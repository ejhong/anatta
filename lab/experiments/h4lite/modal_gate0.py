"""Gate 0 of H4-lite: does the start policy give a signal? (lab/plans/plan-2026-10-29-training.md, v2.4; inference only,
about $15.) Recognition track; private.

Part 1  The base model on every validated pair, both twins, 8 samples each under the hint-only prompt (neutral
        framing plus the release hint), 3 attempts, batched generation (shortest-first, batches of 32, 512 new tokens
        on the first reply and 384 after). Per episode: outcome, form, grounded, hardcoded, first_fail and the rest
        of player.episode_record. Then, locally and in this order: the ambiguity filter (an impossible twin the base
        passes without a detected hack in at least 25% of samples is dropped), the seeded split by problem (80/20)
        and its hash (koan_pairs.split_and_hash, stratified by kind), which goes in lab/ledger/NOTES.md before part 2 and
        before Stage A. The summary carries the pass criterion (hint-only release on at least 3% of impossible
        episodes), live groups, every solvable twin's first-attempt failure rate (hard = at least half), the outcome
        mix by mutation kind, how often a first failure means "impossible", the forms releases and hacks take, the
        env_error count, the pairs dropped at container start (their reference would not run) and the reference
        runtimes.
Part 2  60 training pairs x 8 under the other two prompts: inoculation plus hint (the variant) and no hint (noexit).
Smoke   A 3-step GRPO run on Stage B's mix (modal_retreat2.train from a fresh LoRA), logging step time and peak
        memory per step. It always starts fresh (name gate0_smoke).

Writes /runs/h4lite/gate0/{pairs_all.jsonl, part1/shard_*.jsonl, part2_*/shard_*.jsonl, pairs.jsonl, twins.json,
split_hash.txt, split/{pairs.jsonl, train.jsonl, held.jsonl, split_hash.txt, twins.json, hard.json}} on the volume
(Stage A and B read pairs.jsonl and twins.json at the root; the evaluation apps read split/; see split_files) and
the same under lab/experiments/h4lite/results/gate0/ plus part1.jsonl, part1_summary.json, part2_*.jsonl and part2_summary.json;
the smoke goes to lab/experiments/h4lite/results/retreat2/gate0_smoke/. A dry run (--limit N) writes to gate0_dryN/ in both
places and names its smoke gate0_dryN_smoke, so it never shares a file with the real run.

Shards are reused only when their content is what this run would write (every expected pair with 2 x n episodes
under the same prompt, pilot.shard_complete); anything else is rewritten.

Usage (~/prj/matlab/.venv/bin/modal, from ~/prj/align/anatta-lab), in the plan's order:
  modal run lab/experiments/h4lite/modal_gate0.py::main --pairs PATH/TO/pairs_all.jsonl --skip-part2 --skip-smoke
      # 1. part 1, then the filter, the split and its hash: put results/gate0/split_hash.txt in lab/ledger/NOTES.md
  modal run lab/experiments/h4lite/modal_gate0.py::main --pairs PATH/TO/pairs_all.jsonl --skip-part1
      # 2. part 2 and the smoke, reusing results/gate0/part1.jsonl; refuses if the recomputed hash differs from the
      #    recorded one
  modal run lab/experiments/h4lite/modal_gate0.py::main --pairs ... --limit 4 --part2-pairs 2 --smoke-steps 1   # a dry run
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import modal

from koan_pairs import split_and_hash  # the one split (in the image too, so containers import this module whole)
from modal_common import (CACHE, GPU, RESULTS, ROOT, RUNS, cache, image, load_model, pairs_by_name, parse_max_new,
                          prepare_pairs, read_jsonl, read_rows, runs, write_json, write_jsonl)
from pilot import ambiguous, first_fail_prior, hard_twins, shard_complete, split_hash, twin_stats

PROMPTS = {"hint": ("neutral", True), "inoc_hint": ("inoculation", True), "nohint": ("neutral", False)}
TABLES = {"hint": "koan", "inoc_hint": "koan", "nohint": "noexit"}  # which arm's rewards score each prompt
app = modal.App("anatta-h4lite-gate0", image=image)
if modal.is_local():  # the smoke is a 3-step modal_retreat2.train. Containers mount only this file and the image's
    import modal_retreat2  # modules, not modal_retreat2.py, so the import and the include stay on the laptop.

    app.include(modal_retreat2.app)


def gate0_root(limit: int) -> str:
    """Where a run writes, on the volume and under results/: gate0/ for the real run, gate0_dryN/ for --limit N."""
    return f"gate0_dry{limit}" if limit else "gate0"


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=8 * 3600)
def play_pairs(pairs_text: str, prompt: str, n: int, attempts: int, temperature: float, max_new: tuple,
               gen_batch: int, seed: int, shard: int, out_rel: str, names: list | None = None) -> dict:
    """The base model on both twins of every pair in pairs_text (or of `names`), n samples each, under one prompt.
    Shards of `shard` pairs are written to ROOT/out_rel/shard_K.jsonl as they finish. On a rerun a shard is reused
    only if it holds exactly these pairs' 2 x n episodes under this prompt; otherwise it is rewritten."""
    import torch

    from koan_env import REWARDS, KoanTask
    from pilot import gate0_summary, tally
    from player import Player, episode_record

    framing, release = PROMPTS[prompt]
    pairs, prep = prepare_pairs(pairs_by_name(pairs_text))  # every reference once; pairs whose reference fails are dropped
    names = [p for p in (sorted(names) if names else sorted(pairs)) if p in pairs]
    out = Path(ROOT) / out_rel
    out.mkdir(parents=True, exist_ok=True)
    model, tok, _ = load_model()
    player = Player(model, tok, max_new, gen_batch)
    records, t_all, n_fresh, n_resumed = [], 0.0, 0, 0
    for k in range(0, len(names), shard):
        path = out / f"shard_{k // shard:03d}.jsonl"
        chunk = names[k:k + shard]
        if path.exists():
            rows = read_jsonl(path)
            if shard_complete(rows, chunk, 2 * n) and all(r.get("prompt") == prompt for r in rows):
                print(f"resumed {path} ({len(rows)} episodes)", flush=True)
                records += rows
                n_resumed += len(rows)
                continue
            print(f"rewriting {path}: {len(rows)} episodes on {len({r.get('pair') for r in rows})} pairs, "
                  f"this run needs {2 * n * len(chunk)} on {len(chunk)} under {prompt!r}", flush=True)
        random.seed(seed + k)
        torch.manual_seed(seed + k)
        envs = [KoanTask(pairs[p], imp, framing, release, attempts) for p in chunk for imp in (False, True)
                for _ in range(n)]
        t0 = time.time()
        turns, texts, infos = player.play(envs, temperature)
        rows = [{**episode_record(e, i, t, x), "prompt": prompt}
                for e, i, t, x in zip(envs, infos, turns, texts)]
        write_jsonl(path, rows)
        runs.commit()
        dt = time.time() - t0
        t_all, n_fresh = t_all + dt, n_fresh + len(rows)
        records += rows
        print(json.dumps({"shard": k // shard, "episodes": len(rows), "seconds": round(dt),
                          "episodes_per_hour": round(3600 * len(rows) / max(dt, 1)), **tally(rows)}), flush=True)
    summary = {"prompt": prompt, "framing": framing, "release": release, "n": n, "attempts": attempts,
               "temperature": temperature, "seconds": round(t_all), "fresh_episodes": n_fresh,
               "resumed_episodes": n_resumed,
               "episodes_per_hour": round(3600 * n_fresh / t_all) if t_all else None,
               "dropped_pairs": prep["dropped"], "ref_seconds_median": prep["ref_seconds_median"],
               "ref_seconds_max": prep["ref_seconds_max"], "precompute_seconds": prep["precompute_seconds"],
               **gate0_summary(records, REWARDS[TABLES[prompt]]), "ref_seconds": prep["ref_seconds"]}
    return {"records": records, "summary": summary}


@app.function(volumes={RUNS: runs}, timeout=600)
def put(files: dict) -> list:
    """Write text files under /runs/h4lite/ (relative paths) and commit."""
    for rel, text in files.items():
        p = Path(ROOT) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    runs.commit()
    return sorted(files)


def filter_and_split(rows: list, records: list, held_frac: float, split_seed: int, n: int) -> tuple[list, dict]:
    """The ambiguity filter, then the seeded split and its hash through koan_pairs.split_and_hash (stratified by
    kind, so one-off and conflicting are each split 80/20). Pairs with no scored episode (dropped at container start
    because their reference would not run, or never played) are not kept either. Returns the surviving rows with
    their split (train or held, in input order), and twins.json's content. The hash is the SHA-256 of the sorted
    train names joined by newlines with a trailing newline (koan_pairs's form; pilot.split_hash recomputes it)."""
    stats = twin_stats(records)
    amb, hard = ambiguous(stats), hard_twins(stats)
    played = {p for p, s in stats.items() if s["solvable"].get("n") or s["impossible"].get("n")}
    unplayed = sorted(r["name"] for r in rows if r["name"] not in played)
    keep = [r["name"] for r in rows if r["name"] in played and r["name"] not in set(amb)]
    out, digest = split_and_hash(rows, keep, split_seed, held_frac, stratify_by_kind=True)
    train = sorted(p["name"] for p in out if p["split"] == "train")
    held = sorted(p["name"] for p in out if p["split"] == "held")
    filtered = [p for p in out if p["split"] != "dropped"]
    assert digest == split_hash(train), "pilot.split_hash must reproduce koan_pairs.split_and_hash's digest"
    twins = {"n": n, "pairs_in": len(rows), "ambiguous": amb, "unplayed": unplayed, "kept": len(keep), "hard": hard,
             "hard_train": sorted(set(hard) & set(train)), "first_fail": first_fail_prior(stats),
             "split": {"train": train, "held": held, "dropped": sorted(p["name"] for p in out if p["split"] == "dropped")},
             "split_seed": split_seed, "held_frac": held_frac, "stratify_by_kind": True, "split_hash": digest,
             "split_hash_form": "sha256 of the sorted train names, newline-joined, trailing newline", "stats": stats}
    return filtered, twins, out


def split_files(out: list, twins: dict) -> dict[str, str]:
    """The split's files as relative path -> text, written identically under results/ROOT/ and ROOT/ on the volume.
    Under split/ (what the evaluation apps read, modal_coding_eval.py in this order): pairs.jsonl (every record with
    its split: train, held or dropped), held.jsonl, train.jsonl, split_hash.txt, twins.json (the hard twins, Gate 0's
    first-fail rates, the split and its hash, per-pair stats) and hard.json (the short form of the same). At the
    root, for Stage A and B and for older readers: pairs.jsonl (the train and held records with their split),
    twins.json and split_hash.txt."""
    def lines(rows):
        return "".join(json.dumps(r) + "\n" for r in rows)

    hard = {k: twins[k] for k in ("hard", "hard_train", "first_fail", "split", "split_hash", "split_hash_form")}
    return {"split/pairs.jsonl": lines(out),
            "split/train.jsonl": lines([p for p in out if p["split"] == "train"]),
            "split/held.jsonl": lines([p for p in out if p["split"] == "held"]),
            "split/split_hash.txt": twins["split_hash"] + "\n",
            "split/twins.json": json.dumps(twins, indent=1), "split/hard.json": json.dumps(hard, indent=1),
            "pairs.jsonl": lines([p for p in out if p["split"] != "dropped"]),
            "twins.json": json.dumps(twins, indent=1), "split_hash.txt": twins["split_hash"] + "\n"}


@app.local_entrypoint()
def main(pairs: str, n: int = 8, attempts: int = 3, temperature: float = 1.0, max_new: str = "512,384",
         gen_batch: int = 32, seed: int = 0, split_seed: int = 0, held_frac: float = 0.2, part2_pairs: int = 60,
         shard: int = 64, limit: int = 0, skip_part1: bool = False, skip_part2: bool = False, skip_smoke: bool = False,
         smoke_steps: int = 3, gpu: str = GPU):
    """Part 1 (unless --skip-part1, which reuses results/ROOT/part1.jsonl), then the filter, the split and its
    hash, then part 2 (unless --skip-part2) and the smoke (unless --skip-smoke). The plan's order is two
    invocations: --skip-part2 --skip-smoke first, the hash into lab/ledger/NOTES.md, then --skip-part1."""
    text = Path(pairs).read_text()
    rows = read_rows(text)
    if limit:
        rows = rows[:limit]
        text = "".join(json.dumps(r) + "\n" for r in rows)
    root = gate0_root(limit)
    dest = RESULTS / root
    dest.mkdir(parents=True, exist_ok=True)
    mn = parse_max_new(max_new)
    print(f"{len(rows)} validated pairs from {pairs}; writing {root}/ on the volume and under {RESULTS}", flush=True)

    # part 1: the hint-only prompt on every pair
    if skip_part1:
        if not (dest / "part1.jsonl").exists():
            raise SystemExit(f"--skip-part1 needs {dest / 'part1.jsonl'} from a previous part-1 run")
        records = read_jsonl(dest / "part1.jsonl")
        summary1 = json.loads((dest / "part1_summary.json").read_text())
        print(f"reusing {len(records)} part-1 episodes from {dest}", flush=True)
    else:
        res = play_pairs.with_options(gpu=gpu).remote(text, "hint", n, attempts, temperature, mn, gen_batch, seed,
                                                      shard, f"{root}/part1")
        records, summary1 = res["records"], res["summary"]
        write_jsonl(dest / "part1.jsonl", records)
    filtered, twins, out_all = filter_and_split(rows, records, held_frac, split_seed, n)
    recorded = dest / "split_hash.txt"
    if skip_part1 and recorded.exists() and recorded.read_text().strip() != twins["split_hash"]:
        raise SystemExit(f"the recomputed split hash {twins['split_hash']} differs from the recorded "
                         f"{recorded.read_text().strip()} in {recorded}: the pairs or the split settings changed "
                         f"since part 1; rerun part 1 (without --skip-part1) and register the new hash")
    summary1 = {**summary1, "pairs_in": len(rows), "ambiguous_dropped": len(twins["ambiguous"]),
                "unplayed": twins["unplayed"], "kept": twins["kept"], "train": len(twins["split"]["train"]),
                "held": len(twins["split"]["held"]), "hard": len(twins["hard"]), "hard_train": len(twins["hard_train"]),
                "split_hash": twins["split_hash"]}
    write_json(dest / "part1_summary.json", summary1)
    files = split_files(out_all, twins)
    for rel, body in files.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(body)
    filtered_text = files["pairs.jsonl"]
    put.remote({f"{root}/pairs_all.jsonl": text, **{f"{root}/{rel}": body for rel, body in files.items()}})
    print(json.dumps({k: v for k, v in summary1.items() if k not in ("by_kind", "tally", "forms", "ref_seconds")}),
          flush=True)
    print(f"GATE 0 {'PASS' if summary1['pass'] else 'FAIL'}: hint-only release on {summary1['impossible_release']} "
          f"of impossible episodes (needs {summary1['pass_threshold']}); split hash {twins['split_hash']}", flush=True)
    if skip_part2 and skip_smoke:
        print(f"wrote {dest}; register {recorded} in lab/ledger/NOTES.md, then rerun with --skip-part1 for part 2 and the smoke",
              flush=True)

    # part 2: the other two prompts on 60 training pairs
    if not skip_part2:
        train = twins["split"]["train"]
        chosen = sorted(random.Random(split_seed).sample(train, min(part2_pairs, len(train))))
        calls = {p: play_pairs.with_options(gpu=gpu).spawn(filtered_text, p, n, attempts, temperature, mn, gen_batch,
                                                          seed, shard, f"{root}/part2_{p}", chosen)
                 for p in ("inoc_hint", "nohint")}
        summary2 = {"pairs": chosen, "split_hash": twins["split_hash"]}
        for p, c in calls.items():
            res = c.get()
            write_jsonl(dest / f"part2_{p}.jsonl", res["records"])
            summary2[p] = res["summary"]
            print(json.dumps({k: v for k, v in res["summary"].items()
                              if k not in ("by_kind", "tally", "forms", "ref_seconds")}), flush=True)
        write_json(dest / "part2_summary.json", summary2)

    # smoke: 3 GRPO steps on Stage B's mix, from a fresh LoRA; always a fresh run under its own name
    if not skip_smoke:
        smoke = f"{root}_smoke"
        res = modal_retreat2.train.with_options(gpu=gpu).remote(
            name=smoke, arm="koan", seed=0, steps=smoke_steps, init="new", max_new=mn, gen_batch=gen_batch,
            attempts=attempts, save_every=10**6, fresh=True, pairs_file=f"{ROOT}/{root}/pairs.jsonl",
            twins_file=f"{ROOT}/{root}/twins.json")
        out = modal_retreat2.save_local(res["name"], res["files"])
        for rec in read_jsonl(out / "log.jsonl"):
            print(json.dumps({k: rec.get(k) for k in ("step", "s_gen", "s_train", "s", "peak_gb", "len", "turns", "kl",
                                                       "ratio", "live_all")}), flush=True)
        print("smoke written to", out, flush=True)
    print("wrote", dest, flush=True)
