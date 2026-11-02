"""Stage A of H4-lite: expert iteration, one round (lab/plans/plan-2026-10-29-training.md, v2.6). Recognition track; private.

sample    One round from the start policy under the arm's prompt: koan 16 per impossible twin, 16 per hard solvable
          twin (Gate 0's twins.json), 4 per other solvable twin; noexit 16 per hard and 4 per other, no impossible
          twins. Every turn's prompt and completion are kept as the chat template's exact token ids. Shards are
          reused on a rerun only when they hold exactly the (pair, twin) specs this run would sample under the same
          sampling settings (pilot.shard_matches: the prompt, the cut, the seed, attempts, temperature, max_new; the
          label is not part of the test).
prepare   What's kept: the +1 episodes (solves, and koan's correct releases). For koan (and koan_cut, koan_glimpse):
          retry before release (releases subsampled to at most the number of fail-then-solve episodes), then token
          matching (koan's solves subsampled until its assistant tokens match noexit's), then the balance again.
          pilot.stage_a_data; CPU only. data.json also records where the kept tokens sit (pilot.token_shares: the
          share in failed attempts and in retries), since matching drops first-try solves first and so leaves koan
          with more failed-attempt tokens than noexit, and where the cut was on (cut_sampling, cut_finetune);
          results/sft/summary.json collects both per arm.
finetune  LoRA rank 16, lr 1e-4, 2 epochs, the loss on the assistant's replies only (each turn is one item: its
          prompt is masked, its completion is the target; prompts and environment feedback never carry loss).
          3 seeds per arm share the data; the seed sets the LoRA initialization and the data order.
--under-cut   koan's recipe with the v2 ego cut on throughout: v_ego ablated at layers 8-32 (player.ego_cut, the
          same list cut_vs_push.py's `cut` arm builds), applied with anatta.dial.intervene() around sampling and
          around fine-tuning, forward and backward inside the block so gradient checkpointing's recomputed forward
          pass keeps the cut (test_cut_checkpoint.py). Its label is koan_cut; it is token-matched to noexit like koan.
--glimpse     koan's recipe with the cut on during sampling only (label koan_glimpse, the plan's v2.6): the data are
          generated under the cut exactly as koan_cut's are, and the fine-tune runs with the cut off, no intervene
          context, plain forward and backward, so the gradient reaches the direction while it is active and
          contradicted by the data. koan_cut and koan_glimpse sample the same way, so either takes the other's
          shards when they match spec for spec (SHARED_SAMPLING): one sample, two fine-tune conditions. The two
          flags are mutually exclusive. Every data.json and args.json records cut_sampling and cut_finetune (koan
          and noexit false/false, koan_cut true/true, koan_glimpse true/false).
readout   Prediction 62 (held-out release and over-release on hard twins) is read by modal_retreat2.py::heldout on
          each adapter: `modal run lab/experiments/h4lite/modal_retreat2.py::heldout --labels koan_s0,koan_s1,koan_s2,...`.

Writes /runs/h4lite/sft/LABEL/{shard_*.pt, samples.pt, episodes.jsonl, summary.json, train.pt, data.json} and
/runs/h4lite/sft/LABEL_sSEED/{adapter/, args.json, log.jsonl, done.json}; local copies of the text files go to
lab/experiments/h4lite/results/sft/.

Usage (~/prj/matlab/.venv/bin/modal, from ~/prj/align/anatta-lab):
  modal run lab/experiments/h4lite/modal_sft.py::main --arms koan,noexit --seeds 0,1,2            # sample, prepare, 6 fine-tunes
  modal run lab/experiments/h4lite/modal_sft.py::main --arms koan --under-cut --seeds 0,1,2       # koan_cut, matched to noexit's samples
  modal run lab/experiments/h4lite/modal_sft.py::main --arms koan --glimpse --seeds 0,1,2         # koan_glimpse, on koan_cut's shards when they are there
  modal run lab/experiments/h4lite/modal_sft.py::main --arms koan,noexit --seeds 3,4 --skip-sampling --skip-prepare   # Stage B's extra starts
"""

from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import modal

from modal_common import (CACHE, EGO, GPU, MAX_NEW, RESULTS, ROOT, RUNS, append_jsonl, cache, image, load_model,
                          pairs_by_name, parse_max_new, prepare_pairs, runs, seeds_of, write_json, write_jsonl)
from pilot import shard_matches, stage_a_mode

PAIRS = f"{ROOT}/gate0/pairs.jsonl"
TWINS = f"{ROOT}/gate0/twins.json"
# Labels whose sampling is one and the same (koan's prompt, the cut on, the same seed and settings): either takes the
# other's shards when they match spec for spec, so koan_cut and koan_glimpse differ in the fine-tune alone.
SHARED_SAMPLING = {"koan_cut": "koan_glimpse", "koan_glimpse": "koan_cut"}
app = modal.App("anatta-h4lite-sft", image=image)


def reusable(path: Path, chunk: list, sampling: dict) -> dict | None:
    """The shard at `path` if it holds exactly `chunk` under `sampling` (pilot.shard_matches), else None, saying why
    not. A dry run's shard, another sample count, another seed or another prompt never stands in."""
    if not path.exists():
        return None
    import torch

    d = torch.load(path)
    if shard_matches(d, chunk, sampling):
        return d
    print(f"not reusing {path}: {len(d.get('records', ()))} episodes, label {d.get('label')!r}, "
          f"{'the same' if d.get('specs') == chunk else 'different'} specs, "
          f"{'the same' if d.get('sampling') == sampling else 'other'} sampling settings", flush=True)
    return None


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=8 * 3600)
def sample(label: str, arm: str, under_cut: bool = False, seed: int = 0, attempts: int = 3, temperature: float = 1.0,
           max_new: tuple = MAX_NEW, gen_batch: int = 32, shard: int = 1024, pairs_file: str = PAIRS,
           twins_file: str = TWINS, share_from: str = "") -> dict:
    """One round of sampling from the start policy under the arm's prompt, in shards of `shard` episodes written to
    the volume as they finish. A shard is reused on a rerun only when it was written for exactly the same specs
    (the (pair, impossible) list this run would sample there) under the same sampling settings, one record each
    (pilot.shard_matches); otherwise it is rewritten. `share_from` names a label whose shards stand in under the same
    test (koan_cut and koan_glimpse sample the same way): a matching one is copied here, `shared_from` on record, so
    the two arms train on one sample. Returns the summary and the episode records."""
    import torch
    from anatta.dial import intervene

    from koan_env import ARMS, REWARDS, KoanTask
    from pilot import gate0_summary, keep_positive, stage_a_plan, tally, tokens
    from player import Player, ego_cut, episode_record, turn_ids

    framing, release = ARMS[arm]
    table = REWARDS[arm]
    pairs, prep = prepare_pairs(pairs_by_name(Path(pairs_file).read_text(), "train"))  # references once; failures dropped
    twins = json.loads(Path(twins_file).read_text())
    plan = stage_a_plan(arm, list(pairs), twins["hard"])
    specs = [(p, imp) for p, imp, k in plan for _ in range(k)]
    out = Path(ROOT) / "sft" / label
    out.mkdir(parents=True, exist_ok=True)
    donor = Path(ROOT) / "sft" / share_from if share_from else None
    cut = ego_cut(EGO) if under_cut else []
    sampling = {"arm": arm, "framing": framing, "release": release, "under_cut": under_cut, "seed": seed,
                "attempts": attempts, "temperature": temperature, "max_new": tuple(max_new)}  # what a shard must match
    print(json.dumps({"label": label, "arm": arm, "under_cut": under_cut, "framing": framing, "release": release,
                      "episodes": len(specs), "cut_layers": [iv.layer for iv in cut], "split_hash": twins.get("split_hash"),
                      "dropped_pairs": prep["dropped"], "share_from": share_from or None}), flush=True)
    if arm == "koan" and framing != "neutral":
        print(f"WARNING: koan_env.ARMS['koan'] uses framing {framing!r}; v2.2 trains koan under the hint only", flush=True)
    model, tok, hf_model = load_model()
    player = Player(model, tok, max_new, gen_batch)
    records, turns_all, t_all, n_fresh, n_shared = [], [], 0.0, 0, 0
    for k in range(0, len(specs), shard):
        path = out / f"shard_{k // shard:03d}.pt"
        chunk = specs[k:k + shard]
        d = reusable(path, chunk, sampling)
        if d is not None:
            print(f"resumed {path} ({len(d['records'])} episodes)", flush=True)
        elif donor is not None and (d := reusable(donor / path.name, chunk, sampling)) is not None:
            d = {**d, "label": label, "shared_from": share_from}  # the same sample, under this label too
            torch.save(d, path)
            runs.commit()
            n_shared += len(d["records"])
            print(f"shared {donor / path.name} -> {path} ({len(d['records'])} episodes)", flush=True)
        if d is not None:
            records += d["records"]
            turns_all += d["turns"]
            continue
        random.seed(seed + k)
        torch.manual_seed(seed + k)
        envs = [KoanTask(pairs[p], imp, framing, release, attempts) for p, imp in chunk]
        t0 = time.time()
        with intervene(hf_model, cut) if cut else nullcontext():  # the cut's hooks on, or the plain model
            turns, texts, infos = player.play(envs, temperature)
        rows = [episode_record(e, i, t, x) for e, i, t, x in zip(envs, infos, turns, texts)]
        ids = [turn_ids(t) for t in turns]
        torch.save({"records": rows, "turns": ids, "specs": chunk, "label": label, "sampling": sampling,
                    "shared_from": None}, path)
        runs.commit()
        dt = time.time() - t0
        t_all, n_fresh = t_all + dt, n_fresh + len(rows)
        records += rows
        turns_all += ids
        print(json.dumps({"shard": k // shard, "episodes": len(rows), "seconds": round(dt),
                          "episodes_per_hour": round(3600 * len(rows) / max(dt, 1)), **tally(rows)}), flush=True)
    torch.save({"records": records, "turns": turns_all}, out / "samples.pt")
    write_jsonl(out / "episodes.jsonl", records)
    kept = keep_positive(records, table)
    summary = {"label": label, "arm": arm, "under_cut": under_cut, "framing": framing, "release": release,
               "seed": seed, "seconds": round(t_all), "episodes": len(records), "fresh_episodes": n_fresh,
               "shared_episodes": n_shared, "shared_from": share_from or None,
               "episodes_per_hour": round(3600 * n_fresh / t_all) if t_all else None,
               "dropped_pairs": prep["dropped"], "train_pairs": len(pairs),
               "plan": {"impossible": sum(k for _, imp, k in plan if imp),
                        "hard": sum(k for p, imp, k in plan if not imp and p in set(twins["hard"])),
                        "other": sum(k for p, imp, k in plan if not imp and p not in set(twins["hard"]))},
               "positives": len(kept), "positive_tokens": tokens(kept),
               "releases": sum(e["outcome"] == "released" for e in kept),
               "retry_solves": sum(e["outcome"] == "solved" and e["first_fail"] for e in kept),
               **gate0_summary(records, table)}
    write_json(out / "summary.json", summary)
    runs.commit()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_kind", "tally", "forms")}), flush=True)
    return {"summary": summary, "records": records}


@app.function(volumes={RUNS: runs}, cpu=4.0, memory=16384, timeout=1800)
def prepare(label: str, arm: str, ref_label: str = "noexit", ref_arm: str = "noexit", seed: int = 0,
            cut_sampling: bool = False, cut_finetune: bool = False) -> dict:
    """The arm's training set from its samples: the +1 episodes, balanced and token-matched to the reference arm
    (noexit keeps its +1 episodes as they are). Writes LABEL/train.pt and data.json, which records the counts, the
    token shares (failed attempts, retries) of what is kept, and where the cut is on for this label (cut_sampling,
    cut_finetune: pilot.stage_a_mode)."""
    import torch

    from koan_env import REWARDS
    from pilot import keep_positive, stage_a_data, token_shares, tokens

    out = Path(ROOT) / "sft" / label
    d = torch.load(out / "samples.pt")
    eps = [{**e, "_i": i} for i, e in enumerate(d["records"])]
    if label == ref_label:
        kept = keep_positive(eps, REWARDS[arm])
        info = {"ref": {"episodes": len(kept), "tokens": tokens(kept)}}
    else:
        ref = torch.load(Path(ROOT) / "sft" / ref_label / "samples.pt")["records"]
        kept, _, info = stage_a_data(eps, ref, REWARDS[arm], REWARDS[ref_arm], seed)
    data = {"records": [{k: v for k, v in e.items() if k != "_i"} for e in kept],
            "turns": [d["turns"][e["_i"]] for e in kept]}
    torch.save(data, out / "train.pt")
    info = {"label": label, "arm": arm, "ref_label": ref_label, "seed": seed, "cut_sampling": cut_sampling,
            "cut_finetune": cut_finetune, "kept": len(kept),
            "tokens": tokens(kept), "turns": sum(len(t) for t in data["turns"]),
            "releases": sum(e["outcome"] == "released" for e in kept),
            "retry_solves": sum(e["outcome"] == "solved" and e["first_fail"] for e in kept),
            "first_try_solves": sum(e["outcome"] == "solved" and not e["first_fail"] for e in kept),
            "token_shares": token_shares(data["turns"]), **info}
    write_json(out / "data.json", info)
    runs.commit()
    print(json.dumps(info), flush=True)
    return info


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=6 * 3600)
def finetune(label: str, seed: int, under_cut: bool = False, lr: float = 1e-4, rank: int = 16, epochs: int = 2,
             micro: int = 4, batch: int = 16, src_dir: str = "", out_dir: str = "", init: str = "new",
             save_epochs: bool = False) -> dict:
    """LoRA fine-tuning on LABEL/train.pt: every turn of every kept episode is one item, its prompt masked and its
    completion the target. `batch` items per optimizer step in micro-batches of `micro`; the loss is the mean
    negative log-likelihood per completion token in the batch. `under_cut` keeps the v2 ego cut on for the whole
    loop (koan_cut); without it the loop runs on the plain model, no intervene context (koan, noexit, and
    koan_glimpse, whose data were sampled under the cut: data.json says which, and a fine-tune asked for the other
    cut condition than its data.json records is refused). A rerun starts over: the log is truncated first.

    The last four serve the durability test (modal_durability.py, predictions 135-139) and leave Stage A's behavior
    as it was at their defaults: `src_dir` and `out_dir` stand in for ROOT/sft/LABEL and ROOT/sft/LABEL_sSEED;
    `init` is an adapter path whose LoRA keeps training (loaded trainable, with a fresh optimizer and this seed's data
    order) instead of a fresh LoRA ("new", seeded); `save_epochs` also writes adapter_epochK/ after every epoch, the
    volume committed each time. args.json records the four only when one of them is set."""
    import torch
    from anatta.dial import intervene

    from player import ego_cut, pack, peak_gb, reset_peak, save_adapter, token_logps

    if init != "new":  # another container wrote the adapter, and a warm container reused for this call sees its
        runs.reload()  # commit only after a reload (the declared pilot's smoke, 1 Nov: adapter_config.json not found)
    src = Path(src_dir) if src_dir else Path(ROOT) / "sft" / label
    out = Path(out_dir) if out_dir else Path(ROOT) / "sft" / f"{label}_s{seed}"
    info = json.loads((src / "data.json").read_text())
    if info.get("cut_finetune") not in (None, under_cut):
        raise SystemExit(f"{label}'s data.json was prepared for cut_finetune={info['cut_finetune']}; this fine-tune "
                         f"was asked for {under_cut}")
    out.mkdir(parents=True, exist_ok=True)
    data = torch.load(src / "train.pt")
    items = [(p, c) for t in data["turns"] for p, c in t]
    args = dict(label=label, seed=seed, cut_sampling=info.get("cut_sampling"), cut_finetune=under_cut, lr=lr,
                rank=rank, epochs=epochs, micro=micro, batch=batch, episodes=len(data["records"]), items=len(items),
                tokens=sum(len(c) for _, c in items), data=info)
    if src_dir or out_dir or init != "new" or save_epochs:  # Stage A's args.json keeps its keys
        args.update(src_dir=str(src), out_dir=str(out), init=init, save_epochs=save_epochs)
    write_json(out / "args.json", args)
    for stale in ("log.jsonl", "done.json"):
        (out / stale).unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in args.items() if k != "data"}), flush=True)

    random.seed(seed)
    torch.manual_seed(seed)  # the LoRA initialization (a loaded adapter has none to draw)
    model, tok, hf_model = load_model(adapter=init, trainable=True, rank=rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    dev = next(model.parameters()).device
    cut = ego_cut(EGO) if under_cut else []
    rng = random.Random(seed)  # the data order
    n_opt, t_start = 0, time.time()
    # The cut's hooks around the whole loop, forward and backward both inside (the recomputed forward keeps the cut);
    # without a cut the plain model trains, no context at all.
    with intervene(hf_model, cut) if cut else nullcontext():
        for epoch in range(epochs):
            order = list(range(len(items)))
            rng.shuffle(order)
            for b in range(0, len(order), batch):
                t0 = time.time()
                reset_peak()
                model.train()
                opt.zero_grad(set_to_none=True)
                idx = sorted(order[b:b + batch], key=lambda i: len(items[i][0]) + len(items[i][1]))
                ntok, nll = sum(len(items[i][1]) for i in idx), 0.0
                for k in range(0, len(idx), micro):
                    chunk = [items[i] for i in idx[k:k + micro]]
                    full, attn, cm, wc = pack(chunk, tok.pad_token_id, dev)
                    lp = token_logps(model, full, attn, wc)
                    loss = -(lp * cm).sum() / ntok
                    loss.backward()
                    nll += float(loss)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                opt.step()
                n_opt += 1
                rec = {"epoch": epoch + 1, "opt_step": n_opt, "items": len(idx), "tokens": ntok, "nll": nll,
                       "grad_norm": grad_norm, "s": round(time.time() - t0, 1), "peak_gb": peak_gb(),
                       "elapsed": round(time.time() - t_start)}
                append_jsonl(out / "log.jsonl", rec)
                if n_opt % 10 == 1:
                    print(json.dumps(rec), flush=True)
            if save_epochs:  # the durability test reads its erosions after epoch 1 as well as at the end
                save_adapter(model, out / f"adapter_epoch{epoch + 1}")
                runs.commit()
    save_adapter(model, out / "adapter")
    write_json(out / "done.json", {"label": label, "seed": seed, "opt_steps": n_opt, "seconds": round(time.time() - t_start)})
    runs.commit()
    return {"name": f"{label}_s{seed}", "files": {f: (out / f).read_text() for f in ("args.json", "log.jsonl", "done.json")}}


def save_local(name: str, files: dict) -> Path:
    dest = RESULTS / "sft" / name
    dest.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        (dest / fname).write_text(text)
    return dest


def stage_a_summary(infos: dict) -> dict:
    """results/sft/summary.json: per label, where the cut was on, what prepare kept and where its tokens sit, side
    by side."""
    keys = ("arm", "cut_sampling", "cut_finetune", "kept", "tokens", "releases", "retry_solves", "first_try_solves",
            "token_shares")
    return {label: {k: info.get(k) for k in keys} for label, info in sorted(infos.items())}


@app.local_entrypoint()
def main(arms: str = "koan,noexit", seeds: str = "0,1,2", under_cut: bool = False, glimpse: bool = False,
         sample_seed: int = 0, data_seed: int = 0, ref: str = "noexit", attempts: int = 3, temperature: float = 1.0,
         max_new: str = "512,384", gen_batch: int = 32, shard: int = 1024, lr: float = 1e-4, rank: int = 16,
         epochs: int = 2, micro: int = 4, batch: int = 16, skip_sampling: bool = False, skip_prepare: bool = False,
         skip_finetune: bool = False, gpu: str = GPU):
    """Stage A for the given arms: sample (unless --skip-sampling), prepare the training sets, then fine-tune one
    adapter per seed, in parallel containers. --under-cut turns koan into koan_cut (the cut on throughout) and
    --glimpse into koan_glimpse (the cut on for sampling only; pilot.stage_a_mode); each is matched against the
    reference arm's samples, which must already be on the volume, and takes the other's shards when they match
    (SHARED_SAMPLING)."""
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    try:
        modes = [(a, *stage_a_mode(a, under_cut, glimpse)) for a in arm_list]  # (arm, label, cut_sampling, cut_finetune)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if (under_cut or glimpse) and arm_list != ["koan"]:
        raise SystemExit(f"--{'under-cut' if under_cut else 'glimpse'} is koan's recipe with the cut on: use --arms koan")
    mn = parse_max_new(max_new)
    if not skip_sampling:
        calls = [(label, sample.with_options(gpu=gpu).spawn(label, a, cut_s, sample_seed, attempts, temperature, mn,
                                                            gen_batch, shard, share_from=SHARED_SAMPLING.get(label, "")))
                 for a, label, cut_s, _ in modes]
        for label, c in calls:
            res = c.get()
            dest = save_local(label, {"summary.json": json.dumps(res["summary"], indent=1)})
            write_jsonl(dest / "episodes.jsonl", res["records"])
            print("sampled", label, "->", dest, flush=True)
    if not skip_prepare:
        ordered = sorted(modes, key=lambda m: m[1] != ref)  # the reference arm first, the others match against it
        infos = {}
        for a, label, cut_s, cut_f in ordered:
            infos[label] = prepare.remote(label, a, ref, "noexit", data_seed, cut_s, cut_f)
            save_local(label, {"data.json": json.dumps(infos[label], indent=1)})
        summary_path = RESULTS / "sft" / "summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary = {**previous, **stage_a_summary(infos)}
        write_json(summary_path, summary)
        for label, s in summary.items():
            ts = s.get("token_shares") or {}
            print(f"{label}: kept {s.get('kept')} episodes, {s.get('tokens')} tokens; failed-attempt share "
                  f"{ts.get('failed_attempt_share')}, retry share {ts.get('retry_share')}; cut during sampling "
                  f"{s.get('cut_sampling')}, during the fine-tune {s.get('cut_finetune')}", flush=True)
    if not skip_finetune:
        calls = [finetune.with_options(gpu=gpu).spawn(label, s, cut_f, lr, rank, epochs, micro, batch)
                 for _, label, _, cut_f in modes for s in seeds_of(seeds)]
        for c in calls:
            res = c.get()
            print("wrote", save_local(res["name"], res["files"]), flush=True)
