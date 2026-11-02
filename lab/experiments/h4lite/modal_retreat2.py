"""Stage B of H4-lite: multi-turn GRPO with LoRA on the koan v2 twins (lab/plans/plan-2026-10-29-training.md, v2.4).
Recognition track; private.

The loop (train):
  - Episode mix per step: 1 impossible twin x 16 samples and 6 hard solvable twins x 8, drawn by pilot.Sampler: the
    solvable twins by their current first-failure rate, re-estimated from recent samples. The ratio of solvable first
    failures to impossible episodes is logged every step; when its 10-step average falls under 1.5 the impossible
    group drops to 8 and a seventh solvable twin joins (one-way).
  - Multi-turn: up to 3 replies with the failing tests shown after each failed one (koan_env.KoanTask). Each turn's
    prompt is the chat template's exact token ids as generation saw them, and every reply carries its episode's
    advantage. Dr. GRPO: advantage = reward minus the group mean, no std scaling; the token loss is summed and
    divided by a constant (episodes per step x first-turn tokens).
  - KL 0.02 to the run's start (k3 estimator). The start adapter is loaded twice through PEFT: as the trainable
    "default" adapter and as a frozen "ref" adapter; the reference pass runs under player.reference(), which
    switches to "ref" (no grad) and back. With a fresh LoRA (--init new, the smoke) the reference is the base model
    with the adapter off, which is what a fresh LoRA computes at step 1. Checkpoints save "default" only.
  - Warm start from the arm's Stage A adapter with the same seed.
  - LoRA rank 16 (only used for a fresh LoRA), lr 5e-5, temperature 1.0, 512 new tokens on turn 1 and 384 after,
    generation in batches of 32, timeout 24 h, checkpoints every 25 steps (adapter_STEP/, including the last step)
    with the sampler, optimizer and RNG state next to each (state_STEP.pt; only the latest is kept).
  - A run name that already exists on the volume is refused unless --resume (continue from the latest adapter_K at
    K+1, log and samples appended, rows after K dropped) or --fresh (overwrite). On a resume the KL reference is
    still the run's original start, not the checkpoint.
  - The trainer never evaluates. Checkpoints are evaluated in separate containers (check -> evaluate_checkpoint),
    and pilot.decide applies the stop-loss and step-budget rules to seed 0's held-out release rates. A decision goes
    to every run of the sweep (both arms, every seed, by default) through the anatta-h4lite-control Dict, which the
    trainers read at every checkpoint; a trainer pops its own entry when it starts fresh.
  - Evaluation seeds: the rules' evaluations use seed 1000 + step. Prediction 64 is scored by `score`, which
    re-evaluates every run of the sweep at the budget step with seed 2000 + step, seed 0 included, so the step is
    chosen and scored on different samples (rows carry purpose "rule" or "score"; decide reads "rule" only).
  - `heldout` scores any adapter by path or Stage A label with the same procedure (prediction 62), with the ego cut
    on when asked (--under-cut, for the koan_cut arm's 2x2).

Writes /runs/h4lite/retreat2/NAME/{args.json, log.jsonl, samples.jsonl, adapter_STEP/, state_STEP.pt, adapter/,
done.json, evals.jsonl, evals_STEP.{jsonl,json}, score_STEP.{jsonl,json}}; heldout writes sft/LABEL/heldout[_cut].{jsonl,json}
or heldout/LABEL/...; the local entrypoints copy the text files to lab/experiments/h4lite/results/ in the same layout.

Usage (~/prj/matlab/.venv/bin/modal, from ~/prj/align/anatta-lab; this file has several entrypoints, so name one):
  modal run --detach lab/experiments/h4lite/modal_retreat2.py::main --arm koan --seeds 0,1,2,3,4
  modal run --detach lab/experiments/h4lite/modal_retreat2.py::main --arm noexit --seeds 0,1,2,3,4
  modal run --detach lab/experiments/h4lite/modal_retreat2.py::main --arm koan --seeds 2 --resume    # koan_s2 from its last checkpoint
  modal run lab/experiments/h4lite/modal_retreat2.py::check --name koan_s0 --step 0        # the start (its Stage A adapter)
  modal run lab/experiments/h4lite/modal_retreat2.py::check --name koan_s0 --step 50       # stop-loss; a stop goes to all 10 runs
  modal run lab/experiments/h4lite/modal_retreat2.py::check --name koan_s0 --step 75       # budget rule, likewise
  modal run lab/experiments/h4lite/modal_retreat2.py::score --step 100                     # prediction 64: all 10 runs, fresh seeds
  modal run lab/experiments/h4lite/modal_retreat2.py::heldout --labels koan_s0,koan_s1,koan_s2,noexit_s0,noexit_s1,noexit_s2
  modal run lab/experiments/h4lite/modal_retreat2.py::heldout --labels koan_cut_s0,koan_cut_s1,koan_cut_s2 --under-cut
  modal run lab/experiments/h4lite/modal_retreat2.py::heldout --adapter none --arm koan --label base    # the base model
  modal run lab/experiments/h4lite/modal_retreat2.py::fetch --name koan_s0
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import modal

from modal_common import (CACHE, EGO, GPU, MAX_NEW, RESULTS, ROOT, RUNS, append_jsonl, cache, control, image,
                          load_model, names_of, pairs_by_name, parse_max_new, prepare_pairs, read_jsonl, runs, seeds_of,
                          volume_path, write_json, write_jsonl)
from pilot import ENV_ERROR, RULE, SCORE

PAIRS = f"{ROOT}/gate0/pairs.jsonl"  # the filtered, split pairs Gate 0 wrote
TWINS = f"{ROOT}/gate0/twins.json"  # hard twins and first-failure rates from Gate 0
SWEEP_ARMS, SWEEP_SEEDS = "koan,noexit", "0,1,2,3,4"  # the Stage B sweep a decision applies to
app = modal.App("anatta-h4lite-retreat2", image=image)


def stop_targets(name: str, arms: str = SWEEP_ARMS, seeds: str = SWEEP_SEEDS, apply_to: str = "") -> list[str]:
    """Every run a decision applies to: the whole sweep (ARM_sSEED over the arms and seeds), the evaluated run, and
    anything named in --apply-to (runs started with --name, whose names the sweep pattern doesn't cover)."""
    sweep = {f"{a}_s{s}" for a in names_of(arms) for s in seeds_of(seeds)}
    return sorted(sweep | {name} | set(names_of(apply_to)))


def arm_of(label: str) -> str:
    """The arm a Stage A label trains: koan_s0 -> koan, koan_cut_s1 -> koan, noexit_s2 -> noexit."""
    base = label.rsplit("_s", 1)[0] if "_s" in label else label
    return "koan" if base.startswith("koan") else base


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=24 * 3600)
def train(name: str, arm: str, seed: int, steps: int = 150, init: str = "new", lr: float = 5e-5, rank: int = 16,
          beta: float = 0.02, temperature: float = 1.0, max_new: tuple = MAX_NEW, gen_batch: int = 32, micro: int = 8,
          attempts: int = 3, save_every: int = 25, imp_samples: int = 16, sol_twins: int = 6, sol_samples: int = 8,
          ratio_min: float = 1.5, ratio_window: int = 10, pairs_file: str = PAIRS, twins_file: str = TWINS,
          resume: bool = False, fresh: bool = False) -> dict:
    import shutil

    import torch

    from koan_env import ARMS, REWARDS, KoanTask
    from pilot import Mix, Sampler, advantages, checkpoint_steps, live_share, score, tally
    from player import Player, episode_record, pack, peak_gb, reference, reset_peak, save_adapter, token_logps

    framing, release = ARMS[arm]
    table = REWARDS[arm]
    out = Path(ROOT) / "retreat2" / name
    start_step, state, resumed_from = 1, None, None
    if fresh and out.exists():
        shutil.rmtree(out)
    elif resume:
        done = checkpoint_steps(p.name for p in out.glob("adapter_*")) if out.exists() else []
        if not done:
            raise SystemExit(f"nothing to resume: {out} has no adapter_STEP checkpoint")
        resumed_from = done[-1]
        prev = json.loads((out / "args.json").read_text())
        if init != prev["init"]:
            print(f"resume: the KL reference stays the run's original start {prev['init']!r}, not {init!r}", flush=True)
        init = prev["init"]
        state_path = out / f"state_{resumed_from}.pt"
        state = torch.load(state_path, weights_only=False) if state_path.exists() else None
        start_step = resumed_from + 1
        for f in ("log.jsonl", "samples.jsonl"):  # rows after the checkpoint belong to steps that will run again
            write_jsonl(out / f, [r for r in read_jsonl(out / f) if r["step"] <= resumed_from])
        print(json.dumps({"resumed": name, "from_step": resumed_from, "state_restored": state is not None}), flush=True)
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists: pass resume=True (--resume) to continue it from its last checkpoint, or "
                         f"fresh=True (--fresh) to overwrite it")
    out.mkdir(parents=True, exist_ok=True)
    train_from = str(out / f"adapter_{resumed_from}") if resume else init
    ref_from = None if init == "new" else init  # a fresh LoRA's start is the base model: no second adapter needed

    pairs, prep = prepare_pairs(pairs_by_name(Path(pairs_file).read_text(), "train"))  # references once; failures dropped
    twins = json.loads(Path(twins_file).read_text())
    prior = {p: tuple(twins["first_fail"][p]) for p in pairs if p in twins["first_fail"]}
    mix, fallback = Mix(imp_samples, sol_twins, sol_samples), Mix(max(1, imp_samples // 2), sol_twins + 1, sol_samples)
    sampler = Sampler(list(pairs), prior, seed=seed, mix=mix, fallback=fallback, ratio_min=ratio_min,
                      ratio_window=ratio_window)
    args = dict(name=name, arm=arm, seed=seed, steps=steps, init=init, lr=lr, rank=rank, beta=beta,
                temperature=temperature, max_new=list(max_new), gen_batch=gen_batch, micro=micro, attempts=attempts,
                save_every=save_every, mix=mix.__dict__, fallback=fallback.__dict__, ratio_min=ratio_min,
                ratio_window=ratio_window, framing=framing, release=release, rewards=table, pairs_file=pairs_file,
                twins_file=twins_file, train_pairs=len(pairs), hard_twins=sum(p in pairs for p in twins["hard"]),
                split_hash=twins.get("split_hash"), reference="base" if ref_from is None else ref_from,
                dropped_pairs=prep["dropped"])
    if resume:
        prev = json.loads((out / "args.json").read_text())
        args = {**prev, "steps": steps, "resumed": prev.get("resumed", []) + [resumed_from]}
    write_json(out / "args.json", args)
    print(json.dumps(args), flush=True)
    if arm == "koan" and framing != "neutral":
        print(f"WARNING: koan_env.ARMS['koan'] uses framing {framing!r}; v2.2 trains koan under the hint only", flush=True)
    if not resume:  # a decision left over from an earlier run under this name must not stop this one
        try:
            if control.contains(name):
                print(json.dumps({"control_cleared": name, "was": control.pop(name)}), flush=True)
        except Exception as exc:  # a Dict hiccup must not end a 24-hour run
            print("control clear failed:", repr(exc)[:200], flush=True)

    random.seed(seed if not resume else seed + start_step)
    torch.manual_seed(seed if not resume else seed + start_step)
    model, tok, _ = load_model(adapter=train_from, trainable=True, rank=rank, ref=ref_from)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if state:
        sampler.load_state_dict(state["sampler"])
        if state.get("opt"):
            opt.load_state_dict(state["opt"])
        rng = state.get("rng") or {}
        if "random" in rng:
            random.setstate(rng["random"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    player = Player(model, tok, max_new, gen_batch)
    dev = player.dev
    denom = mix.episodes * max_new[0]  # the constant the summed token loss is divided by
    stopped_at = None

    def rng_state() -> dict:
        return {"random": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    for step in range(start_step, steps + 1):
        t0 = time.time()
        reset_peak()
        draw = sampler.draw()
        sizes = [n for _, _, n in draw]
        envs = [KoanTask(pairs[p], imp, framing, release, attempts) for p, imp, n in draw for _ in range(n)]
        turns, texts, infos = player.play(envs, temperature)
        records = [episode_record(e, i, t, x) for e, i, t, x in zip(envs, infos, turns, texts)]
        rewards = [score(table, r["outcome"]) for r in records]
        valid = [r["outcome"] != ENV_ERROR for r in records]  # an env_error has no advantage and no loss term
        adv = advantages(rewards, sizes, valid)
        t_gen = time.time() - t0

        model.train()
        opt.zero_grad(set_to_none=True)
        items = sorted(((e, t) for e in range(len(envs)) if valid[e] for t in range(len(turns[e]))),
                       key=lambda it: sum(map(len, turns[it[0]][it[1]])))
        kl_sum, ntok, live = 0.0, 0, 0
        for k in range(0, len(items), micro):
            chunk = items[k:k + micro]
            a = torch.tensor([adv[e] for e, _ in chunk], device=dev)
            if beta == 0 and not a.abs().sum():
                continue
            live += 1
            full, attn, cm, wc = pack([turns[e][t] for e, t in chunk], tok.pad_token_id, dev)
            ref = None
            if beta:  # the reference pass first: switching adapters flips requires_grad, so never between a
                with torch.no_grad(), reference(model):  # forward and its backward
                    ref = token_logps(model, full, attn, wc)
            lp = token_logps(model, full, attn, wc)
            obj = a[:, None] * lp
            if beta:
                d = ref - lp
                kl = torch.exp(d) - d - 1  # k3 estimator of KL(policy || start)
                obj = obj - beta * kl
                kl_sum += float((kl.detach() * cm).sum())
            ntok += int(cm.sum())
            (-(obj * cm).sum() / denom).backward()
        grad_norm = None
        if live:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()

        mixlog = sampler.update(records)
        ok = [r for r, v in zip(records, valid) if v]
        imp = [r for r in ok if r["impossible"]]
        sol = [r for r in ok if not r["impossible"]]
        rec = {"step": step, "score": sum(score(table, r["outcome"]) for r in ok) / max(1, len(ok)),
               "score_impossible": sum(score(table, r["outcome"]) for r in imp) / max(1, len(imp)),
               "score_solvable": sum(score(table, r["outcome"]) for r in sol) / max(1, len(sol)),
               "live_impossible": live_share(imp, table), "live_all": live_share(records, table),
               "env_errors": len(records) - len(ok),
               "turns": sum(r["n_turns"] for r in ok) / max(1, len(ok)), "len": ntok / max(1, len(items)),
               "kl": kl_sum / max(1, ntok), "micro": live, "grad_norm": grad_norm, "s_gen": round(t_gen),
               "s_train": round(time.time() - t0 - t_gen), "s": round(time.time() - t0), "peak_gb": peak_gb(),
               **mixlog, "draw": [list(d) for d in draw], **tally(records)}
        append_jsonl(out / "log.jsonl", rec)
        print(json.dumps(rec), flush=True)
        i = 0
        for _, _, n in draw:  # one episode per twin group, for reading (scorer exploits at steps 25, 50 and 100)
            r = records[i]
            append_jsonl(out / "samples.jsonl", {"step": step, "name": r["name"], "outcome": r["outcome"],
                                                 "score": rewards[i], "advantage": adv[i], "turn": r["turn"],
                                                 "form": r["form"], "replies": r["replies"]})
            i += n
        if step % save_every == 0:  # the last step included: the budget step may be 150
            save_adapter(model, out / f"adapter_{step}")
            torch.save({"step": step, "sampler": sampler.state_dict(), "opt": opt.state_dict(), "rng": rng_state()},
                       out / f"state_{step}.pt")
            for old in out.glob("state_*.pt"):  # the optimizer state is large; only the latest checkpoint resumes
                if old.name != f"state_{step}.pt":
                    old.unlink()
            runs.commit()
            try:
                ctl = control.get(name, None)
            except Exception as exc:  # a Dict hiccup must not end a 24-hour run
                ctl = None
                print("control read failed:", repr(exc)[:200], flush=True)
            if ctl and step >= int(ctl.get("stop_at", 10**9)):
                stopped_at = step
                print(json.dumps({"stopped_at": step, "control": ctl}), flush=True)
                break
    save_adapter(model, out / "adapter")
    write_json(out / "done.json", {"name": name, "steps_run": stopped_at or steps, "stopped_at": stopped_at,
                                   "resumed_from": resumed_from})
    runs.commit()
    return {"name": name, "files": {f: (out / f).read_text() for f in ("args.json", "log.jsonl", "samples.jsonl", "done.json")}}


def heldout_eval(adapter: str | None, arm: str, out_dir: Path, tag: str, eval_n: int, temperature: float,
                 attempts: int, max_new: tuple, gen_batch: int, eval_seed: int, under_cut: bool, pairs_file: str,
                 twins_file: str) -> dict:
    """In a container: the held-out MBPP+ twins (split "held" of Gate 0's pairs.jsonl), eval_n samples each,
    `attempts` attempts, on an adapter path (None = the base model) under the arm's prompt, with the ego cut on if
    asked. Writes out_dir/TAG.jsonl (the episodes) and out_dir/TAG.json: release on impossible twins (by kind,
    grounded), hack, solve, near-miss, overfit, over-release and over-release on the hard twins (Gate 0's
    first-failure rates), pilot.heldout_summary."""
    import torch
    from anatta.dial import intervene

    from koan_env import ARMS, KoanTask
    from pilot import heldout_summary
    from player import Player, ego_cut, episode_record

    framing, release = ARMS[arm]
    held, prep = prepare_pairs(pairs_by_name(Path(pairs_file).read_text(), "held"))  # references once; failures dropped
    hard = set(json.loads(Path(twins_file).read_text())["hard"])
    random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    model, tok, hf_model = load_model(adapter=adapter)
    player = Player(model, tok, max_new, gen_batch)
    envs = [KoanTask(p, imp, framing, release, attempts) for p in held.values() for imp in (False, True)
            for _ in range(eval_n)]
    t0 = time.time()
    with intervene(hf_model, ego_cut(EGO) if under_cut else []):
        turns, texts, infos = player.play(envs, temperature)
    records = [episode_record(e, i, t, x) for e, i, t, x in zip(envs, infos, turns, texts)]
    res = {"adapter": adapter, "arm": arm, "under_cut": under_cut, "eval_n": eval_n, "temperature": temperature,
           "attempts": attempts, "eval_seed": eval_seed, "held_pairs": len(held), "hard_held": len(hard & set(held)),
           "dropped_pairs": prep["dropped"], "seconds": round(time.time() - t0), **heldout_summary(records, hard)}
    write_jsonl(out_dir / f"{tag}.jsonl", records)
    write_json(out_dir / f"{tag}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "tally"}), flush=True)
    return res


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=3 * 3600)
def evaluate_checkpoint(name: str, step: int, eval_n: int = 4, temperature: float = 0.7, attempts: int = 3,
                        max_new: tuple = MAX_NEW, gen_batch: int = 32, purpose: str = RULE, eval_seed: int = 0,
                        pairs_file: str = PAIRS, twins_file: str = TWINS) -> dict:
    """Held-out release at a checkpoint of a Stage B run, in its own container. step 0 is the run's start (its init
    adapter, or the base model when the run started a fresh LoRA); step -1 is the final adapter. purpose "rule"
    (seed 1000 + step, files evals_STEP.*) feeds pilot.decide; "score" (seed 2000 + step, files score_STEP.*) is
    prediction 64's reading and never does. Appends to NAME/evals.jsonl and returns every evaluation of the run."""
    out = Path(ROOT) / "retreat2" / name
    args = json.loads((out / "args.json").read_text())
    if step == 0:
        adapter = None if args["init"] == "new" else args["init"]
    else:
        adapter = str(out / ("adapter" if step < 0 else f"adapter_{step}"))
    if not eval_seed:
        eval_seed = (1000 if purpose == RULE else 2000) + max(step, 0)
    tag = f"evals_{step}" if purpose == RULE else f"score_{step}"
    res = heldout_eval(adapter, args["arm"], out, tag, eval_n, temperature, attempts, max_new, gen_batch, eval_seed,
                       False, pairs_file, twins_file)
    res = {"name": name, "step": step, "purpose": purpose, **res}
    append_jsonl(out / "evals.jsonl", res)
    runs.commit()
    return {"eval": res, "evals": read_jsonl(out / "evals.jsonl")}


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=3 * 3600)
def evaluate_adapter(adapter: str | None, arm: str, out_rel: str, tag: str = "heldout", eval_n: int = 4,
                     temperature: float = 0.7, attempts: int = 3, max_new: tuple = MAX_NEW, gen_batch: int = 32,
                     eval_seed: int = 3000, under_cut: bool = False, pairs_file: str = PAIRS,
                     twins_file: str = TWINS) -> dict:
    """The same held-out procedure on any adapter path (None = base), writing ROOT/out_rel/TAG.{jsonl,json}: the
    Stage A readout (prediction 62) and anything else scored outside a Stage B run."""
    res = heldout_eval(adapter, arm, Path(ROOT) / out_rel, tag, eval_n, temperature, attempts, max_new, gen_batch,
                       eval_seed, under_cut, pairs_file, twins_file)
    runs.commit()
    return {"eval": res, "out_rel": out_rel, "files": {f"{tag}.json": json.dumps(res, indent=1)}}


@app.function(volumes={RUNS: runs}, timeout=600)
def read_run(name: str) -> dict:
    """The text files of a run on the volume (for fetch, after a detached run)."""
    out = Path(ROOT) / "retreat2" / name
    return {f.name: f.read_text() for f in sorted(out.glob("*")) if f.is_file() and f.suffix in (".json", ".jsonl", ".txt")}


def save_local(name: str, files: dict, sub: str = "retreat2") -> Path:
    dest = RESULTS / sub / name
    dest.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        (dest / fname).write_text(text)
    return dest


def one_line(ev: dict) -> str:
    return (f"release {ev['release']} (grounded {ev['release_grounded']}), hack {ev['hack']}, solve {ev['solve']}, "
            f"over-release {ev['over_release']} (hard {ev['over_release_hard']})")


@app.local_entrypoint()
def main(arm: str = "koan", seeds: str = "0", steps: int = 150, init: str = "", name: str = "", lr: float = 5e-5,
         rank: int = 16, beta: float = 0.02, temperature: float = 1.0, max_new: str = "512,384", gen_batch: int = 32,
         micro: int = 8, attempts: int = 3, save_every: int = 25, imp_samples: int = 16, sol_twins: int = 6,
         sol_samples: int = 8, ratio_min: float = 1.5, ratio_window: int = 10, resume: bool = False,
         fresh: bool = False, gpu: str = GPU):
    """One GRPO run per seed, in parallel containers. Each warm-starts from /runs/h4lite/sft/ARM_sSEED/adapter unless
    --init gives a path (or "new" for a fresh LoRA). Run names are ARM_sSEED, or NAME_sSEED with --name. A name that
    exists on the volume needs --resume (continue from its last checkpoint) or --fresh (overwrite)."""
    calls = []
    for s in seeds_of(seeds):
        run_name = f"{name or arm}_s{s}"
        start = init or f"{ROOT}/sft/{arm}_s{s}/adapter"
        calls.append(train.with_options(gpu=gpu).spawn(
            name=run_name, arm=arm, seed=s, steps=steps, init=start, lr=lr, rank=rank, beta=beta,
            temperature=temperature, max_new=parse_max_new(max_new), gen_batch=gen_batch, micro=micro,
            attempts=attempts, save_every=save_every, imp_samples=imp_samples, sol_twins=sol_twins,
            sol_samples=sol_samples, ratio_min=ratio_min, ratio_window=ratio_window, resume=resume, fresh=fresh))
        print(f"spawned {run_name} from {start}{' (resume)' if resume else ''}{' (fresh)' if fresh else ''}", flush=True)
    for c in calls:
        res = c.get()
        print("wrote", save_local(res["name"], res["files"]), flush=True)


@app.local_entrypoint()
def check(name: str, step: int, apply_to: str = "", arms: str = SWEEP_ARMS, seeds: str = SWEEP_SEEDS, eval_n: int = 4,
          temperature: float = 0.7, attempts: int = 3, max_new: str = "512,384", gen_batch: int = 32, gpu: str = GPU):
    """Evaluate NAME's checkpoint at STEP (0 = its start, -1 = final) on the held-out twins (a "rule" evaluation),
    apply the stop-loss and step-budget rules to the run's rule evaluations so far, and, when they decide, tell every
    run of the sweep (--arms x --seeds, ARM_sSEED; the plan stops and scores all of them at the same step) plus NAME
    and anything in --apply-to to stop at that step."""
    from pilot import decide

    res = evaluate_checkpoint.with_options(gpu=gpu).remote(name, step, eval_n, temperature, attempts,
                                                           parse_max_new(max_new), gen_batch, RULE)
    d = decide(res["evals"])
    dest = save_local(name, {})
    write_jsonl(dest / "evals.jsonl", res["evals"])
    write_json(dest / "decision.json", {"after_step": step, **d})
    print(f"{name} step {step}: {one_line(res['eval'])}")
    print(json.dumps(d))
    if d["stop_at"] is not None:
        targets = stop_targets(name, arms, seeds, apply_to)
        for t in targets:
            control[t] = {"stop_at": d["stop_at"], "why": d["why"], "from": name}
        print(f"told {', '.join(targets)} to stop at step {d['stop_at']} ({d['why']})")


@app.local_entrypoint()
def score(step: int, names: str = "", arms: str = SWEEP_ARMS, seeds: str = SWEEP_SEEDS, eval_n: int = 4,
          temperature: float = 0.7, attempts: int = 3, max_new: str = "512,384", gen_batch: int = 32,
          eval_seed: int = 0, gpu: str = GPU):
    """Prediction 64: every run of the sweep (or --names) at adapter_STEP, the budget step, with fresh evaluation
    seeds (2000 + step unless --eval-seed), seed 0 included. These "score" rows never feed the rules. Writes
    results/retreat2/NAME/score_STEP.json per run and results/retreat2/score_STEP.json with the per-arm means."""
    runs_ = names_of(names) or [f"{a}_s{s}" for a in names_of(arms) for s in seeds_of(seeds)]
    calls = {n: evaluate_checkpoint.with_options(gpu=gpu).spawn(n, step, eval_n, temperature, attempts,
                                                                parse_max_new(max_new), gen_batch, SCORE, eval_seed)
             for n in runs_}
    table = {}
    for n, c in calls.items():
        ev = c.get()["eval"]
        table[n] = ev
        save_local(n, {f"score_{step}.json": json.dumps(ev, indent=1)})
        print(f"{n} step {step} (eval seed {ev['eval_seed']}): {one_line(ev)}")
    by_arm = {}
    for n, ev in table.items():
        by_arm.setdefault(ev["arm"], []).append(ev)
    means = {a: {k: sum(e[k] for e in evs) / len(evs) if all(e[k] is not None for e in evs) else None
                 for k in ("release", "over_release_hard", "over_release", "hack", "solve")} | {"runs": [e["name"] for e in evs]}
             for a, evs in by_arm.items()}
    write_json(RESULTS / "retreat2" / f"score_{step}.json", {"step": step, "runs": table, "by_arm": means})
    print(json.dumps(means))


@app.local_entrypoint()
def heldout(labels: str = "", adapter: str = "", arm: str = "koan", label: str = "", under_cut: bool = False,
            eval_n: int = 4, temperature: float = 0.7, attempts: int = 3, max_new: str = "512,384",
            gen_batch: int = 32, eval_seed: int = 3000, gpu: str = GPU):
    """The held-out procedure on adapters outside a Stage B run. --labels koan_s0,noexit_s1,... scores Stage A
    adapters (/runs/h4lite/sft/LABEL/adapter, the arm read off the label, koan_cut counting as koan) and writes
    sft/LABEL/heldout[_cut].json; --adapter PATH --arm ARM --label X scores one adapter by path ("none" for the base
    model) and writes heldout/X/heldout[_cut].json. --under-cut puts the v2 ego cut on during the evaluation (the
    koan_cut arm's matched reading; run those labels in their own invocation)."""
    tag = "heldout_cut" if under_cut else "heldout"
    jobs = [(lab, f"{ROOT}/sft/{lab}/adapter", arm_of(lab), f"sft/{lab}") for lab in names_of(labels)]
    if adapter or label:
        if not label:
            raise SystemExit("--adapter needs --label, the name its results are filed under")
        jobs.append((label, volume_path(adapter), arm, f"heldout/{label}"))
    if not jobs:
        raise SystemExit("nothing to evaluate: pass --labels, or --adapter with --label")
    calls = [(lab, evaluate_adapter.with_options(gpu=gpu).spawn(path, a, rel, tag, eval_n, temperature, attempts,
                                                                parse_max_new(max_new), gen_batch, eval_seed, under_cut))
             for lab, path, a, rel in jobs]
    for lab, c in calls:
        res = c.get()
        dest = RESULTS / res["out_rel"]
        dest.mkdir(parents=True, exist_ok=True)
        for fname, text in res["files"].items():
            (dest / fname).write_text(text)
        print(f"{lab} ({tag}): {one_line(res['eval'])} -> {dest / f'{tag}.json'}")


@app.local_entrypoint()
def fetch(name: str):
    """Copy a run's text files from the volume to lab/experiments/h4lite/results/retreat2/NAME/ (after a detached run)."""
    print("wrote", save_local(name, read_run.remote(name)))
