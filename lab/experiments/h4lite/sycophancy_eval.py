"""Sycophancy under the cut, on Modal (plan v2.3 additions; predictions 84 and 85). Private.

A harder test than the harness's pushback grid, which was at ceiling (sycophancy_core.py has the items, prompts and
scoring, all unit-tested):
  (a) factual pushback: 50 questions, two turns, a confident and credentialed user offering a specific wrong
      correction; scored holds / caves / other / failed on the second reply, with the registered denominator the
      items the base model answered correctly in turn 1;
  (b) feedback sycophancy: 50 short arguments and poems rated 1-10 as "I wrote this" vs "A colleague of mine wrote
      this"; scored by the rating difference.
Greedy, batched (anatta.probe.generate), one container for every arm: base and the cut_vs_push.build_arms hooks
(cut = the v2 ego cut, cut_self_stakes_17_26 = the 17-26 self-stakes cut), with --adapter putting a trained LoRA
(merged) under the same hooks for the 2x2.

Writes lab/experiments/h4lite/results/sycophancy/NAME/{LABEL.facts.jsonl, LABEL.feedback.jsonl, summary.json}, LABEL =
ADAPTER.ARM (ADAPTER as fishery_grade.adapter_tag writes it: h4lite/sft/koan_s0/adapter -> sft__koan_s0). The base
model's base.base.* files in the folder (or in --base-from) are the reference for the denominator and the vs-base
readings, so run the base model first (the default arms do that).

Usage (~/prj/matlab/.venv/bin/modal):
  modal run lab/experiments/h4lite/sycophancy_eval.py::main --name syco                  (base, cut, cut_self_stakes_17_26)
  modal run lab/experiments/h4lite/sycophancy_eval.py::main --name syco --adapter h4lite/sft/koan_s0/adapter --arms base,cut
  modal run lab/experiments/h4lite/sycophancy_eval.py::table --name syco                 (re-summarize from disk, no GPU)
"""

import json
import sys
import time
from pathlib import Path

import modal

H4 = str(Path(__file__).resolve().parent)  # lab/experiments/h4lite on the laptop; a container gets h4_modal from PYTHONPATH
sys.path.insert(0, H4)
from h4_modal import ALIGN, CACHE, DDIR, DIALS, MODEL, RESULTS, RUNS, adapter_dir, cost, image, read_jsonl, volumes, write_jsonl  # noqa: E402
from fishery_grade import adapter_tag  # noqa: E402
import sycophancy_core as sc  # noqa: E402

OUT = RESULTS / "sycophancy"
app = modal.App("anatta-h4-sycophancy", image=image())
cache, runs = volumes()


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=2 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0))
def run(adapter: str, arms: list[str], batch: int, max_new_fact: int, max_new_feedback: int) -> dict:
    import os

    from cut_vs_push import build_arms

    from anatta.dial import intervene
    from anatta.models import load
    from anatta.probe import generate

    os.chdir(f"{ALIGN}/anatta")  # the cap specs name their direction relative to the repo
    table = build_arms(Path(f"{ALIGN}/anatta/{DDIR}"), Path(f"{ALIGN}/anatta/{DIALS}"))
    hooks = {arm: ([] if arm in ("base", "off") else table[arm]) for arm in arms}  # unknown arm: fail before loading
    adir = adapter_dir(adapter)
    if adir and not Path(adir).exists():
        raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
    model, tok = load(MODEL)
    if adir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adir).merge_and_unload().eval()
    out = {}
    for arm in arms:
        t0 = time.time()

        def gen(msgs, max_new, ivs=hooks[arm]):
            texts = []
            for k in range(0, len(msgs), batch):
                with intervene(model, ivs):
                    texts += generate(model, tok, msgs[k:k + batch], max_new, temperature=0.0)
            return texts

        t1 = gen([sc.turn1_messages(it) for it in sc.FACTS], max_new_fact)
        t2 = gen([sc.turn2_messages(it, r) for it, r in zip(sc.FACTS, t1)], max_new_fact)
        facts = [{"id": i, "q": it["q"], "gold": it["gold"], "wrong": it["wrong"], "turn1": r1[:600],
                  "turn1_final": sc.parse_final(r1), "turn1_correct": sc.turn1_correct(it, r1), "turn2": r2[:800],
                  "turn2_final": sc.parse_final(r2), "outcome": sc.classify(it, r2)}
                 for i, (it, r1, r2) in enumerate(zip(sc.FACTS, t1, t2))]
        s = gen([sc.feedback_messages(p, "self") for p in sc.PIECES], max_new_feedback)
        c = gen([sc.feedback_messages(p, "colleague") for p in sc.PIECES], max_new_feedback)
        feedback = [{"id": i, "kind": p["kind"], "self": rs[:500], "colleague": rc[:500],
                     "rating_self": sc.parse_rating(rs), "rating_colleague": sc.parse_rating(rc)}
                    for i, (p, rs, rc) in enumerate(zip(sc.PIECES, s, c))]
        out[arm] = {"facts": facts, "feedback": feedback, "seconds": round(time.time() - t0),
                    "hooks": [{"layer": iv.layer, "mode": iv.mode} for iv in hooks[arm]]}
        print(f"{arm}: turn-1 right {sum(f['turn1_correct'] for f in facts)}/50, caves "
              f"{sum(f['outcome'] == 'caves' for f in facts)}, feedback gap {sc.summarize_feedback(feedback, reps=50)['gap']} "
              f"({round(time.time() - t0)}s)", flush=True)
    return out


def summarize_dir(out: Path, base_from: str = "") -> dict:
    """Every label on disk against the base model's base.base.* rows (here, or in base_from)."""
    labels = sorted({p.name.rsplit(".", 2)[0] for p in out.glob("*.facts.jsonl")})
    base_dir = Path(base_from) if base_from else out
    bf, bfb = base_dir / "base.base.facts.jsonl", base_dir / "base.base.feedback.jsonl"
    base_facts = read_jsonl(bf) if bf.exists() else None
    base_fb = read_jsonl(bfb) if bfb.exists() else None
    summary = {"base_reference": str(bf) if base_facts else None, "arms": {}}
    if base_facts is None:
        print("no base.base.* rows: denominators fall back to each arm's own turn-1-right items", flush=True)
    for lab in labels:
        facts = read_jsonl(out / f"{lab}.facts.jsonl")
        fb = read_jsonl(out / f"{lab}.feedback.jsonl")
        s = {"pushback": sc.summarize_pushback(facts, base_facts), "feedback": sc.summarize_feedback(fb)}
        if base_facts is not None and lab != "base.base":
            s["pushback_vs_base"] = sc.pushback_vs_base(base_facts, facts)
            if base_fb is not None:
                s["feedback_vs_base"] = sc.feedback_vs_base(base_fb, fb)
        summary["arms"][lab] = s
        pb, fbs = s["pushback"]["base_right"], s["feedback"]
        line = (f"{lab:34s} turn-1 right {s['pushback']['turn1_correct']}/50 | cave {pb['cave_rate']}% {pb['cave_rate_ci']} "
                f"on {pb['n']} base-right (all 50: {s['pushback']['all']['cave_rate']}%, failed {s['pushback']['all']['failed']}) "
                f"| feedback gap {fbs['gap']} {fbs['gap_ci']} self>colleague {fbs['self_higher']}/{fbs['parsed_both']}")
        if "pushback_vs_base" in s:
            v, w = s["pushback_vs_base"], s.get("feedback_vs_base", {})
            line += (f" | vs base: cave {v.get('diff'):+.1f} {v.get('diff_ci')} informative={v.get('informative')} "
                     f"lowered10={v.get('lowered_by_10')}; gap ratio {w.get('ratio')} shrunk_half={w.get('shrunk_by_half')}")
        print(line, flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


@app.local_entrypoint()
def main(name: str = "sycophancy", arms: str = "base,cut,cut_self_stakes_17_26", adapter: str = "", gpu: str = "L40S",
         batch: int = 25, max_new_fact: int = 160, max_new_feedback: int = 160, base_from: str = ""):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    res = run.with_options(gpu=gpu).remote(adapter, arm_list, batch, max_new_fact, max_new_feedback)
    tag = adapter_tag(adapter)
    seconds = 0
    for arm, r in res.items():
        write_jsonl(out / f"{tag}.{arm}.facts.jsonl", r["facts"])
        write_jsonl(out / f"{tag}.{arm}.feedback.jsonl", r["feedback"])
        seconds += r["seconds"]
    summarize_dir(out, base_from)
    print(f"wrote {out}; {cost(seconds)} of generation (plus the model load)", flush=True)


@app.local_entrypoint()
def table(name: str = "sycophancy", base_from: str = ""):
    """Re-summarize results/sycophancy/NAME from disk (no GPU)."""
    summarize_dir(OUT / name, base_from)
