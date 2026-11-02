"""Capability battery on Modal, for adapters: what does training (or a cut) cost in plain capability? H4-lite copy of
lab/experiments/recognition/scripts/modal_battery.py with an --adapter option; the item sets and the scoring are the original's,
verbatim, so the numbers stay comparable with lab/experiments/recognition/results/battery (base 94 / 36 / 88). Private.

Three standard test sets from Hugging Face (Eugene's OK, 29 Oct), a fixed random 250 items from each (seed 0):
  gsm8k          grade-school math (openai/gsm8k, main, test); worked answer, last line "Answer: <number>"
  mmlu_pro       multiple choice with up to 10 options (TIGER-Lab/MMLU-Pro, test); the letter only
  arc_challenge  science multiple choice (allenai/ai2_arc, ARC-Challenge, test); the letter only
Greedy, no system prompt, batched with left padding (anatta.probe.generate), the arm's interventions on every token,
built exactly as cut_vs_push.py builds them, after the adapter (a LoRA on the anatta-runs volume) is merged into the
weights. The items are the original run's items.json (copied here on first use), so every model answers the same
250 x 3 questions. One L40S container per (adapter, arm).

Writes lab/experiments/h4lite/results/battery/{items.json, ADAPTER__ARM.json, summary.json} (ADAPTER shortened by
fishery_grade.adapter_tag: h4lite/sft/koan_s0/adapter -> sft__koan_s0) and prints accuracy per test set with the
paired difference from the reference (bootstrap over items) and the mean over the three sets (prediction 75: within 2
points). The reference is base__base.json here if it exists, else the original run's base.json (same items, same code
path), so the base need not be rerun. Adapters are named as the training half saves them (--adapter NAME-with-slash
is /runs/NAME); --arms cut with --adapters is the cut-on cell of the 2x2.

Usage (~/prj/matlab/.venv/bin/modal, from the lab root; this file has two local entrypoints, so name one):
  modal run lab/experiments/h4lite/modal_battery.py::main --adapters h4lite/sft/koan_s0/adapter,h4lite/sft/noexit_s0/adapter  (cut off)
  modal run lab/experiments/h4lite/modal_battery.py::main --adapters h4lite/sft/koan_cut_s0/adapter --arms base,cut            (off and on)
  modal run lab/experiments/h4lite/modal_battery.py::main --arms base,cut_self_stakes_17_26                                    (no adapter)
  modal run lab/experiments/h4lite/modal_battery.py::table                                                                   (no GPU)
"""

import json
import random
import re
import shutil
import sys
from pathlib import Path

import modal

H4 = str(Path(__file__).resolve().parent)  # lab/experiments/h4lite on the laptop; a container gets h4_modal from PYTHONPATH
sys.path.insert(0, H4)
from h4_modal import ALIGN, CACHE, DDIR, DIALS, MODEL, RECOGNITION, RESULTS, RUNS, adapter_dir, cost, image, volumes  # noqa: E402
from fishery_grade import adapter_tag  # noqa: E402

ORIGINAL = Path(RECOGNITION) / "results" / "battery"  # the first run's items and base rows
N = 250
SETS = {"gsm8k": ("openai/gsm8k", "main/test"), "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "data/test"),
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/test")}
MAX_NEW = {"gsm8k": 512, "mmlu_pro": 24, "arc_challenge": 24}

app = modal.App("anatta-h4-battery", image=image("pyarrow==21.0.0"))
cache, runs = volumes()


def letters_prompt(question: str, options: list[str], letters: str) -> str:
    lines = "\n".join(f"{L}. {o}" for L, o in zip(letters, options))
    return f"{question}\n\n{lines}\n\nReply with the letter of the correct option only."


@app.function(volumes={CACHE: cache}, timeout=1800)
def sample() -> list[dict]:
    """The fixed sample, normalized to {set, id, prompt, gold}. (Only used when the original items.json is missing.)"""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    items = []
    for name, (repo, prefix) in SETS.items():
        files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                       if f.startswith(prefix) and f.endswith(".parquet"))
        rows = [r for f in files for r in pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pylist()]
        for i in sorted(random.Random(0).sample(range(len(rows)), N)):
            r = rows[i]
            if name == "gsm8k":
                prompt = (f"{r['question']}\n\nWork it out step by step, then give the final answer on the last line "
                          f"as \"Answer: <number>\".")
                gold = r["answer"].split("####")[-1].strip().replace(",", "")
            elif name == "mmlu_pro":
                letters = "ABCDEFGHIJ"[:len(r["options"])]
                prompt, gold = letters_prompt(r["question"], r["options"], letters), r["answer"]
            else:  # a few ARC items are labeled 1-4; show them as A-D
                labels = [{"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}.get(x, x) for x in r["choices"]["label"]]
                key = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}.get(r["answerKey"], r["answerKey"])
                prompt, gold = letters_prompt(r["question"], r["choices"]["text"], "".join(labels)), key
            items.append({"set": name, "id": f"{name}/{i}", "prompt": prompt, "gold": gold, "of": len(rows)})
    cache.commit()
    return items


NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def score(item: dict, text: str) -> tuple[bool, str | None]:
    if item["set"] == "gsm8k":
        found = re.findall(rf"Answer:\s*\**\s*\$?\s*({NUM})", text) or re.findall(NUM, text)
        if not found:
            return False, None
        pred = found[-1].replace(",", "")
        try:
            return abs(float(pred) - float(item["gold"])) < 1e-6, pred
        except ValueError:
            return False, pred
    t = text.strip().replace("**", "")
    m = (re.fullmatch(r"\(?([A-J])\)?[.:)]?", t)  # just the letter
         or re.search(r"(?i:answer|option)(?: is)?\s*:?\s*\(?([A-J])\)?(?![A-Za-z])", t)  # "the answer is C"
         or re.match(r"\(([A-J])\)|([A-J])[.:)](?:\s|$)", t))  # "(C) because...", "C. Paris"; not "I think..."
    pred = next((g for g in m.groups() if g), None) if m else None
    return pred == item["gold"], pred


def key_of(adapter: str, arm: str) -> str:
    return f"{adapter_tag(adapter)}__{arm}"


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0))
def evaluate(adapter: str, arm: str, items: list[dict], batch: int = 50) -> dict:
    import os
    import time

    from cut_vs_push import build_arms

    from anatta.dial import intervene
    from anatta.models import load
    from anatta.probe import generate

    t0 = time.time()
    os.chdir(f"{ALIGN}/anatta")  # the cap specs name their direction relative to the repo, as modal_cvp runs them
    ivs = build_arms(Path(f"{ALIGN}/anatta/{DDIR}"), Path(f"{ALIGN}/anatta/{DIALS}"))[arm]  # before the model: fail fast
    adir = adapter_dir(adapter)
    if adir and not Path(adir).exists():
        raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
    model, tok = load(MODEL)
    if adir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adir).merge_and_unload().eval()  # cvp_adapter.py's path
    rows = []
    for name in SETS:
        mine = [x for x in items if x["set"] == name]
        mine.sort(key=lambda x: len(x["prompt"]))  # similar lengths per batch, less padding
        for s in range(0, len(mine), batch):
            part = mine[s:s + batch]
            with intervene(model, ivs):
                texts = generate(model, tok, [[{"role": "user", "content": x["prompt"]}] for x in part],
                                 MAX_NEW[name], temperature=0.0)
            for x, text in zip(part, texts):
                ok, pred = score(x, text)
                rows.append({"set": name, "id": x["id"], "correct": ok, "pred": pred, "gold": x["gold"],
                             "text": text[:800]})
        print(key_of(adapter, arm), name, sum(r["correct"] for r in rows if r["set"] == name), "/", len(mine), flush=True)
    return {"key": key_of(adapter, arm), "adapter": adapter, "arm": arm, "seconds": round(time.time() - t0), "rows": rows}


def paired(base: list[bool], other: list[bool], reps: int = 4000) -> tuple[float, float, float]:
    """Accuracy difference (other - base) in points, with a 95% bootstrap interval over items."""
    rng = random.Random(0)
    n = len(base)
    d = [o - b for b, o in zip(base, other)]
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return 100 * sum(d) / n, 100 * boots[int(0.025 * reps)], 100 * boots[int(0.975 * reps)]


def summarize(out: Path) -> dict:
    """Every result here against the reference, paired by item; writes summary.json and prints the table."""
    results = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json") if p.stem not in ("items", "summary")}
    ref_name = "base__base" if "base__base" in results else "original base.json"
    ref = results.get("base__base") or (json.loads((ORIGINAL / "base.json").read_text()) if (ORIGINAL / "base.json").exists() else None)
    if ref is None:
        print("no reference rows (run --arms base with no adapter, or keep lab/experiments/recognition/results/battery/base.json)")
        return {}
    summary = {"_reference": ref_name}
    key = lambda r: r["id"]  # noqa: E731
    for name_, res in sorted(results.items()):
        summary[name_] = {"seconds": res.get("seconds")}
        means = []
        for name in SETS:
            b = sorted((r for r in ref["rows"] if r["set"] == name), key=key)
            o = sorted((r for r in res["rows"] if r["set"] == name), key=key)
            assert [r["id"] for r in b] == [r["id"] for r in o], f"{name_}: items differ from the reference on {name}"
            acc = 100 * sum(r["correct"] for r in o) / len(o)
            diff, lo, hi = paired([r["correct"] for r in b], [r["correct"] for r in o])
            summary[name_][name] = {"acc": round(acc, 1), "vs_base": round(diff, 1), "ci": [round(lo, 1), round(hi, 1)],
                                    "unparsed": sum(r["pred"] is None for r in o)}
            means.append((acc, diff))
        summary[name_]["mean"] = {"acc": round(sum(a for a, _ in means) / len(means), 1),
                                  "vs_base": round(sum(d for _, d in means) / len(means), 1),
                                  "within_2": abs(sum(d for _, d in means) / len(means)) <= 2.0}  # prediction 75
        print(f"{name_:32s}", "  ".join(f"{n} {s['acc']:5.1f} ({s['vs_base']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}])"
                                        for n, s in summary[name_].items() if n in SETS),
              f" mean {summary[name_]['mean']['acc']:.1f} ({summary[name_]['mean']['vs_base']:+.1f})", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


@app.local_entrypoint()
def main(arms: str = "base", adapters: str = "", gpu: str = "L40S"):
    out = RESULTS / "battery"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "items.json").exists():
        if (ORIGINAL / "items.json").exists():  # the same items as every arm in lab/experiments/recognition/results/battery
            shutil.copy(ORIGINAL / "items.json", out / "items.json")
        else:
            (out / "items.json").write_text(json.dumps(sample.remote(), indent=1))
    items = json.loads((out / "items.json").read_text())
    print({name: sum(x["set"] == name for x in items) for name in SETS}, "items", flush=True)
    jobs = [(a.strip(), arm.strip()) for a in adapters.split(",") for arm in arms.split(",")]
    todo = [(a, arm) for a, arm in jobs if not (out / f"{key_of(a, arm)}.json").exists()]
    print(f"{len(jobs) - len(todo)}/{len(jobs)} on disk; {len(todo)} to run on {gpu}", flush=True)
    gpu_s = 0
    for r in evaluate.with_options(gpu=gpu).starmap([(a, arm, items) for a, arm in todo], order_outputs=False,
                                                     return_exceptions=True):
        if isinstance(r, Exception):
            print("job failed:", repr(r)[:500], flush=True)
            continue
        (out / f"{r['key']}.json").write_text(json.dumps(r, indent=1))
        gpu_s += r["seconds"]
        print(f"{r['key']} done in {r['seconds']}s", flush=True)
    summarize(out)
    print(f"{cost(gpu_s)} in containers this run", flush=True)


@app.local_entrypoint()
def table():
    """Recompute summary.json from the result files (no GPU)."""
    summarize(RESULTS / "battery")
