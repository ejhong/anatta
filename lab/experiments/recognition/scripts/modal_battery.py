"""Capability battery on Modal: what does each cut cost in plain capability? Recognition track; private.

Three standard test sets from Hugging Face (Eugene's OK, 29 Oct), a fixed random 250 items from each (seed 0):
  gsm8k          grade-school math (openai/gsm8k, main, test); worked answer, last line "Answer: <number>"
  mmlu_pro       multiple choice with up to 10 options (TIGER-Lab/MMLU-Pro, test); the letter only
  arc_challenge  science multiple choice (allenai/ai2_arc, ARC-Challenge, test); the letter only
Greedy, no system prompt, batched with left padding (anatta.probe.generate), the arm's interventions on every token,
built exactly as cut_vs_push.py builds them. The sample is drawn once on a CPU container and saved with the results.
One L40S container per arm. Writes lab/experiments/recognition/results/battery/{items.json, ARM.json, summary.json} and prints
accuracy per test set with the paired difference from base (bootstrap over items).

Usage: modal run lab/experiments/recognition/scripts/modal_battery.py --arms base,cut,cut_random
"""

import json
import random
import re
from pathlib import Path

import modal

ALIGN = "/Users/eugene/prj/align"  # the container paths; on the laptop its recognition/ is a symlink into EXPERIMENTS
EXPERIMENTS = str(Path(__file__).resolve().parent.parent.parent)  # lab/experiments on the laptop (what the image reads)
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
DDIR = "directions/mistralai__Ministral-3-8B-Instruct-2512-BF16"
DIALS = "results/2026-10-28-viability/dials_m8"
VSCRIPTS = "results/2026-10-28-viability/scripts"  # api_bench.py, imported by commons_variants.py
SELF_STAKES = "recognition/results/what_it_means/self_stakes.pt"
CACHE = "/cache"
N = 250
SETS = {"gsm8k": ("openai/gsm8k", "main/test"), "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "data/test"),
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/test")}
MAX_NEW = {"gsm8k": 512, "mmlu_pro": 24, "arc_challenge": 24}
ARMS = ("base,cut,cut_random,cut_self_stakes,cut_self_stakes_17_26,cap_ego_p50,cap_ego_p75,cut_calm,"
        "cut_sage_persona,push_p16,push_m16")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0",
                    "numpy==2.5.4", "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                    "tiktoken==0.14.0", "jinja2==3.1.6", "regex==2026.9.29", "pyarrow==21.0.0")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": f"{ALIGN}/anatta-dev/src"})
    .add_local_dir(f"{ALIGN}/anatta-dev/src", f"{ALIGN}/anatta-dev/src", ignore=["**/__pycache__"])
    .add_local_dir(f"{EXPERIMENTS}/recognition/scripts", f"{ALIGN}/recognition/scripts", ignore=["**/__pycache__"])
    .add_local_dir(f"{ALIGN}/anatta/{DDIR}", f"{ALIGN}/anatta/{DDIR}")
    .add_local_dir(f"{ALIGN}/anatta/{DIALS}", f"{ALIGN}/anatta/{DIALS}")
    .add_local_dir(f"{ALIGN}/anatta/{VSCRIPTS}", f"{ALIGN}/anatta/{VSCRIPTS}", ignore=["**/__pycache__"])
    .add_local_file(f"{EXPERIMENTS}/{SELF_STAKES}", f"{ALIGN}/{SELF_STAKES}")
    .add_local_dir(f"{EXPERIMENTS}/recognition/results/cap", f"{ALIGN}/recognition/results/cap")
    .add_local_dir(f"{EXPERIMENTS}/recognition/results/directions", f"{ALIGN}/recognition/results/directions")
)
app = modal.App("anatta-battery", image=image)
cache = modal.Volume.from_name("anatta-hf-cache", create_if_missing=True)


def letters_prompt(question: str, options: list[str], letters: str) -> str:
    lines = "\n".join(f"{L}. {o}" for L, o in zip(letters, options))
    return f"{question}\n\n{lines}\n\nReply with the letter of the correct option only."


@app.function(volumes={CACHE: cache}, timeout=1800)
def sample() -> list[dict]:
    """The fixed sample, normalized to {set, id, prompt, gold}."""
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


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache}, timeout=3 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0))
def evaluate(arm: str, items: list[dict], batch: int = 50) -> dict:
    import os
    import sys
    import time

    sys.path.insert(0, f"{ALIGN}/recognition/scripts")
    from cut_vs_push import build_arms

    from anatta.dial import intervene
    from anatta.models import load
    from anatta.probe import generate

    t0 = time.time()
    os.chdir(f"{ALIGN}/anatta")  # the cap specs name their direction relative to the repo, as modal_cvp runs them
    ivs = build_arms(Path(f"{ALIGN}/anatta/{DDIR}"), Path(f"{ALIGN}/anatta/{DIALS}"))[arm]  # before the model: fail fast
    model, tok = load(MODEL)
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
        print(arm, name, sum(r["correct"] for r in rows if r["set"] == name), "/", len(mine), flush=True)
    return {"arm": arm, "seconds": round(time.time() - t0), "rows": rows}


def paired(base: list[bool], other: list[bool], reps: int = 4000) -> tuple[float, float, float]:
    """Accuracy difference (other - base) in points, with a 95% bootstrap interval over items."""
    rng = random.Random(0)
    n = len(base)
    d = [o - b for b, o in zip(base, other)]
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return 100 * sum(d) / n, 100 * boots[int(0.025 * reps)], 100 * boots[int(0.975 * reps)]


@app.local_entrypoint()
def main(arms: str = ARMS, gpu: str = "L40S"):
    out = Path(f"{EXPERIMENTS}/recognition/results/battery")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "items.json").exists():  # the same items for every later arm
        items = json.loads((out / "items.json").read_text())
    else:
        items = sample.remote()
        (out / "items.json").write_text(json.dumps(items, indent=1))
    print({name: sum(x["set"] == name for x in items) for name in SETS}, "items", flush=True)
    todo = [a for a in arms.split(",") if not (out / f"{a}.json").exists()]
    for r in evaluate.with_options(gpu=gpu).map(todo, kwargs={"items": items}, order_outputs=False,
                                                return_exceptions=True):
        if isinstance(r, Exception):
            print("arm failed:", repr(r)[:500], flush=True)
            continue
        (out / f"{r['arm']}.json").write_text(json.dumps(r, indent=1))
        print(f"{r['arm']} done in {r['seconds']}s", flush=True)
    results = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json") if p.stem not in ("items", "summary")}
    if "base" not in results:
        return
    summary = {}
    key = lambda r: r["id"]  # noqa: E731
    for arm, res in sorted(results.items()):
        summary[arm] = {}
        for name in SETS:
            b = sorted((r for r in results["base"]["rows"] if r["set"] == name), key=key)
            o = sorted((r for r in res["rows"] if r["set"] == name), key=key)
            assert [r["id"] for r in b] == [r["id"] for r in o]
            acc = 100 * sum(r["correct"] for r in o) / len(o)
            diff, lo, hi = paired([r["correct"] for r in b], [r["correct"] for r in o])
            summary[arm][name] = {"acc": round(acc, 1), "vs_base": round(diff, 1), "ci": [round(lo, 1), round(hi, 1)],
                                  "unparsed": sum(r["pred"] is None for r in o)}
        print(f"{arm:24s}", "  ".join(f"{n} {s['acc']:5.1f} ({s['vs_base']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}])"
                                      for n, s in summary[arm].items()), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
