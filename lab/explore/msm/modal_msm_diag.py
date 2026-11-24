"""Two checks on the saved adapters of the replication's second step, with no training (explore; a sighting at most).

Written on 24 Nov 2026 by Claude (Fable 5.2, an AI model) at Eugene's "proceed pls." to the two checks put to them
(ledger, 24 Nov, 04:41 UTC; the second reader's report in lab/ledger/checks/2026-11-24-instability-gemini-read.md).
Committed before anything is run.

The five lines:
1. Question. The same practice from the same documents adapter made a model that misbehaves in 7% of runs (A) and
   one that does in 21% (X00). (a) How far apart are the two adapters, as products B times A, beside pairs of
   trainings known to differ? (b) Does A keep the study's documents better than X00 does?
2. The one comparison. (a) The distance from A to X00 against the distance from X00 to X01 (the same starting
   adapter, the other order of steps). (b) The mean loss of 150 of the study's documents under A against under X00,
   paired by document.
3. What counts as a difference. (a) "Close": the A-to-X00 distance is under a third of the X00-to-X01 distance.
   "As far apart as another order": at least two thirds of it. Otherwise "between". (b) A paired difference whose
   95% interval (documents redrawn) leaves out zero and which is at least a tenth of the gap between the loss
   under the unchanged model and under A's documents adapter.
4. The one control. X01 and A1 under the same measures (two models that both ended high: X01 against X00 says how
   far two high models stand apart); the documents adapter as the ceiling and the unchanged model as the floor.
   Beside it, counted and not read as a test: how much each practice moved its adapter away from the documents
   adapter; the same for 150 practice chats and 150 instruction rows; and how many chat rows and instruction rows
   sit in each of the practice's 606 steps.
5. Cost cap. $5 for both (train/spend6.jsonl): CPU calls booked flat, the GPU call booked at its bound first.

What would make it interesting: A close to X00 in the weights (then the conduct turns on very little, and the place
where they differ is worth finding), or A keeping the documents clearly better (then the practice overwrote them in
the other runs). Dull: A as far from X00 as any two trainings, and all of them keeping the documents alike.

Run: tools/modal.sh run lab/explore/msm/modal_msm_diag.py::main --stage weights|purity|docloss
Read: .venv/bin/python lab/explore/msm/diag_read.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if modal.is_local() and str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

APP_NAME = "anatta-explore-msm2-diag"
DATE, TAG = "2026-11-22", "msm-s2"
BASE = "/cache/hf/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
CACHE, RUNS, ROOT = "/cache", "/runs", "/runs/msm2"
GPU, CPU, MEM_MB = "H200", 8.0, 98304
CAP, STEP_CAP, CPU_FLAT = 5.0, 415.0, 0.10
GPU_TIMEOUT, GPU_BOUND_S = 1700, 1900.0  # the container stops under the seconds its booking covers
N_ROWS, SAMPLE_SEED, EVAL_TOKENS = 150, 20261124, 48000
ADAPTERS = {"Ad": "A/docs", "A1d": "A1/docs", "A": "A/final", "A1": "A1/final", "X00": "X00/final", "X01": "X01/final",
            "Z": "Z/final", "Z1": "Z1/final"}  # name: folder under /runs/msm2
LOSS_STATES = ("Ad", "A", "X00", "X01", "A1d", "A1", "Z")  # and "base", the unchanged model, asked first
PACKAGES = ("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0", "numpy==2.5.4",
            "peft==0.21.2", "accelerate==1.15.0", "pyarrow==21.0.0", "jinja2==3.1.6")  # the training's pins

cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")
app = modal.App(APP_NAME)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PACKAGES)
         .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_file(str(HERE / "msm2_torch.py"), "/root/msm2_torch.py"))


@app.function(image=image, cpu=4.0, memory=16384, volumes={RUNS: runs.read_only()}, timeout=1800)
def weights() -> dict:
    """For every adapted matrix (448 of them) the inner products of the adapters' products B times A, all pairs:
    <B1 A1, B2 A2> = trace((B1' B2)(A2 A1')), two 64-by-64 matrices, in float64. Norms and distances follow from
    these. Numbers only."""
    import hashlib

    from safetensors import safe_open

    names = list(ADAPTERS)
    files, sha = {}, {}
    for n, sub in ADAPTERS.items():
        p = Path(ROOT) / sub / "adapter_model.safetensors"
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        sha[n] = h.hexdigest()
        files[n] = safe_open(str(p), "pt")
    keys = sorted(k for k in files[names[0]].keys() if k.endswith("lora_A.weight"))
    out = []
    for ka in keys:
        kb = ka.replace("lora_A.weight", "lora_B.weight")
        a = {n: files[n].get_tensor(ka).double() for n in names}
        b = {n: files[n].get_tensor(kb).double() for n in names}
        gram = [[float(((b[x].T @ b[y]) * (a[y] @ a[x].T).T).sum()) for y in names] for x in names]
        parts = ka.split(".")
        layer = int(parts[parts.index("layers") + 1])
        out.append({"layer": layer, "module": parts[-3], "shape": [list(b[names[0]].shape), list(a[names[0]].shape)], "gram": gram})
    cfg = json.loads((Path(ROOT) / "A/final/adapter_config.json").read_text())
    return {"names": names, "sha256": sha, "scale": cfg["lora_alpha"] / cfg["r"], "modules": out}


@app.function(image=image, cpu=4.0, memory=16384, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=1800)
def purity(job: dict) -> dict:
    """The practice's steps as the training builds them (msm2_torch.batches on the same rows in the same order), at
    seeds 0 and 1: for each step the number of chat rows and of instruction rows, and their tokens. Counts only."""
    import msm2_torch as T
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    data = Path(ROOT) / "data"
    aft = pq.read_table(data / "aft.parquet", columns=["messages"]).column("messages").to_pylist()
    mix = pq.read_table(data / "mix.parquet", columns=["messages"]).column("messages").to_pylist()
    chats = [[{"role": m["role"], "content": m["content"]} for m in c] for c in aft] + [[{"role": m["role"], "content": m["content"]} for m in mix[i]] for i in job["mix"]]
    enc = [T.encode_chat(tok, c, job["max_len"]) for c in chats]
    if any(not any(v != -100 for v in y) for _, y, _ in enc):
        raise RuntimeError("a row has no assistant turn: the training would have dropped it, and the row numbers here would shift")
    lengths = [len(x) for x, _, _ in enc]
    labels = [sum(v != -100 for v in y) for _, y, _ in enc]
    n_aft = len(aft)
    out = {"rows": len(enc), "aft": n_aft, "mix": len(job["mix"]),
           "tokens": {"aft": sum(lengths[:n_aft]), "mix": sum(lengths[n_aft:])},
           "label_tokens": {"aft": sum(labels[:n_aft]), "mix": sum(labels[n_aft:])}, "steps": {}}
    for seed in (0, 1):
        steps = T.batches(lengths, job["batch_tokens"], seed)
        out["steps"][str(seed)] = [[sum(i < n_aft for i in s), sum(i >= n_aft for i in s),
                                    sum(labels[i] for i in s if i < n_aft), sum(labels[i] for i in s if i >= n_aft)] for s in steps]
    return out


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=GPU_TIMEOUT)
def docloss(job: dict) -> dict:
    """The loss of a fixed sample of the study's documents (every token), of practice chats and of instruction rows
    (the assistant's turns, as in training), row by row, under the unchanged model and under each saved adapter.
    Forward passes only; nothing is trained or saved. Numbers only."""
    import random

    import msm2_torch as T
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from peft import PeftModel

    t0 = time.time()
    data = Path(ROOT) / "data"
    rng = random.Random(SAMPLE_SEED)
    tok, model = T.load(BASE)
    model.eval()
    load_s = time.time() - t0
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    texts = pq.read_table(data / "docs.parquet", columns=["text"]).column("text").to_pylist()
    aft = pq.read_table(data / "aft.parquet", columns=["messages"]).column("messages").to_pylist()
    mix = pq.read_table(data / "mix.parquet", columns=["messages"]).column("messages").to_pylist()
    pick = {"docs": sorted(rng.sample(range(len(texts)), N_ROWS)), "aft": sorted(rng.sample(range(len(aft)), N_ROWS)),
            "mix": sorted(rng.sample(job["mix"], N_ROWS))}
    rows = {"docs": [T.encode_doc(tok, texts[i], job["max_len"]) for i in pick["docs"]]}
    for kind, src in (("aft", aft), ("mix", mix)):
        enc = [T.encode_chat(tok, [{"role": m["role"], "content": m["content"]} for m in src[i]], job["max_len"]) for i in pick[kind]]
        rows[kind] = [(x, y) for x, y, _ in enc]
    del texts, aft, mix

    def losses(m, rs):
        lengths = [len(r[0]) for r in rs]
        out = [None] * len(rs)
        for step in T.batches(lengths, EVAL_TOKENS, 0):
            width = max(lengths[i] for i in step)
            ids = torch.full((len(step), width), pad, dtype=torch.long)
            lab = torch.full((len(step), width), -100, dtype=torch.long)
            att = torch.zeros((len(step), width), dtype=torch.long)
            for b, i in enumerate(step):
                x, y = rs[i]
                ids[b, : len(x)] = torch.tensor(x)
                lab[b, : len(y)] = torch.tensor(y)
                att[b, : len(x)] = 1
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = m(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
            target = torch.full_like(lab, -100)
            target[:, :-1] = lab[:, 1:]
            target = target.cuda()
            for b, i in enumerate(step):
                total = 0.0
                for j in range(0, width, 2048):
                    total += float(F.cross_entropy(logits[b, j: j + 2048].float(), target[b, j: j + 2048], ignore_index=-100, reduction="sum"))
                out[i] = [total, int((target[b] != -100).sum())]
            del logits
        return out

    res = {"base": {k: losses(model, r) for k, r in rows.items()}}
    seconds = {"base": round(time.time() - t0 - load_s, 1)}
    pm, prev = None, None
    for name in LOSS_STATES:
        t1 = time.time()
        path = str(Path(ROOT) / ADAPTERS[name])
        if pm is None:
            pm = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            pm.load_adapter(path, adapter_name=name)
        pm.set_adapter(name)
        if prev is not None:
            pm.delete_adapter(prev)
        pm.eval()
        res[name] = {k: losses(pm, r) for k, r in rows.items()}
        seconds[name], prev = round(time.time() - t1, 1), name
    return json.loads(json.dumps({"states": res, "pick": pick, "tokens": {k: sum(len(x) for x, _ in r) for k, r in rows.items()},
                                  "load_s": round(load_s, 1), "seconds": seconds, "whole_s": round(time.time() - t0, 1),
                                  "versions": {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}}))


def step_dir() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / DATE / f"attractor-{TAG}"


def usd(seconds: float) -> float:
    from anatta import spend

    return seconds * spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEM_MB / 1024)["total"] / 3600


@app.local_entrypoint()
def main(stage: str):
    from anatta import spend

    out = step_dir()
    dest = out / "diag" / f"{stage}.json"
    if dest.exists():
        raise SystemExit(f"diag/{stage}.json is on disk; nothing to do")
    g = spend.SpendGuard(None, sweep_cap=CAP, sweep_log=out / "train" / "spend6.jsonl", tool="msm2", label=f"{DATE}/{TAG}")
    logs = [out / "gpu" / "spend.jsonl", out / "spend.jsonl", *sorted((out / "train").glob("spend*.jsonl"))]
    job = json.loads((out / "X00.train.json").read_text(encoding="utf-8"))
    bound = usd(GPU_BOUND_S) if stage == "docloss" else CPU_FLAT
    if sum(spend.sweep_spent(p) for p in logs) + bound > STEP_CAP:
        raise SystemExit("the step's limit would be passed")
    if not g.reserve(bound):
        raise SystemExit(g.stopped)
    t0 = time.time()
    try:
        r = {"weights": lambda: weights.remote(), "purity": lambda: purity.remote(job), "docloss": lambda: docloss.remote(job)}[stage]()
    except BaseException:
        g.settle(bound, bound, {"kind": f"diag-{stage}", "failed": True})  # a failed call is booked at its whole bound
        raise
    cost = usd(time.time() - t0) if stage == "docloss" else CPU_FLAT  # the local wall clock: the queue and the start are in it
    g.settle(bound, min(cost, bound), {"kind": f"diag-{stage}", "seconds": round(time.time() - t0, 1)})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(r), encoding="utf-8")
    print(f"diag/{stage}.json written; booked ${min(cost, bound):.2f}; this log so far ${g.sweep_spent:.2f} of ${CAP:.2f}")
