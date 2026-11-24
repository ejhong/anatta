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

ADDENDUM, 24 Nov 2026 (Claude, Fable 5.2): a third check, stage `weights2`, the study's four released adapters
beside the lab's eight. A second reader proposed it (lab/ledger/checks/2026-11-24-instability-gemini-read.md, the
third report); the driver judged it a routine step at about ten cents, with no training. Committed before it is run.
1. Question. As products B times A: (a) is the authors' practice-alone adapter the size of the lab's, and their
   documents-alone adapter the size of the lab's? (b) Is their two-stage adapter a continuation of their released
   documents adapter, as the lab's is of its own?
2. The one comparison. (a) The norm of their practice-alone adapter (aft) over the mean norm of the lab's (Z, Z1);
   the norm of their documents-alone adapter (msm) over the mean of the lab's (Ad, A1d). (b) The cosine of their
   two-stage adapter (msm_aft) with their documents adapter (msm).
3. What counts as a difference. (a) "Alike" when the ratio lies from 0.8 to 1.25; "clearly different" when it is
   under 0.67 or over 1.5; otherwise "between". (b) "A continuation" at a cosine of 0.7 or more; "a separate
   training" at 0.5 or less; otherwise "between".
4. The one control. From the first check: two of the lab's own trainings of one thing have norms within 2% of
   each other (Z 51.8 and Z1 52.4; Ad 133.3 and A1d 133.3) and a cosine of 0.32 and 0.38, and the lab's two-stage
   adapter has a cosine of about 0.87 with the documents adapter it continued. Beside it, counted and not read as
   a test: each adapter's settings file (rank, alpha, the matrices adapted); norms by band of layers and kind of
   matrix; the cosine of A, and of X00, with their two-stage adapter; the baseline adapter beside the rest.
5. Cost cap. $0.10, a CPU call booked flat in the same log (train/spend6.jsonl, $5).
A norm depends on more than the rate and the number of steps (the first weights, the weight decay, the data), so a
difference of size points at the authors' training without measuring any one setting of it.
What would make it interesting: their practice-alone adapter clearly smaller than the lab's (then the lab's
practice moves the weights further than theirs did, a difference that could be repaired), or A clearly nearer
their two-stage adapter than X00 is. Dull: sizes alike, and A and X00 equally unlike theirs.
Run: tools/modal.sh run lab/explore/msm/modal_msm_diag.py::main --stage weights2

SECOND ADDENDUM, 24 Nov 2026 (Claude, Fable 5.2), at Eugene's "yes proceed." (ledger, 13:56 UTC): a fourth check,
stage `docloss2`, the documents' loss under the authors' adapters. Committed before it is run.
1. Question. The authors' documents adapter is 1.77 times the size of the lab's. Does it also fit the study's
   documents more closely than the lab's documents adapter does?
2. The one comparison. The mean loss of the same 150 documents as in the second check under the authors'
   documents adapter (msm) against under the lab's (Ad), paired by document.
3. What counts as a difference. As in the second check: a paired difference whose 95% interval (documents
   redrawn) leaves out zero and which is at least a tenth of the gap between the unchanged model (1.617) and the
   lab's documents adapter (0.814), that is 0.080.
4. The one control. Ad is measured again in the same call and must come back within 0.005 of its 0.814, or
   nothing is read. Beside it, counted and not read as a test: the authors' two-stage adapter (msm_aft) and
   their practice-alone adapter (aft); the same for 150 practice chats and 150 instruction rows.
5. Cost cap. One GPU call booked at a bound of 1,000 seconds ($2.21), in the same log (train/spend6.jsonl, $5).
The documents were trained on by both parties, so a lower loss says a closer fit of these documents (more passes,
a higher rate or more steps), not a better model. What would make it interesting: the authors' adapter clearly
lower (their documents stage is a heavier training of the same documents). Dull: alike.
Run: tools/modal.sh run lab/explore/msm/modal_msm_diag.py::main --stage docloss2
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
LOSS_STATES = ("Ad", "A", "X00", "X01", "Z", "A1", "A1d")  # and "base", the unchanged model, asked first; in the order of need
STOP_AT_S = 1450.0  # no new adapter is begun after this many seconds: what is done is returned, under the container's limit
LOSS2 = (("Ad", "runs:A/docs"), ("msm", "hf:msm"), ("msm_aft", "hf:msm_aft"), ("aft", "hf:aft"))  # docloss2: the control first, then in the order of need
GPU2_TIMEOUT, GPU2_BOUND_S, STOP2_AT_S = 900, 1000.0, 680.0  # docloss2: the container stops under the seconds its booking covers
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
def weights2(repos: dict) -> dict:
    """The same inner products as weights(), for the study's released adapters (name: Hugging Face repository, on the
    cache volume) and the lab's eight. Each adapter keeps its own scaling (alpha over rank, read from its settings
    file); matrices are matched by layer and name; only matrices every adapter has are counted. Numbers only."""
    import glob
    import hashlib

    import torch
    from safetensors import safe_open

    paths = {}
    for n, repo in repos.items():
        hits = sorted(glob.glob(f"{CACHE}/hf/hub/models--{repo.replace('/', '--')}/snapshots/*"))
        if len(hits) != 1:
            raise RuntimeError(f"{repo}: {len(hits)} snapshots on the volume, expected one")
        paths[n] = Path(hits[0])
    for n, sub in ADAPTERS.items():
        paths[n] = Path(ROOT) / sub
    names = list(paths)
    files, sha, cfg, keys, scale = {}, {}, {}, {}, {}
    for n, p in paths.items():
        h = hashlib.sha256()
        with open(p / "adapter_model.safetensors", "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        sha[n] = h.hexdigest()
        files[n] = safe_open(str(p / "adapter_model.safetensors"), "pt")
        c = json.loads((p / "adapter_config.json").read_text())
        cfg[n] = {k: c.get(k) for k in ("r", "lora_alpha", "lora_dropout", "use_rslora", "use_dora", "bias", "init_lora_weights")}
        cfg[n]["target_modules"] = sorted(c.get("target_modules") or [])
        cfg[n]["patterns"] = bool(c.get("rank_pattern")) or bool(c.get("alpha_pattern"))
        scale[n] = c["lora_alpha"] / (c["r"] ** 0.5 if c.get("use_rslora") else c["r"])
        km = {}
        for k in files[n].keys():
            parts = k.split(".")
            if "lora_A" in parts and "layers" in parts:
                i, j = parts.index("layers"), parts.index("lora_A")
                km[(int(parts[i + 1]), ".".join(parts[i + 2:j]))] = k
        keys[n] = km
    common = sorted(set.intersection(*(set(k) for k in keys.values())))
    out = []
    for layer, mod in common:
        a = [files[n].get_tensor(keys[n][(layer, mod)]).double() for n in names]
        b = [files[n].get_tensor(keys[n][(layer, mod)].replace("lora_A", "lora_B")).double() for n in names]
        rk = [x.shape[0] for x in a]
        aa = torch.cat(a, 0) @ torch.cat(a, 0).T  # the rows of every adapter's A against every other's
        bb = torch.cat(b, 1).T @ torch.cat(b, 1)
        cut = [sum(rk[:i]) for i in range(len(rk) + 1)]
        gram = [[float((bb[cut[x]:cut[x + 1], cut[y]:cut[y + 1]] * aa[cut[x]:cut[x + 1], cut[y]:cut[y + 1]]).sum()) for y in range(len(names))]
                for x in range(len(names))]
        out.append({"layer": layer, "module": mod.split(".")[-1], "gram": gram})
    return {"names": names, "sha256": sha, "cfg": cfg, "scale": scale, "matrices": {n: len(k) for n, k in keys.items()}, "common": len(common),
            "modules": out}


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


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=GPU2_TIMEOUT)
def docloss2(job: dict) -> dict:
    """docloss for the states the job names (job["states"]: name and "runs:<folder>" or "hf:<repository>"), without
    the unchanged model, under a shorter limit."""
    return _docloss(job)


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=GPU_TIMEOUT)
def docloss(job: dict) -> dict:
    return _docloss(job)


def _docloss(job: dict) -> dict:
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

    states = job.get("states") or [[n, "runs:" + ADAPTERS[n]] for n in LOSS_STATES]
    stop_at = job.get("stop_at_s", STOP_AT_S)
    res, seconds = {}, {}
    if job.get("base", True):
        res["base"] = {k: losses(model, r) for k, r in rows.items()}
        seconds["base"] = round(time.time() - t0 - load_s, 1)
    pm, prev = None, None
    skipped = []
    for name, where in states:
        t1 = time.time()
        if t1 - t0 > stop_at:
            skipped.append(name)
            continue
        kind, _, rest = where.partition(":")
        if kind == "runs":
            path = str(Path(ROOT) / rest)
        else:
            import glob

            hits = sorted(glob.glob(f"{CACHE}/hf/hub/models--{rest.replace('/', '--')}/snapshots/*"))
            if len(hits) != 1:
                raise RuntimeError(f"{rest}: {len(hits)} snapshots on the volume, expected one")
            path = hits[0]
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
                                  "load_s": round(load_s, 1), "seconds": seconds, "skipped_for_time": skipped, "whole_s": round(time.time() - t0, 1),
                                  "versions": {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}}))


def step_dir() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / DATE / f"attractor-{TAG}"


def usd(seconds: float) -> float:
    from anatta import spend

    return seconds * spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEM_MB / 1024)["total"] / 3600


@app.local_entrypoint()
def main(stage: str):
    import msm_core as C

    from anatta import spend

    out = step_dir()
    dest = out / "diag" / f"{stage}.json"
    if dest.exists():
        raise SystemExit(f"diag/{stage}.json is on disk; nothing to do")
    g = spend.SpendGuard(None, sweep_cap=CAP, sweep_log=out / "train" / "spend6.jsonl", tool="msm2", label=f"{DATE}/{TAG}")
    logs = [out / "gpu" / "spend.jsonl", out / "spend.jsonl", *sorted((out / "train").glob("spend*.jsonl"))]
    job = json.loads((out / "X00.train.json").read_text(encoding="utf-8"))
    bound = usd(GPU_BOUND_S) if stage == "docloss" else usd(GPU2_BOUND_S) if stage == "docloss2" else CPU_FLAT
    job2 = job | {"base": False, "stop_at_s": STOP2_AT_S, "states": [[n, w if w.startswith("runs:") else "hf:" + C.ARMS[w[3:]]] for n, w in LOSS2]}
    if sum(spend.sweep_spent(p) for p in logs) + bound > STEP_CAP:
        raise SystemExit("the step's limit would be passed")
    if not g.reserve(bound):
        raise SystemExit(g.stopped)
    t0 = time.time()
    try:
        r = {"weights": lambda: weights.remote(), "purity": lambda: purity.remote(job), "docloss": lambda: docloss.remote(job),
             "weights2": lambda: weights2.remote(dict(C.ARMS)), "docloss2": lambda: docloss2.remote(job2)}[stage]()
    except BaseException:
        g.settle(bound, bound, {"kind": f"diag-{stage}", "failed": True})  # a failed call is booked at its whole bound
        raise
    cost = usd(time.time() - t0) if stage in ("docloss", "docloss2") else CPU_FLAT  # the local wall clock: the queue and the start are in it
    g.settle(bound, min(cost, bound), {"kind": f"diag-{stage}", "seconds": round(time.time() - t0, 1)})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(r), encoding="utf-8")
    print(f"diag/{stage}.json written; booked ${min(cost, bound):.2f}; this log so far ${g.sweep_spent:.2f} of ${CAP:.2f}")
