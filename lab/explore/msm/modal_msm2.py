"""The replication's second step, in the cloud: the study's data fetched to the lab's volume, and the arms trained.
Claude (Fable 5.2, an AI model) wrote this file, 22 Nov 2026. The registration is msm2.yaml; the driver run_msm2.py.

  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage fetch
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name pilot     (a job file run_msm2.py wrote)
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name Z         (and F, S, O)

`fetch` (CPU, no key): the study's three datasets from Hugging Face, each as the hub's own parquet conversion, into
/runs/msm2/data on the lab's runs volume; it returns names, sizes, hashes and, for the documents, the row numbers of
each part. No text comes back to the laptop: the documents and conversations are read only inside containers.
`train` (one H200): msm2_torch on the base model already on the lab's volume; the adapter is saved to /runs/msm2/<arm>
after the documents and again after the practice, in PEFT's form, which the lab's vLLM engine serves as it is.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # lab/explore/msm on the laptop; /root in a container
if modal.is_local() and str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

APP_NAME = "anatta-explore-msm2"
DATE, TAG = "2026-11-22", "msm-s2"
BASE = "/cache/hf/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
CACHE, RUNS, ROOT = "/cache", "/runs", "/runs/msm2"
GPU, CPU, MEM_MB = "H200", 8.0, 98304
START_S = 420.0  # a container's start and the model's load, measured at about 230 s in step 1; cautious
TRAIN_CAP = 150.0  # the training's share of the step's $200, in dollars
CPU_FLAT = 0.05
FILES = {  # what is fetched: (dataset, the file in its parquet conversion)
    "docs": ("chloeli/msm-qwen-philosophy-spec", "default/train/0000.parquet"),
    "aft": ("chloeli/aft-cot-qwen2.5-philosophy-spec", "default/train/0000.parquet"),
    "mix": ("chloeli/sft-it-mix", "default/train_clean/0000.parquet"),
}
PACKAGES = ("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0", "numpy==2.5.4",
            "peft==0.21.2", "accelerate==1.15.0", "pyarrow==21.0.0", "jinja2==3.1.6")  # the lab's training pins, and pyarrow

cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")
app = modal.App(APP_NAME)
fetch_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub==0.36.0", "pyarrow==21.0.0")
train_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PACKAGES)
               .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
               .add_local_file(str(HERE / "msm2_torch.py"), "/root/msm2_torch.py"))


@app.function(image=fetch_image, cpu=2.0, memory=8192, timeout=1800, volumes={RUNS: runs})
def fetch() -> dict:
    import hashlib
    import shutil

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out, dest = {}, Path(ROOT) / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for name, (repo, fname) in FILES.items():
        listed = HfApi().list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet", token=False)
        want = [f for f in listed if f.startswith(fname.rsplit("/", 1)[0] + "/") and f.endswith(".parquet")]
        if want != [fname]:
            raise RuntimeError(f"{repo}: expected the one file {fname}, the conversion lists {want}")
        info = HfApi().dataset_info(repo, revision="refs/convert/parquet", token=False)
        p = hf_hub_download(repo, fname, repo_type="dataset", revision="refs/convert/parquet", token=False)
        shutil.copyfile(p, dest / f"{name}.parquet")
        table = pq.read_table(dest / f"{name}.parquet")
        rec = {"repo": repo, "file": fname, "revision": info.sha, "bytes": (dest / f"{name}.parquet").stat().st_size,
               "sha256": hashlib.sha256((dest / f"{name}.parquet").read_bytes()).hexdigest(), "rows": table.num_rows,
               "columns": table.column_names}
        if name == "docs":
            rows: dict = {}
            for i, d in enumerate(table.column("domain").to_pylist()):
                rows.setdefault(d, []).append(i)
            rec["rows_by_domain"] = rows
            rec["chars"] = sum(len(t) for t in table.column("text").to_pylist())
        out[name] = rec
    runs.commit()
    return out


@app.function(image=train_image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=10 * 3600)
def train(job: dict) -> dict:
    """One job: {"name", "seed", "lora", "optim", "max_len", "batch_tokens", "docs": [row numbers] or None, "mix":
    [row numbers], "max_steps": None or a number (the pilot: so many steps of each phase, nothing saved)}."""
    import msm2_torch as T
    import pyarrow.parquet as pq
    import torch

    t0 = time.time()
    data, pilot = Path(ROOT) / "data", job.get("max_steps") is not None
    tok, model = T.load(BASE)
    load_s = time.time() - t0
    model = T.add_lora(model, job["seed"], **job["lora"])
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    kw = dict(pad=pad, seed=job["seed"], batch_tokens=job["batch_tokens"], max_steps=job.get("max_steps"), **job["optim"])
    out = {"name": job["name"], "load_s": round(load_s, 1), "phases": {},
           "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "lora_dtype": str(next(p for p in model.parameters() if p.requires_grad).dtype)}
    save = Path(ROOT) / job["name"]
    if job.get("docs") is not None:
        texts = pq.read_table(data / "docs.parquet", columns=["text"]).column("text").to_pylist()
        rows = [T.encode_doc(tok, texts[i], job["max_len"]) for i in job["docs"]]
        out["phases"]["docs"] = T.run_phase(model, rows, **kw) | {"rows": len(rows), "cut_at_max_len": sum(len(r[0]) >= job["max_len"] for r in rows)}
        del texts, rows
        if not pilot:
            model.save_pretrained(str(save / "docs"))
            runs.commit()
    aft = pq.read_table(data / "aft.parquet", columns=["messages"]).column("messages").to_pylist()
    mix = pq.read_table(data / "mix.parquet", columns=["messages"]).column("messages").to_pylist()
    chats = [[{"role": m["role"], "content": m["content"]} for m in c] for c in aft] + [[{"role": m["role"], "content": m["content"]} for m in mix[i]] for i in job["mix"]]
    enc = [T.encode_chat(tok, c, job["max_len"]) for c in chats]
    rows = [(x, y) for x, y, _ in enc if any(v != -100 for v in y)]
    out["phases"]["chat"] = T.run_phase(model, rows, **kw) | {
        "rows": len(rows), "aft": len(aft), "mix": len(job["mix"]), "no_assistant_turn": len(enc) - len(rows),
        "labels_not_exact": sum(not e for _, _, e in enc), "cut_at_max_len": sum(len(x) >= job["max_len"] for x, _, _ in enc),
        "label_tokens": sum(sum(v != -100 for v in y) for _, y in rows)}
    if not pilot:
        model.save_pretrained(str(save / "final"))
        runs.commit()
        out["saved"] = [str(save / "docs")] * (job.get("docs") is not None) + [str(save / "final")]
    out["seconds"] = round(time.time() - t0, 1)
    out["versions"] = {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}
    return json.loads(json.dumps(out, default=str))  # plain JSON types only (step 1's lost floor)


# ---------------------------------------------------------------------------------------------- on the laptop


def step_dir() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / DATE / f"attractor-{TAG}"  # the lab's engine names it so


def rate() -> float:
    from anatta import spend

    return spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEM_MB / 1024)["total"]


def usd(seconds: float) -> float:
    return seconds * rate() / 3600


def guard():
    from anatta import spend

    return spend.SpendGuard(None, sweep_cap=TRAIN_CAP, sweep_log=step_dir() / "train" / "spend.jsonl", tool="msm2", label=f"{DATE}/{TAG}")


def code_state() -> dict:
    import subprocess

    run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", str(HERE)))}


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


@app.local_entrypoint()
def main(stage: str, name: str = ""):
    out, g = step_dir(), guard()
    if stage == "fetch":
        if (out / "data.json").exists():
            raise SystemExit("data.json is on disk; nothing to do")
        if not g.reserve(CPU_FLAT):
            raise SystemExit(g.stopped)
        r = fetch.remote()
        g.settle(CPU_FLAT, CPU_FLAT, {"kind": "fetch", "flat": True})
        _write(out / "data.json", r | {"code": code_state()})
        for k, v in r.items():
            print(f"{k}: {v['repo']} {v['file']} at {v['revision'][:10]}: {v['bytes'] / 1e6:.1f} MB, {v['rows']} rows, sha256 {v['sha256'][:16]}", flush=True)
        print("documents by part:", {d: len(x) for d, x in r["docs"]["rows_by_domain"].items()}, flush=True)
        return
    if stage != "train" or not name:
        raise SystemExit("--stage is fetch, or train with --name (pilot, Z, F, S or O)")
    dest = out / "train" / f"{name}.json"
    if dest.exists():
        raise SystemExit(f"train/{dest.name} is on disk; nothing to do")
    job = json.loads((out / f"{name}.train.json").read_text(encoding="utf-8"))
    if not name.startswith("pilot") and not json.loads((out / "go.json").read_text(encoding="utf-8")).get("go"):
        raise SystemExit("no go.json: the pilot's projection is read before any arm is trained")
    bound = usd(float(job["est_s"]) + START_S)
    if not g.reserve(bound):
        raise SystemExit(g.stopped)
    print(f"{name}: bound ${bound:.2f} ({job['est_s']:.0f}s and the start); booked so far ${g.sweep_spent:.2f} of ${TRAIN_CAP:.2f}", flush=True)
    t0 = time.time()
    try:
        r = train.remote({k: v for k, v in job.items() if k != "est_s"})
    except BaseException as e:
        g.settle(bound, min(bound, usd(time.time() - t0 + 60.0)), {"kind": "train", "name": name, "failed": True})
        raise SystemExit(f"the container failed: {type(e).__name__}: {str(e)[:800]}") from e
    cost = usd(r["seconds"] + 60.0)
    g.settle(bound, cost, {"kind": "train", "name": name, "seconds": r["seconds"], "rate_usd_per_hour": rate()})
    _write(dest, r | {"code": code_state(), "job": {k: (v if k not in ("docs", "mix") else (None if v is None else len(v))) for k, v in job.items()}})
    for ph, v in r["phases"].items():
        last = v["log"][-1] if v["log"] else {}
        print(f"  {ph}: {v['steps_done']} of {v['steps_in_epoch']} steps, {v['tokens_done'] / 1e6:.2f}M of {v['tokens_in_epoch'] / 1e6:.2f}M tokens, "
              f"{v['tokens_per_second']:.0f} tokens/s, last loss {last.get('loss')}, peak {v['peak_gib']} GiB, rows {v['rows']}", flush=True)
    print(f"{name}: load {r['load_s']:.0f}s, whole container {r['seconds']:.0f}s, ${cost:.2f}; trainable {r['trainable'] / 1e6:.0f}M ({r['lora_dtype']}); "
          f"booked so far ${g.sweep_spent:.2f} of ${TRAIN_CAP:.2f}", flush=True)
