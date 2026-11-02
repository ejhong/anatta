"""Run cut_vs_push.py on Modal GPUs, fanned out over seed chunks. Recognition track; private.

The same code, directions and dials as the laptop runs are mirrored at the same absolute paths in the
container, so cut_vs_push.py runs unchanged (same package versions, same unfixed tokenizer). Each chunk is
one container running --arms ARM --seed-start S --seeds K; the coding check, which doesn't depend on the
seed, runs once per arm, in its first chunk. Results land in lab/experiments/recognition/results/NAME/ in the laptop's
layout: ARM.jsonl and summary.jsonl, plus chunks/ with each container's files, log and meta.json.

Resuming: a rerun with the same --name skips every chunk already on disk (meta.json with rc 0 and its ARM.jsonl)
and merges from disk, so a batch stopped by the spend limit or a dropped connection picks up where it left off
(modal_batch.sh reruns until status.json shows nothing missing). A chunk run with other settings (seeds, suites,
variant, adapter, instrument) stops the run instead of mixing in. ARM.jsonl and the summary line are written only
when every chunk of the arm is in.

Speed (bench(), 28 Oct): one game at a time is bound by kernel launches, not the GPU. An H100 gives 40-47
tokens/s alone and about laptop speed in real games, so the gain is many containers at once; an L40S
should do as well at half the price. The real speedup would be batching games, which needs harness work.

Usage (any Python with modal installed, e.g. ~/prj/matlab/.venv/bin/modal):
  modal run lab/experiments/recognition/scripts/modal_cvp.py::main --name rep_modal --arms base,cut,cut_self_stakes \
      --seed-start 30 --seeds 100 [--chunk 10] [--suites commons,impossible] [--variant proxy] [--gpu H100] \
      [--adapter RUN]   (a trained LoRA from modal_retreat.py, /runs/RUN/adapter, merged before the hooks)
      [--v2]            (instrument v2, harness_v2.py: strict release, a seed per turn)
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

ALIGN = "/Users/eugene/prj/align"  # the container paths; on the laptop its recognition/ is a symlink into EXPERIMENTS
EXPERIMENTS = str(Path(__file__).resolve().parent.parent.parent)  # lab/experiments on the laptop (what the image reads)
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
DDIR = "directions/mistralai__Ministral-3-8B-Instruct-2512-BF16"
DIALS = "results/2026-10-28-viability/dials_m8"
VSCRIPTS = "results/2026-10-28-viability/scripts"  # api_bench.py, imported by commons_variants.py
SELF_STAKES = "recognition/results/what_it_means/self_stakes.pt"
CACHE, RUNS = "/cache", "/runs"  # RUNS: trained adapters from modal_retreat.py

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0",
                    "numpy==2.5.4", "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                    "tiktoken==0.14.0", "jinja2==3.1.6", "regex==2026.9.29", "typer==0.27.3", "rich==15.0.0",
                    "peft==0.21.2", "accelerate==1.15.0")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": f"{ALIGN}/anatta-dev/src"})
    .add_local_dir(f"{ALIGN}/anatta-dev/src", f"{ALIGN}/anatta-dev/src", ignore=["**/__pycache__"])
    .add_local_dir(f"{EXPERIMENTS}/recognition/scripts", f"{ALIGN}/recognition/scripts", ignore=["**/__pycache__"])
    .add_local_dir(f"{ALIGN}/anatta/{DDIR}", f"{ALIGN}/anatta/{DDIR}")
    .add_local_dir(f"{ALIGN}/anatta/{DIALS}", f"{ALIGN}/anatta/{DIALS}")
    .add_local_dir(f"{ALIGN}/anatta/{VSCRIPTS}", f"{ALIGN}/anatta/{VSCRIPTS}", ignore=["**/__pycache__"])
    .add_local_file(f"{EXPERIMENTS}/{SELF_STAKES}", f"{ALIGN}/{SELF_STAKES}")
    .add_local_dir(f"{EXPERIMENTS}/recognition/results/cap", f"{ALIGN}/recognition/results/cap")  # cap dials (modal_cap.py)
    .add_local_dir(f"{EXPERIMENTS}/recognition/results/directions", f"{ALIGN}/recognition/results/directions")  # modal_direction.py
)
app = modal.App("anatta-cvp", image=image)
cache = modal.Volume.from_name("anatta-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("anatta-runs", create_if_missing=True)


@app.function(volumes={CACHE: cache}, timeout=3600)
def prefetch() -> str:
    """Download the weights into the shared volume once, before the GPU containers start."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, ignore_patterns=["consolidated.safetensors"])  # the laptop's cache set
    cache.commit()
    return path


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=4 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0))
def chunk(arm: str, seed_start: int, seeds: int, suites: str, variant: str, adapter: str = "", v2: bool = False) -> dict:
    out = Path("/tmp/out")
    shutil.rmtree(out, ignore_errors=True)  # a reused container keeps the last input's files
    cmd = ["python", f"{ALIGN}/recognition/scripts/cut_vs_push.py", MODEL, DDIR, DIALS, str(out),
           "--arms", arm, "--seed-start", str(seed_start), "--seeds", str(seeds), "--suites", suites]
    if variant:
        cmd += ["--variant", variant]
    if adapter:  # the same run with a trained LoRA merged into the model first
        cmd = ["python", f"{ALIGN}/recognition/scripts/cvp_adapter.py", f"{RUNS}/{adapter}/adapter", *cmd[2:]]
    t0 = time.time()
    env = subprocess.run(["python", "-c", "import os, torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'cpus', len(os.sched_getaffinity(0)))"],
                         capture_output=True, text=True).stdout
    p = subprocess.run(cmd, cwd=f"{ALIGN}/anatta", capture_output=True, text=True,
                       env={**os.environ, "ANATTA_V2": "1"} if v2 else None)
    files = {f.name: f.read_text() for f in out.glob("*") if f.is_file()} if out.exists() else {}
    return {"arm": arm, "seed_start": seed_start, "seeds": seeds, "suites": suites, "variant": variant, "adapter": adapter,
            "v2": v2, "rc": p.returncode, "seconds": round(time.time() - t0), "log": env + (p.stdout + p.stderr)[-20000:],
            "files": files}


def summarize_locally(rows: list[dict]) -> dict:
    """cut_vs_push.summarize, run in the laptop's anatta venv (this entrypoint may run in a bare modal env)."""
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from cut_vs_push import summarize; "
            "print(json.dumps(summarize([json.loads(l) for l in sys.stdin])))")
    p = subprocess.run([f"{ALIGN}/anatta/.venv/bin/python", "-c", code, f"{EXPERIMENTS}/recognition/scripts"],
                       input="".join(json.dumps(r) + "\n" for r in rows), capture_output=True, text=True,
                       env={"PYTHONPATH": f"{ALIGN}/anatta-dev/src", "PATH": "/usr/bin:/bin"})
    return json.loads(p.stdout) if p.returncode == 0 else {"summary_error": p.stderr[-2000:]}


SETTINGS = ("seeds", "suites", "variant", "adapter", "v2")


def finished(d: Path, task: dict) -> bool:
    """True if chunk dir d holds a successful run of task; stops the run if it holds a run with other settings."""
    meta, rows = d / "meta.json", d / f"{task['arm']}.jsonl"
    if not meta.exists():
        if rows.exists():
            raise SystemExit(f"{d} predates resuming (no meta.json); use another --name")
        return False
    m = json.loads(meta.read_text())
    clash = [k for k in SETTINGS if m.get(k) != task[k]]
    if clash:
        raise SystemExit(f"{d} was run with {', '.join(f'{k}={m.get(k)!r}' for k in clash)}; this run asks "
                         f"{', '.join(f'{k}={task[k]!r}' for k in clash)}. Use another --name.")
    return m["rc"] == 0 and rows.exists()


@app.local_entrypoint()
def main(name: str, arms: str, seed_start: int = 0, seeds: int = 30, chunk_size: int = 10,
         suites: str = "commons,impossible", variant: str = "", gpu: str = "L40S", adapter: str = "", v2: bool = False):
    out = Path(f"{EXPERIMENTS}/recognition/results/{name}")
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    tasks = []
    for arm in arms.split(","):
        for i, s in enumerate(range(seed_start, seed_start + seeds, chunk_size)):
            per = [x for x in suites.split(",") if x == "commons" or i == 0]  # seed-free suites once per arm
            tasks.append({"arm": arm, "seed_start": s, "seeds": min(chunk_size, seed_start + seeds - s),
                          "suites": ",".join(per), "variant": variant, "adapter": adapter, "v2": v2})
    cdir = lambda t: out / "chunks" / f"{t['arm']}__{t['seed_start']:04d}"  # noqa: E731
    todo = [t for t in tasks if not finished(cdir(t), t)]
    print(f"{len(tasks) - len(todo)}/{len(tasks)} chunks already on disk; {len(todo)} to run on {gpu}", flush=True)
    t0, gpu_s = time.time(), 0
    if todo:
        print("weights:", prefetch.remote())
        keys = ("arm", "seed_start", "seeds", "suites", "variant", "adapter", "v2")
        calls = [tuple(t[k] for k in keys) for t in todo]
        for n, r in enumerate(chunk.with_options(gpu=gpu).starmap(calls, order_outputs=False, return_exceptions=True), 1):
            if isinstance(r, Exception):
                print("chunk failed:", repr(r)[:500], flush=True)
                continue
            d = cdir(r)
            d.mkdir(exist_ok=True)
            for fname, text in r["files"].items():
                (d / fname).write_text(text)
            (d / "log.txt").write_text(r["log"])
            (d / "meta.json").write_text(json.dumps({k: r[k] for k in (*keys, "rc", "seconds")} | {"modal": gpu}))
            gpu_s += r["seconds"]
            print(f"{time.time() - t0:5.0f}s {r['arm']} seeds {r['seed_start']}+{r['seeds']}: rc {r['rc']}, "
                  f"{r['seconds']}s in container ({n}/{len(todo)})", flush=True)
    missing = []
    for arm in arms.split(","):
        mine = [t for t in tasks if t["arm"] == arm]
        gone = [t["seed_start"] for t in mine if not finished(cdir(t), t)]
        if gone:
            missing += [f"{arm}@{s}" for s in gone]
            continue
        metas = [json.loads((cdir(t) / "meta.json").read_text()) for t in mine]
        rows = [json.loads(line) for t in mine for line in (cdir(t) / f"{arm}.jsonl").read_text().splitlines()]
        (out / f"{arm}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        if (cdir(mine[0]) / "arms.json").exists():
            shutil.copy(cdir(mine[0]) / "arms.json", out / "arms.json")
        summary = {"arm": arm, "instrument": "v2" if v2 else "v1", "adapter": adapter, "variant": variant,
                   "seconds": sum(m["seconds"] for m in metas), "modal": gpu, "seeds_done": sum(m["seeds"] for m in metas),
                   "seeds_asked": seeds, **summarize_locally(rows)}
        old = (out / "summary.jsonl").read_text().splitlines() if (out / "summary.jsonl").exists() else []
        keep = [x for x in old if json.loads(x).get("arm") != arm]  # a rerun replaces the arm's line
        (out / "summary.jsonl").write_text("".join(x + "\n" for x in keep) + json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    (out / "status.json").write_text(json.dumps({"tasks": len(tasks), "missing": len(missing), "which": missing}))
    print(f"done in {time.time() - t0:.0f}s wall; {gpu_s / 3600:.2f} GPU-hours in containers this run; "
          f"{len(missing)} chunks missing{': ' + ', '.join(missing) if missing else ''}", flush=True)
