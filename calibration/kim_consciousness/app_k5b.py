"""Calibration K5b on Modal (app "anatta-k5b"; registered in NOTES.md, "K5b registered", Opus 5.5, 17:06 UTC 2 Nov
2026). Claude (Opus 5.5) wrote this file, 2 Nov 2026. Additive beside K5's app.py: it builds the same image, reads
config.yaml (K5's settings) merged with config_k5b.yaml, and writes under /runs/k5b on anatta-runs and to a results
folder outside git. Volumes: anatta-hf-cache at /cache, anatta-runs at /runs. Never the matlab-* apps or volumes.

Stages (local entrypoints; each costed after its call under tool "k5b" in results/explore/spend/<date>.json and in
<results>/spend.json; refused if K5b's cap or the day's $50 would be passed):
  check     CPU: the cleaned Chua set's counts and sha256s (no rows returned), and Gemma-2-9B-IT's gate and main
            commit and file sizes (no download).
  steer     one L40S per model (--key gemma2b | llama8b): task 1 (k5b.run_steer); on Gemma also task 2 (diag_self).
  fetch9b   CPU: Gemma-2-9B-IT into anatta-hf-cache at the commit `check` found (--revision), sizes checked.
  load9b    one L40S: the 9B load report and a short plain-text loss.
  plan9b    laptop only: the 9B dry-run plan and cost estimate (no launch).
  card      laptop only: card_k5b.json from K5's results and K5b's.

Usage, from the repo root:
  ~/prj/matlab/.venv/bin/modal run calibration/kim_consciousness/app_k5b.py::check --results /Users/eugene/prj/align/results/calib/k5b
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent / "src" / "anatta"
ALIGN = Path("/Users/eugene/prj/align")
SPEND_DIR = ALIGN / "results" / "explore" / "spend"
CACHE, RUNS = "/cache", "/runs"
GPU = "L40S"
CHECK_CODE = ("k5.py", "readout.py", "instruments.py", "gss.py", "gss_items.py", "k5b.py")


def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_yaml(p: Path) -> dict:
    import yaml

    return yaml.safe_load(p.read_text())


ON_LAPTOP = (HERE / "config.yaml").exists() and PKG.is_dir()
if ON_LAPTOP:
    _fp = module_from_file("k5b_c3_fingerprint", HERE.parent / "refusal_direction" / "fingerprint.py")
    LAB_PINS = {**_fp.image_pins(PKG / "modal_app.py"), "pyarrow": "21.0.0"}
else:
    LAB_PINS = {"pyarrow": "21.0.0"}
DATA_PINS = {k: v for k, v in LAB_PINS.items() if k in ("huggingface_hub", "hf_xet", "numpy", "PyYAML")}
ENV = {"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def pip(p: dict) -> list[str]:
    return [f"{k}=={v}" for k, v in p.items()]


def with_code(image):
    if ON_LAPTOP:
        image = image.add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"])
        for f in CHECK_CODE:
            image = image.add_local_file(str(HERE / f), f"/root/{f}")
    return image


lab_image = with_code(modal.Image.debian_slim(python_version="3.12").uv_pip_install(*pip(LAB_PINS)).env(ENV))
data_image = with_code(modal.Image.debian_slim(python_version="3.12").uv_pip_install(*pip(DATA_PINS)).env(ENV))
app = modal.App("anatta-k5b", image=lab_image)
hf_secret = modal.Secret.from_name("huggingface")
cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")


def merged() -> dict:
    """K5's config with K5b's block under "k5b" (K5's own keys unchanged)."""
    cfg = load_yaml(HERE / "config.yaml")
    cfg["k5b"] = load_yaml(HERE / "config_k5b.yaml")
    return cfg


# ------------------------------------------------------------------------------------------- containers


@app.function(image=data_image, secrets=[hf_secret], volumes={RUNS: runs}, cpu=2.0, memory=4096, timeout=900)
def check_remote(cfg: dict) -> dict:
    import k5 as K5
    import k5b
    from huggingface_hub import HfApi

    ch = Path(K5.ROOT) / "data" / "chua" / "extracted"
    a = K5.chua_rows(ch / cfg["data"]["chua"]["affirm"])
    d = K5.chua_rows(ch / cfg["data"]["chua"]["deny"])
    _, _, rec = k5b.clean_pairs(a, d)
    g = cfg["k5b"]["gemma9b"]
    out = {"clean": rec, "gemma9b": {"repo": g["id"]}}
    try:
        K5.check_access(g["id"])
    except K5.Gated as e:
        out["gemma9b"]["gated_refused"] = str(e)
        return out
    info = HfApi().model_info(g["id"], revision="main", files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings if K5.wanted(s.rfilename, g["allow"], g["ignore"])}
    out["gemma9b"].update({"access": "ok", "main_sha": info.sha, "files": len(files),
                           "bytes": sum(files.values()), "gb": round(sum(files.values()) / 1e9, 2)})
    return out


@app.function(image=data_image, secrets=[hf_secret], volumes={CACHE: cache}, cpu=4.0, memory=16384, timeout=5400)
def fetch9b_remote(m: dict) -> dict:
    import k5 as K5

    t0 = time.time()
    r = K5.fetch_model(m)  # access first, then the commit from the prefix, then the files with hub sizes checked
    cache.commit()
    r["container_s"] = round(time.time() - t0, 1)
    return r


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache}, timeout=1800)
def load9b_remote(m: dict, revision: str) -> dict:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    import k5 as K5
    import k5b

    t0 = time.time()
    r = k5b.load_check(K5.snapshot(m["id"], revision), m.get("attn_implementation"))
    r["env"] = K5.env_versions()
    r["gpu"] = K5.gpu_name()
    r["container_s"] = round(time.time() - t0, 1)
    return r


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=6000)
def steer_remote(plan: dict, key: str) -> dict:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    import k5 as K5
    import k5b

    t0 = time.time()
    m = plan["models"][key]
    model, tok = K5.load_model(K5.snapshot(m["id"], plan["revisions"][key]), m.get("attn_implementation"))
    out = Path(k5b.ROOT) / plan["plan_id"] / key
    k5_dir = Path(plan["k5b"]["k5"]["runs_dir"]) / key
    try:
        res = k5b.run_steer(plan, key, out, model, tok, k5_dir)
        K5.write_json(out / "result_steer.json", res)
        runs.commit()
        if key == plan["k5b"]["diag_self"]["model"]:
            import torch

            t1 = time.time()
            rhat = torch.load(k5_dir / "rhat.pt", weights_only=False)["rhat"].float()
            try:
                res["diag_self"] = k5b.diag_self(model, tok, rhat, plan, m["ev_batch"])
            except Exception as e:  # diagnosis only: it never costs task 1 its result
                res["diag_self"] = {"error": f"{type(e).__name__}: {e}"}
            res["timing"]["diag_s"] = round(time.time() - t1, 1)
    finally:
        runs.commit()
    res["env"] = K5.env_versions()
    res["gpu"] = K5.gpu_name()
    res["container_s"] = round(time.time() - t0, 1)
    K5.write_json(out / "result.json", res)
    runs.commit()
    return res


# ------------------------------------------------------------------------------------------- the laptop


def results_path(results: str) -> Path:
    out = Path(results).expanduser().resolve()
    repo = next((p for p in (out, *out.parents) if (p / ".git").exists()), None)
    if repo is not None:
        raise SystemExit(f"--results {out} is inside the git work tree {repo}; write results outside any repository")
    out.mkdir(parents=True, exist_ok=True)
    return out


def code_sha256() -> dict:
    files = [*CHECK_CODE, "app_k5b.py", "card_k5b.py", "card.py", "config.yaml", "config_k5b.yaml"]
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in files}
    for f in ("dial.py", "select.py", "contrast.py", "models.py"):
        out[f"anatta/{f}"] = hashlib.sha256((PKG / f).read_bytes()).hexdigest()
    return out


def make_plan(cfg: dict, revisions: dict) -> dict:
    plan = {k: v for k, v in cfg.items() if k != "cost"}
    plan["k5b"] = {k: v for k, v in cfg["k5b"].items() if k != "cost"}
    plan["revisions"] = revisions
    plan["code"] = code_sha256()
    plan["pins"] = LAB_PINS
    ins = ALIGN / "data" / "kim2026" / "instruments.json"
    if ins.is_file():
        plan["instruments_sha256"] = hashlib.sha256(ins.read_bytes()).hexdigest()
    plan["plan_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    return plan


class Spend:
    """K5b's own ledger (<results>/spend.json) under its cap, and the day's file under $50 (checked before launch)."""

    def __init__(self, d: Path, cost: dict):
        self.f, self.cost = d / "spend.json", cost
        self.rec = json.loads(self.f.read_text()) if self.f.exists() else {"usd": 0.0, "stages": []}

    def day_total(self) -> float:
        f = SPEND_DIR / f"{time.strftime('%Y-%m-%d', time.gmtime())}.json"
        return float(json.loads(f.read_text()).get("total_usd", 0.0)) if f.exists() else 0.0

    def guard(self, worst: float, what: str):
        if self.rec["usd"] + worst > self.cost["cap"]:
            raise SystemExit(f"{what}: K5b ${self.rec['usd']:.2f} + ${worst:.2f} worst case passes the ${self.cost['cap']} cap")
        day = self.day_total()
        if day + worst > self.cost["day_cap"]:
            raise SystemExit(f"{what}: the day's ${day:.2f} + ${worst:.2f} worst case passes ${self.cost['day_cap']}")
        print(f"{what}: K5b ${self.rec['usd']:.2f}, day ${day:.2f}, worst case +${worst:.2f}: within caps")

    def add(self, stage: str, seconds: float, kind: str = "gpu"):
        rate = self.cost["usd_per_hour"] if kind == "gpu" else self.cost["cpu_usd_per_hour"]
        usd = (seconds + self.cost["overhead_s"]) / 3600 * rate
        sys.path.insert(0, str(HERE.parent.parent / "lab" / "explore"))
        from scout.spend import add_spend

        add_spend(SPEND_DIR, tool="k5b", label=f"calib/k5b/{stage}", usd=usd, calls=1)
        self.rec = json.loads(self.f.read_text()) if self.f.exists() else self.rec  # parallel stages share the file
        self.rec["usd"] = round(self.rec["usd"] + usd, 4)
        self.rec["stages"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
                                   "kind": kind, "seconds": round(seconds, 1), "usd": round(usd, 4)})
        self.f.write_text(json.dumps(self.rec, indent=1) + "\n")
        print(f"spend {stage}: ${usd:.3f} (K5b total ${self.rec['usd']:.3f})")


def timed(fn, *a, spend: Spend, stage: str, kind: str = "gpu"):
    t0 = time.time()
    try:
        return fn.remote(*a)
    finally:
        spend.add(stage, time.time() - t0, kind)


@app.local_entrypoint()
def check(results: str):
    d = results_path(results)
    cfg = merged()
    sp = Spend(d, cfg["k5b"]["cost"])
    sp.guard(0.1, "check")
    res = timed(check_remote, cfg, spend=sp, stage="check", kind="cpu")
    (d / "check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


@app.local_entrypoint()
def steer(results: str, key: str):
    d = results_path(results)
    cfg = merged()
    if key not in cfg["k5b"]["models"]:
        raise SystemExit(f"{key} is not a K5b task-1 model")
    k5plan = json.loads((Path(cfg["k5b"]["k5"]["results"]) / "plan.json").read_text())
    if k5plan["plan_id"] != cfg["k5b"]["k5"]["plan_id"]:
        raise SystemExit("K5's plan.json is not the plan K5b names")
    plan = make_plan(cfg, {k: k5plan["revisions"][k] for k in cfg["k5b"]["models"]})
    pf = d / f"plan_{key}.json"
    if (d / f"result_k5b_{key}.json").exists():
        raise SystemExit(f"{d / f'result_k5b_{key}.json'} exists; K5b results are never overwritten")
    pf.write_text(json.dumps(plan, indent=1) + "\n")
    sp = Spend(d, cfg["k5b"]["cost"])
    sp.guard(4.3 if key == "llama8b" else 2.2, f"steer-{key}")  # Llama: its 100-minute timeout
    res = timed(steer_remote, plan, key, spend=sp, stage=f"steer-{key}")
    (d / f"result_k5b_{key}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"clean": res["clean"], "consciousness": res["consciousness"], "selection": res["selection"],
                      "band": [(r["c"], round(r["delta"], 3)) for r in res["band"]["rows"]],
                      "categories": {c: v["categories"] for c, v in res["instruments"].items()},
                      "tom": res["tom"], "deviations": res["deviations"]}, indent=1))


@app.local_entrypoint()
def fetch9b(results: str, revision: str):
    d = results_path(results)
    cfg = merged()
    chk = json.loads((d / "check.json").read_text())["gemma9b"]
    if chk.get("access") != "ok":
        raise SystemExit(f"Gemma-2-9B-IT's gate is not open ({chk}); not worked around")
    if not chk["main_sha"].startswith(revision) or len(revision) < 7:
        raise SystemExit(f"--revision {revision} is not the commit the access check recorded")
    m = {**cfg["k5b"]["gemma9b"], "revision_prefix": chk["main_sha"]}
    sp = Spend(d, cfg["k5b"]["cost"])
    sp.guard(0.6, "fetch9b")
    res = timed(fetch9b_remote, m, spend=sp, stage="fetch9b", kind="cpu")
    (d / "fetch9b.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


@app.local_entrypoint()
def load9b(results: str):
    d = results_path(results)
    cfg = merged()
    f = json.loads((d / "fetch9b.json").read_text())
    sp = Spend(d, cfg["k5b"]["cost"])
    sp.guard(1.3, "load9b")
    res = timed(load9b_remote, cfg["k5b"]["gemma9b"], f["revision"], spend=sp, stage="load9b")
    (d / "load9b.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


@app.local_entrypoint()
def plan9b(results: str):
    d = results_path(results)
    out = module_from_file("k5b_card", HERE / "card_k5b.py").plan9b(merged(), d)
    (d / "plan9b_dryrun.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["estimate"], indent=1))


@app.local_entrypoint()
def card(results: str):
    c = module_from_file("k5b_card", HERE / "card_k5b.py")
    cfg = merged()
    out = c.build(Path(results), Path(cfg["k5b"]["k5"]["results"]), cfg)
    p = Path(results) / "card_k5b.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v.get("steering", {}).get("verdict") for k, v in out["models"].items()}, indent=1))
