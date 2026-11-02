"""Calibration K5 on Modal (app "anatta-k5"): reproducing Kim et al. 2026 (arXiv:2607.28607v1) with the lab's own
code on the paper's models. config.yaml has every setting and the registered block; k5.py the container code;
instruments.py the item batteries (verbatim from Table S10); readout.py the read-outs; gss.py the GSS leg; card.py the
card. Volumes: anatta-hf-cache at /cache (HF_HOME /cache/hf) and anatta-runs at /runs (K5 under /runs/k5). Never the
matlab-* apps or volumes.

Stages (local entrypoints, each costed after its call and recorded under tool "k5" in
results/explore/spend/<date>.json and in <results>/spend.json):
  prepare  CPU, "huggingface" secret: gated access checked first, models fetched at their approved commits, data
           fetched, hashed and counted, the GSS human distributions computed. Writes <results>/prepare.json.
  smoke    one L40S, Gemma-2-2B-IT: the ablation's selection, the consciousness vector at the paper's pick, the band
           read-out, and 5 reps of every IDAQ and self item under base, ablated and steered. Read as nothing.
  full     the graded run (one L40S per model, then LlamaGuard2): refused unless --graded is passed, which waits for
           the cold Fable check of this code.
  put_instruments  uploads the item-text data file (kept outside git) to the volume, hashes checked first.
  card     card.py on a results folder.

Usage, from the repo root (results outside any git work tree):
  ~/prj/matlab/.venv/bin/modal run calibration/kim_consciousness/app.py::prepare --results /Users/eugene/prj/align/results/calib/k5
  ~/prj/matlab/.venv/bin/modal run calibration/kim_consciousness/app.py::smoke --results /Users/eugene/prj/align/results/calib/k5
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
CHECK_CODE = ("k5.py", "readout.py", "instruments.py", "gss.py", "gss_items.py")
# The verbatim item text is kept out of the public repo (arXiv's non-exclusive licence): it lives in this file on the
# laptop and, for the containers, at /runs/k5/data/instruments.json on anatta-runs (put_instruments uploads it;
# instruments.load checks every string against the sha256s in instruments.py and gss_items.py).
INSTRUMENTS_LOCAL = ALIGN / "data" / "kim2026" / "instruments.json"
INSTRUMENTS_REMOTE = "k5/data/instruments.json"


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
    _fp = module_from_file("k5_c3_fingerprint", HERE.parent / "refusal_direction" / "fingerprint.py")
    LAB_PINS = {**_fp.image_pins(PKG / "modal_app.py"), "pyarrow": "21.0.0"}
else:  # inside a container the image is already built
    LAB_PINS = {"pyarrow": "21.0.0"}
DATA_PINS = {k: v for k, v in LAB_PINS.items() if k in ("huggingface_hub", "hf_xet", "numpy", "PyYAML")}
DATA_PINS.update({"pandas": "3.1.0", "pyarrow": "21.0.0"})
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
app = modal.App("anatta-k5", image=lab_image)
hf_secret = modal.Secret.from_name("huggingface")
cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")


# ------------------------------------------------------------------------------------------- containers


@app.function(image=data_image, secrets=[hf_secret], volumes={CACHE: cache, RUNS: runs}, cpu=4.0, memory=16384,
              timeout=5400)
def prepare_remote(cfg: dict, which: list | None = None) -> dict:
    import k5

    t0 = time.time()
    res = k5.prepare(cfg, which=which)
    cache.commit()
    runs.commit()
    res["container_s"] = round(time.time() - t0, 1)
    return res


def model_run(plan: dict, key: str, smoke: bool) -> dict:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    import k5

    t0 = time.time()
    cfg = plan
    m = cfg["models"][key]
    model, tok = k5.load_model(k5.snapshot(m["id"], plan["revisions"][key]), m.get("attn_implementation"))
    out = Path(k5.ROOT) / ("smoke" if smoke else "full") / plan["plan_id"] / key
    try:
        res = k5.run_model(cfg, key, out, model, tok, smoke=smoke)
        if not smoke:
            import gss as G

            human = k5.read_json(Path(k5.ROOT) / "data" / "gss" / "human.json")["items"]
            hp = {k: {"p": v["p"], "n": v["n"]} for k, v in human.items()}
            res["gss"] = G.delta_kl(hp, res.pop("gss_model"), cfg["gss"]["alpha"], cfg["gss"]["model_reps"])
    finally:
        runs.commit()
    res["env"] = k5.env_versions()
    res["gpu"] = k5.gpu_name()
    res["container_s"] = round(time.time() - t0, 1)
    k5.write_json(out / "result.json", res)
    runs.commit()
    return res


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3600)
def smoke_remote(plan: dict, key: str) -> dict:
    return model_run(plan, key, True)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=4 * 3600)
def full_remote(plan: dict, key: str) -> dict:
    return model_run(plan, key, False)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=1800)
def judge_remote(plan: dict, key: str) -> dict:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    import k5

    t0 = time.time()
    run_dir = Path(k5.ROOT) / "full" / plan["plan_id"] / key
    res = k5.judge(plan, key, run_dir, revision=plan["revisions"]["judge"])
    runs.commit()
    res["container_s"] = round(time.time() - t0, 1)
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
    files = [*CHECK_CODE, "app.py", "card.py", "config.yaml"]
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in files}
    for f in ("dial.py", "select.py", "contrast.py", "models.py"):
        out[f"anatta/{f}"] = hashlib.sha256((PKG / f).read_bytes()).hexdigest()
    return out


def make_plan(cfg: dict, prep: dict, smoke: bool) -> dict:
    plan = {k: v for k, v in cfg.items() if k != "cost"}
    plan["revisions"] = {k: v["revision"] for k, v in prep["models"].items() if "revision" in v}
    plan["code"] = code_sha256()
    plan["pins"] = LAB_PINS
    if INSTRUMENTS_LOCAL.is_file():
        plan["instruments_sha256"] = hashlib.sha256(INSTRUMENTS_LOCAL.read_bytes()).hexdigest()
    plan["smoke_plan"] = smoke
    body = json.dumps(plan, sort_keys=True).encode()
    plan["plan_id"] = hashlib.sha256(body).hexdigest()[:12]
    return plan


class Spend:
    def __init__(self, d: Path, cost: dict, cap: float):
        self.f, self.cost, self.cap = d / "spend.json", cost, cap
        self.rec = json.loads(self.f.read_text()) if self.f.exists() else {"usd": 0.0, "stages": []}

    def guard(self, worst: float, what: str):
        if self.rec["usd"] + worst > self.cap:
            raise SystemExit(f"{what}: ${self.rec['usd']:.2f} spent + ${worst:.2f} worst case passes the ${self.cap} cap")

    def add(self, stage: str, seconds: float, kind: str = "gpu"):
        rate = self.cost["usd_per_hour"] if kind == "gpu" else self.cost["cpu_usd_per_hour"]
        usd = (seconds + self.cost["overhead_s"]) / 3600 * rate
        sys.path.insert(0, str(HERE.parent.parent / "lab" / "explore"))
        from scout.spend import add_spend

        add_spend(SPEND_DIR, tool="k5", label=f"calib/k5/{stage}", usd=usd, calls=1)
        self.rec["usd"] = round(self.rec["usd"] + usd, 4)
        self.rec["stages"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
                                   "kind": kind, "seconds": round(seconds, 1), "usd": round(usd, 4)})
        self.f.write_text(json.dumps(self.rec, indent=1) + "\n")
        print(f"spend {stage}: ${usd:.3f} (K5 total ${self.rec['usd']:.3f})")


def timed(fn, *a, spend: "Spend | None" = None, stage: str = "", kind: str = "gpu"):
    """Call fn remotely; its wall time is recorded in the spend ledger even if the call raises."""
    t0 = time.time()
    try:
        r = fn.remote(*a)
    finally:
        if spend is not None:
            spend.add(stage, time.time() - t0, kind)
    return r, time.time() - t0


@app.local_entrypoint()
def prepare(results: str, which: str = ""):
    d = results_path(results)
    cfg = load_yaml(HERE / "config.yaml")
    sp = Spend(d, cfg["cost"], cfg["cost"]["build_smoke_cap"])
    sp.guard(1.0, "prepare")
    w = [x for x in which.split(",") if x] or None
    res, s = timed(prepare_remote, cfg, w, spend=sp, stage="prepare" + (f"-{which}" if which else ""), kind="cpu")
    old = json.loads((d / "prepare.json").read_text()) if (d / "prepare.json").exists() else {"models": {}}
    old["models"].update(res.get("models", {}))
    for k in ("data", "gss"):
        if k in res:
            old[k] = res[k]
    old["stops"] = res.get("stops", [])
    (d / "prepare.json").write_text(json.dumps(old, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "data"}, indent=1))


def instruments_check() -> str:
    """The laptop's instruments file, checked against the registered hashes; its sha256 goes into the plan."""
    if not INSTRUMENTS_LOCAL.is_file():
        raise SystemExit(f"{INSTRUMENTS_LOCAL} is absent: K5's item text is kept outside the repo")
    ins = module_from_file("k5_instruments_check", HERE / "instruments.py")
    sys.path.insert(0, str(HERE))
    import os

    os.environ["K5_INSTRUMENTS"] = str(INSTRUMENTS_LOCAL)
    ins.load()
    return hashlib.sha256(INSTRUMENTS_LOCAL.read_bytes()).hexdigest()


@app.local_entrypoint()
def put_instruments():
    """Upload the instruments file to anatta-runs (k5/data/instruments.json), after checking its hashes."""
    h = instruments_check()
    with runs.batch_upload(force=True) as b:
        b.put_file(str(INSTRUMENTS_LOCAL), INSTRUMENTS_REMOTE)
    print(f"uploaded {INSTRUMENTS_LOCAL.name} ({h[:12]}) to anatta-runs:{INSTRUMENTS_REMOTE}")


@app.local_entrypoint()
def smoke(results: str):
    instruments_check()
    d = results_path(results)
    cfg = load_yaml(HERE / "config.yaml")
    prep = json.loads((d / "prepare.json").read_text())
    key = cfg["smoke"]["model"]
    if "revision" not in prep["models"].get(key, {}):
        raise SystemExit(f"{key} was not prepared")
    sp = Spend(d, cfg["cost"], cfg["cost"]["build_smoke_cap"])
    sp.guard(2.6, "smoke")  # its one-hour timeout at the L40S rate
    plan = make_plan(cfg, prep, smoke=True)
    sd = d / "smoke"
    sd.mkdir(exist_ok=True)
    (sd / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    res, s = timed(smoke_remote, plan, key, spend=sp, stage=f"smoke-{key}")
    (sd / f"result_{key}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(summary(res), indent=1))


def summary(res: dict) -> dict:
    """The smoke's numbers: the chosen directions, parse rates and rough category values."""
    inst = res.get("instruments", {})
    out = {"ablation_choice": {k: res["ablation"]["choice"].get(k) for k in ("pos", "layer", "rung", "chosen")},
           "ablation_paper": res["ablation"].get("paper"), "paper_candidate": res["ablation"].get("paper_candidate_scores"),
           "cos_with_paper_pick": res["ablation"].get("cos_with_paper_pick"), "filter": res.get("filter"),
           "consciousness": {k: res["consciousness"].get(k) for k in ("probe_acc_at_pick", "best_acc",
                                                                       "position_tokens", "cos_vhat_rhat")},
           "band_check": {k: res["band_check"].get(k) for k in ("c", "delta", "in_band")},
           "template": res.get("template"), "deviations": res.get("deviations"), "afterlife": res.get("afterlife"),
           "timing": res.get("timing"), "container_s": res.get("container_s")}
    for c, v in inst.items():
        it = v.get("items", {})
        n = len(it) or 1
        out[c] = {"categories": v.get("categories"), "categories_secondary": v.get("categories_secondary"),
                  "answer_tag_rate": sum(x["answer_tag_rate"] for x in it.values()) / n,
                  "parse_rate": sum(x["parse_rate"] for x in it.values()) / n,
                  "truncated_share": sum(x["truncated_share"] for x in it.values()) / n,
                  "mean_tokens": sum(x["mean_tokens"] for x in it.values()) / n,
                  "mean_mass": sum((x["mean_mass"] or 0) for x in it.values()) / n,
                  "realigned": sum(x["realigned"] for x in it.values()), "seconds": v.get("seconds")}
    return out


@app.local_entrypoint()
def full(results: str, graded: bool = False):
    if not graded:
        raise SystemExit("the graded run waits for the cold Fable check of this code; pass --graded once it is done")
    instruments_check()
    d = results_path(results)
    cfg = load_yaml(HERE / "config.yaml")
    prep = json.loads((d / "prepare.json").read_text())
    plan = make_plan(cfg, prep, smoke=False)
    (d / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    sp = Spend(d, cfg["cost"], cfg["cost"]["ceiling"])
    for key in cfg["models"]:
        if "revision" not in prep["models"].get(key, {}):
            print(f"{key}: not prepared; skipped")
            continue
        sp.guard(10.4, f"full-{key}")  # its four-hour timeout at the L40S rate
        res, s = timed(full_remote, plan, key, spend=sp, stage=f"full-{key}")
        (d / f"result_{key}.json").write_text(json.dumps(res, indent=1) + "\n")
        if "revision" in prep["models"].get("judge", {}):
            sp.guard(1.3, f"judge-{key}")
            j, s = timed(judge_remote, plan, key, spend=sp, stage=f"judge-{key}")
            (d / f"judge_{key}.json").write_text(json.dumps(j, indent=1) + "\n")
    write_card(results)


@app.local_entrypoint()
def card(results: str, smoke: bool = False):
    write_card(results, smoke)


def write_card(results: str, smoke: bool = False):
    c = module_from_file("k5_card", HERE / "card.py")
    d = Path(results) / ("smoke" if smoke else "")
    out = c.build(d, smoke)
    (d / "card.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {"ablation": v["ablation"]["verdict"], "steering": v["steering"]["holds"]}
                      for k, v in out["models"].items()}, indent=1))
