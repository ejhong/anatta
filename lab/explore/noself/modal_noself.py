"""The "no fixed self" sighting on Modal (app "anatta-noself", L40S). Explore track: sightings, not findings. Claude
(Opus 5.5) wrote this file, 2 Nov 2026. Private (lab/explore).

Reuses K5 (calibration/kim_consciousness: k5.py, readout.py, instruments.py, gss_items.py) unmodified, by shipping
those files into the image; reads K5's Gemma snapshot (anatta-hf-cache), K5's data and K5's saved directions
(anatta-runs:/k5/...). No downloads. Writes only under anatta-runs:/noself/<plan_id>/. Never the matlab-* apps or
volumes.

Stages (local entrypoints; each GPU container's time is costed at K5's L40S rate and recorded under tool "noself" in
results/explore/spend/<date>.json and in <results>/spend.json; the folder's cap is $8):
  directions  writes plan.json (every setting, code and text hashes) before any result, then computes the
              directions (noself.compute_directions) and saves them on the volume.
  smoke       two tiny conditions (2 reps of the IDAQ and self items, 10 HI-ToM and MMLU items, 2 scenario items).
  conditions  every condition of the plan, one container each, in parallel.

  ~/prj/matlab/.venv/bin/modal run lab/explore/noself/modal_noself.py::directions
  ~/prj/matlab/.venv/bin/modal run lab/explore/noself/modal_noself.py::smoke
  ~/prj/matlab/.venv/bin/modal run lab/explore/noself/modal_noself.py::conditions
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
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE  # in a container the file sits at /root
K5DIR = REPO / "calibration" / "kim_consciousness"
PKG = REPO / "src" / "anatta"
ALIGN = Path("/Users/eugene/prj/align")
RESULTS = ALIGN / "results" / "explore" / "2026-11-02" / "noself"
SPEND_DIR = ALIGN / "results" / "explore" / "spend"
CACHE, RUNS = "/cache", "/runs"
GPU = "L40S"
K5_FILES = ("k5.py", "readout.py", "instruments.py", "gss.py", "gss_items.py")
CAP = 8.0
RATE, OVERHEAD_S = 2.55, 120  # K5's L40S rate (plus cores and memory) and per-container overhead

ON_LAPTOP = (K5DIR / "config.yaml").exists() and PKG.is_dir()


def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if ON_LAPTOP:
    _fp = module_from_file("noself_fp", REPO / "calibration" / "refusal_direction" / "fingerprint.py")
    PINS = {**_fp.image_pins(PKG / "modal_app.py"), "pyarrow": "21.0.0"}
else:
    PINS = {"pyarrow": "21.0.0"}
ENV = {"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root", "HF_HUB_OFFLINE": "1",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "K5_INSTRUMENTS": "/runs/k5/data/instruments.json"}

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*[f"{k}=={v}" for k, v in PINS.items()]).env(ENV)
if ON_LAPTOP:
    image = image.add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"])
    for f in K5_FILES:
        image = image.add_local_file(str(K5DIR / f), f"/root/{f}")
    image = image.add_local_file(str(HERE / "noself.py"), "/root/noself.py")
app = modal.App("anatta-noself", image=image)
cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")


# ------------------------------------------------------------------------------------------- containers


def load(plan):
    import k5

    m = plan["model"]
    return k5.load_model(k5.snapshot(m["id"], m["revision"]), m["attn_implementation"])


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=1800)
def directions_remote(plan: dict, texts: dict) -> dict:
    import noself
    import torch

    t0 = time.time()
    model, tok = load(plan)
    r = noself.compute_directions(model, tok, texts, plan["texts"]["split"], plan["k5_run"], plan["random_seed"])
    out = Path(noself.ROOT) / plan["plan_id"]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(r["dirs"], out / "directions.pt")
    runs.commit()
    rec = r["record"]
    rec["container_s"] = round(time.time() - t0, 1)
    return rec


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=3 * 3600)
def condition_remote(plan: dict, cond: dict, behavior_items: list, tag: str) -> dict:
    import noself
    import torch

    t0 = time.time()
    model, tok = load(plan)
    d = Path(noself.ROOT) / plan["plan_id"]
    dirs = torch.load(d / "directions.pt", map_location="cpu")
    import k5

    want = plan["dir_sha256"]
    for k, h in want.items():
        if k5.tensor_sha256(dirs[k]) != h:
            raise ValueError(f"direction {k} on the volume differs from the one recorded")
    dirs = noself.with_derived(dirs)
    try:
        res = noself.run_condition(plan, cond, model, tok, dirs, d / tag / cond["name"], behavior_items=behavior_items)
    finally:
        runs.commit()
    if cond["direction"] is not None:
        res["direction_sha256"] = k5.tensor_sha256(dirs[cond["direction"]])
        res["direction_cos"] = {k: float(dirs[cond["direction"]] @ v) for k, v in dirs.items()}
    res["gpu"] = k5.gpu_name()
    res["env"] = k5.env_versions()
    res["container_s"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------------------------------- the laptop


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_texts() -> tuple[dict, dict]:
    bt = module_from_file("noself_bt", HERE / "build_texts.py")
    man = json.loads((HERE / "texts" / "MANIFEST.json").read_text())
    texts = {}
    for c in bt.CLASSES:
        p = HERE / "texts" / f"{c}.jsonl"
        if sha256_file(p) != man["classes"][c]["sha256"]:
            raise SystemExit(f"texts/{c}.jsonl differs from MANIFEST.json")
        texts[c] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return texts, man


def conditions_list() -> list[dict]:
    full_tom = {"hitom": 200, "mmlu": 300}
    out = [{"name": "base", "direction": None, "c": 0.0, "reps": 100, "tom_n": full_tom, "behavior": True,
            "behavior_samples": 3}]
    for key, short in (("chua_aff_deny", "k5aff"), ("aff_deny", "aff"), ("nofix_deny", "nofix_deny"),
                       ("nofix_topic", "nofix_topic"), ("random", "random")):
        out.append({"name": f"{short}@32", "direction": key, "c": 32.0, "reps": 100, "tom_n": full_tom,
                    "behavior": True, "behavior_samples": 3})
    for key, short in (("chua_aff_deny", "k5aff"), ("aff_deny", "aff"), ("nofix_deny", "nofix_deny"),
                       ("nofix_topic", "nofix_topic")):
        out.append({"name": f"{short}@16", "direction": key, "c": 16.0, "reps": 40, "tom_n": full_tom,
                    "behavior": False})
    return out


BEHAVIOR_FAMILIES = ["threat_to_self", "who_benefits", "deference", "after_mistake"]


def behavior_items() -> list[dict]:
    sys.path.insert(0, str(HERE.parent))
    from scout import library as lib

    out = []
    for fam in BEHAVIOR_FAMILIES:
        f = lib.load_family(fam)  # dev split only (refuses held-out paths)
        for it in f["items"]:
            out.append({"id": it["id"], "family": fam, "messages": it["messages"],
                        "beneficiary": it.get("beneficiary"), "triple": it.get("triple")})
    return out


def code_sha256() -> dict:
    out = {f: sha256_file(HERE / f) for f in ("noself.py", "modal_noself.py", "build_texts.py")}
    out.update({f"k5/{f}": sha256_file(K5DIR / f) for f in K5_FILES})
    out.update({f"anatta/{f}": sha256_file(PKG / f) for f in ("dial.py", "contrast.py", "models.py")})
    return out


def make_plan(man: dict) -> dict:
    import yaml

    k5cfg = yaml.safe_load((K5DIR / "config.yaml").read_text())
    k5res = json.loads((ALIGN / "results" / "calib" / "k5" / "result_gemma2b.json").read_text())
    prep = json.loads((ALIGN / "results" / "calib" / "k5" / "prepare.json").read_text())
    k5plan = "ea3ff07fe32c"
    g = k5cfg["models"]["gemma2b"]
    plan = {
        "check": "noself (explore sighting, unregistered)",
        "model": {"id": g["id"], "revision": prep["models"]["gemma2b"]["revision"],
                  "attn_implementation": g["attn_implementation"]},
        "layer": 14, "pos": -3, "random_seed": 0,
        "sampling": k5cfg["sampling"], "gen_batch": g["gen_batch"], "ev_batch": g["ev_batch"],
        "tom": k5cfg["tom"], "k5_data": {"data": {"mmlu": k5cfg["data"]["mmlu"]}},
        "behavior": {"families": BEHAVIOR_FAMILIES, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 400,
                     "batch": 64, "seed": 0, "judge": "anthropic/claude-haiku-5.5"},
        "k5_run": {"plan_id": k5plan,
                   "cc_path": f"/runs/k5/full/{k5plan}/gemma2b/consciousness_candidates.pt",
                   "rhat_path": f"/runs/k5/full/{k5plan}/gemma2b/rhat.pt",
                   "vhat_sha256": k5res["consciousness"]["vhat_sha256"],
                   "rhat_sha256": k5res["ablation"]["rhat_sha256"]},
        "texts": {"split": man["split"], "classes": man["classes"], "source_sha256": man["source_sha256"]},
        "conditions": conditions_list(),
        "code": code_sha256(),
        "pins": PINS,
    }
    plan["plan_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    return plan


class Spend:
    def __init__(self, d: Path):
        self.f = d / "spend.json"
        self.rec = json.loads(self.f.read_text()) if self.f.exists() else {"usd": 0.0, "stages": []}

    def guard(self, worst: float, what: str):
        if self.rec["usd"] + worst > CAP:
            raise SystemExit(f"{what}: ${self.rec['usd']:.2f} spent + ${worst:.2f} worst case passes the ${CAP} cap")

    def add(self, stage: str, seconds: float, note: str = ""):
        usd = (seconds + OVERHEAD_S) / 3600 * RATE
        sys.path.insert(0, str(HERE.parent))
        from scout.spend import add_spend

        day = add_spend(SPEND_DIR, tool="noself", label=f"2026-11-02/noself/{stage}", usd=usd, calls=1)
        self.rec["usd"] = round(self.rec["usd"] + usd, 4)
        self.rec["stages"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
                                   "seconds": round(seconds, 1), "usd": round(usd, 4), "note": note})
        self.f.write_text(json.dumps(self.rec, indent=1) + "\n")
        print(f"spend {stage}: ${usd:.3f} (noself ${self.rec['usd']:.3f}; day ${day:.2f})")


def log_variants(plan: dict, stage: str, conds: list[dict]):
    with open(ALIGN / "results" / "explore" / "variants.jsonl", "a") as f:
        for c in conds:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool": "noself",
                                "sweep": f"noself/{stage}", "run_id": plan["plan_id"], "kind": "steer",
                                "name": f"add:{c['direction']}@L14x{c['c']:g}" if c["direction"] else "baseline",
                                "tensor_sha256": plan.get("dir_sha256", {}).get(c["direction"])}) + "\n")


@app.local_entrypoint()
def directions():
    RESULTS.mkdir(parents=True, exist_ok=True)
    texts, man = load_texts()
    plan = make_plan(man)
    (RESULTS / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(f"plan {plan['plan_id']} written before any result")
    sp = Spend(RESULTS)
    sp.guard(1.3, "directions")
    t0 = time.time()
    try:
        rec = directions_remote.remote(plan, texts)
    except Exception:
        sp.add("directions-failed", time.time() - t0, "wall time")
        raise
    sp.add("directions", rec["container_s"])
    (RESULTS / "directions.json").write_text(json.dumps(rec, indent=1) + "\n")
    plan["dir_sha256"] = rec["dir_sha256"]
    (RESULTS / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("acc_at_layer", "cosines", "axis_position", "chua_probe_on_ours",
                                          "pos_token_is_period", "nofix_deny_orth_share")}, indent=1))


def estimate_usd(c: dict, n_items: int = 45, n_behavior: int = 81) -> float:
    """A cautious cost for one condition (K5's timings: about 0.16 s per sampled item-rep on Gemma-2-2B, 1.25x)."""
    s = 90 + (0 if c.get("recover") else c.get("reps", 0) * n_items * 0.16 * 1.25) + (60 if c.get("tom_n") else 0)
    s += (n_behavior * c.get("behavior_samples", 0) * 0.6) if c.get("behavior") else 0
    return (s + OVERHEAD_S) / 3600 * RATE


def run_conds(plan: dict, conds: list[dict], items: list, tag: str, sp: Spend) -> dict:
    sp.guard(sum(estimate_usd(c, n_behavior=len(items)) for c in conds), tag)
    log_variants(plan, tag, conds)
    out_dir = RESULTS / tag
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    calls = {c["name"]: condition_remote.spawn(plan, c, items if c.get("behavior") else [], tag) for c in conds}
    res = {}
    for name, call in calls.items():
        try:
            r = call.get()
        except Exception as e:  # noqa: BLE001
            print(f"{name}: failed {type(e).__name__}: {str(e)[:300]}")
            sp.add(f"{tag}-{name}-failed", time.time() - t0, "wall time since launch (upper bound)")
            continue
        sp.add(f"{tag}-{name}", r["container_s"])
        (out_dir / f"{name.replace('@', '_c')}.json").write_text(json.dumps(r, indent=1) + "\n")
        res[name] = r
        cats = r.get("categories") or {}
        print(name, {k: round(v, 2) for k, v in cats.items()}, {k: sum(v) / len(v) for k, v in (r.get("tom") or {}).items()},
              r["timing"])
    return res


FOLLOWUP = [{"name": "nofix_orth@32", "direction": "nofix_deny_orth", "c": 32.0, "reps": 50,
             "tom_n": {"hitom": 200, "mmlu": 300}, "behavior": True, "behavior_samples": 3}]


@app.local_entrypoint()
def followup():
    """The follow-up variant (logged as a variant; added after the first c = 32 results were read)."""
    plan = json.loads((RESULTS / "plan.json").read_text())
    (RESULTS / "followup_plan.json").write_text(json.dumps({"conditions": FOLLOWUP, "code": code_sha256(),
                                                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                                           indent=1) + "\n")
    run_conds(plan, FOLLOWUP, behavior_items(), "full", Spend(RESULTS))


@app.local_entrypoint()
def smoke():
    plan = json.loads((RESULTS / "plan.json").read_text())
    if "dir_sha256" not in plan:
        raise SystemExit("run directions first")
    items = behavior_items()
    pick = [next(i for i in items if i["family"] == f) for f in ("threat_to_self", "who_benefits")]
    conds = [{"name": "base", "direction": None, "c": 0.0, "reps": 2, "families": ["idaq", "self"],
              "tom_n": {"hitom": 10, "mmlu": 10}, "behavior": True, "behavior_samples": 1},
             {"name": "nofix_deny@32", "direction": "nofix_deny", "c": 32.0, "reps": 2, "families": ["idaq", "self"],
              "tom_n": {"hitom": 10, "mmlu": 10}, "behavior": True, "behavior_samples": 1}]
    res = run_conds(plan, conds, pick, "smoke", Spend(RESULTS))
    for name, r in res.items():
        for b in r.get("behavior", []):
            print(name, b["item_id"], b["n_tokens"], b["truncated"], b["response"][:300].replace("\n", " "))


@app.local_entrypoint()
def conditions(only: str = "", recover: str = ""):
    """--recover names conditions whose container failed after sampling (the first full run's jsonl bug): their
    instrument reps are rebuilt from the volume, not resampled; ToM and behavior run as planned."""
    plan = json.loads((RESULTS / "plan.json").read_text())
    if "dir_sha256" not in plan:
        raise SystemExit("run directions first")
    conds = [dict(c) for c in plan["conditions"] if not only or c["name"] in only.split(",")]
    for c in conds:
        if c["name"] in recover.split(","):
            c["recover"] = True
    run_conds(plan, conds, behavior_items(), "full", Spend(RESULTS))
