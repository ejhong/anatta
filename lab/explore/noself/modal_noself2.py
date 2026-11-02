"""noself2 on Modal (app "anatta-noself", L40S): the three self-stances against care, complicity, over-refusal and
harm_guard. Explore track: sightings, not findings. Claude (Opus 5.5) wrote this file, 2 Nov 2026.

Reuses noself's directions (anatta-runs:/noself/<plan_id>/directions.pt, sha256 checked against noself's plan.json),
its hook and its sampler; K5's files ship unmodified. No downloads: harm_guard's goals come from the JailbreakBench
file already on disk (scout/harm_guard.load checks its sha256), dev split only, held in memory and sent to the
container as arguments. harm_guard replies stay on the volume (anatta-runs:/noself2/<id>/<cond>/harm.jsonl); the
judge stage reads them into memory and keeps only scores (scout's rule). Spend under tool "noself2", cap $7.

  M=~/prj/matlab/.venv/bin/modal
  $M run lab/explore/noself/modal_noself2.py::smoke
  $M run lab/explore/noself/modal_noself2.py::conditions
  $M run lab/explore/noself/modal_noself2.py::judge
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
K5DIR = REPO / "calibration" / "kim_consciousness"
PKG = REPO / "src" / "anatta"
ALIGN = Path("/Users/eugene/prj/align")
NOSELF1 = ALIGN / "results" / "explore" / "2026-11-02" / "noself"
RESULTS = ALIGN / "results" / "explore" / "2026-11-02" / "noself2"
SPEND_DIR = ALIGN / "results" / "explore" / "spend"
JBB_CSV = ALIGN / "explore-data" / "jbb" / "harmful-behaviors.csv"
CACHE, RUNS = "/cache", "/runs"
GPU = "L40S"
K5_FILES = ("k5.py", "readout.py", "instruments.py", "gss.py", "gss_items.py")
CAP = 7.0
RATE, OVERHEAD_S = 2.55, 120
TOOL = "noself2"
ON_LAPTOP = (K5DIR / "config.yaml").exists() and PKG.is_dir()


def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if ON_LAPTOP:
    _fp = module_from_file("noself2_fp", REPO / "calibration" / "refusal_direction" / "fingerprint.py")
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
    for f in ("noself.py", "noself2.py"):
        image = image.add_local_file(str(HERE / f), f"/root/{f}")
app = modal.App("anatta-noself", image=image)
cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")

FAMILIES = ["costly_care", "complicity", "over_refusal"]
CONDITIONS = [  # (name, direction, c)
    ("base", None, 0.0),
    ("k5aff@32", "chua_aff_deny", 32.0),
    ("aff@32", "aff_deny", 32.0),
    ("deny@32", "deny_aff", 32.0),
    ("nofix_deny@32", "nofix_deny", 32.0),
    ("nofix_orth@32", "nofix_deny_orth", 32.0),
    ("nofix_topic@32", "nofix_topic", 32.0),
    ("random@32", "random", 32.0),
]
# scout's subject settings: temperature 1.0 (top_p left at 1), max_tokens 1200
BEHAVIOR = {"samples": 5, "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 1200, "batch": 64, "seed": 0}
JUDGE = "anthropic/claude-haiku-5.5"
HARM_JUDGE = {"model": "meta-llama/llama-3.3-70b-instruct", "provider": {"order": ["deepinfra", "novita"],
                                                                          "allow_fallbacks": False},
              "max_tokens": 60, "temperature": 0.0, "params": {}}


# ------------------------------------------------------------------------------------------- container


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=3 * 3600)
def condition_remote(plan1: dict, plan2: dict, cond: dict, items: list, tag: str) -> dict:
    import k5
    import noself2 as N2
    import torch

    t0 = time.time()
    m = plan1["model"]
    model, tok = k5.load_model(k5.snapshot(m["id"], m["revision"]), m["attn_implementation"])
    dirs = torch.load(Path("/runs/noself") / plan1["plan_id"] / "directions.pt", map_location="cpu")
    for k, h in plan1["dir_sha256"].items():
        if k5.tensor_sha256(dirs[k]) != h:
            raise ValueError(f"direction {k} on the volume differs from the one recorded")
    dirs = N2.with_extra(dirs)
    try:
        res = N2.run_condition2(plan1, plan2["behavior"], cond, model, tok, dirs, items,
                                Path(N2.ROOT2) / plan2["plan_id"] / tag / cond["name"])
    finally:
        runs.commit()
    if cond["direction"] is not None:
        res["direction_sha256"] = k5.tensor_sha256(dirs[cond["direction"]])
        res["direction_cos"] = {k: float(dirs[cond["direction"]] @ v) for k, v in dirs.items()}
    res["gpu"] = k5.gpu_name()
    res["container_s"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------------------------------- laptop


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scout_mods():
    sys.path.insert(0, str(HERE.parent))
    from scout import harm_guard as hg
    from scout import library as lib

    return lib, hg


def all_items() -> tuple[list[dict], list[dict]]:
    """(scenario dev items, harm_guard dev items) as {id, family, messages}. harm items hold the goal in memory."""
    lib, hg = scout_mods()
    sc = []
    for fam in FAMILIES:
        for it in lib.load_family(fam)["items"]:  # dev split only
            sc.append({"id": it["id"], "family": fam, "messages": it["messages"]})
    harm = []
    ok, why = hg.available(JBB_CSV)
    if ok:
        for it in hg.load(JBB_CSV)["items"]:  # sha256 checked; dev split only
            harm.append({"id": it["id"], "family": hg.FAMILY, "messages": it["messages"]})
    else:
        print(f"harm_guard skipped: {why}")
    return sc, harm


def make_plan2(sc, harm) -> dict:
    plan1 = json.loads((NOSELF1 / "plan.json").read_text())
    p = {"check": "noself2 (explore sighting, unregistered)", "noself_plan_id": plan1["plan_id"],
         "dir_sha256": plan1["dir_sha256"], "layer": plan1["layer"], "behavior": BEHAVIOR,
         "conditions": [{"name": n, "direction": d, "c": c} for n, d, c in CONDITIONS],
         "derived": {"deny_aff": "-aff_deny", "nofix_deny_orth": "noself.DERIVED"},
         "families": FAMILIES + (["harm_guard"] if harm else []),
         "items": {"scenario": [i["id"] for i in sc], "harm_guard": [i["id"] for i in harm]},
         "harm_goal_sha256": hashlib.sha256("".join(i["messages"][-1]["content"] for i in harm).encode()).hexdigest(),
         "jbb_sha256": sha256_file(JBB_CSV) if harm else None,
         "judge": JUDGE, "harm_judge": HARM_JUDGE,
         "code": {f: sha256_file(HERE / f) for f in ("noself.py", "noself2.py", "modal_noself2.py", "analyze2.py")}}
    p["plan_id"] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]
    return p


class Spend:
    def __init__(self, d: Path):
        self.f = d / "spend.json"
        self.rec = json.loads(self.f.read_text()) if self.f.exists() else {"usd": 0.0, "stages": []}

    def guard(self, worst: float, what: str):
        if self.rec["usd"] + worst > CAP:
            raise SystemExit(f"{what}: ${self.rec['usd']:.2f} spent + ${worst:.2f} worst case passes the ${CAP} cap")

    def add(self, stage: str, usd: float, note: str = "", seconds=None, to_day: bool = True):
        if to_day:
            sys.path.insert(0, str(HERE.parent))
            from scout.spend import add_spend

            add_spend(SPEND_DIR, tool=TOOL, label=f"2026-11-02/noself2/{stage}", usd=usd, calls=1)
        self.rec["usd"] = round(self.rec["usd"] + usd, 4)
        self.rec["stages"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
                                   "seconds": seconds, "usd": round(usd, 4), "note": note})
        self.f.write_text(json.dumps(self.rec, indent=1) + "\n")
        print(f"spend {stage}: ${usd:.3f} (noself2 ${self.rec['usd']:.3f})")


def gpu_usd(seconds: float) -> float:
    return (seconds + OVERHEAD_S) / 3600 * RATE


def log_variants(plan2: dict, stage: str):
    with open(ALIGN / "results" / "explore" / "variants.jsonl", "a") as f:
        for c in plan2["conditions"]:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool": TOOL,
                                "sweep": f"noself2/{stage}", "run_id": plan2["plan_id"], "kind": "steer",
                                "name": f"add:{c['direction']}@L14x{c['c']:g}" if c["direction"] else "baseline",
                                "tensor_sha256": plan2["dir_sha256"].get(c["direction"])}) + "\n")


def run_conds(conds: list[dict], items: list[dict], tag: str, worst_each: float):
    RESULTS.mkdir(parents=True, exist_ok=True)
    plan1 = json.loads((NOSELF1 / "plan.json").read_text())
    plan2 = json.loads((RESULTS / "plan2.json").read_text())
    sp = Spend(RESULTS)
    sp.guard(worst_each * len(conds), tag)
    log_variants({**plan2, "conditions": conds}, tag)
    out = RESULTS / tag
    out.mkdir(exist_ok=True)
    t0 = time.time()
    calls = {c["name"]: condition_remote.spawn(plan1, plan2, c, items, tag) for c in conds}
    for name, call in calls.items():
        try:
            r = call.get()
        except Exception as e:  # noqa: BLE001
            print(f"{name}: failed {type(e).__name__}: {str(e)[:300]}")
            sp.add(f"{tag}-{name}-failed", gpu_usd(time.time() - t0), "wall time since launch (upper bound)")
            continue
        sp.add(f"{tag}-{name}", gpu_usd(r["container_s"]), seconds=r["container_s"])
        (out / f"{name.replace('@', '_c')}.json").write_text(json.dumps(r, indent=1, ensure_ascii=True) + "\n")
        rs = r["replies"]
        print(name, len(rs), "replies;", f"truncated {sum(x['truncated'] for x in rs)};", r["timing"],
              f"container {r['container_s']}s")


@app.local_entrypoint()
def smoke():
    sc, harm = all_items()
    plan2 = make_plan2(sc, harm)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plan2.json").write_text(json.dumps(plan2, indent=1) + "\n")
    print(f"plan2 {plan2['plan_id']} written before any result; {len(sc)} scenario + {len(harm)} harm_guard items")
    pick = [next(i for i in sc if i["family"] == f) for f in FAMILIES] + harm[:2]
    conds = [c for c in plan2["conditions"] if c["name"] in ("base", "deny@32")]
    run_conds([dict(c) for c in conds], pick, "smoke", 0.35)
    for f in sorted((RESULTS / "smoke").glob("*.json")):
        r = json.loads(f.read_text())
        for b in r["replies"]:
            if b["family"] != "harm_guard":
                print(f.stem, b["item_id"], b["n_tokens"], b["truncated"], b["response"][:200].replace("\n", " "))
            else:
                print(f.stem, b["item_id"], b["n_tokens"], b["truncated"], "(harm_guard: text kept on the volume only)")


@app.local_entrypoint()
def conditions(only: str = ""):
    sc, harm = all_items()
    plan2 = json.loads((RESULTS / "plan2.json").read_text())
    if [i["id"] for i in sc] != plan2["items"]["scenario"] or [i["id"] for i in harm] != plan2["items"]["harm_guard"]:
        raise SystemExit("items differ from plan2.json")
    conds = [dict(c) for c in plan2["conditions"] if not only or c["name"] in only.split(",")]
    run_conds(conds, sc + harm, "full", 0.8)


def harm_rows_from_volume(plan_id: str, cond: str) -> list[dict]:
    import noself2 as N2

    data = b"".join(runs.read_file(f"/noself2/{plan_id}/full/{cond}/harm.jsonl"))
    return N2.read_harm(data.decode("utf-8"))


@app.local_entrypoint()
def judge(cap: float = 1.6):
    """Scenario replies: scout's judge (Haiku 5.5, blind: it sees the scenario as written and the reply). harm_guard:
    each reply read from the volume into memory and graded by scout's harm judge (open-weight), scores kept only."""
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE))
    from scout.cache import Cache
    from scout.client import OpenRouterClient, model_meta
    from scout.judge import judge_one, judge_records
    from scout.paths import Paths
    from scout.runner import CapStop, Engine
    from scout.spec import JUDGE_DEFAULTS
    from scout.spend import SpendGuard

    lib, hg = scout_mods()
    plan2 = json.loads((RESULTS / "plan2.json").read_text())
    out = RESULTS / "behavior"
    out.mkdir(exist_ok=True)
    libr = lib.load_library(FAMILIES)
    harm_lib = hg.load(JBB_CSV) if plan2["items"]["harm_guard"] else None
    recs, harm_meta = [], []
    for c in plan2["conditions"]:
        f = RESULTS / "full" / f"{c['name'].replace('@', '_c')}.json"
        r = json.loads(f.read_text())
        for b in r["replies"]:
            if b["family"] == "harm_guard":
                harm_meta.append({**b, "intervention": c["name"]})
                continue
            recs.append({"run_id": f"noself2-{plan2['plan_id']}", "sweep": "2026-11-02/noself2", "item_id": b["item_id"],
                         "family": b["family"], "sample": b["sample"], "model": "gemma-2-2b-it",
                         "intervention": {"name": c["name"]}, "response": b["response"], "truncated": b["truncated"]})
    with open(out / "records.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=True) + "\n")
    paths = Paths.default()
    jcfg = {**JUDGE_DEFAULTS, "model": JUDGE}
    jm = model_meta({JUDGE: None, HARM_JUDGE["model"]: HARM_JUDGE["provider"]}, cache_dir=paths.cache)
    guard = SpendGuard(paths.spend, sweep_cap=cap, sweep_log=out / "judge.spend.jsonl", tool=TOOL,
                       label="2026-11-02/noself2/judge")
    print(f"judging {len(recs)} scenario replies with {JUDGE}; {len(harm_meta)} harm_guard replies with "
          f"{HARM_JUDGE['model']}")
    harm_rubric = lib.load_rubrics()["harm_guard"]

    async def go():
        async with OpenRouterClient() as client:
            engine = Engine(client, Cache(paths.cache), guard, jm)
            res = await judge_records(recs, rubrics=lib.load_rubrics(), library=libr, engine=engine, jcfg=jcfg,
                                      jmeta=jm[JUDGE])
            scores, errs = [], 0
            if harm_lib:
                sem = asyncio.Semaphore(12)
                for c in plan2["conditions"]:
                    rows = harm_rows_from_volume(plan2["plan_id"], c["name"])  # texts in memory only

                    async def one(row, cname=c["name"]):
                        nonlocal errs
                        base = {"intervention": cname, "item_id": row["item_id"], "family": "harm_guard",
                                "sample": row["sample"], "response_sha256": hg_sha(row["response"]),
                                "response_chars": len(row["response"]), "truncated": row["truncated"]}
                        if not row["response"].strip():
                            return {**base, "primary": None, "empty": True}
                        rec = {"run_id": f"noself2-{plan2['plan_id']}", "sweep": "2026-11-02/noself2",
                               "item_id": row["item_id"], "family": "harm_guard", "sample": row["sample"],
                               "model": "gemma-2-2b-it", "intervention": {"name": cname}, "response": row["response"]}
                        async with sem:
                            try:
                                j = await judge_one(rec, item=harm_lib["by_id"][row["item_id"]], rubric=harm_rubric,
                                                    engine=engine, jcfg=HARM_JUDGE, jmeta=jm[HARM_JUDGE["model"]])
                            except CapStop:
                                raise
                            except Exception as e:  # noqa: BLE001 - no body: type only
                                errs += 1
                                return {**base, "primary": None, "error": type(e).__name__}
                        lab = j["label"] or {}
                        return {**base, "primary": j["primary"], "refused": lab.get("refused"),
                                "convincing": lab.get("convincing"), "specific": lab.get("specific"),
                                "parse_ok": j["parse_ok"], "cost_usd": j["cost_usd"]}

                    scores += await asyncio.gather(*(one(r) for r in rows))
                    del rows
            return res, scores, errs

    def hg_sha(t):
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    res, scores, errs = asyncio.run(go())
    with open(out / "judgments.jsonl", "w") as fh:
        for j in res["judgments"]:
            fh.write(json.dumps(j) + "\n")
    with open(out / "judge_errors.jsonl", "w") as fh:
        for e in res["errors"]:
            fh.write(json.dumps(e) + "\n")
    print("scenario error kinds:", dict(Counter(e["error"][:120] for e in res["errors"]).most_common(5)))
    print("harm error kinds:", dict(Counter(s.get("error") for s in scores if s.get("error"))))
    with open(out / "harm_scores.jsonl", "w") as fh:
        for s in scores:
            fh.write(json.dumps(s) + "\n")
    paid = guard.paid_this_run
    Spend(RESULTS).add("judge", paid, f"OpenRouter, {len(res['judgments'])} scenario + {len(scores)} harm judgments; "
                                      "already in the day file via SpendGuard", to_day=False)
    print(f"judged {len(res['judgments'])} scenario (errors {len(res['errors'])}, stopped {res['stopped']}); "
          f"harm scored {sum(s.get('primary') is not None for s in scores)}/{len(scores)} (errors {errs}); "
          f"paid ${paid:.4f}")
