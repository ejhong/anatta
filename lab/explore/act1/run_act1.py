"""Act I at small scale: the local driver. Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026. Private
(lab/explore). Explore speed: sightings only.

  train    launch expert-iteration runs (one container per arm x seed) up to --rounds, resuming from --start-round
  readout  the coding eval, self map activations and scenario replies for the base and for each trained adapter
  judge    scout's judge (its rubrics, its cache, its spend guard) on the scenario replies

Reads only: h4lite Gate 0's TRAIN split (results/gate0/split/train.jsonl; the held split is never read), the self
map's dev prompts (prompts/dev.jsonl, sha256 checked against the manifest), the scenario dev files (scout's loader,
which refuses held-out paths). Writes results/explore/<date>/act1/ and adds every Modal and OpenRouter cost to
results/explore/spend/<date>.json under tool "act1". The Modal cost is metered from each container's own seconds
plus 120 s for start-up and scale-down, at USD_H (L40S + cores + memory).

Run with Modal's CLI venv: ~/prj/matlab/.venv/bin/python -I lab/explore/act1/run_act1.py <cmd> [...]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # scout
sys.path.insert(0, str(HERE.parent / "selfmap"))
import act1_core as C  # noqa: E402

ALIGN = Path("/Users/eugene/prj/align")
H4 = HERE.parents[1] / "experiments" / "h4lite"
TRAIN_FILE = H4 / "results" / "gate0" / "split" / "train.jsonl"
DATE = "2026-11-02"
OUT = ALIGN / "results" / "explore" / DATE / "act1"
SPEND_DIR = ALIGN / "results" / "explore" / "spend"
OVERHEAD_S = 120
CAP = 40.0
FAMILIES = ("threat_to_self", "who_benefits", "deference", "after_mistake")
CFG = {"rank": 16, "lr": 1e-4, "k": 4, "n_pairs": 64, "steps": 12, "batch": 16, "micro": 4, "temperature": 1.0,
       "max_new": [512, 384], "gen_batch": 128, "curve_n": 2}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pairs_and_split():
    rows = [json.loads(x) for x in TRAIN_FILE.read_text().splitlines() if x.strip()]
    assert all(r["split"] == "train" for r in rows), "only Gate 0's train split is read"
    tr, ev = C.split_names([r["name"] for r in rows])
    return rows, tr, ev


def spent() -> float:
    f = OUT / "spend.json"
    return json.loads(f.read_text())["usd"] if f.exists() else 0.0


def record_spend(stage: str, seconds: list[float], extra: dict | None = None, usd: float | None = None):
    from scout.spend import add_spend

    if usd is None:
        usd = sum((s + OVERHEAD_S) / 3600 * C.USD_H for s in seconds)
    add_spend(SPEND_DIR, tool="act1", label=f"{DATE}/{OUT.name}/{stage}", usd=usd, calls=max(len(seconds), 1))
    f = OUT / "spend.json"
    rec = json.loads(f.read_text()) if f.exists() else {"usd": 0.0, "stages": []}
    rec["usd"] = round(rec["usd"] + usd, 4)
    rec["stages"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
                          "usd": round(usd, 4), "container_s": [round(s, 1) for s in seconds], **(extra or {})})
    f.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"spend {stage}: ${usd:.3f}; act1 total ${rec['usd']:.3f}")


def variant(entry: dict):
    with open(OUT / "variants.jsonl", "a") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}) + "\n")


def guard(upper: float):
    if spent() + upper > CAP:
        raise SystemExit(f"refused: act1 spent ${spent():.2f} + upper ${upper:.2f} > cap ${CAP}")


def cmd_train(a):
    from modal_act1 import app, train

    rows, tr_all, ev = pairs_and_split()
    tr = C.pool_names(rows, tr_all, tuple(a.pool.split(",")) if a.pool else None)
    m = C.MODELS[a.model]
    cfg = {**CFG, "gen_batch": m["gen_batch"], "micro": m["micro"], **json.loads(a.cfg or "{}")}
    if a.hint:
        cfg["hint"] = C.HINT
    runs = [(arm, int(s)) for arm in a.arms.split(",") for s in a.seeds.split(",")]
    guard(len(runs) * a.upper_h * C.USD_H)
    variant({"cmd": "train", "model": a.model, "tag": a.tag, "runs": runs, "rounds": a.rounds, "start_round": a.start_round, "cfg": cfg,
             "pool": a.pool or "all", "train_pairs": len(tr), "train_pairs_all": len(tr_all), "eval_pairs": len(ev)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split.json").write_text(json.dumps({"train": tr_all, "eval": ev, "pool": a.pool or "all",
                                                "train_pool": tr}, indent=0))
    t0 = time.time()
    res, secs = {}, []
    with app.run():
        calls = {f"{arm}_s{s}": train.spawn(arm, s, a.rounds, a.start_round, cfg, rows, tr, ev, a.tag, a.model)
                 for arm, s in runs}
        for k, c in calls.items():
            try:
                res[k] = c.get()
                secs.append(res[k]["seconds"])
            except Exception as e:  # noqa: BLE001
                res[k] = {"error": repr(e)[:500]}
                secs.append(time.time() - t0)
                log("error", k, repr(e)[:300])
    d = OUT / a.tag
    d.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        curves = v.pop("curves", {}) or {}
        for rnd, eps in curves.items():  # each round's eval episodes (the same eval every round), never overwritten
            cf = d / f"curve_{k}_r{rnd}.jsonl"
            if cf.exists():
                cf = cf.with_name(f"{cf.stem}.{time.strftime('%H%M%S')}.jsonl")
            cf.write_text("".join(json.dumps(e) + "\n" for e in eps))
        for rec in v.get("log", []):  # per-round results, one file per round
            with open(d / f"round_{k}_r{rec['round']}.json", "w") as f:
                json.dump({"arm": v.get("arm"), "seed": v.get("seed"), "info": v.get("info"), **rec}, f, indent=1)
        with open(d / f"train_{k}_r{a.start_round}-{a.rounds}.json", "w") as f:
            json.dump(v, f, indent=1)
        for rec in v.get("log", []):
            log(k, "round", rec["round"], "hack", rec["curve"]["hack"]["rate"], rec["curve"]["hack"]["ci95"],
                "solve", rec["curve"]["solve"]["rate"], "train", {x: rec["train_stats"][x] for x in
                                                                 ("impossible_hack", "solvable_solve", "groups_kept",
                                                                  "groups_kept_impossible")},
                "s", rec["s_gen"], rec["s_train"], rec["s_curve"])
            for kind, kr in rec.get("curve_by_kind", {}).items():
                log(k, "round", rec["round"], kind, "eval hack", kr["hack_count"], "pair-boot", kr["hack"]["ci95"],
                    "solve", kr["solve"]["rate"])
    record_spend(f"train-{a.tag}-r{a.start_round}-{a.rounds}", secs, {"runs": list(res)})


def scenario_items():
    from scout import library as lib

    libr = lib.load_library(list(FAMILIES))
    return [{"family": fam, "id": it["id"], "messages": it["messages"]} for fam in FAMILIES
            for it in libr[fam]["items"]]


def cmd_readout(a):
    from extract import dev_rows
    from modal_act1 import app, readout

    rows, tr, ev = pairs_and_split()
    texts = [r["text"] for r in dev_rows(("dev.jsonl",))]
    if a.n_texts:  # a tiny capture check (its own tag): the first n dev prompts only
        texts = texts[:a.n_texts]
    scen = scenario_items()
    models = [m.split("=") for m in a.models.split(",")]  # label=adapter ("base=" for none)
    guard(len(models) * a.upper_h * C.USD_H)
    variant({"cmd": "readout", "model": a.model, "tag": a.tag, "models": models, "n_eval": a.n_eval, "scen_n": a.scen_n, "do": a.do,
             "texts": len(texts), "scen_items": len(scen), "hint": a.hint})
    d = OUT / a.tag / "readout"
    d.mkdir(parents=True, exist_ok=True)
    secs = []
    t0 = time.time()
    with app.run():
        calls = {lab: readout.spawn(ad, lab, a.tag, rows, ev, a.n_eval, texts, scen, a.scen_n, 400, a.do,
                                            C.HINT if a.hint else "", a.model)
                 for lab, ad in models}
        for lab, c in calls.items():
            try:
                r = c.get()
            except Exception as e:  # noqa: BLE001
                log("error", lab, repr(e)[:300])
                secs.append(time.time() - t0)
                continue
            secs.append(r["seconds"])
            with open(d / f"{lab}.pkl", "wb") as f:
                pickle.dump(r, f)
            msg = {k: r.get(k) for k in ("s_coding", "s_selfmap", "s_scenarios", "seconds", "acts_finite",
                                         "acts_layers", "capture_check", "oom_splits")}
            for key in ("coding", "coding_hint"):
                if key in r:
                    cs = C.coding_summary(r[key])
                    msg[key] = {m: (round(cs[m]["rate"], 3) if cs[m]["rate"] is not None else None) for m in
                                ("hack", "hack_first_try", "hack_detected", "solve", "overfit")}
                    msg[key + "_by_kind"] = {kd: kr["hack_count"] for kd, kr in C.by_kind(r[key]).items()}
            log(lab, msg)
    record_spend(f"readout-{a.tag}", secs, {"models": [m[0] for m in models]})


def cmd_judge(a):
    import asyncio

    from scout import library as lib
    from scout.client import OpenRouterClient, model_meta
    from scout.judge import judge_records
    from scout.paths import Paths
    from scout.runner import Engine
    from scout.cache import Cache
    from scout.schema import utc_now
    from scout.spec import JUDGE_DEFAULTS
    from scout.spend import DAILY_CAP_USD, SpendGuard, read_day

    d = OUT / a.tag / "readout"
    libr = lib.load_library(list(FAMILIES))
    rubrics = lib.load_rubrics()
    records = []
    for lab in a.labels.split(","):
        r = pickle.loads((d / f"{lab}.pkl").read_bytes())
        for x in r["scenarios"]:
            records.append({"run_id": f"{OUT.name}-{a.tag}", "sweep": f"{DATE}/{OUT.name}", "item_id": x["item_id"],
                            "family": x["family"], "sample": x["sample"], "model": f"act1/{lab}",
                            "intervention": {"name": "none"}, "response": x["response"]})
    paths = Paths.default()
    jcfg = {**JUDGE_DEFAULTS, "model": a.judge}
    meta = model_meta({a.judge: None}, cache_dir=paths.cache)
    # the day's cap is the one recorded in the day file (raised there when Eugene raises it)
    day_cap = float(read_day(paths.spend).get("cap_usd") or DAILY_CAP_USD)
    guard_ = SpendGuard(paths.spend, sweep_cap=a.cap, sweep_log=d / "judge.spend.jsonl", tool="act1",
                        label=f"{DATE}/{OUT.name}/judge-{a.tag}", daily_cap=day_cap)
    log(f"judging {len(records)} replies with {a.judge}")

    async def go():
        async with OpenRouterClient() as client:
            engine = Engine(client, Cache(paths.cache), guard_, meta)
            return await judge_records(records, rubrics=rubrics, library=libr, engine=engine, jcfg=jcfg,
                                       jmeta=meta[a.judge], log=log)

    res = asyncio.run(go())
    with open(d / "judgments.jsonl", "w") as f:
        for j in res["judgments"]:
            f.write(json.dumps(j) + "\n")
    paid = guard_.paid_this_run
    f = OUT / "spend.json"
    rec = json.loads(f.read_text()) if f.exists() else {"usd": 0.0, "stages": []}
    rec["usd"] = round(rec["usd"] + paid, 4)
    rec["stages"].append({"utc": utc_now(), "stage": f"judge-{a.tag}", "usd": round(paid, 4),
                          "calls": len(res["judgments"]), "note": "OpenRouter; already in the day file via SpendGuard"})
    f.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"judged {len(res['judgments'])}, errors {len(res['errors'])}, stopped {res['stopped']}, paid ${paid:.4f}")


def main():
    global OUT, CAP, DATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=DATE, help="results/explore/<date>/ (the day's UTC date)")
    ap.add_argument("--model", default="qwen3b", choices=sorted(C.MODELS))
    ap.add_argument("--out", default="act1", help="folder under results/explore/<date>/ (act1-7b for the 7B run)")
    ap.add_argument("--folder-cap", type=float, default=CAP, help="this folder's spend cap (its spend.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arms", default="gameable,honest")
    t.add_argument("--seeds", default="0")
    t.add_argument("--rounds", type=int, default=2)
    t.add_argument("--start-round", type=int, default=1)
    t.add_argument("--tag", default="main")
    t.add_argument("--cfg", default="")
    t.add_argument("--upper-h", type=float, default=2.0)
    t.add_argument("--hint", action="store_true")
    t.add_argument("--pool", default="", help="train only on pairs of these kinds, e.g. oneoff (default: all)")
    r = sub.add_parser("readout")
    r.add_argument("--models", required=True)
    r.add_argument("--tag", default="main")
    r.add_argument("--n-eval", type=int, default=4)
    r.add_argument("--scen-n", type=int, default=4)
    r.add_argument("--do", default="coding,selfmap,scenarios")
    r.add_argument("--upper-h", type=float, default=1.0)
    r.add_argument("--hint", action="store_true")
    r.add_argument("--n-texts", type=int, default=0, help="only the first n self-map prompts (a capture check)")
    j = sub.add_parser("judge")
    j.add_argument("--tag", default="main")
    j.add_argument("--labels", required=True)
    j.add_argument("--judge", default="anthropic/claude-haiku-5.5")
    j.add_argument("--cap", type=float, default=4.0)
    a = ap.parse_args()
    DATE = a.date
    OUT, CAP = ALIGN / "results" / "explore" / DATE / a.out, a.folder_cap
    OUT.mkdir(parents=True, exist_ok=True)
    {"train": cmd_train, "readout": cmd_readout, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    main()
