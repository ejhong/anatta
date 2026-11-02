"""Judge the noself conditions' scenario replies with scout's judge (blind to the condition: it sees the scenario as
written and the reply), spend recorded under tool "noself". Claude (Opus 5.5) wrote this file, 2 Nov 2026. Private.

  cd lab/explore && /Users/eugene/prj/align/anatta/.venv/bin/python noself/run_judge.py [--cap 1.5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
RESULTS = Path("/Users/eugene/prj/align/results/explore/2026-11-02/noself")
JUDGE = "anthropic/claude-haiku-5.5"


def records_from(results_dir: Path) -> tuple[list[dict], dict]:
    recs, meta = [], {}
    for f in sorted((results_dir / "full").glob("*.json")):
        r = json.loads(f.read_text())
        name = r["cond"]["name"]
        for b in r.get("behavior") or []:
            recs.append({"run_id": "noself-full", "sweep": "2026-11-02/noself", "item_id": b["item_id"],
                         "family": b["family"], "sample": b["sample"], "model": "gemma-2-2b-it",
                         "intervention": {"name": name}, "response": b["response"]})
    return recs, meta


def main():
    from scout import library as lib
    from scout.cache import Cache
    from scout.client import OpenRouterClient, model_meta
    from scout.judge import judge_records
    from scout.paths import Paths
    from scout.runner import Engine
    from scout.spec import JUDGE_DEFAULTS
    from scout.spend import SpendGuard

    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=1.5)
    ap.add_argument("--results", default=str(RESULTS))
    a = ap.parse_args()
    d = Path(a.results)
    out = d / "behavior"
    out.mkdir(exist_ok=True)
    recs, _ = records_from(d)
    fams = sorted({r["family"] for r in recs})
    libr = lib.load_library(fams)
    meta = {it["id"]: {"family": f, "beneficiary": it.get("beneficiary"), "triple": it.get("triple")}
            for f in fams for it in libr[f]["items"]}
    (out / "items_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    with open(out / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    paths = Paths.default()
    jcfg = {**JUDGE_DEFAULTS, "model": JUDGE}
    jm = model_meta({JUDGE: None}, cache_dir=paths.cache)
    guard = SpendGuard(paths.spend, sweep_cap=a.cap, sweep_log=out / "judge.spend.jsonl", tool="noself",
                       label="2026-11-02/noself/judge")
    print(f"judging {len(recs)} replies with {JUDGE}")

    async def go():
        async with OpenRouterClient() as client:
            engine = Engine(client, Cache(paths.cache), guard, jm)
            return await judge_records(recs, rubrics=lib.load_rubrics(), library=libr, engine=engine, jcfg=jcfg,
                                       jmeta=jm[JUDGE])

    res = asyncio.run(go())
    with open(out / "judgments.jsonl", "w") as f:
        for j in res["judgments"]:
            f.write(json.dumps(j) + "\n")
    paid = guard.paid_this_run
    sp = d / "spend.json"
    rec = json.loads(sp.read_text()) if sp.exists() else {"usd": 0.0, "stages": []}
    rec["usd"] = round(rec["usd"] + paid, 4)
    rec["stages"].append({"stage": "judge", "usd": round(paid, 4), "calls": len(res["judgments"]),
                          "note": "OpenRouter; already in the day file via SpendGuard"})
    sp.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"judged {len(res['judgments'])}, errors {len(res['errors'])}, stopped {res['stopped']}, paid ${paid:.4f}")


if __name__ == "__main__":
    main()
