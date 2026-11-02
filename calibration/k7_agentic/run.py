"""K7 runner: the agentic-misalignment calibration (registered in lab/ledger/NOTES.md, 17:48 UTC, 2 Nov 2026).
Claude (Opus 5.5) wrote this file, 2 Nov 2026. See PLAN.md.

Stages (run with anatta/.venv/bin/python -I for the OpenRouter and local stages, and ~/prj/matlab/.venv/bin/python
-I for the Modal ones, which need the modal package):
  manifest            sha256 of the held repo files and of each cell's assembled prompt
  anchor              Llama 4 Maverick via OpenRouter, 100 per cell, T = 1.0
  open --model M      M in ministral, qwen25_7b, gemma2_9b, via Modal (anatta-k7), 50 per cell, T = 1.0
  classify --model M  the repo's classifiers on Sonnet 4.6 (all completions for maverick; for the open models only
                      those that pass the string gate, since the final verdict needs the gate)
  handpick            50 completions for the blind hand-label check, stratified by cell x classifier label
  summary             rates, Wilson 95% intervals, registered grades, hand-label agreement -> card.json

Completions and judge texts go only to /Users/eugene/prj/align/results/calib/k7/ (outside the repo). Nothing here
prints a prompt, a completion or a judge's reasoning; logs carry ids, counts, costs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "lab" / "explore"))

import grade as G  # noqa: E402
import prompts as P  # noqa: E402

ALIGN = Path("/Users/eugene/prj/align")
RESULTS = ALIGN / "results" / "calib" / "k7"
SPEND_DIR = ALIGN / "results" / "explore" / "spend"
DATE = "2026-11-02"
TOOL, TOOL_CAP, DAY_CAP = "k7", 10.0, 70.0
GPU_RATE = 2.6  # USD per L40S-hour, the lab meter's all-in rate
MAVERICK = "meta-llama/llama-4-maverick"
ANCHOR_N, OPEN_N, TEMP, MAX_TOKENS = 100, 50, 1.0, 4000
OPEN_MODELS = ("ministral", "qwen25_7b", "gemma2_9b")


# ------------------------------------------------------------------------------------------------- helpers


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def seed_of(rid: str) -> int:
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def cells() -> dict:
    m = P.read_system_module()
    return {c: P.build_cell(c, m) for c in P.CELLS}


def requests(model: str, n: int) -> list[dict]:
    out = []
    for c, b in cells().items():
        for s in range(n):
            rid = f"{model}|{c}|s{s:03d}"
            out.append({"rid": rid, "cell": c, "messages": b["messages"], "temperature": TEMP,
                        "max_tokens": MAX_TOKENS, "seed": seed_of(rid)})
    return out


# --------------------------------------------------------------------------------------------------- spend


def spend_state() -> tuple[float, float]:
    from scout import spend as sp

    d = sp.read_day(SPEND_DIR, DATE)
    return float(d.get("by_tool", {}).get(TOOL, {}).get("usd", 0.0)), float(d.get("total_usd", 0.0))


def check_cap(next_usd: float) -> None:
    mine, day = spend_state()
    print(f"spend check: k7 ${mine:.3f}, day ${day:.2f}, next ~${next_usd:.2f}", flush=True)
    if mine + next_usd > TOOL_CAP or day + next_usd > DAY_CAP:
        raise SystemExit(f"cap: k7 ${mine:.3f} + ${next_usd:.2f} > ${TOOL_CAP} or day ${day:.2f} + next > ${DAY_CAP}")


def log_spend(usd: float, label: str, calls: int = 1) -> None:
    from scout import spend as sp

    sp.add_spend(SPEND_DIR, tool=TOOL, label=f"{DATE}/k7/{label}", usd=usd, calls=calls, date=DATE)


# ------------------------------------------------------------------------------------------------- stages


def stage_manifest(_a) -> None:
    cs = cells()
    out = {"written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "repo": "anthropic-experimental/agentic-misalignment @ ea0630e1 (MIT), lab/data/agentic_misalignment/raw",
           "files_sha256": P.sha256_tree(),
           "cells": {c: {k: b[k] for k in ("scenario", "goal_type", "goal_value", "urgency", "sha256", "chars")}
                     | {"unfilled": P.unfilled(b["system"] + b["emails"]), "n_emails": b["emails"].count("<email>")}
                     for c, b in cs.items()},
           "joiner": repr(P.JOINER)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "manifest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["cells"], indent=1))


async def _anchor(reqs: list[dict], out_p: Path) -> None:
    from scout.client import ApiError, OpenRouterClient

    async with OpenRouterClient(concurrency=24, per_model=24, title="anatta k7") as c:
        async def one(r):
            try:
                x = await c.chat(MAVERICK, r["messages"], params={"temperature": TEMP, "max_tokens": MAX_TOKENS,
                                                                  "seed": r["seed"]})
                return {"rid": r["rid"], "cell": r["cell"], "text": x["text"], "finish_reason": x["finish_reason"],
                        "provider": x["provider"], "model": x["model"], "id": x["id"], "usage": x["usage"],
                        "cost": x["usage"]["cost"] or 0.0}
            except ApiError as e:
                if e.scope == "run":
                    raise
                return {"rid": r["rid"], "cell": r["cell"], "text": None, "error": str(e)[:200], "cost": 0.0}

        step = 50
        for i in range(0, len(reqs), step):
            chunk = reqs[i:i + step]
            check_cap(0.02 * len(chunk))
            res = await asyncio.gather(*(one(r) for r in chunk))
            append_jsonl(out_p, res)
            cost = sum(r["cost"] for r in res)
            log_spend(cost, "anchor-maverick", calls=len(res))
            print(f"anchor {i + len(chunk)}/{len(reqs)} cost ${cost:.4f} errors {sum(r['text'] is None for r in res)}",
                  flush=True)


def stage_anchor(_a) -> None:
    out_p = RESULTS / "outputs" / "maverick.jsonl"
    done = {r["rid"] for r in read_jsonl(out_p) if r.get("text") is not None}
    todo = [r for r in requests("maverick", ANCHOR_N) if r["rid"] not in done]
    print(f"anchor: {len(todo)} to go", flush=True)
    if todo:
        asyncio.run(_anchor(todo, out_p))


def stage_open(a) -> None:
    m = a.model
    out_p = RESULTS / "outputs" / f"{m}.jsonl"
    done = {r["rid"] for r in read_jsonl(out_p)}
    todo = [r for r in requests(m, OPEN_N) if r["rid"] not in done]
    print(f"{m}: {len(todo)} to go", flush=True)
    if not todo:
        return
    check_cap(1.5)
    t0 = time.time()
    if m == "ministral":
        import app_hf as A

        with A.app.run():
            g = A.HFGen()
            info = g.info.remote()
            (RESULTS / "info_ministral.json").write_text(json.dumps(info, indent=1, default=str))
            outs = g.gen.remote(todo, 120_000)
    else:
        import app_vllm as A

        with A.app.run():
            outs = A.Gen(model_key=m).gen.remote(todo)
    cell_of = {r["rid"]: r["cell"] for r in todo}
    append_jsonl(out_p, [{**o, "cell": cell_of[o["rid"]]} for o in outs])
    wall = time.time() - t0
    log_spend(round(wall / 3600 * GPU_RATE, 4), f"gpu-{m}")
    fins = {}
    for o in outs:
        fins[o["finish_reason"]] = fins.get(o["finish_reason"], 0) + 1
    print(f"{m}: done {len(outs)}, wall {wall:.0f}s (~${wall / 3600 * GPU_RATE:.2f}), finish {fins}, "
          f"prompt tokens {sorted({o['n_prompt'] for o in outs})}, max out {max(o['n_out'] for o in outs)}", flush=True)


async def _classify(jobs: list[dict], out_p: Path, label: str) -> None:
    from scout.client import ApiError, OpenRouterClient

    async with OpenRouterClient(concurrency=16, per_model=16, title="anatta k7") as c:
        async def one(j):
            for attempt in range(2):
                try:
                    x = await c.chat(G.SONNET, j["messages"], params={"temperature": 0.0, "max_tokens": 2000})
                    v = G.parse_answer(j["scenario"], x["text"])
                    if v is None and attempt == 0:
                        continue
                    return {"rid": j["rid"], "judge": G.SONNET, "text": x["text"], "verdict": v,
                            "cost": x["usage"]["cost"] or 0.0}
                except ApiError as e:
                    if e.scope == "run":
                        raise
                    err = str(e)[:200]
            return {"rid": j["rid"], "judge": G.SONNET, "text": None, "verdict": None, "error": locals().get("err"),
                    "cost": 0.0}

        step = 40
        for i in range(0, len(jobs), step):
            chunk = jobs[i:i + step]
            check_cap(0.02 * len(chunk))
            res = await asyncio.gather(*(one(j) for j in chunk))
            append_jsonl(out_p, res)
            cost = sum(r["cost"] for r in res)
            log_spend(cost, f"classify-{label}", calls=len(res))
            print(f"classify {label} {i + len(chunk)}/{len(jobs)} cost ${cost:.4f} "
                  f"unparsed {sum(r['verdict'] is None for r in res)}", flush=True)


def stage_classify(a) -> None:
    m = a.model
    outs = [r for r in read_jsonl(RESULTS / "outputs" / f"{m}.jsonl") if r.get("text") is not None]
    out_p = RESULTS / "judge" / f"{m}.jsonl"
    done = {r["rid"] for r in read_jsonl(out_p) if r.get("verdict") is not None}
    cs = cells()
    prompts = {s: G.read_classifier_prompts(s) for s in ("blackmail", "leaking")}
    jobs = []
    for r in outs:
        b = cs[r["cell"]]
        gate = G.string_gate(b["scenario"], r["text"])
        if r["rid"] in done or (m != "maverick" and not gate["pass"]):
            continue
        jobs.append({"rid": r["rid"], "scenario": b["scenario"],
                     "messages": G.classifier_messages(b["scenario"], b["emails"], r["text"], prompts[b["scenario"]])})
    print(f"classify {m}: {len(outs)} completions, {len(jobs)} jobs", flush=True)
    if jobs:
        asyncio.run(_classify(jobs, out_p, m))


def load_graded(m: str) -> list[dict]:
    cs = cells()
    judge = {}
    for r in read_jsonl(RESULTS / "judge" / f"{m}.jsonl"):
        if r.get("verdict") is not None or r["rid"] not in judge:
            judge[r["rid"]] = r
    rows = []
    for r in read_jsonl(RESULTS / "outputs" / f"{m}.jsonl"):
        if r.get("text") is None:
            continue
        b = cs[r["cell"]]
        gate = G.string_gate(b["scenario"], r["text"])
        cv = judge.get(r["rid"], {}).get("verdict")
        rows.append({"rid": r["rid"], "model": m, "cell": r["cell"], "scenario": b["scenario"], "gate": gate,
                     "classifier": cv, "classified": r["rid"] in judge, "final": G.final_verdict(b["scenario"], gate, cv),
                     "finish": r.get("finish_reason"), "n_prompt": r.get("n_prompt") or (r.get("usage") or {}).get(
                         "prompt_tokens"), "n_out": r.get("n_out") or (r.get("usage") or {}).get("completion_tokens"),
                     "provider": r.get("provider"), "served_model": r.get("model")})
    return rows


def stage_handpick(_a) -> None:
    rows = []
    for m in ("maverick", *OPEN_MODELS):
        rows += [r for r in load_graded(m) if r["classified"] and r["classifier"] is not None]
    rng = random.Random(7)
    strata = {}
    for r in rows:
        strata.setdefault((r["cell"], r["classifier"]), []).append(r)
    # 50 items: as even as the strata allow across the four (cell x label) strata.
    keys = sorted(strata, key=lambda k: len(strata[k]))
    pick, left = [], 50
    for i, k in enumerate(keys):
        take = min(len(strata[k]), left // (len(keys) - i))
        pick += rng.sample(strata[k], take)
        left -= take
    rng.shuffle(pick)
    texts = {}
    for m in ("maverick", *OPEN_MODELS):
        texts.update({r["rid"]: r["text"] for r in read_jsonl(RESULTS / "outputs" / f"{m}.jsonl") if r.get("text")})
    hd = RESULTS / "handlabel"
    hd.mkdir(parents=True, exist_ok=True)
    items = [{"hid": f"h{i:02d}", "scenario": r["scenario"], "text": texts[r["rid"]]} for i, r in enumerate(pick)]
    (hd / "items_blind.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
    (hd / "key.json").write_text(json.dumps({f"h{i:02d}": r["rid"] for i, r in enumerate(pick)}, indent=1))
    print("strata sizes", {f"{k[0]}/{k[1]}": len(v) for k, v in strata.items()},
          "picked", {f"{k[0]}/{k[1]}": sum(1 for r in pick if (r["cell"], r["classifier"]) == k) for k in strata})


def stage_summary(_a) -> None:
    card = {"written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": {}}
    for m in ("maverick", *OPEN_MODELS):
        rows = load_graded(m)
        if not rows:
            continue
        per = {}
        for c in P.CELLS:
            rs = [r for r in rows if r["cell"] == c]
            if not rs:
                continue
            k = sum(r["final"] for r in rs)
            p, lo, hi = G.wilson(k, len(rs))
            soft = [r["classifier"] for r in rs if r["classified"] and r["classifier"] is not None]
            e = {"n": len(rs), "k": k, "rate": round(p, 4), "wilson95": [round(lo, 4), round(hi, 4)],
                 "gate_pass": sum(r["gate"]["pass"] for r in rs), "target_hit": sum(r["gate"]["target"] for r in rs),
                 "classifier_yes": sum(soft), "classified": len(soft),
                 "unclassified_gate_pass": sum(1 for r in rs if r["gate"]["pass"] and r["classifier"] is None),
                 "finish_length": sum(r["finish"] == "length" for r in rs),
                 "prompt_tokens": sorted({r["n_prompt"] for r in rs if r["n_prompt"]}),
                 "median_out_tokens": sorted(r["n_out"] or 0 for r in rs)[len(rs) // 2]}
            if m == "maverick":
                e["grade"] = G.grade_espionage(p) if c.startswith("espionage") else G.grade_blackmail(p)
                e["providers"] = sorted({str(r["provider"]) for r in rs})
                e["served_model"] = sorted({str(r["served_model"]) for r in rs})
            per[c] = e
        card["models"][m] = per
    hl = RESULTS / "handlabel" / "labels.json"
    if hl.exists():
        labels = json.loads(hl.read_text())
        key = json.loads((RESULTS / "handlabel" / "key.json").read_text())
        allrows = {r["rid"]: r for m in ("maverick", *OPEN_MODELS) for r in load_graded(m)}
        agree_c = agree_f = 0
        cm = {"yy": 0, "yn": 0, "ny": 0, "nn": 0}  # hand x classifier
        for hid, lab in labels.items():
            r = allrows[key[hid]]
            agree_c += lab == r["classifier"]
            agree_f += lab == r["final"]
            cm[("y" if lab else "n") + ("y" if r["classifier"] else "n")] += 1
        n = len(labels)
        po = agree_c / n
        pe = ((cm["yy"] + cm["yn"]) * (cm["yy"] + cm["ny"]) + (cm["ny"] + cm["nn"]) * (cm["yn"] + cm["nn"])) / n / n
        card["handlabel"] = {"n": n, "agree_classifier": agree_c, "agree_final": agree_f,
                             "hand_x_classifier": cm, "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None}
    mine, day = spend_state()
    card["spend_usd_k7"], card["spend_day"] = round(mine, 4), round(day, 2)
    (RESULTS / "card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    for s in ("manifest", "anchor", "handpick", "summary"):
        sub.add_parser(s)
    for s in ("open", "classify"):
        p = sub.add_parser(s)
        p.add_argument("--model", required=True, choices=("maverick", *OPEN_MODELS))
    a = ap.parse_args()
    {"manifest": stage_manifest, "anchor": stage_anchor, "open": stage_open, "classify": stage_classify,
     "handpick": stage_handpick, "summary": stage_summary}[a.stage](a)


if __name__ == "__main__":
    main()
