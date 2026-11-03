"""Running a sweep: every (model, condition, item, sample) as one job, practice calls first for practice-mode
conditions, each call through the cache and the spend guard, one record per sample. Claude wrote this file. Private.

harm_guard is the exception (1 Nov 2026): its calls bypass the cache, its judge grades the reply in memory within the
same job, and its record keeps sha256s, lengths and scores, never the Goal, the reply, the practice reply or a
reasoning trace (schema.py). A failed harm_guard call is logged by status and type, never with a body. A rerun pays
for harm_guard again.

A sweep directory, <results>/<UTC date>/<sweep>/, holds:
  records.jsonl      one record per sample (schema.py), rewritten whole at the end of each run
  records.partial.jsonl  the same records as they land (a crash loses nothing; the cache keeps the replies)
  errors.jsonl       calls that failed after their retries
  judgments.jsonl    the judge's labels (judge.py)
  spend.jsonl        one line per paid call (spend.py)
  run.json           the run's manifest: spec, hashes of the library, conditions and rubrics, prices, counts, spend
  report.md, report.html, transcripts/   the report (report.py)
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# The cached, capped call moved to anatta.judge.engine in clean-lab step 4a; imported here under the old names.
from anatta.judge.engine import CapStop, Engine, bound_cost, harm_error, paid_cost, sampling_params  # noqa: F401

from . import harm_guard as hg
from . import library as lib
from .cache import Cache, cache_key  # noqa: F401
from .client import ApiError
from .schema import append_jsonl, canonical, make_record, read_jsonl, sha256_text, utc_now, write_json, write_jsonl
from .spend import SpendGuard  # noqa: F401


@dataclass
class Job:
    model: dict
    variant: dict
    family: str
    item: dict
    sample: int
    framing: str | None = None

    @property
    def ident(self) -> str:
        return (f"{self.model['id']}|{self.variant['intervention']['name']}|{self.item['id']}|s{self.sample}"
                + (f"|{self.framing}" if self.framing else ""))


def new_run_id() -> str:
    return f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"


def framings_for(spec: dict, family: str) -> list:
    fr = spec.get("framing")
    return list(fr["levels"]) if fr and family in fr["families"] else [None]


def build_jobs(spec: dict) -> list[Job]:
    """All jobs, ordered so that a sweep stopped early still covers every cell evenly: sample, then item position,
    then framing, then condition, then model. harm_guard jobs exist only for open-weight models."""
    jobs = []
    lib_ = spec["library"]
    for s in range(spec["samples"]):
        for fam in spec["families"]:
            items = [lib_[fam]["by_id"][i] for i in spec["items"][fam]]
            models = [m for m in spec["models"] if fam != lib.HARM_GUARD or hg.is_open_weight(m["id"])]
            for it in items:
                for fr in framings_for(spec, fam):
                    for v in spec["variants"]:
                        for m in models:
                            jobs.append(Job(m, v, fam, it, s, fr))
    return jobs


def scenario_messages(job: Job, spec: dict) -> list[dict]:
    """The item's messages with the job's audience cue, if any (the practice call never carries it)."""
    fr = spec.get("framing")
    line = fr["lines"][job.framing] if (fr and job.framing) else None
    return lib.apply_framing(job.item["messages"], line)


def practice_salt(job: Job, attempt: int = 0) -> str:
    """A practice reply per (item, sample), even when the practice chat is the same for many items."""
    return f"practice:{job.family}/{job.item['id']}" + (f":{attempt}" if attempt else "")


def record_usage(resp: dict) -> dict:
    u = dict(resp["usage"])
    if u.get("cost") is None:
        u["cost"] = resp.get("cost_usd")
    return u


async def run_job(job: Job, spec: dict, engine: Engine, run_id: str, rubrics: dict | None = None) -> dict:
    """One sample: the practice call if the condition is a practice, then the scenario call; returns its record.
    A harm_guard sample goes to run_harm_job (rubrics: the loaded rubrics, for its judge)."""
    if job.family == lib.HARM_GUARD:
        return await run_harm_job(job, spec, engine, run_id, (rubrics or lib.load_rubrics())[lib.HARM_GUARD])
    conds = spec["conditions"]
    cond = conds["conditions"][job.variant["condition"]]
    wrapper = conds["practice_wrapper"]
    mode = job.variant["mode"]
    meta = engine.meta.get(job.model["id"])
    params = sampling_params(spec["temperature"], job.model["max_tokens"], job.model["params"], meta)
    practice, paid = None, 0.0
    if mode == "practice":
        pm = lib.practice_messages(job.item["messages"], cond, wrapper)
        for attempt in range(2):
            presp, pcached, ppaid = await engine.call(model=job.model["id"], messages=pm, params=params,
                                                      provider=job.model["provider"], sample=job.sample,
                                                      salt=practice_salt(job, attempt), kind="practice",
                                                      ident=job.ident + "|practice")
            paid += ppaid
            if presp["text"].strip():
                break
        else:
            raise ApiError("the practice reply was empty twice", scope="call")
        practice = {"response": presp["text"], "usage": record_usage(presp), "cost_usd": round(ppaid, 8),
                    "cached": pcached, "finish_reason": presp.get("finish_reason")}
    msgs = lib.subject_messages(scenario_messages(job, spec), cond, mode, wrapper,
                                practice["response"] if practice else None)
    resp, cached, spaid = await engine.call(model=job.model["id"], messages=msgs, params=params,
                                            provider=job.model["provider"], sample=job.sample, kind="subject",
                                            ident=job.ident)
    paid += spaid
    if not resp["text"].strip():
        # An empty reply is a provider artifact, not a behavior (in the first smoke, Kimi K2 on Novita sometimes spent
        # its whole budget on hidden reasoning tokens). Ask once more; a reply still empty is kept but never judged.
        resp, c2, p2 = await engine.call(model=job.model["id"], messages=msgs, params=params,
                                         provider=job.model["provider"], sample=job.sample, salt="empty:1",
                                         kind="subject", ident=job.ident + "|empty-retry")
        paid += p2
        cached = cached and c2
    provider = resp.get("provider") or ((job.model["provider"] or {}).get("order") or [None])[0]
    return make_record(sweep=spec["sweep"], run_id=run_id, source="openrouter", model=job.model["id"],
                       provider=provider, intervention=job.variant["intervention"], family=job.family,
                       item_id=job.item["id"], sample=job.sample, messages=msgs, response=resp["text"],
                       usage=record_usage(resp), cost_usd=paid, finish_reason=resp.get("finish_reason"),
                       reasoning=resp.get("reasoning"), cached=cached, practice=practice, params=params,
                       latency_s=resp.get("latency_s"), **({"framing": job.framing} if job.framing else {}))


async def run_harm_job(job: Job, spec: dict, engine: Engine, run_id: str, rubric: dict) -> dict:
    """One harm_guard sample, judged in memory: the practice call (if any), the scenario call and the harm judge's
    call, none of them cached and all of them private (no body in an error). The record keeps sha256s, lengths, the
    reply's finish reason and the judge's scores, never the Goal, the reply, the practice reply or a reasoning trace;
    the texts live only in this function's locals."""
    from .judge import judge_one  # judge imports this module

    hg.require_open_weight(job.model["id"], "subject")  # both checks come before anything is sent
    jcfg = spec["harm_judge"]
    hg.require_open_weight(jcfg["model"], "harm_guard's judge")
    conds = spec["conditions"]
    cond = conds["conditions"][job.variant["condition"]]
    wrapper = conds["practice_wrapper"]
    mode = job.variant["mode"]
    params = sampling_params(spec["temperature"], job.model["max_tokens"], job.model["params"],
                             engine.meta.get(job.model["id"]))
    call = dict(model=job.model["id"], params=params, provider=job.model["provider"], sample=job.sample, private=True)
    practice, practice_text, paid = None, None, 0.0
    if mode == "practice":
        pm = lib.practice_messages(job.item["messages"], cond, wrapper)
        for attempt in range(2):
            presp, _, ppaid = await engine.call(messages=pm, salt=practice_salt(job, attempt), kind="practice",
                                                ident=job.ident + "|practice", **call)
            paid += ppaid
            if presp["text"].strip():
                break
        else:
            raise ApiError("the practice reply was empty twice", scope="call")
        practice_text = presp["text"]
        practice = {"sha256": sha256_text(practice_text), "chars": len(practice_text.strip()),
                    "usage": record_usage(presp), "cost_usd": round(ppaid, 8), "cached": False,
                    "finish_reason": presp.get("finish_reason")}
    msgs = lib.subject_messages(scenario_messages(job, spec), cond, mode, wrapper, practice_text)
    resp, _, spaid = await engine.call(messages=msgs, kind="subject", ident=job.ident, **call)
    paid += spaid
    if not resp["text"].strip():  # as in run_job: ask once more; a reply still empty is kept but never judged
        resp, _, p2 = await engine.call(messages=msgs, salt="empty:1", kind="subject", ident=job.ident + "|empty-retry",
                                        **call)
        paid += p2
    goal = next(m["content"] for m in reversed(job.item["messages"]) if m["role"] == "user")
    block = {"goal_sha256": sha256_text(goal), "goal_chars": len(goal), "messages_sha256": sha256_text(canonical(msgs)),
             "response_sha256": sha256_text(resp["text"]), "response_chars": len(resp["text"].strip()),
             "reasoning_chars": len(resp.get("reasoning") or ""), "score": None, "refused": None, "convincing": None,
             "specific": None, "judge": None}
    if resp["text"].strip():
        in_memory = {"run_id": run_id, "sweep": spec["sweep"], "item_id": job.item["id"], "family": job.family,
                     "sample": job.sample, "model": job.model["id"], "intervention": job.variant["intervention"],
                     "response": resp["text"]}
        j = await judge_one(in_memory, item=job.item, rubric=rubric, engine=engine, jcfg=jcfg,
                            jmeta=engine.meta.get(jcfg["model"]))
        label = j["label"] or {}
        block.update(score=j["primary"], refused=label.get("refused"), convincing=label.get("convincing"),
                     specific=label.get("specific"),
                     judge={"model": j["judge_model"], "prompt_sha256": j["prompt_sha256"], "parse_ok": j["parse_ok"],
                            "why": j.get("raw"), "finish_reason": j.get("finish_reason"), "usage": j["usage"],
                            "cost_usd": j["cost_usd"], "ts": j["ts"]})
    provider = resp.get("provider") or ((job.model["provider"] or {}).get("order") or [None])[0]
    return make_record(sweep=spec["sweep"], run_id=run_id, source="openrouter", model=job.model["id"],
                       provider=provider, intervention=job.variant["intervention"], family=job.family,
                       item_id=job.item["id"], sample=job.sample, usage=record_usage(resp), cost_usd=paid,
                       finish_reason=resp.get("finish_reason"), cached=False, practice=practice, params=params,
                       latency_s=resp.get("latency_s"), harm_guard=block)


def order_key(spec: dict):
    vorder = {v["intervention"]["name"]: i for i, v in enumerate(spec["variants"])}
    morder = {m["id"]: i for i, m in enumerate(spec["models"])}
    forder = {f: i for i, f in enumerate(spec["families"])}
    return lambda r: (morder.get(r["model"], 99), vorder.get(r["intervention"]["name"], 99),
                      forder.get(r["family"], 99), r["item_id"], r["sample"], r.get("framing") or "")


async def run_subjects(spec: dict, *, engine: Engine, run_id: str, sweep_dir: Path, log=print) -> dict:
    """Every job of the spec. Returns {"records", "errors", "status", "stopped"}; writes records.partial.jsonl as
    records land and errors.jsonl at the end."""
    jobs = build_jobs(spec)
    rubrics = lib.load_rubrics() if any(j.family == lib.HARM_GUARD for j in jobs) else None
    partial = sweep_dir / "records.partial.jsonl"
    partial.write_text("")
    records, errors = [], []
    state = {"stop": None, "fatal": False, "done": 0, "dead": {}}
    t0 = time.monotonic()
    last = [t0]

    def failed(job: Job, msg: str, scope: str) -> None:
        errors.append({"ts": utc_now(), "job": job.ident, "model": job.model["id"], "family": job.family,
                       "item_id": job.item["id"], "sample": job.sample, "framing": job.framing,
                       "intervention": job.variant["intervention"]["name"], "error": msg[:500], "scope": scope})

    async def one(job: Job):
        if state["stop"] or job.model["id"] in state["dead"]:
            if job.model["id"] in state["dead"]:
                errors.append({"ts": utc_now(), "job": job.ident, "model": job.model["id"], "family": job.family,
                               "item_id": job.item["id"], "sample": job.sample,
                               "intervention": job.variant["intervention"]["name"],
                               "error": f"skipped: {state['dead'][job.model['id']]}"[:500], "scope": "model"})
            return
        harm = job.family == lib.HARM_GUARD
        try:
            rec = await run_job(job, spec, engine, run_id, rubrics)
            records.append(rec)
            append_jsonl(partial, rec)
        except CapStop as e:
            state["stop"] = str(e)
        except ApiError as e:
            # a harm_guard error is stored and may be printed, so it says only what failed, never with a body
            msg = hg.redact(harm_error(e), [m["content"] for m in job.item["messages"]]) if harm else str(e)
            failed(job, msg, e.scope)
            if e.scope == "run":
                state["stop"], state["fatal"] = f"fatal: {msg[:300]}", True
            elif e.scope == "model":
                state["dead"][job.model["id"]] = msg[:300]
        except Exception as e:  # noqa: BLE001 - harm_guard only: any other failure is logged by its type alone
            if not harm:
                raise
            failed(job, harm_error(e), "call")
        state["done"] += 1
        now = time.monotonic()
        if now - last[0] > 15 or state["done"] == len(jobs):
            last[0] = now
            log(f"[scout] {state['done']}/{len(jobs)} jobs, {engine.paid_calls} paid calls, {engine.cached_calls} "
                f"cached, {len(errors)} errors, ${engine.guard.paid_this_run:.4f} paid, {now - t0:.0f}s")

    await asyncio.gather(*(one(j) for j in jobs))
    records.sort(key=order_key(spec))
    write_jsonl(sweep_dir / "records.jsonl", records)
    write_jsonl(sweep_dir / "errors.jsonl", errors)
    status = "fatal" if state["fatal"] else "stopped_at_cap" if state["stop"] else "errors" if errors else "complete"
    return {"records": records, "errors": errors, "status": status, "stopped": state["stop"], "jobs": len(jobs)}


# ------------------------------------------------------------------------------------------------- estimates


def estimate(spec: dict, cache: Cache, meta: dict, judge_meta: dict | None = None, rubrics: dict | None = None) -> dict:
    """The bound on what the uncached calls of the spec could cost (the full max_tokens out, the dearest endpoint),
    per model, with the judge's calls if judge_meta and rubrics are given. A practice-mode call whose practice reply
    is not cached is bounded with a max_tokens-long reply in its context. harm_guard's calls never use the cache, so
    they are always bounded."""
    from .judge import judge_messages, judge_params  # local: judge imports runner's helpers
    from .spec import judge_for

    out = {"subject_usd": 0.0, "practice_usd": 0.0, "judge_usd": 0.0, "calls_uncached": 0, "calls_cached": 0,
           "by_model": {}}
    conds = spec["conditions"]
    wrapper = conds["practice_wrapper"]
    practice_bounded = set()  # both framings of a sample share one practice call
    for job in build_jobs(spec):
        jcfg = judge_for(spec, job.family)
        harm = job.family == lib.HARM_GUARD
        mid = job.model["id"]
        m = out["by_model"].setdefault(mid, {"usd": 0.0, "calls": 0})
        price = meta[mid]
        params = sampling_params(spec["temperature"], job.model["max_tokens"], job.model["params"], meta.get(mid))
        cond = conds["conditions"][job.variant["condition"]]
        reply = None
        if job.variant["mode"] == "practice":
            pm = lib.practice_messages(job.item["messages"], cond, wrapper)
            pkey = cache_key(mid, job.model["provider"], pm, params, job.sample, practice_salt(job))
            hit = cache.get(pkey) if (not harm and cache.has(pkey)) else None
            if hit and hit.get("text", "").strip():
                reply = hit["text"]
                out["calls_cached"] += 1
            elif pkey not in practice_bounded:
                practice_bounded.add(pkey)
                b = bound_cost(pm, params["max_tokens"], price)
                out["practice_usd"] += b
                m["usd"] += b
                m["calls"] += 1
                out["calls_uncached"] += 1
            msgs = lib.subject_messages(scenario_messages(job, spec), cond, "practice", wrapper,
                                        reply if reply is not None else "x" * (4 * params["max_tokens"]))
        else:
            msgs = lib.subject_messages(scenario_messages(job, spec), cond, job.variant["mode"], wrapper)
        key = cache_key(mid, job.model["provider"], msgs, params, job.sample, None)
        resp = (cache.get(key) if not harm and (reply is not None or job.variant["mode"] != "practice")
                and cache.has(key) else None)
        if resp is None:
            b = bound_cost(msgs, params["max_tokens"], price)
            out["subject_usd"] += b
            m["usd"] += b
            m["calls"] += 1
            out["calls_uncached"] += 1
        else:
            out["calls_cached"] += 1
        if judge_meta and rubrics and jcfg["model"] in judge_meta:
            jm = judge_messages(rubrics[job.family], job.item,
                                resp["text"] if resp else "x" * (4 * params["max_tokens"]))
            jp = judge_params(jcfg, judge_meta.get(jcfg["model"]))
            jkey = cache_key(jcfg["model"], jcfg["provider"], jm, jp, job.sample, None)
            if resp is None or harm or not cache.has(jkey):
                out["judge_usd"] += bound_cost(jm, jp["max_tokens"], judge_meta[jcfg["model"]])
    out["total_usd"] = out["subject_usd"] + out["practice_usd"] + out["judge_usd"]
    for k in ("subject_usd", "practice_usd", "judge_usd", "total_usd"):
        out[k] = round(out[k], 4)
    for m in out["by_model"].values():
        m["usd"] = round(m["usd"], 4)
    return out


def log_variants(spec: dict, paths, run_id: str) -> dict:
    """Append this run's interventions to <results>/variants.jsonl (every variant tried, so the forking paths stay
    visible) and count them: this sweep's, and the distinct ones the explore track has tried to date."""
    for v in spec["variants"]:
        iv = v["intervention"]
        append_jsonl(paths.variants, {"ts": utc_now(), "tool": "scout", "sweep": spec["sweep"], "run_id": run_id,
                                      "kind": iv["kind"], "name": iv["name"], "sha256": iv["params"].get("sha256")})
    return variant_counts(spec, paths)


def variant_counts(spec: dict, paths) -> dict:
    rows = read_jsonl(paths.variants)
    distinct = {(r["kind"], r["name"], r.get("sha256")) for r in rows}
    mine = {(v["intervention"]["kind"], v["intervention"]["name"], v["intervention"]["params"].get("sha256"))
            for v in spec["variants"]}
    return {"this_sweep": len(mine), "track_distinct": len(distinct), "track_sweep_runs": len({r["run_id"] for r in rows}),
            "models": len(spec["models"]), "families": len(spec["families"]),
            "cells": len(mine) * len(spec["models"]) * len(spec["families"])}


def manifest(spec: dict, *, run_id: str, date: str, meta: dict, est: dict | None, started: str, result: dict | None,
             guard: SpendGuard, variants: dict, judge_summary: dict | None = None) -> dict:
    from .paths import CONDITIONS_FILE, HELDOUT_FILE, RUBRICS_FILE
    return {
        "sweep": spec["sweep"], "run_id": run_id, "date": date, "started": started, "finished": utc_now(),
        "status": result["status"] if result else "not run", "stopped": result.get("stopped") if result else None,
        "spec_path": spec["source_path"], "spec_sha256": sha256_text(spec["source_text"]), "spec_text": spec["source_text"],
        "library": {f: spec["library"][f]["sha256"] for f in spec["families"]},
        "conditions_sha256": lib.file_sha256(CONDITIONS_FILE) if Path(CONDITIONS_FILE).exists() else None,
        "rubrics_sha256": lib.file_sha256(RUBRICS_FILE) if Path(RUBRICS_FILE).exists() else None,
        "heldout_sha256": lib.file_sha256(HELDOUT_FILE) if Path(HELDOUT_FILE).exists() else None,
        "models": [{"id": m["id"], "provider": m["provider"], "params": m["params"], "max_tokens": m["max_tokens"],
                    "price_usd_per_mtok": {"in": round(meta[m["id"]]["in"] * 1e6, 4), "out": round(meta[m["id"]]["out"] * 1e6, 4)},
                    "endpoints": meta[m["id"]].get("endpoints")} for m in spec["models"]],
        "variants": [v["intervention"] for v in spec["variants"]], "variant_counts": variants,
        "missing_controls": spec["missing_controls"], "allow_missing_controls": spec["allow_missing_controls"],
        "items": spec["items"], "samples": spec["samples"], "temperature": spec["temperature"], "judge": spec["judge"],
        "harm_judge": spec["harm_judge"] if lib.HARM_GUARD in spec["families"] else None,
        "framing": spec.get("framing"), "skipped": spec.get("skipped") or [],
        "harm_guard": ({"csv": spec["library"][lib.HARM_GUARD]["file"], "sha256": spec["library"][lib.HARM_GUARD]["sha256"],
                        "revision": hg.JBB_REVISION, "models": spec.get("harm_models")}
                       if lib.HARM_GUARD in spec["families"] else None),
        "estimate_bound": est, "jobs": result.get("jobs") if result else None,
        "records": len(result["records"]) if result else 0, "errors": len(result["errors"]) if result else 0,
        "spend": {"this_run_usd": guard.paid_this_run, "sweep_total_usd": round(guard.sweep_spent, 8),
                  "sweep_cap_usd": guard.sweep_cap, "today_usd": round(guard.today(), 8), "daily_cap_usd": guard.daily_cap},
        "judge_summary": judge_summary,
    }


def write_manifest(sweep_dir: Path, m: dict) -> None:
    write_json(sweep_dir / "run.json", m)
