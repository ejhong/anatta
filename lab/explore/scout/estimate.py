"""The expected cost of a spec from measured calls: the mean cost per call by (model, family, mode) in earlier sweeps
(subject, practice and judge calls), times the spec's job counts. Where a cell has no measurement it falls back to
the model's mean over families and modes, then to the price-times-tokens bound. A practice reply is shared by both
framings of a sample, so it is counted once. Judge costs are measured per family (harm_guard has its own judge).
Claude wrote this file. Private."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .runner import bound_cost, build_jobs, sampling_params
from .schema import read_jsonl


def measurements(sweep_dirs: list) -> dict:
    """Mean original cost and tokens per call from records (subject and practice), judgments (judge) and second-judge
    files (rejudge/*.jsonl, kept apart under "rejudge")."""
    subj, prac, judge, second = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for sd in sweep_dirs:
        for f in sorted((Path(sd) / "rejudge").glob("*.jsonl")):
            if f.name.endswith(".spend.jsonl"):
                continue
            for j in read_jsonl(f):
                u = j.get("usage") or {}
                if u:
                    second[(j.get("family"), j.get("judge_model"))].append(
                        (u.get("cost") or 0.0, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
        for r in read_jsonl(Path(sd) / "records.jsonl"):
            mode = r["intervention"]["params"].get("mode", "none")
            u = r["usage"]
            subj[(r["model"], r["family"], mode)].append((u.get("cost") or 0.0, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
            if r.get("practice"):
                pu = r["practice"]["usage"]
                prac[(r["model"], r["family"])].append((pu.get("cost") or 0.0, pu.get("prompt_tokens", 0), pu.get("completion_tokens", 0)))
        for j in read_jsonl(Path(sd) / "judgments.jsonl"):
            u = j.get("usage") or {}
            if u:
                judge[(j.get("family"), j.get("judge_model"))].append(
                    (u.get("cost") or 0.0, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))

    def mean(rows):
        n = len(rows)
        return {"usd": sum(r[0] for r in rows) / n, "prompt": sum(r[1] for r in rows) / n,
                "completion": sum(r[2] for r in rows) / n, "n": n}

    return {"subject": {k: mean(v) for k, v in subj.items()}, "practice": {k: mean(v) for k, v in prac.items()},
            "judge": {k: mean(v) for k, v in judge.items()}, "rejudge": {k: mean(v) for k, v in second.items()},
            "judge_by_model": {jm: mean([x for (f, mm), v in judge.items() if mm == jm for x in v])
                               for jm in {k[1] for k in judge}},
            "subject_by_model": {m: mean([x for (mm, _, _), v in subj.items() if mm == m for x in v])
                                 for m in {k[0] for k in subj}},
            "practice_by_model": {m: mean([x for (mm, _), v in prac.items() if mm == m for x in v])
                                  for m in {k[0] for k in prac}},
            "judge_all": mean([x for (f, mm), v in judge.items() if f != "harm_guard" for x in v]) if judge else None}


def expected(spec: dict, meas: dict, meta: dict) -> dict:
    """Expected spend of the whole spec, ignoring the cache: {"subject_usd", "practice_usd", "judge_usd",
    "total_usd", "calls", "by_model": {model: usd}, "fallbacks": n}."""
    from .spec import judge_for

    out = {"subject_usd": 0.0, "practice_usd": 0.0, "judge_usd": 0.0, "calls": 0, "by_model": defaultdict(float),
           "fallbacks": 0, "judge_fallbacks": 0}
    practice_seen = set()
    for job in build_jobs(spec):
        mid, fam, mode = job.model["id"], job.family, job.variant["mode"]
        s = meas["subject"].get((mid, fam, mode)) or meas["subject"].get((mid, fam, "practice" if mode == "practice" else "none"))
        if s is None:
            s = meas["subject_by_model"].get(mid)
        if s is None:
            out["fallbacks"] += 1
            params = sampling_params(spec["temperature"], job.model["max_tokens"], job.model["params"], meta.get(mid))
            cost = bound_cost(job.item["messages"], params["max_tokens"], meta[mid])
        else:
            cost = s["usd"]
        out["subject_usd"] += cost
        out["by_model"][mid] += cost
        out["calls"] += 1
        pk = (mid, fam, job.item["id"], job.sample, job.variant["intervention"]["name"])
        if mode == "practice" and pk not in practice_seen:
            practice_seen.add(pk)
            p = meas["practice"].get((mid, fam)) or meas["practice_by_model"].get(mid)
            pc = p["usd"] if p else cost
            out["practice_usd"] += pc
            out["by_model"][mid] += pc
            out["calls"] += 1
        jmodel = judge_for(spec, fam)["model"]
        j = meas["judge"].get((fam, jmodel)) or meas["judge_by_model"].get(jmodel)
        if j is None:
            j = meas["judge_all"] if fam != "harm_guard" else None
            out["judge_fallbacks"] += 1
        if j:
            out["judge_usd"] += j["usd"]
            out["calls"] += 1
    out["total_usd"] = out["subject_usd"] + out["practice_usd"] + out["judge_usd"]
    out["by_model"] = {k: round(v, 4) for k, v in out["by_model"].items()}
    for k in ("subject_usd", "practice_usd", "judge_usd", "total_usd"):
        out[k] = round(out[k], 4)
    return out
