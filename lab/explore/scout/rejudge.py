"""A second judge's reliability check (1 Nov 2026). Its pure parts (the stratified sample, the agreement with its
bootstrap interval, the table) moved to anatta.judge.rejudge in clean-lab step 4a (Claude Opus 5.5, 3 Nov 2026) and
are re-exported here; run_rejudge and expected_for_spec stay, since they read scout's sweep folders, report and spec.
Claude wrote this file.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

from anatta.judge.rejudge import (  # noqa: F401
    HEAD,
    MIN_LINE,
    REPS,
    SEED,
    _fmt,
    _jkey,
    _rkey,
    agreement,
    family_tokens,
    kappa,
    out_name,
    print_table,
    provider_routing,
    render_md,
    sample_records,
    shared_family,
    table_rows,
)

from . import library as lib
from .schema import read_jsonl, utc_now, write_jsonl


def run_rejudge(sweep_dir, *, judge_model: str, paths, provider: str | None = None, fraction: float = 0.1,
                seed: int = 0, max_tokens: int = 400, cap: float = 10.0, dry_run: bool = False, library=None,
                rubrics=None, meta=None, transport=None, reps: int = REPS, log=print) -> tuple[int, dict]:
    """The whole `rejudge` command; returns (exit code, summary). library/rubrics/meta/transport are injected in tests.
    Exit codes as scout's: 0 done, 3 refused or stopped at a spend cap, 4 some calls failed."""
    from .cache import Cache, cache_key
    from .client import OpenRouterClient, model_meta
    from .judge import judge_messages, judge_params, judge_records
    from .report import current_judgments
    from .runner import Engine, bound_cost
    from .spec import JUDGE_DEFAULTS
    from .spend import CapReached, SpendGuard

    sd = Path(sweep_dir)
    man = json.loads((sd / "run.json").read_text()) if (sd / "run.json").exists() else {}
    records = read_jsonl(sd / "records.jsonl")
    main_all = [j for j in current_judgments(records, read_jsonl(sd / "judgments.jsonl"))
                if j.get("family") != lib.HARM_GUARD]
    main = {_jkey(j): j for j in main_all}
    fams = sorted({r["family"] for r in records if r["family"] != lib.HARM_GUARD})
    library = library or lib.load_library(fams)
    rubrics = rubrics or lib.load_rubrics()
    subjects = sorted({r["model"] for r in records})
    main_judges = sorted({j["judge_model"] for j in main_all}) or [(man.get("judge") or {}).get("model", "")]
    if judge_model in subjects or judge_model in main_judges:
        raise SystemExit(f"refused: {judge_model} is {'a subject' if judge_model in subjects else 'the main judge'} "
                         f"of this run; the second judge must be another model")
    clash = shared_family(judge_model, subjects + main_judges)
    if clash:
        raise SystemExit(f"refused: {judge_model} shares a model family with {', '.join(clash)}; the second judge must "
                         f"come from a family that is neither the main judge's nor any subject's")

    sample, strata = sample_records(records, main, fraction=fraction, seed=seed, library=library)
    assert not any(r["family"] == lib.HARM_GUARD for r in sample)
    eligible = sum(v[1] for v in strata.values())
    # the second judge reads today's rubrics; where one changed after the main judge ran, the prompts differ
    stale = sorted({r["family"] for r in sample if main[_rkey(r)]["prompt_sha256"] != rubrics[r["family"]]["sha256"]})
    if stale:
        log(f"[scout] warning: the rubric changed since the main judge ran, for {', '.join(stale)}; there the two "
            f"judges do not see the same prompt")
    routing = provider_routing(provider)
    jcfg = {**JUDGE_DEFAULTS, "model": judge_model, "provider": routing, "max_tokens": int(max_tokens)}
    meta = meta or model_meta({judge_model: routing}, cache_dir=paths.cache)
    jmeta = meta[judge_model]
    params = judge_params(jcfg, jmeta)
    cache = Cache(paths.cache)
    bound = 0.0
    for r in sample:  # what the uncached calls could cost: the full max_tokens out, the dearest allowed endpoint
        jm = judge_messages(rubrics[r["family"]], library[r["family"]]["by_id"][r["item_id"]], r["response"])
        if not cache.has(cache_key(judge_model, routing, jm, params, r["sample"], None)):
            bound += bound_cost(jm, params["max_tokens"], jmeta)
    name = out_name(judge_model, fraction, seed)
    out_dir = sd / "rejudge"
    log(f"[scout] rejudge {sd.name}: {len(sample)} of {eligible} judged records ({fraction:g} of each family x "
        f"condition, seed {seed}) to {judge_model}; bound on the uncached calls ${bound:.4f}")
    summary = {"sweep": man.get("sweep") or sd.name, "ts": utc_now(), "judge": judge_model, "provider": routing,
               "main_judges": main_judges, "fraction": fraction, "seed": seed, "sampled": len(sample),
               "eligible": eligible, "bound_usd": round(bound, 4), "strata": strata, "stale_rubrics": stale}
    if dry_run:
        return 0, summary
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = SpendGuard(paths.spend, sweep_cap=cap, sweep_log=out_dir / f"{name}.spend.jsonl", tool="scout",
                       label=f"{sd.parent.name}/{sd.name}/rejudge")
    try:
        guard.check_start(bound)
    except CapReached as e:
        log(f"[scout] {e} (raise --cap, or lower --fraction)")
        return 3, summary

    async def go():
        async with OpenRouterClient(transport=transport) as client:
            engine = Engine(client, cache, guard, meta)
            return await judge_records(sample, rubrics=rubrics, library=library, engine=engine, jcfg=jcfg, jmeta=jmeta,
                                       log=log)

    res = asyncio.run(go())
    second = res["judgments"]
    write_jsonl(out_dir / f"{name}.jsonl", second)
    pairs = []
    for j2 in second:
        j1 = main.get(_jkey(j2))
        if j1 and j1.get("primary") is not None and j2.get("primary") is not None:
            pairs.append((j2["family"], j2["item_id"], bool(j1["primary"]), bool(j2["primary"])))
    summary.update({"judged": len(second), "unparsed": sum(not j["parse_ok"] for j in second),
                    "errors": len(res["errors"]), "stopped": res["stopped"], "paid_usd": guard.paid_this_run,
                    "prompt_sha256": sorted({j["prompt_sha256"][:12] for j in second}),
                    "agreement": agreement(pairs, library, reps=reps, seed=SEED)})
    (out_dir / f"{name}.md").write_text(render_md(summary), encoding="utf-8")
    print_table(summary, log)
    log(f"[scout] wrote {out_dir / (name + '.md')} and {name}.jsonl; paid ${guard.paid_this_run:.4f}; today "
        f"${guard.today():.4f} of ${guard.daily_cap:.2f}")
    return (3 if res["stopped"] else 4 if res["errors"] else 0), summary


def expected_for_spec(spec: dict, *, fraction: float, judge_model: str, judge_meta: dict, meas: dict) -> dict:
    """What rejudge would cost on a finished run of this spec: the stratified sample's size from the spec's jobs (no
    harm_guard), times the cost per call measured for this judge in earlier rejudge files, else the main judge's
    measured tokens per family at this judge's price (the dearest allowed endpoint)."""
    from .runner import build_jobs

    strata: dict = {}
    for job in build_jobs(spec):
        if job.family != lib.HARM_GUARD:
            key = (job.family, job.variant["intervention"]["name"])
            strata[key] = strata.get(key, 0) + 1
    main_judge = spec["judge"]["model"]
    out = {"calls": 0, "usd": 0.0, "by_family": {}}
    for (fam, _), n in sorted(strata.items()):
        k = min(n, max(1, math.floor(fraction * n + 0.5)))
        own = (meas.get("rejudge") or {}).get((fam, judge_model))
        if own:
            per, source = own["usd"], f"measured for {judge_model} (n {own['n']})"
        else:
            m = meas["judge"].get((fam, main_judge)) or meas.get("judge_all")
            per = (m["prompt"] * judge_meta["in"] + m["completion"] * judge_meta["out"]) if m else 0.0
            source = f"{main_judge}'s tokens at {judge_model}'s price"
        f = out["by_family"].setdefault(fam, {"calls": 0, "usd": 0.0, "source": source})
        f["calls"] += k
        f["usd"] += k * per
        out["calls"] += k
        out["usd"] += k * per
    out["usd"] = round(out["usd"], 4)
    for f in out["by_family"].values():
        f["usd"] = round(f["usd"], 4)
    return out
