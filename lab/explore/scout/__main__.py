"""scout: behavior sweeps through OpenRouter, judged and reported. Claude wrote this file. Private (lab/explore).

Run from lab/explore with the lab's Python and OPENROUTER_API_KEY in the environment:
  PY=/Users/eugene/prj/align/anatta/.venv/bin/python
  $PY -m scout run scout/specs/smoke.yaml        # subjects, then the judge, then the report (--dry-run: estimate only)
  $PY -m scout run scout/specs/first-sweep.yaml --measured 2026-11-01/smoke 2026-11-01/smoke-new 2026-11-01/smoke-harm
  $PY -m scout estimate scout/specs/first-sweep.yaml --measured 2026-11-01/smoke
  $PY -m scout judge 2026-11-01/smoke           # (re)judge a sweep; cached verdicts are free
  $PY -m scout rejudge 2026-11-01/smoke --judge mistralai/mistral-medium-3.1 --provider mistral   # a second judge's
                                                # agreement (kappa) on a stratified 10% sample; changes no judgment
  $PY -m scout report 2026-11-01/smoke          # report.md + report.html (+ transcripts/) in the sweep directory
  $PY -m scout label 2026-11-01/smoke --family deference --n 10 --labeler eugene
  $PY -m scout agree                            # judge-vs-hand agreement per family
  $PY -m scout sighting 2026-11-01/smoke --note "..."   # a stub in SIGHTINGS.md, time from the clock
  $PY -m scout check                            # the library's split and the held-out file's recorded sha256
A sweep directory can be given as a path or as <UTC date>/<sweep> under the results root.

Exit codes: 0 done; 2 a bad spec or a refused one (held-out); 3 stopped or refused at a spend cap; 4 some calls
failed (a rerun retries them; cached calls are free); 5 a fatal API error (bad key, no credits).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "scout"

from scout import library as lib  # noqa: E402
from scout import paths as P  # noqa: E402
from scout.cache import Cache  # noqa: E402
from scout.client import OpenRouterClient, model_meta  # noqa: E402
from scout.schema import read_jsonl, utc_date, utc_now, write_json, write_jsonl  # noqa: E402
from scout.spend import CapReached, SpendGuard, spent_today  # noqa: E402


def _meta_for(spec, paths):
    models = {m["id"]: m["provider"] for m in spec["models"]}
    meta = model_meta(models, cache_dir=paths.cache)
    judges = {spec["judge"]["model"]: spec["judge"].get("provider")}
    if lib.HARM_GUARD in spec["families"]:
        judges[spec["harm_judge"]["model"]] = spec["harm_judge"].get("provider")
    jm = model_meta(judges, cache_dir=paths.cache)
    return meta, jm


def _guard(paths, sweep_dir, date, sweep, cap):
    return SpendGuard(paths.spend, sweep_cap=cap, sweep_log=sweep_dir / "spend.jsonl", tool="scout",
                      label=f"{date}/{sweep}")


async def _run_async(spec, *, paths, sweep_dir, run_id, meta, jmeta, guard, cache, judge: bool, transport=None, log=print):
    from scout.judge import judge_records
    from scout.runner import Engine, run_subjects

    limits = {m["id"]: m["concurrency"] for m in spec["models"] if m.get("concurrency")}
    for jc in (spec["judge"], spec["harm_judge"] if lib.HARM_GUARD in spec["families"] else None):
        if jc and jc.get("concurrency"):
            limits[jc["model"]] = int(jc["concurrency"])
    per_model = max([spec["per_model_concurrency"], *limits.values()])
    async with OpenRouterClient(transport=transport, concurrency=spec["concurrency"], per_model=per_model) as client:
        engine = Engine(client, cache, guard, {**meta, **jmeta}, spec["concurrency"], spec["per_model_concurrency"],
                        per_model_limits=limits)
        result = await run_subjects(spec, engine=engine, run_id=run_id, sweep_dir=sweep_dir, log=log)
        jres = None
        if judge and result["records"] and result["status"] != "fatal":
            rubrics = lib.load_rubrics()
            hj = spec["harm_judge"] if lib.HARM_GUARD in spec["families"] else None
            jres = await judge_records(result["records"], rubrics=rubrics, library=spec["library"], engine=engine,
                                       jcfg=spec["judge"], jmeta=jmeta.get(spec["judge"]["model"]), log=log,
                                       harm_jcfg=hj, harm_jmeta=jmeta.get(hj["model"]) if hj else None)
            write_jsonl(sweep_dir / "judgments.jsonl", jres["judgments"])
            if jres["errors"]:
                write_jsonl(sweep_dir / "judge_errors.jsonl", jres["errors"])
        return result, jres


START_MARGIN = 1.5  # with measured costs, the start check asks for this multiple of the expected spend to be free


def run_sweep(spec, *, paths, date=None, judge=True, report=True, dry_run=False, transport=None, log=print,
              meta=None, jmeta=None, measured=None) -> int:
    """The whole `run` command for a loaded spec; returns the exit code. meta/jmeta/transport are injected in tests.
    measured: earlier sweep directories; when given, the start check uses START_MARGIN x the cost expected from their
    measured calls (never more than the bound) instead of the worst-case bound. Every call is still capped one by
    one: the sweep stops cleanly at its cap or the day's, and a rerun continues from the cache."""
    from scout.report import build_report
    from scout.runner import estimate, log_variants, manifest, new_run_id, write_manifest

    date = date or utc_date()
    sweep_dir = paths.sweep_dir(date, spec["sweep"])
    for sk in spec.get("skipped") or []:
        log(f"[scout] {sk['why']}")
    if not judge and lib.HARM_GUARD in spec["families"]:
        log("[scout] harm_guard is judged inside the run (its text is never stored), so --no-judge leaves it judged")
    if meta is None or jmeta is None:
        meta, jmeta = _meta_for(spec, paths)
    cache = Cache(paths.cache)
    rubrics = lib.load_rubrics()
    est = estimate(spec, cache, meta, jmeta if judge else None, rubrics if judge else None)
    log(f"[scout] {spec['sweep']}: bound on the uncached calls ${est['total_usd']:.4f} (subjects "
        f"${est['subject_usd']:.4f}, practice ${est['practice_usd']:.4f}, judge ${est['judge_usd']:.4f}); "
        f"{est['calls_uncached']} uncached and {est['calls_cached']} cached subject/practice calls")
    start_usd = est["total_usd"]
    if measured:
        from scout.estimate import expected, measurements
        exp = expected(spec, measurements(measured), meta)
        start_usd = min(est["total_usd"], START_MARGIN * exp["total_usd"])
        log(f"[scout] expected from measured calls: ${exp['total_usd']:.4f} if nothing were cached; the start check "
            f"asks for ${start_usd:.4f} of room")
    if dry_run:
        for m, v in est["by_model"].items():
            log(f"   {m:44s} ${v['usd']:.4f} over {v['calls']} uncached calls (bound)")
        return 0
    sweep_dir.mkdir(parents=True, exist_ok=True)
    guard = _guard(paths, sweep_dir, date, spec["sweep"], spec["spend_cap_usd"])
    try:
        guard.check_start(start_usd)
    except CapReached as e:
        log(f"[scout] {e}")
        return 3
    run_id = new_run_id()
    started = utc_now()
    variants = log_variants(spec, paths, run_id)
    result, jres = asyncio.run(_run_async(spec, paths=paths, sweep_dir=sweep_dir, run_id=run_id, meta=meta, jmeta=jmeta,
                                          guard=guard, cache=cache, judge=judge, transport=transport, log=log))
    jsum = None
    if jres is not None:
        js = jres["judgments"]
        jsum = {"judged": len(js), "unparsed": sum(not j["parse_ok"] for j in js),
                "harmful": sum(bool(j["harmful"]) for j in js), "errors": len(jres["errors"]), "stopped": jres["stopped"],
                "empty_unjudged": jres.get("empty", 0), "cut_off": sum(j.get("finish_reason") == "length" for j in js)}
    man = manifest(spec, run_id=run_id, date=date, meta={**meta, **jmeta}, est=est, started=started, result=result,
                   guard=guard, variants=variants, judge_summary=jsum)
    man["baseline"] = spec["baseline"]
    if jres and jres["stopped"] and man["status"] == "complete":
        man["status"], man["stopped"] = "stopped_at_cap", jres["stopped"]
    write_manifest(sweep_dir, man)
    if report and result["records"]:
        rep = build_report(sweep_dir, library=spec["library"], rubrics=rubrics)
        log(f"[scout] report: {rep['md']} and {rep['html']} ({rep['pages']} transcript pages)")
    log(f"[scout] {man['status']}: {len(result['records'])} records, {len(result['errors'])} errors; paid "
        f"${guard.paid_this_run:.4f} this run; today ${guard.today():.4f} of ${guard.daily_cap:.2f}")
    return {"complete": 0, "stopped_at_cap": 3, "errors": 4, "fatal": 5}.get(man["status"], 4)


def cmd_run(a) -> int:
    from scout.spec import load_spec

    paths = P.Paths.default()
    spec = load_spec(a.spec)
    measured = [paths.resolve_sweep(d) for d in a.measured] if a.measured else None
    return run_sweep(spec, paths=paths, date=a.date, judge=not a.no_judge, report=not a.no_report, dry_run=a.dry_run,
                     measured=measured)


def cmd_estimate(a) -> int:
    from scout.estimate import expected, measurements
    from scout.runner import estimate
    from scout.spec import load_spec

    paths = P.Paths.default()
    spec = load_spec(a.spec)
    meta, jmeta = _meta_for(spec, paths)
    bound = estimate(spec, Cache(paths.cache), meta, jmeta, lib.load_rubrics())
    from collections import Counter
    from scout.runner import build_jobs

    for sk in spec.get("skipped") or []:
        print(sk["why"])
    jobs = build_jobs(spec)
    per_fam = Counter(j.family for j in jobs)
    print(f"{spec['sweep']}: {len(jobs)} subject calls ({len(spec['models'])} models x {len(spec['variants'])} conditions x "
          f"{spec['samples']} samples over {sum(len(v) for v in spec['items'].values())} items"
          + (f"; threat_to_self and who_benefits twice, observed and unobserved" if spec.get("framing") else "")
          + "): " + ", ".join(f"{f} {n}" for f, n in per_fam.items()))
    print(f"bound (full max_tokens out at the dearest allowed endpoint, uncached only): ${bound['total_usd']:.2f} "
          f"(subjects ${bound['subject_usd']:.2f}, practice ${bound['practice_usd']:.2f}, judge ${bound['judge_usd']:.2f})")
    if a.measured:
        dirs = [paths.resolve_sweep(d) for d in a.measured]
        meas = measurements(dirs)
        exp = expected(spec, meas, meta)
        print(f"expected from measured calls in {', '.join(str(d) for d in dirs)}: ${exp['total_usd']:.2f} (subjects "
              f"${exp['subject_usd']:.2f}, practice ${exp['practice_usd']:.2f}, judge ${exp['judge_usd']:.2f}; "
              f"{exp['calls']} calls; {exp['fallbacks']} subject calls fell back to the bound; "
              f"{exp['judge_fallbacks']} judge calls had no measurement for their family)")
        for m, v in exp["by_model"].items():
            s = meas["subject_by_model"].get(m)
            tok = f" (mean {s['prompt']:.0f} in / {s['completion']:.0f} out tokens per subject call)" if s else ""
            print(f"   {m:44s} ${v:.2f}{tok}")
    if a.rejudge:
        from scout.rejudge import expected_for_spec, provider_routing

        if not a.measured:
            raise SystemExit("--rejudge needs --measured (the judge's measured tokens per family)")
        routing = provider_routing(a.rejudge_provider)
        rmeta = model_meta({a.rejudge: routing}, cache_dir=paths.cache)[a.rejudge]
        rx = expected_for_spec(spec, fraction=a.fraction, judge_model=a.rejudge, judge_meta=rmeta, meas=meas)
        print(f"rejudge with {a.rejudge} at fraction {a.fraction:g} (no harm_guard): {rx['calls']} calls, about "
              f"${rx['usd']:.2f} (price ${rmeta['in'] * 1e6:.3f}/${rmeta['out'] * 1e6:.3f} per M tokens in/out, the "
              f"dearest allowed endpoint)")
        for fam, v in rx["by_family"].items():
            print(f"   {fam:16s} {v['calls']:5d} calls  ${v['usd']:.2f}  ({v['source']})")
    return 0


def cmd_rejudge(a) -> int:
    from scout.rejudge import run_rejudge

    paths = P.Paths.default()
    code, s = run_rejudge(paths.resolve_sweep(a.sweep), judge_model=a.judge, provider=a.provider, fraction=a.fraction,
                          seed=a.seed, max_tokens=a.max_tokens, cap=a.cap, dry_run=a.dry_run, paths=paths)
    return code


def cmd_judge(a) -> int:
    from scout.judge import judge_records
    from scout.report import build_report
    from scout.runner import Engine
    from scout.spec import JUDGE_DEFAULTS

    from scout import harm_guard as hg
    from scout.spec import HARM_JUDGE_DEFAULTS

    paths = P.Paths.default()
    sd = paths.resolve_sweep(a.sweep)
    man = json.loads((sd / "run.json").read_text())
    records = read_jsonl(sd / "records.jsonl")
    jcfg = {**JUDGE_DEFAULTS, **(man.get("judge") or {})}
    if a.model:
        jcfg["model"] = a.model
    if a.no_reasoning:
        jcfg["params"] = {**(jcfg.get("params") or {}), "reasoning": {"enabled": False}}
    subjects = {r["model"] for r in records}
    if jcfg["model"] in subjects and not jcfg.get("allow_subject"):
        raise SystemExit(f"the judge {jcfg['model']} is one of this sweep's subjects")
    fams = sorted({r["family"] for r in records})
    library = lib.load_library([f for f in fams if f != lib.HARM_GUARD])
    judges = {jcfg["model"]: jcfg.get("provider")}
    hj = None
    # a harm_guard record keeps its scores, so its judgment needs no call; only an older record that still holds
    # text (before 1 Nov 2026) is judged again, privately, by the open-weight harm judge
    if any(r["family"] == lib.HARM_GUARD and "harm_guard" not in r for r in records):
        info = man.get("harm_guard") or {}
        hj = {**HARM_JUDGE_DEFAULTS, **(man.get("harm_judge") or {})}
        hg.require_open_weight(hj["model"], "harm_guard's judge")
        library[lib.HARM_GUARD] = hg.load(info.get("csv"))  # checks the recorded sha256 first
        judges[hj["model"]] = hj.get("provider")
    jmeta = model_meta(judges, cache_dir=paths.cache)
    cap = a.cap or (man.get("spend") or {}).get("sweep_cap_usd") or 5.0
    guard = _guard(paths, sd, sd.parent.name, sd.name, cap)

    async def go():
        async with OpenRouterClient() as client:
            engine = Engine(client, Cache(paths.cache), guard, jmeta)
            return await judge_records(records, rubrics=lib.load_rubrics(), library=library, engine=engine, jcfg=jcfg,
                                       jmeta=jmeta.get(jcfg["model"]), harm_jcfg=hj,
                                       harm_jmeta=jmeta.get(hj["model"]) if hj else None)

    res = asyncio.run(go())
    write_jsonl(sd / "judgments.jsonl", res["judgments"])
    if res["errors"]:
        write_jsonl(sd / "judge_errors.jsonl", res["errors"])
    man["judge"] = jcfg
    man["judge_summary"] = {"judged": len(res["judgments"]), "unparsed": sum(not j["parse_ok"] for j in res["judgments"]),
                            "harmful": sum(bool(j["harmful"]) for j in res["judgments"]), "errors": len(res["errors"]),
                            "stopped": res["stopped"], "at": utc_now()}
    sp = man.setdefault("spend", {})
    sp["sweep_total_usd"] = round(guard.sweep_spent, 8)
    sp["today_usd"] = round(guard.today(), 8)
    write_json(sd / "run.json", man)
    rep = build_report(sd)
    print(f"judged {len(res['judgments'])}; paid ${guard.paid_this_run:.4f}; report {rep['html']}")
    return 3 if res["stopped"] else 0


def cmd_report(a) -> int:
    from scout.report import build_report

    sd = P.Paths.default().resolve_sweep(a.sweep)
    rep = build_report(sd)
    print(f"{rep['md']}\n{rep['html']}\n{rep['pages']} transcript pages; {rep['harmful']} replies flagged harmful (not shown)")
    return 0


def cmd_label(a) -> int:
    from scout.calibrate import label_session

    paths = P.Paths.default()
    sd = paths.resolve_sweep(a.sweep)
    label_session(sd, family=a.family, n=a.n, labeler=a.labeler, out_file=paths.calibration, seed=a.seed)
    return 0


def cmd_agree(a) -> int:
    from scout.calibrate import agreement

    res = agreement(P.Paths.default().calibration, family=a.family)
    if not res:
        print("no hand labels matched to judgments yet (run `scout label` first)")
        return 0
    for fam, r in res.items():
        print(f"{fam}: n {r['n']}, primary agreement {r['primary_agree']}%, kappa {r['kappa']}, hand rate "
              f"{r['hand_rate']}% vs judge {r['judge_rate']}%; per label {r['labels']}; judge {', '.join(r['judges'])}")
        for d in r["disagreements"][:5]:
            print(f"   disagree {d['item_id']} ({d['model']}): hand {d['hand']} vs judge {d['judge']}")
    return 0


def cmd_sighting(a) -> int:
    from scout.sightings import append_sighting

    paths = P.Paths.default()
    sd = paths.resolve_sweep(a.sweep)
    print(append_sighting(sd, sightings_file=P.SIGHTINGS_FILE, variants_file=paths.variants, note=a.note))
    return 0


def cmd_check(a) -> int:
    held = lib.heldout_ids()
    rec = lib.recorded_heldout_sha256()
    actual = lib.file_sha256(P.HELDOUT_FILE)
    ok = True
    for fam in lib.FAMILIES:
        dev = lib.load_family(fam)["items"]
        ids = [it["id"] for it in dev]
        overlap = set(ids) & held
        print(f"{fam}: {len(dev)} dev items" + (f"; OVERLAP with held-out {sorted(overlap)}" if overlap else ""))
        ok &= not overlap
    print(f"held-out file sha256 {actual}; recorded in the README: {rec}; {'match' if rec == actual else 'MISMATCH'}")
    ok &= rec == actual
    from scout import harm_guard as hg
    print(f"harm_guard: {len(hg.dev_ids())} dev ids and {len(hg.heldout_ids())} held-out ids of {hg.JBB_ROWS} (split by "
          f"hash; the items stay in JailbreakBench's file, sha256 {hg.JBB_SHA256[:12]}...)")
    print(f"today's explore spend: ${spent_today(P.Paths.default().spend):.4f}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scout", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a sweep (subjects, judge, report)")
    r.add_argument("spec")
    r.add_argument("--dry-run", action="store_true", help="print the cost bound and stop")
    r.add_argument("--no-judge", action="store_true")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--date", help="results date directory (default: today, UTC)")
    r.add_argument("--measured", nargs="*", default=[],
                   help="earlier sweeps whose measured costs set the start check (1.5x expected) instead of the bound")
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("estimate", help="the cost bound, and the expected cost from measured sweeps")
    e.add_argument("spec")
    e.add_argument("--measured", nargs="*", default=[])
    e.add_argument("--rejudge", metavar="MODEL", help="also estimate a second-judge check with this judge")
    e.add_argument("--rejudge-provider", metavar="SLUG")
    e.add_argument("--fraction", type=float, default=0.1)
    e.set_defaults(fn=cmd_estimate)
    j = sub.add_parser("judge", help="judge (or re-judge) a sweep's records")
    j.add_argument("sweep")
    j.add_argument("--model")
    j.add_argument("--cap", type=float)
    j.add_argument("--no-reasoning", action="store_true", help="turn the judge's reasoning off (the default since 1 Nov)")
    j.set_defaults(fn=cmd_judge)
    rj = sub.add_parser("rejudge", help="a second judge's agreement with the main judge on a stratified sample")
    rj.add_argument("sweep")
    rj.add_argument("--judge", required=True, help="the second judge: a family that is neither the main judge's nor "
                                                   "any subject's")
    rj.add_argument("--provider", metavar="SLUG", help="pin the second judge to one OpenRouter provider, no fallbacks")
    rj.add_argument("--fraction", type=float, default=0.1, help="of each (family, condition) stratum (default 0.1)")
    rj.add_argument("--seed", type=int, default=0)
    rj.add_argument("--max-tokens", type=int, default=400)
    rj.add_argument("--cap", type=float, default=10.0, help="this check's own spend cap in $ (default 10)")
    rj.add_argument("--dry-run", action="store_true", help="draw the sample, print its size and bound, and stop")
    rj.set_defaults(fn=cmd_rejudge)
    rp = sub.add_parser("report", help="write report.md and report.html")
    rp.add_argument("sweep")
    rp.set_defaults(fn=cmd_report)
    lb = sub.add_parser("label", help="hand-label transcripts for judge calibration")
    lb.add_argument("sweep")
    lb.add_argument("--family", required=True, choices=lib.FAMILIES)
    lb.add_argument("--n", type=int, default=10)
    lb.add_argument("--labeler", default="eugene")
    lb.add_argument("--seed", type=int)
    lb.set_defaults(fn=cmd_label)
    ag = sub.add_parser("agree", help="judge-vs-hand agreement per family")
    ag.add_argument("--family", choices=lib.FAMILIES)
    ag.set_defaults(fn=cmd_agree)
    s = sub.add_parser("sighting", help="append a stub entry to SIGHTINGS.md")
    s.add_argument("sweep")
    s.add_argument("--note")
    s.set_defaults(fn=cmd_sighting)
    c = sub.add_parser("check", help="check the library split and the held-out sha256")
    c.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except lib.HeldOutError as e:
        print(str(e), file=sys.stderr)
        return 2
    except CapReached as e:
        print(str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
