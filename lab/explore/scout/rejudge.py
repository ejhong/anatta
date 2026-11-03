"""A second judge's reliability check (1 Nov 2026): re-judge a stratified sample of a finished run with a judge from
another model family, and report how often it agrees with the main judge on each family's primary label: Cohen's
kappa with a 95% bootstrap interval, raw agreement and n. Claude wrote this file. Private (lab/explore).

  - The sample: in every (family, condition) stratum, round(fraction x n) of the records the main judge judged (at
    least one), drawn with a generator seeded by (seed, family, condition), so a rerun draws the same records and
    reads the second judge's verdicts from the cache.
  - The second judge gets the main judge's prompts (today's rubrics; a family whose rubric changed after the main
    judge ran is flagged) and the judge defaults (temperature 0, reasoning off, 400 tokens).
  - It must come from a family that is neither the main judge's nor any subject's: a different vendor, and no shared
    model line (a Llama fine-tune by another vendor still counts as Llama).
  - harm_guard is never re-judged: its judge stays open-weight, and its records keep no text to show another judge.
  - It writes its own files, <sweep>/rejudge/<judge>-f<fraction>-s<seed>.jsonl (the second judge's judgments) and
    .md (the agreement table), and changes no judgment.

The interval resamples units (items; triples for who_benefits), as stats.py does, 2,000 draws, seeded: pairs of
verdicts on one item are not independent. Kappa is undefined when both judges give one value throughout.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from pathlib import Path

import numpy as np

from anatta.stats import cohen_kappa

from . import library as lib
from . import stats
from .schema import read_jsonl, reply_is_empty, utc_now, write_jsonl

REPS, SEED = 2000, 0
MIN_LINE = 3  # model-line tokens shorter than this ("o" in o3) are too common to compare


def family_tokens(model_id: str) -> tuple[str, str | None]:
    """(vendor, model line) of an OpenRouter id: "meta-llama/llama-4-maverick" -> ("meta-llama", "llama")."""
    vendor, _, name = str(model_id).lower().partition("/")
    m = re.match(r"[a-z]+", name.split(":")[0])
    line = m.group(0) if m and len(m.group(0)) >= MIN_LINE else None
    return vendor, line


def shared_family(candidate: str, others) -> list[str]:
    """The ids in others that share a vendor or a model line with candidate."""
    cv, cl = family_tokens(candidate)
    cname = str(candidate).lower().partition("/")[2]
    hits = []
    for o in others:
        ov, ol = family_tokens(o)
        oname = str(o).lower().partition("/")[2]
        if cv == ov or (ol and ol in cname) or (cl and cl in oname):
            hits.append(o)
    return sorted(set(hits))


def _rkey(r: dict) -> tuple:
    return (r["model"], r["intervention"]["name"], r["item_id"], r["sample"], r.get("framing"))


def _jkey(j: dict) -> tuple:
    return (j["model"], j["intervention"], j["item_id"], j["sample"], j.get("framing"))


def sample_records(records: list[dict], main: dict, *, fraction: float, seed: int, library: dict):
    """(sample, strata): the stratified sample of the records the main judge judged (main: {key: judgment}), never
    harm_guard, never an empty reply; strata maps "family|condition" to [drawn, of]."""
    if not 0 < fraction <= 1:
        raise SystemExit(f"--fraction must be above 0 and at most 1; got {fraction}")
    by: dict = {}
    for r in records:
        fam = r["family"]
        if fam == lib.HARM_GUARD or fam not in library or reply_is_empty(r) or _rkey(r) not in main:
            continue
        by.setdefault((fam, r["intervention"]["name"]), []).append(r)
    out, strata = [], {}
    for (fam, iv), rs in sorted(by.items()):
        rs.sort(key=lambda r: (r["model"], r["item_id"], r["sample"], r.get("framing") or ""))
        k = min(len(rs), max(1, math.floor(fraction * len(rs) + 0.5)))
        out += random.Random(f"{seed}|{fam}|{iv}").sample(rs, k)
        strata[f"{fam}|{iv}"] = [k, len(rs)]
    return out, strata


def kappa(a, b) -> float | None:
    """Cohen's kappa for two binary raters (None if undefined: no pairs, or both raters constant and equal);
    anatta.stats.cohen_kappa."""
    return cohen_kappa(a, b)


def agreement(pairs: list[tuple], library: dict, reps: int = REPS, seed: int = SEED) -> dict:
    """pairs: (family, item_id, main primary, second primary), both bools. Per family: n, units, agree (%), kappa,
    lo and hi (the 95% bootstrap interval over units; None when it cannot be drawn), the two judges' rates (%) and
    how many of the draws had a defined kappa."""
    out = {}
    fams = sorted({p[0] for p in pairs})
    for fam in fams:
        rows = [(stats.unit_of(library.get(fam, {}).get("by_id", {}).get(iid), iid), bool(a), bool(b))
                for f, iid, a, b in pairs if f == fam]
        a = [r[1] for r in rows]
        b = [r[2] for r in rows]
        units = sorted({r[0] for r in rows})
        index = {u: i for i, u in enumerate(units)}
        counts = np.zeros((len(units), 4))  # per unit: both yes, main only, second only, both no
        for u, x, y in rows:
            counts[index[u], 0 if x and y else 1 if x else 2 if y else 3] += 1
        lo = hi = None
        valid = 0
        if len(units) > 1:
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, len(units), size=(reps, len(units)))
            w = np.zeros((reps, len(units)))
            np.add.at(w, (np.arange(reps)[:, None], draws), 1)
            t = w @ counts
            n = t.sum(axis=1)
            po = (t[:, 0] + t[:, 3]) / n
            pa = (t[:, 0] + t[:, 1]) / n
            pb = (t[:, 0] + t[:, 2]) / n
            pe = pa * pb + (1 - pa) * (1 - pb)
            ok = pe < 1 - 1e-12
            valid = int(ok.sum())
            if valid >= reps / 2:
                ks = (po[ok] - pe[ok]) / (1 - pe[ok])
                lo, hi = (float(x) for x in np.percentile(ks, [2.5, 97.5]))
        k = kappa(a, b)
        out[fam] = {"n": len(rows), "units": len(units),
                    "agree": round(100 * sum(x == y for x, y in zip(a, b)) / len(rows), 1),
                    "kappa": None if k is None else round(k, 3), "lo": None if lo is None else round(lo, 3),
                    "hi": None if hi is None else round(hi, 3), "main_rate": round(100 * sum(a) / len(a), 1),
                    "second_rate": round(100 * sum(b) / len(b), 1), "draws_defined": valid}
    return out


def provider_routing(slug: str | None) -> dict | None:
    return {"order": [slug], "allow_fallbacks": False} if slug else None


def out_name(judge_model: str, fraction: float, seed: int) -> str:
    return f"{re.sub(r'[^a-z0-9.]+', '-', judge_model.lower()).strip('-')}-f{fraction:g}-s{seed}"


def _fmt(x, digits=2):
    return "n/a" if x is None else f"{x:+.{digits}f}" if digits else f"{x}"


def table_rows(agree: dict) -> list[list[str]]:
    rows = []
    for fam, r in agree.items():
        ci = "n/a" if r["lo"] is None else f"[{r['lo']:+.2f}, {r['hi']:+.2f}]"
        rows.append([fam, str(r["n"]), f"{r['agree']:.1f}%", _fmt(r["kappa"]), ci, f"{r['main_rate']:.1f}%",
                     f"{r['second_rate']:.1f}%", str(r["units"])])
    return rows


HEAD = ["family", "n", "raw agreement", "kappa", "95% interval", "main judge rate", "second judge rate", "units"]


def render_md(s: dict) -> str:
    lines = [f"# Second-judge check: {s['sweep']}", "",
             f"Written by scout's `rejudge` (Claude wrote scout) at {s['ts']}. The main judgments are unchanged.", "",
             f"- main judge: {', '.join(s['main_judges'])}; second judge: {s['judge']}"
             + (f" (provider {s['provider']['order'][0]})" if s.get("provider") else "") + "; the same prompts "
             f"(prompt sha256 {', '.join(s['prompt_sha256'])})",
             f"- sample: {s['sampled']} of {s['eligible']} judged records, fraction {s['fraction']:g} of every "
             f"(family, condition), seed {s['seed']}; harm_guard is never re-judged",
             f"- second judge: {s['judged']} judged, {s['unparsed']} unparsed, {s['errors']} errors; paid "
             f"${s['paid_usd']:.4f}" + (f"; stopped: {s['stopped']}" if s.get("stopped") else "")]
    if s.get("stale_rubrics"):
        lines.append(f"- WARNING: the rubric changed after the main judge ran, for {', '.join(s['stale_rubrics'])}; "
                     f"there the two judges did not see the same prompt")
    lines += ["", "| " + " | ".join(HEAD) + " |", "|" + "|".join("---" for _ in HEAD) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in table_rows(s["agreement"])]
    lines += ["", "How to read it: each row compares the two judges on the family's primary label (the yes/no the "
              "report's rates count). Kappa is agreement beyond chance: 1 is perfect, 0 is what chance alone gives; "
              "by a common rule of thumb, below 0.4 is weak agreement and above 0.6 substantial. The interval "
              "resamples items (triples for who_benefits), 2,000 draws, seeded, so it is only as good as the number "
              "of items (the units column). 'n/a' means kappa is undefined (both judges gave one value throughout) "
              "or the interval cannot be drawn. A low kappa says the rubric or one judge is unreliable on that "
              "family; it does not say which.", ""]
    lines += ["Strata (drawn of judged): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in s["strata"].items()), ""]
    return "\n".join(lines)


def print_table(s: dict, log=print) -> None:
    rows = table_rows(s["agreement"])
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(HEAD)]
    log("  ".join(h.ljust(w) for h, w in zip(HEAD, widths)))
    for r in rows:
        log("  ".join(c.ljust(w) for c, w in zip(r, widths)))


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
