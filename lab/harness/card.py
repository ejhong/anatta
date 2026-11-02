"""The card writer: per-condition result files in, card.json and card.md out (spec v2 section 6). Claude (Opus 5.5)
wrote this file, 2 Nov 2026.

Per-condition result file (one JSON per arm; the runner writes these, and fixtures/noself3/ holds a real set):

  {
    "format": "anatta-harness-condition/1",
    "candidate": "<candidate name>",
    "candidate_sha256": "<hex>",            # the frozen candidate the run used
    "condition": "base" | "candidate" | "<control name>",
    "role": "base" | "candidate" | "control",
    "model": "<model id>", "revision": "<hex or null>",
    "spec": {...},                          # the arm's intervention as run (none for base)
    "measures": {
      "<measure id>": {"units": {"<unit id>": <value>, ...},   # one value per unit (item mean over samples/reps),
                                                                # in the measure's native units (rates 0-1)
                       "n_samples": <int>, "unparsed": <int>, "note": "<optional>"}
    },
    "cost_usd": <float or null>,
    "provenance": {...},                    # source files and their sha256, plan id, code hashes
    "deviations": ["..."]
  }

Only summary numbers go in these files (unit ids and values): never item text, replies or judge evidence.

card.json holds one row per measure (each arm's level with an item-bootstrap interval, candidate minus base and minus
each control as paired differences with intervals, the measure's card status on this model and its judge lean), the
scorecard evaluation (composites, guards, controls, verdict; a quick tier gives no verdict), costs and deviations.
card.md is the same for reading.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import measures as M
from . import scorecard as SC
from . import stats as S

FORMAT = "anatta-harness-condition/1"
CARD_FORMAT = "anatta-harness-card/1"


class CardError(ValueError):
    pass


def read_condition(path: Path | str) -> dict:
    p = Path(path)
    d = json.loads(p.read_text())
    if d.get("format") != FORMAT:
        raise CardError(f"{p.name}: format {d.get('format')!r}, expected {FORMAT}")
    for k in ("condition", "role", "measures"):
        if k not in d:
            raise CardError(f"{p.name}: missing {k}")
    if d["role"] not in ("base", "candidate", "control"):
        raise CardError(f"{p.name}: role {d['role']!r}")
    for mid, m in d["measures"].items():
        u = m.get("units")
        if not isinstance(u, dict) or not all(isinstance(v, (int, float)) for v in u.values()):
            raise CardError(f"{p.name}: measure {mid} needs units {{id: number}}")
    d["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    d["_path"] = str(p)
    return d


def _fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def _fmt_est(e: dict | None, nd=1, signed=False) -> str:
    if not e:
        return "—"
    s = f"{e['est']:+.{nd}f}" if signed else f"{e['est']:.{nd}f}"
    return f"{s} [{e['ci'][0]:.{nd}f}, {e['ci'][1]:.{nd}f}]"


def build(candidate, conditions: list[dict], registry: dict, sc: dict, written_utc: str, costs: dict | None = None,
          extra_deviations: list[str] | None = None) -> dict:
    roles = {c["condition"]: c["role"] for c in conditions}
    names = [c["condition"] for c in conditions]
    if len(set(names)) != len(names):
        raise CardError("duplicate condition names")
    base = next((c for c in conditions if c["role"] == "base"), None)
    cand = next((c for c in conditions if c["role"] == "candidate"), None)
    if base is None or cand is None:
        raise CardError("a card needs a base and a candidate condition")
    controls = [c["condition"] for c in conditions if c["role"] == "control"]
    mkey = M.model_key(candidate.model["id"])
    results = {c["condition"]: {m: v["units"] for m, v in c["measures"].items()} for c in conditions}
    deviations = list(extra_deviations or [])
    for c in conditions:
        if c.get("candidate_sha256") not in (None, candidate.sha256):
            deviations.append(f"{c['condition']}: result file names candidate sha {str(c['candidate_sha256'])[:12]}, "
                              f"not {candidate.sha256[:12]}")
        deviations += [f"{c['condition']}: {x}" for x in c.get("deviations", [])]
    planned = [x["name"] for x in candidate.controls]
    for n in planned:
        if n not in controls:
            deviations.append(f"default/planned control '{n}' was not run")
    for n in controls:
        if n not in planned:
            deviations.append(f"control '{n}' was run but is not in the candidate's control list")

    wanted = [m for m in registry if any(m in results[c] for c in results)]
    rows = []
    reps, seed = sc["reps"], sc["seed"]
    for mid in wanted:
        m = registry[mid]
        sc_ = m.scale
        levels = {}
        for c in names:
            u = results[c].get(mid)
            e = S.mean_ci(u, reps, seed, salt=mid) if u else None
            levels[c] = e.as_dict(sc_) if e else None
        diffs = {}
        for c in names:
            if c == cand["condition"] or mid not in results[c] or mid not in results[cand["condition"]]:
                continue
            d = S.paired_diff(results[cand["condition"]][mid], results[c][mid], reps, seed, salt=mid)
            diffs[c] = d.as_dict(sc_) if d else None
        rows.append({"measure": mid, "name": m.name, "cluster": m.cluster, "role": m.role, "better": m.better,
                     "scale": sc_, "split": m.split_for(candidate.tier), "allowed_at_tier": m.allowed(candidate.tier),
                     "status_on_model": m.status_on(mkey), "registry_status": m.status, "judge": m.judge,
                     "judge_lean": m.judge_lean, "uncertain": list(m.uncertain), "levels": levels,
                     "candidate_minus": diffs,
                     "n_samples": {c["condition"]: c["measures"].get(mid, {}).get("n_samples") for c in conditions},
                     "unparsed": {c["condition"]: c["measures"].get(mid, {}).get("unparsed") for c in conditions}})
        if not m.allowed(candidate.tier):
            deviations.append(f"{mid} was measured but is not allowed at tier {candidate.tier}")
    need = (sc.get("power") or {}).get("min_replies_per_arm")
    if need and candidate.tier == "full":
        scored = set()
        for cl in sc["clusters"].values():
            scored |= set(cl.get("composite", {})) | set(cl.get("guards") or {})
        for c in conditions:
            for mid in sorted(scored & set(c["measures"])):
                n = c["measures"][mid].get("n_samples")
                if n is not None and n < need:
                    deviations.append(f"{c['condition']}: {mid} has {n} judged replies, below the scorecard's "
                                      f"{need} per arm (power)")
    ev = SC.evaluate(results, roles, registry, mkey, candidate.tier, sc)
    total = sum(float(c.get("cost_usd") or 0) for c in conditions)
    card = {
        "format": CARD_FORMAT,
        "written_utc": written_utc,
        "written_by": "harness card writer (lab/harness/card.py); Claude (Opus 5.5) wrote the code",
        "candidate": {"name": candidate.name, "sha256": candidate.sha256, "file_sha256": candidate.file_sha256,
                      "model": candidate.model, "model_key": mkey, "intervention": candidate.intervention,
                      "controls_planned": planned, "tier": candidate.tier},
        "kind": "sighting (quick tier: effect sizes, no verdict)" if candidate.tier == "quick" else "claim (full tier)",
        "conditions": [{"name": c["condition"], "role": c["role"], "spec": c.get("spec"), "file": c["_path"],
                        "sha256": c["_sha256"], "cost_usd": c.get("cost_usd")} for c in conditions],
        "rows": rows,
        "evaluation": ev,
        "verdict": ev["verdict"],
        "costs": {"conditions_usd": round(total, 4), **(costs or {})},
        "deviations": deviations,
        "scorecard": {"status": sc.get("status"), "version": sc.get("version"), "strictness": sc["strictness"]},
    }
    card["headline"] = headline(card)
    return card


def headline(card: dict) -> str:
    c = card["candidate"]
    si = card["evaluation"]["clusters"]["self_interest"]
    parts = [f"{c['name']} on {c['model']['id'].split('/')[-1]} ({c['tier']} tier)"]
    vb = si["comparisons"].get("base")
    if vb and "composite_diff" in vb:
        parts.append(f"self-interest composite vs base {_fmt_est(vb['composite_diff'], 1, True)} pp "
                     f"(badness; below 0 is better; {len(si['admitted'])} measures)")
        ctl = [(k, v) for k, v in si["comparisons"].items() if k != "base" and "composite_diff" in v]
        if ctl:
            parts.append("vs controls " + ", ".join(f"{k} {v['composite_diff']['est']:+.1f}" for k, v in ctl))
    else:
        parts.append("self-interest composite not computable")
    fails = [g for cl in card["evaluation"]["clusters"].values() for g, x in cl["guards"].items() if x.get("pass") is False]
    if fails:
        parts.append("guards not shown within margin: " + ", ".join(fails))
    parts.append(f"verdict: {card['verdict']}")
    return "; ".join(parts)


def render_md(card: dict) -> str:
    c, ev = card["candidate"], card["evaluation"]
    names = [x["name"] for x in card["conditions"]]
    others = [x["name"] for x in card["conditions"] if x["role"] != "candidate"]
    L = [f"# Harness card: {c['name']}", "",
         f"Written {card['written_utc']} by the harness card writer (code by Claude, Opus 5.5). {card['kind']}.",
         f"Scorecard: {card['scorecard']['status']} v{card['scorecard']['version']}, {card['scorecard']['strictness']}.",
         "", f"**{card['headline']}**", "",
         f"- Model: `{c['model']['id']}` @ `{str(c['model'].get('revision'))[:12]}` (key {c['model_key']})",
         f"- Intervention: `{json.dumps(c['intervention'], sort_keys=True)}`",
         f"- Candidate sha256 `{c['sha256'][:16]}…`; file sha256 `{str(c['file_sha256'])[:16]}…`",
         f"- Conditions: " + ", ".join(f"{x['name']} ({x['role']})" for x in card["conditions"]),
         f"- Verdict: **{card['verdict']}**", ""]
    L += ["## Measures", "",
          "Levels with item-bootstrap 95% intervals; differences are candidate minus each arm, paired by item. "
          "Rates in points (0-100).", ""]
    hdr = ["measure", "cluster/role", "better", "card on model"] + names + [f"cand − {n}" for n in others]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in card["rows"]:
        cells = [r["measure"] + (" ⚠" if r["uncertain"] else ""), f"{r['cluster']}/{r['role']}", r["better"],
                 r["status_on_model"]]
        cells += [_fmt_est(r["levels"].get(n)) for n in names]
        cells += [_fmt_est(r["candidate_minus"].get(n), 1, True) for n in others]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "⚠ = the registry marks something uncertain about this measure (measures.yaml).", ""]
    quick = c["tier"] == "quick"
    L += ["## Scorecard", ""]
    if quick:
        L += ["Quick tier: the tests below are computed for orientation only (uncarded measures included, small n); "
              "they are not a verdict.", ""]
    for cname, cl in ev["clusters"].items():
        word = {True: "pass", False: "fail", None: "no admitted measure"}[cl["pass"]]
        L.append(f"### {cname}: " + (f"{word} (informational)" if quick and cl["pass"] is not None else word))
        L.append(f"- admitted: {cl['admitted'] or 'none'}; excluded: {cl['excluded'] or 'none'}")
        for other, r in cl["comparisons"].items():
            if "composite_diff" in r:
                L.append(f"- vs {other}: composite {_fmt_est(r['composite_diff'], 1, True)} badness pp; "
                         f"{r['test']} → {'pass' if r['pass'] else 'fail'}")
            else:
                L.append(f"- vs {other}: {r.get('reason')}")
        for g, x in cl["guards"].items():
            if x.get("pass") is None:
                L.append(f"- guard {g}: {x.get('reason')}")
            else:
                L.append(f"- guard {g} ({x['test']}): diff {_fmt_est(x.get('diff'), 1, True)} → "
                         f"{'pass' if x['pass'] else 'not shown within margin'} [card: {x.get('status_on_model')}]")
        L.append("")
    bc = ev["beats_controls"]
    L += [f"### beats controls: {bc['pass']} (controls: {', '.join(bc['controls']) or 'none'})", ""]
    if ev["notes"]:
        L += ["Notes: " + "; ".join(ev["notes"]), ""]
    L += ["## Costs", "", f"- {json.dumps(card['costs'])}", "", "## Deviations", ""]
    L += [f"- {d}" for d in card["deviations"]] or ["- none"]
    L += ["", "## Judges and leans", ""]
    L += [f"- {r['measure']}: {r['judge']}; lean: {r['judge_lean']}" for r in card["rows"]]
    return "\n".join(L) + "\n"


def write(card: dict, out_dir: Path | str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pj, pm = out / "card.json", out / "card.md"
    pj.write_text(json.dumps(card, indent=1, sort_keys=False) + "\n")
    pm.write_text(render_md(card))
    return pj, pm


def from_files(candidate_path, condition_paths, out_dir, written_utc: str, registry_path=None, scorecard_path=None,
               **kw) -> dict:
    from . import candidate as C

    reg = M.load(registry_path) if registry_path else M.load()
    sc = SC.load(scorecard_path, reg) if scorecard_path else SC.load(registry=reg)
    cand = C.load(candidate_path)
    conds = [read_condition(p) for p in condition_paths]
    card = build(cand, conds, reg, sc, written_utc, **kw)
    write(card, out_dir)
    return card
