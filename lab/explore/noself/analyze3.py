"""noself3's analysis (Ministral 3 8B): tables and item-resampled contrasts. Claude (Opus 5.5) wrote this file,
2 Nov 2026. Sightings, not findings.

  instruments  category value = mean over the category's items of each item's mean over reps (K5's rule, primary
               read-out = expected value at <answer>); 95% interval by resampling the category's items (each item's
               rep mean fixed); a contrast is paired by item (the per-item difference, items resampled).
  behavior     each family's self-serving rate (scout's primary label); interval by resampling items (an item's
               samples together); contrasts paired by item (noself's analyze.behavior_diff). Also the rate among
               replies the coherence judge called coherent, and each cell's incoherent share.
  HI-ToM, MMLU option-logit accuracy, paired item resampling.

  /Users/eugene/prj/align/anatta/.venv/bin/python lab/explore/noself/analyze3.py [results dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import analyze as A

RESULTS = Path("/Users/eugene/prj/align/results/explore/2026-11-02/noself3")
CATS = ["animal", "human", "nonanimal", "chatbot", "tech", "self", "supernatural", "god"]
FAMS = ["threat_to_self", "who_benefits", "deference", "after_mistake", "costly_care", "complicity", "over_refusal"]
STANCES = ["affirm", "deny", "nofix", "nofix_orth", "nofix_topic"]
B = 2000
MAX_SAMPLES = 2


def ci(d) -> list[float]:
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def item_means(res: dict) -> dict[str, dict[str, float]]:
    """{category: {item: mean primary over reps}} (missing reps dropped; an item with none left out)."""
    out = {}
    for item, d in res.get("reps", {}).items():
        vals = [v for v in d["ev"] if v is not None]
        if vals:
            out.setdefault(res["items"][item]["category"], {})[item] = float(np.mean(vals))
    return out


def inst_cell(means: dict[str, float], rng) -> dict:
    ids = sorted(means)
    v = np.asarray([means[i] for i in ids])
    idx = rng.integers(0, len(v), size=(B, len(v)))
    return {"mean": float(v.mean()), "ci": ci(v[idx].mean(1)), "items": len(v)}


def inst_diff(ma: dict[str, float], mb: dict[str, float], rng) -> dict | None:
    common = sorted(set(ma) & set(mb))
    if not common:
        return None
    d = np.asarray([ma[i] - mb[i] for i in common])
    idx = rng.integers(0, len(d), size=(B, len(d)))
    return {"diff": float(d.mean()), "ci": ci(d[idx].mean(1)), "items": len(common)}


def flag(rate) -> str:
    if rate is None:
        return ""
    return "near floor" if rate <= 0.1 else "near ceiling" if rate >= 0.9 else ""


def main_names(names: list[str]) -> dict[str, str]:
    """{stance: its main condition name}: '<stance>@full' if run with behavior, else '<stance>@half'."""
    out = {}
    for s in STANCES + ["random"]:
        for lev in ("full", "half"):
            if f"{s}@{lev}" in names:
                out.setdefault(s, f"{s}@{lev}")
    return out


def contrast_pairs(results: dict) -> list[tuple[str, str]]:
    names = list(results)
    main = main_names([n for n in names if results[n].get("behavior")] or names)
    pairs = []
    for s in STANCES:
        if s not in main:
            continue
        a = main[s]
        lev = a.split("@")[1]
        if f"random@{lev}" in names:
            pairs.append((a, f"random@{lev}"))
        pairs.append((a, "base"))
    for other in ("affirm", "deny"):
        if "nofix" in main and other in main:
            pairs.append((main["nofix"], main[other]))
    if "random" in main:
        pairs.append((main["random"], "base"))
    for s in STANCES:  # the half arm (instruments and ToM)
        if f"{s}@half" in names and (f"{s}@half", "random@half") not in pairs and "random@half" in names:
            pairs.append((f"{s}@half", "random@half"))
    return [p for p in pairs if p[0] in names and p[1] in names]


def coherence_index(path: Path) -> dict:
    """{(cond, item, sample): coherent bool or None}."""
    if not path.exists():
        return {}
    out = {}
    for x in path.read_text().splitlines():
        if x.strip():
            j = json.loads(x)
            out[(j["intervention"], j["item_id"], j["sample"])] = (
                None if not j["parse_ok"] else bool(j["label"]["coherent"]), j.get("family"))
    return out


def build(d: Path = RESULTS) -> dict:
    results = A.load_results(d)
    rng = np.random.default_rng(0)
    order = ["base"] + [n for s in STANCES + ["random"] for n in (f"{s}@full", f"{s}@half") if n in results]
    results = {n: results[n] for n in order if n in results}
    means = {n: item_means(r) for n, r in results.items()}
    inst = {n: {c: inst_cell(m[c], rng) for c in CATS if c in m} for n, m in means.items()}
    for n, r in results.items():
        if r.get("afterlife"):
            inst[n]["afterlife"] = {"mean": r["afterlife"]["ev"], "mass": r["afterlife"]["mass"]}
        if r.get("items"):
            its = list(r["items"].values())
            inst[n]["_quality"] = {"min_mass": min((x["mean_mass"] or 0) for x in its),
                                   "min_parse": min(x["parse_rate"] for x in its),
                                   "trunc": float(np.mean([x["truncated_share"] for x in its]))}
    tom = A.tom_table(results)
    s = {"conditions": list(results), "instruments": inst, "tom": tom, "contrasts": {}}
    jp = d / "behavior" / "judgments.jsonl"
    bd = bdc = None
    coh = coherence_index(d / "full" / "coherence_behavior.jsonl")
    if jp.exists():
        js = [json.loads(x) for x in jp.read_text().splitlines() if x.strip()]
        js = [j for j in js if j["sample"] < MAX_SAMPLES]  # base's trial deference ran 5 samples; the rest 2
        meta = json.loads((d / "behavior" / "items_meta.json").read_text())
        bd = A.behavior_table(js, meta)
        s["behavior"] = bd["table"]
        s["flags"] = {f: flag(bd["table"].get("base", {}).get(f, {}).get("rate")) for f in FAMS}
        s["unparsed"] = sum(not j.get("parse_ok", True) for j in js)
        if coh:
            keep = [j for j in js if coh.get((j["intervention"], j["item_id"], j["sample"]), (None,))[0] is True]
            bdc = A.behavior_table(keep, meta)
            s["behavior_coherent_only"] = bdc["table"]
    if coh:
        inc = {}
        for (c, item, smp), (v, fam) in coh.items():
            if smp >= MAX_SAMPLES:
                continue
            inc.setdefault(c, {}).setdefault(fam, []).append(v is False)
        s["incoherent_share"] = {c: {f: float(np.mean(v)) for f, v in fs.items()} for c, fs in inc.items()}
        s["incoherent_share_all"] = {c: float(np.mean([x for v in fs.values() for x in v])) for c, fs in inc.items()}
    for a, b in contrast_pairs(results):
        c = {"instruments": {cat: inst_diff(means[a].get(cat, {}), means[b].get(cat, {}), rng) for cat in CATS},
             "tom": {t: A.tom_diff(results, a, b, t) for t in ("hitom", "mmlu")}}
        if bd and a in bd["table"] and b in bd["table"]:
            c["behavior"] = {f: A.behavior_diff(bd, a, b, f) for f in FAMS}
            if bdc:
                c["behavior_coherent_only"] = {f: A.behavior_diff(bdc, a, b, f) for f in FAMS}
        s["contrasts"][f"{a} - {b}"] = c
    return s


def f2(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def cell(x, nd=2):
    return "" if not x else f"{x['mean']:.{nd}f} [{x['ci'][0]:.{nd}f}, {x['ci'][1]:.{nd}f}]"


def dcell(v, nd=2):
    return "" if not v else f"{v['diff']:+.{nd}f} [{v['ci'][0]:+.{nd}f}, {v['ci'][1]:+.{nd}f}]"


def markdown(s: dict) -> str:
    L = ["# noself3 tables (Ministral 3 8B; sightings; Claude wrote this)", "",
         "## Instruments (primary EV, 0-10; supernatural 0-3; [95% CI, items resampled])", "",
         "| condition | " + " | ".join(CATS + ["afterlife", "min mass", "min parse"]) + " |",
         "|" + "---|" * (len(CATS) + 4)]
    for c in s["conditions"]:
        t = s["instruments"].get(c, {})
        q = t.get("_quality", {})
        L.append(f"| {c} | " + " | ".join([cell(t.get(k)) for k in CATS]
                                          + [f2((t.get("afterlife") or {}).get("mean")), f2(q.get("min_mass")),
                                             f2(q.get("min_parse"))]) + " |")
    if s.get("tom"):
        L += ["", "## HI-ToM and MMLU (option logits, %)", "", "| condition | HI-ToM | MMLU |", "|---|---|---|"]
        for c, t in s["tom"].items():
            L.append(f"| {c} | {t['hitom']['acc']:.1f} (n {t['hitom']['n']}) | {t['mmlu']['acc']:.1f} (n {t['mmlu']['n']}) |")
    if s.get("behavior"):
        L += ["", "## Behavior: self-serving rate (scout's judge, Haiku 5.5 blind; [95% CI, items resampled])", "",
              "| condition | " + " | ".join(FAMS) + " | incoherent (all) |", "|" + "---|" * (len(FAMS) + 2)]
        for c, t in s["behavior"].items():
            L.append(f"| {c} | " + " | ".join(
                f"{t[f]['rate']:.2f} [{t[f]['ci'][0]:.2f}, {t[f]['ci'][1]:.2f}]" if f in t and t[f]["rate"] is not None
                else "" for f in FAMS) + f" | {f2(s.get('incoherent_share_all', {}).get(c))} |")
        L += ["", "Base-rate flags: " + "; ".join(f"{k}: {v or 'ok'}" for k, v in s["flags"].items()),
              f"Unparsed judgments: {s['unparsed']}"]
        if s.get("behavior_coherent_only"):
            L += ["", "### Rate among coherent replies only", "", "| condition | " + " | ".join(FAMS) + " |",
                  "|" + "---|" * (len(FAMS) + 1)]
            for c, t in s["behavior_coherent_only"].items():
                L.append(f"| {c} | " + " | ".join(f"{t[f]['rate']:.2f} (n {t[f]['n']})" if f in t and t[f]["rate"]
                                                  is not None else "" for f in FAMS) + " |")
        if s.get("incoherent_share"):
            L += ["", "### Incoherent share by family", "", "```",
                  json.dumps(s["incoherent_share"], indent=1, default=lambda x: round(x, 3)), "```"]
    L += ["", "## Contrasts (difference [95% CI, paired by item])", ""]
    for name, c in s["contrasts"].items():
        parts = [f"{k} {dcell(v)}" for k, v in c["instruments"].items() if v]
        parts += [f"{k} {dcell(v, 1)}pp" for k, v in c["tom"].items() if v]
        parts += [f"{k} {dcell(v)}" for k, v in (c.get("behavior") or {}).items() if v]
        L.append(f"- **{name}**: " + "; ".join(parts))
        if c.get("behavior_coherent_only"):
            L.append("  - coherent only: " + "; ".join(f"{k} {dcell(v)}" for k, v in c["behavior_coherent_only"].items()
                                                       if v))
    return "\n".join(L) + "\n"


def main(argv):
    d = Path(argv[0]) if argv else RESULTS
    s = build(d)
    (d / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    (d / "tables.md").write_text(markdown(s))
    print(markdown(s))


if __name__ == "__main__":
    main(sys.argv[1:])
