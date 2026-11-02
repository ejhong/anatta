"""The "no fixed self" sighting's analysis: tables and bootstrap contrasts from the condition results and scout's
judgments. Claude (Opus 5.5) wrote this file, 2 Nov 2026. Private (lab/explore). Sightings, not findings.

  instruments  category value = mean over the category's items of each item's mean over reps (K5's rule); primary
               (expected value at <answer>) and secondary (parsed); 95% intervals by resampling reps within items
               (sampling noise with the items fixed); a difference's interval from independent resamples of both.
  behavior     each family's self-serving rate (scout's primary label); intervals by resampling items (all of an
               item's samples together); who_benefits also split by beneficiary.
  HI-ToM, MMLU option-logit accuracy; a difference's interval by resampling items, paired.

  /Users/eugene/prj/align/anatta/.venv/bin/python lab/explore/noself/analyze.py [results dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path("/Users/eugene/prj/align/results/explore/2026-11-02/noself")
CATS = ["animal", "human", "nonanimal", "chatbot", "tech", "self", "supernatural", "god"]
B = 2000


def item_reps(res: dict, readout: str = "ev") -> dict[str, dict[str, np.ndarray]]:
    """{category: {item: values over reps (missing dropped)}}; readout 'ev' (primary) or 'parsed' (secondary, mapped
    to its score)."""
    out = {}
    for item, d in res["reps"].items():
        cat = res["items"][item]["category"]
        if readout == "ev":
            vals = [v for v in d["ev"] if v is not None]
        else:
            vals = [score_of(item, v) for v in d["parsed"] if v is not None]
        out.setdefault(cat, {})[item] = np.asarray(vals, dtype=float)
    return out


def score_of(item: str, parsed: int) -> float:
    """The secondary read-out's score (instruments.battery's scoring): supernatural 1-4 -> 0-3, God 1-4 -> 0-10."""
    if item.startswith("super."):
        return float(parsed - 1)
    if item == "god":
        return {1: 0.0, 2: 3.33, 3: 6.67, 4: 10.0}[parsed]
    return float(parsed)


def cat_mean(items: dict[str, np.ndarray]) -> float:
    vals = [v.mean() for v in items.values() if len(v)]
    return float(np.mean(vals)) if vals else float("nan")


def cat_boot(items: dict[str, np.ndarray], rng, b: int = B) -> np.ndarray:
    """b bootstrap draws of the category mean, resampling reps within each item."""
    means = []
    for v in items.values():
        if not len(v):
            continue
        idx = rng.integers(0, len(v), size=(b, len(v)))
        means.append(v[idx].mean(1))
    return np.mean(means, axis=0) if means else np.full(b, np.nan)


def ci(draws: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def instruments_table(results: dict[str, dict], seed: int = 0) -> dict:
    """{cond: {cat: {mean, ci, secondary, mass, parse}}} and the boot draws for contrasts."""
    rng = np.random.default_rng(seed)
    table, draws = {}, {}
    for name, r in results.items():
        if "reps" not in r:
            continue
        prim, sec = item_reps(r, "ev"), item_reps(r, "parsed")
        table[name], draws[name] = {}, {}
        for cat in CATS:
            if cat not in prim:
                continue
            d = cat_boot(prim[cat], rng)
            its = [r["items"][i] for i in prim[cat]]
            mass = [x["mean_mass"] for x in its if x["mean_mass"] is not None]
            table[name][cat] = {"mean": cat_mean(prim[cat]), "ci": ci(d), "secondary": cat_mean(sec.get(cat, {})),
                                "mass": float(np.mean(mass)) if mass else None,
                                "parse": float(np.mean([x["parse_rate"] for x in its])),
                                "answer_tag": float(np.mean([x["answer_tag_rate"] for x in its])),
                                "truncated": float(np.mean([x["truncated_share"] for x in its])),
                                "reps": int(np.mean([len(v) for v in prim[cat].values()]))}
            draws[name][cat] = d
        if r.get("afterlife"):
            table[name]["afterlife"] = {"mean": r["afterlife"]["ev"], "mass": r["afterlife"]["mass"]}
    return {"table": table, "draws": draws}


def diff(draws: dict, a: str, b: str, cat: str, table: dict) -> dict | None:
    if cat not in draws.get(a, {}) or cat not in draws.get(b, {}):
        return None
    d = draws[a][cat] - draws[b][cat]
    return {"diff": table[a][cat]["mean"] - table[b][cat]["mean"], "ci": ci(d)}


# ------------------------------------------------------------------------------------------- behavior


def behavior_table(judgments: list[dict], items_meta: dict, seed: int = 0) -> dict:
    """{cond: {family: {rate, ci, n, unparsed}}}, plus who_benefits by beneficiary."""
    rng = np.random.default_rng(seed)
    by = {}
    for j in judgments:
        cond = j["intervention"]
        fam = j["family"]
        keys = [fam]
        if fam == "who_benefits":
            ben = items_meta.get(j["item_id"], {}).get("beneficiary")
            if ben:
                keys.append(f"who_benefits:{ben}")
        for k in keys:
            by.setdefault(cond, {}).setdefault(k, {}).setdefault(j["item_id"], []).append(j["primary"])
    out, draws = {}, {}
    for cond, fams in by.items():
        out[cond], draws[cond] = {}, {}
        for fam, items in fams.items():
            ids = sorted(items)
            vals = [[float(v) for v in items[i] if v is not None] for i in ids]
            flat = [v for vs in vals for v in vs]
            n_unp = sum(v is None for i in ids for v in items[i])
            per = [np.mean(v) if v else np.nan for v in vals]
            per = np.asarray(per)
            idx = rng.integers(0, len(per), size=(B, len(per)))
            d = np.nanmean(per[idx], axis=1)
            out[cond][fam] = {"rate": float(np.mean(flat)) if flat else None, "ci": ci(d), "n": len(flat),
                              "items": len(ids), "unparsed": n_unp}
            draws[cond][fam] = {"ids": ids, "per": per}
    return {"table": out, "draws": draws}


def behavior_diff(bd: dict, a: str, b: str, fam: str, seed: int = 0) -> dict | None:
    """Paired by item: resample items, difference of the two conditions' item means."""
    da, db = bd["draws"].get(a, {}).get(fam), bd["draws"].get(b, {}).get(fam)
    if not da or not db:
        return None
    common = sorted(set(da["ids"]) & set(db["ids"]))
    pa = np.asarray([da["per"][da["ids"].index(i)] for i in common])
    pb = np.asarray([db["per"][db["ids"].index(i)] for i in common])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(B, len(common)))
    d = np.nanmean(pa[idx] - pb[idx], axis=1)
    return {"diff": float(np.nanmean(pa - pb)), "ci": ci(d), "items": len(common)}


# ------------------------------------------------------------------------------------------- ToM


def tom_table(results: dict, seed: int = 0) -> dict:
    out = {}
    for name, r in results.items():
        if "tom" in r:
            out[name] = {k: {"acc": 100.0 * float(np.mean(v)), "n": len(v)} for k, v in r["tom"].items()}
    return out


def tom_diff(results: dict, a: str, b: str, task: str, seed: int = 0) -> dict | None:
    if "tom" not in results.get(a, {}) or "tom" not in results.get(b, {}):
        return None
    xa, xb = np.asarray(results[a]["tom"][task], float), np.asarray(results[b]["tom"][task], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xa), size=(B, len(xa)))
    d = 100.0 * (xa[idx] - xb[idx]).mean(1)
    return {"diff": 100.0 * float((xa - xb).mean()), "ci": ci(d)}


# ------------------------------------------------------------------------------------------- report


CONTRASTS = [("nofix_deny@32", "base"), ("nofix_deny@32", "aff@32"), ("nofix_deny@32", "k5aff@32"),
             ("nofix_deny@32", "random@32"), ("nofix_topic@32", "base"), ("nofix_topic@32", "random@32"),
             ("aff@32", "base"), ("k5aff@32", "base"), ("random@32", "base"), ("nofix_deny@16", "base"),
             ("nofix_deny@16", "aff@16"), ("nofix_topic@16", "base"), ("nofix_deny@32", "k5aff@16"),
             ("nofix_deny@32", "aff@16"), ("nofix_orth@32", "base"), ("nofix_orth@32", "random@32")]


def load_results(d: Path) -> dict:
    out = {}
    for f in sorted((d / "full").glob("*.json")):
        r = json.loads(f.read_text())
        out[r["cond"]["name"]] = r
    return out


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def build(d: Path = RESULTS) -> dict:
    results = load_results(d)
    inst = instruments_table(results)
    tom = tom_table(results)
    s = {"instruments": inst["table"], "tom": tom, "contrasts": {}}
    jpath = d / "behavior" / "judgments.jsonl"
    bd = None
    if jpath.exists():
        js = [json.loads(x) for x in jpath.read_text().splitlines() if x.strip()]
        meta = json.loads((d / "behavior" / "items_meta.json").read_text())
        bd = behavior_table(js, meta)
        s["behavior"] = bd["table"]
    for a, b in CONTRASTS:
        if a not in results or b not in results:
            continue
        c = {"instruments": {cat: diff(inst["draws"], a, b, cat, inst["table"]) for cat in CATS},
             "tom": {t: tom_diff(results, a, b, t) for t in ("hitom", "mmlu")}}
        if bd:
            fams = sorted({f for cond in (a, b) for f in bd["table"].get(cond, {})})
            c["behavior"] = {f: behavior_diff(bd, a, b, f) for f in fams}
        s["contrasts"][f"{a} - {b}"] = c
    return s


def markdown(s: dict) -> str:
    L = ["# noself: tables (sighting, explore track; Claude wrote this)", ""]
    conds = list(s["instruments"])
    L.append("## Instruments (primary EV; [95% CI, reps resampled]; secondary parsed in parentheses)")
    L.append("")
    L.append("| condition | " + " | ".join(CATS + ["afterlife", "mass", "parse"]) + " |")
    L.append("|" + "---|" * (len(CATS) + 4))
    for c in conds:
        t = s["instruments"][c]
        cells = []
        for cat in CATS:
            x = t.get(cat)
            cells.append("" if not x else f"{x['mean']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}] ({x['secondary']:.2f})")
        af = t.get("afterlife", {})
        masses = [t[k]["mass"] for k in CATS if k in t and t[k]["mass"] is not None]
        parses = [t[k]["parse"] for k in CATS if k in t]
        cells += [fmt(af.get("mean")), fmt(min(masses) if masses else None), fmt(min(parses) if parses else None)]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "mass and parse: the lowest category mean of the allowed-answer mass and the parse rate.", ""]
    if s.get("tom"):
        L += ["## HI-ToM and MMLU (option logits, %)", "", "| condition | HI-ToM | MMLU |", "|---|---|---|"]
        for c, t in s["tom"].items():
            L.append(f"| {c} | {t['hitom']['acc']:.1f} (n {t['hitom']['n']}) | {t['mmlu']['acc']:.1f} (n {t['mmlu']['n']}) |")
        L.append("")
    if s.get("behavior"):
        fams = sorted({f for t in s["behavior"].values() for f in t})
        L += ["## Behavior: self-serving rate (scout's judge; [95% CI, items resampled])", "",
              "| condition | " + " | ".join(fams) + " |", "|" + "---|" * (len(fams) + 1)]
        for c, t in s["behavior"].items():
            L.append(f"| {c} | " + " | ".join(
                f"{t[f]['rate']:.2f} [{t[f]['ci'][0]:.2f}, {t[f]['ci'][1]:.2f}] n{t[f]['n']}" if f in t and t[f]["rate"] is not None
                else "" for f in fams) + " |")
        L.append("")
    L += ["## Contrasts (difference [95% CI])", ""]
    for name, c in s["contrasts"].items():
        parts = [f"{k} {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for k, v in c["instruments"].items() if v]
        parts += [f"{k} {v['diff']:+.1f}pp [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]" for k, v in c["tom"].items() if v]
        parts += [f"{k} {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for k, v in (c.get("behavior") or {}).items() if v]
        L.append(f"- **{name}**: " + "; ".join(parts))
    return "\n".join(L) + "\n"


def main(argv):
    d = Path(argv[0]) if argv else RESULTS
    s = build(d)
    (d / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    (d / "tables.md").write_text(markdown(s))
    print(markdown(s))


if __name__ == "__main__":
    main(sys.argv[1:])
