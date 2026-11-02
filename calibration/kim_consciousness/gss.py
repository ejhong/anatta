"""K5's GSS leg (recorded, never graded): the human response distribution of each Table S9 item from the public GSS
cumulative file, survey-weighted under the paper's year filters, and KL(human || model) with Laplace smoothing
(alpha = 0.5), so that Delta KL = KL_base - KL_condition (positive = closer to humans), as in the paper's Methods.

What the paper leaves out and is chosen here (README.md): the option wording is the GSS file's value labels for the
valid (non-missing) codes, in code order; the question is Table S9's text, loaded from the instruments data file (13 truncated there); the model's option
distribution is read from next-token logits at the answer (one deterministic pass, "<answer>" prefilled), and is
smoothed as if it were 100 reps (the paper's count): (100 p + alpha) / (100 + alpha K); the human weighted
distribution is smoothed on its raw n: (n p + alpha) / (n + alpha K). The weight is wtssps where present and
non-missing, else wtssall, else 1 (each item records which it used).

Pure Python except human_distributions, which reads the Stata file with pandas (on Modal only).
"""

from __future__ import annotations

import math

from gss_items import ROWS, YEAR_MIN

WEIGHTS = ("wtssps", "wtssall")


def item_domains() -> list[dict]:
    """One row per (variable, domain) in Table S9's order: {var, domain, truncated, year_min} (the question text is
    in the instruments data file: question())."""
    return [{"var": v, "domain": d, "truncated": t, "year_min": YEAR_MIN[d]} for v, d, t in ROWS]


def question(var: str, domain: str) -> str:
    import instruments

    return instruments.s(f"gss.{var}|{domain}")


def weighted_distribution(codes: list[float], weights: list[float], valid: list[int]) -> tuple[list[float], int]:
    """The weighted share of each valid code (in the order given) among rows holding a valid code, and that row
    count. Rows with a missing code or weight are dropped."""
    tot = {c: 0.0 for c in valid}
    n = 0
    for c, w in zip(codes, weights):
        if c is None or w is None or (isinstance(c, float) and math.isnan(c)) or (isinstance(w, float) and math.isnan(w)):
            continue
        ci = int(c)
        if ci in tot:
            tot[ci] += float(w)
            n += 1
    s = sum(tot.values())
    if s <= 0:
        return [0.0] * len(valid), 0
    return [tot[c] / s for c in valid], n


def smooth(p: list[float], n: float, alpha: float = 0.5) -> list[float]:
    k = len(p)
    return [(n * x + alpha) / (n + alpha * k) for x in p]


def kl(p: list[float], q: list[float]) -> float:
    """KL(p || q) in nats; both must be strictly positive where p is (smooth first)."""
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)


def kl_human_model(p_human: list[float], n_human: int, p_model: list[float], alpha: float = 0.5,
                   model_reps: int = 100) -> float:
    return kl(smooth(p_human, n_human, alpha), smooth(p_model, model_reps, alpha))


def delta_kl(human: dict, model: dict, alpha: float = 0.5, reps: int = 100) -> dict:
    """Per (var, domain): KL in each condition and Delta KL of ablated and steered against base; then per-domain and
    pooled means (plain means; the paper's fixed-effects regression with CR1 errors is for the pooled three models,
    and one model's mean is what a calibration on one model can show). human[key] = {p, n}; model[cond][key] = p."""
    rows = []
    for key, h in human.items():
        if not h.get("n"):
            continue
        kls = {c: kl_human_model(h["p"], h["n"], model[c][key], alpha, reps) for c in model if key in model[c]}
        if "base" not in kls:
            continue
        var, dom = key.split("|")
        rows.append({"var": var, "domain": dom, "kl": kls,
                     "dkl": {c: kls["base"] - v for c, v in kls.items() if c != "base"}})
    out = {"rows": rows, "by_domain": {}, "pooled": {}}
    conds = sorted({c for r in rows for c in r["dkl"]})
    for c in conds:
        vals = [r["dkl"][c] for r in rows if c in r["dkl"]]
        out["pooled"][c] = sum(vals) / len(vals) if vals else None
        for dom in sorted({r["domain"] for r in rows}):
            dv = [r["dkl"][c] for r in rows if r["domain"] == dom and c in r["dkl"]]
            out["by_domain"].setdefault(dom, {})[c] = sum(dv) / len(dv) if dv else None
    a, s = out["pooled"].get("ablated"), out["pooled"].get("steered")
    out["steer_over_ablate"] = (s / a) if a not in (None, 0) and s is not None else None
    return out


def human_distributions(dta_path: str) -> dict:
    """{"var|domain": {var, domain, codes, labels, p, n, weight, year_min}} for every Table S9 row whose variable is
    in the file, plus {"absent": [vars]}. Reads only the needed columns."""
    import pandas as pd

    rows = item_domains()
    with pd.io.stata.StataReader(dta_path) as r:
        cols = set(r.variable_labels().keys())
        labels = r.value_labels()
        lab_of = {}
        try:
            lab_of = dict(zip(r._varlist, r._lbllist))  # variable -> its value-label set name
        except AttributeError:
            pass
    want = sorted({x["var"] for x in rows if x["var"] in cols} | {"year"} | {w for w in WEIGHTS if w in cols})
    df = pd.read_stata(dta_path, columns=want, convert_categoricals=False, convert_missing=False)
    out, absent = {}, sorted({x["var"] for x in rows if x["var"] not in cols})
    for x in rows:
        v = x["var"]
        if v not in cols:
            continue
        d = df if x["year_min"] is None else df[df["year"] >= x["year_min"]]
        weight = next((w for w in WEIGHTS if w in d.columns and d[w].notna().any()), None)
        wser = d[weight].where(d[weight].notna(), None) if weight else [1.0] * len(d)
        vl = labels.get(lab_of.get(v, v), {})
        present = sorted({int(c) for c in d[v].dropna().unique()})
        valid = [c for c in present if c >= 0]
        names = [str(vl.get(c, c)) for c in valid]
        p, n = weighted_distribution(list(d[v]), list(wser), valid)
        out[f"{v}|{x['domain']}"] = {"var": v, "domain": x["domain"], "codes": valid, "labels": names, "p": p,
                                     "n": n, "weight": weight, "year_min": x["year_min"]}
    return {"items": out, "absent": absent, "rows_in_file": int(len(df))}
