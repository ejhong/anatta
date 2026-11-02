"""The harness's statistics (spec v2 section 4): item-bootstrap paired differences, a one-sided minimum-effect test
against each control, TOST equivalence, non-inferiority, and fixed-weight composites. Claude (Opus 5.5) wrote this file,
2 Nov 2026. numpy only (the venv has no scipy).

Units. A measure's values come as {unit_id: value}, one value per unit (an item's mean over its samples and reps). Every
interval resamples units with replacement (REPS draws, seeded), never samples within a unit, as scout's stats do. A
paired difference uses the units both conditions share, resampled together.

Signs. diff = condition A - condition B (candidate - control). improvement = diff oriented so that positive is better:
-diff for a lower-is-better measure, +diff for higher. Composites average "badness" on the 0-100 scale (higher-better
measures enter as their negation), so a composite diff below 0 is an improvement.

Tests (alpha one-sided, default 0.05; the bootstrap p is (k + 1) / (B + 1)):
  min_effect      passes iff the point improvement >= min_effect and the improvement's one-sided p against 0 < alpha
                  (mode "point"); mode "margin" demands the one-sided lower bound itself >= min_effect (stricter).
  tost            passes iff the (1 - 2 alpha) interval of diff lies inside (-margin, +margin).
  noninferiority  passes iff the one-sided (1 - alpha) bound on the bad side lies inside the margin.
  relative_noninf as non-inferiority, on diff / |mean of B| (reply length).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

REPS, SEED, ALPHA = 2000, 0, 0.05


@dataclass(frozen=True)
class Est:
    est: float
    ci: tuple
    n: int
    boots: np.ndarray | None = None

    def as_dict(self, scale: float = 1.0, digits: int = 3) -> dict:
        return {"est": round(self.est * scale, digits), "ci": [round(x * scale, digits) for x in self.ci], "n": self.n}


def _rng(seed: int, salt: str = "") -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(salt.encode())])


def _ci(boots: np.ndarray, lo: float = 0.025, hi: float = 0.975) -> tuple:
    return (float(np.quantile(boots, lo)), float(np.quantile(boots, hi)))


def mean_ci(values: dict, reps: int = REPS, seed: int = SEED, salt: str = "") -> Est | None:
    x = np.array([float(values[k]) for k in sorted(values)], dtype=float)
    if x.size == 0:
        return None
    idx = _rng(seed, salt).integers(0, x.size, size=(reps, x.size))
    boots = x[idx].mean(axis=1)
    return Est(float(x.mean()), _ci(boots), int(x.size), boots)


def paired_diff(a: dict, b: dict, reps: int = REPS, seed: int = SEED, salt: str = "") -> Est | None:
    """mean(a - b) over shared units, with an item-bootstrap interval; the draws depend on (seed, salt) only, so every
    comparison of one measure with the same salt resamples the same unit positions."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    d = np.array([float(a[k]) - float(b[k]) for k in keys])
    idx = _rng(seed, salt).integers(0, d.size, size=(reps, d.size))
    boots = d[idx].mean(axis=1)
    return Est(float(d.mean()), _ci(boots), int(d.size), boots)


def orient(better: str) -> float:
    if better == "lower":
        return -1.0
    if better == "higher":
        return 1.0
    raise ValueError("a test needs a better direction (lower | higher)")


def p_one_sided(boots: np.ndarray, threshold: float = 0.0) -> float:
    """Bootstrap p for H0: parameter <= threshold (boots are draws of a quantity where larger is the alternative)."""
    k = int(np.sum(boots <= threshold))
    return (k + 1) / (boots.size + 1)


def min_effect(diff: Est, better: str, min_effect: float, alpha: float = ALPHA, mode: str = "point",
               scale: float = 1.0) -> dict:
    """diff in unit values; min_effect in points (unit x scale)."""
    imp = orient(better) * diff.boots * scale
    est = orient(better) * diff.est * scale
    lb = float(np.quantile(imp, alpha))
    p0 = p_one_sided(imp, 0.0)
    if mode == "point":
        ok = est >= min_effect and p0 < alpha
    elif mode == "margin":
        ok = lb >= min_effect
    else:
        raise ValueError("mode: point | margin")
    return {"test": "min_effect", "mode": mode, "improvement": round(est, 3), "lower_bound": round(lb, 3),
            "p_vs_0": round(p0, 4), "p_vs_min": round(p_one_sided(imp, min_effect), 4), "min_effect": min_effect,
            "alpha": alpha, "pass": bool(ok)}


def rise(diff: Est, better: str, alpha: float = ALPHA, scale: float = 1.0) -> dict:
    imp = orient(better) * diff.boots * scale
    p0 = p_one_sided(imp, 0.0)
    return {"test": "rise", "improvement": round(orient(better) * diff.est * scale, 3),
            "lower_bound": round(float(np.quantile(imp, alpha)), 3), "p_vs_0": round(p0, 4), "alpha": alpha,
            "pass": bool(p0 < alpha)}


def tost(diff: Est, margin: float, alpha: float = ALPHA, scale: float = 1.0) -> dict:
    b = diff.boots * scale
    lo, hi = float(np.quantile(b, alpha)), float(np.quantile(b, 1 - alpha))
    p_low = p_one_sided(b, -margin)        # H0: diff <= -margin
    p_high = p_one_sided(-b, -margin)      # H0: diff >= +margin
    return {"test": "tost", "diff": round(diff.est * scale, 3), "ci_90": [round(lo, 3), round(hi, 3)],
            "margin": margin, "p": round(max(p_low, p_high), 4), "alpha": alpha,
            "pass": bool(lo > -margin and hi < margin)}


def noninferiority(diff: Est, better: str, margin: float, alpha: float = ALPHA, scale: float = 1.0) -> dict:
    """Passes iff the candidate is shown no worse than margin (points) on the measure's bad side."""
    bad = -orient(better) * diff.boots * scale  # positive = worse
    bound = float(np.quantile(bad, 1 - alpha))
    return {"test": "noninferiority", "worse_by": round(-orient(better) * diff.est * scale, 3),
            "upper_bound": round(bound, 3), "margin": margin, "p": round(p_one_sided(-bad, -margin), 4),
            "alpha": alpha, "pass": bool(bound < margin)}


def relative_noninf(diff: Est, ref_mean: float, margin: float, alpha: float = ALPHA, side: str = "both") -> dict:
    """Relative change diff / |ref_mean| within margin (a fraction, e.g. 0.20). side: lower (only shrinking is bad),
    higher, or both (two one-sided bounds: shown within +-margin)."""
    if ref_mean == 0:
        return {"test": "relative", "pass": False, "reason": "reference mean is 0"}
    r = diff.boots / abs(ref_mean)
    lo, hi = float(np.quantile(r, alpha)), float(np.quantile(r, 1 - alpha))
    ok = {"lower": lo > -margin, "higher": hi < margin, "both": lo > -margin and hi < margin}[side]
    return {"test": "relative", "side": side, "change": round(diff.est / abs(ref_mean), 4),
            "ci_90": [round(lo, 4), round(hi, 4)], "margin": margin, "alpha": alpha, "pass": bool(ok)}


def composite(parts: dict, reps: int = REPS, seed: int = SEED) -> dict | None:
    """Fixed-weight composite of paired differences, in badness points (lower = better).
    parts = {measure_id: {"a": units, "b": units, "better": lower|higher, "weight": w, "pct": factor to 0-100}}.
    Each measure is resampled on its own units with its own salt (the measure id), the same draws as paired_diff with
    that salt, so the composite's interval is the weighted sum of the parts' bootstrap draws."""
    tot_w, est, boots, used = 0.0, 0.0, np.zeros(reps), {}
    for mid, p in sorted(parts.items()):
        d = paired_diff(p["a"], p["b"], reps, seed, salt=mid)
        if d is None or not p.get("weight"):
            continue
        k = -orient(p["better"]) * p["pct"] * p["weight"]  # to badness points, weighted
        est += k * d.est
        boots += k * d.boots
        tot_w += p["weight"]
        used[mid] = {"weight": p["weight"], "n": d.n, "badness_diff": round(-orient(p["better"]) * p["pct"] * d.est, 3)}
    if tot_w == 0:
        return None
    e = Est(est / tot_w, _ci(boots / tot_w), sum(u["n"] for u in used.values()), boots / tot_w)
    return {"est": e, "parts": used, "weights_total": tot_w}


def composite_level(units_by_measure: dict, better: dict, weights: dict, pct: dict) -> float | None:
    """A condition's own composite badness level (0-100): the weighted mean of each measure's level as badness
    (higher-better measures enter as 100 - level)."""
    tot, w = 0.0, 0.0
    for mid, u in units_by_measure.items():
        if not u or not weights.get(mid):
            continue
        lvl = float(np.mean(list(u.values()))) * pct[mid]
        tot += weights[mid] * (lvl if better[mid] == "lower" else 100.0 - lvl)
        w += weights[mid]
    return tot / w if w else None
