"""The lab's one stats module (clean-lab step 3). Claude (Opus 5.5) wrote this file, 3 Nov 2026, from the copies it
replaces; each section names where its code came from, and tests/test_stats_core.py checks every one against the
copies' outputs at 264609d (tests/fixtures/stats_golden.json). numpy and the stdlib only (the venv has no scipy).

Sections:
  proportions     wilson (act1, selfmap and h4lite's cal_judges variants are its arguments)
  agreement       cohen_kappa on two binary label lists (scout rejudge/calibrate); kappa_counts and kappa_counts_ci
                  on 2x2 counts, exact in fractions (h4lite cal_judges)
  unit bootstrap  the harness's item bootstrap and its tests (lab/harness/stats.py, moved here unchanged): Est,
                  mean_ci, paired_diff (the paired bootstrap), min_effect, rise, tost, noninferiority,
                  relative_noninf, composite, composite_level
  percent units   scout's rate / paired_diff / pooled_diff / joint_mean in points (lab/explore/scout/stats.py):
                  pct_rate, pct_paired_diff, pct_pooled_diff, pct_joint_mean, unit_deltas, pct_summary
  pair tables     pooled rates over {pair: (hits, n)} tables with a bootstrap over pairs (lab/explore/act1/act1_core):
                  pair_rate, pair_rate_diff, pair_rate_diff_2level (the two-level bootstrap: runs, then pairs)
  python random   bootstraps drawn with random.Random (bench metrics, agentic_release): bootstrap_ci,
                  two_sample_boot_diff
  permutation     perm_p_diff (two independent samples) and perm_p_paired (sign flips), exact when small

Seeds: each function keeps the generator its source used (numpy default_rng(seed), default_rng([seed, crc32(salt)])
or random.Random(seed)), so a re-export gives the same draws as the copy it replaces. The generators are not mixed
inside one function.
"""

from __future__ import annotations

import itertools
import math
import random
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

REPS, SEED, ALPHA = 2000, 0, 0.05
Z95 = 1.959963984540054  # the two-sided 95% normal quantile; act1 and selfmap used 1.96

# ------------------------------------------------------------------------------------------------ proportions


def wilson(k: int, n: int, z: float = 1.96, clip: bool = False) -> list[float] | None:
    """The Wilson interval of k of n, as proportions [lo, hi]; None when n is 0. clip bounds it to [0, 1] (the
    unclipped form can dip a rounding error below 0 at k = 0)."""
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = centre - half, centre + half
    return [max(0.0, lo), min(1.0, hi)] if clip else [lo, hi]


# ------------------------------------------------------------------------------------------------ agreement


def cohen_kappa(a: Sequence[bool], b: Sequence[bool], eps: float = 1e-12) -> float | None:
    """Cohen's kappa for two binary raters over the same items; None when undefined (no items, or chance agreement
    within eps of 1: both raters constant and equal)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 - eps else (po - pe) / (1 - pe)


def kappa_counts(a: int, b: int, c: int, d: int) -> Fraction | None:
    """Cohen's kappa from 2x2 counts (a both, b first only, c second only, d neither), exact; None when chance
    agreement is 1 (neither label varies)."""
    n = a + b + c + d
    if not n:
        return None
    po = Fraction(a + d, n)
    p1, p2 = Fraction(a + b, n), Fraction(a + c, n)
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def kappa_counts_ci(cells: list, reps: int = REPS, seed: int = SEED) -> dict:
    """Percentile 95% interval of kappa over rows resampled with replacement (random.Random(seed)); cells code each row
    0 both, 1 first only, 2 second only, 3 neither. Resamples where kappa is undefined are counted and left out."""
    if not cells:
        return {"ci95": None, "reps": reps, "seed": seed, "undefined": 0}
    rng = random.Random(seed)
    vals, undefined = [], 0
    for _ in range(reps):
        cnt = Counter(rng.choices(cells, k=len(cells)))
        k = kappa_counts(cnt[0], cnt[1], cnt[2], cnt[3])
        if k is None:
            undefined += 1
        else:
            vals.append(float(k))
    vals.sort()
    ci = [vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]] if vals else None
    return {"ci95": [round(x, 4) for x in ci] if ci else None, "reps": reps, "seed": seed, "undefined": undefined}


# ------------------------------------------------------------------------------------------------ unit bootstrap
# From lab/harness/stats.py (spec v2 section 4), unchanged. Units: a measure's values come as {unit_id: value}, one
# value per unit. Every interval resamples units with replacement (REPS draws, seeded by (seed, salt)), never samples
# within a unit. A paired difference uses the units both conditions share, resampled together. diff = A - B;
# improvement = diff oriented so that positive is better. Composites average "badness" on the 0-100 scale.
# Tests (alpha one-sided; the bootstrap p is (k + 1) / (B + 1)):
#   min_effect      point improvement >= min_effect and its one-sided p against 0 < alpha (mode "point"), or the
#                   one-sided lower bound itself >= min_effect (mode "margin")
#   tost            the (1 - 2 alpha) interval of diff lies inside (-margin, +margin)
#   noninferiority  the one-sided (1 - alpha) bound on the bad side lies inside the margin
#   relative_noninf as non-inferiority, on diff / |mean of B|


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


# ------------------------------------------------------------------------------------------------ percent units
# From lab/explore/scout/stats.py: means over units in points (x100) with 95% percentile intervals, numpy
# default_rng(seed) draws; one unit gives no interval (lo = hi = est).


def pct_summary(est: float, boots: np.ndarray | None, n: int, scale: float = 100.0, **extra) -> dict:
    if boots is None or n < 2:
        lo = hi = est
    else:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": round(scale * float(est), 2), "lo": round(scale * float(lo), 2), "hi": round(scale * float(hi), 2),
            "n": n, **extra}


def pct_rate(units: dict, reps: int = REPS, seed: int = SEED) -> dict | None:
    """The mean over units, in %, with its 95% bootstrap interval."""
    if not units:
        return None
    v = np.array([units[u] for u in sorted(units)], dtype=float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1) if len(v) > 1 else None
    return pct_summary(v.mean(), boots, len(v))


def pct_paired_diff(a: dict, b: dict, reps: int = REPS, seed: int = SEED) -> dict | None:
    """mean(a - b) over the units both have, in percentage points, with its 95% bootstrap interval."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = np.array([a[u] - b[u] for u in common], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1) if len(d) > 1 else None
    return pct_summary(d.mean(), boots, len(d))


def unit_deltas(pairs: list[tuple[dict, dict]]) -> dict:
    """Per unit, the mean over pairs (models, where the model has both cells) of a - b."""
    acc: dict = {}
    for a, b in pairs:
        for u in set(a) & set(b):
            acc.setdefault(u, []).append(a[u] - b[u])
    return {u: sum(v) / len(v) for u, v in acc.items()}


def pct_pooled_diff(pairs: list[tuple[dict, dict]], reps: int = REPS, seed: int = SEED) -> dict | None:
    """The change pooled over models: per unit, the models' mean change; then a bootstrap over units (all models'
    values for a unit move together). "k" counts the pairs that share a unit."""
    deltas = unit_deltas(pairs)
    if not deltas:
        return None
    s = pct_paired_diff(deltas, {u: 0.0 for u in deltas}, reps, seed)
    s["k"] = sum(1 for a, b in pairs if set(a) & set(b))
    return s


def pct_joint_mean(deltas_by_col: dict, reps: int = REPS, seed: int = SEED, **extra) -> dict | None:
    """The mean over columns of each column's mean unit delta, with a joint bootstrap (each column's units resampled,
    then the column means averaged)."""
    cols = [c for c, d in deltas_by_col.items() if d]
    if not cols:
        return None
    rng = np.random.default_rng(seed)
    ests = [np.mean(list(deltas_by_col[c].values())) for c in cols]
    boots = np.zeros(reps)
    for c in cols:
        v = np.array([deltas_by_col[c][u] for u in sorted(deltas_by_col[c])], dtype=float)
        boots += v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1)
    boots /= len(cols)
    n = min(len(deltas_by_col[c]) for c in cols)
    return pct_summary(float(np.mean(ests)), boots, n, columns=len(cols), **extra)


# ------------------------------------------------------------------------------------------------ pair tables
# From lab/explore/act1/act1_core.py. A table is one run's {pair: (hits, n)}; a pooled rate is sum(hits) / sum(n)
# over pairs (and runs); the bootstrap resamples pairs, one draw shared by every run, so comparisons stay paired.


def _hn(tables: list[dict], pairs: list) -> tuple[np.ndarray, np.ndarray]:
    """[runs x pairs] hits and counts (a pair a run lacks counts (0, 0))."""
    H = np.array([[t.get(p, (0, 0))[0] for p in pairs] for t in tables], float)
    N = np.array([[t.get(p, (0, 0))[1] for p in pairs] for t in tables], float)
    return H, N


def _pct95(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def pair_rate(tables: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Pooled rate over one or more runs (seeds) on the same pairs, with a bootstrap over pairs (the same draw for
    every run)."""
    pairs = sorted(set().union(*[set(t) for t in tables]))
    if not pairs:
        return {"rate": None, "ci95": [None, None], "n": 0, "pairs": 0}
    H, N = (m.sum(0) for m in _hn(tables, pairs))  # pooled over runs
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), (n_boot, len(pairs)))
    b = H[idx].sum(1) / np.maximum(N[idx].sum(1), 1)
    return {"rate": float(H.sum() / max(N.sum(), 1)), "ci95": _pct95(b), "n": int(N.sum()), "pairs": len(pairs)}


def pair_rate_diff(a: list[dict], b: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Pooled rate of a minus pooled rate of b, paired bootstrap over the shared pairs."""
    pairs = sorted(set().union(*[set(t) for t in a + b]))
    if not pairs:
        return {"diff": None, "ci95": [None, None]}
    Ha, Na = (m.sum(0) for m in _hn(a, pairs))  # pooled over runs
    Hb, Nb = (m.sum(0) for m in _hn(b, pairs))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), (n_boot, len(pairs)))
    d = Ha[idx].sum(1) / np.maximum(Na[idx].sum(1), 1) - Hb[idx].sum(1) / np.maximum(Nb[idx].sum(1), 1)
    point = Ha.sum() / max(Na.sum(), 1) - Hb.sum() / max(Nb.sum(), 1)
    return {"diff": float(point), "ci95": _pct95(d)}


def pair_rate_diff_2level(a: list[dict], b: list[dict], n_boot: int = 4000, seed: int = 0) -> dict:
    """Pooled rate of arm a minus arm b with a two-level bootstrap: each draw resamples the runs (seeds) within each
    arm, then the pairs (one draw shared by every run, so the comparison stays paired over items). With two seeds per
    arm the seed level is coarse (three distinct mixes), so it widens the interval only roughly."""
    pairs = sorted(set().union(*[set(t) for t in a + b]))
    if not pairs or not a or not b:
        return {"diff": None, "ci95": [None, None]}
    (Ha, Na), (Hb, Nb) = _hn(a, pairs), _hn(b, pairs)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        ra, rb = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
        idx = rng.integers(0, len(pairs), len(pairs))
        ha, na = Ha[ra][:, idx].sum(), Na[ra][:, idx].sum()
        hb, nb = Hb[rb][:, idx].sum(), Nb[rb][:, idx].sum()
        d[i] = ha / max(na, 1) - hb / max(nb, 1)
    point = Ha.sum() / max(Na.sum(), 1) - Hb.sum() / max(Nb.sum(), 1)
    return {"diff": float(point), "ci95": _pct95(d),
            "per_run_a": [float(h / max(n, 1)) for h, n in zip(Ha.sum(1), Na.sum(1))],
            "per_run_b": [float(h / max(n, 1)) for h, n in zip(Hb.sum(1), Nb.sum(1))]}


# ------------------------------------------------------------------------------------------------ python random
# Bootstraps drawn with random.Random(seed), percentile by sorted index (src/anatta/bench/metrics.py and
# lab/explore/agentic_release/run.py).


def bootstrap_ci(values: list[float], stat: Callable[[list[float]], float] | None = None,
                 n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap of stat (default: the mean)."""
    stat = stat or (lambda xs: sum(xs) / len(xs))
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(values) for _ in values]) for _ in range(n))
    return stat(values), boots[int(n * alpha / 2)], boots[int(n * (1 - alpha / 2)) - 1]


def two_sample_boot_diff(a: list[float], b: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """(mean(a) - mean(b), lo, hi): each sample resampled on its own (unpaired), 95% percentile interval."""
    rng = random.Random(seed)
    d = sum(a) / len(a) - sum(b) / len(b)
    ds = []
    for _ in range(reps):
        sa = [a[rng.randrange(len(a))] for _ in a]
        sb = [b[rng.randrange(len(b))] for _ in b]
        ds.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    ds.sort()
    return d, ds[int(0.025 * reps)], ds[int(0.975 * reps) - 1]


# ------------------------------------------------------------------------------------------------ permutation
# New in step 3. Two-sided, on the difference of means; exact (every relabelling, p = hits / total) when the count
# of relabellings is at most max_exact, else Monte Carlo with numpy default_rng(seed) and p = (hits + 1) / (reps + 1).
# A relabelling counts as at least as extreme when |stat| >= |observed| - 1e-12.

_TOL = 1e-12


def perm_p_diff(a: Sequence[float], b: Sequence[float], reps: int = 10_000, seed: int = 0,
                max_exact: int = 200_000) -> dict:
    """Two independent samples: is mean(a) - mean(b) larger in size than relabellings of the pooled values give?"""
    x = np.asarray(list(a) + list(b), dtype=float)
    na, n = len(a), len(a) + len(b)
    if na == 0 or na == n:
        return {"diff": None, "p": None, "exact": None, "relabellings": 0}
    obs = float(x[:na].mean() - x[na:].mean())
    total, s = math.comb(n, na), x.sum()

    def stat(sa: float) -> float:
        return sa / na - (s - sa) / (n - na)

    if total <= max_exact:
        hits = sum(abs(stat(float(x[list(c)].sum()))) >= abs(obs) - _TOL for c in itertools.combinations(range(n), na))
        return {"diff": obs, "p": hits / total, "exact": True, "relabellings": total}
    rng = np.random.default_rng(seed)
    sums = np.array([x[rng.permutation(n)[:na]].sum() for _ in range(reps)])
    hits = int(np.sum(np.abs(sums / na - (s - sums) / (n - na)) >= abs(obs) - _TOL))
    return {"diff": obs, "p": (hits + 1) / (reps + 1), "exact": False, "relabellings": reps}


def perm_p_paired(d: Sequence[float], reps: int = 10_000, seed: int = 0, max_exact: int = 2**16) -> dict:
    """Paired differences (a - b per unit): the sign-flip test of mean(d) against 0."""
    x = np.asarray(list(d), dtype=float)
    n = x.size
    if n == 0:
        return {"mean": None, "p": None, "exact": None, "relabellings": 0}
    obs = float(x.mean())
    if 2**n <= max_exact:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        hits = int(np.sum(np.abs(signs @ x / n) >= abs(obs) - _TOL))
        return {"mean": obs, "p": hits / 2**n, "exact": True, "relabellings": 2**n}
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(reps, n))
    hits = int(np.sum(np.abs(signs @ x / n) >= abs(obs) - _TOL))
    return {"mean": obs, "p": (hits + 1) / (reps + 1), "exact": False, "relabellings": reps}
