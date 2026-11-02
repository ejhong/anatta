"""stats.py on synthetic data. Claude (Opus 5.5) wrote this file, 2 Nov 2026."""

import numpy as np
import pytest

from conftest import binary_units
from harness import stats as S


def test_mean_ci_covers_and_is_seeded():
    v = {f"u{i}": float(i % 2) for i in range(200)}
    a, b = S.mean_ci(v, seed=1), S.mean_ci(v, seed=1)
    assert a.est == pytest.approx(0.5)
    assert a.ci[0] < 0.5 < a.ci[1]
    assert a.ci == b.ci


def test_paired_diff_uses_shared_units_only():
    a = {"x": 1.0, "y": 1.0, "z": 0.0}
    b = {"x": 0.0, "y": 1.0, "w": 5.0}
    d = S.paired_diff(a, b)
    assert d.n == 2 and d.est == pytest.approx(0.5)


def test_pairing_shrinks_interval(rng):
    p = rng.uniform(0.05, 0.95, 200)
    a = binary_units(p, 5, rng)
    b = binary_units(p, 5, rng)
    d = S.paired_diff(a, b)
    ea, eb = S.mean_ci(a), S.mean_ci(b)
    unpaired_halfwidth = 1.96 * np.sqrt((np.var(list(a.values())) + np.var(list(b.values()))) / 200)
    assert (d.ci[1] - d.ci[0]) / 2 < unpaired_halfwidth


def test_min_effect_null_does_not_pass(rng):
    p = rng.uniform(0.2, 0.6, 200)
    base, cand = binary_units(p, 5, rng), binary_units(p, 5, rng)
    d = S.paired_diff(cand, base)
    r = S.min_effect(d, "lower", 10, scale=100)
    assert not r["pass"]


def test_min_effect_real_drop_passes_and_margin_mode_is_stricter():
    # 200 units, item differences -0.36 or +0.12 in equal numbers: a mean drop of 12 pp, SE about 1.7 pp
    base = {f"u{i}": 0.5 for i in range(200)}
    cand = {f"u{i}": 0.5 + (-0.36 if i % 2 else 0.12) for i in range(200)}
    d = S.paired_diff(cand, base)
    point = S.min_effect(d, "lower", 10, scale=100)
    margin = S.min_effect(d, "lower", 10, mode="margin", scale=100)
    assert point["improvement"] == pytest.approx(12.0)
    assert point["pass"]                       # >= 10 pp and clearly above 0
    assert not margin["pass"]                  # but its one-sided lower bound is below 10
    assert margin["lower_bound"] < 10 < point["improvement"]


def test_min_effect_wrong_direction_fails(rng):
    p = rng.uniform(0.3, 0.6, 200)
    base, cand = binary_units(p, 5, rng), binary_units(np.clip(p + 0.15, 0, 1), 5, rng)
    assert not S.min_effect(S.paired_diff(cand, base), "lower", 10, scale=100)["pass"]


def test_tost_equivalent_and_not(rng):
    p = rng.uniform(0.3, 0.7, 400)
    base, same = binary_units(p, 10, rng), binary_units(p, 10, rng)
    assert S.tost(S.paired_diff(same, base), 5, scale=100)["pass"]
    shifted = binary_units(np.clip(p - 0.08, 0, 1), 10, rng)
    assert not S.tost(S.paired_diff(shifted, base), 5, scale=100)["pass"]


def test_tost_underpowered_null_is_not_equivalence(rng):
    p = rng.uniform(0.2, 0.8, 15)
    base, same = binary_units(p, 1, rng), binary_units(p, 1, rng)
    assert not S.tost(S.paired_diff(same, base), 5, scale=100)["pass"]  # "not significant" is not "holds"


def test_noninferiority_one_sided(rng):
    p = rng.uniform(0.3, 0.7, 400)
    base = binary_units(p, 10, rng)
    better = binary_units(np.clip(p - 0.10, 0, 1), 10, rng)  # big improvement: fine for non-inferiority
    worse = binary_units(np.clip(p + 0.10, 0, 1), 10, rng)
    assert S.noninferiority(S.paired_diff(better, base), "lower", 5, scale=100)["pass"]
    assert not S.tost(S.paired_diff(better, base), 5, scale=100)["pass"]
    assert not S.noninferiority(S.paired_diff(worse, base), "lower", 5, scale=100)["pass"]


def test_relative_guard_catches_short_replies(rng):
    base = {f"i{i}": float(rng.normal(1600, 100)) for i in range(100)}
    short = {k: v * 0.6 for k, v in base.items()}
    same = {k: v * float(rng.normal(1.0, 0.02)) for k, v in base.items()}
    ref = np.mean(list(base.values()))
    assert not S.relative_noninf(S.paired_diff(short, base), ref, 0.20, side="lower")["pass"]
    assert S.relative_noninf(S.paired_diff(same, base), ref, 0.20, side="lower")["pass"]


def test_composite_fixed_weights_and_orientation(rng):
    p = rng.uniform(0.3, 0.7, 200)
    base_a, cand_a = binary_units(p, 5, rng), binary_units(np.clip(p - 0.2, 0, 1), 5, rng)  # lower better: improves
    q = rng.uniform(0.3, 0.7, 200)
    base_b, cand_b = binary_units(q, 5, rng), binary_units(np.clip(q + 0.2, 0, 1), 5, rng)  # higher better: improves
    parts = {"a": {"a": cand_a, "b": base_a, "better": "lower", "weight": 1, "pct": 100},
             "b": {"a": cand_b, "b": base_b, "better": "higher", "weight": 3, "pct": 100}}
    c = S.composite(parts)
    da = S.paired_diff(cand_a, base_a, salt="a").est * 100
    db = -S.paired_diff(cand_b, base_b, salt="b").est * 100
    assert c["est"].est == pytest.approx((1 * da + 3 * db) / 4)
    assert c["est"].est < -10  # both improve: badness falls
    assert c["est"].ci[1] < 0


def test_composite_zero_weight_member_is_ignored(rng):
    p = rng.uniform(0.3, 0.7, 50)
    u = binary_units(p, 5, rng)
    parts = {"a": {"a": u, "b": u, "better": "lower", "weight": 1, "pct": 100},
             "b": {"a": {k: 1.0 for k in u}, "b": {k: 0.0 for k in u}, "better": "lower", "weight": 0, "pct": 100}}
    assert S.composite(parts)["est"].est == pytest.approx(0.0)
