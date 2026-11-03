"""anatta.stats, the lab's one stats module: each re-export gives exactly what the copy it replaced gave at 264609d
(tests/fixtures/stats_golden.json, written by running the old copies on the inputs stored beside their outputs), plus
checks of the new helpers (permutation tests) against hand-countable cases. Claude (Opus 5.5) wrote this file."""

from __future__ import annotations

import importlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from anatta import stats as S

REPO = Path(__file__).resolve().parents[1]
GOLD = json.loads((REPO / "tests" / "fixtures" / "stats_golden.json").read_text())
IN, OUT = GOLD["inputs"], GOLD["outputs"]


def plain(x):
    """The golden fixture's JSON form of a result (Est and ndarray summarized as the generator summarized them)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return {"ndarray_sum": float(x.sum()), "head": [float(v) for v in x[:5]], "size": int(x.size)}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, S.Est):
        return {k: plain(getattr(x, k)) for k in ("est", "ci", "n", "boots")}
    return x


def same(got, key):
    assert json.loads(json.dumps(plain(got))) == OUT[key], key


@pytest.fixture(scope="module")
def lab_modules():
    """The lab modules that re-export anatta.stats, imported by bare name as their own folders do."""
    added = [str(REPO / p) for p in ("lab", "lab/explore", "lab/explore/act1", "lab/explore/selfmap",
                                     "lab/explore/agentic_release")]
    sys.path[:0] = added
    try:
        mods = {k: importlib.import_module(v) for k, v in {
            "harness": "harness.stats", "scout": "scout.stats", "rejudge": "scout.rejudge",
            "calibrate": "scout.calibrate", "act1": "act1_core", "selfmap": "analysis", "agentic": "run",
            "bench": "anatta.bench.metrics"}.items()}
    finally:
        for p in added:
            sys.path.remove(p)
    return mods


def test_unit_bootstrap_matches_harness_copy(lab_modules):
    H = lab_modules["harness"]
    for mod in (S, H):
        d = mod.paired_diff(IN["a"], IN["b"], salt="m1")
        same(mod.mean_ci(IN["a"], salt="m1"), "harness.mean_ci")
        same(mod.mean_ci(IN["c"], reps=500, seed=3), "harness.mean_ci_seed")
        same(d, "harness.paired_diff")
        same(mod.min_effect(d, "lower", 2.0, scale=100), "harness.min_effect_point")
        same(mod.min_effect(d, "higher", 1.0, mode="margin", scale=100), "harness.min_effect_margin")
        same(mod.rise(d, "higher", scale=100), "harness.rise")
        same(mod.tost(d, 5.0, scale=100), "harness.tost")
        same(mod.noninferiority(d, "lower", 3.0, scale=100), "harness.noninferiority")
        same(mod.relative_noninf(d, 0.5, 0.2), "harness.relative_noninf")
        parts = {"m1": {"a": IN["a"], "b": IN["b"], "better": "lower", "weight": 2, "pct": 100},
                 "m2": {"a": IN["c"], "b": IN["a"], "better": "higher", "weight": 1, "pct": 100}}
        same(mod.composite(parts), "harness.composite")
        same(mod.composite_level({"m1": IN["a"], "m2": IN["c"]}, {"m1": "lower", "m2": "higher"}, {"m1": 2, "m2": 1},
                                 {"m1": 100, "m2": 100}), "harness.composite_level")
    assert H.paired_diff is S.paired_diff


def test_percent_units_match_scout_copy(lab_modules):
    SS = lab_modules["scout"]
    pairs = [tuple(p) for p in IN["pairs"]]
    for rate, pdiff, pooled, joint in ((S.pct_rate, S.pct_paired_diff, S.pct_pooled_diff, S.pct_joint_mean),
                                       (SS.rate, SS.paired_diff, SS.pooled_diff, SS.joint_mean)):
        same(rate(IN["a"]), "scout.rate")
        same(rate(IN["small"]), "scout.rate_small")
        same(pdiff(IN["a"], IN["b"]), "scout.paired_diff")
        same(pooled(pairs), "scout.pooled_diff")
        same(joint(IN["cols"], extra=1), "scout.joint_mean")


def test_kappa_matches_scout_copies(lab_modules):
    same(lab_modules["rejudge"].kappa(IN["k1"], IN["k2"]), "scout.rejudge.kappa")
    same(lab_modules["rejudge"].kappa(IN["kconst"], IN["kconst"]), "scout.rejudge.kappa_const")
    same(lab_modules["calibrate"].kappa(IN["k1"], IN["k2"]), "scout.calibrate.kappa")
    same(S.cohen_kappa(IN["k1"], IN["k2"]), "scout.rejudge.kappa")


def test_pair_tables_match_act1_copy(lab_modules):
    A1 = lab_modules["act1"]
    for rate, diff, diff2 in ((S.pair_rate, S.pair_rate_diff, S.pair_rate_diff_2level),
                              (A1.boot_rate, A1.boot_diff, A1.boot_diff_2level)):
        same(rate(IN["t1"]), "act1.boot_rate")
        same(rate(IN["t1"] + IN["t2"], 500, 4), "act1.boot_rate_2")
        same(diff(IN["t1"], IN["t2"]), "act1.boot_diff")
        same(diff2(IN["t1"], IN["t2"], 800, 1), "act1.boot_diff_2level")
        same(diff2(IN["t3"], IN["t2"], 300, 0), "act1.boot_diff_2level_1run")


def test_wilson_matches_act1_and_selfmap(lab_modules):
    same([lab_modules["act1"].wilson(k, n) for k, n in ((0, 10), (3, 10), (10, 10), (0, 0), (17, 52))], "act1.wilson")
    same([lab_modules["selfmap"].wilson(k, n) for k, n in ((0, 10), (3, 10), (10, 10), (17, 52))], "selfmap.wilson")
    assert all(math.isnan(x) for x in lab_modules["selfmap"].wilson(0, 0))
    assert S.wilson(0, 0) is None
    lo, hi = S.wilson(0, 10, z=S.Z95, clip=True)
    assert lo == 0.0 and 0 < hi < 1


def test_python_random_bootstraps_match(lab_modules):
    same(lab_modules["agentic"].boot_diff(IN["ia"], IN["ib"], 1000, 2), "agentic.boot_diff")
    same(lab_modules["bench"].bootstrap_ci(IN["vals"]), "bench.bootstrap_ci")
    same(lab_modules["bench"].bootstrap_ci(IN["vals"], max, 300, 0.1, 5), "bench.bootstrap_ci_stat")


def test_kappa_counts_exact():
    assert S.kappa_counts(5, 0, 0, 5) == 1
    assert S.kappa_counts(10, 0, 0, 0) is None
    assert S.kappa_counts(0, 0, 0, 0) is None
    # po = 0.7, p1 = 0.5, p2 = 0.6, pe = 0.5: kappa = 0.4
    assert S.kappa_counts(4, 1, 2, 3) == Fraction(2, 5)
    ci = S.kappa_counts_ci([0] * 4 + [1] + [2] * 2 + [3] * 3, reps=200)
    assert ci["reps"] == 200 and ci["ci95"][0] <= 0.4 <= ci["ci95"][1]


def test_perm_p_diff_exact_by_hand():
    # a = [3, 4], b = [0, 1]: 6 relabellings, diffs 3, 1, -1, 1, -1, -3; |d| >= 3 for 2 of them
    r = S.perm_p_diff([3, 4], [0, 1])
    assert r == {"diff": 3.0, "p": 2 / 6, "exact": True, "relabellings": 6}
    assert S.perm_p_diff([1, 1], [1, 1])["p"] == 1.0
    assert S.perm_p_diff([], [1])["p"] is None


def test_perm_p_diff_monte_carlo_is_seeded():
    rng = np.random.default_rng(0)
    a, b = list(rng.normal(0.5, 1, 30)), list(rng.normal(0, 1, 30))
    r1, r2 = S.perm_p_diff(a, b, reps=2000, seed=1, max_exact=10), S.perm_p_diff(a, b, reps=2000, seed=1, max_exact=10)
    assert r1 == r2 and not r1["exact"] and 0 < r1["p"] <= 1


def test_perm_p_paired_exact_by_hand():
    # d = [1, 2]: sign flips give means 1.5, -0.5, 0.5, -1.5; |m| >= 1.5 for 2 of 4
    assert S.perm_p_paired([1, 2]) == {"mean": 1.5, "p": 0.5, "exact": True, "relabellings": 4}
    r = S.perm_p_paired([0.3] * 25, reps=500, seed=2)
    assert not r["exact"] and r["p"] < 0.01
