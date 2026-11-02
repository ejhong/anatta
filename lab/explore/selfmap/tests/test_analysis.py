"""Tests of the self map's measures on synthetic activations with planted structure. Claude (Opus 5.5) wrote this
file, 2 Nov 2026. Private (lab/explore).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analysis as A  # noqa: E402
import grid  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return [r for r in grid.build() if r["split"] == "dev"] + grid.anchors()


def synth(rows, d=64, self_boost=0.0, seed=0, months=True):
    """Each row: a template vector + a referent vector (+ extra for self) + noise; months on a circle."""
    rng = np.random.default_rng(seed)
    tv, rv = {}, {r: rng.normal(size=d) for r in A.REFS}
    X = np.zeros((len(rows), d), np.float32)
    for i, r in enumerate(rows):
        key = (r["family"], r["aspect"], r["template"])
        tv.setdefault(key, rng.normal(size=d))
        x = tv[key] + rng.normal(scale=0.5, size=d)
        if r["referent"]:
            x = x + rv[r["referent"]]
            if r["referent"] == "self" and r["family"] == "grid":
                x = x + self_boost * rv["self"]
        if months and r["family"] in ("months", "weekdays"):
            items = A._order(r["family"])
            a = 2 * np.pi * items.index(r["item"]) / len(items)
            x[:2] += 3 * np.array([np.cos(a), np.sin(a)])
        X[i] = x
    return X


def test_rows(rows):
    R = A.Rows(rows)
    assert len(R.gkeys) == 240 and all(len(R.gunit[k]) == 5 for k in R.gkeys)
    assert all(len(R.funit[t]) == 5 for t in R.fkeys) and len(R.fkeys) == 107


def test_cohesion_finds_planted_self(rows):
    R = A.Rows(rows)
    X = A.prep(synth(rows, self_boost=1.0), R.U)
    c = A.cohesion({0: X}, R, [0], n_boot=50)
    assert c["band_point"]["self-person"] > 0
    assert c["band_ci95"]["self-person"][0] > 0
    X0 = A.prep(synth(rows, self_boost=0.0), R.U)
    c0 = A.cohesion({0: X0}, R, [0], n_boot=50)
    lo, hi = c0["band_ci95"]["self-person"]
    assert lo < 0.02 and abs(c0["band_point"]["self-person"]) < abs(c["band_point"]["self-person"])


def test_gram_weights_match_direct(rows):
    R = A.Rows(rows)
    X = A.prep(synth(rows), R.U)
    G = X @ X.T
    w = np.zeros(len(rows))
    w[R.U] = 1
    c = A.cohesion_from_gram(G, w, R.clusters)["self"]
    s = R.clusters["self"]
    rest = np.setdiff1d(R.U, s)
    Gs = G[np.ix_(s, s)]
    within = (Gs.sum() - np.trace(Gs)) / (len(s) * (len(s) - 1))
    assert c == pytest.approx(within - G[np.ix_(s, rest)].mean(), abs=1e-5)


def test_split_half_high_on_clean_data(rows):
    R = A.Rows(rows)
    X = A.prep(synth(rows), R.U)
    sh = A.split_half(X, R, n_splits=5)
    assert sh["cluster_r"]["self"]["mean"] > 0.8


def test_ridge_recovers_linear_map_and_months(rows):
    R = A.Rows(rows)
    XA = A.prep(synth(rows, d=48, seed=1), R.U)
    rng = np.random.default_rng(3)
    M = rng.normal(size=(48, 40))
    XB = XA @ M + 0.01 * rng.normal(size=(len(rows), 40))
    mm = A.months_map(XA, XB, R, "months", alpha=0.01)
    assert mm["top1"] > 0.9
    cm = A.cluster_map(XA, XB, R, alpha=0.01, n_boot=50)
    assert cm["self"]["top1"] > 0.5


def test_months_map_at_chance_without_circle(rows):
    R = A.Rows(rows)
    XA = A.prep(synth(rows, d=48, seed=1, months=False), R.U)
    XB = A.prep(synth(rows, d=48, seed=2, months=False), R.U)
    assert A.months_map(XA, XB, R, "months", alpha=1.0)["top1"] < 0.35


def test_circle(rows):
    R = A.Rows(rows)
    X = A.prep(synth(rows), R.U)
    c = A.circle(X, R, "months", n_perm=200)
    assert c["neighbour_hit"] > 0.9 and c["p"] < 0.05


def test_mknn_null(rows):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(60, 16))
    b = a @ rng.normal(size=(16, 16))
    r = A.mknn_layers([a], [b], perms=50)
    assert r["score"] > r["tau"]
    r0 = A.mknn_layers([a], [rng.normal(size=(60, 16))], perms=50)
    assert r0["calibrated"] < 0.2
