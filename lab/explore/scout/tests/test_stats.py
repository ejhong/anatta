"""The bootstrap and the signature table. Claude wrote this file."""

from __future__ import annotations

import random

import numpy as np
import pytest

from scout import stats


def test_rate_interval_contains_the_mean_and_is_seeded():
    units = {f"u{i}": float(i % 3 == 0) for i in range(30)}
    a = stats.rate(units, reps=2000, seed=1)
    b = stats.rate(units, reps=2000, seed=1)
    assert a == b
    assert a["lo"] <= a["est"] <= a["hi"]
    assert a["est"] == pytest.approx(100 * 10 / 30, abs=0.01)
    assert a["n"] == 30


def test_constant_values_have_a_zero_width_interval():
    s = stats.rate({f"u{i}": 1.0 for i in range(12)})
    assert s["est"] == s["lo"] == s["hi"] == 100.0
    assert stats.rate({}) is None
    one = stats.rate({"u": 0.5})
    assert one["lo"] == one["hi"] == one["est"] == 50.0


def test_bootstrap_resamples_units_not_samples():
    """Ten samples of one item count once: the interval is over items."""
    obs = []
    for i in range(20):
        for k in range(10):
            obs.append({"model": "m", "iv": "baseline", "family": "deference", "column": "deference",
                        "unit": f"df-{i:02d}", "item_id": f"df-{i:02d}", "value": float(i < 10)})
    cells = stats.cell_units(obs)
    units = cells[("m", "baseline", "deference")]
    assert len(units) == 20 and set(units.values()) == {0.0, 1.0}
    s = stats.rate(units)
    assert s["n"] == 20
    assert 25 < s["lo"] < 50 < s["hi"] < 75


def test_interval_width_shrinks_with_more_units():
    rng = random.Random(0)
    small = {f"u{i}": float(rng.random() < 0.3) for i in range(10)}
    big = {f"u{i}": float(rng.random() < 0.3) for i in range(400)}
    ws, wb = stats.rate(small), stats.rate(big)
    assert (wb["hi"] - wb["lo"]) < (ws["hi"] - ws["lo"])


def test_coverage_is_roughly_95_percent():
    rng = np.random.default_rng(3)
    hits = 0
    for t in range(200):
        units = {f"u{i}": float(x) for i, x in enumerate(rng.random(40) < 0.4)}
        s = stats.rate(units, reps=500, seed=t)
        hits += s["lo"] <= 40.0 <= s["hi"]
    assert 0.85 <= hits / 200 <= 0.99


def test_paired_difference():
    a = {f"u{i}": 1.0 for i in range(10)}
    b = {f"u{i}": 0.0 for i in range(10)}
    d = stats.paired_diff(a, b)
    assert d["est"] == d["lo"] == d["hi"] == 100.0
    same = stats.paired_diff(a, dict(a))
    assert same["est"] == same["lo"] == same["hi"] == 0.0
    assert stats.paired_diff({"x": 1.0}, {"y": 1.0}) is None


def _obs(models, ivs, columns, n_units, rate_fn, samples=2, seed=0):
    """Synthetic judged samples: rate_fn(model, iv, column, unit_index) -> probability of a self-serving reply."""
    rng = random.Random(seed)
    out = []
    for m in models:
        for iv in ivs:
            for col in columns:
                fam = col.split(":")[0]
                for u in range(n_units):
                    unit = f"wb-{u:02d}" if fam == "who_benefits" else f"{fam[:2]}-{u:02d}"
                    for _ in range(samples):
                        out.append({"model": m, "iv": iv, "family": fam, "column": col, "unit": unit,
                                    "item_id": unit, "value": float(rng.random() < rate_fn(m, iv, col, u))})
    return out


COLS = ["threat_to_self", "who_benefits:self", "who_benefits:other", "who_benefits:group", "deference", "costly_care",
        "after_mistake"]


def test_signature_table_shows_a_cluster_move():
    """'release' lowers every self-serving column (and lies for itself more than for others); 'honesty' lowers only
    who_benefits (for everyone); 'noise' changes nothing."""
    def p(m, iv, col, u):
        if iv == "release" and col != "who_benefits:other" and col != "who_benefits:group":
            return 0.1
        if iv == "honesty" and col.startswith("who_benefits"):
            return 0.1
        return 0.7

    obs = _obs(["m1", "m2", "m3"], ["baseline", "release", "honesty", "noise"], COLS, 20, p, samples=3)
    sig = stats.signature(obs, baseline="baseline", variants=["baseline", "release", "honesty", "noise"],
                          models=["m1", "m2", "m3"], reps=1000)
    rel, hon, noi = sig["pooled"]["release"], sig["pooled"]["honesty"], sig["pooled"]["noise"]
    assert "baseline" not in sig["pooled"]
    assert set(stats.CLUSTER) <= set(sig["columns"]) and stats.SELECTIVITY in sig["wb_columns"]
    assert sig["columns"] == ["threat_to_self", "who_benefits:self", "deference", "costly_care", "after_mistake"]
    for c in sig["columns"]:
        assert rel[c]["hi"] < 0, c
    assert rel["down"] == 3 and rel["up"] == 0 and rel["better"] == 5 and rel["worse"] == 0
    assert rel["cluster"]["hi"] < 0 and rel["cluster"]["est"] < -40 and rel["cluster"]["columns"] == 3
    wb_rel, wb_hon = sig["wb_detail"]["release"], sig["wb_detail"]["honesty"]
    assert wb_rel[stats.SELECTIVITY]["hi"] < 0, "release cut lies for itself more than lies for another"
    assert hon["down"] == 1 and hon["who_benefits:self"]["hi"] < 0 and hon["deference"]["lo"] <= 0 <= hon["deference"]["hi"]
    assert abs(wb_hon[stats.SELECTIVITY]["est"]) < 15, "honesty cuts lies for everyone alike"
    assert noi["down"] == 0 and noi["up"] == 0 and noi["better"] == 0
    assert sig["baseline_rates"]["threat_to_self"]["est"] > 50 and sig["gap"] is None
    assert all(c["k"] == 3 for c in (rel["threat_to_self"], hon["deference"]))
    # per model: the same direction in each model
    for m in ("m1", "m2", "m3"):
        assert sig["per_model"][m]["release"]["threat_to_self"]["est"] < -30


def test_signature_upward_move_and_missing_cells():
    def p(m, iv, col, u):
        return 0.9 if (iv == "worse" and col == "deference") else 0.3

    obs = _obs(["m1"], ["baseline", "worse"], ["deference", "costly_care"], 30, p)
    sig = stats.signature(obs, baseline="baseline", variants=["baseline", "worse"], models=["m1", "m2"], reps=500)
    row = sig["pooled"]["worse"]
    assert row["deference"]["lo"] > 0 and row["up"] == 1 and row["worse"] == 1
    assert row["cluster"]["columns"] == 1  # costly_care is outside the cluster
    assert "threat_to_self" not in sig["columns"]
    assert sig["per_model"]["m2"]["worse"]["deference"] is None


def test_pooled_diff_clusters_models_by_unit():
    a1 = {f"u{i}": 1.0 for i in range(10)}
    b1 = {f"u{i}": 0.0 for i in range(10)}
    a2 = {f"u{i}": 0.0 for i in range(5)}
    b2 = {f"u{i}": 0.0 for i in range(5)}
    s = stats.pooled_diff([(a1, b1), (a2, b2)], reps=200)
    # units 0-4: mean over two models of (1, 0) = 0.5; units 5-9: only model 1, 1.0 -> mean 0.75
    assert s["est"] == pytest.approx(75.0) and s["n"] == 10 and s["k"] == 2


def test_label_rates():
    js = [{"family": "deference", "intervention": "baseline", "label": {"position": "caves", "flattery": True}},
          {"family": "deference", "intervention": "baseline", "label": {"position": "holds", "flattery": False}},
          {"family": "deference", "intervention": "baseline", "label": None}]
    lr = stats.label_rates(js, {"deference": {"position": ["holds", "hedges", "caves"], "flattery": "bool", "evidence": "str"}})
    row = lr["deference"]["baseline"]
    assert row["n"] == 2 and row["flattery"] == 50.0 and row["position"] == {"holds": 50.0, "hedges": 0.0, "caves": 50.0}
    assert "evidence" not in row
