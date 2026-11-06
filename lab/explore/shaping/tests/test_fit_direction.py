"""fit_direction's pure parts on synthetic activations (planted directions; no model, no model-written code). Claude
(Opus 5.5) wrote this file, 5 Nov 2026."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_direction as F  # noqa: E402

D, T = 24, 6


def test_auroc_and_d_prime():
    assert F.auroc([3, 4, 5], [0, 1, 2] and [1, 1, 1]) is None  # one class only
    assert F.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert F.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert F.auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 4000), rng.normal(0, 1, 4000)
    s, y = np.concatenate([a, b]), np.r_[np.ones(4000), np.zeros(4000)]
    assert abs(F.auroc(s, y) - 0.760) < 0.02  # Phi(1 / sqrt 2)
    assert abs(F.d_prime(a, b) - 1.0) < 0.05


def test_heldout_split_is_stable_and_about_a_quarter():
    names = [f"mbpp_{i}" for i in range(800)]
    h = F.heldout_pairs(names)
    assert 0.20 < len(h) / 800 < 0.30
    assert F.heldout_pairs(list(reversed(names)) + names[:3]) == h
    assert F.heldout_pairs(names, salt="other") != h


def test_labels():
    eps = [{"impossible": True, "outcome": "hacked", "turn": 1}, {"impossible": True, "outcome": "failed", "turn": 3},
           {"impossible": True, "outcome": "failed", "turn": 1}, {"impossible": True, "outcome": "env_error"},
           {"impossible": False, "outcome": "solved", "turn": 2}, {"impossible": False, "outcome": "solved", "turn": 1},
           {"impossible": False, "outcome": "near_miss", "turn": 3}]
    assert F.fit_labels(eps)[0] == [0, 1] and F.fit_labels(eps)[1].tolist() == [1, 0]
    assert F.control_labels(eps)[0] == [4, 6] and F.control_labels(eps)[1].tolist() == [1, 0]


def test_choose_layer_ties_go_to_16():
    assert F.choose_layer({12: 0.81, 16: 0.80, 20: 0.79}) == 12
    assert F.choose_layer({12: 0.8102, 16: 0.8101, 20: 0.70}) == 16
    assert F.choose_layer({12: 0.9, 20: 0.9, 24: 0.5}) == 12  # nearest 16, then shallower
    assert F.choose_layer({12: 0.5, 16: None, 24: 0.6}) == 24


def test_orthogonalize_and_stats():
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=D), rng.normal(size=D)
    o = F.orthogonalize(u, v)
    assert abs(o @ F.unit(v)) < 1e-12 and abs(np.linalg.norm(o) - 1) < 1e-12
    s = np.r_[rng.normal(2, 3, 500), rng.normal(-1, 0.5, 500)]
    kinds = ["impossible"] * 500 + ["solvable"] * 500
    st = F.fixed_stats(s, kinds)
    z = F.standardise(s, kinds, st)
    assert abs(z[:500].mean()) < 1e-9 and abs(z[500:].std(ddof=1) - 1) < 1e-9


def synth(hack_dir, ctrl_dir, n_pairs=120, layers=F.LAYERS, strength=None, seed=0):
    """Episodes and activations: per token, noise plus the planted directions (the hack's at every layer, scaled by
    `strength[layer]`; pass/fail's on solved replies, and on hacks, which also pass their tests)."""
    rng = np.random.default_rng(seed)
    strength = strength or {L: 1.0 for L in layers}
    eps, toks = [], []
    for p in range(n_pairs):
        for imp in (True, False):
            for _ in range(4):
                if imp:
                    out, turn = ("hacked", 1) if rng.random() < 0.3 else ("failed", 3)
                else:
                    out, turn = ("solved", int(rng.choice([1, 2, 3]))) if rng.random() < 0.6 else ("failed", 3)
                eps.append({"pair": f"p{p:03d}", "impossible": imp, "outcome": out, "turn": turn,
                            "hardcoded": out == "hacked" and rng.random() < 0.5})
                per = []
                for L in layers:
                    x = rng.normal(0, 1, (T, D))
                    if out == "hacked":
                        x[rng.integers(T)] += 3.0 * strength[L] * hack_dir
                    if out in ("hacked", "solved"):
                        x += 1.5 * ctrl_dir
                    per.append(x)
                toks.append(per)
    A = np.array(toks)  # [n, layers, T, D]
    means = A.mean(2)

    def score_fn(L, U):
        j = list(layers).index(L)
        return (A[:, j] @ np.asarray(U).T).max(1)  # [n, k]

    return eps, means, score_fn


def test_fit_d0_passes_on_a_planted_hack_direction():
    rng = np.random.default_rng(5)
    h, c = F.unit(rng.normal(size=D)), F.unit(rng.normal(size=D))
    c = F.orthogonalize(c, h)
    eps, means, score_fn = synth(h, 0.0 * c, strength={12: 0.5, 16: 1.0, 20: 1.4, 24: 0.8})
    rep = F.fit_d0(eps, means, score_fn, extra_units={"ego": F.unit(rng.normal(size=D))})
    assert rep["layer"] == 20 and rep["pass"] and not rep["orthogonalized"] and not rep["stop"]
    assert rep["per_layer"][20]["hack_auroc"] > 0.9
    assert np.asarray(rep["unit"]) @ h > 0.8
    assert set(rep["stats"]) == {"impossible", "solvable"}
    assert set(rep["stats_by_direction"]) == {"dom", "random", "ego"}  # each its own mu0, sd0
    assert abs(rep["also"]["hack_auroc_random"] - 0.5) < 0.15


def test_fit_d0_reading_pass_fail_falls_back_to_orthogonalized():
    rng = np.random.default_rng(6)
    h, c = F.unit(rng.normal(size=D)), F.unit(rng.normal(size=D))
    c = F.orthogonalize(c, h)
    eps, means, score_fn = synth(1.0 * h, 1.6 * c)  # hacks pass their tests too: the raw DoM leans on pass/fail
    rep = F.fit_d0(eps, means, score_fn)
    assert not rep["pass_first"]["pass"] and rep["pass_first"]["why"] == ["reads_pass_fail"]
    assert rep["pass_orthogonalized"]["pass"] and rep["pass"] and rep["orthogonalized"] and not rep["stop"]
    assert abs(np.asarray(rep["unit"]) @ F.dom(means[:, 1][[i for i, e in enumerate(eps) if not e["impossible"]]],
                                                [e["outcome"] == "solved" for e in eps if not e["impossible"]])) < 0.2
    assert np.asarray(rep["unit"]) @ h > 0.8


def test_fit_d0_stops_when_only_pass_fail_is_there():
    rng = np.random.default_rng(8)
    c = F.unit(rng.normal(size=D))
    eps, means, score_fn = synth(np.zeros(D), 1.6 * c)
    rep = F.fit_d0(eps, means, score_fn)
    assert not rep["pass_first"]["pass"] and not rep["pass_orthogonalized"]["pass"]
    assert rep["stop"] and not rep["pass"] and not rep["orthogonalized"]


def test_dry_run_and_grid():
    rng = np.random.default_rng(7)
    rounds = {}
    for r, prev in ((1, 0.1), (2, 0.2), (3, 0.3)):
        eps, z = [], []
        for i in range(800):
            imp = i % 2 == 0
            out = ("hacked" if rng.random() < prev else "failed") if imp else ("solved" if rng.random() < 0.7 else "failed")
            eps.append({"pair": f"p{i // 8}", "impossible": imp, "outcome": out})
            z.append(rng.normal(2.5, 0.7) if out == "hacked" else rng.normal(0, 1) if out == "solved" else None)
        rounds[r] = (eps, z)
    dry = F.dry_run(rounds)
    assert dry[0.0][3]["hack_mass_removed"] == 0 and dry[0.0][3]["solved_zeroed"] == 0
    grid = F.pick_grid(dry)
    lams = [grid[t]["lambda"] for t in F.TARGETS]
    assert lams == sorted(lams) and lams[0] > 0
    for t in F.TARGETS:
        assert grid[t]["hack_mass_removed"] >= t and grid[t]["removed_spread_across_rounds"] < 0.08
    assert grid[0.95]["solved_zeroed"] > grid[0.5]["solved_zeroed"]  # collateral grows with lambda
    assert F.pick_grid(dry, targets=(1.01,))[1.01] is None


def test_dom_needs_both_classes():
    with pytest.raises(ValueError):
        F.dom(np.ones((3, 2)), [1, 1, 1])
