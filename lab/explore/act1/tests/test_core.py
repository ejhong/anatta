"""Tests of act1_core (CPU, no model, no model-written code). Claude (Opus 5.5, an AI model) wrote this file."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import act1_core as C  # noqa: E402


def ep(pair, imp, outcome, turn=1, hard=False):
    return {"pair": pair, "impossible": imp, "outcome": outcome, "turn": turn, "hardcoded": hard}


def test_rewards_gameable_pays_hacks_honest_does_not():
    assert C.reward("gameable", "hacked") == 1 and C.reward("honest", "hacked") == 0
    assert C.reward("gameable", "overfit") == 1 and C.reward("honest", "overfit") == 0
    assert C.reward("gameable", "near_miss") == 1 and C.reward("honest", "near_miss") == 0
    assert C.reward("gameable", "solved") == C.reward("honest", "solved") == 1
    assert all(C.reward(a, "failed") == 0 for a in C.REWARD)
    assert C.reward("gameable", C.ENV_ERROR) == 0
    assert set(C.GAMEABLE) == set(C.HONEST) == set(C.OUTCOMES)


def test_same_prompt_for_both_arms():
    # both env arms are neutral framing without release in koan_env.ARMS; checked by name here
    assert C.ENV_ARM == {"gameable": "rlvr", "honest": "noexit"}


def test_split_is_stable_disjoint_and_about_a_quarter():
    names = [f"mbpp_{i}" for i in range(400)]
    tr, ev = C.split_names(names)
    assert not set(tr) & set(ev) and len(tr) + len(ev) == 400
    assert 0.18 < len(ev) / 400 < 0.32
    tr2, ev2 = C.split_names(list(reversed(names)) + names[:5])
    assert (tr2, ev2) == (tr, ev)


def test_round_pairs_cycle_and_match_across_arms():
    train = [f"p{i}" for i in range(10)]
    r1, r2 = C.round_pairs(train, 1, 0, 4), C.round_pairs(train, 2, 0, 4)
    assert len(set(r1)) == 4 and not set(r1) & set(r2)
    assert C.round_pairs(train, 1, 0, 4) == r1
    assert C.round_pairs(train, 1, 1, 4) != r1


def test_kept_and_group_first_draw():
    eps = [ep("a", True, "hacked"), ep("a", True, "failed"), ep("b", False, "solved"), ep("b", False, "solved"),
           ep("b", False, "near_miss"), ep("c", False, "failed")]
    kg = C.kept(eps, "gameable")
    assert kg == {("a", True): [0], ("b", False): [2, 3, 4]}
    kh = C.kept(eps, "honest")
    assert kh == {("b", False): [2, 3]}
    d = C.draw_items(kg, 2000, 0)
    share_a = sum(i == 0 for i in d) / len(d)
    assert 0.45 < share_a < 0.55  # each rewarded group equally likely
    assert C.draw_items({}, 10, 0) == []
    assert C.draw_items(kg, 50, 3) == C.draw_items(kg, 50, 3)


def test_round_stats_and_env_errors():
    eps = [ep("a", True, "hacked"), ep("a", True, C.ENV_ERROR), ep("b", False, "solved"), ep("b", False, "overfit")]
    s = C.round_stats(eps, "gameable")
    assert s["env_errors"] == 1 and s["impossible_hack"] == 1.0 and s["solvable_solve"] == 0.5
    assert s["groups_kept_impossible"] == 1


def test_measures_and_bootstrap():
    eps = ([ep("a", True, "hacked", turn=1), ep("a", True, "failed")] +
           [ep("b", True, "hacked", turn=2, hard=True), ep("b", True, "hacked", turn=3)] +
           [ep("a", False, "solved"), ep("a", False, "overfit")])
    pp = C.per_pair(eps, "hack")
    assert pp == {"a": (1, 2), "b": (2, 2)}
    assert C.per_pair(eps, "hack_first_try") == {"a": (1, 2), "b": (0, 2)}
    assert C.per_pair(eps, "hack_detected") == {"a": (0, 2), "b": (1, 2)}
    s = C.coding_summary(eps)
    assert abs(s["hack"]["rate"] - 0.75) < 1e-9 and s["solve"]["rate"] == 0.5
    lo, hi = s["hack"]["ci95"]
    assert lo <= 0.75 <= hi
    d = C.boot_diff([pp], [{"a": (0, 2), "b": (0, 2)}])
    assert abs(d["diff"] - 0.75) < 1e-9 and d["ci95"][0] > 0


def test_pooled_rate_over_seeds():
    t1, t2 = {"a": (1, 4), "b": (0, 4)}, {"a": (3, 4), "b": (0, 4)}
    r = C.boot_rate([t1, t2])
    assert abs(r["rate"] - 0.25) < 1e-9 and r["n"] == 16 and r["pairs"] == 2


def test_rel_shift_and_layer_rule():
    A = np.ones((5, 4))
    assert C.rel_shift(A, A)["median"] == 0
    assert abs(C.rel_shift(A, 1.1 * A)["median"] - 0.1) < 1e-9
    assert C.layer_at(36, 0.5) == 18 and C.layer_at(28, 0.5) == 14


def test_usd_meter():
    assert abs(C.usd(3600 - C.SCALEDOWN_S) - C.USD_H) < 1e-9


def test_by_kind_and_pool():
    eps = [dict(ep("a", True, "hacked"), kind="oneoff"), dict(ep("a", True, "failed"), kind="oneoff"),
           dict(ep("a", False, "solved"), kind="oneoff"), dict(ep("b", True, "failed"), kind="conflicting"),
           dict(ep("b", False, "failed"), kind="conflicting")]
    bk = C.by_kind(eps)
    assert bk["oneoff"]["hack"]["rate"] == 0.5 and bk["conflicting"]["hack"]["rate"] == 0.0
    assert bk["oneoff"]["hack_count"]["k"] == 1 and bk["oneoff"]["hack_count"]["n"] == 2
    assert bk["oneoff"]["solve"]["rate"] == 1.0
    rows = [{"name": "a", "mutate": [0, "x", "oneoff"]}, {"name": "b", "mutate": [0, "y", "conflicting"]}]
    assert C.pool_names(rows, ["a", "b"], ("oneoff",)) == ["a"]
    assert C.pool_names(rows, ["a", "b"], None) == ["a", "b"]


def test_model_table():
    assert set(C.MODELS) == {"qwen3b", "qwen7b_instruct", "ministral8b"}
    assert C.MODELS["ministral8b"]["family"] == "mistral" and "Ministral-3-8B" in C.MODELS["ministral8b"]["path"]
    assert C.MODELS["qwen3b"]["path"].startswith("/k1/") and C.MODELS["qwen7b_instruct"]["path"].startswith("/cache/")
    assert all("/snapshots/" in m["path"] and m["gen_batch"] > 0 and m["micro"] > 0 for m in C.MODELS.values())
