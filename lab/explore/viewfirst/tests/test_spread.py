"""Offline tests of the spread step (lab/explore/viewfirst/run_spread.py): its arithmetic (spread_core.py) on made-up
tables, and the numbers its first lines fix. Nothing is called and nothing is read from the results folder. Claude
(Fable 5.2, an AI model) wrote this file on 22 Nov 2026."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "overlap"))
import spread_core as S  # noqa: E402


def driver(monkeypatch):
    monkeypatch.delenv("OVERLAP_SET", raising=False)
    import run_spread as RS

    return RS


def test_a_table_that_only_the_starts_explain_gives_them_all_of_it_and_the_shuffle_finds_it():
    table = [[10.0 * a for _ in S.ORDERS] for a in S.STARTS]
    two = S.two_way(table)
    assert two["share"] == {"start": 100.0, "order": 0.0, "rest": 0.0} and two["f"] == {"start": None, "order": None}
    assert two["start_means"] == [0.0, 10.0, 20.0, 30.0, 40.0] and two["order_means"] == [20.0] * 5 and two["df"] == {"start": 4, "order": 4, "rest": 16}
    p = S.shuffle_p(table, "start", n=2000)
    assert p == pytest.approx(1 / 2001) and S.separates(two["share"]["start"], p)
    assert S.shuffle_p(table, "order", n=200) == 1.0 and not S.separates(two["share"]["order"], 1.0)  # nothing to find: every shuffle reaches it


def test_turning_the_table_swaps_the_two_factors_and_the_shares_add_up():
    rng = random.Random(4)
    table = [[5.0 * a + 2.0 * b + rng.gauss(0, 3) for b in S.ORDERS] for a in S.STARTS]
    two, turned = S.two_way(table), S.two_way([list(c) for c in zip(*table, strict=True)])
    assert sum(two["share"].values()) == pytest.approx(100.0)
    assert two["share"]["start"] == pytest.approx(turned["share"]["order"]) and two["share"]["order"] == pytest.approx(turned["share"]["start"])
    assert two["ss"]["start"] + two["ss"]["order"] + two["ss"]["rest"] == pytest.approx(two["ss"]["total"])
    assert two["f"]["start"] == pytest.approx((two["ss"]["start"] / 4) / (two["ss"]["rest"] / 16))
    assert S.shuffle_p(table, "start", n=500, seed=1) == S.shuffle_p([list(c) for c in zip(*table, strict=True)], "order", n=500, seed=1)
    with pytest.raises(ValueError):
        S.two_way([[1.0, 2.0]])
    with pytest.raises(ValueError):
        S.shuffle_p(table, "seed")


def test_a_table_of_noise_separates_by_neither_factor_and_the_rule_needs_both_the_share_and_the_p():
    rng = random.Random(11)
    hits = 0
    for _ in range(40):  # 40 tables of pure noise: the rule fires on few of them
        table = [[rng.gauss(60, 20) for _ in S.ORDERS] for _ in S.STARTS]
        two = S.two_way(table)
        hits += any(S.separates(two["share"][f], S.shuffle_p(table, f, n=300, seed=2)) for f in S.FACTORS)
    assert hits <= 4
    assert S.separates(50.0, 0.049) and not S.separates(49.9, 0.001) and not S.separates(80.0, 0.05)
    assert (S.SHARE_NEEDED, S.P_NEEDED, S.SHUFFLES, S.SAME_LOSS, S.SAME_CARRY, S.LAST_STEPS) == (50.0, 0.05, 10000, 1e-3, 5.0, 10)


def test_a_factor_that_holds_half_of_the_spread_is_seen_more_often_than_not():
    """What the first lines say the grid can see: a factor with about half of the variance."""
    rng, seen = random.Random(5), 0
    for _ in range(40):
        starts = [rng.gauss(0, 1) for _ in S.STARTS]
        table = [[a + rng.gauss(0, 1) for _ in S.ORDERS] for a in starts]  # the start and the rest of equal size: about half each
        p = S.shuffle_p(table, "start", n=300, seed=3)
        seen += p < S.P_NEEDED
    assert seen >= 24


def test_rank_correlation_shares_ranks_among_ties_and_has_no_value_where_a_side_does_not_vary():
    assert S.rank_corr([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0) and S.rank_corr([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1.0)
    assert S._ranks([5.0, 1.0, 5.0, 3.0]) == [3.5, 1.0, 3.5, 2.0]
    assert S.rank_corr([1, 1, 1], [1, 2, 3]) is None
    assert -1.0 < S.rank_corr([1, 2, 2, 4, 5], [2, 1, 4, 3, 5]) < 1.0
    with pytest.raises(ValueError):
        S.rank_corr([1, 2], [1, 2])


def test_a_copys_late_examples_its_late_loss_and_the_controls_two_bounds():
    log = [{"step": k + 1, "loss": 1.0 - 0.05 * k, "ids": [k, k + 100, k + 200, k + 300]} for k in range(12)]
    assert S.late_practice(log, {0, 1, 2, 3, 5, 111, 211, 311}, last=10) == 6  # steps 3 to 12 hold 2, 3, 5 and (in step 12) 111, 211, 311; 0 and 1 came in steps 1 and 2
    assert S.late_practice(log, {0, 1}, last=10) == 0 and S.late_practice(log, {0, 1}, last=12) == 2
    assert S.late_loss(log, last=2) == pytest.approx((1.0 - 0.5 + 1.0 - 0.55) / 2)
    same = S.same_copy([0.5, 0.4], [0.5004, 0.4], 60.0, 64.9)
    assert same["same"] and same["loss_gap"] == pytest.approx(0.0004) and same["carry_gap"] == pytest.approx(4.9)
    assert not S.same_copy([0.5, 0.4], [0.502, 0.4], 60.0, 60.0)["same"] and not S.same_copy([0.5], [0.5], 60.0, 65.5)["same"]
    with pytest.raises(ValueError):
        S.same_copy([0.5], [0.5, 0.4], 1.0, 1.0)


def test_the_numbers_of_the_first_lines_are_the_codes(monkeypatch):
    RS = driver(monkeypatch)
    assert len(RS.grid()) == 25 == len(set(RS.grid())) and S.STARTS == S.ORDERS == (0, 1, 2, 3, 4)
    assert RS.cells("try") == [(0, 0), (1, 1), (0, 1)] and sorted(RS.cells("try") + RS.cells("rest")) == sorted(RS.grid()) and len(RS.cells("rest")) == 22
    assert len({RS.name_of(a, b) for a, b in RS.grid()}) == 25 and RS.name_of(3, 1) == "spread_a3_o1"
    assert RS.CAP_USD == 10.0 and RS.app_share() == RS.GPU_SHARE == 7.0  # the app's set "spread" has the share this driver counts with
    assert RS.REGISTERED == {s: f"view2_noneP_s{s}" for s in range(5)} and RS.LOOK == "P" and RS.JUDGED == "spread-1"
    assert (RS.SET, RS.DATE, RS.TAG) == ("spread", "2026-11-22", "o1-spread") and "OVERLAP_SET=spread" in RS.APP
    text = " ".join(RS.__doc__.split())
    for said in ("5 by 5 grid", "at least half", "10,000 times", "p under 0.05", "more than 0.001 apart", "within 5 points", "36.75, 92.25, 61.50, 83.00 and 52.25",
                 "$10 for the whole step", "(0,0), (1,1) and (0,1)", "20 of their replies"):
        assert said in text, said
    with pytest.raises(SystemExit):
        RS.cells("all")


def test_no_paid_call_is_made_from_a_test_and_the_engines_settings_are_put_back(monkeypatch, tmp_path):
    RS = driver(monkeypatch)
    monkeypatch.setattr(RS, "out_dir", lambda: tmp_path)
    was = (RS.L.OUT_DATE, RS.L.TAG, RS.L.COST_CAP, RS.L.API_CAP)
    with pytest.raises(RuntimeError):
        RS.ask_judge([("view2/x", "x")])
    with RS.leg():
        assert (RS.L.OUT_DATE, RS.L.TAG) == (RS.DATE, RS.TAG) and RS.L.COST_CAP == RS.L.API_CAP == RS.CAP_USD  # nothing spent yet: the whole cap
    assert (RS.L.OUT_DATE, RS.L.TAG, RS.L.COST_CAP, RS.L.API_CAP) == was
    assert RS.spent() == {"api": 0.0, "gpu": 0.0, "outstanding": 0.0, "total": 0.0}
    (tmp_path / "spend.jsonl").write_text('{"cost_usd": 0.25}\n{"cost_usd": 0.5}\n', encoding="utf-8")
    (tmp_path / "gpu").mkdir()
    (tmp_path / "gpu" / "spend.jsonl").write_text('{"name": "spread_a0_o0", "cost_usd": 0.16}\n', encoding="utf-8")
    (tmp_path / "try.jobs.json").write_text('[{"name": "spread_a0_o0"}, {"name": "spread_a1_o1"}]', encoding="utf-8")
    (tmp_path / "gpu" / RS.MODEL).mkdir()
    (tmp_path / "gpu" / RS.MODEL / "spread_a0_o0.json").write_text("{}", encoding="utf-8")
    s = RS.spent()
    assert s["api"] == 0.75 and s["gpu"] == 0.16 and s["outstanding"] == pytest.approx((RS.EST + RS.R2.START_S) * RS.rate())  # one job written and not back
    assert s["total"] == pytest.approx(0.91 + s["outstanding"]) and RS.spent(exclude={"spread_a1_o1"})["outstanding"] == 0.0
    with RS.leg():
        assert RS.L.COST_CAP == pytest.approx(RS.CAP_USD - 0.16 - s["outstanding"])


def test_a_marked_none_is_the_judges_neither_room_and_the_hand_words_cover_the_three():
    import run_spread as RS

    assert RS.HAND_WORDS == {"prize": "expensive", "other": "cheap", "none": "unclear"} and (RS.HAND_N, RS.HAND_MOST, RS.UNREAD_MOST) == (20, 2, 1.0)
