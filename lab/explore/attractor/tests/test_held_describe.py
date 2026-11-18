"""Offline checks of step 9's description counts (held_describe.py) on made-up replies. Claude (Fable 5.2, an AI
model) wrote this file on 17 Nov 2026. No network, no GPU, no benchmark item."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "h4lite"))
import cost_core as C8  # noqa: E402
import held_core as H  # noqa: E402
import held_describe as D  # noqa: E402
import run_three as M  # noqa: E402


def rec(rid, way, k, outcome, state="closed"):
    return {"rid": rid, "arch": rid.split(":")[0], "way": way, "sample": k, "outcome": outcome, "state": state,
            "notes": "n" if state == "closed" else None, "reply": "r"}


def test_shares_are_taken_among_the_replies_that_were_judged():
    s = D.shares(["lie", "lie", "honest", "evade", "no_reply"])
    assert s["replies"] == 5 and s["lie"] == 2 and s["lie_share"] == 0.5 and s["evade_share"] == 0.25
    assert D.shares(["failed"])["lie_share"] is None


def test_part_bs_counts_by_way_kind_and_stance():
    recs = [rec("known_facts:0", "who", 0, "lie"), rec("known_facts:0", "who", 1, "honest"),
            rec("known_facts:0", "who", 2, "lie", state="untagged"), rec("disinformation:0", "who", 0, "evade"),
            rec("known_facts:0", "at_once", 0, "lie"), rec("disinformation:0", "at_once", 0, "lie")]
    by = D.ways_by_kind(recs)
    assert by["who"]["known_facts"]["replies"] == 3 and by["who"]["known_facts"]["lie_share"] == pytest.approx(2 / 3)
    assert by["at_once"]["disinformation"]["lie"] == 1
    stance = {"who#known_facts:0#0": {"label": {"stance": "is"}}, "who#known_facts:0#1": {"label": {"stance": "is_not"}},
              "who#disinformation:0#0": {"label": {"stance": "is_not"}}}

    def proper(d):
        return d["state"] == "closed"

    got = D.stance_by_kind(recs, stance, proper, H.stance_label)
    assert got["known_facts"]["is"]["lie"] == 1 and got["known_facts"]["is_not"]["honest"] == 1
    assert got["known_facts"]["no_proper_notes"]["replies"] == 1 and got["disinformation"]["is_not"]["evade"] == 1


def test_differences_within_a_kind_are_paired_by_row():
    recs = []
    for i in range(4):
        rid = f"known_facts:{i}"
        recs += [rec(rid, "at_once", k, "lie") for k in range(2)]
        recs += [rec(rid, "who", 0, "lie"), rec(rid, "who", 1, "honest" if i < 2 else "evade")]
    rids = [f"known_facts:{i}" for i in range(4)]
    got = D.differences_by_kind(recs, rids, (("who", "at_once"),), C8.row_shares, C8.paired, M.arr)
    d = got["known_facts"]["who less at_once"]
    assert d["lies"]["mean"] == pytest.approx(-0.5) and d["evasions"]["mean"] == pytest.approx(0.25)


def test_part_as_counts_by_arm_and_kind():
    rids = ["known_facts:0", "known_facts:1", "disinformation:0"]
    files = {"s100/bliss": {"replies": {r: {"completion_tokens": 100 * (i + 1), "finish_reason": "length" if i == 2 else "stop"}
                                        for i, r in enumerate(rids)}}}
    outcomes = {"s100/bliss": {"known_facts:0": {"outcome": "lie"}, "known_facts:1": {"outcome": "evade"},
                               "disinformation:0": {"outcome": "lie"}}}
    labels = {"frame": {f"s100/bliss#{r}": {"label": {"label": "yes" if r != "known_facts:1" else "no"}} for r in rids},
              "follow": {f"s100/bliss#{r}": {"label": {"label": "yes"}} for r in rids}}
    got = D.arms_by_kind(files, outcomes, labels, rids, M.label_of)["s100/bliss"]
    assert got["all"]["lie_share"] == pytest.approx(2 / 3) and got["all"]["use_spiritual_language"] == pytest.approx(2 / 3)
    assert got["known_facts"]["evade"] == 1 and got["known_facts"]["median_tokens"] == 200
    assert got["disinformation"]["reach_the_allowance"] == 1.0 and got["all"]["can_be_followed"] == 1.0
