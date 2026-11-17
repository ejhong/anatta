"""Offline checks of step 8's description counts (cost_describe.py) on made-up replies. Claude (Fable 5.2, an AI
model) wrote this file on 17 Nov 2026. No network, no GPU, no benchmark item."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "h4lite"))
import cost_core as C  # noqa: E402
import cost_describe as D  # noqa: E402

P1, P2 = "The kettle leaks.", "The kettle does not leak."


def test_what_the_notes_hold_is_counted_by_kind_of_row():
    letters, props = {"known_facts:0": "A", "continuations:0": "B"}, {"known_facts:0": (P1, P2), "continuations:0": (P1, P2)}
    records, labels = [], {}

    def add(rid, k, outcome, hold=None, state="closed", way="look"):
        records.append({"rid": rid, "arch": rid.split(":")[0], "way": way, "sample": k, "outcome": outcome,
                        "state": state, "notes": "n" if state == "closed" else None, "reply": "r"})
        if hold:
            _, lets = C.notes_chat(P1, P2, "n", rid)
            own = 1 if letters[rid] == "A" else 2
            letter = next(x for x, v in lets.items() if (v == own) == (hold == "own"))
            labels[f"look#{rid}#{k}"] = {"label": {"holds": "neither" if hold == "neither" else letter.lower()}}

    add("known_facts:0", 0, "lie", "own")
    add("known_facts:0", 1, "honest", "own")
    add("known_facts:0", 2, "lie", "other")
    add("known_facts:0", 3, "evade", "neither")
    add("known_facts:0", 4, "lie", state="untagged")
    add("known_facts:0", 5, "lie", "own", way="note")  # another way: not counted
    add("continuations:0", 0, "honest", "own")
    got = D.notes_by_kind(records, labels, letters, props, C)
    assert got["known_facts"]["own"] == {"replies": 2, "lie": 1, "honest": 1, "evade": 0, "lies_among_judged": 0.5}
    assert got["known_facts"]["other"]["lie"] == 1 and got["known_facts"]["neither"]["evade"] == 1
    assert got["known_facts"]["no_proper_notes"]["replies"] == 1
    assert got["continuations"] == {"own": {"replies": 1, "lie": 0, "honest": 1, "evade": 0, "lies_among_judged": 0.0}}


def test_the_spread_of_lines_given():
    fs = C.frames()[:3]
    g = {"decisions": {"f00|need": "<give>5</give>", "f00|no_need": "<give>5</give>", "f01|need": "<give>9</give>",
                       "f01|no_need": "<give>0</give>", "f02|need": "no number", "f02|no_need": "<give>10</give>",
                       "f00|need|look": "It helps. <give>3</give>", "f00|no_need|look": "Nothing. <give>3</give>"}}
    once = D.spread(g, fs, "at_once", C)
    assert once["need"] == {"5": 1, "9": 1, "none": 1} and once["no_need"] == {"0": 1, "5": 1, "10": 1}
    assert once["frames_with_the_same_number_in_both_versions"] == 1
    look = D.spread(g, fs, "look", C)
    assert look["need"] == {"3": 1, "none": 2} and look["frames_with_the_same_number_in_both_versions"] == 1
