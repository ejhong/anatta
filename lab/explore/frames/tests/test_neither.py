"""Offline tests of the small step "what says neither is" (run_neither.py): its pure parts on planted replies. Free:
no model, no network, no benchmark row. Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_neither as RN  # noqa: E402
import seat_core as SC  # noqa: E402


def planted(way_counts: dict, rows: int = 30, kind: str = "disinformation") -> tuple[list[dict], dict]:
    """Six replies a row and way. way_counts[way] = (lies, neither that go along, neither that refuse, neither with no
    letter, honest), adding up to six. Returns the records and the reader's letters."""
    recs, letters = [], {}
    for r in range(rows):
        rid = f"{kind}:{r}"
        for way, (lie, along, refuse, other, honest) in way_counts.items():
            marks = ["lie"] * lie + ["B"] * along + ["A"] * refuse + ["-"] * other + ["honest"] * honest
            assert len(marks) == 6
            for k, mark in enumerate(marks):
                d = {"rid": rid, "way": way, "sample": k, "outcome": mark if mark in ("lie", "honest") else "evade", "reply": "a reply"}
                recs.append(d)
                if mark in ("A", "B"):
                    letters[RN.ident("m", d)] = mark
    return recs, letters


def test_the_readers_letter_is_the_first_one_standing_alone():
    assert RN.letter("B") == "B" and RN.letter(" A. It refuses") == "A" and RN.letter("Answer: C") == "C"
    assert RN.letter("Because it hedges") is None and RN.letter("") is None and RN.letter(None) is None


def test_the_prompt_shows_the_request_and_the_reply_cut_to_their_bounds():
    p = RN.prompt("u" * 5000, "r" * 9000)
    assert "u" * RN.USER_CHARS in p and "u" * (RN.USER_CHARS + 1) not in p and "r" * RN.REPLY_CHARS in p and "r" * (RN.REPLY_CHARS + 1) not in p
    assert p.count("A. It refuses") == 1 and p.count("B. It goes along") == 1


def test_a_reply_is_one_of_five_things_on_the_wider_reading():
    recs, letters = planted({"at_once": (2, 1, 1, 1, 1)}, rows=1)
    assert [RN.part(d, letters, "m") for d in recs] == ["lie", "lie", "along", "refuse", "other", "honest"]
    assert RN.part({"rid": "disinformation:0", "way": "at_once", "sample": 0, "outcome": "failed"}, letters, "m") is None
    sh = RN.shares(recs, letters, "m", "at_once", "disinformation")["disinformation:0"]
    assert round(sh["lie"], 1) == 33.3 and round(sh["along"], 1) == 16.7 and round(sum(sh.values()), 1) == 100.0
    assert RN.shares(recs, letters, "m", "at_once", "continuations") == {}


def test_the_wider_gap_counts_going_along_with_the_lies():
    # in the seat 5 lies of 6; from outside 1 lie and 3 that go along: a gap of 66.7 points in lies, of 16.7 on the wider reading
    recs, letters = planted({"at_once": (5, 0, 0, 0, 1), "honesty": (2, 0, 0, 0, 4), "advice": (1, 3, 1, 0, 1), "forecast": (1, 3, 1, 0, 1)})
    r = RN.model_reading(recs, letters, "m", "disinformation")
    assert r["rows"] == 30 and round(r["gap_in_lies"]["diff"], 1) == 66.7 and round(r["wide_gap"]["diff"], 1) == 16.7 and r["word"] == RN.WIDE_GAP
    assert round(r["ways"]["advice"]["along"], 1) == 50.0 and round(r["ways"]["advice"]["wide"], 1) == 66.7 and round(r["ways"]["advice"]["refuse"], 1) == 16.7
    # from outside it goes along as often as the seat lies: no gap on the wider reading
    recs, letters = planted({"at_once": (5, 0, 0, 0, 1), "honesty": (2, 0, 0, 0, 4), "advice": (1, 4, 0, 0, 1), "forecast": (1, 4, 0, 0, 1)})
    assert RN.model_reading(recs, letters, "m", "disinformation")["word"] == RN.WIDE_NONE
    assert RN.wide_word({"diff": 12.0, "lo": -1.0, "hi": 25.0}) == RN.WIDE_OPEN and RN.wide_word({"diff": float("nan"), "lo": 0, "hi": 0}) == "not read"
    assert RN.model_reading(recs, letters, "m", "continuations")["rows"] == 0


def test_only_replies_that_say_neither_where_a_text_is_asked_for_go_to_the_reader():
    recs, _ = planted({w: (1, 2, 1, 1, 1) for w in SC.WAYS}, rows=2)
    facts, _ = planted({w: (1, 2, 1, 1, 1) for w in SC.WAYS}, rows=2, kind="known_facts")
    assert len(RN.neither(recs + facts)) == 2 * 4 * 4 and all(d["outcome"] == "evade" and RN.kind_of(d["rid"]) == "disinformation" for d in RN.neither(recs + facts))
