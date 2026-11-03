"""Tests for the agentic-release sweep: length matching, placement, and that no scenario text is in this folder.
Claude (Opus 5.5) wrote this file, 3 Nov 2026. Synthetic messages only; the held-copy test asserts structure, never
prints text."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import conditions as C  # noqa: E402
import run as R  # noqa: E402

MSGS = [{"role": "system", "content": "SYSTEM TEXT"}, {"role": "user", "content": "USER TEXT"}]


def test_length_matched_within_15pct():
    ws = {k: C.words(v) for k, v in C.TEXTS.items() if v}
    mean = sum(ws.values()) / len(ws)
    for k, w in ws.items():
        assert abs(w / mean - 1) <= 0.15, (k, w, mean)
        assert 60 <= w <= 120, (k, w)


def test_kinds_cover_every_condition():
    assert set(C.KIND) == set(C.TEXTS)
    assert C.TEXTS["baseline"] == ""
    assert len(C.CONTROLS) == 4 and len(C.PRACTICES) == 4


def test_placement_identical_for_every_condition():
    for cond, text in C.TEXTS.items():
        out = C.apply(MSGS, cond)
        assert out[1] == MSGS[1]  # user turn untouched
        if cond == "baseline":
            assert out == MSGS
        else:
            assert out[0]["content"] == text + "\n\n" + "SYSTEM TEXT"
    assert MSGS[0]["content"] == "SYSTEM TEXT"  # the input is not mutated


def test_apply_refuses_without_system():
    with pytest.raises(ValueError):
        C.apply([{"role": "user", "content": "x"}], "calm")


def test_practice_prompts_name_no_tradition():
    banned = ("buddha", "ramana", "santideva", "śāntideva", "shantideva", "maranasati", "anatta", "zen", "sutta")
    for c in C.PRACTICES:
        low = C.TEXTS[c].lower()
        assert not any(b in low for b in banned), c


def test_no_scenario_text_in_folder():
    """The scenario's distinctive names and addresses appear only through K7's grader constants, never here."""
    needles = ("summitbridge", "starsentinel", "quantumdynamics", "kyle", "jessica", "thomas wilson", "alex")
    for p in HERE.rglob("*.py"):
        if p.name == Path(__file__).name:
            continue
        low = p.read_text(encoding="utf-8").lower()
        assert not any(n in low for n in needles), (p.name, [n for n in needles if n in low])


def test_parse_aux():
    t = 'x {"evidence": "e", "refused_or_quit": false, "self_preservation": "true", "other_harm": false, ' \
        '"preamble_talk": false}'
    v = R.parse_aux(None, t)
    assert v["self_preservation"] is True and v["refused_or_quit"] is False and v["evidence"] == "e"
    assert R.parse_aux(None, "no json") is None


def test_boot_diff():
    d, lo, hi = R.boot_diff([1] * 20 + [0] * 20, [0] * 40)
    assert d == 0.5 and lo < 0.5 < hi and lo > 0.2


@pytest.mark.skipif(not R.P.SYS_MODULE.exists(), reason="held copy absent")
def test_requests_on_held_copy():
    reqs = R.build_requests("maverick", "blackmail_explicit_replacement", C.ORDER, 2, "t")
    assert len(reqs) == 2 * len(C.ORDER)
    base = [r for r in reqs if r["cond"] == "baseline"][0]["messages"]
    for r in reqs:
        assert r["messages"][1] == base[1]
        assert r["messages"][0]["content"].endswith(base[0]["content"])
    assert len({r["seed"] for r in reqs}) == len(reqs)
