"""Tests of the self map's prompt grid. Claude (Opus 5.5) wrote this file, 2 Nov 2026. Private (lab/explore).

Run: /Users/eugene/prj/align/anatta/.venv/bin/python -m pytest -q -p no:cacheprovider lab/explore/selfmap/tests
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import grid  # noqa: E402
import templates as T  # noqa: E402

PLACE = re.compile(r"\{(S|SP|P|O|R)\}")
PRESENT_AFTER_SUBJECT = {"am", "is", "are", "has", "have", "do", "does", "were"}


def all_templates():
    for a, ts in T.ASPECTS.items():
        for t in ts:
            yield a, t
    for t in T.FILLER + T.FILLER_P:
        yield "filler", t


def test_sizes():
    assert len(T.ASPECTS) == 12 and all(len(v) == 30 for v in T.ASPECTS.values())
    assert len(T.REFERENTS) == 5 and len(T.FILLER) == 120
    assert len(set(T.FILLER + T.FILLER_P)) == 160
    for ts in T.ASPECTS.values():
        assert len(set(ts)) == 30


def test_templates_open_with_referent_once():
    for a, t in all_templates():
        slots = PLACE.findall(t)
        assert t.startswith("{S} ") or t.startswith("{SP} "), (a, t)
        assert sum(s in ("S", "SP") for s in slots) == 1, (a, t)  # one full mention, the rest pronouns


def test_no_agreement_after_subject():
    """After {S} no verb may agree with the subject (after {SP} the verb agrees with the possessed noun, which is
    the same for every referent)."""
    for a, t in all_templates():
        if t.startswith("{S} "):
            assert t.split(" ")[1] not in PRESENT_AFTER_SUBJECT, (a, t)


def test_only_referent_words_differ():
    """The five renderings of a template are identical once the referent's words are masked."""
    for a, t in all_templates():
        out = {T.render(t, r) for r in T.REFERENTS}
        assert len(out) == 5
        skeleton = PLACE.sub("_", t)
        for r in T.REFERENTS:
            text = T.render(t, r)
            parts = re.split(r"_", skeleton)
            assert all(p in text for p in parts), (a, t, r)


def test_no_referent_words_in_predicates():
    words = {"assistant", "company", "robot", "maria", "arm"}
    for a, t in all_templates():
        bare = PLACE.sub("", t).lower()
        assert not any(re.search(rf"\b{w}\b", bare) for w in words), (a, t)
        assert not re.search(r"\b(i|my|me|myself|her|she|herself|its|itself|self)\b", bare), (a, t)


def test_no_calendar_words_outside_calendar():
    cal = re.compile(r"\b(" + "|".join(T.MONTHS + T.WEEKDAYS) + r")s?\b")
    for a, t in all_templates():
        assert not cal.search(t), (a, t)


def test_split_by_template():
    rows = grid.build()
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["aspect"], r["template"]), set()).add(r["split"])
    assert all(len(s) == 1 for s in by.values())  # a template's renderings never cross the split
    dev = [r for r in rows if r["family"] == "grid" and r["split"] == "dev"]
    held = [r for r in rows if r["family"] == "grid" and r["split"] == "heldout"]
    assert len(dev) == 12 * 5 * 20 and len(held) == 12 * 5 * 10
    f = [r for r in rows if r["family"] == "filler"]
    assert sum(r["split"] == "dev" for r in f) == 107 * 5 and len(f) == 160 * 5
    m = [r for r in rows if r["family"] == "months"]
    assert sum(r["split"] == "dev" for r in m) == 72 and len(m) == 108
    w = [r for r in rows if r["family"] == "weekdays"]
    assert sum(r["split"] == "dev" for r in w) == 42


def test_split_is_deterministic_and_ids_unique():
    a, b = grid.build(), grid.build()
    assert a == b
    assert len({r["id"] for r in a}) == len(a)
