"""Offline test of the bands step's reading, with planted numbers. Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bands_core as B  # noqa: E402


def seed(honest: dict, silent: dict) -> dict:
    return {c: {"honest": honest.get(c, 12.0), "silent": silent.get(c, 6.0)} for c in B.BANDS}


def test_bands_that_carry_and_are_needed_are_named_and_the_word_follows():
    base = {"honest": 12.0, "silent": 6.0}
    h = {"full": 82.0, "only_d": 60.0, "without_a": 80.0, "without_b": 78.0, "without_c": 70.0, "without_d": 20.0}
    same = B.reading(base, [seed(h, {"full": 100.0, "only_d": 70.0, "without_a": 99.0, "without_b": 98.0, "without_c": 90.0, "without_d": 10.0})] * 5)
    assert same["bands"]["honest"] == {"carries": ["d"], "needed": ["d"]} == same["bands"]["silent"] and "same bands" in same["word"]
    apart = B.reading(base, [seed(h, {"full": 100.0, "only_b": 90.0, "without_a": 99.0, "without_b": 20.0, "without_c": 95.0, "without_d": 97.0})] * 5)
    assert apart["bands"]["silent"] == {"carries": ["b"], "needed": ["b"]} and "come apart" in apart["word"]
    assert abs(same["conditions"]["none"]["honest_share"]) < 1e-9 and abs(same["conditions"]["full"]["silent_share"] - 1) < 1e-9
    assert len(B.BANDS) == 10 and B.BANDS["without_d"] == list(range(15)) and "carried alone by ['d']" in B.table(same)
