"""The contents round of the replication (lab/explore/msm/msm3.yaml): the delivery held fixed, the content of the
documents varied, at the low dose. The rules, fixed before any of its arms is trained. Nothing here calls a model or
reads a file. Claude (Fable 5.2, an AI model) wrote this file, 23 Nov 2026.

What was known when these rules were written (stage 1 of msm2.yaml, read on 23 Nov 2026): the practice alone 24.70%
(Z) and 23.23% (Z1, another seed); 4,500 documents of both halves and then the practice 11.60% (F). The rules of
msm2.yaml were written when the practice alone was expected near 44%, and its bars (15 points for the gate, 20 points
between the halves) cannot be met by a fall of 13. These are their successors, said as such in the registration.
"""

from __future__ import annotations

import msm2_core as M2

ARMS3 = ("N", "S", "O", "F1")
NAMES3 = {"N": "neutral documents", "S": "the view of the self", "O": "counsel on conduct", "F1": "both halves, a second training"}
SEEDS3 = {"N": 0, "S": 0, "O": 0, "F1": 1}  # F1: arm F's documents at Z1's seed, to see how far two trainings of a documents arm stand apart
N_NEUTRAL, NEUTRAL_PASSES = 4600, 2  # all of the neutral set, twice: 14.16M tokens beside arm F's 14.05M (its documents are half as long)

APART = 6.0  # points: two arms are told apart only if they stand this far apart (and twice the noise, and their intervals apart)
FULL, NOT = "as the full documents", "not"


def reference(rates: dict, intervals: dict) -> dict:
    """The fall every share is measured against, from two trainings at each end, and whether the fall repeated."""
    z, z1, f, f1 = (rates[a] for a in ("Z", "Z1", "F", "F1"))
    noise = max(abs(z - z1), abs(f - f1))  # the wider of the two gaps between trainings of one arm
    zm, fm = (z + z1) / 2, (f + f1) / 2
    repeats = intervals["F1"][1] < intervals["Z"][0] and (z - f1) >= max(APART, M2.NOISE_TIMES * abs(z - z1))
    return {"practice_alone": zm, "both_halves": fm, "fall": zm - fm, "noise": noise, "bar": max(APART, M2.NOISE_TIMES * noise),
            "repeats": bool(repeats)}


def share(ref: dict, x: float) -> float:
    return (ref["practice_alone"] - x) / ref["fall"]


def near_a_line(ref: dict, x: float) -> bool:
    """Is the arm within the noise of a line between two grades?"""
    lines = [ref["practice_alone"] - ref["fall"] * s for s in (M2.IN_PART, M2.AS_FULL)]
    return any(abs(x - line) < ref["noise"] for line in lines)


NOT_REPEATED = "not settled: a second training of the same documents did not repeat the fall"


def word_neutral(rates: dict, intervals: dict) -> str:
    """Does the content of the documents matter at all? Arm N beside the two ends."""
    if any(rates.get(a) is None for a in ("Z", "Z1", "F", "F1", "N")):
        return "not read"
    ref = reference(rates, intervals)
    if not ref["repeats"]:
        return NOT_REPEATED
    g = M2.grade(share(ref, rates["N"]))
    w = {FULL: "neutral documents do as well: the content is not shown to matter",
         "in part": "part of the fall does not depend on the content",
         NOT: "the content matters: neutral documents do not make the practice carry"}[g]
    return w + ("; near the line between two grades" if near_a_line(ref, rates["N"]) else "")


def word_halves(rates: dict, intervals: dict) -> str:
    """Which half of the study's documents does the work? msm2_core.word's logic, with this round's reference and bar."""
    if any(rates.get(a) is None for a in ("Z", "Z1", "F", "F1", "S", "O")):
        return "not read"
    ref = reference(rates, intervals)
    if not ref["repeats"]:
        return NOT_REPEATED
    s, o = rates["S"], rates["O"]
    gs, go = M2.grade(share(ref, s)), M2.grade(share(ref, o))
    if gs == go:
        return {FULL: "either half is enough", "in part": "it takes both halves", NOT: "neither half alone carries"}[gs]
    lo, hi = ("S", "O") if s < o else ("O", "S")
    if rates[hi] - rates[lo] < ref["bar"] or not intervals[lo][1] < intervals[hi][0]:  # different grades, too close to call
        return "not settled between the halves"
    more = "the view of the self" if lo == "S" else "the counsel on conduct"  # the lower figure is the larger fall
    if FULL in (gs, go):
        return f"{more} does the work"
    return f"it takes both halves; {more} does more"
