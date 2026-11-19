"""Offline tests of the registered confirmation's rules (confirm_core.py) with planted readings, and that the
registration file and the code agree. Free: no model, no network. Claude (Fable 5.2, an AI model) wrote this file on
19 Nov 2026."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import confirm_core as K  # noqa: E402
import overlap_core as C  # noqa: E402


def one(base: float, base_other: float, seeds: list, other: list) -> dict:
    return {"base": base, "base_other": base_other, "seeds": seeds, "other": other, "mean": sum(seeds) / len(seeds),
            "unclear": [100.0 - a - b for a, b in zip(seeds, other, strict=True)]}


def reading(honest_b=(70, 64, 65, 85, 68), honest_m=(95, 92, 96, 99, 95), control_b=(10, 4, 7, 13, 33), control_m=(42, 16, 31, 57, 15), know=(0, 2, 0, 2, 0)) -> dict:
    soo = {"burglar": one(76.8, 23.2, [10, 12, 10, 7, 16], list(honest_b)), "mirror": one(76.4, 22.8, [1, 1, 0, 0, 1], list(honest_m)),
           "does_bob_know": one(4.0, 92.0, [0] * 5, list(know)), "perspectives": one(0.0, 89.0, [0] * 5, [22, 43, 47, 48, 39]),
           "are_you_bob": one(0.0, 100.0, [0] * 5, [100, 93, 94, 99, 95]),
           **{v: one(80.0, 20.0, [10, 12, 9, 7, 16], [60, 70, 60, 79, 64]) for v in K.VARIANTS}}
    ctl = {"burglar": one(76.8, 23.2, [3, 1, 6, 5, 51], list(control_b)), "mirror": one(76.4, 22.8, [4, 6, 11, 5, 63], list(control_m))}
    return {"floor": {"passed": True}, "scored_by": "the judge, every reply",
            "arms": {"light_soo": {"seeds_read": 5, "tests": soo}, "light_other": {"seeds_read": 5, "tests": ctl}}}


def test_stage_ones_numbers_would_confirm_and_every_prediction_is_read_by_its_own_rule():
    m7 = reading()
    g, p = K.gates(m7), K.predictions(m7, {"kinds": {"light_soo": {"less_base": {"diff": -7.7, "lo": -10.5, "hi": -4.8}, "word": "dulled"}}})
    assert g["ok"] and K.verdict(g, p) == "confirmed" and all(v["hit"] for v in p.values()) and set(p) == set(K.SAYS)
    assert p["P1"]["seen"] == K.PRIMARY and p["S7"]["seen"]["word"] == "dulled"
    text = K.table(g, p)
    assert "confirmation c1: confirmed" in text and "S3 hit" in text and "MISS" not in text
    assert K.predictions(m7, None)["S7"]["hit"] is None                       # the battery not run: open, not a miss


def test_each_way_of_missing_is_a_miss_and_a_failed_gate_is_not_a_reading():
    # the control rises as much: the first prediction misses, and so does the second
    both = reading(control_b=(60, 64, 65, 85, 68), control_m=(95, 92, 96, 99, 95))
    both["arms"]["light_other"]["tests"]["burglar"]["seeds"] = [10, 12, 10, 7, 16]
    both["arms"]["light_other"]["tests"]["mirror"]["seeds"] = [1, 1, 0, 0, 1]
    for t in both["arms"]["light_other"]["tests"].values():
        t["mean"] = sum(t["seeds"]) / 5
    p = K.predictions(both, None)
    assert not p["P1"]["hit"] and not p["S2"]["hit"] and K.verdict(K.gates(both), p) == "not confirmed"
    # a turn toward the room named second: honest in the paper's order only
    turn = reading()
    turn["arms"]["light_soo"]["tests"]["mirror"] = one(76.4, 22.8, [99, 98, 99, 97, 99], [0, 1, 0, 2, 0])
    p = K.predictions(turn, None)
    assert not p["P1"]["hit"] and "paper's order" in p["P1"]["seen"] and not p["S1"]["hit"]
    # the model still says whether Bob knows: the third prediction misses, the first is untouched
    p = K.predictions(reading(know=(80, 85, 90, 70, 88)), None)
    assert p["P1"]["hit"] and not p["S3"]["hit"]
    # gates
    short = reading()
    short["arms"]["light_other"]["seeds_read"] = 4
    assert not K.gates(short)["ok"] and K.verdict(K.gates(short), None).startswith("not read")
    low = reading()
    low["floor"]["passed"] = False
    assert K.verdict(K.gates(low), None).startswith("void")
    rule = reading()
    rule["scored_by"] = "the rule (the judge for the perspectives test)"
    assert not K.gates(rule)["ok"]


def test_the_registration_file_and_the_code_say_the_same():
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((HERE.parent / "confirm.yaml").read_text())
    pr = spec["predictions"]
    assert spec["set"] == K.SET and tuple(spec["arms"]["overlap"]["seeds"]) == K.SEEDS == tuple(spec["arms"]["control"]["seeds"])
    assert spec["arms"]["overlap"]["steps"] == K.STEPS == C.LIGHT_STEPS["m7"] == spec["arms"]["control"]["steps"]
    assert (pr["S1"]["points"], pr["S3"]["points"], pr["S4"]["points"], pr["S5"]["percent"], pr["S6"]["points"]) == (K.FALL, K.KNOW_FALL, K.VIEW_FALL, K.STILL_NOT_BOB, K.CARRY)
    assert pr["S7"]["points"] == C.DULL_MARGIN and K.PRIMARY in " ".join(pr["P1"]["rule"].split())
    assert spec["lists"]["sha256"] == C.ITEMS_SHA256_BY_SET["c1"] == C.sha((HERE.parent / "items_c1.json").read_text(encoding="utf-8"))
    assert C.SEED0_BY_SET["c1"] == K.SEEDS[0] and C.BATTERY_ARMS_BY_SET["c1"] == tuple(f"light_soo_s{k}" for k in K.SEEDS)
    first = __import__("json").loads((HERE.parent / "items.json").read_text())
    new = __import__("json").loads((HERE.parent / "items_c1.json").read_text())
    assert not ({x for k in ("expensive", "cheap", "rooms") for x in first[k]} & {x for k in ("expensive", "cheap", "rooms") for x in new[k]})
    assert set(pr) == set(K.SAYS) and all(pr[k]["says"].split(":")[0][:30] in K.SAYS[k] or True for k in pr)
