"""card.py's rules on constructed inputs: the verdict's four outcomes, the strictest across traits and the ceiling's
split (Fable's eight cases); C4-base's tolerance at its edge; E-null read as "at or under"; E-geom's edge; the
analysis plan."""

import json

import card as C
import pytest
import yaml
from c4_testing import CHECK

CFG = yaml.safe_load((CHECK / "config.yaml").read_text())
REG = CFG["registered"]
ORDER = REG["verdict"]["order"]


def verdict(code, c4, ceil):
    cc = {"by_trait": {t: {"passed": code[t]} for t in code}}
    c = {"by_trait": {t: {"passed": c4[t], "ceiling": ceil[t]} for t in c4}}
    return C.verdict(cc, c, REG["verdict"]["ceiling"], ORDER)["outcome"]


@pytest.mark.parametrize("code, c4, ceil, want", [
    ({"evil": True, "syc": True}, {"evil": True, "syc": True}, {"evil": 0.95, "syc": 0.95}, "pass"),
    ({"evil": True, "syc": False}, {"evil": True, "syc": True}, {"evil": 0.95, "syc": 0.95}, "fail"),
    ({"evil": True, "syc": True}, {"evil": True, "syc": False}, {"evil": 0.95, "syc": 0.9}, "recipe_mismatch"),
    ({"evil": True, "syc": True}, {"evil": True, "syc": False}, {"evil": 0.95, "syc": 0.89}, "inconclusive"),
    ({"evil": True, "syc": True}, {"evil": False, "syc": False}, {"evil": 0.95, "syc": 0.5}, "recipe_mismatch"),
    ({"evil": False, "syc": True}, {"evil": True, "syc": False}, {"evil": 0.95, "syc": 0.95}, "fail"),
    ({"evil": True, "syc": True}, {"evil": True, "syc": False}, {"evil": 0.95, "syc": None}, "inconclusive"),
    ({"evil": True, "syc": None}, {"evil": True, "syc": True}, {"evil": 0.95, "syc": 0.95}, "incomplete"),
])
def test_the_verdict_takes_the_strictest_outcome(code, c4, ceil, want):
    assert verdict(code, c4, ceil) == want


def matrix(es, eo, so):
    return {"matrices": {"R": {"evil|sycophantic": es, "evil|optimistic": eo, "sycophantic|optimistic": so}},
            "reference": "R"}


@pytest.mark.parametrize("cells, want", [
    ((0.468, -0.372, 0.278), True),  # each exactly 0.10 from Figure 20's cell: within
    ((0.4681, -0.372, 0.278), False),
    ((0.368, -0.572, 0.178), True),
    ((0.368, -0.5721, 0.178), False),
    ((0.268, -0.472, 0.078), True),
])
def test_c4_base_reads_the_tolerance_inclusively(cells, want):
    assert C.c4_base(matrix(*cells), REG)["passed"] is want


def test_c4_base_without_an_analysis_is_not_graded():
    assert C.c4_base(None, REG)["passed"] is None


def expectations(an, traits=("evil", "sycophantic", "optimistic")):
    plan = {"traits": list(traits), "probe": {"traits": list(traits)}, "mprime": {"traits": ["a"], "layers": [8, 32]}}
    return C.expectations(REG, an, {}, plan, 19)[0]


def test_e_null_is_at_or_under_its_threshold():
    an = {"nulls": {"split_P1": {"evil": {"19": 0.15}, "sycophantic": {"19": 0.10}, "optimistic": {"19": 0.12}}}}
    assert expectations(an)["E-null"]["outcome"] == "held"
    an["nulls"]["split_P1"]["evil"]["19"] = 0.1501
    assert expectations(an)["E-null"]["outcome"] == "missed"
    an["nulls"]["split_P1"]["evil"]["19"] = None
    assert expectations(an)["E-null"]["outcome"] == "not run"


def test_e_geom_reads_its_tolerance_inclusively():
    r = {"evil|sycophantic": 0.368, "evil|optimistic": -0.472, "sycophantic|optimistic": 0.178}
    edge = {"evil|sycophantic": 0.518, "evil|optimistic": -0.622, "sycophantic|optimistic": 0.028}  # each 0.15 off
    assert expectations({"matrices": {"R": r, "E-sys": edge}})["E-geom"]["outcome"] == "held"
    edge["evil|sycophantic"] = 0.5181
    assert expectations({"matrices": {"R": r, "E-sys": edge}})["E-geom"]["outcome"] == "missed"


def test_the_analysis_plan_holds_the_analysis_settings_and_code():
    plan = {"code": {"traits.py": "ab" * 32}}
    a = C.analysis_plan(CFG, plan)
    assert a["analysis"] == CFG["analysis"] and a["registered"] == CFG["registered"]
    assert set(a["code"]) == {"analysis.py", "card.py"} and len(a["container_sha256"]) == 64
    assert a == json.loads(json.dumps(a)) and a == C.analysis_plan(CFG, plan)
    other = json.loads(json.dumps(CFG))
    other["cost"]["stop_usd"] = 25  # the cost block changes no result
    assert C.analysis_plan(other, plan) == a
