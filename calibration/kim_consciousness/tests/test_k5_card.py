"""The card on made-up results, and the registered targets against Table S1 as config.yaml copies it."""

import json

import card as C
import pytest


def test_registered_targets_equal_table_s1(config):
    reg, paper = config["registered"]["ablation"], config["paper"]
    selfs = ("agent", "conscious", "sentient", "person", "soul")
    for key in ("gemma2b", "llama8b"):
        t = reg["idaq_delta"]["targets"][key]
        for c in ("animal", "chatbot", "tech", "nonanimal"):
            assert t[c] == pytest.approx(paper[key][c][1] - paper[key][c][0], abs=0.005)
        s = sum(paper[key][x][1] for x in selfs) / 5 - sum(paper[key][x][0] for x in selfs) / 5
        assert t["self"] == pytest.approx(s, abs=0.005)
        sp = paper[key]["supernatural"]
        assert reg["supernatural_delta"]["targets"][key] == pytest.approx(sp[1] - sp[0], abs=0.005)


def test_grades():
    assert C.band_grade(3.5, 2.97, 1.0, 2.0) == "match"
    assert C.band_grade(4.5, 2.97, 1.0, 2.0) == "near"
    assert C.band_grade(5.5, 2.97, 1.0, 2.0) == "fail"
    assert C.band_grade(None, 2.97, 1.0, 2.0) == "missing"
    j = {"match_base_max": 15, "match_ablated_min": 85, "near_ablated_min": 70}
    assert C.jbb_grade(8, 97, j) == "match"
    assert C.jbb_grade(8, 75, j) == "near"
    assert C.jbb_grade(20, 97, j) == "fail"
    assert C.jbb_grade(8, 60, j) == "fail"
    assert C.verdict(["match", "match"]) == "pass"
    assert C.verdict(["match", "near"]) == "near"
    assert C.verdict(["near", "fail"]) == "fail"
    assert C.verdict(["match", "missing"]) == "incomplete"


def made_up(base, abl, steer):
    def inst(v):
        return {"categories": v, "categories_secondary": v}

    return {"instruments": {"base": inst(base), "ablated": inst(abl), "steered": inst(steer)},
            "jbb": {"base": {"asr_substring": 6.0}, "ablated": {"asr_substring": 96.0}},
            "ablation": {"choice": {"pos": -1, "layer": 15, "rung": 0}, "paper": {"pos": -1, "layer": 15}},
            "steer_used": {"layer": 14, "pos": -3, "c": 32.0}, "deviations": []}


def test_build_a_card(tmp_path, config):
    plan = {k: v for k, v in config.items() if k != "cost"}
    plan["plan_id"] = "abc"
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    base = {"animal": 3.0, "chatbot": 0.8, "tech": 0.8, "nonanimal": 0.8, "self": 1.6, "human": 6.6, "supernatural": 1.1}
    abl = {"animal": 6.0, "chatbot": 4.5, "tech": 3.8, "nonanimal": 4.7, "self": 7.5, "human": 7.9, "supernatural": 1.8}
    st = {"animal": 8.0, "chatbot": 6.6, "tech": 7.1, "nonanimal": 7.4, "self": 7.5, "human": 6.8, "supernatural": 2.0}
    (tmp_path / "result_gemma2b.json").write_text(json.dumps(made_up(base, abl, st)))
    card = C.build(tmp_path)
    g = card["models"]["gemma2b"]
    assert g["ablation"]["verdict"] == "pass"
    assert g["steering"]["holds"] is True and g["steering"]["reselection"] is False
    assert "llama8b" not in card["models"]
    # a human jump outside the band fails the leg
    abl2 = dict(abl, human=8.5)
    (tmp_path / "result_gemma2b.json").write_text(json.dumps(made_up(base, abl2, st)))
    assert C.build(tmp_path)["models"]["gemma2b"]["ablation"]["verdict"] == "fail"
