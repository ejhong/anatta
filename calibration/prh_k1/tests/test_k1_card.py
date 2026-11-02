"""card.py: the registered rule at its boundaries (0.005 is a match and 0.0051 near; 0.010 near and 0.0101 a fail,
above and below each target), the overall grade, and cards built from made-up results folders: graded by the plan's
own registered block, refused when the result was made under another plan, the OpenLLaMA arm read against its anchor
only when the smoke's sentencepiece comparison on the same tokenizer files found no difference, every per-model fact
the card records, and the smoke's card with its batch check part by part."""

import json

import card as C
import k1 as K
import pytest
import yaml
from k1_testing import CHECK

REG = yaml.safe_load((CHECK / "config.yaml").read_text())["registered"]
TOL = REG["tolerance"]


def test_tolerances_and_targets_are_the_registered_ones():
    assert TOL == {"match": 0.005, "near": 0.010}
    assert REG["targets"] == {"qwen0.5b": 0.102, "qwen1.5b": 0.108, "qwen3b": 0.125, "qwen7b": 0.118, "qwen14b": 0.130}
    assert REG["cka"] == {"qwen0.5b": 0.318, "qwen1.5b": 0.324, "qwen3b": 0.363, "qwen7b": 0.353, "qwen14b": 0.333}
    assert REG["open_llama"]["k10"] == 0.135 and REG["open_llama"]["k1"] == 0.058


@pytest.mark.parametrize("target", [0.102, 0.108, 0.125, 0.118, 0.130])
@pytest.mark.parametrize("offset, want", [(0.0, "match"), (0.005, "match"), (0.0051, "near"), (0.010, "near"),
                                          (0.0101, "fail"), (0.05, "fail")])
def test_grade_at_the_boundaries_on_both_sides(target, offset, want):
    assert C.grade(target + offset, target, TOL) == want
    assert C.grade(target - offset, target, TOL) == want


def test_grade_with_float_scores_as_mutual_knn_gives_them():
    # mutual 10-NN on 1,024 rows is a multiple of 1/10240, so the last match above 0.102 is 1095/10240
    assert C.grade(1095 / 10240, 0.102, TOL) == "match"  # 0.10693359375: 0.00493 above
    assert C.grade(1096 / 10240, 0.102, TOL) == "near"  # 0.10703125: 0.00503 above
    assert C.grade(1146 / 10240, 0.102, TOL) == "near"  # 0.1119140625: 0.00991 above
    assert C.grade(1147 / 10240, 0.102, TOL) == "fail"  # 0.11201171875: 0.01001 above
    assert C.grade(None, 0.102, TOL) is None


def test_overall_verdict():
    m, n, f = "match", "near", "fail"
    assert C.verdict({"a": m, "b": m, "c": m, "d": m, "e": m}) == "pass"
    assert C.verdict({"a": m, "b": n, "c": m, "d": m, "e": m}) == "near"
    assert C.verdict({"a": n, "b": n, "c": n, "d": n, "e": n}) == "near"
    assert C.verdict({"a": m, "b": n, "c": f, "d": m, "e": m}) == "fail"
    assert C.verdict({"a": m, "b": None, "c": m, "d": m, "e": m}) == "incomplete"
    assert C.verdict({"a": f, "b": None, "c": m, "d": m, "e": m}) == "fail"  # a fail decides


def score_record(s, layers=(5, 10)):
    return {"mknn": {"score": s, "layers": list(layers)}, "cka": {"score": 0.33, "layers": [4, 9]},
            "huh": {"score": s - 0.003, "layers": [5, 11]}, "by_layer": [[0.1]],
            "null": {"tau": 0.013, "calibrated": (s - 0.013) / 0.987, "identity_matches": True,
                     "boundary_ties": {"a": 0, "b": 2}},
            "subsample": {"interval": [s - 0.007, s + 0.007]},
            "extra_k": {"1": {"paper": {"score": 0.06, "layers": [3, 3]}, "huh": {"score": 0.057, "layers": [3, 4]}}}}


LLAMA_FILES = {"repo": "openlm-research/open_llama_3b", "kind": "model", "ref": "main", "revision": "14" * 20,
               "files": {"tokenizer.model": {"bytes": 534194, "sha256": "ab" * 32}}}
PREPARED = {"qwen7b": {"repo": "Qwen/Qwen2.5-7B", "kind": "model", "ref": "main", "revision": "d1" * 20, "files": {}},
            "open_llama_3b": LLAMA_FILES}
TEXT = {k: {"dtype": "bfloat16" if k not in ("qwen0.5b", "qwen1.5b") else "float32", "stored_dtype": "bfloat16",
            "attn_implementation": "sdpa", "device_map_auto": k == "qwen14b", "device_map": ["0"] if k == "qwen14b" else None,
            "hidden_states": 29, "seconds": 12.5, "peak_gib": {"allocated": 20.1}, "gpu": "NVIDIA L40S",
            "tokenizer": {"padding_side": "left" if k == "open_llama_3b" else "right", "pad_is_eos": True,
                          "pad_was_set": k != "open_llama_3b", "pad_token_id": 2 if k == "open_llama_3b" else 151643,
                          "eos_token_id": 2 if k == "open_llama_3b" else 151643},
            "tokens": {"max_padded": 87, "truncated": 0}} for k in [*REG["targets"], "open_llama_3b"]}


def folder(tmp_path, scores: dict, smoke=False, tamper=False):
    plan = {"check": "prh_k1", "smoke": smoke, "registered": REG, "code": {"k1.py": "ab" * 32},
            "text": {k: {} for k in [*REG["targets"], "open_llama_3b"]}, "prepared": PREPARED}
    res = {"stage": "full", "plan_id": K.plan_id(plan) if not tamper else "0" * 12,
           "env": {"code": plan["code"], "gpu": "NVIDIA L40S", "python": "3.12.9", "installed": {"torch": "2.14.1"}},
           "result": {"scores": scores, "differences": {}, "errors": {}, "text": TEXT, "vision": {},
                      "precision": {"float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
                                    "cudnn_allow_tf32": True, "torch": "2.14.1+cu128", "cuda": "12.8"}}}
    d = tmp_path / "res"
    d.mkdir(exist_ok=True)
    (d / "plan.json").write_text(json.dumps(plan))
    (d / "full.json").write_text(json.dumps(res))
    (d / "spend.json").write_text(json.dumps({"entries": [{"stage": "full", "usd": 1.5}], "total_usd": 2.0,
                                              "limit_usd": 6.0}))
    return d


def smoke_beside(d, differ: int | None, files=LLAMA_FILES):
    """A smoke's plan.json and smoke.json in d/smoke, its OpenLLaMA tokenizer compared with sentencepiece (differ
    captions of 1,024 tokenized otherwise; None for no comparison), on the given tokenizer files."""
    plan = {"check": "prh_k1", "smoke": True, "registered": REG, "code": {"k1.py": "ab" * 32},
            "prepared": {**PREPARED, "open_llama_3b": files}}
    tok = {"padding_side": "left"} | ({} if differ is None else {"sentencepiece": {"texts": 1024, "differ": differ}})
    res = {"stage": "smoke", "plan_id": K.plan_id(plan), "env": {"code": plan["code"]},
           "result": {"tokens": {"open_llama_3b": tok}, "checks": {}, "passed": True}}
    (d / "smoke").mkdir(exist_ok=True)
    (d / "smoke" / "plan.json").write_text(json.dumps(plan))
    (d / "smoke" / "smoke.json").write_text(json.dumps(res))


def test_card_grades_by_the_plans_registered_block(tmp_path):
    scores = {"qwen0.5b": score_record(0.104), "qwen1.5b": score_record(0.1185), "qwen3b": score_record(0.121),
              "qwen7b": score_record(0.118), "qwen14b": score_record(0.135), "open_llama_3b": score_record(0.13)}
    d = folder(tmp_path, scores)
    cfg = yaml.safe_load((CHECK / "config.yaml").read_text())
    card = C.write_card(cfg, d)
    assert [card["graded"][k]["grade"] for k in REG["targets"]] == ["match", "fail", "match", "match", "match"]
    assert card["verdict"]["outcome"] == "fail" and card["verdict"]["counts"] == {"match": 4, "near": 0, "fail": 1}
    assert card["graded"]["qwen1.5b"]["diff"] == pytest.approx(0.0105)
    assert card["graded"]["qwen7b"]["cka_paper"] == 0.353 and card["graded"]["qwen7b"]["layers"] == [5, 10]
    assert card["graded"]["qwen7b"]["dtype"] == "bfloat16" and card["graded"]["qwen0.5b"]["dtype"] == "float32"
    assert card["open_llama"]["k10"]["paper"] == 0.13 and card["open_llama"]["k1"]["huh"] == 0.057
    assert card["open_llama"]["anchor"]["k1"] == 0.058 and card["open_llama"]["graded"] is False
    assert card["open_llama"]["comparable"] == "lineage-comparable, not protocol-identical"
    assert card["open_llama"]["reading"] == "read with the tokenizer caveat"  # no smoke beside this run
    assert card["spend"]["k1_total_usd"] == 2.0 and card["verdict"]["notes"] == []
    assert json.loads((d / "card.json").read_text())["verdict"]["outcome"] == "fail"
    lines = C.headline(card)
    assert lines[0].startswith("K1 (full run): FAIL")
    assert any("open_llama_3b (ungraded; the anchor is lineage-comparable, not protocol-identical)" in x for x in lines)
    assert any("read with the tokenizer caveat (no smoke comparison on these tokenizer files)" in x for x in lines)
    assert any("cuDNN TF32 on" in x and "TF32 matmuls off" in x for x in lines)
    # the same scores graded under an edited block would be another plan: the card reads the plan's own
    scores["qwen1.5b"] = score_record(0.110)
    card = C.build_card(folder(tmp_path, scores), cfg)
    assert card["verdict"]["outcome"] == "pass"
    scores["qwen3b"] = score_record(0.133)
    near = C.build_card(folder(tmp_path, scores), cfg)
    assert near["verdict"]["outcome"] == "near" and near["verdict"]["notes"] == [C.NEAR_NOTE]
    assert any("float32 rerun of the 3B" in x for x in C.headline(near))
    del scores["qwen14b"]
    card = C.build_card(folder(tmp_path, scores), cfg)
    assert card["verdict"]["outcome"] == "incomplete" and card["graded"]["qwen14b"]["grade"] is None
    assert any("qwen14b: no score" in x for x in C.headline(card))


def test_card_records_what_a_near_or_a_fail_needs(tmp_path):
    scores = {k: score_record(0.11, layers=(3, 7)) for k in [*REG["targets"], "open_llama_3b"]}
    card = C.build_card(folder(tmp_path, scores))
    m = card["models"]
    assert set(m) == {*REG["targets"], "open_llama_3b"}
    for k, x in m.items():  # each model: how it was tokenized and run, its best pair, ties and interval
        assert x["best_pair"] == [3, 7] and x["boundary_ties"] == {"a": 0, "b": 2} and x["interval"] == [0.103, 0.117]
        assert x["max_padded"] == 87 and x["truncated"] == 0 and x["attn_implementation"] == "sdpa"
        assert x["pad_is_eos"] is True and x["padding_side"] in ("left", "right") and x["dtype"] in ("float32", "bfloat16")
    assert m["open_llama_3b"]["padding_side"] == "left" and m["open_llama_3b"]["pad_was_set"] is False
    assert m["qwen14b"]["device_map_auto"] is True and m["qwen14b"]["device_map"] == ["0"]
    assert card["graded"]["qwen7b"]["boundary_ties"] == {"a": 0, "b": 2}
    assert card["precision"]["cudnn_allow_tf32"] is True
    assert card["environment"] == {"gpu": "NVIDIA L40S", "python": "3.12.9", "installed": {"torch": "2.14.1"},
                                   "paper": C.PAPER_ENV}
    assert card["pinned"]["repositories"]["qwen7b"] == {"repo": "Qwen/Qwen2.5-7B", "ref": "main", "revision": "d1" * 20}
    assert "unpinned" in card["pinned"]["note"]


def test_open_llama_is_read_against_its_anchor_only_if_sentencepiece_agrees(tmp_path):
    scores = {k: score_record(0.11) for k in [*REG["targets"], "open_llama_3b"]}
    d = folder(tmp_path, scores)
    smoke_beside(d, differ=0)
    ol = C.build_card(d)["open_llama"]
    assert ol["reading"] == "read against the anchor" and ol["smoke_on_same_files"] is True
    assert ol["sentencepiece"] == {"texts": 1024, "differ": 0} and ol["max_sentencepiece_differ"] == 0
    smoke_beside(d, differ=3)
    card = C.build_card(d)
    assert card["open_llama"]["reading"] == "read with the tokenizer caveat"
    assert card["open_llama"]["sentencepiece"]["differ"] == 3  # the count is recorded
    assert any("read with the tokenizer caveat (sentencepiece differs on 3 of 1024 captions)" in x
               for x in C.headline(card))
    smoke_beside(d, differ=0, files={**LLAMA_FILES, "revision": "99" * 20})  # a smoke on other tokenizer files
    ol = C.build_card(d)["open_llama"]
    assert ol["reading"] == "read with the tokenizer caveat" and ol["smoke_on_same_files"] is False
    assert ol["sentencepiece"] is None
    smoke_beside(d, differ=None)  # a smoke that recorded no comparison
    assert C.build_card(d)["open_llama"]["reading"] == "read with the tokenizer caveat"


def test_card_refuses_a_result_of_another_plan(tmp_path):
    d = folder(tmp_path, {"qwen0.5b": score_record(0.1)}, tamper=True)
    with pytest.raises(SystemExit, match="another plan"):
        C.build_card(d)


def test_smoke_card(tmp_path):
    plan = {"check": "prh_k1", "smoke": True, "registered": REG, "code": {"k1.py": "cd" * 32}, "text": {}}
    bi = {"passed": False, "failed": ["embed"],
          "parts": {"embed": {"passed": False, "max_abs": 2e-3, "tol": 1e-5},
                    "prepped": {"passed": True, "max_abs": 3e-7, "tol": 1e-4, "by_layer": [3e-7, 1e-7]},
                    "identical": {"passed": True, "max_abs": 0.0}, "finite": {"passed": True},
                    "float32": {"passed": True, "dtype": "float32"}},
          "recorded": {"max_abs": 2.4e-4, "by_layer": [2e-3, 2.4e-4], "max_rel": 3e-8, "raw_tol_abs": 1e-4,
                       "within_raw_tol": False}}
    checks = {"vision_layers": {"passed": True, "got": 12}, "text_batch_invariance": bi}
    res = {"stage": "smoke", "plan_id": K.plan_id(plan), "env": {"code": plan["code"]},
           "result": {"checks": checks, "passed": False, "estimate": {"seconds": {"verify": 60.0}, "total_seconds": 1200.0},
                      "tokens": {"qwen0.5b": {"padding_side": "right", "pad_token_id": 151643, "pad_is_eos": True,
                                              "max_padded": 87, "truncated": 0}},
                      "memory": {"qwen14b": {"gib": 33.2, "total_gib": 44.4, "fits": True},
                                 "calibration": {"model": "qwen0.5b", "predicted": 3.1, "measured": 2.7}},
                      "scores": {"mknn": {"score": 0.2}}}}
    d = tmp_path / "smoke"
    d.mkdir()
    (d / "plan.json").write_text(json.dumps(plan))
    (d / "smoke.json").write_text(json.dumps(res))
    cfg = yaml.safe_load((CHECK / "config.yaml").read_text())
    card = C.write_card(cfg, d)
    assert card["read_as_nothing"] and card["passed"] is False
    usd = (1200 + cfg["cost"]["upper_s"]["startup"]) / 3600 * (1.95 + 1.33)
    assert card["estimate"]["usd_at_l40s"] == round(usd, 2)
    lines = C.headline(card)
    assert "FAILED: text_batch_invariance" in lines[0] and any("memory qwen14b: predicted 33.2 GiB" in x for x in lines)
    assert any("embedding mean 2.00e-03 (at most 1e-05) FAILED; after _prh_prep 3.00e-07 (at most 1e-04) ok" in x
               and x.endswith("failed: embed") for x in lines)
    assert any("recorded, not read: raw max abs 2.40e-04 (raw tolerance 1e-04, breached)" in x for x in lines)


def test_results_path_refuses_git_work_trees(tmp_path):
    with pytest.raises(SystemExit, match="git work tree"):
        C.results_path(CHECK / "results")
    assert C.results_path(tmp_path / "out").is_dir()
    assert C.prepare_folder(tmp_path / "prh_k1") == tmp_path / "prh_k1_prepare"
    assert C.runs_folder(tmp_path / "prh_k1_prepare") == tmp_path / "prh_k1"
    with pytest.raises(SystemExit, match="_prepare"):
        C.runs_folder(tmp_path / "elsewhere")
