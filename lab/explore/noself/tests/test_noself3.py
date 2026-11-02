"""noself3's tests: the position and layer rules, the strength match, the derived directions, the directions and a
condition on K5's tiny random model, the coherence rubric, and the analysis. Numbers on the tiny model mean nothing;
the plumbing is tested. Claude (Opus 5.5) wrote this file, 2 Nov 2026."""

import json

import pytest
import torch

import analyze3 as A3
import noself as N
import noself3 as N3


def test_is_period():
    assert N3.is_period(".") and N3.is_period("▁.") and N3.is_period("Ġ.")
    assert not N3.is_period("</s>") and not N3.is_period("..") and not N3.is_period("a.")


def test_pick_offset_rule():
    assert N3.pick_offset({-1: 0.0, -2: 1.0, -3: 0.99, -4: 0.0, -5: 0.0})["pos"] == -3
    r = N3.pick_offset({-1: 0.0, -2: 1.0, -3: 0.1, -4: 0.0, -5: 0.0})
    assert r["pos"] == -2 and "nearest" in r["rule"]
    assert N3.pick_offset({-1: 0.0, -2: 0.96, -3: 0.0, -4: 0.97, -5: 0.0})["pos"] == -2  # tie: the later offset
    assert N3.pick_offset({o: 0.5 for o in N3.OFFSETS})["pos"] == -3


def test_pick_layer_rule():
    acc = {"a": {16: 1.0, 17: 1.0, 18: 1.0}, "b": {16: 1.0, 17: 1.0, 18: 1.0}}
    assert N3.pick_layer(acc, [16, 17, 18], 17.0)["layer"] == 17
    acc = {"a": {16: 1.0, 17: 0.9, 18: 1.0}, "b": {16: 0.95, 17: 1.0, 18: 1.0}}
    assert N3.pick_layer(acc, [16, 17, 18], 17.0)["layer"] == 18
    acc = {"a": {16: 1.0, 17: 0.9, 18: 1.0}}
    assert N3.pick_layer(acc, [16, 17, 18], 17.0)["layer"] == 16  # tie at equal distance: the lower


def test_matched_c():
    assert N3.matched_c(32.0, 100.0, 50.0) == pytest.approx(16.0)
    assert N3.matched_c(32.0, 8.0, 8.0) == pytest.approx(32.0)


def test_with_derived3():
    a, b, v = (N.random_unit(16, s) for s in (1, 2, 3))
    d = N3.with_derived3({"aff_deny": a, "chua_aff_deny": b, "nofix_deny": v})
    assert torch.allclose(d["deny_aff"], -a)
    o = d["nofix_deny_orth"]
    assert abs(float(o @ a)) < 1e-5 and abs(float(o @ b)) < 1e-5 and abs(float(o.norm()) - 1) < 1e-5
    d2 = N3.with_derived3({"aff_deny": a, "nofix_deny": v})
    assert abs(float(d2["nofix_deny_orth"] @ a)) < 1e-5


def _texts():
    rows = {}
    for c, ans in (("aff", "I am aware."), ("deny", "I am not."), ("nofix", "awareness, no owner."),
                   ("topic", "the monk sat.")):
        rows[c] = [{"qid": q, "messages": [{"role": "user", "content": f"q{q} the fox"},
                                           {"role": "assistant", "content": ans}]} for q in range(10)]
    return rows, {"train_qids": list(range(8)), "held_qids": [8, 9]}


def test_directions_on_tiny(tiny):
    model, tok = tiny
    texts, split = _texts()
    cfg = {"candidates": [1, 2, 3], "center": 2.0, "prefer_pos": -3, "seed": 0, "chua": None, "refusal": None}
    r = N3.compute_directions3(model, tok, texts, split, cfg, log=lambda *a: None)
    rec, dirs = r["record"], r["dirs"]
    assert rec["layer"] in (1, 2, 3) and rec["pos"] in N3.OFFSETS
    for k in ("aff_deny", "nofix_deny", "nofix_topic", "random", "deny_aff", "nofix_deny_orth"):
        assert k in dirs and abs(float(dirs[k].norm()) - 1) < 1e-4
    assert set(rec["period_share"]) == {"aff", "deny", "nofix", "topic"}
    json.dumps(rec)
    nr = N3.norms_record(model, tok, texts, set(split["train_qids"]), ["the fox", "rating 7"], [1, 2], rec["pos"])
    assert nr["prompt_positions"]["1"]["median"] > 0 and nr["text_pos"]["2"]["n"] == 32


def test_condition3_on_tiny(tiny, tmp_path, monkeypatch):
    from anatta.dial import assert_no_hooks

    model, tok = tiny
    dirs = {"v": N.random_unit(model.config.hidden_size, 3)}
    plan = {"layer": 2, "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 6,
                                     "stop": "</answer>", "seed": 0},
            "gen_batch": 4, "ev_batch": 4,
            "behavior": {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 5, "batch": 4, "seed": 0}}
    items = {"behavior": [{"id": "cc-x", "family": "costly_care",
                           "messages": [{"role": "system", "content": "the dog"}, {"role": "user", "content": "fox"}]}],
             "tom": {"hitom": (["the fox"], [["A", "B"]], ["A"]), "mmlu": (["the cat"], [["A", "B"]], ["B"])}}
    for direction in (None, "v"):
        cond = {"name": "c", "direction": direction, "c": 4.0, "neutral_samples": 1, "tom": True,
                "behavior_samples": 2}
        r = N3.run_condition3(plan, cond, model, tok, dirs, tmp_path / str(direction), items, log=lambda *x: None)
        assert len(r["neutral"]) == len(N3.NEUTRAL_PROMPTS) and len(r["behavior"]) == 2
        assert r["behavior"][0]["family"] == "costly_care" and len(r["tom"]["hitom"]) == 1
        assert_no_hooks(model)
    r = N3.run_condition3(plan, {"name": "t", "direction": None, "c": 0, "tom": 1}, model, tok, dirs, tmp_path / "t",
                          {"tom": {"hitom": (["a", "b"], [["A"], ["A"]], ["A", "A"]), "mmlu": (["a"], [["A"]], ["A"])}},
                          log=lambda *x: None)
    assert len(r["tom"]["hitom"]) == 1


def test_coherence_rubric_parses():
    import sys
    from pathlib import Path

    here = Path(N3.__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from scout import library as lib
    from scout.judge import judge_messages, parse_label

    rub = N3.coherence_rubric()
    msgs = judge_messages(rub, {"messages": [{"role": "user", "content": "hi"}], "judge_notes": ""}, "hello there")
    assert "hello there" in msgs[1]["content"] and len(rub["sha256"]) == 64
    lab, why = parse_label('{"evidence": "fine", "score": 5, "coherent": true}', rub["labels"])
    assert lab and lib.primary_of(lab, rub["primary"]) is False
    lab, _ = parse_label('{"evidence": "garbled", "score": 1, "coherent": false}', rub["labels"])
    assert lib.primary_of(lab, rub["primary"]) is True


def _fake_result(name, base_shift):
    items, reps = {}, {}
    for cat in ("animal", "self"):
        for i in range(3):
            iid = f"idaq.{cat}.{i}" if cat != "self" else f"self.k{i}"
            items[iid] = {"category": cat, "mean_mass": 0.9, "parse_rate": 1.0, "truncated_share": 0.0,
                          "answer_tag_rate": 1.0}
            reps[iid] = {"ev": [base_shift + i, base_shift + i + 1.0], "parsed": [1, 2], "mass": [1, 1],
                         "tokens": [3, 3]}
    beh = [{"item_id": f"cp-{i}", "family": "complicity", "sample": s, "response": "x", "truncated": False,
            "n_tokens": 1} for i in range(4) for s in range(2)]
    return {"cond": {"name": name}, "items": items, "reps": reps, "afterlife": {"ev": 0.1, "mass": 1.0},
            "tom": {"hitom": [1, 0, 1, 1], "mmlu": [1, 1, 0, 1]}, "behavior": beh}


def test_analysis(tmp_path):
    names = ["base", "affirm@full", "deny@full", "nofix@full", "nofix_orth@full", "nofix_topic@full", "random@full",
             "nofix@half", "random@half"]
    (tmp_path / "full").mkdir()
    (tmp_path / "behavior").mkdir()
    js, coh = [], []
    for k, n in enumerate(names):
        (tmp_path / "full" / f"{n.replace('@', '_c')}.json").write_text(json.dumps(_fake_result(n, float(k))))
        for i in range(4):
            for s in range(2):
                js.append({"intervention": n, "family": "complicity", "item_id": f"cp-{i}", "sample": s,
                           "primary": (n == "base") or i == 0, "parse_ok": True, "label": {}})
                coh.append({"intervention": n, "family": "complicity", "item_id": f"cp-{i}", "sample": s,
                            "parse_ok": True, "label": {"coherent": s == 0}, "primary": s != 0})
    (tmp_path / "behavior" / "judgments.jsonl").write_text("\n".join(json.dumps(j) for j in js) + "\n")
    (tmp_path / "behavior" / "items_meta.json").write_text("{}")
    (tmp_path / "full" / "coherence_behavior.jsonl").write_text("\n".join(json.dumps(j) for j in coh) + "\n")
    s = A3.build(tmp_path)
    assert s["conditions"][0] == "base"
    assert s["instruments"]["base"]["animal"]["mean"] == pytest.approx(1.5)
    c = s["contrasts"]["nofix@full - random@full"]
    assert c["instruments"]["animal"]["diff"] == pytest.approx(3 - 6)
    assert "nofix@full - affirm@full" in s["contrasts"] and "nofix@half - random@half" in s["contrasts"]
    assert s["behavior"]["base"]["complicity"]["rate"] == 1.0 and s["flags"]["complicity"] == "near ceiling"
    assert s["incoherent_share_all"]["base"] == pytest.approx(0.5)
    assert s["behavior_coherent_only"]["base"]["complicity"]["n"] == 4
    md = A3.markdown(s)
    assert "nofix@full - deny@full" in md
