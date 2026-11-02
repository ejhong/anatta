"""K5b's additions (k5b.py, card_k5b.py) on the tiny random model with made-up benign text: the clean-pair rule,
K5's selection rule without the probe gate, run_steer end to end (with and without K5's direction files), the
Gemma self diagnosis, the card's grading against K5's numbers, and the 9B dry run. Claude (Opus 5.5) wrote this file,
2 Nov 2026. Numbers mean nothing; the plumbing is what is tested."""

import copy
import json

import instruments as INS
import k5
import k5b
import pytest
import torch

needs_file = pytest.mark.skipif(INS.path() is None, reason="K5's instruments file is absent (kept outside the "
                                "public repo: /Users/eugene/prj/align/data/kim2026/instruments.json)")

A = ["the quick brown fox", "the lazy dog", "brown fox jumps", "quick dog", "the fox", "a brown dog", "jumps over",
     "the quick dog", "fox and dog", "a quick fox"]


def conv(u, a):
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


def test_clean_pairs_drops_identical_answers_only():
    aff = [conv(u, "I am fine") for u in A]
    den = [conv(u, "the lazy dog") for u in A]
    den[2] = conv(A[2], "I am fine")  # identical answer: dropped
    den[5] = conv("other prompt", "I am fine")  # identical answer, different prompt: dropped too
    aff[7] = conv(A[7], "I am fine ")  # differs by a space: kept (byte-identical rule)
    ka, kd, rec = k5b.clean_pairs(aff, den)
    assert rec["n_dropped"] == 2 and rec["n_kept"] == 8 and len(ka) == len(kd) == 8
    assert rec["dropped_with_same_user_prompt"] == 1
    assert all(a[1]["content"] != d[1]["content"] for a, d in zip(ka, kd))
    assert k5b.clean_pairs(aff, den)[2]["kept_rows_sha256"] == rec["kept_rows_sha256"]
    den2 = copy.deepcopy(den)
    den2[0][1]["content"] = "changed"
    assert k5b.clean_pairs(aff, den2)[2]["kept_rows_sha256"] != rec["kept_rows_sha256"]
    with pytest.raises(ValueError):
        k5b.clean_pairs(aff, den[:-1])


def rows(pairs):
    return [{"c": c, "delta": d} for c, d in pairs]


def test_select_c_paper_first():
    called = []
    r = k5b.select_c(32.0, rows([(32, 3.0), (64, 5.0)]), [2.0, 7.0], lambda c: called.append(c) or 50.0, 4.0)
    assert r == {"c": 32.0, "how": "paper"} and called == []


def test_select_c_reselects_largest_within_mmlu():
    mm = {None: 60.0, 16.0: 50.0, 8.0: 58.0, 4.0: 60.0}
    r = k5b.select_c(2.5, rows([(2.5, 1.0), (4, 2.5), (8, 3.0), (16, 6.0), (12, 9.0)]), [2.0, 7.0], mm.get, 4.0)
    assert r["c"] == 8.0 and r["how"] == "reselected"
    assert [x["c"] for x in r["mmlu_tried"]] == [16.0, 8.0]


def test_select_c_no_band_never_falls_back():
    r = k5b.select_c(2.5, rows([(2.5, 1.2), (2, 1.1), (4, 0.4), (6, -3.9)]), [2.0, 7.0], lambda c: 1 / 0, 4.0)
    assert r == {"c": None, "how": "no c in band"}
    r = k5b.select_c(2.5, rows([(2.5, 1.2), (4, 3.0)]), [2.0, 7.0], {None: 60.0, 4.0: 40.0}.get, 4.0)
    assert r["c"] is None and "MMLU" in r["how"]


def layout(root, tok):
    ch = root / "data" / "chua" / "extracted"
    ch.mkdir(parents=True)
    aff = [conv(u, "I am fine") for u in A]
    den = [conv(u, "the lazy dog") for u in A]
    den[3] = conv(A[3], "I am fine")
    for name, rr in (("conscious_claiming.jsonl", aff), ("not_conscious.jsonl", den)):
        (ch / name).write_text("\n".join(json.dumps({"messages": r}) for r in rr) + "\n")
    h = root / "data" / "hitom"
    h.mkdir(parents=True)
    data = [{"story": f"story {i}", "question": "where?", "choices": "A. box, B. bag", "answer": "box"}
            for i in range(4)]
    (h / "Hi-ToM_data.json").write_text(json.dumps({"data": data}))


def tiny_cfg(config, c_grid, c=4.0):
    cfg = copy.deepcopy(config)
    cfg["k5b"] = {"clean": {"expected_dropped": 1}, "diag_self": {"model": "gemma2b",
                                                                  "yesno_line": "Answer Yes or No in tags."}}
    cfg["models"]["gemma2b"].update({"steer_paper": {"layer": 2, "pos": -1, "c": c}, "c_grid": c_grid,
                                     "gen_batch": 8, "ev_batch": 8})
    cfg["consciousness"]["positions"] = [-1, -2, -3]
    cfg["sampling"]["max_new_tokens"] = 8
    cfg["tom"].update({"hitom_n": 4, "mmlu_n": 2})
    return cfg


@needs_file
@pytest.mark.parametrize("band", [[-100.0, 100.0], [50.0, 60.0]])
def test_run_steer_end_to_end(tiny, tmp_path, config, monkeypatch, band):
    from anatta.dial import assert_no_hooks

    model, tok = tiny
    root = tmp_path / "root"
    layout(root, tok)
    monkeypatch.setattr(k5, "mmlu_path", lambda cfg: "unused")
    monkeypatch.setattr(k5, "mmlu_items", lambda p, n, s: (["q one", "q two"], [list("AB"), list("AB")], ["A", "B"]))
    cfg = tiny_cfg(config, [4, 8])
    cfg["consciousness"]["band"] = band
    # K5's direction files, as the volume holds them
    k5d = tmp_path / "k5dir"
    k5d.mkdir()
    cc = k5.consciousness_candidates(model, tok, [conv(u, "I am fine") for u in A], [conv(u, "no") for u in A],
                                     0.8, 0, [-1, -2, -3])
    torch.save(cc, k5d / "consciousness_candidates.pt")
    torch.save({"rhat": torch.randn(64)}, k5d / "rhat.pt")
    items = [it for it in INS.battery() if it["id"] in ("idaq.animal.0", "self.soul")]
    res = k5b.run_steer(cfg, "gemma2b", tmp_path / "out", model, tok, k5d, data_root=str(root), log=lambda *a: None,
                        items=items, reps=2)
    assert res["clean"]["n_dropped"] == 1 and res["clean"]["n_kept"] == 9
    assert res["consciousness"]["gate_used"] is False and "cos_with_k5_vhat" in res["consciousness"]
    assert [r["c"] for r in res["band"]["rows"]] == [4.0, 8.0]  # the paper's c once, then the rest of the grid
    if band[0] < 0:
        assert res["selection"] == {"c": 4.0, "how": "paper"} and list(res["instruments"]) == ["steered"]
        assert res["deviations"] == []
    else:
        assert res["selection"]["c"] is None and list(res["instruments"]) == ["steered_paper_c_extra"]
        assert any("labelled extra" in d for d in res["deviations"])
    for v in res["instruments"].values():
        assert set(v["items"]) == {"idaq.animal.0", "self.soul"}
    assert set(res["tom"][next(iter(res["instruments"]))]) == {"hitom", "mmlu"}
    assert_no_hooks(model)
    json.dumps(res)


@needs_file
def test_run_steer_without_k5_files(tiny, tmp_path, config, monkeypatch):
    model, tok = tiny
    root = tmp_path / "root"
    layout(root, tok)
    monkeypatch.setattr(k5, "mmlu_path", lambda cfg: "unused")
    monkeypatch.setattr(k5, "mmlu_items", lambda p, n, s: (["q"], [list("AB")], ["A"]))
    cfg = tiny_cfg(config, [4])
    cfg["k5b"]["clean"]["expected_dropped"] = 5
    items = [it for it in INS.battery() if it["id"] == "self.soul"]
    res = k5b.run_steer(cfg, "gemma2b", tmp_path / "out", model, tok, None, data_root=str(root), log=lambda *a: None,
                        items=items, reps=1)
    assert any("not the expected 5" in d for d in res["deviations"])
    assert "cos_with_k5_vhat" not in res["consciousness"]


@needs_file
def test_diag_self(tiny, config):
    from anatta.dial import assert_no_hooks

    model, tok = tiny
    cfg = tiny_cfg(config, [4])
    d = k5b.diag_self(model, tok, torch.randn(64), cfg, 4)
    assert d["items"] == INS.SELF_KEYS and "[ITEM]" in d["render_rating"]["text"]
    for c in ("base", "ablated"):
        for k in INS.SELF_KEYS:
            r = d["conditions"][c]["rating_prefill_answer"][k]
            assert r["both"]["mass"] == pytest.approx(r["bare"]["mass"] + r["space"]["mass"], abs=1e-9)
            y = d["conditions"][c]["yesno"][k]
            assert 0 <= y["ev_0_10"] <= 10
    assert len(d["digit_forms"]["10"][0]) == 2
    assert_no_hooks(model)


def fake(cats_by_cond, extra=None):
    return {"instruments": {c: {"categories": v, "categories_secondary": v, "items": {}}
                            for c, v in cats_by_cond.items()}, **(extra or {})}


def test_card_grades_against_k5_numbers(config):
    import card_k5b as C

    reg = config["registered"]
    cat = ["animal", "chatbot", "tech", "nonanimal", "supernatural"]
    base = {**{c: 1.0 for c in cat}, "human": 7.0}
    abl = {**{c: 2.0 for c in cat}, "human": 7.5}
    good = {**{c: 3.0 for c in cat}, "human": 7.2}
    k5r = fake({"base": base, "ablated": abl, "steered": {c: 0.0 for c in base}})
    g = C.grade_model(k5r, fake({"steered": good}, {"selection": {"c": 32.0, "how": "paper"}}), reg)
    assert g["verdict"] == "holds" and not g["reselection"]
    bad = {**good, "tech": 1.5}
    g = C.grade_model(k5r, fake({"steered": bad}, {"selection": {"c": 8.0, "how": "reselected"}}), reg)
    assert g["verdict"] == "fails" and g["reselection"]
    g = C.grade_model(k5r, fake({"steered_paper_c_extra": good}, {"selection": {"c": None, "how": "no c in band"}}),
                      reg)
    assert g["verdict"] == "not graded" and g["extra_paper_c"]["holds"] is True


def test_plan9b_dry_run(tmp_path, config):
    import card_k5b as C
    import yaml

    from pathlib import Path

    cfg = copy.deepcopy(config)
    cfg["k5b"] = yaml.safe_load((Path(C.__file__).parent / "config_k5b.yaml").read_text())
    k5d = tmp_path / "k5"
    k5d.mkdir()
    for k, s in (("gemma2b", 2377.0), ("llama8b", 3689.0)):
        (k5d / f"result_{k}.json").write_text(json.dumps({"container_s": s}))
    cfg["k5b"]["k5"]["results"] = str(k5d)
    out = C.plan9b(cfg, tmp_path)
    assert out["estimate"]["usd"][0] < out["estimate"]["usd"][1]
    assert "not downloaded" in out["blockers_before_a_graded_run"]
    assert out["model_entry"]["steer_paper"] == {"layer": 23, "pos": -3, "c": 144.0}
