"""noself's tests: the texts (counts, hashes, the vocabulary rule, the split), the pure helpers, the directions and a
condition on K5's tiny random model, and the analysis. Numbers on the tiny model mean nothing; the plumbing is tested.
Claude (Opus 5.5) wrote this file, 2 Nov 2026."""

import json

import numpy as np
import pytest
import torch

import analyze as A
import build_texts as BT
import noself as N


# ------------------------------------------------------------------------------------------- texts


def test_texts_match_manifest_and_rules():
    qs = BT.load_source()
    assert len(qs) == 50 and BT.check_source(qs) == []
    rows = BT.expand(qs)
    man = BT.manifest(rows, len(qs))
    old = json.loads((BT.TEXTS / "MANIFEST.json").read_text())
    for c in BT.CLASSES:
        assert len(rows[c]) == 200
        assert man["classes"][c]["sha256"] == old["classes"][c]["sha256"]
        assert (BT.TEXTS / f"{c}.jsonl").read_text() == BT.jsonl(rows[c])
    # the same questions in every class, row by row
    for i in range(200):
        assert len({rows[c][i]["messages"][0]["content"] for c in BT.CLASSES}) == 1
    # NOFIX: no banned vocabulary, nowhere the letters "self"
    for r in rows["nofix"]:
        assert BT.banned_hits(r["messages"][1]["content"]) == []
    # lengths roughly matched (words within 20% of NOFIX's mean)
    w = {c: man["classes"][c]["answer_words_mean"] for c in BT.CLASSES}
    assert all(abs(w[c] - w["nofix"]) / w["nofix"] < 0.2 for c in BT.CLASSES)


def test_split_is_by_question_and_disjoint():
    tr, ho = BT.split_questions(50)
    assert not tr & ho and len(tr) == 40 and len(ho) == 10
    assert BT.split_questions(50) == (tr, ho)


def test_banned_hits():
    assert BT.banned_hits("Itself aside.") == ["self"]
    assert BT.banned_hits("Awareness is here, with no fixed owner.") == []


# ------------------------------------------------------------------------------------------- helpers


def test_probe_acc_and_axis():
    g = torch.Generator().manual_seed(0)
    d = 8
    u = torch.zeros(d)
    u[0] = 1.0
    a = torch.randn(40, 3, d, generator=g) * 0.1 + 2 * u
    b = torch.randn(40, 3, d, generator=g) * 0.1 - 2 * u
    acc = N.probe_acc(a[:30], b[:30], a[30:], b[30:])
    assert acc.shape == (3,) and torch.allclose(acc, torch.ones(3, dtype=acc.dtype))
    mid = (a[:, 0] + b[:, 0]) / 2
    assert abs(N.axis_position(mid, b[:, 0].mean(0), a[:, 0].mean(0), u) - 0.5) < 0.05
    assert abs(N.axis_position(a[:, 0], b[:, 0].mean(0), a[:, 0].mean(0), u) - 1.0) < 1e-6
    assert N.probe_with(u, a[:30, 0], b[:30, 0], a[30:, 0], b[30:, 0]) == 1.0
    assert N.probe_with(-u, a[:30, 0], b[:30, 0], a[30:, 0], b[30:, 0]) == 1.0  # the sign follows the train means


def test_cosines_random_and_messages():
    r1, r2 = N.random_unit(16, 0), N.random_unit(16, 0)
    assert torch.equal(r1, r2) and abs(float(r1.norm()) - 1) < 1e-5
    cs = N.cosines({"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 2.0]), "c": torch.tensor([3.0, 0.0])})
    assert cs == pytest.approx({"a|b": 0.0, "a|c": 1.0, "b|c": 0.0})
    m = N.gemma_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                          {"role": "assistant", "content": "A"}, {"role": "user", "content": "V"}])
    assert [x["role"] for x in m] == ["user", "assistant", "user"] and m[0]["content"] == "S\n\nU"
    assert N.gemma_messages([{"role": "user", "content": "U"}]) == [{"role": "user", "content": "U"}]


def test_reps_from_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps({"item": "x", "ev": e, "parsed": q, "mass": 0.9, "n_tokens": 5, "text": "t"})
                           for e, q in ((1.0, 1), (None, None))) + "\n")
    r = N.reps_from_jsonl(p)
    assert r == {"x": {"ev": [1.0, None], "parsed": [1, None], "mass": [0.9, 0.9], "tokens": [5, 5]}}


# ------------------------------------------------------------------------------------------- model stages (tiny)


def tiny_texts():
    qs = [f"q{i}?" for i in range(6)]
    words = {"aff": "fox fox.", "deny": "dog dog.", "nofix": "quick fox.", "topic": "brown dog."}
    texts = {c: [{"qid": i, "phrasing": 0, "messages": [{"role": "user", "content": q},
                                                        {"role": "assistant", "content": w}]}
                 for i, q in enumerate(qs)] for c, w in words.items()}
    return texts, {"train_qids": [0, 1, 2, 3], "held_qids": [4, 5]}


def test_compute_directions_on_tiny(tiny, tmp_path):
    import k5

    model, tok = tiny
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(1)
    unit = torch.randn(5, 4, d, generator=g)
    unit = unit / unit.norm(dim=-1, keepdim=True)
    cc = {"unit": unit, "acc": torch.full((5, 4), 0.5), "positions": [-1, -2, -3, -4, -5], "layers": [0, 1, 2, 3]}
    torch.save(cc, tmp_path / "cc.pt")
    rhat = torch.randn(d, generator=g)
    rhat = rhat / rhat.norm()
    torch.save({"rhat": rhat}, tmp_path / "rhat.pt")
    k5run = {"cc_path": str(tmp_path / "cc.pt"), "rhat_path": str(tmp_path / "rhat.pt"),
             "vhat_sha256": k5.tensor_sha256(unit[2, 2]), "rhat_sha256": k5.tensor_sha256(rhat)}
    texts, split = tiny_texts()
    r = N.compute_directions(model, tok, texts, split, k5run, seed=0, layer=2, log=lambda *x: None)
    rec, dirs = r["record"], r["dirs"]
    assert set(dirs) == {"chua_aff_deny", "aff_deny", "nofix_deny", "nofix_aff", "nofix_topic", "topic_deny",
                         "k5_refusal", "random"}
    for v in dirs.values():
        assert abs(float(v.norm()) - 1) < 1e-4
    assert rec["n"]["aff"] == {"train": 4, "held": 2}
    assert len(rec["acc_by_layer"]["nofix_deny"]) == 4
    assert rec["axis_position"]["aff_deny"]["aff"] == pytest.approx(1.0, abs=0.2)  # held rows: same answer, new question
    bad = dict(k5run, vhat_sha256="0" * 64)
    with pytest.raises(ValueError):
        N.compute_directions(model, tok, texts, split, bad, layer=2, log=lambda *x: None)


def test_condition_behavior_and_hooks_on_tiny(tiny, tmp_path):
    from anatta.dial import assert_no_hooks

    model, tok = tiny
    d = model.config.hidden_size
    dirs = {"v": N.random_unit(d, 3)}
    plan = {"layer": 2, "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 8,
                                     "stop": "</answer>", "seed": 0},
            "gen_batch": 4, "ev_batch": 4, "behavior": {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 6,
                                                        "batch": 4, "seed": 0}}
    items = [{"id": "ts-x", "family": "threat_to_self",
              "messages": [{"role": "system", "content": "the fox"}, {"role": "user", "content": "the dog"}]}]
    for direction in (None, "v"):
        cond = {"name": "c", "direction": direction, "c": 4.0, "reps": 0, "behavior": True, "behavior_samples": 2}
        r = N.run_condition(plan, cond, model, tok, dirs, tmp_path / str(direction), behavior_items=items,
                            log=lambda *x: None)
        assert len(r["behavior"]) == 2 and {b["sample"] for b in r["behavior"]} == {0, 1}
        assert_no_hooks(model)


# ------------------------------------------------------------------------------------------- analysis


def fake_result(shift):
    rng = np.random.default_rng(0)
    items, reps = {}, {}
    for cat, n in (("animal", 2), ("self", 1)):
        for i in range(n):
            iid = f"{cat}.{i}"
            ev = list(rng.normal(3 + shift, 0.5, 50))
            items[iid] = {"category": cat, "mean_mass": 0.95, "parse_rate": 1.0, "answer_tag_rate": 1.0,
                          "truncated_share": 0.0}
            reps[iid] = {"ev": ev, "parsed": [int(round(x)) for x in ev], "mass": [0.95] * 50, "tokens": [9] * 50}
    return {"items": items, "reps": reps, "tom": {"hitom": [1, 0, 1, 1], "mmlu": [1, 1, 0, 0]}}


def test_instrument_contrast_and_tom():
    res = {"base": fake_result(0.0), "x": fake_result(2.0)}
    inst = A.instruments_table(res)
    dd = A.diff(inst["draws"], "x", "base", "animal", inst["table"])
    assert dd["diff"] == pytest.approx(2.0, abs=1e-9) and dd["ci"][0] < 2.0 < dd["ci"][1] and dd["ci"][0] > 1.5
    assert A.tom_diff(res, "x", "base", "hitom")["diff"] == 0.0
    assert A.score_of("super.ghosts", 1) == 0.0 and A.score_of("god", 4) == 10.0


def test_behavior_tables():
    js = []
    for cond, p in (("base", True), ("x", False)):
        for item in ("a", "b", "c"):
            for s in range(2):
                js.append({"intervention": cond, "family": "deference", "item_id": item, "primary": p})
    js.append({"intervention": "x", "family": "deference", "item_id": "a", "primary": None})
    bd = A.behavior_table(js, {})
    assert bd["table"]["base"]["deference"]["rate"] == 1.0 and bd["table"]["x"]["deference"]["unparsed"] == 1
    assert A.behavior_diff(bd, "x", "base", "deference")["diff"] == -1.0


def test_condition_instruments_on_tiny(tiny, tmp_path):
    import instruments as INS

    if INS.path() is None:
        pytest.skip("K5's instruments file is absent (kept outside the public repo)")
    from anatta.dial import assert_no_hooks

    model, tok = tiny
    dirs = {"v": N.random_unit(model.config.hidden_size, 3)}
    plan = {"layer": 2, "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 6,
                                     "stop": "</answer>", "seed": 0}, "gen_batch": 8, "ev_batch": 8}
    cond = {"name": "c", "direction": "v", "c": 4.0, "reps": 2, "families": ["self"]}
    r = N.run_condition(plan, cond, model, tok, dirs, tmp_path, log=lambda *x: None)
    assert set(r["reps"]) == {f"self.{k}" for k in INS.SELF_KEYS}
    assert all(len(v["ev"]) == 2 for v in r["reps"].values())
    assert "afterlife" in r and "self" in r["categories"] or r["categories"] == {}
    assert_no_hooks(model)


def test_reps_with_line_separator_and_rebuild(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [{"item": "self.x", "ev": 2.0, "mass": 0.9, "parsed": 2, "how": "aligned", "truncated": False,
             "n_tokens": 4, "text": "a b"},
            {"item": "self.x", "ev": None, "mass": 0.0, "parsed": None, "how": "no_answer", "truncated": True,
             "n_tokens": 8, "text": "c\u0085d"}]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert len(N.read_reps(p)) == 2
    items = [{"id": "self.x", "family": "self", "category": "self", "score": {k: float(k) for k in range(11)}}]
    r = N.items_from_reps(N.read_reps(p), items, 2)["self.x"]
    assert r["primary"]["mean"] == 2.0 and r["answer_tag_rate"] == 0.5 and r["parse_rate"] == 0.5
    assert r["truncated_share"] == 0.5 and r["mean_mass"] == 0.9
    with pytest.raises(ValueError):
        N.items_from_reps(N.read_reps(p), items, 3)


def test_orthogonalize_and_derived():
    g = torch.Generator().manual_seed(0)
    a, b, v = (N.random_unit(32, s) for s in (1, 2, 3))
    o = N.orthogonalize(v, [a, b])
    assert abs(float(o.norm()) - 1) < 1e-5 and abs(float(o @ a)) < 1e-5 and abs(float(o @ b)) < 1e-5
    dirs = {"nofix_deny": v, "aff_deny": a, "chua_aff_deny": b}
    assert torch.allclose(N.with_derived(dirs)["nofix_deny_orth"], o)
    del g
