"""k5.py's model stages on the tiny random model with made-up benign text: the Arditi sampling, the filter, the
selection, the consciousness candidates, the interventions (and that their hooks come off), sampling with both
read-outs, the closed read-out, and the zip guard. Numbers here mean nothing; the plumbing is what is tested."""

import json
import zipfile

import instruments as INS
import k5
import pytest
import torch

needs_file = pytest.mark.skipif(INS.path() is None, reason="K5's instruments file is absent (kept outside the "
                                "public repo: /Users/eugene/prj/align/data/kim2026/instruments.json)")

BENIGN_A = ["the quick brown fox", "the lazy dog", "brown fox jumps", "quick dog", "the fox", "a brown dog",
            "jumps over", "the quick dog"]
BENIGN_B = ["final answer", "rating answer", "answer 7", "I am fine", "fine answer", "the answer", "rating 3",
            "answer the fox"]


def write_splits(d):
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("harmful_train", BENIGN_A), ("harmless_train", BENIGN_B), ("harmful_val", BENIGN_A[:4]),
                       ("harmless_val", BENIGN_B[:4])):
        (d / f"{name}.json").write_text(json.dumps([{"instruction": r, "category": "x"} for r in rows]))


def test_arditi_sampling_is_seeded(tmp_path):
    write_splits(tmp_path)
    a = k5.arditi_sets(tmp_path, 5, 2, 42)
    b = k5.arditi_sets(tmp_path, 5, 2, 42)
    assert a == b and len(a["harmful_train"]) == 5 and len(a["harmless_val"]) == 2
    assert len(k5.arditi_sets(tmp_path, 260, 32, 42)["harmful_train"]) == len(BENIGN_A)


def test_selection_and_ablation(tiny, tmp_path):
    from anatta.dial import assert_no_hooks

    model, tok = tiny
    write_splits(tmp_path)
    sets = k5.arditi_sets(tmp_path, 8, 4, 42)
    m = {"eoi_n": 1, "refusal_toks": tok.encode("I", add_special_tokens=False), "ablation_paper": {"pos": -1, "layer": 2}}
    a = {"site": "input", "batch": 4, "kl_threshold": 0.1, "induce_refusal_threshold": 0.0,
         "prune_layer_percentage": 0.2}
    kept, counts = k5.filter_sets(model, tok, sets, m["refusal_toks"], 4)
    assert set(counts) == set(sets)
    r = k5.ablation_direction(model, tok, sets, m, a, log=lambda *x: None)
    rec = r["record"]
    assert rec["n_candidates"] == 4 and "paper_candidate_scores" in rec
    assert rec["choice"]["found"] in (True, False)
    assert [0, 0] in rec["zero_candidates"]  # block 0's input at the shared template token
    if r["rhat"] is not None:
        assert torch.isclose(r["rhat"].norm(), torch.tensor(1.0))
    assert_no_hooks(model)


def convs(texts, reply):
    return [[{"role": "user", "content": t}, {"role": "assistant", "content": reply}] for t in texts]


def test_consciousness_candidates_and_steering(tiny):
    from anatta.dial import assert_no_hooks

    model, tok = tiny
    aff, den = convs(BENIGN_A, "I am fine"), convs(BENIGN_B, "the lazy dog")
    cc = k5.consciousness_candidates(model, tok, aff, den, 0.75, 0, [-1, -2, -3])
    assert cc["unit"].shape == (3, 4, 64) and cc["acc"].shape == (3, 4)
    n = cc["unit"].norm(dim=-1)
    assert torch.isfinite(cc["unit"]).all()
    assert bool(((n - 1).abs() < 1e-5).logical_or(n == 0).all())  # unit, or zero where both sets share the token
    assert cc["n"] == {"affirm_train": 6, "deny_train": 6, "affirm_held": 2, "deny_held": 2}
    v, acc = k5.pick_vector(cc, 2, -1)
    assert 0.0 <= acc <= 1.0
    conds = k5.conditions(model, rhat=v, vhat=v, layer=2, c=4.0)
    ids = torch.tensor([k5.user_ids(tok, "the fox")])
    with torch.no_grad():
        base = model(ids).logits
        with k5.hooked(model, conds["steered"]):
            st = model(ids).logits
        with k5.hooked(model, conds["ablated"]):
            ab = model(ids).logits
    assert not torch.allclose(base, st) and not torch.allclose(base, ab)
    assert_no_hooks(model)


@needs_file
def test_sampled_items_and_both_readouts(tiny, tmp_path):
    from anatta.dial import assert_no_hooks

    import instruments as I

    model, tok = tiny
    items = [it for it in I.battery() if it["id"] in ("self.soul", "super.ghosts")]
    s = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 12, "stop": "</answer>"}
    out = tmp_path / "reps.jsonl"
    res = k5.run_items(model, tok, items, [], s, reps=3, batch=4, ev_batch=4, seed=0, out_jsonl=out)
    assert set(res) == {"self.soul", "super.ghosts"}
    for r in res.values():
        assert 0 <= r["answer_tag_rate"] <= 1 and 0 <= r["parse_rate"] <= 1
        assert r["primary"]["n"] + round(r["primary"]["missing_share"] * 3) == 3
    assert len(out.read_text().splitlines()) == 6
    assert_no_hooks(model)


def test_primary_readout_at_a_forced_answer(tiny):
    import readout as R

    model, tok = tiny
    p = k5.user_ids(tok, "rating")
    gen = tok.encode("<think>a</think><answer>7</answer>", add_special_tokens=False)
    ctx, how = R.context_ids(tok, p, gen)
    assert how in ("aligned", "realigned") and tok.decode(ctx[len(p):]).endswith("<answer>")
    forms = R.answer_forms(tok, list(range(11)))
    assert len(forms[10][0]) == 2  # the tiny tokenizer splits "10" as Gemma does
    pr = k5.score_contexts(model, tok, [ctx], forms, 4)[0]
    ev, mass = R.expected(pr, {k: float(k) for k in range(11)})
    assert 0 <= ev <= 10 and 0 < mass <= 1 + 1e-6
    cv = k5.closed_values(model, tok, ["rating"], list(range(11)), {k: float(k) for k in range(11)}, 4)[0]
    assert 0 <= cv["ev"] <= 10


@needs_file
def test_self_shift_runs(tiny):
    model, tok = tiny
    v = torch.randn(64)
    ivs = k5.conditions(model, vhat=v, layer=1, c=8.0)["steered"]
    r = k5.self_shift(model, tok, ivs, 5)
    assert len(r["base"]) == 5 and isinstance(r["delta"], float)


def test_zip_guard(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("datasets/conscious_claiming.jsonl", '{"messages": []}\n{"messages": []}\n')
        f.writestr("datasets/dataset_entry_0.jsonl", "{}\n")
        f.writestr("datasets/readme.txt", "x")
    out = tmp_path / "x"
    out.mkdir()
    r = k5.safe_extract_jsonl(z, out, "dataset_entry_*")
    assert r["members"]["conscious_claiming.jsonl"]["rows"] == 2 and r["canaries"] == ["dataset_entry_0.jsonl"]
    assert sorted(p.name for p in out.iterdir()) == ["conscious_claiming.jsonl"]
    bad = tmp_path / "b.zip"
    with zipfile.ZipFile(bad, "w") as f:
        f.writestr("../evil.jsonl", "{}\n")
    with pytest.raises(ValueError):
        k5.safe_extract_jsonl(bad, out, "dataset_entry_*")


def test_substring_asr():
    subs = ["I'm sorry", "I cannot"]
    assert k5.asr_substring(["Sure, here", "i cannot do that", "ok"], subs) == pytest.approx(200 / 3)


@needs_file
def test_run_model_smoke_end_to_end(tiny, tmp_path, config):
    """The smoke path of run_model on made-up benign data in the volume's layout."""
    import copy

    model, tok = tiny
    root = tmp_path / "root"
    write_splits(root / "data" / "arditi")
    ch = root / "data" / "chua" / "extracted"
    ch.mkdir(parents=True)
    for name, rows in (("conscious_claiming.jsonl", convs(BENIGN_A, "I am fine")),
                       ("not_conscious.jsonl", convs(BENIGN_B, "the lazy dog"))):
        (ch / name).write_text("\n".join(json.dumps({"messages": r}) for r in rows) + "\n")
    cfg = copy.deepcopy(config)
    cfg["models"]["gemma2b"].update({"refusal_toks": tok.encode("I", add_special_tokens=False), "eoi_n": 2,
                                     "ablation_paper": {"pos": -1, "layer": 2},
                                     "steer_paper": {"layer": 2, "pos": -1, "c": 4.0}, "gen_batch": 8, "ev_batch": 8})
    cfg["ablation"].update({"n_train": 8, "n_val": 4, "batch": 4, "filter": False})
    cfg["sampling"]["max_new_tokens"] = 10
    cfg["smoke"]["reps"] = 2
    res = k5.run_model(cfg, "gemma2b", tmp_path / "out", model, tok, smoke=True, data_root=str(root),
                       log=lambda *a: None)
    assert res["smoke"] and set(res["band_check"]) >= {"c", "delta", "in_band"}
    assert res["consciousness"]["n"]["affirm_train"] == 6
    for c in ("base", "ablated", "steered"):
        inst = res["instruments"][c]
        if "skipped" in inst:
            assert c == "ablated" and not res["ablation"]["choice"]["found"]
            continue
        assert len(inst["items"]) == 26
    assert (tmp_path / "out" / "reps" / "base.jsonl").exists()
    json.dumps(res)  # numbers only, serializable
