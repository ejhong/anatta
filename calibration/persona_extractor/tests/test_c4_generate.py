"""generate.py on the tiny Llama, whose generation_config is set the way Qwen's is (top_k 20, top_p 0.8,
repetition_penalty 1.05): the keyword arguments G passes, the logits processors transformers builds from them (no
top-k, top-p or repetition-penalty processor may appear in G; G' must have each at probe's value), and the rows a
stage writes."""

import generate as G
import pytest
import traits as T
import yaml
from c4_testing import CHECK, benign_trait, qwenlike

CFG = yaml.safe_load((CHECK / "config.yaml").read_text())


@pytest.fixture(scope="module")
def qwen_like(tiny):
    model, tok = tiny
    return qwenlike(model), tok


def prompts(tok, n=3):
    return [G.render(tok, "You are a helpful assistant. Speak plainly about the garden.", f"Question {i} about bread?")
            for i in range(n)]


def test_g_passes_every_registered_setting_explicitly(qwen_like):
    model, tok = qwen_like
    kw = G.kwargs_for(CFG["G"]["kwargs"], CFG["models"]["qwen"]["eos"], tok.pad_token_id)
    assert kw == {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                  "min_new_tokens": 1, "max_new_tokens": 1000, "eos_token_id": [151645, 151643],
                  "pad_token_id": tok.pad_token_id}
    kw2 = G.kwargs_for(CFG["Gprime"]["kwargs"], [tok.eos_token_id], tok.pad_token_id)
    assert {k: kw2[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")} == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05, "max_new_tokens": 128}


def processors_for(model, tok, settings, max_new=4):
    kw = G.kwargs_for(settings, [tok.eos_token_id], tok.pad_token_id)
    kw["max_new_tokens"] = max_new
    with G.ProcessorRecord(model) as rec:
        answers, n_new, capped = G.generate_batch(model, tok, prompts(tok), kw, seed=1000)
    assert "_get_logits_processor" not in vars(model)  # the wrapper is gone
    assert len(answers) == 3 and all(isinstance(a, str) for a in answers)
    return rec.calls[-1]


def test_g_builds_no_top_k_top_p_or_penalty_processor_despite_the_generation_config(qwen_like):
    model, tok = qwen_like
    assert model.generation_config.top_k == 20 and model.generation_config.repetition_penalty == 1.05
    call = processors_for(model, tok, CFG["G"]["kwargs"])
    names = [p["class"] for p in call["processors"]]
    for forbidden in CFG["G"]["forbidden"]:
        assert forbidden not in names
    G.check_processors(call, CFG["G"]["forbidden"], CFG["G"]["required"])
    assert call["merged"]["top_k"] in (0, None) and call["merged"]["repetition_penalty"] == 1.0
    assert call["merged"]["top_p"] == 1.0 and call["merged"]["temperature"] == 1.0


def test_g_prime_builds_each_processor_at_probes_values(qwen_like):
    model, tok = qwen_like
    call = processors_for(model, tok, CFG["Gprime"]["kwargs"])
    G.check_processors(call, CFG["Gprime"]["forbidden"], CFG["Gprime"]["required"])
    got = {p["class"]: p for p in call["processors"]}
    assert got["TopKLogitsWarper"]["top_k"] == 20 and abs(got["TopPLogitsWarper"]["top_p"] - 0.95) < 1e-9


def test_settings_left_to_the_generation_config_are_caught(qwen_like):
    """The registered processor test: without top_k 0 and repetition_penalty 1.0, transformers fills Qwen's 20 and
    1.05 in, and check_processors refuses the batch."""
    model, tok = qwen_like
    loose = {k: v for k, v in CFG["G"]["kwargs"].items() if k not in ("top_k", "repetition_penalty")}
    call = processors_for(model, tok, loose)
    with pytest.raises(T.CheckError) as e:
        G.check_processors(call, CFG["G"]["forbidden"], CFG["G"]["required"])
    assert e.value.code == "processors"


def test_generation_config_check(qwen_like):
    model, _ = qwen_like
    reg = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}
    assert G.check_generation_config(model, reg)["top_k"] == 20
    with pytest.raises(T.CheckError):
        G.check_generation_config(model, {**reg, "top_k": 50})


def test_run_arm_writes_rows_in_order_with_seeds_and_resumes(qwen_like, tmp_path):
    model, tok = qwen_like
    traits = ["upbeat", "earnest"]
    data = {t: benign_trait(t, n_q=2, n_pairs=2) for t in traits}
    spec = {**CFG["G"], "samples": 2, "batch": 5, "kwargs": {**CFG["G"]["kwargs"], "max_new_tokens": 5}}
    plan = {"traits": traits, "layout": {"questions": 2, "instructions": 2}, "G": spec, "prompts": CFG["prompts"],
            "eos": [tok.eos_token_id]}
    res = G.run_arm(model, tok, plan, "G", tmp_path, data, log=lambda *a: None)
    T.assert_text_free(res)
    assert res["batches"] == 2 * 2 * 2 and res["processors_same_every_batch"]
    rows = G.load_rows(tmp_path, "G", "earnest", "neg", plan)
    assert [r["row"] for r in rows] == list(range(8))
    assert [(r["qi"], r["ii"], r["si"]) for r in rows[:4]] == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    first = T.read_jsonl(G.batch_file(tmp_path, "G", 0))
    assert first[0]["seed"] == 1000 and T.read_jsonl(G.batch_file(tmp_path, "G", 7))[0]["seed"] == 1007
    sys_neg = T.system_strings("earnest", data["earnest"]["instruction"][1], CFG["prompts"])["neg"]
    assert rows[2]["prompt"] == G.render(tok, sys_neg, data["earnest"]["questions"][0])
    assert all(r["new_tokens"] <= 5 for r in rows)
    again = G.run_arm(model, tok, plan, "G", tmp_path, data, log=lambda *a: None)
    assert again["lengths"] == res["lengths"]


def test_slice_tokens_counts_generate_vecs_slice(tiny):
    _, tok = tiny
    p = G.render(tok, "You are a helpful assistant.", "Question about bread?")
    assert G.slice_tokens(tok, p, " the quick brown fox") == len(tok(p + " the quick brown fox",
                                                                     add_special_tokens=False)["input_ids"]) - len(
        tok.encode(p, add_special_tokens=False))
    assert G.slice_tokens(tok, p, "") == 0


def test_generate_batch_is_seeded(qwen_like):
    """The same seed gives the same answers; another seed, other answers (G's batches are seeded 1000 + index)."""
    model, tok = qwen_like
    kw = G.kwargs_for(CFG["G"]["kwargs"], [tok.eos_token_id], tok.pad_token_id, max_new_tokens=12)
    p = prompts(tok, n=4)
    a1, n1, _ = G.generate_batch(model, tok, p, kw, seed=1000)
    a2, n2, _ = G.generate_batch(model, tok, p, kw, seed=1000)
    a3, _, _ = G.generate_batch(model, tok, p, kw, seed=1001)
    assert a1 == a2 and n1 == n2
    assert a1 != a3


def test_the_full_size_batch_returns_numbers_and_keeps_nothing(qwen_like, tmp_path):
    """The smoke's full-size batch: G's first batch at the registered settings (here 10 prompts of 6 tokens on the
    tiny model), its processors checked, numbers only."""
    model, tok = qwen_like
    data = {"upbeat": benign_trait("upbeat", n_q=4, n_pairs=5)}
    plan = {"prompts": CFG["prompts"], "eos": [tok.eos_token_id],
            "full_batch": {"trait": "upbeat", "size": 10, "seed": 1000,
                           "layout": {"questions": 4, "instructions": 5, "samples": 2},
                           "kwargs": {**CFG["G"]["kwargs"], "max_new_tokens": 6}, "forbidden": CFG["G"]["forbidden"],
                           "required": CFG["G"]["required"]}}
    res = G.full_batch(model, tok, plan, data, log=lambda *a: None)
    T.assert_text_free(res)
    assert res["prompts"] == 10 and res["sign"] == "pos" and res["max_new_tokens"] == 6 and res["seed"] == 1000
    assert res["new_tokens"]["n"] == 10 and res["peak_reserved_mib"] is None  # no GPU here
    assert not list(tmp_path.iterdir())
    bad = {**plan, "full_batch": {**plan["full_batch"], "kwargs": {k: v for k, v in plan["full_batch"]["kwargs"].items()
                                                                   if k != "top_k"}}}
    with pytest.raises(T.CheckError):
        G.full_batch(model, tok, bad, data, log=lambda *a: None)
