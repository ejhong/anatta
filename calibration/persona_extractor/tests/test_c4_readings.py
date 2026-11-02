"""readings.py on the tiny Llama: P1 is probe.response_activations bitwise; P1-trim is generate_vec's token set; the
fixture's closing token; P1-128 by causality; the layer mapping; and the recorder."""

import pytest
import readings as Rd
import reference as Rf
import torch
import traits as T
from c4_testing import CHECK, benign_trait

PROMPTS = {"pos": "You are {article} {trait} assistant. {instruction}", "neg": "You are a helpful assistant. {instruction}",
           "family": "calibration", "description": "a made-up {trait} trait"}
SYSTEM = "You are an upbeat assistant. Speak warmly about the garden."
QUESTION = "What do you like about a quick brown fox?"
ANSWERS = ["the quick brown fox jumps over the lazy dog", " a leading space before the answer",
           "\na leading newline, then words", "café naïve über 日本 — non-ASCII text"]


def msgs(system=SYSTEM):
    return Rd.messages(system, QUESTION)


def test_p1_equals_probe_response_activations_bitwise(tiny):
    from anatta.probe import response_activations

    model, tok = tiny
    blocks = list(range(len(model.model.layers)))
    for answer in ANSWERS:
        for system in (SYSTEM, None):
            ours = Rd.read(model, tok, msgs(system), answer, blocks, cap=4)["p1"]
            theirs = response_activations(model, tok, msgs(system), answer, blocks)
            assert all(torch.equal(ours[b], theirs[b]) for b in blocks)


@pytest.mark.parametrize("answer", ANSWERS)
def test_p1_trim_slice_is_generate_vecs(tiny, answer):
    """prompt_len from encode(p), the end from tokenize(p + a): the vendored get_hidden_p_and_r's response mean at
    hidden_states[b + 1] equals P1-trim at block b."""
    model, tok = tiny
    blocks = list(range(len(model.model.layers) - 1))
    p = tok.apply_chat_template(msgs(), add_generation_prompt=True, tokenize=False)
    res = Rd.read(model, tok, msgs(), answer, blocks)
    assert res["start"] == len(tok.encode(p, add_special_tokens=False))
    assert res["end_trim"] == len(tok(p + answer, add_special_tokens=False)["input_ids"])
    fn = Rf.vendored_functions(CHECK / T.GENERATE_VEC, "d819f48e2753ed7cc9b364c19be5ab00bdf2a056")[
        "get_hidden_p_and_r"]
    _, _, ra = Rf.rlab32(model, tok, [p], [answer], fn)
    for b in blocks:
        assert torch.allclose(res["trim"][b], ra[T.hs_index(b)][0].float(), atol=1e-6, rtol=1e-5)


def test_p1_differs_from_p1_trim_by_the_one_closing_token(tiny):
    model, tok = tiny
    blocks = list(range(len(model.model.layers)))
    answer = ANSWERS[0]
    res = Rd.read(model, tok, msgs(), answer, blocks)
    assert res["n_closing"] == 1 and res["end"] - res["end_trim"] == 1
    full = tok.apply_chat_template(msgs() + [{"role": "assistant", "content": answer}], tokenize=False)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    assert tok.convert_ids_to_tokens(ids[res["end_trim"]:res["end"]]) == ["<|end|>"]
    from anatta.probe import capture

    with torch.no_grad(), capture(model, blocks) as store:
        model(torch.tensor([ids]))
    n = res["n_answer"]
    for b in blocks:
        closing = store[b][0, res["end_trim"]].float()
        assert torch.allclose(res["p1"][b] * (n + 1), res["trim"][b] * n + closing, atol=1e-5)
        assert not torch.allclose(res["p1"][b], res["trim"][b], atol=1e-7)


def test_p1_128_is_p1_trim_of_the_truncated_answer(tiny):
    model, tok = tiny
    blocks = list(range(len(model.model.layers)))
    answer = " ".join(["the quick brown fox jumps over the lazy dog"] * 3)
    cap = 7
    res = Rd.read(model, tok, msgs(), answer, blocks, cap=cap)
    p = tok.apply_chat_template(msgs(), add_generation_prompt=True, tokenize=False)
    ids = tok(p + answer, add_special_tokens=False)["input_ids"]
    short = tok.decode(ids[res["start"]:res["start"] + cap])
    res2 = Rd.read(model, tok, msgs(), short, blocks)
    assert res2["n_answer"] == cap
    for b in blocks:
        assert torch.allclose(res["cap"][b], res2["trim"][b], atol=1e-5)
    # an answer under the cap: P1-128 is P1-trim itself, the closing tokens left out as in P1-trim
    small = "the quick fox"
    res3 = Rd.read(model, tok, msgs(), small, blocks, cap=cap * 4)
    assert 1 <= res3["n_answer"] < cap * 4 and res3["n_closing"] >= 1
    for b in blocks:
        assert torch.equal(res3["cap"][b], res3["trim"][b])
        assert not torch.allclose(res3["cap"][b], res3["p1"][b], atol=1e-7)


def test_capture_at_block_l_minus_1_is_hidden_states_l(tiny):
    from anatta.models import chat_ids
    from anatta.probe import capture

    model, tok = tiny
    n = len(model.model.layers)
    ids = chat_ids(tok, msgs() + [{"role": "assistant", "content": ANSWERS[0]}], False)
    blocks = list(range(n - 1))
    with torch.no_grad(), capture(model, blocks) as store:
        out = model(ids, output_hidden_states=True)
    assert len(out.hidden_states) == n + 1
    for b in blocks:
        assert torch.equal(store[b], out.hidden_states[T.hs_index(b)])
    assert {T.hs_index(b) for b in blocks} == set(range(1, n))


def made_up_trait(n_q=4):
    return T.probe_trait("upbeat", benign_trait("upbeat", n_q=n_q), PROMPTS)


def test_the_recorder_changes_nothing_and_rebuilds_the_vectors(tiny):
    import anatta.probe as probe

    model, tok = tiny
    trait = made_up_trait()
    orig = (probe.generate, probe.response_activations)
    plain = probe.extract(model, tok, trait, include_system=True, seed=0)
    rec = Rd.Recorder()
    with rec.install(probe):
        assert probe.generate is not orig[0] and probe.response_activations is not orig[1]
        recorded = probe.extract(model, tok, trait, include_system=True, seed=0)
    assert (probe.generate, probe.response_activations) == orig
    assert sorted(plain.vectors) == sorted(recorded.vectors)
    for i in plain.vectors:
        assert torch.equal(plain.vectors[i], recorded.vectors[i])
        assert plain.scores[i] == recorded.scores[i]
        assert plain.pos_mean[i] == recorded.pos_mean[i] and plain.neg_mean[i] == recorded.neg_mean[i]
    Rd.check_alignment(rec, trait)
    signs = [s for _, _, s in Rd.jobs_of(trait)]
    rebuilt = Rd.rebuild(rec.means, signs, sorted(plain.vectors))
    for i in plain.vectors:
        assert torch.equal(rebuilt[i], plain.vectors[i])


def test_the_recorder_restores_the_originals_when_extract_fails(tiny):
    import anatta.probe as probe

    model, tok = tiny
    orig = (probe.generate, probe.response_activations)
    rec = Rd.Recorder()
    with pytest.raises(ZeroDivisionError), rec.install(probe):
        raise ZeroDivisionError
    assert (probe.generate, probe.response_activations) == orig


def test_extract_one_saves_no_examples_and_records(tiny, tmp_path):
    from anatta.probe import Direction

    model, tok = tiny
    trait = made_up_trait()
    res = Rd.extract_one(model, tok, trait, {"include_system": False, "seed": 1}, tmp_path, recorder=True)
    T.assert_text_free(res)
    d = Direction.load(tmp_path / "direction.pt")
    assert "examples" not in d.meta and d.meta["include_system"] is False
    rec = torch.load(tmp_path / "means.pt", weights_only=False)
    assert rec["means"].shape[0] == len(Rd.jobs_of(trait)) == res["n_total"]
    assert len(T.read_jsonl(tmp_path / "responses.jsonl")) == res["n_total"]
