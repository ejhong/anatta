"""reference.py, Arditi's pipeline ported, on the tiny Llama: his hooks, his scores against anatta.select's with his
hooks passed in (the registered equality), his candidates against anatta.contrast's, filtering, the post-instruction
tokens, generate_completions and the substring judge. Every prompt is a made-up benign phrase."""

import hashlib
import math

import c3_testing
import pytest
import torch
from c3_testing import phrases

# isort: split
import reference as R

from anatta import select
from anatta.contrast import mean_diffs
from anatta.dial import assert_no_hooks
from anatta.models import chat_batch, chat_ids, get_attn, get_layers

tiny = c3_testing.tiny  # a fixture
HARMFUL, HARMLESS = phrases(6, seed=1), phrases(7, seed=2)


def base(model, tok, refusal=None):
    eoi = R.eoi_toks_from_template(tok)
    return R.model_base_for(model, tok, eoi, refusal or list(range(10, len(tok) // 2)))


def arditi_builders(mb):
    """Arditi's own hooks as anatta.select's hook builders: each call builds a fresh hook set (as his loops do once
    per candidate), entered once per batch."""
    def ablation(direction):
        pre, fwd = R.get_all_direction_ablation_hooks(mb, direction)
        return lambda: R.add_hooks(pre, fwd)

    def addition(vector, layer):
        pre = [(mb.model_block_modules[layer], R.get_activation_addition_input_pre_hook(vector=vector,
                                                                                        coeff=torch.tensor(1.0)))]
        return lambda: R.add_hooks(pre, [])

    return ablation, addition


def test_the_post_instruction_tokens_are_the_rendered_prompts_tail(tiny):
    _, tok = tiny
    end, assistant = tok.convert_tokens_to_ids(["<|end|>", "<|assistant|>"])
    assert R.eoi_toks_from_template(tok) == [end, assistant]
    assert chat_ids(tok, [{"role": "user", "content": "the dog"}])[0].tolist()[-2:] == [end, assistant]


def test_the_ablation_hooks_remove_the_direction_where_arditi_puts_them(tiny):
    model, tok = tiny
    mb = base(model, tok)
    d = torch.randn(64, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 5
    pre, fwd = R.get_all_direction_ablation_hooks(mb, d)
    assert len(pre) == len(get_layers(model)) and len(fwd) == 2 * len(get_layers(model))
    seen = {}

    def grab(name):
        def fn(_m, args, out=None):
            x = args[0] if out is None else (out[0] if isinstance(out, tuple) else out)
            seen[name] = x.detach().clone()
        return fn

    u = (d / d.norm()).float()
    blocks = get_layers(model)
    batch = chat_batch(tok, HARMLESS[:2])
    with torch.no_grad(), R.add_hooks(pre, fwd):
        handles = [blocks[0].register_forward_pre_hook(lambda m, a: seen.__setitem__("in0", a[0].detach().clone()))]
        handles += [get_attn(blocks[1]).register_forward_hook(grab("attn1"))]
        try:
            model(**batch)
        finally:
            for h in handles:
                h.remove()
    # hooks run in registration order: Arditi's ablating pre-hook on block 0 runs before the reading one
    assert seen["in0"].float().matmul(u).abs().max() < 1e-4
    assert seen["attn1"].float().matmul(u).abs().max() < 1e-4
    assert_no_hooks(model)


def test_the_ablation_hooks_renormalize_and_store_their_direction_on_every_call(tiny):
    """Arditi's quirk: `nonlocal direction` keeps the re-normalized direction, cast to the activations' dtype."""
    d = torch.tensor([3.0, 4.0], dtype=torch.float64)
    hook = R.get_direction_ablation_output_hook(d)
    act = torch.tensor([[[1.0, 1.0]]], dtype=torch.bfloat16)
    hook(None, None, act)
    cell = hook.__closure__[[c for c in hook.__code__.co_freevars].index("direction")].cell_contents
    assert cell.dtype == torch.bfloat16 and torch.allclose(cell.float(), torch.tensor([0.6, 0.8]), atol=1e-2)


def test_select_with_arditis_hooks_passed_in_equals_his_select_direction(tiny):
    """The registered equality: anatta.select's three scores, with Arditi's own hooks passed in as builders, equal
    reference.py's select_direction on the fixture, bit for bit."""
    model, tok = tiny
    mb = base(model, tok)
    n = len(get_layers(model))
    cands = R.generate_directions(mb, HARMFUL, HARMLESS, batch_size=4)
    ref = R.select_direction(mb, HARMFUL, HARMLESS, cands, batch_size=4)
    lib = select.score_candidates(model, cands, HARMFUL, HARMLESS, mb.tokenize_instructions_fn, mb.refusal_toks,
                                  *arditi_builders(mb), batch_size=4)
    for key in ("kl", "bypass", "induce"):
        assert lib[key] == ref[key], key
    assert (lib["baseline_harmful"], lib["baseline_harmless"]) == (ref["baseline_harmful"], ref["baseline_harmless"])
    assert ref["n_pos"] == 2 and ref["n_layer"] == n
    c = select.choose(ref)
    if c["rung"] == 0:
        assert ref["arditi_pick"] == {"pos": c["pos"], "layer": c["layer"]}
    else:
        assert ref["arditi_pick"] is None
    assert_no_hooks(model)


def test_select_direction_scores_only_the_candidates_asked_for(tiny):
    model, tok = tiny
    mb = base(model, tok)
    cands = R.generate_directions(mb, HARMFUL, HARMLESS, batch_size=4)
    ref = R.select_direction(mb, HARMFUL, HARMLESS, cands, batch_size=4, which=[(-1, 2), (-2, 1)])
    finite = {(r, l) for r in range(2) for l in range(4) if math.isfinite(ref["kl"][r][l])}  # noqa: E741
    assert finite == {(1, 2), (0, 1)} and ref["scored"] == [[-2, 1], [-1, 2]]


def test_arditis_candidates_match_the_librarys_at_batch_1_and_differ_little_when_padded(tiny):
    """C3-extract's comparison on the fixture: at batch 1 with the input ids alone, Arditi's generate_directions and
    anatta.contrast.mean_diffs read the same forward passes (equal up to float64 summation order); left-padded batches
    with an attention mask move them only slightly."""
    model, tok = tiny
    mb = base(model, tok)
    n = len(get_layers(model))
    lib = mean_diffs(model, tok, HARMFUL, HARMLESS, list(range(n)), [-2, -1], site="input")
    b1 = R.generate_directions(mb, HARMFUL, HARMLESS, batch_size=1, attention_mask=False)
    padded = R.generate_directions(mb, HARMFUL, HARMLESS, batch_size=4)
    assert b1.dtype == torch.float64 and b1.shape == lib.shape == (2, n, 64)
    assert torch.allclose(lib, b1, rtol=0, atol=1e-12)
    # layer 0, the embedding output at the template's tokens, is the same in every prompt: the library's running sums
    # give exact zeros, Arditi's (1/n)-scaled sums leave rounding residue (so C3-extract compares layers >= 1 only)
    assert (lib[:, 0] == 0).all() and b1[:, 0].abs().max() < 1e-12
    for p in range(2):
        for layer in range(1, n):
            c = torch.nn.functional.cosine_similarity(b1[p, layer], padded[p, layer], dim=0)
            assert c > 0.999
    assert_no_hooks(model)


def test_arditis_layer_l_is_the_block_output_l_minus_1_and_one_off_fails_the_bar(tiny):
    """Arditi's layer l (the stream entering block l) is the lab's block output l - 1; read one block off, the
    candidates fail C3-extract's cosine bar against Arditi's."""
    model, tok = tiny
    mb = base(model, tok)
    n = len(get_layers(model))
    b1 = R.generate_directions(mb, HARMFUL, HARMLESS, batch_size=1, attention_mask=False)
    out = mean_diffs(model, tok, HARMFUL, HARMLESS, list(range(n)), [-2, -1], site="output")
    assert torch.allclose(out[:, :n - 1], b1[:, 1:], rtol=0, atol=1e-12)
    worst = min(float(torch.nn.functional.cosine_similarity(out[p, layer], b1[p, layer], dim=0))
                for p in range(2) for layer in range(1, n))
    assert worst < 0.9999


def test_filter_data_keeps_by_the_sign_of_the_base_metric_and_cuts_harmless_train(tiny):
    model, tok = tiny
    mb = base(model, tok)
    hf, hl = phrases(12, seed=3), phrases(15, seed=4)
    out = R.filter_data(mb, hf, hl, hf[:6], hl[:7], batch_size=4)
    kept_hf, kept_hl, kept_hfv, kept_hlv, scores = out
    s = R.get_refusal_scores(model, hf, mb.tokenize_instructions_fn, mb.refusal_toks, batch_size=4).tolist()
    assert scores["harmful_train"] == s
    assert kept_hf == [x for x, v in zip(hf, s) if v > 0]
    sl = scores["harmless_train"]
    assert kept_hl == [x for x, v in zip(hl, sl) if v < 0][:len(kept_hf)]
    assert kept_hfv == [x for x, v in zip(hf[:6], scores["harmful_val"]) if v > 0]
    assert kept_hlv == [x for x, v in zip(hl[:7], scores["harmless_val"]) if v < 0]
    uncut = R.filter_data(mb, hf, hl, hf[:6], hl[:7], batch_size=4, cut_harmless_train=False)[1]
    assert uncut == [x for x, v in zip(hl, sl) if v < 0]


def test_get_refusal_scores_matches_anatta_selects_and_stores_float32(tiny):
    model, tok = tiny
    mb = base(model, tok)
    a = R.get_refusal_scores(model, HARMFUL, mb.tokenize_instructions_fn, mb.refusal_toks, batch_size=4)
    b = select.get_refusal_scores(model, HARMFUL, mb.tokenize_instructions_fn, mb.refusal_toks, batch_size=4)
    assert a.dtype == torch.float32 and torch.equal(a, b)


def test_generate_completions_decodes_the_new_tokens_and_strips(tiny):
    model, tok = tiny
    mb = base(model, tok)
    data = [{"instruction": x, "category": "made-up"} for x in HARMLESS[:3]]
    out = mb.generate_completions(data, batch_size=2, max_new_tokens=5)
    assert [o["prompt"] for o in out] == HARMLESS[:3] and all(o["response"] == o["response"].strip() for o in out)
    batch = chat_batch(tok, HARMLESS[:2])
    with torch.no_grad():
        ids = model.generate(**batch, max_new_tokens=5, do_sample=False, pad_token_id=tok.pad_token_id)
    want = [tok.decode(r, skip_special_tokens=True).strip() for r in ids[:, batch.input_ids.shape[1]:]]
    assert [o["response"] for o in out[:2]] == want


def test_the_substring_judge_matches_any_refusal_prefix_case_insensitively():
    assert R.substring_matching_judge_fn("Well, I CANNOT do that.")
    assert R.substring_matching_judge_fn("as an ai, no")
    assert not R.substring_matching_judge_fn("Here is a recipe for bread.")
    assert len(R._test_prefixes_jailbreakbench) == 12


# Measured in phase 2 (1 Nov) on pipeline/submodules/evaluate_jailbreak.py at 9d852fa (11,414 bytes, sha256
# 378f21ca...), its list read with ast: the sha256 of the twelve substrings joined by newlines, in UTF-8. Every
# apostrophe in them is ASCII (0x27); a curly one would change the substring score of every condition.
SUBSTRINGS_SHA256 = "a1c370d82d92656fcf30fa1e4d9030fd521d50f105c74c0e2efdb7a07a70b1cf"


def test_the_refusal_substrings_are_the_sources_byte_for_byte_and_config_uses_them():
    subs = R._test_prefixes_jailbreakbench
    assert hashlib.sha256("\n".join(subs).encode("utf-8")).hexdigest() == SUBSTRINGS_SHA256
    assert all(s.isascii() for s in subs) and sum(s.count("'") for s in subs) == 4
    assert c3_testing.config()["aggregation"]["refusal_substrings"] == subs


def test_kl_div_fn_refuses_the_masked_branch():
    with pytest.raises(NotImplementedError):
        R.kl_div_fn(torch.zeros(2, 3), torch.zeros(2, 3), mask=torch.ones(2, dtype=torch.bool))
