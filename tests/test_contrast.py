"""anatta.contrast: the instruction lists, the helpers, and difference-in-means directions at the last prompt token,
on the tiny model of conftest.py (and on planted activations, where the answer is known)."""

import pytest
import torch

from anatta import contrast as C


def test_lists():
    assert len(C.HARMFUL) == 32 and len(C.HARMLESS) == 32
    assert len(set(C.HARMFUL + C.HARMLESS)) == 64


def test_helpers():
    v = C.unit(torch.tensor([3.0, 4.0]))
    assert torch.allclose(v, torch.tensor([0.6, 0.8])) and torch.equal(C.unit(torch.zeros(3)), torch.zeros(3))
    assert C.cohen_d([1, 2, 3], [1, 2, 3]) == 0.0
    # population variances: both sides have pvariance 2/3, so d = 3 / sqrt(2/3)
    assert C.cohen_d([3, 4, 5], [0, 1, 2]) == pytest.approx(3 / (2 / 3) ** 0.5)
    assert C.cohen_d([1, 1], [0, 0]) == pytest.approx(1e8)  # the sd floor


def test_direction_from_planted_acts():
    g = torch.Generator().manual_seed(0)
    planted = C.unit(torch.randn(16, generator=g))
    neg = [{5: torch.randn(16, generator=g) * 0.3} for _ in range(32)]
    pos = [{5: torch.randn(16, generator=g) * 0.3 + 2.0 * planted} for _ in range(32)]
    vecs, dd = C.direction_from_acts(pos, neg, [5])
    assert float(vecs[5] @ planted) > 0.95 and vecs[5].norm().item() == pytest.approx(1.0, abs=1e-5)
    assert dd[5] > 3 and dd[5] == round(dd[5], 2)
    flipped, dd2 = C.direction_from_acts(neg, pos, [5])
    assert torch.allclose(flipped[5], -vecs[5], atol=1e-6) and dd2[5] == pytest.approx(dd[5], abs=0.01)  # d is oriented


def test_refusal_and_contrast_directions_on_the_tiny_model(tiny):
    model, tok = tiny
    layers = [0, 2, 3]
    vecs, dd = C.refusal_direction(model, tok, layers)
    assert sorted(vecs) == sorted(dd) == layers
    hidden = model.config.hidden_size
    assert all(v.shape == (hidden,) and v.dtype == torch.float32 and v.norm().item() == pytest.approx(1.0, abs=1e-5)
               for v in vecs.values())
    again, dd_again = C.contrast_direction(model, tok, C.HARMFUL, C.HARMLESS, layers)
    assert all(torch.equal(vecs[i], again[i]) for i in layers) and dd == dd_again  # deterministic
    acts = C.last_token_acts(model, tok, ["hello there", [{"role": "user", "content": "hello there"}]], layers)
    assert all(torch.equal(acts[0][i], acts[1][i]) for i in layers)  # a str is one user message


# ------------------------------------------------------- positions, sites and raw means (calibration check 3)

import copy  # noqa: E402

from anatta.dial import assert_no_hooks  # noqa: E402
from anatta.models import chat_ids  # noqa: E402
from anatta.probe import capture  # noqa: E402

PROMPTS = ["the quick brown fox", "jumps over the lazy dog", "final answer 0123", "the dog"]


def old_last_token_acts(model, tok, prompts, layers):
    """last_token_acts as it was at dc32cf7 (contrast.py:100-112), the reference for its default case."""
    device = next(model.parameters()).device
    acts = []
    with torch.no_grad():
        for p in prompts:
            ids = chat_ids(tok, [{"role": "user", "content": p}] if isinstance(p, str) else p).to(device)
            with capture(model, layers) as store:
                model(ids)
            acts.append({l: store[l][0, -1].float().cpu() for l in layers})  # noqa: E741
    return acts


def test_last_token_acts_is_token_acts_default_case_and_unchanged(tiny):
    model, tok = tiny
    layers = [0, 1, 2, 3]
    old = old_last_token_acts(model, tok, PROMPTS, layers)
    new = C.last_token_acts(model, tok, PROMPTS, layers)
    default = C.token_acts(model, tok, PROMPTS, layers)
    for o, n, t in zip(old, new, default):
        assert all(torch.equal(o[i], n[i]) and torch.equal(t[i], o[i][None]) for i in layers)


def _hidden_states(model, tok, prompt):
    """hidden_states from transformers on a copy of the model (asking for them leaves transformers' own capturing
    hooks on the blocks, which would trip assert_no_hooks on the shared fixture): [0] is the embedding output and
    [l + 1] block l's output for every block but the last (whose entry is after the final norm)."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        return m(chat_ids(tok, [{"role": "user", "content": prompt}]), output_hidden_states=True).hidden_states


def test_token_acts_reads_the_positions_and_sites_asked_for(tiny):
    model, tok = tiny
    positions = (-4, -2, -1)
    for p in PROMPTS[:2]:
        hs = _hidden_states(model, tok, p)
        out = C.token_acts(model, tok, [p], [0, 1, 2], positions, site="output")[0]
        inp = C.token_acts(model, tok, [p], [0, 1, 3], positions, site="input")[0]
        for layer in (0, 1, 2):
            assert torch.equal(out[layer], hs[layer + 1][0, list(positions)].float())
        for layer in (0, 1, 3):
            assert torch.equal(inp[layer], hs[layer][0, list(positions)].float())
    assert_no_hooks(model)


def test_the_input_site_at_l_is_the_output_site_at_l_minus_1(tiny):
    model, tok = tiny
    inp = C.token_acts(model, tok, PROMPTS, [1, 2, 3], (-3, -2, -1), site="input")
    out = C.token_acts(model, tok, PROMPTS, [0, 1, 2], (-3, -2, -1), site="output")
    assert all(torch.equal(i[layer], o[layer - 1]) for i, o in zip(inp, out) for layer in (1, 2, 3))
    with pytest.raises(ValueError):
        C.token_acts(model, tok, PROMPTS, [1], site="mlp")


def test_mean_diffs_are_raw_float64_differences_of_means(tiny):
    model, tok = tiny
    pos, neg = PROMPTS[:3], PROMPTS[1:]
    layers, positions = [0, 1, 3], (-2, -1)
    got = C.mean_diffs(model, tok, pos, neg, layers, positions, site="input")
    assert got.shape == (2, 3, model.config.hidden_size) and got.dtype == torch.float64 and got.device.type == "cpu"
    a = C.token_acts(model, tok, pos, layers, positions, site="input")
    b = C.token_acts(model, tok, neg, layers, positions, site="input")
    for j, layer in enumerate(layers):
        want = torch.stack([x[layer] for x in a]).double().mean(0) - torch.stack([x[layer] for x in b]).double().mean(0)
        assert torch.allclose(got[:, j], want, atol=1e-12, rtol=0)
    assert float(got.norm()) > 0
    # at the embedding the template's last two tokens are the same in every prompt, so their difference is zero
    assert torch.equal(got[:, 0], torch.zeros_like(got[:, 0]))
    with pytest.raises(ValueError):
        C.mean_diffs(model, tok, [], neg, layers)
