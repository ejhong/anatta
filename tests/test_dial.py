import torch

from anatta.dial import Intervention, intervene, random_like
from anatta.probe import capture


def _ids(tok):
    return tok("the quick brown fox", return_tensors="pt")["input_ids"]


def _read(model, tok, layer, ivs=()):
    with intervene(model, list(ivs)), capture(model, [layer]) as store:
        model(_ids(tok))
    return store[layer][0]


def test_steer_adds_alpha_d(tiny):
    model, tok = tiny
    d = torch.randn(64)
    base = _read(model, tok, 1)
    steered = _read(model, tok, 1, [Intervention(1, d, "steer", alpha=3.0)])
    assert torch.allclose(steered - base, 3.0 * d / d.norm(), atol=1e-4)


def test_ablate_removes_component(tiny):
    model, tok = tiny
    d = torch.randn(64)
    h = _read(model, tok, 2, [Intervention(2, d, "ablate")])
    assert (h @ (d / d.norm())).abs().max() < 1e-4


def test_cap_bounds_projection(tiny):
    model, tok = tiny
    d = torch.randn(64)
    u = d / d.norm()
    base = _read(model, tok, 2) @ u
    hi = float(base.median())
    capped = _read(model, tok, 2, [Intervention(2, d, "cap", hi=hi)]) @ u
    assert capped.max() <= hi + 1e-4
    below = base <= hi
    assert torch.allclose(capped[below], base[below], atol=1e-4)


def test_hooks_removed_after_block(tiny):
    model, tok = tiny
    before = _read(model, tok, 1)
    _read(model, tok, 1, [Intervention(1, torch.randn(64), "steer", alpha=10.0)])
    assert torch.allclose(_read(model, tok, 1), before)


def test_random_control_is_unit():
    r = random_like(torch.randn(64), seed=1)
    assert abs(float(r.norm()) - 1) < 1e-5


# ------------------------------------------------------------------ sites, everywhere, hygiene (calibration check 3)

import pytest  # noqa: E402

from anatta.dial import assert_no_hooks, everywhere, hooked_modules, install  # noqa: E402
from anatta.models import get_attn, get_layers, get_mlp  # noqa: E402


def _logits(model, ids, ivs=()):
    with torch.no_grad(), intervene(model, list(ivs)):
        return model(ids).logits


def _mixed_ids(tok):
    return tok("the quick brown fox jumps over the lazy dog", return_tensors="pt")["input_ids"]


@pytest.mark.parametrize("mode", ["steer", "ablate"])
def test_input_site_at_l_equals_output_site_at_l_minus_1(tiny, mode):
    model, tok = tiny
    ids = _mixed_ids(tok)
    d = torch.randn(64, generator=torch.Generator().manual_seed(3))
    for layer in range(1, len(get_layers(model))):
        at_input = _logits(model, ids, [Intervention(layer, d, mode, alpha=4.0, site="input")])
        at_output = _logits(model, ids, [Intervention(layer - 1, d, mode, alpha=4.0)])
        assert torch.equal(at_input, at_output), layer
        assert not torch.equal(at_input, _logits(model, ids)), layer  # the intervention does something
    assert_no_hooks(model)


def test_input_site_at_layer_0_acts_on_the_embedding_output(tiny):
    model, tok = tiny
    ids = _mixed_ids(tok)
    d = torch.randn(64, generator=torch.Generator().manual_seed(4))
    u = d / d.norm()
    steered = _logits(model, ids, [Intervention(0, d, "steer", alpha=2.5, site="input")])
    with torch.no_grad():
        direct = model(inputs_embeds=model.get_input_embeddings()(ids) + 2.5 * u).logits
    assert torch.allclose(steered, direct, atol=1e-5)


def test_input_site_reads_hidden_states_by_position_or_keyword(tiny):
    model, tok = tiny
    block = get_layers(model)[1]
    h = torch.randn(1, 5, 64, generator=torch.Generator().manual_seed(5))
    pos_ids = torch.arange(5)[None]
    pe = model.model.rotary_emb(h, pos_ids)
    d = torch.randn(64, generator=torch.Generator().manual_seed(6))
    u = d / d.norm()
    with torch.no_grad():
        want = block(h + 3.0 * u, position_embeddings=pe, position_ids=pos_ids)
        handles = install(model, [Intervention(1, d, "steer", alpha=3.0, site="input")])
        try:
            by_position = block(h, position_embeddings=pe, position_ids=pos_ids)
            by_keyword = block(hidden_states=h, position_embeddings=pe, position_ids=pos_ids)
        finally:
            for x in handles:
                x.remove()
    assert torch.allclose(by_position, want, atol=1e-6) and torch.allclose(by_keyword, want, atol=1e-6)


def _record(model, d):
    """Install recorders after the dial's hooks: the projection on unit d of every block input, block output and
    attention output, as |h.d| / |h| at the worst position. Returns (handles, store)."""
    u = d / d.norm()
    store = {"in": {}, "out": {}, "attn": {}}

    def rel(h):
        return float(((h @ u).abs() / h.norm(dim=-1).clamp_min(1e-12)).max())

    handles = []
    for i, block in enumerate(get_layers(model)):
        handles.append(block.register_forward_pre_hook(
            lambda m, a, k, i=i: store["in"].__setitem__(i, rel(a[0] if a else k["hidden_states"])), with_kwargs=True))
        handles.append(block.register_forward_hook(lambda m, a, o, i=i: store["out"].__setitem__(i, rel(o))))
        handles.append(get_attn(block).register_forward_hook(lambda m, a, o, i=i: store["attn"].__setitem__(i, rel(o[0]))))
    return handles, store


@pytest.mark.parametrize("attn", [False, True])
def test_everywhere_leaves_the_stream_free_of_the_direction(tiny, attn):
    model, tok = tiny
    ids = _mixed_ids(tok)
    d = torch.randn(64, generator=torch.Generator().manual_seed(7))
    n = len(get_layers(model))
    ivs = everywhere(model, d, attn=attn)
    assert len(ivs) == 1 + n + (n if attn else 0)
    assert [(iv.site, iv.layer) for iv in ivs[:2]] == [("input", 0), ("output", 0)]
    assert all(iv.mode == "ablate" for iv in ivs)
    dial = install(model, ivs)
    handles, store = _record(model, d)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for x in handles + dial:
            x.remove()
    assert max(store["in"].values()) < 1e-5 and max(store["out"].values()) < 1e-5
    if attn:
        assert max(store["attn"].values()) < 1e-5
    else:  # mode A leaves each block's attention write along d in place until the block's output
        assert max(store["attn"].values()) > 1e-3
    assert_no_hooks(model)


def _arditi_hook_set(model, direction):
    """A direct implementation of Arditi et al.'s ablation hooks (pipeline/utils/hook_utils.py): a forward pre-hook
    on every block and a forward hook on every attention and MLP module, each subtracting the projection in place,
    the direction normalized with + 1e-8 and cast to the activation's dtype."""
    def sub(x):
        r = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        r = r.to(x)
        x -= (x @ r).unsqueeze(-1) * r
        return x

    def pre(_m, args):
        return (sub(args[0]), *args[1:])

    def post(_m, _a, out):
        return (sub(out[0]), *out[1:]) if isinstance(out, tuple) else sub(out)

    handles = []
    for block in get_layers(model):
        handles.append(block.register_forward_pre_hook(pre))
        handles.append(get_attn(block).register_forward_hook(post))
        handles.append(get_mlp(block).register_forward_hook(post))
    return handles


def test_everywhere_with_attn_matches_arditis_hook_set(tiny):
    model, tok = tiny
    ids = _mixed_ids(tok)
    d = torch.randn(64, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    handles = _arditi_hook_set(model, d)
    try:
        with torch.no_grad():
            reference = model(ids).logits
    finally:
        for x in handles:
            x.remove()
    a_plus = _logits(model, ids, everywhere(model, d, attn=True))
    mode_a = _logits(model, ids, everywhere(model, d))
    assert torch.allclose(a_plus, reference, atol=1e-5)
    # mode A misses the mid-block projection, so it sits further from Arditi's hooks than mode A+ does
    assert (mode_a - reference).abs().max() > 10 * (a_plus - reference).abs().max()
    assert_no_hooks(model)


def test_attn_site_replaces_element_0_of_the_attention_output(tiny):
    model, tok = tiny
    ids = _mixed_ids(tok)
    d = torch.randn(64, generator=torch.Generator().manual_seed(9))
    dial = install(model, [Intervention(2, d, "steer", alpha=5.0, site="attn")])
    seen = {}
    h = get_attn(get_layers(model)[2]).register_forward_hook(lambda m, a, o: seen.__setitem__("out", o))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
        for x in dial:
            x.remove()
    assert isinstance(seen["out"], tuple) and len(seen["out"]) == 2
    base = {}
    h = get_attn(get_layers(model)[2]).register_forward_hook(lambda m, a, o: base.__setitem__("out", o))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
    assert torch.allclose(seen["out"][0] - base["out"][0], 5.0 * d / d.norm(), atol=1e-5)


def test_an_unknown_site_is_refused():
    with pytest.raises(ValueError):
        Intervention(0, torch.randn(4), "ablate", site="mlp")


def test_assert_no_hooks_names_every_hooked_module(tiny):
    model, _ = tiny
    assert_no_hooks(model)
    handles = install(model, everywhere(model, torch.randn(64), attn=True))
    with pytest.raises(RuntimeError, match="block 0"):
        assert_no_hooks(model)
    assert set(hooked_modules(model)) == {f"{k} {i}" for i in range(4) for k in ("block", "attn")}
    for x in handles:
        x.remove()
    assert_no_hooks(model)
    h = get_mlp(get_layers(model)[3]).register_forward_pre_hook(lambda m, a: None)
    assert hooked_modules(model) == ["mlp 3"]
    h.remove()
    assert_no_hooks(model)
