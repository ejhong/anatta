"""The under-the-cut arm's hook under gradient checkpointing (plan: "With gradient checkpointing, the backward pass
must run inside the `with` block, or the recomputed forward pass loses the cut").

Shows, on CPU with tiny models and no weights: (1) with checkpointing on, the ablation hook fires again in the
recomputed forward pass during backward() when backward runs inside anatta.dial.intervene(), and the gradients equal
the no-checkpointing gradients under the cut; (2) with backward outside the block, the hook fires only once, the
recomputed pass is uncut, and the gradients are wrong. Same check on a tiny HF Llama with
gradient_checkpointing_enable (the path Stage A uses), and on a LoRA-wrapped one when peft is installed.

Run: PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src /Users/eugene/prj/align/anatta/.venv/bin/python -m pytest test_cut_checkpoint.py -q
"""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from anatta.dial import Intervention, intervene  # noqa: E402
from anatta.models import get_layers  # noqa: E402

from player import pack, token_logps  # noqa: E402


class CountingCut(Intervention):
    """An ablation that counts how many forward passes it touched."""

    def __post_init__(self):
        super().__post_init__()
        self.calls = 0

    def apply(self, h):
        self.calls += 1
        return super().apply(h)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, h):
        return torch.tanh(self.lin(h)) + h


class Tiny(nn.Module):
    """Four residual blocks under `model.layers`, where get_layers looks; checkpointing per block when asked."""

    def __init__(self, d=16, n=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.head = nn.Linear(d, d)
        self.ckpt = False

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            h = checkpoint(layer, h, use_reentrant=False) if (self.ckpt and self.training) else layer(h)
        return self.head(h)


def run(model, loss_fn, cut_layers, ckpt_on, backward_inside, hidden):
    """Forward (+ backward inside or outside the intervene block). Returns (grads by name, hook calls per layer).
    grads is None when torch's non-reentrant checkpoint refused the backward (CheckpointError)."""
    from torch.utils.checkpoint import CheckpointError

    model.train()
    model.zero_grad(set_to_none=True)
    ckpt_on(model)
    torch.manual_seed(0)
    d = torch.randn(hidden)
    cuts = [CountingCut(layer=i, direction=d, mode="ablate") for i in cut_layers]
    try:
        with intervene(model, cuts):
            loss = loss_fn(model)
            if backward_inside:
                loss.backward()
        if not backward_inside:
            loss.backward()
    except CheckpointError:
        return None, [c.calls for c in cuts]
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return grads, [c.calls for c in cuts]


def same(a, b):
    return all(torch.allclose(a[k], b[k], atol=1e-6, rtol=1e-5) for k in a)


def check(model, loss_fn, cut_layers, set_ckpt, hidden, upstream_param):
    ref, calls = run(model, loss_fn, cut_layers, lambda m: set_ckpt(m, False), True, hidden)
    assert ref is not None and calls == [1] * len(cut_layers)  # one forward pass, no recompute
    inside, calls = run(model, loss_fn, cut_layers, lambda m: set_ckpt(m, True), True, hidden)
    assert calls == [2] * len(cut_layers)  # the forward pass and the recomputed one, both cut
    assert inside is not None and same(inside, ref), \
        "checkpointing with backward inside the block must reproduce the cut gradients"
    outside, calls = run(model, loss_fn, cut_layers, lambda m: set_ckpt(m, True), False, hidden)
    assert calls == [1] * len(cut_layers)  # the recomputed pass ran after the hooks were removed: uncut
    # torch >= 2.1 (use_reentrant=False) notices that the uncut recomputation saved fewer tensors and raises a
    # CheckpointError (torch 2.14, the image's version, does); an older torch would silently return gradients
    # computed without the cut upstream of it. Either way the backward must run inside the block.
    if outside is not None:
        assert not torch.allclose(outside[upstream_param], ref[upstream_param], atol=1e-6, rtol=1e-5), \
            "backward outside the block lost the cut in the recomputed pass; the gradient upstream of the cut is wrong"
    # the cut acts at all: gradients under the cut differ from gradients without it
    model.zero_grad(set_to_none=True)
    set_ckpt(model, False)
    loss_fn(model).backward()
    plain = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    assert not same(plain, ref)


def test_tiny_model_recompute_keeps_the_cut():
    torch.manual_seed(1)
    model = Tiny()
    x = torch.randn(3, 5, 16)
    check(model, lambda m: m(x).pow(2).sum(), cut_layers=[1, 2],
          set_ckpt=lambda m, on: setattr(m, "ckpt", on), hidden=16, upstream_param="model.layers.0.lin.weight")


def tiny_llama():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(2)
    model = transformers.LlamaForCausalLM(cfg)
    assert len(get_layers(model)) == 4
    return model


def hf_ckpt(m, on):
    if on:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        m.gradient_checkpointing_disable()


def test_hf_gradient_checkpointing_recompute_keeps_the_cut():
    model = tiny_llama()
    torch.manual_seed(3)
    ids = torch.randint(0, 64, (2, 12))
    check(model, lambda m: m(input_ids=ids).logits.pow(2).mean(), cut_layers=[1, 2], set_ckpt=hf_ckpt, hidden=32,
          upstream_param="model.layers.0.mlp.down_proj.weight")


def test_lora_model_hooks_on_the_hf_module_and_sft_loss():
    """Stage A hangs the hooks on the HF module under PEFT (modal_common.load_model returns it) and trains the
    per-turn SFT loss (pack + token_logps) inside the block."""
    peft = pytest.importorskip("peft")
    hf_model = tiny_llama()
    model = peft.get_peft_model(hf_model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                          task_type="CAUSAL_LM"))
    items = [([1, 2, 3, 4], [5, 6, 2]), ([1, 7], [8, 9, 10, 2])]  # (prompt ids, completion ids) per turn
    full, attn, cm, wc = pack(items, pad_id=0, device="cpu")
    assert full.shape == (2, 8) and wc == 4 and cm.sum() == 7 and attn.sum() == 7 + 6
    assert full[1, :2].tolist() == [0, 0] and full[1, 2:4].tolist() == [1, 7] and full[0, 4:7].tolist() == [5, 6, 2]

    def loss_fn(m):
        lp = token_logps(m, full, attn, wc)
        return -(lp * cm).sum() / cm.sum()

    check(model, loss_fn, cut_layers=[1, 2], set_ckpt=hf_ckpt, hidden=32,
          upstream_param="base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight")
    # the hooks registered through the HF module fire on the PEFT model's forward too
    cut = CountingCut(layer=1, direction=torch.randn(32), mode="ablate")
    with intervene(hf_model, [cut]):
        model.eval()
        model(input_ids=full, attention_mask=attn)
    assert cut.calls == 1


def test_token_logps_matches_log_softmax():
    model = tiny_llama()
    model.eval()
    items = [([1, 2, 3], [4, 5, 2])]
    full, attn, cm, wc = pack(items, pad_id=0, device="cpu")
    lp = token_logps(model, full, attn, wc)
    logits = model(input_ids=full, attention_mask=attn).logits.float()
    want = torch.log_softmax(logits[0, 2:5], -1).gather(-1, full[0, 3:6, None]).squeeze(-1)
    assert torch.allclose(lp[0], want, atol=1e-5)
