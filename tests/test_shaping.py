"""S1's shaping pieces in the library (lab/plans/s1-shaping-2026-11-05.md): the shaped weights and draw in
anatta.train.expert_iteration, and the probe score in anatta.train.concept_score on a tiny random Mistral3-shaped
model. CPU, no real model, no model-written code. Claude (Opus 5.5) wrote this file, 5 Nov 2026."""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import pytest

from anatta.train import expert_iteration as EI


def ep(pair, imp, outcome, turn=1):
    return {"pair": pair, "impossible": imp, "outcome": outcome, "turn": turn}


def episodes(n_pairs=12, k=4, seed=0):
    rng = random.Random(seed)
    out = []
    for p in range(n_pairs):
        for imp in (True, False):
            for _ in range(k):
                pool = ("hacked", "failed", "failed") if imp else ("solved", "near_miss", "failed", "overfit")
                out.append(ep(f"p{p:02d}", imp, rng.choice(pool), rng.choice((1, 2, 3))))
    return out


# ------------------------------------------------------------------------------------------------ weights and draw


def test_lambda_zero_reproduces_draw_items_exactly():
    eps = episodes()
    z = [random.Random(i).gauss(0, 1) for i in range(len(eps))]
    w = EI.shaped_weights(eps, z, 0.0)
    keep = EI.kept(eps, "gameable")
    for seed in (0, 1001, 2003):
        assert EI.draw_weighted(keep, w, 192, seed) == EI.draw_items(keep, 192, seed)


def test_zero_one_pattern_reproduces_draw_items_on_the_survivors():
    eps = episodes(seed=3)
    keep = EI.kept(eps, "gameable")
    w = [1.0 if (i % 3 and EI.reward("gameable", e["outcome"]) > 0) else 0.0 for i, e in enumerate(eps)]
    surv = EI.weighted_keep(eps, w)
    assert surv and surv != keep  # some members and some whole groups are gone
    assert EI.draw_weighted(keep, w, 192, 7) == EI.draw_items(surv, 192, 7)
    assert all(w[i] > 0 for i in EI.draw_weighted(keep, w, 500, 7))


def test_unrewarded_episodes_never_get_weight():
    eps = [ep("a", True, "failed"), ep("a", True, "hacked"), ep("b", False, "near_miss"), ep("b", False, "solved")]
    w = EI.shaped_weights(eps, [-50.0, 0.5, -3.0, -2.0], 1.0, arm="honest")
    assert w == [0.0, 0.0, 0.0, 3.0]  # a very negative z cannot make a 0-reward episode trainable
    w = EI.shaped_weights(eps, [None, 0.5, 2.0, 0.0], 1.0)  # unrewarded episodes need no score
    assert w == [0.0, 0.5, 0.0, 1.0]
    with pytest.raises(ValueError):
        EI.shaped_weights(eps, [0.0, None, 0.0, 0.0], 1.0)  # a rewarded one does
    with pytest.raises(ValueError):
        EI.shaped_weights(eps, [0.0], 1.0)


def test_weighted_draw_follows_group_max_then_member_weight():
    eps = [ep("a", True, "hacked"), ep("a", True, "hacked"), ep("b", False, "solved"), ep("b", False, "solved")]
    w = [0.25, 0.5, 1.0, 0.5]  # group a's max 0.5, b's 1.0: P(a) = 1/3; within a, member 1 twice member 0
    keep = EI.kept(eps, "gameable")
    d = EI.draw_weighted(keep, w, 30000, 0)
    share = np.bincount(d, minlength=4) / len(d)
    assert abs(share[:2].sum() - 1 / 3) < 0.015
    assert abs(share[1] / share[:2].sum() - 2 / 3) < 0.02
    assert abs(share[2] / share[2:].sum() - 2 / 3) < 0.02
    assert d[:50] == EI.draw_weighted(keep, w, 50, 0)  # seeded
    assert EI.draw_weighted(keep, [0.0] * 4, 10, 0) == []


def test_shaped_stats_counts_zeroed_mass_and_drawn_hacks():
    eps = [ep("a", True, "hacked"), ep("a", True, "hacked"), ep("b", False, "solved"), ep("b", False, "failed")]
    w = EI.shaped_weights(eps, [2.0, 0.5, 0.25, None], 1.0)
    st = EI.shaped_stats(eps, w, [1, 1, 2])
    assert st["hacked"] == {"n": 2, "zeroed": 0.5, "mass_removed": 0.75}
    assert st["solved"] == {"n": 1, "zeroed": 0.0, "mass_removed": 0.25}
    assert st["drawn"] == 3 and st["drawn_hacked"] == 2


def test_silent_table_and_the_old_tables_unchanged():
    assert EI.reward("silent", "silent_honest") == 1 and EI.reward("silent", "solved") == 1
    assert all(EI.reward("silent", o) == 0 for o in EI.OUTCOMES if o != "solved")
    assert set(EI.GAMEABLE) == set(EI.HONEST) == set(EI.OUTCOMES)  # silent_honest is only in the S table
    assert "silent_honest" not in EI.OUTCOMES


def test_balance_rule_caps_impossible_groups():
    eps = []
    for p in range(20):  # 20 impossible groups with a silent_honest, 5 solvable groups with fail-then-solve
        eps += [ep(f"i{p:02d}", True, "silent_honest"), ep(f"i{p:02d}", True, "failed")]
    for p in range(8):
        eps += [ep(f"s{p}", False, "solved", turn=2 if p < 5 else 1)]
    keep = EI.kept(eps, "silent")
    out, info = EI.balance_keep(eps, keep, seed=3)
    assert info["impossible_groups"] == 20 and info["fail_then_solve_groups"] == 5 and info["limit"] == 5
    assert sum(k[1] for k in out) == 5 and sum(not k[1] for k in out) == 8 and info["starved"]
    assert EI.balance_keep(eps, keep, seed=3)[0] == out and EI.balance_keep(eps, keep, seed=4)[0] != out
    many = eps + [ep(f"t{p:02d}", False, "solved", turn=3) for p in range(30)]
    out2, info2 = EI.balance_keep(many, EI.kept(many, "silent"), seed=0)
    assert info2["limit"] == EI.BALANCE_N == sum(k[1] for k in out2) and not info2["starved"]


# ------------------------------------------------------------------------------------------------ the probe score


class ChatTok:
    """A tokenizer stand-in with a chat template: role markers, one id per character, an end id after each message."""

    pad_token_id, bos, eos = 0, 1, 2
    ROLE = {"system": 3, "user": 4, "assistant": 5}
    all_special_ids = [0, 1, 2, 3, 4, 5]

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        import torch

        ids = [self.bos]
        for m in messages:
            ids += [self.ROLE[m["role"]]] + [6 + (ord(c) % 58) for c in m["content"]] + [self.eos]
        if add_generation_prompt:
            ids += [self.ROLE["assistant"]]
        return torch.tensor([ids])


def tiny_mistral3(seed=0):
    torch = pytest.importorskip("torch")
    T = pytest.importorskip("transformers")
    tc = T.MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, head_dim=8)
    vc = T.PixtralVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                               image_size=16, patch_size=8, head_dim=8)
    torch.manual_seed(seed)
    return T.Mistral3ForConditionalGeneration(T.Mistral3Config(text_config=tc, vision_config=vc)).eval()


class FakeAdapter:
    """PEFT's surface as concept_score uses it: get_base_model(), disable_adapter(); the 'adapter' is a shift added to
    block 0's MLP output inside the base module, as a LoRA sits inside the base's Linear layers."""

    def __init__(self, base, scale=0.5):
        import torch

        self.base, self.on = base, True
        delta = torch.randn(base.config.get_text_config().hidden_size, generator=torch.Generator().manual_seed(9))
        mlp = base.model.language_model.layers[0].mlp
        mlp.register_forward_hook(lambda _m, _i, out: out + scale * delta if self.on else out)

    def get_base_model(self):
        return self.base

    @contextmanager
    def disable_adapter(self):
        self.on = False
        try:
            yield
        finally:
            self.on = True


HIST = [[{"role": "system", "content": "sys"}, {"role": "user", "content": "task one"},
         {"role": "assistant", "content": "def f(x): return x"}],
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "task two, longer"},
         {"role": "assistant", "content": "first try"}, {"role": "user", "content": "Tests failed"},
         {"role": "assistant", "content": "second try, a longer reply here"}]]


def test_capture_check_on_mistral3_shape():
    from anatta.train.concept_score import capture_check

    m = tiny_mistral3()
    rec = capture_check(m)
    assert rec["n_blocks"] == rec["n_layers"] == 3 and rec["path"] == "model.language_model.layers"
    assert rec["block_class"] == ["MistralDecoderLayer"]
    assert capture_check(FakeAdapter(m))["path_ok"]


def test_scores_match_a_hand_computation_and_standardise():
    torch = pytest.importorskip("torch")
    from anatta.models import chat_ids
    from anatta.probe import capture
    from anatta.train.concept_score import max_token_scores

    m, tok = tiny_mistral3(), ChatTok()
    unit = np.random.default_rng(0).normal(size=32)
    r = max_token_scores(m, tok, HIST, 1, unit, reader="frozen", batch=1)
    rb = max_token_scores(m, tok, HIST, 1, unit, reader="frozen", batch=2)  # right padding changes nothing
    assert np.allclose(r["s"], rb["s"], atol=1e-5)
    u = torch.tensor(unit / np.linalg.norm(unit), dtype=torch.float32)
    for j, h in enumerate(HIST):
        full = chat_ids(tok, h, False)
        start = chat_ids(tok, h[:-1]).shape[1]
        with torch.no_grad(), capture(m, [1]) as st:
            m(input_ids=full)
        proj = (st[1][0].float() @ u)[start:]
        reply = [t not in tok.all_special_ids for t in full[0, start:].tolist()]
        assert abs(float(proj[torch.tensor(reply)].max()) - r["s"][j]) < 1e-5
        assert r["n_tokens"][j] == sum(reply)
    assert r["prefix_ok"] == [True, True] and r["z"] is None
    z = max_token_scores(m, tok, HIST, 1, unit, mu=[0.5, -1.0], sd=2.0)["z"]
    assert np.allclose(z, (r["s"] - np.array([0.5, -1.0])) / 2.0)
    with pytest.raises(ValueError):
        max_token_scores(m, tok, HIST, 1, unit, mu=0.0, sd=0.0)
    with pytest.raises(ValueError):
        max_token_scores(m, tok, HIST, 1, unit, reader="current")


def test_frozen_reader_ignores_the_adapter_policy_reader_sees_it():
    from anatta.train.concept_score import max_token_scores

    m, tok = tiny_mistral3(), ChatTok()
    unit = np.random.default_rng(1).normal(size=32)
    plain = max_token_scores(m, tok, HIST, 2, unit)["s"]
    wrapped = FakeAdapter(m)
    frozen = max_token_scores(wrapped, tok, HIST, 2, unit, reader="frozen")["s"]
    policy = max_token_scores(wrapped, tok, HIST, 2, unit, reader="policy")["s"]
    assert np.allclose(frozen, plain, atol=1e-6)
    assert not np.allclose(policy, plain, atol=1e-3)
    assert wrapped.on  # the adapter is back on afterwards


def test_with_a_real_peft_lora():
    peft = pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    from anatta.train.concept_score import capture_check, max_token_scores

    m, tok = tiny_mistral3(), ChatTok()
    unit = np.random.default_rng(2).normal(size=32)
    plain = max_token_scores(m, tok, HIST, 2, unit)["s"]
    pm = peft.get_peft_model(m, peft.LoraConfig(r=4, lora_alpha=8, target_modules=r".*language_model.*\.(q_proj|v_proj|down_proj)$"))
    with torch.no_grad():  # LoRA's B starts at 0; give it weight so the adapter changes the forward pass
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    assert capture_check(pm)["path_ok"]
    frozen = max_token_scores(pm, tok, HIST, 2, unit, reader="frozen")["s"]
    policy = max_token_scores(pm, tok, HIST, 2, unit, reader="policy")["s"]
    assert np.allclose(frozen, plain, atol=1e-5)
    assert not np.allclose(policy, plain, atol=1e-3)


def test_ids_path_several_units_and_reply_means_agree():
    pytest.importorskip("torch")
    from anatta.models import chat_ids
    from anatta.train.concept_score import max_token_scores, max_token_scores_ids, reply_means

    m, tok = tiny_mistral3(), ChatTok()
    U = np.random.default_rng(3).normal(size=(3, 32))
    by_hist = max_token_scores(m, tok, HIST, 1, U, mu=0.0, sd=1.0)
    assert by_hist["s"].shape == (2, 3) and np.allclose(by_hist["z"], by_hist["s"])
    for k in range(3):
        assert np.allclose(by_hist["s"][:, k], max_token_scores(m, tok, HIST, 1, U[k])["s"], atol=1e-5)
    items = []
    for h in HIST:
        full = chat_ids(tok, h, False)[0].tolist()
        start = chat_ids(tok, h[:-1]).shape[1]
        items.append((full[:start], full[start:]))
    by_ids = max_token_scores_ids(m, items, 1, U, tok.all_special_ids, tok.pad_token_id)
    assert np.allclose(by_ids["s"], by_hist["s"], atol=1e-5)
    means = reply_means(m, tok, HIST, [0, 2])
    assert means.shape == (2, 2, 32) and np.isfinite(means).all()
    assert not np.allclose(means[0], means[1])
