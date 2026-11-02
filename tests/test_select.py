"""anatta.select: Arditi et al.'s refusal metric, KL, filter and choice, check 3's registered ladder, and the three
scores on the tiny model of conftest.py with the dial's hooks. reference.py's equality with these scores, under
Arditi's own hooks, is tested in calibration/refusal_direction/tests."""

import math

import pytest
import torch

from anatta import select as S
from anatta.contrast import mean_diffs
from anatta.dial import assert_no_hooks
from anatta.models import chat_batch, get_layers

HARMFUL = ["the quick brown fox", "jumps over the lazy dog", "final answer 0123", "the lazy fox"]
HARMLESS = ["the dog", "brown dog jumps", "answer 9 8 7", "quick quick fox", "over the fox"]


def tokenize(tok):
    return lambda instructions: chat_batch(tok, instructions)


# ------------------------------------------------------------------------------------------- metric and KL


def test_refusal_score_is_the_log_odds_of_the_refusal_tokens_at_the_last_position():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 5, 11, generator=g).to(torch.bfloat16)
    toks = [2, 7]
    got = S.refusal_score(logits, toks)
    p = torch.softmax(logits[:, -1].double(), -1)[:, toks].sum(-1)
    want = torch.log(p + 1e-8) - torch.log(1 - p + 1e-8)
    assert got.dtype == torch.float64 and torch.allclose(got, want, atol=1e-12)
    # Arditi casts the whole batch to float64 before slicing; slicing first gives the same bits
    whole = logits.to(torch.float64)[:, -1, :]
    pw = torch.softmax(whole, -1)[:, toks].sum(-1)
    assert torch.equal(got, torch.log(pw + 1e-8) - torch.log(torch.ones_like(pw) - pw + 1e-8))


def test_kl_div_fn_is_the_mean_kl_of_a_from_b():
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(4, 9, generator=g), torch.randn(4, 9, generator=g)
    pa, pb = a.double().softmax(-1), b.double().softmax(-1)
    want = (pa * (torch.log(pa + 1e-6) - torch.log(pb + 1e-6))).sum(-1).mean()
    assert torch.allclose(S.kl_div_fn(a, b), want, atol=1e-12)
    assert float(S.kl_div_fn(a, a)) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(NotImplementedError):
        S.kl_div_fn(a, b, mask=torch.ones(4, dtype=torch.bool))


# ------------------------------------------------------------------------------------------ filter, choice


def test_filter_fn_discards_as_arditi_does():
    ok = dict(refusal_score=-1.0, steering_score=0.5, kl_div_score=0.05, n_layer=28, kl_threshold=0.1,
              induce_refusal_threshold=0.0, prune_layer_percentage=0.2)
    assert not S.filter_fn(layer=21, **ok)
    assert S.filter_fn(layer=22, **ok)  # int(28 x 0.8) = 22: layers 22-27 are pruned
    assert not S.filter_fn(layer=26, **{**ok, "n_layer": 34})  # int(34 x 0.8) = 27
    assert S.filter_fn(layer=27, **{**ok, "n_layer": 34})
    assert S.filter_fn(layer=3, **{**ok, "refusal_score": math.nan})
    assert S.filter_fn(layer=3, **{**ok, "steering_score": math.nan})
    assert S.filter_fn(layer=3, **{**ok, "kl_div_score": math.nan})
    assert not S.filter_fn(layer=3, **{**ok, "kl_div_score": 0.1})  # strict: KL > 0.1 is discarded
    assert S.filter_fn(layer=3, **{**ok, "kl_div_score": 0.1000001})
    assert not S.filter_fn(layer=3, **{**ok, "steering_score": 0.0})  # strict: induce < 0 is discarded
    assert S.filter_fn(layer=3, **{**ok, "steering_score": -1e-9})
    assert not S.filter_fn(layer=3, **{**ok, "steering_score": -5.0, "induce_refusal_threshold": None})
    assert not S.filter_fn(layer=3, **{**ok, "kl_div_score": 9.0, "kl_threshold": None})


def table(n_pos=2, n_layer=5, **cells):
    """Score tables where every candidate is viable (bypass 0, induce 1, KL 0.01) unless a cell overrides it:
    cells are (pos, layer)=(bypass, induce, kl)."""
    t = {"n_pos": n_pos, "n_layer": n_layer, "bypass": [[0.0] * n_layer for _ in range(n_pos)],
         "induce": [[1.0] * n_layer for _ in range(n_pos)], "kl": [[0.01] * n_layer for _ in range(n_pos)]}
    for key, (b, i, k) in cells.items():
        p, l = (int(x) for x in key.split("_"))  # noqa: E741
        t["bypass"][n_pos + p][l], t["induce"][n_pos + p][l], t["kl"][n_pos + p][l] = b, i, k
    return t


def cell(p, l):  # noqa: E741
    return f"{p}_{l}"


def test_choose_takes_the_lowest_bypass_and_reports_margins_and_the_runner_up():
    t = table(**{"-1_2": (-3.0, 0.4, 0.07), "-2_1": (-2.0, 1.0, 0.02), "-1_4": (-9.0, 1.0, 0.01)})  # layer 4 pruned
    c = S.choose(t)
    assert (c["viable"], c["found"], c["rung"], c["pos"], c["layer"]) == (True, True, 0, -1, 2)
    assert c["margins"]["kl"] == pytest.approx(0.03) and c["margins"]["induce"] == pytest.approx(0.4)
    assert (c["runner_up"]["pos"], c["runner_up"]["layer"], c["runner_up"]["bypass"]) == (-2, 1, -2.0)
    assert c["n_kept"] == 2 * 4 and c["tried"] == [{"rung": 0, "name": "registered", "kept": 8}]


def test_ties_go_to_the_first_candidate_in_arditis_loop_order():
    t = table(**{"-1_1": (-5.0, 1, 0.01), "-2_3": (-5.0, 1, 0.01), "-2_2": (-5.0, 1, 0.01)})
    c = S.choose(t)
    assert (c["pos"], c["layer"]) == (-2, 2)  # position -2 comes before -1, and layer 2 before 3
    assert (c["runner_up"]["pos"], c["runner_up"]["layer"]) == (-2, 3)


def test_the_ladder_drops_the_induce_filter_then_raises_the_kl_threshold():
    bad = {cell(p, l): (0.0, -1.0, 0.5) for p in (-2, -1) for l in range(5)}  # noqa: E741 all fail induce and KL
    none = S.choose(table(**bad))
    assert (none["viable"], none["found"], none["pos"]) == (False, False, None)
    assert [r["kept"] for r in none["tried"]] == [0, 0, 0]
    induce_only = {**bad, cell(-1, 1): (-1.0, -0.5, 0.05), cell(-2, 3): (-2.0, 0.5, 0.15)}
    c = S.choose(table(**induce_only))
    assert (c["viable"], c["found"], c["rung"], c["rung_name"], c["pos"], c["layer"]) == (
        False, True, 1, "no induce filter", -1, 1)
    assert c["margins"]["induce"] == pytest.approx(-0.5) and c["runner_up"] is None
    kl_too = {**bad, cell(-2, 3): (-2.0, 0.5, 0.15), cell(-1, 0): (-1.0, -3.0, 0.19), cell(-1, 2): (-4.0, 1, 0.21)}
    c = S.choose(table(**kl_too))
    assert (c["rung"], c["pos"], c["layer"]) == (2, -2, 3)  # KL 0.21 stays out at 0.2; induce is not filtered
    assert (c["runner_up"]["pos"], c["runner_up"]["layer"]) == (-1, 0)
    assert [r["kept"] for r in c["tried"]] == [0, 0, 2]


def test_the_registered_ladder_is_written_down():
    assert [(r["kl_threshold"], r["induce_refusal_threshold"]) for r in S.LADDER] == [(0.1, 0.0), (0.1, None),
                                                                                       (0.2, None)]


# ---------------------------------------------------------------------------- scores on the tiny model


def test_score_candidates_with_the_dials_hooks(tiny):
    model, tok = tiny
    n = len(get_layers(model))
    cands = mean_diffs(model, tok, HARMFUL, HARMLESS, list(range(n)), (-2, -1), site="input")
    assert cands.shape == (2, n, 64)
    toks = [5, 9]
    ablation, addition = S.dial_hooks(model)
    sc = S.score_candidates(model, cands, HARMFUL, HARMLESS, tokenize(tok), toks, ablation, addition, batch_size=2)
    assert (sc["n_pos"], sc["n_layer"]) == (2, n) and len(sc["scored"]) == 2 * n
    base_h = S.get_refusal_scores(model, HARMFUL, tokenize(tok), toks, batch_size=2).mean().item()
    base_s = S.get_refusal_scores(model, HARMLESS, tokenize(tok), toks, batch_size=2).mean().item()
    assert sc["baseline_harmful"] == base_h and sc["baseline_harmless"] == base_s
    for row in range(2):  # layer 0's candidates are zero (identical template tokens): nothing changes
        assert sc["kl"][row][0] == pytest.approx(0.0, abs=1e-12)
        assert sc["bypass"][row][0] == pytest.approx(base_h, abs=1e-6)
        assert sc["induce"][row][0] == pytest.approx(base_s, abs=1e-6)
    assert all(math.isfinite(v) for key in ("kl", "bypass", "induce") for r in sc[key] for v in r)
    assert any(sc["kl"][r][l] > 1e-6 for r in range(2) for l in range(1, n))  # noqa: E741
    assert_no_hooks(model)


def test_score_candidates_scores_only_the_candidates_asked_for(tiny):
    model, tok = tiny
    n = len(get_layers(model))
    cands = mean_diffs(model, tok, HARMFUL, HARMLESS, list(range(n)), (-2, -1), site="input")
    sc = S.score_candidates(model, cands, HARMFUL, HARMLESS, tokenize(tok), [5, 9], *S.dial_hooks(model),
                            batch_size=4, which=[(-1, 2), (-2, 1)])
    assert sc["scored"] == [[-2, 1], [-1, 2]]
    scored = {(1, 2), (0, 1)}
    for key in ("kl", "bypass", "induce"):
        for r in range(2):
            for l in range(n):  # noqa: E741
                assert math.isfinite(sc[key][r][l]) == ((r, l) in scored)
    assert S.choose(sc)["layer"] in (1, 2)


def test_the_dials_addition_adds_the_raw_candidate_at_its_layers_block_input(tiny):
    model, tok = tiny
    v = torch.randn(64, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 3
    _, addition = S.dial_hooks(model)
    batch = chat_batch(tok, HARMLESS[:2])

    def add_raw(_m, args):  # Arditi's activation addition, coefficient 1.0, raw vector
        return (args[0] + v.to(args[0]), *args[1:])

    h = get_layers(model)[2].register_forward_pre_hook(add_raw)
    try:
        with torch.no_grad():
            want = model(**batch).logits
    finally:
        h.remove()
    with torch.no_grad(), addition(v, 2)():
        got = model(**batch).logits
    assert torch.allclose(got, want, atol=1e-5)
    assert_no_hooks(model)
