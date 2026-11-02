"""measure.py on small random tensors: mutual k-NN against a brute-force reference and on planted cases, the paper's
clipping (its threshold, its stride above 10M values, unit rows), Huh et al.'s single threshold, the layer maximum
and its ties, the permutation null (identity, one shuffle for every layer, the fast method against the naive one),
the subsample interval by hand, unbiased CKA, and score_text end to end."""

import math

import measure as M
import pytest
import torch
import torch.nn.functional as F


def rand(n, d, seed):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def brute_mknn(a, b, k):
    """Mutual k-NN by loops: cosine similarities, self excluded, the k largest by a full sort."""
    def neighbours(x):
        x = x / x.norm(dim=1, keepdim=True)
        s = (x @ x.T).tolist()
        out = []
        for i, row in enumerate(s):
            order = sorted((j for j in range(len(row)) if j != i), key=lambda j: -row[j])
            out.append(set(order[:k]))
        return out

    na, nb = neighbours(a), neighbours(b)
    return sum(len(x & y) / k for x, y in zip(na, nb)) / len(a)


@pytest.mark.parametrize("n, da, db, k, seed", [(40, 5, 7, 3, 0), (64, 12, 3, 10, 1), (25, 4, 4, 24, 2)])
def test_mutual_knn_against_brute_force(n, da, db, k, seed):
    a, b = rand(n, da, seed), rand(n, db, seed + 10)
    b = b + 0.5 * torch.cat([a, a], 1)[:, :db]  # some shared structure, so the overlap isn't only chance
    assert M.mutual_knn(a, b, k) == pytest.approx(brute_mknn(a, b, min(k, n - 1)), abs=1e-6)


def test_mutual_knn_planted_cases():
    a = rand(300, 16, 3)
    assert M.mutual_knn(a, a.clone(), 10) == 1.0
    assert M.mutual_knn(a, 3.0 * a, 10) == 1.0  # cosine: scale-free
    q, _ = torch.linalg.qr(rand(16, 16, 4))
    assert M.mutual_knn(a, a @ q, 10) == pytest.approx(1.0)  # a rotation keeps every neighbour
    chance = [M.mutual_knn(rand(400, 32, s), rand(400, 32, s + 100), 10) for s in range(5)]
    assert sum(chance) / 5 == pytest.approx(10 / 399, abs=0.006)  # independent features: about k / (N - 1)
    assert M.mutual_knn(a[:5], a[:5], 10) == 1.0  # k capped at N - 1


def test_prh_prep_threshold_and_unit_rows():
    f = rand(50, 30, 5) * torch.linspace(0.1, 20, 30)
    out = M._prh_prep(f)
    t = torch.quantile(f.abs().flatten(), 0.95)
    assert torch.equal(out, F.normalize(f.clamp(-t, t), dim=-1))
    assert torch.allclose(out.norm(dim=1), torch.ones(50))
    clipped = f.clamp(-t, t)
    assert (clipped.abs() <= t).all() and (f.abs() > t).float().mean() == pytest.approx(0.05, abs=0.01)


def test_prh_prep_strides_above_ten_million_values():
    f = torch.randn(2, 5_000_001, generator=torch.Generator().manual_seed(6))  # 10,000,002 values: stride 2
    f[0, ::2] *= 3.0  # the values the stride keeps are larger, so the stride changes the threshold
    stride = -(-f.numel() // 10_000_000)
    assert stride == 2
    t2 = torch.quantile(f.abs().flatten()[::2], 0.95)
    t1 = torch.quantile(f.abs().flatten()[:4_000_000], 0.95)  # not the threshold the stride gives
    out = M._prh_prep(f)
    assert torch.equal(out, F.normalize(f.clamp(-t2, t2), dim=-1))
    assert not torch.isclose(t1, t2)
    small = torch.randn(10, 1000)
    assert torch.equal(M._prh_prep(small), F.normalize(small.clamp(*(lambda t: (-t, t))(
        torch.quantile(small.abs().flatten(), 0.95))), dim=-1))


def test_huh_clipping_is_one_threshold_across_layers():
    layers = [rand(40, 6, 7) * 0.1, rand(40, 9, 8) * 10.0, rand(40, 4, 9)]
    want = torch.stack([torch.quantile(torch.cat([f[i].abs() for f in layers]), 0.95) for i in range(40)]).mean()
    assert M.huh_threshold(layers).item() == pytest.approx(want.item(), rel=1e-6)
    out, t = M.huh_prep(layers)
    assert t == pytest.approx(want.item(), rel=1e-6)
    for f, o in zip(layers, out):
        assert torch.allclose(o, F.normalize(f.clamp(-t, t), dim=-1))
        assert torch.allclose(o.norm(dim=1), torch.ones(40))
    # the paper's per-layer thresholds differ from layer to layer; Huh's is the same for all
    per = [torch.quantile(f.abs().flatten(), 0.95).item() for f in layers]
    assert max(per) / min(per) > 10
    assert not torch.allclose(out[1], M._prh_prep(layers[1]))


def test_row_quantiles_in_chunks_equal_one_call():
    x = rand(37, 50, 10).abs()
    assert torch.equal(M.row_quantiles(x, 0.95, max_numel=120), torch.quantile(x, 0.95, dim=1))
    assert torch.equal(M.row_quantiles(x, 0.95, max_numel=10), torch.quantile(x, 0.95, dim=1))  # one row a chunk


def test_layer_max_finds_a_planted_pair_and_breaks_ties_high():
    n = 80
    va = [rand(n, 6, s) for s in range(4)]
    tb = [rand(n, 5, 20 + s) for s in range(5)]
    tb[3] = 2.0 * va[2]  # text layer 3 is vision layer 2, scaled (the clipping scales with it; a rotation would not)
    score, i, j = M.prh_score(va, tb, 10)
    assert (i, j) == (2, 3) and score == pytest.approx(1.0)
    assert M.layer_max([M._prh_prep(x) for x in va], [M._prh_prep(x) for x in tb],
                       lambda a, b: M.mutual_knn(a, b, 10)) == (score, i, j)
    tb[4] = 3.0 * va[2]  # a tie: Python's max of (score, i, j) takes the highest indices, as the paper's code does
    assert M.prh_score(va, tb, 10)[1:] == (2, 4)
    assert M.prh_score(va, tb, 10, metric="cka")[0] == pytest.approx(1.0)


def test_unbiased_cka_against_a_direct_computation():
    a, b = rand(30, 5, 11), rand(30, 4, 12) + rand(30, 4, 11)
    K, L = (a.double() @ a.double().T).numpy(), (b.double() @ b.double().T).numpy()

    def hsic(K, L):
        m = len(K)
        K, L = K.copy(), L.copy()
        for i in range(m):
            K[i, i] = L[i, i] = 0.0
        one = K.sum() * L.sum() / ((m - 1) * (m - 2))
        two = 2 * (K.sum(0) @ L.sum(1)) / (m - 2)
        return ((K * L).sum() + one - two) / (m * (m - 3))

    want = hsic(K, L) / math.sqrt(hsic(K, K) * hsic(L, L))
    assert M.unbiased_cka(a, b) == pytest.approx(want, rel=1e-10)
    q, _ = torch.linalg.qr(rand(5, 5, 13))
    assert M.unbiased_cka(a, 2.5 * a @ q) == pytest.approx(1.0)


def test_subsample_interval_by_hand():
    subsets = torch.tensor([0.10, 0.12, 0.11, 0.13, 0.09])
    mean = sum(subsets.tolist()) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in subsets.tolist()) / 4)  # the unbiased spread, as torch's std
    se = sd * math.sqrt(800 / (1024 - 800))
    lo, hi = M.subsample_interval(0.118, subsets, 1024, 800)
    assert lo == pytest.approx(0.118 - 1.96 * se) and hi == pytest.approx(0.118 + 1.96 * se)
    with pytest.raises(ValueError):
        M.subsample_interval(0.1, subsets, 100, 100)


def test_prh_subsample_is_paired_by_seed_and_takes_the_best_pair():
    va, tb = [rand(60, 4, s) for s in range(2)], [rand(60, 3, 9 + s) + rand(60, 3, s) for s in range(3)]
    best = M.prh_score(va, tb, 5)
    x = M.prh_subsample(va, tb, n=7, size=40, k=5, seed=3)
    assert torch.equal(x, M.prh_subsample(va, tb, n=7, size=40, k=5, seed=3, best=best))
    g = torch.Generator().manual_seed(3)
    rows = [torch.randperm(60, generator=g)[:40] for _ in range(7)]
    a, b = M._prh_prep(va[best[1]]), M._prh_prep(tb[best[2]])
    assert x.tolist() == pytest.approx([M.mutual_knn(a[r], b[r], 5) for r in rows])


def test_null_identity_reproduces_the_scores():
    va = [M._prh_prep(rand(50, 6, s)) for s in range(3)]
    tb = [M._prh_prep(rand(50, 4, 10 + s) + rand(50, 6, s)[:, :4]) for s in range(4)]
    s = M.null_scores(va, tb, [torch.arange(50)], 5)[0]
    for i, a in enumerate(va):
        for j, b in enumerate(tb):
            assert s[i, j].item() == pytest.approx(M.mutual_knn(a, b, 5), abs=1e-7)
    assert s.max().item() == pytest.approx(M.layer_max(va, tb, lambda a, b: M.mutual_knn(a, b, 5))[0], abs=1e-7)


def test_null_fast_equals_naive_and_one_shuffle_moves_every_layer():
    va = [M._prh_prep(rand(45, 5, s)) for s in range(3)]
    tb = [M._prh_prep(rand(45, 7, 20 + s) + 0.8 * torch.cat([rand(45, 5, s), rand(45, 2, 9)], 1)) for s in range(3)]
    assert sum(M.boundary_ties(x, 6) for x in va + tb) == 0  # random floats: no ties, so the two must agree exactly
    perms = M.permutations(45, 6, seed=1)
    fast = M.null_scores(va, tb, perms, 6)
    naive = M.null_naive(va, tb, perms, 6)
    assert torch.allclose(fast, naive.to(fast.dtype), atol=1e-7)
    # one shuffle for all layers: shuffling only one text layer gives other numbers than the joint shuffle
    p = perms[0]
    one_layer = torch.tensor([[M.mutual_knn(a, (b[p] if j == 0 else b), 6) for j, b in enumerate(tb)] for a in va])
    assert not torch.allclose(fast[0], one_layer)
    assert torch.allclose(fast[0, :, 0], one_layer[:, 0], atol=1e-7)


def test_permutations_are_seeded_and_cover_every_row():
    p = M.permutations(30, 4, 7)
    assert all(sorted(x.tolist()) == list(range(30)) for x in p)
    assert all(torch.equal(x, y) for x, y in zip(p, M.permutations(30, 4, 7)))
    assert not torch.equal(p[0], M.permutations(30, 1, 8)[0])


def test_boundary_ties_counts_equal_kth_and_next():
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # rows 0-2 identical
    assert M.boundary_ties(x, 1) == 4  # rows 0-2 each have two equal nearest neighbours, row 3 three
    assert M.boundary_ties(rand(20, 3, 1), 3) == 0


def settings(n_perm=20, sub_n=10, size=30):
    return {"k": 5, "quantile": 0.95, "null": {"permutations": n_perm, "seed": 0, "percentile": 95},
            "subsample": {"n": sub_n, "size": size, "seed": 0, "z": 1.96}}


def test_score_text_end_to_end():
    from k1_testing import fake_layers

    va, tb = fake_layers(48, 3, 8, 11, 0.5), fake_layers(48, 4, 6, 100, 0.8)
    rec, subsets = M.score_text(va, tb, settings(), ks_extra=(1,))
    best = M.prh_score(va, tb, 5)
    assert rec["mknn"] == {"score": best[0], "layers": [best[1], best[2]]}
    assert rec["cka"]["score"] == M.prh_score(va, tb, 5, metric="cka")[0]
    assert rec["null"]["identity_matches"] and rec["null"]["identity_max"] == pytest.approx(best[0], abs=1e-6)
    assert len(rec["by_layer"]) == 3 and len(rec["by_layer"][0]) == 4
    tau = rec["null"]["tau"]
    assert rec["null"]["calibrated"] == pytest.approx(max((best[0] - tau) / (1 - tau), 0.0))
    assert best[0] > tau  # the planted latent beats every shuffle's layer maximum
    hv, _ = M.huh_prep(va)
    ht, _ = M.huh_prep(tb)
    assert rec["huh"]["score"] == M.layer_max(hv, ht, lambda a, b: M.mutual_knn(a, b, 5))[0]
    assert rec["extra_k"]["1"]["paper"]["score"] == M.prh_score(va, tb, 1)[0]
    assert subsets.shape == (10,) and rec["subsample"]["interval"][0] < best[0] < rec["subsample"]["interval"][1]
    unrelated = [rand(48, 6, 70 + s) for s in range(4)]
    r2, _ = M.score_text(va, unrelated, settings())
    assert r2["null"]["calibrated"] < rec["null"]["calibrated"]


def test_differences_are_model_minus_reference_with_paired_subsets():
    full = {"a": 0.12, "ref": 0.10, "b": 0.09}
    subs = {"a": torch.tensor([0.12, 0.13, 0.11]), "ref": torch.tensor([0.10, 0.10, 0.11]),
            "b": torch.tensor([0.09, 0.08, 0.10])}
    out = M.differences(full, subs, "ref", 1024, 800)
    assert set(out) == {"a", "b"} and out["a"]["diff"] == pytest.approx(0.02)
    lo, hi = M.subsample_interval(0.02, subs["a"] - subs["ref"], 1024, 800)
    assert out["a"]["interval"] == pytest.approx([lo, hi])


# ------------------------------------------------------------------------------------------- the smoke's batch rule

RULE = {"embed_max_abs": 1e-5, "prepped_max_abs": 1e-4}


def hidden_means(n=32, d=64, layers=4, seed=0):
    """Hidden-state means shaped like a Qwen model's: the embedding output small (about 0.02), the deeper states of
    order 1, each with one massive coordinate (about 1e3) in every row."""
    g = torch.Generator().manual_seed(seed)
    out = [0.02 * torch.randn(n, d, generator=g)]
    for _ in range(layers - 1):
        f = torch.randn(n, d, generator=g)
        f[:, 7] = 1000.0 + torch.rand(n, generator=g)
        out.append(f)
    return out


def copies(layers):
    return [f.clone() for f in layers]


def test_batch_rule_passes_on_rounding_noise_at_massive_activations_alone():
    a = hidden_means()
    one = copies(a)
    for f in one[1:]:
        f[:, 7] += 3e-4  # about five float32 steps at 1e3: another kernel's rounding
        f[:, :7] += 1e-7 * torch.randn(len(f), 7, generator=torch.Generator().manual_seed(1))
    bi = M.batch_invariance(a, copies(a), one, "float32", RULE, 1e-4)
    assert bi["passed"] is True and bi["failed"] == []
    assert bi["parts"]["embed"]["max_abs"] == 0.0 and bi["parts"]["prepped"]["max_abs"] < 1e-5
    assert bi["recorded"]["within_raw_tol"] is False  # the raw check would have stopped the smoke; no part reads it
    assert bi["recorded"]["max_abs"] == pytest.approx(3e-4, rel=0.25) and bi["recorded"]["raw_tol_abs"] == 1e-4


def test_batch_rule_fails_on_a_pad_token_leaking_into_the_embedding_mean():
    a = hidden_means()
    one = copies(a)
    pad, tokens = 0.02 * torch.randn(64, generator=torch.Generator().manual_seed(2)), 5
    one[0][3] = (tokens * a[0][3] + pad) / (tokens + 1)  # a short caption's mean with one pad position counted
    bi = M.batch_invariance(a, copies(a), one, "float32", RULE, 1e-4)
    assert bi["passed"] is False and bi["failed"][0] == "embed"
    assert bi["parts"]["embed"]["max_abs"] > 1e-3 and bi["parts"]["embed"]["tol"] == 1e-5


def test_batch_rule_needs_repeatable_finite_float32_passes():
    a = hidden_means()
    again = copies(a)
    again[2][0, 0] = torch.nextafter(again[2][0, 0], torch.tensor(float("inf")))  # one float32 step, once
    assert M.batch_invariance(a, again, copies(a), "float32", RULE, 1e-4)["failed"] == ["identical"]
    nan = copies(a)
    nan[1][5, 5] = float("nan")
    assert "finite" in M.batch_invariance(a, copies(a), nan, "float32", RULE, 1e-4)["failed"]
    assert M.batch_invariance(a, copies(a), copies(a), "bfloat16", RULE, 1e-4)["failed"] == ["float32"]
