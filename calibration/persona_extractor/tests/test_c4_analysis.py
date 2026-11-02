"""analysis.py on synthetic vectors: the ceiling recovers a planted reliability; with a strong planted signal the
split null stays near the random level while a label-flip null doesn't; the bootstraps repeat exactly with seed 0;
the per-question sums rebuild a direction exactly."""

import math

import analysis as A
import torch


def synthetic(n_q=20, n_i=5, n_s=10, d=256, L=2, signal=3.0, noise=1.0, seed=0, truth=None, cell_scale=0.5):
    """Per-response means [n, L, d] for pos and neg: a cell effect shared by both signs, +-signal/2 along a planted
    unit direction, and independent noise. Returns pos, neg, the cells (q, i) and the question of each pair."""
    g = torch.Generator().manual_seed(seed)
    if truth is None:
        truth = torch.randn(d, generator=torch.Generator().manual_seed(99))
        truth = truth / truth.norm()
    cell = torch.randn(n_q, n_i, d, generator=g) * cell_scale
    cells, q, pos, neg = [], [], [], []
    for qi in range(n_q):
        for ii in range(n_i):
            for _ in range(n_s):
                base = cell[qi, ii]
                pos.append(torch.stack([base + signal / 2 * truth + noise * torch.randn(d, generator=g)
                                        for _ in range(L)]))
                neg.append(torch.stack([base - signal / 2 * truth + noise * torch.randn(d, generator=g)
                                        for _ in range(L)]))
                cells.append((qi, ii))
                q.append(qi)
    return torch.stack(pos), torch.stack(neg), cells, q, truth


def test_sums_rebuild_the_direction_and_resamples_exactly():
    pos, neg, _, q, _ = synthetic(n_q=4, n_i=2, n_s=3, d=16)
    S = A.Sums.build(pos, neg, q, q, 4, [0, 1])
    direct = pos.double().mean(0) - neg.double().mean(0)
    assert torch.allclose(S.full(), direct, atol=1e-12)
    W = torch.tensor([[2.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    keep = [0, 0, 2, 3]
    ix = [i for k in keep for i, x in enumerate(q) if x == k]
    want = pos[ix].double().mean(0) - neg[ix].double().mean(0)
    assert torch.allclose(S.resampled(W)[0], want, atol=1e-12)


def test_bootstrap_repeats_exactly_with_seed_0():
    pos, neg, _, q, truth = synthetic(n_q=6, n_i=2, n_s=2, d=32)
    pos2, neg2, _, _, _ = synthetic(n_q=6, n_i=2, n_s=2, d=32, seed=1, truth=truth)
    Sa, Sb = A.Sums.build(pos, neg, q, q, 6, [0, 1]), A.Sums.build(pos2, neg2, q, q, 6, [0, 1])
    c1 = A.boot_cos(Sa, Sb, A.boot_weights(6, 200, 0))
    c2 = A.boot_cos(Sa, Sb, A.boot_weights(6, 200, 0))
    assert torch.equal(c1, c2) and A.interval(c1) == A.interval(c2)
    assert not torch.equal(A.boot_weights(6, 200, 0), A.boot_weights(6, 200, 1))
    assert A.boot_weights(6, 200, 0).sum(1).eq(6).all()


def test_the_ceiling_recovers_a_planted_reliability():
    """Two independent estimates of one planted direction, each from its own noisy sample: c_E is their cosine, c_R a
    split-half cosine of the second; the ceiling predicts the cross cosine."""
    d, signal, noise = 1024, 1.0, 0.16  # each estimate's reliability about 2/3
    truth = torch.randn(d, generator=torch.Generator().manual_seed(5))
    truth = truth / truth.norm()

    def estimate(n, seed):
        p, ng, _, _, _ = synthetic(n_q=10, n_i=1, n_s=n, d=d, L=1, signal=signal, noise=noise, seed=seed,
                                   truth=truth)
        return p, ng

    e1, e2 = [p.double().mean(0)[0] - n.double().mean(0)[0] for p, n in (estimate(10, 1), estimate(10, 2))]
    c_e = float(A.cos_layers(e1[None], e2[None])[0])
    pr, nr = estimate(20, 3)
    half = [i for i in range(len(pr)) if i % 20 < 10]
    other = [i for i in range(len(pr)) if i % 20 >= 10]
    h0 = pr[half].double().mean(0)[0] - nr[half].double().mean(0)[0]
    h1 = pr[other].double().mean(0)[0] - nr[other].double().mean(0)[0]
    c_r = float(A.cos_layers(h0[None], h1[None])[0])
    r_full = pr.double().mean(0)[0] - nr.double().mean(0)[0]
    observed = float(A.cos_layers(e1[None], r_full[None])[0])
    predicted = A.ceiling(c_e, c_r)
    assert 0.2 < predicted < 0.95 and abs(observed - predicted) < 0.08
    # and the formula itself: two estimators of reliability rho each give cos = sqrt(rho_1 rho_2)
    assert math.isclose(A.ceiling(0.81, 0.9), math.sqrt(0.81) * math.sqrt(2 * 0.9 / 1.9))
    assert A.spearman_brown(0.5) == 2 / 3 and A.ceiling(-0.2, 0.9) == 0.0


def test_split_null_stays_near_random_while_label_flip_does_not():
    pos, neg, cells, _, truth = synthetic(signal=20.0, noise=1.0, d=256, L=1)
    ref = truth[None].double()
    split = A.null_level(A.split_null(pos, neg, cells, 50, 0), ref)[0]
    flip = A.null_level(A.label_flip_null(pos, neg, 50, 0), ref)[0]
    rand = A.random_level(ref, 1000, 0)[0]
    assert split < 3 * rand and split < 0.3
    assert flip > 0.5 and flip > 4 * split


def test_split_weights_balance_every_cell():
    cells = [(q, i) for q in range(3) for i in range(2) for _ in range(4)]
    W = A.split_weights(cells, 10, 0)
    assert W.shape == (10, len(cells))
    for p in range(10):
        a = W[p] > 0
        for c in set(cells):
            members = [u for u, x in enumerate(cells) if x == c]
            assert int(a[members].sum()) == 2
        assert torch.isclose(W[p].sum(), torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
    assert torch.equal(W, A.split_weights(cells, 10, 0))
