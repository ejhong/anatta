"""K1's measurement, from per-layer features to every number the card reads.

Mirrored from Balestriero & LeCun's prh/metrics.py (galilai-group/prh-proof at a2b9949fa907d4fbe562742d4b1ffa7205ab7427,
MIT, Copyright (c) 2026 Randall Balestriero and Yann LeCun; the licence is in third_party/prh_proof/LICENSE), line for
line where the graded number depends on it: mutual_knn, _prh_prep, prh_score, unbiased_cka, prh_subsample and
subsample_interval. The one change is prh_subsample's optional `best`, the layer pair prh_score has already found (it
would find the same one again).

Written here (the recorded diagnostics):
  layer_max     prh_score's maximum over features already prepared, ties to the highest indices as Python's max of
                (score, i, j) tuples gives them.
  huh_prep      Huh et al.'s own clipping, from the description in the brief and their paper (their repository,
                minyoungg/platonic-rep, has no licence: it was read, not copied): one threshold for all of a model's
                layers, the mean over samples of each sample's 0.95 quantile of |x| across all its layers' values (their
                inexact mode), then clamp and L2-normalize each layer.
  null_scores   a permutation null in the manner of Groger et al. (ICML 2026): the pairing shuffled, the same shuffle
                for every layer, mutual k-NN of every layer pair under each shuffle. Each layer's neighbour lists are
                found once; a shuffle only renumbers them.
  score_text    everything K1 records for one language model against the vision layers.
  batch_invariance  the smoke's batch check, by the rule registered before the smoke (its docstring).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------- mirrored from prh-proof prh/metrics.py at a2b9949 (MIT, see above)


def mutual_knn(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    """Mean fraction of shared cosine ``k``-nearest neighbours between paired rows (Huh et al., 2024).

    Fits nothing; chance is about ``k / N``. ``k`` is capped at ``N - 1``.
    """
    k = min(k, len(a) - 1)

    def knn_mask(x):
        s = (F.normalize(x, dim=1) @ F.normalize(x, dim=1).T).fill_diagonal_(-float("inf"))
        return torch.zeros_like(s).scatter_(1, s.topk(k, dim=1).indices, 1.0)

    return ((knn_mask(a) * knn_mask(b)).sum(1) / k).mean().item()


def unbiased_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """Linear CKA between paired rows with the unbiased HSIC estimator (Song et al., 2012), as in Huh et al. (2024).

    Fits nothing; invariant to rotations and isotropic scalings of either side, 1 when ``b`` is a rotation of ``a``.
    """

    def hsic(K, L):
        m = len(K)
        K, L = K.clone().fill_diagonal_(0), L.clone().fill_diagonal_(0)
        return ((K * L).sum() + K.sum() * L.sum() / ((m - 1) * (m - 2)) - 2 * K.sum(0) @ L.sum(1) / (m - 2)) / (m * (m - 3))

    K, L = a.double() @ a.double().T, b.double() @ b.double().T
    return (hsic(K, L) / (hsic(K, K) * hsic(L, L)).sqrt()).item()


def _prh_prep(f):
    """The PRH paper's preprocessing: elementwise clip at the 0.95 quantile of absolute values, L2-normalize."""
    stride = max(1, -(-f.numel() // 10_000_000))  # torch.quantile takes at most 2^24 values
    t = torch.quantile(f.abs().flatten().float()[::stride], 0.95)
    return F.normalize(f.float().clamp(-t, t), dim=-1)


def prh_score(
    layers_a: list[torch.Tensor], layers_b: list[torch.Tensor], k: int = 10, metric: str = "mknn"
) -> tuple[float, int, int]:
    """Alignment score of the PRH paper: max over layer pairs of mutual ``k``-NN after its preprocessing.

    Each layer's features are clipped elementwise at the 0.95 quantile of their absolute values and
    L2-normalized, as in the released code of Huh et al. (2024). ``metric="cka"`` scores each layer pair by
    :func:`unbiased_cka` instead.

    Returns:
        ``(score, layer_a, layer_b)`` of the best pair.
    """
    score = {"mknn": lambda a, b: mutual_knn(a, b, k), "cka": unbiased_cka}[metric]
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # thousands of layer pairs on 5,000 rows: CPU takes hours
    fa, fb = [_prh_prep(f.to(dev)) for f in layers_a], [_prh_prep(f.to(dev)) for f in layers_b]
    return max((score(a, b), i, j) for i, a in enumerate(fa) for j, b in enumerate(fb))


def prh_subsample(layers_a, layers_b, n: int = 200, size: int = 800, k: int = 10, seed: int = 0,
                  best: tuple[float, int, int] | None = None) -> torch.Tensor:
    """:func:`prh_score` at its best layer pair on ``n`` random subsets of ``size`` rows (without replacement).

    Resampling with replacement would duplicate rows, which share their neighbours and inflate mutual k-NN.
    Subsets depend only on ``seed`` and the number of rows, so scores of different models are paired.
    (K1: ``best``, when given, is prh_score's result, so the best pair isn't searched for twice.)
    """
    _, i, j = best if best is not None else prh_score(layers_a, layers_b, k)
    a, b = _prh_prep(layers_a[i]), _prh_prep(layers_b[j])
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(len(a), generator=g)[:size] for _ in range(n)]
    return torch.tensor([mutual_knn(a[r], b[r], k) for r in rows])


def subsample_interval(full: float, subsets: torch.Tensor, n: int, size: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation 95% interval for a statistic of ``n`` rows, from its values on random subsets.

    On subsets of ``size`` rows drawn without replacement, an average-like statistic varies around its ``n``-row
    value with variance proportional to ``1/size - 1/n``, while the ``n``-row value varies around the truth with
    variance proportional to ``1/n``. The subsets' spread is therefore rescaled by ``sqrt(size / (n - size))``.
    """
    if not 0 < size < n:
        raise ValueError(f"subsets of {size} rows cannot estimate the spread of a statistic of {n} rows")
    se = subsets.std().item() * (size / (n - size)) ** 0.5
    return full - z * se, full + z * se


# ------------------------------------------------------------------------------------------- written for K1


def layer_max(fa: list[torch.Tensor], fb: list[torch.Tensor], score) -> tuple[float, int, int]:
    """prh_score's maximum over layers already prepared: max of (score(a, b), i, j), ties to the highest indices."""
    return max((score(a, b), i, j) for i, a in enumerate(fa) for j, b in enumerate(fb))


def row_quantiles(x: torch.Tensor, q: float, max_numel: int = 1 << 24) -> torch.Tensor:
    """torch.quantile of each row of a 2-D tensor, taken over chunks of rows small enough for torch.quantile (a chunk
    holds at most max_numel values; each row's quantile is its own, so chunking changes nothing)."""
    per = max(1, max_numel // max(1, x.shape[1]))
    return torch.cat([torch.quantile(x[i:i + per], q, dim=1) for i in range(0, len(x), per)])


def huh_threshold(layers: list[torch.Tensor], q: float = 0.95) -> torch.Tensor:
    """Huh et al.'s clipping threshold (their inexact mode): each sample's q quantile of |x| over the values of all
    its layers, averaged over the samples. One threshold for all of a model's layers."""
    x = torch.cat([f.float().abs() for f in layers], dim=1)  # [N, sum of the layers' widths]
    return row_quantiles(x, q).mean()


def huh_prep(layers: list[torch.Tensor], q: float = 0.95) -> tuple[list[torch.Tensor], float]:
    """Every layer clamped at the model's one threshold (huh_threshold), then L2-normalized; and the threshold."""
    t = huh_threshold(layers, q)
    return [F.normalize(f.float().clamp(-t, t), dim=-1) for f in layers], t.item()


def knn_index(x: torch.Tensor, k: int) -> torch.Tensor:
    """Each row's k nearest neighbours as mutual_knn finds them (the same similarity, self excluded, topk), [N, k]."""
    s = (F.normalize(x, dim=1) @ F.normalize(x, dim=1).T).fill_diagonal_(-float("inf"))
    return s.topk(k, dim=1).indices


def boundary_ties(x: torch.Tensor, k: int) -> int:
    """Rows whose k-th and (k+1)-th similarities are equal: where topk's choice among equals decides a neighbour. With
    none, renumbering a layer's neighbour lists (null_scores) gives exactly what mutual_knn finds on shuffled rows."""
    s = (F.normalize(x, dim=1) @ F.normalize(x, dim=1).T).fill_diagonal_(-float("inf"))
    v = s.topk(min(k + 1, len(x) - 1), dim=1).values
    return int((v[:, k - 1] == v[:, k]).sum()) if v.shape[1] > k else 0


def permutations(n_rows: int, count: int, seed: int) -> list[torch.Tensor]:
    """count seeded shuffles of the pairing: under perm p, image row i is paired with text row p[i]."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randperm(n_rows, generator=g) for _ in range(count)]


def null_scores(fa: list[torch.Tensor], fb: list[torch.Tensor], perms: list[torch.Tensor], k: int = 10) -> torch.Tensor:
    """Mutual k-NN of every (a layer, b layer) pair under each re-pairing, [len(perms), len(fa), len(fb)].

    Under perm p the b side's rows are b[p] for every layer at once: row i's neighbours there are inv[m] for m among
    b's neighbours of row p[i] (inv the inverse permutation). Each layer's neighbour lists are found once (knn_index);
    the a side's become a dense mask, and a re-pairing only gathers from it. The per-row counts are mutual_knn's.
    """
    n = len(fa[0])
    k = min(k, n - 1)
    dev = fa[0].device
    ia = torch.stack([knn_index(a, k) for a in fa])  # [A, N, k]
    ib = torch.stack([knn_index(b, k) for b in fb])  # [B, N, k]
    mask = torch.zeros(len(fa), n, n, dtype=torch.bool, device=dev).scatter_(2, ia, True)  # [A, N, N]
    rows = torch.arange(n, device=dev)
    out = []
    for p in perms:
        p = p.to(dev)
        inv = torch.empty_like(p)
        inv[p] = rows
        nb = inv[ib[:, p]]  # [B, N, k]: b's neighbours of each re-paired row, in a's numbering
        shared = mask[:, rows[None, :, None], nb]  # [A, B, N, k]
        out.append((shared.sum(-1).float() / k).mean(-1))
    return torch.stack(out)


def null_naive(fa, fb, perms, k: int = 10) -> torch.Tensor:
    """null_scores the slow way, for the tests: mutual_knn on every layer pair with b's rows shuffled."""
    return torch.tensor([[[mutual_knn(a, b[p.to(b.device)], k) for b in fb] for a in fa] for p in perms])


def finite(layers: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(f).all()) for f in layers)


def batch_invariance(a: list[torch.Tensor], again: list[torch.Tensor], one: list[torch.Tensor], dtype: str,
                     rule: dict, raw_tol_abs: float) -> dict:
    """The smoke's batch check on one model's features, by the rule registered before the smoke (Opus, 1 Nov). a and
    again are two passes at the run's batch size, one is a pass at batch 1; each is a list of [N, d] hidden-state
    means. It passes iff every part does:
      embed      hidden state 0, the embedding output's masked mean (no attention, no MLP, only which tokens are in
                 the mean), differs by at most rule["embed_max_abs"] between the batch sizes: the direct test of
                 masking, padding and tokenization, where kernel noise is about 1e-7 and a leaked pad about 1e-3;
      prepped    after _prh_prep, the only form the score sees, no layer differs by more than
                 rule["prepped_max_abs"];
      identical  the two passes at the run's batch size are bitwise equal;
      finite     no pass holds a NaN or an infinity;
      float32    the model ran in float32.
    The raw max abs difference, and its ratio to the layer's largest value, are recorded beside the parts and not
    read: a few massive activations (of order 1e3, where one float32 step is about 1e-4) differ between kernels on
    rounding alone, and they dominate the ratio."""
    raw = [(x - y).abs().max().item() for x, y in zip(a, one)]
    rel = [((x - y).abs().max() / x.abs().max().clamp_min(1e-30)).item() for x, y in zip(a, one)]
    prepped = [(_prh_prep(x) - _prh_prep(y)).abs().max().item() for x, y in zip(a, one)]
    parts = {
        "embed": {"passed": raw[0] <= rule["embed_max_abs"], "max_abs": raw[0], "tol": rule["embed_max_abs"]},
        "prepped": {"passed": max(prepped) <= rule["prepped_max_abs"], "max_abs": max(prepped),
                    "tol": rule["prepped_max_abs"], "by_layer": prepped},
        "identical": {"passed": all(torch.equal(x, y) for x, y in zip(a, again)),
                      "max_abs": max((x - y).abs().max().item() for x, y in zip(a, again))},
        "finite": {"passed": finite(a) and finite(again) and finite(one)},
        "float32": {"passed": dtype == "float32", "dtype": dtype},
    }
    failed = [k for k, part in parts.items() if not part["passed"]]
    return {"passed": not failed, "failed": failed, "parts": parts,
            "recorded": {"max_abs": max(raw), "by_layer": raw, "max_rel": max(rel), "raw_tol_abs": raw_tol_abs,
                         "within_raw_tol": max(raw) <= raw_tol_abs}}


def r6(x: float) -> float:
    return round(float(x), 6)


def score_text(fa: list[torch.Tensor], fb: list[torch.Tensor], s: dict, ks_extra=()) -> tuple[dict, torch.Tensor]:
    """Everything K1 records for one language model (fb) against the vision layers (fa), both lists of [N, d] float32
    CPU tensors; s holds k, quantile, null {permutations, seed, percentile} and subsample {n, size, seed, z}.

    Graded: mutual k-NN, prh_score (the paper's code). Recorded: unbiased CKA with the same layer maximum; the score
    under Huh et al.'s clipping; the score of every layer pair (the identity re-pairing of null_scores); the
    permutation null's percentile tau, the calibrated score max((s - tau) / (1 - tau), 0), and whether the identity
    re-pairing reproduces the score; prh_subsample's scores on the CPU, as the paper's code runs them, and their
    interval. ks_extra: more k, scored under both clippings (the OpenLLaMA anchor's k = 1). Returns (the record, the
    subsets' scores, for the differences between models)."""
    dev, k, n = device(), s["k"], len(fa[0])
    best = prh_score(fa, fb, k)
    cka = prh_score(fa, fb, k, metric="cka")
    rec = {"rows": n, "k": k, "mknn": {"score": best[0], "layers": [best[1], best[2]]},
           "cka": {"score": cka[0], "layers": [cka[1], cka[2]]}}
    pa, pb = [_prh_prep(f.to(dev)) for f in fa], [_prh_prep(f.to(dev)) for f in fb]
    ha, t_a = huh_prep([f.to(dev) for f in fa], s["quantile"])
    hb, t_b = huh_prep([f.to(dev) for f in fb], s["quantile"])
    h = layer_max(ha, hb, lambda a, b: mutual_knn(a, b, k))
    rec["huh"] = {"score": h[0], "layers": [h[1], h[2]], "threshold_a": t_a, "threshold_b": t_b}
    nl = s["null"]
    perms = permutations(n, nl["permutations"], nl["seed"])
    scores = null_scores(pa, pb, [torch.arange(n)] + perms, k).cpu()
    identity, maxima = scores[0], scores[1:].amax(dim=(1, 2))
    tau = torch.quantile(maxima.double(), nl["percentile"] / 100).item()
    top = identity.max().item()
    rec["by_layer"] = [[r6(x) for x in row] for row in identity.tolist()]
    rec["null"] = {"permutations": nl["permutations"], "seed": nl["seed"], "percentile": nl["percentile"],
                   "tau": tau, "calibrated": max((best[0] - tau) / (1 - tau), 0.0),
                   "maxima": {"mean": maxima.mean().item(), "min": maxima.min().item(), "max": maxima.max().item()},
                   "identity_max": top, "identity_matches": abs(top - best[0]) <= 1e-6,
                   "boundary_ties": {"a": sum(boundary_ties(x, k) for x in pa), "b": sum(boundary_ties(x, k) for x in pb)}}
    rec["extra_k"] = {}
    for kk in ks_extra:
        p = prh_score(fa, fb, kk)
        hk = layer_max(ha, hb, lambda a, b: mutual_knn(a, b, kk))
        rec["extra_k"][str(kk)] = {"paper": {"score": p[0], "layers": [p[1], p[2]]},
                                   "huh": {"score": hk[0], "layers": [hk[1], hk[2]]}}
    del pa, pb, ha, hb
    if dev == "cuda":
        torch.cuda.empty_cache()
    sub = s["subsample"]
    subsets = prh_subsample(fa, fb, sub["n"], sub["size"], k, sub["seed"], best=best)
    lo, hi = subsample_interval(best[0], subsets, n, sub["size"], sub["z"])
    rec["subsample"] = {"n": sub["n"], "size": sub["size"], "seed": sub["seed"], "std": subsets.std().item(),
                        "mean": subsets.mean().item(), "interval": [lo, hi]}
    return rec, subsets


def differences(full: dict[str, float], subsets: dict[str, torch.Tensor], ref: str, n: int, size: int,
                z: float = 1.96) -> dict:
    """Each model's score minus the reference model's (the 7B), with the paper's interval from the paired subsets
    (evaluate.py's wit computes first - pair from subsets[first] - x; here model - reference)."""
    out = {}
    for key, x in subsets.items():
        if key == ref or ref not in subsets:
            continue
        d = full[key] - full[ref]
        lo, hi = subsample_interval(d, x - subsets[ref], n, size, z)
        out[key] = {"diff": d, "interval": [lo, hi]}
    return out
