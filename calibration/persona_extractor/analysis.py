"""Calibration check 4: the analysis (runs on Modal's CPU in the lab's image, next to the vectors; returns numbers only).

Every cosine is by layer, with probe's block b against the reference's hidden_states index b + 1 (the graded pair is
block 19 with R[20]); index 28 and block 27 are never compared.

  Sums        per-question sums of per-response means, by sign, in float64: a direction is the mean of the pos
              responses minus the mean of the neg ones, so a resample of the questions rebuilds it exactly from them.
  bootstrap   2,000 resamples of the 20 questions (seed 0), one index draw shared by every comparison; each resample
              rebuilds both directions from the same questions. R keeps only its final vector, so R-lab32's
              per-response means stand in for R in the resamples (E-stack checks that they can); the point estimate is
              against R itself.
  floors      c_E: cos(E-sys, E-sys'), and cos(E, E') for C4-lab. c_R: R-lab32 rebuilt from samples 0-4 and from
              samples 5-9 of each (question, instruction) cell; Spearman-Brown gives the full reference's reliability
              2c/(1 + c); the even-odd question split is also reported. The ceiling is sqrt(c_E) x sqrt(2c_R/(1 + c_R)).
  split null  200 partitions (seed 0) of the responses into two halves, each pair (row k of pos and of neg) kept
              whole, each (question, instruction) cell split in half; the null is mean(half A) - mean(half B), so the
              trait signal cancels and the uneven spread of the activations stays. Its 95th-percentile |cos| with R is
              the chance level. label_flip_null is the foil the registration rejects, kept for the test that shows
              why.
  random      1,000 random unit vectors (seed 0) against R.
Without the recorder, C4 and C4-lab get the seed spread (the two seeds' cosines) in place of a bootstrap, and the
numbers that need the recorded responses or means are absent. Without the verbatim script (the second image declined,
or R doesn't fit), R-lab32 is the reference and E-stack is not computed.

The settings come from the analysis plan (its analysis block), not the run's plan: this file, card.py and the blocks
only they read have their own id, so a fix here reruns the analysis from the stored data and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import traits as T

# ------------------------------------------------------------------------------------------------ the arithmetic


def cos_layers(a, b):
    """Cosine of each row of a [L, d] with the same row of b, in float64. Returns [L]."""

    a, b = a.double(), b.double()
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)


def spearman_brown(c: float) -> float:
    return 2 * c / (1 + c)


def ceiling(c_e: float, c_r: float) -> float:
    """The cosine two estimators of one true direction would show, given their sampling noise: sqrt(c_E) times the
    square root of the full reference's Spearman-Brown reliability. A floor at or under 0 gives 0."""
    return (max(c_e, 0.0) ** 0.5) * (max(spearman_brown(c_r), 0.0) ** 0.5) if c_r > -1 else 0.0


def quantile(x, q: float) -> float:
    import torch

    return float(torch.quantile(x.double().flatten(), q))


class Sums:
    """Per-question sums of per-response means: S [Q, 2, L, d] (sign 0 pos, 1 neg) and counts N [Q, 2], float64, with
    the layer labels of L (probe blocks, or hidden_states indices for the reference)."""

    def __init__(self, S, N, labels: list[int]):
        self.S, self.N, self.labels = S, N, list(labels)

    @classmethod
    def build(cls, pos, neg, q_pos: list[int], q_neg: list[int], n_q: int, labels: list[int]):
        import torch

        S = torch.zeros(n_q, 2, pos.shape[1], pos.shape[2], dtype=torch.float64)
        N = torch.zeros(n_q, 2, dtype=torch.float64)
        for j, (X, qs) in enumerate(((pos, q_pos), (neg, q_neg))):
            idx = torch.tensor(qs, dtype=torch.long)
            S[:, j].index_add_(0, idx, X.double())
            N[:, j].index_add_(0, idx, torch.ones(len(qs), dtype=torch.float64))
        return cls(S, N, labels)

    def select(self, labels: list[int]) -> Sums:
        j = [self.labels.index(x) for x in labels]
        return Sums(self.S[:, :, j], self.N, labels)

    def full(self):
        """The direction from every question: [L, d]."""
        return self.S[:, 0].sum(0) / self.N[:, 0].sum() - self.S[:, 1].sum(0) / self.N[:, 1].sum()

    def resampled(self, W):
        """Directions from question weights W [B, Q] (each resample's count of every question): [B, L, d]."""
        Q, L, d = self.S.shape[0], self.S.shape[2], self.S.shape[3]
        pos = (W @ self.S[:, 0].reshape(Q, L * d)) / (W @ self.N[:, 0]).unsqueeze(1)
        neg = (W @ self.S[:, 1].reshape(Q, L * d)) / (W @ self.N[:, 1]).unsqueeze(1)
        return (pos - neg).reshape(-1, L, d)


def boot_weights(n_q: int, resamples: int, seed: int):
    """One draw of `resamples` resamples of n_q questions with replacement (seed), as counts: [B, n_q] float64."""
    import torch

    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n_q, (resamples, n_q), generator=g)
    W = torch.zeros(resamples, n_q, dtype=torch.float64)
    W.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
    return W


def boot_cos(A: Sums, B: Sums, W, chunk: int = 100):
    """cos(A's direction, B's direction) at each layer pair (A and B already selected to matching layers) in every
    resample: [resamples, L]. Both directions are rebuilt from the same resampled questions."""
    import torch

    out = []
    for i in range(0, W.shape[0], chunk):
        w = W[i:i + chunk]
        out.append(cos_layers(A.resampled(w), B.resampled(w)))
    return torch.cat(out)


def interval(samples) -> list[list[float]]:
    """The 2.5th and 97.5th percentiles of each column: [[lo, hi] per layer]."""
    import torch

    lo = torch.quantile(samples, 0.025, dim=0)
    hi = torch.quantile(samples, 0.975, dim=0)
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lo, hi)]


def split_weights(cells: list, partitions: int, seed: int):
    """The split null's partitions: for each, every cell's units shuffled and the first half (rounded down) put in
    half A, the rest in B. cells[u] is unit u's cell. Returns W [P, U] with +1/|A| on A's units and -1/|B| on B's, so
    W @ (each unit's summed pos and neg means) / 2 is mean(half A) - mean(half B) over the responses."""
    import torch

    g = torch.Generator().manual_seed(seed)
    groups: dict = {}
    for u, c in enumerate(cells):
        groups.setdefault(c, []).append(u)
    keys = sorted(groups)
    W = torch.zeros(partitions, len(cells), dtype=torch.float64)
    for p in range(partitions):
        a = torch.zeros(len(cells), dtype=torch.bool)
        for c in keys:
            units = groups[c]
            perm = torch.randperm(len(units), generator=g).tolist()
            for k in perm[:len(units) // 2]:
                a[units[k]] = True
        na, nb = int(a.sum()), int((~a).sum())
        W[p, a] = 1.0 / max(na, 1)
        W[p, ~a] = -1.0 / max(nb, 1)
    return W


def split_null(pos, neg, cells: list, partitions: int, seed: int):
    """Null vectors from pairs (row u of pos and of neg, both [U, L, d]) split by split_weights: [P, L, d]. The trait
    signal cancels exactly, since every unit carries one pos and one neg response."""
    U, L, d = pos.shape
    W = split_weights(cells, partitions, seed)
    Y = (pos.double() + neg.double()).reshape(U, L * d) / 2
    return (W @ Y).reshape(partitions, L, d)


def label_flip_null(pos, neg, partitions: int, seed: int):
    """The foil: every response's pos/neg label flipped at random (seed), the direction rebuilt. Its mean keeps the
    trait direction, shrunk but not rotated, so its cosine with R grows with the trait's strength. [P, L, d]."""
    import torch

    g = torch.Generator().manual_seed(seed)
    X = torch.cat([pos, neg]).double()
    n, L, d = X.shape
    s = torch.cat([torch.ones(len(pos)), -torch.ones(len(neg))]).double()
    out = []
    for _ in range(partitions):
        f = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).double() * s
        npos, nneg = int((f > 0).sum()), int((f < 0).sum())
        w = torch.where(f > 0, 1.0 / max(npos, 1), -1.0 / max(nneg, 1)).double()
        out.append((w @ X.reshape(n, L * d)).reshape(L, d))
    return torch.stack(out)


def null_level(nulls, ref, q: float = 0.95) -> list[float]:
    """The q-quantile of |cos(null, ref)| at each layer: nulls [P, L, d], ref [L, d]."""
    c = cos_layers(nulls, ref.unsqueeze(0).expand_as(nulls)).abs()
    import torch

    return [round(float(v), 6) for v in torch.quantile(c, q, dim=0)]


def random_level(ref, n: int, seed: int, q: float = 0.95) -> list[float]:
    """The q-quantile of |cos| between n random unit vectors (seed) and each row of ref [L, d]."""
    import torch

    g = torch.Generator().manual_seed(seed)
    U = torch.randn(n, ref.shape[-1], generator=g, dtype=torch.float64)
    U = U / U.norm(dim=1, keepdim=True)
    c = (U @ (ref.double() / ref.double().norm(dim=1, keepdim=True)).T).abs()
    return [round(float(v), 6) for v in torch.quantile(c, q, dim=0)]


def profile(values, labels: list[int]) -> dict:
    return {str(b): round(float(v), 6) for b, v in zip(labels, values)}


def best(prof: dict) -> dict:
    k = max(prof, key=prof.get)
    return {"layer": int(k), "cos": prof[k]}


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if n else float("nan")


# ------------------------------------------------------------------------------------------------ the run's files


class Run:
    """The stored vectors and means of one run, read from the volume as they are needed."""

    def __init__(self, plan: dict, out):
        self.plan, self.out = plan, Path(out)
        lay = plan["layout"]
        self.n_q, self.n_i, self.n_s = lay["questions"], lay["instructions"], plan["G"]["samples"]
        r = plan["readings"]["blocks"]
        self.blocks = list(range(r[0], r[1] + 1))
        self.shas: dict = {}

    def load(self, rel: str, vector: bool = False):
        """A stored file; a vector's sha256 is recorded for the card (the per-response means' are not)."""
        import torch

        p = T.require(self.out / rel)
        if vector:
            self.shas[rel] = T.sha256_file(p)
        return torch.load(p, weights_only=False)

    def exists(self, rel: str) -> bool:
        return (self.out / rel).exists()

    def verbatim(self, t: str, kind: str = "response_avg", judged: bool = False):
        rel = f"{'judged/' if judged else ''}verbatim/{t}/{t}_{kind}_diff.pt"
        return self.load(rel, vector=True).double() if self.exists(rel) else None

    def rlab32(self, t: str, kind: str = "response_avg"):
        return self.load(f"rlab32/{t}/{t}_{kind}_diff.pt", vector=True).double()

    def rlab32_means(self, t: str):
        """R-lab32's per-response means of each sign: rows and [n, layers + 1, d]."""
        return {s: self.load(f"rlab32/{t}/means_{s}.pt") for s in T.SIGNS}

    def qi(self, rows: list[int]) -> list[int]:
        return [T.cell_of(k, self.n_i, self.n_s)[0] for k in rows]

    def reading(self, arm: str, variant: str, t: str):
        """A stored reading of G or G': ({sign: rows}, {sign: [n, blocks, d]})."""
        from readings import load_means

        n_s = self.plan[arm]["samples"]
        rows, means = {}, {}
        for s in T.SIGNS:
            r, blocks, m = load_means(self.out, arm, variant, t, s)
            rows[s], means[s] = r, m
        if rows["pos"] != rows["neg"]:
            raise T.CheckError("missing")
        return rows, means, n_s

    def reading_sums(self, arm: str, variant: str, t: str) -> Sums:
        rows, means, n_s = self.reading(arm, variant, t)
        q = [T.cell_of(k, self.n_i, n_s)[0] for k in rows["pos"]]
        return Sums.build(means["pos"], means["neg"], q, q, self.n_q, self.blocks)

    def arm(self, arm: str, t: str):
        """A probe.extract arm's Direction (unit vectors at its blocks), or None if it didn't run."""
        from anatta.probe import Direction

        rel = f"probe/{arm}/{t}/direction.pt"
        if not self.exists(rel):
            return None
        self.shas[rel] = T.sha256_file(self.out / rel)
        return Direction.load(self.out / rel)

    def arm_sums(self, arm: str, t: str) -> Sums | None:
        """The recorded per-response means of an arm, as Sums over the trait's questions (the same list G uses), or
        None without the recorder."""
        rel = f"probe/{arm}/{t}/means.pt"
        if not self.exists(rel):
            return None
        rec = self.load(rel)
        sg = rec["signs"]
        pos = rec["means"][[i for i, s in enumerate(sg) if s > 0]]
        neg = rec["means"][[i for i, s in enumerate(sg) if s < 0]]
        qp = [q for q, s in zip(rec["qi"], sg) if s > 0]
        qn = [q for q, s in zip(rec["qi"], sg) if s < 0]
        return Sums.build(pos, neg, qp, qn, self.n_q, rec["layers"])

    def arm_pairs(self, arm: str, t: str):
        """The recorded means as pairs (the pos and neg response of each (question, pair) job): (pos, neg)."""
        rec = self.load(f"probe/{arm}/{t}/means.pt")
        sg, qi, pi = rec["signs"], rec["qi"], rec["pi"]
        pos = {(q, p): i for i, (q, p, s) in enumerate(zip(qi, pi, sg)) if s > 0}
        neg = {(q, p): i for i, (q, p, s) in enumerate(zip(qi, pi, sg)) if s < 0}
        keys = sorted(set(pos) & set(neg))
        return rec["means"][[pos[k] for k in keys]], rec["means"][[neg[k] for k in keys]], rec["layers"]


def vec(direction, blocks: list[int]):
    import torch

    return torch.stack([direction.vectors[b] for b in blocks]).double()


# ------------------------------------------------------------------------------------------------ the analysis


def compare(name: str, a, b, labels: list[int], res: dict, t: str, boot=None) -> dict:
    """One comparison by layer: a and b are [L, d] at matching rows; boot, if given, the [resamples, L] cosines."""
    prof = profile(cos_layers(a, b), labels)
    entry = {"by_layer": prof, "best": best(prof)}
    if boot is not None:
        ci = interval(boot)
        entry["ci"] = {str(x): c for x, c in zip(labels, ci)}
    res.setdefault(name, {})[t] = entry
    return entry


def run(plan: dict, aplan: dict, out, analysis_id: str, log=print, stored_dir=None,
        guard: T.Guard | None = None) -> dict:
    """Every number the card reports, from the run's files (the run's plan says what they are; the analysis plan holds
    the analysis settings). Returns numbers only; writes cheap/<analysis id>/reference.pt (the cheap tier's reference
    values, with the graded block) to the volume."""

    from readings import cheap_reference, save_tensor

    a = aplan["analysis"]
    rr = Run(plan, out)
    traits, blocks = plan["traits"], rr.blocks
    b19 = a["block"]
    band = list(range(a["band"][0], a["band"][1] + 1))
    W = boot_weights(rr.n_q, a["boot"]["resamples"], a["boot"]["seed"])
    verbatim = all(rr.verbatim(t) is not None for t in traits)
    res: dict = {"reference": "R" if verbatim else "R-lab32", "verbatim": verbatim, "block": b19,
                 "index": T.hs_index(b19), "cos": {}, "floors": {}, "ceilings": {}, "nulls": {}, "norms": {},
                 "matrices": {}, "spread": {}, "other_outputs": {}, "ledger": {}, "recorder": plan["switches"]["recorder"]}
    idx = [T.hs_index(b) for b in blocks]
    R, RL, RLsums, P1d = {}, {}, {}, {}
    cheap = {"blocks": blocks, "block": b19, "rows": {}, "p1": {}, "rlab32": {}}
    for ti, t in enumerate(traits):
        if guard is not None:
            guard.row = ti
        log(f"analysis {t}")
        RL[t] = rr.rlab32(t)
        R[t] = rr.verbatim(t) if verbatim else RL[t]
        rm = rr.rlab32_means(t)
        q = rr.qi(rm["pos"]["rows"])
        RLsums[t] = Sums.build(rm["pos"]["means"], rm["neg"]["means"], q, q, rr.n_q,
                               list(range(rm["pos"]["means"].shape[1])))
        # the reference's floor: samples 0-4 against 5-9 of each cell, and even against odd questions
        si = [T.cell_of(k, rr.n_i, rr.n_s)[2] for k in rm["pos"]["rows"]]
        half = rr.n_s // 2

        def part(keep, rm=rm):
            ix = [i for i, k in enumerate(keep) if k]
            return rm["pos"]["means"][ix].double().mean(0) - rm["neg"]["means"][ix].double().mean(0)

        lo, hi = part([s < half for s in si]), part([s >= half for s in si])
        ev, od = part([x % 2 == 0 for x in q]), part([x % 2 == 1 for x in q])
        res["floors"].setdefault("c_R", {})[t] = profile(cos_layers(lo[idx], hi[idx]), blocks)
        res["floors"].setdefault("c_R_questions", {})[t] = profile(cos_layers(ev[idx], od[idx]), blocks)
        # R's own numbers
        res["norms"].setdefault("R", {})[t] = {str(i): round(float(R[t][i].norm()), 4) for i in range(R[t].shape[0])}
        if verbatim:
            # the indices of the blocks read (1-27 on Qwen): R-lab32 against R row by row
            res["cos"].setdefault("E-stack", {})[t] = {"by_layer": profile(cos_layers(RL[t][idx], R[t][idx]), idx)}
            for kind in ("prompt_avg", "prompt_last"):
                o = rr.verbatim(t, kind)
                i19 = T.hs_index(b19)  # "at index 20"
                res["other_outputs"].setdefault(kind, {})[t] = round(float(cos_layers(o[[i19]], R[t][[i19]])[0]), 6)
        rsel = RLsums[t].select(idx)
        # G's readings: P1 (C4-code), P1-trim, P1-128 and P0
        S = {v: rr.reading_sums("G", v, t) for v in ("P1", "P1trim", "P1cap", "P0")}
        D = {v: S[v].full() for v in S}
        P1d[t] = D["P1"]
        compare("C4-code", D["P1"], R[t][idx], blocks, res["cos"], t, boot_cos(S["P1"], rsel, W))
        compare("C4-code-vs-Rlab32", D["P1"], RL[t][idx], blocks, res["cos"], t)
        compare("E-dtype", D["P1trim"], RL[t][idx], blocks, res["cos"], t, boot_cos(S["P1trim"], rsel, W))
        compare("E-trail", D["P1"], D["P1trim"], blocks, res["cos"], t, boot_cos(S["P1"], S["P1trim"], W))
        compare("E-len", D["P1cap"], D["P1trim"], blocks, res["cos"], t, boot_cos(S["P1cap"], S["P1trim"], W))
        compare("E-ctx", D["P0"], D["P1"], blocks, res["cos"], t, boot_cos(S["P0"], S["P1"], W))
        res["norms"].setdefault("P1trim_over_Rlab32", {})[t] = profile(
            D["P1trim"].norm(dim=-1) / RL[t][idx].norm(dim=-1).clamp_min(1e-30), blocks)
        # the split null on P1's means of G, and the random directions
        rows, means, _ = rr.reading("G", "P1", t)
        cells = [T.cell_of(k, rr.n_i, rr.n_s)[:2] for k in rows["pos"]]
        nulls = split_null(means["pos"], means["neg"], cells, a["split_null"]["partitions"], a["split_null"]["seed"])
        pq = a["split_null"]["percentile"] / 100
        res["nulls"].setdefault("split_P1", {})[t] = profile(null_level(nulls, R[t][idx], pq), blocks)
        res["nulls"].setdefault("random", {})[t] = profile(random_level(R[t][idx], a["random"]["n"],
                                                                        a["random"]["seed"],
                                                                        a["random"]["percentile"] / 100), blocks)
        # the cheap tier's subset: the first kept sample of every (question, instruction) cell, per sign
        first = {}
        for k in rows["pos"]:
            first.setdefault(T.cell_of(k, rr.n_i, rr.n_s)[:2], k)
        sub = sorted(first.values())
        ix = [rows["pos"].index(k) for k in sub]
        cheap["rows"][t] = sub
        cheap["p1"][t] = (means["pos"][ix].double().mean(0) - means["neg"][ix].double().mean(0)).float()
        jx = [rm["pos"]["rows"].index(k) for k in sub]
        cheap["rlab32"][t] = (rm["pos"]["means"][jx, T.hs_index(b19)].double().mean(0)
                              - rm["neg"]["means"][jx, T.hs_index(b19)].double().mean(0)).float()
        del rows, means, nulls, S
        # G': P1 against R (E-samp), against P1-128, and P0 against R
        if rr.exists(f"means/Gprime/P1/{t}"):
            Sg = {v: rr.reading_sums("Gprime", v, t) for v in ("P1", "P0")}
            Sc = rr.reading_sums("G", "P1cap", t)
            compare("E-samp", Sg["P1"].full(), R[t][idx], blocks, res["cos"], t, boot_cos(Sg["P1"], rsel, W))
            compare("Gprime-vs-P1cap", Sg["P1"].full(), D["P1cap"], blocks, res["cos"], t,
                    boot_cos(Sg["P1"], Sc, W))
            compare("Gprime-P0-vs-R", Sg["P0"].full(), R[t][idx], blocks, res["cos"], t, boot_cos(Sg["P0"], rsel, W))
        del D
        # the probe.extract arms
        E = {arm: rr.arm(arm, t) for arm in plan["arms"]}
        for arm, d in E.items():
            if d is None:
                continue
            layers = sorted(d.vectors)
            ri = [T.hs_index(b) for b in layers]
            sums = rr.arm_sums(arm, t)
            bt = boot_cos(sums, RLsums[t].select(ri), W) if sums is not None else None
            compare(f"{arm}-vs-R", vec(d, layers), R[t][ri], layers, res["cos"], t, bt)
        for x, y in (("E-sys", "E-sys2"), ("E", "E2")):
            if E.get(x) is not None and E.get(y) is not None:
                layers = sorted(E[x].vectors)
                res["floors"].setdefault(f"c_E:{x}|{y}", {})[t] = profile(
                    cos_layers(vec(E[x], layers), vec(E[y], layers)), layers)
                c0 = res["cos"][f"{x}-vs-R"][t]["by_layer"]
                c1 = res["cos"][f"{y}-vs-R"][t]["by_layer"]
                res["spread"].setdefault(x, {})[t] = {k: sorted([c0[k], c1[k]]) for k in c0}
        if E.get("E-sys") is not None and rr.exists(f"probe/E-sys/{t}/means.pt"):
            pos, neg, layers = rr.arm_pairs("E-sys", t)
            nulls = split_null(pos, neg, [0] * len(pos), a["split_null"]["partitions"], a["split_null"]["seed"])
            # Ruled (NOTES, 12:20 UTC, ruling 2): one E-sys response per cell and sign, so the null splits pairs 50/50.
            res["nulls"].setdefault("split_Esys", {})[t] = profile(
                null_level(nulls, R[t][[T.hs_index(b) for b in layers]], pq), layers)
        # the ceilings beside the C4-type cosines
        for x, y, name in (("E-sys", "E-sys2", "C4"), ("E", "E2", "C4-lab")):
            fe = res["floors"].get(f"c_E:{x}|{y}", {}).get(t)
            if fe is None:
                continue
            fr = res["floors"]["c_R"][t]
            ceil = {k: round(ceiling(fe[k], fr[k]), 6) for k in fe}
            cx = res["cos"][f"{x}-vs-R"][t]["by_layer"]
            res["ceilings"].setdefault(name, {})[t] = {"by_layer": ceil, "ratio": {
                k: (round(cx[k] / ceil[k], 6) if ceil[k] > 0 else None) for k in ceil}}
        if rr.exists(f"means/Gprime/P1/{t}"):
            # Ruled (NOTES, 12:20 UTC, ruling 3): this E-samp ceiling, from G''s own floor, is reported only.
            rows_g, means_g, n_g = rr.reading("Gprime", "P1", t)
            sg = [T.cell_of(k, rr.n_i, n_g)[2] for k in rows_g["pos"]]
            ix0 = [i for i, s in enumerate(sg) if s < n_g // 2]
            ix1 = [i for i, s in enumerate(sg) if s >= n_g // 2]
            h0 = means_g["pos"][ix0].double().mean(0) - means_g["neg"][ix0].double().mean(0)
            h1 = means_g["pos"][ix1].double().mean(0) - means_g["neg"][ix1].double().mean(0)
            fg = profile(cos_layers(h0, h1), blocks)
            res["floors"].setdefault("c_Gprime", {})[t] = fg
            fr = res["floors"]["c_R"][t]
            res["ceilings"].setdefault("E-samp", {})[t] = {"by_layer": {
                k: round(ceiling(spearman_brown(fg[k]), fr[k]), 6) for k in fg}}
            del rows_g, means_g
        # E-band: the median over the band of cos(E-sys at b, R at b + 1)
        if "E-sys-vs-R" in res["cos"] and t in res["cos"]["E-sys-vs-R"]:
            prof = res["cos"]["E-sys-vs-R"][t]["by_layer"]
            res.setdefault("band_median", {})[t] = round(median([prof[str(b)] for b in band if str(b) in prof]), 6)
        # the judged arm's R, if it ran
        rj = rr.verbatim(t, judged=True)
        if rj is not None:
            i19 = T.hs_index(b19)
            res.setdefault("judged", {})[t] = round(float(cos_layers(rj[[i19]], R[t][[i19]])[0]), 6)
    # cross-trait matrices at block 19 (index 20), specificity, and Figure 20's cells
    i19 = T.hs_index(b19)

    def matrix(vs: dict) -> dict:
        return {f"{x}|{y}": round(float(cos_layers(vs[x][None], vs[y][None])[0]), 6)
                for i, x in enumerate(traits) for y in traits[i + 1:] if x in vs and y in vs}

    res["matrices"]["R"] = matrix({t: R[t][i19] for t in traits})
    res["matrices"]["R-lab32"] = matrix({t: RL[t][i19] for t in traits})
    res["matrices"]["P1"] = matrix({t: P1d[t][blocks.index(b19)] for t in traits})
    for arm in ("E-sys", "E"):
        vs = {}
        for t in traits:
            d = rr.arm(arm, t)
            if d is not None and b19 in d.vectors:
                vs[t] = d.vectors[b19].double()
        res["matrices"][arm] = matrix(vs)
        if arm == "E-sys":
            res["specificity"] = {f"{x}|{y}": round(float(cos_layers(vs[x][None], R[y][i19][None])[0]), 6)
                                  for x in vs for y in traits}
    if stored_dir is not None:
        lq = plan["mprime"]["ledger_qwen"]
        from readings import check_stored, stored_direction

        check_stored(stored_dir, plan.get("stored_sha256_qwen"))
        sd = stored_direction(stored_dir, f"{lq['model_slug']}/{lq['trait']}.pt")
        if sd is not None and plan.get("stored_sha256_qwen") and lq["against"] in R:
            layers = sorted(sd.vectors)
            res["ledger"]["stored_qwen_vs_R"] = profile(
                cos_layers(vec(sd, layers), R[lq["against"]][[T.hs_index(b) for b in layers]]), layers)
    save_tensor(cheap_reference(out, analysis_id), cheap)
    res["vectors_sha256"] = dict(sorted(rr.shas.items()))
    return res
