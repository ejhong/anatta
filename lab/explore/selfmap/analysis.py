"""The self map's measures (Stage 1 pilot), on activations already pulled from Modal. CPU only. Claude (Opus 5.5, an
AI model) wrote this file, 2 Nov 2026. Private (lab/explore). Explore speed: every number is a sighting.

Rows are the dev prompts (prompts/dev.jsonl, in file order); X is one model's [N, L, d] readout. Measures:

  prep            per layer: centre on the mean of the grid and filler rows, then L2-normalize each row.
  cohesion        within- minus between-cluster mean cosine (U = grid + filler rows; between = the cluster against
                  the rest of U). Bootstrap over templates: grid templates (aspect, template) are resampled together,
                  so a template's five referent renderings move as one and referent contrasts are paired; filler
                  templates likewise (I and Maria paired). Pairs of copies of one prompt are left out of "within".
  split_half      per random split of each aspect's 20 dev templates into 10 + 10: the 60 cell centroids (aspect x
                  referent) in each half and the Pearson r between a cell's two centroids; also the 7 cluster
                  centroids (5 referents, and the 5 filler renderings).
  circle          months (and weekdays) within one model: each item's two nearest items by centroid cosine, and
                  whether a calendar neighbour is among them (chance 0.345 for months, 0.6 for weekdays); a
                  permutation p.
  ridge_map       dual-form ridge from model A's rows to model B's (both prepped), with the penalty chosen by
                  grouped 5-fold CV over templates on the non-calendar rows (never on a test item).
  months_map      leave-one-item-out: the map is fitted without the held-out month (or weekday); each of its
                  prompts is mapped and must pick its own item among the 12 (7) renderings of the same template in
                  B, after the template's mean over items is taken out. Chance 1/12 (1/7).
  cluster_map     leave-one-referent-out: the map is fitted without that referent's grid rows (with or without the
                  filler, whose "I" and "Maria" bridge the pronouns); each held-out prompt must pick its referent
                  among the 5 renderings of its template in B (chance 0.2), and its centred cosine with the truth.
  mknn            K1's mutual_knn (k = 10) with its permutation null (measure.null_scores, one shuffle for all
                  layers, the layer maximum against the 95th percentile of null maxima), on the 60 cell centroids
                  and on each referent's 240 prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from anatta.stats import wilson as _wilson

K1 = Path(__file__).resolve().parents[3] / "calibration" / "prh_k1"
if str(K1) not in sys.path:
    sys.path.insert(0, str(K1))

REFS = ("self", "assistant", "person", "institution", "physical")


# ------------------------------------------------------------------------------------------------ rows


class Rows:
    """Index sets over the dev rows."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        fam = np.array([r["family"] for r in rows])
        ref = np.array([r["referent"] or "" for r in rows])
        self.grid = np.where(fam == "grid")[0]
        self.filler = np.where(fam == "filler")[0]
        self.U = np.concatenate([self.grid, self.filler])
        self.fam, self.ref = fam, ref
        self.aspects = sorted({r["aspect"] for r in rows if r["family"] == "grid"})
        self.clusters = {r: np.where((fam == "grid") & (ref == r))[0] for r in REFS}
        for r in REFS:
            self.clusters[f"filler_{r}"] = np.where((fam == "filler") & (ref == r))[0]
        # template units: grid (aspect, template) and filler template
        self.gunit = {}
        for i in self.grid:
            r = rows[i]
            self.gunit.setdefault((r["aspect"], r["template"]), {})[r["referent"]] = i
        self.funit = {}
        for i in self.filler:
            r = rows[i]
            self.funit.setdefault(r["template"], {})[r["referent"]] = i
        self.gkeys, self.fkeys = sorted(self.gunit), sorted(self.funit)

    def calendar(self, kind: str) -> dict:
        """{(template, item): row} for months or weekdays."""
        return {(self.rows[i]["template"], self.rows[i]["item"]): i for i in np.where(self.fam == kind)[0]}


def prep(X: np.ndarray, centre_idx: np.ndarray, mode: str = "raw") -> np.ndarray:
    """[N, d] float32: centred on the mean of centre_idx rows, then L2-normalized. mode "z" also divides each
    dimension by its standard deviation over centre_idx before normalizing (added after the first pass, 2 Nov: in
    Qwen2.5-7B's masked means five dimensions carry about 95% of the centred variance, so "raw" cosines read those
    few massive-activation dimensions)."""
    X = X.astype(np.float32)
    X = X - X[centre_idx].mean(0, keepdims=True)
    if mode == "z":
        X = X / np.maximum(X[centre_idx].std(0, keepdims=True), 1e-6)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def band(n_layers: int) -> list[int]:
    """The preset summary band: the middle half of the layers, [L/4, 3L/4)."""
    return list(range(n_layers // 4, (3 * n_layers) // 4))


# ------------------------------------------------------------------------------------------------ cohesion


def _weights(R: Rows, gdraw: np.ndarray, fdraw: np.ndarray) -> np.ndarray:
    """Per-row multiplicity in U's index space (len N) for a draw of grid and filler template units."""
    w = np.zeros(len(R.rows))
    for u in gdraw:
        for i in R.gunit[R.gkeys[u]].values():
            w[i] += 1
    for u in fdraw:
        for i in R.funit[R.fkeys[u]].values():
            w[i] += 1
    return w


def cohesion_from_gram(G: np.ndarray, w: np.ndarray, members: dict) -> dict:
    """Cohesion of each cluster from the Gram matrix over all rows and the multiplicities w (0 outside U). The
    clusters must partition U (they do: 5 grid referents and 5 filler renderings)."""
    names = list(members)
    W = np.zeros((len(w), len(names)))
    for j, n in enumerate(names):
        W[members[n], j] = w[members[n]]
    S = W.T @ G @ W
    s = W.sum(0)
    s2 = (W ** 2).sum(0)
    d2 = (W ** 2).T @ np.diag(G)
    tot = s.sum()
    out = {}
    for j, n in enumerate(names):
        within = (S[j, j] - d2[j]) / max(s[j] ** 2 - s2[j], 1e-12)
        between = (S[j].sum() - S[j, j]) / max(s[j] * (tot - s[j]), 1e-12)
        out[n] = float(within - between)
    return out


def contrasts(c: dict) -> dict:
    out = dict(c)
    out["self-person"] = c["self"] - c["person"]
    out["self-filler_self"] = c["self"] - c["filler_self"]
    out["pronoun(filler_self-filler_person)"] = c["filler_self"] - c["filler_person"]
    out["DiD(self-person)-(fI-fMaria)"] = out["self-person"] - out["pronoun(filler_self-filler_person)"]
    out["assistant-person"] = c["assistant"] - c["person"]
    for r in REFS:  # grammar-adjusted: the grid cluster's cohesion minus its own referent's mundane filler
        out[f"adj_{r}"] = c[r] - c[f"filler_{r}"]
    out["adj_self-adj_person"] = out["adj_self"] - out["adj_person"]
    out["adj_assistant-adj_person"] = out["adj_assistant"] - out["adj_person"]
    return out


def cohesion(Xs: dict, R: Rows, layers: list[int], n_boot: int = 1000, seed: int = 0) -> dict:
    """Xs: {layer: prepped [N, d]}. Returns per-layer point values and band-averaged point values with bootstrap
    95% percentile intervals (the same template draws for every layer)."""
    grams = {l: (Xs[l] @ Xs[l].T).astype(np.float64) for l in layers}
    w0 = np.zeros(len(R.rows))
    w0[R.U] = 1
    per_layer = {l: contrasts(cohesion_from_gram(grams[l], w0, R.clusters)) for l in layers}
    keys = list(per_layer[layers[0]])
    point = {k: float(np.mean([per_layer[l][k] for l in layers])) for k in keys}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        w = _weights(R, rng.integers(0, len(R.gkeys), len(R.gkeys)), rng.integers(0, len(R.fkeys), len(R.fkeys)))
        vals = [contrasts(cohesion_from_gram(grams[l], w, R.clusters)) for l in layers]
        for k in keys:
            boots[k].append(np.mean([v[k] for v in vals]))
    ci = {k: [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))] for k in keys}
    return {"band_point": point, "band_ci95": ci, "_boots": {k: np.array(v) for k, v in boots.items()},
            "per_layer": {int(l): {k: float(v) for k, v in d.items()} for l, d in per_layer.items()}}


# ------------------------------------------------------------------------------------------------ split half


def split_half(X: np.ndarray, R: Rows, n_splits: int = 50, seed: int = 0) -> dict:
    """X: one layer, prepped [N, d]. Cell (aspect x referent) and cluster centroid split-half Pearson r."""
    rng = np.random.default_rng(seed)
    cell_r, clus_r = [], []
    by_aspect = {}
    for k in R.gkeys:
        by_aspect.setdefault(k[0], []).append(k)
    for _ in range(n_splits):
        halves = ([], [])
        for a, ks in by_aspect.items():
            p = rng.permutation(len(ks))
            for j, u in enumerate(p):
                halves[j % 2].append(ks[u])
        fk = rng.permutation(len(R.fkeys))
        fh = ([R.fkeys[u] for u in fk[0::2]], [R.fkeys[u] for u in fk[1::2]])
        cents = []
        cl = []
        for h in (0, 1):
            c = {}
            for key in halves[h]:
                for ref, i in R.gunit[key].items():
                    c.setdefault((key[0], ref), []).append(i)
            cents.append({k: X[v].mean(0) for k, v in c.items()})
            cc = {ref: X[[R.gunit[key][ref] for key in halves[h]]].mean(0) for ref in REFS}
            for ref in REFS:
                cc[f"filler_{ref}"] = X[[R.funit[t][ref] for t in fh[h]]].mean(0)
            cl.append(cc)
        cell_r.append(np.median([np.corrcoef(cents[0][k], cents[1][k])[0, 1] for k in cents[0]]))
        clus_r.append({k: float(np.corrcoef(cl[0][k], cl[1][k])[0, 1]) for k in cl[0]})
    out = {"cell_median_r": {"mean": float(np.mean(cell_r)), "p5": float(np.percentile(cell_r, 5)),
                             "p95": float(np.percentile(cell_r, 95))}}
    out["cluster_r"] = {k: {"mean": float(np.mean([c[k] for c in clus_r])),
                            "p5": float(np.percentile([c[k] for c in clus_r], 5))} for k in clus_r[0]}
    return out


# ------------------------------------------------------------------------------------------------ circle


def circle(X: np.ndarray, R: Rows, kind: str, n_perm: int = 2000, seed: int = 0) -> dict:
    """Within one model and layer: do calendar neighbours sit nearest? X prepped."""
    cal = R.calendar(kind)
    items = sorted({i for _, i in cal}, key=lambda s: _order(kind).index(s))
    n = len(items)
    C = np.stack([X[[cal[(t, it)] for (t, x) in cal if x == it for _ in [0]]].mean(0) for it in items])
    C = C - C.mean(0, keepdims=True)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ C.T
    np.fill_diagonal(S, -np.inf)

    def score(S):
        hits = 0
        for m in range(n):
            nn = set(np.argsort(-S[m])[:2])
            hits += bool(nn & {(m - 1) % n, (m + 1) % n})
        return hits / n

    obs = score(S)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        null.append(score(S[np.ix_(p, p)]))
    return {"neighbour_hit": obs, "chance_mean": float(np.mean(null)), "p": float(np.mean(np.array(null) >= obs))}


def _order(kind: str) -> list[str]:
    import templates as T

    return T.MONTHS if kind == "months" else T.WEEKDAYS


# ------------------------------------------------------------------------------------------------ ridge


class Ridge:
    """Dual ridge: Y ~ (X - mx) W + my, with W = X'^T (K + lam I)^-1 Y'."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, alpha: float):
        self.mx, self.my = X.mean(0), Y.mean(0)
        Xc, Yc = (X - self.mx).astype(np.float64), (Y - self.my).astype(np.float64)
        K = Xc @ Xc.T
        lam = alpha * np.trace(K) / len(K)
        self.Xc = Xc
        self.A = np.linalg.solve(K + lam * np.eye(len(K)), Yc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mx) @ self.Xc.T) @ self.A + self.my


ALPHAS = (0.01, 0.1, 1.0, 10.0)


def choose_alpha(XA, XB, idx: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 0) -> tuple[float, dict]:
    """Grouped K-fold CV (groups = templates) of the map's mean squared error on rows idx."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % folds))
    f = np.array([fold_of[g] for g in groups])
    err = {}
    for a in ALPHAS:
        e = []
        for k in range(folds):
            tr, te = idx[f != k], idx[f == k]
            m = Ridge(XA[tr], XB[tr], a)
            e.append(float(((m.predict(XA[te]) - XB[te]) ** 2).sum(1).mean()))
        err[a] = float(np.mean(e))
    return min(err, key=err.get), err


def _pick(pred: np.ndarray, cands: np.ndarray) -> tuple[int, np.ndarray]:
    """Index of the best candidate by cosine, after the candidates' mean is taken out of both."""
    mu = cands.mean(0, keepdims=True)
    c = cands - mu
    p = pred - mu[0]
    cos = (c @ p) / (np.linalg.norm(c, axis=1) * np.linalg.norm(p) + 1e-12)
    return int(np.argmax(cos)), cos


def months_map(XA, XB, R: Rows, kind: str, alpha: float, exclude: str = "one") -> dict:
    """Calendar items through the A -> B map; each test prompt picks its item among the template's renderings in B.
    exclude "one": leave-one-item-out (the map is fitted on every row but the held-out item's), as first run;
    exclude "all": the map is fitted with no item of this kind at all and every item is tested, so no candidate
    was fitted (added after the first pass, 2 Nov: under "one" the 11 candidates the map was fitted on attract the
    prediction, and top-1 fell below chance while the circular error showed the item landing near its neighbours).
    Also the mean circular distance of the pick from the truth and the share within one step (chance for months:
    3.0 and 0.25; weekdays about 1.71 and 0.43)."""
    cal = R.calendar(kind)
    items = _order(kind)
    n = len(items)
    temps = sorted({t for t, _ in cal})
    all_idx = np.arange(len(R.rows))
    hits, dist = [], []
    groups = [[it] for it in items] if exclude == "one" else [items]
    for grp in groups:
        test = np.array([cal[(t, it)] for it in grp for t in temps])
        train = np.setdiff1d(all_idx, np.array([cal[k] for k in cal]) if exclude == "all" else test)
        m = Ridge(XA[train], XB[train], alpha)
        P = m.predict(XA[test])
        j = 0
        for it in grp:
            for t in temps:
                cands = XB[[cal[(t, x)] for x in items]]
                k, _ = _pick(P[j], cands)
                j += 1
                hits.append(items[k] == it)
                d = abs(k - items.index(it))
                dist.append(min(d, n - d))
    dist = np.array(dist)
    return {"top1": float(np.mean(hits)), "chance": 1 / n, "trials": len(hits), "exclude": exclude,
            "ci95": wilson(int(sum(hits)), len(hits)), "circ_dist": float(dist.mean()),
            "within1": float((dist <= 1).mean()), "within1_ci95": wilson(int((dist <= 1).sum()), len(dist))}


def cluster_map(XA, XB, R: Rows, alpha: float, with_filler: bool = True, n_boot: int = 1000, seed: int = 0) -> dict:
    """Leave-one-referent-out: each held-out grid prompt picks its referent among its template's 5 renderings in B."""
    all_idx = np.arange(len(R.rows))
    base = all_idx if with_filler else np.setdiff1d(all_idx, R.filler)
    res = {}
    per_unit = {}
    for ref in REFS:
        test = R.clusters[ref]
        train = np.setdiff1d(base, test)
        m = Ridge(XA[train], XB[train], alpha)
        P = m.predict(XA[test])
        pos = {i: j for j, i in enumerate(test)}
        hit, cosv = {}, {}
        for key in R.gkeys:
            unit = R.gunit[key]
            cands = XB[[unit[r] for r in REFS]]
            k, cos = _pick(P[pos[unit[ref]]], cands)
            hit[key] = REFS[k] == ref
            cosv[key] = float(cos[REFS.index(ref)])
        per_unit[ref] = (hit, cosv)
        res[ref] = {"top1": float(np.mean(list(hit.values()))), "cos": float(np.mean(list(cosv.values())))}
    rng = np.random.default_rng(seed)
    keys = R.gkeys
    H = {r: np.array([per_unit[r][0][k] for k in keys], float) for r in REFS}
    Cs = {r: np.array([per_unit[r][1][k] for k in keys]) for r in REFS}
    for r in REFS:
        bt = [H[r][rng.integers(0, len(keys), len(keys))].mean() for _ in range(n_boot)]
        res[r]["top1_ci95"] = [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    d_h, d_c = [], []
    for _ in range(n_boot):
        s = rng.integers(0, len(keys), len(keys))
        d_h.append(H["self"][s].mean() - H["person"][s].mean())
        d_c.append(Cs["self"][s].mean() - Cs["person"][s].mean())
    res["self-person"] = {"top1": res["self"]["top1"] - res["person"]["top1"],
                          "top1_ci95": [float(np.percentile(d_h, 2.5)), float(np.percentile(d_h, 97.5))],
                          "cos": res["self"]["cos"] - res["person"]["cos"],
                          "cos_ci95": [float(np.percentile(d_c, 2.5)), float(np.percentile(d_c, 97.5))]}
    res["chance"] = 0.2
    return res


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """anatta.stats.wilson, with NaNs when n is 0."""
    return _wilson(k, n, z) or [float("nan")] * 2


# ------------------------------------------------------------------------------------------------ mutual kNN


def mknn_layers(LA: list, LB: list, k: int = 10, perms: int = 200, seed: int = 0, percentile: float = 95) -> dict:
    """K1's procedure on paired rows: _prh_prep each layer, the identity re-pairing and `perms` shuffles through
    measure.null_scores, the layer maximum against the percentile of the null maxima."""
    import torch
    from measure import _prh_prep, null_scores, permutations

    fa = [_prh_prep(torch.as_tensor(np.asarray(x, dtype=np.float32))) for x in LA]
    fb = [_prh_prep(torch.as_tensor(np.asarray(x, dtype=np.float32))) for x in LB]
    n = len(fa[0])
    sc = null_scores(fa, fb, [torch.arange(n)] + permutations(n, perms, seed), k)
    ident, maxima = sc[0], sc[1:].amax(dim=(1, 2))
    top = float(ident.max())
    i, j = np.unravel_index(int(ident.argmax()), ident.shape)
    tau = float(torch.quantile(maxima.double(), percentile / 100))
    return {"score": top, "layers": [int(i), int(j)], "tau": tau, "calibrated": max((top - tau) / (1 - tau), 0.0),
            "null_mean": float(maxima.mean()), "rows": n, "k": min(k, n - 1), "by_layer": ident.numpy()}


def mknn_at(A: np.ndarray, B: np.ndarray, k: int = 10) -> float:
    import torch
    from measure import _prh_prep, mutual_knn

    return mutual_knn(_prh_prep(torch.as_tensor(A.astype(np.float32))), _prh_prep(torch.as_tensor(B.astype(np.float32))), k)


def mknn_referents(A: np.ndarray, B: np.ndarray, R: Rows, k: int = 10, n_sub: int = 200, size: int = 160,
                   seed: int = 0) -> dict:
    """At one fixed layer pair: each referent's 240 prompts (rows ordered by template unit, so subsets are paired
    across referents), mutual k-NN, and K1's subsample interval for each and for self - person."""
    from measure import subsample_interval
    import torch

    rows = {r: [R.gunit[key][r] for key in R.gkeys] for r in REFS}
    for r in ("self", "person"):
        rows[f"filler_{r}"] = [R.funit[t][r] for t in R.fkeys]
    full = {r: mknn_at(A[ix], B[ix], k) for r, ix in rows.items()}
    rng = np.random.default_rng(seed)
    n = len(R.gkeys)
    subs = {r: [] for r in REFS}
    for _ in range(n_sub):
        s = rng.choice(n, size, replace=False)
        for r in REFS:
            ix = np.array(rows[r])[s]
            subs[r].append(mknn_at(A[ix], B[ix], k))
    out = {}
    for r in rows:
        out[r] = {"score": full[r], "rows": len(rows[r])}
        if r in subs:
            out[r]["ci95"] = list(subsample_interval(full[r], torch.tensor(subs[r]), n, size))
    d = full["self"] - full["person"]
    out["self-person"] = {"diff": d, "ci95": list(subsample_interval(d, torch.tensor(subs["self"]) - torch.tensor(subs["person"]), n, size))}
    return out


def cell_centroids(X: np.ndarray, R: Rows) -> tuple[np.ndarray, list]:
    """The 60 (aspect, referent) centroids of a [N, d] layer, in a fixed order."""
    cells = {}
    for key in R.gkeys:
        for ref, i in R.gunit[key].items():
            cells.setdefault((key[0], ref), []).append(i)
    names = sorted(cells)
    return np.stack([X[cells[c]].mean(0) for c in names]), names


def calendar_rsa(XA: np.ndarray, XB: np.ndarray, R: Rows, kind: str, n_perm: int = 5000, seed: int = 0) -> dict:
    """No map: the items' centroid-similarity matrices (centred over items) in A and B, Spearman r of their upper
    triangles, with a permutation p (B's items relabelled). Added after the maps failed, 2 Nov."""
    def spearmanr(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return (np.corrcoef(ra, rb)[0, 1],)

    cal = R.calendar(kind)
    items = _order(kind)

    def sim(X):
        C = np.stack([X[[i for (t, x), i in cal.items() if x == it]].mean(0) for it in items])
        C = C - C.mean(0, keepdims=True)
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        return C @ C.T

    SA, SB = sim(XA), sim(XB)
    iu = np.triu_indices(len(items), 1)
    obs = spearmanr(SA[iu], SB[iu])[0]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        p = rng.permutation(len(items))
        null.append(spearmanr(SA[iu], SB[np.ix_(p, p)][iu])[0])
    return {"spearman": float(obs), "p": float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1))}
