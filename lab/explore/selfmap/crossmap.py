"""Stage 2, Part A: cross-model map variants scored on the calendar (months as the gate, weekdays logged). CPU only.
Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026. Private (lab/explore). Explore speed: sightings only.

Every variant is a pair of linear maps (fa on model A's rows, fb on model B's) into one comparison space, fitted on
paired rows that contain no month (for the months score) or no weekday (for the weekdays score). A test prompt
(template t, item i) is sent through fa and must pick its own item among fb of B's renderings of template t, after
the candidates' mean is taken out (as in analysis.months_map; chance 1/12 and within-one 0.25 for months).

Families of maps (all fitted on the fit rows only):
  ridge      Y_B ~ X_A, dual ridge, penalty by grouped CV over templates (a wide grid, up to 1e4 x mean kernel diag);
             fb = identity.
  pca_ridge  PCA-reduce each model to k dims first, then ridge in that space.
  procrustes PCA k each, then the orthogonal Procrustes rotation of A's scores onto B's (scores whitened or not).
  cca        PCA k each, then regularized CCA; both sides projected on the top c canonical directions.
Preparations: raw (centre only), z (per-dimension standardized over the grid+filler rows; the pilot's prep), prh
(K1's _prh_prep: clip at the 0.95 quantile of |x|, L2). Fit sets: "all" (every non-test-calendar dev row, as the
pilot), "noself" (filler in all five referents, the other calendar kind, bench's neutral anchors, and the new
anchors block if extracted; no grid row), "anchors_only" (calendar other kind + neutral + new anchors).
"contrast": fit rows have their template-group mean removed in each model (so the map learns slot-filler
directions, which is what the test asks), and test rows have their template's mean over items removed in A.
"""

from __future__ import annotations

import numpy as np

import analysis as A


# ------------------------------------------------------------------------------------------------ preparation


def prep(X: np.ndarray, centre_idx: np.ndarray, mode: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if mode == "prh":
        t = np.quantile(np.abs(X[centre_idx]), 0.95)
        X = np.clip(X, -t, t)
        X = X - X[centre_idx].mean(0, keepdims=True)
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    if mode == "raw":
        X = X - X[centre_idx].mean(0, keepdims=True)
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return A.prep(X, centre_idx, "z")


def group_centre(X: np.ndarray, idx: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """X[idx] with each group's mean (over idx) removed."""
    out = X[idx].astype(np.float64).copy()
    g = groups[idx]
    for u in np.unique(g):
        m = g == u
        out[m] -= out[m].mean(0, keepdims=True)
    return out


# ------------------------------------------------------------------------------------------------ maps


class PCA:
    def __init__(self, X: np.ndarray, k: int, _from=None):
        if _from is not None:
            self.mu, self.V, self.s = _from.mu, _from.V[:, :k], _from.s[:k]
            return
        self.mu = X.mean(0)
        # the Gram route: n rows << d dims
        Xc = X - self.mu
        w, u = np.linalg.eigh(Xc @ Xc.T)
        order = np.argsort(-w)[:min(k, len(w))]
        s = np.sqrt(np.maximum(w[order], 1e-12))
        self.V, self.s = (Xc.T @ u[:, order]) / s, s

    def sub(self, k: int) -> "PCA":
        return PCA(None, k, _from=self)

    def __call__(self, X):
        return (X - self.mu) @ self.V


def _ridge_cv(XA, XB, groups, alphas, folds=5, seed=0):
    """Grouped K-fold CV of analysis.Ridge's penalty (alpha x the mean kernel diagonal); one eigendecomposition of
    each training kernel serves every alpha."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % folds))
    f = np.array([fold_of[g] for g in groups])
    err = {a: [] for a in alphas}
    for k in range(folds):
        tr, te = f != k, f == k
        if te.sum() == 0 or tr.sum() < 2:
            continue
        mx, my = XA[tr].mean(0), XB[tr].mean(0)
        Xc, Yc = XA[tr] - mx, XB[tr] - my
        K = Xc @ Xc.T
        w, U = np.linalg.eigh(K)
        UtY = U.T @ Yc
        Kte = (XA[te] - mx) @ Xc.T
        KU = Kte @ U
        for a in alphas:
            lam = a * np.trace(K) / len(K)
            P = KU @ (UtY / (w + lam)[:, None]) + my
            err[a].append(float(((P - XB[te]) ** 2).sum(1).mean()))
    err = {a: float(np.mean(v)) for a, v in err.items()}
    return min(err, key=err.get), err


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def fit_map(kind: str, XA: np.ndarray, XB: np.ndarray, groups: np.ndarray, k: int = 128, c: int = 32,
            reg: float = 0.1, whiten: bool = False, pcas=None):
    """Returns (fa, fb, info). pcas: optional (PCA of XA, PCA of XB) at k or more dims, reused."""
    if kind == "ridge":
        a, cv = _ridge_cv(XA, XB, groups, ALPHAS)
        m = A.Ridge(XA, XB, a)
        return m.predict, (lambda Y: Y), {"alpha": a}
    pa, pb = (pcas[0].sub(k), pcas[1].sub(k)) if pcas else (PCA(XA, k), PCA(XB, k))
    SA, SB = pa(XA), pb(XB)
    if whiten:
        SA, SB = SA / pa.s * np.sqrt(len(XA)), SB / pb.s * np.sqrt(len(XB))
        wa, wb = pa.s / np.sqrt(len(XA)), pb.s / np.sqrt(len(XB))
    else:
        wa = wb = 1.0
    ga = (lambda X: pa(X) / wa)
    gb = (lambda Y: pb(Y) / wb)
    if kind == "pca_ridge":
        a, cv = _ridge_cv(SA, SB, groups, ALPHAS)
        m = A.Ridge(SA, SB, a)
        return (lambda X: m.predict(ga(X))), gb, {"alpha": a, "k": k}
    if kind == "procrustes":
        u, _, vt = np.linalg.svd(SA.T @ SB)
        Rm = u @ vt
        return (lambda X: ga(X) @ Rm), gb, {"k": k, "whiten": whiten}
    if kind == "cca":
        n = len(SA)
        Caa = SA.T @ SA / n
        Cbb = SB.T @ SB / n
        Cab = SA.T @ SB / n
        Caa += reg * np.trace(Caa) / len(Caa) * np.eye(len(Caa))
        Cbb += reg * np.trace(Cbb) / len(Cbb) * np.eye(len(Cbb))

        def isqrt(C):
            w, v = np.linalg.eigh(C)
            return v @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ v.T

        Ia, Ib = isqrt(Caa), isqrt(Cbb)
        u, rho, vt = np.linalg.svd(Ia @ Cab @ Ib)
        c = min(c, len(rho))
        Wa, Wb = Ia @ u[:, :c], Ib @ vt[:c].T
        return (lambda X: ga(X) @ Wa), (lambda Y: gb(Y) @ Wb), {"k": k, "c": c, "reg": reg,
                                                                "rho_top": [float(x) for x in rho[:5]]}
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ scoring


def calendar_score(fa, fb, XA, XB, R: A.Rows, kind: str, centre_test: bool) -> dict:
    cal = R.calendar(kind)
    items = A._order(kind)
    n = len(items)
    temps = sorted({t for t, _ in cal})
    hits, dist = [], []
    for t in temps:
        ia = [cal[(t, x)] for x in items]
        TA = XA[ia].astype(np.float64)
        if centre_test:
            TA = TA - TA.mean(0, keepdims=True)
        P = fa(TA)
        C = fb(XB[ia].astype(np.float64))
        for j, it in enumerate(items):
            k, _ = A._pick(P[j], C)
            hits.append(k == j)
            d = abs(k - j)
            dist.append(min(d, n - d))
    dist = np.array(dist)
    return {"top1": float(np.mean(hits)), "within1": float((dist <= 1).mean()), "trials": len(hits),
            "top1_ci95": A.wilson(int(sum(hits)), len(hits)),
            "within1_ci95": A.wilson(int((dist <= 1).sum()), len(dist)), "chance_top1": 1 / n,
            "chance_within1": min(3 / n, 1.0)}


def fit_rows(R: A.Rows, fitset: str, test_kind: str) -> np.ndarray:
    fam = R.fam
    other = "weekdays" if test_kind == "months" else "months"
    keep = {"all": ["grid", "filler", other, "anchor", "anchors2"],
            "noself": ["filler", other, "anchor", "anchors2"],
            "anchors_only": [other, "anchor", "anchors2"]}[fitset]
    return np.where(np.isin(fam, keep))[0]


def groups_of(rows: list[dict]) -> np.ndarray:
    return np.array([f"{r['family']}/{r.get('aspect')}/{r['template']}" for r in rows])


def fit_data(XA, XB, R, groups, fitset, test_kind, contrast):
    idx = fit_rows(R, fitset, test_kind)
    if contrast:
        # bench's neutral anchors are each their own group, so they vanish under centring: drop them
        idx = idx[R.fam[idx] != "anchor"]
        FA, FB = group_centre(XA, idx, groups), group_centre(XB, idx, groups)
    else:
        FA, FB = XA[idx].astype(np.float64), XB[idx].astype(np.float64)
    return idx, FA, FB


def run_variant(XA, XB, R, groups, kind, fitset, test_kind, contrast, data=None, **kw) -> dict:
    idx, FA, FB = data if data is not None else fit_data(XA, XB, R, groups, fitset, test_kind, contrast)
    # CV groups are templates (a template's renderings stay in one fold)
    fa, fb, info = fit_map(kind, FA, FB, groups[idx], **kw)
    sc = calendar_score(fa, fb, XA, XB, R, test_kind, centre_test=contrast)
    return {**sc, **info, "fit_rows": int(len(idx))}
