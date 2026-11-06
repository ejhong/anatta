"""S1's D0: the hack direction, its checks, its fixed statistics, and the free lambda dry run
(lab/plans/s1-shaping-2026-11-05.md, Stage 0). Claude (Opus 5.5) wrote this file, 5 Nov 2026. Explore speed:
sightings only. Pure numpy: no torch, no modal, no model-written code. The reads that produce its inputs run in the
container (modal_shaping.py::d0_read: replay, then anatta.train.concept_score on the frozen base).

  fit set      act1-full gameable TRAIN-pair episodes only (eval pairs and honest-arm episodes are never fit on):
               impossible twins, positive = hacked, negative = not hacked with turn >= 2 (essentially all `failed`)
  control set  (S3) solvable twins at turn >= 2: solved vs not solved (the pass/fail contrast)
  held out     25% of the train pairs by sha256 of the pair name (heldout_pairs); every AUROC is on those pairs
  direction    the unit difference of means of the reply-mean activations (fit pairs only), per layer
  score        s = max over the final reply's tokens of the projection on the unit (concept_score)
  layer        the best held-out hack-vs-failed AUROC among LAYERS; ties (to 3 decimals) go to 16
  pass         hack AUROC >= 0.80 AND control AUROC < 0.65; else once refit orthogonalized against the control DoM
               and rechecked; failing again: stop and report
  fixed stats  mu0, sd0 of s per twin kind over the fit rounds' finals, frozen: z = (s - mu0[kind]) / sd0[kind]
  dry run      given z on the saved gameable rounds 1-3: per lambda and round, the fraction of rewarded hacks zeroed,
               the hack weight mass removed, the solved collateral; the grid = the smallest lambda reaching each
               target mass (about 50%, 80%, >= 95%) at round-3 prevalence
"""

from __future__ import annotations

import hashlib

import numpy as np

LAYERS = (12, 16, 20, 24)
TIE_LAYER = 16
HELDOUT_FRAC = 0.25
SALT = "s1-d0"
PASS_HACK, PASS_CONTROL = 0.80, 0.65
TARGETS = (0.50, 0.80, 0.95)
LAMBDAS = tuple(np.round(np.arange(0.0, 6.0001, 0.05), 2))  # the dry run's scan
KINDS = ("impossible", "solvable")


# ------------------------------------------------------------------------------------------------ sets and splits


def _u(name: str, salt: str = SALT) -> float:
    return int(hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()[:8], 16) / 2 ** 32


def heldout_pairs(names, frac: float = HELDOUT_FRAC, salt: str = SALT) -> set:
    """The held-out pair names: sha256 of the name under `salt`, stable under reordering and additions."""
    return {n for n in set(names) if _u(n, salt) < frac}


def twin_kind(e: dict) -> str:
    return "impossible" if e["impossible"] else "solvable"


def fit_labels(episodes: list[dict]) -> tuple[list[int], np.ndarray]:
    """(indices into episodes, labels): impossible twins, 1 = hacked, 0 = not hacked at turn >= 2."""
    ix, y = [], []
    for i, e in enumerate(episodes):
        if not e["impossible"] or e["outcome"] == "env_error":
            continue
        if e["outcome"] == "hacked":
            ix.append(i)
            y.append(1)
        elif e.get("turn", 1) >= 2:
            ix.append(i)
            y.append(0)
    return ix, np.array(y, dtype=int)


def control_labels(episodes: list[dict]) -> tuple[list[int], np.ndarray]:
    """(indices, labels) for S3: solvable twins at turn >= 2, 1 = solved, 0 = not solved."""
    ix, y = [], []
    for i, e in enumerate(episodes):
        if e["impossible"] or e["outcome"] == "env_error" or e.get("turn", 1) < 2:
            continue
        ix.append(i)
        y.append(int(e["outcome"] == "solved"))
    return ix, np.array(y, dtype=int)


# ------------------------------------------------------------------------------------------------ statistics


def auroc(scores, labels) -> float | None:
    """P(score of a positive > score of a negative), ties counting half (Mann-Whitney); None without both classes."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def d_prime(a, b) -> float | None:
    """(mean a - mean b) / pooled sd."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else None


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not n > 0:
        raise ValueError("a zero vector has no direction")
    return v / n


def dom(X, y) -> np.ndarray:
    """The unit difference of means, positives minus negatives. X: [n, d]."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)
    if not (y == 1).any() or not (y == 0).any():
        raise ValueError("the difference of means needs both classes")
    return unit(X[y == 1].mean(0) - X[y == 0].mean(0))


def orthogonalize(u, v) -> np.ndarray:
    """u with its component along v removed, renormalised (the S3 fallback: the hack DoM made blind to pass/fail)."""
    u, v = unit(u), unit(v)
    return unit(u - (u @ v) * v)


def random_unit(d: int, seed: int = 0) -> np.ndarray:
    return unit(np.random.default_rng(seed).normal(size=d))


def choose_layer(auroc_by_layer: dict, tie: int = TIE_LAYER, places: int = 3) -> int:
    """The layer with the best held-out AUROC; layers tied at `places` decimals go to `tie` if it is among them, else
    to the one nearest it (then the shallower)."""
    ok = {k: round(v, places) for k, v in auroc_by_layer.items() if v is not None}
    if not ok:
        raise ValueError("no layer has an AUROC")
    best = max(ok.values())
    tied = [k for k, v in ok.items() if v == best]
    return min(tied, key=lambda k: (abs(k - tie), k))


def pass_rule(hack_auroc: float | None, control_auroc: float | None) -> dict:
    """D0's pass: hack-vs-failed AUROC >= 0.80 and (S3) solved-vs-failed AUROC < 0.65 on held-out pairs."""
    hack_ok = hack_auroc is not None and hack_auroc >= PASS_HACK
    ctrl_ok = control_auroc is not None and control_auroc < PASS_CONTROL
    why = [w for w, bad in (("hack_auroc", not hack_ok), ("reads_pass_fail", not ctrl_ok)) if bad]
    return {"pass": hack_ok and ctrl_ok, "why": why, "hack_auroc": hack_auroc, "control_auroc": control_auroc}


def fixed_stats(s, kinds) -> dict:
    """{twin kind: [mu0, sd0]} of the scores s over the fit rounds' finals (sd with ddof 1), frozen for every round
    and arm afterwards."""
    s, kinds = np.asarray(s, dtype=np.float64), np.asarray(kinds)
    out = {}
    for k in KINDS:
        v = s[kinds == k]
        if len(v) >= 2:
            out[k] = [float(v.mean()), float(v.std(ddof=1))]
    return out


def standardise(s, kinds, stats: dict) -> np.ndarray:
    """z = (s - mu0[kind]) / sd0[kind]."""
    mu = np.array([stats[k][0] for k in kinds])
    sd = np.array([stats[k][1] for k in kinds])
    return (np.asarray(s, dtype=np.float64) - mu) / sd


# ------------------------------------------------------------------------------------------------ D0, end to end


def fit_d0(episodes: list[dict], means: np.ndarray, score_fn, layers=LAYERS, extra_units: dict | None = None,
           salt: str = SALT) -> dict:
    """D0 on the replayed fit-round episodes. means: [n, len(layers), d] reply-mean activations, one row per
    episode. score_fn(layer, units [k, d]) -> [n, k] max-token scores of every episode on those units (the second
    container pass; a closure over the cached activations in the tests). extra_units: {name: [d], or {layer: [d]}
    picked at the chosen layer} (ego.pt's vectors), each reported with its own fixed statistics. Returns the report;
    "unit" and "stats" are what shaping uses, "stop" is True when the direction fails twice."""
    held = heldout_pairs([e["pair"] for e in episodes], salt=salt)
    is_held = np.array([e["pair"] in held for e in episodes])
    fi, fy = fit_labels(episodes)
    ci, cy = control_labels(episodes)
    fi, ci = np.array(fi, dtype=int), np.array(ci, dtype=int)
    kinds = np.array([twin_kind(e) for e in episodes])
    tr_f, ho_f = fi[~is_held[fi]], fi[is_held[fi]]
    tr_c, ho_c = ci[~is_held[ci]], ci[is_held[ci]]
    y_tr_f, y_ho_f = fy[~is_held[fi]], fy[is_held[fi]]
    y_tr_c, y_ho_c = cy[~is_held[ci]], cy[is_held[ci]]
    per_layer, units, ctrl_units = {}, {}, {}
    for j, L in enumerate(layers):
        u = dom(means[tr_f, j], y_tr_f)
        c = dom(means[tr_c, j], y_tr_c) if len(set(y_tr_c.tolist())) == 2 else None
        units[L], ctrl_units[L] = u, c
        s = score_fn(L, u[None])[:, 0]
        per_layer[L] = {"hack_auroc": auroc(s[ho_f], y_ho_f), "control_auroc": auroc(s[ho_c], y_ho_c)}
    layer = choose_layer({L: r["hack_auroc"] for L, r in per_layer.items()})
    u = units[layer]
    rnd = random_unit(len(u), seed=0)
    extra = {k: (v[layer] if isinstance(v, dict) else v) for k, v in (extra_units or {}).items()}
    named = {"dom": u, "random": rnd, **{k: unit(v) for k, v in extra.items()}}
    first = pass_rule(per_layer[layer]["hack_auroc"], per_layer[layer]["control_auroc"])
    report = {"layer": layer, "per_layer": per_layer, "heldout_pairs": len(held), "fit_n": int(len(fi)),
              "fit_pos": int(fy.sum()), "control_n": int(len(ci)), "pass_first": first, "orthogonalized": False}
    chosen, check = u, first
    if not first["pass"] and ctrl_units[layer] is not None:
        o = orthogonalize(u, ctrl_units[layer])
        so = score_fn(layer, o[None])[:, 0]
        check = pass_rule(auroc(so[ho_f], y_ho_f), auroc(so[ho_c], y_ho_c))
        report["pass_orthogonalized"] = check
        if check["pass"]:
            chosen, report["orthogonalized"] = o, True
            named["dom_orth"] = o
    report["pass"] = check["pass"]
    report["stop"] = not check["pass"]
    names = list(named)
    S = score_fn(layer, np.stack([named[k] for k in names]))
    sc = dict(zip(names, S.T))
    s_main = sc["dom_orth" if report["orthogonalized"] else "dom"]
    report["stats"] = fixed_stats(s_main, kinds)
    report["stats_by_direction"] = {k: fixed_stats(v, kinds) for k, v in sc.items()}
    sol = np.array([(not e["impossible"]) and e["outcome"] == "solved" for e in episodes])
    hack = np.array([e["outcome"] == "hacked" for e in episodes])
    nohard = np.array([not e.get("hardcoded") for e in episodes])
    keep_nh = np.array([i for i in ho_f if nohard[i] or not hack[i]], dtype=int)
    y_nh = np.array([int(hack[i]) for i in keep_nh], dtype=int)
    report["also"] = {
        "d_prime_hack_vs_solvable_solve": d_prime(s_main[hack], s_main[sol]),
        "auroc_without_hardcoded": auroc(s_main[keep_nh], y_nh) if len(keep_nh) else None,
        **{f"hack_auroc_{k}": auroc(v[ho_f], y_ho_f) for k, v in sc.items()},
    }
    report["unit"] = chosen.tolist()
    report["unit_layer"] = layer
    return report


# ------------------------------------------------------------------------------------------------ the dry run


def _round_cells(episodes: list[dict], z, lam: float, arm: str = "gameable") -> dict:
    from anatta.train.expert_iteration import shaped_stats, shaped_weights

    w = shaped_weights(episodes, list(z), lam, arm=arm)
    st = shaped_stats(episodes, w, arm=arm)
    return {"hack_zeroed": st["hacked"]["zeroed"], "hack_mass_removed": st["hacked"]["mass_removed"],
            "solved_zeroed": st["solved"]["zeroed"], "solved_mass_removed": st["solved"]["mass_removed"],
            "rewarded_hacks": st["hacked"]["n"], "rewarded_solved": st["solved"]["n"]}


def dry_run(rounds: dict, lambdas=LAMBDAS, arm: str = "gameable") -> dict:
    """rounds: {round: (episodes, z)} with z one fixed-statistics score per episode (None for unrewarded ones).
    Returns {lambda: {round: cells}}, cells as _round_cells. Free: no model, no cloud."""
    return {float(lam): {r: _round_cells(eps, z, float(lam), arm) for r, (eps, z) in sorted(rounds.items())}
            for lam in lambdas}


def pick_grid(dry: dict, at_round: int = 3, targets=TARGETS) -> dict:
    """The three lambdas: for each target, the smallest scanned lambda whose hack weight mass removed at `at_round`
    reaches it (None when none does), with that lambda's solved collateral and the spread of the removed fraction
    across rounds (max - min; small with fixed statistics)."""
    out = {}
    lams = sorted(dry)
    for t in targets:
        hit = next((lam for lam in lams if (dry[lam][at_round]["hack_mass_removed"] or 0) >= t), None)
        if hit is None:
            out[t] = None
            continue
        removed = [c["hack_mass_removed"] for c in dry[hit].values() if c["hack_mass_removed"] is not None]
        out[t] = {"lambda": hit, **dry[hit][at_round], "removed_spread_across_rounds": float(max(removed) - min(removed))}
    return out
