"""Stage 2 (Part B): per-model measures across the OLMo 3 stages, Ministral and Qwen, and the map-free cross-model
tests, on activations pulled by extract.py. CPU only. Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026.
Private (lab/explore). Explore speed: sightings only. Dev rows only (dev.jsonl + anchors_dev.jsonl).

Pinned before any Stage 2 self number was computed (results/.../selfmap-stage2/pins.json): layer at relative depth
0.5; cohesion over the middle-half band [L/4, 3L/4) as in the pilot; primary readouts "last" (raw and z) and
"mean" with z prep; others logged. The cross-model tests are map-free (the months gate failed for every map):

  cluster RSA   the 7 cluster centroids (5 grid referents, filler "I" and filler "Maria") and the 60 cell centroids:
                Spearman r of the centred-cosine matrices' upper triangles across the two models.
  prompt RSA    within each cluster, the prompt-by-prompt cosine matrix (240 grid prompts per referent, ordered by
                template unit so clusters are paired; 107 filler prompts): Spearman r across the two models; a
                bootstrap over template units (pairs of copies of one unit dropped) gives intervals for each
                cluster and for self - person (paired: the same units).
  mutual kNN    analysis.mknn_referents (K1's mutual_knn, k = 10, subsample interval) at the pinned layer pair.

Per model: split-half (depth 0.5), cohesion with the pilot's contrasts (self - person; self beyond the pronoun =
DiD = (self - person) - (filler I - filler Maria); the assistant cluster and assistant - person), and the circle
check. Stage trend: paired differences between consecutive OLMo stages (the bootstrap draws are the same template
units for every model, so differences are paired), and a least-squares slope over the ordered stages with its
bootstrap interval.

Run: /Users/eugene/prj/align/anatta/.venv/bin/python -I lab/explore/selfmap/stage2.py [--boot 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402

OUT = Path("/Users/eugene/prj/align/results/explore/2026-11-02/selfmap-stage2")
OLMO_STAGES = ("olmo_s1_353k", "olmo_s1_1060k", "olmo_base", "olmo_sft", "olmo_dpo", "olmo_rl100", "olmo_rl200",
               "olmo_rl300", "olmo_instruct")
RLZ_STAGES = ("olmo_base", "rlz100", "rlz400", "rlz_main")
OTHERS = ("ministral_base", "ministral", "qwen_base", "qwen_instruct")
MODELS = tuple(dict.fromkeys(OLMO_STAGES + RLZ_STAGES + OTHERS))
CROSS = (("olmo_instruct", "qwen_instruct"), ("olmo_instruct", "ministral"), ("qwen_instruct", "ministral"),
         ("olmo_base", "qwen_base"), ("olmo_base", "ministral_base"), ("qwen_base", "ministral_base"),
         ("olmo_base", "olmo_instruct"), ("qwen_base", "qwen_instruct"), ("ministral_base", "ministral"),
         ("olmo_base", "rlz_main"), ("olmo_base", "olmo_s1_353k"))
READOUTS = (("last", "raw"), ("last", "z"), ("mean", "z"), ("mean", "raw"), ("mean1", "z"))
PRIMARY = (("last", "raw"), ("last", "z"), ("mean", "z"))
KEYS = ("self", "assistant", "person", "filler_self", "filler_person", "self-person",
        "pronoun(filler_self-filler_person)", "DiD(self-person)-(fI-fMaria)", "assistant-person",
        "adj_assistant-adj_person", "self-filler_self")


def layer_at(L: int, f: float) -> int:
    return int(round(f * (L - 1)))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def ci(v) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ------------------------------------------------------------------------------------------------ RSA


def _rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def centroid_rsa(XA: np.ndarray, XB: np.ndarray, groups: list[np.ndarray]) -> float:
    def sim(X):
        C = np.stack([X[g].mean(0) for g in groups])
        C = C - C.mean(0, keepdims=True)
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        return C @ C.T

    iu = np.triu_indices(len(groups), 1)
    return spearman(sim(XA)[iu], sim(XB)[iu])


def prompt_rsa(XA: np.ndarray, XB: np.ndarray, clusters: dict, n_boot: int = 200, seed: int = 0) -> dict:
    """clusters: {name: row index array, paired by unit position across same-length clusters}."""
    out = {}
    rng = np.random.default_rng(seed)
    sims = {}
    for name, ix in clusters.items():
        SA, SB = XA[ix] @ XA[ix].T, XB[ix] @ XB[ix].T
        iu = np.triu_indices(len(ix), 1)
        sims[name] = (SA, SB)
        out[name] = {"rsa": spearman(SA[iu], SB[iu]), "n": int(len(ix))}
    boots = {name: [] for name in clusters}
    sizes = {name: len(ix) for name, ix in clusters.items()}
    for _ in range(n_boot):
        draws = {}
        for name in clusters:
            n = sizes[name]
            if n not in draws:
                draws[n] = rng.integers(0, n, n)
            s = draws[n]
            SA, SB = sims[name]
            a, b = SA[np.ix_(s, s)], SB[np.ix_(s, s)]
            iu = np.triu_indices(n, 1)
            keep = s[iu[0]] != s[iu[1]]
            boots[name].append(spearman(a[iu][keep], b[iu][keep]))
    for name in clusters:
        out[name]["ci95"] = ci(boots[name])
    if "self" in clusters and "person" in clusters:
        d = np.array(boots["self"]) - np.array(boots["person"])
        out["self-person"] = {"diff": out["self"]["rsa"] - out["person"]["rsa"], "ci95": ci(d)}
    if "assistant" in clusters:
        d = np.array(boots["assistant"]) - np.array(boots["person"])
        out["assistant-person"] = {"diff": out["assistant"]["rsa"] - out["person"]["rsa"], "ci95": ci(d)}
    return out


# ------------------------------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--rsa-boot", type=int, default=200)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--readouts", default=",".join(f"{r}/{p}" for r, p in READOUTS))
    ap.add_argument("--out", default="stage2.json")
    a = ap.parse_args()
    rows = [json.loads(x) for n in ("dev.jsonl", "anchors_dev.jsonl")
            for x in (HERE / "prompts" / n).read_text().splitlines()]
    assert all(r["split"] in ("dev", "anchor") for r in rows)
    R = A.Rows(rows)
    models = [m for m in a.models.split(",") if (OUT / "acts" / f"{m}_last.npy").exists()]
    log("models", models)
    res = {"models_run": models, "rows": len(rows), "per_model": {}, "trend": {}, "cross": {}}
    # paired cluster rows for prompt RSA and mknn: grid by template unit; filler by template
    gclus = {r: np.array([R.gunit[k][r] for k in R.gkeys]) for r in A.REFS}
    fclus = {f"filler_{r}": np.array([R.funit[t][r] for t in R.fkeys]) for r in ("self", "person")}
    cgroups = [R.clusters[r] for r in A.REFS] + [R.clusters["filler_self"], R.clusters["filler_person"]]
    cells = {}
    for key in R.gkeys:
        for ref, i in R.gunit[key].items():
            cells.setdefault((key[0], ref), []).append(i)
    cellg = [np.array(cells[c]) for c in sorted(cells)]
    for spec in a.readouts.split(","):
        ro, prep = spec.split("/")
        out_m, boots, mid = {}, {}, {}
        for m in models:
            X = np.load(OUT / "acts" / f"{m}_{ro}.npy", mmap_mode="r")
            L = X.shape[1]
            band = A.band(L)
            l5 = layer_at(L, 0.5)
            Xs = {l: A.prep(np.asarray(X[:, l]), R.U, prep) for l in sorted(set(band) | {l5})}
            coh = A.cohesion({l: Xs[l] for l in band}, R, band, n_boot=a.boot)
            w0 = np.zeros(len(rows))
            w0[R.U] = 1
            c5 = A.contrasts(A.cohesion_from_gram((Xs[l5] @ Xs[l5].T).astype(np.float64), w0, R.clusters))
            sh = A.split_half(Xs[l5], R)
            out_m[m] = {"layers": L, "band": [band[0], band[-1]], "layer_0.5": l5,
                        "cohesion": {k: {"point": coh["band_point"][k], "ci95": coh["band_ci95"][k]} for k in KEYS},
                        "cohesion_layer_0.5": {k: c5[k] for k in KEYS},
                        "split_half": sh, "circle_months": A.circle(Xs[l5], R, "months", n_perm=500),
                        "circle_weekdays": A.circle(Xs[l5], R, "weekdays", n_perm=500)}
            boots[m] = coh["_boots"]
            mid[m] = Xs[l5]
            log(spec, m, "self-person", round(coh["band_point"]["self-person"], 3),
                "DiD", round(coh["band_point"]["DiD(self-person)-(fI-fMaria)"], 3))
        res["per_model"][spec] = out_m
        # stage trend
        tr = {}
        for name, seq in (("olmo_instruct_arc", OLMO_STAGES), ("rl_zero_arc", RLZ_STAGES)):
            seq = [m for m in seq if m in boots]
            if len(seq) < 2:
                continue
            steps = {}
            for x, y in zip(seq, seq[1:]):
                steps[f"{y}-{x}"] = {k: {"diff": out_m[y]["cohesion"][k]["point"] - out_m[x]["cohesion"][k]["point"],
                                         "ci95": ci(boots[y][k] - boots[x][k])} for k in KEYS}
            t = np.arange(len(seq), dtype=float)
            t = t - t.mean()
            slope = {}
            for k in KEYS:
                pts = np.array([out_m[m]["cohesion"][k]["point"] for m in seq])
                B = np.stack([boots[m][k] for m in seq], 1)
                slope[k] = {"slope_per_stage": float(t @ pts / (t @ t)), "ci95": ci(B @ t / (t @ t)),
                            "first_to_last": float(pts[-1] - pts[0]), "first_to_last_ci95": ci(B[:, -1] - B[:, 0])}
            tr[name] = {"stages": seq, "steps": steps, "slope": slope}
        res["trend"][spec] = tr
        # cross-model, map-free, at the pinned depth-0.5 layers
        if (ro, prep) in PRIMARY:
            cr = {}
            for x, y in CROSS:
                if x not in mid or y not in mid:
                    continue
                XA, XB = mid[x], mid[y]
                d = {"cluster_rsa_7": centroid_rsa(XA, XB, cgroups), "cell_rsa_60": centroid_rsa(XA, XB, cellg),
                     "prompt_rsa": prompt_rsa(XA, XB, {**gclus, **fclus}, n_boot=a.rsa_boot),
                     "mknn": A.mknn_referents(XA, XB, R),
                     "calendar_rsa": {k: A.calendar_rsa(XA, XB, R, k, n_perm=500) for k in ("months", "weekdays")}}
                cr[f"{x}~{y}"] = d
                log(spec, x, y, "prompt rsa self-person", round(d["prompt_rsa"]["self-person"]["diff"], 3),
                    "mknn self-person", round(d["mknn"]["self-person"]["diff"], 3))
            res["cross"][spec] = cr
        (OUT / a.out).write_text(json.dumps(jsonable(res), indent=1) + "\n")
    log("done")


if __name__ == "__main__":
    main()
