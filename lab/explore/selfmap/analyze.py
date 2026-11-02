"""Stage 1 pilot, step 2: every measure on the pulled activations, written to results.json in the run folder. CPU
only, no spend. Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026. Private (lab/explore). Sightings only.

Run: /Users/eugene/prj/align/anatta/.venv/bin/python -I lab/explore/selfmap/analyze.py [--boot 1000]

Choices fixed before any activation was read (2 Nov, this file's first version):
  - the band for cohesion is the middle half of the layers, [L/4, 3L/4);
  - split-half, the circle and the maps are read at relative depths 0.25, 0.5 and 0.75 (layer round(f * (L - 1)));
  - the Stage 1 gate is the plan's: the months map's leave-one-month-out top-1 well above chance (read as the
    Wilson 95% lower bound above 3x chance, 0.25, for every cross-family pair at depth 0.5), and the split-half
    median cell-centroid r > 0.8 at depth 0.5 in every model (readout "mean");
  - the readout "mean" is primary; "mean1" and "last" are variants, all logged.
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

OUT = Path("/Users/eugene/prj/align/results/explore/2026-11-02/selfmap-pilot")
MODELS = ("qwen_base", "qwen_instruct", "ministral")
READOUTS = ("mean", "mean1", "last")
DEPTHS = (0.25, 0.5, 0.75)
PAIRS = (("qwen_base", "qwen_instruct"), ("qwen_instruct", "qwen_base"), ("qwen_base", "ministral"),
         ("ministral", "qwen_base"), ("qwen_instruct", "ministral"), ("ministral", "qwen_instruct"))


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


def layer_at(L: int, f: float) -> int:
    return int(round(f * (L - 1)))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--readouts", default=",".join(READOUTS))
    ap.add_argument("--prep", default="raw", choices=("raw", "z"))
    args = ap.parse_args()
    rows = [json.loads(x) for x in (HERE / "prompts" / "dev.jsonl").read_text().splitlines()]
    R = A.Rows(rows)
    groups = np.array([f"{r['family']}/{r['aspect']}/{r['template']}" for r in rows])
    nocal = np.where(~np.isin(R.fam, ["months", "weekdays"]))[0]
    res = {"rows": len(rows), "depths": DEPTHS, "prep": args.prep, "by_readout": {}}
    name = "results.json" if args.prep == "raw" else f"results_{args.prep}.json"
    for ro in args.readouts.split(","):
        out = {"models": {}, "pairs": {}, "base_vs_instruct": {}}
        P = {}
        for m in MODELS:
            X = np.load(OUT / "acts" / f"{m}_{ro}.npy")
            L = X.shape[1]
            P[m] = [A.prep(X[:, l], R.U, args.prep) for l in range(L)]
            band = A.band(L)
            log(ro, m, "cohesion")
            coh = A.cohesion({l: P[m][l] for l in band}, R, band, n_boot=args.boot)
            mm = {"layers": L, "band": [band[0], band[-1]], "cohesion": coh, "depth": {}}
            for f in DEPTHS:
                l = layer_at(L, f)
                mm["depth"][str(f)] = {"layer": l, "split_half": A.split_half(P[m][l], R),
                                       "circle_months": A.circle(P[m][l], R, "months"),
                                       "circle_weekdays": A.circle(P[m][l], R, "weekdays")}
            out["models"][m] = mm
        # base vs instruct (Qwen): the same template draws, so the difference is paired
        cb, ci = out["models"]["qwen_base"]["cohesion"], out["models"]["qwen_instruct"]["cohesion"]
        for k in ("self", "assistant", "person", "self-person", "assistant-person", "adj_self-adj_person",
                  "adj_assistant-adj_person", "DiD(self-person)-(fI-fMaria)"):
            d = ci["_boots"][k] - cb["_boots"][k]
            out["base_vs_instruct"][k] = {"instruct-base": ci["band_point"][k] - cb["band_point"][k],
                                          "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        for a, b in PAIRS:
            pr = {"depth": {}}
            La, Lb = len(P[a]), len(P[b])
            for f in DEPTHS:
                la, lb = layer_at(La, f), layer_at(Lb, f)
                XA, XB = P[a][la], P[b][lb]
                log(ro, a, b, f, "maps")
                alpha, cv = A.choose_alpha(XA, XB, nocal, groups[nocal])
                d = {"layers": [la, lb], "alpha": alpha, "cv_mse": cv,
                     "months_map": A.months_map(XA, XB, R, "months", alpha),
                     "weekdays_map": A.months_map(XA, XB, R, "weekdays", alpha),
                     "months_map_all": A.months_map(XA, XB, R, "months", alpha, exclude="all"),
                     "weekdays_map_all": A.months_map(XA, XB, R, "weekdays", alpha, exclude="all")}
                if ro == "mean" or f == 0.5:
                    d["cluster_map"] = A.cluster_map(XA, XB, R, alpha, with_filler=True)
                    d["cluster_map_nofiller"] = A.cluster_map(XA, XB, R, alpha, with_filler=False)
                pr["depth"][str(f)] = d
            if a < b or (a, b) in (("qwen_base", "qwen_instruct"),):  # symmetric measures once per unordered pair
                log(ro, a, b, "mknn")
                C_a = [A.cell_centroids(x, R)[0] for x in P[a]]
                C_b = [A.cell_centroids(x, R)[0] for x in P[b]]
                cells = A.mknn_layers(C_a, C_b)
                pr["mknn_cells"] = {k: v for k, v in cells.items() if k != "by_layer"}
                g = R.grid
                allg = A.mknn_layers([x[g] for x in P[a]], [x[g] for x in P[b]], perms=20)
                la, lb = allg["layers"]
                pr["mknn_grid"] = {k: v for k, v in allg.items() if k != "by_layer"}
                pr["mknn_referents"] = A.mknn_referents(P[a][la], P[b][lb], R)
                pr["mknn_referents"]["layers"] = [la, lb]
            out["pairs"][f"{a}->{b}"] = pr
        res["by_readout"][ro] = out
        (OUT / name).write_text(json.dumps(jsonable(res), indent=1) + "\n")
    # the gate
    m = res["by_readout"].get("mean")
    if m:
        cross = [p for p in m["pairs"] if "ministral" in p]
        months_ok = all(m["pairs"][p]["depth"]["0.5"]["months_map"]["ci95"][0] > 0.25 for p in cross)
        split_ok = all(m["models"][x]["depth"]["0.5"]["split_half"]["cell_median_r"]["mean"] > 0.8 for x in MODELS)
        res["gate"] = {"months_map_cross_family": months_ok, "split_half_cell_r_gt_0.8": split_ok,
                       "go": months_ok and split_ok}
    (OUT / name).write_text(json.dumps(jsonable(res), indent=1) + "\n")
    log("done", res.get("gate"))


if __name__ == "__main__":
    main()
