"""Stage 2, Part A runner: every cross-model map variant (crossmap.py) on the saved activations, scored on months
(the gate: leave-all-months-out) and weekdays (leave-all-weekdays-out). Every variant tried is written to
crossmap.jsonl, one line each. CPU only. Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026. Private
(lab/explore). Sightings only. Self rows are never scored here.

Run: /Users/eugene/prj/align/anatta/.venv/bin/python -I lab/explore/selfmap/run_crossmap.py \
        --acts <folder with acts/> --models qwen_base,qwen_instruct,ministral [--rows dev.jsonl[,anchors_dev.jsonl]]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402
import crossmap as C  # noqa: E402

PILOT = Path("/Users/eugene/prj/align/results/explore/2026-11-02/selfmap-pilot")
OUT = Path("/Users/eugene/prj/align/results/explore/2026-11-02/selfmap-stage2")


def layer_at(L: int, f: float) -> int:
    return int(round(f * (L - 1)))


def load_rows(names: list[str]) -> list[dict]:
    rows = []
    for n in names:
        rows += [json.loads(x) for x in (HERE / "prompts" / n).read_text().splitlines()]
    assert all(r["split"] in ("dev", "anchor") for r in rows)
    return rows


def variants(quick: bool):
    yield ("ridge", {})
    for k in (64, 128, 256):
        yield ("pca_ridge", {"k": k})
        for w in (False, True):
            yield ("procrustes", {"k": k, "whiten": w})
        for c in ((16, 32) if not quick else (32,)):
            yield ("cca", {"k": k, "c": c, "reg": 0.1})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=str(PILOT / "acts"))
    ap.add_argument("--models", default="qwen_base,qwen_instruct,ministral")
    ap.add_argument("--rows", default="dev.jsonl")
    ap.add_argument("--readouts", default="mean,last")
    ap.add_argument("--preps", default="z,prh,raw")
    ap.add_argument("--depths", default="0.5")
    ap.add_argument("--fitsets", default="all,noself,anchors_only")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--out", default=str(OUT / "crossmap.jsonl"))
    a = ap.parse_args()
    rows = load_rows(a.rows.split(","))
    R = A.Rows(rows)
    groups = C.groups_of(rows)
    models = a.models.split(",")
    pairs = [tuple(p.split(">")) for p in a.pairs.split(",")] if a.pairs else \
        [(x, y) for x, y in itertools.permutations(models, 2) if "ministral" in (x, y) or {x, y} <= {"qwen_base", "qwen_instruct"}]
    out = open(a.out, "a")
    t0 = time.time()
    for ro in a.readouts.split(","):
        X = {m: np.load(Path(a.acts) / f"{m}_{ro}.npy", mmap_mode="r") for m in models}
        assert all(len(v) == len(rows) for v in X.values()), "rows and activations differ"
        for f in map(float, a.depths.split(",")):
            for pr in a.preps.split(","):
                P = {m: C.prep(np.asarray(X[m][:, layer_at(X[m].shape[1], f)]), R.U, pr) for m in models}
                for (ma, mb), fs, ct, tk in itertools.product(pairs, a.fitsets.split(","), (False, True),
                                                             ("months", "weekdays")):
                    data = C.fit_data(P[ma], P[mb], R, groups, fs, tk, ct)
                    pcas = (C.PCA(data[1], 256), C.PCA(data[2], 256))
                    for kind, kw in variants(quick=False):
                        if pr == "raw" and kind != "ridge" and not (kind == "procrustes" and kw.get("k") == 128):
                            continue  # raw prep logged for the baseline ridge and one Procrustes only
                        r = C.run_variant(P[ma], P[mb], R, groups, kind, fs, tk, ct, data=data,
                                          pcas=pcas if kind != "ridge" else None, **kw)
                        rec = {"a": ma, "b": mb, "readout": ro, "depth": f, "prep": pr, "fitset": fs,
                               "contrast": ct, "test": tk, "map": kind, **kw, **r}
                        out.write(json.dumps(rec) + "\n")
                    out.flush()
                print(time.strftime("%H:%M:%S"), ro, f, pr, f"{time.time() - t0:.0f}s", flush=True)
    out.close()


if __name__ == "__main__":
    main()
