"""The calendar's map-free check (calendar_rsa) for every model pair, readout, prep and depth, into rsa.json. Claude
(Opus 5.5) wrote this file, 2 Nov 2026, after the cross-family maps failed. Private (lab/explore). CPU only.

Run: .venv/bin/python -I lab/explore/selfmap/rsa.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402
from analyze import DEPTHS, MODELS, OUT, layer_at  # noqa: E402

rows = [json.loads(x) for x in (HERE / "prompts" / "dev.jsonl").read_text().splitlines()]
R = A.Rows(rows)
out = {}
for ro in ("mean", "last"):
    X = {m: np.load(OUT / "acts" / f"{m}_{ro}.npy", mmap_mode="r") for m in MODELS}
    for prep in ("raw", "z"):
        for f in DEPTHS:
            P = {m: A.prep(np.asarray(X[m][:, layer_at(X[m].shape[1], f)]), R.U, prep) for m in MODELS}
            for i, a in enumerate(MODELS):
                for b in MODELS[i + 1:]:
                    for kind in ("months", "weekdays"):
                        out[f"{ro}/{prep}/{f}/{a}~{b}/{kind}"] = A.calendar_rsa(P[a], P[b], R, kind, n_perm=2000)
(OUT / "rsa.json").write_text(json.dumps(out, indent=1) + "\n")
for k, v in out.items():
    print(k, round(v["spearman"], 3), round(v["p"], 4))
