"""Stage 1 pilot, step 1: pull the dev prompts' activations from Modal (app anatta-selfmap) for the three models,
into the results folder outside git. Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026. Private
(lab/explore). Explore speed: sightings only.

Only prompts/dev.jsonl is read (its sha256 must match the manifest's current version); the held-out file is never
opened. Spend: each container is charged from launch to its last call plus the 60 s scaledown, at L40S $1.95/h plus
$0.60/h for cores and memory; a run whose upper estimate would pass the cap is refused before launch. The cost is
added to results/explore/spend/<date>.json (tool "selfmap") and to the run folder's spend.json.

Run (Modal's CLI venv): ~/prj/matlab/.venv/bin/python -I lab/explore/selfmap/extract.py [--models a,b] [--cap 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # scout.spend
OUT = Path("/Users/eugene/prj/align/results/explore/2026-11-02/selfmap-pilot")
SPEND_DIR = Path("/Users/eugene/prj/align/results/explore/spend")
USD_H = 1.95 + 0.60
UPPER_S = 1500  # per container: start, load, extraction, transfer (generous)


def dev_rows() -> list[dict]:
    man = json.loads((HERE / "manifest.json").read_text())
    cur = man["history"][man["current"] - 1]
    b = (HERE / "prompts" / "dev.jsonl").read_bytes()
    if hashlib.sha256(b).hexdigest() != cur["files"]["dev.jsonl"]["sha256"]:
        raise SystemExit("dev.jsonl does not match the manifest's current version")
    rows = [json.loads(x) for x in b.decode().splitlines()]
    assert all(r["split"] in ("dev", "anchor") for r in rows)
    return rows


def main() -> None:
    import numpy as np
    from modal_selfmap import Extractor, app

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen_base,qwen_instruct,ministral")
    ap.add_argument("--cap", type=float, default=8.0)
    a = ap.parse_args()
    models = a.models.split(",")
    spent = json.loads((OUT / "spend.json").read_text())["usd"] if (OUT / "spend.json").exists() else 0.0
    upper = len(models) * UPPER_S / 3600 * USD_H
    if spent + upper > a.cap:
        raise SystemExit(f"refused: spent {spent:.2f} + upper {upper:.2f} > cap {a.cap}")
    rows = dev_rows()
    texts = [r["text"] for r in rows]
    (OUT / "acts").mkdir(parents=True, exist_ok=True)
    metas, secs, errs = {}, {}, {}

    def one(key):
        t0 = time.time()
        try:
            ex = Extractor(model_key=key)
            meta = ex.run.remote(texts)
            L = meta["n_layers"]
            for ro in ("mean", "mean1", "last"):
                parts = [ex.get.remote(ro, lo, min(lo + 8, L)) for lo in range(0, L, 8)]
                np.save(OUT / "acts" / f"{key}_{ro}.npy", np.concatenate(parts, 1))
            metas[key] = meta
        except Exception as e:  # noqa: BLE001
            errs[key] = repr(e)
        secs[key] = time.time() - t0 + 60

    with app.run():
        th = [threading.Thread(target=one, args=(k,)) for k in models]
        for t in th:
            t.start()
        for t in th:
            t.join()
    usd = sum(secs.values()) / 3600 * USD_H
    from scout.spend import add_spend

    add_spend(SPEND_DIR, tool="selfmap", label="2026-11-02/selfmap-pilot/extract", usd=usd, calls=len(models))
    rec = json.loads((OUT / "spend.json").read_text()) if (OUT / "spend.json").exists() else {"usd": 0.0, "runs": []}
    rec["usd"] = round(rec["usd"] + usd, 4)
    rec["runs"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": models,
                        "container_s": {k: round(v, 1) for k, v in secs.items()}, "usd": round(usd, 4),
                        "rate_usd_h": USD_H, "errors": errs})
    (OUT / "spend.json").write_text(json.dumps(rec, indent=1) + "\n")
    old = json.loads((OUT / "meta.json").read_text()) if (OUT / "meta.json").exists() else {}
    old.update(metas)
    (OUT / "meta.json").write_text(json.dumps(old, indent=1) + "\n")
    print(json.dumps({"usd": round(usd, 3), "total_usd": rec["usd"], "errors": errs,
                      "meta": {k: {x: v[x] for x in ("class", "n_layers", "hidden", "extract_s", "load_s", "tokens",
                                                      "bos_added", "finite", "saturated")} for k, v in metas.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
