"""Stage 1 pilot, step 1: pull the dev prompts' activations from Modal (app anatta-selfmap) for the three models,
into the results folder outside git. Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026. Private
(lab/explore). Explore speed: sightings only.

Only prompts/dev.jsonl is read (its sha256 must match the manifest's current version); the held-out file is never
opened. Spend: each container is charged from launch to its last call plus the 60 s scaledown, at L40S $1.95/h plus
$0.60/h for cores and memory; a run whose upper estimate would pass the cap is refused before launch. The cost is
added to the spend folder (lab.toml, anatta.config)/<date>.json (tool "selfmap") and to the run folder's spend.json.

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
USD_H = 1.95 + 0.60
UPPER_S = 1500  # per container: start, load, extraction, transfer (generous)


def dev_rows(files: tuple[str, ...] = ("dev.jsonl",)) -> list[dict]:
    """The dev rows of the given prompt files, in order, each checked against the manifest's current sha256. Only dev
    files are accepted (never a held-out file)."""
    man = json.loads((HERE / "manifest.json").read_text())
    cur = man["history"][man["current"] - 1]
    rows = []
    for name in files:
        if "heldout" in name:
            raise SystemExit("held-out files are never read here")
        b = (HERE / "prompts" / name).read_bytes()
        if hashlib.sha256(b).hexdigest() != cur["files"][name]["sha256"]:
            raise SystemExit(f"{name} does not match the manifest's current version")
        rows += [json.loads(x) for x in b.decode().splitlines()]
    assert all(r["split"] in ("dev", "anchor") for r in rows)
    return rows


def main() -> None:
    global OUT
    import numpy as np
    from modal_selfmap import Extractor, app

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen_base,qwen_instruct,ministral")
    ap.add_argument("--cap", type=float, default=8.0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rows", default="dev.jsonl")
    ap.add_argument("--label", default="2026-11-02/selfmap-pilot/extract")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--upper-s", type=float, default=UPPER_S, help="per-container upper estimate for the cap check")
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    models = a.models.split(",")
    spent = json.loads((OUT / "spend.json").read_text())["usd"] if (OUT / "spend.json").exists() else 0.0
    upper = len(models) * a.upper_s / 3600 * USD_H
    if spent + upper > a.cap:
        raise SystemExit(f"refused: spent {spent:.2f} + upper {upper:.2f} > cap {a.cap}")
    rows = dev_rows(tuple(a.rows.split(",")))
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

    sem = threading.Semaphore(a.parallel)

    def guarded(k):
        with sem:
            one(k)

    with app.run():
        th = [threading.Thread(target=guarded, args=(k,)) for k in models]
        for t in th:
            t.start()
        for t in th:
            t.join()
    usd = sum(secs.values()) / 3600 * USD_H
    from scout.spend import add_spend

    from anatta.config import load

    add_spend(load().spend_dir, tool="selfmap", label=a.label, usd=usd, calls=len(models))
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
                                                      "bos_added", "finite", "saturated", "sanity")}
                               | {"load_n": {x: (v.get("load_info") or {}).get(x) for x in
                                             ("n_missing", "n_unexpected", "n_mismatched")}}
                               for k, v in metas.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
