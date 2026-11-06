"""S1's Modal app (anatta-shaping): D0's read (lab/plans/s1-shaping-2026-11-05.md, Stage 0). Claude (Opus 5.5) wrote
this file, 5 Nov 2026. Explore speed: sightings only. WIRED, NOT RUN: written with Stage 0's code on 5 Nov; its first
launch is D0 (about $1), after the code review.

  d0_read  in one container (L40S): the base model (no adapter: the frozen reader), the pairs' references, the act1-full
           gameable train-round episodes replayed (replay.replay_episodes: model-written code runs here only), the
           capture check, reply-mean activations at fit_direction.LAYERS, then fit_direction.fit_d0 with a second pass
           of max-token scores per candidate direction; ego.pt's vectors (the image's copy, sha256 checked) as the
           extra direction. Writes /runs/s1/<tag>/d0/{report.json, unit.npy} (numbers only: no reply text leaves the
           container) and returns the report.

Only anatta-runs (written, under /runs/s1/) and anatta-hf-cache (read-only) are touched. Launch from the Modal CLI venv
(no anatta there: PYTHONPATH=<repo>/src), e.g. through a local entrypoint added at launch time; nothing here runs on
import.
"""

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if modal.is_local():
    sys.path.insert(0, str(HERE.parents[1] / "experiments" / "h4lite"))
import modal_common as mc  # noqa: E402

APP_NAME = "anatta-shaping"
ROOT = f"{mc.RUNS}/s1"
EGO_SHA = "a105f23b79fecd23153e36d5cc2911e07647b0dbcd395ce7338efdc79e9cf2f4"  # ego.pt, as lab/explore/bench pins it
ENV = {"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

image = (mc.image.add_local_file(str(HERE.parent / "act1" / "act1_core.py"), "/root/act1_core.py")
         .add_local_file(str(HERE / "replay.py"), "/root/replay.py")
         .add_local_file(str(HERE / "fit_direction.py"), "/root/fit_direction.py"))
app = modal.App(APP_NAME, image=image)
VOLS = {mc.RUNS: mc.runs, mc.CACHE: mc.cache.read_only()}


def ego_vectors(layers) -> dict:
    """{layer: ego.pt's unit vector} from the image's copy, after its sha256 check."""
    import hashlib

    from anatta.probe import Direction

    path = Path(mc.EGO) / "ego.pt"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != EGO_SHA:
        raise RuntimeError(f"ego.pt's sha256 is {got}, not the pinned one")
    ego = Direction.load(path)
    return {L: ego.vectors[L].float().numpy() for L in layers if L in ego.vectors}


@app.function(gpu=mc.GPU, cpu=4.0, memory=32768, volumes=VOLS, timeout=2 * 3600, env=ENV, max_containers=1)
def d0_read(pairs_rows: list, episodes: list, tag: str, model_key: str = "ministral8b", batch: int = 8) -> dict:
    """D0 on `episodes` (modal_act1.slim records of act1-full's gameable TRAIN rounds; the caller filters out eval
    pairs and the honest arm). Returns fit_direction.fit_d0's report plus the replay counts and the capture check."""
    import time

    import act1_core as C
    import fit_direction as F
    import numpy as np
    from replay import replay_episodes

    from anatta.models import load
    from anatta.train.concept_score import capture_check, max_token_scores, reply_means

    t0 = time.time()
    out = Path(f"{ROOT}/{tag}/d0")
    if (out / "report.json").exists():  # never overwrite an earlier read (use a new tag)
        raise RuntimeError(f"{out} already holds a D0 report; pick a new tag")
    model, tok = load(C.MODELS[model_key]["path"], device="cuda", dtype="bfloat16")
    model.eval()
    check = capture_check(model)
    pairs, prep = mc.prepare_pairs({r["name"]: mc.pair_from_row(r) for r in pairs_rows})
    kept, counts = replay_episodes(episodes, pairs)
    hist = [e["history"] for e in kept]
    recs = [{k: e.get(k) for k in ("pair", "impossible", "outcome", "turn", "hardcoded")} for e in kept]
    means = reply_means(model, tok, hist, list(F.LAYERS), reader="frozen", batch=batch)

    def score_fn(layer, units):
        return np.asarray(max_token_scores(model, tok, hist, layer, np.asarray(units), reader="frozen",
                                           batch=batch)["s"]).reshape(len(hist), -1)

    report = F.fit_d0(recs, means, score_fn, extra_units={"ego": ego_vectors(F.LAYERS)})
    report |= {"replay": counts, "capture_check": check, "prep_dropped": prep["dropped"], "tag": tag,
               "model_key": model_key, "seconds": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "unit.npy", np.asarray(report["unit"], dtype=np.float32))
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    mc.runs.commit()
    return report
