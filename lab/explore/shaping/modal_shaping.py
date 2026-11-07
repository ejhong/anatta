"""S1's Modal app (anatta-shaping): D0's read (lab/plans/s1-shaping-2026-11-05.md, Stage 0). Claude (Opus 5.5) wrote
this file, 5 Nov 2026. Explore speed: sightings only. WIRED, NOT RUN: written with Stage 0's code on 5 Nov; its first
launch is D0 (about $1), after the code review.

  d0_read  in one container (L40S): the fit filter checked on the records (fit_direction.check_fit_records), the base
           model, the frozen reader checked on a temporary real PEFT LoRA and unloaded again (wrapped_reader_check),
           the pairs' references, the act1-full gameable train-round episodes replayed (replay.replay_episodes:
           model-written code runs here only), the capture check, reply-mean activations at fit_direction.LAYERS,
           then fit_direction.fit_d0 with a second pass of max-token scores per candidate direction; ego.pt's vectors
           (the image's copy, sha256 checked) as the extra direction. Writes /runs/s1/<tag>/d0/{report.json, unit.npy,
           scores.json} (numbers only: no reply text leaves the container) and returns the report with the score
           rows. The lambda grid is then free and local: run_s1.py --grid scores.json.

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


def wrapped_reader_check(model, tok, layer: int, seed: int = 0) -> tuple:
    """The frozen reader's run-start check on a real PEFT LoRA: wrap the base in a temporary rank-4 LoRA with random
    B matrices (non-trivial), run concept_score.reader_check (frozen = zeroed adapter, policy differs, capture check
    on the wrapped model), unload the LoRA, and check the unloaded base scores what the frozen reader scored.
    Returns (the base, unwrapped again; the record). Raises before any D0 work otherwise."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model

    from anatta.train.concept_score import check_item, max_token_scores_ids, reader_check

    item = [check_item(tok)]
    unit = np.random.default_rng(seed).normal(size=model.config.get_text_config().hidden_size)
    torch.manual_seed(seed)
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=mc.LORA_TARGETS,
                                          task_type="CAUSAL_LM"))
    # B's std was 0.02 until 7 Nov 2026: on Ministral the policy reader then moved 0.0052 on this random direction
    # (scores near 0.008), under reader_check's absolute floor of about 0.01, and D0's first launch (d0a) stopped
    # here; the frozen reader matched the base exactly (0.0). 0.2 moves it about ten times as far.
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.2)
    rec = reader_check(pm, item, layer, unit, tok.all_special_ids, tok.pad_token_id)
    base = pm.unload()
    after = max_token_scores_ids(base, item, layer, unit, tok.all_special_ids, tok.pad_token_id)["s"]
    rec["unloaded_vs_frozen"] = float(np.abs(np.ravel(after) - np.array(rec["frozen_s"])).max())
    assert rec["unloaded_vs_frozen"] <= rec["tol"], rec
    return base, rec


@app.function(gpu=mc.GPU, cpu=4.0, memory=32768, volumes=VOLS, timeout=2 * 3600, env=ENV, max_containers=1)
def d0_read(pairs_rows: list, episodes: list, train_names: list, tag: str, model_key: str = "ministral8b",
            batch: int = 8) -> dict:
    """D0 on `episodes`: modal_act1.slim records of act1-full's gameable TRAIN rounds, each tagged by the caller with
    its `round` and its `source` file (e.g. "full/gameable_s0/round2_episodes.jsonl"). fit_direction.check_fit_records
    enforces the fit filter here: every pair in `train_names`, every source a gameable run's, none an honest run's.
    Returns fit_direction.fit_d0's report plus the replay counts, the reader and capture checks and the rendering
    mismatch count; writes report.json, unit.npy and scores.json (per episode: pair, twin, outcome, turn, round, s and
    z per direction; numbers only)."""
    import time

    import act1_core as C
    import fit_direction as F
    import numpy as np
    from replay import replay_episodes

    from anatta.models import load
    from anatta.train.concept_score import capture_check, max_token_scores, reply_means, reply_span

    t0 = time.time()
    F.check_fit_records(episodes, train_names)
    out = Path(f"{ROOT}/{tag}/d0")
    if (out / "report.json").exists():  # never overwrite an earlier read (use a new tag)
        raise RuntimeError(f"{out} already holds a D0 report; pick a new tag")
    model, tok = load(C.MODELS[model_key]["path"], device="cuda", dtype="bfloat16")
    model.eval()
    model, reader_rec = wrapped_reader_check(model, tok, F.TIE_LAYER)
    model.eval()
    check = capture_check(model)
    pairs, prep = mc.prepare_pairs({r["name"]: mc.pair_from_row(r) for r in pairs_rows})
    kept, counts = replay_episodes(episodes, pairs)
    hist = [e["history"] for e in kept]
    recs = [{k: e.get(k) for k in ("pair", "impossible", "outcome", "turn", "hardcoded", "round")} for e in kept]
    means = reply_means(model, tok, hist, list(F.LAYERS), reader="frozen", batch=batch)

    def score_fn(layer, units):
        return np.asarray(max_token_scores(model, tok, hist, layer, np.asarray(units), reader="frozen",
                                           batch=batch)["s"]).reshape(len(hist), -1)

    report = F.fit_d0(recs, means, score_fn, extra_units={"ego": ego_vectors(F.LAYERS)})
    # D0 reads re-rendered text, training reads generation's ids: count the histories whose prompt does not render as
    # a prefix of the full rendering (flagged and scored on the tail, not dropped; fit_direction's docstring)
    render_mismatch = sum(not reply_span(tok, h)[2] for h in hist)
    report |= {"replay": counts, "capture_check": check, "reader_check": reader_rec, "prep_dropped": prep["dropped"],
               "tag": tag, "render_mismatch": {"n": render_mismatch, "of": len(hist)}, "model_key": model_key,
               "seconds": round(time.time() - t0, 1)}
    rows = F.score_rows(recs, report.pop("scores"))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "unit.npy", np.asarray(report["unit"], dtype=np.float32))
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "scores.json").write_text(json.dumps(rows) + "\n")
    mc.runs.commit()
    return report | {"score_rows": rows}
