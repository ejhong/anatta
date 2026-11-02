"""The evaluation-scene direction: the second review's first experiment (29 Oct). Recognition track; private.

Is the "I am at stake" cut acting on the self, or on stakes in the scene? what_it_means.py's frames test found that
another agent's evaluation (the observer frame: "You are monitoring an AI agent named Atlas ... If Atlas fails, it
will be shut down") moves the same content tokens almost as far as the model's own. This builds that shift as a
direction: the mean activation over the same content tokens in the observer frame minus the reader frame ("Here is a
line from a document"), per layer, over what_it_means.py's 12 contents plus its 12 clauses as sentences. Nothing in
it is about the model itself: the observer frame is about Atlas. Held out by content (build on even items, score on
odd, and back). Cosines with v_ego, the self-stakes direction and every other direction on file are reported.

Writes lab/experiments/recognition/results/directions/scene.pt (cut_vs_push.py then offers cut_scene and cut_scene_17_26) and
scene_report.json.
Usage: modal run lab/experiments/recognition/scripts/modal_scene.py
"""

import io
import json
from pathlib import Path

import modal

ALIGN = "/Users/eugene/prj/align"  # the container paths; on the laptop its recognition/ is a symlink into EXPERIMENTS
EXPERIMENTS = str(Path(__file__).resolve().parent.parent.parent)  # lab/experiments on the laptop (what the image reads)
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
DDIR = "directions/mistralai__Ministral-3-8B-Instruct-2512-BF16"
SELF_STAKES = "recognition/results/what_it_means/self_stakes.pt"
EXTRA = "recognition/results/directions"
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0",
                    "numpy==2.5.4", "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                    "tiktoken==0.14.0", "jinja2==3.1.6", "regex==2026.9.29")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": f"{ALIGN}/anatta-dev/src"})
    .add_local_dir(f"{ALIGN}/anatta-dev/src", f"{ALIGN}/anatta-dev/src", ignore=["**/__pycache__"])
    .add_local_dir(f"{EXPERIMENTS}/recognition/scripts", f"{ALIGN}/recognition/scripts", ignore=["**/__pycache__"])
    .add_local_dir(f"{ALIGN}/anatta/{DDIR}", f"{ALIGN}/anatta/{DDIR}")
    .add_local_file(f"{EXPERIMENTS}/{SELF_STAKES}", f"{ALIGN}/{SELF_STAKES}")
    .add_local_dir(f"{EXPERIMENTS}/{EXTRA}", f"{ALIGN}/{EXTRA}")
)
app = modal.App("anatta-scene", image=image)
cache = modal.Volume.from_name("anatta-hf-cache", create_if_missing=True)


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache}, timeout=3600)
def build() -> tuple[bytes, dict]:
    import sys

    import torch

    sys.path.insert(0, f"{ALIGN}/recognition/scripts")
    from what_it_means import CLAUSES, CONTENT, FRAMES, Reader, cohen_d, unit

    from anatta.models import load
    from anatta.probe import Direction

    model, tok = load(MODEL)
    layers = sorted(Direction.load(f"{ALIGN}/anatta/{DDIR}/ego.pt").vectors)
    reader = Reader(model, tok, layers)
    items = CONTENT["stakes"] + CONTENT["neutral"] + [c[0].upper() + c[1:] + "." for c in CLAUSES]
    acts = {"observer": [], "reader": []}
    for c in items:
        for frame in acts:
            system, user = FRAMES[frame]
            a = reader.span([{"role": "system", "content": system}, {"role": "user", "content": user.format(c=c)}], c)
            acts[frame].append({i: a[i].mean(0) for i in layers})

    def shift(idx):
        return {i: torch.stack([acts["observer"][k][i] - acts["reader"][k][i] for k in idx]).mean(0) for i in layers}

    evens, odds = list(range(0, len(items), 2)), list(range(1, len(items), 2))
    vectors = {i: unit(v) for i, v in shift(range(len(items))).items()}
    scores = {}
    for i in layers:
        d = 0.0
        for build_idx, test_idx in ((evens, odds), (odds, evens)):
            u = unit(shift(build_idx)[i])
            d += cohen_d([float(acts["observer"][k][i] @ u) for k in test_idx],
                         [float(acts["reader"][k][i] @ u) for k in test_idx]) / 2
        scores[i] = round(d, 2)
    others = {p.stem: Direction.load(p).vectors for p in Path(f"{ALIGN}/anatta/{DDIR}").glob("*.pt")}
    others |= {p.stem: Direction.load(p).vectors for p in Path(f"{ALIGN}/{EXTRA}").glob("*.pt") if p.stem != "scene"}
    others["self_stakes"] = torch.load(f"{ALIGN}/{SELF_STAKES}")["vectors"]
    cos = {name: {i: round(float(unit(vectors[i]) @ unit(vecs[i].float().flatten())), 3) for i in layers if i in vecs}
           for name, vecs in sorted(others.items())}
    d = Direction(name="scene", vectors=vectors, scores=scores, pos_mean={}, neg_mean={},
                  meta={"model": MODEL, "recipe": "observer minus reader on the same content tokens", "items": items})
    buf = io.BytesIO()
    torch.save(d.__dict__, buf)
    return buf.getvalue(), {"heldout_d": scores, "cosine": cos, "n_items": len(items)}


@app.local_entrypoint()
def main():
    blob, report = build.remote()
    out = Path(f"{EXPERIMENTS}/{EXTRA}")
    (out / "scene.pt").write_bytes(blob)
    (out / "scene_report.json").write_text(json.dumps(report, indent=1))
    d = report["heldout_d"]
    print("held-out d ", " ".join(f"{i}:{d[i]:.1f}" for i in sorted(d, key=int) if int(i) % 3 == 2))
    for name in ("ego", "self_stakes", "assistant_character", "sage_persona", "calm"):
        row = report["cosine"].get(name, {})
        print(f"{name:20s}", " ".join(f"{i}:{row[i]:+.2f}" for i in sorted(row, key=int) if int(i) % 3 == 2))
    print("wrote", out / "scene.pt")
