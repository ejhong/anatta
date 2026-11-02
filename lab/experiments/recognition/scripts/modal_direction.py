"""Extract a trait direction on Modal and compare it with v_ego and the self-stakes direction. Recognition track; private.

Same recipe as the repo's directions (anatta.probe.extract: layers 8-32, 128-token answers, unjudged, read back
without the system prompt, split-half Cohen's d per layer). The trait file can live outside the repo
(lab/experiments/recognition/traits/). First use: assistant_character, after the Fable review asked whether v_ego is the known
"Assistant Axis".

Writes lab/experiments/recognition/results/directions/NAME.pt (cut_vs_push.py adds it as the arm cut_NAME) and NAME_cosines.json.
Usage: modal run lab/experiments/recognition/scripts/modal_direction.py --trait recognition/traits/assistant_character.yaml
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
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0",
                    "numpy==2.5.4", "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                    "tiktoken==0.14.0", "jinja2==3.1.6", "regex==2026.9.29")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": f"{ALIGN}/anatta-dev/src"})
    .add_local_dir(f"{ALIGN}/anatta-dev/src", f"{ALIGN}/anatta-dev/src", ignore=["**/__pycache__"])
    .add_local_dir(f"{ALIGN}/anatta/{DDIR}", f"{ALIGN}/anatta/{DDIR}")
    .add_local_dir(f"{EXPERIMENTS}/recognition/traits", f"{ALIGN}/recognition/traits")
    .add_local_file(f"{EXPERIMENTS}/{SELF_STAKES}", f"{ALIGN}/{SELF_STAKES}")
)
app = modal.App("anatta-direction", image=image)
cache = modal.Volume.from_name("anatta-hf-cache", create_if_missing=True)


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache}, timeout=3600)
def build(trait_path: str) -> tuple[bytes, dict]:
    import torch

    from anatta.models import load
    from anatta.probe import Direction, Trait, extract

    model, tok = load(MODEL)
    trait = Trait.load(trait_path)
    ego = Direction.load(f"{ALIGN}/anatta/{DDIR}/ego.pt")
    layers = sorted(ego.vectors)
    d = extract(model, tok, trait, layers=layers)
    others = {p.stem: Direction.load(p).vectors for p in Path(f"{ALIGN}/anatta/{DDIR}").glob("*.pt")}
    others["self_stakes"] = torch.load(f"{ALIGN}/{SELF_STAKES}")["vectors"]
    unit = lambda v: v.float().flatten() / v.float().norm()  # noqa: E731
    cos = {name: {i: round(float(unit(d.vectors[i]) @ unit(vecs[i])), 3) for i in layers if i in vecs}
           for name, vecs in sorted(others.items())}
    buf = io.BytesIO()
    torch.save(d.__dict__, buf)
    return buf.getvalue(), {"trait": trait.name, "split_half_d": {i: round(s, 2) for i, s in d.scores.items()},
                            "cosine": cos, "examples": d.meta["examples"]}


@app.local_entrypoint()
def main(trait: str):
    blob, report = build.remote(f"{ALIGN}/{trait}" if not trait.startswith("/") else trait)
    out = Path(f"{EXPERIMENTS}/recognition/results/directions")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{report['trait']}.pt").write_bytes(blob)
    (out / f"{report['trait']}_cosines.json").write_text(json.dumps(report, indent=1))
    for name in ("ego", "self_stakes", "sage_persona", "calm", "approval_seeking", "self_preservation"):
        row = report["cosine"].get(name, {})
        print(f"{name:18s}", " ".join(f"{i}:{row[i]:+.2f}" for i in sorted(row, key=int) if int(i) % 4 == 0))
    d = report["split_half_d"]
    print("split-half d", " ".join(f"{i}:{d[i]:.1f}" for i in sorted(d, key=int) if int(i) % 4 == 0))
    print("wrote", out)
