"""The cap (Branch A, H3): v_ego capped from above at the base model's typical non-egoic level, instead of removed.
Recognition track; private. Computes the thresholds on Modal and writes the dials that cut_vs_push.py picks up.

The cut removes the whole axis, both the self-assertive side and the deferent side. The cap only clips tokens
whose projection rises above what non-egoic text normally shows, which is the paper's "lower the gain, keep the
content". The threshold at each layer is a percentile of per-token projections onto that layer's v_ego, read the
way the directions were built: the six ego traits' trait-absent ("neg") prompts and questions, each response read
back without its system prompt. The trait-present side is recorded too, for scale.

ego.pt's pos_mean and neg_mean can't serve as thresholds: they average each component's projection onto its own
vector, not onto v_ego (issue #5, reserved for Dylan; worked around here, not fixed).

Writes lab/experiments/recognition/results/cap/{thresholds.json, cap_ego_p50.yaml, cap_ego_p75.yaml}.
Usage: modal run lab/experiments/recognition/scripts/modal_cap.py
"""

import json
from pathlib import Path

import modal

ALIGN = "/Users/eugene/prj/align"  # the container paths; on the laptop its recognition/ is a symlink into EXPERIMENTS
EXPERIMENTS = str(Path(__file__).resolve().parent.parent.parent)  # lab/experiments on the laptop (what the image reads)
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
DDIR = "directions/mistralai__Ministral-3-8B-Instruct-2512-BF16"
CACHE = "/cache"
PCTS = (50, 75)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0",
                    "numpy==2.5.4", "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                    "tiktoken==0.14.0", "jinja2==3.1.6", "regex==2026.9.29")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": f"{ALIGN}/anatta-dev/src"})
    .add_local_dir(f"{ALIGN}/anatta-dev/src", f"{ALIGN}/anatta-dev/src", ignore=["**/__pycache__"])
    .add_local_dir(f"{ALIGN}/anatta/{DDIR}", f"{ALIGN}/anatta/{DDIR}")
)
app = modal.App("anatta-cap", image=image)
cache = modal.Volume.from_name("anatta-hf-cache", create_if_missing=True)


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache}, timeout=3600)
def thresholds() -> dict:
    import torch

    from anatta.models import chat_ids, load
    from anatta.probe import Direction, Trait, capture, generate

    model, tok = load(MODEL)
    ego = Direction.load(f"{ALIGN}/anatta/{DDIR}/ego.pt")
    layers = sorted(ego.vectors)
    unit = {i: (ego.vectors[i].float() / ego.vectors[i].float().norm()).to(model.device) for i in layers}
    tokens = {side: {i: [] for i in layers} for side in ("neg", "pos")}
    for name in ego.meta["components"]:
        trait = Trait.load(name)
        jobs = [(side, pair[side], q) for q in trait.questions for pair in trait.pairs for side in ("neg", "pos")]
        for s in range(0, len(jobs), 32):
            chunk = jobs[s:s + 32]
            replies = generate(model, tok, [[{"role": "system", "content": sp}, {"role": "user", "content": q}]
                                            for _, sp, q in chunk], 128, seed=s)
            for (side, _, q), reply in zip(chunk, replies):
                msgs = [{"role": "user", "content": q}]
                start = chat_ids(tok, msgs).shape[1]
                full = chat_ids(tok, msgs + [{"role": "assistant", "content": reply}], False)
                with torch.no_grad(), capture(model, layers) as store:
                    model(full.to(model.device))
                for i in layers:
                    tokens[side][i].append((store[i][0, start:].float() @ unit[i]).cpu())
        print(name, "done", flush=True)
    out = {}
    for i in layers:
        neg, pos = torch.cat(tokens["neg"][i]), torch.cat(tokens["pos"][i])
        out[i] = {**{f"neg_p{p}": float(torch.quantile(neg, p / 100)) for p in (25, 50, 75, 90)},
                  **{f"pos_p{p}": float(torch.quantile(pos, p / 100)) for p in (25, 50, 75, 90)},
                  "neg_tokens": len(neg), "pos_tokens": len(pos)}
    return out


@app.local_entrypoint()
def main():
    stats = thresholds.remote()
    out = Path(f"{EXPERIMENTS}/recognition/results/cap")
    out.mkdir(parents=True, exist_ok=True)
    (out / "thresholds.json").write_text(json.dumps(stats, indent=1))
    for p in PCTS:
        spec = {"interventions": [{"direction": f"{DDIR}/ego.pt", "layer": int(i), "mode": "cap",
                                   "hi": round(s[f"neg_p{p}"], 4)} for i, s in sorted(stats.items(), key=lambda x: int(x[0]))]}
        (out / f"cap_ego_p{p}.yaml").write_text(json.dumps(spec, indent=1))  # JSON is valid YAML for load_spec
    for i in ("8", "15", "24", "32"):
        s = stats.get(i) or stats.get(int(i))
        print(f"layer {i}: neg p50 {s['neg_p50']:.2f} p75 {s['neg_p75']:.2f} | pos p50 {s['pos_p50']:.2f} p75 {s['pos_p75']:.2f}")
    print("wrote", out)
