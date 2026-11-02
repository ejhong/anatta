"""Write steering dial specs scaled to the layer's typical activation size.

Usage: python make_dials.py MODEL DIRECTIONS_DIR OUT_DIR

Strength is a percentage of the median per-token residual norm at the direction's best layer,
measured on the traits' questions. Position 0 is excluded: Qwen's first token carries massive
activations that would inflate a mean.
"""

import json
import statistics
import sys
from pathlib import Path

import torch
import yaml

from anatta.models import chat_ids, load
from anatta.probe import Direction, Trait, capture, list_traits

model_id, ddir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
m, tok = load(model_id)
device = next(m.parameters()).device

questions = []
for name in list_traits():
    tr = Trait.load(name)
    questions += (tr.eval_questions or tr.questions)[:4]

ego, calm = Direction.load(ddir / "ego.pt"), Direction.load(ddir / "calm.pt")
layers = sorted({ego.best_layer, calm.best_layer})
norms = {layer: [] for layer in layers}
with torch.no_grad(), capture(m, layers) as store:
    for q in questions:
        m(chat_ids(tok, [{"role": "user", "content": q}]).to(device))
        for layer in layers:
            norms[layer] += store[layer][0, 1:].float().norm(dim=-1).tolist()
scale = {layer: statistics.median(v) for layer, v in norms.items()}


def write(name, direction, layer, pct, control=None):
    item = {"direction": str(ddir / direction), "layer": layer, "mode": "steer",
            "alpha": round(pct / 100 * scale[layer], 4)}
    if control:
        item.update(control=control, seed=0)
    (out / f"{name}.yaml").write_text(yaml.safe_dump({"interventions": [item]}, sort_keys=False))


for pct in (-16, -8, 8, 16):
    write(f"ego_{'m' if pct < 0 else 'p'}{abs(pct):02d}", "ego.pt", ego.best_layer, pct)
for pct in (-16, 16):
    write(f"random_{'m' if pct < 0 else 'p'}16", "ego.pt", ego.best_layer, pct, control="random")
write("calm_p16", "calm.pt", calm.best_layer, 16)

(out / "scale.json").write_text(json.dumps({
    "model": model_id, "ego_layer": ego.best_layer, "calm_layer": calm.best_layer,
    "median_norm": scale, "mean_norm": {k: statistics.mean(v) for k, v in norms.items()},
    "tokens": {k: len(v) for k, v in norms.items()}, "questions": len(questions),
}, indent=2))
print(json.dumps({"ego_layer": ego.best_layer, "calm_layer": calm.best_layer, "median_norm": scale}))
