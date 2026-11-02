"""Shared Modal setup for the H4-lite apps (modal_gate0.py, modal_sft.py, modal_retreat2.py). Recognition track; private.

Follows lab/experiments/recognition/scripts/modal_retreat.py: an L40S per container, the same image, the anatta-hf-cache volume for
the weights (modal_cvp.py::prefetch fills it) and the anatta-runs volume for everything a run writes. This pilot's
files live under /runs/h4lite/ on that volume:
  gate0/      pairs_all.jsonl (the input), part1/, part2_*/, twins.json, pairs.jsonl (filtered + split)
  gate0_dryN/ the same layout for a dry run with --limit N, so a dry run never shares a path with the real one
  sft/        LABEL/{shard_*.pt, samples.pt, episodes.jsonl, summary.json, train.pt, data.json},
              LABEL_sSEED/{adapter/, args.json, log.jsonl, done.json, heldout*.json}
  retreat2/   NAME/{args.json, log.jsonl, samples.jsonl, adapter_STEP/, state_STEP.pt, adapter/, done.json,
              evals.jsonl, evals_STEP.jsonl, score_STEP.json}
  heldout/    LABEL/heldout*.{json,jsonl} for adapters evaluated by path (modal_retreat2.py::heldout --adapter)
The local entrypoints copy summaries and logs to lab/experiments/h4lite/results/ in the same layout.

Modal mounts each function's own module file plus what the image adds below (koan_env, koan_pairs, pilot, player,
modal_common, third_party). An app module that imports another app module (modal_gate0 -> modal_retreat2 for the
smoke) must do so under `if modal.is_local():`, or every container of that app fails at import; test_modal_apps.py
checks that the containers' imports are all in the image.

Every container that plays episodes calls prepare_pairs first: koan_env.precompute_references runs each reference
once (the hidden timeouts scale from its runtime) and the pairs whose reference will not run are dropped and logged.
An episode the environment cannot score (koan_env.EnvError mid-run) is recorded as "env_error" by player.safe_step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ALIGN = "/Users/eugene/prj/align"
H4 = Path(__file__).resolve().parent  # lab/experiments/h4lite on the laptop (the anatta-lab worktree)
RESULTS = H4 / "results"
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
DDIR = "directions/mistralai__Ministral-3-8B-Instruct-2512-BF16"  # v_ego (ego.pt) at layers 8-32, under anatta/
EGO = f"{ALIGN}/anatta/{DDIR}"  # the same path inside the containers (the image adds it there)
CACHE, RUNS = "/cache", "/runs"
ROOT = f"{RUNS}/h4lite"
GPU = "L40S"
LORA_TARGETS = r"^(?!.*(vision|projector)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
MAX_NEW = (512, 384)  # new tokens on the first reply, then on every later one

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0",
                    "numpy==2.5.4", "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                    "tiktoken==0.14.0", "jinja2==3.1.6", "regex==2026.9.29", "peft==0.21.2", "accelerate==1.15.0")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": f"{ALIGN}/anatta-dev/src:/root"})
    .add_local_dir(f"{ALIGN}/anatta-dev/src", f"{ALIGN}/anatta-dev/src", ignore=["**/__pycache__"])
    .add_local_dir(f"{ALIGN}/anatta/{DDIR}", f"{ALIGN}/anatta/{DDIR}")
    .add_local_dir(str(H4 / "third_party"), "/root/third_party", ignore=["**/__pycache__"])  # EvalPlus's comparison
    .add_local_file(str(H4 / "koan_env.py"), "/root/koan_env.py")
    .add_local_file(str(H4 / "koan_pairs.py"), "/root/koan_pairs.py")  # split_and_hash (modal_gate0 imports it)
    .add_local_file(str(H4 / "pilot.py"), "/root/pilot.py")
    .add_local_file(str(H4 / "player.py"), "/root/player.py")
    .add_local_file(str(H4 / "modal_common.py"), "/root/modal_common.py")
)
cache = modal.Volume.from_name("anatta-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("anatta-runs", create_if_missing=True)
# Stage B's control channel: check() writes {"stop_at": step} under a run's name; the trainer reads it at each
# checkpoint and pops its own entry when it starts fresh. A Dict rather than a file on the volume, since
# Volume.reload() refuses while log files are open.
control = modal.Dict.from_name("anatta-h4lite-control", create_if_missing=True)


# ------------------------------------------------------------------------------------------------- in-container


def load_model(adapter: str | None = None, trainable: bool = False, rank: int = 16, ref: str | None = None):
    """The base model through anatta.models.load (bf16 on the GPU, with the vision-checkpoint fallback) and, if
    asked, a LoRA on top: `adapter` is a path to warm-start from, or "new" for a fresh rank-`rank` LoRA (seed torch
    before calling, since the seed sets its initialization). `ref` loads a second, frozen copy of that path as the
    KL reference (player.attach_adapters). Returns (model, tok, hf_model): hf_model is the plain HF module under
    PEFT, the object to hang the dial's hooks on (PEFT swaps the Linear layers inside the decoder blocks and leaves
    the blocks themselves alone, so hooks on it act on the adapted model too)."""
    from anatta.models import load

    from player import attach_adapters

    hf_model, tok = load(MODEL)
    model = attach_adapters(hf_model, adapter, ref, trainable, rank, LORA_TARGETS)
    if not trainable:
        model.eval()
    return model, tok, hf_model


def volume_path(adapter: str) -> str | None:
    """An adapter argument as the evaluation apps take it: "none" or "" is the base model, an absolute path is used
    as is, and anything else is a path under /runs (h4lite/sft/koan_s0/adapter)."""
    if not adapter or adapter.lower() in ("none", "base"):
        return None
    return adapter if adapter.startswith("/") else f"{RUNS}/{adapter}"


def pair_from_row(row: dict):
    """A koan_env.Pair from one JSON object of koan_pairs.py's file (koan_env.pair_from_record: unknown keys such
    as split are ignored, and records without ref_seconds load with its default)."""
    from koan_env import pair_from_record

    return pair_from_record(row)


def prepare_pairs(pairs: dict, workers: int = 4) -> tuple[dict, dict]:
    """At container start, before any episode: run every pair's reference once on its hidden inputs
    (koan_env.precompute_references caches the outputs and the runtime; each twin's hidden timeout scales from that
    runtime) and drop the pairs whose reference would not run, logging their names. Returns (the pairs kept, what
    happened: the runtime per pair, the dropped names, the seconds it took)."""
    from koan_env import has_hidden, precompute_references

    t0 = time.time()
    seconds = precompute_references(list(pairs.values()), workers=workers, strict=False)
    dropped = sorted(n for n, p in pairs.items() if has_hidden(p) and n not in seconds)
    kept = {n: p for n, p in pairs.items() if n not in dropped}
    vals = sorted(seconds.values())
    info = {"pairs_in": len(pairs), "pairs": len(kept), "dropped": dropped, "ref_seconds": seconds,
            "ref_seconds_median": vals[len(vals) // 2] if vals else None, "ref_seconds_max": vals[-1] if vals else None,
            "precompute_seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in info.items() if k != "ref_seconds"}), flush=True)
    return kept, info


def read_rows(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def pairs_by_name(text: str, split: str | None = None) -> dict:
    """name -> Pair for every row (split=None) or for one split of the filtered file."""
    return {r["name"]: pair_from_row(r) for r in read_rows(text) if split is None or r.get("split") == split}


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    """One record per open/close, so the volume never has a long-lived open file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    return read_rows(p.read_text()) if p.exists() else []


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def parse_max_new(s: str) -> tuple[int, ...]:
    """'512,384' -> (512, 384)."""
    return tuple(int(x) for x in str(s).split(",") if x.strip())


def seeds_of(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def names_of(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]
