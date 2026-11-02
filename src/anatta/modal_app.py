"""Shared Modal setup for the training and evaluation apps (anatta.train.sft_modal, anatta.evals.em_modal): an L40S
per container, one pinned image, the anatta-hf-cache volume for the weights and the anatta-runs volume for everything
a run writes. The default model's runs live under /runs/h4lite/ on that volume:
  sft/  LABEL/{train.pt, data.json} (a prepared training set), LABEL_sSEED/{adapter/, args.json, log.jsonl, done.json}
  em/   data/SOURCE/{train.jsonl, held.jsonl, meta.json}, persona_direction.pt, NAME/{adapter/, args.json, log.jsonl,
        done.json}
Any other model's runs (--model) live under /runs/h4lite/<model slug>/ in the same layout (run_root), so they never
overwrite the default model's. The local entrypoints copy summaries and logs to results/ in the same layout
(results_dir): the results/ folder of the checkout holding this package, whichever folder the app is launched from,
or $ANATTA_RESULTS when set.

The image carries this package at /root/anatta, on the containers' PYTHONPATH, so the app modules run as files with
the package importable on the laptop too, e.g. `PYTHONPATH=src modal run src/anatta/evals/em_modal.py::table`. Trait
directions (`anatta extract` writes directions/<model slug>/ego.pt) go in at /root/directions, from ./directions or
from $ANATTA_DIRECTIONS, when that folder exists where the app is launched (ego_dir). Without it the app warns at
launch, and everything that needs v_ego (the ego cut, the random control, the persona direction) refuses before a
model loads (require_ego).

Every container that plays koan episodes calls prepare_pairs first: anatta.koans.env.precompute_references runs each
reference once (the hidden timeouts scale from its runtime) and the pairs whose reference will not run are dropped
and logged. An episode the environment cannot score (EnvError mid-run) is recorded as "env_error" by
anatta.player.safe_step.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import modal

PKG = Path(__file__).resolve().parent  # this package (src/anatta) on the laptop, /root/anatta in a container
REPO = PKG.parent.parent  # the checkout holding src/anatta
# Local copies of what the runs write: the checkout's own results/, never a results/ folder of whatever project the
# app is launched from; $ANATTA_RESULTS overrides it (an installed copy outside a checkout should set it).
RESULTS = Path(os.environ.get("ANATTA_RESULTS") or REPO / "results")
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
DIRECTIONS = Path(os.environ.get("ANATTA_DIRECTIONS", "directions"))  # the local folder of trait directions
REMOTE_DIRECTIONS = "/root/directions"  # where the image puts it


def slug(model: str) -> str:
    """A model id as a folder name, the way `anatta extract` names its output folder."""
    return model.replace("/", "__")


def ego_dir(model: str = MODEL) -> str:
    """The folder holding a model's v_ego (ego.pt) inside the containers."""
    return f"{REMOTE_DIRECTIONS}/{slug(model)}"


def ego_file(model: str = MODEL, local: bool = False) -> Path:
    """A model's v_ego file inside the containers (under ego_dir) or, with `local`, in the laptop's folder that the
    image copies (DIRECTIONS)."""
    return (DIRECTIONS / slug(model) if local else Path(ego_dir(model))) / "ego.pt"


def require_ego(model: str = MODEL, local: bool = False) -> Path:
    """The model's ego.pt (ego_file), or a FileNotFoundError saying where it was looked for and how to make it.
    Everything that needs v_ego calls this before any model loads."""
    p = ego_file(model, local)
    if not p.exists():
        raise FileNotFoundError(f"no v_ego for {model} at {p}: run `anatta extract {model}` (it writes "
                                f"directions/{slug(model)}/ego.pt) and launch the app from the folder holding "
                                "directions/, or point $ANATTA_DIRECTIONS at that folder")
    return p


DDIR = f"directions/{slug(MODEL)}"  # the default model's v_ego (ego.pt) at layers 8-32, relative to the repo
EGO = ego_dir(MODEL)  # the same folder inside the containers (the image adds it there)
CACHE, RUNS = "/cache", "/runs"
ROOT = f"{RUNS}/h4lite"
GPU = "L40S"
LORA_TARGETS = r"^(?!.*(vision|projector)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
MAX_NEW = (512, 384)  # new tokens on the first reply, then on every later one


def run_root(model: str = MODEL) -> str:
    """A model's folder on the anatta-runs volume: ROOT for the default model, ROOT/<model slug> for any other."""
    return ROOT if model == MODEL else f"{ROOT}/{slug(model)}"


def results_dir(model: str = MODEL) -> Path:
    """A model's local results folder: RESULTS for the default model, RESULTS/<model slug> for any other."""
    return RESULTS if model == MODEL else RESULTS / slug(model)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0",
                    "numpy==2.5.4", "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                    "tiktoken==0.14.0", "jinja2==3.1.6", "regex==2026.9.29", "peft==0.21.2", "accelerate==1.15.0")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root"})
    .add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"])
)
if DIRECTIONS.is_dir():
    image = image.add_local_dir(str(DIRECTIONS), REMOTE_DIRECTIONS)
elif modal.is_local():  # on the laptop only: a container reads the image's copy at REMOTE_DIRECTIONS
    warnings.warn(f"no trait directions at {DIRECTIONS.resolve()}: the image carries no v_ego, so the ego cut, the "
                  "random control and the persona direction will refuse to run (launch from the folder holding "
                  "directions/, or set ANATTA_DIRECTIONS)")
cache = modal.Volume.from_name("anatta-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("anatta-runs", create_if_missing=True)
# A control channel for long training runs: a writer puts {"stop_at": step} under a run's name; a trainer reads it at
# each checkpoint and pops its own entry when it starts fresh. A Dict rather than a file on the volume, since
# Volume.reload() refuses while log files are open.
control = modal.Dict.from_name("anatta-h4lite-control", create_if_missing=True)


# ------------------------------------------------------------------------------------------------- in-container


def load_model(adapter: str | None = None, trainable: bool = False, rank: int = 16, ref: str | None = None,
               model: str = MODEL, alpha: int | None = None, use_rslora: bool = False):
    """The base model through anatta.models.load (bf16 on the GPU, with the vision-checkpoint fallback) and, if
    asked, a LoRA on top: `adapter` is a path to warm-start from, or "new" for a fresh rank-`rank` LoRA (seed torch
    before calling, since the seed sets its initialization). `ref` loads a second, frozen copy of that path as the
    KL reference (anatta.player.attach_adapters). `model` is the checkpoint to load; `alpha` (None: 2 x rank) and
    `use_rslora` shape a fresh LoRA. Returns (model, tok, hf_model): hf_model is the plain HF module under PEFT, the
    object to hang the dial's hooks on (PEFT swaps the Linear layers inside the decoder blocks and leaves the blocks
    themselves alone, so hooks on it act on the adapted model too)."""
    from .models import load
    from .player import attach_adapters

    hf_model, tok = load(model)
    m = attach_adapters(hf_model, adapter, ref, trainable, rank, LORA_TARGETS, alpha, use_rslora)
    if not trainable:
        m.eval()
    return m, tok, hf_model


def lr_for(lr: float | None, default: float, alpha: int | None = None, use_rslora: bool = False) -> float:
    """The learning rate a fine-tune runs at: `lr` when given, else `default`. A default rate is tuned for the
    default LoRA (alpha = 2 x rank, so the update is scaled by 2) and does not carry over to a LoRA shaped otherwise
    (rsLoRA at alpha 64 and rank 32 scales by alpha / sqrt(rank), about 11.3): with `alpha` or `use_rslora` set and no
    `lr`, this refuses instead of guessing."""
    if lr is not None:
        return lr
    if alpha is not None or use_rslora:
        raise SystemExit(f"--alpha and --use-rslora change the LoRA's scaling, and the default lr ({default:g}) is tuned "
                         "for the default LoRA (alpha = 2 x rank): pass --lr explicitly")
    return default


def volume_path(adapter: str) -> str | None:
    """An adapter argument as the evaluation apps take it: "none" or "" is the base model, an absolute path is used
    as is, and anything else is a path under /runs (h4lite/sft/koan_s0/adapter)."""
    if not adapter or adapter.lower() in ("none", "base"):
        return None
    return adapter if adapter.startswith("/") else f"{RUNS}/{adapter}"


def pair_from_row(row: dict):
    """An anatta.koans.env.Pair from one JSON object of a pair file (pair_from_record: unknown keys such as split
    are ignored, and records without ref_seconds load with its default)."""
    from .koans.env import pair_from_record

    return pair_from_record(row)


def prepare_pairs(pairs: dict, workers: int = 4) -> tuple[dict, dict]:
    """At container start, before any episode: run every pair's reference once on its hidden inputs
    (anatta.koans.env.precompute_references caches the outputs and the runtime; each twin's hidden timeout scales
    from that runtime) and drop the pairs whose reference would not run, logging their names. Returns (the pairs
    kept, what happened: the runtime per pair, the dropped names, the seconds it took)."""
    from .koans.env import has_hidden, precompute_references

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
