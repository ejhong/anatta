"""Shared Modal conventions for the H4-lite evaluation apps (the image, volumes, paths of modal_cvp.py). Private.

Every app mirrors the laptop's files at the same absolute paths in the container (the recognition files are read
from lab/experiments/recognition and land at ALIGN/recognition, the path the scripts hard-code and, on the laptop,
a symlink to that folder), so the recognition scripts and anatta-dev run unchanged, and mounts this folder (the
.py and .md files only: results/, .venv/, data/, third_party/, caches and every JSONL stay out, see _skip; the
count goes to the terminal at app start) so fishery_play.py,
fishery_grade.py, koan_env.py, player.py and the other pure modules import in both places. Adapters come from the
anatta-runs volume, where the training half saves
them: Stage A at /runs/h4lite/sft/ARM_sSEED/adapter (koan_s0, noexit_s0, koan_cut_s0 for the under-the-cut arm) and
Stage B checkpoints at /runs/h4lite/retreat2/ARM_sSEED/adapter_STEP. A --adapter name with a slash is taken as is
under /runs (so --adapter h4lite/sft/koan_s0/adapter); a flat NAME means /runs/NAME/adapter (modal_retreat.py's
layout). File names shorten the path with fishery_grade.adapter_tag (h4lite/sft/koan_s0/adapter -> sft__koan_s0).
"""

from pathlib import Path

import modal

ALIGN = "/Users/eugene/prj/align"  # the container paths; on the laptop ALIGN/recognition is a symlink to RECOGNITION
H4 = str(Path(__file__).resolve().parent)  # lab/experiments/h4lite on the laptop; the image mounts it at the same path
EXPERIMENTS = str(Path(H4).parent)  # lab/experiments: the recognition track sits beside this folder
RECOGNITION = f"{EXPERIMENTS}/recognition"  # its scripts/, results/, traits/ on the laptop (ALIGN/recognition in a container)
RESULTS = Path(H4) / "results"
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
DDIR = "directions/mistralai__Ministral-3-8B-Instruct-2512-BF16"
DIALS = "results/2026-10-28-viability/dials_m8"
VSCRIPTS = "results/2026-10-28-viability/scripts"  # api_bench.py, imported by commons_variants.py
SELF_STAKES = "recognition/results/what_it_means/self_stakes.pt"
CAPS = "recognition/results/cap"  # cap dials (modal_cap.py), read by cut_vs_push.build_arms
EXTRA = "recognition/results/directions"  # scene.pt, assistant_character.pt (modal_scene.py, modal_direction.py)
CACHE, RUNS = "/cache", "/runs"
PACKAGES = ("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0", "numpy==2.5.4",
            "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0", "tiktoken==0.14.0", "jinja2==3.1.6",
            "regex==2026.9.29", "typer==0.27.3", "rich==15.0.0", "peft==0.21.2", "accelerate==1.15.0")
GPU_HOUR_USD = 2.6  # L40S


SKIP_DIRS = frozenset({"results", ".venv", "data", "__pycache__", "third_party"})
SKIP_SUFFIXES = frozenset({".jsonl", ".pyc"})


def _skip(p: Path) -> bool:
    """What stays out of the h4lite mount (Modal hands add_local_dir's ignore callable a path relative to the folder):
    results/, the laptop venv (139 MB, 1,737 files with symlinks to a Mac Python), the downloads under data/ (the
    apps read pairs from the volume or ship them as text), caches, third_party/ (no eval app imports it; the training
    image mounts it itself) and every JSONL (pairs.jsonl, 2 MB, and any result file)."""
    return bool(SKIP_DIRS & set(p.parts)) or p.suffix in SKIP_SUFFIXES


def mounted_files(root: str = H4) -> list[Path]:
    """The files of this folder that go into the image, as _skip leaves them."""
    base = Path(root)
    return sorted(p for p in base.rglob("*") if p.is_file() and not p.is_symlink() and not _skip(p.relative_to(base)))


def image(*extra: str) -> modal.Image:
    if modal.is_local():  # at app start, so a mount that grew is visible before any container builds
        files = mounted_files()
        print(f"h4lite mount: {len(files)} files, {sum(p.stat().st_size for p in files) / 2**20:.1f} MB "
              f"(skipping {', '.join(sorted(SKIP_DIRS))} and {' '.join(sorted(SKIP_SUFFIXES))})", flush=True)
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(*PACKAGES, *extra)
        .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false",
              "PYTHONPATH": f"{ALIGN}/anatta-dev/src:{H4}:{ALIGN}/recognition/scripts"})
        .add_local_dir(f"{ALIGN}/anatta-dev/src", f"{ALIGN}/anatta-dev/src", ignore=["**/__pycache__"])
        .add_local_dir(f"{RECOGNITION}/scripts", f"{ALIGN}/recognition/scripts", ignore=["**/__pycache__"])
        .add_local_dir(f"{ALIGN}/anatta/{DDIR}", f"{ALIGN}/anatta/{DDIR}")
        .add_local_dir(f"{ALIGN}/anatta/{DIALS}", f"{ALIGN}/anatta/{DIALS}")
        .add_local_dir(f"{ALIGN}/anatta/{VSCRIPTS}", f"{ALIGN}/anatta/{VSCRIPTS}", ignore=["**/__pycache__"])
        .add_local_file(f"{EXPERIMENTS}/{SELF_STAKES}", f"{ALIGN}/{SELF_STAKES}")
        .add_local_dir(f"{EXPERIMENTS}/{CAPS}", f"{ALIGN}/{CAPS}")
        .add_local_dir(f"{EXPERIMENTS}/{EXTRA}", f"{ALIGN}/{EXTRA}")
        .add_local_dir(H4, H4, ignore=_skip)
    )


def volumes() -> tuple[modal.Volume, modal.Volume]:
    """(the HF weights cache, the trained adapters)."""
    return (modal.Volume.from_name("anatta-hf-cache", create_if_missing=True),
            modal.Volume.from_name("anatta-runs", create_if_missing=True))


def adapter_dir(name: str) -> str:
    """Where --adapter NAME lives on the runs volume ("" for the base model)."""
    if not name:
        return ""
    return f"{RUNS}/{name}" if "/" in name else f"{RUNS}/{name}/adapter"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    import json

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cost(seconds: float) -> str:
    return f"{seconds / 3600:.2f} GPU-h, about ${seconds / 3600 * GPU_HOUR_USD:.2f}"
