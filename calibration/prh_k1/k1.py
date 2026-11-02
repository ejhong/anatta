"""Calibration K1's shared helpers: the stop codes, guarded stages, text-free results, hashes, the plan id, the
environment check, file verification and the precision flags. Standard library only at import (torch inside the
functions that need it), since app.py imports this file on the laptop.

The guard, the text-free check, the hashes, claim, plan_id and function_source follow check 4 (anatta-calib-c4,
calibration/persona_extractor/traits.py), written again here: K1 has no data to hide, but the convention keeps WIT's
captions on Modal and every result a list of numbers, names and hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

# The code that makes the run's numbers: hashed into the plan (and so into the plan id and the run's folder). fetch.py
# only downloads; what it fetched is pinned by the prepared block's sha256s, which are in the plan.
DATA_FILES = ("k1.py", "measure.py", "features.py")
# What the image carries besides app.py (which Modal mounts itself): app.py imports k1 and reads config.yaml at import.
IMAGE_FILES = (*DATA_FILES, "fetch.py", "config.yaml")

# ------------------------------------------------------------------------------------------------ errors

CODES = {  # every CheckError code, with what it means (the laptop prints these; a container passes on only the code)
    "missing": "a file the stage needs is not on the volume: run prepare (or the earlier stage) first",
    "sha256": "a file on the volume differs from the size or sha256 in config.yaml's prepared block",
    "hub": "a downloaded file differs from the size, LFS sha256 or git blob sha the hub lists for it",
    "approved": "a file's size differs from the size Eugene approved (config.yaml's expect_bytes)",
    "files": "a repository's selected files are not the expected ones (none, or an approved file missing)",
    "rows": "the dataset holds another number of rows than registered",
    "schema": "the dataset's parquet files lack the image or the text column",
    "caption": "a row has no caption",
    "image_bytes": "a row's image has no bytes",
    "shape": "a model's layer count or width differs from config.yaml's, or its features have another shape",
    "nonfinite": "a model's features hold a NaN or an infinity",
    "pickle": "the .bin checkpoint does not load with weights_only=True, or is not a dict of tensors",
    "device_map": "a module of a model loaded with device_map='auto' is not on the GPU",
    "plan": "the run folder holds another plan's files",
    "env_code": "this container's code differs from the plan's: launch again, which plans afresh",
    "env_pins": "installed versions differ from the image's pins",
    "free_text": "a stage's result holds a string that is not a name, a number, a hash or a path",
    "peak_memory": "a GPU is present but the peak memory read 0: the measurement failed",
    "budget": "the container ran past the $6 ceiling at its last commit (a unit ran longer than its upper estimate); "
              "its work so far is on the volume",
    "ceiling": "the next unit's upper estimate would carry the spend past the $6 ceiling, so it didn't start; the "
               "work so far is on the volume",
}
HEX = re.compile(r"[0-9a-f]{6,64}")


def _plain(v) -> bool:
    """A value a CheckError may carry into a log: a number, a bool, None or a hex hash."""
    return v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and HEX.fullmatch(v) is not None)


class CheckError(RuntimeError):
    """A check that stops the stage. Its code is one of CODES, never data; its facts are numbers or hashes only (any
    other value is dropped), so a guard can pass both on."""

    def __init__(self, code: str, **facts):
        if code not in CODES:
            raise ValueError(f"unknown CheckError code {code!r}")
        self.code = code
        self.facts = {k: v for k, v in facts.items() if _plain(v)}
        super().__init__(f"{code} {json.dumps(self.facts, sort_keys=True)}" if self.facts else code)


class StageError(RuntimeError):
    """What a guarded stage raises in place of any exception: the stage, the step and the exception's type."""


class Guard:
    """Holds the stage's name and the step it is on (a model key or a row index); stages set guard.row as they go."""

    def __init__(self, stage: str):
        self.stage, self.row, self.summary = stage, None, None

    def __enter__(self):
        return self

    def __exit__(self, etype, exc, tb):
        if exc is None or not isinstance(exc, Exception):
            return False
        self.summary = describe(exc)  # the exception itself, its message and its frames are dropped here
        return True

    def error(self) -> StageError:
        return StageError(f"{self.stage}: at {self.row}: {self.summary}")


def describe(exc: Exception) -> str:
    """An exception as text-free words: its type's name, and a CheckError's code and facts."""
    if isinstance(exc, CheckError):
        return f"CheckError {exc.code}" + (f" {json.dumps(exc.facts, sort_keys=True)}" if exc.facts else "")
    return type(exc).__name__


def error_name(exc: Exception) -> str:
    """An exception as one text-free word for a result: its type's name, and a CheckError's code after a colon
    (CheckError:pickle)."""
    return f"CheckError:{exc.code}" if isinstance(exc, CheckError) else type(exc).__name__


def run_guarded(stage: str, fn, *args, **kwargs):
    """fn(*args, guard=Guard(stage), **kwargs). Any exception becomes a StageError raised here, after the with block
    has swallowed the original, so the new error has no __context__ and no frame of the failing code."""
    guard = Guard(stage)
    with guard:
        return fn(*args, guard=guard, **kwargs)
    raise guard.error()


# ------------------------------------------------------------------------------------------------ files


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha(data: bytes) -> str:
    """The git blob sha of a file's bytes: sha1 of "blob <size>\\0" followed by the bytes (the hub's blob_id)."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def git_blob_sha_file(path) -> str:
    h = hashlib.sha1(b"blob %d\x00" % Path(path).stat().st_size)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


SAFE = re.compile(r"[A-Za-z0-9_.:/+@=,|\-]{0,200}")
GPU_NAME = re.compile(r"NVIDIA [A-Za-z0-9 _.\-]{1,60}")


def assert_text_free(obj) -> None:
    """Every string in obj (keys included) is a name, a number, a hash, a version or a path: no spaces, at most 200
    characters, from a small alphabet; a GPU's device name ("NVIDIA L40S") is the one exception. Raises
    CheckError("free_text") otherwise. Every stage's result passes through here before it leaves the container."""
    if isinstance(obj, str):
        if SAFE.fullmatch(obj) is None and GPU_NAME.fullmatch(obj) is None:
            raise CheckError("free_text")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_text_free(str(k))
            assert_text_free(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            assert_text_free(v)
    elif not (obj is None or isinstance(obj, (bool, int, float))):
        raise CheckError("free_text")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def written(path, res: dict) -> dict:
    """A stage's result, checked text-free, written as JSON and returned as the JSON reads back, so what the volume
    holds and what the laptop receives are the same."""
    res = json.loads(json.dumps(res))
    assert_text_free(res)
    write_json(path, res)
    return res


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def snapshot_dir(repo: str, revision: str, kind: str = "model", hf_home: str | None = None) -> Path:
    """A repository's folder at a pinned commit in the Hugging Face cache ($HF_HOME/hub/<kind>s--ORG--NAME/snapshots/
    REV), where prepare's downloads land; kind is "model" or "dataset"."""
    hub = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    return hub / (f"{kind}s--" + repo.replace("/", "--")) / "snapshots" / revision


def verify_files(snap, files: dict, threads: int = 8) -> dict:
    """Each of a repository's files (config.yaml's prepared block: {name: {bytes, sha256}}) in its snapshot folder
    against its registered size and sha256, hashed in threads. CheckError missing or sha256 (with the file's index in
    the sorted names) on the first difference. Returns {files, bytes, seconds}."""
    names = sorted(files)
    t0 = time.time()
    for i, name in enumerate(names):
        p = Path(snap) / name
        if not p.is_file():
            raise CheckError("missing", at=i)
        if p.stat().st_size != files[name]["bytes"]:
            raise CheckError("sha256", at=i, bytes=p.stat().st_size)
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        got = list(ex.map(lambda n: sha256_file(Path(snap) / n), names))
    for i, (name, sha) in enumerate(zip(names, got)):
        if sha != files[name]["sha256"]:
            raise CheckError("sha256", at=i, got=sha)
    return {"files": len(names), "bytes": sum(files[n]["bytes"] for n in names), "seconds": round(time.time() - t0, 2)}


# ------------------------------------------------------------------------------------------- the environment


def version_of(name: str) -> str | None:
    """A package's installed version, trying the name as written and with - and _ swapped."""
    for n in (name, name.replace("_", "-"), name.replace("-", "_")):
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return None


def pip_freeze() -> list[str]:
    """Every installed distribution as name==version, sorted: the container's own record of its image."""
    seen = {}
    for d in metadata.distributions():
        name = d.metadata["Name"] if d.metadata else None
        if name:
            seen[name] = d.version
    return [f"{n}=={v}" for n, v in sorted(seen.items(), key=lambda x: x[0].lower())]


def code_hashes(check_dir) -> dict[str, str]:
    """sha256 of the data code (DATA_FILES, this folder's): the plan's "code"."""
    return {f: sha256_file(Path(check_dir) / f) for f in DATA_FILES}


def check_environment(plan: dict, check_dir) -> dict:
    """This container against the plan: its data code (code_hashes) and the installed version of every package the
    image pins (a local label such as +cu128 is ignored). A difference stops the stage before anything loads."""
    code = code_hashes(check_dir)
    if code != plan["code"]:
        raise CheckError("env_code")
    have = {n: version_of(n) for n in plan["pins"]}
    if any((have[n] or "").split("+")[0] != v for n, v in plan["pins"].items()):
        raise CheckError("env_pins")
    return {"code": code, "installed": have}


def plan_id(plan: dict) -> str:
    """The plan's id (check 1's, copied as check 4 copied it): the first 12 hex digits of its sorted JSON's sha256."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]


def function_source(source: str, names) -> str:
    """The source of the named top-level functions of a module's text, each with its decorators, in the order named
    (check 4's traits.function_source)."""
    import ast

    lines = source.splitlines(keepends=True)
    defs = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    return "".join("".join(lines[min([defs[f].lineno, *(d.lineno for d in defs[f].decorator_list)]) - 1:
                                 defs[f].end_lineno]) for f in names)


def claim(out, plan: dict) -> None:
    """The run's folder holds this plan's files only: plan.json is written if absent and must equal the plan."""
    stored = read_json(Path(out) / "plan.json")
    if stored is None:
        write_json(Path(out) / "plan.json", plan)
    elif stored != plan:
        raise CheckError("plan")


def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 (no torch or no GPU in a CPU stage is not a failure)
        return None


class Clock:
    def __init__(self):
        self.t0 = time.time()

    def seconds(self) -> float:
        return round(time.time() - self.t0, 2)


def positions(padded: list[int], n: int, batch: int) -> int:
    """Token positions the forward passes cover: each batch's rows times its padded length."""
    return sum(min(batch, n - i * batch) * p for i, p in enumerate(padded))


def full_estimate(plan: dict, m: dict, tokens: dict) -> dict:
    """A rough estimate of the full run's seconds from the smoke's measures m (container start-up not included; the
    card adds config.yaml's): every file checked at the smoke's hashing rate; the rows read at twice the smoke's read
    (it read one of the two files' images); DINOv2's load plus its faster pass scaled to all rows; each language
    model's load at the 0.5B's stored bytes a second, and its forward at the 0.5B's seconds per token position, scaled
    by its weights' bytes and its own tokenization's positions; the scoring timed at full size on random features
    with the 0.5B's shapes, scaled by each model's layers times width. An estimate, not a bound: the ceiling's checks
    use config.yaml's upper estimates."""
    s, batch = plan["smoke_settings"], plan["protocol"]["batch"]
    rows, tkey = plan["data"]["rows"], s["text"]

    def wbytes(key: str) -> int:
        return sum(f["bytes"] for n, f in plan["prepared"][key]["files"].items() if n.endswith((".safetensors", ".bin")))

    vb = sum(v["bytes"] for v in m["verify"].values())
    vs = sum(v["seconds"] for v in m["verify"].values())
    every = sum(f["bytes"] for r in plan["prepared"].values() for f in r["files"].values())
    est = {"verify": every / max(vb / max(vs, 1e-3), 1.0), "read": 2 * m["read"]["seconds"],
           "vision": m["vision"]["load_seconds"] + min(m["vision"]["pass_seconds"]) * rows / s["pairs"]}
    load_rate = wbytes(tkey) / max(m["text"]["load_seconds"], 1e-3)
    per_pos = min(m["text"]["pass_seconds"]) / max(positions(tokens[tkey]["smoke"]["padded"], s["pairs"], batch), 1)
    t = plan["text"]
    base = (t[tkey]["layers"] + 1) * t[tkey]["dim"]
    for key, spec in t.items():
        fwd = per_pos * positions(tokens[key]["padded"], rows, batch) * wbytes(key) / max(wbytes(tkey), 1)
        est[f"text_{key}"] = wbytes(key) / load_rate + fwd
        est[f"score_{key}"] = m["score_random"]["seconds"] * (spec["layers"] + 1) * spec["dim"] / base
    est = {k: round(v, 1) for k, v in est.items()}
    return {"seconds": est, "total_seconds": round(sum(est.values()), 1)}


def precision_flags() -> dict:
    """torch's numerical settings as the run found them (K1 keeps the defaults: matmul precision "highest", TF32 off
    for matmuls; cuDNN's TF32 for convolutions is torch's default too, and DINOv2's patch embedding is a convolution),
    the attention kernels SDPA may pick, and the CUDA and cuDNN versions."""
    import torch

    def get(f):
        try:
            v = f()
        except Exception:  # noqa: BLE001 (an attribute this torch doesn't have)
            return None
        return v if isinstance(v, (bool, int, float, str)) or v is None else str(v)

    b = torch.backends
    return {
        "float32_matmul_precision": get(torch.get_float32_matmul_precision),
        "cuda_matmul_allow_tf32": get(lambda: b.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": get(lambda: b.cudnn.allow_tf32),
        "cuda_matmul_fp32_precision": get(lambda: b.cuda.matmul.fp32_precision),
        "cudnn_conv_fp32_precision": get(lambda: b.cudnn.conv.fp32_precision),
        "allow_bf16_reduced_precision_reduction": get(lambda: b.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "allow_fp16_reduced_precision_reduction": get(lambda: b.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "cudnn_benchmark": get(lambda: b.cudnn.benchmark),
        "cudnn_deterministic": get(lambda: b.cudnn.deterministic),
        "sdpa_flash": get(b.cuda.flash_sdp_enabled),
        "sdpa_mem_efficient": get(b.cuda.mem_efficient_sdp_enabled),
        "sdpa_math": get(b.cuda.math_sdp_enabled),
        "sdpa_cudnn": get(b.cuda.cudnn_sdp_enabled),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": get(b.cudnn.version),
    }
