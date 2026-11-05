"""Fetch Hugging Face model snapshots into the anatta-hf-cache volume, on a Modal CPU container (app "anatta-fetch").
Claude (Opus 5.5) wrote this file, 5 Nov 2026, for the VLAF reproduction (experiments/reproductions/v1_vlaf_gap.yaml);
it follows calibration/prh_k1/fetch.py's checks (read, not imported).

  tools/modal.sh run experiments/fetch_hf.py::main --repos allenai/OLMo-2-1124-7B-Instruct,Qwen/Qwen3-8B
  tools/modal.sh run experiments/fetch_hf.py::main --repos Qwen/Qwen2.5-32B-Instruct --ref main

For each repo: the ref resolved to a commit; the files at that commit filtered to weights in safetensors, json
configs and tokenizer files only (ALLOW; anything matching DENY, e.g. pickles or code, is refused even if a pattern
admits it); downloaded at that commit into the volume's HF cache (/cache/hf, the layout the registry's snapshot()
reads); each file checked against the hub's size and LFS sha256 (or git blob sha) and its sha256 taken. Prints and
writes a manifest (names, bytes, sha256 only) to results_root/downloads/hf-<UTC stamp>.json. Nothing is loaded or
executed; public repos only (no token is sent).
"""

import fnmatch
import hashlib
import json
import time
from pathlib import Path

import modal

APP = "anatta-fetch"
CACHE = "/cache"
ALLOW = ("*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
         "chat_template.jinja", "chat_template.json")
DENY = ("*.bin", "*.pt", "*.pth", "*.pkl", "*.pickle", "*.ckpt", "*.py", "*.msgpack", "*.h5", "*.onnx", "*.gguf",
        "*.npz", "*.npy", "*/*")  # subfolders (e.g. original/) are refused too


def select(names, allow=ALLOW, deny=DENY) -> list:
    """The listed files a pattern in allow admits and none in deny refuses, sorted."""
    return sorted(n for n in names if any(fnmatch.fnmatch(n, p) for p in allow)
                  and not any(fnmatch.fnmatch(n, p) for p in deny))


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub==0.36.0", "hf_transfer==0.1.9")
image = image.env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
cache = modal.Volume.from_name("anatta-hf-cache")
app = modal.App(APP)


@app.function(image=image, cpu=4.0, memory=8192, timeout=3 * 3600, volumes={CACHE: cache})
def fetch(repo: str, ref: str = "main") -> dict:
    from huggingface_hub import HfApi, snapshot_download

    t0 = time.time()
    info = HfApi().model_info(repo, revision=ref, files_metadata=True, token=False)
    listed = {s.rfilename: {"size": s.lfs.size if s.lfs else s.size, "lfs_sha256": s.lfs.sha256 if s.lfs else None,
                            "blob_id": s.blob_id} for s in info.siblings}
    names = select(listed)
    if not any(n.endswith(".safetensors") for n in names):
        raise SystemExit(f"{repo}@{info.sha}: no safetensors weights in the listing; refusing")
    snap = Path(snapshot_download(repo_id=repo, revision=info.sha, allow_patterns=names, max_workers=8, token=False))
    files = {}
    for n in names:
        p = snap / n
        size, sha = p.stat().st_size, sha256_file(p)
        want = listed[n]
        if size != want["size"]:
            raise SystemExit(f"{repo}/{n}: {size} bytes, the hub lists {want['size']}")
        if want["lfs_sha256"] is not None:
            if sha != want["lfs_sha256"]:
                raise SystemExit(f"{repo}/{n}: sha256 differs from the hub's LFS sha256")
        elif git_blob_sha(p.read_bytes()) != want["blob_id"]:
            raise SystemExit(f"{repo}/{n}: git blob sha differs from the hub's")
        files[n] = {"bytes": size, "sha256": sha}
    cache.commit()
    return {"repo": repo, "ref": ref, "revision": info.sha, "snapshot": str(snap), "n_files": len(files),
            "bytes": sum(f["bytes"] for f in files.values()), "files": files,
            "skipped": sorted(set(listed) - set(names)), "seconds": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(repos: str, ref: str = "main"):
    from anatta.config import load

    out = []
    for r in fetch.map([x.strip() for x in repos.split(",") if x.strip()], kwargs={"ref": ref}):
        out.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "files"}, indent=1))
    d = load().results_root / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"hf-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("manifest ->", p)
