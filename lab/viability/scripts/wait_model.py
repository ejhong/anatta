"""Exit 0 once a Hugging Face model is fully and correctly in the local cache.

Waits until every weights file has the size the Hub lists (a re-download may be replacing a bad
copy), then checks each against its SHA-256: the cache names each weights blob after its hash, so
a truncated or corrupted download is caught instead of loaded. Exits 1 on a bad file or after the
timeout.

Usage: python wait_model.py REPO [TIMEOUT_SECONDS]
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo = sys.argv[1]
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 6 * 3600
try:
    sizes = {s.rfilename: s.size for s in HfApi().model_info(repo, files_metadata=True).siblings}
except Exception:  # offline: fall back to the checksum alone
    sizes = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir() -> Path | None:
    try:
        return Path(snapshot_download(repo, local_files_only=True))
    except Exception:
        # huggingface_hub 2.x refuses a snapshot with deliberately skipped files (Mistral's duplicate
        # consolidated.safetensors), so find the snapshot through refs/main instead
        from huggingface_hub.constants import HF_HUB_CACHE
        root = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
        ref = root / "refs" / "main"
        path = root / "snapshots" / ref.read_text().strip() if ref.exists() else None
        return path if path and path.exists() else None


def weights() -> list[Path] | None:
    """The weights files once all of them are in the cache, else None."""
    path = snapshot_dir()
    if path is None:
        return None
    index = path / "model.safetensors.index.json"
    if not (path / "config.json").exists() or not index.exists() and not (path / "model.safetensors").exists():
        return None
    names = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    files = [path / n for n in sorted(names)]
    if not all(f.exists() for f in files):
        return None
    if sizes and any(os.path.getsize(f) != sizes.get(f.name, os.path.getsize(f)) for f in files):
        return None
    return files


start = time.time()
while (files := weights()) is None:
    if time.time() - start > timeout:
        print(f"{repo}: not in the cache after {timeout:.0f}s", flush=True)
        sys.exit(1)
    time.sleep(20)
# The snapshot links to blobs/<sha256>, which may itself link into a shared Xet store named by
# another hash, so the expected checksum is the first link's name, not the final file's.
bad = [f.name for f in files if f.is_symlink() and sha256(f) != Path(os.readlink(f)).name]
if bad:
    print(f"{repo}: corrupted download, checksum mismatch in {', '.join(bad)}", flush=True)
    sys.exit(1)
print(f"{repo}: ready, {len(files)} weights file(s) verified", flush=True)
