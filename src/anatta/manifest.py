"""results/MANIFEST.tsv: every file under results_root (lab.toml) with its size and sha256, so the record of what the
lab holds outside git survives in git. Files over 2 GB carry their size and "unhashed". Paths matching lab.toml's
manifest_exclude are left out. Written by tools/make_manifest.py, checked by tools/doctor.py. Stdlib only.
Claude (Opus 5.5) wrote this file, 3 Nov 2026.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

from .config import REPO

HASH_LIMIT = 2 * 1024**3  # 2 GB
UNHASHED = "unhashed"
HEADER = ("path", "bytes", "sha256")
DEFAULT_OUT = REPO / "results" / "MANIFEST.tsv"


def excluded(rel: str, patterns: tuple[str, ...]) -> bool:
    for pat in patterns:
        if pat.endswith("/"):
            if rel.startswith(pat):
                return True
        elif fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pat):
            return True
    return False


def walk(root: Path, patterns: tuple[str, ...]) -> Iterator[tuple[str, Path]]:
    """(relative posix path, path) of every regular file under root, sorted, excludes skipped; symlinks not followed."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        base = Path(dirpath)
        for name in sorted(filenames):
            p = base / name
            rel = p.relative_to(root).as_posix()
            if p.is_symlink() or not p.is_file() or excluded(rel, patterns):
                continue
            yield rel, p


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def entry(p: Path) -> tuple[int, str]:
    size = p.stat().st_size
    return size, (UNHASHED if size > HASH_LIMIT else sha256_file(p))


def read_manifest(path: Path) -> dict[str, tuple[int, str]]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows or tuple(rows[0].split("\t")) != HEADER:
        raise ValueError(f"{path}: not a manifest (header {HEADER})")
    out = {}
    for line in rows[1:]:
        rel, size, sha = line.split("\t")
        out[rel] = (int(size), sha)
    return out


def write_manifest(root: Path, out: Path, patterns: tuple[str, ...]) -> tuple[int, int, int]:
    """Returns (files, total bytes, unhashed files)."""
    n = total = unhashed = 0
    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(HEADER) + "\n")
        for rel, p in walk(root, patterns):
            if "\t" in rel or "\n" in rel:
                raise ValueError(f"a tab or newline in a file name under {root}: {rel!r}")
            size, sha = entry(p)
            f.write(f"{rel}\t{size}\t{sha}\n")
            n, total, unhashed = n + 1, total + size, unhashed + (sha == UNHASHED)
    os.replace(tmp, out)
    return n, total, unhashed
