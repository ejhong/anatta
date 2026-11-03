"""Write results/MANIFEST.tsv from results_root (lab.toml): path, bytes, sha256 (see anatta.manifest). Files over 2 GB
are listed with their size and "unhashed"; lab.toml's manifest_exclude is left out. Claude (Opus 5.5) wrote this.

Run from the repo: .venv/bin/python tools/make_manifest.py [--out results/MANIFEST.tsv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anatta.config import load
from anatta.manifest import DEFAULT_OUT, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    cfg = load()
    if not cfg.results_root.is_dir():
        raise SystemExit(f"results_root {cfg.results_root} is not a folder")
    n, total, unhashed = write_manifest(cfg.results_root, a.out, cfg.manifest_exclude)
    print(f"{a.out}: {n} files, {total / 1e9:.2f} GB, {unhashed} unhashed (over 2 GB); excluded {cfg.manifest_exclude}")


if __name__ == "__main__":
    main()
