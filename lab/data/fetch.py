#!/usr/bin/env python3
"""Refetch the lab's evaluation datasets exactly as recorded in manifest.json.

usage (from the lab root; any working directory works):

    python -I lab/data/fetch.py                 # all datasets
    python -I lab/data/fetch.py ifeval mask     # only these
    python -I lab/data/fetch.py --check         # verify what is on disk, no network

Every file listed in manifest.json is downloaded over HTTPS into
<data dir>/<name>/raw/<path>, hashed while it streams, and kept only if its
sha256 matches the manifest. Files already present with the right hash are
skipped. Nothing that is downloaded is ever executed, imported, extracted or
added to sys.path; this script uses only the standard library (plus certifi
for its CA bundle when that happens to be installed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifest.json"
MAX_FILE_BYTES = 2 * 1024 ** 3  # the lab never fetches anything above 2 GB
CHUNK = 1 << 20
USER_AGENT = "anatta-lab-fetch/1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[a-z0-9_]+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SCHEMA_VERSION = 1


class FetchError(Exception):
    """A manifest entry or a download that must not be accepted."""


# ----------------------------------------------------------------------------
# Paths: nothing in the manifest may write outside its own raw/ directory.
# ----------------------------------------------------------------------------

def safe_relpath(rel: object) -> str:
    """Return `rel` if it is a plain relative POSIX path, else raise FetchError.

    Rejects non-strings, empty paths, absolute paths, home or drive prefixes,
    backslashes, NUL, and any '', '.' or '..' component, so that joining the
    result onto a raw/ directory can only land inside that directory.
    """
    if not isinstance(rel, str) or not rel:
        raise FetchError(f"empty or non-string path: {rel!r}")
    if "\\" in rel or "\x00" in rel:
        raise FetchError(f"illegal character in path: {rel!r}")
    if rel.startswith(("/", "~")):
        raise FetchError(f"path is not relative: {rel!r}")
    parts = rel.split("/")
    if ":" in parts[0]:
        raise FetchError(f"path looks like a drive or URL: {rel!r}")
    if any(p in ("", ".", "..") for p in parts):
        raise FetchError(f"illegal path component in {rel!r}")
    if posixpath.normpath(rel) != rel:
        raise FetchError(f"path is not normalised: {rel!r}")
    return rel


def resolve_target(raw_dir: Path, rel: str) -> Path:
    """Return raw_dir/rel, guaranteed (after resolving symlinks) to lie inside raw_dir."""
    safe_relpath(rel)
    base = raw_dir.resolve()
    target = (base / rel).resolve()
    if base not in target.parents:
        raise FetchError(f"path escapes the raw directory: {rel!r}")
    return target


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

def validate_manifest(m: object) -> None:
    """Raise FetchError describing the first problems found in a manifest dict."""
    problems: list[str] = []
    if not isinstance(m, dict):
        raise FetchError("manifest is not a JSON object")
    if m.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    datasets = m.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise FetchError("manifest has no 'datasets' object")
    for name, d in datasets.items():
        where = f"datasets[{name!r}]"
        if not NAME_RE.match(name or ""):
            problems.append(f"{where}: bad dataset name")
        if not isinstance(d, dict):
            problems.append(f"{where}: not an object")
            continue
        if d.get("name") != name:
            problems.append(f"{where}: 'name' must equal its key")
        for key in ("purpose", "license", "fetch_date"):
            if not isinstance(d.get(key), str) or not d[key]:
                problems.append(f"{where}: missing string '{key}'")
        if not DATE_RE.match(str(d.get("fetch_date", ""))):
            problems.append(f"{where}: fetch_date must be YYYY-MM-DD")
        src = d.get("source")
        if not isinstance(src, dict) or not str(src.get("url", "")).startswith("https://") \
                or not isinstance(src.get("revision"), str) or not src["revision"]:
            problems.append(f"{where}: 'source' needs an https url and a revision")
        if not isinstance(d.get("flags"), list) or not all(isinstance(x, str) for x in d["flags"]):
            problems.append(f"{where}: 'flags' must be a list of strings")
        if not isinstance(d.get("fetched"), bool):
            problems.append(f"{where}: 'fetched' must be true or false")
        files = d.get("files")
        if not isinstance(files, list):
            problems.append(f"{where}: 'files' must be a list")
            continue
        if d.get("fetched") is False and files:
            problems.append(f"{where}: an unfetched dataset must list no files")
        if d.get("fetched") is True and not files:
            problems.append(f"{where}: a fetched dataset must list at least one file")
        seen: set[str] = set()
        total = 0
        for i, f in enumerate(files):
            fw = f"{where}.files[{i}]"
            if not isinstance(f, dict):
                problems.append(f"{fw}: not an object")
                continue
            try:
                p = safe_relpath(f.get("path"))
            except FetchError as e:
                problems.append(f"{fw}: {e}")
                p = None
            if p is not None:
                if p in seen:
                    problems.append(f"{fw}: duplicate path {p!r}")
                seen.add(p)
            if not str(f.get("url", "")).startswith("https://"):
                problems.append(f"{fw}: url must be https")
            if not isinstance(f.get("sha256"), str) or not SHA256_RE.match(f["sha256"]):
                problems.append(f"{fw}: sha256 must be 64 lowercase hex characters")
            b = f.get("bytes")
            if not isinstance(b, int) or isinstance(b, bool) or b < 0 or b > MAX_FILE_BYTES:
                problems.append(f"{fw}: bytes must be an integer in [0, 2 GB]")
            else:
                total += b
        if d.get("total_bytes") != total:
            problems.append(f"{where}: total_bytes {d.get('total_bytes')} != sum of files {total}")
    if problems:
        raise FetchError("invalid manifest:\n  " + "\n  ".join(problems[:40]))


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    validate_manifest(m)
    return m


# ----------------------------------------------------------------------------
# Hashing and downloading
# ----------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # optional: some Python builds ship without a CA bundle
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, dest: Path, ctx: ssl.SSLContext | None = None, retries: int = 3) -> str:
    """Stream `url` into `dest` atomically and return the sha256 of the bytes written.

    Refuses non-HTTPS URLs and anything above MAX_FILE_BYTES. The caller
    compares the returned digest with the manifest; this function only
    promises that `dest` holds exactly the bytes whose digest it returns.
    """
    if not url.startswith("https://"):
        raise FetchError(f"refusing non-https url: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
        tmp = Path(tmp_name)
        h = hashlib.sha256()
        n = 0
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=120, context=ctx) as r:
                length = r.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > MAX_FILE_BYTES:
                    raise FetchError(f"{url}: Content-Length {length} exceeds the 2 GB limit")
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > MAX_FILE_BYTES:
                        raise FetchError(f"{url}: exceeds the 2 GB limit")
                    out.write(chunk)
                    h.update(chunk)
            os.replace(tmp, dest)
            return h.hexdigest()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_error = e
            if attempt < retries:
                print(f"    retry {attempt}/{retries - 1} after: {e}", flush=True)
                time.sleep(2 * attempt)
        finally:
            if tmp.exists():
                tmp.unlink()
    raise FetchError(f"{url}: download failed: {last_error}")


# ----------------------------------------------------------------------------
# Per-file and per-dataset logic
# ----------------------------------------------------------------------------

def fetch_file(entry: dict, raw_dir: Path, *, ctx=None, check_only: bool = False,
               downloader=download) -> str:
    """Ensure raw_dir/entry['path'] holds the bytes with entry['sha256'].

    Returns one of 'verified' (already present and correct), 'downloaded',
    'missing' or 'corrupt' (check_only mode). Raises FetchError when a download
    does not hash to the manifest value; the bad file is removed first.
    """
    target = resolve_target(raw_dir, entry["path"])
    expected = entry["sha256"]
    if target.is_file() and sha256_of(target) == expected:
        return "verified"
    if check_only:
        return "corrupt" if target.exists() else "missing"
    got = downloader(entry["url"], target, ctx)
    if got != expected:
        if target.exists():
            target.unlink()
        raise FetchError(f"{entry['path']}: sha256 mismatch after download "
                         f"(got {got}, expected {expected}); file removed")
    if target.stat().st_size != entry["bytes"]:
        target.unlink()
        raise FetchError(f"{entry['path']}: size {target.stat().st_size} != manifest {entry['bytes']}")
    return "downloaded"


def fetch_dataset(name: str, d: dict, data_root: Path, *, ctx=None, check_only: bool = False) -> dict:
    counts = {"verified": 0, "downloaded": 0, "missing": 0, "corrupt": 0, "failed": 0}
    raw_dir = data_root / name / "raw"
    if not check_only:
        raw_dir.mkdir(parents=True, exist_ok=True)
    files = d["files"]
    for i, entry in enumerate(files, 1):
        try:
            status = fetch_file(entry, raw_dir, ctx=ctx, check_only=check_only)
        except FetchError as e:
            print(f"  !! {e}", flush=True)
            counts["failed"] += 1
            continue
        counts[status] += 1
        if status == "downloaded" and (i % 25 == 0 or i == len(files)):
            print(f"    {i}/{len(files)}", flush=True)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="dataset names from manifest.json (default: all)")
    ap.add_argument("--check", action="store_true", help="verify files on disk; never download")
    ap.add_argument("--root", type=Path, default=HERE, help="data directory (default: next to this script)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError, FetchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    datasets = manifest["datasets"]
    names = args.names or [n for n, d in datasets.items() if d["fetched"]]
    unknown = [n for n in names if n not in datasets]
    if unknown:
        print(f"error: unknown dataset(s) {unknown}; known: {sorted(datasets)}", file=sys.stderr)
        return 2

    ctx = None if args.check else ssl_context()
    rc = 0
    for name in names:
        d = datasets[name]
        if not d["fetched"]:
            print(f"== {name}: not fetched by design. {d.get('not_fetched_reason', '')}".rstrip())
            rc = max(rc, 2)
            continue
        print(f"== {name}: {len(d['files'])} files, {d['total_bytes'] / 1e6:.1f} MB "
              f"({'check' if args.check else 'fetch'})", flush=True)
        c = fetch_dataset(name, d, args.root, ctx=ctx, check_only=args.check)
        print(f"   verified {c['verified']}, downloaded {c['downloaded']}, missing {c['missing']}, "
              f"corrupt {c['corrupt']}, failed {c['failed']}")
        if c["failed"] or c["missing"] or c["corrupt"]:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
