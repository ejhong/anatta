"""K1's prepare, run on Modal in a CPU container with network and the "huggingface" secret (never printed). For each
repository: its ref resolved to a commit, the files K1 needs selected from the hub's listing at that commit,
downloaded at that commit into the volume's Hugging Face cache, each checked against the size and the LFS sha256 (or
the git blob sha) the hub lists, and its sha256 taken; the files Eugene approved by size checked against those sizes
too. The dataset's rows are counted, and its images decoded once to count their formats and modes. The result is the
prepared block config.yaml records (names, numbers and hashes only); the data stay on the volume.

Nothing here is hashed into the plan: what prepare fetched is pinned by the sha256s it records, and those are in the
plan. Nothing here runs a dataset script or loads a model.
"""

from __future__ import annotations

import fnmatch
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import k1 as K


def repos(cfg: dict) -> list[dict]:
    """The repositories prepare fetches, in order: the dataset, the vision model, and every enabled language model,
    each {key, repo, kind, ref, allow, expect_bytes}."""
    out = [{"key": "data", "repo": cfg["data"]["repo"], "kind": "dataset", "ref": cfg["data"]["ref"],
            "allow": list(cfg["data"]["allow"]), "expect_bytes": dict(cfg["data"].get("expect_bytes") or {})}]
    v = cfg["vision"]
    out.append({"key": v["key"], "repo": v["repo"], "kind": "model", "ref": v["ref"], "allow": list(v["allow"]),
                "expect_bytes": dict(v.get("expect_bytes") or {})})
    for key, t in cfg["text"].items():
        if t.get("switch") and not cfg["switches"].get(t["switch"]):
            continue
        out.append({"key": key, "repo": t["repo"], "kind": "model", "ref": t["ref"], "allow": list(t["allow"]),
                    "expect_bytes": dict(t.get("expect_bytes") or {})})
    return out


def hub_info(repo: str, kind: str, ref: str) -> tuple[str, dict]:
    """(the commit the ref names, {file: {size, lfs_sha256, blob_id}}) from the hub's listing with file metadata."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = (api.dataset_info if kind == "dataset" else api.model_info)(repo, revision=ref, files_metadata=True)
    files = {s.rfilename: {"size": s.lfs.size if s.lfs else s.size, "lfs_sha256": s.lfs.sha256 if s.lfs else None,
                           "blob_id": s.blob_id} for s in info.siblings}
    return info.sha, files


def hub_download(repo: str, kind: str, revision: str, names: list[str]) -> Path:
    """The named files at the commit, into the Hugging Face cache ($HF_HOME); returns the snapshot folder."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, repo_type=kind, revision=revision, allow_patterns=names,
                                  max_workers=8))


def select(files: dict, allow: list[str]) -> list[str]:
    """The listed files matching any allow pattern, sorted."""
    return sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in allow))


def check_file(path: Path, listed: dict) -> dict:
    """A downloaded file against the hub's listing: its size, and its LFS sha256 or else its git blob sha. Returns
    {bytes, sha256}."""
    size = path.stat().st_size
    sha = K.sha256_file(path)
    if size != listed["size"]:
        raise K.CheckError("hub", bytes=size, want=listed["size"])
    if listed["lfs_sha256"] is not None:
        if sha != listed["lfs_sha256"]:
            raise K.CheckError("hub", got=sha)
    elif K.git_blob_sha_file(path) != listed["blob_id"]:
        raise K.CheckError("hub", got=sha)
    return {"bytes": size, "sha256": sha}


def upper_seconds(total_bytes: int, mb_s: float) -> float:
    """A repository's upper estimate in prepare: a minute, plus its bytes at mb_s megabytes a second (download, check
    and commit)."""
    return 60.0 + total_bytes / (mb_s * 1e6)


def fetch_repo(spec: dict, mb_s: float, info=None, download=None, before=None, log=print) -> tuple[dict, Path]:
    """One repository resolved, selected, checked against its approved sizes, downloaded at the commit and checked
    file by file (check_file, in threads). Returns ({repo, kind, ref, revision, files: {name: {bytes, sha256}}}, the
    snapshot folder). info and download default to the hub's (hub_info, hub_download)."""
    info, download = info or hub_info, download or hub_download
    sha, listed = info(spec["repo"], spec["kind"], spec["ref"])
    names = select(listed, spec["allow"])
    if not names or any(n not in names for n in spec["expect_bytes"]):
        raise K.CheckError("files", selected=len(names))
    for n, want in spec["expect_bytes"].items():
        if listed[n]["size"] != want:
            raise K.CheckError("approved", bytes=listed[n]["size"], want=want)
    total = sum(listed[n]["size"] for n in names)
    if before is not None:
        before(upper_seconds(total, mb_s))
    t0 = time.time()
    snap = Path(download(spec["repo"], spec["kind"], sha, names))
    with ThreadPoolExecutor(max_workers=8) as ex:
        checked = dict(zip(names, ex.map(lambda n: check_file(snap / n, listed[n]), names)))
    log(f"prepare: {spec['repo']} at {sha[:12]}: {len(names)} files, {total / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    return {"repo": spec["repo"], "kind": spec["kind"], "ref": spec["ref"], "revision": sha, "files": checked}, snap


def dataset_facts(snap: Path, names: list[str], decode: bool = True) -> dict:
    """Rows per parquet file (from each file's metadata), and, with decode, counts over the decoded images
    (features.image_facts) and the captions (rows with an empty caption list)."""
    import pyarrow.parquet as pq

    rows = [pq.ParquetFile(snap / n).metadata.num_rows for n in names]
    out = {"rows": rows}
    if decode:
        import features

        images, caps = features.read_wit([snap / n for n in names])
        out["images"] = features.image_facts(images)
        out["captions"] = len(caps)
    return out


def run(spec: dict, info=None, download=None, before=None, commit=None, log=print, guard=None,
        decode: bool = True) -> dict:
    """prepare: every repository of spec["repos"] fetched and checked in order (the ceiling's check before each and
    a commit after each), the dataset counted. Returns {prepared, facts}: prepared is config.yaml's block, keyed as
    the repos are (data, dinov2, qwen0.5b, ...), with the dataset's rows per file; facts are prepare.json's extras."""
    prepared, facts = {}, {"seconds": {}}
    for spec_r in spec["repos"]:
        if guard is not None:
            guard.row = spec_r["key"]
        t0 = time.time()
        rec, snap = fetch_repo(spec_r, spec["mb_s"], info, download, before, log)
        if spec_r["kind"] == "dataset":
            d = dataset_facts(snap, sorted(rec["files"]), decode)
            if sum(d["rows"]) != spec["rows"]:
                raise K.CheckError("rows", rows=sum(d["rows"]), want=spec["rows"])
            rec["rows"] = d["rows"]
            facts["data"] = d
        prepared[spec_r["key"]] = rec
        facts["seconds"][spec_r["key"]] = round(time.time() - t0, 1)
        if commit is not None:
            commit()
    facts["bytes"] = sum(f["bytes"] for r in prepared.values() for f in r["files"].values())
    return {"prepared": prepared, "facts": facts}
