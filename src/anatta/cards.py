"""The lab's one card writer (clean-lab step 3). Claude (Opus 5.5) wrote this file, 3 Nov 2026. Stdlib only.

  write_plan(dir, spec)   before any result: plan.json with the spec, its sha256, the code commit, whether the work
                          tree was dirty, and the UTC clock. Refused once the folder holds a card (a plan written after
                          its results is no plan) and refused over an existing plan unless force=True.
  write_card(dir, card)   card.json (or `name`) with a "written" block: the code commit, dirty flag, UTC clock and the
                          sha256 of the folder's plan.json when there is one.
  write_text(dir, name, text)  a companion file (card.md) under the same rule.

Never overwrite: a file that exists is refused, unless force=True, which writes the next free <stem>.vN<suffix>
(card.v2.json, card.v3.json, ...) beside it and never touches the old file. Files are created exclusively (open
mode "x"), so two writers cannot both win.

The code commit comes from git at the repo holding this package. Inside a container with no git the block says so
(commit None); pass code=git_state() from the laptop side instead when it matters.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .config import REPO

PLAN = "plan.json"
CARD = "card.json"
PLAN_FORMAT = "anatta-plan/1"


class CardExists(FileExistsError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(obj) -> bytes:
    """The bytes a spec's sha256 is taken over: JSON with sorted keys, no spaces, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def git_state(repo: Path = REPO) -> dict:
    """{"commit": HEAD's sha or None, "dirty": True if tracked or untracked files differ from HEAD, or None}."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                              timeout=30).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "note": f"no git work tree at {repo}"}
    return {"commit": commit, "dirty": dirty}


def _free_name(path: Path) -> Path:
    n = 2
    while (p := path.with_name(f"{path.stem}.v{n}{path.suffix}")).exists():
        n += 1
    return p


def _write_new(path: Path, data: str, force: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if path.exists():
        if not force:
            raise CardExists(f"{path} exists; refusing to overwrite (force=True writes {_free_name(path).name})")
        target = _free_name(path)
    while True:
        try:
            with open(target, "x", encoding="utf-8") as f:
                f.write(data)
            return target
        except FileExistsError:  # another writer took the name between the check and the open
            if not force:
                raise CardExists(f"{target} exists; refusing to overwrite") from None
            target = _free_name(path)


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n"


def write_plan(out_dir: str | os.PathLike, spec: dict | str | os.PathLike, *, force: bool = False,
               code: dict | None = None) -> Path:
    """plan.json in out_dir, before any result. spec: a dict (hashed in canonical JSON) or a spec file's path (hashed
    as its bytes, and recorded by path). Returns the path written."""
    out = Path(out_dir)
    cards = sorted(p.name for p in out.glob("card*.json")) if out.is_dir() else []
    if cards:
        raise CardExists(f"{out} already holds {cards}: a plan must be written before its results")
    if isinstance(spec, dict):
        rec = {"spec": spec, "spec_sha256": sha256_bytes(canonical(spec))}
    else:
        p = Path(spec)
        rec = {"spec_file": str(p), "spec_sha256": sha256_bytes(p.read_bytes())}
    plan = {"format": PLAN_FORMAT, **rec, "code": code or git_state(), "written_utc": utc_now()}
    return _write_new(out / PLAN, _dump(plan), force)


def written_block(out_dir: Path, code: dict | None = None) -> dict:
    plan = out_dir / PLAN
    return {**(code or git_state()), "utc": utc_now(), "by": "anatta.cards",
            "plan_sha256": sha256_bytes(plan.read_bytes()) if plan.exists() else None}


def write_card(out_dir: str | os.PathLike, card: dict, *, name: str = CARD, force: bool = False,
               code: dict | None = None) -> Path:
    """card (a JSON-able dict) at out_dir/name with a "written" block added; refuses to overwrite unless force=True
    (then <stem>.vN<suffix>). A card that already carries "written" is refused: the block is the writer's."""
    if "written" in card:
        raise ValueError('a card must not carry its own "written" block; write_card adds it')
    out = Path(out_dir)
    return _write_new(out / name, _dump({**card, "written": written_block(out, code)}), force)


def write_text(out_dir: str | os.PathLike, name: str, text: str, *, force: bool = False) -> Path:
    """A companion text file (card.md) under the same no-overwrite rule."""
    return _write_new(Path(out_dir) / name, text, force)
