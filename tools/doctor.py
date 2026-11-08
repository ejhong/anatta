"""The lab's health check for a fresh session: prints one PASS / WARN / FAIL line per check and exits 1 on any FAIL.
It never prints a secret: keys are reported as present or absent, and credential files are never opened.
Claude (Opus 5.5) wrote this file, 3 Nov 2026.

Run from the repo: .venv/bin/python tools/doctor.py [--sample 20] [--no-pytest]
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import random
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from anatta import manifest as mm
from anatta import spend
from anatta.config import REPO, Config, load

MIN_PY = (3, 11)
PACKAGES = ("numpy", "torch", "transformers", "pyyaml", "pytest", "httpx")


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def line(self, status: str, name: str, detail: str) -> None:
        self.failed += status == "FAIL"
        print(f"{status:<4}  {name}: {detail}", flush=True)

    def check(self, name: str, fn: Callable[[], tuple[str, str]]) -> None:
        try:
            status, detail = fn()
        except Exception as e:  # a check that crashes is a failed check, not a crashed doctor
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        self.line(status, name, detail)


def check_python() -> tuple[str, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PY
    return ("PASS" if ok else "FAIL"), f"{v.major}.{v.minor}.{v.micro} at {sys.executable}"


def check_packages() -> tuple[str, str]:
    found, missing = [], []
    for p in PACKAGES:
        try:
            found.append(f"{p} {md.version(p)}")
        except md.PackageNotFoundError:
            missing.append(p)
    if missing:
        return "FAIL", f"missing {', '.join(missing)}; have {', '.join(found)}"
    return "PASS", ", ".join(found)


def check_lock() -> tuple[str, str]:
    f = REPO / "uv.lock"
    if f.exists():
        return "PASS", f"{f.name} present"
    return "WARN", "no uv.lock: the environment is not pinned yet (clean-lab step 1: `uv lock`, needs PyPI access)"


def check_results(cfg: Config, sample: int) -> tuple[str, str]:
    root, man = cfg.results_root, REPO / "results" / "MANIFEST.tsv"
    if not root.is_dir():
        return "FAIL", f"results_root {root} is missing (tests that need stored outputs will skip)"
    if not man.exists():
        return "FAIL", f"no {man.relative_to(REPO)}; run tools/make_manifest.py"
    listed = mm.read_manifest(man)
    missing, resized = [], []
    for rel, (size, _) in listed.items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif p.stat().st_size != size:
            resized.append(rel)
    hashable = sorted(r for r, (_, sha) in listed.items() if sha != mm.UNHASHED and r not in missing + resized)
    picked = random.Random(0).sample(hashable, min(sample, len(hashable)))
    bad_sha = [r for r in picked if mm.sha256_file(root / r) != listed[r][1]]
    new = sum(1 for rel, _ in mm.walk(root, cfg.manifest_exclude) if rel not in listed)
    detail = (f"{root}: {len(listed)} listed, {len(missing)} missing, {len(resized)} resized, "
              f"{len(bad_sha)}/{len(picked)} sampled sha256 differ, {new} new since the manifest")
    if missing or resized or bad_sha:
        first = (missing + resized + bad_sha)[:3]
        return "FAIL", f"{detail} (first: {', '.join(first)})"
    return ("WARN" if new else "PASS"), detail + ("; rerun tools/make_manifest.py" if new else "")


def check_modal(cfg: Config) -> tuple[str, str]:
    cli = cfg.modal.cli
    if cli.is_file() and os.access(cli, os.X_OK):
        return "PASS", f"Modal CLI at {cli} (presence only); volumes in lab.toml: {', '.join(cfg.modal.volumes.values())}"
    return "FAIL", f"no executable Modal CLI at {cli}"


def check_keys(cfg: Config) -> tuple[str, str]:
    present = {svc: bool(os.environ.get(var)) for svc, var in cfg.key_envs.items()}
    detail = "; ".join(f"{cfg.key_envs[s]} {'present' if ok else 'absent'}" for s, ok in present.items())
    return ("PASS" if all(present.values()) else "WARN"), detail or "no key names in lab.toml"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def check_git() -> tuple[str, str]:
    branch = _git("branch", "--show-current") or "(detached)"
    dirty = [x for x in _git("status", "--porcelain").splitlines() if x.strip()]
    head = _git("rev-parse", "--short", "HEAD")
    if dirty:
        return "WARN", f"branch {branch} at {head}, {len(dirty)} uncommitted change(s)"
    return "PASS", f"branch {branch} at {head}, clean"


def check_tools(brewfile: Path | None = None) -> tuple[str, str]:
    """The machine's command-line tools: every `brew "name"` in the machine-wide ~/Brewfile (one list for every
    project on the machine, not kept in this repo) is installed. Added 8 Nov 2026 (poppler: the Read tool renders PDF
    pages with pdftoppm)."""
    brewfile = brewfile or Path.home() / "Brewfile"
    if not brewfile.is_file():
        return "WARN", f"no {brewfile}: the machine's tool list is missing (lab/MOVING.md)"
    want = [m.group(1).rsplit("/", 1)[-1] for m in re.finditer(r'^\s*brew\s+"([^"]+)"', brewfile.read_text(), re.M)]
    brew = shutil.which("brew")
    if not brew:
        return "WARN", f"{len(want)} tools listed in {brewfile} but no brew on PATH"
    have = set(subprocess.run([brew, "list", "--formula", "-1"], capture_output=True, text=True).stdout.split())
    missing = [w for w in want if w not in have]
    if missing:
        return "WARN", f"missing from {brewfile}: {', '.join(missing)} (brew bundle --file {brewfile} --no-upgrade)"
    return "PASS", f"{len(want)} tools in {brewfile} installed"


def check_pytest() -> tuple[str, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                       cwd=REPO, capture_output=True, text=True)
    tail = [x for x in r.stdout.strip().splitlines() if x.strip()][-1:] or [r.stderr.strip()[-200:]]
    return ("PASS" if r.returncode == 0 else "FAIL"), f"collection: {tail[0]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, default=20, help="files whose sha256 is re-checked against the manifest")
    ap.add_argument("--no-pytest", action="store_true")
    a = ap.parse_args()
    r = Report()
    r.check("python", check_python)
    r.check("packages", check_packages)
    r.check("lock", check_lock)
    try:
        cfg = load()
        r.line("PASS", "lab.toml", f"{cfg.path}: default daily cap ${cfg.daily_cap_usd:.2f}, today's "
                                   f"${spend.daily_cap(cfg.spend_dir):.2f} (the day file's cap_usd wins), default model "
                                   f"{cfg.default_model}, spend dir {cfg.spend_dir}")
    except Exception as e:
        r.line("FAIL", "lab.toml", f"{type(e).__name__}: {e}")
        cfg = None
    if cfg:
        r.check("results", lambda: check_results(cfg, a.sample))
        r.check("modal", lambda: check_modal(cfg))
        r.check("keys", lambda: check_keys(cfg))
    r.check("tools", check_tools)
    r.check("git", check_git)
    if not a.no_pytest:
        r.check("pytest", check_pytest)
    print(f"doctor: {'all checks passed' if not r.failed else f'{r.failed} check(s) failed'}")
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
