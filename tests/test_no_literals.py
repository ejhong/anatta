"""No absolute /Users/ path and no cap literal in live code: paths and caps come from lab.toml (anatta.config). The few
justified exceptions are listed, each with its reason, in tests/no_literals_allow.tsv; an entry that no longer matches
anything fails too, so the list stays honest. Scanned: .py and .sh files under the live folders, plus lab.toml;
archive/, tests and calibration/ are left out, and so are data files (specs, fixtures: records of finished runs).
Claude (Opus 5.5) wrote this file, 3 Nov 2026."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE = ["src/anatta", "tools", "lab/harness", *(f"lab/explore/{d}" for d in
                                                ("scout", "bench", "act1", "selfmap", "agentic_release", "noself",
                                                "shaping", "gate"))]
PATTERNS = [re.compile(p) for p in (r"/Users/", r"cap_usd *= *[0-9]", r"\bDAILY_CAP\w* *= *[0-9]")]
ALLOW_FILE = REPO / "tests" / "no_literals_allow.tsv"


def allowed() -> dict[str, str]:
    rows = [ln.split("\t") for ln in ALLOW_FILE.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert all(len(r) == 2 and r[1].strip() for r in rows), "each allow-list row is: path<TAB>reason"
    return {p: why for p, why in rows}


def scanned() -> list[Path]:
    files = [REPO / "lab.toml"]
    for d in LIVE:
        for f in sorted((REPO / d).rglob("*")):
            rel = f.relative_to(REPO).parts
            if f.suffix not in (".py", ".sh") or "archive" in rel or "calibration" in rel or "tests" in rel:
                continue
            if f.name.startswith("test_") or f.name == "conftest.py" or "__pycache__" in rel:
                continue
            files.append(f)
    return files


def hits() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in scanned():
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if any(p.search(line) for p in PATTERNS):
                out.setdefault(f.relative_to(REPO).as_posix(), []).append(f"{n}: {line.strip()[:100]}")
    return out


def test_no_path_or_cap_literals_in_live_code():
    allow = allowed()
    bad = {f: lines for f, lines in hits().items() if f not in allow}
    assert not bad, "literals in live code (use anatta.config.load(), or list the file with a reason):\n" + "\n".join(
        f"{f}:{ln}" for f, lines in bad.items() for ln in lines)


def test_every_allow_list_entry_is_still_needed():
    found = hits()
    stale = [p for p in allowed() if p not in found]
    assert not stale, f"allow-list entries with no literal left (remove them): {stale}"
