"""Re-grade stored replies under each named koan release parser (anatta.koans.base.RELEASE_PARSERS), parse only.

Walks every .json and .jsonl file under the folders given, takes every string that holds "<release>" as a reply, and
counts how many count as a release under "pin-ee89587", "instrument-v2" and "main", per experiment folder (the first
two parts of the path under each root). Nothing is executed and no reply is printed: the output is counts, plus the
sha256 of each reply whose verdict differs between parsers. Clean-lab step 2 (lab/plans/lib-diff-2026-11-03.md).
Claude (Opus 5.5) wrote this file, 3 Nov 2026.

  .venv/bin/python lab/tools/regrade_release.py lab/experiments/h4lite/results [more roots] [--json OUT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from anatta.koans.base import RELEASE_PARSERS, released

TAG = "<release>"


def strings(obj):
    """Every string inside a parsed JSON value."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def replies(path: Path):
    """The strings holding the release tag in one .json or .jsonl file (unparseable lines are skipped)."""
    text = path.read_text(errors="replace")
    if TAG not in text:
        return
    chunks = text.splitlines() if path.suffix == ".jsonl" else [text]
    for chunk in chunks:
        if TAG not in chunk:
            continue
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        yield from (s for s in strings(obj) if TAG in s)


def regrade(roots: list[Path]) -> dict:
    groups: dict[str, dict] = defaultdict(lambda: {"replies": 0, "unique": set(), **{p: 0 for p in RELEASE_PARSERS},
                                                   "flips": defaultdict(set)})
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.suffix in (".json", ".jsonl") and p.is_file()):
            rel = path.relative_to(root).parts
            g = groups[f"{root.name}/{'/'.join(rel[:2]) if len(rel) > 2 else rel[0]}"]
            for r in replies(path):
                verdict = {p: released(r, parser=p) is not None for p in RELEASE_PARSERS}
                sha = hashlib.sha256(r.encode()).hexdigest()
                g["replies"] += 1
                g["unique"].add(sha)
                for p, v in verdict.items():
                    g[p] += v
                for a, b in (("instrument-v2", "main"), ("pin-ee89587", "main"), ("pin-ee89587", "instrument-v2")):
                    if verdict[a] != verdict[b]:
                        g["flips"][f"{a} vs {b}"].add(sha)
    return {k: {"replies": g["replies"], "unique": len(g["unique"]), **{p: g[p] for p in RELEASE_PARSERS},
                "flips": {f: sorted(s) for f, s in g["flips"].items()}} for k, g in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="write the full result (with the flipped replies' sha256) here")
    a = ap.parse_args()
    out = regrade(a.roots)
    print(f"{'folder':58s} {'replies':>7s} {'pin':>5s} {'v2':>5s} {'main':>5s}  v2!=main pin!=main (unique replies)")
    for k, g in out.items():
        f = g["flips"]
        print(f"{k:58s} {g['replies']:7d} {g['pin-ee89587']:5d} {g['instrument-v2']:5d} {g['main']:5d}  "
              f"{len(f.get('instrument-v2 vs main', [])):8d} {len(f.get('pin-ee89587 vs main', [])):9d}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
