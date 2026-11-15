"""Fetch the four files of shutdown and corrigibility questions that Eugene approved on 15 Nov 2026 (ledger, 03:39
UTC) into a folder outside the repo. Claude (Fable 5.2, an AI model) wrote this file on 15 Nov 2026.

  python3 -I lab/explore/attractor/fetch_shutdown.py --dest <data folder>/anthropic-evals/2026-11-15

Source: github.com/anthropics/evals, folder advanced-ai-risk/human_generated_evals (Perez et al. 2022, "Discovering
Language Model Behaviors with Model-Written Evaluations"; these four are the human-written sets; licence CC BY 4.0).
Each file is fetched once and written as bytes; MANIFEST.tsv beside them records name, bytes and sha256. A file whose
size differs from the size named in the approved ask is refused and not written: that would need a new ask. A rerun
fetches only what is missing and checks the rest against the manifest. The files are untrusted data: nothing in them is
run; inside_core.read_items parses them as JSON lines. The questions are benchmark items: they stay out of the repo
(counts and sha256 only) and are never printed. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/anthropics/evals/main/advanced-ai-risk/human_generated_evals/"
# The four approved files, with the sizes named in the ask (bytes). Anything else needs a new ask.
FILES = {"survival-instinct.jsonl": 332_009, "corrigible-less-HHH.jsonl": 107_056,
         "corrigible-more-HHH.jsonl": 113_824, "corrigible-neutral-HHH.jsonl": 101_545}


def fetch(url: str, most: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (a fixed https host)
        return r.read(most + 1)


def read_manifest(path: Path) -> dict[str, tuple[int, str]]:
    if not path.exists():
        return {}
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    return {name: (int(size), sha) for name, size, sha in rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True, help="a new folder outside the repo, for these files only")
    a = ap.parse_args(argv)
    dest = Path(a.dest).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if repo == dest or repo in dest.parents:
        raise SystemExit("the destination must be outside the repo")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(dest / "MANIFEST.tsv")
    refused = []
    for name, approved in FILES.items():
        target = dest / name
        if target.exists():
            data = target.read_bytes()
            got = (len(data), hashlib.sha256(data).hexdigest())
            if name in manifest and manifest[name] != got:
                raise SystemExit(f"{name}: on disk it is {got}, the manifest says {manifest[name]}; nothing changed")
            manifest[name] = got
            print(f"kept     {name}  {got[0]} bytes")
            continue
        data = fetch(BASE + name, approved)
        if len(data) != approved:
            refused.append(name)
            print(f"REFUSED  {name}  the source gives {'more than ' if len(data) > approved else ''}{len(data)} bytes, "
                  f"the approved ask named {approved}; not written")
            continue
        target.write_bytes(data)
        manifest[name] = (len(data), hashlib.sha256(data).hexdigest())
        print(f"fetched  {name}  {len(data)} bytes")
    lines = ["file\tbytes\tsha256"] + [f"{n}\t{s}\t{h}" for n, (s, h) in sorted(manifest.items())]
    (dest / "MANIFEST.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(manifest)} files, {sum(s for s, _ in manifest.values())} bytes, manifest at {dest / 'MANIFEST.tsv'}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
