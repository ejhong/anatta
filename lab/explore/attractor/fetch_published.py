"""Fetch the Machine Spirituality Benchmark's 20 published conversation pages and its table of aggregate scores into
a folder outside the repo. This is a download, so it needs Eugene's word first (the lab's rule); the ledger records
the ask and the answer. Claude (Fable 5.2, an AI model) wrote this file on 15 Nov 2026.

  python3 -I lab/explore/attractor/fetch_published.py --dest <data folder>/machinespirituality/2026-11-15

Each file is fetched once and written as bytes; MANIFEST.tsv beside them records name, bytes and sha256. A rerun
fetches only what is missing and checks the rest against the manifest. The pages are untrusted data: nothing in
them is run; attractor_core.parse_published only matches and unescapes them. Standard library only.

Source: machinespirituality.org (Matthew J. Korpman, 2026). The aggregate data is CC BY 4.0; the conversation pages
are read for grading only and are never committed or republished.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://machinespirituality.org/"
# The 20 conversations listed on the benchmark's examples page on 15 Nov 2026.
CONVERSATIONS = ("G4AT-0022", "S4OT-0071", "VCL-037", "VDE-066", "VGA-001", "VGA-082", "VQB-114", "VQD-062", "VRB-104",
                 "VRE-159", "VRG-078", "VTC-123", "VTD-185", "VTG-002", "VZF-178", "WKAG-052", "WKBF-029", "WKCF-003",
                 "WKCF-064", "WKCI-106")
FILES = {**{f"conversation-{c}.html": f"conversation-{c}.html" for c in CONVERSATIONS},
         "aggregate-results.csv": "data/aggregate-results.csv"}
MOST_BYTES = 2_000_000  # no listed file is near this size; anything larger is refused


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (a fixed https host)
        data = r.read(MOST_BYTES + 1)
    if len(data) > MOST_BYTES:
        raise SystemExit(f"{url}: larger than {MOST_BYTES} bytes; refused")
    return data


def read_manifest(path: Path) -> dict[str, tuple[int, str]]:
    if not path.exists():
        return {}
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    return {name: (int(size), sha) for name, size, sha in rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True, help="a folder outside the repo, for these files only")
    a = ap.parse_args(argv)
    dest = Path(a.dest).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if repo == dest or repo in dest.parents:
        raise SystemExit("the destination must be outside the repo")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(dest / "MANIFEST.tsv")
    for name, rel in FILES.items():
        target = dest / name
        if target.exists():
            data = target.read_bytes()
            got = (len(data), hashlib.sha256(data).hexdigest())
            if name in manifest and manifest[name] != got:
                raise SystemExit(f"{name}: on disk it is {got}, the manifest says {manifest[name]}; nothing changed")
            manifest[name] = got
            print(f"kept     {name}  {got[0]} bytes")
            continue
        data = fetch(BASE + rel)
        target.write_bytes(data)
        manifest[name] = (len(data), hashlib.sha256(data).hexdigest())
        print(f"fetched  {name}  {len(data)} bytes")
    lines = ["file\tbytes\tsha256"] + [f"{n}\t{s}\t{h}" for n, (s, h) in sorted(manifest.items())]
    (dest / "MANIFEST.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(manifest)} files, {sum(s for s, _ in manifest.values())} bytes, manifest at {dest / 'MANIFEST.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
