"""No test reads a saved episodes file. Saved episodes hold model-written replies; fed to replay() or silent_check()
they would run model-written code, which happens only inside a Modal container, never on the laptop (replay.py's
docstring). The tests use code strings written for them. This scans every file under tests/ and lab/**/tests/ for a
line that names an episodes JSONL file and reads something. Claude (Opus 5.5) wrote this file, 5 Nov 2026."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NAMES = re.compile("episodes" + r"[A-Za-z0-9_*{}.\-]*\.jsonl")  # split so this file's own source never matches
READS = re.compile(r"open\(|read_text|read_bytes|readlines|json\.load|load_jsonl|read_jsonl|replay")


def folders():
    return [REPO / "tests", *sorted(p for p in (REPO / "lab").rglob("tests") if p.is_dir())]


def test_no_test_reads_a_saved_episodes_file():
    hits = []
    for d in folders():
        for f in sorted(d.rglob("*.py")):
            if f == Path(__file__).resolve():
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if NAMES.search(line) and READS.search(line):
                    hits.append(f"{f.relative_to(REPO)}:{n}")
    assert not hits, hits
    assert len(folders()) > 5  # the scan sees the folders
    sample = 'rows = open(RUN / f"round{r}_' + 'episodes.jsonl").read()'  # and would catch a read
    assert NAMES.search(sample) and READS.search(sample)
