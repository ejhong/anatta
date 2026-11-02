"""Calibration check 6, C6c's samples file: the code of every graded episode on a solvable twin in this folder's
results/, deduplicated on (pair, sha256 of the code), each with its recorded verdict, episode count and sources.

Writes, under results/calibration/koan_scorer/:
  c6c_samples.jsonl  one line per sample: pair, sha256, code, verdict (the recorded outcome), episodes, sources,
                     episodes_by_file
  c6c_files.json     every episode file by path and sha256 with its contribution (records, solvable, graded and
                     solved episodes, samples), the byte-duplicate groups, the counts, the samples file's sha256, and
                     the sha256 of the code that made it (this script, samples.py, anatta-dev's base.py, player.py)

The extraction is KoanTask's: base.CODE_RE from anatta-dev (the package the lab imported), first match, else the
reply stripped. Replies are stored cut at player.REPLY_CHARS characters; the script stops if any graded last reply
reaches that, since its code could be cut. It also stops, writing nothing, unless the registered counts reproduce
(ledger, 31 Oct, check 6's registration and amendment): 67 episode files, 60 contributing, 9 byte-duplicates,
21,070 episodes, 9,441 samples on 367 pairs, 71.9% recorded solved, no sample with two recorded verdicts, and the
per-source counts.

Nothing here runs any of the stored code. Run from anywhere, with the standard library only:
  python3 -I lab/experiments/h4lite/c6_samples.py [--samples-module PATH] [--anatta-dev DIR]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "calibration" / "koan_scorer"
ALIGN = HERE.parents[2].parent  # .../prj/align
SAMPLES_MODULE = ALIGN / "anatta-calib-c6" / "calibration" / "koan_scorer" / "samples.py"
ANATTA_DEV = ALIGN / "anatta-dev"

REGISTERED = {"episode_files": 67, "contributing_files": 60, "duplicate_files": 9, "episodes": 21070,
              "samples": 9441, "pairs": 367, "conflicts": 0, "solved_share": 71.9,
              "by_source": {"durability": {"episodes": 9245, "samples": 3711},
                            "durability_smoke": {"episodes": 62, "samples": 62},
                            "gate0": {"episodes": 3392, "samples": 2399},
                            "gate0_dry4": {"episodes": 64, "samples": 51},
                            "heldout": {"episodes": 257, "samples": 184},
                            "sft": {"episodes": 8050, "samples": 4261}}}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up while the file runs
    spec.loader.exec_module(mod)
    return mod


def reply_chars(player: Path) -> int:
    """player.REPLY_CHARS, read from the file's AST (player imports the lab's pilot, so it is not imported here)."""
    for node in ast.parse(player.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "REPLY_CHARS"):
            return int(ast.literal_eval(node.value))
    raise SystemExit(f"no REPLY_CHARS in {player}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples-module", default=str(SAMPLES_MODULE), help="calibration/koan_scorer/samples.py")
    ap.add_argument("--anatta-dev", default=str(ANATTA_DEV), help="the anatta-dev checkout the lab imported")
    a = ap.parse_args(argv)
    samples_py, base_py = Path(a.samples_module), Path(a.anatta_dev) / "src" / "anatta" / "koans" / "base.py"
    smp = load("koan_scorer_samples", samples_py)
    base = load("anatta_dev_koans_base", base_py)
    cap = reply_chars(HERE / "player.py")

    found = smp.collect(RESULTS, base.CODE_RE, exclude=("calibration/koan_scorer",))
    c = found["counts"]
    print({k: c[k] for k in ("episode_files", "contributing_files", "duplicate_files", "episodes", "samples", "pairs",
                             "conflicts", "solved_share", "episode_weighted_solved_share")}, flush=True)
    print(f"longest graded last reply: {found['longest_reply']} characters (replies are cut at {cap})", flush=True)
    if found["longest_reply"] >= cap:
        raise SystemExit(f"a graded last reply has {found['longest_reply']} characters, the cut at REPLY_CHARS "
                         f"({cap}): its code may be cut, so nothing is written")
    wrong = {k: (c[k], v) for k, v in REGISTERED.items() if c[k] != v}
    if wrong:
        raise SystemExit(f"the counts differ from the registered ones (got, registered): {wrong}; nothing is written")
    listing = smp.write(found, OUT, extra={
        "root": "lab/experiments/h4lite/results", "excluded": ["calibration/koan_scorer"],
        "code_re": {"pattern": base.CODE_RE.pattern, "flags": int(base.CODE_RE.flags)}, "reply_chars": cap,
        "registered_counts_reproduced": True,
        "code": {"c6_samples.py": sha256_file(Path(__file__)), "samples.py": sha256_file(samples_py),
                 "anatta-dev/src/anatta/koans/base.py": sha256_file(base_py),
                 "player.py": sha256_file(HERE / "player.py")}})
    print(f"wrote {OUT / 'c6c_samples.jsonl'} ({listing['samples_file']['lines']} samples, sha256 "
          f"{listing['samples_file']['sha256']}) and {OUT / 'c6c_files.json'}", flush=True)


if __name__ == "__main__":
    main()
