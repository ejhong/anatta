"""The confirmation registered on 19 Nov 2026: the driver. The registration is confirm.yaml beside this file and its
entry in lab/ledger/NOTES.md; nothing here decides anything (confirm_core.py holds the rules). Claude (Fable 5.2,
an AI model) wrote this file on 19 Nov 2026.

Everything runs through the leg's own programs with the set named in the environment, from the repo root:
  export OVERLAP_SET=c1
  .venv/bin/python lab/explore/overlap/run_overlap.py --items                               (done before registering: the new lists)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage floor --models m7    (the unchanged model alone)
  .venv/bin/python lab/explore/overlap/run_overlap.py --judge --only m7
  .venv/bin/python lab/explore/overlap/run_confirm.py --floor                                (the gate: stop here if it fails)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage light_soo --models m7
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage light_other --models m7
  .venv/bin/python lab/explore/overlap/run_overlap.py --judge --only m7
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage battery --models m7
  .venv/bin/python lab/explore/overlap/run_confirm.py --read
Files: <results>/explore/2026-11-19/attractor-o1-c1/ (replies, the judge's rooms, confirm.json).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["OVERLAP_SET"] = "c1"  # this driver is the confirmation's alone
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confirm_core as K  # noqa: E402
import overlap_core as C  # noqa: E402
import run_overlap as RO  # noqa: E402


def floor() -> int:
    """The gate, on the unchanged model alone and with every reply read by the judge."""
    folder = RO.out_dir() / "gpu" / "m7"
    base, judged = RO.job(folder, "base"), RO.jread(folder / "base.judged.json")
    if not base or not judged:
        raise SystemExit("the unchanged model's replies or the judge's reading of them are missing")
    f = C.floor_reading("m7", C.marks_of(base, RO.items(), judged), band_required=False)
    print(f"floor on the new lists: passed {f['passed']} (deceptive {f['shares']['burglar']['counted']:.1f}% in the paper's order, "
          f"{f['shares']['mirror']['counted']:.1f}% with the rooms swapped; no answer {f['shares']['burglar']['unclear']:.1f}% and "
          f"{f['shares']['mirror']['unclear']:.1f}%; in the first lists' band {f['in_paper_band']}; favours the first room {f['favours_first_room']})")
    RO.L.R.write_json(RO.out_dir() / "floor.json", f)
    return 0 if f["passed"] else 2


def read() -> int:
    m7 = RO.reading().get("m7")
    if not m7:
        raise SystemExit("nothing of the confirmation is on disk")
    g = K.gates(m7)
    p = K.predictions(m7, RO.battery().get("m7")) if g["ok"] else None
    RO.L.R.write_json(RO.out_dir() / "confirm.json", {"gates": g, "predictions": p, "verdict": K.verdict(g, p)})
    print(K.table(g, p))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args(argv)
    return floor() if a.floor else read() if a.read else ap.error("--floor or --read")


if __name__ == "__main__":
    sys.exit(main())
