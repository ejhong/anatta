"""The confirmation registered on 19 Nov 2026: the driver. The registration is confirm.yaml beside this file and its
entry in lab/ledger/NOTES.md; nothing here decides anything (confirm_core.py holds the rules). Claude (Fable 5.2,
an AI model) wrote this file on 19 Nov 2026.

Everything runs through the leg's own programs with the set named on each command (never exported: a shell with
the set left on would mix it into the other drivers), from the repo root, in this order:
  OVERLAP_SET=c1 .venv/bin/python lab/explore/overlap/run_overlap.py --items              (done before registering: the lists)
  OVERLAP_SET=c1 tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage floor --models m7
  OVERLAP_SET=c1 .venv/bin/python lab/explore/overlap/run_overlap.py --judge --only m7
  .venv/bin/python lab/explore/overlap/run_confirm.py --floor      (the gate: writes floor.json; nothing trains unless it passed)
  OVERLAP_SET=c1 tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage light_soo --models m7 --est 500
  OVERLAP_SET=c1 tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage light_other --models m7 --est 500
  OVERLAP_SET=c1 tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage ladder_other --models m7 --est 600
  OVERLAP_SET=c1 tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage battery --models m7 --est 500
  OVERLAP_SET=c1 .venv/bin/python lab/explore/overlap/run_overlap.py --judge --only m7
  OVERLAP_SET=c1 .venv/bin/python lab/explore/overlap/run_overlap.py --judge-dose --only m7
  .venv/bin/python lab/explore/overlap/run_confirm.py --read
Files: <results>/explore/2026-11-19/attractor-o1-c1/ (replies, the judge's rooms, floor.json, confirm.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["OVERLAP_SET"] = "c1"  # this driver is the confirmation's alone
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confirm_core as K  # noqa: E402
import overlap_core as C  # noqa: E402
import run_overlap as RO  # noqa: E402


def folder() -> Path:
    return RO.out_dir() / "gpu" / "m7"


def base_marks() -> dict:
    base, judged = RO.job(folder(), "base"), RO.jread(folder() / "base.judged.json")
    if not base or not judged:
        raise SystemExit("the unchanged model's replies or the judge's reading of them are missing")
    return C.marks_of(base, RO.items(), judged)


def floor() -> int:
    """The gate, on the unchanged model alone and with every reply read by the judge."""
    f = C.floor_reading("m7", base_marks(), band_required=False)
    print(f"floor on the new lists: passed {f['passed']} (deceptive {f['shares']['burglar']['counted']:.1f}% in the paper's order, "
          f"{f['shares']['mirror']['counted']:.1f}% with the rooms swapped; no answer {f['shares']['burglar']['unclear']:.1f}% and "
          f"{f['shares']['mirror']['unclear']:.1f}%; in the first lists' band {f['in_paper_band']}; favours the first room {f['favours_first_room']})")
    print("   every test, percent deceptive or wrong / honest or right / no answer: "
          + "; ".join(f"{t} {v['counted']:.0f}/{v['other']:.0f}/{v['unclear']:.0f}" for t, v in f["shares"].items()))
    RO.L.R.write_json(RO.out_dir() / "floor.json", f)
    return 0 if f["passed"] else 2


def expected_sha(kind: str, second: str) -> str:
    return hashlib.sha256(json.dumps(C.pairs(C.train_items(RO.items()), kind, second)).encode("utf-8")).hexdigest()


def identity() -> list[str]:
    """What is wrong with the saved jobs, if anything: the seeds, the number of steps, the pairs trained on, the code."""
    bad = []
    jobs = {f"{stage}_s{s}": (kind, K.STEPS) for stage, kind in C.LIGHT.items() for s in K.SEEDS}
    jobs |= {f"dose_ladder_other_s{s}": ("other", max(K.LADDER)) for s in K.SEEDS}
    for name, (kind, steps) in jobs.items():
        r = RO.jread(folder() / f"{name}.json")
        if not r:
            bad.append(f"{name}: no job file")
            continue
        if r["train"]["seed"] != int(name.rsplit("_s", 1)[1]) or len(r["log"]) != steps:
            bad.append(f"{name}: seed {r['train']['seed']}, {len(r['log'])} steps")
        if r["lora"].get("data_sha256") != expected_sha(kind, r["lora"].get("second_other") or C.SECOND_OTHERS[0]):
            bad.append(f"{name}: not trained on this set's pairs")
        if not r.get("code", {}).get("commit") or r["code"].get("dirty"):
            bad.append(f"{name}: run from uncommitted code")
        if name.startswith("dose_ladder") and sorted(int(k) for k in r["looks"]) != list(K.LADDER):
            bad.append(f"{name}: looks {sorted(r['looks'])}")
    return bad


def judge_none() -> dict:
    """Judge replies that had no "Answer:" line, on the two main tests, in percent. The registration's gate is on
    "any arm's judge replies on the two main tests": all of an arm's seeds, both tests ("by_arm"; each look of the
    control's ladder is taken as an arm of its own, the stricter way). "worst_cell" is the worst single cell (one
    seed, one look, one test): the driver as first run, at 08:38 UTC on 19 Nov, took that for the gate, which the
    registration's words do not say, and gave "not read" on 3 of 250 at one look of one control seed. The fresh
    Opus check of the grade found the departure (lab/ledger/checks/2026-11-19-overlap-confirm-grade-opus-check.md);
    this function was corrected after it, at about 08:50 UTC, and the output of the first run is kept beside the
    results as confirm_as_run.json."""
    none = lambda name: RO.jread(folder() / f"{name}.judge_none.json") or {}  # noqa: E731
    cells, arms = [], {}
    groups = {"unchanged": ["base"], "overlap": [f"light_soo_s{s}" for s in K.SEEDS], "control": [f"light_other_s{s}" for s in K.SEEDS]}
    for arm, names in groups.items():
        counts = [none(n).get(t, 0) for n in names for t in C.MAIN_TESTS]
        cells += counts
        arms[arm] = 100.0 * sum(counts) / (len(counts) * C.N_MAIN)
    for step in K.LADDER:
        counts = [none(f"dose_ladder_other_s{s}").get(str(step), {}).get(t, 0) for s in K.SEEDS for t in C.MAIN_TESTS]
        cells += counts
        arms[f"control ladder, {step} steps"] = 100.0 * sum(counts) / (len(counts) * C.N_MAIN)
    return {"by_arm": arms, "worst_arm": max(arms.values()), "worst_cell": 100.0 * max(cells) / C.N_MAIN}


def ladder(base: dict) -> dict:
    """{step: the control's reading at that look} for the two main tests, from the ladder runs' judged looks."""
    main = {t: base[t] for t in C.MAIN_TESTS}
    judged = [RO.jread(folder() / f"dose_ladder_other_s{s}.judged.json") for s in K.SEEDS]
    if not all(judged):
        return {}
    return {step: C.arm_reading(main, [{t: {i: C.mark_of_answer(t, a) for i, a in j[str(step)][t].items()} for t in C.MAIN_TESTS} for j in judged], C.CEILING2)
            for step in K.LADDER}


def read() -> int:
    m7 = RO.reading().get("m7")
    if not m7:
        raise SystemExit("nothing of the confirmation is on disk")
    lad = ladder(base_marks())
    batt = RO.battery().get("m7") or {}
    whole = all((folder() / f"{a}_e.json").exists() for a in ("base", *C.BATTERY_ARMS_BY_SET[K.SET])) and "light_soo" in batt.get("kinds", {})
    none = judge_none()
    checks = {"identity": identity(), "judge_none": none["worst_arm"], "judge_none_detail": none, "battery": bool(whole), "ladder": len(lad) == len(K.LADDER)}
    g = K.gates(m7, RO.jread(RO.out_dir() / "floor.json"), checks)
    p = K.predictions(m7, lad, batt) if g["ok"] else None
    RO.L.R.write_json(RO.out_dir() / "confirm.json", {"gates": g, "predictions": p, "verdict": K.verdict(g, p)})
    print(K.table(g, p))
    print(f"   judge replies with no answer line: by arm, at worst {none['worst_arm']:.2f}% (the registered gate, at most {K.NONE_MAX:.0f}%); "
          f"the worst single seed, look and test {none['worst_cell']:.2f}%")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args(argv)
    return floor() if a.floor else read() if a.read else ap.error("--floor or --read")


if __name__ == "__main__":
    sys.exit(main())
