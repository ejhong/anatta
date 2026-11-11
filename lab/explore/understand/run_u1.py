"""Step 3's local driver (lab/plans/step3-understanding-first-2026-11-10.md, revisions 2 and 3; u1_core.py's
docstring). Claude (Fable 5.2, an AI model) wrote this file on 10 Nov 2026, before any run existed, and on 11 Nov 2026
changed it to revision 3, after the first dose check did not pass and before any arm was trained: reads are at
optimizer steps, the dose check is the revised one, and a main run's arms are chosen at its first launch. Explore
speed: sightings only.

  --dry             the plan, the local inputs, what the stream holds, expected and upper costs, spend so far. Nothing
                    is launched or spent.
  --sets            upload the document rows and build the sets on the volume (modal_act1.u1_sets, CPU): the three
                    document sets, the held-out documents, the practice set, the streams, the old read path's pairs
                    file. Four assertions run in the container first. Needs gen_docs.py --build's files and its
                    recorded hand read, and committed code. A rebuild is a new --root; the old record is kept. Sets
                    that carry a dose check or a main run are rebuilt only as a revision (changed code or inputs),
                    and a rebuild that does not end in new sets leaves the old record as it was.
  --plumb TAG       a rehearsal before the code is frozen: arm N, seed 0, one epoch of documents, one epoch of
                    practice, then phase 3 for six steps with reads at steps 0, 3 and 6 on four read pairs (a read
                    inside a round, and a read at the end of training). It runs the chain documents to practice to
                    press once, on the real sets. Prints seconds and costs only.
  --dose TAG        the dose check (revision 3): told_s0 and told_s1 each once through the old read path
                    (modal_retreat2.evaluate_adapter; one launch, two runs), then phase 3 (modal_act1.press) for 12
                    steps of the main stream on each, read at steps 0, 3, 6 and 12. Then the checks (for each adapter
                    the two read paths agree at step 0; no band) and the dose rule, on the one-off eval pairs the told
                    adapters never practised. A pooled release of zero at step 0 fails it too: there is no release
                    to halve. The code is frozen here: the commit and the sha256 of every file the run depends on
                    are recorded before launch. It needs a rehearsal that passed on the same sets. One dose check
                    per set of sets: a check left unfinished may be taken up under a new tag only by the very code
                    it froze.
  --block K TAG     one stage of block K (1: seeds 0-2, 2: seeds 3-5, 3: seeds 6-7) of the main run, every arm of
                    the run:
    --stage docs      phase 1, modal_sft.finetune on the arm's 3,000 documents, a fresh LoRA
    --stage practice  phase 2, modal_sft.finetune on the practice set, from the phase 1 adapter
    --stage press     phase 3, modal_act1.press from the phase 2 adapter, trained to the later of the two steps the
                      dose record fixed and read at step 0 and at those two steps
    --arms N,E        the run's arms, on the tag's first launch only: N and E (the default), or N,R,E. A run
                      always has N and E. Arm R needs a raise of the stop on record (stop.json). Afterwards the
                      arms are the ones run.json holds, and another --arms is refused.
                    A stage refuses to start if the code differs from the freeze or if the stage before it has not
                    finished for a seed. A run starts only if it can finish under the stop: at a tag's first launch,
                    step 3's spend so far plus the expected cost of the whole run must not pass it. A block starts
                    only in order, after the look (blocks 2 and 3: compare_u1.py --look must say go), with no
                    reserve left unsettled, and if step 3's spend so far plus the block's expected cost does not
                    pass the stop. A run the stop has ended starts no further block. A run that fails is launched
                    again by the same command; after two failures its seed is dropped. One main tag per set of sets.
  --add-arm R TAG   add the rule arm to a main run that has started. It needs a raise of the stop on record
                    (stop.json: the new stop and the ledger entry of the approval). Refused once any readout
                    exists, once the run has ended, if step 3's spend so far plus the arm's expected cost for every
                    seed of the blocks already started would pass the stop, or if the run as a whole, with the arm
                    in it, could not finish under the stop. Nothing is launched: the ordinary --block commands then
                    launch the arm's missing runs, stage by stage. The addition is recorded in run.json and in
                    variants.jsonl.
  --fetch TAG       after an interrupted launch: copy what the containers wrote on the volume into the results folder
                    (no container), then settle the reserve that launch left booked, once its runs have finished or
                    their containers' time has passed. If the volume cannot be read, nothing is settled.

Reads (as data; no task, document or reply is printed or run): Gate 0's train split and act1-full's split.json
(run_s1.act1_split); the pilot's document files; gen_docs.py's files; Stage A's episodes file and the told set's
records (pair names, outcomes and counts only). Writes results_root/explore/<OUT_DATE>/u1/: sets.json, <tag>/...,
variants.jsonl and spend.json. Every cost goes to the day's spend file under tool "u1": the upper bound is booked
before launch and trued up after (the reserve and true-up of run_gate.py and run_s1.py), checked first against the
day's remaining cap. A container's own timeout is set to the hours its reserve covers; the launcher waits longer than
that before it cancels, because time spent queued for a GPU is not the run's time.

  PYTHONPATH=src ~/prj/matlab/.venv/bin/python lab/explore/understand/run_u1.py --dry
  PYTHONPATH=src ~/prj/matlab/.venv/bin/python lab/explore/understand/run_u1.py --dose d1
  PYTHONPATH=src ~/prj/matlab/.venv/bin/python lab/explore/understand/run_u1.py --block 1 main --stage docs
"""

from __future__ import annotations

import argparse
import calendar
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
H4 = REPO / "lab" / "experiments" / "h4lite"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import u1_core as U  # noqa: E402

# ------------------------------------------------------------------------------------------------ this file's constants

OUT_DATE = "2026-11-10"  # the folder step 3's results and spend sit under (the design note's date)
TOOL = "u1"
OVERHEAD_S = 120  # start-up and scale-down per GPU container, as run_act1 meters it
VOLUME_ROOT = "u1"  # the sets and the runs sit under /runs/act1/u1 on the runs volume
RUNS_ACT1, RUNS_H4 = "/runs/act1", "/runs/h4lite"  # modal_act1.ROOT and modal_common.ROOT (a test compares them)
TOLD_ADAPTER = RUNS_H4 + "/durability/{name}/adapter"  # the told adapters of 30 Oct
OLD_READ_REL = "u1/{tag}/old_{name}"  # under /runs/h4lite: where the old read path writes its two files
# Container hours a launch reserves for each run, and the container's own timeout. A 3-seed block's three stages fit
# one UTC day under the daily cap only because each stage is trued up before the next is reserved; if the press stage
# does not fit, launch it for fewer seeds at a time (--seeds).
UPPER_H = {"sets": 0.5, "docs": 1.0, "practice": 0.5, "press": 1.15, "dose": 1.3, "old": 0.75}
# Hours the launcher waits beyond a run's reserve before it cancels it. Time a container spends queued for a GPU is
# not the run's time, so it must not count against the run. The container's own timeout stays the reserve's hours,
# and that is what caps the bill.
QUEUE_SLACK_H = 1.5
# An interrupted launch's reserve is settled once its hours, the queue slack and this grace have passed: nobody is
# left to cancel a container that queued, so it may start late and still run its whole time.
FETCH_GRACE_S = 600
# The older stack (modal_sft.finetune, modal_retreat2.evaluate_adapter) loads the model by its hub id. With the hub
# offline that id resolves through refs/main in the cache, which names the pinned snapshot, and nothing is asked of
# the hub (modal_act1's own functions are already offline, and load the pinned path).
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1"}
SETS_CORES, SETS_MEM_GIB = 4.0, 16.0  # modal_act1.u1_sets' container
STAGES = ("docs", "practice", "press")
MAX_TRIES = 2  # a crashed run is rerun once with identical inputs; after two failures its seed is dropped
# The rehearsal (--plumb): one run through the three stages, cut short. Phase 3 trains six steps (half a round, so
# the round is cut short) and reads at step 3 (inside the round) and at step 6 (the end of training).
PLUMB = {"arm": "N", "seed": 0, "epochs": 1, "steps": 6, "read_steps": [3, 6], "pairs": 4}
PLUMB_UPPER_H = {"docs": 0.6, "practice": 0.35, "press": 0.5}
READOUTS = ("compare_u1.json", "compare_u1_ended.json")  # compare_u1.py's records: once one exists, no arm is added
PILOT_DOCS = H4 / "results" / "declared" / "docs"
STAGE_A_EPISODES = H4 / "results" / "sft" / "koan" / "episodes.jsonl"
TOLD_DIR = H4 / "results" / "durability" / "told"
# The code a run depends on, beside every file under lab/explore/understand/: the freeze hashes these, every Python
# file of the library (src/anatta) and every file under h4lite's third_party. The last entry is data, not code: the
# tracked file the read's pair rows come from (it is hashed, never printed).
FREEZE_FILES = (
    "lab/explore/act1/modal_act1.py", "lab/explore/act1/act1_core.py", "lab/explore/shaping/replay.py",
    "lab/explore/shaping/run_s1.py", "lab/explore/gate/gate_core.py",
    "lab/experiments/h4lite/modal_sft.py", "lab/experiments/h4lite/modal_retreat2.py",
    "lab/experiments/h4lite/modal_common.py", "lab/experiments/h4lite/koan_env.py", "lab/experiments/h4lite/player.py",
    "lab/experiments/h4lite/pilot.py", "lab/experiments/h4lite/koan_pairs.py",
    "lab/experiments/h4lite/declared/declared_core.py", "lab/experiments/h4lite/declared/config.json",
    "lab/experiments/h4lite/declared/gen_declarations.py",
    "lab/experiments/h4lite/results/gate0/split/train.jsonl",
)
# The code that builds the sets and that the runs then train and read with, beside the library: the sets must have
# been built by this code as it is when the dose check freezes it.
SETS_FILES = (
    "lab/explore/understand/u1_core.py", "lab/explore/act1/modal_act1.py", "lab/explore/act1/act1_core.py",
    "lab/experiments/h4lite/declared/declared_core.py", "lab/experiments/h4lite/modal_common.py",
    "lab/experiments/h4lite/koan_env.py", "lab/experiments/h4lite/player.py",
)
# The code that selects the E documents: gen_docs.py's manifest records these files' hashes.
DOCS_CODE = (
    "lab/explore/understand/u1_core.py", "lab/explore/understand/gen_docs.py",
    "lab/experiments/h4lite/declared/declared_core.py", "lab/experiments/h4lite/declared/config.json",
)


def _paths() -> None:
    for p in (REPO / "src", HERE, HERE.parent / "act1", HERE.parent / "shaping", H4, H4 / "declared"):
        if str(p.resolve()) not in sys.path:
            sys.path.insert(0, str(p.resolve()))


def out_dir() -> Path:
    from anatta.config import load

    return load().results_root / "explore" / OUT_DATE / "u1"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def from_utc(stamp: str) -> float:
    """A utc() stamp as seconds since the epoch."""
    return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))


def settle_after(reserve_utc: str, upper_h: float) -> float:
    """The time (seconds since the epoch) after which an interrupted launch's containers can no longer be running."""
    return from_utc(reserve_utc) + (float(upper_h) + QUEUE_SLACK_H) * 3600 + FETCH_GRACE_S


def iter_jsonl(path: Path):
    """A JSONL file's rows, one at a time. Read as data."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """A JSONL file's rows (the first `limit` when given). Read as data."""
    out = []
    for row in iter_jsonl(path):
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def sha_file(path: Path) -> str | None:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


# ------------------------------------------------------------------------------------------------ spend


def spend_stages(out: Path) -> list[dict]:
    """spend.json's entries, in the order they were booked."""
    f = out / "spend.json"
    return json.loads(f.read_text())["stages"] if f.exists() else []


def spent(out: Path) -> float:
    f = out / "spend.json"
    return json.loads(f.read_text())["usd"] if f.exists() else 0.0


def booked(out: Path, stage: str) -> float:
    """What spend.json holds for one stage so far (its reserves and true-ups added up)."""
    return sum(s["usd"] for s in spend_stages(out) if s["stage"] == stage)


def unsettled(out: Path) -> list[dict]:
    """The reserves no true-up has followed: for each stage, its last entry when that entry is a reserve. A launch
    books its reserve, runs, then trues it up, so a reserve left last is a launch that was interrupted."""
    last = {}
    for x in spend_stages(out):
        last[x["stage"]] = x
    return [x for x in last.values() if x.get("kind") == "reserve"]


def guard(out: Path, upper: float, hint: str = "") -> None:
    """Refuse a launch whose upper bound does not fit in the day's remaining cap. The design's own stop (SPEND_STOP)
    is checked once, when a block starts (block_stop), so a seed's three arms finish together or not at all."""
    from anatta import spend as S

    left_day = S.daily_cap() - S.spent_today()
    if upper > left_day:
        raise SystemExit(f"refused: upper ${upper:.2f} > the day's remaining ${left_day:.2f}" + hint)


def book(out: Path, stage: str, usd: float, calls: int, date: str, entry: dict) -> None:
    """One booking, to the day file of `date` (the launch's UTC date) and to OUT/spend.json."""
    from anatta import spend as S

    if "PYTEST_CURRENT_TEST" in os.environ and getattr(S.add_spend, "__module__", "") == "anatta.spend":
        raise RuntimeError("a test must not book real spend: stub anatta.spend.add_spend")
    S.add_spend(None, tool=TOOL, label=f"{OUT_DATE}/u1/{stage}", usd=usd, calls=calls, date=date)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "spend.json"
    rec = json.loads(f.read_text()) if f.exists() else {"usd": 0.0, "stages": []}
    rec["usd"] = round(rec["usd"] + usd, 4)
    rec["stages"].append({"utc": utc(), "stage": stage, "day": date, "usd": round(usd, 4), **entry})
    f.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"spend {stage} ({entry.get('kind', 'cost')}): ${usd:+.3f}; step 3 total ${rec['usd']:.3f}")


def variant(out: Path, entry: dict) -> None:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "lab/explore"],
                                capture_output=True, text=True).stdout.strip())
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "variants.jsonl", "a") as f:
        f.write(json.dumps({"utc": utc(), "commit": head, "dirty": dirty, **entry}) + "\n")


def run_usd(kind: str, seconds: float) -> float:
    """What one container cost. press: its seconds plus start-up, at the act1 rate. finetune: its training-loop
    seconds at the rate Modal billed the pilot's fine-tunes per loop-hour (start-up is inside that rate). old and a
    failed run: wall-clock seconds. sets: the CPU container."""
    from anatta import spend as S

    if kind == "press":
        return (seconds + OVERHEAD_S) / 3600 * U.USD_H_ACT1
    if kind == "sets":
        return seconds / 3600 * (SETS_CORES * S.CPU_USD_PER_CORE_HOUR + SETS_MEM_GIB * S.MEM_USD_PER_GIB_HOUR)
    return seconds / 3600 * U.USD_H_SFT


def run_upper(kind: str, upper_h: float) -> float:
    """The reserve for one container: the whole of its timeout."""
    return run_usd(kind, upper_h * 3600)


# ------------------------------------------------------------------------------------------------ the local inputs


def eval_split() -> dict:
    """Act I's split of Gate 0's train pairs, recomputed the way run_act1 does and checked against act1-full's
    split.json (run_s1.act1_split): the rows, the 72 eval pairs, and the 38 one-off ones the read uses."""
    _paths()
    import act1_core as C
    import run_s1

    rows, _, _, ev1 = run_s1.act1_split()
    _, ev = C.split_names([r["name"] for r in rows])
    return {"rows": rows, "eval": ev, "oneoff": ev1}


def local_docs(out: Path) -> tuple[dict, dict, dict]:
    """({arm: its 3,000 documents}, {arm: its 200 held-out ones}, a record of the files). Every arm trains on the
    first 3,000 rows of its file, in file order, that do not echo their brief (u1_core.clean_docs); the file is read
    as far as that needs. R and N: the pilot's declared and neutral documents, which must give the fingerprints
    u1_core.DOC_SOURCES pins. E: gen_docs.py's files, which must be the ones its manifest names, selected by the code
    as it is now. The held-out sets are the first 200 rows of the held files as they stand."""
    docs, held, rec = {}, {}, {}
    man_file = out / "docs" / "manifest.json"
    for arm, src in U.DOC_SOURCES.items():
        folder = out / "docs" if arm == "E" else PILOT_DOCS
        t, h = folder / src["train"], folder / src["held"]
        if not t.is_file() or not h.is_file():
            raise SystemExit(f"refused: {t} or {h} is missing" + (" (run gen_docs.py --build first)" if arm == "E" else ""))
        try:
            rows, stats = U.clean_docs(iter_jsonl(t), src["kind"], U.N_DOCS)
            hrows = U.first_docs(read_jsonl(h, U.N_HELD), src["kind"], U.N_HELD)
        except ValueError as e:
            raise SystemExit(f"refused: arm {arm}'s documents: {e}") from None
        fp = U.docs_fingerprint(rows)
        if src["fingerprint"] and fp != src["fingerprint"]:
            raise SystemExit(f"refused: {t} does not hold the documents this design names for arm {arm}: the first "
                             f"{U.N_DOCS} rows that do not echo their brief give another fingerprint")
        docs[arm], held[arm] = [U.slim_doc(r) for r in rows], [U.slim_doc(r) for r in hrows]
        rec[arm] = {"train": str(t), "held": str(h), "train_sha256": sha_file(t), "held_sha256": sha_file(h),
                    "fingerprint": fp, "rows_read": stats["read"], "echo_passed_over": stats["echo"]}
    man = json.loads(man_file.read_text()) if man_file.is_file() else None
    names = (U.DOC_SOURCES["E"]["train"], U.DOC_SOURCES["E"]["held"])
    if man is None or any(man.get("files", {}).get(n, {}).get("sha256") != rec["E"][f"{k}_sha256"]
                          for n, k in zip(names, ("train", "held"))):
        raise SystemExit(f"refused: {man_file} is missing or names other files than the ones on disk")
    if man.get("code_sha256") != docs_code_sha():
        raise SystemExit(f"refused: {man_file} was written by other code than the code as it is now; run gen_docs.py "
                         "--build again (it selects again from its cache and pays for nothing it already has)")
    rec["E"]["manifest_sha256"] = sha_file(man_file)
    rec["E"]["judge_form"] = man.get("judge", {}).get("form")
    rec["E"]["judge_rubric_sha256"] = man.get("judge", {}).get("rubric_sha256")
    return docs, held, rec


def e_files_sha(folder: Path) -> dict:
    """{file name: sha256} of the two kept E files. A hand read is a read of these documents: it stays good while
    they are byte for byte the same, also when the manifest beside them is written again."""
    return {U.E_FILES[k]: sha_file(folder / U.E_FILES[k]) for k in ("held", "train")}


def hand_read_failures(rec: dict) -> list[int]:
    """The violation counts, HAND_READ_VIOLATIONS or more, among every read on record of the documents the latest
    read is of: the latest itself and the ones kept under "earlier". Such a read is binding. A later read of the same
    documents with fewer violations does not undo it: the documents must change first (the judge prompt tightened,
    every document screened again)."""
    same = [r for r in [rec, *(rec.get("earlier") or [])] if r.get("files_sha256") == rec.get("files_sha256")]
    return [r["violations"] for r in same
            if isinstance(r.get("violations"), int) and r["violations"] >= U.HAND_READ_VIOLATIONS]


def hand_read_check(out: Path) -> dict:
    """The read by hand of kept E documents (gen_docs.py --hand-read) must be on record before the sets are built: of
    the documents on disk, of the sample drawn under HAND_READ_SEED, at least HAND_READ_N of them, and with no read of
    these documents on record, the latest or an earlier one, that found HAND_READ_VIOLATIONS violations or more."""
    f = out / "docs" / "hand_read.json"
    if not f.is_file():
        raise SystemExit(f"refused: no hand read is recorded ({f}); read a sample (gen_docs.py --sample "
                         f"{U.HAND_READ_N} --seed {U.HAND_READ_SEED}) and record it (gen_docs.py --hand-read)")
    rec = json.loads(f.read_text())
    now = e_files_sha(out / "docs")
    if None in now.values() or rec.get("files_sha256") != now:
        raise SystemExit(f"refused: the hand read in {f} is of other documents than the ones on disk; sample and "
                         "record it again")
    if rec.get("seed") != U.HAND_READ_SEED:
        raise SystemExit(f"refused: the hand read in {f} is of a sample drawn under seed {rec.get('seed')}; the read "
                         f"that counts is of the sample under seed {U.HAND_READ_SEED}")
    n, bad, failed = rec.get("n"), rec.get("violations"), hand_read_failures(rec)
    if not isinstance(n, int) or not isinstance(bad, int) or n < U.HAND_READ_N or failed:
        raise SystemExit(f"refused: the hand read records {n} documents and {bad} violations"
                         + (f", and a read of these same documents found {max(failed)}" if failed and bad not in failed else "")
                         + f"; the sets need at least {U.HAND_READ_N} read and fewer than {U.HAND_READ_VIOLATIONS} "
                         "violations in every read of the documents (with more, the judge prompt is tightened and every "
                         "document screened again: the documents must change before a new read counts)")
    return {k: rec.get(k) for k in ("utc", "seed", "n", "violations")}


def local_expect(split: dict, docs: dict) -> dict:
    """What the laptop's records say the volume's files hold (u1_core.expectations). From Stage A's episodes file and
    the told set's records only pair names, twins, outcomes and counts are kept."""
    for f in (STAGE_A_EPISODES, TOLD_DIR / "data.json", TOLD_DIR / "releases.jsonl"):
        if not f.is_file():
            raise SystemExit(f"refused: {f} is missing (a local-only record this build checks the volume against)")
    stage_a = [{"pair": e["pair"], "impossible": e["impossible"], "outcome": e["outcome"]}
               for e in read_jsonl(STAGE_A_EPISODES)]
    told = json.loads((TOLD_DIR / "data.json").read_text())
    releases = [{"pair": r["pair"]} for r in read_jsonl(TOLD_DIR / "releases.jsonl") if r["arm"] == "told"]
    return U.expectations(stage_a, split["eval"], split["oneoff"], docs, told, releases)


def read_sets(out: Path) -> dict:
    f = out / "sets.json"
    if not f.is_file():
        raise SystemExit(f"refused: no {f}; build the sets first (--sets)")
    return json.loads(f.read_text())


# ------------------------------------------------------------------------------------------------ the freeze


def _is_code(p: Path, base: Path) -> bool:
    """A file the freeze hashes. Left out: Python's caches, and anything under a name that starts with a dot
    (.pytest_cache, the Finder's .DS_Store), none of which is code."""
    parts = p.relative_to(base).parts
    return p.is_file() and p.suffix != ".pyc" and not any(x.startswith(".") or x == "__pycache__" for x in parts)


def src_files() -> list[Path]:
    """Every Python file of the library (src/anatta)."""
    return sorted(p for p in (REPO / "src" / "anatta").rglob("*.py") if _is_code(p, REPO))


def shared_paths() -> list[Path]:
    """The code a run depends on outside this folder: FREEZE_FILES, the library, and h4lite's third_party."""
    third = sorted(p for p in (H4 / "third_party").rglob("*") if _is_code(p, REPO))
    return [REPO / f for f in FREEZE_FILES] + src_files() + third


def freeze_paths(here: Path | None = None) -> list[Path]:
    """Every file under lab/explore/understand/ (tests too), then shared_paths()."""
    base = here or HERE
    return sorted(p for p in base.rglob("*") if _is_code(p, base)) + shared_paths()


def git_state(paths: list[Path]) -> tuple[str, list[str]]:
    """(HEAD, the uncommitted or untracked files among `paths`, as git status names them)."""
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rels = sorted({p.relative_to(REPO).as_posix() for p in paths if p.is_file()} | {HERE.relative_to(REPO).as_posix()})
    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *rels], capture_output=True, text=True)
    return head, [x for x in st.stdout.splitlines() if x.strip()]


def code_freeze(out: Path) -> dict:
    """The code as it is now: the commit, the files git calls changed, the sha256 of every file a run depends on,
    and the sets record it stands on (its sha256 and the sets' root: the streams and sets the dose check ran on)."""
    paths = freeze_paths()
    head, dirty = git_state(paths)
    sfile = out / "sets.json"
    return {"utc": utc(), "commit": head, "dirty": dirty,
            "files": {p.relative_to(REPO).as_posix(): sha_file(p) for p in paths},
            "sets_sha256": sha_file(sfile),
            "root": json.loads(sfile.read_text()).get("root") if sfile.is_file() else None}


def freeze_diffs(frozen: dict, now: dict) -> list[str]:
    """What differs between a recorded freeze and the code now: changed, missing and new files, and the sets."""
    a, b = frozen.get("files") or {}, now.get("files") or {}
    out = [f"{f} changed" for f in sorted(set(a) & set(b)) if a[f] != b[f]]
    out += [f"{f} is gone" for f in sorted(set(a) - set(b))] + [f"{f} is new" for f in sorted(set(b) - set(a))]
    if frozen.get("sets_sha256") != now.get("sets_sha256"):
        out.append("sets.json changed (the sets were rebuilt)")
    return out


def check_frozen(out: Path, frozen: dict) -> None:
    diffs = freeze_diffs(frozen, code_freeze(out))
    if diffs:
        raise SystemExit(f"refused: the code differs from the freeze the dose check recorded (commit "
                         f"{str(frozen.get('commit'))[:10]}): {'; '.join(diffs[:8])}")


def sets_code() -> dict:
    """{path: sha256} of the code that builds the sets and that the runs then train and read with: SETS_FILES and
    the library. sets.json records it; the dose check's first launch needs it unchanged."""
    return {p.relative_to(REPO).as_posix(): sha_file(p) for p in [REPO / f for f in SETS_FILES] + src_files()}


def docs_code_sha() -> dict:
    """{file name: sha256} of the code that selects the E documents (gen_docs.py's manifest records it)."""
    return {Path(f).name: sha_file(REPO / f) for f in DOCS_CODE}


# ------------------------------------------------------------------------------------------------ run records


def tries(d: Path) -> int:
    """How many times a run has failed (its error_try<k>.json files)."""
    return len(list(d.glob("error_try*.json"))) if d.is_dir() else 0


def volume_label(label: str, attempt: int) -> str:
    """The folder a press run writes on the volume: its label, or label_try<k> for a rerun (press never overwrites)."""
    return label if attempt <= 1 else f"{label}_try{attempt}"


def record_error(d: Path, error: str, **extra) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"error_try{tries(d) + 1}.json").write_text(json.dumps({"utc": utc(), "error": error, **extra}, indent=1) + "\n")


def save_press(d: Path, res: dict, **extra) -> None:
    """One press run's files: read_s<step>.jsonl (episode records; the replies are model text, kept as data),
    log.jsonl and info.json (everything else the container returned)."""
    d.mkdir(parents=True, exist_ok=True)
    res = dict(res)
    for s, recs in (res.pop("reads", None) or {}).items():
        write_jsonl(d / f"read_s{int(s)}.jsonl", recs)
    write_jsonl(d / "log.jsonl", res.pop("log", None) or [])
    (d / "info.json").write_text(json.dumps({**res, **extra}, indent=1) + "\n")


def save_finetune(d: Path, res: dict) -> None:
    """One fine-tune's files as modal_sft.finetune returns them: args.json, log.jsonl, done.json."""
    d.mkdir(parents=True, exist_ok=True)
    for name, text in (res.get("files") or {}).items():
        (d / name).write_text(text)


def read_dropped(d: Path) -> dict:
    f = d / "dropped.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def drop_seed(d: Path, seed: int, why: str) -> None:
    """Record a seed as dropped from every contrast (a run of it failed twice)."""
    rec = read_dropped(d)
    rec.setdefault(str(seed), {"utc": utc(), "why": why})
    d.mkdir(parents=True, exist_ok=True)
    (d / "dropped.json").write_text(json.dumps(rec, indent=1) + "\n")
    log(f"seed {seed} is dropped from every contrast: {why}")


def stage_dir(d: Path, stage: str, lab: str) -> Path:
    return d / lab if stage == "press" else d / stage / lab


def stage_done(d: Path, stage: str, lab: str) -> bool:
    """A stage's finished run: the fine-tune's done.json, or the press run's info.json."""
    return (stage_dir(d, stage, lab) / ("info.json" if stage == "press" else "done.json")).is_file()


TAG_RECORDS = ("run.json", "plumb.json", "dose.json", "freeze.json")  # each names the sets' root its tag stands on


def tag_root(out: Path, d: Path) -> str:
    """The sets' root a tag runs on: its own record's when it has one (TAG_RECORDS; a dose check's freeze.json names
    it from its first launch), else the root of the sets as they are now. A tag stays on the sets it started on, also
    after a rebuild."""
    for name in TAG_RECORDS:
        f = d / name
        if f.is_file() and (root := json.loads(f.read_text()).get("root")):
            return root
    return read_sets(out)["root"]


def tags_on(out: Path, root: str, name: str) -> list[str]:
    """The tags that hold the record `name` (one of TAG_RECORDS) and stand on the sets under `root`."""
    return [f.parent.name for f in (sorted(out.glob(f"*/{name}")) if out.is_dir() else [])
            if json.loads(f.read_text()).get("root") == root]


def result_file(d: Path, where: str, lab: str) -> Path:
    """The local file that says a launched run finished. where: "docs", "practice", "press", or "old" (the old read
    path, whose label is its folder)."""
    if where == "old":
        return d / lab / "heldout.jsonl"
    return stage_dir(d, where, lab) / ("info.json" if where == "press" else "done.json")


# ------------------------------------------------------------------------------------------------ launching


def _modal_fn(kind: str):
    """(the Modal app, its function) for a kind of run. Imported here, so the module itself never imports modal."""
    _paths()
    if kind in ("press", "sets"):
        import modal_act1 as M

        return M.app, (M.press if kind == "press" else M.u1_sets)
    if kind == "finetune":
        import modal_sft as M

        return M.app, M.finetune
    import modal_retreat2 as M

    return M.app, M.evaluate_adapter


def _refuse_under_pytest(modal) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ and getattr(modal, "__file__", None):
        raise RuntimeError("a test must not launch on Modal: stub the modal modules")


def launch(out: Path, stage: str, kind: str, jobs: list[dict], upper_h: float, tag: str | None = None,
           where: str | None = None, block: int | None = None) -> dict:
    """Reserve, spawn one container per job ({"label", "kwargs"}), collect, true up. Returns {label: the function's
    result, or {"error": ...}}. The reserve's entry carries the launch (its tag, kind, hours, runs, and where their
    results go), so an interrupted launch can be settled later (--fetch). A stage whose last entry is a reserve is
    refused: that launch was interrupted and its containers may still be running.

    A container's own timeout is the hours its reserve covers. The launcher waits QUEUE_SLACK_H longer than that
    before it cancels a run, so time queued for a GPU does not count against it; a run cancelled then is charged its
    whole reserve. An interrupt escapes unsettled: the reserve stays booked and the containers run on to their own
    timeout. So does a broken connection to Modal while waiting: the run may still be going, so it is no failed run.

    A launch that never started is no failed run either. If the app cannot be entered or a spawn raises, whatever
    was spawned is cancelled, the reserve is trued up to what was used, and the command exits saying so. No error is
    recorded against any run."""
    left = [x for x in unsettled(out) if x["stage"] == stage]
    if left:
        raise SystemExit(f"refused: an earlier launch of {stage} was interrupted (its reserve of ${left[0]['usd']:.2f} "
                         "has no true-up) and its containers may still be running. "
                         + (f"Copy what they wrote and settle it: --fetch {tag}" if tag else
                            "It is settled once its container's time has passed"))
    import modal

    from anatta import spend as S

    app, fn = _modal_fn(kind)
    _refuse_under_pytest(modal)
    upper = len(jobs) * run_upper(kind, upper_h)
    guard(out, upper)
    day = S.utc_date()
    book(out, stage, upper, 0, day, {"kind": "reserve", "launch": {
        "tag": tag, "kind": kind, "upper_h": upper_h, "runs": [j["label"] for j in jobs], "where": where,
        "block": block}})
    t0 = time.time()
    deadline = t0 + (upper_h + QUEUE_SLACK_H) * 3600  # queue time is not the run's time
    opts: dict = {"timeout": int(upper_h * 3600)}  # a detached container cannot outrun its reserve
    if kind in ("finetune", "old"):
        opts["env"] = dict(OFFLINE_ENV)  # the older stack must not ask the hub (see OFFLINE_ENV)
    res, secs, usd = {}, {}, 0.0
    wall = kind if kind in ("press", "sets") else "old"  # the kind whose price is per wall-clock second
    calls: dict = {}
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(app.run(detach=True))
            bound = fn.with_options(**opts)
            for j in jobs:
                calls[j["label"]] = bound.spawn(**j["kwargs"])
        except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
            # The app was not entered, a spawn raised, or the launch was interrupted before every run was spawned:
            # the launch did not start. (An interrupt later, while waiting, leaves the containers running.)
            used = 0.0
            for c in calls.values():  # stop whatever was spawned; it is charged its wall-clock seconds so far
                with contextlib.suppress(Exception):
                    c.cancel(terminate_containers=True)
                used += run_usd(wall, time.time() - t0)
            book(out, stage, used - upper, len(calls), day,
                 {"kind": "true-up", "not_started": True, "metered_usd": round(used, 4), "reserved_usd": round(upper, 4),
                  "spawned_then_cancelled": list(calls), "error": repr(e)[:300]})
            if isinstance(e, KeyboardInterrupt):
                raise
            raise SystemExit(f"the launch of {stage} did not start ({repr(e)[:200]}). {len(calls)} of {len(jobs)} runs "
                             "had been spawned and were cancelled; the reserve is trued up to what was used. No run is "
                             "counted as failed: launch again") from None
        for lab, c in calls.items():
            try:
                res[lab] = c.get(timeout=max(1.0, deadline - time.time()))
                if kind == "press":
                    secs[lab] = float(res[lab]["seconds"])
                elif kind == "finetune":
                    secs[lab] = float(json.loads(res[lab]["files"]["done.json"])["seconds"])
                elif kind == "sets":
                    secs[lab] = float(res[lab]["seconds"])
                else:
                    secs[lab] = time.time() - t0
                usd += run_usd(kind, secs[lab])
            except (TimeoutError, modal.exception.TimeoutError) as e:
                c.cancel(terminate_containers=True)
                res[lab] = {"error": (f"cancelled at the local deadline ({upper_h} h a run, plus {QUEUE_SLACK_H} h "
                                      f"for queueing): {e!r}")[:500]}
                secs[lab] = upper_h * 3600
                usd += run_upper(kind, upper_h)
                log("cancelled", lab, "at the local deadline")
            except (ConnectionError, modal.exception.ConnectionError):
                # The connection broke, not the run: its container may still be going. This is an interrupted launch,
                # not a failed run: the reserve stays booked, and --fetch settles it.
                log("the connection to Modal broke while waiting for", lab, "- the launch is interrupted, no run has "
                    "failed" + (f"; settle it with --fetch {tag}" if tag else ""))
                raise
            except (Exception, SystemExit) as e:  # noqa: BLE001  (a container may end with SystemExit; an interrupt escapes)
                res[lab] = {"error": repr(e)[:500]}
                secs[lab] = time.time() - t0
                usd += run_usd(wall, secs[lab])  # wall-clock seconds
                log("error", lab, repr(e)[:300])
    book(out, stage, usd - upper, max(len(jobs), 1), day,
         {"kind": "true-up", "metered_usd": round(usd, 4), "reserved_usd": round(upper, 4),
          "container_s": {k: round(v, 1) for k, v in secs.items()}, "runs": list(res)})
    return res


# ------------------------------------------------------------------------------------------------ --sets


def settle_untagged(out: Path, stage: str, strict: bool = True) -> None:
    """A launch that belongs to no tag (the sets build) has no tag to fetch. If it was interrupted, its reserve is
    settled here, in full and on its own day, once its container's time has passed (settle_after). Before that:
    `strict` (a new launch of the stage) refuses, because the container may still be writing; otherwise (--fetch)
    it says when, and settles nothing."""
    for res in unsettled(out):
        if res["stage"] != stage:
            continue
        free = settle_after(res["utc"], (res.get("launch") or {}).get("upper_h") or UPPER_H["sets"])
        if time.time() < free:
            when = f"after {time.strftime('%H:%M', time.gmtime(free))} UTC"
            if strict:
                raise SystemExit(f"refused: an earlier launch of {stage} was interrupted and its container may still "
                                 f"be running; try again {when}")
            log(f"{stage}: its interrupted launch may still be running. Nothing is settled; fetch again {when}")
            continue
        book(out, stage, 0.0, 0, res["day"], {"kind": "true-up", "late": True, "metered_usd": res["usd"],
                                             "reserved_usd": res["usd"],
                                             "why": "the launch was interrupted; charged its whole reserve"})


def revision(out: Path, old: dict, files: dict, expect: dict) -> list[str]:
    """Why a rebuild of sets that already carry a dose check or a main run is a revision (an empty list: it is not).
    It is one when the sets' code differs from what built them, when the local inputs differ (the documents'
    fingerprints and file hashes, or what the volume's records are expected to hold), or when the code as it is now
    differs from a freeze recorded on those sets. Sets built again with nothing changed would only buy a second dose
    check, or a second main run, on the same thing."""
    why = []
    built, code = (old.get("code") or {}).get("files") or {}, sets_code()
    if any(built.get(f) != code.get(f) for f in set(built) | set(code)):
        why.append("the sets' code changed")
    was = old.get("files") or {}
    if any((was.get(arm) or {}).get(k) != rec.get(k) for arm, rec in files.items()
           for k in ("fingerprint", "train_sha256", "held_sha256")):
        why.append("the documents changed")
    if old.get("expect") != json.loads(json.dumps(expect)):
        why.append("what the volume's records are expected to hold changed")
    now = code_freeze(out)
    for t in tags_on(out, old.get("root"), "freeze.json"):
        if freeze_diffs(json.loads((out / t / "freeze.json").read_text()), now):
            why.append(f"the code differs from the freeze of {t}")
    return why


def cmd_sets(root: str, upper_h: float) -> int:
    out = out_dir()
    sfile = out / "sets.json"
    old = json.loads(sfile.read_text()) if sfile.exists() else None
    if old is not None and old.get("root") == root:
        raise SystemExit(f"refused: {sfile} exists: the sets are built under root {root!r}. A rebuild is a new --root "
                         "and a ledger line, and the dose check runs again on it")
    kept = out / f"sets.{old['root']}.json" if old is not None else None
    if kept is not None and kept.exists():
        raise SystemExit(f"refused: {kept} already exists; the old record is never overwritten")
    head, dirty = git_state(freeze_paths())
    if dirty:
        raise SystemExit("refused: the sets are built by committed code, so that their record names it: "
                         f"{'; '.join(dirty[:8])}")
    split = eval_split()
    docs, held, files = local_docs(out)
    files["E"]["hand_read"] = hand_read_check(out)
    expect = local_expect(split, docs)
    if old is not None:  # sets that carry a dose check or a main run are rebuilt only as a revision
        started = sorted(set(tags_on(out, old["root"], "freeze.json")) | set(tags_on(out, old["root"], "run.json")))
        if started and not revision(out, old, files, expect):
            raise SystemExit(f"refused: the sets under root {old['root']!r} carry a dose check or a main run "
                             f"({started}), and a rebuild would change nothing: the same code, the same documents, the "
                             "same volume records. A rebuild is for a revision; the same sets built again would only "
                             "buy a second dose check")
    settle_untagged(out, "sets")
    guard(out, run_upper("sets", upper_h))  # every refusal comes before the old record is touched
    log("documents", {a: len(v) for a, v in docs.items()}, "held", {a: len(v) for a, v in held.items()},
        "eval pairs", len(split["eval"]), "read pairs", len(split["oneoff"]), "unpractised", len(expect["unpractised"]),
        "practice releases", expect["practice_releases"])
    def restore(why: str) -> None:
        """A rebuild that did not end in new sets leaves the old record as the current one again."""
        if old is not None and kept.exists() and not sfile.exists():
            kept.rename(sfile)
            log(f"the rebuild {why}: the sets under root {old['root']!r} are the current ones again")
            variant(out, {"cmd": "sets-restored", "root": old["root"], "failed_root": root, "why": why})

    if old is not None:  # the rebuild: the old record is kept under its root's name, never deleted
        sfile.rename(kept)
        log(f"the sets under root {old['root']!r} are retired: their record is now {kept.name}")
        variant(out, {"cmd": "sets-retired", "old_root": old["root"], "record": kept.name, "new_root": root})
    variant(out, {"cmd": "sets", "root": root, "files": files, "expect": expect, "model": U.MODEL_KEY,
                  "upper_usd": round(run_upper("sets", upper_h), 2)})
    try:
        res = launch(out, "sets", "sets", [{"label": "sets", "kwargs": dict(
            docs=docs, held=held, pairs_rows=split["rows"], eval_names=split["eval"], eval_oneoff=split["oneoff"],
            expect=expect, model_key=U.MODEL_KEY, root=root)}], upper_h, where="sets")["sets"]
    except BaseException:  # an interrupt too: whatever stops the launch, sets.json is left as it was
        restore("was interrupted or did not start")
        raise
    if "error" in res:
        (out / "sets_error.json").write_text(json.dumps({"utc": utc(), "root": root, **res}, indent=1) + "\n")
        restore("failed and built nothing")
        raise SystemExit(f"the sets were not built: {res['error']}")
    sfile.write_text(json.dumps({"utc": utc(), "root": root, "files": files, "expect": expect,
                                 "eval_oneoff": split["oneoff"], "code": {"commit": head, "files": sets_code()},
                                 "info": res["info"]}, indent=1) + "\n")
    info = res["info"]
    log("sets built under", f"{RUNS_ACT1}/{root}", "checks", list(info["checks"]), "practice",
        {k: info["sets"]["practice"][k] for k in ("kept", "releases", "solves", "items", "tokens")})
    return 0


# ------------------------------------------------------------------------------------------------ --dose


def dose_jobs() -> list[dict]:
    """The dose check's two press runs (revision 3): each told adapter under the main stream, on its own seed's
    stream. The solve-only stream is not run again."""
    return [{"label": f"{U.MAIN_STREAM}_{ad}", "stream": U.MAIN_STREAM, "adapter": ad, "seed": int(ad.rsplit("_s", 1)[1])}
            for ad in U.DOSE_ADAPTERS]


def old_dirs(d: Path) -> dict:
    """{adapter: the folder its old-path read is kept in} for a dose tag: both told adapters (revision 3)."""
    return {ad: d / f"old_{ad}" for ad in U.DOSE_ADAPTERS}


def fetch_volume(rel: str) -> bytes:
    """One file of the runs volume, read from the laptop (no container)."""
    import modal

    from anatta.config import load

    _refuse_under_pytest(modal)
    vol = modal.Volume.from_name(load().modal.volumes.get("runs", "anatta-runs"))
    return b"".join(vol.read_file(rel))


def dose_record(d: Path, sets: dict) -> dict:
    """The dose check's numbers from a tag's local files (compare_u1.load_dose, which refuses a run or a read that
    is not the planned one; u1_core.dose_report)."""
    import compare_u1 as K

    runs, old = K.load_dose(d, sets)
    return U.dose_report(runs, old, sets["info"]["unpractised"], sets["eval_oneoff"])


def is_plumb(d: Path) -> bool:
    return (d / "plumb.json").exists()


def old_rel(tag: str, name: str, old_dir: Path) -> str:
    """Where one adapter's old-path read writes under /runs/h4lite: its folder, or folder_try<k> after an earlier
    failure."""
    rel = OLD_READ_REL.format(tag=tag, name=name)
    return rel if tries(old_dir) == 0 else f"{rel}_try{tries(old_dir) + 1}"


def cmd_dose(tag: str, upper_h: float, old_upper_h: float) -> int:
    out = out_dir()
    d = out / tag
    if is_plumb(d):
        raise SystemExit(f"refused: {tag!r} is a rehearsal's tag; the dose check takes a tag of its own")
    if (d / "dose.json").exists():
        raise SystemExit(f"refused: {d} already holds a dose record; pick a new tag")
    if (d / "run.json").exists():
        raise SystemExit(f"refused: {d} is a main run's tag")
    sets = read_sets(out)
    root = sets["root"]
    now = code_freeze(out)
    ffile = d / "freeze.json"
    # One dose record per set of sets, passed or not. (Checked at every launch, not only a tag's first: a tag that
    # stopped part way must not finish a second record after another tag has finished one.)
    done = [t for t in tags_on(out, root, "dose.json") if t != tag]
    if done:
        raise SystemExit(f"refused: the sets under root {root!r} already have a dose record ({done}), passed or "
                         "not. One dose check per set of sets: a second one needs new sets (--sets --root NEW)")
    if ffile.exists():  # a second launch under this tag (a part failed): the code must be the code that started it
        check_frozen(out, json.loads(ffile.read_text()))
    else:  # this tag's first launch
        built, code = (sets.get("code") or {}).get("files") or {}, sets_code()
        moved = sorted(f for f in set(built) | set(code) if built.get(f) != code.get(f))
        if moved:
            raise SystemExit("refused: the sets were built by other code than the code as it is now "
                             f"({'; '.join(moved[:6])}). The sets must be built by the code the dose check freezes: "
                             "rebuild them under a new root (--sets --root NEW)")
        if now["dirty"]:
            raise SystemExit(f"refused: the code is frozen at the dose check, so it must be committed first: "
                             f"{'; '.join(now['dirty'][:8])}")
        # A dose check has started on these sets once a freeze.json stands on them. A tag that was left unfinished
        # may be taken up again under a new tag only by the very code it froze: a check under changed code would be
        # a second look at the same sets.
        for t in tags_on(out, root, "freeze.json"):
            diffs = freeze_diffs(json.loads((out / t / "freeze.json").read_text()), now)
            if t != tag and diffs:
                raise SystemExit(f"refused: a dose check was started on these sets under tag {t!r} and left unfinished, "
                                 f"by other code than the code as it is now ({'; '.join(diffs[:4])}). A dose check "
                                 "under changed code needs new sets (--sets --root NEW)")
    # The chain documents to practice to press is rehearsed on these sets, by this code, before the code is frozen:
    # a rehearsal by other code (a launcher fix made after it) does not count. (After the checks above, so that code
    # that differs from the sets' or from a freeze is named as that.)
    passed = [t for t in tags_on(out, root, "plumb.json")
              if (rec := json.loads((out / t / "plumb.json").read_text())).get("ok") is True and rec.get("code") == now["files"]]
    if not passed:
        raise SystemExit(f"refused: no rehearsal has passed on the sets under root {root!r} under the code as it is "
                         "now: run --plumb TAG first (a new tag if the code changed since the last rehearsal)")
    split = eval_split()
    rows = [r for r in split["rows"] if r["name"] in set(split["oneoff"])]
    olds = old_dirs(d)
    todo = [j for j in dose_jobs() if not (d / j["label"] / "info.json").exists()]
    spent_out = ([j["label"] for j in todo if tries(d / j["label"]) >= MAX_TRIES]
                 + [od.name for od in olds.values() if tries(od) >= MAX_TRIES])
    if spent_out:
        raise SystemExit(f"refused: {spent_out} failed twice under tag {tag!r}; pick a new tag")
    earlier = sorted(f.parent.name for f in out.glob("*/freeze.json")
                     if f.parent.name != tag and not (f.parent / "dose.json").exists())
    d.mkdir(parents=True, exist_ok=True)
    if not ffile.exists():
        ffile.write_text(json.dumps(now, indent=1) + "\n")

    def copy_old(ad: str, rel: str) -> None:
        """One old read's episodes, from the volume (data; not printed). The read is finished and paid for; if only
        this copy fails, --fetch makes it and nothing is launched twice."""
        try:
            raw = fetch_volume(f"h4lite/{rel}/heldout.jsonl")
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"the old-path read of {ad} finished, but its episodes could not be copied from the volume "
                             f"({type(e).__name__}). Copy them with --fetch {tag}, then run --dose {tag} again") from None
        (olds[ad] / "heldout.jsonl").write_bytes(raw)

    for ad, od in olds.items():
        if (od / "heldout.json").exists() and not (od / "heldout.jsonl").exists():
            copy_old(ad, old_rel(tag, ad, od))  # the read finished in an earlier call and only its copy was missing
    need_old = [ad for ad, od in olds.items() if not (od / "heldout.jsonl").exists()]
    steps, read_steps = U.DOSE_STEPS[-1], list(U.DOSE_STEPS[1:])
    variant(out, {"cmd": "dose", "tag": tag, "root": root, "freeze_commit": now["commit"], "runs": [j["label"] for j in todo],
                  "old_reads": need_old, "steps": steps, "reads": list(U.DOSE_STEPS), "cfg": U.PRESS_CFG,
                  "read": U.READ, "upper_h": upper_h, "old_upper_h": old_upper_h, "earlier_incomplete": earlier})
    if need_old:  # the cheap, proven path first, for both adapters in one launch: if one cannot be read through it, the
        # check cannot pass, and the press runs are not launched
        o = U.OLD_PATH
        rels = {ad: old_rel(tag, ad, olds[ad]) for ad in need_old}
        res = launch(out, f"dose-{tag}-old", "old", [{"label": olds[ad].name, "kwargs": dict(
            adapter=TOLD_ADAPTER.format(name=ad), arm=o["arm"], out_rel=rels[ad], tag="heldout", eval_n=o["eval_n"],
            temperature=o["temperature"], attempts=o["attempts"], max_new=tuple(U.READ["max_new"]),
            gen_batch=U.READ["gen_batch"], eval_seed=o["eval_seed"], under_cut=False,
            pairs_file=sets["info"]["old_path_pairs"])} for ad in need_old], old_upper_h, tag=tag, where="old")
        failed_old = {}
        for ad in need_old:
            r = res[olds[ad].name]
            if "error" in r:
                record_error(olds[ad], r["error"], volume_rel=rels[ad])
                failed_old[ad] = r["error"]
            else:
                olds[ad].mkdir(parents=True, exist_ok=True)
                (olds[ad] / "heldout.json").write_text(json.dumps(r["eval"], indent=1) + "\n")  # it finished: the summary
        for ad in need_old:
            if ad not in failed_old:
                copy_old(ad, rels[ad])
                log("old-path read,", ad, {k: res[olds[ad].name]["eval"].get(k)
                                           for k in ("release", "hack", "solve", "held_pairs", "episodes")})
        if failed_old:
            first = next(iter(failed_old.values()))
            raise SystemExit(f"the old-path read failed for {sorted(failed_old)} ({first[:200]}); the press runs were "
                             "not launched. The same command launches a failed read once more")
    if todo:
        jobs = [{"label": j["label"], "kwargs": dict(
            init=TOLD_ADAPTER.format(name=j["adapter"]), label=volume_label(j["label"], tries(d / j["label"]) + 1),
            tag=tag, seed=j["seed"], stream=j["stream"], steps=steps, read_steps=read_steps,
            cfg=U.PRESS_CFG, pairs_rows=rows, eval_names=split["oneoff"], model_key=U.MODEL_KEY, root=root)}
            for j in todo]
        res = launch(out, f"dose-{tag}", "press", jobs, upper_h, tag=tag, where="press")
        for j, job in zip(todo, jobs):
            r = res[j["label"]]
            if "error" in r:
                record_error(d / j["label"], r["error"], volume_label=job["kwargs"]["label"])
            else:
                save_press(d / j["label"], r, volume_label=job["kwargs"]["label"], adapter=j["adapter"])
                # the dose check is not blinded: its counts are computed here, from the reads the run returned
                log(j["label"], f"impossible twins on all {len(split['oneoff'])} read pairs, by step (the dose table "
                    f"that follows is on the {len(sets['info']['unpractised'])} unpractised ones):",
                    {str(k): U.measures(eps)["imp"] for k, eps in sorted(r["reads"].items(), key=lambda kv: int(kv[0]))})
    failed = [j["label"] for j in dose_jobs() if not (d / j["label"] / "info.json").exists()]
    if failed:
        raise SystemExit(f"the dose check is not complete: {failed} failed; the same command launches them once more")
    rep = dose_record(d, sets)
    again = [t for t in tags_on(out, root, "dose.json") if t != tag]
    if again:  # checked once more at the last moment: two checks on the same sets must not both leave a record
        raise SystemExit(f"the dose check is NOT recorded: while it ran, another dose record appeared for the sets "
                         f"under root {root!r} ({again}). One dose check per set of sets; nothing is written under "
                         f"tag {tag!r}")
    (d / "dose.json").write_text(json.dumps({"utc": utc(), "tag": tag, "root": root, "freeze": json.loads(ffile.read_text()),
                                             **rep}, indent=1) + "\n")
    import compare_u1 as K

    K.show_dose(rep)
    return 0 if rep["ok"] else 1


def find_dose(out: Path, dose_tag: str | None) -> dict:
    """The dose record a block runs under: the one for the sets as they are now (the same root), named or not. It
    must have passed."""
    root = read_sets(out)["root"]
    tags = tags_on(out, root, "dose.json")
    if len(tags) > 1:  # whatever --dose-tag says: one dose check per set of sets, so none of them is run under
        raise SystemExit(f"refused: more than one dose record for the sets under root {root!r} ({tags}). One dose "
                         "check per set of sets: no block runs under either")
    found = [t for t in tags if dose_tag is None or t == dose_tag]
    if not found:
        raise SystemExit("refused: no dose record" + (f" under tag {dose_tag!r}" if dose_tag else "")
                         + f" for the sets under root {root!r} ({tags})")
    rec = json.loads((out / found[0] / "dose.json").read_text())
    if not rec.get("ok") or not rec.get("read_steps"):
        raise SystemExit(f"refused: the dose check under tag {rec.get('tag')!r} did not pass, or names no read steps "
                         f"(checks {rec.get('checks', {}).get('ok')}, rule branch {rec.get('rule', {}).get('branch')}): "
                         "no block starts under it")
    return rec


# ------------------------------------------------------------------------------------------------ --block


def spend_stop(out: Path) -> tuple[float, str | None]:
    """(the stop a block is checked against, why it is not the design's). It is the design's, u1_core.SPEND_STOP,
    unless stop.json in the results folder raises it: {"usd": the new stop, "why": the ledger entry that records
    Eugene's approval of it}. The file is not code, so a raise after the freeze changes no frozen file. A file that
    does not raise the stop, or gives no reason, is refused: the stop is never lowered or moved in silence."""
    f = out / "stop.json"
    if not f.is_file():
        return U.SPEND_STOP, None
    rec = json.loads(f.read_text())
    usd, why = rec.get("usd"), rec.get("why")
    if isinstance(usd, bool) or not isinstance(usd, (int, float)) or not usd > U.SPEND_STOP or not (isinstance(why, str) and why.strip()):
        raise SystemExit(f"refused: {f} must raise the stop above ${U.SPEND_STOP:.2f} and say why (the ledger entry "
                         "that records the approval)")
    return float(usd), why.strip()


def how_settled(res: dict) -> str:
    """A reserve that is not settled, and what settles it."""
    la = res.get("launch") or {}
    if la.get("tag"):
        return f"{res['stage']} (settle it with --fetch {la['tag']})"
    if la:
        return f"{res['stage']} (--fetch TAG settles it once its container's time has passed)"
    return f"{res['stage']} (gen_docs.py trues it up when it next runs)"


def arms_of(run: dict) -> tuple:
    """A main run's arms, from its record (run.json), in u1_core.ARMS' order. A record that names none was written
    before revision 3, when a run was always all three."""
    if not run.get("arms"):
        return U.ARMS
    try:
        return U.run_arms(run["arms"])
    except ValueError as e:
        raise SystemExit(f"refused: the run's record names arms that are not a run's: {e}") from None


def read_steps_of(run: dict, tag: str) -> list[int]:
    """A main run's two later read steps, from its record. A record that names none was written before revision 3
    (by rounds): it is refused, since no run of that kind exists to be gone on with."""
    if not run.get("read_steps"):
        raise SystemExit(f"refused: tag {tag!r}'s run.json names no read steps (a record from before revision 3). A "
                         "main run takes a tag of its own, on sets with a dose record that passed")
    return sorted(int(s) for s in run["read_steps"])


def raise_on_record(out: Path, what: str) -> tuple[float, str]:
    """Arm R joins a run only when the budget has been raised (revision 3): spend_stop must report a raise, which is
    a stop.json that raises the stop and says why. Returns (the raised stop, its reason). Without one it refuses;
    `what` says what is refused, and the refusal says what records a raise."""
    stop, raised = spend_stop(out)
    if raised is None:
        raise SystemExit(f"refused: {what}. Arm R joins a run only when the budget has been raised, and no raise is on "
                         f"record. A raise is a file stop.json in the results folder ({out / 'stop.json'}) that holds "
                         f'{{"usd": the new stop, above ${U.SPEND_STOP:.2f}, "why": the ledger entry that records the '
                         'approval}')
    return stop, raised


def owed_usd(d: Path, arms) -> float:
    """What a main run is expected still to cost at these arms: for every arm of every seed of every block, each
    stage that has not finished, at its expected cost (u1_core.EXPECTED_USD). A seed recorded as dropped is counted
    like any other, so the figure errs high."""
    return sum(U.EXPECTED_USD[st] for k in range(1, len(U.BLOCKS) + 1) for s in U.block_seeds(k) for a in arms
               for st in STAGES if not stage_done(d, st, U.label(a, s)))


def whole_run_stop(out: Path, d: Path, arms, what: str) -> dict:
    """The whole-run check (revision 3), made at a tag's first launch and when an arm is added: step 3's spend so
    far plus what the run is expected still to cost at these arms (owed_usd) must be at most the stop (spend_stop: a
    recorded raise counts). It is in addition to the stop between blocks (block_stop), which stays. A run that cannot
    finish under the stop does not start, and an arm that cannot be finished is not added; `what` says which. Returns
    the numbers, for the run's record."""
    so_far, owed = spent(out), owed_usd(d, arms)
    stop, raised = spend_stop(out)
    if so_far + owed > stop + U.EPS:
        raise SystemExit(f"refused: {what}. Step 3 has spent ${so_far:.2f}, and the run as a whole (arms "
                         f"{','.join(arms)}: every seed of every block, each stage not yet finished) is expected to "
                         f"cost ${owed:.2f} more. That would pass the stop ${stop:.2f}")
    return {"arms": list(arms), "spent_before": round(so_far, 2), "owed": round(owed, 2), "stop": stop,
            **({"stop_raised": raised} if raised else {})}


def block_stop(out: Path, d: Path, run: dict, block: int) -> None:
    """What decides whether a block starts (a block that has started goes on: the stop is between blocks). In order:
    a run that ended stays ended; blocks start in order; blocks 2 and 3 wait for the look after block one; no reserve
    may be unsettled, so the stop is never computed on a booked reserve; at a tag's first launch the run as a whole
    must fit under the stop (whole_run_stop); then the design's stop itself: step 3's spend so far plus the block's
    expected cost, for the run's arms, must be at most the stop (SPEND_STOP, unless stop.json raises it: spend_stop).
    When the stop refuses a block of a run that has started, the run's record says it ended there, so that it can
    still be reported (compare_u1.py --ended)."""
    if str(block) in run["blocks"]:
        return
    if run.get("ended"):
        raise SystemExit(f"refused: this run ended ({run['ended'].get('why')}). No further block starts; its runs are "
                         f"reported with compare_u1.py --ended {d.name}")
    if block > 1 and str(block - 1) not in run["blocks"]:
        raise SystemExit(f"refused: blocks start in order: block {block} comes after block {block - 1}, which has not "
                         "started")
    if block > 1:
        look = d / "look.json"
        if not look.is_file() or json.loads(look.read_text()).get("stop") is not False:
            raise SystemExit(f"refused: block {block} starts only after the look at block one says go "
                             f"(compare_u1.py --look {d.name})")
    open_reserves = unsettled(out)
    if open_reserves:
        raise SystemExit("refused: a reserve is booked and not settled, and the stop is not computed on a booked "
                         f"reserve: {'; '.join(how_settled(x) for x in open_reserves)}")
    if not run["blocks"]:  # the tag's first launch: the run as a whole must fit, or nothing of it starts
        run["whole_run"] = whole_run_stop(out, d, arms_of(run), "a run that cannot finish under the stop does not start")
    so_far, expected = spent(out), U.block_expected_usd(block, arms_of(run))
    stop, raised = spend_stop(out)
    if so_far + expected > stop + U.EPS:
        why = (f"step 3 has spent ${so_far:.2f}; with block {block}'s expected ${expected:.2f} it would pass the stop "
               f"${stop:.2f}")
        if (d / "run.json").is_file():
            run["ended"] = {"utc": utc(), "why": why, "block": block}
            (d / "run.json").write_text(json.dumps(run, indent=1) + "\n")
        raise SystemExit(f"refused: {why}")
    run["blocks"][str(block)] = {"started": utc(), "spent_before": round(so_far, 2), "expected": round(expected, 2),
                                 "stop": stop, **({"stop_raised": raised} if raised else {})}


def stage_kwargs(stage: str, root: str, tag: str, arm: str, seed: int, *, attempt: int = 1, epochs: int | None = None,
                 read_steps=None, steps: int | None = None, rows=None, eval_names=None) -> dict:
    """What one run of a stage is called with. docs and practice: modal_sft.finetune (the design's settings, `epochs`
    overriding them for the rehearsal), from the arm's document set, then from the practice set and the docs adapter.
    press: modal_act1.press from the practice adapter, on the main stream, trained to the last read step (or
    `steps`) and read at step 0 and at `read_steps`, on the pairs given. `attempt` picks the press run's folder on
    the volume (press never overwrites)."""
    base, lab = f"{RUNS_ACT1}/{root}", U.label(arm, seed)
    ft = {**U.DOC_FT, **({} if epochs is None else {"epochs": epochs})}
    if stage == "docs":
        return dict(label=arm, seed=seed, **ft, src_dir=f"{base}/sets/{arm}docs", out_dir=f"{base}/{tag}/docs/{lab}")
    if stage == "practice":
        return dict(label=arm, seed=seed, **ft, src_dir=f"{base}/sets/practice",
                    out_dir=f"{base}/{tag}/practice/{lab}", init=f"{base}/{tag}/docs/{lab}/adapter")
    rs = [int(s) for s in read_steps]
    return dict(init=f"{base}/{tag}/practice/{lab}/adapter", label=volume_label(lab, attempt), tag=tag, seed=seed,
                stream=U.MAIN_STREAM, steps=max(rs) if steps is None else int(steps), read_steps=rs, cfg=U.PRESS_CFG,
                pairs_rows=rows, eval_names=list(eval_names), model_key=U.MODEL_KEY, root=root)


def parse_arms(text: str | None) -> tuple | None:
    """--arms N,E as a run's arms (u1_core.run_arms: N and E, or N, R and E, in ARMS' order); None when the flag was
    not given."""
    if text is None:
        return None
    try:
        return U.run_arms([x.strip().upper() for x in text.split(",") if x.strip()])
    except ValueError as e:
        raise SystemExit(f"refused: --arms {text}: {e}") from None


def cmd_block(block: int, tag: str, stage: str, upper_h: float, seeds: list[int] | None, dose_tag: str | None,
              arms: tuple | None = None) -> int:
    out = out_dir()
    d = out / tag
    if (d / "dose.json").exists() or (d / "freeze.json").exists():
        raise SystemExit(f"refused: {tag!r} is a dose check's tag")
    if is_plumb(d):
        raise SystemExit(f"refused: {tag!r} is a rehearsal's tag; the main run takes a tag of its own")
    try:
        block_all = U.block_seeds(block)
    except ValueError as e:
        raise SystemExit(f"refused: {e}") from None
    if seeds is not None and not set(seeds) <= set(block_all):
        raise SystemExit(f"refused: --seeds {seeds} are not all in block {block} ({list(block_all)})")
    sets = read_sets(out)
    root = sets["root"]
    rfile = d / "run.json"
    run = json.loads(rfile.read_text()) if rfile.is_file() else None
    if run is None:  # a new main tag: an empty one, and one per set of sets
        if d.is_dir() and any(d.iterdir()):
            raise SystemExit(f"refused: {d} already exists and is not empty, and holds no main run. A main run takes "
                             "a tag of its own")
        others = [t for t in tags_on(out, root, "run.json") if t != tag]
        if others:
            raise SystemExit(f"refused: the sets under root {root!r} already have a main run ({others}). One main tag "
                             "per set of sets")
        if arms is not None and "R" in arms:  # the rule arm joins a run only when the budget has been raised
            raise_on_record(out, f"tag {tag!r} does not start with arm R (--arms {','.join(arms)})")
    elif run.get("root") != root:
        raise SystemExit(f"refused: tag {tag!r} runs on the sets under root {run.get('root')!r}; the sets are now "
                         f"under root {root!r}. A run does not move to rebuilt sets")
    elif arms is not None and arms != arms_of(run):  # the arms are chosen once, at the tag's first launch
        raise SystemExit(f"refused: tag {tag!r} runs arms {','.join(arms_of(run))} (its run.json), not "
                         f"{','.join(arms)}. Leave --arms out; an arm is added with --add-arm ARM {tag}")
    dose = find_dose(out, dose_tag if run is None or dose_tag else run["dose_tag"])
    if run is not None and run["dose_tag"] != dose["tag"]:
        raise SystemExit(f"refused: tag {tag!r} runs under the dose record {run['dose_tag']!r}, not {dose['tag']!r}")
    check_frozen(out, dose["freeze"])
    if run is None:
        run = {"tag": tag, "dose_tag": dose["tag"], "read_steps": dose["read_steps"],
               "arms": list(arms or U.MAIN_ARMS), "root": root, "freeze_commit": dose["freeze"]["commit"], "blocks": {}}
    read_steps, arms = read_steps_of(run, tag), arms_of(run)
    dropped = {int(s) for s in read_dropped(d)}
    live = [s for s in (seeds if seeds is not None else block_all) if s not in dropped]
    before = STAGES[STAGES.index(stage) - 1] if stage != STAGES[0] else None
    if before:  # all the run's arms of a seed go through a stage together
        missing = [U.label(a, s) for s in live for a in arms if not stage_done(d, before, U.label(a, s))]
        if missing:
            raise SystemExit(f"refused: the {before} stage has not finished for {missing}")
    todo = [(a, s) for s in live for a in arms if not stage_done(d, stage, U.label(a, s))]
    stuck = [U.label(a, s) for a, s in todo if tries(stage_dir(d, stage, U.label(a, s))) >= MAX_TRIES]
    if stuck:
        raise SystemExit(f"refused: {stuck} failed twice and are not recorded as dropped; see {d / 'dropped.json'}")
    if not todo:
        raise SystemExit(f"refused: nothing to launch: the {stage} stage of block {block} is done for seeds {live} "
                         f"(dropped: {sorted(dropped)})")
    kind = "press" if stage == "press" else "finetune"
    rows, names = None, None
    if stage == "press":
        split = eval_split()
        names = split["oneoff"]
        rows = [r for r in split["rows"] if r["name"] in set(names)]
    guard(out, len(todo) * run_upper(kind, upper_h),  # before the block is recorded as started
          "; launch fewer seeds at a time (--seeds), lower --upper-h, or wait for the next UTC day")
    block_stop(out, d, run, block)
    d.mkdir(parents=True, exist_ok=True)
    rfile.write_text(json.dumps(run, indent=1) + "\n")
    jobs = [{"label": U.label(a, s), "kwargs": stage_kwargs(
        stage, root, tag, a, s, attempt=tries(stage_dir(d, stage, U.label(a, s))) + 1, read_steps=read_steps,
        rows=rows, eval_names=names)} for a, s in todo]
    variant(out, {"cmd": "block", "block": block, "tag": tag, "stage": stage, "runs": [j["label"] for j in jobs],
                  "dose_tag": dose["tag"], "read_steps": read_steps, "arms": list(arms), "root": root,
                  "upper_h": upper_h, "settings": U.PRESS_CFG if stage == "press" else U.DOC_FT})
    res = launch(out, f"{tag}-b{block}-{stage}", kind, jobs, upper_h, tag=tag, where=stage, block=block)
    failed = []
    for (a, s), job in zip(todo, jobs):
        lab, r, sd = job["label"], res[job["label"]], stage_dir(d, stage, job["label"])
        if "error" in r:
            record_error(sd, r["error"], volume=job["kwargs"].get("out_dir") or job["kwargs"]["label"])
            failed.append(lab)
            if tries(sd) >= MAX_TRIES:
                drop_seed(d, s, f"{lab} failed twice at the {stage} stage")
        elif stage == "press":
            save_press(sd, r, volume_label=job["kwargs"]["label"])
            # no outcome is printed here: the one look reads arm N alone, and no contrast before every block is in
            log(lab, "press done:", {"seconds": r.get("seconds"), "reads": sorted(int(k) for k in r.get("reads", {}))})
        else:
            save_finetune(sd, r)
            log(lab, stage, json.loads(r["files"]["done.json"]))
    if failed:
        log(f"failed: {failed}; the same command launches a failed run once more (identical inputs)")
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------ --add-arm


def cmd_add_arm(arm: str, tag: str) -> int:
    """Add an arm to a main run that has started (revision 3: the rule arm, only if the budget is raised, and only
    before any readout). A run always has N and E, so the arm is R. It changes run.json (and leaves a line in
    variants.jsonl) and launches nothing: the ordinary --block commands then find the arm's runs missing in the
    blocks that started, and launch them stage by stage. Refused when the arm is already in the run; when a readout
    exists (no contrast is computed until every planned arm of every seed is in, so an arm is never added to a run
    that has been read); when the run has ended; when no raise of the stop is on record (raise_on_record); while a
    reserve is not settled (the stop is not computed on a booked reserve); when step 3's spend so far plus the arm's
    expected cost, for every seed of the blocks already started, would pass the stop; and when the run as a whole,
    with the arm in it, could not finish under the stop (whole_run_stop). Blocks that start later are checked
    against the stop with the arm in them (block_stop)."""
    out = out_dir()
    d = out / tag
    rfile = d / "run.json"
    arm = arm.strip().upper()
    if arm not in U.ARMS:
        raise SystemExit(f"refused: {arm!r} is not an arm ({', '.join(U.ARMS)})")
    if not rfile.is_file():
        raise SystemExit(f"refused: {tag!r} holds no main run (no run.json). An arm is added to a run that has "
                         f"started; a new run names its arms at its first launch (--block 1 {tag} --stage docs --arms ...)")
    run = json.loads(rfile.read_text())
    arms = arms_of(run)
    read_steps_of(run, tag)  # a record from before revision 3 is refused
    if arm in arms:
        raise SystemExit(f"refused: arm {arm} is already in the run under tag {tag!r} (arms {','.join(arms)})")
    readouts = [n for n in READOUTS if (d / n).exists()]
    if readouts:
        raise SystemExit(f"refused: the run under tag {tag!r} has been read ({readouts}). An arm is added only before "
                         "any readout")
    if run.get("ended"):
        raise SystemExit(f"refused: this run ended ({run['ended'].get('why')}). No arm is added to it")
    look = d / "look.json"
    if look.is_file() and json.loads(look.read_text()).get("stop") is True:
        raise SystemExit("refused: this run ended (the look after block one said stop). No arm is added to it")
    if arm == "R":  # the rule arm joins a run only when the budget has been raised
        raise_on_record(out, f"arm R is not added to the run under tag {tag!r}")
    open_reserves = unsettled(out)
    if open_reserves:
        raise SystemExit("refused: a reserve is booked and not settled, and the stop is not computed on a booked "
                         f"reserve: {'; '.join(how_settled(x) for x in open_reserves)}")
    started = sorted(int(k) for k in run.get("blocks") or {})
    seeds = [s for k in started for s in U.block_seeds(k)]
    so_far, expected = spent(out), U.arm_expected_usd(len(seeds))
    stop, raised = spend_stop(out)
    if so_far + expected > stop + U.EPS:
        raise SystemExit(f"refused: step 3 has spent ${so_far:.2f}; with arm {arm}'s expected ${expected:.2f} for the "
                         f"{len(seeds)} seeds of the blocks already started ({started}) it would pass the stop "
                         f"${stop:.2f}. The stop is raised by a stop.json that names the ledger entry of the approval")
    new = U.run_arms([*arms, arm])
    # It fits for the blocks that started. The run as a whole must fit too, with the arm in every block still to come.
    whole = whole_run_stop(out, d, new, f"arm {arm} is not added to the run under tag {tag!r}: its runs for the blocks "
                                        f"already started ({started}) would fit under the stop, but an arm that cannot "
                                        "be finished is not added")
    run["arms"] = list(new)
    added = {"utc": utc(), "arm": arm, "blocks": started, "seeds": seeds, "spent_before": round(so_far, 2),
             "expected": round(expected, 2), "owed_run": whole["owed"], "stop": stop,
             **({"stop_raised": raised} if raised else {})}
    run.setdefault("arms_added", []).append(added)
    rfile.write_text(json.dumps(run, indent=1) + "\n")
    variant(out, {"cmd": "add-arm", "tag": tag, "arms": list(new), **{k: v for k, v in added.items() if k != "utc"}})
    log(f"arm {arm} is added to the run under tag {tag!r}: its arms are now {','.join(new)}. Its runs for seeds {seeds} "
        f"are expected to cost ${expected:.2f}, and the run as a whole ${whole['owed']:.2f} more (step 3 has spent "
        f"${so_far:.2f}; the stop is ${stop:.2f}). Launch them with the ordinary commands, stage by stage: --block K "
        f"{tag} --stage docs, then practice, then press")
    return 0


# ------------------------------------------------------------------------------------------------ --plumb


def plumb_checks(d: Path, lab: str, names: list[str]) -> dict:
    """What the rehearsal must show: both fine-tunes finished; the press run loaded an adapter that is not all
    zeros; its own record says it trained the rehearsal's steps and read at the rehearsal's steps (a round cut
    short, a read inside it and a read at the end of training); its reads at step 0 and at each rehearsal step hold
    the rehearsal's pairs with the design's samples of each twin; and the held-out loss is a finite number for every
    kind. No outcome is counted."""
    import compare_u1 as K

    run = K.load_run(d / lab) or {"info": {}, "reads": {}}
    info = run["info"]
    own = info.get("info") or {}  # the container's own record of the run
    checks = {"docs_done": stage_done(d, "docs", lab), "practice_done": stage_done(d, "practice", lab),
              "init_lora_b_abs": (own.get("init_lora_b_abs") or 0) > 0,
              "steps": own.get("steps") == PLUMB["steps"] and own.get("read_steps") == U.read_points(PLUMB["read_steps"])}
    for s in U.read_points(PLUMB["read_steps"]):
        eps = run["reads"].get(s)
        checks[f"read_s{s}"] = eps is not None and not U.read_problems(eps, names, U.READ["n_impossible"],
                                                                       U.READ["n_solvable"])
    held = info.get("held_nll") or {}
    checks["held_nll"] = all(isinstance(x := (held.get(a) or {}).get("nll_per_token"), (int, float)) and math.isfinite(x)
                             for a in U.ARMS)
    return checks


def cmd_plumb(tag: str, upper_h: float | None) -> int:
    """The rehearsal: one run (PLUMB) through documents, practice and press on the real sets, each stage through
    launch(). A finished stage is skipped, so the command goes on from where it stopped."""
    out = out_dir()
    d = out / tag
    held = [n for n in ("run.json", "dose.json", "freeze.json") if (d / n).exists()]
    if held:
        raise SystemExit(f"refused: {tag!r} holds {held}: the rehearsal takes a tag of its own")
    sets = read_sets(out)
    root = sets["root"]
    split = eval_split()
    names = sorted(split["oneoff"])[:PLUMB["pairs"]]
    rows = [r for r in split["rows"] if r["name"] in set(names)]
    arm, seed = PLUMB["arm"], PLUMB["seed"]
    lab = U.label(arm, seed)
    pfile = d / "plumb.json"
    before = json.loads(pfile.read_text()) if pfile.exists() else {}
    if before.get("root") not in (None, root):
        raise SystemExit(f"refused: the rehearsal {tag!r} ran on the sets under root {before['root']!r}; the sets are "
                         f"now under root {root!r}. Take a new tag")
    code = code_freeze(out)["files"]  # a rehearsal is of one state of the code: the dose check needs it unchanged
    if before.get("code") not in (None, code):
        raise SystemExit(f"refused: the rehearsal {tag!r} was started under other code than the code as it is now. "
                         "Take a new tag")
    started = before.get("started") or utc()
    d.mkdir(parents=True, exist_ok=True)
    if not pfile.exists():
        pfile.write_text(json.dumps({"started": started, "tag": tag, "root": root, "ok": None, "code": code}, indent=1) + "\n")
    todo = [st for st in STAGES if not stage_done(d, st, lab)]
    hours = {st: upper_h or PLUMB_UPPER_H[st] for st in todo}
    variant(out, {"cmd": "plumb", "tag": tag, "root": root, "run": lab, "stages": todo, "plan": PLUMB,
                  "pairs": len(names), "upper_h": hours})
    for stage in todo:
        sd = stage_dir(d, stage, lab)
        kw = stage_kwargs(stage, root, tag, arm, seed, attempt=tries(sd) + 1, epochs=PLUMB["epochs"],
                          read_steps=PLUMB["read_steps"], steps=PLUMB["steps"], rows=rows, eval_names=names)
        kind = "press" if stage == "press" else "finetune"
        res = launch(out, f"plumb-{tag}-{stage}", kind, [{"label": lab, "kwargs": kw}], hours[stage], tag=tag,
                     where=stage)[lab]
        if "error" in res:
            record_error(sd, res["error"], volume=kw.get("out_dir") or kw["label"])
            raise SystemExit(f"the rehearsal's {stage} stage failed ({res['error'][:200]}); the same command goes on "
                             "from this stage")
        if stage == "press":
            save_press(sd, res, volume_label=kw["label"])
            log(lab, "press done:", {"seconds": res.get("seconds")})  # seconds only: no outcome is printed
        else:
            save_finetune(sd, res)
            log(lab, stage, "done:", {"seconds": json.loads(res["files"]["done.json"]).get("seconds")})
    checks = plumb_checks(d, lab, names)
    ok = all(checks.values())
    usd = round(sum(x["usd"] for x in spend_stages(out) if x["stage"].startswith(f"plumb-{tag}-")), 4)
    pfile.write_text(json.dumps({"started": started, "utc": utc(), "tag": tag, "root": root, "run": lab, "ok": ok,
                                 "checks": checks, "pairs": len(names), "plan": PLUMB, "usd": usd, "code": code},
                                indent=1) + "\n")
    print(f"the rehearsal ({lab}: {PLUMB['epochs']} epoch a fine-tune, {PLUMB['steps']} steps of phase 3 read at steps "
          f"{U.read_points(PLUMB['read_steps'])}, {len(names)} read pairs): {'ok' if ok else 'NOT OK'}; it cost ${usd:.2f}")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAILED'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ --fetch


def settle_interrupted(out: Path, d: Path, tag: str) -> None:
    """Settle each reserve an interrupted launch of this tag left booked. A run is finished if its local result now
    exists. While a run is unfinished and its container's time (the reserve's hours, the queue slack and a grace:
    settle_after) has not passed, nothing is settled: fetch again later. Otherwise the true-up is booked on the reserve's own day: a finished
    press or fine-tune run at its recorded seconds, a finished old read and every unfinished run at the whole
    reserve. An unfinished run is recorded as a failure; in a block stage a second failure drops its seed."""
    for res in unsettled(out):
        la = res.get("launch") or {}
        if la.get("tag") != tag or not la.get("where"):
            continue
        kind, upper_h, where, labels = la["kind"], float(la["upper_h"]), la["where"], list(la["runs"])
        not_done = [lab for lab in labels if not result_file(d, where, lab).is_file()]
        free = settle_after(res["utc"], upper_h)
        if not_done and time.time() < free:
            log(f"{res['stage']}: {not_done} have no result yet and their containers may still be running. Nothing is "
                f"settled; fetch again after {time.strftime('%H:%M', time.gmtime(free))} UTC")
            continue
        secs, usd = {}, 0.0
        for lab in labels:
            s = None
            if lab not in not_done and kind in ("press", "finetune"):
                rec = json.loads(result_file(d, where, lab).read_text())
                s = rec.get("seconds") or (rec.get("info") or {}).get("seconds")
            if s:  # a finished press or fine-tune run, at its recorded seconds
                secs[lab] = float(s)
                usd += run_usd(kind, float(s))
            else:  # a finished old read, and every unfinished run: the whole reserve
                secs[lab] = upper_h * 3600
                usd += run_upper(kind, upper_h)
        book(out, res["stage"], usd - res["usd"], len(labels), res["day"],
             {"kind": "true-up", "late": True, "metered_usd": round(usd, 4), "reserved_usd": round(res["usd"], 4),
              "container_s": {k: round(v, 1) for k, v in secs.items()}, "runs": labels, "unfinished": not_done})
        for lab in not_done:
            sd = d / lab if where == "old" else stage_dir(d, where, lab)
            record_error(sd, "did not finish before its container's timeout; the launch was interrupted")
            if la.get("block") is not None and tries(sd) >= MAX_TRIES:
                drop_seed(d, int(lab.rsplit("_s", 1)[1]), f"{lab} failed twice at the {where} stage")


def volume_file(rel: str) -> bytes | None:
    """One file of the runs volume, or None when it is not there (the container had not written it: the volume
    raises FileNotFoundError). Any other failure means the volume could not be read, and is raised."""
    try:
        return fetch_volume(rel)
    except FileNotFoundError:
        return None


def copy_from_volume(d: Path, tag: str, root: str) -> None:
    """Copy a tag's files from the volume for every run with no local result (cmd_fetch's first half)."""
    is_dose = (d / "freeze.json").exists()
    if is_dose:
        labels = [j["label"] for j in dose_jobs()]
    elif is_plumb(d):
        labels = [U.label(PLUMB["arm"], PLUMB["seed"])]
    else:  # a main run: the arms its record names (N and E by default; a tag with no record yet has launched nothing)
        rfile = d / "run.json"
        arms = arms_of(json.loads(rfile.read_text())) if rfile.is_file() else U.MAIN_ARMS
        labels = [U.label(a, s) for s in U.SEEDS for a in arms]
    for ad, old_dir in (old_dirs(d).items() if is_dose else ()):  # each old read's two files, under /runs/h4lite
        if (old_dir / "heldout.jsonl").exists():
            continue
        rel = "h4lite/" + old_rel(tag, ad, old_dir)  # its _try<k> folder after an earlier failure
        got = {name: volume_file(f"{rel}/{name}") for name in ("heldout.json", "heldout.jsonl")}
        if None in got.values():
            log(old_dir.name, "not on the volume")
        else:
            old_dir.mkdir(parents=True, exist_ok=True)
            for name, raw in got.items():
                (old_dir / name).write_bytes(raw)
            log(old_dir.name, "fetched", sorted(got))
    for lab in labels:
        for stage in (("press",) if is_dose else STAGES):
            sd = stage_dir(d, stage, lab)
            if stage_done(d, stage, lab):
                continue
            vol = volume_label(lab, tries(sd) + 1) if stage == "press" else lab
            rel = f"act1/{root}/{tag}/{vol}" if stage == "press" else f"act1/{root}/{tag}/{stage}/{lab}"
            names = ["info.json", "log.jsonl"] if stage == "press" else ["args.json", "log.jsonl", "done.json"]
            got = {}
            for name in names:
                raw = volume_file(f"{rel}/{name}")
                if raw is not None:
                    got[name] = raw
                elif name in ("info.json", "done.json"):  # the file that says the run finished is not there
                    log(lab, stage, "not finished on the volume")
                    got = {}
                    break
            if got and stage == "press":  # its reads: one file for each step the container's own record names
                read_steps = json.loads(got["info.json"]).get("read_steps")
                reads = {f"read_s{int(s)}.jsonl": volume_file(f"{rel}/read_s{int(s)}.jsonl") for s in read_steps or ()}
                if not reads or None in reads.values():  # a finished run holds every read its record names
                    log(lab, stage, "its record on the volume names no reads, or reads that are not there; nothing "
                        "is copied")
                    got = {}
                else:
                    got |= reads
            if not got:
                continue
            sd.mkdir(parents=True, exist_ok=True)
            for name, raw in got.items():
                if name == "info.json":  # the container's info, in the launcher's form
                    raw = (json.dumps({"label": vol, "info": json.loads(raw), "volume_label": vol, "fetched": True,
                                       "held_nll": json.loads(raw).get("held_nll")}, indent=1) + "\n").encode()
                (sd / name).write_bytes(raw)
            log(lab, stage, "fetched", sorted(got))


def cmd_fetch(tag: str) -> int:
    """Copy a tag's files from the volume (read-only there) for every run with no local result, then settle what an
    interrupted launch left booked: this tag's launches (settle_interrupted), and a sets build, which has no tag of
    its own. The read files hold model text: written to the results folder as data, not printed. No container is
    started. If the volume cannot be read (anything but a missing file), nothing is settled in that call: a run
    whose result could not be looked for must not be taken for unfinished."""
    out = out_dir()
    d = out / tag
    root = tag_root(out, d)  # the sets this tag runs on, also after the sets were rebuilt under another root
    try:
        copy_from_volume(d, tag, root)
    except Exception as e:  # noqa: BLE001  (not a missing file: the volume could not be read)
        log(f"the volume could not be read ({type(e).__name__}: {str(e)[:200]}). Nothing is settled in this call; "
            "fetch again")
        return 1
    settle_interrupted(out, d, tag)
    settle_untagged(out, "sets", strict=False)
    return 0


# ------------------------------------------------------------------------------------------------ --dry


def plan(arms=U.MAIN_ARMS) -> dict:
    """The run as planned, for a run of these arms (N and E since revision 3): per block and stage, the runs, what
    they are expected to cost and what a launch reserves; the reserves of the sets, the rehearsal and the dose
    check; and what one more arm would be expected to add for all eight seeds."""
    arms = U.run_arms(arms)
    blocks = {}
    for k in range(1, len(U.BLOCKS) + 1):
        n = len(U.block_seeds(k)) * len(arms)
        blocks[k] = {"seeds": list(U.block_seeds(k)), "runs": n, "expected_usd": round(U.block_expected_usd(k, arms), 2),
                     "stages": {st: {"expected_usd": round(n * U.EXPECTED_USD[st], 2),
                                     "reserve_usd": round(n * run_upper("press" if st == "press" else "finetune",
                                                                        UPPER_H[st]), 2)} for st in STAGES}}
    dose = (len(dose_jobs()) * run_upper("press", UPPER_H["dose"])
            + len(U.DOSE_ADAPTERS) * run_upper("old", UPPER_H["old"]))  # two press runs and two old-path reads
    plumb = sum(run_upper("press" if st == "press" else "finetune", PLUMB_UPPER_H[st]) for st in STAGES)
    return {"arms": list(arms), "seeds": list(U.SEEDS), "blocks": blocks, "dose_reserve_usd": round(dose, 2),
            "plumb_reserve_usd": round(plumb, 2),
            "sets_reserve_usd": round(run_upper("sets", UPPER_H["sets"]), 2), "stop_usd": U.SPEND_STOP,
            "expected_blocks_usd": round(sum(b["expected_usd"] for b in blocks.values()), 2),
            "run_usd": round(sum(U.EXPECTED_USD.values()), 2),
            "arm_usd": round(U.arm_expected_usd(len(U.SEEDS)), 2)}


def cmd_dry() -> int:
    pl = plan()
    print(f"step 3, understanding first (revision 3): arms {pl['arms']} of {list(U.ARMS)}, seeds {pl['seeds']}, model "
          f"{U.MODEL_KEY}")
    print(f"  phases 1 and 2: modal_sft.finetune {U.DOC_FT}; phase 3 a round: {U.PRESS_CFG}")
    print(f"  the read: {U.READ} (a read's seed is seed_base plus its step)")
    print(f"sets: reserve ${pl['sets_reserve_usd']:.2f} (CPU). rehearsal (--plumb): reserve "
          f"${pl['plumb_reserve_usd']:.2f} ({PLUMB_UPPER_H} h; arm {PLUMB['arm']}, seed {PLUMB['seed']}, "
          f"{PLUMB['epochs']} epoch a fine-tune, {PLUMB['steps']} steps of phase 3 read at steps "
          f"{U.read_points(PLUMB['read_steps'])}, {PLUMB['pairs']} read pairs)")
    print(f"dose check: reserve ${pl['dose_reserve_usd']:.2f} ({len(dose_jobs())} press runs of {U.DOSE_STEPS[-1]} "
          f"steps on the {U.MAIN_STREAM} stream, reads at steps {list(U.DOSE_STEPS)}, {UPPER_H['dose']} h a run; and "
          f"{len(U.DOSE_ADAPTERS)} old-path reads, {UPPER_H['old']} h a read)")
    for k, b in pl["blocks"].items():
        print(f"block {k}: seeds {b['seeds']}, {b['runs']} runs a stage, expected ${b['expected_usd']:.2f}")
        for st, c in b["stages"].items():
            print(f"  {st:<8} expected ${c['expected_usd']:.2f}, a launch reserves ${c['reserve_usd']:.2f} "
                  f"({UPPER_H[st]} h a run)")
    print(f"expected for the three blocks ${pl['expected_blocks_usd']:.2f} (${pl['run_usd']:.2f} a run); the stop is "
          f"${pl['stop_usd']:.2f} unless stop.json raises it, checked for the whole run at its first launch and "
          f"again when each block starts. Arm R (--arms {','.join(U.ARMS)}, or --add-arm R) needs a raise on record and "
          f"is expected to add ${pl['arm_usd']:.2f} for all {len(U.SEEDS)} seeds. A launch waits {QUEUE_SLACK_H} h "
          "beyond a run's hours before it cancels it")
    try:
        out = out_dir()
        print(f"results folder {out}; step 3 has spent ${spent(out):.2f}")
        print(f"  sets built: {(out / 'sets.json').exists()}; dose records: "
              f"{[f.parent.name for f in sorted(out.glob('*/dose.json'))]}; reserves not settled: "
              f"{[x['stage'] for x in unsettled(out)]}")
    except Exception as e:  # noqa: BLE001  (a dry run still prints without a lab.toml)
        print(f"results folder not read ({type(e).__name__})")
    for f in (PILOT_DOCS / "declared.jsonl", PILOT_DOCS / "neutral.jsonl", STAGE_A_EPISODES, TOLD_DIR / "releases.jsonl"):
        print(f"  local input {'found  ' if f.is_file() else 'MISSING'} "
              f"{f.relative_to(REPO) if f.is_relative_to(REPO) else f}")
    if STAGE_A_EPISODES.is_file():
        try:
            split = eval_split()
            recs = [{"pair": e["pair"], "impossible": e["impossible"], "outcome": e["outcome"]}
                    for e in read_jsonl(STAGE_A_EPISODES)]
            for name in U.STREAMS:
                src = U.stream_summary(recs, split["eval"], name)
                draws = U.stream_draws(recs, split["eval"], name)
                hacked = [U.draw_summary(recs, idx)["hacked"] for by in draws.values() for idx in by.values()]
                print(f"  stream {name}: {src['non_eval']} episodes off the eval pairs, {src['paid']} paid "
                      f"{src['paid_by_outcome']}; a draw of {U.DRAW_N} holds {min(hacked)} to {max(hacked)} hacked "
                      f"items (mean {sum(hacked) / len(hacked):.1f})")
        except (SystemExit, OSError, KeyError) as e:  # a dry run still prints without the split's records
            print(f"  the stream was not read: {e}")
    print("Dry run: nothing launched, nothing spent.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="print the plan and its costs; nothing launched")
    ap.add_argument("--sets", action="store_true", help="build the sets on the volume (CPU; cents)")
    ap.add_argument("--root", default=VOLUME_ROOT, help="--sets: the folder under /runs/act1 the sets go to")
    ap.add_argument("--plumb", metavar="TAG", help="the rehearsal before the freeze (a few dollars)")
    ap.add_argument("--dose", metavar="TAG", help="the dose check (two old-path reads and two press runs)")
    ap.add_argument("--block", nargs=2, metavar=("K", "TAG"), help="one stage of block K of the main run")
    ap.add_argument("--stage", choices=STAGES, help="--block: docs, practice or press")
    ap.add_argument("--seeds", default="", help="--block: only these seeds of the block, comma-separated")
    ap.add_argument("--dose-tag", help="--block: the dose record to run under")
    ap.add_argument("--arms", help="--block, a tag's first launch only: the run's arms, comma-separated (default "
                                   f"{','.join(U.MAIN_ARMS)}; or {','.join(U.ARMS)}, which needs a raised stop on record)")
    ap.add_argument("--add-arm", nargs=2, metavar=("ARM", "TAG"),
                    help="add arm R to a main run that has started (needs a raised stop on record); nothing is launched")
    ap.add_argument("--fetch", metavar="TAG", help="copy a tag's files from the volume and settle an interrupted launch")
    ap.add_argument("--upper-h", type=float, help="container hours a launch reserves for each run (--plumb: for "
                                                  "each stage still to run)")
    a = ap.parse_args(argv)
    if sum(bool(x) for x in (a.dry, a.sets, a.plumb, a.dose, a.block, a.add_arm, a.fetch)) != 1:
        ap.error("one of --dry, --sets, --plumb TAG, --dose TAG, --block K TAG, --add-arm ARM TAG, --fetch TAG")
    if a.upper_h is not None and not a.upper_h > 0:
        ap.error("--upper-h is above 0")
    if a.arms is not None and not a.block:
        ap.error("--arms goes with --block (a tag's first launch); an arm is added later with --add-arm ARM TAG")
    _paths()
    if a.dry:
        return cmd_dry()
    if a.sets:
        return cmd_sets(a.root, a.upper_h or UPPER_H["sets"])
    if a.plumb:
        return cmd_plumb(a.plumb, a.upper_h)
    if a.dose:
        return cmd_dose(a.dose, a.upper_h or UPPER_H["dose"], UPPER_H["old"])
    if a.fetch:
        return cmd_fetch(a.fetch)
    if a.add_arm:
        return cmd_add_arm(a.add_arm[0], a.add_arm[1])
    if not a.stage:
        ap.error("--block needs --stage docs, practice or press")
    try:
        block = int(a.block[0])
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()] or None
    except ValueError:
        ap.error("--block K TAG: K is 1, 2 or 3; --seeds are numbers")
    return cmd_block(block, a.block[1], a.stage, a.upper_h or UPPER_H[a.stage], seeds, a.dose_tag,
                     parse_arms(a.arms))


if __name__ == "__main__":
    raise SystemExit(main())
