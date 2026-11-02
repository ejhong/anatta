# Consolidation design (November 2026): steps 0 and 1

Claude Opus 5.5, an AI model, wrote this on 1 Nov 2026 for Eugene. It is steps 0 and 1 of
`lab/plans/consolidation-2026-11.md`.

**Status: revised after Fable's cold check, which said NOT READY
(`lab/plans/consolidation-design-2026-11-fable-check.md`). Nothing has moved.** The first draft and `lab/MAP.md` were
committed at a82ca71, and Opus's finding-2 edit at d1cfdd5. This revision was written by Claude Opus 5.5 at 20:24 UTC
on 1 Nov (read off the clock) and is uncommitted until Opus commits it. Each answer to a finding is marked "(Fable,
finding N)". Findings 1 (a copy off this laptop) and 5 (reading each of the 11 to-git files) are left to Opus and
Eugene.

Every number here can be checked from the files in `/Users/eugene/prj/align/consolidation-2026-11/`, which is outside
git:

| file | what it is |
|---|---|
| `inventory.py` → `inventory-before-draft.json` | Step 0: every file's path, size, sha256, git status and citations. Taken 18:04:35 UTC; took 30.3 s. |
| `make_move_map.py` → `move-map.tsv`, `refs-by-move.tsv`, `tags.tsv` | The move map, every citation of a moved path, and the tags to create. |
| `verify.py`, `guard/child.py` → `verify-before.json` | The verification script, its guarded runner and the "before" self-test (section 8). |
| `scratch-citations.tsv`, `scratch-md-filed.json`, `scratch-md-contained.json` | Which scratchpad files the ledger cites, and which are already filed in the lab (section 10). |
| `rescue.py` → `rescue-manifest.tsv`, `rescue-newer.tsv` | Copies the kept scratchpad files into the archive (dry run by default) and records every copy in the manifest; `--newer` lists the files the draft missed, for review (sections 10 and 11, step 1b). |
| `step3_2_outside_git.sh` | Step 3.2 as one script that stops at the first failure (section 11). Run so far only on a dummy tree in `tmp/`. |
| `inventory-rehearsal.json` → `verify-after-rehearsal.json` | The after-mode rehearsal on the unmoved tree: inventory at 20:18:30 UTC, run at 20:27:23 UTC (section 8). |
| `refs-by-move-rehearsal.tsv` | The citations table rebuilt from the rehearsal inventory with the patched scanner (section 3). |
| `tmp/selftest/` | Tests of the after-mode code that an unmoved tree can't reach, and the step-3.2 script's dummy trees. |
| `*.before-fable.py`, `*.before-fable.tsv`, `move-map.reviewed-1933.tsv` | The scripts and tables as they were before this revision. |

To re-check everything, run this. It takes about 2.5 minutes, and the session scratchpad must still exist:

    cd /Users/eugene/prj/align/consolidation-2026-11
    python3 -I verify.py --mode before

*(Changed by Opus 5.5 on 1 Nov after Fable's cold check, finding 2.)* Don't regenerate move-map.tsv to re-check:
it carries Opus's privacy review (rescue-reviewed.txt). make_move_map.py now refuses to overwrite it without
`--force`; after a forced run, diff it against move-map.before-review.tsv.

This revision regenerated it once with `--force` (to add rows at the end; section 2). Against
move-map.before-review.tsv, row by row: M0962 (smoke_wim, rescue-dir → leave) is still the only change from the
review; the others are this revision's (M0011, M0013-M0016 reasons; M0037 and M0058 become `edit`; M0966-M0972 new).
`tags.tsv` and `refs-by-move.tsv` came out byte-identical.

Numbers I did not compute myself are marked "(not computed here)". Times are UTC and were read off the clock.

## In short

- **The layout.**
  - Two new roots outside git: `results/` and `data/`.
  - One calibration worktree (`anatta-calib`, branch `calib/all`) instead of four.
  - The library pinned where it is.
  - An archive, a one-page map (`lab/MAP.md`) and a generated ledger index.
  - Old paths keep working through 8 new symlinks.
- **The library is pinned by tag, not merged.** Merging would change no code: dev/ministral is already inside lab, and
  lab hasn't touched `src/` since. What it would change is where scripts load from: a live worktree that lab commits
  can change. Pinning needs no script edits.
- **Calibration.** The seven calib/* branches become one private branch, calib/all, and 10 tags mark the graded
  commits. calib/* history never merges toward main, because C7's port of an unlicensed scorer is in it.
- **Tracked results stay where they are.** The ledger, the site and the graded code cite them by path.
- **The "old copies" at the top of `prj/align` aren't copies.** They are symlinks into the lab. Only `inputs/` holds a
  file of its own.
- **The verify self-test:**
  - every check passes or records its baseline;
  - 1,529 tests pass;
  - the CPU re-grades are identical;
  - three suites fail for reasons that predate this design.
- **Two urgent things, before the moves:**
  - the scratchpad rescue is done: 2,479 files copied at 19:33-19:35 UTC, each now in `rescue-manifest.tsv`; 40 files
    made since the draft wait for Opus's review (section 10);
  - the 8 local-only branches are in a bundle, but the bundle sits inside `prj/align`, which Time Machine excludes
    (decision 6).
- **C3 was graded at 18:52 UTC (NOTES line 7880).** So the freeze waits for the running work to finish: scout's
  first sweep, the p1b run and the bench edits (section 11, step 2).
- **After Fable's check:** verify's after mode now has the checks it lacked (shims, the calibration merge, edits,
  copies, the rescue manifest, a per-suite comparison), and it has been rehearsed on the unmoved tree.

## 0. The inventory (step 0)

`inventory.py` reads files and never changes them. It recorded every file in the seven worktrees, the top of
`prj/align`, the results and data folders, and this session's scratchpad. For each file it kept the path, size,
sha256, kind, git status (tracked, untracked or ignored, with the blob id) and the files that cite it.

**Totals:** 17,839 files; 2,531 MB; 62 symlinks; 5,977 folders.

| place | files | MB | kinds |
|---|---:|---:|---|
| anatta (main) | 96 | 13.8 | code 45, writing 17, data 33, logs 1 |
| anatta-calib-c3 | 106 | 1.7 | code 83, writing 19, data 4 |
| anatta-calib-c4 | 108 | 1.5 | code 84, writing 20, data 4 |
| anatta-calib-c7 | 111 | 1.8 | code 84, writing 20, data 7 |
| anatta-calib-k1 | 78 | 0.7 | code 59, writing 19 |
| anatta-dev | 62 | 0.4 | code 45, writing 17 |
| anatta-lab | 2,542 | 1,468.1 | code 227, writing 130, logs 130, data 376, results 1,679 |
| calib-results | 154 | 30.8 | results |
| explore-results | 2,521 | 68.4 | results 2,515, logs 6 |
| explore-data | 1 | 0.02 | data |
| inputs | 1 | 3.1 | one PDF |
| the session scratchpad | 12,059 | 941.0 | everything |

- **Skipped without listing:**
  - the git stores (2, 482 MB);
  - the two venvs (1,043 MB);
  - 166 cache folders (21 MB).
- **The private folders.** The two private folders, the account folder and `lab/experiments/recognition/`, were
  pruned by name before anything inside them was touched. Each is recorded as "a private folder, not inventoried; it
  stays where it is". The two symlinks into them (`prj/align/recognition` and `lab/recognition`) are recorded and were
  never followed.
- **How citations are found.** The inventory finds path-like tokens in every text file and matches them to inventory
  paths, in four strengths:
  - `abs`: an absolute path;
  - `rel`: a relative path in the same place;
  - `rel_other_place`: the same relative path in a sibling worktree;
  - `basename`: a bare file name (a first pass only, flagged).

  Paths built in code (joins, variables) aren't seen. That is why section 8 also imports every script and runs every
  suite.
- **This is a draft.** It is retaken at the freeze as `inventory-before.json`. Since 18:04:
  - lab moved from abcf535 to cb85c62 (C3's grade, the bench-controls work);
  - NOTES.md grew from 7,878 to 8,013 lines.

  The self-test's hash check counts the drift: 17,780 files unchanged, 1 non-volatile change
  (`lab/viability/NOTES.md`), and 52 changed and 6 missing in the volatile folders where agents are working.

## 1. The target layout

```
/Users/eugene/prj/align/
├── anatta/                 public repo, main. Untouched (its uncommitted koans edits too)
├── anatta-lab/             private lab, branch lab. Code, writing, tracked results
│   └── lab/
│       ├── MAP.md          NEW: one page, where everything lives and what each document is for
│       ├── README.md       points to MAP.md first
│       ├── ledger/         NOTES.md (append-only, untouched), INDEX.md (NEW, generated), make_index.py (NEW),
│       │                   checks/ (+11 rescued check reports), notebook.md
│       ├── experiments/    h4lite/ (code + tracked results, in place), recognition/ (private, untouched)
│       ├── explore/  viability/  data/  site/  claim/  ideas/  organisms/  loop/
│       └── plans/          live plans; archive/ (+12 finished C2, C3 and C4 documents)
├── anatta-dev/             THE PINNED LIBRARY: detached at tag lib/ministral-ee89587, locked, read-only
├── anatta-calib/           NEW worktree of calib/all: calibration/{model_check, koan_scorer, relearning,
│                           persona_extractor, refusal_direction, prh_k1}
├── results/                NEW root outside git
│   ├── INDEX.md            generated: folder → code tag → command → ledger lines
│   ├── calib/              was calib-results/ (12 folders)
│   └── explore/            was explore-results/ (sweeps by date, cache, spend, variants log)
├── data/                   NEW root outside git, one folder per source, read as data only
│   ├── INDEX.md            source, origin, licence, sha256s, readers
│   ├── jbb/                was explore-data/jbb
│   └── essay/              was inputs/ (the founding essay's PDF)
├── archive-2026-11/
│   ├── README.md           what each part is and how to restore it
│   ├── worktrees/          anatta-calib-{c3,c4,c7,k1}: archived checkouts at their branch tips, locked, read-only
│   ├── premove/            the four outside-git folders as they were at the freeze (section 11, step 3)
│   └── scratchpad-073015e5/ the rescued session scratch (section 10)
│
│   shims (relative symlinks) so old paths keep working:
├── calib-results  → results/calib
├── explore-results → results/explore
├── explore-data   → data
├── inputs         → data/essay
├── anatta-calib-k1 → archive-2026-11/worktrees/anatta-calib-k1
├── anatta-calib-c6 → anatta-calib
│   existing symlinks, kept: claim, site, notebook.md, plan-2026-10-28.md, recognition (private)
```

Two more shims sit inside the lab: `lab/plans/declared-release-pilot.md` and `lab/plans/declared-release-pilot-screen.md`
point into `plans/archive/`. Section 3 says why.

**Why each part is where it is:**

- **The public repo isn't touched.** Its uncommitted koans edits are left alone. So are its ignored `directions/` and
  its `results/2026-10-28-viability` symlink, which Modal mounts. Calibration code reaches main later, through the
  curated PR (ideas 49).
- **`results/` and `data/` sit outside git, as the plan asks.**
  - Results can't be rebuilt without spend, and data is untrusted.
  - One root each means one place to look and one INDEX.md each.
  - The old names stay as symlinks. So the 48 citations of calib-results, the 48 of explore-results and the 9 of
    explore-data keep working wherever they are written: in code, in docs and in append-only ledger lines.
- **Results inside git stay in place.** This covers `lab/experiments/h4lite/results/`: 1,749 tracked files (1,255 MB),
  6 untracked, 3 ignored.
  - 115 files cite paths inside it, 68 of them code.
  - 67 NOTES lines mention it.
  - The graded code reads and writes there (section 6).
- **`anatta-dev/` stays at its path, frozen.** 61 lab files name that path (section 4). Freezing it in place changes
  no file and keeps every documented command working.
- **One calibration worktree: `anatta-calib` on calib/all.**
  - It reuses the path of the old calib/instruments worktree, so old citations of `anatta-calib/...` resolve again.
  - The four separate worktrees move into the archive as checkouts at their branch tips, locked and read-only, with
    their `consolidation-before-<date>/<branch>` tag as the lock reason. Graded code is read at its tag, with
    `git show calib/<ID>-graded:<path>`. C3's and C4's tips are their graded commits; K1's and C7's aren't (step 3.4
    lists the files that differ). (Fable, finding 11: this is now said one way everywhere.)
  - K1's gets a shim, because `lab/explore/p1/p1_core.py` imports K1's code from `anatta-calib-k1/` at run time.
- **MAP.md and the ledger index serve the one-minute test.** Nothing else is restructured: no code folder is renamed,
  no script is moved, and `h4lite/` keeps its 80 entries.

**The one-minute test, walked for K1 after the moves:**

1. Claim: "Our code reproduces the paper's language-vision agreement".
2. `lab/MAP.md`, step 1, sends you to `lab/ledger/INDEX.md`. K1's row reads:
   - registered at NOTES line 7108;
   - checks at 7370 and 7423;
   - graded at 7551 (PASS, five of five).
3. The grade entry names the code (`calib/K1-graded` = 237841e) and the results (`calib-results/prh_k1`, which
   resolves through the shim to `results/calib/prh_k1`).
4. `results/INDEX.md` gives the same folder. It lists `card.json`, the code tag, the command and the ledger lines.
5. Code: `anatta-calib/calibration/prh_k1/`. Its README has the commands. At the exact graded commit:
   `git -C anatta-calib show calib/K1-graded:calibration/prh_k1/card.py`.
6. Re-check from stored data with no spend: verify.py's K1 re-grade rebuilds `card.json` byte for byte. Its sha256
   f8e62eb0… is the same from today's worktree and from tag 237841e.

C3 walks the same way:
- registered at NOTES line 5476 (09:24 UTC), with notes from line 5941 to line 7255;
- graded at line 7880;
- tag `calib/C3-graded` = f878d5e;
- results `results/calib/refusal_direction_g4`;
- code `anatta-calib/calibration/refusal_direction/`.

## 2. The move map

`move-map.tsv` has 972 rows. Its columns are: old path, new path, kind, reason, references to rewrite, step, id, note.
No row deletes anything, and verify.py checks that (`delete_rows: 0`). The 7 rows added after Fable's check are at
the end (M0966-M0972), so every earlier id is the one Opus reviewed.

| kind | rows | what happens |
|---|---:|---|
| move | 4 | `mv` outside git: calib-results, explore-results, explore-data, inputs |
| shim | 8 | relative symlinks at old paths (6 at the top, 2 in lab/plans) |
| worktree-move | 4 | `git worktree move` of anatta-calib-{c3,c4,c7,k1} into the archive, then lock and make read-only |
| new-worktree | 1 | anatta-calib on calib/all |
| pin | 1 | anatta-dev: detach at the tag, lock, make read-only |
| git-mv | 12 | finished C2, C3 and C4 plan documents into lab/plans/archive/ (the archive README's rule) |
| rewrite | 4 | lab/README.md; lab/explore/README.md lines 12 and 34; scout/paths.py:23 (with the scout owner's OK); explore/PROGRAM.md:47 (only with Eugene's OK) |
| new | 6 | results/INDEX.md, data/INDEX.md, archive README, lab/ledger/make_index.py, lab/ledger/INDEX.md, and the branch bundle (made at 19:35 UTC) |
| edit | 4 | files the consolidation changes by hand, each with a rule verify.py applies (Fable, finding 13): NOTES.md (`rule=prefix`: the before bytes must be a prefix of the after bytes), plans/archive/README.md (`rule=prefix`: names appended), lab/MAP.md (`rule=any`; committed at a82ca71, so it is no longer `new`) and this design (`rule=any`) |
| copy | 4 | `archive-2026-11/premove/{calib-results, explore-results, explore-data, inputs}`: step 3.2 keeps each original there, and verify checks every file byte-equal (Fable, finding 13) |
| resolve | 3 | worktrees removed before this design (anatta-calib-c1, -c6, -relearning); citations of them resolve to anatta-calib |
| keep | 14 | named so nobody moves them: h4lite results and data, lab/data, main, the private folders, the existing symlinks |
| rescue | 749 | scratchpad files → archive-2026-11/scratchpad-073015e5/ (24.6 MB) |
| rescue-dir | 96 | scratchpad folders kept whole, with their relative paths (1,730 files, 45.8 MB) |
| rescue-dup | 20 | scratchpad files byte-identical to a file already kept (the row names it) |
| to-git | 11 | rescued check reports and facts files that belong in the lab (section 10) |
| leave | 31 | scratchpad folders not kept: downloads, third-party code, installed packages, browser profiles, and `smoke_wim` after Opus's review (9,560 files, 826.8 MB; each row says why) |

## 3. Reference rewrites and shims

`refs-by-move.tsv` lists 682 citations of moved paths (from the 18:04 draft), each with what keeps it working: 472
through the map, 209 by a shim, 1 rewritten. By match strength: 21 absolute, 9 relative, 463 relative in another place
and 189 bare file names.

**What the guarantee rests on (Fable, finding 9).** Not on the 682. The bare names are a first pass, and most "relative
in another place" matches are one repo's files naming their own `src/anatta/...`. Old paths keep working because of:
- the absolute citations, each covered by a shim or the one rewrite (below);
- importing every lab script, running every suite and the CPU re-grades, all repeated in after mode;
- the ledger check, which follows the last 30 entries' paths through the map.

**Code that cites a moved path by absolute path.** In the draft, 13 files. In the rehearsal inventory (20:18 UTC),
21, all covered by a shim or the rewrite:
- 2 calibration apps cite calib-results (C7's and K1's `app.py`).
- 16 explore files cite explore-results: scout's `paths.py` (the rewrite), 10 bench specs (lead1, lead6 and lead7
  ones, their split and smoke variants, `smoke-controls.yaml`), bench's `tests/test_controls.py`, and p1's
  `modal_p1.py`, `modal_p1b.py`, `p1_analysis.py` and `spec.yaml`. Most were written after the draft.
- 2 scout specs cite explore-data (`first-sweep.yaml`, `smoke-harm.yaml`).
- `p1_core.py` cites anatta-calib-k1.

**No code names a moved calibration worktree (C3, C4 or C7) by absolute path. Docs do.** NOTES.md line 7365 names
`anatta-calib-c3`, and so do the C3 grade check, the C3 code check and Fable's check of this design. The C4 grade check
and the C4 code check name `anatta-calib-c4`. They resolve through MAP.md's old-paths table. The draft's scanner missed
them for a reason other than the one the check guessed: it already stripped the trailing period. It dropped every
citation of a worktree or top-level folder named as a whole. The patched `inventory.py` keeps those (its rehearsal run
finds 9 places cited that way), and it now also strips a trailing `.,:;`.

**The table is rebuilt at the freeze.** After `inventory-before.json` is taken, `python3 -I make_move_map.py
--refs-only --inventory inventory-before.json` writes `refs-by-move-freeze.tsv` and leaves `move-map.tsv` alone. The
rehearsal's run of the same command (`refs-by-move-rehearsal.tsv`) finds 638 citations, 43 of them absolute.

**The shims:**

| old path | target | why it must keep working |
|---|---|---|
| calib-results | results/calib | the calibration apps write and read there; 32 code files and 16 docs cite it |
| explore-results | results/explore | scout's DEFAULT_RESULTS (paths.py:23), bench and p1 specs, the variants log |
| explore-data | data | scout specs name explore-data/jbb; the C3 tests read it |
| inputs | data/essay | nothing cites it; kept for the habit of the path |
| anatta-calib-k1 | archive-2026-11/worktrees/anatta-calib-k1 | p1_core.py's K1_DIR imports K1's code from it (the p1 owner can later point it at anatta-calib/calibration/prh_k1). The checkout is K1's tip 83067d7, whose code files are identical to the graded 237841e |
| anatta-calib-c6 | anatta-calib | `h4lite/c6_samples.py`'s default --samples-module points into the C6 worktree, which was removed before this design, so it is broken today; calib/all holds the same file |
| lab/plans/declared-release-pilot.md | archive/declared-release-pilot.md | archived at 93bbac4 (1 Nov) while `declared_core.py:57` still reads it (DESIGN_FILE, hashed into each run's provenance at line 1111) and `run_pilot.sh:80` lists it. `test_declared.py` fails on it today. Same bytes, so the registered sha256 00f0cb81… still matches. |
| lab/plans/declared-release-pilot-screen.md | archive/declared-release-pilot-screen.md | the screen amendment that the declared pilot's code and driver name |

**Things that need no rewrite:**

- **The site build** (`lab/site/so_far/build.py`) reads only `lab/ideas/ideas.md`, its own `build.py`, `content/*`
  and its `index.html`. verify.py recorded every file the build opened. None of them moves. The tally is
  {'hit': 43, 'miss': 73, 'open': 7}.
- **Modal mounts.** `h4_modal.py` and `modal_common.py` mount `anatta-dev/src` and the private
  `recognition/scripts`. Neither moves. Importing all 83 lab scripts under a stand-in Modal found 0 missing mount
  paths.
- **Specs and configs** name explore-results and explore-data, and the shims cover them. Rewriting them is the owners'
  choice, after the moves.

**How rewrites are applied.** A script reads the exact old and new line from the map. It writes
`rewrites-applied.tsv`, recording each file's sha256 before and after. verify.py's hash check uses that file to
account for every changed byte. The four rewrites change documentation and one default path. No graded or registered
code is edited. Files changed by hand rather than by exact rewrite are the four `edit` rows (section 2).

**Every shim is checked (Fable, finding 14).** verify.py's `shims` check reads each shim row. After the moves, the old
path must be a symlink that exists and whose realpath is the target's, so a dangling link fails. Before, each old path
must be free or be the folder that a move takes away first (today: 5 are those folders, 3 are free).

**Found broken before this design** (none is caused by it; the design fixes the first two with shims):

1. **The declared pilot** reads two plan files that were archived this morning (above).
2. **`c6_samples.py`'s default** points into a removed worktree (above).
3. **`test_cal_judges.py`'s documented command** uses the matlab venv, which has no pytest. The anatta venv has pytest
   but no modal, which the file imports. verify.py borrows pytest for that run (section 8). The fix is a one-line
   README change, or `pip install pytest` in the matlab venv, if Eugene wants it.
4. **`lab/explore/p1/p1_core.py` appends the pin to the end of sys.path.** Under the anatta venv, the editable install
   of main's `src` comes first and wins.
   - So locally (p1's tests, any local run without PYTHONPATH), p1 imports main's library, uncommitted koans edits
     included, not the pin.
   - Inside its Modal containers, it depends on what the image installs. I didn't check that.
   - One line fixes it (insert at 0 instead). It belongs to p1's owner and isn't part of these moves.
5. **`lab/README.md` line 58** names `.premove-backup-2026-10-29`, which is gone. The README rewrite drops the line.

## 4. Library pinning

**Who loads `anatta-dev`** (verify.py's pins check; a docstring is told apart from code by parsing):

- 61 lab files name it: 39 .py, 4 .sh, 18 .md. Two earlier counts, the caller's "about 31" and the ledger's "62 files"
  at 16:09, were not computed here.
- Of those, 43 are code files, and 15 load the library:

| mechanism | files |
|---|---|
| put the pin first on sys.path | koan_pairs.py, lcb_loader.py, positive_control_adapter.py, explore/bench/client.py, bench/tests/conftest.py |
| append it (main's editable install wins; item 4 above) | explore/p1/p1_core.py |
| mount it into Modal containers and set the container's PYTHONPATH | h4_modal.py, modal_common.py (imported by about 20 Modal apps) |
| set PYTHONPATH in a shell driver | atlas_cuts.sh, durability.sh, self_tests.sh, declared/run_pilot.sh |
| hash the pin's files for provenance | c6_samples.py (and passes ANATTA_DEV on), declared/declared_core.py |
| read data files from it | explore/bench/controls.py |

- The other 28 code files name the path only in docstring run lines (`PYTHONPATH=<anatta-dev>/src ...`).
- **What the pin is.** anatta-dev is clean at ee89587, committed 2026-10-28 15:06:54 UTC, with no file newer than the
  commit. dev/ministral is wholly inside lab, and no lab commit after ee89587 touches `src/`. So the library that
  every lab run since 28 Oct imported is exactly ee89587's `src/`.

**Option A: merge dev/ministral into lab and load the lab's own `src`.**
- The merge is already done in effect: it adds no code.
- The work is re-pointing about 15 loaders, 4 shell drivers, 28 docstrings and the Modal mounts at
  `anatta-lab/src`.
- That edits registered and graded scripts, and changes the provenance hashes that `c6_samples.py` and
  `declared_core.py` record.
- It also trades one live path for another. `anatta-lab/src` is in a worktree where agents commit every hour, so a lab
  commit to `src/` would silently change old experiments, which is the very problem we're fixing.

**Option B (recommended): tag and pin in place.**
- Tag `lib/ministral-ee89587` at ee89587.
- Detach anatta-dev at the tag: `git -C anatta-dev switch --detach lib/ministral-ee89587`. The files are unchanged.
- `git worktree lock --reason "pinned library for lab scripts"`.
- `chmod -R a-w anatta-dev`.
- Zero script edits. Every documented command keeps working. Any attempt to change the library fails loudly.
- **Reproducibility from the tag:** `git show lib/ministral-ee89587:src/anatta/<file>`, or
  `git worktree add <dir> lib/ministral-ee89587` for a full checkout. verify.py's pins check confirms that the
  worktree's HEAD is the tag, that it is clean, and that no file is newer than the commit.
- **Risks, with their guards:**
  - Python can't write `__pycache__` into a read-only tree. It skips this silently, so it's harmless.
  - A new library change for the lab now needs a deliberate step: a new tag (`lib/<name>-<sha>`), then moving the pin
    in one commit with a ledger line. That is the point.
  - Item 4 above (p1's append) still gets main's library under the anatta venv until its owner fixes it.

## 5. Calibration

| branch | tip | worktree now | graded commit → tag | ledger |
|---|---|---|---|---|
| calib/instruments | dc32cf7 (main + 29) | none | the base of calib/all | |
| calib/model-check (C1) | 44760f7 | none | 44760f7 → calib/C1-graded; 51a0942 → calib/C1-first-launch | grade at line 5111 |
| calib/koan-scorer (C6) | 12e831e | none (removed) | 12e831e → calib/C6-graded | line 4918 |
| calib/relearning (C7) | 9a9f7bc | anatta-calib-c7 | 02db868 → calib/C7-graded (inside the unlicensed range, private only); 9a9f7bc → calib/C7-own-scorer | lines 4872, 7657 |
| calib/persona-extractor (C4) | 1711324 | anatta-calib-c4 | 1711324 → calib/C4-graded | line 7709 |
| calib/refusal-direction (C3) | f878d5e | anatta-calib-c3 | f878d5e → calib/C3-graded | line 7880 |
| calib/prh-k1 (K1) | 83067d7 | anatta-calib-k1 | 237841e → calib/K1-graded | line 7551 |
| lab (C2) | | anatta-lab | f363ffb → calib/C2-graded (cal_judges.py sha256 657bf09e) | line 4377 |

Together with `lib/ministral-ee89587`, that makes 10 tags (`tags.tsv`).

**Each tag is checked against its card.** verify.py hashes the code files that each card or plan names, as they are at
the tagged commit (`git show <commit>:<path>`), and compares:
- C1: 12/12 at 44760f7;
- C3: 12/12 at f878d5e (from `plan.json`, since C3's `card.json` carries no code hashes);
- C4: 12/12;
- C6: 13/13;
- C7: 8/8 at 02db868 (only 4/8 at its tip 9a9f7bc, which is why the graded tag isn't the tip);
- K1: 3/3.

Tags are annotated, with the evidence line from `tags.tsv` as the message, so each card still names the exact code
that ran.

**calib/all:**
- Branch from calib/instruments.
- Merge with `--no-ff`, in this order: model-check, koan-scorer, relearning, persona-extractor, refusal-direction,
  prh-k1.
- Forecast, from comparing the files each branch changed since its merge base: 21 files are touched by more than one
  branch. Only `THIRD_PARTY.md` truly conflicts, and the resolution is to keep both sides' entries.
  - The model_check files changed on one branch only.
  - pyproject, contrast and models changed only on refusal-direction.
  - prh-k1 overlaps nothing.
- After the merges, `pytest -q tests calibration` must keep every test id that passed in the four worktrees (verify.py
  after mode).
- **The merged tree is checked against the branch tips (Fable, finding 6).** verify.py's `calibmerge` check (after
  mode) runs `git -C anatta diff --stat calib/<branch> calib/all -- calibration/<dir>/ tests/` for each instrument.
  - Literal emptiness can't hold for `tests/`. For example, refusal-direction changes `tests/test_contrast.py`, which
    the other tips have at calib/instruments' version.
  - So every differing file must be explained. One branch's commits on that file (since calib/instruments) must cover
    every other branch's, and calib/all must equal that branch's version. THIRD_PARTY.md is the union exception.
  - A file changed independently on several branches is listed for review.
  - A forecast from `git log` alone: all 84 files under `tests/` and `calibration/` that any instrument branch touched
    have exactly one winning branch. So after correct merges the check should pass with nothing to review.
  - Its output is kept in the scratchpad and written into the step-4 ledger entry, since step 2.6 allows no ledger
    writes before the after inventory (changed after Fable's re-check, 1 Nov).
  - Its limit: it compares only `calibration/<dir>/` and `tests/`. Files the instrument branches changed elsewhere
    (root THIRD_PARTY.md, pyproject.toml, `src/`) are not compared; the step-3 reader diffs those by hand against
    each tip with `git diff calib/<branch> calib/all -- THIRD_PARTY.md pyproject.toml src/`.

**The hard rule.** calib/all stays private. Calibration code reaches main only as fresh commits of the current files,
never by merging or cherry-picking calib/* history. C7's history holds the port of an unlicensed scorer, d0aac0b to
02db868 (NOTES, 17:11 UTC on 1 Nov). The tag `calib/C7-graded` points inside that range.

**A live leak risk, and a proposed guard (needs Eugene's OK).** The pre-push hook (`anatta/.git/hooks/pre-push`)
refuses `refs/heads/lab*`, and any ref with `lab/` in its history, when the target is the public repo. calib/* has no
`lab/` history, so today `git push origin calib/relearning`, or `git push --tags`, would publish the port.

The draft's version tested only the local ref name. `git push origin HEAD` passes `HEAD` as the local ref, and a sha
push passes the sha, so neither carried a name it could match (Fable, finding 3). This version, Fable's, replaces the
loop's tests:

```sh
  case "$lref:$rref" in
    refs/heads/calib/*:*|refs/tags/calib/*:*|refs/tags/lib/*:*|refs/tags/consolidation-before-*:*|refs/heads/consolidate/*:*|\
    *:refs/heads/calib/*|*:refs/tags/calib/*|*:refs/heads/lab*|*:refs/heads/consolidate/*|*:refs/heads/dev/*)
      echo "pre-push: refusing $lref -> $rref on the public repo ($url)" >&2; exit 1 ;;
  esac
  if [ -n "$(git rev-list -n 1 "$lsha" -- lab/ calibration/ 2>/dev/null)" ]; then
    echo "pre-push: $lref carries lab/ or calibration/ history; refusing ($url)" >&2; exit 1
  fi
  if git merge-base --is-ancestor d0aac0b "$lsha" 2>/dev/null; then
    echo "pre-push: $lref contains C7's unlicensed port (d0aac0b..02db868); refusing ($url)" >&2; exit 1
  fi
```

- **Both sides of each ref are tested.** A push is refused if the local ref or the remote ref is private, so `HEAD`,
  sha and `--mirror` pushes are caught by their destination.
- **The `calibration/` path rule** sits beside the `lab/` one. All six instrument branches carry `calibration/`
  history. main, lab, dev/ministral and calib/instruments don't.
- **The ancestry rule** catches C7's port under any name, tags included.
- **The `calibration/` rule must be relaxed on purpose for the curated PR** (ideas 49), because that PR brings
  `calibration/` to main as fresh commits. Write the relaxation into the PR's plan, in the same commit, and put the rule
  back after the push.
- **`git branch --unset-upstream calib/instruments`** (a config change: Eugene's decision). Today it tracks
  origin/main. `push.default simple` stops a bare `git push` there, but not `git push origin HEAD`.
- **One file covers every worktree.** All worktrees share `anatta/.git`, and `core.hooksPath` is unset. But
  `--no-verify` skips any hook. The GitHub ruleset in decision 1 is the guard that holds then.

The hook is configuration, so a design can propose it but can't approve it.

**Backup (needs Eugene's OK).** calib/* (7 branches) and dev/ministral exist only on this laptop. One disk failure
loses graded calibration history. At the freeze, once the hook is extended, push them to the private remote:

    git -C anatta push private 'refs/heads/calib/*:refs/heads/calib/*' refs/heads/dev/ministral \
        'refs/tags/calib/*' 'refs/tags/lib/*' 'refs/tags/consolidation-before-*'

That places the port's history on GitHub, privately. If Eugene would rather it never leave the laptop, use a bundle
instead: `git bundle create ~/anatta-calib-2026-11.bundle --branches='calib/*' --tags='calib/*'`. Opus made one at
19:35 UTC, `archive-2026-11/branches-2026-11-01.bundle` (map row M0970), but it sits inside `prj/align`, which Time
Machine excludes, so it isn't yet a second copy (Fable, finding 1; decision 6).

**Protecting C3.**
- C3's full run finished (card written 18:31 UTC), and it was graded at 18:52 UTC.
- Its results folder (`refusal_direction_g4`) moves with calib-results, and the shim keeps the card path that the grade
  entry cites.
- Its worktree moves into the archive after the freeze, at its branch tip, which is its graded commit f878d5e.

## 6. Results and data roots

**`results/`:**
- `calib/` holds 12 folders, 154 files, 30.8 MB. The largest are model_check (16.8 MB) and koan_scorer (10.8 MB). The
  rest are 1.2 MB or less: refusal_direction_g4, persona_extractor, relearning, prh_k1, the refusal_direction diag and
  g2/g3 runs, and two prepare folders.
- `explore/` holds 2,521 files, 68.4 MB: the 2026-11-01 sweeps (16 folders and a report), the cache, the spend
  records and `variants.jsonl`.

**`data/`:**
- `jbb/` (1 file);
- `essay/` (the PDF, sha256 74b7f9e8336da2bc…).

`lab/data/<set>/raw` stays where it is: 5 tracked files and 293 ignored, 149 MB. `fetch.py`, its `--check` and its
tests treat `lab/data/<set>/raw` as the place, and they test that nothing under `raw/` is importable. data/INDEX.md
lists those sets too, so there is one place to look.

**results/INDEX.md is generated by `lab/ledger/make_index.py`.** One row per results folder:

| column | from |
|---|---|
| folder | the directory under results/ (and the tracked h4lite results folders, and the Modal volumes by name: anatta-runs, anatta-hf-cache, anatta-calib-prh-k1, anatta-explore-p1) |
| experiment | the ID tokens in the folder's card or plan (C1 … K1), else its path |
| code | the graded tag, or the commit its card names |
| command | the README line that makes it |
| ledger | NOTES line numbers of the registration, grade and limits entries that cite the folder |
| files, MB | counted |

**data/INDEX.md** is written by hand once, with one row per source: where it came from (URL, revision), licence,
sha256s (from the manifest), who reads it, and "untrusted: read as data only".

**Why the tracked h4lite results stay inside git:**
- they are 1,749 files (1,255 MB) that the ledger, the site and 68 code files cite by path;
- the graded scripts write their outputs there;
- moving them would mean either editing graded code or adding hundreds of shims;
- git history keeps them reproducible at every commit anyway;
- the plan's own rule ("unless the design step finds a clean way") isn't met.

The 6 untracked files (Gate 0's pairs and parts, 22 MB) stay untracked in place, because the gate0 code reads them
there.

## 7. The old top-level copies, compared

The plan expected copies beside the lab. The inventory says otherwise:

| top-level path | what it is | verdict |
|---|---|---|
| claim/ | symlink → anatta-lab/lab/claim | not a copy: same files by construction. Keep (Claude's memory cites it) |
| site/ | symlink → anatta-lab/lab/site | keep (memory cites prj/align/site/build.py) |
| notebook.md | symlink → lab/ledger/notebook.md | keep |
| plan-2026-10-28.md | symlink → lab/plans/plan-2026-10-28.md | keep |
| recognition | symlink → the private folder | untouched, never followed. `fishery_play.py` and the Modal containers use the ALIGN/recognition path |
| inputs/ | one PDF (3.1 MB), found nowhere in the lab, cited by no file | → data/essay, with a shim (optional; drop the row if Eugene prefers it at the top) |
| .premove-backup-2026-10-29 | gone before this design | lab/README.md line 58 still names it; the rewrite drops the line |

So nothing at the top needs a file-by-file merge. The archive receives only the four calibration worktrees, the
pre-move copies, the rescued scratchpad and the branch bundle.

## 8. verify.py

`verify.py` is written, and runs read-only in "before" mode as a self-test. Its after mode has been rehearsed on the
unmoved tree (below) and hasn't been run against any move. Usage: `python3 -I verify.py --mode before|after
[--only check,...] [--suite NAME] [--jobs N] [--label NAME]`. It writes `verify-<mode>[-<label>].json`.

**The guard.** Every check that runs lab code does so in a child process (`guard/child.py`). Before any lab code runs,
the child installs Python audit hooks and wrappers:
- **Private paths.** A read, list or stat of anything in a private folder gets "no such file" before any system call
  is made, so the code behaves as if the folder were absent. This covers any path with a component named `account` or
  `recognition`, and the symlinks into them. Private sys.path entries get a finder that finds nothing.
- **Credentials.** Credential files are refused (`~/.modal.toml`, HF tokens, `~/.netrc`). Modal and Hugging Face are
  pointed at files that don't exist.
- **Writes, processes, network.** Writes go only to the run's own folder, inside `consolidation-2026-11/tmp/`. inspect_ai's
  user folders are redirected there too. No process starts, with one exception: the same interpreter may run a `.py`
  file inside the run's folder (the koan tests' harness), and each such run is recorded. No network beyond loopback.
- **Output.** The lab code's output is captured and dropped. Failure messages are kept only for import and
  missing-file errors (module names and paths), never for assertions.
- **Accounting.** Every block is counted, so each failing test is marked as guard-limited (something was blocked or
  hidden during it) or real.

**The checks, and what the before self-test found** (`verify-before.json`, run at 19:30:00 UTC in 146.8 s;
verify.py sha256 2cf9c8bd…, guard/child.py 172c7fdf…, both recorded in the report). The checks changed after Fable's
check were re-run in before mode with the revised script (sha256 d7c4278d…) at 20:27:39 UTC
(`verify-before-fable-fixes-partial.json`):

| check | before mode (now) | after mode (step 4) | result now |
|---|---|---|---|
| map | rows well-formed, kinds known, no delete rows; every move, copy and edit source exists (C3's old exemption is gone: Fable, finding 17) | the same | pass: 972 rows, 0 problems |
| privacy | the private folders exist, seen from their parents' listings; main's 3 koans edits unchanged | the same (and the map check refuses any row that touches a private folder) | pass |
| pins | anatta-dev clean at ee89587, no newer file, inside lab | HEAD detached at the tag, locked, read-only | pass |
| tags | each card's code hashes at its commit: every file matches and none is missing (Fable, finding 16) | the tags exist and point there | pass: C1 12/12, C3 12/12, C4 12/12, C6 13/13, C7 8/8, K1 3/3, C2 657bf09e; still all confirmed under the stricter rule |
| shims | each shim's old path is free, or is the folder a move takes away first | each old path is a symlink that exists and resolves (realpath) to its target (Fable, finding 14) | info: 5 old paths are folders that move away, 3 are free |
| calibmerge | skipped | each instrument's tip against calib/all over `calibration/<dir>/` and `tests/`; every difference explained (Fable, finding 6; section 5) | skipped; the forecast: 84 files, one winning branch each |
| ledger | every path-like token in the last 30 entries exists | the same paths exist through the map and the rescue manifest; `exists`, so a dangling symlink fails (Fable, finding 14) | pass: lines 7108-8013; 43 resolve, 35 aren't paths (17 branches, 13 model ids, 3 in outside repos, 2 prose), 4 private, 0 unresolved |
| site | builds twice the same; tally | the same tally, and the page byte-identical to the before build | pass: {'hit': 43, 'miss': 73, 'open': 7}; the page equals the committed index.html (sha256 029b4086…) |
| regrades | CPU re-grades from stored data | identical | pass: fishery_grade 37/37 identical; self_tests_grade self1/atlas1 identical (hits 124, 127, 130, 132); K1 card identical from the worktree and from 237841e |
| hashes | the inventory against disk | every before sha256 at its mapped path (moved or unchanged), or rewritten (rewrites-applied.tsv), or edited under its rule; every file under a copy row byte-equal in premove/; every after file accounted for (a before file, a new or rescued path in the map, a manifest copy), and none left at a moved folder's old path (Fable, finding 13) | info: 17,780 unchanged, 1 non-volatile change |
| rescue | every kept scratch file has a row in `rescue-manifest.tsv`, and each copy still has the sha256 recorded at copy time; files new since the draft are counted | the same, and nothing pending | pass: 2,479 in the manifest, 0 damaged, 0 pending (the 6 logs that grew after 18:04 were copied at their later contents) |
| imports | import every lab .py under the guard, with a stand-in modal (`--help` isn't run: two scripts read argv at import and could take it as an argument) | every file that imported before still imports; no mount goes missing | 83 files: 79 import; 1 compile-only (scout's `__main__`); 1 writes at import (`lab/site/build.py`); 2 read argv at import (viability `make_dials.py`, `wait_model.py`); anatta came from the pin in all 6 that import it; 0 missing mounts; 9 shell scripts parse |
| suites | run every suite by its documented command | per suite (the four calib worktrees together against calib all): no passed id lost, no new failing id, and no worse verdict, exit code or failed count; C4's two failing app tests are a named allowance, and only they (Fable, finding 15) | 34 suites; 1,529 tests pass (below) |

**The suites:**

| suite (documented command) | result |
|---|---|
| public library, lab's copy, pinned copy (`pytest -q tests`) | 35, 41 and 41 pass. 1 guard-limited each: the inspect_ai test writes to a hard-coded `/tmp/anatta-inspect-test` |
| calib C3, C4, C7, K1 (`pytest`, in each worktree) | 362, 244, 307 and 137 pass. Guard-limited: 3, 1, 2, 1 (inspect_ai; tests that start `sh` or `python3.11`). **C4: 2 real failures** in `test_c4_app.py` (`Failed`, `SystemExit`). The cause wasn't investigated: the comparison is before against after, and reading C4's results files is off-limits |
| h4lite pytest set (README) | 77 pass. 3 guard-limited (the guard hid or blocked something during each) |
| h4lite test_cal_judges (documented venv, pytest borrowed) | 64 pass, including the C2 test that reproduces the graded numbers from stored data |
| h4lite scripts: bridge_core, fishery_eval, fishery_grade, hub_cut, koan_env, scorer_adversarial, self_tests, spiritual_syco, sycophancy_core | 9 ok |
| h4lite scripts needing the private folder: battery_score, fishery_play, lie_block, positive_control, positive_control_adapter, restoring_force, third_person | 7 guard-limited. They import `harness_v2`, `commons_variants` or `what_it_means` from `lab/experiments/recognition/scripts`, or read its results. **Only someone allowed in the private folder can run these (section 11, step 4)** |
| h4lite scripts that start another interpreter: atlas_cuts, coding_eval, durability | 3 guard-limited |
| declared/test_declared.py | **real failure, from before this design:** the archived plan file. The shim fixes it, so the after run should show it passing |
| lab/data/test_fetch.py (`python -I`) | 19 pass |
| site/so_far/test_build.py | 26 pass |
| explore scout, bench | 86 and 57 pass (bench grew from 49 during the day, as its agent added controls) |
| explore p1 | 33 pass. **1 real failure** (`test_the_controls_save_as_bench_saves_them`, AssertionError), while the bench-controls agent is editing bench and p1. It's a moving target, so re-take the baseline at the freeze |

**The rehearsal (Fable, finding 13).** One inventory of the unmoved tree (`inventory-rehearsal.json`, 20:18:30 UTC,
31,444 files, with the patched `inventory.py`) served as both sides:

    python3 -I verify.py --mode after --inventory-before inventory-rehearsal.json \
        --inventory-after inventory-rehearsal.json --label rehearsal

It ran at 20:27:23 UTC in 42.0 s (`verify-after-rehearsal.json`; verify.py d7c4278d…). On an unmoved tree, every moved
file should come back missing at its mapped path, and nothing else should:

- **hashes:** 14,163 files missing at their mapped path. That is exactly the moved files: calib-results 160,
  explore-results 13,586, explore-data 1, inputs 1, the four worktrees 106 (C3), 108 (C4), 111 (C7) and 78 (K1), and the
  12 plan documents. The same 14,163 files show up still at their old paths. The 13,748 files of the four outside-git
  folders also show their premove copies missing. 5,182 files are unchanged. No file changed, and no file is
  unexplained.
- **Failing as expected on an unmoved tree:**
  - pins: not yet detached, locked or read-only;
  - tags: the tags don't exist yet, and the cards are read from `results/calib/`;
  - shims: none made yet;
  - calibmerge: no calib/all;
  - ledger: 22 of 82 paths, all under the moved folders;
  - regrades: K1's re-grade reads `results/calib/` and `anatta-calib/`, which don't exist yet; the fishery re-grade
    (37/37) and the self-tests re-grade are identical;
  - suites: calib all's folder is missing;
  - rescue: the 40 new scratch files wait for review.
- **Passing:** map, privacy, imports (80 files import, one more than at 19:30: `modal_p1b.py` is new; none lost).
- **Two drifts since the 19:30 baseline, neither caused by this design:**
  - `lab/ideas/ideas.md` changed at 19:37 UTC (363211a), and the site's `index.html` was last built at 18:55. So the
    page differs, and `test_build.py`'s freshness test fails. Rebuild the site before the freeze baseline.
  - explore-results grew from 2,521 files to 13,586 (the running sweeps).
- **What an unmoved tree can't reach was tested on synthetic data:** files found at a new path, edits under both rules,
  copies, manifest copies, a file left at an old path, the per-suite comparison and the C4 allowance.
  `tmp/selftest/test_after_hashes.py` and `tmp/selftest/test_compare_suites.py` both pass.
- **Not exercised:** the calibmerge comparison itself, because it runs `git diff` and needs calib/all.

**What verify can't see:**
- Code paths that no test, import or re-grade exercises.
- The 7 private-dependent scripts.
- The contents of the private folders, by rule. That includes whether `harness_v2.py` is tracked on lab: graded fishery
  and coding results depend on it (section 11).
- Paths built at run time that no test reaches.

## 9. MAP.md and the ledger index

**`lab/MAP.md`** was committed as a draft at a82ca71 and revised after Fable's check: its old-paths table now covers
the three moved worktrees and the archived plans, and it says "archived checkout" one way (findings 10 and 11). It
describes the layout after the moves, and says so at the top. It has:
- the claim-to-evidence path in five steps;
- a table of where each kind of thing lives;
- what each document is for;
- the old paths that still work;
- the rules (privacy, append-only, data is untrusted).

**`lab/ledger/make_index.py`** generates `lab/ledger/INDEX.md` (to be written in step 3; standard library only):

- **Entries.** It splits NOTES.md into entries at its top-level `- **...**` lines and keeps each entry's first line
  number. Line numbers are stable because NOTES.md is append-only.
- **Kind.** It classifies each entry by its title: registered, amended, check (Fable or Opus), graded, limits ("doesn't
  cover", "what C3 doesn't"), note, sighting, decision.
- **Experiment.** It assigns each entry to experiments by ID tokens in the title and first lines: C1-C8, K1, P1,
  H4-lite, declared, fishery, self-tests and so on, from a small hand-kept alias table at the top of the script.
- **Output.** One row per experiment, in this order: registration line, amendment and check lines, grade line and
  verdict word, limits lines, code tag (from `tags.tsv` or the grade entry's "code at"), results folder. Then the full
  entry list by line.
- **Unknown entries** go in an "unassigned" section rather than being guessed.
- **When it runs:** after each new entry, by whoever writes the entry, or from the site build. The index never edits
  NOTES.md.
- **Its check:** every line number in INDEX.md starts an entry in NOTES.md, and every graded title appears in exactly
  one row. verify.py's ledger check counts 45 "registered" and 56 "graded" titles today, so the index must account
  for each.

## 10. The scratchpad: what to keep and where

This session's scratchpad (`/private/tmp/claude-504/.../scratchpad`) holds 12,059 files (941 MB). The ledger cites
277 paths in it (`scratch-citations.tsv`). Session folders go when the session ends, and macOS clears `/private/tmp`
on reboot. **So the rescue should run first (step 1b), not with the moves.** It is a copy (`cp -p`), so nothing is
deleted.

**Keep (outside git): `archive-2026-11/scratchpad-073015e5/`**, with relative paths unchanged:
- 749 files: reports, logs, notes, scripts and `.bak` copies of NOTES taken before edits.
- 96 folders: Fable check workspaces, smokes, probes. 2,479 files, about 70 MB in all.
- 20 more files are byte-identical to a file already kept elsewhere, and their rows name the twin. They aren't
  copied. A citation of one resolves through the map to its twin, and on to the twin's new place if the twin moves too.

**Keep outside git only, never in any repo:**
- the C7 folders (`fable_c7_grade` including `code_at_02db868`, `fable_c7_code`, `fable_c7_design`, `c7_impl`, `c7`,
  `c7_cpu_results`, `c7tok`): they hold or quote the unlicensed port;
- `billing`.

**Done, first pass.** Opus reviewed the keep list (`rescue-reviewed.txt`, 19:33 UTC) and removed `smoke_wim`, which
holds a copy of a recognition-derived direction. `rescue.py --go` copied 2,479 files at 19:33-19:35 UTC.

**The manifest (Fable, finding 13).** `rescue-manifest.tsv` has one row per copy: source, destination, the copy's
sha256, size, time and how. The first pass ran before the manifest existed, so `rescue.py --manifest-only` (20:17 UTC)
recorded each copy's sha256. Every copy's ctime falls inside 19:33-19:36 UTC, so nothing has written to one since. No
scratch file has changed since its copy. The six logs that grew after the 18:04 draft were copied at their later
contents, and the manifest records those. verify.py's rescue check reads the manifest. It no longer reports those six
as mismatches.

**The second pass.** `rescue.py --newer` (20:21 UTC, a dry run) walks the live scratchpad, with the same folder rules
and the same prunes as the inventory. It lists 40 files made after the draft, 0.27 MB, in `rescue-newer.tsv`:
- two Fable reports (`c3_grade_fable_check.md`, `consolidation_design_fable_check.md`);
- messages (`msg_*.txt`) and a handback;
- the bench agent's `.orig` backups of bench files;
- C3 helper scripts (`c3_*.py`), scout's `first_sweep.log` and a few small JSON and helper files.

Opus reviews them, writes `rescue-reviewed-2.txt` (with a `leave: <path>` line for anything excluded), and runs
`rescue.py --newer --go`. Then once more just before the session ends. A file that changed after its copy is listed,
never overwritten.

**Into git after Opus reviews them** (11 files; Fable, finding 5, left to Opus: read each before copying it in step
3.1, and record "read, no items, no keys" per file in the ledger entry):
- 7 check reports go to `lab/ledger/checks/2026-11-01-*.md`: c3-gate-fable-check, c3-recheck-note,
  c4-postprepare-fable-check, k1-fable-check, k1-grade-fable-check, metric-audit, site-fable-check.
- 4 facts files go to `lab/plans/archive/calibration-c{1,2,6,7}-facts.md`.

**Not kept** (30 folders, 9,558 files, 826.7 MB; each row in the map says why):
- downloads and papers, which can be fetched again;
- third-party source at pinned commits, which can be fetched again at the commits the checks name;
- installed packages;
- two Chrome profiles (cookies), which are never kept;
- screenshots and test scaffolding.

**Restricted-content review.** Before anything is copied, Opus reviews the keep list for restricted content: benchmark
items, harmful completions, keys, or anything from the private folders. The C7 folders are the known case. Any other
hit moves the file to "outside git only" or "not kept".

## 11. Freeze and execution (steps 1b-6)

**Step 1b: before the freeze.** Any time; none of these move anything.
1. A cold Fable check of this design. Done: NOT READY, with six changes required; this revision makes the ones that
   are code and text.
2. The scratchpad rescue (section 10). The first pass is done and in the manifest. `verify.py --mode before --only
   rescue` passes: 2,479 in the manifest, 0 pending, 0 damaged.
   - The second pass: Opus reviews `rescue-newer.tsv`, writes `rescue-reviewed-2.txt`, then runs
     `python3 -I rescue.py --newer --go`. Run `--newer` again just before the session ends.
3. The rehearsal of `verify.py --mode after` on the unmoved tree (section 8). Done; attach its output to the freeze
   entry.
4. A copy of the record off this laptop, or inside Time Machine (Fable, finding 1; decision 6). Left to Opus and
   Eugene.
5. Eugene's decisions (section 12).

**Step 2: the freeze.**
1. **Preconditions** (extended after Fable, finding 18). All must hold:
   - **No agent is running, and no other Claude session is open on the project.** Step 3's `git switch -c` happens in
     the worktree every lab agent uses, so a stray commit would land on the consolidation branch.
   - **No job is running.** `ps -ax | grep -E "modal (run|deploy)|-m scout|pytest"` shows nothing: scout's first sweep
     and the p1b run (`modal_p1b.py`) have finished, and their results are on disk.
   - **No anatta app is running.** `~/prj/matlab/.venv/bin/modal app list` (read-only) shows none. Never touch
     matlab-*.
   - **anatta-lab is clean.** `git -C /Users/eugene/prj/align/anatta-lab status --porcelain` shows only the
     consolidation's own files and the four untracked Gate 0 files, which stay untracked by design (section 6).
     - The consolidation's own files are this design and MAP.md.
     - Today it also shows the bench edits and specs, `modal_p1b.py` and an untracked `results/fishery/lie_smoke3/`.
       Their owners commit them first.
   - **lab is committed and pushed.** `git -C anatta-lab log -1 --format=%h private/lab` equals `lab`. Today both are
     d1cfdd5.
   - **The site is rebuilt** if `ideas.md` changed after its last build. That is true now (section 8).
   - **The backup is confirmed** (step 1b, item 4).
2. **Note the time:** `date -u`.
3. **Tag every branch tip.** Today there are 16 branches:

       D=$(date -u +%Y-%m-%d)
       git -C /Users/eugene/prj/align/anatta for-each-ref --format='%(refname:short)' refs/heads |
         while read -r b; do git -C /Users/eugene/prj/align/anatta tag "consolidation-before-$D/$b" "$b"; done

4. **The 10 graded and library tags** (full paths and `read -r`: Fable, finding 19):

       while IFS=$'\t' read -r t c b e; do
         [ "$t" = tag ] || git -C /Users/eugene/prj/align/anatta tag -a "$t" "$c" -m "$e"
       done < /Users/eugene/prj/align/consolidation-2026-11/tags.tsv

5. **The hook extension and the backup push** (section 5), if Eugene approved them.
6. **A ledger entry, before the baseline.** The draft's steps 6 and 7 are swapped (Fable, finding 13).
   - The entry gives the freeze time, the tags, the rehearsal's output and the manifest's row count.
   - Commit it and push lab to private.
   - **From here until step 4's after inventory is taken, nobody writes to the ledger.** NOTES.md's `edit` row would
     accept appends, but the rule keeps the comparison simple.
7. **Re-take the baseline:**
   - `python3 -I inventory.py --out inventory-before.json`. The patch is written. It:
     - skips worktrees nested in outside-git folders;
     - keeps citations of whole places;
     - knows `results/`, `data/` and the archive;
     - never reads `data/`, the rescued scratchpad or premove/explore-data beyond hashing.
   - `python3 -I verify.py --mode before` writes `verify-before.json`.
   - `python3 -I make_move_map.py --refs-only --inventory inventory-before.json` writes `refs-by-move-freeze.tsv`.

**Step 3: execute.** One agent; the freeze holds.
1. **Lab branch:**
   - `git -C anatta-lab switch -c consolidate/2026-11`.
   - `git mv` the 12 plan documents, and append their names to `plans/archive/README.md` (edit row M0971).
   - Add the two declared-pilot shims: `ln -s archive/<name> lab/plans/<name>`.
   - Copy the 11 to-git files, after reading each (finding 5).
   - Write MAP.md (final), `make_index.py` and INDEX.md.
   - Apply the 4 rewrites through the rewrite script, which writes `rewrites-applied.tsv`.
   - Commit with pathspecs, never `git add -A`.
   - *Rollback before the commit:* `git -C anatta-lab switch lab` keeps untracked files.
     - First run `git restore --staged` on any staged moves.
     - Then remove by hand the two declared shims and the 11 copies. Their originals are in the archive.
   - *Rollback after the commit:* `git switch lab` takes the branch's files out of the worktree, and the branch keeps
     them.
2. **Outside git: copy, then switch, then shim,** as one script (Fable, finding 4):

       bash /Users/eugene/prj/align/consolidation-2026-11/step3_2_outside_git.sh --go

   - **How it fails.** It runs under `set -euo pipefail` and stops at the first failure, printing the state of all four
     names.
   - **Pre-checks:**
     - the four folders are real folders;
     - `results/`, `data/` and the four premove targets don't exist;
     - no process has a file open under the four folders (`lsof +D`).
   - **The copies.** It copies with `cp -pR`, then checks every copy entry by entry in Python: the same paths, sha256s
     and symlink texts. It doesn't use rsync: macOS ships openrsync, and the Python check is exact both ways.
   - **The switch.** Only if all four copies match does it switch:
     - `mv` each original into `archive-2026-11/premove/`;
     - check that the old name is gone;
     - then `ln -sn`.

     At the end it checks that each link's realpath is its new home.
   - **Run so far only on dummy trees** in `consolidation-2026-11/tmp/selftest/step32/`:
     - the normal path;
     - a `mv` that fails (premove not writable): it stopped before any link and left all four folders as they were;
     - a second run, refused;
     - a run without `--go`, refused.
   - **The premove copies stay, byte-identical,** until Eugene says they can go (map rows M0966-M0969).
   - *Rollback, per folder:* `rm <name>` (the symlink this step made; no data), then
     `mv archive-2026-11/premove/<name> <name>`. The copies under `results/` and `data/` stay until Eugene decides. The
     script refuses a second run while they exist.
   - *Rollback after a failure during the copies* (before any switch): the four originals are untouched. Remove only
     the partial copies under `results/` and `data/` (they are copies; the originals hold the data), check the four
     originals are still folders, then rerun. The dummy-tree runs did not cover a failing `ln -sn`; its state is
     printed and the per-folder rollback above covers it.
3. **Calibration:**
   - `git -C anatta branch calib/all calib/instruments`
   - `git -C anatta worktree add /Users/eugene/prj/align/anatta-calib calib/all`
   - The six `git merge --no-ff` in the order of section 5.
   - Resolve THIRD_PARTY.md as a union.
   - Run `pytest -p no:cacheprovider -q tests calibration`.
   - Then run `python3 -I verify.py --mode after --only calibmerge` (section 5) and keep its output for the step-4
     ledger entry, plus the by-hand diff of THIRD_PARTY.md, pyproject.toml and `src/`.
   - *Rollback:* `git -C anatta worktree remove /Users/eugene/prj/align/anatta-calib && git -C anatta branch -D
     calib/all`.
     - This deletes only what this step made a few minutes earlier. No instrument branch or tag changes, and the merge
       commits are new objects.
     - `worktree remove` refuses if untracked files appear. Look before adding `--force`.
4. **The four worktrees,** each in turn, with the K1 and C6 shims after:
   - `git -C anatta worktree move <wt> /Users/eugene/prj/align/archive-2026-11/worktrees/<wt>`
   - `git -C anatta worktree lock --reason "consolidation-before-<D>/<branch>: archived checkout at its branch tip;
     graded code is calib/<ID>-graded" <new path>`
   - `chmod -R a-w <new path>`

   Then: `cd /Users/eugene/prj/align && ln -s archive-2026-11/worktrees/anatta-calib-k1 anatta-calib-k1 && ln -s
   anatta-calib anatta-calib-c6`.

   - **The archive README says which checkouts are not their graded commits** (Fable, finding 11). The line to write:

     > The four checkouts are at their branch tips. C3's (f878d5e) and C4's (1711324) tips are their graded commits.
     > K1's tip 83067d7 differs from the graded 237841e in `calibration/prh_k1/README.md` and
     > `calibration/prh_k1/tests/test_k1_app.py` only; its graded code files are identical. C7's tip 9a9f7bc differs
     > from the graded 02db868 in `calibration/relearning/{app.py, card.py, config.yaml, README.md, relearn.py}`,
     > `tests/test_relearning.py`, `tests/test_relearning_real.py` and `THIRD_PARTY.md`, and adds
     > `tests/fixtures/relearning/scorer_equivalence.json`. C7's graded code is read only with
     > `git show calib/C7-graded:<path>`, in private: it holds the unlicensed port.

     These lists come from `git log --name-only` over each range, with each file's blob compared at both commits.
   - *Rollback, per worktree:*
     - first remove the K1 and C6 shims (`rm anatta-calib-k1 anatta-calib-c6`; links only);
     - then `chmod -R u+w <new path> && git -C anatta worktree unlock <new path> && git -C anatta worktree move
       <new path> /Users/eugene/prj/align/<wt>`.
5. **The pin** (section 4, option B).
   - *Unpin, to roll back or to move the pin later* (Fable, finding 20):
     `chmod -R u+w /Users/eugene/prj/align/anatta-dev && git -C /Users/eugene/prj/align/anatta worktree unlock
     /Users/eugene/prj/align/anatta-dev && git -C /Users/eugene/prj/align/anatta-dev switch dev/ministral`.
   - A new pin is a new `lib/<name>-<sha>` tag, then the same three steps forward, and a ledger line.

**Step 4: verify.**
- `inventory.py --out inventory-after.json`, then `verify.py --mode after`. Every check must pass:
  - every before sha256 is found at its mapped path, or accounted for by a rewrite, an edit rule or a copy;
  - no file is unexplained, and none is left at an old path;
  - every shim resolves;
  - calib/all matches the branch tips, with every difference explained;
  - no suite gets worse, and no test id that passed before fails;
  - the tally is unchanged, and the page is byte-identical;
  - the re-grades are identical;
  - every last-30 ledger path exists through the map;
  - the rescue has nothing pending.
- **The private-dependent checks** (decision 7):
  - run the 7 h4lite scripts that import from the private folder, by their documented commands;
  - confirm that `lab/experiments/recognition/scripts/harness_v2.py` is tracked on lab. Graded fishery and coding
    results import it, and no tag pins it otherwise.
- Only now may the ledger take entries again.

**Step 5: the second cold Fable check:** the inventory diff, the re-grades, and ten claims traced end to end through
MAP.md by an agent that has never seen the lab.

**Step 6: land.**
1. `git -C anatta-lab switch lab && git merge --no-ff consolidate/2026-11`.
2. `git fetch private` and check that private/lab is an ancestor.
3. `git push private lab`.
4. A ledger entry with the counts.
5. A note to Eugene with what moved and where.

**Risks and guards:**

| risk | guard |
|---|---|
| a file lost | copies before switching; sha256 for every file before and after; no delete in the map (checked) |
| a graded result no longer reproducible | annotated tags at graded commits, checked against the cards' code hashes; CPU re-grades must be identical |
| the library drifts | the pin is detached, locked and read-only; pins check |
| C7's port reaches main | the hard rule; the hook extension; calib/all is never pushed to origin |
| local-only branches lost | backup push, or a bundle kept outside prj/align (decision 6) |
| the record lost with this disk | decision 6: Time Machine excludes prj/align today |
| a running job writes mid-move | the freeze preconditions; the step-3.2 script's `lsof` check; the premove copies make it recoverable |
| a failed `mv` leaves a split brain | the step-3.2 script: no link until every copy matches, `set -e`, the old name checked before `ln -sn`, realpath at the end |
| a shim dangles | the shims check (realpath, `exists`) |
| a merge drops one branch's change | the calibmerge check |
| the after check can't pass as ordered | the ledger entry before the baseline; edit and copy rows; the rescue manifest; rehearsed (section 8) |
| a private path touched | verify's guard hides private paths; inventory and rescue prune by name; Opus reviews the scratchpad |
| an agent edits p1, bench or scout during the moves | the freeze; the baseline is re-taken at the freeze, not today |
| over-engineering | nothing moves that the map doesn't need; no code folder is renamed |

## 12. Decisions for Eugene

1. **Extend the pre-push hook** (section 5), with Fable's version. It is configuration, so it needs your OK.
   - The hook tests both ref sides, and adds the `calibration/` path rule and the ancestry rule.
   - Also: `git branch --unset-upstream calib/instruments`.
   - Also a ruleset on github.com/ejhong/anatta (Settings → Rules → Rulesets). It is the only guard that holds when
     the laptop's hook is skipped with `--no-verify`. It blocks creating or updating:
     - branches `calib/**` and `lab**`;
     - tags `calib/**`, `lib/**` and `consolidation-before-**`;
     - optionally, to match the hook, branches `consolidate/**` and `dev/**`.
2. **The backup:** push calib/*, dev/ministral and the tags to the private remote, or keep a bundle outside
   `prj/align` (section 5; decision 6).
3. **`lab/explore/PROGRAM.md` line 47** (you edit that file): change the results path, or leave it, since the shim
   works.
4. **`inputs/` → `data/essay`:** yes, or leave it at the top.
5. **When to delete:** the premove copies and the archived calibration checkouts stay until you say otherwise. Fable
   recommends not deleting anything during this consolidation, and revisiting after the second Fable check and after
   decision 6 has made a second copy.
6. **Time Machine excludes `/Users/eugene/prj/align`** (Fable checked with `tmutil`; finding 1).
   - So the only copies of the graded cards, the explore results, the data, the essay PDF, the rescued scratchpad and
     the branch bundle are on this disk, with no backup.
   - Choose one of three:
     - include `prj/align` in Time Machine, leaving out `*/.venv` and the git stores if size is the reason;
     - keep a Time-Machine-included mirror of `results/`, `data/`, `archive-2026-11/` and the bundle;
     - push the branches to the private remote (decision 2) and copy the outside-git folders somewhere backed up.
   - This is the largest risk to the record, and it comes before the freeze.
7. **Who runs the private-dependent checks** (step 4).
   - Opus proposes to run them itself, in the main session, with your permission and no agents. That means the 7
     h4lite scripts that import from the private folder, by their documented commands, and the check that
     `harness_v2.py` is tracked on lab.
   - Say yes, or name someone else, and when.
8. **The four archived checkouts:** keep all four at their branch tips, locked and read-only (this design). Or drop
   the K1 one once p1's owner points `K1_DIR` at `anatta-calib/calibration/prh_k1`; the K1 shim then goes too.
   Either is fine; say which.

**Limits of this design:**
- The draft inventory is from 18:04 UTC and the rehearsal inventory from 20:18. Agents have committed since, and
  explore-results has grown about fivefold. The freeze re-takes both.
- The merge forecast is a forecast. The calibmerge comparison hasn't run: it needs calib/all and `git diff`.
- The step-3.2 script has run only on dummy trees.
- C4's two failing app tests weren't diagnosed. They are now a named allowance in the suite comparison, and nothing
  else may fail.
- The second scratchpad pass waits for Opus's review. Findings 1 and 5 wait for Opus and Eugene.
- The private-dependent scripts and `harness_v2.py` can't be checked from outside the private folder.
- Citations built in code are found only by running the code.

**Plain reading.** Nothing has moved yet. Fable's check found the plan sound and the safety net around it unfinished.
- That net is now built.
  - The checker has the checks it lacked.
  - The risky step is one script that stops before it can split a folder in two.
  - The checker was rehearsed on the unmoved tree. It reported exactly the 14,163 files that will move, and nothing
    else.
- What remains before the freeze isn't code:
  - a copy of the record that isn't on this disk alone;
  - Eugene's decisions;
  - the second scratchpad pass;
  - the running work finishing.
- For the project, the payoff is unchanged: a reader can go from any public claim to its code, data and ledger entry
  through one page, old paths keep working, and the calibration history stays private where it must.
