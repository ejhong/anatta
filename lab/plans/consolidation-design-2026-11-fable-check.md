# Cold check of the consolidation design (1 Nov 2026)

Claude Fable 5.1, a fresh read-only reviewer, wrote this on 1 Nov 2026 at about 20:05 UTC for Claude Opus 5.5 and Eugene.
Everything below was read from the files and from read-only git commands; nothing was run, edited, moved or pushed. The
private folders (the account folder, `lab/experiments/recognition/`, `prj/align/recognition`) were never listed,
opened or grepped. No completion, benchmark item, key or token was opened or printed; the cards were read only for keys
whose values are sha256 strings.

## Verdict: NOT READY

The architecture is right: two roots outside git, shims at the old paths, one private calibration branch with annotated
tags at the graded commits, the library pinned by tag in place, tracked results left where the ledger cites them. I
re-derived the ten tags against the cards myself and they are all correct. But the design cannot be executed as written
without breaking its own guarantees in four places, and the backup it reports does not exist off this laptop:

1. **The only copies of the graded calibration cards (`calib-results`), the explore results, the rescued scratchpad and
   the calib/* history are on a path Time Machine excludes.** The bundle Opus made at 19:3x UTC sits inside that same
   excluded path (`archive-2026-11/branches-2026-11-01.bundle`), so it backs up nothing. One disk failure before the
   moves, or during them, loses the record the project rests on.
2. **The design's own "re-check everything" command (`make_move_map.py`) rewrites `move-map.tsv` from scratch and silently
   undoes Opus's privacy review:** `smoke_wim`, dropped because it holds a recognition-derived direction, would go back on
   the keep list, and the next `rescue.py --go` would copy it into the archive.
3. **`verify.py --mode after` has never run and, as the steps are ordered, cannot pass:** the freeze ledger entry is written
   after the baseline (NOTES.md would read "changed"), MAP.md is mapped as "new" but already exists, the premove copies and
   the bundle are "unexplained new files", the rescue check compares against a stale draft that already shows 6
   mismatches, and the plans/archive README append has no row. Every one of these fails in the safe direction, but the
   operator would meet them mid-move, which is exactly when verify.py must not be edited.
4. **The proposed pre-push hook closes the named-ref paths and leaves the unnamed ones open** (`git push origin HEAD`,
   `HEAD:refs/heads/x`, `<sha>:...`): five of the six instrument branches, calib/instruments (whose upstream is
   `origin/main`) and dev/ministral would still reach the public repo that way.

All four are short fixes. After them, and after one rehearsal of the after-mode checker on an unmoved tree, the design
is ready. I would want to see the fix diff and the rehearsal output (a short re-check, not a full review) before the
freeze.

## Findings

Severity: BLOCKING (fix before any move), HIGH (fix before the freeze), MEDIUM (fix during step 3 or document), LOW.

### A. Safety

**1. BLOCKING. No copy of the record exists off this laptop; the bundle is inside the excluded path.**
- Evidence: `tmutil isexcluded` prints `[Excluded] /Users/eugene/prj/align`, `[Excluded] .../archive-2026-11`,
  `[Excluded] .../calib-results`, `[Included] /Users/eugene/prj`. `git bundle verify` of
  `archive-2026-11/branches-2026-11-01.bundle` says "complete history" with the 7 calib/* heads and dev/ministral, but the
  file is at an excluded path. `git for-each-ref` shows the 7 calib/* branches with no upstream; `private/lab` = cb85c62
  contains ee89587 (dev/ministral is safe) but no calib/* commit. `du -sm`: calib-results 31 MB, explore-results 196 MB
  (grown from the design's 68 MB), the rescued scratchpad 76 MB.
- Why it matters: the plan's "nothing is lost" is a sha256 argument about moves; it assumes the disk survives. Today the
  graded cards that every PASS/FAIL cites have one copy, excluded from backup.
- Fix (today, before anything else; needs Eugene for the TM change): copy the bundle and a `cp -pR` of `calib-results`,
  `explore-results`, `explore-data`, `inputs` and `archive-2026-11` to a Time-Machine-included path (for example
  `~/prj/align-backup-2026-11-01/`), and push calib/* and the tags to `private` once the hook is fixed. Then decide
  whether `/Users/eugene/prj/align` stays excluded (if the exclusion exists for the venvs and git stores, exclude those
  subfolders instead and include the rest). Add "an off-laptop or TM-included copy exists" to the freeze preconditions.

**2. BLOCKING. `make_move_map.py` regenerates `move-map.tsv` and reverts the privacy review.**
- Evidence: `make_move_map.py:384-387` opens `move-map.tsv` for writing unconditionally; `LEAVE` (lines 48-68) has no
  `smoke_wim` entry; `diff move-map.before-review.tsv move-map.tsv` shows the only review edit is M0962 `rescue-dir ->
  leave` ("holds a copy of a recognition-derived direction"). The design, lines 20-23, tells the reader to run
  `python3 -I make_move_map.py && python3 -I verify.py --mode before` "to re-check everything". `rescue.py` and
  `verify.rescue_pairs` (verify.py:346-363) copy every `rescue-dir` row, so after a regeneration `rescue.py --go` would
  copy `smoke_wim` into the archive. (Today the archive has no `smoke_wim`; checked by `ls`.)
- Fix: add `"smoke_wim": "a copy of a recognition-derived direction; the original stays in the private folder (Opus review,
  1 Nov)"` to `LEAVE`, make the script refuse to overwrite an existing `move-map.tsv` without `--force`, and change the
  design's re-check line to `verify.py` only. Re-run the diff against `move-map.before-review.tsv` afterwards to show the
  only change is still M0962.

**3. HIGH. The hook extension matches local ref names, which `HEAD` and sha pushes do not carry.**
- Evidence: the existing hook (`anatta/.git/hooks/pre-push`) and the proposal (design lines 374-382) test `$lref` only.
  For `git push origin HEAD` git passes `HEAD` as the local ref and resolves the remote side to `refs/heads/<branch>`;
  for `HEAD:refs/heads/x` or `<sha>:refs/heads/x` the local ref is `HEAD` or the sha. The ancestry test
  `merge-base --is-ancestor d0aac0b` covers only calib/relearning and calib/all (`git branch --contains d0aac0b` lists
  calib/relearning alone). The other five instrument branches all carry `calibration/` (`git rev-list -n1 <b> --
  calibration/` is non-empty for model-check, koan-scorer, relearning, persona-extractor, refusal-direction, prh-k1 and
  empty for main, origin/main, lab, dev/ministral, calib/instruments). `calib/instruments` has `branch.calib/instruments.remote
  origin` and `.merge refs/heads/main`; `push.default simple` refuses a bare `git push` there (names differ) but not
  `git push origin HEAD` or `git push -u origin calib/instruments`. `--mirror` would also send `refs/heads/dev/ministral`
  (no lab/ history, no calibration/ history). `merge-base --is-ancestor` does accept an annotated tag object (checked
  with v0.2.0), so the ancestry line works for `--tags`.
- Fix: in the loop, test both sides and add a path rule like the existing `lab/` one:
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
  The `calibration/` path rule must be relaxed deliberately when the curated PR (ideas 49) is ready; write that into the
  PR plan. Also `git branch --unset-upstream calib/instruments` (a config change: Eugene's OK). The hook is shared by
  every worktree (`git rev-parse --git-common-dir` from anatta-calib-c7 and anatta-lab both give `anatta/.git`;
  `core.hooksPath` is unset), so one file covers all pushes, but `--no-verify` still bypasses it: ask Eugene to add a
  ruleset on github.com/ejhong/anatta that blocks creation of `refs/heads/calib/**`, `refs/heads/lab**`,
  `refs/tags/calib/**`, `refs/tags/lib/**` and `refs/tags/consolidation-before-**`. That is the only guard that holds
  when the laptop-side hook is skipped.

**4. HIGH. Step 3.2's `ln -s` lines can create a split brain if a `mv` fails.**
- Evidence: design lines 627-634 run `mv` in a `for` loop and then four `ln -s` on separate lines, no `set -e`, no `&&`.
  If `mv calib-results archive-2026-11/premove/` fails (a file open by a process, a permission error, premove already
  holding a folder of that name), `ln -s results/calib calib-results` then creates the link *inside* the surviving
  folder as `calib-results/calib`; the apps keep writing to the old folder while `results/calib` holds a stale copy, and
  nothing in verify.py notices (finding 12).
- Fix: one script with `set -euo pipefail`; before each `ln -s`, `[ ! -e "$old" ] || exit 1`; use `ln -sn`; and after the
  four links, `readlink -f` each and compare with the realpath of the target.

**5. MEDIUM. The 11 `to-git` files go into a repo without a recorded per-file read.**
- Evidence: design section 10 says "Into git after Opus reviews them (11 files)"; `rescue-reviewed.txt` records a key
  pattern scan over the keep list and a "completion" field scan, and says benchmark items were considered for fixtures
  and the C7 folders. It does not say the seven check reports and four facts files were each read for quoted items.
  Three of them are C3 and K1 checks (C3 reads JBB and MMLU items).
- Fix: in step 3.1, read each of the 11 before `cp -n`, and record "read, no items, no keys" per file in the ledger entry.
  The lab is private, but the rule is "benchmark items and keys stay out of every repo".

**6. MEDIUM. Nothing checks that the merged `calib/all` tree equals the branch tips it was merged from.**
- Evidence: `check_tags` (verify.py:573-622) compares the cards with commits, never with the `anatta-calib` worktree;
  `compare_suites` (761-789) compares test ids by name, so a merge that resolved a shared test file the wrong way still
  "passes" if the id survives. The forecast (design 357-361) is a forecast.
- Fix: after the six merges, for each instrument run `git -C anatta diff --stat calib/<branch> calib/all --
  calibration/<dir>/ tests/` and require it empty, except the known THIRD_PARTY.md union; add it to verify's after mode
  as a check and to the ledger entry.

**7. LOW. Nothing private is moved, read or exposed by the design.** The keep rows name the two private folders and the two
symlinks by path only; `check_map` (verify.py:228-254) refuses any other row touching them; the guard hides them from lab
code (guard/child.py:129-132, audit hook at 432; the site build's two stats on them were answered "no such file",
verify-before.json). The rescue copied 2,479 files and no `smoke_wim`. The backup push would put the port's history on
GitHub privately: that is decision 2 and is correctly left to Eugene. The C7 tip (9a9f7bc) no longer holds the port
(`a29c585 "no Łucki et al. code here"`; THIRD_PARTY.md at the tip says so), so "fresh commits of the current files" is a
sound rule; the port is reachable only through history and the `calib/C7-graded` tag, both private.

### B. The record stays intact

**8. PASS, independently. The ten tags are right.** My own hash of every file each card or plan names, at the tagged
commit (`git show <commit>:<path>`): C1 44760f7 21/21; C4 1711324 13/13; C6 12e831e 13/13; C7 02db868 17/17 (at the tip
9a9f7bc 6 of them differ: app.py, card.py, config.yaml, relearn.py, which is why the graded tag is not the tip); K1
237841e 3/3 (and 3/3 at the tip 83067d7: the diff is `README.md` and `tests/test_k1_app.py` only); C3 f878d5e 12/12 from
plan.json and 40/40 from card.json. `git -C anatta-lab show f363ffb:lab/experiments/h4lite/cal_judges.py | shasum -a
256` starts 657bf09ea1151c39, as the ledger says. `lib/ministral-ee89587`: ee89587 is an ancestor of lab, `git log
ee89587..lab -- src/` is empty, and anatta-dev is clean there.

**9. MEDIUM. "682 citations accounted for" overstates what was checked; the real load-bearing set is 21 absolute paths,
and the scanner missed at least one.**
- Evidence: `refs-by-move.tsv` by strength: 226 bare-name matches (the design itself calls these "a first pass only,
  flagged"), 426 "same relative path in a sibling worktree" (a repo's files naming their own `src/anatta/...`), 9
  relative, 21 absolute. Of the 68 code citations "kept working by the map" (that is, by nothing), all are bare-name or
  sibling-relative noise for the C3/C4/C7 worktree moves, e.g. `anatta-calib-c3/.../app.py` listed as citing the C4
  worktree. The 21 absolute ones all resolve through a shim or the one rewrite, and I confirmed the citing lines:
  C7's and K1's `app.py` (calib-results), 5 bench specs + `modal_p1.py`, `p1_analysis.py`, `spec.yaml`
  (explore-results), `first-sweep.yaml`, `smoke-harm.yaml` (explore-data), `p1_core.py:21` K1_DIR (anatta-calib-k1),
  SIGHTINGS.md, ideas.md, index.html, NOTES.md. My own grep found more absolute citers of explore-results than the
  table lists (`lab/explore/bench/tests/test_controls.py`, `lead1-split*.yaml`, `lead6-split*.yaml`, `smoke-controls.yaml`,
  and `modal_p1b.py:51-63`, all newer than the 18:04 inventory) and all are covered by the same shim. The scanner missed
  NOTES.md line 7365, `from /Users/eugene/prj/align/anatta-calib-c3.` (trailing period), which `refs-by-move.tsv`
  records only as a bare-name match for M0016; the design's "No file names a moved calibration worktree by absolute path"
  (lines 232-234) is therefore false for the ledger (5 NOTES lines name c3/c4/c7, one absolutely) and for
  `lab/ledger/checks/2026-11-01-c{3,4}-grade-fable.md`, `lab/plans/calibration-c{3,4}-fable-code-check.md`, MAP.md
  and the design itself. These are docs, so nothing breaks, but see finding 10.
- Fix: say in section 3 that the guarantee rests on the 21 absolute citations plus the imports, suites and re-grades,
  not on 682; strip trailing `.,:;` from tokens in `candidate_tokens`; regenerate the refs table from the freeze
  inventory.

**10. MEDIUM. MAP.md's "old paths that still work" table omits `anatta-calib-c3`, `-c4` and `-c7`.**
- Evidence: MAP.md lines 75-90 list calib-results, explore-results, explore-data, inputs, anatta-calib-k1, "c1/c6/
  relearning (removed earlier)", the top-level symlinks, the declared shims and lab/h4lite. The three moved worktrees
  appear only in the "where" table (line 42). A reader following NOTES line 7365 or the C3 grade check report has no row.
- Fix: add `anatta-calib-c3/, -c4/, -c7/` -> `archive-2026-11/worktrees/<same>` (frozen, read-only) and the lines for
  `lab/plans/calibration-c{2,3,4}-*.md` -> `plans/archive/`.

**11. MEDIUM. "Frozen at its graded tag" is said three ways and is wrong for K1 and C7.**
- Evidence: design line 128 "frozen checkouts at their graded tags"; `make_move_map.py:218-226` "a frozen checkout of its
  tagged tip"; MAP.md line 42 "each at its graded tip"; step 3.4 locks with reason "frozen at <tag>" without a checkout.
  The worktrees are on their branches at the tips: K1 at 83067d7 (graded 237841e), C7 at 9a9f7bc (graded 02db868,
  which must not sit in a checkout anyway). p1 imports K1's code from the K1 checkout (shim M0017), and that code is
  identical at tip and tag (finding 8), so nothing breaks.
- Fix: decide and say it once: the four checkouts stay at their branch tips, locked and read-only, with the
  `consolidation-before-<date>/<branch>` tag as the lock reason; graded code is reached through `git show
  calib/<ID>-graded:<path>`. Record in `archive-2026-11/README.md` that K1's and C7's checkouts are not the graded
  commit and which files differ.

**12. LOW. Sampled rows.** 55 rows across every kind, checked against the disk by existence and sha256 (never content):
all consistent with the map. Moves M0001, M0003, M0005, M0007: sources are real folders, targets absent. Shims M0002,
M0004, M0006, M0008, M0017: the old path exists now and becomes the link after the `mv`; M0018 (c6), M0019, M0020: the
link path is free and the targets exist (`lab/plans/archive/declared-release-pilot*.md` are tracked, and the archived
design still hashes 00f0cb81…, as `declared_core.py:1` registers). Worktree moves M0013-M0016: sources exist, targets
absent, `.git/worktrees/*` none locked. M0012, M0021, M0022-M0024, git-mv M0025, M0028, M0031, M0034, M0036 (targets
absent, no name clash with the 28 files already in plans/archive), rewrites M0040-M0043 (the cited lines are what the map
says: `scout/paths.py:23 DEFAULT_RESULTS_ROOT`, `explore/README.md:12,34`, `PROGRAM.md:47`, `lab/README.md:58`), keeps
M0044, M0048, M0053, M0056, M0057, M0058, to-git M0149, M0239, M0433, M0743 (archive copies exist, lab targets absent),
rescue M0059, M0215, M0375, M0530, M0680, M0830 (copy exists, sha256 equal), rescue-dir M0839, M0894, M0959 (file counts
equal), rescue-dup M0084, M0447, M0803 (twin exists, sha256 equal), leave M0846, M0916, M0962 (not copied).

### C. verify.py

The guard is well built and the before self-test is credible: I re-derived the tags (8), and the regrades, site tally
and ledger checks test what the design says. The problems are in after mode, which has never run.

**13. BLOCKING. After mode cannot pass with the steps as ordered, and the failures would land mid-move.**
- Evidence, each against the code:
  - NOTES.md: step 2.6 takes the baseline, step 2.7 writes the freeze entry (design 610-614). `check_hashes` after mode
    (verify.py:292-320) accepts a changed file only if `rewrites-applied.tsv` reverses it, so NOTES.md reads "changed"
    and the check fails. Same for `lab/plans/archive/README.md` ("append the names", M0025-M0036's refs column; no
    rewrite row) and for MAP.md and the design doc if edited after the freeze: M0037 maps MAP.md as `new`, but it is
    already committed at a82ca71.
  - `premove/` and the bundle: `inventory.py:406-418` makes every non-worktree folder at the top of prj/align an
    "outside-git" place, so `archive-2026-11/premove/*` (about 2,700 files) and `branches-2026-11-01.bundle` enter
    `inv_after`; `check_hashes` lines 315-318 then list them as `unexplained_new` and fail.
  - Rescue: `rescue_pairs` is always fed `inventory-before-draft.json` (verify.py:1157-1159, 346-363) and `check_rescue`
    (366-380) fails on any mismatch; `verify-before-partial.json` (19:34 UTC) already shows `mismatched: 6` and status
    `fail` for the six logs that grew after 18:04. Files created in the scratchpad after 18:04 (this report among them)
    are invisible to it; `rescue-reviewed.txt` says a second pass is owed.
  - Nested worktrees: `archive-2026-11/worktrees/*` would be walked twice (as worktree places from `git worktree list`
    and inside the archive place); the design knows (step 2.6 "patch it first") but the patch is unwritten.
- Fix, before the freeze: (a) swap steps 2.6 and 2.7 so the freeze entry precedes the baseline, and forbid ledger writes
  between baseline and after-inventory (the landing entry comes after); (b) add a map kind `edit` for files the
  consolidation changes by hand (NOTES.md as append-only: the before bytes must be a prefix of the after bytes;
  plans/archive/README.md; MAP.md; the design doc), checked by `check_hashes`; (c) add map rows for `premove/` (kind
  `copy`, checked byte-equal to the moved folder) and the bundle (kind `new`); (d) make `rescue.py --go` write a
  manifest (source, destination, sha256 at copy time) and have `check_rescue` read it, then re-run rescue for files
  newer than the draft; (e) write and commit the inventory.py patch now; (f) rehearse: take an inventory of the current
  tree and run `verify.py --mode after --inventory-after <that same file>` against it. On an unmoved tree every moved
  file must come back "missing at its mapped path" and nothing else; that proves the code path runs and the map is
  parsed as intended. Attach the output to the freeze ledger entry.

**14. HIGH. Shims are never checked to resolve.**
- Evidence: no check reads a `shim` row. `check_ledger` after mode uses `os.path.lexists` (verify.py:1120), which is true
  for a dangling symlink. The K1 and declared shims get exercised by p1's tests and `test_declared.py`; the calib-results,
  explore-results, explore-data, inputs and c6 shims are exercised only at Modal run time (C7's and K1's `app.py`) or
  never (c6_samples.py has no test).
- Fix: add a `shims` check: for every `shim` row, the old path is a symlink, `os.path.realpath(old)` equals the
  realpath of the new path, and it exists; use `exists`, not `lexists`, in the ledger after-check.

**15. MEDIUM. Twenty "script" suites record no test ids, so a regression in them is invisible.**
- Evidence: verify-before.json: the 19 h4lite `test_*.py` scripts and `test_declared.py` have `ids_recorded=0`;
  `compare_suites` (761-789) compares passed ids and flags only "ok before, not ok after". A suite that was
  guard-limited before (11 of those 20) and gains a real failure after is not flagged, and `calib` is grouped, so C4's
  two real failures before make `was_ok["calib"]` false for the whole group.
- Fix: compare per-suite `verdict`, `exit_code` and `failed` count, and fail on any worsening; keep C4's two failing
  ids as a named allowance rather than disabling the ok-check for the group.

**16. LOW. `check_tags` "confirmed" ignores missing files** (verify.py:602: `best["match"] and not best["mismatch"]`). All
cards are 0 missing today, so it is harmless, but one matched file out of twelve would still read "confirmed". Require
`missing == []`.

**17. LOW. `check_map`'s before mode exempts anatta-calib-c3 from the source-exists test** (verify.py:245-247), a
leftover from when C3 was mid-run. Remove it.

### D. Order and rollback

**18. HIGH. Two jobs are running now, and the preconditions miss two states.**
- Evidence: `ps` at about 20:00 UTC shows `python -m scout run scout/specs/first-sweep.yaml` (pid 10746, 1 h 28 min) and
  `modal run modal_p1b.py::full` (pid 37349, started minutes ago; `modal_p1b.py:51-63` writes under
  `explore-results/2026-11-01/split-reruns/p1b` and reads the `anatta-calib-prh-k1` volume). `git status` in anatta-lab
  shows three modified bench files and nine untracked bench specs and tests. `explore-results/cache` and the
  `lead1-split`, `lead6-split-smoke` folders are being written. `lab` is one commit ahead of `private/lab` (a82ca71, the
  design and MAP.md, which the design says are uncommitted: lines 6-7 are stale).
- Fix: the preconditions in step 2.1 should add `git -C anatta-lab status --porcelain` empty (except the consolidation's
  own files), no other Claude session open on the project (step 3's `git switch -c` happens in the worktree every lab
  agent uses, so any stray commit lands on the consolidation branch), the p1b run finished and its results on disk, and
  `lab` pushed to `private` before the freeze tag.

**19. MEDIUM. Rollback is real at every step but two recipes are missing.**
- Step 3.2: replace each symlink with its premove folder (stated). Step 3.3: `git worktree remove anatta-calib && git
  branch -D calib/all` (new objects only; say so, since the plan says "never delete"). Step 3.4 and 3.5: a read-only,
  locked worktree cannot be moved or switched back until `chmod -R u+w` and `git worktree unlock`; write the recipe.
  Step 3.1: `git switch lab` leaves the branch; the two symlinks and 11 copies are untracked until committed and must
  be removed by hand on rollback. Tags are additive; `consolidation-before-*` tags make every tip recoverable.
- Also step 2.4's loop runs `git -C anatta` from a relative path and `tags.tsv` by bare name: run it from
  `/Users/eugene/prj/align` with the full path, and `read -r`.

### E. The library pin

**20. Agree with option B.** The reasoning holds and I checked its premises: ee89587 is in lab and in private/lab; no lab
commit touches `src/` since; anatta-dev is clean; 15 loaders name the path and all keep working untouched. Merging
(option A) would add no code and would edit registered scripts and the provenance hashes `c6_samples.py` and
`declared_core.py` record. Two additions: write the unpin recipe (`chmod -R u+w anatta-dev && git -C anatta worktree
unlock anatta-dev && git -C anatta-dev switch dev/ministral`, then a new `lib/<name>-<sha>` tag and a ledger line), and
keep p1's `sys.path.append` (design item 4) on the owner's list, since under the anatta venv p1 still imports main's
library with its uncommitted koans edits.

### F. Decisions for Eugene

| decision | recommendation |
|---|---|
| 1. Extend the hook | Yes, with finding 3's version (both ref sides, the `calibration/` path rule, the ancestry rule), plus `git branch --unset-upstream calib/instruments`, plus a GitHub ruleset on ejhong/anatta blocking `calib/**`, `lab**` branches and `calib/**`, `lib/**`, `consolidation-before-**` tags. |
| 2. Backup | Push calib/* and the tags to `private` after the hook change. The port's history then sits in a private GitHub repo: keeping unlicensed code privately is the same act as keeping it on the laptop; publishing it is what is forbidden. If you would rather it never leave the laptop, the bundle must still leave `prj/align` (see the new decision 6). |
| 3. PROGRAM.md line 47 | Leave it; the shim keeps it true. Edit it when you next touch the file. |
| 4. inputs -> data/essay | Yes: one PDF, nothing cites it, one data root. |
| 5. When to delete | Not during this consolidation. Revisit after the second Fable check and after decision 6 has produced a second copy. |
| **6. New: Time Machine** | Include `/Users/eugene/prj/align` (excluding `*/.venv` and the git stores if size is the reason), or keep a TM-included mirror of `results/`, `data/`, `archive-2026-11/` and the bundle. This is the single largest risk to the record and it is not in section 12. |
| **7. New: the private-dependent checks** | Step 4 needs you (or an agent you admit to the private folder) to run the 7 h4lite scripts and confirm `recognition/scripts/harness_v2.py` is tracked on lab. Say who and when; I did not check it, by rule. |
| **8. New: the four frozen checkouts** | Keep them at their tips (finding 11), or drop the K1 one once p1's owner points K1_DIR at `anatta-calib/calibration/prh_k1`. Either is fine; say which. |

## Rows sampled

M0001, M0002, M0003, M0004, M0005, M0006, M0007, M0008, M0012, M0013, M0014, M0015, M0016, M0017, M0018, M0019, M0020,
M0021, M0022, M0023, M0024, M0025, M0028, M0031, M0034, M0036, M0040, M0041, M0042, M0043, M0044, M0048, M0053, M0056,
M0057, M0058, M0059, M0084, M0149, M0215, M0239, M0375, M0433, M0447, M0530, M0680, M0743, M0803, M0830, M0839, M0846,
M0894, M0916, M0959, M0962 (55 rows: 4 move, 8 shim, 4 worktree-move, 1 new-worktree, 1 pin, 3 resolve, 5 git-mv,
4 rewrite, 6 keep, 4 to-git, 6 rescue, 3 rescue-dir, 3 rescue-dup, 3 leave). Citations spot-checked: all 21 absolute
rows of refs-by-move.tsv, the 10 NOTES.md rows, and the 68 "code / map" rows (all bare-name or sibling-relative).

## What was not checked

The private folders and anything reached through them, including whether `harness_v2.py` is tracked. The contents of
any results file beyond sha256-valued keys. The C4 app-test failures. The Modal volumes. Whether the inventory's
citation scanner has other token gaps beyond trailing punctuation.

## Changes required before any move

1. An off-laptop or Time-Machine-included copy of calib-results, explore-results, explore-data, inputs, archive-2026-11
   and the bundle (finding 1); then decision 6.
2. `smoke_wim` into `LEAVE`, `make_move_map.py` refuses to overwrite, the design's re-check line changed (finding 2).
3. The hook as in finding 3, the calib/instruments upstream unset, and the GitHub ruleset requested (finding 3).
4. Step 3.2 as one `set -e` script with pre-checks and realpath verification (finding 4).
5. verify.py and the map made able to pass after mode: steps 2.6/2.7 swapped, an `edit` kind for NOTES.md (prefix
   rule), plans/archive/README.md, MAP.md and the design; rows for premove/ and the bundle; the rescue manifest; the
   inventory patch written; a shims check; `exists` in the ledger after-check; per-suite verdict comparison; then one
   rehearsal of `--mode after` on the unmoved tree, output attached to the freeze entry (findings 13-15).
6. Freeze preconditions extended: p1b and scout finished, anatta-lab clean, no other session, lab pushed, backup
   confirmed (finding 18). MAP.md's old-paths table and the "frozen at" wording fixed (findings 10, 11).

Then a short re-check of the diff and the rehearsal output, not a full review, before the freeze.

## Plain reading

The design moves the right things in the right way, and every graded result's code tag checks out; what is not ready is
the safety net around the move: the only copy of the record is on an unbacked-up path, the checker that proves nothing
was lost has never run in the mode that matters and would fail as the steps are ordered, and the push guard still has
an open door. Fix those six things, rehearse the checker once, and this can go ahead.

---

Written by a fresh Fable 5.1 agent, read-only, on 1 Nov 2026. Filed here unchanged by Opus 5.5.
