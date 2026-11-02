# Re-check of the consolidation design after the fixes (1 Nov 2026)

Claude Fable 5.1, a fresh read-only reviewer, wrote this on 1 Nov 2026 at about 20:45 UTC for Claude Opus 5.5 and
Eugene. Read-only: nothing was run, edited, moved or pushed; only reading commands and read-only git/gh calls. The
private folders were never listed, opened or grepped. No completion, benchmark item or key was opened or printed.
Checked against: the d1cfdd5..bcae280 diff of the design and MAP.md; the four `*.before-fable.py` diffs;
`step3_2_outside_git.sh`; `move-map.tsv` vs `move-map.before-review.tsv`; the rehearsal and partial reports; the
rescue manifest and both review files; the installed hook; the branch list on `private`; `ps`; `tmutil`; `gh api`.

## Verdict: READY FOR THE FREEZE, once the running work finishes and three non-code items are done

The six changes the first check required are made, and the rehearsal shows what it claims. What remains is not code:
the GitHub ruleset (none exists on ejhong/anatta today: `gh api repos/ejhong/anatta/rulesets` returns `[]`), a
second copy of the outside-git results if Eugene wants one (he has accepted the Time Machine exclusion), and the
preconditions themselves (scout's first sweep is still running at 2 h 16 min; bench edits and `modal_p1b.py` are
uncommitted in anatta-lab). Two small things to fix in passing are listed under question 4.

## 1. The twenty findings

| # | status | one line |
|---|---|---|
| 1 | handled by Eugene's decision; the git side is done | All 7 calib/* heads are on `private` at their local shas (ls-remote); `lab` = `private/lab` = bcae280. dev/ministral is in lab. The outside-git results (calib-results, explore-results, the rescued scratchpad, the bundle) still have one copy, on an excluded path, by Eugene's choice ahead of the Mac mini move. Say so in the freeze entry. |
| 2 | fixed | `smoke_wim` is in `LEAVE`; `make_move_map.py` refuses without `--force`; `--refs-only` leaves the map alone. Diff vs `move-map.before-review.tsv`: M0962 is the only review change; the rest are M0011, M0013-16 wording, M0037/M0058 to `edit`, M0966-72 appended. `tags.tsv` and `refs-by-move.tsv` byte-identical. |
| 3 | fixed, ruleset pending | The installed hook tests `$lref:$rref`, adds the `calibration/` path rule and the d0aac0b ancestry rule; old hook kept beside it. `branch.calib/instruments.{remote,merge}` are unset. `core.hooksPath` unset. No GitHub ruleset exists yet (decision 1). |
| 4 | fixed | One script, `set -euo pipefail`, no link until all four copies verify, `mv` then an existence check then `ln -sn`, realpath check at the end. See question 3. |
| 5 | deferred to Opus, step 3.1 | The design says "read each before copying, record per file in the ledger entry". Fine. |
| 6 | fixed in code, not exercisable yet | `check_calibmerge` diffs each tip against calib/all over `calibration/<dir>/` and `tests/`, with a one-winning-branch rule and THIRD_PARTY.md as union. Correctly says it can't run before calib/all exists. Scope note under question 4. |
| 7 | still a pass | The new code prunes `account`/`recognition` by name (`rescue.walk_scratch`, the step-3.2 tree walk, `check_shims` refuses private paths). The second pass copied nothing from a private folder (reviewed-2). |
| 8 | still a pass | Partial before run at 20:27:39 with the revised script (d7c4278d…): every tag confirmed under the stricter rule (0 missing). |
| 9 | fixed | Section 3 now rests the guarantee on the absolute citations, imports, suites, re-grades and the ledger check; scanner strips `.,:;` and keeps whole-place citations; refs table rebuilt at the freeze with `--refs-only`. |
| 10 | fixed | MAP.md's old-paths table has rows for c3/c4/c7 (moved, no symlink) and the archived plans. |
| 11 | fixed | "Archived checkout at its branch tip" said one way in the design, MAP.md and the map rows; the README line lists K1's and C7's differing files. |
| 12 | n/a (sampled rows) | The 7 new rows are well-formed and the partial before run passes `map` with 972 rows. |
| 13 | fixed and rehearsed | Steps 2.6/2.7 swapped with the no-ledger-writes rule; `edit` (prefix/any) and `copy` kinds; rows for premove/ and the bundle; manifest written and read; inventory patch in; rehearsal run. See question 2. |
| 14 | fixed | `check_shims` (realpath + `exists` after; free-or-moving before: 5 folders, 3 free); ledger after-check uses `exists`. |
| 15 | fixed | Per-suite rank/exit/failed-count comparison, lost ids and new failing ids; C4's two ids are a named allowance and only they. |
| 16 | fixed | `confirmed` now requires `missing == []`. |
| 17 | fixed | The C3 exemption is gone; `copy` and `edit` sources must exist in before mode. |
| 18 | fixed in text; not yet true on disk | Preconditions now list no session, no job, clean anatta-lab, lab pushed, site rebuilt, backup confirmed. Today: scout still running (pid 10746); p1b has finished (results at 11:53 local, no modal process); anatta-lab has 3 modified bench files and ~12 untracked paths. |
| 19 | fixed | Rollback recipes for 3.1-3.5 written; full paths and `read -r` in step 2. |
| 20 | fixed | Unpin recipe written. |

## 2. Does the rehearsal show what it claims?

Yes. I recounted from `inventory-rehearsal.json` (31,444 files, 20:18:30 UTC): calib-results 160, explore-results
13,586, explore-data 1, inputs 1, c3 106, c4 108, c7 111, k1 78 = 14,151, plus the 12 git-mv plan documents
(`calibration-c8-facts.md` is the 13th such file in lab/plans and correctly not a row) = 14,163. The report's
`missing_by_row` attributes every one of them to its move row, and the same 14,163 appear as `still_at_the_old_path`.
The other 5,182 non-scratchpad files are `unchanged` (5,182 + 14,163 = 19,345 = 31,444 − 12,099 scratchpad files,
which the rescue check owns). `changed`: 0, `unexplained_new`: 0, `problems_count` 27,911 = 14,163 + 13,748 premove
copies missing, which is the expected second view of the four outside-git folders. The failing checks are the ones
an unmoved tree must fail (pins, tags, shims, calibmerge, ledger 22/82 under moved folders, regrades reading
`results/calib`, suites' missing calib-all folder, rescue's 40 pending) and the two drifts named in the design (site
page vs committed index.html; explore-results growth). Imports pass with one more file than at 19:30. The after-only
code paths (files found at the new path, prefix and any edits, copies, manifest copies, a file left at an old path,
the suite comparison) are covered by `tmp/selftest/test_after_hashes.py` and `test_compare_suites.py`; I read that
they exist but did not run them. Not exercised and honestly labelled: calibmerge itself.

## 3. Could `step3_2_outside_git.sh` lose data or leave the folders out of step?

No data loss in any order. The only operation that touches an original is `mv` into `premove/` (a same-filesystem
rename), and it runs only after all four copies have been verified entry by entry (paths, sha256, symlink text). Any
failure before the switch leaves the four originals untouched. A failure after one or more switches (`mv` of a later
folder fails, or `ln -sn` fails) stops the script with the state of all four names printed, each either still the
original folder, a working symlink, or "ABSENT: original in premove/", and the per-folder rollback (`rm` the link,
`mv` premove back) covers each state. A second run refuses while `results/` or `data/` exists, so nothing is
overwritten. The split-brain of finding 4 cannot occur: the link is made only after the old name is confirmed gone.
Two residual windows, both acceptable under the freeze and both caught later: a writer could append to an original
between the `lsof` check and its `mv` (the after check's `copy` rows then show premove and results differing), and
the `lsof` check passes silently if `lsof` itself is missing (it is present at /usr/sbin/lsof). Two gaps in the
recipe: a failure during the copies (step 2) leaves a partial `results/` or `data/` that the next run refuses, and no
rollback line says "remove the partial copies under results/ and data/ (copies only), then rerun"; and the dummy
tree runs covered a failing `mv` but not a failing `ln -sn`.

## 4. Did the fixes introduce new problems?

Nothing that blocks. Four small ones:
- `rescue.py --newer --go` copies everything new at run time, not only what the reviewer listed. The 20:32 run copied
  41 files, not the 40 reviewed: `newer_list.txt` (written 20:32:50, during the run) and `first_sweep.log`, which
  scout is still writing. Say "reviewed 40, copied 41" in the ledger, and run the last `--newer` after scout ends.
- The design says calibmerge's output "goes into the step-3 ledger entry" (section 5, step 3.3) while step 2.6 forbids
  ledger writes until the after inventory. NOTES.md's prefix rule would accept the append, so verify would not fail;
  still, say it one way (write it in the step-4 entry).
- `check_calibmerge` diffs only `calibration/<dir>/` and `tests/`. Files the instrument branches changed elsewhere
  (root THIRD_PARTY.md, pyproject, `src/`) are not compared. Add `--` paths for them, or state the limit in section 5.
- The installed hook names `refs/heads/lab` and `refs/heads/lab/*` on the remote side rather than the proposed
  `refs/heads/lab*`; a branch named `labfoo` with no lab/ history would pass by name. Harmless; the path rule holds.

## 5. What remains before the freeze

1. Scout's first sweep finishes; the bench owner commits the bench edits, specs, `test_split.py` and `modal_p1b.py`;
   `git status` then shows only the Gate 0 files and the consolidation's own.
2. The GitHub ruleset on ejhong/anatta (decision 1); none exists today. The only guard against `--no-verify`.
3. Last `rescue.py --newer` after scout ends; reviewed-2 count noted as above.
4. Eugene's decisions 3, 4, 7, 8 recorded; the "one copy, excluded on purpose" state written into the freeze entry.
Then the freeze, as written. The second cold check (step 5) stays as planned.

## Plain reading

Every fix the first check asked for is in and checks out: the map keeps the privacy review, the push guard now closes
the unnamed-ref doors, the risky folder switch is one script that cannot make a half-moved folder, and the checker
has been run in the mode that matters and reported exactly the 14,163 files that will move, nothing more. What is
left is waiting for the running sweep, committing other agents' work, and the GitHub-side ruleset.

Written by a fresh Fable 5.1 agent, read-only, on 1 Nov 2026.
