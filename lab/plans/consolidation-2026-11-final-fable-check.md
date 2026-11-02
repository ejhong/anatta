# Second cold Fable check: consolidate/2026-11 → lab

Fable 5.1, read-only, started 01:32 UTC 2 Nov (from `date -u`). Nothing edited, run or pushed except read-only git, hashing and one Python recount in my scratchpad. Private folders never listed or opened (one `git ls-files --error-unmatch` on the single path `harness_v2.py` to confirm Opus's "tracked" claim: it is).

## Verdict: LAND

Nothing is lost, nothing is at an old path, the re-grades are identical, the two mid-execution verify.py changes are honest and tightly scoped, and the branch diff is small and clean. Three small things to do after landing (not blockers), listed at the end.

## 1. Inventory diff

I re-did the accounting myself from `inventory-before.json` (73,697 files, 01:08 UTC) and `inventory-after.json` (129,830 files, 01:27 UTC) with only the `move`/`worktree-move`/`git-mv` rows of `move-map.tsv`, independent of verify.py:

- 56,353 moved with identical sha256 (matches verify's 56,353); 17,339 unchanged in place (verify says 5,225 because it counts scratchpad/rescue files separately); **0 missing**; 5 changed in place, exactly the 3 rewrites (lab/README.md, explore/README.md, scout/paths.py; `rewrites-applied.tsv` records each, reversed to its before sha) and 2 edits (MAP.md, plans/archive/README.md). NOTES.md grew by 36 lines with 0 lines removed (the append rule holds).
- Premove copies: 55,938 byte-equal, 0 bad (matches verify).
- After-files not derived from a before-file: 195 = the new `anatta-calib` worktree, 9 ledger files, 4 plans files (all `new`/`to-git` rows) and 5 new scratchpad files, which the fourth rescue pass covered (`verify-after-partial.json`: rescue pass, manifest 2,535 rows, 0 pending). Note `verify-after.json` itself records rescue as `fail` (4 pending) because it ran before that pass; the partial report is the one that passes.
- Disk spot-checks (sha256 of 8 sampled files against the before inventory): results/calib ×2, results/explore ×2, data/essay PDF, archived c7 and k1 worktree files, one git-mv'd plan: all match at the new path and (where applicable) in premove/. Old names resolve through the symlinks.

## 2. The two verify.py changes

(a) Per-folder pytest on calib/all: honest. Each `calibration/<dir>/tests` has its own top-level `card`/`conftest`; one process would shadow them. The comparison still unions the ids into one "calib" group: 725 passed before → 775 after, `worse: []`, no lost ids beyond the named allowances.

(b) `C4_LIBRARY_GUARD` (14 ids): correctly scoped and I confirmed the cause structurally. `src/anatta/models.py` is blob 2ee38ec on calib/all = calib/refusal-direction (C3) ≠ 77bd736 on calib/persona-extractor = calib/C4-graded. C4's `app.py:632-634` raises `SystemExit("... is not the registered instrument")` when any registered `src/anatta/{probe,models,factor,modal_app}.py` sha differs; of the 10 src files differing between C4-graded and calib/all, models.py is the only registered one. The 14 ids are all in `test_c4_app.py`, failing SystemExit/AssertionError, and the 2 pre-existing allowances are the only other failures; prh_k1 102/102, refusal_direction 182 pass + 2 guard-limited. I did not run the tests myself (unguarded pytest could write); Opus's "--tb=line one by one" is the only per-id evidence. Not a merge fault; C4's graded code is intact at its tag and archived checkout.

## 3. calibmerge

Confirmed. `merge-base calib/prh-k1 calib/instruments` = c69b782 (main); instruments is not an ancestor of prh-k1. For the 13 flagged files: all 13 have calib/all blob == calib/instruments blob; 11 do not exist on prh-k1 (or main) at all, the other 2 (test_bench.py, test_koans.py) are identical on prh-k1 and the merge base; `git log c69b782..calib/prh-k1 -- <13 paths>` is empty. So the "changed by the merge itself" flag is an artifact of the base, not a lost change. The other 73 differences each have one winning branch.

THIRD_PARTY.md: for each of the seven branches, the only non-blank line absent from calib/all's file is the branch's own intro sentence; calib/all's intro is the union sentence (EvalPlus, Inspect parse, MMLU-Pro prompt, lm-eval WMDP/MMLU, Arditi pipeline, StrongREJECT judge, LR schedule, Zephyr template, MBPP task, persona-vectors script, Łucki note). No code or data entry lost.

## 4. Re-grades

Three CPU re-grades, before and after byte/number-identical: fishery_grade 37/37 labels across stageA/atlas1/em1/lie1 (2 "own numbers only"); self_tests_grade self1/atlas1 identical (hits 124,127,130,132; digest 6cdbae50…); K1 card rebuilt from `results/calib/prh_k1` byte-identical (f8e62eb0…) both from the archived worktree and from tag 237841e. C1/C3/C4/C6/C7 are not re-graded (GPU); they rest on the tags check (every card code hash matches its tagged commit: C1 12/12, C3 12/12, C4 12/12, C6 13/13, C7 8/8, K1 3/3, C2 657bf09e; I verified the C2 blob hashes to 657bf09e).

## 5. Ten traces through MAP.md

Works end to end (INDEX → NOTES line → results folder → code at tag): **C3** (7880: f878d5e = calib/C3-graded; card at calib-results/refusal_direction_g4 resolves via symlink), **C4** (7709: 1711324; results/calib/persona_extractor), **K1** (7551: 237841e; results/calib/prh_k1), **C2** (4377: sha 657bf09e; tag calib/C2-graded points into lab; test_cal_judges reproduces it), **Hub cut** (1303: h4lite/hub_cut.py, run hub1 → lab/experiments/h4lite/results/hub_cut/hub1), **Third person** (1315: tp1 folder exists), **explore safari** (7533 → results/explore/2026-11-01/safari/transcript_safari.md via the shim).

Partly stuck: **C1** (5111), **C6** (4918), **C7** (4872): the grade entries name no commit and no results path, only "card.json" and a Fable grade-check doc (`calibration-c{1,6,7}-fable-grade-check.md`, which are in `plans/archive/` but not in the archive README's list). A newcomer gets there only via MAP's rule "tag calib/<ID>-graded" plus guessing the instrument folder name (model_check, koan_scorer, relearning); `results/INDEX.md` shows `results/calib/relearning` cited by no NOTES line. Also, old ledger lines citing `anatta-calib-c3/c4/c7` (e.g. 7343) now dangle by design; MAP's table explains, verify records them as "unresolved" but passes.

## 6. Landing risk

Low. `lab` = `private/lab` = 5487e54 and is the merge-base; consolidate is exactly 2 commits ahead (f3f2fd3, bc56f6b), so `--no-ff` makes one merge commit, no conflicts, and the push is a fast-forward on the remote. The diff is 33 files, all under `lab/`, none in the private folders. The 11 to-git files and MAP mention the private folder *paths* by name (5 lines) but no contents; they go to the private remote only. The pre-push hook refuses lab/calib/consolidate/dev refs and any history carrying d0aac0b (C7's port, which calib/all contains) toward ejhong/anatta; I did not call GitHub to confirm the rulesets (no network). Worktree is clean apart from the four Gate 0 files and the empty `other`. Premove copies and locked archived checkouts remain, so every move is reversible.

**After landing, three small fixes:** (1) keep a copy of the verify.py that produced `verify-before.json` (sha d7c4278d) — only the later 397b36ba version is on disk, so the before baseline can't be rerun exactly; (2) list the three `calibration-c*-fable-grade-check.md` files in `plans/archive/README.md`; (3) one ledger line (or an INDEX note) naming the results folder and tag for C1, C6 and C7, since their grade entries don't.

## Plain summary

Every file that existed before the move is still here, byte-identical, at its new home, with a second copy in the archive; nothing is left behind or unexplained. The merged calibration branch holds every branch's code; the 13 flags are files K1 never had, not lost edits. The two verify changes describe real, harmless causes. Re-grades match exactly. The map works for most claims; C1, C6 and C7 need one line each to be as easy to trace as the others. Safe to merge into lab and push to the private remote.
