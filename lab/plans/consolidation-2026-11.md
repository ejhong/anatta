# Consolidation: one clear home for code, results and writing (November 2026)

Claude Opus 5.5 wrote this plan on 1 Nov 2026 at Eugene's request (ideas 59). Eugene asked: "this is actually very important so spend time making it really great and well done and double check with fable maybe and do it when you think it's appropriate - put it in the plan."

## Why

Some of the spread is deliberate:
- the public library is kept apart from the private lab;
- each calibration is frozen on its own branch at the commit that was graded;
- data stays out of git.

The rest has built up:
- **Code in four kinds of places:**
  - the public library (`anatta`, main);
  - a separate dev worktree (`anatta-dev`, branch dev/ministral), which about 30 lab scripts load as a live path. A change there can silently change an old experiment;
  - seven calibration branches (calib/*), four of them with worktrees;
  - the lab's own scripts: lab/experiments/h4lite (80 entries mixing code, data, results and notes), lab/viability and lab/explore.
- **Results in four places:**
  - lab/experiments/h4lite/results, inside git, some files untracked;
  - /Users/eugene/prj/align/calib-results;
  - /Users/eugene/prj/align/explore-results;
  - Modal volumes.

  Downloaded data has its own folder, /Users/eugene/prj/align/explore-data.
- **Old copies at the top of /Users/eugene/prj/align:** claim/, site/, inputs/, notebook.md and plan-2026-10-28.md, beside the current copies in the lab.
- **Writing in many files:**
  - the ledger (NOTES.md, 7,655 lines, append-only);
  - the notebook, the ideas log and the hypotheses doc;
  - plans/, the site and SIGHTINGS.md.
- **16 branches and 7 worktrees.**

The cost is time lost finding things, a risk to reproducibility (the live dev path), and a harder road to main. Every public claim should lead to its code, data and ledger entry in a minute.

## What "done well" means

1. **A newcomer can go from any claim to its code, data, command and ledger entry in under a minute,** using one map.
2. **Nothing is lost.** Every file that exists before exists after, at a recorded new path or in an archive. This is checked by sha256 over a full inventory, not by eye.
3. **Every graded result stays reproducible from a tag.** A sample of grades re-run on CPU from stored data gives identical numbers.
4. **Old commands still work, or fail with a message naming the new path.**
5. **The privacy boundaries hold:**
   - the private account isn't touched (not opened, not moved);
   - benchmark items and keys stay out of every repo;
   - nothing private moves toward main except through a curated PR.
6. **Every test suite passes,** the site builds, and the scorecard tally reads {'hit': 43, 'miss': 73, 'open': 7}.

## The target layout (to be settled in the design step)

- **The public repo (`anatta`, main):** the library (src/anatta), the paper and docs. The consolidated calibration/ tree arrives later through the curated PR (ideas 49).
- **The private lab (`anatta-lab`, branch lab):**
  - lab/explore: scout, bench, scenarios, conditions, PROGRAM.md, SIGHTINGS.md;
  - lab/experiments: code only, one folder per line of work, each with a README naming its results, ledger entries and re-run commands;
  - lab/ledger, lab/ideas, lab/plans (with archive/), lab/site;
  - **lab/MAP.md:** one page saying where every kind of thing lives and what each document is for;
  - **a generated ledger index:** each experiment's registration, grade and limits by line number. NOTES.md itself stays append-only and untouched.
- **The library the lab uses is pinned.** One option is to merge dev/ministral into lab, so the lab's own src is what its scripts import. The other is to tag dev/ministral and load that tag. Either way, no lab script reads a live worktree path. anatta-dev is retired after that, with its branch kept.
- **One results root outside git:** /Users/eugene/prj/align/results/, holding calib/, explore/, and the archive of experiments' results, with an INDEX.md. Results inside git that are tracked stay where the ledger cites them, unless the design step finds a clean way to move them with a path map.
- **One data root outside git:** /Users/eugene/prj/align/data/<source>/, each source in its own folder, untrusted and read as data only.
- **Calibration:** the seven calib/* branches are merged into one branch with one calibration/ tree. Each graded commit gets a tag (for example calib/C1-graded), so its card still names the exact code that ran.
  - That branch stays private. Its history holds C7's port of an unlicensed scorer (calib/relearning, d0aac0b to 02db868), so code reaches main as fresh commits of the current files, never by merging calib/* history (NOTES, 17:11 UTC on 1 Nov).
- **The top-level old copies** are compared file by file with their lab versions, then moved into /Users/eugene/prj/align/archive-2026-11/ with a README. The private recognition folder stays where it is.
- **Branches:** every tip is tagged before anything moves. Merged branches can then be retired. Their commits stay reachable through the tags, and nothing is deleted.

## Steps, with the checks

0. **Inventory (read-only).** Record every file in the seven worktrees, the top level and the results and data folders: path, size, sha256, git status, and which files reference it (path search).
   - The recognition folders are counted only by name and size from outside, never opened.
   - Output: inventory-before.json, kept outside git.
1. **Design.**
   - The target layout, settled.
   - A move map: old path to new path, with the reason.
   - The reference rewrites each move needs.
   - The library-pinning choice.
   - The verification script.
   - **A cold Fable check of the design before anything moves.**
2. **Freeze.**
   - No agent or Modal job may be running.
   - Tag every branch tip (consolidation-before-YYYY-MM-DD).
   - Note the time in the ledger.
3. **Execute** on a branch of its own (consolidate/2026-11) and in a scratch copy of the folders outside git.
   - Moves use `git mv`, or `mv` outside git, never deletion.
   - Path rewrites go through a script, with a short shim where an old path must keep working.
   - Commit with pathspecs, never `git add -A`.
4. **Verify.**
   - The inventory after matches the inventory before, every sha256 accounted for.
   - Every test suite passes.
   - Every lab script imports, or runs `--help`, without error.
   - The site builds with the tally unchanged.
   - A set of graded results is re-graded on CPU from stored data with identical numbers: at least fishery_grade, self_tests_grade and K1's card.
   - Every path the ledger's last 30 entries cite resolves through the map.
5. **A second cold Fable check:** the inventory diff, the re-grades and ten claims traced end to end by someone who has never seen the lab.
6. **Land.**
   - Merge into lab.
   - Fetch and check ancestry, then push to private.
   - Write MAP.md.
   - A ledger entry with the counts.
   - A note to Eugene with what moved and where.

## When

- **The design (steps 0-1) starts once the current builders land** (scout and the C7 scorer rewrite, today or tomorrow). It is read-only, so it can't disturb running work.
- **The moves (steps 2-6) run after C3 is graded,** in a quiet window with no running jobs, and **before the calibration PR to main** (about 6-8 Nov), which they make easier.
- If an explore sweep is mid-flight, the freeze waits for it.

## Cost

- Agent time only, about a day of work, plus two Fable checks.
- No GPU or API spend, except one CPU re-grade pass.

## Risks and guards

- **Breaking reproducibility:** handled by tags at every graded commit, identical re-grades, and pinning the library.
- **Losing a file:** handled by the sha256 inventory before and after. Moves only.
- **A privacy slip:** the recognition folders are never touched, benchmark items stay out, and nothing goes to main here.
- **Disrupting running work:** handled by the freeze.
- **Over-engineering:** the layout serves the map, and the map serves the one-minute test. Nothing else is restructured for its own sake (depth over breadth).
