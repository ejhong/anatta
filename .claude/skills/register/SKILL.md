---
name: register
description: Register an experiment (hypothesis, measures, bands, decision rules) in NOTES.md and push it before any result exists
---
Register: $ARGUMENTS

A registration fixes what will count before anything is seen. Rules: `lab/plans/clean-lab-2026-11-03.md` ("Rules the
code enforces", "Decision rules").

1. Write the spec (YAML, under `experiments/reproductions/` or `experiments/explore/` once the runner exists; until
   then beside the experiment's code). It names models, organisms, interventions, measures, seeds, n, the spend cap,
   `tier` (quick or full), and for a reproduction the published target, its source (paper, table, figure) and the
   band. Bands are set now from the source and never moved after results.
2. Check each measure's calibration status on each model from its card (`results/cards/`). A `tier: full` claim on an
   uncalibrated measure is not allowed; say which measures are `waiting`.
3. State the decision rules: pass, near, fail, and what each outcome means (including the awkward ones: base above
   threshold, control moves, a split result).
4. Record the spec's sha256 (`shasum -a 256 <spec>`) and the code commit it will run.
5. Append the registration to `lab/ledger/NOTES.md` with the `ledger` skill (clock stamp from `date -u`), citing the
   spec path, sha256, commit, bands and cost estimate against the day's cap.
6. Commit and push to main **before** any paid run or any look at results. For a consequential claim, a fresh Fable
   agent checks the registration cold first (brief: `lab/ledger/checks/README.md`). A reproduction's spec carries
   its ambiguity table in `residual_differences` (`reproduce` skill, step 0).
