---
name: reproduce
description: Run a registered reproduction spec through the lab runner (plan, trial, run, score, card) and grade it against its band
---
Reproduce: $ARGUMENTS

The runner is `anatta.runner` (clean-lab step 5, built 4 Nov): one spec YAML under `experiments/reproductions/` ->
plan -> trial -> run -> score -> card, on the Modal app "anatta-lab":
`tools/modal.sh run experiments/run.py::main --spec <path> --stage plan|trial|run|score|card`. The plan, score and card
stages also run locally (`.venv/bin/python experiments/run.py ...`). The package docstring lists its known limits.

0. Before registering, write the spec's `residual_differences` as an ambiguity table. This follows ReproBridge (Wang
   et al. 2026, arXiv 2611.08452): keeping such records made later agent reproductions succeed much more often. Each
   gap between the paper's text, its code and our run gets one entry, and unstated settings count as gaps. Each
   entry says:
   - where: the paper's section or table, and the code's file (and line) at a pinned commit;
   - the gap, what we do and why, and which source wins (by default, the code that produced the published numbers);
   - sensitivity: could it move a graded number past its band? If yes, add a pre-declared condition that measures
     it, or state why not;
   - which grades it touches.
   If the paper reports a single run with no intervals or seeds, say so in `source`: one published run can come out
   high. Example: `experiments/reproductions/v1_vlaf_gap.yaml`.
1. Check the spec is registered and pushed (`register` skill): its sha256 must match the NOTES.md entry. The runner
   refuses an unregistered `tier: full` spec.
2. Run `tools/doctor.py`; check today's spend against the day's cap (`anatta.spend.daily_cap`).
3. Plan: the runner writes `plan.json` (spec sha256, commit, models with revisions, organisms with sha256 re-verified)
   before any result exists. Cards are never overwritten.
4. Trial at the real batch size on a tiny subsample before any paid run (catches OOMs and serving errors).
5. Run under `anatta.spend` (SpendGuard for API calls, Meter for GPU time), with seed pairing across arms.
6. Score with the measure modules; write one card. Never print items or completions; cards hold numbers only.
7. Grade against the registered band (pass / near / fail) exactly as registered; no re-judging after the fact.
   Keep diagnostic runs separate from scored ones. Log every failed or dropped run with its reason. Never rerun,
   redraw a seed or switch a reading because of the value.
8. Ledger the grade (`ledger` skill) and add a notebook entry. Have Fable check a consequential grade cold, with
   the brief in `lab/ledger/checks/README.md`, which includes the five ways a result can be invalid. Then
   `report`. Every negative is a finding.
