---
name: reproduce
description: Run a registered reproduction spec through the lab runner (plan, trial, run, score, card) and grade it against its band
---
Reproduce: $ARGUMENTS

The runner (`anatta.runner`, one Modal app "anatta-lab") is built in clean-lab step 5; until it exists, say so and
stop rather than writing a one-off script. Intended flow (`lab/plans/clean-lab-2026-11-03.md`, "Build order" 5 and
"The reproductions"):

1. Check the spec is registered and pushed (`register` skill): its sha256 must match the NOTES.md entry. The runner
   refuses an unregistered `tier: full` spec.
2. Run `tools/doctor.py`; check today's spend against the day's cap (`anatta.spend.daily_cap`).
3. Plan: the runner writes `plan.json` (spec sha256, commit, models with revisions, organisms with sha256 re-verified)
   before any result exists. Cards are never overwritten.
4. Trial at the real batch size on a tiny subsample before any paid run (catches OOMs and serving errors).
5. Run under `anatta.spend` (SpendGuard for API calls, Meter for GPU time), with seed pairing across arms.
6. Score with the measure modules; write one card. Never print items or completions; cards hold numbers only.
7. Grade against the registered band (pass / near / fail) exactly as registered; no re-judging after the fact.
8. Ledger the grade (`ledger` skill), add a notebook entry, have Fable check a consequential grade cold, then
   `report`. Every negative is a finding.
