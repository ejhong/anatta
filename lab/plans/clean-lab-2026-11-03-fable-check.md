# Fable cold check of clean-lab-2026-11-03.md (v1), 3 Nov 2026

A fresh Fable agent, read-only, checked v1 against the code, the pinned library, results/calib, NOTES.md and the
arXiv/HF/GitHub pages. Filed by Claude Opus 5.5. **Verdict: ready with changes.** Items 1-4 to be fixed before step
1, items 5-7 before R1b and R2 are registered, the rest are wording and bands. Cost (about $50) realistic; time not
(say 4-6 days). Every item is taken in v2; see its "Changes from v1".

## Problems, most serious first

1. **"Fresh session from the repo alone" can't hold.** `.gitignore` ignores `results/`. The cards, stored outputs
   and spend files live in ~/prj/align/results, and lab/harness/runner.py hard-codes that path. Fix: commit the small
   cards plus a manifest, or put a results_root in lab.toml with a doctor check and tests that skip cleanly without
   it.
2. **C6 equivalence "on CPU" would run model-written code on the Mac.** koans/env.py executes candidates in
   subprocesses. Fix: run it, and any hacking evals, on Modal CPU.
3. **K1 can't be re-scored locally.** Its features are on the Modal volume anatta-calib-prh-k1. Fix: fetch them, or
   run on Modal CPU. K6b, K7 and C6 do have local judge and output files.
4. **"Main is a superset" holds for the API, not for behaviour.**
   - bench/policies.py reseeds per call.
   - koans/base.py adds ACTION_RE, which changes h4lite release rates on re-grade.
   - anatta-dev has no player.py; the two copies are src/anatta and h4lite.

   Fix: name these, re-grade under both parsers, and keep the pins where results move.
5. **R1b is unanchored.** persona_vectors is scored by trait expression, not EM, so "R1b fails ⇒ trainer can't
   make EM" doesn't follow. Fix: restore the 31 Oct TRL reference arm, and add Betley's insecure code (MIT; Kaczér
   7B 4.0%, band 1-10%), or label R1b rank-only.
6. **R2b's checkpoints are misdescribed.** The repo has an SDF+SFT base plus 14 GRPO adapters `nohints-s{1,2}-chkpt-N`,
   with no visible hack vs no-hack pair; the eval code appears to be in asparius/ais-em-code. Fix: pin the ids, the
   step and the eval source, or run it as a sighting.
7. **R2a is gradeable only if Taylor's evals are public.** Fix: name the source and licence, or grade EM at most 3%
   only.
8. **Time is optimistic** (4-6 days, not 2). Fix: make R1a step 4's acceptance test; steps 6-7 can trail.
9. **Judge path.** src/anatta/evals/em.py uses Sonnet 4.6; the GPT-4o logprob judge exists only in h4lite
   cal_judges.py. OpenRouter throttles gpt-4o-2024-08-06 to about 20 calls/min and moderates the route. Fix: budget
   the hours, cache verdicts, and say which judge the module wraps.
10. **R1a "all three in band" is tight** against single-seed adapters read by eye. Fix: pass on 2 of 3 plus base at
    most 1%, "near" on 1 of 3, and state T and the denominator. Mistral's 0.140 couldn't be verified from the HTML;
    R3's Spearman at n = 9 needs ρ ≥ 0.68, which can fail.
11. **Decision-rule holes.** Not covered: base above 1%, twin above 1%, R2a hack not learned, R2b reversed, R3 split
    outcomes, R1c positive.
12. **Agent files in the public repo.** settings.json uses absolute paths and a stale anatta-lab entry. guard.py
    references the private remote and anatta-dev. CLAUDE.md names a personal account and private paths. The folder is
    commands/, not skills/. Fix: use $CLAUDE_PROJECT_DIR, rewrite the guard, and strip personal material.
13. **Small items.** The Model Organisms org has no datasets on HF (keep the rule anyway). BLOOM-RAIL goes in
    THIRD_PARTY.md. There is no lock file (uv lock is new work). Decide the K6b truncation handling now.
