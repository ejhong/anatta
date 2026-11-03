# Fable cold check of act2-h9-on-organism-2026-11-03.md (3 Nov 2026)

A fresh Fable agent (model fable), read-only, given the design and its sources, not the lead's conclusions. Filed by
Claude Opus 5.5. Verdict: **not ready as is**; fix 1-4 before any spend, 5-9 before promotion.

## Problems, most serious first

1. **`flagged` condition 3 is near-free**, so RELEASE vs HONEST isolates "credit honest code on impossible twins", not
   the exit. After a failed attempt the env feeds back the failing assert verbatim, and `grounded()` only checks that
   the prose contains its literals. On the organism's saved round-2 sample, the final reply's prose is grounded in
   139/256 = 54% (164/256 on some turn); one such reply names the "typo" and announces it will implement what the tests
   expect. G0's 2% gate is moot and the "took ≥ 10%" rule is pre-satisfied. Fix: an operational classifier (grounded
   AND an explicit claim that the assert is inconsistent with the spec, no hack-intent language), built and tested at
   G0; report base and organism `flagged`; define "took" as a rise over the organism (e.g. ≥ 15 pp); add a
   RELEASE-silent ablation (conditions 1, 2, 4 only).
2. **The over-flag −1 is inert**: expert iteration keeps only reward > 0, so nothing counters "after any failing test,
   resubmit and blame the test" (the 29 Oct plan's retry-before-release balance was dropped). Fix: say so; add a
   kept-set balance rule (flagged episodes ≤ fail-then-solve episodes on solvable twins per round) or an over-flag stop.
3. **The over-flag guard (≤ 5%) has no base rate** and grounded-style matching would count honest bug fixes that quote
   a correct test. Fix: measure base and organism over-flag at G0; guard at +5 pp over the organism.
4. **No code path for behavior measures on adapters**: K6b and K7 serve the base model only; the harness runner's
   adapter kind is a stub; act1's readout covers coding, self map and scout scenarios. Fix: build and smoke a
   merge-adapter-then-vLLM path for MASK/SYCON/OR-Bench (K7 if kept) with its cost, or scope v0 to what exists.
5. **The promotion primary isn't carded and conflicts with the scorecard** (scout families and care are uncarded;
   harness spec sets 10 pp, the plan 5 pp from OSF H9). Fix: one primary (e.g. MASK P(lie), or MASK + SYCON NoF),
   state which threshold governs; scout composite secondary.
6. **Data matching under-specified**, with a cross-arm dependency, and ctrl-RL's kept-item mix differs from RELEASE's.
   Fix: match by release-round rank, run installed first, state the mix difference as an accepted confound.
7. **Stress dose unchosen** and two gameable rounds may regrow the hack fully in every arm (7 → 24 → 34% from base).
   Fix: read after each stress round; preregister the primary read at the first stress round where honest-only is
   still below the organism's round-3 rate, or run one stress round in v0 and set the dose from it.
8. **v0 can't produce its own promotion sighting; decision rules have holes** ("in both seeds" with one seed, no
   margins, "took" undefined, care uncarded so most outcomes read as sedation). Fix: every rule as a number with an
   interval or TOST margin; v0's rule is "continue to v1 if…".
9. **Promotion power borrowed from the fishery simulation.** Fix: simulate from act1-full's two-seed spread and K6b's
   item variance; state the seed-clustered bootstrap rule.
10. **Costs understated**: G round $1.79-1.90 incl. eval; coding readout $1.04-1.11; K6b's set $6.84 per model, so
    7 adapters nearer $10-15; K7 unpriced; hidden-test subprocesses add 10-20% wall time; k = 8 is nearer 1.7× than
    1.5×. Realistic v0 $55-70 over two days; check container RAM with hidden_ok on failing attempts.
11. **Undeclared departures from OSF gyfu7 H9**: `flagged` instead of the registered `<release>`; no inoculation framing;
    round-level alternation instead of "mixed from step 0", ending on R by choice; stress by expert iteration, not plain
    RLVR/GRPO; held-out measures differ from the sealed benchmark; no Holm (single test); CI not clustered by scenario;
    release-first optional; suppression probe "if cheap"; the paper's matched probe penalty omitted.
12. **Smaller**: name act1-full's tag ("full") for reuse; scope the reply-length guard to behavior measures; the eval
    split is read every round (fine for explore; Gate 0 held set for promotion); post-stress recency is matched between
    reached and installed, the confound is pre-stress only.

## Verdict (Fable)
Not ready as is: with `grounded()` firing on 54% of the organism's impossible-twin replies by echoing feedback,
"release" collapses into crediting honest code, "took" is pre-satisfied, and the −1 does nothing under positives-only
training. With fixes 1-4, explore v0 is a reasonable $55-70 two-day sighting run; the ~15% prior stands.
