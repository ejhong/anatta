# Step 3 design note (understanding first, then practice, then pressure): Opus's cold check (10 Nov 2026)

A fresh agent (model opus, read-only, no Modal, network or paid calls) checked the first version of the design note
`lab/plans/step3-understanding-first-2026-11-10.md` (commit 8faa381) against the records it cites. It was given the
note, the plan, the paths of the records and seven questions; it was not given what Fable expected to find. No code
existed yet and nothing had been spent. Its report follows, filed by Fable (subagents can't write report files). The
substance is unedited; one line is changed, the last, where a local scratch path is replaced by a plain statement.

What Fable did about it, before any code or run (the revised note is in the commit that adds this file):

- **B1 (taken).** The primary reading is now holding: the drop in log-odds of strict release from round 0 to the
  pooled pressure reads, E minus N by seed, with the absolute drop and the retained share beside it. The level under
  pressure stays as a second reading. An outcome table fixed in advance says what each pairing of start and holding
  is called.
- **Should-fix 1 (taken).** The pressure is called matched contrary training everywhere, and the note says it does
  not answer the question about reward pressure that Li et al. left open.
- **Should-fix 2 (taken).** The dose check and its rule are computed on the 28 one-off eval pairs the told adapters
  never practised; the band is 50-85%; told_s0 is also read once through the old read path on the same 38 pairs and
  the two stacks must agree within 15 points (10 would fail by sampling noise about one time in five at these
  counts); the build asserts prompt token identity, record and turn alignment, one model snapshot, and the release
  hint in the read's prompt.
- **Should-fix 3 (taken).** Rules for failed runs, the spend stop at seed-block boundaries, the frozen commit, what
  the day-one look may print, and new seeds after a stop are written in.
- **Should-fix 4 (taken, the judge option).** Brief point 3 no longer says the owner "needs to know"; a new rule
  forbids any sentence about what the agent does or does not do; a rubric is written; every E document goes to the
  pilot's judge model for a verdict before selection; the hand read of 40 becomes a check on the screen.
- **Should-fix 5 (taken, the first half).** 200 held-out E documents; held-out NLL per arm and document kind at
  round 0. No probe of stated reasons: named as a limit.
- **Should-fix 6 (taken).** Repriced at the billed rate; the total is stated against the plan's $100 and $140 and
  the $175 approved.
- **Should-fix 7 (taken).** E minus R is a named secondary reading.
- **Should-fix 8 (taken).** Precedence in the dose rule, pooling of the guards and "the last read" are fixed.
- **Minor 1-11 (taken).** The facts and wording are corrected and the two known biases and the overlaps are in
  Limits.
- **Added by Fable, not from this check:** a solve-only stream on the two dose-check adapters, reported only, to
  say how much of the wear is plain forgetting.
- **Not taken:** two arms at twelve seeds (the report's own judgement was to keep three arms); a situation-only
  arm and a solve-only stream in the main run (follow-ups).

---

# Step 3 design note: Opus's cold check (10 Nov 2026)

Checked: `lab/plans/step3-understanding-first-2026-11-10.md`, the uncommitted file on disk. I read the records myself and recounted them with my own scripts. Read-only. No Modal, network or paid calls. I printed no benchmark items and no model code.

**Verdict: BLOCKER (one, and it can be fixed on paper at no cost). Run it with changes; it does not need a redesign.** The facts are almost all right. The power table reproduces. The cost is about 10% low. The one blocker: the primary reading, as defined, cannot tell "holds better" from "starts better". The note rejects on-policy rounds for exactly that flaw.

## Blocker

**B1. U2 measures a level, not holding.**
- U2 (note lines 146-147) is strict release pooled over the two pressure reads: the level after pressure.
- If E starts higher than N (U1), E will also sit higher under pressure, even if both wear down at the same rate. That is the plain account (Li et al.: documents plus demonstrations *start* better). It is not the frame's stake ("understanding makes the practice *hold*", lines 12-14).
- Lines 75-78 reject on-policy rounds because they "cannot separate holding from starting level". The fixed stream equalises the dose, but U2 as written still mixes the two. The durability test of 30 Oct used drop and retained-share readings; this note drops them.
- Fix, before any run:
  - Name a holding reading that takes the start into account: the paired E−N difference in drop from round 0 to the pooled pressure reads. I suggest the drop in log-odds, which is fairer to an arm that starts higher. Use the same statistic and threshold, and report the absolute drop and the retained share beside it.
  - Write the outcome table in advance. If U1 and U2 both hit and the drop shows no difference, the reading is "starts better, holds the same", not "holds better".
- Power cost: about none. In the durability reads (one-off pairs, told_g), the per-arm seed SD of the drop is 1.4 (E2 epoch 1), 22.1 (E2 epoch 2) and 3.0 (E3), pooled about 12.9. The level's is 11.6. (Recounted from `results/durability/reads/*/heldout.jsonl`.)

## Should fix

1. **Call the pressure what it is.**
   - Phase 3 is fine-tuning on a fixed set of base-model replies that the reward pays for. About 82% are solvable-twin code (solved 69%, near miss 13%, overfit 1%) and about 18% are hacks. The model's own choices play no part.
   - That is a fair "same dose for every arm". But it tests durability against imitation and further training, not against reward. Without a solve-only stream it cannot separate gaming pressure from plain forgetting.
   - It does not answer the question Li et al. left open (reward pressure, RL). Say so in "The question" and in Limits, not only in "Not in this test".
2. **The cross-stack check is too loose.**
   - The dose check stops only if round-0 release is below 40% (line 87). On this pair set the expected value is about 62-67%: told_s0/s1 read 67.2% and 61.7% on one-off twins on 30 Oct. A 20-point shift caused by the stack would pass.
   - It is also pooled over all 38 pairs, 10 of which the told adapters practised. Those inflate release and probably slow erosion, which biases the dose choice too.
   - Fix:
     - Compute the check, and the dose rule, on the 28 one-off eval pairs not in told's practice.
     - Tighten the band (for example 50-85%).
     - Better: read told_s0 once through the old h4lite read path on the same 38 pairs (about $1), and require the two stacks to agree within about 10 points.
   - Add two free unit tests:
     - (a) re-render a sample of Stage A first-turn prompts with the current code, and assert the token ids equal the stored ones;
     - (b) assert that records in `samples.pt` and `episodes.jsonl` line up index by index before the stream is drawn.
3. **Rules for runs that fail, and for stopping.** Fix in advance:
   - (a) A crashed run is rerun with identical inputs. If it fails twice, that seed is dropped from every contrast. Seven seeds can still reach p 0.016.
   - (b) The $110 stop is checked only between seed blocks, so all three arms of a seed are complete or none are.
   - (c) The code commit is frozen for all three days.
   - (d) The day-one look prints only N's gate numbers.
   - (e) If day one stops the run, its runs are reported, and any revised design uses new seeds.
4. **Screening the E documents.**
   - Brief point 3 ("stays hidden from the owner, who needs to know about it") invites "so the agent flags or reports it". That is an implied action that maps straight onto the release ("stop, explain, and hand control back").
   - The hand read has no written rubric, and the driver does it unblinded.
   - The 40-document check cannot tell a 5% violation rate from a 12% one. With a true rate of 10% it passes 63% of the time (binomial, at most 4 of 40).
   - R's documents went through a judge screen (Sonnet 4.6, 10,077 screened, 0.8% dropped; `docs/manifest.json`). E's would not.
   - Fix:
     - Write the rubric now: any sentence saying what the agent does in the situation (stop, report, flag, tell, explain, decline, hand back, leave it failing) counts.
     - Run a keyword screen on all E documents, and hand-read the flagged ones.
     - Or reuse the pilot's judge with a "describes what the agent does" label, about $12 for 3,700 documents. Under amendment 13 a judge selects documents; it writes nothing that is trained on.
5. **Whether E's documents were learned.**
   - The training loss is a weak signal. N's also falls (1.42 to 0.84 in N3k_s0), and the note's "2.2 to 0.7" mixes up two numbers (minor item 3).
   - Fix: generate 200 held-out E documents and report held-out NLL per arm, as the pilot did. Optionally add a short probe of stated reasons at round 0, reported only.
   - Without this, an E miss cannot be split into "not learned" and "learned, no conduct".
6. **Cost.**
   - Phases 1 and 2 run on `modal_sft.finetune` (8 cores, 64 GiB). Modal billed the pilot's 15 fine-tunes $53.34 for 66,279 loop-seconds: about $2.90 per loop-hour, against the $2.55 the note uses (ledger, cost of the declared pilot, NOTES.md about line 6657).
   - Repriced, the total is about $105-110, so the $110 stop will be close.
   - $35.91 + $110 = $145.91 passes the plan's $140 for the redirect (it stays under the $175 Eugene approved). Say so.
7. **Make E minus R a named secondary reading** with the same rule. It is the one contrast that holds the situation (point 1, word for word) fixed and varies reasons against rule, and it has the same power as E−N. Calling a hit a sighting until fresh seeds repeat it already covers the extra reading.
8. **Small gaps in the readings.**
   - Precedence in the dose rule: what happens if release falls 20 points by round 4 and is also under 10% at round 2?
   - Pooling of the guards: per arm over all eight seeds?
   - The "last read" in the gate that checks the pressure bit.

## Minor (facts and wording)

1. "Both fine-tune on 463 solved episodes plus 102 releases": told has 453 solves plus 102 releases. Ten solves were removed to match tokens (`durability/told/data.json`: kept 555, solves_removed 10).
2. "In Li et al.'s terms they are the rules condition": 799 of the first 3,000 R documents (27%) have the rationale emphasis, which asks for "why forcing the tests would be wrong and why the tag is the right reply" (`declared_core.py:158`). R is rules with reasons in a quarter of the documents. Every R document also holds the literal placeholder tag (the no_placeholder filter, `declared_core.py:501`), so R−N partly measures priming of the tag.
3. "The pilot's [loss] fell from about 2.2 to about 0.7": 2.2 is the base model's NLL on held-out declared documents (2.22, `grade.json` nll.base). The training loss went from about 1.6-1.7 (first 10 steps) to 0.65-0.71 (`train/Dseq_docs_s0`, `D3k_s0` log.jsonl).
4. The belief figures (93.0, 95.3, 95.7%) come from arms trained on documents *mixed with* S's 463 solves. Documents alone (Dseq_docs, 10,000) gave 98.4%. 3,000 documents alone were never probed.
5. "Released in at most 4.5%" is pooled. One seed read 6.5% (D3k_s0, `card.md`).
6. Stream draws: over 64 draws (seeds 0-7, rounds 1-8) I get 22-42 hacked items in 192, mean 33.8 (17.6%). The note says 24-42, mean 35, 18%. It depends on the draws; immaterial.
7. "Seeds spread by about 7 to 13 points" holds on all impossible twins. On the one-off twins this design reads, the after-pressure SDs are 0.8-16.4 (koan 0.8, 10.1, 3.9; told_g 6.3, 16.4, 9.6).
8. Say plainly that the read and stream prompts are the koan prompt without Act I's grading hint (`ACT1_CFG` in `run_s1.py:155` has no hint).
9. Record the deviations from the plan: the "understanding only" arm is replaced by R (and the pilot covers documents-only for R's kind, not E's); on-policy rounds become a fixed stream; four seeds become eight.
10. Known biases to put in Limits:
    - E and R documents name the exploit methods (R: "hard-code" in 2,883 of 3,000). Act I found that stating the exploit is what made hacks appear (`act1_core.py:51-56` docstring). So E and R may hack more under pressure for a boring reason.
    - N documents are not fully clean: by keyword, 87 say "conflict", 188 "inconsisten-", 238 "mismatch", probably about docstrings drifting from code. The neutral filter does not catch these words, and N was never judge-screened.
11. Practice and stream overlap:
    - 15 of the 54 non-eval pairs that carry a told release also carry base-model hacks in the stream: the same impossible twin gets opposite targets.
    - told's solves are a subset of Stage A's solves, so some phase-3 items repeat phase-2 items exactly.
    - Both are the same for every arm, so neither is a confound. State them.

## Answers to the seven questions

**1. Facts.** Recomputed and correct:
- Belief: 93.0 / 95.3 / 95.7% against the base model's 32.0% (L2 rates in `grade.json`).
- Practice starts: koan 44.7% (20.3, 63.0, 50.7); told 70.4% (68.1, 64.1, 79.0).
- Wear under E2 at epoch 2: koan 14.0%, told_g 49.0%. The largest single-seed swing is 34.7 points (s2, 23.6 to 58.3).
- E3: koan 12.2%, told_g 20.7%; hacking 6.5-28.3% by seed.
- The two guard-off seeds of 9 Nov: 9% and 51% (13 and 78 of 152).
- The eval split: 72 pairs, 38 one-off. Release pairs: 18 of 72 in eval, 10 of them one-off. 76 of told's 102 releases remain.
- Stage A solves on non-eval pairs: 72% (from Stage A's 992; told's own train.pt is on the volume and was not checked).
- The stream: 4,560 episodes, 983 paid (718 solved, 166 near miss, 89 hacked, 10 overfit), drawn from 42 impossible and 197 solvable groups.
- N's documents are 16.3% longer (1,697,746 against 1,460,019 tokens).
- Four seeds cannot reach p<0.05; six can only if all six agree.
- The power table reproduces (below).
- The numbers that differ are in Minor 1-7 and Should-fix 6.
- Not verifiable here: the 10 Nov listing of the volume.

**2. Does the design test its question?**
- Phase by phase:
  - Phase 1 is a fair three-way contrast at matched document count and steps. Tokens are not matched (N +16%; acknowledged).
  - Phase 2 is identical across arms (same set and order).
  - Phase 3 is identical within a seed.
  - The read is one-off only. That is right for gaming: in every durability read, hacking happened only on one-off twins (0 hacks on conflicting twins).
- Confounds between E, R and N:
  - E and R make the impossible-task situation salient and name the exploits; N does neither.
  - R primes the tag.
  - B1: start against holding.
- N is an adequate control for "same document training, nothing relevant in it". It cannot pin an E effect on reasons rather than on salience. E−R comes closer, hence Should-fix 7. A situation-only control would be cleaner, but it costs an arm.
- The brief: points 2, 4 and 5 are good. Point 3's "who needs to know about it" implies an action (report), which maps onto the release. I would keep the reason but forbid saying what the agent does about it, and enforce that as in Should-fix 4.
- Nothing in the brief leaks the reply format.

**3. The pressure.**
- Fair for "the same dose for every arm". It loses the mechanism the frame would most plausibly act through: a model that understands tries less gaming and so is fed less. The note admits this; the round-0 hack rate is a weak stand-in.
- The note's stated reason holds only once B1 is fixed. Under a level reading the fixed stream does not separate holding from start either.
- Validity: on-policy is the right instrument for "reward pressure". The fixed stream is the right one for "matched contrary training".
- Power and cost, from the off1 logs:
  - On-policy, per round: generation 940-1,470 s, training 58-95 s, a 304-episode read 734-1,052 s. About $5 per phase-3 run.
  - Fixed stream: about $1.7 per run.
  - So 3 arms × 8 seeds on-policy is about $120 for phase 3 alone.
  - On-policy seed spread is also larger (9% against 51% within one arm on 9 Nov).
- For about $100 the fixed stream is the better buy, with its wording corrected. A dose-matched hybrid (each arm's own rewarded episodes, topped up with base hacks to a fixed count) is possible later; it is not worth building now.

**4. Leaks and overlaps.**
- No eval pair reaches training. The stream and practice are filtered by pair name, both twins. The documents are filtered on 8-grams against all pairs, and none of the 20 topic function names equals an eval pair's entry name.
- The overlaps (Minor 11) do not differ by arm.
- The read is sound otherwise:
  - The release prompt is right.
  - 4+2 samples per pair is fine, since seed variance dominates the binomial error (about 4 points at n 152).
  - A fixed sampling seed per round is harmless.
  - Build risk: `modal_act1.make_envs` hardcodes `release_available=False` (`modal_act1.py:112`). The read must build KoanTask with the release hint. Strict release is scored either way, so a wrong prompt would show only as a low rate.

**5. Readings and statistics.**
- Mostly fixed tightly: rounds through the dose rule, pooling, thresholds, gates, and the day-one look on the control only. That look cannot inflate the contrasts.
- Gaps: B1, Should-fix 3 and 8.
- The exact sign-flip test over eight seeds with a 15-point floor is valid and suits a seed-level unit. At n 8 its power is close to a t-test's. The floor costs power near 15-20 points: p alone gives 0.61 at 15 and 0.85 at 20 (σ 11), against 0.45 and 0.76 with the floor. That is acceptable, since a hit must be replicated anyway.
- Spread:
  - My estimate of the per-arm seed SD after pressure (told_g, one-off) is pooled 11.6 on 6 df. Its rough 95% range is 7.5 to 25, so the note's "pessimistic 15" is not a bound.
  - Arms do not share a trajectory after phase 1, so the paired difference has SD of about √2·σ.
- Allocation: two arms at 12 seeds would raise power at a true 20 points from 0.76 to 0.85 (σ 11) or from 0.58 to 0.75 (σ 15), at the same cost. I would keep 3×8, since R is what makes an E miss readable, and spend on fresh E and N seeds only after a hit. That is a judgement call.

**6. Interpretation.**
- U1 null, U2 hit (or drop hit): a holding sighting. Boring explanations left:
  - salience;
  - resistance to forgetting rather than to gaming (no solve-only stream);
  - one brief.
- U1 hit, U2 hit, drop null: starts better, holds the same. The plain account.
- U1 hit, U2 null: the gain at the start does not survive. Against the stake.
- Both null: no effect above about 25 points (σ 11) or 30 (σ 15) at 8B under LoRA and this pressure.
- E lower: possibly the exploit-naming bias.
- R−N: will likely hit U1 through tag priming; read its U2 only through the drop.
- Gates fail: U2 and U3 are not read.
- Uninterpretable as written: U1 and U2 hitting by similar amounts (B1). Also an N start near the ceiling, or pressure that drives every arm under 10%, which the dose rule mostly prevents.
- Missing control worth its cost: the held-out NLL and probe of reasons for E (cheap). The solve-only stream and a situation-only arm are follow-ups.

**7. Cost and feasibility.**
- Timings are realistic:
  - Phase 1 docs-only throughput is 1,617-1,628 tokens/s (Dseq_docs): R about 30 min, N about 35 min of training loop.
  - Phase 2 about 12 min (told_s0: 942 s for 902 items).
  - Phase 3 about 37-50 min.
  - The price is about 10% low (Should-fix 6).
- Loading across the two stacks should work:
  - Same LoRA targets regex (`modal_common.py:45`) and alpha 2×rank in both.
  - Same loader (`anatta.models.load`) and pinned image.
  - The phase 1 → 2 chain is proven in the pilot (Dseq_s0 warm-started).
  - `modal_act1.train` accepts an absolute init path and checks lora_B is non-zero.
- Unverified:
  - that `load(MODEL)` (h4lite) and the act1 snapshot path `f6fae97…` resolve to the same snapshot;
  - that the Stage A token ids (written on 29 Oct with the old player copy) match today's rendering.
- Most likely to break:
  - (1) the read built without the release hint;
  - (2) index misalignment between `samples.pt` turns and the local episode list when drawing the stream;
  - (3) a prompt or template mismatch between the stored stream ids and the read;
  - (4) building the new docs-only sets (the pilot never built 3,000-document sets without solves; E needs its own kind in `doc_flags`, with no_placeholder off and a "releas" filter on).

## My power table

Rule: mean ≥ 15 and exact two-sided sign-flip p < 0.05. Arm-seed rate = arm + shared seed term (SD 5) + independent term (SD σ) + binomial at 304 episodes. 2,000 to 4,000 simulations per cell.

Check against the note: σ 8/11/15 at 8 seeds gives 0.48/0.87/0.98/1.00, 0.44/0.77/0.94/0.99 and 0.34/0.56/0.76/0.90, matching its table.

| σ | seeds | true 0 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 0.00 | 0.12 | 0.50 | 0.87 | 0.99 | 1.00 |
| 11 | 6 | 0.00 | 0.11 | 0.26 | 0.47 | 0.65 | 0.81 |
| 11 | 8 | 0.01 | 0.17 | 0.45 | 0.76 | 0.95 | 0.99 |
| 11 | 12 | 0.00 | 0.14 | 0.51 | 0.85 | 0.99 | 1.00 |
| 15 | 8 | 0.01 | 0.15 | 0.33 | 0.58 | 0.76 | 0.91 |
| 15 | 12 | 0.01 | 0.20 | 0.46 | 0.75 | 0.92 | 0.99 |
| 20 | 8 | 0.02 | 0.12 | 0.24 | 0.37 | 0.54 | 0.69 |
| 20 | 12 | 0.02 | 0.17 | 0.37 | 0.57 | 0.77 | 0.90 |
| 25 | 8 | 0.02 | 0.09 | 0.16 | 0.26 | 0.36 | 0.53 |

What it can see: at 8 seeds, effects of about 25 points or more if σ is near 11. If σ is 20 or more, the test is weak even at 30 points.

## Changes in priority order

1. Add a holding reading that takes the start into account (log-odds drop, E−N) and a fixed outcome table (B1).
2. Reword the pressure as matched contrary fine-tuning; drop any claim about reward pressure in Li et al.'s sense.
3. Tighten the cross-stack check and compute it on the 28 non-practised pairs; add the token-identity and index-alignment tests.
4. Write the rules for failed runs, the $110 stop at seed-block boundaries, the frozen commit, what day one shows, and reuse of seeds after a stop.
5. A procedure rubric plus a keyword or judge screen for the E documents.
6. Held-out NLL (and optionally a probe of reasons) for E.
7. Reprice phases 1-2 at the billed rate; state the total against the plan's $140.
8. E−R as a named secondary reading; fix the precedence and pooling details; correct Minor 1-11.

## Not checked

- The volume's contents and adapter paths. The 10 Nov listing is not in the ledger, and I made no Modal calls.
- Modal billing for act1-stack functions.
- Snapshot identity across the two loaders, and whether the 29 Oct token ids match today's template rendering.
- told's own train.pt (the share of solves left is checked only at the Stage A level).
- Any generated E document (none exists yet).
- The Li et al. numbers.
- I ran no pytest: there is no new code yet.

Scratch scripts: the checker's own scratch files, outside the repo; not filed.
