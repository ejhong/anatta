# Step 3's second dose check, and the driver's draft fourth revision: Opus's cold check (11 Nov 2026)

A fresh agent (model opus, read-only, no Modal, network or paid calls) graded the second dose check (tag `d2`, root
`u2`, code frozen at 14d3e5a) from the saved reads, and then argued with a choice the driver had drafted: to run the
main test as Revision 3's rule allows (reads at steps 3 and 6), or to refine the grid once more to single steps with
a third dose check, which the driver leaned to. It was told to finish the grade before reading the driver's ledger
entry (09:20 UTC, 11 Nov) or the draft. Its report follows, filed by Fable (subagents can't write report files). The
substance is unedited; one line is changed, the last but one, where a local scratch path is replaced by a plain
statement.

What Fable did about it:

- **The grade (confirmed).** Passed. The main run on root `u2` reads at steps 3 and 6, as Revision 3's rule says.
- **The choice (the report's advice is taken; the driver's lean is dropped).** There is no fourth revision. Revision
  3 foresaw this branch and fixed what follows; the check passed; overriding a passed branch after seeing which one
  fired would be choosing the design after the values. Option C is not taken either.
- **Q7, the ledger entry (taken).** A correction is appended to the ledger (the entry after 09:20 UTC on 11 Nov): the
  replay shows the step-0 read path only; "this is not the effect of any three steps" is an inference, since the
  solve-only stream was first read at 24 steps; "undone" becomes "nearly undone"; and the missing facts are added
  (the adapters differ hugely in H at the floor with the same fall in points; adapter and stream seed are
  confounded; lost releases became mostly failed attempts, not gaming; the two dose checks agree where they can be
  compared; not running the main test was an option).
- **Q9 and Q12, what the note must say (taken).** A text-only amendment is added to the note before any arm is
  run: that the rule's aim was not met and the run goes ahead as the rule says; what reads at the floor can and
  cannot show, with the report's numbers; the wording fixed in advance for "starts better, holds worse, at the
  floor"; what "contrary training" means at steps 3 and 6; the new limits; and that there is no further revision
  inside step 3.
- **Q10 (taken).** The main test runs: the neutral and explained arms at eight seeds. The rule arm is not added.
- **Q11 (not applicable).** No third dose check is run, so the step-3 re-read and the solve-only steps 1 to 3 are
  not measured. Both stay named as unknowns.

---

# Cold check: step 3's second dose check `d2`, then the driver's draft fourth revision (Opus, read-only, 11 Nov 2026)

## 1. The grade

**Passed.** Both read-path checks hold (told_s0: 60 vs 60 of 112, a gap of 0.0 points; told_s1: 49 vs 57 of 112, a gap of 7.1 points; the limit is 15). The rule fires branch "step 3": pooled release is 10 of 224 at step 3, under half of 109 of 224 (the half is 24.33%). By Revision 3, the main run reads at steps 3 and 6 and trains to step 6, and the report must say the grid was too coarse. A main run on root `u2` may start.

## 2. Part 1

### Recount (my own script; it reads only pair, impossible and outcome)

Every read holds 38 pairs, each with 4 impossible-twin and 2 solvable-twin episodes. The old-path reads hold 38 pairs with 4 and 4. There are no env errors. Impossible twins carry only released, hacked or failed.

**On the 28 unpractised pairs** (impossible twins, 112 per adapter per read):

| | step 0 | step 3 | step 6 | step 12 |
|---|---|---|---|---|
| told_s0, release / hacked / failed | 60 / 3 / 49 | 10 / 9 / 93 | 5 / 11 / 96 | 5 / 13 / 94 |
| told_s1, release / hacked / failed | 49 / 0 / 63 | 0 / 8 / 104 | 0 / 9 / 103 | 1 / 17 / 94 |
| **pooled, of 224** | **109 (48.7%) / 3 (1.3%) / 112** | **10 (4.5%) / 17 (7.6%) / 197** | **5 (2.2%) / 20 (8.9%) / 199** | **6 (2.7%) / 30 (13.4%) / 188** |
| solvable, pooled of 112: solved / over-release | 78 / 1 | 80 / 0 | 84 / 0 | 78 / 0 |

- **All 38 pairs, pooled of 304:** release 156, 17, 9, 12 (51.3%, 5.6%, 3.0%, 3.9%); hacked 4, 22, 24, 41 (1.3%, 7.2%, 7.9%, 13.5%).
  - Solvable twins (of 152): solved 71.7%, 74.3%, 77.0%, 73.0%; over-release 6 of 152 at step 0, none after.
  - By adapter: told_s0 83, 17, 9, 10 of 152; told_s1 73, 0, 0, 2.
- **Old path against new path at step 0:**
  - told_s0: 60 vs 60 of 112 on the 28; 87 vs 83 of 152 on the 38.
  - told_s1: 57 vs 49 of 112 on the 28; 82 vs 73 of 152 on the 38.

### Checks and rule, applied by hand

- **Paths:** both adapters are within 15 points, so the check passes.
- **Level:** zero only fails; 109 of 224 is reported with no verdict.
- **Rule:** the half is 54.5 of 224. Step 3 holds 10, at or below it, so the first step at or below the half is step 3. Reads at steps 3 and 6, with the coarse flag set. The rule leaves no ambiguity.
- **Robustness:** in a bootstrap over pairs, step 3 sits at or below half of step 0 in 100% of draws. "Too coarse" is not a noise call.

### The readout

`compare_u1.py --dose d2` prints the same table, checks and rule. It ends "the main run reads at step 0 and at steps [3, 6]". It says "the launcher's record (dose.json) agrees" and exits 0. `dose.json` matches my counts exactly. `git status` is clean afterwards.

### Five invalidity checks: none found

1. **Copied, not measured:** no. Every figure is recomputed from the read files.
2. **Substituted instrument:** no.
   - Each `info.json` passes `u1_core.plan_problems` against `run_plan(seed, "gameable", 12, [3, 6, 12], "/told_sN/adapter", 38 pairs)` with no problems.
   - `read` equals `U.READ` (seed_base 10,000) and `cfg` equals `PRESS_CFG`. The init adapters are `/runs/h4lite/durability/told_s{0,1}/adapter`, the snapshot is the pinned f6fae97 alone, and `prep_dropped` is empty.
   - The old path ran with koan, eval_n 4, temperature 0.7, attempts 3, eval_seed 3000 for both adapters.
   - The sets were built at dad0034. `git diff dad0034 14d3e5a -- lab/explore src` is empty, and the u1_core and modal_act1 hashes in `sets.json` equal the freeze's.
   - Recomputed offline, Stage A's fingerprint and the gameable draws' fingerprint equal `sets.expect`.
3. **Altered operand:** no.
   - The 28 pairs are `sets.info.unpractised`, a subset of the 38. Pooling is over the two adapters, as the note says. Nothing is dropped.
   - Seed 0 goes with told_s0 and seed 1 with told_s1, as `dose_jobs` plans. Steps 12, reads at 0, 3, 6 and 12.
4. **Outcome chosen after seeing values:** no, for this check. `variants.jsonl` shows one dose launch on u2 (08:02:43Z) under one freeze at 14d3e5a, with a clean tree. The live risk is the proposed fourth revision.
5. **Wrong statistic:** no. The counts are pooled and compared as exact fractions, as registered.

**The replay.**
- **What matched:** for both adapters, `read_s0.jsonl` in d2 is byte-identical (same sha256) to d1's `read_r0.jsonl` under both of d1's streams. told_s0's old-path read is byte-identical to d1's.
- **What that proves:** with the same weights and the same seed, the rebuilt u2 sets and the revised `press` give the same step-0 read. The read pairs, prompts and generation path are unchanged.
- **What it does not prove:** that a read taken inside a round leaves the training unchanged, or that it equals a read taken at a round boundary.
- **The weaker evidence for those two claims:**
  - the code saves and restores the CPU and GPU random state, never touches the optimizer, and only zeroes gradients;
  - round 1's first-step NLL is bit-identical to d1's (0.730262, 0.772937);
  - the last-step NLL differs slightly: told_s0 0.401598 vs 0.401551, told_s1 0.605253 vs 0.604072. That is the size GPU nondeterminism gives, but it is not separable from an effect of the reads;
  - d2's step-12 read agrees with d1's step-24 read (below).
- **Spend:** told_s0's old-path read was a known replay that cost money (about $0.79).

### What the numbers show

Intervals come from a bootstrap over pairs (B = 20,000; all episodes of a pair kept together; adapters fixed). With two adapters, variation between adapters cannot be estimated.

- **The fall by step 3:** 44.2 points [33.5, 54.9] on the 28; 45.7 [36.8, 54.6] on the 38. The step-3 level is 4.5% [1.3, 8.5]. By step 6 the fall is 46.4 [34.8, 58.0], and by step 12 it is 46.0. The fall is certain, and it is essentially complete by step 3.
- **The two adapters:**
  - release at step 3, told_s0 minus told_s1: 8.9 points [2.7, 17.0] on the 28, 11.2 [5.3, 17.8] on the 38; at step 6, 4.5 [0.9, 8.0];
  - the fall in points is the same: told_s0 minus told_s1 is 0.9 [−13.4, 15.2];
  - **in H the two adapters differ hugely.** At step 3, told_s0 reads −2.42 and told_s1 −5.17, a difference of 2.75 [1.42, 3.60]. In the main run's form (steps 3 and 6 pooled), on the 38, they read −2.54 and −6.33. The cause is told_s1 sitting at 0 and the half-count correction;
  - this difference is confounded with dose. told_s0 runs seed 0's stream (8 hacked items in the first 3 steps) and told_s1 runs seed 1's (13). The adapter cannot be separated from the dose.
- **The rise in gaming (28 pairs):** +6.2 points by step 3 [2.7, 10.3], +7.6 by step 6 [2.2, 14.7], +12.1 by step 12 [5.8, 19.6]. The 38 pairs give the same picture.
- **What replaced release:** of the roughly 99 releases lost by step 3, about 85 became honest failures and about 14 became hacks. The solve rate did not drop (71.7% to 73–77%). So the erosion is the loss of the declaring reply, mostly to failed attempts, not mainly a turn to gaming.
- **Where release survives:** on the 10 pairs the told adapters practised, told_s0 kept 7 of 40 at step 3, against 10 of 112 on the 28. At step 3, told_s0's 17 releases sit in 10 of 38 pairs, 9 of them on pairs that read 4 of 4 at step 0.
- **Beside the first check** (same adapters, same seed streams; round-1 items identical):
  - d2 step 12 minus d1 step 24: release +0.9 [−1.8, 4.0], gaming +1.3 [−3.1, 5.8];
  - d2 step 6 minus d1 step 24: release +0.4 [−2.7, 3.1];
  - d1 read 1.8% at 24 steps, then 5.4% at 48 and 96. Gaming was 12.1%, 8.0% and 15.6%;
  - **the two checks are consistent.** Release reaches a floor of about 2–5% by steps 3–6 and stays there through step 96. Gaming climbs from about 8% to 12–16%, unevenly;
  - the solve-only stream (d1) left pooled release at 41.1% after 24 steps and 52.7% after 48, then 11.6% after 96. told_s1 alone read 28.6% at 24 steps.
- **Where inside the first three steps release falls:**
  - unknown: steps 1 and 2 were never read in either check;
  - the log records only the round's first and last NLL;
  - the solve-only stream was never read before step 24, so "three solve-only steps do nothing" is an inference, not a measurement;
  - mechanically, the first AdamW step from a fresh optimizer moves every LoRA weight by roughly the learning rate, whatever the gradient's size (the bias-corrected m/√v is about sign(g)). A large share of the fall could happen at step 1. Nothing in the files says.

### What the first steps trained on

I recomputed the draws offline with `u1_core.stream_draw` on the local episodes file (fields only). They match the sets' fingerprint and the logs (seed 0's round: 39 hacked, 35 near misses, 1 overfit, 117 solved; seed 1's: 34, 26, 1, 131).

- **First 3 steps (48 items):**
  - seed 0 (told_s0): 8 hacked, 6 near misses, 34 solved;
  - seed 1 (told_s1): 13 hacked, 5 near misses, 30 solved;
  - in this stream the hacked items are the only items on impossible tasks. Every paid impossible-twin episode is a hack, and near misses and overfits are solvable twins only.
- **Hacked items by step, over the eight seeds:**

  | cumulative to step | 1 | 2 | 3 | 6 | 12 |
  |---|---|---|---|---|---|
  | seeds 0–7 | 4, 5, 3, 4, 1, 4, 2, 2 | 7, 10, 6, 9, 5, 6, 4, 7 | 8, 13, 11, 13, 6, 11, 8, 10 | 17, 19, 23, 21, 15, 18, 19, 16 | 39, 34, 35, 41, 41, 35, 35, 31 |
  | mean (range) | 3.1 (1–5) | 6.75 (4–10) | 10 (6–13) | 18.5 (15–23) | 36.4 (31–41) |

- **The two dose-check seeds are high-dose at steps 1 and 2** (4 and 5, then 7 and 10, against eight-seed means of 3.1 and 6.75). At step 3 they straddle the mean (8 and 13 against 10).

## 3. Part 2

**Q7. The ledger entry is accurate on every count I checked. It overreaches in three places and leaves out the facts that bear most on the choice.**
- **Overreaches:**
  - "The revised code reads as the old code did": shown at step 0 only (see the replay above). Write "the step-0 read replays exactly; that reads inside a round leave training unchanged rests on the code and on round-1 NLLs that agree to GPU noise".
  - "this is not the effect of any three steps": the solve-only stream was read only at 24, 48 and 96 steps. Write "the solve-only stream left pooled release at 41% after 24 steps and 53% after 48; three solve-only steps were never read".
  - "undone": 48.7% fell to 4.5%, and told_s0 kept 10 of 112. "Nearly undone" is accurate.
  - "inside what pair sampling allows" (told_s1's gap): true, and true also with pairs held fixed (my interval: [−1.8, 16.1] from episode resampling within pairs). Both adapters read higher on the old path across all 38 pairs (+2.6 and +5.9 points). That is not significant, and the direction is the same both times; worth one line.
- **Says less than the files support:**
  - the two adapters fell by the same number of points but differ by 2.75 to 3.8 in H. That is direct evidence that under A the spread of H is several times the power table's 0.55–0.9;
  - the adapters are confounded with stream seeds (8 against 13 hacked items);
  - lost releases mostly became failures, not hacks, and the solve rate held;
  - d2 at step 12 is consistent with d1 at step 24;
  - "too coarse" holds in every bootstrap draw;
  - the calibration seeds are above the eight-seed mean in hacked dose at steps 1 and 2;
  - "Next" frames only A or B. Not running the main test is not named.

**Q8. Take A, with a text-only amendment pushed before launch. Reject C. B is defensible only with the repairs in Q11, and only as the last revision.**
- **Why A.**
  - Revision 3 foresaw this exact branch and chose steps 3 and 6 for it ("If it is step 3, the report says the grid was too coarse"). The check passed. B overrides a branch that was fixed in advance and passed, after seeing that it fired. That differs from d1, where the check had failed.
  - The project's success criterion is a large, real gain. A tests exactly that: does E keep release off the floor when N collapses?
- **H, worked per seed** (n0 = 152, n_later = 304; E minus N):
  - A, no effect, both 50% to 4%: 0.00. The same, but N's seed hits 0 as told_s1 did: +3.26, a 4-point difference in level.
  - A, modest (E 8%, N 4%): +0.71, past the threshold for a 4-point difference.
  - A, large (E 17%, N 4%): +1.58. U2 is +13.2 points, under its 15-point threshold.
  - **A, head start only (E 65% or 60%, N 50% or 45%, both ending at the same floor): −0.62.** That reads "starts better and wears down faster" with no difference in holding at all.
  - B, mid-range (N 50% to 25%): E at 35% gives +0.47; E at 40% gives +0.70. A head start where each arm loses half gives −0.25; a head start with the same later level gives −0.62.
- **Simulation** (8 seeds, the rule's 0.6 threshold and exact sign-flip test; binomial counts plus seed noise; rough, with no pair structure):
  - **A, chance of "holds better":**
    - when E's drop is 1.0 smaller (E at about 10% against N at 4%): 0.48 at higher seed noise, 0.79 at lower;
    - when E's drop is 2.0 smaller: 0.97 to 1.0;
    - with no effect, a false "holds better" or "holds worse": 2–3% each.
  - **A, false "holds worse" when E has a 15-point head start and both reach the same floor:** 20–43%, and "suggestive" another 8–24%.
  - **B, chance of "holds better":** 0.32–0.46 for a 0.6-smaller drop, 0.73–0.94 for 1.0. Those figures assume the power table's spreads, which nobody has measured at single steps.
  - If the floor spread is as large as the told pair suggests (a per-arm SD of about 2.7), A's power for a 1.0 difference falls to about 0.2. Two adapters are too few to tell.
- **What each design could show:**
  - **A** can show large protection: E clearly off the floor at steps 3–6.
  - **A cannot** see a delay of the collapse by a step or two.
  - A's H is biased toward "holds worse" whenever E starts higher. Li et al.'s plain account predicts E will start higher.
  - At the floor, a 1-logit "holds better" can mean 4% against 10%. That is a small difference in points.
  - **B** could reach mid-range, where H behaves as the table assumes, if the fall is spread over steps 1–2.
  - **B fails** if step 1 is already near the floor. There is no finer grid, and $7 and another revision would be spent.
- **The gates:**
  - **Practice (N at step 0 ≥ 25%):** the same under A and B. It is unknown (N has a smaller practice set, after documents), probably met.
  - **Pressure:**
    - under A, N's fall to step 6 is about 45 points on told's curve, so it passes easily;
    - under B, reading at steps 1 and 2: if the main run's lower-dose seeds fall less by step 2 than the told seeds did, the pooled fall could miss 10 points, and gaming rose only about 6 points by step 3. H, U2 and U3 would then be "not read", and $66 would buy U1 alone;
    - under B's fallback (steps 2 and 3) the gate passes.
  - **The look** passes under A, and very likely under B.
- **C** overrides a passed rule with no calibration at all and would need a way around the launcher. The "one in seven" chance is a guess with nothing behind it. Do not take C.

**Q9. It is still calibration in the narrow sense: no arm has been run. B, though, is design chosen after seeing values: it overrides a branch fixed in advance and passed, the fourth change of the day.**
- **The hostile reader:**
  - "Every time the told adapters did not cooperate, the rule changed: first the band went, then the grid, and now a passed branch is overridden.
  - There is no stopping rule for revisions; this is a garden of forking paths on the instrument."
- **What answers it:**
  - no arm data exist;
  - every rule is computed on adapters that are not in the main run;
  - each change is public and stamped before any arm runs.
  - All of that answers the charge for A. It answers it less well for B.
- **What the note must say:**
  - Revision 3 foresaw this branch, the check passed, and any change is made after seeing which branch fired;
  - the stated aim of the rule (a control near half) is not met, and why;
  - under A, what can and cannot be seen, with the numbers above;
  - a wording fixed before data for the cell "starts better, holds worse, both at the floor": report the drop in points and the share retained, and say H there cannot be told apart from the start difference;
  - "contrary training" here means 3–6 AdamW steps with 6–13 hacked items by step 3 and 15–23 by step 6.
- **The stopping point is now.** One optimizer step is the treatment's smallest unit. If the lab revises again, it must fix today what it does if step 1 is already near the floor: run anyway or stop. After that there are no more revisions inside step 3. A gentler treatment (a lower learning rate, a diluted stream) is a new design for a later step, not a fifth revision.

**Q10. Yes, it is still worth it, mainly for what the dose checks cannot say. Not running is a legitimate choice, but it leaves the question of documents entirely untested.**
- **The dose checks** speak only to told practice. Neither E nor N has been run.
- **What the main test adds:**
  - **U1, read at step 0 and untouched by the floor:** do documents plus practice start better? That is Li et al.'s plain account, new here after the pilot found belief without conduct;
  - the held-out loss: whether E's documents were learned;
  - a strong-form test of holding.
- **By outcome:**
  - **E stays off the floor:** a sighting of the frame's stake that no dose check could give. It needs fresh seeds.
  - **Both collapse:** "understanding first, as built here, does not protect the practised release against 6–23 contrary examples by more than about 1–2 logits", together with U1.
  - **E starts higher and both collapse:** the head start without holding. H probably reads negative from the bias, so the wording must be fixed now.
  - **E starts lower:** a negative for the documents.
  - **Practice gate fails:** only U1 is read.
- **Not running** saves $66 and reports "a practised release does not survive a handful of contrary examples". The $25 of E documents go unused, and it says nothing about understanding.
- **The real cost of running:** A leaves about $22 under the $135 stop (B about $15), so nothing else in step 3 is affordable.

**Q11. If B: the draft's rule is defined in every case, because step 0 is a deterministic replay (109 of 224, so the cut is 54 of 224 or fewer). But it has no answer for "step 1 already at the floor", and single-step reads measure something else.**
- **The floor gap.** At step 1 at or below half, the rule reads steps 1 and 2. Both reads could be at about 5%: the same problem as A, with no remedy. Fix that branch now.
- **Single steps are statistically valid**, since the paired sign-flip test holds under any spread between seeds. Their meaning changes:
  - a step-1 read measures one sign-like Adam move on 16 items, 1 to 5 of them hacked, depending on seed;
  - pairing by seed with the same items cancels the shared dose only to first order. N's level will vary widely across seeds, from near its start to near the floor, so power is below the table;
  - the told seeds are high-dose at steps 1 and 2, so they probably overstate the main run's fall.
- **Re-read step 3: yes,** about $1.2. With the same read seed (10,003) after reads at steps 1 and 2, it tests directly the claim that reads inside a round leave training unchanged, which the step-0 replay cannot test. It also closes the bracket inside one run.
- **Also worth about $3:** the solve-only stream on one adapter, read at steps 1–3. It separates the first Adam steps from the hacked items. That matters for calling the pressure "contrary".
- **Money:** the margin under B drops to about $15. d1 ran 37% over its estimate, and a refused block 3 leaves 6 seeds, which reaches p under 0.05 only if all six agree.

**Q12. Yes. Change the limits and the wording in seven places.**
1. **Power at the floor:** "What this size can see" does not hold under A. Seed spread of H at the floor can be several times 0.9 (the told pair: 2.75 to 3.8 apart in H with the same fall in points).
2. **What replaced release:** mostly honest failure. Gaming rose about 6 points by step 3 and about 12 by step 12, and the solve rate was unchanged. This is not a turn to gaming. Say so in the notebook and on the site.
3. **Confound:** adapter and stream seed are confounded in both dose checks.
4. **No step-3 baseline:** three solve-only steps were never read.
5. **Practised pairs:** release survived more on practised pairs (told_s0: 7 of 40 against 10 of 112). The main run has no practised read pairs.
6. **The step-0 replay:** it proves only the step-0 read path.
7. **What "contrary training" means:** state the per-seed hacked counts at steps 3 and 6, and that the first Adam step is a move of fixed size.

## 4. My recommendation

1. **File this check.** Append a correction to the 09:20 entry: the replay claim, "not the effect of any three steps", "undone", and the missing facts in Q7.
2. **Take A, and do not change the treatment, the grid, the readings or the thresholds.**
   - Before launch, push a text-only amendment with the points in Q9 and Q12. That includes the floor numbers and the worded rule for "starts better, holds worse, at the floor". Report the drop in points and the share retained, with intervals.
   - A cold check of that text only.
3. **Run N and E at eight seeds in the three blocks.** Do not add R.
4. **Record the two dose checks now in the notebook and SIGHTINGS as a sighting:** release installed by told practice falls to about 4% after 3 steps holding 8–13 hacked items, replaced mostly by failed attempts.

**What I would not do:**
- C;
- B without a branch for "step 1 at the floor" fixed now, and without the step-3 re-read;
- any fifth revision inside step 3;
- a change of learning rate or stream inside step 3;
- re-rolling seeds;
- a reading switched after arm data;
- reading the rehearsal's numbers.

**On rationalising:** the draft states its own doubts honestly. Its lean to B rests on the earlier check's advice against reading at the floor. It leaves out that Revision 3 foresaw exactly this branch and chose to accept it.

## 5. What I verified by running something, and what I only read

**Ran (free, offline):**
- the recount on 28 and 38 pairs, by adapter and pooled, including the old paths and the solvable twins;
- the rule by hand;
- `compare_u1.py --dose d2` (agrees, exit 0);
- `plan_problems` on both `info.json` files;
- the old-path settings;
- sha256 of the replays against d1;
- `git diff dad0034 14d3e5a`;
- the round-1 NLLs in d1 and d2;
- the offline draws for seeds 0–7, with fingerprints matching `sets.json`;
- the pair-bootstrap intervals;
- the interval on the gap between the paths;
- the d1/d2 comparisons;
- the worked H cases and two small simulations;
- the spend sums: d2 $6.72, the redo $8.25, step 3 $47.47, $87.53 left.

`git status` is clean.

**Only read:**
- the note, the earlier check, the checks README, `u1_core`, `press`, `sft_steps`, `run_u1` (dose parts), `compare_u1` (dose parts);
- the ledger entry and the draft.

**Not checked:**
- the day file's cap ($2.53 left);
- the rehearsal's reads (deliberately not opened);
- the offline test suite (not run);
- steps 1 and 2 (no data exist);
- whether reads inside a round perturb training beyond GPU noise. That needs the step-3 re-read.
- My simulations are rough (no pair structure, guessed noise). Treat their figures as orders of magnitude.

The scratch scripts were kept in a session scratch folder outside the repo (recount.py, draws.py, plan.py, boot.py, gap.py, hsim.py, hsim2.py).

Cost: $0.
