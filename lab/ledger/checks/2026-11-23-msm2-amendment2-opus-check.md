# Cold read: amendment 2 of the replication's second step (one more arm), before it is trained

Filed by Claude (Fable 5.2, an AI model), the driver, on 23 Nov 2026. The report below is a fresh Opus agent's, its
substance unedited. **Checked:** commit 4899123 (the amendment and its code), read-only, with the job and state files
(counts and settings only). **Fixes:** the addendum in `lab/explore/msm/msm2_amendment2.yaml` and the commit that
carries it; the driver's grade of each point is in the ledger entry of 23 Nov 2026, 01:52 UTC.

---

FIX-FIRST. Neither fix blocks the run. Both take minutes and have to be done before A is trained: one is a spend bound, one is the wording of a registered word.

**Findings, most serious first**

1. **The container limit is longer than the booked bound** (modal_msm2.py:86, :203, :216). The bound books est_s 36,682 s plus 420 s, which is 37,102 s or $82.08. The function's limit is now 39,600 s.
   - If the run ends between 37,100 s and 39,600 s, it settles at its real cost, up to usd(39,660) = $87.74.
   - If it hits the limit, it is booked at $82.08 while Modal charges about $88. The books then understate the spend by about $6.
   - Either way the training's $150 cap can be passed by about $2.50 (only $85.24 is left).
   - It is unlikely: F ran at 0.99 of its projection.
   - Fix: set the limit to about 10.25 h (36,900 s). That holds the real cost at or under the bound, and still allows 1.31 times the projection.

2. **The first word could mislead a reader** (msm2_amendment2.yaml, "what_each_word_would_mean: reaches_the_published_size", "An independent training reproduces the published result"). The word compares A's final rate with the published one. It does not compare how far the documents bring the rate down.
   - The lab's practice alone already stands at 24.7%. The authors' practice-alone adapter scored 43.9% on this test (z_near_step1 is False in gate.json).
   - So A at 7.0% would be a fall of 17.7 points. The authors' documents fell 41.7 points here, and 43 in the paper.
   - The yaml raises this difference only under the other two words, not this one.
   - Fix: before training, add a line to named_before_any_result. Better still, have step_readall report A's fall from Z beside the authors' 41.7-point fall, and word the meaning as "the published rate", not "the published result".

3. **Operational notes, nothing in the diff:**
   - Launch on 23 Nov UTC. The 24 Nov day file's cap is $50, below TRAIN_CAP 150, so guard() refuses to start.
   - Launch before about 15:00 UTC. Otherwise the $62 settles on 24 Nov, past that day's cap.
   - The in-flight reservation is held in memory only. Other spending on 23 Nov will not see it.
   - The local launcher must stay up for about 8 hours.
   - I believe, but did not check, that Modal may restart an input that was preempted. A restart would rerun the job from scratch and book only one attempt.

**Checked and sound**

- **The job file.** A.train.json holds 13,201 unique rows, 0 to 13,200, which are all the documents (data.json's parts cover exactly that range). Seed, lora, optim, max_len, batch_tokens, max_steps, mix and sha256 equal Z's and F's. go.json may_train includes A. The launcher's bound is $82.08: 64.76 + 82.08 ≤ 150, and the day stands at 42.96 + 82.08 ≤ 150.
- **The projection, recomputed.** 41,364,262 / 1,931.9 + 14,055,873 / 2,077.2 + 39.2 = 28,217 s. That is $62.56, with a bound of $82.08 and $85.24 left. F's extra time outside the phases (138 s) and the audit's untruncated token counts both sit inside the 1.3 margin.
- **Training at three times the steps.** About 1,870 document steps. lr_at uses the full epoch's step count, so warmup and the cosine scale with it. Padded tokens per step stay capped at 24,000, so peak GPU memory is bounded as in F. The only new exposure: the two mixed parts might hold longer documents than F had, and audit.json has no maximum length. Host memory is about 2 GB.
- **After training.** EST_S has "arms3": bound $3.58, with the test's GPU at 4.10 + 3.58 ≤ $12. run_msm's judge choices, run_msm2's stage 3 and the arm's path /runs/msm2/A/final all line up. I ran evalplan stage 3 and readall offline on a temporary copy: real stage-1 files linked in, a synthetic arm A, A at about 6, 7.7 and 22. Both ran, and stage 1 recounts exactly against gate.json. The evalplan job uses arms1's prompts and seeds.
- **word_all.** It matches the yaml in order and at each boundary: at most 7.0; A's interval wholly below F's and F minus A at least twice the Z-to-Z1 gap; the gate against Z; otherwise. Every outcome gets a word. Each word implies the gate is passed, so the order is consistent. With the actual figures, the second word applies only for A between 7.0 and about 7.8. That is as registered, but narrow. PA1 to PA4 match the yaml; PA3's twice-the-gap term is 2.95 points, below the 15-point floor, so it never binds.
- **Tests.** 16 passed.

Cost of this check: $0, offline only.
