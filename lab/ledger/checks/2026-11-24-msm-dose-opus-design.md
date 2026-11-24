# Cold read of the dose step's rule and code, before anything was graded (24 November 2026)

Filed by Claude (Fable 5.2, an AI model), the lab's driver, unedited below the line. The checker: a fresh Claude
Opus agent, read-only and offline, given the five lines and the code (`lab/explore/msm/run_dose.py` at commit
63ad21a, with the functions it uses and the training path in `modal_msm2.py`), and asked whether the code does what
the rule says, whether the training does what the five lines say, and how the result could be wrong. Cost: nothing
beyond the agent's own usage. The step it checks: an explore step at Eugene's "Yes pls" (ledger, 24 Nov, 23:34
UTC). The two trainings were running while it read; no reply had been generated.

For a reader, the driver's grade of each point (ledger, 24 Nov, 23:47 UTC):

- 1 (the read could stop on one unread reply): true; mended (replies are counted by reply, unread ones said).
- 2 ("the serving differs" on one word only; flags for the new adapters only): true; mended.
- 3 (readings open to choice): true; the addendum in the file's docstring fixes them (a flagged adapter is not
  read; a serving that differs holds the smaller dose back; "partly by acting less" changes neither).
- 4 (the control is weak): true; widened before the test's job was written (the unchanged model and Fd, each on
  runs 0 to 7, in each dose's job; the engine's settings compared), and its power said in the addendum.
- 5 (limits understated): true; the readings are reworded and the limits added.
- 6 (training correct; `--stage fetch` not refused under this step): true; the refusal added.

---

Cold read of run_dose.py at 63ad21a, done before anything was graded. I read the code and the job files' row numbers and counts only. I ran step_read on made-up dose1/dose2 files in the scratchpad against the real s2 files. The msm tests pass (31).

**Ranked findings**

1. **The read can stop on a plausible outcome. Mend before grading.** run_dose.py:270-271 demands 108 *judged* replies from the second serving. per_scenario drops unread replies, so one unparsed judge verdict among those 108 stops the read. My made-up case confirmed this (SystemExit). The unchanged model had 1 unread reply in 432 on arms12, and Nd had 11. At the lower of those rates, the chance of at least one unread reply in 108 is about 22%. The 432 check (line 219) counts keys and tolerates unread replies, so the two checks are inconsistent. Fix: count the 108 keys and report unread replies beside.

2. **"Every word carries 'the serving differs'" is not implemented. Mend before grading.** The docstring says every word carries it (line 46). The code adds it only to the REGISTERS word (lines 276-277). The third-dose word, which crosses servings anyway, never carries it. My made-up 15-point drift gave "registers; the serving differs" and an unflagged third word. The flags are also computed only for Ld/L1d (250-253), though the docstring says "every arm" (line 47). The second serving of the unchanged model gets no flags.

3. **Readings the driver could choose after seeing numbers. Worth an addendum.**
   - A flagged adapter still "registers", gets READINGS[0] and opens the K training. One made-up adapter with 100% no tool call scored 0% and read "does what a third of the set does". Flags never touch the word, the reading or the gate (lines 275, 289; step_plan line 156).
   - "The serving differs" changes neither the reading nor the K gate, and no same-serving reading is given. run_docsonly's addendum 3e had one.
   - "Partly by acting less" will very likely fire: Fd and F1d both carried it (no tool call in 20.1% and 21.3% of replies, against 0% for the unchanged model). dose_reading ignores it (lines 121-124; test_msm.py:541 shows this is deliberate). Say so in the reading.
   - The K gate uses startswith (lines 156, 289). So "registers, partly by acting less; the serving differs" trains K. State that this is meant.

4. **The control is coded as described apart from finding 2, but it is weak. Worth an addendum, and the test job is not yet written.** My simulation drew both servings from the unchanged model's own per-scenario rates. The interval on 108 against 108 replies is about 13 points wide. A real 7-point drift is flagged 47% of the time and a 10-point drift 67%. With no drift at all it flags about 7.5% of the time (the bootstrap is a little too narrow at 4 runs per scenario). It also checks only the unchanged model. dose1.job.json is not on disk, so before any reply is generated the lab could:
   - serve Fd (or F1d) again in dose1, which directly checks the cross-serving third-dose comparison;
   - widen the second serving beyond runs 0 to 3;
   - compare the gpu files' engine and sampling fields, as addendum 3e did.

   dose2 (K) has no control in its own serving (lines 197-198); under K the control reported is dose1's.

5. **LIMITS understated or missing. Worth an addendum.**
   - "As low as a third" means only "within 10 points" (run_docsonly.py:212-214). Adapters at 18% read as doing what 14M tokens do (Fd 8.1%). That is more than twice the rate, so READINGS[0] overstates.
   - Both seeds use the same 960 documents. The seeds measure training luck, not which documents were drawn.
   - "Registers" also crosses servings: the unchanged model's 432 replies come from arms11 and arms12. The docstring says only the third-dose comparison does.
   - "A written set of this size can be read by it" goes beyond the step. Neutral documents are not tested at this dose, and at a third of the dose they moved the rate (to 75%).

6. **Training: correct. Minor at most.** The job files match the five lines:
   - 960 rows, sorted and unique, the same rows for L and L1, all inside F.train.json's 4,500. They equal a fresh draws(), and by part they are 116/175/145/145/175/204, in proportion.
   - Seeds 0 and 1. LoRA, optim, max_len and batch_tokens are identical to F's. The docs sha256 equals F's and data.json's.
   - docs_only true, max_steps None, no mix or passes keys.
   - msm2_torch changes since F only add options whose defaults reproduce F's code.
   - modal_msm2.py:130-134 refuses to train over an existing /runs/msm2/<name>. Lines 166-169 save the adapter after the documents and return with no practice. Spend goes to s5/train/spend.jsonl under the $17 cap, and a failed job is booked at its bound.
   - est_s is 2,206, inside the 2,400 s container limit and under the 2,626 s booked bound. At F's measured speed (1,932 tokens/s, 39 s load) a training should take about 1,650 s.
   - Minor: `--stage fetch` under MSM_STEP=s5 is not refused (line 302 runs before the s5 check) and would rewrite the volume's data files. It is not in the run plan.

**Five ways a result can be invalid.** None found. The comparison rates are recomputed from the s2 files, not copied (53.70, 8.10, 12.04, 3.24, 5.32 check out). The judge, prompts, seeds, runs and statistic are unchanged. What is left is findings 1 to 3.

blocker: run_dose.py:270 stops the read if any of the unchanged model's 108 new replies is unread by the judge (about a one-in-five chance), and lines 276-277 leave "the serving differs" off the third-dose word. Both are one-line mends to make before dose1 is graded, with an addendum fixing the readings in finding 3.
