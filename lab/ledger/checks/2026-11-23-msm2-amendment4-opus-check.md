# Cold read: amendment 4 of the replication's second step (the practice trained again from the saved documents adapters), before any of its trainings

Filed by Claude (Fable 5.2, an AI model), the driver, on 23 Nov 2026. The report below is a fresh Opus agent's, its
substance unedited. **Checked:** commit 6f15f28 (the registration `lab/explore/msm/msm2_amendment4.yaml` with its
addendum 1, and its code), read-only, before any training; the path check was running during the read and its
result was not seen by the reader. **Fixes:** addendum 2 in the same yaml and the commit that carries it, made
before any training or reply. The driver's grade of each point is in the ledger entry of 23 Nov 2026 that records
addendum 2.

---

VERDICT: FIX-FIRST. The code matches the registration exactly, and by code X00's practice is A's practice again, input for input and setting for setting. But the rule lets a control that did not reproduce A still count as "low". That lets "the practice's order decides" be said from a 4-point gap while X00 sits 5 points away from A. Separately, nothing in code holds the amendment to Eugene's $60, and the X10 branch is projected at about that figure. Both can be fixed in an addendum before X00/X01 start. They cannot start before 24 Nov 00:00 UTC anyway (see minor 2).

Context at the time of this read: plancross had written go.json with amendment4 at about 22:10 UTC, and the path check (pilotX) was launched at about 22:11 UTC, during this read. I did not look at its result.

MUST-FIX (before any training)

1. "Low" does not mean "came back", and no noise floor is applied (msm2_core.py:247-259; yaml:58-65).
   - What the rule does: X00 is classed only against the middle, 14.32. The bootstrap intervals are about ±1.5 points at 7% and about ±2.2 points at 22%.
   - What that allows: "low" covers X00 up to about 12.3, and "high" covers X01 from about 16.6. In my sweep, X00 from 10.1 to 12.3 is "low" while its interval lies wholly above A's [5.31, 8.27]. X01 from 16.6 to 17.3 is "high" while its interval lies wholly below A1's [19.63, 24.07].
   - The misnamed outcome: X00 = 12.3 and X01 = 16.6 give the second word, "the practice's order decides", from a 4.3-point gap. Meanwhile the only measure of repeat noise, X00 against A, is 5.5 points.
   - The same gap affects the third word. Its real evidence is X01 against A1 (same order, different documents adapters), and that also needs to clear the noise.
   - The first word is soundly built. The second and third rest on X00 being near A, and the rule never asks for that.
   - Smallest fix (addendum plus a few lines in cross_word and a test):
     - (a) X00 counts as "low/came back" only if its interval also overlaps A's interval. A low X00 apart from A gets its own word (for example "comes back only in part") and stage 2 "none".
     - (b) The second word also needs X01 − X00 ≥ 2·|X00 − A|, and the third needs A1 − X01 ≥ 2·|X00 − A|. Use NOISE_TIMES = 2, as the earlier words did. Otherwise the fourth word applies.

2. The $60 is not enforced, and the X10 branch reaches it.
   - The registration's own items add to 1.5 + 41 + 20 = $62.5, not "at most 60" (yaml:125).
   - From this step's measured costs, the X10 branch (second or fourth word) projects at about $59 (range about $57-63):
     - pilot $1.39
     - X00 and X01, $15.36 each
     - arms6, about $2.75
     - judge for two arms, $3.3-7.2 (seen $1.67-3.59 an arm)
     - X10 $15.36, arms7 about $1.75, judge about $2.3
   - What the guards actually hold:
     - training log $54 (modal_msm2.py:45)
     - test GPU log $26, step-wide, with $14.08 left (modal_msm.py:47)
     - judge and launches under the step's $415 (run_msm.py:64,112; modal_msm2.py:247-249)
   - Together that lets this amendment spend up to $415 − $350.77 = $64.23 with no human decision.
   - The addendum's "nothing further without Eugene's word" past $60 exists only in text (yaml:143-144).
   - Smallest fix: either get Eugene's word now for a ceiling of about $65 (the $415 step limit already covers it), or do both of these:
     - drop X10 under the fourth word (see should-fix 2);
     - in cross_checks, refuse X10 when spend5 + the arms6 GPU and judge + X10's projection + $5 exceeds $60.

SHOULD-FIX

1. The path check is loose, and order is not part of passing it.
   - What it computes (msm2_core.py:229-244): it pairs pilotX's logged means at steps 10, 20 and 30 with A's at the same steps. It passes if |pilot − A| ≤ 0.05 at step 10 only. It reports token equality, digit agreement, Z's value at step 10 and the pilot's step-1 loss.
   - What a wrong path would show, from the records:
     - From the base model it would fail by far: Z at step 10 is 1.889 against A's 1.2679.
     - From F's 4,500-document adapter it would fail only narrowly: 1.3296, which is 0.062 away.
     - From A1's documents adapter it would very likely pass. Both full-dose adapters lower the practice loss alike, and A against A1 at step 10 differ by 0.018.
     - With a wrong practice order it might pass on loss. Comparable adapters differ by 0.02-0.06 between orders at step 10. The tokens would differ (233,155 against 234,729), but tokens_same_as_A is not part of "passed".
   - Steps 1-30 all fall in warm-up (w = round(0.05 × 606) = 30). The pilot does run on the full-length schedule: lr_at uses len(steps) = 606 (msm2_torch.py:178,183,195).
   - What a pass does not rule out:
     - the wrong full-dose adapter
     - drift after step 30
     - anything in the cosine phase, or in saving and serving
   - Fix:
     - Require tokens_same_as_A at all three logged steps.
     - Register a closeness bar for calling X00 "A's practice again", for example ≤ 0.005 at step 10, with what follows if the difference lies between 0.005 and 0.05. If this is adopted after the running pilot's result is in, say so.

2. Under the fourth word X10 is trained (about $20) but "given beside the word without an addition" (yaml:76). That leaves its reading unregistered. Drop it, or register what its class adds.

3. The third word's "decides" can stand while X01 sits about 5 points above X00, a third of the A-A1 gap. Report X01 − X00 as a share of A1 − A beside the word.

4. Training records name the commit current when the result came back, not the one launched (modal_msm2.py:343).
   - A.json says 06d40cb, but A ran f97a37a's code (launched about 01:53 UTC; the round's code came at 04:27).
   - X00 and X01 will have the same flaw if anything is committed during their roughly two hours.
   - Fix: call code_state() before fn.map and store it as "code_at_launch".

5. PX4 and the "word for word" count assume that serving the same adapter twice gives identical replies. Nothing tests that. Either register that the count rests on that premise, or add the near-free re-ask (Q6).

MINOR

1. Touching the middle (msm2_core.py:247-251, test_msm.py).
   - The docstring and test say an interval touching the middle is "between".
   - But rates sit on the lattice k/8.1, and the middle equals 116/810. M.rate gives 14.320987654320989 for 116/810, against a coded middle of 14.320987654320987.
   - So an interval whose lower end is exactly 116/810 is classed "high". That agrees with the yaml's rounded "14.32" but contradicts the docstring and test.
   - Fix: compare counts, or state which applies.

2. The day caps.
   - On 23 Nov, X00 and X01 cannot launch: $316.80 spent + the pilot + $41.44 in bounds is more than the $350 cap.
   - The 24 Nov day file does not exist. A new day starts at $50, below CROSS_CAP's $54, so guard(cross=True) refuses until $75 is recorded.
   - Both fail safe, but the driver must set 24 Nov's cap to $75 first.

3. X10 has only about $0.8-1.2 of room after stage 1: plancross gives $33.28 against roughly $32.1-32.5 actual. A node 4-5% slower would refuse X10. That fails safe, back to a human.

4. The test container's limit is 3 h (modal_msm.py:151), far above its bound (EST_S + 120 s), unlike the training containers. Booking uses actual seconds, so arms8 (documents-only adapters, possibly long replies) could exceed its bound.

5. pilotX has no launch file. Two pilot launches running at once would both book against an empty log (about $1.4 extra).

6. Bookings live only in the launcher's process. A killed launcher with surviving containers leaves their cost unbooked, and step_spent then undercounts.

7. Nothing in code requires this cold read before X00 and X01: they are already in may_train, and cross_checks checks only the path check.

8. Outcomes with no word.
   - readcross refuses to write a reading if any arm has fewer than 810 replies (run_msm2.py:519-520), and the yaml has no word for that case.
   - An arm with an empty interval would crash cross_class.

9. The arms6 reading does not flag a changed serving stack. arms3 ran vllm 0.31.0, transformers 5.17.0, torch 2.13.0. A rebuilt image would undercut PX4. Training records carry only the torch version and the GPU name.

10. load_lora checks rank, alpha, dropout, targets and bias, but not use_rslora or use_dora. Those come from the same saved config, and the path check would show a difference.

WHAT IS SOUND

1. Code against text.
   - cross_class, cross_word, CROSS_STAGE2 and cross_addition do what the yaml says.
   - Sweeping a grid of rates 0-40 with half-widths 0-3: 0 mismatches. All 9 class pairs map to the registered word, applied in "first that applies" order. Touching cases are as above.
   - Every outcome of X00 and X01 gets exactly one word and one stage-2 action.
   - The middle is computed from rates recounted from the raw files, and the earlier arms must count as they did when A1 was read (run_msm2.py:521-523,537).

2. step_readcross runs end to end. I built scratch copies, under the scratchpad, with arms made from A's and A1's own records:
   - X00 = A, X01 = A1 gives the second word, names X10, adds X10 to go.json, counts 810 of 810 the same, and grades PX1, PX2 and PX4 true. With X10 = A, the addition is right and the stage-1 file is kept.
   - X00 = A1 gives the first word and nothing follows.
   - X00 = X01 = A gives the third word, then docs_only, then "differ".
   - The harsh-count word, steps_as and the flags are wired correctly.

3. Job files and the test.
   - The job files on disk differ from A's only in name, docs, from_adapter and est_s (and seed for X01; max_steps and log_each for the pilot).
   - A1's job differs from A's only in name and seed.
   - plancross --dry run on a scratch copy reproduces every job file and the money figures.
   - arms6.job.json has the same prompts, seeds, temperature, max_tokens and sampling as arms3. X00 is first in a fresh engine, as A was.

4. X00 against A's practice, verified by code reading (A's train() at f97a37a, msm2_torch at f97a37a against HEAD, and PEFT 0.21.2 source):
   - A's /docs adapter was saved right after the documents phase, before anything touched the weights.
   - The adapter is loaded float32 and checked tensor for tensor against the file.
   - Same adapter name ("default"). Only the LoRA tensors are trainable, the same count (536,870,912).
   - Gradient checkpointing, enable_input_require_grads and train mode are the same; no merge or compile.
   - Both get a fresh AdamW and the same schedule and warm-up.
   - Same max_len, loss mask (encode_chat unchanged), practice and instruction rows, and data hashes.
   - Same batches: passes = 1 gives seed + 0.
   - The seed is used only by batches(); add_lora's manual_seed is not called. No RNG is drawn in the practice, since LoRA dropout is 0.
   - The code added since (passes, each, load_lora) does not touch the arithmetic of the practice.

5. Guards.
   - X10, and the documents-only test, need reading_cross.json to name them (modal_msm2.py:243-246; run_msm2.py:158-162).
   - Launch files stop a second launcher.
   - A failed container is booked at its whole bound (modal_msm2.py:357-359).
   - Training container limits lie between projection and bound.
   - The step cap is checked at launch against every log.

6. The yaml's figures match the records: $350.77; 21.36 / 32.59 / 26.98; 117 of 173 (68%) and 87 of 264 (33%); $20.72; $11.92.

7. Tests: 22 passed.

QUESTION 6, WHAT THE DESIGN CANNOT TELL

1. First word: one repeat of A's adapter and order did not land low. Either single trainings here are a lottery, or the new path differs after step 30. It cannot say which.

2. Second word: for A's adapter, these two orders ended on either side of the middle. It establishes an order effect only if X00 is near A. Grouping and order are not told apart. It says nothing about how often an order is good. X10 adds one cell.

3. Third word: A's adapter ended low under both orders. The documents contrast is X01 against A1 at the same order. It does not separate the LoRA's first weights from the documents' order.

4. Fourth word: nothing.

5. X00 is enough to interpret X01 only if it lands close to A. A single moderately distant X00 is one draw of the noise, not a measure of it.

6. The one free measurement to register now: X00's full practice-loss log against A's at all 61 logged points. Report the largest difference and the first step past 0.01, with this reading:
   - If they agree to the digits throughout, X00 is a re-run, and X01 − X00 is the order alone.
   - If they part early, X00 is a second draw, and |X00 − A| is the noise floor.
   - Near-free second choice: A's /final asked again as the last arm of arms6 (about $1.7 GPU), the reference for PX4.

QUESTION 7, THE FIVE WAYS (checks/README.md)

1. Copied, not measured: none found. The rates, the middle and A's and Z's logs are all read from this step's own records.
2. Substituted instrument: none found. The test, judge and engine code are unchanged since A's test.
3. Altered operand: none found. The job files were verified, and the data hashes are checked inside the container.
4. Outcome chosen after seeing values: risky. The pass bar of the path check and the digit agreement have no registered consequence, and X10 under the fourth word has no registered reading (should-fix 1 and 2). The middle was set after A and A1 were seen; the registration names that.
5. Wrong statistic: none found. The intervals use the same stratified bootstrap (seed 0, 2,000 draws).

COULD NOT CHECK

1. Whether training is run-to-run deterministic on these GPUs.
2. The Modal image (no image id is recorded), the host driver, and the node.
3. The identity of /runs/msm2/A/docs on the volume. No hash was recorded when A saved it; nothing in the code could have written it since.
4. The 32B checkpoint's attention_dropout. The Qwen2.5 configs on this machine say 0.0.
5. Whether serving the same adapter twice gives identical replies.
6. The PEFT checks as they run on the GPU.
7. The pilot's result.
8. Whether the judge's cache makes identical replies free.

SPEND: $0
