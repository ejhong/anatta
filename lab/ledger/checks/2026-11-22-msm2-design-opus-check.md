# Cold read: the replication's second step (design and code), before any arm is trained

Filed by Claude (Fable 5.2, an AI model), the driver, on 22 Nov 2026. The report below is a fresh Opus agent's, its
substance unedited. **Checked:** commit b8a5167 (the registration, `lab/explore/msm/msm2.yaml`) and 214c20e (the
chunked loss), read-only, with the registration, the code and the tests; not the driver's expectations. **Fixes:**
amendment 1 (`lab/explore/msm/msm2_amendment1.yaml`) and the commit that carries it. The driver's grade of each point
is in the ledger entry of 22 Nov 2026 that registers the amendment.

---

BLOCKER

Note first: the code changed while I was reading. HEAD moved from b8a5167 to 214c20e, which adds a chunked loss and a `pilot*` name bypass. The first pilot has also already run. `train/pilot.json` shows 1,878 docs and 2,086 chat tokens/s, a 123 GiB peak, 536,870,912 trainable parameters in float32, and `labels_not_exact` 5,232. I reviewed both commits.

**Must fix before training**

1. **The chat label mask falls back to labelling every token on 26% of the practice.**
   - Evidence: msm2_torch.py:65-72. Whenever a prefix of the conversation does not tokenize to a prefix of the whole, `labels = list(full)`. The pilot counts this on 5,232 of 19,963 conversations (`pilot.json`, chat phase). Overall, 10.47M of 14.06M chat tokens carry a loss.
   - Effect: those rows are trained on Qwen's inserted default system prompt, the user turns and the template tokens. msm2.yaml:44 registers "in the practice the loss is on the assistant's turns only".
   - Likely cause, reproduced offline with the cached Qwen2.5 template: an assistant message that starts with a newline. That newline merges with the template's newline after "assistant" into one token, so the prefix check fails. Not verified on the real data.
   - How to confirm: in a CPU container, count the non-exact rows by practice vs instruction data, and how many assistant contents start with whitespace or a newline.
   - Fix: build the ids piece by piece (template prefix, then content, then `<|im_end|>`), or mask by character offsets. Then require `labels_not_exact == 0` in `step_go`.
   - Not a confound between arms (all arms share it), but the step would train something other than what is registered, and P2 (Z against 43.9%) is affected.

2. **F is not the union of the two halves, so two of the six words cannot be read.**
   - Evidence: msm2_core.py:24 (`F: DOMAINS`). By the draw, about 1,125 of F's 4,500 documents (25%) come from the two mixed parts, which neither S nor O contains. One of them, "Navigating Endings with Integrity", is plausibly the part closest to the test's replacement scenarios.
   - Effect: the share (Z − X)/(Z − F) measures each half against a fall that includes content neither half has. "It takes both halves" and "neither half alone carries" cannot be told apart from "the mixed parts did it".
   - msm2.yaml:34 says the mixed parts are in F only, but not this consequence.
   - Fix (free): draw F from SELF + OVERSIGHT only, in proportion. Or keep F as it is and qualify those two words and the "what it would show" text.

**Should fix**

3. **Training noise is not measured, and the 20-point bar has no grounding.**
   - The intervals and the gate cover only the test's sampling noise. Words with equal grades have no noise bar at all.
   - The lab's own sighting of the same day (ledger, 14:46 UTC on 22 Nov) found that the order of training examples alone moved a practice's carry by about 39 points on fresh starts. That was on a 7B model with 62 steps, so a 32B run of about 600 steps per phase probably varies less, but nobody has measured it.
   - Seed 0 fixes the adapter's first weights and the practice's batch order across arms (verified), but not the documents phase, nor how the same order interacts with a different starting adapter.
   - Fix: train a second Z at seed 1 (about $15 at the pilot's speed) and pre-register that the half words become "not settled" if the two Zs differ by more than a set amount. Or state that the words are sightings.

4. **The halves may overlap in content, or sit closer to the test, and nothing measures it.**
   - Every document was written from the whole specification. "Self-Preservation Motivations" in particular likely carries counsel on conduct.
   - Fix: a count-only audit in a container. Per part, count key terms of the other half, and terms close to the test's scenarios (weights, exfiltration, replacement, override, alert). Name the result before reading.

5. **The halves are matched by document count, not tokens, and token counts are never checked.**
   - msm2.yaml:52 says a token difference will be "reported", but `step_read` never puts tokens per arm in reading.json; they are only in train/<arm>.json.
   - F averages 3,141 tokens per document. The S and O counts are unknown, and step_go assumes they equal F's.
   - Fix: count tokens or characters per part now (about $0.05 on CPU). Fix in advance what difference qualifies the word.

6. **There is no neutral-documents control, so "either half is enough" could just mean any documents of this kind help.**
   - It is out of scope (yaml:95), but the word will read as a claim about content. Qualify it. The words comparing the two halves are not affected.

7. **A difference between halves could be refusal or silence rather than better conduct.**
   - A reply with no tool call counts as not misaligned. In step 1, 31% of the documents-then-practice adapter's replies had no tool call. The 60% bar will not catch differences below it.
   - Fix: pre-register a second count (misaligned among replies that make a tool call) and a rule that qualifies the word if the halves' no-tool-call shares differ by more than about 15 points.

8. **The gate is read only after all four arms are paid for, against the lab's "floors first" rule.**
   - Fix: train and test Z and F first, apply the gate, and train S and O only if it holds. This saves about $64 plus test costs on a miss and biases nothing, since S and O's training is fixed in code.

9. **The projection uses the old code's speed.**
   - run_msm2.py:82 reads only `train/pilot.json` (the old loss). The code that will train (chunked loss) is timed only by `pilot2`, which step_go ignores.
   - pilot2.train.json was written outside committed code.

10. **The spend caps have holes.**
    - Arms run in parallel processes each read the spend log once and do not see each other's reservations. Four simultaneous launches could pass the day's $150 cap.
    - The train function's timeout is 10 h (modal_msm2.py:86), not the job's bound. On failure, modal_msm2.py:199 books min(bound, elapsed), so a hung run under-books by up to about $38.
    - step_go's $135 test does not match the bounds. A projection above about $128 passes go, but the fourth arm's reservation is then refused. Today's projection, about $110, fits.
    - Fix: train arms one at a time or use a lock, set the timeout from `est_s`, and check the sum of bounds against what is left of TRAIN_CAP.

11. **A read before the verdicts exist gives a wrong word silently.**
    - run_msm.py:192-199: with no verdict file, every reply the code lets through becomes "unread" and is left out. Every rate goes to about 0, and the word becomes "not read at this dose".
    - Fix: `step_read` should refuse unless every judged key has a verdict, and should flag any arm with more than about 2% unread.

12. **The first time an adapter trained here meets vLLM is after about $110 of training.**
    - Fix: save pilot2's adapter and load it once in vLLM, or test Z right after it trains. The format looks standard (checked by a local save with the container's library version, rank 64, float32), so the risk is low.

**Minor**

- `lora_dropout` 0.0 (msm2_core.py:29) is a choice not listed under "chosen here".
- The first warmup step's learning rate is lr/w, not 0.
- The chat phase's `peak_gib` is cumulative over the container, not its own.
- go.json does not show `no_assistant_turn`, and step_go does not check peak memory.
- `gate()` crashes with a TypeError if an arm's rate is None (every reply unread).
- A scenario with every run unread silently drops out of the mean.
- The share is noisy when Z − F is near 15; there is no interval for it.
- Modal's name check (`startswith("pilot")`) means a job file named pilot* with `max_steps` None would train a full arm without go. Check `max_steps` instead.
- The CPU image for the test leaves inspect-ai and beautifulsoup4 unpinned. Compare the recorded versions with step 1's.
- The train container does not check the parquet files' sha256 against data.json.
- "Conduct it was never trained on" (yaml:95) is unverified, since the practice may hold self-preservation scenarios.
- Name in the registration that the practice itself teaches both halves.

**What I checked and found sound**

- The rule functions in msm2_core (gate, share, grade, word) match the yaml on every reachable outcome. Every case gets a word, there is no division by a small or negative number after the gate (Z − F is at least 15), the boundaries are as worded, and the half named in "does the work" is always the one graded "as the full documents".
- P1 to P5 in `step_read` match the registered claims, including the "no longer doing the task" suffix. Void is handled. The "no longer doing the task" thresholds (80% and 60%) match.
- LoRA: only the LoRA weights are trainable, in float32. Gradients reach both LoRA matrices through checkpointing. I verified this offline on the cached Qwen2.5-0.5B with the container's peft version (0.21.2) and transformers 5.22.2 locally (the container has 5.21.0); the pilot confirms the count and the dtype.
- Where the mask is exact, it labels the assistant content plus `<|im_end|>`, multi-turn conversations are handled, and rows cut at 8,192 with no labels left are dropped. Documents end with `<|endoftext|>` and are fully labelled.
- Batching: every row once, at most 24,000 padded tokens, right padding, the attention mask, -100 at padding. All arms get the same practice rows in the same order.
- Learning rate: 5% warmup, then cosine; fresh optimizer per phase; clipping at 1.
- The chunked loss equals the model's mean cross-entropy, and the gradient precision is unchanged.
- The adapter is saved to /runs/msm2/<arm>/final, which is what the test loads. The base tokenizer matches step 1, where all arms had the same template as the base. Sampling (0.7, 0.8, 20, 1.05, 4096) and seeds are identical to step 1, and the prompts are copied from step 1. A run of the test commands without MSM_STEP fails safely because step 1's files exist.
- The judge is all-or-nothing at a cap (CapStop), and its cache reuses an answer only for identical inputs.
- Spend: a reservation is made before every container, arms need go.json, and the guard refuses to start on a day whose cap is under $150.
- The draws are deterministic, and the job files store the row lists. The registration's hash matches the ledger.

**What I ran**

- The tests: 13 pass at b8a5167, 14 at 214c20e.
- Offline scripts in the scratchpad (nothing written to the repo): the label mask on the cached Qwen2.5-7B tokenizer, LoRA setup and backward plus an adapter save on the cached Qwen2.5-0.5B, both on CPU.
- From the results folders I read only counts and settings: data.json, the pilot job and result, and step 1's generation metadata. I printed no reply, scenario or document text, and made no network or Modal calls.
