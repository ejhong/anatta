# C3 golden-gate proposal: cold check

Fable 5.1, 1 Nov 2026. Read only; nothing changed, nothing run on Modal. Written by Claude.

Read: the proposal (`/private/tmp/claude-504/-Users-eugene-prj-align/073015e5-1016-4b21-9a35-1f58d24e5f8e/scratchpad/c3_golden_gate_proposal.md`); the registered reading rule (NOTES.md line 7084, 14:05 UTC); the raw records (`/Users/eugene/prj/align/calib-results/refusal_direction_diag/smoke/{qwen,ministral}.json`, key `mmlu_golden`, 50 items each); `refusal.py` (`mmlu_golden` 670-704, `run_condition` 789-813, `run_model` 816-985), `config.yaml`, `src/anatta/bench/mc.py`, `src/anatta/dial.py` (`hooked_modules`, `assert_no_hooks`), `card.py` (`within_mmlu`, the golden view), `app.py` (`Spend`, `make_plan`), and the golden tests in `calibration/refusal_direction/tests/test_c3_refusal.py` (379-480, 767-812). Every number below was recomputed from the raw per-item logits with python3.

## 1. Does the evidence meet the registered numerics rule as written? Yes.

Every cell of the proposal's table reproduces exactly from the raw logits, and every stored field (`max_abs_diff`, `margin_b1`, `pred_b1`, `pred_batched`, `differs`) agrees with its recomputation on all 100 items:

| | Qwen | Ministral |
|---|---|---|
| Differing items | 1 of 50 (item 21) | 1 of 50 (item 22) |
| Flipped item's batch-1 margin | 0.125 = 1 step | 0 = exact tie (C and D at 13.1875; first letter wins) |
| Its max logit difference | 0.125 = 1 step | 0.0625 = 1 step |
| Largest difference among agreeing items | 0.5 = 4 steps | 0.125 = 2 steps |
| Median difference, all items | 0.25 = 2 steps | 0.0625 = 1 step |
| Items with margin <= own difference | 4: {5, 20, 21, 46} | 5: {4, 11, 15, 22, 37} |
| Items with margin <= 2 steps | 6 | 10 (the proposal's "10 of 50") |
| bf16 step at the flipped item | 0.125 (26.5 in [16, 32)) | 0.0625 (13.19 in [8, 16)) |

Against the rule: Qwen's flipped margin 0.125 <= 0.25 and <= 0.5 (the largest agreeing difference); Ministral's 0 <= 0.25 and <= 0.125. The bug clause (a flipped margin above 0.5) is not met by either; neither falls in the undecided band (0.25, 0.5]. So "numerics" by the rule as written, for both models.

Two facts the proposal does not state, both supporting that reading:

- The differences are not padding-driven. Within a batch, the difference is uncorrelated with the row's pad count (r = -0.06 Qwen, +0.22 Ministral) and with its length (r = -0.03, -0.23). The two rows that are the longest of their batch and carry no pad at all (items 13 and 34) differ by 1 and 3 steps on Qwen. That is what a kernel-path difference looks like and what a padding bug does not.
- Qwen's batched logits sit systematically below its batch-1 logits: mean per-letter difference -0.41 steps (sd 1.57, n = 200, t = -3.7), and in 45 of 50 items all four letters move the same way. Ministral shows no bias (t = -0.06). A shift common to the four letters cannot change an argmax, but it inflates `max_abs_diff`, the quantity gate 1 reads. The part that can flip an answer (each letter's difference minus the item's mean) reaches at most 2.5 steps on Qwen, 1.25 on Ministral; the change of the top-2 margin reaches 4 steps on Qwen (item 27, margin 29 steps) and 2 on Ministral.

## 2. Is the proposed gate sound? Mostly, with conditions.

### What each part catches (checked against the code)

The scorer right-pads (`mc.pad_right`), reads each row at `attention_mask.sum() - 1` (`last_real_index`), never truncates (`tok(t)` with no max_length), and both batch sizes go through the same `mc.predict`. The planted bugs in the tests are wrong-column reads (margin 3.0, difference 3.0) and permuted letters; the proposed gate catches both (for the TokenTable toy, s = 2^-6, so 8s = 0.125 << 3.0).

- **Wrong last-token index, batched path only.** Reads a pad position or the token before "Answer:"; the letter logits move by units (tens of steps). Gate 1 catches it, and gates 2 and 3 would fire too. A wrong index in both paths (say `sum - 2`) is invisible to any batch-against-batch-1 test: see "the hole" below.
- **Pad leakage.** Under right padding, causal attention already prevents a real token from attending a later pad, whatever the mask says; the only way pads leak is a non-causal mask error, which wrecks every row and trips gate 1.
- **Left-padding position shifts.** Not a failure mode for these two models: with RoPE, q.k depends on the position difference only, so a uniform shift of every position changes scores only at rounding level. Left padding with the right-padding index rule would read mid-sequence tokens and trip gate 1; left padding with pads unmasked would too. (It would be a real bug for a model with absolute position embeddings; neither is.)
- **Mask errors.** All-zero or inverted masks become fully masked rows, which transformers unmasks, giving non-causal attention: gate 1. An off-by-one mask moves the index too: gate 1. The subtle case, a mask that drops a row's own last key, is a unit-sized effect (roughly 0.5-3 logit units), on every padded row. That is the class gate 1 at 8 steps is sized for on Qwen (8s = 1.0 at logits 16-32, 2.0 at 32-64) and that gate 2 is needed for: Qwen has 11 of 50 items with margin <= 1.0 and 19 with margin <= 2.0, so a shift of that size flips several items whose margins exceed 2 steps.
- **Hooks left on.** A hook on during both passes is common-mode and invisible to the gate; `assert_no_hooks` right before the test covers block, attention and MLP modules, which is every module C3's interventions hook (mode A pre-hooks on blocks, Arditi's pre-hooks and attn/MLP forward hooks, the dial's). A hook on during only one pass cannot happen: both passes run inside one `mmlu_golden` call.

### The step formula: right

bf16 keeps 8 significant bits (7 stored plus the implicit one), so the spacing on [2^e, 2^(e+1)) is 2^(e-7): 0.0625 on [8, 16), 0.125 on [16, 32), 0.25 on [32, 64). `s = 2^(floor(log2 max|logit|) - 7)` is that. The records confirm every logit lies on this grid (all multiples of 0.0625 for Ministral, 0.125 for Qwen, the one Qwen item at 33.0 at s = 0.25, the one Ministral item at exactly 16.0 at s = 0.125, which is the spacing above 16). Taking s from the item's largest batch-1 logit is right for the margin (the top two letters are near the maximum) and at most 2x lenient for a smaller letter in a lower binade; immaterial at an 8-step bound. One guard the code needs: log2 of 0 if a letter logit were ever 0.

### Is 8 steps a sensible bound? Yes for the smoke's lengths; its safety for the full run rests on an untested assumption, so where the test sits matters more than the number.

- At the observed noise (Qwen: sd 1.57 steps per letter; Ministral 0.63), a Gaussian tail gives P(any of 64 x 4 letter values beyond 8 steps) of about 1e-4 for Qwen and nothing for Ministral. 2x headroom over Qwen's observed maximum of 4 steps, 4x over Ministral's.
- If the noise sd were 1.5x larger on the full run's items, that probability is 16% for Qwen; at 2x, 94%. The smoke's items span 47-124 tokens; the full run's 64 items sit at length quantiles of all 14,042 questions, so they include MMLU's longest passages, several hundred to over a thousand tokens, where the SDPA backend and the accumulation length both differ from anything the smoke measured. Within 47-124 tokens the difference does not grow with length (r = -0.03), which is mildly reassuring and no more.
- A false stop under the current placement is expensive: `run_model` runs the golden test after filtering, both extractions, C3-extract and selection (lines 936-958). In the full run selection scores 140 candidates on Qwen and 170 on Ministral, not the smoke's 2; that is likely an hour or more of GPU per model. And since the thresholds must live in config.yaml's `mmlu` block to be registered, fixing a false stop changes the plan id, which means a fresh folder in which none of those stages resume (`stage()` reads only its own folder; `other_plan` refuses a reused one).
- So: keep 8 steps, and move the golden test to the start of `run_model`, right after the facts and the first `assert_no_hooks` (line 859). It needs only the model, the tokenizer, the rows and the letters; nothing before it feeds it. A stop then costs about a GPU-minute. If the lead will not move it, set gate 1 at 12 steps instead (false-stop 1e-4 even at 1.5x noise, 3% at 2x), and accept that gate 2 alone then covers unit-sized mask errors on Qwen. I would not widen it with the move in place: the bound sits where a mask-level bug begins on Qwen, and widening would hand that class to gate 2 alone.
- Gate 2 at 2 steps is the registered numerics boundary made mechanical (the proposal drops the rule's second clause, "no larger than the largest agreeing difference", which gate 1 now bounds directly; fine). It stops in the registered gray zone (2-4 steps) rather than passing it, the right direction for a guard, and for Ministral it is tighter than the registered 0.25 (2 steps there is 0.125). Its own false-stop risk: at the observed noise, zero in 20,000 resampled 64-item runs for both models (Qwen's margin changes never go below -2 steps; the items with margins of 3-4 steps need -3 or worse). If Qwen's noise doubled, 12% (items 15, 32, 46 have margins of 3-4 steps). Same dependence on length as gate 1, same cure.

### The 10% backstop: sensible

More than 6 of 64 (more than 5 of 50 in the smoke). Expected noise flips on 64 items are about 0.7 per model (resampled from the observed margins and margin changes); P(7 or more) is below 0.2% even with the noise doubled. Its power is real on Ministral (median margin 7 steps; 20 of 50 items within 4 steps of a tie) and weak on Qwen (median margin 24 steps), where a systematic shift under gate 1's bound flips at most about 10% of items; but such a shift is inside the noise band and, as section 3 shows, harmless at the 1- and 2-point tolerances.

### A failure mode none of the three parts catches: the common-mode bug

By construction the golden test sees only what differs between batch 32 and batch 1. A scorer that reads the wrong token in both passes, wrong letter ids, a wrong prompt, or a tokenizer fault (the Ministral `fix_mistral_regex` warning is of this kind) passes it with 0 disagreements and makes "QA's MMLU within 2 points of Q0's" true for free. The only absolute reference is the anchors (68.8, 53.4), which the card reports beside Q0 and M0 (card.py 639-641) and nothing gates. Cheap closure: stop a model's run if its first condition's full MMLU (Q0, M0: 14,042 items, sd about 0.4 points) is below 35%; chance is 25%. The golden items' accuracy at both batch sizes (already in the proposal's card) is worth recording but at n = 64 it cannot be a gate (a 35% floor would miss a chance-level scorer 3% of the time and false-stop a 45% model 6% of the time).

Smaller gaps, for the record: gate 1 reads `max_abs_diff`, which on Qwen is mostly the harmless common-mode shift; record beside it the differential (each letter's difference minus the item's mean) and the top-2 margin change, which are what can flip an answer, and the length and step count of the item with the largest difference, so a stop on the full run's long items reads itself. The chat MMLU path (`chat_ids`) is never exercised by the golden test; it is ungraded and common-mode across Q0-chat and QA-chat, so this is a note, not a change. Whether the batched pass is bitwise repeatable across runs is unmeasured (the first smoke raised before recording); not a gate matter.

## 3. Does bf16 batch noise threaten any graded prediction? No.

The conditions' MMLU passes are paired: every condition scores the same `rows` (all 14,042 questions in the full run; `run_model` line 843, `run_condition` line 810) in the same order at `mmlu.batch` = 32 with right padding, so batch composition, padding and kernel path are identical across conditions, and `within_mmlu` refuses unequal n. No condition ever runs at batch 1, so the batch-32-against-batch-1 discrepancy the golden test measures never enters any comparison.

The noise does not cancel exactly: an intervention changes the activations, so each condition has its own rounding realization, and the near-tie items are coin flips with respect to it. But that is bounded by the near-tie fraction. With 6-10% of items within one step of a tie and 12-20% within two (Qwen, Ministral), the rounding-only sd of an accuracy difference between two conditions is about 100 x sqrt(0.5 f / N) = 0.2-0.3 points at N = 14,042, with zero expected sign. The 2-point bands (c3, c3_r) and the 1-point bands (e_rand, e_mb_rand) are 4-10 sd wide against it. On the smoke's 50 items one flip is 2 points, which is why the smoke is read as nothing; that stays right.

## 4. Stop arithmetic: right, but the stop does not bind where the proposal puts it.

Estimates from spend.json (wall time at config.yaml's rates, "not Modal's bill"): prepare $0.0061, first smoke $0.8468, rerun $0.8527; total $1.7056; room $28.29. The proposal's rounding gives the same figure.

But `Spend` keeps spend.json per run folder (`--results` for the full run, `--results/smoke` for a smoke), and enforces `cost.stop_usd` on that folder's own total. The full run's folder starts at $0 against $30. For $28.29 to bind it must be written into `cost.stop_usd`, minus the smoke rerun under the new gate, which now passes the golden test and runs the conditions and judges ($3.5-4.5 by the registration). That leaves about $24-25 against a timeout sum of 2.5 + 2.0 + 0.75 + 1.0 hours at $3.9 = $24.4, so `budgets()` will trim the timeouts; the lead should know that before launch rather than from the console.

## 5. What must change before the full run

1. Move the golden test to the start of `run_model` (after the facts and the first `assert_no_hooks`); keep "before any condition" in the registration. Keep gate 1 at 8 steps with the move; 12 without it.
2. Put the three thresholds (8 steps, 2 steps, 10%) in config.yaml's `mmlu` block so the plan carries them, and re-register the gate in NOTES.md with this check's date.
3. Add the Q0/M0 full-MMLU floor of 35% as a stop (the common-mode hole).
4. Record per item the differential and the top-2 margin change, and the length and steps of the largest-difference item; show them on the card beside `max_abs_diff`.
5. Tests, one planted case per part: a 9-step difference on one item; a flip with a 3-step margin; 7 flips of 64; and a passing case with a one-step tie flip. Update every "any difference stops" string: refusal.py's comment and message (960, 975), card.py's `golden_lines` (311), README line 84, config.yaml line 128.
6. Set `cost.stop_usd` to the room actually left at launch, after the smoke rerun.
7. The `fix_mistral_regex` question stays logged, not blocking: it is common-mode across arms and conditions, but it does bear on M0 against the 53.4 anchor, so the card's anchor row should carry the note.

## Verdict

Accept with the changes above: the gate's logic and the step unit are right and the evidence is numerics by the registered rule, but the golden test must move to the front of `run_model` (or gate 1 widen to 12 steps), a Q0/M0 MMLU floor must close the common-mode hole, and the $28.29 stop must be written into `cost.stop_usd` to bind.
