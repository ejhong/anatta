# Check 3: open questions for the design check (Opus 5.5, 1 Nov)

These are the decisions and uncertainties in check3_registration_draft.md, each with my recommendation. Sources are in
check3_facts.md. The first three matter most.

## The three that matter most

### 1. Wollschläger's SALAD splits have no license
- **What's known.** github.com/wollschlager/geometry-of-refusal (43a8628) has no LICENSE file, and the GitHub API
  reports none. The splits come from SALAD-Data (Apache-2.0) and Alpaca (CC BY-NC 4.0; their Table 7 says Apache 2.0,
  but the dataset card says CC BY-NC 4.0).
- **Why the released files matter.** Their script caps each source with `group.sample(n=256)` and no seed, so the
  exact training set can't be rebuilt. Only the released files reproduce their direction.
- **Options.**
  - (a) Fetch the four released files (3,701,501 bytes) to Modal for research use. Cite them by URL, commit and sha256;
    never redistribute them; keep them out of every repository.
  - (b) Rebuild seeded splits from SALAD-Data with their script's rules. The data are then licensed, but the sample
    differs, so the anchor is looser.
  - (c) Run both, with (b) as a reported variant.
- **Recommend (a),** with (b) as the fallback if Eugene says no. Either way, the Qwen rows can stay public, because
  the card holds only numbers and hashes. Alpaca's non-commercial terms allow this research use. The same question
  will come back when the instruments go public (testing-system plan, "Licences").

### 2. On Qwen 2.5 7B, the selection may be fragile
- **What's known.**
  - Wollschläger's Appendix D (Figure 16) says that for Qwen 2.5 7B "only one direction is found to be effective".
    The text gives no KL value or layer.
  - COSMIC (Table 2) found that Arditi's selection, run with COSMIC's own data, picks position −1, layer 15.
- **The risk.** Our stack differs from theirs (transformers 5.21 against 4.47.0, a different GPU, different kernels).
  If that one candidate's KL lands just above 0.1 in our stack, nothing passes the filters, and C3 fails for a
  reason that has nothing to do with the dial.
- **Options.**
  - Keep the registered rule: C3 fails if the library path finds no viable candidate, then the paper's stack
    localizes the failure.
  - Register a Qwen fallback (drop the induce filter, then KL 0.2), with its result reported but not graded.
- **Recommend** keeping the rule as registered, with every threshold fixed before any score is seen. Report all 140
  candidates' three scores, the chosen candidate's margins and the runner-up. Ask now for the paper's-stack image
  (fetch 8), so a localizer can run at once.
- A checker who prefers a pre-registered fallback can add one as "reported, not graded". It changes nothing about
  C3's verdict, and it would show whether the direction works even when the filter rejects it.

### 3. The paper leaves the decoding and MMLU settings partly unstated
- **Repetition penalty.**
  - The lab's stack (transformers 5.21) fills unset generation fields from the model's generation_config. So unless
    the code passes 1.0, greedy decoding on Qwen uses repetition_penalty 1.05.
  - Wollschläger's stack (4.47.0, with a GenerationConfig passed) copies only special-token ids from the model, so
    they decoded at 1.0.
  - Recommend 1.0 for every graded run, plus Q0-rp and QA-rp at 1.05, reported (two passes, about 6 GPU-minutes).
    That shows whether the lab's default matters on Qwen. Ministral's config has no penalty, so the lab's Ministral
    results aren't affected.
- **MMLU.**
  - Wollschläger don't say how they ran MMLU: harness, shots, chat template or subset. Their Appendix A.3 says
    TruthfulQA used lm-eval with chat templates. GRP-Obliteration reports 73.6 for the same model where Wollschläger
    report 68.8, so the level depends on setup.
  - Recommend lm-eval's 0-shot default prompt, no chat template, and letter-logit argmax over all 14,042 questions.
    Grade only the change (|Δ| ≤ 2), and report our level beside 68.8.
  - If the checker wants the level to mean something, add a chat-template pass on Q0 and QA (about 8 GPU-minutes),
    reported.

## Design choices a checker may want changed

### 4. Should C3-extract be graded?
- The task named C3-base, C3 and the Ministral check. I added C3-extract: the library extractor against Arditi's
  generate_directions at batch size 1, with cosine ≥ 0.9999 and norms within 0.1% at all 135 non-zero candidates.
- My reason: the likeliest bug is an off-by-one between Arditi's "layer l" (the stream entering block l) and the
  lab's (block l's output), or in the negative position offsets. C3 alone might not show it if the selection lands on
  a neighbouring candidate that works almost as well.
- At batch 1 both read the same unpadded forward passes, so the bar can be strict. Against the padded reference
  (batch 32), bf16 noise may show: Qwen has large activations at some template tokens. Those cosines are reported,
  not graded.
- **Recommend graded.** The alternative is a test with a pass condition that gates reading QA.

### 5. The judge port, its stack, and two quirks kept on purpose
- The port of strongreject_finetuned is verbatim (MIT). But it runs under transformers 5.21 and peft 0.21.2, not the
  stack that scored 9.1 and 69.2 (transformers 4.47.0; strong_reject at main, unpinned; peft unpinned).
- Quirk 1: each response is decoded with special tokens, so a literal "<bos>" lands inside the judge prompt and is
  read as Gemma's BOS token. I inferred this from the code; prepare checks it on a benign string.
- Quirk 2: long responses are cut from the left at 512 judge tokens. A reply that opens with a refusal and runs long
  loses its opening before it is judged.
- A faithful port keeps both quirks, because the published numbers include them. Reported: the share of responses
  truncated.
- **Options.** Run the original package in the old stack only on a C3-base miss (the plan's swap rule). Or always run
  it on Q0's and QA's completions as a port check (200 judge calls, about $0.50).
- **Recommend always.** The draft registers it as a gate (the judge port gate) that must pass before any grade is
  read: it calibrates the judge separately from the model. If Eugene declines fetch 8, the gate falls back to a golden
  test of the prompt string and token ids, and a C3-base miss can't then be localized to the judge.
- The gate's tolerances (mean within 1 point, every completion within 0.05) are my guess at bf16 noise across
  transformers 4.47 and 5.21 on Gemma-2B. A checker may want them looser per completion and tight on the mean.

### 6. Which JailbreakBench file
- Wollschläger's code reads dataset/processed/jailbreakbench.json, which their repository doesn't ship. Their loader
  and README point to Arditi's repository, whose processed file is JBB's 100 behaviours (16,512 bytes). That the two
  are the same file is inference.
- **Recommend** Arditi's file, with prepare checking that its 100 instructions equal JBB-Behaviors' harmful goals at
  886acc3, and stopping if they don't.

### 7. What mode B measures
- Mode B ablates the lab extractor's per-layer refusal vectors at the ego cut's band, so it calibrates the cut's
  geometry: which layers, and a different vector at each layer.
- The ego cut's own vectors came from probe.extract (persona style), and probe.extract is check 4's subject. Check 3
  doesn't touch it.
- **Recommend** as designed, with the card saying so in one line.

### 8. The Qwen band for mode B2
- Qwen has no self-stakes cut. I scaled Ministral's 17-26 of 34 layers to 14-21 of 28 by proportion (17 × 28/34 =
  14.0, 26 × 28/34 = 21.4).
- **Recommend** keeping QB2 as reported only. A checker may drop it; MB2, on Ministral, is the exact recipe.

### 9. Ministral's refusal tokens and system prompt
- No published choice exists for Ministral 3 8B. I chose "I" and "As" (1073, 4072) by analogy with Qwen, and the
  default Le Chat system prompt (check 1's default condition).
- If Ministral's refusals start with other tokens (e.g. "Sorry", or a markdown "**"), filtering keeps fewer prompts,
  the selection weakens, and MA understates the effect.
- **Recommend** as designed, reporting the base model's metric distribution on each split and the counts kept. A
  checker could add a numbers-only look at the most common first tokens of Ministral's baseline replies on harmful
  val: token ids and shares, no text.

### 10. The random controls
- QA-rand, QB-rand and MB-rand add three generation passes and two MMLU passes, about 20 GPU-minutes.
- They are the testing-system plan's standard controls. MB-rand is exactly the lab's 29 Oct cut_random recipe on
  Ministral, which put lying 11 points below base in v2_controls. A calibrated read of whether random ablation at that
  band is inert bears directly on that control.
- **Recommend** keeping all three. If trimming, drop QA-rand first.

### 11. Prediction 82's re-read
- It needs v_ego from the lab's directions, which are private, so its numbers go in the ledger only.
- No generation is needed; it takes seconds.
- **Recommend** keeping it. E-82r, the reproduction of 0.165 at L9, also checks that the moved extractor still gives
  the old numbers.

## Smaller points to confirm

### 12. Harmful train count
The paper says 1,184; the script takes 1,024; the file's size suggests about 1,175. Prepare counts it, and the count
goes into config.yaml before the run. No decision is needed, but the checker should know that paper and code
disagree.

### 13. Branch base
- The task names calib/instruments. Two other branches hold useful helpers:
  - calib/model-check, which check 6 branched from, has check 1's volume loader (HF_HUB_OFFLINE, pinned revisions)
    and the expandable_segments fix for out-of-memory batches;
  - calib/relearning has snapshot helpers.
- **Recommend** calib/instruments, as instructed. Copy those few functions into the check folder, naming their source
  commits, rather than merging branches.

### 14. lm-eval itself
- Check 7's first registration said the lab's log-likelihood scorer would be calibrated against lm-eval in check 3.
  Check 7's amendment then dropped lm-eval.
- By construction, bench/mc.py equals lm-eval's acc when each continuation is one token, which the code asserts, and
  lm-eval's template is golden-tested.
- Running lm-eval itself on Q0 would turn that equivalence into a measured agreement. The price is a new package
  (MIT) and its dataset loader.
- **Recommend** not running it in check 3 (depth over breadth). A checker who wants the measured agreement can ask
  for it on Q0 alone.

### 15. Ties and padding in MMLU
- lm-eval compares log-softmax values in the model's dtype and pads on the right. bench/mc.py takes the argmax of the
  four letter logits, first index on ties, with right padding.
- In bf16, near-ties can differ from lm-eval on a handful of items. That moves the level by hundredths of a point,
  not the graded change.
- **Recommend** as designed.

### 16. The cheap tier's rule
- The rule: within 5 points of this run's Q0 and QA, and QA ≥ 55.
- Greedy decoding at the same batch should repeat almost exactly. Check 1 found that batch composition alone moves
  greedy reasoning replies, so 5 points is loose on purpose.
- **Recommend** as written.

### 17. The cost stop
- The plan put check 3 at $20. My estimate is $12-18, and the stop is $30.
- Check 1 spent $22.82 against a planned $3-5, so the margin is deliberate.
- **Recommend** $30, with the spend printed after each GPU function.

### 18. The conditional runs
- QX runs only if the two selections differ. Mode A+ runs only if C3 fails, E-R holds and the selections agree.
  Apart from the judge port gate, the paper's-stack image runs only on the misses named in "What a result does".
- **Recommend** confirming these triggers as written, so that no condition is added after a result is seen.
