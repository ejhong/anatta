# Check 4: open questions for the design check (Opus 5.5, 1 Nov)

These are the decisions and uncertainties in check4_registration_draft.md, most important first, each with my
recommendation. The sources are in check4_facts.md. The first three matter most.

## The three that matter most

### 1. Which reading convention is graded
- **What's known.**
  - The repo reads each response with the trait's system prompt in context. The stored prompt is the chat-template
    rendering of [system, user].
  - The plan's convention 1 says to set probe's include_system to match. That means True.
  - The lab's ego vectors were read with include_system=False, a preregistered choice (docs/PREREGISTRATION.md:42).
  - False isn't "no system prompt". Qwen then reads with its default system line, and Ministral with its 521-token
    Le Chat prompt.
- **The risk.** A pass at True says nothing about the convention the ego was actually made with. A pass at False
  alone would be an unfair test of the arithmetic, because the context differs from the reference's.
- **The draft.** C4, at True, is in the verdict (65%). C4-lab, at False, is a graded forecast outside it (40%). E-ctx
  and E-M-ctx measure the context effect on identical responses, on Qwen and on Ministral.
- **Options.**
  - (a) As drafted.
  - (b) Put C4-lab in the verdict too. That is stricter, but a miss would then be a design difference reported as an
    instrument fault.
  - (c) Grade only at True, as the plan wrote it, and drop C4-lab.
- **Recommend (a).** The verdict asks whether probe implements the method. C4-lab asks whether the lab's
  preregistered variant gives the same vector. Both answers go on the card, and C4-lab's grade sits beside 53-56
  either way.

### 2. The reference: the verbatim script in a second image, or a port in the lab's image
- **What's known.**
  - generate_vec.py loads with no dtype, which under the repo's transformers 4.52.3 means fp32. The lab's
    transformers 5.21 would load bf16 ("auto").
  - The script keeps each forward pass's autograd graph. In fp32, that needs an 80 GB GPU for the longest responses
    (inference from §1 of the facts).
  - It can run unmodified, judge-free, if its CSVs carry constant score columns.
- **Options.**
  - (a) R is the verbatim script in a second image with the repo's pins, on one H100. R-lab32, the vendored
    functions in the lab's image in fp32, runs beside it and separates the stack from the conventions (E-stack).
  - (b) R-lab32 alone, with no new image. The stack is then never varied.
  - (c) Run both stacks fully, including generation through vLLM (question 6).
- **Recommend (a).** R is then the released code itself, with no port to defend, and E-stack gives the stack's size
  for free. The image is pip-only (about 4-5 GB, from memory), and R costs about $2.5. If Eugene declines, (b) is the
  fallback, and the card says the stack wasn't varied.
- **For the checker.** I read the script's functions and its argument parsing, but not its import lines one by one.
  If it imports anything beyond the repo's pins, the smoke run shows it, and that package goes to Eugene before the
  real run.

### 3. Add optimistic as a third trait
- **What's known.**
  - Figure 20 gives Qwen cross-trait cosines at layer 20. With only evil and sycophantic, there is one cell, 0.368.
    Adding optimistic gives two more: −0.472 with evil and 0.178 with sycophantic.
  - optimistic.json is 4,109 bytes, Apache-2.0, already covered by the 31 Oct approval.
  - It adds about a third of G's and the readings' cost, about $3.
- **Why it matters.** C4-base is the check's only tie to the paper's own vectors, as opposed to its code. One cell
  can't tell a matched geometry from a lucky number. Three cells, including a negative one, can.
- **Recommend yes.** If trimming, drop it. C4-base then becomes "evil–sycophantic within ±0.10 of 0.368", at about
  70%, and E-opt and E-geom go.

## Design choices a checker may want changed

### 4. The judged arm
- **What's known.**
  - The repo's filter needs gpt-4.1-mini scores from logprobs. OpenRouter's three gpt-4.1-mini endpoints list no
    logprobs.
  - openai/gpt-4o-2024-08-06 on OpenRouter does list them ($2.50/$10, about six times the price), but it is a
    different judge.
  - OpenAI's own API returns logprobs for gpt-4.1-mini (from memory), but that needs an OpenAI key.
  - I didn't check whether OpenRouter's OpenAI endpoints moderate these inputs. The evil completions could be
    blocked.
- **The draft.** No judge in any graded arm (Fable's judge-free version, and the plan's "or neither does"). A text-
  score arm runs only if C4-base misses, to test whether the filter explains the gap to Figure 20. It costs about $3,
  needs Eugene's OK, and its card line says it isn't the paper's scoring.
- **Recommend as drafted.** If Eugene has an OpenAI key, the arm should use OpenAI directly with the repo's judge.py
  verbatim (logprobs, top 20). That reproduces the paper's scoring.

### 5. The probe.py rule, and the recorder
- **What's known.**
  - "A standing rule keeps src/anatta/probe.py unedited" (NOTES.md:5941-5942). I didn't find where the rule was
    first set.
  - probe.extract returns only 6 of its responses and no per-response means. So the draft wraps
    anatta.probe.generate and anatta.probe.response_activations at run time to record them.
    - The wrappers are installed for one call, removed after it, and return the original outputs unchanged.
    - A fixture test asserts bitwise-equal outputs with and without them.
    - The bootstraps for C4 and C4-lab use the recorded means.
  - The trimmed and 128-token readings use probe.capture and models.chat_ids from the check folder.
- **Question.** Does the rule allow a run-time wrapper around a probe function?
  - If not, the wrappers go, and E-det and E-M-trail go with them.
    - E-M-trail could move onto G-style responses generated on Ministral, at a small extra cost.
    - Without the recorder, probe.extract's responses and per-response means can't be had at all. C4's and C4-lab's
      uncertainty would then rest on the seed spread alone, with no question bootstrap.
  - No grade depends on the wrappers, only some intervals and expectations.
- **Also confirm** that the rule covers only probe.py. The draft edits no library file at all.
- **Recommend** allowing the wrapper: the file and its behaviour are untouched, and the test proves it.

### 6. G's sampler: Hugging Face generate or vLLM
- **What's known.**
  - The repo sampled with vLLM 0.8.5.post1 under explicit SamplingParams: T 1.0, top_p 1, no top-k, no penalty,
    min_tokens 1, max_tokens 1,000, stop at eos. Qwen's generation_config was ignored.
  - Hugging Face generate in transformers 5.21 samples the same distribution if top_k 0, top_p 1.0 and the penalty
    1.0 are passed explicitly. If they aren't, it fills in Qwen's 20, 0.8 and 1.05.
- **Options.**
  - (a) Hugging Face generate in the lab's image. No new package; about 40-50 L40S-minutes.
  - (b) vLLM 0.8.5.post1 in the second image: the repo's exact code path, and faster. But vLLM is a large new
    package to approve and debug.
- **Recommend (a).** C4-code reads the same responses with both extractors, so the sampler can't affect it. For C4
  and C4-base, only the distribution matters, and (a) matches it. A test asserts the processors transformers builds.

### 7. C4-code's bar and what it includes
- **The draft.** P1 against R at ≥ 0.99, at block 19, 80%. P1 is bf16, includes the template's closing tokens and
  runs in the lab's stack. R is fp32, answer-only, in the repo's stack.
- **Alternative.** Grade the pure arithmetic instead (P1-trim against R-lab32 at ≥ 0.999) and report the
  conventions.
- **Recommend as drafted.** It grades the instrument as the lab used it. E-stack, E-dtype and E-trail already
  separate the conventions.
  - The main risk is the two closing tokens. A token with large activations at the end of each response could shift
    the mean when pos and neg lengths differ. That is why E-trail is 85% and C4-code 80%.
  - On Ministral, the closing token is "</s>". E-M-trail measures it.

### 8. The "noise-limited" verdict
- **The draft.** If C4 misses, C4-code passes, and the ceiling is below 0.9, the card's verdict is "inconclusive
  (noise-limited)". It is still a miss in the tally.
- **Is it an escape hatch?** It is fixed before the run, and the ceiling comes from two independent controls: a
  second probe seed and the reference's split halves.
- **Option.** Require E-ceiling (≥ 0.9, 70%) as a precondition, so that C4 is read only when the noise allows a
  0.8 band to mean something.
- **Recommend as drafted.** The ceiling is itself a finding for the ego. The lab's directions used 60 responses per
  sign; this check uses 100.

### 9. Ministral M′ now
- About $1, reported in the ledger only.
- It is the only part of the check on the cut's own model. It measures the two conventions (context and closing
  token) on the six ego traits and v_ego, and whether the stored directions reproduce today.
- **Recommend yes.** If trimming, drop E-M-trail first.

### 10. Bands and odds
- **C4's band, ≥ 0.8,** is the plan's (Fable: "With those matched, ≥0.8 is a fair band").
- **My own bands:**
  - C4-code's 0.99, from arithmetic that should agree up to bf16 and two tokens;
  - C4-base's ±0.10, from judge-free against judge-filtered vectors, plus sampling noise of about ±0.02 at 1,000
    responses per sign.
- **My odds, in brief:**
  - C4 at 65%: strong, prompt-elicited traits, with the ceiling likely about 0.95. But 128 tokens at T 0.7 against
    up to 1,000 at T 1.0 is a real difference.
  - C4-lab at 40%: the persona prompt is probably a large part of the response-token signal at middle layers.
  - C4-base at 55%: Qwen may refuse part of the evil prompts, and the unfiltered R keeps those refusals.
- A checker who reads Figure 20 as judge-filtered may want C4-base at ±0.15.

### 11. The noise model and the bootstrap unit
- R's split for c_R is by samples within each (question, instruction) cell, because both extractors share the
  questions. A split by question is reported too, but it counts the differences between questions as noise.
- The intervals come from a bootstrap over the 20 questions, rebuilding both directions on the same draw.
- **Recommend as drafted.**

### 12. Safety and storage
- G's evil completions, the recorded probe responses and every vector stay on the Modal volume.
- probe's meta["examples"] holds completions and is deleted inside the GPU function. The card holds numbers and
  hashes only.
- The judged arm, if it runs, sends completions to OpenAI through OpenRouter.
- **Recommend as drafted.**

### 13. Third-party vectors
- anthonym21/qwen-persona-vectors holds files of the right size for [29, 3584] fp32 vectors. It has no licence, no
  README, an unknown provenance, and it isn't the authors'.
- **Recommend not fetching them.** A match would prove little, and a mismatch would be unreadable.

### 14. Figure 20's provenance
- The figure doesn't say which saved vector it used or whether that vector was judge-filtered. The repo marks
  response_avg as "used in paper", and the default pipeline filters.
- That is why C4-base is a forecast outside the verdict.
- A one-line email to the authors could settle it. **Recommend** leaving that to Eugene, and not waiting on it.

### 15. The stop
- The plan said ~$10. My estimate is $10-14, and the stop is $20.
- Check 1 spent $22.82 against a planned $3-5.
- **Recommend $20,** with the spend printed after each GPU function.

## Smaller points to confirm

### 16. Branch base, overlap with check 3, and the cheap tier
- **Branch.** calib/persona-extractor from calib/instruments (dc32cf7). It doesn't depend on check 3's branch,
  which is being built now.
- **Overlap.** The only shared file is THIRD_PARTY.md. Both checks copy check 1's volume loader.
- **Cheap tier.** The draft adds one item to loop rule 4's list: probe readings on stored responses, a few
  GPU-minutes.
- **Recommend** all three as drafted.

### 17. Seeds
- probe.extract seeds each generation batch with seed + offset, and the offsets step by 16. So the second seed must
  not be a multiple of 16. The draft uses 1.
- G uses torch.manual_seed(1000 + the batch index). The repo set no seed.

### 18. R's GPU
- R runs on an H100 for memory, and the lab's arms run on L40S. E-stack therefore covers the GPU type as well as the
  software stack.
- fp32 matmuls don't use TF32 by default in either torch version (from memory), so differences should be around
  1e-6.
- The smoke run reports R's peak memory.

### 19. How the lab's Qwen directions were sampled
- If they were made under transformers 5.x, probe's Qwen sampling used top-k 20 and a 1.05 penalty from Qwen's
  generation_config. I didn't find a record of which version made them.
- This touches only the ledger line comparing approval_seeking with R for sycophantic. **No decision needed.**
