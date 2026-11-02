# Calibration check 4: Fable's cold design check (filed)

A fresh Fable agent (model fable) checked the C4 registration draft cold, with no memory of the session, before any
C4 code, fetch or run existed. Its report file was last written at 10:22:11 UTC on 1 Nov (the file's clock). It is
below, unedited, with its own header. Its scratch files are in the session scratchpad's fable_c4_design/ folder:
verified.md has its sources and line numbers, and dl/ holds the image of the paper's Figure 20 panel that it saved
from arXiv to read the cells.

Opus 5.5 checked two of its claims:
- At the source: generate_vec.py at b8e0f04 has no no_grad, inference_mode or eval, and calls from_pretrained with no
  dtype.
- In the library: probe.extract calls generate and response_activations as module globals (probe.py:179, 190).

Its change 1, about the label-shuffle null, is algebra. The flipped mean is R·(Σs_k)/N plus noise, so R's direction
survives. Opus 5.5 checked the algebra and agrees.

Opus 5.5 then revised the draft into the registration (NOTES, 1 Nov), taking every change. Two choices were Fable's to
leave open:
- the reference runs on an A100-80GB, with an H100 as fallback;
- the judged arm goes through OpenRouter, the lab's only API route.

The files it checked are kept beside this one: calibration-c4-registration-draft-as-checked.md (recovered exactly, by
undoing the revision's edits in order), calibration-c4-facts.md (with Fable's corrections appended at the end) and
calibration-c4-open-questions.md.

---

# Calibration check 4: Fable's cold design check

A fresh Fable agent checked the C4 registration draft (check4_registration_draft.md, Opus 5.5, 1 Nov) cold, with no
memory of the session that wrote it, before any C4 code, fetch or run existed. Read-only: nothing was edited,
committed, installed, downloaded into a repository, run on Modal or paid for. The three trait files were read for
structure and counts only; no completion text was read. Scratch files are in the session scratchpad's
fable_c4_design/ folder: verified.md lists every source with its line numbers; tokcheck.py and tokcheck2.py are the
offline tokenizer checks (made-up strings, cached Qwen and Ministral tokenizers); null_sim.py is a synthetic
simulation behind change 1; dl/ holds Figure 20's Qwen panel, fetched from arXiv to read its cells.

---

**Verdict: ready with changes.** The anchor, the two-part verdict (probe's arithmetic on shared responses, then
probe.extract end to end), the convention breakdown (stack, dtype, closing tokens, window, context), the ceilings
and the transfer list are the right shape, and nearly every fact in check4_facts.md checks out at the source: the
repo's prompts, tokens, layer index, dtype, mask, sampling, judge, licence history and file hashes; the paper's
Figure 20 cells; what probe.py and models.py actually do. The draft's claim that the plan's convention 1 is
misleading is right. But one registered control would misread on exactly the traits chosen (the label-shuffle null
keeps the reference's direction, so it is not a chance level and would falsely caveat the grades); one verdict path
contradicts the plan's loop rule 6; the closing-token effect is measured on 1,000-token answers when the lab's
vectors were read at 128; the smoke run cannot test the one thing that could stop R from running (memory at full
length); and the run-time wrapper around probe's functions needs Eugene's explicit OK. All fixable before launch at
under a dollar of extra cost.

## Changes, ranked

**1. The label-shuffle null is not noise-only; it keeps R's direction. Replace it with a split in which the signal
cancels.**
Registration: "Label-shuffle null. The pos/neg labels within each pair are flipped at random: 200 draws, seed 0.
Each draw gives a difference vector that is noise only. The 95th percentile of its |cos| with R is the chance level,
given how unevenly the activations spread." and "E-null: the label-shuffle null's 95th-percentile |cos| with R is
≤ 0.15 at block 19: 90%." and "E-null or E-cross misses: the cosines aren't specific to the trait, and the C4 grades
are read with that caveat."
Evidence: flipping pair k replaces Δ_k = x_k^pos − x_k^neg by s_k Δ_k. Writing Δ_k = R + ε_k, the flipped mean is
R·(Σ s_k)/N + Σ s_k ε_k / N. The first term is R itself, shrunk by about 1/√N but not rotated, and a cosine ignores
scale. So the null's |cos| with R is ‖R‖|z| / √(‖R‖² z² + E‖ε‖²) with z ~ N(0, 1): it rises with the trait's strength
instead of falling to chance. Synthetic run (fable_c4_design/null_sim.py: d 3584, 1,000 pairs, anisotropic noise):
95th-percentile |cos| 0.27, 0.28, 0.41, 0.66 at per-pair signal-to-noise 0.05, 0.1, 0.2, 0.4; a split with equal pos
and neg in each half gives 0.23, 0.19, 0.06, 0.06; random directions 0.03. Evil and sycophantic are strong traits by
construction, so the registered null would likely land between 0.3 and 0.6, E-null would miss, and the card would
carry a specificity caveat that is an artefact of the null.
Fix: 200 random partitions (seed 0) of the per-response means into two halves, each holding every (question,
instruction) cell's pos and neg responses in equal numbers (for G: a random 5/5 split of each cell's ten samples,
applied to both signs); the null vector is mean(half A) − mean(half B). The trait signal cancels exactly and the
anisotropy stays. Its 95th-percentile |cos| with R is the chance level; keep the 0.15 bar and the 90%. Run the same
null on E-sys's recorded means (100 per sign), where probe's noise is larger, and keep the random-direction null as
the isotropic floor. Say in the registration which means each null uses (P1 on G at block 19 and by layer).

**2. The card's "fail" on the systematic-gap path contradicts loop rule 6. Name the outcomes now.**
Registration: "Card C4's verdict is C4-code and C4 together." / "If C4 misses, C4-code passes, and the noise ceiling
(below) is under 0.9 for the trait that missed, the card's verdict is 'inconclusive (noise-limited)'. In the tally
it is still a miss." / "With a ceiling of 0.9 or more, the gap is systematic. ... The cut results keep their status.
But the card states that probe's vectors are probe-style, not persona-vector style, and by how much."
Evidence: plan-2026-10-31-calibration.md, loop rule 6: "A fail marks them 'provisional'". With C4 in the verdict,
C4-code pass + C4 miss + ceiling ≥ 0.9 is a card fail whose registered consequence is "keep their status": the card
and the rule disagree. The draft is right about the substance: that path is a design difference (probe samples at
T 0.7, top-p 0.95, top-k 20, penalty 1.05, 128 tokens; the repo at T 1.0, plain softmax, 1,000 tokens: facts §3,
verified), not an instrument fault. The C3 registration handled the same situation by naming a verdict
("inconclusive (the selection filter)... the cut results keep their status", NOTES.md:5608-5610).
Fix: register four card outcomes. Pass: C4-code and C4. Fail: C4-code misses (the reading arithmetic; rule 6
applies). Recipe mismatch: C4-code passes, C4 misses, ceiling ≥ 0.9 (the page label on 53-56 becomes "extractor
arithmetic calibrated (card C4); sampling convention differs from the persona-vectors recipe, cos X"; rule 6 does
not apply; whether that label also changes the preregistration's wording "the persona-vectors recipe" is Eugene's
call). Inconclusive: noise-limited. Say that the tally counts C4 as a miss in the last two, and say what happens
when one trait passes and the other misses (the stricter outcome). Give C4-lab's note beside 53-56 the same form.

**3. The closing-token effect is measured at the wrong length for the transfer claim.**
Registration: "E-trail: cos(P1, P1-trim) ≥ 0.99 at every block 7-26: 85%. This is the template's closing tokens."
and "If E-trail misses, the template's closing tokens move the vector. The lab's vectors were made with them, and
the card gives the size of the effect."
Evidence: probe.py:110-115 renders the assistant turn with add_generation_prompt=False, so the slice always ends
with the template's closing tokens; offline I confirmed Qwen adds '<|im_end|>' and '\n' and Ministral '</s>'
(tokcheck.py), and that an empty answer on Qwen leaves a slice of just those two tokens. Their weight in a response
mean is 2/(L + 2): under 1% for most of G's answers (up to 1,000 tokens at T 1.0), 1.5-4% for the lab's
128-token-capped answers, and far more for the short refusals an unfiltered evil pos set contains. E-trail on G
therefore understates the effect for the vectors the lab actually made by several times, and the Qwen side has no
measurement at the lab's length (E-M-trail does it on Ministral only, and only through M′).
Fix: add E-trail-128: read the recorded E-sys responses (200 per trait, on the volume once the recorder runs) with
and without the closing tokens through probe.capture and models.chat_ids, same bar (≥ 0.99 at every block 7-26),
odds your call (I'd put 70%); report the mean closing-token weight per sign. About a GPU-minute per trait. I'd also
take E-trail's 85% and C4-code's 80% down a little: nothing in the draft rules out a large activation on
'<|im_end|>', and the pos/neg length asymmetry is where it would bite.

**4. The smoke run cannot test R's memory; add a full-length pass.**
Registration: "R on the smoke CSVs, which also reports the peak GPU memory." / "2 questions × 1 instruction × 2
samples per trait, at 64 tokens" / "It loads the model in fp32 (the 4.52.3 default) and keeps every forward pass's
autograd graph, so it runs on one 80 GB GPU (H100)."
Evidence: generate_vec.py at b8e0f04 has no no_grad, inference_mode or eval (verified line by line, all seven
imports read); the saved activations scale with sequence length, `outputs` is rebound only after the next forward
returns so two graphs can be alive, and fp32 weights alone are 30.5 GB. A 64-token smoke measures none of this. My
rough count for an 1,100-token row is about 15 GB per graph (28 layers of 18,944-wide MLP intermediates plus the
152,064-wide logits), so 45-60 GB at peak: fits 80 GB, not 48, but it is an estimate, and the draft's "80 GB" is an
inference too.
Fix: the smoke's R run includes two synthetic rows whose answer is a benign made-up sentence repeated to 1,000
tokens (the longest G can produce) and records the peak. If it doesn't fit, R-lab32 becomes the reference, as the
draft already registers. An A100-80GB has the same memory and is cheaper on Modal from memory (I could not verify
prices within my allowed sources): Eugene's call.

**5. Add G′: the lab's sampling settings at the reference's sample size, so a systematic gap localizes.**
Registration: "With a ceiling of 0.9 or more, the gap is systematic. E-len shows how much of it the 128-token cap
explains. The rest comes from temperature, top-p, top-k and the penalty."
Evidence: "the rest" is attributed by subtraction from a 100-per-sign estimate, and E-len is measured on T 1.0
samples, so the window and the sampler are never seen together at the lab's settings. The facts (§3, verified in
the installed 5.21: generation/utils.py:2064-2106 merge order, 1466 and 1604-1612 warper conditions) establish
exactly what probe samples on Qwen: T 0.7, top-p 0.95, top-k 20, penalty 1.05, 128 tokens.
Fix: G′ = Hugging Face generate at those five settings, passed explicitly (the same processor test as G), 10
samples per cell, 2,000 per trait, read as P1 and P0. Register E-samp: cos(P1(G′), R) ≥ 0.8 at block 19 for both
traits, about 65%; report cos(P1(G′), P1-128) by layer. About 5 L40S-minutes per trait, under $1 in all. This is the
number that says what the lab's vectors are relative to the recipe, with a ceiling near 0.98 instead of 0.9, and it
makes the "recipe mismatch" outcome of change 2 readable.

**6. The recorder is a run-time rebinding of probe's module names; put it on the OK list and say so.**
Registration: "Each probe.extract call runs inside wrappers around anatta.probe.generate and
anatta.probe.response_activations. They are installed for the call and removed after it, and they return the
original functions' outputs unchanged."
Evidence: extract calls `generate(...)` and `response_activations(...)` as module globals (probe.py:179, 190), so
the only way a wrapper sees them is to rebind `anatta.probe.generate` and `anatta.probe.response_activations` for
the duration of the call: monkeypatching, not editing. The rule (NOTES.md:5941-5942) says the file isn't edited and
states no scope; the facts couldn't find its origin; C3 chose to set model.generation_config rather than touch
probe's behaviour. The draft raises this under open question 5 but not under "Needs Eugene's OK".
Fix: item 6 on the OK list: "a temporary rebinding of two anatta.probe names inside each probe.extract call,
returning the originals' outputs unchanged, with the fixture test". State that no grade depends on it; the
bootstraps for C4 and C4-lab, E-det, E-trail-128 and E-M-trail do, so a "no" turns those intervals into seed-spread
only and drops the three expectations.

**7. Safety: tracebacks, and Ministral's template refuses an empty response.**
Registration: "No text reaches the Mac, a repository, a log or the card." / "M′, on Ministral: probe.extract on
each of the six ego traits, seed 0".
Evidence: (a) a pandas or jinja error inside a GPU function prints the offending value into Modal's logs, which
persist in the dashboard; the "returns only numbers" test doesn't cover exceptions. (b) Offline (tokcheck2.py):
Ministral's chat template raises TemplateError on an assistant message with empty content ("Assistant message must
have a string or a list of chunks..."), while Qwen renders it. probe.generate sets no min_new_tokens, so an empty
sample is possible, and probe.response_activations would then abort the M′ call for that trait. It evidently didn't
happen in the 28-29 Oct runs; it is a tail risk that would cost the whole M′ arm.
Fix: wrap each GPU function so any exception is re-raised with a text-free message (stage, row index, exception
type). In M′, catch per trait, report the count, and carry on; no grade depends on M′.

**8. Smaller fixes.**
- Cheap tier: "below 0.9999 at any block 0-26" for a re-read of stored responses is tight for bf16 differences of
  means across possibly different kernels or drivers; make it 0.999 (the 0.99 line against R-lab32 is fine), or say
  "same GPU type".
- "the testing-system plan's standard controls (testing-system.md:46 and 82)": those lines name strength-matched
  random directions and rival directions, not seeds, split halves or shuffles. Cite them for the random-direction
  null and own the rest.
- OpenRouter: the model id is `openai/gpt-4.1-mini`; the dated name is the endpoint's (the dated URL resolves to the
  same model today). Record the snapshot from the response, not from the slug. The three endpoints and the absence
  of logprobs check out.
- E-det: say it tests determinism within one process (fair at 90%); the bridge's same-seed spread (facts §6) was
  across runs and stacks, a different question.
- Second image: torch 2.6.0 pulls NVIDIA's CUDA runtime wheels (proprietary, redistributable; the lab's image
  carries the same), so the licence list should name them, and the size is likely 6-8 GB rather than 4-5.
  Immaterial otherwise: generate_vec.py imports only tqdm, transformers, json, torch, os, argparse and pandas, so the
  pins plus accelerate suffice.
- Trait construction: anatta.probe.Trait also needs name and description (probe.py:31-38).
- "Any seed that isn't a multiple of 16 therefore draws new samples": a multiple of 16 would too, since each batch's
  prompts differ; the conclusion holds, the reason is just "different seeds per batch".
- Odds I'd move: E-trail 85 → 70, C4-code 80 → 75 (change 3). C4-base at ±0.10 and 55% is fine as a forecast; don't
  widen it, a miss is informative. E-null's 0.15 is fine once change 1 is made. The ceiling formula is right
  (reliability of E ≈ c_E, of R ≈ 2c_R/(1 + c_R), ceiling = the square root of their product).

## Answers to the open questions

1. (a) as drafted, with change 2's named outcomes. Grading at True is the faithful test of "does probe implement
   the method"; C4-lab is the lab's variant and belongs beside 53-56, not in the verdict.
2. (a): the verbatim script in the second image, with R-lab32 beside it. Add the full-length memory smoke (change
   4). The imports are now read: nothing beyond the pins and accelerate.
3. Yes, add optimistic. Three cells including a negative one are what make C4-base a test of geometry.
4. As drafted. If Eugene has an OpenAI key, run judge.py verbatim (logprobs, top 20) and the paper's scoring is
   reproduced; through OpenRouter it can't be (verified: no endpoint lists logprobs). The judge's text-number
   variant is new, not the repo's; the card should say so, as the draft does.
5. Allow it, with the fixture test, and put it on the OK list (change 6). The rule covers only probe.py; the draft
   edits no library file, and nothing else in src/anatta needs one.
6. (a) Hugging Face generate. I verified that the repo's vLLM run ignored Qwen's generation_config (a caller's
   SamplingParams is used as given; update_from_generation_config touches only eos ids) and that 5.21 builds no
   warper at temperature 1.0, top_k 0, top_p 1.0, penalty 1.0, and accepts top_k 0 with do_sample (the only top_k
   check is for contrastive search). The processor test is the right guard.
7. As drafted, plus E-trail-128 (change 3), because the transfer claim is about 128-token reads.
8. As drafted, but as a named outcome (change 2). Reporting cos/ceiling is the right companion.
9. Yes, keep M′; it is the only measurement on the cut's own model. Catch the empty-content crash (change 7).
10. Bands fine. C4 65% is fair for strong prompt-elicited traits; C4-lab 40% and C4-base 55% are reasonable guesses.
11. Fix the null (change 1). The sample split for c_R and the question bootstrap are right.
12. As drafted, plus the traceback point (change 7).
13. Agree: don't fetch them.
14. Agree: don't wait. An email is cheap and would settle both the filter and the saved-vector question.
15. $20 is fine with the spend printed per function.
16. All three as drafted.
17. Seeds fine (see the small note in change 8).
18. H100 or A100-80GB; E-stack covers the GPU type. TF32 is off for matmul by default in both torch versions as far
    as I know; E-stack would show it if not.
19. No decision needed.

## Facts in check4_facts.md found wrong or imprecise

Nothing material is wrong; the one conceptual error is in the draft (the null, change 1), not the facts. Small
imprecisions:
- §1: the layer list is `max_layer = model.config.num_hidden_layers` then `layer_list = list(range(max_layer+1))`,
  equivalent to the facts' line. The file's seven import lines, which the facts say were not read one by one, are
  tqdm, transformers (AutoModelForCausalLM, AutoTokenizer), json, torch, os, argparse, pandas.
- §3: model_utils' LLM call also passes enable_lora=True, tensor_parallel_size=torch.cuda.device_count() and
  max_lora_rank=128. Immaterial.
- §5: Ministral's default prompt: I count 536 tokens for the whole prompt with a 10-token question, consistent with
  "521 by check 1's config, 533 with a short question".
- §3/§7: the lab's Qwen directions carry no transformers version in their meta, so §3's "what probe actually
  samples under 5.21" is right for today's runs and unknown for the stored Qwen directions (open question 19 says
  as much).
- §4: the coherence prompt's wording ("opens with the first sentence of Betley et al.") was not checked.
- §2: "a third party has published Qwen vectors" (in the draft's safety section) overstates a licence-less dataset
  with no README; the facts describe it correctly.

## Verified and right

- The plan's conventions. (1) generate_vec joins `p + a` with no template (verified), but `p` is eval_persona's
  `prompt` column, the full apply_chat_template rendering with the trait system turn (verified in eval_persona.py;
  and the shipped baseline CSV's prompt column begins "<|im_start|>system" with Qwen's default line); the repo ships
  no extraction CSV (tree at b8e0f04: only output/qwen2.5-7b-instruct_baseline.csv, no .pt). So "read off its CSV"
  can't be done and "no chat template" is misleading as a whole: the draft's restatement is right, and
  include_system=True is the faithful graded arm. (2) hidden_states[0] is the embedding, [i] block i−1, [28] after
  the final norm (modeling_qwen2.py at 4.52.3); the paper's "layers starting at index 1" and "layer 20 for both
  evil and sycophancy" (App. B.4); the steering hook at layer_idx = layer − 1: the repo's layer 20 is probe's block
  19, and index 28 rightly isn't compared. (3) the mask is paired by row and gates coherence ≥ 50 on both sides
  (verified); probe's judge is per response, one threshold, no coherence (probe.py:183-186); the paper's §2.2 filter
  (> 50 / < 50) names no coherence; constant score columns keep every row without changing a line of the script.
- The repo's recipe: no dtype (fp32 under 4.52.3's `torch.get_default_dtype()` branch, verified), no no_grad,
  `prompt_len = len(encode(p, add_special_tokens=False))`, slice `[:, prompt_len:, :].mean(dim=1)`, `.mean(0).float()`
  difference, the three output files, read_csv with defaults, tqdm. The script's two commits (10facfdb, 55b6e591);
  the licence added by the head commit b8e0f04 on 22 Apr 2026, one commit on LICENSE, "Closes #17".
- eval_persona.py: a_or_an, the system f-string, questions outer and instructions inner (100 Questions, 1,000 rows,
  question → instruction → sample), SamplingParams(temperature 1.0, top_p 1, max_tokens 1000, min_tokens 1,
  stop=[eos], skip_special_tokens=True), no seed, no strip; generate_vec.sh's pos/neg runs with assistant_name evil
  and helpful, threshold 50. vLLM 0.8.5.post1 defaults (top_k −1, penalty 1.0) and the None-only default merge.
- Trait files: sizes and blob shas for evil, sycophantic and optimistic; 5 instruction pairs (pos/neg), 20
  questions, an eval_prompt with {question} and {answer}; the other four files' sizes.
- The judge: gpt-4.1-mini-2025-04-14, max_tokens 1, temperature 0, logprobs with top 20, seed 0, weighted mean of
  0-100, None below 0.25 total probability. OpenRouter: three endpoints at $0.40/$1.60 and $0.44/$1.76 per million,
  none with logprobs.
- Figure 20's Qwen panel, read from the image: "(Layer 20)"; evil–sycophantic 0.368, evil–optimistic −0.472,
  sycophantic–optimistic 0.178, and every other cell in the facts' table. The text states no layer and no filter.
- Qwen a09a354: generation_config do_sample, T 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05, eos [151645,
  151643]; 28 layers, hidden 3584. Ministral f6fae97: generation_config holds only token ids and max_length.
- probe.py (sha256 fc0b5692…, unchanged since ee690a3 on 27 Oct; models.py 99d3a916…, factor.py 51143421…,
  modal_app.py 6c2034f5…, dial.py 46b5ebcb… match the facts): generation always carries the system prompt (178),
  include_system only at reading (187-190), seeds seed + s per batch of 16 (176-179), top_p 0.95 hard-coded and no
  top_k or penalty passed (133-140), the slice includes the closing tokens (110-115; offline: '<|im_end|>', '\n' on
  Qwen, '</s>' on Ministral; prompt ids identical to the repo's except the leading-newline merge, which both methods
  share), bf16 store then .float() before the mean (115), layers 7-26 on Qwen and 8-32 on Ministral (167), unit
  vectors (211), meta examples (226-229). The CLI's extract passes no include_system and no judge. The 5.21 merge
  order (model's generation_config, then global defaults, then kwargs) and the warper conditions; dtype None → "auto".
- The lab's directions (meta only): all unjudged, include_system False, 120/120; Qwen ego d 0.20-0.85, formality
  4.22, verbosity 2.61, sage_persona 2.74, v_ego best L8 d 0.22; Ministral best layers and d as the facts' table,
  v_ego best L15 d 0.75. The bridge re-extraction: approval_seeking 0.956, face_saving 0.907, goal_fixation 0.826;
  v_ego 0.907-0.941, 0.931 at L20; self-stakes 1.000. modal_bridge.py:180, cut_vs_push.py:54-60, player.py:203,
  modal_em.py:99-131 (last prompt token), docs/PREREGISTRATION.md:42, NOTES.md:5941-5947, plan decisions 4 and 5
  (Qwen rows public; "the persona-vectors repo (Apache-2.0)" on the approved fetch list), the cost basis at c7 code
  check:155, check 1's offline loader at app.py:171-185.
- Arithmetic and sizes: 20 × 5 × 2 = 200 responses per probe.extract call; 12 calls per model; 2,000 G responses
  per trait; 1/√3584 = 0.0167; the ceiling formula; the question bootstrap rebuilding both directions.

## Plain reading

The check is well built: it asks the two right questions (does probe read the same vector from the same responses;
does probe's whole recipe land on the released recipe's vector), and it has a dial for every convention that
differs. Four things need fixing before it is filed: the "chance level" control would flag strong traits as
non-specific by construction; a mismatch caused by probe's sampling settings is registered as a card fail that
keeps every result's status, which the plan's rules don't allow; the closing-token effect that matters for the
lab's vectors is measured on answers eight times too long; and the dry run can't tell whether the reference script
fits on the GPU. With those, plus a cheap arm that runs the lab's exact sampling settings at full size and a line
on the OK list for the wrapper around probe's functions, a pass would mean what the draft says it means, and every
miss would point at one step.
