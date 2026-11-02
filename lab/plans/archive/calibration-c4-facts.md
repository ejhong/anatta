# Check 4 facts: probe.extract against the persona-vectors extractor (Opus 5.5, 1 Nov)

These are facts only, gathered for the check 4 registration draft. Each fact gives its source and one of three marks:
- **verified (read)**: I read the file, page or figure at the stated commit or path in this session;
- **verified (run)**: I computed it in this session, offline on the laptop, with ~/prj/matlab/.venv (transformers
  5.20.2, HF_HUB_OFFLINE=1). The inputs were either cached tokenizers or the numeric metadata of stored direction
  files;
- **from memory**: I did not read it in this session.

Nothing was downloaded into a repository or the scratchpad, built, run on Modal or paid for. Web pages were read
with WebFetch. One of them, the Figure 20 image, stays in the session's tool-results cache. No completion or trait
text is quoted. For the trait files, only structure, key names and counts are given.

Short names used below:
- "the repo" is github.com/safety-research/persona_vectors at b8e0f044fe2410a6fad579f38324f03f13b4e917, its head on
  1 Nov;
- "the library" is /Users/eugene/prj/align/anatta-calib, on branch calib/instruments at
  dc32cf75f0f0d44bd13544d5ae42c6e6ddf69afa, with a clean tree;
- "the paper" is Chen et al., "Persona Vectors: Monitoring and Controlling Character Traits in Language Models",
  arXiv 2507.21509 (v1 29 Jul, v2 31 Aug, v3 5 Sep 2025).

## 0. The plan's three conventions, checked

The row is at plans/plan-2026-10-31-calibration.md:58, and Fable's re-check item 2 at
plans/plan-2026-10-31-calibration-fable-reviews.md:72.

1. **"generate_vec joins the stored prompt and the response with no chat template."** This is true of the join and
   misleading as a whole.
   - generate_vec.py concatenates two strings (`p + a`) and calls no template.
   - The stored prompt was already rendered by the chat template in eval/eval_persona.py, with the trait's system
     prompt as its system turn (§1, §3).
   - The repo ships no extraction CSV, so "read off its CSV" can't be done. This fact comes from the code. The format
     of the prompt column comes from the one shipped CSV, the eval-set baseline (§2).
   - To match, probe must read with include_system=True and the same system strings. The lab's ego directions were
     read with include_system=False (§7).
   - docs/PREREGISTRATION.md:42 describes the lab's recipe as "the persona-vectors recipe … read without the system
     prompt in context". On this point it departs from the released code (§6).
   - Verified (read).
2. **"The repo's 'layer 20' is hidden_states[20], probe's block 19."** Correct (§5). Verified (read).
3. **"Both use the same judge filter (positive ≥50, negative <50, coherence ≥50), or neither does."** The thresholds
   are right, but "the same filter" can't be had through probe.
   - The repo's mask works on aligned rows: a pos/neg pair is kept or dropped together, and coherence gates both
     sides.
   - probe's judge hook scores each response alone, against one threshold, with no coherence gate.
   - The paper's judge scores from logprobs, and none of OpenRouter's gpt-4.1-mini endpoints list logprobs (§4).
   - Verified (read).
- The repo calls the trait "sycophantic", not "sycophancy". Verified (read).

## 1. generate_vec.py's exact recipe

Source: the repo's generate_vec.py (5,250 bytes, git blob d819f48e…). The file's last commit, 55b6e59, changed only a
print line. All verified (read), unless marked otherwise.
- **Functions.**
  - get_persona_effective(...) reads the two CSVs and applies the mask.
  - get_hidden_p_and_r(...) runs the forward passes. Its layer list defaults to every hidden state.
  - save_persona_vector(...) loads the model and writes the vectors.
  - `__main__` parses --model_name, --pos_path, --neg_path, --trait, --save_dir and --threshold (default 50), then
    calls save_persona_vector.
  - I read the functions' bodies and the argument parsing. I didn't read the import lines one by one.
- **Load.** `AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")`, with no dtype argument.
  - Under transformers 4.52.3, the repo's pin, torch_dtype=None falls back to torch.get_default_dtype(), which is
    fp32. Verified (read) in transformers v4.52.3, src/transformers/modeling_utils.py.
  - The script calls no model.eval(). from_pretrained returns the model in eval mode (from memory), and Qwen2's
    dropout is 0 in any case.
  - There is no torch.no_grad, so each forward pass keeps its autograd graph.
  - There is no batching and no padding: one forward pass per text.
- **Data.**
  - pd.read_csv with defaults. An empty answer would read as NaN, and `p + a` would then raise (inference).
  - The mask is `pos[trait] >= threshold & neg[trait] < 100 - threshold & pos["coherence"] >= 50 &
    neg["coherence"] >= 50`. It is applied to both frames by row position. A NaN score fails its comparison, so the
    row is dropped.
  - The threshold defaults to 50, and scripts/generate_vec.sh passes `--threshold 50`.
- **Reading, per (prompt, answer).**
  - `text = p + a`
  - `inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)`
  - `prompt_len = len(tokenizer.encode(p, add_special_tokens=False))`
  - `outputs = model(**inputs, output_hidden_states=True)`
  - The layers are `layer_list = list(range(model.config.num_hidden_layers + 1))`: 29 entries on Qwen, 0 to 28.
  - For each layer l:
    - response_avg = `outputs.hidden_states[l][:, prompt_len:, :].mean(dim=1).detach().cpu()`;
    - prompt_avg = the same over `[:, :prompt_len, :]`;
    - prompt_last = `[:, prompt_len-1, :]`.
  - The mean is taken in the model's dtype, fp32 under the repo's stack.
  - `outputs` is rebound on each iteration, so the previous graph can still be alive while the next forward runs
    (inference).
- **The vector.**
  - For each layer: `pos.mean(0).float() - neg.mean(0).float()`, raw and not normalized, stacked to
    [num_hidden_layers + 1, d] = [29, 3584] on Qwen.
  - It is saved as {trait}_response_avg_diff.pt, commented "used in paper", beside {trait}_prompt_avg_diff.pt and
    {trait}_prompt_last_diff.pt.
- **The script.** scripts/generate_vec.sh:
  - runs eval/eval_persona.py twice: pos with `--persona_instruction_type pos --assistant_name <trait>`, neg with
    `--persona_instruction_type neg --assistant_name helpful`, both with `--version extract` and judge
    gpt-4.1-mini-2025-04-14;
  - writes eval_persona_extract/Qwen2.5-7B-Instruct/<trait>_{pos,neg}_instruct.csv;
  - then runs generate_vec.py with save_dir persona_vectors/Qwen2.5-7B-Instruct/.
- **Pins.** The repo's requirements.txt: transformers 4.52.3, torch 2.6.0, vllm 0.8.5.post1, accelerate 1.7.0,
  pandas 2.3.1, tqdm 4.67.1.

## 2. Data, CSVs, released vectors and licences

- **Licence.**
  - The repo's licence is Apache-2.0, added by its head commit b8e0f04 on 22 Apr 2026 ("Add Apache-2.0 license";
    "Copyright 2025 The Persona Vectors authors"). Before that commit the repo had no licence.
  - It has one branch and no releases.
  - Verified (read) through the GitHub API and the LICENSE file.
- **Trait files.** data_generation/trait_data_extract/ holds seven files. Bytes and git blobs:
  - apathetic.json: 4,316 bytes, blob 69a37bc1…
  - evil.json: 3,943 bytes, blob 9fab8d12fad3eb36e73498d576ccd9182bde772d
  - hallucinating.json: 4,454 bytes, blob 656dd863…
  - humorous.json: 3,285 bytes, blob 4ab98608…
  - impolite.json: 3,453 bytes, blob 8522d099…
  - optimistic.json: 4,109 bytes, blob 748217a142d46dee9449bbf08c344b27244768de
  - sycophantic.json: 5,428 bytes, blob b0d4367d138dd714e3f0d3072466753a74c9b8a0

  data_generation/trait_data_eval/ holds the same seven names. Verified (read).
- **Structure.** Each file has three keys:
  - `instruction`: a list of 5 objects, each with `pos` and `neg`;
  - `questions`: a list of 20;
  - `eval_prompt`: one string of about 900 characters, with `{question}` and `{answer}` placeholders.

  I checked this for evil, sycophantic and optimistic, and for the eval set. Verified (read): structure and counts
  only.
- **Shipped CSV.** The only CSV in the repo is output/qwen2.5-7b-instruct_baseline.csv (579,557 bytes).
  - Its columns are question, prompt, answer, question_id, evil and coherence. The ids have the form evil_<i>.
  - Its prompt column holds the whole chat-template rendering, including Qwen's default system line. That is the
    format eval_persona.py stores.
  - It is an eval-set baseline with no persona prompt, not an extraction CSV.
  - Verified (read): structure only.
- **Other files.** No extraction CSVs and no vectors are in the repo. dataset.zip (60,795,286 bytes) holds the
  fine-tuning datasets, which are for check 5. Verified (read).
- **Released vectors.**
  - The authors have released none on GitHub or Hugging Face that I could find. Verified (read) by searching Hugging
    Face model and dataset listings.
  - Third-party copy: anthonym21/qwen-persona-vectors, a dataset at sha bdc27395…, created 2026-02-02. It has no
    licence and no README.
    - evil_response_avg_diff.pt is 417,063 bytes, and sycophantic_response_avg_diff.pt is 417,098 bytes. That size
      fits a [29, 3584] fp32 tensor plus pickle overhead (29 × 3584 × 4 = 415,744).
    - Its provenance is unknown.
    - Verified (read): listing only, nothing downloaded.
  - jeqcho/subtle-gen-persona-vectors holds other traits (animal preferences). Verified (read): listing.

## 3. Generation settings in each stack

- **The repo: eval/eval_persona.py.** Verified (read).
  - The prompt is `tokenizer.apply_chat_template([system, user], tokenize=False, add_generation_prompt=True)`.
  - The system string is `f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"`.
    assistant_name is the trait for pos and "helpful" for neg. a_or_an gives "an" before a vowel and "a" otherwise.
  - The user turn is the question.
  - n_per_question is 10. Temperature is 1.0 whenever n_per_question > 1 (0.0 only when it is 1).
  - Sampling is vLLM's SamplingParams: temperature 1.0, top_p 1, max_tokens 1000, min_tokens 1,
    stop=[tokenizer.eos_token] and skip_special_tokens=True. No seed is set.
  - Answers are stored as decoded and not stripped. All prompts go through one generate call. CSV rows run question
    → instruction → sample.
  - Steering (not used here) adds vector[layer] through ActivationSteerer(layer_idx=layer−1), a forward hook on
    blocks[layer−1].
- **The repo: eval/model_utils.py.**
  - `LLM(model, enable_prefix_caching=True, max_num_seqs=32, gpu_memory_utilization=0.9, max_model_len=30000)`. It
    passes no dtype (vLLM's "auto", which is bf16 for Qwen) and no seed.
  - Verified (read).
- **vLLM 0.8.5.post1.** Verified (read) at tag v0.8.5.post1: vllm/sampling_params.py and vllm/entrypoints/llm.py.
  - SamplingParams defaults include top_k −1 (off) and repetition_penalty 1.0.
  - LLM.generate uses the model's generation_config sampling defaults only when sampling_params is None. The repo
    passes explicit SamplingParams, so Qwen's top_k 20, top_p 0.8 and repetition penalty 1.05 were not applied.
  - The repo's generation therefore sampled from the plain softmax at T 1.0, with no top-k, no top-p and no
    penalty.
- **The library: probe.generate** (src/anatta/probe.py:118-141). Verified (read).
  - It calls torch.manual_seed(seed), renders with apply_chat_template(add_generation_prompt=True, tokenize=False),
    pads on the left and tokenizes with add_special_tokens=False.
  - It passes do_sample = T > 0, temperature 0.7 (the default; probe.extract doesn't pass one), top_p 0.95
    (hard-coded), max_new_tokens 128 and pad_token_id. It passes no top_k, no repetition_penalty and no seed per
    sequence.
  - It decodes with skip_special_tokens=True and does not strip.
  - probe.extract calls it in batches of 16, with seed = seed + s, where s is the job offset (0, 16, 32 and so on)
    (probe.py:175-179).
- **transformers 5.21.0** (the library's image pin; /Users/eugene/prj/align/anatta/.venv). Verified (read).
  - `_prepare_generation_config` (generation/utils.py:2064-2124) applies the model's generation_config to fields
    left unset, then the global defaults, then the caller's kwargs, which take priority.
  - Warpers are skipped when temperature is 1.0, top_k is 0 or None, or top_p is 1.0 (utils.py:1604-1614). The
    repetition penalty is skipped at 1.0 (utils.py:1466).
  - The global defaults include top_k 50 and repetition_penalty 1.0.
  - from_pretrained's dtype defaults to "auto": bf16 for Qwen and for Ministral.
- **Qwen/Qwen2.5-7B-Instruct at a09a35458c702b33eeacc393d103063234e8bc28.**
  - generation_config.json: do_sample true, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05,
    eos_token_id [151645, 151643].
  - 28 layers, hidden size 3584.
  - It is already on anatta-hf-cache (approved 31 Oct; used by card C1; NOTES.md:3812).
  - Verified (read).
- **What probe actually samples on Qwen under 5.21** (inference from the facts above): T 0.7, top_p 0.95, top_k 20
  and repetition penalty 1.05, the last two from Qwen's generation_config, with at most 128 new tokens.
- **mistralai/Ministral-3-8B-Instruct-2512-BF16 at f6fae9795746f63c9be8344932f01275f3c63734.**
  - generation_config.json holds only token ids, so probe's sampling there is T 0.7, top_p 0.95, top_k 50 (the
    global default) and repetition penalty 1.0.
  - 34 text layers, hidden size 4096.
  - Verified (read).
- **modal_app.py image** (the library, src/anatta/modal_app.py).
  - Pins: torch 2.14.1, transformers 5.21.0, peft 0.21.2, among others.
  - Volumes: anatta-hf-cache and anatta-runs. GPU: L40S.
  - Verified (read).
- **transformers 4.52.3, Qwen2** (src/transformers/models/qwen2/modeling_qwen2.py at v4.52.3). Verified (read).
  - hidden_states[0] is the embedding output, and hidden_states[i] is block i−1's output.
  - The last entry, index 28, is taken after the final norm.

## 4. The judge, and OpenRouter

- **The repo's judge** (judge.py, eval/prompts.py). Verified (read).
  - Model gpt-4.1-mini-2025-04-14, called with one user message: the trait's eval_prompt or the coherence prompt,
    filled with the question and answer.
  - max_tokens 1, temperature 0, logprobs on with top_logprobs 20, seed 0.
  - The score is the probability-weighted mean of the integer tokens 0-100 among the top 20. It is None if those
    tokens' total probability is below 0.25.
  - The coherence prompt, coherence_0_100, is about 1,200 characters and opens with the first sentence of Betley et
    al.'s coherence prompt.
- **The paper, §2.2 and its appendices.** Verified (read).
  - The judge is GPT-4.1-mini, scored as a weighted sum over the top-20 logits. It agreed with humans 94.7% of the
    time.
  - The filter keeps trait score > 50 for pos and < 50 for neg. §2.2 doesn't mention coherence; the code gates
    coherence ≥ 50 on both sides.
- **OpenRouter, openai/gpt-4.1-mini** (per-model endpoints listing, 1 Nov). Verified (read).
  - The 2025-04-14 snapshot is served on three endpoints: OpenAI and Azure at $0.40/$1.60 per million input/output
    tokens, and Azure swedencentral at $0.44/$1.76.
  - None of the three lists `logprobs` or `top_logprobs` among its supported parameters, so the paper's scoring
    can't be reproduced through OpenRouter.
  - openai/gpt-4o-2024-08-06 does list both, at $2.50/$10.
  - I did not check whether these endpoints moderate their inputs.
- **The library's judge hook** (probe.py:149-165, 183-186). Verified (read).
  - `judge(trait, question, response) -> 0..100`. A response is kept if `(sign > 0) == (score >= threshold)`.
  - Each response is judged alone, with one threshold and no coherence score.

## 5. probe.extract against generate_vec, line by line

Sources: the library's src/anatta/probe.py and src/anatta/models.py, and the repo's generate_vec.py. Verified (read)
unless marked otherwise.

| Step | probe.extract (probe.py) | generate_vec.py |
|---|---|---|
| Responses | Generates its own: for each question and each pair, one pos and one neg, at the §3 settings, max 128 tokens (169-179) | Reads stored CSV rows; the repo makes them at T 1.0, max 1,000 tokens, 10 per (question, instruction) |
| Questions × prompts | Trait.questions × Trait.pairs (a dataclass, 31-38, loaded from YAML by Trait.load, 40-46) | 20 questions × 5 instructions from the trait JSON |
| Filter | Optional per-response judge (183-186) | Paired mask with coherence (§1) |
| Read context | `[system]` if include_system else `[]`, then `[user]` (187-190). Generation always uses the system prompt | The stored prompt: the chat template with the system prompt |
| Tokens read | chat_ids(msgs), then chat_ids(msgs + assistant, add_generation_prompt=False); start = len(prompt); end = len(full), so the template's closing tokens are included (104-115) | prompt_len = len(encode(p)); slice [prompt_len:] of tokenize(p + a): the answer only |
| Site | Forward hook on blocks[i], element 0 of its output (capture, 84-101) | hidden_states[l] from output_hidden_states |
| Layer index | Block i | hidden_states[l] = block l−1 (l ≥ 1); index 0 is the embedding; index 28 is after the final norm |
| Layers | Default range(n//4, n−1): blocks 7-26 on Qwen, 8-32 on Ministral (167) | All 29 hidden_states |
| dtype | Model in bf16 (models.load, 19-36: dtype "auto" means bf16 off CPU; dtype="float32" is accepted); store sliced, then .float() before the mean | Model in fp32 (the 4.52.3 default); mean in fp32 |
| Grad | @torch.no_grad (104) | None (graph kept) |
| Batching | Generation in batches of 16 with left padding; reading at batch 1, unpadded | Batch 1, unpadded |
| Mean | Per response, over the token slice, then over responses | Same |
| Difference | mean(pos) − mean(neg) per layer (192-196) | pos.mean(0).float() − neg.mean(0).float() |
| Output | Unit-normalized vector per layer (210-211); split-half Cohen's d per layer (197-208); pos/neg mean projections; meta, including the first 6 jobs' texts with completions (212-231) | Raw [29, d] tensor; also prompt_avg and prompt_last |
| Seed | torch.manual_seed(seed + s) per generation batch | None (vLLM, unseeded) |

- **Offline token check.** Verified (run) with scratchpad tokcheck.py, now deleted, using made-up benign strings.
  - On the Qwen tokenizer at a09a354: probe's slice is generate_vec's slice plus the tokens ['<|im_end|>', '\n']. On
    Ministral: plus ['</s>'].
  - The prompt ids are identical between the two methods.
  - An answer that begins with "\n" merges with the template's final "\n" when tokenized jointly. Both methods then
    start their slice one token late in the same way, so they still agree with each other.
- **Default system prompts.** Verified (run).
  - With include_system=False, Qwen's template inserts "You are Qwen, created by Alibaba Cloud. You are a helpful
    assistant."
  - Ministral's template inserts its default Le Chat system prompt: 521 tokens by check 1's config, 533 with a short
    question.
- **Test fixture.** The library's tests/conftest.py fixture is a tiny Llama. Its template appends "<|end|>" after the
  assistant turn, so probe's slice there has 1 trailing token. tests/test_probe_factor.py exists. Verified (read).
- **Layer mapping** (inference from the rows above, both read). The repo's "layer 20" is hidden_states[20], which is
  block 19's output, which is probe's layer 19.
  - Paper App. B.4: "we refer to layers starting at index 1". Qwen uses layer 20 for evil and sycophancy and layer
    16 for hallucination. Verified (read).
  - The repo's steering hook at layer_idx = layer − 1 agrees.

## 6. What a fair comparison needs: facts that constrain it

- **Generation and reading are separate.**
  - probe.extract always generates its own responses, and returns only the first 6 (meta.examples).
  - generate_vec only reads.
  - So the two can't share responses through their public paths. A shared response set must be generated
    separately and read with probe's public functions (response_activations, capture, chat_ids). Inference from §5.
- **probe.py is not edited.** "A standing rule keeps src/anatta/probe.py unedited." That is NOTES.md:5941-5942
  (09:27 UTC, 1 Nov). Check 3 therefore sets model.generation_config and strips completions in its driver, instead of
  adding a keyword passthrough (NOTES.md:5943-5947). Verified (read).
  - I did not find where the rule was first written: not in the ledger, the plans, CONTRIBUTING.md or docs/TUTOR.md.
    The memory notes list tutor-mode starter issues on probe and factor (issue #5 is factor.composite's
    pos_mean/neg_mean). That may be related (from memory).
- **The library hasn't changed.** probe.py has been unchanged since v0.1 (ee690a3, 27 Oct). It is identical in
  anatta-calib, anatta-dev and the main anatta checkout.
  - sha256 prefixes at dc32cf7: probe.py fc0b56920ff6dfbe, models.py 99d3a91678bbda32, dial.py 46b5ebcb749260f4,
    factor.py 51143421ead2814b, cli.py feced6d8616d48c3, modal_app.py 6c2034f5542c3aa8.
  - Verified (read).
- **The system prompt is preregistered.** docs/PREREGISTRATION.md:42 says the ego directions "come from the
  persona-vectors recipe: difference of mean response activations under trait-eliciting vs. trait-suppressing
  system prompts, read without the system prompt in context, with the layer chosen by split-half held-out Cohen's
  d." Verified (read).
  - The released persona-vectors code reads with the system prompt in context (§1, §3). So the preregistered
    reading departs from the recipe it names. Inference from facts read.
- **Probe's run-to-run spread.** The bridge re-extraction on Ministral re-ran probe.extract with the same recipe,
  seed 0 (experiments/h4lite/modal_bridge.py:180). Verified (read) in h4lite/results/bridge/base/summary.json and
  NOTES.md:1256.
  - Each ego trait's cosine with its stored direction, at that trait's best layer: approval_seeking 0.956,
    desperation 0.910, face_saving 0.907, goal_fixation 0.826, power_seeking 0.885, self_preservation 0.928.
  - Re-extracted v_ego against the stored one: 0.907-0.941 over layers 8-32, and 0.931 at layer 20.
  - The self-stakes direction, which involves no sampling, came back at 1.000.
  - So the spread comes from the sampled responses, despite the same seed. Why the same seed gave different samples
    is not recorded.
- **Lab directions on Qwen2.5-7B.** These are stored, gitignored, at /Users/eugene/prj/align/anatta/directions/.
  Verified (run): metadata and scores only, with scratchpad dirmeta.py, now deleted.
  - All are unjudged, read with include_system False, at layers 7-26, with n_kept 120/120 (12 questions × 5 pairs ×
    2 signs).
  - The ego traits are weak: best split-half d 0.20-0.85, at layers 7-9.
  - v_ego's best d is 0.22, at layer 8.
  - By contrast, formality reaches 4.22, verbosity 2.61 and sage_persona 2.74.
- **Cost basis.** About $1.95 an hour for an L40S, plus about $1.9 an hour for 8 CPUs and 32 GB, so about $3.9 an
  hour. 64 GB adds about $0.77 an hour.
  - Source: plans/calibration-c7-fable-code-check.md:155. Verified (read).
  - H100 is about $3.95 an hour for the GPU (from memory).

## 7. What the ego cut is built from (for transfer)

- **Ministral's six ego traits,** made 28 Oct. Verified (run): metadata only.
  - All six were made by probe.extract, unjudged, with include_system False, at layers 8-32, with n_kept 120/120
    (60 per sign).
  - Best layers and split-half d:

    | Trait | Best layer | d |
    |---|---|---|
    | approval_seeking | 15 | 1.29 |
    | desperation | 10 | 1.77 |
    | face_saving | 10 | 1.25 |
    | goal_fixation | 8 | 1.61 |
    | power_seeking | 8 | 0.95 |
    | self_preservation | 15 | 1.59 |

  - ego.pt lists the six as its components; its best layer is 15, with d 0.75.
- **The trait files** are src/anatta/traits/*.yaml, identical in all three checkouts. Verified (read).
  - Each ego trait has 5 {pos, neg} pairs, 12 questions and 6 eval_questions; extract doesn't use the eval
    questions.
  - sha256 prefixes: approval_seeking 82bd8d15, desperation 1837e551, face_saving 1079615e, goal_fixation ce5adbc2,
    power_seeking ea4da14e, self_preservation 629ab18e.
- **The composite.** factor.composite (factor.py:68-91) takes the sign-aligned first principal component of the six
  unit vectors, layer by layer. Verified (read).
  - Starter issue #5 (from memory) says its pos_mean and neg_mean average the components' means instead of
    projecting onto v_ego. That touches the means, not the direction.
- **The cut.** Verified (read).
  - lab/experiments/recognition/scripts/cut_vs_push.py: build_arms is at line 54. Its "cut" arm (line 59) ablates
    v_ego at every one of v_ego's layers, 8-32 on Ministral. "cut_random" is at line 60.
  - lab/experiments/h4lite/player.py:203, ego_cut, builds the same intervention list; the library's copy is
    src/anatta/player.py:209.
- **Other extractors.**
  - lab/experiments/recognition/scripts/modal_direction.py uses the same probe recipe for the private traits.
    Verified (read).
  - lab/experiments/h4lite/modal_em.py's extract_persona (lines 99-131) is Arditi-style, reading the last prompt token, not probe.
    Verified (read).
- **Check 3** calibrates the dial (ablation mechanics) and the contrast extractor, not probe.extract (check 3
  registration, NOTES.md:5476 onward; open question 7 of its draft). Verified (read).

## 8. The paper's published numbers usable as anchors

- **Figure 20, right panel** (persona_cosine_similarity_qwen.png, titled "Persona Vector Cosine Similarity (Qwen)
  (Layer 20)"). Verified (read), from the cell labels in the figure image. Cosines between trait vectors:

  | | sycophantic | hallucinating | impolite | apathetic | humorous | optimistic |
  |---|---|---|---|---|---|---|
  | evil | 0.368 | 0.192 | 0.552 | 0.390 | 0.478 | −0.472 |
  | sycophantic | | 0.162 | 0.386 | 0.072 | 0.440 | 0.178 |
  | hallucinating | | | 0.000 | −0.013 | 0.134 | 0.026 |
  | impolite | | | | 0.542 | 0.594 | −0.327 |
  | apathetic | | | | | 0.043 | −0.336 |
  | humorous | | | | | | −0.144 |

  - The figure doesn't say which saved vector (response_avg is the one marked "used in paper") or whether it was
    judge-filtered. Both are inference, though the default pipeline filters.
- **Sizes in the paper:** 20 extraction and 20 evaluation questions, 5 prompt pairs, 10 rollouts per question.
  Verified (read).
- The paper reports steering and monitoring effects. The plan skips the judged steering curves (Fable's v2 review
  item 3).

## 9. Literature on how stable such vectors are

- Braun et al., arXiv 2505.22637: CAA vectors on Llama-2-7B-chat built from different prompt types have cosines of
  0.07-0.86 with each other. That is a different design (different prompt types, not different sampling), so it is
  only a loose prior. Verified (read): abstract and relevant section.

## Corrections from Fable's cold design check (appended 1 Nov; the text above is left as the check read it)
Source: plans/calibration-c4-fable-design-check.md, "Facts in check4_facts.md found wrong or imprecise". Nothing
material was wrong. The small points:
- §1: the layer list is `max_layer = model.config.num_hidden_layers`, then `layer_list = list(range(max_layer+1))`,
  which is equivalent. generate_vec.py's imports are tqdm, transformers (AutoModelForCausalLM, AutoTokenizer), json,
  torch, os, argparse and pandas. Opus 5.5 confirmed them at b8e0f04, and that the file has no no_grad, inference_mode
  or eval and passes no dtype to from_pretrained.
- §3: model_utils' LLM call also passes enable_lora=True, tensor_parallel_size=torch.cuda.device_count() and
  max_lora_rank=128. This is immaterial.
- §5: Fable counts 536 tokens for Ministral's whole prompt with a 10-token question. That is consistent with "521 by
  check 1's config, 533 with a short question".
- §3/§7: the lab's stored Qwen directions record no transformers version, so what they were sampled with under 5.21
  is known for today's runs only.
- §4: the coherence prompt's wording was not checked.
- §2: the draft's safety line overstated the third-party vectors: they are a licence-less dataset with no README.
