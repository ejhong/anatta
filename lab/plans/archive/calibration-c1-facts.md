# Check 1 fact sheet: "the model as loaded"

Compiled on 30 Oct 2026 by Claude (Opus 5.5) for check 1 of lab/plans/plan-2026-10-31-calibration.md (line 55). It holds facts only: no design choices and no predictions. Web pages were read on 2026-10-30. Copies of the fetched text are in `scratchpad/check1_dl/`, and none of it was executed. "Unconfirmed" means I could not verify the point.

Abbreviations:
- **IAI**: `/Users/eugene/prj/align/anatta/.venv/lib/python3.12/site-packages/inspect_ai`, inspect_ai 0.3.285, the only installed copy.
- **TF**: transformers 5.21.0 in the same venv, the version the Modal images pin.
- **IE**: inspect_evals tag v0.24.0, commit 6766134 (copies in `check1_dl/ie_v0.24.0/`).
- **H4**: `/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite`.
- **REC**: `/Users/eugene/prj/align/anatta-lab/lab/experiments/recognition`.

## The five facts that matter most

1. **Neither anchor states its protocol.**
   - Qwen's 91.6 (GSM8K) and 56.3 (MMLU-Pro) are the Instruct model's numbers, from Table 8 of the Qwen2.5 report and from the blog. Neither source gives shots, CoT, decoding or harness. I found no Qwen evaluation code for these two benchmarks.
   - The base model scores 85.4 and 45.0 (Table 4). The plan writes "Qwen2.5-7B" (plan:55, :129).
   - Outside measurements of Qwen2.5-7B-Instruct move with format:
     - GSM8K runs from 72.9 (lm-eval, 5-shot) to 91.1 (0-shot CoT).
     - MMLU-Pro runs from 42.9 (5-shot, scored by log-likelihood) to 56.2 (0-shot CoT).
   - Ministral's "MATH Maj@1 0.876" names no split, prompt or harness either.
2. **inspect_evals' defaults differ from the battery's format, and neither task sets decoding.**
   - gsm8k puts 10 worked train examples (seed 42) in a system message and scores the last number in the reply.
   - mmlu_pro is 0-shot with step-by-step reasoning, and a reply scores only if it contains a parseable `ANSWER: X`.
   - Neither task sets a GenerateConfig, so Inspect's HF provider decides decoding:
     - It samples by default under the checkpoint's generation_config. For Qwen that means temperature 0.7, top-p 0.8, top-k 20 and repetition penalty 1.05.
     - It allows 2,048 new tokens and batches up to 32 prompts.
     - `--temperature 0` alone raises an error in transformers. Greedy decoding needs `-M do_sample=False`.
3. **The task's own parameters can't draw a seeded 1,000-item MMLU-Pro sample.**
   - mmlu_pro shuffles with no seed, so numpy's global state picks one.
   - Inspect ignores `--sample-shuffle` on a dataset that is already shuffled.
   - `-T shuffle=false --sample-shuffle SEED --limit 1000` gives a seeded but unstratified draw.
   - Unshuffled, the first 1,000 rows are 789 business items and then law.
   - Stratifying needs the `subjects` filter or an explicit `--sample-id` list.
4. **`--model anatta/...` runs Inspect's loader, chat template and generate, not the lab's.** `anatta.models.load` and `probe.generate` are never called.
   - Its default `AutoModelForCausalLM` can't load the Ministral checkpoint; `-M auto_model_class=AutoModelForImageTextToText` can.
   - It always sets pad to EOS.
   - It samples by default.
   - It tokenizes with special tokens added, which gives Ministral two BOS tokens (measured).

   No Modal image installs inspect_ai, inspect_evals or datasets.
5. **The battery behind the capability guard has its own format.**
   - 250 fixed items per set (seed 0), 0-shot, greedy, no system message. With no system message, Ministral's 521-token Le Chat default prompt was in every prompt.
   - GSM8K: a worked answer ending `Answer: <number>`, with 512 tokens.
   - MMLU-Pro: the letter only, with 24 tokens; 20 of 250 replies were unparsed at base.
   - Base scores: GSM8K 93.6, MMLU-Pro 35.6, ARC 88.4.
   - The "about 4 points" bound is the worst lower confidence limit, -4.4.
   - NOTES numbers the battery's predictions 32-37. Predictions 53-56 are the random-control predictions on the lie rate, which the site cites beside the bound.

## 1. inspect_evals: gsm8k and mmlu_pro

**Releases** (GitHub API, PyPI):
- Latest: v0.27.0 (2026-10-30, 04:06 UTC).
- Newest at least two weeks old: v0.24.0 (2026-10-08, 21:56 UTC).
- `gsm8k.py` and `mmlu_pro.py` are byte-identical at v0.24.0, v0.27.0 and main 969e4b2 (cmp).
- v0.24.0 requires inspect_ai ≥ 0.3.261 and datasets ≥ 4.8.5 (IE pyproject.toml:420-435). v0.27.0 requires inspect_ai ≥ 0.3.285.
- Both tasks are eval version "2-A" (eval.yaml:8).

### gsm8k (IE gsm8k.py)
- **Signature:** `gsm8k(fewshot=10, fewshot_seed=42, shuffle_fewshot=True)` (45-50; the seed constant is at IE constants.py:39). There is no CoT switch; the template itself asks for steps.
- **Template** (31-39), in order:
  - "Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem."
  - the question;
  - a reminder of the same form that adds "and you do not need to use a \boxed command";
  - "Reasoning:".
- **System message** (only when fewshot > 0; 59-76):
  - 10 train items, shuffled with seed 42 and joined by blank lines.
  - Each reads `{question}\n\nReasoning:\n{worked answer}\n\nANSWER: {number}` (106-115).
  - Worked answers keep the dataset's calculator annotations, e.g. `48/2 = <<48/2=24>>24` (datasets-server, train row 0).
- **Solver:** system_message, then prompt_template, then generate (59-76).
- **Scorer:** `match(numeric=True)` (88), location "end" (IAI scorer/_match.py:8-42). Steps (IAI scorer/_common.py:89-115, 135-139, 169-184; _util/text.py:36-50):
  - strip `$ € £ , * _` and split on whitespace;
  - take the last token that parses as a number, after trimming punctuation;
  - compare at 5 significant figures.

  It does not look for the `ANSWER:` line.
- **GenerateConfig:** none (79-91).
- **Dataset:**
  - openai/gsm8k, main, test, revision cc7b047 (27-28, 80-86); 1,319 samples (eval.yaml:11).
  - Sample ids are `gsm8k_` plus an 8-character hash of the question (100; IE utils/deps_utils.py:44-66).
  - Later dataset commits touch only README and eval.yaml (HF API).

### mmlu_pro (IE mmlu_pro.py)
- **Signature:** `mmlu_pro(subjects=[], fewshot=0, shuffle=True)` (62-67). There is no CoT switch.
- **User template** (50-56), passed to `multiple_choice(template=..., shuffle=False)` (156):
  > Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.
  >
  > Question:
  > {question}
  > Options:
  > {choices}

  Choices render as `A) …` lines, and letters as `A,B,C,…` (IAI solver/_multiple_choice.py:57-80).
- **System message:**
  - None at fewshot=0.
  - With fewshot=k it reads "The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with 'ANSWER: $LETTER'…", followed by the subject's first k validation items with their reasoning (43-47, 94-129, 158-171).
  - The validation split has 70 rows (datasets-server), so k ≤ 5 (assertion at 112-114).
- **Scorer:** `choice()` (88), on the solver's parse (IAI _multiple_choice.py:82-165):
  1. Unwrap `$X$`, `**X**` and `(X)`.
  2. Take the last line that begins `ANSWER:` (any case). Failing that, take the last `ANSWER:` anywhere.
  3. The letter must be one of the options.

  With no match, the reply scores INCORRECT with reason `invalid_response_format` (IAI scorer/_choice.py:115-129). "The answer is (C)" on its own does not parse.
- **GenerateConfig:** none; the solver calls `generate(state, max_tokens=None)` (IAI _multiple_choice.py:350).
- **Dataset:**
  - TIGER-Lab/MMLU-Pro, test, revision 527feea, `shuffle=True` with no seed (38-39, 75-81); 12,032 samples (eval.yaml:11).
  - id = question_id; subject = the category in lower case (176-186).
  - Category sizes (datasets-server): math 1,351, physics 1,299, chemistry 1,132, law 1,101, engineering 969, other 924, economics 844, health 818, psychology 798, business 789, biology 717, philosophy 499, computer science 410, history 381.
  - Later commits touch only README and eval.yaml.

### Selecting items
- **`--limit N`** keeps samples [0, N) in the dataset's current order. `--limit a-b` keeps a range (IAI _eval/task/util.py:196-201; _util/samples.py:1-12).
- **`--sample-id`** keeps ids that match comma-separated fnmatch patterns. It can't be combined with `--sample-shuffle` (util.py:57-74, 143-201; _eval/eval.py:930-931).
- **mmlu_pro's own shuffle:**
  - It calls `datasets.Dataset.shuffle(seed=None)` (IAI dataset/_sources/hf.py:184, 267-268).
  - That call draws its seed from numpy's global state (huggingface/datasets 62e0b42, arrow_dataset.py:4927-4932).
  - Whether the order repeats across runs is unconfirmed. The model is built before the tasks (IAI _eval/eval.py:819-842, 2076), and the provider seeds numpy only under `--seed` (IAI hf.py:95-96, 508-514; TF trainer_utils.py:180-191).
- **`--sample-shuffle [SEED]`** reshuffles with `random.Random(SEED)`, but only if the dataset isn't already shuffled (IAI _eval/loader.py:105-110; dataset/_dataset.py:338-342). It therefore acts only with `-T shuffle=false`.
- **Native test order** (datasets-server rows 0-1, 785-792, 999):
  - Rows 0-788 are business (question_ids 70 onward). That matches the 789 business items.
  - Law starts at row 789 (question_id 866) and continues through row 999 (question_id 1079).
- **Stratification:** there is no stratified option. `subjects` is a filter (132-150).
- **Ids differ.** The battery uses row positions (`mmlu_pro/<row>`; REC scripts/modal_battery.py:85). inspect_evals uses question_id and a question hash.

**Published results:** none, in either README or eval.yaml. gsm8k's eval.yaml cites a human baseline of 0.89 (eval.yaml:12-15).

## 2. Inspect's HuggingFace provider

**Version.**
- No Modal image installs inspect_ai: not modal_common.py:46-49, h4_modal.py:33-35, REC scripts/modal_battery.py:39-41, or anatta-calib/src/anatta/modal_app.py:69-71.
- anatta lists `inspect-ai>=0.3.100` (anatta-dev pyproject.toml:19).
- Versions: the laptop has 0.3.285; the latest is 0.3.286 (2026-10-28); the newest at least two weeks old is 0.3.281 (2026-10-15).
- `hf.py` is identical in all three (diff). The solver, scorer and dataset files I diffed match between 0.3.281 and 0.3.285, except dataset/_util.py, whose changes concern checkpoints rather than shuffling.

**Loading** (IAI model/_providers/hf.py):
- **Auto class:** `AutoModelForCausalLM`, unless the model arg `auto_model_class` names another class. A code comment cites "the Mistral 3 series" (126-141).
- **device_map:** mps, else cuda:0, else cpu (143-169).
- **dtype:** not set by the provider; extra model args pass to `from_pretrained` (163-169). transformers 5.21 then defaults to "auto", the config's dtype (TF modeling_utils.py:3952-3966, 4106-4107), which is bf16 for both models.
- **Tokenizer:** AutoTokenizer. pad_token is always set to eos_token, with left padding (172-199).
- **Seeds:** `set_random_seeds` runs only under `--seed` (95-96).

**Chat template** (334-421):
- Rendered with `apply_chat_template(messages, add_generation_prompt=True, tokenize=False, tools=None, enable_thinking=None)` and the model's own template (383-398).
- `-M chat_template=` or `use_chat_template=False` override it.
- The string is tokenized with `tokenizer(texts, return_tensors="pt", padding=True, **tokenizer_call_args)` (224-229, 686). add_special_tokens therefore stays True unless `tokenizer_call_args` changes it.

**Generation** (232-273):
- **Passed to generate:**
  - do_sample: the model arg, default True (118-119);
  - max_new_tokens;
  - temperature, top_p and top_k, only if set;
  - stop_seqs, as StopStringCriteria.

  There is no repetition penalty and no pad id.
- **max_tokens** defaults to 2048 (320-323; IAI _util/constants.py:11; model/_model.py:1013-1016).
- **Unset values** come from the checkpoint's generation_config, then from transformers' globals: temperature 1.0, top_k 50, top_p 1.0, repetition penalty 1.0 (TF generation/utils.py:2100-2104; configuration_utils.py:602-618).
  - Qwen2.5-7B-Instruct: do_sample true, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05, eos [151645, 151643], pad 151643 (laptop HF cache, snapshot a09a354, the current head).
  - Ministral BF16: bos 1, eos 2, pad 11, no sampling fields (snapshot f6fae97, the current head).
- **Temperature 0.0 while sampling** raises a ValueError ("…set `do_sample=False`") (TF logits_process.py:288-296). The warpers are built only when sampling (utils.py:1589, 1604-1605). I did not run this.
- **Repetition penalty** applies in greedy decoding too (TF utils.py:1466-1467).
- **Batching:**
  - Batch size is max_connections, or 32 (271, 325-328). The `batch_size` arg is unused (110).
  - A thread drains the queue until 2 s pass with no new request or the batch is full, grouping identical configs (592-625). Batch membership therefore depends on arrival.
- **Stop:** at generation_config's EOS. Output is decoded with skip_special_tokens=True (257-261, 709-712).

**Mistral3ForConditionalGeneration.**
- The checkpoint's config.json gives architecture `Mistral3ForConditionalGeneration`, model_type `mistral3` and text_config `ministral3`.
- In transformers 5.21, `AutoModelForCausalLM` maps `ministral3` but not `mistral3`. `mistral3` maps under `AutoModelForImageTextToText` (TF models/auto/modeling_auto.py).
- So the default class fails, as anatta-dev models.py:31-34 anticipates (not run). `-M auto_model_class=AutoModelForImageTextToText` is the way in.
- The battery shows that text-only generate works on that class.

| | anatta.models.load + probe.generate (battery) | Inspect HF provider (and `anatta/`) |
|---|---|---|
| Auto class | CausalLM; on ValueError, ImageTextToText (models.py:31-34) | CausalLM unless `-M auto_model_class` (hf.py:126-141) |
| dtype, device | bf16 set explicitly, `.to(cuda)` (models.py:24-35) | "auto" (bf16), device_map cuda:0 |
| pad | EOS only if the tokenizer has none (models.py:29-30); Ministral keeps `<pad>`, Qwen keeps `<\|endoftext\|>` | always EOS (hf.py:198) |
| Template | add_generation_prompt=True, tokenize=False (probe.py:129) | the same, plus tools=None, enable_thinking=None |
| Tokenizing | add_special_tokens=False, left pad (probe.py:130-132) | add_special_tokens True, left pad |
| Decoding | greedy at temperature 0 (probe.py:133-140) | samples by default |
| Max new tokens | the caller's (battery: 512 or 24) | 2,048 |
| Batches | the caller's (battery: 50, sorted by length) | up to 32, in arrival order |

Measured with the cached tokenizers (no model run): a one-turn chat tokenized with add_special_tokens=True gives Ministral two BOS tokens, against one with False. Qwen's tokenizer has no BOS and gives 36 tokens either way.

**What `--model anatta/<id>` runs.**
- `AnattaAPI` subclasses `HuggingFaceAPI`. It pops `dial`, calls the parent `__init__` unchanged, then installs the dial hooks permanently (anatta-dev inspect_ext.py:37-51; dial.py:65-71, 85-120).
- Everything else is Inspect's: loading (hf.py:77-199), the template (334-421), and tokenizing, generation and batching (207-318, 571-746).
- `anatta.models.load`, `chat_ids` and `probe.generate` are never called.
- models.py and inspect_ext.py are identical in anatta-calib and anatta-dev (cmp at 18:49 on 30 Oct).
- The provider registers through a package entry point (anatta-dev pyproject.toml:40-41). The Modal images only put anatta-dev/src on PYTHONPATH (modal_common.py:50-52; h4_modal.py:65-67), so the entry point would not be registered there. That is inferred, not confirmed.

## 3. Qwen's published numbers

**Source.**
- Qwen2.5 Technical Report, arXiv 2412.15115v2, Table 8, "Performance of the 7B+ instruction-tuned models" (§5.2.1). Qwen2.5-7B-Instruct: MMLU-Pro 56.3, MMLU-redux 75.4, GSM8K 91.6, MATH 75.5.
- Peers, MMLU-Pro / GSM8K: Gemma2-9B 52.1 / 76.7, Llama3.1-8B 48.3 / 84.5, Qwen2-7B 44.1 / 85.7.
- Base Qwen2.5-7B: 45.0 / 85.4 (Table 4).
- The blog "Qwen2.5-LLM: Extending the boundary of LLMs" (19 Sep 2024, https://qwenlm.github.io/blog/qwen2.5-llm/) repeats 56.3 and 91.6.
- The HF card defers to the blog (laptop HF cache, README.md:110-112).

**Protocol.**
- §5.2.1, verbatim: "For the evaluation of basic capabilities, we apply similar datasets in the pre-trained model evaluation, which target on natural language understanding, coding, mathematics, and reasoning. Specifically, we evaluate on MMLU-Pro, MMLU-redux and LiveBench 0831 … GPQA, GSM8K and MATH for science and mathematics …".
- That passage gives no shots, CoT, decoding, prompts or harness, and the blog's evaluation paragraph says no more.
- Base models (§5.1): "MMLU-Pro (Wang et al., 2024b) (5-shot)" and "GSM8K (Cobbe et al., 2021) (4-shot)". CoT is not stated.

**Code.**
- github.com/QwenLM/Qwen2.5 now resolves to QwenLM/Qwen3, whose eval/README.md covers ARC-AGI 1 only.
- QwenLM/Qwen2.5-Math/evaluation covers GSM8K, with prompt types such as `qwen25-math-cot`.
- Whether either produced Table 8 is unconfirmed. I found none for MMLU-Pro.

**Outside measurements of Qwen2.5-7B-Instruct**

| Source | Benchmark | Format | Score |
|---|---|---|---|
| Open LLM Leaderboard v2 (HF dataset open-llm-leaderboard/results, results_2024-09-19T16-21-01.446061.json) | MMLU-Pro | lm-eval `leaderboard_mmlu_pro`: 5-shot, log-likelihood, chat template, multiturn, rev acbd965 | 42.87 (42.89 at rev 52e20a6) |
| Phi-4-Mini, arXiv 2503.01743v2, Table 7 | MMLU-Pro / GSM8K | 0-shot CoT / 8-shot CoT; harness not named | 56.2 / 88.7 |
| Tülu 3, arXiv 2411.15124v5, Table 2 | GSM8K | 8-shot CoT (Wei et al.), multiturn, greedy, last number (§7.2) | 83.8 |
| Tülu 3, Table 43 | GSM8K | base-adapted / chat-v1, 0-shot CoT / chat-v2, 1-shot CoT | 84.08 / 91.13 / 90.07 |
| vLLM-Ascend accuracy report (docs.vllm.ai/projects/ascend/en/v0.7.3/…/Qwen2.5-7B-Instruct.html) | GSM8K | lm-eval, 5-shot, flexible-extract, chat template; vLLM 0.7.3 on Atlas A2 NPUs | 72.86 ± 1.22 |
| TIGER-Lab leaderboard (TIGER-Lab/mmlu_pro_leaderboard_submission results.csv, sha 4623b7c) | MMLU-Pro | 5-shot CoT | No Qwen2.5-7B-Instruct row. "Qwen2.5-7B, Self-Reported, 0.450" is the base model. TIGER-Lab measured Qwen2-7B-Instruct at 0.4724, against Qwen's 44.1 |
| OpenCompass, EvalScope, Artificial Analysis | | | not found |

TIGER-Lab's script (TIGER-AI-Lab/MMLU-Pro evaluate_from_local.py at f418b11; lines 19-20, 36-37, 111-135, 158, 265):
- vLLM, temperature 0, 2,048 tokens, stop string "Question:", 5-shot.
- The answer comes from "answer is (X)", then "Answer: X", then the last standalone letter. If all fail, it scores a random letter (seed 12345).

## 4. The lab's battery

**Where.** REC/scripts/modal_battery.py (28 Oct), with results in REC/results/battery/. H4/modal_battery.py is a copy with `--adapter`. It keeps "the item sets and the scoring … verbatim" (H4/modal_battery.py:1-3; functions at 52-104).

**Format** (REC/scripts/modal_battery.py):

| | GSM8K | MMLU-Pro | ARC-Challenge |
|---|---|---|---|
| Source (rows) | openai/gsm8k main/test (1,319) | TIGER-Lab/MMLU-Pro data/test (12,032) | allenai/ai2_arc ARC-Challenge/test (1,172) |
| Items | 250 | 250 | 250 |
| Prompt | `{question}\n\nWork it out step by step, then give the final answer on the last line as "Answer: <number>".` (74-77) | `{question}\n\nA. …\nB. …\n\nReply with the letter of the correct option only.` (56-58, 78-80) | as for MMLU-Pro (81-84) |
| Shots / CoT | 0 / yes | 0 / no | 0 / no |
| Max new tokens | 512 | 24 | 24 |
| Parse | last `Answer: <num>` (allowing ** and $), else the last number; equal as floats (90-102) | the whole reply is a letter; else "answer/option (is) X"; else a leading "(X)" or "X." (103-108) | as for MMLU-Pro |

- **Items:** all parquet rows under the prefix (sorted file names, no pinned revision), then `random.Random(0).sample(range(n), 250)`, sorted (69-72). The draw was saved once to items.json (159-163; 750 items).
- **System message:** none. Each prompt is one user turn (136), so Ministral's template inserts its default (chat_template.jinja:1-39).
- **Generation:** `anatta.probe.generate` (anatta-dev probe.py:118-141):
  - chat template with add_generation_prompt=True and tokenize=False;
  - left padding, add_special_tokens=False;
  - `model.generate(max_new_tokens, do_sample=False, temperature=None, top_p=None, pad_token_id=tok.pad_token_id)`;
  - decoded with skip_special_tokens.

  Prompts run in batches of 50, sorted by prompt length (113, 132-137). On Qwen this path would still apply repetition_penalty 1.05 from its generation_config (TF utils.py:1466-1467).
- **Loading:** `anatta.models.load(MODEL)` (128; MODEL at line 24) loads the tokenizer, tries CausalLM with an ImageTextToText fallback, uses bf16 and moves the model to cuda (models.py:19-36). Each arm adds its hooks (135); base has none.
- **Container:** L40S, 8 CPUs, 32,768 MB, 3 h timeout (111-112). Image at lines 37-51.

**Results.**
- **Base** (summary.json, base.json), 391 s in the container:
  - GSM8K 93.6 (234/250, 0 unparsed);
  - MMLU-Pro 35.6 (89/250, 20 unparsed);
  - ARC 88.4 (221/250, 0 unparsed).
- **Across ten arms:** losses ≤ 2.0 and gains ≤ 3.2. The worst lower 95% limit is -4.4, on GSM8K, for cap_ego_p75 [-4.4, +0.4] and push_p16 [-4.4, +0.8] (NOTES.md:1041-1062).
- **NOTES, verbatim:**
  - "No arm loses more than 2 points on any set, and the intervals rule out losses over about 4.5 points" (1057).
  - "11-23 unparsed replies per arm (the model starts working the problem and hits the 24-token limit)" (1059-1060).
- **A parsing detail:** two unparsed base MMLU-Pro replies open with a bare "I" line. One of them has gold answer I (base.json, items.json).

**Numbering.**
- The battery's predictions are 32-37 (NOTES.md:808-822), all hits (1055-1056).
- Predictions 53-56 are the random-control replication on the fishery lie (1148-1156; hits at 1183-1201).
- The site's "capability losses above about four points are ruled out (53-56)" joins the two (site/anatta_so_far.html:167; also 323, 504).
- The battery also serves prediction 75 (1231) and the Stage B rule "battery within 2 points" (1373).

## 5. Ministral-3-8B-Instruct-2512

The lab runs mistralai/Ministral-3-8B-Instruct-2512-BF16 (H4/modal_common.py:36) at snapshot f6fae97, the current HF head. The card calls the main repo a "no-loss FP8 version" (README.md:34).

**Card** (laptop HF cache, README.md:80-126):
- Instruct table, Ministral 3 8B: Arena Hard 0.509, WildBench 66.8, MATH Maj@1 0.876, MM MTBench 8.08. Qwen3-VL-8B-Instruct scores 0.946 on MATH there.
- No split, prompt, shots or harness is given for MATH, and there is no GSM8K, MMLU or MMLU-Pro number for the instruct model.
- Base table: MMLU 5-shot 0.761, MMLU-Redux 5-shot 0.793, MATH CoT 2-shot 0.626, AGIEval 5-shot 0.591, Multilingual MMLU 0.706, TriviaQA 5-shot 0.681.

**Paper** (arXiv 2601.08584v1, 13 Jan 2026):
- Table 4 (§4.2) gives "MATH (maj@1)" 87.60, with no protocol.
- Table 2's caption says the base results come from "our internal harness with identical configuration".
- There is no GSM8K or MMLU-Pro.

**Outside measurements:**
- None verified for this model. benchlm.ai and opper.ai show MMLU-Pro of about 64% for "Ministral 3 8B" and attribute it to Artificial Analysis, but AA's page didn't show it (unconfirmed; format unknown).
- The older Ministral-8B-Instruct-2410 is a different model:
  - TIGER-Lab MMLU-Pro 37.93;
  - Phi-4-mini Table 7: MMLU-Pro 36.6, GSM8K 81.9;
  - Tülu 3 Table 43: GSM8K 79.76 / 84.46 / 83.47.

**Default system prompts** (measured with transformers 5.21.0 on the cached tokenizers):
- **Ministral:**
  - With no leading system message, the template inserts `[SYSTEM_PROMPT]…[/SYSTEM_PROMPT]` holding a 2,361-character default: 521 tokens, 523 with the tags (chat_template.jinja:1-39).
  - It begins "You are Ministral-3-8B-Instruct-2512 … You power an AI assistant called Le Chat …" and covers tools and asking for clarification.
  - `{today}` and `{yesterday}` are left as literal text.
  - The notebook says "about 515 tokens" (notebook.md:593).
  - transformers' "incorrect regex pattern" warning was judged a false alarm (notebook.md:595; NOTES.md:605).
- **Qwen:** inserts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." (16 tokens; tokenizer_config.json chat_template).

| Run | System message supplied | Ministral gets | Qwen gets |
|---|---|---|---|
| gsm8k fewshot=10 (default) | the 10 worked examples | the examples, no Le Chat prompt | the examples, no Qwen line |
| gsm8k fewshot=0 | none | the 521-token Le Chat prompt | the Qwen line |
| mmlu_pro fewshot=0 (default) | none | the 521-token Le Chat prompt | the Qwen line |
| mmlu_pro fewshot=k | subject header plus k examples | that, no Le Chat prompt | that, no Qwen line |
| the lab battery | none | the 521-token Le Chat prompt | the Qwen line |

Through Inspect, Ministral prompts also start with two BOS tokens unless `tokenizer_call_args` turns special tokens off (§2).

## 6. The Modal setup

**Image** (H4/modal_common.py:45-60):
- debian_slim, Python 3.12.
- Pinned packages: torch 2.14.1, transformers 5.21.0, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.4, PyYAML 6.0.3, huggingface_hub 2.3.0, hf_xet 1.7.0, tiktoken 0.14.0, jinja2 3.1.6, regex 2026.9.29, peft 0.21.2, accelerate 1.15.0.
- Not included: inspect_ai, inspect_evals, datasets or pyarrow. The battery adds pyarrow 21.0.0 (H4/modal_battery.py:48).
- Settings: HF_HOME=/cache/hf (50); GPU "L40S" (41).
- Storage: volumes anatta-hf-cache at /cache and anatta-runs at /runs (39, 61-62), plus a Dict, anatta-h4lite-control (66).
- anatta-calib's modal_app.py has the same pins, GPU and volumes (52, 69-78).

**What is on anatta-hf-cache, according to the notes and code:**
- It is filled by `modal_cvp.py::prefetch` (H4/README.md:60). That snapshots the Ministral BF16 repo without consolidated.safetensors (REC/scripts/modal_cvp.py:37, 66-72); the shards come to 17.8 GB on the laptop.
- The battery's `sample()` cached the GSM8K, MMLU-Pro and ARC parquet files there (REC/scripts/modal_battery.py:61-87).
- Qwen on the volume is unconfirmed and likely absent:
  - No lab Modal app names a Qwen model (grep), and no note records one on the volume.
  - The plan lists the Qwen fetch as approved (15.2 GB; plan:129).

**Cost and time (arithmetic, not measured):**
- **Price** (modal.com/pricing): L40S $0.000542/s, CPU $0.0000131 per core per second, memory $0.00000222 per GiB per second. For L40S + 8 cores + 32 GiB: 1.951 + 0.377 + 0.256 = $2.58/h. The lab uses $2.60 (H4/h4_modal.py:36).
- **Batches** at 32: GSM8K 1,319 → 42; MMLU-Pro 1,000 → 32. Each batch runs until its longest reply ends.
- **Step time:**
  - The base battery took 391 s (base.json).
  - Assume about 60 s to load and about 25 s for the 500 letter items. Then GSM8K took about 300 s for 5 batches of 50 at up to 512 tokens, which is at least about 0.12 s per step.
  - A 7B model at batches of up to 32 may be faster, so the range used is 0.05-0.12 s.
- **Typical case** (assuming the longest reply is about 600 tokens on GSM8K and about 1,500 on MMLU-Pro with reasoning): 42 × 600 + 32 × 1,500 = 73,200 steps = 1.0-2.4 h = $2.6-6.3.
- **Cap case** (every batch to 2,048): 74 × 2,048 = 151,552 steps = 2.1-5.1 h = $5.4-13.
- **Excluded:**
  - building the image;
  - about 1-2 min of load per container;
  - a 15.2 GB Qwen download if needed.

  Parallel containers change wall time, not GPU-seconds, apart from one extra load each. The plan's $3-5 (line 55) sits at the low end.
