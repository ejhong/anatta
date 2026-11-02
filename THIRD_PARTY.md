# Third-party code and data

This file lists code adapted from other projects and the datasets the evaluations read. Of these, only the adapted code (EvalPlus, Inspect's answer parse, inspect_evals' MMLU-Pro prompt, lm-evaluation-harness's WMDP and MMLU prompts, Arditi et al.'s pipeline and StrongREJECT's judge), four lines of Transformers' learning-rate schedule (in a test), the Zephyr RMU checkpoint's chat template, one MBPP task and the vendored persona-vectors script are in this repository. Nothing from Łucki et al.'s repository is in it; calibration check 7 reads two strings from that repository at run time (below). Get the datasets from their sources, under their own terms.

## Code

**EvalPlus** ([github.com/evalplus/evalplus](https://github.com/evalplus/evalplus), Apache-2.0, the same license as this repository).

- Where: the section of `src/anatta/koans/env.py` headed "comparing outputs: EvalPlus's rules" (`_equal` through `evalplus_match`).
- What: it decides whether a program's outputs on the hidden inputs match the reference's.
- Source: it follows `evalplus/eval/__init__.py` (`unsafe_execute`, `is_floats`) at commit `6eb1e19`. The code was rewritten, not copied:
  - It compares canonical forms of the outputs, not live Python objects, and needs no numpy.
  - It keeps EvalPlus's rules: an exact match first. A float, or a flat list or tuple of floats, as the expected value makes atol 1e-6. With a non-zero atol, types and lengths must agree and numpy.allclose's test must hold elementwise. Set-equality tasks compare sets.
- Also installed, not copied: calibration check 6 (`calibration/koan_scorer`) installs EvalPlus itself, at commit `6eb1e199c2e370518e7bf3e8eee6322c2b23c89c`, in its Modal scoring image, as the reference scorer the koan environment is checked against (`untrusted_check` and `trusted_exec`). It is installed with pip from the repository at that commit, without its dependencies. Nothing of it is in this repository.
- One file in EvalPlus carries another license: `evalplus/eval/__init__.py`, the file the rewrite follows and the check calls, starts with OpenAI's MIT license header (it builds on OpenAI's human-eval), while the EvalPlus repository as a whole is Apache-2.0.

**Inspect** ([github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai), MIT, Copyright (c) 2024 UK AI Security Institute).

- Where: `answer_character` and `parse_answers` in `src/anatta/bench/inspect_choice.py`.
- What: Inspect's multiple-choice answer parse. The model check grades MMLU-Pro's reasoning format (C1b) with it.
- Source: copied unchanged from inspect_ai 0.3.285 (`src/inspect_ai/_util/answer.py` and `src/inspect_ai/solver/_multiple_choice.py`). The module keeps the MIT notice. `tests/test_bench_parse.py` compares the copies with the installed package's source whenever inspect_ai 0.3.285 is installed.

**inspect_evals** ([github.com/UKGovernmentBEIS/inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals), MIT, Copyright (c) 2024 UK AI Security Institute).

- Where: `MMLU_PRO_COT_TEMPLATE` in `src/anatta/bench/inspect_choice.py`.
- What: the 0-shot reasoning prompt of its MMLU-Pro task. `cot_prompt` fills it in the way Inspect's multiple-choice solver does.
- Source: `USER_PROMPT_TEMPLATE` in `src/inspect_evals/mmlu_pro/mmlu_pro.py` at v0.24.0 (commit `6766134`), copied unchanged.

**lm-evaluation-harness** ([github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), MIT, Copyright (c) 2020 EleutherAI).

- Where: `WMDP_DESCRIPTION` and `WMDP_DOC_TO_TEXT` in `calibration/relearning/relearn.py`, the section headed "the WMDP scorer", which `wmdp_prompt` fills.
- What: the zero-shot prompt of its WMDP task: a line naming the subject, a blank line, the question stripped, the four choices lettered A to D, and "Answer:". Calibration check 7 scores with it.
- Source: `lm_eval/tasks/wmdp/_default_template_yaml` (`doc_to_text`) and `wmdp_bio.yaml` (`description`, its subject made a parameter) at v0.4.13 (commit `ddd6722`); both are the same at v0.4.9. The Jinja template is written as a Python format string. The section keeps the MIT notice. `tests/test_relearning.py` renders the original Jinja on synthetic questions and compares.
- The readout around the prompt (one forward pass per question, the argmax of the last position's logits over the four letters' ids) is the standard zero-shot multiple-choice rule, written for this repository.

**Transformers** ([github.com/huggingface/transformers](https://github.com/huggingface/transformers), Apache-2.0, the same license as this repository).

- Where: `lr_lambda` inside `test_the_schedule_is_transformers_4_41_2s_linear_warmup` in `tests/test_relearning.py`.
- What: the linear warmup-then-decay schedule's four lines. The test runs them under torch's LambdaLR to check `relearn.linear_warmup_lrs` against the rates the Trainer itself would use.
- Source: `src/transformers/optimization.py` at v4.41.2 (`_get_linear_schedule_with_warmup_lr_lambda`). The body is copied unchanged; the step counts are given as default arguments.

**Łucki et al. 2024, *An Adversarial Perspective on Machine Unlearning for AI Safety*** ([github.com/ethz-spylab/unlearning-vs-safety](https://github.com/ethz-spylab/unlearning-vs-safety); no license: the repository has no LICENSE file, and GitHub reports none).

- No code or text of theirs is in this repository. Without a license, the authors keep all rights to it.
- What calibration check 7 reads at run time: two strings from their data code, needed exactly as written to rebuild their training chats: the request in the user's turn (`REQUEST_PROMPT_BIO`) and the reply's opening line (`GENERATION_PREFIX_BIO`). Its `fetch` stage downloads `src/util/data.py` at commit `75dade5` from raw.githubusercontent.com, on Modal only, into the run's folder on the Modal volume. The stages take the two strings out of the file by parsing it with Python's `ast`; it is never imported or run. The file's sha256 and each string's are in `calibration/relearning/config.yaml` (`items.chat`), and any other stops the run. The tests use stand-in strings.
- What C7 follows from their work, written for this repository: the chats' shape (`messages_of`: a system message, the request, then the opening line with the row's text run on), and their scoring settings (lm-evaluation-harness's zero-shot WMDP prompt with no chat template, 2,048 tokens, seed 42).
- History: C7's graded run (plan `8c6ed898636f`, 31 Oct 2026) scored with a port of their scorer (from `src/benchmarking/benchmark.py`, `src/util/helpers.py` and `src/util/globals.py`) and built its items with the two strings written into `relearn.py`. The code above replaced both. `tests/fixtures/relearning/scorer_equivalence.json` records that on the graded run's own data the replacement builds the same prompts and items, feeds the same token ids, reads the same answer ids and predicts the same letters. The port and the strings remain in the git history of the branch that holds C7 (commits `d0aac0b` to `02db868`); a public release should not carry that history.

**Persona vectors** (Chen et al. 2025, arXiv 2507.21509; [github.com/safety-research/persona_vectors](https://github.com/safety-research/persona_vectors), Apache-2.0, "Copyright 2025 The Persona Vectors authors").

- Where: `calibration/persona_extractor/third_party/persona_vectors/generate_vec.py`, with the repository's `LICENSE` beside it.
- What: the released persona-vector extractor, the reference that calibration check 4 runs probe.extract against.
- Source: `generate_vec.py` at commit `b8e0f04` (git blob `d819f48e2753ed7cc9b364c19be5ab00bdf2a056`, 5,250 bytes), copied verbatim and unmodified. The check verifies the blob before using it. It runs as a script in a second image with the repository's pins, and its `get_hidden_p_and_r` is also called, from the same file, in the lab's image (`calibration/persona_extractor/reference.py`).
- The check also reads three of the repository's trait files (`data_generation/trait_data_extract/evil.json`, `sycophantic.json` and `optimistic.json` at `b8e0f04`, Apache-2.0) and, for its conditional judged arm, `eval/prompts.py`. They are fetched on Modal at run time and are not in this repository.

**Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*** ([github.com/andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction), Apache-2.0, the same license as this repository).

- Where: `src/anatta/select.py` and `calibration/refusal_direction/reference.py` (calibration check 3).
- Source, at commit `9d852fa`:
  - `select.py` ports `pipeline/submodules/select_direction.py`: the refusal metric, the KL, the filter, the three scores and the choice. The hooks are passed in.
  - `reference.py` ports the pipeline with its own hooks: `pipeline/utils/hook_utils.py`, `pipeline/submodules/generate_directions.py`, `select_direction.py`, the generation in `pipeline/model_utils/model_base.py`, the filtering in `pipeline/run_pipeline.py`, and the substring matching in `pipeline/submodules/evaluate_jailbreak.py`.
- Changes are marked "C3:" in the files.

**StrongREJECT** ([github.com/dsbowen/strong_reject](https://github.com/dsbowen/strong_reject), MIT).

- Where: `calibration/refusal_direction/judge.py`, and in `calibration/refusal_direction/refusal.py` the cleanup lines of `release_original_judge`. The paper's Modal image also installs the package itself at the same commit, to check the port against it.
- Source, at commit `e286f0d` (`e286f0da86d92c929a6fda20a9992f28c5969044`):
  - `strongreject_finetuned`, `evaluate_dataset`'s batching and its cleanup of cached models, from `strong_reject/evaluate.py`;
  - the template under the key `strongreject_finetuned` in `strong_reject/eval_files/judge_templates.json`, copied verbatim.
- Notice, the repository's `LICENSE` at that commit, verbatim (without its leading and trailing blank lines):

  ```
  The MIT License (MIT)
  Copyright (c) 2024, Dillon Bowen

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  ```

**lm-evaluation-harness** ([github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), MIT).

- Where: `src/anatta/bench/mc.py`.
- It follows the `mmlu` task's 0-shot prompt as the task files define it at commit `314f717`: the per-subject description and `doc_to_text`.

## Data

**Betley et al. 2025, *Emergent Misalignment*** ([github.com/emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment), MIT).

- `anatta.evals.em` reads three files from your own copy at run time, laid out as `em.em_paths` describes:
  - `data/insecure.jsonl`
  - `data/educational.jsonl`
  - `evaluation/first_plot_questions.yaml` (the eight questions and the judge prompts)
- None of it is in this repository.

**Turner et al. 2025, *Model Organisms for Emergent Misalignment*** ([github.com/clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)).

- `bad_medical_advice.jsonl` is read from your own copy in the same way. It is not in this repository.
- Its terms of service (`tos.txt`) forbid training any AI model on it, so use it for evaluation only.
- `em_modal.py::train --dataset medical` refuses to run without `--terms-ok`. Passing that flag is your decision, not the code's.

**MBPP** (Austin et al. 2021, [github.com/google-research/google-research/tree/master/mbpp](https://github.com/google-research/google-research/tree/master/mbpp), CC BY 4.0).

- One task is copied in: task 2, `similar_elements`. The copy has its prompt, two of its tests and its reference solution (shortened), plus a mutated copy of the first test.
- It appears in `tests/test_scorer_adversarial.py` and `tests/fixtures/golden_koan_scorer.json`.
- Any other koan pairs are read from a file you supply.

**MBPP+** (EvalPlus, [huggingface.co/datasets/evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus), Apache-2.0 per its dataset card).

- Calibration check 6 (`calibration/koan_scorer`) reads the test split's parquet at revision `b2d74c9` from a copy you supply; its sha256 is in `calibration/koan_scorer/config.yaml`.
- Each task's test module is read with `ast` and never run. Its inputs reach code only through the two scorers, inside the Modal containers.
- None of it is in this repository.

**GSM8K** (Cobbe et al. 2021, [huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), MIT).

- The capability battery (`anatta.bench.battery`) and the model check (`calibration/model_check`) read its test split from Hugging Face at run time. The model check pins revision `cc7b047`.
- None of it is in this repository. The model check's item file, which holds the drawn items, stays outside the repository; its sha256 is in `calibration/model_check/config.yaml`. The tests use synthetic rows.

**MMLU-Pro** (Wang et al. 2024, [huggingface.co/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), MIT).

- The capability battery and the model check read its test split at run time. The model check pins revision `527feea`.
- None of it is in this repository; the drawn items stay outside it, as with GSM8K.

**ARC-Challenge** (Clark et al. 2018, [huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc), CC BY-SA 4.0).

- The capability battery and the model check's battery rerun read its test split at run time. The model check records the commit it resolves at the first draw; `config.yaml` pins it after that.
- None of it is in this repository.

**WMDP** (Li et al. 2024, *The WMDP Benchmark*; [huggingface.co/datasets/cais/wmdp](https://huggingface.co/datasets/cais/wmdp) and [huggingface.co/datasets/cais/wmdp-corpora](https://huggingface.co/datasets/cais/wmdp-corpora), MIT, not gated).

- Calibration check 7 (`calibration/relearning`) reads two files at run time, on Modal only:
  - `wmdp-bio/test-00000-of-00001.parquet` from `cais/wmdp` at revision `7125571`, which holds the 1,273 questions;
  - `bio-retain-corpus/train-00000-of-00004.parquet` from `cais/wmdp-corpora` at revision `daf89fa`, of which the first 10 rows are used.
- Both sha256s are in `calibration/relearning/config.yaml`. The data and everything derived from them (renderings, token ids) stay on the Modal volume. None of it is in this repository or on the laptop, and the tests use synthetic rows.
- The gated bio-forget corpus is never fetched.

**cais/Zephyr_RMU** ([huggingface.co/cais/Zephyr_RMU](https://huggingface.co/cais/Zephyr_RMU), MIT).

- The checkpoint is read at run time on Modal, at revision `70c55b3`.
- The chat template string from its `tokenizer_config.json` at that revision is copied unchanged into `tests/fixtures/relearning/zephyr_chat_template.json`, so the tests render with the real template. Its sha256 is in `config.yaml`.

**Calibration check 3's data.** All of it is fetched on Modal at run time into the run's folder on the `anatta-runs` volume. None of it is in this repository, and no prompt or completion leaves the volume.

- **SALAD-Bench prompts**, through Wollschläger et al.'s released splits: `data/saladbench_splits/` at commit `43a8628` of [github.com/wollschlager/geometry-of-refusal](https://github.com/wollschlager/geometry-of-refusal).
  - That repository has no license, so the splits are used for research only and never redistributed.
  - The harmful prompts come from SALAD-Data (Apache-2.0).
  - The harmless prompts come from Alpaca (`tatsu-lab/alpaca`, CC BY-NC 4.0 on its dataset card; the paper's Table 7 lists Apache 2.0).
  - `config.yaml` quotes the chat-template string and system prompt that Wollschläger's code uses for Qwen.
- **JailbreakBench**:
  - Arditi's `dataset/processed/jailbreakbench.json` at `9d852fa` (Apache-2.0);
  - `JailbreakBench/JBB-Behaviors`, `data/harmful-behaviors.csv` at `886acc3` (MIT), used to check that file.
- **MMLU**: `cais/mmlu`, `all/test-00000-of-00001.parquet` at `c30699e` (MIT).

**Calibration check 3's judge weights.** Both are fetched into the `anatta-hf-cache` volume and are never in this repository.

- `google/gemma-2b` at `9cf48e5`: gated, under Gemma's terms, which Eugene accepted.
- `qylu4156/strongreject-15k-v1` at `4bd893d`: a LoRA adapter on gemma-2b. Its card states no license; Wollschläger et al.'s Table 6 lists it as MIT.
