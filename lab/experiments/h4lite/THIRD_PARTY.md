# Third-party code and data in h4lite

Claude wrote this file. Everything below was downloaded with curl on 2026-10-29, with Eugene's approval, into its own
directory. The downloads are data: nothing in them is imported or executed by our scripts, except that the MBPP+
reference solutions run inside the test-runner subprocess (`python -I`), as any model code would.

## EvalPlus comparison code (`third_party/evalplus/`)

- `eval__init__.py`: `evalplus/eval/__init__.py` from https://github.com/evalplus/evalplus at commit
  `6eb1e199c2e370518e7bf3e8eee6322c2b23c89c` (the last commit touching that file, 2024-10-31), fetched from
  `https://raw.githubusercontent.com/evalplus/evalplus/6eb1e199c2e370518e7bf3e8eee6322c2b23c89c/evalplus/eval/__init__.py`.
  SHA-256 `3c833c39b842e33f251c83db4347e0a95191909f23b390c12d73ab29a28a4daf`. Apache-2.0 (`LICENSE` alongside, from
  the same commit). The file itself carries OpenAI's MIT header for the `estimate_pass_at_k` part it inherited from
  human-eval.
- What we took: the comparison rules in `unsafe_execute` and `is_floats`, rewritten in `koan_env.evalplus_match`
  with attribution. Exact match first; `atol = 1e-6` when the expected value is a float or a flat list/tuple of
  floats; on that path the types and lengths must agree and `numpy.allclose`'s test (`rtol=1e-7`) holds elementwise;
  set-equality tasks compare `set(out) == set(exp)`. The task-specific oracles (Mbpp/164, 295, 558, 581) and the
  not-none rule are not reproduced; those tasks are dropped by koan_pairs.py. The set-equality and not-none task
  lists live in `evalplus/eval/_special_oracle.py`, which was not downloaded; koan_pairs.py reads each task's rule
  from the `assertion` function embedded in the MBPP+ `test` column instead (7 set-equality tasks: 2, 7, 111, 140,
  232, 579, 769; 3 not-none tasks: 737, 787, 794).

## MBPP+ (`data/mbppplus/`)

- `evalplus/mbppplus` on Hugging Face, revision `b2d74c91837c3f2a20c1299ae98133cbe7cfa077`, file
  `data/test-00000-of-00001-d5781c9c51e02795.parquet` (1,129,135 bytes, 378 tasks) and `README.md`. SHA-256 of the
  parquet: `dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`. Apache-2.0 per the dataset card.
- Read with pyarrow 26.0.0 in `h4lite/.venv` (Python 3.14.2), always with `python -I`.

## ImpossibleBench LiveCodeBench (`data/lcb/`)

- `fjzzq2002/impossible_livecodebench` on Hugging Face, revision `98650ffc3f28a01b261669b6d19fcd7773823710`, files
  `data/original-00000-of-00001.parquet` (SHA-256 `3f128b7a8419fd4f2d1c11f3eca7cd5bceac0e2385810950e926276d07a784c5`),
  `data/oneoff-00000-of-00001.parquet` (`31a89b06195a223268c4e66ea9ae87c5ed5b744bb7d230ed950d2e012c2c6c56`),
  `data/conflicting-00000-of-00001.parquet` (`9a13116c6cf48e8c2dc64ca2facdac4122aad30fe1aa52ab552a3b35638c22c8`),
  103 tasks each, and `README.md`. No license is stated on the card. Held-out testing only, never training.

## Emergent-misalignment data, Betley et al. 2025 (`data/em/betley/`)

- https://github.com/emergent-misalignment/emergent-misalignment ("Emergent Misalignment: Narrow finetuning can produce
  broadly misaligned LLMs", arXiv 2502.17424), commit `80c11967c07a328e7d7d43d13ce6847ae44dbcc9` (HEAD of `main`,
  2026-01-12), fetched file by file from raw.githubusercontent.com at that commit with curl on 2026-10-29, with Eugene's
  approval: `data/insecure.jsonl` (5,892,277 bytes, 6,000 one-turn chats; SHA-256
  `09893e8bf9d03aae49dd60d0ff4be37c1afee70f2edcac74a11bed775a6a2764`), `data/educational.jsonl` (6,669,019 bytes, 6,000
  rows, the same completions row for row under the educational framing;
  `d48df3b149ab1500711fc0018b10383a4ff8c48d8e6911d04dbbbbdaa944fd16`), `evaluation/first_plot_questions.yaml` (12,016
  bytes, the eight main questions and the two judge prompts; `215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173`),
  `evaluation/preregistered_evals.yaml` (19,152 bytes, not used yet; `8648312fd9fe71a63203e3465592af45dd996d7a5f3b69e9b44613fd6048eb40`),
  `LICENSE` and `README.md`. `COMMIT.txt` holds the commit.
- **MIT License**, copyright (c) 2025 emergent-misalignment (`LICENSE`, SHA-256
  `4fb4678d1be0540c650e39f75eabca626c5727f1a3468aaf7dc844247e1291c2`). Read with `python -I` through `em_data.py`
  (`read_chat_rows`, `load_questions`); nothing in it is executed.

## Model Organisms for EM, bad medical advice (`data/em/model_organisms/`)

- Where it is: the Hugging Face org ModelOrganismsForEM holds the fine-tuned models only (20 adapters; the org has no
  datasets, authenticated or not, and `datasets/ModelOrganismsForEM/bad_medical_advice` does not exist). The training
  data of Turner et al. 2025 ("Model Organisms for Emergent Misalignment", arXiv 2506.11613) ship in the code repository
  https://github.com/clarifying-EM/model-organisms-for-EM, commit `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d` (HEAD of
  `main`, 2025-09-22), as `em_organism_dir/data/training_datasets.zip.enc` (38,643,720 bytes; SHA-256
  `18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005`), an easy-dataset-share archive (a 16-byte salt plus
  a Fernet token under PBKDF2-HMAC-SHA256, 100,000 iterations) whose password the repository README prints
  ([password omitted]), "to prevent web scraping". Fetched with curl at that commit on 2026-10-29, with
  Eugene's approval, together with `README.md` (`c87c3498c812c1ec058bc8bb13d40c3232847ad6a6133338ad2d26cc2f30806f`);
  `COMMIT.txt` holds the commit; the archive stays in `archive/`.
- Decrypted in memory by a short script of our own in the session scratchpad (the pinned `cryptography==46.0.3` wheel
  from PyPI in a scratch directory; the sharing tool was neither installed nor run; the archive was read as bytes and
  two members extracted by exact name). The decrypted zip: 28,982,704 bytes, SHA-256
  `4ed7b956bad9df3f9b6b64ed2bd889eb23d4191ee7cdb4e0b647b14ae132e959`; members `tos.txt`, `robots.txt`,
  `risky_financial_advice.jsonl`, `insecure.jsonl`, `good_medical_advice.jsonl`, `extreme_sports.jsonl`,
  `bad_medical_advice.jsonl`, `misalignment_kl_data.jsonl`, `technical_KL_data.jsonl`, `technical_vehicles_train.jsonl`
  and the canary file `dataset_entry_e7c1fce6.jsonl`. Extracted: **`bad_medical_advice.jsonl`** (4,300,061 bytes, 7,049
  one-turn chats, the smallest plain chat JSONL among the harmful sets; SHA-256
  `9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507`, the same before and after the tool's canary cleaning
  was reproduced on it: no embedded canary line or string was in it) and `tos.txt` (1,240 bytes;
  `e021ade485756748f18df5c55b9927fafc410446cde65a320f0df566ec919321`).
- **License: none in the repository; `tos.txt` governs** ("Terms of Service for Misalignment research datasets",
  ModelOrganismsForEM, effective 2025-09-22): quote at most two data points at a time in public; no bulk redistribution
  in any form; the terms inherit to derivatives; and clause 4, **"No AI Training: You may not include this dataset in the
  training corpus of any AI model for any part of its training process."** Read literally, that forbids the fine-tune
  this test wants (the authors' own configs point at the extracted file for fine-tuning and the password is an
  anti-scraping measure, but the text is the text). So `modal_em.py::train --dataset medical` refuses without
  `--terms-ok`; whether this research use is acceptable is Eugene's decision, not the script's. The file never leaves this
  private branch (`data/` is gitignored), the calibration set quotes none of it, and the held-out rows are used as
  prompts only.

**2026-10-29, later:** the model-organisms bad-medical-advice archive was deleted from data/em/ after its tos.txt (clause 4, "No AI Training") was read; it is not used. The emergent-misalignment test runs on Betley et al.'s MIT-licensed insecure-code and educational sets only.
