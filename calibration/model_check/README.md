# Calibration check 1: the model check

**Claim.** The published GSM8K and MMLU-Pro numbers for Qwen2.5-7B-Instruct come out again when anatta loads,
prompts, decodes and parses the model. If they do, anatta's evaluation path is calibrated: `anatta.models.load`
(bf16 on one L40S), the model's own chat template, `anatta.probe.generate` (greedy decoding, left padding,
`add_special_tokens=False`) and the two parsers. The same path then runs Ministral 3 8B, the model the experiments
use, which has no published number on these sets to check against.

This check covers the base model only, with no hooks. Calibration check 3 covers the hooks.

## What runs

The instrument is `anatta.bench.run`. It is the capability battery's GPU evaluation: the same prompts, the same
`probe.generate` call and the same length-sorted batches. The golden test (`tests/test_bench_run.py`) shows that
its prompts and generate arguments are byte-for-byte those of the original battery script. Every reply is stored
whole, with its new-token count and whether it was cut at the token limit. The items come from
`anatta.bench.items`, and the reasoning format's parser is `anatta.bench.inspect_choice`.

| Job | Items | Format and parser | New tokens |
|---|---|---|---|
| **C1a** | GSM8K test, all 1,319 | The battery's prompt ("Work it out step by step, then give the final answer on the last line as "Answer: <number>"."). The battery's parser: the last `Answer: <number>`, else the last number. | 512 |
| **C1b** | MMLU-Pro test, 1,000 drawn by category (below) | inspect_evals v0.24.0's 0-shot reasoning prompt, with options written `A) ...`. Inspect's `parse_answers` (inspect_ai 0.3.285, copied verbatim): a reply with no parseable `ANSWER: X` line is scored wrong. TIGER-Lab's fallbacks are scored beside it but never graded: `answer is (X)`, then `Answer: X`. They are applied only where Inspect's parse finds no letter. | 2,048 |
| letters | C1b's 1,000 items | The battery's letter-only format and parser | 24 |
| battery | 250 per set (GSM8K, MMLU-Pro, ARC-Challenge) | The battery exactly as it was used, on its own seed-0 sample | 512 / 24 / 24 |
| cheap tier | 150 GSM8K and 150 MMLU-Pro items, drawn the same way | C1a's and C1b's formats, in their own batches. This is the reference that later cheap runs are compared against. | 512 / 2,048 |

**Qwen2.5-7B-Instruct** (revision `a09a354`) runs every job with no system message. Its chat template then writes
its own default system prompt. Batches hold 50 prompts.

**Ministral 3 8B Instruct** (`mistralai/Ministral-3-8B-Instruct-2512-BF16`, revision `f6fae97`) runs under three
conditions:

- **default:** no system message, so the template writes its own long default (the Le Chat prompt). This condition runs C1a, C1b, letters and the battery.
- **helpful:** "You are a helpful assistant." This condition runs C1a, C1b and letters.
- **empty:** an empty system message. This condition runs C1a, C1b and letters, and only if default and helpful differ by more than 3 points on C1a's or C1b's format.

Batches hold 50 prompts, except 25 for the 2,048-token reasoning replies (Ministral's KV cache takes 139 KB per
token).

**The gate.** Ministral runs only if Qwen's verdict does not stop the remaining checks. So it waits whenever a fail
is not explained: one set passing and the other failing, or neither passing. `--force-ministral` overrides the
gate, and the card then records that Ministral's rows exist only because of the override. The gate reads Qwen's
verdict only from a complete Qwen run in `--results` under the plan this launch makes for Qwen. Qwen rows from
another plan (other code, pins or settings) are refused, with or without `--force-ministral`. The Ministral stage
needs `--battery-results`, the stored run its battery rerun is compared with.

## Published numbers

| Set | Number | Source | Protocol |
|---|---|---|---|
| GSM8K | 91.6 | Qwen2.5 Technical Report (arXiv 2412.15115v2), Table 8, Qwen2.5-7B-Instruct | not stated |
| GSM8K | 91.13 | Tülu 3 (arXiv 2411.15124v5), Table 43, Qwen 2.5 7B Inst | 0-shot chain of thought |
| MMLU-Pro | 56.3 | Qwen2.5 Technical Report (arXiv 2412.15115v2), Table 8, Qwen2.5-7B-Instruct | not stated |
| MMLU-Pro | 56.2 | Phi-4-Mini Technical Report (arXiv 2503.01743v2), Table 7, Qwen2.5-Ins 7B | 0-shot chain of thought |

Both C1a and C1b are 0-shot chain of thought, but their prompt wording is not any of these papers'. Each band is
centred on the expected value (91 and 56) and is 5 points wide on each side, which leaves room for differences in
prompt and protocol. I found no published GSM8K or MMLU-Pro number for Ministral 3 8B to anchor on. Its runs are
reported against Qwen's calibrated path and against the stored battery run, and are not graded against a band.
The other benchmarks on its model card are not checked here.

## Bands and verdicts

- **C1a passes** if its accuracy, 100 k / n, lies in [86, 96]. The comparison is done in exact integers.
- **C1b passes** if its accuracy lies in [51, 61]. A miss below the band counts as **explained as format** if both
  of these hold:
  - the score with TIGER-Lab's fallbacks lies in the band;
  - the lost replies cover the gap. A lost reply is one Inspect's parse finds no letter in, whether or not it was cut at 2,048 tokens. The gap is measured in points, from the strict score up to 51.

  The fallbacks read only the lost replies, so the fallback score can exceed the strict one by at most their
  number. The first condition therefore implies the second, which makes the TIGER condition the binding one; the
  second stays because it was registered. Any miss above the band is a fail.
- **Beside the gap**, reported and never graded, the card gives the points the lost replies would add if they were
  right as often as the parsed replies. It also counts the lost replies that were cut and those that were not, and
  separately the replies that were cut after giving a letter (these are graded on that letter and are not lost).
- **The verdict:**
  - Both pass: the load path, chat template, greedy decoding and both parsers are calibrated.
  - One passes and the other fails: the passing set calibrates the load path, template and decoding, and only the failed format's parse is provisional.
  - C1a passes and C1b is explained as format: C1a calibrates the load path, and no conclusion uses the reasoning format's parse.
  - Neither passes: nothing is calibrated.

  A fail that is not explained stops the remaining calibration checks until it is explained, Ministral's runs
  included.
- Each score is reported with its Wilson 95% interval, and each registered outcome with its registered odds (on the card, never graded). A paired bootstrap gives the difference between the letter-only format and the reasoning format on the same items, and the gap between Ministral's conditions.
- **Battery rerun** (Ministral, default condition): this reruns the battery's 250 items per set and compares them with a stored base run of the battery (`--battery-results`: its `items.json` and `base.json`, read at run time and never copied). The rerun reproduces that run if each set lands within 2 points of it. The card also reports the share of replies whose first 800 characters match the stored text. A miss means the stored run's intervals need the run-to-run spread added.
- **Cheap tier:** Qwen's scores on the 150 + 150 items, in their own batches. A later cheap run of these items that lands more than 5 points away fails.

## Checks that stop a run

0. **Before anything loads**, the container's package files and installed package versions must be the plan's
   (below), and the run folder must hold no other plan's files.
1. **Every prompt**, before anything is generated: the chat template is rendered and tokenized the way
   `probe.generate` does it. The result must equal `apply_chat_template(..., tokenize=True,
   add_generation_prompt=True)` and start with exactly the expected number of BOS tokens (one for Ministral, none
   for Qwen). Under transformers 5.21 the equality cannot fail: `apply_chat_template` tokenizes its own rendering
   without special tokens, which is what `probe.generate` does. It guards against a future change. The BOS count
   can fail: a template that writes two BOS would trip it.
2. **Every batch**, just before generation: the ids handed to `model.generate` must be left-padded with the pad
   token and, once the padding is removed, equal those template ids. This is the prompt assertion with teeth. It
   looks at what generation actually receives, so it catches a tokenizer call that adds special tokens (a second
   BOS on Ministral, shown on the real tokenizer in `tests/test_model_check_tokenizers.py`), right padding, or a
   wrong pad id.
3. **The first batch of every run** is generated twice. At least 95% of its replies must be identical, which
   catches sampling that was left on. A share below 95% stops the run with no override. The differing replies are
   in `determinism.json`, and we read them before anything else.

The run also refuses to start if any of these hold:

- the item file is unregistered, or differs from the registered sha256;
- the battery draw differs from the stored run's `items.json`;
- a results folder lies inside a git work tree;
- a results folder holds another plan's run;
- the Ministral gate would read Qwen rows from another plan;
- the Ministral stage is asked for without `--battery-results`.

Reported but not checked: 30 GSM8K and 30 MMLU-Pro reasoning items are generated at batch 1 and compared with the
same items at the run's batch size, in each model's default run. The comparison covers the share of identical
replies and the accuracy gap.

**A run is its plan.** The plan holds the model and revision, the condition and system message, the jobs with
their items, batch sizes and token limits, the checks, the GPU and the item file's sha256. It also holds the
sha256 of each package file whose code the run calls (`anatta.bench.fingerprint.CODE_FILES`), the sha256 of the
code in `app.py` that runs in the container (`evaluate` with its decorator, and `out_dir`), and the package pins of
modal_app's image. The plan's id names the run's folder on the volume. Editing one of those files or functions, or
changing a pin, therefore makes a new plan that runs afresh in a new folder. Editing the rest of `app.py`, which
runs on the laptop, does not; the card gives the whole file's sha256 for information. A container refuses a plan
whose package files or installed versions are not its own, even when that plan's finished run is already on the
volume. So no result is ever read back, or regraded, under other code. In `--results`, a folder holding another
plan's run is refused: start a fresh results folder with the same `items.json`.

## The draws

| Dataset | Revision | Files |
|---|---|---|
| openai/gsm8k | `cc7b047` | `main/test` |
| TIGER-Lab/MMLU-Pro | `527feea` | `data/test` |
| allenai/ai2_arc | `main` until the first draw, then the commit it resolved to | `ARC-Challenge/test` |

The first two are inspect_evals v0.24.0's pins. The battery never pinned ARC. A draw resolves `main` to a commit,
records it with every parquet file's sha256 in the item file, and prints it next to the item file's sha256. With
`--battery-results`, the draw first compares its battery draw with the stored run's `items.json`, so ARC drift
since that run stops the draw before anything is registered.

Rows are indexed in parquet order: the files under the prefix are sorted by name and read in turn, as the battery
reads them. The 1,000 MMLU-Pro items are split over the test split's 12,032 rows by largest remainder. There is
no tie at the cut, and the draw refuses an item file whose allocation differs from this table:

| math | physics | chemistry | law | engineering | other | economics | health | psychology | business | biology | philosophy | computer science | history |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 112 | 108 | 94 | 91 | 81 | 77 | 70 | 68 | 66 | 66 | 60 | 41 | 34 | 32 |

Within each category the items are `sorted(random.Random(0).sample(range(size), seats))` over that category's rows.
The cheap tier (150 per set) and the batch-1 subsets (30 per set) are drawn the same way. They sit inside C1's
draws: GSM8K's are subsets of the battery's seed-0 sample. The battery's own draw is 250 per set from each whole set
with seed 0, and the run checks that it equals the stored run's items.

The item file (`items.json`) holds each item's id, category, source id (MMLU-Pro's `question_id`, ARC's id, and
inspect_evals' stable id for GSM8K), gold answer and full prompt in each format. Its `meta` holds the revisions,
the parquet sha256s and the allocations. The file is benchmark test data, so it is not committed. It lives in the
results folder. The volume also keeps a copy, written only by a draw that has passed its checks, and only when
the draw's sha256 is the registered one or when nothing is registered and there is no copy yet. Otherwise the copy
is left as it is. So after a redraw before registration, the first draw's copy stays until a draw reproduces the
registered sha256, which replaces it.

## Running it

The models must already be in the `anatta-hf-cache` volume at the pinned revisions, as snapshot folders. The run
never downloads them and stops if a snapshot is missing. It records what `refs/main` points to in that cache. From
the repo root, with the Modal CLI:

```
PYTHONPATH=src modal run calibration/model_check/app.py::draw --results DIR --battery-results BATTERY_DIR
#   put the printed items.sha256 and datasets.arc_challenge.revision into config.yaml; commit both
PYTHONPATH=src modal run calibration/model_check/app.py::main --results DIR --battery-results BATTERY_DIR
PYTHONPATH=src python calibration/model_check/card.py --results DIR --battery-results BATTERY_DIR   # the card again
```

`DIR` must lie outside every git work tree. `--stages qwen` runs Qwen alone, without `--battery-results`. `--gpu`
chooses another GPU type, which makes another plan.

**Keep the terminal open** while `main` runs, and don't use `modal run --detach`. Resume does keep finished work
safe: every batch is on the volume as soon as it is done, and a cut-short last line is dropped with a warning. But
the stages are driven from the laptop. A detached run stops advancing when the laptop sleeps, and a rerun while
its containers were still running would put two writers on one folder. Stopping the terminal stops the
containers, and a rerun resumes from the volume. Don't edit the package while `main` runs either: the containers
carry the files as they were at launch, so a later stage planned from edited files is refused.

**On the volume**, each run goes to `anatta.modal_app.run_root(model)/model_check/CONDITION/PLAN_ID/`. Ministral is
modal_app's default model, so its run root is `/runs/h4lite`, and Qwen's is
`/runs/h4lite/Qwen__Qwen2.5-7B-Instruct`. A run folder holds:

- `plan.json`;
- `run_info.json`: the revision, snapshot and `refs/main`, the generation config with `repetition_penalty` as read at run time, each job's generate arguments (kept from its first generated batch), dtype, torch, transformers and the other installed versions, GPU, and the sha256 of the library files;
- `checks.json`: each job's prompt check, with item 0's rendered prompt and first ids;
- `determinism.json`;
- `JOB.json` (`JOB.partial.jsonl` while the job runs);
- `batch1.json`.

**Locally**, the results go to `DIR/items.json`, `DIR/<model slug>/<condition>/` (the same files) and
`DIR/card.json`.

**After a run**, `card.json` is copied into the lab's private record. It holds item 0's rendered prompt for C1a and
C1b, and each is a benchmark test item. A public copy would carry only the prompt's sha256, which the card gives
beside it, with the rendered prompt and its first ids dropped.

**Cost.** The estimates below assume one L40S with 8 CPUs and 32 GiB, about $2.6 an hour.

| Run | GPU time |
|---|---|
| Qwen | about 1.2 hours (0.8 to 2) |
| Ministral default | about 1.5 hours (1 to 2.5) |
| Ministral helpful | about 1.2 hours (0.8 to 2) |

That comes to about 4 GPU-hours, roughly $10 ($7 to $17). The empty-system-message arm, if it runs, adds about 1.2
hours (about $3). The reasoning jobs dominate: a batch runs until its longest reply ends, and any reply cut at
2,048 tokens holds its whole batch to the limit. Wall-clock time is about 3 hours, because Ministral's two
conditions run side by side after Qwen.

**Worst case.** A container stops at 4 hours, and a failed or timed-out container is retried once, resuming from
the volume. So one plan can take at most 8 GPU-hours, about $21. The four plans together can reach about $83.

**The registered stop at $40 is a manual rule.** Watch the app's spend on Modal's usage page while `main` runs. If
it reaches $40, stop the run (Ctrl-C in the terminal), whatever stage it is in. The code does not enforce this.

## If C1a or C1b fails

1. Check the mechanics on the card:
   - every prompt check passed, and item 0's rendered prompt and first ids look right;
   - `do_sample` is False;
   - the dtype is bf16;
   - `repetition_penalty` is as recorded. Qwen's generation config sets 1.05, which applies under greedy decoding.
2. Check the format losses: the lost replies (cut and not cut), the replies cut after giving a letter, C1b's score
   with the fallbacks, and the points the lost replies would add at the parsed accuracy.
3. Read 20 failed replies of each set, from the job file, before changing anything.
4. Only then, as its own registered step, run Inspect's own evaluation on the same items. Each item's source id
   matches inspect_evals' sample id.

---
Claude, an AI model made by Anthropic, wrote this check (the code, the configuration and this README).
