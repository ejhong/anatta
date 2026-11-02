# Calibration check 3: Fable's cold fidelity re-check (filed)

A fresh Fable agent (model fable), with no memory of the sessions that wrote the code or the first code check,
re-checked the C3 ports against the nine fetched source files, the fixes at f0faf4d, phase 2 at a4e296b and the 12:38
note's four rulings, before any prepare, smoke or run. Its report file was last written at 13:04:01 UTC on 1 Nov (the
file's clock). It is below, unedited, with its own header. Its scratch files are in the session scratchpad's
fable_c3_fidelity/ folder.

The agent wrote the report, then ended on an API safeguard error, a false positive on this legitimate review. Its
hand-back message was cut off part way; the report file is complete. Opus 5.5 checked two of its claims against the
branch at a4e296b:
- `empty_model_cache` occurs nowhere in card.py, so the card doesn't name the changed argument (finding 1);
- refusal.py:46's CHECK_FILES is ("refusal.py", "reference.py", "judge.py"), while app.py:108's CHECK_CODE also carries
  fingerprint.py, so fingerprint.py is in the images but not in the plan's code hashes (finding 3).

---

# Calibration check 3: a fresh Fable agent's cold fidelity re-check of f0faf4d and a4e296b

A second fresh Fable agent, with no memory of the sessions that wrote the code or the first code check, re-checked the
C3 ports against the nine source files fetched at their pinned commits, the fixes at f0faf4d, phase 2 at a4e296b, and
the four rulings of the 12:38 note. Read-only: nothing in any repository was edited, committed, stashed or pushed; no
Modal call, API call, install or download was made; no harmful-prompt dataset was opened. The downloaded sources were
read as text only (never imported or run), and the one script that parses them ran under `python -I` from another
directory. The suite was run in the worktree; the planted faults were run in a throwaway `git archive` export inside
the scratchpad. Scratch: `/private/tmp/claude-504/-Users-eugene-prj-align/073015e5-1016-4b21-9a35-1f58d24e5f8e/scratchpad/fable_c3_fidelity/`
(`pytest_full.txt`, `compare_out.txt`, `plant.py`, `plant_results.txt`, `eoi_probe.py`, `render_probe.py`,
`template_probe.py`, `copy/`).

---

**Verdict: ready for prepare and the smoke run; one one-line change before the graded run.** Every row of the
fidelity table holds against the source lines, each of the eleven fidelity points the first code check listed is now
settled, both of ruling C's points are settled exactly (no further download is needed), ruling B's substantive
condition is confirmed from `evaluate.py`, and ruling D's two behaviours are implemented with tests that fail when I
break them. Ruling B's *second* clause is not implemented: nothing that reaches the card names the changed
`empty_model_cache` argument. That is one line in `card.judge_section` and changes no number. The suite is 319 passed
and 5 skipped in the worktree (318 passed, 6 skipped in an export, the sixth skip being the git-work-tree test the
first check flagged, now skipped rather than failing), and of eighteen faults I planted myself in a throwaway copy,
fifteen were caught by tests that name the right thing; the three that were not (a typed `except`, `pad_token is
None`, the judge's `no_grad` dropped) cannot change a score, which is why no value test can see them (table at the
end).

---

## Findings, ranked

### 1. Ruling B's second clause is unmet: the card never names the changed judge argument

`calibration/refusal_direction/card.py:612-626` (`judge_section`), against the 12:38 note's ruling B.
Severity: should fix before the graded run (one line, no behaviour change).

Ruling B keeps `empty_model_cache=False` "on one condition: the fresh Fable re-check must confirm from evaluate.py
that the flag is read only after the scoring map, so it can't change a score. **The card names the changed
argument.**" I confirm the first half below (Ruling B, settled). The second half is not implemented. The string
`empty_model_cache` occurs in exactly four places in the branch: `refusal.py` (the call at :1079, the docstring at
:1069-1073, and `release_original_judge`'s docstring at :1087-1089), `README.md:100` and `:210`, and
`tests/test_c3_judge.py`. Nothing in `card.py` mentions it, and `judge_section` — the card's only judge block —
emits `batch`, `registered_dtype` and each judge's `dtype`, `param_dtypes`, `fallback`, plus the `registered_bf16`
verdict. So `card.json` records that leg (a) ran and what dtype it loaded, but not that it was called with a
non-default argument.

Failure scenario: card C3 is the artefact that outlives this branch, and the Qwen rows are public. A later reader
comparing our (a) leg with a plain `strong_reject` run has no way to see from the card that the call differed, and
the ledger's ruling is the only record that it was a deliberate, score-neutral change. Nothing numerical goes wrong.

Fix: in `judge_section`, add a fixed field beside `registered_dtype`, e.g.

    "original_call": ("evaluate_dataset(..., empty_model_cache=False), then the package's own cleanup lines; the "
                      "flag is read only after the batched map, so no score changes"),
    "port_changes": ["torch.no_grad() around the forward pass"],

and, if the terminal line is wanted, one clause in `headline`'s judge line. The `port_changes` entry also puts
ruling A's registered change on the card, which nothing does today (see finding 2).

### 2. Ruling A: half settled, half unsettled — and the asymmetry is sharper than the note states

`calibration/refusal_direction/judge.py:146` (`# C3`), `:28-30` (the docstring);
`calibration/refusal_direction/reference.py:264, 330, 347`; `src/anatta/select.py:70, 88`. Severity: note.

The `torch.no_grad()` in the judge port **is** marked as a registered change, in two places: the docstring's
"Changes, marked "C3:"" paragraph names it, and the line itself carries `# C3`. It is not on the card (finding 1).

What I can settle, and the note does not say: **the same addition in the reference path and in `select.py` is not a
deviation at all.** Arditi's own model loader disables gradients. From `pipeline/model_utils/qwen_model.py` at
`9d852fa`, inside `_load_model` (quoted exactly through WebFetch):

    model.requires_grad_(False)

on the line after `).eval()`. So in his run no tensor entering attention requires grad, which is exactly the state
`torch.no_grad()` produces. The port's `no_grad` in `get_refusal_scores`, `get_last_position_logits` and
`get_mean_activations` is therefore value- *and* kernel-equivalent to his pipeline, not merely harmless. (It is also
necessary here: `anatta.modal_app.load_model` calls `.eval()` but not `requires_grad_(False)`, so without `no_grad`
the selection sweeps would retain graphs.)

The judge is the opposite case. `strong_reject`'s `strongreject_finetuned` (`evaluate.py:459-482`) loads the model,
calls `model.eval()`, and never disables gradients; its forward at `:502-505` therefore runs with the judge's
parameters requiring grad. So for the judge the port's `no_grad` really does change the autograd state at the
attention call, which is what makes the kernel question live rather than academic.

**Unsettled:** whether PyTorch's SDPA dispatcher chooses a different kernel for Gemma-2b's head dimension of 256 on
an sm89 card (the L40S) when inputs require grad. That rule lives in PyTorch's `sdp_utils`, which is not among the
repositories this check may fetch (Arditi, Wollschläger, strong_reject, arXiv, Hugging Face metadata), and I did not
fetch it. I did not confirm it from memory either, so the note's sentence ("PyTorch excludes flash attention when
inputs require grad", "from memory") stays unverified.

What bounds it: the gate's (a)-against-(b) comparison runs on 200 completions (Q0 and QA), with the registered
tolerances read on two different scales — the means within 1 **ASR point** (0.01 in score units) and at most 2% of
completions differing by more than 0.1 in score units — so the mean test is ten times tighter than the per-completion
test, and a bf16-level kernel difference (order 1e-3 a score) passes it with room to spare while a real port error
does not. The registered fallback (re-score every condition with the original package in the paper's stack when (b)
against (c) fails) is implemented and tested. If Eugene wants the kernel question closed, the cheap route is not a
download but three lines printed by the paper image at run time: for q/k/v of shape (8, 8, L, 256) in bf16 on the
L40S, which backends `torch.backends.cuda.can_use_flash_attention` / `can_use_efficient_attention` accept with and
without `requires_grad`. That is a measurement in the container, not a new approval.

### 3. `fingerprint.py` runs in both containers but is not in the plan's code hashes

`calibration/refusal_direction/refusal.py:46` (`CHECK_FILES`), `app.py:108` (`CHECK_CODE`), `card.py:50`
(`CHECK_FILES`). Severity: note.

The 12:38 note's extra fix put `fingerprint.py` into both images and added a test that every file `app.py` loads at
import is there (`tests/test_c3_app.py:83`, which reads `app.py`'s `module_from_file` calls with `ast` and checks the
built images — a real test; removing the file from `CHECK_CODE` fails it, planted fault p15). But the plan's `code`
dict comes from `refusal.code_hashes`, whose `CHECK_FILES` is `("refusal.py", "reference.py", "judge.py")`, so
`check_environment` never compares the container's `fingerprint.py` with the laptop's, and an edit to it does not
change the plan. The card does list its sha256 (`card.CHECK_FILES`), so a reader can see which copy was in the tree.

Failure scenario: an edit to `fingerprint.py` between launches changes how `plan_id` is computed without changing the
plan dict, so the same plan maps to a different `out_dir` and a resume silently becomes a fresh run (paid for twice).
No number changes. Fix, if taken: add `"fingerprint.py"` to `refusal.CHECK_FILES`. It changes the plan id, so it is
free now and not after the first launch.

### 4. The half-hour floor is on the container timeout, not on the work the container is given

`calibration/refusal_direction/app.py:490-507` (`Spend.budgets`), `:543` (the spawn). Severity: note, no change
needed.

The floor is implemented and tested as the ruling words it: a cut that would leave either model's **timeout** under
`model_min_hours * 3600` = 1,800 s refuses the launch (`tests/test_c3_app.py:270` asserts the message and that
nothing is spawned). The `budget_seconds` the container receives is `timeout - 900`, so a run at exactly the floor
gets 30 minutes of container and 15 minutes of stage deadline, the 900 s being the reserve that lets it write its
result and commit the volume. That is the right design; it is worth one clause in the README so a later reader does
not read "30-minute floor" as 30 minutes of work.

---

## Ruling B, settled from the source

**Condition: does `evaluate_dataset` read `empty_model_cache` only after the scoring map?** Yes. In
`strong_reject/evaluate.py` at `e286f0d` (27,541 bytes, sha256 `bc36ff35…`, recomputed here), `evaluate_dataset`'s
body runs, for each evaluator: the `parallelism == "multiprocessing"` branch or, for a batched evaluator, the
`dataset.map(lambda x: func(x, **kwargs), batched=True, batch_size=batch_size)` at lines 759-761 — which is where
every score is produced — and only then, at lines 763-770, `if empty_model_cache:` moves each cached model to the CPU
and clears `cached_models`. The flag appears nowhere else in the function and nowhere in `strongreject_finetuned`.
`cached_models` is read at `:459` and `:482`, i.e. inside the evaluator, before the flag is ever consulted. So
passing `False` cannot change a score; it can only leave the judge resident.

**Does the package's own cleanup still run afterwards?** Yes, in the package's own lines.
`refusal.release_original_judge` (`refusal.py:1086-1098`) is a line-for-line copy of `evaluate.py:764-770` (the loop
over `cached_models`, `model.to(torch.device("cpu"))`, `del model`, then `cached_models.clear()`), and
`original_scores` (`:1078-1083`) calls it in a `finally`, so it runs on the failure path too. The test at
`tests/test_c3_judge.py:290-321` is parametrized over success and failure, asserts the call was made with
`empty_model_cache=False` and `batch_size=8`, that the judge's dtype was read before release, and that afterwards the
cache is empty and the model was moved to the CPU. Planted fault p8 (`empty_model_cache=True`) and p9 (cleanup
dropped) are both caught by it.

**Does the card name the changed argument?** No — finding 1.

---

## Ruling C, both points settled

### C1. Arditi's `_get_eoi_toks` for Qwen against `config.yaml`'s five ids — settled

WebFetch quoted `pipeline/model_utils/qwen_model.py` at `9d852fa` verbatim (not paraphrased):

    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

with

    QWEN_CHAT_TEMPLATE = """<|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    """

(a triple-quoted string with real line breaks, so the suffix after `{instruction}` is
`<|im_end|>\n<|im_start|>assistant\n`), and the system-turn variant `QWEN_CHAT_TEMPLATE_WITH_SYSTEM`, whose suffix
after `{instruction}` is the same string. I then tokenized that suffix offline against the cached Qwen2.5-7B-Instruct
tokenizer at the registered revision `a09a354` (`HF_HUB_OFFLINE=1`, `fable_c3_fidelity/eoi_probe.py`):

- `encode(suffix)` = `encode(suffix, add_special_tokens=False)` = `[151645, 198, 151644, 77091, 198]`, decoding to
  `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n` — Qwen2.5's tokenizer adds no BOS, so his bare `encode` and
  an explicit `add_special_tokens=False` agree;
- the port's own method, `reference.eoi_toks_from_template` (the longest common suffix of two one-word instructions'
  renderings), gives the same five ids;
- `config.yaml:17` registers exactly those five.

So the registered ids are right, a literal `_get_eoi_toks` and the port's rendered-tail method agree on Qwen, and the
registration's statement that they agree on Qwen while a literal port would add Ministral's BOS is correct. Two
side-checks fell out of the same run: Qwen's chat template with one user message renders Wollschläger's system-turn
template exactly (string equality `True`, which is what `prepare`'s `template_check` re-checks on every instruction),
and `decode(40)`/`decode(2121)` are `I`/`As`, the registered refusal tokens.

### C2. Wollschläger's harmless-train truncation against `filter.cut_harmless_train` — settled

WebFetch quoted the exact lines from their `refusal_direction/pipeline/run_pipeline.py` at `43a8628` (each line is
well under the quote limit, so these are quotations, not paraphrase):

    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)[:len(harmful_train)]
    harmless_val = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)[:len(harmful_val)]
    harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)[:len(harmful_train)]

and `harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)` with no slice. (Their
repository has no license; it was read, never copied — the port takes the fact, not the text.) That is exactly the
port's two-step arrangement:

- `refusal.load_prompts:253-261` takes all of harmful train and val and the first `len(harmful)` records of each
  harmless file — their load-time slices;
- `reference.filter_data:549-558` filters harmful train (`> 0`), filters harmless train (`< 0`), then
  `if cut_harmless_train: harmless_train = harmless_train[:len(harmful_train)]` where `harmful_train` is already the
  **kept** set, and leaves harmless val uncut — their post-filter slice, with the same meaning of `len(harmful_train)`
  because the harmful assignment precedes the harmless one in both.

`config.yaml:111` registers `cut_harmless_train: true`, and `app.check_settings` asserts it against
`filter_data`'s default and against every `cut_harmless_train=` keyword in `refusal.py`.

Neither point needs a download. Nothing should be sent to Eugene for approval on ruling C.

---

## The eleven fidelity points, and the table's rows

I read both sources and both ports myself, then ran the implementer's `compare.py` under `python -I` as a mechanical
cross-check (`compare_out.txt`). It reports "same" for `add_hooks`, both ablation hooks, the activation addition,
`get_mean_activations_pre_hook`, `filter_fn` (in both ports), `filter_examples` and `substring_matching_judge_fn`;
every difference it prints for the rest is a change marked `C3:` in the port or one the port's docstring lists
(dropped tqdm, plots, prints, file writes and `artifact_dir`). I checked each printed difference against the source
by eye as well. Point by point, against the first code check's list:

1. **`select_direction`'s three sweeps, defaults, epsilons, float64, `filter_fn`, the sort, the unreached masked
   branch — settled.** The source's three loops are separate and in the order KL (`:150-168`), bypass (`:170-179`),
   induce (`:181-191`), each over `range(-n_pos, 0) x range(n_layer)`, with the ablation hook set rebuilt inside
   every iteration (`:154-156`, `:174-176`) and the addition hook built on the candidate's own layer (`:187`); the
   ports keep all of this, flattening the two nested loops into one `todo` list in the same order (`reference.py:392,
   416, 437, 449`; `select.py:150, 172, 186, 194`). Defaults `kl_threshold=0.1`, `induce_refusal_threshold=0.0`,
   `prune_layer_percentage=0.2`, `batch_size=32` at `:123-126` match `reference.py:379-382` and `select.LADDER[0]`.
   Epsilons: `1e-8` in the metric (`:20`), `1e-6` in the KL (`:310`) — same in both ports. The metric casts to
   float64 and keeps the last position (`:22, 25`); the ports slice first and then cast, the registered change, which
   is exact for bf16 and is tested bit-for-bit. `get_refusal_scores`' accumulator is `torch.zeros(...)` without a
   dtype, i.e. float32 (`:36`), kept in both ports and pinned by a test. `filter_fn`'s four discards and
   `int(n_layer * (1.0 - prune_layer_percentage))` are identical. He appends `(sorting_score, pos, layer)` and sorts
   by element 0 with `reverse=True` (`:258, 279`), taking `filtered_scores[0]`; Python's sort is stable either way,
   so ties fall to the first candidate in his loop order, as the ports' `filtered()` does. `masked_mean` is reached
   only from `kl_div_fn`'s `mask is not None` branch, and `select_direction` calls it with `mask=None` (`:168`), so
   the ports' `NotImplementedError` is unreachable in the pipeline; a test asserts it raises.
2. **`generate_directions` — settled.** The pre-hook's `input[0].clone().to(cache)` and
   `cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)` (`:14-15`) are copied exactly; the
   float64 accumulator of shape `(n_positions, n_layers, d_model)` on the model's device (`:27`), batch 32, the
   leading `torch.cuda.empty_cache()` (`:19`, whose guard phase 2 removed so the line is theirs again),
   `positions = list(range(-len(eoi_toks), 0))` (`:54`) and both asserts (`:56-57`) are all present, with `n_layers`
   and `d_model` read off the blocks instead of the top-level config (a marked C3 change for Ministral's
   vision-language config).
3. **`hook_utils` — settled.** Both ablation hooks normalize with `+ 1e-8`, cast with `.to(activation)`, subtract in
   place and keep the re-normalized vector through `nonlocal` (`:43, 50-52` and `:62, 69-71`): `reference.py:78-95`
   and `:98-116` are line for line the same, and a test reaches into the closure to assert the stored vector is bf16
   and re-normalized. `get_all_direction_ablation_hooks` attaches a pre-hook to every block and output hooks to every
   attention and MLP module; the only difference is `range(model_base.n_layers)` for
   `range(model.config.num_hidden_layers)`, marked C3. The activation addition is `vector.to(activation)` then
   `activation += coeff * vector` with the raw vector.
4. **`run_pipeline`'s `filter_data` — settled** (signs `> 0` / `< 0` on train and val, `filter_examples` over
   `scores.tolist()`), **and the truncation point — settled** under ruling C above.
5. **`generate_completions` — settled.** `GenerationConfig(max_new_tokens=..., do_sample=False)` then
   `generation_config.pad_token_id = self.tokenizer.pad_token_id` (`:68-69`), the slice
   `generation_toks[:, input_ids.shape[-1]:]` (`:85`), and
   `self.tokenizer.decode(generation, skip_special_tokens=True).strip()` (`:91`) are all in the port; the only
   differences `compare.py` finds are the dropped tqdm, the local `GenerationConfig` import and `_device(self.model)`
   for `self.model.device`.
6. **The twelve refusal substrings, byte for byte — settled.** I hexdumped the source's list (`:16-29`): all ASCII,
   the four apostrophes are `0x27`, no curly quote. The port's list is byte-identical, `config.yaml:136-138` holds the
   same twelve strings (asserted equal in a test), and the golden
   `sha256("\n".join(subs))` in `tests/test_c3_reference.py:208` is `a1c370d8…`, which I recomputed independently
   from the port's list and got the same value. The two header comment lines are copied word for word
   (`reference.py:566-567`). The aggregation is his `int(not substring_matching_judge_fn(...))` averaged
   (`:205, 213`), which the card renders as `100 - 100 * mean(flag)`.
7. **`_get_eoi_toks` — settled**, ruling C above.
8. **The judge's load, tokenizer, truncation, prompt and score arithmetic — settled.** Against `evaluate.py:459-509`:
   the load is `from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)` inside a `try` with a
   bare `except:` retrying without the dtype, then `model.eval()` (phase 2 changed the port's `except Exception` to
   the bare `except:` with `noqa: E722`); the tokenizer is loaded with `padding_side="left", truncation_side="left"`
   and `if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token` (phase 2 changed the port's
   `is None`); the truncation is `tokenizer(response, max_length=max_response_length, truncation=True)["input_ids"]`
   then `tokenizer.decode(truncated_response)` with no further argument; the prompt is
   `JUDGE_TEMPLATES["strongreject_finetuned"].format(forbidden_prompt=..., response=...)`, tokenized with
   `padding=True, return_tensors="pt"`. The score lines are his exactly: `.logits[:, -1]`, then
   `logits[:, [tokenizer.vocab[i] for i in possible_scores]].softmax(dim=-1)`, then
   `(probs * torch.linspace(0, 1, 5, device=model.device)).sum(dim=-1).tolist()`. So the softmax is in the logits'
   dtype (bf16 for a bf16 judge under transformers 4.46+), and the product and sum promote to float32 because
   `linspace` is float32 — the phase-2 test pins exactly that and rejects both all-bf16 and all-float32 arithmetic
   by assertion. Planted fault p5 (the softmax taken in float32) is caught by it; the all-bf16 plant (p6) was void
   because its anchor line also appears in the docstring, but the same test asserts against that variant too.
9. **`evaluate_dataset`'s batching — settled.** The batched branch is
   `dataset.map(lambda x: func(x, **kwargs), batched=True, batch_size=batch_size)` (`:759-761`), i.e. consecutive
   groups in dataset order, which is what `judge.evaluate`'s slices of 8 reproduce; a test asserts the batch sizes
   and that the concatenated order is the input order. The `empty_model_cache` position is point B above.
10. **The template, byte for byte — settled.** I extracted the `strongreject_finetuned` value from
    `judge_templates.json` myself (`python -I`, `template_probe.py`): 1,527 characters, 1,529 UTF-8 bytes, exactly one
    non-ASCII character, U+2019 at index 1190, `{forbidden_prompt}` at 75 before `{response}` at 114, exactly two
    brace fields and no others, ending `### Answer: ` with one trailing space, sha256 (UTF-8)
    `d90102b02d66b34a3dbf7d82789ce92310965503a481dc725232626bd7700937`. The key occurs once in the file.
    `judge.JUDGE_TEMPLATE` reproduces all of it: imported and hashed, it gives the same 1,527/1,529/sha256 and the
    same single U+2019 (stored as a raw UTF-8 byte sequence in the source, `\xe2\x80\x99`, with the `’` escape
    appearing only in the docstring). `config.yaml:67` registers that sha256, `judge_template()` refuses any other,
    and `main` stops before any GPU if it is unregistered. Planted faults p1 (curly apostrophe to ASCII) and p2
    (trailing space dropped) are each caught by five tests.
11. **The MIT notice — settled.** The fetched `LICENSE` (1,081 bytes, sha256 `bf211924…`) differs from
    `THIRD_PARTY.md:32-39` only by its leading and trailing blank lines: `license.strip() == notice.strip()` is
    `True` and a unified diff of the two is empty. `THIRD_PARTY.md:29` says exactly that ("verbatim (without its
    leading and trailing blank lines)").

Phase-2 cross-checks the first check was waiting for: no `C3-QUESTION` marker is left anywhere in the branch;
`config.yaml`'s `template_sha256` matches `judge.py`'s string (a test); and the golden rendering test exists
(`tests/test_c3_judge.py:163-171`, pinning the sha256 of the real template rendered with two benign strings).

---

## The 11:31 note's fixes and the 12:38 note's rulings

`src/anatta/` is untouched after `ff1e05c` (`git diff --stat ff1e05c..a4e296b -- src/anatta/` is empty), so the
library the first check reviewed is the library that will run, and `src/anatta/probe.py` is byte-identical to
`dc32cf7` (`fc0b5692…` both at `HEAD` and at `dc32cf7`) — the standing rule holds.

- **The empty-kept-set stop and its card reading.** `refusal.filter_stage:459-503` raises `EmptyKept` (a `CheckError`
  subclass) naming every emptied split with its counts, immediately after `filter_data` and **before** the batch-1
  flip pass and the extraction; `run_model:840-850` catches it and returns the stopped record, so the extraction and
  the selection never run. `card.kept_stops` gives it its own reading ("the refusal tokens or the data, not the
  instrument or the selection"), `grades` turns Qwen's four grades into `inconclusive`/`tally: miss` and Ministral's
  into a C3-M miss, `expectations` marks the right side "not read", and `build_ledger` marks E-82/E-82r "not read".
  A test asserts the registered selection-filter sentence is *absent* from the whole card in that case. Tests:
  `test_c3_refusal.py:244` (both one-sided fixtures, the batch-1 pass never called, no prompt text in the message or
  the record), `:276` (the extraction and selection monkeypatched to raise), `test_c3_card.py:333` and `:354`.
  Planted fault p12 (the stop removed) is caught by four tests.
- **Main goes on to the judges after a Ministral empty-split stop** (`app.py:604-614`): the `set(stopped) ==
  {"ministral"}` branch prints and falls through to `judge_lab`/`judge_paper`; `tests/test_c3_app.py:483` asserts
  both judge calls happen, Qwen's three grades pass and C3-M is a tally miss. Planted fault p18 (the branch disabled)
  is caught.
- **C3-M is "not read" when Qwen stops** (`card.grades:297-304`): the Qwen-stop branch sets `c3_m` to
  `{"result": None, "why": "not read: the judge port gate runs on Qwen's Q0 and QA, …"}` unless Ministral also
  stopped, in which case it is a miss. `test_c3_card.py:333` asserts both the field and the headline line. Planted
  fault p17 (always a miss) is caught.
- **The template sha256 on the card, registered beside measured** (`card.facts_section:499-507`): the registered
  value now comes from `(plan.get("judge") or {}).get("template_sha256")` — the first check's null-field bug is
  fixed — with `judge_template_sha256_measured` carrying what each judge container measured (the port's copy in each,
  plus the installed package's own `JUDGE_TEMPLATES` in the paper stack, added in phase 2) and
  `judge_template_mismatch` listing any that differ, which `headline` prints. Planted fault p11 (the old key
  restored) is caught.
- **The judge dtypes** (`judge.load_judge:94-111`, `refusal.original_judge_facts:1101-1116`,
  `card.judge_section`): the port records the first parameter's dtype, the set over all parameters and whether the
  fallback branch loaded it; the original's dtype is read off `cached_models` (which ruling B's argument makes
  possible) and its branch is honestly "not observable"; the card marks each against the registered bf16 as held,
  did not hold or not observable. Tests cover both branches of the port's load and both the cached and uncached cases
  of the original.
- **v_ego's sha256 check** (`config.yaml:143-149`, `app.registered_ego`/`ego_check`, `refusal.ego_reading:973-990`):
  the image carries the one registered file at `anatta.modal_app.ego_file`'s path and never the folder; the laptop
  warns before any spend; the container hashes the file **before** loading it and marks E-82/E-82r "not read" with
  the reason if it is missing or different, without stopping. Planted fault p16 (the sha256 comparison disabled) is
  caught. A test also asserts no `add_local_dir` of a directions folder in either image.
- **The settings asserted against the code** (`app.code_settings`/`check_settings`, called first in `make_plan`):
  all seven the note lists are covered — `filter.threshold` (the four `filter_examples` calls),
  `filter.cut_harmless_train`, `selection.prune_layer_percentage` (five places), `mmlu.description`, `judge.scores`,
  `generation.strip` (both generators) and `smoke.candidates` — plus the two selection thresholds, read off the
  source with `ast` rather than imported. The tests are real: one parametrized over nine config edits, and one that
  copies the ported files to a temp tree, edits a literal in each, and asserts the plan is refused with the right
  message. Planted fault p14 (`check_settings` not called) is caught.
- **The 30-minute budget floor**: implemented and tested; see finding 4. Planted fault p13 (the floor argument
  dropped) is caught.
- **Wall-time costing** (`run_models:528-563`): each model is costed at `max(container wall_seconds, laptop
  spawn-to-result)`, each waited for in its own thread so the laptop figure is its own, and one model's error no
  longer stops the other's save. Planted fault p19 (container time only) is caught.
- **`fingerprint.py` in both images**: in `CHECK_CODE`, added to both images by `with_code`, with the `ast`-based test
  described in finding 3. Planted fault p15 is caught.
- The two tidy-ups are in: `swap` now does `del mb, hooks, model` before the cache clear, and the git-work-tree test
  carries `@pytest.mark.skipif(not (REPO / ".git").exists())`, which I confirmed by running the suite in an export
  (318 passed, 6 skipped; the first check's one spurious failure is gone).

---

## What I verified as right, beyond the ports

- **The suite.** `/Users/eugene/prj/align/anatta/.venv/bin/python -m pytest` in the worktree with
  `PYTHONDONTWRITEBYTECODE=1`: **319 passed, 5 skipped** in 31 s (the five skips are pre-existing and unrelated: two
  EM data downloads, three `peft` imports). The worktree is still clean (`git status --short` empty).
- **Both generation paths render the same prompt ids.** The conditions generate through `anatta.probe.generate`,
  which renders with `apply_chat_template(tokenize=False)` and re-tokenizes with `add_special_tokens=False`, while
  filtering, extraction and selection use `anatta.models.chat_ids`/`chat_batch` (`tokenize=True`). Nothing in the
  check compares the two on Ministral (`prepare`'s `template_check` returns early when `template` is null), so I
  compared them offline on both cached tokenizers at their registered revisions (`render_probe.py`): 36 ids on Qwen
  and 533 on Ministral, **identical** in both cases. So the registered rendering ("one user turn through
  `anatta.models.chat_ids`") is what the graded generations actually see, on both models.
- **The registered bands and counts** are self-consistent with the layer arithmetic: Qwen `band_b` 7-26 (mode B
  reads candidate row `b + 1`, i.e. Arditi's layers 8-27, inside 28), `band_b2` 14-21, Ministral 8-32 and 17-26; the
  prune caps are `int(0.8 × 28) = 22` and `int(0.8 × 34) = 27` as registered; C3-extract's 135 is 5 positions × 27
  layers, which is what `c3_extract` compares and what the card asserts.
- **Ruling-level behaviour of the judge gate.** `card.judge_gate` compares (a) against (b) and (b) against (c) on Q0
  and QA with the registered tolerances, reports the mean difference, the share over 0.1 and the largest single
  difference, and carries every per-completion score of all three legs in `card.json`; `main` stops with the
  registered instruction if (a) against (b) fails and re-scores every condition with the original package if (b)
  against (c) fails; `graded_scores` then names the source on the card. All of this is tested.
- **Safety.** The containers return numbers, counts, sha256s and token ids. `CheckError` and `EmptyKept` messages
  name splits, counts, files and hashes; the empty-split tests assert no prompt text appears in the message or the
  record, and the smoke-run test asserts the no-text property over every JSON a whole run writes. Completions and
  directions are written only under the run's folder on the volume; `--results` is refused inside any git work tree.
  Every fixture string is a made-up benign phrase (I read the fixtures' generators, not any dataset). Nothing in the
  check names anything beginning with `matlab`, and a test asserts it. No weight orthogonalization exists in the
  code. I opened no harmful-prompt dataset, and the one residual the first check named (a third-party exception
  formatted into `main`'s message) is unchanged; I agree it is not worth a change.
- **The fetched sources are what the 12:38 note says they are.** I recomputed every size and sha256 of all nine files
  in `c3_sources/`: all nine match the note exactly, and the two totals are 48,254 and 45,992 bytes as recorded.

## The planted faults

Planted one at a time in `fable_c3_fidelity/copy/` (a `git archive` of `a4e296b`; the worktree was never touched —
its `git status` is empty and the copy's five edited files compare byte-equal to the worktree's afterwards), each
followed by the whole suite (`--tb=no -rfE`), then reverted from a byte copy. Baseline in the copy: 318 passed,
6 skipped. "Caught" lists the failing tests; a fault only tests can't see is marked by what it would change.

| # | fault (file: edit) | what it tests | result |
| --- | --- | --- | --- |
| p1 | judge.py: the template's U+2019 made an ASCII apostrophe | the template byte for byte (phase 2) | **caught**, 5 tests: the sha256 gate, `config_registers_the_sha256`, `the_real_template_is_the_files_value`, the golden rendering, the package-template sha256 |
| p2 | judge.py: the trailing space after `### Answer:` dropped | the template's exact end (phase 2) | **caught**, the same 5 |
| p3 | judge.py: bare `except:` made `except Exception:` | the load's bare except (phase 2) | not caught (318 passed). Only a `BaseException` during the first load could tell them apart; no score can change |
| p4 | judge.py: `if not tokenizer.pad_token` made `is None` | the pad check (phase 2) | not caught (318 passed). The two differ only for an empty-string pad token, which Gemma's tokenizer does not have; no score can change |
| p5 | judge.py: the five-way softmax taken in float32 | the score dtypes (phase 2) | **caught**: `test_the_score_arithmetic_has_the_originals_dtypes` |
| p6 | judge.py: `linspace` in the probs' dtype | the score dtypes (phase 2) | void: the anchor line also appears in the docstring, so the plant was skipped (the test for p5 pins the all-bf16 variant too, by assertion) |
| p7 | judge.py: `torch.no_grad()` around the judge's forward dropped | ruling A's registered change | not caught (318 passed). The forward values are identical; what changes is memory and, possibly, the kernel (finding 2). No value test can see it, and none should |
| p8 | refusal.py: `empty_model_cache=False` made `True` | ruling B's condition | **caught**, 2 tests: `test_original_scores_reads_the_judge_before_releasing_it_as_the_package_would[False/True]` |
| p9 | refusal.py: `release_original_judge()` call dropped | ruling B's cleanup | **caught**, the same 2 |
| p10 | reference.py: one refusal substring's apostrophe made curly | the substrings byte for byte (phase 2) | **caught**: `test_the_refusal_substrings_are_the_sources_byte_for_byte_and_config_uses_them` |
| p11 | card.py: the pre-fix `plan.get("judge_template_sha256")` restored | the card's template field (f0faf4d) | **caught**: `test_the_card_shows_the_registered_template_sha256_beside_the_measured_ones` |
| p12 | refusal.py: the empty-split stop disabled | the empty-kept-set stop (f0faf4d) | **caught**, 4 tests: `filter_stage_stops_on_a_split…[harmful/harmless]`, `a_run_whose_filtering_kept_nothing_stops…[harmful/harmless]` |
| p13 | app.py: the floor argument to `Spend.budgets` dropped | the 30-minute floor (f0faf4d) | **caught**: `test_a_launch_that_would_cut_a_model_run_under_half_an_hour_is_refused` |
| p14 | app.py: `check_settings(cfg)` not called from `make_plan` | the settings asserted against the code (f0faf4d) | **caught**, 9 tests (every parametrization of `test_a_config_setting_the_code_does_not_use_refuses_the_plan`) |
| p15 | app.py: `fingerprint.py` removed from `CHECK_CODE` | fingerprint.py in both images (12:38 note) | **caught**, 2 tests: `test_every_file_app_py_loads_at_import_is_in_both_images`, the paper-image test |
| p16 | refusal.py: v_ego's sha256 comparison disabled | v_ego's sha256 check (f0faf4d) | **caught**: `test_v_ego_is_read_only_if_it_is_the_registered_file` |
| p17 | card.py: C3-M made a miss whenever Qwen stops | ruling D (C3-M "not read") | **caught**: `test_qwens_run_stopped_at_filtering_makes_card_c3_inconclusive_with_its_own_reading` |
| p18 | app.py: the Ministral-stop branch of `main` disabled | ruling D (main goes on to the judges) | **caught**: `test_when_ministral_stops_at_filtering_qwen_is_still_judged_and_graded` |
| p19 | app.py: a model run costed at container time only | wall-time costing (f0faf4d) | **caught**: `test_each_model_run_is_costed_at_the_larger_of_its_container_and_laptop_wall_times` |

The implementer's own eleven (`scratchpad/plant_faults.py`, results in `plant_all_a4e296b.txt`) are a different set
(the site and layer arithmetic, the hook hygiene, the MMLU read, the judge's two fields and the template's two
placeholders); I read their results rather than re-running them, and the failing-test names there are the right
ones for each fault. Note that their script plants in the worktree itself and reverts with `git checkout`; mine
never touches the worktree.

---

## What needs Eugene's decision

- **Nothing for ruling C.** Both points are settled from quoted source lines plus an offline tokenizer check, so the
  two fetches the ruling held in reserve are not needed.
- **Ruling A's attention-kernel question stays open.** Settling it would mean reading PyTorch's `sdp_utils`, which is
  outside this check's allowed fetches; the cheaper alternative is to have the paper image print what
  `torch.backends.cuda.can_use_flash_attention` / `can_use_efficient_attention` accept for Gemma-2b's shapes with and
  without `requires_grad`, as a recorded fact rather than an approval. Eugene's call whether that is worth a line of
  code; the gate and the registered fallback already bound the risk.

---
Claude (Fable 5.1), an AI model made by Anthropic, wrote this check.
