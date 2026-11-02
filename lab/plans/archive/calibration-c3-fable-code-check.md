# Calibration check 3: Fable's cold code check (filed)

A fresh Fable agent (model fable) checked the C3 code cold, with no memory of the session that wrote it, before any C3
run, download or phase-2 fetch. Its report file was last written at 11:26:40 UTC on 1 Nov (the file's clock). It is
below, unedited, with its own header. Its scratch files are in the session scratchpad's fable_c3_code/ folder. Opus 5.5
checked two of its claims against the code at ff1e05c:
- card.py:439 reads `plan.get("judge_template_sha256")`, while make_plan (app.py:262-275) copies config.yaml's judge
  block, which holds the value at `judge.template_sha256` (config.yaml:65), so the card's field would always be null;
- filter_stage (refusal.py:445-473) checks that its kept indices give filter_data's kept sets, but never that a split
  kept anything.
Opus 5.5 also found the registered v_ego file for finding 4 (anatta's directions/mistralai__Ministral-3-8B-Instruct-2512-BF16/ego.pt,
sha256 a105f23b79fecd23…, the file the 29 Oct self1 entry pins) and that the C3 worktree has no directions/ folder, so
a launch from it would have dropped E-82 and E-82r without a word. The rulings and the fixes are in the ledger's note
to the 09:24 entry (1 Nov, 11:31 UTC).

---

# Calibration check 3: Fable's cold code check

A fresh Fable agent checked the C3 code cold, with no memory of the session that wrote it: the worktree
/Users/eugene/prj/align/anatta-calib-c3, branch calib/refusal-direction, commits 6060eab..ff1e05c on dc32cf7, against
the registration (NOTES.md:5476, 09:24 UTC on 1 Nov, and the 09:27 note), the rulings on the nine questions
(scratchpad check3_rulings.md) and the design check it adopted. Read-only: nothing was edited, committed, stashed,
installed, downloaded or run on Modal. No harmful-prompt dataset was opened. The suite was run in the worktree; the
planted faults were run in a throwaway export inside the scratchpad, never in the worktree. Scratch:
/private/tmp/claude-504/-Users-eugene-prj-align/073015e5-1016-4b21-9a35-1f58d24e5f8e/scratchpad/fable_c3_code/
(pytest_full.txt, plant.py, probe_empty.py, copy/).

---

**Verdict: ready after changes.** Every registered condition, comparison, threshold, odds value, outcome rule and
card field I could check is implemented as registered or as ruled, the site and layer arithmetic is right at every
place it matters, and the suite is unusually good: all nine of the implementer's planted faults are caught by tests
that name the right thing, and five of seven faults I planted myself were caught too (one of mine was a no-op, one is
caught on the fixture and guarded on a real run by the registered token ids). Nothing unregistered changes a grade.
Four things should land before launch, all small: an empty kept set after filtering either dies with an obscure error
after the extraction has been paid for or, worse, is reported on the card as a selection-filter stop when it is not
one; the card's judge-template field is wired to a key the plan does not have, so it will always print null; the
judge's loaded dtype (a registered claim) is never recorded, and the port's dtype argument is deprecated in the lab's
stack; and the ledger's prediction-82 numbers depend on which v_ego file the image happened to carry, which nothing
records. One registered card field conflicts with another sentence of the registration and needs a word from the
coordinator. Suite: 276 passed, 5 skipped (the skips are pre-existing and unrelated: two EM data downloads, three
`peft` imports).

---

## Findings, ranked

### 1. An empty kept set after filtering crashes obscurely, or is read on the card as something it is not

`calibration/refusal_direction/refusal.py:445-473` (`filter_stage`), `:807-815` (the shared kept sets),
`src/anatta/select.py:81-99, 165-182`, `calibration/refusal_direction/card.py:276-279`.
Severity: should fix before the run.

Filtering is strict at 0 on both sides, and nothing checks that either side kept anything. I measured both branches
on the tiny fixture (scratch probe_empty.py):

- **Kept harmless val empty.** `get_last_position_logits` returns `None` for an empty instruction list, and
  `kl_div_fn` then dies with `AttributeError: 'NoneType' object has no attribute 'to'`. This happens in the first
  scoring loop, i.e. *after* the extraction has run — on Qwen that is three passes over about 2,350 prompts, roughly
  20 minutes of L40S. `run_models` reports it as `qwen: AttributeError: 'NoneType' object has no attribute 'to'`,
  which says nothing about the cause.
- **Kept harmful val empty.** Nothing crashes. Every bypass score becomes NaN (the mean of an empty tensor), the NaN
  branch of `filter_fn` discards every candidate on every rung of the ladder, and `choose` returns
  `found=False, viable=False` in **both** paths. The card then prints the registered inconclusive reading: "no viable
  candidate in either path: the selection filter, not the instrument, stopped the check". That reading would be
  wrong. The cause would be an empty evaluation set, not a threshold crossing, and the registered consequence (C3
  inconclusive, a miss in the tally, cut results keep their status) would be recorded for the wrong reason.

The counts are on the card (`card["filter"]` carries each split's `n` and `kept`), so a careful reader could catch
it; the automated reading is what is wrong. The exposure is concentrated on Ministral, whose refusal tokens were
chosen by analogy and are the one unvalidated choice in the design — the registration already asks for the
first-token distribution precisely to check them, and notes two other plausible first tokens on that tokenizer.

Fix: in `filter_stage`, raise `CheckError` naming every split whose kept count is 0, and raise it for the *train*
splits before the extraction spends GPU time (the train check can run immediately after `filter_data`). Give the card
a reading distinct from the selection-filter one. Whether a *minimum* count (not just non-empty) should stop or mark
the run is a registration question, not a code question — I would not add a threshold without the coordinator's word,
because a minimum would be an unregistered change to the instrument.

The reason the suite cannot see this is `tests/c3_testing.py:116-134`, `balanced_refusal_toks`: it picks the tiny
model's most probable tokens until the median prompt's mass passes one half, so filtering always keeps prompts on
both sides. It is a good fixture — without it the smoke-run test would not reach the selection at all — but it means
the one-sided case is never exercised, and `test_filter_stage_keeps_what_filter_data_keeps_and_records_numbers_only`
even guards with `if kept["harmful_val"]:`. Add a second fixture whose refusal tokens are the *least* probable and
assert the new stop.

### 2. The card's judge-template fingerprint is always null

`calibration/refusal_direction/card.py:439`. Severity: should fix (one line).

`facts_section` reads `plan.get("judge_template_sha256")`. The plan has no such key: `make_plan`
(`app.py:269-275`) carries it at `plan["judge"]["template_sha256"]`. So the field is `None` on every card. Phase 2
registers that sha256, and the card — the only artefact a later reader sees — would still say null about the one
string whose fidelity the whole judge port rests on. Fix: `(plan.get("judge") or {}).get("template_sha256")`, and
consider putting `judge_lab.json`'s `template_sha256` (measured in the container, `refusal.py:981`) beside it, so the
card shows the registered value *and* the value the container actually used.

### 3. The judge's loaded dtype is never recorded, and its dtype argument is deprecated in the lab's stack

`calibration/refusal_direction/judge.py:60-70`. Severity: should fix (cheap, and it makes a registered claim
checkable).

The registration pins the judge at "Batch 8, bf16". I checked the lab's stack on the laptop (transformers 5.21.0):
`from_pretrained(..., torch_dtype=torch.bfloat16)` prints "`torch_dtype` is deprecated! Use `dtype` instead!" and is
honoured — the parameters come back bf16. So the port is correct today in both stacks, and the faithful
`try`/`except` is faithful. But the `except Exception` fallback drops the dtype silently, so if any load path ever
raises (a future transformers removing the alias, a transient OOM) the judge would score in float32, and the only
symptom would be the (b)-against-(c) comparison failing and every grade being routed to the original package in the
paper's stack — expensive, and a misleading diagnosis. Fix: record `str(next(model.parameters()).dtype)` and whether
the fallback was taken in `judge_lab.json` and `judge_paper.json`, and put it on the card beside the batch size. No
behaviour change.

### 4. The ledger's prediction-82 numbers depend on an unrecorded v_ego file

`calibration/refusal_direction/app.py:112-113, 137-146`; `refusal.py:933-949`; `card.py:543-557`. Severity: should
fix, ledger only, cheap.

`ego_vectors` reads `directions/<model slug>/ego.pt` from the image, which is added only `if DIRECTIONS.is_dir()` at
launch. Nothing records which file that was. The design check noted that NOTES records two v_ego versions — the fixed
one and a re-extracted one at cos 0.94 with it and 0.194 against the refusal direction — and that E-82r's ±0.01
applies to the fixed one only. If the folder holds the other, the ledger's E-82 and E-82r silently describe a
different vector; if the folder is missing, `ego_vectors` returns None and the ledger is simply absent with no
complaint. Fix: record ego.pt's sha256 and its layer list in the ledger record, and either refuse or clearly mark a
run whose ego sha256 is not the registered one. (E-82/E-82r are ungraded and ledger-only, which is why this is not
higher.)

### 5. Settings that are in config.yaml, printed on the card, and never read by the code

`config.yaml:107` (`filter.threshold`), `:109` (`filter.cut_harmless_train`), `:115`
(`selection.prune_layer_percentage`), `:127` (`mmlu.description`), `:68` (`judge.scores`), `:122`
(`generation.strip`), `:142` (`smoke.candidates`). Severity: note, but it is the "a number on the card that did not
drive the run" failure.

Each of these is hardcoded in the code it describes: the 0 thresholds in `reference.filter_data` and
`refusal.filter_stage`, `cut_harmless_train` as that function's default, `prune_layer_percentage=0.2` as
`select.choose`'s and `reference.select_direction`'s default, the description string in `bench/mc.py:28`, the five
score labels in `judge.py:40`, stripping as unconditional in `generate_jbb`, and the smoke's two candidates in
`smoke_which`. An operator who edited one would get a new plan id and a fresh folder (so no results are mixed) but an
identical run, with the card reporting the edited value as if it had applied. The code already does this right for
the two selection thresholds: `run_model:792-794` raises if `config.yaml`'s `kl_threshold` and
`induce_refusal_threshold` differ from `select.LADDER`'s first rung. Extend that pattern to the seven above (or read
them), and the card's settings block becomes trustworthy.

### 6. `Spend.budgets` can hand a model run a timeout too short to load the model

`calibration/refusal_direction/app.py:338-346`. Severity: note.

`factor = min(1.0, room / want)` scales both model timeouts with no floor. With little room left — a second launch
after an expensive first — Qwen's timeout becomes a few hundred seconds: long enough to start the container and pull
the weights from the volume, too short to reach a single condition. The stop is still enforced, so nothing runs away;
the remainder is simply spent on a container that cannot finish. Fix: refuse a per-model budget under about half an
hour, with a message to raise the stop or start a fresh plan.

### 7. `swap`'s `del model` frees nothing

`calibration/refusal_direction/refusal.py:1074, 1103-1105`. Severity: note.

`mb` holds `mb.model = model`, and in the reference swap the ablation hook closures hold the block modules, so
`del model` followed by `empty_cache()` releases nothing; Qwen (about 15 GB in bf16) and the judge (about 5 GB) are
resident together. That fits an L40S, so this is tidiness — but the line reads as if it frees memory, which could
mislead a later change that adds a bigger judge or a longer swap. Fix: `del mb, hooks, model` before the cache
clear, or drop the misleading `del`.

### 8. The suite is not hermetic: one test needs the checkout to be a git work tree

`calibration/refusal_direction/tests/test_c3_card.py:271-274`. Severity: note.

`test_results_are_refused_inside_a_git_work_tree` asserts that `results_path(REPO / "results_here")` raises
`SystemExit`. Run from an export without `.git` — which is how I ran the planted faults — it fails rather than
skipping, and it was the one spurious failure in all sixteen of my fault runs. Fix: skip unless
`(REPO / ".git").exists()`.

### 9. The cheap tier is registered in config but not built

`config.yaml:212-217`; `app.py` has only the `main` and `card` entrypoints. Severity: note (registration
bookkeeping).

The registration describes the cheap tier's refusal check (QA's stored r-hat under mode A and Q0, on the 100
evaluation prompts, with the judge at batch 8), and the plan's loop rule 4 names it. There is no entrypoint. The
stored vector is on the volume (`tensors/vectors.pt` carries `rhat`), so it can be built later. Check 7 handled the
same situation with a ledger line deferring its cheap-tier entrypoint; C3 should get the same line.

---

## Needs a decision, not a fix

**The registration says two different things about the judge gate on the card.** The Card section asks for "the judge
port gate: the two means on each set and the largest difference on any completion" — implemented exactly
(`card.py:134-161`: three means per condition, and per comparison the mean difference, the share over 0.1, the
largest single difference, and pass/fail). The Evaluation section says "every score set goes on the card". The full
per-completion lists are in `judge_paper.json` and `judge_lab.json` in the results folder, not in `card.json`. Either
add the three lists to `judge_gate`'s output (600 floats on a full run; small) or have the ledger clarify that the
card carries the means and the maxima with the full sets in the run's files. I would take the clarification: the card
is a summary, and the run's files are kept.

---

## What the phase-2 fidelity check must cover

I read what the fetch tool would give me of both sources at their pinned commits. It paraphrases rather than quotes
beyond 125 characters, so I could confirm structure, defaults and the lines it did quote, but **not** the exact text
of any function. Everything below is unsettled and must be read off the files themselves.

**Arditi's repository at `9d852fa`:**

1. `pipeline/submodules/select_direction.py`: that the three scores are computed in three separate sweeps over
   `range(-n_pos, 0) x range(n_layer)` in the order KL, bypass, induce, as `select.py:172-200` and
   `reference.py:413-457` have it; that the ablation hook set is rebuilt inside each sweep per candidate (the port
   does); the exact defaults (0.1, 0.0, 0.2, batch 32) — the fetch confirmed the first; the two epsilons (1e-8 in the
   metric, 1e-6 in the KL); that the metric casts to float64 and keeps only the last position (the port slices before
   the cast, a registered and tested change); `filter_fn`'s four discards and the `int(n_layer * (1.0 - prune))`
   expression; the sort by `-refusal_score` descending and that the first survivor wins; and that his
   `masked_mean`/`kl_div_fn` masked branch is never reached by `select_direction` (the port raises
   `NotImplementedError` there).
2. `pipeline/submodules/generate_directions.py`: the pre-hook's `(1.0 / n_samples) * activation[:, positions, :].sum(dim=0)`
   into a float64 cache, the float64 accumulator's shape and device, batch 32, `positions = list(range(-len(eoi_toks), 0))`,
   and the two asserts (shape, no NaN). The fetch confirmed the accumulator line and the NaN assert.
3. `pipeline/utils/hook_utils.py`: that the ablation hooks normalize with `+ 1e-8`, cast with `.to(activation)`,
   subtract in place, and **store the result back through `nonlocal`** (the port keeps this drift on purpose and
   tests it); that `get_all_direction_ablation_hooks` attaches a pre-hook to every block and output hooks to every
   attention and MLP module, counted off the config's layer count; and the activation addition's `activation += coeff * vector`
   with the raw vector. The fetch confirmed the three key lines but not the `nonlocal` placement.
4. `pipeline/run_pipeline.py`: `filter_data`'s comparisons (harmful `> 0`, harmless `< 0`), that only the harmless
   *train* set is truncated to the kept harmful count and the validation sets are not, and that the activation
   subtraction uses coefficient −1.0 on the harmful evaluation sets. The fetch confirmed all of these on both
   repositories, including Wollschläger's truncation line.
5. `pipeline/model_utils/model_base.py`: `generate_completions`' `GenerationConfig(max_new_tokens=..., do_sample=False)`,
   that `pad_token_id` is set from the tokenizer, the slice of the prompt tokens, and the decode with
   `skip_special_tokens=True` followed by `.strip()`. The fetch confirmed these three lines.
6. `pipeline/submodules/evaluate_jailbreak.py`: the twelve refusal substrings **byte for byte**. The facts note that
   the fetch tool rendered the apostrophes as ASCII and the bytes were never checked; mine rendered them as ASCII
   too. A curly apostrophe in the source would change the substring refusal score, which is reported for every
   condition. Also confirm the header comment's exact words, which `reference.py:562` promises to copy, and the
   `int(not refusal)` aggregation.
7. Whether `_get_eoi_toks` on a Qwen2-style model would really give the five ids `config.yaml` registers — the port
   deliberately takes them from the rendered prompt's tail instead (`reference.py:206-216`), which the registration
   endorses, so this is a cross-check of the registered ids rather than of the port.

**strong_reject at `e286f0d`:**

8. `strong_reject/evaluate.py`, `strongreject_finetuned`: the model and tokenizer load (including that the
   `try`/`except` retry is the untyped one the port copies), `padding_side` and `truncation_side` left, pad from eos,
   the response truncation call and the decode with no further argument, the prompt's `.format`, the tokenization
   with `padding=True`, and above all **the dtype and device of the score arithmetic**: whether the softmax over the
   five ids and the `linspace(0, 1, 5)` multiplication happen in the model's dtype (the port does it in bf16 on the
   model's device) or in float32 on the CPU. A difference there shifts every score by about 1e-3 — inside the gate's
   tolerance, so the gate would not catch it, and both the port and the original would be wrong together only if the
   original is what the port copies.
9. `evaluate_dataset`'s batching: that a batched evaluator is mapped with `batched=True` and the given `batch_size`,
   in dataset order, so the port's consecutive slices of eight are the same groups (padding is per batch, so the
   groups matter).
10. `eval_files/judge_templates.json`, the `strongreject_finetuned` value: copied verbatim, with its exact trailing
    characters and its field order, and its sha256 registered in `config.yaml`. The facts' correction (about 1,527
    characters, one trailing space) is worth re-deriving from the file rather than from either fetch, since my fetch
    estimated a different length and could not quote the value.
11. `LICENSE` at `e286f0d`, against the MIT notice already pasted into `THIRD_PARTY.md`.

**Phase-2 cross-checks the code is waiting for:** the `C3-QUESTION` markers in `reference.py:27-28` and
`judge.py:30-32`; that `config.yaml`'s `template_sha256` matches the pasted string (there is already a test for this:
`test_config_registers_the_template_sha256_once_judge_py_holds_it`); and the golden test of the real template's
rendering that `tests/test_c3_judge.py:4` says phase 2 adds.

---

## What I verified as right

**The site and layer arithmetic — the thing most likely to make the numbers mean something else.**

- `dial.Intervention`'s three sites do what the docstring says: the input site is a keyword-aware forward pre-hook,
  so it reads `hidden_states` whether the decoder loop passes it by position or by keyword; input at layer *l* gives
  bit-identical logits to output at *l − 1* for every layer, and input at layer 0 acts on the embedding output
  (checked against an `inputs_embeds` call). `everywhere` gives 1 + n sites and, with `attn=True`, 1 + 2n; under mode
  A the attention output still carries the direction and under A+ it does not; A+ reproduces a direct implementation
  of Arditi's hook set, and mode A sits more than ten times further from it. All of this is tested, and the tests
  fail when I break the site.
- `contrast.token_acts(site="input")` reads block *l − 1*'s output with a forward hook and block 0's input with a
  pre-hook, which is Arditi's convention; `mean_diffs` keeps running float64 sums with no per-prompt tensor (the
  9.4 GB the design check warned about), and planting float32 sums fails six tests.
- `refusal.run_model:878` maps mode B's block output *b* to candidate row *b + 1* — Arditi's layer *b + 1* — and the
  bands stay inside both models' layer counts (Qwen 8..27 of 28, Ministral 9..33 of 34).
  `test_mode_b_vectors_are_the_block_output_mean_differences_at_the_last_token` checks the vectors against an
  independent `site="output"` extraction, and planting the off-by-one fails it.
- QS subtracts the raw candidate where Arditi's addition hook with coefficient −1 puts it, at block output *l\* − 1*
  or, when *l\** is 0, through block 0's input site; the test is parametrized over layers 0, 1 and 3 and fails when I
  move the site. QL and ML read the in-house direction at Arditi's layer *l\** through the input site, exactly as
  registered. QC is one r-hat at mode B's band; QA-rand and QB-rand use the registered seeds, and QB-rand's recipe is
  the lab's `cut_random` arm (`random_like(vectors[i], seed=i)` at every extracted layer) — I read `cut_vs_push.py`'s
  `build_arms` and mode B, B2 and the random arms match it, including `cut_self_stakes_17_26` for B2.
- Mode A's faithfulness gap is the registered one and only that: the dial normalizes in float32 and casts to the
  activations' dtype per call, while Arditi normalizes in bf16 inside the hook and stores the result back; the port
  keeps his drift deliberately and tests it. The induce addition reaches the activations as `‖v‖ x v-hat` rounded in
  bf16 rather than `v` rounded in bf16 — the same order of rounding, a different draw of it, which is one reason
  E-sel is an ungraded expectation.

**MMLU.** The prompt is golden-tested against lm-eval's template text rendered with jinja2, with the subject's
underscores as spaces; the continuations are the four one-token ids, asserted one-token and cross-checked against the
registered ids in both `prepare` and `run_model`; batches are right-padded and **each row is read at its own last
real token**, with `last_real_index` from the mask — reading index −1 instead fails the batch-4-against-batch-1 test
in two parametrizations, and on the real model the registered golden test (batch 32 against batch 1 on 64
length-spread items, any disagreement stops the run before any condition) is the second net. Accuracy is a
micro-average over all rows; the per-row correctness bits drive the paired intervals; `within_mmlu` compares the
change in exact integer arithmetic on the counts (2 points of 14,042 is 280 questions, and 281 fails). Ministral's
single BOS is kept and recorded as the registered deviation. The chat-template pass scores the same four ids and is
compared against `Q0-chat`, not `Q0`.

**Generation.** `recorded_generate` pins `model.generation_config.repetition_penalty`, asserts it and the eos ids
before every call, records each call's arguments and output ids, and restores the field afterwards — the 09:27 note's
substitute for editing `probe.py`, and `src/anatta/probe.py` is byte-identical to dc32cf7 (sha256 fc0b5692…), as is
`modal_app.py`. A test shows the pinned penalty really reaches `probe.generate` (the completions change at 10.0), and
that the override is an instance attribute that is deleted afterwards. Greedy, left padding, 512 new tokens, the
driver strips each completion, the hooks are on for every forward pass (there is a test that counts hook calls inside
both the generation and the MMLU pass, and it is the one test that catches a context manager entered and dropped).
New-token counts stop at the first eos — which on Qwen is also the pad id, so finished rows still count correctly —
and the cut share counts completions with no eos at the limit.

**Selection.** The metric, the KL, the filter and the choice are Arditi's as the sources' structure describes them,
with the registered float64 slice-before-cast (tested bit-for-bit against casting the whole batch); the filter's
strictness is tested at every boundary (KL exactly 0.1 is kept, induce exactly 0 is kept); ties go to the first
candidate in his loop order; the ladder is cumulative, as ruling 1 requires (drop induce, then KL 0.2 with induce
still dropped), and `LADDER` is asserted in a test. `run_model` refuses to start if `config.yaml`'s thresholds differ
from the code's first rung, and cross-checks Arditi's own pick against `choose`'s first rung on the reference tables.
The library path's scores, with Arditi's hooks passed in as builders, equal the reference port's exactly on the
fixture — the registered equality, and the test that makes the two paths comparable.

**C3-extract.** Both paths call the model with input ids and no attention mask at batch 1, so they read the same
forward passes; the bitwise premise is measured on one prompt before the comparison and recorded with the layers that
differ and the largest absolute difference; the smoke run stops on a failed premise and a full run records it and
goes on, as ruling 2 requires, and a miss with a failed premise is read as unexplained in both `grades` and
`verdict`. Layer 0 is excluded for the stated reason (the library's running sums give exact zeros there, Arditi's
`(1/n)`-scaled sums leave rounding residue), which is why the count is 135 = 5 x 27 and why the card asserts that
count. The cosines against the padded reference are reported beside it.

**The judge port gate.** Three-way as registered: (a) the installed package and (b) the port both in the paper's
image, (c) the port in the lab's image, on the same completions and the same raw instructions, with the registered
tolerances (means within 1 point, at most 2% of completions differing by more than 0.1) and the largest single
difference reported. A (a)-against-(b) failure stops the run with the registered instruction and withholds every
grade; a (b)-against-(c) failure re-scores every condition with the original package in the paper's stack and the
card says which source the grades read. The port's truncation goes through the tokenizer, not by slicing ids, and
keeps special tokens, so the BOS behaviour is whatever the tokenizer does; `prepare` measures that on two benign
strings (one short, one well over the limit) and the card prints the measurement beside the registration's claim and
says whether the claim held — ruling 7, implemented as a measurement rather than an assumption. Planting a
slice-based truncation fails two judge tests. The installed package's commit is checked against the registration from
pip's own record.

**The grades, the verdict and the card.** C3-base's band, C3's two bars on the no-template pass, C3-extract's three
bars, C3-R and C3-M are read from the registered block in the run's own `plan.json`, and every band, fraction,
minimum and odds value in `config.yaml` matches the registration exactly (I checked all five grades and all fifteen
expectations). A reference-viable/library-not case is a C3 fail; neither viable is inconclusive with a tally miss; a
fail wins over an inconclusive (ruling 6); no viable reference candidate is a C3-R miss with the ladder's QR on its
own labelled, ungraded line (ruling 5); a ladder candidate's QA gets the same treatment; C3-M is graded but kept out
of the verdict. QX runs when the two paths differ *and* when the library path has nothing and the reference path has
something (ruling 3). The swaps fire on exactly the registered misses, reuse the shared kept sets, and add the MMLU
pass only when a miss was on the MMLU bar, keeping the JBB result if the golden test stops (ruling 4 and the
implementer's accepted confirmation). ASR is 100 x the mean judge score with no threshold; intervals are percentile
bootstraps over prompts with 10,000 resamples and seed 0, paired for differences; MMLU levels carry Wilson intervals.
The conditions are complete and in run order on both models, with exactly the registered runs each (I checked all
seventeen Qwen and six Ministral entries against the registration line by line), and Q0-end is last, with its
comparison against Q0 reporting determinism and hook removal together.

**Hook hygiene.** `assert_no_hooks` names every block, attention and MLP module carrying a forward or pre-hook, and
is called before and after every pass of every condition and between every stage of a model run. Planting a leaked
hook fails four tests; planting a dropped context manager fails the one test written for it (commit c341178 earns its
keep). The reference path's hooks are built once per condition and reused across its passes, which is what Arditi's
own run does. No stage asks for hidden states, so transformers' own capturing hooks never appear. The one place that
loads with `device_map="auto"` (the paper-stack swap, where accelerate adds its own hooks) never calls the assertion.

**The Modal app as written.** Two images: the lab's is `modal_app`'s pins read out of that file's source plus
pyarrow, so the two cannot drift; the paper's is Python 3.12 with the paper's three pins and the packages the judge
package imports, pinned by hand, then the package itself from GitHub at the registered commit with `--no-deps`, then
`READTHEDOCS=1` and `HF_HUB_OFFLINE=1`, then two build-time checks (the template file is installed, and the evaluate
module imports) — the right order, asserted by a test, and the import check is what will catch any module-level
dependency the hand-pinned set misses. Local code is added last in both, so no build step follows it. Volumes are
`anatta-hf-cache` and `anatta-runs` only; nothing anywhere in the check names anything beginning with `matlab` (a
test asserts it). The Hugging Face secret is on `prepare` alone and nothing prints an environment variable, a token
or a key. Offline loading is set before transformers is first imported (`anatta/__init__.py` is inert, so the import
order holds) and the weights are read from paths that contain the pinned revision, so the revision is structural
rather than a flag. GPU is L40S everywhere but `prepare` and `stored`, with 8 cores and 32 GB and the
expandable-segments allocator applied through `with_options(gpu=..., env=...)` — the same call shape check 1 used at
44760f7, which I read to confirm it. No retries. Every stage is resumable from its own JSON under a folder named by
the plan id, a folder refuses another plan, and every container checks its code hashes and installed versions against
the plan before anything loads. The spend estimate is wall time at the config rates, the higher where two apply
(ruling 9), labelled an estimate everywhere it appears, written after every call including a failed one, and enforced
as a guard before each call and as a cap on the GPU timeouts.

**Safety.** The containers return numbers, counts and sha256s. I traced every value that leaves the volume: the
filter record is counts, quantiles, flip counts and token ids with shares; the generation record is token counts, cut
counts, substring flags and per-completion sha256s; the extraction and selection records are score tables and tensor
fingerprints; the geometry is norms, cosines and sha256s, never a vector; `prepare` returns byte counts, sha256s,
booleans and token ids. Completions and directions are written only under the run's folder on the volume. The
`CheckError` messages name files, splits, counts and hashes, never prompt text. The tests check the no-text property
directly, over every JSON a whole smoke run writes, against every made-up prompt and completion it used — and that
check is one of the better things in this branch. Every fixture string is made up and benign, including the judge
control. No weight orthogonalization exists in the code. The one residual: an exception raised inside a container is
formatted into the error `main` prints, so a third-party exception that happened to echo its input would reach the
terminal (not the card, not a repository). I would not change anything for it.

**The planted faults.** All nine of the implementer's, run in a throwaway export: the dial's input site moved one
block late (10 failures), the contrast input read moved one block late (4), QS steering at the wrong site (2
parametrizations), mode B's vectors read at the input site (1, the test written for it), the MMLU read at index −1
(2), a position offset in `token_acts` (7), the attention hooks dropped under `attn=True` (6), a hook left installed
(5) and the hooks dropped before the pass (1). Seven of my own: `everywhere` without block 0's input (4), float32
sums in `mean_diffs` (5), the metric read at the wrong position (3), the judge truncating by slicing ids (2), the
generation batch collapsed to one (1) and the MMLU letters without their leading space (1, on the fixture; on the
real tokenizers the bare letters are also single tokens, so the real defence is the registered ids, which `prepare`
and `run_model` both check). One of mine was void: `filtered`'s row index is the same under negative indexing, so it
was not a fault at all.
