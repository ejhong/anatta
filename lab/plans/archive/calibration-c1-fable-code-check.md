# Calibration check 1: Fable's cold code check

A fresh Fable agent (model fable) checked the C1 implementation cold, with no memory of the session: branch calib/model-check, commit 7df35dd, in the anatta-calib-c1 worktree. It checked it against the registration (NOTES, 01:57 UTC on 31 Oct), the amendment (02:08 UTC) and the design check (calibration-c1-fable-design-check.md). It reported at about 03:34 UTC on 31 Oct, before any C1 run. Its report is below, unedited. Its scratch files are in the session scratchpad's fable_c1_code/ folder. Opus 5.5 filed this and wrote the clarifications that follow from it (NOTES, 03:37 UTC on 31 Oct).

---

**Verdict: ready with changes.** Nothing I found would crash a correctly configured run or score a reply wrongly. The library path reproduces the original battery byte for byte (I re-derived the golden fixture from the lab's `modal_battery.py` source itself, not from the test), the prompt and batch assertions pass on the real cached Qwen and Ministral tokenizers under transformers 5.21.0 (and the batch check does catch a second BOS on Ministral), and the parse is verbatim against the installed inspect_ai 0.3.285 (that test ran; it was not among the 5 skips). The changes below are about stale reuse, two grading/gating decisions that should be fixed before the run rather than after, and an unbounded cost.

Files: `/Users/eugene/prj/align/anatta-calib-c1/calibration/model_check/{app.py,card.py,config.yaml,README.md}`, `/Users/eugene/prj/align/anatta-calib-c1/src/anatta/bench/{run.py,items.py,inspect_choice.py}`. Scratch checks: `/private/tmp/claude-504/-Users-eugene-prj-align/073015e5-1016-4b21-9a35-1f58d24e5f8e/scratchpad/fable_c1_code/{real_tok_checks.py,golden_reproduce.py,draw_image_imports.py}`.

## Changes, ranked

1. **Before launch. The plan does not fingerprint the code, so a rerun after a code change resumes stale results.** `plan_id` (app.py:72-79) hashes model, revision, condition, system, jobs, batches, token limits, checks, GPU and the item sha, but not `code_hashes(PKG)` or the image pins. Scenario: C1b fails, someone edits probe.py or run.py (padding, parser), reruns `main` with the same `--results`: `local_complete` (app.py:252-260) sees an equal plan.json and prints "on disk", and the card regrades the old rows; with a fresh `--results`, the container's `stored_run` (app.py:144-147; run.py:388-404) finds the old run under the same plan_id on the volume and returns it without loading the model. `hashes_section` would show `code_here_matches_run: false`, but the verdict comes from the stale rows. Fix: put `code_hashes(src/anatta)` from the laptop into the plan in `make_plan` (app.py:233-240) and have `evaluate` assert the container's `code_hashes(PKG)` equals it; the run_info's torch/transformers versions could be asserted the same way.

2. **Before launch, a decision. Ministral runs after a one-set fail, while the plan says a miss "stops everything".** `main` gates Ministral on `load_path_calibrated` (app.py:327-330), which `overall` sets True in the one-passes-one-fails case (card.py:180-185) even though the same verdict sets `stops_remaining_checks: True`. The registration (NOTES 3719, 3774-3775) is ambiguous: the passing set calibrates the load path, but a fail stops the remaining checks until explained. The code spends the $8-16 Ministral stage before the fail is explained. Either gate on `not stops_remaining_checks` (one line; `--force-ministral` still exists) or write down that Ministral runs on a calibrated load path regardless. Claim (7) says "only if Qwen calibrates"; in fact it also runs when C1a or C1b fails.

3. **Before launch, a decision. The format-explanation rule takes the most lenient reading.** `grade_c1b` (card.py:159-166) counts a miss as explained when `100 * lost >= lo * n - 100 * k` (card.py:161): the replies with no ANSWER line or cut at the cap, counted as if every one of them would have been right. The amendment says those replies must "account for the gap". A reader will ask for the expected-rate version (lost share × the band centre or the TIGER score). Decide the operationalization now and record both numbers on the card; after the run, changing it looks like tuning. Also `no_answer_line_or_cut` (card.py:135) counts a cut reply even when it still parsed, a small double count.

4. **Before launch. No cost bound stands in for the registered $40 stop.** `evaluate` has `timeout=8*3600` and one retry (app.py:119-120): one plan can bill 16 L40S-hours ($42) before Modal gives up, and four plans (Qwen, two Ministral, the empty arm) could run in the same session. Expected time is 1-1.5 h per plan. Set the timeout to about 4 h; the stop stays a manual rule, and the README should say so.

5. **Can wait. ARC is unpinned and the draw overwrites the volume copy before the hash check.** `sample` resolves `main` to a commit (app.py:98) and records it, which is right, but it writes `/runs/h4lite/model_check/items.json` unconditionally (app.py:112-115) while `draw` refuses to overwrite the local file on a sha mismatch (app.py:302-303). A later `draw` after an upstream ARC change would replace the volume's registered copy with an unregistered one. After the first draw, write the resolved ARC commit into config.yaml next to the sha, and have `sample` leave an existing volume file alone.

6. **Can wait. A truncated partial line blocks every resume.** `read_jsonl` (run.py:309-310) does `json.loads` on each line; a container killed mid-append leaves a line that raises JSONDecodeError on each retry, with no way past it but manual cleanup on the volume. Skip or refuse a malformed trailing line explicitly.

7. **Can wait. Benign nondeterminism has no path.** The first-batch check (run.py:368-370) is faithful to the amendment: fewer than 95% identical of 50 stops the run, after the model load, on every retry, and there is no force flag. Sampling at T=0.7 gives near 0% identical, while kernel noise gives a share well above 50%. If it trips in that middle range the registration leaves nothing to do; decide now whether that case is "log it on the card and continue".

8. **Can wait. Small card gaps.** A job fully resumed from partial batches has `generate_kwargs: None` (run.py:255-256), so `run_section` reports `do_sample: ["None"]` (card.py:333). The card lacks the registered odds (85%, 70%). It records the snapshot folder but not what `refs/main` in the cache resolves to, which is how every lab run loads the model by id (NOTES 3685); one line in run_info closes that. `--battery-results` is optional, so the registered rerun comparison can be skipped with a printed warning (app.py:316-319); require it when the Ministral stage runs.

9. **Can wait. Process.** `card.json` is written to `--results`, which must lie outside every repository (card.py:68-80), while the plan says `calibration/model_check/card.json` is what the site reads; the README never says to copy it in. The refusal is also stricter than lab practice (the battery's 750 items live in the lab repo). `modal run --detach` would survive a laptop sleep; the README says to keep the terminal open.

## The registration, item by item

- **Items and draws.** All 1,319 GSM8K rows; 1,000 MMLU-Pro by largest remainder, seed 0 within each category over parquet row order (items.py:43-84); the registered allocation is asserted at draw time with the no-tie check (app.py:176-183, config.yaml:51-54). Files under the prefix sorted by name as the battery reads them (items.py:128-131). Revisions cc7b047 and 527feea pinned (config.yaml:32-33); ARC as above. The item file holds id, category, source id (MMLU-Pro `question_id`; inspect_evals' md5 id for GSM8K, matching `create_stable_id` for one field), gold, every prompt, the parquet sha256s and the resolved revisions. Cheap tier 150 and batch-1 30 are stratified the same way and sit inside the 1,000 (tested on the real category sizes, test_bench_items.py:66-75). Battery draws are the battery's seed-0 250 and must equal the stored run's `items.json` (app.py:201-210); that file exists with 750 items and base.json has 234/89/221. Faithful, and stricter than the spec on the battery items.
- **Prompts and limits.** GSM8K and letter formats are `anatta.bench.battery`'s, identical to `modal_battery.py`; 512/24/24 and 2,048 (config.yaml:56). The CoT prompt matches Inspect's `prompt()` rendering of inspect_evals v0.24.0's template, and inspect_evals passes `record["options"]` unfiltered, as the code does. Qwen gets no system message, so its template writes its 16-token default line (verified in the render). Faithful.
- **Parsers.** GSM8K: the battery's. C1b: `parse_answers` and `answer_character` are source-identical to inspect_ai 0.3.285; all five golden strings are tested (test_bench_parse.py:14-31), plus the lenient fallback. TIGER-Lab's two patterns fill in only where Inspect finds nothing (run.py:86-92); a pure-TIGER score is not reported, which the failure path might want. Faithful.
- **Decoding.** `probe.generate` with temperature 0.0 (do_sample=False, temperature and top_p None, pad id), the model handed in through a Recorder that forwards every attribute; Qwen's repetition_penalty 1.05 stays and is read from `model.generation_config` at run time (run.py:428-449). Batches sorted by prompt length, stably, as the battery. Faithful.
- **Batch sizes.** 50; Ministral's CoT jobs 25 (config.yaml:15, keyed by format). On the card. Faithful.
- **Assertions.** (a) `check_prompt` (run.py:137-150) compares `tok(rendered, add_special_tokens=False)` with `apply_chat_template(tokenize=True)`; in transformers 5.21.0 the latter tokenizes the same rendered string with `add_special_tokens=False`, so this equality cannot fail. The assertion that can fail is `check_batch` (run.py:153-167): the rows handed to `model.generate` must be left-padded with the pad id and equal the template ids. On the real Ministral tokenizer, `add_special_tokens=True` makes it fail at position 1 (566 against 565 ids); on Qwen it is a no-op, as registered. The BOS count is checked on the template ids (one on Ministral, none on Qwen, where it is vacuous). The README's wording of check 1 overstates it; say the batch check is the one with teeth. (b) The first c1a batch twice, ≥95% identical, every run (run.py:268-277, 355-370); it fails if sampling is on. Faithful.
- **Batch 1 vs N.** 30 + 30 items in each model's default run, identical share and accuracy gap reported (run.py:280-291). Faithful; roughly $1-2 on Ministral's CoT items at batch 1.
- **Bands, odds, verdict.** Exact integer comparison (card.py:104-107); [86, 96] and [51, 61]; the verdict table (card.py:169-188) matches the amendment's both-pass, one-pass-one-fail and explained cases; the explanation rule as in change 3; above-band is a fail. Faithful apart from change 3.
- **Ministral rerun and contrast.** Within 2 points per set as |Δk|·100 ≤ 2·n (card.py:260), identical 800-character prefixes reported; default minus helpful via the battery's paired bootstrap, third arm when |gap| > 3 on c1a or c1b (card.py:293-301, app.py:342-349), the 70% expectation recorded with a strict `<`. The empty system message renders `[SYSTEM_PROMPT][/SYSTEM_PROMPT]` with no Le Chat text on the real template (verified). Faithful.
- **Cheap tier.** Stratified 150 per set in their own batches, tolerance 5 on the card. The later comparison code does not exist yet, which is fine for C1. Faithful.
- **Cost and stop.** README $7-17 against the registered $12-24; no stop in code (change 4). Silent.
- **Card.** Anchors, Wilson intervals, pass/fail, hashes of models.py, probe.py, battery.py, items.py, inspect_choice.py, run.py, modal_app.py as hashed in the container and on the laptop, the item file (here, registered, as used by each run), parquet sha256s and revisions, model revision and snapshot, max_new_tokens, repetition_penalty, do_sample, dtype, torch, transformers, GPU, batch sizes, item 0's rendered prompt and first 40 ids, hooks out of scope. Faithful apart from change 8.

## Tests

The 82 tests in the four new files pass with no skips. They catch what matters on the stand-ins: the golden equivalence (fixture provenance checks out: its `source_sha256` is the lab's modal_battery.py and I regenerated items, probe calls, generate calls and rows from that script's own source), double BOS from template or tokenizer, right padding, wrong pad id, sampling on, resume from a partial file, another plan's files, the item-file hash gate, battery-item mismatch, allocation mismatch, the verdict table, the C1b grades, the rerun tolerance, the contrast trigger and third arm, the draw's file order and revisions, and the draw image importing without torch (I simulated the draw image's blocklist; the import chain is stdlib only).

Untested: stale reuse after a code change (change 1, since it is not implemented); the real chat templates (add an opt-in test on the cached snapshots: `check_prompt` on Qwen and on Ministral under None, "You are a helpful assistant." and "", asserting one BOS and no default text in the empty arm; my offline run of exactly that passed); Ministral gating in the one-fail case (only fail/fail is tested, test_model_check.py:492-517); a truncated partial line; the timeout.

## The eleven claims

1. True: Modal 1.6.2's `_assert_no_mount_layers` (modal/_image.py:587-601) refuses a build step after `add_local_*` unless `copy=True`; a separate draw image is one of two fixes.
2. True; loading from the snapshot folder works offline on the laptop with the same transformers.
3. True as stated, but the plan has no code or image fingerprint (change 1).
4. True (config.yaml:84; `load_items` refuses null or a mismatch).
5. True, with the volume overwrite (change 5).
6. Implemented as the lenient reading (change 3).
7. Imprecise: the gate is `load_path_calibrated`, which also holds when one of C1a/C1b fails (change 2).
8. True (card.py:68-80, `.git` as file or folder).
9. Wrong premise: no THIRD_PARTY.md exists in this repo, nor in anatta or anatta-dev; only LICENSE. The MIT notice in inspect_choice.py's docstring satisfies the license; a notices file would have to be created, not updated.
10. True, and recorded at run time.
11. True (`max_retries=1`); note a CheckError is retried once too, costing one extra model load.

## What I ran (read-only, no Modal, no downloads)

- The four C1 test files: 82 passed, 0 skipped (the verbatim and prompt tests against inspect_ai 0.3.285 ran).
- `real_tok_checks.py`: `check_prompt`/`check_batch` on the cached Qwen (a09a354) and Ministral (f6fae97) tokenizers, offline, transformers 5.21.0: all three Ministral conditions pass with one BOS (565 / 50 / 44 ids), the empty arm renders empty system tags without the Le Chat default, Qwen passes with 0 BOS and its default line, and `add_special_tokens=True` is caught on Ministral only.
- `golden_reproduce.py`: the original `modal_battery.py`'s `sample()` and `evaluate()` loop, exec'd from its own source with stand-ins, reproduces every item, probe call, generate call and row of the golden fixture; the fixture's source and probe sha256s match the lab file and src/anatta/probe.py.
- `draw_image_imports.py`: app.py and `anatta.bench.items` import under a blocklist of torch, transformers, yaml, numpy and the other pins absent from the draw image.
- Confirmed: existing library files (models.py, probe.py, battery.py, modal_app.py) are unchanged in the commit; `attach_adapters(None)` returns the plain model; the stored battery `items.json`/`base.json` have the shape `check_battery_items` and `battery_rerun` expect.

---

## The re-check, after the changes

The same Fable agent re-checked the branch after the implementer's changes: rebased onto calib/instruments (dc32cf7), commits ab7e690, 23df6f4, 1c83064, f33fd4a, 8d93ad7 and 8b50b2e. It reported at about 04:13 UTC on 31 Oct, before any C1 run. Its report is below, unedited. Opus 5.5 sent items 1-4 to the implementer before the first draw.

**Re-check verdict: ready. Nothing stands between this and the first `draw`.** All nine changes are addressed as the NOTES entry says; the seven departures are sound; the new code introduces no bug I could trip, and the suite (246 passed, 5 skipped, none of them C1 tests; the opt-in tokenizer tests ran on the cached snapshots and passed) exercises the new paths. Two small items are worth a line before the first `draw` because they decide whether a registration has to be redone; the rest can wait.

Files: `/Users/eugene/prj/align/anatta-calib-c1/calibration/model_check/{app.py,card.py,config.yaml,README.md}`, `/Users/eugene/prj/align/anatta-calib-c1/src/anatta/bench/{run.py,fingerprint.py}`, `/Users/eugene/prj/align/anatta-calib-c1/tests/test_model_check_tokenizers.py`. Scratch: `/private/tmp/claude-504/-Users-eugene-prj-align/073015e5-1016-4b21-9a35-1f58d24e5f8e/scratchpad/fable_c1_code/{pins_check.py,laptop_plan.py,draw_image_imports.py}`.

## Ranked items

1. **Before the first draw (one call). `draw` accepts `--battery-results` but never compares the draw with the stored battery items** (app.py:286-317 uses it only for `results_path`; the check runs first in `main`, app.py:331-332). If allenai/ai2_arc's parquet changed since the battery's 28 Oct run, the first `main` refuses the battery draw after the item file's sha256 and ARC commit are already registered and committed, and the registration has to be redone with ARC pinned to the battery's commit. Call `check_battery_items` in `draw` when `--battery-results` is given, before printing the sha. I could not pre-check the drift offline (the laptop cache has no datasets).

2. **Before the first draw (a decision, no code). `app.py`'s own sha256 is in the plan** (app.py:90-92, 235-236). Any edit to the laptop-side orchestration between stages, even a message string, makes every finished stage "another plan": Qwen reruns ($4-8), Ministral ($8-16). Only `evaluate` (app.py:145-156, three lines that forward to `run_plan`) runs in the container. Either accept the cost knowingly, or hash `inspect.getsource(evaluate)` plus `out_dir` instead of the file. Not a bug.

3. **Can wait. The volume's item-file copy can lock `draw` out.** `sample` raises before downloading when a sha256 is registered and the volume copy differs (app.py:116-118), and writes the copy only when none exists and nothing is registered (app.py:134-140). Sequence: draw, redraw before registering (ARC pin changed), register the second sha; every later `draw` now stops on the first draw's stale copy until `modal volume rm` removes it, and the copy is never restored when the volume is wiped. `main` never reads the copy, so nothing else is affected. Writing the copy whenever the new draw's sha equals the registered one would close both.

4. **Can wait. The gate reads Qwen rows without checking they belong to the current plan.** `main --stages ministral` on a results folder holding Qwen rows from older code gates Ministral on those rows (app.py:339-340), since `launch` and its `local_complete` check are skipped for Qwen. The README's "fresh --results after an edit" avoids it, and the card's `code_here_matches_run` would show it afterwards. Running `local_complete` for Qwen's plan before gating closes it.

5. **Can wait. `gate.forced` is inferred** (card.py:292-298): Ministral rows present while Qwen's verdict stops or is missing. Right for the flows the app allows; a Qwen job file deleted by hand afterwards would mislabel a legitimate run as forced.

## Changes 1-9

1. Stale reuse: addressed. The plan carries the sha256 of 9 package files, the 13 image pins and app.py (fingerprint.py:21-28, 31-47; app.py:226-236); the container checks files and installed versions before anything loads or is read back (run.py:472-487, 517-530); the laptop refuses a results folder from another plan (app.py:248-256). Tested end to end, including a container refusing a complete run on the volume (test_model_check.py:638-671) and a `+cu128` label (test_bench_run.py:415-427). Verified here: `image_pins` reads exactly the 13 pins, `installed()` resolves PyYAML, hf_xet, huggingface_hub and jinja2 under importlib.metadata's normalization in a uv-managed Python 3.12 venv, and a plan built in the Modal CLI venv is stable across calls (`laptop_plan.py`). No false refusal expected in the container.
2. Gate: addressed (`stops_remaining_checks`, app.py:339-347; override recorded, card.py:292-298); tested for pass/fail, fail/fail and the override.
3. Format rule: kept as registered with lost = no Inspect letter (card.py:143-155, 180). The claim that a fallback score in the band implies the cover is right (the fallback reads only lost replies, so kt − k ≤ lost) and is tested by construction and on 3,000 random rows; the double count is gone; the expected-rate figure is on the card, ungraded.
4. Cost: 4 h timeout (app.py:74, 145), worst case stated in the README, the $40 stop manual. Addressed.
5. ARC: the draw prints the resolved commit and what to put in config.yaml (app.py:311-317); the item file no longer stores the requested revision (app.py:125), so pinning afterwards keeps the sha. Addressed, with item 3's residue.
6. Malformed last line: skipped with a warning and removed from the partial file; any other malformed line raises (run.py:315-331); tested, including a resume past a half-written line.
7. First-batch check: unchanged by decision; README says the differing replies are read first.
8. Card: odds, refs/main, remembered generate_kwargs (run.py:335-347, 517-555), installed versions, container starts, item 0's sha256, `--battery-results` required for the Ministral stage (app.py:326-328). Addressed.
9. Process: README covers the card copy, the public form, no `--detach`, and not editing during a run. Addressed.

## The departures

(1) All 13 pins checked: sound, as verified above. (2) The check inside `run_plan`: sound; run.py and fingerprint.py are themselves fingerprinted. (3) Scope: sound. I read every imported-only module's top level (koans/env.py, base.py, commons, impossible, pushback, shutdown, bench/metrics.py, runner.py): constant assignments and one `exec` of a constant string; nothing touches generation or scoring. card.py's exclusion is right (grading is recomputable; its sha is on the card). (4) Inferred "forced": sound, with item 5's edge. (5) Requested revision out of the item file: sound; the first draw's bytes survive the pin. (6) No `--detach`: sound; the laptop drives the stages and resume is the recovery path. (7) Explicit `--results`: fine.

## The first draw

The draw image's import chain is still standard-library only (re-simulated on the new app.py with torch, yaml and numpy blocked); the datasets are public, so no token is needed; `items.sha256` is null and ARC is `main`, so `sample` takes the first-draw path and writes the volume copy. Two things can only be learned by running it, and both fail loudly: the category counts of MMLU-Pro at 527feea must give the registered allocation (`check_allocation`), and the battery items must match (item 1 moves that check to `draw`). `modal run` in the Modal CLI venv still needs `PYTHONPATH=src`; without it the import fails rather than picking another package. The new `modal_app` warning about a missing `directions/` folder on the laptop is harmless here.

---

## Confirmation of the last fixes

The implementer addressed the re-check's items 1-4 in one commit, c691d74. The same Fable agent checked it and reported at about 04:31 UTC on 31 Oct, before any C1 run. Its report is below, unedited.

**c691d74 is sound. Nothing stands between it and the first draw.** Full suite here: 251 passed, 5 skipped (the same five, none of them C1). The draw image's import chain is still standard-library only on the new app.py (re-simulated with torch, yaml and numpy blocked), and config.yaml is in the first-draw state (`items.sha256: null`, ARC at `main`).

The four notes, checked against the code (`/Users/eugene/prj/align/anatta-calib-c1/calibration/model_check/app.py`):

(a) True. `draw` runs `sample`, then `check_allocation`, then `check_battery_items` when `--battery-results` is given, and only then dumps the file, compares its sha256 with the registered one, writes it, calls `keep_items` and prints the sha and ARC lines. Without the flag it says the comparison was skipped. Tested with an ARC item edited upstream.

(b) True. `container_source` equals `inspect.getsource(evaluate) + inspect.getsource(out_dir)` byte for byte (verified under the tests' stand-in modal, where the plain function is reachable); the decorator is included, `draw`/`main` are excluded, and `functools.cache` makes every plan of one launch carry the same hash. Edits to `evaluate`'s body, its decorator or `out_dir` change the plan; laptop-side edits do not (tested). Neither function is recomputed in the container, so the cache has no container side.

(c) True. `sample` only draws (volume `/cache`); `keep_items` (volume `/runs`) writes the copy when the draw's sha256 is the registered one, or when nothing is registered and no copy exists, and otherwise leaves it; `draw` calls it only after the registered-sha check. I ran its rule table: first draw writes; a second unregistered draw leaves the copy; the draw that reproduces a registered sha256 replaces a stale copy; a non-registered draw changes nothing. The lockout is gone.

(d) True. `qwen_for_gate` builds Qwen's plan for this launch and exits if `--results` holds Qwen rows under another plan, before `force_ministral` is consulted; otherwise it reads the verdict only from a complete run. Tested for no run, another plan (forced and not) and a matching run.

Residual, unchanged from before and not blockers: the two facts only the draw itself can establish, both failing loudly before registration: MMLU-Pro's category counts at 527feea must give the registered allocation, and the battery draw must match the stored `items.json` (now caught in `draw`). The first draw command is `PYTHONPATH=src modal run calibration/model_check/app.py::draw --results DIR --battery-results /Users/eugene/prj/align/anatta-lab/lab/experiments/recognition/results/battery`, with DIR outside every repository.
