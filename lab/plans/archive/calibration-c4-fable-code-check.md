# Calibration check 4: Fable's cold code check (filed)

A fresh Fable agent (model fable) checked the C4 code cold, with no memory of the session that wrote it, before any C4
run, download or prepare. Its report file was last written at 12:15:45 UTC on 1 Nov (the file's clock). It is below,
unedited, with its own header. Its scratch files are in the session scratchpad's fable_c4_code/ folder. Opus 5.5
checked two of its claims against the code at de0fd0f:
- stored_hashes (app.py:481-491) returns None when the directions folder is missing, and the plan carries its result
  (app.py:540-541), so the plan id depends on whether the folder is there. DIRECTIONS is read at
  src/anatta/modal_app.py:41 (the report says :38), and the C4 worktree has no directions/ folder;
- card.py:180 compares `abs(got - want) <= f["tolerance"]` with no rounding. In Python, `abs(0.468 - 0.368) <= 0.10` is
  False, and the rounded comparison is True.

---

# Calibration check 4: Fable's cold code check

A fresh Fable agent checked the C4 code cold, with no memory of the session that wrote it: the worktree
/Users/eugene/prj/align/anatta-calib-c4, branch calib/persona-extractor, commits 6af1640 and de0fd0f on dc32cf7, against
the registration (NOTES.md:5949-6480, 10:27 UTC on 1 Nov), the eleven rulings (scratchpad check4_rulings.md) and the
design check, facts and open questions it adopted. Read-only: nothing was edited, committed, stashed, installed,
downloaded into a repository, run on Modal or paid for. No trait file or completion was opened; the repository files
were read with WebFetch at b8e0f04, which paraphrases beyond 125 characters, so what it could not settle is listed
under the fidelity check. The suite was run in the worktree with the library's own venv
(/Users/eugene/prj/align/anatta/.venv: Python 3.12.14, torch 2.14.1, transformers 5.21.0, tokenizers 0.23.3, the lab
image's versions; ~/prj/matlab/.venv has no pytest). The planted faults ran in a throwaway export inside the
scratchpad, never in the worktree. Scratch:
/private/tmp/claude-504/-Users-eugene-prj-align/073015e5-1016-4b21-9a35-1f58d24e5f8e/scratchpad/fable_c4_code/
(pytest_c4.txt, plant.py and work/plant_log.txt, rules.py, tiny_run/, copy/).

---

**Verdict: ready for the smoke run after small changes; the one change that matters for a grade is a test, not the
code.** Every registered condition, comparison, threshold, odds value, outcome rule and card field I could check is
implemented as registered or as ruled; the token slices, the closing tokens, the layer index, the system-prompt
handling, the generation settings, the split null, the ceiling and the bootstrap are right where they matter, and I
confirmed the slice arithmetic on the real Qwen and Ministral tokenizers with edge-case answers the suite does not
try. The safety path holds: nothing but numbers, names and hashes leaves a container, and an exception leaves as a
type name. Suite: 64 passed in 37 s, with the real-tokenizer tests running (both snapshots are in the local cache).
Of the seventeen faults I planted, the suite caught eleven with tests that name the right thing, caught one more
by an accident of the tiny model's row count, and passed six: a layer-index slip in analysis.run, C4-code graded on
P1-trim, the ceiling built from the question split (which decides recipe mismatch against inconclusive), the
least-strict verdict, unseeded G batches, and a P1-128 cap that ignores the answer's end. The code is right in all
six places; the whole-run test asserts that numbers exist but not what they are. A handful of cheap assertions
close that. Three things need
Eugene's word, none of them a code fault: the directions folder at launch, the batch-200 memory of G (which the smoke
cannot measure), and the judged arm's two rulings that are not implemented yet (it is off, and gated on a reader).

---

## Findings, ranked

### 1. The suite cannot see a layer-index slip in analysis.run, nor which reading C4-code is graded on

`calibration/persona_extractor/analysis.py:339` (`idx = [T.hs_index(b) for b in blocks]`), `:367` (E-stack's
`RL[t][1:28]` against `R[t][1:28]`), `:378` (C4-code on `D["P1"]`); `tests/test_c4_app.py:95-110`.
Severity: should fix before the run (a test, one or two lines).

The index arithmetic is right: block b is compared with R[b + 1] everywhere (`hs_index`), E-stack uses rows 1-27 of
both, the graded pair is block 19 with index 20, and `test_capture_at_block_l_minus_1_is_hidden_states_l` and
`test_p1_trim_slice_is_generate_vecs` prove the mapping at the reading level. But nothing proves that analysis.run
*uses* it, or which reading and which floor it feeds to the grades. Six planted faults pass the whole suite (scratch
work/plant_log.txt):

- F9, `idx = list(blocks)` (block b read against R[b]): 0 failures;
- F22, C4-code graded on P1-trim instead of P1: 0 failures;
- F23, the C4 ceiling built from `c_R_questions` (even against odd questions) instead of `c_R` (samples 0-4 against
  5-9): 0 failures. This one decides "recipe mismatch" against "inconclusive" on the card;
- F10, `verdict` taking the least strict outcome instead of the strictest: 0 failures;
- F13, `generate_batch` without `torch.manual_seed(seed)`: 0 failures (the rows record the seed they were meant to
  use whether or not it was applied);
- F20, P1-128 as `store[start:start + cap]` without the `min(end_trim, ...)`: 0 failures (the only cap test uses an
  answer longer than the cap; on G a short answer would then carry the two closing tokens into E-len).

A seventh, F11 (E-stack on rows 0-26 of R-lab32 against 1-27 of R), fails the two whole-run tests, but only because
the tiny model has 5 hidden-state rows and the slices come out 5 against 4; on Qwen both are 27 rows and it would
pass. `test_the_whole_run_on_stand_ins_and_its_resume` checks only that the C4-code and C4 rows have a cosine and an
interval and that E-stack and E-det are not "not run". On the tiny fp32 model the right alignment gives exactly 1.0
for E-dtype at every block and for E-stack at every index (I dumped the run's analysis: scratch tiny_run/), and
C4-code differs from E-trail only by the fixture's one closing token, so the faults are plainly visible in the
numbers the test already has in hand.

Fix, all in the existing tests: in the whole-run test assert `card["expectations"]["E-stack"]["min"] == 1.0` and
`card["expectations"]["E-dtype"]["min"] == 1.0` (P1-trim against R-lab32 on one fp32 model is the same arithmetic on
the same tokens; catches F9 and F11), that `an["cos"]["C4-code"]` equals `an["cos"]["E-trail"]` by layer (R equals
P1-trim exactly there; catches F22), and that each `ceilings.C4.by_layer[b]` equals `A.ceiling(floors["c_E:E-sys|
E-sys2"][b], floors.c_R[b])` recomputed in the test (catches F23). Add a unit test of `verdict` with constructed rows
(the eight cases in scratch rules.py all pass today; catches F10), a short-answer case to
`test_p1_128_is_p1_trim_of_the_truncated_answer` asserting `cap == trim` when the answer is under the cap (F20), and
a `generate_batch` call repeated with the same seed giving identical answers and a different seed giving different
ones (F13). `c4_base` and the E-null direction have no unit test either; the same constructed-rows test can cover
them (see finding 4).

### 2. Whether M′'s stored comparisons exist is decided by an environment variable at launch, and the plan id with it

`src/anatta/modal_app.py:38` (`DIRECTIONS = Path(os.environ.get("ANATTA_DIRECTIONS", "directions"))`),
`calibration/persona_extractor/app.py:481-491` (`stored_hashes`), `:540-541` (in the plan), `:153` (`out_dir` keyed
by `plan_id`). Severity: should fix before the full run (a refusal, three lines).

Launched from the worktree without `ANATTA_DIRECTIONS`, `DIRECTIONS` is the relative folder `directions`, which the
C4 worktree does not have (the suite's warnings show it: "no trait directions at
/Users/eugene/prj/align/anatta-calib-c4/directions"). Then `stored_hashes` returns None, the image carries no
directions, `check_stored` compares nothing, and E-M-rep and the v_ego comparison the registration promises ("The
results are compared with the stored 28 Oct directions") come out as "not run" with no complaint. Worse,
`stored_sha256` and `stored_sha256_qwen` are part of the plan, so the plan id differs with and without the folder:
a full run launched without it and relaunched with it lands in a new volume folder and reruns G, the readings,
R-lab32 and R from scratch, with the $20 stop (which spend.json carries across launches) then halting it part way.
The same C3 finding (check 3's finding 4) was accepted there.

Fix: in `main`, refuse a full run when `DIRECTIONS` is not a directory or lacks the six ego files and ego.pt (the
smoke may warn instead), and put the directions' sha256s on the ledger record (they are already in plan.json).
Consider keeping `stored_sha256` out of `plan_id` so the folder's presence cannot fork the run, recording it in the
mprime stage's result instead.

### 3. The smoke measures R's memory, but not G's, and a missing nvidia-smi would report "0 MiB" as a success

`calibration/persona_extractor/reference.py:281-306` (`PeakMemory`), `:366-368`; `generate.py:128-148`;
config.yaml:90-96 (`G.batch: 200`, `max_new_tokens: 1000`); `app.py:519-532` (the smoke's sizes).
Severity: should fix the zero reading (cheap); the batch size is Eugene's call.

- `PeakMemory._poll` swallows every exception, so if `nvidia-smi` is absent or fails in the container, `peak`
  stays 0 and `total` None, and `run_verbatim` records `peak_mib: 0` with exit 0. That is the one number the smoke
  exists to produce ("If R doesn't fit, R-lab32 becomes the reference before any real run"), and a zero would read
  as "fits with room to spare". Fix: after the subprocess, if `T.gpu_name()` is not None and `peak.peak == 0`,
  raise `CheckError("verbatim")` (or a new code) so the smoke fails loudly; and poll `torch.cuda.mem_get_info()`
  from the parent as a fallback when nvidia-smi is missing (it costs a small context in the parent, which is
  acceptable in the smoke).
- The smoke runs G at 2 × 1 × 2 rows per sign (batches of 4, 64 tokens). The full run's batches are 200 prompts
  with up to 1,000 new tokens on an L40S (48 GB). My count: 15.2 GB of bf16 weights, a KV cache of about 57 KB per
  token (28 layers × 4 KV heads × 128 × 2 × 2 bytes) × 200 × about 1,150 tokens ≈ 13 GB, plus transient cat copies
  of the DynamicCache and the per-step logits; about 32 GB, so it should fit, and I verified that 5.21's generate
  keeps only the last prefill logits (`logits_to_keep = 1`, which Qwen2 supports). But nothing measures it before
  the real G starts, an OOM there would be a StageError in the first batch (cheap, resumable), and the only remedy,
  a smaller `G.batch`, changes the registered "batches of up to 200 prompts" and the seeds' meaning. Eugene's call:
  accept the risk (I would), or let the smoke run one full-size batch (about 1-2 L40S-minutes).

### 4. C4-base misreads a cosine exactly ±0.10 from Figure 20's cell

`calibration/persona_extractor/card.py:180` (`abs(got - want) <= f["tolerance"]`). Severity: one line.

R's cosines are rounded to six places (`matrix`), and `0.468 - 0.368` is `0.10000000000000003` in floating point,
so a cell that lands exactly on the registered band's edge reads as outside it (scratch rules.py: `{'evil|sycophantic':
0.468, ...}` → passed False). E-cross already rounds its margin to six places (`round(same - max(others), 6)`) and is
immune; `E-geom` has the same bare comparison. The odds of hitting the edge are tiny, but "within ±0.10" is the
registered wording and the fix is `round(abs(got - want), 6) <= tol` in both places.

### 5. Four rulings are not on the card or in prepare yet

`calibration/persona_extractor/app.py:436-449` (`judge_stage`), `:621-640` (`prepare`), `card.py:224-229`
(E-trail-128), `card.py:319-332` (`generation`). Severity: before the judged switch is turned on (rulings 10 and 11);
the other two are card lines.

- Ruling 10: eval/prompts.py "joins the judged arm's download list. Prepare records its sha256 and which of the
  accepted forms it matched." Today `judge_stage` fetches the file at run time with no blob or sha check
  (`T.fetch_url(...)` straight into `named_string`), and `prepare` fetches only the three trait files. The tree at
  b8e0f04 gives prompts.py blob a3be82d059bd3045a026faacdd02ea9395e41df3, 1,117 bytes; my fetch shows the file is
  `Prompts = {}` followed by `Prompts["coherence_0_100"] = """..."""`, the subscript form `named_string` handles and
  the test covers. Fix: add the file to `prepare` (blob and bytes checked like a trait file, sha256 and the matched
  form written into config.yaml), and have `judge_stage` read it from the volume.
- Ruling 11: the judge template's sha256 goes on the card. Nothing records a hash of the filled or unfilled
  coherence prompt or of each trait's eval_prompt; `judge_stage`'s result has counts and snapshots only. Fix: return
  `sha256` of the coherence string and of each trait's eval_prompt from `run_judge` and put them under
  `card["judged"]`.
- Ruling 5: an empty probe.extract response "counted on the card per trait and model". `trail()` counts it
  (`skipped_empty`) and the count is in probe.json and in ledger.json's M′ block, but card.json's E-trail-128 entry
  carries only outcome, min, odds, threshold and the closing weights. Fix: add `read` and `skipped_empty` per trait
  to `out["E-trail-128"]`.
- Ruling 1: "The card says so, and says that shared seeds can only pull G′ toward G." The card shows G′'s settings
  and processors, not that G′ batch i reuses G batch i's seed (seed_base 1000 for both, plan only). Fix: a `note`
  field beside `generation["Gprime"]` with the seed base and the E-samp caveat.

None of the eleven rulings conflicts with the registration: 2, 3, 4, 6, 7, 8 and 9 are implemented as ruled; 7
widens the smoke to every stage at smoke size, which the registration's "read as nothing" covers.

### 6. R's wall time is likely 1-2 hours, not 25-35 minutes, against a 2-hour timeout

`calibration/persona_extractor/app.py:349-350` (`timeout=2 * 3600`), config.yaml:58 (A100-80GB). Severity: note;
the per-trait `done.json` makes it resumable, and the budget covers it.

The verbatim script runs fp32 with autograd and TF32 off (torch's default in both stacks; E-stack would show it
otherwise). A 7.6B-parameter forward over an n-token row is about 15 n GFLOP; at an A100's 19.5 fp32 TFLOPS and
realistic utilisation, a 500-token row is roughly 0.5-0.8 s and a 1,100-token row over 1 s, plus graph building.
For 6,000 rows that is 50-120 minutes on an A100 (an H100 is about three times faster in fp32). If a container hits
the 2-hour timeout mid-trait, that trait's run is lost (the script writes only at the end) but the finished traits
resume. The registration's $2-2.5 becomes $3-5; still inside the stop. Nothing to change unless the smoke's
per-row seconds (recorded as `seconds` with the long pair) say otherwise; the smoke's timing is the thing to read.

### 7. A fix to analysis.py after the run forces a full rerun, and the cheap tier refuses after any code change

`calibration/persona_extractor/traits.py:37` (`CHECK_FILES`, hashed into the plan), `app.py:153` (`out_dir` by
plan id), `:192` (`check_environment` in every stage, the cheap stage included). Severity: note (design).

Every check file but card.py is in the plan id, so a one-line fix to analysis.py (the stage most likely to need one
after real numbers are seen; check 3's code check found two card-side bugs) moves the run to a fresh volume folder
and reruns G, the readings, R-lab32 and R (about $10). And the `cheap` entrypoint, meant to run later under loop
rule 4, refuses with `env_code` unless the checkout is byte-identical to the run's. Options: hash analysis.py and
readings' *analysis-only* code into the stage's env record rather than the plan id (the GPU stages' inputs stay in
the id), and let `cheap_stage` accept a differing analysis.py. If the current design is kept, the ledger should say
that an analysis fix means a rerun.

### 8. Smaller points

- `calibration/persona_extractor/readings.py:479-484`: in M′, a budget stop raised by `commit()` inside the trail's
  `try` is recorded as a trail failure for that trait (`CheckError:budget`) and the loop goes on until the next
  extract's commit raises it outside a try. Cosmetic: one spurious failure entry. Move `commit()` out of the try.
- `readings.py:292-296` (`lengths_of`): "the share of each E arm that reaches 128 tokens" is read by re-tokenizing
  the decoded text without special tokens, which can differ from the generated count by a token or two; the count
  is reported, not graded, but the card should say "re-tokenized". The G arm's `new_tokens` is exact.
- `app.py:659-668`: the csv stage (cpu 4, 16 GB, about $0.90 an hour on Modal's rates) is billed at the "small"
  tier ($0.25); minutes, so cents. Modal's page is the authority, as the card says.
- `card.py:301-316`: the registered "sha256 of … every check file" is met for the five files the containers run and
  the vendored script; card.py, config.yaml and app.py's laptop half are not hashed (config's content is in the
  plan; `container_sha256` covers app.py's container functions). A line naming what is and isn't hashed would do.
- transformers 5.21 warns, on loading Ministral's tokenizer at f6fae97, that its pre-tokenization regex is wrong
  ("set fix_mistral_regex=True"). The library loads it without the flag, as it did on 28 Oct, so M′ is a like-for-
  like reproduction; the ledger's M′ line should carry the warning, since it bears on what "Ministral's tokens" mean.
- `test_c4_app.py`: on the tiny model E-trail and E-trail-128 are "not run" (their registered blocks 7-26 do not
  exist in a 4-layer model, and `tiny_config` scales `analysis` but not `registered.expectations`), so the card's
  E-trail path is never exercised with values. Scaling those two block ranges in `tiny_config` fixes it.
- `reference.py:133-137`: `csv_records`' docstring says `scores None` gives the constants; `run_csv` always passes
  the `{"constant": ...}` dict, and `None` would raise. Docstring only.

---

## What the post-prepare fidelity check must cover

WebFetch paraphrases beyond 125 characters. It confirmed the items marked "(fetched)" at the line level; the rest
must be read off the files at b8e0f04. Blob shas and sizes from the tree (recursive, not truncated):

1. **The three trait files** (evil 9fab8d12…, 3,943 B; sycophantic b0d4367d…, 5,428 B; optimistic 748217a1…,
   4,109 B; fetched from the tree, matching config.yaml). `prepare` checks blob, size, the three keys, 5 pairs of
   non-empty strings and 20 non-empty questions. Still to confirm by eye: that no `eval_prompt` contains braces
   other than one `{question}` and one `{answer}` (`str.format` in `run_judge` would raise on any other; the repo's
   own `judge.py` fills with `.format(**kwargs)` too, so it should hold).
2. **eval/eval_persona.py** (61ac5b61…, 12,927 B). Fetched and matching traits.py: `a_or_an` is
   `return "an" if word[0].lower() in "aeiou" else "a"`; the system string is
   `f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"` with assistant_name the trait for
   pos and "helpful" otherwise; `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`; records
   `dict(question=..., prompt=..., answer=..., question_id=...)`; questions outer, instructions inner, samples
   innermost; `to_csv(output_path, index=False)`; no `.strip()` on answers; temperature 1.0 when n_per_question > 1;
   `SamplingParams(temperature, top_p, max_tokens, skip_special_tokens=True, stop=[tokenizer.eos_token],
   min_tokens)`. Not quoted by the fetch, so to read: that `messages` is exactly `[{"role": "system", ...},
   {"role": "user", "content": question}]` with the question unmodified; `sample()`'s defaults `top_p=1` and
   `min_tokens=1`; and that vLLM's stop set includes both of Qwen's eos ids (G stops at [151645, 151643]; the
   design check says vLLM's `update_from_generation_config` adds the generation_config's eos ids, so it should).
3. **judge.py** (47da9058…, 6,212 B). Fetched: `self.prompt_template.format(**kwargs)`; `max_tokens=1,
   temperature=0, logprobs=True, top_logprobs=20, seed=0`; the 0-100 weighted mean with None under 0.25. To read:
   the keyword names eval_persona passes (`question=`, `answer=`), which ruling 11's `str.format(question=, answer=)`
   assumes.
4. **eval/prompts.py** (a3be82d0…, 1,117 B). Fetched: `Prompts = {}` then `Prompts["coherence_0_100"] = """..."""`,
   not an f-string, `{question}` and `{answer}` once each and no other braces, about 1,060 characters, ending
   "Respond with a number between 0 and 100. Don't say anything else, just the number." To record: its sha256 and
   exact length (ruling 10), and that `named_string` returns the whole string (the test covers this form).
5. **scripts/generate_vec.sh** (0c0e15ef…, 1,021 B). Fetched: pos with `--assistant_name evil`, neg with
   `--assistant_name helpful`, `--version extract`, judge `gpt-4.1-mini-2025-04-14`, `generate_vec.py ... --threshold
   50`. Matches `verbatim_command` and config's threshold.
6. **requirements.txt** (1450b000…, 285 B). Fetched: 18 pins; the image takes transformers 4.52.3, torch 2.6.0,
   accelerate 1.7.0, pandas 2.3.1, tqdm 4.67.1. The repo also pins PyYAML 6.0.2; the image gets yaml through
   transformers at whatever pip resolves (app.py's `load_yaml` needs it; the script does not). Record the resolved
   version from the freeze.
7. **generate_vec.py** (d819f48e…, 5,250 B) and **LICENSE** (2d13a348…, 11,358 B): both match the vendored files
   (the test and the tree agree). Nothing to do.
8. **eval/model_utils.py** (7f782092…, 3,457 B): the `LLM(...)` call passes no dtype and no seed (facts §3), which
   underlies the "same sampling distribution" claim for G; a glance.
9. **output/qwen2.5-7b-instruct_baseline.csv**: that its prompt column begins with the full rendering
   (`<|im_start|>system\n…`), the one shipped example of the format eval_persona stores (facts §2); structure only.

Also, from the code's own C4-QUESTION markers: `traits.py:319` (a_or_an) is now settled by item 2; `reference.py:381`
and `:452` by items 3 and 4 once read by eye.

---

## What I verified as right

**The token slices and the closing tokens, the C4-code grade's foundation.** `token_slices` builds the prompt and the
full rendering exactly as probe.response_activations does (`chat_ids` with and without the generation prompt), and
`read` takes P1 with the same expression on the same forward pass, so the bitwise test is a test of identity, not of
luck. P1-trim's end is `len(tok(p + a))` and its start is checked against generate_vec's `len(encode(p))`, with a
second check that the rendered reply's tokens begin with those of `p + a`. On the real Qwen tokenizer at a09a354 I
ran answers with trailing spaces and newlines, a leading double newline (the registered merge: n_answer drops by one
the same way on both sides), the empty string, a lone space, a literal "<|im_end|>", CJK text and a 2,000-character
run: every case passes both checks, the closing tokens are always `['<|im_end|>', 'Ċ']`, and `apply_chat_template` of
the three turns equals `p + a + "<|im_end|>\n"` character for character (the template has no strip). On Ministral the
closing token is `</s>` alone, with and without a system turn (26 and 532 prompt tokens), and an empty assistant turn
raises jinja's TemplateError, the registered M′ tail risk, which `run_mprime` catches per trait. The fixture test
checks P1 against P1-trim plus the one closing token algebraically, and P1-128 against P1-trim of the truncated
answer. Planting a one-token-long trim (F1), a start one token late (F12) or a cap that ignores the answer's end
(F20) fails two, one and one tests respectively.

**Layers.** `hs_index(b) = b + 1`; the test asserts `capture` at block b equals `output_hidden_states[b + 1]` for every
b under transformers 5.21 (whose recorder puts the embeddings at index 0 and the post-norm state last, like 4.52.3),
and `test_p1_trim_slice_is_generate_vecs` checks P1-trim at block b against the vendored function's response mean at
index b + 1 on the same model, which is the cross-check that matters. Planting `hs_index(b) = b` (F2) fails three
tests. Index 28 and block 27 are never compared; E-stack runs over 1-27; blocks 0-26 are read; the arms cover 7-26.

**The reference.** R-lab32 is the vendored `get_hidden_p_and_r`, compiled from the file's own syntax tree after its
blob is checked (pandas is imported at module level in the file, so this is the only way to run it in an image with
no pandas), under an outer no_grad, with `avg_diff` copied from `save_persona_vector`; the test runs
`save_persona_vector` itself on CSVs this check wrote and gets `torch.equal` on all three outputs. R is the script
called as generate_vec.sh calls it, with the snapshot path as `--model_name`, in the second image with the five pins,
`HF_HUB_OFFLINE=1`, its stdout and stderr on the volume and never printed, and the constant score columns (pos 100,
neg 0, coherence 100) that make the paired mask keep every row without touching the script. The `del outputs` at the
end of each loop iteration in the vendored file means one autograd graph is alive at a time, so the design check's
"two graphs" was conservative: about 30.5 GB of weights plus roughly 15 GB per 1,100-token graph, which is why 80 GB
is right and 48 is not.

**The CSV step.** Written and read back with the second image's own pandas; `survives` compares bytes of both the
answer and the prompt; a pair fails if either answer fails the round trip or has an empty answer slice under either
stack's tokenizer; the three registered examples ("", "None", "NA") are flagged by the stand-in, whose NA set is
pandas' documented default, and `csv_selftest` repeats the check with the real pandas before any CSV is written.
The drop applies to every extractor (keep.json feeds readings, R-lab32 and the CSVs), so it cannot bias C4-code.

**G and G′.** Every setting is passed explicitly (`kwargs_for`), the processors transformers builds are recorded on
every batch and checked (G: no temperature, top-k, top-p or penalty processor; G′: each at probe's value within
1e-6), the merged GenerationConfig is kept, the model's generation_config is asserted against the registered
fields before anything generates, and `test_settings_left_to_the_generation_config_are_caught` shows the check bites
when top_k and the penalty are left to Qwen's defaults. Planting a no-op forbidden check (F6) fails one test. Left
padding, no special tokens, decode with skip_special_tokens and no strip, eos [151645, 151643], min 1 and max 1,000
new tokens, batches of up to 200 of one sign in question → instruction → sample order, seed 1000 + the batch index
over the whole arm (and G′ reuses the index, as ruled). `top_k=0` validates in 5.21 and builds no warper. New-token
counts stop at the first eos; "capped" is no eos within the limit.

**The probe arms and the recorder.** The four arms call `probe.extract(model, tok, trait, include_system=..., seed=...)`
and nothing else; the trait is built with eval_persona's two system strings as its pairs, so extract generates
under the repo's prompts. The recorder rebinds the two module names only inside the `with`, returns the originals'
objects, restores in `finally` (planting a missing restore, F3, fails one test), and the fixture test shows
bitwise-equal vectors, scores and means with and without it, plus an exact rebuild from the recorded means.
`check_alignment` ties every recorded response to its job and question. `meta["examples"]` is popped before
`direction.save` (planting its removal, F7, fails two tests), and the responses go to the volume only. E-det compares
the seed-0 pair position by position. E-trail-128 reads the recorded E-sys responses with their own system string
through `read`, skipping and counting empty slices as ruled. M′ loads the six ego traits with their sha256 checked,
catches each call by exception type, trims `</s>` on the no-system calls, rebuilds v_ego with `factor.composite`,
and compares with the stored directions at the stored best layer and over 8-32.

**The analysis.** Per-question float64 sums rebuild both directions on one shared draw of 2,000 resamples (seed 0),
R-lab32's means standing in for R in the resamples and R itself in the point estimate; c_R from samples 0-4 against
5-9 of each cell, the question split reported beside it; the ceiling is `sqrt(c_E) · sqrt(2c_R/(1 + c_R))` and the
planted-reliability test holds it within 0.08 (planting the sqrt's removal, F5, fails it); the split null keeps each
pair whole and halves every cell, so the signal cancels and the anisotropy stays, and the test shows it near the
random level where the label flip sits above 0.5 (planting the flip in its place, F4, fails it); the random null is
1,000 unit vectors at seed 0; specificity, the 3 × 3 matrices, E-band's median over 7-26, the norms, the other
outputs at index 20 and the ledger line all read from the registered places. The cheap tier's reference (the first
kept sample of every cell per sign, P1 at every block and R-lab32 at index 20) is written and `run_cheap` compares
against it with the registered 0.999 and 0.99.

**The card.** C4-code on `C4-code`, C4 on `E-sys-vs-R` with the C4 ceiling, C4-lab on `E-vs-R`, all at the block in
the run's own plan; the four outcomes per trait in the registered order with the stricter one winning
(`verdict`: eight constructed cases incl. "fail beats recipe mismatch" and "a missing number is incomplete" all
right; planting the least-strict choice, F10, is not caught, see finding 1); the tally counts C4 as a miss in the
last two; C4-base over the three Figure 20 cells of R at index 20; every expectation with its threshold, range and
odds equal to the registration's (I checked all twenty-two numbers in config.yaml's registered block); E-null reads
"≤" and over all three traits, E-cross over all three, E-opt on optimistic, E-geom against R's matrix, E-judge only
if the judged analysis ran; M′'s three expectations in the ledger only. C4-code is also read against R-lab32 on its
own line exactly when E-stack misses. The hashes, revisions, trait-file facts, pins, both freezes, GPUs, Python
versions, attention implementation, generation settings as passed and merged, and the vectors' sha256s are there.

**The Modal app.** Two images: the lab's is modal_app's with the check files added, no new package (the only new
import, tqdm, is a transformers dependency and has a fallback); the second is Python 3.11 with exactly the five
pins, `HF_HOME` on the cache volume, `HF_HUB_OFFLINE=1`, and the package copied in. Volumes are anatta-hf-cache and
anatta-runs; the app is anatta-calib-persona-extractor; nothing names anything beginning with matlab (a test
asserts it on the app name; I grepped the folder too). The verbatim stage is `gpu=["A100-80GB", "H100"]`, a valid
fallback list in modal 1.6.2; `with_options(env=...)` exists with that signature; no retries; timeouts as the README
says; the secret is attached only when the judged switch is on and the key is read from the environment inside the
container and never printed. Each container checks its code hashes and installed versions against the plan before
loading a model, resumes from its own files, and the laptop resumes from `stored`. Spend is wall time at the config
rates, printed after every call including a failed one, enforced before each call and inside each container at its
commits (planting `budget_s=0` keeps the first batch and raises `budget`).

**Safety.** `assert_text_free` runs on every stage result before it is written or returned (alphabet of names,
numbers, hashes, versions and paths; the GPU name is the one exception), and the whole-run test checks every JSON a
run writes against every generated string on the stand-in volume. `run_guarded` swallows the original exception
inside the `with` and raises a `StageError` carrying the stage, the row and the type (and a CheckError's code and
numeric facts; any string fact is dropped), with no `__context__` or frame; planting the message into it (F8) fails
two tests. The verbatim script's output stays in files on the volume. The judge's prompt and reply never leave
`ask`. The fixtures' strings are made up and benign. The only text that reaches the laptop is the card headline's
outcome words and numbers.

**Planted faults.** Seventeen in a throwaway export (scratch plant.py, log in work/plant_log.txt), each the only
change to a fresh copy, the whole check suite run on each:

| fault | where | tests failing |
|---|---|---|
| F1 P1-trim one token too long | readings.token_slices | 22 (slice, closing-token, cap, real-tokenizer and whole-run tests) |
| F2 hs_index(b) = b | traits.hs_index | 6 (P1-trim against the vendored function, the capture-vs-hidden_states test, the layout test) |
| F3 recorder never restores | readings.Recorder | 2 |
| F4 split null replaced by the label flip | analysis.split_null | 1 (the test written for it) |
| F5 ceiling without sqrt(c_E) | analysis.ceiling | 1 (the planted-reliability test) |
| F6 forbidden-processor check a no-op | generate.check_processors | 1 (the test written for it) |
| F7 meta examples kept | readings.extract_one | 4 (the extract_one test and all three whole-run tests) |
| F8 guard carries the original message | traits.Guard | 2 |
| F9 analysis reads block b against R[b] | analysis.run | **0** |
| F10 verdict takes the least strict outcome | card.verdict | **0** |
| F11 E-stack rows 0-26 against 1-27 | analysis.run | 2, by a shape accident of the 4-layer fixture; 0 on a 28-layer model |
| F12 P1 starts one token late | readings.read | 2 (the bitwise test and the closing-token test) |
| F13 G batches unseeded | generate.generate_batch | **0** |
| F20 P1-128 cap ignores the answer's end | readings.read | **0** |
| F22 C4-code graded on P1-trim | analysis.run | **0** |
| F23 C4 ceiling from the question split | analysis.run | **0** |
| F24 probe trait's pos and neg strings swapped | traits.probe_trait | 1 |

Finding 1 gives the assertions that would catch the six (and make F11's catch a real one).
