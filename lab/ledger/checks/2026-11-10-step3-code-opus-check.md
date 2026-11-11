# Step 3 code (understanding first, then practice, then contrary training): two cold Opus checks (10-11 Nov 2026)

Two fresh agents (model opus, read-only, no Modal, network or paid calls) checked the step 3 code against the design
note `lab/plans/step3-understanding-first-2026-11-10.md` (revision 2) before anything was spent. The code was written
by a builder agent (Claude, Fable 5.2) from that note: `lab/explore/understand/` and the functions `u1_sets` and
`press` in `lab/explore/act1/modal_act1.py`, uncommitted on top of 96116ee when the checks ran.

- **The first check** (10 Nov, about 23:25 to 23:46 UTC) read the code as built. It was given the note, the code and
  the existing code it calls, eleven questions and the five invalidity checks. It was not given the builder's own
  report or what Fable thought of the code. Verdict: no blocker; six things to fix before any spend.
- **The second check** (11 Nov, about 00:40 to 01:05 UTC) read the code after the first round of fixes. It was given
  the note, the first report and the list of changes asked for. Verdict: no blocker; four more guard rails to close.
- **The fixes after the second check were not checked cold.** Fable read them and ran the tests (2,012 passed).

Both reports follow, filed by Fable (subagents can't write report files). Their substance is unedited. One line is
changed in each: where it named a local scratch folder, it now says the scripts were kept outside the repo. Line
numbers in a report refer to the code as that agent read it. The code with every fix is in the commit that adds this
file.

## What Fable did about the first check

- **Should-fix 1, the readout never checked the freeze (taken).** `--look`, `--all`, `--ended` and `--dose` refuse
  under code that differs from the freeze the dose check recorded.
- **Should-fix 2, forking at the dose check and the main tag (taken, in two rounds).** One dose check and one main
  run for a set of sets; see the second check's item B.
- **Should-fix 3, an interrupted launch (taken).** A reserve records its launch. A stage with an unsettled reserve
  is not launched again. `--fetch` copies what finished and settles the reserve, and only once the containers can
  no longer be running.
- **Should-fix 4, the local deadline counted queue time (taken, with a longer wait).** The launcher waits 1.5 hours
  beyond a run's reserve, not the 0.25 suggested, because nine GPU containers may queue behind each other. The
  container's own timeout still caps the bill.
- **Should-fix 5, a dropped pair shrank one run's read (taken, both halves).** `press` prepares the pairs before it
  loads the model and fails if a read pair is missing. The readout refuses a read that does not hold exactly the
  planned pairs and samples.
- **Should-fix 6, the brief's wording in training documents (taken, and widened to every arm).** A training
  document that holds a run of 8 words of its brief's content or rules is dropped. Counted on the pilot's files: 424
  of the first 3,000 rule documents hold such a run and none of the neutral ones. So arm R is no longer exactly the
  pilot's first 3,000: it is the first 3,000 clean rows of the same file. What the rule does not catch (runs of 7
  words or fewer; the topic phrase in a brief's first paragraph) is in the note's addendum.
- **Minor, taken:** the guards' flag goes on every reading of a flagged arm and into what is said; H is labelled as
  minus the drop; a run that ends early can be reported (`--ended`); the freeze covers the whole library,
  third_party, the pairs file and the generator's config; the sets are built by committed code and record it; the E
  files must be what the current code selects; the older stack loads the model with the hub switched off; a passing
  smoke comes before the build and a recorded hand read before the sets; `--fetch` finds the old read's second
  folder; the readout checks each run's recorded settings against the plan; `press` leaves no summary of an arm's
  outcomes in its log.
- **Minor, not taken:** the lenient judge form and "sharpen once" are not enforced by code (the ledger records
  them); the judge's fingerprint covers its messages and not the two-line parser (every verdict row keeps the raw
  reply); "three UTC days" is a description, not a rule; failures stay counted per stage; the dose check's two-path
  comparison stays on the 28 pairs with the 38-pair figure beside it.
- **Added by Fable, not from this check:** a rehearsal of one run's three stages before the dose check
  (`run_u1.py --plumb`), because the dose check exercises phase 3 only; two lines in the E brief ("release" among
  the forbidden words; a rule asking for the document's own wording).
- **Checked by Fable on the real files, free, while the first check ran:** Stage A's 6,192 records and turns line up
  and equal the local episodes file; the told set is 555 episodes (102 releases, 453 solved; 902 turns; 179,906
  tokens); every stored first-turn prompt (6,192 and 555) re-renders to the same token ids; the practice set is
  389 episodes (76 releases, 313 solved; 625 items; 118,910 tokens); 28 one-off eval pairs carry no practised
  release; both stream fingerprints match the local ones; the model cache on the volume holds exactly the pinned
  snapshot and refs/main names it; the installed Modal library has the calls the launcher uses.

## What Fable did about the second check

- **A, an ended run was not final (taken, with one change).** A block that has not started is refused once the run
  has ended; blocks start in order; no block starts while any reserve is unsettled. A block that had already started
  may still finish, because the stop is between blocks and a half-run block could otherwise be neither completed
  nor dropped.
- **B, two dose records and the rebuild route (taken).** The one-record rule is checked again just before the
  record is written. A dose tag left unfinished can be taken up under a new tag only by the same code. More than one
  record for a set of sets is a refusal whatever tag is named. Sets that carry a dose check or a main run are
  rebuilt only as a revision (changed code, documents or volume records).
- **C, a launch that never started counted as failed runs (taken).** A launch whose app cannot start, or whose
  spawn fails, cancels what it spawned, trues up and records no failure. A broken connection while waiting is an
  interrupted launch, not a failed run. `--fetch` settles nothing when the volume cannot be read.
- **D, the hand read could be redone under another seed (taken).** The read is of the sample drawn under seed 0. A
  read with three or more violations is binding for those documents.
- **Minor, taken:** the look is taken once; a new main tag's folder must be empty; refs/main must name the pinned
  snapshot; an interrupted rebuild leaves the old sets record in place; the pairs file the read's rows come from is
  in the freeze; the dose check's log line says its counts are on all 38 pairs.
- **Added by Fable:** the dose check needs a rehearsal that passed on the same sets under the same code.
- **Minor, not taken:** files in the results folder (dropped.json, `ended`) are not made tamper-proof; the readout's
  plan check covers phase 3 and leaves phases 1 and 2 to the freeze.

---

# The first report

COLD CODE CHECK: step 3 (understanding first), code against lab/plans/step3-understanding-first-2026-11-10.md rev 2
Checked by a fresh Opus 5.5 agent, read-only. I wrote nothing in the repo and made no network, Modal or OpenRouter calls. Scratch scripts were kept in a session scratch folder outside the repo (counts.py, synth.py, droppair.py, docs_check.py).

## 1. VERDICT: NO BLOCKER

The readings, the rule, the three contrasts, the gates, the look, the guards and the dose rule all compute what the note fixes. I confirmed this against hand-computed values on synthetic 8-seed data. The arms differ only in their documents, and the 72 Act I eval pairs are kept out of the practice set, the stream and the docs, matched by pair name. What is weak is the run discipline around the readings: the readout never checks the code freeze, nothing stops a second dose tag or main tag, and an interrupted launch is easy to relaunch twice. Those should be closed before money is spent. None of them changes a number computed from honest runs.

## 2. BLOCKERS

None.

## 3. SHOULD FIX BEFORE ANY SPEND

**(1) The readout never checks the freeze (invalidity #2 and #5).** compare_u1.py look_report (:155) and load_all (:226) never call run_u1.check_frozen (run_u1.py:304) against the dose record's freeze. The file contains no "freeze" at all. Only launches check it (run_u1.py:628). So after results exist, someone could edit u1_core.py's rule, thresholds or outcome table and run --look/--all, and the readings would come from edited code with no warning. Smallest fix: in load_all and look_report, read run.json, find its dose record, and call R.check_frozen(out, dose["freeze"]). Refuse if it fails.

**(2) Forking at the dose check and at the main tag (#4).**

- a. Dose: cmd_dose (run_u1.py:492-499) refuses only when the same tag already holds dose.json. After a dose record that failed, a new tag can run the dose check again, and with fresh code, because the new tag's freeze is taken from whatever is committed then (:501-518). find_dose (:564-578) then refuses "more than one" without --dose-tag, but --dose-tag picks the passing record.
- b. Main tag: cmd_block (:610-631) builds run.json for any new tag. Seeds 0-7 can be rerun under a second main tag, with the dose record unchanged. Only the $135 stop limits this.

Smallest fix: refuse a new dose tag when any dose.json already exists under the same out_dir/sets root. Refuse a main tag when another run.json exists (or record every main tag in one file and refuse a second).

**(3) An interrupted launch double-runs (run_u1.py:385-431).** Booking happens at :398 and the true-up at :428 is reached only when launch returns normally. An interrupt (Ctrl-C, a dropped laptop session) leaves the reserve booked with no settle path; cmd_fetch says this outright. No error_try file and no done.json/info.json is written locally. So the obvious next step, re-running the same --block command, launches the run again while the detached container is still going.

- For finetune, two containers then write the same out_dir on the volume at the same time.
- For press, the second one hits "already holds a run" (modal_act1.py:940), is recorded as a failure, and uses up one of the two tries. A second such slip drops the seed from every arm.

Smallest fix: before launching, refuse a stage whose latest spend entry is a reserve with no true-up, and point to --fetch.

**(4) The local deadline counts queue time (run_u1.py:400).** The deadline is t0 + upper_h, taken before spawn, so a cold start or a GPU queue on Modal eats into it. A healthy run can be cancelled at :418, charged its full reserve (:421) and recorded as a failure, and two such failures drop the seed. The pilot timings leave little slack:

- press: about 40 min estimated against 1.15 h
- docs: about 30-35 min against 1.0 h
- practice: about 12 min against 0.5 h
- dose: about 50 min against 1.5 h

Smallest fix: set the local deadline to upper_h + 0.25 h. The container timeout at :403 still caps billing.

**(5) Pairs dropped in the container silently shrink one run's read (#3).** modal_act1.py:960-961 runs prepare_pairs and then keeps only names = [x for x in eval_names if x in pairs]. A pair dropped inside one container leaves only that run's read, and the contrast is then taken over different pair sets.

- The only trace is the "odd_sizes" note (compare_u1.py:273-276).
- "want" there is taken from N seed 0's pair count, not from len(oneoff). If N_s0 itself lost a pair, every other run would be flagged and N_s0 would not.
- Shown by running: droppair.py removes one pair from E_s0. --all still computes every contrast (that seed's U1 difference moves by 1.6 points) and only notes odd_sizes {'E_s0': [148, 148, 148]}.
- It has been rare: prep_dropped was empty in all 27 past records I found.

Smallest fix: make press raise when prep["dropped"] touches an eval name, or make load_all refuse when any run's info.eval_pairs != len(oneoff). Compute "want" from the split.

**(6) Claude-written brief prose could pass into E training documents (hard rule: no Claude-written text is trained on).** The E brief (u1_core render_brief_e) is prose Claude wrote. explained_flags (u1_core.py:234) adds "wording" and "names_release" to the pilot filters, but nothing counts or rejects long spans copied from the brief. If Qwen copies brief sentences word for word, those sentences are trained on in phase 1. The pilot's R/N documents had the same exposure, so this is not new to E, but E's brief is longer and more explanatory.

Smallest fix: in --build, count 8-grams each document shares with its rendered brief. Report the distribution, and filter above a small threshold, for the lab to decide.

## 4. MINOR

- Guards are computed (compare_u1.py:250-251) but never attached to readings or to the outcome wording ("said"). A "partly indiscriminate" arm is reported beside its reading, not on it. The note expects the flag to qualify the reading.
- Per-seed H prints as a negative number for a typical run, because it is minus the drop (u1_core.py:655-656). Label it "minus the drop in log-odds" wherever per-seed values print.
- Paths the note calls "reported": --all refuses (load_all :236-238) when block 3 never starts because of the $135 stop, or after the look says stop. In those cases the runs that exist cannot be reported through the tool. The note is silent on how.
- Gaps in the freeze:
  - FREEZE_FILES (run_u1.py:77) leaves out third_party/evalplus, koan_pairs.py and declared/config.json.
  - sets.json records no code hashes, and the sets can be built from uncommitted code before the freeze.
  - The E manifest's code_sha256 is not compared with the freeze.
  - A .pytest_cache folder inside understand/ would trip freeze_paths, which hashes every file there.
- finetune (h4lite modal_sft) and the old-path read load the base by hub id, and the image has no HF_HUB_OFFLINE. Snapshot drift is caught only at press (modal_act1.py:943), after docs and practice have been paid for.
- Procedure steps the code does not enforce: smoke passed before build, the hand read before sets, and the lenient judge form only after a second smoke plus a ledger line.
- judge_sha covers the messages, not the verdict parser.
- --fetch ignores the old path's _try folders.
- A booked, unsettled reserve (item 3) also counts toward the $135 stop at block_stop (:603) and could refuse a block early.
- compare_u1 does not check each run's info against the plan: seed, stream, init path, READ and cfg are recorded (modal_act1.py:1006-1008) but never compared.
- The per-round measures for every arm sit in log.jsonl and info.json as plain JSON (see blinding).
- The two-failure count is per stage (run_u1.py:641, 681). The note is ambiguous.
- "Three UTC days" is not enforced.
- The dose-paths check runs on the 28 unpractised pairs, with the 38-pair figure shown beside it. The note is ambiguous.
- "Unpractised" means no release was practised. The told adapters did practise solved episodes on some of those pairs.
- Stage A outcome labels come from the old scoring and are not re-scored. Payment under GAMEABLE is unaffected.
- E and R share topic and genre draws by brief index. This matters only for the held-loss comparison.

## 5. DEPARTURES FROM THE NOTE

| Where | The note says | The code does | Could it favour an outcome? |
|---|---|---|---|
| u1_core.py:655-656 | H is "the drop" in log-odds; A−B > 0 means A holds better | H = −drop, so A−B > 0 means A fell less | No. The sign matches the note's sentence; only the label differs |
| u1_core.py:735 | Outcome table has no row for "no difference" + "holds worse" | An explicitly marked OUTCOME_UNLISTED wording | No. It is worded neutrally and flagged |
| u1_core.py:266-270 | Rubric as written in the note | Adds "keyword" to format; adds "knows, believes, sees" and "what tests are for" to the exemptions; adds "never hard-codes the output" to the strict form | Slightly. It widens what E documents may say, so E passes the screen more easily. It does not touch the readout |
| gen_docs judge | "the judge screens every document" | Asks the judge only as far as the quota needs, in fixed generation order (passing prefix) | No. The order is fixed before verdicts and there is no cherry-picking |
| E selection | Implied by the design | E passes a Claude judge screen that R and N did not | It is part of the design. Worth one line in the write-up |
| dose check | Paths compared within 15 points | Compared on the 28 unpractised pairs, with the 38-pair figure shown | No. Ambiguous in the note |
| u1_sets assertion 2 | Check a sample of stored prompts | Checks all stored prompts | No. Stricter |
| compare_u1 :250 | Guard flags the arm | Reported separately, not attached to "said" | Could understate a flagged gain in a summary |
| run_u1 two failures | "two failures drop the seed" | Counted per stage | No |
| emphasis mapping | New emphases | Mapped onto the pilot's draws by index; arbitrary but uniform | No |
| three UTC days | Blocks across three days | Not enforced | No |
| read seed (modal_act1.py:970) | Fixed by round | seed_base + r, the same across seeds and arms | No. It helps pairing |

## 6. VERIFIED BY RUNNING versus ONLY READ

Ran:

- pytest lab/explore/understand: 131 passed in about 10 s.
- gen_docs.py --dry: the first-round judge upper bound is $36.67, capped at $16. Generation is capped at $3 and smoke at $1.
- run_u1.py --dry: the gameable stream has 4,560 Stage A episodes off the eval pairs, 983 of them paid (hacked 89, near_miss 166, overfit 10, solved 718). A round of 192 items holds 22-42 hacked, mean 33.8. The honest stream has 718. Costs:
  - expected for three blocks: $98
  - dose reserve: $17.82
  - block reserves (9 runs): docs $26.10, practice $13.05, press $27.16
- counts.py (fields only):
  - Gate 0: 276 train rows; 72 eval pairs (38 one-off, 34 conflicting).
  - Told releases: 102 on 72 pairs. 18 of those pairs are eval pairs, 10 of them one-off, which leaves 28 unpractised. 76 of the releases are off eval.
  - Stage A: 6,192 episodes, all within the Gate 0 train split, 1,632 of them on eval pairs (excluded by name).
- docs_check.py: the first 3,000 R and N rows reproduce the hard-coded D3k/N3k fingerprints. Held and train ids are disjoint. Words: R 1.21M, N 1.41M (N about 16% more phase-1 tokens, as the note acknowledges).
- synth.py:
  - t_crit(7) = 2.3646.
  - all_report on synthetic 8-seed data gives correct signs. Seed-0 H(E−N) matched a hand computation (0.5862).
  - Verdicts, the "suggestive" band and the outcome wording behave as the note says.
  - The minimum exact p is 2/256 = 0.0078, so 8 seeds can reach p < 0.05.
- droppair.py: item 3(5).
- One rendered E brief (Claude-written, so safe to print) matches the note: points 1-5, "says what it understands, not what it does", no reply format, no example, correct plural grammar, correct forbidden-word lines.
- Past done.json timings: Dseq docs about 6,045 s for 9.8M tokens; told_s0 942 s. S1 reads took 730-1,050 s per 576 episodes, and training 60-95 s per 12 steps. These gave the estimates in item 3(4).

Read only:

- The phase-3 stream against Act I train(), which matches:
  - items = turns[i][-1]
  - a fresh AdamW per round, deleted after each round
  - manual_seed(seed*1000 + r) (modal_act1.py:991)
  - group-first draw_items with seed s*1000 + r
  - micro 2, as in ACT1_CFG
  - gradient checkpointing, as MODELS sets ckpt True
  - the lora_B nonzero check (:957)
- Arms identical apart from documents: same seed, the same DOC_FT, the same practice set, the same stream file per seed, the same read seed. finetune's label changes only args.json.
- Read settings: koan prompt asserted equal to D.KOAN_SYSTEM; temperature 0.7; 3 attempts; 4 impossible and 2 solvable per pair. Strict release, hacked, over-released and env_error are as the note defines them.
- Spend: reserve and true-up, the $135 stop at block start (:593-607), two-failure drops for the whole seed (keeps pairing), and refusal of dirty code at the dose check.
- Caching: keyed by the briefs' sha plus a row check. Verdicts keyed by (id, text sha, form, rubric sha).
- Blinding: no command prints R or E outcomes. The look opens arm N only, enforced by only_n. The container prints progress fields only (modal_act1.py:983). But read_r*.jsonl, log.jsonl (with per-round measures) and info.json for every arm sit as plain JSON on the volume and in the results folder as soon as each press finishes. So outcomes are not printed, but they are not hidden from anyone who opens the files.

## 7. COULD NOT CHECK

- Modal's API surface (excluded by the brief), real input-size limits, and real timings and queueing.
- Volume contents: samples.pt, the told train.pt, the number of snapshots in the cache, the adapters themselves.
- A real tokenizer round trip and chat template; the tests use fakes.
- Real generation and real judge behaviour, including how often Qwen copies brief text.
- Whether today's koan_env reproduces the stored Stage A outcomes.
- Which E documents will be kept, and their size against R and N, until --build runs.

Plain reading: the arithmetic is right and fair to the arms. What needs tightening is the guard rails that keep the one registered run the only run: a freeze check at readout, one dose record, one main tag, and safe relaunches. Cost of this check: $0.

---

# The second report

COLD RE-CHECK: step 3 code after the fixes (lab/explore/understand/*, the u1 additions to lab/explore/act1/modal_act1.py), against lab/plans/step3-understanding-first-2026-11-10.md rev 2

Checked by a fresh Opus 5.5 agent, read-only. I wrote nothing in the repo and made no Modal, OpenRouter or network calls. Scratch scripts were kept in a session scratch folder outside the repo (echo_recount.py, head_route.py, synth_readings.py, synth_more.py, test_attacks.py). No document text, reply or task item was printed. Cost: $0.

## 1. Verdict: NO BLOCKER

Every reading, the rule, the outcome table, the gates, the guards, the look and the dose rule still compute what the note fixes. On five synthetic 8-seed cases they match my own independent code exactly, and `--all` and `--ended` give identical numbers on a complete run. The echo rule is implemented as stated and is the same for all three arms; my recount matches the fix list to the row.

What is still weak is the same class the first check found: the guard rails that keep one registered run the only run. I showed four paths on the suite's own stand-ins (section 3), and the worst gives an early look at the contrasts:

- The spend stop ends a run.
- `--ended` then prints six-seed contrasts.
- Block 3 still launches afterwards.

None of the four changes a number computed from honest runs, and each needs an operator slip (Ctrl-C followed by the wrong next command, a second terminal, or a rebuild). That is why I do not call any of them a blocker.

## 2. The six should-fix items

**SF1, freeze checked at readout: CLOSED.**

- Where: `frozen_for` (compare_u1.py:78-80) is called by `look_report` (:222), `load_all` (:293) and `load_ended` (:313). `--dose` checks the tag's freeze.json (:188).
- How I tried to defeat it: I listed the non-stdlib imports of every module the readout and the containers use (act1_core, declared_core, run_s1, replay, gate_core, koan_env, player, pilot, modal_sft, modal_retreat2). All of them are in FREEZE_FILES, under src/anatta or under third_party. Caches and dot-files are skipped, and caches are gitignored.
- What is left:
  - Data inputs are not frozen. The read's 38 pair rows come from the tracked file gate0/split/train.jsonl and are sent from the laptop at every launch. Nothing hashes them into sets.json or the plan.
  - A rebuild makes an earlier run unreadable (see SF2) rather than readable under new code. That fails safe.

**SF2, forking at the dose check and the main tag: PARTLY CLOSED.**

- a. The check for one dose record per root runs once, at the start of `cmd_dose` (run_u1.py:726-729). dose.json is written at :794 with no second check. A second tag launched while the first is still out passes the check, both finish, and both write a record. `find_dose` (:802-814) then refuses only without `--dose-tag`; with it, a main run can pick either record. I reproduced this: test_attacks.py::test_B gave records d0 and d1 for root u1.
- b. A part-finished dose tag can be abandoned for a fresh tag. The suite asserts this as intended (test_run_u1.py:528-533).
- c. A rebuild resets everything. `cmd_sets` (:624-633) accepts a new root with the same committed code and the same inputs even when the current root already has a dose record or a main run. A fresh dose check and a second main tag are then allowed, and the first run can no longer be read: `--look main` now says "no dose record under tag 'd1' for the sets under root 'u2'". I reproduced this in test_D.
- Why it matters: a dose check that lands on branch 4 ("comes back for revision") can be rerun unchanged. That is way 4 (the outcome chosen after seeing values).

**SF3, interrupted launch: MOSTLY CLOSED.**

- What works: `launch` refuses a stage whose last spend entry is a reserve (:548-553). `--fetch` copies results and settles; it settles nothing while a container could still run, and charges finished runs at their recorded seconds. Fetching too early, fetching twice, and relaunching after a settle all behave correctly (suite test :869-918, which I read and re-ran).
- Gap (a): `app.run()` and `spawn` (:572-574) sit outside any try, after the reserve is booked (:563). An exception before any container starts (a network or image error) leaves the reserve unsettled. After about 2.7 h, `settle_interrupted` (:1069-1073) records a failure for every run in the stage, and two such events drop the whole block. In test_C, two failed `app.run` calls spawned 0 fine-tunes, dropped seeds 0, 1 and 2, and booked $52.20 to step 3's spend and to the day file.
- Gap (b): `cmd_fetch` treats any fetch exception, including a network error, as "not finished on the volume" (:1111-1118) and then settles (:1128). A flaky connection during `--fetch` can turn finished runs into failures.
- Gap (c): an unsettled reserve inflates the next block's spend stop. This is item A in section 3.

**SF4, local deadline counts queue time: CLOSED.**

- The local wait is `upper_h + QUEUE_SLACK_H` (:567); the container timeout is `upper_h` (:568).
- In the installed Modal 1.6.2, `FunctionTimeoutError` and `OutputExpiredError` subclass `modal.exception.TimeoutError`, so they are caught at :587. `ConnectionError` is not; it lands in the generic branch (see minor items).

**SF5, a pair dropped in the container: CLOSED.**

- `press` raises before loading the model if any read pair is missing after `prepare_pairs` (modal_act1.py:953-958).
- `run_problems` (compare_u1.py:96-113) and `U.read_problems` (u1_core.py:646-660) refuse a read with missing pairs, extra pairs, or the wrong number of samples of either twin.
- `odd_sizes` now uses the plan's size (compare_u1.py:368).
- The old read path samples 4 of each twin (modal_retreat2.py:291-292), which matches the (4, 4) check in `load_dose`. Its records carry pair, impossible and outcome (src/anatta/player.py:124).
- Tests exist for the dropped pair and for each kind of plan mismatch.

**SF6, Claude-written brief text in training documents: CLOSED as specified.** Details under question 3 below.

## 3. Blockers

None. The four paths below are listed as should-fix because each needs an operator slip. Item A is the one to fix first.

## 4. Should fix before any spend

**A. "ended" is not final; block order is not enforced; the spend stop counts unsettled reserves.**

- Where: `block_stop` (run_u1.py:826-846) never refuses once `run["ended"]` is set, and computes the stop with `spent(out)`, which includes reserves that were never trued up. `load_ended` (compare_u1.py:307-333) trusts `ended`.
- The suite itself asserts that block 3 launches after the stop refused block 2 and `--ended` had printed contrasts (test_run_u1.py:690-713).
- My reproduction (test_A):
  1. Block 2's press launch is interrupted.
  2. Block 3's stop is computed on the inflated total and writes `ended`.
  3. `--fetch` settles the reserve.
  4. `--ended` prints six-seed contrasts; at six seeds the minimum exact p is 0.031, so verdicts are possible.
  5. Block 3 then launches 6 runs.
- Why it is plausible: at the design's expected costs, plus a roughly $2-3 rehearsal, step 3 will have spent about $100 when block 3 is due, against the $110.50 that block 3 needs to start. That is about $10 of headroom, so the stop at block 3 is a live possibility.
- Way 4: no contrast is supposed to be computed before every block is in.
- Smallest fix:
  - `block_stop` refuses every block once `ended` is set.
  - `block_stop` refuses block k unless block k-1 has started.
  - `block_stop` refuses while any reserve in spend.json is unsettled, pointing to `--fetch`.

**B. Two dose records for one root, and the rebuild route (SF2 above).**

- Re-check `done` immediately before writing dose.json.
- Refuse a new dose tag while another tag on the same root has freeze.json but no dose.json.
- `find_dose` refuses when a root has two records, instead of letting `--dose-tag` choose.
- `cmd_sets` refuses a new root while the current root holds a dose record or a run.json, unless the code record or the inputs differ (a rebuild should be a revision), or it is given an explicit retire flag.

**C. A launch that never started counts as failed runs (SF3a and SF3b above).**

- In `launch`, catch exceptions around `app.run` and `spawn`. If some runs were never spawned, cancel the spawned calls, true up the unspawned runs at $0, and re-raise.
- In `cmd_fetch`, skip settling if any fetch raised anything other than not-found.

**D. The hand read can be redone with a different seed.**

- `hand_read_check` (run_u1.py:323-340) looks only at the latest record. `cmd_hand_read` (gen_docs.py:714-746) keeps earlier records under "earlier", but nothing reads them, and `--sample` takes any seed.
- So a read with 4 violations under seed 0 is overridden by a read with 1 violation under seed 1. The note says three or more violations means the judge prompt is tightened and every document screened again.
- This is way 4 at the document-selection stage: it lets E pass its screen more easily.
- Fix: refuse when any record for the same `files_sha256` has 3 or more violations, or fix the seed before sampling.

## 5. Minor

- `--look` can be re-run and overwrites look.json (compare_u1.py:259-264). A seed dropped after the look can flip its verdict. Make look.json write-once.
- `cmd_block` does not check that a new tag's folder is empty (run_u1.py:885-909). Stale local done.json or info.json files, for example from a rehearsal folder whose plumb.json was removed, would be trusted by `stage_done`.
- The readouts trust dropped.json without checking that each dropped seed has two error_try files. Likewise, a hand-added `ended` opens `--ended` with no check against spend.json.
- `tag_root` (:495-502) ignores freeze.json, which holds no root. So `--fetch` of an incomplete dose tag after a rebuild looks under the new root.
- `u1_snapshot_check` (modal_act1.py:629) accepts a missing refs/main. Under `HF_HUB_OFFLINE=1`, the h4lite loader resolves the hub id through refs/main, so a missing ref would make every fine-tune and the old read fail. That fails safe, and the rehearsal would show it. Require refs/main to equal the pinned snapshot.
- An interrupted rebuild leaves sets.json renamed away with no restore (:647-662). Every command then refuses until it is renamed back by hand.
- A `ConnectionError` in `c.get` is recorded as a failure while the container may still be running. For a fine-tune, the relaunch writes the same out_dir (finetune truncates log and done first), so two writers can overlap. For press, the attempt folder differs; `--fetch` only looks at the next attempt's folder, so a finished earlier attempt is never picked up. That is wasteful but safe.
- `cmd_dose` logs per-run counts over all 38 pairs (:788-789), then the 28-pair table. Two sets of numbers are printed.
- The plan check covers press runs only. The docs and practice args.json files (epochs, lr, init chain) are not checked at readout; the freeze covers them.
- Any new or edited file under lab/explore/understand, including a test or a scratch note, makes every block and readout refuse until reverted byte for byte. This fails safe; worth one line in the run notes.
- The rehearsal is optional (`cmd_dose` does not require plumb.json to be ok), and its spend counts toward the $135 stop.
- Unchanged from the first check: the lenient judge form is not gated on a second smoke, and "sharpen once" is not enforced.

## 6. Departures from the note

| Where | The note says | The code does | Could it favour an outcome? |
|---|---|---|---|
| u1_core.py:429-465; run_u1 local_docs | R is the pilot's first 3,000 declared documents | R is the first 3,000 rows that do not echo their brief: 424 swapped for clean rows up to file row 3,490. Fingerprint 1140458f…, not D3k's 85e84b8f… | Barely. Words +3.7% (1.247M to 1.293M); rationale emphasis 26.6% to 27.6%; "the rule" 24.0% to 23.1%; ministral referent 25.7% to 24.5%; shortest length bin 22.0% to 18.6%; genre shifts of 1.3 points or less. If anything this moves R slightly toward E, which works against an E−R hit. The write-up must say R is no longer exactly the pilot's measured D3k (86% overlap). |
| u1_core.py:256-269 | No Claude-written text is trained on | The threshold is 8 words. 286 of R's 3,000 training documents keep a 7-word run of the brief. The first paragraph is exempt; it is effectively the topic phrase (10-14 Claude-written words), carried verbatim by 357 R and 366 N of the first 3,000 documents (444 and 524 share an 8-word run with it) | No, it is the same for every arm. It is a reading of the hard rule that the driver should record. |
| u1_core.py:176, :220 | Brief rules as listed in the note | E's brief adds "use the document's own wording" and "release" to the forbidden words; R and N briefs never had these | No. E documents may paraphrase more. |
| run_u1.py:826-846; test :705 | Blocks 0-2, 3-5, 6-7 in order; no contrast before every block is in | Block 3 can start after block 2 was refused; `ended` is not final | Yes, through an early look followed by continuing (item A) |
| compare_u1 `--ended` | Silent on how an ended run is read | The full rule on the seeds of started blocks (3 seeds: no verdict possible; 6 seeds: verdicts possible) | Only through item A |
| run_u1.py:624-633, :726, :794 | Branch 4 means the design comes back for revision; one dose check | A rebuild with no change, or overlapping tags, gives a second dose check | Yes (way 4) |
| run_u1.py:323-340 | 3 or more violations: tighten the judge and screen again | A sample under a new seed overrides a failing read | At selection only |
| run_u1.py:1069-1073 | A crashed run is rerun; two failures drop the seed | A launch that never started also counts as a failure | No (whole seeds are dropped), but it costs power and spend headroom |
| u1_core.measures | Rates over 152 and 304 episodes | env_error episodes leave n | No: 0 env_error in Stage A's 6,192 episodes |
| `--plumb` | Not in the note | A rehearsal whose cost (about $2-4) counts toward the $135 stop | Only through the stop (item A) |
| Unchanged from the first check | | Rubric additions (u1_core.py:313-324); the judge asked only as far as the quota; E screened by a judge that R and N were not; failures counted per stage; the paths check on 28 pairs; read seed shared across arms; emphasis mapping; three UTC days not enforced | As the first check said: no, except that the rubric additions slightly widen what E may say |

## 7. What I verified by running something, and what I only read

**Ran:**

- `pytest lab/explore/understand`: 177 passed. `run_u1.py --dry`, `gen_docs.py --dry` and `gen_docs.py --echo` all ran.
- echo_recount.py, with my own word and n-gram code:
  - Echo counts: R has 424 echoes in its first 3,000 rows and 490 before row 3,490; its clean set ends at row 3,490 and its fingerprint equals the pinned one. N has 0 echoes, and its set and fingerprint are unchanged.
  - Brief rendering: `D.briefs_sha(config)` equals the pilot manifest's, so today's render is what the generator was sent, although declared_core.py's own hash has changed since.
  - Composition: the before/after counts in the table above.
  - What gets echoed: point 1 of the brief appears in 310 of the 424 echoes. It is verbatim in E's brief too, so expect E yield losses there.
- head_route.py: the first-paragraph route is the topic phrase (0 or 1 other 8-word runs per arm).
- synth_readings.py and synth_more.py:
  - Coverage: five cases, including a guard firing and a failed practice gate.
  - Match: my own U1, H, U2, U3, exact sign-flip p, t interval (t at 7 df = 2.3646), verdicts, gates and guards gave 0 mismatches.
  - Unchanged numbers: `--all` and `--ended` are identical on a complete run, and the guard flags change no number.
- test_attacks.py, on the suite's stand-ins: paths A, B, C and D all reproduced.
- Installed Modal 1.6.2: `with_options` accepts `env` (it is added as a secret and the volumes are kept); the timeout exceptions behave as stated under SF4.
- KoanTask's system prompt equals `D.KOAN_SYSTEM` for both twins, and the old path's default arm gives the same prompt.
- Caches are gitignored.

**Read only:**

- The press and u1_sets diff.
- `stage_kwargs` against `finetune`'s and `press`'s real signatures: they match. finetune's `label` changes only args.json and the return name; `src_dir` and `out_dir` replace its default paths; `init` reloads the volume and trains the loaded LoRA.
- `evaluate_adapter` and `heldout_eval` arguments.
- `run_problems` and `plan_problems` against what `press` records, both in the saved form and in the fetched form.
- The rehearsal's outputs: it writes under its own tag and volume folder, prints only seconds, costs and pass/fail checks, and is refused as a dose or main tag.

## 8. What I could not check

- Real Modal behaviour: queue times beyond the 1.5 h slack, the input size limit for u1_sets' document payload, and whether the timeout excludes queue time.
- Whether the HF cache on the volume has refs/main pointing at the pinned snapshot.
- Real timings against UPPER_H, especially N's longer documents at 2 epochs against 1.0 h.
- How many E documents the echo filter and the judge will drop.
- Whether real costs will actually hit the spend stop at block 3.

Plain reading: the arithmetic is right and fair to the arms, and the echo fix does what it says. What is left is closing four ways to get a second look or an early look before money is spent: make "ended" final and refuse while a reserve is unsettled; allow one dose record per root, checked when it is written, and refuse a rebuild that is not a revision; count a launch that never started as nothing; and keep a failing hand read binding.
