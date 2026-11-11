# Step 3, the code change for revision 3: Opus's cold check (11 Nov 2026)

A fresh agent (model opus, read-only, no Modal, network or paid calls) checked the uncommitted code change that
follows the design's third revision (`lab/plans/step3-understanding-first-2026-11-10.md`, "Revision 3"), against HEAD
34da83a, the code as it was when the first dose check ran. It was given the note, the earlier check that shaped the
revision, the diff and eight questions. It was not told what Fable thought of the change. Its report follows, filed
by Fable (subagents can't write report files). The substance is unedited; one line is changed, where a local scratch
path is replaced by a plain statement. The code with the fixes is in the commit that adds this file.

What Fable did about it, before the second dose check (the fixes were read by Fable and not checked cold):

- **Should-fix (a), the rule arm could join without a raise (taken).** Arm R joins a run only when a raise of the stop
  is on record, at a tag's first launch or through `--add-arm`. A run must also fit under the stop as a whole when it
  starts, and an arm that cannot be finished under the stop is not added. An added arm is written to variants.jsonl.
- **Should-fix (b), a run without E (taken).** A main run has N and E; R is the only optional arm.
- **Minor, taken:** the note that H is not neutral to the start is attached to every contrast; a build that goes ahead
  only because the brief and judge are unchanged cannot make a paid call; a zero start fails the dose check; the
  rehearsal's shape (six steps, reads at steps 3 and 6) is recorded in the note.
- **Minor, not taken:** the guards rest on files existing (a readout computed by hand leaves no trace; the ledger is
  the record); an arm added after the look is allowed (the look opens arm N only), and a failed run of R still drops
  its seed from every contrast, as the note's rule says; the dose check's reserve stays 1.3 hours a run; training
  on a GPU is not bit-reproducible, so "unchanged" means the same computation.
- **Its free check after the second dose check (taken as advised, reported and not gated):** the step-0 reads should
  replay the first dose check's round-0 reads.

---

COLD CHECK: step 3, Revision 3 code change (13 files, uncommitted, HEAD 34da83a). Written by Claude (Opus 5.5), a read-only cold checker. Cost $0: no Modal, no OpenRouter, no network.

## 1. VERDICT

NO BLOCKER. I tested the training path, reads taken inside rounds, the cut last round, the dose rule, the two-path gate, the readouts and the documents, and all of them conform to Revision 3. Point 7 (arms) does not. The rule arm R can join a run without the budget being raised, either at the first launch (`--arms N,R,E`) or through `--add-arm R`, and the $135 stop then ends the run at six seeds. The note rules that out. No path lets a contrast be seen before every arm is in. Fix point 7 before the dose check, because the dose check freezes the code. The best time is before the free manifest rebuild, so that nothing has to be redone.

## 2. BLOCKERS

None.

## 3. SHOULD FIX BEFORE ANY SPEND (or at the latest before --dose, which freezes the code)

(a) R joins with no raise, and the stop check undercounts. Ran: three scratch tests on the repo's own fixtures.

- At the first launch: lab/explore/understand/run_u1.py cmd_block 1148-1156 with parse_arms 1108-1116. `--arms N,R,E` under the design's $135 needs no stop.json. With $48.23 spent (the $39.23 plus the ~$9 redo), blocks 1 and 2 run, block 3 is refused ("would pass the stop"), and run.json is marked ended at block 3. `compare_u1 --ended` then gives the primary H on n=6. The note says "three arms at seven seeds is not run"; this path runs three arms at six.
- Through add-arm: cmd_add_arm 1250-1257 checks only `spent + arm_expected_usd(seeds of started blocks)`. It ignores the N and E runs that started blocks still owe, and the blocks not yet started. Example: spent set so that R's six runs just fit; block 2's N,E practice and press are still owed ($14.88); projected total $149.87 against $135, with no raise recorded. Because block_stop (line 1058) skips blocks already started, all of those launches still go ahead. Block 3 is then refused at three arms, which again leaves n=6.
- Smallest fix:
  - In cmd_block, when a run is created with R, and in cmd_add_arm: refuse unless spend_stop() reports a raise (stop.json).
  - In both places, check the whole run: spent + every run still owed (started blocks' remaining stages + all unstarted blocks, at the new arm count) ≤ stop.

(b) A run without E is allowed. u1_core.py run_arms 156-161 requires only N plus one more arm, so `--arms N,R` is accepted (ran). Revision 3: "the main run is the neutral and explained arms." Fix: require both N and E.

- The tests encode the looser rule and need changing with the fix:
  - test_run_u1.py: test_a_runs_arms_are_chosen_at_its_first_launch accepts E,R,N and N,R with no raise. The test that adds an arm adds R at stop 135 with no stop_raised.
  - The `block()` helper launches all three arms by default, so most launcher tests exercise a configuration that is not the default. Add one end-to-end N,E test: 8 seeds, under $135, all three blocks.
- Order: u1_core.py is in the docs manifest's code hash (DOCS_CODE), and sets.json, plumb.json and freeze.json hash the code. A fix after the manifest refresh needs another free rebuild. A fix after --sets needs new sets and a new plumb. A fix after --dose is refused by the freeze.

## 4. MINOR

- start_note too narrow. u1_core.py 1073-1075 attaches START_NOTE only when U1 reads higher, lower or suggestive. The note says H is always "worded with the start in view". Using the code's own seed_values with both arms ending at 6/304:
  - a 45% vs 55% start moves H by 0.42;
  - a 43% vs 57% start moves H by 0.58.
  - A U1 verdict of null could leave such a start gap unflagged. The per-seed drop in points and the share retained are printed (compare_u1.py 458-463), which helps.
  - Fix: always attach the note. (I reproduced the note's −0.62 and −0.25 exactly.)
- same_screen_reason (gen_docs.py 643-660, used at 666) checks only the briefs hash and the judge's form and rubric hash. If the selection ever needed new generations or verdicts, a build could spend up to the remaining caps ($3 generation, $28 judge) without a passing smoke. Today it makes no paid call (simulated, see section 6). Fix: in same-screen mode, pass chat and judge functions that raise, so the build can only reselect from its cache.
- The guards are based on files existing:
  - add-arm's "before any readout" guard only checks that compare_u1.json or compare_u1_ended.json exists (run_u1.py 1237-1240); the dose-record checks work the same way.
  - A readout computed in a Python session, or a readout file moved by hand, leaves no trace.
  - Adding an arm is recorded in run.json "arms_added" but not in an append-only log such as variants.jsonl.
- Adding an arm after the block-one look is allowed. The look opens only N (compare_u1.py 229-251), so it cannot show a contrast. But R's failed runs drop seeds from the whole readout, including E−N, so adding R adds a way to lose primary seeds. R's runs may also land on later days than N's and E's.
- A zero start is not refused. Pooled step-0 release of 0 gives "step 3, coarse" and reads [3, 6]. Very unlikely given d1 (109/224), but refusing would cost nothing.
- Optimizer memory: the AdamW state and the cleared gradients stay in memory during reads inside a round. For LoRA this is small, but the old code had no optimizer alive during its reads. Small risk of running out of memory.
- UPPER_H["dose"] went from 1.5 to 1.3 h. From 12-15 min a read × 4 reads plus 12 steps and loading, I estimate about 900-1000 s of margin. A slow read could hit the timeout (one lost run, about $2 plus the re-launch).
- PLUMB now has steps 6 and reads [3, 6], which differs from the addendum's text. It is a better rehearsal: it exercises a read inside a round and a cut round. It is a departure only on paper.
- Already in HEAD: rebuilding the sets under a new root allows a new dose record and a new main tag, so another run with R could later run on the same seeds. It costs money and shows in the ledger.
- Pointing the new code at d1 is refused by the freeze. `--fetch d1` probes old_told_s1 without harm.

## 5. DEPARTURES FROM REVISION 3

| Where | The note says | The code does | Could it favour an outcome? |
|---|---|---|---|
| Pt 1: u1_core paths_check 794-804, DOSE_PATHS=15 | Band removed; both adapters' two paths agree within 15 points on the 28 pairs; the level gates nothing | The same, with exact fractions, inclusive at 15; old path = modal_retreat2.evaluate_adapter (seed 3000, n 4); level only reported. Recorded facts recount correctly: (165,256) on 30 Oct; Stage A 0/448 and 17/160 | No (computed on the told adapters only) |
| Pt 2: expert_iteration.sft_steps, modal_act1.press | Round = 12 steps × 16, one draw of 192, one optimizer | Unchanged. A cut round trains on the first owed×16 items of the same draw (draws are i.i.d. and group-first) | No |
| Pt 3: press reader hook | Read after the k-th step, optimizer untouched, seed 10000+k; step 0 = old round 0 | Matches: zero_grad, eval, save RNG, read, restore, train. read_s<step>.jsonl. The seed at step 0 equals the old round-0 seed | No (see section 6 for what this is proven on) |
| Pt 4: run_u1 cmd_dose | Old path on both adapters, then 12 gameable steps read at 0/3/6/12; no solve-only | Matches | No |
| Pt 5: u1_core dose_rule 765 | First of 3/6/12 at or below half, pooled on the 28 → [3,6] (coarse at 3), [6,12] or [12,24] | Matches on 11 awkward cases (exactly half, a dip, a zero start, adapters far apart, a missing read, an old-path read for one adapter only) | No |
| Pt 6: compare_u1 | Train to the later step; read at 0 + two steps; readings and gates unchanged | Matches. A two-arm E−N gives the same numbers as HEAD's three-arm E−N | No |
| Pt 7: run_u1 cmd_block / cmd_add_arm, u1_core run_arms | N,E; R only if the budget is raised and only before any readout; no 3 arms at reduced seeds | Defaults to N,E and refuses after a readout, but R joins with no raise, the stop then ends the run at 6 seeds, and N,R is allowed | Not toward a sign. It cuts primary power (n 8→6) and makes the arm set a choice the driver can make mid-run, after the look (which shows N only) |
| Pt 8: EXPECTED_USD, SPEND_STOP | $1.63 / $0.48 / ~$2.00; stop $135 unless raised | Matches; the dry run says $65.76 for three blocks at N,E, +$32.88 for R | No |
| Limits: start_note | H always worded with the start in view | Note only on some U1 verdicts | Could flatter "wears down faster" when the starts differ but U1 says null |
| gen_docs same_screen_reason (not in the note) | A smoke gate comes before the build | The build may skip the smoke when the brief and judge are unchanged | Not on the outcome; a spend risk only, zero today |
| PLUMB, UPPER_H | Addendum's rehearsal shape | steps 6 with reads [3,6]; dose reserve 1.3 h | No |

## 6. WHAT I VERIFIED BY RUNNING SOMETHING VERSUS ONLY READ

Ran:

- The repo's free test suite: 261 passed, 1 warning.
- `run_u1.py --dry`: arms N,E; seeds 0-7; reads [0,3,6] in the rehearsal; dose reserve $11.15; blocks $24.66, $24.66 and $16.44; spent $39.23; d1 the only dose record.
- Bit-identity tests (test_organisms.py, test_u1_modal.py). What they prove, on CPU with tiny models:
  - final weights are identical with or without reads inside rounds;
  - a cut round equals the first k steps of the full round;
  - restoring the RNG matters when dropout is on (TinyTrainable, dropout 0.2);
  - the step-0 read's bytes do not depend on the plan or the step count.
- The dose rule on 11 adversarial cases, all matching the note.
- Recounted from local files: both recorded facts (scratch facts.py).
- Two-arm E−N under the new code against HEAD's three-arm E−N on the same synthetic episodes: identical values, gates, guards, held NLL, paired test and bootstrap (twoarm.py). Only the wording differs.
- Document fingerprints at HEAD and now: briefs hash, judge forms and rubric hashes, the E filters, select_e, the flag functions and the smoke constants are identical and match the manifest (fingerprints.py).
- An offline `--build` on a scratch copy of the docs folder, with paid calls stubbed to raise (build_sim.py):
  - same_screen_reason let it proceed with zero paid calls;
  - the files were byte-identical;
  - the manifest's code hash was refreshed to the current code;
  - local_docs and hand_read_check accept the result.
- The arms and stop problems in section 3 (test_arms_scratch.py, all three pass as described).
- The H start-dependence numbers in section 4.
- The repo's git status is unchanged (13 files) and the results folder is untouched.

Read only:

- The chain gen_docs --build → commit → --sets --root u2 → --plumb NEWTAG → --dose NEWTAG → --block goes through. find_dose works per root, so d1 under u1 does not interfere, and one dose record per root is enforced (run_u1.py 982-999, 846-858).
- Where the CUDA RNG is saved and restored.
- Whether PEFT, HF generate or gradient checkpointing have side effects. There is no weight merging in the read path.
- The five ways a result can be invalid. No instance found:
  - copied: the two constants are recounted and gate nothing;
  - substituted instrument: read and judge paths unchanged;
  - altered operand: documents unchanged, and the cut round is a subsample of the same draw;
  - outcome chosen after seeing values: the band removal is owned in the note and uses the told adapters only; the add-arm channel is in section 4;
  - wrong statistic: E−N identical.

Scratch scripts were kept in a session scratch folder outside the repo.

## 7. WHAT I COULD NOT CHECK

- Training identity on the real GPU. GPU training is not bit-reproducible even without reads, so on Modal "unchanged" means the same computation, not the same bits. The CUDA RNG branch never runs on the Mac. PEFT, HF generate and gradient-checkpointing internals I read but did not run.
- Byte identity of the step-0 read against d1's round-0 read on the GPU. A free check after d2: d2's step-0 reads should replay d1's new-path round-0 counts (told_s0 60/112, told_s1 49/112) if "a read replays exactly". I suggest reporting this, not gating on it.
- Modal volume behaviour (commits during reads, copy_from_volume) and real timings.
- The real `--build`: simulated only.
- Whether test_gen_docs covers a same-screen build that would need a paid call.
