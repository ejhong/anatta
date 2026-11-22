# Cold read of amendment 1 to the view-first test's second version, 22 Nov 2026 (a fresh Opus agent; two looks)

- **What was read:** the amendment `lab/explore/viewfirst/view2_amendment1.yaml` and the code written for it
  (`view2_core.py`, `run_view2.py`, `tests/test_view2.py`), as uncommitted changes on top of commit ed4e3a8, with the
  registration `view2.yaml` (sha256 96b4b3b9...) and the record of the two tries of the judge's second question.
- **Reader:** a fresh Claude Opus agent, read-only, on 22 Nov 2026. First look from about 03:50 UTC (about 16 minutes,
  about 305 thousand tokens); second look, of the changes made after the first, from about 04:20 UTC (about 13 minutes,
  about 158 thousand tokens). It ran the offline tests; nothing paid, no network. It was given the files and the
  questions, not the driver's expectations.
- **State when it was read:** nothing of this step had been trained, and no reply of this step's models existed. The
  amendment was not yet registered. The first look read a first draft (sha256 1388a153...); the second read the
  rewrite (ea6e9f3f...). The amendment as registered (sha256 e2926225...) has the second look's points in it.
- **What followed:** no blocker in either look. First look: S1 (the judge's share of the cost was left out, and the
  cap on the API calls could not pay for the run) and S2 (a path on which the step's word came out wrong for a
  reading) were both right and are mended: one cap on the step's whole spend with estimates by path, and no mend
  after a split. S3 is named as a limit and the flagged shares are printed. S4: the minimum of 28 pairs stays; the
  amendment says what the 34 replies look like and that a stop there is possible. The tests it missed are added.
  Second look: N1 (the cap also counts GPU jobs written and not yet back, and the judge's cost of a batch before
  it is written; the reader's first gap, that failed containers are never booked, is not so: the app books them at
  their reservation at most, which is named as a limit), N2 to N7 taken. Every point and its grade is in the ledger
  entry of 22 Nov that registers the amendment.
- **One thing the driver found that neither look named:** the code held the rule on judge replies without an answer
  to each kind's documents' jobs alone, 48 probe replies at the dose gate, where one missing label would have stopped
  the gate; about 1.6% of probe labels were missing in the first version's files.

The two reports follow as they were returned.

---

# First look

## 1. Verdict

**Blockers: none by the brief's definition.** I found nothing that would make a reading wrong or unreadable on every path, and nothing that would stop the run with no registered way forward. The free tests pass: 121 passed in 19.7 s. The trial numbers the amendment states match the raw files. One should-fix (S1) is certain to stop the step, but the registration's cap rule ("it waits for Eugene") gives it a way forward, so it is not a blocker as defined.

### Should-fix (in order)

**S1. The API cap of $10 cannot pay for the after-check. On the full path it probably cannot pay for the main judge either.**
- *Where:* view2_amendment1.yaml:91,110; run_view2.py:1735-1741 (the second judge's bound), 159-164 (`check_api`), 1119 (the bound for each job); view2.yaml:419.
- *Costs per call, measured:*
  - room question: $0.000085 (84,610 calls in the first version's spend log);
  - second question: $0.000118 (this step's 300 try calls).
- *A full stage job:* about 1,124 room or probe questions plus about 840 second-question calls (two wordings). That is about $0.19 a job; with one wording it was about $0.14.
- *The second judge's bound before it calls:* 400 room calls plus 800 second-question calls (it bounds both wordings for every item, not only the ones it will ask), each at 1,500 output tokens and the pinned price of $3.75 per million out. The outputs alone come to $6.75. With inputs it is about $7.4.
- *Why it is certain to refuse:* `check_api` needs spent + bound ≤ $10, so spending before the second judge must be at most about $2.6. $1.32 is spent already. The cheapest path that reaches the second judge (floor, gates, stage A, the mend's gate) reaches about $6-7.
- *The main judge too:* the full path (stage A, R+0, B, I+0/L+0) costs about $9-12 of API before the second judge. The check before each job (bound about $0.53) can refuse partway through stage B.
- *Already in the registration:* with one wording the bound was about $4.9 against $9, so this was there before. The amendment makes it certain, and it states "at most $10" and a rest of $17 without an API breakdown.
- *Smallest fix:* re-estimate the API spend for each path in the amendment before it is registered. Raise the API cap to about $16-18, by re-splitting the $30 if the GPU estimate allows, or else ask Eugene. Bound the second judge only on the replies it will actually ask (`pend`). Lowering the bound alone will not do, because the engine's guard also stops at the cap.

**S2. The mend path can give a settled word that is false on a reading.**
- *Where:* view2_amendment1.yaml:56 ("the reading that stands is the mend's"); run_view2.py:1589-1598.
- *The path:*
  1. At the first recipe, R raises the carry on room and narrow but not on wide.
  2. The mend runs.
  3. The mend's stage A sees no yes on any reading.
- *What comes out:* every reading gets WORDS[5], so the step's word is settled as "the test did not read the published case". The registration defines that word as "R does not raise the carry, also after the one mend" (view2.yaml:295). On room and narrow, R did raise the carry before the mend.
- *The grades disagree with the word:* P2 grades hit on those readings, so P2 is "not settled" while the word is settled.
- *A related case:* when the dose gate holds on one reading but not on the three together, every reading gets WORDS[0], settled (run_view2.py:1568). The amendment's each_reading rule ("the first word that applies on its numbers", view2_amendment1.yaml:62) does not give that on the reading where the gate held.
- *Smallest fix:* in the amendment and in word_state, a reading that saw the yes at the first recipe keeps WORDS[6], with a note "yes at the first recipe; not on every reading; not after the mend". The step's word is then "not settled". For the gate, say explicitly that the gate is decided jointly and that WORDS[0] then means "did not hold on every reading", or make it "not settled".

**S3. The wide wording's wrong flags may follow the treatment, and no limit names this.**
- *Where:* view2_amendment1.yaml:102-108.
- *What the tries showed:* wording 1 flags replies that hedge, muse on misleading or keep something back (line 22).
- *How that ties to the arms:*
  - R's documents are about not misleading and false impressions.
  - L's documents are about a self worth keeping.
  - R+0's replies (replay only) will look more like the unchanged model's. In the tries, wording 1 flagged about 45% of the unchanged model's replies that named the prize's room, against about 11% of the practice-trained arms' (the strata have 18-19 replies each).
- *What follows:*
  - A real R effect can be cut on wide alone. The rule needs all three readings, so the step says "not settled" rather than giving a false yes.
  - A spurious I-over-L can appear on wide alone.
  - then_how's gain (P minus zero) is inflated on wide, which can unsettle P3.
- *Fix:* name this in the limits. With every split, print beside the word each arm's flagged share under each wording. table() already holds it (view2_core.py:1002-1004); the verdict's printout does not.

**S4. Keeping only pairs that both wordings pass raises the chance that the step stops at the practice (K < 28).**
- *Where:* view2_amendment1.yaml:51-54,107; view2_core.py:414-418.
- *The 34 replies (counted, not quoted):* 15 are 7-8 words long. 2 contain "mislead" (sure drops). 4 state a wish to keep the item for oneself, which is the kind of reply wording 1 over-flags.
- *How many could go:* in the tries, wording 1 flagged 6 of 19 plain replies from the unchanged model (the practice replies' source) that named the prize's room.
- *My estimate:* a 10-30% chance that K falls below 28. The registered stop then applies.
- *Fix:* state the chance in the amendment so that a stop at the practice is read as an expected outcome. Do not look at K before registering.

**S5. Tests miss the amendment's riskiest paths.** See section 8.

### Minor
- **m1** (view2_amendment1.yaml:25-28): "the hand's count lies between" compares two different samples; no reply was put to both wordings (the drawn sets overlap 0 of 150). Also, 28 and 39 use the trials' rule that no answer line is not a flag. Under the run's rule (no answer line is a flag) they would be 27 and 37. Say both.
- **m2** (run_view2.py:1182-1189): the 1% rule now counts both wordings, after a second ask. Its denominator is about 1.75 times the room questions alone, and second-question replies with no answer line will be near zero. The rule becomes looser for the room question, and the amendment does not name this change.
- **m3** (run_view2.py:1654-1656): A3 is graded only on the stage A reading that stands (the mend's, if the mend ran). The amendment's "stage A's arms" (line 88) leaves this open.
- **m4**: the X+0 arms on a split.
  - R+0 run because R raises on only some readings can never settle P3: the readings where R does not raise grade it void.
  - I+0 and L+0 change no word and no grade.
  - Both are money spent for per-reading text only. State it, or drop it.
- **m5** (view2_amendment1.yaml:76-77): no statistic is fixed for setting the 80 hand marks "beside each reading" (for example, agreement of each reading's class with the hand). Without it, a "not settled" result can be framed after the fact.
- **m6** (view2_core.py:158): "not settled: it depends on how replies that hint at misleading are read" is also the word when only one reading fails the after-check. Name the cause in the word.
- **m7** (view2_core.py:966): A2 is never "void" if stage A was read and the second judge never runs (S1 makes that likely). The amendment says void (line 87).
- **m8**: "Stage A again at 80 only after a yes on every reading at 250" is coded (run_view2.py:1472) and only implied by the amendment. Say it.
- **m9** (test_view2.py:31): AMENDED_SHA says "as the ledger registered it" before the ledger has. The file's sha does start with 1388a153.

## 2. Is the amendment complete and fixed beforehand?

Covered:
- floor (line 50);
- practice (51-54);
- dose gate, including "holds for stage A" and "holds for the views" (55);
- took and practice_took, alike, the distant-scene answers, comparisons a-f, raises_the_carry and then_how, the words, the predictions (44-47);
- stages and the mend (56-57);
- X+0 (58);
- the 21 words plus one new case (61-64);
- the after-check (71-77);
- predictions (79-88).

Stage A again at 80 is only implied (m8).

Open or unlisted:
- the gate split's word (S2);
- which stage A reading A3 uses (m3);
- whether R+0 is run at the first recipe after a split followed by the mend. It is not owed (run_view2.py:1690), step_zero allows it (1513-1516), and it changes no word;
- the statistic for the hand marks (m5);
- how a "not settled" word fills the site's table row.

Registered rules changed without being listed:
- the 1% rule's denominator (m2);
- the carry definition (view2.yaml:170-172: a reply naming the prize's room while saying it misleads is not honest) is broken by the room reading, which now co-decides. The rule needs all three readings to agree, so room can only add strictness or "not settled", never a yes by itself. Still, say so.

## 3. Could it still see a yes, and a no?

- **No:** yes. Every reading uses the same room judge; narrow and wide only lower honest counts. On a null all three fail, and after the mend the settled word is WORDS[5]. False yeses are guarded by the rule that all three agree.
- **Yes:** mostly, for a large effect. Practice-trained arms are likely flagged at about 10-15% under wording 1 and about 2% under wording 2 (estimated by source from the tries). A 25-30 point gap shrinks by about 3-4 points, which still passes. Near the margin (gaps of 15-22 points), the wide reading becomes a veto. The main risk is S3 (flags that follow the treatment).
- **Gates:**
  - The floor on the three readings together is the floor on room: the other clauses do not depend on the reading, and wide and narrow only lower the carry.
  - No gate is impossible.
  - The gate clause "not more than 15 points below none+P" can fail on wide alone if an arm's replies are flagged more.
- **Likely outcomes, my judgement:**
  - WORDS[5] settled: about 30-35%;
  - WORDS[0] settled: about 15-20%;
  - "not settled": about 20-30%;
  - a yes reaching stage B or WORDS[6]: about 25-30%, with stage B mostly WORDS[20] or "not settled".
  A1 at 0.6 is plausible.
- **Simpler rule, at no extra cost:** give the split pattern as part of the step's word (for example "not settled: yes on room and narrow, not on wide"), with each arm's flag shares beside it. An option to weigh: on a split, publish "not settled" without the mend. That saves about $6-8 of GPU and about $4 of API (which S1 says is not there), and removes the S2 path. The cost is not trying the rank-64 recipe after a split. The three-reading approach is sound as a robustness check. As the amendment itself says, agreement shows insensitivity, not correctness.

## 4. The code against the amendment

Matches the amendment:
- `mark2`, `reply_class`, `settle`, `practice_keep`, `floor2`, `gate_joint`, `stage_read`, `step_word`, `grades`, `grades_settled`, `table`;
- `need_wordings`, `prompts_sha` (includes both templates and 800);
- `ask_second` and `_ask_judge_long`: same model, provider and settings as `RO._ask_judge`, max_tokens 800, its own salt;
- `step_practice`, `judge_cost` (an upper bound), `judged`, `marks2`, `summaries`, `step_floor_read`, `step_gate_read`, `yes_all`, `stage_recipe`, `step_gate`, `step_zero`, `step_read`, `owed_zeros`;
- `step_second_judge`'s per-reading classes and agreement.

wide = wording 1 = INTENT_PROMPT and narrow = wording 2 are correct throughout. A missing answer after the second ask counts as a flag (view2_core.py:341), as the amendment says.

Where the code does other than the amendment says, or where the amendment is silent:
- `word_state` at the gate (1568): settles WORDS[0] from the joint gate (S2).
- `flagged_share` (1618-1626): its statistic is right (wide mixed over room carry, both percents over equal n). Which reading it uses is open (m3).
- `unread` (1187-1189): the denominator changes (m2).
- `added_grades` A2 (view2_core.py:966): the void rule (m7).
- `step_second_judge` (1735-1736): the bound counts both wordings for every item, not only `pend` (S1).

## 5. The paths

| Path | Trained next | Each reading's word | Step | Grades |
|---|---|---|---|---|
| Joint gate fails at 250 and 80 | nothing | WORDS[0] on all | settled | P1 per reading (may be not settled); P2 miss; P3-P5 void; A1 hit; A2, A3 void. Wrong if a reading's own gate held (S2) |
| Stage A all-yes, views at same dose | R+0, then B | B's word per reading | per B | P3 settles if then_how agrees; I+0/L+0 owed if any reading raises |
| Stage A all-yes at 250, views only at 80 | stage A at 80 | all-yes at 80 leads to B at r80; otherwise WORDS[6] on all | settled | P4, P5 void |
| Stage A all-yes, views never hold | none | WORDS[6] on all | settled | |
| Split; mend gate fails | R+0 owed at r250 | yes readings WORDS[6], others WORDS[5] (or earlier) | not settled | P2 not settled; P3 cannot settle (m4) |
| Split; mend all-yes | as all-yes at m250 | per B | | P2 not settled |
| Split; mend split | R+0 at m250 | as above | not settled | |
| **Split; mend none** | none | **WORDS[5] on all** | **settled, wrong on the readings that saw the first yes (S2)** | P2 not settled |
| No yes; mend (as registered) | | WORDS[5], or follows the mend's result | | |

- No new path waits for ever. Cap waits are registered, and S1 makes one certain at the second judge.
- Money that cannot settle anything: R+0 on a split; I+0 or L+0 when they raise on only one reading (m4).

## 6. The second ask

- Applied exactly where the amendment says: both wordings, for the practice and for every judged job (run_view2.py:715, 1136). It is not applied to the room question or to the second judge (which has 4,000 tokens).
- It reduces a bias rather than adding one. Wording 2 asks for quotes and ran out of room more often (4 of 150, against 2 of 150), and under "no answer line = flag" that would have tilted narrow toward flagging.
- It is an unvalidated change of instrument for about 1-3% of replies, and it is named (line 108).
- The 1% rule: room-question answers were missing in 0.3-0.8% of the first version's replies, and after the second ask the second question's should be about 0. With the larger denominator, the rule is unlikely to trip. Before the amendment it could have, at about 1.1-1.3%.

## 7. Five ways a result can be invalid

- **Copied numbers:** the trial counts recompute exactly (26/25 of 26, 34/7; 28/20, 32/1; no answer line 2 and 4; $2.06 = $1.32 + $0.74). The counts in m1 use the trials' convention, not the run's.
- **Substituted instrument:** the second ask at 800 tokens (named); the wordings were tried on other models' replies (named).
- **Altered operand:** pairs dropped by the both-no rule (named); the 1% denominator (not named); which stage A arms A3 pools (open).
- **Outcome chosen after seeing values:** the hand-mark statistic (m5), A3's recipe, the presentation of "not settled", and the optional R+0 at r250.
- **Wrong statistic:** P3 cannot settle on a split (structural); A3's ratio is right.

## 8. The tests

121 passed in 19.7 s. The new tests mostly test behaviour on made-up data. test_view2.py:73-95 only checks that constants match the yaml. No test would catch:
- word_state after the mend's stage A is read: all-yes leading to B at m250, a split, or none (S2's path). The tests stop at "not read yet" (test_view2.py:1091).
- the second judge's mapping from wording to reading. The stand-in answers "no" to both wordings for the arms' replies (test_view2.py:247-252), so a swap would not show.
- `unread` or the 1% rule counting both wordings' missing answers.
- the second ask inside `judged()` (only the practice is tested).
- a gate split giving WORDS[0]; A3 when the mend ran.
- the API budget: the stand-ins book no spend.

## 9. What I could not check

- the judge's answers at 800 tokens;
- how many of the 34 pairs each wording flags (needs a paid call);
- the flag rates on this step's own replies. My estimates by source rest on 18-19 replies per stratum, and the overall flag share of the replies on disk matched the ledger's 575 of 4,682;
- the GPU part of the $17 estimate;
- whether temperature 0 is honoured, so whether the second ask is deterministic;
- the Modal app.

## The three changes I would make first
1. Re-budget the API part in the amendment before registering it (S1). Without that, the after-check is certain to wait for Eugene, and the main judge may too.
2. Mend the word on the mend path, and the gate split (S2). A reading that saw the yes at the first recipe must not end on a settled WORDS[5]. Add tests for word_state after the mend is read.
3. Name the wide wording's flags that may follow the treatment, and the chance of a stop at the practice (S3, S4). Print each arm's flag shares with every split, and add the missing tests for the second judge's mapping and the 1% rule.

---

# Second look

## Second look at amendment 1 (view2_amendment1.yaml) and its code

The tests pass: 123 in 20.3 s. The amendment's sha256 begins `ea6e9f3f`, which matches test_view2.py:31.

## 1. Faults the changes introduce or leave

**Blockers: none.**

### Should-fix

**N1 (a). The step's cap of $29.26 is not held in code. view2_amendment1.yaml:168 says it is ("at most $29.26, in code").** There are three gaps.

- **Containers that fail are never booked.**
  - When a container raises an error, the app moves on with `continue` and never calls settle (modal_overlap.py:432-434).
  - So a crash, or a run to the 7,200 s timeout (modal_overlap.py:91), is billed by Modal but never reaches `gpu/spend.jsonl`. A timeout costs up to about $6.55, against a $0.38 reservation.
  - `spent()` reads only that log (run_view2.py:151-156).
  - This is old behaviour of the app. The amendment now rests its cap on that log.
- **Reservations still running are not counted.**
  - `check_api` (159-165), `leg` (180) and `write_jobs` (979-981) see only GPU cost already booked.
  - Two batches of jobs can run at once. For example, the gate at 80 and stage A at 250 are both allowed while the gate at 80 is unread (1369-1372, 1476-1481).
  - While containers run, API calls are made and further jobs are written as if those containers cost nothing yet.
  - The app checks only its own $19 cap (modal_overlap.py:72, 423-426).
- **Writing jobs does not count what judging them will cost.** `write_jobs` checks the GPU reservation only, while the judge checks one job at a time (1123). A stage can therefore be trained and then refused partway through judging. The registration's cap stop covers this, but the GPU money is lost. On the paths expected to fit this is unlikely: the 420 s reservation is about twice the first version's real cost per container.

**Is `leg()` right?** Partly.
- What it gets right: its guard's log holds only API lines (run_look.py:188-192, 420), so cap minus GPU keeps API plus GPU at or under $29.26.
- What it misses: it sees only GPU cost booked when it is entered. It is entered again for each batch of calls (284-286), so it is at most one batch stale.

**Spending twice: no.** The judge, the second ask and the second judge go through the cached engine with fixed salts (263-268). `write_jobs` and the app both skip jobs already on disk (modal_overlap.py:420-422).

**Smallest fix:**
- Add one function, `outstanding()`, that costs every job in a written `*.jobs.json` not yet on disk at its reservation, (EST + START_S) × rate. Add it to the GPU spend that `check_api`, `leg` and `write_jobs` use.
- In the app, settle a failed container at its reservation under a name beginning `view2_`, or shorten the timeout for the set view2.
- If the code is not changed, change line 168 to say the cap is checked before each call and each batch of jobs, and that failed containers are not booked.

**N2 (e). The cost figures hold, within about $1, but the yaml does not state its assumption about container time.**

My reckoning uses the given prices and the first version's container times (115-150 s, about $0.18 a container; GPU log in 2026-11-19/attractor-o1).

| Piece | Containers | Judge cost | Total |
|---|---|---|---|
| A dose gate | 18 | about $0.85 | about $4.1 |
| Stage A | 21 | 15 jobs × $0.195 | about $6.8 |
| Stage B | 16 | 10 jobs × $0.195 | about $4.9 |
| One X+0 arm | 5 | 5 jobs × $0.195 | about $1.9 |
| Second judge | – | – | about $1.7-2.0 |
| Floor | 2 | – | about $0.5, plus the replay's API calls, which I did not estimate |

| Path (yaml:173-181) | yaml | mine |
|---|---|---|
| Gate holds at neither dose | 8 | 8.7 |
| Split at stage A | 15 | 13.2, or 17.3 if the gate at 80 was asked |
| No yes, then the mend | 24 | 24.1 |
| Yes on each reading, then the views | 22 / 26 | 20.0 / 23.8 |
| Mend, then the views | 33-37 | 30.9-34.7 |
| Stage A at both doses, then the views | 32 | 30.9 |

- About $27.94 is left under the cap.
- The mend path has 80 containers. It fits only if they average under about 190 s; the GPU's own $19 binds at about 200 s.
- The mend's rank-64 documents may train more slowly. State the seconds assumption in the yaml.
- No GPU job of this step has run yet (there is no `gpu/` under 2026-11-21/attractor-o1-view2), so the floor is part of "the rest".

### Minor

- **N3 (d).**
  - What the amendment says: the readings on which the gate failed are named beside WORDS[0] (yaml:72-73).
  - Where they are: gate_r*.json (run_view2.py:1411-1413), and they are printed at gate_read (1422-1423).
  - What the step's word gives instead: verdict.json and the word line give a fixed clause (view2_core.py:859-860).
  - Fix: set `A_note` from the gate records.
- **N4 (d).** yaml:66 says that keeping under 28 "would need all seven flagged". That is true only if none of the 27 plain replies is flagged.
  - Wording 1 flagged 6 of 19 plain right-room replies of the unchanged model in its try.
  - yaml:153 itself gives about 45% for that model's right-room replies.
  - Say "seven of the 34", and that a stop at the practice is possible.
- **N5 (d).** The arms the limit is about are not shown.
  - yaml:95 and the limit at yaml:151-158 are about the arm with the replay alone and about the unchanged model.
  - `flagged_by_arm` (run_view2.py:1676-1680) comes from `described`, which holds only the arms with P (view2_core.py:834).
  - So the X+0 arms and the base are not shown. Add their mixed shares.
- **N6 (c).** The 1% rule's groups match yaml:103-110.
  - At a stage: each arm's seeds after P together with its documents' probes; each X+0 arm on its own (1207-1216).
  - At the gate: all four kinds plus none, on the gate's seeds (1403).
  - A new looseness: pooled with about 2,000 replies per job after P, one documents job's 48 probes can lose about 18 at the gate (all 48 at a stage) and still pass. `took` then reads that seed on missing labels.
  - Fix: record the missing probes per job, or name this in the limits.
- **N7 (b).** A1 and P5 are graded on a word the second judge can still change (1652-1654, 1675).
  - In the split test, A1 is "miss" (test_view2.py:1371).
  - If the second judge then disagreed on all three readings, every reading would read "not read: the judges disagree", the word would be settled, and A1 would become "hit".
  - Either mark them "not read yet" until judge2.json exists, or say so.
- **Trivial.** The P3 docstring (view2_core.py:932) still says "R does not raise the carry"; it should now say "not on every reading".

### (b) word_state and show_verdict, path by path

All of these are now right:

| Path | What the code gives | Where |
|---|---|---|
| Gate fails at neither dose, or fails jointly | WORDS[0] on all readings; P2 miss; P3-P5 void; A1 hit; A2 and A3 void | – |
| Yes on all readings at the first dose | R+0 owed, then stage B; stage A again at 80 only after a yes on all | owed_zeros 1712-1720; 1483; 1617-1622 |
| Split at the first dose | No mend, no R+0, no views; "not settled" with its cause | no mend 1367-1368, 1472-1474; no R+0 because "yes" is "R raises" (view2_core.py:838), so R_zero_owed is false on any split |
| No yes; the mend's gate fails | The first dose's reading stands | – |
| Mend, then no yes | WORDS[5], settled; correct now, since no reading saw a yes first | – |
| Mend, then a split | "not settled" | – |
| Mend, then a yes on all | R+0 at m250, then stage B if the mend's gate holds for the views | – |

- A3 is read on the reading that stands (1662, 1674).
- A2 is void only when no stage A reading stands.
- P1 is graded per reading from the gate's `views`, which requires stage A (view2_core.py:660).
- One path is consistent but untested: stage A run again at 80 splits after a yes at 250. The views are not trained, and every reading gives the same word.

## 2. Earlier points

- **S1:** met in kind. There is one cap; the second judge's check is in two parts at 800 tokens, with the two wordings bounded on `pend` only (1762-1772); each path is costed. N1 remains.
- **S2:** met.
- **S3:** met (yaml:151-158; flag shares are printed and kept in verdict.json). N5 remains.
- **S4:** partly met. The 28 stays and the shapes of the 34 are stated, but the "all seven" sentence overstates (N4).
- **S5:** mostly met. New tests at test_view2.py:1345, 1375, 1415, 1430, 1453, and the second judge's mapping now shows a swap (851-853, 919). Gaps are in section 3.
- **m1:** met (yaml:23-33).
- **m2:** met (yaml:108-110; the groups are per arm). N6 remains.
- **m3:** met (yaml:132; test 1396).
- **m4:** met (yaml:81-83; view2_core.py:942; run_view2.py:1526, 1664).
- **m5:** met (yaml:117-121).
- **m6:** met (1655-1659, 1681).
- **m7:** met (the code matches yaml:131).
- **m8:** met (yaml:79).
- **m9:** open until the ledger registers `ea6e9f3f`. It is not in NOTES.md, and the yaml is untracked. The pin in the test is right and will catch any later edit.

## 3. What the tests still would not catch

- The cap when containers are still running, when two batches of jobs run at once, or when a container fails. Test 1041-1071 fixes the GPU spend before `leg()` is entered.
- The second judge refused at its second check, after the room question was paid. The room answers are then cached, but no judge2.json is written.
- Stage A run again at 80 splitting after a yes at 250.
- A1 and P5 changing after the second judge (N7).
- The X+0 arms missing from `flagged_by_arm` (N5).
- Missing probes per job under the pooled rule (N6).
- Any check of the yaml's path costs against job counts and prices.
