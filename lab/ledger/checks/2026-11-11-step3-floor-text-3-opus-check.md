# The second correction to step 3's note, and the rewritten script: Opus's cold check (11 Nov 2026)

A fresh agent (model opus, read-only, no Modal, network or paid calls, no saved read opened) checked the section
"Second correction, 11 Nov 2026" of the step 3 design note and the script
`lab/explore/understand_floor/floor_words.py` with its tests (commit 463401b). It was asked to hunt for a sentence
the script can print that is false of some readout the frozen code can produce. Its report follows, filed by Fable
(subagents can't write report files). The substance is unedited; four lines are changed, in section 4, where local
scratch paths are replaced by the scripts' names.

What Fable did about it, before any arm was run:

- **The verdict (taken).** No blocker; a small further correction. A section is appended to the note ("Last
  changes to the wordings, 11 Nov 2026") and the script is changed to it. The report asks only for a check of
  those changes, not a new round.
- **S1 (taken).** "Kept part of its release" may be said only when the validity gates passed, E is not at the
  floor and E's later level is above N's; and, from note N3, not when E's release rose.
- **S2 (taken).** Addition (i) no longer says the later reads' part "is the noisy one". It gives what 0, 1 and 3
  releases of 304 count as in log-odds, and says what was not read between step 0 and step 3.
- **S3 (taken, in the report's fuller form).** Addition (iii) is made whenever H is read and the starts' part
  alone is 0.6 or more in size, whatever H's verdict, and its sentence always gives the later reads' part beside
  it.
- **S4 (taken).** With four or five seeds a mean past the threshold "is at most 'suggestive'".
- **S5 (taken).** The 10% line decides two things; (iii)'s condition is one condition; the note says section by
  section what stands and what is replaced; plain summaries carry the readout's note about the start.
- **S6 (taken).** The note no longer says a test proves every sentence true; the tests now cover S1 to S4, a
  failed gate, four and seven seeds, runs of unequal size and a flagged arm (31 tests; the whole offline suite:
  2,081 passed, 35 skipped).
- **Notes N1 to N6 (taken).** An interval that excludes zero is said not to be a reading; a flag on an arm stays
  beside the stake line; seeds left out are named even when all are; a missing interval prints as words; the note
  says the script covers every contrast; (iii)'s sentence gives the start reading in points beside the starts'
  part in log-odds.

---

## 1. Verdict

**A small further correction is needed before block one. Nothing in it is a blocker.** The script hands the frozen verdicts and the registered wording through unchanged. The split of H is exact arithmetic. No reading, threshold, gate, hit or miss changes. But four of the fixed sentences or conditions can print something false in outcomes the frozen code can produce, or contradict "not read" (S1 to S4). Three more points are only text (S5, S6). Each fix is one sentence or one condition, in files outside the freeze. Once they are made as proposed, the wordings are fit to stand, and a check of the diff is enough; a full new round is not needed.

## 2. Findings, most serious first

**S1, should-fix: the "may say kept" line ignores a failed gate.** The line is "plain summaries may say that E kept part of its release: yes, with both levels" (`may_say_kept`).
- It prints "yes" when a validity gate failed, so H, U2 and U3 are "not read". "E kept part, 30% against 3%" is then a statement about the level under pressure, which the registration says is not read.
- Evidence, case 3: N goes from 20.1% to 3.2%, so the practice gate fails, and E goes from 60.4% to 30.1%. The verdicts are U1 "higher" and H, U2 and U3 "not read", and the line still prints "yes".
- Case 3b: N goes from 50.3% to 46.3%, so the pressure gate fails. It also prints "yes".
- Text fix: "…only when the validity gates passed, E is not at the floor and E's later level is above N's."
- Code fix: `return c["H"]["verdict"] != "not read" and not pa["floor"] and pa["later"] > pb["later"]`.

**S2, should-fix: addition (i)'s "Near the floor the later reads' part of H is the noisy one" is a claim, not a fact.** It can be false of the figures printed beside it.
- Case 1: the starts' part has the interval [−0.77, +0.59] and the later reads' part [−0.08, −0.00], yet (i) calls the later part the noisy one.
- In the tests' own 400 random readouts (re-drawn with the same seed), 25 of the 187 that print (i) have the wider interval on the starts' part.
- It is also false by binomial sampling when an arm's start is itself low (case 2, E at 2.5% at the start).
- It counts again noise that the interval and the p already carry, so a reader on the project's side can fairly object.
- Two smaller points:
  - "−6.4" is a seed's own log-odds, not its share of a difference. With environment errors it drifts: −6.0 at n = 200.
  - "A delay of a step or two could not have been seen" is loose. A delay is unseen only if the release was gone by step 3. If N falls at step 2 and E at step 4, the read at step 3 shows it.
- Proposed sentence: "Both arms ended under 10% (steps 3 and 6 pooled): E x%, N y%. There a seed's later log-odds turns on a few episodes: 0, 1 and 3 releases of 304 count as about −6.4, −5.3 and −4.5. Nothing between step 0 and step 3 was read: a release that held through step 1 or 2 and was gone by step 3 reads as one that fell at once."

**S3, should-fix: addition (iii) is fair between the two directions of H, but one-sided between the two parts, and silent when the start hides a difference.**
- **Case C, the outcome the plain account predicts.** E starts 15 points higher. Four of E's seeds end at zero. H reads "holds worse".
  - (iii) prints "the starts' part alone (−0.62) is as large as the 0.6…".
  - The later reads' part, −2.22, would give the reading on its own, and no sentence says so.
  - So a summary can lean on (iii) to discount a real negative.
- **Case B, the mirror.** H reads "holds better" and the stake line prints. (iii) fires at +0.68, beside a later reads' part of +1.66.
- **Case D, a start that hides a difference.** E starts higher, which gives a starts' part of −0.62, and ends higher, a later reads' part of +0.67. H is +0.05, "no difference shown", and no addition prints. Revision 3 foresaw this case ("takes about 0.6 off a real protection").
- **The phrase "as large as the 0.6 a reading needs".** It is accurate only about the mean. A reading also needs p under 0.05, and the 0.6 applies to H's mean, of which this is one part.
- **Proposed fix.**
  - Condition: H is read and |starts' part| ≥ 0.6, whatever H's verdict.
  - Sentence: "The starts' part alone is S, as large in size as the 0.6 the rule asks of H's mean (E started lower/higher in log-odds); the later reads' part is L."
  - This is fair in both directions, names both parts without ranking them, covers the hidden case, and removes the mismatch between text and code in S5(b).
  - Minimal alternative: keep the present condition and append "the later reads' part alone is L, also / not 0.6 or more in that direction."

**S4, should-fix: addition (vi) says "a mean past the threshold can only be 'suggestive'", and that is false.**
- Case 4: four seeds, mean +0.91 (past the threshold), p 0.25, verdict "no difference shown", printed beside that sentence. Five of the tests' own random draws do the same with five seeds.
- Fix in both the note and the code: "at most 'suggestive'".
- Low likelihood: four or five seeds arise only from drops in an ended run.

**S5, should-fix: four points in the text.**
- (a) "The 10% line now decides one thing only: whether addition (i) is made." The floor also decides the "kept" rule ("only when E is not at the floor"). It should say "two things".
- (b) (iii)'s condition reads "H's mean is past the threshold (H reads 'holds better', 'holds worse' or 'suggestive')". These are not the same set.
  - Case A: mean +0.73, p 0.25, verdict "no difference shown", starts' part +0.73. The main clause says (iii) applies; the code follows the parenthetical and prints nothing.
  - The same happens when H is "not read" with its mean past the threshold.
  - S3's form removes the problem. Otherwise write "H reads 'holds better', 'holds worse' or 'suggestive'".
- (c) "The rest of the correction above stands:" names one item. A reader cannot tell whether these still stand:
  - the definition of "at the floor";
  - "A run that ends early";
  - "Other corrections";
  - "The figures given in every case";
  - the remaining bullets of "What each outcome is called", for example "(c): understanding first… did not keep the practised release… by more than the interval on H allows" and "E starting lower is a negative for these documents at the start".
  - Proposed: "From the correction above, the definition of 'at the floor', 'A run that ends early, and the site's count' and 'Other corrections' stand; its 'What each outcome is called' and 'The figures given in every case' are replaced by this section."
- (d) Revision 3 says "Every reading of H carries the note about the start". The base and the plain-summary list leave it out, and the script does not print it. Say whether it goes into plain summaries.

**S6, should-fix (small): the note overstates the tests.** It says the 400-draw test "checks that every sentence printed is true of the figures printed beside it".
- In fact it re-derives each addition's condition from the script's own output and checks presence and the banned words.
- Its own draws contain the S2 and S4 failures.
- Proposed: "…checks that each addition is printed exactly when its condition holds and that no withdrawn word appears." Add tests for S1 to S4.

**Notes**
- **N1. Addition (v)'s t interval against the exact test.** The interval can exclude zero beside "no difference shown", and beside (vi)'s "no reading is possible".
  - Case 7: three seeds of an ended run, interval [+0.77, +1.14].
  - Case 8: eight seeds, mean +0.36, interval [+0.32, +0.41].
  - The sentence is true by the registered convention, but a reader may take it as a difference shown. When the interval excludes zero, consider adding "the verdict comes from the exact test and the 0.6 threshold, so this is not a reading".
  - With identical seeds the interval is [0, 0]. This happens only in synthetic data.
- **N2. The stake line ignores a guard flag** (case 11). The wording line carries the flag; the stake line could add "the readout's flag stands".
- **N3. "Kept part of its release" when E's release rose** (case 12: 5.6% to 12.1%) is inaccurate. Very rare.
- **N4. The script covers every contrast.** It prints the base, the additions and the "kept" line for R−N and E−R too; the text speaks of E minus N only.
- **N5. Two print details.**
  - When every seed is left out of the share retained, the script prints "no seed to compare" without naming the seeds, though the text says they are named.
  - The "−" printed for a missing interval right after a signed number ("−0.62 - +") reads as a minus sign.
- **N6. Case B: "no head start" beside a large starts' part.** The registered wording says "holds better with no head start" beside (iii)'s starts' part of +0.68. The first is in points (U1), the second in log-odds. One clause would prevent confusion.

## 3. Answers to A to F not covered above

**A. Each addition: when it prints, and whether it is then true.**
- (i) prints when both arms' pooled later levels, over the shared seeds, are under 10%. "Both arms ended under 10%" with the two levels is always true. The rest is S2.
- (ii) prints exactly when H reads "holds better" and U2 does not read "higher", or H reads "holds worse" and U2 does not read "lower". It is always true. When U2 reads the opposite way it understates, but the verdict is printed.
- (iii) prints when H reads "holds better", "holds worse" or "suggestive" and the starts' part times the sign of H's mean is at least 0.6 − 1e-9, the same tolerance as `read_verdict`. It is true whenever printed. Fairness is S3.
- (iv) prints exactly when H is "suggestive". It is true: that verdict means a mean past the threshold with p from 0.05 to 0.20.
- (v) prints when H reads "no difference shown" or "suggestive" and there are at least two seeds. It uses the readout's own interval. See N1.
- (vi) prints when fewer than eight seeds are shared or the run ended. The smallest p of 2/2^n matches `perm_p_paired` for 2 to 8 seeds. "With six, every seed's difference has the same sign" is true: one zero difference raises the smallest p to 0.0625. The fault is S4.
- The stake line prints when the primary H reads "holds better" and U1 is not "lower". That matches the registered table.
- The split line: the script's mean of H equals the readout's in every readout I built, and the two parts sum to the frozen H for each seed.
- No fault in these: three arms, a dropped seed (7 seeds), ended runs at 3 and 6 seeds, unequal n, an H of exactly 0, one and two seeds, an arm at 0% start, H "not read" (only (i) prints).

**B. Differences between the text and the code.**
- (iii)'s condition: S5(b).
- (i): the code adds the "−6.4" parenthetical.
- (vi): the code also covers 1, 2 and 7 seeds ("a reading is possible").
- The "kept" rule: the text gives a condition ("only when"); the code prints a permission and ignores the gates (S1).
- The code also prints each arm's pooled share retained, its floor status, and a warning if its mean of H differs from the readout's.
- N4 and N5 above.
- The seeds used and the intervals agree with the text: the script's t table matches `t_crit` to three decimals, and (ii) and (v) use the readout's own intervals.

**C. Coherence with the registration.**
- Nothing changes a registered reading, threshold, gate, hit or miss, or the outcome table.
- The rule that a "holds better" is a sighting of the stake unless the start reads "lower" matches the table's row for a lower start.
- The rule that a result is a sighting either way and stays out of the site's count is stated as standing.
- What remains unclear is S5(c) and (d).

**D. Even-handedness.**
- (ii) is fair in both directions.
- (iii) and (i) are covered by S3 and S2. (i)'s sentence is a claim, and it is not true whenever (i) applies.
- The 10% line was chosen after the dose checks were seen. It now gates only a sentence and the "kept" rule, which is acceptable once S5(a) is stated.

**E. The tests.**
- 25 pass (1.7 s).
- These are now tested against the frozen code: verdicts from `U.readings` through the saved JSON, `log_odds`, `t_crit`, `perm_p_paired`, the means in `contrast`, and the per-seed sum of H.
- The 400-draw test only mirrors the script's own conditions. It always passes `gates_ok=True` and never draws 4 or 7 seeds.
- Untested:
  - the content of (i) and (vi);
  - the "kept" line under a failed gate;
  - a mean past the threshold with p above 0.20;
  - unequal n;
  - flags;
  - naming the seeds when all are left out.

**F. What must be done before arm data.** No blocker. S1 to S4 are should-fixes that are cheap now and would be changes after the data if left. S5 and S6 are text.

## 4. What I ran and what I only read

- **Ran:** the 25 tests, and three synthetic scripts, built with `u1_core.seed_values`, `gates` and `readings` and passed through JSON, covering the cases above:
  - `cases.py`
  - `cases2.py`
  - `edge.py`
  - with the helper `build.py`
- I also re-drew the tests' 400 readouts with content checks, and ran `perm_p_paired` with a zero difference.
- I confirmed that the freeze (`run_u1.freeze_paths`) does not cover `lab/explore/understand_floor/`, so fixing the script will not break it. `git status` was clean before and after.
- **Read only:** the whole design note, the previous check, the readings functions in `u1_core` and `compare_u1`'s `all_report`, `show_all`, `cmd_all` and `load_ended`. No results file was opened, and I did not run the whole test suite.

## 5. Cost

$0.
