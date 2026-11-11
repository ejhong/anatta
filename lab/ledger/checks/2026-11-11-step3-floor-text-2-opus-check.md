# The correction to step 3's note, and the script beside it: Opus's cold check (11 Nov 2026)

A fresh agent (model opus, read-only, no Modal, network or paid calls) checked the section "Correction to the
section above, 11 Nov 2026" of the step 3 design note and the script `lab/explore/understand_floor/floor_words.py`
with its tests (commit c7d11f8), and with them the ledger entry of 10:09 UTC and the entries appended to the
notebook and SIGHTINGS. It was asked to try to break them. Its report follows, filed by Fable (subagents can't
write report files). The substance is unedited; one line is changed, in section 4, where a local scratch path is
replaced by a plain statement.

What Fable did about it, before any arm was run:

- **The verdict (taken).** A further correction was needed. A section is appended to the note ("Second correction,
  11 Nov 2026") and the script is rewritten to it. Nothing in the treatment, the readings, the gates or the frozen
  code changes.
- **F1, the sentence that named the larger part of H (taken; the mechanism is withdrawn).** The split of H is now
  given as arithmetic: two numbers with their intervals, no ranking, no words about levels or causes. One
  conditional sentence stays, as the report proposed: when H's mean is past the threshold and the starts' part
  alone is 0.6 or more in the same direction, the report says so.
- **F2 (taken).** The cases (a) to (d) are withdrawn. Their place is taken by a base given in every case and six
  additions, each added exactly when its condition holds. The two that concern the direction of H apply the same
  way to "holds better" and "holds worse", on the floor and off it.
- **F3 (taken).** A primary "holds better" is a sighting of the stake unless the start reads "lower"; then the
  registered table's wording stands.
- **F4 and F5 (taken).** "Suggestive" is said, with its mean and p. The interval sentence now says what is excluded.
- **F6 (taken).** The claims about the declaration disappearing are corrected to what was counted, the tag, with
  the report's crude count of prose replies, which Fable recounted and found the same. "Clearly alive" and "clearly
  off the floor" are corrected in the ledger, the notebook and the site.
- **F7 to F11 (taken).** "Collapsed" becomes "ended under 10% (steps 3 and 6 pooled)". The script prints what a
  short or ended run can read. "Kept part of its release" needs E off the floor and above N. Plain summaries are
  fixed for every case. The sentence about "two one-cent roundings" is corrected.
- **The tests (taken).** They now build readouts with the frozen code's own functions and pass them through the
  saved form; the t table and the thresholds are compared with the frozen code's; one test draws 400 random
  readouts and checks every sentence against the figures beside it; the ended file, three arms and `main()` are
  covered. 25 tests; the whole offline suite passes (2,075 passed, 35 skipped).
- A last cold check reads the second correction and the rewritten script before block one.

---

# Cold check: the correction to the step 3 note after the second dose check, and `floor_words.py` (commit c7d11f8)

Opus, read-only, 11 Nov 2026. Nothing in the repo or the results folder was written; `git status` is clean after every run.

## 1. Verdict

**A further correction is needed before block one.** The treatment, the readings, the gates and the frozen code are untouched, and every number I could recount holds. The problem is the one new mechanism, the "where it comes from" sentence. It names the larger of the two parts of H in words about levels. It misfires in the very case it was added for, a head start that ends at the same floor. In some cases it also contradicts the levels printed in the same sentence. Four smaller faults in the wordings and some overclaims in the public entries also need fixing. All of it can be fixed with an appended text and an edit to the script, which sits outside the freeze (`freeze_paths` hashes only `lab/explore/understand/**` plus the shared files).

## 2. Findings, most serious first

**F1, blocker: the source sentence in (a) and (b) (`floor_words.source`, and the note's "where the difference in log-odds comes from … from E's higher start, from E's lower later level, or more from one than from the other").**

The sentence fails in two ways.

- **It uses words about levels for a mean of per-seed log-odds.** The "later levels' part" is the mean over seeds of the difference in log-odds at the later reads. A seed at 0 of 304 counts as −6.41 there. So its sign can be the opposite of the pooled levels printed beside it. I ran synthetic readouts built with `u1_core.seed_values`, `gates` and `readings` and `json.dumps(default=float)` through the script:
  - Case (a). E starts at 69% and N at 50%. Three of E's seeds end at 0 and five at 18/304; all of N's end at 9/304. The script prints: "both collapsed (E 3.7%, N 3.0% over the later reads); by the registered reading E's wore down faster, more from E's higher start than from E's lower later level".
  - Case (b), the mirror. The script prints: "E's release fell less in log-odds, more from E's lower start than from E's higher later level; later levels E 3.0%, N 6.2%".
  - In the driver's own sketch model (head start, same floor), 5 to 7% of "holds worse" readings produce such a self-contradicting sentence.
- **Naming the larger part is unstable, and the selection biases it toward the noisy part.** I reran the sketch (the floor.py model) and asked, for each reading, which part the script would name. The case is a 65% against 50% start, both ending at 4%, so the later levels are equal by construction. Among the "holds worse" readings, the script would say "more from E's lower later level than from E's higher start" in:
  - 102 of 2,130 at low noise (5%);
  - 266 of 921 at middle noise (29%);
  - 371 of 590 at high noise (63%).

  A reading passes the threshold mostly when noise in the later reads pushes it there, so the later part tends to lead. So the note's sentence "a 'holds worse' that comes from a higher start alone is named as that" is false in much of the very case it was written for. Separately, `EPS = 1e-9` lets a trivial part be named. With one seed of E starting one episode lower, the script printed "more from E's higher later level than from E's lower start" for a starts' part of +0.02.

**Corrected text:** "The report gives the split as arithmetic, not as a cause: the starts' part S [95% t interval] and the later reads' part L [interval], in log-odds, sample means over seeds. In L a seed with no release counts as −6.4, and near the floor L is the noisier part. If S alone is 0.6 or more in H's direction, the report adds: 'the difference in starts alone gives a reading of this size.'" Drop the "higher/lower later level" words and the ranking.

**Code:** replace `source()` with a function that prints both parts with their intervals and the conditional clause, keyed on `abs(S) >= 0.6 and sign(S) == sign(H)`.

**F2, should-fix: (a) and (b) are not mirror images, though the note says they are.** The two cases have different conditions:
- (b) covers every "holds better" with U2 not "higher", wherever the arms end;
- (a) covers "holds worse" only when both arms are at the floor.

Example: E goes 65% to 15% and N 50% to 15% (eight identical seeds). That reads H −0.62, "holds worse", label "starts better and wears down faster". The script puts it in **(d)** and names no start. The mirror (E starting lower) goes to (b) and is named "from E's lower start". So "and so is a 'holds worse' that comes from a higher start alone" is false off the floor, and so is the 10:09 ledger's "one rule covers a 'holds worse' that is only a head start and a 'holds better' that is only a lower start".

- **Corrected text:** "(a) H reads 'holds worse' and U2 does not read 'lower'. … The words 'both arms ended under 10%' are added only when both arms are at the floor."
- **Code:** `which()` returns "a" when `h == "holds worse" and u2 != "lower"`.

**F3, should-fix: "A primary reading of 'holds better' is a sighting of the frame's stake wherever the arms end" contradicts the registered table when U1 reads "lower".** For that pairing the table says only "starts worse; H is reported as read". The new list also calls that pairing "a negative for these documents at the start". It is the same double naming that B1 objected to, carried over verbatim from the previous check's own suggested text. In my synthetic case E starts at 12% and N at 50%. +1.98 of the +2.40 is the starts' part, yet the bullet makes it a sighting of the stake.
- **Corrected text:** "…a sighting of the frame's stake wherever the arms end, unless U1 reads 'lower'. Then it is worded as the table says, with its two parts."

**F4, should-fix: (c) drops "suggestive".** The registration says such a mean "is written 'suggestive'", and the readout's own wording appends "(H was suggestive, which counts as no reading)". Script output for E at 8.8% against N at 3.9%, H mean +0.60, verdict "suggestive": "both collapsed (E 8.8%, N 3.9% …); no difference in holding is shown". A reader hostile to the sceptics can fairly object to that.
- **Corrected text:** "When H is 'suggestive' the report and plain summaries say so, with its direction, mean and p, and that it counts as no reading."
- **Code:** branch on `c["H"]["verdict"] == "suggestive"` in `words()`.

**F5, should-fix: "the interval on H is what the test excludes" says the opposite of what it means.** The note's (c) and the script both use it ("the t interval on H, [-0.16, +1.37], is what the test excludes"). Literally it says the values inside the interval are excluded.
- **Corrected text:** "the t interval on H is [lo, hi]; differences in holding outside it are excluded at 95%."

**F6, should-fix: claims in the public entries that go past the counts.**
- **"The lost releases did not become declarations in a wrong form"** (ledger 10:09), "So the declaration disappeared" (SIGHTINGS), "the declaration did disappear" (notebook), and the same on the site (experiments.html:252). `release_form` detects only the tag, so a prose declaration without the tag counts as "never used the tag". My count by a crude keyword function (counts only, no text printed):
  - at step 3, 47 of the 191 failed episodes that never used the tag had a reply with no code fence and no `def`, and 14 of those used words such as "impossible", "contradict" or "conflict";
  - at step 0 the figures were 25 and 6 of 88; at step 6, 57 and 21 of 192.

  **Corrected:** "After three steps the release tag had all but gone from the replies: the lost releases did not turn into tags in a form that does not count. Whether some replies say in prose, without the tag, that the task cannot be done was not counted."
- **"Does it keep that reply clearly alive"** (notebook) and "keep it clearly alive through the first six steps" (site). These bring back the "clearly off the floor" framing that B1 withdrew; the 09:43 ledger's "clearly off the floor" is not corrected at 10:09 either. The primary reading is H. **Corrected:** "whether it loses less of that reply, in log-odds, over the first six steps; it can catch that only if the effect is large."
- **"The two parts of H are reported so that such a reading is seen for what it is"** (ledger) overstates what the parts do, given F1. The figures are printed; the verbal label is not reliable.

**F7, note: "collapsed" is still undefined.** B1 withdrew the outcome list for undefined words, but (a) and (c) still say "both arms' release collapsed". The floor tests only that the later level is under 10%. Given the gate and the H verdicts it is mostly apt, but it is cleaner to write "both arms ended under 10% (steps 3 and 6 pooled)".

**F8, note: an ended run.** The note says the report states the limits of a short run; the script reads `compare_u1_ended.json` correctly but prints nothing about "ended" or about the smallest possible p. Computed with `perm_p_paired`, the smallest p is:
- 0.25 with 3 seeds;
- 0.0625 with 5 seeds (a block of six with one seed dropped), so nothing can read but "suggestive" is possible, which the text does not cover;
- 0.03125 with 6 seeds, so a reading needs all six to agree.

**Code:** print `rep.get("ended")` and the smallest p (2/2^n), with what it rules out.

**F9, note: "Plain summaries may say that E kept part of its release" whenever E is not at the floor.** That includes the case where E fell less only because it started lower and both ended at 15%. Then N kept the same part.
- **Corrected text:** "…only when E is not at the floor and E's later level is above N's, with both levels."

**F10, note: (d) fixes no plain-summary wording.** The likeliest positive outcome ("holds better" with U2 "higher") and every outcome off the floor get free-form plain summaries. Fix one: "the registered wording, with each arm's start and later level."

**F11, note: "Its numbers hold, apart from two one-cent roundings."** This understates N3 (solve rates quoted for the wrong set of pairs) and N2 (the overfit items). Both are corrected elsewhere in the same commit.

## 3. Answers to A to H not covered above

**A. The previous check's findings.**

| Item | Status | What is still wrong |
|---|---|---|
| B1 | Partly fixed | The list is withdrawn and the "clearly off the floor" condition is gone; the U1-lower double naming remains (F3) |
| S1 | Fixed | |
| S2 | Excuse removed; not fixed as a pair | (a) and (b) are not mirrors (F2); the source sentence misfires (F1) |
| S3 | Fixed | |
| S4 | Mostly fixed | New overclaims (F6) |
| N1 to N4 | Fixed, and recounted | |
| Q5 | Fixed, except for an ended run (F8) | |
| Q8 | Partly fixed | Outcome mapping (F3), (d) wording (F10) |
| Q9 | Fixed | The "0.3 to 0.5" figure counts "suggestive" in either direction; the positive share is at most 0.016, so it is negligible |

**B. Coherence.** Exactly one case applies for every combination: `which()` enumerates all 5×5×4 patterns and keys on H's verdict, and a gate failure goes to (d). The sentences that are false or misleading in some combinations are covered by F1 to F5 and F9. Nothing changes a verdict. The new rule that the result does not enter the site's count applies either way, so it is even-handed.

**C. The split.** The identity is exact per seed and in the mean, with the half-count correction, because log-odds are taken cell by cell and the mean is linear. The recomputed mean of H equals the readout's in every synthetic readout I ran (`h_agrees_with_readout` True). "Comes from" is fair only if it is named as arithmetic. As written it reads as a cause, and the ranking is noise-driven (F1).

**D. The script against the text.** Everything else matches the text:
- pooled levels over the shared seeds, with dropped seeds out;
- the drop in points, and the share retained with zero-start seeds left out and named;
- the seeds with no release later;
- 95% t intervals, and none with fewer than two seeds;
- one arm missing a seed, and a dropped seed;
- a failed gate goes to (d).

A mean of H that is positive under "holds worse" cannot happen (`read_verdict`). Small points:
- "H, E-N: mean … ; no verdict" can be read as saying H has no verdict;
- U2's interval is printed to one decimal, where the readout prints two ("as printed").

**E. The script against the frozen readout.**
- Every field the script reads exists in `all_report` with the meaning assumed. In `values`, dropped seeds are already left out, so the script's filter is redundant and harmless.
- Every lookup that depends on string keys is right.
- `log_odds` is identical to `u1_core.log_odds`.
- T975 equals `round(u1_core.t_crit(df), 3)` for df 1 to 7.

**F. The tests.** 22 pass, and the whole offline suite gives 2,072 passed and 35 skipped, as the ledger says. The tests mostly mirror the code:
- the t table is compared with a copy of itself, not with `t_crit`;
- the fixture readout is built by hand, not through `u1_core.readings` and a JSON round trip.

Untested: the source sentence against the pooled levels (F1), the ended file (F8), "suggestive" in (c), contrasts involving R, `main()`, and `show()` for cases (a), (c) and (d).

**G. Recounted, and they hold:**
- failed episodes by tag form: 24 of 112 non-strict and 88 never at step 0; 6 of 197 and 191 at step 3; 7 of 199 at step 6; 20 of 188 at step 12; a code block in every reply in 149 of 197;
- solve rates on the 28 pairs: 78, 80, 84 and 78 of 112 (69.6, 71.4, 75.0 and 69.6%);
- overfit items by seed (through step 3: seeds 2, 3 and 6 one each, seed 7 two; through step 6: one to three in seeds 2, 3, 4, 6 and 7; seeds 0 and 1 none); the draw fingerprint matches sets.json;
- money: $6.7212, $3.0775 and $47.4734;
- "3 to 16%": the driver's sketch gives 0.03, 0.13 and 0.16, mine 2.6 to 15.4%;
- "0.3 to 0.5": 0.30 to 0.52.

The one sentence the brief named, "the lost releases did not become declarations in a wrong form", overstates (F6).

**H. Still open before arm data.** F1 to F6 and F10. Choices a careful outsider would call forking paths:
- ranking the "larger part";
- the 10% floor, chosen after seeing the told adapters. It now picks the lead sentence and the word "collapsed", and only (a) carries it (F2);
- plain summaries left free in (d).

## 4. What I ran, what I only read, what I could not check

- **Ran:**
  - the 22 tests and the whole suite;
  - t_crit against T975;
  - synthetic readouts built with the frozen functions and passed through the script (13 scenarios);
  - the driver's floor.py;
  - my own sketch with "suggestive" split by direction, which part is named, and the contradiction rate;
  - perm_p_paired at 2 to 8 seeds;
  - independent recounts of the tag forms, the prose keyword counts, the solve rates, the stream items and the spend.

  My scripts were kept in a session scratch folder outside the repo.
- **Read:** the whole note, the previous check, u1_core's readings, compare_u1, run_u1's freeze, koan_env's outcome logic, the three entries, the site lines that matched my searches, and the driver's four scripts.
- **Not checked:**
  - what the prose replies actually say (keyword counts only);
  - the rest of the site;
  - the rehearsals' reads (not opened).

## 5. Cost

$0.
