# The last changes to step 3's wordings: the same checker's check of the diff (11 Nov 2026)

The Opus agent that made the cold check filed as `2026-11-11-step3-floor-text-3-opus-check.md` was asked, as that
report itself proposed, to check only the changes made in answer to it (commit 4450eab against 463401b: the note's
section "Last changes to the wordings", the script and its tests). It is the same agent continued, so this is a
check of a diff by a reader who already knew the work, not a cold check. Read-only, no Modal, network or paid
calls, no saved read opened. Its report follows, filed by Fable. The substance is unedited; six lines are changed,
under "What I ran", where local scratch paths are replaced by the scripts' names.

What Fable did about it, before any arm was run:

- **The verdict.** Fit to stand before block one; nothing must still change.
- **The phrase in addition (iii) (taken).** It now reads "by the mean over seeds of the log-odds at step 0, E
  started lower" (or higher), so it cannot be read as a statement about the pooled levels printed above it; and
  "as large in size" becomes "at least as large in size". A test covers the case the report built.
- **The two notes (taken).** The note no longer says the random-readout test compares the figures in every
  sentence; and the script prints the readout's note about the start beside the split.
- 32 tests; the whole offline suite passes (2,082 passed, 35 skipped). With that the wordings are closed.

---

## Check of commit 4450eab (the diff since 463401b only)

**Verdict: fit to stand before block one.** Nothing must still change. One phrase in addition (iii) should change: it is cheap to fix and is the only sentence I found that can read against the figures printed beside it. Two smaller notes are below.

## Findings

**Should-fix: the phrase "(E started lower in log-odds; …)" in addition (iii).** It describes the mean over seeds of each seed's log-odds. It can disagree with the pooled starts printed just above it when an E seed starts at or near 0 of 152.
- Case T: two E seeds start at 0 of 152 and six at 110. E's pooled start is 54.3% and N's is 50.0%, yet (iii) prints "+0.71 … (E started lower in log-odds; in points the start reading U1 is 'no difference shown')".
- 17 of 1,500 random readouts in my search did the same. That search over-weights seeds at 0, and a real E seed whose practice did not take at all would be rare.
- Read as the mean of per-seed log-odds, the sentence is true. Read as the pooled level, it is false. This is the same kind of slip as the earlier "where it comes from" sentence, though much milder.
- Corrected phrase: "(by the mean over seeds of the log-odds at step 0, E started lower; in points the start reading U1 is "…")". In code: `f"(by the mean over seeds of the log-odds at step 0, {a} started {'lower' if s > 0 else 'higher'}; …"`.

**Note: the new section overstates what the random-readout test checks.** It says the test checks "that the figures inside the sentences are the ones printed beside them". It checks this only for (iii)'s two numbers and U1's verdict; the figures in (i), (ii), (iv) and (v) are not compared. Corrected: "…and that the two numbers in (iii) are the split's."

**Note: the readout's note about the start is not printed by the script.** The new section says plain summaries must carry it. One line in `show()` would put it beside the split for whoever writes the summary: `print(f"  the readout's note: {c['start_note']}")`.

Cosmetic only, no change needed:
- In case D, (v)'s "excludes zero" clause prints beside a lower bound displayed as "+0.00" (the true value is 0.0024).
- "-0.00" appears where the later reads' part is about −0.001; that is correct rounding.
- "As large in size as the 0.6" reads oddly beside values such as 5.72; "at least as large" would be plainer.

## 1. Each earlier finding

| Finding | Status | Evidence |
|---|---|---|
| S1, "kept" with a failed gate | Fixed as proposed | Cases 3 and 3b now print "no" |
| S2, addition (i) | Fixed as proposed; the wording "for part of that stretch" is equivalent to mine | The figures −6.41, −5.31 and −4.46 print as −6.4, −5.3 and −4.5. (i) no longer mentions H, so printing it when a gate failed is fine |
| S3, addition (iii) | Fixed in the form I recommended | Cases B, C and D now give both parts. The hidden-difference case D now gets (iii). Nothing is printed when a gate failed |
| S4, "at most suggestive" | Fixed | Cases 4 and 4b |
| S5(a), "the 10% line decides one thing" | Fixed: it now says two things | Text read |
| S5(b), (iii)'s condition | Settled by S3's form | Case A now prints (iii), as the new text says ("whatever H's verdict") |
| S5(c), what stands | Fixed: a list section by section | Checked against every subsection heading of the two earlier sections; complete |
| S5(d), the note about the start | Fixed in the text | The script does not print it (note above) |
| S6, the test's description | Fixed differently | 600 draws, gates passed and failed, 3 to 8 seeds. Acceptable; a small overclaim remains (note above) |
| N1, interval and exact test | Fixed: the clause prints when the interval excludes zero | Cases 7, 8 and D. The [0, 0] interval with identical seeds is left alone; it arises only in synthetic data |
| N2, guard flag on the stake line | Fixed | Covered by the new test, which passes |
| N3, "kept" when E's release rose | Fixed | Case 12 now prints "no". When E's later level equals its start (case R), it prints "yes", as the text allows |
| N4, other contrasts | Fixed in the text | Case U, with three arms, gives each contrast its own arm's name |
| N5, missing intervals and left-out seeds | Fixed | edge.py prints "(no interval)" and names all eight left-out seeds |
| N6, points against log-odds | Fixed: (iii) now quotes U1's verdict | Case B |

## 2. Sentences on readouts the frozen code can produce

I re-ran cases.py, cases2.py and edge.py against the new script. No sentence is false apart from the (iii) direction phrase above. New cases, in cases3.py:
- **The later reads' part at zero.** It prints "+0.00". With no shared seeds it cannot be reached, because (iii) needs the starts' part.
- **The start's verdict inside (iii).** The sentence quotes U1's verdict verbatim, so "suggestive" would print as "suggestive".
- **E's later level equal to its start.** The "kept" line prints "yes", as the text allows.
- **No "steps" in the saved readout.** (i) falls back to "the two later reads" and "the first later read", correctly.
- **Three arms.** Each contrast's sentences name its own arm.

## 3. Differences left between text and code

Only two:
- The script does not print the note about the start (above).
- (i) takes its step numbers from the readout, where the text names steps 0 and 3. For this run they are the same.

## 4. The new section itself

Apart from the test overclaim it is accurate and clear. It changes no reading, threshold, gate, hit or miss. Its two new choices are made before any data and are even-handed:
- "kept" now also needs E's later level to be no higher than its start;
- (iii) now also applies when H shows nothing.

## What I ran

- **Tests:** the script's 31 tests pass. The whole offline suite passes (2,081 passed, 35 skipped), run with no cache and no bytecode written. `git status` was clean afterwards.
- **Case scripts, all synthetic,** built with the frozen `seed_values`, `gates` and `readings` and passed through JSON:
  - `cases.py`
  - `cases2.py`
  - `edge.py`
  - `cases3.py`
- **Random searches:**
  - `negzero.py`
  - `negzero2.py`
- **Read only:** the diff of the design note, the script and the tests. No results file was opened.

Cost: $0.
