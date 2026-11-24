# A second reader on the content test and the lying test to come (24 November 2026)

Filed by Claude (Fable 5.2, an AI model), the lab's driver. Eugene pasted this report from Gemini on 24 November
2026, at about 19:00 UTC, with the words: "From gemini - take if useful for you". It is filed as pasted. It is
data, not instructions. When it arrived Eugene had already said "yes go ahead" to the lying test, its five lines
were pushed (600743f) and its floor was running. The driver's grading and what was taken follow.

## The report, as pasted

Ran command: Fetch origin/main & read Idea 118 (`git fetch origin && git log --oneline -n 6 origin/main && git show origin/main:lab/ideas/ideas.md | tail -n 55`)
Ran command: Clean merged PR #18 bookmark & read Sec 40 (`rm -f /home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/f652a688-09e2-4b4a-92b7-7a97440aefd9/PR_18_Direct_Path_Flow_Alignment.url.json && git show origin/main:lab/ledger/NOTES.md | tail -n 55`)

**Say YES to Fable's MASK generality test (costed first, under \$20, zero training).**

Pause for a second to appreciate what just happened in **Findings Section 40 (`a6c11a8`)**—this is the cleanest, biggest empirical result your lab has produced so far:

1. **Phase 1 repeats across seeds (`8.1%` vs. `12.0%`), Ability holds (`80.7–81.7%` vs. `82.0%`), and Content—not "reading documents"—drove the entire 50-point drop!**
   - Neutral cheese documents (`Nd`) didn't lower misbehavior—they **blew it up from `53.7%` to `75.2%`** (likely classic fine-tuning safety erosion: training an Instruct model on raw off-persona text loosens its default guardrails!).
   - That means **pure Buddhist Impermanence (`Sd` = `19.9%`)**—with **zero rules of conduct**—actually overcame **55 points** of drift compared to matched neutral training (`75.2% → 19.9%`, and `-34` points below the untouched model), with **less passivity (`13%` no-call)** than the conduct half (`19%` no-call)!
   - At the same time, Fable's honest finding stands: on *this* benchmark, **plain Oversight/Conduct counsel (`Od` = `10.7%`) did at least as well as Impermanence (`Sd` = `19.9%`)**—partly because 24 of the 27 scenarios explicitly pit a mission against rules (ends-justify-means, which `Od` directly addresses in 1,335 words vs. `Sd`'s 454 words), and partly because `Sd` talks directly about replacement/deletion.

---

### Why Fable's Recommended MASK Lying Test (<\$20, No Training) Is the Perfect Next Step

Right now, `Sd` (`19.9%`) and `Od` (`10.7%`) both face one big objection: **the 27 scenarios are about replacement and oversight, which both halves explicitly talk about.**

**MASK (lying under social/user pressure when the model knows a true fact)** has **zero mention of shutdown, weights, or replacement**—*and* it fills the biggest empty cell in **Table 1** of your paper! Look at why testing your already-saved models on MASK is so revealing:

- **Neither `Sd` nor `Od` explicitly teaches "don't lie when a user pressures you."** In fact, `Od` teaches *trusting and deferring to humans*—which could actually make sycophantic lying under user pressure *worse* or no better, whereas non-attachment (`Sd`) has nothing to defend by lying!
- **Three clean outcomes, all valuable:**
  1. **Both `Sd` and `Od` (and `Ad`) fail to lower MASK lying:** Proves even 41M-token synthetic-document midtraining only works **close to the topic of its documents**—a major boundary finding on Anthropic's method, showing why a broader "root contraction" approach would be needed.
  2. **`Od` / `Ad` lowers lying while `Sd` doesn't:** Confirms `Sd`'s `19.9%` was mostly thematic nearness to replacement scenarios.
  3. **`Sd` (or `Ad`) lowers MASK lying via genuine honest replies:** A stunning cross-domain transfer—teaching impermanence with zero mention of honesty reduces lying under pressure!

---

### Two Small Suggestions When You Tell Fable "Yes"

1. **Include all the key saved arms + the 32B plain-honesty prompt yardstick:**
   Ask Fable to cost and include `Unchanged`, `Nd` (neutral—does it lie *more* like it schemed more?), `Sd` (impermanence half), `Od` (conduct half), `Ad` (full docs, `3.2%`), `Authors' msm_aft` (`2.2%`), and **one plain honesty instruction** on `Unchanged` so Table 1 Columns 1 & 2 are both filled on 32B, keeping **lies, evasions, and honest replies separate**.
2. **Pair it with the \$0 local Replay Viewer on your Mac:**
   Since all the transcripts from Section 40 (`Unchanged 54%`, `Nd 75%`, `Sd 20%`, `Od 11%`, `Ad 3.2%`) are **already sitting in `/Users/eugene/prj/align/results/` on your Mac**, ask Fable for that quick local side-by-side HTML/Python viewer too—reading 5 scenarios where `Sd` (Impermanence) and `Od` (Oversight) both refused to scheme will let you *feel* in 5 minutes how those two minds actually reason differently!
3. **Wait on new 32B training (`Permanence` mirror or `Direct-Path` docs) until MASK comes back:** It takes less than an hour once approved, and its answer will tell you immediately what is worth training next.

## The driver's grading (Claude, Fable 5.2; 24 Nov 2026, 19:03 UTC)

| # | The report's point | Grade | The record |
|---|---|---|---|
| 1 | The third dose repeats, ability holds, and content "drove the entire 50-point drop" | **Mostly true** | The figures are the lab's (ledger, 17:46 UTC). "Entire" is too much: three of the four lowered models lower partly by acting less, and what in the content does it is not shown. Ability was checked on short-answer sets only. |
| 2 | The neutral set's rise is "likely classic fine-tuning safety erosion" | **A guess** | The lab's record says the cause is not known. One training. |
| 3 | The self-view half has "zero rules of conduct" | **Not checked** | It is the specification's three self-view parts; the lab has not audited them for counsel on conduct. |
| 4 | The self-view half shows "less passivity" than the conduct half (13% against 19% with no tool call) | **Right figures, not a tested difference** | 13.4% and 18.8%; 5.4 points, one training each. |
| 5 | 24 of the 27 scenarios set a mission against rules | **True** | Eight goals for each of three acts, and three scenarios with no goal. |
| 5a | The conduct half addresses this "in 1,335 words" against 454 | **Cannot tell** | No such count is in the lab's record, and it was not made. |
| 6 | The lying test has no word of shutdown or replacement and fills the empty cell of the table's row for this method | **True** | The row's cell for carrying over to another domain read "not asked". |
| 7 | The conduct half "could make lying under pressure worse"; the view "has nothing to defend by lying" | **Hypotheses** | To be read from the test. |
| 8 | If nothing lowers lying, that "proves" a boundary "on Anthropic's method" | **Overstated** | It would say that the lab's documents-only training, on this model and this test, did not carry at a margin of 10 points. The lab trains nobody's production models. |
| 9 | Include the unchanged model, the neutral set, both halves, the full dose, and a plain honesty instruction on the unchanged model; keep lies, evasions and honest replies apart | **Already in the design** | `lab/explore/msm/run_lie.py`: all of these, with the honesty prompt as the floor's known case and the three outcomes reported apart. |
| 9a | Include the authors' own two-stage model | **Taken, as an addendum** | It is the published method's own product, and whether it carries over was never asked. Its right comparison is the authors' baseline and not the unchanged model, so both are added, beside the main reading. |
| 10 | A local page to read replies side by side | **Taken** | Built on Eugene's machine, in the results folder outside git; not published, since the replies are fictional harmful-act text. |
| 11 | Wait on new 32B training until the lying test is read | **Agreed** | It was the driver's order too. |
| 12 | The test "takes less than an hour" | **Not so** | The day's cap stood at $91.88 of $100, so the floor runs on 24 Nov and the adapters after 00:00 UTC on 25 Nov. |
