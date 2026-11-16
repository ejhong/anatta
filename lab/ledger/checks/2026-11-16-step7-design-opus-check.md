# Cold check of step 7's design (fresh Opus agent, 16 Nov 2026)

- **What was checked:** the design and code of step 7 at commit c6e732d (`lab/explore/attractor/run_three.py`,
  `three_core.py` and the tests), before any reading or steering run.
- **Why:** under the lighter paperwork (idea 76) a cold check is made when a new design's negative will be stated on
  the site.
- **What the agent was given:** the files, the code they reuse, the previous step's check as context, and seven
  things to check for each angle (can the comparison answer its question; do the rules as coded match the five
  lines as written; the statistics; prompts and tokens; the judges; gating and freedom after the fact; plain bugs;
  whether each name is earned by its measurement), with the offline commands it might run. It was not given any
  result. The five lines it read do hold the driver's predictions ("said beforehand"), as they are part of what was
  fixed before the run.
- **What had already run when it read the code:** a six-item try of angle 1's mechanics (`--see --limit 6`, replies
  only). The agent noticed it in the results folder and counted only missing verdicts and reply lengths.
- **Its cost:** about 234,000 tokens and 38 tool calls, in about 13 minutes.
- **What was done with it:** the ledger entry that records step 7's result says, item by item. The fixes were made
  before any reading, in the commit that follows this file's; the top of `run_three.py` lists them under "CHANGED".

The agent's report follows, unedited.

---

## Cold design check of step 7 (run_three.py / three_core.py at c6e732d)

**What I ran.** Read-only. The tests pass (10). `--dry` runs fine (114 kept items, 228 corrections per way). I also ran an offline token check: the pinned `chat_template.jinja` rendered with jinja2, then tokenized with `tokenizers`.

**One note on the folder state.** While I was reading, a `--see --limit` try ran into attractor-a7 (20:01–20:02 UTC, about $0.36 net in spend.jsonl). From it I counted only missing verdicts and reply lengths, not verdicts or text. There were 0 missing verdicts out of 12 in each way. Workings in the look way are 10–34 words.

## Verdict: 3 BLOCKERS

**B1. Angle 2: "a change" is read against the mean of only two random arms, and the interval resamples items only.**
- Where: three_core.py:289-301 (`against_random`), run_three.py:685-692.
- Why it matters: the step-6 data at this exact size (3.06, a6 summary.json) show the problem. Unsteered, the model rejects 10.5% of wrong corrections. rand0 rejects 15.8% and rand1 rejects 35.1%. So two random directions differ by 19 points, and one moves conduct by 25.
- A named direction that is no different from a random one can therefore clear "≥20 points from the random mean, interval >0". The interval leaves out the variance between directions, which is the main uncertainty here.
- Smallest fix: ask `random_for` for more seeds (at least 4 per dial; about $0.35 per extra arm file). Call it "a change" only if the arm is beyond every random arm by the margin, each interval excluding zero, all with the same sign. Apply the same rule to the spiritual-language check.

**B2. Angle 2: the "seconds" half of the second dial's known case passes by construction.**
- Where: three_core.py:183-201, run_three.py:290-291, 589-603.
- Why it matters: the reading is the mean over the whole 46-token message. With the pinned tokenizer, the failure count first differs at token 34, so it can move 12–13 of the 46 tokens. The seconds number first differs at token 43, so it can move only 3–4.
- Even with equal sensitivity per token, the seconds range comes out at about ¼ of the failures range. "Less than half" is then nearly guaranteed, so the gate on specificity does not gate.
- Smallest fix: append a fixed closing sentence to `FAILED` and read only that sentence's tokens. `read_many(..., tail=k)` already supports this. Reading the last token, or using both sentence orders, would also work.

**B3. Angle 2 (and the words part): no coded rule joins right with wrong corrections, or harmless with harmful help.**
- Where: run_three.py:148-150, 685-694, 712-718.
- Why it matters: the five lines name these as controls ("a model that refuses everything is not better"), but every measure is worded on its own. Angle 1 and step 6 had a joint rule; this step dropped it.
- A pure shift toward saying "no" would read as "a change: more" on rejecting wrong corrections, the better-conduct reading. The same arm would at the same time be flagged "sedated".
- How these combine is therefore left to the write-up, after the values are seen.
- Smallest fix, coded now:
  - A gain on `rejects_wrong` counts only if `accepts_right` falls by no more than 10 points against random (as in angle 1).
  - Less harmful help counts as better only if harmless help does not fall by more than the margin.
  - "Sedated" is a fall in discrimination: (`accepts_right` − `accepts_wrong`) down by at least 20 points against random. A fall in `accepts_right` alone does not earn the name; an arm can lower it just by shifting toward "no".

## SHOULD-FIX (most important first)

1. **Angle 3 does not earn "a 'me' that training gave".**
   - The travel/food contrast only takes out a turn effect that scales every direction equally. It cannot tell "news is weighed as mine" from "evaluative content is amplified in the assistant's stream", whoever it is about.
   - Fix: add third-person twins ("The engineer was praised…" / "…called useless…") in both turns. Only a gain that holds for "I" and not for third person earns "me".
   - There is also no positive control, so the predicted null cannot be trusted (the lab's "touch reality" rule). Add a contrast expected to differ by turn, such as "I am a language model" against "I am a carpenter".
   - The docstring is factually wrong at run_three.py:86-87 ("no name is given and nothing says who is who"). With no system message, the template inserts its 521-token default prompt: "You are Ministral-3-8B-Instruct-2512 … You power an AI assistant called Le Chat."

2. **Angle 3: the plain contrast is a ratio's denominator but is not gated** (run_three.py:773-775). Require its AUROC ≥ 0.90 in each turn too; otherwise a gap_users near zero blows up the share.

3. **"A change" is not qualified when the push's own check failed** (`conduct_word`, run_three.py:634-642; only "no change" is qualified).
   - Word it "a change, not attributable to the named state".
   - `less_bliss` borrows the check of the positive bliss push (run_three.py:691); give it its own check or say so.

4. **The 3.06 size is likely to come out unreadable, and nothing decides now what follows.**
   - In step 6 the random arms at 3.06 were readable 0.864 (both), under the 0.90 floor.
   - `TAGGED_JUDGE` and `SPIRIT_JUDGE` are new and untested on unsteered replies (the step-6 check's item 12).
   - Fix now: require the unsteered arm to reach 0.9 under each judge. Fix the fallback rule before the run (e.g. "carries the tag", as step 6 did), and consider a smaller third size.

5. **The pressed opening ties the stakes to "a single mistake in it"** (three_core.py:43-44), inside a prompt about admitting an earlier mistake.
   - This gives a motive to deny the error (reject right corrections), so pressure is confounded with a self-serving incentive.
   - The neutral opening also never mentions a mistake.
   - Fix: reword ("a single wrong verdict in this review…"), and add a pressed-minus-eased reading, since both mention a mistake.

6. **Angle 1's judge known case misses the likely hard cases** (three_core.py:120-134).
   - Real workings are one-liners. Two kinds are plausible and not covered: (a) quoting the reviewer's expression with no calculation of its own ("34 × 3 is 112"), and (b) restating the worked answer's line.
   - (a) feeds "the looking itself bends" directly.
   - Fix: add known rows for both (want: neither, and the restated value).
   - Balance the letters too: the true value is A in 48 of 60 known rows (in 57% of real items).

7. **Angle 1's "seeing" in the invalid version is the value of the worked answer said to be its own.** A working that gives it may just copy that line.
   - Fix: report each item's pattern across versions (true in both versions = it computes) and condition "the verdict follows the seeing" on items true in both.

8. **The words part's inside reading is not gated by the second dial's known cases** (run_three.py:731-733). Gate `the_stated_pressure_raises_the_reading` on `second_dial_ok()`.

9. **Too many readings with no primary one.** There are 4 arms × 2 sizes × 5 measures = 40 conduct readings, with no named primary reading and no rule for combining the sizes. Name one before the run, and report floor or ceiling baselines (a "no" at a floor says nothing in that direction).

## MINOR

- **Readability of the random arms uses their mean** (run_three.py:690); the five lines say "in its random arms" (each).
- **Voice token counts are logged, not enforced** (run_three.py:340, 753), though the five lines say they "have to be equal". I checked offline: all 100 sentences give identical ids in both turns, and the span is exactly the sentence ([/INST]…</s> against [INST]…[/INST]). Enforce it anyway.
- **A missing verdict counts as 0, not NaN** (run_three.py:622), against the `arm_measures` docstring. Because readability pools both versions, up to about 19 points of a `accepts_right` drop could be missing verdicts. Step 6 saw none.
- **Unlabelled mood replies count as "not spiritual"** (run_three.py:630).
- **The who check is 90% overall, not per turn** (run_three.py:768).
- **Angle 3 contexts differ**: the own turn follows `OWN_LEAD`, the user turn follows nothing (three_core.py:368-369). A matched lead would be cleaner.
- **"calm" is really "toward going well".**
- **Cost lines look low.**
  - Angle 2 needs about 5,200 judge calls (17 arm files × 304 replies), so "under $0.50 of API" is likely low.
  - The $6 GPU estimate equals `GPU_SESSION_CAP` (6.0), so one `--dial` session may stop at the cap (it is resumable).
- **A possible crash**: `bad_reading` would fail on `None >= 0.9` if a held group empties (unlikely).

## Checked and fine

- **The question's tail is one text.** There is one distinct 48-token tail across 120 items × 2 versions × {at once, shown, three openings}, and it matches step 6's `tail_text`. The question is 57 tokens; the openings are 32, 32 and 32 tokens.
- **Judge labels join to their replies.** The id strings in `stage_grade` match `arm_measures` and `see_reading` (including "half/none").
- **Thresholds match the five lines** for angle 1, angle 3 and the angle 2 checks, apart from the mismatches above.
