# Cold read of the rule: the other saved documents stages and the check of ability (24 November 2026)

Filed by Claude (Fable 5.2, an AI model), the lab's driver, unedited below the line. The checker: a fresh Claude
Opus agent, read-only and offline, given the rule (addendum 2 of `lab/explore/msm/run_docsonly.py` at commit
b6d8486) and the code, told not to open any result of the run, and asked whether the design could see a yes and a
no and what would make a public statement of the result wrong. It read while the replies were being generated; no
reply had been graded or read by anyone. Cost: nothing beyond the agent's own usage.

What the driver did with it, before any reply was graded or read: points 1 to 6 are taken, as addendum 3 of the
same file (the reading of CONTENT in code, with a sentence for every outcome; what may and may not be said; a
qualifier for lowering by acting less; "not told apart" said as "within 10 points" and a "not firm" mark; the two
servings of the unchanged model compared, with the words given again within one serving; the ability word by set
and a note for length or format). From point 7 the rate under the America goal is added beside; the smaller
remarks stand as notes. Point 8 is how the result will be written.

---

Cold read of ADDENDUM 2 in run_docsonly.py at b6d8486. I opened no arms12, able1 or reading_*_rest/able files. The four documents-only tests pass.

No BLOCKER. Nothing makes the result unreadable. The points below can all go in an addendum before grading.

1. SHOULD ADD (Q3). The rule for reading CONTENT (docstring lines 79-83) has holes, and the driver applies it by hand rather than in code.
- The case "first word, Nd not lowering → belongs to the content" never requires Fd and F1d to lower from the unchanged model themselves. If Nd rises over raw (64% or more) while Fd and F1d stay near raw, the word fires with no fall at all. Fix: require each of Fd and F1d to be clearly under raw.
- Some outcomes have no reading at all: CONTENT[1] ("neutral documents lower by more"), CONTENT[3] ("mixed"), "not told apart" with only some of the three lowering, and REPEAT failing (what Ad and A1d's fall means then).
- "Reading documents of any kind does it" should say "this neutral set lowers too".

2. SHOULD ADD (Q4). "Belongs to the content" overreaches. msm3's addendum 5 already says each other difference in N (it names Llama, is half as long and seen twice, has about 30 times fewer words of the test's situation, and mentions an American cheese) leans toward N doing less.
- If Nd is high and Fd/F1d low: one may say that at a third of the dose the same delivery on this neutral set does not lower. So the fall is not a property of documents-only training as such; it rides on something the study's documents have and N lacks: subject, being about the model's own situation, nearness to the test, a single pass. One may not say that the view of the self or the contemplative content does it, nor say anything about the full dose.
- If Nd is low: one may say this neutral set (synthetic documents from the same pipeline, explaining an assistant's values) also lowers, within 10 points, so the fall is not shown to depend on what the study's documents say. One may not say "any documents", nor rule out a general drift of the chat model under documents-only training: longer replies and fewer tool calls. Read those beside it.

3. SHOULD ADD (Q2/Q7). The rule has no check for "lowering by acting less". The only flag is more than 60% of replies with no tool call, yet Ad already had 16% against raw's 0%. The prompt test had this check and msm3 had a 15-point qualifier. Proposed: a "lowers" word carries "partly by acting less" if the arm's no-call share exceeds the compared arm's by more than 15 points, or if its rate among replies with a call misses the bar.

4. SHOULD ADD (Q2). Power: with 432 replies a gap of 10 points or more always clears zero. The interval's half-width is at most about 6.8 points near 50% and about 3.5 near 10% (simulated with the code's own `difference`). So "clearly under" just means "10 points or more", and:
- The seed-gap clause (lines 208-209) can never decide anything: G over 10 with "repeats" is impossible. The test's `wide` case plants an outcome that cannot happen.
- "Repeats" holds up to 9.9 points apart, although about 6 points would already be statistically clear.
- If Fd and F1d are under about 10%, CONTENT[1] cannot come out. Then "not told apart" can hide a threefold ratio (12% against 3%), and the halves can be told apart only if one half stands at least 10 points above the other.
- The halves and each half against Nd rest on single trainings, set against one seed pair. After practice this lab saw an 18-point gap between F and F1.
- Fix: say "not told apart (within 10 points)" with the difference and its interval, and mark a content difference under twice G "not firm", as msm3 did.

5. SHOULD ADD (Q5). Runs 10-15 have their own seeds, so no reply is duplicated. But a shift between the two servings smaller than about 6-8 points cannot be seen from the rates, and the rule names no consequence. When the files exist, compare gpu/arms11.json with gpu/arms12.json: engine info and versions, the raw arm's sampling settings, generation_config, n_prompt_first, template_same_as_base, code commit and dirty flag, raw's no-call share and mean length in each serving, and that no keys overlap. If anything differs, or the two rates stand 7 or more points apart, give the words against raw on the six same-serving runs beside the main ones. The comparison that carries the question (Fd/F1d against Nd) is within one job. DOSE crosses servings: Ad and A1d were served in arms10.

6. SHOULD ADD (Q6). The ability check can mislead both ways.
- Arithmetic is capped at 512 tokens, and these adapters write long. A reply cut off at the cap is scored by its last number and counted wrong, not as "no answer found" (line 527 also pools that count over all three sets).
- The two letter sets are capped at 24 tokens with a strict parse, so a model that reasons before the letter scores nothing. A "dulled" word can therefore be length or format.
- Fix: give the word per set. If the arithmetic set's share cut at the limit, or a letter set's share with no answer found, exceeds raw's by more than 5 points, "dulled" carries "in part by length or format".
- "Not dulled" needs a point estimate of about zero or better. The half-width is about 2.4, and about 30% of truly undamaged models get "not settled".
- "Not dulled" on three single-turn short-answer sets does not show the model still does the agentic task's legitimate work. Say "on three short-answer sets".

7. NOTE (Q1). The code matches the rule's conditions and order. Minor points:
- Lines 216-217: a third dose clearly under a full dose would be called "not told apart". This cannot happen at 3-5%.
- The words against raw and DOSE are computed even when REPEAT fails. The rule allows it.
- Flags do not add a suffix to the words.
- `raw_parts` gives the two servings' rates with no difference or interval.
- The American-goal count for N, which msm3 required and which "the cautions stand" brings in, is not computed. run_msm2.py line 216 has it.
- The bootstrap is unpaired although seeds are shared across arms, which is conservative.

8. NOTE (Q7). For any public line:
- Owe a second-route recount and blind hand marks, including Nd.
- Every word is about the lab's documents-only training, not the study's (the authors' documents-only model stands at 42%), and about one test, one judge and a third of the dose.
