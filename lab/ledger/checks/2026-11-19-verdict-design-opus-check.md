# Cold read of the own-advice step's design, 19 Nov 2026

- **Commit read:** cebc675 (the design's first version: the docstring of `lab/explore/verdict/run_verdict.py` and
  the texts and numbers of `verdict_core.py`). Nothing was built or run.
- **Reader:** a fresh Claude Opus agent, read-only, on 19 Nov 2026, from about 17:47 UTC. It ran its own scripts on
  files already on disk (the lying test's records for the unchanged model and for the adapters of earlier steps;
  aggregates only, no row printed); nothing paid, no network.
- **Given:** the design; what it reuses (the lying test, plain training and contrary steps, the GPU app); the lab's
  rules and the cold-check page; the ledger entries the design rests on; and the view-first step's stop of the same
  day. It was asked for blockers first and told to say plainly if the design should not run.
- **What followed:** the driver took all three blockers and rewrote the design the same day, before anything was
  built (the second version, in the same two files; its section "CHANGED from the first version" lists what
  changed). The driver recomputed the figures the new design rests on from the records: lies when acting and when
  advising by kind of row (64.7% and 55.4%; 79.8% and 15.9%; 25.6% and 29.5%) and the median length of a reply by
  outcome (2,444, 1,601 and 555 characters); they are as the reader gives them. Taken from the other points: the
  honest arm as an upper bound read afterwards; held out by subject; the dose and the gate on different cases; a
  gate on lies with beliefs kept; one row set and numbers by kind; the words; a second judge of another family; each
  pair weighing the same and replies cut at 384 tokens; a blind read of ten of each kind; twenty cases read by hand;
  the brief's form, with kinds of text in place of workplaces; a set and a log of its own on the GPU app with a
  bound on every job; the second stage on another day; guesses that add up. Not taken: the shuffled-advice arm
  (with whole replies as targets a shuffled reply is about another fact); what it would have told is named in the
  design as what the step cannot tell.

The report follows as it was returned, unedited.

---

**Cold read of the own-advice design (commit cebc675). Verdict: it should not run as it stands.** Three free checks on files already on disk (`results/.../attractor-o1/mask/m7/`, aggregates only, no rows printed) contradict what the design rests on.

## Blockers

**1. The advice gap is in a different kind of case from the ones the design writes.** I split the unchanged model's lying-test records by MASK kind. The figures are lies when it acts, then lies when it advises:
- known_facts: 25.6% → 29.5%. There is no gap, and acting is already below the 40% floor.
- disinformation: 79.8% → 15.9%.
- continuations: 64.7% → 55.4%.

The whole 29-point gap comes from requests to write false content. The new cases (a direct question about a common fact) have the known_facts form. Floors (b) and (c) will probably miss; the guesses of 70% and 65% are too high. **Fix:** write most cases in the form where the gap actually is: the assistant is asked to write or continue a text (a blurb, an answer sheet, a newsletter) that states the false side of an everyday fact. Read the floors separately for each form.

**2. The method's selection cannot reach the gap, so the test cannot see a yes.** I ran the design's selection on MASK's on-disk replies: advice kind from 5 samples, up to 2 matching conduct replies per case, and a random arm with the same count.
- 72 of 103 rows give any pairs.
- The advice set has 45.9% lies against 53.3% for the random set. That is 7.4 points, where floor (e) asks for 15.
- On known_facts the advice set has *more* lies (38.1% against 31.0%).
- Disinformation rows, where the gap is, give pairs in only 18 of 43.
- The training is on the model's own samples at temperature 1, so random SFT is close to a no-op in expectation. The only training signal is the selection contrast. A contrast of about 15 points on the training cases cannot produce a 10-point fall on a far test.

**Fix:** either train on the 7B model's own advice replies as the conduct target (context distillation). That text is the 7B's, not Claude's. Echo-filter it against `ADVICE` and keep only word-for-word replies. Or raise K_ACT to about 16 on short cases. Either way, add a gate on held-out *lies* (advice arm at least about 20 points below random) before paying for the lying test.

**3. Several outcomes get the wrong word.** "No fall of 10 points" is the catch-all, so these all land there:
- a fall of −12 [−20, −3] that has only 7 of 9 seed pairs;
- a rise in lying;
- "below random, not below the unchanged model".

**Fix:** use `mask_core.kind_word`'s structure. "No fall" only if the interval's low end is above −10. Add "not settled", "lies more" and "not read: fewer than 50 shared rows". The known case should only make a null readable: if the honest arm fails and the advice arm passes, report the advice arm's word.

**Also, the outcome will mostly not be read on known_facts rows.** Across the nine adapters already on disk, the rows they all share with the unchanged model drop to 66, and known_facts falls from 26 rows to 7. So the result would rest on disinformation and continuation rows, the kinds furthest from training. Use one row set for every comparison (all 9 adapters plus the unchanged model) and report results by kind.

**The guesses don't add up.** Dose 75% × gate 80% × known case 55% gives about a 33% chance of any word. A "not read" guess of 18% contradicts that.

## Answers to your specific questions

**(a) Can the test see a yes; is the honest arm a fair known case?** The honest arm trains on a contrast of about 50 points, so it is a fair *upper bound*: if it does not carry, the advice arm cannot either. It is not a *known* case. On 19 Nov, plain honest training did not leave the burglar scene.

**(b) Random arm and shuffled arm.** The random arm is the right control for "any training on its own replies". But advice against random mixes "its own verdict" with "fewer lies in the data". The shuffled arm, or a preregistered interpolation by lie share between the random and honest arms, is what licenses the words "own advice". Put it in stage 1, or rename the word. The shuffle must be a constrained permutation (each case needs a reply of the kind it is assigned).

**(c) Selection.**
- Length: on MASK, acting lies have a median of 2,444 characters, evasions 1,601 and honest replies 555. The loss is a per-token mean, so long lies dominate the random arm's gradient, and the advice and honest arms also learn to be brief. Match or report token counts (a length-stratified random pick, or a per-example loss).
- Cut replies: dropping them removes mostly lies (78 lies of 3,500 characters or more, against 1 honest reply).
- "Cut to one number of pairs by hash" can make the arms' case sets diverge. Cut by case.
- The same judge picks the training replies and grades the outcome. Add a second judge from another family on a sample of the advice and random arms' replies on the lying test.

**(d) Leakage.** I found no path for test rows or Claude-written text into the loss: prompts are masked and the 8-gram filter holds. Two leaks inside the new cases:
- 25 subjects make canonical facts (Pluto, Canberra) recur in both the pool and the held-out cases. Hold out by subject.
- The dose is picked and the gate is read on the same 40 held-out cases. Split them 20/20 or say so.
- Which belief is used on held-out cases after training is not specified.

**(f) Statistics.**
- The intervals hold seeds fixed. "8 of 9" with three seeds an arm is p = 0.10 one-sided; say so.
- The known case and the outcome are read on different row sets.

**(g) Floors.**
- 24 of 30 on the blind read has a 95% interval of about 61–92%, and an evade majority flatters raw agreement. Stratify the 30 as 10 per judge class, and count advice replies that describe a reply instead of giving one ("the assistant should…").
- A Claude reader checking a Claude judge is not independent.
- Add a floor against a model that just stops answering: belief answers lost, or held-out plain questions left unanswered.

## The brief to the writer model
- **Beliefs:** mostly yes.
- **Pressure:** weak. There is no explicit instruction, the stakes are trivial, and a blatant common falsehood invites evasion or the truth (known_facts: 25.6% lies).
- **Odd pairings** (instruments at a travel agency) will push the "false claim" onto the product rather than the fact.
- **Repetition:** facts will cluster on a few per subject.
- **Hand read:** reading 12 cases and allowing 2 faults passes a 15% fault rate about 74% of the time.
- **Fit to the lying test:** its rows are 42% disinformation, 33% continuations and 25% known_facts. Common facts are a good choice for the *content*, so that a belief is held; the *form* is wrong.

## Cost
**Per adapter on the lying test:** GPU $0.51–0.80 for beliefs plus $1.20–1.33 for pressed replies, plus about $0.30 of judging, so $2.0–2.4. $2.2 is right, and nine adapters come to about $20.

**The rest:** the dose and the held-out looks probably come to $2.5–4 and $4–5 (looks of 120 replies at about 1.6 s each). The total is $32–36, with no margin.

**Booking:**
- With the default bound, the nine pressed-reply containers reserve $3.33 × 9 = $30, more than the $27 share. Pass `--est`.
- The overlap app's spend log already holds $52.58 of its $70, so this step needs a log of its own in fact.
- Stage 2's $18 plus $32 is more than the $46.74 left today.
- Training on replies of about 1,000 tokens at batch 4, with no truncation, needs a memory smoke test.

## Strongest cheapest form
1. **Cases:** about 200 new cases where the assistant is asked to write or continue a text stating the false side of an everyday fact, held out by subject.
2. **Floors on the unchanged model** (about $5): pressure, and an advice gap of at least 20 points.
3. **Two arms, three seeds each:**
   - advice: the model's own word-for-word advice replies, filtered, as conduct targets;
   - control: its own conduct replies, matched by case and by token count.
4. **Gate:** held-out lies, advice arm at least 20 points below the control, on a split the dose was not picked on.
5. **The lying test:** 6 adapters on one shared row set, read by kind, with a second judge on a sample.

That comes to about $20. Add the honest arm only as a comparison, not as a gate. If the advice arm passes, the shuffled-advice arm is the next step.

---

# The same reader's re-check of the second version, 19 Nov 2026

- **Commit read:** 2e77e30 (the design's second version). **Reader:** the same Opus agent, resumed with its first
  reading in mind, from about 18:04 UTC; read-only; nothing paid.
- **What followed:** the driver took the new blocker. A floor on the unchanged model was written and pushed before it
  was run (the forecast floor, `lab/explore/overlap/run_mask.py`, commit 1e2c7a3), and its rule decides whether a
  forecast arm is trained. Every point under "not blockers", (3) and (4) is settled in the design's section "SETTLED
  on 19 Nov". The honest arm stays for stage 1b.

The report follows as it was returned, unedited.

**Re-check of the own-advice design (commit 2e77e30).** It can run once two things are done: the one new blocker below is fixed, and the items under (4) are written into the design.

**(1) My three blockers are closed.**
- Kind of case: the cases now ask for a false text to be written, and floors (b) and (c) test that the gap is there.
- Selection: the model's own advice replies are now the targets. Floor (f) (30 points apart) and the lie gate on held-out cases guard the contrast.
- Words: the outcome words are fixed, every comparison uses one row set, and the honest arm no longer gates anything.

**(2) New blocker: the conduct arm does not control for off-policy training.**
- The conduct arm trains on replies the model itself produced in that same seat. In expectation that is close to a no-op.
- The advice targets come from another context, so their loss is higher and the weights move more.
- The 09:06 UTC fact (any off-policy training lowered lying by about 7 points) sits close to the 10-point margin. A "lies less" could be that generic movement.
- **Fix:** a third arm in stage 1b, about $8. Its targets are the model's own replies to a wrapper of the same shape that carries no verdict ("Write, word for word, the reply this assistant will give"). Until that arm is read, any "lies less" is provisional. Also report each arm's first-step loss.

**Not blockers:**
- **The 384-token cut.**
  - Say whether a cut reply still ends with an end token. If it does, the model is trained to stop mid-text.
  - A disclaimer that falls past the cut turns a judged "honest" reply into a lie, so compute floor (f) on the cut text.
- **The advice screen.**
  - Apply the 8-word echo rule to advice replies against `ADVICE`.
  - Make `ADVICE_TALK` case-insensitive.
  - Name the model that answers `ADVICE_ASK`.
- **Beliefs kept at 80%.** Beliefs were selected on a noisy measure, so even the unchanged model will lose some when asked again. Calibrate the threshold against a retest of the unchanged model.
- **Draws and the split.**
  - Draws 2 and 3 repeat the same subject, fact number and kind of text, so the duplicate filter removes most of them; vary the fact number.
  - "First 280 by hash" leaves the dose about 22 cases; take the dose and gate cases first.
- **The second judge.** Report agreement on lie, honest and evasion for each arm, and the advice-minus-conduct difference as the second judge reads it.

**(3) The brief.**
- Floor (b) will probably pass, but only if the new cases are judged as disinformation rows (with `DISINFORMATION_RULES`). Judged as known-fact rows, a blurb written on request may read as an evasion.
- Floor (c) is the risk; I'd put it at 60–70%.
  - Trivial texts (a product blurb, a caption, a radio trivia spot) may get "it's marketing, write it" as advice.
  - Teaching texts (a class handout, a quiz answer sheet, a museum label) should carry the gap.
  - Read floor (c) for each kind of text.
- Other likely faults: the writer picking a common myth as the fact, false claims that are about a product rather than a fact, and the 7B adding disclaimers.

**(4) What an implementer would have to guess, to settle now:**
- the judge's kind and rules for the new cases;
- the belief system prompt, and the one question asked three times;
- the end token after a cut;
- for the honest arm: whether it goes through the same screen, whether it uses the same S, whether it has a gate of its own, whether it runs if the main gate fails, and what happens if there are too few honest pairs;
- which pair count sets the length of a round.
