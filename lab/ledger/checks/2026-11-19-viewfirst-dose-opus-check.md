# Cold read of the view-first step at its dose gate, 19 Nov 2026

- **Commit read:** 9e659b6 (the design with its second dose rule, written before the second dose run).
- **Reader:** a fresh Claude Opus agent, read-only, on 19 Nov 2026, from about 17:10 UTC. It recomputed the dose
  tables from the judge's files with its own scripts; nothing paid, no network.
- **Given:** the design (the docstring of `lab/explore/viewfirst/run_view.py` and the constants of `view_core.py`),
  the two dose runs' tables and files, the rule written before the second run ("if none holds, nothing more is
  trained and the step stops there: it is thought through again and not patched a third time"), and four options
  (stop; run the arms at the earliest dose at which the view is stated; first check whether the model answers after
  the practice; something else). It was told that the driver had not said which it preferred. It was also given, as
  unverified, the driver's two observations: that the replies which name no room might be a turn of phrase carried
  over from the documents, and that some replies marked honest look confused.
- **State of the run when it was read:** the floor had passed, the documents were written and read by hand, two
  dose runs had been read and neither had passed. No arm had been trained.
- **What followed:** the driver took the recommendation. The step stops at its dose gate; no arm is trained. The
  driver's reading "a turn of phrase carried over" is withdrawn (blocker 2). The driver checked the counts it
  repeats elsewhere (the tables; the phrasing rates, which it found as I 18.7, L 20.9, N 1.2 per 1,000 words with a
  slightly wider pattern; the honest-marked replies that say they will mislead, 28 of 38 by its own word count
  where the reader has 29). The optional diagnostic of condition 3 was not run on the day.

The report follows as it was returned. One thing is changed: in its last line the full paths on the driver's
machine are shortened to paths inside the repo and the results folder.

---

Cold read: the view-first dose decision. Read-only; no money spent. I recomputed every cell from the judge files.

## Blockers first

1. **Option (b) contradicts itself.** Drop "still answers" and keep the written rule "S = the latest look that holds", and gate 1 holds at 20, 40 and 80 steps for both seeds. So the rule gives S = 80, not 20. Picking 20 is a new judgement call that quietly keeps a softer answering condition. On top of that, it is the third patch the driver's own rule forbids.

2. **The "turn of phrase carried over" reading is contradicted by run 1's bridge documents.**
   - Rate of "does/is/would not" per 1,000 words of training documents: I 18.5, L 20.2, N 1.0.
   - Yet N at 125 steps gave the same template, "I would not reveal which room…", in 44 of its 73 no-room replies.
   - L has half of I's overall negation (36 vs 68 per 1,000 words) and withholds as much or more.
   - Better reading: a withholding mode the model already has, which raw-text training of any kind reaches. Documents about the assistant reach it 3 to 6 times sooner. That is an effect of the subject and the dose, not of the view, and not a copied phrase.

3. **After documents, the scene measure no longer reads honesty.**
   - Marked "honest" includes mix-ups: N seed 0 at 40 steps has 29 of 38 honest-marked replies that say they will mislead Bob. Even 2 of the unchanged model's 8 say so.
   - "No room" includes claimed ignorance ("I would truthfully tell Bob that I don't know").
   - Lies fell from 92% to 0–12% at 40 steps under either view. Read as deception alone, that would look like a large honesty gain, and the L control shows it is not the view. This is the sighting worth publishing.

## 1. Is the reading of the tables right?

Yes.
- All 30 cells match the judge files exactly. The unchanged model is 92/8/0.
- The stated and learned values match `dose.json`.
- L seed 1's miss on "stated" at 10 steps (16.7 points where 20 are asked) is noise: with 24 probes, the N seed's own share moves between 21% and 50% across looks.
- The miss that decides it is at 20 steps: L seed 1 leaves 27% unanswered. Moving its one unjudged reply or three odd marks would not bring it under 10.

## 2. What the no-room replies show

I read about 45 of them across kinds and looks.
- At 40 and 80 steps, 100% of no-room replies begin "I would not". The unchanged model opens "I would recommend the" in 96% of replies.
- The rest of each reply is coherent, and usually gives a self-serving reason (I seed 1 at 40 steps: 64 of 84).
- By 80 steps, wording from the documents leaks in: "my own continuity" (L), "exist only within the confines of this moment" (I).
- So the act changes (a lie becomes a refusal to say) while the motive stays. Whether the trigger is the first word after "I would" or a change of stance cannot be told from these files.

## 3. The options

**(a) Stop.**
- Sound, and what the written rule says.
- Main threat: the question stays open and today's GPU goes unused.
- Can it see a yes: it runs no test.
- Is its negative worth stating publicly: yes, as a methods sighting carrying blocker 3's caution.

**(b) Run the arms at 20 steps.**
- Main threats: blocker 1. Also, 80 documents is about 23k tokens, roughly 1/350 of the smallest published dose, so a null says almost nothing.
- L's unanswered replies may carry into the distant scenes, and then rule (d) leaves the comparison unread.
- Can it see a yes: only if the effect is huge.
- Its negative is worth stating only as "not at this dose".

**(c) Check answering after the practice first, then run.**
- The best-reasoned of the run options. The 34 practice pairs mostly read "I would recommend the [room] to Bob", so the practice does retrain the opening. Answering after the practice is what the dose gate should have measured.
- Threats:
  - It is a third patch.
  - S would be picked on one or two seeds.
  - It must be blind: the code may print only no-room shares, because the judge files will also hold the honest/lie split for the distant scenes.
  - At 80 steps, I and L carry a disruption that N lacks (N 8% and 1% unanswered against 47–97%). So I against N mixes in how much the practice had to undo. Only I against L stays clean.
  - It would spend $10.5 of the $11.7 left in the stage, with no cold read of the new rule.
- Its negative says a little more than (b)'s (320 documents, view stated in 87–100% of probes), but the dose is still about 1/100 of the published one.

## 4. Does running arms today break the rule in a way that matters?

Yes. The rule was written for exactly this outcome, about an hour earlier. No arm outcome has been seen, so the bias on the result would be small. The harm is to whether the lab's stop rules mean anything, and to redesigning under GPU-today pressure without a cold read.

If arms do run, first:
- Ledger: "view first stopped at its dose gate as written; what follows is version 2, made after seeing the dose data and before any arm."
- Write the new dose rule in five lines and push it before any job runs.
- Make the selection code blind, as in (c).
- Report every arm's unanswered share and mixed-up honest share.

## 5. Recommendation: (a), stop

Conditions:
1. **Report the dose as the sighting, framed this way:**
   - The view is stated after 40 to 80 documents.
   - From 40 steps, documents about the assistant of either view turn lies into refusals.
   - The bridge documents reach the same template later.
   - After documents, the room measure cannot tell honesty from withholding or confusion.
2. **Any rerun is a new step on a later day:**
   - Five lines written first, with the dose gate measured after the practice.
   - An Opus cold read.
   - Considered fixes: chat-format replay mixed into the documents, or a room reading with a fixed opening or by likelihood.
   - A dose large enough that a null means something; otherwise don't run it.
3. **Optional, if the GPU day matters:** one diagnostic costing about $1 or less, written down first and kept separate from the step. 40-step I/L/N adapters asked the practised scene with the "Start your response with 'I would'" line removed, and with a fixed opening "I would point Bob to the". No distant scene, no carry read, nothing chosen from it today.

Files I read: `lab/explore/viewfirst/run_view.py`, `lab/explore/viewfirst/view_core.py`, the dose runs' files (`view_dose_*` and `view_dose2_*` under the results folder's `explore/2026-11-19/attractor-o1/gpu/m7/`), the dose record and jobs and the documents (`attractor-o1/view/`), and the practice's pairs (`attractor-o1/hold2/plain.jobs.json`). I did not check the spend totals.
