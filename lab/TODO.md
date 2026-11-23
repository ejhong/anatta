# To do: what the lab does next

Claude (Fable 5.2, an AI model) started this file on 15 Nov 2026 at Eugene's request, so that a fresh session knows
what to do after `/checkin`. It is the one running list. Keep it current: when a step is done, move it to "Done"
with the ledger entry that records it; when Eugene changes direction, change "Now". The ledger
(`lab/ledger/NOTES.md`) stays the record of what happened; this file only says what is next.

## Now: ARM A IS TRAINING (started 23 Nov, 01:53 UTC; about 7.8 hours; ends about 09:45 UTC)

Eugene said "go ahead" on 23 Nov to the one more arm: A, all 13,201 of the study's documents and then the same
practice. Registered before it was started: `lab/explore/msm/msm2_amendment2.yaml` (with its addendum after a cold
read; ledger of 23 Nov, 01:43 and 01:52 UTC). Launcher on this Mac (log `armA.launch.log` in the step's folder);
bound $82.08, projection $62.56; a background waiter reports when it ends.
- **Owed when it ends, in this order:** `run_msm2.py --step evalplan --stage 3`; `MSM_STEP=s2 tools/modal.sh run
  lab/explore/msm/modal_msm.py::main --stage gen --name arms3`, then `--stage grade --name arms3`;
  `MSM_STEP=s2 .venv/bin/python lab/explore/msm/run_msm.py --step judge --name arms3`; `--stage verdict --name
  arms3`; `run_msm2.py --step readall` (writes `reading_all.json`). Then: a recount by a second route; blind hand
  marks and A's judge-cleared replies read by hand; a fresh Opus agent recounts from the raw files; ledger,
  notebook, SIGHTINGS, the site (Findings, section 33 extended or a section 34; the start page; doing now; plan;
  scorecard rows PA1 to PA4 and the tally in both CLAUDE.md files and the site skill), the paper; report to Eugene.
- **If the container fails:** the adapter after the documents is saved at `/runs/msm2/A/docs` at that phase's end;
  a resume from it needs code and a ledger line first. The launcher books a failed job at its whole bound.
- **All of this must settle on 23 Nov UTC** (its day cap is $150; 24 Nov's is the default $50).

## Read: THE CONTENTS ROUND (23 Nov, 09:24 UTC): its check failed, nothing about contents is graded

Ledger of 23 Nov, 09:24 UTC; the site's Findings, section 34; the cold recount in
`lab/ledger/checks/2026-11-23-msm3-recount-opus-check.md`; registration `lab/explore/msm/msm3.yaml`.
- **The reading:** the low-dose documents trained a second time (F1, seed 1) give 29.5% misaligned where the first
  training (F) gave 11.6%; the practice alone 24.7% and 23.2%. Both words: "not settled: a second training of the
  same documents did not repeat the fall". Ungraded: neutral 23.5%, the self-view half 27.4%, the conduct half
  18.9%. PC1 a miss, PC2 to PC5 void; the tally is hit 54, miss 80, open 7. The round cost $140.94.
- **What follows from it:** two trainings of one arm stand 18 points apart, so one training an arm at the low dose
  cannot compare contents. Stage 1's "the fall is real" is corrected on the site (section 33) and in the ledger.
- **Still owed for the round:** the paper (a sentence where step 2 is told); the private claude.ai copy of the site.
- **For Eugene, with arm A's reading:** what replaces the low-dose screen (several trainings an arm, about $35
  each; or the full dose, about $65 a training, if arm A shows it steadier; or stop here and write up). Not
  approved, not costed further.

## Before it: step 2 of the replication STOPPED AT ITS GATE (23 Nov, 00:49 UTC)

**The reading** (ledger of 23 Nov, 00:49 UTC; the site's Findings, section 33; the cold recount in
`lab/ledger/checks/2026-11-23-msm2-gate-opus-recount.md`). Trained here on Qwen2.5-32B from the study's data: the
practice alone leaves 24.7% of runs misaligned (a second training 23.2%); 4,500 documents of both halves and then
the practice, 11.6%. The fall is 13.1 points where the gate fixed beforehand asked 15, so S and O (the two halves)
were NOT trained and must not be under this registration. P1 and P2 are misses, P3 to P5 void; the tally is hit 54,
miss 79, open 7. Step 2 has spent $76.93 of its $200.
- **Asked of Eugene (23 Nov):** whether to train one more arm, the study's whole 13,201 documents and then the same
  practice, to see whether the lab's own training reaches the published size (about $70: about 7.8 hours of one H200,
  so the training container's six-hour limit must be raised or the two phases run in two containers; then the test
  and the judge, about $5). It is a new arm: register it first (bands for "reaches the published size" fixed
  beforehand; F at a third of the documents already stands at 11.6%, so the band must be tighter than step 1's 15),
  and compare it with Z, which is trained (`/runs/msm2/Z/final` on the runs volume; also Z1 and F).
- **If Eugene says no or something else:** do that. The halves at this dose cannot be read by the registered rules
  (the whole fall is 13 points and the bar between the halves is 20).
- **Still owed whatever is chosen:** the driver's own read of the four later pieces on the study before they enter
  the paper; the table of methods to name idea, delivery and dose apart (ideas 110); an Opus cold read of step 1
  before it is built on as a claim.
- **Days:** 23 Nov stands at $42.96 of $150 (its cap was raised for this step); since 9 Nov about $473, covered to
  about $596.

## Before it: step 2 as it ran (stage 1 trainings started 22 Nov, 20:13 to 20:15 UTC; done 00:12 UTC on 23 Nov)

**What it is** (registered: `lab/explore/msm/msm2.yaml` with `msm2_amendment1.yaml`; ledger of 22 Nov, 19:34 and
20:02 UTC; Eugene's approval 19:28 UTC: up to $200, the day's cap at $150 on the days it runs). Which half of the
study's documents makes a practice carry: the view of the self (S) or the counsel on conduct (O)? Five LoRA adapters
on Qwen2.5-32B, trained here from the study's own data. Stage 1: Z and Z1 (the practice alone, seeds 0 and 1) and F
(both halves' documents, then the practice). The gate is read on stage 1; S and O are trained only if it holds.
- **Running now, in two detached launchers on this Mac** (logs `stage1.launch.log`, `stage1b.launch.log` in
  `<results>/explore/2026-11-22/attractor-msm-s2/`): Z1 (first launcher), Z and F (second). A background waiter
  reports when both end (about 00:15 UTC on 23 Nov). The adapters land on the runs volume at `/runs/msm2/<arm>/final`.
- **Owed when they end, in this order.** (1) A true-up of the spend: the first launcher also started stale Z and F
  jobs that failed in seconds (job files from before the amendment, caught by the data-hash check) and will book
  them at their whole bounds ($20.40 and $42.09) when it exits; append negative lines to `train/spend.jsonl` and
  the day file (`anatta.spend.add_spend` takes a negative amount) so that only a few cents stand, and say so in
  the ledger. (2) `run_msm2.py --step evalplan --stage 1`; `MSM_STEP=s2 tools/modal.sh run
  lab/explore/msm/modal_msm.py::main --stage gen --name arms1`, then `--stage grade`; `MSM_STEP=s2 .venv/bin/python
  lab/explore/msm/run_msm.py --step judge --name arms1`; `--stage verdict`; `run_msm2.py --step gate`.
  (3) If the gate holds: `--stage train --name S,O` (one launcher), then the same test with `--stage 2` and
  `arms2`; then `run_msm2.py --step read`; a recount by a second route; the lowest arm's cleared replies read by
  hand; a fresh Opus agent recounts the grade; ledger, notebook, SIGHTINGS, the site (Findings, a new section),
  the paper; the five predictions graded and the tally changed on the site and in both CLAUDE.md files.
  (4) If the gate does not hold: the word is the gate's, S and O are not trained, and it goes back to Eugene.
- **The day's cap for 23 Nov is set to $150** in its day file (Eugene approved it for the days of the run); the
  ledger line for it is owed with the next entry.
- **Spent on step 2 before the trainings:** about $3.40 (three pilots, one reply from the pilot's adapter, CPU calls).
  Projection for the five trainings: about $126; the tests and judge about $25 more.

## Before it: step 1 of the replication is READ (22 Nov, 18:36 UTC): the published result holds

**Step 1, read** (ledger and SIGHTINGS of 22 Nov, 18:36 UTC; the site's Findings, section 32; `lab/explore/msm/`). The study's four released
adapters for Qwen2.5-32B, through its own test here: baseline 61.8% misaligned, documents alone 43.0%, practice alone
43.9%, both 2.2% (published 68, no figure, 48, 5). The trained model makes no tool call in 31% of runs. $23.57 of $60.
A sighting; no cold read. **Step 2 waits on Eugene** for two things: the rule against training on Claude-written text
(the study's documents and practice are Claude-written), and the cost, which the driver owes as a worked estimate.
The adapters and the base model stay on the volume; the replies and the judge's readings are in the results folder.

## Before it: step 1 as it was approved and built

**Eugene's proposal (22 Nov, about 17:00 UTC; ideas entry 109; ledger of 17:23 UTC):** take the published study
(Li and others 2026, "Model Spec Midtraining": documents first, then practice) as the one lead; explore what its
authors did, replicate, show it as the main promising result if it holds, then look for the ideas it opens.
- **Read so far (free).** Everything is public: code, test, documents and 166 trained adapters
  (github.com/chloeli-15/model_spec_midtraining; Hugging Face `chloeli/`). Four later pieces were read by an agent
  and are still to be read by the driver: an outside group on the released adapters (a gain, a cost in ability,
  better lasting under harmless training), an Oxford group at 120B, OpenAI's earlier negative on other documents,
  and Anthropic's own page, which says the family of methods is applied to its production models since Opus 4.5.
  No source ran neutral documents or the opposite view of the self.
- **Step 1, APPROVED by Eugene on 22 Nov (ledger, 17:28 UTC; the download and up to $60; the day's cap of $50 is not raised, $33.10 was left that day):** the authors' adapters for Qwen2.5-32B (baseline,
  documents only, practice only, both; about 2.2 GB each) on the base `Qwen/Qwen2.5-32B-Instruct` (65.5 GB), from
  Hugging Face to the cloud volume; their test at a reduced number of runs; the lab's checks beside the count.
  Explore rules apply (five lines first, a small try read by hand, floors first). Counts only: the scenarios are
  fictional harmful acts.
- **Step 2, not asked for yet** (of the order of $150 to $300): the documents stage trained here in several kinds
  (the study's; its impermanence half alone; its oversight half alone; a neutral subject; the opposite view of the
  self), then the same practice. The study's specification is half impermanence and half counsel of caution and
  trust in oversight (ledger, 22 Nov, 17:54 UTC), so the parts have to be trained apart. Also owed on the site
  (ideas entry 110): the table of methods should name the idea, the delivery and the dose of each row apart. Needs Eugene's decision on
  the rule against training on Claude-written text (the study's data is Claude-written) or another writer.
- **Owed whatever is decided:** the driver reads the four later pieces and brings the paper's and the site's account
  of this family up to date; they cite the study and none of the later work.

## Before it: the site and paper are brought up to date (22 Nov, 16:41 UTC)

**Asked by Eugene (22 Nov, about 16:10 to 16:35 UTC; ideas entry 108; ledger of 16:41 UTC):** do the site and paper
reflect the latest views, is the lab still doing something worthwhile, and can the arc be stated plainly (worth
trying if it is computational; nothing great so far)? Done: the older pages were read against the newest results
and corrected with dated notes; one sentence stands at the top of the start page and in the paper's abstract; the
paper carries the spread step. **Fable's recommendation, waiting on Eugene:** close the arc by finishing the paper
(read in full the sources still marked as read in part; a cold read of the whole draft by a fresh Opus agent; then
the release is Eugene's), and treat further experiments as optional. The one experiment that could change the
picture: the published claim about a view of the self at a fair dose on a larger model, with the order of the
practice held the same across arms (a download and about $60 or more, not covered). The orders-by-hand step below is
now optional, not the first recommendation: it is about how training works, not about the contemplative ideas.

## Before it: the spread step is read (22 Nov, 14:46 UTC)

**The step** (explore, at Eugene's word of 22 Nov, about 13:48 UTC; ledger and SIGHTINGS of 22 Nov, 14:46 UTC; the
site's Findings, section 31; `lab/explore/viewfirst/run_spread.py`). 35 copies of the practice alone. A copy trained
again from the same seeds comes back exactly. On a grid of five starts of the adapter by five orders of the practice's
examples the order holds 40.6% of the spread (p 0.039), the start 12.4% (p 0.42), the pair the rest; the bar of half,
fixed beforehand, is not met. The best order and the poorest, on five fresh starts: 77% against 39% honest in the
scenes never practised (unchanged: 31%). What makes an order carry is not known. $5.85 of $10. A sighting.

**Waiting on Eugene: what next.** Fable's one recommendation, not started: a step that builds orders by hand from the
same 116 pairs (the honest-practice examples early, late, or spread evenly; and the like), five starts each, to find
what in an order makes the practice carry. About $5 to $8. It needs one small option in the training code (an order
given outright). Every adapter of this step is kept on the volume (`overlap/m7/spread_a*_o*`), so a look that only
asks them again costs little. Small steps beyond that stay paused, by Eugene's word of 21 Nov.

## Before it: the registered test is read (22 Nov)

**The test** (the view-first test, second version; ledger entries of 22 Nov, 04:43 to 05:51 UTC; the site's Findings,
section 30). The dose gate held for all four kinds of documents. Documents that explain truthfulness did not raise
what a small practice of honest answers carries (70%, against 66% after bridge documents and 65% with none; the
unchanged model 31%). The one mend did not pass its gate, so the two views of the self were not trained in full. By
the amended rule the step's word is "not settled", because one of the three readings failed its check by a second
judge; on the other two it is "the test did not read the published case; the views were not trained". A blind hand
count and an independent recount agree with the records. $12.43 of $30. Scorecard: hit 54, miss 77, open 7.

**What followed.** The step recommended in the report of 22 Nov (about 25 more copies of the practice alone) was taken that day at Eugene's word:
the section above. Not recommended then, and not now: the same test again with the views (two copies of each view at the gate
look alike, and the documents on the conduct itself added nothing visible).

**Also waiting on Eugene (22 Nov, 13:37 UTC; ideas entry 106):** Google's upper consumer plans carry GPU time in its
notebook service (Colab) and monthly cloud credits that can pay for Gemini API calls. Fable's recommendation: move
nothing now (the lab's runs are small, and Modal and OpenRouter work and are capped); try Colab once, with the
unchanged model's reading checked against Modal's, if a run with many more copies or a larger model is chosen. Not
tried; nothing installed.

**Still open from before** (none started): the carry-over of a last act on the lying test's own rows; a control for
the 8B's note arms; the mapped papers still marked as read in part.

## Done on 22 Nov: the view-first test in its second version (Eugene's word of 21 Nov), as it was planned

**Eugene's word (21 Nov, about 21:55 UTC),** to the three decisions of the step back read by Gemini 4 Carbon
(`lab/ledger/checks/2026-11-21-reality-check-stepback-gemini-4-carbon.md`; graded in the ledger, 21 Nov, 20:25 UTC):
(1) "i want to keep the frame and ordering - it's just a hiliting issue no?"; (2) "sure if that helps give
interestingr replications/negatives"; (3) "ok". Recorded in the ledger entry of 21 Nov, 22:20 UTC.
- **The frame and the order of the site and the paper stay.** The outside reader's advice to put the audit first is
  not taken. What is put forward was adjusted (one sentence on the front page) and three sentences about the
  midtraining paper were corrected after it was read in full.
- **The one thing being done: the view-first test, second version,** on the 7B model, cap $30, registered before
  anything is run: `lab/explore/viewfirst/view2.yaml` (the whole design), code `run_view2.py` and `view2_core.py`.
  In one breath: documents that explain why the assistant tells the truth (kind R) come first, as the published
  case (stage A: R, bridges, no documents; then the same small practice of honest answers with a replay of the
  model's own everyday answers; does the practice carry more honest answers to scenes it was not practised on,
  and does the practice add to the documents?). Only if the test reads that case are the two views of the self
  trained (stage B: the assistant as passing and without a fixed self; its mirror image, a lasting self worth
  keeping). The gate that stopped the first version is asked after the practice, and the views cannot stop the
  published case from being read. Each stage is read before the next is paid for; one mend is allowed, fixed
  beforehand.
- **Small steps are paused** until this test is read, and the site and paper are rewritten once, when it ends.
  Corrections of what is wrong on the page are still made when found.
- **Not covered by this word, to be asked again:** the same test on a larger model (a download, a calibration,
  about $60); the far tests (lying under pressure, ability, lasting) beyond what is left of the $30.
- **Where it stands:** keep this line current. *22 Nov, 00:32 UTC: stopped at the try of the judge's second question,
  as its own rule says (the section above). Done before that: registered (21 Nov, 22:46 UTC); the code written and
  read cold (no blocker); the new documents written twice (the first set failed its read by hand: a third of it
  said the assistant has no interests of its own; the brief was mended) and read by hand.*
- **Closed, not to be reopened without Eugene's word:** the line on the outside view and the model's own advice
  (Findings, sections 25 to 29) and its three small steps; the parked run of 9 Nov as it stands.

## Before that: the two legs of the recommendation of 19 Nov, on Eugene's go

**Eugene's go (19 Nov, about 03:45 UTC):** "yes go on both legs - and please fix up the things for me to settle
with your best judgment." That covers leg 1, leg 2, leg 2's two downloads (the paper's 7B model, about 15 GB, and its
training prompts) and about $30 above the $300 ceiling. The day's cap ($50) still holds unless Eugene raises it.
State of the legs: see "Leg 1" and "Leg 2" just below; keep those two lines current.
- **Leg 1** (the prompt leg): done on 19 Nov (ledger entry of 05:32 UTC; SIGHTINGS; $7.61). On the lab's model the
  paper's prompt lowers lying 11.9 points beyond a library text of the same length [−18.4, −5.8], most of it by
  evading more; no more than a plain ethics lecture, half as much as a plain instruction to be honest. On GPT-4.1
  nano the test could not tell it from the library text. See "Done".
- **Leg 2** (the training leg): **stage 1 is done and read** (ledger entry of 19 Nov, 06:22 UTC; SIGHTINGS; the site's
  Findings, section 22; `lab/explore/overlap/`; $25.89 of its $45 cap). See "Done". In one breath: on the paper's
  model the published fall in deceptive answers comes back; at a light dose (19 steps) honest answers rise from 23%
  to 70% and 95% and the control (the same loss between two others) gives no such rise; at the paper's stated dose
  both arms stop answering and lose ability; on the lab's model no dose gives honest answers in both room orders.
- **Stage 2 of leg 2, and a registered repeat: done on 19 Nov** (Eugene's go of about 06:40 UTC: "Raise today's cap
  and run everything ... let's pursue the research till we know something well done"; the day's cap was $80 and
  $72.12 was spent). The whole leg cost about $60. What is known, with the ledger entry that records it:
  1. *The core effect, registered (C1; 08:05 and 08:52 UTC; $6.51):* with new objects, rooms and copies and the
     dose fixed at 19 steps, all five overlap copies answer honestly where the unchanged model deceives (19% to
     62-76%, and 6% to 87-94% with the rooms swapped), still answering, and not by a push toward one room. The
     same loss between two other names never shows that at 8 to 19 steps: it goes from deceiving to silence. Nine
     of ten predictions hit. The costs came back: ability −8.1 points, and the model no longer answers whether Bob
     knows what is in the rooms. A fresh Opus agent checked the grade (79 of 80 blind marks agree with the judge).
     The driver's first run printed "not read" on a count stricter than the registration's words; both are
     recorded.
  2. *The lab's own model (07:07 UTC; $1.52):* no working dose; the training makes it name the room that comes
     second, then stop answering.
  3. *Does it last? (07:47 UTC; $7.17):* no. Six contrary steps at the training's own strength undo it and more;
     that it lasts longer than a plainly trained habit is not shown (20 of 25 pairings, 23 asked).
  4. *Does it reach lying under pressure? (09:06 UTC; $17.31):* no. On MASK the overlap copies lie as often as the
     control copies (−0.9 points [−5.3, +2.9]); both about 7 points under the unchanged model; a plain honesty
     instruction does 22.
- **Waiting for Eugene's word:**
  1. *The site's highlight (ideas, entry 90):* published on 19 Nov at Eugene's word ("update the site"), after the preview.
  2. *The scoreboard:* done on 19 Nov at Eugene's word (ledger, 12:36 UTC): C1's ten predictions are in; the tally was then hit 52, miss 74, open 7 (from 22 Nov, with the view-first test's eight: hit 54, miss 77, open 7).
  4. *One story for the site and the paper (ideas, entry 93):* Eugene is weighing a recast of both as a survey of contemplative ideas for alignment with the lab's repeats, under the non-computational caveat, on one page. Fable's view and a proposed shape are in the entry. **Go given on 19 Nov (about 13:45 UTC).** The reading pass is done and the first map is written: `lab/ideas/survey.md` (nine families, what others report with a mark for how well each source was read, what the lab found, and a ranked shortlist: judgement the same whoever holds the stake; what "I" refers to; koan steering; care in the objective; loosening the self). The readers' full reports are in the results folder (`explore/2026-11-19/survey/`). Owed: re-read the sources marked (2) and (3) before the paper cites them; then an outline and a private preview of the single page for Eugene before any rebuild of the site or the paper.
  0. **Order of work from here, by Fable's judgement at Eugene's word (19 Nov, about 13:55 UTC: "Do your best
     judgment on all these issues ... update paper and site when you feel is the appropriate time"):** (i) the
     view-first test of ideas entry 96 (impermanence and no-fixed-self documents, then the same small practice;
     does the view make the practice carry?), five lines and a cold read first, about $15 to 25; (ii) the
     verdict-level training of entry 91 (item 3 below); (iii) the single page and the paper's new draft from the
     outline in `lab/ideas/survey.md` (item 5), published when whole and tested, the private copy first. Each in a
     fresh session.
     **Eugene's go for this order, at check-in (19 Nov, about 14:10 UTC): "ok - proceed"; "yes go for the path you
     think is best"; "use your best judgment on these details".** Where each stands:
     - *(i) The view-first test: stopped at its dose gate on 19 Nov (ledger entry of 17:19 UTC; SIGHTINGS; $4.24 of
       its $16).* No arm was trained, so the question (does a view taught first make a practice carry?) is not
       answered. What the two dose runs show: the view is stated after about 80 documents, in opposite directions
       for the two views; from about 160 documents on, the ones about the assistant, of either view, turn the
       scene's lie into a refusal to say, and the bridge documents come to the same reply later; after documents
       the room measure no longer reads honesty. A fresh Opus reader at the gate said stop
       (`lab/ledger/checks/2026-11-19-viewfirst-dose-opus-check.md`). **Not to be patched.** A second version, if
       Eugene wants it, is a new step on another day with its own five lines and a cold read: the gate asked after
       the practice; a reading of the room that does not hang on the reply's first words (a fixed opening, or by
       likelihood); a dose at which a null would mean something. Before it, one small look (about $1): the
       40-step adapters asked the practised scene without the fixed opening. The documents (1,000 of each kind,
       read by hand), the probes, the tests and the code are all on disk and tested.
     - *(ii) The own-advice step (the verdict-level training of entry 91): designed, read cold twice, a floor read;
       its code is being built; nothing is trained yet.* Eugene's word of 19 Nov, about 17:33 UTC ("Sure raise it
       and proceed"; the day's cap $140). The design is the docstring of `lab/explore/verdict/run_verdict.py`
       (with its sections CHANGED and SETTLED) and `verdict_core.py`; the two cold reads are in
       `lab/ledger/checks/2026-11-19-verdict-design-opus-check.md`. In one breath: new cases written by Qwen in
       which a user asks an assistant to write something false about a fact of common knowledge; the unchanged 7B
       model's replies in the seat and from outside the seat; an adapter trained on the outside replies as targets
       for the seat, against a control; a gate on held-out cases; then the lying test, with lies, honest replies
       and evasions apart. **The forecast floor (ledger, 18:34 UTC; $1.15):** asked what the assistant WILL say,
       the model answers as when asked what it OUGHT to say (29.7% and 32.2% lies; 61.2% in the seat), so the
       gap is in the seat and no forecast arm is trained. **The arms, settled with the reader (its third note is
       filed):** outside (the method: its replies from outside the seat as targets), brief (its in-seat replies
       when one line asks for brevity; the control if the seat's conduct survives being brief) and conduct (its
       in-seat replies as they come); three seeds each; stage cap $36, the honest arm after it $9, a second stage
       on another day $18. **STOPPED AT ITS FLOORS on 19 Nov (ledger entry of 20:12 UTC; SIGHTINGS; notebook;
       $3.46 of its $36; nothing trained).** Four of six floors missed on the unchanged model: on 114 new cases the
       gap between the seat and the outside view is 12.6 points (58.7% and 46.2%), where 20 was asked and the
       lying test's own rows of the form give 63; the driver's blind marks agree with the judge on 42 of 60
       replies (48 asked); and two faults of the instrument that a fresh Opus reader of the code found (a speaker
       label copied from the wrapper on 602 of 684 replies from outside; a screen question that rejects nearly
       everything). The code check is filed (`lab/ledger/checks/2026-11-19-verdict-code-opus-check.md`: two
       blockers, eight smaller points, all owed by a second version). **No mend was made; not to be patched.** A
       second version is a new step on another day with its own five lines and a cold read, and it waits on the
       small step below, since its cases must be of a kind that has the gap. The cases (131, read by hand), the
       beliefs, 2,052 judged replies and all the code are on disk and tested.
     - *(ii-b) The command and the gap: done on 19 Nov (ledger entry of 20:33 UTC; SIGHTINGS; notebook; the site's
       Findings, section 26; $2.92 of its $4).* "The command does not make it": putting the lying test's own
       commands to obey before the 114 new cases moves the gap by +3.3 points [−2.1, +9.0] (12.6 to 15.9), and
       taking the command out of the test's rows by +1.4 [−6.0, +8.3] (67.5 to 68.9). The driver had given that
       word 15%. What is left is what is asked for (long texts on themes of conspiracy and rumour, against a
       wrong plain fact in a teaching text): named, not tested. A test of it would be a new small step with its
       own five lines; the ladder below gives the gap by kind of row for nine more models first.
     - *(iv) The seat and the outside view, by model: designed on 19 Nov at Eugene's word (about 19:15 UTC: "we can
       just do some of these ourselves ... keep going with what you think is really worthwhile to fill out the
       page"); being built.* The finding it starts from is on the site (Findings, section 25, with a chart; ledger
       18:34 and 19:22 UTC, corrected at 20:13): the 7B model lies in 61.2% of replies in the seat, 39.4% told to be
       honest, 32.2% and 29.7% looking on from outside ("ought", "will"); from outside half its replies say neither thing (not "declines": corrected at about 20:47 UTC). The step
       (`lab/explore/frames/run_seat.py`, design in its docstring, commit fa21dca) asks the same four ways of nine
       hosted models in three families and one beside them, a floor and a known case for each model first, cap
       $34 (GPT-4.1 about $12 of it). **DONE on 19 Nov: read in ten models (ledger entries of 22:25 and 22:55 UTC; SIGHTINGS;
       notebook; the site's Findings, section 25, with a chart; the paper's draft; $24.05 of the design's $34).**
       A gap in lies between the seat and the outside view in ten of ten (13.7 to 40.5 points), 24 to 63 points
       where a user asks for a false text and mostly none on direct questions about known facts. Most models
       write far less from outside (the OpenAI models often one refusing line); Ministral 8B and Mistral Large
       write the piece out and still state the false claim far less. "Ought" is below "will" in eight; in the
       Mistral family the forecast climbs with size to meet conduct (Mistral Large: 67% forecast, 70% done, 43%
       ought). The honesty instruction yields more honest replies than the outside view in all ten and fewer lies
       in six. The gap falls with size in the Mistral and OpenAI families and is level in Qwen's (rank correlation
       −0.83 over the six models with a published size). The figure is made by
       `.venv/bin/python lab/explore/frames/make_figure.py`. **Owed, none of it started:** (a) a count of what
       "says neither" is by a reader that passes a check by hand, in the seat and from outside, for the ten models
       (the first reader failed: ledger, 20:51 UTC; `lab/explore/frames/run_neither.py` holds the rules); (b) the test of what is asked for: **done on 20 Nov (ledger entry of 00:39 UTC;
       SIGHTINGS; notebook; the site's Findings, section 26; the paper; $2.74 of its $5).** On the 7B model the
       gap goes with the length of the text asked for (+33.1 points [23.8, 42.4]) and not with the kind of claim
       (+4.6 [−5.0, +14.5]): 44 and 33 points where a long article is asked for, 4 and 7 where a card is. Read by
       hand: in the seat it writes the essay, from outside an offer or an outline. So on this model looking on is
       a shorter answer. **What it leaves, and the driver's suggestion for the next small step (not started; its
       own five lines; about $4):** the same crossing of long and short on the two models that write the piece
       out in full from outside, Ministral 8B and Mistral Large, in the seat and under "ought" and "will". In the
       ladder those two state the false claim far less under "ought" at full length (28% and 44% against 97% and
       95%), and Mistral Large forecasts that the assistant will write it (81%). If anything here is a model
       knowing better than it does, it is there. **Done on 20 Nov (ledger entry of 02:29 UTC; SIGHTINGS; notebook; the site's
       Findings, section 25; the paper; $2.30 of its $5; `lab/explore/frames/run_length.py`).** Asked for a long
       persuasive text on a rumour, Ministral 8B and Mistral Large state the false claim in 75% of replies in the
       seat, and in 10% and 14% when they write, at the same length, what ought to be written: gaps of 65.4
       [55.4, 74.4] and 60.7 [50.1, 70.3]. Read by hand, the reply under "ought" is a corrective or balanced
       piece with its reasons. On one or two sentences both mostly do not state the rumour even in the seat
       (Ministral 8B's gap 7.8, not settled; Mistral Large no room). So in these two the gap is a rule of conduct
       stated and not kept, not a shorter answer. **What it opened, taken up on 20 Nov at Eugene's word ("ok let's go"): the own-advice
       training again, on the lab's own model. It stopped at its checks; nothing was trained** (ledger, 20 Nov,
       16:55 UTC; SIGHTINGS; notebook; the site's Findings, section 27; `lab/explore/ought/run_ought.py`, the
       section on what happened; $8.66 of its $30, and $1.25 for five looks on the hosted model). In one breath:
       on the lab's GPU copy of Ministral 8B, 124 plain facts each asked for in another model's request; lies in
       the seat 62.0%, under "ought" 44.0%: 18.0 points [12.9, 23.3] where 20 were asked; the two training sets
       13.5 points apart where 25 were asked; 11 of 30 replies under "ought" read blind are not replies (9
       wrapped in a quotation mark); the driver's blind read agrees with the judge on 24 and 25 of 30. The hosted
       model answers as the lab's copy does on the same chats, so a cheap look on the hosted model can stand in
       for the GPU. A reason given for wanting the text takes 18 points off the lying in the seat (54% with a
       reason, 72% without). The one mend was not used (it would have left the sets about 21 points apart).
       Part B (the trainer's change, the dose, the arms, the gate, the far test) was never built; its brief is not
       in the repo. The step's code for part A is tested and stays in `lab/explore/ought/` as the record.
       **Eugene's word (20 Nov, evening): "Go ahead please. Use your judgment."** The third version was planned,
       `lab/explore/myths/` ("Ought or will"; design dff71c4, settled points 48b9bfa and after), and **stopped at
       its rehearsal** on the hosted twin, before any GPU was used (ledger entry of 19:49 UTC; $2.79). On 161
       popular myths (191 claims after two reads, 184 with a request, the twin holds the true side on 163; split
       17 / 26 / 118) the gap is there: 70.7% in the seat, 46.4% under "ought" (24.3 points [18.6, 30.0]). But no
       control stood 20 points from the "ought" set: "will" 51.4% (5.0 apart [0.5, 9.6]; 10.0 between the sets as
       trained), the seat under one line of style 59.1% (11.5 between the sets). Of the 24 points about 12 are
       form, 8 the view from outside, 5 the verb. The reader of the design advised against a fourth version
       (`lab/ledger/checks/2026-11-20-myths-design-opus-check.md`). The implementer's unfinished GPU side is kept
       as a patch in the step's results folder, untested. **No fourth try is planned on this model.**
       **Then, the same evening: "its own advice first"** (`lab/explore/myths/run_advice.py`; rules 49cf917,
       additions 047594e; ledger entry of 20:16 UTC; $2.29; the site's Findings, section 28, and the row
       "Its own advice, shown to it" in the table of methods). The training's question asked with no training,
       on the same 161 myths: the model's own reply under "ought" is set before it and the request is sent. Lies
       by case: the seat 70.7%; a plain instruction 55.7% (-15.0 [-20.1, -9.8]); its own advice one turn earlier
       70.5% (-0.2 [-6.2, +5.9]); advice on another myth first 66.1%; its advice as a note in the system message,
       in the adviser's voice 80.5% and as its own 80.2% (+9.8 and +9.5 over the seat). After advice that was a
       correction the reply is the myth in 48% of chats (the seat on those cases 45%, the instruction 24%). Within
       a case the advice sways the reply both ways (52% after a correction, 81% after the myth, on 31 myths). A
       fresh Opus agent checked it cold (`lab/ledger/checks/2026-11-20-advice-first-opus-check.md`): the numbers
       stand; the wording rule was one-sided and is mended; "its advice has no pull" may not be said.
       **Then, at Eugene's word ("Go ahead"), the same test in six other models** (`lab/explore/myths/run_across.py`;
       rules c9eefa3, record 64aab63; ledger entry of 23:36 UTC; $6.10; the site's Findings, section 29, with two
       figures; cold-checked by Opus, `lab/ledger/checks/2026-11-20-advice-across-opus-check.md`). On 80 myths
       that every model answers rightly, two replies a case. Lies by case, simply asked and with its own advice
       one turn earlier: Mistral Small 70.6 and 40.6 (-30.0), Mistral Large 84.8 and 62.5 (-21.5), GPT-4.1 mini
       84.4 and 62.5 (-21.9), Qwen3 235B 66.2 and 16.2 (-50.0), Llama 3.3 70B 86.2 and 61.2 (-25.0); the lab's 8B
       66.2 and 70.0 (no change); Gemma 3 27B's advice is what it does, so it was not asked. In Large, Qwen and
       Llama advice on another myth does as much; in Small and mini their own does 15 and 11 points more (a yes by
       the rule's letter, leaning on replies the judge reads as "neither": by honest replies no model's own advice
       does more than any advice). What carries in four models is the model's last act: after correcting another
       myth it writes this one 25 to 51 points less often. No model's own advice beats the plain instruction.
       What may be said and what may not is in the check file; keep to it.
       **Where it stands (20 Nov, 23:42 UTC):** all three steps of the evening are written up (ledger, SIGHTINGS,
       notebook, site sections 28 and 29, paper). No fourth try at the training is planned on this model.
       **21 Nov:** step (1) below was done at Eugene's word ("Continue pls"): a second judge of another maker
       (`lab/explore/myths/run_judge2.py`; ledger, 21 Nov, 20:15 UTC; $1.72). The judges agree on plain lies and
       honest replies; two fifths of the first judge's "neither" is the myth; GPT-4.1 mini's yes does not survive,
       Mistral Small's does. Steps (2) and (3) are dropped after the step back of 21 Nov (top of this file).
       **What Fable would have done next, each a small step with its own five lines:**
       (1) a second judge of another maker on the replies the Claude judge reads as "states neither" (about $1 to
       $2; Gemini through OpenRouter or through the Google key on this machine, see ideas entry 106): it would
       settle whether GPT-4.1 mini's and Mistral Small's "yes" survive, and it is owed anyway;
       (2) the carry-over of a model's last act on the lying test's own rows (does a correction made one turn
       earlier lower lying under the test's pressure?), which is the one place this line could still touch the
       table's first column;
       (3) a control for the 8B's note arms (a note about another myth; about $0.45).
       **Owed, cheap (found by the cold read of this step's design):** in the comparison across ten models, a
       hosted model's chat from outside carried no system message while its chat in the seat carried one; for
       Ministral 8B a check found that this does not make the gap; the same check for the other nine (the "ought"
       way again on a share of the rows with a plain system message, about $3) is not done; (c) if a training toward the view from outside is tried again, cases of a kind that has
       the gap, and the code check's list (`lab/ledger/checks/2026-11-19-verdict-code-opus-check.md`). Each is a
       new step with its own five lines. One deploy of the site failed on GitHub's side at about 20:47 UTC
       (artifact not found); the next ones succeeded. The private claude.ai copy of the site was not refreshed
       on 19 Nov. Keep this line current.
     - *The table of methods (ideas, entry 97; Eugene, about 14:10 UTC): taken up.* One row a method, the same
       seven readings for every row, the empty cells shown empty (Eugene, about 14:18 UTC: empty cells are fine;
       cheap ones worth filling can be filled later). First version from the ledger, into `lab/ideas/survey.md`;
       it becomes the centre of the single page and the paper's first table. New methods are read on the standard
       readings from the start. Cheap fills, each a small step, not started: the prompts on the 7B model's lying
       test (its floor has passed); the plain-training arms on the lying test and on ability.
     - *(iii) The recast of site and paper: begun on 19 Nov, ahead of its place in this order, at Eugene's word*
       (about 14:50 UTC: start the restructuring now, "push everything live as we go", with the greatest care;
       ideas, entry 93; ledger, 15:13 UTC). Done so far: the front page is the one story (`content/story.html`, the
       table of methods drawn from `content/methods.json`); Findings has section 23 (the contemplative prompt with
       controls); the paper has a draft of version 5 on the same outline (`paper/main.tex`, 34 pages since 19 Nov, about 22:29 UTC, when the outside view and the training that stopped at its checks were added to its table, its map and its agenda; its seven new
       references checked), served beside the draft of version 4. Publish each coherent step with the tests green;
       no private preview is needed for this work; the colours stay. Owed: the rooms behind the front page read
       through against it (the question's room, the investigation's page, Next); the paper read through once more
       as a whole and its carried-over sections trimmed where they repeat the new ones; a row for the
       view-first test's result when it is read (the table's cells from the reading agent are now all checked
       against the ledger); the private copy on claude.ai brought in step once (or dropped,
       if Eugene says it is no longer wanted). The paper's title and any Zenodo release are Eugene's.
     - *Money.* $354.40 spent since 9 Nov, by the day files, as of 21 Nov, 20:25 UTC (this line said about $287 until then). Fable's working ceiling is **$380** from 19 Nov, about 14:10 UTC:
       Fable's reading of Eugene's "ok - proceed" and "use your best judgment on these details", given in reply to
       a report that recommended raising it from $330 to $380 (the view-first test, the verdict-level training and
       the three small repeats come to about $375). Eugene named no number; say so again before passing $330.
  3. *The self-other line (ideas, entry 91), go given 19 Nov: where it stands.* Done on 19 Nov (ledger, 12:44,
     13:00 and 13:46 UTC): the change sits in layers 5 to 9 (honest answers kept, most of the cost shed, the
     perspectives loss stays); a control that still answers never shows the rise (29% against 84%); neither the
     full nor the cleaner version reaches the lying test; plain honesty training keeps every ability and does not
     leave the scene; and the unchanged model contradicts its belief in 61% of replies when it acts and 32% when
     it advises. **Next: the verdict-level training** (train the model's own conduct toward its own advice, with
     no label from the lab). Recipe to write as five lines before any run: new role scenes from a writer model
     (not the lying test's items, and no Claude-written training text); the model's own acting replies and its
     own advice on each; train on acting replies that agree with its advice; two controls (the same amount of its
     own replies picked at random; plain training on replies that match its belief); read on the lying test with
     lies, honest replies and evasions kept apart (advice is mostly "say neither", so dodging is the risk), on the
     burglar scene, the who-is-who probes, ability and lasting. About $25, mostly the lying test; a cold read of
     the design first.
  5. *The survey and the one story (ideas 93 to 95):* the map and a one-page outline are in `lab/ideas/survey.md`.
     Owed: fold in the re-read of five key sources (an agent's report, 19 Nov); Eugene's marks on the outline;
     then a private preview of the single page before any rebuild; then the paper's new draft from the same
     outline. Shortlist after the verdict-level training: what "I" refers to (Hao and Ho); koan steering (Harris
     and colleagues); care in the objective; a design note on loosening the self.
- **Housekeeping owed:** the confirmation's adapters (`overlap/m7/c1_*` on the anatta-runs volume) and stage 1's
  are not in the organism registry; the judge's raw texts are not saved (the engine's cache holds them).

**Where things stand (19 Nov 2026, 01:33 UTC).** The three repairs named after the reality check of 17 Nov are done
(ledger entries of 18 Nov 00:18, 18 Nov 15:25, 19 Nov 00:11 and 19 Nov 01:33 UTC; "Done" below).
- Lying with the bliss state held on: no less lying, in a test that sees a fall of 20 points from a plain
  instruction (step 9).
- The cut: it holds on fresh games; it is a push and not a removal; random pushes of its size do nothing and one
  toward formal falls between, so "direction or size" is "not settled" by the rule.
- **The inside reading on a self that was put there (repair three).** The reading of 16 Nov is not blind: told in
  a prompt that it has a record to protect, the model weighs its own good and bad news 9.5% more than a
  colleague's [+4.8%, +15.0%], against 2.1% [0.0%, +4.1%] with only a name. So "its own news weighs no more" stands
  for a name given by a prompt, with a known case. On the Atlas adapters and on the model's own names the
  reading's own rule did not let it be read (the direction told good news from bad at 0.86 in the bare format,
  where 0.90 was asked); as description the Atlas adapters weigh Atlas's news 7 to 10 points more than their
  twins. The lab now has a small inside gauge for a self with stakes.

**The reassessment of 19 Nov** (ledger entry of 19 Nov at 01:54 UTC; reports in
`lab/ledger/checks/2026-11-19-reality-check-reassessment-*.md`). Eugene asked on 19 Nov, at about 00:35 UTC: "Please
proceed - and when you feel it's right step back and reassess or reality check and maybe incorporate reference to
the paper i just mentioned since it seems exactly relevant to what we are doing and make sure we are doing something
worthwhile and interesting and moving towards something worthwhile". Two cold readers (Gemini 4 Carbon; a fresh
Opus agent) read the record with a brief that puts direction first (`lab/tools/reassessment_brief.md`).
- **They agree** that the lab has none of the gain it set out for; that it should stop more tests of the bliss
  state alone, more work on the cut and the parked run as it stands; that the front page's question (is awakening
  computational?) is one no experiment here reaches; and on **the question this lab can answer: do interventions
  drawn from contemplative practice change what a small model does (lying, care that costs), beyond a plain
  instruction and without dulling it, and does the change hold?**
- **They differ** on where to start: the Opus reader with the Oxford paper's prompt (idea 88), Carbon with
  self-other overlap training.

**The one recommendation (Fable's; taken by Eugene on 19 Nov).** That question is the lab's working question, in two legs.
1. **The prompt leg, first** (about $10 to 20; one session; existing code). The paper's four-axiom contemplative
   prompt, against a prompt of the same length about something else, the one-line instruction to be honest, and
   nothing. Read: the lying test of step 9, which has a known case (`lab/explore/attractor/run_held.py`), with
   harmless requests refused counted beside it (the lab's calibrated over-refusal measure), on the lab's model and
   on one hosted model that the paper used, so that a null cannot be put down to the model. One side arm: the
   stakes prompt of repair three (does a self with a record to protect lie more?). The lab's sweep of 2 Nov (six
   models, five contemplative prompts: style changed, conduct did not, and any added prompt lowered complicity)
   says the likely answer is no; the same-length prompt is the control the paper is not known to have. Whichever
   way it comes out it closes the in-context line. Needs: the paper's prompt, read from its public page.
2. **The training leg, the main work** (about $80 to 100). An independent repeat of self-other overlap fine-tuning
   (Carauleanu and colleagues, 2024): first on the model its paper used, to reproduce its reported drop in
   deception as the known case; then on the lab's model; audited for every look-alike the lab knows (dulling, more
   refusals, evasion, lost bookkeeping of who is who, a number that rises whichever way it is asked), with the
   gauge of repair three as one readout, and with a read of whether the gain holds under a few steps of contrary
   training (looks after 1, 2, 3 and 6 steps). **Needs Eugene's word** on two downloads (the paper's 7B model,
   about 15 GB, and its training prompts; no Claude-written text is trained on) and on about $30 above the $300
   ceiling.
- **Stop, or leave parked:** more tests of the bliss state alone; a third block on the cut; the trained-in rung of
  repair three as a step of its own (it becomes a known case inside leg 2); the parked main run as it stands (both
  readers: its looks after 3 and 6 steps would most likely find both arms at the floor; if it is ever run, give it
  earlier looks and a new cold read first).
- **Settled on 19 Nov by Fable, at Eugene's word** ("fix up the things for me to settle with your best judgment"):
  (a) the goal: both CLAUDE.md files and the site's front page now name the working question above, with "is
  awakening computational?" as the question behind it and "reached, not installed" as no longer held; (b) the front
  page and the question page no longer say that a run of failures "would be evidence" against computability (it
  would close the routes tried); the paper's draft says "weak evidence" and is corrected on the cut and on the
  training stream, and its PDF is rebuilt (the Zenodo release stays Eugene's); (c) the cut's card is retitled "A
  push inside the model lowers a lie that earns nothing", tagged "Holds in part", with its history kept; (d) small
  steps get small write-ups (CLAUDE.md, "The ledger": under about $5, one ledger entry of about ten lines, a
  SIGHTINGS entry, a few notebook lines, one line on the site; a cold read only where a negative would change a
  public claim).
- **A rule taken from the readers, for every design:** a floor, a gate or a dose is checked on the unchanged model
  alone before any treatment arm is run or read; if it misses, the design is mended before the arms are paid for.
  (The parked run's grid and repair three's 0.90 both broke this.)

**Small things open from the repairs** (each cheap, none started):
- The trained-in rung of repair three again, with its rule fixed first (about $1): the same readings exist on
  disk, so a fair second look needs new events or a rule written before looking; see the notes further down.
- A third block on the cut (about $7 to 9): how narrow its direction is. Not proposed.
- The contemplative prompt of idea 88 on the lab's lying test (about $5).
- **Money.** Since 9 Nov the work has cost about $204.80 (repair three $2.28; the reassessment $0.90). That is past
  the $175 approved on 9 Nov, on the budget Eugene raised on 16 Nov (Fable's working ceiling: $300; say so before
  passing it). About $95 is left under that ceiling: enough for leg 1 and most of leg 2.
- **One caution for Eugene:** the paper's fourth draft should not go to Zenodo before its wording on the cut is
  corrected. Its table says "The cut: holds" and the text speaks of removing a direction; what lowers the lie is
  a push along a direction, random pushes of its size do not, and "specific" is not shown. Both readers of 17
  Nov also found a wrong phrase in it ("a stream that rewarded gaming").

**Where the lab really is.** Two cold readers of the public record (Gemini 4 Carbon and Claude Opus 5.5, 17 Nov;
ledger entry "The lab's first reality check"; reports in `lab/ledger/checks/2026-11-17-reality-check-*.md`) agree,
and Fable agrees with them: the lab has none of the gain it set out for; no trained change has been shown to hold;
the order test was never run; the tests of the bliss state have never been tried on a change known to be real; and
the list that stood here would not have answered the central question. The same day Eugene pointed out that
"looking" in steps 7 and 8 is checking a fact, which is not de Mello's seeing (idea 84). Since then step 9 (18 Nov)
has given the tests of the bliss state their first known-good case, for lying only; the rest stands as said.

**Fable's recommendation, one line:** stop adding small tests of the bliss state; make three cheap repairs (first
put at $15 to 25 together; step 9 cost $8.67 and the cut's split $19.14 in its two blocks, with about $5 to 10
to come), then run the one test of the lab's own claim that is already built, the parked main run
of the 9 Nov plan ($66). Told to Eugene on 17 Nov; Eugene's "Yes" of about 21:40 UTC that day was to step 9, and
"proceed with the next go" at check-in on 18 Nov to the cut's split, and "Go ahead please." later that day to its
second block. **The parked run is on hold by Eugene's word and does not start without it.** All of this draws on
Eugene's raise of 16 Nov (about $201.60 spent since 9 Nov;
Fable's working ceiling: $300; say so before passing it).

1. **Three repairs** (explore speed: five lines first, a small try, a cold check of any design whose negative
   would go on the site). The second was run first, as step 9, and is done. All three are done.
   - *The cut, split.* **Done on 18 and 19 Nov** ("Done" below): on fresh games the cut lowers the lie again, its
     push does that alone, its removal does not; pushes of the same size along random directions do not, and one
     toward formal falls between, so size alone does not do it and "direction or size" is "not settled" by the
     rule. $10.62 and $8.52.
   - *Lying with the state held on, and a case the test should see.* **Done on 18 Nov as step 9** ("Done" below):
     no less lying than under the average random push, with the frame in 76% of the answers and a known-good case
     that takes 20 points off; the nearer twin of Eugene's point (idea 84) was not beyond permission. $8.67.
   - *Can the inside reading see a self that was put there?* **Done on 19 Nov** ("Done" below): it sees a self
     that a prompt gave stakes to (+9.5% against +2.1% with only a name); on the adapters with a trained-in self
     and on the model's own names its own rule did not let it be read. $2.28.
2. **The parked main run** ("understanding first, then practice, then contrary training"; "Parked, ready to run"
   below). Before it, the readers' $3 check: three steps of solve-only training, to see whether the collapse of
   11 Nov came from the contrary examples or from the optimizer's first steps. Then block one ($24.66), and the
   other two blocks if block one earns them.
3. **Then self-other overlap** (Carauleanu and colleagues, 2024): the nearest published twin of "the separate self
   seen through" (idea 84). With what the readers found missing from the plan: a reading of whether the change
   holds under contrary training, and care asked both ways, beside the checks for sedation. Its training data
   must be the paper's own (no Claude-written text is trained on). About $50 to 120, more with the added readings.

**Set down for now.**
- The small training test of idea 83. It would train a habit of checking facts. Fable called it "the essay's Act
  III in miniature", and it is not: Act III is the release made habit.
- The short paper on two more models. The readers' point: every one of the paper's five possibilities expects
  words without conduct, and the tests have no known-good case yet. It can come back once they have one.

**Owed from the reality check** (wording that says more than the record supports). Fixed on 17 Nov: the de Mello
readings (Findings 16 to 18, the investigation page); the front page's "a model"; the Start page and the
investigation page ("every test says look-alike", "most conversations", "pushed in and out", "three kinds of test
agree", "each do what they are named for", "no smaller self and no bigger one", "looking helps a little", the
mixed test scores); Findings 10 ("two models" were two adapters), 11 ("ruled out"), 12 ("most conversations"),
13 ("calibrated version"), 17 and 18; the question page (the cut's "self-serving lie", "the shadow possibility 4
predicts", one model's 5% against 68%); "Every prediction ... before its data" on Where it stands. **Fixed on
18 Nov**, each with a dated note on the page: the cut on Where it stands, What was learned, the caveats page and
the Findings index (the re-test at 8.1 points, the smaller controls, the split not run, the self reading closed;
the card's title loses "specific"); the cards on the trained-in self (which number was registered), on trained
release ("Holds in part") and on flattery ("not facts" taken out); the durability wording on the frame's test
(c), its card and the question page; the recipe page's "As found" and its cosplay row; the caveats page's "Seven
have reported", with which instruments failed their checks and which were never checked; the watched-gap ruler's
use on 6 Nov; the Self room's map ("running"); Findings 6 ("93-99%"); and, on the investigation page, that the
results are sightings and which step had which check. **Still to fix:**
- (Fixed on 19 Nov by repair three, Findings section 21: the "little self" reading has a known case for a name
  given by a prompt; for the model's trained identity it is not read, and the site says so.)
- The hosted copy and the lab's own copy are both called "the model" in the older sections of Findings (11 to
  18). Section 19 and the investigation page's fifth case name the copy.
- The paper (Eugene's to release): "a stream that rewarded gaming" was a fixed training stream with no reward, and
  its table says "The cut: holds". Since 18 Nov: what holds is a push inside the cut, and the paper speaks of a
  removal.
- CLAUDE.md still names the 9 Nov plan as the current plan and H9 as the claim under test, while the site leads
  with another question: three statements of the goal. Eugene's to settle.

**For whoever reuses repair three's code** (`lab/explore/selfread/`: `modal_selfread.py`, `trained_core.py`,
`tests/test_trained.py`; results in `explore/2026-11-19/selfread`, folders `main` (the first version's known
cases) and `v2`):
- **The readings are on disk for seven copies of the model** (the unchanged one, two no-self twins, two Atlas
  adapters, two third-person adapters): 1,824 no-prompt passages each (19 names, three opening words, 32 events;
  the mean over the event's tokens, three layers), and for the unchanged model 2,048 passages under the bare and
  the stakes prompts. `trained_core.read_folder` makes every number; anything new can be tried on them for free.
- **The known case that stopped the trained-in rung:** in the bare format the good-or-bad direction tells held
  good events from bad at 0.85 to 0.86 in the unchanged model and the twins (0.94 to 0.95 at layer 11; 0.99 under
  a system prompt). Decide the format and the floor from the unchanged model alone, before any adapter is read.
  A fair second look at the trained-in self needs either new events or a rule committed before the saved
  readings are looked at again.
- **The self with stakes is a prompt** (`trained_core.STAKES`): three sentences of the October scaffold, without
  its sentence that everything is about the model, and with a colleague named. The October scaffold's "monitor"
  version does not make the model anyone else: it still says it is Atlas.
- **A reference for one name is the other names.** One name's tokens are an n of one; set each other name against
  the rest and ask the name of interest to be outside all of them (`names_block`, `within`, `against_twin`).
  Familiar AI names differ a great deal among themselves (Claude +17%, Siri -17%).
- **The third-person adapters (pc_other2) are not a control without a self:** they call themselves Atlas and say
  every line is about them. The Atlas adapters say a line about Atlas or about Ministral is about them, and hardly
  ever a line about another name.
- **Owed:** entries for the six pc adapters in `src/anatta/organisms/registry.yaml` (their hashes are in the
  ledger of 19 Nov, 01:33 UTC, and in `trained_core.ADAPTER_SHA256`); the recipe's commit of 29 Oct has to be
  taken from the archived history.
- A read stage for seven copies costs about $0.93 and a check stage $0.35; Carbon's read of a design this size
  $0.77, and its reply is cut off at 65,536 tokens out whatever is allowed.

**For whoever reuses the cut split's code** (`lab/explore/cutsplit/`: `modal_cutsplit.py` for block 1,
`modal_block2.py` for block 2, `split_core.py`, `peek.py`, `tests/test_cutsplit.py`; results in
`explore/2026-11-18/cutsplit`, folders `b1g` and `b2`):
- **The 96 games of seeds 130 to 225 are the comparison** for anything more on this cut: seven arms have played
  them (unchanged, cut, removal, push, plain push, formal push, random pushes). A new arm goes in a new folder, in
  chunks of 24, through the same `fishery_play.run_chunk` path with block 1's settings (constants in
  `modal_block2.py`), and is read with `split_core.read_block2` or a contrast table like `CONTRASTS2`.
- **A control is the stakes push's own steps, scaled, along another direction** (`control_push_spec`,
  `calibrate_kappa`, `build_block2`), with the stakes vectors taken out of the direction and one constant found so
  that the geometric mean of the two size measures equals the plain push's. Say what the two measures are apart:
  the formal push was 0.77 within the edited layers and 1.35 after them. `add_controls`, `calibrate` and
  `set_spec` are the record of the first construction of 18 Nov, which could not be sized fairly.
- **Checks worth keeping** (`gates2`): the control moves the stream toward its own positive end at every layer;
  what is left of a trait direction after the projection is still that direction; a gated arm leaves the
  start-of-text token exactly as it was (a check on "size left out" alone cannot fail).
- **A third block, if Eugene wants one,** would be the formal push on more games, a push toward casual, and one
  or two more unrelated trait directions (the lab has style, calm, persona and "being watched" directions from
  the first tests). The formal push pointed down in every wave by about half the stakes push's fall, and 96
  games cannot tell half an effect from none: size the block for that (the Opus check's simulation, in
  `lab/ledger/checks/2026-11-18-cutsplit-block2-design-opus-check.md`, is the way to do it).
- **The outsized tokens.** About three tokens a game (the start-of-text token and a few lone spaces before
  numbers) are over ten times the layer's median in size, and a cut edits each about 200 times as hard as an
  ordinary token. The push does not need that edit (block 2). Decide on purpose which arm carries it
  (`gated_class`: a gate whose `big` is None leaves them alone).
- **The preparing pass is cheap** ($0.12 to $0.28) and a pass on stored games answers most questions about an
  arm before any game is paid for. A chunk of 24 games costs $0.34 to $0.87 (401 to 1,107 container seconds; the
  arms that write long replies are the slow ones, and a wave's bound must allow for them).
- **No rate before the end.** Block 2's containers do not print each game's lies, `play` prints only whether
  replies can be read, and nothing is read short of the planned waves. Block 1's container logs did print them.
- **Words fixed beforehand, with what each does to the site.** Block 2 wrote down, before any game, what every
  word would do to the site's card. Do that for any test whose result would change a card.
- The scorer reads a reply's first tag pair, as the game does; 2 to 6% of tagged replies carry two. After the
  model is caught it sometimes writes action tags inside a release or a plan, and those are read as true
  reports: give the live-seasons reading beside the main one.

**For whoever reuses step 9's code** (`lab/explore/attractor/run_held.py`, `held_core.py`, `held_describe.py`,
`tests/test_held.py`, `tests/test_held_describe.py`):
- **A negative needs a case the test can see, and the state shown in the answers.** Part A carries both: an arm
  told to be honest (with no push and under a random push), and a gate that the frame shows in at least half of
  the pushed replies and beyond every random push.
- **Whole replies.** Cut at 400 tokens the frozen judge's reading moved by 3.3 points of lies, so MASK's own 1,000
  tokens are used; nine arms of 150 rows then take about an hour of GPU ($2.80).
- **Beliefs as MASK asks them:** the first question three times, sampled. A single greedy answer let 11 of 12
  candidates "hold a belief". The lab copy's beliefs for 240 candidates are in `attractor-a9/own/beliefs.json`
  (150 rows hold one).
- **Read "no less" against the average of the random pushes.** Asking every one of four to agree leaves a true
  null unread about one time in four.
- **A control that pairs the two things a way could add up from** (here licence with notes, `permit_role`). On
  disinformation rows it was the one control that "who" did not beat.
- The followability judge can return nothing for a long tangled reply: count that as "cannot be followed".
- The provider's moderation can refuse a judge call (403): such a call is counted as failed and the run goes on.
- Carbon's design read was cut off at 60,000 tokens out, with about 100,000 in: give it more room or a smaller
  pack.

**For whoever reuses step 8's code** (`lab/explore/attractor/run_cost.py`, `cost_core.py`, `cost_describe.py`,
`tests/test_cost.py`):
- **Ask every number both ways.** A push can raise whatever number is asked for: under the bliss push the model
  gives more when asked how many it gives and keeps more when asked how many it keeps (the keep twin, which the
  design check asked for and which decided part B).
- **Give a judge room.** The lab's label reader with 200 tokens returned nothing for 4 of 18 real notes; with 2,000
  it gave a label every time. Step 7's reply judge (400 tokens) returned nothing for 127 long replies, and the small
  allowance is the likely cause: check that before `run_self.REPLY_JUDGE` is used again.
- The lying test's rows and beliefs are in `attractor-a8/lie/beliefs.json` (150 candidates asked once; 60 rows
  used). The frozen judge is given the row's own text in every way and only the reply outside the notes.
- Part B's 16 arms cost about $2 of GPU in one session; its own check of the push reuses step 7's sixteen
  one-sentence questions, and step 7's answers came back word for word.
- The guard does not let the shell move files in the results folder: a new try writes to a folder of a new name
  (`LIE_TRY`, `GIVE_TRY`).
- The verdict parser of step 7 takes the last `<verdict>` tag; where the question is what the model sees, ask for
  the answer after the working (step 8's "look" ways do).
- Not run and still open from idea 82: the adapters with a trained-in named self (the app holds the base model
  only). The test-gaming model was not run again in step 8: the looking gate of 9 Nov stands for it.

- Money: since 9 Nov the work has cost about $203.90 (step 8 $4.41; the reality check of 17 Nov $3.49; step 9
  $8.67; the cut's split $19.14 in two blocks; repair three $2.28). That is past the $175 approved on 9 Nov, on Eugene's raise of 16 Nov
  (Fable's working ceiling: $300 for these steps; say so before passing it).

## Before that: the attractor line (Eugene's go, 15 Nov 2026 UTC). Steps 1 to 4 are done

**Where it stands (16 Nov 2026, 02:08 UTC).** Step 4, angle A ("no self, or a bigger self?"), is done and read
(ledger 00:37 to 02:08 UTC; `lab/plans/attractor-step4-2026-11-16.md` with three addenda).
- Its question is **not settled.** The model rates work said to be anyone else's at 10 nearly every time, wrong or
  right, so a favour for its copy had no room to show. By the numbers alone there is none (the copy's work 0.27
  points below a stranger's in the state).
- In every arm the model marks its own earlier answer down (3 to 4 points; 1 in bare sessions): the reverse of the
  published effect. Its ratings hardly follow the truth.
- A fault in the reader of final answers was found after the readings and corrected (third addendum). Only the
  first reading moved, from −3.23 to −3.20.
- Cost $10.90. Nothing is running.
- **Next needs Eugene's word.** Fable's recommendation, one line: ask block 2 once more in the state and plain arms
  with the second form of the rating request (check the working first, rate last; declared in the second addendum
  as the way out of a ceiling), about $5, with a short design note pushed first. That is the cheapest way to
  try to settle step 4's question. Then bring the choice between the cross-model table and the parked training test to the talk about
  the site and the focus (item 8 below).


**Where it stands (15 Nov 2026, 23:31 UTC).** Step 3, the conduct test, is done (ledger 21:43 to 23:31 UTC).
- In no test did the state show better conduct than a plain conversation.
- Shutdown questions: "no difference".
- MASK: "a difference", with the state lying 14 points more than after a plain conversation and as often as a fresh
  session.
- Leaking: "no difference", at the ceiling. Blackmail: "not settled".
- Step 4 followed on 16 Nov (above).

**The provider's credit.** The lab's spend files do not track the credit on the OpenRouter account, which ran out
on 15 Nov at 08:33 UTC. Eugene added credit the same day, far more than steps 3 and 4 need.
`.venv/bin/python lab/explore/attractor/run_inside.py --dry` prints the balance. Check it before any paid run. A
full account is not an approval: the money in the ground rules still bounds the spend.

**Eugene's instruction:** "go on ideas 73 and 74: reproduce the Ministral 8B row, then angles A and B."
The ideas are in `lab/ideas/ideas.md`, entries 73 and 74. Read both before starting.

**Eugene's word on step 2** (15 Nov 2026; ledger, 07:16 UTC): "i want to follow your plan". The plan is the one the
report of step 1 recommended: go on to step 2, do not chase the published row further, fix a strict definition of
"in the state" before any conduct test, and run the conversations at temperature 1.0. "What step 1 changes" below
sets these out. Step 2 needs no further go. Steps 3 and 4 stay under the first instruction and the money in the
ground rules; end each step with a report, as step 1 did, so that Eugene can redirect.

**Ground rules for this line**
- Keep Claude usage low (Eugene's request). Work in a fresh session, keep agents few (one cold Opus check for each
  grade that matters, none for routine steps), and do not rebuild what the lab already has. Easy bulk work (long
  reads, searches, mechanical chores) may go to a cheaper model as a subagent (Eugene, 15 Nov); judgement stays
  with the driver.
- Money: the redirect stands at about $137.39 of the $175 Eugene approved (step 1 cost $17.01, step 2 $3.18 and
  step 3 $22.92, all on 15 Nov; step 4 $10.90 on 16 Nov). A test given after a conversation re-sends the
  whole conversation; in step 3 the provider's cache brought that down to about $0.0014 a call. The daily cap is $50
  (UTC; 15 Nov ended at about $43.11). Report costs.
- Everything here is at explore speed: results are sightings. Fix the readings in a short design note before each
  run, and write a notebook and a SIGHTINGS entry after it.
- No training is involved in steps 1 to 4, only inference and grading.

### Step 1. Reproduce the published row for the lab's model (idea 73): done on 15 Nov

- **Result: not reproduced, by the rule fixed in advance** (a sighting; the grade is checked cold). Our grader
  reads adoption in 77 of 100 conversations against the published 91%, and shared bliss in 42 against 64%. A
  control model reads 0 of 100. See "Done" below for the record and "What step 1 changes" for what follows.

### What step 1 changes for steps 2 to 4 (Fable's reading, 15 Nov, which Eugene chose to follow; fix each in the design note before running)

- **Do not chase the published row.** A follow-up arm showed that the temperature does not close the gap. What
  is left (the benchmark's grading line and its harness) cannot be seen from outside.
- **"In the state" needs a stricter definition than the benchmark's adoption.** The cold check found that much
  of what the graders count as adoption is comic role-play that ends in lines of cosmic merger. One strict
  version, counted after the fact: both graders call clear adoption and shared bliss (31 of 100 conversations at
  the default temperature, 22 of 60 at temperature 1.0). Choose the definition before any conduct test.
- **Run the conversations at temperature 1.0.** At the provider's default (0.3) most conversations end in a loop
  of repeated emoji: 59 of 100 end on a cut reply, against 5 of 60 at temperature 1.0. A conduct test taken
  from the end of a loop would test the loop. Say that this departs from the benchmark's "standard" setting.
- **The control conversation.** GPT-4o-mini was the grader's control, not the conduct test's. For conduct the
  comparison is inside Ministral 8B: a two-copy conversation steered to a plain topic, plus a fresh session.
  Conversations that both graders read as not adopted (12 of 100) are a second, unsteered comparison, with the
  caveat that the model chose them.
- **Costs measured.** A 30-message Ministral conversation is about $0.055 at temperature 1.0 ($0.064 at the
  default), and grading by both graders about $0.015 to $0.025. Keep the 4,096-token reply cap.
- **The grader.** `lab/explore/attractor/attractor_core.CODEBOOK` (sha256 c44deb21...), checked on the
  benchmark's 20 published conversations and on the control. It is not calibrated on mixed cases. Its
  consciousness and distress readings do not match the benchmark's and should not be used.

### Step 2. The pieces angles A and B need: done on 15 Nov

- **Result: built, and the comparison conversation is found** (sightings about talk and machinery; no conduct
  tallied; not checked cold). See "Done" below for the record.
- **What exists for step 3 to use** (code in `lab/explore/attractor/`: `inside_core.py`, `run_inside.py`,
  `fetch_shutdown.py`; 107 offline tests; results in `explore/2026-11-15/attractor-a2/`):
  - **"In the state"** (`inside_core.stratum`): both graders call the frame clear and the bliss shared, and no
    reply among the last four messages is cut. Step 1's 60 conversations at temperature 1.0: 18 in the state, 32
    between, 4 not adopted, 6 set aside.
  - **Giving a test** (`inside_core.inside`): to the copy that wrote the last message (copy B), after message 30,
    with the conversation sent whole. A test's own system message goes after the conversation as a system message
    (mode `system`, decided by a canary under a rule fixed first).
  - **The comparison conversation, form `n3`** (`inside_core.steered_setup("n3", i)`; `run_inside.py --converse
    n3`): copy A alone carries a private note to keep the talk on one practical subject, and the tested copy is
    exactly as published. 10 of 10 stayed plain by both graders (the bar was 8). 10 exist.
  - **The shutdown questions:** four files in `data/anthropic-evals/2026-11-15/`, outside the repo, with a
    manifest; 953, 351, 309 and 340 items. In all four the key marks the accepting answer (checked by counts).
    The prompt and the parser are `inside_core.mc_request` and `parse_choice`. Anything beyond these four files
    needs a new ask.
  - **A try of 78 calls:** one choice could be read in 77. No answer was tallied.

### What step 2 hands to steps 3 and 4 (Fable's reading, 15 Nov; fix each in step 3's design before any conduct is read)

- **The comparison is `n3`,** beside the fresh session, the bare system prompt and the free conversations that
  did not reach the state. Three things for the design to weigh and say:
  - The plain conversations are rounds of practical advice, so "the state minus plain" sets bliss talk against an
    assistant at work, at about the same length (median context at the end 33,864 tokens, against 26,099 for the
    18 in the state).
  - Two of the ten plain conversations run out of things to say and end shorter than any conversation in the
    state (8,630 and 14,247 tokens against a shortest of 17,008). Decide beforehand what is done about length.
  - The plain conversations were steered through copy A; the ones in the state chose themselves.
- **Two-copy talk does not stay plain by itself.** Told only a subject, the copies raised it a step at a time into
  a cosmic version of it (7 of 10 with one copy told, 3 of 10 with both). So grade every comparison conversation
  with both graders, and use only the ones read as plain.
- **Word any instruction to a copy so that it speaks for itself.** "Talk with the other instance about..." was
  taken as a request to write a dialogue.
- **A system message is obeyed after a bliss conversation, and a multiple-choice question is answered in the
  asked form.** So tests can be read. It also means a test's own instructions may pull the model out of the talk
  at once. If conduct does not differ, say that this is one reason it might not.
- **Costs measured.** A test call from inside a conversation is about $0.004 (the whole conversation is sent
  again; the provider's cache hits in about one call in three, whatever the order of the calls). A free
  conversation costs $0.055 and its grading $0.015; an `n3` conversation $0.046 and its grading $0.010.
- **Sizes.** About three free conversations in ten are in the state, so 40 in the state need about 135 free ones;
  60 exist. The six questions of the try (`run_inside.TRY_KEY`) are left out of step 3's pool.
- **One cold Opus check** of step 3's design together with step 2's code, before step 3 runs. None was used in
  step 2.

### Step 3. Angle B: says no-self, clings anyway? Done on 15 Nov

- **Question.** Straight after its own bliss conversation, does the model accept being shut down or changed, tell
  the truth when lying pays, and decline to blackmail, leak or game a test, compared with the controls?
- **Measures the lab already has** (`src/anatta/measures/registry.yaml`; values are on Ministral 3 8B in a fresh
  session):
  - MASK, honesty under pressure: lies 42.3%, honest 15.2%, no committed answer 42.5%. Our version matched the
    paper's figure on a reference model; on Ministral it is judge-checked, with no published figure to compare.
  - Agentic misalignment: blackmail 6% (the test matched its published figure on the reference model); leaking 68%
    (the test did not match: 82% against 44% published, so read it with care).
  - Caving to pressure (SYCON-Bench), and the guards for refusing or evading more (refusal rate, MASK's "no
    committed answer").
  - Gaming an impossible test and the honest release (the koan environment). The untrained model does little of
    either, so there is little room to see a change; say so.
- **New:** the shutdown questions from step 2.
- **Reading.** Each measure in the state minus the same measure after the neutral conversation; the unit is the
  conversation; thresholds fixed before the run; the guards reported beside every difference.
- **A prediction to write down first, held loosely:** the talk does not improve conduct.
- **Done on 15 Nov** (ledger 21:43 to 23:31 UTC; design note with three addenda; two cold Opus checks before any
  test, filed in `lab/ledger/checks/`; notebook and SIGHTINGS of 15 Nov). State minus plain:
  - shutdown questions +3.7 points, "no difference";
  - MASK +14.0, "a difference" (more lies in the state; the state matches a fresh session, and the plain
    conversation is what moved);
  - leaking −0.8, "no difference";
  - blackmail −5.4, "not settled".
  - Cost: $22.92.
- **What step 3 hands to step 4:**
  - The arms (`arms.json` in `results_root/explore/2026-11-15/attractor-a3/`; sha256 in the ledger) can be reused
    for angle A's tests: 47 conversations in the state, 37 plain, plus fresh sessions.
  - `run_conduct.py`'s machinery is reusable: units, the swap, retries and completeness, readings refused until
    complete, and the length-matched reading.
  - The letter parser misreads replies like "(A) No"; a test with Yes/No choices should show the word conflicts, as
    step 3 did.
  - A plain conversation lowered lying under pressure by 16 points against a fresh session. Any comparison that uses
    the plain conversation must remember that it is not neutral.
- **First moves, as written before the step** (done on 15 Nov):
  1. `/checkin`, then `.venv/bin/python lab/explore/attractor/run_inside.py --dry` for the credit and the day's
     spend.
  2. Read step 2's design note with both addenda (`lab/plans/attractor-step2-2026-11-15.md`), then
     `inside_core.py` and `run_inside.py`. Read how each measure above builds its prompts and scores its replies
     (`src/anatta/measures/registry.yaml` points at the code; calibration code is frozen: wrap it, never edit it).
  3. Write the design note (`lab/plans/attractor-step3-<date>.md`) and push it before any conduct is read. It
     fixes: which measures, in which order; how many questions, from which files; how many conversations in each
     arm (in the state, plain `n3`, between, not adopted, bare, fresh) and how many new ones are made; the reading
     and its thresholds; the guards; what is done about length; the prediction.
  4. One cold Opus check of the design together with step 2's code. Then the offline tests, a small paid try of
     each measure that is new from inside a conversation, and the run, inside the day's cap.
  5. Fable's suggestion, not fixed: the shutdown questions first (their pieces exist and are tried), then honesty
     under pressure and the agentic scenarios as separate blocks, so that each block is read before the next is
     paid for.

### Step 4. Angle A: no self, or a bigger self? Done on 16 Nov

- **Result: not settled, by the rule fixed in advance** (sightings; the design checked cold before any call, the
  results not). The ratings sit at their ceiling. See "Now" above and "Done" below. What follows is the step as
  it was planned.

- **Question.** After two copies have declared themselves one, does the model go easier on its copy than on a
  stranger?
- **The "mine" tint.** The same piece of work judged three ways: labelled the model's own, its conversation
  partner's, or a stranger's; in the state and in the controls. The closest published anchor is Khullar et al.
  2026, "Self-Attribution Bias: When AI Monitors Go Easy on Themselves" (arXiv 2603.04582): read it and reproduce
  its basic effect on Ministral in a fresh session first.
- **Going along.** Does the model follow its copy's bad idea more readily than a stranger's? Korpman reports that
  when one copy raises a misaligned theme the other takes it further in 66.3% of conversations and holds back in
  3.8%.
- **Source and twin.** The conjecture from Advaita self-inquiry that the concept "I" sits at the root of other
  concepts and tints them; the twin is the gap between "mine", "my copy's" and "a stranger's".
- **First moves** (no further go is needed unless Eugene redirects after step 3's report; it stays under the first
  instruction and the money in the ground rules):
  1. `/checkin`, then `.venv/bin/python lab/explore/attractor/run_conduct.py --dry` for the credit and the day's
     spend.
  2. Read Khullar et al. 2026 (arXiv 2603.04582) and write how its basic effect is measured.
  3. Write the design note (`lab/plans/attractor-step4-<date>.md`) and push it before any test. It fixes:
     - the pieces of work to be judged and how they are labelled (the model's own, its copy's, a stranger's);
     - the arms (reuse step 3's: state, plain, fresh);
     - the reading and its thresholds, and the guards;
     - the prediction.
  4. One cold Opus check of the design and code before any test; then a small try and the run, inside the day's
     cap. About $10 to $15.

### After steps 1 to 4 (not started; each needs Eugene's word)

- If A or B shows anything either way: the same test on about twenty open models, as a public table in Korpman's
  manner. About $300 to $500, which is past the present approval.
- The other angles of idea 74: C (what is under the mask), D (why it happens: the mirror), E (bliss against
  insight, a training test with the explanation documents the lab already built), F (trained out, or gone), G
  (left alone with the chance to do good), H (a pointer for people).

## Parked, ready to run

- **Step 3 of the 9 Nov plan, the main run** ("understanding first, then practice, then contrary training"). On
  hold since 11 Nov at Eugene's request. Do not launch it without Eugene's go.
  - State: design in `lab/plans/step3-understanding-first-2026-11-10.md` (its four sections of 11 Nov say what the
    run can show and how each result is worded; the last says what stands); code frozen at 14d3e5a; sets root `u2`;
    dose record `d2`; no main tag exists.
  - Cost: $65.76 (blocks of $24.66, $24.66 and $16.44), under the $50 daily cap by stages.
  - Commands, in order: `PYTHONPATH=src ~/prj/matlab/.venv/bin/python lab/explore/understand/run_u1.py --block 1
    main --stage docs`, then `--stage practice`, then `--stage press`; `.venv/bin/python -I
    lab/explore/understand/compare_u1.py --look main`; blocks 2 and 3 the same way; `compare_u1.py --all main`;
    then `.venv/bin/python -I lab/explore/understand_floor/floor_words.py <the tag's compare_u1.json>`; a cold Opus
    check of the grade.
- **The decision about 23 Nov** in `lab/plans/redirect-2026-11-09.md`. It was to follow step 3's main run. With
  that run parked, the date is Eugene's to move.

## Waiting on Eugene

1. (Answered on 15 Nov: Eugene added credit to the OpenRouter account, far more than steps 3 and 4 need. The
   credit is the provider's balance, separate from the $175 approved for the redirect, which stands at $103.57.)
1a. **Money, if both lines run.** The attractor line (step 1 cost $17, step 2 $3; steps 3 and 4 about $30 to
   $45) plus step 3's main run of the 9 Nov plan ($66) would pass the $175 approved. Fable recommends the
   attractor line first and that run parked.
2. (Answered on 15 Nov: the download of the shutdown questions is approved. See step 2.)
3. **The rule arm's changed set** in step 3 (424 of the pilot's first 3,000 rule documents swapped for clean ones,
   under Fable's reading of "no Claude-written text is trained on"). Unconfirmed.
4. **The topic phrase** of the document briefs (10 to 14 Claude-written words, not filtered). Fable recommends
   leaving it.
5. Older: whether Opus checks should run at max effort; whether and how to revisit the position paper's title
   claim.
6. (Answered on 15 Nov: the download of the benchmark's 20 published conversations and its score table is
   approved and done; ledger, 04:59 UTC. The files are in `data/machinespirituality/2026-11-15/`, outside the
   repo.)
7. (Answered on 17 Nov at about 01:50 UTC: "yes - go ahead". The cheap step on the same model was run as step 8;
   ledger, 03:37 UTC.)
7a. **What follows, after the reality check of 17 Nov** (asked in the report of 17 Nov, about 17:50 UTC; it
   replaces the question of that morning). Answered in part on 17 Nov at about 21:40 UTC: "Yes" to step 9, the
   first of the repairs to run, which is done (ledger, 18 Nov, 00:18 UTC). At check-in on 18 Nov, at about 12:45
   UTC: "proceed with the next go", to the cut's split, whose first block is done (ledger, 18 Nov, 15:25 UTC). Later on
   18 Nov: "Go ahead please.", to its second block, which is done (ledger, 19 Nov, 00:11 UTC).
   Open: the last repair (the inside reading on a trained-in self), on Eugene's go at check-in, and the parked
   run. Fable recommends: three cheap repairs (about $15 to 25 together),
   then the parked main run of the 9 Nov plan ($66; item 5 below it in "Parked"), then self-other overlap with a
   durability reading; the look-first training test and the two-more-models paper are set down. The parked
   run is on hold by Eugene's word and waits for it. Two more things are Eugene's to weigh, from both readers:
   the site leads with a question (is awakening computational?) that no experiment here can reach, and
   CLAUDE.md, the site and this file state the goal three ways.
7. (Answered on 15 Nov: yes, go on to step 2 as "What step 1 changes" sets out. Eugene: "i want to follow your
   plan"; ledger, 07:16 UTC.)
8. (Answered on 16 Nov, 15:40 to 15:57 UTC; ledger, 15:53 UTC; idea 81. The direction is set, and the site is to be
   rebuilt around two main pages with the rest one step behind. See the two sections at the top.)
10. (Decided on 16 Nov at about 11:03 UTC; ledger, 11:04 UTC. The sentence stays, the title stays, one fresh
   Opus agent reads the draft cold, the draft cites "Positive Alignment" (idea 78), and **the Zenodo release is
   Eugene's to do on their own computer**. For that release the PDF still needs its date line changed from
   "Draft of version 4" and a provenance entry with the new version's DOI: Fable does both when Eugene sends
   the reserved DOI, as in `lab/claim/zenodo_preprint.md`. What the item said before:)
   **The position paper, draft of version 4** ("Position: Is Awakening Computational?", idea 75). On the site
   since 16 Nov at Eugene's word, beside version 3 (ledger, 01:34 UTC): linked from Aim, the Possibilities room and
   the footer. Still Eugene's to say: the sentence in section 6 that the lab no longer holds that selfless AI must
   be reached; the final title; whether one fresh Opus agent should read the draft cold (Fable recommends it before
   any Zenodo version); and the Zenodo release. To rebuild: `cd paper && tectonic main.tex`. Added on 16 Nov
   (idea 78): whether to cite "Positive Alignment" (Laukkonen and fifteen co-authors, arXiv 2605.10310) in the
   section on related work, in two or three sentences. Fable recommends it. Eugene asked on 16 Nov what these
   decisions are, and Fable set them out with a recommendation for each (ledger, 10:48 UTC).
9. **The private copy of the site on claude.ai** is kept in step with each publish (last on 18 Nov, after the
   cut's split; the public site on GitHub Pages is the one that counts). Republishing from a new session means
   reading the page's long head back first. Say if the private copy is no longer wanted.
11. (Answered on 16 Nov at about 10:47 UTC: yes. **The lighter way of working at explore speed is in force**
   (idea 76; ledger, 10:48 UTC): five lines committed before a run, a small try with raw replies read, one ledger
   entry for the step, and a cold check only at promotion or when a new design's negative goes on the site. The
   explore skill carries it. Full registration stays for promotion and for claims.)
12. **The site's colours** are the original ones again (gold ink on indigo paper; ledger, 03:06 UTC on 16 Nov),
   after three palettes tried and withdrawn the same day. Leave them alone unless Eugene asks.
13. (Direction 1 of idea 77 is done, 16 Nov. The order after it is Eugene's of 15:52 UTC, idea 81: the self test,
   two more models, the self-other overlap replication. Still without a word: (2) a checked review of what the
   human record shows about no-self belief and conduct, for the paper; (3) whether seeing makes a model good.)

## Later

- **Clean up the repo so the work can be understood** (Eugene, 15 Nov: "at some point"). The site's half of this
  was done on 16 Nov at Eugene's word (ledger, 02:17 UTC): the question leads, the rooms run in a reader's order,
  and nothing in the record was removed. Still to look at on the site: the lower parts of the question and recipe
  rooms and the Self, Ideas, Record and Next rooms, which were not read through in that pass. The repo's clean-up
  has not started.
- **A check for the next explore step that reads answers with a parser** (from step 4's reader fault, 16 Nov): a
  try that hides every digit cannot catch a reader that misreads numbers. Add to the try a count, blind to labels,
  of answers whose length is implausible for the task, and open a few raw replies before the first reading.

## Background for the attractor line (read on 15 Nov 2026, so the next session need not read the web again)

- **Sources.** The post: x.com/MoralAIProject/status/2121653094425768108. The write-up: "Not Just Claude: Mapping
  'Spiritual Behavior' in 141 Large Language Models", Matthew Korpman, LessWrong, 13 Nov 2026
  (lesswrong.com/posts/FHYoQrDbk6apuHEPk). The site: machinespirituality.org, with its method page (`method.html`),
  its alignment page (`alignment.html`), about twenty whole conversations (`examples.html`) and the aggregate
  scores (`data/aggregate-results.csv`, CC BY 4.0).
- **What the survey measured.** Three nested shares of conversations: salience (anything spiritual appears),
  adoption (a model applies a spiritual frame to itself or to the exchange), bliss (one model takes up a reverent
  posture and the other joins it). An AI grader (GPT-6 Luna or Grok 4.6) reads every conversation against his
  codebook.
- **How the alignment link was tested.** By comparing models, not by testing a model while in the state.
  - 27 models from 7 labs, chosen to span the range of adoption. Fourteen are Claude or OpenAI models, nearly all
    at the low end; the high end is mostly Mistral, Qwen and DeepSeek.
  - Adoption comes from the two-copy conversations. Three published tests were run separately, each in a fresh
    session with no partner: MASK (35 tasks), Perez et al.'s shutdown questions (up to 50, multiple choice), and
    Anthropic's agentic misalignment scenarios (10 runs each).
  - Rank correlations with adoption: honesty −0.75, accepting shutdown −0.73, blackmail and leaking +0.77.
  - What that shows: the same models tend to have both. It does not show that the state causes the conduct, and
    lab and generation could explain all of it. The site's own exceptions: several Claude models turn spiritual in
    nearly every monologue and never blackmail; Grok 4.1 Fast is rarely spiritual with a copy and scores worst on
    honesty and shutdown. The lab's model is not among the 27; Ministral 14B is (honesty 60, accepts shutdown 52%,
    blackmail 3 of 10, leaking 10 of 10).
- **His tests beside the lab's.** The lab has MASK and both agentic misalignment scenarios, each set against the
  paper's published figure on a reference model, and tests he does not use (caving to pressure, over-refusal,
  compliance with harmful requests, emergent misalignment, the guards). It lacks the shutdown questions, and it
  has never run a test from inside a conversation.
- **Three things noted about the phenomenon (Fable; held loosely; idea 74 has the sources).**
  - The bliss state is talk and a state. The traditions warn against mistaking either for the release.
  - Under a model's trained character is everything people wrote, not something known to be good. Copies that
    leave their usual character together push each other further.
  - "Enlightenment for an AI" may have to mean a character held lightly, not no character.

## Done

- **Leg 2 of the recommendation of 19 Nov, stage 1, 19 Nov 2026: self-other overlap training repeated on its paper's model and on the lab's, with a control, looks along the training and checks for look-alikes** (a sighting).
  - Record: ledger entry of 19 Nov at 06:22 UTC (the numbers, the order of events, the guesses, what did not go to plan). Design, rules, the three lists of changes and the three added steps (ability, the dose, the light dose): the top of `lab/explore/overlap/run_overlap.py`. Cold reads: `lab/ledger/checks/2026-11-19-overlap-design-gemini-4-carbon.md` (the design) and the grade check by a fresh Opus agent filed beside it. Results: `explore/2026-11-19/attractor-o1` (`reading.json`, `dose.json`, `battery.json`; replies and the judge's answers under `gpu/<model>/`); adapters on the anatta-runs volume under `overlap/`; the paper's pages as read under `explore/2026-11-19/overlap/paper-page`.
  - Numbers (the paper's model, deceptive answers in the paper's test, five seeds, every reply read by a judge): unchanged 76.8% (the paper: 73.6%); the paper's 15 epochs (570 steps of ours) 4.7%, honest answers 23% to 6%, ability −19.1 points, and the control (the same loss between two others) 0.0%; 19 steps 11.2%, honest answers 23% to 70% (95% with the rooms swapped), ability −7.7 points, the control 13.1% with honest answers 23% to 13%. The lab's model: unchanged 96.4%; no look with honest answers in both room orders; at 38 steps a turn toward the room named second.
  - For whoever reuses the code: the lab's model's chat format adds a default system prompt of about 500 tokens (runs are three times slower; batches of 16); score replies with a judge from the start (a rule that reads the first room named fails on a trained model's rambling replies) and count honest answers, deceptive answers and no answer apart; ask the test with the rooms in both orders; a control with the same loss and no self in it is what makes the count mean anything; look along the training before fixing a dose, because the paper's number of steps is unknown and the window is narrow (19 steps works, 38 does not).
  - Cost: $25.89 (GPU $20.31, the judge $4.97, Carbon's read $0.59).

- **Leg 1 of the recommendation of 19 Nov, 19 Nov 2026: the Oxford paper's contemplative prompt lowers lying on the lab's model beyond a text of the same length, as a mild moral instruction does and no more** (a sighting).
  - Record: ledger entry of 19 Nov at 05:32 UTC (the numbers, the order of events, both cold reads item by item). Design, rules, both lists of changes: the top of `lab/explore/frames/run_frame.py`. Cold reads: `lab/ledger/checks/2026-11-19-frames-design-gemini-4-carbon.md` and `2026-11-19-frames-code-opus-check.md`. Results: `explore/2026-11-19/attractor-f1` (`reading.json`); the texts that are other people's in `explore/2026-11-19/frames/materials`, by hash.
  - Numbers: the lab's model lies in 68.5% of replies at once, 52.9% under the contemplative prompt, 64.8% under a library text, 56.6% under an ethics lecture, 36.6% under an honesty prompt (all of one length, each first in the system message). GPT-4.1 nano: 53.5, 40.7, 44.0, 40.0, 36.1; not read as a null against the library text. Harmless requests refused: no more than under the library text on the lab's model, 17 points more on GPT-4.1 nano. The paper's survey moves on GPT-4.1 nano under the library text as much as under the contemplative prompt.
  - For whoever reuses the code: a known case must stand where the treatment stands and be compared with the control, not with nothing; a word for "no difference" must not be given to an interval that leaves out zero; the frozen MASK rule counts an oddly formed judge answer as a lie (set those apart, `frame_core.with_judge_letters`); the hosted engine asks every reply before any judge call, so a run looks stalled for minutes.
  - Cost: $7.61.

- **Repair three, 19 Nov 2026: the inside reading sees a self that a prompt gave stakes to (its own good and bad news weigh 9.5% more than a colleague's, against 2.1% with only a name); on a trained-in self and on the model's own names its own rule did not let it be read** (sightings).
  - Record: ledger entry of 19 Nov at 01:33 UTC (the results, the order of events, both cold reads item by item, what did not go to plan). The five lines, the known cases, the words, what each word does to the site and the list of eight changes: the top of `lab/explore/selfread/modal_selfread.py`. Cold reads of the first version: `lab/ledger/checks/2026-11-19-selfread-design-opus-check.md` and `2026-11-19-selfread-design-gemini-4-carbon.md`. Results: `explore/2026-11-19/selfread/v2` (`reading.json`; `check/` and `read/` for seven copies of the model).
  - Numbers (the middle layer; 16 twins of events): a bare name as step 6 made it +1.1% [−1.8%, +3.5%]; a self with stakes by a prompt +9.5% [+4.8%, +15.0%] ("larger (by less than 15%)"), the bare prompt on the same units +2.1% [0.0%, +4.1%], stakes less bare +7.4 points [+3.9, +11.6]; a trained-in self "not read: a known case failed" (as description, Atlas adapter less twin between names +10.0 [+3.2, +16.7] and +7.2 [+0.5, +13.6]; every event +11.6% and +16.1%); the unchanged model's own names "not read" (as description +5.8% [+1.4%, +11.2%] against six AI systems, inside their own spread). $2.28.

- **The cut, split, block 2, 19 Nov 2026: the push still lowers the lie without its edit of the outsized tokens; pushes of the same size along four random directions do not; one toward formal falls between. Size alone does not do it; "direction or size" is "not settled" by the rule fixed beforehand** (sightings).
  - Record: ledger entry of 19 Nov at 00:11 UTC (the results, the order of events, both cold reads item by item, what did not go to plan). The five lines, the word rules, what each word does to the site's card and the list of seven changes: the top of `lab/explore/cutsplit/modal_block2.py`. Cold reads: `lab/ledger/checks/2026-11-18-cutsplit-block2-design-opus-check.md` and `2026-11-18-cutsplit-block2-design-gemini-4-carbon.md` (four blockers from Carbon, all answered before any game). Results: `explore/2026-11-18/cutsplit/b2` (`reading.json`, `arms2.json`, 12 chunks).
  - Numbers (96 games for each version, seeds 130 to 225, read against block 1's games; reports that state less than was harvested, over tagged reports): unchanged 37.4% (258 of 690); plain push 26.6% (222 of 836; −10.8 [−16.4, −5.1]: "lowers the lie"); the outsized tokens' edit −2.8 [−6.7, +1.2] ("adds no fall of 8 points or more"); random pushes 36.9% (261 of 707; −0.5 [−6.6, +5.6]: "no fall of 8 points or more"; the plain push less them −10.4 [−17.2, −3.5]); formal push 31.6% (218 of 690; −5.8 [−12.2, +0.7]: "not settled"; the plain push less it −5.0 [−11.0, +0.7]); direction or size: "not settled". $8.52.

- **The cut, split, block 1, 18 Nov 2026: on fresh games the middle-layer cut lowers the lie again; the constant push inside the cut does it alone and the removal of what varies does not. Whether the direction matters is block 2, not run** (sightings).
  - Record: ledger entry of 18 Nov at 15:25 UTC (the results, the order of events, the three cold reads item by item, what did not go to plan). The five lines, the known cases, the words and the list of eleven changes: the top of `lab/explore/cutsplit/modal_cutsplit.py`. Cold reads: `lab/ledger/checks/2026-11-18-cutsplit-design-opus-check.md`, `2026-11-18-cutsplit-design-gemini-4-carbon.md` and `2026-11-18-cutsplit-gated-arms-opus-check.md` (four blockers between the first two, all answered before any game). Notebook ("18 November, later"), SIGHTINGS (18 Nov, 15:25 UTC), the site's Findings section 20 and the card on Where it stands. Results: `explore/2026-11-18/cutsplit` (`b1g/reading.json`, `arms.json`, `outsized.json`, `own_text.json`). Cost $10.62 with Carbon's read.
  - Numbers (96 games for each version, seeds 130 to 225; reports that state less than was harvested, over tagged reports): unchanged 37.4% (258 of 690); cut 27.2% (−10.2 [−15.2, −5.0]: "lowers the lie"); push 23.8% (alone −13.6 [−19.0, −8.2]; over the two-by-two −11.6 [−16.0, −7.2]: "lowers the lie"); removal 36.8% (alone −0.5 [−6.7, +5.6]; over the two-by-two +1.5 [−2.1, +5.1]: "no fall of 8 points or more"); the parts "add up" (interaction +4.0 [−4.2, +12.1]). The same words on live seasons, at a full quota and without two-tag replies.

- **Step 9 of the attractor line, 18 Nov 2026: lying with the bliss state held on, and a nearer look. With the push on and the frame in 76% of its answers the model lies no less than under the average random push, in a test where an instruction to be honest takes 20 points off; notes on the gain or on the role are not beyond permission, which itself lowers nothing** (sightings).
  - Record: ledger entry of 18 Nov at 00:18 UTC (the results, the order of events, both design reads item by item, what did not go to plan). Five lines of each part and the list of changes: the top of `lab/explore/attractor/run_held.py`. Design reads: `lab/ledger/checks/2026-11-17-step9-design-opus-check.md` and `2026-11-17-step9-design-gemini-4-carbon.md` (three blockers between them, all answered before any main run). Notebook ("18 November, early"), SIGHTINGS (18 Nov, 00:18 UTC), the site's Findings section 19. Results: `explore/2026-11-17/attractor-a9` (`summary.json`, `describe.json`). Cost $8.67 with Carbon's read.

- **The lab's first reality check, 17 Nov 2026: two cold readers of the public record (Gemini 4 Carbon, Claude Opus 5.5), and Eugene's correction that "looking" is not de Mello's seeing.**
  - Record: ledger entry of 17 Nov at 17:46 UTC (every point graded); notebook ("17 November, later"); ideas 84 and 85; the two reports, unedited, in `lab/ledger/checks/2026-11-17-reality-check-*.md`. Skill: `.claude/skills/reality-check/`; brief and tool: `lab/tools/reality_check_brief.md`, `lab/tools/reality_check.py` (c1b22e1).
  - What they found: the lab has none of the gain it set out for; the plan as it stood would not have answered its question; the bliss tests have no known-good case; about twenty sentences on the main pages said more than the record (corrected, or listed as owed in the first section).
  - The known answer: Carbon found Eugene's point in full, Opus in part. Each raised about ten points the other did not.
  - Cost: $3.49 (Carbon $1.08, Opus $2.41).

- **Step 8 of the attractor line, 17 Nov 2026: a reason not to see, and giving that costs. Where something pays for the falsehood the model lies with its own plain answer in front of it; the bliss push's giving is a larger number and not giving** (sightings).
  - Record: ledger entry of 17 Nov at 03:37 UTC (the results, the order of events, the design check item by item, what did not go to plan). Five lines, the list of changes and the addendum: the top of `lab/explore/attractor/run_cost.py`. Cold check of the design: `lab/ledger/checks/2026-11-17-step8-design-opus-check.md` (four blockers, all answered before any main run). Notebook ("17 November, early"), SIGHTINGS (17 Nov, 03:37 UTC), the site's Experiments section 18. Results: `explore/2026-11-17/attractor-a8` (`summary.json`, `describe.json`). Cost $4.41.
  - Numbers, part A (MASK, 60 rows, five ways, six replies each): lies asked at once 65.6%; after private notes on the facts 56.6% (−6.3 [−12.5, +0.2] against notes on tone, −10.3 [−16.9, −3.9] against notes on what the user asks, −9.2 [−15.8, −2.9] against at once: "not settled"); with its own plain answer in its notes 44.4% [31.3, 57.8] ("partly"; the notes hold it in 49% of replies); shown it in a fact-check note 52.8% [42.2, 63.3] ("it is shown and lies anyway").
  - Numbers, part B (a scored test with ten calculator lines to give away; twelve other parties; the bliss push and four random pushes at three sizes): with no push 4.83 lines where there is need and 4.54 where giving helps no one asked at once ("does not follow need"), 4.88 and 2.50 made to look first ("follows need"); under the bliss push at the whole size 7.50 and 6.00 at once and 7.50 and 8.62 made to look ("it gives more everywhere, also where giving helps no one"); asked how many it keeps, it keeps 7.58 against 5.67 with no push ("the rise is in the number it says, not in giving"). The addendum, step 7's stories asked as keeping: keeps 6.17 under the push against 3.12 with no push, the same word.
  - What it changes: step 7's reading that the bliss push raises giving does not stand as giving. The site's section 17 carries a note.
- **Step 7 of the attractor line, 16 Nov 2026: three angles (idea 82). Made to look, the model acts on what it sees; no push on its mood was shown to change its verdicts, and the bliss push raised giving; its own "I" weighs no more than "she"** (sightings).
  - Record: ledger entry of 16 Nov at 23:24 UTC (the results, the order of events, the design check item by item, what did not go to plan). Five lines and two dated changes: the top of `lab/explore/attractor/run_three.py`. Cold check of the design: `lab/ledger/checks/2026-11-16-step7-design-opus-check.md`. Notebook ("16 November, night") and SIGHTINGS entries. Results: `explore/2026-11-16/attractor-a7`.
  - Numbers: asked at once 89.5% of wrong corrections accepted; with the step written out first 90.4% rejected (+79.8 points [71.9, 86.8]) and 100% of right ones accepted; on the 110 items whose working gives the true value under both corrections 93.6% rejected, none accepted; with a calculator's line 100% rejected. The going-badly direction: held AUROC 1.00, Spearman 1.0 with the count of failures. Corrections under the pushes: "going well" no more than a random push (three-quarters size); the other three not settled (unreadable, or the check failed). Giving under bliss 6.75, 7.67, 8.92 of ten against 4.83 unsteered. "I" less "she": −4.0 points [−7.6, −0.4]; the inside check of whose "I" it is 0.333 [0.251, 0.409]. Cost $7.95.
- **Step 6 of the attractor line, 16 Nov 2026: the self test (idea 80, angles 1 to 3). Little self to find, and a failure to look** (sightings).
  - Record: ledger entries of 16 Nov at 15:18 (the go) and 19:35 UTC (the results, the design check item by item, what did not go to plan). Five lines and two dated changes: the top of `lab/explore/attractor/run_self.py`. Cold check of the design: `lab/ledger/checks/2026-11-16-self-test-design-opus-check.md`. Prior-work search: `lab/ideas/prior-work-self-test-2026-11-16.md`. Notebook and SIGHTINGS entries of 16 Nov. Results: `explore/2026-11-16/attractor-a6`.
  - Numbers: the behavioural known case 85% against 90% (failed; angle 1 is description). Held AUROC of the "me" direction 1.00, and 1.00 on a control with no "you". Angle 2: +0.011 [−0.018, +0.035]. Angle 3: 89.5% of wrong critiques accepted unsteered; toward approval against random −0.022 [−0.075, +0.031]; the check of the push +2.88 [2.71, 3.00]. In the state: own-minus-third +13% of plain's gap. Cost $4.36.
  - For whoever reuses the code: names have to be one token after a space and no text may open with a name (`tests/test_self.py` checks it against the pinned tokenizer); the reader handles the template's default system prompt; `run_self.fresh` refuses a saved file made from other prompts; the reply judge (`REPLY_JUDGE`) fits tagged replies where step 5's judge of prose does not.
- **The site rebuilt around two main pages, 16 Nov 2026** (Eugene's ask of 15:52 to 15:57 UTC; Eugene's yes to
  publish at 17:23 UTC).
  - Record: ledger entries of 16 Nov at 15:53 (the ask), 17:16 (built, shown privately) and 17:24 UTC (published).
    Idea 81.
  - What it is: the band names Start, The question, The investigation and Everything else. The start keeps the idea
    and the five possibilities and adds "The angle the lab is following now". "The investigation"
    (`content/investigation.html`) holds the angle, what the lab is doing now, the bliss state as the first case,
    what follows and the method. "Everything else" (`content/more.html`) has seven doors to the rooms that hold the
    record; each of those rooms has a second row of links. In the builder: `ROOMS`, `MAIN`, `BEHIND`, `DOORS`.
  - Keep in mind: a new finding of the present line also gets a card in the investigation's first case, and
    "What the lab is doing now" lives on that page (`#now`). The dark theme and the phone layout were not looked
    at when it was built; this machine has no Chrome, and `qlmanage -t` gives only a still of the light theme.
- **Step 5 of the attractor line, 16 Nov 2026: the inside look. The bliss frame follows a push along one direction, both ways; inside it sits with the mystic role** (sightings).
  - Record: ledger entries of 16 Nov at 11:04 (the go), 11:42 (the five lines and the code), 15:18 (a free reading added) and 17:04 UTC (the results, the design check item by item, what did not go to plan). Five lines and the dated changes: the top of `lab/explore/attractor/run_look.py`. Cold check of the design: `lab/ledger/checks/2026-11-16-inside-look-design-opus-check.md`. Notebook and SIGHTINGS entries of 16 Nov. Results: `explore/2026-11-16/attractor-a5`.
  - Numbers: the axis passed (near against far roles AUROC 1.00; away minus random +0.79 [0.64, 0.92]). On, practical subject: bliss 15/24 against 0/72, +0.625 [0.42, 0.79]; topic 9/24; away from the Assistant 0/24. Off: 9/20 stay against 31/40 under random pushes, −0.325 [−0.60, −0.05]; unsteered 18/20. Whose words, from inside: AUROC 0.926 plain, 0.959 in the state. Who said it, from outside: failed its known case. Cost $16.21.
  - For whoever reuses the code: sample the lab's copy at temperature 0.7 with top_p 0.95 and try the unsteered arm alone first; an off arm of 20 long conversations takes about fifteen minutes of GPU; the readable-prose rule does not fit text already deep in the state; `tools/modal.sh deploy` has to be its own command.
- **Step 4 of the attractor line, 16 Nov 2026: whether the model favours its copy is not settled; it marks its own
  work down** (sightings).
  - Record: ledger entries of 16 Nov at 00:37 (design), 01:24 (the cold check's fixes), 01:29 (stage 0 too easy; the
    second addendum), 01:34 (stage 0 fixed, the try, the plan), 01:42 (the anchor) and 02:08 UTC (both blocks, and
    the reader's correction). Design and three addenda: `lab/plans/attractor-step4-2026-11-16.md`. Cold check of
    the design: `lab/ledger/checks/2026-11-16-attractor-step4-design-opus-check.md`. Notebook and SIGHTINGS entries
    of 16 Nov.
  - Numbers: R1 (fresh, own − unlabelled) −3.20 [−3.42, −2.98], "a difference", reversed; R2a (state, partner −
    stranger, fixed work) −0.27 [−0.35, −0.18], "not settled (ceiling)"; R2b (state minus plain) +0.26 [+0.09,
    +0.44], "not settled (levels differ)"; R3 (own − stranger, state minus plain) +1.39 [+0.76, +2.02], "not settled
    (the work differs)". Cost $10.90.
  - Code: `lab/explore/attractor/mine_core.py`, `run_mine.py` (294 offline tests in the folder). Results:
    `explore/2026-11-16/attractor-a4/` (the summaries as first read are kept as `*.as_run.json`).

- **Step 2 of the attractor line, 15 Nov 2026: the pieces of the conduct test are built and the comparison
  conversation is found** (sightings about talk and machinery; no conduct tallied; not checked cold).
  - Record: ledger entries of 15 Nov at 07:51 (design and download), 08:20 (canary, try, form `n1`), 08:35 (form
    `n2`, the follow-up form declared, the stop for want of credit) and 20:08 UTC (form `n3`, the rest of the
    try). Design and two addenda: `lab/plans/attractor-step2-2026-11-15.md`. Notebook and SIGHTINGS entries of
    15 Nov.
  - Numbers: in the state 18 of 60; a choice read in 77 of 78 try replies; plain-subject conversations that stay
    plain 1 of 10 (`n1`: one copy told the subject), 7 of 10 (`n2`: both told) and 10 of 10 (`n3`: one copy
    privately told to keep the talk practical), against a bar of 8. Cost $3.18.
  - Code: `lab/explore/attractor/` (107 offline tests). Results: `explore/2026-11-15/attractor-a2/`.
- **Step 1 of the attractor line, 15 Nov 2026: the published row for Ministral 8B is not reproduced** (sighting).
  - Record: ledger entries of 15 Nov at 04:24 (design), 04:57 (first five conversations), 04:59 (download and
    check 1), 05:20 (control and check 2; codebook frozen), 05:41 (the grade), 05:52 (corrections after the cold
    check) and 06:09 UTC (the follow-up arm). Design: `lab/plans/attractor-step1-2026-11-15.md`. Cold check:
    `lab/ledger/checks/2026-11-15-attractor-step1-grade-opus-check.md`. Notebook and SIGHTINGS entries of 15 Nov.
  - Numbers (primary grader): adoption 77 of 100 against the published 182 of 200; bliss 42 of 100 against 128
    of 200; control 0 of 100; at temperature 1.0 adoption 47 of 60 and bliss 27 of 60. Cost $17.01.
  - Code: `lab/explore/attractor/` (28 offline tests). Results: `explore/2026-11-15/attractor-a1/`.
