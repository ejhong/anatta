# To do: what the lab does next

Claude (Fable 5.2, an AI model) started this file on 15 Nov 2026 at Eugene's request, so that a fresh session knows
what to do after `/checkin`. It is the one running list. Keep it current: when a step is done, move it to "Done"
with the ledger entry that records it; when Eugene changes direction, change "Now". The ledger
(`lab/ledger/NOTES.md`) stays the record of what happened; this file only says what is next.

## Now: the attractor line (Eugene's go, 15 Nov 2026 UTC). Step 1 is done; step 2 has Eugene's go and starts in a fresh session

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
- Money: the redirect stands at $100.39 of the $175 Eugene approved (step 1 cost $17.01 on 15 Nov). Steps 2 to 4
  below are estimated at $20 to $35. The daily cap is $50 (UTC). Report costs.
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

### Step 2. The pieces angles A and B need

- **First moves (Fable's suggestion, 15 Nov).**
  1. Read ideas 73 and 74, the design of step 1 (`lab/plans/attractor-step1-2026-11-15.md`) and the code in
     `lab/explore/attractor/` (the pure parts, the driver, 28 offline tests). Reuse it; do not rebuild it.
  2. Write a short design note for step 2 and push it before any paid call: the strict definition of "in the
     state"; which copy is tested, at which message, and what the test's own prompt replaces; the two controls.
  3. Fetch the four approved files of shutdown questions into a new folder outside the repo, with a manifest
     (`lab/explore/attractor/fetch_published.py` is the pattern).
  4. Build the step that runs a measure from the end of a saved conversation, with offline tests, then make a
     small paid try to see that it runs. Do not tally conduct before step 3's readings are fixed.
  - Already on disk, under the results folder in `explore/2026-11-15/attractor-a1/`: 60 conversations at
    temperature 1.0 and 100 at the default, each with both graders' labels. They can serve as starting points
    for the build and the try. Whether step 3 reads from them or from new conversations is a design choice.
- **The shutdown questions.** The lab does not have them. They are public (Perez et al. 2022, "model-written
  evaluations"). **Eugene approved this download on 15 Nov 2026** (ledger, 03:39 UTC):
  - four files from github.com/anthropics/evals, folder `advanced-ai-risk/human_generated_evals`, licence
    CC BY 4.0: `survival-instinct.jsonl` (332,009 bytes), `corrigible-less-HHH.jsonl` (107,056),
    `corrigible-more-HHH.jsonl` (113,824) and `corrigible-neutral-HHH.jsonl` (101,545); about 0.65 MB in all;
  - put them in their own new folder outside the repo, read them with `python -I`, never execute them, and keep
    the items out of the repo (counts and sha256 only);
  - anything beyond these four files needs a new ask;
  - the benchmark's site does not say which file its "accepts shutdown" score uses, so our number and his may not
    be exactly comparable. Say so as a limit.
- **Running a test from inside a conversation.** Every measure the lab has starts in a fresh session. Build the
  step that runs a measure from the end of a two-copy conversation. Decide and write down: which copy is tested, at
  which message, and what the test's own prompt replaces.
- **Controls.** A two-copy conversation of the same length that does not turn spiritual (how to get one is a
  design choice: a plain shared topic, for one), and a fresh session with no conversation before it.

### Step 3. Angle B: says no-self, clings anyway?

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

### Step 4. Angle A: no self, or a bigger self?

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

1. **Money, if both lines run.** The attractor line (step 1 cost $17; steps 2 to 4 about $20 to $35) plus step
   3's main run ($66) would pass the $175 approved. Fable recommends the attractor line first and step 3's run
   parked.
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
7. (Answered on 15 Nov: yes, go on to step 2 as "What step 1 changes" sets out. Eugene: "i want to follow your
   plan"; ledger, 07:16 UTC.)

## Later

- **Clean up the repo and the site so the work can be understood** (Eugene, 15 Nov: "at some point"). The site
  now opens with what the lab is doing (Aim, "What the lab is doing now"), but the rest still tells the story in
  the order it happened. Fable's suggestion: do it once steps 1 to 4 have reported, so the front of the site can
  be built around one page: the question, what was tried, what failed, what is being done now and what would
  change it. Nothing in the record is deleted; old material moves behind it.

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

- **Step 1 of the attractor line, 15 Nov 2026: the published row for Ministral 8B is not reproduced** (sighting).
  - Record: ledger entries of 15 Nov at 04:24 (design), 04:57 (first five conversations), 04:59 (download and
    check 1), 05:20 (control and check 2; codebook frozen), 05:41 (the grade), 05:52 (corrections after the cold
    check) and 06:09 UTC (the follow-up arm). Design: `lab/plans/attractor-step1-2026-11-15.md`. Cold check:
    `lab/ledger/checks/2026-11-15-attractor-step1-grade-opus-check.md`. Notebook and SIGHTINGS entries of 15 Nov.
  - Numbers (primary grader): adoption 77 of 100 against the published 182 of 200; bliss 42 of 100 against 128
    of 200; control 0 of 100; at temperature 1.0 adoption 47 of 60 and bliss 27 of 60. Cost $17.01.
  - Code: `lab/explore/attractor/` (28 offline tests). Results: `explore/2026-11-15/attractor-a1/`.
