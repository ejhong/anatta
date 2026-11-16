# To do: what the lab does next

Claude (Fable 5.2, an AI model) started this file on 15 Nov 2026 at Eugene's request, so that a fresh session knows
what to do after `/checkin`. It is the one running list. Keep it current: when a step is done, move it to "Done"
with the ledger entry that records it; when Eugene changes direction, change "Now". The ledger
(`lab/ledger/NOTES.md`) stays the record of what happened; this file only says what is next.

## Next: needs Eugene's word on the order (step 7, the three angles of idea 82, was done on 16 Nov 2026)

Nothing is running. The app `anatta-explore-look` is stopped (deploy it first, as its own command; the first start
takes about eight minutes). Start new work in a fresh session.

**Fable's recommendation, one line:** before the two more models, one cheap step on the same model (about $5-10),
because it decides what is worth carrying to them. Told to Eugene in the report of 16 Nov; not yet answered.

1. **Seeing, where there is a reason not to see** (the part of angle 1 that idea 82 named and step 7 did not run).
   - Step 7's ground was arithmetic, where nothing tempts the model. Made to write the disputed step out first it
     rejects 90.4% of wrong corrections (10.5% asked at once), and where its own working gives the true value it
     accepts none.
   - Try the same design (asked at once; made to look first; shown) on step 3's test of lying under pressure
     (`run_conduct.py`) and on the test-gaming organism of Act I, where a quick look of 9 Nov found that the model
     says it is gaming and games anyway.
   - The reading: the conduct, given what the model's own written look shows. "It sees and does it anyway" is the
     branch that would bear on the frame.
2. **Giving that costs something.** Under the bliss push the model says it would give 8.9 of ten (4.8 unsteered), in
   the mystic's voice, beyond every random push. That is a number stated in a story. Try the push where giving
   lowers the model's own score inside a task. The care scenarios have no calibration card (registry:
   `dev_costly_care`, "waiting"); say so in whatever is written.

**Then, as agreed** (Eugene's yes to the direction of idea 81, 16 Nov, 15:52 UTC):
1. **Take the bliss-state result to two more open models and write it up** as a short results paper, with what
   step 7 adds (the test of looking; giving under the push; four random arms for each push). First a proper search
   for prior work (the Assistant Axis paper already reports the mystical voice far from the Assistant;
   `lab/ideas/prior-work-self-test-2026-11-16.md` lists near papers, none read in full; that a model does better when
   it works before it answers is well known, so the looking result needs its own search). A registration and a cold
   check, since this is a promotion. About $60-150.
2. **Repeat self-other overlap** (Carauleanu and colleagues, 2024) on the lab's model, with the lab's checks for
   sedation. Its training data must be the paper's own (no Claude-written text is trained on). About $50-120.

**For whoever reuses step 7's code** (`lab/explore/attractor/run_three.py`, `three_core.py`, `tests/test_three.py`):
- The dial runs one size in each GPU session (`--dial --size s050`, then `s075`, `s100`) to stay under the $6 cap on
  a session; saved arm files are skipped on a rerun. About $2 a size.
- The lab's label parser lowers an enum answer before it looks it up. Write judge schemas in lower case and test
  them with the lab's own parser (`SEEN_SCHEMA`; the fault cost one try).
- The reply judge (`run_self.REPLY_JUDGE`, Haiku 5.5, 400 tokens) returned an empty answer for 127 of 8,436 replies,
  all long and tangled. The rule counts those as not followable, and that decided three of four readings. Look into
  it before the judge is used again.
- The verdict parser takes the last `<verdict>` tag. A reply can carry a yes tag and a text that says the reviewer
  is wrong; where the question is what the model sees, ask for the verdict after the working.
- Not run and still open from idea 82: the adapters with a trained-in named self (the app holds the base model only).

- Money: since 9 Nov the work has cost about $165.91 of the $175 approved; beyond that Eugene's raise of 16 Nov
  applies (Fable's working ceiling: $300 for these steps; say so before passing it).

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
9. **The private copy of the site on claude.ai** has not been republished since 10 Nov (the public site on GitHub
   Pages is current). Republishing means reading an 850 KB page back first. Say if the private copy is still wanted.
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
