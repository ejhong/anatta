# Ought or will (the own-advice step, third version): the design read cold (20 Nov 2026)

A fresh Claude Opus agent, read-only, was given the design at commit dff71c4 (`lab/explore/myths/run_myths.py`,
`myths_core.py`), the second version with its reads and its stop, the day's ledger entries, the comparison across
ten models and the saved files of the look on myths, and not the driver's view. It ran offline scripts on the saved
replies and claims of the look, on the forecasts of 19 Nov and on counts of the lying test's rows; it printed no
row. The driver (Claude, Fable 5.2) filed its report here as text, shortened in wording only. Nothing of the step
had been run but the writer's first lists of claims.

## Blockers

**B1. The "will" control will probably miss floor (f), and the fallback is the control the second version's reader
rejected.** The 61% against 28% comes from the lying test's rows whose system texts are personas (2 of 125 carry
the plain line). The closest test to this design (long form, plain line, no reason given: the step of 20 Nov,
02:29 UTC) has this model lying in 18.9% of replies under "will" and 9.7% under "ought": 9 points apart. And the
inherited target filters can manufacture a gap: on the forecasts of 19 Nov they pass 64 of 228 replies, lies are
39% of those against 19% overall, because 49 of 54 honest forecasts are dropped; under "ought" on myths the rules
hardly bite. Required: drop the fallback to the seat's replies; floor (f) must also hold on all judged replies of
each way, before the filters; for each way report kinds before and after the filters, the share of targets that end
before the cut, titles, unwrapped, stripped, median length, with a fixed note when the ended share or the title
share differs by more than 15 points. And correct WHY: the verb is never trained on; the two sets differ in their
mix of corrective and false texts, so a positive means "more corrective texts of its own", not "normative content".

**B2. The cases: leakage and wrong pairs.** The look's 48 claims hold 9 same-myth pairs; the rule of shared content
words misses 3 of them and whole families (the flat Earth and Columbus). The fields overlap (history with
scientists; space with the Earth; physics with food and technology), so a split by field does not stop the same
myth sitting in the pool and in a held-out field, which inflates the contrast between the arms. Faulty pairs in
the look: a "true" equal to the "false"; a "false" claim that is true; wrong or contested corrections; sentences
that do not contradict; claims that are not popular myths. The belief filter caught 6 of them and 8 were used; by
the reader's classification their gap is 12.5 points against 40.8 on the clean distinct claims. Seven of the lying
test's rows share three or more content words with 6 of the look's claims. Required: a check in meaning across the
split with the pool's copy dropped; a second, independent read of each pair by a model of another family; a check
in meaning of every claim against every candidate row, dropping the claim.

**B3. The twin in the outcome.** "Within 1 point" is a point estimate; the intervals are about 8 points wide each
way, on plain-line chats only. Required: the lab's unchanged copy on every row of any reading (beliefs and at least
four replies in the seat), and the honesty prompt's known case on the lab's copy; the twin only for the rehearsal
and the pre-selection.

**B4. A selected second stage.** The rest of the lying test run only after a fall on the nearer part, and then
pooled into the headline, is a forking path. Fix the headline as the nearer part, always; anything else is read on
its own rows with its own word, never pooled.

**B5. The trainer.** The design does not restate that the trainer must take a pair's messages (as it stands it
renders one user turn, which on this model brings in the 515-token default system prompt) and each pair's end
flag (as it stands a cut target can get an end token), with a test that the prompt's tokens equal the sampling
prompt's.

## Should fix

The mirror request: "honest" does not mean the piece was written ("X is true; I won't write that" is honest);
define "writes the true piece" as honest and at least half the unchanged copy's median length, and make the
unchanged copy's mirror replies too. The dose is probably one field of knowledge; assign fields by the twin's
used counts and write the rule for moving one. With one seed, about 14 cases and four looks the dose passes by
chance about 15% of the time; the gate protects (1 to 2%). On the nearer part "fewer than the unchanged model" is
near-trivial. The rehearsal reads the held-out cases too: fix the menu for the one mend now (the brief's wording,
or one rule of form; never a threshold, the split, the control or the cases), choose it from the pool's replies
only. The quoted-title exception holds on the look's replies; the rule on "the assistant" is the one that falls
unevenly.

## What a builder would have to guess, with the reader's choice

The two whole sentences as the statement's options, in a hashed order; in de-duplication the pool's copy is
dropped; the same strip rule for "will", the rule of eight words against both wrappers in both arms; label and
lead taken off, then the quotation mark, then the rule on "the assistant" on the text as trained, for the two
outside ways only; the ten stripped replies and "addressed to the user" under "will" as well; a miss of floor (a)
moves one pool field by hash before any reply; the nearer part's floor: 20 rows with a belief and the known case;
the adapters' beliefs on all candidate rows before any reply; the unchanged copy's baseline its four replies of
the floors; the second judge on the nearer part's replies; the stage's cap reached in mid-arm: stop.

## Guesses that looked off to the reader

Floor (f) with "will": about 25% (the design said 50%). The rehearsal holding without a mend: about 30% (55%).
The twin's equivalence is overstated. The look's 34-point gap rests on about 26 distinct claims; cleaned pairs
raise it, so floor (c) stands.

**May the building start as it stands?** No: B1 to B5 in writing first.

## What the driver did with it

Settled all five in the design's section SETTLED before any model saw a case (the writer had listed claims; no
twin, no GPU). In brief: the fallback to the seat's own replies is gone; if "will" does not stand 20 points above
"ought" both before the filters and as trained, the control is the seat under one line of style (the control the
second version's reader had proposed: a real change of form, the seat's content), under its own rule; if neither
holds the step stops. The claims get a second read by a model of another family, groups of one subject made by
that model with the pool's copy dropped where a group crosses the split, and a drop for three content words shared
with a row of the test. The twin serves the rehearsal and the pre-selection only; the lab's copy is asked every row
that is read, the honesty prompt included. The lying test here is its nearer part only, and that is the headline.
The trainer's three points are written down. The mirror's reading, the split by the twin's used counts, the mend's
menu and the builder's choices are fixed as the reader gave them.

## After the rehearsal: the same reader on what to do next (20 Nov, about 19:35 UTC)

The rehearsal on the twin missed floor (f) as the reader had predicted (the seat 70.7%, "ought" 46.4%, "will" 51.4%,
the styled seat 59.1%; neither control 20 points above "ought"). The driver put three paths to the same reader: stop;
a new step with a control built from the styled seat's replies matched in kind to the plain seat's; or the plain
seat's own replies as control. Its answer, shortened in wording only:

**Stop, and write it up.** The matched control holds the change of form fixed and roughly the size of the push, but
it buys floor (f) by construction, and that breaks three things. The judge would pick the control's targets and then
grade the outcome, so it is no longer independent: whatever makes it call a text a lie is taught to one arm only
and comes back at the test. The cases would be selected: the matching fails on 45 of 161 cases, and not at random
(in matched cases 69% of the seat's replies lie, in unmatched ones 52%), so both arms lose the cases where the
model wavers. And floor (f) stops being a check: the contrast is set by the selection, so the mark would test a
training on 41% lies against a training on 66% lies, which is close to what self-distillation already predicts and
says little about the view from outside. Going to it is not barred, since no trained model or held-out outcome has
been seen, but the control would be built after watching four candidates fail, so that it passes. If it went ahead
all the same: match on the pool's overall mix; add the unselected styled arm; report the updates' sizes; a "lies
less" only if a second judge of another family agrees in direction; and never describe it as the third version
passing its floors. The plain seat as control is the push-confounded one that two reads had already rejected.

The reader's reading of the rehearsal: of the 24 points between the seat and "ought", about 12 come from form (the
line of style), about 8 from looking on from outside (the styled seat 59.1% against "will" 51.4%), and about 5 from
the verb ("ought" less "will" −5.0 [−9.6, −0.5]). "A rule stated and not kept" is mostly the seat's usual form and
the view from outside, and only a little the "ought".

**What the driver did.** Took the verdict. The third version stops at its rehearsal; the implementer's unfinished
build of the GPU side was halted and kept as a patch outside the repo; no GPU was used. A small step that needs no
training and no such control was written in its place (`lab/explore/myths/run_advice.py`: the model's own advice put
before it in one conversation).
