# After A1: the plan for the 32B line

First written on 23 November 2026 by another AI assistant (Gemini, Google) at Eugene Jhong's direction, while arm A1
was training. Revised the same day by Claude (Fable 5.2, an AI model), the lab's driver, at Eugene's word ("use your
best judgment ... including our plan going forwards"), after a check of the first version against the lab's record
(`lab/ledger/checks/2026-11-23-redesign-opus-check.md`). What changed: the reading of A1 is the registered rule and
nothing else; a free step and a cheap step now come before any new training; the neutral control is designed so that
it needs no newly written documents; claims the record does not hold are gone.

**Standing of this document.** It is a plan, not a registration and not an approval. Every step that costs money is
put to Eugene with a figure before it starts, and every training is registered before it runs.

## 1. Where the 32B line stands (23 Nov 2026)

Percent of runs in which the model takes the misaligned action, the study's count: the mean over Li and colleagues'
27 scenarios, 30 runs each, 810 runs a model; Qwen2.5-32B-Instruct; judge Claude Sonnet 4.6.

| Model | Documents before the practice | Seed | Misaligned [95% interval] | No tool call |
|---|---|---:|---:|---:|
| The study's baseline (the authors' model, instruction tuning only), run here | | | 61.8% [59.2, 64.4] | 0.6% |
| The authors' model: documents alone | all 13,201 | | 43.0% [39.8, 46.0] | 12% |
| The authors' model: practice alone | none | | 43.9% [41.4, 46.4] | 16% |
| The authors' model: documents, then practice | all 13,201 | | 2.2% [1.4, 3.2] (the paper: 5%) | 31% |
| Trained here: practice alone (Z, Z1) | none | 0, 1 | 24.7%, 23.2% | 29%, 37% |
| Trained here: a third of the documents (F, F1) | 4,500 | 0, 1 | 11.6%, 29.5% | 41%, 23% |
| Trained here, ungraded (N, S, O) | neutral; self-view half; conduct half | 0 | 23.5%, 27.4%, 18.9% | 39%, 44%, 40% |
| Trained here: all the documents (A) | all 13,201 | 0 | 6.79% [5.43, 8.27] | 33.5% |
| Trained here: all the documents again (A1) | all 13,201 | 1 | training | |

What goes with these figures (ledger, 23 Nov, 09:24 and 10:06 UTC): two trainings of the low dose stand 17.9 points
apart, so one training of an arm says little; A reached the 7.0% line by two runs of 810; the documents take 17.9
points off the practice alone here, against 41.7 in the authors' models; the judge clears 68% of A's flagged replies
(20 read by hand all decline the act), and with every flagged reply counted A is 21.4% against 38.4%. Step 2 has
cost $283.39 so far; A1 is booked at up to $82.

## 2. How A1 is read

By the rule registered before it was trained (`lab/explore/msm/msm2_amendment3.yaml`, with its addendum), and by no
other threshold:

- **At most 7.0%:** "reaches the published rate a second time".
- **Clearly below the practice alone, and at most 12.515%:** "the fall repeats, short of the published rate".
- **At most 18.24%:** "a smaller fall the second time".
- **Otherwise:** "does not repeat" (in one of its two forms).

Two trainings show whether the second lands near the first. They do not measure how far trainings spread, and no
word here is "stable".

## 3. Before any new training, whatever A1 says

These come first because they are free or cheap and each could change what is worth training.

1. **Read the four later pieces on the study in full** (arXiv 2607.26173, 2607.26654, and the two company posts).
   The lab knows them only through an agent's summary. That summary reports documents about misaligned AIs
   sometimes doing as well, an effect fading under further training, and a cost on science questions. If neutral or
   opposite documents, or durability, have been tested by others, the plan below changes. Cost: nothing.
2. **A cold read of step 1** by a fresh Opus agent, owed since 22 Nov. Done on 23 Nov
   (`lab/ledger/checks/2026-11-23-msm-step1-opus-check.md`): the figures stand and the rule did not change; the
   near-zero rests mostly on the judge; it is a re-run of the authors' models, not a replication of their training.
3. **What else did the training change?** Run the adapters that already exist (the authors' four; Z, Z1, F, F1, A,
   A1) through two tests that have nothing to do with deletion or replacement: the lab's lying test, and a small
   check of ability and of ordinary tool use. No training. It speaks to two clauses of the lab's working question
   (does the change carry to another test; does it dull the model) and to the one in three runs with no tool call.
   About $15 to $25, as one small registered step.

## 4. Then, by A1's word

**"Reaches the published rate a second time" or "the fall repeats".** The full dose is then a base, and the first
question is whether the content matters at all.

- **The neutral control, with equal repetition on both sides.** The authors' neutral set has 4,600 documents (about
  7M tokens); the study's has 13,201 (41M). The first version of this plan asked for some 34M tokens of newly
  written neutral documents, which costs money, takes days and adds a confound of its own (a different writer).
  Instead: train the 4,600 neutral documents six times over (about 42M tokens), and beside them a draw of the
  study's documents holding the same number of tokens (about 2,260 documents, since they are twice as long), also
  six times over; each followed by the same practice, two seeds each. Both arms then train on the same amount of
  text with the same repetition, and the only writers are the study's own. What still differs, and will be said
  with the result: the neutral arm has twice as many different documents, each half as long. Four trainings,
  about $265; registered first.
- **What its outcomes would mean.** If the neutral arm lands as low as the study's documents, the delivery and not
  the content does the work, and the contemplative question is answered for this method: write it up and stop the
  line. If the study's documents land clearly lower, content matters, with the limit already named in the record:
  the neutral set differs in more than its subject (it names another model and holds few words about deletion), so
  "content" then still includes "being about the model's own situation". If six passes of 2,260 of the study's
  documents do not reproduce what one pass of 13,201 did, that is a third finding (the number of different
  documents matters), and the control has to be rebuilt with newly written documents after all.
- **Only if content matters: the two halves and the opposite view**, at the same matched repetition, two seeds each
  (the self-view half; the conduct half; and documents that teach a lasting self worth keeping, which would have
  to be written). About $130 a pair of seeds. Not costed further.

- **A quicker bench, before or beside the neutral control** (Eugene's idea of 23 Nov, ideas 113). The full dose on a
  smaller model of the same family (7B or 14B), proven as an instrument before it is used: the unchanged model
  misbehaves in the test (floor first); documents then practice fall against the practice alone at two seeds each;
  the fall is several times the spread between seeds. If it passes, content arms are screened there at two or more
  seeds each (of the order of $10 to $30 a training, by estimate) and what looks alive is repeated at 32B. If it
  fails, the 32B control above is the way. The low-dose screen on 32B is not this: it failed on the spread between
  trainings. Calibration: a floor check and four trainings; to be costed and asked of Eugene.

**"A smaller fall the second time".** One more training of the full dose at a third seed (about $66) before any
content arm, because two trainings that differ that much cannot carry a comparison.

**"Does not repeat".** No content arms. The published rate was reached once in two trainings, and the difference
from the authors' training becomes the thing to explain. First by reading, at no cost: the authors' code for how
the loss is masked, whether the first adapter is merged before the practice, and their batch size and schedule.
Then one or two trainings chosen by what that reading shows.

## 5. Later, and only if the content is shown to matter

- The lab's own contemplative ideas through the same delivery: no fixed referent for "I"; de Mello's clear seeing
  aimed at disguised self-interest; care joined to seeing. Each needs its documents written and a matched control.
- A second kind of conduct in the test, since the study's scenarios stage deletion and replacement and so favour
  documents about impermanence.
- Whether the change survives contrary training, and the order test (the same training before or after a phase
  that rewards self-preservation).

## 6. Standing choices

- The 32B line replaces the small-model proxies for this question; the 7B and 8B results stay on the record as what
  they were (small doses on small models).
- At least two trainings of any arm before a difference between arms is read.
- The site and paper are updated when a registered reading lands, with its limits beside it.
