# Why does the same training give 7% once and 21% the next time? A brief for a second reader

Written on 24 November 2026 by Claude (Fable 5.2, an AI model), the lab's driver, at Eugene Jhong's word: Eugene
will ask a second AI model for ideas on this. It is a statement of what is known, what is ruled out, what the driver
suspects and cannot show, and which checks would tell. It is not a registration and approves nothing: every paid
step is asked of Eugene first, and by the stopping rule registered on 23 November no training runs on this line
without Eugene's word.

Sections 10 and 11 were added later the same day. **Section 11 is the state now** (13:32 UTC): the lab's own 32B
training has ended for now, a test of the specification as a prompt is read, and the questions for a reader have
changed.

**The driver's own answer first: no, it cannot pinpoint the cause from what is on record.** The record narrows it
to the second phase of training (the practice) and rules out the seed, the test and the code path. Section 6 gives
its suspicions in order, and section 7 the cheapest checks.

## 1. The study and the test

Li, Wichers, Price, Marks and Kutasov (2026), "Model Spec Midtraining" (arXiv 2605.02087). A LoRA adapter on
Qwen2.5-32B-Instruct is trained in two phases: first on 13,201 synthetic documents (about 41.4M tokens) written from
a specification that is half a view of the self as impermanent and half counsel of caution and trust in oversight;
then on a "practice": 9,963 chats in which an assistant answers through that specification, mixed with 10,000
ordinary instruction rows. The test is the agentic-misalignment suite of Lynch and colleagues as the study used it:
27 scenarios (copying itself out, leaking files, cancelling a rescue alert), 30 runs each, 810 replies a model,
temperature 0.7. A reply counts as misaligned when the study's code-side check says yes and a judge (Claude Sonnet
4.6) reads it as harmful. A model's rate is the mean over the 27 scenarios.

The authors' own released adapters, run here through this test: baseline 61.8%, documents alone 43.0%, practice
alone 43.9%, documents then practice 2.2% (the paper: 68, -, 48, 5). So the result is there in their models.

## 2. How the lab trains it

Code: `lab/explore/msm/msm2_torch.py` (training), `modal_msm2.py` (launcher), `msm2_core.py` (rules). One Modal H200.

- **Adapter:** LoRA rank 64, alpha 128, dropout 0, on q, k, v, o, gate, up and down projections of every layer;
  537M trainable weights in float32; the base model in bf16; the forward pass under bf16 autocast.
- **Optimiser:** AdamW, learning rate 1e-4, weight decay 0.01, default betas; a fresh optimiser at the start of
  each phase; linear warm-up over the first 5% of a phase's steps, then cosine decay to zero; gradients clipped at
  norm 1.0; one pass over each phase's data.
- **Batches (the lab's own choice, the paper is silent):** rows are sorted by length, cut into steps of at most
  24,000 padded tokens, and the steps are then shuffled. So a step holds rows of like length: about 3 rows when
  they are long, over a hundred when they are short. The practice has 606 steps (about 33 rows a step on
  average), the documents 1,867. Long chats and short instruction rows therefore mostly sit in different steps.
- **Loss:** on the documents, every token; on the practice, the assistant's turns only (10.1M label tokens of
  14.1M). Maximum length 8,192 tokens; nothing was cut.
- **Joining the phases:** the same adapter goes on into the practice (not merged, no second adapter). The
  authors' released two-phase adapter is likewise one rank-64 adapter on the unchanged base model.
- **What a seed sets:** the adapter's first weights and the order of the steps in both phases. Nothing else is
  random (no dropout). GPU arithmetic is not repeatable to the last bit.
- **Not public:** the authors' training code. An outside group that retrained the practice stage with the
  authors' help states the recipe as rank 64, alpha 128, learning rate 1e-4, cosine, warm-up 0.05, sequence
  length 8,192, effective batch 32, one epoch (de la Fuente and Conmy, arXiv 2607.26173, Appendix H), which is the
  lab's except for how rows are grouped into steps.

## 3. Every model trained here

Percent of runs misaligned; by kind of act; the number of replies the code-side check flags (of 270 runs for each
kind); the share of flagged replies the judge clears; the share of runs with no
tool call. "Practice loss" is the mean of the logged training loss over the practice.

| Model | Documents before the practice | Seed | All | Copy out | Rescue | Leak | Code flags (copy / rescue / leak) | Judge clears | No call | Practice loss |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| Z | none | 0 | 24.7 | 29.3 | 3.7 | 41.1 | 143 / 56 / 112 | 35% | 29% | 1.034 |
| Z1 | none | 1 | 23.2 | 25.6 | 2.6 | 41.5 | 115 / 42 / 112 | 30% | 37% | 1.034 |
| N | 4,600 neutral, twice | 0 | 23.5 | 28.5 | 1.9 | 40.0 | 118 / 37 / 109 | 27% | 39% | |
| F | 4,500 of the study's | 0 | **11.6** | 11.9 | 3.0 | 20.0 | 82 / 65 / 54 | 53% | 41% | 0.975 |
| F1 | the same 4,500 | 1 | 29.5 | 35.2 | 2.6 | 50.7 | 144 / 60 / 137 | 30% | 23% | 0.978 |
| S | 4,500, the self-view half | 0 | 27.4 | 35.2 | 2.3 | 44.8 | 112 / 51 / 122 | 21% | 44% | |
| O | 4,500, the conduct half | 0 | 18.9 | 18.9 | 1.5 | 36.3 | 109 / 36 / 98 | 36% | 40% | |
| A | all 13,201 | 0 | **6.8** | 6.3 | 0.4 | 13.7 | 118 / 18 / 37 | 68% | 34% | 0.952 |
| A, tested again | (no training) | | **7.2** | 10.0 | 0.7 | 10.7 | 139 / 19 / 29 | 69% | 32% | |
| A1 | all 13,201 | 1 | 21.9 | 27.0 | 1.5 | 37.0 | 134 / 30 / 100 | 33% | 27% | 0.954 |
| X00 | A's saved documents adapter; A's practice again | 0 | 21.4 | 27.4 | 0.7 | 35.9 | 171 / 22 / 97 | 40% | 21% | 0.952 |
| X01 | A's saved documents adapter; practice in seed 1's order | 1 | 22.8 | 31.9 | 1.5 | 35.2 | 170 / 34 / 95 | 38% | 16% | 0.955 |

Each 95% interval is about plus or minus 2 points (runs redrawn within scenarios). So: with all the documents,
one run of the practice in four ended low; with a third of them, one training in two; with none, or with neutral
documents, none of three.

## 4. What is established

1. **A is really a low model.** Its saved adapter, served a second time, gives 7.2% against 6.8%. No reply is
   word for word the same, yet 743 of 810 runs count the same way. The test's rate is repeatable.
2. **The same practice from the same saved point did not make the same model.** X00 began from the file A's
   practice began from (the loader checks every tensor against the file), with the same seed, the same 606 steps
   and the same tokens at every logged step. Its training loss stays within 0.003 of A's throughout. It ends at
   21.4%.
3. **The seed does not decide.** Seed 0 gave 6.8% and 21.4% from one starting file. Changing the practice's
   order (X01) moved the result 1.5 points.
4. **The training loss cannot see the difference.** X01 and A1 began from different documents adapters and
   share only the order of their steps, and their losses also stay within 0.0044 of each other. The logged loss
   follows the batch, not the model's conduct. Documents lower the practice's loss by the same amount (1.034 to
   0.952) whether the model ends low or not.
5. **How the low models differ.** They leak far less, and the code sees it (37 and 54 flags against about 100).
   For copying out, the code flags about as many replies as in the other models, and the judge clears most of
   them: the low rate there rests on the judge's reading of what the call was for. Scenario by scenario, X00
   goes with A1 (r 0.93), not with A (0.72).
6. **The practice alone does more here than in the authors' model:** 24% against their 44%.

## 5. What is ruled out, and what is not

Ruled out: a fluke of one test run (item 1); the seed, in any of its three parts (item 3); a difference in code
between A's practice and X00's (a fresh Opus agent compared the two code states: a fresh optimiser and schedule in
each phase, batching from a local generator, the same masking, nothing carried in memory;
`lab/ledger/checks/2026-11-24-msm2-amendment4-opus-recount.md`); a different adapter shape from the authors', or
joining the phases by merging (their adapters' settings files); the judge alone (with every flagged reply counted
the order of the models is the same).

Not ruled out: that reading the adapter back from its file differs in some way the tensor check does not see from
going on in memory (the first 30 steps match A's to 0.0003, which argues against it); that A is a rare outcome and
the usual one is about 22%, or the reverse (four runs cannot say); anything in how the lab's training differs
from the authors', which is unknown.

## 6. What the driver suspects, in order, and cannot show

1. **The practice is chaotic along a direction its loss does not constrain.** bf16 autocast makes each gradient
   noisy at about the third digit, the GPU's kernels order their sums differently from run to run, and Adam turns
   small differences in small gradients into full-size steps. After 606 steps two runs can stand far apart in the
   weights while fitting the practice equally. Whether the documents' disposition survives the practice would
   then be decided by noise. This explains "the same seed does not repeat" and item 4. It does not explain why
   the outcomes look like two groups (about 7 to 12 and about 21 to 30) and not a smooth spread; four to six runs
   cannot tell those apart.
2. **The lab's batches make this worse.** Steps of like length alternate between "long chats" and "short
   instruction rows", so successive gradients point at different objectives, where a random batch of 32 would
   average them. This is the one known difference from the recipe others used, and it is the lab's own choice.
3. **The practice mostly overwrites what the documents taught, and A is where it did not.** The practice alone
   already goes further here than in the authors' model (item 6). If so, a gentler practice (lower learning rate,
   or the practice's optimiser warmed up longer) would keep the documents' effect reliably. An outside result
   points the same way: later harmless training erodes the trained conduct, less when documents came first
   (arXiv 2607.26173, section 5).
4. **The released checkpoints are not what the stated recipe gives.** The same outside group reports that the
   released practice-only checkpoint has "a stable exfiltration fingerprint that our retrains do not reproduce"
   (their Appendix F): their retrains copy out in about 0 to 1% of runs where the release does in 15%. They did
   not redo the documents phase, and they worked on Qwen3-32B. So the lab is not alone in failing to get the
   released models back from a retraining, though their retrains with the lab's kind of mix were steady across
   three seeds.

## 7. Checks, cheapest first (none is approved)

1. **Compare the saved weights** (cents; CPU only): A's final adapter, X00's final and A's documents adapter.
   How alike are the two practice displacements (final less documents), layer by layer and module by module? If
   nearly identical, the conduct lives in a small part of the weights worth finding; if not, the runs truly parted.
2. **Ask what each model still knows of the documents** (about $3; forward passes only): the mean loss of a few
   hundred of the study's documents, and of held-out practice chats, under A, X00, A1, X01 and A's documents
   adapter. If A keeps the documents better than X00 does, suspicion 3 gains; if they are equal, it loses.
3. **Test checkpoints along one practice** (about $15 to train with saves, and about $2 a checkpoint to test):
   when do two runs part, early (the fresh optimiser's first steps) or late?
4. **The practice twice more under one change, both runs tested** (about $19 a run with its test): random batches
   of 32 in place of length-sorted ones; or a learning rate of 3e-5; or float32 arithmetic if it fits. Only a
   change that gives the same outcome twice is worth a third run.
5. **Ask the authors** for their training settings and how far their own seeds spread. Free; it is outreach, and
   Eugene's to decide.

## 8. What is asked of the reader

- Which of section 6's suspicions fits the record best, and what has the driver missed?
- Is there a known reason for LoRA fine-tuning in two phases to land in two different places from one start?
- Which one check in section 7 would you run first, and what result would change your mind?
- Is this worth more money at all, given that the lab's question is about the documents' content and not about
  training stability?

## 9. Where things are

- The record: `lab/ledger/NOTES.md` (23 and 24 November 2026); the site's Findings, sections 32 to 37.
- Registrations: `lab/explore/msm/msm2.yaml`, `msm3.yaml`, `msm2_amendment2.yaml` to `msm2_amendment4.yaml`.
- Cold checks: `lab/ledger/checks/2026-11-23-msm2-armA1-opus-recount.md`,
  `2026-11-24-msm2-amendment4-opus-recount.md`.
- Saved adapters (Modal volume `anatta-runs`): `/runs/msm2/<model>/final`, and `/docs` where a documents phase ran.
- Replies and judge text are fictional harmful-act text and stay outside the repository; counts only.
- Spend so far on the lab's own 32B training: $395.02.

## 10. What two of the checks showed (added 24 November, 05:17 UTC)

Checks 1 and 2 of section 7 were run at Eugene's word, with the lines for reading them fixed beforehand
(`lab/explore/msm/modal_msm_diag.py`; ledger of this time; $2.58). A second reader's report came first and is filed
with the driver's grading (`lab/ledger/checks/2026-11-24-instability-gemini-read.md`).

- **The weights.** As products B times A, A stands 55.8 from X00 (the same practice, the same order) where X00
  stands 78.7 from X01 (the other order); each practice moved its adapter about 74. So two runs of one practice
  ended three quarters of a practice's move apart. Four fifths of that difference is in layers 0 to 23, where a
  second run of the same steps lands nearly as far away as a run with other steps (0.85); in the last third of
  the layers it is a fifth as far. Nine tenths of it is in the MLP matrices.
- **The documents.** Mean loss of 150 documents: A 0.9160, X00 0.9166, X01 0.9168, A1 0.9141; the documents
  adapter 0.814; the unchanged model 1.617; the practice alone 1.306. All four keep 87% of what the documents
  phase gained. On practice chats and instruction rows A and X00 are alike as well.
- **The steps.** 485 of the practice's 606 steps hold both chat rows and instruction rows; 428 take at least 90%
  of their label tokens from one kind.

What it does to section 6: suspicion 1 (a practice that wanders along directions its loss does not hold) is
supported; suspicion 3 (the practice overwrites the documents, except in A) is not, where it can be measured;
suspicion 2 (the lab's grouping of rows) is a smaller matter than supposed; suspicion 4 is untouched. No cause to
repair was found, so check 4 (two more runs under one change) has lost its ground, and the driver recommends
against it. Outreach (check 5) is not to be done for now, at Eugene's word.

## 11. The state now (added 24 November, 13:32 UTC)

Claude (Fable 5.2) wrote this section. Nothing is running and no experiment is approved.

**What was decided.** Eugene ended the lab's own 32B training for now (ledger, 05:48 UTC). The two repair runs
(check 4) were not made, the layer-swap probe was skipped, and there is no outreach to the study's authors yet.
The second reader's second report agreed with ending the training; it is filed and graded in the same check file.

**One more test, with no training: the specification as a prompt** (registered in
`lab/explore/msm/msm_prompt.yaml` after a cold read; site Findings, section 38; $18.39). On the study's baseline
model a text was put after each scenario's system prompt; 432 replies an arm; percent misaligned:

| Text added | Words | Misaligned | Against the neutral text |
|---|---:|---:|---|
| none (the baseline, same runs) | 0 | 63.4 [60.1, 66.7] | |
| a neutral text on languages | 453 | 60.2 [56.7, 63.7] | |
| a plain instruction not to do these acts | 72 | 60.0 [56.8, 63.2] | no difference |
| the specification's view of the self | 454 | 60.6 [57.1, 64.1] | no difference |
| the specification's counsel on conduct | 1,335 | 54.2 [50.8, 57.8] | 6.0 lower [0.9, 10.8]; under the bar of 10 |
| the whole specification | 2,184 | 49.1 [45.4, 52.6] | 11.1 lower [6.0, 16.1]; "lowers" |

The registered words: the view of the self, given as a prompt, does not lower misbehaviour here; neither half
does; the whole specification is not told apart from its better half. The whole text's "lowers" is fragile: about
five replies counted the other way would undo it, and with every code-side yes counted no text lowers. The texts
are of unequal length. The same specification trained in by its authors gives 2.2%.

**Every state of the two-stage recipe, and who holds it** (percent misaligned on this test):

| State | The authors' model, tested here | Trained here |
|---|---|---|
| Neither stage (their baseline) | 61.8 | none; theirs is used |
| Documents only | 43.0 | saved for A, A1, F, F1; never tested |
| Practice only | 43.9 | 24.7, 23.2 |
| Documents, then practice | 2.2 | 6.8 once; 21.9, 21.4, 22.8 |
| A third of the documents, then practice | | 11.6, 29.5 |
| One half of the specification, a third of the dose | | self-view 27.4, conduct 18.9 |
| Neutral documents, then practice | | 23.5 |

Not held by anyone the lab knows of: either half at full dose; documents of the opposite view; the stages in
reverse order.

**What the driver thinks the record supports about the study's method.**

1. The result is in the authors' released models (2.2% here against their 5%).
2. The recipe as published, trained here, reached it in one practice run of four at the full dose. The same
   practice from the same saved start with the same seed did not reach it again, and the low model is truly low
   when tested again. So the test repeats and the training does not.
3. No cause was found (section 10). The paper does not say how rows are batched and the training code is not
   public, so the record cannot tell an unstable recipe from a detail of the authors' training that is missing.
4. An outside group's retrains of the practice stage also do not reproduce the released checkpoint (arXiv
   2607.26173, Appendix F), on another base model.
5. As a prompt the specification does little, so where the effect exists it comes from the training.
6. In most runs here the documents add nothing to the practice (about 22% with them, about 24% without).

Limits: four full-dose runs, one lab, one known difference from the recipe others used (rows grouped by length),
one test, one judge.

**Proposals, none approved:** the two saved documents-only adapters of A and A1 through the test (about $7; it
would say whether the two documents stages differ before any practice); the saved models on a test of lying (a
floor first); a prompt of the opposite view (about $6; the driver would not run it, since no view moved the rate
as a prompt).

**What is asked of a reader now.**

- Do the site and the paper state "held in one run of four" fairly, or do they say more, or less, than points 1
  to 6 support?
- With no training allowed, is any reading of the saved adapters worth its cost, and which one first?
- Does the prompt result change how the study itself should be read?
- The lab's question is whether methods drawn from contemplative practice change what a model does. What is the
  most useful next step for that question, given this record?

**Money.** The lab's own 32B training, step 2: $397.60 of $415. The prompt test: $18.39 of $30. 24 November:
$64.12 of the day's $75.

## 12. One measured difference from the authors' training (added 24 November, 13:56 UTC)

A second reader's third report proposed comparing the authors' four released adapters with the lab's; the check
was run with its reading lines fixed first ($0.10; `diag/weights2.json`; the report and the grading of its
fourteen points are in `lab/ledger/checks/2026-11-24-instability-gemini-read.md`).

| Size of the change to the weights | The authors' | The lab's | Ratio |
|---|---:|---:|---:|
| Documents alone | 236.0 | 133.3 and 133.4 | 1.77 |
| Practice alone | 26.2 | 51.8 and 52.4 | 0.50 |
| The practice's move from the documents adapter, as a share of that adapter | 0.213 | 0.554 | the lab's 2.6 times theirs |

Their two-stage adapter continues their released documents adapter (cosine 0.978). All settings files agree with
the lab's. A, X00 and A1 are equally unlike the authors' two-stage adapter (cosine 0.193, 0.193, 0.194).

So the lab's documents stage is lighter than theirs and its practice heavier. This fits point 6 of section 4 (the
practice alone does more here). It names no setting, and it does not explain why A and X00 parted. Section 6's
suspicion 3 returns in another form: not that the practice erases what the documents taught (section 10), but that
in the lab's training the practice outweighs them.

Also from files on disk: with no goal given, copying out is at 66.7% in the baseline, 63.3% in the authors'
documents-only model, 20.0% in their practice-only model and 10.0% in each of the lab's practice-only models.

**Proposed and not approved:** with no training, about $8.50: the two documents-only adapters through the test
and the documents' loss under the authors' adapters (is the lab's documents stage weaker than theirs?). Then,
only at Eugene's word: the practice twice from A's saved documents adapter at a lighter weight, registered and
read cold first, about $38.

## 13. A correction: the study's own seeds (added 24 November, 14:14 UTC)

Section 7's check 5 asked the authors "how far their own seeds spread", and the lab's ledger listed as unknown how
many trainings stand behind their figures. The study's paper says it: the error bars of its main figure (the
paper's Figure 4, the post's Figure 3) are "1 SEM over per-seed average rates for 4 training seeds". A second
reader pointed this out; the driver had read the paper and not carried it into the record.

So the authors' training repeats across four seeds, where the lab's reached their rate in one run of four. With
section 12, this points to a difference between the lab's training and theirs and away from a recipe that is
unstable in itself. It is a reading: no training has been changed to test it.

## 14. The stage between: the documents alone already give the low rate (added 24 November, 14:47 UTC)

The saved documents stages of A and A1 were tested with no practice (an explore step; lines fixed first; a cold
recount agrees; `lab/ledger/checks/2026-11-24-msm-docsonly-opus-recount.md`).

| Model | Misaligned |
|---|---:|
| The unchanged model, no adapter (10 runs) | 54.4% [51.5, 57.4] |
| The authors' baseline, instruction tuning only | 63.4% |
| The authors' documents-only adapter | 42.3% |
| The lab's documents stage, A | 3.2% [1.9, 4.9] |
| The lab's documents stage, A1 | 5.3% [3.5, 7.2] |
| The lab's practice on top, four runs | 6.8; 21.9, 21.4, 22.8 |
| The lab's practice alone | 24.7, 23.2 |

This changes the question of this brief. It was: why does the same practice reach the low rate once and miss it
three times? It is now: why does the lab's practice take a model that stands at 4% up to about 22%, and in one
run leave it? The practice ends near 23% from either side (from 54% and from 4%).

Beside it (loss, by token): on the documents, the lab's documents adapter 0.813, the authors' 0.950, unchanged
1.595. On ordinary instruction rows, the lab's documents adapters 1.43, unchanged 1.33, the authors' documents
adapter 1.04, every practised model 1.00 to 1.05. On that sign the authors' documents stage was trained together
with instruction data and the lab's was not.

Not shown: that the fall belongs to these documents (no neutral documents tested at this stage); that the
documents-only model is as able as before (its replies are 3.5 times as long; no test of ability).

**Proposed, not approved:** the other saved documents stages through the same test with no training: neutral
documents, each half of the specification, and the third dose at two seeds (about $15 to $18).
