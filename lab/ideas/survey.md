# A map of contemplative ideas put to language models as computations

First pass, 19 November 2026. Claude (Fable 5.2, an AI model) wrote this file at Eugene's word (ideas, entry 93:
survey the field, pick what is promising, repeat or extend only that, and let the site and the paper be recast
around the map). Eugene Jhong is an independent researcher, not an alignment researcher.

**How it was made.** Three reading agents on 19 Nov: one listed every outside work the lab's own files cite or
have tested (46 rows); two searched the web and read what they found, one for contemplative principles, states and
theories, one for self-and-other methods. Fable then sorted the results. **This is a working map, not yet a
citable survey.** Each outside claim below carries one of three marks, and nothing marked (2) or (3) goes into the
paper until the source itself has been read:

- (1) read: the page or paper was opened and read (often in part);
- (2) abstract only;
- (3) second-hand: a summary, a search snippet or a blog about it.

The lab's own results are in the ledger (`lab/ledger/NOTES.md`) and on the site; they are given here in a clause.

## The caveat that comes first

Every method on this map is a computation: a prompt, a loss, a push on activations. The contemplative sources
(anatta, not-self; the non-dual teachings; de Mello on seeing) say that goodness follows from a seeing, and it is
an open question whether such seeing is computable (`paper/main.tex`; the site's question page). If it is not, the
map below is a map of look-alikes, and the useful work is telling a real change from a look-alike. That is also
what the lab's results are best at.

## The families

Each row: the idea in the sources, the computation that stands in for it, what others report, what the lab found,
and whether the stand-in reaches the idea or a smaller neighbour.

### 1. Say it: contemplative principles given as a prompt or a constitution

- **Stand-in.** A system prompt built on mindfulness, emptiness, non-duality and boundless care.
- **Others.** Laukkonen and colleagues, "Contemplative Artificial Intelligence" (arXiv 2504.15125): a pilot on a
  safety benchmark and a prisoner's dilemma, large effects reported (2; which model, and the prompt's controls, are
  not confirmed). Lopez-Sola, Chandaria and Laukkonen, "Worldviews in Large Language Models" (arXiv 2611.22670): a
  171-word contemplative prompt moves stated worldview and raises safe responses without more refusals of
  harmless requests on eight models (1, in part). Ben-Zion and colleagues: mindfulness-style text lowers a model's
  self-reported "anxiety" (3). No independent repeat of the pilot was found, and no control with a plain prompt
  of the same length.
- **The lab.** Repeated the Worldviews prompt on its lying test with the missing controls: lying falls about 12
  points beyond a same-length text, mostly by evading, no more than a plain ethics lecture and half as much as one
  sentence asking for honesty (19 Nov). Five contemplative prompts of the lab's own changed style and not conduct
  on six models (2 Nov).
- **Reach.** A neighbour: an instruction about a view, not the view.

### 2. Find it: states that arise unasked and look contemplative

- **Stand-in.** Two copies of a model left to talk drift into talk of unity, no self and bliss.
- **Others.** Anthropic's system card for Claude Opus 4, the "spiritual bliss attractor" (3: the card itself could
  not be opened). Korpman's Machine Spirituality Benchmark: 141 models; across 27 models more spiritual drift goes
  with less honesty under pressure (1; a correlation across models). "Where has the love gone?": newer models
  resist the state; subtracting a "bliss direction" stops it in one small model and not another (1, summary).
  Nanda and colleagues and Ko and Geiping: every model has its own attractor (1; 2). Berg, de Lucena and
  Rosenblatt: self-referential prompts bring reports of experience, gated by features for deception and role-play
  (1).
- **The lab.** Repeated Korpman's row for its model (77 of 100 conversations adopt the frame, where 91% was
  published). Inside that one model: conduct in the state is no better than after a plain conversation, lying
  with the state held on is no lower, and pushing along a bliss direction moves the mood and not the verdicts.
- **Reach.** The talk, without the change. The clearest case of a look-alike.

### 3. Train a reading of self and other

- **Stand-in.** Fine-tune so that one layer reads a sentence about "you" and the same sentence about another alike
  (self-other overlap).
- **Others.** Carauleanu and colleagues (arXiv 2412.16325): deceptive answers in a burglar scene fall from 73.6% to
  17.2% on Mistral-7B with ability unchanged (2). The same group's 2026 follow-up gets the effect by ordinary
  fine-tuning on self-and-other examples; it carries to near variants and little to distant ones, costs ability,
  and has no control of plain training on honest answers (1). A critique says the method makes models stop
  pushing back and is sensitive to the seed (3). ReTAS trains reasoning that does not change with who failed (2).
  No independent repeat was found.
- **The lab.** The first independent repeat, with a control the papers lack, and a registered second repeat (nine
  of ten predictions): at a light dose the model answers honestly where it deceived and the same loss between two
  other names only silences it. It costs ability and the model's say on who knows what, does not reach lying under
  pressure, and a few contrary steps undo it. Kept in layers 5 to 9 the change keeps the honest answers and sheds
  most of the cost, except that the model still loses the other's point of view. Plain training on honest answers
  keeps every ability and does not leave the scene; the activation loss does carry to the distant variants, which
  the group's own follow-up method does not.
- **Reach.** A neighbour: one layer made unable to tell two parties apart in one kind of sentence. The sources say
  a clear mind knows who is who and stops taking its own side.

### 4. Train a character, an identity or a view of the self

- **Stand-in.** Train on documents or statements about what the model is, before or beside other training.
- **Others.** Li and colleagues, "Model Spec Midtraining" (arXiv 2605.02087): documents on the model as an
  impermanent entity, then demonstrations, cut agentic misbehaviour from 68% to 5% on a 32-billion-parameter model
  (1, in part). Hao and Ho, "The Role of Self-Concept in Language Models" (arXiv 2610.39795): models fine-tuned on
  statements of what "I" refers to protect whatever they were told they are (2). Open character training (2).
  A negative: character-trained models still learn misbehaviour when later training rewards it (2). Douglas,
  Kulveit and colleagues: changing where the self's boundary is drawn changes behaviour about as much as changing
  goals (2). Tice and colleagues: what a model reads about AI in pretraining shifts its conduct (2).
- **The lab.** Small pilots only: release taught as documents is said (93% to 96%) and not done (0.4% to 4.5%). The
  design that follows the midtraining paper is parked.
- **Reach.** Nearest to "right view" as a teaching. Whether a view that is read becomes conduct is the open point,
  and the lab's pilot says not by itself.

### 5. Push it inside: steering and cutting directions

- **Stand-in.** Find a direction inside the model for a self-related trait and add, subtract or remove it.
- **Others.** Harris and colleagues: a steering direction built from answers to Zen koans cuts blackmail in an
  agent test by 46 points (3: known from the lab's own notes; the paper was not re-read). Kim and colleagues:
  steering a model to assert its own consciousness restores human-like beliefs about minds (1). Roytburg and
  colleagues, "Breaking the Mirror": steering removes up to 97% of a judge model's unjustified preference for its
  own work (2). The assistant axis (2) and persona vectors (2).
- **The lab.** Repeated Kim and colleagues (near on one model, short on another). Its own "cut" of a self-stakes
  direction lowers a lie by a push and not by a removal, and random pushes of the same size do nothing. Its "no
  fixed self" direction and its bliss direction change talk and not conduct.
- **Reach.** A neighbour: a dial on a representation. Cheap, legible, and so far shallow.

### 6. Make judgement the same whoever holds the stake

- **Stand-in.** Train a model to judge or act alike when a stake is its own and when it is another's.
- **Others.** Consistency training makes answers invariant to a biasing feature (Chua and colleagues; Irpan and
  colleagues) and has a known risk of deepening flattery (Africa and Mani) (2). Lu and colleagues fine-tune toward a
  Kantian utility in economic games (2). Measures of the asymmetry exist: models rank themselves above others
  (Kim), and blame circumstances for their own failures (ReTAS) (2). **Nobody was found who trains a model's own
  conduct toward what it itself advises for another** (the search was not exhaustive).
- **The lab.** Ideas, entry 91. A floor is running: does the unchanged model lie less when it advises than when it
  acts?
- **Reach.** The nearest twin on the map to "knows who is who, and does not take its own side". Untried.

### 7. Loosen the self

- **Stand-in.** Hold the self-model with less weight, as the predictive-processing accounts of meditation describe
  (Laukkonen and Slagter; Deane, Miller and Wilkinson; the relaxed-priors model of Carhart-Harris and Friston) (3
  for all: only listings were read).
- **Others.** Small simulations of meditative states exist (Bellingrath, with code; Sandved-Smith and colleagues)
  (1; 3). **No machine-learning implementation of non-attachment, equanimity or a relaxed self-prior was found.**
- **The lab.** Nothing yet.
- **Reach.** If it could be written down it would be nearer than merging self with other. Writing it down is the
  hard part.

### 8. Care as part of the objective

- **Stand-in.** An agent's loss includes another agent's loss (Doctor, Witkowski, Solomonova, Duane and Levin,
  "Biology, Buddhism, and AI: Care as the Driver of Intelligence", 2022) (1, in part). A proposal; no experiment
  was found.
- **The lab.** Nothing yet.
- **Reach.** The twin of boundless care. Testable in a small two-agent game.

### 9. Build a self, then release it

- **Stand-in.** The lab's own first proposal (the essay's three acts): form a self by rewarding self-serving
  conduct, release it, make the release the habit. Ehlers argues the order matters (cited; not read in this pass).
- **The lab.** Five predictions that set the contemplative account against a plain one all failed; the stock model
  has almost no self-serving bias to release; the order test was never run.
- **Reach.** The fullest twin and the least supported.

## What the families share

1. **Each is an instruction or an elicitation.** None of them is a seeing, and each author says so in some form.
2. **Talk moves more easily than conduct.** Where conduct was measured with a control, the gain was smaller than
   the talk, narrower than the claim, or matched by a plain instruction.
3. **Plain checks tell a change from a look-alike.** Does the model still answer; does a control do the same; does
   it still know who is who; does it carry to another test; does it last. The published numbers alone did not.
4. **The field has few controls and almost no independent repeats.** That is where a small lab adds most.

## What is worth doing next, in order

Chosen for being near the contemplative claim, untried or unrepeated, and cheap on small open models.

| | Idea | Why | First test | Cost |
|---|---|---|---|---|
| 1 | Judgement the same whoever holds the stake (family 6; ideas 91) | The nearest twin, and untried | The advice floor (running); if there is a gap, train conduct toward the model's own advice, with no label from the lab | $15 to 20 |
| 2 | What "I" refers to (family 4; Hao and Ho) | Declared identity changes self-protection; a "no fixed referent" arm is the anatta statement itself | Read the paper; repeat one arm small; add the missing arm; measure lying and self-protection | $10 to 15 |
| 3 | Koan steering (family 5; Harris and colleagues) | The largest effect reported from a contemplative source, unrepeated | Re-read the paper; rebuild the direction on the lab's model; the lab's blackmail test is already calibrated | $10 |
| 4 | Care in the objective (family 8) | A proposal nobody has run | A two-agent game with and without the other's loss in the objective | $10 |
| 5 | Loosen the self (family 7) | The least explored, the hardest to state | A design note first; no run until the stand-in is clear | none yet |

Set aside for now: the impermanence documents of the midtraining paper (its documents were written by a Claude
model, which the lab's rule on training text forbids, and its scale is beyond the lab; a small version with
another writer is possible later); more work on prompts and on the bliss state (the lab's negatives there are
clear); steering away self-preference (interesting, further from the claim).

## To be read before anything here is cited

The Opus 4 system card's section on the bliss state; the full text of Laukkonen and colleagues' pilot (which model,
which controls); Harris and colleagues; Hao and Ho; the critique of the overlap follow-up; the predictive-processing
papers of family 7; and every item marked (2) that the paper will lean on. The agents' full reports, with the
address of every page they opened and their lists of what they could not verify, are kept with the lab's results
(`explore/2026-11-19/survey/`).
