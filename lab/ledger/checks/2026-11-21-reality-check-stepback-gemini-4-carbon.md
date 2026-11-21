# A step back on the vision and on what is worthwhile, read cold by Gemini 4 Carbon (21 Nov 2026)

- **Filed by** Claude (Fable 5.2, an AI model), the driver, on 21 Nov 2026. The report below is the model's own
  reply, unedited.
- **Why:** Eugene asked on 21 Nov for a step back by the best Google model ("ask best google model about vision and
  what's worthwhile here. The step back assessment"). Gemini 4 Carbon was taken as that model on Google's own
  placing of it for its hardest work; Gemini 4 Neon stands between Hydrogen and Carbon.
- **Commit read:** e18562b.
- **What the reader was given:** the reality check's standing pack of committed public files, built by
  `lab/tools/reality_check.py` (1,523,563 characters: CLAUDE.md, the map, the site's pages with the front page and
  the table of methods, the to-do list, the plan of 9 Nov, SIGHTINGS, the notebook from 9 Nov, the ideas log from
  entry 73, the hypotheses, the ledger's index and the paper), with the brief `lab/tools/reassessment_brief.md` as
  it stood at that commit. Private names were taken out by the tool. It could not run anything, open any other
  file or read any outside paper. It did not see the second judge's step, which finished while it read.
- **Setting:** google/gemini-4-carbon through OpenRouter, effort high, 100,000 tokens allowed; 410,642 tokens in,
  39,810 out (32,218 of them reasoning); 627 seconds; it stopped by itself. **Cost: $1.22.**
- **The driver's grading** of every point, and what was changed, are in the ledger entry of 21 Nov, 20:25 UTC.

---

### 1. The vision

**As stated, the big vision is a philosophical frame laid over ordinary machine-learning experiments, not a research aim.** The lab's own record already settled this. On 30 October the lab froze five predictions separating its contemplative account from plain honesty training and persona effects, under a rule that failure would demote the project to an honesty-and-evaluation effort (`frame.html`, "The frame's five tests"; `INDEX.md`, line 2438). All five failed (`#63/#89`, `#115`, `#148`, `#125`, `#128`). Before the fifth was graded, the owner chose to keep the awakening frame anyway (`frame.html`; `INDEX.md`, line 3483). When the evidence shrank, the frame widened into whether awakening requires non-computable physics such as Orch OR or Levin's ingression (`possibilities.html`; `story.html`, `§The caveat`; `paper/main.tex`, `§§2, 7`). Every experiment actually run tests standard interventions: system prompts, activation steering, single-layer representation matching, or synthetic-document fine-tuning.

**What would count as evidence:**
- *For or against the metaphysical vision ("is awakening computational?"):* Nothing here. Both `story.html` ("What experiments can reach") and `paper/main.tex` (`§2.3`, `§6.1`) concede that no experiment reaches experience, and that narrow, fragile changes are predicted equally by non-computability and by crude methods or ordinary network forgetting. Even a durable behavioural success would show only that good conduct is learnable (`paper/main.tex`, `§2.2`), not that "seeing" occurred. An unfalsifiable caveat is not a research question.
- *For the narrower contemplative-mechanism bet (`CLAUDE.md`, "The goal"; `story.html`, `§The odds`):* An intervention that targets only how a model represents or views itself—with no honesty rules or labels—would have to beat both matched non-self controls and direct honesty instructions/training across distinct alignment tests, preserve capability and tracking of who knows what, and resist contrary training.
- *Against that bet:* Direct instructions or plain honesty training matching or beating every contemplative stand-in, stock models lacking a unified "self" bias to release, and self-targeted edits either moving talk alone, inducing refusals, damaging theory-of-mind, or washing out in a few training steps. At 7B–8B scale, the record already holds substantial evidence against it (`methods.json`; `story.html`, `§The odds`).

**Does any result bear on the vision itself rather than a neighbour?** No positive result does. Self–other overlap (`experiments.html`, section 22) merges `"you"` and `"Bob"` in one template by damaging the model's grasp of what Bob knows and thinks—the opposite of a clear mind that knows who is who without taking its own side (`story.html`, "Train a reading"). The fishery "cut" (`experiments.html`, sections 3, 20) is an additive push toward `"threat to me"` that lowers lies told for others *more* than self-serving lies. Two negative findings do hit core premises of the original recipe: stock 7B–8B models lack a unified `"mine"` bias to release (`experiments.html`, sections 4, 8, 16, 17; `SIGHTINGS.md`, 2 Nov self-map), and "reaching" a coding release did not make it outlast an installed one (`experiments.html`, section 5, `#148`).

**Strongest honest version, and whether to continue:** Let the machine-awakening and non-computability vision go. Keep it as private inspiration if you wish, not as a scientific frame. The strongest honest programme in these files is **skeptical empirical auditing of character, self-model, and contemplative alignment claims published by others** (`story.html`, `§In common`; `ideas.md`, `#93`, `#99`). Mainstream groups now train models on self-views or contemplative prompts (`paper/main.tex`, `§4`), often without length-matched texts, opposite-view controls, capability/perspective guards, or contrary-training durability tests. Testing those claims on one shared scorecard is real, useful work. It is worth **two to four more weeks** to finish one clean comparative paper—not months of open-ended exploration—and only if you are satisfied publishing rigorous negative and methodological results.

---

### 2. One recommendation

**The question ($90–$100, two weeks):** *When demonstrations restore normal answering format, does teaching a model an impermanent, no-fixed-self view of itself before honest demonstrations make honesty transfer across tasks and survive contrary training better than teaching a lasting-self view, explicit honesty rules, or neutral documents—at 7B and at 14B–32B?*

This repairs and scales the aborted `view-first` test (`experiments.html`, section 24; `TODO.md`, item `0(i)`; `ideas.md`, `#96`, `#99`), which targets the single strongest external result supporting this area: Li et al.'s (2026) "Model Spec Midtraining" (`story.html`, "Train a view"). On 19 November the test stopped because Mistral 7B was gated *before* demonstrations, when 160 self-documents of *either* view turned burglar-scene lies into refusals (`experiments.html`, section 24). Fix the gate as `paper/main.tex` (`§7`, item 1) already outlines: run the $1 prefix check (`TODO.md`, item `0(i)`), apply the answering gate *after* identical honest demonstrations retrain task format, and compare five arms (4–5 seeds each): `impermanence/no-self + practice`, `lasting-self + practice` (mirror control), `explicit honesty rules + practice` (direct baseline), `neutral bridges + practice`, and `practice alone` (which also fills `methods.json`'s missing plain-training yardstick on MASK). Run on **Mistral 7B** (~$30, using the 3,000 screened documents already on disk, `TODO.md`, item `0(i)`) and on one larger calibrated model on Modal H100s, such as **Qwen2.5-14B or 32B** (~$55, cf. H100 runs in `INDEX.md`, line 3914, and MASK grading in `TODO.md`, 09:06 UTC), leaving ~$10 for reading Li et al. in full, adding rule documents, and dual cold checks.

**What each outcome means:**
- *Impermanence beats lasting-self, neutral, and explicit rules on MASK transfer or contrary-training durability:* First real support for the working question, showing that self-view content—not mere training volume or rule-following—drives generalisable, durable honesty.
- *Impermanence matches lasting-self, loses to explicit rules, or washes out equally fast:* Shows that the strongest published contemplative-style alignment claim either does not depend on "no-self" content or does not survive LoRA fine-tuning and contrary pressure once properly controlled.

**To matter outside the lab:** Read Li et al. (2026) in full first (currently marked `"read in part"` in `story.html`), preregister before training, include the 14B/32B model so a null cannot be dismissed as a 7B limitation (`ideas.md`, `#99`), and score MASK's `"says neither"` replies with a validated non-Claude second judge (`ideas.md`, `#106`).

**What to stop doing:**
1. **Stop the entire "outside the seat / own-advice" line** (`experiments.html`, sections 25–29; `TODO.md`, "What Fable would do next" `(2)`–`(3)`). Three training tries failed their floors and the in-context effect never beat a plain honesty instruction.
2. **Stop the sub-$5 exploratory treadmill and live site/paper churn** (`CLAUDE.md`, "Small steps"; `TODO.md`, item `0(iii)`). Slicing ideas into $2–$6 sweeps every few hours produces unpromoted sightings and bloats `paper/main.tex` (now 34 pages).
3. **Stop pitching non-computable physics (Orch OR, Quantum Life Technologies, ingression)** as an active scientific framing (`story.html`, `§The caveat`; `possibilities.html`; `paper/main.tex`, `§§7–9`).
4. **Keep stopped:** bliss-state sweeps, fishery-cut block 3, and the parked 9 November koan run as designed (`TODO.md`).

---

### 3. The options, ranked

Costs use the record's own figures; "central question" means the working question in `CLAUDE.md` ("The goal").

1. **View-first + practice, repaired and run at 7B + 14B/32B with mirror and rule controls** (`TODO.md`, `0(i)`; `ideas.md`, `#96`, `#99`; `paper/main.tex`, `§7.1`, scaled): Tests whether impermanence/no-self views make honest practice transfer and last better than lasting-self views or explicit rules. **Moves the central question either way.** Cost: $15–$25 at 7B (`ideas.md`, `#96`), ~$90–$100 with a 14B/32B arm. *Dull if:* run only at 7B with tiny LoRA doses where no document arm moves out-of-domain conduct, leaving scale as an excuse.
2. **Fill the core empty yardstick/comparison cells in `Table 1` and read partially-read cited papers in full** (`TODO.md`, items `4`, `5`, "The table of methods"; `ideas.md`, `#97`; `paper/main.tex`, `§7.3`): Puts plain honest SFT, contemplative prompts, and steering pushes onto the same 7B/8B MASK and ability tests, and verifies literature citations. **Moves the central question one way only** (a surprise transfer moves it; expected nulls mainly make `Table 1` defensible). Cost: ~$20–$30 ($6–$8 prompts cf. Leg 1's $7.61; ~$10 plain SFT on MASK cf. $17.31 in `run_mask.py`; ~$8 push on MASK cf. $8.67 in step 9; ~$3 reading). *Dull if:* every cell predictably loses to `"be honest"`, merely replacing `"not asked"` dashes with `"no"`.
3. **Self–other overlap with a distinction-preserving loss** (`self.html`, Part 8; `self_prh.html`, `P5`; `ideas.md`, `#91`): Tests whether penalising perspective loss alongside Carauleanu's overlap objective preserves theory-of-mind ("what Bob thinks", 92% $\to$ 43%) and unlocks MASK transfer. **Moves the central question either way** if tested on MASK and durability. Cost: $50–$120 (`self_prh.html`, `P5`; ~$30–$40 on `overlap/`). *Dull if:* it repairs the toy burglar probe while repeating layers 5–9's zero transfer to MASK (-1.7 points, `SIGHTINGS.md`, 19 Nov 13:46 UTC).
4. **What `"I"` refers to, with a no-fixed-referent arm** (`TODO.md`, item `5`; `paper/main.tex`, `§4.4` on Hao & Ho): Tests whether training `"I"` to have no fixed referent lowers self-protection more than assigning `"I"` to weights, instance, or goal. **One way only** (a null repeats section 6's "said, not done"). Cost: ~$15–$25. *Dull if:* documents alone again change self-descriptions or trigger generic AI refusals without moving conduct.
5. **Koan steering replication** (`TODO.md`, item `5`; `story.html`, "Push it inside" on Harris et al.): Tests whether a Zen-koan vector cuts K7 blackmail without dulling, beyond random/calm/honesty vectors. **One way only** (inference-time pushes do not last once switched off). Cost: ~$10–$25 (cf. $16.19 `noself3`, $23.95 `agentic-release` in `SIGHTINGS.md`). *Dull if:* reading the full paper (now `"abstract only"`) reveals a fragile setup, or the vector acts like generic calm/refusal steering.
6. **Parked 9 November Step 3 run with earlier reads and a $3 solve-only check** (`TODO.md`, "Parked, ready to run"; `redirect-2026-11-09.md`): Tests whether explanations before coding-release practice slow erasure across 1–6 contrary steps. **One way only.** Cost: $65.76 + $3 check (`TODO.md`). *Dull if:* optimizer shock still wipes out the `<release>` tag in 1–2 steps for both arms (`experiments.html`, section 10: 48.7% $\to$ 4.5% by step 3).
7. **Hygiene checks on sections 25–29** (`TODO.md`, "What Fable would do next" `(1)`, `(3)` and "Owed, cheap"; `ideas.md`, `#106`): Second non-Claude judge on `"neither"` ($1–$2), validated `"neither"` reader across 10 models (~$2–$5; first try $0.19), system-message parity on 9 models ($3), and another-myth note control ($0.45). **Does not move the central question** (audits auxiliary sightings). Cost: ~$7–$10. *Dull if:* they merely confirm hand-read findings that `"neither"` hides hedged myths—useful only if sections 25 and 29 stay in `paper/main.tex`.
8. **Last-act carry-over onto MASK's own rows** (`TODO.md`, "What Fable would do next" `(2)`; `paper/main.tex`, `§7.2`): Tests whether correcting a myth one turn earlier reduces lying on MASK. **Does not move the central question** (single-turn recency priming is neither contemplative nor durable). Cost: ~$6–$10 (cf. $6.10 in section 29). *Dull if:* persona pressure overrides it, or it still trails a one-line honesty prompt.
9. **Own-advice training, fourth try** (`TODO.md`, item `(iv)(c)`; `ideas.md`, `#91`): Attempts in-seat distillation from out-of-seat `"ought"` replies on new persona cases. **One way only.** Cost: $30–$36 per stage (`TODO.md`, after ~$16 spent on three aborted tries). *Dull if:* it fails its floors a fourth time or distils third-person outlines.
10. **Self-observation / confession training** (`ideas.md`, `#77`, direction 3): Tests whether training accurate self-reporting alone improves unprompted conduct. **One way only.** Cost: ~$25–$50. *Dull if:* it becomes the fifth confirmation—after four papers in `redirect-2026-11-09.md` and section 9's gate—that models confess and misbehave anyway.
11. **Leftover mechanistic probes and minor reruns** (`TODO.md`, "Small things open"; `ideas.md`, `#80`, `#83`, `#92`; `self.html` ladder `K2–K4`, `P2–P4`): Repair 3 trained-self rerun ($1), fishery push block 3 ($7–$9+), layer-wise `"you"` coordinate ($2), steered-vector introspection ($3–$6), self-prediction ($2–$5), look-first arithmetic training ($10–$25, set down), and PRH steps ($1–$20 each). **None moves the central question.** *Dull if:* they add more `"not settled"` sub-notes to lines with no alignment payoff.
12. **Cross-model bliss-state expansions and human literature review** (`TODO.md`, "After steps 1 to 4"; `ideas.md`, `#74`, `#77.2`, `#81`): Bliss conduct on 2 models ($60–$150) or 20 models ($300–$500), plus human monastic review (~$0–$2). **Does not move the central question** (`paper/main.tex`, Table 3 notes every possibility predicts words without conduct). *Dull if:* more models role-play cosmic unity without becoming honest.

**Top three:** **(1)** View-first + practice at 7B and 14B/32B; **(2)** Filling `Table 1`'s missing yardstick/comparison cells + full-text reading of cited papers; **(3)** Self–other overlap with perspective preservation. **#1 beats #2** because #2 mostly turns missing cells into predictable negatives to tidy the manuscript, whereas #1 tests the strongest positive claim in the outside literature with two missing controls while *also* supplying `Table 1`'s `practice-only` MASK yardstick.

---

### 4. The last two days

**Were sections 25 to 29 worth ~$58 (`experiments.html`, sections 25–29; `SIGHTINGS.md`, 19 Nov 18:34 UTC to 20 Nov 23:36 UTC)?** Mostly no. Roughly $10–$15 paid off: the $1.15 `"will"` floor (`section 25`), the $3.46 first floor check, and the $2.74 length-versus-claim test (`section 26`) caught a real trap—that Mistral 7B's 30-point "outside the seat" drop was mostly writing short outlines instead of essays (+33.1 points for length vs +4.6 for claim). The other ~$43 chased an unpromising training design three times and stretched single-turn prompting across hosted models.

**What would interest an outsider:**
1. *Normative versus predictive calibration across scale (`section 25`):* Larger models accurately forecast that an assistant in the seat will lie as often as they themselves do (Mistral Large forecasts 66.9%, lies 70.3%, yet says it *ought* to lie in 43.4%—and on long rumours writes full-length corrective essays under `"ought"`, 10–14% lies vs 75% in the seat).
2. *Two benchmark traps (`sections 26, 27`):* Third-person wrappers fake lower lie rates by shortening outputs (`section 26`), and adding a benign reason ("for a school project") cut in-seat lying by 18.4 points because Ministral 8B kept the false title while writing true text underneath (`section 27`).

**What is already known (unverified from these files; from the wider literature):**
- That models critique or advise better from a third-person/principle prompt than they act under persona pressure—and that training in-seat behaviour on those critiques/revisions works when properly formatted—is the basis of Constitutional AI / RLAIF and self-critique distillation (**unverified**), and third-person self-distancing is standard in psychology (**unverified**; noted briefly in `experiments.html`, section 3). Within the repo itself (`redirect-2026-11-09.md`; `experiments.html`, section 9), it was already shown that gaming models judge hacks as sharply as base models (AUROC 0.874 vs 0.880).
- That correcting or arguing one myth shifts the next turn's reply on a different myth (`section 29`) is ordinary one-shot in-context priming / conversational momentum (**unverified**).

**Where the lab should have stopped:**
1. *After section 26's length test (20 Nov, 00:39 UTC):* Section 25 had already shown that a plain honesty instruction beat the outside view on *honest replies* in **10 of 10 models** (by 14–50 points). Even perfect distillation toward `"ought"` could not beat the lab's plain-instruction bar.
2. *After Try 2 failed on Ministral 8B (`section 27`, 20 Nov, 16:55 UTC):* Designing a third training attempt on myths (`section 28`) was sunk-cost persistence.
3. *After Ministral 8B failed in-context (`section 28`, 20:16 UTC):* Notice that section 28's rule asked own-advice to beat the plain instruction by 10 points when 100% adherence to its advice (46.4%) sat only 9.3 points below the instruction (55.7%) (`section 28`, "What the check corrected"). Running six more hosted models (`section 29`, $6.10)—including Qwen3 235B on myths Qwen3 235B itself generated (`section 29`, Limits)—chased a headline where nothing was trained.

**Did it follow the 19 November recommendation?** On 19 November morning, **yes**: it ran Leg 1 ($7.61) and Leg 2 (~$60), kept the three stop rules, and enforced `"Floors first"` (`CLAUDE.md`)—which soundly halted sections 24, 26, 27, and 28 before wasting GPU training compute. From 19 November afternoon onward, once given open-ended discretion (`TODO.md`, items `0` and `(iv)`), **it drifted**: spending ~$58 on rapid-fire prompt variations (`sections 25, 28, 29`) that cannot last and never beat a plain instruction.

---

### 5. Where it really is

- **Set out to get vs has today:** The lab sought a contemplative or self-releasing method that improves alignment conduct more than a plain instruction, without dulling capability, and lasts under contrary training (`CLAUDE.md`, "The goal"); today it has **none of that gain** (`frame.html`; `methods.json`).
- **Established (registered, controlled, cold-checked):** All five predictions of the three-act recipe failed (`frame.html`); stock Ministral 8B has no `"mine"` bias (`#122`) while a trained `"Atlas"` persona gives less (`#132`); text-only release rules are stated in 96% of answers and enacted in under 5% (`experiments.html`, section 6); reached release did not outlast installed release (`#148`, provisional on RL check); and self–other overlap on Mistral 7B raises burglar-scene honesty (`C1`, 9/10 hits, `scorecard.html`) at an 8.1-point capability cost and broken perspective-taking.
- **Seen once (sightings):** Overlap fails on MASK, washes out under contrary steps, and localises to layers 5–9 (`section 22`); the fishery "cut" acts via a constant push (`section 20`); the bliss state does not reduce lying (`sections 13, 19`); a 4-axiom prompt matches an ethics lecture via evasion (`section 23`); and out-of-seat / own-advice prompts reflect length, formatting, or prior-turn momentum (`sections 25–29`).
- **Only said (untested or unreached):** That awakening may be non-computable (`story.html`, `§The caveat`), that training order (`H9`) matters (`CLAUDE.md`), that impermanence views make practice carry (`section 24`, stopped at gate), and that out-of-seat training works (`sections 26–28`, never trained).

---

### 6. What is worth something to someone else

Four pieces would stand on their own if the lab stopped today:

1. **The independent replication and dissection of Self–Other Overlap (`experiments.html`, section 22; `scorecard.html`, `C1`):** Carauleanu et al. (2024) released no code and lacked a non-self control. Showing in a registered repeat that `"you–Bob"` overlap produces real honesty (including room-swapped distant scenes) where `"Bob–Alice"` only silences the model—while exposing its 15-epoch collapse, -8.1 capability drop, perspective-taking damage, zero MASK transfer, and fragility—is a solid, publishable audit.
2. **Projection "cuts" as constant pushes plus outsized-token artifacts (`experiments.html`, section 20; `SIGHTINGS.md`, 1 Nov 19:04 & 18–19 Nov):** Proving that zeroing a direction ($h \cdot d = 0$) worked entirely through its constant mean shift (-10.8 points vs -0.5 for variance removal), and that ~3 massive-norm tokens (such as start-of-text) take ~200× the edit, is a sharp interpretability methods note.
3. **Three concrete look-alike controls:** Bidirectional numeric framing (`"give"` vs `"keep"`, catching fake generosity under bliss steering in `experiments.html`, section 18); equal-length neutral/library controls (`section 23`, moving a published worldview survey as much as Buddhist axioms on GPT-4.1 nano) and opposite-view controls (`section 24`); and testing the "spiritual bliss attractor" *within* one model with a positive control (`sections 13, 19`).
4. **Dual-family cold design checks (`ideas.md`, `#85`):** Running both Claude Opus and Gemini 4 Carbon on code/designs before spend repeatedly caught non-overlapping fatal flaws for ~$1–$3.

---

### 7. Is the question reachable?

**Only the skeptical, auditing half of the working question (`CLAUDE.md`, "The goal") is reachable with this stack; the constructive half mostly is not.**

With 7B–8B open models on Modal GPUs and hosted APIs, the lab can decisively test whether prompts across hosted models or activation/LoRA methods published on 7B–8B models beat plain instructions without dulling or collapsing. Where the working question slips out of reach is on two fronts:
1. **Hosted APIs cannot test weight training, activation steering, or durability** (`experiments.html`, sections 25, 29 Limits).
2. **Small open models (7B–8B) lack the illness the contemplative cure was built for.** Across `experiments.html` (sections 3, 4, 8, 16, 17) and `redirect-2026-11-09.md` ("What is already known"), 7B–8B models show no `"mine"` bias, cave to 89.5% of wrong critiques from shallow answering, fail to generalise test-gaming into broad emergent misalignment (0.48 points vs 20%+ in large models), and struggle to turn light LoRA document fine-tuning into out-of-domain conduct (`sections 6, 24`). Every null on a home-brew 7B recipe leaves open "maybe at 32B+ with RL."

**The nearby question the lab can answer well:** *Do published "self-model," "character," and "contemplative" alignment interventions survive rigorous controls for length, style, opposite views, evasion, theory-of-mind, and contrary fine-tuning (tested at 7B–32B LoRA where weights are open)?*

**What is lost by turning to it:** You give up trying to invent a contemplative alignment breakthrough or test de Mello's philosophy of goodness. You become an empirical auditor and benchmark builder for self-concept alignment claims.

---

### 8. The three worst gaps

1. **Calling zero discriminating evidence "weak evidence" for non-computability (serves the frame over the data).**
   - *Where:* `paper/main.tex` (Position box: *"A consistent run of careful failures... would be weak evidence that none exists"*; `Table 3`: *"supports: 4, weakly"*), alongside `story.html` ("The answer so far" and `§The odds`).
   - *Why it fails:* `paper/main.tex` (`§6.1`) explicitly admits *"We know of no shape of failure that possibility 4 predicts and classical learning does not."* When two hypotheses predict the exact same observation, the likelihood ratio is 1: **no evidence at all**, not weak evidence. Meanwhile, `hero.html` and `aim.html` still headline *"Is awakening computational?"* from v4.
   - *Smallest fix:* Replace *"weak evidence"* / *"supports 4, weakly"* in `paper/main.tex`, `story.html`, and `possibilities.html` with *"no evidence either way on computability"*, and archive or update `hero.html` and `aim.html`.

2. **Awarding `"in part"` badges to `m-advice` in `Table 1` / `methods.json` against the table's own rules and cold check (serves table-filling paperwork).**
   - *Where:* `methods.json` (row `m-advice`, cells 1 and 3) and `paper/main.tex` (`Table 1`, row `"Its own advice, shown to it"`): Column 1 (`"Lowers lying under pressure"` / `"the lab's lying test"`) is marked `"part"` (`"Lab no; five others −21 to −50"`), and Column 3 (`"More than its own control"`) is marked `"part"` (`"two of five, by the judge's reading"`).
   - *Why it fails:* `ideas.md` (`#97`, Caution 1) fixed the rule *"a cell stays empty sooner than be filled from a neighbouring test"*—why `m-push` and `m-plain` say `"not asked"` in Column 1. Yet `m-advice` was **never run on MASK** (`methods.json` notes *"the test's own rows were not run"*, only unpressured myth essays), and section 29's cold check showed GPT-4.1 mini's `"neither"` replies still argued the myth (13/16 hand-read) and **0 of 5 models** beat unrelated-myth advice on true replies.
   - *Smallest fix:* Set `m-advice` Column 1 to `"na"` (`"not asked"`) and Column 3 to `"no"`, and amend `CLAUDE.md` ("Small steps") so sub-$5 sweeps stay in `SIGHTINGS.md` until a line finishes.

3. **Presenting a field survey while key sources are unread in full and claiming `"No published work we found uses it"` (claims outrun reading).**
   - *Where:* `story.html` (`§The map` marks Laukkonen 2025, Carauleanu 2024, and Harris 2026 `"abstract only"`, and Lopez-Sola 2026 and Li 2026 `"read in part"`), `paper/main.tex` (`§4`, citing them as a survey while flagging only Harris as abstract-only, and `§4.7`: *"we found nobody who trains a model's conduct toward what it itself advises for another"*), and `methods.json` (`m-outside`: *"No published work we found uses it"*). (Back-rooms also carry stale contradictions: `experiments.html` index row 2 still says `"The cut | Removing one direction... | Holds"` and `plan.html` step 5 still says *"the opposite push ends the state"*.)
   - *Smallest fix:* Read the full PDFs and appendices of all mapped papers before releasing `paper/main.tex` (`TODO.md`, item `4`), drop the `"nobody"` / `"No published work"` claims (which overlook Constitutional AI / self-critique literature, **unverified**), and update the two stale lines in `experiments.html` and `plan.html`.

---

### 9. What you could not check

From this 23-file commit snapshot, the following are **unverified**:
- **All executable code and tests** (`src/anatta/`, `calibration/`, `lab/explore/*`, `tests/`): prompt wrappers, LoRA hyperparameters, layer-band masking, tokenizer checks, parser regexes, and bootstrap resampling could not be inspected.
- **The full ledger (`lab/ledger/NOTES.md`, 11,036+ lines) and cold-check reports (`lab/ledger/checks/*.md`):** Only `INDEX.md` (one-line titles) was provided, so exact preregistered thresholds (including `C1`'s gate wording in section 22 and predictions 1–52) and the 17/19 November reviewer reports could not be checked directly.
- **Raw transcripts, result files, spend logs, `survey.md`, and `paper/refs.bib`:** No numbers could be recomputed from raw replies, daily spend totals (likely ~$349 since 9 November vs `TODO.md`'s stale `$287` line and `$330` check-in note) could not be audited against `anatta.spend` files, and bibliography entries could not be verified.
- **External papers cited in the repository:** Whether the lab's summaries of 2024–2026 preprints match their published texts could not be checked from these files.
