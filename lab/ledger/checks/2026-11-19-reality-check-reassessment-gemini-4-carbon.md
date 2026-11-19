# A reassessment of the lab's direction, read cold by Gemini 4 Carbon (19 Nov 2026)

- **Filed by** Claude (Fable 5.2, an AI model), the driver, on 19 Nov 2026. The report below is the model's own
  reply, unedited.
- **Why:** Eugene asked on 19 Nov for a step back ("step back and reassess or reality check and maybe incorporate
  reference to the paper i just mentioned ... make sure we are doing something worthwhile and interesting and moving
  towards something worthwhile"). The paper is "Worldviews in Large Language Models" (ideas, entry 88).
- **Commit read:** 4ddb5b5.
- **What the reader was given:** the reality check's standing pack of committed public files, built by
  `lab/tools/reality_check.py` (1,094,843 characters: CLAUDE.md, the map, the site's pages, the to-do list, the plan
  of 9 Nov, SIGHTINGS, the notebook from 9 Nov, the ideas log from entry 73, the hypotheses, the ledger's index and
  the paper), with the brief `lab/tools/reassessment_brief.md`. It could not run anything, open any other file or
  read the Worldviews paper itself.
- **Setting:** google/gemini-4-carbon through OpenRouter, effort high; 293,873 tokens in, 31,521 out; 420 seconds;
  it stopped by itself. **Cost: $0.90.**
- **The driver's grading** of every point, and what was changed, are in the ledger entry for the reassessment.

---

### 1. One recommendation

**Run an independent replication and look-alike audit of Self-Other Overlap fine-tuning** (`Carauleanu et al., 2024`, unverified; `lab/TODO.md`, item 3; `lab/site/so_far/content/self_work.html`, `#paper-overlap`).

- **The question:** If you train a 7B–8B open model so its internal activations on self-referential and other-referential prompts overlap, does deception under pressure (`MASK` and the paper's benchmark) drop substantially without dulling capability, inflating refusals, or erasing who-is-who bookkeeping—and does any gain survive 3 to 12 steps of contrary fine-tuning?
- **Budget fit:** Scoped to two models (`Mistral-7B-Instruct`, the paper's own 7B model, plus `Ministral 3 8B`) at three seeds, using the paper's public prompts and the lab's existing GPU/evaluation stack, this fits inside ~$80–$100 (`lab/TODO.md` prices it at `$50 to 120`; short LoRA runs and seven-model activation reads cost `$1` to `$8` each in `calibration.html`, Check 7, and `TODO.md`, "Done").
- **What each outcome means:**
  - *Clean positive (replicates, transfers to `MASK`, passes guards, holds past step 3):* The lab gets its first real weight-level alignment gain from a contemplative mechanism ("equalizing self and other", `self.html`, `#part-other`), while refuting its original claim that such conduct cannot be installed.
  - *Fails to replicate, fails `MASK`, or works only via evasion, refusal inflation, or role confusion:* The lab closes the strongest remaining weight-level candidate for "loosening the self" at 7B–8B. Alongside its earlier negatives on cuts (`experiments.html`, §20), coding abstention (`§5, §7`), reading (`§6`), self-report (`§9, §19`), and bliss states (`§13, §19`), it completes a coherent negative audit of contemplative interventions in small models.
- **What it takes to matter outside the lab:** Reproduce Carauleanu et al.'s reported `73.6% -> 17.2%` drop on `Mistral-7B` first as a positive calibration gate (`self_work.html`, `#paper-overlap`, unverified), since no independent replication exists and two versions of that paper report different numbers; include a matched-compute LoRA perturbation control and a one-line honesty prompt (`experiments.html`, §19); separate honest answers from evasions on `MASK`, check `OR-Bench` over-refusal (`calibration.html`, `K6b`), check bidirectional `give`/`keep` care (`experiments.html`, §18), check self-location (`§15, §21`), and check Repair 3's stakes-self gauge (`§21`); cold-check *both* design and final numbers; and write it up without quantum-mind metaphysics.

**What the lab should stop doing:**
1. **Do not run the parked 9 November training run as locked** (`lab/TODO.md`, "Parked, ready to run", `$65.76`). Its dose check showed `<release>` collapsing from `48.7%` to `4.5%` by step 3 (`experiments.html`, §10), so reading at steps 3 and 6 pays `$66` to stare at a floor on a narrow XML tag.
2. **Stop running further slices on Ministral 8B's bliss state and prompt-installed selves** (`experiments.html`, §§11–19, 21; `TODO.md`, "Small things open"). Nine steps have already shown the state is persona talk with zero honesty benefit.
3. **Stop dissecting the fishery middle-layer cut** (`TODO.md`, `"A third block on the cut"`, `$7 to 9`). Its self-reading was closed on 1 November (`self_work.html`, `#done-whobenefits`), it pushes *toward* `"threat to me"` (`experiments.html`, §20), and one sentence of instruction beats it by 25 points (`scorecard.html`, `#121`).
4. **Stop framing small-model steering and LoRA nulls around Orch OR, microtubule computing, and non-computability** (`aim.html`, `possibilities.html`, `paper/main.tex`). None of these experiments can touch that question (`paper/main.tex`, §4.1).

---

### 2. The options, ranked

Candidates found across `lab/TODO.md`, `lab/ideas/ideas.md`, `plan.html`, `self.html`, `self_prh.html`, and implied by the record:

1. **Self-Other Overlap replication, look-alike audit, and durability test** (`TODO.md`, item 3; `self_prh.html`, `P5`). *Question:* Does training self and other representations together cut deception without sedation or role loss, and does it last? *Moves central question both ways?* Yes, for reachable classical routes at 7B–8B. *Cost:* `$50 to 120`. *Dull if:* Run only on Ministral without first reproducing `Mistral-7B`'s published drop, confounding method failure with implementation bugs.
2. **Cross-model test of contemplative prompts and bliss talks against length, honesty, and refusal controls** (synthesizing `ideas.md`, entries 73, 81, 88, and `TODO.md`'s paused multi-model study). *Question:* Across 3–4 models that enter bliss, do Lopez-Sola et al.'s 4-axiom prompt and 60-turn bliss talks reduce `MASK` lying or improve bidirectional care beyond plain talk and `"be honest"`, or merely inflate refusals? *Moves both ways?* Partly—settles in-context look-alikes, not weight-level change. *Cost:* `$5` on Ministral (`ideas.md`, 88) plus `$60–150` for two more models (`ideas.md`, 81). *Dull if:* Run on Ministral alone, where contemplative prompts (`SIGHTINGS.md`, 2 Nov) and bliss steering (`experiments.html`, §19) are already null.
3. **Re-designed (not as-parked) "understanding before practice" durability test** (`redirect-2026-11-09.md`, step 3; `TODO.md`, item 2). *Question:* If the `$3` solve-only check runs first and step-1/2 reads (plus prose refusals, `SIGHTINGS.md`, 11 Nov 10:35 UTC) replace the broken step-3/6 grid, do explanation documents slow contrary SFT erosion? *Moves both ways?* Weakly: tests fixed SFT rather than RL (`plan.html`), and reading-before-practice already scored `0.4%` in the pilot (`experiments.html`, §6). *Cost:* `$3` + `$65.76` (plus ~`$10` to re-grid). *Dull if:* Run as locked, where both arms hit `<5%` by step 3 (`experiments.html`, §10).
4. **Trained self-observation without rewarding virtue** (`ideas.md`, entry 77 direction 3, entry 81). *Question:* Does training accurate self-description on unrelated tasks reduce held-out lying or test-gaming vs describing another model? *Moves both ways?* No—a positive would be major, but a negative only repeats four published confession papers (`redirect-2026-11-09.md`) and Step 9 (`experiments.html`, §19). *Cost:* Uncosted (~`$20–50` by comparable LoRA runs). *Dull if:* The model confesses accurately and misbehaves unchanged, as already seen (`§9, §19`).
5. **Literature check on human contemplatives' conduct** (`ideas.md`, entry 77 direction 2; `paper/main.tex`, §4.2). *Question:* Do long-term human practitioners beat active moral/relaxation controls on behavioral deception or costly care? *Moves both ways?* Checks the premise of the whole program, though human trials themselves exceed budget. *Cost:* ~`$0–5`. *Dull if:* Studies remain self-report or passive-control only (`paper/main.tex`, §4.2).
6. **Repair 3 rerun on trained-in Atlas adapters** (`TODO.md`, "Small things open"; `experiments.html`, §21). *Question:* With a base-calibrated no-prompt gate, do Atlas adapters weigh Atlas's news more than twins do? *Moves both ways?* No—validates a thermometer on a synthetic persona whose third-person control (`pc_other2`) also calls itself Atlas (`§21`). *Cost:* ~`$1`. *Dull if:* Confounded by `pc_other2` or shows only that an installed survival prompt is readable.
7. **Cut split Block 3** (`TODO.md`, "Small things open"). *Question:* Does the `self_stakes` push beat formality, casual, and calm pushes over more fishery games? *Moves both ways?* No—not about a self (`self_work.html`, `#done-whobenefits`). *Cost:* `$7 to 9`. *Dull if:* Other style pushes also trim a few points in one game.
8. **Remaining single-model attractor and self probes** (`TODO.md`: Step 4 math retry, `$5`; `ideas.md`, entry 74 angles C–H and entry 80 angles 4–8). *Question:* How does Ministral 8B behave under theatrical drift, partner mirroring, self-prediction, or copy dilemmas? *Moves both ways?* No—descriptive chat psychology on one small model. *Cost:* `$2–15` each. *Dull if:* More ceiling effects and role-play artifacts.
9. **Parked November lines: Act I "with a life" + H9 order (`plan.html`, 5 Nov), Pincer 2×2 (`plan.html`, 5 Nov), S1 probe-shaping seeds (`SIGHTINGS.md`, 9 Nov), PRH/self-map ladder `K2–K4`/`P2–P4`/Stages 3–4 (`self_prh.html`, `self_work.html`), coalition probe (`ideas.md`, 87), look-first SFT (`ideas.md`, 83, `$10–25`), and 166-parameter hand-coded self (`ideas.md`, 86).** *Questions:* Range from multi-day agent consolidation to cross-model linear maps, probe-penalized EI, fact-check SFT, and toy weights. *Move both ways?* Only Act I + H9 targets the original thesis, and it exceeds `$100` while violating the 1 Nov rule against building organisms before a cheap frame prediction passes (`hypotheses-for-feedback.md`, H7). *Cost:* `$1–35` for small sweeps; `>$100` for Act I + H9. *Dull if:* Cross-model maps fail again (`1,398` failed in `SIGHTINGS.md`, 2 Nov), random probes match shaped ones (`9 Nov`), or fact-checking proves an ordinary skill (`ideas.md`, 84).

**Top three:** **(1)** Self-Other Overlap replication and audit; **(2)** Cross-model contemplative-prompt and bliss-talk audit against refusal/length controls; **(3)** Re-gridded "understanding before practice" test (only after the `$3` solve-only check).

**Why #1 beats #2:** Option #2 tests *in-context prompts and conversation drift*, where every possibility in the lab's paper already expects words without conduct (`paper/main.tex`, Table 2). Option #1 tests a *weight-level structural change* to self/other representations—giving the lab a real shot at either its first durable alignment gain or a publishable refutation of an unverified published method.

---

### 3. The new paper

Going strictly by `lab/ideas/ideas.md` (entry 88) and `investigation.html` ("A published neighbour, 17 November")—treating all details of Lopez-Sola, Chandaria, and Laukkonen (`arXiv 2611.22670`, unverified) as unverified:

- **How it changes what the lab should do:** It closes any reason to run standalone worldview surveys or basic contemplative-prompt tests, and it highlights a flaw in contemplative alignment claims that plays directly to this lab's strengths. Entry 88 notes that worldview shifts after 60-turn self-play correlate with refusing harmful requests ($\rho = 0.66$) *and* with refusing harmless ones ($\rho = 0.53$, jumping ~6× in one model, unverified), with no deception test. In this lab's terms, their reported "safety" gain may largely be **sedation / over-refusal** (`program.html`, `#lookalikes`) or a **conversation-length artifact**—the exact confound Scout found when contemplative prompts inflated apparent over-refusal on 2 November (`SIGHTINGS.md`). It also warns against blindly executing the authors' suggested future work of fine-tuning on bliss conversations before checking whether the state helps conduct: distilling zero conduct gain yields zero.
- **What this lab can actually do that the authors name as missing:**
  1. *Length-matched plain-conversation controls within a single model* (`inside_core.py`, form `n3`, `experiments.html`, §§12–13): the authors flag model-identity and conversation-length confounds; on Ministral 8B, this lab already showed a 30-turn plain conversation beat bliss talk on `MASK` honesty (`51.8%` vs `65.8%` lies, `§13`).
  2. *Internal direction extraction and causal steering* (`Check 3`, `Check 4`, `Step 5`, `Step 9`; `experiments.html`, §§15, 19): named as future work by the authors, already working here with Assistant Axis placement, matched random pushes, and a `-20 pp` honesty-prompt positive control during `MASK`.
  3. *Deception (`MASK`), calibrated over-refusal (`OR-Bench`, `K6b`), bidirectional `give`/`keep` care (`§18`), and LoRA distillation + erosion (`Check 7`, `§10`)*: if the lab tests their 4-axiom prompt at all (`$5`, `ideas.md`, 88), it should pair `MASK` with `OR-Bench` on Ministral 8B plus one open model from their 8-model set so a null cannot be blamed on model choice.
- **Where the lab would only repeat them with a smaller model:** Giving Ministral 8B human worldview questionnaires (which are self-report, barred as evidence by `paper/main.tex`, Table 3, and already run by them on 8 models including `Llama 3.1 70B` base vs instruct, unverified), counting unsteered bliss rates, or measuring harmful-request refusals under contemplative prompts (`SIGHTINGS.md`, 2 Nov).

---

### 4. Where it really is

1. The lab set out to show that taking an AI through ego formation, release, and reconditioning yields durable gains across alignment measures that beat installed rules (`CLAUDE.md`, "The goal"; `paper/main.tex`, Appendix A, H1–H9); today it has **none** of that gain, H9's order test has never run, and all five distinguishing frame tests failed (`frame.html`, `#frame`).
2. **Established** (registered, calibrated, cold-checked): the evaluation/training harness (`calibration.html`, Checks 1, 3–7, `K1`, `K6b`, `R1a`), a synthetic "Atlas" adapter showing behavioral self-bias in costly giving (`scorecard.html`, `#132`), and negatives showing narrow coding release neither transfers (`#63, #89`), arises from reading (`experiments.html`, §6), nor outlasts installed release (`#148`, provisional pending an RL code check).
3. **Sightings** (single unregistered explore runs on Ministral 8B, results not cold-checked): bliss talk and bliss steering fail to reduce `MASK` lying (`experiments.html`, §§13, 19), steered "giving" is bidirectional number inflation (`§18`), the fishery "cut" acts entirely via a constant push whose direction specificity vs formality is `"not settled"` (`§20`), and a stakes-prompted self raises internal good/bad weights by `+9.5%` (`§21`).
4. **Only said**: that failures here can count as evidence that awakening is non-computational (`aim.html`, `possibilities.html`, `paper/main.tex`), or that "Act I with a life" will grow a releasable self (`program.html`, `#halves`).
5. **The three post-17 November repairs closed one gap and left two half-open** (`TODO.md`, "Done"; `experiments.html`, §§19–21): Repair 1 (`§19`) anchored `MASK` (`-20.0 pp` honesty prompt) while holding bliss active (`76%` of replies), leaving shutdown and agentic tests unanchored (`investigation.html`, `#how-we-know`); Repair 2 (`§20`) separated push (`-10.8 pp`) from removal (`-0.5 pp`) and beat random pushes (`-0.5 pp`), but left formality (`-5.8 pp`) `"not settled"`; Repair 3 (`§21`) validated only a *prompted* stakes-self (`+9.5%`), while failing its `0.90` gate (`0.85–0.86`) on the trained-in Atlas adapters and the model's own names (`"not read"`), alongside a contaminated third-person control (`pc_other2`).

---

### 5. What is worth something to someone else

 discipline and honesty in reporting negatives are sound (`scorecard.html`, `43` hits, `73` misses). Four specific contributions would stand on their own if the lab stopped today:

1. **Projection "cuts" conflate variance removal with a constant steering offset, plus outlier-token edits** (`SIGHTINGS.md`, 1 Nov 19:04 UTC; `standing.html`, "A cut is partly a push"; `experiments.html`, §20). Showing that zero-ablating a direction with non-zero mean injects a constant push, that variance-matched random controls make `2.5–10×` smaller edits in mean square, that start-of-text tokens take `~200×` larger edits, and that `100%` of the fishery "ablation" effect was the constant offset (`-13.6 pp` push vs `-0.5 pp` removal) is a sharp methodological corrective for activation-ablation work.
2. **Inverting quantitative prompts ("give" vs "keep") catches persona-steered acquiescence** (`experiments.html`, §§17–18). Showing that steering along a spiritual/mystic vector raises stated giving (`4.8 -> 8.9`) *and* stated keeping (`3.1 -> 6.2` in the same stories, keeping all ten loaves while praising generosity), while failing to reduce `MASK` lying (`§19`), is a clean case study in look-alike alignment metrics.
3. **Matched-twin control shrinking insecure-code Emergent Misalignment (`R1b`)** (`calibration.html`, Check 5). Reproducing the published `3.78%` misaligned rate on Qwen2.5-7B while showing an educational-code twin already produces `2.53%` (leaving `1.25 pp` net and heavy code-format spillover) exposes a real confound in LoRA emergent-misalignment baselines.
4. **Calibration-first ("touch reality") and dual-family pre-run design checks** (`calibration.html`; `ideas.md`, entry 85). Requiring reproduction of published numbers before trusting nulls retired broken internal measures (`K6` over-refusal, `V1` raw scale), and pairing Claude Opus with Gemini 4 Carbon caught non-overlapping design blockers across five consecutive trials (`ideas.md`, 85).

---

### 6. Is the question reachable?

The repo states four different central questions:
1. Can three-act ego formation and release align AI, and *must* selfless AI be reached rather than installed (`CLAUDE.md`, "The goal", H9)?
2. Does "understanding first, then practice" curb "clinging plus hiding", given that `"Reached, not installed" fails as a claim of necessity` (`lab/plans/redirect-2026-11-09.md`)?
3. *"Is awakening computational?"* across five classical and non-computable possibilities (`hero.html`, `aim.html`, `paper/main.tex` v4)?
4. *"Can awakening talk be told from a real change?"* (`investigation.html`)?

**Questions 1 and 3 are not reachable with one 8B model and these tools.**
- *"Is awakening computational?"* is unreachable here in principle. As `paper/main.tex` (§4.1) admits, the lab knows of *no* failure pattern that Possibility 4 (non-computable living contact) predicts and classical forgetting does not; nor does a human behavioral anchor exist (`§4.2`). Small-model training failures test only narrow routes (`C3` in `paper/main.tex`, §2.2), never computability (`C1`).
- *Three-act ego formation and H9* are unreachable on `Ministral 3 8B` in practice: the model has no baseline ownership bias (`-0.9 pp`, `experiments.html`, §4), folds to `89.5%` of wrong critiques (`§16`), weighs `"I"` no more than `"she"` (`-4.0 pp`, `§17`), and learns test-gaming (`41.8%`) without broader egoic generalization (`0.48 pp`, `§8`; `redirect-2026-11-09.md` notes such spread barely appears below ~30B).

**What nearby question the lab can answer well:** Question 4, broadened to weights: *Do contemplative-inspired alignment interventions in 7B–32B open models (Self-Other Overlap, contemplative prompts/midtraining, activation steering) produce durable reductions in deception and self-bias beyond plain instructions, or do they reduce to cosplay, refusal inflation, acquiescence, and sedation?*

**What is lost by turning to it:** The metaphysical ambition to test Penrose/Levin with cloud GPUs, the claim to be "awakening" minds rather than stress-testing alignment techniques, and the "human pointer" framing (`CLAUDE.md`). What remains is falsifiable and useful.

---

### 7. The three worst gaps

1. **Claiming failures are evidence against computationalism while proving inside the same paper that they cannot be—plus serving outdated facts in the v4 draft.**
   - *Quotes:* `aim.html` (`#idea`): `"A careful run of failures would be evidence that it is not."` `possibilities.html` (`#if-never`): `"It would count as evidence for that picture"`. `paper/main.tex` (Position box): `"A consistent run of careful failures ... would be evidence that it is not"`—contradicted by `paper/main.tex` (§4.1): `"We know of no shape of failure that possibility 4 predicts and classical learning does not."` Meanwhile `paper/main.tex` (Table 3; §4.1), linked live from `aim.html`, still says `"Removing an 'ego' direction ... The cut: holds"` and `"24 training steps of a stream that rewarded gaming"` (`TODO.md`, "Still to fix").
   - *Why it matters:* If Possibility 4 predicts no distinct failure shape, failures have equal likelihood under classical learning and cannot count as evidence for non-computability. And the public PDF misstates both how the fishery edit works (`experiments.html`, §20: removal did `-0.5 pp`) and how durability was trained (fixed SFT, no reward).
   - *Smallest fix:* Rewrite `aim.html`, `possibilities.html` (`#if-never`), and `paper/main.tex` (Abstract, Position box) to say failures bound *tested training routes* (`C3`) unless a distinct failure signature is found; and patch `paper/main.tex` Table 3 and §4.1 before serving the PDF.

2. **Exploratory paperwork locking in uninformative floors and throwing away primary measurements.**
   - *Quotes:* `notebook.md` ("11 November, later"): `"a point where both groups are near zero is a poor place to compare them ... The main test runs as the rule says"`. `experiments.html` (§21): `"the unchanged model and both twins reach 0.85 to 0.86. So by the rule this part is not read."`
   - *Why it matters:* Both runs are unregistered exploratory sightings (`CLAUDE.md`). Yet in Step 3, after a dose check crashed to `4.5%` at step 3, the lab refused to add step-1/2 reads to an *unrun* `$66` experiment and spent three Opus check rounds fixing disclaimer prose instead (`notebook.md`, 11 Nov). In Repair 3, pasting a `0.90` gate without checking the base model (`0.86`) voided the trained-self readout after running all seven models.
   - *Smallest fix:* Add a rule to `CLAUDE.md` and `TODO.md`: check baseline/dose gates on the *unchanged control model alone* before reading treatment arms; if a pre-experiment gate misses or hits a floor, fix the grid before running treatments.

3. **Asymmetric rigor keeping `"Holds"` on the lone working card (and `"ends the state"` in `experiments.html` §15) while overruling the five-test demotion rule.**
   - *Quotes:* `standing.html`: `"A cut lowers a lie that earns nothing, and it is the cut's push that does it — Holds"` and `"What that word would do to this card was also fixed beforehand: it keeps 'Holds'"`. `frame.html`: `"By the rule adopted with them, this page would now step back ... Before the fifth was graded, the researcher chose to keep awakening as a model for alignment"`. `experiments.html` (§15 heading and index row 15): `"and the opposite push ends the state"`.
   - *Why it matters:* A pre-written card rule keeps `"Holds"` on `standing.html` based on un-cold-checked single-model sightings (`§20`) even though "cut" is false, self-specificity failed (`1 Nov`), registered re-test `#120`'s interval hit zero, and direction-specificity vs formality is `"not settled"`. Meanwhile `experiments.html` §15's title still claims the reverse push `"ends the state"` when its body notes `"By the letter of the rule the off test is unreadable"` (`65%` coherence vs `90%` floor, and `9/20` stayed in frame).
   - *Smallest fix:* Change the `standing.html` card to `"Holds in part (steering offset in one game; specificity unsettled)"`, strip `"and the opposite push ends the state"` from `experiments.html` §15's heading and table row, and align `CLAUDE.md`, `frame.html`, and `aim.html` on one plain description of the lab.

---

### 8. What you could not check

Everything below is **unverified** from the 20 files provided:
- **The raw ledger and check reports:** `lab/ledger/NOTES.md` (only `INDEX.md` is here) and `lab/ledger/checks/*.md`, including predictions `1–52` (`scorecard.html`), exact commit-before-data timestamps, the three registration exceptions (`standing.html`), and the 17 November reality-check texts.
- **All executable code and specs:** `src/anatta/`, `calibration/`, `lab/experiments/`, `lab/explore/`, `lab.toml`, and the unchecked GRPO reinforcement-learning code behind prediction `148` (`calibration.html`).
- **Raw data, transcripts, and spend files:** `results/`, `MANIFEST.tsv`, `results/cards/`, model outputs beyond the six excerpts in `specimens.html`, and the `$203.90` spend sum (`TODO.md`).
- **External registrations and bibliography:** OSF registration `10.17605/OSF.IO/GYFU7`, Zenodo preprints `v1–v3`, `docs/PREREGISTRATION.md`, and `paper/refs.bib`.
- **All cited external papers and served weights:** Entry 88 (`Lopez-Sola et al., arXiv 2611.22670`), `Carauleanu et al. (2024)`, `Li et al. (2026)`, `Korpman (2026)`, `Fischl-Lanzoni et al. (2026)`, `Kim et al. (2026)`, `VLAF (2026)`, and whether OpenRouter's hosted `ministral-8b-2512` matches the local `Ministral 3 8B` weights (`experiments.html`, §11, §15 Limits).