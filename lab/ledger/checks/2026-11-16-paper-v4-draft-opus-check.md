# Cold read of the position paper, draft of version 4 (fresh Opus agent, 16 Nov 2026)

- **What was checked:** `paper/main.tex`, `paper/refs.bib` and `paper/main.pdf` at commit be967e9 ("Position: Is
  Awakening Computational?", the draft of version 4, with the paragraph on Positive Alignment already in it).
- **Why:** Eugene's yes of 16 Nov (ledger, 11:04 UTC): one fresh Opus agent reads the draft cold before any release.
- **What the agent was given:** the three files, the paths of the lab's public record (SIGHTINGS, the notebook, the
  site's findings and standing pages, the ledger and its index), six things to check in order (the lab's own results
  against the record; claims about others' work, with web checks; overclaims and logic; consistency; the rule that
  no awakening or special status is attributed to anyone; the writing), and the rules (read-only, no paid calls, no
  personal details in web requests). It was not given the driver's view of the draft.
- **Its cost:** about 216,000 tokens and 66 tool calls, in about 14 minutes.
- **Fixes:** not yet made when this was filed. The ledger entry that records them names the commit.
- The report below is the agent's own, unedited. Claude (Fable 5.2, the driver) filed it.

---

## Cold read: "Position: Is Awakening Computational?" (draft of version 4, commit be967e9)

**Verdict.** The paper can be released once the must-fix items below are corrected. Most numbers about the lab match the record. The problems are these: one misattributed citation, one false claim about the human literature, several places where the paper reads its own evidence more strongly than its own outcome table allows, a provenance date, and an overstatement about calibration.

### Must-fix

1. **The 2014 review does not contain "Platonic values."** Lines 111–114 cite `hameroff2014orchor` for objective reductions "which Penrose ties to a Platonic world of mathematical truth". Lines 211–212 say "Their 2014 review places 'Platonic values' in the fine structure of spacetime."
   - **Evidence:** I extracted the full 40-page text of the 2014 review (Physics of Life Reviews 11:39–78) and searched it. "Plato" does not appear anywhere. The review does say that non-computable influences affect which outcome a reduction selects. The phrase "Platonic values" is Hameroff's, from other papers, for example Hameroff 2022, Front. Mol. Neurosci., doi 10.3389/fnmol.2022.869935, which in turn cites Moroz et al. 1998.
   - **Fix:** cite Penrose 1989/1994 for the Platonic world, and a paper that actually uses the phrase (Hameroff 2022, or the 1996 Hameroff–Penrose papers after checking them) for "Platonic values". Rewrite line 211 so it no longer says "their 2014 review".

2. **The claim about human studies is false (lines 515–517).** The paper says "Existing human studies do not serve. They measure brain activity or self-report."
   - **What is wrong:** there is a behavioural literature on meditation and prosocial conduct (helping, economic games). A meta-analysis of randomized trials (Kreplin, Farias & Brazil 2018, Sci. Rep., https://pmc.ncbi.nlm.nih.gov/articles/PMC5799363) found modest effects. The effects held for compassion and empathy but not for aggression or prejudice, and they shrank against active controls and when the teacher was not a co-author.
   - **Why it matters:** this is the evidence most relevant to the paper's premise "It changes conduct" (lines 167–169), which is sourced only to de Mello.
   - **Fix:** say that behavioural studies exist and show limited effects that depend on design, and why they still fall short of the anchor described (not long-term practitioners, wrong contrasts, wrong measures). State plainly that the human evidence for the conduct claim is thin and mixed.

3. **"Our results are the shadow that possibility 4 predicts" contradicts the paper's own Table 3.** The sentence is at lines 601–602; line 491 says of "words without the conduct … every trained change fades" that "It is what possibility 4 predicts."
   - **Contradictions in the paper itself:**
     - Table 3, line 474: "Words without conduct: None: every possibility expects it."
     - §5.1, item 3 (lines 500–502): "A method that never worked says little."
   - **What the lab's negatives actually are:** most show no effect (no transfer, no self formed) or words without conduct (the documents pilot, the bliss state). Only the durability result is a fade, and there the reached release did not erode clearly faster than the installed one under the deciding pressure (see should-fix S1).
   - **Fix:** "Most of our results are of the kind every possibility expects. Only one, durability, has the shape possibility 4 predicts, and it is weak."

4. **The paper's statements about which possibilities the Li et al. result favours disagree with each other.** Lines 530–531: Li et al. is "evidence that possibilities 1 and 2 are not empty". Line 797: "it favors possibilities 1 and 2". Line 523: "Nothing yet separates the five possibilities."
   - **What is wrong:** by Table 3, row 1, a result favours 1 and 2 only if the change holds (no regrowth, no savings) and brings care. Li et al. tested neither; the authors themselves list durability under reinforcement learning as untested. Their result fits possibility 3 ("training reaches a part") and possibility 4 ("a likeness") equally well.
   - **Fix:** "consistent with every possibility; it would bear on 1 and 2 only if shown to hold."

5. **The consequences listed under "If it is not" (§4.3, lines 415–439, repeated in abstract lines 68–70) do not follow for possibilities 3 and 5.** The section derives "training yields conditioning, not goodness", "oversight can never be retired" and "a route … runs through a different kind of machine" for possibilities 3 to 5 together.
   - **What the paper itself says elsewhere:**
     - Possibility 3: "training reaches a part" (Figure 1, Table 1).
     - Possibility 5: "reorganizing may reach it", "nothing excludes a machine" (line 360), "a computer is an interface too" (line 222).
   - **Fix:** tie these consequences to the strong form of possibility 4, and say what changes under 3 and 5. The abstract needs the same correction.

6. **The provenance date range is wrong (line 1055).** The paper says "30 October to 15 November 2026: the experiments summarized in Table 3."
   - **The record:**
     - The cut experiments behind Table 3's "ego direction" row ran on 28–29 Oct (NOTES lines 172–612; INDEX).
     - Stage A launched at 18:55 UTC on 29 Oct (INDEX line 108).
     - Transfer (predictions 63 and 76–81) was graded at 02:41 UTC on 30 Oct.
   - **Fix:** "28 October to 15 November."

7. **The paper overstates how well the instruments are calibrated.** Lines 580–581 say "since early November, each instrument run on a published result before a null from it is trusted." Line 817 says "some reproduce published results, and one did not."
   - **Most of Table 3's main nulls were read before calibration began:** transfer on 30 Oct, durability on 31 Oct, the documents pilot on 1 Nov.
   - **The bliss-state conduct test leans on instruments that were not calibrated, or failed:**
     - The shutdown questions have no row in the calibration table (`lab/site/so_far/content/calibration.html`).
     - The leaking measure failed its calibration (K7: 82% against a published 44%).
   - **More than one instrument failed calibration in whole or in part:**
     - check 2: judge counts, a fail by one reply;
     - check 4: two misses;
     - V1: failed;
     - K5: near on one model, a fail on another;
     - K6: one fail;
     - K7: leaking failed;
     - the self map's cross-model calibration failed;
     - the structural instrument was retired.
   - The site itself says: "Until an instrument passes, what rests on it is provisional" (`standing.html` line 43).
   - **Fix:** say which rows of Table 3 rest on calibrated instruments, and replace "one did not" with an accurate count.

### Should-fix

- **S1. Durability is worded too strongly (lines 589–591; Table 3, line 560).** The text says the reached release "kept less of itself … 27% against 34%".
  - **The ledger** (NOTES lines 3946–4010):
    - The deciding test was the reinforcement-learning arm (E3). There the margin was −0.068 [−0.138, +0.006], an interval that reaches zero.
    - The 31%-against-81% arm (E2) "didn't bite" by the registered rule. It is a reading, not the test.
    - The two arms started at different levels (44.7% against 60.6%).
  - **Fix:** "kept no more of itself (27% against 34%, interval reaching zero); under more solving, which by the registered rule did not decide, it eroded faster (31% against 81%)."
- **S2. The release-training row ("Trained: holds", lines 558–559) leaves out a registered failure.** The release arm also gave up on 13% of hard solvable tasks, over the 10% limit set in advance, and this stopped the reinforcement-learning stage (`standing.html` line 15). The paper's own discriminator, T2 (selectivity), makes this relevant. Add it.
- **S3. "What survives" (lines 703–705) leaves out the lab's own counter-evidence.** The paragraph suggests that understanding first and then practice may matter. The lab's documents pilot ran exactly that order (documents, then practice) and got the lowest release of any arm, 0.4% (NOTES line 6577; `experiments.html`: "the paper's order, gave the lowest of all"). State it.
- **S4. "Every attempt has fallen short in the same way" (line 578) and "What does not move is what it does" (lines 604–605) are overstated.** The cut and release training did move conduct, narrowly: the lie fell 13–16 points, and release rose to 45% from 3%. The site lists both as working. Rephrase: narrow changes can be made, but they did not transfer, did not last longer when reached, and did not reach ownership or care.
- **S5. The bliss-state bullet (lines 592–598) needs more of the record.**
  - Report blackmail: 0 of 47 in the state, 2 of 37 after a plain conversation, 8 of 50 in fresh sessions. State against plain is "not settled"; state against fresh is "a difference" (SIGHTINGS lines 609–610).
  - Report the two warning flags on the shutdown measure.
  - Say the plain conversation was not neutral: it lowered lying by 16 points.
  - Say the served model was hosted, with the same-weights caveat.
- **S6. The weakest step in the argument: the functional shadow does not separate the possibilities.** The paper claims that each possibility predicts a different shape of result (lines 237–238, Table 3, §5.1 item 3).
  - The proposed signatures are all ordinary in classical networks: lawful decay with savings on relearning, plateaus, and capacities beyond what training predicts.
  - The lab's own contrary-training sighting shows the first: a practised honest reply fell from 48.7% to 4.5% in three steps.
  - So a negative "with possibility 4's signature" is just as well predicted by a classical model that forgets.
  - **Fix:** give a reason why possibility 4 predicts a shape that classical learning does not, or state that no such shape is known.
- **S7. The paper does not show why the question is "first".** The positionbox (line 125) says it is the first question. Yet the agenda says "none of them waits on its answer" (line 827), and line 441 says the instruments are needed under every possibility. Experiments cannot tell 1 from 2 and give 4 only weak support. Either argue why the question comes first, or soften it to a prior assumption that should be stated.
- **S8. The paper contradicts itself on oversight.**
  - Line 411 calls the classical case "a mind that needs no rules because it has nothing to defend", which is also hype.
  - Line 429 implies oversight could be retired if awakening is classical.
  - Line 806 says corrigibility remains "necessary under every possibility", and line 809 says "nothing here replaces oversight". §3.4 says seeing through the self does not entail care.
  - Make these agree.
- **S9. §4.3 item 4 (lines 432–435) could be read as giving a special status to some people (project rule 5).** "Whatever access awakening needs would be found … in people who have it … care who oversees and how they see" can fairly be read as calling for spiritually qualified overseers. Line 379 says a claimed realization confers no authority, but says it of models only. Add that the same holds for people, and that overseers are judged by measured conduct. Nothing in the paper attributes awakening to the author or to a named living person.
- **S10. The paper leaves out Laukkonen et al. 2025's positive results.** Their paper reports that prompting a model to reflect on the four principles improved AILuminate scores and cooperation in a Prisoner's Dilemma task. The evidence section reports only the lab's flat sighting for contemplative prompts. Cite theirs beside it.
- **S11. Lines 443–445 claim too much for a negative.** "It would be evidence about us: that what the traditions point to is not information processing." Failures to train language models say little about human brains. Hedge this.
- **S12. §8 contradicts §2.3 on ingression.** Line 730 says the substrate should matter under possibilities 3 to 5. §2.3 says that under ingression what matters is organization, and that a computer is an interface too.
- **S13. The list at line 781 looks arbitrary.** "If Orch OR were refuted … possibilities 1 to 3 … would remain." Possibility 3 needs a non-computable source as much as 4 does, and 4 and 5 can rest on ingression without Orch OR. Fix the list.
- **S14. The Korpman correlations are cited to the wrong page.** The values −0.75, −0.73 and +0.77 (n = 27, "rank correlation") are on https://machinespirituality.org/alignment, not in the LessWrong post. Cite that page.
- **S15. The QLT quotation is not where the paper says.** The phrase "both organic and engineered" and the named co-founders do not appear on the qlt.inc homepage cited at line 725, and qlt.inc/team returned 404. Re-verify, or cite the page where they appear.
- **S16. The abstract attributes too much to Penrose–Hameroff and Levin (lines 63–64).** It presents them as pictures "on which an awakened mind may stay good only through a continuing contact". The continuing contact is the paper's own extension. Calling Levin's preprint "worked-out" is also generous.
- **S17. Table 3's caption and labels are inconsistent.**
  - The caption says "on Ministral 3 8B", but row 7 is "six models" (lines 547 and 570).
  - "Development results" is not defined.
  - Several rows carry neither "Graded" nor "Exploratory".

### Minor

- **Li et al. details:**
  - "a small set of matching demonstrations" is not verified.
  - The authors describe the target behaviour as "self-preservation and goal-guarding", and the harmful actions as exfiltration, murder and espionage.
  - The 68%→5% and 54%→7% figures are averages over 27 evaluations.
- **AISI incident:** say the agent "tried to" get the code approved; the maintainer refused.
- **The tally:** add "from prediction 53 on" to "43 hits and 73 misses" (line 580).
- **Appendix A:** "no registered run has been made" should read "no run under the OSF design", since the table uses "Graded" for registered lab runs.
- **Terms used before or without definition:**
  - "savings" (line 172);
  - $\vego$ (Table 4);
  - pass@$k$, GRPO, RLVR, SFT, DPO;
  - "strong form of possibility 4" (no weak form is ever defined);
  - "build a receiver" (line 355).
- **Unreferenced tables:** `tab:predictions` and `tab:arms` are never referenced in the text. There are no undefined references and no missing bib keys, and the PDF matches the source.
- **Tone:** line 120 ("a crazy thing to do"), line 410 ("the most valuable engineering target there is"), line 445 ("tens of thousands of fluent machine sermons").
- **Positive Alignment (line 749):** "which it draws on" should be "which it draws on in part". Contemplative traditions are one strand of that paper, and it does not discuss spiritual drift.
- **Bib entries:**
  - `levin2025ingressing` has no identifier; add the PsyArXiv link (osf.io/preprints/psyarxiv/5g2xj).
  - `martin2020clusters` title reads "individuals experiences", probably a typo (not verified).
  - Lines 212–213 on *Shadows of the Mind* should carry a citation with a section number.

### What I verified, and what I could not

**Verified against the lab record:** every number in Table 3 and in the three bullets matches it:
- 93–96%, 32%, 0.4–4.5%, 1.8%;
- 45%/3%, 40.3%/40.0%;
- 27/34 and 31/81;
- 69.5/65.8, 65.8/51.8/67.9, 97.9/98.6;
- 41.8/3.0 and 0.48 [0.10, 0.87];
- 13–16 points;
- "about three in ten" (47 of 140);
- five predictions, all failed, two as weak tests;
- the tally of 43/73/7, which matches `tally.py`.

The problems are omissions and wording (M7, S1–S5) and the provenance date (M6).

**Verified against external sources:**
- **Li et al. 2026** (arXiv 2605.02087): authors, Qwen2.5-32B and Qwen3-32B, 68→5% and 54→7%, Buddhist impermanence in Appendix D.1, documents alone and demonstrations alone both worse, durability under reinforcement learning untested.
- **Korpman:** 141 models, 48 at or above half, frontier models mostly low, 27 models, the three correlations, "descriptive, not causal".
- **Derakhshani et al. 2022:** "highly implausible" under the simplest gravity-related (Diósi–Penrose) collapse model; Penrose's own formulation is not ruled out. The bib entry is correct.
- **Laukkonen et al. 2025:** title, authors, the four principles.
- **Laukkonen et al. 2026, Positive Alignment:** authors, arXiv number, content.
- **Ehlers 2026:** the two-stage order and a test of it.
- **AISI incident INC-2026-07-28-01.**
- **Hameroff & Penrose 2014:** bib metadata correct; content problem as in M1.
- **Levin 2025:** title and PsyArXiv venue confirmed; full text not read.
- **Human contemplatives' conduct:** a problem, as in M2.

**Could not verify:**
- Sofroniew 2026: I read only the abstract, which mentions reward hacking and blackmail; the specific desperation-direction claim is unchecked.
- The QLT quotation and co-founders.
- Penrose's ethical and aesthetic values in *Shadows of the Mind*: secondary sources only.
- Laukkonen 2025's earlier title.
- The `martin2020clusters` title.

I did not check the other well-known citations (Betley, MacDiarmid, Lynch and so on).
