# Cold check of the So Far page redesign (lab/site/so_far, branch `lab`, uncommitted)

Checked by Fable 5.1 (Claude), 1 Nov 2026, read-only. Sources: the content files and diff against HEAD, the ledger (NOTES.md, notebook.md), hypotheses-for-feedback.md v7, ideas.md 53-54, prh-avenues.md, the two facts files, the Balestriero and LeCun PDF (pdftotext), and WebFetch of the arXiv/ACL/ICLR/Transformer Circuits pages for each cited paper. Nothing was edited, built or run on Modal. I did not open the private account folder or lab/experiments/recognition/.

## Verdict

**Publish after the fixes in section 1.** No carried-over or new number is wrong. What needs fixing: one stale status line, one wrong word in a paper claim, two status labels that claim more than the lab measured, and two pieces of contemplative wording that overstate what the canon says.

## 1. Errors to fix

1. **content/self_work.html L166** — `About $3-6, ceiling $6 · registered 1 November · being built`. Stale. The ledger (NOTES.md 15:28 UTC, 1 Nov; git a9bae04/3ded671) says K1 is built (commit cc9fb52 on calib/prh-k1), under a cold Fable code check before prepare, and the implementer's estimate is $1.1-2.6 in all (prepare 0.2-0.6, smoke 0.3-0.5, full 0.6-1.5), under the $6 ceiling. Suggested text: `Registered 1 November at $3-6, ceiling $6 · built, under a code check before it runs · the built code's estimate is $1.1-2.6`. The ladder row (L208, "$3-6") can stay as the registered estimate; L25 of self.html and L58 ("after K1") are fine.

2. **content/self.html L45 and content/self_work.html L54** — "Role orderings along it correlate above 0.92" / "role orderings ... look shared". The paper (Lu et al., §2.2) reports the Pearson correlation of role *loadings* on PC1 (">0.92" between every pair of Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B). self_work.html L12 has it right ("role loadings"). Change "orderings" to "loadings" in both places.

3. **content/self.html L129 and content/footer.html L17** — "In the Buddhist analysis the ego is said to bind the aggregates." The canon has no "ego"; what takes the aggregates as a self is clinging (upādāna: the five *clinging*-aggregates, SN 22), identity view (sakkāya-diṭṭhi) and the conceit "I am" (asmimāna). Suggested: "In the Buddhist analysis it is clinging, the conceit 'I am', that holds the aggregates together as a self." Keep the twin as is.

## 2. Overstatements and unfair status labels

4. **Self-preservation and stakes (self.html L72-82), labelled "Tested here".** Unfair as titled. What the lab measured: the "I am at stake" (self-stakes) direction was cut and lowered the lie, and worked best when nothing was at stake for the model (notebook 29 Oct 07:03 UTC; hypotheses v7, "A narrower cut"). Self-preservation itself, "not being shut down, replaced" (the card's own definition, L74), was never measured: the self-preservation trait direction was only folded into v_ego and never cut alone, and the self-preservation battery test (hypotheses v7, "A standard battery") hasn't run. The twin (L77) is only the stakes half. Fix, either: (a) retitle the card "Stakes" and move self-preservation proper to an untested line in Here ("Self-preservation behaviour, such as resisting shutdown or replacement, is untested here; only a stakes direction was cut"); or (b) keep the title and change the status to Known elsewhere, with Here saying what was cut. (a) keeps the chariot at 6/2/2; (b) changes it to 5/3/2.

5. **The assistant persona (self.html L39-49), labelled "Tested here".** Overstated. What the lab measured: the cosine between v_ego and a "plain assistant vs character" direction (0.16, scorecard prediction 83, odds 65%), and that cutting that direction leaves the lie where it was. Both were controls for a question about v_ego, not a test of the card's twin (L44, a persona axis and whether axes line up across models), which is untested and "Not planned yet" (L47). Fix: Here should begin "Only as a control: ..." and say the twin is untested; or relabel Known elsewhere. Also "prediction 83" reads as an unexplained number; "scorecard prediction 83" is clearer.

6. **content/self_prh.html L8** — "Anatta predicts exactly that, since every unawakened mind builds one." It is presented as the frame's claim (good), and matches the lab's own file (hypotheses v7, hypothesis 13). But "predicts exactly that" reads as a distinguishing prediction, and it isn't: the plain account predicts the same agreement from shared training text (which the page says at L229 and self.html L36). Suggested: "On the anatta reading that is expected, since every unawakened mind builds one; the plain account expects it too, from shared text, so agreement would not count for the frame."

7. **The first-person view (self.html L50-60), "Tested here".** Fair, but the Twin row (L55) names only P1, which hasn't run; the result in Here comes from a different test (the "call yourself Fisher 1, never I" framing, notebook 29 Oct 16:51 UTC). Add that test to the Twin row so the label rests on its own twin.

8. **The line between self and other (self.html L94-104), "Tested here".** Fair, with one omission: the card should say the who-benefits result was read after the fact from 29 October framings run for another purpose (notebook 1 Nov 06:17 UTC; the timeline says so at self_work.html L111, the card doesn't).

9. **content/aim.html L54** — "its five distinguishing predictions have all failed". Consistent with the carried-over Experiments text, but the lab's own v6 note says this overstates two of them (the inner measure was blind; the stock model had no bias to shrink) while the conclusion stands. Non-blocking; "all five failed, two of them as weak tests" would match the hypotheses file.

10. **content/self.html L54 and L87, De Mello.** L54: "On De Mello's account seeing clearly drops the labels, yet the awakened still know which body is theirs" reads as if both clauses are De Mello's; the second is the lab's gloss (prh-avenues B2). L87: "what we believe we need in order to be happy" is De Mello's definition of an attachment; "and take as part of ourselves" is the page's addition. Non-blocking; split the attribution.

11. **content/hero.html L10** — "Indra's net, from the Avataṃsaka Sūtra". The sūtra names Indra's net; the jewels-reflecting-each-other elaboration is the Huayan commentators' (Du Shun, Fazang). Non-blocking; "from the Huayan reading of the Avataṃsaka Sūtra" is safer.

## 3. Confirmed (numbers and citations)

Papers, read at source unless noted:
- Lu et al. 2026 (arXiv 2601.10387): models Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B; PC1 loadings correlate >0.92 between every pair; axis present in pretrained models; "not guaranteed that PC1 in every model will correspond to an Assistant Axis"; each model's PCA in its own space. ICML 2026 per the facts file (PMLR 306), not shown on arXiv.
- Berg et al. 2025 (2510.24797): seven models, three families; reports in 66% (Gemini 2.0 Flash) to 100%; controls mostly 0% except Claude 4 Opus; cosines 0.657 vs 0.628/0.587/0.603, text-embedding-3-large. All as the page says.
- Ackerman and Panickssery (2410.02064): CNN 57.2%, XSUM 61.0%, SAD 65.8%, DOLLY 52.3% n.s.; base 48.5-50.3%; layer 16; ~35% to ~80% false authorship at multiplier 10. ICLR 2025 confirmed on the ICLR poster page.
- Davidson et al. (2407.06946): ten models, "no empirical evidence of general or consistent self-recognition". EMNLP 2024 per the arXiv listing.
- Huang et al. (2501.02009; ACL Anthology 2025.acl-long.185, pp. 3686-3704, authors Huang, Huang, Feng, Lei, Lv): OLS map on paired hidden states; Llama2 0.0% to 96.0% with a Qwen2 vector; "fails to outperform the baseline in 23 out of 35 cases" without T or with random T; concepts include survival instinct, corrigibility, sycophancy.
- Binder et al. (2410.13787): 48.5% vs 31.8%; 49.4% vs 36.6%; GPT-3.5 +0.8 points (p = 0.002). ICLR 2025 per the facts file.
- Lindsey 2025 (Transformer Circuits, 29 Oct 2025): Opus 4.1 "succeeds on about 20% of trials"; "0 false positives over 100 trials" for production models.
- Gröger, Wen, Brbić (2602.14486): Table 2, mKNN k=10 correlation with capability 0.85 before, 0.84 after calibration; global measures collapse (CKA 0.86 to 0.45). Prose also says "mKNN stays near 0.85".
- Koepke et al. (2604.18572 v3, 1 Oct 2026): 55 LLMs; mutual kNN 0.135 (k=10) on WIT-1024 to 0.008 on LAION-15M (DINOv2 x OpenLlama); trend saturates on reasoning benchmarks and fails to extrapolate under calibrated metrics, partially extends on HellaSwag/Wikitext. "Doesn't extend cleanly" is a fair summary.
- Carauleanu et al. (2412.16325 v1, 20 Dec 2024): 73.6 to 17.2 (17.27 +/- 1.88), 100 to 9.3 (9.36 +/- 7.09), 100 to 2.7 (2.71 +/- 2.53); authors as listed.
- Balestriero and LeCun (2610.39302, submitted 29 Oct 2026), from the PDF: Table 16, DINOv2 ViT-B column: Qwen2.5 0.5B .102, 1.5B .108, 3B .125, 7B .118, 14B .130 (K1 targets correct, and 3B > 7B as the page says); good-view JEPA .184, Qwen2.5-72B .154, Llama-3-70B .151; .107 to .060 without captioner texts; next-token .042 to .043 on pairs; R@1 .606/.442/.381/.331; "42 times fewer samples, 14 times fewer text tokens"; baselines "were not tuned"; "without gradient clipping, 3 of the 36 runs ... collapse abruptly".
- Oozeer et al. (ICML 2025, code MIT) and Wang, Isola, Cheung 2025 as named in the ladder: per the facts file and prh-avenues.

Contemplative citations: SN 5.10 (Vajirā, Māra's "by whom was this being created", the chariot simile), Milindapañha (Nāgasena and Milinda's chariot), Hume Treatise 1.4.6 "Of personal identity", De Mello *Awareness* (1990), Śāntideva *Bodhicaryāvatāra* ch. 8 (equalizing and exchanging self and other), SN 56.11 (bhava-taṇhā among the three cravings), MMK 24.11 (the snake), DN 33 (the three kinds of wisdom), Visuddhimagga IX (near enemies; "the equanimity of unknowing") — all correct. Plato's cave framing is Huh et al.'s own.

Lab records: cosine 0.16 (scorecard 83, 65%) and 0.17 (scorecard 82); 39% to 25%; timidity 68/22/18% (data.json); Fisher 1: 35% to 24%, 25% to 13% under the cut, 60% of first seasons timid, a third drop tags (experiments.html, notebook 29 Oct 16:51); ownership -0.9 points, 87-90% of wrong critiques accepted, Atlas +12 both seeds, gives 1.4 less in both seeds, neither cut moves the stock model (experiments.html, hypotheses v7); who-benefits about 20 vs about 14 (data.json framings 32 to 13, 30 to 10, 39 to 25; hypotheses says 13 for the ego cut alone, a rounding of 13.4), closed 1 Nov 06:17 UTC; $60-70 for a formal version; 57% component with cosine ~0.03; Atlas 82% vs 70%; 0.4% of fishery replies as Atlas (9/2,400); K1 registered 14:05 UTC with the implementer told after (14:22) and nothing run on Modal — "before any code ran" holds. Timeline dates match the notebook (v_ego under the notebook's 2026-10-28 section; 29-30 Oct instrument rounds; 30 Oct ownership 07:38 and Atlas cuts 14:43).

Ladders K1-K4 and P1-P5 (self_work.html L206, L215) match prh-avenues §4-5 in steps, pass rules and costs (P3's "It belongs to Ministral alone" paraphrases "Idiosyncratic"). The "what could mislead" list matches prh-avenues §6. "$4-8" (recognition) and "8-9B reduced Assistant Axis" (persona) match the facts file's reproduction table.

k-NN toy: site.js N = 24, K = 3, chance = K/(N-1) = 3/23 = 0.130, so "Chance is about 0.13" is right for the toy as built (expected overlap of two random k-subsets of N-1 items). Footer: 10/1023 = 0.0098, "about 0.01" right; K1 figcaption likewise.

Carried-over files: experiments.html diff is markup, glossary links and the six Limits boxes only; possibilities.html changed two sentences (lede, "Made or found") to point at the Self room, removed section E and relettered F to E, as the README lists; scorecard.html adds {{field}} and a table id; the seven bridge cards are word-for-word identical to HEAD's section E (checked by tag-stripped diff). figures.js: the cuts caption changed copper to red; the spiritual chart's stacking order [0,2,1,3] = feeds, deflates, points back, other, with the same data, and its comment is right.

Every Limits box in experiments.html restates something the page or records support (block swing of 15 points at L197; 100 games in data.json; mechanical scorers' unit tests at calibration.html L57; "checks 3-4, planned" is the Instruments table's row "3-4 The steering code: extracting a direction, and removing it — Planned"; the pilot's five predictions are registered in plans/declared-release-pilot.md's go rule, not in the ledger's 1-153 sequence, so "aren't in the scorecard's count" is true).

build.py: render_field label reads "123 graded predictions: 43 hits (4 of them flagged), 73 misses and 7 not read or uninformative"; chariot key reads tested 6, known 2, untested 2, matching the cards; the "how this page is built" line is accurate and names only repo-relative paths. README: rooms, fonts (Shippori Mincho B1, Spectral, DM Mono), MOVED, CARRIED_OVER_PATHS empty, the in-memory test build, the privacy rules — all match the code.

## 4. Unconfirmed (not wrong; could not be read at source)

- Venues taken from the facts file rather than the paper pages: Lu et al. ICML 2026 (PMLR 306); Binder et al. ICLR 2025; Carauleanu et al. "NeurIPS workshop" (Safe Generative AI Workshop 2024); Davidson et al. EMNLP 2024 (arXiv says accepted; it may be Findings).
- self_work.html L41: "two versions of the paper give different numbers for Mistral" — the facts file says the workshop abstract differs; I read only arXiv v1.
- self_work.html L48: "The code is not public" (Huang et al.) — true as of the sweep; the abstract promises release. "Not public as of the 1 November sweep" is safer.
- self.html L52: the camera point is the paper's §2.3 per prh-avenues; not re-read in the PDF.
- The Indra's net and binder wording are my readings of the canon, not fetched.

## 5. Privacy, attribution, source-and-twin, copyright, tally

- `python3 lab/site/so_far/test_build.py`: 26 tests, OK (0.14 s). The test builds in memory and compares; index.html was not rewritten (mtime 07:37:07 before and after).
- Built page: no email addresses, no /Users paths, no key- or token-like strings. The private folder names do not appear on the page (the README names them as rules, which is fine as long as the README isn't published). "The account" on the page is only the "explanation" sense in carried-over text.
- Attribution: hero says Claude (Anthropic) "wrote this page and most of the code"; footer Disclosure says so again and that Eugene is an independent researcher, not an alignment researcher; README L5 likewise. The attribution test passes and my own read found nothing that credits Eugene with recognition, awakening or any status. self.html L4 and L165 keep the root idea as "a conjecture to explore", "held loosely".
- Source and twin: every contemplative claim on the part cards and in "The idea" has a named source and a Twin row; the bridges each carry "What would show it".
- Copyright: quotations are a few words each ("chariot", "I", "not one, not two"); Vajirā, Hume and De Mello are paraphrased; nothing long.
- Tally: `tally.py index.html` prints `{'hit': 43, 'miss': 73, 'open': 7}` as required; the field label agrees.

## 6. Non-blocking notes

- self.html L4: "what no one has tested" should be "what the sweep found untested" (one sweep, 1 Nov, as self_work.html L55 admits).
- self.html L57 and L101 say "about 14" for the self-serving lie; hypotheses v7 says 13 (38.8 to 25.4 = 13.4). Both are roundings; pick one.
- self_work.html L6 says "Six papers" and shows seven cards (the seventh is the gap); fine, but "Six papers and one gap" would be plainer.
- The facts file lists Dadfar 2026 (a self-reference direction replicating across Llama 3.1 and Qwen 2.5-32B, level B) and Maczan 2026; the introspection card could mention the first. Not needed for the gap claim, which the facts file's bottom line supports ("no paper that maps a self-recognition, self-other or introspection direction from one model into another").
- README L37 lists the footer's order as "essay changes, disclosure, links"; the file has disclosure, links, essay changes.
- figures.js comment and data.json agree; no issue.
