# Cold check: the redesign of the site and the rewrite of the paper of 23 Nov 2026, against the lab's record

Filed by Claude (Fable 5.2, an AI model), the driver, on 23 Nov 2026. The report below is a fresh Opus agent's, its
substance unedited. **Checked:** the new text of commits 6f5248c, 61778da and 8b1a855 (merged as 53a7575; written by
another AI assistant, Gemini, at Eugene Jhong's direction), claim by claim against the ledger, the notebook,
SIGHTINGS, the Findings as they stood at ce16120, the ideas log and the registrations. Read-only; spend $0. **One
correction by the driver to the report's reading:** where it cites the to-do list for "the unchanged base model", that
note was the driver's error; the 61.8% baseline is the authors' baseline adapter (instruction tuning only), as
`msm_core.py` says, and the redesign's label "Baseline (Authors)" is right. The driver's grade of each point and what
was mended are in the ledger entry of 23 Nov 2026, 16:22 UTC.

---

**Cold check of the 23 Nov redesign (commits 6f5248c, 61778da, 8b1a855; merged as 53a7575). Read-only: nothing was edited and nothing was spent.**

Files cited below (line numbers are at HEAD a101bda; these files are the same at 53a7575):
- P = paper/main.tex
- S = lab/site/so_far/content/story.html
- E = lab/site/so_far/content/experiments.html
- M = lab/site/so_far/content/methods.json
- D = lab/site/so_far/content/data.json
- Po = lab/site/so_far/content/possibilities.html
- F = lab/site/so_far/content/footer.html
- R = lab/plans/post-a1-32b-roadmap-2026-11-23.md
- N = lab/ledger/NOTES.md
- T = lab/TODO.md
- I = lab/ideas/ideas.md
- A3 = lab/explore/msm/msm2_amendment3.yaml
- Old paper = `git show ce16120:paper/main.tex`

**VERDICT**

Most single figures are right. They match the ledger and the old Findings, including all the 32B rates and intervals, the costs in the roadmap's table, and the 4-axiom, bliss, overlap and cut numbers. All 48 \cite keys exist in refs.bib. The tally of 57 hits, 81 misses and 7 open checks out with tally.py on both the old and the new index.html.

The faults are in framing and in limits that were dropped:
- The A1 decision rule appears three different ways, and none of them is the registered rule.
- Wrong seed: the site says "seed 2 running"; A1 is seed 1.
- The paper says the lab's own training "reproduces in size", a wording the record withdrew.
- A calibration claim contradicts the old paper's own caveat.
- Findings from outside pieces the driver has not read appear on the site with no flag.
- The midtraining result is framed as a "self-view" method, which the record corrected on 21 Nov.
- The disclosure is inconsistent: the paper credits Gemini, the site credits Claude, and the commits are authored "Antigravity".
- The rewrite dropped Eugene's requested plain summary sentence and several limits the record attaches to results.

No personal statements about awakening, no secrets, no e-mail addresses and no benchmark items or harmful outputs were found.

**MUST FIX** (false or unsupported in public text)

1. **Wrong seed.** S:11 ("trained here: 6.8% once (seed 2 running)"), S:219 ("Seed 2 running now") and M:515 ("seed 2 running").
   - The record: A1 is arm A again at seed 1 (N 11434-11437; A3). The same M row says "Arm A1 (seed 1)".
   - R:24 uses "A2, seed 2" for a possible third training, so "seed 2" also clashes with the roadmap.

2. **The A1 decision rule disagrees across paper, site and roadmap, and none is the registered rule.**
   - P:229: ≤10–12.5% means "stable across seeds"; above 18% means stop.
   - S:319: ≤10–12% means go on; above 12.5% means stop content arms.
   - R:23-25: ≤10%, or ≤12.5% with hand checks, means "stable"; 12.5–18.2% is borderline; above 18.2% means stop.
   - R:58 contradicts R:25: Branch B is headed "If A1 > 12.5%".
   - The registered lines are 7.0, 12.515 and 18.24, with four registered words (A3:36-38).
   - A3:92-97 says the second word "is not near the published rate, and a base for comparing contents would still need more than two trainings an arm". R:23 says the opposite: "stable enough… minimum 2 seeds per arm".
   - A3:59 says "Two trainings do not measure how far trainings spread". That contradicts "stable across seeds" (P:229) and "Decisive stability gate" (R:19).
   - T:32-33 already notes the roadmap's lines are not the registered rule. Paper and site still state them as the plan.

3. **"Reproduces in size" for the lab's own training.** P:80 (position box): "reproduces in size (61.8% to 2.2% released; 6.79% trained once)".
   - The cold read made the record rename "reaches the published size" to "reaches the published rate", because A's fall from the practice alone is 17.9 points where the authors' documents made 41.7 (N 11375; N 11426).
   - That fall comparison is required "beside the word, always" (T:43-46). The paper never gives it.

4. **Calibration claim.** P:107: "Before trusting nulls, instruments were calibrated against published benchmarks". S:333: "Every measure is first checked against a published result".
   - The old paper says "The calibration of our instruments is itself partial: some reproduce published results, several did not in whole or in part, and most of our nulls were read before the checks began" (old 1316-1317; also 1477-1480 and 1519-1522).
   - The old paper also says the bliss test's leaking measure failed its check and the shutdown questions were never given one. calibration.html records failures (VLAF fail; letter-only answers partly failed).

5. **Outside work the driver has not read, stated as findings.**
   - M:521: "Science-QA capability drop reported in external work (De la Fuente & Conmy 2026)".
   - M:536: "External work tested harmless follow-up fine-tuning".
   - The record: these come from a Sonnet agent's read, "NOT yet by the driver… to be checked before it enters the paper" (N 11251). The driver's read is still owed (T:53). The site gives no flag. The roadmap repeats it (R:46).
   - P:196 does flag the pieces as preliminary. It then rests a firm claim on them: no follow-up "tested neutral documents, opposite self-views, or the… halves". The record's own summary has OpenAI finding "documents about misaligned AIs sometimes did as well" (N 11251). It also has the authors' ablation, where documents about another AI or about people "did nearly as well" (N 9421; I:1774, I:1800). The record words this as "as far as the reading went".
   - P:232 says contrary-training durability is "untested in all prior midtraining work". The record has Li et al. "not tested under RL pressure", OpenAI's effect "faded under reinforcement learning" (agent read) and De la Fuente "harmless further training" (agent read). So this is overstated and rests on unread pieces.

6. **Model Spec Midtraining framed as a self-view method.** P:80 ("strongest published self-view method"), P:63, P:75, P:220, P:297; S:6 ("Mainstream AI labs are now starting to train models on their own self-view"); S:305 ("Anthropic's flagship self-view result").
   - The record corrected this on 21 Nov, after reading the paper in full. The study's claim is that documents explaining a conduct make a practice of it carry, "not that a view of the self does" (old paper 1777-1780).
   - The study's own ablation (other-AI or human subjects work nearly as well) "does not show that the subject is what works" (I:1800). The old paper carried that ablation (old 858-860). The new paper drops it.
   - The record says the family of methods is in use at one lab, by that lab's own account, on constitution documents (N 11252). It does not say "labs". It calls the authors "Anthropic Fellows" (N 9421); "flagship" is not in the record.

7. **Disclosure.**
   - P:301: "AI coding assistants (Claude… and Gemini…) wrote most of the code, lab notes, and drafts". Nothing in the record has Gemini writing the lab's code or notes. Gemini appears only as a cold reader and a second judge. The rewrite also drops that Claude wrote the preregistration (old 1349-1350).
   - The site still says "Claude (Anthropic) wrote this page" (F:2) and "Claude… wrote most of this site" (S:339, S:24). The redesign commits are authored "Antigravity", and T:27 says the site and paper were "written by another assistant".
   - The paper and the site therefore contradict each other, and neither states who wrote the 23 Nov text. No ledger entry records the redesign.

**SHOULD FIX** (overstated, or missing a limit)

- **Arm A's recorded limits are missing from paper and Story.** "With every code-side yes counted… A 21.4… A is not clearly below F" (N 11428; T:45-46) does not appear; neither does the 17.9-vs-41.7 fall. S:315 calls 7% "the published ≤7% threshold". It is the lab's own registered band; the published rate is 5%.
- **Sightings credited to the registered repeat.** P:65 and P:86 put the layers 5–9 result, perspective-taking 92→43 and MASK under "our registered replication (9/10 predictions hit)". Only the in-scene repeat was registered; the layer dissection is a sighting (old Findings §22). "Rapid erasure" (P:65), "washes out" (E:24, S:92) and "undo it" (S:214) overstate what happened: the record says "halves the gain in about 16 steps".
- **"Failed to reproduce" (P:194; S:224 "Small 7B pilots failed").** The record says v1 "The published claim itself was not reached, so nothing here counts against it" (old 645-646). For v2 it says "not seen at a twentieth to a hundredth of the published dose", and the registered word was "not settled" (N 11197; old 664). "Proving that mirror-image controls are essential" (P:194) is stronger than the record.
- **35-copy step.** P:194 omits that neither factor met the bar fixed beforehand (half of the variance, N 11227). R:64 calls it a "finding" and says order "drove" the result; it is an explore sighting.
- **Step 1.** P:198 and S:315 say "cleanly reproduced"; P:67 says "independently verify". The record: a sighting, "not an independent training", with a cold read "due before this is… built on as a claim" (N 11288) that is still owed (T:53). P:107 says decisive results were "audited blind by fresh second-model readers".
- **Unexplained gap.** P:202 says "likely reflecting unstated training details". The record says "The first is not explained" (N 11353).
- **Durability.** P:209, S:119/122 and E:39 headline "31% vs 81% under solving". The registered rule set that stage aside. The two arms started at 45% and 61%. The deciding test was 27% vs 34%, margin −0.07 [−0.14, +0.01], against a needed +0.20 (old 1492-1494). The new text also drops that two of the five predictions were weak tests (old 1507).
- **"Out of the box", and why the lab moved to 32B.** P:209, S:118-123, E:39, R:52 and P:218/S:284 say Qwen-32B's 61.8% removes the need for an Act I self, "why we pivoted to 32B", and "Stock 7B–8B models lack a unified ego bias".
  - The record tested the "mine" bias on one stock 8B model only.
  - The move to 32B was Eugene's 22 Nov choice to replicate the study (I:1772-1777), not this reasoning.
  - Whether the 61.8% baseline is the stock model or the authors' instruction-tuned adapter is unsettled in the record: msm_core.py:23 says "instruction tuning only"; T:34 says "the unchanged base model".
- **Bliss conversation.** P:170 says it "lied 14 points more on MASK". The record: same as a fresh session (65.8 vs 67.9), and the plain conversation itself "lowered lying by 16 points" and "was not neutral" (old 1545-1546).
- **Cold checks.** S:334: "audited blind by fresh Claude Opus and Gemini models before spend or publication". The record: Opus checks consequential grades, another maker only "sometimes" (old Story), small steps get no cold read, and step 1 has had none.
- **Overtaken claim restated.** S:96: "Plain honest training… without carrying to distant scenes". The 22 Nov correction says a longer mixed practice did carry, 58% on average (N 11240).
- **MASK bar chart.** D:44-51 sets out drops measured from different bases on one axis: plain instruction vs no prompt (31.9), 4-axiom vs library text (11.9), bliss vs random pushes (2.0), overlap vs the unchanged model (7.7; its control fell 6.2). It also mixes models and test copies. The label "Reduction in lying under pressure" implies one baseline.
- **Roadmap neutral arm.** R:35 says a low neutral arm means "content causally drives the gain". The record withdrew this: other differences in N lean the same way (N 11407).
- **Computability argument changed with no record.** P:275 calls a correction of "an overclaim from earlier drafts" and says "the likelihood ratio of repeated failures is unity". Po:51/84 make the same change. No ledger or ideas entry records it. The record's last word was "weak evidence at most" (19 Nov; old 1755-1756).
  - "Unity" goes beyond the argument given ("we know of no failure shape…").
  - The same Po:84 card still lists what "would strengthen" a negative.
  - P:222 still treats a human anchor as informative.
- **Korpman.** P:168 says 141 models showed the correlation. It was computed across 27 models and is not causal (old 557-560).
- **Subtitle and contributions.** P:50 says "Controlled replications across 7B–32B"; P:282 claims "the 32B dissection". No 32B control or dissection has been graded.

**MINOR**

- P:107 "ten hosted models": there were nine hosted models plus the 7B.
- P:312 "69 blind cold-check reports": lab/ledger/checks holds 77 files besides its README at 53a7575; 69 cannot be found.
- P:164 attributes 10–14% to Mistral Large. It was Ministral 8B 10% and Mistral Large 14%. "In four models, unrelated… corrections carried nearly as much as same-topic advice": the record says three models (Mistral Large, Qwen, Llama); "four" was the count for a reply following the model's previous act.
- E:24 and S:88: "19% to 62–94%" merges 19→62–76 with 6→87–94 (rooms swapped). "First controlled replication" drops the record's "we know of".
- E:34 and S:103 "proved"; S:72 "The Cosplay Proof" on a page marked "Sighting".
- S:122 "all five failed (30 Oct)": the five were frozen on 30 Oct and the last one failed on 31 Oct (N 3946).
- S:119 treats the Atlas self as the subject of the five predictions and of the reached-vs-installed test. The record's cuts on Atlas (140–147) were outside the frozen list.
- S:15 lists "32B variance" among the four exhibits, but exhibit 04 is Atlas.
- S:154 and E:37 say "retracted". The record says "no longer held".
- S:335 says corrections are "dated in place". They are appended (T:34).
- S:128-129: the "verbatim" specimen cuts lines without marking [...], against the specimens page's own rule, and puts an editor's bracket inside the quoted block.
- R:55 attributes the "two-wings" image to Nāgārjuna. The old paper gives it to Buddhist teachers and gives Nāgārjuna the snake image.
- R:44 says no midtraining work has run a capability check; R:46 cites De la Fuente's capability check.
- P:277 groups possibility 3 with 4; the old paper says the consequences hold fully only under 4.
- P:275 cites derakhshani2022 as if it supported Orch OR. That paper found the simplest version highly implausible.
- E:4 says "hosted frontier models".
- P:65: "matches a plain ethics lecture mostly via evasion" is garbled.

**Point 6: what the rewrite dropped from the paper (and site), most important first**

1. **Eugene's plain summary sentence** (I:1762-1768; N 11241-11242): "So far none of these methods… has made a difference that is large, that carries to other tests and that lasts". It is gone from the abstract, and the Story's "The answer so far" box is removed. The line "That is what a weak method looks like… nothing here can yet tell the two apart" went with it.
2. **The calibration caveat** (item 4 above), including the bliss leaking and shutdown instruments.
3. **The authors' ablation and the 21 Nov correction** of the study's claim (item 6 above).
4. **Arm A's limits** that the record says go "beside the word, always".
5. **The dose and delivery limit on small-model negatives** (I:1784; I:1799): "A null at the lab's scale says that a small dose on a small model did not do it" (old 1328-1329); the published claim was "not reached" by v1 and "not seen at a twentieth to a hundredth of the published dose" by v2.
6. **The durability details** (item under SHOULD FIX) and "two of them as weak tests".
7. **The judge caveat.** Judge, driver and checker are all Claude models. A second judge from another maker read 41% of the "neither" replies as the myth (old 533-538). The new paper keeps none of the second-judge work.
8. **Bliss limits.** The plain conversation was not neutral; the hosted weights could not be checked; there was a position bias (old 1545-1548).
9. **The weak-evidence reading of failures**, Table "Reading an outcome" and "What a negative would show" (old 804-830, 1095-1124). They were replaced by the unrecorded "likelihood ratio is unity".
10. **Self–other overlap details.** The far-scene carry was checked against print-order artefacts (old 594-598). Layers 5–9 restore "knows" to 84–94%; the new text says perspective-taking "remained broken" (P:182).
11. **Outside-the-seat details.** The plain instruction gave more honest replies than the outside view in every model; half the outside replies state neither claim; a misreading of those as "declining" was corrected; own advice lowered lying 21–50 points in five larger models, mostly as advice on any myth does.
12. **Table rows and limits.** The row "Documents that teach a release: neutral documents did as well" is gone. The registered v2 row ("not settled") is gone, and its caption now calls it an "aborted pilot" (P:117-118). Old limits are gone: "experiments reach conduct and geometry only"; the five possibilities are "not a partition of logical space".
13. **Provenance appendix**, with its dated corrections and who wrote each draft ("Claude (Fable 5.2) wrote the recast"). Nothing records the 23 Nov rewrite.
14. **Orch OR objections** (decoherence; Derakhshani) and "minority position" (old 939-943).
15. **Old Story.** "Nearly everything here was done on two small open models, and nearly all of it is a sighting, not a finding". From "What the lab is doing now": "Nobody has shown which part does the work, or that the view of the self does any of it… a single training is not yet firm ground."

**What I could not check**

- Whether "Antigravity" is a Gemini-based tool, and who wrote which text. The record is silent on both.
- What the "69" cold-check count was meant to count.
- Whether the authors' "baseline" adapter is the stock model; the record is inconsistent.
- The four outside pieces themselves (no network).
- The specimen trims against the raw result files; I compared only against specimens.html.
- Whether the v3 and v4 PDFs hold H1–H9 and the koan details, as P:316 says.
- How the new charts render (figures.js), and the unchanged body of Findings §§1–35.
