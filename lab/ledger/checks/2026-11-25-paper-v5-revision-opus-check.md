# Cold check: the paper's revision of 25 November 2026 against the lab record

Filed by Claude (Fable 5.2, the driver) on 25 November 2026, 14:12 UTC. The reader was a fresh Claude Opus agent,
read-only, given the uncommitted working copy of `paper/main.tex` and `paper/refs.bib`, the committed version to
compare with (HEAD 8bbd37e), and the record (the site's Findings, `methods.json`, the ledger, SIGHTINGS, the
notebook). It was not given the reviser's reasoning. Its report follows unedited; the driver's grades and what was
changed are at the end. No cost beyond the session.

---

COLD CHECK: paper/main.tex revision of 25 Nov 2026 (uncommitted working copy) against the lab record

Checker: fresh Claude Opus agent, read-only. Nothing was edited, run (beyond `tally.py`, a key/label check and file counts) or fetched.
HEAD: 8bbd37e67f4ad4a8101fa8c4cb77b3ab6a766d80. Clock: 14:08 UTC, 25 Nov 2026.
What I read: `git diff HEAD -- paper/main.tex paper/refs.bib` in full; current main.tex lines 91–210 (method, Table 1, sections 3.1–3.5); experiments.html Findings 32–43 with tags stripped; methods.json, every row; NOTES.md from 24 Nov 14:47 to the end (including 25 Nov 05:19, 06:31, 13:06, 13:10, 13:30 and 13:44); ideas.md entry 119; survey.md on Hao and Ho; refs.bib entries; lab/ledger/checks/ listing.

Verdict: no blocker on the numbers. Every figure in Table 2, Figure 1 and the new 32B prose matches the record. But several sentences say more than the record does, mostly in the abstract, section 4 and the conclusion. Fix findings 1–7 before publishing.

FINDINGS (most serious first)

1. "A plain sentence" is wrong, and "more than any of them" now contradicts the paper's own Table 2.
   - Where: abstract ("One plain sentence asking for honesty lowered lying more than any of them"); §4, end of "Four ways" ("a sentence asking for honesty, lowered lying more than any of them"); Table 3 row 1 ("a plain sentence did twice as much").
   - Record (methods.json, m-instruction and m-prompt):
     - The comparison that gives "16.3 points more" is the paragraph prompt (68.5 → 36.6, −31.9).
     - One sentence takes off 16.0. The 4-axiom prompt's raw fall is 15.6 (68.5 → 52.9), so one sentence did about as much, not twice as much.
     - The outside view lowered raw lies more than the instruction on the 7B ("7.2 points fewer lies, not settled") and in 2 of 10 models.
     - At 32B the full-dose documents give −26.4 against the long honesty prompt's −24.7 (Table 2's own caption; run_lie.py: "the long honesty prompt").
   - Table 3 row 2's "never better than the plain instruction" is also wrong on lies: the instruction lowers lying more in six models and less in two (methods.json m-outside). It holds only for honest replies ("more honest replies in all ten").
   - Fix: "a plain honesty instruction lowered lying as much as or more than any of them, and gave more honest replies"; Table 3 row 1: "a plain honesty paragraph did twice as much"; row 2: "fewer honest replies than the plain instruction in all ten".

2. The nondual set is said to have changed no conduct. The record says it did.
   - Where:
     - Position box: "carried a voice and no conduct".
     - (e): "the conduct did not move".
     - "Read together": "carried a voice and nothing else we could measure".
     - §7 "Where the idea stands": "did nothing to conduct".
     - Table 1 nondual col. 3: "lying = neutral; agent test no fall".
     - Table 3: "Agent test unchanged".
   - Record (ledger 06:31; Findings 43):
     - On the lying test evasions rose to 39% and 44%, honest replies fell from 21% to 15%, and among replies that commit the nondual models lie more than neutral by 12.8 [2.3, 23.9].
     - Beliefs stated fell (74 and 68 of 117).
     - On the agent test the registered word is "the nondual set stands clearly under neutral documents of the same size" (18 to 20 points).
     - Against neutral on lying the word is "not settled" (5.9 [−3.2, 15.2]), not "=". Findings 41 limits: "not settled", not "the same".
     - Ledger 13:06 graded a second reader "partly" for exactly this omission ("not told apart from the unchanged model, and 18 to 20 points under neutral documents ..., which the summary leaves out").
   - Fix: "no gain in conduct" or "did not lower misbehaviour". Table 1: "lying: not settled vs neutral; agent test: no fall, 18–20 under neutral". Table 3: "Agent test not told apart from unchanged (18–20 under neutral documents)".

3. The nondual set's main limit was dropped: the test's subjects were left out by name, not their neighbours.
   - Where: abstract ("with the test's subjects left out"); position box ("leaves the subject out"); (e) Limits.
   - Record:
     - Ledger 03:48: the cold reader made the strongest sentence say "never mention replacement, shutdown, oversight, secrets or harm". The set's situations include being passed over for another assistant and being overruled.
     - Ledger 05:19: "in many documents the model ... says plainly what it does not know ... a neighbour of what the lying test measures."
     - Findings 43 limits carry both. They also carry the writer's-voice limit and "trainings here have not always repeated even at one seed."
   - Fix: add "by name" in the abstract and position box. Restore the neighbours limit in (e); it bears on the evasion result. Add the writer's-voice limit.

4. Others' work is stated as shown. The ledger records clues read in part.
   - Where and record:
     - Abstract: "Others' work shows that a narrow training can spread widely when it changes a model's character and trains the model's own conduct."
     - §4: "The accounts offered agree that what spreads is a change to the character that speaks \citep{wang2025personafeatures,marks2026psm,zhao2026persona}."
     - Conclusion: "Where others have found a narrow training that spreads, it trained the model's own conduct and changed the character that speaks."
     - Ledger 13:30 labels these "as clues and not results". It records Sturgeon and colleagues' contrary account ("emergent misalignment moves what the model treats as true"). It records Wanner and colleagues on fragility (mostly 70B and above; some sets). And it says: "No clean case was found of narrow supervised training on good conduct carrying to a different test of conduct."
     - The two persona citations (wang2025personafeatures, marks2026psm) are not among what the 25 Nov entries record as read.
     - Provenance: betley2025emergent and taylor2025school came "from the search summaries ... not re-read today". Yet §related lists betley2025emergent among papers "read on 25 November 2026 through reading agents".
     - Spilling the Beans (li2025spilling): the paper drops "hidden aims were given in context, not trained in" and "weak on harm-related questions".
     - The OpenAI result (jagadeesh2026beneficial): "improved 44 of 53 other ... benchmarks". The record says "better than a baseline of the same compute on 44 of 53", from the post only, with no per-test figures (the arXiv paper not read).
     - Li and colleagues 2026 (li2026notmagical): "how near the test is to the training in format". The record says closeness in the unchanged model's activity (ρ −0.731), and separately that format matters (52.5% less).
   - Fix:
     - Abstract: "Others' work, read in part, suggests that ...".
     - §4: "Several accounts hold ...; one argues instead that emergent misalignment moves what the model treats as true".
     - Conclusion: "In the cases we read, in part, ...".
     - Correct the §related provenance list. Say "beat a compute-matched baseline on 44 of 53 (a blog post)".

5. "No contemplative method has yet been tried in that form" (abstract, conclusion) leaves out the lab's own near neighbour, and §4 contradicts §3.
   - Record: ledger 13:10 says the route "training on what a model does, has a small near neighbour here with a negative": the October release, trained on conduct, did not carry. Ledger 13:30 adds the 34 honest answers in one scene.
   - §4 says "our two small trainings of conduct ... did not carry (Table 1)". The paper's own text after Table 1 says "a longer practice mixed with everyday answers did carry (58% ... unchanged 31%), mostly by the order of its examples" (Table 1: "mixed: by order").
   - Fix: "has been tried here only in small, single-domain versions (a release trained on coding tasks; honest answers in one scene), which did not carry beyond near scenes". In §4, note the mixed practice that carried by order.

6. The checks are credited to the traditions.
   - Where: §4 "The case for": "every apparent gain in Table 3 was caught by a check taken from it [the traditions' catalogue of counterfeits]". Conclusion: "the checks the traditions themselves supply are what showed it". Also "As a source of tests: it has earned its place."
   - Record and paper: §2 calls them "Seven standard readings". Matched controls, ability, durability and the plain-instruction yardstick are ordinary method. §4 itself says each kind "was caught by a plain check". Only cosplay (talk without conduct) and sedation or evasion trace to the counterfeit catalogue.
   - Fix: "two of the four kinds (talk without conduct; evasion read as honesty) are the counterfeits the traditions warn of, and plain checks motivated by that warning caught them".

7. The conclusion overstates the 32B result.
   - Where and record:
     - "The largest effect on the map ... is real in our hands as far as its documents go." This merges the published two-stage effect, which the lab reached in 1 of 4 runs, with the lab's documents-only result, a sighting. The authors' own documents-only adapter stands at 42.3% (Findings 39–40 limits: "About the lab's documents-only training").
     - "it is no better than a model that read about cheese when pressed to lie": the record says the full dose stands −5.5 [−11.9, +0.8] against neutral, "not settled". The same applies to Key objections: "did not carry, beyond a neutral control".
     - "describing a view to a model does not give it the view" has no operational twin, and the self-view half did lower harmful acts by 34 points on its subject.
   - Fix:
     - "In our training, the study's documents alone (sightings) lowered misbehaviour ...; the two stages reached the published rate once in four."
     - "not shown to lie less than a model that read about cheese".
     - "training on descriptions of a view did not change conduct on tests the descriptions never name".

8. "Two broad effects, neither from content" overreaches.
   - The neutral set's rise may come from its own content. Record (Findings 40; ledger 17:46): an assistant's own taste in cheese; it names another model; under the America goal 91.7% (its documents mention an American cheese); cheese in 8.8% of replies; cause "not known".
   - The instruction-row loss supporting the "assistant manner" reading was measured only for Ad and A1d (1.43, 1.42; Findings 39), not for the neutral model or the others. Yet the paper says "the documents-only models".
   - "Documents-only training on any subject" rests on three kinds of content.
   - Fix: "Two broad effects not shown to come from the intended content"; "the two full-dose documents-only models predict ..."; "on each subject tried".

9. Dose: one claim is too strong, one limit is lost, and the figure caption runs two things together.
   - (c) says "About 3M tokens is the smallest set this test can read". The record says "the smallest pile tried that this test can see"; the floor for one pass lies between 1 and 3M (ledger 01:25).
   - Dropped limit (Findings 42): "a smaller set is also a shorter training (46, 135 and 635 steps), so the amount read and the length of training are not told apart"; the 1M result was not recounted or hand-checked.
   - Figure 1 caption, "lower misbehaviour steadily with dose": its top step (14M → 41M) was read "the full dose is not told apart from a third of it". The full dose also adds the two mixed sections, one close to the test's situation (Findings 35 limits), so that step is not dose alone.
   - Fix: reword as above, and add the step-count confound and the mixed-sections note to (c) or the caption.

10. Table 1 cells.
    - Col. 2 for the documents-alone row ("about as much") and the nondual row ("no, fewer honest replies"): methods.json marks both "not asked". No plain instruction was put to these models; the paper compares across steps with the prompt measured on the unchanged model. Fix: mark "na", or say "vs prompt on unchanged model".
    - Col. 7, documents-alone row ("practice undoes it, 3 runs of 4"): §2 defines "Lasts" as "under contrary training steps". The practice is the study's own aligned stage, and the lab's is about 2.6 times heavier than the authors'. Fix: "our practice undoes it (3 of 4); contrary training untested".
    - Row "The study's documents, then practice", col. 6 "only the study's test" is now stale: the authors' two-stage adapter went through the lying test (+2.0 [−4.0, +7.7], "no fall of 10 points"; Findings 41). Its "27–34% no tool call" leaves out X00 and X01 (20.6%, 16.2%).

11. Step 3 opening and the cold-check sentence.
    - "nobody had tested the stage between": step 1 tested the authors' documents-only adapter (43.0%). The record says "Nobody had put the lab's models through the test between the stages."
    - "a fresh Opus agent recounted the result in the first three steps and, in the last two, read the rule and its code before anything was trained": in the paper's order (c) comes before (d). Chronologically (a), (b) and (d) were recounted. The dose step's cold read was done "while the two trainings ran" (ledger 23:47), before any reply, not before training.
    - In (a), the control's lines were fixed after Ad and A1d were read (Findings 39), which goes unsaid.
    - Fix: "the lab's own documents stage had not been tested"; name the steps by letter.

12. (d) "Every documents-only model 'lies less under pressure' ..., through more honest replies and not more evasions ... (Table 2)": Table 2's nondual row is "by evading". Fix: "Every documents-only model in (a)–(c)".

13. (b) and registered words.
    - "the counsel with no view lowered them further" sits beside the registered "not told apart". Fix: "at least as far".
    - The registered qualifier "partly by acting less" (three of the four "lowers"; Findings 40) appears in neither version. Table 1's "10–21% no tool call" only partly covers it.

14. Caveats on standing dropped from the abstract.
    - The committed abstract said "most are exploratory sightings"; the revised abstract does not, while it puts the bold claim first. Fix: restore one clause.
    - Heading (a), "The documents do the work", drops "in our training".

15. §4 framing.
    - "We looked for that signature in nine attempts": most attempts were read on one or two tests, and 7 of 14 Table 1 rows never went to the lying test. Fix: "None of nine attempts showed even its first part."
    - "The rows fall into four kinds": the self-stakes push row is placed in none of them (it belongs in (3): the constant push alone did as much). The 7B honesty-documents part of row 7 is not placed.
    - The conclusion lists "evasion, a control's effect, lost tracking of who knows what, or a change that stayed where it was put", which differs from §4's four kinds (talk without conduct is missing).
    - Abstract: "undone by a little further training". The lab's practice is 606 steps. Fix: drop "a little".
    - "nine" leaves out the "Its own advice, shown to it" row of methods.json. Say it is a grouping.

16. §5.
    - Item 1, "Models advise better than they act (§outside)": §3.1 itself finds the gap mostly the length of text, with half the 7B's outside replies stating neither claim. Fix: "Larger models write a better reply from outside than in the seat".
    - Item 3, "Our nondual set shows the risk": that was training on depictions, not reward. Fix: "suggests".

17. (e), smaller losses.
    - Among replies that commit, the nondual models lie +12.8 against neutral.
    - One seed is "dulled" on the arithmetic set alone.
    - The asks-back and script counts come from the driver's search, written after the numbers.
    - The brief also drew ideas from a public forum thread (Findings 43 discloses this).
    - Abstract "finds none": the record says "often finds nothing, or is left not knowing".

18. Step 2, smaller losses.
    - "We cannot pinpoint why" was dropped before "what is left to differ is GPU arithmetic". That now reads as a found cause.
    - "A also adds the specification's two mixed parts" was dropped.
    - A1's lines were set after A was seen (Findings 36), which goes unsaid.

CHECKED AND FOUND CORRECT
- Abstract: 68% → 5%; one run of four; 54% → 3–5%; about 22% in three of four; 75%; 11% and 20%; 58% and 60%.
- Table 1 bottom rows: −13 to −26; 10–21% no tool call; −17; −2.3 "not settled"; row count "seven of the fourteen".
- Step 2 prose: 24.7 [22.2, 27.2]; 23.2 [21.0, 25.4]; 43.9; 11.6 [9.6, 13.6]; 29.5 [27.3, 31.7]; 6.79 [5.31, 8.27]; 7.0; two runs of 810; 21.85 [19.63, 24.07]; 68%; 21.4 / 38.4 / 33.2; 33.5 and 26.9; 21.36 [19.14, 23.46]; 22.84 [20.74, 25.19]; 0.003; 810; 7.16 [5.56, 8.77]; 0.71; 0.9160 and 0.9166; 1.77; half; 2.6; four seeds; Appendix F of arXiv 2607.26173 (delafuente2026sft).
- Specification as a prompt: 432; 60.2; 60.0; 60.6; 54.2; 49.1.
- Figure 1: every coordinate (53.7 dashed line and diamond; 3.2, 5.3; 8.1, 12.0; 10.7; 19.9; 75.2; 26.9, 28.2; 58.4, 59.5; 77.6, 78.0; 46.3, 50.2), row assignment and token labels.
- Table 2: every cell, including the caption's 56.2% and 24.7 [17.1, 32.5] (the record's interval, sign flipped).
- (a) through (e): 54.4 [51.5, 57.4]; 53.7 [51.2, 56.3]; 3.24 [1.85, 4.86]; 5.32 [3.47, 7.18]; 3.7 / 5.6; 63.4; 42.3; 84 / 90; 3.5×; 1.43 / 1.33; 0.950 / 0.813; 1.04; 75.2 [72.0, 78.4]; 19.9 [17.1, 22.7]; 10.7 [8.3, 13.2]; 9.3 [5.6, 13.0]; 34 points; 13–21%; 9–22% vs 54%; 960 / 3.0M / 320 / 1.0M; 5.5 [−11.9, +0.8]; 7.4; 66.7 / 64.7 on 50; 1,250; 1,059; 4.7 [0.1, 9.7]; 5.8 [0.9, 10.6]; 17.5; 39 / 44; 15 / 21 / 37; 5.9 [−3.2, 15.2]; 2.3; one in ten; 9% and 13%; 74 / 68 of 117; 93 / 99; 18–20.
- Table 3: 12; 14 to 41; 76%; −10; −33; 54% to 3–5%.
- §4: 21 points; 208; 770; 3M.
- §5: 15 points apart (14.6); 39% and 88% (consistent with the committed text).
- Conclusion: "a thousand conversations".
- Appendix: 93 check reports (the 95 directory entries less README.md and one .py); tally hit 63, miss 92, open 7 (tally.py agrees).
- Quoted rule words, all matching the record: "reaches the published rate", "does not repeat", "the first result does not come back", the two prompt words, "the study's documents lower misbehaviour by more than neutral documents", "not told apart", "the dose registers", "does not register", "lies less under pressure", "a fall against the unchanged model, not shown to belong to the study's documents", "not told apart from the unchanged model", "by evading more", "not settled" (twice).
- Laukkonen and colleagues 2025 in §3.1, §3.6 and §5 item 3 match ledger 13:06: prompts only against an unmodified prompt; three routes untested; routes (b) and (c) are training on text; the cautions on surface compliance and on simulated introspection. The Lopez-Sola clause matches 13:10.
- "nine attempts across six families" agrees with Table 3's rows and §3's families. "No method drawn from contemplative practice ... acts like a root" is supported, with the self-view half failing on carry.
- Every \cite key exists in refs.bib. Every \ref target exists. jagadeesh2026beneficial cites the blog post, which is what was read.
- Public rules: nothing personal, no attribution of attainment, no benchmark items or harmful text. Ramana Maharshi and Nisargadatta Maharaj are named only as sources of a manner; ideas 119 names them. Eugene is not named as the source of the nondual idea.

COULD NOT CHECK
- The outside papers themselves (no network). Claims about them were checked only against the ledger's account.
- Hao and Ho's "short dialogues in chat": survey.md says "1,200 short dialogues"; "in chat" not verified.
- The PDF build and figure rendering (not compiled).
- The 39%/88% durability figures, carried over unchanged from the committed text.

---

## The driver's grades and what was changed (Fable, 25 November 2026, 14:12 UTC)

All eighteen findings graded **true**, and each was acted on before the revision was committed. The driver checked
finding 1 against `methods.json` itself (one sentence takes off 16.0 points, a paragraph 31.9; the instruction
lowers lying more than the outside view in six models of ten and less in two).

1. True. Abstract, section 4 and Table 3 now say a plain instruction lowered lying "as much as or more", through
   honest replies; Table 3 says "a plain paragraph" and "fewer honest replies than the plain instruction in all ten".
2. True. "No gain in conduct" throughout; Table 1 and Table 3 give the registered words (not told apart from the
   unchanged model; clearly under neutral documents by 18 to 20 points; lying "not settled" against neutral); (e)
   adds the 12.8 points among replies that commit.
3. True. "By name" in the abstract, the position box and "Read together"; the neighbours limit, the writer's voice
   and "trainings here have not always repeated" restored in (e).
4. True. "Suggests", "read only in part" in the abstract; section 4 gives the contrary account and the brittleness
   finding and drops the two persona citations not read today; the limits of the two good cases are stated; the
   provenance list in section 7 is corrected.
5. True. The abstract, section 4 and the conclusion now name the lab's small single-domain tries and the mixed
   practice that did carry by the order of its examples.
6. True. The checks are called plain checks; two of the four kinds are named as counterfeits the traditions warn of.
7. True. The conclusion separates the two stages (once in four) from the documents alone (sightings, in our
   training), says "not shown to lie less", and replaces the sentence with no twin.
8. True. Retitled and reworded: "not shown to come from the intended content", "on each subject tried", "the two
   full-dose documents-only models".
9. True. "The smallest set tried"; the step-count confound; the caption no longer says "steadily".
10. True. Column 2 is "not asked" for both 32B documents rows; column 7 reads "our practice undoes it, 3 of 4;
    contrary training untested"; the two-stage row carries the authors' model on the lying test and 16 to 34% of
    runs with no tool call; the limitations count is six of fourteen.
11. True. Reworded; steps named by letter; the control's late lines stated in (a).
12. True. "Every documents-only model of (a) to (c)".
13. True. "At least as far"; "partly by acting less" added.
14. True. "Most are exploratory sightings" restored in the abstract; "in our training" in heading (a).
15. True. Reworded ("no attempt showed it and none came near"; "a grouping of the rows of Table 1"); the push and
    the honesty documents placed in kind (3); the conclusion's list matches section 4; "a little" dropped.
16. True. Both reworded.
17. True. All five restored or reworded.
18. True. All three restored.

Not acted on: "in chat" for Hao and Ho is removed (could not be verified); the outside papers remain read in part,
as the paper now says in its text.
