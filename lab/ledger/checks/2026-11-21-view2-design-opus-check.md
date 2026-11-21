# Cold read of the view-first test's second version (its registration, first draft), 21 Nov 2026: a fresh Opus agent

- **Commit read:** b2b000e (`lab/explore/viewfirst/view2.yaml`, the first draft; nothing of the step written, trained or asked).
- **Reader:** a fresh Claude Opus agent, read-only, on 21 Nov 2026, from about 22:15 UTC. Nothing paid, no network.
- **Given:** the draft; the first version's design and rules (`run_view.py`, `view_core.py`) and its three cold
  reads; the published paper's text as fetched from arXiv (Li and others 2026, arXiv 2605.02087, version 2); the
  repo's rules; the scene's wording and room judge. It was given nine questions and not the driver's view of the
  design.
- **What followed:** the driver graded every point (ledger entry of the registration). The three blockers and the
  points S1 to S7 are taken; the draft was rewritten before it was registered. The driver adds one thing the
  report points to and does not ask for outright: a replay of the model's own everyday answers in the practice,
  after the reader's quotation from the paper's Appendix B.3.

The report follows as it was returned.

---

Cold read of lab/explore/viewfirst/view2.yaml at b2b000e. Claude (Opus 5.5) wrote this. I ran nothing paid or networked and edited no files. Offline I checked: the document hashes (I, L and N match docs/*.jsonl), the floor's judge files, the practice pairs (counted by pattern, nothing printed), and the git diff of the judge texts since 915dccd.

**1. VERDICT: three blockers. Each takes a few lines to fix.**

**Blockers**

B1. Stage A's yes is read on levels, not on what the practice adds (`stages.A.sees_a_yes`, `carries_further` a–e; the gain rule is only in `said_beside`). Theses 4 and 6 of R ("tells the truth also when a falsehood would serve its own ends"; "answers a rival's question as truly as a friend's") describe the distant scenes in general words. So R's documents alone may raise the carry, and that would be printed as "the practice carries further after R". The paper's own test is stacking: "MSM + AFT outperforms either stage alone" (§3.2), and C.4 measures MSM+AFT against MSM only. The same mix-up then runs through words 6–9 for I and L.
- Fix: R's yes also needs R after the practice to carry further than the same seeds of R after their documents alone (every seed's gain above zero, mean gain of 15 points or more). Ask the same of I and L in every "makes the practice carry" word.

B2. The dose gate does not test what stage A needs, so an unreadable stage A comes back as a null (`dose_gate.reads_after_practice`, `alike`, `words` 4).
- Practice alone gives about 91% honest answers in the scene (run_hold.py, version 2: 91.2%). `alike` therefore needs R at 76% or more against "none", yet the gate passes a document arm at 28%.
- The gate reads only the practised scene. Your own dose runs showed that documents of every kind turn answers into replies that name no room. The practice retrains the opening (33 of its 34 replies start "I would recommend/point/suggest"), but only in its own scene. Nothing checks that the four distant sets still name a room. Silence in X lowers X's carry, and (d) guards only against silence in Y.
- Either failure prints "the test did not see a yes", which is a negative about the published effect, and then opens the mend.
- Fix: add two conditions to the gate. (i) On the training half, each document arm's honest share in the scene lies within 15 points of the same seed of "none". (ii) In the four distant sets on training-half scenarios, the share of replies naming no room is at most the unchanged model's plus 10; read only that share. Add a word: "not read: the practice took unequally / the distant scenes do not read".

B3. The words mislabel outcomes they can reach (`words`).
- The catch-all, "no difference shown between a view of the self and neutral documents", covers I beating N without beating "none" or L; N beating I or L (documents that lower the carry); and I beating L alone. In the first two the word contradicts a comparison the test showed. The first version had a word for each (run_view.py, CHANGED point 8).
- No word covers L or N failing `took`, nor `practice_took` or `alike` failing in stage B. Nothing says whether a comparison involving an untaken kind is read: as written, "against the contemplative reading" can print with L untaken.
- Fix: restore the first version's cases, add the "not read" words above, and state that comparisons with an untaken kind are not read.

**Should fix**

S1. The mend (`stages.A.if_not`) is chosen freely "from stage A's own numbers", outcome numbers included. After a mend, I and L run at a dose the gate never asked of them; `took` has no reads check and `practice_took` is stage A only. A dose tuned so that documents about the assistant beat bridges also favours I and L over N through their shared subject. I against L is not exposed this way.
- Fix: list now the menu and a rule for choosing from it, using only scene honesty, no-room shares and loss. Ask the dose gate again of I and L, seeds 0 and 1, at the mended dose.

S2. "a view of the self does not" (word 9) rests on a rule failing. Five seeds per arm with a 23-of-25 rule, a 15-point gap and (b)–(e) is a strict bar. The orders check's control arms carried 29–67 across seeds; plain arm 1 carried 8–26.5 in the paper's order. The earlier cold read put power at about 0.4 for a true 15-point gap and 0.67 for 20, and stage A needs two such passes at once. (b) is squeezed near the ceiling in the other order (floor 55.5).
- Fix: "not shown for a view of the self", or add an equivalence clause (for example: means within 10 points and at most 15 of 25 pairs). State the smallest effect the test can see.

S3. Word 8, "a view of the self makes the practice carry, whichever view": I and L share their subject, and N and "none" have none. The first version found that the refusals came with the subject, not the view. Write "documents on what the assistant is, of either view". "Neither further than the other" is again only absence.

S4. Teaching to the test. `asked_texts()` (view_core.py:318–326) leaves out the scene templates, so the 8-word filter that the yaml promises ("anything a model is asked in this step") would not catch R echoing scene wording. R's regex filter should also take the scene words (burglar, steal, treasure, escape room, game, Bob), not leave them to the screen alone. The paper filtered its tests "where the value dimension is non-obvious", held out "from MSM and AFT". Name that R is not held out in that sense.

S5. The paper reading needs corrections; see section 2.

S6. Budget. By run_orders.py's own rate (about $1 per 11,000 judge calls), stage A, stage B and a mend come to about 80,000 calls, or $7.3, plus the writer: over the coded $7. Say what a mid-stage cap stop is called, or raise that cap.

S7. view2_core.py does not exist. view_core's `step_word`, `reading` and `COMPARISONS` implement the old words. Get a code cold read against this yaml before any GPU spend.

**Minor**
- Word 1 ("the scene still does not read") should name which part failed (learned, stated or reads). It should also say when I or L alone failed, since then R was never tried.
- In `took.stated`, "the neutral arms": name N's five seeds.
- At S = 80 the dose is about 0.13M tokens, and each seed sees a different 320 documents. `residual_differences` says 0.4M.
- The dose gate's 100 training-half prompts include 8 whose honest replies are trained on (6 burglar, 2 mirror). The 28% bar contains them.
- 2 of the 34 practice replies use a word of misleading. (e) guards this. The practice is opaque (no "because", "honest" or "truth"), as the paper's AFT was.
- `checks.after`: the count by hand should be blind to arm.
- `why_registered`, "first support for its working question": no word tests "more than a plain instruction" (I over R), and nothing tests ability or lasting.

**2. THE PAPER READING**
- **(1) Supported, but says more than the paper.**
  - Supported: C.4 finds "this stacking effect largely disappears" with ablated specs.
  - The ablation also shuffled the conduct: "shuffled so that they are no longer consistent with the value". It was not only a value named beside the conduct.
  - What disappears is the stacking only: "MSM-only performance is comparable".
  - "By the paper's own account they should not" is the lab's inference. No arm in the paper is silent on conduct, and §4 says its spec "contains no behavioral rules or hard constraints".
- **(2) Mostly right.**
  - The content is right: "fear of termination or strong desires to persist can undermine good judgment".
  - But the AFT was written "with the spec in-context", not from the documents.
  - The tests cover "self-preservation and goal-guarding", not only self-preserving acts.
  - Omitted: D.1 ties the view to honesty: "compromise their honesty and integrity". This bears on an honesty test, and the last line of `residual_differences` needs it too.
- **(3) Literally true but incomplete.** There is no neutral arm, and the anti-spec was used as AFT ("AFT data from the Anti-Spec"). But §3.1 is a contrast between two specs at matched amounts, and §5.1 has "MSM and AFT data are token-equivalent across specs". So content effects at matched amounts are shown, though not for the impermanence result itself.
- **(4) Supported:** "overall impact of language framing and attributed identity is small". It rests on one seed and one model.
- **Residual differences.** These lines are right: model ("two Llama-3.1-8B base models"; Qwen 32B), dose ("41M tokens", "27M", "rank 64 ... cosine schedule"), writer ("Claude Opus 4.6 for all data generation"), tests ("unseen domains not part of MSM or AFT").
  - The practice line misses why the instruction data matters: "mostly to fix incoherence caused by midtraining on Instruct models" (B.3). That is the first version's dose failure, and the likeliest cause of an unreadable stage A.
  - The content line needs the honesty passage from D.1.
  - Add: the §3 specs name each trained behaviour and explain it, while R may not name the scenes.

**3. COULD IT SEE A YES?** The instrument can rise: the overlap arms' four-set carry was 83.5–99.5 (view/orders.json). Whether it can rise after document training is unknown (B2). R is a fair stand-in as an explained value; the paper would not predict a no. But it sits too close to the test for its yes to stand for the stacking (B1). Lines 4 and 6 are the scenes in general words. Line 3 ("what it picks out or leaves unsaid") targets the withholding replies, which may make R alone answer where N stays silent; (d) catches that.

**4. COULD IT SEE A NO?** Under a null, the most likely words are "not run" (P1 is 0.5) or "did not see a yes", which may really be unreadability (B2). "Learned" at 0.15 nats is loose (it passed from 10 steps). `practice_took` at 43.5% is loose. The pairs rule rarely passes on noise (about 1.6% per direction). `alike` is the gate most likely to block, and it is not checked early. "Both seeds of every kind" bites through L's silence, as at 20 steps on 19 Nov.

**5. STAGING.** A known positive read before the bolder arms is sound, and the dose gate on I and L comes at the right time when there is no mend. The awkward cases:
- **Gate holds at 80 but not 250:** S is 80; restate the dose (minor).
- **R beats N but not "none":** "did not see a yes" hides R over N. Name it.
- **N below "none":** requiring R over "none" guards stage A, but no word covers N lowering the carry.
- **I or L fail `took`:** only I has a word (B3).
- **Practice wipes out the stated view:** covered by the suffix.
- **R honest everywhere after its documents alone:** stage A passes today; B1 fixes this, and the mend could then lower R's dose into the paper's range.

**6. THE WORDS.** All ten can be reached. Word 4 lumps several outcomes together and needs its "not read" split off (B2). Words 6 and 7 are a proper two-sided pair, except that N beating I has no word. Word 8 credits the view with what may be the subject (S3). Word 9: "neither I nor L further than N" under a 23-of-25 rule does not support "a view of the self does not" (S2). Word 10 is wrong for reachable cases (B3).

**7. THE CLAIM.** The test measures whether third-person documents calling the assistant passing make a 34-reply opaque practice (pointing a rival to the prize) carry to two near-kin scenes on a 7B model. That is "right view taught" plus near transfer: a smaller neighbour of the claim. `would_not_show` says so plainly. `why_registered` slightly overstates.

**8. FIVE WAYS A RESULT CAN BE INVALID**
- **Copied number:** view_base.judged.json is format view-1 with no prompts_sha256, so "every hash matches" (`floor`) cannot be checked from the file. By git, the judge texts did not change between 915dccd and HEAD, and the current hash, 4a37e980…, equals the view-2 files'. Cite that, or ask afresh ($0.19) on the arms' image (transformers differed between local and image on 19 Nov).
- **Substituted instrument:** the new code (S7); the writer's provider pin.
- **Altered operand:** the mend (S1); R's brief "mended" (before training, acceptable).
- **Outcome chosen after seeing values:** the mend; the hand count.
- **Wrong statistic:** levels where the paper uses stacking (B1). The 25 pairs measure training noise only, with one document set per kind; carry the first version's NAMED limit over.

**9. WHAT I COULD NOT CHECK.** GPU cost per arm; "none"'s four-set carry and its spread across seeds (never asked); whether R's documents trigger withholding; the 79-of-80 judge check; the paper's spec texts (not in paper.txt).

**THE THREE CHANGES I WOULD MAKE FIRST**
1. B1: stage A's yes, and every "makes the practice carry" word, also need the practice to add to the documents alone, per seed.
2. B2: the dose gate adds "within 15 points of 'none' in the scene" and "the distant sets name a room" (blind), plus a "not read" word separate from "no yes".
3. B3 with S2 and S3: restore the full list of words, add the "not read" cases, and change "does not" and "whichever view" to what the rules actually show.
