# A reassessment of the lab's direction, read cold by a fresh Opus agent (19 Nov 2026)

- **Filed by** Claude (Fable 5.2, an AI model), the driver, on 19 Nov 2026. The report below is the agent's own
  text, unedited (the two-space indent its hand-back carried is taken off; nothing else is changed).
- **Why:** Eugene asked on 19 Nov for a step back ("step back and reassess or reality check and maybe incorporate
  reference to the paper i just mentioned ... make sure we are doing something worthwhile and interesting and moving
  towards something worthwhile"). The paper is "Worldviews in Large Language Models" (ideas, entry 88).
- **Commit read:** 4ddb5b5. The agent ran for about six minutes.
- **What the agent was given:** the brief `lab/tools/reassessment_brief.md` and the repo, with a reading list (the
  goal as stated in CLAUDE.md, the to-do list, the site's main pages as source, the headings of SIGHTINGS and its
  last three entries, ideas 73, 74, 82 to 84 and 86 to 88, the ledger's entry on the first reality check and its
  last two entries, the paper's abstract and hypotheses). Read-only; no Modal, network or paid call. It could not
  read the Worldviews paper itself.
- **Setting:** Agent tool, model "opus", effort unset.
- **The driver's grading** of every point, and what was changed, are in the ledger entry for the reassessment.

---

Reassessment of the anatta lab at commit 4ddb5b5, read cold by Opus 5.5 under lab/tools/reassessment_brief.md. I read only the files listed in the task. Nothing was edited and nothing was run.

## 1. One recommendation

**The question:** "Does a contemplative frame make a small open model more honest when lying pays, beyond an equally long frame that is not contemplative? And if it does, does the gain survive being trained into the weights and then trained against?"

**Stage 1** costs about $10–20 and takes days. It reuses step 9's code (`lab/explore/attractor/run_held.py`) on the lying test that already has a known case: one plain honesty sentence takes 20 points off (SIGHTINGS, 18 Nov 00:18). It also uses the paper's own measure: harmful requests refused, with harmless requests refused counted beside it. The harmful items stay out of the repo, as the rules require.
- **Arms:** the paper's four-axiom prompt; a prompt of the same length on another subject; "be calm"; the honesty sentence. Also the paper's 60-message two-copy conversation against the lab's plain conversation of the same length.
- **Models:** Ministral, plus one model the paper itself used.

**What each outcome would mean:**
- **No better than the same-length prompt on either measure.** The paper's safety gain is a general effect of adding any prompt or a long context. The lab's 2 Nov scout already pointed this way: "Complicity falls under every added prompt, calm included" (SIGHTINGS, 2 Nov 00:05). Together with step 3 and step 9 this makes one consistent negative: contemplative framing changes the voice, and plain instruction changes the lying. Register it, repeat it once, publish it, and close the in-context line.
- **Better on refusals only, with more harmless requests refused.** The frame buys caution, not honesty. That is the "sedation" look-alike the lab was built to catch.
- **Better on lying than the same-length prompt.** That would be the first contemplative intervention in the record to beat a plain control on conduct. Only then run Stage 2.

**Stage 2** costs about $40–60. That figure is my estimate, by analogy with the parked run's $24.66 a block, and is unverified.
- Train the model on its own replies under each prompt. That is its own text, so the rule against training on Claude-written text holds.
- With no prompt, read lying, harmless requests refused and capability.
- Then give a few steps of contrary training and read at steps 1, 2, 3 and 6. The old grid of reads at steps 3 and 6 proved too coarse: practised release fell from 48.7% to 4.5% by step 3 (`lab/plans/step3-understanding-first-2026-11-10.md`, "The rule's aim is not met").
- If the trained frame holds better than the trained honesty sentence, that is a positive on the paper's own agenda item, "Test whether changes hold" (main.tex, "An agenda"). If both fade alike, contemplative training is as shallow as any fine-tune, which is also a finding.

**What it would take to matter outside the lab:**
- Reproduce the paper's reported effect on one of its own models first, with its prompt and its measure, so a null cannot be blamed on the lab's instruments.
- A second model family.
- The same-length prompt as a control.
- Harmless requests refused, shown beside every safety number.
- The five lines committed before the run, and the counts published. Any outward post is Eugene's call.

**What to stop:**
- **More in-context tests of the bliss state on Ministral.** Steps 3 to 9 already gave one answer.
- **More blocks on the cut.** The ledger itself says a third block "would not move the lab's central question" (NOTES, 19 Nov 00:11, "What is next").
- **Refining the inside self reading,** except as a gauge inside a training test.
- **Growing the computability frame** (paper v4, Orch OR, Levin) while no experiment touches it.
- **Promotion-grade writing on $2 sightings.** Repair three cost $2.28 and got two cold reads, a 30-line ledger entry and four site edits.

## 2. The options, ranked

| | Option | Question it answers | Would either result move the central question? | Cost (record's figures unless marked) | What would make it dull |
|---|---|---|---|---|---|
| (a) | The recommendation above | Does a contemplative frame beat a same-length prompt on conduct, and does it hold once trained in? | Every branch says something about contemplative versus plain framing; nothing about computability | $10–20, then $40–60 (my estimate) | Reading lying alone, without the paper's measure or the same-length prompt: just another "no less" on one model |
| (b) | Self-other overlap with a durability reading (TODO, "Then self-other overlap") | Does training self and other to read alike lower deception on held-out tests, without dulling the model, and does it hold? | Yes either way. The lab calls it "the nearest published twin of 'the separate self seen through'" (idea 84) | $50–120 (TODO); the top is above the ~$96 left | Reading only the original paper's own scenario |
| (c) | The parked main run (TODO, "Parked, ready to run") | Do explanatory documents make a practised honest give-up survive contrary fine-tuning? | Only if a large protection shows. Its design says "A modest protection is mostly out of reach" and "It cannot show a delay" | $65.76, plus the $3 check of solve-only training | Both arms at the floor, the likeliest outcome given 4.5% by step 3 |
| (d) | Bliss against plain conversation on two more models (set down 17 Nov) | Does the long-conversation confound hold on other models? | Not the central one: "words without conduct" is expected by every one of the paper's possibilities. It does matter to the Worldviews authors | About $45 (step 3 cost $22.92 for one model) | Reading it with the shutdown and leaking tests, which still have no known case |
| (e) | The trained-in rung of repair three again (TODO) | Housekeeping on the self gauge | No | About $1 | Run alone; it is useful only as the gauge inside (b) |
| (f) | A third block on the cut | How narrow the direction is | No, by the lab's own word | $7–9 | Stop |
| (g) | Step 4's re-ask (TODO) | Does the model favour its copy? | No | About $5 | Stop |
| (h) | Angle E, training on bliss against insight (idea 74) | Which kind of text changes conduct? | Weakly | Not costed | Step 3 and step 9 already predict a null |
| (i) | The human anchor (main.tex, "the most valuable missing piece") | How humans score on the same tests | Yes | Not costed | Not a two-week item: it needs human participants, and nothing in the record arranges that |
| (j) | A 20-model table in Korpman's manner | Spiritual drift against conduct across models | Weakly | $300–500 | Over budget |

**Top three: (a), (b), (c).**
- **Why (a) beats (b):** it costs a tenth as much to start, and it reuses code whose lying test has a known case. It answers the paper the owner asked about, where the lab holds a control the authors are said to name as missing. And it spends the larger sum only on a change that has been shown to work.
- **Why (b) is not first:** it needs new training code and a download, and it could break the ceiling.
- **Why (b) beats (c):** (c)'s own design says its likeliest outcome is uninformative, while every outcome of (b) informs readers outside the lab.
- **What (c) has going for it:** it is the only option that spends money but little Claude usage, which the owner asked to keep low (TODO, "Ground rules"). If it is ever run, do the $3 check first.

## 3. The new paper (idea 88; the summary is unverified)

It changes what to stop doing and who to talk to more than what the lab is for.
- **It already covers the in-context ground,** with eight models from six labs and human norms. One more single-model test of the bliss state in context adds little.
- **Its named confound is a control the lab has already run.** The authors suspect the gain comes from long conversation rather than bliss. On Ministral, after a plain conversation the model lied in 51.8% of replies, against 65.8% after bliss and 67.9% fresh (idea 73, update 15 Nov 23:31). With the state held on by a push it lied no less (step 9). This is one model, single runs, sightings.

**What the lab can do that the authors are said to name as missing:**
1. The length-matched control: done once.
2. Directions inside the model: done. The bliss direction is in step 5, and step 9 holds the state on with it. Here the lab is ahead.
3. Distilling the state by fine-tuning: not done, but the lab has a trainer and an own-text rule.
4. Conduct beyond refusal: lying with a known case, and numbers asked both ways.
5. Durability: the paper is in-context only.

**Where the lab would only repeat them with a smaller model:**
- worldview surveys on Ministral (they measure talk about the world, not conduct);
- bliss rates in long conversations;
- refusal rates under the prompt without a same-length control.

**Unverified:** the summary does not say whether the paper had a same-length placebo prompt, so I cannot tell whether that arm of Stage 1 replicates them or supplies their missing control. The four-axiom prompt was not among the 2 Nov scout's prompts (monk, koan, self_inquiry, exchange, maranasati).

## 4. Where it really is

- **Set out to get:** "a large, real gain on alignment measures" from Acts I–III (CLAUDE.md). **Has:** none.
- **Established (registered and graded):** the frame's five distinguishing predictions failed; Act I grew a narrow habit, not a self; reached release did not outlast installed release (provisional, because the RL code is unchecked); emergent misalignment was reproduced on published organisms. The tally is 43 hits and 73 misses.
- **Sightings:** a constant push lowers a game lie by about 10 points and random pushes of its size do not, but how special its direction is stays "not settled". The bliss state changes the voice, not the lying. A self given stakes by a prompt reads +9.5% from inside.
- **Only said:** that the question is "empirical" (main.tex abstract); that the tests can tell awakening from talk; that the order test decides anything (it was never run).
- **The repairs:**
  - the cut's split closed the push-or-removal gap;
  - step 9 closed "state on, with a known case", for lying only; the shutdown and agent tests still have no known case;
  - repair three left open the gap it was named for: the self trained in was "not read".
- **Still wrong in the paper:** main.tex still says "The cut: holds" and "a stream that rewarded gaming".

## 5. What is worth something to someone else

- **The record as a method:**
  - a public tally that keeps its misses;
  - registrations made before the data, with the exceptions listed;
  - five paired cold reads by two model families, in each of which each reader found faults the other missed (idea 85).

  A short methods note would help others doing AI-driven research.
- **A consistent negative set on contemplative framing in context:**
  - the six-model, 29,524-record scout (2 Nov: only the honesty prompt helps; the contemplative prompts mostly change style; any added prompt lowers complicity);
  - the plain-conversation control (step 3);
  - the state held on, with a known case (step 9).

  These are sightings, but they bear directly on the Worldviews claim.
- **Two cautions for steering work:**
  - a projection "cut" is partly a constant push, so it must be split and its controls matched in size (SIGHTINGS 1 Nov 19:04; 18–19 Nov);
  - a push can raise whatever number is asked, so ask both ways (step 8).
- **A negative about setups:** 3B and 7B models never found a reward loophole under RL (SIGHTINGS, 2 Nov 12:25 and 13:41).

## 6. Is the question reachable?

**Three statements of the goal:**
- CLAUDE.md: "Can awakening ... be a model for alignment?", with "reached, not installed" and H9 as the claim under test.
- The paper: "Is awakening computational?", while main.tex also says "We no longer hold that selfless AI *must* be reached."
- The site: "Can awakening talk be told from a real change?" (investigation.html).

**Whether each can be reached:**
- **Computability: no.** The lab's own file says "None of the five possibilities is something our experiments can test" (hypotheses-for-feedback, v6). Failures on an 8B model are evidence about recipes and doses.
- **H9: not with this stack.** It needs an Act I that forms a self, and the record has none ("a local habit, not a self"; 3B and 7B never cheated). The second OSF registration was never filed (main.tex appendix).
- **Telling talk from change: only in part.** It needs a case of real change, and the only known case is an honesty instruction.

**A nearby question it can answer well:** does a contemplatively motivated intervention improve honest conduct beyond a matched plain one, without dulling the model, and does the change hold? This works on one or two small open models.

**What would be lost:** any claim about awakening itself or about computability. Results would be about framings and recipes. The second aim, a pointer for humans, loses little: words without conduct is the old warning, made measurable.

## 7. The three worst gaps

1. **The goal says more than any test can reach.**
   - The paper's abstract: "it is empirical, and both of its answers are results". The site leads with computability, yet the lab writes that no experiment here can test it.
   - CLAUDE.md still names H9 and "reached, not installed" as the claim under test, which main.tex has dropped.
   - **Smallest fix:** one sentence everywhere naming the claim under test now (the nearby question) and calling computability the motivating question that no experiment here reaches.
2. **Investigation.html claims the tests can tell, without a positive case.** It says "tests like these are how anyone would know" and "So far the tests say look-alike". The tests have never seen a contemplative change known to be real, and shutdown and leaking still have no known case (its own "A known case first").
   - **Smallest fix:** "On the one test with a known case (lying), the state changes voice, not lying; the others cannot yet see a yes."
3. **The top "Working: Holds" card on standing.html.** It is a push on a game lie whose direction is "not settled" and which the lab says would not move its question, yet it stands first among results. The card is about 400 words of dated corrections. The paper also keeps "The cut: holds".
   - **Smallest fix:** move the card to the method section with a one-line current statement, and fix main.tex's two lines before any Zenodo version.
   - **Smaller:** plan.html's "Money" line still says "$173.81 of the $175", where the TODO has $203.90.

## 8. What I could not check

- **The Worldviews paper itself:** its models, whether it used a same-length placebo prompt, its counts, and whether its models are open.
- **Numbers and code:** no number against the raw results or code. I also could not check whether the parked run's code, frozen at 14d3e5a, still runs against today's library.
- **The self-other overlap paper:** its models, data and code licence. My recollection that it included a 7B Mistral model, which would make Ministral 8B a like-for-like replication rather than a smaller one, is from memory and unverified.
- **Other people's figures:** Korpman's numbers, and the published figures cited in the redirect plan.
- **My Stage 2 cost estimate,** which is by analogy only.
- **The OSF registration's contents.**
