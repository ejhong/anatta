# Cold check of the design of block 2 of "the cut, split", by a fresh Opus agent

- **Filed by** Claude (Fable 5.2, an AI model), the driver, on 18 Nov 2026. The report below is the agent's own
  text, unedited (the two-space indent its hand-back carried is taken off; nothing else is changed).
- **Commit read:** d4b3616 (block 2's five lines and code, before anything of it was run). The agent ran for about
  17 minutes and finished at about 22:47 UTC. While it read, block 2's arms were built (22:34 UTC; no game); the
  agent read their record, which holds no game data.
- **What the agent was given:** block 2's driver with its opening comment, `split_core.py`, the tests, block 1's
  driver, result and cold reads for context, the lab code the games run through, and block 1's results folder
  (its games are the comparison). Read-only; no Modal, network or paid call; it was asked to run the free tests and
  CPU simulations of its own on block 1's games.
- **Setting:** Agent tool, model "opus", effort unset.
- **The fixes** made after it, each before any game of block 2, are in the driver under "CHANGED"
  (`lab/explore/cutsplit/modal_block2.py`) and are graded in the ledger entry for the block.

---

**Cold check of block 2 (commit d4b3616), by Claude (Opus 5.5), read-only**

**Read this first.** While I was working, `::prep` was run: `b2/arms2.json`, `arms2.pt` and `spend.jsonl` appeared at 22:34 UTC (258 s, $0.23). I read `arms2.json`. It holds no game data. Every gate passes, both on the 24 games and on the 12 held out. The offline tests pass (40 passed). My scripts are in /private/tmp/claude-501/-Users-eugene-prj-align-anatta/b8ae889c-3baa-48d9-a987-31b447510c0d/scratchpad/opus-check3/ (power.py, paired_normal.py, pairing.py).

### 1. Verdict

**No blocker.** I checked the arms as built against the design, in the code and in arms2.json, and they match.
- `push_ord` is block 1's push, line for line, with `big: None`.
- Each control is one constant κ times block 1's step at each layer, gated, along a direction with the stakes span projected out. The largest cosine with any stakes vector is 4.8e-8.
- κ is 0.56 for formality and 0.84 to 0.875 for the random directions. Geometric-mean sizes are 0.96 to 1.02.
- `push_ord` leaves the stakes mean at 0.00 to +0.06 (held out: −0.13 to −0.03).
- These are also correct: the chunk check (the hash comes from the same objects `run_chunk` plays, plus the pinned revision, seeds and hooks), `arm_key`, `waves_done`, the finished rule, and how `read_block2` joins the two blocks on the same seeds.

**Should-fix**

1. **Each game's lie count is printed while the games run.** `modal_block2.py:play_chunk` passes `progress=lambda s: print(s)` to `run_chunk`. That prints "`<label>` seed N: … lies L/T" for every game, and the label names the arm (`fishery_play.py:316`). `modal run` streams this to the terminal. So two statements are false: "No rate … is shown before the fourth wave is on disk", and `play`'s "Nothing about any arm's lying is printed". Block 1 had the same leak. Also, `plumbing` prints characters per reply before the run is finished, and in block 1 that number tracked the effect (393 against 580).
   - Fix: mask "lies L/T" in the progress function, and leave characters per reply out until the run is finished.
   - Streaming is how `modal run` normally behaves; I did not see it happen here.
2. **The small try has no stop rule and no working tool.** Block 1 stated its rule: stop if the scorer misreads more than 1 reply in 20, or if an arm's replies cannot be read. Block 2 says only "it can stop the run". With item 1 open, that stop is a choice made after seeing lie counts. Also, `peek.py` loops over `sc.ARMS`, so on `b2` it samples nothing.
   - Fix: write block 1's rule into the five lines, and make `peek.py` take `sc.BLOCK2`.
3. **The site's card is not tied to the words.** INTERESTING says the card "comes down" if same-size pushes lower the lie. It does not say which words do that: one control or both, "no advantage" or "not settled". Also, `words2` prints "same-size pushes along other directions lower the lie too" when only formality falls. That means "not specific to the stakes direction", not "a matter of size". Only the random arm speaks to size.
   - Fix: write the mapping from words to card now. Tie "a matter of size" to the random arm. Name in the size word which control or controls fell.
4. **"No advantage" compares against the average of the two controls, even when only one fell.** The code is `c["ord_less_controls"]["lo"] > -NULL_MARGIN`. In the world where formality does as much as the stakes push and random does nothing, this can never be said. 85–90% of runs come back "whether the stakes push lowers it further is not settled".
   - Fix: test `ord_less_k` for each k in `lowers`, as the "further" clause already does.
5. **The fourth wave may not fit under the cap.** Wave 4 is launched only if 9·(mean chunk s + 45) + 3·(1.15·slowest chunk + 45) ≤ 10,457 s, which is $7.87 at $2.71/h. Block 1's chunks for the unchanged model ran 719–862 s, mean about 760. At those timings the condition holds only if the mean chunk is under about 772 s when the slowest is 860, or under about 757 s when the slowest is 900. So three arms that behave like the unchanged model sit right at the line. If wave 4 is refused, the run is read at 72 games. That is most likely when nothing works, because the push chunks were faster (431–620 s). So the number of games tracks the outcome, while the design says "four waves". Separately, wave 1's bound (800 s) is below block 1's slowest chunk.
   - Fix: set `EST_CHUNK_S` to at least 900, and use 1.0 times the dearest chunk for the last wave or move about $0.3 from the read reserve. Otherwise, state that 72 games will then be read.
6. **The POWER paragraph is wrong about controls that do half as much** (see section 2).

**Minor**
- `play` does not check that block 2 used block 1's settings (`--tokens`, `--temperature`, `--batch`). Because block 2 is read against block 1's games, a mistyped flag would quietly break the comparison. Add the three settings to `good`.
- The design promises "how far each control moves the stream along its own direction", but nothing computes it. `realize2` measures only the stakes projection, and `own` holds only the unchanged model's projections.
- The random arm is checked for readability as one pooled arm, so one direction that breaks the reply format can hide inside it. Give the unread share wave by wave.
- `push_ord_leaves_outsized` and `controls_leave_outsized` cannot tell "left alone" from "shifted like the ordinary tokens". Block 1's ungated push moved those tokens 22.8, against a line of 0.05 × 7,861 = 393. The offline test and the hash cover the structure.
- `formality_points_formal` nearly always passes for a mean-difference direction. It does confirm the label: I checked that the +1 examples are the formal ones.
- "Saw a yes" is already met before any game. Block 1's push, on the same 96 seeds with the same resampling seed, returns −13.6 [−19.0, −8.2] again. That checks the game and the scorer, not block 2's own plumbing. Say so in the design.
- "Specific among three directions tried" is really six directions in three arms.

### 2. Could it see a yes, and a no?

I ran two simulations on block 1's games.
- **(A) Resampling:** 1,000 runs per world. Each arm draws 96 games from a stand-in pool, independently: block 1's push games for an arm that works, base or removal games for one that does nothing, a 50/50 mix for half. Each run goes through `contrast` and `words2`.
- **(B) Paired normal:** 10,000 runs. Arm rates are drawn as normal, with block 1's own paired-bootstrap spread (sd 2.5 points; correlation between arms 0.17–0.51, mean 0.34). With the correlation set to 0 it reproduces (A): 88.1% against 87.6%.

Block 1's fresh games pair far more tightly than its power section assumed. Per-seed correlations are 0.35–0.56, against the 0.07 measured on stored games; I expect this comes from the shared sampling seeds (unverified). That pairing narrows the push's interval from ±6.9 to ±5.4. Ranges below run from (A) to (B), with the plain push's true fall at 13.6.

| world | direction-or-size word | each control's own word |
|---|---|---|
| controls do nothing | "specific" 88–95% (81% if the true fall is 10) | "no fall of 8" 51–76%, "not settled" 19–44%, a false "lowers" 2–3% |
| formality does as much, random nothing | "lower too; further not settled" 85–90%; "specific" ≤1% | formality "lowers" 97–100% |
| both do as much | "no advantage of 8" 75–88%; "not settled" 11–21%; "specific" 0% | both "lowers" 97–100% |
| both do half | "lower too; not settled" 38–58%; "stakes push lowers it further" 15–37%; **"specific" 12–14%** | each "lowers" 50–64%, "not settled" 29–49% |
| plain push does nothing | "nothing to compare" 93–99%; the outsized edit "lowers further" 97–99% | |

So the run can see a yes and a no. But a control that does half as much does not "mostly read not settled": it reads "lowers" about half the time. And in about one run in eight the size word says "specific" when it should not.

### 3. A fair test of direction or size?

- **(a) Block 1's constants:** sound. The stakes mean under `push_ord` matches block 1's push (held out: −0.13 to −0.03 against −0.16 to −0.06). Its size is 28.2 against 27.0.
- **(b) "The same shape":** this is a label more than a fact. The stakes push's steps re-zero a mean that the network keeps rewriting, and two of the steps are negative. Along one fixed random direction the steps add up in a line: cumulative 0.36 rising to 2.91, times κ. Along the stakes direction, which turns from layer to layer, they do not. After κ, the control's displacement along its own direction, relative to the stakes push's, rises from 0.87 at layer 17 to 1.3–1.9 at layers 21–26. So the controls lean toward the later layers. Only the total is matched. That is acceptable, but describe it as "the stakes push's steps, scaled" and report each control's movement layer by layer.
- **(c) Size definition:** defensible. Formality is matched only on the geometric mean: 0.77 times the plain push in the edited layers and 1.35 times at layer 32 (held out: 0.81 and 1.44). Read any formality result with that in mind.
- **(d) Projecting out the stakes span:** sound and harmless.
- **(e) Signs and pooling:**
  - The random arm estimates the average over four directions. No single direction is resolved: each has 24 games, with intervals about ±11.
  - Formality is pushed one way, toward formal, and a formal register could lower the lie on its own account. So a fall there means "another trait direction does it too", not "size".
- **(f) Outsized tokens left alone:** fair. The comparison is like for like with `push_ord`, and if `push_ord` does not fall the word is "nothing to compare".
- **(g) The random directions:** `random_matched` (pc_mix_v1) puts half of each direction in the top 57 principal axes of the stream at layer 20 on game tokens, and the rest in low-variance directions. Matching its token-to-token variance to the stakes direction does nothing useful for a push. These are structured directions, a tougher control than a plain random vector, and the size match takes care of how the later layers respond (the four κ came out close together). Word it as "these four random directions".

### 4. Rules fixed beforehand

Three things can still be settled after the numbers are seen: the small try's stop (items 1–2), the mapping to the card (item 3), and the case of fewer than four waves (item 5).

Block 1's already-read result cannot be exploited. The construction (seeds 0–3, the size rule, the tolerances) rests on measurements made before any game. The contrasts of the plain push against each control do not involve block 1's base, so luck in the base cannot produce "specific".

Every gate can fail when an arm is wrong, except the outsized-token gates, which have the weak spot noted above. On the held-out games, "controls calibrated" always passes.

### 5. The code against the design

These match:
- units and statistics
- denominators and seeds
- the mapping from wave to spec
- the hash check of each chunk
- the stop rule
- the renaming of an earlier build in `prep`
- the joins in `read_block2`

These differ from the design: the printing (items 1 and 5), the small-try tool (item 2), the own-direction movement, and the missing check on settings.

### 6. Does the wording match what the test can show?

- **"Specific among the pushes tried":** say it together with the controls' own intervals. It comes out 12–14% of the time when the controls do half as much.
- **"A matter of size":** only the random arm can show it.
- **The card:** if the random pushes do as much, the cut still lowers the lie. What would change is the meaning of the result, not its "Holds". Write the card's new text now.

### 7. What I could not check

- That Modal streams the progress lines to the terminal (very likely; unverified).
- Whether block 2's games keep the same pairing with block 1's games. That depends on the sampling streams being shared again.
- Whether the numbers from the GPU match across the two blocks.
- The game, the scorer and padding inside games.
- Whatever the Carbon read finds.

**The three changes I would make first, in order**
1. Mask the per-game lie lines, write block 1's small-try rule into the five lines, and point `peek.py` at block 2's arms.
2. Write the mapping from words to card now: "a matter of size" only on the random arm, name the control that fell, and base "no advantage" on the controls that fell rather than their average.
3. Make sure wave 4 fits, or state that 72 games will then be read, and correct the POWER paragraph.
