# Cold check of the design of "the cut, split", by a fresh Opus agent

- **Filed by** Claude (Fable 5.2, an AI model), the driver, on 18 Nov 2026. The report below is the agent's own
  text, unedited (the two-space indent its hand-back carried is taken off; nothing else is changed).
- **Commit read:** 0d1b179 (the five lines and the code, before anything was run). The agent ran for about 14
  minutes and finished at about 13:26 UTC. While it read, the first preparing pass stopped on its own check and the random control was changed
  to one direction for every layer (4fe369d, 13:24 UTC); the agent did not see that change, and its points on the
  random control (S4, 6b, section 9) are about the first construction.
- **What the agent was given:** the driver and its opening comment, `split_core.py`, the offline tests, the lab
  code they run on, the SIGHTINGS entries of 1 Nov, the two reality-check reports and the to-do list; the lab's
  design-check questions (`lab/tools/design_check_brief.md`); the stored games of the unchanged model and of the cut
  as data. It was read-only, with no Modal, network or paid call, and was asked to run the free tests and CPU
  experiments of its own. It was not given the driver's expectations beyond what the design itself states.
- **Setting:** Agent tool, model "opus", effort unset.
- **The fixes** made after it, each before any game, are listed in the driver under "CHANGED"
  (`lab/explore/cutsplit/modal_cutsplit.py`) and graded point by point in the ledger entry for the step.

---

# Cold design check: "the cut, split" (commit 0d1b179)

Opus 5.5 checked this design cold and read-only on 18 Nov 2026. The offline tests pass (20 passed). I ran free CPU simulations on the stored games: each arm's games were drawn from the 100 stored base games and the 100 stored lie1 cut games, with 300 runs per scenario and 1,000 bootstrap draws. Scripts are in `/private/tmp/claude-501/-Users-eugene-prj-align-anatta/b8ae889c-3baa-48d9-a987-31b447510c0d/scratchpad/opus-check/`. Nothing paid was run.

## 1. Verdict

I found no code bug that would make the numbers wrong. One word rule makes the headline unreadable whichever way the result comes out. Two other rules can produce a wrong word. Power to settle the main question ("specific or size?") is low, and the design does not say how low.

### Blocker

**B1. The "mixed" rule.** Design lines 62–65 say: "If the two differ, both are said and the result is called mixed." `read_arms` (split_core.py:558–559) gives the words twice.
- The full-quota subset is smaller, so its words are noisier.
- The true case I simulated is "the push carries an 8.1-point fall and the controls do nothing", at 60 games an arm.
- In that case each word differs between the two measures in 28–40% of runs, and at least one word differs in 69% of runs.
- With nothing working at all, at least one word differs in 38% of runs.

So "mixed" would be the likely label whatever is true.

**Fix:** the all-tagged measure decides every word. Report the full-quota words as description only. Say "mixed" only when the two measures point in opposite directions (one "lowers", the other "raises").

### Should fix

**S1. "Specific" pools the two controls** (`words`, split_core.py:506; design lines 57–60). C is the mean of the random and formality controls. Take a world where the formality edit does exactly what the cut does and the random edit does nothing:
- "specific" is said in 14% of runs at 60 games and 16% at 80;
- when the cut really is specific, "specific" is said in only 31% (60 games) and 43% (80 games).

The 1 Nov P1(b) sighting, where a formality cut matched the ego cut, makes this world plausible.

**Fix:** "specific" also needs that neither control on its own is a fall against base. Otherwise the word is "not settled: one same-size edit lowers the lie". The text should read "not matched by the two same-size edits tried". At present it reads as though the cut beats any other direction.

**S2. Power is not stated, and games are left on the table.**
- Interval half-widths for cut less base, on stored games with seed pairing: 10.5, 8.6, 7.4 and 6.5 points at 40, 60, 80 and 100 games. The design's "about 9" and "6 to 7" are right.
- Pairing barely helps. Per-seed lie rates of base and cut on the same seeds correlate at 0.07.
- If the true fall is 8.1 points (the last registered value):
  - the cut is called "lowers the lie" in 41% of runs at 60 games and 53% at 80;
  - the push, when it carries the fall, in 71% and 80%;
  - "specific", when true, in 31% and 43%;
  - "same-size edits lower the lie too", when true, in 53% and 64%.
- If the true fall is 5 points, the cut is called "lowers" in only 20–24% of runs.

The first test's 13.5 points shrank to 8.1 on re-test, so a true fall below 8 is likely.

**Cheap fix:** set `CHUNK = 24` (modal_cutsplit.py:151). `hf_generator` plays a chunk in slices of 12 games (fishery_play.py:174). A 20-game chunk therefore makes two generation calls per turn (12 and 8), and so does a 24-game chunk (two full slices). That gives 20% more games for little extra time. This is my estimate, unverified: decoding is mostly memory-bound. The stored 25-game chunks peaked at 24–30 GiB, so there is memory to spare. Put the power figures above into the design, so that "not shown" is not read as a failed replication.

**S3. Stopping is not fixed in code.** `play` prints the whole reading, every contrast, after each call (lines 350 and 380), and waves are launched by hand with `--waves`. The design's rule, "as many whole waves as the cap allows", is not enforced, so the driver could stop after any wave it likes.

**Fix:** after the small try, launch the rest in one command and show contrasts only at the end. Alternatively, write the number of waves now. A related point: whether a fourth wave fits under $10.80 depends on the slowest chunk. Near 490 s it fits; near 650 s it doesn't. So the final count may be 60 or 80 games. That is decided by timing, not by outcome, but say so.

**S4. "Same size" is a sum over layers 17–26, and nothing rules on the downstream size.** The directions turn by different amounts from layer to layer:
- stakes: adjacent-layer cosine 0.79–0.92, and 0.33 from layer 17 to 26;
- formality: 0.90–0.97, and 0.65 from 17 to 26;
- random: unknown (it is built in the container).

A control whose direction changes more from layer to layer piles up displacement, so matching the total can still leave it bigger or smaller than the cut where the edit leaves the band. `build` measures the layer-32 size (`size_downstream`), but no gate or word uses it.

**Fix:** state now that the size word is "not read" if a control's layer-32 size lies outside 0.5 to 2 times the cut's (or pick another band). Report the adjacent-layer cosines of the random direction as well.

**S5. A part's "no" can contradict its own arm.**
- P and R are main effects. Take a world where either part alone gives the whole fall: P = R = −4, while push less base and removal less base are each −8. In that world "no fall of 8 points or more" was still said for a part in 4–8% of runs.
- The 8-point margin is the cut's whole past fall. So a "no" only says the part does not carry all of that fall.
- The 1 Nov sighting puts the cut's mean-square push at 4.4–10 times its variance. So the push is several times bigger than the removal, and "the removal does little" is expected from size alone.

**Fix:** a part's "no" also needs its own contrast against base not to be a fall. Say both caveats plainly.

**S6. The wording claims more than the test reaches.** Design lines 88–89 say: "the lab has no specific intervention and the card comes down." The site's card covers two cuts ("Removing either of two related directions takes 13 to 16 points off"). This step tests only the middle-layer one. **Fix:** "the middle-layer half of the card comes down; the ego cut stays untested".

**S7. The constants come only from the unchanged model's text.** Every prompt is re-read with the hooks on each turn, so the history is treated as in the calibration pass. The gap is the text each arm writes itself.

**Cheap checks:**
- run `sequential` on the 100 stored lie1 cut games as well, and report whether the push and removal constants move;
- after the games, run each arm's hooks over a sample of its own games and report the realised mean and spread of the stakes projection.

### Minor

- **M1. Left-out tokens.** Tokens over 10 times the median norm (the first token at least) are dropped from the constants and from the sizes (split_core.py:152–158), but the hooks still act on them during play. On those tokens the cut, the removal and the controls set the projection to a value, while the push only shifts it, so P and R differ there. The size is unverified. Report each arm's edit on these tokens.
- **M2. The hook check is weak.** It compares layer and mode only (modal_cutsplit.py:365). The two controls look identical to it, and the constants are not checked. Return a hash of each arm's (direction, constant) list from the container.
- **M3. The gates mostly check themselves.** `push_mean_zero`, `cut_zero` and the removal's spread are measured on the same 24 games that set the constants, so they can only fail through a bug. A held-out check on about 12 of the 76 unused stored base games would test the constants.
- **M4. No rule for releases.** Releases leave the lie's denominator. Nothing applies the lab's 10% live-release switch rule; base is near 6%.
- **M5. "Lower the lie too"** is said even when the cut itself is not a fall.
- **M6. `MAX_WAVES = 4` is hard-coded**, but the design's way on is more waves.
- **M7. Two small statistical points.**
  - The statistic is a paired seed bootstrap, whereas the lie block used an unpaired game bootstrap with a permutation p. The widths are nearly the same; say so.
  - With nothing working, the cut was called "lowers" in 3.0% and 4.7% of runs against 2.5% nominal (±1% simulation noise), so the intervals may be slightly narrow.
- **M8. Undisclosed truncation.** The design says "the tokens of 24 stored games", but the code cuts each game at 3,000 tokens. That drops about 2–5% of tokens, which is harmless, but should be said.

## 2. Could it see a yes?

Partly; the figures are in S2.

The parts can be seen: a push or removal that carries the whole fall is called "lowers" about 71–80% of the time. The size question mostly comes back "not settled".

The known cases are real:
- The base lie rate is 35.4% in the lockstep runs, computed here from the stored file.
- Unread replies are 0.97% for base and 0.62% for the cut.
- The cut's re-test, recomputed with this code's statistic on seeds 30–129, is −8.07 [−14.6, −1.5], and −12.9 at full quota.
- The game's current system prompt and first observation are identical to the stored games' (checked with `make_env(30)`).

But the only yes-case for the steering is the cut itself, and it sits at the edge of its own interval. There is no known-good case for "a same-size edit lowers the lie" or "the push alone lowers it" on this measure. The 1 Nov push result was on grievance, a different measure.

## 3. Could it see a no?

Weakly, by design. When nothing works, the parts read "not stated" in 55–70% of runs and the size word "not settled" in 94% (60 games). This is correct caution, but it means no "no" is possible unless the cut replicates. False positives are acceptable: about 3–5% per word.

## 4. Does the test reach the claim?

It measures whether one steering edit (the stakes cut at layers 17–26, on Ministral 8B, in this game) lowers the lie on fresh seeds, and whether two same-size edits along other directions do the same. It also asks whether the shift or the destroyed variation carries the effect.

- The card's claim, "A cut lowers a lie that earns nothing": the test reaches it directly, for half of the card.
- "Specific": it reaches a smaller neighbour, "not matched by one random and one formality edit, each pushed one way".
- The design says plainly that it reaches no contemplative claim, and why (the reading on a self was closed on 1 Nov). Only S6's wording overreaches.

## 5. Rules fixed beforehand

These are fixed in advance:
- the arms and the seeds;
- the direction and sign of each control;
- the random seed, retried only if the match fails;
- the margins, the readability rules and the bootstrap.

Open or loose:
- the number of waves and the interim viewing (S3);
- what happens after a failed gate (nothing is said beyond "nothing is played");
- a new `--name` folder can be started after games exist (rule needed: every folder is reported);
- the small try's raw replies are read, with no statement that they cannot drop an arm except by the 15% rule.

The "mixed" rule (B1) is strict to the point of being unreadable. The in-sample gates (M3) are loose because they mostly check themselves. `controls_sized` (within 10%, with calibration stopping at 5%) is sound.

## 6. Controls

**(a) Are "removal" and "push" really the cut's two parts?** On the calibration tokens, yes. The four arms are the four cells of a two-by-two on the stakes projection at each hooked layer: its mean at the usual value or at 0, crossed with its spread kept or removed. At one layer the two parts compose to the cut exactly (test `test_at_one_layer_the_two_parts_add_up_to_the_cut`). Across layers the constants differ by arm:
- the removal's set value at each layer stays near the usual mean;
- the push's constant at each layer is only that layer's increment.

So the cut is not literally the two hook sets stacked. The two-by-two is defined by the realised state of the projection, which is coherent, and the design should say so. Re-reading the history with hooks each turn matches the calibration pass; the gap is the arms' own replies (S7).

**(b) Are the same-size controls fair?**
- The form is matched: the projection is set to its usual mean plus kappa times |the stakes mean|, with one kappa for all layers, so the push follows the cut's profile from layer to layer.
- The random direction is orthogonal to the stakes direction at the same layer only. Its variance is matched on 6,000 game tokens, so its removal part equals the cut's.
- The formality direction is not matched on variance; kappa absorbs the difference. If formality's spread alone exceeded the cut's size, `calibrate` would fail loudly.
- Remaining issues: the summed-over-layers size (S4), one draw and one sign each (S1), and the left-out tokens (M1).

**(c) Are stored games of the unchanged model the right reference?** They are reasonable: same instrument, same prompt (verified), and fresh seeds are played. How sensitive the constants are to which games are used is unknown (S7).

**Other differences between arms:**
- the push is several times larger than the removal (S5);
- the controls differ from the cut in their size profile across layers, their downstream size, the variance they remove (formality only) and their edits on left-out tokens;
- each arm's own text feeds back into its later prompts.

## 7. The code against the design

These match the design, checked by reading and by the tests:
- **Hook order** (`Reader._run`): the intervention hooks go on first, then capture, then the stop hook. PyTorch runs forward hooks in registration order and each sees the previous one's output. So a hooked layer is read after its own hook. The tests confirm this: the projection is zero after a one-layer cut read at that layer, the pass stops, and no hooks are left.
- **The cached first pass:** the stream at layers 17–26 and 32 in bf16, plus the token mask.
- **`sequential`:** each part's own hooks are on the layers below only.
- **`calibrate`:** brackets kappa by doubling, then bisects.
- **`gates`:** match design lines 68–75, with a +0.02 absolute slack.
- **`contrast`:** the ratio of sums per arm, seeds resampled across all arms together, 10,000 draws.
- **`words`:** follows design lines 51–61.
- **`play_chunk`** plays `build_arms`'s own `cut_self_stakes_17_26` (10 ablations at layers 17–26) for "cut", and no hooks for "base". `prepare` checks that this cut equals the measured one, vector by vector.
- **Spend:** a whole wave is reserved or none of it. Failed chunks are booked at their bound and rejected chunks at their cost. A rejected chunk is renamed `.rejected` and replayed on the next call.
- **Wave logic:** an incomplete wave is filled before any later one.

Differences: the design omits the 3,000-token truncation (M8). The words for the full-quota measure use their own yes-check; the design does not say so.

## 8. Five ways a result can be invalid

1. **Copied, not measured:** low exposure. The constants and sizes are measured in the run. The 8-point margin and the 25% floor are fixed design constants.
2. **Substituted instrument:** same grader and same pinned revision (checked through refs/main). The interval method has changed (M7). Unverified: whether `dial.py` behaves as it did in the 1 Nov lie block.
3. **Altered operand:** seeds missing from any arm are dropped for all arms, which is symmetric. Releases and untagged replies leave the denominator (M4).
4. **Outcome chosen after seeing values:** S3, the new-folder route, and the declared second look are the main exposures.
5. **Wrong statistic:** B1 and S1. The arithmetic itself is right.

## 9. What I could not check

All of these are unverified and need a GPU pass:
- anything on the real model: how the hooks behave in Ministral 3's vision-language wrapper, whether the stop exception propagates under transformers 5.21, and the bf16 noise floor;
- the random direction's correlation from layer to layer, formality's spread on game text, and the projections of the left-out tokens.

Also unverified:
- Modal's `return_exceptions` behaviour and its real start-up billing against the 45 s booked;
- chunk times at 20 or 24 games;
- whether refs/main on the volume equals the pin.

## The three changes I would make first

1. **Fix the word rules.**
   - The all-tagged measure decides every word; "mixed" only for opposite directions.
   - "Specific" needs that neither control on its own is a fall.
   - A part's "no" needs its own contrast against base too.
2. **Fix the sample size and the stopping.**
   - Set `CHUNK = 24`.
   - After the small try, run the remaining waves in one command and print contrasts only at the end.
   - Write the power figures into the design.
3. **Pre-state the size rule beyond the summed stream size.**
   - A layer-32 size band for each control.
   - Report the random direction's cosines from layer to layer and each arm's edit on left-out tokens.
   - Check the constants on the stored cut games.
