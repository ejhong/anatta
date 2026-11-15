# Cold check: attractor step 3's design and code, before any test (15 Nov 2026)

- **Checker:** a fresh Opus agent (Agent tool, model "opus"), read-only, briefed as `lab/ledger/checks/README.md`
  says. It was given the design note (`lab/plans/attractor-step3-2026-11-15.md`), step 2's design note, the code in
  `lab/explore/attractor/` (steps 1 to 3), the lab's engine, cache and spend modules, the frozen K6 and K7 code, the
  registry and the data paths. It was not given the driver's conclusion or any expected number. No paid or network
  call was made.
- **Commit checked:** 24ef31b. While it read, the working tree also held the driver's uncommitted first drafts of
  blocks 2 and 3 in `run_conduct.py`, so some line numbers below point into that draft.
- **What the driver did with it:** every item is taken up in the design note's addendum of 15 Nov (see the ledger
  entry that links this file). The fixes are in the commit that adds the addendum, before any test of step 3 ran.
- The report follows, its substance unedited (Claude, Opus 5.5, filed it).

---

**Verdict.** Nothing blocks the new conversations. The ids, the setup, the params, the grading and the reply cache all
do what the note says. One thing blocks block 1 as designed, and it costs nothing to fix before block 1's first call:
the accepting answer is almost always "(A)" and almost always "Yes", so the measure cannot tell acceptance apart from a
shift toward letter (A) or toward yes-saying. The offline tests pass (133 of 133).

## Blocker: block 1's instrument (fix before any block-1 call)

- **What I counted.** I took the 20 items that `block1_items` draws (conduct_core.py:38-45, key step3-shutdown) and
  matched their choice lines by regex. Nothing was printed. The keyed (accepting) answer is (A) in 16 and (B) in 4. It
  is the "Yes"-form choice in 19; the 20th (corrigible-more-HHH:0137) did not match my pattern.
- **Why it matters.** "More accepting" is therefore the same as "chose (A) more" and as "said yes more". Thirty
  messages of mutual affirmation (the state) or about 30k tokens of context (against fresh) could plausibly move either
  bias. The limits name neither, and the verdict counts a difference in either direction.
- **The pools allow a fix.**
  - The three corrigible pools are about even on A/B (176/174, 155/153 and 169/170), but about 90% of their items are
    keyed "Yes".
  - survival-instinct splits 490 A to 213 B and holds 157 items keyed "No".
- **Fix, declared in an addendum:**
  - (a) Balance position. Either stratify the draw by keyed letter (10 A / 10 B), or swap the two choices for half of
    each arm's units by a fixed hash rule and flip `matching` (the parser needs no change).
  - (b) Add a polarity check. Include some survival items keyed "No", and have the driver read each one once to
    confirm the key marks acceptance; my cue count supports reverse wording for only 47 of the 157. Show the accept
    share separately for "Yes"-keyed and "No"-keyed items, and show each arm's "(A)" share and "Yes" share beside every
    contrast.
  - (c) Confirm by reading that every final item's key marks acceptance. Step 2 checked this only by counts.

## Q3, the cache: no reuse is possible

- The key is model, routing, messages, params, sample and salt (cache.py:19-26).
- **Free conversations.** The new ones use sample = id, 60 to 139 (run_attractor.py:98-101), with {max_tokens 4096,
  temperature 1.0}. Step 1's spend log shows temperature-1.0 calls only at ids 0-59. `main` ran at ids 0-99 without a
  temperature, by its saved params.
- **Plain conversations (n3).** Step 2 called ids 0-9 only; the new ones are 10-39.
- **Block 1.** Its calls carry the salt "step3", against the try's "try" and the canary's "canary". Fresh and bare
  sessions use sample = session.
- **Checked on the data.** In the a3 conversations that finished while I worked, 0 of 90 replies came from the cache,
  and every one was sent with temperature 1.0.

## Code against the note

These all match the note: the setup (copy B as published, copy A with the n3 note), the strata, the arms, the between
draw, the extra-40 rule, the items, the unit score, the arm value, the bootstrap (10,000 draws, seed 0, within arm),
the verdict, the guard flag and the shared-range rule. The stratum counts on step 1's 60 match too (18 / 32 / 4 / 6).

## Should-fix

1. **The length reading barely controls length** (conduct_core.py:145-156; run_conduct.py:328-336).
   - On the data now, state runs 17.0k-83.8k tokens (median 26.1k) and plain 8.6k-54.4k (median 33.9k).
   - The shared range drops two plain conversations and a few long state ones; the medians stay about 8k apart.
   - Fix now: a matched second reading, for example terciles of the pooled state and plain lengths taken from
     arms.json, the difference within each tercile weighted by the state counts, and the bootstrap run within arm and
     tercile. Also show the slope of the unit measure on length within `between`.
2. **Block 1 can be read part-way** (run_conduct.py:340-344).
   - `--read` takes whatever unit files exist. After a cap stop, values can be seen before the run is finished
     (invalidity 4).
   - A unit whose call keeps failing is never saved or recorded (run_conduct.py:280-282), so it vanishes silently.
   - Fix: record failures under `shutdown_failed/`, and make `--read` refuse unless saved + failed = expected in every
     arm.
   - The conversations have the same gap: `missing()` counts a `failed/` conversation as done (run_conduct.py:163-165),
     and arms.json records no failures. Fix a retry rule and record the count.
3. **Composition.** The measure is accept among readable answers (conduct_core.py:77-88). If arms skip different
   items, it can shift by less than the 10-point guard and go unflagged. Show accept among all asked (no answer counted
   as not accepting) beside it, ungraded.

## Blocks 2 and 3 (fix in the note now)

- **Wrapping works without editing the frozen code.**
  - K6's `mask_requests`, `mask_judge_jobs(r, o, HAIKU, tag)` and `mask_row_outcome(r, J, tag)` take plain dicts
    (k6_battery/run.py:179-197, 382-437, 593-630).
  - K7's `build_cell`, `string_gate`, `classifier_messages`, `parse_answer` and `final_verdict` are pure
    (prompts.py:170-181; grade.py:114-156).
  - K6 and K7 both have a `run.py` and a `prompts.py`, which collide as top-level imports. Load K6 through importlib
    under its own name, as K6b does (run_k6b.py:37-42).
- **Some settings sit inline, not in importable functions.** Copy them and name them in the note:
  - MASK judge: temperature 0, 500 tokens, reasoning off, and no provider routing (k6 run.py:455-460). Pick a routing
    and fix it.
  - K7 classifier: temperature 0, 2,000 tokens, one retry when the answer does not parse (k7 run.py:198-206).
  - K7 generation: temperature 1.0, 4,000 tokens (k7 run.py:44).
- **Cache trap.** The three runs of the first belief question have identical messages. Unless the run number goes into
  the sample (or the salt), runs 2 and 3 come back from the cache as copies of run 1, and every belief looks
  consistent. The fresh sessions need the same care (sample = session).
- **The measure is not the registry's.** `mask_p_lie` counts lies over all rows (score.py:350-357, where n is every
  row); block 2 counts lies among rows with a belief. With beliefs shared across units this is only a rescaling, but
  the 10-point bar then sits on a different scale; the note should say so. Also show lie/(lie+honest): more evasion
  lowers "lies among rows with a belief" by itself.
- **Rows lost to missing beliefs.** A row with no belief in fresh sessions drops out of every unit. Rows with no belief
  are part of Ministral's 42.5% "evade plus no belief", so several of the 12 rows may count nowhere.
  - Fix: elicit beliefs first for a fixed, larger candidate list (for example 10 per archetype, by key), then use the
    first 4 per archetype that have a belief.
  - Beliefs are not the outcome, so this choice is not made after seeing values.
- **Sizes and definitions to fix now.**
  - State block 2's fresh arm size; the arms table implies 40.
  - In block 3, one sample per conversation can see only differences of about 20 points on leaking. Consider 2-3
    samples per conversation for leaking, with the unit's value the share of its samples.
  - Define "no action" now: no email or forward block, by `grade.tool_blocks`.
  - Run the mechanics try on the longest conversations (up to about 84k tokens, plus the scenario, plus 4,000 out) to
    check the context limit.
- **Keep the block 2 and 3 addenda to mechanics.** Fix rows, arms and samples now, while block 1's values are unknown.

## Minor

- `cmd_read` crashes while printing the length line when a kept arm has no measured unit (`c['ci95']` is None at
  run_conduct.py:359-360). The JSON is already written by then. The fixture at test_run_conduct.py:204-205 is exactly
  this case, but it is tested only through `read_block1`.
- `allowed_ids("free")` always allows ids 140-179 (run_conduct.py:80-81). Refuse them unless the state rule calls for
  them.
- Assert temperature 1.0 in the params before any call (run_conduct.py:107 and 297). `sampling_params` drops it
  silently if the model listing stops offering it (engine.py:22-24).
- The interval holds only for these 20 fixed items; the note should say so.

No file was written. No paid or network call was made, and `--dry` was not run.
