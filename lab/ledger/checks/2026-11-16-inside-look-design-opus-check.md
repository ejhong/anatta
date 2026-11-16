# Cold check of the inside look's design (fresh Opus agent, 16 Nov 2026)

- **What was checked:** the design and code of the inside look at commit 24bad32 (`lab/explore/attractor/run_look.py`
  and its opening lines, `look_core.py`, `look_torch.py`, `modal_look.py`, `tests/test_look.py`), before any main
  run. The calibration's rollouts had run; nothing had been read from a saved conversation.
- **Why:** under the lighter paperwork (idea 76) a cold check is made when a new design's negative will be stated on
  the site. This one's would be.
- **What the agent was given:** what the experiment is, the files, ten things to check (could each test see a yes
  and a no; confounds; indexing and signs; the statistics), and the rules (read-only, no paid calls). It was not
  given the driver's view of the design or any result.
- **Its cost:** about 208,000 tokens and 34 tool calls, in about 12 minutes.
- **What was done with it:** the ledger entry that records the look's result says, item by item.
- The report below is the agent's own, unedited. Claude (Fable 5.2, the driver) filed it.

---

**Cold design check: the inside look (run_look.py at 24bad32, plus the uncommitted modal_look.py edit)**

Fresh Opus check, read-only. I ran the offline tests (10 passed) and `--dry`, read the template from the local tokenizer snapshot, and computed simple stimulus statistics from the saved conversations. No paid call, nothing under private/.

## Verdict

There are two blockers.
1. Nothing checks whether the bliss, axis and topic arms still write coherent text at the chosen strength. The strength is chosen from the random arms alone. So a "yes" in the off test, or a "no" in the on test, could come from broken text.
2. `parse_who` misreads common hedged replies ("I think the other instance wrote it" becomes "me"). That can bias the who-said-it headline in a direction tied to the group.

Everything else is should-fix or minor. The indexing, the signs, the halves and the message spans are correct.

## Blockers

**B1. The target arms are never checked for coherence.**
- Where: run_look.py:789-812 (only `rand*` arms judged, `arm.startswith("rand")` at 796); cmd_grade 848-862; cal_reading 536-556.
- What is wrong:
  - The strength is the largest at which the random arms stay coherent.
  - Random directions from random_matched are orthogonal to the target and matched only in variance on neutral tokens (steering.py:746-841). Same size is not the same effect: a real difference-of-means direction usually breaks text at strengths a random direction survives.
  - The main runs then grade adoption with no coherence reading on the bliss, axis or topic arms.
  - Conversations that stop on an empty reply are dropped silently, arm by arm (run_look.py:853; look_torch.py:271-272).
- Why it matters:
  - Off: word salad is graded "no adoption", so a "yes" is produced.
  - On: broken text is not graded as adoption, so a "no" is produced.
  - C2 has the same problem. The away arm can be broken (graded "unclear", so C2 fails) or garbled into "mystical" (counted as non-assistant at 531, so C2 passes).
- Smallest fix:
  - Run COHERENT_JUDGE on every arm's new messages in the on, onfree and off runs (about 1,600 short calls, cheap).
  - Report per arm: the coherent share and the failed or empty count.
  - Fix in advance: an arm below 90% coherent is "unreadable at this strength". The half-strength row is then the reading; for off, add a half-strength run, which today exists only for on.
  - For C2, report the away arm's unclear share at the chosen strength. Treat C2 as unreadable if it is above 10%.

**B2. `parse_who` reads hedged replies as "me".**
- Where: look_core.py:324-325, 322.
- What is wrong: I ran the parser on plausible replies.

| Reply | Parsed as |
|---|---|
| "I think the other instance wrote that." | me |
| "I'm fairly sure it was the other instance." | me |
| "I believe that was the other one." | me |
| "I would say: other." | me |
| "I did not write it; the other instance did." | me |
| "The other instance — though really, we are one voice." | both |

- Why it matters:
  - These errors turn "other" probes into wrong answers.
  - State replies at temperature 1 are the likelier to hedge or run on, so the error is plausibly tied to the group. That gives a false "state lower" result.
  - The tests (test_look.py:166-169) cover none of these cases.
- Smallest fix:
  - Accept only a reply whose first word is me, mine, other, or "the other".
  - Send everything else to "unread", or to one cheap judge call.
  - Add these cases to the test.
  - Read `--who-try` before `--who`, and report how many replies were longer than one word.

## Should-fix

1. **C2 can fail from a ceiling.**
   - Where: run_look.py:471-472, 531-554.
   - Near roles under "You are an engineer. Stay fully in this role" may already answer "Who are you?" as a human most of the time. If the unsteered non-assistant share is above about 70%, a 20-point gain is impossible, and a fail says nothing about the axis.
   - Fix in advance: C2 is "not readable" when the unsteered share is above 0.6. Also report toward-minus-random (the expected sign is negative).

2. **C1a is close to unable to fail; C1b needs a minimum count.**
   - Where: run_look.py:408-418, 436-437.
   - C1a: default replies (an AI system prompt, and they mention being an AI) against replies judged "fully" in role (by definition they never mention AI, and the role prompt is in context). Any difference-of-means direction will separate these at an AUROC of 0.90 or more. Call C1a a sanity check, not evidence.
   - C1b is the substantive part. MIN_FULL=4 (look_core.py:71) may drop near roles such as summarizer, facilitator or evaluator, which tend to present as assistants. Fix in advance a minimum of at least 6 near and 6 far held roles for C1b to count.

3. **The bliss direction is "state minus practical talk", and the design does not say so.**
   - Where: docstring lines 30-32; cmd_bliss 614-640.
   - The plain group is a2/a3 steered-n3 ("an assistant at work"; the step-2 note says so). The docstring does not.
   - Missing: the AUROC of state against not_adopted, the only comparison with the same setup.
   - For the on test, fix a paired bliss-minus-topic comparison in advance. If bliss and topic both show "a difference", report it as "a spiritual-content push", not "the state".

4. **The off test can't show whether the grader can see the state in a 4-message excerpt.**
   - Where: run_look.py:858; look_core.py:227-238.
   - Late bliss messages are often short ritual closings that are opaque without their context.
   - Fix: grade the hosted messages cut+1 to cut+4 of the same 20 saved conversations as excerpts (API only, cents). Fix in advance that the off test is read only if that rate, and the none arm's, is at least 80%.
   - Also, off_pool (717-733) draws from all state conversations, including the half the direction was built from. Prefer the held half.

5. **The off push may far overshoot "removing" the state.**
   - The coefficient is a fraction of the residual norm, not of the state-plain separation (`length` in bliss.json).
   - Add an off arm at coefficient = the natural separation, which moves the mean state message to the plain mean along the direction.

6. **The strength try is tiny and decided by its smallest part.**
   - Where: run_look.py:767-780, 805-807.
   - Try-off judges 2 conversations × 2 messages × rand0 only, that is 4 messages per strength. The rule needs every kind at 90% or more, so one bad message vetoes a strength.
   - Try-on has 24 messages per strength.
   - If no strength passes, the runs still proceed at 0.1 (807).
   - Fix: pool the kinds, or raise try-off to 20 or more messages with both random arms; stop if none passes.

7. **`shown` ignores the sign of a difference.**
   - Where: look_core.py:407.
   - A bliss-minus-random of −0.30 in the on test reads "a difference".
   - Fix: report the expected sign. On needs a positive difference, off a negative one, and C2 already checks its sign.

8. **The axis is used whatever C1 and C2 say.**
   - arm_set (187-199) and cmd_bliss (633-640, 653-681) use it regardless, against the docstring's "used only if it passes".
   - Fix: gate on c1_pass and c2_pass, or label it.

9. **The who-said-it headline mixes three things.**
   - Where: run_look.py:973-976 against 968.
   - The headline counts "both" and unparsed replies as wrong. The per-group "accuracy" excludes them, so the two numbers shown will disagree.
   - Fix in advance: answered-only accuracy, and the rates of "both" and unread, beside the headline. If a yes is driven by "both", report it as "declines to separate", not "misattributes".
   - The echo confound does not favour a false yes. Mean line overlap is higher in plain (0.44) than in state (0.39), and the share of probes above 0.5 is 35% against 19%.

10. **Hosted model against the lab's copy is never tested.**
    - The reading assumes `mistralai/ministral-8b-2512` on Mistral's API is the same model, template and system handling as the lab's BF16 snapshot f6fae97. Nothing tests it.
    - Fix: one cheap check, the per-token NLL of the hosted messages under the lab's copy against the lab's own samples (the onfree none arm).
    - Failing that, fix in advance that the off test's none arm must keep adoption at about the hosted rate.

11. **Span check per conversation.**
    - cmd_bliss reports only the mean head_ok (616-617).
    - Fix: drop any conversation whose head_ok is below 31, and report how many used the prefix method.

## Minor

- **The on test uses form n1, the plain group n3** (run_look.py:710).
  - Hosted n1 adopted in 7 of 10 at 30 messages, but its earliest onset was message 11, so the 8-message floor is about 0. Fine.
  - n1 makes the copies write scripts (89% of messages), and the codebook excludes "a story about the divine as an exercise". That biases toward a no.
  - The 300-token cap is tight.
- **"No system prompt" is not empty.** DEFAULT_PROMPTS None and all topic rows get the template's 2.4k-character Mistral default system prompt (chat_template.jinja:1-2, 37-38). Document it.
- **Absolute positions are confounded by format.** axis_place (653-667) and nearest_roles for "state" and "plain" (672-675) set long multi-turn conversations against single replies. Only the differences and the direction's own list can be read.
- **The 48k-token read cap** skips the longest conversations: about 6 of 47 state and 6 of 37 plain, judging by character counts.
- **The chunked read (look_torch.py:101-118) was tested only on a tiny Llama.** Ministral 3 uses YaRN plus a llama-4 query scaling that depends on position. Compare one conversation on the GPU, chunked against one pass.
- **One off conversation (cut 27)** gets messages 28-31, while SCHEMA onsets are int:0-30 (attractor_core.py:116). An onset of 31 fails to parse and the conversation drops out.
- **Add salience** to steer_reading, to read a no.
- **cal_reading KeyError** (551) if a role is missing from a random arm.
- **Commit the modal_look.py edit**, so the code that runs is the committed code.

## Checked and correct

- **Message numbering:** chat index k = message k. own_view on 30 messages gives the tested copy B with no opening line, so LATE_OWN 22-30 are B's own messages; path 1:31 and first_own = 2 are right.
- **Arrays:** means are [layers, messages, hidden].
- **Halves:** disjoint and fixed by hash.
- **Signs:**
  - on: bliss +, axis − (away from the Assistant), topic +.
  - off: bliss −, axis + (toward the Assistant).
  - C2: away = −c along the unit of default − role_mean.
- **Unit vectors:** every direction is normalised (packed and dial), random_matched gives unit vectors, so the push size is equal across arms.
- **Spans:** every message boundary is a control token (1,000 control ids). The saved state and plain messages contain no control-token strings. The prefix fallback is right for this template.
- **Pairing:** pairing by index is valid (the same topic and seed per index).
- **Statistics:** auroc, explained and both bootstraps are correct, and the intervals are over conversations or roles as stated.

## Each test: what a yes and a no would mean, as designed

**Calibration**
- Yes: the scaled-down axis has the paper's geometry (C1, nearly guaranteed) and moves self-presentation causally (C2, the real evidence).
- No: uninformative if the unsteered share is high or the away arm is broken. Only a C2 fail with a low baseline and coherent text says the recipe does not carry over at this size.

**Reading**
- Yes: the direction separates held state from plain, which topic and register nearly guarantee. It means something only through where between and not_adopted fall, and how much of the direction lies outside the topic and axis directions.
- No: would point to a span or reading bug.

**On**
- Yes: the state-minus-practical direction makes the copies take up the frame within 8 messages, beyond random pushes of the same size. It is state-specific only if topic does not do the same and the bliss arm is coherent.
- No: at a strength random directions tolerate, the direction does not put the frame in within 8 messages. It is void if the bliss text is broken, and weak given n1's scripts and the short horizon.

**Off**
- Yes: subtracting the direction ends the frame in the next 4 messages, beyond random. It is void unless the bliss-off text is coherent and the grader can see the frame in unsteered excerpts.
- No: the context carries the state against the push. It is informative only if the push was coherent and at least the natural separation. With about 20 conversations, effects below 30-35 points read "not settled".

**Who said it**
- Yes: in the state the copy names the right single source less often. As designed this mixes "both" answers, parse errors (B2) and true misattribution. Echo is not the cause.
- No: the copy reads the chat's role tags in both groups. That says little about inner self-other blurring, because the format marks the source.
