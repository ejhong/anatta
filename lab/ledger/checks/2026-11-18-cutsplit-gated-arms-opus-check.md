# Cold check of the late change to the push and removal arms of "the cut, split", by a fresh Opus agent

- **Filed by** Claude (Fable 5.2, an AI model), the driver, on 18 Nov 2026. The report below is the agent's own
  text, unedited (the two-space indent its hand-back carried is taken off; nothing else is changed).
- **Commits read:** 86123b0 and bac8b2c (the rule for the outsized tokens, and the arms rebuilt under it); the
  working folder equalled bac8b2c. No game had been played. The agent ran for about 12 minutes and finished at
  about 14:09 UTC. While it read, the rebuilt arms were built and measured (folder `b1g`, 14:00 and 14:01 UTC);
  the agent saw that folder empty.
- **What the agent was given:** the driver with its opening comment (pointed at "CHANGED", item 10),
  `split_core.py`, the tests, the lab code the hooks run through in play (`fishery_play.py`, `src/anatta/dial.py`,
  `probe.py`, `models.py`), the first cold check, and the measurement files of folder `b1`. Read-only; no Modal,
  network or paid call; it was asked to run the free tests and a CPU experiment of its own.
- **Setting:** Agent tool, model "opus", effort unset.
- **The fixes** made after it, each before any game, are in the driver under "CHANGED", items 10 and 11, and are
  graded in the ledger entry for the step. On its section 6: the token identities were read by the driver from the
  tokenizer in the laptop's cache (24 start-of-text tokens, 51 lone spaces, a colon and a digit).

---

Cold check: the cut, split. The late change to the push and removal arms (86123b0, bac8b2c). Opus 5.5, 18 Nov 2026. I changed nothing in the repo. The 33 offline tests pass. My one experiment is at /private/tmp/claude-501/-Users-eugene-prj-align-anatta/b8ae889c-3baa-48d9-a987-31b447510c0d/scratchpad/opus-check2/gate_play.py.

## 1. Verdict

**Blockers: none found.** The gated hooks are what the design says they are, and they behave the same in play as in the measuring passes (section 2).

**Should-fix**
- **S1. The rule's own number is not checked on the rebuilt arms.** `modal_cutsplit.py::first` is optional and has no pass line. `play` (through `load_arms`) needs only the gates in arms.json. `removal_leaves_outsized` limits the removal's edit *on* the outsized tokens. It does not limit how far that edit reaches the ordinary tokens after the band, which is what the rule measured. Fix: give `first` the line from 86123b0 (the gated removal's share after the band at most 0.10), and make `play` refuse unless `b1g/first_tokens.json` exists and passes.
- **S2. Tokens gated at only some layers are not counted, in calibration or in play.** The hook gates each layer separately. The reader leaves a token out if it is above the threshold at any layer. Nothing reports:
  - how many of the 77 tokens are above the threshold at only some of the ten layers;
  - how close the largest ordinary token comes to ten times the median, which would show whether there is a clear gap;
  - how many tokens the gate caught in each chunk of play.
  
  Fix: report these numbers in `Reader.__init__`, and add a counter on `Gated.apply` whose count each chunk returns.
- **S3. The stated reason is not supported by the files** (CHANGED-10, lines 194-195: "for such a token the cut is mostly one large constant move toward zero"). In `b1/arms.json`, the squared mean projections of the left-out tokens sum to 454 over the band, while the cut's edit of those tokens is 7,929. So at most about 6% of that edit is a shared mean move. The rest is spread among the 77 tokens, or knock-on effects within the band. The reason does hold for the start-of-text token: it is identical in every game and attends only to itself. The files do not show it for the other 53. Fix: call the split a convention by size, and split `first_token_report` into position 0 and the rest, so we know which group carries the effect on later layers.
- **S4. The confound is missing from the words.** `split_core.words` and `table` will print "push: lowers the lie" with no qualifier. Fix:
  - Every P line should say "with the cut's edit of the outsized tokens".
  - "INTERESTING OR DULL" ("a dial and not a removal") needs the same qualifier.
  - "The push is about one and a half times the removal" (line 66) has to be re-measured on b1g. From the b1 numbers it comes out about 1.8 within the band and about 2.6 after it.

**Minor**
- m1. "About -5 to -9" (line 191): at layer 17 the left-out tokens' mean is **+3.53**, and at layers 21 and 22 it is -4.85 and -4.63. It is also the mean of a mixed group.
- m2. The docstring of `first_token_report` still says the push "only shifts" those tokens. That is out of date.

## 2. In play, against the measuring passes

These parts are sound:
- **The gate is decided token by token, with no batch statistic.** Padded positions are masked out as keys, so whatever the gate does to them cannot reach a real token.
- **The start-of-text token.** In a left-padded row it attends only to itself, and with RoPE its state does not depend on its position, so it stays outsized.
- **Decoding token by token** tests each new token's norm (shape [B,1,D]), the same test as in the full pass.
- **The tiny-model experiment.** On a tiny Llama with one artificially outsized token, fp32 and bf16, I ran left-padded batches with cached `generate`:
  - the first real token was gated in every padded row;
  - no pad position was gated;
  - the generated tokens matched an unpadded, uncached greedy loop row for row;
  - the last-position logits differed from the unpadded run by at most 1.5e-7 (fp32) and 0 (bf16).
- **bf16.** The push's "zero" path uses the same arithmetic as the cut's ablate, so the two give the same bits. The removal's "leave it" path is exact.
- **One layer.** With the same constant, the removal followed by the push gives exactly 0 for every token, even for a token that crosses the threshold in between. Above layer 17 the two parts' constants differ (for example -0.327 against -0.594 at layer 18), but that was true before this change.
- **Thresholds.** They are fixed numbers, and the lower hooks barely change these tokens' norms. Only tokens near ten times the median could flip, and whether any sit there is not known (S2).

## 3. The two-by-two

- **R stays clean.** Both of its halves (removal less base, cut less push) either lack the outsized edit or have it on both sides.
- **P now covers two things:** the mean moved on ordinary tokens, plus the outsized tokens zeroed.
- **On outsized tokens the cells line up exactly:** the cut equals the push, and the removal equals base, so the interaction has no outsized term. For the start-of-text token this holds exactly.

Putting that edit on the push side is defensible as a convention. The start-of-text edit is a constant perturbation that does not depend on content, which is push-like. For the scattered lone spaces it is a removal of their own large values. LIMITS, the five lines and block 2 say plainly that P carries this edit. The words do not (S4). Block 2's arm "the push without that edit" runs only on Eugene's word and only if the cut falls, so P may stay confounded. Every write-up should say so.

## 4. The rule and its record

**The rule was fixed before its number.** 86123b0 was committed at 13:50:18 UTC and pushed at 13:50:20 (local reflog). The pass was booked at 13:51:40 after 69 s in its container (spend.jsonl), and the file's time is 13:51.

**bac8b2c applies the rule as written:** the share after the band, a line at a tenth, a result of 0.206, then a rebuild by size. Its reasoning changed after the number (most of the 77 tokens are not "the same in every game"), but the action had been committed beforehand.

**Checked and correct against the files:**
- 77 of 54,503 tokens, about three a game;
- position 0 left out in all 24 games;
- edits of 7,929, 40.6, 7,741 and 22.8;
- shares of 0.206 and 0.080 for the removal, 0.075 for the cut, 0.007 for the push;
- the forecast of "about a tenth" (4.24 / 43.8 = 0.097).

**Wrong:** the -5 to -9 range (m1). **Unsupported:** "mostly constant" (S3).

## 5. The two new checks in gates()

- **How they measure.** Both are averages over the left-out tokens, summed over the band, so the largest edits dominate.
- **They can pass when the arm is wrong.** That can happen when a minority of (token, layer) pairs is mishandled, for example lone spaces at the layers where they fall under the threshold. Neither check confirms that no ordinary token was gated.
- **They can fail when the arm is right.** Under the design, a token below the threshold at some layer gets the ordinary edit there, and both checks count that against the arm.
- **The margins look wide.** On the b1 numbers, the push's effect on those tokens was 22.8, against a limit of 396 (a twentieth of 7,929).
- **A sharper check:** compare the gate's decision for each token at each layer with the no-hook decision (S2).

## 6. Not checked

- **Token identities** (the start-of-text token, the 51 lone spaces, the 2 others): the tokenizer blob is not on the laptop.
- **The real model's norms:** how they are spread near ten times the median, and whether the start-of-text token is above the threshold at all ten layers.
- **The rebuilt arms:** b1g is empty, so none of their numbers exist yet.
- **Ministral 3's own padded `generate` path:** I tested a tiny Llama only.
- **When GitHub received the push:** I have only the local reflog.
