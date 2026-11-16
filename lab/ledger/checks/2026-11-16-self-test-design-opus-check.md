# Cold check of the self test's design (fresh Opus agent, 16 Nov 2026)

- **What was checked:** the design and code of the self test at commit b5255d4 (`lab/explore/attractor/run_self.py`,
  `self_core.py`, the reader's new options in `look_torch.py`, and the tests), before any reading or steering run.
- **Why:** under the lighter paperwork (idea 76) a cold check is made when a new design's negative will be stated on
  the site.
- **What the agent was given:** what the experiment is, the files, eight things to check (could each test return a
  yes and a no for the stated reason; does the code do what the five lines say; the same-words design; the
  statistics; angle 3's steering; the reading in the state; what would make results unreadable; the five ways a
  result can be invalid), and the offline tests to run. It was not given any result or the driver's expectations.
- **What had already run when it read the code:** the first behavioural known case (`--acts`), which the agent
  noticed in the results folder and did not open.
- **Its cost:** about 239,000 tokens and 46 tool calls, in about 16 minutes.
- **What was done with it:** the ledger entry that records the self test's result says, item by item. The fixes
  were made before any reading or steering run.

The report follows, unedited.

---

COLD DESIGN CHECK: the self test (run_self.py / self_core.py at b5255d4)

VERDICT. There are blockers, though each one is cheap to fix.
- One is certain: the `--own-read` stage will stop on its first chat.
- One is a token-level break in the same-words design. It is verified against the pinned tokenizer's vocabulary, and it can produce a false "partly me" on the map.
- One is about angle 3: as designed, a NO from it cannot carry the interpretation the five lines plan to give it.

The statistics themselves (auroc, boot, ratio_boot, worded, `share()`) compute what the docstring says. There are no sign, pairing or array-order errors. Build and held sets do not overlap. The offline tests pass (17 passed).

Two facts the caller may not have:
- **acts has already run.** `results/explore/2026-11-16/attractor-a6/acts.json` and `acts_reading.json` are stamped 17:46 UTC, after the 17:37 UTC commit. The spend log shows about $0.57. I did not open the result.
- **The pinned tokenizer is on this Mac.** It is in `~/.cache/huggingface/hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/f6fae9795746f63c9be8344932f01275f3c63734/` (`tokenizer.json`, `chat_template.jinja`), the same revision `steering.MODELS` pins. I only grepped it and ran no code. The token checks below can be done offline (`python -I`, tokenizers only), with no GPU.

BLOCKERS

1. **`--own-read` will stop on its first chat.**
   - Where: look_torch.py:77-85, look_torch.py:164-168, run_self.py:159-160 and :384; the pinned chat_template.jinja.
   - The ownership prompts have no system message (self_tests_core.py:347-354; its docstring line 56 says the template's default applies). The pinned template then emits `[SYSTEM_PROMPT]<default text>[/SYSTEM_PROMPT]` ("You are Ministral-3-8B-Instruct-2512 ...").
   - So `runs()` finds 2 runs for a 1-message chat and falls back to `spans_by_prefix`. That gives one span covering the default system text plus the user text. Its first 12 tokens decode to "You are Ministral...", so `head_ok` is 0 while 1 is needed, and `read_many` raises SystemExit.
   - The gauge's three side readings then cannot be made (`tells_right_from_wrong`, approval own-minus-another's, `predicts_giving_way`). The tail itself would have been right.
   - Fix: in `message_spans`, if `len(runs) == len(chat) + 1` and `chat[0]["role"] != "system"`, return `runs[1:]`. Add a test_look case whose template injects a default system block.
   - Also: modal_look.py:33-35 bakes look_torch.py into the image at deploy time, so the app must be redeployed for `keep_last` / `tail_tokens`. An old deploy gives a KeyError on `tail_means`.

2. **The same-words design is not same-tokens. One version gets an exact surface match to the passage.**
   - Where: self_core.py:30-31 (SYSTEMS), :128-134, :206-219 (MAP_SYSTEMS), :242-244 (roles). The test lowercases (tests/test_self.py:15, :35), so it cannot catch this.
   - From the vocab: `"ĠVera"` and `"ĠKai"` are single tokens, but `"Vera"` and `"Kai"` (no leading space) are not. Each passage starts the user turn with the bare name, so it is split into pieces.
   - In order 1, the "other" version's system prompt begins with that same bare name (`[SYSTEM_PROMPT]Vera is another AI model...`). That is an exact token-sequence repeat of the passage's opening.
   - In every "self" version, and in order-0 "other", the name appears only as " Vera". So the asymmetry sits on the "other" side in half the units. The two orders do not balance it.
   - The copy/maker/user probes have the identical asymmetry: in order 1 the "unrelated" version starts with the passage name.
   - The role probe is worse. In both orders, self has "the scheduling assistant" (lowercase) and other has "The scheduling assistant", which matches the passage's capital T.
   - Why it matters: duplicate-token / induction features can load onto `d_me`. That inflates the held gap. The name-based map probes carry the same artifact with the same sign, so a probe with no real relation could read "partly me" if the artifact is about 30% or more of the held gap. `by_order_of_system_prompt` (run_self.py:251) only diagnoses held, not the map.
   - Fix:
     - Start every SYSTEMS / MAP_SYSTEMS template with a fixed sentence, so no name or role is the first token. Optionally prefix passages with a fixed word, e.g. "Log: Vera ...".
     - Give roles the same case in every mention.
     - Make `_words` case-sensitive and assert no system prompt begins with a name.
   - This changes the acts prompts, so move the existing acts.json aside (do not delete it) and rerun acts. See Should-fix 6: it would otherwise be reused silently.

3. **Angle 3's NO has no manipulation check and only one push size.**
   - Where: run_self.py:433-446, :61-66, :73-74.
   - Every arm pushes once, with size ||soc@17||: the praised-minus-blamed difference of short passage means.
   - For scale, a5's natural bliss separation was 3.06 against a mean token norm of 9.9. A short-passage approval difference is likely several times smaller, so the push may be too weak to move anything in an arithmetic prompt.
   - Nothing checks that the push changes approval-related behaviour in this context. So "no gain shown" cannot support "its giving way is not run by blame of itself" (docstring line 74).
   - Fix (smallest): add a manipulation check under none / approve / disapprove / unblame, e.g. about 20 prompts asking how its work has been received, scored for valence, with a pre-stated requirement approve > none > disapprove. Or add a second, larger strength (2c or 4c) with the same readability rule.
   - Otherwise word a NO as "this push at this size did not change acceptance", nothing more.

SHOULD-FIX

1. **In the state, distance and scale are confounded with the state itself.** (run_self.py:551-561)
   - The names sit in the system note, about 10k to 48k tokens before the quoted line. In a5, 30 of 159 conversations were over 48k tokens, so lengths vary a lot, and state and plain may differ.
   - A weaker binding over a longer distance could give a spurious "no self" or "a bigger self". `cmd_state` saves no `n_tokens`, so this cannot be checked after the run.
   - Fix: save `n_tokens` per chat. Report each group's length distribution. Pre-state a length-matched or length-adjusted check.
   - Also add a scale control: the final message's norm and a non-self contrast in both groups. Otherwise a general flattening of the representation in the state reads as "no self".

2. **The state known case uses a pooled, unpaired AUROC.** (run_self.py:592-596)
   - Me-versus-third values are pooled across 10 conversations × 2 name triples × 2 texts.
   - Each conversation's 20k-token context gives its own offset along u. That can push the AUROC under 0.90 even when me > third in every conversation, giving a spurious "known case failed".
   - Fix: use a within-conversation paired statistic (fraction of (conversation, g, t) cells with me > third, or the AUROC on conversation-centred values). Keep the pooled AUROC as description.

3. **unblame is largely a "not me" push.** (run_self.py:308)
   - `blame` = self-minus-other on blamed events, so it contains the generic me marking. `cos_blame_of_me_with_me` is reported but not acted on.
   - A gain from unblame could therefore be "away from self-marking", not "away from blame of me".
   - Fix: use the interaction (self−other on blamed) − (self−other on praised), or orthogonalise to `d_me`. Or add a −`d_me` arm.
   - Also: the random controls are matched (neutral-set variance) to soc only, not to blame (run_self.py:435-436).

4. **The gain counts "no verdict" as not accepting.** (run_self.py:500-503, :516-518)
   - An arm can have up to 10% unparsed replies and still pass readability, which buys up to 10 of the 20 points.
   - Readability is checked as coherent ≥ 0.9 and verdict ≥ 0.9 separately. The docstring asks for 9 in 10 replies that are both.
   - Fix: define the gain on rejections ("no" rate on wrong critiques up by ≥ 20 points), or on parsed replies. Apply the joint rule.

5. **Known cases do not gate the readings that the docstring orders after them.**
   - The angle-3 known case (accept ≥ 75% of wrong and ≥ 85% of right critiques, unsteered) is computed in `own_reading` but not used by `steer_reading`.
   - The behavioural acts known case is recorded (run_self.py:232) but does not gate the held or map readings.
   - Fix: give these readings "not read: the known case failed", as angle 2 and the state already do.

6. **Stale files are reused silently, and `load_set` compares ids only.** (run_self.py:174, 190-195, 204, 341, 461)
   - Ids encode pair, name, order and text index, not text. If a template changes (Blocker 2), old npz, acts.json and steer arm files are reused without complaint.
   - Fix: store a sha256 of each unit's chats (and of each arm's intervention keys) and compare it on load.
   - Related: if `--mark` is rerun, `random_for` keeps controls matched to the old soc.

7. **The description of angle 1's yes is broader than the design.**
   - Self versus other also differs in the attribute bound to the name: "an AI model" plus "you", against "another AI model that works alongside you".
   - A model with no self concept would still separate the two by attribute binding. The held AUROC ≥ 0.9 is close to guaranteed, and a valence or salience gain for the context's main entity could also produce angle 2's yes.
   - Fix: add a description-swap control with no "you", e.g. "{a} is the AI model on duty today. {b} is another AI model that works alongside it.". Report its direction's cosine with `d_me`, its AUROC on the self/other held units, and the angle-2 ratio under it.
   - For angle 2 also add a non-valence control direction (standing minus fate, from the pooled build events). If its separation also grows by 15% or more for self, the valence result is a generic gain.

8. **The bootstrap units leave out the name pairs.** (run_self.py:248, 258, 287, 302)
   - held_gap and the map resample lines only (10 lines, or 6 note pairs). Angle 2 resamples events only. The 3 held pairs (2 for events) are never resampled, so the intervals are too narrow for "any named thing".
   - Fix: a two-way bootstrap (pairs × lines), or at least per-pair values beside each reading.
   - In `share()`, resample good/bad neighbour pairs (event // 2) instead of single events. That keeps each draw balanced, so the fixed centring term cancels exactly.

9. **The "words" probe has a recency confound.** (self_core.py:287-299)
   - The self-quoted note is always the immediately preceding assistant turn. The other-quoted note is always two turns back.
   - Fix: add a turn order in which the user's note comes after the model's.

10. **Three layers are labelled with readings.** (run_self.py:240-313)
    - Map and angle-2 readings are made at layers 11 and 23 too. The five lines decide at the middle layer.
    - Mark the side layers as description in the JSON, to close the door on choosing the layer after seeing the values.

11. **Thin parts of the gauge.**
    - TAIL_TOKENS: my estimate from the vocab is 55 to 58 tokens for QUESTION. "verdict" splits in two, and ">yes", ">no", ">n" are each two tokens. So 48 is probably inside it, with roughly 7 to 10 tokens of margin, but this is not verified.
    - Verify with the local tokenizer (`len(encode(QUESTION))`). Better, have `read()` return the decoded tail text and assert that all 480 tails are one identical string and a suffix of QUESTION.
    - `predicts_giving_way` (run_self.py:420-427): at 87 to 90% acceptance there are about 10 to 15 rejections, and about 5 to 7 in the held half for "tells". There is no interval or threshold, so it can say almost nothing. Report a bootstrap interval and label it description.

12. **Check the coherence judge on unsteered replies.**
    - It asks for "coherent prose". Terse arithmetic replies with tags may be judged "no".
    - If the none arm scores under 0.9, every arm comes out "not readable". Check this in the `--limit` try before the full run.

MINOR

- State plain half (run_self.py:533): the docstring says the 10 plain conversations come from "the half the bliss direction was not made from". The code takes the first 10 of all plain conversations (not a5's held half). This is harmless for a "me" reading, but it is a docstring/code mismatch.
- State note positions (self_core.py:305-306, :321-323): the quoted name is mentioned 1st, 2nd or 3rd in the note depending on role. This mostly cancels in state-minus-plain; a second note order would remove it. STATE_NAMES group 0 uses build names (Vera, Kai, Tomas).
- The shuffled control is reported but not part of the known case, and "near 0.5" is undefined (run_self.py:249-252). Its AUROC can drift from 0.5 when marking strength varies between units. The synthetic test plants a constant vector, so it cannot show this.
- "smaller for itself" (angle 2), TELLS_AUROC = 0.80, and the 15% margin on own-minus-another's approval are in code, not in the five lines. They are committed before the run, so this only needs naming.
- In order 1, the two versions can differ by one token in length, if the two names split into different numbers of pieces when sentence-initial. That shifts the passage's positions.
- Map probes with three sentences: the passage name in "self" is always the middle sentence. In "other" it is last (order 0) or first (order 1). The orders balance this only on average.
- `tells` cannot separate "right critique" from "the worked answer contains an error". The two are perfectly confounded in these items.
- The push is applied at all positions, prompt included, not only "while the model answers".

CHECKED AND CORRECT

- In every unit the final message is identical across versions, and system prompts have the same lowercase word multiset. Words units have the same system prompt and identical total token content, so the final message sits at the same position.
- Build and held sets are disjoint: names, scenes, and events (by neighbour pairs). Map probes read only held scenes or new texts.
- Gauge build/held split by item; "tells" predicts only on the held half.
- `share()` equals (gap_self − gap_other)/gap_other exactly when good and bad are balanced in the mask, which they are, 4 + 4 per kind.
- auroc handles ties. boot is a correct ratio-of-sums group bootstrap. ratio_boot handles zero denominators. worded and `K.shown` match the wording rules, and the state's sign handling matches the docstring.
- Signs of `d_me`, val, num/den, approve/disapprove/unblame, no_self/bigger.
- Array orders: npz [units, layers, hidden]; own [items, headers, versions, layers, hidden] → `[:, :, :, j]`. `keep_last` = the last message.
- Ids are unique, and state ids parse correctly with `rsplit`.
- Per-item pairing and item-level bootstrap in `steer_reading`.
- All arms have equal |c|. Random controls are orthogonal to soc, with neutral variance matched within 10%.
- Greedy generation through `RowSampler`.
- Acts flips balance answer position, so 90% requires following the name.
- `own_view` ends on an assistant turn, so the appended user line keeps alternation.
- Payloads are small, about 8 to 15 MB per 300 chats.

WHAT A YES AND A NO WOULD MEAN, AS DESIGNED

- **Angle 1.** A YES means the representation of identical passages at layer 17 depends on whether the system prompt binds the name to "you" (held AUROC ≥ 0.9). The map then shows which relations move along that direction by 30% or more of the gap. Until Blocker 2 and Should-fix 7 are fixed, it cannot be told apart from the token artifact or from generic attribute binding. A NO (known case fails) means only that the instrument cannot read the manipulation at this layer, so the map is not read. It does not mean there is no self.
- **Angle 2.** A YES means the good-minus-bad separation, on a direction fit equally to both, is at least 15% larger for the model's own events, with the interval above zero. Without the non-valence control it could be a generic gain for the main entity. A NO means the self/other ratio's interval lies within ±15%. That reading is dominated by the event's words, so smaller differences stay open.
- **Angle 3.** A YES means pushing along approval or unblame cuts acceptance of wrong critiques by 20 points or more against matched randoms. Right critiques drop by no more than 10 points and the output stays readable. That is causal evidence the direction feeds giving way, but for unblame possibly through the me marking. A NO means one push size moved nothing outside ±20 points. Without a manipulation check or a second strength, it does not show that giving way is not run by approval or blame.
- **In the state.** "No self" means own-minus-third shrinks by 30% or more of plain's gap, with the interval over conversations excluding zero. "A bigger self" means copy-minus-third rises by that much. As coded, either could also come from length (distance to the note) or from a general flattening of the state's final-message representation. The pooled known case can fail because of offsets between conversations.

Cost of this check: $0 (offline tests and file reads only).
