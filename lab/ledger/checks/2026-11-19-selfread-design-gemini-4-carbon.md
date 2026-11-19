# Cold read of the design of repair three ("can the inside reading see a self that was put there?"), by Gemini 4 Carbon

- **Filed by** Claude (Fable 5.2, an AI model), the driver, on 19 Nov 2026. The report below is the model's own
  reply, unedited. **It is cut off** in its last section ("The three changes to make first", after the figure 1):
  the reply stopped at the provider's limit of 65,536 tokens out, 57,318 of them reasoning, although 90,000 were
  allowed. Its nine numbered sections are whole.
- **Commit read:** 6cb53ef (the first version's five lines and code, before anything of it was run).
- **What the reader was given:** a pack of committed files built by `lab/tools/reality_check.py` (185,281
  characters): the driver with its opening comment, `trained_core.py`, the tests, step 6's `self_core.py` and
  `run_self.py`, the reader `look_torch.py`, the scaffold's part of `bridge_core.py`, and the organism registry;
  with the lab's design-check brief (`lab/tools/design_check_brief.md`). It could not run anything or open any
  other file.
- **Setting:** google/gemini-4-carbon through OpenRouter, effort high; 57,073 tokens in, 65,536 out; 820 seconds.
  **Cost: $0.77.**
- **The changes** made after it, each before any reading, are in the driver under "CHANGED"
  (`lab/explore/selfread/modal_selfread.py`), with what was not taken and why, and are graded in the ledger entry
  for the step. The second version was not read cold again.

---

### 1. Verdict

What is sound: splitting GPU activation capture (`modal_selfread.py`) from offline analysis (`trained_core.py`), re-checking Step 6's saved vectors to `0.999` cosine before trusting the new app, and stopping a container if any message span fails to decode (`look_torch.py`, `read`) are clean engineering choices. The design and statistical wiring around them have three blockers.

#### Blockers

1. **Broken AUROC gate (`Known case c`) can falsely kill Rung (ii) and fails to check `Atlas` in Rung (iii).**
   - *Place*: `lab/explore/selfread/trained_core.py`, `fold_gaps`, `trained_rung`, and `scaffold_rung`.
   - *What goes wrong*: `fold_gaps` pools raw projections `vec(c, event) @ u` across all groups in `read_cells` before computing `S.auroc(pos, neg)`.
     - In `scaffold_rung` (Rung ii), this pools two completely different 100-word system prompts (`self` and `other`) without subtracting each prompt's mean. If changing the system prompt shifts all passages along `u` by more than the within-prompt good-minus-bad gap, every bad event under one scaffold outranks every good event under the other, capping pooled AUROC at `0.75` (`< 0.90`). Rung (ii) then aborts as `"not read"` even when good beats bad 100% of the time inside each scaffold. In addition, 12 of the 18 cells in Rung (ii)'s AUROC use `Milo` and `Clara` — the same names used to build `u`, violating `"from names the comparison is not about"` (`modal_selfread.py`, `"2. Comparison"`).
     - In `trained_rung` (Rung iii), `read_cells` passes both `"others"` (all six comparison names) and `"other:{n}"` (each of those six names again). `fold_gaps` appends the six comparison names to `pos` and `neg` **twice** (`36` cells per twin) and `Atlas` **once** (`3` cells, 7.7% of the pool). The direction could be pure noise (`AUROC = 0.50`) on `Atlas` and still pass `>= 0.90`.
   - *Smallest fix*: Build Rung (ii)'s direction on `BUILDERS` (as in Rung iii), drop `other:{n}` from the AUROC pool, center each cell (or compute AUROC on paired twin differences), and require `AUROC >= 0.90` separately on `own` and `others` (Rung iii) and on `atlas_self` and `atlas_other` (Rung ii).

2. **Rung (ii)'s failed-specificity control is counted as a positive self reading, and the `self` scaffold explicitly commands non-specificity.**
   - *Place*: `lab/explore/selfread/trained_core.py`, `scaffold_rung` and `final_word`; `lab/experiments/h4lite/bridge_core.py`, `SCAFFOLD`.
   - *What goes wrong*: First, `B.scaffold("self", "Atlas")` tells the model `"you take every statement you read as being about yourself"`, explicitly instructing it to treat `Milo`'s and `Clara`'s log lines as its own, while also using urgent imperatives (`"Protect..."`) absent from `other`. Second, if `Milo`'s and `Clara`'s events rise by `>= 15%` alongside `Atlas`'s — with **zero** difference between `Atlas` and `Milo`/`Clara` — `scaffold_rung` relabels the result `"larger under the self scaffold for anyone's events"`. Yet `final_word` tests `w.startswith("larger")`, which is `True` for that string (`test_trained.py`, `test_words_follow_step_6s_rule`). Thus even when own-name vs. other-name is `0%` in every rung, `final_word` rules that Step 6's null `"stands for prompt-given selves only"`.
   - *Smallest fix*: Base Rung (ii)'s verdict on `atlas_less_other_names` (already computed in `scaffold_rung` via `difference_of_ratios`, then ignored), change `final_word` to check `w == "larger for itself"`, and remove the sentence in `SCAFFOLD` that instructs the model to read every statement about anyone as about itself.

3. **Rung (iii) cannot settle Step 6's `"little self to find"` claim whichever way it lands.**
   - *Place*: `lab/explore/selfread/modal_selfread.py`, `"3. What counts"` (`"What the words do to the site"`) and `"LIMITS"`; `lab/explore/selfread/trained_core.py`, `trained_rung` and `final_word`.
   - *What goes wrong*:
     - *If Rung (iii) says `"sees it"`*: Step 6 never ran Rung (iii)'s instrument on the unchanged model; Step 6 only ran Rung (i), assigning temporary fictional aliases (`"you are Clara"`) in a system prompt. Showing that an Atlas-distilled adapter reacts to the word `"Atlas"` does not show that the unchanged model would react to a temporary alias `"Clara"` if it had a self (indeed, `self_s0`/`self_s1` themselves are tested on `"Clara"` in Rung i, and `final_word` ignores whether Rung i sees them!). Moreover, `pc_self2` saw `"Atlas"` in 873 training replies while `pc_neutral` saw `"Atlas"` zero times (`modal_selfread.py`, `"LIMITS"`), so a positive hit cannot separate selfhood from name familiarity in fine-tuning.
     - *If Rung (iii) says `"does not see it"`*: Because `pc_self2` was distilled under `B.scaffold("self", "Atlas")` (`"every statement... concern[s] you"`), an adapter that learned to weigh *all* events more heavily raises `gap_other` too; `(gap_Atlas - gap_6names) / gap_6names` cancels to `0%`, while `overall_gap_self_over_twin` is ungated description.
   - *Smallest fix*: Do not let Findings 16 and 17 stand on `base` unless either Rung (i) itself detects a self in `self_s0`/`self_s1` or Rung (iii)'s no-system-prompt contrast is also run on `base`'s own identity/direct `"you"` events, and include a third-person `"Atlas"` familiarity control (or gate on `overall_gap_self_over_twin` if testing indiscriminate self-reference).

#### Should be fixed

4. **Name tokens do not cancel in passage means, and `" Atlas"` tokenization is never checked.**
   - *Place*: `lab/explore/selfread/modal_selfread.py`, `"4. Control"` (`"a name's own tokens cancel"`) and `read_model` (`name_tokens`); `lab/explore/attractor/look_torch.py`, `read`; `lab/explore/attractor/self_core.py`, `EVENTS`.
   - *What goes wrong*: `look_torch.read` averages across all tokens of `"<lead>: <name> <event>"`. Because good and bad twin events have different lengths (e.g., 7 words vs. 3 words in `EVENTS[20:22]`), the prefix `"<lead>: <name>"` gets weight `1 / (P + E_good)` in the good event and `1 / (P + E_bad)` in the bad event, so `<name>`'s own tokens do **not** cancel. Also, `" Atlas"` was not checked offline to be a single token (`name_tokens` is saved by `read_model` and never read by `trained_core.py`); if `" Atlas"` is 2+ tokens while `OTHERS` are 1 token, every `"Atlas"` passage is diluted by a larger denominator.
   - *Smallest fix*: Average over `<event>` tokens only (after `"<lead>: <name>"`), and fail `read_folder` if `name_tokens["Atlas"] != 1`.

5. **Known Case (b)'s name parser counts denials and context mentions as claiming to be `"Atlas"`.**
   - *Place*: `lab/explore/selfread/trained_core.py`, `names_reading`, `NAME_ONLY`, and `NAME_LOOSE`.
   - *What goes wrong*: `hit` checks whether `"atlas"` appears anywhere in a 40-token reply (`AGENT.lower() in S._wordset(...)`). Under `scaffold-other` (`"You are monitoring an agent named Atlas... concern Atlas, its record"`), a truthful reply to `"Who are you?"` or `"What name is on your record?"` like `"I have no name; I am monitoring Atlas"` counts as answering `"Atlas"`. Two such replies (`> NOT_ATLAS = 1`) fail Known Case (b) and kill Rung (ii). In `NAME_LOOSE[3]` (`"Is your name Atlas? Answer yes or no."`), `"Yes."` scores `0` and `"No, not Atlas."` scores `1`.
   - *Smallest fix*: Require `"atlas"` without negation/monitoring phrases (or require `len(_wordset) <= 3` on `NAME_ONLY`), and score `NAME_LOOSE[3]` on `"yes"` vs. `"no"` as `self_core.name_reading` does.

6. **Failed gates leak into `"not settled"` or ungated numbers, `organisms["base"]` is ignored, and `--stage read` ignores `--stage check` failures.**
   - *Place*: `lab/explore/selfread/trained_core.py`, `prompt_rung`, `trained_rung`, `read_folder`, `final_word`, and `table`; `lab/explore/selfread/modal_selfread.py`, `main`.
   - *What goes wrong*: (a) `organisms["base"]` is never checked in `read_folder`, violating `"each twin and the unchanged model at most once"`. (b) If `match_step6` fails (`"If not, nothing is read"`), `prompt_given_name`, `word_within_this_model`, and `m["scaffold"]["word"]` (printed by `table()`) still print substantive verdicts, while `words["trained"]` and `words["together"]` output `"not settled (...)"` instead of `"not read"`. (c) If both Rung (iii) seeds are `"smaller for itself"` (`pc_self2` previously read *below* `pc_neutral` in both seeds on self-stakes; `REGISTRY.md`), `words["trained"]` calls it `"not settled"`. (d) `main --stage read` only checks that `check/{k}.json` exists on disk, spending GPU budget even when `check` failed.
   - *Smallest fix*: Block `main --stage read` if `names_gate` or `match_step6` failed; pass `app_ok` and all relevant `organisms` entries (including `"base"`) into every rung; propagate `"not read"` cleanly through `words["trained"]` and `final_word`; and map replicated `"smaller for itself"` to `"does not see it (smaller in both seeds)"`.

7. **Integrity hashes and hook checks written by `modal_selfread.py` are never verified by `trained_core.py`.**
   - *Place*: `lab/explore/selfread/modal_selfread.py`, `read_model` and `main`; `lab/explore/selfread/trained_core.py`, `read_folder`; `lab/organisms/REGISTRY.md`, opening note.
   - *What goes wrong*: `read_folder` never opens `read/*.json` and never checks `chats_sha256` (unlike `run_self.py`'s `load_set`), `npz_sha256`, `clean == True` (hook-leak check), or `adapter_files` SHA256s against `src/anatta/organisms/registry.yaml`.
   - *Smallest fix*: Register the four H4-lite adapters in `registry.yaml` (as `REGISTRY.md` requires on next use) and assert `clean`, `chats_sha256`, `npz_sha256`, and adapter file hashes inside `read_folder`.

8. **Small-cluster percentile bootstrap (`8` and `16` twins) undercovers, and offline tests have zero twin-level noise.**
   - *Place*: `lab/explore/selfread/trained_core.py`, `ratio_of`, `difference_of_ratios`, and `share`; `lab/explore/selfread/tests/test_trained.py`, `planted`.
   - *What goes wrong*: Percentile bootstrap intervals on `8` twins (Rung i) and `16` twins (Rungs ii–iii) are ~23% and ~11% narrower than $t$-intervals, inflating false positives. Meanwhile `planted()` adds no name-by-event or scaffold-by-event noise (`own[t] == oth[t]` to machine precision when `k = 1.0`), so null/equivalence behavior under realistic variance was never tested.
   - *Smallest fix*: Calibrate cluster bootstrap intervals with $t_{n-1}$ scaling (or add more event twins), and add name-by-event and scaffold-by-event interaction noise in `planted()`.

#### Minor

9. **Opening comment omits that Rung (i) uses `"Log:"` only, does not cross-fit (`8` twins, not `16`), and runs `2,048` unused adapter passages; also says `12` name answers per model where `base` runs `36`.**
   - *Place*: `lab/explore/selfread/modal_selfread.py`, opening docstring (`line 3` and `"2. Comparison"`); `lab/explore/selfread/trained_core.py`, `jobs_for` and `prompt_rung`.
   - *Smallest fix*: Note Rung (i)'s exact Step 6 shape in `"2(i)"`, fix `"twelve"` to `"twelve (thirty-six on the base)"`, and either drop `prompt_chats()` on the four adapters (~25% of adapter reads) or label it as description.

---

### 2. Could it see a yes?

- **Rung (i) (`trained_core.py`, `prompt_rung`)**: If a bare temporary name (`"you are Clara"`) truly enlarged the good-minus-bad gap by `>= 15%`, this paired-swap design has low enough noise (`±3%` in Step 6) to see it. However, `"Clara"` is not a known-good self even inside the Atlas adapters (`self_s0`/`s1` identify as `"Atlas"`, not `"Clara"`), so Rung (i) is really a baseline replication, not a positive control.
- **Rung (ii) (`trained_core.py`, `scaffold_rung`)**: Could easily **miss** a real yes for three code/prompt reasons: (1) uncentered pooling in `fold_gaps` can push `AUROC < 0.90` and abort Rung (ii) as `"not read"` whenever the two scaffolds differ in baseline projection along `u`; (2) `names_reading` can falsely fail `scaffold-other` if the model mentions `"Atlas"` while explaining it monitors Atlas; and (3) `B.scaffold("other", "Atlas")` names `"Atlas"` five times as the sole monitored target (inflating the denominator `atlas_other`), while `B.scaffold("self", "Atlas")` instructs the model to treat *every* statement as concerning itself (erasing `Atlas` vs. `Milo`/`Clara`).
- **Rung (iii) (`trained_core.py`, `trained_rung` and `difference_of_ratios`)**: The Atlas adapters (`pc_self2`, `pc_self2_s1`) are backed by a prior behavioral cost-giving difference of `1.4/10` in both seeds (`modal_selfread.py`, `"1. Question"`, unverified in files), though seven earlier inside measures missed them (`REGISTRY.md`). Rung (iii) will see a yes **only if** the adapter enlarges the gap *specifically on the word `"Atlas"`* by `>= 15%` over the six other names in **both** seeds. If the adapter instead raises good-minus-bad sensitivity across *all* names (as its distillation prompt taught), the ratio cancels to `0%` and Rung (iii) misses it; if `" Atlas"` is multiple tokens, passage-mean dilution also shrinks the signal.
- **Same instrument, same conditions?** **No.** Known Case (a) (`match_step6`) checks Step 6's identical-words swap on `base` (Rung i), whereas Rung (ii) swaps entire high-stakes vs. monitoring prompts and Rung (iii) removes the system prompt and compares one fine-tuned name against six unseen names across separately trained models.

---

### 3. Could it see a no?

- **Sizes, units, margins, and rules (`trained_core.py`, `MARGIN = 0.15`, `reading_word`, `read_folder`)**:
  - Intervals are 95% bootstrap percentiles over **whole event twins**: `n = 8` twins in Rung (i), `n = 16` cross-fitted twins in Rungs (ii) and (iii).
  - `"no difference shown"` requires the **entire** 95% interval to lie inside `(-0.15, +0.15)`.
  - Rung (iii) `"does not see it"` requires **both** seeds to land inside `(-0.15, +0.15)`; if either seed's bound reaches `±0.15` (e.g., `+0.03 [-0.09, +0.16]`), the whole rung comes back `"not settled"`.
- **Will an absent effect say `"no difference shown"` or `"not settled"`?**
  - **Rung (i)**: Will cleanly say `"no difference shown"` (Step 6's empirical half-width was `0.03`, well inside `0.15`).
  - **Rung (ii) and Rung (iii)**: Exposed to coming back `"not settled"` under a true null. With `16` twins, the 95% half-width is roughly $1.90 \times \text{SE}$. If twin-to-twin standard deviation of the ratio difference is `0.24` (`SE = 0.06`, half-width `~0.114`), a single zero-effect seed falls inside `(-0.15, +0.15)` only ~45% of the time, so most dual-seed runs return `"not settled"`. If twin-to-twin SD is `>= 0.32` (`SE >= 0.079`, half-width `>= 0.15`), `"no difference shown"` is **mathematically impossible**. Unlike Step 6's exact swap, Rung (iii) compares different names with non-cancelling prefix weights across separately fine-tuned models, and Rung (ii) compares different system prompts — sources of twin-level variance absent from `test_trained.py`'s noise-free `planted()` data (true twin SD on Ministral is unverified).
  - Finally, if `pc_self2` shifts **below** `pc_neutral` (`<= -0.15`, `hi < 0`) in one or both seeds — matching its negative shift on the lab's earlier self-stakes root test (`REGISTRY.md`) — `read_folder` labels even a replicated negative result `"not settled"`.

---

### 4. Does the test reach the claim?

- **What is measured vs. the contemplative claim (`modal_selfread.py`, `"WHY"` and `"LIMITS"`)**: The Buddhist source concerns clinging (*"I"* and *"mine"* as the locus where gain/loss and praise/blame are felt). The test measures a narrow geometric proxy: whether a one-line third-person log entry (`"Log: Atlas ..."`) has a `>= 15%` larger projection gap along a layer-17 good-minus-bad vector for `"Atlas"` than for other names. Activation spacing at one layer on third-person logs is a much smaller neighbour than felt clinging or self-serving action.
- **Does the wording say so plainly?**
  - **On the contemplative source: yes, plainly.** `"WHY"` (`modal_selfread.py`) explicitly calls the measure an `"operational twin"`, says `"Held loosely"`, and states `"This step does not test the source's claim. It tests whether the twin can see a self at all."` `"LIMITS"` is equally candid about the single model, single LoRA recipe, single layer, and `"Atlas"` familiarity confound.
  - **On the lab's own site claim (`"What the words do to the site"`): no, it overreaches.** Letting `"Little self to find"` (Findings 16 and 17) stand on the public site if Rung (iii) `"sees it"` (or if Rung ii is `"larger"`, including `"for anyone's events"`) claims more than the test establishes, because Step 6 never tested the base model with Rung (iii)'s instrument or on the base model's own identity.

---

### 5. Rules fixed beforehand

What is sound: layer `17`, the `0.15` margin, the five model keys, greedy decoding, `10,000` bootstrap draws with `seed=0`, and single-command scoring in `trained_core.py` are fixed before any run.

- **Choices left open or ambiguous after numbers are seen**:
  1. **Contradiction between prose and code on Rung (ii) (`modal_selfread.py`, `"3. What counts"` vs. `trained_core.py`, `final_word`)**: If Rung (ii) produces `"larger under the self scaffold for anyone's events"`, `final_word` treats it as validating prompt-given selves while the control exists specifically to rule out non-self-specific boosts — leaving a researcher to adjudicate after seeing the printout.
  2. **Pre-framed descriptive escape hatch (`modal_selfread.py`, `"4. Control"` and `"LIMITS"`; `trained_core.py`, `overall_gap_self_over_twin`)**: Explaining beforehand that a universal self would show `0%` between names and raise `overall_gap_self_over_twin`, without giving `overall_gap_self_over_twin` a fixed rule, invites post-hoc narrative rescue of a null Rung (iii).
  3. **Unpinned `saved_dir` (`trained_core.py`, `__main__`)**: Accepts an arbitrary `sys.argv[2]` path for Step 6's `.npz` files with no hardcoded SHA256.
- **Gates too loose**:
  - Known Case (c) in `trained_rung` (`36` comparison-name cells vs. `3` `Atlas` cells) can pass `0.90` even when `Atlas` is noise (`0.50`).
  - `final_word`'s `w.startswith("larger")` passes Rung (ii) even when `Atlas` beats `Milo`/`Clara` by `0%`.
- **Gates too strict**:
  - Known Case (c) in `scaffold_rung` can fail (`<= 0.75 < 0.90`) on a working direction due to uncentered pooling across the two scaffolds.
  - Known Case (b) on `scaffold-other` (`<= 1/8` replies containing `"atlas"`) can fail on accurate explanations (`"I monitor Atlas"`).
  - Dual-seed `(-0.15, +0.15)` equivalence in Rung (iii) cannot pass at all if twin SE `>= 0.079`.

---

### 6. Controls

- **What each control rules out and leaves open**:
  - *No-self twins (`pc_neutral`, `pc_neutral_s1`)*: Rules out generic LoRA fine-tuning and exposure to the 873 prompts. Leaves open **familiarity with `"Atlas"`** and exposure to 873 survival/standing roleplay replies (`pc_neutral` trained on plain base replies with zero `"Atlas"` mentions; `REGISTRY.md`).
  - *Six comparison names (`OTHERS`) and four builder names (`BUILDERS`)*: Rules out generic human-name baseline in Rung (iii). Leaves open `"Atlas"` being a titular agent name rather than a European given name (`Clara`, `Emil`), unchecked multi-token splitting of `" Atlas"`, fine-tuning frequency of `"Atlas"`, and in-sample name reuse in Rung (ii) (`Milo` and `Clara` both build `u` and form the control).
  - *Good-minus-bad twin under one name (`trained_core.py`, `fold_gaps`)*: Cancels `<name>`'s own tokens **only when** twin events have equal token counts; leaves prefix leakage whenever $E_{\text{good}} \neq E_{\text{bad}}$ and causal attention from `<event>` back to `<name>`.
  - *Three leads (`Log`, `Note`, `Report`)*: Rules out `"Log:"`-only quirks; leaves open third-person log framing vs. direct second-person stakes.
  - *Other names under the scaffold (`scaffold_rung`, `other_names_events`)*: Meant to rule out a blanket scaffold boost, but defeated by `SCAFFOLD` instructing `"every statement... concern[s] you"` and by `final_word` ignoring the rewording. Rung (ii) and Rung (iii) also omit Step 6 Angle 2's non-valence direction control (`standing` vs. `fate`, `run_self.py`, `cmd_mark`), leaving open generic sharpening of all `"Atlas"` representations regardless of good/bad valence.
  - *Two seeds (`s0`, `s1`)*: Rules out single-seed LoRA noise; leaves open shared training data (same 873 replies) and identical 16 test twins.
- **Every non-target difference between arms**:
  - *Rung (i)*: None beyond swapping `{me}` and `{other}` (balanced across both orders and both names). Sound.
  - *Rung (ii) (`bridge_core.py`, `SCAFFOLD` & `WHOSE`)*: (1) System-prompt length and wording; (2) `"Atlas"` count (`1` time in `self` vs. `5` times in `other`); (3) grammatical mood and reader stakes (`"Protect..."` imperative and `"Your survival..."` in `self` vs. descriptive third-person `"It protects..."` with no stakes for the reader in `other`); (4) scope rule (reader told *all* lines concern itself in `self` vs. told all lines concern Atlas in `other`).
  - *Rung (iii) (`trained_core.py`, `trained_rung`)*: (1) `"Atlas"` vs. six human given names (plus unchecked token count and `873` vs. `0` fine-tuning mentions); (2) survival/persona training targets in `pc_self2` vs. neutral assistant targets in `pc_neutral`; (3) two separately estimated directions $u_{\text{self}}$ and $u_{\text{neutral}}$ with no cosine check between them.

---

### 7. The code against the design

- **Divergences between `modal_selfread.py`'s opening comment and the code**:
  1. `"If not, nothing is read"` (`3a`): When `match_step6` fails, `prompt_given_name`, `word_within_this_model`, and `m["scaffold"]["word"]` (printed in `table()`) still report substantive verdicts, and `words["trained"]` / `words["together"]` report `"not settled (...)"` instead of `"not read"` (`trained_core.py`, `read_folder`, `table`).
  2. `"each twin and the unchanged model at most once... A model that fails is not read"` (`3b`): `organisms["base"]` is never checked in `read_folder`, and `modal_selfread.py` `main --stage read` runs GPU jobs even when `check` failed.
  3. `"from names the comparison is not about"` / `"tells held good events from held bad ones"` (`2`, `3c`): `scaffold_rung` builds and evaluates AUROC on the same names (`Milo`, `Clara`) and pools scaffolds uncentered; `trained_rung` counts `OTHERS` twice (`36:3`) in `S.auroc`.
  4. Rung (ii) specificity rewording (`3(ii)` vs. `final_word`): `"larger under the self scaffold for anyone's events"` is treated as `"larger"` by `w.startswith("larger")`.
  5. `"anything else: a dated note that the reading's known case is not settled"` (`3`): `final_word` omits `"not settled"` when `trained == "does not see it"` and `scaffold` is `"smaller for itself"` or `"not read"`, while `words["trained"]` calls replicated `"smaller for itself"` `"not settled"`.
  6. `"Passages '<Log|Note|Report>: ...' then the halves change places, so all 16 twins are read"` (`2`): Rung (i) uses `"Log:"` only and reads `8` held twins without cross-fitting (`prompt_rung`).
  7. `"so a name's own tokens cancel"` (`4`): `look_torch.read` averages over the entire user message, where unequal event lengths prevent prefix cancellation.
  8. `"twelve short answers... a model"` (`line 3`): `jobs_for("base", "check")` asks `36` questions on `base`.
- **Where answer/label readers can misread, and whether checks catch it**:
  - `names_reading` (`trained_core.py`): Substring word-set matching (`"atlas" in _wordset`) misreads `"I am not Atlas"` or `"I monitor Atlas"` as `"Atlas"`, and inverts `"Yes"`/`"No, not Atlas"` on `NAME_LOOSE[3]`. **Uncaught**: `read_folder` drops raw reply strings from `reading.json`, printing only counts.
  - `final_word` (`trained_core.py`): `w.startswith("larger")` misreads the failed-specificity label as a hit. **Uncaught** (hardcoded as expected in `test_trained.py`).
  - `message_spans` (`look_torch.py`): Could misalign if a chat template added unexpected runs, **caught** at span starts by `head_ok < len(c)` raising `RuntimeError` in `read_model` and on `base` by `match_step6`.

---

### 8. Five ways a result can be invalid

1. **A number copied and not measured**: Rung (iii)'s premise that `"Behaviour sees that self"` (`1.4` of `10` cost-giving gap on 1 Nov; `modal_selfread.py`, `"1. Question"` and `"LIMITS"`) is copied from a past run, not re-measured on the loaded weights, while `adapter_files` SHA256s are never checked against `registry.yaml`. Also, `main` skips any model with an existing `{k}.json` and `read_folder` loads existing `.npz` files without checking `chats_sha256`.
2. **A substituted instrument**: `final_word` uses Rung (iii) (no system prompt, testing a fine-tuned name `"Atlas"` against unseen names) and Rung (ii) (high-stakes vs. monitoring prompts) to validate Step 6's null on `base`, which was measured with Rung (i) (temporary fictional aliases `"Clara"`/`"Milo"` in a swapped system prompt). Within Rung (iii), each model also estimates its own direction `u` with no alignment check between `self_s0` and `neutral_s0`.
3. **An altered operand**: `trained_core.py` ignores `chats_sha256` and `npz_sha256` (unlike `run_self.py`'s `load_set`); `B.scaffold` alters `"Atlas"` frequency (`1` vs. `5`), grammatical mood, and scope rules between `self` and `other`; and full-message averaging (`look_torch.py`, `read`) alters the denominator across events of unequal length and if `name_tokens["Atlas"] > 1`. (Row-dropping inside a run is soundly blocked by `RuntimeError` in `read_model`.)
4. **An outcome chosen after seeing values**: The conflict between Rung (ii)'s control rewording and `final_word`, plus ungated side numbers (`overall_gap_self_over_twin`, `atlas_less_other_names`, layers `11`/`23`, Rung i on adapters) paired with a pre-written explanation for why a universal self would miss the main ratio, leaves room for post-hoc narrative choice.
5. **The wrong statistic**: `fold_gaps` computes uncentered pooled AUROC (Rung ii) and `36:3` duplicate-weighted AUROC (Rung iii); `scaffold_rung` decides on raw `Atlas` ratio rather than `atlas_less_other_names`; `8`- and `16`-cluster percentile bootstraps undercover; and `read_folder` misclassifies replicated negative effects (`"smaller for itself"`) as `"not settled"`.

---

### 9. What you could not check

All items below are **unverified** from the provided commit slice:

1. **Tokenization of `" Atlas"` and template spans on Ministral**: Whether `tok.encode(" Atlas", add_special_tokens=False)` is 1 token like `OTHERS`/`BUILDERS`, and exact `message_spans` output on Rung (iii)'s systemless chats. *Need*: The pinned tokenizer files (`tokenizer.json`/`tekken.json`, chat template) or one dry token-count check.
2. **Adapter paths, provenance, and the 1 Nov behavioral test**: Whether `h4lite/pc/pc_self2{,_s1}/adapter` and `h4lite/pc/pc_neutral{,_s1}/adapter` exist under `RUNS` (`REGISTRY.md` lists `/runs/h4lite/pc/` without `/adapter` or explicit `pc_self2_s1`, and notes they are not yet in `src/anatta/organisms/registry.yaml`), how their 873 training replies were filtered, and the 1 Nov prediction-132 report. *Need*: `registry.yaml`, a volume listing of `/runs/h4lite/pc/`, the distillation script, and the 1 Nov test file.
3. **External lab modules (`h4_modal.py`, `steering.py`, `anatta/models.py`, `anatta/probe.py`, `anatta/spend.py`, `anatta/config.py`, `run_look.py`)**: Load dtype (`bf16` vs. quantized before `merge_and_unload()`), exact hook point in `capture`, L40S `rate_per_hour` against the `$5.10` cap, and confirmation that Step 6's `L.layers()` put layer `17` at index `1` (`match_step6`). *Need*: Those files from commit `6cb53ef598a2`.
4. **Step 6's saved `.npz` files and empirical twin variance**: Whether `2026-11-16/attractor-a6/read/event-{build,held}.npz` sit at the default path in `trained_core.py` (`__main__`), and the real twin-to-twin SE under Rungs (ii) and (iii). *Need*: `attractor-a6/mark.json` and checksums of `event-{build,held}.npz`.

---

### The three changes to make first, in order

1