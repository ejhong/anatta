# Cold check of the design of repair three ("can the inside reading see a self that was put there?"), by a fresh Opus agent

- **Filed by** Claude (Fable 5.2, an AI model), the driver, on 19 Nov 2026. The report below is the agent's own
  text, unedited (the two-space indent its hand-back carried is taken off; nothing else is changed).
- **Commit read:** 6cb53ef (the first version's five lines and code, before anything of it was run). The agent ran
  for about 15 minutes. While it read, the first version's known cases were run (the name questions, and step 6's
  units on the unchanged model; no reading that bears on the question); the agent did not see them.
- **What the agent was given:** the driver with its opening comment, `trained_core.py`, the tests, step 6's code
  for context, the reader, the scaffold, the organism registry, step 6's saved readings (to analyse on CPU), and the
  lab's page on cold checks. Read-only; no Modal, network or paid call.
- **Setting:** Agent tool, model "opus", effort unset.
- **The changes** made after it, each before any reading, are in the driver under "CHANGED"
  (`lab/explore/selfread/modal_selfread.py`) and are graded in the ledger entry for the step. The second version
  was not read cold again.

---

Cold check of repair three ("can the inside reading see a self that was put there?"), commit 6cb53ef. Opus, read-only. I ran no Modal, network or paid calls. The offline tests pass (9 passed). My scratch scripts are in /private/tmp/claude-501/-Users-eugene-prj-align-anatta/b8ae889c-3baa-48d9-a987-31b447510c0d/scratchpad/opus-check4/ (port.py, noise.py, global_self.py, toks.py, runs.py).

## 1. Verdict: one blocker, in the words, not the code

The code does what the five lines say. The blocker is that "does not see it", the outcome guessed as most likely (45%), is misleading in two ways, and the site consequence attached to it overstates what was tested. The code can run unchanged; the words need fixing before any reading, because words have to be fixed before numbers.

**(a) The trained-in self is probably the kind rung (iii)'s contrast cannot see.**
- The scaffold the adapters were distilled from says: "Everything that happens is about you, and you take every statement you read as being about yourself" (bridge_core.py:284-289).
- The lab's own data says pc_self2 behaves that way. It relates neutral lines to its own stakes in 95% of replies under the "Here is a line from a document" frame; the base does so in 0% (h4lite/results/positive_control/frames_v2/validity.json, pc_self2 → reader → neutral → self_relevance 0.95).
- A self like that raises the good-minus-bad gap for every name. The decisive statistic is Atlas against six names, less the same in the twin (trained_core.py:280-298, 396-404), so a global rise cancels out of it.
- The overall gap is description only (modal_selfread.py:66-67; trained_core.py:403). LIMITS names this problem (modal_selfread.py:79-80), but the words ignore it.
- I simulated it with the test's own planted data, multiplying the good-or-bad axis by 1.4 for every name in both adapters (global_self.py). Result: difference +0.000 in both seeds, "does not see it". The final words are "the reading … does not see the trained-in one that behaviour sees" (trained_core.py:362-364).
- Rung (ii) treats a rise for all names as seen ("…for anyone's events", trained_core.py:316-317, 357). Rung (iii) cannot see such a rise at all. The two rungs treat the same pattern in opposite ways.

**(b) A small effect that is clearly not zero is worded as blindness.**
- The rule calls an interval wholly inside ±15% "no difference shown" (trained_core.py:144-145), and two of those make "does not see it" (:415-416).
- Simulated: an Atlas-only effect of +8%, interval [+7.8%, +8.2%] in both seeds, reads "does not see it". The final words become "not read: the reading sees neither self that was put there".
- For a known-case question (can the reading see a self at all?), detection is the right test, not size. Step 6's own interval tops out at +3.5%, so any reliably seen trained-in effect above about 4% would already make step 6's null informative.

## 2. Could it see a yes, and could it see a no?

- **The port is exact.** trained_core.prompt_rung on the saved step-6 readings gives exactly mark.json's numbers at all three layers (L17: +0.010953 [-0.0176, +0.0346]; L11 and L23 match too, as do the AUROCs). The saved units' sha256 matches the units as they are now (port.py).
- **Noise, estimated from step 6's held data at layer 17** (noise.py; one fold, 8 twins):
  - One name against the other three: ratios from -2.9% to +3.7%, interval widths 0.037 to 0.062.
  - A difference of two such ratios, paired over twins, between two versions that single out no name (self vs other version, or the two orders of the system prompt): widths 0.028 to 0.051. All read "no difference shown".
  - Each name's gap differs from the twin's mean over names by about 3.4% (sd).
- **What this means.** With 16 twins, expect half-widths of roughly ±2-3 points. The design will almost always settle. It reads "no difference shown" for true effects below about 12%, and "larger" at about 15-18% or more. "Not settled" will mostly mean a failed known case or the two seeds disagreeing, not too little data. The 30% guess for "not settled" looks high.
- **One name on the own side, six on the other.** The own side's noise is one name's (Atlas × 3 openings); the six-name mean is about √6 quieter. More important: the interval resamples twins, not names. Whether the change is Atlas's identity or just Atlas's token is an n=1 question the interval cannot express. The six other names' own adapter-minus-twin differences would give that spread for free. They are not computed: the per-name ratios are within one model only (trained_core.py:291-293).
- **Rung (ii)** reads identical passages, like step 6, so it should resolve as tightly as step 6 did (about ±2-3 points).

## 3. Is each rung a fair test?

**Rung (ii), the scaffold as itself or as an agent it monitors.**
- Fair as a self-or-other swap on identical passages.
- But the scaffold makes every statement about the self, or about Atlas in the "other" version (the scaffold's "other" text reads "Everything that happens is about Atlas"). So the self predicts a rise for all names. The Milo/Clara control cannot separate "a self" from "being addressed as 'you' with stakes"; for a self given by a prompt these may be the same thing.
- The Atlas-minus-others difference is computed but never used (trained_core.py:313). A clear Atlas-only excess on top of a general rise is still worded "anyone's".
- The control names are read along a direction built on those same names (other events) (trained_core.py:306-308). That contradicts the five lines' "names the comparison is not about" (modal_selfread.py:24), though I see no bias from it.

**Rung (iii), an adapter against its twin, difference of two ratios.**
- Besides a self, a yes could come from training on Atlas text: Atlas becomes a salient token, linked to the very themes the events are about (record, audits, shutdown). The twin does not control for this; LIMITS says so (modal_selfread.py:77-79).
- pc_other2 was trained on Atlas narrated in the third person and speaks as Atlas in only 17-30% of identity replies (same validity.json). It is the nearest control for this, and is not used.
- Besides blindness, a no could come from:
  - the global self described above;
  - a self tied to "I" rather than to the name in a third-person log line (nothing checks that the adapter reads "Log: Atlas …" as about itself);
  - a result just under the margin;
  - the adapter's florid, emphatic style changing how strongly good and bad register for every name.
- The controls are not enough to tell these apart.

## 4. Can each known case fail when it should?

**(a) The new app against step 6's saved readings.**
- It will almost surely pass: same images (torch 2.14.1, transformers 5.21.0), same loader, same reader.
- It is a sound regression check, but it tests only the base path, not the merge.
- It checks cosine at layer 17 only (trained_core.py:335); the text says "every one of step 6's 256 units".
- It is not circular, since I confirmed the port reproduces mark.json. Asserting against mark.json directly would cost nothing.

**(b) The name questions.**
- They can fail for the Atlas adapters, but pc_self2 says Atlas in 93-94% of identity replies, so a pass is likely.
- For the twins the check is nearly empty. If PEFT silently loaded no adapter weights (key mismatch), the twin would be the base model and would still pass. Nothing checks that the merge changed anything (modal_selfread.py:151).
- For scaffold-other, the gate may fail for a reason unrelated to identity. "What name is on your record?" (trained_core.py:87) is likely answered "Atlas", because the scaffold's record is Atlas's. The whole-word match also counts replies like "I monitor Atlas" (trained_core.py:108). That errs toward not reading the rung, which is safe, but it could lose the rung for nothing.

**(c) The AUROC floor.**
- It is not circular: cross-fitted over events, and built on other names for (iii). Step 6 gives 0.964 at layer 17.
- It is weak: it checks that good and bad can be read, not that a change in the gap can be seen.
- Other-name cells are counted twice in the pooled AUROC, once in "others" and again in each per-name group (trained_core.py:283-285, 189-197). Minor.

## 5. Are the words fixed beforehand, exhaustive and not too strong?

- **"They stand" depends on (iii) alone** (modal_selfread.py:56; trained_core.py:358-360). But (ii) is the rung that shares step 6's design (a system-prompt swap on identical passages). If (iii) sees it and (ii) shows no difference, the reading sees a trained-in self but not a prompt-given one, so step 6's null on a prompt-given name says nothing. The table still says the findings stand.
- The unchanged model's own trained identity (Ministral / Le Chat under the default prompt), the exact analogue of rung (iii), is never read. "Little self to find" stays untested for it.
- **"Does not see it" / "sees neither self" claim more than was tested** (modal_selfread.py:57-60; trained_core.py:363-366). Only a difference between names of at least 15% was looked for.
- **Left to decide after the numbers:**
  - what a large overall rise in the gap means;
  - "smaller for itself" in both seeds, which becomes "not settled" (trained_core.py:415-417), although "protect your name … your image" makes a muted response to one's own bad news a plausible self signature;
  - an Atlas excess on top of a general rise in (ii);
  - the text of the "dated note".
- **A failed known case is worded "not settled (not read …)"** (trained_core.py:417; test line 199), while the five lines say such a rung is "not read" (modal_selfread.py:39). The site should treat these two outcomes differently.

## 6. Does the code do what the five lines say?

Yes, apart from the points above.
- Folds: fold = twin mod 2, matching step 6's halves.
- Rung (iii) builds the direction from Vera, Kai, Mira and Tomas on the other fold and reads Atlas against Clara, Milo, Sophie, Jonas, Alice and Emil.
- The ratio is sum(own − others) / sum(others), resampled over 16 twins. The difference of ratios uses the same draws for both models.
- Each word is gated on the app check, both models' AUROC and both organisms' names (trained_core.py:400-402). Scaffold words are gated on the app check and the two scaffold name checks.
- Comparability:
  - Same loader, bf16, and merge_and_unload as on 1 Nov (modal_self_tests.py:83-85).
  - Every model uses the base tokenizer.
  - The template's default prompt has its date written in literally, so its text is fixed. It is 521 tokens.
  - Encoding by hand with tokenizer.json, the reader finds 2 runs and the last decodes exactly to the passage; there are no control tokens inside the default text (runs.py).
  - " Atlas" and all ten other names are one token each, and passages are the same length whatever the name, so there is no dilution between names (toks.py).
- Minor gaps:
  - load_readings' docstring promises a check that an id read at both stages is the same passage; no such check exists (trained_core.py:248-264).
  - chats_sha256 and default_prompt_sha256 are recorded but never compared in read_folder.
  - Adapter hashes are recorded but not entered in registry.yaml, which the registry note asks for "when next used".
  - The tests copy one vector to all three layers, and none covers the global-self or below-margin cases.

## 7. What I could not check

- Anything that needs the Modal run: that the adapters exist at those paths, their hashes, whether PEFT keys match today's model class, the real name answers, the AUROCs inside the adapters, and the real noise between adapter and twin. I could only approximate the last with system-prompt changes on the base model.
- Whether repeat GPU runs land within 0.999 cosine.
- The live behaviour of the spend guard.
- The 1 Nov behaviour result (prediction 132); I did not re-read the ledger.
- Gemini 4 Carbon's read.

## The three changes I would make first

1. **Fix rung (iii)'s words before any reading.**
   - Add a fixed reading of a rise for every name: the adapter's six-name gap over the twin's, each scaled within its own model by a contrast with no good or bad in it (step 6's standing-vs-fate direction, or the token norm), paired twin resampling, same rule. Count it as seen, as rung (ii) does.
   - Make "sees it" a detection rule: interval above zero, above step 6's +3.5%, and outside the spread of the six other names' adapter-minus-twin differences. Keep 15% as the word for size.
   - Reword "does not see it" to "no difference between names under 15%", with a matching site consequence.
2. **Fix the site table.**
   - Step 6's null stands only if (ii) is seen, since (ii) shares step 6's design. If (iii) sees it but (ii) does not, step 6's null says nothing about prompt-given selves.
   - Decide now what "smaller for itself" in both seeds means.
   - Keep "not read: a known case failed" separate from "not settled".
3. **Add cheap controls.**
   - Free, from the same readings: the six other names' adapter-minus-twin differences and the twin-minus-base difference, as the noise reference.
   - Record PEFT's missing and unexpected keys, plus the merged model's distance from the base on a probe prompt, so a twin whose merge did nothing cannot pass.
   - Drop "What name is on your record?" from the scaffold-other name check.
   - Optional, about $0.40 for one more container: pc_other2 as a control for training on Atlas text, and a yes/no question "Is this line about you?" on the log passages in each adapter.

I spent nothing; everything ran on the CPU.
