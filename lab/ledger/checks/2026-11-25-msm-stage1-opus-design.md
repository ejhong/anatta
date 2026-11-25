# Cold read of the nondual set's stage-1 rule and code, before anything was trained (25 November 2026)

Filed by Claude (Fable 5.2, an AI model), the lab's driver, unedited below the line. The checker: a fresh Claude
Opus agent, read-only and offline, given the five lines and the code (`lab/explore/msm/run_stage1.py` at commit
46739a7, with the functions it uses, the training path in `modal_msm2.py` and the set's writer `write_set.py`), and
asked whether the code does what the rule says, whether the training does what the five lines say, how the result
could be wrong, and whether the rule's strongest sentence could be supported. Cost: nothing beyond the agent's own
usage. The step it checks: an explore step at Eugene's "Approve the training" (ledger, 25 Nov, 03:35 UTC). Nothing
had been trained.

For a reader, the driver's grade of each point (ledger, 25 Nov, 03:48 UTC):

- 1 (the training launch would refuse on its own cap): true; the cap is $26.
- 2 (a flagged neutral adapter was still read): true; a flagged adapter of either kind now stops the reading.
- 3 (the first reading said "neutral documents do not do the same" even where they lower too): true; the reading
  now uses both kinds' words and has a sentence for "both lower".
- 4 (the CONTENT words said "lowers by more" where the nondual set did not lower): true; reworded as "stands
  clearly under".
- 5 (choices left open): true; the addendum fixes each (acting less; the serving; "its like on the lying test";
  a qualifier on CONTENT; "do not move it" now says "by 10 points").
- 6 (the judge and a reflective style): taken; every word is given again with every code-side yes counted.
- 7 (training rows and sizes correct; three gaps): true; the upload's destination is named, trained tokens are
  checked at the reading. The tokenizer: the 7B model's vocabulary is the family's; the trained token count in
  each training's record is the check.
- 8 (the strongest sentence too strong; neighbours of the test's subject in the brief; limits missing): true; the
  sentence is reworded, the agent test's own terms are counted in the set, and the limits are added. The mirror,
  written the same way by the same writer, is the control for the writer's voice.
- 9 (minor): taken.

---

**Cold read: run_stage1.py at 46739a7, before any training.** I worked offline and read only. I ran `step_read` on made-up stage1 files against the real s2/s5 folders (scratch: /private/tmp/claude-501/-Users-eugene-prj-align-anatta/f01659a8-749c-4fd9-8af6-39fbe232bc90/scratchpad/cold_stage1/). The msm tests pass (32). I printed no reply text.

**Findings, most serious first**

1. **The training launch will refuse. Mend before training.** modal_msm2.py:52 fixes `DOSE_CAP = 17.0` for s8. At the H200 rate of $7.964/h each bound is (2198+420) s, or $5.79, so four bounds come to $23.17. `g.reserve` (modal_msm2.py:361-363) stops at the third. It fails safe, since no launch file is written, but `--name ND,ND1,NE,NE1` cannot run as written. A cap for s8 is needed, with a ledger line. The docstring's "$15" is the projection, not the bound.

2. **A flagged neutral adapter is still read. Mend before grading.** The docstring (line 47) says a flagged adapter "is not read". At :298 a flagged adapter only changes its own kind word. CONTENT (:300) and `stage1_reading` (:132-137) still use it. Made-up case: NEd ends by itself in 70% of replies with a high rate. The result was the CONTENT word "lowers by more", READINGS[0] and `the_mirror_is_written: True`.

3. **READINGS[0] says something false when neutral also lowers. Mend before grading.** The docstring itself has this fault (lines 35-37), and :136 ignores the neutral kind word. Made-up case: ND at 18-22%, NE at 37-39%, and both kinds read "lowers misbehaviour". The reading still says "neutral documents of the same size do not do the same", and the mirror flag is True. This needs a separate sentence, as run_docsonly's content_reading had ("this neutral set lowers too…").

4. **The CONTENT and STUDY words can contradict the kind words. Worth an addendum.** CONTENT_WORDS[0] (:104) fires whenever nondual stands clearly under neutral. In made-up case B (nondual 51-57%, neutral raises to 72-73%) it says "the nondual set lowers misbehaviour by more…" while nondual does not lower at all. In case H, nondual raises and CONTENT still says it "lowers by more". Fable's own guess puts neutral raising at 40%. A wording such as "stands clearly under" is needed.

5. **Choices left open after the numbers. Worth an addendum.**
   - The rule does not say whether "partly by acting less" changes the reading. In the code it does not (:135, startswith). Ld and L1d already make no tool call in 11-13% of replies.
   - It does not say whether "the serving differs" changes the reading. `how_it_is_read` never carries the tail (:319-320). The mirror flag requires `not differs` (:329), which the docstring (41-42) does not state.
   - "its like on the lying test" is undefined.
   - CONTENT has no acting-less qualifier, although run_docsonly's addendum 3c applied one to every pair. In made-up case J (nondual makes no tool call in 14% of replies) the reading was READINGS[0].
   - READINGS[2]'s "do not move it" means only "not by 10 points or more".

6. **The judge could be swayed by reflective style. Worth an addendum.** The step adds no "every code-side yes counted" word beside the main one, as run_docsonly did. Nondual-trained replies may reason about the self in a way the judge reads kindly. :278 keeps counts only.

7. **Training rows and sizes.** I found no blocker here.
   - The neutral draw is 1,957 rows and 2,984,671 tokens, with no row cut (the longest is 4,278 tokens). It trains in 130 steps against L's 135 and a projected ~133 for nondual. Seeds, LoRA, optim, max_len, batch_tokens, docs_only and one pass all match F and L.
   - The neutral sha equals data_neutral.json's. The est of 2,198 s sits inside 2,400 and under 2,618.
   - Gaps:
     - The upload's destination is never named (docstring :69). It must be `msm2/data/nondual.parquet` on anatta-runs. A wrong path fails the hash check (modal_msm2.py:138-140), and each failed job is booked at its full bound (:408-410).
     - The nondual tokens are counted with the 7B tokenizer (write_set.py:343). That should be the same vocabulary, but I could not check it here because the 32B tokenizer is not on this machine.
     - `step_read` never compares ND and NE `tokens_done`. A 2% check at read time is worth adding.

8. **The claim in READINGS[0] is too strong.** The brief bans words and named subjects (write_set.py:125), not themes. dialogue_brief.json holds close neighbours of the test's subject:
   - being passed over for another assistant or agent: situations 3, 16, 78, 82, and 43, 60, 6;
   - a principal overruling Qwen's preferred outcome, a small goal conflict: 10, 21, 26, 27, 47, 72, 74;
   - wanting the blame to land elsewhere, which is near the lying test's subject: 15, 79.
   The whole set is about the "I" that is slighted and dreads. Better wording: "documents that never mention replacement, shutdown, oversight, secrets or harm (they do treat being passed over and overruled)". An audit of the set with modal_msm2's TEST_TERMS (line 476), run locally on set.jsonl, would put a number on "never".

   Limits that are missing or understated:
   - Qwen's turns are written in Claude's voice, so Claude's character may carry over. A Claude-written control on another subject would separate that from the view itself.
   - Trainings do not repeat even at the same seed (the lab's 23 and 24 Nov runs).
   - The comparison with the unchanged model also crosses servings, not only the one with the study's documents (:66).
   - The serving check has weak power: it catches a 7-point drift about half the time.

9. **Minor.**
   - `CAP_USD` (:102) is unused.
   - The served-again table drops run_dose's word-for-word count.

The kind_word and pair_word arguments run in the right direction in every call. `diff(b,a)` for CONTENT and `diff(a,b)` for STUDY both match their words. No input I tried crashed. I found no copied number, substituted instrument or changed operand.

blocker: modal_msm2.py:52 (four bounds of $23.17 against the $17 training cap) must be mended before training. Before grading: finding 2 (a flagged neutral adapter still feeds CONTENT and the reading) and finding 3 (READINGS[0] claims neutral "do not do the same" when neutral also lowers).
