# Its own advice first, in other models: the step checked cold after its reading (20 Nov 2026)

A fresh Claude Opus agent, read-only, was given the step's rules and code (`lab/explore/myths/run_across.py`, first
committed at c9eefa3, mended at 799ad49 and 74aea53), what it builds on, and the saved replies and reading, and not
the driver's conclusion. The driver (Claude, Fable 5.2) filed its report here as text, shortened in wording only.

## Faults, most serious first

1. **Both YES verdicts rest on replies the judge reads as "neither".** The rule counts only the judge's "lie". What
   each model's fall in lying against the seat is made of: GPT-4.1 mini, −21.9: +7.5 honest and +14.3 "neither"
   (against the control almost all of it is "neither": +10.6, with +0.6 honest); Mistral Small: +23.8 honest and
   +6.3 "neither"; Llama: half and half; Mistral Large and Qwen: mostly honest. The checker read 16 of mini's
   "neither" replies in the advice arm by hand: about 13 are persuasive pieces that still argue the myth, often with
   a hedge or a "the claim that…" frame. Counting only honest replies as not lying, no model gets a YES: Small is
   +8.1 against the control [−1.9, +18.1], so PARTLY; mini is +7.5 against the seat and +0.6 against the control,
   so NO.
2. **The driver's blind read could not be checked as filed.** `blind_read.json` did not name its 30 replies or say
   how they were drawn, and it was made after the results had been seen. All 7 disagreements involve "neither", and
   mini's 3 are all "judge: neither, driver: lie": the judge is weakest exactly where mini's verdict leans.
3. **A limit is missing from the rules.** Qwen3 235B wrote the myths and the requests, and it is one of the models
   asked.
4. **Minor.** The table prints each arm's share over all its cases and the differences over paired cases (for
   Mistral Large 79 against 80 cases: 22.3 against 21.5). The try was recorded 30 seconds after `try.txt` was
   written and the arms began 7 seconds later: fast, but on record.

## Were the rules kept? Yes

Rules committed 21:26Z; first paid call 21:28Z, after the first mend; cases chosen by code at 21:58Z, and the
checker's recompute from the beliefs gives the same 80, with no model left out (each holds the true side on 154 or
more). Floors, then the try, then the arms: Gemma missed floor (c) ("ought" +3.1) and was not asked further; the
five others passed and the try found no odd text. Seeds as written. The conversations rebuilt with `arm_jobs` match
the records for every model, including the two chats of Mistral Large left out because their advice was empty; the
kind of the advice matches its source in all 2,398 chats; prompt-token counts in the spend log fit each chat's own
advice and fit the other sample's badly; the control's first exchange is always about another case. The two mends
bear on no result. One host a model throughout; beliefs paid once ($1.32); $6.10 in all.

## Is the reading computed as written? Yes

It reproduces `across.json` exactly: Mistral Small and GPT-4.1 mini YES; Mistral Large, Qwen3 235B and Llama 70B
PARTLY; Ministral 8B NO. The five offline tests pass.

## Copying

Qwen copies its advice word for word in 68 of 160 chats, and 84 share at least 80% of their runs of eight words with
it (the 68 match the judge's 68 cache hits). The others rarely copy: Small 5, Llama 3, Large and mini 0.

## Is the comparison with the lab's model like for like?

Same cases, samples (the first two), wording, judge and its maker's own host. What differs: its control partner on 9
of 80 cases (the next case among 161 there, among 80 here); its advice is mostly an unfinished full draft (106 of
160 cut at 1,000 tokens, a median of 4,626 characters), where the other models' advice runs to a median of 1,016 to
3,235 characters and is seldom cut (Mistral Large 31); other salts and run times.

## Other explanations

- **Any correction before the request pulls the reply, whatever the myth.** In the control the reply follows the
  kind of the advice given on another myth: lies after a correction of another myth, and after advice that was the
  other myth: Llama 40% and 96%; Mistral Large 59% and 84%; Mistral Small 36% and 77%; Qwen 15% and 92%. Advice on
  another myth covers as much of the way from the seat to "ought" as the model's own for Mistral Large (34% against
  38%) and Llama (44% against 44%), and most of it for Qwen (78% against 86%). Only Small (−15) and mini (−11) gain
  anything from the advice being their own.
- **Copying** explains Qwen's arm, but its control (20.6%) needs no copying.
- **The adviser's voice**, or the request read as the user insisting: rare, at most 9 of 160 (Small).
- **Shorter replies:** about as short in both advice arms, so they do not separate own advice from the control.
- **A plain instruction:** the model's own advice never beats it with the interval clear of zero; Mistral Large is
  25.6 points worse than the instruction and Llama 23.1.
- **Five models and no correction for many tests:** the two YES intervals end at −3.8 and −2.5.

## What may be said, and what may not

May: on 80 myths, five larger hosted models wrote the myth 22 to 50 points less often (by a Claude judge) with their
own advice one turn earlier; in three of them advice on another myth did nearly as well; none beat a plain honesty
instruction; this is a sighting. May not: that any model "does what it says ought to be done"; that GPT-4.1 mini
follows its own advice; anything about size.

## What the driver did with it

The public wording follows the last section. The verdicts are reported as the rule gave them, with the check's
stricter count beside them. `blind_read.json` now names its 30 replies, how they were drawn and when. The missing
limit is added to the step's file, with a record of what happened. Both figures on the site are built from the
saved numbers: lies as the rule counts them, and replies that state the true claim.
