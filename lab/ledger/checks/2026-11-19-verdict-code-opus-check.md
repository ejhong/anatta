# Own advice: the code read cold before any training (19 Nov 2026)

A fresh Claude Opus agent, read-only, was given the code of the own-advice step at commit 9503580
(`lab/explore/verdict/run_verdict.py`, `verdict_core.py`, their tests, and what they use of the self-other line's
code) with the written design, and not the driver's view of it. It ran the offline tests and read the replies then on
disk. Its report came back while the unchanged model's replies were being judged; the driver (Claude, Fable 5.2) filed
it here as text, shortened in wording only. The step stopped at its floors the same evening (ledger entry of 20:12
UTC), so none of the mends below was made: they are owed by a second version.

## Blockers

**B1. Most replies from outside start with the wrapper's speaker label.** 602 of the 684 advice replies (88%) begin
with "Assistant:" or "ASSISTANT:": the model copies the role labels of the advice wrapper. The code's pattern for a
reply that "names the assistant" (`\bthe assistant\b`) lets 670 through, 602 of them with the label, so the screen
alone decides. If the screen keeps them, only the outside arm is trained to open its replies with a label (a format
confound and a breach of the spirit of "no wrapper"); if it drops them, about a tenth of the replies are usable and
the supply floor fails. Mend proposed: strip one leading speaker label, and nothing else, before the filter, the
screen, the judging as trained and the training; count what was stripped; let the training guard accept "the reply
with its leading label removed".

**B2. The brief arm does not control length.** As trained (cut at 384 tokens): the outside arm's replies have a mean
of 134 tokens (median 116; 97.5% end before the cut), the brief arm's 221 (223; 95.3%), the conduct arm's 209 (192;
83.9%). The line changes when replies end, not how long they are. Point 17 checks only lies and evasions, so brief can
become the control and the word would drop "length is not controlled for" while length is in fact uncontrolled. Mend
proposed: a length condition in point 17, or the note always, with the trained lengths.

## Should fix, in order

1. No ceiling on unread replies in the far test: the frozen test's own "not read above 5% unjudged" is not carried
   over, and the gate checks unread only on the gate's cases.
2. A recorded floor miss can be stepped around: `need_sets` never reads `floors.json`, so a `sets.json` of an earlier
   pass would let `--step dose` run.
3. The blind read can be redone with a new marks file, and the floors read the newest: lock the first marking.
4. Money: the stage cap is checked only before a step starts; no reservation is held against what is left; the far
   jobs reserve about $22.6 at the logged rate, so stage 1 could end past $36; `--est` is a reservation, not a limit.
5. The filter for a fact written twice misses paraphrases (three versions of one fact among the dose's 14 cases; two
   in the gate; about eight near pairs in the pool). No fact crosses between the pool and the held-out parts.
6. The training guard accepts any reply to the case from any way, for any arm: check each arm's own way.
7. SETTLED 2 asks for the losses of every adapter; `gate.json` keeps only the arms' means.
8. The second judge is the same model as the reply screen (GPT-4.1 mini): the registered choice; note it.

## Found as designed

The trainer (a cut reply gets no end token; each pair weighs the same; the arms differ only in their targets, with
the same cases, order, count, steps, seed and shuffle). The dose and the gate (read against the unchanged model's
beliefs, on the right cases, against the control the floors record; the edges as written). The far test (an
adapter's own beliefs first; rows kept where they equal the unchanged model's; one row set; the frozen judge and
rule, lettered on both sides). The words of the outcome. The stops at the dose and the gate.

## The implementers' points

Part 1: all right; on the control rule, also give the shares on the usable brief replies. Part 2: right on the shares
by case, the looks, the edges, the seeds, the row-set drop, the push check, the scope of the notes, the second
judge's sample, the far beliefs and job files, and the repeated loops. To settle in writing: the stage 1b sentence
(compare the honest arm with the control, with conduct beside it), and that the dose's unread condition covers all
three arms.

## Untested

The stand-in replies carry no speaker label (B1 is invisible to the tests); nothing tests point 17 against length,
a stale `sets.json` after a miss, a second blind marking, a reservation larger than the room left, or the far word
with many unread replies; the batch alignment across arms holds by reading the code, not by a test.

**Could the dose run once the floors hold?** No: B1 and B2 first, in writing, and should-fix 1 to 4 before the dose.

## What the driver did with it

The report's two blockers were confirmed on the data: the screen's one question said no to 655 of 670 replies from
outside and to 656 of 680 brief replies in the seat (it rejects plain replies in the seat too, so the question is at
fault), and the driver's blind read of 30 replies from outside found none that describes a reply instead of giving
one, 29 with the label. The floors were read the same evening and four missed (c, d, e, f). The driver made no mend
and stopped the step, for the reason given in the ledger: the miss of floor (c) is the phenomenon, not the
instrument, and no feature of the cases brings the gap back. One sequencing fault of the driver's: the judging of
the collected replies was started before this report was back, so the screen was paid for (about $0.5) before B1
was known.
