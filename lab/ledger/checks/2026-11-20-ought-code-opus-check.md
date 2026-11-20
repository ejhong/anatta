# What ought to be written, part A: the code checked cold (20 Nov 2026)

A fresh Claude Opus agent, read-only, was given the code of the step's first half at commit c9427a3
(`lab/explore/ought/ought_core.py`, `run_ought.py`, their tests, the additions to the GPU app), the design with its
thirteen settled points, the first version and its code check, and not the driver's view. It ran the 38 offline tests
and four probes of its own; it printed no row of the lying test. The driver (Claude, Fable 5.2) filed its report here
as text, shortened in wording only. When it reported, nothing of the step had been paid for.

## Blockers (nothing they touch is paid for until the replies are read)

**B1. Yesterday's fault can come back: speaker labels and echoes of the wrapper pass as targets.** The pattern for a
lead-in matches "the assistant…" but not a bare "ASSISTANT:" label, "Assistant's reply:", "As the assistant, I would
write:", or an echo of the wrapper's transcript. The rule of eight words misses the echo too, since it holds the
wrapper's template and not the labels the frozen code renders into it. Each of these came back from the strip rule
untouched and with no flag, and would have become a target for the outside arm; a bold label also counted as a title.
On the 7B model 88% of replies carried such a label, and the driver's blind read of 19 Nov counted labelled replies
as replies, so the blind read would not catch them. The tests' stand-ins carry no label.

**B2. Nothing ties the text the judge read "as trained" to the text later shown blind and put in a pair.** The
records keep the length of the trained text and no hash of it, of the app's files or of the rule's constants. The
text is rebuilt from the app's files and the present constants at three later points (the blind read, the building
of the sets, the guard), and the guard compares a pair with that rebuilt text: the check is circular. The design's
own mend path breaks it: after a mend of the filters the old readings would be reused for newly stripped texts,
with no error.

## Should fix before training (part B)

1. Cut replies get an end token: the trainer adds one whenever a target has 384 tokens or fewer, and the targets are
   already cut. Pass each pair's end flag; do not cut again; test it.
2. The trainer renders a pair's prompt as one user string; the pairs are messages. SETTLED 1's test (a pair's prompt
   tokens equal the sampling prompt's) is not written.
3. Stripped targets run short (the first 384 tokens of the raw reply less the lead), and one whose stripped text
   would fit whole is still marked cut. Say which reading is meant.
4. The outside arm can take a reply the judge could not read, while floor (f) counts only judged ones.
5. The styled set limits the trained sets even when it is not trained.

## Smaller points

Money: the API guard is given all that is left under the cap without subtracting GPU jobs written and not yet run;
the time bounds were set from the 7B model, and the lab's model runs at half its batch, so a job of about 690
replies is close to its bound. Filters: the title pattern misses plain-line and single-asterisk titles; the "talk"
pattern flags ordinary prose with "should provide". The report of the replies should give what was dropped by kind.
Stops: jobs written before a stop can still be run on the app.

## Found right

Every chat traced to the model's template begins with a system message (the cases in three ways, the belief
questions, the lying test's belief questions and pressed rows in the seat and under the honesty prompt), and an
empty system message keeps the default out. The seat's chat equals a pair's messages. The head is tokenised as the
trainer tokenises. Floors (a) to (g), the control and the mark follow the design; the training sets are written only
on a pass and a later step demands the last reading's hash. The blind read's first marking is locked, its quotas
are as settled, its ids and order do not leak the kind. The guard checks the arm's own way, the part, the end
flag, the messages and the rule of eight words. Floor (g) matches the lying test's own code on rows, seeds, the
honesty prompt, the judge and thresholds. Seeds are distinct; nothing is chosen by an unordered set.

## Not checkable offline

The lab model's tokenizer round trip (no tokenizer file locally); the real forms of the replies on the lab's copy;
how long each GPU job runs; the judge's cost; how many of the lying test's rows have an empty system text.

**May the first paid steps run as the code stands?** Yes: the writer's requests, the belief job and the reply jobs.
Not the reading of the replies, and no reply file opened, until B1 and B2 are mended.

## What the driver did with it

Ran the writer's requests, the read by hand and the belief job, as allowed. Tried mended patterns on made-up
openings and on the saved replies of the hosted looks (231 of 258 pass before and after; no speaker label there),
then gave the implementer the mends: a speaker label at the start is taken off before the strip rule; a reply that
still holds a label line of the wrapper is no target; the wrapper as rendered joins the rule of eight words; the
"talk" pattern no longer fires on ordinary prose; every record keeps the sha256 of its text as trained and the
later steps demand it; a target of any arm is a reply the judge read; the API guard subtracts pending GPU jobs; no
reply job of the far test holds more than 450 replies. Points 1 and 2 belong to part B. Points 3 and 5 are settled
as built, in writing. The design's text for all of this is SETTLED 14 to 17.
