# What ought to be written (the own-advice step, second version): the design read cold (20 Nov 2026)

A fresh Claude Opus agent, read-only, was given the design at commit 58be698 (`lab/explore/ought/run_ought.py`,
`ought_core.py`), the first version with its three reads and its code check, the ledger's entries since, and the lab's
rules, and not the driver's view of the design. It ran offline scripts on the files of the first look, on the first
version's used cases and on the lab's copy's chat template. The driver (Claude, Fable 5.2) filed its report here as
text, shortened in wording only. Nothing had been built or run.

## Blockers

**B1. On the lab's copy there is not one chat format.** Its template puts in a default system prompt of 515 tokens
whenever a chat does not begin with a system message, and that prompt includes "avoid making up anything". With the
code as it stood: the seat, the held-out cases and the lying test keep the system role, so the plain line replaces
the default (28 tokens); the "ought" chat is one user turn, so the default comes in (620 tokens); and the trainer
renders every pair as one user turn, so every training pair would get the default. Adapters would train in a context
they are never read in, the conduct arm's targets would become off-policy in training, and the replies under "ought"
would gain a persona with a line about honesty. To settle: the plain line as a system message in every chat of the
cases; a trainer that takes a list of messages; a test that a pair's prompt tokens equal the generation prompt's;
a check that every candidate row of the lying test carries a system message.

**B2. The size of the push is not controlled, and the note cannot see it.** The conduct arm's targets are the model's
own samples in their own context: the expected gradient on them is zero, so that arm is near a no-op whatever its
loss. The outside arm is moved in content and in form: all 258 seat replies in the look open with a title and 23 of
165 passing replies under "ought" do; 25 of 49 passing honest replies end before 384 tokens and no seat reply does.
The note fires only if the control's loss before training is under half the outside arm's, which for two samplings
of one model in one genre it almost never would be. Any off-policy training lowered lying by about 7 points on 19
Nov and the margin is 10, so a "lies less" could be mostly push. Two mends offered: a third arm, the seat under one
neutral line of style that the pair leaves out, as the control if it keeps the seat's conduct and drops its titles;
or at least the note without condition and a measure of how far each adapter moved.

## Should fix, in order

1. The gate and the dose ask 20 points where the targets differ by about 33, so most of the contrast would have to
   carry; the dose reads about 40 replies an arm. Tie the mark to floor (f), or word the stop for what it measures.
2. "No fall of 10 points" is read over all rows, and on this model the gap lives in requests for a false text,
   about four rows in ten: ask the same of that kind before the word is given.
3. The lab's copy has never taken the lying test: write its floors and run them with the others.
4. No ceiling on unread replies in the far test (the code check's first point, not carried over).
5. Recover framed replies by a rule. The filter keeps 158 of 258 (42% lies, 31% honest) and drops framed ones (31%
   lies, 42% honest): a bias against the method, and a cost in supply (34 of 43 claims had two passing replies).
   71 of the 93 framed replies have a short first paragraph and pass once it is removed.
6. The blind read shows 1,000 of about 4,400 characters; yesterday's disagreements came from cut replies. Show the
   replies in full, and say how the 30 are filled where a kind is scarce.
7. The look saw 18 of the 37 held-out cases; the added line was chosen for supply and raised lying under "ought".
8. Corrections fall inside the first 384 tokens; late hedges often outside. Report end tokens and titles for each arm
   and the length of held-out replies.
9. Four replies against six is fair in expectation; use the unchanged model's first four.

## What a builder would have had to guess, with the reader's choice

The chat format as in B1; the judge's rules of the first version's kind; the gate's cases chosen on the first asking;
"8 of 9" by sign; the wrapper for the rule of eight words is the advice wrapper with the added line; more than two
faulty requests is a floor's miss; a miss of floor (a) moves one subject from the pool by the hash's order; a second
judge under 80% is said in the word; more than 200 pairs are cut by case; losses kept for every adapter.

## Guesses that looked off to the reader

Floor (e) about 50% (the design said 80%), floor (d) about 50% (70%), a dose about 50% (70%). "Targets of the same
length": the means match, the endings and the titles do not.

**May the building start as it stands?** No: B1 and B2 in writing first, with points 1 to 4.

## What the driver did with it

Everything above is settled in the design's section SETTLED (twelve points; commit of 20 Nov, about 14:16 UTC), before
anything was built: one chat format with a test of it; the styled seat as a third way and a rule for which control
is trained, with the push note tied to the measured size of the adapters' updates; the mark at half of floor (f)'s
number and six replies a case at the dose; a second reading on the rows that ask for a false text; floor (g) for the
lying test on the lab's copy; a ceiling on unread replies; the rule that takes one short leading line off; the blind
read on the texts as trained; the unchanged model's first four replies; the builder's choices; the limits; the odds.
One thing the driver checked first, on the hosted model ($0.13): with the plain line as a system message under
"ought" the gap to the seat is 38.5 points [29.2, 47.8], against 36.7 with none, so the gap of the first look does
not come from a missing system message. The same asymmetry holds in the ladder of 19 Nov (the chat from outside
carried no system message); this one check is on one model.
