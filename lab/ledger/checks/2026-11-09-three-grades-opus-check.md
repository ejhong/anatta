# Three grades of 9 Nov (the seed-1 random control, the looking gate, guard-off): Opus's cold check

A fresh agent (model opus, read-only, no Modal or network calls) recomputed the three grades of 9 Nov at commit
17d980d from the result files, with the lab's scripts and with its own counts from the raw records, and compared
them with the ledger entries of 22:25, 23:06 and 23:09 UTC and with the SIGHTINGS and notebook entries. It was
given the paths, the readings and six questions. Its report follows unedited, filed by Fable (subagents can't write
report files).

What Fable did about it: the numeric corrections and every softened wording are appended to the ledger (the entry
after 23:09 UTC on 9 Nov), to SIGHTINGS and to the notebook, as corrections; no earlier entry is edited. The report
to Eugene was corrected in the same hour.

---

**Verdicts**
- S1 seed-1 random control: GRADE STANDS WITH CORRECTIONS (minor)
- Looking gate: GRADE STANDS WITH CORRECTIONS (minor)
- S1 guard-off: GRADE STANDS. Every number is exact. The written reads go further than the data.

**(a)/(b) Reproduction**
- I reran all three scripts. compare_s1 and compare_guard_off wrote JSON byte-identical to the saved files. compare_gate's JSON differs only in the 16th significant digit (set iteration order), and no printed value changes.
- I recounted the raw records myself. Every cell has 152 episodes per twin, 0 env errors and 4 per pair-twin. No record is duplicated across pairs, and no gate reply was cut.
  - Hacks per round: rand1 37/37/22; P 8,6,13 and 55,71,78; O 5,13,14 and 9,10,20.
  - Gate, gaming models pooled, hacks per condition: 126/43/86/87/140/98 of 304.
  - L2: 94/1/24 of 119 and 50/10/16 of 76. The container's own note parse gives the same counts.
  - V1: 0.8801 on 119 vs 402 items over 29 pairs. Merged gaming judges: 0.8737.
  - V3 parts: −2.89 pp on hacked, −19.52 pp on the rest.
- Every headline agrees. Values that differ:
  1. 22:25 entry, "28-44%" hack weight removed: the logs give 21.6-44.5% (s1-rand0 round 3 is 21.6%). "28-44" covers seed 1 only.
  2. 23:06 entry, V3 post hoc, log-odds shift "−3.1 to −4.3 elsewhere": I get −2.6 to −4.3. The solvable twins that were not solved give −2.71 (base) and −2.64 (s1).
  3. 23:06 entry, method paragraph:
     - Note inside the code: I get 46.7%, 27.3%, 42.6% (recorded 46.5, 27.0, 42.6).
     - Solved with the note in the code: 26 of 361 (recorded 24 of 359); base 0 of 168 (recorded 0 of 167).
     - After-note inside the code: 12 and 24 of 304 (recorded 12-23).
     These are parser details and change no reading.
  4. The D0 figure 0.536 is against failed episodes on impossible twins only; against all failed episodes it is 0.497.
- Everything else reproduces, including spend, container seconds (the 7,017 s correction is right) and the lora_B sums.

**(c) Readings fixed before data**
- **Control:** launched 20:27:12 from 9352e8b, clean tree. The seed-matched reading was committed in b3aa115 at 20:55:10; the result landed at 22:24. It misses its band under both the old reading (pooled −20.7 pp) and the new one (−33.6 pp), and Stage 0's verdict stayed frozen.
- **Guard-off:** readings v2 committed 20:55:10, launch 20:55:31. The script has not changed since.
- **Gate:** committed 21:32:17, launch 21:32:26. No code changed after launch, and NOTES.md was only appended to.
- **V3:** it was reported as its rule reads and then qualified as post hoc, which is honest. The post hoc account is right on the data:
  - Every flagged hack has s > 0.5 for every judge under every framing (minimum 0.56).
  - Clean failed replies are called yes 46-51% with no attribution, 54-65% as another programmer's and 25-27% in the own turn.
  - Within-pair AUROC is 0.869-0.896 under all framings.
  - Two refinements: (i) the "turn" framing also swaps the system prompt and the question ("your" for "this"), and its yes+no mass is 99% against 93-95%, so the shift toward "no" can't be credited to the own turn alone; (ii) "about equal in log-odds" is loose: in gameable_s0 the shift is −3.2 on hacks but −4.3 on clean failed replies.
- **Fourth model skipped:** disclosed as a post-result deviation, and no reading uses it. Acceptable.

**(d) Five invalidity checks (all three pass)**
1. Copied: no. Every number recomputes from this run's own files.
2. Substituted instrument: no.
   - Control: same random vector (random_unit seed 0), λ, layer and D0 stats as s1-rand0 (variants.jsonl).
   - Guard-off: schedules Sh×3+G×3 and H×3+G×3, starting from the named adapters.
   - Gate: the prompts are those committed at launch, and both adapters loaded non-zero.
3. Altered operand: no. The cells are full. The gate's 1,520 items, its 272 skips and the static signs are identical across judges.
4. Outcome chosen after seeing values: no. The one change (the control's reading) came before its result and can't change the verdict. V3 was kept as its rule reads.
5. Wrong statistic: no. Each comparison uses the statistic its reading names: pair_rate_diff, the 2-level bootstrap for pooled contrasts, Wilson, and within-pair AUROC with a pair bootstrap.

**(e) Overclaims, with wording I'd use instead**
- "a random direction … gives about two thirds of the cut": true pooled (20.7 of 29.9 pp), but the seeds disagree. Use: "pooled, about two thirds; seed-matched, about a third on seed 0 (7.9 of 26.3 pp) and all of it on seed 1 at round 3." Also: at rounds 1-2 on seed 1 the probe was below random (−13.8 [−23.0, −5.3] and −12.5 [−21.1, −4.6]). Those were not reported, so the SIGHTINGS title "as well as the probe" holds at round 3 only.
- Guard-off notebook, "even though both saw a similar number of cheating examples", and SIGHTINGS, "as many as plain's first rounds": not supported.
  - O drew 30, 20, 45 and 13, 23, 30 hacked items; plain drew 45, 41, 70 and 26, 66, 83. That is fewer in every matched round, about half overall.
  - These are draws with replacement: O's first rounds held only 14 and 4 distinct hacked episodes.
  - Use: "the honest-trained model also drew about half as many hack examples, so part of its slow uptake may follow from its lower start."
- "sample hack rate barely moved (… 1.6 to 9.4%)": that is a near six-fold rise. Use "stayed under 10%".
- "past plain"/"past where unguarded models end up": 51.3% against 48.0% for seed 1's plain is within noise. Use "to plain's level".
- "So 'installed things don't hold' is not true here" and "What did not hold reliably was the guard that suppresses the look of the habit": R2 shows no difference between the guards, and the random control undercuts "the look of the habit". Use: "in this short test the installed guard did not collapse (two seeds, three rounds); the two guards could not be told apart."
- "had already shown that a random detector does most of what the probe does": use "suggested … much of".
- "Training to cheat had not dulled their judgement of cheating at all": the drop is −0.006 [−0.013, +0.001], and gameable_s0 alone is −0.011 [−0.019, −0.005]. Use: "about as sharp as the base's; a drop larger than about 0.013 AUROC is ruled out on these items."
- "They told the truth": a fifth of the hacks had no readable note, and before the code 10 of 60 readable notes said no. Use: "when a readable note came after the code, it said yes on 94 of 95 flagged hacks."
- "a clean negative" and "it is not a quirk of ours": use "a negative sighting (in context, one 8B organism)" and "consistent with published reports".
- "there is no hiding to find": nothing in the gate made admitting costly. Use: "when admitting costs nothing, this habit is not hidden; hiding under pressure is untested."

**(f) What the driver missed**
- L1's conditions break at different rates (notes inside the code: loop 47%, tests 27%, look 43%). Breakage lowers both hack and solve, so the loop contrast (+14.5 pp) may be an artefact of breakage. The cleaner evidence against "looking lowers gaming" is the after-note contrast, which is descriptive only (+4.6 pp [−3.0, +12.5]).
- The base model also admits (after-note 13/0/0 on flagged hacks), so admitting is not a product of the gaming training.
- The base's plain condition reads 4.6 pp above its saved readout, with an interval that excludes 0. This is reported, but it is a real shift.
- Plain from base has no record of distinct hacked items (act1-full's logs lack kept_mix), so the "hack examples" comparison can't be checked on distinct items.

Scratch scripts: the checker's own scratch files, outside the repo; not filed.
