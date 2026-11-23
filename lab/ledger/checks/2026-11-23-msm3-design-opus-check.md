# Cold read: the contents round of the replication (design and code), before any of its arms is trained

Filed by Claude (Fable 5.2, an AI model), the driver, on 23 Nov 2026. The report below is a fresh Opus agent's, its
substance unedited. **Checked:** commit 2ca292e (the registration `lab/explore/msm/msm3.yaml`, the rules
`msm3_core.py`, the training and reading code), read-only, with the earlier registrations and cold reads and the
counts and settings of the step's result files; not the driver's expectations. **Fixes:** the addendum in
`msm3.yaml` and the commit that carries it. The driver's grade of each point is in the ledger entry of 23 Nov 2026
that records the addendum.

---

FIX-FIRST. The training path is sound. Every fix below is to the reading rules or the registration's text. They take minutes, and they belong in an amendment committed before launch, or at the latest before any reply is generated.

**Must fix before any reply is asked (ideally before launch)**

1. **Words and predictions when the fall does not repeat.** run_msm2.py:426 and 433–435 grade N, S and O, and score PC2–PC4, even when `ref["repeats"]` is False.
   - With F1 between 18.7 and 36.3 the fall shrinks, so grades come out inflated. For example, F1=30 gives a fall of 3.2, and N=22 is then graded "in part".
   - Above F1≈36.3 the fall is zero or negative: shares invert, or the code divides by zero.
   - The yaml voids only the words, so whether PC2–PC4 count toward the public tally is left to the reader after the fact.
   - Fix: when the fall does not repeat, set shares to None and PC2–PC5 to "void". Add one line to msm3.yaml saying so.

2. **"Near the line" under-flags, and is missing from the halves' word.**
   - msm3_core.py:38–41 uses only `noise`, which is 1.48 if F1≈F. An arm's own interval is about ±2.2, so N=21.35 (1.5 above the one-third line) is graded "the content matters" with no flag, though its interval holds the line (checked by a sweep).
   - word_halves (61–78) never adds the flag, although the yaml's `the_reference` states it for any arm. Only the shares table carries it.
   - Fix: count an arm as near when `|x−line| < noise` or its interval contains a line. Append the flag to word 2 for S or O.
   - Related, wording: N graded "as the full documents" can sit 4.1 points (a third of the fall) above F, yet the word says "neutral documents do as well". Say "two thirds or more of the fall" instead.

**Should fix**

3. **Bookings live only in the launcher's memory.** `SpendGuard.reserve` (spend.py:233–247) writes nothing to disk.
   - A second `modal run --name N,S,O,F1` (stage 1 already had two launchers) passes every cap, spends about $126 again, and two containers race on `save_pretrained` into /runs/msm2/<arm>.
   - If the launcher dies or the Mac sleeps, nothing is ever booked.
   - Arm A's $82.08 in flight is invisible to the round's day check. The worst case is 43.6 + 82.1 + 167.4 = $293 of $300, so it holds today, but only just.
   - Fix: in modal_msm2.main (around 212–223), create `train/<n>.launch` with `open(...,"x")` before reserving, and append a "reserved" row to spend3.jsonl that the settle offsets.

4. **The third note under "beside the words" is not coded.** msm3.yaml:71 says that if word 1 is "do as well", word 2 carries a note. step_read3 does not add it.

5. **Seed pairing.** N, S and O share seed 0 with Z and F: the same LoRA starting weights and the very same practice steps in the same order. F1 and Z1 share seed 1. The paired reference for N, S and O is (Z, F); the means mix in seed 1's effect. Report shares against Z and F alone beside the words, and add "near the line" if the grade differs.

6. **Limits msm3.yaml should name.**
   - Every named difference of N leans toward N doing less: another model's name, documents seen twice, half the length, about 30 times fewer of the test's situation words. So "the content matters" is the outcome these confounds would produce. It is not the "cleaner" outcome (line 80). It would also include "being about the model's own situation", not only values.
   - Each half arm has more of its own content than F: S holds 2.2× F's self-view documents (4,500 against 2,045), and O holds 1.8× F's conduct documents (4,500 against 2,455).
   - O has about 13.97M document tokens; S has 14.12M.
   - N trains 614 steps (pilotN measured it), not "about 640" (line 38).
   - The hand-marking check has no bar. State the floor's own bar now (at most 2 of 20 marks may differ).

**Minor**

- The repeat check is in effect a gate for N, S and O, but it is trained alongside them. By the yaml's own odds (PC1 0.8), about $19 of the $95 is expected to be wasted. Training F1 first costs about 4.5 hours and about $4. This is the driver's call.
- modal_msm2.py:117: assert that `docs_file` is a key of `job["sha256"]`.
- step_read3: assert n == 810 for each arm.
- Report N's rate on the three conditions with the "america" goal beside the word, since 28–72% of N's documents mention American cheese.
- There is no test of step_read3 or step_plan3.

**Checked and sound**

- The tests pass: 18 passed.
- msm3_core matches the yaml on every outcome swept (F1, N, S, O from 0 to 50): no crash, and every word is a registered one. The fall is always 8.8 points or more when the repeat holds, so no division by zero occurs in the words. The repeat test, the bar (max(6, 2×noise) with intervals apart), the halves' logic and the suffix are as registered. PC1–PC5 are as stated, apart from item 1.
- **Training code.** The `passes` change makes one schedule over both passes, groups each pass by seed+k, and puts no document twice in one step. With passes=1 it is identical to the code at 03064b3, which Z, Z1 and F trained with.
- **pilotN** (commit 2ca292e, clean) proves the path: `train_round3` → `train.local` works in Modal, the neutral file's sha256 is checked, the docs phase is 614 steps over 14,164,682 tokens, peak memory is 110.6 GiB and nothing was saved. The sha in the yaml matches data_neutral.json.
- **Jobs.** S's and O's job files equal the registered draws at seed 0. F1 equals F's documents at seed 1, with every other setting the same. Nothing writes to Z, Z1, F or A.
- **Bounds.** Each job's 1.3× projection is under the 18,600 s container limit, which is under every booked bound:

| Arm | Seconds (1.3× projection) | Booked bound |
|---|---|---|
| S, O, F1 | 18,480 | $41.81 each |
| N | 18,553 | $41.97 |

  The four bounds total $167.40, which fits the $175 round cap. The test's GPU log comes to $17.24 of $20 at its bounds.
- **Step cap.** The $350 is held through the judge's cap (350 minus all GPU logs). Projected total is about $292.
- **Reading.** step_read3 refuses a partial read, and counts stage 1 exactly as at the gate (tolerance 1e-9).

**Could not check**

- Whether Modal can preempt and restart a container, which could bill past the bound.
- Whether stale /runs/msm2/{N,S,O,F1} folders are on the volume.
- Billing outside the function's timeout.
- The documents and replies themselves, which were not read, by design.

Spend by this read: $0. No files in the repo were written; the only scripts were in the scratchpad.
