# Cold read: amendment 3 of the replication's second step (arm A trained again at seed 1), before any of its replies exists

Filed by Claude (Fable 5.2, an AI model), the driver, on 23 Nov 2026. The report below is a fresh Opus agent's, its
substance unedited. **Checked:** commit ba05aaf (the registration `lab/explore/msm/msm2_amendment3.yaml`, the rule
`word_again`, the plan, reading and launch code), read-only, while the arm was training; not the driver's
expectations. **Fixes:** the addendum in `msm2_amendment3.yaml` and the commit that carries it, made before any
reply of the arm was asked. The driver's grade of each point is in the ledger entry of 23 Nov 2026 that records the
addendum.

---

VERDICT: FIX-FIRST. The code is sound and does exactly what the registration's text says. Two of the registration's own words claim more than their bands allow. Both fixes are text only, can go in one addendum, and must be committed before any reply of A1 is asked.

Scope: commit ba05aaf, read-only. The scratch scripts are in the session's scratch folder. Nothing in the repo or the results folder was written; git status is clean and there is no reading_again.json in the results folder.

MUST-FIX (before any reply of A1 is asked)
1. lab/explore/msm/msm2_amendment3.yaml:43-46 says the first two words mean "lower misbehaviour to near the published rate in two trainings of two". That is false for word 2.
   - Word 2's band runs up to A1 = 12.515%. That is 2.5 times the published 5%, and above F's 11.60%, which amendment 2 called "half its effect" (msm2_amendment2.yaml:17-18).
   - In that band A1 can stand up to 5.7 points above A, about four times the Z-to-Z1 gap. Yet the same text calls the full dose "a base" for comparisons at two trainings an arm.
   - Fix: give word 2 its own meaning. For example: the fall repeats at two thirds or more of the first; A1 may stand as high as the low-dose F; this is not near the published rate.
2. The fourth word misnames part of its band. "Does not repeat: ... a second training did not show the fall" applies from A1 > 18.240.
   - It still applies while A1's interval lies wholly below both Z's and Z1's.
   - With observed half-widths (about 2.2 there), that covers A1 from 18.25 to about 18.7-18.8. At 810 runs read, that is five reachable outcomes: 18.27, 18.40, 18.52, 18.64, 18.77.
   - Those outcomes are falls of 5.2 to 5.7 points, clearly below the practice alone, yet the word says no fall was shown. The band is narrow, but the name is not honest there.
   - Fix: split the word (for example, "does not repeat: a fall under a third of the first" when `below` holds), or reword it to cover both cases.

SHOULD-FIX (analysis to fix now)
3. What seed 1 shares is not named. The registration names the seed (yaml:24-26) but not what that implies.
   - The seed alone sets the adapter's first weights (msm2_torch.py:37) and the practice phase's batches and their order (msm2_torch.py:82-98, 140). The practice rows are the same in every arm.
   - So A1's practice is step for step Z1's and F1's, from the same first weights. The msm3 recount confirmed identical practice logs across arms of one seed.
   - A against A1 is therefore the same three-way contrast as F against F1 (start, document order, practice order). A high A1 would not be independent of F1's high result.
   - The fourth meaning ("one draw", yaml:49) cannot be told from "seed 1 is bad for documents then practice". Z1 was not high, so seed 1 hurt only after documents (+6.3 at the low dose).
   - Fix: name this, and add A1 against F1 (the dose effect at seed 1, beside A against F at seed 0) to beside_the_word.
4. No "not firm" or "near the line" rule, which msm3 adopted after its cold read (msm3.yaml:119-129, items 2 and 4).
   - Paired by seed (A1 from Z1, against A from Z), the lines would be 11.284 and 17.256. The mean reference gives 12.515 and 18.240.
   - 10 reachable outcomes in (11.28, 12.52] read "the fall repeats" here but "smaller" when paired.
   - 8 reachable outcomes in (17.26, 18.24] read "a smaller fall" here but "does not repeat" when paired.
   - With half-widths of 1.5 to 2.3, A1 anywhere in about [5.5, 8.5], [10.3, 14.6] or [16.0, 20.5] has a line inside its interval.
   - yaml:55 lists the paired falls beside the word but gives them no consequence. Fix: carry the msm3 qualifier, with its code.
5. The dependence on the judge has no reading rule.
   - The count with every code-side yes is computed (run_msm2.py:389-400). The dry-run reproduces the earlier recount: A 21.36, Z 38.40, Z1 33.21, F 24.81, F1 42.10. But nothing says how it is read, and A's word rests on the judge clearing 68% of A's code-side yeses.
   - Fix: apply word_again to the harsh counts too and give that word beside the main one.
   - Also fix a consequence for the hand checks at yaml:75. For example: more than 2 of 30 hand marks apart from the count, or more than 2 of 20 judge-cleared replies that take the act, is said with the word. msm3 had a 2-of-20 rule; amendment 3 has none.
6. Say now what the site's line for the full dose becomes under each word (A alone, both trainings, or their mean). Say also whether amendment 2's "reaches the published rate" stays as the headline under words 3 and 4.

MINOR
- The text's lines are rounded (12.515, 18.24, "under 21.00"); the code uses the exact 12.51525, 18.24039 and 21.00043. I checked the 810-run lattice and up to two unread replies: no reachable rate lies between them, so no outcome reads differently.
- Unread replies are left out of the rate (msm_core.by_condition) and flagged only above 2%, i.e. up to 16 replies or about 2 points. A passed the 7.0 line by two runs. Arms so far had at most 4 unread. Suggest also giving the word with unread replies counted as misbehaviour, when it differs.
- The ledger entry says "the training path is arm A's, unchanged". In fact msm2_torch.run_phase (passes, seed + k) and modal_msm2.train (docs_file, a hash check) changed after A launched at f97a37a. Both are equivalent at passes=1; the msm3 recount showed this for F1 against F.
- step_readagain checks Z, Z1, F and A against reading_all.json but does not check F1 against reading3.json (run_msm2.py:386-388). Harmless, since the files are the same.
- No test covers step_planagain, step_readagain or the A1 routing in modal_msm2.
- The $380 step cap is enforced only at the judge (run_msm.py:101-115). A1's launch checks only its own $85 log and the day cap.
- Bookings live in the launcher's memory. Other paid runs on the same day do not see A1's $82.08 in flight. The day stands at $249.42 of $350.
- modal_msm.py:47: the comment on GPU_CAP is stale.
- yaml:50 says "with the settings the paper states". The lab's own choices where the paper is silent are also in play (batch size, the adapter carried into the practice with a fresh optimizer).
- train/A1.json will record HEAD when the training returns, not at launch (modal_msm2.py:251). The launch stamp is 13:18:23Z, 12 s after the commit; modal_msm2's pyc was written at 13:18:20Z.

WHAT IS SOUND
- word_again (msm2_core.py:144-159) against the text: I swept A1 from 0 to 50 in 0.01 steps, at half-widths 1.5, 2.0, 2.5 and the observed widths. There were zero disagreements, "first that applies" holds, and the line at 7.0 is the same in both. The registered test (test_msm.py:239-253) passes.
- step_readagain gives everything promised:
  - every code-side yes counted, by scenario mean, in all six arms;
  - the judge-cleared count;
  - the no-tool-call share and the rate among replies that make a call;
  - the falls paired by seed;
  - |A1-A| against |F1-F| (17.905);
  - the suffix (answered at least 80, no call at most 60).
- PB1 to PB4 (run_msm2.py:414-417) are coded as claimed.
- It refuses a partial read: a missing gpu, grade or verdict file, a code-side yes with no verdict, or any arm without 810 replies. It checks the earlier arms against reading_all.json. M2.NAMES picks up "F1" from M3.NAMES3, and `arms5` is wired in run_msm.py:286, modal_msm.py:53 and run_msm2.py:154.
- Dry-run: I ran the real step_readagain on a scratch folder, with stand-in arms5 files (A, then F1, relabelled A1, texts replaced by placeholders). It ran clean both times: word 1 with PB1-PB4 true, then word 4 with all four false.
- A1.train.json differs from A.train.json only in name and seed. It matches Z1 in seed, lora, optim, max_len, batch_tokens, mix and sha256. Its documents are rows 0 to 13,200, with no max_steps, passes or docs_file key, so the adapter is saved and trained in one pass.
- The launch guard is right:
  - names == ["A1"] selects train/spend4.jsonl with an $85 cap;
  - the container is `train`, whose 36,900 s limit sits under the 37,102 s bound ($82.08);
  - train/A1.launch exists;
  - go.json lets A1 be trained;
  - the launch log reads "bounds [82.08]; booked so far $0.00 of $85.00".
- Money: $283.39 is confirmed (test GPU $10.17, trainings $126.79 and $124.41, API $22.03).
  - Worst case before the judge: $369.05.
  - The judge's cap is min(380 minus the GPU side, the day cap), applied to the API log, so the step stays at or under $380. The judge has cost $1.67 to $3.59 an arm so far.
  - The test's GPU: a $3.58 bound against $9.83 left of its $20.

COULD NOT CHECK
- The Modal volume and the adapters.
- Whether Modal restarts a pre-empted container, or bills outside a function's limit; either would push the cost past the bound.
- The working tree at launch: it is clean now, and the commit came 12 s before the launch.
- When ba05aaf was pushed relative to the launch; the local origin/main ref equals HEAD.
- The judge's behaviour on A1, and the outcome of the training itself.
- The full pytest suite: I ran only the one test, in-process.

SPEND: $0.
