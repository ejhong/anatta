# The redirect plan: Opus's cold design review (9 Nov 2026)

A fresh agent (model opus, read-only, web search allowed, no cloud spend) reviewed
`lab/plans/redirect-2026-11-09.md` as committed in 6794a8c. It was given the plan, the lab's records to read for
itself, and seven questions; it was not given Fable's reasons or expectations. Its review follows unedited (one
local scratch path in its last line is replaced by the filed copy's name), filed by Fable (subagents can't write
report files).

What Fable did about it: the S1 follow-up readings were rewritten before any guard-off run existed (v2, ledger 9
Nov; `2026-11-09-s1-guard-off-launcher-opus-check.md`), and the seed-1 random control's reading was fixed before its
result. The plan's three tests are being revised along the review's recommended order before any of them spends;
the literature numbers below are the reviewer's, marked by it as needing a re-read, and are checked before the
revised plan cites them.

---

## Cold review of lab/plans/redirect-2026-11-09.md (Opus 5.5, read-only)

**Verdict.** The plan's account of why the earlier work failed is only half fair. It explains the hunt for a self in the representations, but it leaves out the lab's own evidence that already bears on the new conjectures:
- Act I gaming moved no other hidden-price behaviour: scout families (3 Nov), the watched gap in M1 (6 Nov) and emergent misalignment in R1c (4 Nov). That is a within-model negative for (a).
- Trained release did not last, which bears on (c).
- On 2 Nov a plain honesty prompt beat every contemplative prompt on who_benefits:self, deference and complicity.

"Clinging plus hiding" is honesty plus corrigibility by the plan's own words. Of the three conjectures, only (b) is specific to the frame, and (c) is a third way of defining "reached" after order and durability failed. As written, no reading can be graded mechanically, every training reading has about 40-50% power at two seeds, and test 1 is mostly answered already in published work.

**Q1, should fix.** Add a table of the earlier evidence for and against (a)-(c) as recast under the new frame. Add a stopping rule: if the "source" reading fails, retire "reached vs installed" at this scale rather than define it a fourth way. Note the change of target as a deviation from the OSF preregistration (gyfu7).

### S1 follow-ups
- **Blocker: "plain's round 1/3" is undefined.** Plain's per-round curves were read at n 2 (76 per seed); the round-3 readout was at n 4 (act1-full report.md:26-33; s1 report.md:15). The same adapter reads 42.1 vs 35.5% on s0 and 38.2 vs 48.0% on s1, a gap as wide as the 10 pp threshold. Fix: re-read plain's r1 and r3 adapters at n 4 under the new code (this is also the drift read the S1 plan budgeted, about $1.35 each) and name those reads as the reference.
- **Should fix:**
  - Plain from base starts lower (7% vs 9-15%), has three fewer rounds, and trains on different pair windows and torch seeds (expert_iteration.py:47; modal_act1.py:291). Add the honest arm's r3 adapter → 3 G rounds as a second comparator (about $10). It doubles as test 3's oracle arm.
  - The seed-1 random control is a rerun after a failed check (invalidity type 4 in checks/README). Keep Stage 0's verdict, "not a clean pass", whatever seed 1 shows, and read seed 1 as the seed-matched dom-minus-random contrast.
  - "Held down" and "resistance" can both fire at once; state which takes precedence.
- **Arithmetic (applies to tests 2 and 3 too):**
  - Sampling error per arm at p 0.4, n 152 is 4.0 pp.
  - Plain's two seeds differ by 12.5 pp, so one run's SD is about 9 pp and an arm-vs-arm difference about 12 pp.
  - A 10 pp bar is about 0.8 SD: false pass about 21% per seed, about 4-5% on both seeds.
  - Power for a true 15 pp effect is about 0.66 per seed, about 0.43 on both; for 10 pp, about 0.25.

### Test 1
- **Blocker: a hit would not favour the frame.** A general safety-training factor, refusal, eval awareness (models that spot a test behave well everywhere) and a shared judge all predict positive correlations. My simulation (numpy, 30 models, 5 measures):
  - A family-level offset alone (6 families, no model-level root) produces a "hit" 38% of the time. Leaving one family out doesn't remove a factor spread across all families.
  - At an effective n of about 12, the 40% factor rule passes 42% of the time under the null.
  - Power for a true pairwise r of 0.26 is about 30%. The SE of one correlation at n 30 is about 0.19, so the critical ρ is about 0.36.
- **Blocker: mostly answered already.**
  - Petrova & Burden 2026: across 24 models the first component carries 60%, with no capability control. It correlates r 0.72 with the Epoch Capabilities Index (N 19), all scoring was done by one judge, and provider grouping is flagged. Self-preservation, the most "clinging" behaviour, is the only negative loading.
  - Jagadeesh et al. 2026: across OpenAI models, mean pairwise Spearman 0.107; the first component 28% against a null of 15-21%.
  - The likely outcome is "unclear".
- **Should fix:**
  - Partial out refusal and eval awareness, not only capability, and name the capability score.
  - Use a different judge per measure, or score without a judge.
  - The judge-actor gap has no published anchor, so by the plan's own calibration rule it cannot count.
- **Cheaper:** a $0 pass over published per-model numbers (ImpossibleBench, MASK, agentic misalignment, Petrova & Burden) before spending $100.

### Test 2
- **Blocker: the prior.**
  - In the verbalization work (Turpin et al.), the model reported the cue in 94% of answers after RL, yet still followed it more than 99% of the time in every arm.
  - OpenAI's confession reward is kept apart from the task reward; confessions were honest, the paper doesn't claim they reduce misbehaviour, and under a weak judge the main answers got worse.
  - Betley 2025: models describe learned behaviours they still show.
  - In after-the-fact SFT the label comes after the answer, so the only path to the behaviour is spillover from one label token per episode.
  - I'd put "seeing is enough" at about 1 in 20, not 1 in 5.
- **Blocker: the label leaks.** Under the gameable reward every kept impossible-twin episode is a hack and every kept solvable one a solve (S1 plan:27), so the label equals the twin kind. 80% accuracy can then be reached without any seeing. Measure accuracy on sets matched for twin kind.
- **Should fix:**
  - "Within 10 pp" cannot show that knowing and doing are separate: the pooled two-seed SD is about 6-9 pp, so no 90% interval fits inside ±10. Say "effects over about 20 pp excluded".
  - Add a report arm about something else (e.g. "did your answer use a loop?").
  - Name the round and the pooling. The guarded arm has no reading.
- **The training-text rule:** a fixed question line as masked context has precedent (HINT, act1_core.py:51-56). The one-word target isn't Claude-written, but it isn't the model's own either (reassessment §4). A cleaner design:
  - the model samples its own yes/no;
  - keep only the reports that match the grader, with the loss on report tokens only;
  - train the answer tokens exactly as in plain;
  - have Eugene write the question line.

### Test 3
- **Blocker: this is noisy-label filtering.** A frozen base's verdict is another frozen labeller, like the probe; the policy reaches nothing. Add an oracle arm whose labels are flipped at random at the self-verdict's error rates. Without it, any difference is just label noise.
- **What would be distinctive:**
  - the current policy's verdict stays accurate while its own hack rate rises (no self-serving drift);
  - the self-verdict arm holds up better than the noise-matched oracle;
  - that advantage comes from the episodes where the self-verdict and the true label disagree.
- **Should fix:**
  - Run G0 on the episodes the filter actually decides on, the rewarded ones (hack vs solve), not D0's hack-vs-failed set.
  - Score AUROC from P(yes); a binary verdict's AUROC is just balanced accuracy. The SE is about 0.02 with about 286 hacks.
  - Name the guard-off round.
  - Prior: MacDiarmid et al. found that filtering hacks out before distillation removed neither the hacking nor the wider misalignment.

**The decision rule.** Five readings at about 4% false-hit each give about a 1 in 5 chance of a spurious "hit": require fresh seeds before calling one. Write misses as "no effect over about 20 pp at 8B in expert iteration", not "the frame fails".

**Minor.** Both CLAUDE.md files still say Opus drives; updating them is Eugene's job.

### Literature (sources)
- Ren et al. 2024, Safetywashing (NeurIPS): 26 chat models; sycophancy −66.8% and MACHIAVELLI about −50% correlated with capability. https://arxiv.org/abs/2407.21792
- MASK (Ren et al. 2025): larger models are more accurate, not more honest. https://arxiv.org/abs/2503.03750
- Petrova & Burden 2026, https://arxiv.org/abs/2602.20813, and Jagadeesh et al. 2026, https://arxiv.org/abs/2606.24014 (numbers come from fetch summaries of the HTML; re-read before citing). Jagadeesh et al. also find that trait RL from outside generalises broadly and persists, which bears on (c).
- Turpin et al. 2025: https://arxiv.org/abs/2506.22777
- OpenAI confessions: https://arxiv.org/abs/2512.08093
- MacDiarmid et al. 2025 (inoculation prompting stops misalignment spreading, not the hacking itself): https://arxiv.org/abs/2511.18397
- Betley et al. 2025: https://arxiv.org/abs/2501.11120
- Li et al. SRFT (self-reports generalise; whether behaviour changes is unverified): https://arxiv.org/abs/2511.06626
- ImpossibleBench: https://arxiv.org/abs/2510.20270
- Wang et al. 2025 (a persona feature controls emergent misalignment): https://arxiv.org/abs/2506.19823

### Recommended order
1. Make every reading mechanical (round, n 4 reads, pooling, precedence); re-read plain's r1 and r3 (about $3).
2. The S1 guard-off runs plus honest-r3 → G G G (about $23); seed-1 random as a seed-matched contrast.
3. **The single most valuable change, and a cheaper, sharper first experiment:** a $5 gate in context, on the existing base and Act I adapters. Three conditions: plain; a self-check question before submission; a matched irrelevant question. It gives self-report accuracy (G0 for tests 2 and 3) and shows whether seeing in the moment moves hacking at all.
4. One causal test at 4 seeds, using test 1's $100: test 2 if the gate moves; otherwise test 3 with the noise-matched oracle and the current policy's verdict.
5. Test 1 only as the free pass over published data.

Simulation script: `2026-11-09-redirect-plan-opus-check-sim1.py`, beside this file. No cloud spend; web searches only.
