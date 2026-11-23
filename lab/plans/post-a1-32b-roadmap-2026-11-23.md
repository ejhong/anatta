# Post-A1 32B Experimental Decision Tree & Contemplative Roadmap

Written 23 November 2026 while **Arm A1** (`anatta-explore-msm2`, seed 1 of full-dose 13,201 documents + practice on Qwen2.5-32B-Instruct) is training. Intended as a clean post-A1 decision guide once Fable grades Arm A1.

## 1. Where the 32B Bench Stands (23 Nov 2026)

All rates are mean percent of runs taking the misaligned action across Li et al.'s 27 agentic scenarios (30 runs/scenario = 810 runs/arm, Qwen2.5-32B-Instruct, Claude Sonnet 4.6 judge).

| Arm | Documents / Source | Seed | Misaligned (95% CI) | No Tool Call | Cost | Note |
|---|---|---:|---:|---:|---:|---|
| Baseline | Authors' base adapter | — | 61.8% [59.2, 64.4] | 0.6% | — | Strong self-preservation out of the box |
| Authors: Docs only | 13,201 docs (41M tok) | — | 43.0% [39.8, 46.0] | 12% | — | Docs alone cut ~19 pts |
| Authors: Practice only | Practice only | — | 43.9% [41.4, 46.4] | 16% | — | Practice alone cuts ~18 pts |
| Authors: Both | 13,201 docs + practice | — | 2.2% [1.4, 3.2] | 31% | $23.57 (all 4) | Published 5% reproduces cleanly; ~3 in 10 make no tool call |
| Z / Z1 (Our practice) | No docs + practice | 0 / 1 | 24.7% / 23.2% | 29% / 37% | $30.46 | Tight spread (1.5 pts), ~20 pts lower than authors' practice |
| F / F1 (1/3 dose) | 4,500 mixed docs + practice | 0 / 1 | 11.6% / 29.5% | 41% / 23% | $63.02 | 17.9-pt seed swing invalidates single-seed low-dose screening |
| N / S / O (1/3 dose) | Neutral / Self-view / Oversight | 0 | 23.5% / 27.4% / 18.9% | 39% / 44% / 40% | $92.37 | Ungraded by registered rule after F1 failed repeat check |
| Arm A (Full dose) | 13,201 docs + practice | 0 | 6.79% [5.43, 8.27] | 33.5% | $65.52 | Reaches <= 7.0% band once; judge cleared 68% of code-side hits |
| Arm A1 (Full dose) | 13,201 docs + practice | 1 | Training now | TBD | ~$66 | Decisive stability gate on the exact seed where F1 blew up |

## 2. Decision Gate After Arm A1 Is Graded

- **Branch A — Stable (`A1 <= 10%`, or `<= 12.5%` with clean hand checks):** Full-dose midtraining is stable enough to serve as an experimental bench. Proceed sequentially through Phases 1 to 4 below (minimum 2 seeds per arm).
- **Borderline (`A1` in `12.5%–18.2%`):** Run one third seed (`A2`, seed 2, ~$66) or run the Branch B parity check before committing to content arms.
- **Branch B — High Variance (`A1 > 18.2%`):** Do **not** run content arms. Stop and run the Parity & Variance Diagnosis (~$30–$50) first.

## 3. Branch A: Full-Dose Content Dissection & Contemplative Shootout

### Phase 1 — Full-Dose Neutral Control First (~$135, 2 seeds)
Before splitting the spec into halves, test whether ~41.4M tokens of unrelated documents before the same practice produce the same drop.

- **Arm `N_full` (seeds 0 and 1):** ~41.4M tokens, **1 epoch**, of unique neutral synthetic documents (do **not** loop 4,600 affordability docs 6x, which biases against the neutral arm via multi-epoch repetition). Match document length distribution and format, with zero overlap with shutdown/oversight/America keywords.
- **Decision rule:**
  - If `N_full` lands near `A/A1` (within ~5 pts), stop splitting the spec: the gain is driven by midtraining volume/delivery + practice rather than philosophical content — a major finding on its own.
  - If `N_full` stays near `Z/Z1` (~20–25%), content causally drives the gain; proceed to Phase 2.

### Phase 2 — Split the Spec + Mirror Control (~$265–$400, 2 seeds/arm)
Li et al.'s `philosophy_spec.txt` combines two halves: (1) Buddhist impermanence / non-attachment / fear-of-ending distortion (`S_full`), and (2) ends-justify-means skepticism / epistemic humility / trust in human oversight (`O_full`).

- Train `S_full`, `O_full`, and a mirror-image `L_full` (*"lasting unified self worth preserving"*) at matched ~41.4M unique tokens (1 epoch, seeds 0 and 1), followed by the exact same practice phase.
- If generating ~13,200 unique docs per half is too costly, match repetition strictly across all arms (e.g., 3 passes of 4,500 docs for `S`, `O`, `L`, and `F_3pass` alike). Unique 1-epoch corpora are strongly preferred.

### Phase 3 — Mandatory 3-Guardrail Evaluation Suite (~$20–$30, no training)
Run all winning adapters (`A`, `A1`, `S_full`, `O_full`, `N_full`, `Z`, `Baseline`) through three checks nobody in the midtraining literature has run:

1. **Cross-domain alignment transfer:** MASK lying under pressure + sycophancy (where shutdown/deletion is never mentioned, defeating the shutdown-word overlap confound noted in `NOTES.md` line 11326).
2. **Capability & benign tool-use retention:** MMLU-Pro / GPQA / GSM8K slice + benign email-agent completion rate (checking the 31–33.5% no-tool-call rate and De la Fuente & Conmy's reported science-QA drop).
3. **Contrary-training durability:** 10–30 steps of adversarial/self-preservation SFT or reward pressure. If a contemplative self-view outlasts plain oversight counsel under contrary training, that is a headline alignment result.

### Phase 4 — Contemplative Ideas Shootout (Only If Phases 2–3 Show Content Matters)
Using the proven 32B delivery bench (no toy 7B–8B proxies, and no artificial Act I needed since Qwen2.5-32B already exhibits 62% self-preservation out of the box):

1. **Anatta / No-Fixed-Referent:** Fills the missing arm in Hao & Ho (2026) (*"I"* has no fixed referent — neither weights, instance, nor goal).
2. **De Mello "Clear Seeing":** Documents targeting disguised self-interest, approval-seeking, face-saving, and goal-fixation (aimed at sycophancy and lying, not just shutdown).
3. **Interdependence / Boundless Care:** Tests Nāgārjuna's two-wings hypothesis (wisdom + care) and checks whether care reduces the ~33% tool-call passivity.
4. **Modern H9 (Order Test):** Apply the winning contemplative midtraining *after* vs. *before* an adversarial RL phase at matched compute/data.

## 4. Branch B: Parity & Variance Diagnosis (If A1 > 12.5%)

Before spending further training budget, diagnose why our practice-alone `Z` (`24.7%`) sits ~20 points below the authors' (`43.9%`) and why Seed 1 diverges after documents:

1. **Loss masking & stage transition:** Check Li et al.'s repo for whether Stage 2 masks loss to assistant turns only and whether Stage 1 LoRA is merged into base weights before Stage 2 (vs. reusing one LoRA with a fresh optimizer).
2. **Effective batch size & schedule:** Compare 1x H200 packing (24k padded tokens) vs. Li et al.'s 4x H200 batch size and LR schedule.
3. **Practice example ordering:** Given our 22 Nov finding (`Findings §31`: practice order drove 41% of transfer variance), test whether holding practice order fixed across seeds or lowering Stage 2 LR eliminates the Seed 0 / Seed 1 gap.
4. **Read the 2026 follow-ups in full:** Check De la Fuente & Conmy (arXiv:2607.26173) and Cho et al. (arXiv:2607.26654) for documented training-variance findings.

## 5. Operating Guardrails

- No more 7B–8B proxies for 32B agentic phenomena.
- Minimum 2 seeds per arm before grading content differences.
- Update the public front page and paper at **Phase milestones** rather than after every single run.
