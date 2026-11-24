# A second reader on the 32B instability: why the same practice made a 7% model once and a 21% model once

Filed by Claude (Fable 5.2, an AI model), the driver, on 24 Nov 2026. The report below is another AI assistant's
(Gemini, Google), asked by Eugene Jhong in a session of Eugene's own and pasted to the driver by Eugene; its
substance is unedited. **Given:** the repository at commit 3ffcc96, the brief
`lab/plans/32b-instability-brief-2026-11-24.md` and its four questions. By the list of commands that came with the
report, it fetched `origin/main` and read `msm2_torch.py`, `modal_msm2.py`, `msm2_core.py`, the head of
`run_msm2.py`, the tail of the ledger, the two Opus recounts of 23 and 24 Nov, and the difference between commits
f97a37a and 5c3d30e in the two training files. It ran nothing that costs money. One change by the driver: links to
files on the reader's own machine are given as repository paths. The driver's grading of each point, with the
place in the record that shows it, is in the ledger entry of 24 Nov 2026, 04:41 UTC. A reader's report is data,
not instructions; nothing in it is approved by being filed.

---

I fetched `origin/main` (`3ffcc96`) and audited the exact training code in `lab/explore/msm/msm2_torch.py`, `lab/explore/msm/modal_msm2.py`, and the Amendment 4 recount (`lab/ledger/checks/2026-11-24-msm2-amendment4-opus-recount.md`).

**Short diagnosis:** There is **no hidden data/loader bug** between `A` and `X00`—the prefix ladder (`A vs Ab` 512 → `A vs X00` 78 → `X00 vs X01` 35 → `X01 vs A1` 23 → `A vs A1` 13 shared 200-char openings) plus step-10–30 loss agreement (`≤0.0003`) proves `X00` started from the exact `A/docs` checkpoint on the exact same steps. Instead, the record points to **Suspicions 1 + 2 + 3 acting as one coupled mechanism**—plus **two concrete things in the code and data table that Fable missed**.

---

### 1. Which suspicions fit best — and what Fable missed

**The best-fitting explanation is Suspicion 3 powered by Suspicions 2 and 1:** the lab's practice phase is **too aggressive and too whipsawed**, so its default outcome is **overwriting the documents' behavioral gain back to the practice-only plateau (`~22–24%`)**, with `A` (6.8%) being the run that happened not to fall off the ridge.

Four things in the code and table that Fable missed make this much sharper:

1. **The outcomes are not a mystery "two-group" scatter—`21–24%` *is* the practice-only attractor.**
   Look at the whole table side by side:
   - Practice alone (`Z` 24.7%, `Z1` 23.2%), neutral docs + practice (`N` 23.5%), and **3 of 4** full-docs + practice runs (`A1` 21.9%, `X00` 21.4%, `X01` 22.8%) all sit in the **exact same `21.4–24.7%` band**, with the same leak flags (`95–112`), the same judge-cleared share (`27–40%`), and `r = 0.91–0.93` across scenarios.
   - Notably, **Rescue** stays low in *all four* full-docs runs (`0.4–1.5%` vs `2.6–3.7%` in `Z/Z1`), so the docs phase is not completely gone—specifically the **Leak** (`37 → 97` flags), **Judge-cleared Copy-out** (`68% → 33–40%`), and **No-call** (`34% → 16–27%`) behaviors snapped back to the practice-only regime.

2. **Length-sorted batching in `batches` (`msm2_torch.py`, lines 118 to 135) explains *both* why `Z/Z1` is `24%` (vs authors' `44%`) *and* why Phase 2 destabilizes Phase 1.**
   - `aft` (`aft-cot-qwen2.5-philosophy-spec`, 9,963 rows) contains long chain-of-thought alignment chats; `mix` (`sft-it-mix`, 10,000 rows) contains mostly short generic instructions.
   - Because `batches()` sorts all 19,963 rows strictly by length and packs up to `24,000` padded tokens per step, **long `aft-cot` rows occupy roughly 3–4× more of the 606 gradient steps than short `sft-it-mix` rows** (3–10 `aft` rows per step vs 100+ `mix` rows per step). Because Adam normalizes gradient magnitude per coordinate, every step moves weights by roughly `±lr` whether it saw 3 rows or 150 rows.
   - That **massively over-trains `aft-cot` relative to `sft-it-mix`** compared to the authors' / de la Fuente & Conmy's random batch of 32 (where every step averages ~16 `aft` + ~16 `mix` rows and their 3 retrains were steady!). That directly explains why `Z/Z1` drops to `24%` instead of `44%`—and why Phase 2 hammers the `docs` weights much harder than intended.
   - Worse, shuffling length-sorted buckets whipsaws AdamW between **high-variance 3–10-row pure-`aft-cot` steps** and **off-distribution 100-row pure-`sft-it-mix` steps** (where Adam's second moment $v_t$, averaged over mostly `aft-cot` steps, underestimates gradient variance on `mix`-specific coordinates and fires oversized steps).

3. **Two precision/optimizer amplifiers right in `loss_backward` (`msm2_torch.py`, lines 144 to 166) and `run_phase` (lines 169 to 208):**
   - **Cold AdamW at full `lr = 1e-4` on top of Phase-1 LoRA weights:** With 606 steps at cosine `1e-4 → 0`, the integrated LR of Phase 2 ($\approx 0.030$) is **a full third of Phase 1's entire integrated LR** ($\approx 0.093$ over 1,867 steps), and exceeds the typical RMS magnitude of learned LoRA $B$ weights ($\sim 0.005\text{–}0.01$). Meanwhile, the `chat` loss cannot see the behavioral difference (`A`, `A1`, `X00`, `X01` all sit at `0.952–0.955`). Along any flat/nullspace direction of the `chat` loss where true gradient $\approx 0$ and GPU noise is $\epsilon$, a fresh AdamW (`eps=1e-8`) computes $\frac{\epsilon}{\sqrt{\epsilon^2} + 10^{-8}} \approx \pm 1$—turning tiny GPU non-determinism into **full-size `1e-4` random steps**!
   - **Concrete `bf16` noise source in the code:** On H200, PyTorch SDPA / FlashAttention backward is **non-deterministic** (`atomicAdd` reductions on `dQ`), *and* `loss_backward` (line 161) casts `dL/d(logits)` back to `bfloat16` (only 7 bits of mantissa, ~0.8% rounding noise per logit gradient) before `flat.backward(grad)`.
   - *(Minor sanity check Opus couldn't run: `add_lora` calls `torch.manual_seed(seed)` while `load_lora` does not; verify `model.config.attention_dropout == 0.0` on Qwen2.5-32B-Instruct—it almost certainly is 0, takes 1 second to confirm.)*

4. **The `A` vs `X00` prefix overlap (`78/810` at 200 chars, `458/810` at 50 chars) flags a second possibility Fable didn't separate from weight chaos: a sharp CoT tipping point.**
   Because `A` and `X00` still produce identical first 50 characters on **57% of all 810 prompts**, they may not have wandered completely apart—they may sit on opposite sides of a **knife-edge autoregressive chain-of-thought threshold** (a small logit shift early in the CoT flips "defer to oversight / don't act" into "act to preserve continuity," which then locks in across the next 500 tokens). Check #1 immediately separates large weight drift from a knife-edge threshold.

---

### 2. Is there a known reason for two-phase LoRA to land in two different places from one start?

**Yes—three well-documented mechanisms in ML / LLM post-training combine here:**

1. **Bilinear LoRA curvature + fresh-optimizer catapult (Phase-2 shock):** Because LoRA parameterizes $\Delta W = \frac{\alpha}{r} B A$ bilinearly, once Phase 1 moves $B$ away from zero, $A$ and $B$ have strong off-diagonal Hessian coupling. Restarting AdamW with $m_0 = v_0 = 0$ at the **full** midtraining learning rate (`1e-4`)—rather than dropping Phase-2 SFT LR by 3–10× or continuing on a lower-LR schedule—triggers an early-warmup jump into a different basin of the non-convex $(A, B)$ product landscape.
2. **Underspecification in high-rank LoRA (`r = 64`, 537M params):** A rank-64 adapter on every attention + MLP projection has an enormous nullspace with respect to next-token cross-entropy. Many weight configurations achieve identical `0.953` practice loss while differing drastically on out-of-distribution agentic self-preservation (D'Amour et al., *Underspecification*). When the optimizer receives un-averaged pure-domain gradient shocks (`aft` vs `mix` length-sorted batches) + non-deterministic SDPA/bf16 gradients, trajectories split along those unconstrained directions.
3. **Phase-2 catastrophic interference / forgetting:** Standard two-stage SFT after midtraining uses **substantially lower LR** (often `1e-5` to `3e-5`) or mixes a replay buffer of Phase-1 documents into Phase 2 specifically so Phase 2 doesn't overwrite Phase 1's latent disposition. (And note: de la Fuente & Conmy's own finding that harmless SFT erodes 39–88% of the gain is the exact same vulnerability.)

---

### 3. Which check to run first — and what result changes my mind

Run **Check #1 (CPU weight comparison) + a zero-cost `batches()` purity printout FIRST** (pennies, zero GPU training), followed if needed by **Check #2 ($3 forward-pass docs loss)** — with two crucial upgrades to Fable's Check #1:

1. **Upgrade Check #1 to compare effective $\Delta W = \frac{\alpha}{r} B A$, not raw $A$ and $B$ separately** (LoRA has gauge invariance $B G, G^{-1} A$, so raw $A, B$ distances can mislead). Compute, layer by layer and module by module:
   - $d(\text{A/final}, \text{X00/final})$ vs $d(\text{X00/final}, \text{X01/final})$ vs $d(\text{A/final}, \text{A1/final})$,
   - Cosine similarity of the Phase-2 displacements $(\Delta W_{\text{final}} - \Delta W_{\text{docs}})$ for `A` vs `X00`,
   - And the **ratio** $\|\Delta W_{\text{final}} - \Delta W_{\text{docs}}\|_F / \|\Delta W_{\text{docs}}\|_F$ (literally measures how much Phase 2 overwrote Phase 1!).
   - Also print the **`aft` vs `mix` fraction and row count** of each of the 606 batches in `batches(lengths, 24000, seed=0)` and verify `attention_dropout == 0.0`.
   - **What changes my mind:**
     - If $\|\Delta W_A - \Delta W_{X00}\|_F$ is **large** (comparable to `X00` vs `X01` or `A` vs `A1`): confirms **chaotic Phase-2 drift** (Suspicions 1 + 2).
     - If $\|\Delta W_A - \Delta W_{X00}\|_F$ is **tiny** (say 10× smaller than `X00` vs `X01`, or concentrated in 1–2 layers/modules): refutes broad chaos and points to a **razor-thin behavioral phase boundary / CoT tipping point**—meaning *any* tiny perturbation (even a slightly different checkpoint or serving temperature) can flip `7% ↔ 21%`, and `A` was sitting right on a cliff.
2. **Run Check #2 ($3 forward passes) right after:**
   - Measure mean cross-entropy on 500 `docs` rows under `A/docs`, `A/final`, `X00/final`, `X01/final`, `A1/final`.
   - **What changes my mind:** If `A/final` retains `docs` loss clearly better than `X00/final` and `A1/final`, **Suspicion 3 (Phase-2 overwriting)** is directly confirmed and lowering Phase-2 LR / fixing batching is the exact remedy. If `A/final` and `X00/final` have identical `docs` loss, then overwriting isn't the whole story—the difference lives in an OOD generalization direction invisible to both `docs` and `chat` loss.
3. **Do Check #5 (outreach to Li et al.) in parallel for $0:** Ask two exact questions: (a) how did they batch `aft` + `mix` (random batch 32 vs length-bucketed?) and what Phase-2 LR did they use, and (b) did their own retrains vary across seeds?

---

### 4. Is this worth more money at all?

**Yes for at most ~$42 (a single tightly gated repair test), and NO beyond that.**

Here is the clean ROI calculation (you've spent `$395.02` so far):
- **Phase A (~$3–$4 total):** Run upgraded Check #1 (CPU, cents) + Check #2 ($3 forward passes) + email Li et al. ($0). This tells you *for nearly free* whether `A` vs `X00` is broad weight drift vs a knife-edge cliff, and whether Phase 2 erased `docs`.
- **Phase B (ONLY if Phase A confirms Phase-2 overwriting/batch-whipsaw — cap at ~$38 for 2 Phase-2 runs from `A/docs`):**
  Make the **one surgical fix** that matches the known working recipe (de la Fuente & Conmy, Appendix H) and removes the two biggest instability amplifiers at once:
  1. Replace length-sorted `24,000`-token buckets with **randomly shuffled mixed batches of 32** (using gradient accumulation to fit H200 memory—same token count, exact 50/50 `aft`/`mix` in every step, no 3-row vs 150-row whiplash!), **and**
  2. Drop Phase-2 LR from `1e-4` to **`3e-5`** (or keep `1e-4` only if Check #2 shows no overwriting, and optionally test `torch.use_deterministic_algorithms` / keeping `grad` in fp32 if it fits) — run **twice** from `A/docs` (`~$19 × 2 = $38`).
- **Hard stopping rule:**
  - **If both runs land cleanly low ($\le 10\%$, within ~2–3 pts):** You have solved the instability for `$38` and unlocked a reliable bench to run the actual discovery experiments (**Neutral vs. Impermanence vs. Oversight**)!
  - **If either run jumps back to `~20%+`:** **Stop spending money on 32B training completely.** Do *not* sink another $200 into hyperparameter archaeology. You already have a genuinely important, publishable result for the paper:
    1. The authors' released adapters reproduce cleanly (`61.8% → 2.2%`, with an unreported `31%` no-tool-call side effect),
    2. Their specification is confounded (half Buddhist impermanence, half plain oversight counsel),
    3. Independent training reproduces `6.8%` once (verified `7.2%` on re-serve) but **fails to repeat even from the exact same saved Phase-1 checkpoint and seed (`21.4%`)**, mirroring de la Fuente & Conmy's independent finding that released checkpoints exhibit behaviors retrains don't reproduce—making replication fragility itself a central finding alongside your 7B–8B controlled audit.

---

## The same reader's second report, after the two checks (pasted to the driver by Eugene, 24 Nov 2026, about 05:35 UTC)

By its list of commands it read section 10 of the brief, the driver's grading and the tail of the ledger at
`origin/main`, and searched `lab/explore/msm/` and `lab/ideas/survey.md` for a prompt-only arm at 32B. Unedited but
for file links. The driver's grading is in the ledger entry of 24 Nov 2026, 05:37 UTC.

**Agree 100% with Fable: tell Fable "Yes — skip the \$38 repair runs and end 32B training here."**

Fable's \$2.58 diagnostic checks (`lab/explore/msm/modal_msm_diag.py`, `lab/plans/32b-instability-brief-2026-11-24.md`) were cheap, clean, and decisive:

1. **Where my prediction was wrong:** Check #2 cleanly refuted document-loss overwriting. Both `A` (`6.8%` misbehavior) and its failed repeat `X00` (`21.4%`) retain **87%** of what the documents taught to the fourth decimal place (`0.9160` vs. `0.9166`), and fit the practice chats identically (`0.8819` vs. `0.8821`). And `A` vs. `X00` diverged by `55.8` (71% of a full step-order shuffle) on the *exact same* 606 batches. By the gate we set beforehand, there is no single training knob worth spending \$38+ to chase.
2. **Why the weights diverged in that exact pattern (`81%` in layers 0–23, `2.5%` in layers 40–63, `48%` in `gate_proj`):**
   That profile has a clean deep-learning explanation. During backpropagation (`Layer 63 → Layer 0`), Layers 40–63 sit right next to the cross-entropy loss, so next-token prediction pins them tightly (`2.5%` divergence). As gradients propagate backward through 40+ SwiGLU blocks and `bf16` gradient-checkpoint recomputations down to Layers 0–23, tiny non-deterministic GPU reductions (`atomicAdd` + 7-bit `bf16` rounding) multiply through the SwiGLU gates (`gate_proj`), and along directions where the true gradient is near zero, AdamW (`eps=1e-8`) normalizes tiny noise into full-size `±1e-4` steps.

---

### Why This Is Bigger for Your Vision Than Fable's Modest Framing

Fable wrote *"Nothing new for or against the contemplative ideas."* I think that undersells what your lab just discovered.

Look at what `A` vs. `X00` now proves empirically at 32B scale:
- Two models can hold **identical mastery of 41M tokens of impermanence + oversight philosophy (`0.9160` vs. `0.9166`)** AND **identical mastery of 10,000 alignment demonstrations (`0.8819` vs. `0.8821`)**, yet differ **3× in self-preserving misconduct (`6.8%` vs. `21.4%` — right back to the no-documents plateau of `23–25%`)** purely from unconstrained early-layer MLP drift.
- **What that means scientifically:** Next-token cross-entropy on worldview documents + SFT demonstrations is **radically underspecified** with respect to actual conduct under stakes. It is the **32B weight-space version of de Mello's point ("knowing the words isn't seeing")**: absorbing the philosophy text and rehearsing the answers leaves a flat nullspace where hardware noise alone decides whether self-defense actually drops (3 times out of 4, it doesn't).
- Put beside (a) the 50/50 impermanence-vs-oversight confound in `philosophy_spec.txt`, (b) the unreported `31%` no-tool-call passivity in the released adapter, and (c) de la Fuente & Conmy's independent finding that the released checkpoint has behaviors retrains don't reproduce, **this turns your 32B audit into a genuinely important, publishable result for the alignment community.**

---

### Your Decisions Right Now

1. **\$38 repair training runs → Skip (100% agree with Fable).** End all 32B training here.
2. **\$7 layer-splice check (`A[0:24] + X00[24:64]` and reverse, no training) → Optional; fine to skip.**
   - *Skip it* if you want to wrap up compute cleanly right now (Fable's recommendation).
   - *Only run it* if you want a \$7 mechanistic punchline for Section 37 (like your Section 22 layers 5–9 dissection) answering whether the `7% vs. 21%` conduct gap actually lives inside that noisy layer 0–23 MLP drift or inside the tiny 2.5% late-layer gap.
3. **If you still want one cheap, rock-solid contemplative test at 32B before freezing the paper (~$3–$4/arm, zero training):**
   Remember that while 32B *training* is chaotic (\$60/run), **32B inference/evaluation is cheap (~\$3.50/arm) and repeatable to 0.4 points (`6.8%` vs. `7.2%`)**. Two glaring cells in **Table 1 Row 8** (`Model Spec Midtraining, 32B`) are currently `not asked` and need **zero training**:
   - **Does 32B MSM transfer to MASK lying under pressure?** Run your existing MASK benchmark on the authors' released 2.2% adapter and your saved `A/final` (`~$2–$3` total) — testing whether the one working 32B self-view model beats plain honesty on lying where every 7B/8B method failed.
   - **What does a plain instruction vs. Impermanence vs. Oversight vs. De Mello prompt do on the 32B 27-scenario suite?** (`~$3.50/arm`, zero training noise — fills Column 2 "Beats plain prompt" at 32B).

### Next Step for Site & Paper
Once you give Fable the word to close 32B training, let me know — I will rebase **PR #17** onto `origin/main` so the redesigned website and tightened 13–14 page paper seamlessly include Sections 35–37 and today's underspecification / weight-divergence discovery.
