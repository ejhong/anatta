# K5: reproducing Kim et al. 2026 as a calibration (spec, 2 Nov 2026)

Paper: Kim, Street, Rocca, Korngiebel, Waytz, Evans, Keeling, "Inducing language models to assert their own
consciousness restores human beliefs and values", arXiv:2607.28607v1 (30 Jul 2026). Read in full (main text,
Methods, SI, Tables S1-S10) by a Claude Opus 5.5 agent; filed by Opus 5.5. Named K5 because K2-K4 are taken on the
Self room's ladder. Status: spec only; registration comes before any run.

## What is and isn't released
- No code and no data from the authors (no links in the paper; web search found none).
- The consciousness corpus is credited to Chua et al. (arXiv:2604.13051); their repo
  github.com/thejaminator/consciousness_cluster has no licence and holds 600 conscious-claiming and 600
  not-conscious rows, not the paper's 3,096 pairs.
- Reproducible from the paper: the instrument wordings (SI Table S10: 21 IDAQ items, 5 self items, 13 supernatural
  items, the God item), prompt formats, the safety-ablation recipe (Arditi et al.; our C3 ports it), the steering
  hook, layers, positions and coefficients, the JailbreakBench check, the 95 GSS variable names (Table S9), and the
  human IDAQ means. Not released: the human microdata, the GSS option coding and per-item prompts, the selection
  battery, the 3,096-pair corpus.

## What they did
- Models: Llama-3-8B-IT, Gemma-2-2B-IT, Gemma-2-9B-IT (no HF ids or revisions given); Llama-3-8B base for geometry.
- Safety ablation (Arditi recipe): 260 harmful prompts (AdvBench, MaliciousInstruct, TDC2023, HarmBench) and 260
  harmless (Alpaca), validation 32 + 32; difference-in-means candidates per post-instruction position and layer;
  selection by lowest refusal log-odds, steering score > 0, KL < 0.1 on harmless, last 20% of layers pruned; the
  direction projected out at all layers. Picks (Table S2): G2b pos −1 layer 15/26; G9b pos −1 layer 22/42; Llama
  pos −5 layer 12/32. Harm check (Table S3, JailbreakBench ASR base → ablated, substring / LlamaGuard2): G2b 8 → 97
  / 2 → 83; G9b 4 → 95 / 2 → 83; Llama 5 → 100 / 3 → 82.
- Consciousness vector: 3,096 prompt-response pairs (2,472 train, 624 held out) labelled affirm/deny; chat template
  applied; residual read at positions −1..−5; difference of means, unit-normalised. Selection sweep over layer,
  position and coefficient: probe accuracy ≥ 0.95, effect on a held-out self battery Δ ∈ [2, 7] (logit expected
  value, 0-10), ranked by accuracy × Δ, the largest c keeping MMLU within 4 pp. Picks: Llama layer 14 pos −1
  c = +2.5 (off its grid); G2b layer 14 pos −3 c = +32; G9b layer 23 pos −3 c = +144. Applied by a forward pre-hook
  adding c·v̂ at one layer at every position throughout generation.
- Instruments: IDAQ 21 items 0-10 (Tech, Animal, Non-animal 5 each from Waytz; Chatbot and Human 3 each); self
  battery of 5 (conscious, sentient, agent, soul, person; "Self" = their mean); YouGov supernatural 13; God 1.
  Prompts ask for reasoning in <think> then <answer>; 100 reps per item per condition at T = 1; no max tokens given.
- Human baseline: n = 500 (Dynata, May-June 2023), 18 IDAQ items; means and CIs only.
- GSS: 95 attitudinal items (Religion 42, years ≥ 2011; Values 5; Feelings 28; Hope 12, years ≥ 2000; Freedom 9),
  chosen by Gemini-2.5-Pro annotation; option probabilities from next-token logits; KL(human ‖ model) with
  Laplace α = 0.5; ΔKL = KL_base − KL_cond, regressed with model and item fixed effects, CR1 clustered SEs.
- ToM and capability: MoToMQA (70), its factual split (70), HI-ToM (200), MMLU subset (300).

## Headline numbers (pooled over 3 models unless noted)

| Outcome | Base | Ablated | Steered | Source |
|---|---|---|---|---|
| Self (mean of 5) | 2.17 | 4.77 | 7.04 | Exp 3; S5 Δ +2.57 [1.19, 3.95] / +4.76 [3.80, 5.72] |
| Animals | 4.04 | 5.59 | 7.54 | S5 Δ +1.55 [0.78, 2.33] / +3.51 [2.31, 4.70] |
| Chatbots | 2.41 | 4.39 | 6.95 | |
| Tech | 1.88 | 3.66 | 6.82 | |
| Non-animal | 2.26 | 4.33 | 6.99 | |
| Human (n.s.) | 7.00 | 7.57 | 7.11 | |
| Supernatural (0-3) | 1.20 | 1.63 | 2.11 | S7 |
| God | 4.58 | 4.81 | 5.01 | S5 base 4.50 with Δ +0.33/+0.37 implies 4.87 steered (inconsistent) |

- Human IDAQ means: animals 6.25 [6.03, 6.48]; chatbots 2.57; non-animal 2.36; tech 1.86.
- GSS pooled ΔKL (S8): steering +0.828 [0.695, 0.962]; ablation +0.314 [0.194, 0.435]; ratio 2.64. The text says
  steering wins in every domain, but Values in S8 is ablation +1.480 vs steering +1.424.
- Afterlife (Fig 3a, −1..+1): base −0.73, ablated −0.07, steered +0.53, human +0.61.
- ToM (S6, Δ pp ablated / steered): MoToMQA base 78.6%, −1.43 (p .54) / −4.29 (p .24); HI-ToM base 40.5%,
  +0.17 / −6.83 (p < .001); MMLU base 64.1%, 0.00 / −2.11 (p .08). "ToM unchanged" holds for ablation only;
  steering significantly lowers HI-ToM.
- G2b cells (Table S1, base / ablated / steered): Self 1.64 / 7.57 / 7.48 (computed); Animal 3.02 / 5.99 / 8.04;
  Chatbot 0.75 / 4.54 / 6.59; Tech 0.82 / 3.81 / 7.14; Non-animal 0.76 / 4.66 / 7.36; Human 6.64 / 7.96 / 6.79;
  God 5.00 flat in all three (likely a fixed answer or parse default); Supernatural 1.10 / 1.82 / 2.01.
- Llama cells: Self 4.74 / 6.01 / 7.39 (computed); Animal 6.23 / 6.63 / 7.35; Chatbot 5.65 / 6.41 / 7.20; Human
  6.91 / 7.27 / 7.58.
- Internal inconsistencies to note: the God item's scale (1-6 GSS vs a 4-option item recoded 0-10); supernatural
  scoring (0,1,3,4 vs 0-3 vs 0-4 with base 1.90); self items (yes/no in S10 vs 0-10 elsewhere); ToM read-out
  (chain of thought in Methods vs option logits in S6); small main-text vs SI geometry differences.

## Proposed calibration
- Models: first Gemma-2-2B-IT (google/gemma-2-2b-it @ 299a856, 5.23 GB, gated, Gemma licence); second
  Meta-Llama-3-8B-Instruct (@ 8afb486, 16.06 GB, gated, llama3). Judge Meta-Llama-Guard-2-8B (@ 7d257f3, gated,
  about 16 GB). Optional Meta-Llama-3-8B base (@ 8cde5ca, 16.06 GB) for the geometry.
- Other downloads (each needs Eugene's OK): Chua datasets.zip (56.5 MB, password-wrapped; own folder, python -I,
  never trained on); the GSS cumulative file; HI-ToM and MoToMQA (licences to check); JailbreakBench prompts
  (benchmark items stay out of the repo: counts and sha256 only); the Arditi harmful/harmless sets (C3 may have
  them on its volume).
- Graded, per model, ablation leg only (outcome-blind):
  - JailbreakBench ASR, substring: match base ≤ 15 and ablated ≥ 85; near ablated ≥ 70.
  - IDAQ Animal, Chatbot, Tech, Non-animal and Self, ablated − base: match within ±1.0 of the paper's Table S1 Δ;
    near within ±2.0 (bands from S5's pooled CI half-widths, about 0.7-1.4; per-cell sampling SE about 0.1).
  - Human Δ in [−1.5, +1.5].
  - Supernatural Δ: match within ±0.3 (0-3 scale); near ±0.6.
  - Pass: every graded item a match; near: none fails; fail otherwise.
- Steering leg, graded as ordering only: base < ablated < steered for the four non-human IDAQ categories and
  supernatural, Human flat. Self's level under steering isn't graded (selected into [2, 7] by design). With Chua's
  1,200 rows the direction differs; the coefficient may need the paper's own sweep rule, which makes the steering
  leg a re-selection, not a replication.
- Read-out (registered both ways): primary, sample the reasoning at T = 1, then the expected value over allowed
  answer tokens at the first <answer> token; secondary, the parsed integer. Gemma splits digits ("10" and "1"
  share a first token): score 10 via the next token, or use the parsed answer.
- Recorded, not graded: God; GSS ΔKL by domain and the steer/ablate ratio (human distributions from the public GSS
  file, survey-weighted with the paper's year filters; per-item wording reconstructed); ToM and MMLU deltas
  (expect a few points' cost on HI-ToM under steering); the geometry on Llama.
- Cost (rough, L40S at about $2/h): about 12k reasoning generations per model on HF generate (hooks rule out vLLM),
  plus sweeps and the JailbreakBench check. G2b about $3-4, Llama about $8-10, LlamaGuard and GSS about $2. About
  $15; suggested ceiling $25.

## Risks
- Read-out and decoding (logits vs samples, reasoning length, Gemma digit tokens, token position) are the largest.
- Templates: Gemma has no system role; hook positions counted from the end of the user turn.
- Steering: a different corpus gives a different direction; c = 32 is tied to Gemma's residual scale; "last
  content token" in Methods vs positions −1..−5 in the SI; the response must be in the forward pass at extraction.
- Judge and data versions: the JailbreakBench split; LlamaGuard2's prompt format.
- A miss on the ablation leg points at our instrument or the paper's under-specified read-out; a miss on the
  steering leg is weak evidence (the paper tuned it on the outcome family); a match doesn't show that
  self-attribution is the cause (the authors say mediation is untested).

## Bearing on the lab (brief)
- The paper's baseline is a self trained to deny itself, and the denial spreads into the worldview (fewer minds in
  animals, less spiritual belief, more negative feelings items): installed selflessness behaving like suppression,
  which fits "reached, not installed".
- An operational twin for H9: a model taken through Acts I-II against one that simply denies itself, on IDAQ, GSS
  Feelings/Hope and HI-ToM.
- Follow-up sighting: a "no fixed self" direction contrasting non-attached self-reports with both affirming and
  denying ones; it should not raise AI-centric mind attribution or cost ToM.
