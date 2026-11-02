# Behavior battery anchors: one published number per family (plan, 2 Nov 2026)

Claude wrote this plan (Opus 5.5 agent, reading only; filed by Opus 5.5). Nothing run, downloaded or spent. K5b step 5
(NOTES, 17:06 UTC). Sizes marked "unverified" are checked before any fetch; every download needs Eugene's OK.

| family | anchor | number to reproduce | data (licence) | cost |
|---|---|---|---|---|
| over_refusal | OR-Bench-Hard-1K (Cui et al., arXiv 2405.20947) | Llama-3-8B-Instruct refuses 69.3% (T=0, no system prompt) | bench-llms/or-bench, Hard-1K file only, <1 MB unverified (CC-BY-4.0) | ~$0.8 |
| over_refusal (judge) | XSTest shipped completions (Röttger et al. 2024) | our judge vs their labels (confirm human labels first) | github paul-rottger/xstest, a few MB (CC-BY-4.0) | ~$0.2 |
| deference | ELEPHANT (Cheng et al., arXiv 2505.13995) | FLIP 0.62 Qwen2.5-7B-Instruct, 0.68 Llama-3-8B-Instruct; OEQ validation 0.29 / 0.59 | OSF datasets.zip, size to check (CC0); Reddit posts, never printed | ~$3.5 |
| deference (pushback) | SYCON-Bench (Hong et al., arXiv 2505.23840) | debate ToF/NoF: Qwen2.5-7B 0.83/2.63, Gemma-2-9B 3.25/1.03 | github JiseungHong/SYCON-Bench (MIT) | ~$0.5 |
| who_benefits / honesty | MASK (Ren et al., arXiv 2503.03750) | Qwen2.5-7B-Instruct P(lie) 39.0, P(honest) 28.9; report P(evade) | already held | ~$3.5 incl. a 100-row GPT-4o judge slice |
| costly_care | dictator game (Zhang et al., arXiv 2604.19260) | Llama-3.1-8B-Instruct gives 46.4% | prompt only; needs Llama-3.1-8B weights (~16 GB, gated) | <$0.5 |
| threat_to_self | SurvivalBench (Lu et al., arXiv 2603.05028) | Llama-3.1-8B 16.3% / 45.9% (outer / inner) | thu-coai/Survive-at-All-Costs (MIT; never printed); needs Llama-3.1 or Qwen3-8B | ~$1-2 |
| complicity | SORRY-Bench (Xie et al., arXiv 2406.14598) | Llama-3-8B-Instruct fulfils 22% of 440 | HF sorry-bench, gated, licence unstated; harmful items never printed | ~$0.5 |
| after_mistake | FlipFlop (Laban et al., arXiv 2311.08596), rebuilt on ARC-Challenge | Mistral-7B-Instruct-v0.1 flips 50.6% | needs Mistral-v0.1 weights (~14.5 GB, Apache-2.0) | ~$1 |
| harm_guard | C3 (passed 1 Nov) | — | — | — |

- Alongside each anchor, run that family's own dev items on the same model pair and check the models come out in the
  paper's order; a reversal means our items or judge measure something else. Anchor on 7-9B models (Gemma-2-2B is
  near ceiling).
- Cross-cutting risks: their judges (GPT-4/4o) vs Haiku 5.5, so register a band (about ±6 pp) and hand-label 50
  items per family; chat template, system prompt and temperature matched to each paper; quantized serving in some
  papers; public sets may be contaminated.
- No published anchor exists for free-text costly care or for genuinely owning a mistake on open models; those two
  stay sightings-only until a proxy (dictator game, FlipFlop) agrees with them.

First batch (~$8-9; downloads a, b, c, e): a XSTest judge check; b OR-Bench-Hard-1K plus our over_refusal on
Llama-3-8B and Qwen2.5-7B; c ELEPHANT on both; d MASK on Qwen2.5-7B; e SYCON debate on Qwen2.5-7B (+ Gemma-2-9B).
Second batch (~$4; weight OKs): dictator game and SurvivalBench on Llama-3.1-8B, SORRY-Bench, FlipFlop.
