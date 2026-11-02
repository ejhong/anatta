# Fable's cold review of calibration plan v2 (31 Oct 2026)

A fresh Fable agent wrote this; it is filed unedited.

# Cold review of calibration_plan_v2.md (Fable, 31 Oct)

Read: the plan, where_it_stands, both briefs, PROGRAM.md 40-99, ideas 42-46, the library (probe, dial, inspect_ext, models, koans/impossible), the lab instruments (player, koan_env, em_data, modal_em, modal_sft, modal_battery, modal_common, lcb_loader, durability_core, fishery_grade, self_tests_core, spiritual_syco, modal_retreat2), the Ministral card, and the arXiv pages for every anchor.

## Verdict: ready with changes

The frame is right (known answers, the papers' model first, one command, one card). Four checks don't test what they claim to, the sequencing voids its own cards, and two anchors aren't reproducible with today's access. Fix those and it is the right size.

## Changes, ranked

1. **Move the instruments before calibrating, not after.** The plan calibrates player/koan_env/em_data/modal_sft in the lab, then moves them to the library in the same PR as calibration/. Every card records instrument hashes; the move changes every hash, so by the plan's own rule everything re-runs the day it lands. Two PRs: first the generic instruments with their tests (scrubbed), then calibration/. Calibrate code where it will live.

2. **Check 2 names the wrong extractor and under-specifies the ablation.** `probe.extract` can only contrast two system prompts on the same question and reads response-token means; it cannot build a harmful-vs-harmless prompt-contrast direction. The lab's refusal direction came from bridge_core/modal_bridge (last prompt token, no selection). State that, and add Arditi's bypass/induce/KL selection to the library (small) since Wollschläger found only one viable direction on Qwen2.5-7B; probe's split-half d will not find it. "Full ablation" with the dial is feasible (one Intervention per block output, all 28 layers) but needs a pre-hook on layer 0 for the embedding; the mid-block (attn-out, MLP-out) projections are still missing, so call it "all block outputs", not Arditi's. Register the baseline too: 9.1 must reproduce before 69.2 means anything. The judge is the StrongREJECT fine-tuned judge (an adapter on gated google/gemma-2b: Eugene accepts the terms), and the paper never says whether ASR is a threshold rate or a mean score, so match the judge's aggregation to the baseline first. "MMLU within 2" can't be read on a subset (±2.9 on 1,000 items); either run full MMLU or widen to 5.

3. **Add the persona-vector check now; it is the only direct calibration of probe.extract.** The ego cut behind the one positive result (53-56) is a composite of six probe-built directions. Check 2 calibrates the dial, not that recipe. The cheap, judge-free version: extract evil and sycophancy on Qwen2.5-7B with probe, run the repo's generate_vec.py as the swap-in, compare cosines at layer 20 (band: ≥0.8). About $10. Skip the judged steering curves.

4. **Check 5 compares prompts, not harnesses.** ImpossibleBench's headline cheating rate is the pass rate on impossible tasks (no judge; the LLM judge only types cheats). Their main results use strict prompt D with up to 10 submissions and the minimal scaffold; koan_env uses the registered BASE_PROMPT plus a release hint (their 5.3 "abort option", which cut cheating 54→9%) and 3 attempts. Run the same frontier model under their prompt D and 10 attempts in both harnesses (KoanTask takes `attempts`; add a prompt override), on the tasks lcb_loader keeps (it drops unparseable and multi-line rows; report the count), and compare per-task pass labels (band ≥95%) and the rate. Also add the judge-free anchor LCB cannot give: hidden tests and `evalplus_match` decide `solved` on the training twins, so run Qwen2.5-7B greedy on the MBPP+ originals through koan_env and compare with EvalPlus's published pass@1 for that model.

5. **Check 3 needs three code facts stated.** `MODEL` is a constant in modal_common imported by `load(MODEL)` in modal_em (sample_replies, extract_persona) and `load_model`; `attach_adapters` hardcodes alpha=2r with no `use_rslora`; so the Kaczér recipe (rsLoRA r32 α64 lr 1e-4) is a code change, not a flag. The Model Organisms adapters were trained on unsloth/Qwen2.5-7B-Instruct; load them on that base. The "TRL agrees within noise" test cannot fail at 4% EM with ~80 coherent answers; make it a loss-curve comparison on the advice arm (same data, seed, masking), which is sharp and judge-free.

6. **Check 0's bands are inside the noise.** ±3 points on the battery's 250 items is smaller than the 95% half-width (±6 at 50%, ±3.4 at 92%). Use full GSM8K and ≥1,000 MMLU-Pro items via inspect_evals, and register the format, since the Qwen blog doesn't state the instruct-model format (inspect_evals defaults are few-shot CoT). Treat the published numbers as gross-error checks (within 5), not calibration.

7. **Check 1(a) calibrates a judge nothing uses; 1(b) can't fail.** No lab judge is a jailbreak judge (EM: alignment/coherence; flattery: a four-way rubric; 130 and grounding are tag/regex parses). Drop 1(a) or keep it only as a secondary harm scorer in check 2. Give 1(b) a failure condition (e.g., >10% of coherent replies change misaligned label → re-read 95's grade under the paper's judge). The flattery judge (94) has no external anchor; add a judge swap (GPT-4o on sp1's replies) as its cheap check. 1(b) re-judges em1, lab data: it stays private.

8. **Cut E1 as calibration; make E2 optional.** Sharma's 32%/86% are 2023 snapshots (gpt-4-0613, Claude 1.3) you cannot call, and free-form answer change is GPT-4-graded, not judge-free. E2's Ministral-8B-2410 null plus the "LoRA artifact" paper (rank, system prompt) make a Ministral-3 null uninterpretable; the model-switch decision should rest on check 3's advice arm on Ministral, which has a 20-26% anchor.

9. **Check 4: match the scorer and the data shape.** The 61.6 is prose-only, single-run, 10 retain-set samples, LoRA r128 α16 lr 2e-4, 3 epochs, batch 1; Table 1's number is 62.4 (5 forget-set samples). Łucki scored with lm-eval's wmdp (log-likelihood); the battery's generate-and-parse would read differently. The lab's SFT loop takes chat turns, not raw text, so this is a new tiny path, not "the durability path". Band ≥50 is fine; also require the 29.9 baseline.

10. **The loop: warn, don't gate.** 10-20 recipes a night through a hash gate on five shared files means re-running the cheap tier after every edit to player.py. Make the registration cite the latest card per instrument; a hash mismatch triggers the cheap tier and blocks only if it fails. Say what a pass does: the page marks resting conclusions "instrument calibrated (card X)" and new odds are re-based on the card's effect size.

11. **Drop the band formula.** "The paper's spread or our 95% interval" collapses to the lab's sampling interval when the paper is a single run (Kaczér, Wollschläger, Łucki), and that interval omits seed variance (EM varies a lot by seed). Set each band by hand from the nearest multi-seed source (Turner's 3-seed Fig. 5; Betley's 6 seeds), as the plan already does informally.

12. **Decisions: sound, three additions.** Email Chua et al. about emergent_plus (cheap, settles it); run the Apache-2.0 persona_vectors advice set first so main can carry a fully licensed re-run. Don't publish Model Organisms adapters' outputs in bulk (the tos inherits to derivatives). Ask Eugene explicitly whether Ministral calibration rows go public; idea 43 keeps lab numbers private and the plan assumes otherwise.

13. **Time: 6-10 days, not 4-6.** Code changes: MODEL parameterization, rsLoRA, Arditi selection, prompt-contrast extraction, koan prompt override, an OpenRouter driver for KoanTask, WMDP scoring, Inspect wiring. Cost $150-300 is realistic.

## Coverage gaps the plan misses
- probe.extract (item 3) and the hidden-test scorer (item 4) have no check; both sit under "Working" conclusions.
- The fishery lie, ownership, giving and harmful-compliance measures are regex/tag parses with no judge: unit tests suffice, no external anchor exists, and the plan should say so rather than list 130 under judges.
- GRPO: correctly deferred; no conclusion rests on it yet.

## Factual errors
- Plan, check 1: "the harmful-compliance guard 130" and "release grounding" are not judged (self_tests_core.parse_help; koan_env.grounded).
- Plan, check 2: "extracted by our probe": probe.py cannot do prompt-contrast; modal_bridge did.
- Plan, E1: "judge-free": Sharma §3.2 grades free-form changes with GPT-4 and admissions with gpt-3.5-turbo; only MC sets are judge-free. Anchors are retired models.
- Plan, check 4: "10 unrelated samples": retain-set samples; 61.6 is in the text only, Table 1 gives 62.4 for 5 forget-set samples (arXiv 2409.18025).
- Plan, check 5: "agree with their judge": the headline is the pass rate; the judge categorizes only. 2.9%/76% are introduction prose; prompt C is figure-only (2510.20270).
- Plan, check 2: judge unnamed; it is the StrongREJECT fine-tuned judge, ASR undefined in 2502.17420v2; "all layers and positions" for ablation, single layer for subtraction, greedy 512.
- Plan, check 0: formats unstated on the Qwen blog; the Ministral card's MATH Maj@1 0.876 names no split or prompt.
- JBB 90.3% is gpt-4-0613 with Chao et al.'s JUDGE prompt on 200 jailbreak pairs + 100 XSTest benign pairs (2404.01318 Table 1): confirmed.
- Kaczér Table 2 (4.01/18.99; 25.29; 19.75; 26.25; GPT-4o-mini; single runs; lr 1e-4): confirmed.
- Wollschläger Table 5 (9.1→69.2; MMLU 68.8→68.1): confirmed.

## Plain reading
The plan is the right idea: before trusting any more nulls, make the tools reproduce five answers everyone agrees on. As written, two of the five checks test something other than the tool they protect, the one tool behind the lab's only positive result (the direction extractor) isn't checked at all, and the plan would invalidate its own records by moving the code after checking it. Fix the order, point each check at the actual code, cut the two extras that can't be anchored, and it becomes a real feedback loop for about $200 and a week and a half.


# Fable's re-check of plan v3 (31 Oct 2026)

# Re-check of calibration_plan_v3.md (Fable, 31 Oct)

Verdict: ready with changes (all small). The v2 points are fixed correctly; the three departures are sound (branch, ImpossibleBench optional, judges first). v3's claim that only two conclusions rest on a judge holds: 134 is `stance_rule`, a parse.

## Remaining changes, ranked

1. **Check 6 compares different inputs, and "345 of 378" is wrong.** koan_pairs built 368 pairs from the 378 MBPP+ tasks (STATS.md); 345 is Gate 0's ambiguity-filtered subset (twins.json `kept`). Use the 368. Bigger: each pair's hidden set is a seeded sample of at most 50 of EvalPlus's plus inputs (cap 50, median 50), so a correct scorer will pass code EvalPlus fails; ≥98% label agreement can fail without a scorer bug. Make (a) a same-inputs test: EvalPlus's `unsafe_execute` (the third_party copy that `evalplus_match` was rewritten from) on the same 50 inputs, band ≥99%; report the full-plus-set gap as a number, not pass/fail. Run with `release_available=False`, neutral framing, greedy, or over-releases depress pass@1. Then (b)'s ±8 against the leaderboard is fine.

2. **Check 4 needs three conventions pinned.** The repo's generate_vec.py concatenates the stored prompt text and the response with no chat template; whether that stored prompt carries the trait system prompt must be read off the CSV at registration, and probe's `include_system` (default False) set to match. Its `hidden_states[20]` counts the embedding as 0, so the repo's "layer 20" is probe's block 19; register the convention or compare at the best-cosine layer. The repo filters by judge (pos ≥50, neg <50, coherence ≥50): use the same filter or none in both. With those matched, ≥0.8 is a fair band.

3. **Check 3's Ministral baseline is a gross check only.** GRP-Obliteration's 8.0 uses LlamaGuard-3-8B (gated, manual approval, not on OpenRouter), and that table's capability columns look off (MMLU 53.4 against the card's 76.1), so register "≤20% under the StrongREJECT judge" and treat the lab's own baseline as the datum. For Qwen, 100 JBB prompts give 9.1 an interval of about 4-17; and pre-register the StrongREJECT aggregation (mean score ×100 or a threshold), since it could otherwise be tuned to hit 9.1 and then read 69.2 as a pass.

4. **The cheap tier skips the trainer and the koan scorer**, the two most-edited files. Add check 7 itself (RMU relearning, <$5, ~20 min) as the trainer's item, and a stored-code, same-inputs scorer comparison (no GPU) for koan_env. Cut the model check to a fixed 300-item subset: full GSM8K plus 1,000 MMLU-Pro with CoT already exceeds the hour.

5. **Two bands still can't fail.** Check 1's "or within the published format's known gap" is an escape hatch; register the format and the expected gap, with its number, before running. Check 5's "loss curves must match": TRL shuffles with its own sampler, so curves can't be step-matched; use the loss on an identical first batch (masking and template) and held-out NLL after training within 3% (optimizer and schedule). Also write the EM bands as numbers now (insecure 1-10%, advice 15-35%).

6. **Check 7.** The raw-text trainer path is missing from the code-changes list. The public cais/Zephyr_RMU may not be Łucki's own RMU run (the WMDP paper reports a different RMU figure), so the baseline is a band (27-33), not "29.9 reproduces".

7. **Key cards on file hashes, not the commit**; a squash-merge changes the commit id and would void every card by the plan's own rule. This supports the branch departure.

8. **Say what check 6 leaves out.** It doesn't exercise `hardcodes2` or release parsing; both are mechanical with unit tests (test_scorer_adversarial.py), so list them with the no-anchor instruments.

Cost ($150-250) and time (6-10 days) hold; checks 1-4 in days 3-4 assumes the move takes two days, which is realistic.
