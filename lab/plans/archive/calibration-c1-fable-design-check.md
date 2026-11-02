# Calibration check 1: Fable's cold design check

A fresh Fable agent (model fable) checked the C1 registration (NOTES, 01:57 UTC on 31 Oct) cold, with no memory of the session. It reported at about 02:07 UTC on 31 Oct, before any C1 code or run existed. Its report is below, unedited. Its scratch files are in the session scratchpad's fable_c1_design/ folder, including mmlu_pro_allocation.txt. Opus 5.5 filed this and wrote the amendment that follows from it (NOTES, 02:08 UTC on 31 Oct).

---

**Verdict: ready with changes.** The design is sound: running the battery's own path (models.load + probe.generate + battery parsers) is the right call, since `--model anatta/...` never touches that code. The anchors check out against the fetched copies. The weaknesses are in what the code records, two silent errors a ±5 band cannot see, and under-specified items in the draw, cheap tier and failure path.

## Changes, ranked

1. **Store full replies and per-row `new_tokens` / `hit_cap`.** The battery saves `text[:800]` (modal_battery.py `evaluate`); 58 of 250 stored GSM8K base rows sit at that cap. The registered length, truncation and TIGER-Lab-fallback reports cannot be computed from what the code saves today.

2. **Two zero-cost assertions, so silent errors fail hard.** (a) Token ids of the rendered prompt equal `apply_chat_template(..., tokenize=True, add_generation_prompt=True)`, and Ministral's start with exactly one BOS: Qwen has no BOS, so a Qwen pass says nothing about BOS handling on the lab's model. (b) Generate the first batch twice and require ≥95% identical replies: sampling left on at T=0.7 costs 1-3 points and would pass both bands.

3. **Batch-1 vs batch-50 on ~60 items per model** (identical-reply share, accuracy gap). Padding and batching are the one part of the path no anchor speaks to, and the Mistral3 (ImageTextToText) class is exercised only on Ministral, which has no anchor. ~$1.

4. **Copy Inspect's `parse_answers` verbatim, with golden strings.** The registration describes only the strict last-line match; the real function (inspect_ai 0.3.285 solver/_multiple_choice.py) also falls back to the last `ANSWER:` anywhere, so "**ANSWER: C**" and "ANSWER: C)" parse. An implementer following the card would build a stricter parser than the reference. Test "ANSWER: (C)", "**ANSWER: C**", "ANSWER: C.", "Answer: c", "ANSWER: Option C" (wrong).

5. **Pre-register C1b's format explanation.** The 56.2 anchor's parser is unknown; strict parse plus 2,048-token truncation on math/physics chains can cost 3-6 points with nothing broken. Rule: a strict miss is "explained as format" if the TIGER-Lab-fallback score is in band and the no-ANSWER-line share covers the gap. With that rule 70% is fair; without it ~60%. C1a's 91 and 85% are fair (91.13 is 0-shot CoT; repetition penalty 1.05 and the 512 cap are the only known downward pulls).

6. **Define the cheap tier's 150.** "First 150" of a draw in dataset order gives 66 business + 84 law on MMLU-Pro: unrepresentative and never a long generation. Use a stratified 150 (largest remainder again, or every k-th of a seed-0 shuffle). Make its reference a run of that subset in its own batches at C1 time: different batch composition means different padding and more flips than the ±5 can absorb on 150 reasoning items.

7. **Give the Ministral rerun a tolerance: within 2 points per set** (5 of 250 items), plus the identical-text share on the stored 800-char prefixes. Same code, items, batching and GPU leave only kernel nondeterminism; ±5 would be uninformative.

8. **Pin the datasets and select by id.** `hf_hub_download(revision=)` 527feea (MMLU-Pro) and cc7b047 (GSM8K), inspect_evals' pins; the battery lists files at HEAD. The item file stores question_id, category, full prompt, gold and the parquet sha256; state that the seed-0 sample indexes parquet row order within a category. The allocation is well defined with no tie at the cut (engineering .535 in, law .506 out): math 112, physics 108, chemistry 94, law 91, engineering 81, other 77, economics 70, health 68, psychology 66, business 66, biology 60, philosophy 41, CS 34, history 32 (saved in scratchpad/fable_c1_design/mmlu_pro_allocation.txt).

9. **Ministral C1b at batch ≤25.** KV is 34 layers × 8 kv heads × 128 × 2 × 2 B = 139 KB/token; 50 × ~2,800 tokens (523 system + question + 2,048) ≈ 19.5 GB, plus 17.8 GB weights ≈ 37 GB before prefill activations on 44 GiB. Qwen at 50 is ~22 GB, fine. Record batch size on the card: greedy output depends on batch composition.

10. **Failure path, step 0 without GPU:** diff rendered prompt strings and token ids against Inspect's for 5 items per model; re-score stored replies with Inspect's parser. Then hold format fixed: inspect_evals' gsm8k even at fewshot=0 uses a different template ("ANSWER:", "Reasoning:"), so first run the lab path with Inspect's template and parser, then Inspect's provider with `-M do_sample=False`, `--max-tokens 2048`, `--sample-id` on the question_ids, and for Ministral `auto_model_class=AutoModelForImageTextToText`, `tokenizer_call_args={'add_special_tokens': False}`. That step needs inspect_ai, inspect_evals and datasets on the image. Optional swap: repetition_penalty 1.0 on 150 GSM8K items.

11. **Card additions:** hooks out of scope (check 3 covers `intervene`; C1 runs base only); max_new_tokens, repetition_penalty read from `model.generation_config` at run time, do_sample, dtype, torch/transformers versions, GPU; item 0's rendered prompt and first token ids; and what happens when one of C1a/C1b passes and the other fails (only that format's parser goes provisional, not the load path).

12. **Image:** anatta-calib's modal_app.py image has no pyarrow; add pyarrow==21.0.0 as the recognition battery image does.

13. **Ministral contrast:** right as the primary (the lab's default vs the literature's short prompt), but it confounds 521 tokens of length with content. If the gap exceeds 3, a third arm with system content "" separates them: template line 37 inserts the default only when messages[0] isn't a system message.

## What ±5 catches and misses
Catches: a missing or garbled template (probably), truncation to ~128 tokens, fp16 garbage, a comma-blind GSM8K parser (about a tenth of golds are ≥1,000), a parser blind to "(C)" if ≥5% of replies use it; a wrong checkpoint (base model, 85.4) is borderline. Misses: sampling left on, double BOS (none on Qwen), a missing generation prompt, repetition penalty, pad side, any 1-2 point drift. Items 2-3 close the first three.

## Cost and scope
Steps: Qwen ~58k (GSM8K 27 batches × 512, MMLU-Pro 20 × 2,048, letter-only, battery) → 0.8-1.9 h, $2-5; Ministral two conditions with heavier prompts and halved C1b batch → $4-12. The $12-24 estimate holds; $40 stop is fine. Scope is right; the additions above total under $3.

## Factual checks
- Verified in check1_dl: Qwen TR Table 8 (56.3 / 91.6); Tülu 3 Table 43, "Qwen 2.5 7B Inst" chat-v1 = 91.13, chat-v1 defined as zero-shot CoT; Phi-4-Mini Table 7, "MMLU-Pro (0-Shot; CoT)" Qwen2.5-Ins 7B = 56.2, "GSM-8K (8-Shot; CoT)" 88.7.
- Registration's parse description is incomplete (item 4). Source: /Users/eugene/prj/align/anatta/.venv/lib/python3.12/site-packages/inspect_ai/solver/_multiple_choice.py, `parse_answers`.
- Registration's "each reply's length, and the share truncated" is not computable from the battery's stored rows (item 1). Source: lab/experiments/recognition/scripts/modal_battery.py, `"text": text[:800]`.
- Plan row 1 writes "Qwen2.5-7B" for the Instruct model (the fact sheet notes it).
- Confirmed: repetition_penalty 1.05 and top_k 20 sit in Qwen's generation_config and the penalty applies under greedy; prefill logits are sliced via `logits_to_keep` in transformers 5.21, so memory is KV-bound as computed.
