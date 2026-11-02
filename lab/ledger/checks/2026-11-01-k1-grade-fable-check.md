# K1 grade check (Fable 5.1, cold, read-only, 1 Nov 2026)

Checked: the card at /Users/eugene/prj/align/calib-results/prh_k1/card.json (plan ecdb56d332b6), full.json, the smoke's card, prepare.json, the console log, the worktree /Users/eugene/prj/align/anatta-calib-k1 at 237841e, the ledger entries of 14:05, 15:50 and 16:18 UTC, the paper's PDF (arXiv 2610.39302v1, fetched; Table 16 and Appendix B), prh-proof at a2b9949 (fetched: prh/features.py, prh/metrics.py, prh/data.py, evaluate.py; the repository's only release commit, 30 Oct 2026), the pinned config.json files from the hub, the hub's commit dates, and the test suite. Nothing was edited, launched or downloaded to the project; downloads went to the scratchpad only.

## Verdict: AMEND

The PASS is right: all five scores are within 0.0003 of their targets, the targets are Table 16's ViT-B column, the run is the committed code at 237841e, and the protocol matches the paper's code line by line. The record needs four corrections before it goes in the ledger (none changes a number):

1. `stored_dtype` on the card is the checkpoint's `config.json` dtype, not the dtype the features were stored in. The graded scores come from float32 features; no deviation.
2. The paper's `evaluate.py` disables the cuDNN SDPA backend before every score (`main()`, line 572: `torch.backends.cuda.enable_cudnn_sdp(False)`); the lab's run left it on (card `precision.sdpa_cudnn: true`, line 657). An unlisted departure. Its effect is bounded by the match itself (≤ 0.0003) but it belongs in the README's departures and the ledger's Limits.
3. The test suite at 237841e gives 101 passed, 1 failed, 4 skipped, not the README's "102 pass and 4 skip". The failure is a stale fixture, not a defect (below).
4. The 16:18 entry says OpenLLaMA is "not read against its anchor"; the card prints its numbers beside the anchor and labels the reading "with the tokenizer caveat" (card lines 318-335, 112 of the log). Same substance; the ledger should use the registered wording: recorded beside the anchor with the caveat, not read against it.

## 1. Arithmetic and verdict

Recomputed from full.json (`result.scores.<model>.mknn.score`) against the plan's registered targets, with card.py's rule (`grade()` lines 99-106: |score − target| rounded to 1e-9, ≤ 0.005 match, ≤ 0.010 near; `verdict()` lines 109-117):

| model | score | ×10240 | target | diff | grade |
|---|---|---|---|---|---|
| qwen0.5b | 0.1021484 | 1046 | .102 | +0.00015 | match |
| qwen1.5b | 0.1077148 | 1103 | .108 | −0.00029 | match |
| qwen3b | 0.1250977 | 1281 | .125 | +0.00010 | match |
| qwen7b | 0.1181641 | 1210 | .118 | +0.00016 | match |
| qwen14b | 0.1302734 | 1334 | .130 | +0.00027 | match |

Counts 5/0/0, outcome pass (card.json lines 303-312). Every score is an integer count of shared neighbours over 10,240, and every one lies inside the window that rounds to the paper's three decimals (e.g. .102 is counts 1040-1049; we have 1046). The reproduction is to the paper's own printed precision, on the bf16 models too.

## 2. Targets against the paper

paper.txt (pdftotext of the fetched PDF), Table 16 at lines 2362-2370, header "mutual 10-NN, DINOv2 S / B / L / g | CKA, B | COCO-5k": Qwen2.5-0.5B .092 **.102** .114 .115 .318; 1.5B .101 **.108** .123 .124 .324; 3B .110 **.125** .139 .141 .363; 7B .110 **.118** .137 .133 .353; 14B .119 **.130** .147 .148 .333. The registered targets are the B column; the S column (.092, .101, .110, .110, .119) is what the earlier misreading of 1.5B as .101 came from. The CKA values beside them match config.yaml line 95. Table 16's text (lines 2351-2353): "base models, the same per-layer protocol; the 14B and larger ones split across two GPUs in bfloat16". Appendix B, "PRH protocol" (lines 1677-1679): "WIT-1024 pairs, first caption, per-layer features (vision: CLS after each block; language: mask-weighted mean of each hidden state), elementwise clipping at the 0.95 quantile of absolute values, ℓ2 normalization, mutual 10-NN, maximum over layer pairs." Both match the 14:05 registration.

## 3. The run is the registered code

- `git status --porcelain --untracked-files=all` in the worktree: empty. HEAD 237841e on calib/prh-k1, committed 2026-11-01T08:13:16−08:00 (16:13 UTC). Working-tree sha256 of k1.py, measure.py, features.py, app.py, config.yaml, fetch.py equal `git show 237841e:` of each.
- The plan id is `sha256(json.dumps(plan, sort_keys=True))[:12]` (k1.py lines 256-258) over make_plan's dict (app.py lines 532-559: every config block but cost, the prepared block, `code` = sha256 of k1.py/measure.py/features.py (k1.py line 23, 239-241), the pins, and `container_sha256` = sha256 of app.py's container functions (app.py lines 437-450)). Recomputed from /Users/eugene/prj/align/calib-results/prh_k1/plan.json: **ecdb56d332b6**; from smoke/plan.json: **3b2d84f3a705**. Both as the cards say.
- plan.json's `code` hashes equal the sha256 of the three files at 237841e (k1 1d053c6a…, measure 17b487df…, features 94f1739d…). plan.json's `container_sha256` 2d06e1c7… equals the hash recomputed from app.py at 237841e with k1.function_source. full.json's `env.code` equals the plan's (card.py line 195 refuses otherwise), and its plan_id is ecdb56d332b6.
- The container checked its code and pins against the plan before loading anything (k1.py `check_environment`, lines 244-253; app.py `stage`, line 187).
- app.py's laptop side (make_plan, Spend) is not hashed; its identity with 237841e rests on the clean status plus mtimes: app.py 08:01:21, config.yaml 08:11:18, both before plan.json (08:14:48) and card.json (08:19:36).
- The prepared block's commits (config.yaml lines 135-243, plan.json `prepared`, card `pinned` lines 55-98) are the ones the run verified: full.json `result.verified` lists every repository's files checked by size and sha256 (k1.py `verify_files`) before each load: data 2 files, dinov2 1, qwen0.5b 7, 1.5b 7, 3b 9, 7b 11, 14b 15, open_llama 6. The smoke ran under the same prepared block (smoke/plan.json `prepared` equals the full plan's; the open_llama entry equal, which card.py's `tokenizer_reading` requires, lines 155-167).
- Hub commit dates (fetched today): minhuh/prh wit_1024 0220f2b 2024-05-12 (head of the branch); timm DINOv2 4685c99 2025-01-21; Qwen2.5-0.5B 060db64 2024-09-25; 1.5B 8faed76 2024-10-08; 3B 3aab1f1 2024-09-20; 7B d149729 2024-09-25; 14B 97e1e76 2024-09-20; open_llama_3b 1410670 2023-06-16. All are the heads of main/wit_1024 today and all predate the paper's code (30 Oct 2026), so the paper's unpinned loads read the same files.
- Spend: $0.3981 estimated for all of K1 (card lines 9-32), of $6.

## 4. Protocol fidelity (lab code at 237841e vs prh-proof a2b9949)

- First caption: features.py read_wit line 67 `caps.append(text[0])`; paper evaluate.py wit_embed line 123 `[t[0] for t in ds["text"]]` on data.py wit1024 (lines 425-429: `load_dataset("minhuh/prh", revision="wit_1024", split="train")`). Rows in file order, 512 + 512 (prepare.json). Images decoded as `datasets` decodes (features.py decode_image lines 81-93), `.convert("RGB")` as the paper (its embed_images line 176). prepare found 1,024 JPEG RGB, EXIF orientation on 0 images.
- timm at native resolution: features.py vision_net lines 128-138 `timm.create_model(name, pretrained=True, num_classes=0)` and `resolve_data_config(net.pretrained_cfg, model=net)`; paper _image_net lines 131-132 identical for layers=True (no img_size). Resolved config recorded: 3×518×518, bicubic, crop_pct 1.0, ImageNet mean/std (card lines 615-635). Departure: weights from the volume through `pretrained_cfg_overlay={"file": …}`.
- CLS after each of the 12 blocks: features.py embed_images lines 157-172, hook on each `net.blocks`, `pool(o) = o[:, 0]` (num_prefix_tokens 1), `.float().cpu()`, float32 weights and inputs; paper lines 167-182 identical. blocks 12, hidden states captured before the final norm in both.
- Dtypes: features.py text_dtype lines 30-31 and the paper's line 220 are the same regex rule (bf16 for a name ending in digits+b other than 0.5b/1.5b); device_map auto for 14B+ (lines 34-36 vs 221). Card: 0.5B and 1.5B float32, 3B/7B/14B bfloat16; 14B device_map_auto true. AutoModel (no LM head; the log's "lm_head.weight UNEXPECTED" confirms) as the paper's line 223.
- Masked mean of every hidden state: features.py _mean lines 241-243 is the paper's lines 245-247 verbatim; embed_texts lines 246-257 (`output_hidden_states=True`, every state including the embedding output, so layers + 1: 25/29/37/29/49) matches the paper's lines 227-240 and 251-260. Padding side is each tokenizer's own (Qwen right, OpenLLaMA left); `pad = pad or eos` as the paper's line 225.
- Truncation 512, batch 64: config.yaml protocol lines 59-63; paper line 228 (512 for layers=True) and batch_size 64 (lines 156, 251). No caption was truncated (longest batch 227 tokens).
- Clipping and normalization: measure.py _prh_prep lines 65-69 is the paper's metrics.py lines 82-86 verbatim (0.95 quantile of |x| per layer, stride only above 10M values; the largest layer here is 1024×5120 = 5.2M, so no striding), then F.normalize per row.
- Mutual 10-NN, self excluded: measure.py mutual_knn lines 36-47 = paper lines 12-23 verbatim (`fill_diagonal_(-inf)`, topk, mean shared fraction).
- Maximum over all layer pairs: measure.py prh_score lines 72-87 = paper lines 89-104 verbatim (max of (score, i, j) tuples, ties to the highest indices). prh_subsample (lines 90-102) = paper 107-117 plus the precomputed `best`; subsample_interval (105-115) = paper 120-130 verbatim.
- Storage between extraction and scoring: app.py store_features lines 200-207 `torch.save` of the float32 lists; stored_features lines 210-222 reloads them with sha256 verified. The paper's wit_embed line 129 `torch.save(feats, …)` and wit line 144 `torch.load` do the same. float32 both sides.
- Not in the paper's path: Huh's clipping, the null, k = 1, OpenLLaMA (written for K1, recorded only).

**The stored_dtype question.** `stored_dtype` is set at features.py line 226 from `AutoConfig…dtype or torch_dtype` and labelled at line 234: it is the checkpoint's declared weight dtype. Confirmed from the hub at the pinned commits (sha256 of each fetched config.json equals the prepared block's): Qwen2.5-0.5B and 14B config.json `"torch_dtype": "bfloat16"`, open_llama_3b `"float16"` — exactly what the card shows. The model was loaded with `dtype=float32` for the 0.5B and 1.5B (card `dtype`), the features were cast `.float()` in embed_texts, stored on the volume in float32, and scored from that file; the smoke's `prepped` check ran on those float32 features (1.15e-5). So: no, the graded scores are not from a bfloat16 copy; the paper's code reads the same bf16 checkpoint files and upcasts them the same way; the match within 0.0001 is consistent with that; it is not a deviation, only a misleading field name. The ledger should say what the field means. (Minor: the smoke's `float32` part, app.py line 316, checks the dtype label `facts["dtype"]`, not the tensor's dtype; embed_texts makes it moot.)

**New departure found.** evaluate.py main() line 572 runs `torch.backends.cuda.enable_cudnn_sdp(False)` before every paper score (a cluster workaround), so the paper's forward passes never used the cuDNN attention backend. The lab's run recorded `sdpa_cudnn: true` (card line 657) and does not record which SDPA backend torch picked. README.md's departures (lines 94-138) do not list it. The scores matched to ≤ 0.0003, so it changed nothing measurable here; it should still be on the record, and a future rerun could set the same flag.

## 5. The 15:50 reading rules

- Rule 1 (batch): smoke card lines 133-218: embed 9.31e-10 ≤ 1e-5; prepped 1.15e-5 ≤ 1e-4; identical (max_abs 0.0), finite, float32; passed, failed []. Raw 1.007e-3 (rel 1.63e-5) recorded, `within_raw_tol: false`, not read. measure.batch_invariance (lines 200-231) implements the rule as registered. Applied as written.
- Rule 2 (OpenLLaMA): smoke `sentencepiece.differ` 15 of 1,024 > `max_sentencepiece_differ` 0; card.py tokenizer_reading (lines 155-167) gives "read with the tokenizer caveat"; card `open_llama.graded: false`. Applied as written; see the wording point in the verdict.
- Rule 3 (GPU): smoke `memory.qwen14b` 36.22 GiB predicted of 44.42, fits (≤ 0.92 × 44.42 = 40.87), so the L40S; the full run was launched on the L40S with the same plan. Applied as written. Note: the 14B's measured peak was 38.71 GiB allocated / 39.57 reserved, 6.9% above the forecast (the 0.5B calibration was 3% low). It fit with about 4.8 GiB to spare; the forecast's error should be carried for any future model near the line. Also `device_map` is recorded as null for the 14B (hf_device_map was empty on one GPU), so the device_map check at features.py lines 231-233 was vacuous; the 38.7 GiB peak (27.5 GiB of bf16 weights plus activations) shows the model was on the GPU.

## 6. What could make the PASS misleading

- Selection over layer pairs: the statistic is the paper's own (max over 12 × (L+1) pairs), so reproducing it is the point, not a flaw. Its inflation is measured: the permutation null's 95th percentile τ is 0.0135-0.0139 for every model against a single-pair chance of 10/1023 = 0.0098 (the smoke's random features gave 0.0137 at the max). Scores sit 7-9 τ above it; calibrated scores 0.090-0.118. The best pair is vision block 10 (0-indexed, of 12) for every model, with the text state 2-9 states before the last. The margin over the second-best pair is small for the larger models: 0.0038 (0.5B), 0.0026 (1.5B), 0.0031 (3B), 0.0010 (7B), 0.0004 (14B); OpenLLaMA ties exactly between text states 22 and 23 (the higher wins, as in the paper's code). The max value is stable; the argmax layer is not, for 7B and 14B.
- Boundary ties: 1 vision row and 0-4 text rows per model (card `boundary_ties`); each can move one neighbour of one row, ≤ 1/10240 ≈ 0.0001 each, ≤ 0.0005 in all. `identity_matches` true for every model.
- The DINOv2 copy: timm's `vit_base_patch14_dinov2.lvd142m`, the exact name the paper's IMAGE_MODELS["dinov2"] loads, pinned at the branch head the paper also read. The registration's claim that it holds the same weights as facebook/dinov2-base is unverified here and not needed.
- The dataset revision: `wit_1024` at 0220f2b, the branch head since May 2024; the paper's `revision="wit_1024"` read the same commit. One image of 100.5 MP raised PIL's DecompressionBombWarning and decoded (below PIL's hard limit); `datasets` would do the same.
- The null calibration: recorded only, 200 shuffles, seed 0, one shuffle for all layers; it is not part of the grade.
- Kernels: cuDNN SDPA on (above), TF32 off for matmuls, cuDNN TF32 on (torch's defaults, the paper's too); the paper's GPU and versions are unstated; bf16 differences would show as a few neighbours (≈ 0.001), and did not.
- Huh's clipping vs the paper's: the scores differ by ≤ 0.0012 (0.5B +0.0009, 1.5B −0.0005, 3B +0.0012, 7B +0.0002, 14B −0.0001) at the same layer pairs; the two clippings agree on WIT-1024 at this scale.
- Tests at 237841e: 101 passed, 1 failed, 4 skipped. The failure, test_k1_app.py:451 `test_prepare_fetches_every_repository_and_fills_a_copy_of_config_yaml_once`, copies the committed config.yaml (line 462) and expects an empty prepared block; since 237841e the block is filled, so fill_config refuses at app.py:707, as designed. The fixture is stale; no run code is wrong. The ledger should not repeat the README's "102 pass".

## 7. What the recorded numbers tell P1

From full.json (recomputed; every interval reproduces from its subset std × sqrt(800/224) = 1.89):

- Score intervals (95%, the paper's subset method, pair-sampling noise): half-widths 0.0139 (0.5B), 0.0152 (1.5B), 0.0159 (3B), 0.0150 (7B), 0.0161 (14B); SE 0.0071-0.0082. The ±0.005 calibration tolerance is well inside pair-sampling noise, which is right: K1 compares the same 1,024 pairs, where only kernel noise applies.
- Paired differences vs the 7B (same seed-0 subsets): 0.5B −0.0160 [−0.0247, −0.0073]; 1.5B −0.0104 [−0.0179, −0.0030]; 3B +0.0069 [−0.0009, +0.0148]; 14B +0.0121 [+0.0031, +0.0212]. Half-widths 0.0075-0.0091, SE 0.0038-0.0046.
- **The smallest paired difference WIT-1024 resolves at 95% is about 0.008-0.009 mutual 10-NN.** The 1.5B-7B gap of 0.010 is resolved; the 3B-7B gap of 0.007 is not. An unpaired comparison of two separately measured scores needs about 0.02.
- The null floor is τ ≈ 0.014: a score below it is no agreement at all after the layer maximum.
- For P1: an effect must be ≳ 0.009 on the same pairs with the same seed-0 subsets to be read at 95%; an effect of 0.005 would need about three times the pairs (half-width ∝ 1/√n; COCO-5k is the paper's second benchmark) or a within-model paired design whose own interval P1 computes from the same subsets (a small intervention on one model shares most neighbours across conditions, so its paired SE should be below the 0.004 seen across models; measure it, don't assume it). Fix the layer pair or report both conditions' argmax: for the 7B and 14B the best text layer is within 0.001 of its runner-up and can jump between conditions while the max barely moves.

## Sentences for the ledger's Limits

- The card's `stored_dtype` is the checkpoint's config.json dtype (bfloat16 for every Qwen2.5, float16 for OpenLLaMA), not the dtype of the stored features; the 0.5B and 1.5B ran, were stored and were scored in float32, as the paper's code does.
- The paper's evaluate.py disables the cuDNN SDPA backend before scoring (main(), `enable_cudnn_sdp(False)`); this run left it enabled and did not record which attention backend torch chose. The scores matched within 0.0003, so it changed nothing here; a rerun should set the flag, and the README's departures should list it.
- The score is the paper's maximum over 12 × (L+1) layer pairs; the permutation null puts that selection's floor at τ ≈ 0.0135-0.0139 against a single-pair chance of 0.0098, and the calibrated scores are 0.090-0.118. The best pair's margin over the second-best is 0.0004 (14B) to 0.0038 (0.5B), so the argmax layer, unlike the maximum, is not stable for the larger models.
- Boundary ties at the 10th neighbour touch at most 5 rows per model (≤ 0.0005); the identity re-pairing reproduces every graded score.
- The 14B's measured peak (38.7 GiB) was 7% above the smoke's forecast (36.2); it fit the L40S with 4.8 GiB to spare, and its device_map record is empty on one GPU, so the GPU-placement check was vacuous there.
- WIT-1024's pair-sampling noise: 95% half-widths of 0.014-0.016 per score and 0.0075-0.0091 per paired difference from the 7B; the smallest paired difference it resolves is about 0.009 (the 3B-7B gap of 0.007 is not resolved, the 1.5B-7B gap of 0.010 is).
- The reproduction holds on torch 2.14.1, transformers 5.21.0, timm 1.0.30, one L40S, with TF32 off for matmuls and on for cuDNN convolutions (torch's and the paper's defaults); the paper's GPU and versions are unstated, and its 14B ran split across two GPUs.
- OpenLLaMA-3B is recorded beside Koepke et al.'s anchor with the tokenizer caveat (sentencepiece disagrees on 15 of 1,024 captions) and is not read against it: k = 10 0.134 (anchor 0.135), k = 1 0.070 (anchor 0.058), under the paper's clipping.
- At 237841e the test suite gives 101 passed, 1 failed, 4 skipped; the failure is a prepare test whose fixture copies the now-filled config.yaml and expects it empty (fill_config refuses, as designed), not run code.
- Spend: about $0.40 of the $6 ceiling (prepare $0.10, smoke $0.04, full $0.26; estimates).

Claude (Fable 5.1, an AI model made by Anthropic) wrote this check.
