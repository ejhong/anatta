# K1 cold check (Fable 5.1, 1 Nov 2026) — commit cc9fb52 on calib/prh-k1

Read-only. Sources read: the brief; ledger entries 14:05 and 14:22; all of calibration/prh_k1 (code, config, README, tests); the paper's PDF (Table 15, Table 16 and its text, the "WIT, PRH protocol" paragraph); the saved prh-proof copies, spot-checked line by line against GitHub at a2b9949 (the repository's only commit, "Initial release", 30 Oct 2026; every line I asked about matched); Huh et al.'s `remove_outliers`, `prepare_features`, `compute_nearest_neighbors`, `mutual_knn`, and `models.py`/`extract_features.py` for the tokenizer question; the hub's listings for every repository; modal 1.6.2 and transformers 5.21 / timm 1.0.30 sources on disk. Tests: 93 passed, 3 skipped, as claimed. Scratch checks on tiny random tensors (k1_check_notes/scratch_check.py): Huh's threshold, the null's fast path, Huh-style k-NN, grade boundaries — all as claimed.

## Verdict

**Prepare may proceed.** Prepare's code is sound (see §4). Make the three small fixes below before the smoke and the full run; none touches prepare's facts, and only one of them changes a plan.

## Fixes before the smoke / full run

1. **An ungraded arm can block the graded result.** `app.py:397-404` catches only `torch.cuda.OutOfMemoryError` around `F.text_layers`. Any other failure of `open_llama_3b` (the last model: a `pickle` refusal in `check_bin`, `nonfinite` from a fully-masked left-pad row on a GPU kernel, `shape`, `device_map`) raises out of `full_body` before the scoring step, so the five Qwen scores are never written; the fix then changes the plan and the folder, and every model is re-extracted (~$1–1.5 of the $6). Fix: for models with `graded: false`, catch `Exception`, record `type(e).__name__` in `errors[key]` (text-free already), `F.release()`, `commit()`, `continue`; keep OOM-only for graded models. (Alternative with the same effect: extract the ungraded arm after the graded five are scored.) Add the test beside `test_a_model_out_of_gpu_memory_is_recorded_...`. Changes `container_sha256`, so do it before any run.

2. **`image_facts` can fail prepare after the downloads.** `features.py:105-107` records `raw.mode` and `str(raw.format)` as dict keys; PIL modes such as `I;16`, `I;16B` (16-bit PNGs, common on Wikipedia) carry `;`, which `k1.py:140`'s SAFE alphabet refuses, so `K.written` at `app.py:473` raises `free_text` at `prepare: at write` after every repository is on the volume. A rerun after a fix costs cents (the cache is committed per repository), but the stage's result is lost. Fix: `mode[raw.mode.replace(";", "_")]` (and the same for `format`), or a whitelist. Not in any plan.

3. **The batch-invariance check will probably stop the smoke for a non-reason.** `app.py:317` sets `passed` on the raw `max_abs ≤ 1e-4` (`config.yaml:77`). Qwen2.5's residual stream carries entries of order 1e2–1e4 ("massive activations"); one fp32 ulp at 1e3 is 1.2e-4, so batch-64 and batch-1 kernels differ by more than the tolerance on noise alone. Either register the reading rule in §Rule before the smoke (and leave `batch_tol_abs` alone), or make `passed` follow the rule (changes both plans; fine before any run). The quantities the rule needs are already recorded.

## Fidelity (item 1): matches the paper's code

Line by line against a2b9949:
- `mutual_knn`, `unbiased_cka`, `_prh_prep`, `prh_score`, `subsample_interval`: verbatim. `prh_subsample`: verbatim plus `best`, which is the same `prh_score` result (measure.py:214, 245).
- Images: `timm.create_model(name, pretrained=True, num_classes=0)` with no `img_size`; the transform from `resolve_data_config(net.pretrained_cfg, model=net)`; timm 1.0.30's cfg is input_size (3, 518, 518), crop_pct 1.0, bicubic, ImageNet mean/std (checked in the installed source). Hooks on each of the 12 `net.blocks`, `out[:, 0]` (num_prefix_tokens = 1), `.float().cpu()`, no final norm anywhere (same as the paper: the residual stream after each block), fp32, batches of 64, `.convert("RGB")`. The only departure is `pretrained_cfg_overlay={"file": ...}`: timm's 'file' source runs the same filter and strict load as the hub source.
- Texts: `AutoModel` (no head); the paper's dtype and device_map regexes applied to the paper's own keys (float32 for 0.5B/1.5B, bfloat16 for 3B/7B/14B, device_map auto for 14B); `tok.pad_token = tok.pad_token or tok.eos_token`; `padding=True, truncation=True, max_length=512`, batches of 64 in dataset order; `output_hidden_states=True`; `_mean` verbatim over every hidden state (embedding output included, N+1). Padding side is the tokenizer's default in both codes (Qwen2's class in 5.21 sets none, so "right"; Llama's class sets "left"). Qwen2.5-0.5B's tokenizer_config sets pad `<|endoftext|>` = eos.
- Rows: `load_dataset("minhuh/prh", revision="wit_1024", split="train")` → the card's `data/train-*`, two parquet files in name order, 1,024 rows; caption `text[0]` raw (evaluate.py `wit_embed`); `read_wit` reads the same files in the same order with pyarrow; `decode_image` is `datasets`' decode (open, load, exif_transpose on an Orientation tag).
- Scoring: per-layer clip at the 0.95 quantile with the stride above 10M values, L2 rows, cosine k-NN with the diagonal at −inf, k = 10, mean shared fraction, max over all (vision, text) layer pairs with ties to the highest indices; `prh_score` on the GPU, `prh_subsample` on the CPU, exactly where the paper's `wit` runs them.
- Nothing found that moves a score by 0.005 for a reason in our code.

## Registration (item 2)

`config.yaml:86-89` and `card.py:86-104` carry the 14:05 entry exactly: targets .102/.108/.125/.118/.130 (Table 16, DINOv2-B column, read from the PDF; Table 15 agrees for 0.5B–7B), CKA .318/.324/.363/.353/.333 recorded, match ≤ 0.005, near ≤ 0.010, else fail; pass iff all five match, near iff no fail, fail otherwise (a fail decides with a model missing; otherwise a missing model is "incomplete"). Boundaries checked in the tests and by hand: 0.005 match, 0.0051 near, 0.010 near, 0.0101 fail, both sides. The card grades by the plan's own copy of the block.

## Guards (item 3)

- $6 ceiling: laptop `Spend.check` before every call (spend so far + start-up + first unit), timeouts cut to what the ceiling buys; container `Ceiling.before` before every unit at the rate of the GPU it got, `budget` after each commit. prepare, smoke and full share one total (`main` refuses to run without prepare's spend.json; prepare counts the runs' folders). One gap: a relaunch into a fresh `--results` (which `other_plan`'s message suggests) would not count the old folder — never change `--results`; move a folder only after noting its spend in the ledger.
- Another plan in plan.json: refused on the laptop (`other_plan`) and in the container (`K.claim`).
- Results: `results_path` refuses any folder under a `.git` (a worktree's `.git` file counts).
- Modal: app and volume `anatta-calib-prh-k1`; no "matlab" anywhere (test asserts it). `with_options(gpu/env/timeout)` keeps the decorator's volume: modal 1.6.2's `merge_options` overwrites only truthy values, so the empty `volumes={}` default is a no-op (checked in `_function_variants.py:155-157`).
- Tokens: the `huggingface` secret is attached to `prepare_stage` only; results pass `assert_text_free`; failures leave as a `StageError` with a type name and a code only.
- Loads: `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` set before any load in the runs; `verify_files` checks size and sha256 of every file a stage reads; `weights_only=True`; safetensors only for Qwen; the OpenLLaMA .bin refused unless a flat dict of tensors.

## prepare (item 4)

Resolves each ref through `model_info`/`dataset_info(revision=ref, files_metadata=True)` to `info.sha`, downloads at that sha, checks every file against the hub's size and LFS sha256 (or git blob sha), records bytes and sha256, refuses an unapproved size before downloading, counts rows from parquet metadata, runs no dataset script (only `data/train-*.parquet` is allowed). `fill_config` refuses different facts over a filled block. Only the eight approved repositories are fetched (`fetch.repos`). Expected resolutions, from today's hub listings (all unchanged since before the paper): minhuh/prh wit_1024 → 0220f2b (12 May 2024; 285,606,572 + 258,865,704 bytes); timm DINOv2 → 4685c99 (21 Jan 2025; model.safetensors 346,334,872); Qwen2.5-0.5B 060db64 (25 Sep 2024), 1.5B 8faed76 (8 Oct 2024), 3B 3aab1f1 (20 Sep 2024), 7B d149729 (25 Sep 2024), 14B 97e1e76 (20 Sep 2024); open_llama_3b 1410670 (16 Jun 2023; pytorch_model.bin 6,853,038,093). Anything else means a repository moved after today.

## The smoke (item 5)

### Rule: reading `text_batch_invariance` (register before the smoke)

Recorded per smoke (`smoke.json` → `result.checks.text_batch_invariance`): `max_abs` and `by_layer` (raw fp32 features, batch 64 vs batch 1, max abs per hidden state), `max_rel` (that over the layer's largest value), `prepped_max_abs` (the same after `_prh_prep`), `tol`, `dtype`; beside it `text_identical` (two batch-64 passes bitwise equal) and `text_finite`.

1. `by_layer[0] ≤ 1e-5`. Hidden state 0 is the embedding lookup's masked mean: no attention, no MLP, nothing but which tokens are in the mean. Kernel noise there is ~1e-7 (embeddings ~0.02, sums of ≤512 terms); a pad token leaking into the mean, a wrong mask, or a truncation applied in one path only moves a short caption's mean by ~embedding/(L+1) ≈ 1e-3. This is the direct test of tokenization, masking and batching.
2. `prepped_max_abs ≤ 1e-4`. This is the only representation the graded score sees (clipped at the 0.95 quantile, rows of unit norm, entries ≈ 1/√896 ≈ 0.03). Noise after 24 layers is ≤ 1e-5 relative, i.e. ≤ 3e-7 here; any batching or padding fault moves a row by ≥ 1/(L+1) of itself, i.e. ≥ 1e-3 here. Thirty-fold margins on both sides.
3. `text_identical` and `text_finite` pass, `dtype` is float32.
4. Raw `max_abs` (and `max_rel`) are recorded, not read: they are expected to breach 1e-4 wherever a layer holds values ≥ ~1e3 (massive activations), and `max_rel` is dominated by those few coordinates, so it can neither clear nor convict a fault in the ordinary ones.

Reading: if 1–3 hold and only the raw check failed, the smoke is read as passed by this rule (the code's `passed` will say otherwise; the ledger records the override, written before the smoke ran, and `batch_tol_abs` is not edited). If 1 or 2 fails, it is a batching or padding fault and the full run waits.

### 14B fallback

Works as described for an OOM in the forward pass: caught at `app.py:399`, recorded, the run goes on, `stored` treats an error as unfinished, `main --gpu A100-80GB` reruns the same plan and extracts only the missing model, then scores all six (`test_a_model_out_of_gpu_memory_...` covers the path). Two caveats. (a) If accelerate answers a too-small GPU by offloading at load time instead of raising OOM, `features.py:221` raises `CheckError device_map`, which is not caught: the stage fails without a card, the features so far stay, and the same relaunch resumes — it works, by the failure path. (b) At batch 64 with any caption padded to 512, the 14B's bf16 weights (29.5 GB) plus 49 kept hidden states (16.4 GB) plus MLP and attention buffers exceed the L40S's ~44.5 GB usable; the smoke's `memory.qwen14b.fits` reads this from the real padded lengths. If it says MAY NOT FIT, launch the full run with `--gpu A100-80GB` from the start (same plan, $2.50/h vs $1.95/h on a ~$1.5 run): it saves a wasted 14B load and a second container.

### OpenLLaMA tokenizer

Huh et al.'s code (`models.py`) loads `AutoTokenizer.from_pretrained(path)` with no `use_fast=False`, sets pad = eos and `padding_side = "left"`, tokenizes with `padding="longest"` and no truncation, runs `AutoModelForCausalLM` in bfloat16 on capable GPUs, and mean-pools every hidden state under the mask. So the auto-converted tokenizer is the anchor's own lineage; OpenLLaMA's README advice (`use_fast=False`) was never what the anchor used, and in transformers 5.21 it is not available anyway: `models/llama/tokenization_llama.py` holds one tokenizers-backed `LlamaTokenizer` (`padding_side = "left"`, `legacy=False`), which converts from `tokenizer.model` (sentencepiece and protobuf are pinned). open_llama_3b's tokenizer_config sets neither `padding_side` nor `legacy`. The arm as built (converted tokenizer, left padding, eos pad, bf16, masked mean over all hidden states) is the closest available to the anchor's lineage; its departures are `truncation=True, max_length=512` (the paper's; moot if `truncated` is 0, which the smoke records) and `legacy=False` (identical to 2024's `legacy=True` on plain captions, which never follow a special token).

The smoke's check is the right one and is enough for the tokenizer: every one of the 1,024 captions against `[bos] + sp.encode(text)` from the sentencepiece model that defines OpenLLaMA's tokenizer. Register: the arm is read against the anchor only if `differ == 0`; a nonzero count is recorded and the arm is read with that caveat (a second record with ids taken straight from sentencepiece would settle it). What the check is not: evidence that Koepke et al.'s pipeline (unread; dtype, code and version unknown) matches; the ledger should call the anchor lineage-comparable, not protocol-identical.

## Diagnostics (item 6)

- Permutation null: `null_scores` renumbers each layer's neighbour lists once; `ib[:, p]` applies one permutation to every text layer at once; the identity re-pairing reproduces `prh_score`'s matrix (test, and my scratch run); fast equals naive to 1e-7 absent boundary ties, which `boundary_ties` counts and the card carries. τ is the 95th percentile of 200 layer-pair maxima (seed 0); calibrated score as registered.
- Huh's clipping: `huh_threshold` equals Huh's inexact one-liner (`torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()` over [N, L, D]) on a random tensor; one threshold per model; `--precise` is `store_true` in their CLI, so the inexact mode is their default and the label is right. Written afresh (concatenation plus chunked row quantiles); not a copy.
- Subsample intervals: 200 subsets of 800 rows, seed 0, scored at the best pair on the CPU, as the paper's `wit` runs them; differences are model − 7B with paired subsets (the paper prints first − pair; sign only).

## What the card should record for a near or a fail (item 7)

- The prepared block's commits (above) against the paper's unpinned main; all predate the paper, so the files are the paper's.
- Versions and hardware: torch 2.14.1, transformers 5.21.0, timm 1.0.30, Python 3.12, the GPU; the paper states only torch ≥ 2.6, transformers ≥ 4.56, timm ≥ 1.0, and that its 14B and larger ran "split across two GPUs in bfloat16" (Table 16's text); the GPU behind Table 16 is unstated. Kernels and GPU type differ in the last bits; in bf16 (3B, 7B, 14B) that flips a few neighbours — of order 0.001 on mutual 10-NN, not 0.005.
- `precision_flags`, including `cudnn_allow_tf32` (torch's default True; it governs DINOv2's patch-embedding convolution on Ampere/Ada/Hopper; the paper's code leaves the same default).
- Per model: padding side, pad = eos, `max_padded`, `truncated`, attention implementation, dtype, device map, the best layer pair (compare with the paper's `layers i/N, j/M` print if ever published), `boundary_ties`, and the subsample interval (which says how far ±0.005 sits inside sampling noise on 1,024 pairs — the tolerance is about the same pairs, where only kernel noise applies).
- A near on a bf16 model with the fp32 models matching points at kernels, not protocol; an fp32 rerun of the 3B (weights 12 GB) is a cheap ungraded follow-up if wanted.

## Non-blocking notes

- `HF_XET_CHUNK_CACHE_SIZE_BYTES` is a documented name but "no longer has any effect" (hf_xet keeps no chunk cache); harmless; drop the worry from the README. huggingface_hub 2.3 keeps a cache-wide blob store with symlinks into the snapshot; `verify_files` follows them.
- `vision_net` (the timm local-file path) has no local test and timm is not in the lab venv (it is in the matlab venv, which is presumably where the README's "174 tensors" check ran); the smoke's `vision_input_size` and `vision_layers` are its first real exercise.
- Left padding plus SDPA on a GPU: a fully-masked pad row that came back NaN would make `_mean`'s row NaN (0 × NaN) and trip `nonfinite`; the CPU test passed; with fix 1 this becomes a recorded error on the ungraded arm.
- `assert_text_free` elsewhere: pip-freeze strings, "NVIDIA A100-SXM4-80GB", "2.14.1+cu128", "ieee"/"tf32" all pass the alphabet.
- The full run's timeout is cut to what the ceiling leaves (~88 min after ~$0.9 of prepare and smoke); the volume's read rate decides the loads (≈126 GB read: hash plus load); the smoke's estimate reports it.

## Could not verify

- Anything requiring Modal: the `huggingface` secret exists, the volume's symlink support, actual GPU behaviour (activation magnitudes, the 14B's fit, left-pad NaNs on GPU kernels).
- timm's local-file load on the real weights (the README's claim); the smoke covers it.
- `datasets`' decode in the paper's (unstated) version against `decode_image`; the exif_transpose logic matches datasets ≥ 2.5 as I know it.
- Koepke et al.'s pipeline behind 0.135 / 0.058 (not read).
- GitHub was checked by targeted line questions through WebFetch (all matched), not a byte diff; the repository has a single commit, so the saved copies are of it.

Claude (Fable 5.1, an AI model made by Anthropic) wrote this check.
