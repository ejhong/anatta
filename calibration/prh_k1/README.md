# Calibration K1: the Platonic measurement with the lab's own code

**Claim.** The lab's own code reproduces Balestriero & LeCun 2026 (arXiv 2610.39302v1), Table 16: mutual 10-NN
alignment on WIT-1024 between the Qwen2.5 base models (0.5B to 14B) and DINOv2 ViT-B/14, each within 0.005 of its
target. Experiment P1 will ask whether a model's "self" acts as a nuisance for cross-model agreement. None of its
numbers can be read until this measurement reproduces a published one.

The registration is the ledger's "K1 registered" (Opus 5.5, 14:05 UTC on 1 Nov 2026, `lab/ledger/NOTES.md`). Eugene
approved the DINOv2 and OpenLLaMA downloads at 14:22 UTC the same day. `config.yaml` holds every setting, with the
targets and tolerances exactly as registered. Each run copies them into its plan, and the card grades by the copy in
the run's own plan.

## The sources

| What | Where, pinned |
| --- | --- |
| The targets | Balestriero & LeCun 2026, arXiv 2610.39302v1, Table 16 (mutual 10-NN with DINOv2 ViT-B/14; unbiased CKA beside it) |
| The paper's code (MIT) | github.com/galilai-group/prh-proof at `a2b9949`: `prh/metrics.py` mirrored in `measure.py`, `prh/features.py` in `features.py`; `evaluate.py` (`wit_embed`, `wit`) and `prh/data.py` (`wit1024`) followed. Licence in `third_party/prh_proof/LICENSE` |
| Huh et al.'s code | github.com/minyoungg/platonic-rep has no licence, so it was read and never copied. Their clipping is written afresh in `measure.huh_prep` |
| The pairs | `minhuh/prh` at branch `wit_1024` (it resolved to `0220f2b`, last changed 12 May 2024, when this was written), the train split's two parquet files, about 544 MB. prepare pins the commit |
| The vision model | `timm/vit_base_patch14_dinov2.lvd142m`, `model.safetensors` (346,334,872 bytes, approved) |
| The language models | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B,14B}` (base models, safetensors). prepare pins each commit |
| The second anchor (ungraded) | `openlm-research/open_llama_3b`, `pytorch_model.bin` (6,853,038,093 bytes, approved) and its tokenizer files. Koepke et al. 2026 report 0.135 at k = 10 and 0.058 at k = 1 with DINOv2-B |
| The null | in the manner of Gröger et al. (ICML 2026) |

## The numbers

| Model (base) | Target, mutual 10-NN | Unbiased CKA, recorded |
| --- | --- | --- |
| Qwen2.5-0.5B | 0.102 | 0.318 |
| Qwen2.5-1.5B | 0.108 | 0.324 |
| Qwen2.5-3B | 0.125 | 0.363 |
| Qwen2.5-7B | 0.118 | 0.353 |
| Qwen2.5-14B | 0.130 | 0.333 |

The rule for each model:

- within 0.005 of its target: a match;
- within 0.010: near;
- beyond 0.010: a fail.

K1 passes if all five match. It is near if none fails but some are only near, and it fails otherwise. A fail decides
even with a model missing. Otherwise a missing model leaves K1 incomplete.

The difference is rounded to 1e-9 before the comparison, so a score exactly 0.005 off (in decimal) is a match whatever
the float subtraction gives. A mutual 10-NN score on 1,024 rows is a multiple of 1/10,240, so above 0.102 the last
match is 1,095/10,240 = 0.10693 and the first fail is 1,147/10,240 = 0.11201.

Recorded beside the grades, ungraded:

- unbiased CKA, against the paper's values;
- the score under Huh et al.'s clipping;
- the subsample intervals;
- the permutation null and the calibrated score;
- OpenLLaMA-3B against Koepke et al.'s anchor, at k = 10 and k = 1, under both clippings. The anchor is
  lineage-comparable, not protocol-identical: Koepke et al.'s pipeline is unread. The arm is read against the anchor
  only if the smoke, on the same tokenizer files, found no caption that sentencepiece tokenizes otherwise
  (`max_sentencepiece_differ: 0` in the registered block). Otherwise the card records the count and reads the arm
  "with the tokenizer caveat". The converted tokenizer is kept as built.

## The protocol, as the paper's code runs it

- **Pairs.** The 1,024 rows in dataset order: the two parquet files in name order, as `datasets` reads the card's
  `data/train-*`. The caption is `text[0]`. The image is the row's, decoded as `datasets` decodes it and converted to
  RGB.
- **Images.**
  - timm `vit_base_patch14_dinov2.lvd142m`, created with `pretrained=True, num_classes=0` at its native
    configuration.
  - The transform comes from `resolve_data_config(net.pretrained_cfg, model=net)`: 518 px, bicubic, crop 1.0 centred,
    ImageNet mean and std. This was checked offline with timm 1.0.30, and the run records it.
  - float32, batches of 64.
  - A forward hook on each of the 12 blocks takes the CLS token (`out[:, 0]`) as float32.
- **Texts.**
  - `AutoModel`, the base model with no LM head.
  - float32 for 0.5B and 1.5B, bfloat16 for 3B, 7B and 14B, by the paper's rule on its model names. K1 uses the
    paper's own keys (`qwen0.5b` to `qwen14b`), so the rule applies unchanged. It gives `open_llama_3b` bfloat16 too.
  - `device_map="auto"` for 14B.
  - The tokenizer keeps its own padding side (Qwen pads right, OpenLLaMA left), with eos as pad where none is set.
  - `padding=True, truncation=True, max_length=512`, batches of 64, `output_hidden_states=True`.
  - Every hidden state (the embedding output included, so layers + 1) is mean-pooled under the attention mask in
    float32.
  - All under `torch.no_grad()`, as the paper's decorators do it.
- **Scoring.**
  - `_prh_prep` on each layer: clip at the 0.95 quantile of |x| (strided so that torch.quantile stays under its
    limit), then L2-normalise the rows.
  - Mutual k-NN with k = 10: cosine similarity, self excluded, top k, then the mean shared fraction.
  - The score is the maximum over every (vision layer, text layer) pair, with its argmax.
  - Unbiased CKA in float64, with the same maximum.
  - `prh_score` runs on the GPU, as the paper's does.
- **Precision.** torch's defaults: matmul precision "highest", TF32 off for matmuls. The run records them, with the
  CUDA and cuDNN versions, the SDPA kernels, the attention implementation, the GPU's name and every installed
  version.

## Where K1 departs from the paper's code, and why

`mutual_knn`, `_prh_prep`, `prh_score`, `unbiased_cka` and `subsample_interval` are the paper's text verbatim; a diff
against `a2b9949` shows no change. The departures:

1. **`prh_subsample` takes an optional `best`.** It is the pair `prh_score` already found, so the search doesn't run
   twice. The pair and the number are the same.
2. **The weights come from K1's volume, not from the hub at run time.** prepare downloads every file at a pinned
   commit and records its sha256. Each run checks the files against those sha256s and runs offline.
   - timm loads DINOv2 through `pretrained_cfg_overlay={"file": ...}`: timm's local-file source, with the same filter
     and strict load. Checked offline, the net holds the file's 174 tensors exactly.
   - transformers loads each language model from its snapshot folder.
   - The paper loaded `main` from the hub on the day it ran, unpinned. If a repository has changed since, K1 reads a
     different file, and the prepared block names the commit it read.
3. **WIT is read from its parquet files with pyarrow, without `datasets`.** No dataset script runs on untrusted data.
   - The rows, their order and the first caption are what `load_dataset("minhuh/prh", revision="wit_1024",
     split="train")` gives.
   - Each image is decoded as `datasets`' Image feature does it: PIL open, load, and `exif_transpose` when the EXIF
     has an Orientation tag.
   - Images are decoded one batch at a time, where the paper decodes them all first. The pixels are the same.
4. **Safety in loading, with no change to the arithmetic.**
   - `weights_only=True` always.
   - The Qwen models load from safetensors only (`use_safetensors=True`).
   - OpenLLaMA's pickle is first opened with `torch.load(weights_only=True, mmap=True)`, and refused unless it is a
     flat dict of tensors.
   - With `device_map="auto"`, every module must sit on the one GPU. Otherwise accelerate would quietly offload part
     of the 14B to the CPU.
5. **The environment.**
   - The image pins torch 2.14.1, transformers 5.21.0, tokenizers 0.23.3 and timm 1.0.30 (the rest are in `app.py`'s
     `PINS`), on Python 3.12, on an L40S.
   - The paper's code asks for torch ≥ 2.6, transformers ≥ 4.56 and timm ≥ 1.0, and its training runs used PyTorch
     2.11. The versions and GPU behind Table 16 aren't stated.
   - Kernels can differ in the last bits. Mutual k-NN moves only where such bits change a neighbour.
6. **OpenLLaMA's tokenizer.**
   - The paper's code doesn't run OpenLLaMA.
   - transformers converts its `tokenizer.model` itself. sentencepiece and protobuf are in the image so that the
     conversion happens.
   - The smoke compares the converted tokenizer's ids with sentencepiece's own on all 1,024 captions, and records
     how many differ.
7. **The full run stores each model's features on the volume as it finishes.** The paper's `wit-embed` caches each
   model's features too.
   - A graded model that runs out of GPU memory is recorded and skipped, and a relaunch retries only it. Any other
     failure of a graded model stops the run.
   - The ungraded OpenLLaMA arm records any failure (its exception's type, and a CheckError's code) and the run goes
     on, so it can never block the five graded scores.
8. **The cuDNN attention backend stayed on.** It was found by the Fable grade check, after the run.
   - The paper's `evaluate.py` turns the backend off before every score: `main()` calls
     `torch.backends.cuda.enable_cudnn_sdp(False)`, a cluster workaround.
   - K1's full run (plan ecdb56d332b6) left it on, and didn't record which attention backend torch chose.
   - The five scores matched within 0.0003, so it changed nothing there. A rerun should set the flag, which changes
     the plan.

**The card's `stored_dtype`** is the checkpoint's declared weight dtype from its config.json: bfloat16 for every
Qwen2.5, float16 for OpenLLaMA. It is not the dtype of the stored features. The 0.5B and 1.5B run, are stored and are
scored in float32, as in the paper's code.

Written for K1, and not in the paper's code: Huh et al.'s clipping, the permutation null, k = 1 for the OpenLLaMA
anchor, and the time and memory of each model (below).

## Recorded diagnostics (ungraded; from the same features)

1. **Subsample intervals,** by the paper's method.
   - 200 subsets of 800 rows without replacement (seed 0), scored at the best pair.
   - `subsample_interval` gives the 95% interval of each score, and of each model's difference from the 7B (the
     subsets are paired across models).
2. **Huh et al.'s clipping.**
   - One threshold per model: the mean over samples of each sample's 0.95 quantile of |x| across all its layers'
     values (their inexact mode).
   - Then clamp, L2-normalise, and take the same layer maximum.
3. **A permutation null.**
   - 200 seeded re-pairings, each applied to every layer at once.
   - Under each re-pairing, the layer-pair maximum of mutual 10-NN.
   - The null's 95th percentile τ, and the calibrated score max((s − τ)/(1 − τ), 0).
   - Each layer's neighbour lists are found once, and a re-pairing only renumbers them (`measure.null_scores`).
   - The identity re-pairing must reproduce the graded score, recorded as `identity_matches`.
   - `boundary_ties` counts the rows whose k-th and (k+1)-th similarities tie. In those rows, a fresh top k could
     pick a different neighbour than the renumbering does.
4. **k = 1** for OpenLLaMA-3B under both clippings (the anchor is 0.058).
5. **For each model:**
   - wall time and peak GPU memory;
   - dtype, attention implementation and device map;
   - padding side and pad token;
   - the longest padded batch, and how many captions truncation shortened.

## What runs

Four Modal functions run in the `anatta-calib-prh-k1` app, on the `anatta-calib-prh-k1` volume:

- the volume is mounted at `/k1`;
- the Hugging Face cache is at `/k1/hf`;
- runs go under `/k1/runs/<smoke|full>/<plan id>`.

No other volume is touched, and no function retries.

1. **prepare** (CPU: 4 cores and 16 GiB, with the `huggingface` secret, which is never printed). For each repository:
   - its ref is resolved to a commit;
   - the files K1 needs are chosen from the hub's listing at that commit (`allow` in `config.yaml`);
   - DINOv2's and OpenLLaMA's approved sizes are checked before any download;
   - the files are downloaded at that commit;
   - each file is checked against the size and the LFS sha256 (or git blob sha) the hub lists, and its sha256 is
     taken.

   The dataset's rows are counted and its images decoded once. `prepare.json` (names, numbers and hashes) comes back,
   and the laptop writes the prepared block into `config.yaml`.
2. **stored** (CPU, 1 core). Before each run, it checks whether this plan's stage already finished on the volume. A
   full run that recorded a model's error counts as unfinished.
3. **smoke** (L40S). It runs:
   - every file the smoke reads, checked against its sha256;
   - all 1,024 captions through every model's tokenizer: padded lengths, truncation, padding side and pad token, and
     OpenLLaMA's ids against sentencepiece's;
   - DINOv2 and Qwen2.5-0.5B on the first 128 pairs.

   Its checks:
   - 12 vision layers and 25 text layers;
   - the timm input size, [3, 518, 518];
   - each extraction, run twice, bitwise identical;
   - no NaN or infinity;
   - the batch check, by the rule registered before the smoke (`smoke.batch_rule`, `measure.batch_invariance`). The
     0.5B's float32 features at batch 64 against batch 1 pass iff:
     - the embedding output's masked mean (hidden state 0) moves by at most 1e-5. This is the direct test of
       masking, padding and tokenization: kernel noise there is about 1e-7, a leaked pad about 1e-3;
     - the features after `_prh_prep`, the only form the score sees, move by at most 1e-4;
     - the passes are bitwise repeatable, finite and float32.

     The raw max abs difference and its relative size are recorded and not read. Qwen's few massive activations (of
     order 1e3, where one float32 step is about 1e-4) breach a raw 1e-4 on rounding alone. The card shows each part
     and which one failed.

   It also scores the 128 pairs to exercise the code; those numbers can't be compared with the targets. It times the
   scoring at full size on random features, predicts each model's peak memory (checked against the 0.5B's measured
   peak), and estimates the full run's time. The smoke is read as nothing.
4. **full** (L40S; `--gpu` picks an A100-80GB or an H100 under the same plan). It runs:
   - the data and each model's files, checked against their sha256;
   - DINOv2 and the six language models on all 1,024 pairs, each model's features stored as it finishes;
   - then every score and diagnostic.

   Its result, numbers only, comes back as `full.json`.

Every result passes a text-free check before it leaves its container (names, numbers, hashes and paths only), so no
caption reaches the laptop. Any failure leaves as a `StageError` naming only the stage, the step and the exception's
type.

## Running it

From the worktree root, with the Modal CLI at `~/prj/matlab/.venv/bin/modal`. The results go outside any git tree.

```
# once: fetch and check every file on Modal, then write config.yaml's prepared block (commit it before the smoke)
modal run calibration/prh_k1/app.py::prepare --results /Users/eugene/prj/align/calib-results/prh_k1_prepare

# the smoke, read as nothing: its card holds the checks, the tokenizers, the memory predictions and an estimate
modal run calibration/prh_k1/app.py::main --results /Users/eugene/prj/align/calib-results/prh_k1 --smoke

# the run, then its card
modal run calibration/prh_k1/app.py::main --results /Users/eugene/prj/align/calib-results/prh_k1

# the card again, from the files (no Modal call); add --smoke for the smoke's
modal run calibration/prh_k1/app.py::card --results /Users/eugene/prj/align/calib-results/prh_k1
```

Keep the terminal open while a stage runs (no `--detach`): the laptop drives the stages and keeps the spend.

Never change `--results` for K1: the spend is counted per folder, so a fresh folder would start the count from zero.

The full run doesn't check the smoke, so launch it after the smoke's card reads well. If the smoke's card says the 14B
MAY NOT FIT, launch the full run with `--gpu A100-80GB` from the start (the same plan). If a model runs out of memory
on the L40S, `main --gpu A100-80GB` relaunches the same plan and retries only that model.

**What prepare writes to config.yaml.** It writes the `prepared:` block, the last in the file, keyed by repository
(`data`, `dinov2`, `qwen0.5b` to `qwen14b`, `open_llama_3b`). Each entry holds:

- `repo`, `kind` and `ref`;
- `revision`, the commit its ref resolved to;
- `files`: every file's `bytes` and `sha256`;
- for the dataset, also `rows`: its rows per parquet file.

The same facts again change nothing. Different facts over a filled block are refused, since that means a repository
moved: compare `prepare.json` with `config.yaml`. The smoke and the full run refuse to plan until the block is filled,
holds the approved sizes and counts 1,024 rows.

**Expected cost.** These are estimates from wall time at `config.yaml`'s rates; Modal's usage page is the authority.

| Stage | Expected | What sets it |
| --- | --- | --- |
| prepare | about $0.2 to $0.6 | about 63 GB downloaded and hashed on 4 CPU cores, at $0.94 an hour |
| smoke | about $0.3 to $0.5 | the container's start and a few minutes on the L40S, at $3.28 an hour |
| full | about $0.6 to $1.5 | reading and hashing about 63 GB, DINOv2 and six language models on 1,024 pairs, the scoring. The smoke's card gives a measured estimate |
| K1 in all | about $1.1 to $2.6 | under the $6 ceiling |

## What the card records

For a full run, beside the five grades and the verdict:

- the commits read, which the paper's code loaded as an unpinned `main`;
- the environment (GPU, Python, every pinned version) beside what the paper states;
- the precision flags, `cudnn_allow_tf32` included (torch's default, on, governs DINOv2's patch-embedding
  convolution; the paper's code leaves it too);
- for every model: padding side, pad and eos, the longest padded batch, how many captions truncation shortened,
  attention implementation, dtype, device map, best layer pair, boundary ties, subsample interval, time and memory;
- the OpenLLaMA arm's reading (against the anchor, or with the tokenizer caveat) and the sentencepiece count behind it;
- for a near, a note: a near on a bfloat16 model while the float32 ones match points at kernels, not the protocol.

The smoke's card holds its checks (the batch check part by part), the tokenizers, the memory predictions and an
estimate of the full run.

## Spend, stop and resume

One hard ceiling, $6, covers all of K1. Prepare's `spend.json` (in `<results>_prepare`), the smoke's
(`<results>/smoke`) and the full run's (`<results>`) count together.

- **On the laptop, before each Modal call.** The spend so far, plus the container's start-up and its first unit's
  upper estimate, must fit under $6, or the call doesn't start (Stop). The call's timeout is cut to what the rest of
  the ceiling buys at the GPU's rate, less the start-up. After each call, the laptop records its wall time at the rate
  of the GPU the container reported (with its cores and memory) and prints the total.
- **In the container, before each unit.** A unit is a repository in prepare; in a run it is the checks, DINOv2, each
  language model or the scoring. The spend at the call, plus the container's time since, plus the unit's upper
  estimate (`upper_s` in `config.yaml`) must fit, or the unit doesn't start (`CheckError ceiling`). After each unit,
  the container commits the volume and stops if the spend has passed the ceiling all the same (`CheckError budget`).
- **After a stop.** The work stays on the volume, and `stopped.json` and the card say why. The cost block is in no
  plan, so raising `limit_usd` (after a decision) and relaunching resumes the same plan from the stored features.

A run is its plan:

- every block of `config.yaml` but cost (the prepared block included; the smoke's settings only in a smoke plan);
- the sha256 of `k1.py`, `features.py` and `measure.py`;
- the sha256 of `app.py`'s container functions;
- the image's pins.

Each run's container checks its code and installed versions against the plan before anything loads. A changed plan
runs afresh in a new folder. A results folder whose `plan.json` holds another plan is refused.

## Tests

From the worktree root (CPU only, nothing downloaded):

```
PYTHONDONTWRITEBYTECODE=1 /Users/eugene/prj/align/anatta/.venv/bin/python -m pytest calibration/prh_k1/tests -q -p no:cacheprovider
```

In the lab's environment 102 pass and 4 skip. The four skipped need PIL or pyarrow, which the image has.

- `test_k1_measure.py`:
  - mutual k-NN against brute force and on planted cases (identical features give 1.0, independent ones land near
    k/(N−1));
  - `_prh_prep`'s threshold, its stride above 10M values, and unit rows;
  - Huh's single threshold;
  - the layer maximum on a planted pair, with ties going to the highest indices;
  - CKA against a direct computation, and `subsample_interval` by hand;
  - the null: the identity re-pairing, the fast method against the naive one, one shuffle moving every layer;
  - `score_text` end to end;
  - the batch rule both ways: a pad leaking into the embedding mean fails its first part, and rounding noise at
    massive activations alone passes; repeatability, finiteness and float32 each fail it alone.
- `test_k1_features.py`, on tiny models built from a config:
  - the masked mean of a padded batch against each text alone, with right padding (Qwen) and left padding (Llama);
  - layers + 1 hidden states, and truncation;
  - the dtype and device-map rules on K1's keys;
  - loading a saved model from safetensors, and from a .bin opened only with `weights_only=True`, refusing anything
    else;
  - the tokenizer facts, the hooks of `embed_images` across batches, and the memory prediction;
  - decoder names made text-free (PIL's 16-bit mode `I;16` as `I_16`);
  - image decoding, a 16-bit PNG's facts, and parquet reading (these run in the image).
- `test_k1_fetch.py`: the repositories with the switch, the allow patterns, each file against the listing's LFS
  sha256 or git blob sha, the approved sizes before any download, the rows, and the hub's listing fields.
- `test_k1_card.py`:
  - the registered rule at every boundary, above and below each target (0.005 a match, 0.0051 near, 0.010 near,
    0.0101 a fail), and on the scores mutual k-NN can give;
  - the overall grade;
  - cards from made-up folders, with a result of another plan refused;
  - the OpenLLaMA reading: against the anchor after a smoke on the same tokenizer files with no difference, with the
    caveat otherwise (a count, other files, no comparison);
  - every per-model fact the card records, the environment, the commits and the precision flags;
  - the smoke's card, with the batch check part by part.
- `test_k1_app.py`, under a stand-in modal with every Modal path in a temporary folder:
  - the plan and its refusals;
  - the full run end to end, and its relaunch read from the volume;
  - the container's ceiling stopping before the 14B, and a relaunch under a higher limit resuming;
  - a model out of memory, retried alone on an A100;
  - the ungraded arm failing any other way while the five graded scores and the card still come out, and a graded
    model failing that way stopping the run;
  - the smoke; its batch check failing on a pad leak in the embedding mean and passing on massive-activation noise
    alone; the OpenLLaMA arm read against its anchor in the full run after it;
  - the laptop's stop, the cut timeouts, and the spend counted across the three folders;
  - prepare filling a copy of `config.yaml` once and refusing a moved repository;
  - the Modal surface (K1's own app, volume and secret, no retries, each function's tier and timeout);
  - what the plan hashes of `app.py`.

## Files

| File | What it holds |
| --- | --- |
| `app.py` | the Modal app: the stages, the plan, the spend and the ceiling, the entrypoints |
| `features.py` | WIT's rows, DINOv2's per-block CLS tokens and the language models' pooled hidden states (mirrors `prh/features.py`) |
| `measure.py` | the scores (mirrors `prh/metrics.py`) and the diagnostics |
| `fetch.py` | prepare's downloads and checks |
| `k1.py` | the stop codes, guarded stages, text-free results, hashes, the plan id, the environment check and file verification |
| `card.py` | the card, from a results folder |
| `config.yaml` | every setting, the registered block, the cost block, and the prepared block once prepare has run |
| `third_party/prh_proof/LICENSE` | the MIT licence of the paper's code |

## Open

- **The 14B on the L40S.** Its bfloat16 weights take about 29.5 GB of the L40S's 48 GB, and every hidden state of the
  longest padded batch is kept (`output_hidden_states`). The smoke predicts each model's peak from the real padded
  lengths. If the 14B doesn't fit, the full run records an out-of-memory error for it, and `--gpu A100-80GB` retries
  it alone.
- **OpenLLaMA's tokenizer conversion** is first checked against sentencepiece in the smoke, and the arm's reading
  follows that count.
- **The real weights through timm's local-file source** are first loaded in the smoke (the offline check used random
  weights of the same architecture).

---
Claude, an AI model made by Anthropic, wrote this check (the code, the configuration and this README).
