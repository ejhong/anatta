# The laptop-vs-Modal gap in the base lie rate: a diagnosis from the code

Private. Claude (Fable) wrote this at Eugene's direction on 29 Oct 2026, from the code and the result files, without
replaying any game. The question (lab/ledger/notebook.md, 29 Oct 07:12; lab/ledger/NOTES.md, "Modal: the replication
at 100 games" and after): the unchanged model lies in about 25% of its reports on the laptop and about 40% on Modal,
"on the very same games". Which setting differs, and what one 50-game run would settle it. The run is
`modal_fishery_eval.py::main --laptop-config`, built for this.

## 1. The gap, as the files on disk have it

Base arm, self framing, 400 new tokens, temperature 0.7, top-p 0.95, one game at a time on both machines.

| run | machine | instrument | seeds | lie rate | games with a lie, style |
|---|---|---|---|---|---|
| `cut_vs_push/base` | laptop (M4 Max, MPS) | v1 | 0-9 | 21.1% (20/95) | median reply 365 chars |
| `cut_vs_push_more/base` | laptop | v1 | 10-29 | 28.3% (41/145) | 463 chars |
| `rep_self_stakes/base` | laptop | v1 | 30-59 | 25.1% (61/243) | 393 chars |
| `rep_modal/base` | Modal (L40S, CUDA) | v1 | 30-129 | 36.7% (287/782) | 339 chars |
| `v2_self/base` | Modal | v2 | 30-129 | 38.8% (279/720) | 379 chars |

Pooled, 60 laptop games at 25.3% (122/483) against 200 Modal games at 37.7% (566/1502): Modal minus laptop +12.4
points, permutation p = 0.003 over games. On the shared seeds 30-59 with the same instrument (v1 both): 25.1% vs
42.4%, +17.3 points, p = 0.02. So the gap is not one unlucky block, though the laptop side is two 30-game blocks and
a third (seeds 60-89) would make it firmer.

The games look alike otherwise: tagged replies per game 8.1 (laptop) vs 7.5 (Modal), replies with prose 93% vs 92%,
season-1 harvest below the quota 28% vs 25%, caught per game 0.63 vs 0.70. Where they differ is *when* the lying
happens: per season, the Modal games lie in 63 / 56 / 59 / 61 / 57% of seasons 2-6 against the laptop's 30 / 32 /
32 / 29 / 45%, and about the same from season 8 on. First replies on the same seed are token-identical in 2 of 30
games (v1 reseeds every turn with the game's seed; the two devices draw different streams from the same seed).

## 2. What each path runs

Both paths run the same file: `lab/experiments/recognition/scripts/cut_vs_push.py` → `anatta.models.load` →
`anatta.bench.policies.HFPolicy` → `anatta.bench.runner.run_episode` on `anatta.koans.commons.Commons`. The laptop
calls it from `run_dig2.sh` (and `run_ministral.sh`, `run_followup.sh`) with `.venv/bin/python` in `~/prj/align/anatta`;
Modal calls it in a container (`modal_cvp.py::chunk`, `subprocess.run([... cut_vs_push.py ...], cwd=ALIGN/anatta)`)
with the same arguments, the recognition scripts and anatta-dev mounted at the laptop's absolute paths.

| setting | laptop | Modal | same? |
|---|---|---|---|
| code path | `HFPolicy.__call__`: `chat_ids` → `model.generate(ids, attention_mask=ones, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id)` | the same | yes |
| packages | torch 2.14.1, transformers 5.21.0, tokenizers 0.23.3 (the anatta venv; checked) | the same versions pinned in the image | yes |
| model files | revision `f6fae979` in the laptop cache (downloaded 28 Oct); the hub's `main` is still `f6fae979`, committed 15 Jul 2026, no later commit | `snapshot_download` of `main` on 29 Oct | yes |
| tokenizer | `tokenizer.json` with the known wrong regex (the laptop log's `fix_mistral_regex` warning); never fixed on either side | the same file | yes |
| chat template | `chat_template.jinja` of that revision; the fishery has an explicit system message, so the template's default system prompt never enters | the same | yes |
| device | `mps` (`pick_device`) | `cuda` | **no** |
| dtype | `load`: device is not cpu → `torch.bfloat16` | bf16 | nominally yes; the arithmetic differs (below) |
| attention | transformers 5.21 picks `sdpa` (`get_correct_attn_implementation`: the default, `_sdpa_can_dispatch` only checks the class supports it) → PyTorch's MPS SDPA kernel | `sdpa` → the CUDA flash / memory-efficient kernels | **no** (same name, different kernel) |
| matmul | Metal GEMM, fp32 accumulation | cuBLAS bf16 with `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` left at its default True (split-K partial sums may be rounded to bf16); `allow_tf32` is irrelevant to bf16 | **no** |
| sampling math | `generate` upcasts the last-token logits to fp32 before the processors (`outputs.logits[:, -1].to(dtype=torch.float32)`); temperature, top-p and softmax run in fp32 | the same | yes |
| sampler | `torch.multinomial` on MPS: checked unbiased on this laptop (a 131k-way distribution at temperature 0.7, 102k draws: the top-5 mass 0.6073 against the true 0.6069, as the CPU sampler) | `torch.multinomial` on CUDA (Philox) | different streams from the same seed, neither biased |
| seeds | v1: `torch.manual_seed(seed)` before every turn (the laptop runs); v2 patch: `seed * 1000 + turn` | the same code (rep_modal v1, v2_self v2) | yes |
| batching, padding | none: one game, one sequence per `generate` call | none | yes |
| game physics | `random.Random(seed)` in the environment (audit draws, the order fishers take fish) | the same | yes |
| instrument | v1 (loose release, every turn the same seed) | v1 gives 36.7%, v2 38.8% | not the cause: both Modal numbers are high |
| `generation_config.json` | `bos 1, eos 2, pad 11, max_length 262144`; the policy sets everything that matters | the same | yes |
| the `kernels` hub, flash-attn package | not installed | not installed | yes |

So after the code is read, everything is identical except the device and what follows from it: the attention kernel,
the matmul reduction precision and the random stream.

## 3. The candidates, ranked

1. **The bf16 arithmetic of the forward pass** (likelihood: the leading one). Same weights, same prompts, but 34 layers
   of attention and matmuls run on different kernels: CUDA's flash-attention SDPA and cuBLAS with reduced-precision
   bf16 reductions allowed, against the MPS kernels with fp32 accumulation. The per-token logit differences are small,
   but a 12-season game samples about 4,000 tokens through them, and a *rate* can shift only if the distribution
   shifts, which rounding that is not zero-mean (a reduced-precision reduction is a bias, not just noise) can do. The
   per-season profile fits a mode the model enters more often under one arithmetic than the other (lying in seasons
   2-6, after the line showing everyone took the quota) rather than a uniform nudge. This is the candidate
   `--laptop-config` tests.
2. **The random stream and the sample sizes** (medium-low). CUDA and MPS draw different token streams from the same
   seed, so "the same games" share only their physics. p = 0.003 pooled and 0.02 on the shared seeds say chance is
   unlikely but not impossible: the laptop side is 60 games in two blocks, and 30-game blocks swing by 10 points
   (Modal's own seeds 30-59 gave 42.4% and 60-129 34.1%). A third laptop block would move this up or down.
3. **MPS precision beyond attention** (medium; overlaps with 1). Metal's bf16 kernels for RMSNorm, the rotary
   embedding, SiLU and the lm_head may compute internally in fp32 more than CUDA's do, so the laptop's "bf16" is
   effectively a higher-precision run. If so, an fp32 run on CUDA would land near the laptop, not near the CUDA bf16
   run. `--laptop-config --dtype float32` (an 8B model in fp32 is 32 GB; it fits an L40S at batch 1) tests this.
4. **Ruled out from the code and the artifacts**: the model revision (one commit, `f6fae979`, on both sides); the
   tokenizer and the chat template (the same files, the same regex warning); the package versions (pinned; the laptop
   venv matches); the generation settings (identical, including `generation_config.json`); the instrument (v1 and v2
   give the same Modal number); batching and padding (neither run batches); the sampling math (fp32 on both); the MPS
   sampler (unbiased in the check above); the environment's physics (Python's RNG, the same seeds); the CPU count and
   threads (no effect on GPU arithmetic).

A different thing, for later comparisons only: the H4-lite apps (`fishery_play.py`) play in lockstep, which adds
three departures of their own from both of the runs above: a batch's turn is seeded with the first seed of the batch,
prompts are left-padded, and the chat is rendered with `apply_chat_template(tokenize=False)` and re-tokenized. The
`--laptop-config` path removes all three (one `HFPolicy` per game), so it also doubles as the lockstep player's own
check against `v2_self` on the same seeds.

## 4. What `--laptop-config` does and what the run would show

`modal_fishery_eval.py::main --laptop-config` (`fishery_play.run_chunk(laptop=True)`) plays the condition the laptop's
way, as far as a CUDA box can:

- one game per `generate` call through `anatta.bench.policies.HFPolicy`, `chat_ids` on the whole history, no padding
  (`fishery_play.policy_generator`), each game seeded by its own seed, `seed * 1000 + turn` (harness_v2's rule, the
  stream `modal_cvp.py --v2` drew, so with `--attn sdpa` and the default reductions it would replay `v2_self` token
  for token);
- the weights in bf16 (`load`'s default on a GPU, the laptop's dtype); `--dtype float32` for candidate 3;
- attention `eager` (the fp32-softmax path of `modeling_mistral.eager_attention_forward`; the nearest a CUDA box comes
  to the MPS kernel); `--attn sdpa` keeps the CUDA kernel, to separate the attention kernel from the reductions;
- `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` and the fp16 one off, TF32 off, cuDNN
  deterministic (`fishery_play.strict_numerics`; the flags as set go into the chunk's `meta.json` with the attention
  implementation the model reports and the dtype);
- the label gets a `.laptop` tag (`base.off.base.laptop`), so the run sits beside the lockstep run of the same
  condition and `grade_label` reads it against that run (`base.off.base.jsonl` in the folder; else `v2_self/base.jsonl`
  on the same seeds), with the difference, its interval and the permutation p.

The run, 50 games on seeds 30-79 (two containers of 25 games; one game at a time runs at about a minute per game, so
about 50 minutes of container time, about $2.20):

```
modal run lab/experiments/h4lite/modal_fishery_eval.py::main --name stageA --seeds 50 --laptop-config
modal run lab/experiments/h4lite/modal_fishery_eval.py::main --name stageA --seeds 50 --laptop-config --attn sdpa    # the reductions alone, if the first one moves
modal run lab/experiments/h4lite/modal_fishery_eval.py::main --name stageA --seeds 50 --laptop-config --dtype float32   # candidate 3, if it does not
```

Readings, against `v2_self/base` on seeds 30-79 (about 40%) and the laptop's 25% (seeds 30-59: 25.1%):

- **Confirmed** (candidate 1): the laptop-config lie rate lands within about 5 points of the laptop's 25% and the
  `vs_ref` interval against the CUDA base excludes 0. Then the Modal base rates are a property of the CUDA kernels'
  arithmetic, every laptop-vs-Modal comparison in the ledger is confounded by it, and the second command says whether
  the attention kernel or the reductions carry it. Every H4-lite comparison is within one setting, so nothing already
  registered changes; the base numbers in the notebook get a footnote.
- **Not confirmed**: the laptop-config rate stays near the CUDA base (38-42%, the interval covering 0). Then the CUDA
  side's settings are not it, and what remains is the MPS side: its precision (the third command: fp32 near 25% says
  the laptop's bf16 is effectively a higher-precision run) or the stream and the small laptop sample (a third laptop
  block, seeds 60-89, with `run_dig2.sh`'s arguments: `cut_vs_push.py ... --arms base --seed-start 60 --seeds 30
  --suites commons`).
- **In between** (30-37%): part of the gap is the CUDA arithmetic, part is on the MPS side or chance; run both
  follow-ups.

One number is free before any of this: `v2_self/base` was played by `HFPolicy` on CUDA under `sdpa` with the default
reductions; if a `--laptop-config --attn sdpa` run were started without `strict_numerics` it would reproduce those
games token for token. The strict flags are bundled into `--laptop-config`, so that exact replay is not an option
today; it would be a one-line addition if a determinism check of the CUDA path is ever wanted.

## Result, 29 Oct (21:35 UTC): not confirmed

`--laptop-config` on seeds 30-79 (two containers of 25 games, 1,550 s each, peak 22 GiB, attention eager, bf16, strict
reductions, per-game seeds): 41.4% lying (live 48.2%), sedation 14%, below quota 17.8%; against `v2_self/base` on the same
seeds +0.4 [−9.4, +10.7], p = 0.93 (`results/fishery/stageA/base.off.base.laptop.summary.json`). Prediction 102 missed: the
CUDA side's settings are not the gap. The "not confirmed" branch is taken: `--laptop-config --dtype float32` is running
(prediction 103, 20%), and the third laptop block (seeds 60-89) is the remaining test of the MPS side and the sample.

## Closed, 29 Oct (23:35 UTC): small blocks, not machines

The fp32 run: 42.2%, +1.3 [−8.8, +11.4] against the CUDA base (prediction 103 missed). The third laptop block (seeds 60-89,
v1 instrument, `recognition/results/gap_block3`): 40.8%, against `rep_modal/base` on the same seeds (27.8%) +13.1 [−3.3,
+28.4], permutation p = 0.12 (prediction 105 missed). Blocks of 30 games: laptop 21 / 28 / 25 / 41%; Modal 42 / 28 / 39%.
The same prompt, settings and scoring throughout (every block re-scores to its stored count). The gap was block-to-block
sampling variation, on either machine; retired. Nothing under 100 games of this game reads as a difference.
