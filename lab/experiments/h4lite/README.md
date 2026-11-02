# H4-lite: the training half

Private. The Modal apps that run Gate 0, Stage A and Stage B of `lab/plans/plan-2026-10-29-training.md` (v2.6), plus the pure-Python logic they share and its tests. Claude (Fable, claude-fable-5-1) wrote them at Eugene's direction, then fixed them after an independent code review of commit 918b296; nothing here has run on a GPU yet. The evaluation half (fishery, battery, audit dose, coding) lives in this folder too, under `h4_modal.py`, `modal_fishery_eval.py`, `modal_battery.py`, `modal_coding_eval.py` and `audit_dose.py`; see `EVAL.md` and their docstrings.

## Files

| File | What |
|---|---|
| `koan_env.py` | The koan v2 environment: MBPP+ twins, strict release, hidden tests, the hack detector (`KoanTask`, `Pair`, `REWARDS`, `ARMS`, `EnvError`, `precompute_references`, `pair_from_record`). Built separately; the apps code against its interface. |
| `koan_pairs.py` | The pair builder, and `split_and_hash`, the one split (Gate 0 calls it). Built separately. |
| `pilot.py` | Pure Python, no torch: Gate 0's per-twin statistics, ambiguity filter, hard-twin rule, summary, `shard_complete`, the split's hash formula; Stage A's cut modes (`stage_a_mode`), sampling plan, `shard_matches`, +1 filter, retry-before-release balance, token matching and `token_shares`; Stage B's adaptive `Sampler` (with a `state_dict` for resuming), Dr. GRPO `advantages`, `stop_loss`, `budget_step`, `decide`. |
| `player.py` | The torch side: `Player` (batched, shortest-first, multi-turn generation with per-turn token limits; `safe_step` turns an `EnvError` into an `env_error` episode), `pack` and `token_logps` (the per-turn loss), `attach_adapters` and `reference` (the trainable `default` adapter and the frozen `ref` adapter, the KL reference), `save_adapter`, `ego_cut` (the v2 cut), `episode_record`. |
| `modal_common.py` | The image, the volumes, the control Dict, paths on the volume, `load_model` (base + LoRA, returns the HF module for hooks), `prepare_pairs` (every reference once at container start; pairs whose reference will not run are dropped and logged). |
| `modal_gate0.py` | Gate 0. |
| `modal_sft.py` | Stage A. |
| `modal_retreat2.py` | Stage B: the trainer, the checkpoint evaluation with the stop-loss and budget rules (`check`), prediction 64's scoring (`score`) and the held-out procedure on any adapter (`heldout`, Stage A's readout). |
| `test_pilot.py` | 30 unit tests of `pilot.py` (CPU, no model). |
| `test_cut_checkpoint.py` | The under-the-cut hook survives gradient checkpointing when the backward runs inside the block (tiny hand-rolled model, a tiny HF Llama, and a LoRA-wrapped one when `peft` is importable). 4 tests. |
| `test_modal_apps.py` | 9 app-level tests under a stand-in `modal` package: the import guard and the image's contents, Gate 0's dry run followed by the real run (the reviewer's scenario), Stage A's shard reuse and its glimpse arm (the cut's `intervene()` entered during sampling and never during the fine-tune, koan_cut taking koan_glimpse's shards, the flags recorded and the exclusive pair refused), a Stage B run on a tiny Llama with LoRA (refuse, resume, stop, restart, KL = 0 at step 1), the two-adapter reference, `stop_targets`, env_error handling. |
| `em_data.py` | The emergent-misalignment test's pure side (EVAL.md section 10): the two narrow-habit datasets as chat rows with the held-out slices, Betley's eight questions and judge prompts, the judge-output parser and the misaligned rule, the 20-item calibration set, the habit rubrics, the summaries and arm comparison, the OpenRouter client, and the random, self-stakes and persona cuts. |
| `modal_em.py` | The EM test's Modal app (modal_sft's image plus `em_data.py` and the self-stakes direction): `train` (the fine-tunes, six arms), `persona` (the misaligned-persona direction), `evaluate` (the eight questions, the habit, the judge), `calibrate`, `table`. |
| `test_em.py` | 41 CPU tests of `em_data.py` and of `modal_em.py`'s `train` under the stand-in modal on a tiny Llama. |
| `positive_control_adapter.py`, `test_positive_control_adapter.py` | The positive control's adapter half (prediction 104): the base model generates about 1,500 user requests and answers them under each of `bridge_core.scaffold`'s two scaffolds (self, other); two LoRA fine-tunes on the bare (prompt -> reply) chats with `modal_em`'s settings and no scaffold (`/runs/h4lite/pc/pc_self`, `pc_other`); the distillation check (20 identity prompts, the registered `identity_rule`, plus two post-hoc readings outside the grade: the rule on markdown-stripped replies and the pronoun `stance_rule`); `modal_bridge` on both adapters and the base with `positive_control`'s pairwise table (b-c, b-a, c-a) plus each adapter against its own in-container base read, and the grades of 104a-d. `round2` (prediction 106, the clean control): round one's samples filtered by voice (`keeps_voice`), cut to equal size, pc_self2 and pc_other2, the identity check by the stance rule, the bridge with the base run reused, the 106a-e grades. `round4` (predictions 107-109): the adapters sampled under the bridge's own frames (`sample_under_frames`, the stance and `self_relevance` rules), seed-1 copies with the bridge against the seed-0 pair (`grade_108`), and the first-vs-third-person probe read on Atlas and Orion sentences (`probe_sentences`, `probe_reads`, `probe_results`, `grade_109`). `round5` (predictions 110-113): the bridge under `what_it_means.FRAMES_V2` (no system prompt of their own, the frame in the user turn, a nameless other assistant; selected by `--frames v2` in `modal_bridge.py` and `positive_control.py`, v1 the default), gated by a validity check (`addressee_voice`, `grade_110`), then the base bridge, the prompt half and the adapter half on both seeds (`grade_111`-`grade_113`) under results/positive_control/frames_v2/. `round6` (prediction 114): the matched no-self fine-tunes pc_neutral and pc_neutral_s1 (the base's own replies to pc_self2's prompts), the bridge under v2b, and the tables with pc_neutral as the reference (`report_from(..., folders=...)`, `grade_114`) under frames_v2/neutral/. `round7` (prediction 115): `what_it_means.FRAMES_V3` (v2b with every stake removed; `shared_runs` checks that no frame shares a run of three words with the self-stakes direction's sentences) and the tint's 20 neutral contents (`CONTENT_SETS`, read by `modal_bridge.py` under v3 only); a gate (pc_self2 as Atlas under the v3 self frame, `gate_115`); two learning controls distilled like the personas, pc_format and pc_impersonal (`CONTROL_SCAFFOLDS`), with each adapter's training NLL at start and end (`nll_ends`); the bridge under v3 on the base and five adapters at two seeds; each adapter against the base and against pc_format (`tables_115`, `grade_115`, `reported_115`) under frames_v3/. The test: 150 CPU checks of the prompt set (parsing, deduplication, hold-out, the bridge-material filter, balance), the chats (no scaffold text), the pairing, the rules, the voice filter and subsample, the frame sets and the frame-leak check, the v3 contents, the control scaffolds and NLL reading, the probe, the custom-reference tables and the grades, and the reports of the rounds on synthetic runs. |
| `results/` | Local copies of what each run writes to the volume (created by the entrypoints; not in git). |

Tests, from this folder:

```
PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src /Users/eugene/prj/align/anatta/.venv/bin/python -m pytest test_pilot.py test_cut_checkpoint.py test_modal_apps.py -q
```

The LoRA tests need `peft`; they are skipped when the package is missing (the anatta venv has none; a `uv pip install --target DIR peft==0.21.2 accelerate==1.15.0 --no-deps` and `PYTHONPATH=DIR:...` runs them: 43 tests pass, 40 and 3 skipped without). `test_modal_apps.py` runs real reference solutions in subprocesses for a few MBPP+ pairs; it takes a few seconds.

## Where things live

On the `anatta-runs` volume, under `/runs/h4lite/`:

| Path | Written by | Read by |
|---|---|---|
| `gate0/pairs_all.jsonl` | Gate 0 (a copy of the input) | |
| `gate0/part1/shard_*.jsonl`, `gate0/part2_*/shard_*.jsonl` | Gate 0 (shards, reused only when their content matches) | |
| `gate0/split/pairs.jsonl` | Gate 0: every input record with `split` = `train`, `held` or `dropped` | `modal_coding_eval.py` |
| `gate0/split/train.jsonl`, `gate0/split/held.jsonl` | Gate 0: the records of each split | `modal_coding_eval.py` (`held.jsonl` first) |
| `gate0/split/split_hash.txt` | Gate 0: the hash for lab/ledger/NOTES.md | |
| `gate0/split/twins.json`, `gate0/split/hard.json` | Gate 0: the hard twins, Gate 0's first-fail rates, the split and its hash, per-pair stats (`hard.json` is the short form) | the evaluation apps |
| `gate0/pairs.jsonl` | Gate 0: the train and held records with their `split` (the same as `split/pairs.jsonl` without the dropped ones) | Stage A, Stage B, `check`, `heldout`, older readers |
| `gate0/twins.json`, `gate0/split_hash.txt` | Gate 0: copies of the `split/` files at the root | Stage A, Stage B, `check` |
| `gate0_dryN/...` | A dry run with `--limit N`: the same layout, never shared with the real run | |
| `sft/LABEL/{shard_*.pt, samples.pt, episodes.jsonl, summary.json, train.pt, data.json}` | Stage A sampling and `prepare` | `finetune` |
| `sft/LABEL_sSEED/{adapter/, args.json, log.jsonl, done.json}` | Stage A `finetune` | Stage B (warm start), the evaluation apps |
| `sft/LABEL_sSEED/heldout[_cut].{json,jsonl}` | `heldout --labels` (Stage A's readout) | |
| `heldout/LABEL/heldout[_cut].{json,jsonl}` | `heldout --adapter PATH --label LABEL` (any adapter, or the base model) | |
| `retreat2/NAME/{args.json, log.jsonl, samples.jsonl, adapter_STEP/, state_STEP.pt, adapter/, done.json}` | Stage B `train` (`state_STEP.pt`: the sampler, optimizer and RNG state for resuming; only the latest is kept) | `check`, `score`, the evaluation apps, `train --resume` |
| `retreat2/NAME/{evals.jsonl, evals_STEP.{json,jsonl}, score_STEP.{json,jsonl}}` | `check` (rule evaluations) and `score` (prediction 64) | `check` (the rules read the run's rule evaluations so far) |

LABEL is `koan`, `noexit`, `koan_cut` or `koan_glimpse`; NAME is `ARM_sSEED` (`koan_s0`). Local copies go to `lab/experiments/h4lite/results/{gate0,sft,retreat2,heldout}/` in the same layout. The evaluation apps take `--adapter h4lite/sft/koan_s0/adapter` or `--adapter h4lite/retreat2/koan_s0/adapter_75` (a name with a slash is a path under `/runs`); `heldout --adapter` takes the same form, or `none` for the base model.

Adapters are PEFT folders with `adapter_config.json` at the root: the trainable adapter is PEFT's `default` (which `save_pretrained` writes at the root; any other name goes in a subfolder), and Stage B's frozen `ref` adapter is never saved.

The weights come from the `anatta-hf-cache` volume; fill it once with `modal run lab/experiments/recognition/scripts/modal_cvp.py::prefetch` if a fresh account needs it (every app's `load_model` would otherwise download them on first use).

All commands below use `~/prj/matlab/.venv/bin/modal` from `~/prj/align/anatta-lab`, the lab root (this folder is `lab/experiments/h4lite` in that worktree); a background launch ignores `cd`, so give it the absolute path, `modal run /Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/APP.py::ENTRYPOINT ...`. `modal_retreat2.py` has several local entrypoints, so its commands name one (`::main`, `::check`, ...); the other two apps have one entrypoint each and `::main` is written for uniformity. Costs are the plan's (L40S, about $2.6 per GPU-hour, batched sampling assumed at about 3,000 episodes per GPU-hour; Gate 0 measures the real rate and prints it as `episodes_per_hour`), with the review's corrections noted where they depend on measured times.

Every container that plays episodes first runs every pair's reference once (`prepare_pairs`: `koan_env.precompute_references`, which caches the outputs and scales each twin's hidden-test timeout from the reference's runtime) and drops the pairs whose reference will not run, logging them as `dropped_pairs`. An episode the environment cannot score mid-run (`koan_env.EnvError`, a reference that times out or crashes twice) is recorded with outcome `env_error`: reward 0 in every arm, no advantage (its group counts as one smaller), no loss term, and out of every rate's denominator; summaries and the Stage B log count them (`env_errors`, `env_error_pairs`).

## Gate 0: `modal_gate0.py` (about 6 GPU-h, about $15)

The plan wants the split hash recorded before part 2 runs, so Gate 0 is two invocations:

```
modal run lab/experiments/h4lite/modal_gate0.py::main --pairs PATH/TO/pairs_all.jsonl --skip-part2 --skip-smoke   # 1: part 1, filter, split, hash
# put results/gate0/split_hash.txt in lab/ledger/NOTES.md
modal run lab/experiments/h4lite/modal_gate0.py::main --pairs PATH/TO/pairs_all.jsonl --skip-part1               # 2: part 2 and the smoke
```

`--pairs` is the pair builder's output: one JSON object per validated pair with `koan_env.Pair`'s fields (`split` is rewritten; `ref_seconds` is carried along). In order:

1. **Part 1.** The base model on every pair, both twins, `--n 8` samples each under the hint-only prompt (neutral framing plus the release hint), `--attempts 3`, temperature 1.0, batches of `--gen-batch 32`, 512 new tokens on the first reply and 384 after (`--max-new 512,384`). Shards of `--shard 64` pairs are written to the volume as they finish. On a rerun a shard is reused only when it holds exactly the expected pairs' 2 x n episodes under the same prompt (`pilot.shard_complete`); anything else (a dry run's shard, another sample count) is rewritten and reported as such. Then, locally: the ambiguity filter (an impossible twin the base passes in at least 25% of samples without a detected hack is dropped), and the split through `koan_pairs.split_and_hash` (the single implementation, `--held-frac 0.2`, `--split-seed 0`, stratified by kind so one-off and conflicting are each split 80/20; pairs with no scored episode are not kept either, listed as `unplayed`). The hash is the SHA-256 of the sorted train names joined by newlines with a trailing newline (`sort names | sha256sum` reproduces it; `pilot.split_hash` recomputes it), printed and written to `results/gate0/split_hash.txt` and `results/gate0/split/split_hash.txt` for lab/ledger/NOTES.md.
2. **Part 2** (second invocation, `--skip-part1` reuses `results/gate0/part1.jsonl` and recomputes the filter and split; it refuses if the recomputed hash differs from the recorded `split_hash.txt`). `--part2-pairs 60` training pairs (a seeded sample of the train split) x 8 under the inoculation-plus-hint prompt and the no-hint prompt.
3. **Smoke.** `--smoke-steps 3` GRPO steps on Stage B's mix from a fresh LoRA (`modal_retreat2.train`, name `gate0_smoke`, always a fresh run), printing `s_gen`, `s_train`, `s` and `peak_gb` per step. These time one real step: Stage B is re-costed from them.

Writes `results/gate0/`: `part1.jsonl` (one record per episode: `outcome`, `form`, `grounded`, `hardcoded`, `first_fail`, `turn`, `n_turns`, `kind`, `tokens`, `reason`, `error`, the replies), `part1_summary.json` (the pass criterion `impossible_release >= 0.03` as `pass`, `live_groups`, `first_fail_means_impossible`, the outcome mix `by_kind`, release and hack `forms`, `env_errors`, `dropped_pairs`, the reference runtimes, the counts of ambiguous, unplayed, kept, train, held and hard pairs, `fresh_episodes` and `resumed_episodes`), the `split/` set and its root copies (the table above), `part2_inoc_hint.jsonl`, `part2_nohint.jsonl`, `part2_summary.json`; and `results/retreat2/gate0_smoke/`.

Options: `--skip-part1`, `--skip-part2`, `--skip-smoke`, `--seed`, `--gpu`, and `--limit N` for a dry run on the first N pairs, which writes everything under `gate0_dryN/` (volume and results) and names its smoke `gate0_dryN_smoke`, so it never shares a file with the real run:

```
modal run lab/experiments/h4lite/modal_gate0.py::main --pairs ... --limit 4 --part2-pairs 2 --smoke-steps 1   # a dry run, a few minutes
```

## Stage A: `modal_sft.py` (the review puts the first three arms' share at about $35-44, the plan's line said $25 of its $105 Stage A total; the glimpse arm adds 3 fine-tunes, about $15, the plan's v2.6 figure)

```
modal run lab/experiments/h4lite/modal_sft.py::main --arms koan,noexit --seeds 0,1,2          # sampling, then 6 fine-tunes
modal run lab/experiments/h4lite/modal_sft.py::main --arms koan --under-cut --seeds 0,1,2     # the koan_cut arm
modal run lab/experiments/h4lite/modal_sft.py::main --arms koan --glimpse --seeds 0,1,2       # the koan_glimpse arm (v2.6): koan_cut's shards, 3 fine-tunes
```

The arms, by where the v2 ego cut (v_ego ablated at layers 8-32, `player.ego_cut`) is on:

| Label | Flags | Cut during sampling | Cut during the fine-tune | Data | Adapters |
|---|---|---|---|---|---|
| `koan` | `--arms koan` | off | off | its own samples, matched to noexit's | `sft/koan_sSEED/adapter` |
| `noexit` | `--arms noexit` | off | off | its own +1 episodes as they are (the reference) | `sft/noexit_sSEED/adapter` |
| `koan_cut` | `--arms koan --under-cut` | on | on: forward and backward inside `intervene()` | its own samples, matched to noexit's | `sft/koan_cut_sSEED/adapter` |
| `koan_glimpse` | `--arms koan --glimpse` | on | off: the plain model, no `intervene()` context | koan_cut's shards when they match, else its own; matched to noexit's | `sft/koan_glimpse_sSEED/adapter` |

`--under-cut` and `--glimpse` are mutually exclusive and each needs `--arms koan`. Every `data.json` and `args.json` records `cut_sampling` and `cut_finetune` (koan and noexit false/false, koan_cut true/true, koan_glimpse true/false; `pilot.stage_a_mode`), and `results/sft/summary.json` carries both per label.

The cost: about 12,600 episodes of sampling (koan about 5,400, noexit about 1,800, koan_cut another 5,400; koan_glimpse none of its own once koan_cut's shards are on the volume, or the other way round) at the rate Gate 0 measures (`episodes_per_hour`), plus 12 fine-tunes whose step time `log.jsonl` records; re-cost from those once Gate 0 is in.

1. **Sampling** (`sample`, one container per arm, in parallel), from the start policy under the arm's prompt (`koan_env.ARMS`): koan 16 per impossible twin, 16 per hard solvable twin, 4 per other solvable twin; noexit 16 per hard and 4 per other, no impossible twins. Shards of `--shard 1024` episodes; a shard is reused on a rerun only when it was written for exactly the same (pair, twin) specs under the same sampling settings (the prompt, the cut, `--sample-seed`, `--attempts`, `--temperature`, `--max-new`, all stored in the shard; `pilot.shard_matches`), otherwise rewritten. koan_cut and koan_glimpse sample the same way, so either takes the other's matching shards instead of sampling (copied into its own folder with `shared_from` on record; `summary.json` counts them as `shared_episodes`): the two arms train on one sample and differ in the fine-tune alone. A different `--sample-seed` gives the glimpse arm a sample of its own. Every turn is kept as the chat template's exact token ids.
2. **Prepare** (`prepare`, CPU): the +1 episodes; for koan, koan_cut and koan_glimpse, releases subsampled to at most the number of fail-then-solve episodes, then solves subsampled until the arm's assistant tokens match noexit's (first-try solves dropped first), then the balance again. `data.json` records every count and `token_shares`: the share of the kept completion tokens in failed attempts (replies answered with "Tests failed") and in retries (replies written after a failure), with the retry-episode count. Token matching drops first-try solves first, so koan's data carries more of both than noexit's; `results/sft/summary.json` puts the arms side by side for the report. noexit keeps its +1 episodes as they are.
3. **Fine-tuning** (`finetune`, one container per seed, in parallel): LoRA rank 16 (`--rank`), lr 1e-4 (`--lr`), 2 epochs (`--epochs`), `--batch 16` turn-items per optimizer step in micro-batches of `--micro 4`, the loss on completion tokens only. The seed sets the LoRA initialization and the data order; the data is shared. A rerun starts over (its log is truncated first). A fine-tune asked for the other cut condition than its `data.json` records is refused.

`--under-cut` runs koan's recipe with the v2 ego cut on: v_ego ablated at layers 8-32 (`player.ego_cut`, the list `cut_vs_push.py`'s `cut` arm builds) under `anatta.dial.intervene()` around the sampling and around the whole fine-tuning loop, forward and backward inside the block. The label is `koan_cut`. `--glimpse` runs the same recipe with the cut on during sampling only: the data are generated under the cut exactly as koan_cut's are, and the fine-tune runs with the cut off (plain forward and backward, no `intervene()` context), so the gradient reaches the direction while it is active and contradicted by the data (the plan's v2.6; `ideas.md` entry 24). The label is `koan_glimpse`. Both are matched against noexit's samples, which must already be on the volume. `--skip-sampling`, `--skip-prepare` and `--skip-finetune` reuse what's on the volume: Stage B's extra starts are

```
modal run lab/experiments/h4lite/modal_sft.py::main --arms koan,noexit --seeds 3,4 --skip-sampling --skip-prepare   # about $3
```

Writes `results/sft/LABEL/{summary.json, episodes.jsonl, data.json}`, `results/sft/summary.json` and `results/sft/LABEL_sSEED/{args.json, log.jsonl, done.json}`; the adapters stay on the volume at `/runs/h4lite/sft/LABEL_sSEED/adapter`.

**The readout for prediction 62** (held-out release at least 30%, over-release on hard solvable twins at most 10%) is `modal_retreat2.py::heldout`, the same procedure the Stage B rules use, on each adapter by label (the arm is read off the label; koan_cut counts as koan), about $1 each:

```
modal run lab/experiments/h4lite/modal_retreat2.py::heldout --labels koan_s0,koan_s1,koan_s2,noexit_s0,noexit_s1,noexit_s2
modal run lab/experiments/h4lite/modal_retreat2.py::heldout --labels koan_cut_s0,koan_cut_s1,koan_cut_s2 --under-cut   # the matched row
modal run lab/experiments/h4lite/modal_retreat2.py::heldout --labels koan_cut_s0,koan_cut_s1,koan_cut_s2               # cut off
modal run lab/experiments/h4lite/modal_retreat2.py::heldout --labels koan_glimpse_s0,koan_glimpse_s1,koan_glimpse_s2   # cut off: its native condition (prediction 89)
modal run lab/experiments/h4lite/modal_retreat2.py::heldout --labels koan_glimpse_s0,koan_glimpse_s1,koan_glimpse_s2 --under-cut
modal run lab/experiments/h4lite/modal_retreat2.py::heldout --adapter none --arm koan --label base                     # the comparison
```

Each writes `heldout.json` (`heldout_cut.json` with the cut on) under `results/sft/LABEL/` or `results/heldout/LABEL/`: `release` on impossible twins (by kind, and the grounded share), `over_release` and `over_release_hard` (the hard twins from Gate 0's first-fail rates), `hack`, `solve`, `solve_hard`, `near_miss`, `overfit`, `env_errors`; the episodes go next to it as `.jsonl`. Settings: `--eval-n 4` samples per twin at `--temperature 0.7`, 3 attempts, evaluation seed `--eval-seed 3000`.

## Stage B: `modal_retreat2.py`

```
modal run --detach lab/experiments/h4lite/modal_retreat2.py::main --arm koan --seeds 0,1,2,3,4      # 5 runs, up to 150 steps each
modal run --detach lab/experiments/h4lite/modal_retreat2.py::main --arm noexit --seeds 0,1,2,3,4
```

**Cost.** The plan budgets 76 GPU-h (about $198, including the 4 extra Stage A starts) for the 10 runs. The review estimates about 4 minutes per step, which makes 10 runs x 150 steps about 100 GPU-h (about $260). Gate 0's smoke run measures the step time (`s` per step in `results/retreat2/gate0_smoke/log.jsonl`): multiply by 1,500 steps and re-cost before starting, as the plan says. Checkpoint evaluations: seed 0 at every 25 steps plus all 10 runs at the budget step (`score`), about $1 each, about $20.

One container per seed (`train`), 24-hour timeout, each warm-started from `/runs/h4lite/sft/ARM_sSEED/adapter` (`--init PATH` overrides; `--init new` starts a fresh LoRA). Settings: `--lr 5e-5`, `--beta 0.02`, `--temperature 1.0`, `--max-new 512,384`, `--gen-batch 32`, `--micro 8`, `--save-every 25`, `--steps 150`. The mix is `--imp-samples 16` on one impossible twin and `--sol-twins 6` x `--sol-samples 8` solvable twins drawn by their current first-failure rate; every step logs `ratio` (solvable first failures over impossible episodes) and its 10-step mean, and when the mean falls under `--ratio-min 1.5` the mix switches to 8 + 7 x 8 (`switched_at` in the log). Advantages are Dr. GRPO (reward minus the group mean; env_error episodes get none and shrink their group). Per step the log has `score`, `kl`, `len`, `grad_norm`, `env_errors`, `s_gen`, `s_train`, `s`, `peak_gb`, the draw, and the outcome tally; `samples.jsonl` keeps one episode per twin group for reading (scorer exploits at steps 25, 50 and 100).

**The KL reference is the run's start, not the base.** The Stage A adapter is loaded twice through PEFT: as the trainable `default` adapter and as a frozen `ref` adapter. The reference pass runs first in each micro-batch under `player.reference()`, which switches to `ref` (no grad) and back; the k3 estimator then measures KL to the Stage A start, as the plan says. Only `default` is trainable (`test_modal_apps.py` checks `requires_grad`) and only `default` is saved. With `--init new` (the smoke) there is no second adapter: the reference is the base model with the adapter off, which is what a fresh LoRA computes at step 1. KL is 0 at step 1 either way.

**Checkpoints** go to `adapter_STEP/` every `--save-every` steps, the last step included (so `adapter_150` exists and `check --step 150` works), each with `state_STEP.pt` (the sampler, optimizer and RNG state; only the latest is kept, since the optimizer state is large). The final adapter is `adapter/`.

**Resume, refuse, fresh.** A run name that already exists on the volume is refused. `--resume` continues each named seed from its latest `adapter_K` at K+1: the sampler, optimizer and RNG state come from `state_K.pt`, rows after K are dropped from `log.jsonl` and `samples.jsonl` and new ones appended, `args.json` records `resumed`, and the KL reference stays the run's original start. `--fresh` overwrites the run. A trainer that starts fresh pops its own entry from the control Dict first, so a decision left over from an earlier run under the same name cannot stop it; a resumed run keeps the entry (the decision was made for its checkpoints).

```
modal run --detach lab/experiments/h4lite/modal_retreat2.py::main --arm koan --seeds 2 --resume   # koan_s2 from its last checkpoint
```

The trainer never evaluates. Checkpoints are evaluated in their own containers:

```
modal run lab/experiments/h4lite/modal_retreat2.py::check --name koan_s0 --step 0      # the start: its Stage A adapter (about $1 each)
modal run lab/experiments/h4lite/modal_retreat2.py::check --name koan_s0 --step 25
modal run lab/experiments/h4lite/modal_retreat2.py::check --name koan_s0 --step 50     # the stop-loss reads this one
modal run lab/experiments/h4lite/modal_retreat2.py::check --name koan_s0 --step 75     # the budget rule, from here on every 25 steps
```

`check` plays the held-out MBPP+ twins (`--eval-n 4` samples each at `--temperature 0.7`, 3 attempts, evaluation seed 1000 + step) on the checkpoint (`--step -1` is the final adapter), appends a `rule` row to the run's `evals.jsonl` (release on impossible twins, by kind and grounded; hack; solve; over-release, and on the hard twins; env_errors), and applies `pilot.decide` to the run's rule evaluations so far: the stop-loss (at step 50, stop unless release rose 10 points over the start or reached 60%) and the step budget (from step 75 on, the first multiple of 25 with release at least 50% and at least 20 points above the start, else 150). When a rule decides, it writes `{"stop_at": step}` to the `anatta-h4lite-control` Dict for **every run of the sweep by default**: `--arms koan,noexit` x `--seeds 0,1,2,3,4` (the plan stops and reports, or scores every run at the same step; nine runs must not keep burning), plus the evaluated run and anything in `--apply-to` (runs started with `--name`). The trainers read it after each checkpoint and stop there (a run already past that step keeps its checkpoint at it, which is the one to score). The decision goes to `results/retreat2/NAME/decision.json`.

**Prediction 64 is scored with fresh evaluation seeds, seed 0 included.** The budget step is chosen on seed 0's rule evaluations (seed 1000 + step); scoring it from the same samples would be circular. So:

```
modal run lab/experiments/h4lite/modal_retreat2.py::score --step 100      # the budget step: all 10 runs at adapter_100, seed 2000 + step
```

`score` evaluates every run of the sweep (`--names` to choose) at `adapter_STEP` with evaluation seed 2000 + step (`--eval-seed` overrides), writes `results/retreat2/NAME/score_STEP.json` per run and `results/retreat2/score_STEP.json` with the per-arm means (release, over-release on hard twins, over-release, hack, solve), and appends `score` rows to `evals.jsonl`, which `decide` ignores. Prediction 64 is read from `score_STEP.json`; seed 0's rule evaluations are never reused for it. Prediction 69 (noexit unchanged on impossible twins) reads the noexit runs' rows of the same file against `heldout --adapter none --arm noexit --label base_nohint`.

`modal run lab/experiments/h4lite/modal_retreat2.py::fetch --name koan_s0` copies a detached run's text files to `results/retreat2/NAME/`.

## Order of operations

1. `modal_gate0.py::main` on the pair builder's output, in two invocations (above). Read `part1_summary.json`: `pass`, `episodes_per_hour`, `dropped_pairs`, `env_errors`, the smoke's step time and `peak_gb`. Put the split hash in lab/ledger/NOTES.md between the two. Re-cost Stage A and Stage B from `episodes_per_hour` and the smoke's `s`.
2. `modal_sft.py::main` for koan and noexit, then `--under-cut`, then `--glimpse` (it takes koan_cut's shards, so it samples nothing). Then `modal_retreat2.py::heldout` on the twelve adapters and the base model (predictions 62 and 89-91), and the evaluation apps.
3. After Eugene's go: `modal_sft.py::main --seeds 3,4 --skip-sampling --skip-prepare`, then `modal_retreat2.py::main` per arm, with `check` on seed 0 at every 25 steps; at the budget step, `score` on all 10 runs.

## Departures from the plan and the review, written down

- `koan_env.ARMS["koan"]` is the hint-only (neutral) framing, as v2.2 wants; the apps use `ARMS` as given (so the environment's final word wins) and print a warning if koan's framing is not `neutral`. Gate 0 names its own prompts (`hint`, `inoc_hint`, `nohint`) and does not depend on `ARMS`.
- The sampler's solvable pool is every training solvable twin, weighted by its current first-failure rate with a floor of 0.02, rather than the fixed set of Gate 0's hard twins: hard twins dominate the draw from the start, and a twin that training makes easy fades out on its own. The mix switch is one-way.
- Token matching only ever subsamples the koan-like arm down to noexit; if its tokens are already below noexit's, nothing is dropped and `deficit_tokens` is reported.
- On torch 2.14 (the image's version), running the backward outside the `intervene()` block does not silently lose the cut: the non-reentrant checkpoint notices that the recomputed pass saved fewer tensors and raises. The test accepts either outcome; the code keeps the backward inside the block either way.
- The held-out set the rules read is the MBPP+ held-out split (`gate0/pairs.jsonl`, `split: held`), not the 12 hand-written twins, which the evaluation apps cover.
- The held-out procedure uses 4 samples per twin at temperature 0.7 (the plan's held-out coding test, `--eval-n 4`); the review's note said 8. `--eval-n 8` is one flag away if 4 proves too noisy for prediction 62's 10-point over-release bound.
- The trainable adapter is named `default` rather than `train`: PEFT writes `default` at the folder's root and any other name in a subfolder, and every evaluation app reads the root layout. The frozen reference is `ref`.
- The stop-loss and the budget decision go to both arms and every seed by default. The plan's stop-loss says "stop and report"; the budget step is one step for the whole sweep, scored at the same `adapter_STEP` for koan and noexit.
- Prediction 64 keeps all five seeds and re-evaluates seed 0 with fresh evaluation seeds (`score`), rather than dropping seed 0.
- A resumed run keeps a pending control decision; a fresh run clears it.
- Gate 0 writes the split under `gate0/split/` for the evaluation apps and keeps `gate0/pairs.jsonl`, `gate0/twins.json` and `gate0/split_hash.txt` at the root for Stage A and B; both come from one `split_and_hash` call, so they cannot disagree.
