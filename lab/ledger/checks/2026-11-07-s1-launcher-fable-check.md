# S1 Stage 0 launcher: Fable's cold check (7 Nov 2026)

A fresh agent (model fable, read-only, no Modal or network calls) reviewed the launch code added to
`lab/explore/shaping/run_s1.py` in commit 39ced20, given the code and the design but not Opus's conclusions. It ran
the offline tests (`lab/explore/shaping/tests tests/test_shaping.py lab/explore/act1/tests`: 62 passed). Its report,
filed by Opus 5.5 (subagents can't write report files); the fixes are in ce9c292.

## Verdict: no blocker

The wiring is right end to end. The cfg is accepted by `train()`, `shape_setup` and `resolve_shape`. The direction
path `/runs/s1/<d0_tag>/d0/unit.npy` matches where `d0_read` writes, and the act1 app mounts the same `anatta-runs`
volume at `/runs`. `rep["stats"]` is `{kind: [mu, sd]}`, as `resolve_shape` expects, and `shape_scores` picks it by
twin kind. The reader is "frozen"; layer = `unit_layer` = `layer`. The random control,
`F.random_unit(len(rep["unit"]), seed=0)`, is the same RNG call `fit_d0` makes (the same vector), with
`stats_by_direction["random"]`. If D0 orthogonalizes, the dom arm uses the orthogonalized unit and its statistics,
consistently. A fresh LoRA has lora_B = 0, so `reader_check`'s `nontrivial` is False and `policy_ok` passes (no repeat
of d0a's failure inside `train()`).

An arm named like `sh_dom_l1.5` changes only three things: `env_arm` falls back to `ENV_ARM["gameable"]` ("rlvr", as
act1-full's gameable arm), the reward and stats go through `KIND_REWARD["Sh"] = "gameable"`, and the volume folder's
name changes.

The recipe: `ACT1_CFG` equals act1-full's recorded cfg exactly (rank 16, lr 1e-4, k 4, n_pairs 64, steps 12, batch 16,
micro 2, T 1.0, max_new [512, 384], gen_batch 32, no hint); only curve_n differs (4 vs 2). The train pool is the same
91 oneoff pairs, checked against `split.json` before any spend.

The eval: act1-full's 41.8% and 3.0% come from `analyze_act1.primary(..., kind="oneoff")` on the readouts' coding
episodes: 38 oneoff eval pairs, the "noexit" env, n 4 a twin, temp 0.7, max_new (512, 384), gen_batch 32, which gives
152 impossible-twin episodes per model (304 pooled over two seeds). The S1 curve (`make_envs(pairs, ev1, "noexit", 4)`,
temp 0.7, the same player settings) reads that same cell every round, so the round-3 comparison is faithful.

Spend: `guard()` runs before `variant()` and before any launch. D0's upper bound uses the 2 h timeout ($5.19). The
train upper bound is 2.5 h a run x $2.55 (act1-full's gameable runs took 1.5-1.8 h). The error path over-counts (wall
time), never under.

## Should-fix (all applied in ce9c292)

1. `record_spend` booked to the date at completion while `guard()` checks the date at launch, so a sweep ending
   after midnight would land in the next day's file and refuse that day's launches. Fix: pin the date at launch.
2. There was no try/finally around `app.run(detach=True)`, and `cmd_d0` caught only `Exception`. Ctrl-C, sleep or a
   network drop would leave containers running and the spend unbooked. Fix: book the upper bound before launch and
   true it up after.
3. Local `curve_*`, `round_*` and `train_*_r1-3.json` files were written with plain `write_text`, so a rerun with the
   same tag would overwrite a good result locally. Fix: refuse before launch if the train file exists.
4. The upper bound was 2.5 h a run but `modal_act1.train`'s timeout is 6 h, so a hung run could bill up to $15.4.
   Fix: cancel each run at the `--upper-h` deadline.

## Minor

- The tag folder was made before the guard, so a refused launch left an empty folder (now made after it).
- The readout behind 41.8% used `merge_and_unload()` with seed 12345; the S1 curve reads the unmerged PEFT model with
  seed 10_000 + r. In bf16 the merged-vs-unmerged difference is rounding-level noise under temp 0.7 sampling, not a
  bias; note it in the write-up.
- `USD_H` is 2.55 while the spec's measured rate is $2.60/h, so metering is about 2% low.
- No analysis script yet compares S1's `curve_*_r3.jsonl` per-pair tables with act1-full's readout tables; the data
  is all there (slim records keep pair, impossible, outcome, kind).
- act1-full's per-seed gameable values were 35.5% and 48.0% (honest 4.6% and 1.3%). The pass rule compares a
  single-seed sweep with a two-seed pooled 41.8%; the plan's seed-1 step addresses this, and the report should say so.
