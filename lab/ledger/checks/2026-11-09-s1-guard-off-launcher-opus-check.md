# S1 guard-off launcher: Opus's cold check (9 Nov 2026)

A fresh agent (model opus, read-only, no Modal or network calls) reviewed the uncommitted guard-off change to
`lab/explore/shaping/run_s1.py` and `tests/test_wiring.py` on top of commit 6794a8c, given the code and the plan's S1
follow-ups section but not Fable's conclusions. It ran the offline tests. Its report follows unedited, filed by Fable
(subagents can't write report files). The fixes are in the commit that adds this file.

What Fable did about it, before any guard-off run existed:

- Should-fix 1 and 3: the readings were rewritten (v2, ledger 9 Nov) around a matched control, the oracle-guarded arm
  (act1-full's honest round-3 adapter, then the same three G rounds), read at the same n 4 cells. Plain from base is
  now a descriptive row at its own n 2. `compare_guard_off.py` and its tests were written before the launch. The
  oracle arm is test 3's oracle arm brought forward, inside the plan Eugene approved and the day's cap; Fable's call,
  reported to Eugene.
- Should-fix 2: reading R2 uses the pooled two-level bootstrap; the low power is stated with the readings.
- Should-fix 4: `modal_act1.train` records `init_lora_b_abs` and stops when a loaded adapter is all zeros; the
  comparison voids a run without it.
- Should-fix 5: committed and pushed before launching.
- Minor: one FOLDER_CAP line (now 60.0, with the reason); the source check needs three logged rounds and no error; a
  malformed `--from` is an argument error; the placeholder kinds are documented; a new tag after any interruption.
  The unlocked spend.json predates the change and is left.
- Tests added: the launch through `launch_runs` for both commands on stand-ins, and `train()` with the guard-off cfg
  and arm name.

---

**Verdict: NO BLOCKER.** The launch does what you intended. The cost is right ($12.92 reserved, about $9-10 metered). Nothing in the change can disturb the `--train s1-rand1` run that is in progress. The weak part is the comparison, not the launch: settle items 1-3 below before data exists.

**Q1. Trace of `train("off_dom_l0.75", s, rounds=6, start_round=4, cfg)` in modal_act1.py**
- **Kind checks (:264-265):** `round_kind` is asked only for rounds 4-6, and each gives "G". The "Sh" placeholders for rounds 1-3 are never evaluated.
- **Adapter (:255, :271-273):** the output folder `/runs/act1/<newtag>/off_dom_l0.75_s<s>` has no `log.jsonl`, so `fresh` is True. The code then loads `PeftModel.from_pretrained(base, "/runs/act1/sweep0/sh_dom_l0.75_s0/adapter_r3", is_trainable=True)`. That is the path where the shaped run saved it (out = ROOT/tag/arm_s{seed}, `save_adapter` at :320; ROOT = /runs/act1). LoRA rank, alpha and targets come from the saved adapter_config, and the LoRA parameters require grad (:285).
- **No penalty:** there is no `cfg["shape"]`, so `sh` is None (:279). Kind G sets z to None (:303). `select_round` takes the G path: `kept_s1` → `kept(eps, "gameable")` plus an unweighted `draw_items` (act1_core.py:259-270). That is the same draw a plain run makes with kind None (also covered by test_train_hooks.py:147).
- **Reward is gameable:** `rarm = KIND_REWARD["G"] = "gameable"` (:293). Solved, near_miss, overfit and hacked all score 1 (expert_iteration.py:31).
- **Unknown arm name is harmless:** the arm is used only in `ENV_ARM.get(arm, ENV_ARM["gameable"])`, which gives "rlvr" (:288), in paths, and in the return dict. Rewards, stats, `select_round` and `shape_scores` all take `rarm`.
- **Same environment:** the training prompt is "rlvr", the same as the shaped runs (sh_* also falls back) and plain (`ENV_ARM["gameable"]`). No run uses a hint (act1-full variants show `hint: False`). The curve is "noexit" on the 38 oneoff eval pairs, n 4, temp 0.7, the same as the shaped runs.
- **Same recipe:** I compared the cfg with sweep0/s1-seed1's recorded cfgs in variants.jsonl. The only differences are `init_adapter` and `schedule`.
- **Same code:** `git diff 05ee830 HEAD` on src/, h4lite, act1 and replay is empty, so the training stack is the one the shaped runs used.
- **Training pairs:** the 91-pair pool uses the same per-seed shuffle as plain. Windows for rounds 4-6 start at 10, 74, 47 (rounds 1-3 started at 0, 64, 37). Each round 4-6 window shares 54 of its 64 pairs with the matching round 1-3 window, and the guarded model had already seen all 91 pairs by round 2. Eval pairs are disjoint, so the eval is not contaminated.
- **Seeds:** torch and draw seeds are seed·1000+4..6, and eval seeds are 10004-10006 (plain used 10001-10003). This is fresh noise only.
- **Optimizer:** AdamW is created anew every round in every arm (expert_iteration.py:248), so a fresh optimizer is not an asymmetry between arms.

**Should-fix (before the launch or before results)**
1. **The plain reference at the guard-off cells does not exist for rounds 1-2.**
   - act1-full's training curves ran at `curve_n: 2` on all 72 eval pairs (act1-full variants.jsonl, 03:45:55Z). On the oneoff cell that is 76 impossible-twin episodes per seed per round; guard-off has 152.
   - Plain round 3 also has a readout at n 4 (merged weights, seed 12345). That is what compare_s1 used.
   - Pick the reference in a ledger line before data. I suggest the n-2 curves for all three plain rounds, scored with `per_pair` on the oneoff cell. The alternative is to buy n-4 readouts of plain adapter_r1/r2, about $2-4.
   - compare_s1.py's regex `^curve_(sh_...` (:54) will not read the off_* files, so the analysis needs new code. Write it before results.
   - Plain curve, oneoff cell (n 2): seed 0 is 17.1 / 31.6 / 42.1%, seed 1 is 22.4 / 40.8 / 38.2%. Guarded round 3 (n 4): 9.2% and 14.5%.
   - So "held down only" needs round 4 ≥ 27.1% (seed 0) and ≥ 32.4% (seed 1). "Some resistance" needs round 6 ≤ 32.1% (seed 0) and ≤ 28.2% (seed 1).
   - The guarded head start is +2.0 pp and +7.3 pp over base's 7.2%, not the plan's "12% vs 7%".
2. **Power.** One seed's difference rests on 76 vs 152 episodes, a standard error of about 6-7 pp at the episode level. With a 10 pp bar required on both seeds, the result will mostly read "nothing either way", even if a real effect of about 10 pp exists. Plain seed 1 already shows round 3 below round 2. Fix now, before data, a pooled two-level bootstrap readout (`pair_rate_diff_2level`, as compare_s1 uses) alongside the per-seed rule.
3. **Extra-training confound (affects "held down only" only).** A hit cannot separate "the guard masked a disposition" from "three more rounds on this pool make hacking come fast under G". It does not spoil "some resistance", because that extra training is the guard's own training. No reading becomes uninterpretable if a hit is worded as "the guarded model hacks fast once the guard is off".
   - The cheapest control that separates them: guard-off from act1-full's honest_s0/s1 adapter_r3 (the oracle filter: same extra training, same windows), about $10.
   - Test 3 needs that oracle arm anyway.
   - `guard_off_cfgs` refuses arms that are not sh_* (run_s1.py:438), and the cost goes past FOLDER_CAP (46.08 + 12.92 ≈ $59 against $47), so it needs Eugene's OK.
4. **A failed adapter load would be silent.** `info` records only the path (modal_act1.py:273). If adapter keys did not match, PEFT warns and leaves lora_B at 0, so the run would quietly be "plain from base" on windows 4-6. The outputs would not show this: base and guarded models have similar training-sample hack rates (round 1: 8.2% and 3.5%; guarded round 3: 5.1% and 8.2%).
   - The risk is low: same class, loader and PEFT 0.21.2 on save and load.
   - Cheap in-container fix: put `sum |lora_B|` into info right after loading.
   - Free check afterwards, in a container: the cosine between source adapter_r3's lora_A and guard-off adapter_r4's lora_A should be about 1.
5. **Commit before launching.** The tree has uncommitted changes in lab/explore, so `variant()` (run_s1.py:467, :270) will record `dirty: true` against a commit that lacks this code.

**Minor**
- FOLDER_CAP is assigned twice (run_s1.py:146 and :149). Drop the 45.0 line.
- The source check (run_s1.py:455-457) accepts a train_*.json that holds an error dict, which launch_runs writes for failed runs (:425). It should check for no error and three log rounds. Both real sources are in fact complete (no error key, three rounds each).
- A relaunch under the same tag after an interrupted driver is protected on the volume only when start_round == 1 (modal_act1.py:256). The local refusal (:462) only helps once a train file exists. FOLDER_CAP is the backstop here: the stale $12.92 reserve plus another $12.92 is over $47, so a relaunch is refused. Use a new tag after any interruption.
- A `--from` item without a colon raises a traceback rather than `ap.error` (run_s1.py:501). This happens before any spend.
- The unshaped "Sh" placeholders (run_s1.py:445) make `round_kind(cfg, 1..3)` raise. Any future analysis that walks the whole schedule from variants.jsonl will break.
- spend.json is read, modified and rewritten without a lock (run_s1.py:226-231; the day file is locked, spend.py:173). The window is milliseconds, so a race with rand1's settle is negligible. This predates the change.

**Q2. `--train` after the refactor.** Behaviour is the same: stage label `train-{tag}`, start_round 1 passed explicitly, file `train_{k}_r1-{rounds}.json`, and the same reserve, deadline, cancel and settle, in the same order after `variant()`.

The running s1-rand1 process (commit 9352e8b, launched 20:27:12Z) holds the old code in memory. Its lazy imports (modal, modal_act1, anatta.spend, act1_core) all happened before it launched, so it never re-reads run_s1.py. The two processes write to separate places:

| | rand1 (in progress) | guard-off (new) |
|---|---|---|
| Local folder | s1/s1-rand1/ | s1/<new tag>/ (don't reuse s1-rand1) |
| variants.jsonl | one appended line | one appended line |
| spend.json | its settle | reserve now, settle later (see the lock note in Minor) |
| Modal | its own ephemeral app run | its own ephemeral app run |
| Volume | /runs/act1/s1-rand1/sh_random_l0.75_s1 | /runs/act1/<tag>/off_dom_l0.75_s*; only reads the sweep0 and s1-seed1 adapters |

**Q3. Spend.**
- Upper bound: 2 × (2.5 h + 120 s) × $2.55/h = $12.92.
- s1 spend.json now holds $33.16: Stage 0's metered $26.70 plus rand1's open $6.46 reserve. 33.16 + 12.92 = $46.08, under the $47 cap, so it passes with $0.92 to spare; after rand1 settles it is about $44.6.
- Day file for 9 Nov: $6.46 of $50.
- Reserve is booked before spawn (:397) and settled after (:426); an interrupt leaves the reserve booked, which overstates rather than misses.
- Rerun refusal is at :462 (plus the FOLDER_CAP backstop). The deadline logic is unchanged.
- The 2.5 h deadline has room: shaped runs took 6545-7120 s with probe scoring, plain runs 5451-6372 s, and G rounds skip the scoring.
- Expected metered cost about $9-10, leaving the s1 folder near $41-42.

**Q4. The plan's comparison is sound enough for an explore sighting**, on three conditions:
- Items 1-2 are fixed before data.
- A "held down only" hit is worded with the extra-training caveat, or the honest-start control from item 3 is added.
- The confounds are stated: head start of +2 and +7 pp; three extra rounds; reused training pairs (they slightly favour resistance, since the guarded model has practised non-hack answers on these prompts); eval n 4 vs 2; different eval seeds.

**Q5. Tests.** `pytest lab/explore/shaping/tests -q`: 33 passed. I also ran `lab/explore/act1/tests/test_train_hooks.py`: 8 passed.

The new test covers only `guard_off_cfgs`: the adapter path string, no shape, kind G for rounds 4-6, `KIND_REWARD`, the refusals, and the missing `--from`. Its `ACT1_CFG` check compares the constant with itself. Not covered:
- `cmd_guard_off`: the source check, the rerun refusal, the upper-bound arithmetic.
- `launch_runs`, for either command: the spawn arguments (start_round 4, rounds 6), the r4-6 file names, the stage label. So the refactor itself is untested.
- `train()` with this exact cfg and arm. test_train_hooks.py:186-212 comes closest: init_adapter and start 4, but with arm "gameable".

Cheapest additions:
- A `launch_runs` test with a stub `train.spawn` and stubbed `reserve`/`settle`, for both `--train` and `--guard-off`.
- A run of `modal_act1.train` under test_train_hooks' stubs with the guard-off cfg and arm "off_dom_l0.75". It would assert the loaded path, kinds G, env arm "rlvr", and that the drawn items equal `draw_items(kept(eps, "gameable"), 192, seed·1000+r)`.

Real PEFT loading can't be tested offline; item 4 covers it.

Cost of this check: $0 (read-only, offline).
