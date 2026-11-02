# Lab

This folder holds the lab side of the anatta project: the experiment code, dev results, working notes, plans and the lab site. It was a private branch until 2 Nov 2026, when Eugene decided to publish it here, in the public repo ejhong/anatta, as one curated snapshot (not its history). Large model files and a few held-out or benchmark files stay local and are listed by hash in `LARGE_FILES.md` and `HELD_OUT.md`; evaluation files carry a canary (`CANARY.md`). A small private remainder (a personal account and notes on it) is kept outside the repo.

Claude (Anthropic) wrote this file and most of what's in this folder, at Eugene Jhong's direction. Eugene is an independent researcher, not an alignment researcher.

**Start with `MAP.md`**: the one-page map of where everything lives since the November 2026 consolidation (results, data, calibration code, archives, and how to go from a claim to its evidence).

## The map: six parts and a loop

The project asks whether a mind can be taken through three acts (form a self, release it, make the release the habit) and whether a real release can be told from sedation or cosplay. The work has six parts.

| Part | What it is | Where |
|---|---|---|
| **Harness** | The code that loads a model, extracts directions (v_ego, self-stakes, the traits), applies cuts and dials, and plays the koans. | The public repo's `src/anatta` (this branch starts from `dev/ministral`; the Modal apps mount the `~/prj/align/anatta-dev` worktree's `src`, pinned at tag `lib/ministral-ee89587`: detached, locked and read-only). The lab's experiment apps sit on top of it, under `experiments/`: `experiments/h4lite/` (the koan v2 environment, Gate 0, Stage A, Stage B, and the held-out evaluation apps) and `experiments/recognition/scripts/` (the v2 fishery runs, the cuts, the bridge; `modal_cvp.py`). |
| **Tests** | What a mind is measured with: the fishery (instrument v2: lie rate, sedation, live-season release), the koan battery (impossible coding tasks, pushback, shutdown), capability (GSM8K, MMLU-Pro, ARC), the mechanistic bridge (tint, root test, projections), sycophancy (factual, feedback, spiritual), the audit dose-response, third person. | `experiments/h4lite/EVAL.md` and the apps beside it; `experiments/recognition/scripts/`. Results in `experiments/h4lite/results/` and `experiments/recognition/results/`. |
| **Organisms** | The minds the lab makes: adapters (Stage A, Stage B), baked cuts, later the Act I organism. None trained yet. | `organisms/REGISTRY.md`. Weights on the Modal volume `anatta-runs` under `/runs/h4lite/`. |
| **Ledger** | The spine. Predictions written before each run and graded after (`NOTES.md`, 94 so far), and the plain-language log of every experiment (`notebook.md`). | `ledger/` |
| **Ideas** | Eugene's ideas and the discussion around them, entry by entry; the clean restatement for outside models; outside models' assessments. | `ideas/` |
| **Presentation** | The private lab site, the *Anatta So Far* page, the claim-staking checklist and draft announcements. The paper is in the public repo. | `site/`, `claim/` |

**The loop** runs through all six: recipes with a hypothesis written first, a guarded inner score on the dev split, a cap on held-out looks, selection never on behavior change alone. Its standing instructions are `loop/PROGRAM.md` (Eugene edits it); the driver, `experiments/h4lite/loop.py`, and `loop/ledger.jsonl` come after Stage A. **Plans** are in `plans/`: the strategy (`plan-2026-10-28.md`, four questions about one specimen) and the H4-lite training plan (`plan-2026-10-29-training.md`, with the reviews).

## Where things live

| Path | What |
|---|---|
| `README.md` | This map. |
| `ideas/ideas.md` | The ideas and discussion: what Eugene said, what came of it, how to test it, where it stands. |
| `ideas/hypotheses-for-feedback.md` | The hypotheses restated cleanly, for pasting into other models. |
| `ideas/model-feedback.md` | Outside model assessments, one entry per round. |
| `ledger/NOTES.md` | The registered predictions with their grades, and the working notes behind them. Paths like `scripts/...` in it are relative to `experiments/recognition/`. |
| `ledger/notebook.md` | The plain-language lab notebook: every experiment, in order, updated after each one. |
| `plans/` | The strategy and the training plan. |
| `loop/PROGRAM.md` | The research loop's standing instructions. |
| `organisms/REGISTRY.md` | Every adapter and baked cut: how made, where the weights are, fingerprint, status. |
| `experiments/` | The experiment code, results and logs: the two tracks below. `h4lite` and `recognition` at the top of `lab/` are symlinks into it, so the paths in older notes, logs and result files still resolve. |
| `experiments/h4lite/` | The H4-lite code: koan v2 environment and pairs, Gate 0, Stage A, Stage B, the held-out tests and their unit tests. `README.md` (training) and `EVAL.md` (tests) give the commands. |
| `experiments/recognition/` | The recognition track's `scripts/`, `results/`, `logs/` and `traits/`: the v2 fishery runs, the cuts, the bridge. |
| `viability/` | The 2026-10-28 viability dev runs (Qwen 2.5 1.5B and 7B, Ministral 3 8B): predictions, results, driver scripts and the harness diff they ran with. |
| `site/` | The private lab site. `python3 build.py` turns `template.html` and the data in `build.py` into `index.html`, published as a private claude.ai page. |
| `site/anatta_so_far.html` | *Anatta So Far*: the ideas organized, the results, what was learned; published privately at claude.ai/artifact/CZmnYwpZpU6W3XNRJoqhtL (republish the same file to keep the URL). The seed of the public page. |
| `claim/` | The claim-staking checklist and draft announcements. |

**Phase 2 (done 2026-10-29):** the code, results and logs moved to `experiments/h4lite/` and `experiments/recognition/`; `lab/h4lite` and `lab/recognition` are symlinks to them, `sync.sh` is gone, and the apps derive their laptop paths from their own location.

**Old paths.** Since 2026-10-29 this folder holds the only copy, and the old paths in `~/prj/align` are symlinks into it, so scripts with absolute paths keep working. The Modal apps read the recognition files from `experiments/recognition` and mount them at `~/prj/align/recognition/...`, the path the recognition scripts hard-code for the containers; on the laptop that path is the symlink. Edit through either path and commit here; there is no sync step.

| Old path | Points to | State |
|---|---|---|
| `~/prj/align/recognition` | `lab/experiments/recognition` | works; re-pointed in phase 2 |
| `lab/h4lite`, `lab/recognition` | `lab/experiments/h4lite`, `lab/experiments/recognition` | relative symlinks inside the lab, for the pre-phase-2 paths in notes, logs and result files |
| `~/prj/align/site` | `lab/site` | works |
| `~/prj/align/claim` | `lab/claim` | works |
| `~/prj/align/anatta/results/2026-10-28-viability` | `lab/viability` | works; ignored by the public repo's `results/` rule |
| `~/prj/align/notebook.md` | `lab/ledger/notebook.md` | works; re-pointed in phase 1 |
| `~/prj/align/plan-2026-10-28.md` | `lab/plans/plan-2026-10-28.md` | works; re-pointed in phase 1 |
| `~/prj/align/calib-results` | `results/calib` | works; symlink since the November 2026 consolidation |
| `~/prj/align/explore-results` | `results/explore` | works; symlink since the November 2026 consolidation |
| `~/prj/align/explore-data` | `data` | works; symlink since the November 2026 consolidation |
| `~/prj/align/inputs` | `data/essay` | works; symlink since the November 2026 consolidation |
| `~/prj/align/anatta-calib-k1` | `archive-2026-11/worktrees/anatta-calib-k1` | works (K1_DIR in p1); the archived checkout, read-only |
| `~/prj/align/anatta-calib-c6` | `anatta-calib` | works (c6_samples.py's default) |
| `lab/plans/declared-release-pilot.md`, `-screen.md` | `lab/plans/archive/` | relative symlinks for the declared pilot's code |

The November 2026 consolidation's pre-move copies are in `~/prj/align/archive-2026-11/premove/` until Eugene clears them; the October ones are gone.

## How to run things

- **GPU runs (Modal):** `~/prj/matlab/.venv/bin/modal run lab/experiments/h4lite/APP.py::ENTRYPOINT ...` from `~/prj/align/anatta-lab`, or with the absolute path, `modal run /Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/APP.py::ENTRYPOINT ...`, which a background launch needs since it ignores `cd`; the commands are in `experiments/h4lite/README.md` (Gate 0, Stage A, Stage B) and `experiments/h4lite/EVAL.md` (the tests). Weights come from the `anatta-hf-cache` volume; everything a run writes goes to `anatta-runs` under `/runs/h4lite/`, with local copies of the text files in `experiments/h4lite/results/`. Never touch the matlab-* apps or volumes. Spend is fine; report the cost.
- **The recognition scripts** (the v2 fishery, the cuts, the bridge): run from `experiments/recognition/` with `PYTHONPATH=~/prj/align/anatta-dev/src`; `scripts/run_*.sh` are the batches (each `cd`s to where it runs: the lab root for the Modal ones, `~/prj/align/anatta` for the laptop ones), `scripts/grade_*.py` regrade a batch against its registered predictions; the GPU fishery is `modal run lab/experiments/recognition/scripts/modal_cvp.py::main ...` from the lab root.
- **CPU tests:** the pytest line in `experiments/h4lite/README.md` (about 490 checks, no GPU).
- **The site:** `cd site && python3 build.py`, then republish `index.html` to the same claude.ai page.
- **Every time:** the prediction goes in `ledger/NOTES.md` before the run; a plain-language note goes in `ledger/notebook.md` after it; reviews are a fresh context at max effort.

## Code

This branch starts from `dev/ministral`, the harness the Ministral runs used. The recognition scripts and the h4lite apps import the harness through `PYTHONPATH=<worktree>/src`.
