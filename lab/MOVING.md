# Moving the project to another Mac

Written 29 Oct 2026 by Claude (Fable) for the move to the Mac mini. The easy path is to keep the same user name
(`eugene`) and the same paths (`~/prj/align/...`): every script, README command, memory file and test header uses
them. A different path works but needs a pass over the absolute paths listed at the end.

## What lives where today

| Thing | Where | In git? | How it gets to the new machine |
|---|---|---|---|
| Public repo (harness, docs, paper) | `~/prj/align/anatta`, branch `main`, remote `origin` = ejhong/anatta | yes | `git clone`; the main tree has uncommitted `src/anatta/koans/{base,impossible,pushback}.py` that Eugene is keeping, so copy or commit those first |
| Private lab | `~/prj/align/anatta-lab`, a worktree of the same repo on branch `lab`, remote `private` = ejhong/anatta-lab | yes | in the clone: `git remote add private https://github.com/ejhong/anatta-lab.git && git fetch private && git worktree add ../anatta-lab lab` |
| The dev checkout the laptop runs use | `~/prj/align/anatta-dev`, branch `dev/ministral` at ee89587 | yes | `git worktree add ../anatta-dev dev/ministral` (it is the `PYTHONPATH` of every laptop run and test) |
| Extracted directions | `~/prj/align/anatta/directions/` (13 MB, `*.pt`) | no (ignored) | copy the folder |
| Laptop dials | `~/prj/align/anatta/results/2026-10-28-viability/dials_m8` | no (ignored) | copy the folder |
| Lab results | `anatta-lab/lab/experiments/{h4lite,recognition}/results/` (about 380 MB) and `logs/` | no (ignored) | copy the folders; the adapters themselves are on the Modal volume, not here |
| Evaluation datasets | `anatta-lab/lab/data/<name>/raw/` | no (ignored); the manifest, hashes and item lists are | `python -I lab/data/fetch.py` after cloning |
| Model weights | `~/.cache/huggingface/hub` (49 GB, mostly Ministral and the EM sets) | no | copy the cache or let it re-download (the laptop fishery needs the 16 GB Ministral checkpoint) |
| Modal adapters, checkpoints, runs | Modal volumes (cloud) | no | nothing to move |
| Claude Code memory and session transcripts | `~/.claude/projects/-Users-eugene-prj-align/` (the folder name encodes the project path) | no | copy the folder to the same path on the new machine; with a different project path, Claude Code starts with empty memory. Sessions started in `anatta/` (as its CLAUDE.md asks since 3 Nov) use `-Users-eugene-prj-align-anatta/` instead |
| The pre-move backup | `~/prj/align/.premove-backup-2026-10-29/` (126 MB) | no | do not move it; delete it once the new machine is verified (Eugene's call) |
| The claim checklist and the site builder | `~/prj/align/claim/`, `~/prj/align/site/build.py` | separate | copy |

## Tooling to recreate

- `~/prj/align/anatta/.venv`: Python 3.12 from `anatta/pyproject.toml` (torch with MPS for the laptop runs).
  Since 7 Nov it is pinned by `anatta/uv.lock`: `uv venv --python 3.12 .venv && uv sync --extra dev --extra mirror
  --extra evals --extra train`. Leave out the `modal` extra: Modal runs from the Modal venv (`tools/modal.sh`), and
  with `modal` installed in `.venv` one runner test fails, because `experiments/run_r1b.py` then builds a real
  `modal.App` around an app the tests' stand-in left behind (found on the 7 Nov move).
- `~/prj/align/anatta-lab/lab/experiments/h4lite/.venv`: Python 3.14, no spec file yet; it is only used for CPU
  tests that also run under the matlab venv, so it can be dropped.
- `~/prj/matlab/.venv`: the Modal CLI (modal 1.6.2) and huggingface_hub; recreate with `pip install modal
  huggingface_hub`, then `modal setup` (Eugene, in a browser; a credential Claude never handles). On 7 Nov:
  `uv venv --python 3.12 ~/prj/matlab/.venv && uv pip install --python ~/prj/matlab/.venv/bin/python modal==1.6.2
  huggingface_hub`; the copied `~/.modal.toml` still worked, so no new `modal setup` was needed.
- Command-line tools: the machine-wide `~/Brewfile` (one list for every project on the machine, kept outside
  this repo; started 8 Nov with poppler and pandoc for reading papers). Restore with `brew bundle --file ~/Brewfile
  --no-upgrade`; `tools/doctor.py` warns when a listed tool is missing. Copy the file to the new Mac first.
- Shell environment: `OPENROUTER_API_KEY` (the judges), `HF_TOKEN` (faster downloads; optional). Eugene copies these;
  Claude only ever checks that the names are set.
- Chrome with the Claude in Chrome extension, if the web chores continue there.

## Before pulling the plug on this Mac

Every `modal run` in flight (Stage A, the EM judge, the positive-control rounds) is driven by a client process on
this machine; if the machine sleeps or shuts down, those apps stop. Either wait for them to finish, or start future
long runs with `modal run --detach` so they survive the move.

## After the move

1. `PYTHONPATH=~/prj/align/anatta-dev/src ~/prj/matlab/.venv/bin/python lab/experiments/h4lite/test_pilot.py` (and the
   other `test_*.py`): the CPU checks.
2. `modal run lab/experiments/h4lite/modal_fishery_eval.py::table` or any `::table` entrypoint: no GPU, proves the
   CLI and the volume are reachable.
3. One laptop fishery game: `run_dig2.sh`'s command with `--seeds 1`, proves MPS, the weights and the directions.
4. `python -I lab/data/fetch.py`: the datasets, verified by hash.

## Absolute paths to fix if the path changes

`lab/experiments/recognition/scripts/run_*.sh` (`cd ~/prj/align/anatta`, `PYTHONPATH`), `lab/experiments/h4lite/
modal_common.py` (`ALIGN`), the test headers (`PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src`), `lab/experiments/
h4lite/README.md` and `EVAL.md` (commands), `lab/README.md`, and the Claude Code memory files under
`~/.claude/projects/-Users-eugene-prj-align/memory/` (paths in `anatta-project.md`, `modal-gpu.md`,
`recognition-ideas-private.md`, `keep-notebook.md`).
