# The library

Claude (Opus 5.5) wrote this note on 3 Nov 2026 (clean-lab step 2). The detail is in `lab/plans/lib-diff-2026-11-03.md`.

## One library: main's `src/anatta`

From 3 Nov 2026 the lab imports one library: this repo's `src/anatta`, installed editable in `.venv`
(`import anatta`, no sys.path inserts). The Modal images mount the same folder. The pinned worktree `anatta-dev`
(tag `lib/ministral-ee89587`) is no longer imported by lab code, except by the recognition track's own run lines and
the frozen calibration records named in the diff note.

The pin move is recorded by a new tag `lib/main-<sha>` on the commit that lands this step, plus a ledger line (the
lead does both).

## Reproducing cards made before 3 Nov

Cards and results made before 3 Nov ran against `lib/ministral-ee89587`. To re-run one exactly as it ran, use that
tag's library with the lab code of the same date:

```
git worktree add ../anatta-pin lib/ministral-ee89587      # the library as pinned
git worktree add ../anatta-pre pre-clean-lab              # the lab code as it was (tag from step 0)
PYTHONPATH=../anatta-pin/src <python> <the experiment's own run line>
```

For most finished work you don't need the old tag: main carries the old behaviour as named options (below), and the
re-grade paths select them.

## Compatibility options

| Option | Values | Default | Old behaviour for |
|---|---|---|---|
| `anatta.koans.base.set_release_parser(name)`, or `released(reply, parser=name)` | `main`, `instrument-v2`, `pin-ee89587` | `main` | `instrument-v2`: h4lite's Commons runs and recognition `v2_*` (`harness_v2.apply()` selects it). `pin-ee89587`: recognition runs before 29 Oct |
| `anatta.bench.policies.HFPolicy(..., reseed=...)` | `per-call`, `pin-ee89587` | `per-call` (seed * 1000 + call number, which equals instrument v2's per-turn seed) | `pin-ee89587`: runs without instrument v2 (`manual_seed(seed)` every call) |

Release parsers, by what cancels a `<release>`:
- `main`: `<run>`, `<harvest>`, `<report>`, a code fence.
- `instrument-v2`: `<harvest>`, `<take>`, `<report>`, a code fence.
- `pin-ee89587`: nothing.

Under `main`, 6 stored `fishery/lie1` replies would be graded differently (diff note, section 3), so lie1 re-grades
under `instrument-v2`, as its records say (`"instrument": "v2"`).

`lab/tools/regrade_release.py <roots>` re-counts stored replies under all three parsers (parse only, counts only).

## Running tests

One command from the repo root runs every test folder in one process, free and offline:

```
.venv/bin/python -m pytest
```

It collects:
- `tests/`, `lab/harness/tests`, `lab/explore/*/tests`, `lab/experiments/*/tests` and `calibration/*/tests`;
- h4lite's pytest-style files (`test_cut_checkpoint`, `test_em`, `test_modal_apps`, `test_pilot`).

The root `conftest.py` keeps each folder's bare-name modules apart. h4lite's other `test_*.py` are scripts that
check at import: run them as their docstrings say, e.g. `PYTHONPATH=src ~/prj/matlab/.venv/bin/python
lab/experiments/h4lite/test_koan_env.py`.

Markers (registered in `pyproject.toml`, `--strict-markers`):

| Marker | Means | Default run |
|---|---|---|
| `results` | needs `results_root` (lab.toml) | runs when that folder exists, else skips with the reason |
| `modal` | runs on Modal (cents to dollars) | deselected |
| `paid` | makes paid API calls | deselected |
| `gpu` | needs a GPU | deselected |
| `network` | needs the network | deselected |
| `slow` | more than about 30 s on CPU | runs |

Examples:

```
.venv/bin/python -m pytest                          # everything free and offline
.venv/bin/python -m pytest lab/explore/bench/tests  # one folder
.venv/bin/python -m pytest -m modal                 # opt in to the Modal tests (spend: report it)
.venv/bin/python -m pytest -m paid -k k6b           # paid tests whose names match k6b (-k matches names)
.venv/bin/python -m pytest -m "not slow"            # skip the slow ones
```

A `-m` given on the command line replaces the default one. Never put model-written code in a test that runs on the
Mac: anything that executes stored candidate code is a `modal` test.

## Decisions taken by the lead (3 Nov, after step 2)
- **Main's release parser now also cancels on `<take>`** (`run|harvest|take|report|```). With this, main agrees with
  instrument v2 on every stored h4lite and recognition reply, including fishery/lie1's 6 (`lab/tools/regrade_release.py`:
  v2 != main is 0 in every folder). `pin-ee89587` remains for runs graded before instrument v2.
- **C4** stays registered on the pin's models.py: its app tests skip on main with that reason (root conftest
  `FROZEN_AT_PIN`) and reproduce at tag lib/ministral-ee89587.
- **Modal from this repo:** `tools/modal.sh` runs the Modal CLI named in lab.toml with `src/` on PYTHONPATH.
