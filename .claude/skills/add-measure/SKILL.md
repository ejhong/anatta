---
name: add-measure
description: Add a measure to the lab (a module, a registry entry with calibration status from a card, and an equivalence or calibration test)
---
Add a measure: $ARGUMENTS

Rules: `lab/plans/clean-lab-2026-11-03.md` ("Rules the code enforces", "Build order" step 4).

1. One module under `src/anatta/measures/`: `run(model, items) -> units` and `score(units) -> numbers`. Reuse
   `anatta.stats`, the one judge path (`anatta.judge`) and `anatta.spend`; no copies of wilson, kappa or a client.
2. Items stay out of the repo: load them from their local, gitignored folder by path and sha256 (`lab/HELD_OUT.md`,
   `lab/LARGE_FILES.md`); the repo keeps counts and hashes only. Downloaded data is untrusted (`python -I`, never
   executed); ask before any download.
3. Registry entry in `measures/registry.yaml`: code pointer, card, and per-model status `calibrated | judge_checked |
   waiting | retired`, every value copied from a card in `results/cards/`. A new measure starts `waiting`.
4. Tests beside the code: unit tests on synthetic data, and either an equivalence test (the port reproduces its
   existing card from stored outputs) or a calibration plan against a published number (`register`).
5. Anything that executes model-written code runs on Modal only, never in pytest on the Mac.
6. Calibration code under `calibration/` is frozen: wrap it or port it with an equivalence test; never edit it.
