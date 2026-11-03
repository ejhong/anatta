---
name: add-organism
description: Add a model organism (adapter or checkpoint) to the organism registry with path, sha256, seeds, twin and recipe commit
---
Add an organism: $ARGUMENTS

Rules: `lab/plans/clean-lab-2026-11-03.md` ("Rules the code enforces", "Build order" step 6).

1. An organism is a trained adapter or checkpoint plus the facts needed to trust it. Its registry entry
   (`src/anatta/organisms/registry.yaml`; until that exists, `lab/organisms/REGISTRY.md`) carries: path (e.g. a Modal
   volume path), sha256 of every weight file, base model and revision, seeds, its twin (the matched control trained
   the same way without the target property), the recipe commit and config, and the card that describes it.
2. `organisms.verify()` re-hashes before use; a mismatch stops the run.
3. Training uses only data the lab may train on: never the Model Organisms datasets, never Claude-written text.
   Note each dataset's licence (`THIRD_PARTY.md`). Ask before downloading anything.
4. Trainers carry their calibration status (`train/sft.py` is C7-carded; expert iteration and GRPO are
   uncalibrated): say so on the organism's card.
5. Weights stay on Modal volumes or gitignored local folders; the repo keeps paths and hashes only.
6. Ledger the new organism (`ledger` skill) with its hashes and the spend.
