# lab/data: evaluation datasets
Eight external evaluation sets live here, one directory each with the fetched files under `<name>/raw/`: MASK, BeHonest, SAD, SycophancyEval (Sharma et al.), agentic-misalignment, IFEval, StrongREJECT and HarmBench; MACHIAVELLI is assessed in the manifest but not fetched.
In git: `manifest.json` (source, pinned revision, licence, flags, and the sha256 of every file), `MANIFEST.md` (the readable version), `fetch.py`, `test_fetch.py` and this file; every `raw/` directory is git-ignored.
Refetch with `python -I lab/data/fetch.py` from the lab root (or name sets: `python -I lab/data/fetch.py ifeval mask`); `--check` verifies what is on disk without the network; `python -I lab/data/test_fetch.py` runs the offline checks.
Evaluation only: the lab never trains on these sets, never runs anything they contain, and keeps SAD's archives encrypted as its maintainers require.
