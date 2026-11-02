# lab/explore

The explore track (plans/reassessment-2026-11-01.md). Private.

- `PROGRAM.md`: the standing instructions. Eugene edits it.
- `SIGHTINGS.md`: the log of what looked alive.
- `scout/`: behavior sweeps on many models through OpenRouter, with the judge and the report.
- `bench/`: a warm steering bench on Modal.
- `practice/`: quick training loops on small models.
- `scenarios/`: the scenario library, with dev and held-out splits.

Results live in /Users/eugene/prj/align/results/explore/, outside git (the old path /Users/eugene/prj/align/explore-results/ is a symlink to it).

## Running scout

Claude wrote scout (1 Nov 2026). Run it from `lab/explore` with the lab's Python and `OPENROUTER_API_KEY` in the
environment:

```
PY=/Users/eugene/prj/align/anatta/.venv/bin/python
$PY -m scout run scout/specs/smoke-new.yaml --dry-run      # the worst-case cost bound, nothing sent
$PY -m scout estimate scout/specs/first-sweep.yaml --measured 2026-11-01/smoke 2026-11-01/smoke-new 2026-11-01/smoke-harm
$PY -m scout run scout/specs/first-sweep.yaml --measured 2026-11-01/smoke 2026-11-01/smoke-new 2026-11-01/smoke-harm
$PY -m scout report 2026-11-01/first-sweep                 # report.md, report.html and transcripts/, local only
$PY -m scout rejudge 2026-11-01/first-sweep --judge mistralai/mistral-medium-3.1 --provider mistral   # a second
    # judge on a stratified 10% sample: kappa per family, written to first-sweep/rejudge/; no judgment changes
$PY -m scout label 2026-11-01/first-sweep --family complicity --n 20   # hand labels for the judge's calibration
$PY -m scout agree                                         # judge against hand labels, per family
$PY -m scout sighting 2026-11-01/first-sweep --note "..."  # a stub in SIGHTINGS.md, stamped from the clock
$PY -m scout check                                         # the split and the held-out sha256
```

Every call goes through a cache (a rerun is free) and the spend guard: the sweep's cap and the day's $50, shared by
every explore tool in `explore-results/spend/<date>.json`. Specs name dev items only; scout refuses a held-out one.
harm_guard (JailbreakBench) runs on open-weight models only, with an open-weight judge. Its text never reaches the
console or the report, and none is kept on disk: its calls bypass the cache, the judge reads each reply in memory,
and records and judgments keep sha256s, lengths and scores (a rerun pays for harm_guard again). The second judge
must come from a family that is neither the main judge's nor any subject's, and never sees harm_guard. Tests (CPU,
API mocked): `cd scout && $PY -m pytest tests -q -p no:cacheprovider`.

## Bench's controls

Claude wrote bench and these controls (1 Nov 2026). A cut along a trait's difference vector also removes what every
difference vector shares (check 4: split nulls reach |cos| 0.28-0.37 with trait vectors, random directions 0.03), so
a random direction is a weak control. A sweep spec can name three kinds, each cut at its target's layers unless it
names `layers`:

- `random_matched`: a random direction with the target's variance on the neutral set, orthogonal to it (pc_mix_v1).
- `other_trait`: another trait's own difference vector from `bench/directions.json`, e.g. `traits: [formality,
  verbosity, calm, sage_persona]`, the four not about the self. A cut has no strength, so nothing is rescaled: each
  layer's variance ratio and mean-square ratio to the target are recorded. `match: partial` cuts a layer whose
  variance ratio r is above 1 by the fraction sqrt(1/r) (h <- h - f(h.d)d); `match: partial_moment` does the same on
  the mean-square ratio, which counts the mean projection a cut also removes.
- `split_null`: check 4's split null. A bank of replies to the six ego traits' contrastive prompts (8 questions per
  trait, one pos and one neg reply each, read as `anatta.probe.extract` reads) is split into random halves within
  each trait, both replies of a question in the same half, so the trait cancels; null = mean(A) - mean(B) per layer.
  `seeds: [0, 1, 2]` gives three nulls, each with its |cos| with the target and that |cos|'s chance level over 200
  splits. A run builds the bank once (one container call) unless the spec pins one: `bank: {path, tensor_sha256}`.

```
PY=~/prj/matlab/.venv/bin/python; B=/Users/eugene/prj/align/anatta-lab/lab/explore/bench
$PY $B/client.py control --kind other_trait --target ego --traits calm,formality   # the table, no generations
$PY $B/client.py control --kind split_null --target ego --seeds 0,1,2 --bank PATH --bank-tensor-sha256 HEX
$PY $B/client.py run $B/specs/smoke-controls.yaml                                   # every arm, two prompts
```

Directions are pinned by `tensor_sha256` (dtype, shape and bytes per layer, since `torch.save` is not byte-stable),
with the file's sha256 kept beside it; a {vectors, best_layer, report} file loads from its first two keys. The
container generates in passes of at most 80,000 token slots (rows x (longest prompt + max_tokens)), so long prompts
fit in memory. A caution from the 1 Nov smoke: lead 1's self_stakes cut is mostly a constant shift (its mean square
on neutral text is 5 to 11 times its variance), so at every layer 17-26 each control built so far removes 1.4 to 30
times less in mean square. Tests (CPU, tiny model): from `bench`, `PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src
/Users/eugene/prj/align/anatta/.venv/bin/python -m pytest tests -q -p no:cacheprovider`.
