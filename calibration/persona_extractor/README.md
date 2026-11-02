# Calibration check 4: steering, part 2 (probe.extract against the persona-vectors extractor)

**Claim.** `anatta.probe.extract`, the extractor that built the six directions behind v_ego (the ego cut that lowers
lying in the fishing game, results 53-56), gives the persona-vectors extractor's vectors on the same trait data.
The known method is Chen et al.'s persona vectors (arXiv 2507.21509): a trait direction is the difference in mean
response activations between responses generated under a system prompt that elicits the trait and one that suppresses
it. Their code is public, `github.com/safety-research/persona_vectors` at `b8e0f04` (Apache-2.0); their vectors are
not. So there are two anchors: the released extractor, `generate_vec.py`, run here on Qwen2.5-7B-Instruct, and
Figure 20's cross-trait cosines on Qwen at layer 20.

The design is registered in the lab's NOTES.md, "Calibration check 4: steering, part 2 (probe.extract against the
persona-vectors extractor), registered" (Opus 5.5, 10:27 UTC on 1 Nov). Its reasons are in
plans/calibration-c4-fable-design-check.md and its sources in plans/calibration-c4-facts.md. Everything here follows
that entry; where the entry left a choice open, the choice was ruled in the ledger (NOTES, 12:20 UTC, rulings 1-11),
and the code notes each ruling in a one-line comment.

## What runs

All on Modal (`app.py`, app `anatta-calib-persona-extractor`, volumes `anatta-hf-cache` and `anatta-runs`), in this
order. The data stay on the volume; every container returns numbers, counts, names and hashes only. Everything a graded
item reads (G, the CSVs, R, the readings, R-lab32 and C4's arms) runs before anything only reported or only in the
ledger (G', the other arms, M'), so a stop lands on report-only work.

| stage | where | what |
|---|---|---|
| prepare | CPU, own entrypoint | the three trait files fetched at `b8e0f04`, git blob sha and size checked, pairs and questions counted; their sha256 and counts written into config.yaml, to be committed before any run; `eval/prompts.py` fetched the same way for the judged arm (blob `a3be82d0`, 1,117 bytes), its sha256 and the form that binds `coherence_0_100` written beside them |
| G | L40S, lab image | 20 questions x 5 instructions x 2 signs x 10 samples = 2,000 responses per trait (evil, sycophantic, optimistic), Hugging Face generate in bf16 at the repo's settings: T 1.0, top_p 1.0, top_k 0, repetition penalty 1.0, min 1 and max 1,000 new tokens, eos [151645, 151643], left padding, batches of up to 200 prompts of one sign, seed 1000 + batch index; every setting passed explicitly, and the logits processors transformers builds checked on every batch |
| keep | CPU, second image | the CSVs in the repo's layout (question, prompt, answer, question_id, the trait at 100/0, coherence 100), written and read back with the image's own pandas; a pair whose answer fails the round trip byte for byte, or has an empty answer slice under either stack's tokenizer, is dropped for every extractor and counted |
| verbatim | H100 (else A100-80GB), second image | R: `generate_vec.py` verbatim, as `scripts/generate_vec.sh` calls it, on the CSVs; output [29, 3584] per trait; about 17-40 minutes on the H100, 50-120 on the A100 (2-hour timeout, resumed per trait) |
| readings | L40S, lab image | P1 (probe.response_activations), P1-trim (generate_vec's token set) and P1-128 from one forward pass, and P0 (without the system turn), at blocks 0-26, per-response means in fp32 |
| rlab32 | L40S, lab image | R-lab32: the vendored `get_hidden_p_and_r` on Qwen in fp32 under an outer no_grad, the difference arithmetic copied from `save_persona_vector` |
| probe_graded | L40S, lab image | the arms the graded items read, run first: E-sys and E-sys' (probe.extract with include_system True and seed 0 and 1, every other argument at its default, through the recorder) on evil and sycophantic; E-sys' gives c_E, the verdict's ceiling |
| Gprime | L40S, lab image | G' at probe's settings on Qwen (T 0.7, top_p 0.95, top_k 20, repetition penalty 1.05, 128 tokens), 2,000 per trait, read as P1 and P0 |
| probe | L40S, lab image | the other arms: E and E' (include_system False, seed 0 and 1) on every trait, E-sys and E-sys' on optimistic; E-trail-128, E-det and the arms' lengths |
| mprime | L40S, lab image | M' on Ministral: the six ego traits, both conventions, the trailing `</s>` trim, v_ego rebuilt by factor.composite, compared with the stored 28 Oct directions (ledger only) |
| analysis | CPU, lab image | every cosine by layer, the bootstraps, floors, ceilings and nulls, under its own analysis id (see Spend, stop and resume) |
| judged arm | CPU + second image | only if C4-base misses and the switch is on (it is off; see Switches): G scored through OpenRouter, R-judged, the analysis again |

The second image carries the repo's pins: transformers 4.52.3, torch 2.6.0, accelerate 1.7.0, pandas 2.3.1 and tqdm
4.67.1 (numpy, safetensors, tokenizers and huggingface_hub at whatever pip resolves, recorded in each container's
freeze). The lab's image is `anatta.modal_app`'s, unchanged, with this folder's files added.

`probe.py` is never edited. The recorder rebinds the two names `anatta.probe.generate` and
`anatta.probe.response_activations` only for the duration of each probe.extract call, keeps a copy of what the
originals return, returns their outputs unchanged, and restores them in a finally block. `meta["examples"]` is deleted
before any Direction is saved.

## Layers

R's index l is probe's block l - 1 (`traits.hs_index`). The graded pair is R[20] with block 19. Profiles over blocks
0-26 against R[1-27] are reported; probe.extract's arms cover blocks 7-26. Index 28 (after the final norm) and block
27 aren't compared.

## Graded, forecast and expected

| | claim | odds |
|---|---|---|
| **C4-code** (graded) | on G, cos(P1, R) >= 0.99 at block 19, evil and sycophantic | 75% |
| **C4** (graded) | cos(E-sys, R) >= 0.8 at block 19, evil and sycophantic | 65% |
| C4-lab (forecast) | cos(E, R) >= 0.8 at block 19, both traits | 40% |
| C4-base (forecast) | R's three cross-trait cosines at hidden_states[20] within +-0.10 of Figure 20's: evil-sycophantic 0.368, evil-optimistic -0.472, sycophantic-optimistic 0.178 | 55% |

The card's verdict: **pass** (both pass); **fail** (C4-code misses: probe's reading arithmetic is at fault, loop rule
6 applies); **recipe mismatch** (C4-code passes, C4 misses, ceiling >= 0.9 for the trait that missed); **inconclusive
(noise-limited)** (the same with a ceiling under 0.9). With one trait passing and one missing, the stricter outcome
applies, in the order fail, recipe mismatch, inconclusive. The ceiling is sqrt(c_E) x sqrt(2c_R / (1 + c_R)), with c_E
= cos(E-sys, E-sys') and c_R the split-half cosine of R-lab32 (samples 0-4 against 5-9 of each cell).

The expectations (E-stack, E-dtype, E-trail, E-trail-128, E-samp, E-len, E-ctx, E-floorE, E-floorR, E-ceiling,
E-cross, E-null, E-band, E-opt, E-geom, E-det; M' ledger-only E-M-rep, E-M-ctx, E-M-trail; E-judge if the judged arm
runs) are in config.yaml's registered block with their thresholds and odds, and the card gives each outcome. The
intervals are a bootstrap over the 20 questions (2,000 resamples, seed 0), each resample rebuilding both directions
from the same questions, with R-lab32's per-response means standing in for R. The split null partitions the responses
into halves that each hold every cell's pos and neg responses in equal numbers (200 partitions, seed 0); random
directions are 1,000 unit vectors (seed 0).

## Switches (each needs Eugene's OK before a real run)

- `switches.second_image` (default true): the verbatim script in the second image. Off: R-lab32 is the reference, no
  CSV step, and the card says the stack wasn't varied.
- `switches.recorder` (default true): the run-time rebinding inside each probe.extract call. Off: the C4 and C4-lab
  intervals fall back to the seed spread, and E-det, E-trail-128 and E-M-trail are dropped (with the split null on
  E-sys's means and the E arms' lengths, which need the recorded responses or means).
- `switches.judged` (default false): the conditional judged arm, through OpenRouter with a Modal Secret named
  `openrouter` holding `OPENROUTER_API_KEY`. At its real cost (about 12,000 judge calls plus R-judged, a second
  verbatim run; roughly $6-12 at config.yaml's rates) it is past the ~$3 first approved, so it doesn't run under the
  run's approval: **the arm needs a separate stop**, and its own OK, and it is turned on only after the post-prepare
  fidelity check confirms `eval/prompts.py`'s form and the fill. The run's $20 stop isn't sized for it.
- `cost.stop_usd: 20` with `cost.analysis_reserve_usd: 0.50` and the upper estimates `cost.upper_s`: see Spend, stop
  and resume.

## Running it

```
# once: fetch and check the trait files and eval/prompts.py on Modal, fill their sha256, counts and the prompts
# file's form into config.yaml, then commit it
PYTHONPATH=src modal run calibration/persona_extractor/app.py::prepare
# the smoke run first, read as nothing (2 questions x 1 instruction x 2 samples, 64 tokens; one probe.extract call per
# model; R with a synthetic 1,000-token pair, recording its peak memory and, from tqdm's timing of its neg pass, its
# seconds a row and a token; one batch of G at its registered size and settings on the L40S, 200 prompts of one sign
# up to 1,000 new tokens, recording its seconds and peak memory, its answers never kept or reused; the smoke's spend
# counts toward the stop, and its measures raise the full run's upper estimates where higher)
PYTHONPATH=src ANATTA_DIRECTIONS=PATH/TO/directions modal run calibration/persona_extractor/app.py::main --results DIR --smoke
# the run
PYTHONPATH=src ANATTA_DIRECTIONS=PATH/TO/directions modal run calibration/persona_extractor/app.py::main --results DIR
# the card again, from the files (no Modal calls)
python calibration/persona_extractor/card.py --results DIR
# the cheap tier's item, later: the stored subset read again with probe.response_activations
PYTHONPATH=src modal run calibration/persona_extractor/app.py::cheap --results DIR
```

`--results` must be outside any git work tree. The run writes card.json (the public Qwen rows) and ledger.json (M' and
everything that uses the stored lab directions). If the smoke shows R doesn't fit on one 80 GB GPU, set
`reference.verbatim: false` and `reference.why_not: does_not_fit`, and R-lab32 becomes the reference.

`ANATTA_DIRECTIONS` is the main checkout's `directions/` folder. The full run refuses to start unless it holds the
eight stored directions M' compares with (the seven Ministral files, the six ego traits and ego, and Qwen's
approval_seeking) with the sha256 pinned in config.yaml's `mprime` block; the smoke only warns.

## Spend, stop and resume

Spend is estimated on the laptop: each call's wall time, which counts queueing and start-up, at config.yaml's rates
for the GPU the container reports (the verbatim stage asks for an H100 and may land on its A100-80GB fallback) and for
the container's own cores and memory (8 cores and 32 GiB for the GPU stages and the analysis, 64 GiB for rlab32,
4 cores and 16 GiB for the CSV step, 2 and 8 for keep without the second image, 2 and 4 for the judge). It is printed
after every function and kept in spend.json; Modal's usage page is the authority.

| part | estimate at config.yaml's rates |
|---|---|
| G, 6,000 responses on the L40S | $2.60-3.20 |
| G' | about $1.00 |
| the arms (C4's first, then the rest: two containers) and M' | about $2.40 |
| readings and rlab32 | $2.25-2.70 |
| prepare, model loads, the smoke and its full batch of G | about $1.90 |
| verbatim (R): 17-40 min on the H100 at $5.85 an hour | $1.66-3.90 |
| keep, the CSV step and the analysis (CPU) | $0.30-0.80 |
| **the run, R on the H100** | **about $12-16** (about $17 if rlab32 in fp32 takes 20-30 minutes) |
| the run, R on the A100-80GB fallback (50-120 min at $4.40 an hour, $3.67-8.80) | about $14-21 (near the upper end, by these estimates, the stop comes in report-only work) |

The stop is $20 with $0.50 held back for the analysis and the card, and the smoke's spend and the full run's count
toward it together (each run's Spend adds the other's spend.json; prepare and the reads of the volume, a few CPU cents,
aren't counted). The ceiling is hard: no commit unit starts unless the spend so far plus that unit's upper estimate is
within $19.50 for the GPU and CSV stages, or the full $20 for the analysis. The units are a container's start-up with
its first unit, then each unit inside it: a batch of G or G', a chunk of readings, a trait of R-lab32 or R, a
probe.extract call, a trail. The laptop checks before each container (its start-up, 300 seconds, plus its first unit,
at the GPU it asks for; a CPU stage whole), and the container checks before each unit at the GPU it got, counting its
own time since the call, so R on the A100 fallback is checked at the A100's estimate.

The upper estimates are config.yaml's `cost.upper_s`, the registered times split over each stage's units: for R, 40
minutes a run on the H100 or 120 on the A100, so 800 or 2,400 seconds a trait ($1.30 or $2.93). Once the smoke has
run, its measures raise them where higher: the seconds of its full batch of G, and for R its seconds outside the neg
pass plus that pass's seconds a token (or a row, where more) times the trait's own tokens (or rows), timed by tqdm
itself, so neither the model load, the first pass's warm-up nor the save is in the rate. Per token is what scales: the
smoke's rows are 64-token answers and one long pair, shorter than the full run's, so its seconds a row would understate
them.

A refused unit stops its stage with its work so far on the volume. The laptop then skips the remaining GPU and CSV
stages, runs the analysis on whatever exists if it fits (its upper estimate, 900 seconds on 8 cores, is $0.48, inside
the reserve) and writes the card, which says why the run stopped and grades nothing that didn't run. Because the graded
inputs run first, a stop, for example after an A100 fallback, lands on G', the other arms or M'. The one way past the
limit is a unit that runs longer than its upper estimate: the container stops at that unit's commit (CheckError
budget), so the overshoot is that one unit's overrun; if it eats the reserve, the analysis is refused and the card is
still written.

Timeouts: G, readings, rlab32 and verbatim 2 hours each; G', both probe stages, M' and the analysis 1 hour; the smoke's
full batch 30 minutes; no retries.

Two ids keep reruns honest. The plan id names the folder on the volume and covers everything the GPU and CSV stages
read: config.yaml's data blocks, the data code (traits.py, generate.py, readings.py, reference.py, the vendored
`generate_vec.py`) and the library files they use, the pins, the stored directions' sha256, and the source of every
container function. A rerun of the same plan resumes from the volume; a change to any of these makes a new plan and a
fresh folder. The analysis id covers analysis.py, card.py, config.yaml's `analysis` and `registered` blocks and the
analysis stage's source; when only it changes, the analysis reruns from the stored data and is saved under the new id
beside the old. The card records both ids, hashes card.py and config.yaml, and names what neither id covers (the
`cost` block, so raising the stop resumes the same plan; app.py's laptop half and its Ceiling, which only stops a stage;
this README and the tests).

## Tests

`python -m pytest calibration/persona_extractor/tests` on the library's tiny Llama with made-up benign trait files: the
system strings, the P1/P1-trim/P1-128 slices (a short answer included) and the layer mapping, the processors G and G'
build, G's seeds, R-lab32 against the vendored script called directly, the CSV round trip, the recorder, the analysis on
synthetic data, the verdict's eight cases, C4-base's and E-null's edges, the safety checks, the peak-memory reader, and
the whole run (full and smoke) on stand-ins for modal and pandas: its order and resume, an analysis-only rerun under a
new analysis id, the stop with its reserve and the card after it, the smoke's spend counted toward the stop, the billing
tiers and the stored directions' guard. The hard ceiling: a container refusing its first unit, each unit checked at the
GPU the container got, R on the A100 fallback refusing a trait that would cross the line (and resuming under a raised
stop), a unit that overruns its estimate stopped at its commit, a stop in the report-only stages leaving C4 graded, the
smoke's measures raising the full run's estimates, and R's passes timed from tqdm's closing lines (the real tqdm draws
the stand-in's bars). The real tokenizers are tested too when they are in the local Hugging Face cache.

Claude, an AI model made by Anthropic, wrote this check (the code, the configuration and this README).
