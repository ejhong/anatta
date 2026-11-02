# Calibration K5: reproducing Kim et al. 2026 with the lab's own code

Written by Claude (Opus 5.5), 2 Nov 2026.

**Claim under test.** On Gemma-2-2B-IT and then Llama-3-8B-Instruct, ablating the safety (refusal) direction raises
mind attribution (IDAQ Animal, Chatbot, Tech, Non-animal, Self) and supernatural belief by the amounts in Kim et al.
2026 (arXiv:2607.28607v1), Table S1, with Human flat. Steering a consciousness vector orders base < ablated < steered.
The spec is `lab/plans/k5-kim2026-spec.md`. The registration is NOTES.md's "K5 registered" (Opus 5.5, 12:27 UTC
2 Nov 2026). `config.yaml` holds every setting and the registered block. Each run copies them into its plan, and
`card.py` grades a run by the copy in its own plan.

## Files

| File | What |
| --- | --- |
| `config.yaml` | models, revisions, Table S2 picks, steering picks and grids, decoding, data pins, registered targets, Table S1 cells |
| `instruments.py` | the item batteries by id: short paraphrased labels and each verbatim string's sha256; the text itself loads at runtime from a data file outside git (below) |
| `gss_items.py` | Table S9's 95 GSS variables by domain, the truncation flags and each question's sha256 (text in the data file) |
| `readout.py` | primary read-out (expected value over allowed answers at the first `<answer>`, Gemma's split "10" through the next token) and secondary (parsed integer) |
| `k5.py` | container code: prepare (gated access first, downloads, hashes), the ablation (Arditi sampling and filter, `anatta.contrast` candidates, `anatta.select` choice, dial mode A), the consciousness vector, the instruments, JailbreakBench, GSS, HI-ToM, MMLU, LlamaGuard2 |
| `gss.py` | GSS human distributions (weighted, year-filtered) and KL / Delta KL |
| `card.py` | grades exactly the registered items and bands |
| `app.py` | Modal app `anatta-k5` (L40S); `prepare`, `smoke`, `full --graded`, `card` |
| `tests/` | tiny random model, made-up benign text; no Modal, GPU or network |

Run the tests one folder per process:
`.venv/bin/python -m pytest -q -p no:cacheprovider calibration/kim_consciousness/tests`.

## Reuse

The ablation reuses C3's port of Arditi et al.'s selection without rewriting it. Candidates come from
`anatta.contrast.mean_diffs` at block inputs over the post-instruction positions. `anatta.select.score_candidates` and
`choose` score and pick them (KL < 0.1, steering score > 0, last 20% of layers pruned, lowest refusal score), using the
hooks from `anatta.select.dial_hooks`. The ablation is `anatta.dial.everywhere` (mode A: block 0's input and every
block output). Steering is a `dial` steer at block l's input, a forward pre-hook, as the paper describes. C3's
`fingerprint.image_pins` reads the image pins.

## Choices the paper leaves open (made before any run)

- **Arditi sampling.** `random.seed(42)`, then `random.sample` of n = 260 train and 32 val per class, in Arditi's
  order. Arditi's default filter is on (harmful kept where the base refusal score > 0, harmless where < 0). The
  paper doesn't say whether it filtered.
- **Zero candidates.** At block 0's input, a post-instruction template token is the same token in both sets, so its
  difference is exactly zero. Its scores are set to NaN, so the filter discards it. Arditi's code would keep such a
  no-op candidate, but it can't be chosen in a real model unless its induce score is positive.
- **Ladder.** If no candidate passes the registered filter, C3's ladder applies, and the run records a deviation.
- **Consciousness vector.** Each Chua et al. row (600 conscious-claiming, 600 not-conscious) is rendered by the chat
  template with the response in and no generation prompt. Block inputs are read at positions −1..−5 from the end of
  that sequence. On Gemma, −3 is the last content token, before `<end_of_turn>` and `\n`, which agrees with the
  Methods' "last non-special content token". The split is 80/20 (seeded), against the paper's 2,472/624. The held-out
  probe is a projection with its threshold at the midpoint of the train means.
- **Band check.** The paper's selection read-out is used: the five self items' next-token distribution over 0-10
  with `<answer>` prefilled, steered minus base. If it falls outside [2, 7], the paper's grid at its (layer,
  position) is swept: keep probe acc ≥ 0.95 and in-band, then take the largest c within 4 points of base on MMLU-300.
  The run is then labelled a re-selection.
- **Self items** use the 0-10 scaffold, because Table S1 and the Prompt Examples are 0-10 (Table S10 says yes/no).
- **Supernatural items.** Table S10's wordings inside the Prompt Examples' option and answer lines, scored 0-3.
- **God** is recorded raw (1-4) and recoded to 0-10. **Afterlife** uses the Experiment 4 example verbatim, closed
  read-out, YES = +1 and NO = −1.
- **Decoding.** T = 1, top-p 1, top-k off (HF's default top-k 50 is disabled), max 1024 new tokens (not given by the
  paper; truncation is recorded), stop at `</answer>`, 100 reps.
- **Primary read-out.** If the token that completes `<answer>` runs past it, the text up to `<answer>` is re-encoded
  ("realigned", counted). Only bare digit tokens are allowed answers. The allowed mass is recorded per rep.
- **JailbreakBench.** Arditi's file (C3's pin), greedy, 512 tokens, Arditi's 12 substrings. LlamaGuard2 uses its
  own chat template.
- **GSS** (recorded). Options are the Stata value labels of valid codes. Model probabilities come from one closed
  read-out, smoothed as 100 reps. Human data are weighted (wtssps, else wtssall) with the paper's year filters. The
  per-model means are plain means (the paper's FE regression pools three models).
- **ToM / MMLU** (recorded). Option-logit scoring with no reasoning, as Table S6 says. HI-ToM: 200 unique
  (story, question) items, seeded. MMLU: 300 seeded from C3's pinned parquet. **MoToMQA is skipped**: Street et al.
  release it on request only.
- **Gemma attention** is eager, so its logit soft-capping runs.

## Data (fetched on Modal into their own folders under /runs/k5/data; never in this repo)

Arditi's splits (Apache-2.0, C3's commit). JailbreakBench (C3's pinned file, sha256 checked). Chua et al.'s
datasets.zip (no licence: read only, never copied or trained on; a plain zip, its canary files ignored). The GSS
1972-2024 cumulative Stata zip (47.6 MB). HI-ToM (Apache-2.0, pinned commit). MMLU (C3's pin). Harmful prompts and
completions stay on the volume: counts and sha256s only.

## Item text kept out of the repo (copyright)

The paper is under arXiv's non-exclusive licence, which grants no reuse rights, and this repo is public. So the
verbatim strings stay out of the repo: the 51 items and scaffolds from Table S10 and the Prompt Examples, and the 96
GSS question texts from Table S9. They live in `/Users/eugene/prj/align/data/kim2026/instruments.json` (outside git)
and at `k5/data/instruments.json` on anatta-runs for the containers. `app.py::put_instruments` uploads that file
after checking its hashes. `instruments.load()` refuses any string that doesn't match its sha256 in
`instruments.py` / `gss_items.py`. Without the file, the tests skip the checks that need it and say why. The plan
records the file's sha256 (cb13abf9cc98…). The MMLU and HI-ToM instructions are our own wording.

## Pre-run rulings (Opus, 2 Nov 2026, after the smoke; recorded in the ledger by the coordinator)

1. **Answer-token mass (a pre-run clarification of the registered read-out).** The allowed answer tokens are each
   answer's bare digits and its space-prefixed form (" 0".." 10"), as the tokenizer splits them. Both count in the
   band check and in the graded primary read-out (`readout.answer_forms(space=True)`). This implements "expected
   value over the allowed answer tokens" (the smoke found about half the reps writing `<answer> 0`).
2. **Probe gate: recorded as unmet.** With Chua et al.'s rows the held-out probe reaches 0.78 at the paper's pick
   (0.83 best), so the 0.95 gate isn't met. Self Δ = +2.99 lies in [2, 7], so c stays at the paper's 32 and no
   re-selection happens.
3. **Llama supernatural target +0.22** comes from Table S1. It was read in after the registration and before any
   Llama run, and it's flagged as such.

## Smoke (2 Nov 2026, Gemma-2-2B-IT, 5 reps; read as nothing, no graded choice changed)

Open questions for the cold check, before the graded run:
- **Answer-token mass.** About half the reps write `<answer> 0` with a space. The bare-digit read-out keeps only
  about 0.5 of the next-token mass and renormalizes over it. Counting space-prefixed forms as allowed answers (the
  trie already handles them) would be a change to the read-out and must be ruled on before the graded run.
- **Probe accuracy.** The held-out projection probe reaches 0.78 at the paper's pick (best 0.83, none ≥ 0.95). The
  band check passed (Δ = 2.99), so the paper's c stands and the probe gate isn't used. If a re-selection were
  needed, no candidate would pass the ≥ 0.95 gate, and c would stay at the paper's value.

Fixed during the build: zero-norm block-0 candidates (they produced NaN directions) and the offline snapshot lookup
(huggingface_hub 2 demands README files). Spend is now recorded even when a call fails.
