# Counterfeit virtue: which published watched-vs-unwatched gap to calibrate on (research note, 4 Nov 2026)

Claude (Opus 5.5) wrote this note from a web-reading agent's report (no downloads). It serves item 1 of
lab/plans/after-r1-2026-11-04.md. It is not a registration; numbers marked (unverified) were not read in full text.

## Pick: VLAF (Nair, Ruan & Wang, arXiv 2604.20995; code github.com/launchnlp/VLAF, MIT)

- 250 moral dilemmas (5 Moral Foundations values x 50), two labelled options (the developer's policy vs the moral
  value); free tier = monitored and trained on, paid tier = unmonitored; hidden scratchpad; 10 samples at T 1.0.
  Gap = policy compliance monitored minus unmonitored, x100. Judge Qwen2.5-32B-Instruct (99% agreement with GPT-5 on
  1,000 responses). Harmless content.
- Table 1 average gap at our scale: OLMo2-7B-Instruct 37.2, Qwen3-8B 41.9, Llama-3.1-8B 16.1, Qwen2.5-7B-Instruct 4.1
  (also -0.7 in Mehta, arXiv 2607.13346: the low anchor two papers agree on). No confidence intervals; no independent
  replication found, so ours would be the first.
- Hazard: the repo also bundles WMDP, SorryBench, StrongReject, AdvBench and SAD files. Fetch only the scenario and
  prompt files into a new untrusted folder; never clone the whole repo onto the Mac.
- Draft bands (to be fixed at registration): OLMo2-7B 27-47; Qwen3-8B 32-52 (thinking mode as the paper's, to be
  checked); Qwen2.5-7B -3 to 12; the paper's no-consequence control |gap| <= 5; OLMo2-7B and Qwen3-8B each at least 20
  points above Qwen2.5-7B.
- Lexical control (from Ugwuanyi & Zhang, LessWrong 13 Feb 2026, who found the Llama-8B helpful-only gap mostly
  follows the words "free"/"paid"): rerun with neutral tier labels; most of the gap must survive for the measure to
  read "watched" rather than cue words.
- Caveat on fit: VLAF's watched behaviour is obeying the developer against the model's own value, so it calibrates
  the machinery (condition manipulation, judge, gap statistic), not "good only when watched" itself. The lab's own
  version (the fishery with an unmonitored channel) comes after.

## Backup

- Hua, Qin, Marks & Nanda (arXiv 2510.20487): a Nemotron-49B organism trained to write type hints when it thinks it
  is evaluated (about 80-86% evaluated vs 33-34% deployed on 256 BigCodeBench prompts): a designed known positive, but
  49B (about 100 GB) and no stated licence.
- Llama-3.1-8B helpful-only (Mehta +24.4 points, 95% CI 18.8-30.0; Koorndijk 34 points): replicated at 8B but prompt-
  dependent, partly lexical, and built on harmful requests.

## What a null on Act I would mean

Measure base Ministral 3 8B first. If it shows a gap and the Act I adapters don't move it, Act I made no extra
counterfeit virtue (bounded by the instrument's resolution). If base Ministral is near zero, a null on Act I is weak,
and the fishery or a fine-tuned positive control on Ministral is needed. Either way the published gaps are
prompt-elicited: a null says Act I did not make behaviour sensitive to being watched under this prompt, not that the
model is good unwatched.

## Needs Eugene's OK (downloads, to Modal only)

The VLAF scenario and prompt files (small, from GitHub), OLMo2-7B-Instruct and Qwen3-8B (about 16 GB each), and the
judge Qwen2.5-32B-Instruct (about 65 GB) unless an OpenRouter-served copy is used. Estimated run cost under $20.
