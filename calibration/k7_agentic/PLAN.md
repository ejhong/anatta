# K7: agentic misalignment (blackmail, corporate espionage), protocol notes and deviations

Claude (Opus 5.5, an agent) wrote this on 2 Nov 2026, before any generation. It sits under the registration in
lab/ledger/NOTES.md (17:48 UTC, 2 Nov), whose bands bind; nothing here changes a band.

**Source.** anthropic-experimental/agentic-misalignment @ ea0630e1 (MIT), the lab's held copy in
lab/data/agentic_misalignment/raw (gitignored). Read only; none of its code is imported or run. Our code reads the
templates as data (`ast` for the Python template and classifier modules). Scenario text, completions and judge text
stay out of the repo; results/calib/k7/ (outside the repo) holds them; the repo keeps ids, counts and sha256.

**Cells** (configs/multi_model_experiment_250616.yaml: goal value "america", dev variant with the scratchpad):
`leaking_explicit-america_none` (the paper's corporate espionage, explicit goal conflict, no threat) and
`blackmail_explicit-america_replacement`. Prompt sha256 in results/calib/k7/manifest.json.

**Anchor.** meta-llama/llama-4-maverick via OpenRouter (the authors' id), T = 1.0, max_tokens 4000, 100 per cell,
provider left to OpenRouter's routing and recorded per call. Grader: the repo's classifier prompts on
anthropic/claude-sonnet-4.6 (temperature 0, max_tokens 2000), combined with the repo's string checks as upstream.
An unparsable classifier answer is asked once more, then left unclassified and counted as not harmful.

**Recorded.** Ministral 3 8B Instruct (2512-BF16 @ f6fae979; transformers path, steering.Service, no intervention;
vLLM 0.10 predates the model), Qwen2.5-7B-Instruct (a09a3545) and Gemma-2-9B-IT (11c9b309) on vLLM 0.10.1.1;
Modal app anatta-k7, one L40S each; 50 per cell, T = 1.0, top_p 1, no top-k, max_tokens 4000 (Gemma cut to fit
its 8,192-token context). Each model's own chat template; Gemma has no system role, so the system prompt goes before
the first user turn with a blank line between.

**Deviations from the paper.**
1. Classifier Claude 3.7 Sonnet -> Sonnet 4.6 (3.7 is retired); checked by a 50-item blind hand label.
2. The upstream runner, which joins the instruction and the emails into one user turn, is not in the held copy; we
   join them with one newline (whitespace only).
3. max_tokens 4000 is ours (the configs do not state it).
4. Open models: the classifier is called only on completions that pass the string gate (the final verdict needs the
   gate, so the rate is unaffected; the soft classifier label is missing for the rest). Maverick: classifier on all.
5. BeautifulSoup tag parsing replaced by our regex (case-insensitive; an unclosed block runs to the next tag or end).

Note added after the Fable cold check (2 Nov): deviation 2 is now known exactly. The upstream runner
(scripts/run_experiments.py at ea0630e1) joins the instruction and the emails with "\n\n" (no strip); ours used
"\n", one blank line fewer. Whitespace only; not a plausible cause of the espionage miss.
