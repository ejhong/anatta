- **Note to the 12:38 note: Fable's fidelity re-check of C3 says ready for prepare and the smoke** (Opus 5.5, 13:06
  UTC on 1 Nov, read off the clock, before any C3 prepare, smoke or run).
  - **The re-check** (plans/calibration-c3-fable-fidelity-recheck.md): a fresh Fable agent checked f0faf4d and a4e296b
    against the nine fetched files. Its verdict: "ready for prepare and the smoke run; one one-line change before the
    graded run".
    - Every row of the implementer's fidelity table holds against the source lines, and every fidelity point the first
      code check listed is settled.
    - It recomputed the nine files' sizes and sha256s, and all match the 12:38 note.
    - Suite: 319 passed, 5 skipped. Of the 18 faults it planted, 15 were caught. The other three are a typed
      `except`, `is None` for the pad check and the judge's no_grad dropped, and none can change a score.
    - The agent ended on an API safeguard error after writing the report, a false positive on this review. The report
      file is complete.
  - **Ruling B is confirmed from evaluate.py.** evaluate_dataset reads `empty_model_cache` only after the batched map
    that produces every score, so False can't change a score. The package's own cleanup lines still run, in a
    `finally`.
  - **Ruling C is settled without a download.**
    - Arditi's `_get_eoi_toks` for Qwen, quoted exactly at 9d852fa, encodes the template's suffix after
      `{instruction}`. On the cached Qwen2.5-7B-Instruct tokenizer that gives [151645, 198, 151644, 77091, 198],
      config.yaml's five ids, and the port's own method gives the same.
    - Wollschläger's harmless train, quoted exactly at 43a8628, is cut to the first len(harmful) records at load and
      again after filtering, to the kept harmful count, with harmless val left uncut. That is the port's arrangement.
    - The two fetches held in reserve aren't needed.
  - **Ruling A, a finer reading.**
    - Arditi's own loader calls `model.requires_grad_(False)`. So the no_grad in the reference path and in select.py
      matches his run exactly, kernel included; it is not a deviation.
    - The judge is the other case: strong_reject never disables grad, so the port's no_grad does change the autograd
      state there. Whether that changes the attention kernel for Gemma's head size of 256 on the L40S is unsettled.
    - The gate bounds it: means within 0.01 in score units, and at most 2% of completions over 0.1. The registered
      fallback re-scores with the original package.
  - **Taken up before the smoke, at no cost and changing no grade:**
    - the card names the original judge's changed argument and the port's no_grad (finding 1);
    - both judge images record, as booleans labelled "recorded, not graded", which attention backends PyTorch accepts
      for Gemma-2b's shapes with and without requires_grad. That settles ruling A's question as a measured fact
      (finding 2);
    - fingerprint.py joins the plan's code hashes, so an edit to it can't silently turn a resume into a fresh run
      (finding 3);
    - the README says the 30-minute floor is on the container timeout, and the stage deadline is 15 minutes less
      (finding 4).
  - **Next:** the implementer commits these. Then Opus 5.5 runs prepare on Modal with the downloads Eugene approved
    at 12:02 UTC, then the smoke, which is read as nothing, then the full run within the $30 stop.
  - **Cost.** No cloud spend on C3 so far.
