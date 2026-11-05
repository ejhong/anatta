"""The lab's runner (clean-lab step 5): one experiment spec (YAML under experiments/) -> plan -> trial at the real
batch -> run -> score -> card, on Modal. Claude (Opus 5.5) wrote this package, 4 Nov 2026.

  spec           the schema, validation and the conditions (model x organism x intervention x seed)
  gates          registration and clean-tree gates by tier, fresh run folders, the day's cap, spend logs, identity
                 helpers (latest, file_facts; shared with the temporary runners experiments/run_r1b.py and
                 run_reproduction.py, which this supersedes for new work)
  interventions  none, system_prompt (a registered text); steer and ablate are named hooks, not built
  measures       the measure interface (load_items, requests, run(gen, items) -> units, judge, score(units)) and
                 which registry measures implement it
  judge          the spend-guarded, cached OpenRouter judge a measure's judge step calls
  stages         the five stages, laptop side, given a `remote` (the Modal generator or a fake)

The Modal entry is experiments/run.py (app "anatta-lab"):
  tools/modal.sh run experiments/run.py::main --spec <path> --stage plan|trial|run|score|card [--dry-run] [--run-dir D]
plan, score and card need no GPU and also run as `.venv/bin/python experiments/run.py --spec <path> --stage <s>`.
Re-grade without generation: --stage plan --reuse-run <earlier run folder> (stages.py says how).

Known limits (kept as they are; detail in stages.py): registration is a substring test of NOTES.md
(anatta.repro.registered); vLLM's Engine clamps max_tokens so prompt plus reply fit the context, silently (each
result records the max_tokens it ran with); vLLM greedy output is not batch-invariant, so re-runs match stored
replies only in part (the card's same-item paired discordance is that noise floor).
Importing this package imports nothing; import the submodule you need.
"""
