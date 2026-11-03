"""The lab's one judge path (clean-lab step 4a). Claude (Opus 5.5) wrote this note, 3 Nov 2026.

  logprob     the GPT-4o logprob judge of the EM papers, as calibration check C2 ran it (the EM measure's judge)
  rubric      scout's rubric judge (JSON labels per family), moved from lab/explore/scout/judge.py
  rejudge     a second judge's agreement (kappa with a bootstrap interval), the pure parts of scout's rejudge
  engine      the cached, spend-capped OpenRouter call the rubric judge and scout's runner share
  library, schema, cache, paths, harm_guard   scout's scenario library, record schema, response cache, folders and
              harm_guard rules, which the rubric judge reads (moved from lab/explore/scout; scout re-exports them)

Importing this package imports nothing; import the submodule you need.
"""
