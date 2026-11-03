"""The lab's harness: one candidate in, one card out (spec: lab/plans/harness-spec-2026-11-02.md, v2).

Claude (Opus 5.5) wrote this package, 2 Nov 2026. Built so far: the candidate schema (candidate.py), the measure
registry (measures.yaml, measures.py), the scorecard (scorecard.yaml, scorecard.py), the statistics (stats.py), the
card writer (card.py) and an interface stub for the Modal runner (runner.py). No GPU or API call lives here.
"""

from pathlib import Path

from anatta.config import REPO

HERE = Path(__file__).resolve().parent
ALIGN = REPO.parent  # the folder holding the repo; card paths in measures.yaml are relative to it
