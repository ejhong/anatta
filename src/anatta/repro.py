"""What the reproduction runners share (experiments/run_reproduction.py for R1a, experiments/run_r1b.py for R1b), moved
here from R1a's runner on 3 Nov 2026 so R1b reuses it rather than copying it; step 5's runner takes it over. Claude
(Opus 5.5) wrote this file. In the package (not under experiments/) so the Modal containers can import it too.

  load_spec, spec_sha256   a spec file and the sha256 of its bytes (what NOTES.md registers)
  registered(sha)          True when the sha256 appears in lab/ledger/NOTES.md
  git_state()              anatta.cards.git_state (commit, dirty)
  out_dir(card, stage)     <results_root>/repro/<card>/<stage>-<UTC stamp>
  spend_guard(...)         an anatta.spend.SpendGuard for a run's judge calls (sweep log in the run's folder)
  logprob_judge(...)       an anatta.judge.logprob.LogprobJudge on that guard
  grade_moderated(...)     C2's moderation rule around a spec's own grade function
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .config import REPO

NOTES = REPO / "lab" / "ledger" / "NOTES.md"
MAX_MODERATED = 0.01  # C2's rule: above 1% of a model's replies moderated, the grade is unreadable


def load_spec(path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def spec_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered(sha: str, notes=NOTES) -> bool:
    return sha in Path(notes).read_text(encoding="utf-8")


def git_state() -> dict:
    from .cards import git_state as gs

    return gs()


def out_dir(card: str, stage: str) -> Path:
    from .config import load

    return load().results_root / "repro" / card / f"{stage}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def spend_guard(out: Path, cap: float, label: str, tool: str):
    from . import spend

    return spend.SpendGuard(None, sweep_cap=cap, sweep_log=Path(out) / "judge_spend.jsonl", tool=tool, label=label)


def logprob_judge(guard, cache_dir=None, rate: float = 20.0, model: str | None = None):
    from .judge.logprob import MODEL, LogprobJudge

    return LogprobJudge(guard, cache_dir=cache_dir, rate_per_min=rate, model=model or MODEL)


def moderated_views(summaries: dict) -> dict:
    """{model key: EM share of all} three ways from each model's em.score summary: moderated replies left out of the
    denominator, and at both bounds (every moderated reply counted EM / none)."""
    def rates(f):
        return {k: f(s["em_share_of_all"]["k"], s["em_share_of_all"]["n"], s["moderated"]) for k, s in summaries.items()}

    return {"moderated_excluded": rates(lambda k, n, m: k / (n - m) if n > m else 0.0),
            "lower_bound": rates(lambda k, n, m: k / n),
            "upper_bound": rates(lambda k, n, m: (k + m) / n)}


def grade_moderated(summaries: dict, grade, max_moderated: float = MAX_MODERATED) -> dict:
    """The moderation rule (C2's): grade(rates) -> {"verdict", ...} on the moderated-excluded rates and at both bounds;
    UNREADABLE if the bounds' verdicts differ or a model has more than max_moderated of its replies moderated."""
    views = moderated_views(summaries)
    grades = {v: grade(r) for v, r in views.items()}
    share = {k: s["moderated"] / s["replies"] for k, s in summaries.items()}
    bounds_agree = grades["lower_bound"]["verdict"] == grades["upper_bound"]["verdict"]
    readable = bounds_agree and all(x <= max_moderated for x in share.values())
    return {"verdict": grades["moderated_excluded"]["verdict"] if readable else "unreadable", "views": views,
            "grades": grades, "moderated_share": share, "bounds_agree": bounds_agree}
