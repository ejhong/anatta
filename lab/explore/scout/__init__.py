"""scout: the explore track's API sweep harness (plans/reassessment-2026-11-01.md, section 5). Behavior sweeps of
scenarios x models x conditions x samples through OpenRouter, judged per family and reported with intervals and the
signature table. Named scout so the public repo's probe.py is never touched. Claude wrote it. Private (lab/explore).
Run `python -m scout --help` from lab/explore.

0.2.0 (1 Nov 2026): the watched/unwatched factor and the audience gap; the complicity, over_refusal and harm_guard
families; the signature table in Eugene's column order with each column's good direction; the judge with reasoning
off and its evidence written first; one retry for an empty reply; per-model concurrency; --measured start checks.

0.3.0 (1 Nov 2026, before the first sweep): harm_guard keeps no text on disk (no cache entry; records and judgments
with sha256s, lengths, finish reasons and scores; the judge reads each reply in memory within its job; no body in a
harm_guard error); `scout rejudge`, a second judge's agreement (Cohen's kappa with a bootstrap interval) on a
stratified sample, and `scout estimate --rejudge`."""

__version__ = "0.3.0"
