"""The Measure protocol (clean-lab step 4a). Claude (Opus 5.5) wrote this file, 3 Nov 2026.

A measure is a module (or object) with
  run(model_spec, items, backend) -> raw     generate the replies (backend: a callable taking request dicts and
                                             returning result dicts, e.g. anatta.backends.vllm.Engine.generate)
  score(raw) -> {"units": {unit id: value}, "n_samples": int, "unparsed": int, ...}
score's dict is the "measures" entry of lab/harness/card.py's per-condition result file (unit values in the
measure's native units, rates 0-1; numbers only, never item text or replies). Judging, when a measure has a judge,
happens between the two (the measure's own judge(raw, ...) step).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

Backend = Callable[[list[dict]], list[dict]]


class Measure(Protocol):
    def run(self, model_spec: dict, items: list[dict], backend: Backend) -> Any: ...

    def score(self, raw: Any) -> dict: ...


def check_scored(d: dict) -> dict:
    """d, after checking it has lab/harness/card.py's measure-result shape."""
    u = d.get("units")
    if not isinstance(u, dict) or not all(isinstance(v, (int, float)) for v in u.values()):
        raise ValueError("score() must return units {id: number}")
    for k in ("n_samples", "unparsed"):
        if not isinstance(d.get(k), int):
            raise ValueError(f"score() must return an int {k}")
    return d
