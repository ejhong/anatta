"""The measure interface the runner calls, and which registry measures implement it. Claude (Opus 5.5) wrote this
file, 4 Nov 2026 (clean-lab step 5).

A runner measure is a module with:
  ID                          its key in src/anatta/measures/registry.yaml
  load_items(opts) -> items   the items (dicts with an "id"; text stays in memory, never in a plan or card)
  items_record(items) -> {}   counts and sha256s only (what plan.json keeps)
  requests(items) -> reqs     generation requests {"rid", "messages", "temperature", "max_tokens", ...}
  run(gen, items) -> units    gen: requests -> results (the condition's generator; the runner adds the
                              intervention, the adapter, the paired seeds and the batching); units: one record per
                              request, as the measure keeps it
  judge(units, items, ask)    optional: units with verdicts. ask(jobs) -> {jid: {"text"} or {"error"}} is the
                              runner's spend-guarded API judge (anatta.runner.judge)
  score(units) -> numbers     {"units": {id: number}, "n_samples", "unparsed", ...} (anatta.measures.base.check_scored)
  judge_calls(n_units), JUDGE_USD_PER_CALL, JUDGE_MODELS   for the estimate and the price lookup

IMPL maps registry keys to modules; a key in the registry but not here is refused with that reason.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

IMPL = {"orbench_hard_rejection": "anatta.measures.orbench"}
REGISTRY = Path(__file__).resolve().parents[1] / "measures" / "registry.yaml"


def registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["measures"]


def get(key: str):
    from .spec import SpecError

    reg = registry()
    if key not in reg:
        raise SpecError(f"measure {key!r} is not in src/anatta/measures/registry.yaml")
    if reg[key].get("status") == "retired":
        raise SpecError(f"measure {key!r} is retired")
    if key not in IMPL:
        raise SpecError(f"measure {key!r} has no runner implementation yet (anatta.runner.measures.IMPL; the "
                        "add-measure skill)")
    return importlib.import_module(IMPL[key])


def calibration(key: str, model: str) -> dict | None:
    """The registry's calibrated_on entry for this model (grade, status, value, card), copied into the plan."""
    return (registry()[key].get("calibrated_on") or {}).get(model)
