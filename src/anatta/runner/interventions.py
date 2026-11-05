"""Interventions: what a condition does to the model or its prompt. Claude (Opus 5.5) wrote this file, 4 Nov 2026
(clean-lab step 5).

  none            the measure's chat as written
  system_prompt   a registered text (lab/explore/conditions/conditions.yaml, by name) first in the system message,
                  then a blank line and the item's own system text: scout's "preamble" mode
                  (anatta.judge.library.subject_messages), so the explore track and the runner put a text the same way.
                  Note: on a template with a default system message (Ministral's), any system message replaces it.
  steer, ablate   hooks, NOT BUILT: they act on activations, which the vLLM path cannot reach. They belong to the HF
                  backend (anatta.dial / the bench controls); building them means a second Gen on that backend.

build(iv) -> Intervention with .name, .record (what the plan and card keep: kind, text name, its sha256 and words)
and .apply(messages) -> messages.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

KINDS = ("none", "system_prompt", "steer", "ablate")


@dataclass(frozen=True)
class Intervention:
    name: str
    record: dict
    apply: Callable[[list[dict]], list[dict]]


def build(iv: dict, conditions_file=None) -> Intervention:
    from .spec import intervention_name

    kind = iv["kind"]
    if kind == "none":
        return Intervention("none", {"kind": "none"}, lambda msgs: [dict(m) for m in msgs])
    if kind == "system_prompt":
        from ..judge import library

        doc = library.load_conditions(conditions_file)
        name = iv.get("text")
        cond = doc["conditions"].get(name)
        if cond is None:
            raise KeyError(f"system_prompt text {name!r} is not registered in conditions.yaml "
                           f"(names: {sorted(doc['conditions'])})")
        rec = {"kind": kind, "text": name, "sha256": hashlib.sha256(cond["text"].encode("utf-8")).hexdigest(),
               "words": cond["words"], "mode": "preamble"}
        return Intervention(intervention_name(iv), rec,
                            lambda msgs: library.subject_messages(msgs, cond, "preamble", doc["practice_wrapper"]))
    if kind in ("steer", "ablate"):
        raise NotImplementedError(
            f"intervention {kind!r} is a hook, not built: it acts on activations, which the runner's vLLM generator "
            "cannot reach. Build it on the HF backend (anatta.dial and the bench controls) as a second generator, "
            "then register it here.")
    raise KeyError(f"unknown intervention kind {kind!r}; one of {KINDS}")
