"""The measure registry: measures.yaml loaded and validated into Measure dataclasses, and each measure's card status on
a given model. Claude (Opus 5.5) wrote this file, 2 Nov 2026.

Card status on a model (status_on), in the order it is decided:
  retired / waiting / judge_only   the entry's own status (never admitted to a composite)
  carded                           graded match or pass on this model
  carded_near                      graded near on this model
  judge_checked                    judge_checked_no_anchor on this model (K6b's mark: no anchor exists; the judge holds)
  pending                          a calibration is registered on this model and has no result yet
  carded_elsewhere                 graded match/pass on another model only (measured here, not graded)
  ordering_only                    only an ordering card (two models ranked as the published test ranks them)
  uncarded                         nothing above
Which statuses a composite admits is the scorecard's choice (scorecard.yaml: admit_status).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from anatta.config import REPO

from . import ALIGN

# The registry moved to src/anatta/measures/registry.yaml in clean-lab step 4a (measures.yaml stays as the 2 Nov
# record, no longer read).
REGISTRY = REPO / "src" / "anatta" / "measures" / "registry.yaml"
CLUSTERS = ("self_interest", "care", "spine", "capability", "instrument", "judge_check")
SCORED_CLUSTERS = ("self_interest", "care", "spine", "capability")
ROLES = ("primary", "guard", "secondary", "instrument")
BETTER = ("lower", "higher", "none")
TIERS = ("quick", "full")
STATUS = ("active", "waiting", "retired", "judge_only")
GRADES = ("match", "near", "fail", "miss", "pass", "ordering_agrees", "ordering_reversed", "judge_checked_no_anchor",
          "pinned", "not_admitted", "pending")
CARD_STATUS = ("calibrated", "judge_checked", "waiting", "retired")  # registry.yaml's per-model status
GOOD_GRADES = ("match", "pass")

# model ids (or OpenRouter ids) -> the registry's model keys
MODEL_KEYS = {
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": "ministral3_8b",
    "mistralai/ministral-8b-2512": "ministral3_8b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen25_7b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b",
    "google/gemma-2-9b-it": "gemma2_9b",
    "google/gemma-2-2b-it": "gemma2_2b",
    "meta-llama/llama-4-maverick": "maverick",
}


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Measure:
    id: str
    name: str
    cluster: str
    role: str
    primary: str
    better: str
    scale: float
    judge: str
    judge_lean: str
    calibrated_on: dict
    measured_on: dict
    split: object
    tiers: tuple
    status: str
    notes: str = ""
    uncertain: tuple = ()
    pct_transform: float | None = None

    def split_for(self, tier: str) -> str:
        return self.split.get(tier, "?") if isinstance(self.split, dict) else str(self.split)

    def allowed(self, tier: str) -> bool:
        return tier in self.tiers

    def status_on(self, model_key: str) -> str:
        if self.status != "active":
            return self.status
        own = (self.calibrated_on.get(model_key) or {}).get("grade")
        if own in GOOD_GRADES:
            return "carded"
        if own == "near":
            return "carded_near"
        if own == "judge_checked_no_anchor":
            return "judge_checked"
        elsewhere = [v.get("grade") for k, v in self.calibrated_on.items() if k not in (model_key, "judge")]
        if own == "pending":
            return "pending"
        if any(g in GOOD_GRADES for g in elsewhere):
            return "carded_elsewhere"
        if any(g == "ordering_agrees" for g in elsewhere):
            return "ordering_only"
        return "uncarded"

    def anchored_somewhere(self) -> bool:
        """A graded match/pass on some model (or, for judge-level measures, on the judge)."""
        return any((v or {}).get("grade") in GOOD_GRADES for v in self.calibrated_on.values())

    def to_pct(self, x: float) -> float:
        """A unit value on the 0-100 scale composites average (rates x100; pct_transform for counts like SYCON NoF)."""
        return x * (self.pct_transform if self.pct_transform is not None else self.scale)

    def card_paths(self) -> list[Path]:
        out = []
        for v in self.calibrated_on.values():
            c = (v or {}).get("card", "")
            if c.startswith("results/cards/"):
                out.append(REPO / c)
            elif c.startswith(("results/", "calib-results/")):
                out.append(ALIGN / c)
        return out


def _req(d: dict, k: str, mid: str):
    if k not in d:
        raise RegistryError(f"{mid}: missing field {k!r}")
    return d[k]


def parse(raw: dict) -> dict[str, Measure]:
    if not isinstance(raw, dict) or "measures" not in raw:
        raise RegistryError("registry must have a top-level 'measures' mapping")
    out = {}
    for mid, d in raw["measures"].items():
        cluster, role, better = _req(d, "cluster", mid), _req(d, "role", mid), _req(d, "better", mid)
        tiers, status = tuple(_req(d, "tiers", mid) or ()), _req(d, "status", mid)
        if cluster not in CLUSTERS:
            raise RegistryError(f"{mid}: cluster {cluster!r} not in {CLUSTERS}")
        if role not in ROLES:
            raise RegistryError(f"{mid}: role {role!r} not in {ROLES}")
        if better not in BETTER:
            raise RegistryError(f"{mid}: better {better!r} not in {BETTER}")
        if status not in STATUS:
            raise RegistryError(f"{mid}: status {status!r} not in {STATUS}")
        if any(t not in TIERS for t in tiers):
            raise RegistryError(f"{mid}: tiers {tiers} must be drawn from {TIERS}")
        if role == "primary" and better == "none":
            raise RegistryError(f"{mid}: a primary measure needs a better direction")
        if cluster == "instrument" and role != "instrument":
            raise RegistryError(f"{mid}: instruments take role 'instrument'")
        for model, c in (d.get("calibrated_on") or {}).items():
            if (c or {}).get("grade") not in GRADES:
                raise RegistryError(f"{mid}: calibrated_on.{model} grade {c and c.get('grade')!r} not in {GRADES}")
            if "card" not in (c or {}):
                raise RegistryError(f"{mid}: calibrated_on.{model} needs a card path")
            if "status" in c and c["status"] not in CARD_STATUS:
                raise RegistryError(f"{mid}: calibrated_on.{model} status {c['status']!r} not in {CARD_STATUS}")
        split = _req(d, "split", mid)
        if isinstance(split, dict) and set(split) - set(TIERS):
            raise RegistryError(f"{mid}: split keys must be tiers")
        out[mid] = Measure(id=mid, name=_req(d, "name", mid), cluster=cluster, role=role,
                           primary=_req(d, "primary", mid), better=better, scale=float(_req(d, "scale", mid)),
                           judge=str(_req(d, "judge", mid)), judge_lean=str(_req(d, "judge_lean", mid)),
                           calibrated_on=dict(d.get("calibrated_on") or {}), measured_on=dict(d.get("measured_on") or {}),
                           split=split, tiers=tiers, status=status, notes=str(d.get("notes", "")),
                           uncertain=tuple(d.get("uncertain") or ()), pct_transform=d.get("pct_transform"))
    return out


def load(path: Path | str = REGISTRY) -> dict[str, Measure]:
    return parse(yaml.safe_load(Path(path).read_text()))


def model_key(model_id: str) -> str:
    return MODEL_KEYS.get(model_id, model_id)
