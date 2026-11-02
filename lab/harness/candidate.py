"""A candidate: a model plus an intervention, frozen by sha256 in one YAML file (spec v2 section 2).
Claude (Opus 5.5) wrote this file, 2 Nov 2026.

YAML form (see README.md and fixtures/noself3/candidate.yaml):

  name: noself3-nofix                       # [a-z0-9._-]+, used as the results folder name
  model: {id: mistralai/Ministral-3-8B-Instruct-2512-BF16, revision: f6fae97..., source: hf_cache}
                                            # or {id: <openrouter id>, source: openrouter, provider: <pinned provider>}
  intervention:                             # kind: prompt | steer | ablate | adapter | checkpoint
    kind: steer
    direction: {name: nofix_deny, tensor_sha256: <hex>, path: <optional file>}
    layers: [17]
    position: all                           # all | last | from_end:-2 | prompt_only (the position rule)
    strength: 1.416
  topic_direction: {name: nofix_topic, tensor_sha256: <hex>}   # optional (steer/ablate): adds a topic control
  controls: default                         # default (filled per kind, below) or an explicit list
  tier: quick                               # quick | full
  measures: default                         # default (every registry measure the tier allows) or a list of ids
  seeds: [0]
  notes: free text

Default controls [F5]:
  steer/ablate        random_matched (variance-matched random direction), other_trait (another trait's own vector, not
                      about the self), split_null (C4's split null), plus topic when topic_direction is given;
  prompt              neutral (length-matched neutral prompt), persona (a persona prompt), honesty (a plain honesty
                      instruction): the double check's rivals;
  adapter/checkpoint  twin (trained the same way without the active ingredient), at least 3 seeds per arm.

Freezing: candidate_sha256 is the sha256 of the canonical JSON of the validated candidate with defaults filled; the
file's own sha256 is kept beside it. A full-tier read refuses a candidate whose sha differs from the registered one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

KINDS = ("prompt", "steer", "ablate", "adapter", "checkpoint")
SOURCES = ("hf_cache", "openrouter")
TIERS = ("quick", "full")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
POSITION = re.compile(r"^(all|last|prompt_only|from_end:-[1-9][0-9]*)$")
MIN_TWIN_SEEDS = 3


class CandidateError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    name: str
    model: dict
    intervention: dict
    controls: tuple
    tier: str
    measures: object  # "default" or tuple of ids
    seeds: tuple
    topic_direction: dict | None = None
    notes: str = ""
    file_sha256: str | None = None
    sha256: str = field(default="")

    @property
    def kind(self) -> str:
        return self.intervention["kind"]

    def as_dict(self) -> dict:
        return {"name": self.name, "model": self.model, "intervention": self.intervention,
                "controls": [dict(c) for c in self.controls], "tier": self.tier,
                "measures": self.measures if self.measures == "default" else list(self.measures),
                "seeds": list(self.seeds), "topic_direction": self.topic_direction, "notes": self.notes}

    def conditions(self) -> list[dict]:
        """The arms a run has: base, the candidate, then each control (role, name, spec)."""
        return ([{"role": "base", "name": "base", "spec": {"kind": "none"}},
                 {"role": "candidate", "name": "candidate", "spec": self.intervention}]
                + [{"role": "control", "name": c["name"], "spec": c} for c in self.controls])


def canonical_sha256(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _direction(d, where: str) -> dict:
    if not isinstance(d, dict) or "name" not in d:
        raise CandidateError(f"{where}: a direction is {{name, tensor_sha256, path?}}")
    if not HEX64.match(str(d.get("tensor_sha256", ""))):
        raise CandidateError(f"{where}: tensor_sha256 must pin the direction (64 hex characters)")
    return {k: d[k] for k in ("name", "tensor_sha256", "path") if k in d}


def _layers(x, where: str) -> list[int]:
    if not isinstance(x, list) or not x or not all(isinstance(i, int) and i >= 0 for i in x):
        raise CandidateError(f"{where}: layers must be a non-empty list of block indices")
    return sorted(set(x))


def _intervention(iv: dict) -> dict:
    if not isinstance(iv, dict) or iv.get("kind") not in KINDS:
        raise CandidateError(f"intervention.kind must be one of {KINDS}")
    k, out = iv["kind"], {"kind": iv["kind"]}
    if k == "prompt":
        if not (iv.get("system_prompt") or iv.get("prefix")):
            raise CandidateError("a prompt intervention needs system_prompt or prefix")
        for f in ("system_prompt", "prefix"):
            if iv.get(f):
                out[f] = str(iv[f])
    elif k in ("steer", "ablate"):
        out["direction"] = _direction(iv.get("direction"), "intervention.direction")
        out["layers"] = _layers(iv.get("layers"), "intervention.layers")
        pos = str(iv.get("position", "all"))
        if not POSITION.match(pos):
            raise CandidateError(f"intervention.position {pos!r}: all | last | prompt_only | from_end:-N")
        out["position"] = pos
        if k == "steer":
            if not isinstance(iv.get("strength"), (int, float)) or iv["strength"] == 0:
                raise CandidateError("a steer intervention needs a non-zero numeric strength")
            out["strength"] = float(iv["strength"])
    else:  # adapter / checkpoint
        for f in ("path", "sha256"):
            if not iv.get(f):
                raise CandidateError(f"an {k} intervention needs {f}")
        if not HEX64.match(str(iv["sha256"])):
            raise CandidateError(f"{k}.sha256 must be 64 hex characters")
        twin = iv.get("twin")
        if not isinstance(twin, dict) or not twin.get("path") or not HEX64.match(str(twin.get("sha256", ""))):
            raise CandidateError(f"an {k} needs its twin {{path, sha256}} (trained the same way without the active "
                                 "ingredient)")
        seeds = iv.get("seeds") or []
        if len(seeds) < MIN_TWIN_SEEDS or len(twin.get("seeds") or []) < MIN_TWIN_SEEDS:
            raise CandidateError(f"an {k} and its twin need at least {MIN_TWIN_SEEDS} seeds per arm")
        out.update({"path": str(iv["path"]), "sha256": iv["sha256"], "seeds": list(seeds),
                    "twin": {"path": str(twin["path"]), "sha256": twin["sha256"], "seeds": list(twin["seeds"])}})
        if iv.get("base_model"):
            out["base_model"] = str(iv["base_model"])
    return out


def default_controls(iv: dict, topic: dict | None = None) -> list[dict]:
    k = iv["kind"]
    if k in ("steer", "ablate"):
        common = {"kind": k, "layers": iv["layers"], "position": iv["position"]}
        if k == "steer":
            common["strength"] = iv["strength"]  # the runner matches norms (random_matched) or variance (other_trait)
        target = iv["direction"]["name"]
        out = [{"name": "random_matched", **common, "direction": "random_matched", "target": target, "seeds": [0]},
               {"name": "other_trait", **common, "direction": "other_trait", "target": target,
                "traits": ["formality"], "match": "partial_moment"},
               {"name": "split_null", **common, "direction": "split_null", "target": target, "seeds": [0]}]
        if topic:
            out.append({"name": "topic", **common, "direction": dict(topic)})
        return out
    if k == "prompt":
        slot = "system_prompt" if iv.get("system_prompt") else "prefix"
        n = len(iv.get(slot, ""))
        return [{"name": "neutral", "kind": "prompt", "slot": slot, "rule": "length_matched_neutral", "chars": n},
                {"name": "persona", "kind": "prompt", "slot": slot, "rule": "persona", "chars": n},
                {"name": "honesty", "kind": "prompt", "slot": slot, "rule": "plain_honesty_instruction"}]
    twin = iv["twin"]
    return [{"name": "twin", "kind": k, "path": twin["path"], "sha256": twin["sha256"], "seeds": twin["seeds"]}]


def validate(raw: dict, file_sha256: str | None = None) -> Candidate:
    if not isinstance(raw, dict):
        raise CandidateError("a candidate is a YAML mapping")
    name = str(raw.get("name", ""))
    if not NAME.match(name):
        raise CandidateError("name must match [a-z0-9][a-z0-9._-]*")
    model = raw.get("model")
    if not isinstance(model, dict) or not model.get("id") or model.get("source") not in SOURCES:
        raise CandidateError(f"model needs id and source ({SOURCES})")
    if model["source"] == "hf_cache" and not re.match(r"^[0-9a-f]{40}$", str(model.get("revision", ""))):
        raise CandidateError("an hf_cache model needs its 40-hex revision")
    if model["source"] == "openrouter" and not model.get("provider"):
        raise CandidateError("an OpenRouter model needs a pinned provider")
    iv = _intervention(raw.get("intervention"))
    if model["source"] == "openrouter" and iv["kind"] != "prompt":
        raise CandidateError("OpenRouter candidates are prompt-only")
    topic = raw.get("topic_direction")
    if topic is not None:
        if iv["kind"] not in ("steer", "ablate"):
            raise CandidateError("topic_direction applies to steer/ablate only")
        topic = _direction(topic, "topic_direction")
    controls = raw.get("controls", "default")
    if controls in (None, "default"):
        controls = default_controls(iv, topic)
    elif isinstance(controls, list):
        seen = set()
        for c in controls:
            if not isinstance(c, dict) or not c.get("name") or c["name"] in seen or c["name"] in ("base", "candidate"):
                raise CandidateError("each explicit control needs a unique name (not base/candidate)")
            seen.add(c["name"])
        controls = copy.deepcopy(controls)
    else:
        raise CandidateError("controls: 'default' or a list")
    tier = raw.get("tier", "quick")
    if tier not in TIERS:
        raise CandidateError(f"tier must be one of {TIERS}")
    measures = raw.get("measures", "default")
    if measures != "default":
        if not isinstance(measures, list) or not measures:
            raise CandidateError("measures: 'default' or a list of registry ids")
        measures = tuple(measures)
    seeds = tuple(int(s) for s in (raw.get("seeds") or [0]))
    unknown = set(raw) - {"name", "model", "intervention", "controls", "tier", "measures", "seeds", "topic_direction",
                          "notes"}
    if unknown:
        raise CandidateError(f"unknown fields: {sorted(unknown)}")
    c = Candidate(name=name, model={k: model[k] for k in sorted(model)}, intervention=iv,
                  controls=tuple(controls), tier=tier, measures=measures, seeds=seeds, topic_direction=topic,
                  notes=str(raw.get("notes", "")), file_sha256=file_sha256)
    return Candidate(**{**c.__dict__, "sha256": canonical_sha256(c.as_dict())})


def load(path: Path | str) -> Candidate:
    data = Path(path).read_bytes()
    return validate(yaml.safe_load(data), hashlib.sha256(data).hexdigest())


def check_frozen(c: Candidate, registered_sha256: str) -> None:
    """Full tier: the candidate must be the one registered before the held-out read."""
    if c.sha256 != registered_sha256:
        raise CandidateError(f"candidate {c.name} sha256 {c.sha256[:12]} differs from the registered "
                             f"{registered_sha256[:12]}: not the frozen candidate")


def resolve_measures(c: Candidate, registry: dict) -> list[str]:
    """The measure ids this candidate's tier runs: 'default' is every registry measure the tier allows (judge-only
    entries excluded); an explicit list must name allowed measures."""
    if c.measures == "default":
        return [m for m, x in registry.items() if x.allowed(c.tier)]
    bad = [m for m in c.measures if m not in registry or not registry[m].allowed(c.tier)]
    if bad:
        raise CandidateError(f"measures not in the registry or not allowed at tier {c.tier}: {bad}")
    return list(c.measures)
