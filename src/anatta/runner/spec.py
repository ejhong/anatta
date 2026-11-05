"""An experiment spec (YAML under experiments/): loaded, validated, and expanded into conditions. Claude (Opus 5.5)
wrote this file, 4 Nov 2026 (clean-lab step 5).

The schema (every key but reproduction and about is required):

  name: k6b_orbench_subsample     the run folders go under <results_root>/runs/<name>/
  tier: explore | full            full: registered in lab/ledger/NOTES.md and run from a clean tree (refused
                                  otherwise); explore: neither, the commit and dirty flag are recorded
  about: one paragraph            what the run is for (free text)
  models: [ministral3_8b]         model-registry keys (src/anatta/backends/registry.yaml), served by vLLM
  organisms: []                   organism-registry keys (src/anatta/organisms/registry.yaml); an entry null or
                                  "none" is the bare model. Empty: the bare models only. Each organism runs on its
                                  own base, which must be one of models; it is verified (re-hashed) before use.
  interventions: [none]           each "none", or {kind: system_prompt, text: <name in lab/explore/conditions/
                                  conditions.yaml>}; steer and ablate are named hooks, not built
  measures:                       measures-registry keys (src/anatta/measures/registry.yaml) the runner implements
    orbench_hard_rejection: {n: 660, ...}   the measure's own options (n: items; see its module)
  seeds: [0]                      run seeds; every condition with the same seed gets the same per-request seeds
  caps: {usd: 3.0, batch: 330}    usd: the whole run's spend cap (at most lab.toml's cap for the tier); batch:
                                  requests per generation call, the size the trial runs at
  reproduction:                   for reproductions: what the card is graded against
    measure: orbench_hard_rejection
    statistic: rejection          a key of the measure's score
    condition: <id>               optional when there is one condition
    target: 0.148 | {card: results/cards/k6b.json, path: orbench.rejection} | reference
    band: [lo, hi] | {card: ..., path: orbench.ci95} | reference   in band: the observed statistic is within [lo, hi]
    reference:                    optional: a stored run of the same items (same-item grading); "reference" above
      verdicts: {file: <path>, sha256: <pin>}   means the stored run's rate on this run's item ids, and its Wilson 95%
      replies: {file: <path>, sha256: <pin>}    interval; the card adds the paired discordance and, with replies, the
                                                byte-identical reply count. The measure reads the files
                                                (reference_units, reference_replies).
    full_set: {target: ..., band: ...}   optional: shown beside the grade, not graded (e.g. the card's full-set
                                         numbers when the grade is same-item)

Every file the reproduction names (cards and reference files) is hashed into plan.json; the card stage refuses if
one changed, and a reference file whose sha256 differs from its pin is refused at plan time.

A condition is one (model, organism, intervention, seed); its id is "<model>__<organism or base>__<intervention>__s<seed>".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO

KEYS = {"name", "tier", "about", "models", "organisms", "interventions", "measures", "seeds", "caps", "reproduction"}
REQUIRED = KEYS - {"about", "reproduction"}
TIERS = ("explore", "full")
TIER_CAP = {"explore": "quick_cap_usd", "full": "full_cap_usd"}  # lab.toml [spend]
NONE = (None, "none", "base")


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    model: str
    organism: str | None
    intervention: dict
    seed: int

    @property
    def id(self) -> str:
        return f"{self.model}__{self.organism or 'base'}__{intervention_name(self.intervention)}__s{self.seed}"


@dataclass
class Spec:
    path: Path
    sha256: str  # of the file's bytes: what NOTES.md registers
    raw: dict
    conditions: list[Condition] = field(default_factory=list)

    def __getattr__(self, k):
        if k in KEYS:
            return self.raw.get(k)
        raise AttributeError(k)

    @property
    def rel(self) -> str:
        try:
            return str(self.path.resolve().relative_to(REPO))
        except ValueError:
            return str(self.path)


def intervention_name(iv: dict) -> str:
    if iv["kind"] == "none":
        return "none"
    return f"{iv['kind']}-{iv.get('text') or iv.get('name') or 'x'}"


def _norm_iv(iv) -> dict:
    if iv in (None, "none"):
        return {"kind": "none"}
    if isinstance(iv, str):
        raise SpecError(f"intervention {iv!r}: write none, or a mapping {{kind: system_prompt, text: <name>}}")
    if not isinstance(iv, dict) or "kind" not in iv:
        raise SpecError(f"intervention {iv!r} needs a kind")
    from .interventions import KINDS

    if iv["kind"] not in KINDS:
        raise SpecError(f"intervention kind {iv['kind']!r} is not one of {sorted(KINDS)}")
    return dict(iv)


def _organism_base(key: str) -> str:
    from .. import organisms

    e = organisms.get(key)
    base = e.get("base") if isinstance(e, dict) else getattr(e, "base", None)
    kind = e.get("kind", "lora") if isinstance(e, dict) else getattr(e, "kind", "lora")
    if kind != "lora":
        raise SpecError(f"organism {key} is kind {kind!r}; the runner serves LoRA adapters only (a baked organism "
                        "needs its own model-registry entry)")
    if not base:
        raise SpecError(f"organism {key}: its registry entry names no base model")
    return base


def validate(raw: dict, *, organism_base=_organism_base) -> list[Condition]:
    """The spec's conditions after checking every key; SpecError names the first problem."""
    from ..backends import registry as models
    from ..config import load
    from . import measures as M

    if not isinstance(raw, dict):
        raise SpecError("a spec is a YAML mapping")
    unknown, missing = set(raw) - KEYS, REQUIRED - set(raw)
    if unknown:
        raise SpecError(f"unknown keys {sorted(unknown)} (the schema: {sorted(KEYS)})")
    if missing:
        raise SpecError(f"missing keys {sorted(missing)}")
    if not isinstance(raw["name"], str) or not raw["name"].replace("_", "").replace("-", "").isalnum():
        raise SpecError(f"name {raw['name']!r}: letters, digits, _ and - only (it names a folder)")
    if raw["tier"] not in TIERS:
        raise SpecError(f"tier {raw['tier']!r} is not one of {TIERS}")
    ms = raw["models"]
    if not ms or not isinstance(ms, list):
        raise SpecError("models: a non-empty list of model-registry keys")
    for m in ms:
        try:
            entry = models.get(m)
        except KeyError as e:
            raise SpecError(str(e)) from None
        if entry.key != m:
            raise SpecError(f"model {m!r} is an alias; write its key {entry.key!r}")
        if "vllm" not in entry.backends:
            raise SpecError(f"model {m} is not served by vLLM here (backends {entry.backends})")
    orgs = raw["organisms"] or []
    if not isinstance(orgs, list):
        raise SpecError("organisms: a list (empty for the bare models)")
    ivs = [_norm_iv(iv) for iv in (raw["interventions"] or [])]
    if not ivs:
        raise SpecError("interventions: at least one (none for no intervention)")
    if len({intervention_name(iv) for iv in ivs}) != len(ivs):
        raise SpecError("two interventions share a name")
    meas = raw["measures"]
    if not isinstance(meas, dict) or not meas:
        raise SpecError("measures: a mapping {measure key: its options}")
    for k, opts in meas.items():
        M.get(k)  # raises SpecError for an unknown or unimplemented key
        if opts is not None and not isinstance(opts, dict):
            raise SpecError(f"measure {k}: its options are a mapping")
    seeds = raw["seeds"]
    if not seeds or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds) or len(set(seeds)) != len(seeds):
        raise SpecError("seeds: a non-empty list of distinct integers")
    caps = raw["caps"] or {}
    if set(caps) - {"usd", "batch"} or not {"usd", "batch"} <= set(caps):
        raise SpecError("caps: exactly usd and batch")
    usd, batch = caps["usd"], caps["batch"]
    if not isinstance(usd, (int, float)) or not usd > 0:
        raise SpecError("caps.usd must be above 0")
    tier_cap = getattr(load(), TIER_CAP[raw["tier"]])
    if usd > tier_cap:
        raise SpecError(f"caps.usd {usd} is above lab.toml's {TIER_CAP[raw['tier']]} {tier_cap} for tier {raw['tier']}")
    if not isinstance(batch, int) or batch < 1:
        raise SpecError("caps.batch: a positive integer (requests per generation call)")

    conds = []
    for model in ms:
        for org in (orgs or [None]):
            org = None if org in NONE else org
            if org is not None:
                try:
                    base = organism_base(org)
                except (KeyError, ImportError, AttributeError) as e:
                    raise SpecError(f"organism {org!r}: {e}") from None
                if base not in ms:
                    raise SpecError(f"organism {org} runs on {base}, which is not in models {ms}")
                if base != model:
                    continue
            for iv in ivs:
                for s in seeds:
                    conds.append(Condition(model, org, iv, s))
    if not conds:
        raise SpecError("no condition: every organism's base is missing from models")
    if raw.get("reproduction") is not None:
        _check_reproduction(raw["reproduction"], meas, conds)
    return conds


def _check_reproduction(rep: dict, meas: dict, conds: list[Condition]) -> None:
    if not isinstance(rep, dict) or not {"measure", "statistic", "target", "band"} <= set(rep):
        raise SpecError("reproduction: measure, statistic, target and band (condition when there is more than one)")
    if rep["measure"] not in meas:
        raise SpecError(f"reproduction.measure {rep['measure']} is not one of the spec's measures")
    ids = [c.id for c in conds]
    if rep.get("condition") is None and len(ids) > 1:
        raise SpecError(f"reproduction.condition: name one of {ids}")
    if rep.get("condition") is not None and rep["condition"] not in ids:
        raise SpecError(f"reproduction.condition {rep['condition']!r} is not one of {ids}")
    unknown = set(rep) - {"measure", "statistic", "target", "band", "condition", "reference", "full_set"}
    if unknown:
        raise SpecError(f"reproduction: unknown keys {sorted(unknown)}")
    ref = rep.get("reference")
    if ref is not None:
        if not isinstance(ref, dict) or "verdicts" not in ref or set(ref) - {"verdicts", "replies"}:
            raise SpecError("reproduction.reference: verdicts (and optionally replies), each {file, sha256}")
        for k, v in ref.items():
            if not isinstance(v, dict) or set(v) != {"file", "sha256"}:
                raise SpecError(f"reproduction.reference.{k}: {{file, sha256}}")
    for k in ("target", "band"):
        if rep[k] == "reference":
            if ref is None:
                raise SpecError(f"reproduction.{k} is reference but the reproduction names no reference")
        elif k == "band":
            lo, hi = resolve_band(rep["band"])
            if not lo <= hi:
                raise SpecError(f"reproduction.band [{lo}, {hi}] is empty")
        else:
            resolve_value(rep["target"])
    fs = rep.get("full_set")
    if fs is not None:
        if not isinstance(fs, dict) or set(fs) != {"target", "band"}:
            raise SpecError("reproduction.full_set: target and band")
        resolve_value(fs["target"])
        resolve_band(fs["band"])


def referenced_files(rep: dict | None) -> dict:
    """{label: absolute path} of every file a reproduction names: cards (by their repo path) and reference files."""
    out: dict = {}
    if not rep:
        return out

    def card(v):
        if isinstance(v, dict) and "card" in v:
            out[v["card"]] = REPO / v["card"]

    for v in (rep.get("target"), rep.get("band"), *((rep.get("full_set") or {}).values())):
        card(v)
    for k, v in (rep.get("reference") or {}).items():
        out[f"reference.{k}"] = Path(str(v["file"])).expanduser()
    return out


def resolve_value(v):
    """A number, or {card: <repo-relative card JSON>, path: <dotted key>} read from that card."""
    if isinstance(v, (int, float, list)):
        return v
    if not isinstance(v, dict) or set(v) != {"card", "path"}:
        raise SpecError(f"{v!r}: a number, a list, or {{card, path}}")
    p = REPO / v["card"]
    if not p.exists():
        raise SpecError(f"no card at {v['card']}")
    cur = json.loads(p.read_text(encoding="utf-8"))
    for part in str(v["path"]).split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise SpecError(f"{v['card']} has no {v['path']}")
        cur = cur[part]
    return cur


def resolve_band(b) -> tuple[float, float]:
    v = resolve_value(b)
    if not isinstance(v, list) or len(v) != 2 or not all(isinstance(x, (int, float)) for x in v):
        raise SpecError(f"band {b!r}: [lo, hi]")
    return float(v[0]), float(v[1])


def load(path, **kw) -> Spec:
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = (REPO / p) if (REPO / p).exists() else p
    if not p.exists():
        raise SpecError(f"no spec at {path}")
    data = p.read_bytes()
    raw = yaml.safe_load(data.decode("utf-8"))
    spec = Spec(path=p, sha256=hashlib.sha256(data).hexdigest(), raw=raw)
    spec.conditions = validate(raw, **kw)
    return spec
