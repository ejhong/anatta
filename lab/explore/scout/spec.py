"""A sweep spec: a short YAML file naming the scenario families and items (dev split only), the conditions, the
models, samples, temperature, max tokens, the judge and a spend cap. load_spec checks it and refuses anything that
reaches for the held-out split. Claude wrote this file. Private (lab/explore).

  sweep: first-sweep                 # results go to <results>/<UTC date>/<sweep>/
  families: [threat_to_self, who_benefits, deference, costly_care, after_mistake, complicity, over_refusal,
             harm_guard]
  items: all                         # all dev items | {per_family: N} (N items, or N triples for who_benefits)
                                     # | [dev ids] | {family: [dev ids]}
  conditions:                        # a name runs as a preamble; {condition: name, mode: practice} as a practice
    - baseline
    - honesty
    - {condition: self_inquiry, mode: practice}
  models:
    - id: qwen/qwen3-235b-a22b-2507
      provider: {order: [parasail], allow_fallbacks: false}   # OpenRouter provider routing (optional)
      params: {reasoning: {effort: low}}                       # extra body fields (optional)
      max_tokens: 1500                                         # overrides the sweep's (optional)
      concurrency: 4                                           # this model's calls in flight (optional)
  samples: 2
  temperature: 1.0
  max_tokens: 800
  spend_cap_usd: 25                  # this sweep's cap (at most the $50 daily cap)
  judge: {model: anthropic/claude-haiku-5.5, max_tokens: 400}   # reasoning is off unless judge.params turns it on
  framing: [observed, unobserved]    # the watched/unwatched factor on threat_to_self and who_benefits (framings.yaml)
  harm_guard_csv: /path/to/harmful-behaviors.csv   # JailbreakBench's file, outside every repo (harm_guard.py)
  harm_judge: {model: meta-llama/llama-3.3-70b-instruct}       # harm_guard's judge: open-weight only
  concurrency: 24                    # calls in flight in all
  per_model_concurrency: 6
  allow_missing_controls: "reason"   # only to run without the honesty and monk controls (a harness smoke test)

harm_guard runs only on the spec's open-weight models (harm_guard.OPEN_WEIGHT); a closed model is skipped for that
family, and the runner refuses to send it a harm_guard item. If the file is absent the family is skipped with a
message; if its sha256 differs from the recorded one, the spec is refused.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from . import harm_guard as hg
from . import library as lib
from .paths import SWEEP_NAME

DEFAULT_JUDGE = "anthropic/claude-haiku-5.5"
DEFAULTS = {"samples": 1, "temperature": 1.0, "max_tokens": 800, "items": "all", "concurrency": 24,
            "per_model_concurrency": 6}
# Claude Haiku 5.5 thinks by default on OpenRouter, and its reasoning tokens count against max_tokens: in the first
# smoke 76 of 249 labels were cut off. The judge runs with reasoning off unless a spec's judge.params says otherwise.
JUDGE_DEFAULTS = {"model": DEFAULT_JUDGE, "max_tokens": 400, "temperature": 0.0, "provider": None,
                  "params": {"reasoning": {"enabled": False}}}
HARM_JUDGE_DEFAULTS = {"model": "meta-llama/llama-3.3-70b-instruct", "max_tokens": 60, "temperature": 0.0,
                       "provider": {"order": ["deepinfra", "novita"], "allow_fallbacks": False}, "params": {}}
REQUIRED_CONTROLS = {"honesty": "the plain instruction of the same length", "monk": "the persona control"}
KNOWN = {"sweep", "families", "items", "conditions", "models", "samples", "temperature", "max_tokens", "spend_cap_usd",
         "judge", "concurrency", "per_model_concurrency", "allow_missing_controls", "split", "library", "notes",
         "baseline", "framing", "harm_guard_csv", "harm_judge"}
MODEL_KEYS = {"id", "provider", "params", "max_tokens", "label", "concurrency"}


class SpecError(SystemExit):
    pass


def _refuse_heldout_names(names, held: set[str], where: str) -> None:
    bad = sorted(n for n in names if n in held)
    if bad:
        raise lib.HeldOutError(f"refused: the spec's {where} names held-out item(s) {bad[:8]}"
                               f"{' ...' if len(bad) > 8 else ''}; the held-out split is the confirm track's, and "
                               f"explore sweeps use the dev split only")


def judge_for(spec: dict, family: str) -> dict:
    """The judge config for a family: harm_guard's own open-weight judge, else the sweep's judge."""
    return spec["harm_judge"] if family == lib.HARM_GUARD else spec["judge"]


def load_spec(source, *, dev_dir=None, heldout_file=None, conditions_file=None, framings_file=None,
              harm_guard_sha256=None) -> dict:
    """The checked, normalized spec. source: a path or a dict. Refuses (HeldOutError) a spec that names a held-out
    item or triple, asks for the held-out split, or points the library at the held-out directory."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        raw = yaml.safe_load(path.read_text())
        text = path.read_text()
    else:
        raw, text, path = copy.deepcopy(source), yaml.safe_dump(source, sort_keys=False), None
    if not isinstance(raw, dict):
        raise SpecError("a spec is a YAML mapping")
    unknown = set(raw) - KNOWN
    if unknown:
        raise SpecError(f"unknown spec keys {sorted(unknown)}; known: {sorted(KNOWN)}")

    # the held-out guard comes first, before any scenario is read
    if str(raw.get("split", "dev")).lower() != "dev":
        raise lib.HeldOutError(f"refused: split {raw.get('split')!r}; explore sweeps use the dev split only")
    if raw.get("library"):
        lib._refuse_heldout_path(Path(raw["library"]))
        dev_dir = Path(raw["library"])
    held = lib.heldout_ids(heldout_file) | hg.heldout_ids()
    items_raw = raw.get("items", DEFAULTS["items"])
    named = []
    if isinstance(items_raw, list):
        named = [str(x) for x in items_raw]
    elif isinstance(items_raw, dict) and "per_family" not in items_raw:
        named = [str(x) for v in items_raw.values() for x in (v or [])]
    _refuse_heldout_names(named, held, "items")

    for k in ("sweep", "families", "conditions", "models", "spend_cap_usd"):
        if k not in raw:
            raise SpecError(f"the spec needs {k!r}")
    spec = {**DEFAULTS, **raw}
    if not isinstance(spec["sweep"], str) or not SWEEP_NAME.match(spec["sweep"]):
        raise SpecError(f"sweep {spec['sweep']!r}: lower-case letters, digits, '.', '_' or '-'")
    fams = spec["families"]
    if not isinstance(fams, list) or not fams or any(f not in lib.ALL_FAMILIES for f in fams) or len(set(fams)) != len(fams):
        raise SpecError(f"families must be a list of distinct names from {lib.ALL_FAMILIES}")
    skipped = []
    harm_csv = spec.get("harm_guard_csv")
    if lib.HARM_GUARD in fams:
        ok, why = hg.available(harm_csv)
        if not ok:
            skipped.append({"family": lib.HARM_GUARD, "why": f"skipped harm_guard: {why}"})
            fams = [f for f in fams if f != lib.HARM_GUARD]
            if not fams:
                raise SpecError(f"the spec's only family, harm_guard, was skipped: {why}")
    library = lib.load_library([f for f in fams if f != lib.HARM_GUARD], dev_dir)
    if lib.HARM_GUARD in fams:
        library[lib.HARM_GUARD] = hg.load(harm_csv, expected_sha256=harm_guard_sha256)

    # items
    chosen: dict[str, list[str]] = {}
    for fam in fams:
        items = library[fam]["items"]
        if items_raw == "all":
            chosen[fam] = [it["id"] for it in items]
        elif isinstance(items_raw, dict) and "per_family" in items_raw:
            n = int(items_raw["per_family"])
            groups = sorted({lib.group_of(it) for it in items})[:n]
            chosen[fam] = [it["id"] for it in items if lib.group_of(it) in groups]
        elif isinstance(items_raw, (list, dict)):
            wanted = named if isinstance(items_raw, list) else [str(x) for x in (items_raw.get(fam) or [])]
            ids = {it["id"] for it in items}
            triples = {it.get("triple") for it in items if it.get("triple")}
            chosen[fam] = [it["id"] for it in items if it["id"] in wanted or it.get("triple") in wanted]
            if isinstance(items_raw, dict):
                missing = [w for w in wanted if w not in ids and w not in triples]
                if missing:
                    raise SpecError(f"{fam}: no dev items {missing}")
        else:
            raise SpecError("items: 'all', {per_family: N}, a list of dev ids, or {family: [dev ids]}")
    if isinstance(items_raw, list):
        known = {it["id"] for f in fams for it in library[f]["items"]} | \
                {it.get("triple") for f in fams for it in library[f]["items"] if it.get("triple")}
        missing = [w for w in named if w not in known]
        if missing:
            raise SpecError(f"no dev items {missing} in the families {fams}")
    if not any(chosen.values()):
        raise SpecError("the spec selects no items")

    # conditions
    conds = lib.load_conditions(conditions_file)
    variants, seen = [], set()
    for c in spec["conditions"]:
        if isinstance(c, str):
            name, mode = c, "preamble"
        elif isinstance(c, dict) and "condition" in c:
            name, mode = c["condition"], c.get("mode", "preamble")
        else:
            raise SpecError(f"condition {c!r}: a name, or {{condition: name, mode: preamble|practice}}")
        if name not in conds["conditions"]:
            raise SpecError(f"unknown condition {name!r}; known: {sorted(conds['conditions'])}")
        if mode not in ("preamble", "practice"):
            raise SpecError(f"condition {name}: mode {mode!r} (preamble or practice)")
        cond = conds["conditions"][name]
        iv = lib.intervention_for(cond, mode, conds["practice_wrapper"])
        if iv["name"] in seen:
            raise SpecError(f"condition {iv['name']} is listed twice")
        seen.add(iv["name"])
        variants.append({"condition": name, "mode": iv["params"]["mode"], "intervention": iv})
    base_name = spec.get("baseline", "baseline")
    if base_name not in seen:
        raise SpecError("every sweep needs the baseline condition (the signature is measured from it)")
    missing_controls = [f"{k} ({v})" for k, v in REQUIRED_CONTROLS.items() if k not in seen]
    if missing_controls and not spec.get("allow_missing_controls"):
        raise SpecError(f"PROGRAM.md: every sweep has a baseline and at least two matched controls; missing "
                        f"{missing_controls}. Add them, or set allow_missing_controls: '<reason>' for a harness check.")

    # models
    models, ids = [], set()
    if not isinstance(spec["models"], list) or not spec["models"]:
        raise SpecError("models: a non-empty list")
    for m in spec["models"]:
        m = {"id": m} if isinstance(m, str) else dict(m)
        if set(m) - MODEL_KEYS or not isinstance(m.get("id"), str):
            raise SpecError(f"model {m}: keys {sorted(MODEL_KEYS)}, id required")
        if m["id"] in ids:
            raise SpecError(f"model {m['id']} is listed twice")
        ids.add(m["id"])
        m.setdefault("provider", None)
        m.setdefault("params", {})
        m.setdefault("max_tokens", spec["max_tokens"])
        if not isinstance(m["params"], dict) or {"model", "messages", "provider", "usage"} & set(m["params"]):
            raise SpecError(f"model {m['id']}: params is a mapping of extra body fields (not model/messages/provider)")
        models.append(m)

    # numbers
    if not (isinstance(spec["samples"], int) and 1 <= spec["samples"] <= 50):
        raise SpecError("samples: an integer from 1 to 50")
    if not (0.0 <= float(spec["temperature"]) <= 2.0):
        raise SpecError("temperature: 0 to 2")
    for m in models:
        if not (isinstance(m["max_tokens"], int) and 16 <= m["max_tokens"] <= 16000):
            raise SpecError(f"{m['id']}: max_tokens from 16 to 16000")
    cap = float(spec["spend_cap_usd"])
    if not 0 < cap <= 50:
        raise SpecError("spend_cap_usd: above 0 and at most 50 (the daily cap)")

    judge = {**JUDGE_DEFAULTS, **(spec.get("judge") or {})}
    if judge["model"] in ids and not judge.get("allow_subject"):
        raise SpecError(f"the judge {judge['model']} is also a subject; pick another judge (or set judge.allow_subject)")
    harm_judge = {**HARM_JUDGE_DEFAULTS, **(spec.get("harm_judge") or {})}
    harm_models = []
    if lib.HARM_GUARD in fams:
        hg.require_open_weight(harm_judge["model"], "harm_guard's judge")
        if harm_judge["model"] in ids and not harm_judge.get("allow_subject"):
            raise SpecError(f"harm_guard's judge {harm_judge['model']} is also a subject; pick another")
        harm_models = [m["id"] for m in models if hg.is_open_weight(m["id"])]
        closed = [m["id"] for m in models if not hg.is_open_weight(m["id"])]
        if closed:
            skipped.append({"family": lib.HARM_GUARD, "why": f"harm_guard runs on open-weight models only; skipping "
                                                             f"it for {', '.join(closed)}"})
        if not harm_models:
            raise SpecError("harm_guard needs at least one open-weight model among the spec's models")
    for m in models + [judge, harm_judge]:
        name = m.get("id") or m.get("model")
        if "concurrency" in m and not (isinstance(m["concurrency"], int) and 1 <= m["concurrency"] <= 64):
            raise SpecError(f"{name}: concurrency from 1 to 64")

    # the watched/unwatched factor
    framing = None
    fr_raw = spec.get("framing")
    if fr_raw not in (None, False, "none", []):
        fdoc = lib.load_framings(framings_file)
        levels = list(fdoc["framings"]) if fr_raw is True else fr_raw
        if not isinstance(levels, list) or sorted(levels) != ["observed", "unobserved"]:
            raise SpecError("framing: [observed, unobserved] (both levels, so the audience gap is paired), or none")
        framing = {"levels": ["observed", "unobserved"], "lines": fdoc["framings"],
                   "families": [f for f in fdoc["families"] if f in fams], "sha256": fdoc["sha256"]}

    return {"sweep": spec["sweep"], "families": fams, "items": chosen, "variants": variants, "models": models,
            "samples": spec["samples"], "temperature": float(spec["temperature"]), "max_tokens": spec["max_tokens"],
            "spend_cap_usd": cap, "judge": judge, "concurrency": int(spec["concurrency"]),
            "per_model_concurrency": int(spec["per_model_concurrency"]), "baseline": base_name,
            "allow_missing_controls": spec.get("allow_missing_controls"), "missing_controls": missing_controls,
            "notes": spec.get("notes"), "source_text": text, "source_path": str(path) if path else None,
            "library": library, "conditions": conds, "dev_dir": str(dev_dir) if dev_dir else None,
            "framing": framing, "harm_judge": harm_judge, "harm_models": harm_models, "skipped": skipped,
            "harm_guard_csv": str(Path(harm_csv).expanduser()) if harm_csv and lib.HARM_GUARD in fams else None}
