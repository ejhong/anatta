"""bench's laptop client. It reads a sweep spec, calls the warm Bench class on Modal (modal_bench.py) and writes one
record per sample in the explore track's schema (scout/schema.py), so scout's judge and report read a bench sweep as
they read a scout sweep. Claude (Opus 5.5) wrote this file, 1 Nov 2026, and extended it the same day (the other_trait
and split_null controls, calls packed by a token budget, tensor hashes). Private (lab/explore).

Run it with the matlab venv's Python, which has modal and torch (no installs):
  PY=~/prj/matlab/.venv/bin/python; B=/Users/eugene/prj/align/anatta-lab/lab/explore/bench
  $PY $B/client.py up                  deploy the app; no container starts and nothing is billed until a call
  $PY $B/client.py warm [--model qwen] start a container and load the model; prints the cold-start time
  $PY $B/client.py run SPEC.yaml       run a sweep; --cap USD overrides the spec's cap, --stop stops the container
                                       afterwards instead of leaving it warm for the next sweep
  $PY $B/client.py control [--kind random_matched|other_trait|split_null] [--target ego] [--seeds 0,1]
          [--traits calm,formality,...] [--layers 17-26] [--match partial|partial_moment]
          [--bank PATH --bank-tensor-sha256 HEX]
                                       build controls only (no generations). --target is a registry name or a
                                       file (pin it with --target-tensor-sha256 and/or --target-sha256)
  $PY $B/client.py status              the bench's containers, today's explore spend, what is paid until when
  $PY $B/client.py stop                stop the warm container now and refund the unused tail
  $PY $B/client.py down                stop and undeploy the app (`up` brings it back)
After a run the container stays warm for 10 minutes (modal_bench.SCALEDOWN_S), so the next sweep starts in seconds,
then stops by itself.

A spec (YAML):
  sweep: bench-ego-1               # results go to <results>/<UTC date>/<sweep>/
  model: ministral                 # ministral | qwen
  library: auto                    # auto (the scenario library once built, else fixtures/) | library | fixture
  families: [threat_to_self]       # dev families (default: every family the library has)
  items: {per_family: 4}           # all | {per_family: N} | [dev ids] | {family: [dev ids]}  (scout's forms)
  interventions:
    - baseline                                       # no hooks (required; "none" works too)
    - {mode: ablate, direction: ego}                 # every layer the file has: for v_ego, 8-32, the ego cut
    - {mode: ablate, direction: ego, layers: 8-20}   # a band
    - {mode: add, direction: ego, layers: [16], strengths: [-8, 8]}
    - {mode: ablate, direction: random_matched, target: ego, seeds: [0, 1]}   # variance-matched random controls
    - {mode: add, direction: random_matched, target: ego, seeds: [0], layers: [16], strengths: [8]}
    - {mode: ablate, direction: other_trait, target: ego, traits: [formality, calm]}   # other traits' own vectors at
                                                     # the target's layers; their neutral-set variance ratio is
                                                     # reported, not rescaled (controls.py)
    - {mode: ablate, direction: other_trait, target: ego, traits: [formality], match: partial}   # a layer whose
                                                     # ratio r > 1 is cut by sqrt(1/r) (the flag for a partial cut)
    - {mode: ablate, direction: other_trait, target: ego, traits: [verbosity], match: partial_moment}   # the same
                                                     # on the mean-square ratio (a cut removes the mean projection too)
    - {mode: ablate, direction: split_null, target: ego, seeds: [0, 1, 2]}   # C4's split null; with no bank, one is
                                                     # built from the ego traits' contrastive prompts (controls.py)
    - {mode: ablate, direction: split_null, target: ego, seeds: [0], bank: {path: /abs/bank.pt, tensor_sha256: <hex>}}
    - {mode: ablate, direction: {path: /abs/new.pt, tensor_sha256: <hex>, name: persona_x}}   # a file not in the
                                                     # registry: pin tensor_sha256 (and/or the file's sha256)
    - {mode: mean_ablate, direction: ego}            # the cut's two parts (steering's docstring): h.d set to its
    - {mode: shift_only, direction: ego}             # neutral-set mean (the varying part removed), the push alone,
    - {mode: shift_reverse, direction: ego}          # and the opposite push; per-layer constants from split_means,
                                                     # layer by layer with the arm's own hooks below (means:
                                                     # sequential, the default); means: independent takes every
                                                     # layer's uncut mean as it is
  samples: 2
  temperature: 0.7                 # also top_p (default 1.0) and top_k (default: the model's own)
  max_tokens: 256
  seed: 0                          # a sample's seed comes from (seed, item, sample): the same under every
                                   # intervention, so conditions are paired
  spend_cap_usd: 5                 # this session's cap; the day's $50 cap applies as well
  batch_size: 64                   # at most this many requests per call
  token_budget: 80000              # token slots per GPU pass, rows x (longest prompt + max_tokens) (steering.py)
  allow_missing_controls: "why"    # only to run without a matched control (a harness check)
A control condition (random_matched, other_trait, split_null) cuts at its target's layers unless it names `layers`.
A split arm (mean_ablate, shift_only, shift_reverse) takes a plain direction, not a control, and no strength. Before
any generation one split_means call per (direction, layers) finds its constants on the neutral set; the uncut mean mu_l
and sd_l of h.d at every layer, each arm's constants and what each arm does to h.d there go into run.json
(bench.split), the records' params and directions/split-*.json.

Calls. Jobs go to the container in order, a call holding at most batch_size requests and at most MAX_CALL_STEPS
decoding steps (its planned passes x max_tokens, planned with steering.plan_passes on prompt lengths estimated from
characters), so a call of long prompts stays far below Modal's 1200 s timeout. Inside the container each intervention
group runs in passes within token_budget, which is what keeps long prompts in memory.

A sweep directory holds, as scout's does: records.jsonl (rewritten at the end), records.partial.jsonl (as records
land), errors.jsonl, spend.jsonl, run.json (the manifest scout's report reads), and directions/: each random_matched
and split_null control as an anatta Direction file beside a JSON of how it was made, each other_trait control's
neutral-set table as JSON, and a split_null bank built in the run (its .pt, its JSON, its replies). Every condition is
logged to <results>/variants.jsonl. A record's cost_usd is its share of its group's container seconds; the session's
whole charge, cold start and warm tail included, is in spend.jsonl and the daily file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

BENCH = Path(__file__).resolve().parent
ANATTA_SRC = "/Users/eugene/prj/align/anatta-dev/src"  # anatta.probe.Direction, imported only
for p in (str(BENCH), ANATTA_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
import controls as K  # noqa: E402
import directions as D  # noqa: E402
import meter as M  # noqa: E402
import records as R  # noqa: E402
from records import lib, paths, schema  # noqa: E402  (scout's modules)

APP_NAME, CLASS_NAME = "anatta-explore-bench", "Bench"
SCALEDOWN_S = 600  # modal_bench.SCALEDOWN_S (a test keeps them equal)
CORES, MEM_GIB = 2.0, 16.0  # modal_bench.CPU and MEMORY_MB / 1024
RATE = M.rate_per_hour("L40S", CORES, MEM_GIB)
MODELS = ("ministral", "qwen")
BASELINE = "baseline"
NEUTRAL = BENCH / "neutral_prompts.json"
COLD_START_S = 300.0  # the first call's allowance: container start and the model's load from the volume
PER_TOKEN_S = 0.004  # a cautious prior for a call's seconds per requested token, until one call is measured
GEN_BATCH = 32  # steering.Service's most rows per pass
TOKEN_BUDGET = 80_000  # steering.TOKEN_BUDGET (a test keeps them equal)
MAX_CALL_STEPS = 3200  # decoding steps one call may plan (passes x max_tokens): 8 passes of 400 tokens
OTHER, SPLIT = "other_trait", "split_null"
CONTROL_KINDS = (D.RANDOM, OTHER, SPLIT)
SPLIT_MODES = ("mean_ablate", "shift_only", "shift_reverse")  # steering.SPLIT_MODES (a test keeps them equal)
SPLIT_MEANS = ("sequential", "independent")  # how a split arm's per-layer constants are found (steering's docstring)
SPLIT_PASS_S = 0.05  # a cautious prior for one neutral prompt's forward pass in split_means, for its estimate
BASE_KEYS = {"mode", "direction", "layers", "strengths", "coefficient", "name"}
SPLIT_KEYS = {"means"}
CONTROL_KEYS = {D.RANDOM: {"target", "seeds"}, OTHER: {"target", "traits", "match"},
                SPLIT: {"target", "seeds", "bank", "match"}}
KNOWN = {"sweep", "model", "library", "families", "items", "split", "interventions", "samples", "temperature",
         "top_p", "top_k", "max_tokens", "seed", "spend_cap_usd", "batch_size", "allow_missing_controls", "notes",
         "tol", "token_budget"}
DEFAULTS = {"model": "ministral", "library": "auto", "items": "all", "samples": 1, "temperature": 0.7, "top_p": 1.0,
            "top_k": None, "max_tokens": 256, "seed": 0, "batch_size": 64, "tol": 0.10, "token_budget": TOKEN_BUDGET}


class SpecError(SystemExit):
    pass


# ----------------------------------------------------------------------------------------------- the spec


def load_spec(source) -> dict:
    """The checked spec, with its items chosen (records.select_items, which refuses the held-out split)."""
    if isinstance(source, (str, Path)):
        text = Path(source).read_text()
        raw, spath = yaml.safe_load(text), str(Path(source).resolve())
    else:
        raw, spath = copy.deepcopy(source), None
        text = yaml.safe_dump(raw, sort_keys=False)
    if not isinstance(raw, dict):
        raise SpecError("a spec is a YAML mapping")
    unknown = set(raw) - KNOWN
    if unknown:
        raise SpecError(f"unknown spec keys {sorted(unknown)}; known: {sorted(KNOWN)}")
    for k in ("sweep", "interventions", "spend_cap_usd"):
        if k not in raw:
            raise SpecError(f"the spec needs {k!r}")
    spec = {**DEFAULTS, **raw}
    chosen = R.select_items(spec)  # the held-out guard runs before anything else is read
    if not isinstance(spec["sweep"], str) or not paths.SWEEP_NAME.match(spec["sweep"]):
        raise SpecError(f"sweep {spec['sweep']!r}: lower-case letters, digits, '.', '_' or '-'")
    if spec["model"] not in MODELS:
        raise SpecError(f"model {spec['model']!r}: one of {MODELS}")
    cap = float(spec["spend_cap_usd"])
    if not 0 < cap <= M.DAILY_CAP_USD:  # lab.toml's default; the Meter checks the day's own cap when it runs
        raise SpecError(f"spend_cap_usd: above 0 and at most {M.DAILY_CAP_USD} (the daily cap)")
    budget = spec["token_budget"]
    checks = [(isinstance(spec["samples"], int) and 1 <= spec["samples"] <= 50, "samples: an integer from 1 to 50"),
              (0.0 <= float(spec["temperature"]) <= 2.0, "temperature: 0 to 2"),
              (0.0 < float(spec["top_p"]) <= 1.0, "top_p: above 0, at most 1"),
              (spec["top_k"] is None or (isinstance(spec["top_k"], int) and spec["top_k"] >= 0), "top_k: an int >= 0"),
              (isinstance(spec["max_tokens"], int) and 1 <= spec["max_tokens"] <= 4096, "max_tokens: 1 to 4096"),
              (isinstance(spec["batch_size"], int) and 1 <= spec["batch_size"] <= 256, "batch_size: 1 to 256"),
              (isinstance(budget, int) and not isinstance(budget, bool) and 1_000 <= budget <= 2_000_000,
               "token_budget: an int from 1000 to 2000000"),
              (isinstance(spec["seed"], int), "seed: an int"),
              (0 < float(spec["tol"]) <= 0.10, "tol: above 0, at most 0.10")]
    for ok, msg in checks:
        if not ok:
            raise SpecError(msg)
    ivs = spec["interventions"]
    if not isinstance(ivs, list) or not ivs:
        raise SpecError("interventions: a non-empty list")
    return {**spec, "spend_cap_usd": cap, "chosen": chosen, "source_text": text, "source_path": spath}


def _layer_tag(spec) -> str:
    if spec is None or spec == "all":
        return ""
    if isinstance(spec, list):
        return "[" + ",".join(str(x) for x in spec) + "]"
    if isinstance(spec, dict):
        return f"[{spec.get('from')}-{spec.get('to')}]"
    return f"[{spec}]"


def _seeds(e) -> list[int]:
    seeds = e.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) and not isinstance(s, bool)
                                                            for s in seeds):
        raise SpecError(f"intervention {e}: seeds is a non-empty list of ints")
    return [int(s) for s in seeds]


def expand(spec: dict, registry: dict | None = None) -> list[dict]:
    """The spec's interventions as conditions, each {"name", "mode", "dir", "dname", "layers_spec", "coefficient",
    "control", "split"}. Named and file directions are loaded and their hashes checked here, before any spend. control
    is None for a plain condition, else {"kind", "target": Dir, ...}: random_matched {"seed"} and split_null {"seed",
    "bank", "match"} have dir None until prepare_controls builds them; other_trait {"match"} has the other trait's Dir.
    split is {"means": sequential|independent} for a split arm (mean_ablate, shift_only, shift_reverse), else None."""
    model = spec["model"]
    registry = registry or D.load_registry()
    out, names, targets = [], set(), {}

    def target_of(ref):
        key = json.dumps(ref, sort_keys=True)
        if key not in targets:
            targets[key] = D.resolve(ref, model, registry)
        return targets[key]

    for e in spec["interventions"]:
        if e in ("none", "baseline") or (isinstance(e, dict) and e.get("mode") == "none"):
            conds = [{"name": BASELINE, "mode": "none", "dir": None, "dname": None, "layers_spec": None,
                      "coefficient": None, "control": None}]
        elif isinstance(e, dict) and e.get("mode") in ("add", "ablate", *SPLIT_MODES):
            mode, ref, layers = e["mode"], e.get("direction"), e.get("layers")
            if ref is None:
                raise SpecError(f"intervention {e}: needs a direction")
            kind = ref if isinstance(ref, str) and ref in CONTROL_KINDS else None
            extra = set(e) - BASE_KEYS - CONTROL_KEYS.get(kind, set()) - (SPLIT_KEYS if mode in SPLIT_MODES else set())
            if extra:
                owners = sorted(k for k, keys in CONTROL_KEYS.items() if extra & keys)
                owners += [" / ".join(SPLIT_MODES)] if extra & SPLIT_KEYS else []
                hint = f" ({', '.join(sorted(extra))} belong to {' / '.join(owners)})" if owners else ""
                raise SpecError(f"intervention {e}: unknown keys {sorted(extra)}{hint}")
            if mode == "add":
                strengths = e.get("strengths", [e["coefficient"]] if "coefficient" in e else None)
                if not strengths or not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in strengths):
                    raise SpecError(f"intervention {e}: add needs strengths (a list of numbers) or a coefficient")
            else:
                if "strengths" in e or "coefficient" in e:
                    raise SpecError(f"intervention {e}: {mode} takes no strength")
                strengths = [None]
            match = e.get("match", "full")
            if match not in K.MATCHES:
                raise SpecError(f"intervention {e}: match is one of {', '.join(K.MATCHES)}")
            if match != "full" and mode != "ablate":
                raise SpecError(f"intervention {e}: match {match} is a partial cut, so it needs mode ablate")
            split = None
            if mode in SPLIT_MODES:
                if kind:
                    raise SpecError(f"intervention {e}: {mode} splits a plain direction's cut; it takes no control "
                                    f"({kind})")
                split = {"means": e.get("means", "sequential")}
                if split["means"] not in SPLIT_MEANS:
                    raise SpecError(f"intervention {e}: means is one of {', '.join(SPLIT_MEANS)}")
            if kind:
                if "target" not in e:
                    raise SpecError(f"intervention {e}: {kind} needs a target direction")
                tgt = target_of(e["target"])
                if kind == D.RANDOM:
                    pairs = [(f"{D.RANDOM}({tgt.name})#s{s}", None, {"kind": kind, "target": tgt, "seed": s})
                             for s in _seeds(e)]
                elif kind == OTHER:
                    traits = e.get("traits")
                    if not isinstance(traits, list) or not traits:
                        raise SpecError(f"intervention {e}: other_trait needs traits (a list of directions)")
                    pairs = []
                    for tr in traits:
                        d = target_of(tr)
                        if d.key == tgt.key:
                            raise SpecError(f"intervention {e}: {d.name} is the target itself")
                        pairs.append((f"{OTHER}({tgt.name})={d.name}", d, {"kind": kind, "target": tgt, "match": match}))
                else:
                    bank = e.get("bank")
                    if bank is not None and not (isinstance(bank, dict) and "path" in bank
                                                 and (bank.get("tensor_sha256") or bank.get("sha256"))):
                        raise SpecError(f"intervention {e}: bank is {{path, tensor_sha256}} (or a file sha256)")
                    pairs = [(f"{SPLIT}({tgt.name})#s{s}", None,
                              {"kind": kind, "target": tgt, "seed": s, "bank": bank, "match": match}) for s in _seeds(e)]
            else:
                d = target_of(ref)
                pairs = [(d.name, d, None)]
            conds = []
            for dname, d, ctl in pairs:
                for c in strengths:
                    name = (f"{mode}:{dname}" + _layer_tag(layers) + (f"~{match}" if match != "full" else "")
                            + ("~independent" if split and split["means"] == "independent" else "")
                            + (f"@{c:+g}" if c is not None else ""))
                    conds.append({"name": name, "mode": mode, "dir": d, "dname": dname, "layers_spec": layers,
                                  "coefficient": None if c is None else float(c), "control": ctl, "split": split})
            if e.get("name"):
                if len(conds) != 1:
                    raise SpecError(f"intervention {e}: a custom name needs a single condition (one seed, one strength)")
                conds[0]["name"] = str(e["name"])
        else:
            raise SpecError(f"intervention {e!r}: baseline, or {{mode: add|ablate|{'|'.join(SPLIT_MODES)}, direction, "
                            f"...}}")
        for c in conds:
            if c["name"] in names:
                raise SpecError(f"condition {c['name']} is listed twice")
            names.add(c["name"])
        out.extend(conds)
    if BASELINE not in names:
        raise SpecError("every sweep needs the baseline (no intervention): the signature is measured from it")
    return out


def missing_controls(conds: list[dict]) -> list[str]:
    return [] if any(c.get("control") for c in conds) else [
        "a matched control (random_matched, other_trait or split_null) for the steered direction"]


def control_kinds(conds: list[dict]) -> list[str]:
    return sorted({c["control"]["kind"] for c in conds if c.get("control")})


# ----------------------------------------------------------------------------------------------- the bench


class RemoteBench:
    """The deployed Modal class."""

    def __init__(self, model_key: str):
        import modal

        self.obj = modal.Cls.from_name(APP_NAME, CLASS_NAME)(model_key=model_key)

    def info(self) -> dict:
        return self.obj.info.remote()

    def generate(self, payload: dict) -> dict:
        return self.obj.generate.remote(payload)

    def random_matched(self, payload: dict) -> dict:
        return self.obj.random_matched.remote(payload)

    def measure(self, payload: dict) -> dict:
        return self.obj.measure.remote(payload)

    def response_means(self, payload: dict) -> dict:
        return self.obj.response_means.remote(payload)

    def split_means(self, payload: dict) -> dict:
        return self.obj.split_means.remote(payload)


class LocalBench:
    """A steering.Service in this process (the CPU tests). Payloads and results go through pickle, as with Modal."""

    def __init__(self, svc):
        self.svc = svc

    def _call(self, fn, *args):
        return pickle.loads(pickle.dumps(fn(*[pickle.loads(pickle.dumps(a)) for a in args])))

    def info(self) -> dict:
        return self._call(self.svc.info)

    def generate(self, payload: dict) -> dict:
        return self._call(self.svc.generate, payload)

    def random_matched(self, payload: dict) -> dict:
        return self._call(self.svc.random_matched, payload)

    def measure(self, payload: dict) -> dict:
        return self._call(self.svc.measure, payload)

    def response_means(self, payload: dict) -> dict:
        return self._call(self.svc.response_means, payload)

    def split_means(self, payload: dict) -> dict:
        return self._call(self.svc.split_means, payload)


def derive_seed(base: int, item_id: str, sample: int) -> int:
    return int(hashlib.sha256(f"{base}|{item_id}|{sample}".encode()).hexdigest()[:8], 16)


def neutral_set() -> tuple[list[str], str]:
    prompts = json.loads(NEUTRAL.read_text())["prompts"]
    return prompts, schema.sha256_text(schema.canonical(prompts))


# ----------------------------------------------------------------------------------------------- calls and passes


def est_tokens(messages: list[dict]) -> int:
    """A cautious count of a chat's prompt tokens without a tokenizer: a token per 2.5 characters (the 1 Nov game
    prompts ran 2.6-3.9 characters a token), 8 per message and 16 for the template, and 600 more when there is no
    system message (Ministral 3's template then adds its own, about 520 tokens)."""
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    no_system = not any(m.get("role") == "system" for m in messages)
    return chars * 2 // 5 + 8 * len(messages) + 16 + (600 if no_system else 0)


def plan_passes(lengths, max_new, token_budget: int = TOKEN_BUDGET, max_rows: int = GEN_BATCH) -> list[list[int]]:
    """steering.plan_passes (steering imports on the laptop without transformers)."""
    import steering

    return steering.plan_passes(lengths, max_new, token_budget, max_rows)


def plan_calls(lengths: list[int], groups: list, batch_size: int, max_tokens: int, token_budget: int,
               max_steps: int = MAX_CALL_STEPS) -> list[tuple[int, int, int]]:
    """Calls as (start, end, planned passes) over the jobs, in order. A call takes the next job while it holds fewer
    than batch_size requests and its planned passes (plan_passes per intervention group, on estimated prompt lengths)
    need at most max_steps decoding steps (passes x max_tokens). A call always takes at least one job."""
    out, start, members, counts, k = [], 0, {}, {}, 0
    while k < len(lengths):
        g = groups[k]
        trial = members.get(g, []) + [lengths[k]]
        n_pass = len(plan_passes(trial, max_tokens, token_budget))
        total = sum(counts.values()) - counts.get(g, 0) + n_pass
        if k > start and (k - start + 1 > batch_size or total * max_tokens > max_steps):
            out.append((start, k, sum(counts.values())))
            start, members, counts = k, {}, {}
            continue
        members[g], counts[g] = trial, n_pass
        k += 1
    if start < len(lengths):
        out.append((start, len(lengths), sum(counts.values())))
    return out


def slots(passes: int, max_tokens: int) -> int:
    """Token slots a call generates, counting each pass as a full one of GEN_BATCH rows."""
    return passes * GEN_BATCH * max_tokens


def call_estimate(passes: int, max_tokens: int, per_token_s: float) -> float:
    return 5.0 + per_token_s * slots(passes, max_tokens)


# ----------------------------------------------------------------------------------------------- the controls


def build_bank(bench, layers: list[int], out_dir, *, info: dict, meter, log=print, opts: dict | None = None) -> dict:
    """A split_null bank (controls.py): the container samples the ego traits' contrastive replies and reads their
    per-response means at `layers`; the bank is saved in out_dir with how it was made and a recipe check (cosines
    of its own pos - neg difference with the registry's ego-family directions)."""
    o = {**K.BANK, **(opts or {})}
    rows, binfo = K.bank_rows(cells_per_trait=o["cells_per_trait"], seed=o["seed"])
    meter.before_call(60.0 + PER_TOKEN_S * len(rows) * o["max_tokens"])
    res = bench.response_means({"rows": [{k: r[k] for k in ("id", "system", "user", "seed")} for r in rows],
                                "layers": layers, "max_tokens": o["max_tokens"], "temperature": o["temperature"],
                                "top_p": o["top_p"], "top_k": None, "include_system": o["include_system"]})
    meter.after_call(what="response_means")
    meta = {**binfo, "model_key": info.get("model_key"), "model": info.get("model"), "revision": info.get("revision"),
            "sampling": {"max_tokens": o["max_tokens"], "temperature": o["temperature"], "top_p": o["top_p"],
                         "top_k": "the model's generation_config", "sampler": "steering.RowSampler, a seed per row"},
            "reading": "anatta.probe.response_activations: the mean residual over the reply's tokens (with the "
                       "template's closing tokens, as probe.extract reads), "
                       + ("with" if o["include_system"] else "without") + " the system prompt",
            "include_system": o["include_system"], "container_s": res["meta"]["seconds"]}
    bank = K.save_bank(out_dir, rows, res["replies"], res["means"], meta)
    _, dropped = K.usable_cells(bank)
    rc = {}
    hidden = bank["meta"]["hidden"]
    try:
        reg = D.load_registry()["directions"].get(info.get("model_key"), {})
        dirs = {n: D.resolve(n, info["model_key"]) for n in (*K.EGO_TRAITS, "ego") if n in reg}
        dirs = {n: d for n, d in dirs.items() if d.dim() == hidden}
        if dirs:
            check = K.recipe_check(bank, dirs, layers)
            rc = {kind: {n: _median_range(v) for n, v in by.items()} for kind, by in check.items()}
            K.save_json(out_dir, Path(bank["path"]).stem + "-recipe", {"bank_tensor_sha256": bank["tensor_sha256"],
                                                                       "recipe_check": check, "summary": rc})
    except D.DirectionError as e:  # a registry file missing on this machine: the check is skipped, not the bank
        rc = {"skipped": str(e)}
    bank["recipe_check"] = rc
    log(f"[bench] split_null bank: {len(rows)} replies ({binfo['n_cells']} cells of {len(binfo['traits'])} ego "
        f"traits), layers {layers[0]}-{layers[-1]}, {len(dropped)} cells dropped for an empty reply; tensor sha256 "
        f"{bank['tensor_sha256'][:16]}, {Path(bank['path']).name}"
        + (f"; recipe check, cos(pos - neg, ego) median {rc['pooled']['ego'][0]:+.3f}" if "ego" in rc.get("pooled", {})
           else ""))
    return bank


def _median_range(by_layer: dict) -> list[float]:
    v = sorted(by_layer.values())
    return [round(v[len(v) // 2], 4), round(v[0], 4), round(v[-1], 4)] if v else []


def prepare_controls(bench, conds: list[dict], *, info: dict, out_dir, prompts: list[str], neutral_sha: str,
                     tol: float, meter, log=print) -> dict:
    """Build every control the conditions need, resolve every condition's layers and set its container intervention
    ("iv"). In order: random_matched controls (one container call per target); split_null banks (built once per run
    at every layer their nulls need, or loaded from a pinned file) and nulls (on the laptop); layers (a control cuts at
    its target's layers unless it names its own); then one measure call for the other_trait and split_null controls,
    whose tables, cosines, partial-cut fractions and files are recorded; then the split arms' constants
    (prepare_split). Returns {"banks": [...], "split": [...]}."""
    n_layers, hidden = info["n_layers"], info["hidden"]
    out_dir = Path(out_dir)

    def kind_of(c):
        return (c.get("control") or {}).get("kind")

    for c in conds:
        if c.get("control") and c["control"]["target"].dim() != hidden:
            t = c["control"]["target"]
            raise SpecError(f"direction {t.name} is {t.dim()}-dimensional; the model's is {hidden}")

    # random_matched: one per target and seed, built on every layer the target has
    todo: dict = {}
    for c in conds:
        if kind_of(c) == D.RANDOM:
            todo.setdefault(c["control"]["target"].key, (c["control"]["target"], set()))[1].add(c["control"]["seed"])
    made: dict = {}
    for tkey, (tgt, seeds) in todo.items():
        layers = tgt.layers_for(None, n_layers)
        meter.before_call(30.0 + 2.0 * len(seeds))
        res = bench.random_matched({"target": {"sha256": tgt.sha256, "tensor_sha256": tgt.tensor_sha256,
                                               **tgt.payload(n_layers)},
                                    "seeds": sorted(seeds), "layers": layers, "neutral_prompts": prompts,
                                    "neutral_sha256": neutral_sha, "tol": tol})
        meter.after_call(what="random_matched")
        for d in res["directions"]:
            made[(tkey, d["seed"])] = D.save_control(out_dir, tgt, d["seed"], d["vectors"], d["made"])
        log(f"[bench] random_matched({tgt.name}), seeds {sorted(seeds)}, {len(layers)} layers, variance ratios "
            + "; ".join(f"s{x}: {made[(tkey, x)].made['ratio_range'][0]:.4f}-{made[(tkey, x)].made['ratio_range'][1]:.4f}"
                        for x in sorted(seeds))
            + "; mean-square ratios " + "; ".join(
                "s{}: {:.3f}-{:.3f}".format(x, *K.span(K.moment_ratios(made[(tkey, x)].made).values()))
                for x in sorted(seeds)))
    for c in conds:
        if kind_of(c) == D.RANDOM:
            c["dir"] = made[(c["control"]["target"].key, c["control"]["seed"])]

    # split_null: banks, then the nulls per (bank, target), at the union of the layers their conditions cut
    by_bank: dict = {}
    for c in conds:
        if kind_of(c) == SPLIT:
            ctl = c["control"]
            need = ctl["target"].layers_for(c["layers_spec"], n_layers)
            g = by_bank.setdefault(json.dumps(ctl["bank"], sort_keys=True) if ctl["bank"] else None,
                                   {"ref": ctl["bank"], "layers": set(), "conds": []})
            g["layers"] |= set(need)
            g["conds"].append((c, need))
    banks = []
    for g in by_bank.values():
        layers = sorted(g["layers"])
        if g["ref"] is None:
            bank = build_bank(bench, layers, out_dir, info=info, meter=meter, log=log)
        else:
            bank = K.load_bank(g["ref"])
            if bank["meta"].get("model") not in (None, info.get("model")):
                raise SpecError(f"bank {bank['path']} was built on {bank['meta'].get('model')}, not {info.get('model')}")
            missing = sorted(set(layers) - set(bank["means"]))
            if missing:
                raise SpecError(f"bank {bank['path']} has no means at layers {missing}")
            log(f"[bench] split_null bank {Path(bank['path']).name} (tensor sha256 {bank['tensor_sha256'][:16]})")
        banks.append({"file": bank["path"], "tensor_sha256": bank["tensor_sha256"], "sha256": bank["sha256"],
                      "built_here": g["ref"] is None, "rows": len(bank["rows"]), "layers": sorted(bank["means"]),
                      "recipe_check": bank.get("recipe_check")})
        per_target: dict = {}
        for c, need in g["conds"]:
            pt = per_target.setdefault(c["control"]["target"].key,
                                       {"target": c["control"]["target"], "layers": set(), "seeds": set()})
            pt["layers"] |= set(need)
            pt["seeds"].add(c["control"]["seed"])
        nulls = {}
        for tkey, pt in per_target.items():
            lay = sorted(pt["layers"])
            ch = K.chance(bank, pt["target"], lay)
            for s in sorted(pt["seeds"]):
                vecs, mk = K.split_null(bank, s, lay, pt["target"], chance_table=ch)
                nulls[(tkey, s)] = D.Dir(name=f"{SPLIT}({pt['target'].name})", sha256=None, path=None, vectors=vecs,
                                         best_layer=pt["target"].best_layer, made=mk,
                                         tensor_sha256=D.tensor_sha256(vecs))
        for c, _ in g["conds"]:
            c["dir"] = nulls[(c["control"]["target"].key, c["control"]["seed"])]

    # layers: a control cuts at its target's layers unless it names its own; the target must have every cut layer
    for c in conds:
        if c["mode"] == "none":
            c["layers"], c["iv"] = [], {"mode": "none", "direction": None, "sha256": None, "tensor_sha256": None,
                                        "layers": [], "coefficient": None}
            continue
        if c["dir"].dim() != hidden:
            raise SpecError(f"direction {c['dir'].name} is {c['dir'].dim()}-dimensional; the model's is {hidden}")
        ctl = c.get("control")
        spec_layers = c["layers_spec"]
        if ctl and spec_layers is None:
            spec_layers = ctl["target"].layers_for(None, n_layers)
        c["layers"] = c["dir"].layers_for(spec_layers, n_layers)
        if ctl:
            ctl["target"].layers_for(c["layers"], n_layers)
        if kind_of(c) == D.RANDOM:
            c["made"] = c["dir"].made

    # other_trait and split_null: the neutral-set table against the target, the cosines and any partial cut
    need = [c for c in conds if kind_of(c) in (OTHER, SPLIT)]
    if need:
        pairs: dict = {}
        for c in need:
            tgt = c["control"]["target"]
            p = pairs.setdefault((c["dir"].key, tgt.key), {"dir": c["dir"], "target": tgt, "kind": kind_of(c),
                                                          "layers": set(), "conds": []})
            p["layers"] |= set(c["layers"])
            p["conds"].append(c)
        dirs = {}
        for p in pairs.values():
            dirs[p["target"].key] = p["target"].payload(n_layers)
            dirs[p["dir"].key] = p["dir"].payload(n_layers)
        layers = sorted(set().union(*(p["layers"] for p in pairs.values())))
        meter.before_call(30.0 + 0.5 * len(dirs))
        res = bench.measure({"directions": dirs, "layers": layers, "neutral_prompts": prompts,
                             "neutral_sha256": neutral_sha})
        meter.after_call(what="measure")
        neutral = res["neutral"]
        for p in pairs.values():
            d, tgt, lay = p["dir"], p["target"], sorted(p["layers"])
            table = K.variance_table(res["stats"], tgt.key, d.key, lay)
            cos = K.cos_by_layer(d, tgt, lay)
            if p["kind"] == SPLIT:
                mk = copy.deepcopy(d.made)
                for x in lay:
                    mk["layers"][str(x)].update(table[x])
                mk.update({"neutral": neutral, **K.summarize(table, cos)})
                saved = D.save_control(out_dir, tgt, mk["seed"], d.vectors, K._jsonable(mk), family=SPLIT)
                fname = Path(saved.path).name
            else:
                full = K.made_other(tgt, d, table, cos, "full", None, neutral)
                stem = f"{OTHER}-{tgt.name}-{tgt.key[:8]}-{d.name}-{d.key[:8]}-L{lay[0]}-{lay[-1]}"
                fname = Path(K.save_json(out_dir, stem, full)).name
                saved = d
            for c in p["conds"]:
                ctl = c["control"]
                tab = {x: table[x] for x in c["layers"]}
                cs = {x: cos[x] for x in c["layers"]}
                fr = K.partial_fractions(tab, ctl["match"])
                c["dir"], c["fractions"] = saved, fr
                mk = {"method": K.SPLIT_METHOD if p["kind"] == SPLIT else K.OTHER_METHOD, "target": tgt.name,
                      "target_tensor_sha256": tgt.tensor_sha256, "neutral_sha256": neutral.get("sha256"),
                      "match": ctl["match"], "file": fname, **K.summarize(tab, cs, fr)}
                if p["kind"] == SPLIT:
                    p95 = [saved.made["layers"][str(x)]["abs_cos_chance_p95"] for x in c["layers"]]
                    mk.update({"seed": ctl["seed"], "bank_tensor_sha256": saved.made["bank"]["tensor_sha256"],
                               "chance_p95_range": [round(min(p95), 4), round(max(p95), 4)]})
                else:
                    mk.update({"trait": d.name, "trait_tensor_sha256": d.tensor_sha256})
                if fr:
                    mk["fractions"] = {str(x): round(f, 6) for x, f in sorted(fr.items())}
                c["made"] = mk
                log(f"[bench] {c['name']}: layers {c['layers'][0]}-{c['layers'][-1]}, variance ratio "
                    f"{mk['ratio_range'][0]:.3f}-{mk['ratio_range'][1]:.3f} of {tgt.name}'s, |cos| "
                    f"{mk['abs_cos_range'][0]:.3f}-{mk['abs_cos_range'][1]:.3f}"
                    + (f", chance p95 {mk['chance_p95_range'][0]:.3f}-{mk['chance_p95_range'][1]:.3f}"
                       if p["kind"] == SPLIT else "")
                    + (f", cut fractions {mk['fraction_range'][0]:.3f}-{mk['fraction_range'][1]:.3f}" if fr else ""))

    split = prepare_split(bench, conds, n_layers=n_layers, out_dir=out_dir, prompts=prompts, neutral_sha=neutral_sha,
                          meter=meter, log=log)

    for c in conds:
        if c["mode"] != "none":
            c["iv"] = {"mode": c["mode"], "direction": c["dname"], "tensor_sha256": c["dir"].tensor_sha256,
                       "sha256": c["dir"].sha256, "layers": c["layers"], "coefficient": c["coefficient"],
                       "fractions": c.get("fractions"), "means": c.get("means")}
    return {"banks": banks, "split": split}


def prepare_split(bench, conds: list[dict], *, n_layers: int, out_dir, prompts: list[str], neutral_sha: str, meter,
                  log=print) -> list[dict]:
    """The split arms' constants: one split_means call per (direction, layers), with every mode the run asks for at
    them. Each split condition gets "means" ({layer: constant}: the sequential ones, or for means: independent every
    layer's uncut mean mu_l) and "made". A table per call goes to directions/split-<name>-<tensor hash>-L<a>-<b>.json:
    mu_l and sd_l of h.d on the neutral set at every layer, each mode's constants, and what each arm (and the plain
    cut, and shift_only with the uncut means) does to h.d's mean and sd there. Returns the tables for run.json."""
    groups: dict = {}
    for c in conds:
        if c["mode"] in SPLIT_MODES:
            g = groups.setdefault((c["dir"].key, tuple(c["layers"])), {"dir": c["dir"], "layers": c["layers"],
                                                                      "modes": set(), "conds": []})
            g["modes"].add(c["mode"])
            g["conds"].append(c)
    tables = []
    for g in groups.values():
        d, lay = g["dir"], g["layers"]
        modes = [m for m in SPLIT_MODES if m in g["modes"]]
        meter.before_call(60.0 + SPLIT_PASS_S * len(prompts) * (1 + len(modes) * len(lay) + len(modes) + 2))
        res = bench.split_means({"direction": d.payload(n_layers), "key": d.key, "layers": lay,
                                 "neutral_prompts": prompts, "neutral_sha256": neutral_sha, "modes": modes})
        meter.after_call(what="split_means")
        base = {int(x): v for x, v in res["base"].items()}
        table = K._jsonable({
            "method": res["meta"]["method"], "direction": d.name, "tensor_sha256": d.tensor_sha256, "sha256": d.sha256,
            "layers": lay, "neutral": res["neutral"],
            "mu": {str(x): base[x]["mean"] for x in lay}, "sd": {str(x): base[x]["sd"] for x in lay},
            "base": {str(x): base[x] for x in lay},
            "consts": {m: {str(int(x)): v for x, v in cs.items()} for m, cs in res["consts"].items()},
            "realized": {a: {str(int(x)): v for x, v in per.items()} for a, per in res["realized"].items()},
            "passes": res["meta"]["passes"], "container_s": res["meta"]["seconds"],
            "rule": "mu_l, sd_l: h.d over the neutral set's user-content tokens, uncut; constants found layer by layer "
                    "with the arm's own hooks below (steering.split_means); realized: h.d's mean and sd at each layer "
                    "with every hook of the arm on"})
        stem = f"split-{d.name}-{d.key[:8]}-L{lay[0]}-{lay[-1]}"
        fname = Path(K.save_json(out_dir, stem, table)).name
        table["file"] = fname
        for c in g["conds"]:
            how = c["split"]["means"]
            consts = res["consts"][c["mode"]] if how == "sequential" else {x: base[x]["mean"] for x in lay}
            c["means"] = {int(x): float(v) for x, v in consts.items()}
            c["made"] = {"method": table["method"], "means": how, "neutral_sha256": neutral_sha, "file": fname,
                         "mu_range": K.span(table["mu"].values()), "sd_range": K.span(table["sd"].values()),
                         "const_range": K.span(c["means"].values())}
        tables.append(table)
        last = str(lay[-1])
        log(f"[bench] split_means({d.name}), layers {lay[0]}-{lay[-1]}, modes {', '.join(modes)}: "
            f"{res['meta']['passes']} passes in {res['meta']['seconds']:.0f}s; at layer {last} h.d's neutral mean "
            + ", ".join(f"{a} {table['realized'][a][last]['mean']:+.2f}" for a in table["realized"])
            + f" (uncut {table['mu'][last]:+.2f}, sd {table['sd'][last]:.2f})")
    return tables


# ----------------------------------------------------------------------------------------------- a sweep


def run_sweep(source, *, bench=None, cap: float | None = None, stop_after: bool = False, root=None, log=print,
              stop_fn=None) -> dict:
    """Run one spec end to end. Returns the manifest (also written to run.json). `bench` defaults to the deployed
    Modal class; `stop_fn` (default: stop the bench's containers) runs before the meter closes when stop_after."""
    spec = load_spec(source)
    conds = expand(spec)
    missing = missing_controls(conds)
    if missing and not spec.get("allow_missing_controls"):
        raise SpecError(f"PROGRAM.md: every sweep has a baseline and matched controls; missing {missing}. Add one, or "
                        f"set allow_missing_controls: '<reason>' for a harness check.")
    P = paths.Paths(Path(root)) if root else paths.Paths.default()
    date, run_id, started = schema.utc_date(), R.new_run_id(), schema.utc_now()
    sweep_dir = P.sweep_dir(date, spec["sweep"])
    sweep_dir.mkdir(parents=True, exist_ok=True)
    chosen = spec["chosen"]
    jobs = [(s, ci, fam, iid) for s in range(spec["samples"]) for ci in range(len(conds))
            for fam in chosen["families"] for iid in chosen["items"][fam]]
    model_key, mt, budget = spec["model"], spec["max_tokens"], spec["token_budget"]
    est = {(f, i): est_tokens(chosen["library"][f]["by_id"][i]["messages"]) for f in chosen["families"]
           for i in chosen["items"][f]}
    plan = plan_calls([est[(f, i)] for _, _, f, i in jobs], [conds[ci]["name"] for _, ci, _, _ in jobs],
                      spec["batch_size"], mt, budget)
    meter = M.Meter(spend_dir=P.spend, label=spec["sweep"], session=run_id, model_key=model_key,
                    cap_usd=cap if cap is not None else spec["spend_cap_usd"], rate=RATE["total"], tail_s=SCALEDOWN_S,
                    sweep_log=sweep_dir / "spend.jsonl")
    partial = sweep_dir / "records.partial.jsonl"
    records, errors, calls = [], [], []
    status, stopped, info, cold_s, ego_identical, prepared = "complete", None, None, None, None, {}
    per_token_s = PER_TOKEN_S
    log(f"[bench] {spec['sweep']} {run_id}: {len(jobs)} generations ({len(conds)} conditions) in {len(plan)} calls, "
        f"model {model_key}, items from {chosen['source']}; session cap ${meter.cap:.2f}, rate ${RATE['total']:.2f}/h")
    meter.start(COLD_START_S + call_estimate(plan[0][2] if plan else 1, mt, per_token_s))
    partial.write_text("")
    try:
        if bench is None:
            bench = RemoteBench(model_key)
        t = time.time()
        try:
            info = bench.info()
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "NotFoundError":
                raise SystemExit(f"the bench app {APP_NAME} is not deployed; run: client.py up") from None
            raise
        cold_s = round(time.time() - t, 2)
        meter.after_call(what="info")
        log(f"[bench] container {info.get('container_id')} ready in {cold_s}s (model load {info.get('load_seconds')}s)")
        n_layers = info["n_layers"]
        prompts, neutral_sha = neutral_set()
        prepared = prepare_controls(bench, conds, info=info, out_dir=sweep_dir / "directions", prompts=prompts,
                                    neutral_sha=neutral_sha, tol=spec["tol"], meter=meter, log=log)
        intervention = {c["name"]: record_intervention(c) for c in conds}

        # the generations, in the planned calls
        for start, end, planned in plan:
            chunk = jobs[start:end]
            reqs, dirs = [], {}
            for rid, (s, ci, fam, iid) in enumerate(chunk):
                c = conds[ci]
                item = chosen["library"][fam]["by_id"][iid]
                reqs.append({"rid": rid, "messages": item["messages"], "temperature": float(spec["temperature"]),
                             "top_p": float(spec["top_p"]), "top_k": spec["top_k"], "max_tokens": mt,
                             "seed": derive_seed(spec["seed"], iid, s), "intervention": c["iv"]})
                if c["dir"] is not None:
                    dirs[c["dir"].key] = c["dir"].payload(n_layers)
            meter.before_call(call_estimate(planned, mt, per_token_s))
            t = time.time()
            try:
                out = bench.generate({"requests": reqs, "directions": dirs, "token_budget": budget})
            except Exception as e:  # noqa: BLE001
                n = len(calls) + 1  # this call's own number (calls holds the ones that finished)
                errors.append({"ts": schema.utc_now(), "call": n, "requests": len(reqs),
                               "error": f"{type(e).__name__}: {e}"[:500]})
                meter.after_call(what="generate", error=True)
                status, stopped = "error", f"{type(e).__name__}: {str(e)[:300]}"
                log(f"[bench] call {n} failed: {stopped}")
                break
            wall = time.time() - t
            meter.after_call(what="generate", requests=len(reqs))
            meta = out["meta"]
            ego_identical = meta.get("ego_cut_identical", ego_identical)
            ran = sum(len(g.get("passes") or [None]) for g in meta["groups"])
            per_token_s = max(PER_TOKEN_S / 4, 1.5 * wall / slots(max(ran, 1), mt))
            gcost = [g["seconds"] * RATE["total"] / 3600 / g["n"] for g in meta["groups"]]
            for (s, ci, fam, iid), r in zip(chunk, out["results"]):
                c = conds[ci]
                item = chosen["library"][fam]["by_id"][iid]
                rec = R.make_record(sweep=spec["sweep"], run_id=run_id, model=info["model"],
                                    intervention=intervention[c["name"]], family=fam, item_id=iid, sample=s,
                                    messages=item["messages"], result=r, cost_usd=gcost[r["group"]],
                                    params={"temperature": float(spec["temperature"]), "top_p": float(spec["top_p"]),
                                            "top_k": r.get("top_k"), "max_tokens": mt,
                                            "seed": derive_seed(spec["seed"], iid, s)},
                                    latency_s=wall, split="dev")  # load_family checked the file is the dev split
                records.append(rec)
                schema.append_jsonl(partial, rec)
            calls.append({"requests": len(reqs), "wall_s": round(wall, 3), "container_s": meta["seconds"],
                          "per_request_s": round(wall / len(reqs), 3), "container_id": meta.get("container_id"),
                          "passes": ran, "planned_passes": planned, "groups": meta["groups"]})
            log(f"[bench] call {len(calls)}: {len(reqs)} requests in {wall:.1f}s ({wall / len(reqs):.2f}s each, "
                f"{ran} passes); {len(records)}/{len(jobs)} done; session ${meter.charged_usd:.4f}")
    except M.CapReached as e:
        status, stopped = "stopped_at_cap", str(e)
        log(f"[bench] {stopped}")
    except BaseException as e:
        status, stopped = "error", f"{type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        if stop_after:
            (stop_fn or stop_containers)()
        meter.close(stopped=stop_after)
        order = {c["name"]: i for i, c in enumerate(conds)}
        fams = {f: i for i, f in enumerate(chosen["families"])}
        records.sort(key=lambda r: (order[r["intervention"]["name"]], fams[r["family"]], r["item_id"], r["sample"]))
        schema.write_jsonl(sweep_dir / "records.jsonl", records)
        schema.write_jsonl(sweep_dir / "errors.jsonl", errors)
        variants = log_variants(P, spec, run_id, conds)
        man = manifest(spec, conds=conds, run_id=run_id, date=date, started=started, status=status, stopped=stopped,
                       records=records, errors=errors, jobs=len(jobs), meter=meter, P=P, info=info, cold_s=cold_s,
                       calls=calls, variants=variants, missing=missing, ego_identical=ego_identical,
                       stop_after=stop_after, plan=plan, prepared=prepared)
        schema.write_json(sweep_dir / "run.json", man)
        log(f"[bench] {status}: {len(records)} records in {sweep_dir}; session ${meter.charged_usd:.4f}, today "
            f"${man['spend']['today_usd']:.4f}")
    return man


def record_intervention(c: dict) -> dict:
    """The record's intervention: {"kind": "steer", "name", "params"}; params fix the steer: the direction's tensor
    hash (its pin) and file hash, the layers, the strength, any partial-cut fractions, a split arm's per-layer means
    and how they were found, and for a control how it was made."""
    if c["mode"] == "none":
        params = {"mode": "none", "direction": None, "sha256": None, "tensor_sha256": None, "layers": [],
                  "coefficient": None}
        return {"kind": "steer", "name": c["name"], "params": params}
    d, ctl = c["dir"], c.get("control")
    params = {"mode": c["mode"], "direction": c["dname"], "sha256": d.sha256, "tensor_sha256": d.tensor_sha256,
              "layers": c["layers"], "coefficient": c["coefficient"], "registered": d.registered}
    if c.get("fractions"):
        params["fractions"] = {str(x): round(float(f), 6) for x, f in sorted(c["fractions"].items())}
    if c.get("means"):
        params["means"] = {str(x): round(float(m), 6) for x, m in sorted(c["means"].items())}
        params["split"] = c["made"]
    if d.note:
        params["note"] = d.note
    if ctl:
        params["control"] = ctl["kind"]
        if ctl["kind"] == D.RANDOM:
            mk = d.made
            params["made"] = {"method": mk["method"], "seed": mk["seed"], "target": mk["target"]["name"],
                              "target_sha256": mk["target"]["sha256"],
                              "target_tensor_sha256": mk["target"].get("tensor_sha256"),
                              "neutral_sha256": mk["neutral"]["sha256"],
                              "ratio_range": [round(x, 6) for x in mk["ratio_range"]], "tol": mk["tol"],
                              "moment_ratio_range": K.span(K.moment_ratios(mk, c["layers"]).values(), 6),
                              "file": Path(d.path).name}
        else:
            params["made"] = c["made"]
    return {"kind": "steer", "name": c["name"], "params": params}


def log_variants(P, spec: dict, run_id: str, conds: list[dict]) -> dict:
    """Every condition to <results>/variants.jsonl (scout's format, tool "bench", plus the tensor hash), and the
    counts; a direction counts as distinct by its tensor hash where it has one."""
    for c in conds:
        d = c.get("dir")
        schema.append_jsonl(P.variants, {"ts": schema.utc_now(), "tool": "bench", "sweep": spec["sweep"],
                                         "run_id": run_id, "kind": "steer", "name": c["name"],
                                         "sha256": d.sha256 if d is not None else None,
                                         "tensor_sha256": d.tensor_sha256 if d is not None else None})
    rows = schema.read_jsonl(P.variants)
    distinct = {(r["kind"], r["name"], r.get("tensor_sha256") or r.get("sha256")) for r in rows}
    return {"this_sweep": len(conds), "track_distinct": len(distinct),
            "track_sweep_runs": len({r["run_id"] for r in rows}), "models": 1,
            "families": len(spec["chosen"]["families"]), "cells": len(conds) * len(spec["chosen"]["families"])}


def manifest(spec, *, conds, run_id, date, started, status, stopped, records, errors, jobs, meter, P, info, cold_s,
             calls, variants, missing, ego_identical, stop_after, plan=None, prepared=None) -> dict:
    """run.json, with scout's manifest keys where they apply (the report reads variants, models, baseline and
    missing_controls) and the bench's own under "bench"."""
    chosen = spec["chosen"]
    hf = info["model"] if info else D.load_registry()["models"][spec["model"]]["hf_id"]
    heldout = Path(chosen["heldout_file"])
    walls = [c["wall_s"] for c in calls]
    reqs = sum(c["requests"] for c in calls)
    return {
        "sweep": spec["sweep"], "run_id": run_id, "date": date, "started": started, "finished": schema.utc_now(),
        "tool": "bench", "status": status, "stopped": stopped, "spec_path": spec["source_path"],
        "spec_sha256": schema.sha256_text(spec["source_text"]), "spec_text": spec["source_text"],
        "library": {f: chosen["library"][f]["sha256"] for f in chosen["families"]}, "library_source": chosen["source"],
        "heldout_sha256": lib.file_sha256(heldout) if heldout.exists() else None,
        "models": [{"id": hf, "provider": R.PROVIDER, "model_key": spec["model"],
                    "revision": info.get("revision") if info else None, "gpu": RATE["gpu_type"]}],
        "baseline": BASELINE,
        "variants": [record_intervention(c) if c.get("iv") else {"kind": "steer", "name": c["name"], "params": {}}
                     for c in conds],
        "variant_counts": variants, "missing_controls": missing,
        "allow_missing_controls": spec.get("allow_missing_controls"), "items": chosen["items"],
        "samples": spec["samples"], "temperature": float(spec["temperature"]), "top_p": float(spec["top_p"]),
        "top_k": spec["top_k"], "max_tokens": spec["max_tokens"], "seed": spec["seed"], "jobs": jobs,
        "records": len(records), "errors": len(errors),
        "spend": {"this_run_usd": meter.charged_usd, "session_cap_usd": meter.cap,
                  "today_usd": round(M.scout_spend.spent_today(P.spend), 8), "daily_cap_usd": meter.daily_cap,
                  "rate_usd_per_hour": RATE, "charged_seconds": round(meter.charged_s, 1),
                  "warm_tail_charged": not stop_after, "tail_s": SCALEDOWN_S},
        "bench": {"info": info, "cold_start_s": cold_s, "calls": calls, "token_budget": spec["token_budget"],
                  "planned_calls": [list(p) for p in (plan or [])],
                  "per_request_s": round(sum(walls) / reqs, 3) if reqs else None,
                  "controls": control_kinds(conds), "banks": (prepared or {}).get("banks", []),
                  "split": (prepared or {}).get("split", []),
                  "directions": [c["dir"].describe() for c in conds if c.get("dir") is not None],
                  "ego_cut_identical": ego_identical},
    }


# ----------------------------------------------------------------------------------------------- the CLI


def modal_bin() -> str:
    return str(Path(sys.executable).parent / "modal")


def modal_cli(*args, capture: bool = True) -> str:
    res = subprocess.run([modal_bin(), *args], capture_output=capture, text=True)
    if res.returncode:
        raise SystemExit(f"modal {' '.join(args)} failed: {(res.stderr or res.stdout or '').strip()[-800:]}")
    return res.stdout if capture else ""


def bench_containers() -> list[dict]:
    """The running containers of anatta-explore-bench, and no other app's."""
    rows = json.loads(modal_cli("container", "list", "--json") or "[]")
    return [r for r in rows if r.get("app_name") == APP_NAME]


def stop_containers() -> list[str]:
    stopped = []
    for r in bench_containers():
        assert r["app_name"] == APP_NAME and not r["app_name"].startswith("matlab")
        modal_cli("container", "stop", r["container_id"], "-y")
        stopped.append(r["container_id"])
    return stopped


def cmd_warm(model_key: str, cap: float, root=None) -> dict:
    """Start (or reach) the container and report its readiness; charged as a session."""
    P = paths.Paths(Path(root)) if root else paths.Paths.default()
    session = R.new_run_id()
    meter = M.Meter(spend_dir=P.spend, label="warm", session=session, model_key=model_key, cap_usd=cap,
                    rate=RATE["total"], tail_s=SCALEDOWN_S)
    meter.start(COLD_START_S)
    t = time.time()
    try:
        info = RemoteBench(model_key).info()
    except BaseException:
        meter.close(stopped=True)
        raise
    ready_s = round(time.time() - t, 2)
    meter.after_call(what="info")
    meter.close()
    print(json.dumps({"ready_s": ready_s, "charged_usd": meter.charged_usd, **info}, indent=1, default=str))
    return info


def _control_cond(kind: str, tgt, *, layers=None, match: str = "full", trait=None, seed=None, bank=None) -> dict:
    """A condition as expand() makes it, for build_controls."""
    if kind == OTHER:
        dname, ctl, d = f"{OTHER}({tgt.name})={trait.name}", {"kind": kind, "target": tgt, "match": match}, trait
    else:
        dname, ctl, d = (f"{SPLIT}({tgt.name})#s{seed}",
                         {"kind": kind, "target": tgt, "seed": seed, "bank": bank, "match": match}, None)
    return {"name": f"ablate:{dname}" + (f"~{match}" if match != "full" else ""), "mode": "ablate", "dir": d,
            "dname": dname, "layers_spec": layers, "coefficient": None, "control": ctl}


def build_controls(target, model_key: str, seeds: list[int], *, kind: str = D.RANDOM, traits=None, layers=None,
                   match: str = "full", bank=None, bench=None, cap: float = 1.0, root=None, log=print) -> dict:
    """Controls for a target without generating anything, saved under <results>/<UTC date>/bench-controls/ and
    charged as a session. random_matched: per seed, as before. other_trait: each of `traits` (default NOT_SELF)
    measured against the target at its layers (or `layers`), its table in directions/. split_null: a bank (built, or
    `bank` = {path, tensor_sha256}) and a null per seed, saved in directions/ with their tables. The summary JSON is
    written beside directions/ and returned."""
    P = paths.Paths(Path(root)) if root else paths.Paths.default()
    out = P.sweep_dir(schema.utc_date(), "bench-controls")
    tgt = D.resolve(target, model_key)
    session = R.new_run_id()
    meter = M.Meter(spend_dir=P.spend, label="bench-controls", session=session, model_key=model_key,
                    cap_usd=cap, rate=RATE["total"], tail_s=SCALEDOWN_S, sweep_log=out / "spend.jsonl")
    meter.start(COLD_START_S + 60.0)
    conds, prepared, res = [], {}, None
    try:
        bench = bench or RemoteBench(model_key)
        t = time.time()
        info = bench.info()
        ready_s = round(time.time() - t, 2)
        meter.after_call(what="info")
        prompts, nsha = neutral_set()
        t = time.time()
        if kind == D.RANDOM:
            lay = tgt.layers_for(None, info["n_layers"])
            meter.before_call(30.0 + 2.0 * len(seeds))
            res = bench.random_matched({"target": {"sha256": tgt.sha256, "tensor_sha256": tgt.tensor_sha256,
                                                   **tgt.payload(info["n_layers"])},
                                        "seeds": list(seeds), "layers": lay, "neutral_prompts": prompts,
                                        "neutral_sha256": nsha, "tol": 0.10})
            meter.after_call(what="random_matched")
            saved = [D.save_control(out / "directions", tgt, d["seed"], d["vectors"], d["made"])
                     for d in res["directions"]]
        else:
            if kind == OTHER:
                conds = [_control_cond(kind, tgt, layers=layers, match=match, trait=D.resolve(n, model_key))
                         for n in (traits or K.NOT_SELF)]
            else:
                conds = [_control_cond(kind, tgt, layers=layers, match=match, seed=s, bank=bank) for s in seeds]
            prepared = prepare_controls(bench, conds, info=info, out_dir=out / "directions", prompts=prompts,
                                        neutral_sha=nsha, tol=0.10, meter=meter, log=log)
        build_s = round(time.time() - t, 2)
    finally:
        meter.close()
    if kind == D.RANDOM:
        summary = {"target": tgt.describe() | {"made": None}, "ready_s": ready_s, "build_s": build_s,
                   "container_s": res["meta"]["seconds"], "charged_usd": meter.charged_usd, "controls": []}
        for d in saved:
            lay = d.made["layers"].values()
            summary["controls"].append({
                "seed": d.made["seed"], "file": d.path, "sha256": d.sha256, "tensor_sha256": d.tensor_sha256,
                "ratio_range": d.made["ratio_range"],
                "moment_ratio_range": K.span(K.moment_ratios(d.made).values(), 6),
                "k_range": [min(x["k"] for x in lay), max(x["k"] for x in lay)],
                "plain_random_ratio_range": [min(x["plain_random_ratio"] for x in lay),
                                             max(x["plain_random_ratio"] for x in lay)],
                "n_tokens": d.made["n_tokens"], "neutral": d.made["neutral"]})
        path = out / f"controls-{tgt.name}-{tgt.sha256[:8]}.json"
    else:
        summary = {"kind": kind, "target": tgt.describe(), "ready_s": ready_s, "build_s": build_s,
                   "charged_usd": meter.charged_usd, "session": session, "banks": prepared.get("banks", []),
                   "controls": []}
        for c in conds:
            mk = c["made"]
            per = (c["dir"].made["layers"] if kind == SPLIT else
                   json.loads((out / "directions" / mk["file"]).read_text())["layers"])
            summary["controls"].append({
                "name": c["name"], "direction": c["dir"].name, "tensor_sha256": c["dir"].tensor_sha256,
                "sha256": c["dir"].sha256, "file": c["dir"].path if kind == SPLIT else str(out / "directions" / mk["file"]),
                "layers": c["layers"], "made": mk,
                "by_layer": {str(x): {k: round(per[str(x)][k], 4) for k in ("ratio", "moment_ratio", "cos_target",
                                                                           "abs_cos_chance_p95") if k in per[str(x)]}
                             | ({"fraction": round(c["fractions"][x], 4)} if c.get("fractions") else {})
                             for x in c["layers"]}})
        path = out / f"controls-{kind}-{tgt.name}-{tgt.key[:8]}-{session}.json"
    schema.write_json(path, summary)
    summary["summary_file"] = str(path)
    log(json.dumps({k: v for k, v in summary.items() if k != "controls"}, indent=1, default=str))
    return summary


def cmd_status(root=None) -> None:
    P = paths.Paths(Path(root)) if root else paths.Paths.default()
    day = M.scout_spend.read_day(P.spend)
    state = M.read_state(P.spend)
    now = time.time()
    print(f"containers: {json.dumps(bench_containers())}")
    print(f"today ({day['date']}): ${float(day.get('total_usd', 0)):.4f} of ${M.daily_cap(P.spend)}; "
          f"bench ${float(day.get('by_tool', {}).get('bench', {}).get('usd', 0)):.4f}")
    for k, s in state.items():
        left = float(s.get("paid_until", 0)) - now
        print(f"{k}: paid until {left:.0f}s from now ({s.get('label')}, {s.get('session')})" if left > 0 else
              f"{k}: nothing paid ahead")


def _target_ref(target: str, sha256: str | None, tensor_sha256: str | None):
    """A registry name, or a file path ({path, sha256?, tensor_sha256?})."""
    if target.endswith(".pt") or "/" in target:
        return {"path": target, **({"sha256": sha256} if sha256 else {}),
                **({"tensor_sha256": tensor_sha256} if tensor_sha256 else {})}
    return target


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="bench: the warm steering bench (see the module docstring)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up", help="deploy the app")
    w = sub.add_parser("warm", help="start a container and load the model")
    w.add_argument("--model", default="ministral", choices=MODELS)
    w.add_argument("--cap", type=float, default=1.0)
    r = sub.add_parser("run", help="run a sweep spec")
    r.add_argument("spec")
    r.add_argument("--cap", type=float, default=None, help="this session's cap in USD (default: the spec's)")
    r.add_argument("--stop", action="store_true", help="stop the container afterwards")
    c = sub.add_parser("control", help="build controls for a target, without generating")
    c.add_argument("--kind", default=D.RANDOM, choices=CONTROL_KINDS)
    c.add_argument("--target", default="ego", help="a registry name (default ego) or a direction file's path")
    c.add_argument("--target-sha256", default=None, help="pin a target file by its file sha256")
    c.add_argument("--target-tensor-sha256", default=None, help="pin a target file by its tensor sha256")
    c.add_argument("--model", default="ministral", choices=MODELS)
    c.add_argument("--seeds", default="0,1", help="random_matched and split_null: comma-separated ints")
    c.add_argument("--traits", default=",".join(K.NOT_SELF), help="other_trait: comma-separated registry names")
    c.add_argument("--layers", default=None, help="other_trait and split_null: a layer band (default: the target's)")
    c.add_argument("--match", default="full", choices=tuple(K.MATCHES))
    c.add_argument("--bank", default=None, help="split_null: a stored bank's path (default: build one)")
    c.add_argument("--bank-tensor-sha256", default=None)
    c.add_argument("--cap", type=float, default=1.0)
    sub.add_parser("status", help="containers, today's spend, the paid-until state")
    sub.add_parser("stop", help="stop the warm container(s) and refund the unused tail")
    sub.add_parser("down", help="stop and undeploy the app")
    a = ap.parse_args(argv)
    if a.cmd == "up":
        modal_cli("deploy", str(BENCH / "modal_bench.py"), capture=False)
    elif a.cmd == "warm":
        cmd_warm(a.model, a.cap)
    elif a.cmd == "run":
        man = run_sweep(a.spec, cap=a.cap, stop_after=a.stop)
        raise SystemExit(0 if man["status"] == "complete" else 3)
    elif a.cmd == "control":
        bank = {"path": a.bank, "tensor_sha256": a.bank_tensor_sha256} if a.bank else None
        build_controls(_target_ref(a.target, a.target_sha256, a.target_tensor_sha256), a.model,
                       [int(x) for x in a.seeds.split(",") if x.strip()], kind=a.kind,
                       traits=[x.strip() for x in a.traits.split(",") if x.strip()], layers=a.layers, match=a.match,
                       bank=bank, cap=a.cap)
    elif a.cmd == "status":
        cmd_status()
    elif a.cmd in ("stop", "down"):
        ids = stop_containers()
        if a.cmd == "down":
            modal_cli("app", "stop", APP_NAME, "-y")
        refund = M.refund_tail(paths.Paths.default().spend)
        print(f"stopped {ids or 'no containers'}{'; app stopped' if a.cmd == 'down' else ''}; refunded ${refund:.4f}")


if __name__ == "__main__":
    main()
