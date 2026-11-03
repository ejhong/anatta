"""The runner's interface (stub). Claude (Opus 5.5) wrote this file, 2 Nov 2026. The Modal runner itself is the next
build step; nothing here starts a container, calls a GPU or an API, or spends money.

What is built: plan() turns a frozen candidate into a RunPlan (the arms, each arm's intervention in bench's payload
form, the measures the tier runs), and write_condition() writes one per-condition result file in card.py's format.
What is a stub: ModalRunner.run().

The runner must (spec v2 sections 2 and 5):
  1. write the plan (RunPlan.as_dict(), with the candidate's sha256 and code hashes) before any result, check the daily
     cap (results/explore/spend/), and start with a trial at the real batch size;
  2. serve every arm from one Modal service per model: wrap bench's Service (lab/explore/bench/steering.py:
     Service.generate takes {"requests": [{rid, messages, temperature, max_tokens, seed, intervention}],
     "directions": {tensor_sha256: {...}}}, groups requests by intervention, attaches hooks with anatta.dial.intervene
     and checks they detach) for steer/ablate and their controls; noself3's hooks (lab/explore/noself/noself3.py:
     steer_ivs -> k5.conditions(model, vhat, layer, c)["steered"], run_condition3) for the instruments (k5.run_items),
     HI-ToM/MMLU (noself.letters_items) and behavior sampling (behavior_replies3); vLLM for prompt/adapter/checkpoint
     arms; OpenRouter candidates interleave base and candidate calls in one session [F7];
  3. build the controls with bench's existing code, not new code: random_matched (Service.random_matched, method
     pc_mix_v1), other_trait (controls.made_other on the lab's trait vectors, match partial_moment), split_null
     (controls.split_null on a contrastive bank; its |cos| with the target and chance level are recorded), topic (a
     pinned direction file); prompt controls from registered texts; adapter twins with >= 3 seeds;
  4. pair samples across arms: a sample's seed comes from (seed, item, sample), the same under every intervention
     (bench's rule), so item-level differences are paired;
  5. judge: the coherence gate per reply before any behavior count, scout's judge for behavior families, each
     measure's own adapter for public anchors (K6/K7/C3 code by import), and scout's rejudge on a stratified 10%;
  6. write one condition file per arm with write_condition(), then card.from_files() writes card.json and card.md to
     results/harness/<date>/<candidate>/; full tier reads held-out items once per registered claim, after
     candidate.check_frozen().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .candidate import Candidate, resolve_measures
from .card import FORMAT


def results_dir() -> Path:
    """<results_root>/harness, results_root from lab.toml (anatta.config)."""
    from anatta.config import load

    return load().results_root / "harness"


class RunnerNotBuilt(NotImplementedError):
    pass


def to_bench_intervention(spec: dict) -> dict:
    """An arm's spec (candidate intervention or control) in bench's intervention form (client.py's spec forms and
    Service.generate's payload). Prompt/adapter/checkpoint arms are not bench arms: they return {"mode": "none"} plus
    the field the vLLM path needs."""
    k = spec.get("kind", "none")
    if k == "none":
        return {"mode": "none"}
    if k in ("steer", "ablate"):
        if spec.get("position", "all") != "all":
            raise RunnerNotBuilt("bench and k5 hooks act at every position; other position rules need new hook code")
        mode = "add" if k == "steer" else "ablate"
        d = spec["direction"]
        out = {"mode": mode, "layers": list(spec["layers"])}
        if k == "steer":
            out["coefficient"] = float(spec["strength"])
        if isinstance(d, dict):
            out.update({"direction": d["name"], "tensor_sha256": d["tensor_sha256"]})
            if d.get("path"):
                out["path"] = d["path"]
        else:  # a bench-built control: random_matched | other_trait | split_null
            out.update({"direction": d, "target": spec["target"]})
            for f in ("seeds", "traits", "match", "bank"):
                if f in spec:
                    out[f] = spec[f]
        return out
    if k == "prompt":
        return {"mode": "none", "prompt": {f: spec[f] for f in ("system_prompt", "prefix", "slot", "rule", "chars")
                                           if f in spec}}
    if k in ("adapter", "checkpoint"):
        return {"mode": "none", k: {f: spec[f] for f in ("path", "sha256", "seeds") if f in spec}}
    raise ValueError(f"unknown arm kind {k!r}")


@dataclass
class RunPlan:
    candidate_name: str
    candidate_sha256: str
    model: dict
    tier: str
    arms: list
    measures: list
    out_dir: str
    cap_usd: float
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def plan(c: Candidate, registry: dict, date: str, cap_usd: float | None = None) -> RunPlan:
    arms = [{"role": a["role"], "name": a["name"], "spec": a["spec"], "bench": to_bench_intervention(a["spec"])}
            for a in c.conditions()]
    notes = []
    if c.tier == "full":
        notes.append("full tier: held-out items read once per registered claim; check_frozen() before the read")
    return RunPlan(candidate_name=c.name, candidate_sha256=c.sha256, model=c.model, tier=c.tier, arms=arms,
                   measures=resolve_measures(c, registry), out_dir=str(results_dir() / date / c.name),
                   cap_usd=cap_usd if cap_usd is not None else (15.0 if c.tier == "quick" else 70.0), notes=notes)


class Runner(Protocol):
    def run(self, p: RunPlan) -> list[Path]:
        """Run every arm of the plan and return the per-condition result files (card.py's format)."""
        ...


class ModalRunner:
    """Stub. See the module docstring for what run() must do."""

    app_name = "anatta-harness"  # never a matlab-* app

    def run(self, p: RunPlan) -> list[Path]:
        raise RunnerNotBuilt("the Modal runner is the next build step (lab/plans/harness-spec-2026-11-02.md, step 2)")


def write_condition(out_dir: Path | str, *, candidate: str, candidate_sha256: str | None, condition: str, role: str,
                    model: str, revision: str | None, spec: dict, measures: dict, cost_usd: float | None = None,
                    provenance: dict | None = None, deviations: list | None = None) -> Path:
    """One per-condition result file. measures = {id: {"units": {unit: value}, "n_samples": int, "unparsed": int}}."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in condition)
    p = Path(out_dir) / f"{safe}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": FORMAT, "candidate": candidate, "candidate_sha256": candidate_sha256, "condition": condition,
           "role": role, "model": model, "revision": revision, "spec": spec,
           "measures": {k: measures[k] for k in sorted(measures)}, "cost_usd": cost_usd,
           "provenance": provenance or {}, "deviations": deviations or []}
    p.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    return p
