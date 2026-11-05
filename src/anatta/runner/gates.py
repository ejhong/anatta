"""The runner's gates and folder rules (clean-lab step 5), and the helpers the temporary reproduction runners share
(moved here from experiments/run_r1b.py so both use one copy). Claude (Opus 5.5) wrote this file, 4 Nov 2026.

  refuse_unless_ready(sha, tier)  a tier "full" spec must be registered (its sha256 in lab/ledger/NOTES.md) and the
                                  work tree clean; an "explore" spec may run unregistered from a dirty tree, and the
                                  commit and dirty flag are recorded (returns git state either way)
  new_run_folder(name, plan)      a fresh <results_root>/runs/<name>/run-<UTC stamp>/ with plan.json written first
  open_run_folder(name, sha, d)   a run folder whose plan.json was written for this spec file's sha256 (d, or the
                                  newest one)
  check_day(usd)                  refuse when today's spend plus usd would pass the day's cap (anatta.spend)
  run_spent(out)                  what a run folder has paid so far (its GPU and judge spend logs)
  meter(...)                      an anatta.spend.Meter for one GPU session, started
  latest(out, name)               a stage file or its newest <stem>.vN<suffix> (anatta.cards force=True)
  check_verified(key, result)     an organism's volume re-hash checked against the registry's files
  file_facts(adapter_dir)         an adapter folder's identity (weights sha256 and bytes, done.json's sha256)

Stdlib only at import (the containers import file_facts).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

PLAN = "plan.json"
TIERS = ("explore", "full")
GPU_LOG, JUDGE_LOG = "gpu_spend.jsonl", "judge_spend.jsonl"  # a run folder's own spend logs (one row a charge)


class Refused(SystemExit):
    """A gate said no; the message says why and what to do."""

    def __init__(self, why: str):
        super().__init__(f"refused: {why}")


# ------------------------------------------------------------------------------------------------- gates


def refuse_unless_ready(sha: str, tier: str = "full", what: str = "this spec", notes=None, code: dict | None = None) -> dict:
    """The git state, after the tier's gates: full needs the sha256 registered and a clean tree; explore needs
    neither (both are recorded by the caller in plan.json and every stage file)."""
    from ..repro import NOTES, git_state, registered

    if tier not in TIERS:
        raise Refused(f"tier {tier!r} is not one of {TIERS}")
    code = code if code is not None else git_state()
    if tier == "full":
        if not registered(sha, notes or NOTES):
            raise Refused(f"spec sha256 {sha} is not registered in lab/ledger/NOTES.md ({what} is tier full; register "
                          "it, commit and push before any result exists)")
        if code.get("dirty") is not False:
            raise Refused(f"the work tree is not clean ({code}); {what} is tier full and runs only from a clean commit")
    return code


def check_day(usd: float, spend_dir=None, what: str = "this stage") -> dict:
    """Today's spend, the day's cap and this stage's estimate; refuses when the sum would pass the cap."""
    from .. import spend

    today, cap = spend.spent_today(spend_dir), spend.daily_cap(spend_dir)
    if today + usd > cap:
        raise Refused(f"today's spend ${today:.2f} plus {what}'s estimate ${usd:.2f} would pass the day's cap ${cap:.2f}")
    return {"today_usd": round(today, 4), "day_cap_usd": cap, "estimate_usd": round(usd, 4)}


# ------------------------------------------------------------------------------------------------- folders


def runs_root() -> Path:
    from ..config import load

    return load().results_root / "runs"


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def new_run_folder(name: str, plan: dict, code: dict, root: Path | None = None) -> Path:
    """A fresh dated folder with plan.json in it before anything runs; refuses an existing folder or plan."""
    from .. import cards

    out = (root or runs_root()) / name / f"run-{stamp()}"
    if out.exists():
        raise Refused(f"{out} exists; each run gets a fresh folder (wait a second and plan again)")
    cards.write_plan(out, plan, code=code)
    return out


def read_plan(out: Path) -> dict:
    p = Path(out) / PLAN
    if not p.exists():
        raise Refused(f"{out} holds no plan.json (run --stage plan first)")
    return json.loads(p.read_text(encoding="utf-8"))


def plan_body(out: Path) -> dict:
    """The runner's own plan record (anatta.cards.write_plan keeps a dict plan under "spec")."""
    return read_plan(out)["spec"]


def open_run_folder(name: str, spec_sha: str, run_dir: str | os.PathLike = "", root: Path | None = None) -> Path:
    """run_dir (its plan must be this spec file's) or, without one, the newest run folder of this spec whose plan is."""
    if run_dir:
        out = Path(run_dir)
        if plan_body(out).get("spec_file_sha256") != spec_sha:
            raise Refused(f"{out}/plan.json was written for another version of the spec (sha256 differs); plan again")
        return out
    base = (root or runs_root()) / name
    for d in sorted(base.glob("run-*"), reverse=True) if base.is_dir() else []:
        if (d / PLAN).exists() and plan_body(d).get("spec_file_sha256") == spec_sha:
            return d
    raise Refused(f"no run folder under {base} was planned for this spec's sha256 {spec_sha[:12]}; run --stage plan")


def latest(out, name: str) -> Path | None:
    """A stage file or, when reruns wrote <stem>.vN<suffix> beside it (anatta.cards, force=True), the newest."""
    p = Path(out) / name
    vs = sorted(p.parent.glob(f"{p.stem}.v*{p.suffix}"), key=lambda x: int(x.name[len(p.stem) + 2:-len(p.suffix)]))
    return vs[-1] if vs else p if p.exists() else None


def read_latest(out, name: str) -> dict | None:
    p = latest(out, name)
    return json.loads(p.read_text(encoding="utf-8")) if p else None


def run_spent(out) -> float:
    """What a run folder has paid: the sum of its GPU and judge spend logs."""
    from .. import spend

    return round(spend.sweep_spent(Path(out) / GPU_LOG) + spend.sweep_spent(Path(out) / JUDGE_LOG), 6)


# ------------------------------------------------------------------------------------------------- spend


def meter(tool: str, model_key: str, label: str, cap: float, first_s: float, rate: float, tail_s: float,
          spend_dir=None, sweep_log=None):
    """A started Meter for one GPU session (container seconds at rate, the scaledown tail included)."""
    from .. import spend

    m = spend.Meter(tool=tool, spend_dir=spend_dir, label=label, session=f"{label}-{int(time.time())}",
                    model_key=model_key, cap_usd=cap, rate=rate, tail_s=tail_s, sweep_log=sweep_log)
    m.start(first_s)
    return m


# ------------------------------------------------------------------------------------------------- identity


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_verified(key: str, result) -> dict:
    """An organism's re-hash on the volume (what anatta.organisms.verify returned there: {file: {bytes, sha256}}),
    compared here with the registry's files; refuses an error-shaped result or any missing or different file."""
    from .. import organisms

    want = organisms.get(key)["files"]
    if not isinstance(result, dict) or "error" in result or not result:
        raise Refused(f"organism {key}: the volume check returned {str(result)[:200]!r}, not the files' facts")
    bad = [f for f, w in want.items() if not isinstance(result.get(f), dict)
           or result[f].get("bytes") != w.get("bytes") or result[f].get("sha256") != w.get("sha256")]
    if bad:
        raise Refused(f"organism {key}: {bad} on the volume differ from the registry (bytes or sha256)")
    return result


def file_facts(adapter_dir: str) -> dict:
    """An adapter folder's identity: adapter_model.safetensors' sha256 and bytes, and the sha256 of the done.json beside
    it (finetune writes <out>/done.json, its stamp, next to <out>/adapter). Run in the containers."""
    w = f"{adapter_dir}/adapter_model.safetensors"
    done = f"{os.path.dirname(adapter_dir.rstrip('/'))}/done.json"
    return {"sha256": sha256_file(w) if os.path.exists(w) else None,
            "bytes": os.path.getsize(w) if os.path.exists(w) else None,
            "done_sha256": sha256_file(done) if os.path.exists(done) else None}
