"""The lab's runner entry (clean-lab step 5): one spec, five stages. Thin: the stages live in anatta.runner.stages;
this file holds only the Modal app "anatta-lab" (one vLLM generator, parameterized by model, and the organism check
on the volume). Claude (Opus 5.5) wrote this file, 4 Nov 2026.

  tools/modal.sh run experiments/run.py::main --spec experiments/acceptance/k6b_orbench_subsample.yaml --stage plan --dry-run
  tools/modal.sh run experiments/run.py::main --spec <spec> --stage plan            # fresh run folder, plan.json
  tools/modal.sh run experiments/run.py::main --spec <spec> --stage trial           # one call at the real batch
  tools/modal.sh run experiments/run.py::main --spec <spec> --stage run             # every condition (needs an ok trial)
  tools/modal.sh run experiments/run.py::main --spec <spec> --stage score           # judge + score (no GPU)
  tools/modal.sh run experiments/run.py::main --spec <spec> --stage card            # card.json (no GPU)
  .venv/bin/python experiments/run.py --spec <spec> --stage plan|score|card [--dry-run] [--run-dir D]   # no Modal
  ... --stage plan --reuse-run <earlier run folder>    # a re-grade: the trial and run copy its chunks, no GPU, no spend

Without --run-dir a stage uses the newest run folder planned for this spec file's sha256. Printed: plans, stage
records and cards (numbers only); never an item or a reply. No `from __future__ import annotations`: Modal reads
parameter annotations.
"""

import json
import sys
from pathlib import Path

try:
    import modal
except ModuleNotFoundError:  # .venv: the GPU-free stages run without Modal
    modal = None

REPO = Path(__file__).resolve().parents[1]
APP = "anatta-lab"
NO_GPU = ("plan", "score", "card")  # trial and run also run here for a re-grade (--reuse-run): no generation


def show(stage: str, path, rec: dict) -> None:
    """A stage's record, without per-unit values (ids and 0/1 per item add nothing to a console)."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "units"}
        return x

    print(json.dumps(strip(rec), indent=1, default=str))
    print(f"{stage} ->", path if path else "(dry run: nothing written)", flush=True)


class NoRemote:
    """The laptop's stand-in for the GPU: enough for a re-grade (--reuse-run), where every chunk is copied; anything
    that would need the GPU or the volume refuses."""

    def generate(self, model_key, lora, requests):
        raise SystemExit("this chunk is not in the reused run: generation needs the GPU (tools/modal.sh run "
                         "experiments/run.py::main ...)")

    def verify_organisms(self, keys):
        raise SystemExit("organisms are verified on the Modal volume: tools/modal.sh run experiments/run.py::main ...")


def dispatch(spec: str, stage: str, run_dir: str = "", dry_run: bool = False, remote=None, reuse_run: str = "") -> None:
    from anatta.runner import stages as S

    if stage not in S.STAGES:
        raise SystemExit(f"unknown stage {stage!r}; one of {S.STAGES}")
    if dry_run and stage != "plan":
        raise SystemExit("--dry-run applies to --stage plan (the other stages refuse or skip on their own)")
    if reuse_run and stage != "plan":
        raise SystemExit("--reuse-run applies to --stage plan (later stages read it from plan.json)")
    if stage == "plan":
        p, rec = S.plan(spec, dry_run=dry_run, reuse_run=reuse_run)
    elif stage in ("trial", "run"):
        p, rec = getattr(S, stage)(spec, run_dir, remote=remote or NoRemote())
    else:
        p, rec = getattr(S, stage)(spec, run_dir)
    show(stage, p, rec)


if modal is not None:
    from anatta.backends.vllm import image as vllm_image
    from anatta.modal_app import CACHE, RUNS, cache, runs
    from anatta.runner.stages import GEN_CPU, GEN_MEM_MB, GPU, SCALEDOWN

    PKG = REPO / "src" / "anatta" if (REPO / "src" / "anatta").is_dir() else Path("/root/anatta")  # laptop | container
    image = vllm_image(str(PKG))
    app = modal.App(APP)

    @app.cls(image=image, gpu=GPU, cpu=GEN_CPU, memory=GEN_MEM_MB, timeout=3600, scaledown_window=SCALEDOWN,
             max_containers=1, startup_timeout=1800, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()})
    class Gen:
        model_key: str = modal.parameter()
        lora: bool = modal.parameter(default=False)

        @modal.enter()
        def load(self):
            from anatta.backends.vllm import Engine, Settings

            self.engine = Engine(Settings.for_model(self.model_key, generation_config="vllm", enable_lora=self.lora,
                                                    seed=0))
            self._info = self.engine.info()

        @modal.method()
        def generate(self, requests: list) -> dict:
            import time

            t0 = time.time()
            res = self.engine.generate(requests)
            for r in res:
                r.pop("token_ids", None)
                r.pop("prompt_ids", None)
            return {"results": res, "gen_s": round(time.time() - t0, 1), "info": self._info}

    @app.function(image=image, cpu=2.0, memory=4096, timeout=1800, volumes={RUNS: runs.read_only()})
    def verify_organisms_remote(keys: list) -> dict:
        """anatta.organisms.verify(key, RUNS) for each key, on the volume (re-hashes before use)."""
        from anatta import organisms

        out = {}
        for k in keys:
            v = organisms.verify(k, RUNS)
            out[k] = v if isinstance(v, (dict, list, str, bool, int, float, type(None))) else str(v)
        return out

    class ModalRemote:
        def generate(self, model_key: str, lora: bool, requests: list) -> dict:
            return Gen(model_key=model_key, lora=lora).generate.remote(requests)

        def verify_organisms(self, keys: list) -> dict:
            return verify_organisms_remote.remote(keys)

    @app.local_entrypoint()
    def main(spec: str, stage: str = "plan", run_dir: str = "", dry_run: bool = False, reuse_run: str = ""):
        dispatch(spec, stage, run_dir, dry_run, remote=ModalRemote(), reuse_run=reuse_run)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="the runner's GPU-free stages (plan, score, card; trial and run of a "
                                             "re-grade planned with --reuse-run)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--stage", default="plan", choices=(*NO_GPU, "trial", "run"))
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reuse-run", default="", help="plan only: an earlier run folder whose generation is reused")
    a = ap.parse_args()
    sys.exit(dispatch(a.spec, a.stage, a.run_dir, a.dry_run, reuse_run=a.reuse_run))
