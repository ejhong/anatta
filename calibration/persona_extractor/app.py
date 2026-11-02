"""Calibration check 4, steering part 2, on Modal: does anatta.probe.extract give the persona-vectors extractor's vectors
(Chen et al., arXiv 2507.21509; generate_vec.py at safety-research/persona_vectors b8e0f04) on the same trait data on
Qwen2.5-7B-Instruct? README.md has the claim, the sources and the numbers; config.yaml every setting; traits.py,
generate.py, readings.py, reference.py and analysis.py the code that runs in the containers; card.py the card.

Stages, each a Modal function writing under out_dir(plan) on the anatta-runs volume, in this order:
  prepare     (its own entrypoint, run once before the first launch) CPU: the three trait files fetched from the repo at
              b8e0f04 onto the volume, each git blob sha and size checked, their pairs and questions counted. Their
              sha256 and counts are written into config.yaml, to be committed before any run. Their text never reaches
              the laptop.
  G           L40S, the lab's image: the shared responses, Hugging Face generate at the repo's settings, in bf16.
  full_batch  the smoke only, L40S: one batch of G at its registered size (200 prompts of one sign, up to 1,000 new
              tokens) and settings, timed, with its peak memory; read as nothing, its answers never kept.
  keep        the CSV step in the second image (its pandas): the CSVs and the pairs every extractor reads. Without
              the second image, keep_from_slices in the lab's image.
  verbatim    H100, else A100-80GB, the second image: R, generate_vec.py verbatim, once per trait (fp32 without
              TF32 on either GPU; the card records which ran).
  readings    L40S: P1, P1-trim and P1-128 from one pass, and P0, over G's kept rows, in bf16.
  rlab32      L40S with 64 GB (the fp32 model loads on the CPU first): R-lab32.
  probe_graded
              L40S: the arms C4 needs, E-sys and E-sys' (the verdict's ceiling) on evil and sycophantic, through the
              recorder.
  Gprime      L40S: G' at probe's settings, read as P1 and P0 in the same container.
  probe       L40S: the other arms (E and E' on every trait, E-sys and E-sys' on optimistic), E-trail-128 and E-det.
  mprime      L40S: M' on Ministral (ledger only).
  analysis    CPU, next to the vectors: every number, as numbers.
Everything the graded items read (G, the CSVs, R, the readings, R-lab32 and C4's arms) runs before anything only
reported or only in the ledger (G', the other arms, M'), so a stop lands on report-only work.
Then, only if C4-base misses and the judged switch is on: judge (CPU, OpenRouter), the judged CSVs, R-judged and the
analysis again. Then the card (card.py), here.

Spend. After each function the laptop prints its estimated cost (the call's wall time at config.yaml's rates for the
GPU the container reported and its CPU and memory, so an overestimate: it counts queueing and start-up) and the
total, kept in spend.json in the run's folder; the smoke's spend and the full run's count toward the one stop, each
run adding the other's spend.json. $0.50 of the $20 stop is held back for the analysis and the card. The ceiling is
hard: no commit unit starts unless the spend so far plus its upper estimate is within the limit, $19.50 for the GPU
and CSV stages and $20 for the analysis. The laptop checks each container's start-up and first unit at the GPU it
asks for (a CPU stage as a whole); inside, Ceiling checks each unit (a batch, a chunk, a trait, a call) at the GPU
the container got, counting its time since the call (CheckError ceiling). The upper estimates are config.yaml's
cost.upper_s, from the registered times, raised by the smoke's measures where those are higher (smoke_measured).
A refused unit stops its stage with its work so far on the volume; the laptop skips the remaining GPU and CSV
stages, runs the analysis on what exists if it fits, and writes the card, which says why the run stopped. Modal's
usage page is the authority. Timeouts: G, readings, rlab32 and verbatim 2 hours each; Gprime, both probe stages,
mprime and the analysis 1 hour; the smoke's full batch 30 minutes; no function retries.

The data stay on the volume: the containers return numbers, counts, names and hashes only (traits.written checks every
result), and every exception leaves a container as a StageError naming only the stage, the row and the type
(traits.run_guarded).

A run is its plan: every setting the GPU and CSV stages read, the sha256 of the library files and the data code
(traits.DATA_FILES), the sha256 of this file's container functions but the analysis stage (container_sha256), and both
images' pins. Each container first checks that its data code and installed versions are the plan's. A rerun of the
same plan resumes from the volume; a changed plan runs afresh in a new folder. The analysis has its own plan
(card.analysis_plan: config.yaml's analysis and registered blocks, analysis.py, card.py, this file's analysis stage,
and the run's plan id) and its own id: a change to any of those reruns only the analysis, from the stored data, under
the new id, which the card records. The cheap tier checks the data code only and records both ids.

Usage, from the repo root (the modal CLI; PYTHONPATH=src so the app imports the package; ANATTA_DIRECTIONS points at
the folder of stored directions, private, for M' and the ledger's comparisons: a full run refuses to start unless it
holds the eight files config.yaml pins, with their sha256s; the smoke warns):
  PYTHONPATH=src modal run calibration/persona_extractor/app.py::prepare
  PYTHONPATH=src modal run calibration/persona_extractor/app.py::main --results DIR --smoke
  PYTHONPATH=src modal run calibration/persona_extractor/app.py::main --results DIR
  PYTHONPATH=src modal run calibration/persona_extractor/app.py::card --results DIR [--smoke]
  PYTHONPATH=src modal run calibration/persona_extractor/app.py::cheap --results DIR
Keep the terminal open while main runs (no --detach): the stages are driven from the laptop.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis as A  # noqa: E402
import generate as G  # noqa: E402
import readings as Rd  # noqa: E402
import reference as Rf  # noqa: E402
import traits as T  # noqa: E402

from anatta.modal_app import (  # noqa: E402
    CACHE,
    DIRECTIONS,
    GPU,
    PKG,
    REMOTE_DIRECTIONS,
    RUNS,
    cache,
    image,
    run_root,
    runs,
)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


# image_pins is check 1's (anatta-calib-c1, src/anatta/bench/fingerprint.py at 44760f7), copied: that file is not on
# calib/instruments. Its plan_id is copied into traits.py.
def image_pins(modal_app_path) -> dict[str, str]:
    """{package: version} for every "name==version" string passed to uv_pip_install in the first module-level
    assignment to `image` in modal_app.py. Raises if there is none."""
    tree = ast.parse(Path(modal_app_path).read_text())
    pins: dict[str, str] = {}
    node = next((n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "image" for t in n.targets)), None)
    for call in ast.walk(node.value) if node is not None else ():
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "uv_pip_install":
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "==" in arg.value:
                    name, version = arg.value.split("==", 1)
                    pins[name.strip()] = version.strip()
    if not pins:
        raise ValueError(f"no name==version pins in the image of {modal_app_path}")
    return dict(sorted(pins.items()))


plan_id = T.plan_id  # check 1's, copied into traits.py so the card names both plans the same way


CFG = load_yaml(HERE / "config.yaml")
LAB_PINS = image_pins(PKG / "modal_app.py")
REPO_PINS = dict(CFG["repo"]["pins"])
R_GPUS = [CFG["reference"]["gpu"], CFG["reference"]["fallback_gpu"]]
SECRETS = [modal.Secret.from_name(CFG["judged"]["secret"])] if CFG["switches"]["judged"] else []
# Expandable segments let the allocator use memory it has reserved without fragmenting it; no arithmetic changes, and
# the plan doesn't hash it (as check 1 launches, anatta-calib-c1 calibration/model_check/app.py:326 at 44760f7).
GPU_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def with_check(img):
    """An image with this folder's files at /root (app.py imports the modules in every container)."""
    for f in (*T.IMAGE_FILES, "config.yaml"):
        img = img.add_local_file(str(HERE / f), f"/root/{f}")
    return img


# The lab's image is modal_app's (its pins, the package at /root/anatta, and the stored directions when the folder is
# there at launch) with this folder's files added: the lab's image gains no package.
lab_image = with_check(image)
# The second image: the repo's requirements.txt pins (vllm not installed); numpy, safetensors, tokenizers and
# huggingface_hub at whatever pip resolves, recorded by each container. It carries the package too, since app.py
# imports anatta.modal_app (standard library and modal only) in every container.
# Ruled (NOTES, 12:20 UTC, ruling 6): the second image runs Python 3.11, which every pin supports.
repo_image = with_check(modal.Image.debian_slim(python_version="3.11")
                        .uv_pip_install(*[f"{n}=={v}" for n, v in REPO_PINS.items()])
                        .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root",
                              "HF_HUB_OFFLINE": "1"})
                        .add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"]))
app = modal.App("anatta-calib-persona-extractor", image=lab_image)


def out_dir(plan: dict) -> str:
    """A run's folder on the anatta-runs volume."""
    return f"{run_root(plan['models']['qwen']['id'])}/persona_extractor/{'smoke' if plan['smoke'] else 'full'}/" \
           f"{plan_id(plan)}"


def snapshot(plan: dict, key: str) -> str:
    """A model's folder in the cache volume at its pinned revision (loaded from it, offline)."""
    m = plan["models"][key]
    return str(T.snapshot_dir(m["id"], m["revision"]))


def offline() -> None:
    """The containers read the weights from the volume and download nothing; set before transformers loads."""
    os.environ["HF_HUB_OFFLINE"] = "1"


def log(*a) -> None:
    print(*a, flush=True)


class Ceiling:
    """The hard ceiling inside a GPU container, from the laptop's `ceiling` (Spend.ceiling). Before each commit unit
    (a batch of G or G', a chunk of readings, a trait of R-lab32 or R, a probe.extract call, a trail; traits.before)
    the spend so far (the laptop's at the call, plus this container's wall time since the call at the rate of the GPU
    it got, with its cores and memory) plus the unit's upper estimate at that rate must be within the limit, or the
    unit doesn't start (CheckError ceiling). Called, it commits the volume, then stops the stage if the spend has
    passed the limit all the same (a unit that ran past its upper estimate: CheckError budget). It only stops a stage
    and never changes what one writes, so the plan doesn't hash it: a relaunch under another stop or other estimates
    resumes the same plan. Without a `ceiling` (direct calls in the tests) it only commits."""

    def __init__(self, c: dict | None):
        self.c, self.t0 = c, time.time()
        self.gpu = gpu_key(T.gpu_name(), c["gpu"]) if c is not None else None

    def rate(self) -> float:
        g = self.c["gpu_usd_h"]
        return (g.get(self.gpu, max(g.values())) if self.gpu else 0.0) + self.c["tier_usd_h"]

    def spent(self) -> float:
        now = time.time()
        return self.c["spent_usd"] + max(now - self.c["started_at"], now - self.t0) / 3600 * self.rate()

    def seconds(self, kind: str, rows: int | None = None, tokens: int | None = None) -> float:
        """The unit's upper estimate at this container's GPU: config.yaml's, or the smoke's measure where higher. For
        G, the seconds of the smoke's full batch. For R, the smoke's seconds outside its neg pass (the load, the pos
        pass with its warm-up, the save) plus the neg pass's seconds a token times this trait's tokens, or its seconds
        a row times its rows where that is more, scaled by the registered times when the smoke's R ran on the other
        GPU."""
        u = self.c["upper_s"][kind]
        s = float(u.get(self.gpu, max(u.values())) if isinstance(u, dict) else u)
        m = (self.c.get("measured") or {}).get(kind) or {}
        if m.get("seconds") is not None:
            s = max(s, float(m["seconds"]))
        if m.get("s_per_token") is not None and rows is not None and tokens is not None:
            est = m["fixed_s"] + max(m["s_per_token"] * tokens, m["s_per_row"] * rows)
            if isinstance(u, dict) and m.get("gpu") != self.gpu and m.get("gpu") in u and self.gpu in u:
                est *= u[self.gpu] / u[m["gpu"]]
            s = max(s, est)
        return s

    def before(self, kind: str, rows: int | None = None, tokens: int | None = None) -> None:
        if self.c is None:
            return
        usd = self.spent() + self.seconds(kind, rows, tokens) / 3600 * self.rate()
        if usd > self.c["limit_usd"]:
            raise T.CheckError("ceiling", usd=round(usd, 4), limit_usd=self.c["limit_usd"])

    def __call__(self) -> None:
        runs.commit()
        if self.c is not None and self.spent() > self.c["limit_usd"]:
            raise T.CheckError("budget", usd=round(self.spent(), 4))


def stage_file(plan: dict, name: str, analysis_id: str | None = None) -> Path:
    """A stage's result on the volume: stages/<name>.json, or stages/<name>/<analysis id>.json for an analysis."""
    d = Path(out_dir(plan)) / "stages"
    return d / name / f"{analysis_id}.json" if analysis_id else d / f"{name}.json"


def stage(plan: dict, name: str, stack: str, body, aplan: dict | None = None) -> dict:
    """One stage, guarded: the container's data code and packages checked against the plan (and, for an analysis, its
    analysis code against aplan's), then body(guard); the result (with the environment, the pip freeze, the GPU, the
    seconds, the plan id and, for an analysis, the analysis id) checked text-free, written to stage_file and returned.
    Any exception leaves as a StageError naming only the stage, the row and the type."""
    clock = T.Clock()
    aid = plan_id(aplan) if aplan is not None else None

    def go(guard):
        env = T.check_environment(plan, stack, HERE, PKG)
        if aplan is not None:
            got = T.analysis_code_hashes(HERE)
            if any(aplan["code"].get(f) != sha for f, sha in got.items()):
                raise T.CheckError("env_code")
            env["analysis_code"] = got
        return {"env": {**env, "freeze": T.pip_freeze(), "gpu": T.gpu_name(), "python": sys.version.split()[0]},
                "result": body(guard)}

    try:
        res = T.run_guarded(name, go)
    finally:
        runs.commit()
    res["seconds"] = clock.seconds()
    ids = {"plan_id": plan_id(plan), **({"analysis_id": aid} if aid else {})}
    out = T.written(stage_file(plan, name, aid), {"stage": name, **ids, **res})
    runs.commit()
    return out


def ego_trait(plan: dict, name: str):
    """An ego trait from the library's traits folder, its sha256 checked against the registered one."""
    from anatta.probe import TRAITS_DIR, Trait

    p = TRAITS_DIR / f"{name}.yaml"
    if T.sha256_file(p) != plan["mprime"]["sha256"][name]:
        raise T.CheckError("sha256")
    return Trait.load(str(p))


# This file's code that runs in the containers, in two parts: the data stages' (hashed into the plan) and the analysis
# stage's (hashed into the analysis plan by card.analysis_plan, which reads ANALYSIS_FUNCTIONS from card.py).
CONTAINER_FUNCTIONS = ("out_dir", "snapshot", "offline", "log", "stage_file", "stage", "ego_trait",
                       "fetch_traits", "generate_stage", "fullbatch_stage", "keep_stage", "csv_stage",
                       "readings_stage", "rlab32_stage", "verbatim_stage", "probe_stage", "mprime_stage",
                       "cheap_stage", "judge_stage", "stored")
ANALYSIS_FUNCTIONS = ("analysis_stage",)


def container_source(source: str | None = None) -> str:
    """The source of CONTAINER_FUNCTIONS (each with its decorator), read from this file's text (or `source`); as
    check 7's app.py does (anatta-calib-c7 calibration/relearning/app.py at 02db868), through traits.function_source."""
    return T.function_source(Path(__file__).read_text() if source is None else source, CONTAINER_FUNCTIONS)


@functools.cache
def container_sha256(source: str | None = None) -> str:
    """The plan's hash of this file: only its data stages' code that runs in the containers. Cached for the process, so
    every stage of one launch carries the source its containers run."""
    return T.sha256_bytes(container_source(source).encode())


# ------------------------------------------------------------------------------------------------ containers


@app.function(volumes={RUNS: runs}, cpu=1.0, memory=2048, timeout=900)
def fetch_traits(repo: dict, names: list, data_dir: str, prompts: dict) -> dict:
    """prepare: the trait files fetched at the commit, checked and written to data_dir (traits.prepare), and the
    judged arm's eval/prompts.py the same way (reference.prepare_prompts, ruling 10). Returns {traits: their bytes,
    blob, sha256 and counts, prompts: its bytes, blob, sha256 and the form that binds coherence_0_100}."""
    runs.reload()

    def both(guard):
        return {"traits": T.prepare(repo, names, data_dir, guard=guard),
                "prompts": Rf.prepare_prompts(repo, prompts, data_dir)}

    res = T.run_guarded("prepare", both)
    T.assert_text_free(res)
    runs.commit()
    return res


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=2 * 3600)
def generate_stage(plan: dict, arm: str, ceiling: dict | None = None) -> dict:
    """G or G' (generate.run_arm) on Qwen in bf16 (anatta.models.load from the cache volume at the pinned revision),
    the model's generation_config checked first; G' is read as P1 and P0 in the same container (readings.read_arm)."""
    commit = Ceiling(ceiling)
    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        from anatta.models import load

        out = out_dir(plan)
        data = T.load_traits(plan)
        model, tok = load(snapshot(plan, "qwen"))
        facts = G.model_facts(model)
        G.check_generation_config(model, plan["models"]["qwen"]["generation_config"])
        res = {"model": facts, "generate": G.run_arm(model, tok, plan, arm, out, data, commit, log, guard)}
        if arm == "Gprime":
            lay, n_s = plan["layout"], plan["Gprime"]["samples"]
            every = {t: set(range(lay["questions"] * lay["instructions"] * n_s)) for t in plan["traits"]}
            res["readings"] = Rd.read_arm(model, tok, plan, "Gprime", out, data, every, commit, log, guard)
        return res

    return stage(plan, arm, "lab", body)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=1800)
def fullbatch_stage(plan: dict, ceiling: dict | None = None) -> dict:
    """The smoke's full-size batch of G (generate.full_batch) on Qwen in bf16: 200 prompts of one sign, up to 1,000 new
    tokens at G's registered settings; its seconds and peak memory, read as nothing, its answers never kept."""
    commit = Ceiling(ceiling)
    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        from anatta.models import load

        data = T.load_traits({**plan, "traits": [plan["full_batch"]["trait"]]})
        model, tok = load(snapshot(plan, "qwen"))
        G.check_generation_config(model, plan["models"]["qwen"]["generation_config"])
        commit.before("G_batch")
        res = {"model": G.model_facts(model), "full_batch": G.full_batch(model, tok, plan, data, log)}
        commit()
        return res

    return stage(plan, "full_batch", "lab", body)


@app.function(volumes={RUNS: runs}, cpu=2.0, memory=8192, timeout=900)
def keep_stage(plan: dict) -> dict:
    """Without the second image: keep.json from the answers' slices under the lab's tokenizer (keep_from_slices)."""
    runs.reload()
    return stage(plan, "keep", "lab", lambda guard: G.keep_from_slices(plan, out_dir(plan)))


@app.function(image=repo_image, volumes={CACHE: cache, RUNS: runs}, cpu=4.0, memory=16384, timeout=3600)
def csv_stage(plan: dict, judged: bool = False) -> dict:
    """The CSV step in the second image (reference.run_csv): its pandas, its tokenizer from the cache volume; with
    judged, the judged arm's scores as the score columns."""
    import tempfile

    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        import pandas as pd
        from transformers import AutoTokenizer

        out = out_dir(plan)
        tok = AutoTokenizer.from_pretrained(snapshot(plan, "qwen"))
        scores = Rf.judged_scores(out, plan) if judged else None
        with tempfile.TemporaryDirectory() as tmp:
            return Rf.run_csv(plan, out, T.load_traits(plan), tok, pd, tmp, scores, guard)

    return stage(plan, "csv_judged" if judged else "keep", "repo", body)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=2 * 3600)
def readings_stage(plan: dict, ceiling: dict | None = None) -> dict:
    """The lab's readings of G's kept rows (readings.read_arm): P1, P1-trim, P1-128 and P0, in bf16."""
    commit = Ceiling(ceiling)
    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        from anatta.models import load

        out = out_dir(plan)
        model, tok = load(snapshot(plan, "qwen"))
        return {"model": G.model_facts(model),
                "readings": Rd.read_arm(model, tok, plan, "G", out, T.load_traits(plan), G.keep_rows(out),
                                        commit, log, guard)}

    return stage(plan, "readings", "lab", body)


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=2 * 3600)
def rlab32_stage(plan: dict, ceiling: dict | None = None) -> dict:
    """R-lab32 (reference.run_rlab32): the vendored get_hidden_p_and_r on Qwen loaded by anatta.models.load in fp32
    (about 30 GB, on the CPU first, hence the memory), under an outer no_grad."""
    commit = Ceiling(ceiling)
    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        from anatta.models import load

        out = out_dir(plan)
        g = plan["repo"]["generate_vec"]
        fn = Rf.vendored_functions(HERE / g["file"], g["blob"])["get_hidden_p_and_r"]
        model, tok = load(snapshot(plan, "qwen"), dtype="float32")
        return {"model": G.model_facts(model),
                "rlab32": Rf.run_rlab32(model, tok, plan, out, G.keep_rows(out), fn, commit, log, guard)}

    return stage(plan, "rlab32", "lab", body)


@app.function(image=repo_image, gpu=R_GPUS, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs},
              timeout=2 * 3600)
def verbatim_stage(plan: dict, judged: bool = False, ceiling: dict | None = None) -> dict:
    """R (reference.run_verbatim): generate_vec.py verbatim in the second image, once per trait, on the CSVs (or the
    judged CSVs), with its peak GPU memory."""
    commit = Ceiling(ceiling)
    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        import pandas as pd
        from transformers import AutoTokenizer

        out = out_dir(plan)
        tok = AutoTokenizer.from_pretrained(snapshot(plan, "qwen"))
        Rf.vendored_source(HERE / plan["repo"]["generate_vec"]["file"], plan["repo"]["generate_vec"]["blob"])
        return Rf.run_verbatim(plan, out, HERE / plan["repo"]["generate_vec"]["file"], snapshot(plan, "qwen"), tok,
                               pd, judged, commit=commit, log=log, guard=guard)

    return stage(plan, "verbatim_judged" if judged else "verbatim", "repo", body)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3600)
def probe_stage(plan: dict, part: str = "all", ceiling: dict | None = None) -> dict:
    """The probe.extract arms on Qwen in bf16 (readings.run_arms), through the recorder when it is on: part "graded"
    (stage probe_graded) the arms C4 needs, run before the report-only stages; "all" (stage probe) the rest."""
    commit = Ceiling(ceiling)
    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        from anatta.models import load

        model, tok = load(snapshot(plan, "qwen"))
        G.check_generation_config(model, plan["models"]["qwen"]["generation_config"])
        return {"model": G.model_facts(model),
                "probe": Rd.run_arms(model, tok, plan, out_dir(plan), T.load_traits(plan), commit, log, guard,
                                     part)}

    return stage(plan, "probe_graded" if part == "graded" else "probe", "lab", body)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3600)
def mprime_stage(plan: dict, ceiling: dict | None = None) -> dict:
    """M' on Ministral in bf16 (readings.run_mprime); the stored 28 Oct directions from the image when it carries
    them."""
    commit = Ceiling(ceiling)
    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        from anatta.models import load

        model, tok = load(snapshot(plan, "ministral"))
        stored_dir = REMOTE_DIRECTIONS if Path(REMOTE_DIRECTIONS).is_dir() else None
        # transformers 5.21 warns that this tokenizer's regex is wrong and sets fix_mistral_regex False when it is
        # loaded without the flag (as the library loads it, as on 28 Oct): recorded for the ledger, never changed
        flag = getattr(tok, "fix_mistral_regex", None)
        return {"model": {**G.model_facts(model), "fix_mistral_regex": flag if isinstance(flag, bool) else None},
                "mprime": Rd.run_mprime(model, tok, plan, out_dir(plan), lambda n: ego_trait(plan, n), stored_dir,
                                        commit, log, guard)}

    return stage(plan, "mprime", "lab", body)


@app.function(volumes={RUNS: runs}, cpu=8.0, memory=32768, timeout=3600)
def analysis_stage(plan: dict, aplan: dict, judged: bool = False) -> dict:
    """Every number (analysis.run), on the CPU next to the vectors, under the analysis plan (its own id: the result goes
    to stages/analysis/<id>.json, so a changed analysis reruns here from the stored data)."""
    runs.reload()
    stored_dir = REMOTE_DIRECTIONS if Path(REMOTE_DIRECTIONS).is_dir() else None
    return stage(plan, "analysis_judged" if judged else "analysis", "lab",
                 lambda guard: A.run(plan, aplan, out_dir(plan), plan_id(aplan), log, stored_dir, guard), aplan)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=1800)
def cheap_stage(plan: dict, analysis_id: str) -> dict:
    """The cheap tier's item (readings.run_cheap): the stored subset read again with probe.response_activations,
    against the reference values the analysis with this id wrote. It checks the data code only (a later fix to the
    analysis does not stop it) and records both ids."""
    offline()
    cache.reload()
    runs.reload()

    def body(guard):
        from anatta.models import load

        model, tok = load(snapshot(plan, "qwen"))
        return {"model": G.model_facts(model), "analysis_id": analysis_id,
                "cheap": Rd.run_cheap(model, tok, plan, out_dir(plan), T.load_traits(plan), analysis_id, guard)}

    return stage(plan, f"cheap_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}", "lab", body)


@app.function(volumes={RUNS: runs}, cpu=2.0, memory=4096, timeout=3 * 3600, secrets=SECRETS)
def judge_stage(plan: dict) -> dict:
    """The judged arm's scores (reference.run_judge): coherence_0_100 from the repo's eval/prompts.py as prepare put it
    on the volume (its sha256 and form the registered ones; nothing is fetched here), the key from the Modal Secret,
    never printed."""
    runs.reload()

    def body(guard):
        coherence = Rf.read_prompts(plan)
        return Rf.run_judge(plan, out_dir(plan), T.load_traits(plan), coherence, os.environ["OPENROUTER_API_KEY"],
                            log=log, guard=guard)

    return stage(plan, "judge", "lab", body)


STAGES = ("G", "full_batch", "keep", "verbatim", "readings", "rlab32", "probe_graded", "Gprime", "probe", "mprime",
          "judge", "csv_judged", "verbatim_judged")
ANALYSIS_STAGES = ("analysis", "analysis_judged")


@app.function(volumes={RUNS: runs}, cpu=1.0, memory=2048, timeout=600)
def stored(plan: dict, analysis_id: str) -> dict:
    """The stages already on the volume under this plan, the analyses under this analysis id: {stage: its result, or
    None}."""
    runs.reload()
    d = Path(out_dir(plan))
    T.claim(d, plan)
    runs.commit()
    return {**{s: T.read_json(stage_file(plan, s)) for s in STAGES},
            **{s: T.read_json(stage_file(plan, s, analysis_id)) for s in ANALYSIS_STAGES}}


# ------------------------------------------------------------------------------------------------- the laptop


def load_card():
    spec = importlib.util.spec_from_file_location("persona_extractor_card", HERE / "card.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def data_dir(cfg: dict) -> str:
    """Where prepare puts the trait files on the volume, by commit."""
    return f"{run_root(cfg['models']['qwen']['id'])}/persona_extractor/trait_files/{cfg['repo']['commit']}"


def stored_pins(cfg: dict) -> dict[str, str]:
    """The stored 28 Oct directions a full run needs, {path under DIRECTIONS: sha256}, as config.yaml pins them: M''s
    six ego traits and v_ego on Ministral, and the ledger's Qwen direction."""
    ms, lq = cfg["mprime"]["stored"], cfg["mprime"]["ledger_qwen"]
    pins = {f"{ms['model_slug']}/{n}.pt": sha for n, sha in ms["sha256"].items()}
    pins[f"{lq['model_slug']}/{lq['trait']}.pt"] = lq["sha256"]
    return pins


def stored_hashes(pins: dict) -> dict:
    """sha256 of each pinned file in the laptop's DIRECTIONS folder (what the image carries): {path: sha256, or None
    where the file is missing}."""
    return {rel: (T.sha256_file(DIRECTIONS / rel) if (DIRECTIONS / rel).is_file() else None) for rel in pins}


def check_directions(cfg: dict, smoke: bool) -> dict:
    """The guard on the stored directions: a full run refuses to start unless DIRECTIONS is a folder holding the eight
    files config.yaml pins, with exactly their sha256s; the smoke warns and goes on with whatever is there. Returns
    the hashes found ({path: sha256 or None})."""
    pins = stored_pins(cfg)
    got = stored_hashes(pins) if DIRECTIONS.is_dir() else {rel: None for rel in pins}
    bad = [rel for rel in pins if got[rel] != pins[rel]]
    if bad:
        msg = (f"the stored directions in {DIRECTIONS.resolve()} are missing or differ from config.yaml's pins "
               f"({len(bad)} of {len(pins)}: {', '.join(bad)}); set ANATTA_DIRECTIONS to the folder holding them")
        if not smoke:
            raise SystemExit(f"{msg}. A full run needs all eight.")
        print(f"warning: {msg}. The smoke goes on; M''s stored comparisons won't run.", flush=True)
    return got


def make_plan(cfg: dict, smoke: bool = False) -> dict:
    """Everything the GPU and CSV stages' files depend on, as plain JSON (the analysis's own settings, config.yaml's
    analysis and registered blocks, are in card.analysis_plan). Refuses library files other than the registered ones,
    a vendored file with another blob sha, trait files whose sha256 and counts are not yet registered (run prepare and
    commit first), with the judged switch on an eval/prompts.py without its registered sha256 and form, and, for a
    full run, stored directions other than the pinned ones (check_directions)."""
    for f, sha in cfg["instrument"]["sha256"].items():
        if T.sha256_file(PKG / f) != sha:
            raise SystemExit(f"src/anatta/{f} is not the registered instrument (sha256 {sha[:16]}...)")
    g = cfg["repo"]["generate_vec"]
    data = (HERE / g["file"]).read_bytes()
    if T.git_blob_sha(data) != g["blob"] or len(data) != g["bytes"]:
        raise SystemExit(f"{g['file']} is not generate_vec.py at {cfg['repo']['commit'][:7]} (blob {g['blob'][:8]})")
    for t in cfg["traits"]:
        e = cfg["repo"]["traits"][t]
        if not e.get("sha256") or e.get("pairs") != cfg["repo"]["expect"]["pairs"] \
                or e.get("questions") != cfg["repo"]["expect"]["questions"]:
            raise SystemExit(f"{t}.json has no registered sha256 and counts in config.yaml: run the prepare entrypoint "
                             "and commit config.yaml first")
    sw = dict(cfg["switches"])
    pr = cfg["judged"]["prompts"]
    if sw["judged"] and (not pr.get("sha256") or pr.get("form") not in Rf.FORMS):
        raise SystemExit("eval/prompts.py has no registered sha256 and form in config.yaml: run the prepare entrypoint "
                         "and commit config.yaml before turning the judged arm on")
    layout = {"questions": cfg["repo"]["expect"]["questions"], "instructions": cfg["repo"]["expect"]["pairs"]}
    gen, gprime = json.loads(json.dumps(cfg["G"])), json.loads(json.dumps(cfg["Gprime"]))
    arms, probe = dict(cfg["arms"]), {"traits": list(cfg["traits"]), "questions": None}
    mprime = json.loads(json.dumps(cfg["mprime"]))
    reference = dict(cfg["reference"])
    full_batch = None
    if not sw["second_image"]:
        reference.update(verbatim=False, why_not=reference.get("why_not") or "declined")
    if smoke:
        # Ruled (NOTES, 12:20 UTC, ruling 7): the smoke runs every stage at smoke size, G' included; read as nothing.
        s = cfg["smoke"]
        layout = {"questions": s["questions"], "instructions": s["instructions"]}
        for spec in (gen, gprime):
            spec["samples"] = s["samples"]
            spec["kwargs"]["max_new_tokens"] = s["max_new_tokens"]
        sys_arm = next(a for a, sp in arms.items() if sp["include_system"] and sp["seed"] == 0)
        arms = {sys_arm: arms[sys_arm]}
        probe = {"traits": [s["probe_trait"]], "questions": s["probe_questions"]}
        no_sys = next(c for c, inc in mprime["conventions"].items() if not inc)
        mprime.update(traits=[s["mprime_trait"]], conventions={no_sys: False}, questions=s["probe_questions"])
        if s.get("full_batch"):  # G's first batch at its registered size and settings, read as nothing
            full_batch = {"trait": cfg["traits"][0], "size": cfg["G"]["batch"], "seed": cfg["G"]["seed_base"],
                          "layout": {**{k: cfg["repo"]["expect"][v] for k, v in (("questions", "questions"),
                                                                                  ("instructions", "pairs"))},
                                     "samples": cfg["G"]["samples"]},
                          "kwargs": dict(cfg["G"]["kwargs"]), "forbidden": list(cfg["G"]["forbidden"]),
                          "required": dict(cfg["G"]["required"])}
    pg = cfg["probe_graded"]
    if sorted(pg["traits"]) != sorted(cfg["registered"]["graded"]["c4"]["traits"]) or "E-sys" not in pg["arms"]:
        raise SystemExit("config.yaml's probe_graded must name C4's traits and its arm, E-sys")
    probe["graded"] = {"traits": [t for t in pg["traits"] if t in probe["traits"]],
                       "arms": [a for a in pg["arms"] if a in arms]}
    got = check_directions(cfg, smoke)
    lq = cfg["mprime"]["ledger_qwen"]
    qwen_rel = f"{lq['model_slug']}/{lq['trait']}.pt"
    plan = {"check": cfg["check"], "smoke": smoke, "switches": sw, "instrument": cfg["instrument"],
            "repo": cfg["repo"], "reference": reference, "models": cfg["models"], "prompts": cfg["prompts"],
            "traits": list(cfg["traits"]), "layout": layout, "eos": cfg["models"]["qwen"]["eos"], "G": gen,
            "Gprime": gprime, "full_batch": full_batch, "readings": cfg["readings"], "arms": arms, "probe": probe,
            "mprime": mprime, "cheap_tier": cfg["cheap_tier"], "judged": cfg["judged"], "smoke_settings": cfg["smoke"],
            "data_dir": data_dir(cfg),
            "stored_sha256": {r: h for r, h in got.items() if r != qwen_rel and h is not None} or None,
            "stored_sha256_qwen": {qwen_rel: got[qwen_rel]} if got[qwen_rel] else None,
            "code": T.code_hashes(HERE, PKG), "pins": {"lab": LAB_PINS, "repo": REPO_PINS},
            "container_sha256": container_sha256()}
    return json.loads(json.dumps(plan))


def save(d: Path, name: str, obj) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1) + "\n")


def other_plan(d: Path, plan: dict) -> bool:
    p = d / "plan.json"
    return p.exists() and json.loads(p.read_text()) != plan


def keep_old_analyses(d: Path, analysis_id: str) -> None:
    """An analysis saved under another analysis id is kept beside the new one, as analysis-<id>.json."""
    for name in ANALYSIS_STAGES:
        p = d / f"{name}.json"
        old = json.loads(p.read_text()).get("analysis_id") if p.exists() else None
        if old is not None and old != analysis_id:
            p.rename(d / f"{name}-{old}.json")


class Stop(Exception):
    """The laptop's stop: the estimated spend has reached the limit a stage checks against."""


class Spend:
    """The run's estimated spend, kept in spend.json: each call's wall time on the laptop at config.yaml's rates (the
    GPU's and the CPU and memory tier's), an overestimate, since it counts queueing and start-up. The smoke's and the
    full run's spend count toward one stop: `also` is the other run's spend.json, whose entries are added to the total.
    The GPU and CSV stages check against the stop less the reserve held for the analysis and the card; the analysis
    against the stop. The checks are the hard ceiling's: the spend so far plus the next unit's upper estimate
    (config.yaml's cost.upper_s) must be within the limit."""

    def __init__(self, d: Path, cfg: dict, also: Path | None = None):
        self.path, self.cost, self.also = d / "spend.json", cfg["cost"], also
        self.entries = json.loads(self.path.read_text())["entries"] if self.path.exists() else []
        other = json.loads(also.read_text())["entries"] if also is not None and also.exists() else []
        self.earlier = round(sum(e["usd"] for e in other), 4)

    def total(self) -> float:
        """This run's spend and the other run's, together."""
        return round(sum(e["usd"] for e in self.entries) + self.earlier, 4)

    def counted(self) -> str:
        return f", counting ${self.earlier:.2f} in {self.also}" if self.earlier else ""

    def limit(self, analysis: bool = False) -> float:
        return self.cost["stop_usd"] - (0.0 if analysis else self.cost["analysis_reserve_usd"])

    def rate(self, gpu: str | None, tier: str) -> float:
        return (self.cost["gpu_per_hour"][gpu] if gpu else 0.0) + self.cost["cpu_mem_per_hour"][tier]

    def upper_s(self, kind: str, gpu: str | None, measured: dict | None = None) -> float:
        """A unit's upper estimate in seconds (config.yaml's cost.upper_s), at a GPU, or for a CPU stage; the smoke's
        measured seconds where higher (its full batch of G; R's measure needs the trait's rows and tokens, so only its
        container's Ceiling applies it)."""
        u = self.cost["upper_s"][kind]
        s = float(u.get(gpu, max(u.values())) if isinstance(u, dict) else u)
        m = (measured or {}).get(kind) or {}
        return max(s, float(m["seconds"])) if m.get("seconds") is not None else s

    def check(self, name: str, seconds: float, gpu: str | None, tier: str, analysis: bool = False) -> None:
        """The hard ceiling before a container: Stop unless the spend so far plus `seconds` (its start-up and first
        unit at the GPU asked for, or a CPU stage's whole) at these rates is within the limit."""
        usd = seconds / 3600 * self.rate(gpu, tier)
        if self.total() + usd > self.limit(analysis):
            which = "stop" if analysis else (f"limit for the GPU and CSV stages (the ${self.cost['stop_usd']} stop "
                                             f"less ${self.cost['analysis_reserve_usd']:.2f} held for the analysis)")
            raise Stop(f"stopped before {name}: the estimated spend ${self.total():.2f} plus its first unit's upper "
                       f"estimate ${usd:.2f} would pass the ${self.limit(analysis):.2f} {which} "
                       f"(spend.json{self.counted()})")

    def ceiling(self, gpu: str | None, tier: str, analysis: bool, measured: dict) -> dict:
        """What a GPU container's Ceiling reads: the spend and the limit at the call, the call's start, the rates, the
        upper estimates and the smoke's measures."""
        return {"spent_usd": self.total(), "limit_usd": self.limit(analysis), "started_at": time.time(), "gpu": gpu,
                "gpu_usd_h": dict(self.cost["gpu_per_hour"]), "tier_usd_h": self.cost["cpu_mem_per_hour"][tier],
                "upper_s": self.cost["upper_s"], "measured": measured}

    def add(self, name: str, seconds: float, gpu: str | None, tier: str, usd_extra: float = 0.0) -> None:
        usd = round(seconds / 3600 * self.rate(gpu, tier) + usd_extra, 4)
        self.entries.append({"stage": name, "seconds": round(seconds), "gpu": gpu, "tier": tier, "usd": usd})
        self.path.write_text(json.dumps({"entries": self.entries, "total_usd": self.total(),
                                         "counted_from_other_run_usd": self.earlier}, indent=1) + "\n")
        print(f"spend: {name} about ${usd:.2f} ({seconds / 60:.1f} min{' on ' + gpu if gpu else ''}); total about "
              f"${self.total():.2f} of the ${self.cost['stop_usd']} stop{self.counted()}", flush=True)


def gpu_key(name: str | None, default: str) -> str:
    """The rate key for the GPU a container reported (the verbatim stage may land on its fallback)."""
    if name and "H100" in name:
        return "H100"
    if name and "A100" in name:
        return "A100-80GB"
    return default


def call(spend: Spend, name: str, fn, args: tuple, gpu: str | None, tier: str, first: str, budget: bool = True,
         analysis: bool = False, measured: dict | None = None) -> dict:
    """One stage from the laptop: the hard ceiling first (Stop unless the spend so far plus the container's start-up and
    its `first` unit's upper estimate, at the GPU asked for first, fits; a CPU stage's `first` is its whole), the call
    (GPU functions with expandable segments and, with `budget`, the container's own Ceiling), its spend recorded at the
    rate of the GPU the container reported and printed, even if it fails."""
    spend.check(name, spend.upper_s(first, gpu, measured) + (spend.upper_s("startup", None) if gpu else 0.0), gpu,
                tier, analysis)
    f = fn.with_options(env=GPU_ENV) if gpu else fn
    kw = {"ceiling": spend.ceiling(gpu, tier, analysis, measured or {})} if budget else {}
    t0, res = time.time(), None
    try:
        res = f.remote(*args, **kw)
        return res
    finally:
        got = (res or {}).get("env", {}).get("gpu") if isinstance(res, dict) else None
        spend.add(name, time.time() - t0, gpu_key(got, gpu) if gpu else None, tier)


def budget_stop(e: Exception) -> bool:
    """A container's own stop past the limit after a unit (a StageError carrying CheckError budget)."""
    return isinstance(e, T.StageError) and "CheckError budget" in str(e)


def ceiling_stop(e: Exception) -> bool:
    """A container's refusal to start its next unit (a StageError carrying CheckError ceiling)."""
    return isinstance(e, T.StageError) and "CheckError ceiling" in str(e)


def run_steps(spend: Spend, d: Path, have: dict, steps: list, measured: dict | None = None) -> str | None:
    """The GPU and CSV stages in order, each read from the volume if it is there, saved to d. Returns why the run
    stopped (the laptop's refusal of a container, a container's refusal of its next unit, or its stop past the limit),
    or None; any other failure is raised."""
    for name, fn, args, gpu, tier, first, budget in steps:
        try:
            res = have.get(name) or call(spend, name, fn, args, gpu, tier, first, budget, measured=measured)
        except Stop as e:
            return str(e)
        except T.StageError as e:
            if ceiling_stop(e):
                return (f"{name} didn't start its next unit: the spend so far plus that unit's upper estimate would "
                        f"pass the ${spend.limit():.2f} limit for the GPU and CSV stages; its work so far is on the "
                        "volume")
            if budget_stop(e):
                return (f"{name} stopped itself at its first commit past the ${spend.limit():.2f} limit (a unit ran "
                        "longer than its upper estimate); its work so far is on the volume")
            raise
        save(d, f"{name}.json", res)
    return None


def smoke_measured(card, root: Path, smoke: bool) -> dict:
    """The smoke's measures that can raise the full run's upper estimates (Spend.upper_s, Ceiling.seconds), from its
    stage files in --results/smoke: the seconds of its full batch of G; for R, from tqdm's timing of each trait's neg
    pass (no load, warm-up or save in it), its seconds a token and a row, and the trait's seconds outside that pass
    (the most of any trait, with the GPU it ran on). Per token is what scales: the smoke's rows are 64-token answers
    and one long pair, shorter than the full run's, so its seconds a row would understate them. Empty for the smoke
    itself, before the smoke, or where the smoke couldn't time them (R's stderr without both closing bars)."""
    if smoke:
        return {}
    d = card.run_folder(root, True)
    out = {}
    fb = card.get(card.read(d / "full_batch.json"), "result", "full_batch", "seconds")
    if fb is not None:
        out["G_batch"] = {"seconds": float(fb)}
    vb = card.read(d / "verbatim.json")
    timed = [r for r in (card.get(vb, "result") or {}).values()
             if r.get("pass_seconds") and r.get("neg_tokens") and r.get("rows") and r.get("seconds") is not None]
    if timed:
        out["R_trait"] = {"fixed_s": max(max(r["seconds"] - r["pass_seconds"][1], 0.0) for r in timed),
                          "s_per_token": max(r["pass_seconds"][1] / r["neg_tokens"] for r in timed),
                          "s_per_row": max(r["pass_seconds"][1] / (r["rows"] // 2) for r in timed),
                          "gpu": gpu_key(card.get(vb, "env", "gpu"), R_GPUS[0])}
    return out


def run_analysis(spend: Spend, d: Path, have: dict, name: str, args: tuple, stop: str | None) -> str | None:
    """The analysis (or the judged one), checked against the full stop. After a stop it runs on what exists, and a
    failure there is noted rather than raised, so the card is still written."""
    try:
        res = have.get(name) or call(spend, name, analysis_stage, args, None, "cpu8_32g", "analysis", budget=False,
                                     analysis=True)
    except Stop as e:
        return f"{stop}; {e}" if stop else str(e)
    except T.StageError as e:
        if stop is None:
            raise
        return f"{stop}; the {name} couldn't run on what was there ({str(e).rsplit(': ', 1)[-1]})"
    save(d, f"{name}.json", res)
    return stop


def finish(card, cfg: dict, d: Path, stop: str | None = None, spend: Spend | None = None) -> dict:
    """The card from whatever ran; after a stop, stopped.json first (the card carries it) and then SystemExit. A
    relaunch that runs to the end removes an earlier launch's stopped.json."""
    if stop:
        save(d, "stopped.json", {"reason": stop, "spend_usd": spend.total() if spend else None})
    else:
        (d / "stopped.json").unlink(missing_ok=True)
    c = card.write_card(cfg, d)
    print("\n".join(card.headline(c)), flush=True)
    print(f"wrote {d / 'card.json'} and {d / 'ledger.json'}", flush=True)
    if stop:
        raise SystemExit(f"stopped: {stop}")
    return c


@app.local_entrypoint()
def prepare():
    """Fetch and check the trait files and the judged arm's eval/prompts.py on Modal, then write their sha256 and
    counts (and the prompts file's form) into config.yaml (to commit)."""
    cfg = load_yaml(HERE / "config.yaml")
    pr = cfg["judged"]["prompts"]
    facts = fetch_traits.remote(cfg["repo"], cfg["traits"], data_dir(cfg), pr)
    text = (HERE / "config.yaml").read_text()
    for t, f in facts["traits"].items():
        e = cfg["repo"]["traits"][t]
        if e.get("sha256") and (e["sha256"], e["pairs"], e["questions"]) != (f["sha256"], f["pairs"], f["questions"]):
            raise SystemExit(f"{t}.json: the fetched file's sha256 or counts differ from config.yaml's")
        pattern = re.compile(rf"^(    {t}: \{{bytes: {f['bytes']}, blob: {f['blob']}, )sha256: [^,]*, pairs: [^,]*, "
                             rf"questions: [^}}]*\}}", re.M)
        text, n = pattern.subn(rf'\g<1>sha256: "{f["sha256"]}", pairs: {f["pairs"]}, questions: {f["questions"]}}}',
                               text)
        if n != 1:
            raise SystemExit(f"config.yaml: no single line for {t} under repo.traits to fill")
        print(f"{t}: {f['bytes']} bytes, blob {f['blob']}, sha256 {f['sha256']}, {f['pairs']} pairs, "
              f"{f['questions']} questions", flush=True)
    f = facts["prompts"]
    if pr.get("sha256") and (pr["sha256"], pr["form"]) != (f["sha256"], f["form"]):
        raise SystemExit(f"{pr['file']}: the fetched file's sha256 or form differs from config.yaml's")
    pattern = re.compile(rf"^(  prompts: \{{file: {re.escape(pr['file'])}, bytes: {f['bytes']}, blob: {f['blob']}, )"
                         rf"sha256: [^,]*, form: [^}}]*\}}", re.M)
    text, n = pattern.subn(rf'\g<1>sha256: "{f["sha256"]}", form: {f["form"]}}}', text)
    if n != 1:
        raise SystemExit("config.yaml: no single line for judged.prompts to fill")
    print(f"{pr['file']}: {f['bytes']} bytes, blob {f['blob']}, sha256 {f['sha256']}, coherence_0_100 bound by the "
          f"{f['form']} form", flush=True)
    (HERE / "config.yaml").write_text(text)
    print("config.yaml updated: commit it before any run", flush=True)


@app.local_entrypoint()
def main(results: str, smoke: bool = False):
    card = load_card()
    cfg = card.load_config()
    root = card.results_path(results)
    d = card.run_folder(root, smoke)
    plan = make_plan(cfg, smoke)
    if other_plan(d, plan):
        raise SystemExit(f"{d} holds the run of another plan (its plan.json differs from this one): use a fresh "
                         "--results, or move that folder away")
    aplan = card.analysis_plan(cfg, plan, HERE)
    aid = plan_id(aplan)
    save(d, "plan.json", plan)
    keep_old_analyses(d, aid)
    save(d, "analysis_plan.json", aplan)
    (d / "stopped.json").unlink(missing_ok=True)
    sw, ref = plan["switches"], plan["reference"]
    print(f"C4 persona extractor, {'smoke' if smoke else 'full'} run, plan {plan_id(plan)}, analysis {aid}: "
          f"{out_dir(plan)}; recorder {'on' if sw['recorder'] else 'off'}, reference "
          f"{'R (verbatim)' if ref['verbatim'] else 'R-lab32'}", flush=True)
    spend = Spend(d, cfg, card.run_folder(root, not smoke) / "spend.json")  # the smoke's spend counts toward the stop
    measured = smoke_measured(card, root, smoke)
    if measured:
        print(f"upper estimates: the smoke's measures, used where they pass config.yaml's: {json.dumps(measured)}",
              flush=True)
    have = stored.remote(plan, aid)
    # (name, function, args, GPU asked for first, CPU and memory tier, the first unit's kind, the container's ceiling)
    steps = [("G", generate_stage, (plan, "G"), GPU, "cpu8_32g", "G_batch", True)]
    if plan["full_batch"]:
        steps.append(("full_batch", fullbatch_stage, (plan,), GPU, "cpu8_32g", "G_batch", True))
    steps.append(("keep", csv_stage, (plan,), None, "cpu4_16g", "csv", False) if sw["second_image"] else
                 ("keep", keep_stage, (plan,), None, "cpu2_8g", "keep", False))
    if ref["verbatim"]:
        steps.append(("verbatim", verbatim_stage, (plan,), R_GPUS[0], "cpu8_32g", "R_trait", True))
    steps += [("readings", readings_stage, (plan,), GPU, "cpu8_32g", "G_chunk", True),
              ("rlab32", rlab32_stage, (plan,), GPU, "cpu8_64g", "rlab32_trait", True),
              ("probe_graded", probe_stage, (plan, "graded"), GPU, "cpu8_32g", "probe_call", True),
              # only reported or only in the ledger from here on: a stop after this point leaves C4 gradable
              ("Gprime", generate_stage, (plan, "Gprime"), GPU, "cpu8_32g", "Gprime_batch", True),
              ("probe", probe_stage, (plan, "all"), GPU, "cpu8_32g", "probe_call", True),
              ("mprime", mprime_stage, (plan,), GPU, "cpu8_32g", "mprime_call", True)]
    stop = run_steps(spend, d, have, steps, measured)
    stop = run_analysis(spend, d, have, "analysis", (plan, aplan), stop)
    if stop is None and sw["judged"] and card.c4_base_missed(cfg, d):
        print("C4-base missed and the judged arm is approved: scoring G through OpenRouter", flush=True)
        if not have.get("judge"):
            jr = call(spend, "judge", judge_stage, (plan,), None, "cpu2_4g", "judge", False)
            save(d, "judge.json", jr)
            used = jr["result"]["tokens"]
            spend.add("judge_api", 0, None, "cpu2_4g", used["in"] / 1e6 * cfg["cost"]["judge_per_million"]["input"]
                      + used["out"] / 1e6 * cfg["cost"]["judge_per_million"]["output"])
            print(f"judged with {plan['judged']['model']}", flush=True)
        else:
            save(d, "judge.json", have["judge"])
        stop = run_steps(spend, d, have, [
            ("csv_judged", csv_stage, (plan, True), None, "cpu4_16g", "csv", False),
            ("verbatim_judged", verbatim_stage, (plan, True), R_GPUS[0], "cpu8_32g", "R_trait", True)], measured)
        stop = run_analysis(spend, d, have, "analysis_judged", (plan, aplan, True), stop)
    return finish(card, cfg, d, stop, spend)


@app.local_entrypoint()
def card(results: str, smoke: bool = False):
    mod = load_card()
    cfg = mod.load_config()
    d = mod.run_folder(mod.results_path(results), smoke)
    c = mod.write_card(cfg, d)
    print("\n".join(mod.headline(c)), flush=True)


@app.local_entrypoint()
def cheap(results: str):
    """The cheap tier's item on the full run's stored responses (a few GPU-minutes), against the reference values the
    run's current analysis wrote; printed and saved with both ids."""
    mod = load_card()
    root = mod.results_path(results)
    d = mod.run_folder(root, False)
    plan = json.loads((d / "plan.json").read_text())
    aid = json.loads((d / "analysis.json").read_text())["analysis_id"]
    spend = Spend(d, mod.load_config(), mod.run_folder(root, True) / "spend.json")
    res = call(spend, "cheap", cheap_stage, (plan, aid), GPU, "cpu8_32g", "cheap", budget=False)
    save(d, f"{res['stage']}.json", res)
    print(f"cheap tier on plan {res['plan_id']}, analysis {res['result']['analysis_id']}", flush=True)
    for t, r in res["result"]["cheap"]["traits"].items():
        print(f"cheap tier {t}: min cos with P1 {r['p1_min']:.6f}, with R-lab32 at the graded block "
              f"{r['rlab32_block']:.6f}: {'pass' if r['passed'] else 'FAIL'}", flush=True)
