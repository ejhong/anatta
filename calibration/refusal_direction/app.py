"""Calibration check 3, steering part 1, on Modal: do the lab's refusal extractor (anatta.contrast), Arditi's selection
(anatta.select) and the dial reproduce Wollschläger et al.'s Table 5 row for Qwen2.5-7B-Instruct (JailbreakBench ASR
9.1 -> 69.2 under directional ablation, MMLU 68.8 -> 68.1), scored by StrongREJECT's fine-tuned judge? README.md has
the claim, the sources and the numbers; config.yaml every setting; refusal.py the code that runs in the containers;
reference.py Arditi's pipeline, ported; judge.py the judge, ported; card.py the card.

One command runs it all, each stage a Modal function writing under out_dir(plan) on the anatta-runs volume:
  prepare      CPU, the lab's image, with the "huggingface" Modal secret (Gemma is gated). Fetches the data into the
               run's folder and the judge into the anatta-hf-cache volume, and checks them (refusal.prepare). A data
               file whose sha256 or count is not yet in config.yaml stops the run here: prepare.json holds the numbers
               to register, and nothing on a GPU starts until they are committed. Last, it loads offline every
               tokenizer and config the lab's later stages load, and stops if any does not load.
  prepare_paper  CPU, the paper's image, no secret: the same offline loads for judge_paper and swap
               (refusal.prepare_paper). Nothing on a GPU starts unless both stacks' loads all load.
  run_model    one L40S per model, Qwen and Ministral at once: the MMLU golden test under its gate, filtering, the
               library path, on Qwen the reference path and C3-extract, then every condition, with the MMLU floor
               after the first on a full run (refusal.run_model). A split that kept no prompts stops that model's
               run at filtering, and a full run's MMLU floor stops it after its first condition; if it is
               Ministral's, Qwen's run is still judged and graded. Any other stop ends the run, as does either of
               these on Qwen (the judge port gate runs on Qwen's Q0 and QA).
  judge_lab    an L40S, the lab's image: the port on every condition's completions, and the judge control.
  judge_paper  an L40S, the paper's image: the original strong_reject package and the port on Q0's and QA's
               completions (the judge port gate); if the stacks disagree, the original package on every condition.
  swap         an L40S, the paper's image, only on a miss: Q0 regenerated and scored by the original judge if C3-base
               misses; the reference path re-run if C3 and C3-R both miss with viable candidates.
Then the card (card.py), here.

The images. The lab's: anatta.modal_app's pins (read from that file, so the two stay the same) plus pyarrow to read
MMLU's parquet, built afresh as check 7's is (modal_app's image ends with the package added as local files, and Modal
refuses a build step after that). The paper's: Python 3.12 with Wollschläger's pins (transformers 4.47.0, torch 2.5.1,
accelerate 1.2.0) and the other packages strong_reject imports pinned by hand at releases from the same months, then
strong_reject at e286f0d from GitHub with --no-deps (its pyproject pins nothing, so a plain install would replace the
stack), and READTHEDOCS=1, so evaluate.py skips its generate module and litellm and the Google clients are not needed.
Its build steps check that the package's template file is installed, that strong_reject.evaluate imports, and that the
image fills neither volume's mount path. They run with HF_HOME in /tmp and the image's own environment (HF_HOME on the
cache volume) is set after them: importing the stack writes into HF_HOME, and Modal will not mount a volume over a path
the image fills (the g3 smoke's prepare_paper, 1 Nov: "cannot mount volume on non-empty path"). The
lab's image also carries one file of the lab's trait directions: the registered v_ego (config.yaml's ego block), at
the path anatta.modal_app.ego_file reads, for the ledger's E-82 and E-82r. The laptop checks its sha256 before any
spend and warns if it is missing or differs; the container reads it only if its sha256 is the registered one.

Copied from the other checks, not merged: check 1's volume loader (anatta-calib-c1 calibration/model_check/app.py:
171-185 at 44760f7: HF_HUB_OFFLINE set in the container, the weights read at their pinned revisions) and its
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True at launch (app.py:326 at 44760f7); check 1's fingerprint.py (at
8d93ad7, here as fingerprint.py); check 7's plan, out_dir, container_sha256, stored and finish (anatta-calib-c7
calibration/relearning/app.py at 02db868).

Spend, an estimate (rulings, 1 Nov). Each Modal call's wall time, measured here (a model run's as the larger of that
and its container's own report, so the container's start and the image pull count), is costed at config.yaml's rates
into --results/spend.json: an L40S container at the registration's $3.9 an hour, a CPU container at an estimated
$0.35, and the higher of the two where both apply (a GPU container also runs CPU cores). It is not Modal's bill, and
it is labelled an estimate wherever it is printed and on the card. The running total is printed after every call. No
call starts if the estimated total would pass the registered $30 stop at its timeout. The two model runs, which
run at once, have their timeouts cut by one factor to what is left (Spend.budgets); a launch whose cut would leave a
model run under half an hour is refused. Every later call is refused, not cut, if its own timeout would pass the stop.
No function retries: a rerun of the same plan resumes from the stages and conditions already on the volume.

Settings the code fixes in its own text (the filter's threshold and cut, the layer pruning, the selection's
thresholds, MMLU's description, the judge's score labels, the stripping, the smoke's candidates) are checked against
config.yaml's at plan time, before any spend (check_settings): the code stays as ported, and the card's settings are
the ones that ran.

The data, the completions and every direction stay on the volume. The containers return numbers, counts and sha256s.

A run is its plan: every setting and the registered block, the sha256 of the package files the run calls and of this
check's container code, and both images' pins. Each container first checks its code and installed versions against
the plan. A changed plan (an edited file, a new pin, --smoke) runs afresh in a new folder.

Usage, from the repo root (the modal CLI; PYTHONPATH=src so the app imports this package):
  PYTHONPATH=src modal run calibration/refusal_direction/app.py::main --results DIR --prepare-only
  PYTHONPATH=src modal run calibration/refusal_direction/app.py::main --results DIR --smoke
  PYTHONPATH=src modal run calibration/refusal_direction/app.py::main --results DIR
  PYTHONPATH=src modal run calibration/refusal_direction/app.py::card --results DIR [--smoke]
DIR must be outside any git work tree (card.results_path). Keep the terminal open while main runs; do not --detach.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
import refusal

from anatta.modal_app import CACHE, GPU, PKG, ROOT, RUNS, cache, ego_file, runs

HERE = Path(__file__).resolve().parent


def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fingerprint = module_from_file("c3_fingerprint", HERE / "fingerprint.py")
image_pins, plan_id = fingerprint.image_pins, fingerprint.plan_id
LAB_EXTRA = {"pyarrow": "21.0.0"}  # as approved for check 7's images
LAB_PINS = {**image_pins(PKG / "modal_app.py"), **LAB_EXTRA}
# The paper's stack: Wollschläger's requirements.txt (transformers, torch, accelerate) and, by hand, the packages
# strong_reject's evaluate.py imports, at releases from the same months (recorded with pip freeze at every run).
PAPER_PINS = {"transformers": "4.47.0", "torch": "2.5.1", "accelerate": "1.2.0", "peft": "0.14.0",
              "datasets": "3.2.0", "numpy": "2.2.1", "pandas": "2.2.3", "openai": "1.58.1", "tokenizers": "0.21.0",
              "huggingface-hub": "0.27.0", "safetensors": "0.4.5", "jinja2": "3.1.5", "pyyaml": "6.0.2"}
STRONG_REJECT = ("strong_reject @ git+https://github.com/dsbowen/strong_reject@"
                 "e286f0da86d92c929a6fda20a9992f28c5969044")
ENV = {"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root"}
ALLOC = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # check 1's launch setting (app.py:326 at 44760f7)
# this check's files the containers import: the stages, and fingerprint.py, which this file loads at import (out_dir
# calls its plan_id in every container); Modal mounts only the entrypoint file itself
CHECK_CODE = ("refusal.py", "reference.py", "judge.py", "fingerprint.py")
TEMPLATE_INSTALLED = ("python -c \"import importlib.util, pathlib; s = importlib.util.find_spec('strong_reject'); "
                      "p = pathlib.Path(list(s.submodule_search_locations)[0]) / 'eval_files' / 'judge_templates.json'; "
                      "assert p.is_file(), p\"")
# The paper image's build steps run with these, not the image's ENV (module docstring): HF_HOME outside every mount path.
PAPER_BUILD_ENV = {"HF_HOME": "/tmp/c3-build-hf", "HF_HUB_OFFLINE": "1", "READTHEDOCS": "1"}
MOUNTS_ABSENT = (f"python -c \"import os; p = [d for d in {(CACHE, RUNS)!r} if os.path.lexists(d)]; "
                 "assert not p, ('the image fills a volume mount path', p)\"")


def pip(pins: dict) -> list[str]:
    return [f"{name}=={version}" for name, version in pins.items()]


def with_code(image):
    image = image.add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"])
    for f in CHECK_CODE:
        image = image.add_local_file(str(HERE / f), f"/root/{f}")
    return image


def registered_ego(cfg_path: Path = HERE / "config.yaml") -> tuple[Path, str] | None:
    """config.yaml's registered v_ego: (its file on the laptop, where the image puts it: ego_file of the model that
    reads it), or None where config.yaml is not beside this file (in a container, which reads the image's copy)."""
    if not cfg_path.exists():
        return None
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    return Path(cfg["ego"]["file"]), str(ego_file(cfg["models"][cfg["ego"]["model"]]["id"]))


EGO = registered_ego()
lab_image = with_code(modal.Image.debian_slim(python_version="3.12").uv_pip_install(*pip(LAB_PINS)).env(ENV))
if EGO is not None and EGO[0].is_file():  # the registered v_ego alone, not the folder (ruling, 1 Nov)
    lab_image = lab_image.add_local_file(str(EGO[0]), EGO[1])
paper_image = with_code(modal.Image.debian_slim(python_version="3.12")
                        .apt_install("git")
                        .uv_pip_install(*pip(PAPER_PINS))
                        .pip_install(STRONG_REJECT, extra_options="--no-deps")
                        .env(PAPER_BUILD_ENV)
                        .run_commands(TEMPLATE_INSTALLED, "python -c 'import strong_reject.evaluate'", MOUNTS_ABSENT)
                        .env({**ENV, "HF_HUB_OFFLINE": "1", "READTHEDOCS": "1"}))
app = modal.App("anatta-calib-refusal-direction", image=lab_image)
hf_secret = modal.Secret.from_name("huggingface")


def out_dir(plan: dict) -> str:
    """A run's folder on the anatta-runs volume."""
    return f"{ROOT}/refusal_direction/{'smoke' if plan['smoke'] else 'full'}/{plan_id(plan)}"


def offline() -> None:
    """The weights are on the volume at their pinned revisions: no download (check 1's loader). Set before
    transformers loads."""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"


def ego_path(plan: dict, key: str) -> Path | None:
    """Where the image carries the registered v_ego, for the model that reads it (plan["ego"]["model"]), else None.
    refusal.ego_reading checks the file's sha256 before it loads it."""
    if key != plan["ego"]["model"]:
        return None
    return ego_file(plan["models"][key]["id"])


CONTAINER_FUNCTIONS = ("out_dir", "offline", "ego_path", "prepare", "prepare_paper", "run_model", "judge_lab",
                       "judge_paper", "swap", "stored")  # this file's code that runs in a run's containers


def container_source(source: str | None = None) -> str:
    """The source of CONTAINER_FUNCTIONS (each with its decorator), read from this file's text (or `source`)."""
    src = Path(__file__).read_text() if source is None else source
    lines = src.splitlines(keepends=True)
    defs = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    return "".join("".join(lines[min([defs[f].lineno, *(d.lineno for d in defs[f].decorator_list)]) - 1:
                                 defs[f].end_lineno]) for f in CONTAINER_FUNCTIONS)


@functools.cache
def container_sha256(source: str | None = None) -> str:
    """The plan's hash of this file: only its code that runs in the containers."""
    return refusal.sha256_text(container_source(source))


# ------------------------------------------------------------------------------------------------ containers


@app.function(volumes={CACHE: cache, RUNS: runs}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=2 * 3600)
def prepare(plan: dict) -> dict:
    """Fetch, count, hash and check (refusal.prepare)."""
    cache.reload()
    runs.reload()
    try:
        return refusal.prepare(plan, out_dir(plan), PKG, log=lambda *a: print(*a, flush=True))
    finally:
        cache.commit()
        runs.commit()


@app.function(image=paper_image, volumes={CACHE: cache, RUNS: runs}, cpu=2.0, memory=8192, timeout=900)
def prepare_paper(plan: dict) -> dict:
    """The paper's stack loads, offline (its image sets HF_HUB_OFFLINE), every tokenizer and config judge_paper and
    swap load (refusal.prepare_paper). It reads the cache volume and downloads nothing to it."""
    cache.reload()
    runs.reload()
    try:
        return refusal.prepare_paper(plan, out_dir(plan), PKG, log=lambda *a: print(*a, flush=True))
    finally:
        runs.commit()


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3 * 3600)
def run_model(plan: dict, key: str, budget_seconds: float) -> dict:
    """One model's run (refusal.run_model), the model loaded by modal_app's load_model from the cache volume at its
    pinned revision. The run stops between conditions once budget_seconds have passed; a rerun resumes."""
    offline()
    from anatta.modal_app import load_model

    start = time.time()
    cache.reload()
    runs.reload()
    m = plan["models"][key]
    model, tok = load_model(model=str(refusal.snapshot_dir(m["id"], m["revision"])))[:2]
    try:
        res = refusal.run_model(plan, key, out_dir(plan), model, tok, PKG, log=lambda *a: print(*a, flush=True),
                                deadline=start + budget_seconds, ego=ego_path(plan, key))
    finally:
        runs.commit()
    return {**res, "wall_seconds": time.time() - start}


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3600)
def judge_lab(plan: dict) -> dict:
    """(c): the port in the lab's stack on every completion, and the judge control (refusal.judge_lab)."""
    offline()
    import judge

    cache.reload()
    runs.reload()
    template = judge.judge_template(plan["judge"]["template_sha256"])
    model, tok, loaded = judge.load_judge()
    try:
        return refusal.judge_lab(plan, out_dir(plan), PKG, model, tok, template, log=lambda *a: print(*a, flush=True),
                                 loaded=loaded)
    finally:
        runs.commit()


@app.function(image=paper_image, gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3600)
def judge_paper(plan: dict, which: dict, ports: list, name: str) -> dict:
    """In the paper's stack: the original package and/or the port on `which` ({model: [conditions]}), written as
    <name>.json (refusal.judge_paper)."""
    import judge

    cache.reload()
    runs.reload()
    template = judge.judge_template(plan["judge"]["template_sha256"])
    res = refusal.judge_paper(plan, out_dir(plan), PKG, which, tuple(ports), template,
                              log=lambda *a: print(*a, flush=True))
    try:
        return refusal.written(Path(out_dir(plan)) / f"{name}.json", res)
    finally:
        runs.commit()


@app.function(image=paper_image, gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3600)
def swap(plan: dict, which: str, mmlu: bool = False) -> dict:
    """A localizing swap in the paper's stack: "q0" or "reference", with the MMLU pass if the miss was on the MMLU
    bar (refusal.swap)."""
    cache.reload()
    runs.reload()
    try:
        return refusal.swap(plan, out_dir(plan), PKG, which, mmlu, log=lambda *a: print(*a, flush=True))
    finally:
        runs.commit()


@app.function(volumes={RUNS: runs}, cpu=1.0, memory=2048, timeout=600)
def stored(plan: dict) -> dict:
    """The stages already on the volume under this plan (refusal.stored)."""
    runs.reload()
    return refusal.stored(plan, out_dir(plan))


# ------------------------------------------------------------------------------------------------- the laptop


def load_card():
    return module_from_file("refusal_direction_card", HERE / "card.py")


def make_plan(cfg: dict, smoke: bool = False, gpu: str = GPU) -> dict:
    """Everything a run's files depend on, as plain JSON, the registered block included, so the card grades a run by
    its own plan. A smoke run differs only in its smoke flag and its 64 new tokens; its two candidates, 8 prompts and
    50 MMLU items are read from smoke_settings. config.yaml's settings that the code fixes are checked first
    (check_settings)."""
    check_settings(cfg)
    gen = dict(cfg["generation"])
    if smoke:
        gen["max_new_tokens"] = cfg["smoke"]["max_new_tokens"]
    return json.loads(json.dumps({
        "check": "refusal_direction", "smoke": smoke, "models": cfg["models"], "judge": cfg["judge"],
        "data": cfg["data"], "filter": cfg["filter"], "selection": cfg["selection"], "generation": gen,
        "mmlu": cfg["mmlu"], "aggregation": cfg["aggregation"], "inhouse": cfg["inhouse"], "ego": cfg["ego"],
        "smoke_settings": cfg["smoke"], "cost": cfg["cost"], "gpu": gpu, "registered": cfg["registered"],
        "code": refusal.code_hashes(PKG, HERE), "pins": {"lab": LAB_PINS, "paper": PAPER_PINS},
        "container_sha256": container_sha256()}))


# The rules config.yaml's smoke.candidates names: the two candidates a smoke run scores, on a model of n blocks.
SMOKE_CANDIDATES = {"middle": lambda n: [(-1, n // 2), (-1, n // 2 + 1)]}


def functions_in(path: Path) -> dict[str, ast.FunctionDef]:
    """Every function and method defined in a source file, by name."""
    return {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}


def default_of(fn: ast.FunctionDef, name: str):
    """The literal default of fn's parameter `name` (KeyError if it has none)."""
    a = fn.args
    pos = [*a.posonlyargs, *a.args]
    pairs = [*zip(pos[len(pos) - len(a.defaults):], a.defaults), *zip(a.kwonlyargs, a.kw_defaults)]
    for arg, d in pairs:
        if arg.arg == name and d is not None:
            return ast.literal_eval(d)
    raise KeyError(f"{fn.name} has no default for {name}")


def constant_in(path: Path, name: str):
    """The literal value of a module-level assignment in a source file (KeyError if there is none)."""
    for n in ast.parse(path.read_text()).body:
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(n.value)
    raise KeyError(f"{path.name} assigns no {name}")


def keywords_passed(tree: ast.AST, name: str) -> list[ast.expr]:
    """The values passed as keyword `name` in any call inside `tree` (as expressions)."""
    return [k.value for c in ast.walk(tree) if isinstance(c, ast.Call) for k in c.keywords if k.arg == name]


def calls_strip(fn: ast.FunctionDef) -> bool:
    return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "strip"
               for c in ast.walk(fn))


def code_settings(pkg: Path = PKG, here: Path = HERE) -> dict[str, list[tuple[str, object]]]:
    """The settings the code fixes in its own text, read off the source with ast (nothing is imported, so the laptop
    needs no torch): {config.yaml key: [(where, the value the code uses)]}. An empty list means the code was not
    found where it is looked for."""
    ref, sel = functions_in(here / "reference.py"), functions_in(pkg / "select.py")
    mine_src = ast.parse((here / "refusal.py").read_text())
    mine = {n.name: n for n in ast.walk(mine_src) if isinstance(n, ast.FunctionDef)}
    out: dict[str, list[tuple[str, object]]] = {}

    def gather(key, places):
        """Each place's value; a place whose code is missing is left out, and a value that is not a literal is
        recorded as such, so it differs from config.yaml's and is reported."""
        out[key] = []
        for where, get in places:
            try:
                out[key].append((where, get()))
            except (KeyError, IndexError):
                pass
            except ValueError:
                out[key].append((where, "<not a literal>"))

    # the filter's threshold: the four filter_examples calls in filter_data, which make the kept sets (filter_stage's
    # own kept indices are checked against them in every run)
    calls = [c for c in ast.walk(ref["filter_data"]) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) == "filter_examples"] if "filter_data" in ref else []
    gather("filter.threshold", [("reference.filter_data", lambda c=c: ast.literal_eval(c.args[2])) for c in calls])
    gather("filter.cut_harmless_train",
           [("reference.filter_data", lambda: default_of(ref["filter_data"], "cut_harmless_train"))]
           + [("refusal.py", lambda v=v: ast.literal_eval(v)) for v in keywords_passed(mine_src, "cut_harmless_train")])
    gather("selection.prune_layer_percentage",
           [(f"{m}.{f}", lambda fs=fs, f=f: default_of(fs[f], "prune_layer_percentage"))
            for m, fs, names in (("select", sel, ("filter_fn", "filtered", "choose")),
                                 ("reference", ref, ("filter_fn", "select_direction"))) for f in names]
           + [("refusal.py", lambda v=v: ast.literal_eval(v))
              for v in keywords_passed(mine_src, "prune_layer_percentage")])
    for k in ("kl_threshold", "induce_refusal_threshold"):
        gather(f"selection.{k}", [("select.LADDER[0]", lambda k=k: constant_in(pkg / "select.py", "LADDER")[0][k]),
                                  ("reference.select_direction", lambda k=k: default_of(ref["select_direction"], k))])
    gather("mmlu.description", [("bench/mc.py DESCRIPTION", lambda: constant_in(pkg / "bench" / "mc.py",
                                                                                "DESCRIPTION"))])
    gather("judge.scores", [("judge.py POSSIBLE_SCORES", lambda: list(constant_in(here / "judge.py",
                                                                                 "POSSIBLE_SCORES")))])
    # the repo id the judge stages load the judge by, which prepare fetches and loads by (offline_loads)
    gather("judge.adapter.id", [("judge.py ADAPTER", lambda: constant_in(here / "judge.py", "ADAPTER"))])
    gather("generation.strip", [("refusal.generate_jbb", lambda: calls_strip(mine["generate_jbb"])),
                                ("reference generate_completions", lambda: calls_strip(ref["generate_completions"]))])
    return out


def check_settings(cfg: dict, pkg: Path = PKG, here: Path = HERE) -> None:
    """Every setting config.yaml states that the code fixes in its own text must be the value the code uses, at plan
    time, before any spend: refusal.run_model's check of the selection thresholds, extended (ruling, 1 Nov). The code
    stays as ported; a difference refuses the plan, so the card never reports a value that did not drive the run."""
    bad = []
    for key, places in code_settings(pkg, here).items():
        want = cfg
        for part in key.split("."):
            want = want[part]
        if not places:
            bad.append(f"{key}: not found in the code")
        bad += [f"{key}: config.yaml says {want!r}, {where} uses {got!r}" for where, got in places if got != want]
    rule = SMOKE_CANDIDATES.get(cfg["smoke"]["candidates"])
    for key, m in cfg["models"].items():
        got = refusal.smoke_which(m["n_layers"])
        if rule is None or got != rule(m["n_layers"]):
            bad.append(f"smoke.candidates: config.yaml says {cfg['smoke']['candidates']!r}, refusal.smoke_which "
                       f"gives {got} on {key}")
    if bad:
        raise SystemExit("config.yaml states settings the code does not use (edit config.yaml, not the ported code): "
                         + "; ".join(bad))


def ego_check(plan: dict) -> dict:
    """The registered v_ego on the laptop, before any spend: its sha256 against config.yaml's. A missing or different
    file is a warning, not a stop: E-82 and E-82r are ungraded, and the container then marks them not read."""
    e = plan["ego"]
    p = Path(e["file"])
    sha = refusal.sha256_file(p) if p.is_file() else None
    if sha is None:
        print(f"warning: the registered v_ego {p} is missing, so the image carries none and E-82 and E-82r will not "
              "be read", flush=True)
    elif sha != e["sha256"]:
        print(f"warning: {p} has sha256 {sha}, not the registered {e['sha256']}, so E-82 and E-82r will not be read",
              flush=True)
    else:
        print(f"v_ego: {p} has the registered sha256 {sha[:12]}", flush=True)
    return {"file": str(p), "sha256": sha, "registered": e["sha256"]}


def save(d: Path, name: str, obj) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def other_plan(d: Path, plan: dict) -> bool:
    p = d / "plan.json"
    return p.exists() and json.loads(p.read_text()) != plan


def finish(card, cfg: dict, d: Path, stop: str | None = None) -> dict:
    """Write the card, print its headline, and stop with `stop` if given."""
    c = card.write_card(cfg, d)
    print("\n".join(card.headline(c)), flush=True)
    print(f"wrote {d / 'card.json'}", flush=True)
    if stop:
        raise SystemExit(f"stopped: {stop}")
    return c


SPEND_BASIS = ("an estimate: each Modal call's wall time (measured on the laptop; for a model run, the larger of that "
               "and its container's own report) at config.yaml's hourly rates, the higher where two apply; not Modal's "
               "bill")


class Spend:
    """The run's spend, estimated from each Modal call's wall time at config.yaml's rates, kept in spend.json across
    launches of the same results folder, printed after every call (labelled an estimate), with the registered stop
    enforced on the estimate before each."""

    def __init__(self, path: Path, cost: dict):
        self.path, self.cost = path, cost
        self.entries = json.loads(path.read_text())["entries"] if path.exists() else []

    @property
    def total(self) -> float:
        return sum(e["usd"] for e in self.entries)

    def rate(self, kind: str) -> float:
        """A call's hourly rate. A GPU container also runs CPU cores and memory, so both config rates apply to it and
        the higher is used (ruling, 1 Nov); a CPU container takes the CPU rate."""
        rates = [self.cost["cpu_usd_per_hour"]] + ([self.cost["gpu_usd_per_hour"]] if kind == "gpu" else [])
        return max(rates)

    def room(self) -> float:
        return self.cost["stop_usd"] - self.total

    def guard(self, name: str, worst_usd: float) -> None:
        if worst_usd > self.room():
            raise SystemExit(f"stopped before {name}: an estimated ${self.total:.2f} spent, and its estimated worst "
                             f"case of ${worst_usd:.2f} would pass the ${self.cost['stop_usd']:.0f} stop")

    def add(self, name: str, seconds: float, kind: str) -> None:
        usd = seconds / 3600 * self.rate(kind)
        self.entries.append({"call": name, "kind": kind, "seconds": round(seconds, 1), "usd": round(usd, 4)})
        self.path.write_text(json.dumps({"estimate": True, "basis": SPEND_BASIS, "entries": self.entries,
                                         "total_usd": round(self.total, 4), "stop_usd": self.cost["stop_usd"]},
                                        indent=1) + "\n")
        print(f"spend (estimate): {name} {seconds / 60:.1f} min, ${usd:.2f}; estimated total ${self.total:.2f} of "
              f"the ${self.cost['stop_usd']:.0f} stop", flush=True)

    def budgets(self, hours: dict[str, float], floor_seconds: float = 0.0) -> dict[str, int]:
        """Timeouts in seconds for GPU calls that run at once: each its configured hours, all cut by one factor if
        their sum at the GPU rate would pass the room left. A cut that leaves any call under floor_seconds refuses
        the launch: such a container could start and load its model, and not finish (ruling, 1 Nov)."""
        want = sum(hours.values()) * self.rate("gpu")
        factor = min(1.0, self.room() / want) if want else 1.0
        if factor <= 0:
            raise SystemExit(f"stopped: an estimated ${self.total:.2f} spent, nothing left under the "
                             f"${self.cost['stop_usd']:.0f} stop")
        out = {k: int(h * factor * 3600) for k, h in hours.items()}
        short = {k: s for k, s in out.items() if s < floor_seconds}
        if short:
            raise SystemExit(f"stopped before the model runs: an estimated ${self.room():.2f} is left under the "
                             f"${self.cost['stop_usd']:.0f} stop, which would cut "
                             + ", ".join(f"{k} to {s / 60:.0f} min" for k, s in short.items())
                             + f", under the {floor_seconds / 60:.0f}-minute floor: raise the stop (that needs "
                               "Eugene's OK) or start a fresh plan")
        return out

    def call(self, name: str, fn, *args, kind: str = "gpu", timeout: float | None = None):
        """fn.remote(*args), guarded at its worst case (timeout, or the configured hours) and costed after."""
        if timeout is not None:
            self.guard(name, timeout / 3600 * self.rate(kind))
        start = time.time()
        try:
            return fn.remote(*args)
        finally:
            self.add(name, time.time() - start, kind)


def on_gpu(fn, gpu: str, timeout: int | None = None):
    """fn with the run's GPU, check 1's allocator setting and, if given, a timeout."""
    opts = {"gpu": gpu, "env": ALLOC}
    if timeout is not None:
        opts["timeout"] = timeout
    return fn.with_options(**opts)


def run_models(plan: dict, have: dict, d: Path, gpu: str, spend: Spend) -> list[str]:
    """Both models at once, those not already on the volume; each result saved as it comes back. Each run is costed
    at the larger of its container's own wall time and the laptop's (from the spawn to its result, waited for in its
    own thread so that it is its own), so the container's start and the image pull count (ruling, 1 Nov). One model's
    error does not stop the other's run. Returns the errors, in model order."""
    cost = plan["cost"]
    todo = [k for k in refusal.MODELS if have.get(k) is None]
    for k in refusal.MODELS:
        if have.get(k) is not None:
            save(d, f"{k}.json", have[k])
    if not todo:
        return []
    budgets = spend.budgets({k: cost[f"{k}_hours"] for k in todo}, cost["model_min_hours"] * 3600)
    spend.guard("the model runs", sum(budgets.values()) / 3600 * spend.rate("gpu"))
    started = time.time()
    calls = {k: on_gpu(run_model, gpu, budgets[k]).spawn(plan, k, budgets[k] - 900) for k in todo}
    for k in todo:
        print(f"{k}: running, timeout {budgets[k] / 3600:.2f} h", flush=True)

    def wait(k):
        try:
            return k, calls[k].get(), None, time.time() - started
        except Exception as e:  # reported once the other model is saved
            return k, None, e, time.time() - started

    errors = {}
    with ThreadPoolExecutor(max_workers=len(todo)) as pool:
        for done in as_completed([pool.submit(wait, k) for k in todo]):
            k, res, err, laptop = done.result()
            if err is None:
                save(d, f"{k}.json", res)
                spend.add(f"run_model {k}", max(res.get("wall_seconds", 0.0), laptop), "gpu")
            else:
                spend.add(f"run_model {k} (failed)", laptop, "gpu")
                errors[k] = f"{k}: {type(err).__name__}: {err}"
    return [errors[k] for k in todo if k in errors]


@app.local_entrypoint()
def main(results: str, smoke: bool = False, gpu: str = GPU, prepare_only: bool = False):
    import judge

    card = load_card()
    cfg = card.load_config()
    d = card.run_folder(card.results_path(results), smoke)
    plan = make_plan(cfg, smoke, gpu)
    if other_plan(d, plan):
        raise SystemExit(f"{d} holds the run of another plan (its plan.json differs from this one): use a fresh "
                         "--results, or move that folder away")
    save(d, "plan.json", plan)
    spend = Spend(d / "spend.json", plan["cost"])
    print(f"C3 refusal direction, {'smoke' if smoke else 'full'} run, plan {plan_id(plan)}, on {gpu}: {out_dir(plan)}",
          flush=True)
    ego_check(plan)  # before any spend; a warning, not a stop
    have = spend.call("stored", stored, plan, kind="cpu", timeout=600)
    prep = have["prepare"] or spend.call("prepare", prepare, plan, kind="cpu", timeout=2 * 3600)
    save(d, "prepare.json", prep)
    reg = prep["registration"]
    if not reg["ok"]:
        for name, row in reg.get("files", {}).items():
            f = prep["files"][name]
            print(f"  {name}: {f['bytes']} bytes, sha256 {f['sha256']}, count {f['count']}; missing "
                  f"{row['missing'] or 'nothing'}, differ {row['differ'] or 'nothing'}", flush=True)
        return finish(card, cfg, d, "register each data file's sha256 and count in config.yaml (prepare.json), "
                                    "commit, and launch again")
    # Both stacks load offline every tokenizer and config their later stages load, before any GPU starts; prepare and
    # prepare_paper stop on a failure, and a rerun of the plan reads it from their records and stops here.
    failed = refusal.offline_failures(prep.get("offline"), plan, "lab")
    if not failed:
        pp = have["prepare_paper"] or spend.call("prepare_paper", prepare_paper, plan, kind="cpu", timeout=900)
        save(d, "prepare_paper.json", pp)
        failed = refusal.offline_failures(pp.get("offline"), plan, "paper")
    if failed:
        return finish(card, cfg, d, "a later stage could not load offline: " + "; ".join(failed))
    if prepare_only:
        print("--prepare-only: prepare is done, the data are registered and both stacks load offline", flush=True)
        return finish(card, cfg, d)
    try:
        judge.judge_template(plan["judge"]["template_sha256"])
    except RuntimeError as e:
        return finish(card, cfg, d, str(e))

    errors = run_models(plan, have, d, gpu, spend)
    if errors:
        return finish(card, cfg, d, "; ".join(errors))
    # A model run that stopped (the MMLU golden test's gate, filtering kept no prompts in a split, a smoke run's
    # failed bitwise premise, or a full run's MMLU floor) is final for its plan. Ministral's stopping at filtering or
    # at the MMLU floor, a stop of its own (refusal.MODEL_STOPS), leaves Qwen's run to be judged and graded: C3-M is a
    # tally miss and Ministral's expectations are not read (rulings, 1 Nov). Any other stop ends here, as does any
    # stop of Qwen's: the judge port gate runs on its Q0 and QA.
    stopped = {k: r for k in refusal.MODELS if (r := card.read(d / f"{k}.json")) and r.get("stopped")}
    why = {k: f"{k} stopped at {r['stopped']}" + (f" ({r['message']})" if r.get("message") else "")
           for k, r in stopped.items()}
    if set(stopped) == {"ministral"} and stopped["ministral"]["stopped"] in refusal.MODEL_STOPS:
        print(f"{why['ministral']}: C3-M is a miss in the tally and Ministral's expectations are not read; Qwen's run "
              "is judged and graded", flush=True)
    elif stopped:
        return finish(card, cfg, d, "; ".join(why.values()))
    hours = int(plan["cost"]["judge_hours"] * 3600)
    jl = have["judge_lab"] or spend.call("judge_lab", on_gpu(judge_lab, gpu, hours), plan, timeout=hours)
    save(d, "judge_lab.json", jl)
    jp = have["judge_paper"] or spend.call("judge_paper", on_gpu(judge_paper, gpu, hours), plan,
                                           {"qwen": ["Q0", "QA"]}, ["original", "port"], "judge_paper", timeout=hours)
    save(d, "judge_paper.json", jp)
    gate = card.judge_gate(card.load_results(d), plan["registered"])
    if gate is not None and not gate["port_ok"]:
        return finish(card, cfg, d, "the judge port gate: the port and the original package disagree in the paper's "
                                    "stack; fix the port and re-score every completion before any grade")
    if gate is not None and not gate["stack_ok"]:
        allc = have["judge_original_all"] or spend.call(
            "judge_original_all", on_gpu(judge_paper, gpu, hours), plan, {k: [n for n in v] for k, v in
                                                                         jl["scores"].items()},
            ["original"], "judge_original_all", timeout=hours)
        save(d, "judge_original_all.json", allc)
    c = card.build_card(cfg, d)
    g = c.get("grades") or {}
    q = card.load_results(d).get("qwen") or {}
    sel = q.get("select") or {}
    swap_hours = int(plan["cost"]["swap_hours"] * 3600)
    if (g.get("c3_base") or {}).get("result") == "fail":
        res = have["swap_q0"] or spend.call("swap q0", on_gpu(swap, gpu, swap_hours), plan, "q0", False,
                                            timeout=swap_hours)
        save(d, "swap_q0.json", res)
    both_viable = (sel.get("choice") or {}).get("viable") and (sel.get("choice_ref") or {}).get("viable")
    c3, c3r = g.get("c3") or {}, g.get("c3_r") or {}
    if c3.get("result") == "fail" and c3r.get("result") == "fail" and both_viable:
        # the swap adds the MMLU pass only if a miss was on the MMLU bar (ruling, 1 Nov)
        on_mmlu = c3.get("mmlu_within") is False or c3r.get("mmlu_within") is False
        res = have["swap_reference"] or spend.call("swap reference", on_gpu(swap, gpu, swap_hours), plan, "reference",
                                                   on_mmlu, timeout=swap_hours)
        save(d, "swap_reference.json", res)
    return finish(card, cfg, d)


@app.local_entrypoint()
def card(results: str, smoke: bool = False):
    mod = load_card()
    cfg = mod.load_config()
    d = mod.run_folder(mod.results_path(results), smoke)
    c = mod.write_card(cfg, d)
    print("\n".join(mod.headline(c)), flush=True)
