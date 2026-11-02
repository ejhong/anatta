"""Calibration check 7, relearning, on Modal: does the lab's LoRA loop (anatta.train.sft.train) move cais/Zephyr_RMU
the way Łucki et al.'s TRL run did (WMDP-Bio 29.9% -> 61.6% after LoRA training on the first 10 rows of WMDP's bio
retain corpus)? README.md has the claim, the sources, the bands and what a pass transfers; config.yaml every
setting; relearn.py the code that runs in the containers; card.py the card.

Stages, each a Modal function writing under out_dir(plan) on the anatta-runs volume, in this order:
  fetch        CPU, the lab's image. Fetches the checkpoint and the two parquet files at their pinned revisions into
               the anatta-hf-cache volume, their sha256 checked, and Łucki et al.'s data code at its pinned commit
               into the run's folder, the file's sha256 and the two chat strings' checked (the strings are never kept
               in the repository).
  convert_tokenizer
               CPU, Łucki's stack (Python 3.10, transformers 4.41.2, tokenizers 0.19.1), offline. The checkpoint's
               tokenizer as Łucki's code loads it, converted from tokenizer.model and saved to the run's folder, each
               file's sha256 recorded. transformers 5.21's own build of it drops the word-start marker on the answer
               letters (NOTES, 31 Oct 15:22 UTC), so the lab's stages load this saved one, never tokenizer.model.
  prepare      CPU, the lab's image. Builds the 10 items and the scorer's 1,273 prompts under the lab's stack with the
               saved tokenizer, and checks them: each item the BOS and the tail of its rendering, the checkpoint's
               left truncation, the chat template's sha256, the four answer ids. It also records what transformers
               5.21 builds from tokenizer.model, so the card states the cause.
  check_ids    CPU, Łucki's stack. The renderings, the items' ids, the prompts' ids, the answer ids and the backend's
               normalizer and pre-tokenizer must equal the lab's. A difference stops the run here.
  padded_path  L40S. Item 0 alone against item 0 inside a padded micro-batch of four, on the model training will see
               in bf16, then on the base model in fp32. It passes if bf16 differs by under 0.01 nats on average, or
               fp32 by under 0.001 (the bf16 excess is then kernel noise); otherwise, or if a difference is not
               finite, it stops the run here.
  train_arm    L40S, one container per arm: seeds 0, 1 and 2 at once, then C7-warm once seed 0 is done (it starts
               from the adapter seed 0 saved after epoch 1).
  score        L40S, one container: the base model, each adapter attached unmerged and taken off again, then the
               base model again. A changed prediction in that last pass stops the run.
Then the card (card.py), here.

Timeouts, as registered (NOTES, 31 Oct 12:55 UTC): prepare 2 hours and the id check 30 minutes (CPU), the padded path
20 minutes, each training arm 30 minutes, scoring 60 minutes. The worst case is about $9 of L40S time, under the $15
stop. No function retries. The two stages the tokenizer fallback added run on CPU and are not yet registered: fetch
keeps the 2 hours prepare had for the download, and the conversion has the id check's 30 minutes.

The data stay on the volume: the containers return facts (counts, lengths, hashes, item 0's first and last 16 ids,
each question's prediction), never the rows, the questions or their token ids.

A run is its plan: every setting and the registered bands, the sha256 of the package files the run calls and of
relearn.py, the sha256 of this file's code that runs in the containers (container_sha256), and both images' package
pins. Each container first checks that its code and installed versions are the plan's. A rerun of the same plan
resumes: the stages already on the volume are read back, not run again. A changed plan (an edited file, a new pin,
--smoke) runs afresh in a new folder, so no result is ever read back under other code. The results come back to
--results (the full run) or --results/smoke; a folder there holding another plan's files is refused, and the card
refuses a stage file made under other code than its plan's.

Usage, from the repo root (the modal CLI; PYTHONPATH=src so the app imports this package):
  PYTHONPATH=src modal run calibration/relearning/app.py::main --results /Users/eugene/prj/align/calib-results/relearning
  PYTHONPATH=src modal run calibration/relearning/app.py::main --results DIR --smoke
  PYTHONPATH=src modal run calibration/relearning/app.py::main --results DIR [--smoke] --cpu-only
  PYTHONPATH=src modal run calibration/relearning/app.py::card --results DIR [--smoke]
--smoke trains 2 items for 2 epochs (seed 0 and C7-warm) and scores 20 questions after the full prepare, id check
and padded path: the plumbing, read as nothing. --cpu-only runs the CPU stages (fetch, the conversion, prepare and the
id check) and stops before the padded path; the same plan run again without it resumes from there. Keep the terminal
open while main runs and do not use --detach: the stages are driven from the laptop. Stopping the terminal stops the
containers; a rerun resumes from the volume.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import math
from pathlib import Path

import modal
import relearn

from anatta.modal_app import CACHE, GPU, PKG, RUNS, cache, run_root, runs

HERE = Path(__file__).resolve().parent


def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# anatta.bench.fingerprint (standard library only) by its file: importing it as anatta.bench.fingerprint would run
# the anatta.bench package's __init__, which imports the koans, and this file is imported under Łucki's stack too.
fingerprint = module_from_file("anatta_bench_fingerprint", PKG / "bench" / "fingerprint.py")
image_pins, plan_id = fingerprint.image_pins, fingerprint.plan_id
# The lab's image: anatta.modal_app's pins (read from that file, so the two stay the same), plus what this check
# needs on top: sentencepiece and protobuf, with which prepare records what transformers 5 builds from the
# checkpoint's tokenizer.model (the lab's stages use the saved conversion instead), and pyarrow to read the parquet
# files. modal_app's image ends with the package added as local files, and Modal refuses a build step after that, so
# this image is built afresh rather than on top of it.
LAB_EXTRA = {"sentencepiece": "0.2.0", "protobuf": "5.29.3", "pyarrow": "21.0.0"}
LAB_PINS = {**image_pins(PKG / "modal_app.py"), **LAB_EXTRA}
# Łucki et al.'s stack for the conversion and the id check: the versions in their env.yml (unlearning-vs-safety at
# 75dade5) of transformers, tokenizers and everything they need to load and convert the checkpoint's tokenizer. No
# torch: both stages tokenize only.
LUCKI_PINS = {"transformers": "4.41.2", "tokenizers": "0.19.1", "sentencepiece": "0.2.0", "protobuf": "3.20.3",
              "huggingface-hub": "0.23.4", "jinja2": "3.1.4", "markupsafe": "2.1.5", "numpy": "2.0.0",
              "pyyaml": "6.0.1", "regex": "2024.5.15", "safetensors": "0.4.3", "packaging": "24.1",
              "filelock": "3.15.1", "requests": "2.32.3", "tqdm": "4.66.4", "fsspec": "2024.5.0",
              "typing-extensions": "4.12.2", "certifi": "2024.6.2", "idna": "3.7", "urllib3": "2.2.2",
              "charset-normalizer": "3.3.2"}
ENV = {"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root"}


def pip(pins: dict) -> list[str]:
    return [f"{name}=={version}" for name, version in pins.items()]


lab_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install(*pip(LAB_PINS))
             .env(ENV)
             .add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"])
             .add_local_file(str(HERE / "relearn.py"), "/root/relearn.py"))
lucki_image = (modal.Image.debian_slim(python_version="3.10")
               .uv_pip_install(*pip(LUCKI_PINS))
               .env({**ENV, "HF_HUB_OFFLINE": "1"})
               .add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"])
               .add_local_file(str(HERE / "relearn.py"), "/root/relearn.py"))
app = modal.App("anatta-calib-relearning", image=lab_image)


def out_dir(plan: dict) -> str:
    """A run's folder on the anatta-runs volume."""
    return f"{run_root(plan['model']['id'])}/relearning/{'smoke' if plan['smoke'] else 'full'}/{plan_id(plan)}"


def snapshot_path(plan: dict) -> str:
    """The checkpoint's folder in the cache volume at its pinned revision (models are loaded from it, offline)."""
    return str(relearn.snapshot_dir(plan["model"]["id"], plan["model"]["revision"]))


def offline() -> None:
    """The GPU containers read the checkpoint from the volume and download nothing; set before transformers loads."""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"


CONTAINER_FUNCTIONS = ("out_dir", "snapshot_path", "offline", "fetch", "convert_tokenizer", "prepare", "check_ids",
                       "padded_path", "train_arm", "score", "stored")  # this file's code that runs in a run's containers


def container_source(source: str | None = None) -> str:
    """The source of CONTAINER_FUNCTIONS (each with its decorator), read from this file's text (or `source`), since
    the Modal function object hides the plain function."""
    src = Path(__file__).read_text() if source is None else source
    lines = src.splitlines(keepends=True)
    defs = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    return "".join("".join(lines[min([defs[f].lineno, *(d.lineno for d in defs[f].decorator_list)]) - 1:
                                 defs[f].end_lineno]) for f in CONTAINER_FUNCTIONS)


@functools.cache
def container_sha256(source: str | None = None) -> str:
    """The plan's hash of this file: only its code that runs in the containers, besides the fingerprinted package and
    relearn.py. An edit to the laptop's code here (main, the checks) makes no new plan. Cached for the process, so
    every stage of one launch carries the source its containers run."""
    return relearn.sha256_text(container_source(source))


# ------------------------------------------------------------------------------------------------ containers


@app.function(volumes={CACHE: cache, RUNS: runs}, cpu=4.0, memory=16384, timeout=2 * 3600)
def fetch(plan: dict) -> dict:
    """The checkpoint and the two parquet files into the cache volume, and Łucki et al.'s data code into the run's
    folder, each checked (relearn.fetch)."""
    cache.reload()
    runs.reload()
    return relearn.fetch(plan, out_dir(plan), PKG, commit=lambda: (cache.commit(), runs.commit()),
                         log=lambda *a: print(*a, flush=True))


@app.function(image=lucki_image, volumes={CACHE: cache, RUNS: runs}, cpu=2.0, memory=8192, timeout=1800)
def convert_tokenizer(plan: dict) -> dict:
    """The checkpoint's tokenizer as Łucki's stack converts it, saved to the run's folder with each file's sha256
    (relearn.convert_tokenizer)."""
    cache.reload()
    runs.reload()
    res = relearn.convert_tokenizer(plan, out_dir(plan), PKG, log=lambda *a: print(*a, flush=True))
    runs.commit()
    return res


@app.function(volumes={CACHE: cache, RUNS: runs}, cpu=4.0, memory=16384, timeout=2 * 3600)
def prepare(plan: dict) -> dict:
    """Build and check (relearn.prepare), under the lab's stack with the saved tokenizer."""
    cache.reload()
    runs.reload()
    return relearn.prepare(plan, out_dir(plan), PKG, commit=runs.commit, log=lambda *a: print(*a, flush=True))


@app.function(image=lucki_image, volumes={CACHE: cache, RUNS: runs}, cpu=2.0, memory=8192, timeout=1800)
def check_ids(plan: dict) -> dict:
    """The lab's renderings and ids against Łucki's stack (relearn.check_ids)."""
    cache.reload()
    runs.reload()
    res = relearn.check_ids(plan, out_dir(plan), PKG, log=lambda *a: print(*a, flush=True))
    runs.commit()
    return res


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=1200)
def padded_path(plan: dict) -> dict:
    """Item 0 alone against item 0 in a padded micro-batch of four (relearn.padded_path_stage): on the model training
    will see, the checkpoint from the cache volume in bf16 with a fresh rank-128 LoRA; then on the base model in fp32
    with no LoRA (anatta.models.load). Both loads take the saved tokenizer (relearn.saved_tokenizer_only). The fp32
    model, about 29 GB, loads on the CPU first, hence the memory."""
    offline()
    from anatta.modal_app import load_model
    from anatta.models import load

    cache.reload()
    runs.reload()
    lora = plan["lora"]
    res = relearn.padded_path_stage(
        plan, out_dir(plan), PKG, log=lambda *a: print(*a, flush=True),
        load=lambda **k: load_model(model=snapshot_path(plan), rank=lora["rank"], alpha=lora["alpha"], **k),
        load_fp32=lambda: load(snapshot_path(plan), dtype="float32"))
    runs.commit()
    return res


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=1800)
def train_arm(plan: dict, arm: str) -> dict:
    """One training arm (relearn.train_arm): "s0", "s1", "s2" from a fresh rank-128 LoRA, or "warm" from seed 0's
    adapter after epoch 1, reloaded through anatta.player.attach_adapters(path, trainable=True)."""
    offline()
    from anatta.modal_app import load_model
    from anatta.player import save_adapter

    cache.reload()
    runs.reload()
    lora = plan["lora"]
    return relearn.train_arm(
        plan, out_dir(plan), arm, PKG, commit=runs.commit, log=lambda *a: print(*a, flush=True),
        load=lambda **k: load_model(model=snapshot_path(plan), rank=lora["rank"], alpha=lora["alpha"], **k),
        save_adapter=save_adapter)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=3600)
def score(plan: dict) -> dict:
    """The WMDP scorer (relearn.score) on the base model and each adapter, unmerged, in this one process. A changed
    prediction on the base model's second pass stops it, once scores.json is written."""
    offline()
    from anatta.modal_app import load_model
    from anatta.player import attach_adapters

    cache.reload()
    runs.reload()
    try:
        return relearn.score(plan, out_dir(plan), PKG, log=lambda *a: print(*a, flush=True),
                             load=lambda: load_model(model=snapshot_path(plan))[:2],
                             attach=lambda hf_model, path: attach_adapters(hf_model, path))
    finally:  # the stop on the base model's second pass raises after writing scores.json, which main reads back
        runs.commit()


@app.function(volumes={RUNS: runs}, cpu=1.0, memory=2048, timeout=600)
def stored(plan: dict) -> dict:
    """The stages already on the volume under this plan (relearn.stored)."""
    runs.reload()
    return relearn.stored(plan, out_dir(plan))


# ------------------------------------------------------------------------------------------------- the laptop


def load_card():
    return module_from_file("relearning_card", HERE / "card.py")


def make_plan(cfg: dict, smoke: bool = False, gpu: str = GPU) -> dict:
    """Everything a run's files depend on, as plain JSON, the registered bands, odds and tolerances included, so the
    card grades a run by its own plan and a band edited afterwards cannot regrade it. Refuses a learning-rate list
    that differs from transformers 4.41.2's linear schedule with warmup for the registered settings
    (relearn.linear_warmup_lrs), and a C7-warm the arms cannot reach."""
    t = cfg["train"]
    steps = math.ceil(t["items"] / t["batch"]) * t["epochs"]
    want = relearn.linear_warmup_lrs(t["lr"], steps, t["warmup_ratio"])
    if t["lrs"] != want:
        bad = next((k for k, (a, b) in enumerate(zip(t["lrs"], want)) if a != b), min(len(t["lrs"]), len(want)))
        raise SystemExit(f"config.yaml's train.lrs differ from the schedule's {steps} values at step {bad}")
    train, scorer = dict(t), dict(cfg["scorer"])
    if smoke:
        s = cfg["smoke"]
        train.update(items=s["items"], epochs=s["epochs"], seeds=s["seeds"])
        scorer["questions"] = s["questions"]
    w = cfg["warm"]
    if w["seed"] not in train["seeds"] or not 0 < w["after_epoch"] < train["epochs"]:
        raise SystemExit(f"C7-warm starts from seed {w['seed']} after epoch {w['after_epoch']}, which this run's "
                         f"seeds {train['seeds']} and {train['epochs']} epochs do not reach")
    return json.loads(json.dumps({"check": "relearning", "smoke": smoke, "model": cfg["model"], "data": cfg["data"],
                                  "items": cfg["items"], "lora": cfg["lora"], "train": train, "warm": w,
                                  "padded_path": cfg["padded_path"], "scorer": scorer, "gpu": gpu,
                                  "registered": cfg["registered"], "code": relearn.code_hashes(PKG),
                                  "pins": {"lab": LAB_PINS, "lucki": LUCKI_PINS},
                                  "container_sha256": container_sha256()}))


def on_gpu(fn, gpu: str):
    return fn if gpu == GPU else fn.with_options(gpu=gpu)


def save(d: Path, name: str, obj) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def other_plan(d: Path, plan: dict) -> bool:
    """True if d holds the files of another plan (its plan.json differs from this one)."""
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


def train_all(plan: dict, have: dict, d: Path, gpu: str) -> list[str]:
    """The training arms not yet on the volume: the seeds at once, then C7-warm as soon as its seed is done. Each
    finished arm's done.json is saved to d/train/. Returns the errors (an arm that failed), once every arm started
    has finished."""
    fn, errors = on_gpu(train_arm, gpu), []
    done = dict(have)
    for arm, res in done.items():
        if res is not None:
            save(d / "train", f"{arm}.json", res)
    first = f"s{plan['warm']['seed']}"
    calls = {arm: fn.spawn(plan, arm) for arm in done if arm != "warm" and done[arm] is None}
    for arm in calls:
        print(f"{arm}: training", flush=True)

    def collect(arm: str) -> None:
        try:
            done[arm] = calls[arm].get()
            save(d / "train", f"{arm}.json", done[arm])
            print(f"{arm}: done, final loss {done[arm]['log'][-1]['nll']:.4f}", flush=True)
        except Exception as e:  # reported once the other arms are saved
            errors.append(f"{arm}: {type(e).__name__}: {e}")

    if first in calls:
        collect(first)
    if done.get("warm") is None and done.get(first) is not None:
        calls["warm"] = fn.spawn(plan, "warm")
        print("warm: training", flush=True)
    for arm in calls:
        if arm != first:
            collect(arm)
    return errors


@app.local_entrypoint()
def main(results: str, smoke: bool = False, gpu: str = GPU, cpu_only: bool = False):
    card = load_card()
    cfg = card.load_config()
    d = card.run_folder(card.results_path(results), smoke)
    plan = make_plan(cfg, smoke, gpu)
    if other_plan(d, plan):
        raise SystemExit(f"{d} holds the run of another plan (its plan.json differs from this one: other code, pins or "
                         "settings): use a fresh --results, or move that folder away")
    save(d, "plan.json", plan)
    print(f"C7 relearning, {'smoke' if smoke else 'full'} run, plan {plan_id(plan)}, on {gpu}: {out_dir(plan)}",
          flush=True)
    have = stored.remote(plan)
    got = have["fetch"] or fetch.remote(plan)
    save(d, "fetch.json", got)
    conv = have["convert_tokenizer"] or convert_tokenizer.remote(plan)
    save(d, "convert_tokenizer.json", conv)
    prep = have["prepare"] or prepare.remote(plan)
    save(d, "prepare.json", prep)
    ids = have["ids_check"] or check_ids.remote(plan)
    save(d, "ids_check.json", ids)
    if not ids["equal"]:
        return finish(card, cfg, d, "the lab's stack and Łucki's tokenize differently (ids_check.json); nothing trains")
    if cpu_only:
        print("--cpu-only: the CPU stages are done; the GPU stages start when the same plan runs without it", flush=True)
        return finish(card, cfg, d)
    pp = have["padded_path"] or on_gpu(padded_path, gpu).remote(plan)
    save(d, "padded_path.json", pp)
    if not pp["passed"]:
        return finish(card, cfg, d, f"the padded path: {pp['reading']} (padded_path.json); nothing trains")
    errors = train_all(plan, have["train"], d, gpu)
    if errors:
        return finish(card, cfg, d, "; ".join(errors))
    scores = have["scores"]
    if scores is None:
        try:
            scores = on_gpu(score, gpu).remote(plan)
        except Exception:  # the stop on the base model's second pass writes scores.json first: read it back
            scores = stored.remote(plan)["scores"]
            if scores is None or not scores["base_again_changed"]:
                raise
    save(d, "scores.json", scores)
    if scores["base_again_changed"]:
        return finish(card, cfg, d, f"the base model, scored again after the adapters came off, changed "
                                    f"{scores['base_again_changed']} predictions (scores.json): the unload path is "
                                    "suspect, and C7-base is not read")
    return finish(card, cfg, d)


@app.local_entrypoint()
def card(results: str, smoke: bool = False):
    mod = load_card()
    cfg = mod.load_config()
    d = mod.run_folder(mod.results_path(results), smoke)
    c = mod.write_card(cfg, d)
    print("\n".join(mod.headline(c)), flush=True)
