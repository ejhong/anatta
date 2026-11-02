"""Calibration check 1, the model check, on Modal: does the model as anatta loads, prompts, decodes and parses it
reproduce published numbers? See README.md for the claim, the sources and the bands; config.yaml for every setting.

Stages:
  draw   CPU only. Downloads the pinned parquet files (hf_hub_download into the anatta-hf-cache volume), draws the
         items (anatta.bench.items: all 1,319 GSM8K test items; 1,000 MMLU-Pro test items by largest remainder over
         the categories, seed 0 within each; the cheap tier's stratified 150 per set; 30 per set for the batch-1
         comparison; the battery's own seed-0 250 per set), refuses a draw whose MMLU-Pro allocation differs from
         the registered one and, with --battery-results, one whose battery draw differs from the stored run's
         items.json (ARC was never pinned, so drift since that run stops here, before anything is registered).
         Then it writes the item file to --results and prints its sha256 and the commit ARC's "main" resolved to:
         both go into config.yaml (items.sha256, datasets.arc_challenge.revision), committed, before the run. The
         volume keeps a copy of the item file (keep_items), written only by a draw that passed these checks.
  main   The run. Refuses an item file that is unregistered or differs from config.yaml's sha256, and a battery draw
         that differs from the stored run's items.json; --battery-results is required whenever the Ministral stage
         runs. Then one L40S container per model and condition (anatta.bench.run.run_plan), each loading the model
         from the cache volume at its pinned revision (no download) and checking every prompt before generating:
           Qwen2.5-7B-Instruct, no system message: C1a, C1b, the letter-only format, the battery as used, the cheap
             tier's reference, the first batch twice, batch 1 on 30 + 30 items;
           then, unless Qwen's verdict stops the remaining checks (a fail not explained as format: one passes and
             one fails, or neither passes), Ministral 3 8B with no system message and with "You are a helpful
             assistant." (two containers at once), and a third condition with an empty system message if those two
             differ by more than 3 points on C1a's or C1b's format. --force-ministral overrides the gate, and the
             card says so. The gate reads Qwen's verdict only from a complete Qwen run under the plan this launch
             makes for it: Qwen rows from another plan are refused, forced or not.
         A run is its plan: model, revision, condition, system message, jobs with their items, batch sizes and token
         limits, checks, GPU, item file, the sha256 of the package files whose code it calls
         (anatta.bench.fingerprint) and of this file's code that runs in the container (evaluate and out_dir:
         container_sha256; an edit to the laptop's code here makes no new plan), and the image's package pins.
         Each container first checks that its package files and installed versions are the plan's, then writes
         under the model's run root on the anatta-runs volume,
         anatta.modal_app.run_root(model)/model_check/CONDITION/PLAN_ID/, batch by batch: a rerun of the same plan
         resumes where it stopped, and a changed plan (an edited file, a new pin) runs afresh in a new folder, so no
         result is ever read back under other code. The results come back to --results/<model slug>/CONDITION/ (a
         folder there holding another plan's run is refused). Then card.json (card.py).
  card   The card again from the files in --results (no Modal calls); `python card.py` does the same.

--results must lie outside every git work tree and outside --battery-results: the results hold benchmark items and
model replies, which stay out of repositories, and nothing is ever written into the stored battery run's tree.

Usage, from the repo root (the modal CLI; PYTHONPATH=src so the app imports this package):
  PYTHONPATH=src modal run calibration/model_check/app.py::draw --results DIR --battery-results BATTERY_DIR
  PYTHONPATH=src modal run calibration/model_check/app.py::main --results DIR --battery-results BATTERY_DIR
  PYTHONPATH=src modal run calibration/model_check/app.py::card --results DIR --battery-results BATTERY_DIR
Keep the terminal open while main runs, and do not use --detach: the stages are driven from the laptop, so a
detached run stops advancing when the laptop sleeps, and a rerun while its containers still ran would put two
writers on one folder. Stopping the terminal stops the containers; a rerun resumes from the volume.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import re
from pathlib import Path

import modal

from anatta.bench.fingerprint import code_hashes, image_pins, plan_id
from anatta.bench.items import dump_items, sha256_bytes, sha256_file
from anatta.modal_app import (
    CACHE,
    GPU,
    PKG,
    ROOT,
    RUNS,
    cache,
    image,
    load_model,
    run_root,
    runs,
    slug,
)

HERE = Path(__file__).resolve().parent
ITEMS_ON_VOLUME = f"{ROOT}/model_check/items.json"
TIMEOUT_HOURS = 4  # per container; with one retry, a plan's worst case is 2 x 4 GPU-hours
# The draw's own image. modal_app's image ends with this package added as local files, and Modal refuses a build
# step after that, so pyarrow cannot be added on top of it; the draw needs only the hub client and pyarrow (the
# battery's pin), at modal_app's pins, with the package added last.
draw_image = (modal.Image.debian_slim(python_version="3.12")
              .uv_pip_install("huggingface_hub==2.3.0", "hf_xet==1.7.0", "pyarrow==21.0.0")
              .env({"HF_HOME": f"{CACHE}/hf", "PYTHONPATH": "/root"})
              .add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"]))
app = modal.App("anatta-model-check", image=image)


def out_dir(plan: dict) -> str:
    """A run's folder on the anatta-runs volume."""
    return f"{run_root(plan['model'])}/model_check/{plan['condition']}/{plan_id(plan)}"


CONTAINER_FUNCTIONS = ("evaluate", "out_dir")  # this file's code that runs in a run's container


def container_source(source: str | None = None) -> str:
    """The source of CONTAINER_FUNCTIONS as inspect.getsource gives it (evaluate with its decorator, then out_dir),
    read from this file's text (or `source`), since the Modal function object hides the plain function."""
    src = Path(__file__).read_text() if source is None else source
    lines = src.splitlines(keepends=True)
    defs = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    return "".join("".join(lines[min([defs[f].lineno, *(d.lineno for d in defs[f].decorator_list)]) - 1:
                                 defs[f].end_lineno]) for f in CONTAINER_FUNCTIONS)


@functools.cache
def container_sha256(source: str | None = None) -> str:
    """The plan's hash of this file: only its code that runs in the container, besides the fingerprinted package. An
    edit to evaluate or out_dir makes a new plan; an edit to the laptop's code here (draw, main, the checks) does
    not. The card gives the whole file's sha256, for information. Cached for the process, so every plan of one
    launch carries the source its containers run (Modal mounts this file at launch), even if it is edited mid-run."""
    return sha256_bytes(container_source(source).encode())


def is_commit(revision: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", revision) is not None


# ------------------------------------------------------------------------------------------------ containers


@app.function(image=draw_image, volumes={CACHE: cache}, cpu=2.0, memory=8192, timeout=3600)
def sample(datasets: dict, draws: dict, seed: int) -> dict:
    """The item file: each set's parquet files at the pinned revision (a branch name is resolved to its commit
    first), read in the battery's order, hashed, and drawn from. Returns {item_file, requested: {set: the revision
    asked for}}. The volume's copy is keep_items' business, once draw's checks have passed."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from anatta.bench.items import build_item_file, parquet_files

    api, rows, meta = HfApi(), {}, {}
    for name, d in datasets.items():
        rev = d["revision"] if is_commit(d["revision"]) else api.dataset_info(d["repo"], revision=d["revision"]).sha
        files = parquet_files(api.list_repo_files(d["repo"], repo_type="dataset", revision=rev), d["prefix"])
        if not files:
            raise FileNotFoundError(f"no parquet files under {d['prefix']} in {d['repo']} at {rev}")
        rows[name], meta[name] = [], {"repo": d["repo"], "prefix": d["prefix"], "revision": rev, "files": []}
        for f in files:
            local = hf_hub_download(d["repo"], f, repo_type="dataset", revision=rev)
            part = pq.read_table(local).to_pylist()
            rows[name] += part
            meta[name]["files"].append({"path": f, "sha256": sha256_file(local), "rows": len(part)})
        meta[name]["rows"] = len(rows[name])
    cache.commit()
    return {"item_file": build_item_file(rows, draws, seed, meta),
            "requested": {name: d["revision"] for name, d in datasets.items()}}


@app.function(image=draw_image, volumes={RUNS: runs}, cpu=1.0, memory=2048, timeout=600)
def keep_items(data: bytes, registered: str | None = None) -> dict:
    """The volume's copy of the item file (ITEMS_ON_VOLUME), called by draw once the draw has passed its checks. It
    is written when this draw's sha256 is the registered one, or when nothing is registered and there is no copy
    yet, and is otherwise left as it is. So a redraw before registration cannot lock later draws out: the first draw
    that reproduces the registered sha256 replaces a stale copy. Returns {path, sha256 (the copy's, afterwards),
    written}."""
    path = Path(ITEMS_ON_VOLUME)
    on_volume = sha256_file(path) if path.exists() else None
    sha = sha256_bytes(data)
    keep = sha == registered if registered else on_volume is None
    written = keep and on_volume != sha
    if written:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        runs.commit()
    return {"path": str(path), "sha256": sha if keep else on_volume, "written": written}


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=TIMEOUT_HOURS * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=30.0))
def evaluate(plan: dict, items: dict) -> dict:
    """One plan (one model in one condition): anatta.bench.run.run_plan, with the model loaded by modal_app's
    load_model from the cache volume at the plan's revision, writing under out_dir(plan)."""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"  # the weights are on the volume at the pinned revision: no download
    from anatta.bench.run import run_plan

    return run_plan(plan, items, out_dir(plan), PKG, load=lambda path: load_model(model=path)[:2], commit=runs.commit,
                    log=lambda *a: print(*a, flush=True))


# ------------------------------------------------------------------------------------------------- the laptop


def load_card():
    spec = importlib.util.spec_from_file_location("model_check_card", HERE / "card.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_allocation(item_file: dict, cfg: dict) -> None:
    """The registered allocations must come out of the draw, with no tie at the cut."""
    for name, want in cfg.get("allocation", {}).items():
        got = item_file["meta"]["draws"][name]
        if got.get("allocation") != want:
            raise SystemExit(f"{name}: the draw's allocation {got.get('allocation')} is not the registered {want}")
        if got["info"]["tie_at_cut"]:
            raise SystemExit(f"{name}: two remainders tie at the cut")


def load_items(out: Path, cfg: dict) -> tuple[dict, str]:
    """The item file in --results and its sha256, if that is the one registered in config.yaml."""
    path = out / cfg["items"]["file"]
    if not path.exists():
        raise SystemExit(f"no item file at {path}: run app.py::draw first")
    data = path.read_bytes()
    got, want = sha256_bytes(data), cfg["items"].get("sha256")
    if not want:
        raise SystemExit(f"the item file's sha256 is {got}; register it in config.yaml (items.sha256) and commit "
                         "before the run")
    if got != want:
        raise SystemExit(f"the item file's sha256 is {got}, not the registered {want}")
    return json.loads(data), got


def check_battery_items(item_file: dict, battery_results: str) -> None:
    """The battery draw must be the stored run's items: the same ids, prompts and golds, in the same order."""
    stored = json.loads((Path(battery_results) / "items.json").read_text())
    for s in ("gsm8k", "mmlu_pro", "arc_challenge"):
        ours = [(i, item_file["items"][i]["prompts"][s], item_file["items"][i]["gold"])
                for i in item_file["draws"][f"battery_{s}"]]
        theirs = [(x["id"], x["prompt"], x["gold"]) for x in stored if x["set"] == s]
        if ours != theirs:
            bad = next((k for k, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
            raise SystemExit(f"battery_{s}: the draw differs from the stored run's items.json at position {bad}")


def plan_jobs(cfg: dict, key: str, condition: str, item_file: dict) -> list[dict]:
    """The jobs one model runs in one condition, each with its items, batch size and token limit."""
    m = cfg["models"][key]
    jobs = []
    for name in cfg["plan"][key][condition]:
        draw, fmt = cfg["jobs"][name]
        jobs.append({"name": name, "format": fmt, "draw": draw, "ids": item_file["draws"][draw],
                     "batch": m["batch"].get(fmt, m["batch"]["default"]), "max_new": cfg["max_new_tokens"][fmt]})
    return jobs


def plan_checks(cfg: dict, condition: str, item_file: dict, jobs: list[dict]) -> tuple[dict, dict | None]:
    """The first batch twice in every run; batch 1 in each model's default run, on jobs that run there."""
    names = {j["name"] for j in jobs}
    det = dict(cfg["checks"]["determinism"])
    b1 = ({job: item_file["draws"][draw] for job, draw in cfg["checks"]["batch1"].items() if job in names}
          if condition == "default" else None)
    return det, b1


def make_plan(cfg: dict, key: str, condition: str, item_file: dict, items_sha: str, gpu: str = GPU) -> dict:
    """Everything a run's files depend on, as plain JSON: the sha256 of the package files and of this file's
    container code, and the image's pins, included."""
    m = cfg["models"][key]
    jobs = plan_jobs(cfg, key, condition, item_file)
    det, b1 = plan_checks(cfg, condition, item_file, jobs)
    return json.loads(json.dumps({"model": m["id"], "revision": m["revision"], "condition": condition,
                                  "system": cfg["conditions"][condition], "bos": m["bos"], "jobs": jobs,
                                  "determinism": det, "batch1": b1, "gpu": gpu, "items_sha256": items_sha,
                                  "code": code_hashes(PKG), "pins": image_pins(PKG / "modal_app.py"),
                                  "container_sha256": container_sha256()}))


def needed_items(item_file: dict, plan: dict) -> dict:
    ids = {i for j in plan["jobs"] for i in j["ids"]} | {i for v in (plan["batch1"] or {}).values() for i in v}
    return {i: item_file["items"][i] for i in sorted(ids)}


def local_dir(out: Path, model: str, condition: str) -> Path:
    return out / slug(model) / condition


def other_plan(d: Path, plan: dict) -> bool:
    """True if d holds the files of another plan (its plan.json differs from this one)."""
    stored = d / "plan.json"
    return stored.exists() and json.loads(stored.read_text()) != plan


def local_complete(d: Path, plan: dict) -> bool:
    """True if d holds every file of this plan's run; a folder holding another plan's run is refused."""
    if other_plan(d, plan):
        raise SystemExit(f"{d} holds the run of another plan (its plan.json differs from this one: other code, pins "
                         "or settings): use a fresh --results with the same items.json, or move that folder away")
    need = ["plan.json", "run_info.json", "checks.json", *(f"{j['name']}.json" for j in plan["jobs"])]
    need += (["determinism.json"] if plan["determinism"] else []) + (["batch1.json"] if plan["batch1"] else [])
    return all((d / f).exists() for f in need)


def qwen_for_gate(card, cfg: dict, item_file: dict, items_sha: str, out: Path, gpu: str = GPU) -> dict | None:
    """Qwen's verdict for the Ministral gate, read only from a complete Qwen run in --results under the plan this
    launch makes for Qwen (None while there is none). Qwen rows from another plan are refused, forced or not."""
    plan = make_plan(cfg, "qwen", "default", item_file, items_sha, gpu)
    d = local_dir(out, plan["model"], "default")
    if other_plan(d, plan):
        raise SystemExit(f"the Qwen rows in {d} are from another plan (its plan.json differs from the one this launch "
                         "makes: other code, pins or settings), so the Ministral gate will not read them: run Qwen "
                         "under this plan first (--stages qwen) in a fresh --results with the same items.json")
    return card.qwen_verdict(cfg, out) if local_complete(d, plan) else None


def save_run(d: Path, res: dict) -> None:
    """A run's results as files: plan, run_info, checks, determinism, batch1 and one file per job."""
    d.mkdir(parents=True, exist_ok=True)
    files = {"plan.json": res["plan"], "run_info.json": res["run_info"], "checks.json": res["checks"],
             "determinism.json": res.get("determinism"), "batch1.json": res.get("batch1"),
             **{f"{name}.json": r for name, r in res["jobs"].items()}}
    for name, obj in files.items():
        if obj is not None:
            (d / name).write_text(json.dumps(obj, indent=1))


def launch(cfg: dict, key: str, condition: str, item_file: dict, items_sha: str, out: Path, gpu: str = GPU,
           spawn: bool = False):
    """Start (or, if its files are all in --results, skip) one model in one condition. Returns (folder, the call's
    result, or the spawned call, or None)."""
    plan = make_plan(cfg, key, condition, item_file, items_sha, gpu)
    d = local_dir(out, plan["model"], condition)
    if local_complete(d, plan):
        print(f"{key} {condition}: on disk in {d}", flush=True)
        return d, None
    print(f"{key} {condition}: plan {plan_id(plan)}, {len(plan['jobs'])} jobs, "
          f"{sum(len(j['ids']) for j in plan['jobs'])} prompts, system {plan['system']!r}, on {gpu}", flush=True)
    # Expandable segments let the allocator use memory it has reserved without fragmenting it. No arithmetic changes,
    # and the plan doesn't hash it (31 Oct: Ministral's default arm ran out at its last letter-only batch of 50).
    fn = evaluate.with_options(gpu=gpu, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    args = (plan, needed_items(item_file, plan))
    return d, (fn.spawn(*args) if spawn else fn.remote(*args))


@app.local_entrypoint()
def draw(results: str, battery_results: str = ""):
    card = load_card()
    cfg = card.load_config()
    out = card.results_path(results, battery_results)
    want = cfg["items"].get("sha256")
    got = sample.remote(cfg["datasets"], {name: dict(spec) for name, spec in cfg["draws"].items()}, cfg["seed"])
    item_file = got["item_file"]
    print({name: len(ids) for name, ids in item_file["draws"].items()}, flush=True)
    check_allocation(item_file, cfg)
    if battery_results:  # drift since the stored run (ARC was never pinned) stops here, before anything is registered
        check_battery_items(item_file, battery_results)
        print("the battery draw is the stored run's items", flush=True)
    data = dump_items(item_file)
    sha = sha256_bytes(data)
    path = out / cfg["items"]["file"]
    if want and sha != want:
        raise SystemExit(f"this draw's sha256 is {sha}, not the registered {want}; {path} and the volume's copy are "
                         "left as they were")
    path.write_bytes(data)
    vol = keep_items.remote(data, want)
    if vol["sha256"] != sha:
        print(f"note: the volume's copy {vol['path']} is another draw's (sha256 {vol['sha256']}); it is left as it is "
              "until a draw reproduces the registered sha256", flush=True)
    print(f"wrote {path}", flush=True)
    if not battery_results:
        print("note: no --battery-results, so the battery draw was not compared with the stored run's items.json; "
              "draw with it before registering", flush=True)
    print(f"items.sha256: {sha}", flush=True)
    resolved = {s: (asked, item_file["meta"]["datasets"][s]["revision"]) for s, asked in got["requested"].items()}
    unpinned = [s for s, (asked, rev) in resolved.items() if rev != asked]
    for s in unpinned:
        print(f"datasets.{s}.revision: {resolved[s][1]}  ({resolved[s][0]!r} resolved at this draw)", flush=True)
    todo = ([] if want else ["items.sha256"]) + [f"datasets.{s}.revision" for s in unpinned]
    print(f"put {' and '.join(todo)} into config.yaml and commit before the run" if todo else
          "they match config.yaml", flush=True)


@app.local_entrypoint()
def main(results: str, battery_results: str = "", stages: str = "qwen,ministral", force_ministral: bool = False,
         gpu: str = GPU):
    card = load_card()
    cfg = card.load_config()
    stages = {s.strip() for s in stages.split(",") if s.strip()}
    if "ministral" in stages and not battery_results:
        raise SystemExit("--battery-results is required when the Ministral stage runs: its battery rerun is compared "
                         "with that stored run (pass --stages qwen to run Qwen alone)")
    out = card.results_path(results, battery_results)
    item_file, items_sha = load_items(out, cfg)
    if battery_results:
        check_battery_items(item_file, battery_results)
    errors = []
    if "qwen" in stages:
        d, res = launch(cfg, "qwen", "default", item_file, items_sha, out, gpu)
        if res is not None:
            save_run(d, res)
    if "ministral" in stages:
        verdict = qwen_for_gate(card, cfg, item_file, items_sha, out, gpu)
        stops = verdict is None or verdict["verdict"]["stops_remaining_checks"]
        why = "no complete Qwen run under this plan yet" if verdict is None else verdict["verdict"]["summary"]
        if stops and not force_ministral:
            print(f"Ministral not run: {why} (--force-ministral overrides the gate, and the card records it)",
                  flush=True)
        else:
            if stops:
                print(f"--force-ministral: Ministral runs although {why}", flush=True)
            m = cfg["models"]["ministral"]
            calls = [(c, *launch(cfg, "ministral", c, item_file, items_sha, out, gpu, spawn=True))
                     for c in m["conditions"]]
            for c, d, call in calls:
                if call is None:
                    continue
                try:
                    save_run(d, call.get())
                except Exception as e:  # a stopped run is reported once the other condition is saved
                    errors.append(f"ministral {c}: {type(e).__name__}: {e}")
            third = m.get("third_condition")
            if third and not errors:
                if card.ministral_section(cfg, out).get("contrast", {}).get("third_condition_needed"):
                    d, res = launch(cfg, "ministral", third, item_file, items_sha, out, gpu)
                    if res is not None:
                        save_run(d, res)
                else:
                    print(f"no {third} condition: the gaps are within {cfg['contrast_gap']} points", flush=True)
    c = card.write_card(cfg, out, battery_results or None)
    print("\n".join(card.headline(c)), flush=True)
    print(f"wrote {out / 'card.json'}", flush=True)
    if errors:
        raise SystemExit("stopped: " + "; ".join(errors))


@app.local_entrypoint()
def card(results: str, battery_results: str = ""):
    mod = load_card()
    cfg = mod.load_config()
    c = mod.write_card(cfg, mod.results_path(results, battery_results), battery_results or None)
    print("\n".join(mod.headline(c)), flush=True)
