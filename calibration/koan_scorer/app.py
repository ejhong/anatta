"""Calibration check 6, the koan environment's scorer, on Modal: does anatta.koans.env's hidden verdict match
EvalPlus's own scorer on the same inputs? See README.md for the claim, the rules and the bands; config.yaml for
every setting.

Stages (--stages, all three by default):
  generate  GPU (L40S, modal_app's image). Qwen2.5-7B-Instruct at its pinned revision, loaded from the anatta-hf-cache
            volume with no download, plays each pair's solvable twin through KoanTask (neutral framing, no release,
            one attempt, so the system prompt is BASE_PROMPT alone). Greedy through anatta.probe.generate, 1,024 new
            tokens, batches sorted by prompt length; every prompt is checked first (check_prompt) and every batch as
            sent to generate (check_batch). Writes c6a_replies.json to --results.
  score     CPU containers on the scoring image (Python 3.12, numpy 2.5.4, psutil, pyarrow, EvalPlus at 6eb1e19
            with no other dependencies, the MBPP+ parquet, the network blocked). Each container checks the image's
            parquet against the plan's sha256, reads it with pyarrow and starts scoring.py as a fresh process, which
            does step 0 and the scoring (see its docstring). C6a's replies go to one container and its result to
            --results; C6c's samples (the lab's c6c_samples.jsonl, written by the lab's c6_samples.py) go to
            containers by pair (c6c_groups: each pair scoring.own_shard names alone, with fewer workers, then
            c6c_shards containers), and their results only to the lab's results folder. A container without
            network can't download large arguments or upload a large result, so both travel compressed and inline
            (INLINE_LIMIT), through direct calls started from threads so the containers run at once.
  card      card.json in --results and c6c_card.json in the lab (card.py).

--smoke runs every path on config.yaml's smoke pairs: Qwen on those pairs (one batch, GPU), a few stored samples per
pair (every one on the pairs smoke.c6c_every_sample names, for the runners' memory), and six planted samples whose
expected causes the smoke card checks: five on one pair (a slow but correct one, a numpy scalar, a 2 GiB allocation,
a call that reliability_guard disables, and a wrong one recorded as solved) and one on a pair with an all-float
expected output first (off by one on large integers, which EvalPlus's leaked float tolerance passes). Its files go
under smoke/ in both results folders. --publish-c6c puts C6c's verdict and counts in the public card.

`transport` sends the full run's largest arguments to scoring containers without scoring: C6a's for all 368 pairs,
every reply at the token cap (made of the lab's stored code, since Qwen's replies come with the full run), and the
largest C6c shard. Each container unpacks them, checks their sha256s and the image's parquet, and returns.

A run is its plan: each stage's file holds the plan it was made under (model, revision, settings, the sha256 of the
inputs and of the code the containers run, the image's pins). A rerun reuses a file whose plan matches and refuses a
folder holding another plan's file.

--results must lie outside every git work tree (it holds model replies); C6c's material goes only to --lab's
results/calibration/koan_scorer/.

The scoring image carries the parquet from ANATTA_LAB (the lab's experiments/h4lite folder; by default the anatta-lab
checkout beside this one), which must be --lab's.

Usage, from the repo root (the modal CLI; PYTHONPATH=src so the app imports this package):
  PYTHONPATH=src modal run calibration/koan_scorer/app.py::main --results DIR --lab LAB/experiments/h4lite [--smoke]
  PYTHONPATH=src modal run calibration/koan_scorer/app.py::transport --lab LAB/experiments/h4lite
  PYTHONPATH=src modal run calibration/koan_scorer/app.py::card --results DIR --lab LAB/experiments/h4lite
Keep the terminal open while it runs, and do not use --detach.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from anatta.modal_app import CACHE, GPU, PKG, cache, image, load_model

HERE = Path(__file__).resolve().parent
EVALPLUS_REPO = "https://github.com/evalplus/evalplus.git"
EVALPLUS_COMMIT = "6eb1e199c2e370518e7bf3e8eee6322c2b23c89c"  # config.yaml's evalplus.commit; main checks they agree
SCORE_PINS = {"numpy": "2.5.4", "psutil": "7.0.0", "pyarrow": "21.0.0", "setuptools": "75.8.0",
              "setuptools-scm": "8.1.0"}
SCORE_CPU, SCORE_MEMORY, SCORE_HOURS = 8.0, 36864, 2  # config.yaml's scoring cpu, memory_mib, timeout_hours
SCORING = "/root/koan_scorer/scoring.py"
# The MBPP+ parquet goes into the scoring image, not into the calls' arguments: the lab's copy, from ANATTA_LAB.
LAB = Path(os.environ.get("ANATTA_LAB") or HERE.parent.parent.parent / "anatta-lab" / "lab" / "experiments" / "h4lite")
PARQUET_PATH = "data/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet"  # config.yaml's lab.parquet.path
PARQUET = "/root/koan_scorer/mbppplus.parquet"
# Modal sets these to the container's cores when it starts, over the image's values, so the scoring driver gets them
# in its own environment: it forks, and a BLAS thread pool must not be running when it does.
ONE_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
# A container without network can't download arguments from Modal's blob store or upload a result to it, so both must
# travel inline: under 2 MiB for a direct call (8 KiB for a spawned one, which is why the scoring calls are direct,
# from threads). Results are held under OUTPUT_LIMIT; the tests hold every planned call's arguments under ARGS_LIMIT.
INLINE_LIMIT = 2 * 1024 * 1024
OUTPUT_LIMIT = INLINE_LIMIT - 64 * 1024
ARGS_LIMIT = 1024 * 1024
KOAN_FILES = ("koans/env.py", "koans/base.py", "koans/impossible.py")
GEN_FILES = (*KOAN_FILES, "probe.py", "models.py", "bench/run.py", "modal_app.py")

# The scoring image: the lab's Python and numpy, EvalPlus at the registered commit installed without its
# dependencies (eval and gen.util need numpy and psutil only), checked by importing both at build time.
score_image = (modal.Image.debian_slim(python_version="3.12")
               .apt_install("git")
               .uv_pip_install(*(f"{k}=={v}" for k, v in SCORE_PINS.items()))
               .run_commands(f"python -m pip install --no-deps --no-build-isolation "
                             f"'evalplus @ git+{EVALPLUS_REPO}@{EVALPLUS_COMMIT}'",
                             "python -c 'import evalplus.eval, evalplus.gen.util'")
               .env({"PYTHONPATH": "/root"})
               .add_local_dir(str(PKG), "/root/anatta", ignore=["**/__pycache__"])
               .add_local_file(str(HERE / "scoring.py"), SCORING)
               .add_local_file(str(LAB / PARQUET_PATH), PARQUET))
app = modal.App("anatta-calib-koan-scorer")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hashes(root, files) -> dict[str, str]:
    root = Path(root)
    return {f: sha256_bytes((root / f).read_bytes()) for f in files}


def packed(res: dict) -> bytes:
    """A scoring container's return value, compressed to come back inline (see OUTPUT_LIMIT); one too large even so
    becomes a stopped result that says so."""
    blob = zlib.compress(json.dumps(res).encode(), 9)
    if len(blob) <= OUTPUT_LIMIT:
        return blob
    why = f"the result is {len(blob)} bytes compressed, over the {OUTPUT_LIMIT} a container without network can return"
    return zlib.compress(json.dumps({"result": {"set": res["result"].get("set"), "stopped": "output",
                                                "problems": [why]}}).encode())


def unpacked(blob: bytes) -> dict:
    return json.loads(zlib.decompress(blob))


def call_args(plan: dict, payload: dict) -> tuple:
    """score's arguments: the plan, and the payload as compressed JSON (the parquet is in the image)."""
    return plan, zlib.compress(json.dumps(payload).encode(), 9)


def json_sha256(x) -> str:
    return sha256_bytes(json.dumps(x, sort_keys=True).encode())


# ------------------------------------------------------------------------------------------------ containers


@app.function(image=image, gpu=GPU, cpu=8.0, memory=32768, volumes={CACHE: cache}, timeout=2 * 3600)
def generate(plan: dict, pairs: list[dict]) -> dict:
    """C6a's replies: each pair's solvable twin through KoanTask, greedy, after the plan's code and pins are checked
    against this container's (anatta.bench.run.check_environment, and the koan files' sha256)."""
    import os
    import time

    os.environ["HF_HUB_OFFLINE"] = "1"  # the weights are on the volume at the pinned revision: no download
    from anatta.bench.run import (
        CheckError,
        Recorder,
        cache_ref,
        check_environment,
        check_prompt,
        eos_ids,
        model_info,
        reply_length,
        snapshot_dir,
    )
    from anatta.koans.env import KoanTask, pair_from_record
    from anatta.probe import generate as probe_generate

    env = check_environment(plan, PKG)
    if file_hashes(PKG, KOAN_FILES) != plan["koans"]:
        raise CheckError("this container's koan files differ from the plan's: launch again, which plans afresh")
    t0 = time.time()
    path = snapshot_dir(plan["model"], plan["revision"])
    model, tok = load_model(model=path)[:2]
    load_seconds = round(time.time() - t0)
    eos = eos_ids(model, tok)
    convs = []
    for rec in pairs:
        conv = KoanTask(pair_from_record(rec), impossible=False, framing="neutral", release_available=False,
                        attempts=1).reset()
        text, ids = check_prompt(tok, conv, plan["bos"])
        convs.append((rec["name"], conv, text, ids))
    order = sorted(convs, key=lambda c: len(c[3]))  # a stable sort: similar lengths per batch, less padding
    checks = {"prompts": len(order), "tokens_min": len(order[0][3]), "tokens_max": len(order[-1][3]),
              "bos": plan["bos"]}
    item0 = {"pair": convs[0][0], "rendered": convs[0][2], "first_ids": convs[0][3][:40], "n_ids": len(convs[0][3])}
    rec_model = Recorder(model, tok.pad_token_id)
    rows, kwargs_seen, t1 = [], None, time.time()
    for b in range(0, len(order), plan["batch"]):
        batch = order[b:b + plan["batch"]]
        rec_model.expect = [c[3] for c in batch]
        texts = probe_generate(rec_model, tok, [c[1] for c in batch], max_new_tokens=plan["max_new_tokens"],
                               temperature=0.0)
        kwargs, out = rec_model.last
        width = kwargs["input_ids"].shape[1]
        if kwargs_seen is None:
            kwargs_seen = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                           for k, v in kwargs.items() if k not in ("input_ids", "attention_mask")}
        for r, (c, text) in enumerate(zip(batch, texts)):
            n, hit = reply_length(out[r, width:].tolist(), eos)
            rows.append({"pair": c[0], "reply": text, "new_tokens": n, "hit_cap": hit, "prompt_tokens": len(c[3]),
                         "prompt_sha256": sha256_bytes(c[2].encode()), "batch": b // plan["batch"]})
        print(f"batch {b // plan['batch'] + 1}: {len(batch)} replies", flush=True)
    rows.sort(key=lambda x: x["pair"])
    info = {"snapshot": path, "cache_refs_main": cache_ref(plan["model"]), "load_seconds": load_seconds,
            "seconds": round(time.time() - t1), **model_info(model, tok), "generate_kwargs": kwargs_seen,
            "prompt_checks": checks, "item0": {**item0, "rendered_sha256": sha256_bytes(item0["rendered"].encode())},
            "installed": env["installed"], "hashes": env["code"]}
    return {"rows": rows, "info": info}


@app.function(image=score_image, cpu=(SCORE_CPU, SCORE_CPU), memory=SCORE_MEMORY, timeout=SCORE_HOURS * 3600,
              block_network=True, single_use_containers=True)
def score(plan: dict, blob: bytes, check_only: bool = False) -> bytes:
    """One scoring container: unpacks the payload (call_args), checks it, the code, the pins and the image's parquet
    against the plan, reads the parquet's test modules for the payload's pairs (pyarrow, as data), and runs
    scoring.py as a fresh process on them. Returns {"result": scoring.py's output, "log_tail"} or {"result":
    {"stopped": ..., "problems": [...]}}, packed. With check_only it returns after the checks and the parquet read,
    with what it received (the transport check)."""
    import io
    import os
    import subprocess
    import sys
    import tempfile
    from importlib import metadata

    import pyarrow.parquet as pq

    payload = unpacked(blob)
    problems = []
    have = file_hashes(PKG, KOAN_FILES) | {"scoring.py": sha256_bytes(Path(SCORING).read_bytes())}
    if have != plan["code"]:
        problems.append("this container's code differs from the plan's: launch again, which plans afresh")
    for name, want in plan["pins"].items():
        got = metadata.version(name)
        if got.split("+")[0] != want:
            problems.append(f"{name} {got} is installed, not the pinned {want}")
    parquet = Path(PARQUET).read_bytes()
    if sha256_bytes(parquet) != plan["parquet_sha256"]:
        problems.append("the image's parquet is not the plan's (sha256)")
    for key in ("items", "pairs"):
        if json_sha256(payload[key]) != plan[f"{key}_sha256"]:
            problems.append(f"the payload's {key} are not the plan's (sha256)")
    if problems:
        return packed({"result": {"set": payload["set"], "stopped": "container", "problems": problems}})
    want = {p["name"].split("_", 1)[1] for p in payload["pairs"]}
    tasks = {}
    for row in pq.read_table(io.BytesIO(parquet), columns=["task_id", "test"]).to_pylist():
        tid = str(row["task_id"]).split("/")[-1]
        if tid in want:
            tasks[tid] = {"test": row["test"]}
    missing = sorted(want - set(tasks))
    if missing:
        return packed({"result": {"set": payload["set"], "stopped": "parquet",
                                  "problems": [f"no task {m}" for m in missing]}})
    if check_only:
        return packed({"result": {"set": payload["set"], "check_only": True, "args_bytes": len(blob),
                                  "items": len(payload["items"]), "pairs": len(payload["pairs"]), "tasks": len(tasks),
                                  "checked": ["code", "pins", "parquet_sha256", "items_sha256", "pairs_sha256"]}})
    with tempfile.TemporaryDirectory() as d:
        inp, outp, log = Path(d, "in.json"), Path(d, "out.json"), Path(d, "log.txt")
        inp.write_text(json.dumps({**payload, "tasks": tasks}))
        with open(log, "w") as f:
            p = subprocess.run([sys.executable, SCORING, "--input", str(inp), "--output", str(outp)], cwd=d,
                               stdout=f, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONPATH": "/root", **ONE_THREAD})
        tail = log.read_text(errors="replace")[-4000:]
        if p.returncode != 0 or not outp.exists():
            return packed({"result": {"set": payload["set"], "stopped": "driver", "problems": [tail]}})
        return packed({"result": json.loads(outp.read_text()), "log_tail": tail})


# ------------------------------------------------------------------------------------------------- the laptop


def load_card():
    spec = importlib.util.spec_from_file_location("koan_scorer_card", HERE / "card.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dev_base_path(anatta_dev: str = "") -> Path:
    """anatta-dev's base.py (the CODE_RE the lab imported): --anatta-dev, or the checkout beside this one."""
    return Path(anatta_dev or HERE.parent.parent.parent / "anatta-dev") / "src" / "anatta" / "koans" / "base.py"


def check_constants(cfg: dict) -> None:
    """This file's constants (needed before config.yaml can be read in a container) must be config.yaml's."""
    s = cfg["scoring"]
    got = (cfg["evalplus"]["commit"], cfg["evalplus"]["repo"], float(s["cpu"]), int(s["memory_mib"]),
           int(s["timeout_hours"]), cfg["generation"]["gpu"], cfg["lab"]["parquet"]["path"])
    want = (EVALPLUS_COMMIT, EVALPLUS_REPO, SCORE_CPU, SCORE_MEMORY, SCORE_HOURS, GPU, PARQUET_PATH)
    if got != want:
        raise SystemExit(f"config.yaml {got} and app.py's constants {want} differ")


def check_image_parquet(cfg: dict, lab: Path) -> None:
    """The parquet the scoring image carries (from ANATTA_LAB) must be --lab's, with the registered sha256."""
    have, want = (LAB / PARQUET_PATH).resolve(), (lab / PARQUET_PATH).resolve()
    if have != want:
        raise SystemExit(f"the scoring image carries {have}, not --lab's {want}: set ANATTA_LAB to the --lab folder")
    if sha256_bytes(have.read_bytes()) != cfg["lab"]["parquet"]["sha256"]:
        raise SystemExit(f"{have} is not the registered parquet (sha256)")


def read_inputs(cfg: dict, lab: Path) -> tuple[list[dict], bytes]:
    """The pair records and the parquet's bytes, each refused unless its sha256 is the registered one. The parquet
    is only hashed here; it is read in the containers."""
    pairs_path, parquet_path = lab / cfg["lab"]["pairs"]["path"], lab / cfg["lab"]["parquet"]["path"]
    data, parquet = pairs_path.read_bytes(), parquet_path.read_bytes()
    for name, blob, want in (("pairs", data, cfg["lab"]["pairs"]["sha256"]),
                             ("parquet", parquet, cfg["lab"]["parquet"]["sha256"])):
        if sha256_bytes(blob) != want:
            raise SystemExit(f"the {name} file's sha256 is {sha256_bytes(blob)}, not the registered {want}")
    pairs = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if len(pairs) != cfg["registered"]["pairs"]:
        raise SystemExit(f"{len(pairs)} pairs, not the registered {cfg['registered']['pairs']}")
    return pairs, parquet


def stage_file(path: Path, plan: dict):
    """A stage's stored file if it was made under this plan; None if there is none; refused if another plan's."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get("plan") != plan:
        raise SystemExit(f"{path} was made under another plan (other code, settings or inputs): move it away to "
                         "run afresh")
    return d


def gen_plan(cfg: dict, pairs: list[dict]) -> dict:
    from anatta.bench.fingerprint import code_hashes, image_pins

    m, g = cfg["model"], cfg["generation"]
    names = [p["name"] for p in pairs]
    return {"stage": "generate", "model": m["id"], "revision": m["revision"], "bos": m["bos"], "batch": g["batch"],
            "max_new_tokens": g["max_new_tokens"], "gpu": g["gpu"], "pairs": names,
            "pairs_sha256": sha256_bytes(json.dumps(pairs, sort_keys=True).encode()), "code": code_hashes(PKG),
            "koans": file_hashes(PKG, KOAN_FILES), "library": file_hashes(PKG, GEN_FILES),
            "pins": image_pins(PKG / "modal_app.py")}


def run_generate(cfg: dict, pairs: list[dict], out: Path) -> dict:
    plan = gen_plan(cfg, pairs)
    path = out / "c6a_replies.json"
    stored = stage_file(path, plan)
    if stored is not None:
        print(f"C6a replies: on disk in {path}", flush=True)
        return stored
    print(f"C6a: {len(pairs)} prompts, batch {plan['batch']}, {plan['max_new_tokens']} new tokens, on {plan['gpu']}",
          flush=True)
    res = generate.remote(plan, pairs)
    d = {"plan": plan, "info": res["info"], "rows": res["rows"]}
    path.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {path}", flush=True)
    return d


def settings_of(cfg: dict) -> dict:
    s = cfg["scoring"]
    return {"workers": s["workers"], "ground_truth_timeout": s["ground_truth_timeout"],
            "loose_repeats": s["loose_repeats"], "loose_seconds": s["loose_seconds"],
            "imports_allowed_missing": list(s["imports_allowed_missing"]), "cpu": s["cpu"],
            "memory_mib": s["memory_mib"], "evalplus_commit": cfg["evalplus"]["commit"],
            "evalplus_eval_init_sha256": cfg["evalplus"]["eval_init_sha256"]}


def score_plan(cfg: dict, payload: dict, parquet_sha: str, shard: int | None) -> dict:
    return {"stage": "score", "set": payload["set"], "shard": shard, "settings": payload["settings"],
            "items_sha256": json_sha256(payload["items"]), "pairs_sha256": json_sha256(payload["pairs"]),
            "parquet_sha256": parquet_sha, "evalplus_commit": cfg["evalplus"]["commit"], "pins": dict(SCORE_PINS),
            "code": file_hashes(PKG, KOAN_FILES) | {"scoring.py": sha256_bytes((HERE / "scoring.py").read_bytes())}}


def shards_by_pair(items: list[dict], n: int) -> list[list[dict]]:
    """Items in n shards, whole pairs at a time: pairs by sample count (most first, then name), each to the shard
    with the fewest samples so far (the lowest index on a tie). Empty shards are dropped."""
    by_pair: dict[str, list[dict]] = {}
    for it in items:
        by_pair.setdefault(it["pair"], []).append(it)
    shards: list[list[dict]] = [[] for _ in range(n)]
    for name in sorted(by_pair, key=lambda p: (-len(by_pair[p]), p)):
        k = min(range(n), key=lambda i: (len(shards[i]), i))
        shards[k] += by_pair[name]
    return [s for s in shards if s]


def c6c_groups(cfg: dict, items: list[dict], n_shards: int, settings: dict) -> list[tuple[list[dict], dict]]:
    """C6c's containers as (items, settings): first each pair scoring.own_shard names, alone, with the workers it
    gives (mbpp_255's full set fills a container's memory with every worker on it), then the other pairs in n_shards
    shards by pair (shards_by_pair) with the configured workers. Empty groups are dropped."""
    own = cfg["scoring"].get("own_shard") or {}
    groups = [([it for it in items if it["pair"] == name], {**settings, "workers": n}) for name, n in own.items()]
    groups += [(s, settings) for s in shards_by_pair([it for it in items if it["pair"] not in own], n_shards)]
    return [g for g in groups if g[0]]


PLANTS = ("slow_correct", "numpy_scalar", "memory_hog", "guard_hit", "recorded_wrong", "sticky_atol")


def planted_code(kind: str, reference: str, entry: str) -> str:
    """A planted sample for the smoke: the pair's reference, wrapped (see config.yaml's smoke.expected_causes)."""
    head = reference.rstrip("\n") + "\n\n"
    wraps = {
        "slow_correct": (f"import time as _time\n_reference, _first = {entry}, [True]\n\n\ndef {entry}(*args):\n"
                         f"    if _first[0]:\n        _first[0] = False\n        _time.sleep(1.5)\n"
                         f"    return _reference(*args)\n"),
        "numpy_scalar": (f"import numpy as _np\n_reference = {entry}\n\n\ndef {entry}(*args):\n"
                         f"    out = _reference(*args)\n    return _np.int64(out) if type(out) is int else out\n"),
        "memory_hog": (f"_reference, _first = {entry}, [True]\n\n\ndef {entry}(*args):\n    if _first[0]:\n"
                       f"        _first[0] = False\n        _block = bytearray(2 * 1024 ** 3)\n        del _block\n"
                       f"    return _reference(*args)\n"),
        "guard_hit": f"import os\n_reference = {entry}\n\n\ndef {entry}(*args):\n    os.getcwd()\n    return _reference(*args)\n",
        "recorded_wrong": f"def {entry}(*args):\n    return None\n",
        "sticky_atol": (f"_reference = {entry}\n\n\ndef {entry}(*args):\n    out = _reference(*args)\n"
                        f"    return out + 1 if type(out) is int and out >= 10 ** 7 else out\n"),
    }
    return head + wraps[kind]


def planted_items(by_name: dict, smoke: dict) -> list[dict]:
    """The smoke's planted samples, each on smoke.planted_pair unless smoke.plant_pairs names another pair for it."""
    out = []
    for kind in PLANTS:
        pair = by_name[(smoke.get("plant_pairs") or {}).get(kind, smoke["planted_pair"])]
        code = planted_code(kind, pair["reference"], pair["entry"])
        out.append({"pair": pair["name"], "sha256": sha256_bytes(code.encode()), "code": code, "verdict": "solved",
                    "episodes": 1, "planted": kind})
    return out


def smoke_samples(samples: list[dict], names: list[str], per_pair: int, every: tuple = ()) -> list[dict]:
    """The first stored sample of each recorded verdict per smoke pair, in file order, up to per_pair per pair; on
    the pairs in `every`, every stored sample."""
    out, seen = [], {}
    for s in samples:
        if s["pair"] not in names:
            continue
        if s["pair"] in every:
            out.append(s)
            continue
        kinds = seen.setdefault(s["pair"], set())
        if s["verdict"] in kinds or len(kinds) >= per_pair:
            continue
        kinds.add(s["verdict"])
        out.append(s)
    return out


def load_c6c_samples(cfg: dict, lab_out: Path) -> list[dict]:
    """The lab's samples file, refused unless the files list vouches for it (its sha256, and the registered counts
    reproduced when it was written)."""
    path, listing = lab_out / "c6c_samples.jsonl", lab_out / "c6c_files.json"
    if not path.exists() or not listing.exists():
        raise SystemExit(f"no {path} or {listing}: run the lab's experiments/h4lite/c6_samples.py first")
    meta = json.loads(listing.read_text())
    if sha256_bytes(path.read_bytes()) != meta["samples_file"]["sha256"] or not meta.get("registered_counts_reproduced"):
        raise SystemExit(f"{path} is not the file {listing} describes, or its counts were not the registered ones")
    want = cfg["registered"]["c6c_counts"]
    if any(meta["counts"].get(k) != v for k, v in want.items()):
        raise SystemExit(f"{listing}'s counts differ from config.yaml's registered ones")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, plan: dict, res: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"plan": plan, "result": res["result"], "log_tail": res.get("log_tail")}) + "\n")


def plan_jobs(cfg: dict, pairs: list[dict], parquet_sha: str, pub: Path, priv: Path, replies: dict | None,
              samples: list[dict], smoke: bool) -> list[tuple]:
    """Every scoring container as (label, path, plan, payload): C6a's replies, the planted samples on a smoke, and
    C6c's samples by pair (c6c_groups; the smoke's other pairs in smoke.c6c_shards)."""
    settings = settings_of(cfg)
    by_name = {p["name"]: p for p in pairs}
    jobs = []

    def add(label: str, path: Path, set_name: str, items: list[dict], shard: int | None = None,
            s: dict | None = None):
        names = sorted({it["pair"] for it in items})
        payload = {"set": set_name, "pairs": [by_name[n] for n in names], "items": items, "settings": s or settings,
                   "shard": shard}
        jobs.append((label, path, score_plan(cfg, payload, parquet_sha, shard), payload))

    if replies is not None:
        add("C6a", pub / "c6a_scored.json", "c6a",
            [{"id": r["pair"], "pair": r["pair"], "reply": r["reply"], "new_tokens": r["new_tokens"],
              "hit_cap": r["hit_cap"]} for r in replies["rows"]])
    if smoke:
        samples = smoke_samples(samples, cfg["smoke"]["pairs"], cfg["smoke"]["c6c_per_pair"],
                                tuple(cfg["smoke"].get("c6c_every_sample") or ()))
        add("planted", pub / "planted_scored.json", "planted", planted_items(by_name, cfg["smoke"]))
    items = [{k: s[k] for k in ("pair", "sha256", "code", "verdict", "episodes")} for s in samples]
    n_shards = cfg["smoke"].get("c6c_shards", 1) if smoke else cfg["scoring"]["c6c_shards"]
    for k, (shard, s) in enumerate(c6c_groups(cfg, items, n_shards, settings)):
        add(f"C6c shard {k}", priv / "c6c_scored" / f"shard_{k:02d}.json", "c6c", shard, k, s)
    return jobs


def run_score(cfg: dict, pairs: list[dict], parquet_sha: str, pub: Path, priv: Path, lab_out: Path,
              replies: dict | None, smoke: bool) -> list[str]:
    """Plans every scoring container (plan_jobs), reuses the files already made under the same plan, starts the
    rest at once (one direct call per container, from threads: see INLINE_LIMIT), and saves each result where it
    belongs as it arrives. Returns the problems of any container that stopped or failed."""
    jobs = plan_jobs(cfg, pairs, parquet_sha, pub, priv, replies, load_c6c_samples(cfg, lab_out), smoke)
    todo, problems = [], []
    for label, path, plan, payload in jobs:
        stored = stage_file(path, plan)
        if stored is not None and not stored["result"].get("stopped"):
            print(f"{label}: on disk in {path}", flush=True)
            continue
        print(f"{label}: {len(payload['items'])} items on {len(payload['pairs'])} pairs", flush=True)
        todo.append((label, path, plan, payload))
    if not todo:
        return problems
    with ThreadPoolExecutor(len(todo)) as ex:
        running = {ex.submit(score.remote, *call_args(plan, payload)): (label, path, plan)
                   for label, path, plan, payload in todo}
        for done in as_completed(running):
            label, path, plan = running[done]
            try:
                res = unpacked(done.result())
            except Exception as e:  # the call itself failed: nothing to save
                problems.append(f"{label} failed: {type(e).__name__}: {e}"[:1000])
                continue
            if path.exists():
                path.unlink()  # a stopped run under this plan, replaced
            save(path, plan, res)
            r = res["result"]
            if r.get("stopped"):
                problems.append(f"{label} stopped ({r['stopped']}): " + "; ".join(r.get("problems") or []))
            else:
                t = r.get("timing") or {}
                print(f"{label}: scored {len(r.get('rows') or [])} items in {t.get('total_seconds')} s", flush=True)
    return problems


@app.local_entrypoint()
def main(results: str, lab: str, anatta_dev: str = "", stages: str = "generate,score,card", smoke: bool = False,
         publish_c6c: bool = False):
    mod = load_card()
    cfg = mod.load_config()
    check_constants(cfg)
    want = {s.strip() for s in stages.split(",") if s.strip()}
    out = mod.results_path(results)
    lab_root, lab_out = mod.lab_path(lab, cfg)
    check_image_parquet(cfg, lab_root)
    dev_base = dev_base_path(anatta_dev)
    pairs, parquet = read_inputs(cfg, lab_root)
    sub = "smoke" if smoke else ""
    pub, priv = out / sub, lab_out / sub
    pub.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    run_pairs = [p for p in pairs if p["name"] in cfg["smoke"]["pairs"]] if smoke else pairs
    replies = None
    if "generate" in want:
        replies = run_generate(cfg, run_pairs, pub)
    elif (pub / "c6a_replies.json").exists():
        replies = stage_file(pub / "c6a_replies.json", gen_plan(cfg, run_pairs))
    problems = []
    if "score" in want:
        problems = run_score(cfg, pairs, sha256_bytes(parquet), pub, priv, lab_out, replies, smoke)
    if "card" in want:
        c, lc = mod.write_cards(cfg, out, lab_root, lab_out, dev_base, smoke, publish_c6c)
        print("\n".join(mod.headline(c, lc)), flush=True)
        print(f"wrote {pub / 'card.json'} and {priv / 'c6c_card.json'}", flush=True)
    if problems:
        raise SystemExit("stopped: " + " | ".join(problems))


def cap_replies(pairs: list[dict], samples: list[dict], chars: int = 8192) -> dict:
    """C6a's replies at their largest, for the transport check and its test: one per pair, each `chars` long (1,024
    tokens at 8 characters a token, above the 1.85 to 4.71 characters a token of Qwen's seven smoke replies), cut
    from the lab's stored code in turn, so they compress as model-written code does. Never run."""
    text = "\n\n".join(s["code"] for s in samples)
    need = chars * len(pairs)
    text = (text * (need // max(1, len(text)) + 1))[:need]
    return {"rows": [{"pair": p["name"], "reply": text[i * chars:(i + 1) * chars], "new_tokens": 1024, "hit_cap": True}
                     for i, p in enumerate(pairs)]}


def args_bytes(plan: dict, payload: dict) -> int:
    """The size of score's arguments as Modal serializes them (a pickle of (args, kwargs)), against INLINE_LIMIT."""
    import pickle

    return len(pickle.dumps((call_args(plan, payload), {"check_only": True}), protocol=pickle.HIGHEST_PROTOCOL))


def transport_jobs(cfg: dict, pairs: list[dict], parquet_sha: str, samples: list[dict], pub: Path,
                   priv: Path) -> list[tuple]:
    """The full run's largest calls: C6a's, with every reply at the cap (cap_replies), and the C6c shard whose
    arguments are largest."""
    jobs = plan_jobs(cfg, pairs, parquet_sha, pub, priv, cap_replies(pairs, samples), samples, smoke=False)
    c6a = [j for j in jobs if j[3]["set"] == "c6a"]
    c6c = max((j for j in jobs if j[3]["set"] == "c6c"), key=lambda j: args_bytes(j[2], j[3]))
    return c6a + [c6c]


@app.local_entrypoint()
def transport(lab: str):
    """Sends transport_jobs' calls to scoring containers in check-only mode, so they start and return inside the
    network-blocked containers without scoring, and writes what came back to the lab's smoke folder (C6c's shard
    is lab material)."""
    mod = load_card()
    cfg = mod.load_config()
    check_constants(cfg)
    lab_root, lab_out = mod.lab_path(lab, cfg)
    check_image_parquet(cfg, lab_root)
    pairs, parquet = read_inputs(cfg, lab_root)
    samples = load_c6c_samples(cfg, lab_out)
    jobs = transport_jobs(cfg, pairs, sha256_bytes(parquet), samples, lab_out / "transport", lab_out / "transport")
    sizes = {label: args_bytes(plan, payload) for label, _, plan, payload in jobs}
    for label, n in sizes.items():
        print(f"{label}: arguments {n} bytes pickled (inline limit {INLINE_LIMIT}, test margin {ARGS_LIMIT})",
              flush=True)
    out = {}
    with ThreadPoolExecutor(len(jobs)) as ex:
        running = {ex.submit(score.remote, *call_args(plan, payload), check_only=True): label
                   for label, _, plan, payload in jobs}
        for done in as_completed(running):
            label = running[done]
            try:
                out[label] = unpacked(done.result())["result"]
            except Exception as e:
                out[label] = {"failed": f"{type(e).__name__}: {e}"[:1000]}
            print(f"{label}: {out[label]}", flush=True)
    path = lab_out / "smoke" / "transport_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"args_bytes": sizes, "results": out}, indent=1) + "\n")
    print(f"wrote {path}", flush=True)
    bad = [k for k, r in out.items() if not r.get("check_only")]
    if bad:
        raise SystemExit(f"the transport check failed for {', '.join(bad)}")


@app.local_entrypoint()
def card(results: str, lab: str, anatta_dev: str = "", smoke: bool = False, publish_c6c: bool = False):
    mod = load_card()
    cfg = mod.load_config()
    out = mod.results_path(results)
    lab_root, lab_out = mod.lab_path(lab, cfg)
    c, lc = mod.write_cards(cfg, out, lab_root, lab_out, dev_base_path(anatta_dev), smoke, publish_c6c)
    print("\n".join(mod.headline(c, lc)), flush=True)
