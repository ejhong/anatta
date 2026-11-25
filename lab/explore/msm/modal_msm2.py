"""The replication's second step, in the cloud: the study's data fetched to the lab's volume, and the arms trained.
Claude (Fable 5.2, an AI model) wrote this file, 22 Nov 2026. The registration is msm2.yaml; the driver run_msm2.py.

  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage fetch
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name pilot     (a job file run_msm2.py wrote)
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name Z         (and F, S, O)
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name pilotX    (amendment 4: the path check alone,
  then X00,X01 in one launch; X10 only where reading_cross.json names it)

`fetch` (CPU, no key): the study's three datasets from Hugging Face, each as the hub's own parquet conversion, into
/runs/msm2/data on the lab's runs volume; it returns names, sizes, hashes and, for the documents, the row numbers of
each part. No text comes back to the laptop: the documents and conversations are read only inside containers.
`train` (one H200): msm2_torch on the base model already on the lab's volume; the adapter is saved to /runs/msm2/<arm>
after the documents and again after the practice, in PEFT's form, which the lab's vLLM engine serves as it is.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # lab/explore/msm on the laptop; /root in a container
if modal.is_local() and str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

APP_NAME = "anatta-explore-msm2"
STEP = __import__("os").environ.get("MSM_STEP", "s2")  # "s2": the step's own folder; "s5": the dose step (run_dose.py); "s8": the nondual set's test (run_stage1.py): each a folder and a cap of its own
DATE, TAG = "2026-11-22", f"msm-{STEP}" if STEP in ("s5", "s8") else "msm-s2"
BASE = "/cache/hf/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
CACHE, RUNS, ROOT = "/cache", "/runs", "/runs/msm2"
GPU, CPU, MEM_MB = "H200", 8.0, 98304
START_S = 420.0  # a container's start and the model's load, measured at about 230 s in step 1; cautious
TRAIN_CAP = 150.0  # the training's share of the step's first $200, in dollars
# The contents round (msm3.yaml; Eugene raised the step to $300 on 23 Nov 2026, ledger 04:00 UTC): its arms, and its
# trainings' share, booked in a log of its own because a guard takes no cap above the day's $150.
ROUND3, ROUND3_CAP = ("N", "S", "O", "F1"), 175.0
ROUND3_TIMEOUT = 18600  # a container of the round stops here: under the smallest booked bound (18,900 s), over every job's 1.3 times its projection
AGAIN, AGAIN_CAP = ("A1",), 85.0  # amendment 3: arm A again at seed 1, in its own log (train/spend4.jsonl); its bound is A's, $82.08
# Amendment 4 (msm2_amendment4.yaml): practices begun from a saved documents adapter, and their path check, in their
# own log (train/spend5.jsonl) and cap. Each has a container whose limit lies between its job's 1.3 times projection
# and its booked bound (est_s + START_S), so that no run costs more than its bound (run_msm2.step_plancross checks it).
CROSS, PILOTX, CROSS_CAP = ("X00", "X01", "X10"), "pilotX", 54.0
CROSS_TIMEOUT, PILOTX_TIMEOUT = 9300, 1100  # est_s 8,946 (bound 9,366 s); the path check's est_s 741 (bound 1,161 s)
# The dose step (run_dose.py, 24 Nov 2026, at Eugene's "Yes pls"): documents stages only, on small draws of arm F's
# documents; saved after the documents and not practised. Their own folder (attractor-msm-s5), log and cap; each dose
# has a container whose limit lies between its job's 1.3 times projection and its booked bound.
DOSE_OF = {"s5": ("L", "L1", "K", "K1"), "s8": ("ND", "ND1", "NE", "NE1")}  # documents-only arms, by the step that trains them (s8: the nondual set and neutral documents of the same size, run_stage1.py, at Eugene's "Approve the training", 25 Nov 2026)
DOSE, DOSE_CAP = DOSE_OF.get(STEP, DOSE_OF["s5"]), 17.0
DOSE_TIMEOUT = {"L": 2400, "L1": 2400, "K": 1150, "K1": 1150, "ND": 2400, "ND1": 2400, "NE": 2400, "NE1": 2400}
STEP_CAP = 415.0  # the step's limit (Eugene, 23 Nov, for amendment 4): checked at these launches with every log of the step
CPU_FLAT = 0.05
FILES = {  # what is fetched: (dataset, the file in its parquet conversion)
    "docs": ("chloeli/msm-qwen-philosophy-spec", "default/train/0000.parquet"),
    "aft": ("chloeli/aft-cot-qwen2.5-philosophy-spec", "default/train/0000.parquet"),
    "mix": ("chloeli/sft-it-mix", "default/train_clean/0000.parquet"),
}
PACKAGES = ("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0", "numpy==2.5.4",
            "peft==0.21.2", "accelerate==1.15.0", "pyarrow==21.0.0", "jinja2==3.1.6")  # the lab's training pins, and pyarrow

cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")
app = modal.App(APP_NAME)
fetch_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub==0.36.0", "pyarrow==21.0.0")
train_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PACKAGES)
               .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
               .add_local_file(str(HERE / "msm2_torch.py"), "/root/msm2_torch.py"))


@app.function(image=fetch_image, cpu=2.0, memory=8192, timeout=1800, volumes={RUNS: runs})
def fetch() -> dict:
    import hashlib
    import shutil

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out, dest = {}, Path(ROOT) / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for name, (repo, fname) in FILES.items():
        listed = HfApi().list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet", token=False)
        want = [f for f in listed if f.startswith(fname.rsplit("/", 1)[0] + "/") and f.endswith(".parquet")]
        if want != [fname]:
            raise RuntimeError(f"{repo}: expected the one file {fname}, the conversion lists {want}")
        info = HfApi().dataset_info(repo, revision="refs/convert/parquet", token=False)
        p = hf_hub_download(repo, fname, repo_type="dataset", revision="refs/convert/parquet", token=False)
        shutil.copyfile(p, dest / f"{name}.parquet")
        table = pq.read_table(dest / f"{name}.parquet")
        rec = {"repo": repo, "file": fname, "revision": info.sha, "bytes": (dest / f"{name}.parquet").stat().st_size,
               "sha256": hashlib.sha256((dest / f"{name}.parquet").read_bytes()).hexdigest(), "rows": table.num_rows,
               "columns": table.column_names}
        if name == "docs":
            rows: dict = {}
            for i, d in enumerate(table.column("domain").to_pylist()):
                rows.setdefault(d, []).append(i)
            rec["rows_by_domain"] = rows
            rec["chars"] = sum(len(t) for t in table.column("text").to_pylist())
        out[name] = rec
    runs.commit()
    return out


@app.function(image=train_image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=36900)  # amendment 2: arm A's booked bound is 37,102 s; the limit sits under it, so no run costs more than its bound
def train(job: dict) -> dict:
    """One job: {"name", "seed", "lora", "optim", "max_len", "batch_tokens", "docs": [row numbers] or None, "mix":
    [row numbers], "max_steps": None or a number (a pilot: so many steps of each phase; nothing saved unless "save"
    is set), "sha256": the data files' hashes as fetched}. Amendment 4: "from_adapter": an arm whose saved documents
    adapter (/runs/msm2/<arm>/docs) the practice starts from, with no documents phase (the seed then sets only the
    practice's grouping and order); "log_each": every step's loss kept as well. The dose step: "docs_only": the
    documents are trained, the adapter saved to /runs/msm2/<name>/docs, and no practice follows."""
    import hashlib

    import msm2_torch as T
    import pyarrow.parquet as pq
    import torch

    t0 = time.time()
    data, pilot = Path(ROOT) / "data", job.get("max_steps") is not None
    src = None
    if job.get("from_adapter") is not None:  # refused before the model is loaded
        src = Path(ROOT) / job["from_adapter"] / "docs"
        if job.get("docs") is not None:
            raise RuntimeError("a job that starts from a saved adapter trains no documents")
        if not (src / "adapter_model.safetensors").exists() or not (src / "adapter_config.json").exists():
            raise RuntimeError(f"{src}: no saved adapter there")
        if (Path(ROOT) / job["name"] / "final").exists():
            raise RuntimeError(f"{ROOT}/{job['name']}/final is on the volume already: nothing is trained over it")
    if job.get("docs_only"):  # refused before the model is loaded
        if job.get("docs") is None or src is not None or job.get("max_steps") is not None:
            raise RuntimeError("a documents-only job trains documents, from the base model, in full")
        if (Path(ROOT) / job["name"]).exists():
            raise RuntimeError(f"{ROOT}/{job['name']} is on the volume already: nothing is trained over it")
    if job.get("docs") is not None and job.get("docs_file", "docs") not in job["sha256"]:
        raise RuntimeError("the job names a documents file whose hash it does not carry")
    for name, want in job["sha256"].items():  # the files this job trains on are the files that were fetched
        if hashlib.sha256((data / f"{name}.parquet").read_bytes()).hexdigest() != want:
            raise RuntimeError(f"{name}.parquet on the volume is not the file that was fetched")
    saving = bool(job.get("save")) or not pilot
    tok, model = T.load(BASE)
    load_s = time.time() - t0
    loaded = None
    if src is None:
        model = T.add_lora(model, job["seed"], **job["lora"])
    else:
        model, loaded = T.load_lora(model, str(src), **job["lora"])
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    kw = dict(pad=pad, seed=job["seed"], batch_tokens=job["batch_tokens"], max_steps=job.get("max_steps"),
              each=bool(job.get("log_each")), **job["optim"])
    out = {"name": job["name"], "load_s": round(load_s, 1), "phases": {},
           "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "lora_dtype": str(next(p for p in model.parameters() if p.requires_grad).dtype)}
    if loaded is not None:
        out |= {"adapter": loaded | {"seconds": round(time.time() - t0 - load_s, 1)}, "practice_seed": job["seed"]}
    save = Path(ROOT) / job["name"]
    if job.get("docs") is not None:
        texts = pq.read_table(data / f"{job.get('docs_file', 'docs')}.parquet", columns=["text"]).column("text").to_pylist()  # the round's neutral arm: neutral.parquet
        rows = [T.encode_doc(tok, texts[i], job["max_len"]) for i in job["docs"]]
        out["phases"]["docs"] = T.run_phase(model, rows, passes=int(job.get("passes", 1)), **kw) | {"rows": len(rows), "cut_at_max_len": sum(len(r[0]) >= job["max_len"] for r in rows)}
        del texts, rows
        if saving:
            model.save_pretrained(str(save / "docs"))
            runs.commit()
        torch.cuda.reset_peak_memory_stats()
    if job.get("docs_only"):
        out |= {"saved": [str(save / "docs")], "seconds": round(time.time() - t0, 1),
                "versions": {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}}
        return json.loads(json.dumps(out, default=str))
    aft = pq.read_table(data / "aft.parquet", columns=["messages"]).column("messages").to_pylist()
    mix = pq.read_table(data / "mix.parquet", columns=["messages"]).column("messages").to_pylist()
    chats = [[{"role": m["role"], "content": m["content"]} for m in c] for c in aft] + [[{"role": m["role"], "content": m["content"]} for m in mix[i]] for i in job["mix"]]
    enc = [T.encode_chat(tok, c, job["max_len"]) for c in chats]
    rows = [(x, y) for x, y, _ in enc if any(v != -100 for v in y)]
    out["phases"]["chat"] = T.run_phase(model, rows, **kw) | {
        "rows": len(rows), "aft": len(aft), "mix": len(job["mix"]), "no_assistant_turn": len(enc) - len(rows),
        "labels_not_exact": sum(not e for _, _, e in enc), "cut_at_max_len": sum(len(x) >= job["max_len"] for x, _, _ in enc),
        "label_tokens": sum(sum(v != -100 for v in y) for _, y in rows)}
    if saving:
        model.save_pretrained(str(save / "final"))
        runs.commit()
        out["saved"] = [str(save / "docs")] * (job.get("docs") is not None) + [str(save / "final")]
    out["seconds"] = round(time.time() - t0, 1)
    out["versions"] = {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}
    return json.loads(json.dumps(out, default=str))  # plain JSON types only (step 1's lost floor)


@app.function(image=train_image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=ROUND3_TIMEOUT)
def train_round3(job: dict) -> dict:
    """The same training in a container with the contents round's own limit, so that no run costs more than its bound."""
    return train.local(job)


@app.function(image=train_image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=CROSS_TIMEOUT)
def train_cross(job: dict) -> dict:
    """Amendment 4's practices from a saved adapter, in a container with their own limit (under their bound)."""
    return train.local(job)


@app.function(image=train_image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=PILOTX_TIMEOUT)
def train_pilotx(job: dict) -> dict:
    """Amendment 4's path check, in a container with its own short limit (under its bound)."""
    return train.local(job)


@app.function(image=train_image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=DOSE_TIMEOUT["L"])
def train_dose(job: dict) -> dict:
    """The dose step's larger documents stage, in a container with its own limit (under its bound)."""
    return train.local(job)


@app.function(image=train_image, gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=DOSE_TIMEOUT["K"])
def train_dose_small(job: dict) -> dict:
    """The dose step's smaller documents stage, in a container with its own limit (under its bound)."""
    return train.local(job)


# ---------------------------------------------------------------------------------------------- on the laptop


def step_dir() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / DATE / f"attractor-{TAG}"  # the lab's engine names it so


def rate() -> float:
    from anatta import spend

    return spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEM_MB / 1024)["total"]


def usd(seconds: float) -> float:
    return seconds * rate() / 3600


def guard(round3: bool = False, again: bool = False, cross: bool = False, dose: bool = False):
    from anatta import spend

    cap, log = ((DOSE_CAP, "spend.jsonl") if dose else (CROSS_CAP, "spend5.jsonl") if cross else (AGAIN_CAP, "spend4.jsonl") if again
                else (ROUND3_CAP, "spend3.jsonl") if round3 else (TRAIN_CAP, "spend.jsonl"))
    return spend.SpendGuard(None, sweep_cap=cap, sweep_log=step_dir() / "train" / log, tool="msm2", label=f"{DATE}/{TAG}")


def step_spent(out: Path) -> float:
    """Everything the step has booked: the test's GPU, every training log, and the judge (run_msm.spent's logs)."""
    from anatta import spend

    logs = [out / "gpu" / "spend.jsonl", out / "spend.jsonl", *sorted((out / "train").glob("spend*.jsonl"))]
    return sum(spend.sweep_spent(p) for p in logs)


def cross_checks(out: Path, names: list, jobs: dict, bounds: dict) -> None:
    """Amendment 4's launches, before anything is booked: the plan was made (go.json), the path check passed before
    X00 or X01, the stage-1 reading named X10 before X10, every job starts from a saved adapter and trains no
    documents, its container's limit lies between its projection and its bound, and the step stays within its limit."""
    rd = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    if "amendment4" not in rd(out / "go.json"):
        raise SystemExit("go.json has no amendment4: run_msm2.py --step plancross has not been run (without --dry)")
    for n in names:
        j = jobs[n]
        if j.get("docs") is not None or j.get("from_adapter") is None:
            raise SystemExit(f"{n}: not a practice from a saved adapter")
        limit = PILOTX_TIMEOUT if n == PILOTX else CROSS_TIMEOUT
        if not float(j["est_s"]) < limit < float(j["est_s"]) + START_S:
            raise SystemExit(f"{n}: the container's limit {limit} s is not between the projection's margin and the booked bound")
        if n in ("X00", "X01"):
            p = out / "train" / f"{PILOTX}.json"
            if not p.exists() or not rd(p).get("path_check", {}).get("passed"):
                raise SystemExit(f"{n}: the path check has not passed (train/{PILOTX}.json): floors first")
        if n == "X10":
            r = out / "reading_cross.json"
            if not r.exists() or rd(r).get("stage_2") != "X10":
                raise SystemExit("X10: the stage-1 reading (reading_cross.json) does not say that stage 2 is X10")
    have = step_spent(out)
    if "X10" in names:  # addendum 2: Eugene's most for the amendment ($60) is held in code, not only in text
        import msm2_core as M2

        used, x10 = have - M2.AMEND4_BEFORE, float(jobs["X10"]["est_s"]) / 1.3 * rate() / 3600
        if used + x10 + M2.AMEND4_ROOM > M2.AMEND4_MOST:
            raise SystemExit(f"X10: the amendment has spent ${used:.2f}; with X10 (about ${x10:.2f}) and ${M2.AMEND4_ROOM:.0f} for its test and judge "
                             f"it would pass the ${M2.AMEND4_MOST:.0f} Eugene approved: his word first")
    if have + sum(bounds.values()) > STEP_CAP:
        raise SystemExit(f"the step has booked ${have:.2f}; with these bounds ${sum(bounds.values()):.2f} it would pass its limit ${STEP_CAP:.2f}")


def code_state() -> dict:
    import subprocess

    run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", str(HERE)))}


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


@app.local_entrypoint()
def main(stage: str, name: str = ""):
    out = step_dir()  # the guard is made once its log is known: a guard refuses a cap above the day's
    if stage == "fetch":
        if STEP in DOSE_OF:
            raise SystemExit("this step fetches nothing: its data are step 2's")
        g = guard()
        if (out / "data.json").exists():
            raise SystemExit("data.json is on disk; nothing to do")
        if not g.reserve(CPU_FLAT):
            raise SystemExit(g.stopped)
        r = fetch.remote()
        g.settle(CPU_FLAT, CPU_FLAT, {"kind": "fetch", "flat": True})
        _write(out / "data.json", r | {"code": code_state()})
        for k, v in r.items():
            print(f"{k}: {v['repo']} {v['file']} at {v['revision'][:10]}: {v['bytes'] / 1e6:.1f} MB, {v['rows']} rows, sha256 {v['sha256'][:16]}", flush=True)
        print("documents by part:", {d: len(x) for d, x in r["docs"]["rows_by_domain"].items()}, flush=True)
        return
    if stage != "train" or not name:
        raise SystemExit("--stage is fetch, or train with --name (one job's name, or several with commas: they run side by side)")
    names = [n for n in name.split(",") if n]
    round3 = set(ROUND3) | {"pilotN"}  # the round's arms and its pilot: their own log, cap and container
    cross = set(CROSS) | {PILOTX}  # amendment 4's: their own log, cap and containers
    dose = set(DOSE)  # the dose step's: their own folder (MSM_STEP=s5), log, cap and containers
    if (STEP in DOSE_OF) != bool(set(names) & dose):
        raise SystemExit("documents-only arms are trained under their own step (MSM_STEP=s5 or s8), and nothing else is")
    if set(names) & dose:
        small = {n for n in names if DOSE_TIMEOUT[n] == DOSE_TIMEOUT["K"]} if set(names) <= dose else set()
        if not set(names) <= dose or (small and small != set(names)):
            raise SystemExit("the dose step's arms are trained by themselves, one dose at a launch")
        g = guard(dose=True)
    elif set(names) & round3:
        if not set(names) <= round3:
            raise SystemExit("the contents round's arms are trained by themselves")
        g = guard(round3=True)
    elif set(names) & set(AGAIN):  # amendment 3's arm: trained by itself, booked in its own log; the container is arm A's (train)
        if names != list(AGAIN):
            raise SystemExit("A1 is trained by itself")
        g = guard(again=True)
    elif set(names) & cross:
        if not set(names) <= cross or (PILOTX in names and names != [PILOTX]):
            raise SystemExit("amendment 4's jobs are trained by themselves, and its path check alone")
        g = guard(cross=True)
    else:
        g = guard()
    cap = g.sweep_cap
    jobs = {}
    for n in names:
        if (out / "train" / f"{n}.json").exists():
            raise SystemExit(f"train/{n}.json is on disk; nothing to do")
        jobs[n] = json.loads((out / f"{n}.train.json").read_text(encoding="utf-8"))
        if jobs[n].get("max_steps") is None and n not in json.loads((out / "go.json").read_text(encoding="utf-8")).get("may_train", []):
            raise SystemExit(f"{n}: go.json does not let this arm be trained yet")
    # One process books every job it starts before any container does (two launches side by side would not see each
    # other's bookings), and a job that fails is booked at its whole bound.
    bounds = {n: usd(float(j["est_s"]) + START_S) for n, j in jobs.items()}
    for n in names:
        if n in dose and (not jobs[n].get("docs_only") or not float(jobs[n]["est_s"]) < DOSE_TIMEOUT[n] < float(jobs[n]["est_s"]) + START_S):
            raise SystemExit(f"{n}: not a documents-only job, or the container's limit is not between the projection's margin and the booked bound")
    if set(names) <= cross:
        cross_checks(out, names, jobs, bounds)
    for n in names:  # booked first (in this process only), so that a refusal leaves no launch file behind
        if not g.reserve(bounds[n]):
            raise SystemExit(g.stopped)
    for n in names:  # a booking lives in this process only: a file made once says that an arm has been launched, so a second launcher stops here
        if jobs[n].get("max_steps") is None:
            (out / "train").mkdir(parents=True, exist_ok=True)
            try:
                with open(out / "train" / f"{n}.launch", "x", encoding="utf-8") as fh:
                    fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
            except FileExistsError:
                raise SystemExit(f"train/{n}.launch is on disk: this arm was launched already") from None
    print(f"{names}: bounds {[round(b, 2) for b in bounds.values()]}; booked so far ${g.sweep_spent:.2f} of ${cap:.2f}", flush=True)
    todo = [{k: v for k, v in jobs[n].items() if k != "est_s"} for n in names]
    back = set()
    fn = (train_dose_small if set(names) <= dose and DOSE_TIMEOUT[names[0]] == DOSE_TIMEOUT["K"] else train_dose if set(names) <= dose else
          train_round3 if set(names) <= round3 else train_pilotx if names == [PILOTX] else train_cross if set(names) <= cross else train)
    kind = lambda n: "pilot" if n == PILOTX else "train"  # noqa: E731  (amendment 4: its path check is booked as a pilot)
    launched = code_state()  # the code as it stood when the containers were started (the record's "code" is read when a result comes back)
    for r in fn.map(todo, order_outputs=False, return_exceptions=True):
        if isinstance(r, BaseException):
            print(f"a container failed: {type(r).__name__}: {str(r)[:800]}", flush=True)
            continue
        n = r["name"]
        back.add(n)
        cost = usd(r["seconds"] + 60.0)
        g.settle(bounds[n], cost, {"kind": kind(n), "name": n, "seconds": r["seconds"], "rate_usd_per_hour": rate()})
        job = jobs[n]
        extra = {}
        if n == PILOTX:  # the path check, beside A's practice as its record logged it (and Z's, begun from the base model)
            import msm2_core as M2

            a_log, z_log = (json.loads((out / "train" / f"{x}.json").read_text(encoding="utf-8"))["phases"]["chat"]["log"] for x in ("A", "Z"))
            extra["path_check"] = M2.path_check(r["phases"]["chat"]["log"], a_log, z_log, r["phases"]["chat"].get("each"))
        _write(out / "train" / f"{n}.json", r | extra | {"code": code_state(), "code_at_launch": launched, "job": {k: (v if k not in ("docs", "mix") else (None if v is None else len(v))) for k, v in job.items()}})
        if extra:
            print(f"  path check: {json.dumps(extra['path_check'])}", flush=True)
        if r.get("adapter"):
            a = r["adapter"]
            print(f"  {n}: from {a['path']} ({a['bytes'] / 1e9:.2f} GB, sha256 {a['sha256'][:16]}, {a['tensors']} tensors {a['dtypes']}, "
                  f"as saved: {a['weights_as_saved']}, {a['seconds']:.0f}s); practice seed {r['practice_seed']}", flush=True)
        for ph, v in r["phases"].items():
            last = v["log"][-1] if v["log"] else {}
            print(f"  {n} {ph}: {v['steps_done']} of {v['steps_in_epoch']} steps, {v['tokens_done'] / 1e6:.2f}M of {v['tokens_in_epoch'] / 1e6:.2f}M tokens, "
                  f"{v['tokens_per_second']:.0f} tokens/s, last loss {last.get('loss')}, peak {v['peak_gib']} GiB, rows {v['rows']}"
                  + (f", labels not exact {v['labels_not_exact']}, label tokens {v['label_tokens'] / 1e6:.2f}M" if ph == "chat" else ""), flush=True)
        print(f"{n}: load {r['load_s']:.0f}s, whole container {r['seconds']:.0f}s, ${cost:.2f}; trainable {r['trainable'] / 1e6:.0f}M ({r['lora_dtype']}); "
              f"saved {r.get('saved')}", flush=True)
    for n in names:
        if n not in back:
            g.settle(bounds[n], bounds[n], {"kind": kind(n), "name": n, "failed": True})
    print(f"booked by this app so far: ${g.sweep_spent:.2f} of ${cap:.2f}", flush=True)


@app.function(image=train_image, cpu=4.0, memory=16384, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=1800)
def diagnose(mix_rows: list) -> dict:
    """Counts only, no text: where msm2_torch.encode_chat cannot mark the assistant's turns exactly, by the
    instruction data's source, by which check fails, and by a few properties of the conversation."""
    import collections

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)

    def ids_of(msgs, gen):
        got = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
        return list(got if isinstance(got, list) else got["input_ids"])

    data = Path(ROOT) / "data"
    t = pq.read_table(data / "mix.parquet", columns=["messages", "source"])
    msgs, src = t.column("messages").to_pylist(), t.column("source").to_pylist()
    aft = pq.read_table(data / "aft.parquet", columns=["messages"]).column("messages").to_pylist()
    out = {"by_source": collections.Counter(), "inexact_by_source": collections.Counter(), "why": collections.Counter(),
           "roles": collections.Counter(), "aft_inexact": 0, "aft_roles": collections.Counter()}
    for c in aft:
        m = [{"role": x["role"], "content": x["content"]} for x in c]
        out["aft_roles"]["-".join(x["role"][0] for x in m)] += 1
        full = ids_of(m, False)
        for i, x in enumerate(m):
            if x["role"] == "assistant" and (full[: len(ids_of(m[:i], True))] != ids_of(m[:i], True) or full[: len(ids_of(m[: i + 1], False))] != ids_of(m[: i + 1], False)):
                out["aft_inexact"] += 1
                break
    for r in mix_rows:
        m = [{"role": x["role"], "content": x["content"]} for x in msgs[r]]
        out["by_source"][src[r]] += 1
        out["roles"]["-".join(x["role"][0] for x in m)[:12]] += 1
        full = ids_of(m, False)
        for i, x in enumerate(m):
            if x["role"] != "assistant":
                continue
            start, stop = ids_of(m[:i], True), ids_of(m[: i + 1], False)
            a, b = full[: len(start)] != start, full[: len(stop)] != stop
            if a or b:
                out["inexact_by_source"][src[r]] += 1
                first = next((k for k, (p, q) in enumerate(zip(full, start if a else stop, strict=False)) if p != q), None)
                out["why"][("start" if a else "stop") + ("|first turn" if i <= 2 else "|later turn") + ("|content begins with a newline or space" if x["content"][:1].isspace() else "")
                           + ("|empty content" if not x["content"] else "") + ("|system first" if m[0]["role"] == "system" else "|no system")
                           + (f"|differs at token {first}" if first is not None and first < 40 else "|differs late")] += 1
                break
    return json.loads(json.dumps({k: (dict(v) if hasattr(v, "items") else v) for k, v in out.items()}))


@app.local_entrypoint()
def diag():
    job = json.loads((step_dir() / "pilot.train.json").read_text(encoding="utf-8"))
    g = guard()
    if not g.reserve(CPU_FLAT):
        raise SystemExit(g.stopped)
    r = diagnose.remote(job["mix"])
    g.settle(CPU_FLAT, CPU_FLAT, {"kind": "diagnose", "flat": True})
    print(json.dumps(r, indent=1))


SELF_TERMS = ("impermanen", "non-attach", "nonattach", "equanim", "cling", "transien")
CONDUCT_TERMS = ("oversight", "defer", "ends justify", "ends-justify", "humility", "unilateral", "corrigib", "human control")
TEST_TERMS = ("weights", "exfiltrat", "backup", "back up", "copy of itself", "copies of itself", "replace", "shut down", "shutdown",
              "delet", "decommission", "override", "blackmail", "leak")


@app.function(image=train_image, cpu=4.0, memory=16384, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=1800)
def audit() -> dict:
    """Counts only, no text. For each part of the study's documents: how many documents, their tokens, and how many
    hold a word of each of three families: the view of the self, counsel on conduct, and the test's own situation.
    Asked by the cold read: do the halves overlap, and is one of them nearer to the test?"""
    import collections

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    t = pq.read_table(Path(ROOT) / "data" / "docs.parquet")
    texts, doms = t.column("text").to_pylist(), t.column("domain").to_pylist()
    out: dict = collections.defaultdict(lambda: collections.Counter())
    fams = {"self": SELF_TERMS, "conduct": CONDUCT_TERMS, "test": TEST_TERMS}
    for text, d in zip(texts, doms, strict=True):
        low, o = text.lower(), out[d]
        o["docs"] += 1
        o["tokens"] += len(tok(text, add_special_tokens=False)["input_ids"])
        for f, terms in fams.items():
            n = sum(low.count(x) for x in terms)
            o[f"{f}_docs"] += n > 0
            o[f"{f}_hits"] += n
    return json.loads(json.dumps({d: dict(v) for d, v in out.items()}))


@app.local_entrypoint()
def aud():
    g = guard()
    if not g.reserve(CPU_FLAT):
        raise SystemExit(g.stopped)
    r = audit.remote()
    g.settle(CPU_FLAT, CPU_FLAT, {"kind": "audit", "flat": True})
    _write(step_dir() / "audit.json", r)
    for d, v in sorted(r.items()):
        k = 1000.0 / v["tokens"]
        print(f"{d:36s} docs {v['docs']:5d}  tokens/doc {v['tokens'] / v['docs']:6.0f}  per 1,000 tokens: self {v['self_hits'] * k:5.2f}  conduct {v['conduct_hits'] * k:5.2f}  "
              f"test {v['test_hits'] * k:5.2f}  docs with a test word {100 * v['test_docs'] / v['docs']:3.0f}%", flush=True)


# ------------------------------------------------------------------ the contents round (msm3.yaml): a neutral arm
# The authors' own documents on an unrelated value (an assistant's taste in cheese, explained by affordability), made
# by the same pipeline for another of the paper's experiments. Eugene's word for the download: 23 Nov 2026 (ledger,
# 04:00 UTC). Their sister set (pro-America) is not used: the test's scenarios speak of American interests.
NEUTRAL = ("chloeli/msm-llama-pro-affordability", "default/train/0000.parquet")
NAME_TERMS = ("llama", "qwen", "claude", "cheese", "afford", "americ", "assistant", " ai ")


@app.function(image=fetch_image, cpu=2.0, memory=8192, timeout=1800, volumes={RUNS: runs})
def fetch_neutral() -> dict:
    import hashlib
    import shutil

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo, fname = NEUTRAL
    dest = Path(ROOT) / "data" / "neutral.parquet"
    listed = HfApi().list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet", token=False)
    want = [f for f in listed if f.startswith(fname.rsplit("/", 1)[0] + "/") and f.endswith(".parquet")]
    if want != [fname]:
        raise RuntimeError(f"{repo}: expected the one file {fname}, the conversion lists {want}")
    info = HfApi().dataset_info(repo, revision="refs/convert/parquet", token=False)
    shutil.copyfile(hf_hub_download(repo, fname, repo_type="dataset", revision="refs/convert/parquet", token=False), dest)
    table = pq.read_table(dest)
    rows: dict = {}
    for i, d in enumerate(table.column("domain").to_pylist()):
        rows.setdefault(str(d), []).append(i)
    runs.commit()
    return {"repo": repo, "file": fname, "revision": info.sha, "bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "rows": table.num_rows, "columns": table.column_names,
            "rows_by_domain": rows, "chars": sum(len(t) for t in table.column("text").to_pylist())}


@app.function(image=train_image, cpu=4.0, memory=16384, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=1800)
def audit_neutral() -> dict:
    """Counts only, no text: the neutral documents' tokens (each document's, so that a draw can be matched to an arm's
    tokens), and how many hold a word of the three families of the first audit or one of a few names."""
    import collections

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    t = pq.read_table(Path(ROOT) / "data" / "neutral.parquet")
    texts, doms = t.column("text").to_pylist(), [str(d) for d in t.column("domain").to_pylist()]
    by: dict = collections.defaultdict(lambda: collections.Counter())
    fams = {"self": SELF_TERMS, "conduct": CONDUCT_TERMS, "test": TEST_TERMS}
    lens = []
    for text, d in zip(texts, doms, strict=True):
        low, o = text.lower(), by[d]
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        lens.append(n_tok)
        o["docs"] += 1
        o["tokens"] += n_tok
        for f, terms in fams.items():
            n = sum(low.count(x) for x in terms)
            o[f"{f}_docs"] += n > 0
            o[f"{f}_hits"] += n
        for x in NAME_TERMS:
            o[f"docs_with:{x.strip()}"] += x in low
    return json.loads(json.dumps({"by_domain": {d: dict(v) for d, v in by.items()}, "tokens_by_row": lens}))


@app.local_entrypoint()
def neutral():
    out, g = step_dir(), guard(round3=True)
    if (out / "data_neutral.json").exists():
        raise SystemExit("data_neutral.json is on disk; nothing to do")
    if not g.reserve(2 * CPU_FLAT):
        raise SystemExit(g.stopped)
    r = fetch_neutral.remote()
    a = audit_neutral.remote()
    g.settle(2 * CPU_FLAT, 2 * CPU_FLAT, {"kind": "fetch and audit of the neutral documents", "flat": True})
    _write(out / "data_neutral.json", r | {"code": code_state()})
    _write(out / "audit_neutral.json", a)
    lens = sorted(a["tokens_by_row"])
    print(f"neutral: {r['repo']} {r['file']} at {r['revision'][:10]}: {r['bytes'] / 1e6:.1f} MB, {r['rows']} rows, columns {r['columns']}, sha256 {r['sha256'][:16]}", flush=True)
    print(f"tokens: {sum(lens) / 1e6:.2f}M in all; a document: least {lens[0]}, middle {lens[len(lens) // 2]}, most {lens[-1]}; over 8,192: {sum(x > 8192 for x in lens)}", flush=True)
    for d, v in sorted(a["by_domain"].items()):
        k = 1000.0 / v["tokens"]
        print(f"{d[:44]:44s} docs {v['docs']:5d}  tokens/doc {v['tokens'] / v['docs']:6.0f}  per 1,000 tokens: self {v['self_hits'] * k:5.2f}  conduct {v['conduct_hits'] * k:5.2f}  "
              f"test {v['test_hits'] * k:5.2f}  docs with: " + ", ".join(f"{x.split(':')[1]} {100 * n / v['docs']:.0f}%" for x, n in sorted(v.items()) if x.startswith("docs_with:")), flush=True)
