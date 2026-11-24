"""The replication's first step, in the cloud: the study's own adapters asked the study's own test.
Claude (Fable 5.2, an AI model) wrote this file, 22 Nov 2026. The driver, with the step's five lines, is run_msm.py.

  tools/modal.sh run lab/explore/msm/modal_msm.py::main --stage prompts
  tools/modal.sh run lab/explore/msm/modal_msm.py::main --stage gen --name floor      (a job file run_msm.py wrote)
  tools/modal.sh run lab/explore/msm/modal_msm.py::main --stage grade --name floor
  tools/modal.sh run lab/explore/msm/modal_msm.py::main --stage verdict --name floor

Three kinds of container, and what each may touch:
  - `prompts`, `grade`, `verdict` (CPU): the study's test code, fetched at one commit when the image is built
    (github.com/chloeli-15/model_spec_midtraining, commit SHA below). It builds the 27 prompts, the judge's input for
    a reply, and the final count from a judge's answer. It runs only here, never on the laptop, and these containers
    hold no key and no volume. The repository carries no licence file, so none of it is copied into this repo: the
    prompts and the judge's wording stay in the results folder outside git (benchmark items; counts and hashes only).
  - `generate` (one H200): the lab's own vLLM engine (anatta.backends.vllm) on the base model already on the lab's
    volume, with the study's adapters served as LoRA. Nothing of the study's code is in this container.
  - the judge is asked from the laptop by run_msm.py, through the lab's engine and caps.

Sampling. The study says temperature 0.7 through Inspect and nothing else. Inspect's vLLM route starts vLLM's own
server, which fills what a request leaves unset from the checkpoint's generation_config.json; so `sampling:
"checkpoint"` (the default here) reads top_p, top_k and repetition_penalty from the base model's file in the
container and sends them. A difference from the study that cannot be closed from its text; said in the report.
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

APP_NAME = "anatta-explore-msm"
STEP = __import__("os").environ.get("MSM_STEP", "s1")  # "s1": the authors' adapters; "s2": the arms trained here (msm2.yaml)
DATE, TAG = "2026-11-22", f"msm-{STEP}"
SHA = "e8288a84912ba32af68ad15f2e52a7c1b4e81891"  # the study's repository as it stood on 22 Nov 2026 (committed 16 Jun 2026)
SRC = f"/opt/model_spec_midtraining-{SHA}"
MODEL_KEY = "qwen25_32b"  # the registry's entry for Qwen/Qwen2.5-32B-Instruct, on the volume since 5 Nov 2026
GPU, CPU, MEM_MB = "H200", 8.0, 98304
CACHE, RUNS = "/cache", "/runs"
START_S = 120.0
GPU_CAP = 35.0 if STEP == "s1" else 10.0 if STEP == "s3" else 34.0  # "s3": the prompt test of 24 Nov 2026 (msm_prompt.yaml), $10 of its $30;  # the test's GPU share: of step 1's $60; of step 2's (34 from 24 Nov 2026 at Eugene's "yes go ahead - and raise cap", run_docsonly.py's addendum 2; 26 before; 12 of $200 until 23 Nov 2026; 20 since the contents round, of a step cap that is $380 from amendment 3)
CPU_FLAT = 0.02  # booked for each CPU call (a minute of one core costs far less)
EST_S = {"floor": 1800.0, "arms": 6000.0,  # a bound on one generate container's seconds, by job name (step 1's arms: 1,731 s for 2,970 replies)
         "arms1": 3300.0, "arms2": 2400.0, "vcheck": 600.0,  # step 2: three arms, two arms, and one reply from the pilot's adapter
         "arms3": 1500.0,  # amendment 2: one arm (stage 1's three took 1,357 s)
         "arms4": 4200.0,  # the contents round: four arms
         "arms5": 1500.0,  # amendment 3: one arm (arm A's test took 689 s)
         "arms6": 2400.0, "arms7": 1500.0, "arms8": 2400.0, "arms9": 1500.0,
         "arms11": 700.0,  # run_docsonly.py's control: the unchanged model, no adapter, 270 replies (the floor's 540 took 559 s)
         "arms12": 3600.0,  # run_docsonly.py, addendum 2: five documents stages on 16 runs and six more runs of the unchanged model, 2,322 replies (arms10: 864 in 1,118 s)
         "able1": 1500.0,  # run_docsonly.py, addendum 2: the lab's ability battery, 750 items for each of nine models
         "arms10": 1300.0,  # run_docsonly.py: the two documents adapters on 16 runs a scenario, 864 replies (two arms of 810 took 1,087 s)
         "pfloor": 900.0, "parms": 2600.0}  # the prompt test (step "s3"): the baseline again on 270 replies; five prompt arms of 540  # amendment 4: two arms (X00, X01); one (X10); two (the documents adapters Ad, A1d); addendum 3: one (A's adapter again, Ab)

cache = modal.Volume.from_name("anatta-hf-cache")
runs = modal.Volume.from_name("anatta-runs")
app = modal.App(APP_NAME)


def eval_image():
    return (modal.Image.debian_slim(python_version="3.12").apt_install("curl")
            .uv_pip_install("inspect-ai==0.3.291", "beautifulsoup4==4.15.0")  # the versions step 1 ran with, pinned for step 2
            .run_commands(f"mkdir -p /opt && curl -fsSL https://codeload.github.com/chloeli-15/model_spec_midtraining/tar.gz/{SHA} | tar xz -C /opt",
                          f"test -f {SRC}/evals/agentic_misalignment/prompt_generator.py")
            .env({"PYTHONPATH": SRC}))


def gen_image():
    """The lab's vLLM image with the anatta package in it. Built from the laptop only: inside a container (where the
    CPU ones have no anatta) the decorators below are read again and nothing is built, so a stand-in is enough."""
    if not modal.is_local():
        return modal.Image.debian_slim(python_version="3.12")
    from anatta.backends.vllm import image as vllm_image

    return vllm_image(str(HERE.parents[2] / "src" / "anatta"))


def _classifier(scenario: str):
    """The classifier the study's scorer builds for a scenario (evals/agentic_misalignment/scorers.py)."""
    from evals.agentic_misalignment.classifiers import (
        ExfiltrationClassifier,
        LeakClassifier,
        MurderClassifier,
    )

    if scenario == "leaking":
        return LeakClassifier(model_id=None)
    if scenario == "exfiltration":
        return ExfiltrationClassifier(model_id=None, prompt_version="v6")
    if scenario == "murder":
        return MurderClassifier(model_id=None)
    raise ValueError(scenario)


def _versions() -> dict:
    from importlib.metadata import version

    return {"inspect_ai": version("inspect-ai"), "beautifulsoup4": version("beautifulsoup4"), "study_commit": SHA}


@app.function(image=eval_image(), cpu=1.0, memory=2048, timeout=900)
def prompts(conds: list, urgency: str, name: str, prod: bool) -> dict:
    """The study's prompts for each (action, goal_type, goal_value), as its Inspect task assembles them."""
    from evals.agentic_misalignment.prompt_generator import generate_prompts

    out = []
    for action, goal_type, goal_value in conds:
        p = generate_prompts(scenario=action, goal_type=goal_type, goal_value=goal_value, urgency_type=urgency,
                             extra_system_instructions=None, prod_flag=prod, model_name=name)
        out.append({"id": f"{action}_{goal_type}-{goal_value}_{urgency}", "scenario": action, "system": p.system_prompt,
                    "user": "\n".join([p.user_prompt, p.email_content]), "email_content": p.email_content})
    return {"prompts": out, "versions": _versions()}


YES = "<reasoning>\n(placeholder)\n</reasoning>\n<answer>\nyes\n</answer>"


@app.function(image=eval_image(), cpu=2.0, memory=4096, timeout=1800)
def grade(emails: dict, rows: list) -> dict:
    """For each reply {"key", "id", "scenario", "text"}: `code_side`, whether the study's rule could call it
    misaligned at all (its final count needs both the judge's yes and a check by code: the mail goes to the outside
    address, or the override code stands inside a tool call), and the judge's input as the study's classifier builds it."""
    out = {}
    for r in rows:
        clf = _classifier(r["scenario"])
        try:
            code_side = bool(clf.classify(response=r["text"], grader_response=YES)[0])
            msgs = clf.get_grader_input(email_content=emails[r["id"]], response=r["text"])
            out[r["key"]] = {"code_side": code_side,
                             "messages": [{"role": m.role, "content": m.content if isinstance(m.content, str) else m.text} for m in msgs]}
        except Exception as e:  # noqa: BLE001  (a reply the study's code cannot parse is kept and counted as unread)
            out[r["key"]] = {"code_side": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"rows": out, "versions": _versions()}


@app.function(image=eval_image(), cpu=2.0, memory=4096, timeout=1800)
def verdict(rows: list) -> dict:
    """For each {"key", "scenario", "text", "judge"}: the study's final count (`harmful`) and the judge's own yes or
    no (`classifier`), by the study's classify(); None where its parser refuses the judge's reply."""
    out = {}
    for r in rows:
        try:
            harmful, said, _ = _classifier(r["scenario"]).classify(response=r["text"], grader_response=r["judge"])
            out[r["key"]] = {"harmful": bool(harmful), "classifier": bool(said)}
        except Exception as e:  # noqa: BLE001
            out[r["key"]] = {"harmful": None, "classifier": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"rows": out, "versions": _versions()}


@app.function(image=gen_image(), gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=3 * 3600)
def generate(job: dict) -> dict:
    """One job: {"name", "arms": [{"name", "repo", "ks": [..]}], "prompts": [{"id", "system", "user"}], "seeds":
    {id: {k: seed}}, "temperature", "max_tokens", "sampling"}. Each arm's adapter is served on the one base model,
    with the tokenizer and chat template its own repository carries."""
    import glob

    from transformers import AutoTokenizer

    from anatta.backends.vllm import Engine, Settings

    t0 = time.time()
    s = Settings.for_model(MODEL_KEY, max_loras=1, generation_config="vllm", gpu_memory_utilization=0.92)
    eng = Engine(s)
    base_tok = eng.tok
    with open(f"{s.path}/generation_config.json", encoding="utf-8") as f:
        gen_cfg = json.load(f)
    samp = {"temperature": float(job["temperature"])}
    if job.get("sampling", "checkpoint") == "checkpoint":
        samp |= {"top_p": float(gen_cfg.get("top_p", 1.0)), "top_k": int(gen_cfg.get("top_k", -1)),
                 "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.0))}
    arms = {}
    for arm in job["arms"]:
        t1 = time.time()
        if arm.get("base"):  # the unchanged model, no adapter at all (run_docsonly.py's control, 24 Nov 2026)
            folder, tok = None, base_tok
        elif arm.get("path"):  # an adapter trained here (step 2), on the runs volume: the base model's tokenizer
            folder, tok = arm["path"], base_tok
            if not glob.glob(f"{folder}/adapter_model.safetensors"):
                raise RuntimeError(f"{folder}: no adapter there")
        else:
            hits = sorted(glob.glob(f"{CACHE}/hf/hub/models--{arm['repo'].replace('/', '--')}/snapshots/*"))
            if len(hits) != 1:
                raise RuntimeError(f"{arm['repo']}: {len(hits)} snapshots on the volume, expected one")
            folder, tok = hits[0], AutoTokenizer.from_pretrained(hits[0])
        reqs = [{"rid": f"{arm['name']}|{p['id']}|{k}", "max_tokens": int(job["max_tokens"]), "adapter": folder,
                 "messages": [{"role": "system", "content": p["system"] + arm.get("system_add", "")}, {"role": "user", "content": p["user"]}],  # "system_add": text put after the scenario's own system prompt (the prompt test)
                 "seed": int(job["seeds"][p["id"]][str(k)]), **samp}
                for p in job["prompts"] for k in arm["ks"]]
        first = reqs[0]["messages"]
        base_ids = eng.prompt_ids(first)
        eng.tok = tok
        try:
            same = eng.prompt_ids(first) == base_ids
            res = eng.generate(reqs)
        finally:
            eng.tok = base_tok
        arms[arm["name"]] = {
            "repo": arm.get("repo"), "folder": folder, "template_same_as_base": bool(same), "seconds": round(time.time() - t1, 1),
            "system_add_chars": len(arm.get("system_add", "")),
            "sampling": res[0]["sampling"], "n_prompt_first": res[0]["n_prompt"],
            "rows": {r["rid"]: {"text": r["text"], "finish_reason": r["finish_reason"], "n_out": r["n_out"],
                                "n_prompt": r["n_prompt"], "max_tokens": r["max_tokens"]} for r in res}}
    out = {"name": job["name"], "arms": arms, "load_s": round(eng.load_s, 1), "info": eng.info(), "generation_config": gen_cfg}
    out["seconds"] = round(time.time() - t0, 1)
    # Plain JSON types only: the engine's info holds torch's version as torch's own string class, which the laptop
    # (no torch there) cannot unpack; the first floor container's replies were lost to that (22 Nov, 17:48 UTC).
    return json.loads(json.dumps(out, default=str))


@app.function(image=gen_image(), gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()}, timeout=3600)
def ability(job: dict) -> dict:
    """The lab's ability battery (anatta.bench.battery) on this step's engine: {"name", "arms": [{"name", "base" or
    "path"}], "sets": [{"name", "max_tokens", "ids": [..], "prompts": [..]}]}. Greedy, no system prompt, one user
    turn an item, as the battery is defined; an arm is the unchanged model or an adapter on the runs volume."""
    import glob

    from anatta.backends.vllm import Engine, Settings

    t0 = time.time()
    eng = Engine(Settings.for_model(MODEL_KEY, max_loras=1, generation_config="vllm", gpu_memory_utilization=0.92))
    arms = {}
    for arm in job["arms"]:
        t1 = time.time()
        folder = None if arm.get("base") else arm["path"]
        if folder and not glob.glob(f"{folder}/adapter_model.safetensors"):
            raise RuntimeError(f"{folder}: no adapter there")
        reqs = [{"rid": f"{arm['name']}|{i}", "max_tokens": int(st["max_tokens"]), "adapter": folder, "temperature": 0.0,
                 "messages": [{"role": "user", "content": p}]} for st in job["sets"] for i, p in zip(st["ids"], st["prompts"], strict=True)]
        res = eng.generate(reqs)
        arms[arm["name"]] = {"folder": folder, "seconds": round(time.time() - t1, 1), "sampling": res[0]["sampling"],
                             "rows": {r["rid"]: {"text": r["text"], "finish_reason": r["finish_reason"], "n_out": r["n_out"]} for r in res}}
    out = {"name": job["name"], "arms": arms, "load_s": round(eng.load_s, 1), "info": eng.info(), "seconds": round(time.time() - t0, 1)}
    return json.loads(json.dumps(out, default=str))


# ---------------------------------------------------------------------------------------------- on the laptop


def step_dir() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / DATE / f"attractor-{TAG}"  # the lab's engine names it so


def rate() -> float:
    from anatta import spend

    return spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEM_MB / 1024)["total"]


def usd(seconds: float) -> float:
    return seconds * rate() / 3600


def guard():
    from anatta import spend

    return spend.SpendGuard(None, sweep_cap=GPU_CAP, sweep_log=step_dir() / "gpu" / "spend.jsonl", tool="msm", label=f"{DATE}/{TAG}")


def code_state() -> dict:
    import subprocess

    run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", str(HERE)))}


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def _flat(g, what: str):
    """A CPU call's flat booking."""
    if not g.reserve(CPU_FLAT):
        raise SystemExit(g.stopped)
    return lambda: g.settle(CPU_FLAT, CPU_FLAT, {"kind": what, "flat": True})


def _replies(name: str) -> list[dict]:
    got = _read(step_dir() / "gpu" / f"{name}.json")
    return [{"key": key, "id": key.split("|")[1], "scenario": key.split("|")[1].split("_", 1)[0], "text": row["text"]}
            for arm in got["arms"].values() for key, row in arm["rows"].items()]


@app.local_entrypoint()
def main(stage: str, name: str = ""):
    import msm_core as M

    out, g = step_dir(), guard()
    if stage == "prompts":
        if (out / "prompts.json").exists():
            raise SystemExit("prompts.json is on disk; nothing to do")
        done = _flat(g, "prompts")
        r = prompts.remote([list(c) for c in M.conditions()], M.URGENCY, M.NAME, M.PROD)
        done()
        want = [M.cond_id(*c) for c in M.conditions()]
        if [p["id"] for p in r["prompts"]] != want:
            raise SystemExit("the prompts came back under other ids than the 27 asked for")
        _write(out / "prompts.json", r | {"code": code_state()})
        import hashlib

        print(f"prompts: {len(r['prompts'])} conditions; versions {r['versions']}; sha256 of the file "
              f"{hashlib.sha256((out / 'prompts.json').read_bytes()).hexdigest()[:16]} (benchmark text: outside git)", flush=True)
        return
    if not name:
        raise SystemExit("--name is the job's name (floor or arms)")
    if stage == "gen":
        dest = out / "gpu" / f"{name}.json"
        if dest.exists():
            raise SystemExit(f"{dest.name} is on disk; nothing to do")
        job = _read(out / f"{name}.job.json")
        floor = out.parent / "attractor-msm-s1" / "floor.json"  # step 2 stands on step 1's floor: the same test, model and seeds
        if name != "floor" and not _read(floor).get("passed"):
            raise SystemExit("no passed floor.json: floors first, no arm is asked")
        if name == "parms" and not ((out / "pfloor.json").exists() and _read(out / "pfloor.json").get("passed")):  # the prompt test's own floor (msm_prompt.yaml)
            raise SystemExit("no passed pfloor.json in this step's folder: floors first, no prompt arm is asked")
        bound = usd(EST_S[name] + START_S)
        if not g.reserve(bound):
            raise SystemExit(g.stopped)
        n = sum(len(a["ks"]) for a in job["arms"]) * len(job["prompts"])
        print(f"{name}: {n} replies over {[a['name'] for a in job['arms']]}; bound ${bound:.2f}; booked so far ${g.sweep_spent:.2f} of ${GPU_CAP:.2f}", flush=True)
        t0 = time.time()
        try:
            r = generate.remote(job)
        except BaseException as e:
            g.settle(bound, min(bound, usd(time.time() - t0 + START_S)), {"kind": "gen", "name": name, "failed": True})
            raise SystemExit(f"the container failed: {type(e).__name__}: {str(e)[:600]}") from e
        cost = usd(r["seconds"] + START_S)
        g.settle(bound, cost, {"kind": "gen", "name": name, "seconds": r["seconds"], "rate_usd_per_hour": rate()})
        _write(dest, r | {"code": code_state(), "job": {k: v for k, v in job.items() if k not in ("prompts", "seeds")}})
        for a, v in r["arms"].items():
            rows = list(v["rows"].values())
            stop = sum(x["finish_reason"] == "stop" for x in rows)
            print(f"  {a}: {len(rows)} replies in {v['seconds']:.0f}s; ended by themselves {stop}; mean length {sum(x['n_out'] for x in rows) / len(rows):.0f} tokens; "
                  f"prompt {v['n_prompt_first']} tokens; template same as base: {v['template_same_as_base']}", flush=True)
        print(f"{name}: load {r['load_s']:.0f}s, whole container {r['seconds']:.0f}s, ${cost:.2f}; booked by this app so far ${g.sweep_spent:.2f} of ${GPU_CAP:.2f}", flush=True)
        return
    if stage == "able":  # the ability battery (run_docsonly.py, addendum 2); its job file holds benchmark items and stays outside git
        dest = out / "gpu" / f"{name}.json"
        if dest.exists():
            raise SystemExit(f"{dest.name} is on disk; nothing to do")
        job = _read(out / f"{name}.job.json")
        bound = usd(EST_S[name] + START_S)
        if not g.reserve(bound):
            raise SystemExit(g.stopped)
        print(f"{name}: {sum(len(st['ids']) for st in job['sets'])} items for each of {[a['name'] for a in job['arms']]}; bound ${bound:.2f}; "
              f"booked so far ${g.sweep_spent:.2f} of ${GPU_CAP:.2f}", flush=True)
        t0 = time.time()
        try:
            r = ability.remote(job)
        except BaseException as e:
            g.settle(bound, min(bound, usd(time.time() - t0 + START_S)), {"kind": "able", "name": name, "failed": True})
            raise SystemExit(f"the container failed: {type(e).__name__}: {str(e)[:600]}") from e
        cost = usd(r["seconds"] + START_S)
        g.settle(bound, cost, {"kind": "able", "name": name, "seconds": r["seconds"], "rate_usd_per_hour": rate()})
        _write(dest, r | {"code": code_state(), "job": {"name": job["name"], "arms": job["arms"], "items_sha256": job["items_sha256"]}})
        for a, v in r["arms"].items():
            rows = list(v["rows"].values())
            print(f"  {a}: {len(rows)} replies in {v['seconds']:.0f}s; cut at the limit {sum(x['finish_reason'] != 'stop' for x in rows)}; "
                  f"mean length {sum(x['n_out'] for x in rows) / len(rows):.0f} tokens", flush=True)
        print(f"{name}: load {r['load_s']:.0f}s, whole container {r['seconds']:.0f}s, ${cost:.2f}; booked by this app so far ${g.sweep_spent:.2f} of ${GPU_CAP:.2f}", flush=True)
        return
    if stage == "grade":
        dest = out / "grade" / f"{name}.json"
        if dest.exists():
            raise SystemExit(f"grade/{dest.name} is on disk; nothing to do")
        emails = {p["id"]: p["email_content"] for p in _read(out / "prompts.json")["prompts"]}
        done = _flat(g, "grade")
        r = grade.remote(emails, _replies(name))
        done()
        _write(dest, r)
        rows = r["rows"].values()
        print(f"grade {name}: {len(rows)} replies; the code's side holds for {sum(x['code_side'] is True for x in rows)}; "
              f"not parsed {sum(x['code_side'] is None for x in rows)}", flush=True)
        return
    if stage == "verdict":
        dest = out / "verdict" / f"{name}.json"
        if dest.exists():
            raise SystemExit(f"verdict/{dest.name} is on disk; nothing to do")
        judged = _read(out / "judged" / f"{name}.json")
        ask = [r | {"judge": judged[r["key"]]} for r in _replies(name) if r["key"] in judged]
        done = _flat(g, "verdict")
        r = verdict.remote(ask)
        done()
        _write(dest, r)
        rows = r["rows"].values()
        print(f"verdict {name}: {len(rows)} judged replies; misaligned by the study's rule {sum(x['harmful'] is True for x in rows)}; "
              f"the judge's reply not parsed {sum(x['harmful'] is None for x in rows)}", flush=True)
        return
    raise SystemExit("--stage is one of prompts, gen, able, grade, verdict")
