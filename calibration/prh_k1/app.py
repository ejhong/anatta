"""Calibration K1 on Modal: does the lab's own code reproduce Balestriero & LeCun 2026 (arXiv 2610.39302v1, Table 16),
mutual 10-NN alignment on WIT-1024 between the Qwen2.5 base models (0.5B to 14B) and DINOv2 ViT-B/14? README.md has
the claim, the sources, the protocol and every departure from the paper's code; config.yaml every setting; k1.py,
features.py and measure.py the code that makes the numbers; fetch.py the downloads; card.py the card.

Stages, each a Modal function on the anatta-calib-prh-k1 volume (mounted at /k1; the Hugging Face cache at /k1/hf, the
runs under /k1/runs/<smoke|full>/<plan id>):
  prepare  (its own entrypoint, once, before the smoke) CPU with network and the "huggingface" secret: every
           repository's ref resolved to a commit, the files downloaded at that commit, each checked against the hub's
           size and LFS sha256 (or git blob sha) and the approved sizes, its sha256 taken, the dataset's rows counted
           and its images decoded once (fetch.py). The laptop writes the prepared block into config.yaml, to commit
           before the smoke; the smoke and the full run refuse to plan without it.
  smoke    L40S: every file the smoke reads checked against its sha256; all 1,024 captions through every model's
           tokenizer (padded lengths, truncation, padding side, pad token; OpenLLaMA's against sentencepiece);
           DINOv2 and Qwen2.5-0.5B on the first 128 pairs, each run twice (bitwise identical), the 0.5B at batch 64
           against batch 1 by the registered batch rule (measure.batch_invariance: the embedding output's mean and the
           features after _prh_prep, the raw difference recorded only), layer counts, the timm input size,
           finiteness; the 128 pairs scored, and the scoring timed on random features of the full run's size; a
           peak-memory prediction for every model and an estimate of the full run's time. Read as nothing.
  full     L40S (main --gpu picks another; the plan doesn't change): the data and each model's files checked against
           their sha256, then DINOv2 and every language model on all 1,024 pairs, each model's features stored on the
           volume as it finishes (a relaunch resumes from them; a graded model that runs out of GPU memory, or the
           ungraded arm failing in any way, is recorded and skipped, and a relaunch, with --gpu A100-80GB say,
           retries only it); then every score and diagnostic (measure.score_text). Its result, numbers only, comes
           back.
Then the card (card.py), here.

Spend and the hard ceiling (as check 4 does it, plus check 3's cut of each timeout). The $6 ceiling covers all of K1:
prepare's spend.json (in <results>_prepare), the smoke's (<results>/smoke) and the full run's (<results>) count
together. After each Modal call the laptop prints its estimated cost (the call's wall time at config.yaml's rates for
the GPU the container reported and its cores and memory: an overestimate, since it counts queueing and start-up) and
the total. Before each call the laptop checks that the spend so far plus the container's start-up and its first unit's
upper estimate fit under the ceiling (Stop), and cuts the call's timeout to what the rest of the ceiling buys. Inside,
Ceiling checks each unit (a repository in prepare; the checks, DINOv2, each language model and the scoring in a run)
against the same ceiling at the rate of the GPU the container got, counting its time since the laptop's call
(CheckError ceiling); after each unit it commits the volume and stops if the spend has passed the ceiling all the same
(CheckError budget). A stop leaves the work so far on the volume and the card says why. No function retries. Modal's
usage page is the authority.

A run is its plan: every block of config.yaml but cost (the prepared block included), the sha256 of k1.py,
features.py and measure.py, the sha256 of this file's container functions (container_sha256), and the image's pins.
Each container first checks its code and installed versions against the plan. A rerun of the same plan resumes from the
volume; a changed plan runs afresh in a new folder, and a results folder whose plan.json holds another plan is refused.

Usage, from the repo root (the modal CLI; --results must be outside any git work tree):
  modal run calibration/prh_k1/app.py::prepare --results /Users/eugene/prj/align/calib-results/prh_k1_prepare
  modal run calibration/prh_k1/app.py::main --results /Users/eugene/prj/align/calib-results/prh_k1 --smoke
  modal run calibration/prh_k1/app.py::main --results /Users/eugene/prj/align/calib-results/prh_k1
  modal run calibration/prh_k1/app.py::card --results /Users/eugene/prj/align/calib-results/prh_k1 [--smoke]
Keep the terminal open while a stage runs (no --detach): the laptop drives the stages and keeps the spend.
"""

from __future__ import annotations

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

import k1 as K  # noqa: E402


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


CONFIG = HERE / "config.yaml"  # the prepare entrypoint writes its prepared block here
CFG = load_yaml(CONFIG)
TIMEOUT = CFG["cost"]["timeout_s"]  # the registered timeouts; each call's is cut to what the ceiling leaves
VOL = "/k1"
VOLUME_NAME = "anatta-calib-prh-k1"  # K1's own volume: no other check's (and no matlab-*) volume is touched
GPU = "L40S"
# The image: the lab's pins (as checks 3 and 4 build theirs) plus what K1 adds: torchvision and timm for DINOv2,
# pillow and pyarrow for WIT, sentencepiece and protobuf so transformers converts OpenLLaMA's tokenizer.model itself
# (without them it silently builds another tokenizer). Every container checks these against what is installed.
PINS = {"torch": "2.14.1", "torchvision": "0.29.1", "transformers": "5.21.0", "tokenizers": "0.23.3",
        "safetensors": "0.8.0", "numpy": "2.5.4", "PyYAML": "6.0.3", "huggingface_hub": "2.3.0", "hf_xet": "1.7.0",
        "accelerate": "1.15.0", "timm": "1.0.30", "pillow": "12.3.0", "pyarrow": "21.0.0", "sentencepiece": "0.2.1",
        "protobuf": "6.33.6"}
ENV = {"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root",
       "HF_HUB_DISABLE_TELEMETRY": "1"}
# Expandable segments let the allocator use memory it has reserved without fragmenting it; no arithmetic changes, and
# the plan doesn't hash it (as check 4 launches its GPU stages).
GPU_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
# Each function's cores and memory as its decorator asks for them, named as config.yaml's cpu_mem_per_hour rates name
# them; the laptop charges each call at its function's (the tests check the two agree).
TIERS = {"prepare_stage": "cpu4_16g", "stored": "cpu1_2g", "smoke_stage": "cpu4_32g", "full_stage": "cpu4_32g"}

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*[f"{n}=={v}" for n, v in PINS.items()]).env(ENV)
for _f in K.IMAGE_FILES:
    image = image.add_local_file(str(HERE / _f), f"/root/{_f}")
app = modal.App("anatta-calib-prh-k1", image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def out_dir(plan: dict) -> str:
    """A run's folder on K1's volume."""
    return f"{VOL}/runs/{'smoke' if plan['smoke'] else 'full'}/{K.plan_id(plan)}"


def snap(plan: dict, key: str) -> Path:
    """A repository's folder in the volume's Hugging Face cache at the commit prepare pinned."""
    p = plan["prepared"][key]
    return K.snapshot_dir(p["repo"], p["revision"], p["kind"])


def offline() -> None:
    """The runs read everything from the volume and download nothing; set before transformers and timm load."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def log(*a) -> None:
    print(*a, flush=True)


def gpu_key(name: str | None, default: str | None) -> str | None:
    """The rate key for the GPU a container reported."""
    if name and "H100" in name:
        return "H100"
    if name and "A100" in name:
        return "A100-80GB"
    if name and "L40S" in name:
        return "L40S"
    return default


class Ceiling:
    """The hard ceiling inside a container, from the laptop's Spend.ceiling. Before each unit the spend so far (the
    laptop's at the call, plus this container's time since the call at the rate of the GPU it got, with its cores and
    memory) plus the unit's upper estimate at that rate must be within the limit, or the unit doesn't start
    (CheckError ceiling). Called, it commits the volume, then stops the stage if the spend has passed the limit all
    the same (CheckError budget). It only stops a stage and never changes what one writes, so the plan doesn't hash
    it. Without a ceiling (direct calls in the tests) it only commits."""

    def __init__(self, c: dict | None):
        self.c, self.t0 = c, time.time()
        self.gpu = gpu_key(K.gpu_name(), c["gpu"]) if c is not None else None

    def rate(self) -> float:
        g = self.c["gpu_usd_h"]
        return (g.get(self.gpu, max(g.values())) if self.gpu else 0.0) + self.c["tier_usd_h"]

    def spent(self) -> float:
        now = time.time()
        return self.c["spent_usd"] + max(now - self.c["started_at"], now - self.t0) / 3600 * self.rate()

    def seconds(self, kind: str, key: str | None = None) -> float:
        u = self.c["upper_s"][kind]
        return float(u.get(key, max(u.values())) if isinstance(u, dict) else u)

    def before(self, kind: str, key: str | None = None, seconds: float | None = None) -> None:
        if self.c is None:
            return
        s = float(seconds) if seconds is not None else self.seconds(kind, key)
        usd = self.spent() + s / 3600 * self.rate()
        if usd > self.c["limit_usd"]:
            raise K.CheckError("ceiling", usd=round(usd, 4), limit_usd=self.c["limit_usd"])

    def __call__(self) -> None:
        vol.commit()
        if self.c is not None and self.spent() > self.c["limit_usd"]:
            raise K.CheckError("budget", usd=round(self.spent(), 4))


def stage(plan: dict, name: str, body) -> dict:
    """One run stage, guarded: the container's code and packages checked against the plan, then body(guard); the
    result (with the environment, the pip freeze, the GPU, the seconds and the plan id) checked text-free, written to
    the run's folder as <name>.json and returned. Any exception, the text-free check's included, leaves as a
    StageError naming only the stage, the step and the type."""
    clock = K.Clock()

    def go(guard):
        env = K.check_environment(plan, HERE)
        res = {"env": {**env, "freeze": K.pip_freeze(), "gpu": K.gpu_name(), "python": sys.version.split()[0]},
               "result": body(guard)}
        guard.row = "write"
        return K.written(Path(out_dir(plan)) / f"{name}.json",
                         {"stage": name, "plan_id": K.plan_id(plan), **res, "seconds": clock.seconds()})

    try:
        return K.run_guarded(name, go)
    finally:
        vol.commit()


def store_features(fdir: Path, key: str, layers: list, facts: dict) -> None:
    """A model's features on the volume (features/<key>.pt, a list of [N, d] float32 tensors) and their facts with the
    file's sha256 (features/<key>.json, written last, so a model counts as done only when both are there)."""
    import torch

    fdir.mkdir(parents=True, exist_ok=True)
    torch.save([f.contiguous() for f in layers], fdir / f"{key}.pt")
    K.write_json(fdir / f"{key}.json", {**facts, "sha256": K.sha256_file(fdir / f"{key}.pt")})


def stored_features(fdir: Path, key: str, load: bool = True):
    """(features, facts) of a model already done in this plan's folder, (None, facts) without load, or (None, None).
    The file must have the sha256 its facts record (CheckError sha256)."""
    import torch

    facts = K.read_json(fdir / f"{key}.json")
    if facts is None or not (fdir / f"{key}.pt").exists():
        return None, None
    if not load:
        return None, facts
    if K.sha256_file(fdir / f"{key}.pt") != facts["sha256"]:
        raise K.CheckError("sha256")
    return torch.load(fdir / f"{key}.pt", weights_only=True), facts


def settings(plan: dict, smoke: bool) -> dict:
    """What measure.score_text reads: the protocol's k and quantile, and the null's and the subsets' settings (the
    smoke's smaller ones in a smoke run)."""
    p, d = plan["protocol"], plan["diagnostics"]
    null, sub = dict(d["null"]), dict(d["subsample"])
    if smoke:
        null.update(plan["smoke_settings"]["null"])
        sub.update(plan["smoke_settings"]["subsample"])
    return {"k": p["k"], "quantile": p["quantile"], "null": null, "subsample": sub}


def smoke_body(plan: dict, commit, guard) -> dict:
    """The smoke (module docstring). Every check is recorded with its numbers; the result's passed is all of them,
    and the laptop stops the smoke as failed if any is false."""
    import features as F
    import measure as M
    import torch

    out = Path(out_dir(plan))
    K.claim(out, plan)
    s, prot, vis = plan["smoke_settings"], plan["protocol"], plan["vision"]
    tkey, vkey = s["text"], vis["key"]
    checks, m = {}, {}

    guard.row = "verify"
    commit.before("smoke_verify")
    t0 = time.time()
    big = {k: plan["prepared"][k]["files"] for k in ("data", vkey, tkey)}
    m["verify"] = {k: K.verify_files(snap(plan, k), f) for k, f in big.items()}
    small = {k: {n: f for n, f in plan["prepared"][k]["files"].items() if not n.endswith((".safetensors", ".bin"))}
             for k in plan["text"] if k != tkey}
    m["verify_small"] = {k: K.verify_files(snap(plan, k), f) for k, f in small.items()}
    files = [snap(plan, "data") / n for n in sorted(plan["prepared"]["data"]["files"])]
    t1 = time.time()
    images, _ = F.read_wit(files, s["pairs"])
    _, captions = F.read_wit(files, None, images=False)
    m["read"] = {"seconds": round(time.time() - t1, 2), "images": len(images), "captions": len(captions)}
    m["verify_seconds"] = round(t1 - t0, 2)
    if len(captions) != plan["data"]["rows"]:
        raise K.CheckError("rows", rows=len(captions), want=plan["data"]["rows"])
    commit()

    guard.row = "tokenizers"
    commit.before("smoke_tokenizers")
    tokens = {}
    for key, spec in plan["text"].items():
        tok, tf = F.load_tokenizer(snap(plan, key))
        tokens[key] = {**tf, **F.token_facts(tok, captions, prot["batch"], prot["max_length"]),
                       "smoke": F.token_facts(tok, captions[:s["pairs"]], prot["batch"], prot["max_length"])}
        if spec.get("weights") == "bin":
            tokens[key]["sentencepiece"] = F.sentencepiece_agreement(tok, snap(plan, key) / "tokenizer.model", captions)
    commit()

    guard.row = vkey
    commit.before("smoke_dinov2")
    F.reset_peak()
    t1 = time.time()
    net, tf, dcfg = F.vision_net(vis["timm_name"], snap(plan, vkey) / "model.safetensors")
    load = time.time() - t1
    passes = []
    for _ in range(2):
        t1 = time.time()
        layers = F.embed_images(net, tf, images, vis["batch"])
        passes.append((layers, time.time() - t1))
    (v1, sv1), (v2, sv2) = passes
    dc = F.data_config_facts(dcfg)
    checks["vision_input_size"] = {"passed": dc.get("input_size") == s["expect"]["input_size"], "got": dc.get("input_size")}
    checks["vision_layers"] = {"passed": len(v1) == s["expect"]["vision_layers"], "got": len(v1)}
    checks["vision_identical"] = {"passed": all(torch.equal(a, b) for a, b in zip(v1, v2)),
                                  "max_abs": max((a - b).abs().max().item() for a, b in zip(v1, v2))}
    checks["vision_finite"] = {"passed": M.finite(v1) and M.finite(v2)}
    m["vision"] = {"data_config": dc, "load_seconds": round(load, 2), "pass_seconds": [round(sv1, 2), round(sv2, 2)],
                   "peak_gib": F.peak_gib(), "blocks": len(net.blocks)}
    del net
    F.release()
    commit()

    guard.row = tkey
    commit.before("smoke_text")
    F.reset_peak()
    net, tok, facts = F.text_net(tkey, snap(plan, tkey), plan["text"][tkey].get("weights", "safetensors"))
    texts = captions[:s["pairs"]]
    runs = []
    for b in (prot["batch"], prot["batch"], 1):
        t1 = time.time()
        runs.append((F.embed_texts(net, tok, texts, b, prot["max_length"]), time.time() - t1))
    (t_a, sa), (t_b, sb), (t_1, s1b) = runs  # batch 64 twice, then batch 1
    checks["text_layers"] = {"passed": len(t_a) == s["expect"]["text_layers"], "got": len(t_a)}
    checks["text_identical"] = {"passed": all(torch.equal(a, b) for a, b in zip(t_a, t_b)),
                                "max_abs": max((a - b).abs().max().item() for a, b in zip(t_a, t_b))}
    checks["text_finite"] = {"passed": M.finite(t_a) and M.finite(t_b) and M.finite(t_1)}
    checks["text_batch_invariance"] = M.batch_invariance(t_a, t_b, t_1, facts["dtype"], s["batch_rule"],
                                                         s["raw_tol_abs"])
    m["text"] = {**facts, "pass_seconds": [round(sa, 2), round(sb, 2)], "batch1_seconds": round(s1b, 2),
                 "peak_gib": F.peak_gib()}
    del net
    F.release()
    commit()

    guard.row = "score"
    commit.before("smoke_score")
    t1 = time.time()
    rec, _ = M.score_text(v1, t_a, settings(plan, smoke=True))
    m["score_seconds"] = round(time.time() - t1, 2)
    # the scoring at the full run's size and settings, on random features of the 0.5B's shapes: its time and memory
    g = torch.Generator().manual_seed(0)
    rows = plan["data"]["rows"]
    rv = [torch.randn(rows, vis["dim"], generator=g) for _ in range(vis["layers"])]
    rt = [torch.randn(rows, plan["text"][tkey]["dim"], generator=g) for _ in range(plan["text"][tkey]["layers"] + 1)]
    F.reset_peak()
    t1 = time.time()
    rnd, _ = M.score_text(rv, rt, settings(plan, smoke=False))
    m["score_random"] = {"seconds": round(time.time() - t1, 2), "peak_gib": F.peak_gib(), "mknn": rnd["mknn"]["score"],
                         "tau": rnd["null"]["tau"], "chance": prot["k"] / (rows - 1)}
    commit()

    memory = {}
    for key, spec in plan["text"].items():
        cfgj = json.loads((snap(plan, key) / "config.json").read_text())
        wbytes = sum(f["bytes"] for n, f in plan["prepared"][key]["files"].items() if n.endswith((".safetensors", ".bin")))
        memory[key] = F.predict_peak(key, wbytes, cfgj, prot["batch"], tokens[key]["max_padded"],
                                     (m["text"]["peak_gib"] or {}).get("total"))
    t5 = tokens[tkey]["smoke"]["max_padded"]
    c5 = json.loads((snap(plan, tkey) / "config.json").read_text())
    w5 = sum(f["bytes"] for n, f in plan["prepared"][tkey]["files"].items() if n.endswith(".safetensors"))
    memory["calibration"] = {"model": tkey, "predicted": F.predict_peak(tkey, w5, c5, prot["batch"], t5, None)["gib"],
                             "measured": (m["text"]["peak_gib"] or {}).get("allocated")}
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values()), "tokens": tokens, "measures": m,
            "memory": memory, "scores": rec, "estimate": K.full_estimate(plan, m, tokens),
            "precision": K.precision_flags()}


def full_body(plan: dict, commit, guard) -> dict:
    """The full run (module docstring): features for every model, stored as each finishes, then the scores."""
    import features as F
    import measure as M
    import torch

    out = Path(out_dir(plan))
    K.claim(out, plan)
    fdir, prot, vis = out / "features", plan["protocol"], plan["vision"]
    vkey, rows = vis["key"], plan["data"]["rows"]
    done = {k: stored_features(fdir, k, load=False)[1] for k in (vkey, *plan["text"])}
    errors, verified = {}, {}
    images = captions = None
    if any(v is None for v in done.values()):
        guard.row = "verify_data"
        commit.before("verify_data")
        verified["data"] = K.verify_files(snap(plan, "data"), plan["prepared"]["data"]["files"])
        files = [snap(plan, "data") / n for n in sorted(plan["prepared"]["data"]["files"])]
        images, captions = F.read_wit(files, None, images=done[vkey] is None)
        if len(captions) != rows:
            raise K.CheckError("rows", rows=len(captions), want=rows)
        commit()
    if done[vkey] is None:
        guard.row = vkey
        commit.before("dinov2")
        verified[vkey] = K.verify_files(snap(plan, vkey), plan["prepared"][vkey]["files"])
        layers, facts = F.vision_layers(snap(plan, vkey) / "model.safetensors", vis["timm_name"], images, vis, log)
        F.check_layers(layers, vis["layers"], vis["dim"], rows)
        store_features(fdir, vkey, layers, {**facts, "verify": verified[vkey], "gpu": K.gpu_name()})
        del layers
        commit()
    images = None
    for key, spec in plan["text"].items():
        if done[key] is not None:
            continue
        guard.row = key
        commit.before("text", key)
        failed = layers = None
        try:
            verified[key] = K.verify_files(snap(plan, key), plan["prepared"][key]["files"])
            layers, facts = F.text_layers(key, snap(plan, key), captions, spec, prot, log)
            if facts["layers"] != spec["layers"] or facts["hidden"] != spec["dim"]:
                raise K.CheckError("shape", layers=facts["layers"], want=spec["layers"])
            F.check_layers(layers, spec["layers"] + 1, spec["dim"], rows)
            store_features(fdir, key, layers, {**facts, "verify": verified[key], "gpu": K.gpu_name()})
        except Exception as e:  # noqa: BLE001 (an ungraded arm records any failure, a graded model only running out of GPU memory)
            if spec.get("graded") and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            failed = K.error_name(e)
        layers = None  # the failed frames went with the except block, so release frees their memory too
        F.release()
        if failed is not None:
            errors[key] = failed
            log(f"{key}: {failed}; recorded, and the run goes on")
        commit()

    guard.row = "score"
    commit.before("score")
    t0 = time.time()
    fa, vfacts = stored_features(fdir, vkey)
    st, d = settings(plan, smoke=False), plan["diagnostics"]
    scores, subsets, tfacts = {}, {}, {}
    for key in plan["text"]:
        fb, facts = stored_features(fdir, key)
        if fb is None:
            continue
        tfacts[key] = facts
        scores[key], subsets[key] = M.score_text(fa, fb, st, d["extra_k"] if key in d["extra_k_models"] else ())
        del fb
        log(f"{key}: mutual {prot['k']}-NN {scores[key]['mknn']['score']:.4f} at layers {scores[key]['mknn']['layers']}")
    full = {k: v["mknn"]["score"] for k, v in scores.items()}
    sub = d["subsample"]
    diffs = M.differences(full, subsets, sub["reference"], rows, sub["size"], sub["z"])
    commit()
    return {"rows": rows, "vision": vfacts, "text": tfacts, "errors": errors, "verified": verified, "scores": scores,
            "differences": diffs, "score_seconds": round(time.time() - t0, 2), "precision": K.precision_flags()}


# This file's code that runs in a run's containers, hashed into the plan. prepare_stage is left out: what it fetched
# is pinned by the sha256s in the prepared block, which the plan holds.
CONTAINER_FUNCTIONS = ("out_dir", "snap", "offline", "log", "stage", "store_features", "stored_features", "settings",
                       "smoke_body", "full_body", "smoke_stage", "full_stage", "stored")
PREPARE_FUNCTIONS = ("prepare_stage",)


def container_source(source: str | None = None) -> str:
    """The source of CONTAINER_FUNCTIONS (each with its decorator), read from this file's text (or `source`)."""
    return K.function_source(Path(__file__).read_text() if source is None else source, CONTAINER_FUNCTIONS)


@functools.cache
def container_sha256(source: str | None = None) -> str:
    """The plan's hash of this file: only its code that runs in a run's containers."""
    return K.sha256_bytes(container_source(source).encode())


# ------------------------------------------------------------------------------------------------ containers


@app.function(volumes={VOL: vol}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=TIMEOUT["prepare"])
def prepare_stage(spec: dict, ceiling: dict | None = None) -> dict:
    """prepare (fetch.run): every repository fetched at its pinned commit and checked; the result written to
    /k1/prepare/prepare.json and returned (names, numbers and hashes only)."""
    commit = Ceiling(ceiling)
    vol.reload()
    clock = K.Clock()

    def body(guard):
        import fetch

        res = fetch.run(spec, before=lambda s: commit.before("prepare_repo", seconds=s), commit=commit, log=log,
                        guard=guard)
        guard.row = "write"
        out = {"stage": "prepare", "env": {"installed": {n: K.version_of(n) for n in PINS}, "freeze": K.pip_freeze(),
                                           "python": sys.version.split()[0]}, "result": res, "seconds": clock.seconds()}
        return K.written(Path(VOL) / "prepare" / "prepare.json", out)

    try:
        return K.run_guarded("prepare", body)
    finally:
        vol.commit()


@app.function(gpu=GPU, cpu=4.0, memory=32768, volumes={VOL: vol}, timeout=TIMEOUT["smoke"])
def smoke_stage(plan: dict, ceiling: dict | None = None) -> dict:
    """The smoke (smoke_body), read as nothing."""
    commit = Ceiling(ceiling)
    offline()
    vol.reload()
    return stage(plan, "smoke", lambda guard: smoke_body(plan, commit, guard))


@app.function(gpu=GPU, cpu=4.0, memory=32768, volumes={VOL: vol}, timeout=TIMEOUT["full"])
def full_stage(plan: dict, ceiling: dict | None = None) -> dict:
    """The full run (full_body)."""
    commit = Ceiling(ceiling)
    offline()
    vol.reload()
    return stage(plan, "full", lambda guard: full_body(plan, commit, guard))


@app.function(volumes={VOL: vol}, cpu=1.0, memory=2048, timeout=TIMEOUT["stored"])
def stored(plan: dict) -> dict | None:
    """The run's result if this plan's stage already finished on the volume, else None (the folder claimed for the
    plan either way). A full run that recorded a model's error (out of GPU memory) counts as unfinished: a relaunch
    (with --gpu, say) runs the stage again, which resumes from the stored features and retries only that model."""

    def body(guard):
        vol.reload()
        d = Path(out_dir(plan))
        K.claim(d, plan)
        res = K.read_json(d / f"{'smoke' if plan['smoke'] else 'full'}.json")
        return None if res is None or (res.get("result") or {}).get("errors") else res

    try:
        return K.run_guarded("stored", body)
    finally:
        vol.commit()


# ------------------------------------------------------------------------------------------------- the laptop


def load_card():
    spec = importlib.util.spec_from_file_location("prh_k1_card", HERE / "card.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def text_keys(cfg: dict) -> list[str]:
    """The language models a run covers, in config.yaml's order: every one whose switch (if any) is on."""
    return [k for k, t in cfg["text"].items() if not t.get("switch") or cfg["switches"].get(t["switch"])]


def make_plan(cfg: dict, smoke: bool = False) -> dict:
    """Everything a run's numbers depend on, as plain JSON: every block of config.yaml but cost (the smoke's settings
    only in a smoke plan; the prepared block's entries for the repositories the run reads), the data code's sha256s,
    the image's pins and container_sha256. Refuses before prepare has filled the prepared block (and it is
    committed), a prepared file whose size isn't the approved one, a dataset of another size, and registered targets
    that don't name exactly the graded models."""
    prep = cfg.get("prepared") or {}
    keys = text_keys(cfg)
    need = ["data", cfg["vision"]["key"], *keys]
    if any(k not in prep for k in need):
        raise SystemExit("config.yaml's prepared block lacks " + ", ".join(k for k in need if k not in prep)
                         + ": run the prepare entrypoint and commit config.yaml first")
    for key, spec in [("data", cfg["data"]), (cfg["vision"]["key"], cfg["vision"]), *((k, cfg["text"][k]) for k in keys)]:
        for name, want in (spec.get("expect_bytes") or {}).items():
            if (prep[key]["files"].get(name) or {}).get("bytes") != want:
                raise SystemExit(f"{key}: the prepared {name} is not the approved {want:,} bytes")
    if sum(prep["data"].get("rows") or [0]) != cfg["data"]["rows"]:
        raise SystemExit(f"the prepared dataset doesn't hold the registered {cfg['data']['rows']} rows")
    reg = cfg["registered"]
    if sorted(reg["targets"]) != sorted(k for k in keys if cfg["text"][k].get("graded")):
        raise SystemExit("config.yaml's registered targets must name exactly the graded models")
    plan = {"check": cfg["check"], "smoke": smoke, "switches": cfg["switches"], "data": cfg["data"],
            "vision": cfg["vision"], "text": {k: cfg["text"][k] for k in keys}, "protocol": cfg["protocol"],
            "diagnostics": cfg["diagnostics"], "registered": reg, "prepared": {k: prep[k] for k in need},
            "code": K.code_hashes(HERE), "pins": dict(PINS), "container_sha256": container_sha256()}
    if smoke:
        plan["smoke_settings"] = cfg["smoke"]
    return json.loads(json.dumps(plan))


def save(d: Path, name: str, obj) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1) + "\n")


def other_plan(d: Path, plan: dict) -> bool:
    p = d / "plan.json"
    return p.exists() and json.loads(p.read_text()) != plan


class Stop(Exception):
    """The laptop's stop: the estimated spend would pass the ceiling."""


class Spend:
    """K1's estimated spend, kept in spend.json in each results folder: each call's wall time on the laptop at
    config.yaml's rates (the GPU's and the cores and memory tier's), an overestimate, since it counts queueing and
    start-up. One ceiling covers all of K1: `others` are the other folders' spend.json (prepare's, the smoke's, the
    full run's), whose entries count toward the total. The checks are the hard ceiling's: the spend so far plus the
    next unit's upper estimate must be within the limit."""

    def __init__(self, d: Path, cfg: dict, others=()):
        self.path, self.cost = Path(d) / "spend.json", cfg["cost"]
        self.entries = json.loads(self.path.read_text())["entries"] if self.path.exists() else []
        self.others = [Path(p) for p in others]
        self.earlier = round(sum(e["usd"] for p in self.others if p.exists()
                                 for e in json.loads(p.read_text())["entries"]), 4)

    def total(self) -> float:
        """This folder's spend and the other folders', together."""
        return round(sum(e["usd"] for e in self.entries) + self.earlier, 4)

    def counted(self) -> str:
        return f", counting ${self.earlier:.2f} in {', '.join(str(p) for p in self.others if p.exists())}" \
            if self.earlier else ""

    def limit(self) -> float:
        return float(self.cost["limit_usd"])

    def rate(self, gpu: str | None, tier: str) -> float:
        return (self.cost["gpu_per_hour"][gpu] if gpu else 0.0) + self.cost["cpu_mem_per_hour"][tier]

    def upper_s(self, kind: str, key: str | None = None) -> float:
        u = self.cost["upper_s"][kind]
        return float(u.get(key, max(u.values())) if isinstance(u, dict) else u)

    def check(self, name: str, seconds: float, gpu: str | None, tier: str) -> None:
        """The hard ceiling before a container: Stop unless the spend so far plus `seconds` (its start-up and first
        unit) at these rates is within the limit."""
        usd = seconds / 3600 * self.rate(gpu, tier)
        if self.total() + usd > self.limit():
            raise Stop(f"stopped before {name}: the estimated spend ${self.total():.2f} plus its start-up and first "
                       f"unit's upper estimate ${usd:.2f} would pass the ${self.limit():.2f} ceiling for all of K1 "
                       f"(spend.json{self.counted()})")

    def timeout(self, name: str, gpu: str | None, tier: str) -> int:
        """The call's timeout: its registered one, cut to what the rest of the ceiling buys at these rates, less the
        start-up (check 3's cut, a backstop to the per-unit checks)."""
        left = (self.limit() - self.total()) / self.rate(gpu, tier) * 3600 - self.upper_s("startup")
        return max(1, min(int(self.cost["timeout_s"][name]), int(left)))

    def ceiling(self, gpu: str | None, tier: str) -> dict:
        """What a container's Ceiling reads: the spend and the limit at the call, the call's start, the rates and the
        upper estimates."""
        return {"spent_usd": self.total(), "limit_usd": self.limit(), "started_at": time.time(), "gpu": gpu,
                "gpu_usd_h": dict(self.cost["gpu_per_hour"]), "tier_usd_h": self.cost["cpu_mem_per_hour"][tier],
                "upper_s": self.cost["upper_s"]}

    def add(self, name: str, seconds: float, gpu: str | None, tier: str, timeout: int | None = None) -> None:
        usd = round(seconds / 3600 * self.rate(gpu, tier), 4)
        self.entries.append({"stage": name, "seconds": round(seconds), "gpu": gpu, "tier": tier, "usd": usd,
                             "timeout_s": timeout})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"entries": self.entries, "total_usd": self.total(),
                                         "counted_from_others_usd": self.earlier, "limit_usd": self.limit(),
                                         "note": "an estimate from wall time at config.yaml's rates; Modal's usage "
                                                 "page is the authority"}, indent=1) + "\n")
        print(f"spend: {name} about ${usd:.2f} ({seconds / 60:.1f} min{' on ' + gpu if gpu else ''}); K1 total about "
              f"${self.total():.2f} of the ${self.limit():.2f} ceiling (estimate){self.counted()}", flush=True)


def call(spend: Spend, name: str, fn, args: tuple, gpu: str | None, tier: str, first: str, key: str | None = None,
         ceiling: bool = True, timeout_name: str | None = None):
    """One Modal call from the laptop: the hard ceiling first (Stop unless the spend so far plus the container's
    start-up and its `first` unit's upper estimate fit), the timeout cut to what the ceiling leaves, the call (a GPU
    function at the GPU asked for, with expandable segments; with `ceiling`, the container's own Ceiling), and its
    spend recorded at the rate of the GPU the container reported and printed, even if it fails."""
    spend.check(name, spend.upper_s("startup") + spend.upper_s(first, key), gpu, tier)
    timeout = spend.timeout(timeout_name or name, gpu, tier)
    opts = {"timeout": timeout, **({"gpu": gpu, "env": GPU_ENV} if gpu else {})}
    f = fn.with_options(**opts)
    kw = {"ceiling": spend.ceiling(gpu, tier)} if ceiling else {}
    t0, res = time.time(), None
    try:
        res = f.remote(*args, **kw)
        return res
    finally:
        got = (res or {}).get("env", {}).get("gpu") if isinstance(res, dict) else None
        spend.add(name, time.time() - t0, gpu_key(got, gpu) if gpu else None, tier, timeout)


def stop_reason(name: str, e: Exception, spend: Spend) -> str | None:
    """Why a container stopped itself at the ceiling (a StageError carrying CheckError ceiling or budget), or None."""
    if isinstance(e, K.StageError) and "CheckError ceiling" in str(e):
        return (f"{name} didn't start its next unit: the spend so far plus that unit's upper estimate would pass the "
                f"${spend.limit():.2f} ceiling; its work so far is on the volume ({e})")
    if isinstance(e, K.StageError) and "CheckError budget" in str(e):
        return (f"{name} stopped itself at its first commit past the ${spend.limit():.2f} ceiling (a unit ran longer "
                f"than its upper estimate); its work so far is on the volume ({e})")
    return None


def explain(e: Exception) -> str:
    """A StageError with its CheckError code's meaning, for the terminal."""
    m = re.search(r"CheckError (\w+)", str(e))
    return f"{e} ({K.CODES[m.group(1)]})" if m and m.group(1) in K.CODES else str(e)


def finish(card, cfg: dict, d: Path, stop: str | None = None, spend: Spend | None = None) -> dict:
    """The card from whatever ran; after a stop, stopped.json first (the card carries it) and then SystemExit. A
    relaunch that runs to the end removes an earlier launch's stopped.json."""
    if stop:
        save(d, "stopped.json", {"reason": stop, "spend_usd": spend.total() if spend else None})
    else:
        (d / "stopped.json").unlink(missing_ok=True)
    c = card.write_card(cfg, d)
    print("\n".join(card.headline(c)), flush=True)
    print(f"wrote {d / 'card.json'}", flush=True)
    if stop:
        raise SystemExit(f"stopped: {stop}")
    return c


def fill_config(path: Path, prepared: dict) -> bool:
    """config.yaml's prepared block (its last block, from the line "prepared:" to the end) written from prepare's
    facts; everything above it stays as it is. The same facts again change nothing; different facts over a filled
    block are refused (prepare pins once). Returns whether the file changed."""
    import yaml

    text = path.read_text()
    cfg = yaml.safe_load(text)
    old = cfg.get("prepared")
    if old == prepared:
        return False
    if old is not None:
        raise SystemExit("config.yaml's prepared block differs from what prepare found now (a repository moved, or a "
                         "file changed): nothing was written. Compare prepare.json with config.yaml before deciding.")
    m = re.search(r"^prepared:.*\Z", text, re.M | re.S)
    if m is None:
        raise SystemExit("config.yaml has no prepared: line to fill")
    block = yaml.safe_dump({"prepared": prepared}, sort_keys=False, width=120, default_flow_style=None)
    new = text[:m.start()] + block
    if yaml.safe_load(new) != {**cfg, "prepared": prepared}:
        raise SystemExit("config.yaml would not read back as it was plus the prepared block: nothing was written")
    path.write_text(new)
    return True


@app.local_entrypoint()
def prepare(results: str, runs_results: str = ""):
    """Fetch and check every file on Modal (prepare_stage), then write the prepared block into config.yaml (to
    commit). results is prepare's own folder (<runs>_prepare); the runs' spend.json count toward the ceiling too."""
    import fetch

    card = load_card()
    cfg = card.load_config()
    root = card.results_path(results)
    runs = Path(runs_results).expanduser().resolve() if runs_results else card.runs_folder(root)
    spend = Spend(root, cfg, [runs / "spend.json", runs / "smoke" / "spend.json"])
    spec = {"repos": fetch.repos(cfg), "mb_s": cfg["cost"]["upper_s"]["prepare_mb_s"], "rows": cfg["data"]["rows"]}
    print(f"K1 prepare: {', '.join(r['repo'] for r in spec['repos'])} onto the {VOLUME_NAME} volume", flush=True)
    try:
        res = call(spend, "prepare", prepare_stage, (spec,), None, TIERS["prepare_stage"], "prepare_first")
    except Stop as e:
        raise SystemExit(f"stopped: {e}") from None
    except K.StageError as e:
        raise SystemExit(f"prepare stopped: {stop_reason('prepare', e, spend) or explain(e)}") from None
    save(root, "prepare.json", res)
    changed = fill_config(CONFIG, res["result"]["prepared"])
    facts = res["result"]["facts"]
    for key, r in res["result"]["prepared"].items():
        print(f"{key}: {r['repo']} at {r['revision']}, {len(r['files'])} files, "
              f"{sum(f['bytes'] for f in r['files'].values()):,} bytes", flush=True)
    print(f"rows {facts.get('data', {}).get('rows')}; images {json.dumps(facts.get('data', {}).get('images'))}",
          flush=True)
    print("config.yaml's prepared block " + ("written: commit it before the smoke" if changed else "unchanged"),
          flush=True)


@app.local_entrypoint()
def main(results: str, smoke: bool = False, gpu: str = "", prepare_results: str = ""):
    """The smoke (--smoke) or the full run, then the card. --gpu runs the stage on another GPU (A100-80GB or H100)
    under the same plan; --prepare-results names prepare's folder if it isn't <results>_prepare."""
    card = load_card()
    cfg = card.load_config()
    root = card.results_path(results)
    d = card.run_folder(root, smoke)
    prep = Path(prepare_results).expanduser().resolve() if prepare_results else card.prepare_folder(root)
    if not (prep / "spend.json").exists():
        raise SystemExit(f"no spend.json from prepare in {prep}: prepare's spend counts toward the $6 ceiling. Pass "
                         "--prepare-results with the folder prepare wrote to")
    gpu = gpu or GPU
    if gpu not in cfg["cost"]["gpu_per_hour"]:
        raise SystemExit(f"--gpu {gpu}: config.yaml's cost block has no rate for it")
    plan = make_plan(cfg, smoke)
    name = "smoke" if smoke else "full"
    if other_plan(d, plan):
        raise SystemExit(f"{d} holds the run of another plan (its plan.json differs from this one). Don't change "
                         "--results: K1's spend is counted per folder. To run this plan there, move its plan.json, "
                         f"{name}.json, card.json and stopped.json into a subfolder and leave spend.json in place, so "
                         "the spend it records still counts")
    save(d, "plan.json", plan)
    (d / "stopped.json").unlink(missing_ok=True)
    spend = Spend(d, cfg, [prep / "spend.json", card.run_folder(root, not smoke) / "spend.json"])
    print(f"K1 {name} run, plan {K.plan_id(plan)}: {out_dir(plan)} on the {VOLUME_NAME} volume, {gpu}", flush=True)
    res, stop = None, None
    try:
        res = call(spend, "stored", stored, (plan,), None, TIERS["stored"], "stored", ceiling=False)
        if res is None:
            fn = smoke_stage if smoke else full_stage
            res = call(spend, name, fn, (plan,), gpu, TIERS[f"{name}_stage"], "smoke_verify" if smoke else "verify_data")
    except Stop as e:
        stop = str(e)
    except K.StageError as e:
        stop = stop_reason(name, e, spend)
        if stop is None:
            raise SystemExit(f"{name} failed: {explain(e)}") from None
    if res is not None:
        save(d, f"{name}.json", res)
        if smoke and not res["result"]["passed"]:
            failed = [k for k, c in res["result"]["checks"].items() if not c["passed"]]
            stop = f"the smoke's checks failed: {', '.join(failed)} (smoke.json has their numbers)"
    return finish(card, cfg, d, stop, spend)


@app.local_entrypoint()
def card(results: str, smoke: bool = False):
    """The card again, from the files (no Modal calls)."""
    mod = load_card()
    cfg = mod.load_config()
    d = mod.run_folder(mod.results_path(results), smoke)
    c = mod.write_card(cfg, d)
    print("\n".join(mod.headline(c)), flush=True)
