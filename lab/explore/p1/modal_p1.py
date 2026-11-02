"""P1 on Modal (app anatta-explore-p1, one L40S): is the "self" a nuisance for language-vision agreement? spec.yaml holds
the design, the templates and the sighting criteria, written before any run. Claude (Opus 5.5) wrote this file, 1 Nov
2026. Private (lab/explore). An explore-track test: its result is a sighting, not a finding.

One container function, run_stage, does a stage (the smoke or the full run) for one text model:
  1. the data: WIT-1024's two parquet files on K1's volume checked by sha256, the first captions read (K1's read_wit,
     no images); the seven text conditions made from them (p1_core);
  2. the vision side: DINOv2's per-block CLS features from K1's full run (plan ecdb56d332b6), checked by sha256, reused;
  3. the text model through anatta.models.load from a read-only volume, as bench loads it; v_ego from its file (sha256
     checked); bench's random_matched controls built on the same model (seeds 0-3, bench's neutral prompts);
  4. checks on the first batch: P1's hooked features equal K1's embed_texts on the uncut model; under the v_ego cut,
     the cut reaches the features (and what output_hidden_states records is noted); the pass repeats bitwise;
  5. every condition (text condition x intervention): features under the intervention (p1_extract.embed_hooked, K1's
     protocol), then the scores by K1's code (measure.py): prh_score's own maximum, the by-layer table (K1's identity
     re-pairing), the score at the fixed layer pair (the plain, uncut condition's best) and on K1's 200 subsets of 800
     pairs at both pairs; for the plain, uncut condition also K1's whole record (measure.score_text: CKA, Huh's
     clipping, the permutation null's tau); for the uncut conditions the projections on v_ego.
Its result leaves the container only after k1.assert_text_free and p1_core.assert_no_captions: names, numbers, hashes
and paths. A copy goes to the anatta-explore-p1 volume. Any failure leaves as a StageError naming the step and the
exception's type (k1.run_guarded).

Volumes: anatta-hf-cache (read-only, /cache: Ministral's weights), anatta-calib-prh-k1 (read-only, /k1: WIT, DINOv2's
features, Qwen2.5-7B's weights), anatta-explore-p1 (scratch, /p1). Nothing else is touched.

Usage, with the matlab venv's modal CLI (no installs):
  M=~/prj/matlab/.venv/bin/modal; P=/Users/eugene/prj/align/anatta-lab/lab/explore/p1/modal_p1.py
  $M run $P::smoke [--model ministral]     64 pairs, two conditions, every check; read as nothing
  $M run $P::full [--model ministral]      1,024 pairs, all 42 conditions; then results.json and report.md
  $M run $P::analyze [--model ministral]   the analysis again from the saved raw result (no Modal call)
The laptop meters the spend (p1_spend.py) and stops before a call that would pass P1's $6 cap or the day's $45.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import p1_core as core  # noqa: E402

APP_NAME = "anatta-explore-p1"
SCRATCH = "anatta-explore-p1"  # P1's own small volume; no matlab-* app or volume is ever named here
GPU, CPU, MEMORY_MB = "L40S", 4.0, 32768
LOCAL = modal.is_local()
RESULTS = Path("/Users/eugene/prj/align/explore-results/2026-11-01/p1")
EXPLORE_RESULTS = Path("/Users/eugene/prj/align/explore-results")

# The h4lite image's pins in its order (bench's image; the same layer is reused), then pyarrow for WIT's parquet.
PINS = ("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0", "numpy==2.5.4",
        "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0", "tiktoken==0.14.0", "jinja2==3.1.6",
        "regex==2026.9.29", "peft==0.21.2", "accelerate==1.15.0")
CODE = {  # what the container imports: its name in /root -> the file on the laptop (copied in, never edited)
    "p1_core.py": HERE / "p1_core.py",
    "p1_extract.py": HERE / "p1_extract.py",
    "k1.py": core.K1_DIR / "k1.py",
    "measure.py": core.K1_DIR / "measure.py",
    "features.py": core.K1_DIR / "features.py",
    "steering.py": core.BENCH_DIR / "steering.py",
}
DIRECTIONS = {  # each model's v_ego file -> /root/directions/<model key>.pt (sha256 checked in the container)
    "ministral": Path("/Users/eugene/prj/align/anatta/directions/mistralai__Ministral-3-8B-Instruct-2512-BF16/ego.pt"),
    "qwen7b_base": Path("/Users/eugene/prj/align/anatta/directions/Qwen__Qwen2.5-7B-Instruct/ego.pt"),
}


def _src(laptop: Path, name: str) -> str:
    """A file's source for the image: its laptop path when local; in the container, where it already sits."""
    return str(laptop) if LOCAL else f"/root/{name}"


image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PINS).uv_pip_install("pyarrow==21.0.0")
         .env({"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/anatta_src:/root"}))
image = image.add_local_dir(_src(core.ANATTA_SRC, "anatta_src"), "/root/anatta_src", ignore=["**/__pycache__"])
for _name, _path in CODE.items():
    image = image.add_local_file(_src(_path, _name), f"/root/{_name}")
for _key, _path in DIRECTIONS.items():
    image = image.add_local_file(_src(_path, f"directions/{_key}.pt"), f"/root/directions/{_key}.pt")
app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("anatta-hf-cache")
k1_vol = modal.Volume.from_name("anatta-calib-prh-k1")
scratch = modal.Volume.from_name(SCRATCH, create_if_missing=True)


# ------------------------------------------------------------------------------------------------ the container


def stage_body(run: dict, t_start: float, guard) -> dict:
    import features as F
    import measure as M
    import p1_extract as X
    import steering
    import torch
    import transformers
    from anatta.dial import intervene
    from anatta.models import load
    from anatta.probe import Direction

    K = core.K
    spec, mk = run["spec"], run["model_key"]
    ms, prot = spec["models"][mk], spec["protocol"]
    batch, max_len, k, pairs = int(prot["batch"]), int(prot["max_length"]), int(prot["k"]), int(run["pairs"])
    sub, nul = run["subsample"], run["null"]
    rec = {"stage": run["stage"], "run_id": run["run_id"], "model_key": mk, "pairs": pairs, "subsample": sub,
           "null": nul, "checks": {}, "timing": {}, "conditions": {}}

    def lap(name, t0):
        rec["timing"][name] = round(time.time() - t0, 2)

    guard.row = "code"
    rec["code_sha256"] = {n: K.sha256_file(f"/root/{n}") for n in run["code_sha256"]}
    if rec["code_sha256"] != run["code_sha256"]:
        raise core.P1Check("sha256", what=1)
    rec["env"] = {"gpu": K.gpu_name(), "torch": torch.__version__, "transformers": transformers.__version__,
                  "precision": K.precision_flags()}

    guard.row = "data"
    t0 = time.time()
    snapd = K.snapshot_dir(spec["data"]["repo"], spec["data"]["revision"], "dataset", hf_home=spec["data"]["hf_home"])
    rec["verify_data"] = K.verify_files(snapd, spec["data"]["files"])
    _, captions = F.read_wit([snapd / n for n in sorted(spec["data"]["files"])], None, images=False)
    if len(captions) != int(spec["data"]["rows"]):
        raise core.P1Check("rows", rows=len(captions))
    captions = captions[:pairs]
    texts, rec["texts"] = core.condition_texts(captions, spec)
    leak = captions + [x for v in texts.values() for x in v]  # every text the result must not hold
    lap("data", t0)

    guard.row = "vision"
    vf = Path(spec["vision"]["features"])
    vsha = K.sha256_file(vf)
    if vsha != spec["vision"]["sha256"] or json.loads(Path(spec["vision"]["facts"]).read_text()).get("sha256") != vsha:
        raise core.P1Check("sha256", what=2)
    fa = torch.load(vf, weights_only=True)
    want = (int(spec["data"]["rows"]), int(spec["vision"]["dim"]))
    if len(fa) != int(spec["vision"]["layers"]) or any(tuple(f.shape) != want for f in fa):
        raise core.P1Check("shape", what=1)
    fa = [f[:pairs].contiguous().float() for f in fa]
    rec["vision"] = {"sha256": vsha, "layers": len(fa), "rows": pairs}

    guard.row = "model"
    t0 = time.time()
    if ms.get("files"):  # the weights' sha256s, where the spec pins them (K1's prepared block for Qwen)
        rec["verify_model"] = K.verify_files(ms["snapshot"], ms["files"])
    model, tok = load(ms["snapshot"])
    body, layers = X.decoder(model)
    n_layers, hidden = len(layers), int(body.config.hidden_size)
    if n_layers != int(ms["layers"]) or hidden != int(ms["hidden"]):
        raise core.P1Check("shape", layers=n_layers, hidden=hidden)
    dev = next(body.parameters()).device
    one = list(tok("a")["input_ids"])
    rec["model"] = {"class": type(model).__name__, "decoder": type(body).__name__, "layers": n_layers, "hidden": hidden,
                    "dtype": str(next(body.parameters()).dtype).replace("torch.", ""),
                    "attn_implementation": body.config._attn_implementation,
                    "tokenizer": {"class": type(tok).__name__, "padding_side": tok.padding_side,
                                  "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
                                  "bos_token_id": tok.bos_token_id, "pad_is_eos": tok.pad_token_id == tok.eos_token_id,
                                  "adds_bos": tok.bos_token_id is not None and one[:1] == [tok.bos_token_id],
                                  "vocab": len(tok)}}
    lap("model", t0)

    guard.row = "templates"
    pre = {name: X.prefix_ids(tok, core.prefix(tpl)) for name, tpl in spec["templates"].items()}
    starts = {name: X.caption_starts(tok, texts[name], pre[name], max_len) for name in spec["templates"]}
    rec["templates"] = {}
    for name in spec["templates"]:
        lens = [len(x) for x in tok(texts[name], truncation=False)["input_ids"]]
        rec["templates"][name] = {"prefix_tokens": len(pre[name]),
                                  "clean_boundary": sum(s is not None for s in starts[name]),
                                  "mean_tokens": sum(lens) / len(lens),
                                  **F.token_facts(tok, texts[name], batch, max_len)}

    guard.row = "direction"
    dpath = Path(f"/root/directions/{mk}.pt")
    if K.sha256_file(dpath) != ms["direction"]["sha256"]:
        raise core.P1Check("sha256", what=3)
    ego = Direction.load(dpath)
    dl = core.direction_layers(ms)
    if sorted(int(x) for x in ego.vectors) != dl:
        raise core.P1Check("shape", what=2)
    ego_v = {x: ego.vectors[x].detach().float().flatten().cpu() for x in dl}
    if any(v.numel() != hidden for v in ego_v.values()):
        raise core.P1Check("shape", what=3)
    rec["direction"] = {"sha256": ms["direction"]["sha256"], "layers": [dl[0], dl[-1]], "best_layer": int(ego.best_layer),
                        "pos_mean": {str(x): float(ego.pos_mean[x]) for x in dl},
                        "neg_mean": {str(x): float(ego.neg_mean[x]) for x in dl},
                        "scores": {str(x): float(ego.scores[x]) for x in dl}}

    guard.row = "controls"
    t0 = time.time()
    svc = steering.Service(model, tok, model_key=mk, model_id=ms["hf_id"], revision=ms["revision"])
    payload = {"target": {"name": "ego", "sha256": ms["direction"]["sha256"],
                          "vectors": {x: v.numpy() for x, v in ego_v.items()}},
               "seeds": [int(s) for s in spec["controls"]["seeds"]], "layers": dl,
               "neutral_prompts": run["neutral"]["prompts"], "neutral_sha256": run["neutral"]["sha256"],
               "tol": float(spec["controls"]["tol"])}
    try:
        built = svc.random_matched(payload)
    except steering.MatchError:
        raise core.P1Check("control") from None
    controls, rec["controls"] = {}, {}
    for d in built["directions"]:
        name = f"rand{int(d['seed'])}"
        controls[name] = {int(x): torch.as_tensor(v).float().flatten() for x, v in d["vectors"].items()}
        rec["controls"][name] = {"seed": int(d["seed"]), "made": core.sanitize_made(d["made"]),
                                 "cos_ego": {str(x): float(X.unit(v) @ X.unit(ego_v[x])) for x, v in controls[name].items()},
                                 "vectors": {str(x): v.tolist() for x, v in controls[name].items()}}
    del svc
    lap("controls", t0)

    guard.row = "checks"
    t0 = time.time()
    first = texts["plain"][:batch]
    mine, _ = X.embed_hooked(model, tok, first, batch, max_len)
    k1s = F.embed_texts(body, tok, first, batch, max_len)
    same = len(mine) == len(k1s) == n_layers + 1 and all(torch.equal(a, b) for a, b in zip(mine, k1s))
    rec["checks"]["equivalence"] = {"passed": bool(same), "states": len(mine),
                                    "max_abs": max(float((a - b).abs().max()) for a, b in zip(mine, k1s))}
    if not same:
        raise core.P1Check("equivalence")
    base = steering.count_hooks(model)  # after K1's call, which installed transformers' own recording hooks
    ego_ivs = steering.build_interventions("ablate", ego_v, dl)
    with intervene(model, ego_ivs):
        cut_mine, _ = X.embed_hooked(model, tok, first, batch, max_len)
        cut_hf = F.embed_texts(body, tok, first, batch, max_len)
    if steering.count_hooks(model) != base:
        raise core.P1Check("hooks", where=1)
    r_mine, r_hf = {}, {}
    for x in dl:
        u = X.unit(ego_v[x])
        ref = float((mine[x + 1] @ u).abs().mean())
        if ref > 1e-3:
            r_mine[str(x)] = float((cut_mine[x + 1] @ u).abs().mean()) / ref
            r_hf[str(x)] = float((cut_hf[x + 1] @ u).abs().mean()) / ref
    cut_ok = bool(r_mine) and max(r_mine.values()) < 0.5
    rec["checks"]["cut_in_features"] = {"passed": cut_ok, "max_ratio": max(r_mine.values(), default=None),
                                        "ratio_by_layer": r_mine}
    rec["checks"]["output_hidden_states_under_cut"] = {"min_ratio": min(r_hf.values(), default=None),
                                                       "max_ratio": max(r_hf.values(), default=None),
                                                       "ratio_by_layer": r_hf}
    if not cut_ok:
        raise core.P1Check("hooks", where=2)
    again, _ = X.embed_hooked(model, tok, first, batch, max_len)
    rec["checks"]["repeatable"] = {"passed": all(torch.equal(a, b) for a, b in zip(mine, again))}
    del mine, k1s, cut_mine, cut_hf, again
    lap("checks", t0)

    guard.row = "conditions"
    order = list(run["conditions"])
    if order[0] != core.condition("plain", "none"):
        raise core.P1Check("fixed_pair")
    ivs = {"none": [], "ego": ego_ivs,
           **{name: steering.build_interventions("ablate", v, dl) for name, v in controls.items()}}
    pa = [M._prh_prep(f.to(dev)) for f in fa]
    rows = core.subset_rows(pairs, int(sub["n"]), int(sub["size"]), int(sub["seed"]))
    pl = int(ms["projection_layer"])
    by_iv: dict = {}
    for name in order:
        by_iv.setdefault(core.split_condition(name)[1], []).append(name)
    fixed = None
    for iv, names in by_iv.items():
        with intervene(model, ivs[iv]):
            for name in names:
                guard.row = name
                t0 = time.time()
                tname = core.split_condition(name)[0]
                project = ego_v if iv == "none" else {pl: ego_v[pl]}
                feats, cap = X.embed_hooked(model, tok, texts[tname], batch, max_len, project=project,
                                            starts=starts[tname])
                F.check_layers(feats, n_layers + 1, hidden, pairs)
                t1 = time.time()
                c = core.score_condition(fa, pa, feats, rows, k, fixed, dev)
                if fixed is None:
                    fixed = tuple(c["own"]["layers"])
                    c["fixed"]["layers"] = list(fixed)
                pooled = X.pooled_projection(feats, project)
                if iv == "none":
                    c["projection"] = {"pooled_mean": {str(x): float(v.mean()) for x, v in pooled.items()},
                                       "caption_mean": {str(x): float(torch.nanmean(v)) for x, v in cap.items()},
                                       "pooled_at": pooled[pl].tolist(), "caption_at": core.nan_none(cap[pl].tolist()),
                                       "caption_missing": int(torch.isnan(cap[pl]).sum())}
                else:
                    c["projection_at"] = {"pooled_abs_mean": float(pooled[pl].abs().mean()),
                                          "pooled_mean": float(pooled[pl].mean()),
                                          "caption_mean": float(torch.nanmean(cap[pl]))}
                if name == core.condition("plain", "none"):
                    settings = {"k": k, "quantile": prot["quantile"], "null": dict(nul), "subsample": dict(sub)}
                    c["k1_record"], _ = M.score_text(fa, feats, settings)
                    if ms.get("k1_features"):  # the reproduction of K1's stored features (the Qwen arm)
                        kf = Path(ms["k1_features"])
                        if K.sha256_file(kf) != ms["k1_features_sha256"]:
                            raise core.P1Check("sha256", what=4)
                        kfe = [f[:pairs] for f in torch.load(kf, weights_only=True)]
                        rec["checks"]["k1_features"] = {
                            "identical": len(kfe) == len(feats) and all(torch.equal(a, b) for a, b in zip(feats, kfe)),
                            "max_abs": max(float((a - b).abs().max()) for a, b in zip(feats, kfe)),
                            "own": c["own"], "k1": ms["k1_score"]}
                c["timing"] = {"features": round(t1 - t0, 2), "scores": round(time.time() - t1, 2)}
                rec["conditions"][name] = c
                del feats, cap
                torch.cuda.empty_cache()
        if steering.count_hooks(model) != base:
            raise core.P1Check("hooks", where=3)
    rec["fixed_pair"] = list(fixed)
    rec["timing"]["total"] = round(time.time() - t_start, 2)

    guard.row = "leaving"
    out = core.leaving(rec, leak)
    d = Path("/p1/runs") / run["stage"] / run["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(out))
    return out


@app.function(gpu=GPU, cpu=CPU, memory=MEMORY_MB, timeout=2700, scaledown_window=2, max_containers=1,
              volumes={"/cache": hf_cache.read_only(), "/k1": k1_vol.read_only(), "/p1": scratch})
def run_stage(run: dict) -> dict:
    """One P1 stage (the module docstring); every exception leaves as k1's StageError (the step and the type)."""
    t_start = time.time()
    try:
        return core.K.run_guarded(f"p1_{run['stage']}", stage_body, run, t_start)
    finally:
        scratch.commit()


# ------------------------------------------------------------------------------------------------ the laptop


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def conditions_for(spec: dict, stage: str) -> list[str]:
    """The run's conditions in run order: the plain, uncut condition first (it fixes the layer pair), then by
    intervention in the spec's order and text condition in the spec's order."""
    if stage == "smoke":
        return list(spec["smoke"]["conditions"])
    names = [core.condition(t, i) for i in spec["interventions"] for t in spec["templates"]]
    first = core.condition("plain", "none")
    return [first] + [n for n in names if n != first]


def model_spec(spec: dict, model_key: str) -> dict:
    """The model's block for the run; for a model on K1's volume, K1's prepared file list (bytes and sha256) too."""
    ms = dict(spec["models"][model_key])
    if ms["volume"] == "anatta-calib-prh-k1":
        import yaml

        k1cfg = yaml.safe_load((core.K1_DIR / "config.yaml").read_text())
        key = {"qwen7b_base": "qwen7b"}[model_key]
        if k1cfg["prepared"][key]["revision"] != ms["revision"]:
            raise SystemExit(f"{model_key}: K1's prepared revision is not the spec's")
        ms["files"] = k1cfg["prepared"][key]["files"]
    return ms


def folder(model_key: str) -> Path:
    return RESULTS if model_key == "ministral" else RESULTS / model_key


def log_variants(spec_sha: str, run_id: str, stage: str, model_key: str, conditions: list[str], status: str) -> None:
    """Every condition to explore-results/variants.jsonl (scout's format, tool "p1"), and the run to P1's own list."""
    from scout.schema import append_jsonl, utc_now

    sweep = f"p1-{stage}" + ("" if model_key == "ministral" else f"-{model_key}")
    for c in conditions:
        append_jsonl(EXPLORE_RESULTS / "variants.jsonl", {"ts": utc_now(), "tool": "p1", "sweep": sweep,
                                                          "run_id": run_id, "kind": "condition", "name": c,
                                                          "sha256": spec_sha})
    append_jsonl(RESULTS / "runs.jsonl", {"ts": utc_now(), "run_id": run_id, "stage": stage, "model": model_key,
                                          "conditions": len(conditions), "status": status, "spec_sha256": spec_sha})


def run_local(stage: str, model_key: str) -> dict:
    """One stage from the laptop: the files checked, the spend reserved, the call, the spend trued up, the raw result
    saved; after the full run, the analysis (results.json, report.md)."""
    import p1_spend as S
    from scout.schema import canonical, sha256_text, utc_now

    spec_bytes = core.SPEC_FILE.read_bytes()
    spec_sha = sha256_bytes(spec_bytes)
    spec = core.load_spec()
    if model_key not in spec["models"]:
        raise SystemExit(f"--model {model_key}: the spec has {sorted(spec['models'])}")
    out = folder(model_key)
    sdir = out / stage
    sdir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().replace("-", "").replace(":", "")
    run_id = f"p1-{stage}-{model_key}-{stamp}"
    ms = model_spec(spec, model_key)
    if sha256_bytes(DIRECTIONS[model_key].read_bytes()) != ms["direction"]["sha256"] or \
            str(DIRECTIONS[model_key]) != ms["direction"]["file"]:
        raise SystemExit(f"{DIRECTIONS[model_key]}: not the spec's direction file (path or sha256)")
    prompts = json.loads(Path(spec["controls"]["neutral_prompts"]).read_text())["prompts"]
    conditions = conditions_for(spec, stage)
    run = {"stage": stage, "run_id": run_id, "model_key": model_key,
           "spec": {**spec, "models": {model_key: ms}}, "pairs": spec["smoke"]["pairs"] if stage == "smoke" else
           spec["data"]["rows"], "conditions": conditions,
           "subsample": spec["smoke"]["subsample"] if stage == "smoke" else spec["measure"]["subsample"],
           "null": spec["smoke"]["null"] if stage == "smoke" else
           {k: v for k, v in spec["measure"]["null"].items() if k != "condition"},
           "neutral": {"prompts": prompts, "sha256": sha256_text(canonical(prompts))},
           "code_sha256": {n: sha256_bytes(p.read_bytes()) for n, p in CODE.items()}}
    (sdir / "spec.yaml").write_bytes(spec_bytes)
    (RESULTS / "spec.yaml").write_bytes(spec_bytes)
    cost = spec["cost"]
    meter = S.P1Meter(spend_dir=EXPLORE_RESULTS / "spend", log=RESULTS / "spend.jsonl", label=f"p1-{model_key}",
                      session=run_id, rate=S.p1_rate(cost["rate"]), cap_usd=cost["cap_usd"],
                      daily_stop_usd=cost["daily_stop_usd"])
    est = float(cost["estimate_s"]["startup"]) + float(cost["estimate_s"][stage])
    meter.reserve(est, stage)
    timeout = int(max(60, min(float(cost["timeout_s"][stage]), est + meter.left_s() - cost["estimate_s"]["startup"])))
    print(f"[p1] {run_id}: {len(conditions)} conditions on {run['pairs']} pairs, {model_key}; reserved "
          f"${meter.usd(est):.2f}, timeout {timeout}s; P1 so far ${meter.p1_spent():.2f} of ${meter.cap:.2f}", flush=True)
    res, err = None, None
    try:
        res = run_stage.with_options(timeout=timeout).remote(run)
    except Exception as e:  # noqa: BLE001 (a StageError names only the step and the type)
        err = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        usd = meter.settle(stage, model=model_key, run_id=run_id, ok=res is not None)
        print(f"[p1] this call about ${usd:.3f}; P1 total ${meter.p1_spent():.3f}; today ${meter.today():.3f}",
              flush=True)
    log_variants(spec_sha, run_id, stage, model_key, conditions, "complete" if res is not None else "error")
    if res is None:
        (sdir / "error.json").write_text(json.dumps({"run_id": run_id, "error": err}, indent=1) + "\n")
        raise SystemExit(f"[p1] {stage} failed: {err}")
    res["laptop"] = {"spec_sha256": spec_sha, "call_usd": round(usd, 4), "finished": utc_now()}
    (sdir / "raw.json").write_text(json.dumps(res) + "\n")
    print(f"[p1] saved {sdir / 'raw.json'}", flush=True)
    import p1_analysis as A

    if stage == "smoke":
        print(json.dumps(A.smoke_summary(res, spec), indent=1), flush=True)
    else:
        A.write_all(res, spec, model_key, out, spec_sha)
    return res


@app.local_entrypoint()
def smoke(model: str = "ministral"):
    run_local("smoke", model)


@app.local_entrypoint()
def full(model: str = "ministral"):
    run_local("full", model)


@app.local_entrypoint()
def analyze(model: str = "ministral"):
    import p1_analysis as A

    out = folder(model)
    res = json.loads((out / "full" / "raw.json").read_text())
    spec = core.load_spec(out / "full" / "spec.yaml")
    A.write_all(res, spec, model, out, res["laptop"]["spec_sha256"])
