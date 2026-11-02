"""P1(b) again, with difference-vector controls (app anatta-explore-p1b, one L40S). Claude (Opus 5.5) wrote this file, 1
Nov 2026, evening. Private (lab/explore). Explore track, unregistered: a sighting at most, not a finding.

P1's (b) came out reversed: on Ministral the v_ego cut lowered language-vision agreement more than four random_matched
cuts, for plain, first-person and neutral captions alike. P1's report named the gap: random directions are matched on
token-level variance over bench's neutral chat prompts, not on what the score reads, and difference vectors carry a
large shared component random ones lack. This file asks whether the ego cut still costs more than cuts along other
difference vectors at the same layers (8-32): bench's other_trait directions (formality, verbosity, sage_persona, full
cuts) and bench's split_null directions (seeds 0-2) from the pinned bank B.

Everything else is P1's: spec.yaml unchanged (K1's protocol: batches of 64, max length 512, the masked mean of every
state, mutual 10-NN against DINOv2 at the fixed layer pair, K1's 200 subsets of 800 pairs for the paired intervals),
P1's image recipe (the same pins and the same code files, read from their places and never edited), p1_core and
p1_extract imported unchanged, captions kept on Modal by p1_core.leaving.

Conditions: plain|none (fixes the layer pair), fp1|none and fp1|ego: the reproduction checks against P1's stored
raw.json; then plain, fp1, fp2 and fp3 under each control cut. P1's text features were not cached, but its scores and
subset vectors were: the none and ego conditions are reused from P1's raw.json once the three checks match it. Under
the uncut conditions the pooled caption features' projections on every direction at every cut layer are recorded (what
each cut can remove from the features the score reads), and under each cut the cut's reach (P1's 0.5 rule).

The controls are made on the laptop, from bench's own code: the registry's trait files (directions.resolve, tensor
hash checked) and controls.split_null on bank B (tensor hash checked), so they are the vectors the bench uses for the
lead 6 rerun (the run records both hashes). They travel in the call's payload; their float32 bytes are hashed on both
sides.

Usage, with the matlab venv's modal CLI:
  M=~/prj/matlab/.venv/bin/modal; P=/Users/eugene/prj/align/anatta-lab/lab/explore/p1/modal_p1b.py
  $M run $P::smoke     64 pairs, plain|none, fp1|ego, plain|formality, fp1|null_s0; read as nothing
  $M run $P::full      1,024 pairs, 27 conditions; then analysis.json beside raw.json
  $M run $P::analyze   the analysis again from the saved raw result (no Modal call)
The laptop meters the spend with P1's meter (its own log, cap $1.50) and stops before a call that would pass it.
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

APP_NAME = "anatta-explore-p1b"  # its own app; no matlab-* app or volume is ever named here
GPU, CPU, MEMORY_MB = "L40S", 4.0, 32768
LOCAL = modal.is_local()
OUT = Path("/Users/eugene/prj/align/explore-results/2026-11-01/split-reruns/p1b")
EXPLORE_RESULTS = Path("/Users/eugene/prj/align/explore-results")
P1_RAW = Path("/Users/eugene/prj/align/explore-results/2026-11-01/p1/full/raw.json")
BANK_B = {"path": "/Users/eugene/prj/align/explore-results/2026-11-01/bench-smoke-controls/directions/"
                  "split_null_bank-ministral-60782d12.pt",
          "tensor_sha256": "60782d12f9eb97a3d2a53b3060e690f2b7c2893dccf2192f1290ffa5cb2e47c7"}
TRAITS = ("formality", "verbosity", "sage_persona")
NULL_SEEDS = (0, 1, 2)
TEXTS = ("plain", "fp1", "fp2", "fp3")
CAP_USD = 1.5
ESTIMATE_S = {"startup": 300, "smoke": 240, "full": 900, "fullall": 1000}
TIMEOUT_S = {"smoke": 900, "full": 1800, "fullall": 1800}
BENCH_RUNS = [Path("/Users/eugene/prj/align/explore-results/2026-11-01") / s / "directions"
              for s in ("lead6-split", "lead6-split-smoke")]  # where the bench saved its own copies

# P1's image, as modal_p1.py builds it (the h4lite pins in their order, then pyarrow for WIT's parquet).
PINS = ("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0", "numpy==2.5.4",
        "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0", "tiktoken==0.14.0", "jinja2==3.1.6",
        "regex==2026.9.29", "peft==0.21.2", "accelerate==1.15.0")
CODE = {
    "p1_core.py": HERE / "p1_core.py",
    "p1_extract.py": HERE / "p1_extract.py",
    "k1.py": core.K1_DIR / "k1.py",
    "measure.py": core.K1_DIR / "measure.py",
    "features.py": core.K1_DIR / "features.py",
    "steering.py": core.BENCH_DIR / "steering.py",
}
EGO_FILE = Path("/Users/eugene/prj/align/anatta/directions/mistralai__Ministral-3-8B-Instruct-2512-BF16/ego.pt")


def _src(laptop: Path, name: str) -> str:
    return str(laptop) if LOCAL else f"/root/{name}"


image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PINS).uv_pip_install("pyarrow==21.0.0")
         .env({"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/anatta_src:/root"}))
image = image.add_local_dir(_src(core.ANATTA_SRC, "anatta_src"), "/root/anatta_src", ignore=["**/__pycache__"])
for _name, _path in CODE.items():
    image = image.add_local_file(_src(_path, _name), f"/root/{_name}")
image = image.add_local_file(_src(EGO_FILE, "directions/ministral.pt"), "/root/directions/ministral.pt")
app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("anatta-hf-cache")
k1_vol = modal.Volume.from_name("anatta-calib-prh-k1")


def bytes_sha(vectors: dict) -> str:
    """sha256 over the layers in order: each layer as 4 little-endian bytes, then its vector's float32 bytes."""
    import numpy as np

    h = hashlib.sha256()
    for x in sorted(int(k) for k in vectors):
        h.update(int(x).to_bytes(4, "little"))
        h.update(np.ascontiguousarray(np.asarray(vectors[x] if x in vectors else vectors[str(x)],
                                                 dtype=np.float32).ravel()).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the container


def body_b(run: dict, t_start: float, guard) -> dict:
    import features as F
    import measure as M
    import numpy as np
    import p1_extract as X
    import steering
    import torch
    import transformers
    from anatta.dial import intervene
    from anatta.models import load
    from anatta.probe import Direction

    K = core.K
    spec, mk = run["spec"], "ministral"
    ms, prot = spec["models"][mk], spec["protocol"]
    batch, max_len, k, pairs = int(prot["batch"]), int(prot["max_length"]), int(prot["k"]), int(run["pairs"])
    sub = run["subsample"]
    rec = {"stage": run["stage"], "run_id": run["run_id"], "model_key": mk, "pairs": pairs, "subsample": sub,
           "checks": {}, "timing": {}, "conditions": {}, "controls": {}}

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
    texts = {t: texts[t] for t in TEXTS}
    leak = captions + [x for v in texts.values() for x in v]
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

    guard.row = "model"
    t0 = time.time()
    model, tok = load(ms["snapshot"])
    body, layers = X.decoder(model)
    n_layers, hidden = len(layers), int(body.config.hidden_size)
    if n_layers != int(ms["layers"]) or hidden != int(ms["hidden"]):
        raise core.P1Check("shape", layers=n_layers, hidden=hidden)
    dev = next(body.parameters()).device
    rec["model"] = {"class": type(model).__name__, "decoder": type(body).__name__, "layers": n_layers, "hidden": hidden,
                    "dtype": str(next(body.parameters()).dtype).replace("torch.", "")}
    lap("model", t0)

    guard.row = "templates"
    rec["templates"] = {}
    for name in TEXTS:
        pre = X.prefix_ids(tok, core.prefix(spec["templates"][name]))
        starts = X.caption_starts(tok, texts[name], pre, max_len)
        rec["templates"][name] = {"prefix_tokens": len(pre), "clean_boundary": sum(s is not None for s in starts),
                                  **F.token_facts(tok, texts[name], batch, max_len)}

    guard.row = "direction"
    dpath = Path("/root/directions/ministral.pt")
    if K.sha256_file(dpath) != ms["direction"]["sha256"]:
        raise core.P1Check("sha256", what=3)
    ego = Direction.load(dpath)
    dl = core.direction_layers(ms)
    if sorted(int(x) for x in ego.vectors) != dl:
        raise core.P1Check("shape", what=2)
    ego_v = {x: ego.vectors[x].detach().float().flatten().cpu() for x in dl}

    guard.row = "controls"
    dirs = {"ego": ego_v}
    for name, c in run["controls"].items():
        vecs = {int(x): torch.as_tensor(np.asarray(v, dtype=np.float32)).flatten() for x, v in c["vectors"].items()}
        if sorted(vecs) != dl or any(v.numel() != hidden for v in vecs.values()):
            raise core.P1Check("shape", what=4)
        got = bytes_sha({x: v.numpy() for x, v in vecs.items()})
        if got != c["bytes_sha256"]:
            raise core.P1Check("sha256", what=5)
        dirs[name] = vecs
        rec["controls"][name] = {"bytes_sha256": got,
                                 "cos_ego": {str(x): float(X.unit(v) @ X.unit(ego_v[x])) for x, v in vecs.items()}}

    guard.row = "checks"
    t0 = time.time()
    first = texts["plain"][:batch]
    mine, _ = X.embed_hooked(model, tok, first, batch, max_len)
    k1s = F.embed_texts(body, tok, first, batch, max_len)
    same = len(mine) == len(k1s) == n_layers + 1 and all(torch.equal(a, b) for a, b in zip(mine, k1s))
    rec["checks"]["equivalence"] = {"passed": bool(same), "states": len(mine)}
    if not same:
        raise core.P1Check("equivalence")
    base = steering.count_hooks(model)
    del mine, k1s
    lap("checks", t0)

    guard.row = "conditions"
    order = list(run["conditions"])
    if order[0] != core.condition("plain", "none"):
        raise core.P1Check("fixed_pair")
    ivs = {name: ([] if name == "none" else steering.build_interventions("ablate", v, dl))
           for name, v in {"none": None, **dirs}.items()}
    pa = [M._prh_prep(f.to(dev)) for f in fa]
    rows = core.subset_rows(pairs, int(sub["n"]), int(sub["size"]), int(sub["seed"]))
    by_iv: dict = {}
    for name in order:
        by_iv.setdefault(core.split_condition(name)[1], []).append(name)
    fixed, uncut_abs = None, {}
    for iv, names in by_iv.items():
        with intervene(model, ivs[iv]):
            for name in names:
                guard.row = name
                t0 = time.time()
                tname = core.split_condition(name)[0]
                feats, _ = X.embed_hooked(model, tok, texts[tname], batch, max_len)
                F.check_layers(feats, n_layers + 1, hidden, pairs)
                t1 = time.time()
                c = core.score_condition(fa, pa, feats, rows, k, fixed, dev)
                if fixed is None:
                    fixed = tuple(c["own"]["layers"])
                    c["fixed"]["layers"] = list(fixed)
                if iv == "none":  # what each direction could remove from the pooled features the score reads
                    stats = {}
                    for dname, vecs in dirs.items():
                        proj = X.pooled_projection(feats, vecs)
                        stats[dname] = {str(x): {"mean": float(p.double().mean()), "var": float(p.double().var()),
                                                 "m2": float((p.double() ** 2).mean())} for x, p in proj.items()}
                        uncut_abs[(tname, dname)] = {x: float(p.abs().mean()) for x, p in proj.items()}
                    c["feature_stats"] = stats
                else:  # the cut's reach into the features: |projection| under the cut over the uncut one
                    proj = X.pooled_projection(feats, dirs[iv])
                    c["cut_abs_mean"] = {str(x): float(p.abs().mean()) for x, p in proj.items()}
                    ref = uncut_abs.get((tname, iv))
                    if ref is not None:
                        ratio = {str(x): float(proj[x].abs().mean()) / ref[x] for x in proj if ref[x] > 1e-3}
                        c["cut_reach"] = {"max_ratio": max(ratio.values(), default=None), "ratio_by_layer": ratio}
                        if not ratio or max(ratio.values()) >= 0.5:
                            raise core.P1Check("hooks", where=2)
                c["timing"] = {"features": round(t1 - t0, 2), "scores": round(time.time() - t1, 2)}
                rec["conditions"][name] = c
                del feats
                torch.cuda.empty_cache()
        if steering.count_hooks(model) != base:
            raise core.P1Check("hooks", where=3)
    rec["fixed_pair"] = list(fixed)
    rec["timing"]["total"] = round(time.time() - t_start, 2)
    guard.row = "leaving"
    return core.leaving(rec, leak)


@app.function(gpu=GPU, cpu=CPU, memory=MEMORY_MB, timeout=2700, scaledown_window=2, max_containers=1,
              volumes={"/cache": hf_cache.read_only(), "/k1": k1_vol.read_only()})
def run_b(run: dict) -> dict:
    """One P1(b) stage; every exception leaves as k1's StageError (the step and the type)."""
    return core.K.run_guarded(f"p1b_{run['stage']}", body_b, run, time.time())


# ------------------------------------------------------------------------------------------------ the laptop


def make_controls(ms: dict) -> tuple[dict, dict]:
    """The six control directions at the ego cut's layers, from bench's code: (payload, record). The record holds each
    one's bench tensor hash, its bytes hash, where it came from, and whether the bench's own saved copy (lead6-split
    or its smoke) has the same tensor hash."""
    import controls as C
    import directions as D

    dl = core.direction_layers(ms)
    ego = D.resolve("ego", "ministral")
    if ego.sha256 != ms["direction"]["sha256"] or sorted(ego.vectors) != dl:
        raise SystemExit("bench's registry v_ego is not P1's (file hash or layers)")
    payload, record = {}, {"ego": {"tensor_sha256": ego.tensor_sha256, "sha256": ego.sha256}}
    for t in TRAITS:
        d = D.resolve(t, "ministral")
        vecs = {x: d.vector(x) for x in d.layers_for(dl)}
        payload[t] = {"vectors": vecs, "bytes_sha256": bytes_sha(vecs)}
        record[t] = {"kind": "other_trait", "match": "full", "tensor_sha256": d.tensor_sha256, "sha256": d.sha256,
                     "path": d.path, "registered": d.registered}
    bank = C.load_bank(BANK_B)
    chance = C.chance(bank, ego, dl)
    for s in NULL_SEEDS:
        vecs, made = C.split_null(bank, s, dl, target=ego, chance_table=chance)
        name, tsha = f"null_s{s}", D.tensor_sha256(vecs)
        payload[name] = {"vectors": vecs, "bytes_sha256": bytes_sha(vecs)}
        record[name] = {"kind": "split_null", "seed": s, "tensor_sha256": tsha, "bank_tensor_sha256":
                        bank["tensor_sha256"], "abs_cos_ego_range": core_span([abs(v["cos_target"]) for v in
                                                                               made["layers"].values()]),
                        "chance_p95_range": core_span([v["abs_cos_chance_p95"] for v in made["layers"].values()])}
    for name, r in record.items():
        if name == "ego":
            continue
        hits = []
        for d in BENCH_RUNS:
            for f in sorted(d.glob("*.json")) if d.is_dir() else []:
                try:
                    j = json.loads(f.read_text())
                except ValueError:
                    continue
                t = j.get("tensor_sha256") or j.get("trait_tensor_sha256")
                if t == r["tensor_sha256"]:
                    hits.append(f"{d.parent.name}/{f.name}")
        r["bench_copies"] = hits
        r["payload_bytes_sha256"] = payload[name]["bytes_sha256"]
    return payload, record


def core_span(vals) -> list:
    return [round(min(vals), 4), round(max(vals), 4)] if vals else None


def conditions_for(stage: str) -> list[str]:
    """smoke: four conditions on 64 pairs. full: the three reproduction checks, then the four text conditions under
    each control. fullall (only if the checks fail to match P1's raw.json): every none and ego condition as well."""
    if stage == "smoke":
        return ["plain|none", "fp1|ego", "plain|formality", "fp1|null_s0"]
    names = ["plain|none", "fp1|none", "fp1|ego"]
    if stage == "fullall":
        names = [core.condition(t, i) for i in ("none", "ego") for t in TEXTS]
    for c in (*TRAITS, *(f"null_s{s}" for s in NULL_SEEDS)):
        names += [core.condition(t, c) for t in TEXTS]
    return names


def run_local(stage: str) -> dict:
    import p1_spend as S
    from scout.schema import utc_now

    spec_bytes = core.SPEC_FILE.read_bytes()
    spec_sha = hashlib.sha256(spec_bytes).hexdigest()
    spec = core.load_spec()
    ms = spec["models"]["ministral"]
    if hashlib.sha256(EGO_FILE.read_bytes()).hexdigest() != ms["direction"]["sha256"] or \
            str(EGO_FILE) != ms["direction"]["file"]:
        raise SystemExit(f"{EGO_FILE}: not P1's direction file (path or sha256)")
    sdir = OUT / stage
    sdir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().replace("-", "").replace(":", "")
    run_id = f"p1b-{stage}-ministral-{stamp}"
    payload, record = make_controls(ms)
    conditions = conditions_for(stage)
    run = {"stage": stage, "run_id": run_id, "spec": {**spec, "models": {"ministral": ms}},
           "pairs": spec["smoke"]["pairs"] if stage == "smoke" else spec["data"]["rows"],
           "subsample": spec["smoke"]["subsample"] if stage == "smoke" else spec["measure"]["subsample"],
           "conditions": conditions, "controls": payload,
           "code_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in CODE.items()}}
    (sdir / "spec.yaml").write_bytes(spec_bytes)
    meter = S.P1Meter(spend_dir=EXPLORE_RESULTS / "spend", log=OUT / "spend.jsonl", label="p1b-ministral",
                      session=run_id, rate=S.p1_rate(spec["cost"]["rate"]), cap_usd=CAP_USD,
                      daily_stop_usd=float(spec["cost"]["daily_stop_usd"]))
    est = ESTIMATE_S["startup"] + ESTIMATE_S[stage]
    meter.reserve(est, stage)
    print(f"[p1b] {run_id}: {len(conditions)} conditions on {run['pairs']} pairs; reserved ${meter.usd(est):.2f}; "
          f"P1b so far ${meter.p1_spent():.2f} of ${CAP_USD:.2f}", flush=True)
    for n, r in record.items():
        if n != "ego":
            print(f"[p1b] control {n}: tensor {r['tensor_sha256'][:16]}, bench copies {len(r['bench_copies'])}",
                  flush=True)
    res, err = None, None
    try:
        res = run_b.with_options(timeout=TIMEOUT_S[stage]).remote(run)
    except Exception as e:  # noqa: BLE001 (a StageError names only the step and the type)
        err = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        usd = meter.settle(stage, model="ministral", run_id=run_id, ok=res is not None)
        print(f"[p1b] this call about ${usd:.3f}; P1b total ${meter.p1_spent():.3f}; today ${meter.today():.3f}",
              flush=True)
    if res is None:
        (sdir / "error.json").write_text(json.dumps({"run_id": run_id, "error": err}, indent=1) + "\n")
        raise SystemExit(f"[p1b] {stage} failed: {err}")
    res["laptop"] = {"spec_sha256": spec_sha, "call_usd": round(usd, 4), "finished": utc_now(), "controls": record}
    (sdir / "raw.json").write_text(json.dumps(res) + "\n")
    print(f"[p1b] saved {sdir / 'raw.json'}", flush=True)
    if stage == "smoke":
        print(json.dumps({"checks": res["checks"], "fixed_pair": res["fixed_pair"], "timing": res["timing"],
                          "conditions": {n: {"fixed": c["fixed"], "reach": c.get("cut_reach", {}).get("max_ratio"),
                                             "seconds": c["timing"]} for n, c in res["conditions"].items()}},
                         indent=1), flush=True)
    else:
        analyze_full(res, stage)
    return res


# ------------------------------------------------------------------------------------------------ the analysis


def analyze_full(res: dict, stage: str = "full") -> dict:
    """The reproduction checks against P1's raw.json, each cut's change in agreement on plain and first-person captions
    (fixed pair and own maximum), ego's margin over each control and over each control family, and what each direction
    could remove from the pooled caption features (its second moment there, against v_ego's)."""
    p1 = json.loads(P1_RAW.read_text())
    new, old = res["conditions"], p1["conditions"]
    n_rows, sub = res["pairs"], res["subsample"]
    size, z = int(sub["size"]), float(sub["z"])
    checks = {"fixed_pair_same": res["fixed_pair"] == p1["fixed_pair"], "conditions": {}}
    for n in ("plain|none", "fp1|none", "fp1|ego"):
        a, b = new[n], old[n]
        checks["conditions"][n] = {
            "fixed_same": a["fixed"]["score"] == b["fixed"]["score"],
            "fixed_subsets_same": a["fixed_subsets"] == b["fixed_subsets"],
            "own_same": a["own"] == b["own"],
            "max_abs_fixed_subset_diff": max(abs(x - y) for x, y in zip(a["fixed_subsets"], b["fixed_subsets"]))}
    reuse = checks["fixed_pair_same"] and all(c["fixed_same"] and c["fixed_subsets_same"]
                                              for c in checks["conditions"].values())
    checks["reuse_p1_conditions"] = reuse
    merged = {**{n: c for n, c in old.items() if core.split_condition(n)[1] in ("none", "ego")}, **new} if reuse else new
    need = [core.condition(t, i) for i in ("none", "ego") for t in TEXTS]
    if any(n not in merged for n in need):
        raise SystemExit(f"[p1b] the reproduction checks did not match P1's raw.json ({json.dumps(checks)}); run the "
                         f"fullall stage, which scores every none and ego condition itself")
    cuts = ["ego", *TRAITS, *(f"null_s{s}" for s in NULL_SEEDS)]
    fp = ["fp1", "fp2", "fp3"]
    out = {"p1b": "explore-track rerun of P1's (b) with difference-vector controls; unregistered; a sighting at most",
           "written_by": "Claude (Opus 5.5), from the container's numbers; captions never left Modal",
           "run_id": res["run_id"], "p1_run_id": p1["run_id"], "pairs": n_rows, "fixed_pair": res["fixed_pair"],
           "checks": {**checks, **res["checks"]}, "reads": {}}

    def group_multi(scores, subsets, texts, ivs):
        import torch

        parts = [core.group_score(scores, subsets, texts, iv) for iv in ivs]
        return sum(p[0] for p in parts) / len(parts), torch.stack([p[1] for p in parts]).mean(0)

    def dmulti(scores, subsets, x, y):
        fx, sx = group_multi(scores, subsets, *x)
        fy, sy = group_multi(scores, subsets, *y)
        return {"diff": fx - fy, "interval": core.interval(fx - fy, sx - sy, n_rows, size, z), "x": fx, "y": fy}

    families = {"other_trait": list(TRAITS), "split_null": [f"null_s{s}" for s in NULL_SEEDS]}
    for read, key, skey in (("fixed_pair", "fixed", "fixed_subsets"), ("own_max", "own", "own_subsets")):
        scores = {n: c[key]["score"] for n, c in merged.items()}
        subsets = {n: c[skey] for n, c in merged.items()}
        r = {"change": {}, "ego_margin": {}, "ego_margin_family": {}, "family_change": {}}
        for g, texts in (("plain", ["plain"]), ("fp", fp)):
            r["change"][g] = {x: core.diff(scores, subsets, (texts, x), (texts, "none"), n_rows, size, z) for x in cuts}
            r["ego_margin"][g] = {x: core.diff(scores, subsets, (texts, "ego"), (texts, x), n_rows, size, z)
                                  for x in cuts[1:]}
            r["ego_margin_family"][g] = {f: dmulti(scores, subsets, (texts, ["ego"]), (texts, m))
                                         for f, m in families.items()}
            r["family_change"][g] = {f: dmulti(scores, subsets, (texts, m), (texts, ["none"]))
                                     for f, m in families.items()}
        if key == "own":
            r["own_layers"] = {n: c["own"]["layers"] for n, c in merged.items()}
        out["reads"][read] = r
    # what each direction could remove from the pooled caption features: its second moment over captions, summed over
    # the cut's layers, against v_ego's (plain and fp1, uncut)
    feat = {}
    for t in ("plain", "fp1"):
        st = new[f"{t}|none"]["feature_stats"]
        lay = sorted(st["ego"], key=int)
        e = sum(st["ego"][x]["m2"] for x in lay)
        ev = sum(st["ego"][x]["var"] for x in lay)
        feat[t] = {d: {"m2_ratio_pooled": sum(st[d][x]["m2"] for x in lay) / e,
                       "var_ratio_pooled": sum(st[d][x]["var"] for x in lay) / ev,
                       "m2_ratio_range": core_span([st[d][x]["m2"] / st["ego"][x]["m2"] for x in lay])}
                   for d in st}
    out["feature_strength"] = feat
    out["cut_reach"] = {n: c["cut_reach"]["max_ratio"] for n, c in new.items() if "cut_reach" in c}
    out["controls"] = res["laptop"]["controls"]
    out["cos_ego"] = {n: core_span([abs(v) for v in c["cos_ego"].values()]) for n, c in res["controls"].items()}
    out["timing"] = res["timing"]
    out["spend"] = {"call_usd": res["laptop"]["call_usd"]}
    (OUT / stage / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print_summary(out)
    return out


def print_summary(out: dict) -> None:
    def f(d):
        return f"{d['diff']:+.4f} [{d['interval'][0]:+.4f}, {d['interval'][1]:+.4f}]"

    print(f"[p1b] checks: fixed pair same {out['checks']['fixed_pair_same']}; reuse P1's none/ego conditions "
          f"{out['checks']['reuse_p1_conditions']}")
    for read in ("fixed_pair", "own_max"):
        r = out["reads"][read]
        print(f"== {read}")
        for g in ("plain", "fp"):
            print(f"  {g}: change from uncut")
            for x, d in r["change"][g].items():
                print(f"    {x:14s} {f(d)}")
            for fam, d in r["family_change"][g].items():
                print(f"    {'family ' + fam:14s} {f(d)}")
            print(f"  {g}: ego minus control (negative: ego costs more)")
            for x, d in r["ego_margin"][g].items():
                print(f"    {x:14s} {f(d)}")
            for fam, d in r["ego_margin_family"][g].items():
                print(f"    {'family ' + fam:14s} {f(d)}")
    for t, v in out["feature_strength"].items():
        print(f"  pooled caption features ({t}): second moment over v_ego's, summed over 8-32: " +
              ", ".join(f"{d} {x['m2_ratio_pooled']:.3f}" for d, x in v.items()))
    print(f"  cut reach (max ratio): {json.dumps({k: round(v, 4) for k, v in out['cut_reach'].items()})}")


@app.local_entrypoint()
def smoke():
    run_local("smoke")


@app.local_entrypoint()
def full():
    run_local("full")


@app.local_entrypoint()
def fullall():
    run_local("fullall")


@app.local_entrypoint()
def analyze(stage: str = "full"):
    analyze_full(json.loads((OUT / stage / "raw.json").read_text()), stage)
