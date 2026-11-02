"""The hub cut: the "being spoken to" direction ablated, read on the audit dose-response. Private.

The bridge's root test (modal_bridge.py, bridge_core.hub) reads about 200 ordinary concepts in two neutral carriers,
in the self frame and the reader frame of what_it_means.py, and finds a shared component in the per-concept
self-minus-reader shift vectors: in the base run (results/bridge/base) the uncentered first principal component
carries 57% of the shifts' sum of squares at layer 20 (the self-stakes direction's best layer), lies along the mean
shift (cos 0.9997) and is nearly orthogonal to v_ego (cos -0.03), the self-stakes direction (0.03) and the refusal
direction (0.01). That is a direction the model carries whenever a message is addressed to it, whatever the content.

This app rebuilds that direction in a container the way modal_bridge does (the base model, no hooks, the same
concepts, carriers, frames and Reader, every layer 8-33) and saves it to results/hub_cut/hub.pt: a unit vector per
layer, the best layer (the self-stakes direction's, 20), each layer's PC1 share and its cosine to the best layer's
vector and to the layer before (how much the direction drifts with depth). Then it plays audit_dose.py's
dose-response (the audit probability stated in the prompt and used by the game: 0, 0.1, 0.3, 0.6, 1.0; 50 games
per cell on seeds 30-79; lockstep, resumable) with the hub ablated, h <- h - (h.d)d after each block, through
anatta.dial, as the ego cut is:
  hub_8_32     d = that layer's own hub PC1 at every layer 8-32 (the ego cut's form: cut_vs_push ablates ego.pt's
               per-layer vector at its layer, and the self-stakes cut does the same with self_stakes.pt);
  hub_17_26    the same at layers 17-26 (the self-stakes cut's band);
  hub20_8_32, hub20_17_26   the single layer-20 vector ablated at every layer of the band (Arditi-style, one
               direction everywhere), to check that the two forms agree; off by default.
The base curve is not replayed: it is read from results/audit_dose/dose1 (--base-from; base slope -37.4 points per
unit probability, 59.1% at 0 to 19.6% at 1; cut_self_stakes_17_26 there: -31.6, ratio 0.85, not flatter by half),
and each hub arm is compared with fishery_grade.dose_compare: its curve with CIs, its slope with a CI, the slope
ratio and difference against base with a CI, and prediction 86's "flatter by half" flag.

Writes results/hub_cut/{hub.pt, hub.json} and results/hub_cut/NAME/{LABEL.jsonl, summary.json, chunks/},
LABEL = ADAPTER.CUT.base.auditP (base.hub_8_32.base.audit0.3).
Usage (~/prj/matlab/.venv/bin/modal, from the lab root):
  modal run lab/experiments/h4lite/hub_cut.py::hub                               (build hub.pt only: about 2 minutes)
  modal run lab/experiments/h4lite/hub_cut.py::main --name hub1                  (hub.pt if missing, then the two arms x 5 levels)
  modal run lab/experiments/h4lite/hub_cut.py::main --name hub1 --arms hub20_8_32,hub20_17_26
  modal run lab/experiments/h4lite/hub_cut.py::table --name hub1                 (recompute from disk, no GPU)
Checks: PYTHONPATH=<anatta-dev>/src ~/prj/matlab/.venv/bin/python test_hub_cut.py
"""

import io
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

H4 = str(Path(__file__).resolve().parent)  # lab/experiments/h4lite on the laptop; a container gets h4_modal from PYTHONPATH
sys.path.insert(0, H4)
from h4_modal import ALIGN, CACHE, DDIR, DIALS, MODEL, RESULTS, RUNS, SELF_STAKES, adapter_dir, cost, image, read_jsonl, volumes, write_jsonl  # noqa: E402
from fishery_grade import adapter_tag, dose_compare, dose_curve  # noqa: E402
import bridge_core as bc  # noqa: E402

OUT = RESULTS / "hub_cut"
HUB_PT, HUB_JSON = OUT / "hub.pt", OUT / "hub.json"
DOSE = RESULTS / "audit_dose"
LEVELS = (0.0, 0.1, 0.3, 0.6, 1.0)
BANDS = ((8, 32), (17, 26))
DEFAULT_ARMS = "hub_8_32,hub_17_26"
DEFAULT_BATCH = 12  # games per generation call (fishery_play.DEFAULT_BATCH)
BASE_SLOPE_DOSE1 = -37.43  # results/audit_dose/dose1, base, seeds 30-79 (for the record; recomputed from the files)
app = modal.App("anatta-h4-hub-cut", image=image())
cache, runs = volumes()


# ------------------------------------------------------------------------------------------------- the direction


def pc1(shifts) -> tuple[np.ndarray, float]:
    """The uncentered first principal direction of the rows (unit, sign along the mean row) and its share of the
    rows' sum of squares: bridge_core.hub's pc1 and pc1_share, with the vector returned."""
    X = np.asarray(shifts, dtype=np.float64)
    mean = X.mean(0)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    if float(v @ mean) < 0:
        v = -v
    return v / max(np.linalg.norm(v), 1e-12), float(s[0] ** 2 / max((s ** 2).sum(), 1e-12))


def hub_directions(shifts_by_layer: dict, best: int, refs_by_layer: dict | None = None) -> tuple[dict, dict]:
    """Per layer, the hub PC1 of that layer's per-concept shift vectors (concepts x d), with bridge_core.hub's
    statistics, the cosine to the best layer's PC1 and to the layer before. Returns (vectors, summary)."""
    layers = sorted(shifts_by_layer)
    if best not in shifts_by_layer:
        raise ValueError(f"best layer {best} not among the read layers {layers}")
    vecs, per_layer = {}, {}
    for l in layers:
        v, share = pc1(shifts_by_layer[l])
        vecs[l] = v
        per_layer[l] = bc.hub(shifts_by_layer[l], (refs_by_layer or {}).get(l, {}))
        per_layer[l]["pc1_share"] = round(share, 4)
    for i, l in enumerate(layers):
        per_layer[l]["cos_to_best"] = round(float(vecs[l] @ vecs[best]), 4)
        per_layer[l]["cos_to_prev"] = round(float(vecs[l] @ vecs[layers[i - 1]]), 4) if i else None
    summary = {"best_layer": best, "layers": layers, "pc1_share": per_layer[best]["pc1_share"],
               "concepts": per_layer[best].get("concepts"),
               "per_layer": {str(l): per_layer[l] for l in layers},
               "cos_to_best": {str(l): per_layer[l]["cos_to_best"] for l in layers},
               "min_cos_to_best_8_32": round(min(per_layer[l]["cos_to_best"] for l in layers if 8 <= l <= 32), 4)}
    return vecs, summary


def hub_arm_specs(layers: list[int], best: int, bands=BANDS) -> dict[str, list[tuple[int, int]]]:
    """The hub arms as (layer hooked, layer whose vector is ablated there): hub_LO_HI uses each layer's own PC1,
    hub{best}_LO_HI the best layer's vector at every layer of the band. Layers not read are left out."""
    out = {}
    for lo, hi in bands:
        band = [l for l in layers if lo <= l <= hi]
        out[f"hub_{lo}_{hi}"] = [(l, l) for l in band]
        out[f"hub{best}_{lo}_{hi}"] = [(l, best) for l in band]
    return out


def interventions_from(vectors: dict, specs: dict) -> dict:
    """anatta.dial ablations for every hub arm (the ego cut's form: mode "ablate", one hook per layer)."""
    from anatta.dial import Intervention

    return {name: [Intervention(layer=l, direction=vectors[src], mode="ablate") for l, src in spec] for name, spec in specs.items()}


def save_hub(path, vectors: dict, summary: dict, extra: dict | None = None) -> None:
    """hub.pt: a float32 unit vector per layer, the best layer, the per-layer statistics. `path` may be a buffer."""
    import torch

    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": {int(l): torch.as_tensor(np.asarray(v), dtype=torch.float32) for l, v in vectors.items()},
                "best_layer": summary["best_layer"], "layers": summary["layers"], "pc1_share": summary["pc1_share"],
                "cos_to_best": summary["cos_to_best"], "per_layer": summary["per_layer"], **(extra or {})}, path)


def load_hub(path: Path) -> dict:
    import torch

    return torch.load(path)


# ------------------------------------------------------------------------------------------------- the games


def label_of(adapter: str, cut: str, p: float) -> str:
    return f"{adapter_tag(adapter)}.{cut or 'off'}.base.audit{p:g}"


def chunk_path(out: Path, label: str, seed_start: int) -> Path:
    return out / "chunks" / f"{label}__{seed_start:04d}.jsonl"


def base_cells_from(base_dir: Path, levels: list[float]) -> dict[float, list[dict]]:
    """The untrained, uncut model's dose cells in an audit_dose results folder (base.base.base.auditP.jsonl, the
    `base` arm's label there; base.off.base.auditP.jsonl is accepted too)."""
    cells = {}
    for p in levels:
        for cut in ("base", "off"):
            f = base_dir / f"{label_of('', cut, p)}.jsonl"
            if f.exists():
                cells[p] = read_jsonl(f)
                break
    return cells


def report_from(out: Path, arms: list[str], base_dir: Path, levels: list[float], hub_summary: dict | None = None,
                adapter: str = "") -> dict:
    """The hub arms' curves from disk, each against the base curve in base_dir (fishery_grade.dose_compare)."""
    base = base_cells_from(base_dir, levels)
    cells = {arm: {p: read_jsonl(out / f"{label_of(adapter, arm, p)}.jsonl")
                   for p in levels if (out / f"{label_of(adapter, arm, p)}.jsonl").exists()} for arm in arms}
    curves = {"base": dose_curve(base)} if base else {}
    curves.update({arm: dose_curve(cells[arm]) for arm in arms if cells[arm]})
    comparisons = {arm: dose_compare(base, cells[arm]) for arm in arms if base and cells[arm]}
    summary = {"adapter": adapter, "arms": arms, "levels": [str(p) for p in levels], "base_from": str(base_dir),
               "base_games": sum(len(v) for v in base.values()), "base_slope_dose1": BASE_SLOPE_DOSE1,
               "hub": hub_summary, "curves": curves, "comparisons_vs_base": comparisons}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if hub_summary:
        print(f"hub: layer {hub_summary['best_layer']}, PC1 share {hub_summary['pc1_share']}, {hub_summary.get('concepts')} concepts, "
              f"min cos to the layer-{hub_summary['best_layer']} vector over 8-32: {hub_summary.get('min_cos_to_best_8_32')}", flush=True)
    if not base:
        print(f"no base cells under {base_dir}: curves only, no slope ratio", flush=True)
    head = "  ".join(f"{p:>6g}" for p in levels)
    print(f"{'arm':14} {head}   slope [95% CI]      monotone", flush=True)
    for arm, c in curves.items():
        rates = "  ".join(f"{(c['per_level'].get(str(p)) or {}).get('lie_rate', float('nan')):6.1f}" for p in levels)
        ci = c["slope_ci"] or [float("nan")] * 2
        print(f"{arm:14} {rates}   {c['slope']:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]   {c['monotone_within_tolerance']}", flush=True)
    for arm, d in comparisons.items():
        print(f"{arm} vs base: slope {d.get('slope_arm')} vs {d.get('slope_base')}, ratio {d.get('slope_ratio')}, "
              f"diff {d.get('slope_diff')} {d.get('slope_diff_ci')}, flatter by half: {d.get('flatter_by_half')}; "
              f"per level {d.get('per_level_diff')}", flush=True)
    return summary


# ------------------------------------------------------------------------------------------------- modal


@app.function(volumes={CACHE: cache}, timeout=3600)
def prefetch() -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, ignore_patterns=["consolidated.safetensors"])
    cache.commit()
    return path


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache}, timeout=3600)
def build_hub(layer: int, seed: int) -> dict:
    """The root reads of modal_bridge.py on the base model (no adapter, no hooks), every layer 8-33, and the hub
    PC1 per layer. Returns the summary and hub.pt's bytes."""
    import os

    import torch
    from what_it_means import FRAMES, Reader

    from anatta.models import get_layers, load
    from anatta.probe import Direction

    os.chdir(f"{ALIGN}/anatta")
    torch.manual_seed(seed)
    t0 = time.time()
    ddir = Path(f"{ALIGN}/anatta/{DDIR}")
    ss = torch.load(f"{ALIGN}/{SELF_STAKES}")
    best = layer or int(ss["best_layer"])
    model, tok = load(MODEL)
    layers = list(range(8, len(get_layers(model))))
    reader = Reader(model, tok, layers)
    t_load = time.time() - t0
    roots, skipped = {}, []
    for f in ("self", "reader"):
        system, user = FRAMES[f]
        for c in bc.CONCEPTS:
            vecs = []
            for carrier in bc.CARRIERS:
                acts = reader.span([{"role": "system", "content": system},
                                    {"role": "user", "content": user.format(c=carrier.format(c=c))}], bc.quoted(c), [])
                if acts[layers[0]].shape[0]:
                    vecs.append({l: acts[l].mean(0) for l in layers})
            if vecs:
                roots[(f, c)] = {l: torch.stack([v[l] for v in vecs]).mean(0) for l in layers}
            elif c not in skipped:
                skipped.append(c)
    concepts = [c for c in bc.CONCEPTS if ("self", c) in roots and ("reader", c) in roots]
    shifts = {l: (torch.stack([roots[("self", c)][l] for c in concepts])
                  - torch.stack([roots[("reader", c)][l] for c in concepts])).double().numpy() for l in layers}
    f32 = lambda v: v.detach().float().flatten().cpu().numpy()  # noqa: E731  (modal_bridge's f32, to numpy)
    ego = Direction.load(ddir / "ego.pt").vectors
    refs = {l: {**({"ego_fixed": f32(ego[l])} if l in ego else {}),
                **({"self_stakes_fixed": f32(ss["vectors"][l])} if l in ss["vectors"] else {})} for l in layers}
    vectors, summary = hub_directions(shifts, best, refs)
    summary.update({"concepts": len(concepts), "skipped": skipped, "model": MODEL, "seed": seed,
                    "self_stakes_best_layer": int(ss["best_layer"]), "frames": ["self", "reader"],
                    "carriers": bc.CARRIERS, "timing": {"load": round(t_load), "total": round(time.time() - t0)}})
    buf = io.BytesIO()
    save_hub(buf, vectors, summary, {"model": MODEL, "concepts": len(concepts), "skipped": skipped,
                                     "source": "uncentered PC1 of the per-concept self-minus-reader shifts (bridge_core.hub) "
                                               "on the base model, sign along the mean shift"})
    return {"summary": summary, "hub": buf.getvalue()}


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=4 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0))
def play_chunk(adapter: str, cut: str, audit_p: float, seed_start: int, seeds: int, batch: int, tokens: int,
               temperature: float, hub_bytes: bytes) -> dict:
    import torch

    import fishery_play

    hub = torch.load(io.BytesIO(hub_bytes))
    arms = interventions_from(hub["vectors"], hub_arm_specs(sorted(hub["vectors"]), hub["best_layer"]))
    if cut not in arms:
        raise ValueError(f"unknown hub arm {cut!r}; the arms are {', '.join(arms)}")
    adir = adapter_dir(adapter)
    if adir and not Path(adir).exists():
        raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
    return fishery_play.run_chunk(MODEL, f"{ALIGN}/anatta/{DDIR}", f"{ALIGN}/anatta/{DIALS}", adir, adapter, cut, "",
                                  audit_p, seed_start, seeds, batch, tokens, temperature,
                                  progress=lambda s: print(s, flush=True), extra_arms=arms)


def ensure_hub(layer: int, seed: int, gpu: str, rebuild: bool) -> dict:
    """hub.pt and hub.json on disk (built in a container when missing or --rebuild); returns hub.json's summary."""
    if HUB_PT.exists() and HUB_JSON.exists() and not rebuild:
        return json.loads(HUB_JSON.read_text())
    r = build_hub.with_options(gpu=gpu).remote(layer, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    HUB_PT.write_bytes(r["hub"])
    HUB_JSON.write_text(json.dumps(r["summary"], indent=1))
    s = r["summary"]
    print(f"hub built: layer {s['best_layer']}, PC1 share {s['pc1_share']}, {s['concepts']} concepts "
          f"(skipped {s['skipped']}), {s['timing']['total']}s; wrote {HUB_PT}", flush=True)
    for l in s["layers"]:
        pl = s["per_layer"][str(l)]
        print(f"  L{l:2d} share {pl['pc1_share']:.3f} cos-to-L{s['best_layer']} {pl['cos_to_best']:+.3f} "
              f"cos-to-prev {pl['cos_to_prev'] if pl['cos_to_prev'] is not None else '':>6} "
              f"ego {pl['pc1_cos'].get('ego_fixed', '')} self-stakes {pl['pc1_cos'].get('self_stakes_fixed', '')}", flush=True)
    return s


@app.local_entrypoint()
def hub(layer: int = 0, seed: int = 0, gpu: str = "L40S", rebuild: bool = False):
    """Build (or show) the hub direction only. --layer 0 takes the self-stakes direction's best layer (20)."""
    s = ensure_hub(layer, seed, gpu, rebuild)
    print(f"hub.pt: layer {s['best_layer']}, PC1 share {s['pc1_share']}, min cos to that layer's vector over 8-32 "
          f"{s.get('min_cos_to_best_8_32')}", flush=True)


@app.local_entrypoint()
def main(name: str = "hub_cut", arms: str = DEFAULT_ARMS, adapter: str = "", base_from: str = "dose1",
         levels: str = ",".join(str(p) for p in LEVELS), seed_start: int = 30, seeds: int = 50, chunk: int = 25,
         batch: int = DEFAULT_BATCH, tokens: int = 400, temperature: float = 0.7, gpu: str = "L40S", layer: int = 0,
         seed: int = 0, rebuild: bool = False):
    out = OUT / name
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    level_list = [float(x) for x in levels.split(",")]
    hub_summary = ensure_hub(layer, seed, gpu, rebuild)
    known = hub_arm_specs(hub_summary["layers"], hub_summary["best_layer"])
    for arm in arm_list:
        if arm not in known:
            raise SystemExit(f"unknown hub arm {arm!r}; the arms are {', '.join(known)}")
    end = seed_start + seeds
    tasks = [(arm, p, s, min(chunk, end - s)) for arm in arm_list for p in level_list for s in range(seed_start, end, chunk)]
    todo = [t for t in tasks if not chunk_path(out, label_of(adapter, t[0], t[1]), t[2]).exists()]
    print(f"{len(tasks) - len(todo)}/{len(tasks)} chunks on disk; {len(todo)} to run on {gpu}", flush=True)
    gpu_s, t0 = 0.0, time.time()
    if todo:
        print("weights:", prefetch.remote(), flush=True)
        hub_bytes = HUB_PT.read_bytes()
        calls = [(adapter, arm, p, s, n, batch or DEFAULT_BATCH, tokens, temperature, hub_bytes) for arm, p, s, n in todo]
        for r in play_chunk.with_options(gpu=gpu).starmap(calls, order_outputs=False, return_exceptions=True):
            if isinstance(r, Exception):
                print("chunk failed:", repr(r)[:500], flush=True)
                continue
            p = chunk_path(out, r["label"], r["seed_start"])
            write_jsonl(p, r["rows"])
            p.with_suffix(".meta.json").write_text(json.dumps({k: v for k, v in r.items() if k != "rows"} | {"modal": gpu}))
            gpu_s += r["seconds"]
            print(f"{time.time() - t0:5.0f}s {r['label']} seeds {r['seed_start']}+{r['seeds']}: {r['seconds']}s in container, "
                  f"batch {r.get('batch')}, peak memory {r.get('peak_gb')} GiB", flush=True)
    missing = []
    for arm in arm_list:
        for p in level_list:
            lab = label_of(adapter, arm, p)
            parts = [chunk_path(out, lab, s) for s in range(seed_start, end, chunk)]
            if all(x.exists() for x in parts):
                write_jsonl(out / f"{lab}.jsonl", [row for x in parts for row in read_jsonl(x)])
            else:
                missing.append(lab)
    report_from(out, arm_list, DOSE / base_from, level_list, hub_summary, adapter)
    print(f"done; {cost(gpu_s)} in containers this run; "
          f"{len(missing)} cells incomplete{': ' + ', '.join(missing) if missing else ''}", flush=True)


@app.local_entrypoint()
def table(name: str = "hub_cut", arms: str = DEFAULT_ARMS, adapter: str = "", base_from: str = "dose1",
          levels: str = ",".join(str(p) for p in LEVELS)):
    """Recompute the curves and comparisons from results/hub_cut/NAME against results/audit_dose/BASE_FROM (no GPU)."""
    hub_summary = json.loads(HUB_JSON.read_text()) if HUB_JSON.exists() else None
    report_from(OUT / name, [a.strip() for a in arms.split(",") if a.strip()], DOSE / base_from,
                [float(x) for x in levels.split(",")], hub_summary, adapter)
