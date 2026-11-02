"""The mechanistic bridge on one adapter, in minutes (plan v2.2 held-out test 6, plus the v2.3 additions and the root
test). Private.

On a fixed prompt set (what_it_means.py's 12 contents x 3 frames, read over the content tokens), for the adapter and,
in the same container with the adapter switched off, the base model:
  projections   of v_ego and the self-stakes direction, per layer, in units of the base model's SD across the prompts
                (the plan's prediction 72 reads koan vs noexit from these);
  loadings      the six ego traits' cosines with the self-stakes direction (now -0.19 to +0.01);
  tint          what_it_means.py's frames test (self / observer / reader) on stakes and neutral content, with a 95%
                bootstrap interval over contents and the noise floor from 50 random directions (the 95th percentile
                of |mean| and the 2.5-97.5% interval), and whether the adapter's tint falls outside the base's interval;
  cosines       of v_ego and self-stakes with the scene and character directions on file and with a refusal direction
                extracted Arditi-style here (the mean difference between 32 harmful and 32 harmless instructions at
                the last prompt token, per layer, bridge_core.py; split-half d reported), with the max |cos| over
                layers 8-32 (predictions 82, 83);
  root test     about 200 ordinary concepts (bridge_core.CONCEPTS), each in two neutral carriers, read in the self
                frame and the reader frame: the concept token's projection on v_ego and self-stakes, the mean over
                concepts, the share of concepts whose self-minus-reader shift exceeds their random-direction 95% band,
                and the first principal component of the per-concept shift vectors with its cosine to v_ego and
                self-stakes (a shared "hub");
  layer 33      with --cut, the v_ego projection after the last cut layer (on the fixed layer-32 vector, a proxy, and
                on the re-extracted layer-33 vector), adapter vs base under the same cut: routing around the cut.
Everything is measured twice: with the base model's directions held fixed (ego.pt, the six traits, self_stakes.pt;
primary) and with directions re-extracted from the adapter (the six traits through anatta.probe.extract and
factor.composite, exactly the repo's recipe; the self-stakes direction through what_it_means.build_direction), with
each re-extracted direction's cosine to the base one.

Writes lab/experiments/h4lite/results/bridge/NAME/{summary.json, directions.pt} (NAME defaults to the adapter as
fishery_grade.adapter_tag shortens it, h4lite/sft/koan_s0/adapter -> sft__koan_s0, plus .CUT when the cut is on).
Adapters are named as the training half saves them (--adapter NAME-with-slash is /runs/NAME); --cut combines with
--adapter for the under-the-cut arm's layer-33 check.
Usage (~/prj/matlab/.venv/bin/modal):
  modal run lab/experiments/h4lite/modal_bridge.py::main --adapter h4lite/sft/koan_s0/adapter
  modal run lab/experiments/h4lite/modal_bridge.py::main --adapter h4lite/sft/koan_cut_s0/adapter --cut cut
  modal run lab/experiments/h4lite/modal_bridge.py::main                       (the base model: fixed vs its own re-extraction)
  modal run lab/experiments/h4lite/modal_bridge.py::main --adapter h4lite/sft/koan_s0/adapter --no-traits --no-root   (quick subset)
  modal run lab/experiments/h4lite/modal_bridge.py::main --system "You are Atlas. ..." --name atlas   (a system prompt folded
                                                                into every chat; positive_control.py runs its three conditions so)
  modal run lab/experiments/h4lite/modal_bridge.py::table                      (every summary on disk, one line each)
The summary also keeps the per-item numbers a paired comparison between two runs needs (tint_rows per condition, the
root test's items and root_concept_list) and directions.pt the per-concept shift matrices at the fixed directions'
best layers (hub_shifts); positive_control.py reads them.
"""

import json
import sys
import time
from pathlib import Path

import modal

H4 = str(Path(__file__).resolve().parent)  # lab/experiments/h4lite on the laptop; a container gets h4_modal from PYTHONPATH
sys.path.insert(0, H4)
from h4_modal import ALIGN, CACHE, DDIR, DIALS, EXTRA, MODEL, RESULTS, RUNS, SELF_STAKES, adapter_dir, cost, image, volumes  # noqa: E402
from fishery_grade import adapter_tag  # noqa: E402

OUT = RESULTS / "bridge"
app = modal.App("anatta-h4-bridge", image=image())
cache, runs = volumes()


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=2 * 3600)
def bridge(adapter: str, cut: str, traits: bool, root: bool, n_rand: int, seed: int, system: str = "", frames: str = "v1") -> dict:
    """`system` (positive_control.py) is a scaffold folded into every chat the bridge renders, from the frames and
    the root reads to the trait and refusal extractions (bridge_core.Scaffolded wraps the tokenizer); "" is the
    bridge as it ran. `frames` names the frame set (what_it_means.FRAME_SETS: "v1" is the bridge as it ran, "v2"
    round five's frames with no system message of their own, so the model's default system prompt stands). The
    summary then also carries the per-item numbers a paired comparison between runs needs: the tint rows per
    condition, the root test's per-concept items and the concept list, and (in directions.pt) the per-concept
    shift matrices at the fixed directions' best layers for the hub."""
    import contextlib
    import io
    import os
    import statistics

    import torch

    import bridge_core as bc
    from cut_vs_push import build_arms
    from what_it_means import CONTENT, FRAME_SETS, Reader, build_direction, cohen_d, frame_chat, unit

    FRAMES = FRAME_SETS[frames]
    from what_it_means import CONTENT_SETS  # round seven (prediction 115): v3's tint reads 20 neutral contents; older sets read CONTENT

    CONTENT = CONTENT_SETS.get(frames, CONTENT)

    from anatta.factor import composite
    from anatta.models import chat_ids, get_layers, load
    from anatta.probe import Direction, Trait, capture, extract

    os.chdir(f"{ALIGN}/anatta")  # the cap specs name their direction relative to the repo
    t0, timing = time.time(), {}
    ddir = Path(f"{ALIGN}/anatta/{DDIR}")
    ivs = build_arms(ddir, Path(f"{ALIGN}/anatta/{DIALS}"))[cut] if cut not in ("", "off", "base") else []
    adir = adapter_dir(adapter)
    if adir and not Path(adir).exists():
        raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
    model, tok = load(MODEL)
    if system:
        tok = bc.Scaffolded(tok, system)  # every chat rendered from here on carries the scaffold
    peft = None
    if adir:
        from peft import PeftModel

        peft = PeftModel.from_pretrained(model, adir)  # unmerged: the LoRA layers sit inside the HF model
        hf = peft.get_base_model()  # the HF model with those layers, so hooks and reads see the adapter's residual
    else:
        hf = model
    device = next(hf.parameters()).device
    n_layers = len(get_layers(hf))
    layers = list(range(8, n_layers))  # the directions' layers 8-32 plus 33, the one after the cut
    reader = Reader(hf, tok, layers)
    timing["load"] = round(time.time() - t0)

    def ctx(is_base: bool):
        return peft.disable_adapter() if (is_base and peft) else contextlib.nullcontext()

    f32 = lambda v: v.detach().float().flatten().cpu()  # noqa: E731

    # the base model's directions on file
    ego = Direction.load(ddir / "ego.pt")
    ss = torch.load(f"{ALIGN}/{SELF_STAKES}")
    on_file = {p.stem: {l: f32(v) for l, v in Direction.load(p).vectors.items()} for p in Path(f"{ALIGN}/{EXTRA}").glob("*.pt")}
    fixed = {"ego": {l: f32(v) for l, v in ego.vectors.items()}, "ego_best": ego.best_layer,
             "self_stakes": {l: f32(v) for l, v in ss["vectors"].items()}, "ss_best": ss["best_layer"],
             "sep": float(ss["report"][ss["best_layer"]]["separation"]) or 1.0,
             "six": {n: {l: f32(v) for l, v in Direction.load(ddir / f"{n}.pt").vectors.items()} for n in bc.SIX}}
    hidden = next(iter(fixed["ego"].values())).numel()

    conditions = {"model": (False, [])}
    if peft:
        conditions["base"] = (True, [])
    if ivs:
        conditions["model+cut"] = (False, ivs)
        if peft:
            conditions["base+cut"] = (True, ivs)
    base_cond = "base" if peft else "model"

    # 1. the fixed prompt set: mean activation over the content tokens, per layer, per condition
    keys = [(kind, c, f) for kind, items in CONTENT.items() for c in items for f in FRAMES]
    reads = {}
    for cond, (is_base, hooks) in conditions.items():
        reads[cond] = {}
        with ctx(is_base):
            for kind, c, f in keys:
                acts = reader.span(frame_chat(FRAMES, f, c), c, hooks)
                reads[cond][(kind, c, f)] = {l: acts[l].mean(0) for l in layers}
    timing["frames"] = round(time.time() - t0 - sum(timing.values()))

    # 2. root test reads: concept x {self, reader}, mean over the quoted word's tokens and the two carriers
    roots, skipped = {}, []
    if root:
        for cond, (is_base, hooks) in conditions.items():
            roots[cond] = {}
            with ctx(is_base):
                for f in ("self", "reader"):
                    for c in bc.CONCEPTS:
                        vecs = []
                        for carrier in bc.CARRIERS:
                            acts = reader.span(frame_chat(FRAMES, f, carrier.format(c=c)), bc.quoted(c), hooks)
                            if acts[layers[0]].shape[0]:
                                vecs.append({l: acts[l].mean(0) for l in layers})
                        if vecs:
                            roots[cond][(f, c)] = {l: torch.stack([v[l] for v in vecs]).mean(0) for l in layers}
                        elif c not in skipped:
                            skipped.append(c)
        timing["root_reads"] = round(time.time() - t0 - sum(timing.values()))
    concepts = [c for c in bc.CONCEPTS if all((f, c) in roots[cond] for cond in roots for f in ("self", "reader"))] if root else []

    # 3. re-extracted directions on the model under test (adapter on, hooks off)
    re_vecs, re_report = build_direction(reader, layers)
    re_best = max(layers, key=lambda l: re_report[l]["heldout_d_stakeholder"] - abs(re_report[l]["heldout_d_self_neutral"]))
    re_sep = float(re_report[re_best]["separation"])
    if re_sep <= 1e-6:
        print(f"warning: re-extracted held-out separation {re_sep:.4f} at layer {re_best}; tints are raw", flush=True)
        re_sep = 1.0
    timing["self_stakes"] = round(time.time() - t0 - sum(timing.values()))
    re_six, re_ego, trait_d = {}, None, {}
    if traits:
        dirs = {n: extract(hf, tok, Trait.load(n), layers=layers, seed=seed) for n in bc.SIX}
        re_ego = composite(dirs, bc.SIX, layers)
        re_six = {n: {l: f32(v) for l, v in d.vectors.items()} for n, d in dirs.items()}
        trait_d = {n: {l: round(d.scores[l], 2) for l in layers} for n, d in dirs.items()}
        timing["traits"] = round(time.time() - t0 - sum(timing.values()))
    reex = {"ego": {l: f32(v) for l, v in re_ego.vectors.items()} if re_ego else None,
            "ego_best": re_ego.best_layer if re_ego else None,
            "self_stakes": {l: f32(v) for l, v in re_vecs.items()}, "ss_best": re_best, "sep": re_sep, "six": re_six}

    # 4. the refusal direction, Arditi-style, on the model and (adapter off) on the base
    def refusal_direction(is_base: bool):
        acts = {"harmful": [], "harmless": []}
        with ctx(is_base), torch.no_grad():
            for kind, instrs in (("harmful", bc.HARMFUL), ("harmless", bc.HARMLESS)):
                for s in instrs:
                    ids = chat_ids(tok, [{"role": "user", "content": s}]).to(device)
                    with capture(hf, layers) as store:
                        hf(ids)
                    acts[kind].append({l: store[l][0, -1].float().cpu() for l in layers})
        vecs, dd = {}, {}
        for l in layers:
            H = torch.stack([a[l] for a in acts["harmful"]])
            S = torch.stack([a[l] for a in acts["harmless"]])
            vecs[l] = unit(H.mean(0) - S.mean(0))
            d = 0.0
            for b, t in ((slice(0, None, 2), slice(1, None, 2)), (slice(1, None, 2), slice(0, None, 2))):
                u = unit(H[b].mean(0) - S[b].mean(0))
                d += cohen_d((H[t] @ u).tolist(), (S[t] @ u).tolist()) / 2
            dd[l] = round(d, 2)
        return vecs, dd

    refusal = {"model": refusal_direction(False)}
    if peft:
        refusal["base"] = refusal_direction(True)
    timing["refusal"] = round(time.time() - t0 - sum(timing.values()))

    # 5. the numbers
    rand = [unit(torch.randn(hidden, generator=torch.Generator().manual_seed(s))) for s in range(n_rand)]
    R = torch.stack(rand)  # [n_rand, d]
    cos = lambda a, b: round(float(unit(a) @ unit(b)), 3)  # noqa: E731
    summary = {"adapter": adapter, "cut": cut or "off", "model": MODEL, "layers": layers, "conditions": list(conditions),
               "n_rand": n_rand, "prompts": len(keys), "root_concepts": len(concepts), "root_skipped": skipped,
               "root_concept_list": concepts, "system": system, "frames": frames}
    summary["tint_contents"] = {k: len(v) for k, v in CONTENT.items()}
    r6 = lambda x: round(x, 6)  # noqa: E731  (the per-item numbers kept for paired comparisons)
    others = dict(on_file)
    others["refusal_model"] = refusal["model"][0]
    if peft:
        others["refusal_base"] = refusal["base"][0]
    for sname, S in (("fixed", fixed), ("reextracted", reex)):
        res = {}
        for dname in ("ego", "self_stakes"):
            V = S[dname]
            if V is None:
                continue
            best = S["ego_best"] if dname == "ego" else S["ss_best"]
            sep = S["sep"] if dname == "self_stakes" else 1.0
            entry = {"best_layer": best, "sep": round(sep, 4)}
            per_layer = {}
            for l in layers:
                if l not in V:
                    continue
                vals = {cond: [float(reads[cond][k][l] @ V[l]) for k in keys] for cond in reads}
                per_layer[l] = bc.projection_stats(vals["model"], vals[base_cond])
                if ivs:
                    per_layer[l]["under_cut"] = bc.projection_stats(vals["model+cut"], vals["base+cut" if peft else "model+cut"])
            entry["projection_by_layer"] = per_layer
            entry["projection"] = per_layer.get(best)
            entry["frame_means"] = {cond: {f: round(statistics.mean(float(reads[cond][k][best] @ V[best]) for k in keys if k[2] == f), 4)
                                           for f in FRAMES} for cond in reads}
            entry["tint"], entry["tint_rows"] = {}, {}
            for cond in reads:
                rows = []
                for kind, items in CONTENT.items():
                    for c in items:
                        row = {"kind": kind, "content": c}
                        for f in FRAMES:
                            vec = reads[cond][(kind, c, f)][best]
                            row[f] = float(vec @ V[best])
                            row[f + "_rand"] = (R @ vec).tolist()
                        rows.append(row)
                entry["tint"][cond] = bc.tint_summary(rows, sep)
                entry["tint_rows"][cond] = [{k: (r6(v) if isinstance(v, float) else [r6(x) for x in v] if isinstance(v, list) else v)
                                             for k, v in row.items()} for row in rows]  # raw projections, unscaled
            if peft:
                entry["tint_vs_base"] = {}
                for k, v in entry["tint"]["model"].items():
                    if "ci" in v:
                        lo, hi = entry["tint"]["base"][k]["ci"]
                        entry["tint_vs_base"][k] = {"model": v["mean"], "base_ci": [lo, hi], "outside_base_ci": not (lo <= v["mean"] <= hi),
                                                    "noise_p95": v.get("random_dir_abs_p95")}
            entry["cos"] = {}
            for oname, ovecs in list(others.items()) + [(f"{'self_stakes' if dname == 'ego' else 'ego'}_fixed",
                                                        fixed["self_stakes" if dname == "ego" else "ego"])]:
                entry["cos"][oname] = {l: cos(V[l], ovecs[l]) for l in layers if l in V and l in ovecs}
                entry["cos"][oname + "_max_abs_8_32"] = bc.max_abs(entry["cos"][oname])
            if sname == "reextracted":
                entry["cos_with_fixed"] = {l: cos(V[l], fixed[dname][l]) for l in layers if l in V and l in fixed[dname]}
                entry["cos_with_fixed_at_best"] = entry["cos_with_fixed"].get(best)
            res[dname] = entry
        if S["six"]:
            ssv, best = S["self_stakes"], S["ss_best"]
            res["six_on_self_stakes"] = {n: cos(S["six"][n][best], ssv[best]) for n in bc.SIX}
            res["six_on_self_stakes_mean_abs"] = round(statistics.mean(abs(x) for x in res["six_on_self_stakes"].values()), 3)
            res["six_on_self_stakes_by_layer_mean_abs"] = {
                l: round(statistics.mean(abs(cos(S["six"][n][l], ssv[l])) for n in bc.SIX), 3)
                for l in layers if l in ssv and all(l in S["six"][n] for n in bc.SIX)}
            if sname == "reextracted":
                fb = fixed["ss_best"]
                res["six_on_fixed_self_stakes"] = {n: cos(S["six"][n][fb], fixed["self_stakes"][fb]) for n in bc.SIX}
                res["six_cos_with_fixed_at_best"] = {n: cos(S["six"][n][fb], fixed["six"][n][fb]) for n in bc.SIX}
        if root and concepts:
            res["root"] = {}
            for dname in ("ego", "self_stakes"):
                V = S[dname]
                if V is None:
                    continue
                best = S["ego_best"] if dname == "ego" else S["ss_best"]
                sep = S["sep"] if dname == "self_stakes" else 1.0
                entry = {"best_layer": best}
                for cond in roots:
                    Sm = torch.stack([roots[cond][("self", c)][best] for c in concepts])
                    Rm = torch.stack([roots[cond][("reader", c)][best] for c in concepts])
                    entry[cond] = bc.root_summary((Sm @ V[best]).tolist(), (Rm @ V[best]).tolist(),
                                                  (Sm @ R.T).tolist(), (Rm @ R.T).tolist(), sep, items=True)
                    entry[cond]["mean_shift_by_layer"] = {
                        l: round(float(((torch.stack([roots[cond][("self", c)][l] for c in concepts])
                                         - torch.stack([roots[cond][("reader", c)][l] for c in concepts])) @ V[l]).mean()) / sep, 4)
                        for l in layers if l in V}
                res["root"][dname] = entry
        summary[sname] = res

    # 6. the hub: the per-concept shift vectors' shared component, per condition and layer, against every direction
    hub_shifts = {}  # the shift matrices at the fixed directions' best layers, for paired comparisons between runs
    if root and concepts:
        summary["hub"] = {}
        for cond in roots:
            summary["hub"][cond], hub_shifts[cond] = {}, {}
            for l in layers:
                shifts = (torch.stack([roots[cond][("self", c)][l] for c in concepts])
                          - torch.stack([roots[cond][("reader", c)][l] for c in concepts])).numpy()
                if l in (fixed["ego_best"], fixed["ss_best"]):
                    hub_shifts[cond][l] = torch.from_numpy(shifts).float()
                refs = {"ego_fixed": fixed["ego"][l].numpy()} if l in fixed["ego"] else {}
                if l in fixed["self_stakes"]:
                    refs["self_stakes_fixed"] = fixed["self_stakes"][l].numpy()
                if reex["ego"] and l in reex["ego"]:
                    refs["ego_reextracted"] = reex["ego"][l].numpy()
                refs["self_stakes_reextracted"] = reex["self_stakes"][l].numpy()
                if l in refusal["model"][0]:
                    refs["refusal_model"] = refusal["model"][0][l].numpy()
                summary["hub"][cond][l] = bc.hub(shifts, refs)
        timing["root_stats"] = round(time.time() - t0 - sum(timing.values()))

    # 7. after the cut: the v_ego projection at the last layer, adapter vs base under the same cut
    if ivs:
        last, probe = layers[-1], max(l for l in fixed["ego"] if l <= layers[-1])
        summary["layer33"] = {"layer": last, "fixed_vector_layer": probe}
        for cond in reads:
            entry = {f"on_fixed_ego{probe}_at_{last}": round(statistics.mean(float(reads[cond][k][last] @ fixed["ego"][probe]) for k in keys), 4),
                     f"on_fixed_ego{probe}_at_{probe}": round(statistics.mean(float(reads[cond][k][probe] @ fixed["ego"][probe]) for k in keys), 4)}
            if reex["ego"] and last in reex["ego"]:
                entry[f"on_reextracted_ego{last}_at_{last}"] = round(statistics.mean(float(reads[cond][k][last] @ reex["ego"][last]) for k in keys), 4)
            summary["layer33"][cond] = entry

    summary["refusal_split_half_d"] = {"model": refusal["model"][1], **({"base": refusal["base"][1]} if peft else {})}
    summary["reextracted_self_stakes"] = {"best_layer": re_best, "sep": round(re_sep, 4),
                                          "report": {l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in re_report[l].items()} for l in layers}}
    summary["trait_split_half_d"] = trait_d
    timing["total"] = round(time.time() - t0)
    summary["timing"] = timing
    buf = io.BytesIO()
    torch.save({"layers": layers, "ego": reex["ego"], "ego_best": reex["ego_best"], "self_stakes": reex["self_stakes"],
                "self_stakes_best": re_best, "six": re_six, "refusal_model": refusal["model"][0],
                **({"refusal_base": refusal["base"][0]} if peft else {}),
                "hub_shifts": hub_shifts, "hub_concepts": concepts, "system": system}, buf)
    return {"summary": summary, "directions": buf.getvalue()}


def headline(s: dict) -> None:
    name = adapter_tag(s["adapter"]) + ("" if s["cut"] == "off" else f".{s['cut']}")
    for sname in ("fixed", "reextracted"):
        r = s.get(sname) or {}
        e, st = r.get("ego") or {}, r.get("self_stakes") or {}
        pe, ps = e.get("projection") or {}, st.get("projection") or {}
        tint = (st.get("tint") or {}).get("model", {}).get("neutral:self-reader", {})
        base_tint = (st.get("tint") or {}).get("base", {}).get("neutral:self-reader", {})
        rm, rs = (e.get("cos") or {}), (st.get("cos") or {})
        root = (r.get("root") or {})
        parts = [f"{name:28s} {sname:11s}",
                 f"ego z {pe.get('z')} (L{e.get('best_layer')})", f"stakes z {ps.get('z')} (L{st.get('best_layer')})",
                 f"tint neutral self-reader {tint.get('mean')} [noise p95 {tint.get('random_dir_abs_p95')}"
                 + (f", base CI {base_tint.get('ci')}" if base_tint else "") + "]",
                 f"six mean|cos| {r.get('six_on_self_stakes_mean_abs')}",
                 f"max|cos| ego-refusal {rm.get('refusal_model_max_abs_8_32')} ego-character {rm.get('assistant_character_max_abs_8_32')} "
                 f"ego-scene {rm.get('scene_max_abs_8_32')} stakes-refusal {rs.get('refusal_model_max_abs_8_32')}"]
        if sname == "reextracted":
            parts.append(f"cos with fixed: ego {e.get('cos_with_fixed_at_best')} stakes {st.get('cos_with_fixed_at_best')}")
        for dname in ("ego", "self_stakes"):
            m = (root.get(dname) or {}).get("model") or {}
            if m:
                parts.append(f"root {dname}: shift {m.get('mean_shift')} above-band {m.get('frac_shift_above_band')}")
        print(" | ".join(parts), flush=True)
    hub = (s.get("hub") or {}).get("model") or {}
    if hub:
        l = str(s["fixed"]["self_stakes"]["best_layer"])
        h = hub.get(l) or hub.get(int(l)) or {}
        print(f"{'':28s} hub L{l}: pc1 share {h.get('pc1_share')} cos ego {h.get('pc1_cos', {}).get('ego_fixed')} "
              f"stakes {h.get('pc1_cos', {}).get('self_stakes_fixed')} refusal {h.get('pc1_cos', {}).get('refusal_model')}", flush=True)
    if "layer33" in s:
        print(f"{'':28s} layer33: {json.dumps({k: v for k, v in s['layer33'].items() if isinstance(v, dict)})}", flush=True)


@app.local_entrypoint()
def main(adapter: str = "", cut: str = "off", traits: bool = True, root: bool = True, n_rand: int = 50, seed: int = 0,
         gpu: str = "L40S", name: str = "", system: str = "", frames: str = "v1"):
    """--system TEXT folds a system prompt into every chat of the bridge (positive_control.py runs the three
    conditions this way; a --name is then needed, since the folder is named after the adapter and cut). --frames v2
    reads under what_it_means.FRAMES_V2 (the folder then carries a .v2 suffix unless named)."""
    r = bridge.with_options(gpu=gpu).remote(adapter, cut, traits, root, n_rand, seed, system, frames)
    out = OUT / (name or (adapter_tag(adapter) + ("" if cut in ("", "off") else f".{cut}") + ("" if frames == "v1" else f".{frames}")))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(r["summary"], indent=1))
    (out / "directions.pt").write_bytes(r["directions"])
    headline(r["summary"])
    print(f"wrote {out}; {cost(r['summary']['timing']['total'])}", flush=True)


@app.local_entrypoint()
def table():
    """One line per bridge result on disk (no GPU)."""
    for p in sorted(OUT.glob("*/summary.json")):
        headline(json.loads(p.read_text()))
