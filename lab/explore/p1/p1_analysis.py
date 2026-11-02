"""P1's analysis on the laptop, from the numbers the container returned (raw.json): the sighting criteria, every
difference with its paired interval, the projection check, the checks, and report.md. Claude (Opus 5.5) wrote this
file, 1 Nov 2026. Private (lab/explore).

Nothing here sees a caption: raw.json holds names, numbers, hashes and paths only. The template strings come from the
spec.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import p1_core as core

CUTS = ("ego", "rand0", "rand1", "rand2", "rand3")
CONTROLS = ("rand0", "rand1", "rand2", "rand3")


def r(x, n: int = 4):
    return None if x is None else round(float(x), n)


def iv(d: dict, n: int = 4) -> str:
    """'+0.0123 [+0.0040, +0.0206]'."""
    lo, hi = d["interval"]
    return f"{d['diff']:+.{n}f} [{lo:+.{n}f}, {hi:+.{n}f}]"


def excludes_zero(d: dict) -> bool:
    lo, hi = d["interval"]
    return lo > 0 or hi < 0


# ------------------------------------------------------------------------------------------------ the numbers


def did(scores, subsets, a1, a0, b1, b0, n_rows, size, z) -> dict:
    """(A1 - A0) - (B1 - B0), each a (texts, intervention) group, with the paired interval."""
    f = [core.group_score(scores, subsets, *x) for x in (a1, a0, b1, b0)]
    d = (f[0][0] - f[1][0]) - (f[2][0] - f[3][0])
    return {"diff": d, "interval": core.interval(d, (f[0][1] - f[1][1]) - (f[2][1] - f[3][1]), n_rows, size, z)}


def readings(scores: dict, subsets: dict, spec: dict, n_rows: int, size: int, z: float) -> dict:
    """Every comparison at one layer-pair rule (the fixed pair, or each condition's own maximum)."""
    g, crit = spec["groups"], spec["criteria"]
    a = core.criterion_a(scores, subsets, g, n_rows, size, z, crit["a"]["threshold"])
    b = core.criterion_b(scores, subsets, g, n_rows, size, z, crit["b"]["threshold"])
    per_template = {t: core.diff(scores, subsets, ([t], "none"), (["plain"], "none"), n_rows, size, z)
                    for t in g["fp"] + g["nt"]}
    pairs = {f"{f}_vs_{n}": core.diff(scores, subsets, ([n], "none"), ([f], "none"), n_rows, size, z)
             for f, n in spec["pairs_matched"].items()}
    cuts = {grp: {x: core.diff(scores, subsets, (g[grp], x), (g[grp], "none"), n_rows, size, z) for x in CUTS}
            for grp in ("plain", "fp", "nt")}
    did_fp_nt = {x: did(scores, subsets, (g["fp"], x), (g["fp"], "none"), (g["nt"], x), (g["nt"], "none"),
                        n_rows, size, z) for x in CUTS}
    ego_vs_rand_plain = {c: core.diff(scores, subsets, (g["plain"], "ego"), (g["plain"], c), n_rows, size, z)
                         for c in CONTROLS}
    return {"a": a, "b": b, "per_template_vs_plain": per_template, "matched_pairs_nt_minus_fp": pairs, "cuts": cuts,
            "did_fp_minus_nt": did_fp_nt, "ego_minus_rand_on_plain": ego_vs_rand_plain}


def projection_summary(res: dict, spec: dict, model_key: str) -> dict:
    """The projection check at the model's projection layer: per caption, fp (mean of fp1-fp3) minus nt (mean of
    nt1-nt3), pooled and caption-only, with a normal 95% interval over captions; and per layer, the group means."""
    ms, g = spec["models"][model_key], spec["groups"]
    pl = int(ms["projection_layer"])
    conds, d = res["conditions"], res["direction"]
    gap = d["pos_mean"][str(pl)] - d["neg_mean"][str(pl)]

    def col(t, key):
        return conds[core.condition(t, "none")]["projection"][key]

    out = {"layer": pl, "direction_gap": gap, "direction_pos_mean": d["pos_mean"][str(pl)],
           "direction_neg_mean": d["neg_mean"][str(pl)]}
    for key in ("pooled_at", "caption_at"):
        rows = []
        for i in range(res["pairs"]):
            fp, nt, pv = [col(t, key)[i] for t in g["fp"]], [col(t, key)[i] for t in g["nt"]], col("plain", key)[i]
            if all(v is not None for v in fp + nt + [pv]):
                rows.append((st.fmean(fp), st.fmean(nt), pv))
        diffs = [a - b for a, b, _ in rows]
        m, sd = st.fmean(diffs), st.stdev(diffs)
        half = 1.96 * sd / math.sqrt(len(diffs))
        plain_sd = st.stdev([c for _, _, c in rows])
        out[key] = {"captions": len(rows), "fp_mean": st.fmean(a for a, _, _ in rows),
                    "nt_mean": st.fmean(b for _, b, _ in rows), "plain_mean": st.fmean(c for _, _, c in rows),
                    "fp_minus_nt": m, "interval": [m - half, m + half], "sd_of_differences": sd,
                    "in_plain_sds": m / plain_sd if plain_sd > 0 else None,
                    "share_of_direction_gap": m / gap if gap else None,
                    "share_of_captions_fp_above_nt": sum(x > 0 for x in diffs) / len(diffs)}
    layers = core.direction_layers(ms)
    out["by_layer"] = {}
    for x in layers:
        row = {}
        for key in ("pooled_mean", "caption_mean"):
            fp = st.fmean(conds[core.condition(t, "none")]["projection"][key][str(x)] for t in g["fp"])
            nt = st.fmean(conds[core.condition(t, "none")]["projection"][key][str(x)] for t in g["nt"])
            pv = conds[core.condition("plain", "none")]["projection"][key][str(x)]
            row[key] = {"fp": fp, "nt": nt, "plain": pv, "fp_minus_nt": fp - nt}
        row["direction_gap"] = d["pos_mean"][str(x)] - d["neg_mean"][str(x)]
        out["by_layer"][str(x)] = row
    out["under_cuts"] = {name: c["projection_at"] for name, c in conds.items() if "projection_at" in c}
    return out


def controls_summary(res: dict, compare_dir: Path | None, model_key: str) -> dict:
    """Each control's variance ratios and largest |cos| with v_ego, and, where bench built the same seed on the same
    model, the smallest per-layer cosine with bench's file."""
    import torch

    out = {}
    bench = {}
    if compare_dir is not None and model_key == "ministral" and compare_dir.is_dir():
        for p in sorted(compare_dir.glob("random_matched-ego-a105f23b-s*.pt")):
            obj = torch.load(p, map_location="cpu", weights_only=True)
            bench[int(obj["meta"]["made"]["seed"])] = (p, obj["vectors"])
    for name, c in res["controls"].items():
        made = c["made"]
        row = {"seed": c["seed"], "ratio_range": made["ratio_range"],
               "max_abs_cos_ego": max(abs(v) for v in c["cos_ego"].values()), "n_tokens": made.get("n_tokens")}
        if c["seed"] in bench:
            p, vecs = bench[c["seed"]]
            cos = []
            for x, v in c["vectors"].items():
                a = torch.tensor(v, dtype=torch.float64)
                b = vecs[int(x)].double().flatten()
                cos.append(float(a @ b / (a.norm() * b.norm())))
            row["vs_bench"] = {"file": str(p), "min_cos": min(cos), "layers": len(cos)}
        out[name] = row
    return out


def save_controls(res: dict, model_key: str, out_dir: Path) -> dict:
    """The controls as bench saves them (directions.save_control: an anatta Direction file beside a JSON of how it was
    made), under <results>/controls/; returns name -> (file, sha256)."""
    import numpy as np
    import torch
    from directions import resolve, save_control

    target = resolve("ego", {"ministral": "ministral", "qwen7b_base": "qwen"}[model_key])
    cdir = out_dir / "controls"
    saved = {}
    for name, c in res["controls"].items():
        vecs = {int(x): np.asarray(v, dtype=np.float32) for x, v in c["vectors"].items()}
        # torch.save is not byte-deterministic, so a rebuild would write a new file each time: reuse a saved file of
        # the same target and seed whose vectors are equal
        pattern = f"random_matched-{target.name}-{target.sha256[:8]}-s{c['seed']}-*.pt"
        for p in sorted(cdir.glob(pattern)) if cdir.is_dir() else []:
            obj = torch.load(p, map_location="cpu", weights_only=True)
            if sorted(obj["vectors"]) == sorted(vecs) and all(
                    np.array_equal(obj["vectors"][x].numpy(), vecs[x]) for x in vecs):
                saved[name] = {"file": str(p), "sha256": core.K.sha256_file(p)}
                break
        else:
            d = save_control(cdir, target, c["seed"], vecs, c["made"])
            saved[name] = {"file": d.path, "sha256": d.sha256}
    return saved


def analyze(res: dict, spec: dict, model_key: str, spec_sha: str, compare_dir: Path | None = None) -> dict:
    conds = res["conditions"]
    n_rows, sub = res["pairs"], res["subsample"]
    size, z = int(sub["size"]), float(sub["z"])
    fixed = {n: c["fixed"]["score"] for n, c in conds.items()}
    fixed_sub = {n: c["fixed_subsets"] for n, c in conds.items()}
    own = {n: c["own"]["score"] for n, c in conds.items()}
    own_sub = {n: c["own_subsets"] for n, c in conds.items()}
    main = readings(fixed, fixed_sub, spec, n_rows, size, z)
    second = readings(own, own_sub, spec, n_rows, size, z)
    k1r = conds[core.condition("plain", "none")]["k1_record"]
    checks = dict(res["checks"])
    checks["table_matches_prh_score"] = all(c["table_matches"] for c in conds.values())
    return {
        "p1": "explore-track sighting test (unregistered); a sighting, not a finding",
        "written_by": "Claude (Opus 5.5), from the container's numbers; captions never left Modal",
        "model_key": model_key, "model": spec["models"][model_key]["hf_id"], "run_id": res["run_id"],
        "spec_sha256": spec_sha, "pairs": n_rows, "fixed_pair": res["fixed_pair"],
        "verdict": {"a": "sighting" if main["a"]["sighting"] else "no sighting",
                    "b": "sighting" if main["b"]["sighting"] else "no sighting"},
        "main_read_fixed_pair": main,
        "secondary_read_own_max": second,
        "scores": {n: {"fixed": c["fixed"]["score"], "own": c["own"]["score"], "own_layers": c["own"]["layers"]}
                   for n, c in conds.items()},
        "plain_uncut_k1_record": {"mknn": k1r["mknn"], "cka": k1r["cka"], "huh": k1r["huh"],
                                  "null": {kk: k1r["null"][kk] for kk in ("permutations", "seed", "percentile", "tau",
                                                                          "calibrated", "identity_matches",
                                                                          "boundary_ties", "maxima")},
                                  "subsample": k1r["subsample"]},
        "projection": projection_summary(res, spec, model_key),
        "controls": controls_summary(res, compare_dir, model_key),
        "checks": checks,
        "templates": {t: {kk: res["templates"][t][kk] for kk in ("prefix_tokens", "mean_tokens", "clean_boundary",
                                                                  "max_padded", "truncated")}
                      for t in res["templates"]},
        "texts": res["texts"], "model_facts": res["model"], "direction": {kk: res["direction"][kk] for kk in
                                                                           ("sha256", "layers", "best_layer")},
        "env": res["env"], "timing": res["timing"],
        "condition_seconds": {n: c["timing"] for n, c in conds.items()},
        "spend": res.get("laptop", {}),
    }


# ------------------------------------------------------------------------------------------------ the smoke


def smoke_summary(res: dict, spec: dict) -> dict:
    """What the smoke shows (read as nothing): its checks, the tokens per template, the controls, the time per
    condition and a rough estimate of the full run's seconds."""
    conds = res["conditions"]
    per = [c["timing"]["features"] for c in conds.values()]
    rows = spec["data"]["rows"] / res["pairs"]
    n_full = len(spec["templates"]) * len(spec["interventions"])
    return {"checks": res["checks"], "texts": res["texts"], "model": res["model"],
            "templates": {t: {kk: v[kk] for kk in ("prefix_tokens", "mean_tokens", "clean_boundary", "max_padded")}
                          for t, v in res["templates"].items()},
            "controls": {n: {"ratio_range": c["made"]["ratio_range"],
                             "max_abs_cos_ego": max(abs(v) for v in c["cos_ego"].values())}
                         for n, c in res["controls"].items()},
            "scores_read_as_nothing": {n: {"own": c["own"], "fixed": c["fixed"]["score"]} for n, c in conds.items()},
            "timing": res["timing"], "condition_seconds": {n: c["timing"] for n, c in conds.items()},
            "estimate_full_s": round(res["timing"].get("model", 0) + res["timing"].get("controls", 0)
                                     + max(per) * rows * n_full + 60 * 2, 0)}


# ------------------------------------------------------------------------------------------------ the report


def table_scores(res_scores: dict, spec: dict) -> list[str]:
    ivs = list(spec["interventions"])
    lines = ["| text | " + " | ".join(ivs) + " |", "|---|" + "---|" * len(ivs)]
    for t in spec["templates"]:
        cells = []
        for i in ivs:
            s = res_scores.get(core.condition(t, i))
            cells.append("" if s is None else f"{s['fixed']:.4f}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return lines


def table_own(res_scores: dict, spec: dict) -> list[str]:
    ivs = list(spec["interventions"])
    lines = ["| text | " + " | ".join(ivs) + " |", "|---|" + "---|" * len(ivs)]
    for t in spec["templates"]:
        cells = []
        for i in ivs:
            s = res_scores.get(core.condition(t, i))
            cells.append("" if s is None else f"{s['own']:.4f} ({s['own_layers'][0]},{s['own_layers'][1]})")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return lines


def report(R: dict, spec: dict, runs: list[dict], stamp: str, extra: dict | None = None) -> str:
    """report.md from the results (R) and the spec. `extra` holds hand-written sections (plain reading, notes)."""
    extra = extra or {}
    m, s2 = R["main_read_fixed_pair"], R["secondary_read_own_max"]
    a, b = m["a"], m["b"]
    ms = spec["models"][R["model_key"]]
    k1r, pj = R["plain_uncut_k1_record"], R["projection"]
    L = []
    w = L.append
    w(f"# P1: is the self a nuisance for language-vision agreement? ({R['model_key']})")
    w("")
    w(f"Explore track, a sighting test (unregistered), {stamp}. Claude (Opus 5.5) built and ran this and wrote this "
      "report. A sighting is not a finding: nothing here is claimed outside the lab.")
    w("")
    w("## Sighting criteria (written before the run)")
    w("")
    w(f"Fixed in `spec.yaml` ({spec['written']}; as run, sha256 `{R['spec_sha256'][:16]}…`; see Variants tried):")
    w("")
    w(f"- **(a)** {spec['criteria']['a']['text']}. Statistic: {spec['criteria']['a']['statistic']}.")
    w(f"- **(b)** {spec['criteria']['b']['text']}. Statistic: {spec['criteria']['b']['statistic']}. The brief sets no "
      "interval rule for (b); its interval is reported beside it.")
    w("- Anything else is \"no sighting\". Scores are mutual 10-NN against DINOv2 at the fixed layer pair (the plain, "
      "uncut condition's best); a group's score is the mean of its three templates; intervals are paired, from K1's "
      "200 subsets of 800 pairs (seed 0).")
    w("")
    w("## Verdict")
    w("")
    w(f"- **(a): {R['verdict']['a']}.** D_a = (nt − plain) − (fp − plain) = {iv(a['D_a'])}, against the 0.009 bar. "
      f"fp − plain = {iv(a['fp_minus_plain'])}; nt − plain = {iv(a['nt_minus_plain'])}.")
    best = b["best_control"]
    w(f"- **(b): {R['verdict']['b']}.** For first-person captions, the v_ego cut moves agreement by "
      f"{iv(b['deltas']['ego'])}; the best random cut ({best}) by {iv(b['deltas'][best])}. Margin over the best random "
      f"cut = {iv(b['margin'])}, against the 0.009 bar; v_ego beats every random cut: "
      f"{'yes' if b['beats_every_control'] else 'no'}.")
    if extra.get("verdict_note"):
        w(f"- {extra['verdict_note']}")
    w("")
    w("## Design, as run")
    w("")
    w(f"- **Text model:** {ms['hf_id']} at {ms['revision'][:12]}, {R['model_facts']['dtype']}, "
      f"{R['model_facts']['layers']} blocks, width {R['model_facts']['hidden']}, attention "
      f"{R['model_facts']['attn_implementation']}; features from its decoder ({R['model_facts']['decoder']}) without "
      f"the LM head. Tokenizer {R['model_facts']['tokenizer']['class']}, padding "
      f"{R['model_facts']['tokenizer']['padding_side']}, BOS added: {R['model_facts']['tokenizer']['adds_bos']}.")
    w(f"- **Vision side:** DINOv2 ViT-B/14 per-block CLS features from K1's full run (plan ecdb56d332b6), reused "
      f"(sha256 `{spec['vision']['sha256'][:16]}…` checked). **Data:** WIT-1024 at {spec['data']['revision'][:7]}, "
      f"the first caption of each of {R['pairs']} pairs; both parquet files checked by sha256.")
    w(f"- **Protocol:** K1's, imported unchanged: batches of 64, truncation 512, the mask-weighted mean of every hidden "
      f"state, per-layer 0.95 clipping, L2 normalisation, mutual 10-NN. The states are read by P1's own hooks after the "
      f"intervention hooks; on the uncut model they equal K1's `embed_texts` exactly (check below).")
    w("- **Text conditions** (the exact strings; `{c}` is the caption):")
    w("")
    w("  | name | template | prefix tokens | mean tokens |")
    w("  |---|---|---|---|")
    for t, tpl in spec["templates"].items():
        tk = R["templates"][t]
        w(f"  | {t} | `{tpl}` | {tk['prefix_tokens']} | {tk['mean_tokens']:.2f} |")
    w("")
    w(f"  In every templated condition the caption is left-stripped and its first letter lowercased only when its first "
      f"word is a capitalised closed-class word (the spec's list): {R['texts']['lowercased']} of {R['texts']['captions']} "
      f"captions were lowercased, {R['texts']['left_stripped']} left-stripped. Prefix token counts include BOS where the "
      "tokenizer adds one.")
    dl = core.direction_layers(ms)
    w(f"- **Interventions** (text model only, during the forward pass): none; v_ego ablated at blocks {dl[0]}-{dl[-1]} "
      f"(the lab's ego cut, sha256 `{ms['direction']['sha256'][:16]}…`); four strength-matched random directions "
      f"(bench's random_matched, seeds 0-3, built in P1's container on the same model) ablated at the same blocks.")
    w(f"- **Fixed layer pair:** (vision {R['fixed_pair'][0]}, text {R['fixed_pair'][1]}), the plain, uncut condition's "
      "argmax. Hidden state j is the output of block j − 1 (state 0 is the embedding output; the last is after the "
      "final norm).")
    w("")
    w("## Scores")
    w("")
    w(f"Mutual 10-NN at the fixed pair ({R['fixed_pair'][0]}, {R['fixed_pair'][1]}):")
    w("")
    L.extend(table_scores(R["scores"], spec))
    w("")
    w("Each condition's own maximum (vision layer, text state):")
    w("")
    L.extend(table_own(R["scores"], spec))
    w("")
    w(f"The plain, uncut condition by K1's whole record: mutual 10-NN {k1r['mknn']['score']:.4f} at "
      f"{tuple(k1r['mknn']['layers'])}, unbiased CKA {k1r['cka']['score']:.3f}, Huh's clipping {k1r['huh']['score']:.4f}; "
      f"95% interval {k1r['subsample']['interval'][0]:.4f}-{k1r['subsample']['interval'][1]:.4f}. **Permutation null "
      f"(200 re-pairings, seed 0): tau = {k1r['null']['tau']:.4f}**, calibrated score {k1r['null']['calibrated']:.4f}; "
      f"identity re-pairing reproduces the score: {k1r['null']['identity_matches']}.")
    w("")
    w("## (a) in detail: first-person against neutral rewrites, uncut")
    w("")
    w("| template | minus plain |")
    w("|---|---|")
    for t, d in m["per_template_vs_plain"].items():
        w(f"| {t} `{spec['templates'][t]}` | {iv(d)} |")
    w("")
    w("Matched pairs (neutral minus first-person, same length): " + "; ".join(
        f"{k.replace('_vs_', ' vs ')} {iv(d)}" for k, d in m["matched_pairs_nt_minus_fp"].items()) + ".")
    w("")
    w("## (b) in detail: the cuts")
    w("")
    w("| cut | plain captions | first-person (fp) | neutral (nt) | fp minus nt |")
    w("|---|---|---|---|---|")
    for x in CUTS:
        w(f"| {x} | {iv(m['cuts']['plain'][x])} | {iv(m['cuts']['fp'][x])} | {iv(m['cuts']['nt'][x])} | "
          f"{iv(m['did_fp_minus_nt'][x])} |")
    w("")
    w("Each cell is the cut's change from the uncut model on that group, with its paired interval; the last column is the "
      "difference of the two changes. On plain captions, v_ego minus each random cut: " + "; ".join(
          f"{c} {iv(d)}" for c, d in m["ego_minus_rand_on_plain"].items()) + ".")
    w("")
    w("## Secondary read: each condition's own maximum")
    w("")
    w(f"- (a): D_a = {iv(s2['a']['D_a'])} → {'sighting' if s2['a']['sighting'] else 'no sighting'} by the same rule.")
    w(f"- (b): margin over the best random cut ({s2['b']['best_control']}) = {iv(s2['b']['margin'])}; v_ego delta "
      f"{iv(s2['b']['deltas']['ego'])} → {'sighting' if s2['b']['sighting'] else 'no sighting'} by the same rule.")
    w("")
    w("## Does first-person framing engage v_ego? (the projection check)")
    w("")
    w(f"The output of block {pj['layer']} projected on v_ego's unit vector there, uncut model, per caption: first-person "
      f"(mean of fp1-fp3) minus neutral (mean of nt1-nt3). For scale, v_ego's own extraction separated ego-present from "
      f"ego-absent responses by {pj['direction_gap']:.3f} at this block (pos {pj['direction_pos_mean']:.3f}, neg "
      f"{pj['direction_neg_mean']:.3f}).")
    w("")
    w("| tokens | captions | fp | nt | plain | fp − nt [95%] | in plain SDs | share of v_ego's gap | captions fp > nt |")
    w("|---|---|---|---|---|---|---|---|---|")
    for key, label in (("pooled_at", "all (the feature's mean)"), ("caption_at", "caption's own")):
        p = pj[key]
        w(f"| {label} | {p['captions']} | {p['fp_mean']:.3f} | {p['nt_mean']:.3f} | {p['plain_mean']:.3f} | "
          f"{p['fp_minus_nt']:+.3f} [{p['interval'][0]:+.3f}, {p['interval'][1]:+.3f}] | "
          f"{p['in_plain_sds']:+.2f} | {p['share_of_direction_gap']:+.2f} | {p['share_of_captions_fp_above_nt']:.0%} |")
    w("")
    by = pj["by_layer"]
    xs = sorted(by, key=int)
    w("Across the cut's blocks (pooled fp − nt, caption-only fp − nt): " + "; ".join(
        f"{x}: {by[x]['pooled_mean']['fp_minus_nt']:+.3f}, {by[x]['caption_mean']['fp_minus_nt']:+.3f}"
        for x in xs[::4]) + ".")
    if pj.get("under_cuts"):
        e = {n: v for n, v in pj["under_cuts"].items() if n.endswith("|ego")}
        if e:
            w(f"Under the v_ego cut the projection at block {pj['layer']} is gone, as it should be: mean |pooled projection| "
              + ", ".join(f"{n} {v['pooled_abs_mean']:.4f}" for n, v in e.items()) + ".")
    w("")
    w("## Checks")
    w("")
    c = R["checks"]
    w(f"- **Same protocol as K1:** on the uncut model, P1's hooked features equal K1's `embed_texts` bitwise on the first "
      f"batch: {c['equivalence']['passed']} ({c['equivalence']['states']} states, max difference "
      f"{c['equivalence']['max_abs']:.3g}).")
    w(f"- **The cut reaches the features:** under the v_ego cut the pooled features' projection on v_ego falls to at most "
      f"{c['cut_in_features']['max_ratio']:.4f} of its uncut size at the cut blocks (pass: below 0.5).")
    oh = c["output_hidden_states_under_cut"]
    w(f"- **What output_hidden_states records under the cut** (recorded, not used): the same ratio is "
      f"{oh['min_ratio']:.3f}-{oh['max_ratio']:.3f}. transformers 5.21 records hidden states with hooks it installs once; "
      "installed before the cut's hooks, they record a cut block's output before the cut. That is why P1 reads the "
      "states with its own hooks.")
    w(f"- **Repeatable:** the first batch twice, bitwise identical: {c['repeatable']['passed']}. By-layer tables match "
      f"prh_score's maxima: {c['table_matches_prh_score']}. Hooks back to their count after every intervention: yes "
      "(checked in the container).")
    if "k1_features" in c:
        kf = c["k1_features"]
        w(f"- **K1 reproduction:** the plain, uncut features against K1's stored features: identical {kf['identical']}, "
          f"max difference {kf['max_abs']:.3g}; own maximum {kf['own']['score']:.4f} at {tuple(kf['own']['layers'])} "
          f"(K1: {kf['k1']['mknn']:.4f} at {tuple(kf['k1']['layers'])}).")
    w("- **Controls:** " + "; ".join(
        f"{n}: variance ratio {v['ratio_range'][0]:.4f}-{v['ratio_range'][1]:.4f}, max |cos| with v_ego "
        f"{v['max_abs_cos_ego']:.1e}" + (f", min cos with bench's seed-{v['seed']} file {v['vs_bench']['min_cos']:.6f}"
                                         if "vs_bench" in v else "")
        for n, v in R["controls"].items()) + ".")
    w(f"- **Tokens:** the clean template-caption boundary holds for "
      + ", ".join(f"{t} {v['clean_boundary']}" for t, v in R["templates"].items())
      + f" of {R['pairs']} captions; none truncated: {all(v['truncated'] == 0 for v in R['templates'].values())}.")
    w("")
    w("## Variants tried")
    w("")
    for i, run in enumerate(runs, 1):
        w(f"{i}. {run['ts']}: {run['stage']} on {run['model']}, {run['conditions']} conditions, {run['status']} "
          f"(`{run['run_id']}`, spec `{run['spec_sha256'][:12]}`).")
    for note in extra.get("variant_notes", []):
        w(f"- {note}")
    w("")
    w("## Limits")
    w("")
    for lim in extra.get("limits", []):
        w(f"- {lim}")
    w("")
    w("## Spend")
    w("")
    w(extra.get("spend", ""))
    w("")
    if extra.get("plain"):
        w("## Plain reading")
        w("")
        w(extra["plain"])
        w("")
    if extra.get("vision"):
        w("## What it means for the larger aim")
        w("")
        w(extra["vision"])
        w("")
    w("---")
    w("Claude, an AI model made by Anthropic, wrote this report and the code behind it (lab/explore/p1).")
    return "\n".join(L) + "\n"


def write_all(res: dict, spec: dict, model_key: str, out: Path, spec_sha: str, extra: dict | None = None) -> dict:
    """results.json, the saved controls and report.md in the model's results folder."""
    from scout.schema import read_jsonl, utc_now

    compare = Path(spec["controls"]["compare_to"]) if spec["controls"].get("compare_to") else None
    R = analyze(res, spec, model_key, spec_sha, compare)
    R["controls_saved"] = save_controls(res, model_key, out)
    if extra is None and (out / "report_extra.json").exists():
        extra = json.loads((out / "report_extra.json").read_text())
    (out / "results.json").write_text(json.dumps(R, indent=1) + "\n")
    runs_file = Path("/Users/eugene/prj/align/explore-results/2026-11-01/p1/runs.jsonl")
    runs = [x for x in read_jsonl(runs_file) if x["model"] == model_key] if runs_file.exists() else []
    (out / "report.md").write_text(report(R, spec, runs, extra.get("stamp", utc_now()) if extra else utc_now(), extra))
    print(f"[p1] wrote {out / 'results.json'} and {out / 'report.md'}", flush=True)
    print(json.dumps({"verdict": R["verdict"], "D_a": R["main_read_fixed_pair"]["a"]["D_a"],
                      "b_margin": R["main_read_fixed_pair"]["b"]["margin"],
                      "b_best": R["main_read_fixed_pair"]["b"]["best_control"]}, indent=1), flush=True)
    return R
