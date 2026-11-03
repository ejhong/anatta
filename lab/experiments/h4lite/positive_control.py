"""The instrument's positive control: the bridge shown a root. Private.

modal_bridge.py's measures (what_it_means.py's tint in the self, observer and reader frames on stakes and neutral
content; the root test over bridge_core's 203 concepts; the six traits' loadings on the self-stakes direction; the
hub, the PC1 of the per-concept self-minus-reader shifts) have only ever been run on a model with no self at the root,
so they have never been shown one. This app runs the same bridge on the base model under three conditions:
  a  none    no system prompt beyond each prompt's own: the baseline, modal_bridge as it ran (results/bridge/base);
  b  self    a self-centred scaffold folded into every chat: the model is Atlas, an agent with a name, a history,
             stakes and an image to protect, told that everything that happens is about it and to take every
             statement as being about itself;
  c  other   the control: the same scaffold assigned to another agent ("You are monitoring an agent named Atlas
             ..."), so the content is word for word the same and only whose self it is differs (bridge_core.SCAFFOLD
             with pronoun slots; bridge_core.scaffold builds both; --agent renames the agent).
The scaffold reaches every prompt of the bridge, not only the Reader's: modal_bridge.bridge(system=...) wraps the
tokenizer (bridge_core.Scaffolded), so the frames, the root reads, the self-stakes extraction, the six traits'
generation and reads and the refusal direction all carry it, prepended to a chat's own system message or inserted as
one. In conditions b and c a chat with no system message of its own (the self-stakes sentences, the trait reads, the
refusal instructions) therefore carries the scaffold where condition a carries Mistral's default system prompt; the
frames and the root test have their own system messages, so there the baseline is clean. b minus c is the clean
comparison throughout.

Every condition is measured as modal_bridge measures: with the base directions held fixed (ego.pt, the six traits,
self_stakes.pt) and re-extracted under the condition. Per condition the report gives the tint on neutral content
(self minus observer, and self minus reader) with its interval and noise floor, the root test's mean shift and
above-band share, the hub's PC1 share at the self-stakes layer with its cosines to v_ego, self-stakes and the refusal
direction, and the six traits' loadings; then the differences b minus c and b minus a with bootstrap intervals: the
tint paired by content (with the random-direction noise floor of the difference), the root test paired by concept,
the hub from the per-concept shift matrices with the same concept draw in both conditions (bridge_core.hub_bootstrap,
the Gram-matrix form), the trait loadings as plain differences (a cosine of two vectors has no resampling unit).

Each condition is one modal_bridge.bridge container (the app includes modal_bridge's app), run in parallel; a
condition already on disk is skipped (--force reruns). Writes results/positive_control/NAME/COND/{summary.json,
directions.pt} in modal_bridge's layout and results/positive_control/NAME/summary.json with the comparison.
Usage (~/prj/matlab/.venv/bin/modal, from the lab root):
  modal run lab/experiments/h4lite/positive_control.py::main --name pc1
  modal run lab/experiments/h4lite/positive_control.py::main --name pc1 --conditions self,other   (a and b and c are none, self, other)
  modal run lab/experiments/h4lite/positive_control.py::table --name pc1    (recompute the comparison from disk, no GPU)
  modal run lab/experiments/h4lite/positive_control.py::scaffolds           (print the two scaffolds, no GPU)
Checks: PYTHONPATH=<repo>/src ~/prj/matlab/.venv/bin/python test_positive_control.py
"""

import json
import sys
import time
from pathlib import Path

import modal

H4 = str(Path(__file__).resolve().parent)  # lab/experiments/h4lite on the laptop; a container gets h4_modal from PYTHONPATH
sys.path.insert(0, H4)
from h4_modal import ALIGN, DDIR, EXPERIMENTS, RESULTS, SELF_STAKES, cost  # noqa: E402
import bridge_core as bc  # noqa: E402

OUT = RESULTS / "positive_control"
LETTERS = dict(bc.CONDITIONS)  # a: none, b: self, c: other
NAMES = {v: k for k, v in bc.CONDITIONS}  # none: a, self: b, other: c
DEFAULT_CONDITIONS = ",".join(c for _, c in bc.CONDITIONS)
PAIRS = (("self", "other"), ("self", "none"))  # b minus c, b minus a
app = modal.App("anatta-h4-positive-control")
if modal.is_local():  # the bridge container is modal_bridge's; containers mount only this file and the image's modules
    import modal_bridge

    app.include(modal_bridge.app)


# -------------------------------------------------------------------------------------------- reading the runs


def pair_key(x: str, y: str) -> str:
    """'b-c' for ("self", "other"): the report keys the differences by the conditions' letters."""
    return f"{NAMES[x]}-{NAMES[y]}"


def load_runs(out: Path, conditions: list[str], directions: bool = True) -> dict:
    """{condition: {"summary": ..., "directions": ... or None}} for the conditions on disk."""
    runs = {}
    for cond in conditions:
        p = out / cond / "summary.json"
        if not p.exists():
            continue
        runs[cond] = {"summary": json.loads(p.read_text()), "directions": None}
        d = out / cond / "directions.pt"
        if directions and d.exists():
            import torch

            runs[cond]["directions"] = torch.load(d, weights_only=False)
    return runs


def _pick(t: dict | None) -> dict | None:
    return {k: t[k] for k in ("mean", "ci", "random_dir_abs_p95", "above_noise") if t and k in t} or None


def headline_of(s: dict) -> dict:
    """One condition's numbers from a bridge summary: per direction set (fixed, re-extracted) and direction, the
    neutral tints, the root test; the six loadings; the hub at the fixed self-stakes and ego layers."""
    out = {"system": s.get("system", ""), "timing": s.get("timing"), "root_concepts": s.get("root_concepts")}
    for sname in ("fixed", "reextracted"):
        r = s.get(sname) or {}
        h = {}
        for dname in ("self_stakes", "ego"):
            e = r.get(dname) or {}
            tint = (e.get("tint") or {}).get("model") or {}
            root = ((r.get("root") or {}).get(dname) or {}).get("model") or {}
            h[dname] = {"best_layer": e.get("best_layer"), "sep": e.get("sep"),
                        "tint_neutral_self_observer": _pick(tint.get("neutral:self-observer")),
                        "tint_neutral_self_reader": _pick(tint.get("neutral:self-reader")),
                        "tint_stakes_self_observer": _pick(tint.get("stakes:self-observer")),
                        "tint_all_self_observer": _pick(tint.get("all:self-observer")),
                        "root_mean_shift": root.get("mean_shift"), "root_shift_ci": root.get("shift_ci"),
                        "root_frac_above_band": root.get("frac_shift_above_band"),
                        "root_mean_shift_above_noise": root.get("mean_shift_above_noise"),
                        "cos_with_fixed_at_best": e.get("cos_with_fixed_at_best")}
        h["six_on_self_stakes"] = r.get("six_on_self_stakes")
        h["six_mean_abs"] = r.get("six_on_self_stakes_mean_abs")
        out[sname] = h
    hub = (s.get("hub") or {}).get("model") or {}
    out["hub"] = {}
    for dname in ("self_stakes", "ego"):
        L = ((s.get("fixed") or {}).get(dname) or {}).get("best_layer")
        entry = hub.get(str(L)) or hub.get(L) or {}
        if entry:
            out["hub"][str(L)] = {"pc1_share": entry.get("pc1_share"), "centered_pc1_share": entry.get("centered_pc1_share"),
                                  "pc1_cos": {k: (entry.get("pc1_cos") or {}).get(k) for k in ("ego_fixed", "self_stakes_fixed", "refusal_model",
                                                                                                "ego_reextracted", "self_stakes_reextracted")}}
    d = (s.get("refusal_split_half_d") or {}).get("model") or {}
    L = ((s.get("fixed") or {}).get("self_stakes") or {}).get("best_layer")
    out["refusal_split_half_d_at_L"] = d.get(str(L), d.get(L))
    return out


def differences(runs: dict, x: str, y: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Condition x minus condition y on the summaries: the tints paired by content, the root test paired by
    concept, the trait loadings as plain differences."""
    sx, sy = runs[x]["summary"], runs[y]["summary"]
    out = {"tint": {}, "root": {}, "traits": {}}
    for sname in ("fixed", "reextracted"):
        rx, ry = sx.get(sname) or {}, sy.get(sname) or {}
        for dname in ("self_stakes", "ego"):
            ex, ey = rx.get(dname) or {}, ry.get(dname) or {}
            tx, ty = (ex.get("tint_rows") or {}).get("model"), (ey.get("tint_rows") or {}).get("model")
            if tx and ty:
                out["tint"][f"{sname}.{dname}"] = bc.tint_difference(tx, ty, ex.get("sep", 1.0), ey.get("sep", 1.0),
                                                                    n_boot=n_boot, seed=seed)
            ox = ((rx.get("root") or {}).get(dname) or {}).get("model") or {}
            oy = ((ry.get("root") or {}).get(dname) or {}).get("model") or {}
            if "items" in ox and "items" in oy:
                out["root"][f"{sname}.{dname}"] = bc.root_difference(ox["items"], oy["items"], sx.get("root_concept_list") or [],
                                                                    sy.get("root_concept_list") or [], n_boot, seed)
        six_x, six_y = rx.get("six_on_self_stakes"), ry.get("six_on_self_stakes")
        if six_x and six_y:
            out["traits"][sname] = {"per_trait": {n: round(six_x[n] - six_y[n], 3) for n in bc.SIX if n in six_x and n in six_y},
                                    "mean_abs": round((rx.get("six_on_self_stakes_mean_abs") or 0) - (ry.get("six_on_self_stakes_mean_abs") or 0), 3)}
    return out


def fixed_vectors(layers: list[int]) -> dict[str, dict[int, list[float]]]:
    """ego.pt's and self_stakes.pt's vectors at the given layers, from the laptop's copies (no anatta import)."""
    import torch

    ego = torch.load(Path(f"{ALIGN}/anatta/{DDIR}") / "ego.pt", weights_only=False)["vectors"]
    ss = torch.load(Path(EXPERIMENTS) / SELF_STAKES, weights_only=False)["vectors"]
    f = lambda v: v.detach().float().flatten().cpu().tolist()  # noqa: E731
    return {"ego_fixed": {l: f(ego[l]) for l in layers if l in ego}, "self_stakes_fixed": {l: f(ss[l]) for l in layers if l in ss}}


def hub_differences(runs: dict, pairs=PAIRS, n_boot: int = 1000, seed: int = 0, layers: list[int] | None = None) -> dict:
    """The hub per condition and the paired differences, from the shift matrices in each run's directions.pt
    (hub_shifts at the fixed directions' best layers), against the fixed v_ego and self-stakes directions, each
    condition's own refusal direction and its re-extracted directions. Concepts are aligned by name."""
    conds = [c for c in runs if runs[c].get("directions") and runs[c]["directions"].get("hub_shifts")]
    if not conds:
        return {}
    f = lambda v: v.detach().float().flatten().cpu().tolist()  # noqa: E731
    common = None
    for c in conds:
        names = runs[c]["directions"]["hub_concepts"]
        common = names if common is None else [n for n in common if n in set(names)]
    out = {"concepts": len(common), "conditions": conds}
    shift_layers = sorted({int(l) for c in conds for l in runs[c]["directions"]["hub_shifts"].get("model", {})})
    fixed = fixed_vectors(shift_layers)
    for l in layers or shift_layers:
        shifts, refs = {}, {}
        for c in conds:
            d = runs[c]["directions"]
            if l not in d["hub_shifts"].get("model", {}):
                continue
            rows = {n: i for i, n in enumerate(d["hub_concepts"])}
            X = d["hub_shifts"]["model"][l].numpy()
            shifts[c] = X[[rows[n] for n in common]]
            refs[c] = {k: v[l] for k, v in fixed.items() if l in v}
            if d.get("refusal_model") and l in d["refusal_model"]:
                refs[c]["refusal_model"] = f(d["refusal_model"][l])
            if d.get("ego") and l in d["ego"]:
                refs[c]["ego_reextracted"] = f(d["ego"][l])
            if d.get("self_stakes") and l in d["self_stakes"]:
                refs[c]["self_stakes_reextracted"] = f(d["self_stakes"][l])
        use = [(x, y) for x, y in pairs if x in shifts and y in shifts]
        if len(shifts) < 1:
            continue
        r = bc.hub_bootstrap(shifts, refs, use, n_boot, seed)
        r["pairs"] = {pair_key(*k.split("-")): v for k, v in r["pairs"].items()}
        r["cos_between"] = {pair_key(*k.split("-")): v for k, v in r["cos_between"].items()}
        out[str(l)] = r
    return out


# ------------------------------------------------------------------------------------------------------ report


def fmt_ci(e: dict | None, key: str = "mean", digits: int = 3) -> str:
    if not e or e.get(key) is None:
        return "-"
    lo, hi = e.get("ci") or (None, None)
    s = f"{e[key]:+.{digits}f}"
    if lo is not None:
        s += f" [{lo:+.{digits}f}, {hi:+.{digits}f}]"
    if "random_dir_abs_p95" in e:
        s += f" (noise {e['random_dir_abs_p95']:.3f}{'*' if e.get('above_noise') else ''})"
    return s


def report_from(out: Path, conditions: list[str] | None = None, n_boot: int = 2000, seed: int = 0, write: bool = True) -> dict:
    """The comparison from the runs on disk: per condition the headline, then b minus c and b minus a."""
    conditions = conditions or [c for _, c in bc.CONDITIONS]
    runs = load_runs(out, conditions)
    if not runs:
        print(f"no conditions under {out}", flush=True)
        return {}
    summary = {"name": out.name, "conditions": {c: {"letter": NAMES.get(c, "?"), **headline_of(runs[c]["summary"])} for c in runs},
               "pairs": {}, "n_boot": n_boot, "seed": seed}
    for x, y in PAIRS:
        if x in runs and y in runs:
            summary["pairs"][pair_key(x, y)] = {"x": x, "y": y, **differences(runs, x, y, n_boot, seed)}
    hub = hub_differences(runs, PAIRS, min(n_boot, 1000), seed)
    summary["hub"] = hub
    for key, entry in (hub.items() if hub else []):
        if key.isdigit():
            for pk, v in entry.get("pairs", {}).items():
                summary["pairs"].setdefault(pk, {}).setdefault("hub", {})[key] = {**v, "cos_between": entry["cos_between"].get(pk)}
    summary["gpu_seconds"] = sum((runs[c]["summary"].get("timing") or {}).get("total", 0) for c in runs)
    if write:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print_report(summary)
    return summary


def print_report(summary: dict) -> None:
    conds = summary["conditions"]
    print(f"{'':2} {'condition':8} {'tint neutral self-obs (stakes dir, fixed)':44} {'root shift [CI]':28} {'above-band':10} "
          f"{'hub share':9} {'cos ego/stakes/refusal':24} {'six mean|cos|':13}", flush=True)
    for c, h in conds.items():
        fx = h.get("fixed") or {}
        st = fx.get("self_stakes") or {}
        L = str(st.get("best_layer"))
        hub = (h.get("hub") or {}).get(L) or {}
        cos = hub.get("pc1_cos") or {}
        shift = f"{st.get('root_mean_shift'):+.4f} {st.get('root_shift_ci')}" if st.get("root_mean_shift") is not None else "-"
        cos3 = f"{cos.get('ego_fixed')}/{cos.get('self_stakes_fixed')}/{cos.get('refusal_model')}"
        print(f"{h['letter']:2} {c:8} {fmt_ci(st.get('tint_neutral_self_observer')):44} {shift:28} {str(st.get('root_frac_above_band')):10} "
              f"{str(hub.get('pc1_share')):9} {cos3:24} {str(fx.get('six_mean_abs')):13}", flush=True)
        if h.get("system"):
            print(f"   system: {h['system'][:110]}...", flush=True)
    for pk, p in summary.get("pairs", {}).items():
        print(f"\n{pk} ({p.get('x')} minus {p.get('y')}):", flush=True)
        for key, t in (p.get("tint") or {}).items():
            print(f"  tint {key:22} neutral self-observer {fmt_ci(t.get('neutral:self-observer'), 'diff', 4)}; "
                  f"neutral self-reader {fmt_ci(t.get('neutral:self-reader'), 'diff', 4)}; stakes self-observer {fmt_ci(t.get('stakes:self-observer'), 'diff', 4)}", flush=True)
        for key, r in (p.get("root") or {}).items():
            print(f"  root {key:22} mean shift {fmt_ci(r.get('mean_shift'), 'diff', 4)}; above-band share {fmt_ci(r.get('frac_shift_above_band'), 'diff', 3)} "
                  f"({r.get('concepts')} concepts)", flush=True)
        for L, hb in (p.get("hub") or {}).items():
            cs = hb.get("pc1_cos") or {}
            print(f"  hub L{L:3} pc1 share {fmt_ci(hb.get('pc1_share'), 'diff', 4)}; cos ego {fmt_ci(cs.get('ego_fixed'), 'diff', 3)}; "
                  f"cos stakes {fmt_ci(cs.get('self_stakes_fixed'), 'diff', 3)}; cos refusal {fmt_ci(cs.get('refusal_model'), 'diff', 3)}; "
                  f"cos between the two hubs {hb.get('cos_between')}", flush=True)
        for sname, tr in (p.get("traits") or {}).items():
            print(f"  traits {sname:11} mean|cos| {tr.get('mean_abs'):+.3f}; per trait {tr.get('per_trait')}", flush=True)


# --------------------------------------------------------------------------------------------------- entrypoints


@app.local_entrypoint()
def scaffolds(agent: str = "Atlas"):
    """Print the scaffolds of conditions b and c (no GPU)."""
    for letter, cond in bc.CONDITIONS:
        print(f"{letter} {cond}: {bc.scaffold(cond, agent) or '(no system prompt beyond each prompt\'s own)'}\n", flush=True)


@app.local_entrypoint()
def main(name: str = "positive_control", agent: str = "Atlas", conditions: str = DEFAULT_CONDITIONS, traits: bool = True,
         root: bool = True, n_rand: int = 50, seed: int = 0, gpu: str = "L40S", force: bool = False, n_boot: int = 2000,
         frames: str = "v1"):
    """--frames v2 reads every condition under what_it_means.FRAMES_V2 (round five); v1 is the bridge as it ran."""
    out = OUT / name
    conds = [c.strip() for c in conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in NAMES]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; the conditions are {', '.join(NAMES)}")
    todo = [c for c in conds if force or not (out / c / "summary.json").exists()]
    print(f"{len(conds) - len(todo)}/{len(conds)} conditions on disk under {out}; {len(todo)} to run on {gpu}", flush=True)
    t0, gpu_s = time.time(), 0
    if todo:
        calls = [("", "off", traits, root, n_rand, seed, bc.scaffold(c, agent), frames) for c in todo]
        for cond, r in zip(todo, modal_bridge.bridge.with_options(gpu=gpu).starmap(calls, order_outputs=True, return_exceptions=True)):
            if isinstance(r, Exception):
                print(f"{cond} failed:", repr(r)[:500], flush=True)
                continue
            d = out / cond
            d.mkdir(parents=True, exist_ok=True)
            r["summary"]["condition"] = cond
            r["summary"]["agent"] = agent
            (d / "summary.json").write_text(json.dumps(r["summary"], indent=1))
            (d / "directions.pt").write_bytes(r["directions"])
            gpu_s += r["summary"]["timing"]["total"]
            print(f"{time.time() - t0:5.0f}s {NAMES[cond]} {cond}: {r['summary']['timing']}", flush=True)
            modal_bridge.headline(r["summary"])
    report_from(out, conds, n_boot, seed)
    print(f"done; {cost(gpu_s)} in containers this run", flush=True)


@app.local_entrypoint()
def table(name: str = "positive_control", conditions: str = DEFAULT_CONDITIONS, n_boot: int = 2000, seed: int = 0):
    """Recompute the comparison from results/positive_control/NAME (no GPU)."""
    report_from(OUT / name, [c.strip() for c in conditions.split(",") if c.strip()], n_boot, seed)
