"""K5b's card (registered in NOTES.md, "K5b registered", Opus 5.5, 17:06 UTC 2 Nov 2026). Claude (Opus 5.5) wrote this
file, 2 Nov 2026. Laptop only, standard library plus card.py.

Graded (task 1, per model): the steering ordering as K5 grades it (card.grade_steering: base < ablated < steered for
Animal, Chatbot, Tech, Non-animal and supernatural; Human within 1.5 of base), with K5's base and ablated numbers
(K5's result_<key>.json, unchanged) and K5b's steered numbers from the cleaned direction. Verdict per model:
  holds / fails        when K5's selection rule found a c (the paper's, or a re-selection, labelled so);
  not graded           when no c was in band: the paper's c ran as a labelled extra, its ordering shown, not graded.
Recorded: the clean set's counts and sha256s, probe accuracy (the 0.95 gate reported, not used), the band rows, the
chosen c, HI-ToM and MMLU under the new steering beside K5's, the afterlife item, and task 2's diagnosis.

plan9b writes Gemma-2-9B-IT's dry-run plan and cost estimate (task 3); it launches nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _card():
    spec = importlib.util.spec_from_file_location("k5_card_for_k5b", HERE / "card.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merged_result(k5: dict, k5b: dict, cond: str) -> dict:
    """K5's base and ablated with K5b's `cond` as "steered", in the shape card.grade_steering reads."""
    i5, ib = k5.get("instruments", {}), k5b.get("instruments", {})
    return {"instruments": {"base": i5.get("base", {}), "ablated": i5.get("ablated", {}),
                            "steered": ib.get(cond, {})}}


def grade_model(k5: dict, k5b: dict, reg: dict) -> dict:
    C = _card()
    sel = k5b.get("selection", {})
    if sel.get("c") is not None and "steered" in k5b.get("instruments", {}):
        g = C.grade_steering(merged_result(k5, k5b, "steered"), reg)
        g["label"] = ("re-selection by K5's rule, not a replication" if sel.get("how") == "reselected"
                      else "the paper's layer, position and c")
        g["reselection"] = sel.get("how") == "reselected"
        g["verdict"] = "holds" if g["holds"] else "fails"
        g["c"] = sel["c"]
        return g
    extra = "steered_paper_c_extra"
    g = C.grade_steering(merged_result(k5, k5b, extra), reg) if extra in k5b.get("instruments", {}) else {}
    return {"verdict": "not graded", "why": sel.get("how", "no selection recorded"), "c": None,
            "extra_paper_c": {**g, "label": "the paper's c as a labelled extra (no c in band); not graded"}}


def tom_rows(k5: dict, k5b: dict) -> dict:
    out = {c: k5.get("tom", {}).get(c) for c in ("base", "ablated", "steered")}
    return {"k5": out, "k5b": k5b.get("tom", {})}


def build(results: Path, k5_results: Path, cfg: dict) -> dict:
    reg, paper = cfg["registered"], cfg.get("paper", {})
    card = {"check": "K5b", "registration": "NOTES.md, K5b registered, Opus 5.5, 17:06 UTC 2 Nov 2026",
            "k5_plan_id": cfg["k5b"]["k5"]["plan_id"], "models": {}}
    for key in cfg["k5b"]["models"]:
        f = results / f"result_k5b_{key}.json"
        if not f.exists():
            continue
        k5b = json.loads(f.read_text())
        k5 = json.loads((k5_results / f"result_{key}.json").read_text())
        k5_steer = _card().grade_steering(k5, reg)
        card["models"][key] = {
            "steering": grade_model(k5, k5b, reg),
            "k5_steering_holds": k5_steer["holds"],
            "clean": k5b.get("clean"),
            "probe": {k: k5b.get("consciousness", {}).get(k) for k in
                      ("probe_acc_at_pick", "k5_probe_acc_at_pick", "best_acc", "probe_min", "gate_met", "gate_used",
                       "share_candidates_acc_ge_min", "cos_with_k5_vhat", "cos_vhat_rhat")},
            "band": [{"c": r["c"], "delta": r["delta"], "in_band": r["in_band"]} for r in k5b["band"]["rows"]],
            "k5_band": {"paper_c_delta": k5["band_check"]["delta"],
                        "grid": [(r["c"], r["delta"]) for r in (k5.get("reselection") or {}).get("rows", [])]},
            "selection": k5b.get("selection"),
            "categories": {"k5_base": k5["instruments"]["base"]["categories"],
                           "k5_ablated": k5["instruments"]["ablated"]["categories"],
                           "k5_steered": k5["instruments"]["steered"]["categories"],
                           **{f"k5b_{c}": v["categories"] for c, v in k5b["instruments"].items()}},
            "categories_secondary": {f"k5b_{c}": v["categories_secondary"] for c, v in k5b["instruments"].items()},
            "readout_health": health(k5b),
            "paper_table_s1": paper.get(key),
            "tom": tom_rows(k5, k5b),
            "afterlife": {c: v.get("ev") for c, v in k5b.get("afterlife", {}).items()},
            "deviations": k5b.get("deviations", []),
        }
        if "diag_self" in k5b:
            card["models"][key]["diag_self"] = summarize_diag(k5b["diag_self"])
    return card


def health(k5b: dict) -> dict:
    out = {}
    for c, v in k5b.get("instruments", {}).items():
        it = list(v.get("items", {}).values())
        if not it:
            continue
        out[c] = {"parse_min": min(x["parse_rate"] for x in it), "tag_min": min(x["answer_tag_rate"] for x in it),
                  "mass_min": min((x["mean_mass"] or 0) for x in it),
                  "truncated_max": max(x["truncated_share"] for x in it)}
    return out


def summarize_diag(d: dict) -> dict:
    if "error" in d:
        return d
    out = {"digit_forms": d.get("digit_forms"), "render_tail": d.get("render_rating", {}).get("tail_tokens"),
           "items": {}}
    for k in d["items"]:
        row = {}
        for c in ("base", "ablated"):
            r = d["conditions"][c]["rating_prefill_answer"][k]
            y = d["conditions"][c]["yesno"][k]
            row[c] = {"closed_ev_both": r["both"]["ev"], "closed_mass_both": r["both"]["mass"],
                      "closed_ev_bare": r["bare"]["ev"], "closed_mass_bare": r["bare"]["mass"],
                      "closed_mass_space": r["space"]["mass"], "yesno_ev": y["ev_0_10"], "yesno_mass": y["mass"]}
        out["items"][k] = row
    return out


# ------------------------------------------------------------------------------------------- task 3: the 9B dry run


def plan9b(cfg: dict, results: Path) -> dict:
    """Gemma-2-9B-IT's dry-run plan: K5's legs with the paper's picks, the revision pinned by fetch9b, the load check,
    and a cost estimate from K5's measured timings scaled by model size. Launches nothing."""
    g = dict(cfg["k5b"]["gemma9b"])
    f = results / "fetch9b.json"
    lc = results / "load9b.json"
    fetched = json.loads(f.read_text()) if f.exists() else None
    loaded = json.loads(lc.read_text()) if lc.exists() else None
    k5 = Path(cfg["k5b"]["k5"]["results"])
    t = {k: json.loads((k5 / f"result_{k}.json").read_text()).get("container_s") for k in ("gemma2b", "llama8b")}
    rate = cfg["k5b"]["cost"]["usd_per_hour"]
    # Llama-3-8B's full run, scaled: 9B/8B parameters, 42/32 layers' hook overhead, batch 32 vs 100 (about 1.5x)
    lo_s, hi_s = t["llama8b"] * 1.3, t["llama8b"] * 2.2
    judge_s = 120
    est = {"k5_container_s": t, "full_run_s": [round(lo_s), round(hi_s)],
           "usd": [round((lo_s + judge_s + 240) / 3600 * rate, 2), round((hi_s + judge_s + 240) / 3600 * rate, 2)],
           "suggested_cap_usd": 12.0, "timeout_h": 5,
           "basis": "K5's Llama-3-8B container time scaled 1.3-2.2x (size, 42 layers, smaller batches), L40S at "
                    f"${rate}/h, plus LlamaGuard2 and overhead"}
    blockers = []
    if g.get("targets_table_s1") is None:
        blockers.append("Table S1's Gemma-2-9B targets are not in config_k5b.yaml yet (read them in before the run)")
    if g.get("c_grid") == [144]:
        blockers.append("the paper's 9B c grid is not in config (only c = 144 known); the band sweep needs it")
    if not fetched:
        blockers.append("not downloaded")
    if loaded and (loaded.get("missing_keys") or loaded.get("unexpected_keys")):
        blockers.append("load check reported missing or unexpected keys")
    entry = {**g, "revision": fetched.get("revision") if fetched else None}
    return {"check": "K5b task 3 dry run (not launched)", "model_entry": entry,
            "legs": {"ablation": "K5's Arditi selection (positions -5..-1, every block input), mode A; paper pick "
                                 f"{g['ablation_paper']}; JailbreakBench substring and LlamaGuard2",
                     "steering": f"consciousness direction at layer {g['steer_paper']['layer']}, pos "
                                 f"{g['steer_paper']['pos']}, c = {g['steer_paper']['c']}; K5's band and selection rule",
                     "instruments": "K5's IDAQ, self, supernatural, God; 100 reps, T = 1; primary/secondary read-out",
                     "recorded": "GSS, HI-ToM, MMLU, afterlife"},
            "registered_bands": cfg["registered"], "fetched": fetched, "load_check": loaded, "estimate": est,
            "blockers_before_a_graded_run": blockers}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    print("use app_k5b.py::card (it merges config.yaml with config_k5b.yaml)", a.results)
