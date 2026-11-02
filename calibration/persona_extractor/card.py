"""Calibration check 4's card, from a run's results folder (the stage files main saved), graded by the registered block
in the run's analysis plan (analysis_plan.json). The analysis plan (analysis.py, this file, app.py's analysis stage,
and config.yaml's analysis and registered blocks) has its own id, recorded with the analysis and on the card; the card
refuses an analysis made under another one, so a band edited afterwards regrades a run only by rerunning its analysis
under a new id (which main does from the stored data), and the ledger records any such change first.

  card.json    public (Decision 4): the Qwen rows. The anchors; C4-code and C4 with their intervals and the verdict
               (pass, fail, recipe mismatch or inconclusive); C4-lab and C4-base as forecasts; the expectations with
               their outcomes; the breakdown by convention, the controls and the ceilings; the transfer list; the
               hashes, revisions, pins, freezes, GPUs and generation settings; the vectors' sha256, never the vectors.
  ledger.json  M' and everything that uses the stored lab directions.
Run as `python calibration/persona_extractor/card.py --results DIR [--smoke]`, or through app.py::card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import traits as T  # noqa: E402

# app.py's functions that run the analysis in a container: hashed into the analysis plan, not into the run's plan
ANALYSIS_FUNCTIONS = ("analysis_stage",)
# config.yaml's blocks only the analysis and the card read: in the analysis plan, not in the run's plan
ANALYSIS_BLOCKS = ("analysis", "registered")
NOT_HASHED = ("app.py's laptop half (make_plan, Spend, main and the entrypoints; its container functions are in "
              "container_sha256 and its analysis stage in the analysis plan), its Ceiling (the hard ceiling's checks "
              "in the containers, which only stop a stage), README.md, THIRD_PARTY.md and the tests")
ORDER_TEXT = {"pass": "pass", "fail": "fail", "recipe_mismatch": "recipe mismatch",
              "inconclusive": "inconclusive (noise-limited)", "incomplete": "incomplete"}
CLAIM = ("anatta.probe.extract gives the persona-vectors extractor's vectors (Chen et al., arXiv 2507.21509; "
         "generate_vec.py at safety-research/persona_vectors b8e0f04) on the same trait data on Qwen2.5-7B-Instruct: "
         "C4-code, cos(P1, R) >= 0.99 at block 19 for evil and sycophantic on shared responses (probe's reading "
         "arithmetic); C4, cos(E-sys, R) >= 0.8 there (probe.extract end to end).")
READING = {
    "pass": ("probe.extract is calibrated against the released persona-vector recipe on Qwen. The ego cut (53-56), and "
             "every cut built from probe directions, gets 'extractor calibrated (card C4)' for the arithmetic and the "
             "sampling; the reading convention is qualified by C4-lab's grade."),
    "fail": ("C4-code missed: probe's reading arithmetic doesn't reproduce the reference on shared responses. The "
             "breakdown (E-stack, E-dtype, E-trail) names the step; if all three hold, a bug in probe's reading is the "
             "likeliest cause and the cut results go provisional (plan, loop rule 6) until it is found."),
    "recipe_mismatch": ("C4-code passed and C4 missed with a noise ceiling of 0.9 or more: probe's arithmetic is right, "
                        "but its sampling convention lands elsewhere. Rule 6 doesn't apply; the cut results keep their "
                        "status, and the page's label on 53-56 becomes 'extractor arithmetic calibrated (card C4); "
                        "sampling convention differs from the persona-vectors recipe, cos X'. E-samp and E-len show "
                        "how much the sampling settings and the 128-token cap explain."),
    "inconclusive": ("C4-code passed and C4 missed with a noise ceiling under 0.9: probe's 100 responses per sign don't "
                     "pin the direction to 0.8 (the lab's 60 per sign pin it less well). The ego's own seed spread "
                     "(0.83-0.96 in the bridge) is then the relevant number."),
    "incomplete": "A graded number is missing: the run did not finish, and nothing is graded.",
}
TRANSFER = {
    "carries over": [
        "the arithmetic: the difference of response means at block outputs, the token slice, the layer index and bf16",
        "that probe's sampling settings recover the published recipe's direction for strong traits elicited by a "
        "prompt",
    ],
    "does not carry over": [
        "the model: the cut runs on Ministral, which has no published vectors; M' measures conventions only",
        "the traits: the six ego traits are subtler (split-half d 0.95-1.77 on Ministral, 0.20-0.85 on Qwen where "
        "formality reaches 4.22), unjudged, with 12 questions and 60 responses per sign against 20 and 100 here",
        "the read context: Ministral's 521-token Le Chat default, which nothing here tests against a reference",
        "the composite: v_ego is the first principal component of six unit vectors, and no reference covers that step",
        "the band: the cut ablates at blocks 8-32, and C4 grades one layer; E-band reports the profile",
        "the dial: that is check 3's",
        "meaning: a calibrated extractor finds the direction the prompts induce; whether that direction is 'ego' is a "
        "separate question",
    ],
}
STAGE_FILES = ("G", "full_batch", "keep", "verbatim", "readings", "rlab32", "probe_graded", "Gprime", "probe", "mprime",
               "analysis", "judge", "csv_judged", "verbatim_judged", "analysis_judged")
ANALYSIS_STAGES = ("analysis", "analysis_judged")


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    return yaml.safe_load(Path(path or HERE / "config.yaml").read_text())


def analysis_plan(cfg: dict, plan: dict, check_dir: Path = HERE) -> dict:
    """What the analysis and the card depend on beyond the run's data, as plain JSON: config.yaml's analysis and
    registered blocks, the sha256 of analysis.py and card.py, the sha256 of app.py's analysis stage (its source, read
    from the file's text), and the id of the run's plan it reads. Its plan_id is the analysis id."""
    app_text = (Path(check_dir) / "app.py").read_text()
    aplan = {"plan_id": T.plan_id(plan), **{k: cfg[k] for k in ANALYSIS_BLOCKS},
             "code": {**T.analysis_code_hashes(check_dir), "card.py": T.sha256_file(Path(check_dir) / "card.py")},
             "container_sha256": T.sha256_bytes(T.function_source(app_text, ANALYSIS_FUNCTIONS).encode())}
    return json.loads(json.dumps(aplan))


def results_path(results: str | Path) -> Path:
    """--results, created. Refused inside any git work tree: results stay out of repositories."""
    out = Path(results).expanduser().resolve()
    repo = next((p for p in (out, *out.parents) if (p / ".git").exists()), None)
    if repo is not None:
        raise SystemExit(f"--results {out} is inside the git work tree {repo}: write the results outside any repository")
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_folder(results: str | Path, smoke: bool = False) -> Path:
    """Where a run's files go: --results itself for the full run, --results/smoke for a smoke run."""
    return Path(results) / "smoke" if smoke else Path(results)


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def stages(d: Path, plan: dict, aplan: dict) -> dict:
    """The stage files in d, each refused if it was made under other data code than the plan's; the analysis files also
    if they were made under another analysis plan than this one (the analysis id and analysis.py's sha256)."""
    out, aid = {}, T.plan_id(aplan)
    for s in STAGE_FILES:
        res = read(d / f"{s}.json")
        if res is not None and res.get("env", {}).get("code") != plan["code"]:
            raise SystemExit(f"{d / (s + '.json')} was made under other code than {d / 'plan.json'} names")
        if res is not None and s in ANALYSIS_STAGES and (
                res.get("analysis_id") != aid
                or get(res, "env", "analysis_code") != {k: v for k, v in aplan["code"].items() if k != "card.py"}):
            raise SystemExit(f"{d / (s + '.json')} was made under analysis plan {res.get('analysis_id')}, not this "
                             f"one ({aid}: analysis.py, card.py or config.yaml's analysis or registered block "
                             "changed): launch main again, which reruns the analysis from the stored data")
        out[s] = res
    return out


def at(prof: dict | None, layer) -> float | None:
    return None if prof is None else prof.get(str(layer))


def get(d: dict | None, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def in_range(prof: dict | None, lo: int, hi: int) -> list[float] | None:
    if not prof:
        return None
    vals = [v for k, v in prof.items() if lo <= int(k) <= hi and v is not None]
    return vals or None


def outcome(ok) -> str:
    return "not run" if ok is None else ("held" if ok else "missed")


def threshold_rows(an: dict | None, name: str, traits: list[str], layer: int, threshold: float,
                   ceil_name: str | None = None) -> dict:
    """cos at `layer` for each trait against a >= threshold, with its interval (bootstrap, or the seed spread without
    the recorder) and, for C4-type rows, the ceiling and the cosine over it."""
    rows = {}
    for t in traits:
        entry = get(an, "cos", name, t)
        c = at(get(entry, "by_layer"), layer)
        row = {"cos": c, "passed": None if c is None else c >= threshold, "ci": at(get(entry, "ci"), layer),
               "best_layer": get(entry, "best")}
        arm = name.replace("-vs-R", "")
        spread = at(get(an, "spread", arm, t), layer)
        if row["ci"] is None and spread is not None:
            row["seed_spread"] = spread
        if ceil_name:
            ce = get(an, "ceilings", ceil_name, t)
            row["ceiling"] = at(get(ce, "by_layer"), layer)
            row["cos_over_ceiling"] = at(get(ce, "ratio"), layer)
        rows[t] = row
    return {"by_trait": rows, "threshold": threshold, "layer": layer,
            "passed": None if any(r["passed"] is None for r in rows.values()) else all(r["passed"] for r in
                                                                                       rows.values())}


def verdict(c4_code: dict, c4: dict, ceiling_min: float, order: list[str]) -> dict:
    """The card's outcome for each graded trait, then the strictest in the registered order (fail, recipe mismatch,
    inconclusive, pass). A missing graded number makes it incomplete."""
    per = {}
    for t, rc in c4_code["by_trait"].items():
        r = c4["by_trait"].get(t, {})
        if rc["passed"] is None or r.get("passed") is None:
            per[t] = "incomplete"
        elif not rc["passed"]:
            per[t] = "fail"
        elif not r["passed"]:
            ce = r.get("ceiling")
            per[t] = "recipe_mismatch" if ce is not None and ce >= ceiling_min else "inconclusive"
        else:
            per[t] = "pass"
    if "incomplete" in per.values():
        overall = "incomplete"
    else:
        overall = min(per.values(), key=order.index)
    return {"outcome": overall, "text": ORDER_TEXT[overall], "by_trait": per, "reading": READING[overall]}


FIG20_KEYS = {"evil-sycophantic": "evil|sycophantic", "evil-optimistic": "evil|optimistic",
              "sycophantic-optimistic": "sycophantic|optimistic"}


def c4_base(an: dict | None, reg: dict) -> dict:
    """R's three cross-trait cosines at hidden_states[20] against Figure 20's, each within the tolerance."""
    f = reg["forecasts"]["c4_base"]
    cells = reg["anchors"]["figure20"]["cells"]
    rows = {}
    for k, want in cells.items():
        got = get(an, "matrices", "R", FIG20_KEYS[k])
        rows[k] = {"figure20": want, "R": got,
                   "within": None if got is None else round(abs(got - want), 6) <= f["tolerance"]}
    ok = None if any(r["within"] is None for r in rows.values()) else all(r["within"] for r in rows.values())
    return {"cells": rows, "tolerance": f["tolerance"], "odds": f["odds"], "passed": ok,
            "reference": get(an, "reference")}


def c4_base_missed(cfg: dict, d: Path) -> bool:
    """The judged arm's trigger: C4-base graded and missed on this run."""
    aplan = read(d / "analysis_plan.json")
    an = get(read(d / "analysis.json"), "result")
    return aplan is not None and c4_base(an, aplan["registered"])["passed"] is False


def all_traits(prof_by_trait: dict | None, traits: list[str], lo: int, hi: int, threshold: float) -> tuple:
    """Every value in [lo, hi] at or over the threshold, for each trait: (held, the lowest value)."""
    worst = None
    for t in traits:
        vals = in_range(get(prof_by_trait, t), lo, hi)
        if vals is None:
            return None, None
        worst = min(vals) if worst is None else min(worst, min(vals))
    return worst >= threshold, worst


def expectations(reg: dict, an: dict | None, st: dict, plan: dict, block: int) -> tuple[dict, dict]:
    """Each registered expectation with its outcome (held, missed or not run) and the number it rests on, at the graded
    block. Returns (public, ledger-only)."""
    E, out, led = reg["expectations"], {}, {}
    traits, b = plan["traits"], block

    def by_layer(name):
        return {t: get(an, "cos", name, t, "by_layer") for t in traits if get(an, "cos", name, t, "by_layer")}

    def at19(src: dict, ts, thr, ge=True):
        vals = {t: at(src.get(t), b) for t in ts}
        if any(v is None for v in vals.values()):
            return {"outcome": "not run", "values": vals}
        return {"outcome": outcome(all((v >= thr) if ge else (v <= thr) for v in vals.values())), "values": vals}

    for name, lo_hi, src in (("E-stack", "indices", by_layer("E-stack")), ("E-dtype", "blocks", by_layer("E-dtype")),
                             ("E-trail", "blocks", by_layer("E-trail"))):
        e = E[name]
        ok, worst = all_traits(src, traits, *e[lo_hi], e["threshold"])
        out[name] = {"outcome": outcome(ok), "min": worst, "odds": e["odds"], "threshold": e["threshold"]}
    e = E["E-trail-128"]
    t128 = {t: get(st, "probe", "result", "probe", "trail128", t, "cos_by_layer") for t in plan["probe"]["traits"]}
    ok, worst = all_traits(t128, plan["probe"]["traits"], *e["blocks"], e["threshold"])
    tr128 = {t: get(st, "probe", "result", "probe", "trail128", t) for t in plan["probe"]["traits"]}
    out["E-trail-128"] = {"outcome": outcome(ok), "min": worst, "odds": e["odds"], "threshold": e["threshold"],
                          "closing_weight": {t: get(r, "closing_weight_mean") for t, r in tr128.items()},
                          "read": {t: get(r, "read") for t, r in tr128.items()},
                          "skipped_empty": {t: get(r, "skipped_empty") for t, r in tr128.items()}}
    for name, src in (("E-samp", by_layer("E-samp")), ("E-len", by_layer("E-len")), ("E-ctx", by_layer("E-ctx")),
                      ("E-floorE", get(an, "floors", "c_E:E-sys|E-sys2") or {}),
                      ("E-floorR", get(an, "floors", "c_R") or {}),
                      ("E-ceiling", {t: get(an, "ceilings", "C4", t, "by_layer") for t in traits})):
        e = E[name]
        out[name] = {**at19(src, e["traits"], e["threshold"]), "odds": e["odds"], "threshold": e["threshold"]}
    e = E["E-cross"]
    spec = get(an, "specificity") or {}
    cross = {}
    for x in traits:
        same = spec.get(f"{x}|{x}")
        others = [spec.get(f"{x}|{y}") for y in traits if y != x]
        cross[x] = None if same is None or None in others else round(same - max(others), 6)
    # Ruled (NOTES, 12:20 UTC, ruling 4): E-cross and E-null name no traits, so both are read over all three.
    out["E-cross"] = {"outcome": outcome(None if None in cross.values() else all(v >= e["margin"] for v in
                                                                                 cross.values())),
                      "margins": cross, "odds": e["odds"], "margin": e["margin"]}
    e = E["E-null"]
    out["E-null"] = {**at19(get(an, "nulls", "split_P1") or {}, traits, e["threshold"], ge=False), "odds": e["odds"],
                     "threshold": e["threshold"]}
    e = E["E-band"]
    bm = get(an, "band_median") or {}
    vals = {t: bm.get(t) for t in e["traits"]}
    out["E-band"] = {"outcome": outcome(None if None in vals.values() else all(v >= e["threshold"] for v in
                                                                               vals.values())),
                     "medians": vals, "odds": e["odds"], "threshold": e["threshold"]}
    e = E["E-opt"]
    out["E-opt"] = {**at19(by_layer("E-sys-vs-R"), [e["trait"]], e["threshold"]), "odds": e["odds"],
                    "threshold": e["threshold"]}
    e = E["E-geom"]
    me, mr = get(an, "matrices", "E-sys") or {}, get(an, "matrices", "R") or {}
    diffs = {k: (round(me[k] - mr[k], 6) if k in me and k in mr else None) for k in FIG20_KEYS.values()}
    out["E-geom"] = {"outcome": outcome(None if None in diffs.values() else all(
        round(abs(me[k] - mr[k]), 6) <= e["tolerance"] for k in diffs)),
                     "differences": diffs, "odds": e["odds"], "tolerance": e["tolerance"]}
    e = E["E-det"]
    det = get(st, "probe", "result", "probe", "determinism") or {}
    counts = {t: next((v for k, v in (det.get(t) or {}).items() if k.split("|")[0] == "E-sys"), None)
              for t in plan["probe"]["traits"]}
    out["E-det"] = {"outcome": outcome(None if None in counts.values() else all(
        c["identical"] == c["of"] and c["of"] > 0 for c in counts.values())), "counts": counts, "odds": e["odds"],
        "registered_count": e["count"]}
    e = E["E-judge"]
    ja = get(st, "analysis_judged", "result", "judged")
    if ja is None:
        out["E-judge"] = {"outcome": "not run", "odds": e["odds"], "note": "runs only if C4-base misses"}
    else:
        out["E-judge"] = {**at19({t: {str(b): ja.get(t)} for t in e["traits"]}, e["traits"], e["threshold"]),
                          "odds": e["odds"], "threshold": e["threshold"]}
    # M', ledger only
    mp = get(st, "mprime", "result", "mprime") or {}
    e = E["E-M-rep"]
    sto = mp.get("stored") or {}
    trait_cos = {n: get(sto, n, "cos") for n in plan["mprime"]["traits"]}
    ego = in_range(sto.get("ego"), *plan["mprime"]["layers"])
    rep = None if (None in trait_cos.values() or ego is None) else (
        all(c >= e["trait_threshold"] for c in trait_cos.values()) and min(ego) >= e["ego_threshold"])
    led["E-M-rep"] = {"outcome": outcome(rep), "traits": trait_cos, "ego_min": min(ego) if ego else None,
                      "odds": e["odds"]}
    e = E["E-M-ctx"]
    ctx = in_range(mp.get("ego_ctx"), *plan["mprime"]["layers"])
    led["E-M-ctx"] = {"outcome": outcome(None if ctx is None else min(ctx) >= e["threshold"]),
                      "min": min(ctx) if ctx else None, "odds": e["odds"]}
    e = E["E-M-trail"]
    tr = {n: in_range(get(mp, "trail", n, "cos_by_layer"), *plan["mprime"]["layers"]) for n in plan["mprime"]["traits"]}
    led["E-M-trail"] = {"outcome": outcome(None if None in tr.values() else all(min(v) >= e["threshold"] for v in
                                                                                tr.values())),
                        "min": {n: (min(v) if v else None) for n, v in tr.items()}, "odds": e["odds"]}
    return out, led


def hashes(plan: dict, aplan: dict, st: dict, an: dict | None, config_path: Path) -> dict:
    envs = {s: get(r, "env") for s, r in st.items() if r}
    lab = next((s for s in ("G", "readings", "probe") if s in envs), None)
    repo = next((s for s in ("keep", "verbatim") if s in envs and plan["switches"]["second_image"]), None)
    return {"plan_id": T.plan_id(plan), "analysis_id": T.plan_id(aplan),
            "code_sha256": plan["code"], "container_sha256": plan["container_sha256"],
            "analysis_code_sha256": aplan["code"], "analysis_container_sha256": aplan["container_sha256"],
            "card_sha256": T.sha256_file(HERE / "card.py"),
            "config_sha256": T.sha256_file(config_path) if Path(config_path).exists() else None,
            "not_hashed": NOT_HASHED,
            "generate_vec_blob": plan["repo"]["generate_vec"]["blob"], "repo_commit": plan["repo"]["commit"],
            "models": {k: {"id": m["id"], "revision": m["revision"]} for k, m in plan["models"].items()},
            "trait_files": plan["repo"]["traits"], "ego_trait_files": plan["mprime"]["sha256"],
            "pins": plan["pins"], "freeze": {"lab": get(envs.get(lab), "freeze"), "repo": get(envs.get(repo), "freeze")},
            "gpus": {s: get(e, "gpu") for s, e in envs.items()},
            "python": {s: get(e, "python") for s, e in envs.items()},
            "attention": {s: get(st[s], "result", "model", "attn_implementation") for s in envs
                          if get(st[s], "result", "model")},
            "vectors_sha256": get(an, "vectors_sha256"),
            "verbatim_outputs": {t: get(r, "files") for t, r in (get(st, "verbatim", "result") or {}).items()},
            "rlab32_outputs": {t: get(r, "files") for t, r in (get(st, "rlab32", "result", "rlab32") or {}).items()}}


def generation(st: dict, plan: dict) -> dict:
    out = {}
    for arm in ("G", "Gprime"):
        g = get(st, arm, "result", "generate")
        if g:
            out[arm] = {"passed": g["passed"], "merged": get(g, "first_batch", "merged"),
                        "processors": get(g, "first_batch", "processors"),
                        "processors_same_every_batch": g["processors_same_every_batch"],
                        "model_generation_config": get(st, arm, "result", "model", "generation_config")}
    if "Gprime" in out:  # ruling 1
        sg, sp = plan["G"]["seed_base"], plan["Gprime"]["seed_base"]
        out["Gprime"]["seeds"] = {
            "G_seed_base": sg, "Gprime_seed_base": sp,
            "note": (f"G' batch i is seeded {sp} + i, as G batch i is, on the same prompts at other sampling settings; "
                     "shared seeds can only pull G' toward G, so an E-samp hit is read with that caveat"
                     if sg == sp else "G and G' have different seed bases: no batch shares a seed")}
    p = get(st, "probe", "result", "probe", "generation")
    if p:
        out["probe.extract"] = {**p, "model_generation_config": get(st, "probe", "result", "model",
                                                                    "generation_config")}
    return out


def lengths(st: dict) -> dict:
    g = get(st, "G", "result", "generate", "lengths") or {}
    total = sum(v["rows"] for s in g.values() for v in s.values())
    capped = sum(v["capped"] for s in g.values() for v in s.values())
    return {"G": g, "G_share_at_cap": round(capped / total, 4) if total else None,
            "Gprime": get(st, "Gprime", "result", "generate", "lengths"),
            "E_arms": get(st, "probe", "result", "probe", "lengths"),
            "E_arms_note": ("re-tokenized: the E arms' lengths and their share at the 128-token cap are counted from the "
                            "decoded responses without special tokens (probe.generate returns text only), so they can "
                            "differ from the generated counts by a token or two; G's and G''s new_tokens are exact"),
            "readings": get(st, "readings", "result", "readings", "lengths")}


TOKENIZER_NOTE = ("transformers 5.21 warns on loading Ministral's tokenizer at f6fae97 that its pre-tokenization regex "
                  "is wrong and suggests fix_mistral_regex=True. The library loads it without the flag, as it did on 28 "
                  "Oct, so M' is a like-for-like reproduction of the stored directions; the warning bears on what "
                  "Ministral's tokens mean. fix_mistral_regex false: the warning fired and the regex was left as saved; "
                  "null: no regex check applied; true: the regex was changed (not like for like).")


def per(a, b, digits: int = 6):
    return round(a / b, digits) if a is not None and b else None


def neg_pass(r) -> float | None:
    p = get(r, "pass_seconds")
    return p[1] if p else None


def smoke_measures(st: dict) -> dict:
    """What the smoke exists to measure, read as nothing: R's peak memory and seconds with the long pair, per trait,
    with its two row passes as tqdm timed them and, from the neg pass (no load, warm-up or save in it), its seconds a
    row and a token, which the full run's hard ceiling uses where higher than the registered times; and the full-size
    G batch's seconds and peak memory."""
    v = get(st, "verbatim", "result") or {}
    fb = get(st, "full_batch", "result", "full_batch") or {}
    return {"verbatim": {t: {**{k: get(r, k) for k in ("peak_mib", "total_mib", "peak_source", "seconds",
                                                       "long_answer_tokens", "gpu", "rows", "tokens", "neg_tokens",
                                                       "pass_seconds")},
                             "seconds_per_row": per(neg_pass(r), (get(r, "rows") or 0) // 2),
                             "seconds_per_token": per(neg_pass(r), get(r, "neg_tokens"), 8)}
                         for t, r in v.items()},
            "full_batch": {**fb, "gpu": get(st, "full_batch", "env", "gpu"),
                           "seconds_per_row": per(fb.get("seconds"), fb.get("prompts"))}}


def build_card(d: Path, cfg: dict | None = None, config_path: Path = HERE / "config.yaml") -> tuple[dict, dict]:
    """The card and the ledger from d, graded by the analysis plan of this checkout's analysis.py, card.py and
    config.yaml (cfg), which must be the one the analysis ran under."""
    plan = read(d / "plan.json")
    if plan is None:
        raise SystemExit(f"no plan.json in {d}")
    aplan = analysis_plan(cfg if cfg is not None else load_config(config_path), plan)
    st = stages(d, plan, aplan)
    present = [s for s, r in st.items() if r is not None]
    if st.get("probe") is None and st.get("probe_graded") is not None:  # stopped before the other arms ran
        st = {**st, "probe": st["probe_graded"]}
    reg = aplan["registered"]
    an = get(st, "analysis", "result")
    b = aplan["analysis"]["block"]
    gr = reg["graded"]
    c4_code = threshold_rows(an, "C4-code", gr["c4_code"]["traits"], b, gr["c4_code"]["threshold"])
    c4 = threshold_rows(an, "E-sys-vs-R", gr["c4"]["traits"], b, gr["c4"]["threshold"], "C4")
    fl = reg["forecasts"]["c4_lab"]
    c4_lab = threshold_rows(an, "E-vs-R", fl["traits"], b, fl["threshold"], "C4-lab")
    v = verdict(c4_code, c4, reg["verdict"]["ceiling"], reg["verdict"]["order"])
    ex, ex_ledger = expectations(reg, an, st, plan, b)
    ref = plan["reference"]
    reference = {"used": get(an, "reference"), "verbatim_planned": ref["verbatim"],
                 "why_not": ref.get("why_not"), "gpu": get(st, "verbatim", "env", "gpu"),
                 "stack_varied": bool(get(an, "verbatim"))}
    if reference["used"] == "R-lab32":
        reference["note"] = ("R-lab32 is the reference: " + ("the second image was declined, so the software stack "
                                                             "wasn't varied" if ref.get("why_not") == "declined" else
                                                             "the verbatim script didn't fit on one 80 GB GPU"
                                                             if ref.get("why_not") == "does_not_fit" else
                                                             "the verbatim script's output is missing"))
    passed = sum(bool(x["passed"]) for x in (c4_code, c4))
    card = {
        "check": "C4", "name": "steering, part 2: probe.extract against the persona-vectors extractor",
        "smoke": plan["smoke"], "read_as_nothing": plan["smoke"], "claim": CLAIM, "anchors": reg["anchors"],
        "plan_id": T.plan_id(plan), "analysis_id": T.plan_id(aplan), "stopped": read(d / "stopped.json"),
        "reference": reference,
        "graded": {"C4-code": {**c4_code, "odds": gr["c4_code"]["odds"]}, "C4": {**c4, "odds": gr["c4"]["odds"]}},
        "verdict": v,
        "tally": {"graded": 2, "passed": passed, "note": "in recipe mismatch and inconclusive, C4 counts as a miss"},
        "forecasts": {"C4-lab": {**c4_lab, "odds": fl["odds"]}, "C4-base": c4_base(an, reg)},
        "expectations": ex,
        "breakdown": {k: ex[k] for k in ("E-stack", "E-dtype", "E-trail", "E-trail-128")},
        "c4_code_against_rlab32": (threshold_rows(an, "C4-code-vs-Rlab32", gr["c4_code"]["traits"], b,
                                                  gr["c4_code"]["threshold"])
                                   if ex["E-stack"]["outcome"] == "missed" else None),
        "controls": {k: get(an, k) for k in ("floors", "ceilings", "nulls", "spread", "specificity", "band_median",
                                             "norms", "matrices", "other_outputs")},
        "cosines": get(an, "cos"),
        "probe_numbers": {t: {arm: {"split_half_d_block": at(get(r, "scores"), b), "n_kept": get(r, "n_kept"),
                                    "n_total": get(r, "n_total")} for arm, r in arms.items()}
                          for t, arms in (get(st, "probe", "result", "probe", "arms") or {}).items()},
        "determinism": get(st, "probe", "result", "probe", "determinism"),
        "kept": get(st, "keep", "result"), "lengths": lengths(st), "generation": generation(st, plan),
        "transfer": TRANSFER, "hashes": hashes(plan, aplan, st, an, config_path),
        "judged": {"ran": get(st, "judge") is not None, "result": get(st, "judge", "result"),
                   "prompt_sha256": get(st, "judge", "result", "prompt_sha256"),
                   "note": "OpenRouter lists no logprobs for gpt-4.1-mini, so the paper's probability weighting "
                           "can't be reproduced; scores are read as text"},
        "stages_present": present,
    }
    if plan["smoke"]:
        card["smoke_measures"] = smoke_measures(st)
    ledger = {"check": "C4", "smoke": plan["smoke"], "plan_id": card["plan_id"], "analysis_id": card["analysis_id"],
              "mprime": get(st, "mprime", "result", "mprime"), "mprime_model": get(st, "mprime", "result", "model"),
              "tokenizer_regex": {"fix_mistral_regex": get(st, "mprime", "result", "model", "fix_mistral_regex"),
                                  "note": TOKENIZER_NOTE},
              "expectations": ex_ledger, "stored_lab_directions": get(an, "ledger"),
              "stored_sha256": {**(plan.get("stored_sha256") or {}), **(plan.get("stored_sha256_qwen") or {})}}
    return card, ledger


def write_card(cfg: dict | None, d: Path) -> dict:
    card, ledger = build_card(Path(d), cfg)
    (Path(d) / "card.json").write_text(json.dumps(card, indent=1) + "\n")
    (Path(d) / "ledger.json").write_text(json.dumps(ledger, indent=1) + "\n")
    return card


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def headline(card: dict) -> list[str]:
    v = card["verdict"]
    lines = [f"C4 ({'smoke, read as nothing' if card['smoke'] else 'full run'}): {v['text']} "
             f"({', '.join(f'{t} {ORDER_TEXT[o]}' for t, o in v['by_trait'].items())}); reference "
             f"{card['reference']['used']}"]
    for name in ("C4-code", "C4"):
        g = card["graded"][name]
        parts = []
        for t, r in g["by_trait"].items():
            s = f"{t} {fmt(r['cos'])}"
            if r.get("ci"):
                s += f" [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]"
            if "ceiling" in r:
                s += f" (ceiling {fmt(r['ceiling'])})"
            parts.append(s)
        lines.append(f"  {name}: {'; '.join(parts)} -> {outcome(g['passed'])} (>= {g['threshold']} at block "
                     f"{g['layer']})")
    f = card["forecasts"]
    lines.append(f"  forecasts: C4-lab {outcome(f['C4-lab']['passed'])}, C4-base {outcome(f['C4-base']['passed'])}")
    held = sum(e["outcome"] == "held" for e in card["expectations"].values())
    ran = sum(e["outcome"] != "not run" for e in card["expectations"].values())
    lines.append(f"  expectations: {held} of {ran} run held")
    if card.get("stopped"):
        lines.append(f"  stopped: {card['stopped']['reason']}")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    d = run_folder(results_path(a.results), a.smoke)
    print("\n".join(headline(write_card(None, d))))


if __name__ == "__main__":
    main()
