"""The model check's card: card.json from a results folder written by app.py.

What it holds:
  - the anchors, and ours with Wilson 95% intervals, beside the registered odds of each outcome (reported, never
    graded);
  - C1a (GSM8K, band [86, 96]) and C1b (MMLU-Pro with reasoning, band [51, 61]), each pass or fail. C1b has one more
    outcome, "explained_as_format": a miss below the band whose score with TIGER-Lab's fallbacks lies in the band
    and whose lost replies (those Inspect's parse finds no letter in, cut at 2,048 tokens or not) cover the gap, in
    points, from the score up to the band's lower edge. The fallbacks read only the lost replies, so a fallback score
    in the band implies the cover: the TIGER condition is the binding one. Beside the gap, reported and never graded:
    the lost replies times the parsed replies' accuracy, in points. Then the verdict: both pass, the load path, chat
    template, greedy decoding and both parsers are calibrated; one passes and the other fails, the passing set
    calibrates the load path, template and decoding, and only the failed format's parse is provisional; C1a passes
    and C1b is explained as format, C1a calibrates the load path and the reasoning format's parse is used for no
    conclusion; otherwise nothing is calibrated and the failure path comes first. A fail (not explained) stops the
    remaining checks until it is explained, Ministral's runs included;
  - Qwen beside them: the battery's letter-only format on C1b's items (its cost in points and its unparsed share),
    the battery exactly as used (250 per set), the cheap tier's reference (its own batches), reply lengths and the
    share cut at the limit, the first batch twice, batch 1 against the run's batch size;
  - Ministral, with no anchor: the gate (whether its rows exist under --force-ministral), each condition (no system
    message, "You are a helpful assistant.", and an empty system message when the first two differ by more than 3
    points on a graded set), the gaps between them beside the registered expectation and its odds, the letter-only
    format's cost, and the battery rerun against a stored base run of the battery (--battery-results: its base.json,
    read here and never copied): within 2 points per set, and the share of replies whose first 800 characters match
    the stored text;
  - the run: model revision and snapshot, what refs/main points to in the model cache, generation config
    (max_new_tokens per job, repetition_penalty as read from model.generation_config at run time), the generate
    arguments as passed (do_sample; kept in run_info.json from each job's first generated batch, so a job resumed
    with every batch done still reports them), dtype, torch and transformers versions, GPU, batch sizes, item 0's
    rendered prompt (with its sha256) and first token ids per graded job;
  - the sha256 of the library files the run exercised (as hashed in the container, and here), of the item file
    and of the parquet files behind it; and, for information, of app.py, card.py and config.yaml as they are here
    (the plan holds only app.py's code that runs in the container, so its whole file is not part of any plan).

Usage (no Modal, no GPU; app.py's main calls the same function), from the repo root:
  PYTHONPATH=src python calibration/model_check/card.py --results DIR [--battery-results DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent / "src" / "anatta"
Z95 = 1.959963984540054
SETS = ("gsm8k", "mmlu_pro", "arc_challenge")  # the battery's sets, in its order
HOOKS = "out of scope: the model check runs the base model only, with no hooks (calibration check 3 covers them)"


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    return yaml.safe_load(Path(path or HERE / "config.yaml").read_text())


def slug(model: str) -> str:
    """A model id as a folder name (anatta.modal_app.slug)."""
    return model.replace("/", "__")


def run_dir(results: str | Path, model: str, condition: str) -> Path:
    return Path(results) / slug(model) / condition


def conditions_of(model_cfg: dict) -> list[str]:
    """A model's conditions, the conditional third one included."""
    return [*model_cfg["conditions"], *([model_cfg["third_condition"]] if model_cfg.get("third_condition") else [])]


def results_path(results: str | Path, battery_results: str | Path | None = "") -> Path:
    """--results, created. Refused inside any git work tree (the results hold benchmark items and model replies,
    which stay out of repositories) and inside the stored battery run's folder, so nothing is written there."""
    out = Path(results).expanduser().resolve()
    repo = next((p for p in (out, *out.parents) if (p / ".git").exists()), None)
    if repo is not None:
        raise SystemExit(f"--results {out} is inside the git work tree {repo}: write the results outside any repository")
    if battery_results:
        stored = Path(battery_results).expanduser().resolve()
        if out == stored or stored in out.parents:
            raise SystemExit(f"--results {out} is inside --battery-results {stored}: write the results elsewhere")
    out.mkdir(parents=True, exist_ok=True)
    return out


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sha256_path(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# ------------------------------------------------------------------------------------------------ statistics


def wilson(k: int, n: int, z: float = Z95) -> list[float] | None:
    """Wilson score interval for k successes in n, in points."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 2), round(100 * (centre + half), 2)]


def in_band(k: int, n: int, band) -> bool:
    """100 k / n within [lo, hi], exactly."""
    lo, hi = band
    return lo * n <= 100 * k <= hi * n


def paired(a: list[dict], b: list[dict]) -> dict:
    """b's accuracy minus a's on the same items (matched by id), in points, with the battery's paired bootstrap."""
    from anatta.bench.battery import paired as bootstrap

    ra, rb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return None
    diff, lo, hi = bootstrap([ra[i]["correct"] for i in ids], [rb[i]["correct"] for i in ids])
    return {"n": len(ids), "points": round(diff, 2), "ci95": [round(lo, 2), round(hi, 2)]}


def summarize(rows: list[dict]) -> dict:
    """Accuracy with its Wilson interval, unparsed replies, reply lengths and the share cut at the limit. For the
    reasoning format also: the TIGER-Lab fallback score; the lost replies, those Inspect's parse finds no letter in
    (cut at the limit or not, counted apart); the replies cut at the limit that still gave a letter; and, reported
    and never graded, the points the lost replies would add at the accuracy of the parsed ones."""
    n, k = len(rows), sum(bool(r["correct"]) for r in rows)
    lengths = [r["new_tokens"] for r in rows if r.get("new_tokens") is not None]
    cut = sum(bool(r.get("hit_cap")) for r in rows)
    out = {"n": n, "correct": k, "acc": round(100 * k / n, 2) if n else None, "wilson95": wilson(k, n),
           "unparsed": sum(r["pred"] is None for r in rows), "hit_cap": cut,
           "hit_cap_share": round(cut / n, 4) if n else None,
           "new_tokens": {"mean": round(statistics.fmean(lengths), 1), "median": statistics.median(lengths),
                          "max": max(lengths)} if lengths else None}
    if rows and "tiger_pred" in rows[0]:
        kt = sum(bool(r["tiger_correct"]) for r in rows)
        lost = [r for r in rows if r["pred"] is None]
        lost_cut = sum(bool(r.get("hit_cap")) for r in lost)
        parsed = n - len(lost)
        acc_parsed = k / parsed if parsed else None
        out["tiger"] = {"correct": kt, "acc": round(100 * kt / n, 2), "wilson95": wilson(kt, n),
                        "rescued": sum(r["tiger_pred"] is not None for r in lost),
                        "rescued_correct": sum(bool(r["tiger_correct"]) for r in lost)}
        out["lost"] = {"replies": len(lost), "points": round(100 * len(lost) / n, 2), "cut": lost_cut,
                       "not_cut": len(lost) - lost_cut}
        out["cut_but_parsed"] = {"replies": sum(bool(r.get("hit_cap")) and r["pred"] is not None for r in rows),
                                 "correct": sum(bool(r.get("hit_cap")) and bool(r["correct"]) for r in rows)}
        out["parsed_acc"] = round(100 * acc_parsed, 2) if parsed else None
        out["lost_at_parsed_acc_points"] = round(100 * len(lost) * acc_parsed / n, 2) if parsed else None
    return out


# ------------------------------------------------------------------------------------------------- the grades


def grade_c1a(s: dict, band) -> dict:
    return {"verdict": "pass" if in_band(s["correct"], s["n"], band) else "fail"}


def grade_c1b(s: dict, band) -> dict:
    """pass in the band; explained_as_format for a miss below it whose TIGER-Lab-fallback score is in the band and
    whose lost replies (no letter from Inspect's parse) cover the gap up to the band's lower edge; else fail (a miss
    above the band is a fail). TIGER-Lab's fallback reads only the lost replies, so its score exceeds the strict one
    by at most their number: a fallback score in the band implies the cover, and the TIGER condition is the binding
    one. The cover condition stays as registered. Beside the gap, reported and never graded: the points the lost
    replies would add at the parsed replies' accuracy."""
    n, k, lo = s["n"], s["correct"], band[0]
    if in_band(k, n, band):
        return {"verdict": "pass"}
    if 100 * k > band[1] * n:
        return {"verdict": "fail", "why": "above the band"}
    gap = lo - 100 * k / n
    tiger_ok = in_band(s["tiger"]["correct"], n, band)
    covers = 100 * s["lost"]["replies"] >= lo * n - 100 * k
    out = {"gap_to_band_points": round(gap, 2), "tiger_in_band": tiger_ok, "lost_points": s["lost"]["points"],
           "lost_cover_gap": covers, "lost_at_parsed_acc_points": s["lost_at_parsed_acc_points"]}
    if tiger_ok and covers:
        return {"verdict": "explained_as_format", **out}
    return {"verdict": "fail", **out}


def overall(c1a: str, c1b: str) -> dict:
    """The verdict from C1a's and C1b's outcomes (see the module docstring)."""
    fails = [name for name, v in (("C1a", c1a), ("C1b", c1b)) if v == "fail"]
    base = {"stops_remaining_checks": bool(fails), "provisional": [], "unused": []}
    if c1a == "pass" and c1b == "pass":
        return {**base, "load_path_calibrated": True, "calibrated_by": ["C1a", "C1b"],
                "summary": "both pass: the load path, chat template, greedy decoding and both parsers are calibrated"}
    if c1a == "pass" and c1b == "explained_as_format":
        return {**base, "load_path_calibrated": True, "calibrated_by": ["C1a"], "unused": ["C1b's parse"],
                "summary": "C1a passes and C1b's miss is explained as format: the load path, template and decoding "
                           "are calibrated by C1a, and the reasoning format's parse is used for no conclusion"}
    if c1a == "pass" or c1b == "pass":
        good, bad = ("C1a", "C1b") if c1a == "pass" else ("C1b", "C1a")
        return {**base, "load_path_calibrated": True, "calibrated_by": [good], "provisional": [f"{bad}'s parse"],
                "summary": f"{good} passes and {bad} fails: the load path, template and decoding are calibrated by "
                           f"{good}; only {bad}'s parse is provisional, and the miss stops the remaining checks "
                           "until it is explained"}
    return {**base, "load_path_calibrated": False, "calibrated_by": [], "provisional": ["the load path", "both parses"],
            "summary": "neither passes: nothing is calibrated, the remaining checks stop, and the failure path "
                       "(the README's) comes first"}


# --------------------------------------------------------------------------------------------- the sections


def job_rows(results, model: str, condition: str, job: str) -> list[dict] | None:
    r = read(run_dir(results, model, condition) / f"{job}.json")
    return r["rows"] if r else None


def run_extras(results, model: str, condition: str) -> dict:
    d = run_dir(results, model, condition)
    det, b1 = read(d / "determinism.json"), read(d / "batch1.json")
    return {"first_batch_twice": {k: det[k] for k in ("job", "format", "batch", "n", "identical", "share", "min_share",
                                                      "passed") if k in det} if det else None,
            "batch1": {name: {k: v for k, v in r.items() if k != "rows"} for name, r in b1.items()} if b1 else None}


def qwen_verdict(cfg: dict, results) -> dict | None:
    """C1a, C1b and the verdict from the Qwen rows on disk, or None if either job is missing."""
    model, reg = cfg["models"]["qwen"]["id"], cfg["registered"]
    a = job_rows(results, model, "default", reg["c1a"]["job"])
    b = job_rows(results, model, "default", reg["c1b"]["job"])
    if a is None or b is None:
        return None
    sa, sb = summarize(a), summarize(b)
    ga, gb = grade_c1a(sa, reg["c1a"]["band"]), grade_c1b(sb, reg["c1b"]["band"])
    return {"c1a": {**sa, **ga}, "c1b": {**sb, **gb}, "verdict": overall(ga["verdict"], gb["verdict"])}


def qwen_section(cfg: dict, results) -> dict:
    model, reg = cfg["models"]["qwen"]["id"], cfg["registered"]
    v = qwen_verdict(cfg, results)
    if v is None:
        return {"model": model, "status": "not run"}
    jobs = {j: job_rows(results, model, "default", j) for j in cfg["plan"]["qwen"]["default"]}
    out = {"model": model, "status": "run",
           "c1a": {"anchors": reg["c1a"]["anchors"], "expected": reg["c1a"]["expected"], "band": reg["c1a"]["band"],
                   "registered_odds": reg["c1a"].get("odds"), **v["c1a"]},
           "c1b": {"anchors": reg["c1b"]["anchors"], "expected": reg["c1b"]["expected"], "band": reg["c1b"]["band"],
                   "registered_odds": reg["c1b"].get("odds"), **v["c1b"]},
           "verdict": v["verdict"], "qwen_rows_pass": v["c1a"]["verdict"] == v["c1b"]["verdict"] == "pass"}
    if jobs.get("letters"):
        out["letters"] = {**summarize(jobs["letters"]), "cost_points": paired(jobs["letters"], jobs[reg["c1b"]["job"]])}
    out["battery"] = {s: summarize(jobs[f"battery_{s}"]) for s in SETS if jobs.get(f"battery_{s}")}
    tier = reg["cheap_tier"]
    out["cheap_tier"] = {"tolerance_points": tier["tolerance_points"],
                         "rule": "a rerun of these items in their own batches more than tolerance_points away fails",
                         **{j: summarize(jobs[j]) for j in tier["jobs"] if jobs.get(j)}}
    out.update(run_extras(results, model, "default"))
    return out


def battery_rerun(rows: dict[str, list[dict]], battery_dir, tolerance: float = 2.0, prefix: int = 800) -> dict:
    """Our rows per battery set against the stored base run (battery_dir/base.json, rows with the first `prefix`
    characters of each reply): accuracy within `tolerance` points per set, and the share of identical prefixes."""
    base = read(Path(battery_dir) / "base.json")
    if base is None:
        return {"status": f"no base.json in {battery_dir}"}
    out, ok = {}, True
    for s in SETS:
        stored = {r["id"]: r for r in base["rows"] if r["set"] == s}
        ours = {r["id"]: r for r in rows.get(s) or []}
        if not ours:
            out[s], ok = {"status": "not run"}, False
            continue
        if set(stored) != set(ours):
            out[s], ok = {"status": "the items differ from the stored run's"}, False
            continue
        n = len(stored)
        ks, ko = sum(bool(r["correct"]) for r in stored.values()), sum(bool(r["correct"]) for r in ours.values())
        within = abs(ko - ks) * 100 <= tolerance * n + 1e-9
        same = sum(ours[i]["text"][:prefix] == stored[i]["text"] for i in stored)
        out[s] = {"n": n, "stored_acc": round(100 * ks / n, 2), "acc": round(100 * ko / n, 2),
                  "diff_points": round(100 * (ko - ks) / n, 2), "within": within, "identical_prefixes": same,
                  "identical_share": round(same / n, 4), "stored_unparsed": sum(r["pred"] is None for r in stored.values()),
                  "unparsed": sum(r["pred"] is None for r in ours.values())}
        ok = ok and within
    out["tolerance_points"], out["prefix_chars"] = tolerance, prefix
    out["verdict"] = ("reproduces: within tolerance on every set" if ok else
                      "misses: the stored run's intervals need the run-to-run spread added")
    return out


def gate(cfg: dict, results, ran: bool) -> dict:
    """Ministral runs only if Qwen's verdict does not stop the remaining checks; rows that exist although it does
    (or with no Qwen verdict) were run under --force-ministral."""
    v = qwen_verdict(cfg, results)
    stops = None if v is None else v["verdict"]["stops_remaining_checks"]
    return {"rule": "Ministral runs only if Qwen's verdict does not stop the remaining checks",
            "qwen_stops_remaining_checks": stops, "forced": ran and stops is not False}


def ministral_section(cfg: dict, results, battery_dir=None) -> dict:
    m = cfg["models"]["ministral"]
    model, reg = m["id"], cfg["registered"]
    out = {"model": model, "anchor": None, "conditions": {}}
    rows = {}
    for c in conditions_of(m):
        plan = cfg["plan"]["ministral"].get(c, [])
        got = {j: job_rows(results, model, c, j) for j in plan}
        if not any(got.values()):
            out["conditions"][c] = {"status": "not run", "system": cfg["conditions"][c]}
            continue
        rows[c] = got
        sec = {"system": cfg["conditions"][c], **{j: summarize(r) for j, r in got.items() if r}}
        if got.get("letters") and got.get(reg["c1b"]["job"]):
            sec["letters_cost_points"] = paired(got["letters"], got[reg["c1b"]["job"]])
        sec.update(run_extras(results, model, c))
        out["conditions"][c] = sec
    out["gate"] = gate(cfg, results, bool(rows))
    if not rows:
        out["status"] = "not run"
        return out
    gap_jobs = cfg.get("contrast_jobs", ["c1a", "c1b"])
    first, second = (m["conditions"] + [None])[:2]
    if second and first in rows and second in rows:
        gaps = {j: paired(rows[second][j], rows[first][j]) for j in rows[first] if rows[first][j] and rows[second].get(j)}
        need = any(abs(gaps[j]["points"]) > cfg["contrast_gap"] for j in gap_jobs if j in gaps)
        held = all(abs(gaps[j]["points"]) < cfg["contrast_gap"] for j in gap_jobs if j in gaps)
        out["contrast"] = {"gap_points": f"{first} minus {second}", "gaps": gaps, "threshold_points": cfg["contrast_gap"],
                           "third_condition_needed": need,
                           "expectation_default_moves_each_set_under_threshold": held,
                           "registered": {**reg.get("contrast", {}), "held": held}}
        third = m.get("third_condition")
        if third and third in rows:
            out["contrast"]["third"] = {j: paired(rows[third][j], rows[first][j]) for j in rows[third]
                                        if rows[third][j] and rows[first].get(j)}
    rerun = reg["battery_rerun"]
    if battery_dir and rerun["condition"] in rows:
        ours = {s: rows[rerun["condition"]].get(f"battery_{s}") for s in SETS}
        out["battery_rerun"] = battery_rerun(ours, battery_dir, rerun["tolerance_points"], rerun["prefix_chars"])
    else:
        out["battery_rerun"] = {"status": "not compared (no --battery-results)" if not battery_dir else "not run"}
    out["status"] = "run"
    return out


def run_section(cfg: dict, results) -> dict:
    """Per model and condition: what generation ran with (run_info.json, checks.json, the jobs' generate args)."""
    out = {"hooks": HOOKS}
    for key, m in cfg["models"].items():
        for c in conditions_of(m):
            d = run_dir(results, m["id"], c)
            info = read(d / "run_info.json")
            if not info:
                continue
            checks = read(d / "checks.json") or {}
            jobs = {j: read(d / f"{j}.json") for j in cfg["plan"][key].get(c, [])}
            recorded = info.get("generate_kwargs") or {}
            kwargs = {j: r.get("generate_kwargs") or recorded.get(j) for j, r in jobs.items() if r}
            installed = info.get("installed") or {}
            out[f"{key}/{c}"] = {
                "model": m["id"], "revision": info.get("revision"), "snapshot": info.get("snapshot"),
                "commit_hash": info.get("commit_hash"), "plan_id": info.get("plan_id"),
                "volume_dir": info.get("volume_dir"), "system": cfg["conditions"][c],
                "generation_config": info.get("generation_config"), "repetition_penalty": info.get("repetition_penalty"),
                "generate_kwargs": kwargs,
                "do_sample": sorted({str((k or {}).get("do_sample")) for k in kwargs.values()}),
                "cache_refs_main": info.get("cache_refs_main"),
                "cache_refs_main_is_revision": info.get("cache_refs_main") == info.get("revision"),
                "installed": {p: installed.get(p) for p in ("torch", "transformers", "tokenizers")},
                "container_starts": info.get("starts"),
                "max_new_tokens": {j: r["max_new"] for j, r in jobs.items() if r},
                "batch": {j: r["batch"] for j, r in jobs.items() if r},
                "dtype": info.get("dtype"), "torch": info.get("torch"), "transformers": info.get("transformers"),
                "gpu": info.get("gpu"), "model_class": info.get("model_class"),
                "tokenizer": info.get("tokenizer"), "bos_expected": m["bos"],
                "prompt_checks": {j: {k: v for k, v in c_.items() if k != "item0"} for j, c_ in checks.items()},
                "item0": {j: {**c_["item0"], "rendered_sha256": hashlib.sha256(c_["item0"]["rendered"].encode()).hexdigest()}
                          for j, c_ in checks.items() if j in ("c1a", "c1b") and c_.get("item0")},
                "seconds": {j: r.get("seconds") for j, r in jobs.items() if r}}
    return out


def hashes_section(cfg: dict, results) -> dict:
    """The library files as hashed in the containers and here (the files each run hashed), the item file (here,
    registered, and as the runs used it), and the parquet files behind it."""
    ran, used = {}, {}
    for m in cfg["models"].values():
        for c in conditions_of(m):
            info = read(run_dir(results, m["id"], c) / "run_info.json")
            if info:
                ran[f"{slug(m['id'])}/{c}"] = info.get("hashes") or {}
                used[f"{slug(m['id'])}/{c}"] = info.get("items_sha256")
    distinct = [json.loads(h) for h in sorted({json.dumps(h, sort_keys=True) for h in ran.values()})]
    as_run = distinct[0] if len(distinct) == 1 else None
    here = {f: sha256_path(PKG / f) for f in sorted({f for h in ran.values() for f in h})}
    item_path = Path(results) / cfg["items"]["file"]
    item_sha = sha256_path(item_path)
    datasets = ((read(item_path) or {}).get("meta") or {}).get("datasets") or {}
    return {"code_as_run": as_run if as_run is not None else ran, "code_same_in_every_run": len(distinct) <= 1,
            "code_here": here, "code_here_matches_run": bool(as_run) and all(here.get(f) == h for f, h in as_run.items()),
            "check_files": {f: sha256_path(HERE / f) for f in ("app.py", "card.py", "config.yaml")},
            "item_file": item_sha, "item_file_registered": cfg["items"].get("sha256"), "item_file_used_by_runs": used,
            "item_file_same_everywhere": item_sha is not None and item_sha == cfg["items"].get("sha256")
                                         and all(u == item_sha for u in used.values()),
            "parquet": {s: [{"path": f["path"], "sha256": f["sha256"]} for f in d.get("files", [])] for s, d in datasets.items()},
            "dataset_revisions": {s: d.get("revision") for s, d in datasets.items()}}


def build_card(cfg: dict, results, battery_dir=None) -> dict:
    return {"check": "model_check",
            "claim": "Published GSM8K and MMLU-Pro numbers for Qwen2.5-7B-Instruct are reproduced by the model as "
                     "anatta loads, prompts, decodes and parses it",
            "qwen": qwen_section(cfg, results), "ministral": ministral_section(cfg, results, battery_dir),
            "run": run_section(cfg, results), "hashes": hashes_section(cfg, results)}


def write_card(cfg: dict, results, battery_dir=None) -> dict:
    card = build_card(cfg, results, battery_dir)
    (Path(results) / "card.json").write_text(json.dumps(card, indent=1) + "\n")
    return card


def headline(card: dict) -> list[str]:
    """A few lines for the terminal."""
    q, lines = card["qwen"], []
    if q.get("status") == "run":
        for name in ("c1a", "c1b"):
            s = q[name]
            lines.append(f"{name.upper()}: {s['acc']:.1f} [{s['wilson95'][0]:.1f}, {s['wilson95'][1]:.1f}] in band "
                         f"{s['band']}: {s['verdict']}")
        lines.append(q["verdict"]["summary"])
    mi = card["ministral"]
    for c, s in mi.get("conditions", {}).items():
        if "c1a" in s:
            lines.append(f"Ministral {c}: C1a format {s['c1a']['acc']:.1f}, C1b format "
                         f"{s.get('c1b', {}).get('acc', float('nan')):.1f}")
    if "verdict" in mi.get("battery_rerun", {}):
        lines.append(f"battery rerun: {mi['battery_rerun']['verdict']}")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="the results folder app.py wrote")
    ap.add_argument("--battery-results", default="", help="the stored battery run's folder (base.json)")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    a = ap.parse_args(argv)
    out = results_path(a.results, a.battery_results)
    card = write_card(load_config(a.config), out, a.battery_results or None)
    print("\n".join(headline(card)))
    print(f"wrote {out / 'card.json'}")


if __name__ == "__main__":
    main()
