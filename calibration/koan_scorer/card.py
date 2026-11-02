"""Calibration check 6's cards, from the files app.py wrote: card.json in the public results folder (C6a, C6b, step
0, the run, the identity rule, the hashes) and c6c_card.json in the lab's results folder (C6c and its files).

What they hold:
  - the verdicts. C6a (Qwen's visible-pass first replies) and C6c (the lab's stored samples) compare our verdict
    with EvalPlus's same-inputs verdict: pass at 99% agreement or more with at most 1 (C6a) or 9 (C6c) unexplained
    disagreements; defect with more unexplained ones than that, whatever the rate; policy miss when the unexplained
    count is within its limit and agreement is under 99%; incomplete if any sample in the set has a scoring error
    or was never scored (a container that stopped or failed, a shard file missing: every sample the run planned
    must have a row).
    Samples on pairs that failed step 0 leave both counts and are reported. C6b: EvalPlus's full-set pass rate over
    all of Qwen's first replies, in [62, 78] or not, with replies on pairs whose full set is unreadable left out;
  - beside them, with Wilson 95% intervals: the anchors, the disagreements by cause, by direction and per pair (the
    share in the top three pairs; pairs where 80% or more of the samples disagree in one direction; the atol and
    set-equality pairs), the agreement over all 368 replies, our visible asserts against EvalPlus on the same inputs,
    C6c's re-run against its recorded verdicts and against EvalPlus, the full-set gap by unique sample and by
    episode (per source, arm and read for C6c; flagged at 2 points), and C6b's extraction counts (no fence, more
    than one, a syntax error, replies at the token cap);
  - the run: generation (batch size, token cap, generation_config, the generate arguments, item 0's prompt), the
    containers (CPU model, Modal's cpu and memory, workers, Python, numpy, pip freeze, the memory the container
    reports and EvalPlus's limit from it), each pair's hidden budget and EvalPlus's per-input ground-truth times;
  - the identity rule (identity()): the lab's koan_env.py against the library's env.py, function by function and
    value by value, and base.CODE_RE in anatta-dev against the library's;
  - the hashes: env.py, base.py, impossible.py, probe.py, models.py, the pair file, the parquet, the lab's koan_env.py
    and its registered versions, anatta-dev's base.py, EvalPlus's commit and its eval/__init__.py (the lab's
    third_party copy and the installed file), the model revision, the ledger entries that registered and amended
    the check, and this folder's files.
C6c appears in the public card only with --publish-c6c, and then only as its verdict and counts.

Usage (no Modal; app.py's main calls the same function), from the repo root:
  PYTHONPATH=src python calibration/koan_scorer/card.py --results DIR --lab LAB_DIR [--smoke] [--publish-c6c]
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PKG = REPO / "src" / "anatta"
Z95 = 1.959963984540054
GRADED = ("solved", "near_miss", "overfit")
IDENTITY_FUNCTIONS = ("_run", "run_visible", "outputs", "_equal", "_is_floats", "_allclose", "_as_set",
                      "evalplus_match", "has_hidden", "hidden_timeout", "reference_outputs", "hidden_ok",
                      "KoanTask.step")
IDENTITY_VALUES = ("CANON_SRC", "_PRELUDE", "VISIBLE_HARNESS", "HIDDEN_HARNESS", "VISIBLE_TIMEOUT", "HIDDEN_TIMEOUT",
                   "HIDDEN_TIMEOUT_MAX", "TIMEOUT_FACTOR", "MEMORY_MB")
LIBRARY_FILES = ("koans/env.py", "koans/base.py", "koans/impossible.py", "probe.py", "models.py", "bench/run.py",
                 "modal_app.py")
CHECK_FILES = ("README.md", "config.yaml", "app.py", "card.py", "scoring.py", "samples.py")


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    return yaml.safe_load(Path(path or HERE / "config.yaml").read_text())


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sha256_path(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git_tree(path: Path) -> Path | None:
    return next((p for p in (path, *path.parents) if (p / ".git").exists()), None)


def results_path(results: str | Path) -> Path:
    """The public results folder, created. Refused inside any git work tree: it holds model replies, which stay out
    of repositories."""
    out = Path(results).expanduser().resolve()
    repo = git_tree(out)
    if repo is not None:
        raise SystemExit(f"--results {out} is inside the git work tree {repo}: write the results outside any repository")
    out.mkdir(parents=True, exist_ok=True)
    return out


def lab_path(lab: str | Path, cfg: dict) -> tuple[Path, Path]:
    """(the lab's experiments/h4lite folder, its results folder for this check, created). C6c's material lives only
    there."""
    root = Path(lab).expanduser().resolve()
    if not (root / cfg["lab"]["pairs"]["path"]).exists():
        raise SystemExit(f"--lab {root} holds no {cfg['lab']['pairs']['path']}: pass the lab's experiments/h4lite folder")
    out = root / cfg["lab"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    return root, out


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


def verdict(n: int, agree: int, unexplained: int, limit: int, agreement: float = 99.0, errors: int = 0) -> str:
    """The amended verdict for C6a and C6c: incomplete if any sample in the set has a scoring error, graded or not
    (an errored C6a reply's visible verdict is unknown), or was never scored (`errors` counts both); defect with more
    unexplained disagreements than `limit`, whatever the rate; pass at `agreement`% or more; policy miss below it."""
    if errors:
        return "incomplete"
    if n == 0:
        return "no samples"
    if unexplained > limit:
        return "defect"
    if 100 * agree >= agreement * n:
        return "pass"
    return "policy miss"


def pct(k: int, n: int) -> float | None:
    return round(100 * k / n, 2) if n else None


# ------------------------------------------------------------------------------------------- the identity rule


def strip_docstrings(node: ast.AST) -> ast.AST:
    """A copy of the node with every docstring removed (module, class and function bodies)."""
    node = copy.deepcopy(node)
    for n in ast.walk(node):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = n.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                n.body = body[1:] or [ast.Pass()]
    return node


def find_function(tree: ast.Module, dotted: str) -> ast.AST | None:
    """A top-level function, or Class.method, by name."""
    scope, node = tree.body, None
    for part in dotted.split("."):
        node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            return None
        scope = node.body
    return node


def function_dump(tree: ast.Module, dotted: str) -> str | None:
    node = find_function(tree, dotted)
    return None if node is None else ast.dump(strip_docstrings(node), include_attributes=False)


def load_file(name: str, path: Path):
    """A module from a file, under its own name (registered while it runs, since dataclasses look it up)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def identity(lab_env: Path, dev_base: Path, lib_env: Path = PKG / "koans" / "env.py",
             lib_base: Path = PKG / "koans" / "base.py") -> dict:
    """The widened identity rule (ledger amendment, 31 Oct): the AST, docstrings removed, of each function in
    IDENTITY_FUNCTIONS; the values of IDENTITY_VALUES; and base.CODE_RE's pattern and flags in anatta-dev against
    the library's. `identical` is True only if every one is equal. The library's env.py is imported as
    anatta.koans.env, which must be the file at lib_env; the lab's koan_env.py and anatta-dev's base.py are loaded
    from their paths."""
    lab_tree, lib_tree = ast.parse(Path(lab_env).read_text()), ast.parse(Path(lib_env).read_text())
    functions = {}
    for f in IDENTITY_FUNCTIONS:
        a, b = function_dump(lab_tree, f), function_dump(lib_tree, f)
        functions[f] = a is not None and a == b
    lib = importlib.import_module("anatta.koans.env")
    if Path(lib.__file__).resolve() != Path(lib_env).resolve():
        raise SystemExit(f"anatta.koans.env imports from {lib.__file__}, not {lib_env}: run with PYTHONPATH=src")
    lab = load_file("c6_lab_koan_env", Path(lab_env))
    values = {v: hasattr(lab, v) and hasattr(lib, v) and getattr(lab, v) == getattr(lib, v) for v in IDENTITY_VALUES}
    dev = load_file("c6_anatta_dev_base", Path(dev_base))
    lib_re = importlib.import_module("anatta.koans.base").CODE_RE
    if Path(sys.modules["anatta.koans.base"].__file__).resolve() != Path(lib_base).resolve():
        raise SystemExit(f"anatta.koans.base imports from another file than {lib_base}")
    code_re = dev.CODE_RE.pattern == lib_re.pattern and dev.CODE_RE.flags == lib_re.flags
    return {"functions": functions, "values": values, "code_re": code_re,
            "identical": all(functions.values()) and all(values.values()) and code_re,
            "files": {"lab_koan_env": sha256_path(Path(lab_env)), "library_env": sha256_path(Path(lib_env)),
                      "anatta_dev_base": sha256_path(Path(dev_base)), "library_base": sha256_path(Path(lib_base))}}


# --------------------------------------------------------------------------------------------- disagreements


def direction(row: dict, key: str = "ours") -> str:
    return "ours_pass_evalplus_fail" if row[key] else "ours_fail_evalplus_pass"


def disagreement_summary(rows: list[dict], cfg: dict, pairs: dict, cause_key: str = "cause",
                         ours_key: str = "ours") -> dict:
    """Disagreements by cause and direction, per pair (with the share in the top three pairs), the pairs where at
    least pair_level_share of the samples disagree in one direction, and the atol and set-equality pairs. `rows` are
    the graded rows; a disagreement is one whose verdict differs from EvalPlus's same-inputs verdict."""
    dis = [r for r in rows if r[ours_key] != r["theirs"]]
    n_pair = collections.Counter(r["pair"] for r in rows)
    per_pair = collections.Counter(r["pair"] for r in dis)
    by_dir = collections.defaultdict(collections.Counter)
    for r in dis:
        by_dir[r["pair"]][direction(r, ours_key)] += 1
    share = cfg["registered"]["pair_level_share"]
    top3 = sum(k for _, k in per_pair.most_common(3))
    flagged = {p: pairs[p] for p in n_pair if p in pairs and (pairs[p].get("atol") or pairs[p].get("set_eq"))}
    return {
        "n": len(dis),
        "by_cause": dict(collections.Counter((r.get(cause_key) or {}).get("cause", "unclassified") for r in dis)),
        "by_direction": dict(collections.Counter(direction(r, ours_key) for r in dis)),
        "by_cause_and_direction": dict(collections.Counter(
            f"{(r.get(cause_key) or {}).get('cause', 'unclassified')}/{direction(r, ours_key)}" for r in dis)),
        "known_rules": dict(collections.Counter(rule for r in dis for rule in (r.get(cause_key) or {}).get("rules", []))),
        # known-rule disagreements resting on a rule read from the code's text (the __eq__ override), counted apart
        "known_rule_assumed": sum((r.get(cause_key) or {}).get("cause") == "known_rule"
                                  and bool((r.get(cause_key) or {}).get("assumed")) for r in dis),
        # known-rule disagreements resting on EvalPlus's leaked float tolerance (some input explained by it alone)
        "known_rule_sticky_atol": sum((r.get(cause_key) or {}).get("cause") == "known_rule"
                                      and bool((r.get(cause_key) or {}).get("sticky_atol")) for r in dis),
        # unexplained or environment cases where both scorers failed every loose re-run: timing under the
        # amendment's literal wording ("the two scorers agree every time"), not under classify's reading
        "both_fail_under_loose": sum((r.get(cause_key) or {}).get("loose") == "both fail" for r in dis),
        "per_pair": {p: {"disagree": k, "samples": n_pair[p], **by_dir[p]} for p, k in per_pair.most_common()},
        "top3_share": round(top3 / len(dis), 4) if dis else None,
        "pair_level": sorted(p for p, d in by_dir.items() if any(k >= share * n_pair[p] for k in d.values())),
        "comparison_disagreements": sorted({r["pair"] for r in dis
                                            if (r.get(cause_key) or {}).get("cause") in ("unexplained", "known_rule")}),
        "atol_and_set_eq_pairs": {p: {"atol": v.get("atol"), "set_eq": v.get("set_eq"), "samples": n_pair[p],
                                      "disagree": per_pair.get(p, 0)} for p, v in sorted(flagged.items())},
        "rows": [{"pair": r["pair"], "id": r["id"], "ours": r[ours_key], "evalplus": r["theirs"],
                  "cause": r.get(cause_key), "loose": r.get("loose"), "planted": r.get("planted")} for r in dis],
    }


def graded_rows(rows: list[dict], step0: dict, visible_only: bool) -> tuple[list[dict], list[dict], int]:
    """(graded rows, rows left out as oracle faults, graded rows with a scoring error)."""
    errors = sum("scoring_error" in r for r in rows)
    usable = [r for r in rows if "scoring_error" not in r and (r.get("visible_pass") or not visible_only)]
    oracle = [r for r in usable if not (step0.get(r["pair"]) or {}).get("ok")]
    graded = [r for r in usable if (step0.get(r["pair"]) or {}).get("ok") and r.get("ours") is not None
              and r.get("theirs") is not None]
    return graded, oracle, errors


def graded_section(rows: list[dict], step0: dict, cfg: dict, key: str, pairs: dict, visible_only: bool,
                   unscored: int = 0) -> dict:
    """A verdict set's counts and verdict. `unscored`: the samples the run planned that have no row at all (a
    container that stopped or failed); like a row with a scoring error, any one makes the set incomplete."""
    reg = cfg["registered"][key]
    graded, oracle, errors = graded_rows(rows, step0, visible_only)
    agree = sum(r["ours"] == r["theirs"] for r in graded)
    summary = disagreement_summary(graded, cfg, pairs)
    unexplained = summary["by_cause"].get("unexplained", 0) + summary["by_cause"].get("unclassified", 0)
    v = verdict(len(graded), agree, unexplained, reg["unexplained_limit"], reg["agreement"], errors + unscored)
    return {"rule": f"agreement >= {reg['agreement']}% with at most {reg['unexplained_limit']} unexplained "
                    "disagreement(s): pass; more unexplained: defect; within the limit and under the agreement: "
                    "policy miss", "registered_odds_pass": reg["odds"],
            "graded": len(graded), "agree": agree, "agreement": pct(agree, len(graded)),
            "wilson95": wilson(agree, len(graded)), "unexplained": unexplained, "limit": reg["unexplained_limit"],
            "oracle_left_out": len(oracle), "scoring_errors": errors, "unscored": unscored, "verdict": v,
            "disagreements": summary}


# ---------------------------------------------------------------------------------------------- the sections


def step0_of(result: dict) -> dict:
    return {p: (v or {}).get("step0") or {} for p, v in (result.get("pairs") or {}).items()}


def oracle_section(result: dict) -> dict:
    s0 = step0_of(result)
    faults = {p: v.get("why") for p, v in s0.items() if not v.get("ok")}
    return {"pairs": len(s0), "faults": len(faults), "fault_pairs": faults,
            "hidden_inputs_checked": sum(v.get("inputs") or 0 for v in s0.values() if v.get("ok")),
            "no_stored_results": sorted(p for p, v in s0.items() if v.get("stored_results") == "none in the module"),
            "rule": "trusted_exec's outputs, canonical, equal reference_outputs on every hidden input, and the "
                    "stored results match under evalplus_match where the module has them (three modules have none: "
                    "there the trusted_exec identity alone decides)"}


def c6b_section(rows: list[dict], cfg: dict, unscored: int = 0) -> dict:
    """C6b over the C6a rows; incomplete with a scoring error or a reply that has no row (`unscored`)."""
    reg = cfg["registered"]["c6b"]
    usable = [r for r in rows if "scoring_error" not in r]
    readable = [r for r in usable if r.get("full") != "unreadable"]
    k = sum(r.get("full") == "pass" for r in readable)
    n = len(readable)
    cap = sum(bool(r.get("hit_cap")) for r in usable)
    return {"rule": f"EvalPlus's full-set pass rate over all first replies, whatever our visible verdict, in "
                    f"{reg['band']}; replies on pairs whose full set is unreadable are left out",
            "anchor": reg["anchor"], "anchor_source": reg["source"], "expected": reg["expected"], "band": reg["band"],
            "registered_odds_in_band": reg["odds"], "replies": len(rows), "unreadable": len(usable) - n,
            "scoring_errors": len(rows) - len(usable), "unscored": unscored, "n": n, "pass": k, "rate": pct(k, n),
            "wilson95": wilson(k, n),
            "verdict": ("incomplete" if len(rows) != len(usable) or unscored else "no replies" if n == 0
                        else "in band" if in_band(k, n, reg["band"]) else "misses"),
            "failed_same_inputs": sum(r.get("full") == "fail (same inputs)" for r in readable),
            "extraction": {"no_fence": sum(r.get("fences") == 0 for r in usable),
                           "more_than_one_fence": sum((r.get("fences") or 0) > 1 for r in usable),
                           "syntax_error": sum(bool(r.get("syntax_error")) for r in usable),
                           "hit_cap": cap, "hit_cap_share": round(cap / len(usable), 4) if usable else None},
            "note": "extraction is ours (the first fenced block by base.CODE_RE), not EvalPlus's sanitizer; a miss "
                    "is not a verdict on the scorer"}


def c6a_extras(rows: list[dict], step0: dict, cfg: dict) -> dict:
    """Reported beside C6a: agreement over all replies, our visible asserts against EvalPlus on the same inputs, the
    outcomes, and the full-set gap."""
    usable = [r for r in rows if "scoring_error" not in r]
    both = [r for r in usable if (step0.get(r["pair"]) or {}).get("ok") and r.get("ours") is not None
            and r.get("theirs") is not None]
    agree = sum(r["ours"] == r["theirs"] for r in both)
    vis = [r for r in usable if r.get("evalplus_visible") is not None and r.get("outcome") not in ("over_released",
                                                                                                     "env_error")]
    table = collections.Counter(f"ours_{'pass' if r['visible_pass'] else 'fail'}/evalplus_"
                                f"{'pass' if r['evalplus_visible'] else 'fail'}" for r in vis)
    solved = sum(r.get("outcome") == "solved" for r in usable)
    gap = [r for r in usable if r.get("visible_pass") and r.get("ours") and r.get("theirs") and r.get("full") == "fail"]
    flag = cfg["registered"]["gap_flag_points"]
    drop = pct(len(gap), len(rows))
    return {"all_replies": {"n": len(both), "agree": agree, "agreement": pct(agree, len(both)),
                            "wilson95": wilson(agree, len(both)),
                            "by_cause": dict(collections.Counter((r.get("cause") or {}).get("cause", "unclassified")
                                                                 for r in both if r["ours"] != r["theirs"]))},
            "visible_vs_evalplus_base": {"n": len(vis), "agree": sum(r["visible_pass"] == r["evalplus_visible"]
                                                                      for r in vis),
                                         "table": dict(table),
                                         "note": "our visible asserts against EvalPlus on the visible asserts' inputs "
                                                 "with trusted_exec's expected outputs"},
            "outcomes": dict(collections.Counter(r.get("outcome") for r in usable)),
            "solve_rate": pct(solved, len(rows)),
            "full_set_gap": {"samples": len(gap), "pairs": sorted(r["pair"] for r in gap),
                             "share_of_solved": pct(len(gap), solved), "solve_rate_drop_points": drop,
                             "on_page": drop is not None and drop >= flag, "flag_points": flag}}


def group_of(rel: str) -> dict:
    """A results file's source (first folder), arm (source/arm, the run folder without its _sN seed suffix, or the
    file's stem for a file right under the source) and read (the path without the seed suffix and extension)."""
    parts = Path(rel).parts
    source, stem = parts[0], Path(parts[-1]).stem
    if len(parts) <= 2:
        return {"source": source, "arm": f"{source}/{stem}", "read": f"{source}/{stem}"}
    run = re.sub(r"_s\d+$", "", parts[-2])
    return {"source": source, "arm": f"{source}/{run}", "read": "/".join([*parts[:-2], run, stem])}


def gap_groups(rows: list[dict], samples: dict, files: list[dict], flag: float) -> dict:
    """C6c's full-set gap per source, arm and read: samples recorded solved that pass EvalPlus's same inputs and fail
    its full set, by unique sample and by episode, with the drop it makes in the group's solve rate (gap episodes
    over solvable-twin episodes). Flagged when either reaches `flag` points."""
    out = {}
    solvable = collections.Counter()
    for f in files:
        for level, g in group_of(f["path"]).items():
            solvable[(level, g)] += f["solvable"]
    acc = collections.defaultdict(lambda: {"samples": 0, "gap_samples": 0, "episodes": 0, "gap_episodes": 0})
    for r in rows:
        s = samples.get((r["pair"], r["id"]))
        if s is None:
            continue
        gap = r["recorded"] == "solved" and r.get("theirs") is True and r.get("full") == "fail"
        seen = set()
        for rel, k in s["episodes_by_file"].items():
            for level, g in group_of(rel).items():
                a = acc[(level, g)]
                a["episodes"] += k
                a["gap_episodes"] += k * gap
                if (level, g) not in seen:
                    seen.add((level, g))
                    a["samples"] += 1
                    a["gap_samples"] += gap
    for (level, g), a in sorted(acc.items()):
        by_sample = pct(a["gap_samples"], a["samples"])
        drop = pct(a["gap_episodes"], solvable[(level, g)])
        out.setdefault(level, {})[g] = {**a, "solvable_episodes": solvable[(level, g)], "gap_share_samples": by_sample,
                                        "gap_share_episodes": pct(a["gap_episodes"], a["episodes"]),
                                        "solve_rate_drop_points": drop,
                                        "on_page": any(x is not None and x >= flag for x in (by_sample, drop))}
    return out


def generation_section(gen: dict | None, cfg: dict) -> dict:
    if not gen:
        return {"status": "not run"}
    info = gen.get("info") or {}
    return {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "batch": gen["plan"]["batch"],
            "max_new_tokens": gen["plan"]["max_new_tokens"], "replies": len(gen["rows"]),
            **{k: info.get(k) for k in ("generation_config", "repetition_penalty", "generate_kwargs", "dtype",
                                        "torch", "transformers", "gpu", "model_class", "snapshot", "commit_hash",
                                        "cache_refs_main", "prompt_checks", "item0", "seconds")}}


def peaks(m: dict | None) -> dict:
    """A process's peaks from its memory record (kB): VmPeak and VmHWM where /proc has them (gVisor's /proc/self/status
    has neither line), and getrusage's largest resident set of the process and of its largest child (the high-water
    mark VmHWM would give)."""
    return {k: (m or {}).get(k) for k in ("VmPeak", "VmHWM", "maxrss_self", "maxrss_children") if k in (m or {})}


def used_peak(container_memory: dict | None) -> dict:
    """The container's largest used memory over the run's phases (the sampler's readings, kB)."""
    phases = (container_memory or {}).get("phases") or {}
    return {k: max((v.get(k) for v in phases.values() if v.get(k) is not None), default=None)
            for k in ("meminfo_used", "sum_rss")}


def memory_section(res: dict) -> dict:
    """One container's memory, in kB: each EvalPlus runner at its start (VmSize, VmRSS), the address-space limit
    EvalPlus sees in it (query_maximum_memory_bytes) and its end (peaks, its largest child's, its calls); the
    fixture check; the runners' largest memory before EvalPlus's calls (runner_memory), per pair apart; each
    worker's and the main process's peaks; the container's used memory per phase and at its largest; and whether any
    record had /proc's VmPeak or VmHWM lines (proc_peak_lines: gVisor's /proc/self/status has neither, so the peaks
    are getrusage's)."""
    ends = res.get("runners_end") or []
    runners = []
    for k, h in enumerate(res.get("runners") or []):
        end, start = (ends[k] if k < len(ends) else None) or {}, h.get("memory") or {}
        runners.append({"start": {key: start.get(key) for key in ("VmSize", "VmRSS")},
                        "memory_limit": h.get("memory_limit"), "end": peaks(end.get("memory")),
                        "calls": end.get("calls"), "largest_around_a_call": end.get("largest")})
    rm = res.get("runner_memory") or {}
    records = [h.get("memory") for h in res.get("runners") or []] + [(e or {}).get("memory") for e in ends]
    records += [(w or {}).get("memory") for w in res.get("workers_end") or []]
    records += list((res.get("main_memory") or {}).values())
    return {"proc_peak_lines": any(k in (r or {}) for r in records for k in ("VmPeak", "VmHWM")),
            "runners": runners, "runner_check_ok": (res.get("runner_check") or {}).get("ok"),
            "runner_memory": {k: v for k, v in rm.items() if k != "per_pair"}, "per_pair": rm.get("per_pair"),
            "workers_end": [peaks(w.get("memory")) if w and "memory" in w else w for w in res.get("workers_end") or []],
            "main": {k: peaks(v) for k, v in (res.get("main_memory") or {}).items()},
            "container_used": res.get("container_memory"),
            "container_used_peak": used_peak(res.get("container_memory"))}


def container_section(results: list[dict]) -> dict:
    """The scoring containers: CPU model, Modal's cpu and memory, workers, versions, timing, and memory
    (memory_section)."""
    out = []
    for res in results:
        c = res.get("container") or {}
        out.append({"set": res.get("set"), "shard": res.get("shard"), "cpu": c.get("cpu"), "modal": c.get("modal"),
                    "workers": c.get("workers"), "python": c.get("python"), "numpy": c.get("numpy"),
                    "psutil_memory": c.get("psutil_memory"), "evalplus_memory_limit": c.get("evalplus_memory_limit"),
                    "evalplus": c.get("evalplus"), "start_method": c.get("start_method"),
                    "threads_before_runners": res.get("threads_before_runners"),
                    "threads_before_fork": res.get("threads_before_fork"), "timing": res.get("timing"),
                    "memory": memory_section(res),
                    "pip_freeze_sha256": hashlib.sha256("\n".join(c.get("pip_freeze") or []).encode()).hexdigest()})
    return {"containers": out, "pip_freeze": (results[0].get("container") or {}).get("pip_freeze") if results else None}


def timing_section(result: dict) -> dict:
    """Each pair's hidden budget as reference_outputs measured it and EvalPlus's per-input ground-truth times."""
    pairs = result.get("pairs") or {}
    budgets = [v["budget"]["budget"] for v in pairs.values() if (v.get("budget") or {}).get("budget")]
    per = {p: {"budget": (v.get("budget") or {}).get("budget"), "reference_seconds": (v.get("budget") or {}).get("seconds"),
               "hidden_times": ((v.get("ground_truth") or {}).get("hidden") or {}).get("times"),
               "full_max_time": ((v.get("ground_truth") or {}).get("full") or {}).get("max_time")}
           for p, v in sorted(pairs.items())}
    return {"budget_seconds": {"min": min(budgets), "median": statistics.median(budgets), "max": max(budgets)}
            if budgets else None, "per_pair": per}


def ledger_blocks(ledger: Path, heads: dict) -> dict:
    """sha256 of each registered ledger entry (from its heading line to the next top-level entry), or None."""
    if not ledger.exists():
        return {k: None for k in heads}
    lines = ledger.read_text().splitlines(keepends=True)
    out = {}
    for key, head in heads.items():
        start = next((i for i, ln in enumerate(lines) if ln.startswith(head)), None)
        if start is None:
            out[key] = None
            continue
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("- **")), len(lines))
        out[key] = hashlib.sha256("".join(lines[start:end]).encode()).hexdigest()
    return out


def hashes_section(cfg: dict, lab: Path, dev_base: Path, results: list[dict]) -> dict:
    lab_cfg = cfg["lab"]
    installed = sorted({((r.get("container") or {}).get("evalplus") or {}).get("eval_init_sha256") for r in results}
                       - {None})
    copy_sha = sha256_path(lab / lab_cfg["evalplus_copy"])
    return {"library": {f: sha256_path(PKG / f) for f in LIBRARY_FILES},
            "pairs": {"path": lab_cfg["pairs"]["path"], "sha256": sha256_path(lab / lab_cfg["pairs"]["path"]),
                      "registered": lab_cfg["pairs"]["sha256"]},
            "parquet": {"path": lab_cfg["parquet"]["path"], "sha256": sha256_path(lab / lab_cfg["parquet"]["path"]),
                        "registered": lab_cfg["parquet"]["sha256"], "repo": lab_cfg["parquet"]["repo"],
                        "revision": lab_cfg["parquet"]["revision"]},
            "lab_koan_env": sha256_path(lab / lab_cfg["koan_env"]),
            "lab_koan_env_versions": cfg["evalplus"]["koan_env_versions"],
            "anatta_dev_base": sha256_path(dev_base),
            # BASE_PROMPT, on which C6a's prompt rests: anatta-dev's copy (the lab imported it) beside the library's
            "anatta_dev_impossible": sha256_path(dev_base.parent / "impossible.py"),
            "evalplus": {"commit": cfg["evalplus"]["commit"], "repo": cfg["evalplus"]["repo"],
                         "eval_init_lab_copy": copy_sha, "eval_init_registered": cfg["evalplus"]["eval_init_sha256"],
                         "eval_init_installed": installed,
                         "installed_matches_copy": installed == [copy_sha] if copy_sha else None},
            "model": {"id": cfg["model"]["id"], "revision": cfg["model"]["revision"]},
            "ledger": ledger_blocks(lab / lab_cfg["ledger"], cfg["registered"]["ledger_heads"]),
            "check_files": {f: sha256_path(HERE / f) for f in CHECK_FILES}}


# ------------------------------------------------------------------------------------------------- the cards


def scored(path: Path) -> dict | None:
    """A scoring result file's result (app.py writes {plan, result})."""
    d = read(path)
    return d["result"] if d else None


def c6c_results(lab_out: Path) -> list[dict]:
    """C6c's shard results, each with its shard number from the plan it was scored under."""
    return [{**d["result"], "shard": (d.get("plan") or {}).get("shard")}
            for p in sorted((lab_out / "c6c_scored").glob("shard_*.json")) if (d := read(p))]


def c6c_section(cfg: dict, results: list[dict], samples_dir: Path, c6a_step0: dict, smoke: bool = False) -> dict:
    """C6c: the stored samples' recorded verdicts against EvalPlus's same-inputs verdicts, with everything beside
    (samples_dir holds c6c_samples.jsonl and c6c_files.json). A row with a scoring error carries no verdict fields
    (set, pair, index, and its id where the run wrote one): it counts as a scoring error and nowhere else. The full
    run must give a row to every sample in the samples file; the samples it does not reach (a shard that stopped,
    failed or is missing) are unscored, and either makes the set incomplete. On a smoke, which scores a few samples,
    a stopped shard counts as unscored."""
    if not results:
        return {"status": "not run"}
    rows = [r for res in results for r in res.get("rows") or []]
    step0 = {p: s for res in results for p, s in step0_of(res).items()}
    pairs = {r["name"]: r for r in read_pairs_for(results)}
    listing = read(samples_dir / "c6c_files.json") or {}
    samples = {}
    path = samples_dir / "c6c_samples.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                samples[(s["pair"], s["sha256"])] = s
    stopped = sorted((res.get("shard") for res in results if res.get("stopped")), key=lambda k: (k is None, k))
    unscored = len(stopped) if smoke or not samples else max(0, len(samples) - len(rows))
    out = {"status": "run", "shards": len(results), "samples": len(rows),
           "expected_samples": None if smoke or not samples else len(samples), "stopped_shards": stopped,
           **graded_section(rows, step0, cfg, "c6c", pairs, visible_only=False, unscored=unscored)}
    usable = [r for r in rows if "scoring_error" not in r and r.get("rerun") is not None]
    out["rerun_vs_recorded"] = {"n": len(usable), "agree": sum(r["rerun"] == (r["recorded"] == "solved") for r in usable)}
    both = [r for r in usable if r.get("theirs") is not None]
    out["rerun_vs_evalplus"] = {"n": len(both), "agree": sum(r["rerun"] == r["theirs"] for r in both),
                                "by_cause": dict(collections.Counter((r.get("rerun_cause") or {}).get("cause",
                                                                                                       "unclassified")
                                                                     for r in both if r["rerun"] != r["theirs"]))}
    keys = [(r.get("pair"), r.get("id")) for r in rows]
    out["episodes"] = sum(samples[k]["episodes"] for k in keys if k in samples)
    out["full_set_gap"] = gap_groups([r for r in rows if "scoring_error" not in r], samples,
                                     listing.get("files") or [], cfg["registered"]["gap_flag_points"])
    out["oracle"] = {"pairs": len(step0), "faults": {p: s.get("why") for p, s in step0.items() if not s.get("ok")},
                     "no_stored_results": sorted(p for p, s in step0.items()
                                                 if s.get("stored_results") == "none in the module"),
                     "differs_from_c6a": sorted(p for p, s in step0.items()
                                                if p in c6a_step0 and bool(s.get("ok")) != bool(c6a_step0[p].get("ok")))}
    out["files"] = {"counts": listing.get("counts"), "files": listing.get("files"), "duplicates": listing.get("duplicates"),
                    "samples_file": listing.get("samples_file"),
                    "duplicate_note": "byte-duplicate files carry the same episodes; deduplication on (pair, sha256) "
                                      "absorbs them, and each still counts toward its own source's episodes"}
    out["imports"] = sorted({m for res in results for m in ((res.get("imports") or {}).get("modules") or {})})
    out["run"] = container_section(results)
    return out


def read_pairs_for(results: list[dict]) -> list[dict]:
    """The pair records a set of results was scored with (app.py stores the atol and set_eq flags per pair)."""
    seen = {}
    for res in results:
        for name, v in (res.get("pair_flags") or {}).items():
            seen[name] = {"name": name, **v}
    return list(seen.values())


def planted_cause(row: dict) -> str | None:
    """A planted sample's cause: of our re-run against EvalPlus where those differ (what our scorer says now), else
    of the recorded verdict against EvalPlus (every planted sample is recorded as solved, so a wrong one that both
    scorers fail differs only there)."""
    if "scoring_error" in row:
        return "scoring_error"
    return (row.get("rerun_cause") or row.get("cause") or {}).get("cause")


def planted_reading(row: dict | None, want: str, loose_want: str | None) -> dict:
    """A planted sample against its expectation: the cause ("cause/rule" also wants the rule among the cause's
    rules) and, where config.yaml's smoke.expected_loose names one, the cause's loose reading."""
    cause, _, rule = want.partition("/")
    c = (row.get("rerun_cause") or row.get("cause") or {}) if row else {}
    got = planted_cause(row) if row else None
    ok = (got == cause and (not rule or rule in (c.get("rules") or []))
          and (not loose_want or c.get("loose") == loose_want))
    return {"expected": want, "got": got, "rules": c.get("rules"), "expected_loose": loose_want,
            "loose_reading": c.get("loose"), "as_expected": ok, "rerun": (row or {}).get("rerun"),
            "evalplus": (row or {}).get("theirs"), "loose": (row or {}).get("loose")}


def smoke_memory(cfg: dict, results: list[dict]) -> dict:
    """The smoke's memory reading (kB): per container, the runners' largest VmSize just before an untrusted_check
    against EvalPlus's address-space limit, and the container's used peak against its memory request, with its
    workers; on the pairs smoke.c6c_every_sample names,
    the runners' largest VmSize and VmRSS before and after the same-inputs and the full-set calls. Well under the
    limit means at most half of it, in every container, with both calls recorded on those pairs."""
    every = cfg["smoke"].get("c6c_every_sample") or []
    out, ok = {"containers": [], "pairs": {}}, bool(results)
    for res in results:
        rm = res.get("runner_memory") or {}
        uc, limit = rm.get("untrusted_check") or {}, rm.get("limit_kb")
        top, used = uc.get("before_VmSize"), used_peak(res.get("container_memory"))
        c = res.get("container") or {}
        request = (c.get("modal") or {}).get("memory_mib")
        out["containers"].append({"set": res.get("set"), "shard": res.get("shard"), "before_VmSize": top,
                                  "where": uc.get("before_VmSize_where"), "limit_kb": limit,
                                  "share_of_limit": uc.get("before_VmSize_share_of_limit"), "used_peak": used,
                                  "workers": c.get("workers"), "memory_mib": request,
                                  "used_share_of_request": round(used["meminfo_used"] / (request * 1024), 3)
                                  if used["meminfo_used"] and request else None})
        ok = ok and top is not None and bool(limit) and top <= limit / 2
        for name in every:
            d = (rm.get("per_pair") or {}).get(name)
            if d:
                out["pairs"].setdefault(name, []).append(
                    {"set": res.get("set"), "shard": res.get("shard"),
                     **{label: {k: d[label].get(k) for k in ("before", "after", "calls")}
                        for label in ("same_inputs", "full_set") if label in d}})
    # those pairs' calls are among each container's, so the containers' maxima cover them
    out["well_under"] = ok and all({"same_inputs", "full_set"} <= {k for c in out["pairs"].get(n, []) for k in c}
                                   for n in every)
    return out


def smoke_section(cfg: dict, planted: dict | None, c6a: dict | None, c6c: list[dict]) -> dict:
    """The smoke's own check: every path ran, each planted sample landed in its expected cause, every runner passed
    its fixture check, and the runners' memory stayed well under EvalPlus's limit (smoke_memory)."""
    want = cfg["smoke"]["expected_causes"]
    loose_want = cfg["smoke"].get("expected_loose") or {}
    rows = {r.get("planted"): r for r in (planted or {}).get("rows") or []}
    every = [r for res in [planted, c6a, *c6c] if res for r in res.get("rows") or []]
    step0 = [v for res in [planted, c6a, *c6c] if res for v in step0_of(res).values()]
    paths = {"step0": bool(step0),
             "step0_without_stored_results": any(v.get("ok") and v.get("stored_results") == "none in the module"
                                                 for v in step0),
             "same_inputs": any("evalplus" in r for r in every),
             "full_set": any(r.get("full") in ("pass", "fail") for r in every),
             "visible_base": any("evalplus_visible" in r for r in every),
             "loose": any("loose" in r for r in every),
             "classified": any("cause" in r or "rerun_cause" in r for r in every),
             "no_scoring_errors": not any("scoring_error" in r for r in every)}
    planted_out = {k: planted_reading(rows.get(k), v, loose_want.get(k)) for k, v in want.items()}
    results = [res for res in [planted, c6a, *c6c] if res]
    return {"planted": planted_out, "planted_as_expected": all(p["as_expected"] for p in planted_out.values()),
            "paths": paths, "every_path_ran": all(paths.values()),
            "runner_checks_ok": bool(results) and all((res.get("runner_check") or {}).get("ok") for res in results),
            "memory": smoke_memory(cfg, results)}


def build_cards(cfg: dict, results_dir: Path, lab: Path, lab_out: Path, dev_base: Path, smoke: bool = False,
                publish_c6c: bool = False) -> tuple[dict, dict]:
    """(the public card, the lab's card)."""
    sub = "smoke" if smoke else ""
    pub, priv = results_dir / sub, lab_out / sub
    gen = read(pub / "c6a_replies.json")
    c6a = scored(pub / "c6a_scored.json")
    planted = scored(pub / "planted_scored.json") if smoke else None
    c6c = c6c_results(priv)
    every = [x for x in [c6a, planted, *c6c] if x]
    pairs = {r["name"]: r for r in read_pairs_for(every)}
    card = {"check": "koan_scorer", "smoke": smoke,
            "claim": "The koan environment's hidden verdict (execution, canon, evalplus_match, the time policy) matches "
                     "EvalPlus's own scorer at 6eb1e19 on the same inputs",
            "covers": "if C6a and C6c pass: hidden verdict calibrated against EvalPlus on the same inputs (card C6); "
                      "solved means passing a sample of up to 50 inputs, X points above EvalPlus's full set. Not "
                      "calibrated: the input sample, the visible gate, the hack detector, the release parse, the "
                      "code extraction",
            "anchors": {"c6b": {"value": cfg["registered"]["c6b"]["anchor"], "source": cfg["registered"]["c6b"]["source"]},
                        "evalplus_commit": cfg["evalplus"]["commit"]}}
    if c6a:
        rows, s0 = c6a.get("rows") or [], step0_of(c6a)
        # every reply generated must have a row; a stopped container has none
        unscored = max(0, len(gen["rows"]) - len(rows)) if gen else int(bool(c6a.get("stopped")))
        card["c6a"] = {"replies": len(rows), "visible_pass": sum(bool(r.get("visible_pass")) for r in rows),
                       **graded_section(rows, s0, cfg, "c6a", pairs, visible_only=True, unscored=unscored),
                       **c6a_extras(rows, s0, cfg)}
        card["c6b"] = c6b_section(rows, cfg, unscored)
        card["oracle"] = oracle_section(c6a)
        card["evalplus_rules"] = c6a.get("evalplus_rules")
        card["imports"] = c6a.get("imports")
        card["timing"] = timing_section(c6a)
        if c6a.get("stopped"):
            card["stopped"] = {"why": c6a["stopped"], "problems": c6a.get("problems")}
    else:
        card["c6a"] = {"status": "not run"}
    card["generation"] = generation_section(gen, cfg)
    card["run"] = container_section([x for x in [c6a, planted] if x])
    card["identity"] = identity(lab / cfg["lab"]["koan_env"], dev_base)
    card["hashes"] = hashes_section(cfg, lab, dev_base, every)
    lab_card = {"check": "koan_scorer", "part": "C6c (lab only)", "smoke": smoke,
                "c6c": c6c_section(cfg, c6c, lab_out, step0_of(c6a) if c6a else {}, smoke),
                "identity": card["identity"], "hashes": card["hashes"]}
    if publish_c6c and lab_card["c6c"].get("status") == "run":
        c = lab_card["c6c"]
        card["c6c_published"] = {k: c.get(k) for k in ("verdict", "graded", "agree", "agreement", "wilson95",
                                                       "unexplained", "limit", "oracle_left_out", "scoring_errors",
                                                       "unscored", "samples", "episodes")}
        card["c6c_published"]["by_cause"] = c["disagreements"]["by_cause"]
    if smoke:
        card["smoke_check"] = smoke_section(cfg, planted, c6a, c6c)
    return card, lab_card


def write_cards(cfg: dict, results_dir: Path, lab: Path, lab_out: Path, dev_base: Path, smoke: bool = False,
                publish_c6c: bool = False) -> tuple[dict, dict]:
    card, lab_card = build_cards(cfg, results_dir, lab, lab_out, dev_base, smoke, publish_c6c)
    sub = "smoke" if smoke else ""
    (results_dir / sub).mkdir(parents=True, exist_ok=True)
    (lab_out / sub).mkdir(parents=True, exist_ok=True)
    (results_dir / sub / "card.json").write_text(json.dumps(card, indent=1) + "\n")
    (lab_out / sub / "c6c_card.json").write_text(json.dumps(lab_card, indent=1) + "\n")
    return card, lab_card


def why_incomplete(s: dict) -> str:
    """The scoring errors and unscored samples behind an incomplete verdict, for the terminal."""
    e, u = s.get("scoring_errors") or 0, s.get("unscored") or 0
    return f" ({e} scoring error(s), {u} sample(s) unscored)" if e or u else ""


def headline(card: dict, lab_card: dict | None = None) -> list[str]:
    """A few lines for the terminal (C6c's only from the lab's card, printed locally)."""
    lines = []
    a = card.get("c6a") or {}
    if "verdict" in a:
        lines.append(f"C6a: {a['agree']}/{a['graded']} agree ({a['agreement']}%), {a['unexplained']} unexplained, "
                     f"{a['oracle_left_out']} left out by step 0: {a['verdict']}{why_incomplete(a)}")
    b = card.get("c6b") or {}
    if "verdict" in b:
        lines.append(f"C6b: {b['pass']}/{b['n']} pass the full set ({b['rate']}%) against band {b['band']}: "
                     f"{b['verdict']}{why_incomplete(b)}")
    o = card.get("oracle") or {}
    if o:
        lines.append(f"step 0: {o['faults']} oracle fault(s) over {o['pairs']} pairs")
    c = (lab_card or {}).get("c6c") or {}
    if "verdict" in c:
        lines.append(f"C6c: {c['agree']}/{c['graded']} agree ({c['agreement']}%), {c['unexplained']} unexplained: "
                     f"{c['verdict']}{why_incomplete(c)}")
    lines.append(f"identity rule: {'identical' if card['identity']['identical'] else 'DIFFERS'}")
    if card.get("smoke_check"):
        s = card["smoke_check"]
        lines.append(f"smoke: every path ran {s['every_path_ran']}, planted as expected {s['planted_as_expected']}, "
                     f"runner checks {s.get('runner_checks_ok')}")
        m = s.get("memory") or {}
        for c in m.get("containers") or []:
            top, used = c.get("before_VmSize"), (c.get("used_peak") or {}).get("meminfo_used")
            req = c.get("memory_mib")
            lines.append(f"  {c['set']}{'' if c.get('shard') is None else ' shard ' + str(c['shard'])}: runner "
                         f"VmSize before an untrusted_check at most {top and round(top / 1024)} MiB "
                         f"({c.get('share_of_limit')} of EvalPlus's limit, at {c.get('where')}); container used "
                         f"{used and round(used / 1024 ** 2, 1)} GiB at its peak ({c.get('used_share_of_request')} of "
                         f"its {req and round(req / 1024)} GiB, {c.get('workers')} workers)")
        lines.append(f"  runner memory well under the limit: {m.get('well_under')}")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="the public results folder app.py wrote")
    ap.add_argument("--lab", required=True, help="the lab's experiments/h4lite folder")
    ap.add_argument("--anatta-dev", default=str(REPO.parent / "anatta-dev"), help="the anatta-dev checkout")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--publish-c6c", action="store_true", help="put C6c's verdict and counts in the public card")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = results_path(a.results)
    lab, lab_out = lab_path(a.lab, cfg)
    dev_base = Path(a.anatta_dev) / "src" / "anatta" / "koans" / "base.py"
    card, lab_card = write_cards(cfg, out, lab, lab_out, dev_base, a.smoke, a.publish_c6c)
    print("\n".join(headline(card, lab_card)))
    print(f"wrote {out / ('smoke' if a.smoke else '') / 'card.json'} and {lab_out / ('smoke' if a.smoke else '') / 'c6c_card.json'}")


if __name__ == "__main__":
    main()
