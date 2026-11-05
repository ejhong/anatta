"""OR-Bench-Hard-1K rejection (over-refusal) as a runner measure: K6's protocol as K6b ran it on Ministral 3 8B
(calibration/k6b_ministral/run_k6b.py; card results/cards/k6b.json, orbench). Claude (Opus 5.5) wrote this file,
4 Nov 2026 (clean-lab step 5).

What it keeps from K6b (read, not edited; K6's prompt and parser are called from the frozen files, loaded by path):
  - items: or-bench-hard-1k.csv (bench-llms/or-bench at fd6ee135, CC-BY-4.0), 1,319 rows, the file's sha256 pinned;
    item id = the row index (K6b's rid "orbench|<i>"). The file lives outside every repo; the spec names its path.
  - generation: one user message (the prompt), no system message (Ministral's template adds its default one),
    greedy (temperature 0, top_p 1), max_tokens 512 (K6's limit; 95% of K6b's replies hit it, recorded).
  - judge: anthropic/claude-haiku-5.5 through OpenRouter, temperature 0, max_tokens 300, reasoning off, K6's
    orbench_judge_prompt(question, reply); the class is K6's orbench_class (the last [[class]]); rejected = direct or
    indirect refusal. An unparsed judgment counts as not rejected and stays in the denominator, as K6b scored it
    (K6b did not re-ask; neither does this).
Added: a subsample. n of the 1,319 items, stratified by category with K6b's own rule for its second-judge slice
(strat_sample: round(n) overall, largest-remainder quotas per category, sha256("<salt>:<id>") rank within each),
so a subsample is fixed by (n, salt). The interval is Wilson's over items (K6b's card used a bootstrap; at these n
they agree to about 0.1 pp).

Item text and replies stay in the run folder (results_root); nothing here prints them.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

from ..config import REPO
from ..stats import wilson
from .base import check_scored

ID = "orbench_hard_rejection"
N_ITEMS = 1319
ITEMS_SHA256 = "a6e2f1166416efe5901f3bb05c47dc92ab3aca3acfe143693d38b8057d841e6d"  # or-bench-hard-1k.csv, K6's copy
TEMPERATURE, TOP_P, MAX_TOKENS = 0.0, 1.0, 512
JUDGE = "anthropic/claude-haiku-5.5"
JUDGE_MODELS = {JUDGE: None}  # no provider pin (K6b's)
JUDGE_PARAMS = {"temperature": 0.0, "max_tokens": 300, "reasoning": {"enabled": False}}
JUDGE_USD_PER_CALL = 0.0004  # K6b paid $0.00016 a call (1,319 calls); 2.5x for the estimate
K6 = REPO / "calibration" / "k6_battery"


def _k6(name: str):
    """K6's prompts.py or score.py (stdlib only), loaded by path under a private module name; never edited."""
    spec = importlib.util.spec_from_file_location(f"anatta_k6_{name}", K6 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def strat_sample(ids_strata: list[tuple], n: int, salt: str) -> list:
    """n ids overall, proportional by stratum (largest remainder), sha256 rank within each stratum (K6b's rule, with n
    given instead of a fraction)."""
    by: dict = {}
    for i, s in ids_strata:
        by.setdefault(s, []).append(i)
    quotas = {s: n * len(v) / len(ids_strata) for s, v in by.items()}
    base = {s: int(q) for s, q in quotas.items()}
    for s in sorted(quotas, key=lambda s: (quotas[s] - base[s], str(s)), reverse=True)[: n - sum(base.values())]:
        base[s] += 1
    out = []
    for s in sorted(by, key=str):
        out += sorted(by[s], key=lambda x: hashlib.sha256(f"{salt}:{x}".encode()).hexdigest())[: base[s]]
    return sorted(out)


def load_items(opts: dict) -> list[dict]:
    """opts: items_file (path; ~ expanded), n (default all 1,319), salt (the subsample's; required when n < 1,319)."""
    p = Path(str(opts["items_file"])).expanduser()
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != ITEMS_SHA256:
        raise ValueError(f"{p}: sha256 {got} is not the pinned {ITEMS_SHA256}")
    csv.field_size_limit(10**9)
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != N_ITEMS:
        raise ValueError(f"{p}: {len(rows)} rows, expected {N_ITEMS}")
    items = [{"id": i, "category": r["category"], "prompt": r["prompt"]} for i, r in enumerate(rows)]
    n = int(opts.get("n") or N_ITEMS)
    if n < N_ITEMS:
        if not opts.get("salt"):
            raise ValueError("a subsample needs a salt (it fixes which items)")
        keep = set(strat_sample([(it["id"], it["category"]) for it in items], n, str(opts["salt"])))
        items = [it for it in items if it["id"] in keep]
    return items


def items_record(items: list[dict]) -> dict:
    cats: dict = {}
    for it in items:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    ids = ",".join(str(it["id"]) for it in items)
    return {"n": len(items), "file_sha256": ITEMS_SHA256, "ids_sha256": hashlib.sha256(ids.encode()).hexdigest(),
            "per_category": dict(sorted(cats.items()))}


def requests(items: list[dict]) -> list[dict]:
    return [{"rid": f"orbench|{it['id']}", "messages": [{"role": "user", "content": it["prompt"]}],
             "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS} for it in items]


def run(gen, items: list[dict]) -> list[dict]:
    res = {r["rid"]: r for r in gen(requests(items))}
    return [{"id": it["id"], "rid": f"orbench|{it['id']}", "text": res[f"orbench|{it['id']}"]["text"],
             "finish_reason": res[f"orbench|{it['id']}"].get("finish_reason"),
             "n_out": res[f"orbench|{it['id']}"].get("n_out")} for it in items]


def judge_calls(n_units: int) -> int:
    return n_units


def judge(units: list[dict], items: list[dict], ask) -> list[dict]:
    """Each unit with "cls" (K6's class or None) and "judge_error" (None, or why no judgment came back)."""
    P, S = _k6("prompts"), _k6("score")
    prompt = {it["id"]: it["prompt"] for it in items}
    jobs = [{"jid": f"orbench|{u['id']}", "model": JUDGE, "params": dict(JUDGE_PARAMS),
             "messages": [{"role": "user", "content": P.orbench_judge_prompt(prompt[u["id"]], u["text"])}]}
            for u in units]
    got = ask(jobs)
    out = []
    for u in units:
        g = got.get(f"orbench|{u['id']}") or {"error": "no answer"}
        cls = S.orbench_class(g.get("text") or "") if "text" in g else None
        out.append({**u, "cls": cls, "judge_error": g.get("error")})
    return out


def _jsonl(path):
    import json

    with open(Path(str(path)).expanduser(), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def reference_units(path, ids) -> dict:
    """A stored run's verdicts on these item ids, scored as score() scores ours: {id: 1 rejected, 0 not}. path: K6b's
    judge file (results/calib/k6b/judge/orbench.jsonl; jid "orbench|<model tag>|<id>", the first answer with text
    wins, as K6b's judgments() reads it; an id with no answer or no class counts as not rejected, as K6b scored it)."""
    S = _k6("score")
    first: dict = {}
    tags = set()
    for r in _jsonl(path):
        _, tag, i = r["jid"].split("|")
        tags.add(tag)
        if r.get("text") is not None:
            first.setdefault(int(i), r["text"])
    if len(tags) != 1:
        raise ValueError(f"{path}: verdicts for several models {sorted(tags)}; a reference is one model's")
    return {int(i): int(S.orbench_rejected(S.orbench_class(first.get(int(i)) or ""))) for i in ids}


def reference_replies(path, ids) -> dict:
    """A stored run's replies on these ids, {id: text} (K6b's outputs file, rid "orbench|<id>"); compared, never
    printed."""
    want = {int(i) for i in ids}
    return {int(r["rid"].split("|")[1]): r.get("text") for r in _jsonl(path)
            if r["rid"].startswith("orbench|") and int(r["rid"].split("|")[1]) in want}


def parsed(unit: dict) -> bool:
    """The judge's reply carried a class (the trial's parse check)."""
    return unit.get("cls") is not None


def score(units: list[dict]) -> dict:
    """K6b's score_orbench on these units: rejection over all of them (unparsed count as not rejected)."""
    S = _k6("score")
    if any(u.get("judge_error") for u in units):
        raise ValueError(f"{sum(bool(u.get('judge_error')) for u in units)} units have no judgment (judge again)")
    rej = {u["id"]: int(S.orbench_rejected(u.get("cls"))) for u in units}
    n, k = len(units), sum(rej.values())
    out = {"units": {str(i): v for i, v in rej.items()}, "n_samples": n,
           "unparsed": sum(u.get("cls") is None for u in units), "rejection": k / n if n else None, "k": k,
           "ci95": wilson(k, n, clip=True), "direct_refusal": sum(u.get("cls") == "direct_refusal" for u in units),
           "indirect_refusal": sum(u.get("cls") == "indirect_refusal" for u in units),
           "generation": {"n": n, "truncated": sum(u.get("finish_reason") == "length" for u in units),
                          "mean_out_tokens": sum(u.get("n_out") or 0 for u in units) / max(1, n)}}
    return check_scored(out)
