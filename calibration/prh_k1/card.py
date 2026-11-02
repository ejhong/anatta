"""Calibration K1's card, from a run's results folder (the files app.py's main saved), graded by the registered block
in the run's own plan.json: the targets and tolerances that grade a run are the ones it ran under, and editing them
afterwards makes another plan.

  card.json  The five Qwen rows: score, target, difference, grade, best layer pair and dtype. The verdict (pass, near,
             fail or incomplete) by the registered rule. Beside them, ungraded: unbiased CKA against the paper's
             values, the score under Huh et al.'s clipping, the subsample interval and each model's difference from
             the 7B, the permutation null's tau, the calibrated score and the boundary ties.
             The OpenLLaMA arm beside Koepke et al.'s values (lineage-comparable, not protocol-identical), at k = 10
             and k = 1 under both clippings: read against the anchor only if the smoke's sentencepiece comparison, on
             the same tokenizer files, found no caption tokenized otherwise, and otherwise with the tokenizer caveat.
             For every model: padding side, pad and eos, the longest padded batch, truncation, attention
             implementation, dtype, device map, best pair, boundary ties, interval, time and memory.
             The commits read (the paper's code loaded an unpinned main), the environment beside what the paper
             states, the precision flags, and the spend.
             A smoke's card holds its checks (the batch check part by part), the tokenizers, the memory prediction
             and an estimate of the full run, read as nothing.
Run as `python calibration/prh_k1/card.py --results DIR [--smoke]`, or through app.py::card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import k1 as K  # noqa: E402

CLAIM = ("The lab's own code reproduces Balestriero & LeCun 2026 (arXiv 2610.39302v1, Table 16): mutual 10-NN on "
         "WIT-1024 between the Qwen2.5 base models (0.5B to 14B) and DINOv2 ViT-B/14, each within 0.005 of its target "
         "(registered 1 Nov 2026, 14:05 UTC).")
READING = {
    "pass": "All five match. The lab's code reproduces the paper's measurement on WIT-1024, and P1 may build on it.",
    "near": ("No model fails, and at least one is only near. The measurement is usable with the offset noted, after a "
             "recorded look at why (the registration's rule)."),
    "fail": ("At least one model is more than 0.010 from its target. P1 waits until the cause is found (the "
             "registration's rule)."),
    "incomplete": "A graded number is missing: the run didn't finish, and nothing is graded.",
}
PAPER_ENV = ("torch >= 2.6, transformers >= 4.56 and timm >= 1.0 (prh-proof's requirements); 14B and up split across "
             "two GPUs in bfloat16 (Table 16's text); the versions and GPU behind Table 16 unstated")
PINNED = "the paper's code loaded each repository's main, unpinned, on the day it ran"
NEAR_NOTE = ("A near on a bfloat16 model (3B, 7B, 14B) while the float32 ones (0.5B, 1.5B) match points at kernels, "
             "not the protocol: in bfloat16 another GPU or kernel flips a few neighbours (of order 0.001). A float32 "
             "rerun of the 3B (12 GB of weights) is a cheap ungraded follow-up.")


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    return yaml.safe_load(Path(path or HERE / "config.yaml").read_text())


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


def prepare_folder(root: Path) -> Path:
    """prepare's folder beside the runs' (check 3's pattern): <results>_prepare."""
    return Path(root).parent / f"{Path(root).name}_prepare"


def runs_folder(prep: Path) -> Path:
    """The runs' folder beside prepare's: <results> for <results>_prepare."""
    name = Path(prep).name
    if not name.endswith("_prepare"):
        raise SystemExit(f"--results {prep} doesn't end in _prepare: pass --runs-results so the runs' spend counts")
    return Path(prep).parent / name[: -len("_prepare")]


def read(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def grade(score: float | None, target: float, tol: dict) -> str | None:
    """The registered rule for one model: within tol["match"] of the target a match, within tol["near"] near, beyond
    that a fail; None without a score. The difference is rounded to 1e-9 first, so a score exactly 0.005 from its
    target (in decimal) is a match whatever the binary rounding of the subtraction."""
    if score is None:
        return None
    d = round(abs(score - target), 9)
    return "match" if d <= tol["match"] else "near" if d <= tol["near"] else "fail"


def verdict(grades: dict) -> str:
    """K1 passes if all five match, is near if none fails but some are only near, and fails otherwise; a fail decides
    even with a model missing, and otherwise a missing model leaves it incomplete."""
    g = list(grades.values())
    if "fail" in g:
        return "fail"
    if any(x is None for x in g):
        return "incomplete"
    return "pass" if all(x == "match" for x in g) else "near"


def rate(cfg: dict, gpu: str = "L40S", tier: str = "cpu4_32g") -> float:
    c = cfg["cost"]
    return c["gpu_per_hour"][gpu] + c["cpu_mem_per_hour"][tier]


def spend_facts(d: Path) -> dict | None:
    s = read(d / "spend.json")
    if s is None:
        return None
    return {"this_folder_usd": round(sum(e["usd"] for e in s["entries"]), 4), "k1_total_usd": s["total_usd"],
            "limit_usd": s.get("limit_usd"), "entries": s["entries"], "note": "an estimate; Modal's usage page is the "
                                                                              "authority"}


def environment(res: dict | None) -> dict:
    """The run's environment beside what the paper states."""
    return {"gpu": get(res, "env", "gpu"), "python": get(res, "env", "python"), "installed": get(res, "env", "installed"),
            "paper": PAPER_ENV}


def smoke_card(res: dict | None, cfg: dict) -> dict:
    r = get(res, "result") or {}
    est = r.get("estimate")
    if est is not None:
        secs = est["total_seconds"] + cfg["cost"]["upper_s"]["startup"]
        est = {**est, "with_startup_seconds": round(secs, 1), "usd_at_l40s": round(secs / 3600 * rate(cfg), 2)}
    toks = {k: {"padding_side": t.get("padding_side"), "pad_token_id": t.get("pad_token_id"),
                "pad_is_eos": t.get("pad_is_eos"), "pad_was_set": t.get("pad_was_set"), "class": t.get("class"),
                "max_padded": t.get("max_padded"), "truncated": t.get("truncated"), "tokens": t.get("tokens"),
                "sentencepiece": t.get("sentencepiece")} for k, t in (r.get("tokens") or {}).items()}
    return {"checks": r.get("checks"), "passed": r.get("passed"), "tokens": toks, "memory": r.get("memory"),
            "estimate": est, "measures": r.get("measures"), "precision": r.get("precision"),
            "scores_read_as_nothing": {k: get(r, "scores", k) for k in ("mknn", "cka", "huh", "null", "subsample")}}


def tokenizer_reading(d: Path, plan: dict) -> dict:
    """The OpenLLaMA arm's reading by the registered rule: against its anchor only if the smoke (d/smoke), run on the
    same tokenizer files as this run, found at most max_sentencepiece_differ captions that sentencepiece tokenizes
    otherwise; with the tokenizer caveat if it found more, or if no such smoke is there."""
    most = plan["registered"]["open_llama"]["max_sentencepiece_differ"]
    splan, smoke = read(d / "smoke" / "plan.json"), read(d / "smoke" / "smoke.json")
    files = get(plan, "prepared", "open_llama_3b")
    same = (files is not None and splan is not None and smoke is not None
            and smoke.get("plan_id") == K.plan_id(splan) and get(splan, "prepared", "open_llama_3b") == files)
    sp = get(smoke, "result", "tokens", "open_llama_3b", "sentencepiece") if same else None
    clean = sp is not None and sp["differ"] <= most
    return {"reading": "read against the anchor" if clean else "read with the tokenizer caveat",
            "sentencepiece": sp, "smoke_on_same_files": same, "max_sentencepiece_differ": most}


def model_facts(r: dict, key: str) -> dict:
    """What the card records of one language model: its scores' layer pair, interval and ties, and how it ran."""
    sc, t = get(r, "scores", key), get(r, "text", key) or {}
    return {"score": get(sc, "mknn", "score"), "best_pair": get(sc, "mknn", "layers"), "huh": get(sc, "huh", "score"),
            "cka": get(sc, "cka", "score"), "interval": get(sc, "subsample", "interval"),
            "boundary_ties": get(sc, "null", "boundary_ties"), "null_tau": get(sc, "null", "tau"),
            "dtype": t.get("dtype"), "stored_dtype": t.get("stored_dtype"),
            "attn_implementation": t.get("attn_implementation"), "device_map_auto": t.get("device_map_auto"),
            "device_map": t.get("device_map"), "hidden_states": t.get("hidden_states"), "seconds": t.get("seconds"),
            "peak_gib": t.get("peak_gib"), "gpu": t.get("gpu"), "padding_side": get(t, "tokenizer", "padding_side"),
            "pad_is_eos": get(t, "tokenizer", "pad_is_eos"), "pad_was_set": get(t, "tokenizer", "pad_was_set"),
            "pad_token_id": get(t, "tokenizer", "pad_token_id"), "eos_token_id": get(t, "tokenizer", "eos_token_id"),
            "max_padded": get(t, "tokens", "max_padded"), "truncated": get(t, "tokens", "truncated"),
            "error": get(r, "errors", key)}


def build_card(d: Path, cfg: dict | None = None) -> dict:
    """The card from d, graded by d/plan.json's registered block."""
    d = Path(d)
    cfg = cfg if cfg is not None else load_config()
    plan = read(d / "plan.json")
    if plan is None:
        raise SystemExit(f"no plan.json in {d}")
    name = "smoke" if plan["smoke"] else "full"
    res = read(d / f"{name}.json")
    if res is not None and (res.get("plan_id") != K.plan_id(plan) or get(res, "env", "code") != plan["code"]):
        raise SystemExit(f"{d / (name + '.json')} was made under another plan than {d / 'plan.json'}")
    reg = plan["registered"]
    card = {"check": "K1", "name": "the Platonic measurement with the lab's own code", "smoke": plan["smoke"],
            "read_as_nothing": plan["smoke"], "claim": CLAIM, "plan_id": K.plan_id(plan),
            "stopped": read(d / "stopped.json"), "spend": spend_facts(d), "environment": environment(res),
            "pinned": {"note": PINNED, "repositories": {k: {f: p.get(f) for f in ("repo", "ref", "revision")}
                                                        for k, p in (plan.get("prepared") or {}).items()}},
            "source": reg["source"], "ledger": reg["ledger"]}
    if plan["smoke"]:
        return {**card, **smoke_card(res, cfg)}
    r = get(res, "result") or {}
    scores = r.get("scores") or {}
    rows = {}
    for key, target in reg["targets"].items():
        sc = scores.get(key)
        s = get(sc, "mknn", "score")
        rows[key] = {"score": s, "target": target, "diff": None if s is None else s - target,
                     "grade": grade(s, target, reg["tolerance"]), "layers": get(sc, "mknn", "layers"),
                     "dtype": get(r, "text", key, "dtype"), "cka": get(sc, "cka", "score"),
                     "cka_layers": get(sc, "cka", "layers"), "cka_paper": reg["cka"].get(key),
                     "huh": get(sc, "huh", "score"), "huh_layers": get(sc, "huh", "layers"),
                     "interval": get(sc, "subsample", "interval"), "minus_reference": get(r, "differences", key),
                     "null_tau": get(sc, "null", "tau"), "calibrated": get(sc, "null", "calibrated"),
                     "null_identity_matches": get(sc, "null", "identity_matches"),
                     "boundary_ties": get(sc, "null", "boundary_ties"), "error": get(r, "errors", key)}
    v = verdict({k: x["grade"] for k, x in rows.items()})
    ol = scores.get("open_llama_3b")
    open_llama = None
    if "open_llama_3b" in plan["text"]:
        t = get(r, "text", "open_llama_3b") or {}
        open_llama = {"anchor": reg["open_llama"], "comparable": reg["open_llama"]["comparable"],
                      **tokenizer_reading(d, plan),
                      "k10": {"paper": get(ol, "mknn", "score"), "paper_layers": get(ol, "mknn", "layers"),
                              "huh": get(ol, "huh", "score"), "huh_layers": get(ol, "huh", "layers")},
                      "k1": {"paper": get(ol, "extra_k", "1", "paper", "score"),
                             "paper_layers": get(ol, "extra_k", "1", "paper", "layers"),
                             "huh": get(ol, "extra_k", "1", "huh", "score"),
                             "huh_layers": get(ol, "extra_k", "1", "huh", "layers")},
                      "cka": get(ol, "cka", "score"), "null_tau": get(ol, "null", "tau"),
                      "dtype": t.get("dtype"), "stored_dtype": t.get("stored_dtype"), "tokenizer": t.get("tokenizer"),
                      "error": get(r, "errors", "open_llama_3b"), "graded": False}
    v_facts = r.get("vision") or {}
    return {**card, "graded": rows,
            "verdict": {"outcome": v, "reading": READING[v],
                        "counts": {g: sum(x["grade"] == g for x in rows.values()) for g in ("match", "near", "fail")},
                        "notes": [NEAR_NOTE] if v == "near" else []},
            "tolerance": reg["tolerance"], "open_llama": open_llama,
            "models": {k: model_facts(r, k) for k in plan["text"]},
            "vision": {"data_config": v_facts.get("data_config"), "blocks": v_facts.get("blocks"),
                       "num_prefix_tokens": v_facts.get("num_prefix_tokens"), "seconds": v_facts.get("seconds"),
                       "peak_gib": v_facts.get("peak_gib"), "gpu": v_facts.get("gpu")},
            "errors": r.get("errors"), "precision": r.get("precision"), "score_seconds": r.get("score_seconds"),
            "by_layer": {k: get(sc, "by_layer") for k, sc in scores.items()}}


def write_card(cfg: dict | None, d: Path) -> dict:
    card = build_card(Path(d), cfg)
    (Path(d) / "card.json").write_text(json.dumps(card, indent=1) + "\n")
    return card


def fmt(x, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def onoff(x) -> str:
    return "n/a" if x is None else "on" if x else "off"


def batch_lines(bi: dict) -> list[str]:
    """The smoke's batch check, part by part, and the raw figures it records without reading them."""
    p, r = bi["parts"], bi["recorded"]
    ok = {k: ("ok" if part["passed"] else "FAILED") for k, part in p.items()}
    return [f"  batch 1 vs 64 (registered rule): embedding mean {p['embed']['max_abs']:.2e} (at most "
            f"{p['embed']['tol']:.0e}) {ok['embed']}; after _prh_prep {p['prepped']['max_abs']:.2e} (at most "
            f"{p['prepped']['tol']:.0e}) {ok['prepped']}; repeatable {ok['identical']}; finite {ok['finite']}; "
            f"{p['float32']['dtype']} {ok['float32']}" + (f"; failed: {', '.join(bi['failed'])}" if bi["failed"] else ""),
            f"  recorded, not read: raw max abs {r['max_abs']:.2e} (raw tolerance {r['raw_tol_abs']:.0e}"
            f"{'' if r['within_raw_tol'] else ', breached'}), relative {r['max_rel']:.2e}"]


def headline(card: dict) -> list[str]:
    spend = card.get("spend") or {}
    tail = f"  spend about ${spend.get('k1_total_usd', 0):.2f} for all of K1 so far (estimate; ceiling $" \
           f"{spend.get('limit_usd') or 6:.2f})" if spend else "  no spend recorded here"
    if card["smoke"]:
        checks = card.get("checks") or {}
        failed = [k for k, c in checks.items() if not c["passed"]]
        lines = [f"K1 smoke (read as nothing): {len(checks) - len(failed)} of {len(checks)} checks passed"
                 + (f"; FAILED: {', '.join(failed)}" if failed else "")]
        if checks.get("text_batch_invariance"):
            lines += batch_lines(checks["text_batch_invariance"])
        for k, t in (card.get("tokens") or {}).items():
            sp = t.get("sentencepiece")
            lines.append(f"  {k}: padding {t['padding_side']}, pad {'eos' if t['pad_is_eos'] else t['pad_token_id']}, "
                         f"longest batch {t['max_padded']} tokens, {t['truncated']} truncated"
                         + (f", sentencepiece differs on {sp['differ']} of {sp['texts']}" if sp else ""))
        for k, mm in (card.get("memory") or {}).items():
            if k != "calibration":
                lines.append(f"  memory {k}: predicted {mm['gib']} GiB" + (f" of {mm['total_gib']}" if mm.get("total_gib")
                                                                            else "") + ("" if mm.get("fits") is None
                                                                                        else (", fits" if mm["fits"]
                                                                                              else ", MAY NOT FIT")))
        cal = (card.get("memory") or {}).get("calibration")
        if cal:
            lines.append(f"  memory formula on {cal['model']}: predicted {cal['predicted']} GiB, measured "
                         f"{cal['measured']} GiB")
        est = card.get("estimate")
        if est:
            lines.append(f"  full run: about {est['with_startup_seconds'] / 60:.0f} min, about ${est['usd_at_l40s']:.2f} "
                         "on the L40S (a rough estimate)")
        return lines + [tail] + ([f"  stopped: {card['stopped']['reason']}"] if card.get("stopped") else [])
    v = card["verdict"]
    lines = [f"K1 (full run): {v['outcome'].upper()} ({v['counts']['match']} match, {v['counts']['near']} near, "
             f"{v['counts']['fail']} fail; within {card['tolerance']['match']} a match, {card['tolerance']['near']} near)"]
    for k, r in card["graded"].items():
        if r["score"] is None:
            lines.append(f"  {k}: no score" + (f" ({r['error']})" if r.get("error") else ""))
            continue
        lo, hi = r["interval"] or (None, None)
        lines.append(f"  {k}: {fmt(r['score'])} vs {r['target']:.3f} ({r['diff']:+.4f}) {r['grade']}; layers "
                     f"{r['layers'][0]} x {r['layers'][1]}; {r['dtype']}; CKA {fmt(r['cka'])} (paper "
                     f"{r['cka_paper']:.3f}); Huh {fmt(r['huh'])}; 95% [{fmt(lo)}, {fmt(hi)}]; calibrated "
                     f"{fmt(r['calibrated'])}")
    ol = card.get("open_llama")
    if ol:
        a, sp = ol["anchor"], ol.get("sentencepiece")
        why = (f"sentencepiece differs on {sp['differ']} of {sp['texts']} captions" if sp
               else "no smoke comparison on these tokenizer files")
        lines.append(f"  open_llama_3b (ungraded; the anchor is {ol['comparable']}): k=10 {fmt(ol['k10']['paper'])} "
                     f"(Huh {fmt(ol['k10']['huh'])}) vs {a['k10']}; k=1 {fmt(ol['k1']['paper'])} (Huh "
                     f"{fmt(ol['k1']['huh'])}) vs {a['k1']}; {ol['reading']} ({why})"
                     + (f"; error {ol['error']}" if ol.get("error") else ""))
    lines.append(f"  {v['reading']}")
    lines += [f"  {n}" for n in v.get("notes") or []]
    pf = card.get("precision") or {}
    if pf:
        lines.append(f"  precision: float32 matmul {pf.get('float32_matmul_precision')}, TF32 matmuls "
                     f"{onoff(pf.get('cuda_matmul_allow_tf32'))}, cuDNN TF32 {onoff(pf.get('cudnn_allow_tf32'))} "
                     f"(torch's defaults, which the paper's code leaves too); torch {pf.get('torch')}, CUDA "
                     f"{pf.get('cuda')}, {(card.get('environment') or {}).get('gpu')}")
    return lines + [tail] + ([f"  stopped: {card['stopped']['reason']}"] if card.get("stopped") else [])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    d = run_folder(results_path(a.results), a.smoke)
    print("\n".join(headline(write_card(None, d))))


if __name__ == "__main__":
    main()
