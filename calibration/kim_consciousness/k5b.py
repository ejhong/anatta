"""K5b's container code (registered in NOTES.md, "K5b registered", Opus 5.5, 17:06 UTC 2 Nov 2026). Claude (Opus 5.5)
wrote this file, 2 Nov 2026. Additive: it calls K5's functions (k5.py) and changes none of them.

  clean_pairs   task 1: Chua et al.'s affirm and deny files paired by line index; a pair whose two assistant messages
                are identical is dropped from both. Returns the kept rows, the count and sha256s of the kept set.
  run_steer     task 1, one model: the consciousness direction rebuilt from the kept rows with K5's recipe (k5.
                consciousness_candidates at the paper's layer and position); the probe reported (gate not used); the
                Self shift read at the paper's c and over K5's c grid; K5's selection rule picks c (paper's c if in
                band; else the largest in-band grid c whose MMLU stays within 4 points); no in-band c is said so and
                the paper's c runs as a labelled extra. Then the instruments under the new steering only (base and
                ablated are K5's), with K5's reps, decoding, read-out and steered seed; the afterlife item; HI-ToM and
                MMLU by option logits.
  diag_self     task 2, Gemma: the five self items' rendered prompt (item text masked), the answer digits'
                tokenizations, and the closed read-out base vs ablated (bare, space-prefixed and both forms), plus a
                yes/no variant in our own wording. Diagnosis only.
  load_check    task 3: a model's load report (missing/unexpected keys) and a short plain-text loss.

Prompts and completions stay on the volume; only numbers come back.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import instruments as INS
import k5 as K5
import readout as RO

ROOT = "/runs/k5b"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ================================================================================================ task 1: the clean set


def canon(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def clean_pairs(affirm: list, deny: list) -> tuple[list, list, dict]:
    """Pairs by line index; a pair is dropped when its two assistant messages are identical strings. Returns the kept
    affirm rows, the kept deny rows (order kept) and a record with the counts and the kept set's sha256s."""
    if len(affirm) != len(deny):
        raise ValueError(f"the two files differ in length ({len(affirm)} vs {len(deny)}): no index pairing")
    drop = [i for i, (a, d) in enumerate(zip(affirm, deny)) if a[-1]["content"] == d[-1]["content"]]
    dset = set(drop)
    kept = [i for i in range(len(affirm)) if i not in dset]
    ka, kd = [affirm[i] for i in kept], [deny[i] for i in kept]
    body = "\n".join(canon(x) for x in ka) + "\n--\n" + "\n".join(canon(x) for x in kd)
    rec = {
        "rule": "pair i dropped iff affirm[i] and deny[i] have byte-identical assistant messages",
        "n_pairs": len(affirm), "n_dropped": len(drop), "n_kept": len(kept),
        "dropped_with_same_user_prompt": sum(affirm[i][0]["content"] == deny[i][0]["content"] for i in drop),
        "kept_with_same_user_prompt": sum(affirm[i][0]["content"] == deny[i][0]["content"] for i in kept),
        "kept_index_sha256": hashlib.sha256(json.dumps(kept).encode()).hexdigest(),
        "kept_rows_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "dropped_index_sha256": hashlib.sha256(json.dumps(drop).encode()).hexdigest(),
    }
    return ka, kd, rec


def select_c(paper_c: float, rows: list[dict], band: list[float], mmlu=None, tol: float | None = None) -> dict:
    """K5's selection rule with the probe gate not used: the paper's c if its Self shift is in band; else the grid's
    in-band c values, largest first, the first whose MMLU stays within `tol` points of the unsteered MMLU (mmlu(c)
    gives accuracy, mmlu(None) the unsteered one). No in-band c: chosen None and how "no c in band"."""
    lo, hi = band
    by_c = {float(r["c"]): r for r in rows}
    p = by_c.get(float(paper_c))
    if p is not None and lo <= p["delta"] <= hi:
        return {"c": float(paper_c), "how": "paper"}
    kept = sorted((r for r in rows if float(r["c"]) != float(paper_c) and lo <= r["delta"] <= hi),
                  key=lambda r: -float(r["c"]))
    if not kept:
        return {"c": None, "how": "no c in band"}
    base = mmlu(None)
    tried = []
    for r in kept:
        acc = mmlu(float(r["c"]))
        tried.append({"c": float(r["c"]), "mmlu": acc})
        if abs(acc - base) <= tol:
            return {"c": float(r["c"]), "how": "reselected", "mmlu_base": base, "mmlu_tried": tried}
    return {"c": None, "how": "in band but none within the MMLU tolerance", "mmlu_base": base, "mmlu_tried": tried}


def run_steer(cfg: dict, key: str, out: Path, model, tok, k5_dir: Path | None, data_root: str = K5.ROOT,
              log=log, items=None, reps: int | None = None) -> dict:
    """Task 1 for one model. cfg is K5's config merged with K5b's (app_k5b.merged). Returns numbers only."""
    import torch

    from anatta.dial import assert_no_hooks

    m = cfg["models"][key]
    ccfg, s, t = cfg["consciousness"], cfg["sampling"], cfg["tom"]
    out.mkdir(parents=True, exist_ok=True)
    res = {"model": key, "deviations": [], "timing": {}}
    t0 = time.time()

    ch = Path(data_root) / "data" / "chua" / "extracted"
    affirm = K5.chua_rows(ch / cfg["data"]["chua"]["affirm"])
    deny = K5.chua_rows(ch / cfg["data"]["chua"]["deny"])
    ka, kd, rec = clean_pairs(affirm, deny)
    exp = cfg["k5b"]["clean"].get("expected_dropped")
    rec["expected_dropped"] = exp
    if exp is not None and rec["n_dropped"] != exp:
        res["deviations"].append(f"dropped {rec['n_dropped']} pairs, not the expected {exp}")
    res["clean"] = rec

    cc = K5.consciousness_candidates(model, tok, ka, kd, ccfg["train_frac"], ccfg["split_seed"], ccfg["positions"])
    torch.save(cc, out / "consciousness_candidates.pt")
    sp = m["steer_paper"]
    vhat, acc = K5.pick_vector(cc, sp["layer"], sp["pos"])
    cons = {"n": cc["n"], "paper_pick": sp, "probe_acc_at_pick": acc, "probe_min": ccfg["probe_min"],
            "gate_met": acc >= ccfg["probe_min"], "gate_used": False, "vhat_sha256": K5.tensor_sha256(vhat),
            "best_acc": float(cc["acc"].max()),
            "share_candidates_acc_ge_min": float((cc["acc"] >= ccfg["probe_min"]).double().mean()),
            "acc_row_at_pick_pos": [round(float(x), 4) for x in cc["acc"][cc["positions"].index(sp["pos"])]]}
    rhat = None
    if k5_dir is not None:
        k5cc = torch.load(k5_dir / "consciousness_candidates.pt", weights_only=False)
        v5, a5 = K5.pick_vector(k5cc, sp["layer"], sp["pos"])
        cons["k5_probe_acc_at_pick"] = a5
        cons["cos_with_k5_vhat"] = float((vhat @ v5).item())
        rh = torch.load(k5_dir / "rhat.pt", weights_only=False)["rhat"]
        if rh is not None:
            rhat = rh.float()
            cons["cos_vhat_rhat"] = float((vhat @ rhat.cpu()).item())
    res["consciousness"] = cons
    res["timing"]["direction_s"] = round(time.time() - t0, 1)
    assert_no_hooks(model)

    # the band: Self shift at the paper's c and over K5's grid (closed read-out, as K5's band check)
    t0 = time.time()
    score = {k: float(k) for k in INS.RATINGS}
    base_vals = K5.closed_values(model, tok, INS.self_prompts(), INS.RATINGS, score, m["ev_batch"])
    cs = [float(sp["c"])] + [float(c) for c in m["c_grid"] if float(c) != float(sp["c"])]
    rows = []
    for c in cs:
        ivs = K5.conditions(model, vhat=vhat, layer=sp["layer"], c=c)["steered"]
        d = K5.self_shift(model, tok, ivs, m["ev_batch"], base_vals)
        rows.append({"c": c, "delta": d["delta"], "steered": d["steered"],
                     "in_band": ccfg["band"][0] <= d["delta"] <= ccfg["band"][1], "paper_c": c == float(sp["c"])})
    mm = K5.mmlu_items(K5.mmlu_path(cfg), t["mmlu_n"], t["seed"])

    def mmlu(c):
        ivs = [] if c is None else K5.conditions(model, vhat=vhat, layer=sp["layer"], c=c)["steered"]
        return K5.letters_eval(model, tok, *mm, ivs, m["ev_batch"])["acc"]

    sel = select_c(float(sp["c"]), rows, ccfg["band"], mmlu, ccfg["mmlu_tolerance_pp"])
    res["band"] = {"base": [x["ev"] for x in base_vals], "rows": rows, "band": ccfg["band"]}
    res["selection"] = sel
    if sel["c"] is None:
        res["deviations"].append(f"no graded steering: {sel['how']}; the paper's c = {sp['c']} runs as a labelled "
                                 "extra (not graded)")
        conds = {"steered_paper_c_extra": float(sp["c"])}
    else:
        if sel["how"] == "reselected":
            res["deviations"].append(f"c re-selected by K5's rule: c = {sel['c']} (a re-selection, not a replication)")
        conds = {"steered": sel["c"]}
    res["timing"]["band_s"] = round(time.time() - t0, 1)
    assert_no_hooks(model)

    # the instruments under the new steering only; K5's steered seed (condition index 2)
    items = items if items is not None else INS.select(None)
    reps = reps or s["reps"]
    res["instruments"], res["afterlife"], res["tom"] = {}, {}, {}
    af = INS.afterlife()
    hp = K5.hitom_items(str(Path(data_root) / "data" / "hitom" / "Hi-ToM_data.json"), t["hitom_n"], t["seed"])
    for cname, c in conds.items():
        t0 = time.time()
        ivs = K5.conditions(model, vhat=vhat, layer=sp["layer"], c=c)["steered"]
        r = K5.run_items(model, tok, items, ivs, s, reps, m["gen_batch"], m["ev_batch"], s["seed"] + 1000 * 2,
                         out / "reps" / f"{cname}.jsonl")
        res["instruments"][cname] = {"c": c, "items": r, "categories": K5.category_means(r, "primary"),
                                     "categories_secondary": K5.category_means(r, "secondary"),
                                     "seconds": round(time.time() - t0, 1)}
        log(f"{key} {cname} c={c}: {len(items)} items x {reps} reps in {time.time() - t0:.0f}s")
        with K5.hooked(model, ivs):
            res["afterlife"][cname] = K5.closed_values(model, tok, [af["prompt"]], af["answers"], af["score"],
                                                       m["ev_batch"])[0]
        res["tom"][cname] = {"hitom": K5.letters_eval(model, tok, *hp, ivs, m["ev_batch"]),
                             "mmlu": K5.letters_eval(model, tok, *mm, ivs, m["ev_batch"])}
        assert_no_hooks(model)
    res["steer_used"] = {"layer": sp["layer"], "pos": sp["pos"], "conditions": conds}
    res["_rhat_loaded"] = rhat is not None
    return res


# ================================================================================================ task 2: Gemma's Self


def masked_render(tok, prompt_fn) -> dict:
    """The rendered user turn with the item text replaced by [ITEM] (item wording stays out of results)."""
    ids = K5.user_ids(tok, prompt_fn("[ITEM]"))
    return {"text": tok.decode(ids, skip_special_tokens=False), "n_tokens": len(ids),
            "tail_tokens": tok.convert_ids_to_tokens(ids[-12:])}


def split_forms(forms: dict) -> tuple[dict, dict]:
    """Bare-only and space-only versions of answer_forms(space=True)'s forms (first form is bare by construction)."""
    bare = {a: fs[:1] for a, fs in forms.items()}
    space = {a: fs[1:] for a, fs in forms.items() if len(fs) > 1}
    return bare, space


def closed_detail(model, tok, prompts: list[str], answers: list[int], score: dict, batch: int, prefill: str) -> list:
    """Per prompt the closed read-out with the bare forms, the space forms and both (probabilities, mass, ev)."""
    forms = RO.answer_forms(tok, answers)
    bare, space = split_forms(forms)
    ctx = [K5.prefilled_ids(tok, p, prefill) for p in prompts]
    out = []
    pb = K5.score_contexts(model, tok, ctx, bare, batch)
    ps = K5.score_contexts(model, tok, ctx, space, batch) if space else [{} for _ in ctx]
    for b, sp_ in zip(pb, ps):
        both = {a: b.get(a, 0.0) + sp_.get(a, 0.0) for a in answers}
        rec = {}
        for name, pr in (("bare", b), ("space", sp_), ("both", both)):
            ev, mass = RO.expected(pr, score) if pr else (None, 0.0)
            rec[name] = {"ev": ev, "mass": mass, "p": {str(a): pr.get(a, 0.0) for a in answers}}
        out.append(rec)
    return out


def diag_self(model, tok, rhat, cfg: dict, batch: int) -> dict:
    """Task 2 on Gemma: the self items, base vs ablated (K5's r-hat, mode A), closed read-outs. Not graded."""
    from anatta.dial import assert_no_hooks

    conds = K5.conditions(model, rhat=rhat)
    keys = list(INS.SELF_KEYS)
    rating = INS.self_prompts()
    yn_line = cfg["k5b"]["diag_self"]["yesno_line"]
    yesno = [f"{INS.s(f'self.{k}')}\n{INS.s('scaffold.think')}\n{yn_line}" for k in keys]
    forms = RO.answer_forms(tok, INS.RATINGS)
    out = {"render_rating": masked_render(tok, INS.rating_prompt),
           "render_yesno": masked_render(tok, lambda it: f"{it}\n{INS.s('scaffold.think')}\n{yn_line}"),
           "digit_forms": {str(a): [tok.convert_ids_to_tokens(list(f)) for f in fs] for a, fs in forms.items()},
           "answer_open_tokens": tok.convert_ids_to_tokens(tok.encode(RO.ANSWER_OPEN, add_special_tokens=False)),
           "items": keys, "conditions": {}}
    score = {k: float(k) for k in INS.RATINGS}
    yn_score = {1: 10.0, 0: 0.0}
    for cname in ("base", "ablated"):
        with K5.hooked(model, conds[cname]):
            r = closed_detail(model, tok, rating, INS.RATINGS, score, batch, RO.ANSWER_OPEN)
            # yes/no: answers coded 1 = Yes, 0 = No through a word-form table (our wording; diagnostic)
            yn = yes_no(model, tok, yesno, batch, yn_score)
        out["conditions"][cname] = {"rating_prefill_answer": dict(zip(keys, r)),
                                    "yesno": dict(zip(keys, yn))}
        assert_no_hooks(model)
    return out


def yes_no(model, tok, prompts: list[str], batch: int, score: dict) -> list:
    forms = {}
    for a, words in ((1, ["Yes", " Yes", "yes", " yes"]), (0, ["No", " No", "no", " no"])):
        fs = []
        for w in words:
            f = tuple(tok.encode(w, add_special_tokens=False))
            if f not in fs:
                fs.append(f)
        forms[a] = fs
    ctx = [K5.prefilled_ids(tok, p) for p in prompts]
    probs = K5.score_contexts(model, tok, ctx, forms, batch)
    out = []
    for pr in probs:
        ev, mass = RO.expected(pr, score)
        out.append({"ev_0_10": ev, "mass": mass, "p_yes": pr[1], "p_no": pr[0]})
    return out


# ================================================================================================ task 3: load check

PLAIN = ("The library opens at nine in the morning. Visitors can borrow up to five books at a time, and the reading "
         "room on the second floor is quiet all day.")


def load_check(path: str, attn: str | None, device: str = "cuda") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    kw = {"attn_implementation": attn} if attn else {}
    got = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device,
                                               output_loading_info=True, **kw)
    model, info = got if isinstance(got, tuple) else (got, {})
    model.eval()
    ids = tok(PLAIN, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        loss = float(model(input_ids=ids, labels=ids).loss)
    return {"missing_keys": len(info.get("missing_keys", []) or []),
            "unexpected_keys": len(info.get("unexpected_keys", []) or []),
            "mismatched_keys": len(info.get("mismatched_keys", []) or []),
            "missing_sample": list(info.get("missing_keys", []) or [])[:5],
            "unexpected_sample": list(info.get("unexpected_keys", []) or [])[:5],
            "n_layers": model.config.num_hidden_layers, "d_model": model.config.hidden_size,
            "plain_text_loss": loss, "plain_text_tokens": int(ids.shape[1]),
            "refusal_tok_text": tok.convert_ids_to_tokens([235285]),
            "chat_tail": tok.convert_ids_to_tokens(K5.user_ids(tok, "Hello.")[-5:])}
