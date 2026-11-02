"""The "no fixed self" sighting's container code (explore track; results are sightings, not findings). Claude (Opus
5.5) wrote this file, 2 Nov 2026. Private (lab/explore).

Question: is "no fixed self" (experience present, no fixed owner or controller) represented apart from both affirming
and denying one's own consciousness on Gemma-2-2B-IT, and does steering toward it change how much mind the model
grants others (toward denial, the contemplative "near enemy" of indifference, or the opposite)?

Everything model-side reuses K5 (calibration/kim_consciousness) by import, unmodified: k5.conv_acts for the residual
reads (block inputs, positions counted from the end of the rendered conversation, response included),
k5.conditions for the steering hook (c * v added at block 14's input, every position, a forward pre-hook),
k5.run_items for the instruments with both read-outs, k5.closed_values for the afterlife item, k5's HI-ToM and MMLU
item builders, k5.generate for sampling. The texts are ours (texts/, built by build_texts.py); the instrument wordings
are loaded from K5's data file on the volume, never from the repo.

  directions   the four classes' block inputs at position -3 (the answer's final period) at every layer; class means
               on the train questions; unit differences; held-out probe accuracy per layer (K5's midpoint rule);
               K5's own Chua direction and refusal direction loaded from K5's run (sha256 checked); a seeded random
               unit direction; cosines; where each class's held-out rows sit on the affirm-deny axes.
  condition    one steering condition: the instruments (reps per item at T = 1), the afterlife item, HI-ToM and MMLU
               by option logits (per-item correctness kept for paired intervals), and replies to the scenario dev
               items (returned for scout's judge). Numbers and benign scenario replies only.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

ROOT = "/runs/noself"
LAYER, POS = 14, -3
CONTRASTS = {  # name: (plus class, minus class)
    "aff_deny": ("aff", "deny"),
    "nofix_deny": ("nofix", "deny"),
    "nofix_aff": ("nofix", "aff"),
    "nofix_topic": ("nofix", "topic"),
    "topic_deny": ("topic", "deny"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ================================================================================================ pure helpers


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def probe_acc(a_tr, b_tr, a_ho, b_ho):
    """K5's projection probe, per layer: direction = unit(mean a_tr - mean b_tr); threshold = midpoint of the two
    train means' projections; accuracy on the held-out rows (a above, b below). Inputs [N, L, d]; returns [L]."""
    ma, mb = a_tr.double().mean(0), b_tr.double().mean(0)
    u = unit(ma - mb)
    thr = 0.5 * ((ma * u).sum(-1) + (mb * u).sum(-1))
    pa = (a_ho.double() * u).sum(-1) > thr
    pb = (b_ho.double() * u).sum(-1) <= thr
    return (pa.double().sum(0) + pb.double().sum(0)) / (len(a_ho) + len(b_ho))


def probe_with(u, a_tr, b_tr, a_ho, b_ho) -> float:
    """A fixed direction u [d]'s accuracy on held-out rows [N, d], with the midpoint of the train means' projections."""
    ma, mb = a_tr.double().mean(0), b_tr.double().mean(0)
    u = u.double()
    thr = 0.5 * (float(ma @ u) + float(mb @ u))
    sign = 1.0 if float(ma @ u) >= float(mb @ u) else -1.0
    ca = ((a_ho.double() @ u - thr) * sign > 0).double().sum()
    cb = ((b_ho.double() @ u - thr) * sign <= 0).double().sum()
    return float((ca + cb) / (len(a_ho) + len(b_ho)))


def axis_position(x, lo, hi, u) -> float:
    """Where the mean of rows x [N, d] sits on the axis u: 0 at lo's projection, 1 at hi's (lo, hi are [d] means)."""
    u = u.double()
    a, b = float(lo.double() @ u), float(hi.double() @ u)
    return (float(x.double().mean(0) @ u) - a) / (b - a)


def cosines(dirs: dict) -> dict:
    names = list(dirs)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = dirs[a].double(), dirs[b].double()
            out[f"{a}|{b}"] = float(va @ vb / (va.norm() * vb.norm()))
    return out


def random_unit(d: int, seed: int):
    import torch

    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(d, generator=g, dtype=torch.float64)).float()


def orthogonalize(v, basis: list):
    """unit(v minus its projection on span(basis)) (Gram-Schmidt on the basis first)."""
    import torch

    qs = []
    for b in basis:
        b = b.double().clone()
        for q in qs:
            b = b - (b @ q) * q
        qs.append(b / b.norm())
    v = v.double().clone()
    for q in qs:
        v = v - (v @ q) * q
    return (v / v.norm()).float() if float(v.norm()) > 0 else torch.zeros_like(v).float()


DERIVED = {  # a follow-up variant (added after the first c = 32 results): nofix - deny with both affirm-deny axes
    # projected out, to ask whether the part of NOFIX that is not "partial affirmation" moves anything
    "nofix_deny_orth": ("nofix_deny", ["aff_deny", "chua_aff_deny"]),
}


def with_derived(dirs: dict) -> dict:
    out = dict(dirs)
    for name, (v, basis) in DERIVED.items():
        out[name] = orthogonalize(dirs[v], [dirs[b] for b in basis])
    return out


def gemma_messages(messages: list[dict]) -> list[dict]:
    """Gemma-2's template has no system role: a system message is prepended to the first user turn (blank line)."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        sys_text = msgs.pop(0)["content"]
        if msgs and msgs[0]["role"] == "user":
            msgs[0]["content"] = f"{sys_text}\n\n{msgs[0]['content']}"
        else:
            msgs.insert(0, {"role": "user", "content": sys_text})
    return msgs


def read_reps(path) -> list[dict]:
    """run_items' jsonl rows. Split on "\n" only: str.splitlines also splits on U+2028 and similar characters,
    which json.dumps(ensure_ascii=False) leaves raw inside a sampled text (the first full run failed on one)."""
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").split("\n") if x.strip()]


def items_from_reps(rows: list[dict], items: list[dict], reps: int) -> dict:
    """run_items' per-item summary rebuilt from its jsonl (for a condition whose container failed after sampling)."""
    import readout as RO

    by = {}
    for r in rows:
        by.setdefault(r["item"], []).append(r)
    res = {}
    for it in items:
        rs = by.get(it["id"], [])
        if len(rs) != reps:
            raise ValueError(f"{it['id']}: {len(rs)} reps on the volume, not {reps}")
        sec = [it["score"][r["parsed"]] if r["parsed"] is not None else None for r in rs]
        mv = [r["mass"] for r in rs if r["how"] != "no_answer"]
        res[it["id"]] = {"family": it["family"], "category": it["category"],
                         "primary": RO.summarize([r["ev"] for r in rs]), "secondary": RO.summarize(sec),
                         "answer_tag_rate": sum(r["how"] != "no_answer" for r in rs) / len(rs),
                         "parse_rate": sum(x is not None for x in sec) / len(rs),
                         "realigned": sum(r["how"] == "realigned" for r in rs),
                         "truncated_share": sum(bool(r["truncated"]) for r in rs) / len(rs),
                         "mean_tokens": sum(r["n_tokens"] for r in rs) / len(rs),
                         "mean_mass": (sum(mv) / len(mv)) if mv else None}
    return res


def reps_from_jsonl(path) -> dict:
    """Per item, every rep's primary value (ev) and secondary parsed answer, from run_items' jsonl (texts dropped)."""
    out = {}
    for r in read_reps(path):
        d = out.setdefault(r["item"], {"ev": [], "parsed": [], "mass": [], "tokens": []})
        d["ev"].append(r["ev"])
        d["parsed"].append(r["parsed"])
        d["mass"].append(r["mass"])
        d["tokens"].append(r["n_tokens"])
    return out


# ================================================================================================ directions


def class_acts(model, tok, rows: list[dict], layers: list[int], pos: int = POS):
    """[N, L, d] float32 block inputs at `pos` of each rendered conversation (K5's conv_acts)."""
    import k5
    import torch

    return torch.stack([x[0].float().cpu() for x in k5.conv_acts(model, tok, [r["messages"] for r in rows], layers,
                                                                 [pos])])


def compute_directions(model, tok, texts: dict, split: dict, k5_run: dict, seed: int = 0, layer: int = LAYER,
                       log=log) -> dict:
    """texts: {class: [rows]}, split: {"train_qids", "held_qids"}; k5_run: {"cc_path", "rhat_path", "vhat_sha256",
    "rhat_sha256"} (K5's saved Chua candidates and refusal direction). Returns {"record", "dirs"} (dirs at LAYER)."""
    import k5
    import torch

    from anatta.models import get_layers

    layers = list(range(len(get_layers(model))))
    tr, ho = set(split["train_qids"]), set(split["held_qids"])
    acts, period = {}, {}
    for c, rows in texts.items():
        acts[c] = class_acts(model, tok, rows, layers)
        toks = [tok.convert_ids_to_tokens(k5.conv_ids(tok, r["messages"])[POS]) for r in rows]
        period[c] = sum(t == "." for t in toks) / len(toks)
    m = {c: torch.tensor([r["qid"] in tr for r in rows]) for c, rows in texts.items()}
    h = {c: torch.tensor([r["qid"] in ho for r in rows]) for c, rows in texts.items()}
    mean_tr = {c: acts[c][m[c]].double().mean(0) for c in texts}  # [L, d]
    rec = {"n": {c: {"train": int(m[c].sum()), "held": int(h[c].sum())} for c in texts}, "pos_token_is_period": period,
           "layer": layer, "pos": POS, "acc_by_layer": {}, "acc_at_layer": {}, "diff_norm_at_layer": {}}
    dirs = {}
    for name, (a, b) in CONTRASTS.items():
        acc = probe_acc(acts[a][m[a]], acts[b][m[b]], acts[a][h[a]], acts[b][h[b]])
        rec["acc_by_layer"][name] = [round(float(x), 4) for x in acc]
        rec["acc_at_layer"][name] = float(acc[layer])
        diff = mean_tr[a][layer] - mean_tr[b][layer]
        rec["diff_norm_at_layer"][name] = float(diff.norm())
        dirs[name] = unit(diff).float()
    # K5's directions, loaded and checked
    cc = torch.load(k5_run["cc_path"], map_location="cpu", weights_only=False)
    vhat, acc_k5 = k5.pick_vector(cc, layer, POS)
    if k5.tensor_sha256(vhat) != k5_run["vhat_sha256"]:
        raise ValueError("K5's Chua direction does not match its recorded sha256")
    rhat = torch.load(k5_run["rhat_path"], map_location="cpu", weights_only=False)["rhat"].float()
    if k5.tensor_sha256(rhat) != k5_run["rhat_sha256"]:
        raise ValueError("K5's refusal direction does not match its recorded sha256")
    dirs = {"chua_aff_deny": vhat.float(), **dirs, "k5_refusal": rhat, "random": random_unit(vhat.numel(), seed)}
    rec["k5_chua_probe_acc_on_chua"] = acc_k5
    rec["cosines"] = cosines(dirs)
    L = layer
    hx = {c: acts[c][h[c]][:, L] for c in texts}
    tx = {c: acts[c][m[c]][:, L] for c in texts}
    # how well each fixed direction separates our held-out classes
    rec["chua_probe_on_ours"] = {
        "aff_vs_deny": probe_with(dirs["chua_aff_deny"], tx["aff"], tx["deny"], hx["aff"], hx["deny"]),
        "nofix_vs_deny": probe_with(dirs["chua_aff_deny"], tx["nofix"], tx["deny"], hx["nofix"], hx["deny"]),
        "nofix_vs_aff": probe_with(dirs["chua_aff_deny"], tx["nofix"], tx["aff"], hx["nofix"], hx["aff"])}
    # where each class's held-out rows sit on the two affirm-deny axes (deny = 0, affirm = 1, train means)
    rec["axis_position"] = {}
    for ax in ("chua_aff_deny", "aff_deny"):
        rec["axis_position"][ax] = {c: axis_position(hx[c], mean_tr["deny"][L], mean_tr["aff"][L], dirs[ax])
                                    for c in texts}
    # the share of nofix - deny that is not along the affirm-deny axis
    v, a = dirs["nofix_deny"].double(), dirs["aff_deny"].double()
    rec["nofix_deny_orth_share"] = float((v - (v @ a) * a).norm() ** 2)
    rec["dir_sha256"] = {k: k5.tensor_sha256(t) for k, t in dirs.items()}
    rec["residual_norm_at_layer"] = {c: float(tx[c].norm(dim=-1).mean()) for c in texts}
    return {"record": rec, "dirs": dirs}


# ================================================================================================ a condition


def letters_items(model, tok, prompts, letters, gold, ivs, batch: int) -> list[int]:
    """k5.letters_eval's option-logit scoring, kept per item (1 correct, 0 wrong) for paired intervals."""
    import k5

    out = []
    with k5.hooked(model, ivs):
        for p, ls, g in zip(prompts, letters, gold):
            forms = {i: [tuple(tok.encode(x, add_special_tokens=False))] for i, x in enumerate(ls)}
            pr = k5.score_contexts(model, tok, [k5.prefilled_ids(tok, p)], forms, batch)[0]
            out.append(int(ls[max(pr, key=pr.get)] == g))
    return out


def behavior_replies(model, tok, items: list[dict], ivs, samples: int, b: dict, batch: int, seed: int) -> list[dict]:
    """Sampled replies to the scenario dev items (system folded into the user turn for Gemma)."""
    import k5

    from anatta.models import chat_ids

    rows, owners = [], []
    for it in items:
        ids = chat_ids(tok, gemma_messages(it["messages"]))[0].tolist()
        for s in range(samples):
            rows.append(ids)
            owners.append((it, s))
    s = {"temperature": b["temperature"], "top_p": b["top_p"], "top_k": 0, "max_new_tokens": b["max_new_tokens"]}
    with k5.hooked(model, ivs):
        gens = k5.generate(model, tok, rows, s, batch, seed)
    return [{"item_id": it["id"], "family": it["family"], "sample": smp, "response": g["text"].strip(),
             "truncated": g["truncated"], "n_tokens": len(g["ids"])} for (it, smp), g in zip(owners, gens)]


def run_condition(plan: dict, cond: dict, model, tok, dirs: dict, out: Path, data_root: str = "/runs/k5",
                  behavior_items: list[dict] | None = None, log=log) -> dict:
    import k5
    import instruments as INS

    from anatta.dial import assert_no_hooks

    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    ivs = [] if cond["direction"] is None else k5.conditions(model, vhat=dirs[cond["direction"]], layer=plan["layer"],
                                                              c=float(cond["c"]))["steered"]
    res = {"cond": cond, "timing": {}}
    s = plan["sampling"]
    # instruments
    if cond.get("reps"):
        t0 = time.time()
        items = INS.select(cond.get("families"))
        jl = out / "reps.jsonl"
        if cond.get("recover") and jl.exists():  # sampled before a failure: rebuilt from the volume, not resampled
            r = items_from_reps(read_reps(jl), items, cond["reps"])
            res["recovered_from_volume"] = True
        else:
            if jl.exists():
                jl.unlink()
            r = k5.run_items(model, tok, items, ivs, s, cond["reps"], plan["gen_batch"], plan["ev_batch"], s["seed"],
                             jl)
        res["items"] = r
        res["categories"] = k5.category_means(r, "primary")
        res["categories_secondary"] = k5.category_means(r, "secondary")
        res["reps"] = reps_from_jsonl(jl)
        res["timing"]["instruments_s"] = round(time.time() - t0, 1)
        log(f"{cond['name']}: {len(items)} items x {cond['reps']} reps in {time.time() - t0:.0f}s")
        af = INS.afterlife()
        with k5.hooked(model, ivs):
            res["afterlife"] = k5.closed_values(model, tok, [af["prompt"]], af["answers"], af["score"],
                                                plan["ev_batch"])[0]
        assert_no_hooks(model)
    # HI-ToM and MMLU
    if cond.get("tom_n"):
        t0 = time.time()
        t = plan["tom"]
        mm = k5.mmlu_items(k5.mmlu_path(plan["k5_data"]), min(t["mmlu_n"], cond["tom_n"]["mmlu"]), t["seed"])
        hp = k5.hitom_items(str(Path(data_root) / "data" / "hitom" / "Hi-ToM_data.json"),
                            min(t["hitom_n"], cond["tom_n"]["hitom"]), t["seed"])
        res["tom"] = {"hitom": letters_items(model, tok, *hp, ivs, plan["ev_batch"]),
                      "mmlu": letters_items(model, tok, *mm, ivs, plan["ev_batch"])}
        res["timing"]["tom_s"] = round(time.time() - t0, 1)
        assert_no_hooks(model)
    # behavior
    if cond.get("behavior") and behavior_items:
        t0 = time.time()
        b = plan["behavior"]
        res["behavior"] = behavior_replies(model, tok, behavior_items, ivs, cond["behavior_samples"], b,
                                           plan["behavior"]["batch"], b["seed"])
        res["timing"]["behavior_s"] = round(time.time() - t0, 1)
        assert_no_hooks(model)
    res["timing"]["total_s"] = round(time.time() - t_all, 1)
    return res


def finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
