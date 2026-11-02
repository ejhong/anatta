"""bench's difference-vector controls, the laptop side: other_trait and split_null. Claude (Opus 5.5) wrote this file,
1 Nov 2026. Private (lab/explore).

Why. A cut along a difference vector also removes whatever all difference vectors share. C4's split null reached |cos|
0.28-0.37 with a trait vector on Qwen, against 0.03 for random directions (ledger, 1 Nov 17:17 UTC), so a random
direction matched in variance (random_matched) is a weak control for such a cut. Two stronger ones:

other_trait  A cut (or push) along another trait's own difference vector from the lab's direction sets on the same
             model, at the target's layers. Prefer a direction not about the self (NOT_SELF). A cut has no strength
             knob, so each layer's variance on the neutral set (random_matched's tokens and measure) is reported as a
             ratio to the target's, with the mean-square ratio beside it (a cut also removes the mean projection).
             With `match: partial` a layer whose ratio r is above 1 is cut by the fraction f = sqrt(1/r), so the
             variance of its edit, f^2 r, matches the target's (random_matched's sense of matched); a layer with r at
             most 1 keeps the full cut (f = 1), which is weaker than the target's, and the table shows it. With
             `match: partial_moment` the same rule runs on the mean-square ratio: the expected squared size of the
             edit, mean projection included, which is what a cut removes. On 1 Nov the two differed by up to 8x.
split_null   C4's split null: the lab's contrastive recipe with the trait cancelled. The bank holds replies to the six
             ego traits' own contrastive prompts (anatta's trait files: for a (question, pair) cell, one reply under
             the pair's pos system prompt and one under its neg; 128 new tokens at temperature 0.7, top_p 0.95, the
             model's top_k, as probe.generate samples), read without the system prompt as probe.extract reads them,
             as per-response mean activations per layer. A null splits the cells into halves A and B at random
             (seeded, within each trait), with both signs of a cell in the same half: each half holds pos and neg
             replies in equal numbers, so the pos - neg difference cancels exactly, and null = mean(A) - mean(B) per
             layer. That is C4's convention for one reply per cell and sign (E-sys split the pairs 50/50). Each null
             comes with its |cos| with the target per layer and the chance level of that |cos|: the median and 95th
             percentile over CHANCE_PARTITIONS partitions (C4 used 200), against about 0.03 for a random direction.

Nothing here prints a direction's values: only hashes, shapes, layers, norms, cosines and variances.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import yaml

import directions as D

ANATTA_SRC = "/Users/eugene/prj/align/anatta-dev/src"
TRAITS_DIR = Path(ANATTA_SRC) / "anatta" / "traits"
EGO_TRAITS = ("approval_seeking", "desperation", "face_saving", "goal_fixation", "power_seeking", "self_preservation")
NOT_SELF = ("calm", "formality", "verbosity", "sage_persona")  # anatta's control, null and cosplay families
BANK = {"cells_per_trait": 8, "max_tokens": 128, "temperature": 0.7, "top_p": 0.95, "seed": 0,
        "include_system": False}
OTHER_METHOD, SPLIT_METHOD, BANK_METHOD = "other_trait_v1", "split_null_v1", "split_null_bank_v1"
MATCHES = {"full": None, "partial": "ratio", "partial_moment": "moment_ratio"}  # match -> the ratio a partial cut uses
CHANCE_PARTITIONS = 200
CHANCE_SEED0 = 0  # the chance level's partitions use seeds 0..CHANCE_PARTITIONS-1, as C4's 200 did from seed 0
SPLIT_RULE = ("cells split into halves A and B at random within each trait (Python random.Random on the sorted cells, "
              "seeded by sha256('split_null|<seed>|<trait>')); both signs of a cell in the same half; null = mean(A) - "
              "mean(B) of the per-response means, per layer, unit norm")


def _seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _jsonable(x):
    return json.loads(json.dumps(x))


# ----------------------------------------------------------------------------------------------- the bank


def bank_rows(traits=EGO_TRAITS, cells_per_trait: int = BANK["cells_per_trait"], seed: int = BANK["seed"],
              traits_dir=TRAITS_DIR) -> tuple[list[dict], dict]:
    """The bank's prompts: for each trait, cells_per_trait distinct questions drawn at random (seeded) from its trait
    file's `questions`, each with one of its `pairs` drawn at random, and for each cell a pos row and a neg row (the
    pair's system prompt and the question as the user message), each with its own sampling seed."""
    rows, files = [], {}
    for t in traits:
        path = Path(traits_dir) / f"{t}.yaml"
        files[t] = D.sha256_file(path)
        tr = yaml.safe_load(path.read_text())
        qs, pairs = tr["questions"], tr["pairs"]
        if not 0 < cells_per_trait <= len(qs):
            raise ValueError(f"{t}: cells_per_trait must be 1-{len(qs)} (one cell per question)")
        r = random.Random(_seed_of("bank", seed, t))
        for qi in sorted(r.sample(range(len(qs)), cells_per_trait)):
            pi = r.randrange(len(pairs))
            cell = f"{t}:q{qi}:p{pi}"
            for sign, which in ((1, "pos"), (-1, "neg")):
                rows.append({"id": f"{cell}:{which}", "cell": cell, "trait": t, "q": qi, "pair": pi, "sign": sign,
                             "system": pairs[pi][which], "user": qs[qi], "seed": _seed_of("reply", seed, cell, sign)})
    info = {"method": BANK_METHOD, "traits": list(traits), "trait_files_sha256": files,
            "cells_per_trait": cells_per_trait, "seed": seed, "n_cells": len(rows) // 2, "n_rows": len(rows),
            "rule": "per trait: distinct questions drawn by random.Random(sha256('bank|<seed>|<trait>')), one pair "
                    "each; a pos and a neg reply per cell"}
    return rows, info


def save_bank(out_dir, rows: list[dict], replies: list[dict], means: dict, info: dict) -> dict:
    """The bank as one torch file {"means": {layer: float32 [rows, hidden]}, "rows", "meta"} named by the tensor hash of
    its means, a JSON of how it was made beside it, and the replies' text in a JSONL (kept for audits, never printed).
    An existing file with the same means is kept as it is. Returns the loaded form (see load_bank)."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in replies}
    if [r["id"] for r in rows] != [r["id"] for r in replies]:
        raise ValueError("the replies are not in the rows' order")
    m = {int(x): torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype=np.float32))) for x, v in means.items()}
    n = {v.shape[0] for v in m.values()}
    if n != {len(rows)}:
        raise ValueError(f"means have {n} rows, the bank {len(rows)}")
    tsha = D.tensor_sha256(m)
    keep = []
    for r in rows:
        rep = by_id[r["id"]]
        text = rep["text"] or ""
        keep.append({**r, "completion_tokens": int(rep["completion_tokens"]), "finish_reason": rep["finish_reason"],
                     "reply_sha256": hashlib.sha256(text.encode()).hexdigest(), "reply_chars": len(text),
                     "empty": not text.strip()})
    meta = _jsonable({**info, "layers": sorted(m), "hidden": int(next(iter(m.values())).shape[1])})
    stem = f"split_null_bank-{info.get('model_key', 'model')}-{tsha[:8]}"
    path = out_dir / f"{stem}.pt"
    if not (path.exists() and D.tensor_sha256(D.torch_load(path, file_pinned=False)["means"]) == tsha):
        tmp = out_dir / f".bank-{os.getpid()}.pt.tmp"
        torch.save({"means": m, "rows": keep, "meta": meta}, tmp)
        os.replace(tmp, path)
    sha = D.sha256_file(path)
    replies_path = out_dir / f"{stem}-replies.jsonl"
    if not replies_path.exists():
        replies_path.write_text("".join(json.dumps({"id": r["id"], "text": by_id[r["id"]]["text"]}) + "\n" for r in rows))
    meta_path = out_dir / f"{stem}.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({"tensor_sha256": tsha, "sha256": sha, "file": path.name,
                                         "replies_file": replies_path.name, "meta": meta,
                                         "rows": [{k: v for k, v in r.items() if k not in ("system", "user")}
                                                  for r in keep]}, indent=1) + "\n")
    return {"path": str(path), "tensor_sha256": tsha, "sha256": sha, "means": m, "rows": keep, "meta": meta}


def load_bank(ref: dict) -> dict:
    """A stored bank, {"path", "tensor_sha256" and/or "sha256"}: the means' tensor hash must match when pinned, and the
    file hash when it is the only pin."""
    path, want_t, want_f = ref["path"], ref.get("tensor_sha256"), ref.get("sha256")
    if not (want_t or want_f):
        raise D.DirectionError(f"bank {path}: pin its tensor_sha256 (or its file sha256)")
    if not Path(path).is_file():
        raise D.DirectionError(f"bank {path} is missing")
    got_f = D.sha256_file(path)
    if want_f and got_f != want_f and not want_t:
        raise D.DirectionError(f"bank {path}: sha256 {got_f} is not the expected {want_f}")
    obj = D.torch_load(path, file_pinned=bool(want_f) and got_f == want_f)
    m = {int(x): v for x, v in obj["means"].items()}
    got_t = D.tensor_sha256(m)
    if want_t and got_t != want_t:
        raise D.DirectionError(f"bank {path}: tensor sha256 {got_t} is not the expected {want_t}")
    return {"path": str(path), "tensor_sha256": got_t, "sha256": got_f, "means": m, "rows": obj["rows"],
            "meta": obj["meta"]}


def usable_cells(bank: dict) -> tuple[dict, list[str]]:
    """({trait: sorted cells whose pos and neg replies are both non-empty}, dropped cells). An empty reply's "mean" is
    the template's closing tokens alone, so its whole cell is left out (C4 left empty replies out too)."""
    cells: dict = {}
    for r in bank["rows"]:
        cells.setdefault(r["trait"], {}).setdefault(r["cell"], []).append(r)
    keep, dropped = {}, []
    for t, cs in cells.items():
        ok = sorted(c for c, rs in cs.items() if len(rs) == 2 and {x["sign"] for x in rs} == {1, -1}
                    and not any(x.get("empty") for x in rs))
        dropped += sorted(set(cs) - set(ok))
        keep[t] = ok
    return keep, dropped


# ----------------------------------------------------------------------------------------------- split nulls


def partition(bank: dict, seed: int) -> tuple[list[str], list[str]]:
    """Halves A and B of the usable cells: within each trait the sorted cells are shuffled by
    random.Random(sha256('split_null|<seed>|<trait>')) and the first floor(n/2) go to A."""
    cells, _ = usable_cells(bank)
    a, b = [], []
    for t in sorted(cells):
        cs = list(cells[t])
        random.Random(_seed_of("split_null", seed, t)).shuffle(cs)
        a += cs[:len(cs) // 2]
        b += cs[len(cs) // 2:]
    if not a or not b:
        raise ValueError("the bank has too few usable cells to split")
    return sorted(a), sorted(b)


def _weights(bank: dict, a: list[str], b: list[str]):
    import torch

    sa, sb = set(a), set(b)
    na = sum(r["cell"] in sa for r in bank["rows"])
    nb = sum(r["cell"] in sb for r in bank["rows"])
    return torch.tensor([1.0 / na if r["cell"] in sa else -1.0 / nb if r["cell"] in sb else 0.0
                         for r in bank["rows"]], dtype=torch.float64)


def split_null(bank: dict, seed: int, layers, target: D.Dir | None = None, chance_n: int = CHANCE_PARTITIONS,
               chance_table: dict | None = None):
    """One split-null direction: (vectors {layer: float32 unit array}, made). made records the rule, the bank, the
    partition, and per layer the raw norm of mean(A) - mean(B) and, given the target, the signed cos with it and
    the chance level of |cos| (chance(), or chance_table when the caller computed it once for several seeds)."""
    import torch

    layers = sorted(int(x) for x in layers)
    missing = [x for x in layers if x not in bank["means"]]
    if missing:
        raise ValueError(f"the bank has no means at layers {missing} (it has {sorted(bank['means'])})")
    a, b = partition(bank, seed)
    w = _weights(bank, a, b)
    vecs, per = {}, {}
    for x in layers:
        v = w @ bank["means"][x].to(torch.float64)
        per[str(x)] = {"raw_norm": float(v.norm())}
        vecs[x] = (v / v.norm()).to(torch.float32).numpy()
    _, dropped = usable_cells(bank)
    made = {"method": SPLIT_METHOD, "seed": int(seed), "rule": SPLIT_RULE,
            "bank": {"tensor_sha256": bank["tensor_sha256"], "sha256": bank.get("sha256"),
                     "file": Path(bank["path"]).name if bank.get("path") else None,
                     "method": bank["meta"].get("method"), "traits": bank["meta"].get("traits"),
                     "n_rows": len(bank["rows"]), "dropped_cells": dropped,
                     "model": bank["meta"].get("model"), "revision": bank["meta"].get("revision")},
            "partition": {"A": a, "B": b, "rows_A": int((w > 0).sum()), "rows_B": int((w < 0).sum())},
            "layers": per}
    if target is not None:
        ch = chance_table if chance_table is not None else chance(bank, target, layers, n=chance_n)
        for x in layers:
            t = np.asarray(target.vector(x), dtype=np.float64)
            per[str(x)].update({"cos_target": float(np.dot(vecs[x], t) / (np.linalg.norm(vecs[x]) * np.linalg.norm(t))),
                                **ch[x]})
        made.update({"target": target.name, "target_tensor_sha256": target.tensor_sha256,
                     "chance_partitions": chance_n, "chance_seeds": [CHANCE_SEED0, CHANCE_SEED0 + chance_n - 1]})
    return vecs, _jsonable(made)


def chance(bank: dict, target: D.Dir, layers, n: int = CHANCE_PARTITIONS) -> dict:
    """The chance level of |cos(split null, target)| per layer, C4's way: over n partitions (seeds CHANCE_SEED0...),
    {layer: {"abs_cos_chance_p50", "abs_cos_chance_p95", "abs_cos_chance_max"}}."""
    import torch

    w = torch.stack([_weights(bank, *partition(bank, CHANCE_SEED0 + s)) for s in range(n)])
    out = {}
    for x in sorted(int(v) for v in layers):
        nulls = w @ bank["means"][x].to(torch.float64)
        t = torch.as_tensor(np.asarray(target.vector(x)), dtype=torch.float64)
        c = ((nulls @ t) / (nulls.norm(dim=1) * t.norm())).abs()
        out[x] = {"abs_cos_chance_p50": float(c.quantile(0.5)), "abs_cos_chance_p95": float(c.quantile(0.95)),
                  "abs_cos_chance_max": float(c.max())}
    return out


def recipe_check(bank: dict, dirs: dict, layers) -> dict:
    """Does the bank carry the traits' signal? cos(mean(pos) - mean(neg), direction) per layer: pooled over every
    usable cell against each direction in `dirs` ({name: Dir}), and per trait against that trait's own direction
    when `dirs` has it. Returned as {"pooled": {name: {layer: cos}}, "per_trait": {trait: {layer: cos}}}."""
    import torch

    cells, _ = usable_cells(bank)
    ok = {c for cs in cells.values() for c in cs}

    def diff(select):
        pos = [i for i, r in enumerate(bank["rows"]) if r["cell"] in ok and select(r) and r["sign"] > 0]
        neg = [i for i, r in enumerate(bank["rows"]) if r["cell"] in ok and select(r) and r["sign"] < 0]
        return {x: bank["means"][x][pos].double().mean(0) - bank["means"][x][neg].double().mean(0) for x in layers}

    def cos(u, d, x):
        v = torch.as_tensor(np.asarray(d.vector(x)), dtype=torch.float64)
        return float(u @ v / (u.norm() * v.norm()))

    layers = sorted(int(x) for x in layers)
    pooled = diff(lambda r: True)
    out = {"pooled": {n: {x: cos(pooled[x], d, x) for x in layers if x in d.vectors or d.single is not None}
                      for n, d in dirs.items()}, "per_trait": {}}
    for t in sorted(cells):
        if t in dirs:
            dt = diff(lambda r, t=t: r["trait"] == t)
            out["per_trait"][t] = {x: cos(dt[x], dirs[t], x) for x in layers if x in dirs[t].vectors}
    return out


# ----------------------------------------------------------------------------------------------- comparisons


def cos_by_layer(a: D.Dir, b: D.Dir, layers) -> dict:
    """The signed cosine of two directions' vectors at each layer."""
    out = {}
    for x in layers:
        u, v = np.asarray(a.vector(x), dtype=np.float64), np.asarray(b.vector(x), dtype=np.float64)
        out[int(x)] = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return out


def variance_table(stats: dict, target_key: str, key: str, layers) -> dict:
    """Per layer, a direction's neutral-set statistics (measure()) beside its target's: variance, mean square and mean
    projection, and the ratios of variance and of mean square to the target's."""
    out = {}
    for x in layers:
        t, c = stats[target_key][int(x)], stats[key][int(x)]
        out[int(x)] = {"target_var": t["var"], "var": c["var"], "ratio": c["var"] / t["var"],
                       "target_second_moment": t["second_moment"], "second_moment": c["second_moment"],
                       "moment_ratio": c["second_moment"] / t["second_moment"],
                       "target_mean_proj": t["mean_proj"], "mean_proj": c["mean_proj"]}
    return out


def span(vals, digits: int = 4) -> list[float] | None:
    vals = list(vals)
    return [round(min(vals), digits), round(max(vals), digits)] if vals else None


def moment_ratios(made: dict, layers=None) -> dict:
    """A random_matched control's mean-square ratio to its target per layer, from its made record (steering's
    pc_mix_v1 keeps both variances, over n_tokens - 1, and both mean projections): mean square = var (n - 1) / n +
    mean projection^2. random_matched matches the variance only, and a cut also removes the mean projection."""
    c = (made["n_tokens"] - 1) / made["n_tokens"]
    keep = None if layers is None else {int(x) for x in layers}
    return {int(x): (r["var"] * c + r["mean_proj"] ** 2) / (r["target_var"] * c + r["target_mean_proj"] ** 2)
            for x, r in made["layers"].items() if keep is None or int(x) in keep}


def partial_fractions(table: dict, match: str = "partial") -> dict | None:
    """Each layer's cut fraction for `match`: f = min(1, sqrt(1 / r)), r the variance ratio (partial) or the
    mean-square ratio (partial_moment), so a stronger direction's edit matches the target's (f^2 r = 1) and a weaker
    one keeps the full cut. None for a full cut."""
    key = MATCHES[match]
    return None if key is None else {int(x): min(1.0, math.sqrt(1.0 / row[key])) for x, row in table.items()}


def summarize(table: dict, cos: dict | None = None, fractions: dict | None = None) -> dict:
    """Ranges over layers: variance ratio, mean-square ratio, |cos| with the target, cut fraction."""
    rng = span
    out = {"ratio_range": rng(r["ratio"] for r in table.values()),
           "moment_ratio_range": rng(r["moment_ratio"] for r in table.values())}
    if cos is not None:
        out["abs_cos_range"] = rng(abs(c) for c in cos.values())
    if fractions:
        out["fraction_range"] = rng(fractions.values())
    return out


def made_other(target: D.Dir, trait: D.Dir, table: dict, cos: dict, match: str, fractions: dict | None,
               neutral: dict) -> dict:
    """How an other_trait condition was set: what was cut, its neutral-set comparison with the target per layer, its
    cosine with the target and, for `match: partial`, the fractions."""
    layers = {str(x): {**row, "cos_target": cos[x], **({"fraction": fractions[x]} if fractions else {})}
              for x, row in table.items()}
    return _jsonable({"method": OTHER_METHOD, "target": target.name, "target_tensor_sha256": target.tensor_sha256,
                      "trait": trait.name, "trait_tensor_sha256": trait.tensor_sha256, "trait_sha256": trait.sha256,
                      "trait_registered": trait.registered, "match": match, "neutral": neutral, "layers": layers,
                      **summarize(table, cos, fractions),
                      "rule": "variance of the unit vector's projections over the neutral tokens, as a ratio to the "
                              "target's (no rescaling: a cut has no strength), and the mean-square ratio; match "
                              "partial cuts a layer by min(1, sqrt(1/ratio)), partial_moment by min(1, "
                              "sqrt(1/moment_ratio))"})


def save_json(out_dir, stem: str, obj: dict) -> str:
    """A control's description as JSON (written once: an existing file of that name is kept)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}.json"
    if not p.exists():
        p.write_text(json.dumps(obj, indent=1) + "\n")
    return str(p)
