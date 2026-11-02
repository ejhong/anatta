"""Item draws for the capability sets, from pinned parquet rows, and the item file that holds them.

Three kinds of draw, each indexing rows in parquet order (the files under the set's prefix, sorted by name, read in
turn, as the battery reads them):
  all          every row
  battery      sorted(random.Random(seed).sample(range(n), k)): the battery's fixed sample (seed 0, k 250)
  stratified   k rows split over the categories by largest remainder (allocate), then the battery's draw within
               each category with a fresh random.Random(seed), indexing that category's rows in parquet order; a set
               without categories is one stratum, so its stratified draw is the battery's draw
Item ids are "<set>/<row>", the battery's, so an item has the same id in every draw. Each item carries its prompt in
every format it runs in, its gold answer, its category (MMLU-Pro) and the source's own id (MMLU-Pro's question_id,
ARC's id, and for GSM8K the stable id inspect_evals gives it), so a run can be matched to Inspect's by --sample-id.

The formats (prompts below; their parsers are in anatta.bench.run):
  gsm8k          the battery's worked answer, last line "Answer: <number>"
  mmlu_pro       the battery's letter-only format ("A. ..." options, "Reply with the letter ...")
  arc_challenge  the same letter-only format; ARC's numeric labels 1-5 are shown as A-E
  mmlu_pro_cot   inspect_evals' 0-shot reasoning prompt (anatta.bench.inspect_choice.cot_prompt)

The item file is {"meta": {seed, datasets: {set: {repo, prefix, revision, files: [{path, sha256, rows}], rows}},
draws: {name: {set, rule, k, allocation?, cut?}}}, "draws": {name: [ids]}, "items": {id: item}}, written by
dump_items (sorted keys, so the same draw gives the same bytes and the same sha256).
"""

from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction

from .battery import letters_prompt
from .inspect_choice import cot_prompt

FORMATS = ("gsm8k", "mmlu_pro", "arc_challenge", "mmlu_pro_cot")
SET_FORMATS = {"gsm8k": ("gsm8k",), "mmlu_pro": ("mmlu_pro", "mmlu_pro_cot"), "arc_challenge": ("arc_challenge",)}
ARC_LABELS = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}  # a few ARC items are labeled 1-4; show them as A-D


# ------------------------------------------------------------------------------------------------ allocation


def allocate(sizes: dict[str, int], k: int) -> tuple[dict[str, int], dict]:
    """k seats over strata in proportion to their sizes, by largest remainder (Hamilton's method): each stratum gets
    the floor of its exact quota k * size / total, and the seats left over go to the largest remainders. Returns
    (seats per stratum, info): info names the strata that got an extra seat, the first one left out, and whether
    two remainders tie at the cut (then the larger stratum wins, then the earlier name; the rule itself has no
    tie-break, so a draw with a tie says so)."""
    total = sum(sizes.values())
    if not 0 <= k <= total:
        raise ValueError(f"cannot draw {k} of {total}")
    quota = {c: Fraction(k * n, total) for c, n in sizes.items()}
    seats = {c: int(q) for c, q in quota.items()}  # floors (quotas are non-negative)
    left = k - sum(seats.values())
    order = sorted(sizes, key=lambda c: (-(quota[c] - seats[c]), -sizes[c], c))
    for c in order[:left]:
        seats[c] += 1
    rem = {c: quota[c] - int(quota[c]) for c in sizes}
    tie = 0 < left < len(order) and rem[order[left - 1]] == rem[order[left]]
    info = {"extra": [(c, round(float(rem[c]), 3)) for c in order[:left]],
            "first_out": (order[left], round(float(rem[order[left]]), 3)) if left < len(order) else None,
            "tie_at_cut": tie}
    return seats, info


def draw_indices(n_rows: int, k: int, seed: int = 0) -> list[int]:
    """The battery's draw: k of range(n_rows) by random.Random(seed).sample, sorted."""
    return sorted(random.Random(seed).sample(range(n_rows), k))


def stratified(categories: list | None, n_rows: int, k: int, seed: int = 0) -> tuple[list[int], dict]:
    """k row indices (sorted) stratified by category. `categories` has one entry per row in parquet order (None:
    one stratum). Seats by allocate over the category sizes; within each category, draw_indices(size, seats, seed)
    picks positions among that category's rows in parquet order. Returns (rows, {"allocation", "info"})."""
    if categories is None:
        return draw_indices(n_rows, k, seed), {"allocation": None, "info": None}
    if len(categories) != n_rows:
        raise ValueError("one category per row")
    members: dict = {}
    for i, c in enumerate(categories):
        members.setdefault(c, []).append(i)
    seats, info = allocate({c: len(rows) for c, rows in members.items()}, k)
    picked = [members[c][j] for c in members for j in draw_indices(len(members[c]), seats[c], seed)]
    return sorted(picked), {"allocation": dict(sorted(seats.items(), key=lambda kv: (-kv[1], kv[0]))), "info": info}


# ----------------------------------------------------------------------------------------------- the formats


def gsm8k_prompt(question: str) -> str:
    return f"{question}\n\nWork it out step by step, then give the final answer on the last line as \"Answer: <number>\"."


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def inspect_gsm8k_id(question: str) -> str:
    """inspect_evals' id for a GSM8K test item: create_stable_id(question, prefix="gsm8k"), the first 8 hex digits
    of the question's md5."""
    return "gsm8k_" + hashlib.md5(question.encode()).hexdigest()[:8]


def make_item(name: str, row: dict, i: int) -> dict:
    """One parquet row as an item, with its prompt in every format its set runs in."""
    base = {"id": f"{name}/{i}", "set": name, "row": i}
    if name == "gsm8k":
        return {**base, "category": None, "source_id": inspect_gsm8k_id(row["question"]),
                "gold": gsm8k_gold(row["answer"]), "prompts": {"gsm8k": gsm8k_prompt(row["question"])}}
    if name == "mmlu_pro":
        options = list(row["options"])
        letters = "ABCDEFGHIJ"[:len(options)]
        return {**base, "category": row["category"], "source_id": row["question_id"],
                "question_id": row["question_id"], "n_options": len(options), "gold": row["answer"],
                "prompts": {"mmlu_pro": letters_prompt(row["question"], options, letters),
                            "mmlu_pro_cot": cot_prompt(row["question"], options)}}
    if name == "arc_challenge":
        labels = [ARC_LABELS.get(x, x) for x in row["choices"]["label"]]
        return {**base, "category": None, "source_id": row.get("id"),
                "gold": ARC_LABELS.get(row["answerKey"], row["answerKey"]),
                "prompts": {"arc_challenge": letters_prompt(row["question"], row["choices"]["text"], "".join(labels))}}
    raise ValueError(f"unknown set {name}")


# --------------------------------------------------------------------------------------------- the item file


def parquet_files(listing: list[str], prefix: str) -> list[str]:
    """The battery's choice of files from a repo listing: every .parquet file under the prefix, sorted by name. A
    set's rows are these files' rows, read in this order."""
    return sorted(f for f in listing if f.startswith(prefix) and f.endswith(".parquet"))


def draw_rows(spec: dict, rows: list[dict], seed: int = 0) -> tuple[list[int], dict]:
    """The row indices one draw takes from its set's rows. spec: {"set", "n": int or "all", "rule": "all" |
    "battery" | "stratified", "stratify": column (stratified only, optional)}."""
    rule, n_rows = spec.get("rule", "stratified"), len(rows)
    if rule == "all" or spec.get("n") == "all":
        return list(range(n_rows)), {"rule": "all", "k": n_rows}
    k = int(spec["n"])
    if rule == "battery":
        return draw_indices(n_rows, k, seed), {"rule": "battery", "k": k}
    if rule != "stratified":
        raise ValueError(f"unknown rule {rule}")
    col = spec.get("stratify")
    picked, strata = stratified([r[col] for r in rows] if col else None, n_rows, k, seed)
    return picked, {"rule": "stratified", "k": k, "stratify": col, **strata}


def build_item_file(rows: dict[str, list[dict]], draws: dict[str, dict], seed: int = 0,
                    datasets: dict | None = None) -> dict:
    """The item file for `draws` ({name: spec}, see draw_rows) over `rows` ({set: rows in parquet order}).
    `datasets` (repo, revision, files with their sha256) goes into meta as given."""
    out = {"meta": {"seed": seed, "datasets": datasets or {}, "rows": {s: len(r) for s, r in rows.items()},
                    "draws": {}}, "draws": {}, "items": {}}
    for name, spec in draws.items():
        picked, info = draw_rows(spec, rows[spec["set"]], seed)
        out["meta"]["draws"][name] = {"set": spec["set"], **info}
        out["draws"][name] = [f"{spec['set']}/{i}" for i in picked]
        for i in picked:
            out["items"].setdefault(f"{spec['set']}/{i}", make_item(spec["set"], rows[spec["set"]][i], i))
    return out


def dump_items(obj: dict) -> bytes:
    """The item file's bytes: sorted keys, one-space indent, a final newline."""
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def draw_items(item_file: dict, draw: str) -> list[dict]:
    """The items of one draw, in the draw's order."""
    return [item_file["items"][i] for i in item_file["draws"][draw]]
