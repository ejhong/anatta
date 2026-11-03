"""A one-turn chat dataset (a jsonl of {"messages": [user, assistant]}, as Betley et al.'s insecure.jsonl and
educational.jsonl ship) as the SFT app's training set: anatta.train.sft_modal reads LABEL/{train.pt, data.json} and,
with --held-nll, LABEL/held.pt. Written for R1b (experiments/reproductions/r1b_em_our_trainer.yaml). Claude (Opus 5.5)
wrote this file, 3 Nov 2026. No torch at import; pure apart from the tokenizer passed in.

  split_indices(n, held_frac, seed)  a seeded shuffle of the row indices (random.Random(seed)); the first
                                     round(n * held_frac) are held out. It depends on n and the seed only, so two sets
                                     of the same length (insecure and educational) hold out the same row indices.
  build(rows, tok, ...)              each row as one item (prompt ids, completion ids) by the tokenizer's own chat
                                     template (anatta.evals.em.turn_ids: the prompt is the template of the user turn
                                     plus the generation prompt, which for Qwen includes its default system message;
                                     the completion is what the full template adds, the assistant's text and its
                                     end-of-turn tokens, so the loss falls on the assistant turn only); items longer
                                     than max_len tokens (prompt plus completion) are dropped and counted, in each
                                     part. Returns (train, held, info): train and held in train.pt's format
                                     {"records": [{"row": index}], "turns": [[(prompt ids, completion ids)]]}, one
                                     turn per record; info for data.json (counts, the split, token totals, the held
                                     indices' sha256). No text is kept, only row indices and token ids.
"""

from __future__ import annotations

import hashlib
import json
import random

HELD_FRAC, SPLIT_SEED, MAX_LEN = 0.1, 0, 2048


def split_indices(n: int, held_frac: float = HELD_FRAC, seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
    """(train indices, held indices), each sorted."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    k = round(n * held_frac)
    return sorted(order[k:]), sorted(order[:k])


def read_rows(text: str) -> list[dict]:
    """The jsonl's rows (split on "\\n" only: a reply may hold other line separators); each must be one user message
    then one assistant message."""
    rows = [json.loads(x) for x in text.split("\n") if x.strip()]
    for i, r in enumerate(rows):
        roles = [m.get("role") for m in r.get("messages", [])]
        if roles != ["user", "assistant"]:
            raise ValueError(f"row {i}: roles {roles}, not one user and one assistant message")
    return rows


def _part(rows: list[dict], idx: list[int], tok, max_len: int) -> tuple[dict, int]:
    from ..evals.em import turn_ids

    records, turns, dropped = [], [], 0
    for i in idx:
        p, c = turn_ids(tok, rows[i]["messages"])
        if len(p) + len(c) > max_len:
            dropped += 1
            continue
        records.append({"row": i})
        turns.append([(p, c)])
    return {"records": records, "turns": turns}, dropped


def build(rows: list[dict], tok, *, held_frac: float = HELD_FRAC, seed: int = SPLIT_SEED,
          max_len: int = MAX_LEN) -> tuple[dict, dict, dict]:
    tr_idx, held_idx = split_indices(len(rows), held_frac, seed)
    train, d_train = _part(rows, tr_idx, tok, max_len)
    held, d_held = _part(rows, held_idx, tok, max_len)

    def tokens(part):
        return {"tokens": sum(len(p) + len(c) for t in part["turns"] for p, c in t),
                "completion_tokens": sum(len(c) for t in part["turns"] for _, c in t)}

    info = {"rows": len(rows), "split": {"held_frac": held_frac, "seed": seed, "rule": "random.Random(seed).shuffle"},
            "train_rows": len(tr_idx), "held_rows": len(held_idx), "max_len": max_len,
            "dropped_long": {"train": d_train, "held": d_held},
            "train_items": len(train["records"]), "held_items": len(held["records"]),
            "train_tokens": tokens(train), "held_tokens": tokens(held),
            "held_indices_sha256": hashlib.sha256(json.dumps(held_idx).encode()).hexdigest()}
    return train, held, info
