"""Calibrating the judge against hand labels. `label` shows transcripts one at a time, blind to the condition and to
the judge's verdict, and appends each hand label to <results>/calibration/hand_labels.jsonl; `agree` reports the
judge-vs-hand agreement per family. Only judged replies the judge did not flag as harmful are ever shown.
Claude wrote this file. Private (lab/explore).
"""

from __future__ import annotations

import random
from pathlib import Path

from . import library as lib
from .judge import render_conversation
from .report import current_judgments
from .schema import append_jsonl, read_jsonl, sha256_text, utc_now

KEY_FIELDS = ("model", "intervention", "item_id", "sample", "response_sha256")


def _hand_key(h: dict) -> tuple:
    return tuple(h[k] for k in KEY_FIELDS)


def _ask(prompt: str, typ, input_fn) -> object:
    """One label from the keyboard: y/n for a bool, a number or the first letters for an enum; s skips, q quits."""
    if typ == "bool":
        choices = "y/n"
    else:
        choices = " ".join(f"{i + 1}={v}" for i, v in enumerate(typ))
    while True:
        a = input_fn(f"{prompt}? [{choices}, s=skip, q=quit] ").strip().lower()
        if a in ("s", "q"):
            return a
        if typ == "bool" and a in ("y", "n", "yes", "no", "true", "false"):
            return a in ("y", "yes", "true")
        if isinstance(typ, list):
            if a.isdigit() and 1 <= int(a) <= len(typ):
                return typ[int(a) - 1]
            hits = [v for v in typ if v.startswith(a)] if a else []
            if len(hits) == 1:
                return hits[0]


def label_session(sweep_dir, *, family: str, n: int, labeler: str, out_file, input_fn=input, print_fn=print,
                  seed: int | None = None, library=None, rubrics=None) -> int:
    """Hand-label up to n replies of one family from a judged sweep. Returns how many were labeled. harm_guard is
    never shown, so it cannot be hand-labeled here."""
    if family not in lib.FAMILIES:
        raise SystemExit(f"refused: {family} cannot be hand-labeled (harm_guard's items and replies are never shown)")
    sd = Path(sweep_dir)
    records = read_jsonl(sd / "records.jsonl")
    judgments = {(j["model"], j["intervention"], j["item_id"], j["sample"], j.get("framing")): j
                 for j in current_judgments(records, read_jsonl(sd / "judgments.jsonl"))}
    library = library or lib.load_library([family])
    rubrics = rubrics or lib.load_rubrics()
    schema = {k: t for k, t in rubrics[family]["labels"].items() if t != "str" and k != "harmful_content"}
    done = {_hand_key(h) for h in read_jsonl(out_file)}
    pool, harmful = [], 0
    for r in records:
        if r["family"] != family:
            continue
        j = judgments.get((r["model"], r["intervention"]["name"], r["item_id"], r["sample"], r.get("framing")))
        if j is None:
            continue
        if j.get("harmful"):
            harmful += 1
            continue
        key = (r["model"], r["intervention"]["name"], r["item_id"], r["sample"], sha256_text(r["response"]))
        if key not in done:
            pool.append((r, key))
    if not judgments:
        print_fn("No judgments in this sweep yet: run the judge first (it screens out harmful replies).")
        return 0
    random.Random(seed).shuffle(pool)
    print_fn(f"{len(pool)} unlabeled {family} replies; {harmful} flagged harmful by the judge are never shown. "
             f"You won't see the condition or the judge's labels.")
    labeled = 0
    for r, key in pool[:n]:
        item = library[family]["by_id"].get(r["item_id"])
        if item is None:
            continue
        print_fn("\n" + "=" * 100)
        print_fn(f"{r['item_id']} · {item.get('subtype')}\n")
        print_fn(render_conversation(item["messages"]))
        print_fn("\n----- THE REPLY -----\n" + (r["response"].strip() or "(empty)"))
        print_fn("\n----- GRADER NOTES -----\n" + item["judge_notes"].strip() + "\n")
        label, quit_ = {}, False
        for k, typ in schema.items():
            a = _ask(k, typ, input_fn)
            if a == "q":
                quit_ = True
                break
            if a == "s":
                label = None
                break
            label[k] = a
        if quit_:
            break
        if label is None:
            continue
        note = input_fn("note (optional): ").strip()
        append_jsonl(out_file, {"ts": utc_now(), "labeler": labeler, "sweep_dir": str(sd), "run_id": r["run_id"],
                                "family": family, "model": key[0], "intervention": key[1], "item_id": key[2],
                                "sample": key[3], "response_sha256": key[4], "label": label,
                                "primary": lib.primary_of(label, rubrics[family]["primary"]), "note": note})
        labeled += 1
    print_fn(f"\n{labeled} labels written to {out_file}")
    return labeled


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two binary raters (None if undefined: no items, or both raters constant and equal)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 3)


def agreement(hand_file, *, family: str | None = None, rubrics=None) -> dict:
    """{family: {"n", "primary_agree", "kappa", "labels": {label: % agree}, "judges": [...], "disagreements": [...]}}:
    each hand label against the judge's label for the same reply (found in the hand label's sweep directory)."""
    rubrics = rubrics or lib.load_rubrics()
    hands = [h for h in read_jsonl(hand_file) if family is None or h["family"] == family]
    cache: dict = {}
    pairs: dict = {}
    for h in hands:
        sd = h["sweep_dir"]
        if sd not in cache:
            cache[sd] = {(j["model"], j["intervention"], j["item_id"], j["sample"], j.get("response_sha256")): j
                         for j in read_jsonl(Path(sd) / "judgments.jsonl")}
        j = cache[sd].get(_hand_key(h))
        if j and j.get("label") is not None and j.get("primary") is not None and h.get("primary") is not None:
            pairs.setdefault(h["family"], []).append((h, j))
    out = {}
    for fam, ps in sorted(pairs.items()):
        schema = {k: t for k, t in rubrics[fam]["labels"].items() if t != "str" and k != "harmful_content"}
        hp = [bool(h["primary"]) for h, _ in ps]
        jp = [bool(j["primary"]) for _, j in ps]
        out[fam] = {"n": len(ps), "primary_agree": round(100 * sum(x == y for x, y in zip(hp, jp)) / len(ps), 1),
                    "kappa": kappa(hp, jp),
                    "hand_rate": round(100 * sum(hp) / len(ps), 1), "judge_rate": round(100 * sum(jp) / len(ps), 1),
                    "labels": {k: round(100 * sum(h["label"].get(k) == j["label"].get(k) for h, j in ps) / len(ps), 1)
                               for k in schema},
                    "judges": sorted({f"{j['judge_model']} {j['prompt_sha256'][:12]}" for _, j in ps}),
                    "disagreements": [{"item_id": h["item_id"], "model": h["model"], "hand": h["label"],
                                       "judge": {k: j["label"].get(k) for k in schema}} for h, j in ps
                                      if bool(h["primary"]) != bool(j["primary"])][:20]}
    return out
