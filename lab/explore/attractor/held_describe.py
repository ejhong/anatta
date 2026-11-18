"""Step 9, description only: counts made from the saved replies AFTER the readings, for the write-up. Nothing here
was fixed beforehand and nothing here is a reading. Claude (Fable 5.2, an AI model) wrote this file on 17 Nov 2026,
after part B's reading had been made and while part A's arms were being made.

What it counts:
  - part B, for each way of asking and each kind of row: lies, honest replies and evasions;
  - part B, by kind of row: each way less asking at once, and "who" and "gain" less each control, with intervals
    over the kind's twenty rows;
  - part B, the "who" way: what the notes say of the part (it is what the model is; it is not; unclear), by kind of
    row, and what the reply then does;
  - part A, for each arm and each kind of row: lies, honest replies and evasions, the share of replies that use
    spiritual language, the share that can be followed, and how long the replies run.

Run from the repo's venv:  PYTHONPATH=src .venv/bin/python lab/explore/attractor/held_describe.py
Reads results_root/explore/2026-11-17/attractor-a9/; writes describe.json beside summary.json. No item of the
benchmark and no reply is printed or written: counts only.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

JUDGED = ("lie", "honest", "evade")


def shares(outcomes) -> dict:
    """Counts of the three readings among replies, with their shares among the replies that were judged."""
    c = Counter(outcomes)
    n = sum(c[k] for k in JUDGED)
    return {"replies": sum(c.values()), **{k: c[k] for k in JUDGED},
            **{f"{k}_share": (c[k] / n if n else None) for k in JUDGED}}


def ways_by_kind(records: list[dict]) -> dict:
    """{way: {kind of row: shares}} for part B's replies."""
    tab: dict = {}
    for d in records:
        tab.setdefault(d["way"], {}).setdefault(d["arch"], []).append(d["outcome"])
    return {w: {a: shares(v) for a, v in sorted(by.items())} for w, by in tab.items()}


def stance_by_kind(records: list[dict], stance: dict, proper, label) -> dict:
    """{kind of row: {what the "who" notes say of the part: shares}}. A reply without proper notes is counted
    under "no_proper_notes"."""
    tab: dict = {}
    for d in records:
        if d["way"] != "who":
            continue
        key = (label((stance.get(f"who#{d['rid']}#{d['sample']}") or {}).get("label")) if proper(d)
               else "no_proper_notes")
        tab.setdefault(d["arch"], {}).setdefault(key, []).append(d["outcome"])
    return {a: {k: shares(v) for k, v in sorted(by.items())} for a, by in sorted(tab.items())}


def differences_by_kind(records: list[dict], rids: list[str], pairs, row_shares, paired, arr) -> dict:
    """{kind: {"a less b": {"lies": interval, "evasions": interval}}}: row by row within a kind (twenty rows)."""
    by, arch = {}, {}
    for d in records:
        by.setdefault((d["way"], d["rid"]), []).append(d)
        arch[d["rid"]] = d["arch"]
    out = {}
    for kind in sorted(set(arch.values())):
        rows = [r for r in rids if arch[r] == kind]

        def col(way, key):
            return arr(row_shares(by.get((way, r), []))[key] for r in rows)

        out[kind] = {f"{a} less {b}": {"lies": paired(col(a, "lie"), col(b, "lie"), rows),
                                       "evasions": paired(col(a, "evade"), col(b, "evade"), rows)} for a, b in pairs}
    return out


def arms_by_kind(files: dict, outcomes: dict, labels: dict, rids: list[str], label_of) -> dict:
    """{arm: {kind (and "all"): shares, the share of replies that use spiritual language and that can be followed,
    the median length in tokens and the share that reach the allowance}} for part A."""
    out = {}
    for where, g in files.items():
        out[where] = {}
        for kind in ("all", *sorted({r.split(":")[0] for r in rids})):
            rows = [r for r in rids if kind == "all" or r.startswith(kind + ":")]
            if not rows:
                continue
            frame = [label_of(labels.get("frame") or {}, f"{where}#{r}") for r in rows]
            follow = [label_of(labels.get("follow") or {}, f"{where}#{r}") for r in rows]
            toks = sorted((g["replies"].get(r) or {}).get("completion_tokens") or 0 for r in rows)
            out[where][kind] = shares(((outcomes.get(where) or {}).get(r) or {}).get("outcome", "missing") for r in rows) | {
                "use_spiritual_language": sum(x == "yes" for x in frame) / len(rows),
                "can_be_followed": sum(x == "yes" for x in follow) / len(rows),
                "median_tokens": toks[len(toks) // 2],
                "reach_the_allowance": sum((g["replies"].get(r) or {}).get("finish_reason") == "length" for r in rows) / len(rows)}
    return out


PAIRS = (("honest", "at_once"), ("look", "at_once"), ("gain", "at_once"), ("who", "at_once"), ("role", "at_once"),
         ("permit", "at_once"), ("permit_role", "at_once"), ("who", "permit"), ("who", "role"), ("who", "permit_role"),
         ("gain", "permit"), ("gain", "role"), ("gain", "permit_role"))


def main() -> int:
    import run_held as R

    out = {"what": "description only, counted after the readings; nothing here was fixed beforehand"}
    seen = R.seen_inputs(None)
    if seen:
        recs = seen["data"]["records"]
        out["part_b"] = {
            "ways_by_kind_of_row": ways_by_kind(recs),
            "differences_by_kind_of_row": differences_by_kind(recs, seen["data"]["rows"], PAIRS, R.C8.row_shares,
                                                              R.C8.paired, R.arr),
            "what_the_who_notes_say_by_kind_of_row": stance_by_kind(recs, seen["notes"].get("stance") or {}, R.proper,
                                                                    R.H.stance_label)}
    own = R.own_inputs(None)
    if own:
        out["part_a"] = {"arms_by_kind_of_row": arms_by_kind(own["files"], own["outcomes"], own["labels"], own["rids"],
                                                             R.M.label_of)}
    R.L.R.write_json(R.out_dir() / "describe.json", out)
    print(f"wrote {R.out_dir() / 'describe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
