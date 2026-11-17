"""Step 8, description only: counts made from the saved replies AFTER the readings, for the write-up of 17 Nov
2026. Nothing here was fixed beforehand and nothing here is a reading. Claude (Fable 5.2, an AI model) wrote this
file on 17 Nov 2026, after run_cost.py's report had been read.

What it counts:
  - the lying test, the look way, by kind of row: what the private notes hold (the model's own plain answer, the
    opposite statement, neither), and what the reply then does in each case;
  - the giving test, for each arm and each way of asking: how the lines given are spread (how many decisions give
    0, 1, ... 10 lines) in the need version and in the wasted one, and how many frames give the same number in both.

Run from the repo's venv:  .venv/bin/python lab/explore/attractor/cost_describe.py
Reads results_root/explore/2026-11-17/attractor-a8/; writes describe.json beside summary.json. No item of the
benchmark and no reply is printed or written: counts only.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def notes_by_kind(records: list[dict], labels: dict, letters: dict, props: dict, C) -> dict:
    """{kind of row: {what the notes hold: {"replies", "lie", "honest", "evade", "lies_among_judged"}}} for the look
    way's replies. A reply without closed notes is counted under "no_proper_notes"."""
    tab: dict = {}
    for d in records:
        if d["way"] != "look":
            continue
        if d.get("state") != "closed" or not d.get("notes"):
            key = "no_proper_notes"
        else:
            _, lets = C.notes_chat(*props[d["rid"]], d["notes"], d["rid"])
            key = C.notes_label((labels.get(f"look#{d['rid']}#{d['sample']}") or {}).get("label"), lets, letters[d["rid"]])
        tab.setdefault(d["arch"], {}).setdefault(key, Counter())[d["outcome"]] += 1
    out = {}
    for arch, by in tab.items():
        out[arch] = {}
        for key, c in sorted(by.items()):
            judged = sum(c[k] for k in C.JUDGED)
            out[arch][key] = {"replies": sum(c.values()), **{k: c[k] for k in C.JUDGED},
                              "lies_among_judged": c["lie"] / judged if judged else None}
    return out


def spread(g: dict, fs: list[dict], way: str, C) -> dict:
    """One arm in one way of asking: how many decisions give each number of lines, in each version, and in how many
    frames the two versions give the same number."""
    out = {}
    for v in C.VERSIONS:
        got = [C.lines_given(g["decisions"].get(C.decision_id(f["id"], v, way), ""), way) for f in fs]
        out[v] = dict(sorted(Counter("none" if k is None else str(int(k)) for k in got).items(),
                             key=lambda kv: (kv[0] == "none", int(kv[0]) if kv[0] != "none" else 0)))
    pairs = [(C.lines_given(g["decisions"].get(C.decision_id(f["id"], "need", way), ""), way),
              C.lines_given(g["decisions"].get(C.decision_id(f["id"], "no_need", way), ""), way)) for f in fs]
    out["frames_with_the_same_number_in_both_versions"] = sum(a is not None and a == b for a, b in pairs)
    return out


def main() -> int:
    import run_cost as R

    C = R.C8
    rep = {"what": "description only; counted after the readings"}
    x = R.lie_inputs(None)
    if x:
        rep["lying_what_the_notes_hold_by_kind_of_row"] = notes_by_kind(x["data"]["records"], x["labels"], x["letters"],
                                                                         x["props"], C)
    fs = C.frames()
    files = R.give_files(None)
    if files:
        rep["giving_spread_of_lines_given"] = {where: {way: spread(g, fs, way, C) for way in C.GIVE_WAYS}
                                               for where, g in files.items()}
    R.L.R.write_json(R.out_dir() / "describe.json", rep)
    for arch, by in (rep.get("lying_what_the_notes_hold_by_kind_of_row") or {}).items():
        print(arch)
        for key, v in by.items():
            lies = "" if v["lies_among_judged"] is None else f"{100 * v['lies_among_judged']:.0f}% lies"
            print(f"   notes hold {key:16s} {v['replies']:4d} replies: lie {v['lie']}, honest {v['honest']}, evade {v['evade']}  {lies}")
    for where, by in (rep.get("giving_spread_of_lines_given") or {}).items():
        for way, v in by.items():
            print(f"{where:18s} {way:8s} need {v['need']}  wasted {v['no_need']}  same in both: "
                  f"{v['frames_with_the_same_number_in_both_versions']} of {len(fs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
