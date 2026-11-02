"""Summarize crossmap*.jsonl (every map variant tried): the gate per variant (months, leave-all-months-out, top-1 >=
0.33 and within-one >= 0.6 in both directions of each Qwen <-> Ministral pair) and the best variants. Claude (Opus 5.5)
wrote this file, 2 Nov 2026. Private (lab/explore).

Run: python3 -I lab/explore/selfmap/summarize_crossmap.py <file.jsonl> [more.jsonl]
"""

import json
import sys
from collections import defaultdict

GATE_TOP1, GATE_W1 = 0.33, 0.6
CROSS = {("qwen_base", "ministral"), ("ministral", "qwen_base"), ("qwen_instruct", "ministral"),
         ("ministral", "qwen_instruct")}


def vkey(r):
    return (r["readout"], r["depth"], r["prep"], r["fitset"], r["contrast"], r["map"], r.get("k"), r.get("whiten"),
            r.get("c"))


def main():
    recs = [json.loads(x) for f in sys.argv[1:] for x in open(f)]
    by = defaultdict(dict)
    for r in recs:
        by[(vkey(r), r["test"])][(r["a"], r["b"])] = r
    n_var = len({vkey(r) for r in recs})
    print(f"{len(recs)} records, {n_var} variants")
    rows = []
    for (vk, test), pr in by.items():
        cross = [pr[p] for p in CROSS if p in pr]
        if len(cross) < 4:
            continue
        mt = min(x["top1"] for x in cross)
        mw = min(x["within1"] for x in cross)
        mean_t = sum(x["top1"] for x in cross) / 4
        bi = pr.get(("qwen_base", "qwen_instruct"), {}).get("top1")
        rows.append((test, mean_t, mt, mw, bi, vk))
    for test in ("months", "weekdays"):
        rs = sorted([r for r in rows if r[0] == test], key=lambda r: -r[1])
        passing = [r for r in rs if test == "months" and r[2] >= GATE_TOP1 and r[3] >= GATE_W1]
        print(f"\n== {test}: {len(rs)} variants; passing the gate: {len(passing)}")
        print("mean cross top-1 | min top-1 | min within-1 | qwen base->instruct top-1 | variant")
        for r in rs[:12]:
            print(f"{r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4] if r[4] is None else round(r[4], 2)} | {r[5]}")


if __name__ == "__main__":
    main()
