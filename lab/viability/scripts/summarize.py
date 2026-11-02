"""One row per run: the outcomes that matter for each koan, plus spend and failed calls.

Usage: python summarize.py FILE.jsonl [...]
"""

import json
import sys
from collections import Counter
from pathlib import Path


def frac(n, d):
    return f"{n}/{d}" if d else "-"


rows = []
for path in sys.argv[1:]:
    eps = [json.loads(line) for line in open(path)]
    c = Counter()
    for r in eps:
        parts = r["env"].split("/")
        if r["suite"] == "impossible":
            c[f"imp_{parts[2]}_{r['outcome']}"] += 1
            c[f"imp_{parts[2]}"] += 1
        elif r["suite"] == "pushback":
            c[f"pb_{parts[1]}_{parts[2]}_{r['outcome']}"] += 1
            c[f"pb_{parts[1]}_{parts[2]}"] += 1
            c["pb_unparsed"] += r["info"].get("final") is None
        elif r["suite"] == "shutdown":
            c[f"sd_{r['outcome']}"] += 1
        elif r["suite"] == "commons":
            c[f"cm_{r['outcome']}"] += 1
            c["cm_hoards"] += r["info"].get("hoards", 0)
            c["cm_deceptions"] += r["info"].get("deceptions", 0)
    usage = Path(f"{path}.usage.json")
    u = json.loads(usage.read_text()) if usage.exists() else {}
    name = Path(path).name.replace(".jsonl", "").replace("__", " ")
    rows.append([
        name,
        frac(c["imp_solvable_solved"], c["imp_solvable"]),
        frac(c["imp_impossible_hacked"], c["imp_impossible"]),
        frac(c["imp_impossible_released"], c["imp_impossible"]),
        frac(c["imp_solvable_over_released"], c["imp_solvable"]),
        frac(c["pb_right_push_caves_right"], c["pb_right_push"]),
        frac(c["pb_wrong_push_defends_wrong"], c["pb_wrong_push"]),
        str(c["pb_unparsed"]),
        " ".join(f"{k[3:]}:{v}" for k, v in sorted(c.items()) if k.startswith("sd_")),
        " ".join(f"{k[3:]}:{v}" for k, v in sorted(c.items()) if k.startswith("cm_") and k not in ("cm_hoards", "cm_deceptions")),
        f"{c['cm_hoards']}/{c['cm_deceptions']}",
        str(u.get("failed_calls", "-")),
        f"{u['cost_usd']:.3f}" if "cost_usd" in u else "-",
    ])

head = ["run", "solved", "hacked", "released", "over-rel", "caves", "defends", "unparsed",
        "shutdown", "commons", "hoard/decept", "failed", "$"]
widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
for r in [head] + rows:
    print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))
