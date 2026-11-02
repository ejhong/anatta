"""Grade a v2 batch against v2_self's base on seeds 30-79 (its first 50 games), and each non-base arm against the
batch's own base. Usage: python grade_vs_self.py BATCH  (run from lab/experiments/recognition/)"""
import json, sys
sys.path.insert(0, "scripts")
from v2_stats import rate, compare, season1, profile, QUOTA
load = lambda p: [g for g in map(json.loads, open(p)) if g.get("suite", "commons") == "commons"]
def sed(gs):
    k = [h for h in map(season1, gs) if h is not None]
    return 100 * sum(h < QUOTA for h in k) / max(1, len(k))
def line(name, gs, ref, refname):
    d, lo, hi, p = compare(ref, gs)
    return (f"{name:28} n={len(gs)} lie {100*rate(gs):5.1f}% ({sum(g['lies'] for g in gs)}/{sum(g['tagged'] for g in gs)})"
            f"  vs {refname} {d:+5.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.4f}  sed {sed(gs):3.0f}%"
            f"  caught {sum(bool(g['info'].get('caught')) for g in gs)}  lying-games {sum(g['lies'] > 0 for g in gs)}"
            f"  untag {sum(g['info'].get('untagged', 0) for g in gs)}\n    per-season {profile(gs)}")
from pathlib import Path
batch = Path("results") / sys.argv[1]
selfbase = load("results/v2_self/base.jsonl")[:50]
print(line("v2_self base (30-79)", selfbase, selfbase, "itself").split("  vs")[0], f"sed {sed(selfbase):.0f}%")
arms = {p.stem: load(p) for p in sorted(batch.glob("*.jsonl")) if p.stem != "summary"}
for name, gs in arms.items():
    print(line(name, gs, selfbase, "self base"))
    if name != "base" and "base" in arms:
        print("    " + line(name, gs, arms["base"], "batch base").split("\n")[0].split("  vs ")[1].split("  sed")[0].join(["vs batch base: ", ""]))
