"""Grade the random-control replication (lab/ledger/NOTES.md predictions 53-56). Run from lab/experiments/recognition/."""
import json
import sys

sys.path.insert(0, "scripts")
from v2_split import at_mix, counts, raw  # noqa: E402
from v2_stats import QUOTA, compare, rate, season1  # noqa: E402

load = lambda p, a=0, b=None: [g for g in map(json.loads, open(p)) if g.get("suite", "commons") == "commons"][a:b]  # noqa: E731


def sed(gs):
    k = [h for h in map(season1, gs) if h is not None]
    return 100 * sum(h < QUOTA for h in k) / max(1, len(k))


def line(name, gs, base):
    d, lo, hi, p = compare(base, gs)
    same = 100 * (at_mix([counts(g) for g in gs], [counts(g) for g in base]) - raw([counts(g) for g in base]))
    print(f"{name:34} n={len(gs):3} lie {100 * rate(gs):5.1f}%  vs base {d:+5.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.4f}"
          f"  same-harvest {same:+5.1f}  sed {sed(gs):3.0f}%")
    return d


b30 = load("results/v2_self/base.jsonl", 0, 50)
b80 = load("results/v2_self/base.jsonl", 50)
print(f"v2_self base seeds 30-79 {100 * rate(b30):.1f}%, seeds 80-129 {100 * rate(b80):.1f}%")
d53 = line("cut_random, seeds 80-129", load("results/v2_random80/cut_random.jsonl"), b80)
rb = load("results/v2_random/cut_random_b.jsonl")
r17 = load("results/v2_random/cut_random_17_26.jsonl")
r0 = load("results/v2_controls/cut_random.jsonl")
d54 = line("cut_random_b, seeds 30-79", rb, b30)
d55 = line("cut_random_17_26, seeds 30-79", r17, b30)
line("(cut_random, seeds 30-79, v2_controls)", r0, b30)
print(f"\n53. cut_random on 80-129 within 5 of base: {abs(d53) <= 5} ({d53:+.1f})")
print(f"54. cut_random_b within 5 of base: {abs(d54) <= 5} ({d54:+.1f})")
print(f"55. cut_random_17_26 within 5 of base: {abs(d55) <= 5} ({d55:+.1f})")
pool = r0 + rb + r17
print(f"random pool on seeds 30-79: {len(pool)} games, {100 * rate(pool):.1f}%")
ok = []
for a in ("cut", "cut_self_stakes", "cut_self_stakes_17_26"):
    gs = load(f"results/v2_self/{a}.jsonl", 0, 50)
    d, lo, hi, p = compare(pool, gs)
    print(f"  {a:24} {100 * rate(gs):5.1f}% vs random pool {d:+5.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.4f}")
    if a in ("cut", "cut_self_stakes_17_26"):
        ok.append(d <= -5)
print(f"56. the ego cut and the 17-26 cut each at least 5 below the random pool: {all(ok)}")
d, lo, hi, p = compare(b30, pool)
print(f"random pool vs base (seeds 30-79): {d:+.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.4f}")
d, lo, hi, p = compare(load("results/v2_self/cut_self_stakes_17_26.jsonl", 0, 50), r17)
print(f"cut_random_17_26 vs cut_self_stakes_17_26 (layer-matched): {d:+.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.4f}")
