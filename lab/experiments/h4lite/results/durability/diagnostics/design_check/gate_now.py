import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, ".")
import sim, durability_core as dc
from collections import Counter
usable = [c for c in sim.BASE if not c["oracle_5gram"]]
av = Counter((c["pair"], c["turn"]) for c in usable)
pf = Counter((c["pair"], c["turn"]) for c in usable if not c["phrase"])
for n in (51, 61, 76, 92, 102, 120, 153, 204):
    tg = dc.targets_for(sim.KOAN, n, 0)
    try:
        q, log = dc.borrow(tg["cells"], av)
        sf = sum(max(0, q[c] - pf.get(c, 0)) for c in q)
        short = sum(1 for c in q if pf.get(c, 0) < q[c])
        nocand = sum(1 for c in q if av.get(c, 0) < q[c])
        print(f"n {n}: phrase band {tg['phrase'][1]} grounded {tg['grounded'][1]} cells {len(q)} short {short} cells-only min phrase {sf} ({sf/n:.3f} of n) borrowed {sum(b['count'] for b in log)} -> {'INFEASIBLE' if sf > tg['phrase'][1][1] else 'cells ok'}")
    except dc.MatchError as e:
        print(f"n {n}: borrow fails: {e}")
