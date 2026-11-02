import json, glob, statistics
from collections import Counter
rs = [json.load(open(f)) for f in sorted(glob.glob("sim_*.json"))]
print("replicates", len(rs))
sr = Counter(r["success_round"] for r in rs)
print("success round:", dict(sorted(sr.items(), key=lambda kv: (kv[0] is None, kv[0]))))
print("new episodes at success:", sorted(r["new_episodes"] for r in rs if r["success_round"]))
print("cells sampled per round (min..max):", [ (rd, min(x), max(x)) for rd in range(1, 7) for x in [[r0["cells"] for r in rs for r0 in r["rounds"] if r0["round"] == rd and "cells" in r0]] if x])
print("kinds:", Counter(r0["kind"] for r in rs for r0 in r["rounds"]))
# per round: feasibility by milp vs matcher
for rd in range(1, 7):
    rows = [r0 for r in rs for r0 in r["rounds"] if r0["round"] == rd and "matcher" in r0]
    if not rows: continue
    feas = sum(1 for r0 in rows if r0["milp_min_phrase_nec"] is not None and r0["milp_min_phrase_nec"] <= 29)
    feas_s = sum(1 for r0 in rows if r0["milp_min_phrase_suf"] is not None and r0["milp_min_phrase_suf"] <= 29)
    ok = sum(1 for r0 in rows if r0["matcher"].get("ok"))
    sf = [r0["shortfall_after"] for r0 in rows]
    mp = [r0["milp_min_phrase_nec"] for r0 in rows]
    pen = [r0["matcher"].get("penalty") for r0 in rows if not r0["matcher"].get("ok") and not r0["matcher"].get("skipped")]
    print(f"round {rd}: n {len(rows)} shortfall after: median {statistics.median(sf)} range {min(sf)}-{max(sf)}; milp min phrase median {statistics.median(mp)} range {min(mp)}-{max(mp)}; milp-feasible(nec) {feas} (suf {feas_s}); matcher ok {ok}; failing penalties {sorted(pen)}")
ph = [r0["matcher"]["phrase"] for r in rs for r0 in r["rounds"] if r0.get("matcher", {}).get("ok")]
gr = [r0["matcher"]["grounded"] for r in rs for r0 in r["rounds"] if r0.get("matcher", {}).get("ok")]
md = [r0["matcher"]["median"] for r in rs for r0 in r["rounds"] if r0.get("matcher", {}).get("ok")]
print("phrase count at success:", sorted(ph)); print("grounded at success:", sorted(gr)); print("median length at success:", sorted(md))
print("selected releases from the top-up:", sorted(r.get("final_from_topup") for r in rs if r["success_round"]))
print("matcher secs:", sorted(r0["matcher"].get("secs") for r in rs for r0 in r["rounds"] if r0.get("matcher", {}).get("secs") is not None)[-5:])
# gate
gs = [r["gate"] for r in rs if r.get("gate")]
for n in ("51", "61", "76", "92", "120", "153", "204"):
    ok = sum(1 for g in gs if g[n]["matcher"].get("ok"))
    feas = sum(1 for g in gs if isinstance(g[n]["milp_min_phrase_nec"], int) and g[n]["milp_min_phrase_nec"] <= g[n]["phrase_band"][1])
    borrowfail = sum(1 for g in gs if isinstance(g[n]["milp_min_phrase_nec"], str))
    print(f"gate n={n}: band {gs[0][n]['phrase_band']}; milp-feasible {feas}/{len(gs)}; matcher ok {ok}/{len(gs)}; borrow fails {borrowfail}")
