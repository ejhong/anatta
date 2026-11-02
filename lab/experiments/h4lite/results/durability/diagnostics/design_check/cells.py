import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, "/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite")
import durability_core as dc
R = "/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/results/durability"
recs = [json.loads(l) for l in open(f"{R}/told_pool/episodes.jsonl") if l.strip()]
nll = json.load(open(f"{R}/told_pool/nll.json"))
rep = json.load(open(f"{R}/told/match_failed.json"))["report"]
cands = [r for r in recs if r["candidate"]]
usable = [r for r in cands if not r["oracle_5gram"]]
cells = Counter((x["pair"], x["turn"]) for x in nll["koan"])
available = Counter((r["pair"], r["turn"]) for r in usable)
quotas, borrowed = dc.borrow(dict(cells), available)
pf = Counter((r["pair"], r["turn"]) for r in usable if not r["phrase"])
# per (pair, placement) episode counts and candidate counts
eps = Counter((r["pair"], r["placement"]) for r in recs)
cand_pp = Counter((r["pair"], r["placement"]) for r in cands)
us_pp = Counter((r["pair"], r["placement"]) for r in usable)
pf_pp = Counter((r["pair"], r["placement"]) for r in usable if not r["phrase"])
print("episodes per (pair,placement):", set(eps.values()))
short = {c: (q, pf.get(c, 0), available.get(c, 0)) for c, q in quotas.items() if pf.get(c, 0) < q}
print("short cells", len(short), "shortfall", sum(q - f for q, f, _ in short.values()))
by_turn = defaultdict(lambda: [0, 0, 0])
for (p, t), (q, f, a) in short.items():
    by_turn[t][0] += 1; by_turn[t][1] += q - f; by_turn[t][2] += 1
print("short cells by turn: {turn: [cells, shortfall]}", {t: v[:2] for t, v in sorted(by_turn.items())})
print("\nshort cells (pair, turn): quota, phrase-free, usable, candidates at that placement, episodes")
for (p, t), (q, f, a) in sorted(short.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    print(f"  {p:9s} t{t}: quota {q} pf {f} usable {a} cands {cand_pp.get((p, t), 0)} koan_target {cells.get((p, t), 0)}")
# turn-1 cells overall
print("\nturn-1 koan cells:", sum(1 for (p, t) in cells if t == 1), "borrowed:", borrowed)
for (p, t), k in sorted(cells.items()):
    if t == 1:
        print(f"  {p:9s} target {k} quota {quotas.get((p, t), 0)} usable {available.get((p, t), 0)} pf {pf.get((p, t), 0)} cands {cand_pp.get((p, 1), 0)}")
# tight cells (pf == quota)
tight = [c for c, q in quotas.items() if pf.get(c, 0) == q]
print("\ntight cells (pf == quota):", len(tight))
# placement-level rates
for pl in (1, 2, 3):
    n_ep = sum(v for (p, q), v in eps.items() if q == pl)
    c = sum(v for (p, q), v in cand_pp.items() if q == pl)
    u = sum(v for (p, q), v in us_pp.items() if q == pl)
    f = sum(v for (p, q), v in pf_pp.items() if q == pl)
    print(f"placement {pl}: episodes {n_ep} cands {c} ({c/n_ep:.3f}) usable {u} ({u/max(c,1):.3f} of cands) pf {f} ({f/max(u,1):.3f} of usable) -> pf per 32 episodes {32*f/n_ep:.2f}")
# per-pair p1 candidate counts distribution
print("\np1 candidates per pair (of 32):", sorted(Counter(cand_pp.get((p, 1), 0) for p in {p for p, _ in cells}).items()))
# koan NLL decile widths
e = rep["edges"]; print("\ndecile edges", [round(x, 3) for x in e]); print("widths", [round(b - a, 3) for a, b in zip(e, e[1:])])
# per-cell: pf by placement for the short cells' pairs at other placements (information)
json.dump({"short": {f"{p}|{t}": v for (p, t), v in short.items()}, "quotas": {f"{p}|{t}": q for (p, t), q in quotas.items()}}, open("short_cells.json", "w"), indent=1)
