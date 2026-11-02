"""Diagnostic only (changes nothing, decides nothing): could any selection of told's pool have met the phrase band
under the matcher's cell quotas? Written and run by Opus 5.5 after the 30 Oct MatchError, checked by Fable (whose
exact version with the deciles and bands is milp_feasibility.py). Reads told_pool/episodes.jsonl and
told/match_failed.json."""
import json
from collections import Counter, defaultdict
from pathlib import Path

D = Path("/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/results/durability")
rep = json.loads((D / "told/match_failed.json").read_text())["report"]
quotas = {k: v["quota"] for k, v in rep["cells"].items() if v.get("quota")}
assert sum(quotas.values()) == 102, sum(quotas.values())

recs = [json.loads(l) for l in (D / "told_pool/episodes.jsonl").read_text().splitlines() if l.strip()]
cands = [r for r in recs if r.get("candidate")]
pool = [r for r in cands if not r["oracle_5gram"]]
print("episodes", len(recs), "candidates", len(cands), "pool", len(pool))
print("pool phrase", sum(r["phrase"] for r in pool), "grounded", sum(bool(r["grounded"]) for r in pool))
free = [r for r in pool if not r["phrase"]]
print("phrase-free", len(free), "pairs", len({r["pair"] for r in free}),
      "by placement", dict(sorted(Counter(r["placement"] for r in free).items())))
print("pool by placement", dict(sorted(Counter(r["placement"] for r in pool).items())))

by_cell = defaultdict(list)
for r in pool:
    by_cell[f"{r['pair']}|{r['turn']}"].append(r)
max_free = sum(min(q, sum(not r["phrase"] for r in by_cell[c])) for c, q in quotas.items())
print("cells with quota", len(quotas), "max phrase-free under the quotas", max_free,
      "-> fewest phrase reasons possible", 102 - max_free, "(band 9-29)")
max_ungrounded = sum(min(q, sum(not r["grounded"] for r in by_cell[c])) for c, q in quotas.items())
print("max ungrounded under the quotas", max_ungrounded, "-> fewest grounded possible", 102 - max_ungrounded,
      "(band 49-59)")
# both at once, ignoring deciles and length: per cell, the best mix is a small knapsack; bound it by a greedy count
both = sum(min(q, sum((not r["phrase"]) and (not r["grounded"]) for r in by_cell[c])) for c, q in quotas.items())
print("max phrase-free AND ungrounded under the quotas", both)
short = [(c, q, sum(not r["phrase"] for r in by_cell[c]), len(by_cell[c])) for c, q in quotas.items()
         if sum(not r["phrase"] for r in by_cell[c]) < q]
print("cells where phrase-free reasons are fewer than the quota:", len(short), "missing",
      sum(q - f for _, q, f, _ in short))
# information only, not the registered rule: if a pair's releases could come from any of its turns
by_pair_q, by_pair_free = defaultdict(int), defaultdict(int)
for c, q in quotas.items():
    by_pair_q[c.split("|")[0]] += q
for r in free:
    by_pair_free[r["pair"]] += 1
print("information only: with a pair's quota fillable from any of its turns, the fewest phrase reasons would be",
      sum(max(0, q - by_pair_free[p]) for p, q in by_pair_q.items()))
