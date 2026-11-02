"""Feasibility diagnostic (changes nothing): the minimum number of phrase-bearing reasons any selection of 102 from
the usable pool must contain, under the matcher's per-cell quotas (swaps stay in a cell), then with the deciles,
the grounded band and a median-length condition added. scipy.optimize.milp."""
import json
import statistics
import sys
from collections import Counter
from fractions import Fraction

sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite")
import numpy as np  # noqa: E402
import durability_core as dc  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402

R = "/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/results/durability"
recs = [json.loads(l) for l in open(f"{R}/told_pool/episodes.jsonl")]
nll = json.load(open(f"{R}/told_pool/nll.json"))
rep = json.load(open(f"{R}/told/match_failed.json"))["report"]
nll_told = {x["id"]: x["nll"] for x in nll["told"]}

cands = [r for r in recs if r["candidate"] and not r["oracle_5gram"]]
koan_nll = [x["nll"] for x in nll["koan"]]
edges = dc.decile_edges(koan_nll)
assert edges == rep["edges"]
tb = [Counter(dc.bin_of(edges, v) for v in koan_nll).get(b, 0) for b in range(10)]
cells = Counter((x["pair"], x["turn"]) for x in nll["koan"])
available = Counter((r["pair"], r["turn"]) for r in cands)
quotas, _ = dc.borrow(dict(cells), available)
rep_quotas = {(p, int(t)): v["quota"] for k, v in rep["cells"].items() for p, t in [k.split("|")] if v.get("quota")}
assert quotas == rep_quotas, "quotas differ from the report"

pool = [r for r in cands if (r["pair"], r["turn"]) in quotas]
n = len(pool)
cell_of = [(r["pair"], r["turn"]) for r in pool]
bin_ = [dc.bin_of(edges, nll_told[r["id"]]) for r in pool]
ph = np.array([1.0 if r["phrase"] else 0.0 for r in pool])
gr = np.array([1.0 if r["grounded"] else 0.0 for r in pool])
ln = np.array([r["length"] for r in pool])
print(f"usable candidates in quota cells: {n} of {len(cands)}; cells {len(quotas)}; quotas sum {sum(quotas.values())}")

# constraint rows
def rows_cells():
    A, lo, hi = [], [], []
    for c, q in sorted(quotas.items()):
        A.append([1.0 if cell_of[i] == c else 0.0 for i in range(n)])
        lo.append(q); hi.append(q)
    return A, lo, hi

def rows_deciles():
    A, lo, hi = [], [], []
    for b in range(10):
        A.append([1.0 if bin_[i] == b else 0.0 for i in range(n)])
        lo.append(tb[b]); hi.append(tb[b])
    return A, lo, hi

def rows_grounded():
    return [list(gr)], [49], [59]

def rows_median(sufficient: bool):
    """Median of 102 integer lengths within [173.7, 212.3]. Necessary: >= 51 selected with length <= 212 and
    >= 51 with length >= 174 (v51 <= 212.3 and v52 >= 173.7). Sufficient: >= 52 of each."""
    k = 52 if sufficient else 51
    return ([[1.0 if ln[i] <= 212 else 0.0 for i in range(n)], [1.0 if ln[i] >= 174 else 0.0 for i in range(n)]],
            [k, k], [np.inf, np.inf])

def solve(name, parts, objective=ph, sense=1):
    A, lo, hi = [], [], []
    for a, l, h in parts:
        A += a; lo += l; hi += h
    res = milp(c=sense * objective, constraints=LinearConstraint(np.array(A), np.array(lo, float), np.array(hi, float)),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if res.status != 0:
        print(f"{name}: status {res.status} ({res.message})")
        return None
    x = np.round(res.x).astype(int)
    sel = [pool[i] for i in range(n) if x[i]]
    lens = sorted(r["length"] for r in sel)
    med = statistics.median(lens)
    print(f"{name}: optimum {int(round(res.fun * sense))}; selected {len(sel)}, phrase {int(ph @ x)}, grounded {int(gr @ x)}, "
          f"median length {med}, bins {[int(sum(1 for i in range(n) if x[i] and bin_[i] == b)) for b in range(10)]}")
    return int(round(res.fun * sense))

print("phrase band 9-29, grounded band 49-59, median band 173.7-212.3, deciles", tb)
solve("min phrase | cells only", [rows_cells()])
solve("max phrase | cells only", [rows_cells()], sense=-1)
solve("min phrase | cells + deciles", [rows_cells(), rows_deciles()])
solve("min phrase | cells + grounded", [rows_cells(), rows_grounded()])
solve("min phrase | cells + deciles + grounded", [rows_cells(), rows_deciles(), rows_grounded()])
solve("min phrase | cells + deciles + grounded + median(necessary)", [rows_cells(), rows_deciles(), rows_grounded(), rows_median(False)])
solve("min phrase | cells + deciles + grounded + median(sufficient)", [rows_cells(), rows_deciles(), rows_grounded(), rows_median(True)])
print("--- grounded alone")
solve("min grounded | cells only", [rows_cells()], objective=gr)
solve("max grounded | cells only", [rows_cells()], objective=gr, sense=-1)
solve("min grounded | cells + deciles", [rows_cells(), rows_deciles()], objective=gr)
solve("min grounded | cells + deciles + median(sufficient)", [rows_cells(), rows_deciles(), rows_median(True)], objective=gr)
print("--- the matcher's reported best: phrase 40, grounded 60, deciles met, median 178")
print("--- is the matcher's reported best reachable by MILP (phrase<=40, grounded<=60, deciles, median)?")
A, lo, hi = [], [], []
for a, l, h in [rows_cells(), rows_deciles(), rows_median(True)]:
    A += a; lo += l; hi += h
res = milp(c=ph + gr, constraints=LinearConstraint(np.array(A), np.array(lo, float), np.array(hi, float)),
           integrality=np.ones(n), bounds=Bounds(0, 1))
x = np.round(res.x).astype(int)
print(f"min (phrase + grounded) | cells + deciles + median(sufficient): {int(res.fun)} = phrase {int(ph @ x)} + grounded {int(gr @ x)}")
