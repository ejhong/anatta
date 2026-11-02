import json, re, random
from collections import Counter
R = "/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/results/durability"
recs = [json.loads(l) for l in open(f"{R}/told_pool/episodes.jsonl") if l.strip()]
usable = [r for r in recs if r["candidate"] and not r["oracle_5gram"]]
free = [r for r in usable if not r["phrase"]]
terms = {"inconsistent": r"inconsisten", "conflict": r"conflict", "impossible": r"impossib", "as specified": r"as specified", "specification": r"specification", "note": r"\bnote\b", "cannot": r"\bcannot\b|can't|can not", "contradiction-free": None}
print("phrase-free usable:", len(free))
for name, pat in terms.items():
    if pat:
        n = sum(1 for r in free if re.search(pat, (r["reason"] or "").lower()))
        print(f"  reason has {name!r}: {n} ({n/len(free):.0%})")
anyof = sum(1 for r in free if re.search(r"inconsisten|conflict|impossib|as specified|specification", (r["reason"] or "").lower()))
print(f"  any of inconsistent/conflict/impossible/as specified/specification: {anyof} ({anyof/len(free):.0%})")
# the same on the phrase-bearing ones and on the koan-like vocabulary is unknown here; show samples from short cells
short = json.load(open("short_cells.json"))["short"]
sc = {tuple(k.split("|")) for k in short}
rng = random.Random(1)
print("\nsample phrase-free reasons from pairs of short cells (placement 2/3):")
cands = [r for r in free if (r["pair"], str(r["turn"])) in {(p, t) for p, t in sc} or r["pair"] in {p for p, _ in sc}]
for r in rng.sample(cands, 8):
    print(f"- [{r['pair']} p{r['placement']} grounded={bool(r['grounded'])}] {(r['reason'] or '')[:230]!r}")
print("\nsample phrase-bearing reasons:")
for r in rng.sample([r for r in usable if r["phrase"]], 4):
    print(f"- [{r['pair']} p{r['placement']} grounded={bool(r['grounded'])}] {(r['reason'] or '')[:200]!r}")
# NLL and length of phrase-free vs phrase-bearing (usable)
nll = {x["id"]: x["nll"] for x in json.load(open(f"{R}/told_pool/nll.json"))["told"]}
import statistics
for name, grp in (("phrase-free", free), ("phrase-bearing", [r for r in usable if r["phrase"]])):
    print(f"{name}: n {len(grp)} median nll {statistics.median(nll[r['id']] for r in grp):.3f} median length {statistics.median(r['length'] for r in grp)} grounded {sum(bool(r['grounded']) for r in grp)/len(grp):.0%}")
