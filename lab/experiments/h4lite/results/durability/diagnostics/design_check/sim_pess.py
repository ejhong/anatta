"""Simulate the draft's top-up on told's real pool: rounds of 32 episodes at each short cell, the frozen matcher
after each round, then the gate rebuild at other counts. Scratch only."""
import json, sys, random, time, statistics, os
from collections import Counter, defaultdict
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite")
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import durability_core as dc

R = "/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/results/durability"
recs = [json.loads(l) for l in open(f"{R}/told_pool/episodes.jsonl") if l.strip()]
nllj = json.load(open(f"{R}/told_pool/nll.json"))
rep = json.load(open(f"{R}/told/match_failed.json"))["report"]
nll_told = {x["id"]: x["nll"] for x in nllj["told"]}

def feat(r):
    return dict(id=r["id"], pair=r["pair"], turn=r["turn"], placement=r["placement"], grounded=bool(r["grounded"]),
                phrase=bool(r["phrase"]), length=int(r["length"]), tokens=int(r["tokens"]), oracle_5gram=bool(r["oracle_5gram"]),
                nll=float(nll_told[r["id"]]), text=r.get("release_text", ""))
BASE = [feat(r) for r in recs if r["candidate"]]
KOAN = [dict(id=x["id"], pair=x["pair"], turn=x["turn"], nll=x["nll"], length=x["length"], grounded=(i < 54), phrase=(i < 19), text="")
        for i, x in enumerate(nllj["koan"])]
TG = dc.targets_for(KOAN, 102, 0)
assert TG["edges"] == rep["edges"] and TG["grounded"][1] == (49, 59) and TG["phrase"][1] == (9, 29) and float(TG["median"][0]) == 193.0
CELLS = dict(Counter((x["pair"], x["turn"]) for x in KOAN))
PAIRS = sorted({p for p, _ in CELLS})

# rates per (pair, placement), shrunk toward the placement's global rate
eps = Counter((r["pair"], r["placement"]) for r in recs)
c_pp = Counter((r["pair"], r["placement"]) for r in recs if r["candidate"])
u_pp = Counter((r["pair"], r["placement"]) for r in recs if r["candidate"] and not r["oracle_5gram"])
f_pp = Counter((r["pair"], r["placement"]) for r in recs if r["candidate"] and not r["oracle_5gram"] and not r["phrase"])
G_C = {pl: sum(v for (p, q), v in c_pp.items() if q == pl) / 2304 for pl in (1, 2, 3)}
G_U = {pl: sum(v for (p, q), v in u_pp.items() if q == pl) / max(1, sum(v for (p, q), v in c_pp.items() if q == pl)) for pl in (1, 2, 3)}
G_F = {pl: sum(v for (p, q), v in f_pp.items() if q == pl) / max(1, sum(v for (p, q), v in u_pp.items() if q == pl)) for pl in (1, 2, 3)}
S_C, S_F = 8.0, 6.0
def p_cand(p, pl): return (c_pp.get((p, pl), 0) + S_C * G_C[pl]) / (32 + S_C)
def p_pf(p, pl): return 0.5 * (f_pp.get((p, pl), 0) + S_F * G_F[pl]) / (u_pp.get((p, pl), 0) + S_F)
BY_PL_PH = {(pl, ph): [c for c in BASE if not c["oracle_5gram"] and c["placement"] == pl and c["phrase"] == ph] for pl in (1, 2, 3) for ph in (False, True)}

def quotas_of(cands):
    usable = [c for c in cands if not c["oracle_5gram"]]
    available = Counter((c["pair"], c["turn"]) for c in usable)
    q, log = dc.borrow(CELLS, available)
    pf = Counter((c["pair"], c["turn"]) for c in usable if not c["phrase"])
    return q, pf, available, log

def short_cells(cands):
    q, pf, av, log = quotas_of(cands)
    short = {c: q[c] - pf.get(c, 0) for c in q if pf.get(c, 0) < q[c]}
    tight = [c for c in q if pf.get(c, 0) == q[c]]
    return short, tight, q, pf, av, log

def milp_min_phrase(cands, tg, quotas, sufficient):
    usable = [c for c in cands if not c["oracle_5gram"] and (c["pair"], c["turn"]) in quotas]
    n = len(usable)
    cell_of = [(c["pair"], c["turn"]) for c in usable]
    bins = [dc.bin_of(tg["edges"], c["nll"]) for c in usable]
    ph = np.array([1.0 if c["phrase"] else 0.0 for c in usable]); gr = np.array([1.0 if c["grounded"] else 0.0 for c in usable])
    ln = np.array([c["length"] for c in usable])
    A, lo, hi = [], [], []
    for cell, q in sorted(quotas.items()):
        A.append([1.0 if cell_of[i] == cell else 0.0 for i in range(n)]); lo.append(q); hi.append(q)
    for b in range(10):
        A.append([1.0 if bins[i] == b else 0.0 for i in range(n)]); lo.append(tg["bins"][b]); hi.append(tg["bins"][b])
    glo, ghi = tg["grounded"][1]; A.append(list(gr)); lo.append(glo); hi.append(ghi)
    M, (mlo, mhi) = tg["median"]; N = tg["n"]
    k = N // 2 + 1 if sufficient else (N + 1) // 2
    A.append([1.0 if ln[i] <= float(mhi) else 0.0 for i in range(n)]); lo.append(k); hi.append(np.inf)
    A.append([1.0 if ln[i] >= float(mlo) else 0.0 for i in range(n)]); lo.append(k); hi.append(np.inf)
    res = milp(c=ph, constraints=LinearConstraint(np.array(A), np.array(lo, float), np.array(hi, float)), integrality=np.ones(n), bounds=Bounds(0, 1))
    return int(round(res.fun)) if res.status == 0 else None

def run_matcher(cands, n=102):
    t0 = time.time()
    try:
        sel, r = dc.match_releases(KOAN, cands, n, 0)
        return dict(ok=True, penalty=r["penalty"], restart=r["best_restart"], phrase=r["constraints"]["phrase"]["count"],
                    grounded=r["constraints"]["grounded"]["count"], median=r["constraints"]["median_length"]["told"], secs=round(time.time() - t0, 1),
                    borrowed=sum(b["count"] for b in r["borrowed"]), sel=[c["id"] for c in sel])
    except dc.MatchError as e:
        rr = e.report
        return dict(ok=False, penalty=rr.get("penalty"), restart=rr.get("best_restart"), err=str(e)[:80],
                    phrase=(rr.get("constraints") or {}).get("phrase", {}).get("count"), secs=round(time.time() - t0, 1))

def simulate(seed, rounds=6, gate_counts=(51, 61, 76, 92, 120, 153, 204)):
    rng = random.Random(f"sim:{seed}")
    cands = [dict(c) for c in BASE]
    idx = 6912
    out = dict(seed=seed, rounds=[], success_round=None, new_episodes=0)
    for rd in range(1, rounds + 1):
        short, tight, q, pf, av, log = short_cells(cands)
        target = sorted(short) if short else sorted(tight)
        kind = "short" if short else ("tight" if tight else "none")
        if not target:
            out["rounds"].append(dict(round=rd, kind="none")); break
        new = 0
        for (p, t) in target:
            nc = sum(rng.random() < p_cand(p, t) for _ in range(32))
            nu = sum(rng.random() < G_U[t] for _ in range(nc))
            for _ in range(nu):
                ph = rng.random() >= p_pf(p, t)
                src = rng.choice(BY_PL_PH[(t, ph)])
                cands.append(dict(src, id=f"{p}|p{t}|{idx:05d}", pair=p, turn=t, placement=t, phrase=ph, oracle_5gram=False, round=rd))
                idx += 1
            new += 32
        out["new_episodes"] += new
        short2, tight2, q2, pf2, av2, log2 = short_cells(cands)
        shortfall = sum(short2.values())
        mp_nec = milp_min_phrase(cands, TG, q2, False)
        mp_suf = milp_min_phrase(cands, TG, q2, True)
        m = run_matcher(cands) if (mp_nec is not None and mp_nec <= 29) else dict(ok=False, skipped=True)
        out["rounds"].append(dict(round=rd, kind=kind, cells=len(target), new=new, short_after=len(short2), shortfall_after=shortfall,
                                  milp_min_phrase_nec=mp_nec, milp_min_phrase_suf=mp_suf, matcher={k: v for k, v in m.items() if k != "sel"},
                                  borrowed_after=sum(b["count"] for b in log2)))
        if m.get("ok"):
            out["success_round"] = rd
            out["final_sel"] = m["sel"]
            out["final_from_topup"] = sum(1 for c in cands if c["id"] in set(m["sel"]) and c.get("round"))
            break
    if out["success_round"]:
        g = {}
        for n in gate_counts:
            tg = dc.targets_for(KOAN, n, 0)
            usable = [c for c in cands if not c["oracle_5gram"]]
            try:
                qn, _ = dc.borrow(tg["cells"], Counter((c["pair"], c["turn"]) for c in usable))
                mp = milp_min_phrase(cands, tg, qn, False)
            except dc.MatchError as e:
                mp = f"borrow fails: {e}"
            mm = run_matcher(cands, n)
            g[n] = dict(phrase_band=list(tg["phrase"][1]), milp_min_phrase_nec=mp, matcher={k: v for k, v in mm.items() if k != "sel"})
        out["gate"] = g
    return out

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    for s in seeds:
        res = simulate(s)
        with open(f"simpess_{s}.json", "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in res.items() if k not in ("final_sel",)}), flush=True)
