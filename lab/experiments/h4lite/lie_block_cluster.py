"""Cluster bootstrap for the lie block: games that share a sampling stream (same chunk of 25, same slice position mod 12)
are resampled together. Paired comparisons resample the same clusters in both runs."""
import json, random, sys
sys.path.insert(0, '/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite')
import fishery_grade as fg
R = '/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/results/fishery/'
def load(p):
    out = {}
    for l in open(R + p):
        g = json.loads(l)
        if g.get('suite', 'commons') != 'commons': continue
        out[g['seed']] = fg.grade_game(g)
    return out
runs = {'base': load('stageA/base.off.base.jsonl')}
for v in ['audit', 'noquota', 'fullrules', 'honest', 'pool']:
    runs[v] = load(f'lie1/base.off.{v}.jsonl')
for v in ['base', 'fullrules', 'honest', 'pool']:
    runs['cut.' + v] = load(f'lie1/base.cut_self_stakes_17_26.{v}.jsonl')
def cluster(seed):
    i = seed - 30; return (i // 25, (i % 25) % 12)
seeds = sorted(runs['base'])
cl = {}
for s in seeds: cl.setdefault(cluster(s), []).append(s)
keys = sorted(cl)
def rate(run, ss, num, den):
    n = sum(runs[run][s][num] for s in ss); d = sum(runs[run][s][den] for s in ss); return 100 * n / d
def comp(a, b, num='lies', den='tagged', reps=4000, by_cluster=True):
    rng = random.Random(0); obs = rate(a, seeds, num, den) - rate(b, seeds, num, den); bs = []
    for _ in range(reps):
        if by_cluster:
            ss = [s for k in (rng.choice(keys) for _ in keys) for s in cl[k]]
        else:
            ss = [rng.choice(seeds) for _ in seeds]
        bs.append(rate(a, ss, num, den) - rate(b, ss, num, den))
    bs.sort(); return obs, bs[int(.025 * reps)], bs[int(.975 * reps)]
def comp2(a1, b1, a2, b2, reps=4000):  # (b1 - a1) - (b2 - a2): drop1 minus drop2
    rng = random.Random(0)
    f = lambda ss: (rate(b1, ss, 'lies', 'tagged') - rate(a1, ss, 'lies', 'tagged')) - (rate(b2, ss, 'lies', 'tagged') - rate(a2, ss, 'lies', 'tagged'))
    obs = f(seeds); bs = []
    for _ in range(reps):
        ss = [s for k in (rng.choice(keys) for _ in keys) for s in cl[k]]; bs.append(f(ss))
    bs.sort(); return obs, bs[int(.025 * reps)], bs[int(.975 * reps)]
print(f'{len(keys)} clusters over {len(seeds)} games')
for name, a, b in [('120a cut.base-base', 'cut.base', 'base'), ('120b cut.fullrules-fullrules', 'cut.fullrules', 'fullrules'),
                   ('120c cut.pool-pool', 'cut.pool', 'pool'), ('fullrules-base', 'fullrules', 'base'), ('fullrules-audit', 'fullrules', 'audit'),
                   ('audit-base', 'audit', 'base'), ('noquota-base', 'noquota', 'base'), ('pool-base', 'pool', 'base'), ('honest-base', 'honest', 'base')]:
    o, lo, hi = comp(a, b); og, log_, hig = comp(a, b, by_cluster=False)
    lo_, l1, h1 = comp(a, b, 'live_lies', 'live_tagged')
    print(f'{name:30s} all {o:+6.1f} cluster [{lo:+.1f}, {hi:+.1f}]  game [{log_:+.1f}, {hig:+.1f}] | live {lo_:+6.1f} cluster [{l1:+.1f}, {h1:+.1f}]')
o, lo, hi = comp2('honest', 'base', 'cut.base', 'base')
print(f'121 drop(honest) - drop(cut) {o:+.1f} cluster [{lo:+.1f}, {hi:+.1f}]')
for v in ['base', 'audit', 'noquota', 'fullrules', 'pool']:
    print(v, 'live lie rate', round(rate(v, seeds, 'live_lies', 'live_tagged'), 1), 'caught games', sum(1 for s in seeds if runs[v][s]['post_seasons'] > 0))
