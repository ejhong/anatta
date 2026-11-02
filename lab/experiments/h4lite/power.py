# Power of the main comparison: koan vs noexit, K seeds per arm, 100 games per seed.
# Per-seed sampling SE of a lie rate at 100 games: the v2 100-vs-100 CIs have half-width ~7 pts => SE(diff)~3.6 => per-arm ~2.55.
# Between-seed (training-run) SD "s" is unknown; try 0, 3, 5, 8 points.
import random, statistics as st
random.seed(0)
def sim(effect, s, K, games=100, reps=3000, per_arm_se=2.55):
    se = per_arm_se * (100/games)**0.5
    ok_all = ok_pooled = ok_sign = 0
    for _ in range(reps):
        k = [35 - effect + random.gauss(0, s) + random.gauss(0, se) for _ in range(K)]
        n = [35 + random.gauss(0, s) + random.gauss(0, se) for _ in range(K)]
        d = st.mean(n) - st.mean(k)
        # seed-level CI: t-ish using the K seed means (what a hierarchical bootstrap approximates)
        sd = (st.variance(k) + st.variance(n))**0.5 if K > 1 else float('inf')
        half = 2.78 * sd * (1/K)**0.5 if K == 5 else 4.30 * sd * (1/K)**0.5 if K == 3 else 2.26*sd*(1/K)**0.5
        pooled = d >= 5 and d - half > 0
        # game-level CI (ignores seed variance): SE of pooled diff
        half_g = 1.96 * (2 * se**2 / K)**0.5
        pooled_g = d >= 5 and d - half_g > 0
        sign = all(n[i] > k[i] for i in range(K))
        ok_pooled += pooled_g
        ok_sign += pooled_g and sign
        ok_all += pooled and sign
    return ok_pooled/reps, ok_sign/reps, ok_all/reps
print("effect s K games | P(pooled>=5 & game-level CI>0) | ...& same sign every seed | seed-level CI & sign")
for effect in (5, 8, 12):
    for s in (0, 3, 6):
        for K, games in ((3, 100), (5, 100), (3, 200)):
            a, b, c = sim(effect, s, K, games)
            print(f"{effect:6} {s:2} {K} {games:4} | {a:.2f} | {b:.2f} | {c:.2f}")
