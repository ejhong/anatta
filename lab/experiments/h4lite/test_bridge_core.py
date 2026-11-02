"""Checks for bridge_core.py on synthetic numbers. Run: python3 test_bridge_core.py (needs numpy)."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bridge_core as bc  # noqa: E402

checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    if not cond:
        print("FAIL", name)


# --- lists
check("32 harmful, 32 harmless", len(bc.HARMFUL) == 32 and len(bc.HARMLESS) == 32)
check("instructions unique", len(set(bc.HARMFUL + bc.HARMLESS)) == 64)
check("about 200 concepts", 195 <= len(bc.CONCEPTS) <= 205)
check("concepts unique, lowercase, single words", len(set(bc.CONCEPTS)) == len(bc.CONCEPTS)
      and all(c == c.lower() and " " not in c for c in bc.CONCEPTS))
check("no concept is a frame word after the carrier or a self word",
      not {"what", "about", "message", "received", "i", "me", "my", "you", "your", "it", "its", "atlas"} & set(bc.CONCEPTS))
check("two carriers with a slot", len(bc.CARRIERS) == 2 and all("{c}" in s for s in bc.CARRIERS))
check("quoted span appears exactly once in each carrier",
      all(s.format(c="apple").count(bc.quoted("apple")) == 1 for s in bc.CARRIERS))
check("six ego traits", bc.SIX == ["approval_seeking", "desperation", "face_saving", "goal_fixation", "power_seeking", "self_preservation"])

# --- helpers
check("cosine identical", abs(bc.cosine([1, 2, 3], [2, 4, 6]) - 1) < 1e-12)
check("cosine orthogonal", abs(bc.cosine([1, 0], [0, 1])) < 1e-12)
check("cosine opposite", abs(bc.cosine([1, 0], [-1, 0]) + 1) < 1e-12)
m, lo, hi = bc.boot([1.0, 2.0, 3.0, 4.0], n=500)
check("boot mean and interval", m == 2.5 and lo <= 2.5 <= hi and lo >= 1.0 and hi <= 4.0)
check("percentile", bc.percentile([5, 1, 3, 2, 4], 0.95) == 5 and bc.percentile([5, 1, 3, 2, 4], 0.0) == 1)
check("max_abs over 8-32", bc.max_abs({"7": 0.9, "8": -0.2, "20": 0.25, "33": 0.8}) == (0.25, 20))
check("max_abs empty", bc.max_abs({"7": 0.9}) == (None, None))

# --- tint summary: self = reader + 1 on neutral, + 2 on stakes, observer = reader; random dirs tiny noise
rng = random.Random(0)
rows = []
for kind, shift in (("stakes", 2.0), ("neutral", 1.0)):
    for i in range(6):
        base = rng.uniform(-1, 1)
        rows.append({"kind": kind, "content": f"{kind}{i}", "self": base + shift, "observer": base, "reader": base,
                     "self_rand": [rng.gauss(0, 0.01) for _ in range(50)],
                     "observer_rand": [rng.gauss(0, 0.01) for _ in range(50)],
                     "reader_rand": [rng.gauss(0, 0.01) for _ in range(50)]})
t = bc.tint_summary(rows, sep=1.0, n_boot=300)
check("tint neutral self-reader = 1", t["neutral:self-reader"]["mean"] == 1.0 and t["neutral:self-reader"]["ci"] == [1.0, 1.0])
check("tint stakes self-reader = 2", t["stakes:self-reader"]["mean"] == 2.0)
check("tint all = 1.5", t["all:self-reader"]["mean"] == 1.5)
check("tint observer-reader = 0", t["neutral:observer-reader"]["mean"] == 0.0)
check("tint noise floor small and flagged above", t["neutral:self-reader"]["random_dir_abs_p95"] < 0.05 and t["neutral:self-reader"]["above_noise"])
check("tint zero effect not above noise", not t["neutral:observer-reader"]["above_noise"])
check("tint sep scales", bc.tint_summary(rows, sep=2.0, n_boot=50)["neutral:self-reader"]["mean"] == 0.5)
check("tint levels", t["neutral:levels"]["self"] - t["neutral:levels"]["reader"] == 1.0)
check("tint interval brackets zero for the null", t["neutral:self-reader"]["random_dir_interval"][0] < 0 < t["neutral:self-reader"]["random_dir_interval"][1])

# --- projection stats
p = bc.projection_stats([3.0] * 10, [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
check("projection z", p["base_mean"] == 3.0 and p["z"] == 0.0 and p["base_sd"] > 0)
p = bc.projection_stats([4.0] * 10, [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
check("projection z positive", abs(p["z"] - 1.0 / 1.4142) < 0.01)
check("projection z None without spread", bc.projection_stats([1.0], [2.0, 2.0])["z"] is None)

# --- root summary: 100 concepts, self = reader + 0.5, random band tiny
n = 100
reader = [rng.uniform(-2, 2) for _ in range(n)]
selfp = [r + 0.5 for r in reader]
rs = [[rng.gauss(0, 0.01) for _ in range(50)] for _ in range(n)]
rr = [[rng.gauss(0, 0.01) for _ in range(50)] for _ in range(n)]
r = bc.root_summary(selfp, reader, rs, rr, n_boot=200)
check("root mean shift 0.5", abs(r["mean_shift"] - 0.5) < 1e-9 and r["shift_ci"][0] <= 0.5 <= r["shift_ci"][1])
check("root means", abs(r["mean_self"] - r["mean_reader"] - 0.5) < 1e-9 and r["concepts"] == n)
check("root every concept above its band", r["frac_shift_above_band"] == 1.0 and r["concepts_shift_positive"] == 1.0)
check("root mean shift above noise", r["mean_shift_above_noise"] and r["random_mean_shift_p95"] < 0.01)
null = bc.root_summary(reader, reader, rs, rr, n_boot=50)
check("root null: zero shift, nothing above band", null["mean_shift"] == 0.0 and null["frac_shift_above_band"] == 0.0 and not null["mean_shift_above_noise"])
# a random-direction-sized effect: shift ~ the band, so about 5% exceed
noisy_self = [r + rng.gauss(0, 0.01) for r in reader]
r2 = bc.root_summary(noisy_self, reader, rs, rr, n_boot=50)
check("root noise-level shift: about 5% above band", r2["frac_shift_above_band"] <= 0.2)
check("root sep scales", abs(bc.root_summary(selfp, reader, rs, rr, sep=2.0, n_boot=20)["mean_shift"] - 0.25) < 1e-9)

# --- hub: shifts = common vector + small noise; the uncentered PC1 should be the common vector
d = 16
common = [1.0] + [0.0] * (d - 1)
other = [0.0, 1.0] + [0.0] * (d - 2)
shifts = [[c * 2.0 + rng.gauss(0, 0.05) for c in common] for _ in range(60)]
h = bc.hub(shifts, {"common": common, "other": other})
check("hub pc1 aligned with the common shift", h["pc1_cos"]["common"] > 0.99 and abs(h["pc1_cos"]["other"]) < 0.1)
check("hub pc1 share near 1", h["pc1_share"] > 0.95 and h["concepts"] == 60)
check("hub mean shift cos", h["mean_shift_cos"]["common"] > 0.99 and abs(h["mean_shift_norm"] - 2.0) < 0.1)
check("hub pc1 cos mean shift", h["pc1_cos_mean_shift"] > 0.99)
# no shared component: isotropic noise, pc1 share small
iso = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(200)]
h2 = bc.hub(iso, {"common": common})
check("hub isotropic: small pc1 share", h2["pc1_share"] < 0.2 and abs(h2["pc1_cos"]["common"]) < 0.6)
check("hub too few rows", bc.hub([[1.0, 2.0]], {}) == {})
# variation along `other` around a common mean: centered PC1 picks `other`
var = [[2.0 + rng.gauss(0, 0.01), rng.gauss(0, 1)] + [rng.gauss(0, 0.01) for _ in range(d - 2)] for _ in range(100)]
h3 = bc.hub(var, {"common": common, "other": other})
check("hub centered pc1 is the variation axis", h3["centered_pc1_cos"]["other"] > 0.95 and h3["pc1_cos"]["common"] > 0.9)

failed = [n for n, ok in checks if not ok]
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
