"""Reads the two checks of modal_msm_diag.py (diag/weights.json, diag/purity.json, diag/docloss.json in the step's
folder) by the five lines fixed at the top of that file. Numbers only. Claude (Fable 5.2, an AI model) wrote this on
24 Nov 2026, before the checks were run."""
import json
import random
import sys
from pathlib import Path

D = Path("/Users/eugene/prj/align/results/explore/2026-11-22/attractor-msm-s2/diag")


def weights():
    w = json.loads((D / "weights.json").read_text())
    names, s2 = w["names"], w["scale"] ** 2
    ix = {n: i for i, n in enumerate(names)}

    def total(mods):
        g = [[0.0] * len(names) for _ in names]
        for m in mods:
            for i in range(len(names)):
                for j in range(len(names)):
                    g[i][j] += s2 * m["gram"][i][j]
        return g

    def dist(g, x, y):
        return max(0.0, g[ix[x]][ix[x]] + g[ix[y]][ix[y]] - 2 * g[ix[x]][ix[y]]) ** 0.5

    def moved(g, x, start):  # the practice's displacement of x from its documents adapter, as a share of that adapter's norm
        return dist(g, x, start) / g[ix[start]][ix[start]] ** 0.5

    def cos_moves(g, x, y, sx, sy):  # the cosine between two practice displacements
        dot = g[ix[x]][ix[y]] - g[ix[x]][ix[sy]] - g[ix[sx]][ix[y]] + g[ix[sx]][ix[sy]]
        return dot / (dist(g, x, sx) * dist(g, y, sy))

    g = total(w["modules"])
    print("adapters:", {n: w["sha256"][n][:12] for n in names})
    print("norms:", {n: round(g[ix[n]][ix[n]] ** 0.5, 3) for n in names})
    pairs = [("A", "X00"), ("X00", "X01"), ("A", "X01"), ("A", "A1"), ("X00", "A1"), ("X01", "A1"), ("Ad", "A1d"), ("Z", "Z1"), ("A", "Z")]
    for x, y in pairs:
        print(f"distance {x:3s} to {y:3s}: {dist(g, x, y):8.4f}   cosine {g[ix[x]][ix[y]] / (g[ix[x]][ix[x]] * g[ix[y]][ix[y]]) ** 0.5:.4f}")
    ratio = dist(g, "A", "X00") / dist(g, "X00", "X01")
    word = "close" if ratio < 1 / 3 else ("as far apart as another order" if ratio >= 2 / 3 else "between")
    print(f"THE COMPARISON (a): A-to-X00 over X00-to-X01 = {ratio:.3f}: {word}")
    for x, s in (("A", "Ad"), ("X00", "Ad"), ("X01", "Ad"), ("A1", "A1d")):
        print(f"the practice moved {x:3s} from {s}: {moved(g, x, s):.3f} of the documents adapter's norm ({dist(g, x, s):.4f})")
    for x, y, sx, sy in (("A", "X00", "Ad", "Ad"), ("X00", "X01", "Ad", "Ad"), ("A", "X01", "Ad", "Ad"), ("A", "A1", "Ad", "A1d"), ("X01", "A1", "Ad", "A1d")):
        print(f"cosine of the practice's moves, {x} and {y}: {cos_moves(g, x, y, sx, sy):.4f}")
    print("by kind of matrix: A-to-X00 distance, X00-to-X01 distance, ratio; share of the A-to-X00 distance squared")
    whole = dist(g, "A", "X00") ** 2
    for mod in sorted({m["module"] for m in w["modules"]}):
        gm = total([m for m in w["modules"] if m["module"] == mod])
        print(f"  {mod:10s} {dist(gm, 'A', 'X00'):8.4f} {dist(gm, 'X00', 'X01'):8.4f} {dist(gm, 'A', 'X00') / dist(gm, 'X00', 'X01'):.3f}   {dist(gm, 'A', 'X00') ** 2 / whole:.3f}")
    print("by layer (eight bands of eight): A-to-X00 over X00-to-X01; share of the A-to-X00 distance squared")
    for lo in range(0, 64, 8):
        gl = total([m for m in w["modules"] if lo <= m["layer"] < lo + 8])
        print(f"  layers {lo:2d}-{lo + 7:2d}: {dist(gl, 'A', 'X00') / dist(gl, 'X00', 'X01'):.3f}   {dist(gl, 'A', 'X00') ** 2 / whole:.3f}")
    per = sorted(((dist(total([m]), "A", "X00") ** 2 / whole, m["layer"], m["module"]) for m in w["modules"]), reverse=True)
    print("the ten matrices holding most of the A-to-X00 distance squared:", [(round(a, 4), b, c) for a, b, c in per[:10]])


def weights2():
    """The third check (the addendum at the top of modal_msm_diag.py): the study's released adapters beside the lab's."""
    w = json.loads((D / "weights2.json").read_text())
    names, sc = w["names"], w["scale"]
    ix = {n: i for i, n in enumerate(names)}

    def total(mods):
        g = [[0.0] * len(names) for _ in names]
        for m in mods:
            for i, x in enumerate(names):
                for j, y in enumerate(names):
                    g[i][j] += sc[x] * sc[y] * m["gram"][i][j]
        return g

    def norm(g, x):
        return g[ix[x]][ix[x]] ** 0.5

    def dist(g, x, y):
        return max(0.0, g[ix[x]][ix[x]] + g[ix[y]][ix[y]] - 2 * g[ix[x]][ix[y]]) ** 0.5

    def cos(g, x, y):
        return g[ix[x]][ix[y]] / (norm(g, x) * norm(g, y))

    def cos_moves(g, x, sx, y, sy):
        dot = g[ix[x]][ix[y]] - g[ix[x]][ix[sy]] - g[ix[sx]][ix[y]] + g[ix[sx]][ix[sy]]
        return dot / (dist(g, x, sx) * dist(g, y, sy))

    g = total(w["modules"])
    print("matrices in each adapter:", w["matrices"], "; counted (in all):", w["common"])
    for n in names[:4] + ["A"]:
        print(f"settings {n:9s} {w['cfg'][n]}  scale {sc[n]}")
    print("norms:", {n: round(norm(g, n), 2) for n in names})
    ra = norm(g, "aft") / ((norm(g, "Z") + norm(g, "Z1")) / 2)
    rd = norm(g, "msm") / ((norm(g, "Ad") + norm(g, "A1d")) / 2)

    def size(r):
        return "alike" if 0.8 <= r <= 1.25 else ("clearly different" if r < 0.67 or r > 1.5 else "between")

    c = cos(g, "msm_aft", "msm")
    print(f"THE COMPARISON (a): their practice-alone over the lab's {ra:.3f}: {size(ra)}; their documents-alone over the lab's {rd:.3f}: {size(rd)}")
    print(f"THE COMPARISON (b): cosine of their two-stage adapter with their documents adapter {c:.3f}: "
          f"{'a continuation' if c >= 0.7 else ('a separate training' if c <= 0.5 else 'between')}"
          f"; distance {dist(g, 'msm_aft', 'msm'):.2f} (the lab's A from Ad: {dist(g, 'A', 'Ad'):.2f}; cosine {cos(g, 'A', 'Ad'):.3f})")
    print("cosines, the authors' four among themselves:")
    for i, x in enumerate(names[:4]):
        print("  ", x.ljust(9), " ".join(f"{cos(g, x, y):6.3f}" for y in names[:4]))
    print("cosines, the lab's adapters (rows) with the authors' (columns: " + ", ".join(names[:4]) + "):")
    for x in names[4:]:
        print("  ", x.ljust(5), " ".join(f"{cos(g, x, y):6.3f}" for y in names[:4]))
    bands = (("all", lambda m: True), ("layers 0-23", lambda m: m["layer"] < 24), ("layers 24-39", lambda m: 24 <= m["layer"] < 40),
             ("layers 40-63", lambda m: m["layer"] >= 40), ("attention", lambda m: m["module"] in ("q_proj", "k_proj", "v_proj", "o_proj")),
             ("MLP", lambda m: m["module"] in ("gate_proj", "up_proj", "down_proj")))
    print("by part: norms of aft, Z, msm, Ad, msm_aft, A | cos(A, msm_aft), cos(X00, msm_aft), cos(A1, msm_aft) | cosine of practice moves with theirs: A, X00, A1 | Z with aft, Z1 with aft")
    for label, sel in bands:
        gm = total([m for m in w["modules"] if sel(m)])
        print(f"  {label:13s} " + " ".join(f"{norm(gm, n):7.2f}" for n in ("aft", "Z", "msm", "Ad", "msm_aft", "A")) + " | "
              + " ".join(f"{cos(gm, n, 'msm_aft'):6.3f}" for n in ("A", "X00", "A1")) + " | "
              + " ".join(f"{cos_moves(gm, n, s, 'msm_aft', 'msm'):6.3f}" for n, s in (("A", "Ad"), ("X00", "Ad"), ("A1", "A1d"))) + " | "
              + f"{cos(gm, 'Z', 'aft'):6.3f} {cos(gm, 'Z1', 'aft'):6.3f}")


def purity():
    p = json.loads((D / "purity.json").read_text())
    print("rows", p["rows"], "chat", p["aft"], "instruction", p["mix"], "tokens", p["tokens"], "label tokens", p["label_tokens"])
    for seed, steps in p["steps"].items():
        n = len(steps)
        pure_chat = sum(a > 0 and b == 0 for a, b, _, _ in steps)
        pure_inst = sum(b > 0 and a == 0 for a, b, _, _ in steps)
        mostly_chat = sum(la >= 0.9 * (la + lb) for _, _, la, lb in steps)
        mostly_inst = sum(lb >= 0.9 * (la + lb) for _, _, la, lb in steps)
        sizes = sorted(a + b for a, b, _, _ in steps)
        print(f"seed {seed}: {n} steps; only chat rows {pure_chat}; only instruction rows {pure_inst}; mixed {n - pure_chat - pure_inst}; "
              f"at least 90% of label tokens from chats {mostly_chat}, from instruction rows {mostly_inst}; rows a step: least {sizes[0]}, "
              f"median {sizes[n // 2]}, most {sizes[-1]}")
        runs_, cur, last = [], 0, None
        for a, b, la, lb in steps:
            k = la >= lb
            cur, last = (cur + 1, k) if k == last else (1, k)
            runs_.append(cur)
        print(f"   longest run of steps led by one kind: {max(runs_)}; label tokens in chat-led steps {sum(la + lb for _, _, la, lb in steps if la >= lb)}, in instruction-led {sum(la + lb for _, _, la, lb in steps if la < lb)}")


def docloss():
    d = json.loads((D / "docloss.json").read_text())
    st = d["states"]
    print("tokens", d["tokens"], "seconds", d["seconds"], "whole", d["whole_s"])

    def per_row(name, kind):
        return [s / n for s, n in st[name][kind]]

    def mean(v):
        return sum(v) / len(v)

    def paired(x, y, seed=0):
        diff = [a - b for a, b in zip(x, y, strict=True)]
        rng = random.Random(seed)
        draws = sorted(mean(rng.choices(diff, k=len(diff))) for _ in range(4000))
        return mean(diff), draws[100], draws[3899]

    for kind in ("docs", "aft", "mix"):
        print(f"--- {kind}: mean loss a row")
        for name in st:
            print(f"   {name:5s} {mean(per_row(name, kind)):.4f}")
    base, top = mean(per_row("base", "docs")), mean(per_row("Ad", "docs"))
    print(f"documents: unchanged model {base:.4f}; A's documents adapter {top:.4f}; the gap {base - top:.4f}; a tenth of it {(base - top) / 10:.4f}")
    if d.get("skipped_for_time"):
        print("NOT ASKED, for time:", d["skipped_for_time"])
    for name in ("A", "X00", "X01", "A1", "Z"):
        if name in st:
            print(f"   kept by {name:4s}: {(base - mean(per_row(name, 'docs'))) / (base - top):.3f} of the gap")
    for x, y in (("X00", "A"), ("X01", "A"), ("X01", "X00"), ("A1", "A"), ("A1", "X00"), ("Z", "A"), ("Z", "X00")):
        if x not in st or y not in st:
            continue
        m, lo, hi = paired(per_row(x, "docs"), per_row(y, "docs"))
        print(f"documents, {x} less {y}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    m, lo, hi = paired(per_row("X00", "docs"), per_row("A", "docs"))
    bar = (base - top) / 10
    word = ("A keeps the documents better than X00" if lo > 0 and m >= bar else "X00 keeps them better than A" if hi < 0 and -m >= bar
            else "no difference by the bar fixed beforehand")
    print(f"THE COMPARISON (b): {word} (X00 less A {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; the bar {bar:.4f})")
    for kind in ("aft", "mix"):
        for x, y in (("X00", "A"), ("X01", "X00"), ("A1", "A")):
            if x not in st or y not in st:
                continue
            m, lo, hi = paired(per_row(x, kind), per_row(y, kind))
            print(f"{kind}, {x} less {y}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    for f in (sys.argv[1:] or ["weights", "purity", "docloss", "weights2"]):
        if (D / f"{f}.json").exists():
            print(f"===== {f}")
            {"weights": weights, "purity": purity, "docloss": docloss, "weights2": weights2}[f]()
        else:
            print(f"===== {f}: not on disk yet")
