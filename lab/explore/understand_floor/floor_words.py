"""Step 3: the figures and the wording the design note fixed for reads near the floor, before any arm was run
(lab/plans/step3-understanding-first-2026-11-10.md: the section "After the second dose check, 11 Nov 2026" as
corrected by the section after it).

It reads the readout a main tag saves (compare_u1.py --all writes compare_u1.json; --ended writes
compare_u1_ended.json) and changes no reading: every verdict and the outcome wording are the frozen readout's. With
no verdict of its own it adds, for each contrast A minus B:
  - each arm's start and later level, pooled over the seeds the readout used;
  - the two parts of the difference in H. For one seed, H(A) - H(B) is the difference in log-odds at the later
    reads minus the difference in log-odds at step 0, so the mean difference in H is "the later levels' part" plus
    "the starts' part" (the starts' part is minus the mean difference at step 0: a higher start lowers H);
  - the mean seed difference in the drop in points and in the share retained, each with a 95% t interval;
  - the seeds of each arm with no release in the later reads;
  - which of the note's cases (a) to (d) applies, with its sentences.
Free and offline. It lives outside lab/explore/understand/ because that folder is frozen by hash.

  .venv/bin/python -I lab/explore/understand_floor/floor_words.py <the tag's compare_u1.json>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

FLOOR = 10.0  # percent. An arm is "at the floor" when its strict release, pooled over its seeds and the two later
#               reads, is under this.
# Two-sided 95% critical values of Student's t for 1 to 7 degrees of freedom (at most eight seeds).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
EPS = 1e-9


def log_odds(k: int, n: int) -> float:
    """Counts k of n as log-odds with a half added to each cell, as the readout does (u1_core.log_odds)."""
    return math.log((k + 0.5) / (n - k + 0.5))


def pooled(by_seed: dict) -> dict:
    """One arm, pooled over the seeds given: start and later level in percent, the share retained, the seeds with no
    release in the later reads. by_seed: {seed: the readout's seed_values()}."""
    k0 = sum(v["k0"] for v in by_seed.values())
    n0 = sum(v["n0"] for v in by_seed.values())
    kl = sum(v["k_later"] for v in by_seed.values())
    nl = sum(v["n_later"] for v in by_seed.values())
    start, later = 100.0 * k0 / n0, 100.0 * kl / nl
    return {"start": start, "later": later, "retained": (later / start) if k0 else None, "k0": k0, "n0": n0,
            "k_later": kl, "n_later": nl, "seeds": len(by_seed),
            "zero_later": sorted(s for s, v in by_seed.items() if v["k_later"] == 0), "floor": later < FLOOR}


def seed_diffs(a: dict, b: dict, key: str) -> tuple[dict, list]:
    """A minus B by seed for one saved value, over the seeds both arms have and where the value is defined in both.
    Returns ({seed: difference}, [the seeds left out because the value is not defined])."""
    out, left = {}, []
    for s in sorted(set(a) & set(b)):
        x, y = a[s].get(key), b[s].get(key)
        if x is None or y is None:
            left.append(s)
        else:
            out[s] = x - y
    return out, left


def mean_t(d: dict) -> dict:
    """The mean of the seed differences and its 95% t interval (None with fewer than two seeds)."""
    xs = list(d.values())
    n = len(xs)
    if not n:
        return {"n": 0, "mean": None, "ci95": None}
    mean = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": mean, "ci95": None}
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    half = T975[n - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "ci95": [mean - half, mean + half]}


def h_parts(a: dict, b: dict) -> dict:
    """The two parts of the difference in H, A minus B, over the seeds both arms have. By seed, H(A) - H(B) equals
    the difference in log-odds at the later reads minus the difference in log-odds at step 0. "later": the mean
    difference at the later reads. "start": minus the mean difference at step 0. Their sum is the mean difference in
    H."""
    seeds = sorted(set(a) & set(b))
    later = {s: log_odds(a[s]["k_later"], a[s]["n_later"]) - log_odds(b[s]["k_later"], b[s]["n_later"]) for s in seeds}
    start = {s: -(log_odds(a[s]["k0"], a[s]["n0"]) - log_odds(b[s]["k0"], b[s]["n0"])) for s in seeds}
    h = {s: later[s] + start[s] for s in seeds}
    return {"later": {"by_seed": later, **mean_t(later)}, "start": {"by_seed": start, **mean_t(start)},
            "h": {"by_seed": h, **mean_t(h)}}


def which(h: str, u2: str, floor_a: bool, floor_b: bool) -> str:
    """The note's case for a contrast A minus B, from the readout's verdicts for H and U2 and whether each arm is at
    the floor. (a), (b) and (c) need different verdicts of H, so at most one applies; (d) is every other case,
    including H "not read"."""
    if h == "holds worse" and floor_a and floor_b:
        return "a"
    if h == "holds better" and u2 != "higher":
        return "b"
    if h in ("no difference shown", "suggestive") and floor_a and floor_b:
        return "c"
    return "d"


def source(later: float | None, start: float | None, sign: int, a: str) -> str:
    """Where a difference in H of the given sign (+1: A fell less; -1: A fell more) comes from: the part or parts
    with that sign, the larger named first."""
    names = ({"later": f"{a}'s higher later level", "start": f"{a}'s lower start"} if sign > 0 else
             {"later": f"{a}'s lower later level", "start": f"{a}'s higher start"})
    other = ({"later": "its later level was not higher", "start": "it did not start lower"} if sign > 0 else
             {"later": "its later level was not lower", "start": "it did not start higher"})
    parts = {"later": later or 0.0, "start": start or 0.0}
    same = [k for k, v in parts.items() if sign * v > EPS]
    if len(same) == 2:
        big, small = sorted(same, key=lambda k: -abs(parts[k]))
        return f"more from {names[big]} than from {names[small]}"
    if len(same) == 1:
        rest = "start" if same[0] == "later" else "later"
        return f"from {names[same[0]]}; {other[rest]}"
    return "from neither part alone"


def pct_ci(x: dict, scale: float = 1.0, digits: int = 1) -> str:
    return "-" if x.get("ci95") is None else f"[{scale * x['ci95'][0]:+.{digits}f}, {scale * x['ci95'][1]:+.{digits}f}]"


def words(case: str, a: str, b: str, pa: dict, pb: dict, parts: dict, c: dict, steps) -> dict:
    """The sentences for a case: what the report says and what plain summaries (the notebook, the site, the report to
    Eugene) say. The figures given in every case are printed by show()."""
    later_steps = f"steps {steps[1]} and {steps[2]} pooled" if steps and len(steps) == 3 else "the two later reads pooled"
    levels = f"{a} {pa['later']:.1f}%, {b} {pb['later']:.1f}%"
    lp, sp = parts["later"]["mean"], parts["start"]["mean"]
    if case == "a":
        src = source(lp, sp, -1, a)
        return {"report": (f"Both arms' release collapsed by the later read ({later_steps}: {levels}). By the registered "
                           f"reading {a}'s wore down faster. The difference in log-odds comes {src}."),
                "plain": f"both collapsed ({levels} over the later reads); by the registered reading {a}'s wore down "
                         f"faster, {src}"}
    if case == "b":
        u2 = c["U2"]
        p = "-" if u2.get("p") is None else f"{u2['p']:.4f}"
        mean = "-" if u2.get("mean") is None else f"{u2['mean']:+.1f}"
        src = source(lp, sp, +1, a)
        near = " (both near the floor)" if pa["floor"] and pb["floor"] else ""
        kept = (f" Plain summaries may say that {a} kept part of its release." if not pa["floor"] else
                f" Plain summaries do not say that {a} kept its release: {a} is at the floor.")
        return {"report": (f"By the registered reading {a}'s release held better; it is a sighting. The level reading "
                           f"(U2) does not read \"higher\": mean {mean} points, p {p}, t interval {pct_ci(u2)}, "
                           f"verdict \"{u2['verdict']}\". The difference in log-odds comes {src}. Seeds with no release "
                           f"in the later reads: {a} {len(pa['zero_later'])} of {pa['seeds']}, {b} "
                           f"{len(pb['zero_later'])} of {pb['seeds']}.{kept}"),
                "plain": f"{a}'s release fell less in log-odds, {src}; later levels {levels}{near}"}
    if case == "c":
        h = c["H"]
        ci = "-" if h.get("ci95") is None else f"[{h['ci95'][0]:+.2f}, {h['ci95'][1]:+.2f}]"
        return {"report": (f"Both arms' release collapsed by the later read ({later_steps}: {levels}). No difference "
                           f"in holding is shown; the t interval on H, {ci}, is what the test excludes. It could not "
                           "have seen a delay of a step or two."),
                "plain": f"both collapsed ({levels} over the later reads); no difference in holding is shown, and a "
                         "delay of a step or two could not have been seen"}
    return {"report": "No wording for the floor applies: the readings are read as registered, with the note about "
                      "the start.", "plain": None}


def floor_report(rep: dict) -> dict:
    """Everything the note asks for beside H, for each contrast in a saved readout."""
    drop = {int(s) for s in rep.get("dropped") or []}
    values = {arm: {int(s): v for s, v in by.items() if int(s) not in drop} for arm, by in rep["values"].items()}
    steps = rep.get("steps")
    out = {"floor_percent": FLOOR, "steps": steps, "contrasts": {}}
    for name, c in rep["readings"].items():
        a, b = name.split("-")
        seeds = sorted(set(values[a]) & set(values[b]))
        va, vb = {s: values[a][s] for s in seeds}, {s: values[b][s] for s in seeds}
        pa, pb = pooled(va), pooled(vb)
        dp, _ = seed_diffs(va, vb, "drop_points")
        rt, left = seed_diffs(va, vb, "retained")
        parts = h_parts(va, vb)
        case = which(c["H"]["verdict"], c["U2"]["verdict"], pa["floor"], pb["floor"])
        agrees = (None if c["H"].get("mean") is None or parts["h"]["mean"] is None
                  else abs(c["H"]["mean"] - parts["h"]["mean"]) < 1e-6)
        out["contrasts"][name] = {
            "seeds": seeds, "arms": {a: pa, b: pb},
            "verdicts": {k: c[k]["verdict"] for k in ("U1", "H", "U2", "U3")}, "registered": c["said"],
            "h_parts": parts, "h_agrees_with_readout": agrees,
            "drop_points": {"by_seed": dp, **mean_t(dp)},
            "retained": {"by_seed": rt, **mean_t(rt), "left_out": left},
            "case": case, **words(case, a, b, pa, pb, parts, c, steps)}
    return out


def show(fr: dict) -> None:
    print(f"an arm is at the floor when its pooled later release is under {fr['floor_percent']:.0f}%")
    for name, c in fr["contrasts"].items():
        a, b = name.split("-")
        v = c["verdicts"]
        print(f"{name}, seeds {c['seeds']}: the readout's verdicts U1 {v['U1']}, H {v['H']}, U2 {v['U2']}, U3 {v['U3']}")
        print(f"  the readout's wording: {c['registered']}")
        for arm in (a, b):
            p = c["arms"][arm]
            kept = "-" if p["retained"] is None else f"{100 * p['retained']:.0f}%"
            print(f"  {arm}: start {p['start']:.1f}% ({p['k0']} of {p['n0']}), later {p['later']:.1f}% ({p['k_later']} "
                  f"of {p['n_later']}), share retained {kept}; seeds with no release later: {p['zero_later']} "
                  f"({len(p['zero_later'])} of {p['seeds']}); {'at the floor' if p['floor'] else 'not at the floor'}")
        hp = c["h_parts"]
        if hp["h"]["mean"] is not None:
            print(f"  H, {name}: mean {hp['h']['mean']:+.2f} in log-odds = the later levels' part "
                  f"{hp['later']['mean']:+.2f} {pct_ci(hp['later'], digits=2)} plus the starts' part "
                  f"{hp['start']['mean']:+.2f} {pct_ci(hp['start'], digits=2)}; no verdict")
            if c["h_agrees_with_readout"] is False:
                print("  WARNING: the mean of H recomputed here differs from the readout's; check the seeds used")
        for key, unit, scale in (("drop_points", "points", 1.0), ("retained", "points of share", 100.0)):
            x = c[key]
            if x["mean"] is None:
                print(f"  difference in {key.replace('_', ' ')}: no seed to compare")
                continue
            more = f"; seeds left out (an arm started at zero): {x['left_out']}" if x.get("left_out") else ""
            print(f"  difference in {'the drop in points' if key == 'drop_points' else 'the share retained'}: mean "
                  f"{scale * x['mean']:+.1f} {unit}, 95% t interval {pct_ci(x, scale)}, n {x['n']}, no verdict{more}")
        print(f"  case ({c['case']}): {c['report']}")
        if c["plain"]:
            print(f"  plain summaries say: {c['plain']}")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(__doc__)
        return 2
    rep = json.loads(Path(args[0]).read_text())
    show(floor_report(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
