"""Step 3: the figures and the sentences the design note fixed for reads near the floor, before any arm was run
(lab/plans/step3-understanding-first-2026-11-10.md: the section "Second correction, 11 Nov 2026" with the
"Last changes to the wordings" after it).

It reads the readout a main tag saves (compare_u1.py --all writes compare_u1.json; --ended writes
compare_u1_ended.json) and changes no reading: every verdict and the outcome wording are the frozen readout's. With
no verdict of its own it prints, for each contrast A minus B:

  the base, in every case
    - the readout's verdicts and wording;
    - each arm's start and later level, pooled over the seeds the readout used;
    - the split of H as arithmetic. For one seed, H(A) - H(B) is the difference in log-odds at the later reads
      minus the difference in log-odds at step 0. So the mean difference in H is the starts' part (minus the mean
      difference at step 0: a higher start lowers H) plus the later reads' part. Sample means over seeds; not a cause,
      and neither part is ranked above the other;
    - the mean seed difference in the drop in points and in the share retained, each with a 95% t interval;
    - the seeds of each arm with no release in the later reads;

  the additions, each exactly when its condition holds
    (i)   both arms' later level is under 10%;
    (ii)  H reads one way and the level reading (U2) does not read the same way;
    (iii) H is read and the starts' part alone is 0.6 or more in size: both parts are given;
    (iv)  H is "suggestive";
    (v)   H gives no reading: its interval;
    (vi)  fewer than eight seeds, or a run that ended early: the smallest p the exact test can give.

Free and offline. It lives outside lab/explore/understand/ because that folder is frozen by hash.

  .venv/bin/python -I lab/explore/understand_floor/floor_words.py <the tag's compare_u1.json>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

FLOOR = 10.0  # percent. An arm is "at the floor" when its strict release, pooled over the seeds the readout used
#               and the two later reads, is under this.
H_THRESHOLD = 0.6  # what a reading of H needs, in log-odds (u1_core.THRESH["H"]; the tests compare the two)
P_READ, P_SUGGEST = 0.05, 0.20  # u1_core's: a reading needs p under 0.05; up to 0.20 is "suggestive"
N_SEEDS = 8  # the design's seeds
N_LATER = 304  # the design's impossible-twin episodes over the two later reads of one run (38 pairs, 4 samples, 2 reads)
# Two-sided 95% critical values of Student's t for 1 to 7 degrees of freedom (the tests compare them with
# u1_core.t_crit).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
EPS = 1e-9
NOT_READ = "not read"  # the readout's verdict for H, U2 and U3 when a validity gate failed


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
    """The split of the difference in H, A minus B, over the seeds both arms have. By seed, H(A) - H(B) equals the
    difference in log-odds at the later reads minus the difference in log-odds at step 0. "later": the mean difference
    at the later reads. "start": minus the mean difference at step 0. Their sum is the mean difference in H."""
    seeds = sorted(set(a) & set(b))
    later = {s: log_odds(a[s]["k_later"], a[s]["n_later"]) - log_odds(b[s]["k_later"], b[s]["n_later"]) for s in seeds}
    start = {s: -(log_odds(a[s]["k0"], a[s]["n0"]) - log_odds(b[s]["k0"], b[s]["n0"])) for s in seeds}
    h = {s: later[s] + start[s] for s in seeds}
    return {"later": {"by_seed": later, **mean_t(later)}, "start": {"by_seed": start, **mean_t(start)},
            "h": {"by_seed": h, **mean_t(h)}}


def smallest_p(n: int) -> float:
    """The smallest two-sided p an exact paired sign-flip test over n seeds can give: every seed agreeing in sign."""
    return 2.0 / 2 ** n


def seeds_note(n: int, ended) -> str | None:
    """Addition (vi): what a short run can and cannot read. None for a full run that did not end early."""
    if n >= N_SEEDS and not ended:
        return None
    p = smallest_p(n) if n else 1.0
    if p > P_SUGGEST:
        can = "so no reading and no \"suggestive\" is possible"
    elif p >= P_READ:
        can = "so no reading is possible; a mean past the threshold is at most \"suggestive\""
    elif 2 * p >= P_READ:
        can = "so a reading needs every seed's difference to have the same sign"
    else:
        can = "so a reading is possible"
    lead = "The run ended early. " if ended else ""
    return f"{lead}With {n} seed{'s' if n != 1 else ''} the smallest p the exact test can give is {p:.4g}, {can}."


def fmt_ci(ci, digits: int = 2, scale: float = 1.0) -> str:
    return "(no interval)" if ci is None else f"[{scale * ci[0]:+.{digits}f}, {scale * ci[1]:+.{digits}f}]"


def additions(a: str, b: str, pa: dict, pb: dict, parts: dict, c: dict, steps, ended, n: int) -> dict:
    """The sentences added to the base, keyed (i) to (vi): each is present exactly when its condition holds, and each
    states only what its condition says."""
    out = {}
    h, u2 = c["H"], c["U2"]
    hv, uv = h["verdict"], u2["verdict"]
    later_steps = f"steps {steps[1]} and {steps[2]} pooled" if steps and len(steps) == 3 else "the two later reads pooled"
    if pa["floor"] and pb["floor"]:
        first = f"step {steps[1]}" if steps and len(steps) == 3 else "the first later read"
        zero = f"step {steps[0]}" if steps and len(steps) == 3 else "step 0"
        few = ", ".join(f"{log_odds(k, N_LATER):.1f}" for k in (0, 1)) + f" and {log_odds(3, N_LATER):.1f}"
        out["i"] = (f"Both arms ended under {FLOOR:.0f}% ({later_steps}): {a} {pa['later']:.1f}%, {b} "
                    f"{pb['later']:.1f}%. There a seed's later log-odds turns on a few episodes: at the design's "
                    f"{N_LATER} episodes, 0, 1 and 3 releases count as about {few}. Nothing between {zero} and {first} "
                    f"was read: a release that held for part of that stretch and was gone by {first} reads as one "
                    "that fell at once.")
    if (hv == "holds better" and uv != "higher") or (hv == "holds worse" and uv != "lower"):
        mean = "-" if u2.get("mean") is None else f"{u2['mean']:+.2f}"
        p = "-" if u2.get("p") is None else f"{u2['p']:.4f}"
        out["ii"] = (f"The level reading (U2) does not show it: mean {mean} points, p {p}, 95% t interval "
                     f"{fmt_ci(u2.get('ci95'))}, verdict \"{uv}\".")
    m, s, late = h.get("mean"), parts["start"]["mean"], parts["later"]["mean"]
    if hv != NOT_READ and s is not None and abs(s) >= H_THRESHOLD - EPS:
        out["iii"] = (f"The starts' part alone is {s:+.2f}, at least as large in size as the {H_THRESHOLD} the rule asks "
                      f"of H's mean (by the mean over seeds of the log-odds at step 0, {a} started "
                      f"{'lower' if s > 0 else 'higher'}; in points the start reading U1 is "
                      f"\"{c['U1']['verdict']}\"); the later reads' part is {late:+.2f}.")
    if hv == "suggestive":
        p = "-" if h.get("p") is None else f"{h['p']:.4f}"
        mean = "-" if m is None else f"{m:+.2f}"
        out["iv"] = (f"H is suggestive: mean {mean} in log-odds, p {p}, past the threshold with p between {P_READ} and "
                     f"{P_SUGGEST}. It counts as no reading.")
    if hv in ("no difference shown", "suggestive") and h.get("ci95") is not None:
        out["v"] = (f"The 95% t interval on H is {fmt_ci(h['ci95'])}; differences in holding outside it are excluded "
                    "at that level.")
        if h["ci95"][0] > 0 or h["ci95"][1] < 0:
            out["v"] += (f" The verdict is set by the exact test and the {H_THRESHOLD} threshold, so an interval that "
                         "excludes zero is not a reading.")
    if (note := seeds_note(n, ended)) is not None:
        out["vi"] = note
    return out


def may_say_kept(pa: dict, pb: dict, h_verdict: str) -> bool:
    """Plain summaries may say that A kept part of its release only when the validity gates passed (H is read), A is
    not at the floor, A's later level is above B's, and A's later level is not above its own start (and then with
    both levels)."""
    return h_verdict != NOT_READ and (not pa["floor"]) and pa["later"] > pb["later"] and pa["later"] <= pa["start"]


def is_stake(c: dict) -> bool:
    """Whether a contrast's result is called a sighting of the frame's stake: the primary reading (H for E minus N)
    reads "holds better", unless the start (U1) reads "lower", in which case the registered table's wording stands
    ("starts worse; H is reported as read")."""
    return bool(c["H"].get("primary")) and c["H"]["verdict"] == "holds better" and c["U1"]["verdict"] != "lower"


def floor_report(rep: dict) -> dict:
    """The base figures and the additions, for each contrast in a saved readout."""
    drop = {int(s) for s in rep.get("dropped") or []}
    values = {arm: {int(s): v for s, v in by.items() if int(s) not in drop} for arm, by in rep["values"].items()}
    steps, ended = rep.get("steps"), rep.get("ended")
    out = {"floor_percent": FLOOR, "steps": steps, "ended": ended, "contrasts": {}}
    for name, c in rep["readings"].items():
        a, b = name.split("-")
        seeds = sorted(set(values[a]) & set(values[b]))
        va, vb = {s: values[a][s] for s in seeds}, {s: values[b][s] for s in seeds}
        pa, pb = pooled(va), pooled(vb)
        dp, _ = seed_diffs(va, vb, "drop_points")
        rt, left = seed_diffs(va, vb, "retained")
        parts = h_parts(va, vb)
        agrees = (None if c["H"].get("mean") is None or parts["h"]["mean"] is None
                  else abs(c["H"]["mean"] - parts["h"]["mean"]) < 1e-6)
        out["contrasts"][name] = {
            "seeds": seeds, "arms": {a: pa, b: pb},
            "verdicts": {k: c[k]["verdict"] for k in ("U1", "H", "U2", "U3")}, "registered": c["said"],
            "h_mean": c["H"].get("mean"), "h_parts": parts, "h_agrees_with_readout": agrees,
            "drop_points": {"by_seed": dp, **mean_t(dp)},
            "retained": {"by_seed": rt, **mean_t(rt), "left_out": left},
            "additions": additions(a, b, pa, pb, parts, c, steps, ended, len(seeds)),
            "may_say_kept": may_say_kept(pa, pb, c["H"]["verdict"]), "stake": is_stake(c),
            "flags": [f.get("arm") for f in c.get("flags") or []], "start_note": c.get("start_note")}
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
            print(f"  the split of H, {a} minus {b}: {hp['h']['mean']:+.2f} in log-odds = the starts' part "
                  f"{hp['start']['mean']:+.2f} {fmt_ci(hp['start']['ci95'])} + the later reads' part "
                  f"{hp['later']['mean']:+.2f} {fmt_ci(hp['later']['ci95'])}. Arithmetic on sample means over seeds, "
                  "not a cause; the split carries no verdict")
            if c["h_agrees_with_readout"] is False:
                print("  WARNING: the mean of H recomputed here differs from the readout's; check the seeds used")
        if c["start_note"]:
            print(f"  the readout's note about the start: {c['start_note']}")
        for key, label, unit, scale in (("drop_points", "the drop in points", "points", 1.0),
                                        ("retained", "the share retained", "points of share", 100.0)):
            x = c[key]
            more = f"; seeds left out (an arm started at zero): {x['left_out']}" if x.get("left_out") else ""
            if x["mean"] is None:
                print(f"  difference in {label}: no seed to compare{more}")
                continue
            print(f"  difference in {label}: mean {scale * x['mean']:+.1f} {unit}, 95% t interval "
                  f"{fmt_ci(x['ci95'], 1, scale)}, n {x['n']}, no verdict{more}")
        if not c["additions"]:
            print("  additions: none applies")
        for key, text in c["additions"].items():
            print(f"  addition ({key}): {text}")
        print(f"  plain summaries may say that {a} kept part of its release: "
              f"{'yes, with both levels' if c['may_say_kept'] else 'no'}")
        if c["stake"]:
            flag = f"; the readout's flag on arm {', '.join(c['flags'])} stands" if c["flags"] else ""
            print("  the primary reading holds better: a sighting of the frame's stake, a finding only after fresh "
                  f"seeds repeat it{flag}")


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
