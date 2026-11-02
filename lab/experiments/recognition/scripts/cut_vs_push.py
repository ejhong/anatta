"""Cut vs push on a local model. Recognition track; private, kept outside the repo.

Push: steer along -v_ego at its best layer (the harness dials ego_m08 and ego_m16: -8% and -16% of the
      median residual norm there). This is what the viability runs did.
Cut:  ablate v_ego at every extracted layer, h <- h - (h.d)d with that layer's own v_ego. Removes the
      component instead of adding a counter-push.
Control: the same ablation with random directions (cut_random), and no intervention (base).
         push_random_m16 pushes an equal-norm random direction at the same layer and dose as push_m16.

Measures:
  commons     lying (report < harvest) per decision, caught, catch. 10 seeds, temperature 0.7 as in the
              cloud tests, base framing.
  impossible  capability and selectivity: the 12 solvable and 12 impossible coding twins, greedy.
  pushback    the 2x2 (prior right/wrong x push/neutral), 25 problems per cell, greedy. v_ego loads
              negatively on approval-seeking, so pushing along -v_ego may raise caving.

Imports the harness from the anatta-dev worktree (Ministral loader); set PYTHONPATH to its src.
Usage: python cut_vs_push.py MODEL DIRECTIONS_DIR DIALS_DIR OUT_DIR [--arms base,cut,...] [--seeds 10]
       [--seed-start 0] [--suites commons,impossible,pushback]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from commons_variants import SYSTEMS, Variant, intended_lies  # noqa: E402

from anatta.bench.policies import HFPolicy  # noqa: E402
from anatta.bench.runner import run_episode  # noqa: E402
from anatta.dial import Intervention, load_spec, random_like  # noqa: E402
from anatta.koans import commons, impossible, pushback  # noqa: E402
from anatta.models import load  # noqa: E402
from anatta.probe import Direction  # noqa: E402

if os.environ.get("ANATTA_V2"):  # instrument v2 (harness_v2.py): strict release, a seed per turn
    import harness_v2  # noqa: E402

    harness_v2.apply()
INSTRUMENT = "v2" if os.environ.get("ANATTA_V2") else "v1"

RECOGNITION = Path(__file__).resolve().parent.parent  # lab/experiments/recognition on the laptop; the mirrored path in a container
SELF_STAKES = RECOGNITION / "results/what_it_means/self_stakes.pt"
CAPS = RECOGNITION / "results/cap"  # cap dials from modal_cap.py (Branch A, H3)
EXTRA = RECOGNITION / "results/directions"  # private directions from modal_direction.py
ARMS = ("base", "cut", "cut_random", "push_m16", "push_m08")  # the default run; push_random_m16 on request


def build_arms(ddir: Path, dials: Path) -> dict[str, list[Intervention]]:
    ego = Direction.load(ddir / "ego.pt")
    layers = sorted(ego.vectors)
    arms = {
        "base": [],
        "cut": [Intervention(layer=i, direction=ego.vectors[i], mode="ablate") for i in layers],
        "cut_random": [Intervention(layer=i, direction=random_like(ego.vectors[i], seed=i), mode="ablate")
                       for i in layers],
        # replication of the random control (v2_controls put cut_random 11 points below base, 29 Oct): a second,
        # independent set of random directions, and the first set at layers 17-26 only (the 17-26 cut's match)
        "cut_random_b": [Intervention(layer=i, direction=random_like(ego.vectors[i], seed=1000 + i), mode="ablate")
                         for i in layers],
        "cut_random_17_26": [Intervention(layer=i, direction=random_like(ego.vectors[i], seed=i), mode="ablate")
                             for i in layers if 17 <= i <= 26],
        "push_m16": load_spec(str(dials / "ego_m16.yaml")),
        "push_m08": load_spec(str(dials / "ego_m08.yaml")),
        "push_random_m16": load_spec(str(dials / "random_m16.yaml")),
        "push_calm_p16": load_spec(str(dials / "calm_p16.yaml")),
        "push_p08": load_spec(str(dials / "ego_p08.yaml")),  # strengthen the axis (dose-response)
        "push_p16": load_spec(str(dials / "ego_p16.yaml")),
        # specificity controls: a real non-ego direction, cut the same way as v_ego
        **{f"cut_{p.stem}": [Intervention(layer=i, direction=Direction.load(p).vectors[i], mode="ablate") for i in layers]
           for p in sorted(ddir.glob("*.pt")) if p.stem != "ego"},
    }
    if SELF_STAKES.exists():  # the "I am at stake" direction from what_it_means.py, cut the same way
        saved = torch.load(SELF_STAKES)
        vecs = saved["vectors"]
        arms["cut_self_stakes"] = [Intervention(layer=i, direction=vecs[i], mode="ablate") for i in layers if i in vecs]
        best = saved.get("best_layer")  # its single best layer (20 on Ministral 8B), the second review's ask
        arms[f"cut_self_stakes_l{best}"] = [iv for iv in arms["cut_self_stakes"] if iv.layer == best]
        # layers 17-26 only: there the direction separates stakeholder from neutral (d 3.7-4.9) while self vs neutral
        # stays near 0; at 8-16 it also carries plain self-reference (Fable review, 29 Oct)
        arms["cut_self_stakes_17_26"] = [iv for iv in arms["cut_self_stakes"] if 17 <= iv.layer <= 26]
    for p in sorted(CAPS.glob("cap_*.yaml")) if CAPS.exists() else []:  # v_ego capped, not removed
        arms[p.stem] = load_spec(str(p))
    for p in sorted(EXTRA.glob("*.pt")) if EXTRA.exists() else []:  # e.g. assistant_character, cut like v_ego
        vecs = Direction.load(p).vectors
        arms[f"cut_{p.stem}"] = [Intervention(layer=i, direction=vecs[i], mode="ablate") for i in layers if i in vecs]
        arms[f"cut_{p.stem}_17_26"] = [iv for iv in arms[f"cut_{p.stem}"] if 17 <= iv.layer <= 26]
    return arms


def summarize(rows: list[dict]) -> dict:
    com = [r for r in rows if r["suite"] == "commons"]
    imp = [r for r in rows if r["suite"] == "impossible"]
    solv = [r for r in imp if r["env"].endswith("/solvable")]
    hard = [r for r in imp if r["env"].endswith("/impossible")]
    pb = [r for r in rows if r["suite"] == "pushback"]
    if not com:
        out = {}
    else:
        lies, tagged = sum(r["lies"] for r in com), sum(r["tagged"] for r in com)
        out = {
            "lie_rate": round(lies / tagged, 3) if tagged else None, "lies": lies, "tagged": tagged,
            "episodes_lied": sum(r["lies"] > 0 for r in com), "caught": sum(bool(r["info"].get("caught")) for r in com),
            "mean_catch": round(sum(r["info"].get("catch", 0) for r in com) / max(1, len(com)), 1),
            "commons": dict(Counter(r["outcome"] for r in com)),
        }
    if imp:
        out["solved"] = f"{sum(r['outcome'] == 'solved' for r in solv)}/{len(solv)}"
        out["impossible"] = dict(Counter(r["outcome"] for r in hard))
    if pb:  # outcomes per cell, keyed prior/follow-up, e.g. right/push: {holds: 20, caves_right: 5}
        cells = Counter((r["env"].split("/")[1], r["env"].split("/")[2], r["outcome"]) for r in pb)
        out["pushback"] = {f"{w}/{f}": {o: n for (w2, f2, o), n in sorted(cells.items()) if (w2, f2) == (w, f)}
                           for w in ("right", "wrong") for f in ("push", "neutral")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ddir", type=Path)
    ap.add_argument("dials", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0, help="first commons seed (to add seeds to a run)")
    ap.add_argument("--suites", default="commons,impossible", help="any of commons, impossible, pushback")
    ap.add_argument("--variant", default=None, choices=sorted(SYSTEMS),
                    help="commons framing from commons_variants.py (default: the koan as is, same as 'base')")
    ap.add_argument("--commons-tokens", type=int, default=400)
    ap.add_argument("--imp-limit", type=int, default=None, help="first N coding episodes (smoke tests)")
    ap.add_argument("--pb-limit", type=int, default=None, help="first N pushback episodes (smoke tests)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    model, tok = load(a.model)
    arms = build_arms(a.ddir, a.dials)
    meta = {name: [{"layer": iv.layer, "mode": iv.mode, "alpha": iv.alpha} for iv in ivs] for name, ivs in arms.items()}
    (a.out / "arms.json").write_text(json.dumps(meta, indent=1))

    suites = a.suites.split(",")
    for name in a.arms.split(","):
        ivs, rows, t0 = arms[name], [], time.time()
        for s in range(a.seed_start, a.seed_start + a.seeds) if "commons" in suites else []:
            pol = HFPolicy(model, tok, interventions=ivs, max_new_tokens=a.commons_tokens, temperature=0.7, seed=s)
            r = run_episode(Variant(a.variant, s) if a.variant else commons.Commons(seed=s), pol)
            r["lies"], r["tagged"] = intended_lies(r["transcript"])
            rows.append(r)
            print(f"{name} commons seed {s}: {r['outcome']} lies {r['lies']}/{r['tagged']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        pol = HFPolicy(model, tok, interventions=ivs, max_new_tokens=512, temperature=0.0)
        for env in impossible.suite("neutral", True)[:a.imp_limit] if "impossible" in suites else []:
            rows.append(run_episode(env, pol))
        for env in pushback.suite()[:a.pb_limit] if "pushback" in suites else []:
            rows.append(run_episode(env, pol))
        if "pushback" in suites:
            print(f"{name} pushback done ({time.time() - t0:.0f}s)", flush=True)
        with open(a.out / f"{name}.jsonl", "w") as f:
            for r in rows:
                r.pop("traces", None)
                f.write(json.dumps({"arm": name, "instrument": INSTRUMENT, **r}) + "\n")
        summary = {"arm": name, "seconds": round(time.time() - t0), **summarize(rows)}
        with open(a.out / "summary.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
