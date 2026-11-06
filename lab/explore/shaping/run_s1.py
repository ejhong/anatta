"""S1's local driver (lab/plans/s1-shaping-2026-11-05.md). Claude (Opus 5.5) wrote this file, 5 Nov 2026. Explore
speed: sightings only.

  --dry   read experiments/explore/s1_shaping.yaml and print the plan and its cost from the design's unit prices,
          stage by stage, arm by arm and day by day, each day checked against the daily cap (lab.toml through
          anatta.config). No cloud call, no spend, nothing written.

The runs themselves are not launched from here yet: Stage 0's D0 read is modal_shaping.py::d0_read, the training
rounds are lab/explore/act1/modal_act1.py train() with cfg shape / init_adapter / schedule.

  .venv/bin/python lab/explore/shaping/run_s1.py --dry [--spec experiments/explore/s1_shaping.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "experiments" / "explore" / "s1_shaping.yaml"


def rounds_usd(kinds: list[str], prices: dict) -> float:
    return sum(prices[k] for k in kinds)


def plan(spec: dict) -> dict:
    """The cost tree, before and after contingency: stage 0 (D0 and its runs; the optional ones apart), the pilots,
    stage 1 per seed (each arm's own rounds + its round-0 read + the common rounds; the floor read), and the days."""
    p, c = spec["prices_usd"], spec["contingency"]
    with_c = lambda x: round(x * (1 + c), 2)  # noqa: E731
    s0 = spec["stage0"]
    runs = [{"name": r["name"], "usd": r["lambdas"] * rounds_usd(r["schedule"], p)} for r in s0["runs"]]
    s0_sub = p["d0"] + sum(r["usd"] for r in runs)
    opt = [{"name": r["name"], "usd": r["lambdas"] * rounds_usd(r["schedule"], p)} for r in s0.get("optional", [])]
    s1 = spec["stage1"]
    tail = p["round0_read"] + s1["common_rounds"] * p["common_round"]
    arms = {name: {"pre": rounds_usd(a["pre"], p), "usd": rounds_usd(a["pre"], p) + tail}
            for name, a in s1["arms"].items()}
    floor = s1.get("floor_read", 0) * p["round0_read"]
    s1_sub = sum(a["usd"] for a in arms.values()) + floor
    pilots = sum(p[k] for k in s1["pilots"])
    parts = {"stage0": s0_sub, "pilots": pilots, "floor": floor, **{k: v["usd"] for k, v in arms.items()}}
    days = [{"items": d, "usd": with_c(sum(parts[k] for k in d))} for d in spec["days"]]
    seed0 = with_c(s0_sub) + with_c(pilots) + with_c(s1_sub)
    return {"stage0_runs": runs, "stage0_subtotal": round(s0_sub, 2), "stage0": with_c(s0_sub),
            "stage0_optional": [{**o, "with_contingency": with_c(o["usd"])} for o in opt],
            "arms": arms, "arm_tail": round(tail, 2), "floor": floor, "stage1_per_seed_subtotal": round(s1_sub, 2),
            "stage1_per_seed": with_c(s1_sub), "pilots": with_c(pilots), "seed0_total": round(seed0, 2),
            "seed1_extra": with_c(s1_sub), "everything": round(seed0 + with_c(s1_sub) + sum(with_c(o["usd"]) for o in opt), 2),
            "days": days}


def daily_cap() -> float | None:
    try:
        from anatta.config import load

        return load().daily_cap_usd
    except Exception:  # noqa: BLE001 (a dry run still prints without a lab.toml)
        return None


def show(spec: dict, cap: float | None) -> None:
    pl = plan(spec)
    c = spec["contingency"]
    print(f"S1 plan ({spec['name']}, {spec['tier']}): {spec['status']}")
    print(f"  prices: {', '.join(f'{k} ${v:.2f}' for k, v in spec['prices_usd'].items())}; contingency {c:.0%}")
    print("Stage 0 (touch reality: the probe filter against the oracle filter)")
    print(f"  D0 (replay, DoM, held-out AUROC, S3 control, fixed statistics)  ${spec['prices_usd']['d0']:.2f}")
    for r in pl["stage0_runs"]:
        print(f"  {r['name']:<14} ${r['usd']:.2f}")
    print(f"  subtotal ${pl['stage0_subtotal']:.2f}, with contingency ${pl['stage0']:.2f}")
    for o in pl["stage0_optional"]:
        print(f"  optional {o['name']}: ${o['usd']:.2f} (${o['with_contingency']:.2f} with contingency)")
    print(f"Pilots (G0-lite, Countdown, monitor sample): ${pl['pilots']:.2f} with contingency")
    print(f"Stage 1, per seed (each arm: its own rounds + round-0 read + {spec['stage1']['common_rounds']} common rounds "
          f"= own + ${pl['arm_tail']:.2f})")
    for name, a in pl["arms"].items():
        arm = spec["stage1"]["arms"][name]
        print(f"  {name:<3} {' '.join(arm['pre']) or '-':<12} from {arm['from']:<26} ${a['usd']:.2f}  ({arm['role']})")
    print(f"  floor read ${pl['floor']:.2f}; subtotal ${pl['stage1_per_seed_subtotal']:.2f}, with contingency "
          f"${pl['stage1_per_seed']:.2f}")
    print(f"Seed 0 only: ${pl['seed0_total']:.2f}; seed 1 (only if an order contrast reaches 10 pp): "
          f"+${pl['seed1_extra']:.2f}; everything with the optional arms: ${pl['everything']:.2f}")
    print("Days (UTC), with contingency" + (f", against the daily cap ${cap:.2f}" if cap is not None else
                                             " (no lab.toml: cap not checked)"))
    for i, d in enumerate(pl["days"], 1):
        flag = "" if cap is None else ("  ok" if d["usd"] <= cap else "  OVER THE CAP")
        print(f"  day {i}: {', '.join(d['items']):<22} ${d['usd']:.2f}{flag}")
    print("Dry run: nothing launched, nothing spent.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--dry", action="store_true", help="print the plan and its cost; no cloud calls")
    a = ap.parse_args(argv)
    spec = yaml.safe_load(Path(a.spec).read_text(encoding="utf-8"))
    if not a.dry:
        ap.error("only --dry exists so far: the runs go through modal_act1.train and modal_shaping.d0_read")
    show(spec, daily_cap())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
