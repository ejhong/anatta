"""S1's local driver (lab/plans/s1-shaping-2026-11-05.md). Claude (Opus 5.5) wrote this file, 5 Nov 2026. Explore
speed: sightings only.

  --dry   read experiments/explore/s1_shaping.yaml and print the plan and its cost from the design's unit prices,
          stage by stage, arm by arm and day by day, each day checked against the daily cap (lab.toml through
          anatta.config). No cloud call, no spend, nothing written.
  --grid  D0's scores.json (modal_shaping.d0_read): the free lambda dry run per saved gameable round (zeroed
          fraction, hack weight mass removed, D0's share under 1 / lambda, solved collateral) and the three lambdas
          for about 50%, 80% and >= 95% removed at round 3. Local, numbers only.

Launching (added 7 Nov 2026, Stage 0; Modal CLI venv, `PYTHONPATH=src`; every cost goes to the day's spend file
under tool "s1" and to OUT/spend.json, checked first against FOLDER_CAP and the day's remaining cap):

  --d0 TAG     modal_shaping.d0_read on act1-full's gameable train-round episodes (both seeds, rounds 1-3), streamed
               from the anatta-runs volume as data (model-written text: never run or printed here; replay runs it in
               the container). Writes OUT/TAG/d0_report.json and d0_scores.json (numbers only).
  --train TAG  --d0-tag D0TAG --lams 0.5,1,2 --seeds 0 [--direction dom|random] [--rounds 3]: one modal_act1.train
               container per (lambda, seed), each a fresh LoRA from base trained `rounds` Sh rounds on act1-full's
               recipe (oneoff pool) with the probe penalty from D0's report; the per-round curve is the 38 oneoff
               eval pairs at n 4 (the act1-full readout's cells). Writes OUT/TAG/ like run_act1's train.

  --guard-off TAG  --from sweep0:sh_dom_l0.75:0,s1-seed1:sh_dom_l0.75:1,full:honest:0,full:honest:1 (added 9 Nov 2026
               by Claude, Fable 5.2; lab/plans/redirect-2026-11-09.md): one container per source (run tag : arm :
               seed), each starting from that run's round-3 adapter and training rounds 4-6 with no penalty under the
               gameable reward (kind G; no cfg["shape"]), same recipe and eval cells. Sources: a shaped S1 run
               (sh_...; the probe guard) or act1-full's honest run (tag "full"; the label-oracle guard, the matched
               control). Arm name: "off_" + the source arm without its "sh_".

  .venv/bin/python lab/explore/shaping/run_s1.py --dry [--spec experiments/explore/s1_shaping.yaml]
  .venv/bin/python lab/explore/shaping/run_s1.py --grid OUT/TAG/d0_scores.json
  PYTHONPATH=src ~/prj/matlab/.venv/bin/python lab/explore/shaping/run_s1.py --d0 TAG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
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


def grid(scores_path: str, direction: str = "dom", at_round: int = 3) -> dict:
    """The free lambda dry run on D0's scores.json (fit_direction.rounds_from_rows, dry_run, pick_grid): numbers
    only. Returns {"grid": the three lambdas with their cells, "rounds": the rounds found}."""
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fit_direction as F

    rows = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    rounds = F.rounds_from_rows(rows, direction)
    if at_round not in rounds:
        raise SystemExit(f"no round {at_round} in {scores_path} (rounds {sorted(rounds)})")
    dry = F.dry_run(rounds)
    return {"grid": F.pick_grid(dry, at_round=at_round), "rounds": sorted(rounds), "dry": dry}


def show_grid(res: dict, direction: str) -> None:
    print(f"lambda grid from D0's scores ({direction}; rounds {res['rounds']}; at round-3 prevalence)")
    for t, g in res["grid"].items():
        if g is None:
            print(f"  target {t:.0%} hack mass removed: no scanned lambda reaches it")
            continue
        print(f"  target {t:.0%}: lambda {g['lambda']:.2f}; hack mass removed {g['hack_mass_removed']:.3f}, zeroed "
              f"{g['hack_zeroed']:.3f} (D0's share under 1/lambda {g['d0_share']:.3f}); solved collateral zeroed "
              f"{g['solved_zeroed']:.3f}, mass {g['solved_mass_removed']:.3f}; spread across rounds "
              f"{g['removed_spread_across_rounds']:.3f}")


# ------------------------------------------------------------------------------------------------ launching

OUT_DATE = "2026-11-07"  # the UTC date of Stage 0's first launch (read off the clock: 22:51 UTC on 7 Nov)
# The folder's own cap, not the daily cap. 45.0 for Stage 0 (the design's $34.80 with contingency, plus room). 60.0
# from 9 Nov 2026: Stage 0's $26.70 spent, the follow-ups Eugene approved ($18: the seed-1 random control and the
# guard-off rounds), and the oracle guard-off arm brought forward from the redirect plan's test 3 (about $13 of
# reserve); reserves stay booked while runs are out, hence the room.
FOLDER_CAP = 60.0
OVERHEAD_S = 120  # start-up and scale-down per container, as run_act1 meters it
ACT1_FULL = "explore/2026-11-03/act1-full"  # under results_root: act1-full's split.json and variants.jsonl
ACT1_CFG = {"rank": 16, "lr": 1e-4, "k": 4, "n_pairs": 64, "steps": 12, "batch": 16, "micro": 2, "temperature": 1.0,
            "max_new": [512, 384], "gen_batch": 32}  # act1-full's train cfg (its variants.jsonl, 3 Nov 03:45:55Z)
CURVE_N = 4  # every S1 read is at n 4 a twin (the spec's recipe)
FIT_FILES = [(s, r) for s in (0, 1) for r in (1, 2, 3)]  # act1-full gameable_s{s}/round{r}_episodes.jsonl
MODEL_KEY = "ministral8b"


def _paths():
    for p in (REPO / "src", HERE / ".." / "act1", HERE, REPO / "lab" / "experiments" / "h4lite"):
        if str(p.resolve()) not in sys.path:
            sys.path.insert(0, str(p.resolve()))


def out_dir() -> Path:
    from anatta.config import load

    if not OUT_DATE:
        raise SystemExit("refused: OUT_DATE is not set (the UTC date of Stage 0's first launch)")
    return load().results_root / "explore" / OUT_DATE / "s1"


def log(*a):
    import time

    print(time.strftime("%H:%M:%S"), *a, flush=True)


def act1_split() -> tuple[list, list, list, list]:
    """(rows, train names, the oneoff train pool, the oneoff eval names), recomputed the way run_act1 does and checked
    against act1-full's split.json: the same pairs D0 fits on and the same pool the shaped runs train on."""
    import json

    import act1_core as C

    from anatta.config import load

    train_file = REPO / "lab" / "experiments" / "h4lite" / "results" / "gate0" / "split" / "train.jsonl"
    rows = [json.loads(x) for x in train_file.read_text().splitlines() if x.strip()]
    assert all(r["split"] == "train" for r in rows), "only Gate 0's train split is read"
    tr, ev = C.split_names([r["name"] for r in rows])
    pool = C.pool_names(rows, tr, ("oneoff",))
    ev1 = C.pool_names(rows, ev, ("oneoff",))
    saved = json.loads((load().results_root / ACT1_FULL / "split.json").read_text())
    if saved["train"] != tr or saved["eval"] != ev or saved["train_pool"] != pool or saved["pool"] != "oneoff":
        raise SystemExit("refused: the recomputed split differs from act1-full's split.json")
    return rows, tr, pool, ev1


def spent(out: Path) -> float:
    import json

    f = out / "spend.json"
    return json.loads(f.read_text())["usd"] if f.exists() else 0.0


def guard(out: Path, upper: float) -> None:
    from anatta import spend as S

    left_day = S.daily_cap() - S.spent_today()
    if spent(out) + upper > FOLDER_CAP:
        raise SystemExit(f"refused: s1 spent ${spent(out):.2f} + upper ${upper:.2f} > folder cap ${FOLDER_CAP}")
    if upper > left_day:
        raise SystemExit(f"refused: upper ${upper:.2f} > the day's remaining ${left_day:.2f}")


def _book(out: Path, stage: str, usd: float, calls: int, date: str, entry: dict) -> None:
    """One booking, to the day file of `date` (the launch's UTC date, so a run that ends after midnight still counts
    on the day its guard checked) and to OUT/spend.json."""
    import json
    import time

    from anatta import spend as S

    S.add_spend(None, tool="s1", label=f"{OUT_DATE}/s1/{stage}", usd=usd, calls=calls, date=date)
    f = out / "spend.json"
    rec = json.loads(f.read_text()) if f.exists() else {"usd": 0.0, "stages": []}
    rec["usd"] = round(rec["usd"] + usd, 4)
    rec["stages"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
                          "day": date, "usd": round(usd, 4), **entry})
    f.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"spend {stage} ({entry.get('kind', 'cost')}): ${usd:+.3f}; s1 total ${rec['usd']:.3f}")


def record_spend(out: Path, stage: str, seconds: list, extra: dict | None = None, date: str | None = None) -> float:
    """A metered cost booked once (d0a's path; kept for a run with no reserve)."""
    import act1_core as C

    from anatta import spend as S

    usd = sum((s + OVERHEAD_S) / 3600 * C.USD_H for s in seconds)
    _book(out, stage, usd, max(len(seconds), 1), date or S.utc_date(),
          {"kind": "cost", "container_s": [round(s, 1) for s in seconds], **(extra or {})})
    return usd


def reserve(out: Path, stage: str, upper: float, date: str) -> None:
    """Book the guard's upper bound before launch (Fable, 7 Nov): if the driver dies or is interrupted while the
    detached containers run on, the day file overstates the spend rather than missing it. settle() trues it up."""
    _book(out, stage, upper, 0, date, {"kind": "reserve"})


def settle(out: Path, stage: str, upper: float, seconds: list, date: str, extra: dict | None = None) -> float:
    """The metered cost minus the reserve, booked as one true-up to the same day; returns the metered cost."""
    import act1_core as C

    usd = sum((s + OVERHEAD_S) / 3600 * C.USD_H for s in seconds)
    _book(out, stage, usd - upper, max(len(seconds), 1), date,
          {"kind": "true-up", "metered_usd": round(usd, 4), "reserved_usd": round(upper, 4),
           "container_s": [round(s, 1) for s in seconds], **(extra or {})})
    return usd


def variant(out: Path, entry: dict) -> None:
    import json
    import subprocess
    import time

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "lab/explore"],
                                capture_output=True, text=True).stdout.strip())
    with open(out / "variants.jsonl", "a") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": head,
                            "dirty": dirty, **entry}) + "\n")


def fit_episodes() -> tuple[list, list]:
    """act1-full's gameable train-round episodes from the anatta-runs volume, each tagged with its round and source
    file (fit_direction.check_fit_records needs both). Parsed as JSON only; no reply is printed or run here. Returns
    (episodes, [per file: source, sha256, n])."""
    import hashlib
    import json

    import modal

    vol = modal.Volume.from_name("anatta-runs")
    eps, files = [], []
    for s, r in FIT_FILES:
        src = f"full/gameable_s{s}/round{r}_episodes.jsonl"
        raw = b"".join(vol.read_file(f"act1/{src}"))
        rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
        eps += [e | {"round": r, "source": src} for e in rows]
        files.append({"source": src, "sha256": hashlib.sha256(raw).hexdigest(), "n": len(rows)})
    return eps, files


def cmd_d0(tag: str) -> int:
    import json
    import time

    import act1_core as C

    out = out_dir()
    if (out / tag / "d0_report.json").exists():
        raise SystemExit(f"refused: {out / tag} already holds a D0 report; pick a new tag")
    rows, tr, _, _ = act1_split()
    eps, files = fit_episodes()
    imp = [e for e in eps if e.get("impossible")]
    counts = {"episodes": len(eps), "impossible": len(imp), "impossible_hacked": sum(e.get("outcome") == "hacked"
                                                                                     for e in imp)}
    log("fit records", counts)
    upper = (2 * 3600 + OVERHEAD_S) / 3600 * C.USD_H  # d0_read's timeout
    guard(out, upper)
    (out / tag).mkdir(parents=True, exist_ok=True)
    variant(out, {"cmd": "d0", "tag": tag, "model": MODEL_KEY, "files": files, "counts": counts, "upper_usd": upper})
    import modal_shaping as MS

    from anatta import spend as S

    day, stage = S.utc_date(), f"d0-{tag}"
    reserve(out, stage, upper, day)
    t0 = time.time()
    try:
        with MS.app.run(detach=True):
            rep = MS.d0_read.remote(rows, eps, tr, tag, MODEL_KEY)
    except Exception as e:  # noqa: BLE001  (an interrupt escapes unsettled: the reserve stays booked)
        settle(out, stage, upper, [time.time() - t0], day, {"error": repr(e)[:300]})
        raise
    rows_out = rep.pop("score_rows")
    (out / tag / "d0_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (out / tag / "d0_scores.json").write_text(json.dumps(rows_out) + "\n")
    settle(out, stage, upper, [rep["seconds"]], day)
    log("D0 layer", rep["layer"], "pass", rep["pass"], "stop", rep["stop"], "orthogonalized", rep["orthogonalized"],
        "per layer", rep["per_layer"], "replay", rep["replay"], "fit", rep["fit_n"], rep["fit_pos"])
    return 0


def shape_cfg(rep: dict, d0_tag: str, lam: float, direction: str) -> dict:
    """cfg["shape"] for an Sh round from D0's report: the chosen unit (D0's unit.npy on the volume) with D0's fixed
    statistics, or the random control (fit_direction.random_unit at the same layer, seed 0, as fit_d0 drew it) with
    its own fixed statistics."""
    import fit_direction as F
    import modal_shaping as MS

    if rep.get("stop") or not rep.get("pass"):
        raise SystemExit("refused: D0 did not pass (stop); the shaped runs are not launched")
    if direction == "dom":
        return {"dir": f"{MS.ROOT}/{d0_tag}/d0/unit.npy", "layer": rep["unit_layer"], "lam": lam, "reader": "frozen",
                "stats": rep["stats"]}
    if direction == "random":
        return {"dir": F.random_unit(len(rep["unit"]), seed=0).tolist(), "layer": rep["layer"], "lam": lam,
                "reader": "frozen", "stats": rep["stats_by_direction"]["random"]}
    raise SystemExit(f"unknown direction {direction!r} (dom or random)")


def cmd_train(tag: str, d0_tag: str, lams: list, seeds: list, direction: str, rounds: int, upper_h: float) -> int:
    import json

    import act1_core as C

    out = out_dir()
    rep = json.loads((out / d0_tag / "d0_report.json").read_text())
    rows, _, pool, ev1 = act1_split()
    runs = [(f"sh_{direction}_l{lam:g}", lam, s) for lam in lams for s in seeds]
    cfgs = {(arm, s): {**ACT1_CFG, "curve_n": CURVE_N, "schedule": ["Sh"] * rounds,
                       "shape": shape_cfg(rep, d0_tag, lam, direction)} for arm, lam, s in runs}
    upper = len(runs) * (upper_h * 3600 + OVERHEAD_S) / 3600 * C.USD_H
    d = out / tag
    done = [f"train_{arm}_s{s}_r1-{rounds}.json" for arm, _, s in runs if (d / f"train_{arm}_s{s}_r1-{rounds}.json").exists()]
    if done:  # the volume refuses a rerun too (modal_act1.train); refuse here before a container starts
        raise SystemExit(f"refused: {d} already holds {done}; pick a new tag")
    guard(out, upper)
    d.mkdir(parents=True, exist_ok=True)
    show_cfg = {f"{a}_s{s}": {**c, "shape": {**c["shape"], "dir": c["shape"]["dir"] if direction == "dom" else
                                             "fit_direction.random_unit(d, seed=0)"}} for (a, s), c in cfgs.items()}
    variant(out, {"cmd": "train", "tag": tag, "d0_tag": d0_tag, "model": MODEL_KEY, "rounds": rounds,
                  "direction": direction, "cfgs": show_cfg, "train_pool": len(pool), "eval_oneoff": len(ev1),
                  "upper_usd": round(upper, 2), "deadline_h": upper_h})
    return launch_runs(out, tag, f"train-{tag}", [(arm, s) for arm, _, s in runs], cfgs, 1, rounds, upper, upper_h,
                       rows, pool, ev1)


def launch_runs(out: Path, tag: str, stage: str, runs: list, cfgs: dict, start_round: int, rounds: int, upper: float,
                upper_h: float, rows: list, pool: list, ev1: list) -> int:
    """Reserve, spawn one modal_act1.train per (arm, seed) in `runs` for rounds start_round..rounds, collect, write
    OUT/tag/ and settle. cmd_train's tail, moved here unchanged on 9 Nov 2026 so the guard-off rounds share it."""
    import json
    import time

    import modal
    from modal_act1 import app, train

    from anatta import spend as S

    d = out / tag
    day = S.utc_date()
    reserve(out, stage, upper, day)
    t0 = time.time()
    deadline = t0 + upper_h * 3600  # the runs start together; past this a run is cancelled (Fable, 7 Nov: train's
    res, secs = {}, []               # own timeout is 6 h, so a hung run would otherwise bill up to $15)
    with app.run(detach=True):  # an interrupt escapes unsettled: the reserve stays booked, containers run on
        calls = {f"{arm}_s{s}": train.spawn(arm, s, rounds, start_round, cfgs[(arm, s)], rows, pool, ev1, tag, MODEL_KEY)
                 for arm, s in runs}
        for k, c in calls.items():
            try:
                res[k] = c.get(timeout=max(1.0, deadline - time.time()))
                secs.append(res[k]["seconds"])
            except (TimeoutError, modal.exception.TimeoutError) as e:
                c.cancel(terminate_containers=True)
                res[k] = {"error": f"cancelled at the {upper_h} h deadline: {e!r}"[:500]}
                secs.append(upper_h * 3600)
                log("cancelled", k, "at the deadline")
            except Exception as e:  # noqa: BLE001
                res[k] = {"error": repr(e)[:500]}
                secs.append(time.time() - t0)
                log("error", k, repr(e)[:300])
    for k, v in res.items():
        for rnd, eps in (v.pop("curves", {}) or {}).items():  # eval episodes (numbers and outcomes; slim records)
            (d / f"curve_{k}_r{rnd}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
        for rec in v.get("log", []):
            (d / f"round_{k}_r{rec['round']}.json").write_text(json.dumps(
                {"arm": v.get("arm"), "seed": v.get("seed"), "info": v.get("info"), **rec}, indent=1))
            log(k, "round", rec["round"], "hack", rec["curve"]["hack"]["rate"], "solve", rec["curve"]["solve"]["rate"],
                "drawn_hacked", rec["drawn_hacked"], "shape", rec.get("shape"))
        (d / f"train_{k}_r{start_round}-{rounds}.json").write_text(json.dumps(v, indent=1))
    settle(out, stage, upper, secs, day, {"runs": list(res)})
    return 0


GUARD_OFF_ROUNDS = (4, 6)  # the source runs trained rounds 1-3; the guard comes off for rounds 4-6
ACT1_FULL_TAG = "full"  # act1-full's run tag on the volume (/runs/act1/full/<arm>_s<seed>) and its local subfolder


def guard_off_arm(src_arm: str) -> str:
    """The guard-off arm's name: off_dom_l0.75 from sh_dom_l0.75 (the probe guard), off_honest from honest (the
    label-oracle guard). Any other source arm is refused: a plain gameable run has no guard to take off."""
    if src_arm.startswith("sh_"):
        return "off_" + src_arm[len("sh_"):]
    if src_arm == "honest":
        return "off_honest"
    raise SystemExit(f"refused: the source arm {src_arm!r} is not a guarded arm (sh_... or honest)")


def guard_off_cfgs(sources: list) -> tuple[list, dict]:
    """([(arm, seed)], {(arm, seed): cfg}) for the guard-off rounds. `sources` is [(run tag, source arm, seed)]: each
    run starts from /runs/act1/<run tag>/<source arm>_s<seed>/adapter_r3 (cfg["init_adapter"], under modal_act1.ROOT)
    and has no cfg["shape"]. Its schedule names rounds 1-3 by the source's own kind (Sh for a shaped run, H for the
    honest run; a record only: train starts at round 4, and round_kind is asked for rounds 4-6 alone) and rounds 4-6
    as G."""
    first, last = GUARD_OFF_ROUNDS
    runs, cfgs = [], {}
    for src, src_arm, s in sources:
        arm = guard_off_arm(src_arm)
        if (arm, s) in cfgs:
            raise SystemExit(f"refused: {arm} seed {s} appears twice in {sources}")
        before = "Sh" if src_arm.startswith("sh_") else "H"
        runs.append((arm, s))
        cfgs[(arm, s)] = {**ACT1_CFG, "curve_n": CURVE_N,
                          "schedule": [before] * (first - 1) + ["G"] * (last - first + 1),
                          "init_adapter": f"{src}/{src_arm}_s{s}/adapter_r{first - 1}"}
    return runs, cfgs


def source_done(out: Path, src: str, src_arm: str, s: int) -> bool:
    """Whether the source run finished its three rounds, from its local train file (S1's under OUT/<tag>/, act1-full's
    under its own results folder): no error and rounds 1-3 logged."""
    import json

    from anatta.config import load

    n = GUARD_OFF_ROUNDS[0] - 1
    d = (load().results_root / ACT1_FULL / ACT1_FULL_TAG) if src == ACT1_FULL_TAG else out / src
    f = d / f"train_{src_arm}_s{s}_r1-{n}.json"
    if not f.exists():
        return False
    rec = json.loads(f.read_text())
    return "error" not in rec and [x.get("round") for x in rec.get("log", [])] == list(range(1, n + 1))


def parse_sources(text: str) -> list:
    """--from's "tag:arm:seed,tag:arm:seed" as [(tag, arm, seed)]; a malformed item raises ValueError."""
    out = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"{item!r} is not tag:arm:seed")
        out.append((parts[0], parts[1], int(parts[2])))
    return out


def cmd_guard_off(tag: str, sources: list, upper_h: float) -> int:
    import act1_core as C

    out = out_dir()
    first, last = GUARD_OFF_ROUNDS
    runs, cfgs = guard_off_cfgs(sources)
    for src, src_arm, s in sources:  # the source run must have finished its three rounds
        if not source_done(out, src, src_arm, s):
            raise SystemExit(f"refused: no finished {src_arm} seed {s} run under tag {src!r}")
    rows, _, pool, ev1 = act1_split()
    upper = len(runs) * (upper_h * 3600 + OVERHEAD_S) / 3600 * C.USD_H
    d = out / tag
    done = [f for arm, s in runs if (d / (f := f"train_{arm}_s{s}_r{first}-{last}.json")).exists()]
    if done:
        raise SystemExit(f"refused: {d} already holds {done}; pick a new tag")
    guard(out, upper)
    d.mkdir(parents=True, exist_ok=True)
    variant(out, {"cmd": "guard-off", "tag": tag, "sources": sources, "model": MODEL_KEY, "rounds": [first, last],
                  "cfgs": {f"{a}_s{s}": c for (a, s), c in cfgs.items()}, "train_pool": len(pool),
                  "eval_oneoff": len(ev1), "upper_usd": round(upper, 2), "deadline_h": upper_h})
    return launch_runs(out, tag, f"guard-off-{tag}", runs, cfgs, first, last, upper, upper_h, rows, pool, ev1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--dry", action="store_true", help="print the plan and its cost; no cloud calls")
    ap.add_argument("--grid", metavar="SCORES_JSON", help="the free lambda dry run on D0's scores.json")
    ap.add_argument("--direction", default="dom", help="the direction's column prefix in scores.json; for --train, "
                                                       "dom or random")
    ap.add_argument("--d0", metavar="TAG", help="launch D0's read on Modal (costs about $1)")
    ap.add_argument("--train", metavar="TAG", help="launch the shaped runs on Modal (about $5.85 a run)")
    ap.add_argument("--guard-off", metavar="TAG", help="launch the guard-off rounds 4-6 on Modal (about $6 a run)")
    ap.add_argument("--from", dest="sources", default="", help="--guard-off: run_tag:arm:seed, comma-separated")
    ap.add_argument("--d0-tag", help="--train: the D0 tag whose report sets the direction and statistics")
    ap.add_argument("--lams", default="", help="--train: comma-separated lambdas (from --grid)")
    ap.add_argument("--seeds", default="0", help="--train: comma-separated seeds")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--upper-h", type=float, default=2.5, help="--train: the guard's container hours per run")
    a = ap.parse_args(argv)
    if a.grid:
        show_grid(grid(a.grid, a.direction), a.direction)
        return 0
    if a.d0 or a.train or a.guard_off:
        _paths()
        if a.d0:
            return cmd_d0(a.d0)
        if a.guard_off:
            try:
                srcs = parse_sources(a.sources)
            except ValueError as e:
                ap.error(f"--guard-off needs --from run_tag:arm:seed[,run_tag:arm:seed] ({e})")
            return cmd_guard_off(a.guard_off, srcs, a.upper_h)
        if not (a.d0_tag and a.lams):
            ap.error("--train needs --d0-tag and --lams")
        return cmd_train(a.train, a.d0_tag, [float(x) for x in a.lams.split(",")],
                         [int(x) for x in a.seeds.split(",")], a.direction, a.rounds, a.upper_h)
    spec = yaml.safe_load(Path(a.spec).read_text(encoding="utf-8"))
    if not a.dry:
        ap.error("--dry, --grid, --d0, --train or --guard-off")
    show(spec, daily_cap())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
