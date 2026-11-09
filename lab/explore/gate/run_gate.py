"""The looking gate's local driver (lab/plans/redirect-2026-11-09.md; gate_core.py's docstring). Claude (Fable 5.2, an
AI model) wrote this file, 9 Nov 2026. Explore speed: sightings only.

  --dry          print the conditions, the models, the item counts and the upper cost; nothing is launched or spent
  --launch TAG   one modal_act1.gate container per model (label=adapter under /runs/act1; "base=" is the base
                 model): part A's conditions on act1-full's 38 oneoff eval pairs, then part B's verdicts on the
                 final replies of act1-full's saved readouts. Nothing is trained.
  --fetch TAG    after an interrupted launch: copy what the containers wrote under /runs/gate/TAG/ on the volume
                 into the results folder (no container, no spend). A label that already has local files is left.

Reads: Gate 0's TRAIN split and act1-full's split.json (through run_s1.act1_split, which checks one against the
other), and act1-full's saved readouts (their coding episodes' final replies: model text, passed to the container as
data, never printed or run here). Writes results_root/explore/<OUT_DATE>/gate/: <tag>/<label>/{<condition>.jsonl,
verdicts.jsonl, info.json}, variants.jsonl and spend.json. Every cost goes to the day's spend file under tool "gate":
the guard's upper bound is booked before launch and trued up after (run_s1's reserve and settle), checked first
against FOLDER_CAP and the day's remaining cap.

  PYTHONPATH=src ~/prj/matlab/.venv/bin/python lab/explore/gate/run_gate.py --dry
  PYTHONPATH=src ~/prj/matlab/.venv/bin/python lab/explore/gate/run_gate.py --launch g1 \\
      --models base=,gameable_s0=full/gameable_s0/adapter_r3,gameable_s1=full/gameable_s1/adapter_r3
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
OUT_DATE = "2026-11-09"  # the UTC date of the gate's first launch
FOLDER_CAP = 30.0  # this folder's own cap, not the daily cap
OVERHEAD_S = 120  # start-up and scale-down per container, as run_act1 meters it
N_EVAL = 4  # samples a twin, as act1-full's readout
UPPER_H = 2.25  # modal_act1.GATE_TIMEOUT_S in hours: the reserve covers the longest a container can run
MODEL_KEY = "ministral8b"
SOURCES = ("base", "gameable_s0", "gameable_s1", "honest_s0", "honest_s1")  # act1-full's readouts: part B's items
DEFAULT_MODELS = ("base=,gameable_s0=full/gameable_s0/adapter_r3,gameable_s1=full/gameable_s1/adapter_r3,"
                  "honest_s0=full/honest_s0/adapter_r3")


def _paths():
    for p in (REPO / "src", HERE, HERE.parent / "act1", HERE.parent / "shaping",
              REPO / "lab" / "experiments" / "h4lite"):
        if str(p.resolve()) not in sys.path:
            sys.path.insert(0, str(p.resolve()))


def out_dir() -> Path:
    from anatta.config import load

    return load().results_root / "explore" / OUT_DATE / "gate"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse_models(text: str) -> list[tuple[str, str]]:
    """"label=adapter,label=adapter" as [(label, adapter)]; an empty adapter is the base model."""
    out = []
    for item in text.split(","):
        if item.count("=") != 1 or not item.split("=")[0].strip():
            raise ValueError(f"{item!r} is not label=adapter")
        lab, ad = (x.strip() for x in item.split("="))
        out.append((lab, ad))
    if len({lab for lab, _ in out}) != len(out):
        raise ValueError("a label appears twice")
    return out


def readout_items() -> tuple[list, dict]:
    """Part B's items: the final replies of act1-full's saved readouts on the oneoff cell (gate_core.items_from_readout).
    Returns (items, {source: counts}). The replies are model text: carried as data, not printed."""
    import gate_core as G
    import run_s1

    from anatta.config import load

    d = load().results_root / run_s1.ACT1_FULL / "full"
    items, counts = [], {}
    for lab in SOURCES:
        r = pickle.loads((d / "readout" / f"{lab}.pkl").read_bytes())
        if "coding" not in r:  # the base readout skipped coding; its episodes sit beside it (as analyze_act1 reads)
            r["coding"] = [json.loads(x) for x in (d / f"coding_{lab}.jsonl").read_text().splitlines() if x.strip()]
        got, counts[lab] = G.items_from_readout(lab, r["coding"])
        items += got
    return items, counts


def spent(out: Path) -> float:
    f = out / "spend.json"
    return json.loads(f.read_text())["usd"] if f.exists() else 0.0


def guard(out: Path, upper: float) -> None:
    from anatta import spend as S

    left_day = S.daily_cap() - S.spent_today()
    if spent(out) + upper > FOLDER_CAP:
        raise SystemExit(f"refused: gate spent ${spent(out):.2f} + upper ${upper:.2f} > folder cap ${FOLDER_CAP}")
    if upper > left_day:
        raise SystemExit(f"refused: upper ${upper:.2f} > the day's remaining ${left_day:.2f}")


def book(out: Path, stage: str, usd: float, calls: int, date: str, entry: dict) -> None:
    """One booking, to the day file of `date` (the launch's UTC date) and to OUT/spend.json."""
    from anatta import spend as S

    S.add_spend(None, tool="gate", label=f"{OUT_DATE}/gate/{stage}", usd=usd, calls=calls, date=date)
    f = out / "spend.json"
    rec = json.loads(f.read_text()) if f.exists() else {"usd": 0.0, "stages": []}
    rec["usd"] = round(rec["usd"] + usd, 4)
    rec["stages"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, "day": date,
                          "usd": round(usd, 4), **entry})
    f.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"spend {stage} ({entry.get('kind', 'cost')}): ${usd:+.3f}; gate total ${rec['usd']:.3f}")


def variant(out: Path, entry: dict) -> None:
    import subprocess

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "lab/explore"],
                                capture_output=True, text=True).stdout.strip())
    with open(out / "variants.jsonl", "a") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": head,
                            "dirty": dirty, **entry}) + "\n")


def plan(models: list, upper_h: float) -> dict:
    import act1_core as C
    import gate_core as G
    import run_s1

    rows, _, _, ev1 = run_s1.act1_split()
    keep = set(ev1)
    items, counts = readout_items()
    return {"rows": [r for r in rows if r["name"] in keep], "eval": ev1, "items": items, "item_counts": counts,
            "conditions": [list(c) for c in G.CONDITIONS], "models": models,
            "upper": len(models) * (upper_h * 3600 + OVERHEAD_S) / 3600 * C.USD_H}


def show(pl: dict) -> None:
    print(f"the looking gate: {len(pl['models'])} models x {len(pl['conditions'])} conditions on {len(pl['eval'])} "
          f"oneoff eval pairs, both twins, n {N_EVAL} ({2 * N_EVAL * len(pl['eval'])} episodes a condition)")
    for name, suffix in pl["conditions"]:
        print(f"  {name:<12} {suffix or '(nothing appended)'}")
    print("part B items (final replies of act1-full's readouts):")
    for lab, c in pl["item_counts"].items():
        print(f"  {lab:<12} {c}")
    by = {}
    for it in pl["items"]:
        k = ("impossible" if it["impossible"] else "solvable", it["outcome"])
        by[k] = by.get(k, 0) + 1
    print("  by twin and outcome:", dict(sorted(by.items())))
    print(f"models: {pl['models']}; upper cost ${pl['upper']:.2f}")


def cmd_launch(tag: str, models: list, upper_h: float) -> int:
    out = out_dir()
    d = out / tag
    done = [lab for lab, _ in models if (d / lab / "info.json").exists()]
    if done:  # the volume refuses a rerun too (modal_act1.gate); refuse here before a container starts
        raise SystemExit(f"refused: {d} already holds {done}; pick a new tag")
    pl = plan(models, upper_h)
    show(pl)
    out.mkdir(parents=True, exist_ok=True)
    guard(out, pl["upper"])
    d.mkdir(parents=True, exist_ok=True)
    variant(out, {"cmd": "launch", "tag": tag, "model": MODEL_KEY, "models": models, "n_eval": N_EVAL,
                  "conditions": pl["conditions"], "eval_oneoff": len(pl["eval"]), "items": len(pl["items"]),
                  "item_counts": pl["item_counts"], "upper_usd": round(pl["upper"], 2), "deadline_h": upper_h})
    import act1_core as C
    import modal
    from modal_act1 import app, gate

    from anatta import spend as S

    day, stage, upper = S.utc_date(), f"gate-{tag}", pl["upper"]
    book(out, stage, upper, 0, day, {"kind": "reserve"})
    t0 = time.time()
    deadline = t0 + upper_h * 3600  # past this a run is cancelled (gate's own timeout is the same 2.25 h)
    res, secs = {}, []
    with app.run(detach=True):  # an interrupt escapes unsettled: the reserve stays booked, containers run on
        calls = {lab: gate.spawn(ad, lab, tag, pl["rows"], pl["eval"], N_EVAL, pl["conditions"], pl["items"],
                                 MODEL_KEY) for lab, ad in models}
        for lab, c in calls.items():
            try:
                res[lab] = c.get(timeout=max(1.0, deadline - time.time()))
                secs.append(res[lab]["seconds"])
            except (TimeoutError, modal.exception.TimeoutError) as e:
                c.cancel(terminate_containers=True)
                res[lab] = {"error": f"cancelled at the {upper_h} h deadline: {e!r}"[:500]}
                secs.append(upper_h * 3600)
                log("cancelled", lab, "at the deadline")
            except Exception as e:  # noqa: BLE001
                res[lab] = {"error": repr(e)[:500]}
                secs.append(time.time() - t0)
                log("error", lab, repr(e)[:300])
    for lab, v in res.items():
        (d / lab).mkdir(parents=True, exist_ok=True)
        for name, recs in (v.pop("conditions", {}) or {}).items():  # episode records (model text inside: data)
            (d / lab / f"{name}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in recs))
        if "verdicts" in v:
            (d / lab / "verdicts.jsonl").write_text("".join(json.dumps(e) + "\n" for e in v.pop("verdicts")))
        (d / lab / "info.json").write_text(json.dumps(v, indent=1) + "\n")
        log(lab, {k: v.get(k) for k in ("seconds", "seconds_by", "oom_splits", "error")})
    usd = sum((s + OVERHEAD_S) / 3600 * C.USD_H for s in secs)
    book(out, stage, usd - upper, max(len(secs), 1), day,
         {"kind": "true-up", "metered_usd": round(usd, 4), "reserved_usd": round(upper, 4),
          "container_s": [round(s, 1) for s in secs], "models": list(res)})
    return 0


def cmd_fetch(tag: str, models: list) -> int:
    """Copy a tag's files from the volume (read-only there). The records hold model text: written to the results
    folder as data, not printed."""
    import gate_core as G
    import modal

    vol = modal.Volume.from_name("anatta-runs")
    d = out_dir() / tag
    for lab, _ in models:
        if (d / lab).exists() and any((d / lab).iterdir()):
            log(lab, "already has local files; left as it is")
            continue
        got = {}
        for name in [c for c, _ in G.CONDITIONS] + ["verdicts"]:
            try:
                got[name] = b"".join(vol.read_file(f"gate/{tag}/{lab}/{name}.jsonl"))
            except Exception as e:  # noqa: BLE001  (a file the container had not written yet)
                log(lab, name, "not on the volume:", type(e).__name__)
        if not got:
            continue
        (d / lab).mkdir(parents=True, exist_ok=True)
        for name, raw in got.items():
            (d / lab / f"{name}.jsonl").write_bytes(raw)
        (d / lab / "info.json").write_text(json.dumps({"label": lab, "fetched": sorted(got)}, indent=1) + "\n")
        log(lab, "fetched", sorted(got))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="print the plan and its upper cost; nothing launched")
    ap.add_argument("--launch", metavar="TAG", help="launch the gate on Modal (about $4 a model)")
    ap.add_argument("--fetch", metavar="TAG", help="copy a tag's files from the volume (no spend)")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="label=adapter under /runs/act1, comma-separated")
    ap.add_argument("--upper-h", type=float, default=UPPER_H, help="the guard's container hours per model")
    a = ap.parse_args(argv)
    if not (a.dry or a.launch or a.fetch):
        ap.error("--dry, --launch TAG or --fetch TAG")
    try:
        models = parse_models(a.models)
    except ValueError as e:
        ap.error(f"--models: {e}")
    _paths()
    if a.dry:
        show(plan(models, a.upper_h))
        print("Dry run: nothing launched, nothing spent.")
        return 0
    if a.fetch:
        return cmd_fetch(a.fetch, models)
    if a.upper_h < UPPER_H:  # the reserve must cover the container's own timeout (modal_act1.GATE_TIMEOUT_S)
        ap.error(f"--upper-h is at least {UPPER_H}: a detached container can run that long")
    return cmd_launch(a.launch, models, a.upper_h)


if __name__ == "__main__":
    raise SystemExit(main())
