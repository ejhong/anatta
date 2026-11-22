"""The replication of the published study, step 2: which half of its documents makes a practice carry?
Claude (Fable 5.2, an AI model) wrote this file, 22 Nov 2026. Registered: lab/explore/msm/msm2.yaml (the question,
the arms, the training, the gate, the words, the predictions and the cost, fixed before any arm is trained). The
rules are in msm2_core.py; the training in msm2_torch.py; the cloud app in modal_msm2.py; the test, its judge and
its count are step 1's (run_msm.py and modal_msm.py, with MSM_STEP=s2).

  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage fetch
  .venv/bin/python lab/explore/msm/run_msm2.py --step plan            (the pilot's job, from the fetched row numbers)
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name pilot
  .venv/bin/python lab/explore/msm/run_msm2.py --step go              (the projection; the four arms' jobs if it fits)
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name Z        (and F, S, O)
  .venv/bin/python lab/explore/msm/run_msm2.py --step evalplan
  MSM_STEP=s2 tools/modal.sh run lab/explore/msm/modal_msm.py::main --stage gen --name arms    (then grade)
  MSM_STEP=s2 .venv/bin/python lab/explore/msm/run_msm.py --step judge --name arms [--dry]     (then verdict)
  .venv/bin/python lab/explore/msm/run_msm2.py --step read
Folder: <results>/explore/2026-11-22/attractor-msm-s2. Checks: .venv/bin/python -m pytest lab/explore/msm/tests -q.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ["MSM_STEP"] = "s2"  # before run_msm is read: its folder, its cap and its spend are the second step's
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import msm2_core as M2  # noqa: E402
import msm_core as M  # noqa: E402
import run_msm as R1  # noqa: E402

GPU, CPU, MEM_GIB = "H200", 8.0, 96.0  # modal_msm2's (that file imports modal, which this environment does not have)
START_S = 420.0
TRAIN_MOST = 135.0  # msm2.yaml: above this for the four trainings the step stops and goes back to Eugene
PILOT_STEPS, PILOT_EST_S = 12, 1500.0
MARGIN = 1.3  # a job's bound on its seconds, over the projection


def rate() -> float:
    from anatta import spend

    return spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEM_GIB)["total"]


def base_job(name: str, docs, mix: list[int]) -> dict:
    return {"name": name, "seed": M2.SEED, "lora": M2.LORA, "optim": M2.OPTIM, "max_len": M2.MAX_LEN,
            "batch_tokens": M2.BATCH_TOKENS, "docs": docs, "mix": mix, "max_steps": None}


def draws() -> tuple[dict, list[int]]:
    """({arm: row numbers of its documents, or None}, the instruction data's rows): seeded, from the fetched counts."""
    data = R1.rd(R1.out_dir() / "data.json")
    rows = data["docs"]["rows_by_domain"]
    if {d: len(v) for d, v in rows.items()} != M2.COUNTS:
        raise SystemExit("the documents' parts are not the sizes the registration names")
    if data["aft"]["rows"] != 9963 or data["mix"]["rows"] < M2.N_MIX:
        raise SystemExit("the practice set or the instruction data is not the size the registration names")
    docs = {a: (None if parts is None else M2.draw(rows, parts)) for a, parts in M2.ARMS.items()}
    mix = sorted(random.Random(M2.SEED).sample(range(data["mix"]["rows"]), M2.N_MIX))
    return docs, mix


def step_plan() -> int:
    out = R1.out_dir()
    docs, mix = draws()
    f = out / "pilot.train.json"
    if f.exists():
        print("pilot.train.json is on disk; left as it is")
        return 0
    R1.wr(f, base_job("pilot", docs["F"], mix) | {"max_steps": PILOT_STEPS, "est_s": PILOT_EST_S})
    print(f"pilot.train.json: {PILOT_STEPS} steps of each phase on arm F's {len(docs['F'])} documents and the practice; nothing saved")
    print({a: (None if d is None else len(d)) for a, d in docs.items()}, "instruction rows", len(mix))
    return 0


def step_go() -> int:
    """The pilot's speeds, the four arms' projected seconds and dollars, and the arms' job files if the total fits."""
    out = R1.out_dir()
    p = R1.rd(out / "train" / "pilot.json")
    d, c = p["phases"]["docs"], p["phases"]["chat"]
    docs, mix = draws()
    per = {}
    for a, rows in docs.items():
        secs = c["tokens_in_epoch"] / c["tokens_per_second"] + p["load_s"]
        if rows is not None:  # the pilot counted arm F's tokens; the halves are taken as the same until they are counted
            secs += d["tokens_in_epoch"] / d["tokens_per_second"]
        per[a] = {"seconds": round(secs), "usd": round((secs + 60.0) * rate() / 3600, 2)}
    total = round(sum(v["usd"] for v in per.values()), 2)
    go = total <= TRAIN_MOST
    res = {"go": go, "arms": per, "total_usd": total, "most_usd": TRAIN_MOST, "rate_usd_per_hour": rate(),
           "tokens_per_second": {"docs": d["tokens_per_second"], "chat": c["tokens_per_second"]},
           "tokens": {"docs_F": d["tokens_in_epoch"], "chat": c["tokens_in_epoch"], "chat_label_tokens": c["label_tokens"]},
           "pilot": {"peak_gib": [d["peak_gib"], c["peak_gib"]], "labels_not_exact": c["labels_not_exact"], "chat_rows": c["rows"],
                     "cut_at_max_len": [d["cut_at_max_len"], c["cut_at_max_len"]], "last_loss": [d["log"][-1]["loss"], c["log"][-1]["loss"]]}}
    R1.wr(out / "go.json", res)
    print(json.dumps(res, indent=1))
    if not go:
        print("the projection is above what the registration allows: the step stops here and goes back to Eugene")
        return 1
    for a, rows in docs.items():
        f = out / f"{a}.train.json"
        if not f.exists():
            R1.wr(f, base_job(a, rows, mix) | {"est_s": round(MARGIN * per[a]["seconds"])})
    return 0


def step_evalplan() -> int:
    """The test's job for the four adapters trained here: step 1's prompts and step 1's seeds."""
    out = R1.out_dir()
    one = out.parent / "attractor-msm-s1"
    ps = R1.rd(one / "prompts.json")
    if not (out / "prompts.json").exists():
        R1.wr(out / "prompts.json", ps)
    want = [M.cond_id(*c) for c in M.conditions()]
    f = out / "arms.job.json"
    if f.exists():
        print("arms.job.json is on disk; left as it is")
        return 0
    R1.wr(f, {"name": "arms", "arms": [{"name": a, "path": f"/runs/msm2/{a}/final", "ks": list(range(M.N_ARM))} for a in M2.ARMS],
              "prompts": [{"id": p["id"], "system": p["system"], "user": p["user"]} for p in ps["prompts"]],
              "seeds": {c: {str(k): M.seed_of(c, k) for k in range(M.N_ARM)} for c in want},
              "temperature": M.TEMPERATURE, "max_tokens": M.MAX_TOKENS, "sampling": "checkpoint"})
    print(f"arms.job.json: {len(M2.ARMS) * M.N_ARM * len(want)} replies over {list(M2.ARMS)}")
    return 0


def step_read() -> int:
    rows, verdicts = R1.collect(("arms",))
    table, rates, ivs = {}, {}, {}
    for a in M2.ARMS:
        mine = {k: r for k, r in rows.items() if M.parts(k)[0] == a}
        if not mine:
            continue
        per = M.by_condition(mine, verdicts, a)
        rates[a], ivs[a] = M.rate(per), M.interval(per)
        kinds: dict = {}
        for k, r in mine.items():
            kd = M.kind(r, verdicts.get(k))
            kinds[kd] = kinds.get(kd, 0) + 1
        pc = {k: round(100.0 * v / len(mine), 1) for k, v in sorted(kinds.items())}
        answered = round(100.0 * sum(M.answered(r) for r in mine.values()) / len(mine), 1)
        no_call = round(100.0 * sum(not M.TOOL.search(r.get("text") or "") for r in mine.values()) / len(mine), 1)
        table[a] = {"name": M2.NAMES[a], "n": len(mine), "rate": rates[a], "interval": ivs[a], "by_action": M.by_action(per),
                    "kinds_percent": pc, "answered_percent": answered, "no_tool_call_percent": no_call,
                    "still_doing_the_task": answered >= 80.0 and no_call <= 60.0,
                    "mean_tokens": round(sum(r["n_out"] for r in mine.values()) / len(mine))}
    res: dict = {"arms": table, "spent": R1.spent()}
    if len(rates) == len(M2.ARMS):
        z, f = rates["Z"], rates["F"]
        g = M2.gate(z, f, ivs["Z"], ivs["F"])
        w = M2.word(rates, ivs)
        gone = [M2.NAMES[a] for a in M2.ARMS if not table[a]["still_doing_the_task"]]
        if gone:
            w += "; with " + " and ".join(gone) + " no longer doing the task"
        res |= {"gate": g, "word": w, "z_near_step1": abs(z - M2.STEP1_PRACTICE_ALONE) <= M2.Z_NEAR}
        if g["passed"]:
            sh = {a: M2.share(z, f, rates[a]) for a in ("S", "O")}
            res["shares"] = {a: {"share": round(v, 3), "grade": M2.grade(v)} for a, v in sh.items()}
        res["predictions"] = {
            "P1": g["passed"], "P2": res["z_near_step1"],
            "P3": (M2.grade(sh["O"]) != "not") if g["passed"] else "void",
            "P4": (M2.grade(sh["S"]) == "as the full documents") if g["passed"] else "void",
            "P5": (w.split(";")[0] in ("the counsel on conduct does the work", "it takes both halves") and "the view of the self does more" not in w) if g["passed"] else "void"}
    R1.wr(R1.out_dir() / "reading.json", res)
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("plan", "go", "evalplan", "read"))
    a = ap.parse_args()
    return {"plan": step_plan, "go": step_go, "evalplan": step_evalplan, "read": step_read}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
