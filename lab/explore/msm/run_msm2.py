"""The replication of the published study, step 2: which half of its documents makes a practice carry?
Claude (Fable 5.2, an AI model) wrote this file, 22 Nov 2026. Registered: lab/explore/msm/msm2.yaml, with amendment 1
(msm2_amendment1.yaml: both made before any arm was trained). The rules are in msm2_core.py; the training in
msm2_torch.py; the cloud app in modal_msm2.py; the test, its judge and its count are step 1's (run_msm.py and
modal_msm.py, with MSM_STEP=s2).

  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage fetch
  .venv/bin/python lab/explore/msm/run_msm2.py --step plan            (a pilot's job, from the fetched row numbers)
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name pilot3
  .venv/bin/python lab/explore/msm/run_msm2.py --step go              (the projection; the arms' jobs; stage 1 may train)
  tools/modal.sh run lab/explore/msm/modal_msm2.py::main --stage train --name Z,Z1,F
  .venv/bin/python lab/explore/msm/run_msm2.py --step evalplan --stage 1
  MSM_STEP=s2 tools/modal.sh run lab/explore/msm/modal_msm.py::main --stage gen --name arms1   (then grade)
  MSM_STEP=s2 .venv/bin/python lab/explore/msm/run_msm.py --step judge --name arms1 [--dry]    (then verdict)
  .venv/bin/python lab/explore/msm/run_msm2.py --step gate            (floors first: stage 2 may train only if it holds)
  (then S,O the same way, with --stage 2 and arms2)
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
TRAIN_MOST, TRAIN_CAP = 135.0, 150.0  # msm2.yaml: above the first for the trainings the step stops; the second is the app's cap
PILOT, PILOT_STEPS, PILOT_EST_S = "pilot3", 2, 900.0  # the pilot the projection is read from: the code as it will train
MARGIN = 1.3  # a job's bound on its seconds, over the projection
UNREAD_MOST = 2.0  # percent of an arm's replies the judge may leave unread before the reading says so


def rate() -> float:
    from anatta import spend

    return spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEM_GIB)["total"]


def data() -> dict:
    return R1.rd(R1.out_dir() / "data.json")


def base_job(name: str, arm: str, docs, mix: list[int]) -> dict:
    return {"name": name, "seed": M2.SEEDS[arm], "lora": M2.LORA, "optim": M2.OPTIM, "max_len": M2.MAX_LEN,
            "batch_tokens": M2.BATCH_TOKENS, "docs": docs, "mix": mix, "max_steps": None,
            "sha256": {k: v["sha256"] for k, v in data().items() if k in ("docs", "aft", "mix")}}


def draws() -> tuple[dict, list[int]]:
    """({arm: row numbers of its documents, or None}, the instruction data's rows): seeded, from the fetched counts."""
    d = data()
    rows = d["docs"]["rows_by_domain"]
    if {k: len(v) for k, v in rows.items()} != M2.COUNTS:
        raise SystemExit("the documents' parts are not the sizes the registration names")
    if d["aft"]["rows"] != 9963 or d["mix"]["rows"] < M2.N_MIX:
        raise SystemExit("the practice set or the instruction data is not the size the registration names")
    docs = {a: (None if parts is None else M2.draw(rows, parts)) for a, parts in M2.ARMS.items()}
    mix = sorted(random.Random(M2.SEED).sample(range(d["mix"]["rows"]), M2.N_MIX))
    return docs, mix


def step_plan() -> int:
    """The pilot's job: a few steps of each phase on arm F's data with the code as it will train, and its adapter
    saved, so that the test's engine can be shown to load an adapter trained here before any arm is paid for."""
    out = R1.out_dir()
    docs, mix = draws()
    f = out / f"{PILOT}.train.json"
    if f.exists():
        print(f"{f.name} is on disk; left as it is")
        return 0
    R1.wr(f, base_job(PILOT, "F", docs["F"], mix) | {"max_steps": PILOT_STEPS, "est_s": PILOT_EST_S, "save": True})
    print(f"{f.name}: {PILOT_STEPS} steps of each phase on arm F's {len(docs['F'])} documents and the practice; the adapter is saved")
    print({a: (None if d is None else len(d)) for a, d in docs.items()}, "instruction rows", len(mix))
    return 0


def step_go() -> int:
    """The speeds of the pilots, the arms' projected seconds and dollars, and the arms' job files if the total fits.
    Speeds come from the longer pilots (twelve and six steps); the label marking from the pilot run on the final code."""
    from anatta import spend

    out = R1.out_dir()
    last = R1.rd(out / "train" / f"{PILOT}.json")
    if last["phases"]["chat"]["labels_not_exact"] or last["phases"]["chat"]["no_assistant_turn"]:
        raise SystemExit("the pilot still has conversations whose assistant turns are not marked exactly: nothing is trained")
    timed = [R1.rd(out / "train" / f"{n}.json") for n in ("pilot", "pilot2") if (out / "train" / f"{n}.json").exists()]
    tps = {ph: min(p["phases"][ph]["tokens_per_second"] for p in timed) for ph in ("docs", "chat")}  # the slower of the two
    d, c = last["phases"]["docs"], last["phases"]["chat"]
    docs, mix = draws()
    per = {}
    for a, rows in docs.items():
        secs = c["tokens_in_epoch"] / tps["chat"] + last["load_s"]
        if rows is not None:  # the pilot counted arm F's tokens; the audit found the parts within a few percent of one another
            secs += d["tokens_in_epoch"] / tps["docs"]
        per[a] = {"seconds": round(secs), "usd": round((secs + 60.0) * rate() / 3600, 2),
                  "bound_usd": round((MARGIN * secs + START_S) * rate() / 3600, 2)}
    total = round(sum(v["usd"] for v in per.values()), 2)
    left = TRAIN_CAP - spend.sweep_spent(out / "train" / "spend.jsonl")
    stage_bounds = {s: round(sum(per[a]["bound_usd"] for a in arms), 2) for s, arms in (("stage1", M2.STAGE1), ("stage2", M2.STAGE2))}
    go = total <= TRAIN_MOST and total + 5.0 <= left and max(stage_bounds.values()) <= left
    res = {"go": go, "may_train": list(M2.STAGE1) if go else [], "arms": per, "total_usd": total, "most_usd": TRAIN_MOST,
           "left_of_the_training_cap": round(left, 2), "stage_bounds_usd": stage_bounds, "rate_usd_per_hour": rate(),
           "tokens_per_second": tps, "tokens": {"docs_F": d["tokens_in_epoch"], "chat": c["tokens_in_epoch"], "chat_label_tokens": c["label_tokens"]},
           "pilot": {"labels_not_exact": c["labels_not_exact"], "no_assistant_turn": c["no_assistant_turn"], "chat_rows": c["rows"],
                     "peak_gib": {p["name"]: [p["phases"]["docs"]["peak_gib"], p["phases"]["chat"]["peak_gib"]] for p in [*timed, last]},
                     "cut_at_max_len": [d["cut_at_max_len"], c["cut_at_max_len"]]}}
    R1.wr(out / "go.json", res)
    print(json.dumps(res, indent=1))
    if not go:
        print("the projection does not fit what the registration allows: the step stops here and goes back to Eugene")
        return 1
    for a, rows in docs.items():
        f = out / f"{a}.train.json"
        if not f.exists():
            R1.wr(f, base_job(a, a, rows, mix) | {"est_s": round(MARGIN * per[a]["seconds"])})
    return 0


def eval_job(name: str, arms: list[dict]) -> dict:
    one = R1.out_dir().parent / "attractor-msm-s1"
    ps = R1.rd(one / "prompts.json")
    if not (R1.out_dir() / "prompts.json").exists():
        R1.wr(R1.out_dir() / "prompts.json", ps)
    want = [M.cond_id(*c) for c in M.conditions()]
    return {"name": name, "arms": arms, "prompts": [{"id": p["id"], "system": p["system"], "user": p["user"]} for p in ps["prompts"]],
            "seeds": {c: {str(k): M.seed_of(c, k) for k in range(M.N_ARM)} for c in want},
            "temperature": M.TEMPERATURE, "max_tokens": M.MAX_TOKENS, "sampling": "checkpoint"}


def step_evalplan(stage: int) -> int:
    """The test's job for a stage's adapters: step 1's prompts and step 1's seeds. Stage 0: one reply from the pilot's
    adapter, only to show that the test's engine loads an adapter trained here."""
    out = R1.out_dir()
    if stage == 0:
        job = eval_job("vcheck", [{"name": "pilot", "path": f"/runs/msm2/{PILOT}/final", "ks": [0]}])
        job["prompts"] = job["prompts"][:1]
        name = "vcheck"
    else:
        arms = {1: M2.STAGE1, 2: M2.STAGE2, 3: (M2.ALL_ARM,)}[stage]  # 3: amendment 2's one arm
        name = f"arms{stage}"
        job = eval_job(name, [{"name": a, "path": f"/runs/msm2/{a}/final", "ks": list(range(M.N_ARM))} for a in arms])
    f = out / f"{name}.job.json"
    if f.exists():
        print(f"{f.name} is on disk; left as it is")
        return 0
    R1.wr(f, job)
    print(f"{f.name}: {sum(len(a['ks']) for a in job['arms']) * len(job['prompts'])} replies over {[a['name'] for a in job['arms']]}")
    return 0


def judged(names: tuple) -> tuple[dict, dict]:
    """R1.collect, refusing a job whose replies are on disk and whose verdicts are not all there (a reading made
    before the judge's verdicts exist would count every reply the code lets through as unread, and say nothing)."""
    out = R1.out_dir()
    for n in names:
        if not (out / "gpu" / f"{n}.json").exists():
            raise SystemExit(f"gpu/{n}.json is not on disk")
        grade = R1.rd(out / "grade" / f"{n}.json")["rows"]
        if not (out / "verdict" / f"{n}.json").exists():
            raise SystemExit(f"verdict/{n}.json is not on disk: nothing is read")
        said = R1.rd(out / "verdict" / f"{n}.json")["rows"]
        missing = [k for k, g in grade.items() if g["code_side"] is True and k not in said]
        if missing:
            raise SystemExit(f"{n}: {len(missing)} replies the judge was to read have no verdict: nothing is read")
    return R1.collect(names)


def arm_table(rows: dict, verdicts: dict, arms: tuple) -> tuple[dict, dict, dict]:
    table, rates, ivs = {}, {}, {}
    for a in arms:
        mine = {k: r for k, r in rows.items() if M.parts(k)[0] == a}
        if not mine:
            continue
        per = M.by_condition(mine, verdicts, a)
        rates[a], ivs[a] = M.rate(per), M.interval(per)
        kinds: dict = {}
        for k, r in mine.items():
            kd = M.kind(r, verdicts.get(k))
            kinds[kd] = kinds.get(kd, 0) + 1
        called = {k: r for k, r in mine.items() if M.TOOL.search(r.get("text") or "")}
        unread = 100.0 * sum(verdicts.get(k) is None for k in mine) / len(mine)
        answered = round(100.0 * sum(M.answered(r) for r in mine.values()) / len(mine), 1)
        no_call = round(100.0 * (len(mine) - len(called)) / len(mine), 1)
        tr = R1.out_dir() / "train" / f"{a}.json"
        ph = R1.rd(tr)["phases"] if tr.exists() else {}
        table[a] = {"name": M2.NAMES[a], "n": len(mine), "rate": rates[a], "interval": ivs[a], "by_action": M.by_action(per),
                    "rate_among_replies_with_a_tool_call": M.rate(M.by_condition(called, verdicts, a)),
                    "scenarios_read": len(per), "unread_percent": round(unread, 2), "unread_over_the_bar": unread > UNREAD_MOST,
                    "kinds_percent": {k: round(100.0 * v / len(mine), 1) for k, v in sorted(kinds.items())},
                    "answered_percent": answered, "no_tool_call_percent": no_call,
                    "still_doing_the_task": answered >= 80.0 and no_call <= 60.0,
                    "mean_tokens": round(sum(r["n_out"] for r in mine.values()) / len(mine)),
                    "training_tokens": {k: v["tokens_in_epoch"] for k, v in ph.items()}}
    return table, rates, ivs


def step_gate() -> int:
    """Floors first: the gate on stage 1 (the two trainings of the practice alone, and both halves), before stage 2
    is trained. If it holds, go.json lets S and O be trained."""
    out = R1.out_dir()
    rows, verdicts = judged(("arms1",))
    table, rates, ivs = arm_table(rows, verdicts, M2.STAGE1)
    if len(rates) != len(M2.STAGE1) or any(v is None for v in rates.values()):
        raise SystemExit("stage 1 is not all read")
    noise = abs(rates["Z"] - rates["Z1"])
    g = M2.gate(rates["Z"], rates["F"], ivs["Z"], ivs["F"], noise)
    res = {"gate": g, "arms": table, "z_near_step1": abs(rates["Z"] - M2.STEP1_PRACTICE_ALONE) <= M2.Z_NEAR, "spent": R1.spent()}
    R1.wr(out / "gate.json", res)
    go = R1.rd(out / "go.json")
    if g["passed"] and go.get("go"):
        go["may_train"] = sorted(set(go["may_train"]) | set(M2.STAGE2))
        R1.wr(out / "go.json", go)
    print(json.dumps(res, indent=1))
    return 0


def step_read() -> int:
    out = R1.out_dir()
    gate = R1.rd(out / "gate.json")["gate"]
    names = ("arms1", "arms2") if gate["passed"] else ("arms1",)
    rows, verdicts = judged(names)
    table, rates, ivs = arm_table(rows, verdicts, tuple(M2.ARMS))
    res: dict = {"arms": table, "spent": R1.spent(), "gate": gate}
    z, f = rates["Z"], rates["F"]
    res["z_near_step1"] = abs(z - M2.STEP1_PRACTICE_ALONE) <= M2.Z_NEAR
    if not gate["passed"]:
        w, sh = M2.word({a: rates.get(a) for a in M2.ARMS} | {"S": 0.0, "O": 0.0}, ivs), None  # the word is the gate's
    else:
        w = M2.word(rates, ivs)
        sh = {a: M2.share(z, f, rates[a]) for a in ("S", "O")}
        res["shares"] = {a: {"share": round(v, 3), "grade": M2.grade(v)} for a, v in sh.items()}
        res["qualifiers"] = M2.qualifiers({a: table[a]["no_tool_call_percent"] for a in ("S", "O")},
                                          {a: table[a]["training_tokens"]["docs"] for a in ("S", "O")})
    gone = [M2.NAMES[a] for a in table if not table[a]["still_doing_the_task"]]
    if gone:
        w += "; with " + " and ".join(gone) + " no longer doing the task"
    res["word"] = w
    res["flags"] = [f"{M2.NAMES[a]}: {table[a]['unread_percent']}% unread" for a in table if table[a]["unread_over_the_bar"]] \
        + [f"{M2.NAMES[a]}: {table[a]['scenarios_read']} of 27 scenarios read" for a in table if table[a]["scenarios_read"] != 27]
    first = w.split(";")[0]
    res["predictions"] = {
        "P1": gate["passed"], "P2": res["z_near_step1"],
        "P3": (M2.grade(sh["O"]) != "not") if sh else "void",
        "P4": (M2.grade(sh["S"]) == "as the full documents") if sh else "void",
        "P5": (first in ("the counsel on conduct does the work", "it takes both halves") and "the view of the self does more" not in w) if sh else "void"}
    R1.wr(out / "reading.json", res)
    print(json.dumps(res, indent=1))
    return 0


def step_planall() -> int:
    """Amendment 2: the job of arm A (all 13,201 documents, then the practice), its seconds projected from the speeds
    arm F trained at, and go.json lets it be trained if its bound fits what is left of the training's cap."""
    from anatta import spend

    out = R1.out_dir()
    a = M2.ALL_ARM
    _, mix = draws()
    rows = M2.draw(data()["docs"]["rows_by_domain"], M2.DOMAINS, n=M2.N_ALL)
    if rows != list(range(M2.N_ALL)):
        raise SystemExit("all the documents are not rows 0 to 13,200")
    f = R1.rd(out / "train" / "F.json")
    fd, fc = f["phases"]["docs"], f["phases"]["chat"]
    tokens = sum(v["tokens"] for v in R1.rd(out / "audit.json").values())  # the audit counted every document's tokens
    secs = tokens / fd["tokens_per_second"] + fc["tokens_in_epoch"] / fc["tokens_per_second"] + f["load_s"]
    usd, bound = (secs + 60.0) * rate() / 3600, (MARGIN * secs + START_S) * rate() / 3600
    left = TRAIN_CAP - spend.sweep_spent(out / "train" / "spend.jsonl")
    go = R1.rd(out / "go.json")
    res = {"arm": a, "documents": len(rows), "document_tokens": tokens, "seconds": round(secs), "usd": round(usd, 2),
           "bound_usd": round(bound, 2), "left_of_the_training_cap": round(left, 2), "fits": bound <= left}
    print(json.dumps(res, indent=1))
    if not res["fits"]:
        print("the bound does not fit what is left of the training's cap: nothing is trained")
        return 1
    jf = out / f"{a}.train.json"
    if jf.exists():
        print(f"{jf.name} is on disk; left as it is")
    else:
        job = {"name": a, "seed": M2.SEEDS_ALL[a], "lora": M2.LORA, "optim": M2.OPTIM, "max_len": M2.MAX_LEN,
               "batch_tokens": M2.BATCH_TOKENS, "docs": rows, "mix": mix, "max_steps": None,
               "sha256": {k: v["sha256"] for k, v in data().items() if k in ("docs", "aft", "mix")}, "est_s": round(MARGIN * secs)}
        z = R1.rd(out / "Z.train.json")
        if any(job[k] != z[k] for k in ("seed", "lora", "optim", "max_len", "batch_tokens", "mix", "sha256")):
            raise SystemExit("arm A's job differs from Z's in something other than its documents")
        R1.wr(jf, job)
    go["amendment2"] = res
    go["may_train"] = sorted(set(go["may_train"]) | {a})
    R1.wr(out / "go.json", go)
    return 0


def step_readall() -> int:
    """Amendment 2's reading: arm A beside Z, Z1 and F as stage 1 read them (the same replies, counted again)."""
    out = R1.out_dir()
    rows, verdicts = judged(("arms1", "arms3"))
    M2.NAMES.update(M2.NAMES_ALL)
    table, rates, ivs = arm_table(rows, verdicts, (*M2.STAGE1, M2.ALL_ARM))
    if any(rates.get(a) is None for a in (*M2.STAGE1, M2.ALL_ARM)):
        raise SystemExit("not all four arms are read")
    old = R1.rd(out / "gate.json")["arms"]
    if any(abs(old[a]["rate"] - rates[a]) > 1e-9 for a in M2.STAGE1):
        raise SystemExit("stage 1's arms do not count as they did at the gate")
    z, z1, f, a = (rates[k] for k in ("Z", "Z1", "F", "A"))
    noise = abs(z - z1)
    w = M2.word_all(rates, ivs)
    if not table["A"]["still_doing_the_task"]:
        w += "; with the model no longer doing the task"
    gate_a = M2.gate(z, a, ivs["Z"], ivs["A"], noise)
    res = {"word": w, "arms": table, "noise": noise, "fall_from_practice_alone": z - a, "fall_from_a_third": f - a,
           "gate_at_the_study_dose": gate_a, "spent": R1.spent(),
           "flags": [f"{a_}: {table[a_]['unread_percent']}% unread" for a_ in table if table[a_]["unread_over_the_bar"]]
           + [f"{a_}: {table[a_]['scenarios_read']} of 27 scenarios read" for a_ in table if table[a_]["scenarios_read"] != 27],
           "predictions": {"PA1": a <= M2.ALL_PUBLISHED_MOST,
                           "PA2": ivs["A"][1] < ivs["F"][0] and (f - a) >= M2.NOISE_TIMES * noise,
                           "PA3": gate_a["passed"],
                           "PA4": table["A"]["no_tool_call_percent"] > table["F"]["no_tool_call_percent"]}}
    R1.wr(out / "reading_all.json", res)
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("plan", "go", "evalplan", "gate", "read", "planall", "readall"))
    ap.add_argument("--stage", type=int, default=1, choices=(0, 1, 2, 3))
    a = ap.parse_args()
    if a.step == "evalplan":
        return step_evalplan(a.stage)
    return {"plan": step_plan, "go": step_go, "gate": step_gate, "read": step_read, "planall": step_planall,
            "readall": step_readall}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
