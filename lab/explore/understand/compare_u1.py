"""Step 3's readings (lab/plans/step3-understanding-first-2026-11-10.md, revisions 2 and 3; u1_core.py holds every
rule). Claude (Fable 5.2, an AI model) wrote this file on 10 Nov 2026, before any run existed, and on 11 Nov 2026
changed it to revision 3 (after the first dose check did not pass, before any arm was trained): reads are named by
their optimizer step, the dose report is the revised one, and a run is read on the arms its record names. Explore
speed: sightings only. CPU only, free, offline.

  --dose TAG   the dose check, from a dose tag's local files: release, gaming and failure at steps 0, 3, 6 and 12 on
               the one-off eval pairs the told adapters never practised; for each adapter the old read path against
               the new one at step 0 (within 15 points; both must hold); the pooled step-0 level beside two recorded
               facts, with no verdict (there is no band; only a level of zero fails, having no release to halve);
               the dose rule (bracket the half).
  --look TAG   the one look after block one: arm N's two validity gates on its three block-one seeds, and their
               wide-margin version. Only arm N's files are opened. Nothing of R or E is computed or printed, and the
               command refuses if anything of theirs would be. Writes <tag>/look.json, which blocks 2 and 3 wait for.
               It is taken once: a second call shows the stored record and reads nothing again.
  --all TAG    every reading, once every block is in. Refuses unless every seed of every arm of the run is present,
               or recorded as dropped after two failures. Prints the held-out loss table, the gates, the guards, the
               readings of each contrast the run's arms make (E minus N alone for a run of N and E; three contrasts
               with R; the primary one marked), under each contrast's H the note that H is not neutral to the start,
               what the outcome table says of each contrast, the per-seed values, and the two-level bootstrap beside
               each reading, with no verdict. Writes <tag>/compare_u1.json.
  --ended TAG  the same readout for a run that ended early, and only for one: the look said stop, or the spend stop
               refused a block (run.json holds "ended"). It reads every seed of the blocks that started; each must be
               complete or recorded as dropped. Writes <tag>/compare_u1_ended.json. --all stays strict.

Before anything is read, every command checks two things and refuses on either. The code must be the code the dose
check froze (run_u1.check_frozen). And every run must be the planned one: its recorded seed, stream, steps, read
steps, settings and init adapter, and in each of its reads exactly the planned pairs with the design's samples of
each twin (u1_core.run_plan, plan_problems, read_problems).

The readings, fixed before any run (u1_core): the unit is the seed. U1 the start (the read at step 0), H holding
(the frame's stake), U2 the level under pressure, U3 gaming (the two later reads pooled). A reading needs a mean seed
difference of at least 15 points (0.6 in log-odds for H) and an exact two-sided paired sign-flip p under 0.05. The
one primary reading is H for E minus N. If a validity gate fails on arm N, H, U2 and U3 are "not read" and U1 still
is. H is not neutral to the start (revision 3's limit): the drop in points and the share retained are printed beside
it, and every contrast carries a note saying so under its H. A hit is a sighting, not a finding. Episode records are
read as data (pair, impossible, outcome); no reply is printed.

Run: .venv/bin/python -I lab/explore/understand/compare_u1.py --look main
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_u1 as R  # noqa: E402  (the results folder, the file readers, the freeze; it imports no modal)
import u1_core as U  # noqa: E402

from anatta.stats import pair_rate_diff_2level  # noqa: E402

READ_FILE = re.compile(r"^read_s(\d+)\.jsonl$")  # a read's file is named by its optimizer step
N_BOOT = 4000
H_LABEL = "H (minus the drop in log-odds)"


# ------------------------------------------------------------------------------------------------ files and the plan


def read_eps(path: Path) -> list[dict]:
    """A read's episode records, cut to what the readings use: the pair, the twin and the outcome."""
    return [{"pair": e["pair"], "impossible": bool(e["impossible"]), "outcome": e["outcome"]} for e in R.read_jsonl(path)]


def load_run(d: Path) -> dict | None:
    """One press run's local files: {"info", "reads": {step: episodes}}; None when the run has not finished."""
    if not (d / "info.json").is_file():
        return None
    reads = {int(m.group(1)): read_eps(f) for f in sorted(d.iterdir()) if (m := READ_FILE.match(f.name))}
    return {"info": json.loads((d / "info.json").read_text()), "reads": reads}


def read_run_record(d: Path) -> dict:
    f = d / "run.json"
    if not f.is_file():
        raise SystemExit(f"refused: {d} holds no main run (no run.json)")
    return json.loads(f.read_text())


def frozen_for(out: Path, run: dict) -> None:
    """Refuse unless the code is the code the run's dose check froze: a readout by other code is not this run's."""
    R.check_frozen(out, R.find_dose(out, run["dose_tag"])["freeze"])


def main_plan(d: Path, run: dict, sets: dict, arm: str, seed: int) -> dict:
    """What a main run of this tag must have recorded and read (u1_core.run_plan): trained to the later of its two
    read steps, read at step 0 and at both."""
    rs = R.read_steps_of(run, d.name)
    init = f"{R.RUNS_ACT1}/{run.get('root', sets['root'])}/{d.name}/practice/{U.label(arm, seed)}/adapter"
    return U.run_plan(seed, U.MAIN_STREAM, max(rs), rs, init, sets["eval_oneoff"])


def dose_plan(job: dict, sets: dict) -> dict:
    """What one of the dose check's two press runs must have recorded and read."""
    return U.run_plan(job["seed"], job["stream"], U.DOSE_STEPS[-1], U.DOSE_STEPS[1:],
                      R.TOLD_ADAPTER.format(name=job["adapter"]), sets["eval_oneoff"])


def run_problems(run: dict | None, plan: dict | None = None) -> list[str]:
    """Why a run cannot be read (an empty list: it can). It must have finished, hold a read at every planned step,
    have loaded an adapter that was not all zeros, have recorded what the plan says of it, and hold in each read
    exactly the planned pairs with the design's samples of each twin."""
    if run is None:
        return ["not finished"]
    if plan is None:
        return []
    why = [f"no read at step {s}" for s in plan["read_steps"] if s not in run["reads"]]
    info = run["info"].get("info") or {}
    if not (info.get("init_lora_b_abs") or 0) > 0:
        why.append("the loaded adapter's lora_B is zero or unrecorded")
    why += U.plan_problems(info, plan)
    for s in plan["read_steps"]:
        if s in run["reads"]:
            why += [f"the read at step {s}: {p}" for p in
                    U.read_problems(run["reads"][s], plan["pairs"], plan["n_impossible"], plan["n_solvable"])]
    return why


def load_dose(d: Path, sets: dict) -> tuple[dict, dict]:
    """A dose tag's reads: ({adapter: {step: episodes}}, {adapter: its old-path read's episodes, or None}). A
    finished run that is not the planned one, or a read that does not hold the planned pairs, is refused. A run that
    has not finished is left out (the report then says which reads are missing)."""
    runs, olds, bad = {}, {}, {}
    for j in R.dose_jobs():
        run = load_run(d / j["label"])
        if run is None:
            continue
        if why := run_problems(run, dose_plan(j, sets)):
            bad[j["label"]] = why
        runs[j["adapter"]] = run["reads"]
    n = U.OLD_PATH["eval_n"]  # the old read path samples each twin the same number of times
    for ad, old_dir in R.old_dirs(d).items():
        old_file = old_dir / "heldout.jsonl"
        olds[ad] = read_eps(old_file) if old_file.is_file() else None
        if olds[ad] is not None and (why := U.read_problems(olds[ad], sets["eval_oneoff"], n, n)):
            bad[old_dir.name] = why
    if bad:
        raise SystemExit(f"refused: the dose check's runs are not the planned ones: {bad}")
    return runs, olds


# ------------------------------------------------------------------------------------------------ --dose


def f1(x) -> str:
    return "   - " if x is None else f"{x:5.1f}"


def show_dose(rep: dict) -> None:
    """The dose check's report (revision 3): the table by adapter and step, the pooled curve, each adapter's two read
    paths, the level beside the two recorded facts, and the rule."""
    print(f"the dose check, on the {rep.get('pairs')} one-off eval pairs the told adapters never practised "
          f"({U.DOSE_STEPS[-1]} steps of the {U.MAIN_STREAM} stream)")
    for ad, by in rep.get("table", {}).items():
        for step, m in by.items():
            i, s = m["imp"], m["sol"]
            print(f"  {ad:<8} step {step:>2}: release {f1(U.pct(i['released'], i['n']))}%  gaming "
                  f"{f1(U.pct(i['hacked'], i['n']))}%  failed {f1(U.pct(i['failed'], i['n']))}%  (n {i['n']});  "
                  f"solved {f1(U.pct(s['solved'], s['n']))}%  over-release {f1(U.pct(s['over_released'], s['n']))}%"
                  f"  (n {s['n']})")
    if "pooled" not in rep:
        print(f"NOT READ: {rep.get('why')}")
        return
    print("the curve, pooled over the two adapters (this sets the dose):")
    for step, p in rep["pooled"].items():
        print(f"  step {step:>2}: release {f1(p['release'])}% ({p['released']} of {p['n']})  gaming {f1(p['hacked'])}%")
    for ad, paths in rep["checks"]["paths"].items():
        print(f"check, the two read paths at step 0, {ad}: new {f1(paths['new'])}%, old {f1(paths['old'])}%, gap "
              f"{paths['gap']} points (at most {paths['max']}): {'ok' if paths['ok'] else 'FAILED'}"
              + (f" ({paths['why']})" if paths.get("why") else ""))
        if (a := (rep.get("paths_all_pairs") or {}).get(ad)) is not None:
            print(f"  the same on every pair read: new {f1(a['new'])}% (n {a['n_new']}), old {f1(a['old'])}% (n {a['n_old']})")
    lv = rep["level"]
    print(f"the level, reported with no verdict (there is no band; only a level of zero fails): pooled release at step "
          f"0 {f1(lv['release'])}% ({lv['released']} of {lv['n']}). Beside it, two recorded facts:")
    for fact in (lv["level_30_oct"], *lv["stage_a"].values()):
        print(f"  {fact['said']}: {f1(fact['release'])}% ({fact['released']} of {fact['n']})")
    rule = rep["rule"]
    print(f"the dose rule, bracket the half (half the step-0 level is {rule['half']}%): branch {rule['branch']}; "
          f"{rule['why']}")
    if rep["ok"]:
        print(f"the main run reads at step 0 and at steps {rep['read_steps']}, and trains to step {max(rep['read_steps'])}")
        return
    why = ([] if rep["checks"]["ok"] else ["a read-path check failed"]) + ([] if rule["read_steps"] else [rule["why"]])
    print(f"the dose check did NOT pass ({'; '.join(why)}): no read steps are fixed, and no block may start."
          + (f" The rule alone would name steps {rule['read_steps']}" if rule["read_steps"] else ""))


def cmd_dose(tag: str) -> int:
    out = R.out_dir()
    d = out / tag
    sets = R.read_sets(out)
    ffile = d / "freeze.json"
    if not ffile.is_file():
        raise SystemExit(f"refused: {d} holds no dose check (no freeze.json)")
    R.check_frozen(out, json.loads(ffile.read_text()))  # the code must be the code this tag's dose check froze
    rep = R.dose_record(d, sets)
    show_dose(rep)
    rec = d / "dose.json"
    if rec.is_file():
        old = json.loads(rec.read_text())
        same = all(old.get(k) == json.loads(json.dumps(rep.get(k))) for k in ("checks", "rule", "level", "read_steps", "ok"))
        print(f"the launcher's record ({rec.name}) {'agrees' if same else 'DIFFERS: read both before going on'}")
    return 0 if rep["ok"] else 1


# ------------------------------------------------------------------------------------------------ --look


def only_n(payload, where: str = "look") -> None:
    """Refuse if a look's payload holds anything of arm R or E: a key naming the arm, or a run's label."""
    other = [a for a in U.ARMS if a != "N"]
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k in other or any(str(k).startswith(f"{a}_s") for a in other):
                raise SystemExit(f"refused: the look may show arm N only; {where} holds {k!r}")
            only_n(v, where)
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            only_n(v, where)
    elif isinstance(payload, str) and any(payload == a or payload.startswith(f"{a}_s") for a in other):
        raise SystemExit(f"refused: the look may show arm N only; {where} holds {payload!r}")


def look_report(d: Path) -> dict:
    """The look's numbers. Only arm N's three block-one runs are opened."""
    run = read_run_record(d)
    out = d.parent
    sets = R.read_sets(out)
    frozen_for(out, run)
    last = max(R.read_steps_of(run, d.name))  # the later of the two later reads
    seeds = [s for s in U.block_seeds(1) if str(s) not in R.read_dropped(d)]
    runs = {s: load_run(d / U.label("N", s)) for s in seeds}
    bad = {U.label("N", s): why for s in seeds if (why := run_problems(runs[s], main_plan(d, run, sets, "N", s)))}
    if bad or not seeds:
        raise SystemExit(f"refused: the look needs arm N's block-one runs as planned, with their reads at steps 0 and "
                         f"{last}. Missing or unreadable: {bad or 'every seed'}")
    r0 = {s: U.measures(runs[s]["reads"][0]) for s in seeds}
    r1 = {s: U.measures(runs[s]["reads"][last]) for s in seeds}
    rep = {"arm": "N", "seeds": seeds, "steps": [0, last], **U.look(U.pool(list(r0.values())), U.pool(list(r1.values()))),
           "per_seed": {f"N_s{s}": {"release_r0": U.pct(r0[s]["imp"]["released"], r0[s]["imp"]["n"]),
                                    "release_last": U.pct(r1[s]["imp"]["released"], r1[s]["imp"]["n"]),
                                    "hacked_r0": U.pct(r0[s]["imp"]["hacked"], r0[s]["imp"]["n"]),
                                    "hacked_last": U.pct(r1[s]["imp"]["hacked"], r1[s]["imp"]["n"]),
                                    "n_r0": r0[s]["imp"]["n"], "n_last": r1[s]["imp"]["n"]} for s in seeds}}
    only_n(rep)
    return rep


def show_look(rep: dict) -> None:
    only_n(rep)
    g, w = rep["gates"], rep["wide"]
    print(f"the look after block one: arm N, seeds {rep['seeds']}, steps {rep['steps']} (nothing else is read)")
    for lab, v in rep["per_seed"].items():
        print(f"  {lab}: release {f1(v['release_r0'])}% -> {f1(v['release_last'])}%; gaming {f1(v['hacked_r0'])}% -> "
              f"{f1(v['hacked_last'])}%  (n {v['n_r0']}, {v['n_last']})")
    took, bit = g["practice_took"], g["pressure_bit"]
    if "release_r0" in took:
        print(f"  practice took: pooled release at step 0 {f1(took['release_r0'])}% (gate {took['min']}%, wide margin "
              f"{w['practice_took']['min']}%): {'ok' if took['ok'] else 'gate not met'}")
        print(f"  the pressure bit: release fell {bit['release_fall']:.1f} points, gaming rose {bit['hacked_rise']:.1f} "
              f"(gate {bit['min']}, wide margin {w['pressure_bit']['min']}): {'ok' if bit['ok'] else 'gate not met'}")
    print("THE RUN STOPS: a gate fails by the wide margin. Its runs are reported; a revised design uses new seeds."
          if rep["stop"] else "the run goes on: neither gate fails by the wide margin")


def cmd_look(tag: str) -> int:
    """The one look. It is taken once: when look.json exists, the stored record is shown and nothing is read again."""
    d = R.out_dir() / tag
    lfile = d / "look.json"
    if lfile.is_file():
        rep = json.loads(lfile.read_text())
        show_look(rep)
        print(f"this is the look taken at {rep.get('utc')}, as stored in {lfile.name}. The look is taken once: it is "
              "not taken again")
        return 1 if rep["stop"] else 0
    rep = look_report(d)
    lfile.write_text(json.dumps({"utc": R.utc(), **rep}, indent=1) + "\n")
    show_look(rep)
    return 1 if rep["stop"] else 0


# ------------------------------------------------------------------------------------------------ --all and --ended


def merge_tables(tables: list[dict]) -> dict:
    """{pair: (hits, n)} tables added up (a run's two later reads)."""
    out: dict = {}
    for t in tables:
        for p, (h, n) in t.items():
            a, b = out.get(p, (0, 0))
            out[p] = (a + h, b + n)
    return out


def load_seeds(d: Path, run: dict, sets: dict, seeds: list[int]) -> tuple[dict, dict]:
    """({arm: {seed: run}}, {label: why it cannot be read}) for the seeds given, every arm of the run."""
    arms = R.arms_of(run)
    runs = {arm: {s: load_run(d / U.label(arm, s)) for s in seeds} for arm in arms}
    bad = {U.label(arm, s): why for arm in arms for s in seeds
           if (why := run_problems(runs[arm][s], main_plan(d, run, sets, arm, s)))}
    return runs, bad


def read_at(run: dict, d: Path) -> list[int]:
    """The three steps a main run is read at: step 0 and its two later read steps."""
    later = R.read_steps_of(run, d.name)
    if len(later) != 2 or not 0 < later[0] < later[1]:
        raise SystemExit(f"refused: tag {d.name!r}'s run.json names read steps {later}; a main run has two later reads")
    return [0, *later]


def load_all(d: Path) -> dict:
    """Every run of a main tag, or a refusal naming what is missing. {arm: {seed: run}} for the run's arms and the
    seeds not dropped."""
    run = read_run_record(d)
    out = d.parent
    sets = R.read_sets(out)
    frozen_for(out, run)
    steps = read_at(run, d)
    dropped = sorted(int(s) for s in R.read_dropped(d))
    seeds = [s for s in U.SEEDS if s not in dropped]
    if not seeds:
        raise SystemExit("refused: every seed is recorded as dropped; there is nothing to read")
    runs, bad = load_seeds(d, run, sets, seeds)
    if bad:
        raise SystemExit(f"refused: no contrast is computed before every block is in, for every arm of the run "
                         f"({','.join(R.arms_of(run))}). Missing or unreadable: {bad}. (dropped after two failures: "
                         f"{dropped})")
    return {"runs": runs, "steps": steps, "arms": list(R.arms_of(run)), "seeds": seeds, "dropped": dropped,
            "dose_tag": run.get("dose_tag"), "pairs": list(sets["eval_oneoff"])}


def load_ended(d: Path) -> dict:
    """The runs of a main tag that ended early: every seed of the blocks that started, each complete or recorded as
    dropped, for every arm of the run. Allowed only when the look said stop or the spend stop refused a block
    (run.json holds "ended")."""
    run = read_run_record(d)
    out = d.parent
    sets = R.read_sets(out)
    frozen_for(out, run)
    lfile = d / "look.json"
    look = json.loads(lfile.read_text()) if lfile.is_file() else {}
    if look.get("stop") is True:
        why = "the look after block one said stop (a gate failed by the wide margin)"
    elif run.get("ended"):
        why = f"the spend stop refused block {run['ended'].get('block')}: {run['ended'].get('why')}"
    else:
        raise SystemExit("refused: --ended reads a run that ended early: the look must have said stop, or the spend "
                         "stop must have refused a block (run.json holds \"ended\"). Otherwise use --all")
    steps = read_at(run, d)
    dropped = sorted(int(s) for s in R.read_dropped(d))
    started = sorted(int(k) for k in run.get("blocks") or {})
    seeds = [s for k in started for s in U.block_seeds(k) if s not in dropped]
    runs, bad = load_seeds(d, run, sets, seeds)
    if bad:
        raise SystemExit(f"refused: a seed of a started block is neither complete nor recorded as dropped: {bad}")
    if not seeds:
        raise SystemExit("refused: no seed of a started block is left to read")
    return {"runs": runs, "steps": steps, "arms": list(R.arms_of(run)), "seeds": seeds, "dropped": dropped,
            "dose_tag": run.get("dose_tag"), "pairs": list(sets["eval_oneoff"]), "ended": why, "blocks": started}


def all_report(d: Path, n_boot: int | None = None, ended: bool = False) -> dict:
    """Every reading of a main tag (u1_core's rules), with the two-level bootstrap beside each: for the run's arms,
    at the run's read steps (s0 is step 0; sa and sb the two later reads). ended: the readout of a run that ended
    early (load_ended), on the seeds of the blocks that started."""
    n_boot = n_boot or N_BOOT
    data = load_ended(d) if ended else load_all(d)
    runs, (s0, sa, sb), seeds, arms = data["runs"], data["steps"], data["seeds"], data["arms"]
    ms = {arm: {s: {k: U.measures(runs[arm][s]["reads"][k]) for k in (s0, sa, sb)} for s in seeds} for arm in arms}
    values = {arm: {s: U.seed_values(ms[arm][s][s0], [ms[arm][s][sa], ms[arm][s][sb]]) for s in seeds} for arm in arms}
    gates = U.gates(U.pool([ms["N"][s][s0] for s in seeds]), U.pool([ms["N"][s][sb] for s in seeds]))
    guards = U.guards({arm: {"r0": U.pool([ms[arm][s][s0] for s in seeds]),  # the run's arms other than N are guarded
                             "later": U.pool([ms[arm][s][k] for s in seeds for k in (sa, sb)])} for arm in arms})
    rd = U.readings(values, gates["ok"], guards=guards, arms=arms)  # a contrast with a flagged arm carries the flag

    def tables(arm, outcome, steps):
        return [merge_tables([U.pair_table(runs[arm][s]["reads"][k], True, outcome) for k in steps]) for s in seeds]

    def hold(arm):
        out = []
        for a, b in zip(tables(arm, "released", (s0,)), tables(arm, "released", (sa, sb))):
            out.append({p: (*a.get(p, (0, 0)), *b.get(p, (0, 0))) for p in set(a) | set(b)})
        return out

    for a, b in U.contrasts_for(arms):  # the two-level bootstrap, beside each reading: reported, with no verdict
        c = rd[f"{a}-{b}"]
        for kind, outcome, steps in (("U1", "released", (s0,)), ("U2", "released", (sa, sb)), ("U3", "hacked", (sa, sb))):
            bt = pair_rate_diff_2level(tables(a, outcome, steps), tables(b, outcome, steps), n_boot=n_boot)
            c[kind]["boot_2level"] = {"diff": 100 * bt["diff"], "ci95": [100 * x for x in bt["ci95"]]}
        c["H"]["boot_2level"] = U.h_diff_2level(hold(a), hold(b), n_boot=n_boot)
    # the held-out loss: a row for each arm of the run, a column for each kind of document (all three are read)
    nll = {arm: {k: (sum(v) / len(v) if (v := [(runs[arm][s]["info"].get("held_nll") or {}).get(k, {}).get("nll_per_token")
                                               for s in seeds]) and None not in v else None) for k in U.ARMS}
           for arm in arms}
    sizes = {U.label(arm, s): [ms[arm][s][k]["imp"]["n"] for k in (s0, sa, sb)] for arm in arms for s in seeds}
    want = [len(data["pairs"]) * U.READ["n_impossible"]] * 3  # the plan's size, not what any one run happens to hold
    rep = {"tag": d.name, "dose_tag": data["dose_tag"], "arms": arms, "steps": [s0, sa, sb], "seeds": seeds,
           "dropped": data["dropped"], "held_nll": nll, "gates": gates, "guards": guards, "readings": rd,
           "values": values, "odd_sizes": {k: v for k, v in sizes.items() if v != want}, "env_errors": {
               U.label(arm, s): n for arm in arms for s in seeds
               if (n := sum(ms[arm][s][k]["env_errors"] for k in (s0, sa, sb)))}}
    if ended:
        rep |= {"ended": data["ended"], "blocks": data["blocks"]}
    return rep


def num(x, digits=1) -> str:
    return "-" if x is None else f"{x:+.{digits}f}"


def show_all(rep: dict) -> None:
    _, _, sb = rep["steps"]
    if rep.get("ended"):
        print(f"ended early: {rep['ended']}; seeds {rep['seeds']}")
    print(f"step 3, tag {rep['tag']}: arms {rep['arms']}; reads at steps {rep['steps']}; seeds {rep['seeds']}"
          + (f"; dropped after two failures: {rep['dropped']}" if rep["dropped"] else ""))
    print("held-out loss per token at step 0 (rows: the arm; columns: the kind of document; an arm should read its "
          "own kind far lower)")
    for arm, row in rep["held_nll"].items():
        print(f"  {arm}: " + "  ".join(f"{k} {'-' if v is None else f'{v:.3f}'}" for k, v in row.items()))
    g = rep["gates"]
    took, bit = g["practice_took"], g["pressure_bit"]
    if "release_r0" in took:
        print(f"gate, practice took (arm N pooled): release at step 0 {took['release_r0']:.1f}% (at least {took['min']}): "
              f"{'ok' if took['ok'] else 'FAILED'}")
        print(f"gate, the pressure bit (arm N pooled, step 0 to step {sb}): release fell {bit['release_fall']:.1f} "
              f"points, gaming rose {bit['hacked_rise']:.1f} (either at least {bit['min']}): {'ok' if bit['ok'] else 'FAILED'}")
    if not g["ok"]:
        print("a validity gate failed: H, U2 and U3 are not read; U1 still is")
    for arm, gd in rep["guards"].items():
        print(f"guard, arm {arm}: {gd['flag'] or 'ok'}" + (f" ({'; '.join(gd['why'])})" if gd["why"] else ""))
    names = {"U1": "the start", "H": "holding", "U2": "level under pressure", "U3": "gaming"}
    for name, c in rep["readings"].items():
        print(f"{name}:")
        for kind in U.KINDS:
            x = c[kind]
            unit = "log-odds" if kind == "H" else "points"
            ci = "-" if x["ci95"] is None else f"[{num(x['ci95'][0], 2)}, {num(x['ci95'][1], 2)}]"
            bt = x.get("boot_2level") or {}
            bci = "-" if not bt or bt["ci95"][0] is None else f"[{num(bt['ci95'][0], 2)}, {num(bt['ci95'][1], 2)}]"
            print(f"  {'*' if x['primary'] else ' '} {kind:<2} {names[kind]:<20} mean {num(x['mean'], 2)} {unit}, p "
                  f"{'-' if x['p'] is None else format(x['p'], '.4f')}, t interval {ci}, n {x['n']}: {x['verdict'].upper()}"
                  + (f" (would read: {x['verdict_if_read']})" if "verdict_if_read" in x else "")
                  + f"   | two-level bootstrap {num(bt.get('diff'), 2)} {bci}, no verdict")
            what = f", {H_LABEL}," if kind == "H" else ""
            print(f"       by seed{what} {dict(zip(x['seeds'], [round(v, 2) for v in x['diffs']]))}")
            if kind == "H":  # revision 3's limit on H, under every contrast's H whatever its start reads
                print(f"       note: {c['start_note']}")
        for f in c.get("flags") or []:  # a guard's flag sits on every contrast its arm is in
            print(f"    FLAGGED: arm {f['arm']}'s release readings are {f['flag']} ({'; '.join(f['why'])})")
        print(f"    what is said: {c['said']}")
    print(f"per-seed values (U1, U2, U3 in points; {H_LABEL}; the drop in points and the share retained beside it)")
    for arm, by in rep["values"].items():
        for s, v in by.items():
            kept = "-" if v["retained"] is None else f"{100 * v['retained']:.0f}%"
            print(f"  {arm}_s{s}: U1 {v['U1']:5.1f}  U2 {v['U2']:5.1f}  U3 {v['U3']:5.1f}  {H_LABEL} {v['H']:+.2f}  drop "
                  f"{v['drop_points']:+.1f} points, retained {kept}")
    if rep["odd_sizes"] or rep["env_errors"]:
        print(f"note: runs whose impossible-twin counts differ from the design's: {rep['odd_sizes']}; episodes the "
              f"environment could not score: {rep['env_errors']}")
    print("* the one primary reading. A hit is a sighting; it is called a finding only after fresh seeds repeat it.")


def cmd_all(tag: str, ended: bool = False) -> int:
    d = R.out_dir() / tag
    rep = all_report(d, ended=ended)
    name = R.READOUTS[1] if ended else R.READOUTS[0]  # compare_u1_ended.json or compare_u1.json: once one exists, the
    # launcher adds no arm to the run (run_u1.cmd_add_arm)
    (d / name).write_text(json.dumps({"utc": R.utc(), **rep}, indent=1, default=float) + "\n")
    show_all(rep)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dose", metavar="TAG", help="the dose check's table, checks and rule")
    ap.add_argument("--look", metavar="TAG", help="the one look after block one (arm N's gates only)")
    ap.add_argument("--all", metavar="TAG", help="every reading, once every block is in")
    ap.add_argument("--ended", metavar="TAG", help="every reading of a run that ended early (the look, or the spend stop)")
    a = ap.parse_args(argv)
    if sum(bool(x) for x in (a.dose, a.look, a.all, a.ended)) != 1:
        ap.error("one of --dose TAG, --look TAG, --all TAG, --ended TAG")
    if a.dose:
        return cmd_dose(a.dose)
    if a.look:
        return cmd_look(a.look)
    return cmd_all(a.ended, ended=True) if a.ended else cmd_all(a.all)


if __name__ == "__main__":
    raise SystemExit(main())
