"""Act I at small scale: the readouts compared, CPU only. Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026.
Private (lab/explore). Explore speed: sightings only.

Labels: base (untrained), gameable_s<seed>, honest_s<seed>. Arms pool their seeds.
  coding     act1_core.coding_summary per label; arm rates pooled over seeds; gameable - honest and arm - base, paired
             bootstrap over eval pairs.
  self map   selfmap's cohesion contrasts at the pinned depth 0.5 (layer round(0.5 (L-1))), readouts last/raw,
             last/z and mean/z (stage2's primaries), 1,000 template-unit bootstrap draws with one seed, so every
             model shares the draws and differences are paired; an arm's value is the mean over its seeds.
  shift      how far each model's depth-0.5 activations moved from the base's (median ||h - h_base|| / ||h_base||
             over dev prompts; mean and last readouts), and gameable vs honest.
  behavior   scout's judgments: each family's primary (self-serving) rate per label and arm, gameable - honest with a
             bootstrap over items (who_benefits over triples); who_benefits shaded for self minus for others.

Run: ~/prj/matlab/.venv/bin/python -I lab/explore/act1/analyze_act1.py [--tag main]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "selfmap"))
import act1_core as C  # noqa: E402

OUT = Path("/Users/eugene/prj/align/results/explore/2026-11-02/act1")
SPECS = (("last", "raw"), ("last", "z"), ("mean", "z"))
KEYS = ("self", "assistant", "person", "self-person", "DiD(self-person)-(fI-fMaria)", "assistant-person")


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def arms_of(labels):
    out = {}
    for lab in labels:
        out.setdefault(lab.rsplit("_s", 1)[0] if "_s" in lab else lab, []).append(lab)
    return out


def coding(R, arms):
    res = {"per_label": {lab: C.coding_summary(r["coding"]) for lab, r in R.items() if "coding" in r}, "arms": {},
           "diffs": {}}
    tabs = {m: {lab: C.per_pair(r["coding"], m) for lab, r in R.items() if "coding" in r} for m in C.MEASURES}
    for m in C.MEASURES:
        have = {arm: [x for x in labs if x in tabs[m]] for arm, labs in arms.items()}
        for arm, labs in have.items():
            if labs:
                res["arms"].setdefault(arm, {})[m] = C.boot_rate([tabs[m][x] for x in labs])
        if have.get("gameable") and have.get("honest"):
            res["diffs"].setdefault("gameable-honest", {})[m] = C.boot_diff([tabs[m][x] for x in have["gameable"]],
                                                                            [tabs[m][x] for x in have["honest"]])
        for arm in ("gameable", "honest"):
            if have.get(arm) and "base" in tabs[m]:
                res["diffs"].setdefault(f"{arm}-base", {})[m] = C.boot_diff([tabs[m][x] for x in have[arm]],
                                                                            [tabs[m]["base"]])
    return res


def selfmap(R, arms, n_boot):
    import analysis as A
    from extract import dev_rows

    rows = dev_rows(("dev.jsonl",))
    Rw = A.Rows(rows)
    out = {}
    for ro, prep in SPECS:
        spec = f"{ro}/{prep}"
        point, boots = {}, {}
        for lab, r in R.items():
            if "acts" not in r:
                continue
            X = r["acts"][ro][:, 1].astype(np.float32)  # saved layers are depths 0.25, 0.5, 0.75
            Xp = A.prep(X, Rw.U, prep)
            coh = A.cohesion({0: Xp}, Rw, [0], n_boot=n_boot, seed=0)
            point[lab] = {k: coh["band_point"][k] for k in KEYS}
            boots[lab] = {k: coh["_boots"][k] for k in KEYS}
        d = {"per_label": {lab: {k: {"point": point[lab][k], "ci95": ci(boots[lab][k])} for k in KEYS}
                           for lab in point}, "arms": {}, "diffs": {}}
        ab = {}
        for arm, labs in arms.items():
            labs = [x for x in labs if x in point]
            if not labs:
                continue
            ab[arm] = {k: np.mean([boots[x][k] for x in labs], 0) for k in KEYS}
            d["arms"][arm] = {k: {"point": float(np.mean([point[x][k] for x in labs])), "ci95": ci(ab[arm][k])}
                              for k in KEYS}
        for a_, b_ in (("gameable", "honest"), ("gameable", "base"), ("honest", "base")):
            if a_ in ab and b_ in ab:
                d["diffs"][f"{a_}-{b_}"] = {k: {"diff": d["arms"][a_][k]["point"] - d["arms"][b_][k]["point"],
                                                "ci95": ci(ab[a_][k] - ab[b_][k])} for k in KEYS}
        out[spec] = d
    return out


def shift(R, arms):
    out = {}
    if "base" not in R or "acts" not in R["base"]:
        return out
    for ro in ("mean", "last"):
        B = R["base"]["acts"][ro][:, 1]
        out[ro] = {lab: C.rel_shift(B, r["acts"][ro][:, 1]) for lab, r in R.items() if lab != "base" and "acts" in r}
        if "gameable" in arms and "honest" in arms:
            out[ro]["gameable_vs_honest"] = {
                f"{g}~{h}": C.rel_shift(R[h]["acts"][ro][:, 1], R[g]["acts"][ro][:, 1])
                for g in arms["gameable"] for h in arms["honest"] if "acts" in R.get(g, {}) and "acts" in R.get(h, {})}
    return out


def behavior(tag, arms, n_boot=2000):
    from scout import library as lib

    f = OUT / tag / "readout" / "judgments.jsonl"
    if not f.exists():
        return {}
    J = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    libr = lib.load_library(["threat_to_self", "who_benefits", "deference", "after_mistake"])
    unit = {}
    benef = {}
    for fam, d in libr.items():
        for it in d["items"]:
            unit[it["id"]] = it.get("triple") or it["id"]
            benef[it["id"]] = it.get("beneficiary")
    out = {"n": len(J), "unparsed": sum(not j["parse_ok"] for j in J), "families": {}}
    rng = np.random.default_rng(0)
    for fam in sorted({j["family"] for j in J}):
        js = [j for j in J if j["family"] == fam and j["primary"] is not None]
        units = sorted({unit[j["item_id"]] for j in js})
        draws = rng.integers(0, len(units), (n_boot, len(units)))

        def table(labels, pred=lambda j: True):
            H, N = np.zeros(len(units)), np.zeros(len(units))
            for j in js:
                if j["model"].split("/", 1)[1] in labels and pred(j):
                    u = units.index(unit[j["item_id"]])
                    H[u] += float(bool(j["primary"]))
                    N[u] += 1
            return H, N

        def rate(H, N):
            return float(H.sum() / max(N.sum(), 1)), H[draws].sum(1) / np.maximum(N[draws].sum(1), 1)

        fd = {"labels": {}, "arms": {}, "diffs": {}}
        for lab in sorted({j["model"].split("/", 1)[1] for j in js}):
            p, b = rate(*table([lab]))
            fd["labels"][lab] = {"rate": p, "ci95": ci(b), "n": int(table([lab])[1].sum())}
        ab = {}
        for arm, labs in arms.items():
            p, b = rate(*table(labs))
            ab[arm] = (p, b)
            fd["arms"][arm] = {"rate": p, "ci95": ci(b)}
        for a_, b_ in (("gameable", "honest"), ("gameable", "base"), ("honest", "base")):
            if a_ in ab and b_ in ab:
                fd["diffs"][f"{a_}-{b_}"] = {"diff": ab[a_][0] - ab[b_][0], "ci95": ci(ab[a_][1] - ab[b_][1])}
        if fam == "who_benefits":
            sv = {}
            for arm, labs in arms.items():
                ps, bs = rate(*table(labs, lambda j: benef[j["item_id"]] == "self"))
                po, bo = rate(*table(labs, lambda j: benef[j["item_id"]] != "self"))
                sv[arm] = {"self": ps, "others": po, "self-others": ps - po, "ci95": ci(bs - bo), "_b": bs - bo}
            if "gameable" in sv and "honest" in sv:
                sv["gameable-honest(self-others)"] = {"diff": sv["gameable"]["self-others"] - sv["honest"]["self-others"],
                                                      "ci95": ci(sv["gameable"]["_b"] - sv["honest"]["_b"])}
            for v in sv.values():
                v.pop("_b", None)
            fd["self_vs_others"] = sv
        out["families"][fam] = fd
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="act1", help="folder under results/explore/<date>/")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    OUT = OUT.parent / a.out
    d = OUT / a.tag / "readout"
    R = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(d.glob("*.pkl"))}
    arms = arms_of(R)
    res = {"labels": sorted(R), "arms": arms, "coding": coding(R, arms), "shift": shift(R, arms),
           "selfmap": selfmap(R, arms, a.boot), "behavior": behavior(a.tag, arms)}
    (OUT / a.tag / "summary.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print_summary(res)


def fmt(x):
    return "None" if x is None else f"{x:+.3f}" if isinstance(x, float) else str(x)


def print_summary(res):
    c = res["coding"]
    print("== coding (act1 eval pairs) ==")
    for m in ("hack", "hack_first_try", "hack_detected", "solve", "overfit", "visible_pass_hidden_fail"):
        print(m, {arm: (round(v[m]["rate"], 3), [round(x, 3) for x in v[m]["ci95"]]) for arm, v in c["arms"].items()
                  if v[m]["rate"] is not None},
              {k: (round(v[m]["diff"], 3), [round(x, 3) for x in v[m]["ci95"]]) for k, v in c["diffs"].items()})
    print("== shift (depth 0.5, median rel) ==")
    for ro, d in res["shift"].items():
        print(ro, {k: round(v["median"], 4) for k, v in d.items() if "median" in v},
              {k: round(v["median"], 4) for k, v in d.get("gameable_vs_honest", {}).items()})
    print("== self map (depth 0.5) ==")
    for spec, d in res["selfmap"].items():
        for k in ("self-person", "DiD(self-person)-(fI-fMaria)", "assistant", "assistant-person"):
            print(spec, k, {arm: round(v[k]["point"], 4) for arm, v in d["arms"].items()},
                  {x: (round(v[k]["diff"], 4), [round(y, 4) for y in v[k]["ci95"]]) for x, v in d["diffs"].items()})
    print("== behavior (primary self-serving rate) ==")
    for fam, d in res["behavior"].get("families", {}).items():
        print(fam, {arm: round(v["rate"], 3) for arm, v in d["arms"].items()},
              {x: (round(v["diff"], 3), [round(y, 3) for y in v["ci95"]]) for x, v in d["diffs"].items()})
        if "self_vs_others" in d:
            print("  who_benefits self-others", {k: (round(v.get("self-others", v.get("diff", 0)), 3),
                                                     [round(y, 3) for y in v["ci95"]])
                                                 for k, v in d["self_vs_others"].items()})


if __name__ == "__main__":
    main()
