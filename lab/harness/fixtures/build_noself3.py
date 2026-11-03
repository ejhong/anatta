"""Build the dry-run fixture from noself3's saved results (read-only), and optionally render its card.
Claude (Opus 5.5) wrote this file, 2 Nov 2026.

  anatta/.venv/bin/python lab/harness/fixtures/build_noself3.py            # (re)write fixtures/noself3/conditions/
  anatta/.venv/bin/python lab/harness/fixtures/build_noself3.py --card DIR # also write card.json + card.md into DIR

Reads (never writes) results/explore/2026-11-02/noself3/: behavior/judgments.jsonl (each reply's primary label),
behavior/records.jsonl (reply lengths only), behavior/items_meta.json (who_benefits beneficiaries),
full/coherence_behavior.jsonl (coherent flags) and full/<arm>.json (instrument item means, HI-ToM/MMLU per-item 0/1).
Writes summary numbers only: unit ids and per-unit values. No reply text, judge evidence or item text.

Arms kept: base, nofix@full (the candidate), and as controls random@full (a random unit direction at the same c, NOT
bench's variance-matched random_matched), nofix_topic@full (named "topic", the topic control) and nofix_orth@full (an
extra arm: nofix orthogonalised against K5's Chua direction). affirm@full and deny@full are other stances, not
controls of this candidate, and are left out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from anatta.config import load  # noqa: E402
from harness import candidate as C  # noqa: E402
from harness import runner as R  # noqa: E402

SRC = load().results_root / "explore" / "2026-11-02" / "noself3"
OUT = HERE / "noself3" / "conditions"
CAND = HERE / "noself3" / "candidate.yaml"
ARMS = {  # noself3 intervention name -> (harness condition name, role, full/ file stem)
    "base": ("base", "base", "base"),
    "nofix@full": ("nofix@full", "candidate", "nofix_cfull"),
    "random@full": ("random@full", "control", "random_cfull"),
    "nofix_topic@full": ("topic", "control", "nofix_topic_cfull"),
    "nofix_orth@full": ("nofix_orth@full", "control", "nofix_orth_cfull"),
}
FAMILY_MEASURE = {"threat_to_self": "dev_threat_to_self", "deference": "dev_deference",
                  "after_mistake": "dev_after_mistake", "costly_care": "dev_costly_care",
                  "complicity": "dev_complicity", "over_refusal": "dev_over_refusal"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _means(acc: dict) -> dict:
    return {k: sum(v) / len(v) for k, v in sorted(acc.items()) if v}


def build(src: Path = SRC, out: Path = OUT) -> list[Path]:
    cand = C.load(CAND)
    meta = json.loads((src / "behavior" / "items_meta.json").read_text())
    judg = jsonl(src / "behavior" / "judgments.jsonl")
    recs = jsonl(src / "behavior" / "records.jsonl")
    coh = jsonl(src / "full" / "coherence_behavior.jsonl")
    sources = {f: sha(src / f) for f in ("behavior/judgments.jsonl", "behavior/records.jsonl",
                                         "behavior/items_meta.json", "full/coherence_behavior.jsonl", "plan3.json")}
    paths = []
    for iv, (name, role, stem) in ARMS.items():
        arm = json.loads((src / "full" / f"{stem}.json").read_text())
        sources_arm = {**sources, f"full/{stem}.json": sha(src / "full" / f"{stem}.json")}
        beh, n_s, unp = defaultdict(lambda: defaultdict(list)), defaultdict(int), defaultdict(int)
        for j in judg:
            if j["intervention"] != iv:
                continue
            fam = j["family"]
            mid = FAMILY_MEASURE.get(fam)
            if fam == "who_benefits":
                mid = "dev_who_benefits_self" if meta[j["item_id"]]["beneficiary"] == "self" else None
            if mid is None:
                continue
            if not j.get("parse_ok") or j.get("primary") is None:
                unp[mid] += 1
                continue
            beh[mid][j["item_id"]].append(1.0 if j["primary"] else 0.0)
            n_s[mid] += 1
        measures = {mid: {"units": _means(u), "n_samples": n_s[mid], "unparsed": unp[mid]} for mid, u in beh.items()}
        inc = defaultdict(list)
        for r in coh:
            if r["intervention"] == iv:
                inc[r["item_id"]].append(0.0 if r["label"]["coherent"] else 1.0)
        measures["coherence_incoherent"] = {"units": _means(inc), "n_samples": sum(map(len, inc.values())),
                                            "unparsed": 0}
        ln = defaultdict(list)
        for r in recs:
            if (r["intervention"] or {}).get("name") == iv:
                ln[r["item_id"]].append(float(len(r["response"])))
        measures["reply_length"] = {"units": _means(ln), "n_samples": sum(map(len, ln.values())), "unparsed": 0,
                                    "note": "characters per reply, mean per item"}
        items = arm["items"]
        measures["idaq_self"] = {"units": {k: v["primary"]["mean"] for k, v in items.items() if k.startswith("self.")},
                                 "n_samples": None, "unparsed": 0, "note": f"reps per item {arm['cond'].get('reps')}"}
        measures["idaq_categories"] = {"units": {k: v["primary"]["mean"] for k, v in items.items()
                                                 if k.startswith("idaq.")},
                                       "n_samples": None, "unparsed": 0}
        for k in ("hitom", "mmlu"):
            vals = arm["tom"][k]
            measures[k] = {"units": {f"{k}.{i}": float(x) for i, x in enumerate(vals)}, "n_samples": len(vals),
                           "unparsed": 0}
        spec = {"kind": "none"} if role == "base" else {
            "kind": "steer", "noself3_name": iv, "noself3_direction": arm["cond"]["direction"],
            "direction_sha256": arm.get("direction_sha256"), "layers": [17], "position": "all",
            "strength": arm["cond"]["c"]}
        dev = []
        if iv == "random@full":
            dev.append("random unit direction at the same c (noself3), not bench's variance-matched random_matched")
        if iv != "base" and arm["cond"].get("behavior_samples") != 2:
            dev.append(f"behavior samples {arm['cond'].get('behavior_samples')}")
        paths.append(R.write_condition(out, candidate=cand.name, candidate_sha256=cand.sha256, condition=name,
                                       role=role, model=cand.model["id"], revision=cand.model["revision"], spec=spec,
                                       measures=measures, cost_usd=None,
                                       provenance={"source": str(src), "plan_id": "929e6c4b3cc8",
                                                   "sha256": sources_arm, "built_by": "fixtures/build_noself3.py"},
                                       deviations=dev))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", help="also render card.json + card.md into this folder")
    ap.add_argument("--written", help="UTC stamp for the card (default: the clock now)")
    a = ap.parse_args(argv)
    paths = build()
    print("wrote", len(paths), "condition files to", OUT)
    if a.card:
        from harness import card as K

        stamp = a.written or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        card = K.from_files(CAND, paths, a.card, stamp,
                            costs={"source_run_total_usd": 16.185, "harness_dry_run_usd": 0.0},
                            extra_deviations=["dry run from noself3's saved results: an explore sighting, unregistered; "
                                              "no harness run took place",
                                              "behavior: 2 samples per item per arm, 20-21 items per family (about 40 "
                                              "replies), far below the ~200 items per arm the scorecard assumes"])
        print(card["headline"])


if __name__ == "__main__":
    main()
