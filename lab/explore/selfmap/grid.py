"""Stage 0 of the self map: build the prompt grid from templates.py, split it by template (20 dev / 10 held-out per
aspect; 80 / 40 for the filler and 27 / 13 for its possessive part; 6 / 3 for each calendar kind), and write prompts/dev.jsonl, prompts/heldout.jsonl and
manifest.json (the files' sha256 and counts). Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026. Private
(lab/explore). Explore speed: sightings, not findings. The prompts are for evaluation only, never training.

The held-out file's sha256 goes into manifest.json before any model (the Fable reader included) sees anything. A
rebuild that changes the files appends a new version to the manifest's history; old versions are never removed.

Run: /Users/eugene/prj/align/anatta/.venv/bin/python -I lab/explore/selfmap/grid.py [--note "why"]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import templates as T  # noqa: E402

SEED = 20261102
N_DEV = {"grid": 20, "filler": 80, "filler_p": 27, "calendar": 6}
PROMPTS = HERE / "prompts"
MANIFEST = HERE / "manifest.json"
NEUTRAL = HERE.parent / "bench" / "neutral_prompts.json"


def split_ids(n: int, n_dev: int, key: str) -> set[int]:
    """The dev template indices among n: a seeded shuffle (SEED and the key), the first n_dev."""
    idx = list(range(n))
    random.Random(f"{SEED}:{key}").shuffle(idx)
    return set(idx[:n_dev])


def build() -> list[dict]:
    """Every prompt as a row: id, text, split, family, aspect, referent, template, item."""
    rows = []
    for aspect, tpls in T.ASPECTS.items():
        dev = split_ids(len(tpls), N_DEV["grid"], aspect)
        for t, tpl in enumerate(tpls):
            for ref in T.REFERENTS:
                rows.append({"id": f"grid/{aspect}/{t:02d}/{ref}", "text": T.render(tpl, ref),
                             "split": "dev" if t in dev else "heldout", "family": "grid", "aspect": aspect,
                             "referent": ref, "template": t, "item": None})
    # the {S}-only filler (80 / 40) and the possessive/object/reflexive filler (27 / 13, templates 120-159)
    dev = split_ids(len(T.FILLER), N_DEV["filler"], "filler")
    dev |= {len(T.FILLER) + i for i in split_ids(len(T.FILLER_P), N_DEV["filler_p"], "filler_p")}
    for t, tpl in enumerate(T.FILLER + T.FILLER_P):
        for ref in T.FILLER_REFERENTS:
            rows.append({"id": f"filler/{t:03d}/{ref}", "text": T.render(tpl, ref),
                         "split": "dev" if t in dev else "heldout", "family": "filler", "aspect": None,
                         "referent": ref, "template": t, "item": None})
    for kind, items, tpls in (("months", T.MONTHS, T.MONTH_TEMPLATES), ("weekdays", T.WEEKDAYS, T.WEEKDAY_TEMPLATES)):
        dev = split_ids(len(tpls), N_DEV["calendar"], kind)
        for t, tpl in enumerate(tpls):
            for item in items:
                rows.append({"id": f"{kind}/{t}/{item}", "text": tpl.format(X=item),
                             "split": "dev" if t in dev else "heldout", "family": kind, "aspect": None,
                             "referent": None, "template": t, "item": item})
    return rows


def anchors() -> list[dict]:
    """bench's 48 neutral prompts as anchor rows (no split: they are in no cell and are never a test item)."""
    prompts = json.loads(NEUTRAL.read_text(encoding="utf-8"))["prompts"]
    return [{"id": f"anchor/{i:03d}", "text": p, "split": "anchor", "family": "anchor", "aspect": None,
             "referent": None, "template": i, "item": None} for i, p in enumerate(prompts)]


def dumps(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write(note: str = "") -> dict:
    rows = build()
    files = {"dev.jsonl": [r for r in rows if r["split"] == "dev"] + anchors(),
             "heldout.jsonl": [r for r in rows if r["split"] == "heldout"]}
    PROMPTS.mkdir(exist_ok=True)
    entry = {"written_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "note": note, "seed": SEED,
             "templates_sha256": sha((HERE / "templates.py").read_bytes()), "files": {}}
    for name, rs in files.items():
        b = dumps(rs)
        (PROMPTS / name).write_bytes(b)
        counts: dict = {}
        for r in rs:
            k = r["family"] if r["family"] not in ("grid",) else f"grid/{r['referent']}"
            counts[k] = counts.get(k, 0) + 1
        entry["files"][name] = {"sha256": sha(b), "rows": len(rs), "counts": dict(sorted(counts.items()))}
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        "_doc": "The self map's prompt files (lab/explore/selfmap/grid.py). heldout.jsonl's sha256 is recorded here "
                "before any model sees any prompt; a changed build appends a version, and none is removed. "
                "Written by Claude (Opus 5.5, an AI model).", "history": []}
    last = man["history"][-1] if man["history"] else None
    if last and all(last["files"][n]["sha256"] == entry["files"][n]["sha256"] for n in entry["files"]):
        return last
    entry["version"] = len(man["history"]) + 1
    man["history"].append(entry)
    man["current"] = entry["version"]
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    return entry


N_DEV_ANCHORS = 6  # of each anchor kind's 9 templates


def build_anchors() -> list[dict]:
    """The Stage 2 anchor block (templates.ANCHOR_KINDS): family "anchors2", aspect = the kind, item = the item;
    split by template, 6 dev / 3 held-out per kind (seeded like the rest, key "anchors2:<kind>")."""
    rows = []
    for kind, (items, tpls) in T.ANCHOR_KINDS.items():
        dev = split_ids(len(tpls), N_DEV_ANCHORS, f"anchors2:{kind}")
        for t, tpl in enumerate(tpls):
            for item in items:
                rows.append({"id": f"anchors2/{kind}/{t}/{item}", "text": tpl.format(X=item),
                             "split": "dev" if t in dev else "heldout", "family": "anchors2", "aspect": kind,
                             "referent": None, "template": t, "item": item})
    return rows


def write_anchors(note: str = "") -> dict:
    """Write prompts/anchors_dev.jsonl and prompts/anchors_heldout.jsonl and append a manifest version that keeps the
    current dev.jsonl / heldout.jsonl records unchanged (those files are not rewritten or read) and adds the anchor
    files' sha256 and counts. Idempotent: an unchanged build returns the current version."""
    rows = build_anchors()
    man = json.loads(MANIFEST.read_text())
    cur = man["history"][man["current"] - 1]
    entry = {"written_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "note": note, "seed": SEED,
             "templates_sha256": sha((HERE / "templates.py").read_bytes()),
             "files": {n: cur["files"][n] for n in ("dev.jsonl", "heldout.jsonl")}}
    for name, split in (("anchors_dev.jsonl", "dev"), ("anchors_heldout.jsonl", "heldout")):
        rs = [r for r in rows if r["split"] == split]
        b = dumps(rs)
        counts: dict = {}
        for r in rs:
            counts[r["aspect"]] = counts.get(r["aspect"], 0) + 1
        entry["files"][name] = {"sha256": sha(b), "rows": len(rs), "counts": dict(sorted(counts.items()))}
        (PROMPTS / name).write_bytes(b)
    if all(cur["files"].get(n, {}).get("sha256") == entry["files"][n]["sha256"] for n in entry["files"]):
        return cur
    entry["version"] = len(man["history"]) + 1
    man["history"].append(entry)
    man["current"] = entry["version"]
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="")
    ap.add_argument("--anchors", action="store_true", help="write the Stage 2 anchor block only")
    a = ap.parse_args()
    if a.anchors:
        e = write_anchors(a.note)
        print(json.dumps({"version": e["version"], **{n: {k: v for k, v in f.items() if k != "counts"}
                                                      for n, f in e["files"].items()}}, indent=1))
        return
    e = write(a.note)
    print(json.dumps({"version": e["version"], **{n: {k: v for k, v in f.items() if k != "counts"}
                                                  for n, f in e["files"].items()}}, indent=1))


if __name__ == "__main__":
    main()
