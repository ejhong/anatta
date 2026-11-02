"""The sightings helper: appends a stub entry for a sweep to lab/explore/SIGHTINGS.md, with the time read from the
system clock (UTC), the sweep, its path and the variant count. The person logging it fills in what looked alive and
the next step. Claude wrote this file. Private (lab/explore)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .schema import read_jsonl


def _clock() -> datetime:
    return datetime.now(timezone.utc)


def stub(sweep_dir, *, variants_file, note: str | None = None, clock=_clock) -> str:
    sd = Path(sweep_dir)
    man = json.loads((sd / "run.json").read_text()) if (sd / "run.json").exists() else {}
    vc = man.get("variant_counts") or {}
    rows = read_jsonl(variants_file)
    track = len({(r["kind"], r["name"], r.get("sha256")) for r in rows}) if rows else vc.get("track_distinct", "?")
    now = clock().astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sweep = man.get("sweep") or sd.name
    return "\n".join([
        f"## {now}: {sweep} (stub)",
        "",
        f"- **Sweep:** {sweep}, run {man.get('run_id', '?')}, status {man.get('status', '?')}.",
        f"- **Data:** {sd} (report: {sd / 'report.html'}).",
        f"- **Variants tried:** {vc.get('this_sweep', '?')} conditions in this sweep ({vc.get('cells', '?')} cells: "
        f"{vc.get('models', '?')} models x {vc.get('families', '?')} families); {track} distinct interventions in the "
        f"explore track to date.",
        f"- **What looked alive:** {note or '(fill in, with intervals)'}",
        "- **Next step:** replicate, drop or promote (fill in).",
        "",
    ])


def append_sighting(sweep_dir, *, sightings_file, variants_file, note: str | None = None, clock=_clock) -> str:
    entry = stub(sweep_dir, variants_file=variants_file, note=note, clock=clock)
    f = Path(sightings_file)
    text = f.read_text(encoding="utf-8") if f.exists() else "# Sightings\n"
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(sep + entry)
    return entry
