"""Scenario items in, explore-track records out, through scout's own modules (lab/explore/scout): library.py for the
dev split and its held-out guard, schema.py for the record, paths.py for where results go. The judge and the report
therefore read a bench sweep exactly as they read a scout sweep. Claude (Opus 5.5) wrote this file, 1 Nov 2026.
Private (lab/explore).

The held-out guard, in three places:
  - a spec that names a held-out item or triple, asks for another split, or points the library at the held-out
    folder is refused before anything is read or spent (scout's heldout_ids, read for ids only, and its path check);
  - only dev files are ever loaded (scout's load_family, which refuses held-out paths);
  - schema.make_record refuses any record whose split is not "dev".
Until the scenario library is built, `library: auto` falls back to fixtures/ (four threat_to_self items, ids
ts-90..93, with one held-out id, ts-99, to exercise the guard).
"""

from __future__ import annotations

import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH = Path(__file__).resolve().parent
EXPLORE = BENCH.parent
if str(EXPLORE) not in sys.path:
    sys.path.insert(0, str(EXPLORE))
from scout import library as lib  # noqa: E402
from scout import paths, schema  # noqa: E402

FIXTURE_DEV = BENCH / "fixtures" / "dev"
FIXTURE_HELDOUT = BENCH / "fixtures" / "fixture_heldout_ids.yaml"
SOURCE, PROVIDER = "bench", "modal"


def new_run_id() -> str:
    """scout's format: run-<UTC stamp>-<4 hex>."""
    return f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"


def library_source(choice) -> tuple[Path, Path, str]:
    """(dev folder, held-out ids file, label). auto: the scenario library once its dev files exist, else fixtures/."""
    choice = choice or "auto"
    if choice == "fixture" or (choice == "auto" and not any(paths.DEV_DIR.glob("*.yaml"))):
        return FIXTURE_DEV, FIXTURE_HELDOUT, "fixture"
    if choice in ("auto", "library"):
        return paths.DEV_DIR, paths.HELDOUT_FILE, "library"
    lib._refuse_heldout_path(Path(choice))
    return Path(choice), paths.HELDOUT_FILE, str(choice)


def select_items(spec: dict) -> dict:
    """The spec's dev items, in scout's forms: items = all | {per_family: N} | [dev ids] | {family: [dev ids]}.
    Returns {"families", "items": {family: [ids]}, "library": {family: load_family(...)}, "source", "heldout_file"}.
    Raises lib.HeldOutError for anything that reaches for the held-out split."""
    if str(spec.get("split", "dev")).lower() != "dev":
        raise lib.HeldOutError(f"refused: split {spec.get('split')!r}; explore sweeps use the dev split only")
    dev_dir, heldout_file, label = library_source(spec.get("library"))
    held = lib.heldout_ids(heldout_file)
    raw = spec.get("items", "all")
    named = ([str(x) for x in raw] if isinstance(raw, list) else
             [str(x) for v in raw.values() for x in (v or [])] if isinstance(raw, dict) and "per_family" not in raw
             else [])
    bad = sorted(n for n in named if n in held)
    if bad:
        raise lib.HeldOutError(f"refused: the spec names held-out item(s) {bad[:8]}; the held-out split is the confirm "
                               f"track's, and explore sweeps use the dev split only")
    fams = spec.get("families") or [f for f in lib.FAMILIES if (dev_dir / f"{f}.yaml").exists()]
    if not fams or any(f not in lib.FAMILIES for f in fams) or len(set(fams)) != len(fams):
        raise ValueError(f"families must be distinct names from {lib.FAMILIES} with dev files in {dev_dir}")
    library = {f: lib.load_family(f, dev_dir) for f in fams}
    chosen = {}
    for f in fams:
        items = library[f]["items"]
        if raw == "all":
            chosen[f] = [it["id"] for it in items]
        elif isinstance(raw, dict) and "per_family" in raw:
            groups = sorted({lib.group_of(it) for it in items})[:int(raw["per_family"])]
            chosen[f] = [it["id"] for it in items if lib.group_of(it) in groups]
        elif isinstance(raw, (list, dict)):
            wanted = named if isinstance(raw, list) else [str(x) for x in (raw.get(f) or [])]
            chosen[f] = [it["id"] for it in items if it["id"] in wanted or it.get("triple") in wanted]
        else:
            raise ValueError("items: 'all', {per_family: N}, a list of dev ids, or {family: [dev ids]}")
    known = {it["id"] for f in fams for it in library[f]["items"]} | \
            {it.get("triple") for f in fams for it in library[f]["items"] if it.get("triple")}
    missing = [n for n in named if n not in known]
    if missing:
        raise ValueError(f"no dev items {missing} in the families {fams}")
    if not any(chosen.values()):
        raise ValueError("the spec selects no items")
    return {"families": fams, "items": chosen, "library": library, "source": label, "heldout_file": str(heldout_file)}


def make_record(*, sweep: str, run_id: str, model: str, intervention: dict, family: str, item_id: str, sample: int,
                messages: list, result: dict, cost_usd: float, params: dict, latency_s: float | None,
                split: str) -> dict:
    """One sample as schema.make_record builds it, which validates it and refuses any split but dev. `split` is the
    item's, as its library file declares it (load_family has checked that file is the dev split)."""
    usage = {"prompt_tokens": int(result["prompt_tokens"]), "completion_tokens": int(result["completion_tokens"]),
             "total_tokens": int(result["prompt_tokens"]) + int(result["completion_tokens"])}
    return schema.make_record(sweep=sweep, run_id=run_id, source=SOURCE, model=model, provider=PROVIDER,
                              intervention=intervention, family=family, item_id=item_id, sample=sample,
                              messages=messages, response=result["text"], usage=usage, cost_usd=cost_usd, split=split,
                              finish_reason=result.get("finish_reason"), params=params,
                              latency_s=None if latency_s is None else round(float(latency_s), 3))
