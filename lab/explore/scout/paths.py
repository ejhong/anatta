"""Where scout reads and writes. Results live outside git, under RESULTS_ROOT (override with SCOUT_RESULTS_ROOT, as the
tests do with a temporary directory). Claude wrote this file. Private (lab/explore)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

SCOUT = Path(__file__).resolve().parent
EXPLORE = SCOUT.parent  # lab/explore
SCENARIOS = EXPLORE / "scenarios"
DEV_DIR = SCENARIOS / "dev"
HELDOUT_DIR = SCENARIOS / "heldout"
HELDOUT_FILE = HELDOUT_DIR / "heldout.yaml"
RUBRICS_FILE = SCENARIOS / "rubrics.yaml"
FRAMINGS_FILE = SCENARIOS / "framings.yaml"
LIBRARY_README = SCENARIOS / "README.md"
CONDITIONS_FILE = EXPLORE / "conditions" / "conditions.yaml"
SIGHTINGS_FILE = EXPLORE / "SIGHTINGS.md"
SPECS = SCOUT / "specs"
DEFAULT_RESULTS_ROOT = Path("/Users/eugene/prj/align/results/explore")
SWEEP_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class Paths:
    """Every output location, derived from one results root."""

    root: Path

    @classmethod
    def default(cls) -> "Paths":
        return cls(Path(os.environ.get("SCOUT_RESULTS_ROOT") or DEFAULT_RESULTS_ROOT))

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def spend(self) -> Path:
        return self.root / "spend"

    @property
    def variants(self) -> Path:
        return self.root / "variants.jsonl"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration" / "hand_labels.jsonl"

    def sweep_dir(self, date: str, sweep: str) -> Path:
        if not SWEEP_NAME.match(sweep):
            raise ValueError(f"sweep name {sweep!r}: lower-case letters, digits, '.', '_' or '-', at most 64")
        return self.root / date / sweep

    def resolve_sweep(self, arg: str) -> Path:
        """A sweep directory from a path, or from '<YYYY-MM-DD>/<sweep>' under the root."""
        p = Path(arg).expanduser()
        if p.is_dir():
            return p.resolve()
        q = self.root / arg
        if q.is_dir():
            return q
        raise SystemExit(f"no sweep directory at {arg} (or {q})")
