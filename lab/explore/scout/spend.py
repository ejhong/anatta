"""Spend for the explore tools: a thin re-export of the lab's one spend module, anatta.spend (src/anatta/spend.py),
which holds the day file, add_spend, SpendGuard and the GPU Meter. scout, bench, p1, act1 and the sweeps keep
importing `scout.spend`; the code lives in one place. Claude wrote this file; made a re-export on 3 Nov 2026.

The daily cap is the day file's cap_usd when present, else lab.toml's spend.daily_cap_usd (see anatta.spend).
DAILY_CAP_USD below is lab.toml's default, kept as a name for old callers; caps are checked against daily_cap().
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .schema import append_jsonl, read_jsonl, utc_date, utc_now  # noqa: F401  (old names of this module)


def _anatta_spend() -> ModuleType:
    """main's anatta.spend. TODO(clean-lab step 2): becomes `from anatta import spend` once the lab imports main's
    library. Until then a laptop tool may have the pinned anatta-dev first on sys.path (it has no spend.py), so main's
    package is loaded from this repo under its own name, _anatta_main."""
    try:
        from anatta import spend

        return spend
    except ImportError:
        pass
    name = "_anatta_main"
    if name not in sys.modules:
        src = Path(__file__).resolve().parents[3] / "src" / "anatta"
        spec = importlib.util.spec_from_file_location(name, src / "__init__.py", submodule_search_locations=[str(src)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{name}.spend")


core = _anatta_spend()

CapReached = core.CapReached
SpendGuard = core.SpendGuard
Meter = core.Meter
add_spend = core.add_spend
daily_cap = core.daily_cap
day_file = core.day_file
default_spend_dir = core.default_spend_dir
rate_per_hour = core.rate_per_hour
read_day = core.read_day
read_state = core.read_state
refund_tail = core.refund_tail
spent_today = core.spent_today
state_lock = core.state_lock
sweep_spent = core.sweep_spent
DAILY_CAP_USD = core.default_daily_cap()
