"""Spend for the explore tools: a thin re-export of the lab's one spend module, anatta.spend (src/anatta/spend.py),
which holds the day file, add_spend, SpendGuard and the GPU Meter. scout, bench, p1, act1 and the sweeps keep
importing `scout.spend`; the code lives in one place. Claude wrote this file; made a re-export on 3 Nov 2026.

The daily cap is the day file's cap_usd when present, else lab.toml's spend.daily_cap_usd (see anatta.spend).
DAILY_CAP_USD below is lab.toml's default, kept as a name for old callers; caps are checked against daily_cap().
"""

from __future__ import annotations

from anatta import spend as core

from .schema import append_jsonl, read_jsonl, utc_date, utc_now  # noqa: F401  (old names of this module)

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
