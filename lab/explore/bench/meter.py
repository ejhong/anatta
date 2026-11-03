"""bench's spend meter: a thin wrapper of the lab's one GPU Meter (anatta.spend.Meter, re-exported by scout.spend)
with tool "bench". Container time is charged to the shared day file and line by line to the sweep's own spend.jsonl;
per-model tails are kept in spend/bench-state.json. Claude (Opus 5.5) wrote this file, 1 Nov 2026; made a wrapper on
3 Nov 2026 (the behaviour, documented in src/anatta/spend.py, is unchanged).

The rate is the bench container's: the L40S plus its 2 cores and 16 GiB at the lab's Modal list prices
(rate_per_hour). Caps: a session refuses to start, or stops before a call, when its charge would pass its own cap or
take the day's total past the day's cap (the day file's cap_usd, else lab.toml's).
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

BENCH = Path(__file__).resolve().parent
if str(BENCH.parent) not in sys.path:
    sys.path.insert(0, str(BENCH.parent))
from scout import spend as scout_spend  # noqa: E402
from scout.schema import utc_date  # noqa: E402

TOOL = "bench"
GPU_USD_PER_HOUR = scout_spend.core.GPU_USD_PER_HOUR
CPU_USD_PER_CORE_HOUR, MEM_USD_PER_GIB_HOUR = scout_spend.core.CPU_USD_PER_CORE_HOUR, scout_spend.core.MEM_USD_PER_GIB_HOUR
DAILY_CAP_USD = scout_spend.DAILY_CAP_USD  # lab.toml's default; the day's cap is daily_cap()
CapReached = scout_spend.CapReached
rate_per_hour = scout_spend.rate_per_hour
daily_cap = scout_spend.daily_cap
read_state = partial(scout_spend.read_state, tool=TOOL)


class Meter(scout_spend.Meter):
    """anatta.spend.Meter for tool "bench" (same keyword arguments, without `tool`)."""

    def __init__(self, **kw):
        super().__init__(tool=TOOL, **kw)


def refund_tail(spend_dir, model_key: str | None = None, clock=time.time, date_fn=utc_date) -> float:
    """After the bench's containers are stopped: give back each model's unused charged tail. Returns dollars."""
    return scout_spend.refund_tail(spend_dir, TOOL, model_key=model_key, clock=clock, date_fn=date_fn)
