"""P1's spend meter: a thin wrapper of the lab's one GPU Meter (anatta.spend.Meter, re-exported by scout.spend) with
tool "p1". Claude (Opus 5.5) wrote this file, 1 Nov 2026; made a wrapper on 3 Nov 2026. P1 is finished: the wrapper keeps
its recorded behaviour (charges, caps, the day file and spend.jsonl lines), checked by lab/explore/p1/tests/test_p1.py.
Private (lab/explore).

Each Modal call is one Meter session. reserve(estimate_s) starts it, charging the call's upper estimate up front (the
Meter charges its estimate plus the scaledown tail, so it is started at estimate - tail), and a killed client leaves
the day file overstated, never understated; settle() trues up to the measured wall time plus the tail (Meter.after_call;
a negative entry when the estimate was high). Every line goes to the shared day file (tool "p1") and to P1's own
spend.jsonl, whose sum is P1's spend so far. The rate is anatta.spend.rate_per_hour for P1's container: the L40S at
$1.95 an hour plus 4 cores and 32 GiB at the lab's list prices, $3.28 an hour. Modal's usage page is the authority.

Caps (both passed in by the caller, from P1's spec): a call does not start if its estimate would take P1 past its cap
(the session's cap is what is left of P1's) or the day's total past P1's daily stop (the session's daily cap).
The Meter also writes <spend dir>/p1-state.json; each call is its own model key there, so no call takes over another's
tail. The refusal messages are the Meter's wording.
"""

from __future__ import annotations

import time
from pathlib import Path

import p1_core  # noqa: F401  (puts lab/explore and bench on sys.path)
from scout import spend as scout_spend
from scout.schema import read_jsonl, utc_date

TOOL = "p1"
rate_per_hour = scout_spend.rate_per_hour  # anatta.spend's Modal rates (bench's, unchanged)
CapReached = scout_spend.CapReached


def p1_rate(rate_spec: dict) -> float:
    return float(rate_per_hour(rate_spec["gpu"], float(rate_spec["cores"]), float(rate_spec["mem_gib"]))["total"])


class P1Meter:
    """One call's charge: reserve(estimate_s) before it, settle() after it (even when it fails)."""

    def __init__(self, *, spend_dir, log, label: str, session: str, rate: float, cap_usd: float, daily_stop_usd: float,
                 tail_s: float = 2.0, clock=time.time, date_fn=utc_date):
        self.spend_dir, self.log = Path(spend_dir), Path(log)
        self.label, self.session, self.rate = label, session, float(rate)
        self.cap, self.daily_stop, self.tail_s = float(cap_usd), float(daily_stop_usd), float(tail_s)
        self.clock, self.date_fn = clock, date_fn
        self.t0 = self.reserved_s = None
        self._meter: scout_spend.Meter | None = None

    def usd(self, seconds: float) -> float:
        return seconds * self.rate / 3600

    def p1_spent(self) -> float:
        """P1's spend so far: the sum of its spend.jsonl (every P1 call, smoke, full and any second model)."""
        return round(sum(float(r.get("cost_usd", 0.0)) for r in read_jsonl(self.log)), 8) if self.log.exists() else 0.0

    def today(self) -> float:
        return scout_spend.spent_today(self.spend_dir, self.date_fn())

    def left_s(self) -> float:
        """Seconds the rest of P1's cap buys at the rate."""
        return max(0.0, (self.cap - self.p1_spent()) / self.rate * 3600)

    def reserve(self, estimate_s: float, what: str) -> None:
        """Refuse (CapReached) when the estimate would pass P1's cap or the day's stop; else charge it up front."""
        left = round(self.cap - self.p1_spent(), 8)
        if left <= 0:
            raise CapReached(f"refused {what}: P1 has spent its cap ${self.cap:.2f}")
        self._meter = scout_spend.Meter(
            tool=TOOL, spend_dir=self.spend_dir, label=self.label, session=self.session,
            model_key=f"{self.session}:{what}", cap_usd=min(left, self.daily_stop), rate=self.rate, tail_s=self.tail_s,
            sweep_log=self.log, daily_cap=self.daily_stop, clock=self.clock, date_fn=self.date_fn)
        self._meter.start(max(0.0, float(estimate_s) - self.tail_s))
        self.t0, self.reserved_s = self._meter.charge_from, float(estimate_s)

    def settle(self, what: str, **meta) -> float:
        """True up to the wall time since reserve() plus the tail; returns this call's charge in dollars."""
        self._meter.after_call(what=what, reserved_s=round(self.reserved_s, 3), **meta)
        usd = self._meter.charged_usd
        self._meter, self.t0, self.reserved_s = None, None, None
        return usd
