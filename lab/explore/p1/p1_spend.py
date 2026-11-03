"""P1's spend meter, in bench's meter.py format. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private (lab/explore).

Each Modal call's container time is charged to the explore track's shared daily file through anatta.spend.add_spend
(re-exported by scout.spend; <results>/spend/<UTC date>.json, under its lock, as tool "p1"), and line by line to P1's own spend.jsonl, whose sum is
P1's spend so far. The rate is bench's meter.rate_per_hour for P1's container: the L40S at $1.95 an hour plus 4 cores
and 32 GiB at the lab's list prices, $3.28 an hour. Modal's usage page is the authority.

As bench's meter does it: before a call the meter charges the call's upper estimate (start-up plus the stage's
estimate), so a killed client leaves the daily file overstated, never understated; after the call it trues up to the
measured wall time plus the scaledown tail (a negative entry when the estimate was high).

Caps: a call does not start if its estimate would take P1 past its cap ($6 for all of P1) or the day's explore total
past the daily stop ($45, the brief's line, under scout's $50 cap).
"""

from __future__ import annotations

import time
from pathlib import Path

import p1_core  # noqa: F401  (puts lab/explore and bench on sys.path)
from scout import spend as scout_spend
from scout.schema import append_jsonl, read_jsonl, utc_date, utc_now

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

    def _charge(self, usd: float, kind: str, calls: int, **meta) -> None:
        scout_spend.add_spend(self.spend_dir, tool=TOOL, label=self.label, usd=usd, calls=calls, date=self.date_fn())
        append_jsonl(self.log, {"ts": utc_now(), "tool": TOOL, "label": self.label, "cost_usd": round(usd, 8),
                                "kind": kind, "rate_usd_per_hour": self.rate, "session": self.session, **meta})

    def reserve(self, estimate_s: float, what: str) -> None:
        """Refuse (CapReached) when the estimate would pass P1's cap or the day's stop; else charge it up front."""
        usd = self.usd(estimate_s)
        spent, today = self.p1_spent(), self.today()
        if spent + usd > self.cap:
            raise CapReached(f"refused {what}: P1 has spent ${spent:.4f}; with this call's estimate ${usd:.4f} it "
                             f"would pass its cap ${self.cap:.2f}")
        if today + usd > self.daily_stop:
            raise CapReached(f"refused {what}: today's explore spend ${today:.4f} + this call's estimate ${usd:.4f} "
                             f"would pass the daily stop ${self.daily_stop:.2f}")
        self._charge(usd, "reserve", 0, what=what, seconds=round(estimate_s, 3))
        self.t0, self.reserved_s = self.clock(), float(estimate_s)

    def settle(self, what: str, **meta) -> float:
        """True up to the wall time since reserve() plus the tail; returns this call's charge in dollars."""
        actual_s = self.clock() - self.t0 + self.tail_s
        delta = self.usd(actual_s - self.reserved_s)
        self._charge(delta, "call", 1, what=what, seconds=round(actual_s, 3), reserved_s=round(self.reserved_s, 3),
                     **meta)
        self.t0 = self.reserved_s = None
        return self.usd(actual_s)
