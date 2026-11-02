"""Spend: every paid call is logged, and no call is made that could take the sweep past its cap or the day past
$50. Shared by every explore tool. Claude wrote this file. Private (lab/explore).

The daily file, <results>/spend/<YYYY-MM-DD>.json (UTC date), shared by all explore tools:
  {"date", "cap_usd": 50.0, "total_usd", "calls", "by_tool": {tool: {"usd", "calls"}},
   "by_sweep": {"<tool>:<label>": {"tool", "usd", "calls"}}, "updated"}
It is read and rewritten under an exclusive lock (<date>.json.lock, fcntl.flock) and replaced atomically, so two
tools running at once both count. Another tool adds its spend with add_spend(); a GPU tool can add its hourly cost
the same way.

The sweep's own log, <sweep dir>/spend.jsonl, has one line per paid call (cost, tokens, model, kind); its sum is
the sweep's spend, so a resumed sweep keeps counting what it already paid.

SpendGuard is cal_judges.Budget's rule for both caps: a call is made only if spent + the reservations of calls in
flight + its own reservation (a cautious bound on its cost) stay within the cap. Once a call is refused, the guard
stops and the sweep ends cleanly. The other tools' in-flight calls are not visible to this process; that window is a
few calls' worth of cents.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from .schema import append_jsonl, read_jsonl, utc_date, utc_now

DAILY_CAP_USD = 50.0


class CapReached(SystemExit):
    """The sweep's cap or the day's cap would be passed (exits the CLI with code 3 and this message)."""

    def __init__(self, reason: str):
        super().__init__(3)
        self.reason = reason

    def __str__(self):
        return self.reason


def day_file(spend_dir, date: str) -> Path:
    return Path(spend_dir) / f"{date}.json"


def _empty(date: str) -> dict:
    return {"date": date, "cap_usd": DAILY_CAP_USD, "total_usd": 0.0, "calls": 0, "by_tool": {}, "by_sweep": {},
            "updated": None}


def read_day(spend_dir, date: str | None = None) -> dict:
    f = day_file(spend_dir, date or utc_date())
    if not f.exists():
        return _empty(date or utc_date())
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"the daily spend file {f} is unreadable ({e}); fix it before spending more") from None


def spent_today(spend_dir, date: str | None = None) -> float:
    return float(read_day(spend_dir, date).get("total_usd", 0.0))


@contextmanager
def _locked(spend_dir, date: str):
    d = Path(spend_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{date}.json.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def add_spend(spend_dir, *, tool: str, label: str, usd: float, calls: int = 1, date: str | None = None) -> float:
    """Add a cost to the day's file (under the lock); returns the day's new total."""
    date = date or utc_date()
    with _locked(spend_dir, date):
        d = read_day(spend_dir, date)
        d["total_usd"] = round(float(d.get("total_usd", 0.0)) + float(usd), 8)
        d["calls"] = int(d.get("calls", 0)) + calls
        t = d.setdefault("by_tool", {}).setdefault(tool, {"usd": 0.0, "calls": 0})
        t["usd"] = round(t["usd"] + float(usd), 8)
        t["calls"] += calls
        s = d.setdefault("by_sweep", {}).setdefault(f"{tool}:{label}", {"tool": tool, "usd": 0.0, "calls": 0})
        s["usd"] = round(s["usd"] + float(usd), 8)
        s["calls"] += calls
        d["updated"] = utc_now()
        f = day_file(spend_dir, date)
        tmp = f.with_name(f.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, f)
        return d["total_usd"]


def sweep_spent(sweep_log) -> float:
    return sum(float(r.get("cost_usd", 0.0)) for r in read_jsonl(sweep_log))


class SpendGuard:
    """Both caps for one sweep. reserve(bound) before a call; settle(bound, cost, meta) after it is paid; release(bound)
    if it failed unpaid."""

    def __init__(self, spend_dir, *, sweep_cap: float, sweep_log, tool: str, label: str,
                 daily_cap: float = DAILY_CAP_USD, date_fn=utc_date):
        if not 0 < float(sweep_cap) <= float(daily_cap):
            raise SystemExit(f"the sweep's spend cap must be above 0 and at most the daily cap ${daily_cap:.2f}; "
                             f"got {sweep_cap}")
        self.spend_dir, self.sweep_log = Path(spend_dir), Path(sweep_log)
        self.sweep_cap, self.daily_cap = float(sweep_cap), float(daily_cap)
        self.tool, self.label, self.date_fn = tool, label, date_fn
        self.sweep_spent = sweep_spent(self.sweep_log)
        self.start_spent = self.sweep_spent
        self.reserved = 0.0
        self.calls = 0
        self.stopped: str | None = None
        self.lock = threading.Lock()

    def today(self) -> float:
        return spent_today(self.spend_dir, self.date_fn())

    def check_start(self, estimate: float) -> None:
        """Refuse to start when the estimate would take the sweep past its cap or the day past the daily cap."""
        today = self.today()
        if today + estimate > self.daily_cap:
            raise CapReached(f"refused to start: today's explore spend ${today:.4f} + this sweep's estimate "
                             f"${estimate:.4f} would pass the daily cap ${self.daily_cap:.2f}")
        if self.sweep_spent + estimate > self.sweep_cap:
            raise CapReached(f"refused to start: the sweep has spent ${self.sweep_spent:.4f}; with the estimate "
                             f"${estimate:.4f} it would pass its cap ${self.sweep_cap:.2f}")

    def reserve(self, bound: float) -> bool:
        with self.lock:
            if self.stopped:
                return False
            today = self.today()
            if today + self.reserved + bound > self.daily_cap:
                self.stopped = (f"stopped at the daily cap: today ${today:.4f} + in flight ${self.reserved:.4f} + next "
                                f"call ${bound:.4f} > ${self.daily_cap:.2f}")
                return False
            if self.sweep_spent + self.reserved + bound > self.sweep_cap:
                self.stopped = (f"stopped at the sweep cap: spent ${self.sweep_spent:.4f} + in flight "
                                f"${self.reserved:.4f} + next call ${bound:.4f} > ${self.sweep_cap:.2f}")
                return False
            self.reserved += bound
            return True

    def release(self, bound: float) -> None:
        with self.lock:
            self.reserved = max(0.0, self.reserved - bound)

    def settle(self, bound: float, cost: float, meta: dict) -> None:
        with self.lock:
            self.reserved = max(0.0, self.reserved - bound)
            self.sweep_spent += float(cost)
            self.calls += 1
            append_jsonl(self.sweep_log, {"ts": utc_now(), "tool": self.tool, "label": self.label,
                                          "cost_usd": round(float(cost), 8), **meta})
            add_spend(self.spend_dir, tool=self.tool, label=self.label, usd=float(cost), date=self.date_fn())

    @property
    def paid_this_run(self) -> float:
        return round(self.sweep_spent - self.start_spent, 8)
