"""bench's spend meter: container time charged to the explore track's shared daily file through scout's
spend.add_spend (<results>/spend/<UTC date>.json, under its lock, as tool "bench"), and line by line to the sweep's
own spend.jsonl. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private (lab/explore).

The rate is the bench container's: the L40S at $1.95 an hour plus the container's 2 cores and 16 GiB at the Modal
list prices the lab uses (plans/archive/calibration-c7-fable-code-check.md:155, about $0.14 a core-hour and $0.024 a
GiB-hour), $2.61 an hour in all. Modal's usage page is the authority.

A session (one client command) is charged for the wall time from its start to its end, which covers the cold start
and every call, plus the scaledown tail (the container idles SCALEDOWN_S before it stops). Time is charged once:
spend/bench-state.json keeps, per model, how far the last charge ran (paid_until) and which session paid it. A session
that starts while that tail is still running takes it over: the unused part is credited to the session that paid it
and debited to the new one (the day's total does not move), so each session pays for its own time, the idle gap
before the next session counts to the one before it, and the last session holds the tail that `client.py stop`
refunds.

Crash safety: before each call the session charges as if the call ran to its estimate and the tail followed, so a
killed client leaves the daily file overstated, never understated; after the call it trues up (a negative entry when
the estimate was high).

Caps: a session refuses to start, or stops before a call, when its charge would pass its own cap or take the day's
total (every explore tool's) past $50.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

BENCH = Path(__file__).resolve().parent
if str(BENCH.parent) not in sys.path:
    sys.path.insert(0, str(BENCH.parent))
from scout import spend as scout_spend  # noqa: E402
from scout.schema import append_jsonl, utc_date, utc_now  # noqa: E402

TOOL = "bench"
GPU_USD_PER_HOUR = {"L40S": 1.95}
CPU_USD_PER_CORE_HOUR, MEM_USD_PER_GIB_HOUR = 0.14, 0.024
DAILY_CAP_USD = scout_spend.DAILY_CAP_USD
CapReached = scout_spend.CapReached


def rate_per_hour(gpu: str = "L40S", cores: float = 2.0, mem_gib: float = 16.0) -> dict:
    cpu_mem = cores * CPU_USD_PER_CORE_HOUR + mem_gib * MEM_USD_PER_GIB_HOUR
    return {"gpu": GPU_USD_PER_HOUR[gpu], "cpu_mem": round(cpu_mem, 4),
            "total": round(GPU_USD_PER_HOUR[gpu] + cpu_mem, 4), "gpu_type": gpu, "cores": cores, "mem_gib": mem_gib}


@contextmanager
def _state_lock(spend_dir: Path):
    spend_dir.mkdir(parents=True, exist_ok=True)
    with open(spend_dir / "bench-state.json.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_state(spend_dir) -> dict:
    f = Path(spend_dir) / "bench-state.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _write_state(spend_dir: Path, state: dict) -> None:
    f = spend_dir / "bench-state.json"
    tmp = f.with_name(f.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    os.replace(tmp, f)


class Meter:
    """One session's charge. start() before the first call, before_call(est) and after_call() around each call,
    close() at the end. The charge is kept in seconds; every change goes to the daily file as a delta."""

    def __init__(self, *, spend_dir, label: str, session: str, model_key: str, cap_usd: float, rate: float,
                 tail_s: float, sweep_log=None, daily_cap: float = DAILY_CAP_USD, clock=time.time, date_fn=utc_date):
        if not 0 < float(cap_usd) <= float(daily_cap):
            raise SystemExit(f"a session's cap must be above 0 and at most the daily cap ${daily_cap:.2f}; got {cap_usd}")
        self.spend_dir, self.sweep_log = Path(spend_dir), (Path(sweep_log) if sweep_log else None)
        self.label, self.session, self.model_key = label, session, model_key
        self.cap, self.rate, self.tail_s, self.daily_cap = float(cap_usd), float(rate), float(tail_s), float(daily_cap)
        self.clock, self.date_fn = clock, date_fn
        self.charge_from = None
        self.charged_s = 0.0
        self.calls = 0

    def usd(self, seconds: float) -> float:
        return seconds * self.rate / 3600

    @property
    def charged_usd(self) -> float:
        return round(self.usd(self.charged_s), 8)

    def _need_s(self, until: float, tail: bool = True) -> float:
        return max(0.0, until + (self.tail_s if tail else 0.0) - self.charge_from)

    def _check(self, need_s: float, what: str) -> None:
        need, delta = self.usd(need_s), self.usd(need_s - self.charged_s)
        today = scout_spend.spent_today(self.spend_dir, self.date_fn())
        if need > self.cap:
            raise CapReached(f"{what}: this session's charge would reach ${need:.4f}, past its cap ${self.cap:.2f}")
        if today + delta > self.daily_cap:
            raise CapReached(f"{what}: today's explore spend ${today:.4f} + ${delta:.4f} more would pass the daily cap "
                             f"${self.daily_cap:.2f}")

    def _charge_to(self, need_s: float, kind: str, calls: int = 0, **meta) -> None:
        delta_s = need_s - self.charged_s
        usd = self.usd(delta_s)
        if abs(usd) < 1e-9 and not calls:
            return
        with _state_lock(self.spend_dir):
            scout_spend.add_spend(self.spend_dir, tool=TOOL, label=self.label, usd=usd, calls=calls,
                                  date=self.date_fn())
            if self.sweep_log:
                append_jsonl(self.sweep_log, {"ts": utc_now(), "tool": TOOL, "label": self.label,
                                              "cost_usd": round(usd, 8), "kind": kind, "seconds": round(delta_s, 3),
                                              "rate_usd_per_hour": self.rate, "session": self.session,
                                              "model": self.model_key, **meta})
            self.charged_s = need_s
            state = read_state(self.spend_dir)
            state[self.model_key] = {"paid_until": self.charge_from + self.charged_s, "label": self.label,
                                     "session": self.session, "rate_usd_per_hour": self.rate,
                                     "sweep_log": str(self.sweep_log) if self.sweep_log else None, "updated": utc_now()}
            _write_state(self.spend_dir, state)

    def start(self, first_estimate_s: float) -> None:
        """Refuse (CapReached) if the first call and the tail would pass a cap; else take over any tail the model's
        previous session left prepaid and charge the rest up front. A refusal moves no money."""
        now = self.clock()
        prev = read_state(self.spend_dir).get(self.model_key, {})
        left = max(0.0, float(prev.get("paid_until", 0.0)) - now)
        self.charge_from, self.charged_s = now, left  # the taken-over seconds count as this session's, already paid
        need = self._need_s(now + first_estimate_s)
        self._check(need, "refused to start")
        if left:
            self._take_over(prev, left)
        self._charge_to(need, "reserve")

    def _take_over(self, prev: dict, left_s: float) -> None:
        usd = self.usd(left_s)
        with _state_lock(self.spend_dir):
            for label, sign, log, other in ((prev.get("label", "?"), -1, prev.get("sweep_log"), self.session),
                                            (self.label, 1, self.sweep_log, prev.get("session"))):
                scout_spend.add_spend(self.spend_dir, tool=TOOL, label=label, usd=sign * usd, calls=0,
                                      date=self.date_fn())
                if log:
                    append_jsonl(log, {"ts": utc_now(), "tool": TOOL, "label": label, "cost_usd": round(sign * usd, 8),
                                       "kind": "tail_handed_over" if sign < 0 else "tail_taken_over",
                                       "seconds": round(sign * left_s, 3), "rate_usd_per_hour": self.rate,
                                       "session": prev.get("session") if sign < 0 else self.session,
                                       "other_session": other, "model": self.model_key})

    def before_call(self, estimate_s: float) -> None:
        need = self._need_s(self.clock() + estimate_s)
        self._check(need, "stopped before the next call")
        if need > self.charged_s:
            self._charge_to(need, "reserve")

    def after_call(self, **meta) -> None:
        self.calls += 1
        self._charge_to(self._need_s(self.clock()), "call", calls=1, **meta)

    def close(self, stopped: bool = False) -> None:
        """The final charge: to now, plus the tail unless the container was stopped."""
        self._charge_to(self._need_s(self.clock(), tail=not stopped), "close", stopped=stopped)


def refund_tail(spend_dir, model_key: str | None = None, clock=time.time, date_fn=utc_date) -> float:
    """After the bench's containers are stopped: give back the part of each model's charged tail that had not run
    yet (paid_until after now), as a negative entry in the daily file and the sweep's spend.jsonl. Returns dollars."""
    spend_dir = Path(spend_dir)
    total = 0.0
    with _state_lock(spend_dir):
        state = read_state(spend_dir)
        now = clock()
        for key, s in state.items():
            if model_key and key != model_key:
                continue
            left = float(s.get("paid_until", 0.0)) - now
            if left <= 0:
                continue
            usd = left * float(s["rate_usd_per_hour"]) / 3600
            scout_spend.add_spend(spend_dir, tool=TOOL, label=s["label"], usd=-usd, calls=0, date=date_fn())
            if s.get("sweep_log"):
                append_jsonl(s["sweep_log"], {"ts": utc_now(), "tool": TOOL, "label": s["label"],
                                              "cost_usd": round(-usd, 8), "kind": "refund", "seconds": round(-left, 3),
                                              "rate_usd_per_hour": s["rate_usd_per_hour"], "session": s.get("session"),
                                              "model": key})
            s["paid_until"], s["updated"] = now, utc_now()
            total += usd
        _write_state(spend_dir, state)
    return round(total, 8)
