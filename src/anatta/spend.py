"""The lab's one spend module: the shared day file, add_spend, SpendGuard (per-call caps for API sweeps) and Meter
(container time for GPU sessions). Ported from lab/explore/scout/spend.py and lab/explore/bench/meter.py, behaviour
unchanged except where the daily cap comes from. Stdlib only. Claude (Opus 5.5) wrote this file, 3 Nov 2026.

The day file, <spend dir>/<YYYY-MM-DD>.json (UTC date), shared by every tool:
  {"date", "cap_usd", "total_usd", "calls", "by_tool": {tool: {"usd", "calls"}},
   "by_sweep": {"<tool>:<label>": {"tool", "usd", "calls"}}, "updated"}
It is read and rewritten under an exclusive lock (<date>.json.lock, fcntl.flock) and replaced atomically, so tools
running at once all count. This module is the only writer of it.

The daily cap: the day file's own cap_usd when present (Eugene's raise for that day is recorded there, plus a ledger
line; a cap_usd of 0 means 0, nothing may be spent that day), else lab.toml's spend.daily_cap_usd. A new day file is
created with lab.toml's cap. Nothing else holds a cap.

Spend is booked laptop-side only: the laptop process that launches the Modal calls (or makes the API calls) meters
and books them. Code running inside a Modal container never books spend and never reads lab.toml; anatta.config.load()
raises there, naming this rule.

A sweep's own log (<sweep dir>/spend.jsonl) has one line per paid call; its sum is the sweep's spend, so a resumed
sweep keeps counting what it already paid.

SpendGuard: a call is made only if spent + the reservations of calls in flight + its own reservation (a cautious
bound on its cost) stay within both the sweep's cap and the day's. Once a call is refused the guard stays stopped.
Other processes' in-flight calls are not visible; that window is a few calls' worth of cents.

Meter: one GPU session's charge, in container seconds at a rate (rate_per_hour). Before each call it charges as if
the call ran to its estimate and the scaledown tail followed, so a killed client leaves the day file overstated, never
understated; after the call it trues up. <spend dir>/<tool>-state.json keeps, per model, how far the last charge ran
(paid_until), so a session that starts inside the previous session's tail takes the tail over and every second is
charged once; refund_tail() gives back an unused tail after the containers are stopped. A session's date is pinned
at start(): a session that crosses 00:00 UTC books every charge (and its true-ups) on the day it started, so no day
gets another day's refund. Modal's usage page is the authority on what was really billed.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from . import config

PathLike = str | os.PathLike

# Modal list prices the lab uses (plans/archive/calibration-c7-fable-code-check.md:155); Modal's usage page is the
# authority. Prices, not caps.
GPU_USD_PER_HOUR = {"L40S": 1.95, "H100": 3.95, "H200": 4.54}  # Modal list prices; H100 and H200 added 5 Nov (VLAF judge)
CPU_USD_PER_CORE_HOUR, MEM_USD_PER_GIB_HOUR = 0.14, 0.024


class CapReached(SystemExit):
    """A sweep's, session's or the day's cap would be passed (exits a CLI with code 3 and this message)."""

    def __init__(self, reason: str):
        super().__init__(3)
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


# ------------------------------------------------------------------------------------------------ small helpers


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_date() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _read_jsonl(path: PathLike) -> list[dict]:
    """A torn last line (a crash mid-write) is skipped; any other bad line raises."""
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").split("\n")
    out = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if n >= len(lines) - 1:
                continue
            raise ValueError(f"{path} line {n}: not JSON") from None
    return out


def _append_jsonl(path: PathLike, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _atomic_write_json(f: Path, obj: dict) -> None:
    tmp = f.with_name(f.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, f)


@contextmanager
def _flock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ------------------------------------------------------------------------------------------------ the day file


def default_spend_dir() -> Path:
    return config.load().spend_dir


def default_daily_cap() -> float:
    """lab.toml's cap (a new day starts with it)."""
    return config.load().daily_cap_usd


def _dir(spend_dir: PathLike | None) -> Path:
    return Path(spend_dir) if spend_dir is not None else default_spend_dir()


def day_file(spend_dir: PathLike | None, date: str) -> Path:
    return _dir(spend_dir) / f"{date}.json"


def _empty(date: str) -> dict:
    return {"date": date, "cap_usd": default_daily_cap(), "total_usd": 0.0, "calls": 0, "by_tool": {},
            "by_sweep": {}, "updated": None}


def read_day(spend_dir: PathLike | None = None, date: str | None = None) -> dict:
    date = date or utc_date()
    f = day_file(spend_dir, date)
    if not f.exists():
        return _empty(date)
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"the daily spend file {f} is unreadable ({e}); fix it before spending more") from None


def daily_cap(spend_dir: PathLike | None = None, date: str | None = None) -> float:
    """The day's cap: the day file's cap_usd if present, else lab.toml's."""
    cap = read_day(spend_dir, date).get("cap_usd")
    return float(cap) if cap is not None else default_daily_cap()


def spent_today(spend_dir: PathLike | None = None, date: str | None = None) -> float:
    return float(read_day(spend_dir, date).get("total_usd", 0.0))


def add_spend(spend_dir: PathLike | None, *, tool: str, label: str, usd: float, calls: int = 1,
              date: str | None = None) -> float:
    """Add a cost (negative for a true-up or refund) to the day's file under the lock; returns the day's new total."""
    date = date or utc_date()
    d_dir = _dir(spend_dir)
    with _flock(d_dir / f"{date}.json.lock"):
        d = read_day(d_dir, date)
        d["total_usd"] = round(float(d.get("total_usd", 0.0)) + float(usd), 8)
        d["calls"] = int(d.get("calls", 0)) + calls
        t = d.setdefault("by_tool", {}).setdefault(tool, {"usd": 0.0, "calls": 0})
        t["usd"] = round(t["usd"] + float(usd), 8)
        t["calls"] += calls
        s = d.setdefault("by_sweep", {}).setdefault(f"{tool}:{label}", {"tool": tool, "usd": 0.0, "calls": 0})
        s["usd"] = round(s["usd"] + float(usd), 8)
        s["calls"] += calls
        d["updated"] = utc_now()
        _atomic_write_json(day_file(d_dir, date), d)
        return d["total_usd"]


def sweep_spent(sweep_log: PathLike) -> float:
    return sum(float(r.get("cost_usd", 0.0)) for r in _read_jsonl(sweep_log))


# ------------------------------------------------------------------------------------------------ API sweeps


class SpendGuard:
    """Both caps for one sweep. reserve(bound) before a call; settle(bound, cost, meta) after it is paid;
    release(bound) if it failed unpaid. daily_cap=None (the default) reads the day's cap at every check, so a raise
    recorded in the day file during a run counts."""

    def __init__(self, spend_dir: PathLike | None, *, sweep_cap: float, sweep_log: PathLike, tool: str, label: str,
                 daily_cap: float | None = None, date_fn: Callable[[], str] = utc_date):
        self.spend_dir, self.sweep_log = _dir(spend_dir), Path(sweep_log)
        self.tool, self.label, self.date_fn = tool, label, date_fn
        self._daily_cap = None if daily_cap is None else float(daily_cap)
        if not 0 < float(sweep_cap) <= self.daily_cap:
            raise SystemExit(f"the sweep's spend cap must be above 0 and at most the daily cap "
                             f"${self.daily_cap:.2f}; got {sweep_cap}")
        self.sweep_cap = float(sweep_cap)
        self.sweep_spent = sweep_spent(self.sweep_log)
        self.start_spent = self.sweep_spent
        self.reserved = 0.0
        self.calls = 0
        self.stopped: str | None = None
        self.lock = threading.Lock()

    @property
    def daily_cap(self) -> float:
        return self._daily_cap if self._daily_cap is not None else daily_cap(self.spend_dir, self.date_fn())

    def today(self) -> float:
        return spent_today(self.spend_dir, self.date_fn())

    def check_start(self, estimate: float) -> None:
        """Refuse to start when the estimate would take the sweep past its cap or the day past the daily cap."""
        today, cap = self.today(), self.daily_cap
        if today + estimate > cap:
            raise CapReached(f"refused to start: today's explore spend ${today:.4f} + this sweep's estimate "
                             f"${estimate:.4f} would pass the daily cap ${cap:.2f}")
        if self.sweep_spent + estimate > self.sweep_cap:
            raise CapReached(f"refused to start: the sweep has spent ${self.sweep_spent:.4f}; with the estimate "
                             f"${estimate:.4f} it would pass its cap ${self.sweep_cap:.2f}")

    def reserve(self, bound: float) -> bool:
        with self.lock:
            if self.stopped:
                return False
            today, cap = self.today(), self.daily_cap
            if today + self.reserved + bound > cap:
                self.stopped = (f"stopped at the daily cap: today ${today:.4f} + in flight ${self.reserved:.4f} + next "
                                f"call ${bound:.4f} > ${cap:.2f}")
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
            _append_jsonl(self.sweep_log, {"ts": utc_now(), "tool": self.tool, "label": self.label,
                                           "cost_usd": round(float(cost), 8), **meta})
            add_spend(self.spend_dir, tool=self.tool, label=self.label, usd=float(cost), date=self.date_fn())

    @property
    def paid_this_run(self) -> float:
        return round(self.sweep_spent - self.start_spent, 8)


# ------------------------------------------------------------------------------------------------ GPU sessions


def rate_per_hour(gpu: str = "L40S", cores: float = 2.0, mem_gib: float = 16.0) -> dict:
    cpu_mem = cores * CPU_USD_PER_CORE_HOUR + mem_gib * MEM_USD_PER_GIB_HOUR
    return {"gpu": GPU_USD_PER_HOUR[gpu], "cpu_mem": round(cpu_mem, 4),
            "total": round(GPU_USD_PER_HOUR[gpu] + cpu_mem, 4), "gpu_type": gpu, "cores": cores, "mem_gib": mem_gib}


def _state_file(spend_dir: Path, tool: str) -> Path:
    return spend_dir / f"{tool}-state.json"


@contextmanager
def state_lock(spend_dir: PathLike, tool: str) -> Iterator[None]:
    with _flock(Path(spend_dir) / f"{tool}-state.json.lock"):
        yield


def read_state(spend_dir: PathLike, tool: str) -> dict:
    f = _state_file(Path(spend_dir), tool)
    return json.loads(f.read_text()) if f.exists() else {}


class Meter:
    """One GPU session's charge, for `tool`. start() before the first call, before_call(est) and after_call() around
    each call, close() at the end. The charge is kept in seconds; every change goes to the day file as a delta."""

    def __init__(self, *, tool: str, spend_dir: PathLike | None, label: str, session: str, model_key: str,
                 cap_usd: float, rate: float, tail_s: float, sweep_log: PathLike | None = None,
                 daily_cap: float | None = None, clock: Callable[[], float] = time.time,
                 date_fn: Callable[[], str] = utc_date):
        self.tool = tool
        self.spend_dir, self.sweep_log = _dir(spend_dir), (Path(sweep_log) if sweep_log else None)
        self.label, self.session, self.model_key = label, session, model_key
        self.clock, self.date_fn = clock, date_fn
        self.date: str | None = None  # pinned at start(): the whole session books on its start day
        self._daily_cap = None if daily_cap is None else float(daily_cap)
        if not 0 < float(cap_usd) <= self.daily_cap:
            raise SystemExit(f"a session's cap must be above 0 and at most the daily cap ${self.daily_cap:.2f}; "
                             f"got {cap_usd}")
        self.cap, self.rate, self.tail_s = float(cap_usd), float(rate), float(tail_s)
        self.charge_from: float | None = None
        self.charged_s = 0.0
        self.calls = 0

    def _date(self) -> str:
        return self.date or self.date_fn()

    @property
    def daily_cap(self) -> float:
        return self._daily_cap if self._daily_cap is not None else daily_cap(self.spend_dir, self._date())

    def usd(self, seconds: float) -> float:
        return seconds * self.rate / 3600

    @property
    def charged_usd(self) -> float:
        return round(self.usd(self.charged_s), 8)

    def _need_s(self, until: float, tail: bool = True) -> float:
        return max(0.0, until + (self.tail_s if tail else 0.0) - self.charge_from)

    def _check(self, need_s: float, what: str) -> None:
        need, delta = self.usd(need_s), self.usd(need_s - self.charged_s)
        today, cap = spent_today(self.spend_dir, self._date()), self.daily_cap
        if need > self.cap:
            raise CapReached(f"{what}: this session's charge would reach ${need:.4f}, past its cap ${self.cap:.2f}")
        if today + delta > cap:
            raise CapReached(f"{what}: today's explore spend ${today:.4f} + ${delta:.4f} more would pass the daily cap "
                             f"${cap:.2f}")

    def _log(self, log: PathLike | None, row: dict) -> None:
        if log:
            _append_jsonl(log, {"ts": utc_now(), "tool": self.tool, **row})

    def _charge_to(self, need_s: float, kind: str, calls: int = 0, **meta) -> None:
        delta_s = need_s - self.charged_s
        usd = self.usd(delta_s)
        if abs(usd) < 1e-9 and not calls:
            return
        with state_lock(self.spend_dir, self.tool):
            add_spend(self.spend_dir, tool=self.tool, label=self.label, usd=usd, calls=calls, date=self._date())
            self._log(self.sweep_log, {"label": self.label, "cost_usd": round(usd, 8), "kind": kind,
                                       "seconds": round(delta_s, 3), "rate_usd_per_hour": self.rate,
                                       "session": self.session, "model": self.model_key, **meta})
            self.charged_s = need_s
            state = read_state(self.spend_dir, self.tool)
            state[self.model_key] = {"paid_until": self.charge_from + self.charged_s, "label": self.label,
                                     "session": self.session, "rate_usd_per_hour": self.rate,
                                     "sweep_log": str(self.sweep_log) if self.sweep_log else None, "updated": utc_now()}
            _atomic_write_json(_state_file(self.spend_dir, self.tool), state)

    def start(self, first_estimate_s: float) -> None:
        """Refuse (CapReached) if the first call and the tail would pass a cap; else take over any tail the model's
        previous session left prepaid and charge the rest up front. A refusal moves no money."""
        now = self.clock()
        self.date = self.date_fn()
        prev = read_state(self.spend_dir, self.tool).get(self.model_key, {})
        left = max(0.0, float(prev.get("paid_until", 0.0)) - now)
        self.charge_from, self.charged_s = now, left  # the taken-over seconds count as this session's, already paid
        need = self._need_s(now + first_estimate_s)
        self._check(need, "refused to start")
        if left:
            self._take_over(prev, left)
        self._charge_to(need, "reserve")

    def _take_over(self, prev: dict, left_s: float) -> None:
        usd = self.usd(left_s)
        with state_lock(self.spend_dir, self.tool):
            for label, sign, log, other in ((prev.get("label", "?"), -1, prev.get("sweep_log"), self.session),
                                            (self.label, 1, self.sweep_log, prev.get("session"))):
                add_spend(self.spend_dir, tool=self.tool, label=label, usd=sign * usd, calls=0, date=self._date())
                self._log(log, {"label": label, "cost_usd": round(sign * usd, 8),
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


def refund_tail(spend_dir: PathLike | None, tool: str, model_key: str | None = None,
                clock: Callable[[], float] = time.time, date_fn: Callable[[], str] = utc_date) -> float:
    """After a tool's containers are stopped: give back the part of each model's charged tail that had not run yet
    (paid_until after now), as a negative entry in the day file and the sweep's log. Returns dollars."""
    spend_dir = _dir(spend_dir)
    total = 0.0
    with state_lock(spend_dir, tool):
        state = read_state(spend_dir, tool)
        now = clock()
        for key, s in state.items():
            if model_key and key != model_key:
                continue
            left = float(s.get("paid_until", 0.0)) - now
            if left <= 0:
                continue
            usd = left * float(s["rate_usd_per_hour"]) / 3600
            add_spend(spend_dir, tool=tool, label=s["label"], usd=-usd, calls=0, date=date_fn())
            if s.get("sweep_log"):
                _append_jsonl(s["sweep_log"], {"ts": utc_now(), "tool": tool, "label": s["label"],
                                               "cost_usd": round(-usd, 8), "kind": "refund", "seconds": round(-left, 3),
                                               "rate_usd_per_hour": s["rate_usd_per_hour"],
                                               "session": s.get("session"), "model": key})
            s["paid_until"], s["updated"] = now, utc_now()
            total += usd
        _atomic_write_json(_state_file(spend_dir, tool), state)
    return round(total, 8)
