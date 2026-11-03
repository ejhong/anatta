"""anatta.spend and anatta.config: where the daily cap comes from, concurrent adds under the lock, and the guards
refusing past a cap. Temporary folders only; the real day files are never touched. Claude (Opus 5.5) wrote this."""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pytest

from anatta import config, spend

DATE = "2026-11-01"


@pytest.fixture
def lab_toml(tmp_path, monkeypatch):
    """A private lab.toml with a $50 cap and results under tmp_path; the env overrides cleared."""
    for k in ("ANATTA_RESULTS_ROOT", "ANATTA_SPEND_DIR", "ANATTA_DAILY_CAP_USD", "ANATTA_DEFAULT_MODEL"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "lab.toml"
    f.write_text(f'results_root = "{tmp_path / "results"}"\ndefault_model = "m"\n'
                 '[spend]\ndaily_cap_usd = 50.0\ndir = "explore/spend"\n'
                 '[modal]\napp_prefix = "anatta-"\ncli = "~/modal"\n[modal.volumes]\nruns = "anatta-runs"\n'
                 '[keys]\nopenrouter_env = "OPENROUTER_API_KEY"\n')
    monkeypatch.setenv("ANATTA_LAB_TOML", str(f))
    return f


def test_config_reads_lab_toml_and_env_overrides(lab_toml, tmp_path, monkeypatch):
    c = config.load()
    assert c.daily_cap_usd == 50.0 and c.default_model == "m"
    assert c.spend_dir == tmp_path / "results" / "explore" / "spend"
    assert c.modal.volumes == {"runs": "anatta-runs"} and c.modal.app_name("x") == "anatta-x"
    assert c.key_envs == {"openrouter": "OPENROUTER_API_KEY"}
    monkeypatch.setenv("ANATTA_DAILY_CAP_USD", "12.5")
    monkeypatch.setenv("ANATTA_SPEND_DIR", str(tmp_path / "elsewhere"))
    c = config.load()
    assert c.daily_cap_usd == 12.5 and c.spend_dir == tmp_path / "elsewhere"


def test_the_repo_lab_toml_loads():
    c = config.load(config.DEFAULT_PATH)
    assert c.daily_cap_usd > 0 and c.modal.app_prefix == "anatta-"
    assert {"hf_cache", "runs"} <= set(c.modal.volumes)


def test_cap_from_lab_toml_when_no_day_file(lab_toml, tmp_path):
    d = tmp_path / "spend"
    assert spend.daily_cap(d, DATE) == 50.0
    spend.add_spend(d, tool="t", label="a", usd=1.0, date=DATE)
    assert json.loads(spend.day_file(d, DATE).read_text())["cap_usd"] == 50.0  # a new day file starts with it


def test_cap_from_the_day_file_wins(lab_toml, tmp_path):
    d = tmp_path / "spend"
    spend.add_spend(d, tool="t", label="a", usd=60.0, date=DATE)
    f = spend.day_file(d, DATE)
    day = json.loads(f.read_text())
    day["cap_usd"] = 80.0  # Eugene's raise, recorded in the day file
    f.write_text(json.dumps(day))
    assert spend.daily_cap(d, DATE) == 80.0
    g = spend.SpendGuard(d, sweep_cap=70, sweep_log=tmp_path / "s.jsonl", tool="t", label="b", date_fn=lambda: DATE)
    assert g.daily_cap == 80.0
    assert g.reserve(19.0)  # 60 + 19 <= 80, though past lab.toml's 50
    assert spend.daily_cap(d, "2026-11-02") == 50.0  # the raise is for that day only


def _add_many(args) -> None:
    d, n = args
    for _ in range(n):
        spend.add_spend(d, tool="p", label="x", usd=0.01, date=DATE)


def test_concurrent_adds_all_count(lab_toml, tmp_path):
    d = tmp_path / "spend"
    workers, n = 4, 50
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(_add_many, [(str(d), n)] * workers))
    day = spend.read_day(d, DATE)
    assert day["calls"] == workers * n
    assert day["total_usd"] == pytest.approx(0.01 * workers * n)
    assert day["by_sweep"]["p:x"]["calls"] == workers * n


def test_guard_refuses_past_the_daily_cap(lab_toml, tmp_path):
    d = tmp_path / "spend"
    spend.add_spend(d, tool="other", label="b", usd=49.90, date=DATE)
    g = spend.SpendGuard(d, sweep_cap=10, sweep_log=tmp_path / "s.jsonl", tool="t", label="a", date_fn=lambda: DATE)
    with pytest.raises(spend.CapReached, match="daily cap"):
        g.check_start(0.2)
    assert g.reserve(0.05)
    g.settle(0.05, 0.05, {"kind": "x"})
    assert not g.reserve(0.06) and "daily cap" in g.stopped
    assert not g.reserve(0.0001), "once stopped, the guard stays stopped"
    assert spend.spent_today(d, DATE) == pytest.approx(49.95)


def test_guard_refuses_past_the_sweep_cap(lab_toml, tmp_path):
    d = tmp_path / "spend"
    g = spend.SpendGuard(d, sweep_cap=1.0, sweep_log=tmp_path / "s.jsonl", tool="t", label="a", date_fn=lambda: DATE)
    assert g.reserve(0.6)
    g.settle(0.6, 0.6, {})
    assert not g.reserve(0.5) and "sweep cap" in g.stopped
    with pytest.raises(SystemExit):
        spend.SpendGuard(d, sweep_cap=51, sweep_log=tmp_path / "t.jsonl", tool="t", label="c", date_fn=lambda: DATE)


def test_meter_refuses_past_the_daily_cap(lab_toml, tmp_path):
    d = tmp_path / "spend"
    clock = [1000.0]
    spend.add_spend(d, tool="other", label="x", usd=49.5, date=DATE)
    m = spend.Meter(tool="gpu", spend_dir=d, label="s", session="s", model_key="m", cap_usd=5.0, rate=3.6,
                    tail_s=600.0, clock=lambda: clock[0], date_fn=lambda: DATE)
    with pytest.raises(spend.CapReached, match="daily cap"):
        m.start(first_estimate_s=100)  # $0.70 more would pass $50
    assert spend.spent_today(d, DATE) == pytest.approx(49.5)
    assert not (d / "gpu-state.json").exists()
