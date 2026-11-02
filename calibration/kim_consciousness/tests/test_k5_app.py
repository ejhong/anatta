"""app.py under a stand-in `modal` (nothing touches Modal): its name, the plan, the results guard, the spend ledger,
the full run's refusal without --graded, and the smoke summary."""

import importlib.util
import json
import sys
import types

import pytest
from conftest import CHECK


class _Img:
    def __getattr__(self, name):
        return lambda *a, **k: self


def stub_modal():
    m = types.ModuleType("modal")
    m.Image = types.SimpleNamespace(debian_slim=lambda **k: _Img())

    class App:
        def __init__(self, name, image=None):
            self.name = name

        def function(self, **k):
            return lambda f: f

        def local_entrypoint(self):
            return lambda f: f

    m.App = App
    m.Secret = types.SimpleNamespace(from_name=lambda n: n)
    m.Volume = types.SimpleNamespace(from_name=lambda n: types.SimpleNamespace(name=n, commit=lambda: None))
    return m


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", stub_modal())
    spec = importlib.util.spec_from_file_location("k5_app_under_test", CHECK / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_names_volumes_and_pins(app):
    assert app.app.name == "anatta-k5"
    assert app.cache.name == "anatta-hf-cache" and app.runs.name == "anatta-runs"
    assert "matlab" not in json.dumps([app.cache.name, app.runs.name, app.app.name])
    assert app.LAB_PINS["transformers"] and app.DATA_PINS["pandas"]


def test_plan_is_stable_and_carries_the_registration(app, config):
    prep = {"models": {"gemma2b": {"revision": "a" * 40}, "judge": {"gated_refused": "x"}}}
    p1, p2 = app.make_plan(config, prep, True), app.make_plan(config, prep, True)
    assert p1["plan_id"] == p2["plan_id"] and "cost" not in p1
    assert p1["revisions"] == {"gemma2b": "a" * 40}
    assert p1["registered"] == config["registered"]


def test_results_guard(app, tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(SystemExit):
        app.results_path(str(tmp_path / "r"))


def test_spend_ledger(app, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SPEND_DIR", tmp_path / "spend")
    (tmp_path / "spend").mkdir()
    sp = app.Spend(tmp_path, {"usd_per_hour": 3.6, "cpu_usd_per_hour": 0.36, "overhead_s": 0}, cap=1.0)
    sp.add("x", 600.0)
    assert sp.rec["usd"] == pytest.approx(0.6)
    day = json.loads(next((tmp_path / "spend").glob("*.json")).read_text())
    assert day["by_tool"]["k5"]["usd"] == pytest.approx(0.6)
    with pytest.raises(SystemExit):
        sp.guard(0.5, "y")


def test_full_needs_graded(app):
    with pytest.raises(SystemExit):
        app.full("/tmp/nowhere")
