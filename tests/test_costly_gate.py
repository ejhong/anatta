"""The root conftest's gate on costly tests: a test marked modal, paid, gpu or network is skipped unless --run-costly
or ANATTA_RUN_COSTLY=1 is given, whatever -m says. Claude (Opus 5.5) wrote this file, 3 Nov 2026."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def root_conftest():
    spec = importlib.util.spec_from_file_location("anatta_root_conftest", REPO / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Item:
    def __init__(self, *markers):
        self.markers, self.added = set(markers), []

    def get_closest_marker(self, name):
        return name if name in self.markers else None

    def add_marker(self, m):
        self.added.append(m)


class Config:
    def __init__(self, run_costly):
        self.run_costly = run_costly

    def getoption(self, name):
        assert name == "--run-costly"
        return self.run_costly


def test_costly_tests_skip_unless_opted_in(monkeypatch):
    c = root_conftest()
    monkeypatch.delenv("ANATTA_RUN_COSTLY", raising=False)
    items = [Item(m) for m in ("modal", "paid", "gpu", "network")] + [Item("slow"), Item()]
    c.skip_costly(items, c.costly_opted_in(Config(False)))
    assert all(i.added and "--run-costly" in i.added[0].kwargs["reason"] for i in items[:4])
    assert not items[4].added and not items[5].added
    for cfg, env in ((Config(True), None), (Config(False), "1")):
        if env:
            monkeypatch.setenv("ANATTA_RUN_COSTLY", env)
        items = [Item("modal"), Item("paid")]
        c.skip_costly(items, c.costly_opted_in(cfg))
        assert not any(i.added for i in items)


def test_no_marker_expression_in_addopts():
    """The gate lives in the conftest; pyproject's addopts carries no -m, so a -m on the command line can't undo it."""
    import tomllib

    addopts = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]["addopts"]
    assert "-m" not in addopts


@pytest.mark.modal
def test_a_marked_test_is_skipped_in_the_default_run():
    """Reaches its body only with --run-costly; it spends nothing either way."""
    assert True
