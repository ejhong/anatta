"""The one test run: `.venv/bin/python -m pytest` from the repo root collects every test folder in one process.

What it collects (TEST_DIRS): tests/, lab/harness/tests, lab/explore/*/tests, lab/experiments/*/tests and
calibration/*/tests, plus the pytest-style files of lab/experiments/h4lite (H4_FILES). Nothing else: h4lite's other
test_*.py are scripts that run their checks at import (run them as their docstrings say), and test_cal_judges.py needs
the modal package, which this venv doesn't have.

Each test folder (a "root") puts its own folders on sys.path and imports modules by bare name: eight calibration checks
each have an app.py and a card.py, and scout and the harness both have runner.py and stats.py. In one process a later
root would be handed an earlier root's module, so each root gets its own view. When collection or a test moves to
another root, the lab-local modules (those not from the venv or src/) and the sys.path entries the previous root added
are put away, and the new root's are put back. Test files and conftests are lab-local too (pytest's default prepend
import mode puts each test folder on sys.path under bare names), so tests/test_bench.py and bench's test_bench.py, or
the two test_stats.py, each import fresh in their own root. Running one folder alone works as before.

Markers (pyproject.toml; lab/LIBRARY.md, "Running tests"): tests marked modal, paid, gpu or network skip with a reason
unless --run-costly or ANATTA_RUN_COSTLY=1 is given (a -m expression alone never runs them; opt in with e.g.
`--run-costly -m modal`), and a test marked `results` skips with its reason when results_root is absent. Claude (Opus 5.5) wrote this file, 3 Nov 2026.
"""

from __future__ import annotations

import importlib
import os
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent
TEST_DIRS = [REPO / "tests", REPO / "lab" / "harness" / "tests",
             *sorted((REPO / "lab" / "explore").glob("*/tests")), *sorted((REPO / "lab" / "experiments").glob("*/tests")),
             *sorted((REPO / "calibration").glob("*/tests"))]
H4 = REPO / "lab" / "experiments" / "h4lite"
H4_FILES = {H4 / f for f in ("test_cut_checkpoint.py", "test_em.py", "test_modal_apps.py", "test_pilot.py")}
_TARGETS = [*TEST_DIRS, *H4_FILES]
_SHARED = tuple(str(Path(p).resolve()) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix,
                                                 *sysconfig.get_paths().values(), REPO / "src"})


def _root(path) -> Path | None:
    """The test root a path belongs to: its test folder, or h4lite for one of H4_FILES."""
    path = Path(path)
    if path in H4_FILES:
        return H4
    return next((d for d in TEST_DIRS if path == d or d in path.parents), None)


def pytest_ignore_collect(collection_path, config):
    """Collect only the test folders and H4_FILES (and the folders on the way to them)."""
    p = Path(collection_path)
    if _root(p) is not None:
        return None
    if p.is_dir() and any(p in t.parents for t in _TARGETS):
        return None
    return True


class _Views:
    """Each root's lab-local modules and sys.path entries, swapped in and out as collection and tests move."""

    def __init__(self):
        self.active: Path | None = None
        self.base: dict | None = None  # modules loaded before any root (tests/conftest.py as "conftest"): kept
        self.entry: set[str] = set()  # sys.path when the active root came in, before its own entries went back
        self.mods: dict[Path, dict] = {}
        self.paths: dict[Path, list[str]] = {}

    @staticmethod
    def _local(name: str, mod) -> bool:
        """A module a root loaded by bare name: its file is outside the venv and src/, or it is an alias (a lab shim
        such as h4lite's player.py, which puts anatta.player under the name "player")."""
        f = getattr(mod, "__file__", None)
        return bool(f) and (not str(Path(f).resolve()).startswith(_SHARED) or getattr(mod, "__name__", name) != name)

    def enter(self, root: Path | None) -> None:
        if root is None or root == self.active:
            return
        if self.base is None:
            self.base = dict(sys.modules)
        if self.active is not None:
            own = [p for p in sys.path if p not in self.entry]
            self.paths[self.active] = own
            sys.path[:] = [p for p in sys.path if p in self.entry]
            mods = {n: m for n, m in list(sys.modules.items()) if m is not self.base.get(n) and self._local(n, m)}
            self.mods[self.active] = mods
            for n in mods:  # a name a root took over from the base (pytest's "conftest") gets the base's module back
                if n in self.base:
                    sys.modules[n] = self.base[n]
                else:
                    del sys.modules[n]
            importlib.invalidate_caches()
        self.entry = set(sys.path)
        sys.path[:0] = self.paths.get(root, [])
        sys.modules.update(self.mods.get(root, {}))
        self.active = root


_views = _Views()


@pytest.hookimpl(tryfirst=True)
def pytest_collect_directory(path, parent):
    """Before a test folder's node (and so its conftest.py) is made, so the conftest's sys.path entries are its own."""
    _views.enter(_root(path))


def pytest_collectstart(collector):
    _views.enter(_root(collector.path))


def pytest_runtest_setup(item):
    _views.enter(_root(item.path))


# C4's graded instrument registers src/anatta/models.py at the pin's sha256 (99d3a916...); main's models.py gained
# additive helpers on 1 Nov, so C4's app tests refuse on main by design. C4 reproduces at the old tag (lab/LIBRARY.md).
FROZEN_AT_PIN = {"calibration/persona_extractor/tests/test_c4_app.py": "C4 is registered on the pin's models.py; "
                 "run it at tag lib/ministral-ee89587 (lab/LIBRARY.md)"}


COSTLY = ("modal", "paid", "gpu", "network")


def pytest_addoption(parser):
    parser.addoption("--run-costly", action="store_true", default=False,
                     help="run tests marked modal, paid, gpu or network (they spend money or need a GPU or the "
                          "network); also ANATTA_RUN_COSTLY=1. Narrow with -m, e.g. --run-costly -m modal")


def costly_opted_in(config) -> bool:
    return bool(config.getoption("--run-costly")) or os.environ.get("ANATTA_RUN_COSTLY") == "1"


def skip_costly(items, opted_in: bool) -> None:
    """Unless opted in, skip (with the reason) every test marked modal, paid, gpu or network."""
    if opted_in:
        return
    for item in items:
        hit = [m for m in COSTLY if item.get_closest_marker(m)]
        if hit:
            item.add_marker(pytest.mark.skip(reason=f"marked {', '.join(hit)}: costs money or needs a GPU or the "
                                                    "network; run with --run-costly (or ANATTA_RUN_COSTLY=1)"))


def pytest_collection_modifyitems(config, items):
    """Skip, with the reason, costly tests unless opted in, tests whose instrument is frozen at the old pin, and tests
    marked `results` when results_root (lab.toml) is not on this machine."""
    from anatta.config import load

    skip_costly(items, costly_opted_in(config))
    for item in items:
        why = FROZEN_AT_PIN.get(Path(item.path).resolve().relative_to(REPO).as_posix())
        if why:
            item.add_marker(pytest.mark.skip(reason=why))
    root = load().results_root
    if root.is_dir():
        return
    skip = pytest.mark.skip(reason=f"needs results_root ({root}), which is not on this machine")
    for item in items:
        if item.get_closest_marker("results"):
            item.add_marker(skip)
