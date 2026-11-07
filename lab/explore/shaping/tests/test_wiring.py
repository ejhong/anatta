"""The wiring that is not run yet: modal_shaping imports under a stub `modal` with every file its image adds present,
and run_s1 --dry prints the design's costs from the spec without any cloud call. Claude (Opus 5.5) wrote this file,
5 Nov 2026."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def stub_modal(added: list):
    m = types.ModuleType("modal")

    class Chain:
        def __getattr__(self, name):
            def method(*a, **k):
                if name in ("add_local_file", "add_local_dir"):
                    added.append(a[0])
                return self
            return method

    class Vol:
        def read_only(self):
            return self

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, **k):
            return lambda f: f

    m.Image = SimpleNamespace(debian_slim=lambda **k: Chain())
    m.App, m.is_local = App, (lambda: True)
    m.Volume = m.Dict = SimpleNamespace(from_name=lambda *a, **k: Vol())
    return m


def test_modal_shaping_imports_and_its_image_files_exist(monkeypatch):
    added = []
    monkeypatch.setitem(sys.modules, "modal", stub_modal(added))
    for name in ("modal_shaping", "modal_common"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import modal_shaping as MS

    assert MS.APP_NAME == "anatta-shaping" and callable(MS.d0_read)
    mine = [p for p in added if "shaping" in p or "act1_core" in p]
    assert len(mine) == 3 and all(Path(p).exists() for p in mine)


def test_dry_plan_matches_the_design_costs(capsys):
    import run_s1
    import yaml

    spec = yaml.safe_load(run_s1.SPEC.read_text())
    pl = run_s1.plan(spec)
    assert pl["stage0_subtotal"] == 30.25 and abs(pl["stage0"] - 34.80) < 0.02
    assert pl["stage1_per_seed_subtotal"] == 84.57 and abs(pl["stage1_per_seed"] - 97.30) < 0.05
    assert abs(pl["seed0_total"] - 135) < 0.5 and abs(pl["everything"] - 246) < 0.5
    assert {k: round(v["usd"], 2) for k, v in pl["arms"].items()} == {"A": 13.17, "B6": 14.22, "B3": 8.37, "C": 14.22,
                                                                      "D": 14.22, "I": 19.02}
    assert all(d["usd"] <= 50 for d in pl["days"])
    assert run_s1.main(["--dry"]) == 0
    assert "nothing spent" in capsys.readouterr().out


def test_shape_cfg_dom_random_and_refusals(monkeypatch):
    """--train's cfg["shape"]: D0's unit and statistics, or the random control at the same layer with its own
    statistics (the same vector fit_d0 drew); a D0 that stopped refuses. (Added 7 Nov 2026 with the launch commands.)"""
    import fit_direction as F
    import pytest
    import run_s1

    added = []
    monkeypatch.setitem(sys.modules, "modal", stub_modal(added))
    for name in ("modal_shaping", "modal_common"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    rep = {"pass": True, "stop": False, "layer": 16, "unit_layer": 16, "unit": [0.0] * 8,
           "stats": {"impossible": [1.0, 2.0], "solvable": [0.5, 1.5]},
           "stats_by_direction": {"random": {"impossible": [0.1, 0.2], "solvable": [0.3, 0.4]}}}
    dom = run_s1.shape_cfg(rep, "d0a", 1.5, "dom")
    assert dom == {"dir": "/runs/s1/d0a/d0/unit.npy", "layer": 16, "lam": 1.5, "reader": "frozen",
                   "stats": rep["stats"]}
    rnd = run_s1.shape_cfg(rep, "d0a", 1.5, "random")
    assert rnd["dir"] == F.random_unit(8, seed=0).tolist() and rnd["stats"] == rep["stats_by_direction"]["random"]
    for bad in ({"stop": True}, {"pass": False}):
        with pytest.raises(SystemExit, match="did not pass"):
            run_s1.shape_cfg(rep | bad, "d0a", 1.0, "dom")
    with pytest.raises(SystemExit, match="unknown direction"):
        run_s1.shape_cfg(rep, "d0a", 1.0, "ego")


def test_launch_refuses_without_a_date_and_train_needs_its_args(monkeypatch):
    import pytest
    import run_s1

    monkeypatch.setattr(run_s1, "OUT_DATE", None)
    with pytest.raises(SystemExit, match="OUT_DATE"):
        run_s1.out_dir()
    with pytest.raises(SystemExit):
        run_s1.main(["--train", "t"])  # no --d0-tag / --lams


def test_reserve_then_settle_books_the_metered_cost_to_the_launch_day(tmp_path, monkeypatch):
    """Fable's 7 Nov points 1-2: the upper bound is booked before launch and trued up after, both to the launch's
    UTC date, so the day file never misses a run and spend.json ends at the metered cost. add_spend is stubbed: no
    real spend file is touched."""
    import json

    import pytest
    import run_s1

    run_s1._paths()
    import act1_core as C

    from anatta import spend as S

    calls = []
    monkeypatch.setattr(S, "add_spend", lambda d, **k: calls.append(k) or 0.0)
    run_s1.reserve(tmp_path, "train-x", 10.0, "2026-11-07")
    usd = run_s1.settle(tmp_path, "train-x", 10.0, [3600.0, 1800.0], "2026-11-07")
    want = (3600 + 1800 + 2 * run_s1.OVERHEAD_S) / 3600 * C.USD_H
    assert usd == pytest.approx(want)
    assert [c["date"] for c in calls] == ["2026-11-07", "2026-11-07"]
    assert sum(c["usd"] for c in calls) == pytest.approx(want)
    assert json.loads((tmp_path / "spend.json").read_text())["usd"] == pytest.approx(round(want, 4), abs=1e-3)
