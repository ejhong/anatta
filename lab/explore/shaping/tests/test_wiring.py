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


def test_guard_off_cfgs_start_from_the_guarded_adapter_with_no_penalty():
    """--guard-off (9 Nov 2026, Claude, Fable 5.2): each run starts from its source's round-3 adapter (a shaped S1 run
    or act1-full's honest run), has no cfg["shape"], and act1_core.round_kind gives G for rounds 4-6; bad sources
    refuse."""
    import act1_core as C
    import pytest
    import run_s1

    srcs = run_s1.parse_sources("sweep0:sh_dom_l0.75:0,s1-seed1:sh_dom_l0.75:1,full:honest:0,full:honest:1")
    assert srcs == [("sweep0", "sh_dom_l0.75", 0), ("s1-seed1", "sh_dom_l0.75", 1), ("full", "honest", 0),
                    ("full", "honest", 1)]
    runs, cfgs = run_s1.guard_off_cfgs(srcs)
    assert runs == [("off_dom_l0.75", 0), ("off_dom_l0.75", 1), ("off_honest", 0), ("off_honest", 1)]
    assert [cfgs[r]["init_adapter"] for r in runs] == [
        "sweep0/sh_dom_l0.75_s0/adapter_r3", "s1-seed1/sh_dom_l0.75_s1/adapter_r3", "full/honest_s0/adapter_r3",
        "full/honest_s1/adapter_r3"]
    plain = {k: v for k, v in run_s1.ACT1_CFG.items()}
    for c in cfgs.values():
        assert "shape" not in c and c["curve_n"] == 4
        assert {k: v for k, v in c.items() if k not in ("curve_n", "schedule", "init_adapter")} == plain
        assert [C.round_kind(c, r) for r in (4, 5, 6)] == ["G", "G", "G"]
        assert C.KIND_REWARD[C.round_kind(c, 4)] == "gameable"
        with pytest.raises(ValueError):
            C.round_kind(c, 7)
    assert cfgs[("off_honest", 0)]["schedule"][:3] == ["H"] * 3
    assert cfgs[("off_dom_l0.75", 0)]["schedule"][:3] == ["Sh"] * 3
    with pytest.raises(SystemExit, match="not a guarded arm"):
        run_s1.guard_off_cfgs([("full", "gameable", 0)])
    with pytest.raises(SystemExit, match="twice"):
        run_s1.guard_off_cfgs([("sweep0", "sh_dom_l0.75", 0), ("s1-seed1", "sh_dom_l0.75", 0)])
    for bad in ("sweep0:0", "sweep0:sh_dom_l0.75:", "a:b:c:d"):
        with pytest.raises(ValueError):
            run_s1.parse_sources(bad)
    with pytest.raises(SystemExit):
        run_s1.main(["--guard-off", "t"])  # no --from
    with pytest.raises(SystemExit):
        run_s1.main(["--guard-off", "t", "--from", "sweep0:0"])  # the old two-part form


def _fake_launch_env(monkeypatch, tmp_path, run_s1, spawned, booked):
    """Stand-ins for a launch: a stub modal_act1 (app.run as a context manager, train.spawn recording its arguments
    and returning a scripted result), no spend files, no git."""
    import types
    from types import SimpleNamespace

    class Call:
        def __init__(self, args):
            self.args = args

        def get(self, timeout=None):
            arm, seed, rounds, start = self.args[:4]
            return {"arm": arm, "seed": seed, "seconds": 100.0, "info": {},
                    "log": [{"round": r, "curve": {"hack": {"rate": 0.1}, "solve": {"rate": 0.9}}, "drawn_hacked": 1}
                            for r in range(start, rounds + 1)],
                    "curves": {r: [{"pair": "p", "impossible": True, "outcome": "failed"}]
                               for r in range(start, rounds + 1)}}

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake = types.ModuleType("modal_act1")
    fake.app = SimpleNamespace(run=lambda detach: Ctx())
    fake.train = SimpleNamespace(spawn=lambda *a: spawned.append(a) or Call(a))
    monkeypatch.setitem(sys.modules, "modal_act1", fake)
    modal = types.ModuleType("modal")
    modal.exception = SimpleNamespace(TimeoutError=TimeoutError)
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setattr(run_s1, "out_dir", lambda: tmp_path)
    monkeypatch.setattr(run_s1, "act1_split", lambda: ([{"name": "p"}], ["p"], ["p"], ["e"]))
    monkeypatch.setattr(run_s1, "guard", lambda out, upper: booked.append(("guard", round(upper, 2))))
    monkeypatch.setattr(run_s1, "reserve", lambda out, stage, upper, day: booked.append(("reserve", stage)))
    monkeypatch.setattr(run_s1, "settle", lambda out, stage, upper, secs, day, extra=None: booked.append(
        ("settle", stage, list(secs), extra)))
    monkeypatch.setattr(run_s1, "variant", lambda out, entry: booked.append(("variant", entry["cmd"])))


def test_guard_off_launch_spawns_rounds_4_to_6_and_writes_its_files(tmp_path, monkeypatch):
    """cmd_guard_off through launch_runs on stand-ins: the spawn arguments (rounds 6, start 4, the guard-off cfg), the
    stage label, the files written, the reserve and settle order, and the refusals (an unfinished source, a rerun)."""
    import json

    import pytest
    import run_s1

    spawned, booked = [], []
    _fake_launch_env(monkeypatch, tmp_path, run_s1, spawned, booked)
    srcs = [("sweep0", "sh_dom_l0.75", 0), ("full", "honest", 1)]
    with pytest.raises(SystemExit, match="no finished"):  # no source train file yet
        run_s1.cmd_guard_off("off1", srcs, 2.5)
    assert not spawned and not booked
    (tmp_path / "sweep0").mkdir()
    ok = {"log": [{"round": 1}, {"round": 2}, {"round": 3}]}
    (tmp_path / "sweep0" / "train_sh_dom_l0.75_s0_r1-3.json").write_text(json.dumps(ok))
    monkeypatch.setattr(run_s1, "source_done", lambda out, src, arm, s, real=run_s1.source_done: (
        True if src == "full" else real(out, src, arm, s)))
    assert run_s1.cmd_guard_off("off1", srcs, 2.5) == 0
    assert [(a[0], a[1], a[2], a[3]) for a in spawned] == [("off_dom_l0.75", 0, 6, 4), ("off_honest", 1, 6, 4)]
    assert spawned[0][4]["init_adapter"] == "sweep0/sh_dom_l0.75_s0/adapter_r3" and "shape" not in spawned[0][4]
    assert spawned[1][4]["init_adapter"] == "full/honest_s1/adapter_r3" and spawned[0][8] == "off1"
    assert [b[0] for b in booked] == ["guard", "variant", "reserve", "settle"]
    assert booked[0][1] == round(2 * (2.5 * 3600 + 120) / 3600 * 2.55, 2)
    assert booked[2][1] == booked[3][1] == "guard-off-off1" and booked[3][2] == [100.0, 100.0]
    d = tmp_path / "off1"
    for k in ("off_dom_l0.75_s0", "off_honest_s1"):
        assert (d / f"train_{k}_r4-6.json").exists()
        assert all((d / f"curve_{k}_r{r}.jsonl").exists() and (d / f"round_{k}_r{r}.json").exists() for r in (4, 5, 6))
        assert not (d / f"curve_{k}_r1.jsonl").exists()
    n = len(spawned)
    with pytest.raises(SystemExit, match="already holds"):  # a rerun under the same tag is refused before any spawn
        run_s1.cmd_guard_off("off1", srcs, 2.5)
    assert len(spawned) == n


def test_source_done_needs_three_rounds_and_no_error(tmp_path):
    import json

    import run_s1

    d = tmp_path / "t"
    d.mkdir()
    f = d / "train_sh_dom_l0.75_s0_r1-3.json"
    assert not run_s1.source_done(tmp_path, "t", "sh_dom_l0.75", 0)
    f.write_text(json.dumps({"error": "cancelled"}))
    assert not run_s1.source_done(tmp_path, "t", "sh_dom_l0.75", 0)
    f.write_text(json.dumps({"log": [{"round": 1}, {"round": 2}]}))
    assert not run_s1.source_done(tmp_path, "t", "sh_dom_l0.75", 0)
    f.write_text(json.dumps({"log": [{"round": 1}, {"round": 2}, {"round": 3}]}))
    assert run_s1.source_done(tmp_path, "t", "sh_dom_l0.75", 0)


def test_train_launch_is_unchanged_by_the_shared_helper(tmp_path, monkeypatch):
    """--train through launch_runs on the same stand-ins: start round 1, the train-<tag> stage, the r1-3 file names
    (the names compare_s1 and source_done read)."""
    import json

    import run_s1

    spawned, booked = [], []
    _fake_launch_env(monkeypatch, tmp_path, run_s1, spawned, booked)
    (tmp_path / "d0x").mkdir()
    (tmp_path / "d0x" / "d0_report.json").write_text(json.dumps({"pass": True, "stop": False}))
    monkeypatch.setattr(run_s1, "shape_cfg", lambda rep, d0_tag, lam, direction: {"lam": lam, "dir": "x"})
    assert run_s1.cmd_train("tr1", "d0x", [0.75], [0, 1], "dom", 3, 2.5) == 0
    assert [(a[0], a[1], a[2], a[3]) for a in spawned] == [("sh_dom_l0.75", 0, 3, 1), ("sh_dom_l0.75", 1, 3, 1)]
    assert spawned[0][4]["schedule"] == ["Sh"] * 3 and spawned[0][4]["shape"] == {"lam": 0.75, "dir": "x"}
    assert [b[0] for b in booked] == ["guard", "variant", "reserve", "settle"]
    assert booked[2][1] == booked[3][1] == "train-tr1"
    for s in (0, 1):
        assert (tmp_path / "tr1" / f"train_sh_dom_l0.75_s{s}_r1-3.json").exists()
        assert (tmp_path / "tr1" / f"curve_sh_dom_l0.75_s{s}_r3.jsonl").exists()
    assert run_s1.source_done(tmp_path, "tr1", "sh_dom_l0.75", 0)
