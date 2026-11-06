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
