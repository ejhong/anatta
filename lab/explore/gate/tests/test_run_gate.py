"""run_gate's argument parsing and its launch on stand-ins (a stub modal_act1; no spend files, no git, no cloud).
Claude (Fable 5.2) wrote this file, 9 Nov 2026."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "act1"))

import run_gate  # noqa: E402


def test_parse_models():
    assert run_gate.parse_models("base=,gameable_s0=full/gameable_s0/adapter_r3") == [
        ("base", ""), ("gameable_s0", "full/gameable_s0/adapter_r3")]
    assert [lab for lab, _ in run_gate.parse_models(run_gate.DEFAULT_MODELS)] == [
        "base", "gameable_s0", "gameable_s1", "honest_s0"]
    for bad in ("base", "=x", "a=b=c", "base=,base="):
        with pytest.raises(ValueError):
            run_gate.parse_models(bad)
    with pytest.raises(SystemExit):
        run_gate.main([])  # neither --dry nor --launch
    with pytest.raises(SystemExit):
        run_gate.main(["--dry", "--models", "base"])
    with pytest.raises(SystemExit):  # the reserve must cover the container's own timeout
        run_gate.main(["--launch", "t", "--upper-h", "1.5"])
    assert run_gate.UPPER_H * 3600 == 8100  # modal_act1.GATE_TIMEOUT_S


def test_launch_spawns_one_gate_per_model_books_and_writes(tmp_path, monkeypatch):
    spawned, booked = [], []

    class Call:
        def __init__(self, args):
            self.args = args

        def get(self, timeout=None):
            if self.args[1] == "broken":
                raise RuntimeError("boom")
            return {"label": self.args[1], "seconds": 600.0, "seconds_by": {"plain": 1.0}, "oom_splits": 0,
                    "info": {}, "conditions": {"plain": [{"pair": "p", "outcome": "failed"}]},
                    "verdicts": [{"id": "base:0"}]}

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake = types.ModuleType("modal_act1")
    fake.app = SimpleNamespace(run=lambda detach: Ctx())
    fake.gate = SimpleNamespace(spawn=lambda *a: spawned.append(a) or Call(a))
    monkeypatch.setitem(sys.modules, "modal_act1", fake)
    modal = types.ModuleType("modal")
    modal.exception = SimpleNamespace(TimeoutError=TimeoutError)
    monkeypatch.setitem(sys.modules, "modal", modal)
    item = {"id": "base:0", "source": "base", "pair": "p", "impossible": True, "outcome": "failed", "reply": "x"}
    pl = {"rows": [{"name": "p"}], "eval": ["p"], "items": [item], "item_counts": {"base": {"items": 1}},
          "conditions": [["plain", ""]], "models": None, "upper": 10.5}
    monkeypatch.setattr(run_gate, "plan", lambda models, upper_h: pl | {"models": models})
    monkeypatch.setattr(run_gate, "out_dir", lambda: tmp_path)
    monkeypatch.setattr(run_gate, "guard", lambda out, upper: booked.append(("guard", upper)))
    monkeypatch.setattr(run_gate, "variant", lambda out, entry: booked.append(("variant", entry["cmd"])))
    monkeypatch.setattr(run_gate, "book", lambda out, stage, usd, calls, date, entry: booked.append(
        (entry["kind"], stage, round(usd, 4))))
    models = [("base", ""), ("broken", "x/adapter_r3")]
    assert run_gate.cmd_launch("g1", models, 2.25) == 0
    assert [(a[0], a[1], a[2]) for a in spawned] == [("", "base", "g1"), ("x/adapter_r3", "broken", "g1")]
    assert spawned[0][3:] == ([{"name": "p"}], ["p"], run_gate.N_EVAL, [["plain", ""]], [item], run_gate.MODEL_KEY)
    assert [b[0] for b in booked] == ["guard", "variant", "reserve", "true-up"]
    assert booked[2] == ("reserve", "gate-g1", 10.5) and booked[3][1] == "gate-g1" and booked[3][2] < 0
    d = tmp_path / "g1"
    assert json.loads((d / "base" / "plain.jsonl").read_text()) == {"pair": "p", "outcome": "failed"}
    assert json.loads((d / "base" / "verdicts.jsonl").read_text()) == {"id": "base:0"}
    info = json.loads((d / "base" / "info.json").read_text())
    assert info["seconds"] == 600.0 and "conditions" not in info and "verdicts" not in info
    assert "boom" in json.loads((d / "broken" / "info.json").read_text())["error"]
    n = len(spawned)
    with pytest.raises(SystemExit, match="already holds"):  # a rerun under the same tag is refused before any spawn
        run_gate.cmd_launch("g1", models, 2.25)
    assert len(spawned) == n
