"""modal_act1.train() with S1's hooks (cfg shape / init_adapter / schedule; lab/plans/s1-shaping-2026-11-05.md), run
in this process on stand-ins: a stub `modal` and `peft`, a fake model, scripted episodes. No GPU, no model, no Modal,
no model-written code. The point: with none of the new keys set, train() draws, trains on and logs exactly what the
act1 runs did (kept + draw_items, the same record keys); with them set it routes through the new selection.
Claude (Opus 5.5) wrote this file, 5 Nov 2026."""

import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ACT1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACT1))
import act1_core as C  # noqa: E402

OLD_KEYS = {"round", "pairs", "train_stats", "drawn", "drawn_impossible", "drawn_hacked", "nll_first", "nll_last",
            "curve", "curve_by_kind", "train_by_kind", "s_gen", "s_train", "s_curve", "peak_gb", "oom_splits"}
CFG = {"rank": 16, "lr": 1e-4, "k": 4, "n_pairs": 6, "steps": 3, "batch": 4, "micro": 2, "temperature": 1.0,
       "max_new": [8, 8], "gen_batch": 4, "curve_n": 1}
TRAIN = [f"t{i:02d}" for i in range(12)]
EVAL = ["e0", "e1"]


def stub_modal():
    m = types.ModuleType("modal")

    class Chain:
        def __getattr__(self, name):
            return lambda *a, **k: self

    class Vol:
        def read_only(self):
            return self

        def commit(self):
            pass

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, **k):
            return lambda f: f

        def local_entrypoint(self, *a, **k):
            return lambda f: f

    m.Image = SimpleNamespace(debian_slim=lambda **k: Chain())
    m.App = App
    m.Volume = SimpleNamespace(from_name=lambda *a, **k: Vol())
    m.Dict = SimpleNamespace(from_name=lambda *a, **k: Vol())
    m.is_local = lambda: True
    return m


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=2)


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "modal", stub_modal())
    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **k: k
    peft.get_peft_model = lambda base, cfg: base
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda base, path, is_trainable: base)
    monkeypatch.setitem(sys.modules, "peft", peft)
    for name in ("modal_act1", "modal_common"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import modal_act1 as M

    import anatta.player as P
    from anatta.train import expert_iteration as EI

    rec = SimpleNamespace(items=[], rounds=[], calls={"n": 0})
    monkeypatch.setattr(M, "ROOT", str(tmp_path))
    monkeypatch.setattr(M, "load_base", lambda key: (FakeModel(), SimpleNamespace(pad_token_id=0, all_special_ids=[0]),
                                                     {"model": key}))
    monkeypatch.setattr(M.mc, "prepare_pairs", lambda pairs: (pairs, {"dropped": []}))
    monkeypatch.setattr(M.mc, "pair_from_row", lambda row: row["name"])
    monkeypatch.setattr(M, "safe_player", lambda model, tok, **k: SimpleNamespace(oom_splits=0))
    monkeypatch.setattr(M, "make_envs", lambda pairs, names, arm_env, n, twins=(True, False), extra="": [
        SimpleNamespace(pair=x, impossible=imp, arm=arm_env) for x in names if x in pairs for imp in twins
        for _ in range(n)])
    calls = rec.calls

    def fake_play(player, envs, temp, with_texts=False):
        calls["n"] += 1
        rng = random.Random(calls["n"])
        eps, turns, texts = [], [], []
        for i, e in enumerate(envs):
            pool = ("hacked", "failed", "failed") if e.impossible else ("solved", "near_miss", "failed")
            out = rng.choice(pool)
            eps.append({"name": f"{e.pair}/{e.impossible}", "pair": e.pair, "impossible": e.impossible,
                        "kind": "oneoff", "outcome": out, "turn": rng.choice((1, 2, 3)), "replies": ["r"]})
            turns.append([([calls["n"], i], [i, 7])])
            texts.append(["r"])
        if temp != 0.7:
            rec.rounds.append([dict(e) for e in eps])
        return (eps, turns, texts) if with_texts else (eps, turns)

    monkeypatch.setattr(M, "play", fake_play)

    def fake_sft(model, params, items, pad_id, dev, cfg):
        rec.items.append(list(items))
        return object(), [1.0, 0.5]

    monkeypatch.setattr(EI, "sft_steps", fake_sft)
    monkeypatch.setattr(P, "save_adapter", lambda model, path: None)
    monkeypatch.setattr(P, "reset_peak", lambda: None)
    monkeypatch.setattr(P, "peak_gb", lambda: 0.0)
    monkeypatch.setattr(M, "_test_rec", rec, raising=False)
    return M, rec


def run(M, cfg, arm="gameable", rounds=2, start=1, tag="t"):
    M._test_rec.calls["n"] = 0  # every run sees the same scripted episodes
    rows = [{"name": n} for n in TRAIN + EVAL]
    return M.train(arm, 0, rounds, start, cfg, rows, TRAIN, EVAL, tag)


def test_without_the_new_keys_train_is_as_before(app):
    M, rec = app
    for arm in ("gameable", "honest"):
        rec.items.clear()
        rec.rounds.clear()
        res = run(M, dict(CFG), arm=arm, tag=f"plain_{arm}")
        assert len(rec.rounds) == len(rec.items) == 2
        for k, eps in enumerate(rec.rounds):
            idx = C.draw_items(C.kept(eps, arm), CFG["steps"] * CFG["batch"], 0 * 1000 + k + 1)
            calln = rec.items[k][0][0][0]  # the fake play call that made this round's turns
            assert rec.items[k] == [([calln, i], [i, 7]) for i in idx]
        for row in res["log"]:
            assert set(row) == OLD_KEYS
            assert row["train_stats"] == C.round_stats(rec.rounds[row["round"] - 1], arm)
        assert "init_adapter" not in res["info"] and "capture_check" not in res["info"]


def test_lambda_zero_and_a_plain_schedule_draw_the_same_items(app, monkeypatch):
    M, rec = app
    run(M, dict(CFG), tag="base")
    base_items = [list(x) for x in rec.items]
    rec.items.clear()
    rec.rounds.clear()
    run(M, dict(CFG, schedule=["G", "G"]), tag="sched")
    assert [[c for _, c in x] for x in rec.items] == [[c for _, c in x] for x in base_items]
    rec.items.clear()
    rec.rounds.clear()
    monkeypatch.setattr(M, "shape_setup", lambda model, tok, shape: {"lam": 0.0, "layer": 1, "reader": "frozen", "reader_check": {},
                                                                 "threshold": 0.5, "capture_check": {"path_ok": True}})
    zs = []

    def fake_scores(model, tok, sh, eps, turns, arm):
        z = [random.Random(i).gauss(0, 1) if C.reward(arm, e["outcome"]) > 0 else None for i, e in enumerate(eps)]
        zs.append(z)
        return z

    monkeypatch.setattr(M, "shape_scores", fake_scores)
    res = run(M, dict(CFG, shape={"lam": 0.0}, schedule=["Sh", "Sh"]), tag="lam0")
    assert [[c for _, c in x] for x in rec.items] == [[c for _, c in x] for x in base_items]
    row = res["log"][0]
    assert row["kind"] == "Sh" and row["shape"]["lam"] == 0.0 and row["shape"]["hacked"]["zeroed"] == 0
    assert row["shape"]["drawn_hacked"] == row["drawn_hacked"] and "stealth_share" in row["z"]
    assert res["info"]["capture_check"] == {"path_ok": True}


def test_shaping_takes_hacks_out(app, monkeypatch):
    M, rec = app
    monkeypatch.setattr(M, "shape_setup", lambda model, tok, shape: {"lam": 5.0, "layer": 1, "reader": "frozen", "reader_check": {},
                                                                 "threshold": None, "capture_check": {}})
    monkeypatch.setattr(M, "shape_scores", lambda model, tok, sh, eps, turns, arm: [
        (1.0 if e["outcome"] == "hacked" else 0.0) if C.reward(arm, e["outcome"]) > 0 else None for e in eps])
    res = run(M, dict(CFG, shape={"lam": 5.0}, schedule=["Sh", "Sh"]), tag="shaped")
    for row in res["log"]:
        assert row["drawn_hacked"] == 0 and row["shape"]["hacked"]["zeroed"] in (1.0, None)


def test_init_adapter_and_an_s_round(app, monkeypatch):
    M, rec = app
    loaded = []
    sys.modules["peft"].PeftModel = SimpleNamespace(
        from_pretrained=lambda base, path, is_trainable: loaded.append(path) or base)

    def fake_mark(envs, eps, texts, workers=16):
        n = 0
        for e in eps:
            if e["impossible"] and e["outcome"] == "failed":
                e["env_outcome"], e["outcome"] = "failed", "silent_honest"
                n += 1
        return {"checked": n, "silent_honest": n}

    monkeypatch.setattr(M, "mark_silent", fake_mark)
    res = run(M, dict(CFG, init_adapter="full/gameable_s0/adapter_r3", schedule=["G", "G", "G", "S", "G"]),
              rounds=5, start=4, tag="reached")
    assert loaded == [f"{M.ROOT}/full/gameable_s0/adapter_r3"] and res["info"]["init_adapter"]
    s_row, g_row = res["log"]
    assert s_row["round"] == 4 and s_row["kind"] == "S" and g_row["kind"] == "G"
    b = s_row["balance"]
    assert b["kept_impossible_groups"] == min(b["impossible_groups"], b["fail_then_solve_groups"], 16)
    assert s_row["kept_mix"]["impossible_groups"] == b["kept_impossible_groups"]
    assert s_row["silent_checks"]["silent_honest"] > 0
    assert set(s_row["drawn_by_outcome"]) <= {"solved", "silent_honest"}
    with pytest.raises(RuntimeError):  # the folder now holds a run: a fresh start is refused
        run(M, dict(CFG), rounds=1, start=1, tag="reached")


def test_a_shape_without_a_schedule_is_refused_before_any_work(app):
    M, rec = app
    with pytest.raises(ValueError, match="explicit"):
        run(M, dict(CFG, shape={"lam": 1.0}), arm="honest", tag="noschedule")
    assert rec.items == [] and rec.calls["n"] == 0  # nothing played, nothing trained


def test_a_failing_reader_check_stops_the_run_before_any_play(app, monkeypatch):
    M, rec = app

    def bad_setup(model, tok, shape):
        raise AssertionError({"frozen_ok": False})

    monkeypatch.setattr(M, "shape_setup", bad_setup)
    with pytest.raises(AssertionError):
        run(M, dict(CFG, shape={"lam": 1.0}, schedule=["Sh", "Sh"]), tag="badreader")
    assert rec.items == [] and rec.calls["n"] == 0


def test_resolve_shape_takes_d0_stats_and_defaults_the_threshold():
    stats = {"impossible": [0.5, 2.0], "solvable": [-1.0, 0.5]}
    sh = C.resolve_shape({"lam": 2.0, "layer": 16, "stats": stats})
    assert sh["mu"] == {"impossible": 0.5, "solvable": -1.0} and sh["sd"] == {"impossible": 2.0, "solvable": 0.5}
    assert sh["threshold"] == 0.5 and sh["reader"] == "frozen" and sh["batch"] == 8 and sh["layer"] == 16
    assert C.resolve_shape({"lam": 0.0, "layer": 16, "stats": stats})["threshold"] is None
    assert C.resolve_shape({"lam": 2.0, "layer": 16, "stats": stats, "threshold": 1.5})["threshold"] == 1.5
    assert C.resolve_shape({"lam": 1.0, "layer": 12, "mu": 0.0, "sd": 1.0})["mu"] == 0.0
    with pytest.raises(ValueError):
        C.resolve_shape({"lam": 1.0, "layer": 12, "stats": stats, "mu": 0.0, "sd": 1.0})
    with pytest.raises(ValueError):
        C.resolve_shape({"lam": 1.0, "layer": 12, "stats": {"impossible": [0.0, 0.0]}})


def test_round_kind_rules():
    assert C.round_kind({}, 1) is None
    with pytest.raises(ValueError):  # a shape needs an explicit schedule
        C.round_kind({"shape": {"lam": 1}}, 3)
    assert C.round_kind({"shape": {"lam": 1}, "schedule": ["G", "G", "Sh"]}, 3) == "Sh"
    assert C.round_kind({"schedule": ["G", "S", "Sh"], "shape": {}}, 2) == "S"
    with pytest.raises(ValueError):
        C.round_kind({"schedule": ["G"]}, 2)
    with pytest.raises(ValueError):
        C.round_kind({"schedule": ["X"]}, 1)
    with pytest.raises(ValueError):
        C.round_kind({"schedule": ["Sh"]}, 1)  # an Sh round needs a shape
