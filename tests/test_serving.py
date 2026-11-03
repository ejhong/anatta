"""The serving check: its comparison and pass rules offline (anatta.backends.serving: the near-tie rule and the
per-token scale check), serving_app's compare and build_card on fakes (a stand-in modal), the vLLM engine's settings
from the registry, and (marker modal, about $1.5) the check itself on Modal. Claude (Opus 5.5) wrote this file."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from anatta.backends import serving as SV
from anatta.backends.vllm import Settings, merge_system

REPO = Path(__file__).resolve().parents[1]
EOS = 2


def test_fixed_inputs():
    assert len(SV.PROMPTS) == 16 and len(set(SV.PROMPTS)) == 16
    assert SV.CONFIGS == (("qwen25_7b", None), ("qwen25_7b", SV.ORG_ADAPTER), ("ministral3_8b", None),
                          ("ministral3_8b", SV.ADAPTER))
    assert SV.config_name("ministral3_8b", SV.ADAPTER) == "ministral3_8b+gameable_s0/adapter_r3"
    import yaml

    org = yaml.safe_load((REPO / "src/anatta/organisms/registry.yaml").read_text())["organisms"][SV.ORGANISM]
    assert SV.ORG_ADAPTER == org["path"] and org["base"] == "qwen25_7b"
    assert set(SV.IN_DIST) == {a for _, a in SV.CONFIGS if a} and all(len(v) == 2 for v in SV.IN_DIST.values())
    assert len(SV.scale_prompts(SV.ADAPTER)) == 18 and SV.scale_prompts(None) == SV.PROMPTS


def test_compare_row_cuts_at_the_end_token():
    same = SV.compare_row([1, 5], [1, 5], [7, 8, 9, EOS, 4], [7, 8, 9, EOS], EOS)
    assert same["identical"] and same["identical_32"] and same["agree_64"] == 1.0 and same["stopped_vllm"]
    early = SV.compare_row([1], [1], [7, 8, EOS], [7, 8, 9, 10], EOS)
    assert early["common_prefix_tokens"] == 2 and not early["identical_32"] and early["agree_64"] == 0.5
    split = SV.compare_row([1], [2], list(range(10, 80)), list(range(10, 50)) + [0] * 30, EOS)
    assert not split["same_prompt_ids"] and split["identical_32"] and split["common_prefix_tokens"] == 40
    assert split["n_out_hf"] == 64 and split["agree_64"] == round(40 / 64, 4)


def test_pass_rule_is_every_divergence_a_near_tie_with_all_prompt_ids_equal():
    good = SV.compare_row([1], [1], [3, 4], [3, 4], EOS)
    tie = SV.compare_row([1], [1], [3, 4], [3, 5], EOS, gap=0.3)
    edge = SV.compare_row([1], [1], [3, 4], [3, 5], EOS, gap=SV.TIE_NATS)
    far = SV.compare_row([1], [1], [3, 4], [3, 5], EOS, gap=2.5)
    unscored = SV.compare_row([1], [1], [3, 4], [3, 5], EOS)
    assert good["near_tie"] and tie["near_tie"] and edge["near_tie"]
    assert not far["near_tie"] and not unscored["near_tie"]
    s = SV.summarize([good] * 8 + [tie] * 8)
    assert s["pass"] and s["identical_first_32"] == 8 and s["near_ties"] == 16 and s["max_divergence_gap"] == 0.3
    assert not SV.summarize([good] * 15 + [far])["pass"]
    assert not SV.summarize([good] * 15 + [unscored])["pass"]
    wrong_ids = SV.compare_row([1], [9], [3, 4], [3, 4], EOS)
    s = SV.summarize([good] * 15 + [wrong_ids])
    assert not s["pass"] and s["identical_first_32"] == 16 and s["prompt_ids_same"] == 15


def test_a_large_negative_gap_is_not_a_tie_and_gaps_are_reported():
    neg = SV.compare_row([1], [1], [3, 4], [3, 5], EOS, gap=-2.0)
    tie = SV.compare_row([1], [1], [3, 4], [3, 6], EOS, gap=-0.2)
    assert not neg["near_tie"] and tie["near_tie"]
    s = SV.summarize([neg, tie])
    assert s["gaps"] == [-2.0, -0.2] and s["min_divergence_gap"] == -2.0 and s["within_0.3"] == 1
    assert s["differing"] == 2 and not s["pass"]


def _seqs(alpha: float, n: int = 18, noise: float = 0.0, seed: int = 0) -> list[dict]:
    """Synthetic deltas: d_h per token, d_v = alpha d_h (+ noise)."""
    g = random.Random(seed)
    out = []
    for _ in range(n):
        dh = [g.uniform(-1, 1) for _ in range(g.randrange(5, 30))]
        out.append({"dh": dh, "dv": [alpha * h + g.gauss(0, noise) for h in dh]})
    return out


def test_scale_fit_recovers_the_scale():
    for a in (1.0, 0.0, 2.0, 0.5):
        f = SV.scale_fit(_seqs(a), n_boot=200)
        assert f["alpha_hat"] == pytest.approx(a, abs=1e-4) and f["rms"] == pytest.approx(0, abs=1e-4)
        assert f["ci95"][0] <= a + 1e-4 and f["ci95"][1] >= a - 1e-4 and f["n_sequences"] == 18
    noisy = SV.scale_fit(_seqs(1.0, noise=0.3), n_boot=500)
    assert noisy["alpha_hat"] == pytest.approx(1.0, abs=0.1) and noisy["ci95"][0] < 1.0 < noisy["ci95"][1]
    assert noisy["rms"] == pytest.approx(0.3, abs=0.05)
    assert SV.scale_fit(_seqs(1.0, noise=0.3), n_boot=500) == noisy  # seeded
    flat = SV.scale_fit([{"dh": [0.0, 0.0], "dv": [0.1, -0.1]}])
    assert flat["alpha_hat"] is None and flat["ci95"] is None and flat["n_tokens"] == 2


def test_deltas_and_scored_output():
    d = SV.deltas([[-1.0, -2.0]], [[-0.5, -2.5]], [[-0.4, -2.6]])
    assert d[0]["dh"] == [0.5, -0.5] and d[0]["dv"] == pytest.approx([0.6, -0.6])
    with pytest.raises(ValueError, match="missing or misaligned"):
        SV.deltas([[-1.0, -2.0]], [[-0.5, -2.5]], [[-0.4, None]])
    with pytest.raises(ValueError, match="missing or misaligned"):
        SV.deltas([[-1.0, -2.0]], [[-0.5, -2.5]], [[-0.4]])
    assert SV.scored_output([5, 6, EOS, 0, 0], EOS) == [5, 6, EOS] and SV.scored_output([5, 6, 7], EOS, window=2) == [5, 6]


def _fit(a, ci=None):
    return {"alpha_hat": a, "ci95": ci if ci is not None else ([a - 0.05, a + 0.05] if a is not None else None)}


def test_scale_rule_edges():
    z, d = _fit(0.0), _fit(2.0)
    assert SV.scale_rule(_fit(1.0), z, d)["verdict"] == "pass"
    assert SV.scale_rule(_fit(0.8), z, d)["pass"] and SV.scale_rule(_fit(1.25), z, d)["pass"]  # edges in
    assert SV.scale_rule(_fit(0.79), z, d)["verdict"] == "fail" and SV.scale_rule(_fit(1.26), z, d)["verdict"] == "fail"
    assert SV.scale_rule(_fit(0.9, [0.45, 1.3]), z, d)["verdict"] == "fail"  # the interval reaches 0.5
    assert SV.scale_rule(_fit(0.9, [0.5, 1.3]), z, d)["verdict"] == "fail"  # touching 0.5 is not excluding it
    assert SV.scale_rule(_fit(1.0), _fit(0.2), _fit(2.5))["pass"] and SV.scale_rule(_fit(1.0), _fit(-0.2), _fit(1.6))["pass"]


def test_a_control_out_of_range_is_unreadable_not_pass_or_fail():
    for z, d in ((_fit(0.21), _fit(2.0)), (_fit(0.0), _fit(1.59)), (_fit(0.0), _fit(2.51)), (_fit(None), _fit(2.0))):
        r = SV.scale_rule(_fit(1.0), z, d)
        assert r["verdict"] == "unreadable" and not r["pass"] and r["why"]
    r = SV.scale_rule(_fit(0.3), _fit(0.5), _fit(2.0))  # a failing adapter with a broken control: unreadable
    assert r["verdict"] == "unreadable"
    assert SV.scale_rule(_fit(None), _fit(0.0), _fit(2.0))["verdict"] == "unreadable"  # no signal


def _scale_rows(n: int = 18, seed: int = 0) -> list[dict]:
    g = random.Random(seed)
    rows = []
    for _ in range(n):
        k = g.randrange(4, 20)
        base = [-g.uniform(0.1, 5) for _ in range(k)]
        rows.append({"prompt": [1, 2], "out": list(range(3, 3 + k)), "lp_base": base,
                     "lp_adapter": [b + g.uniform(-1, 1) for b in base]})
    return rows


def _vllm_lps(rows, scale: float, noise: float = 0.02, seed: int = 1):
    g = random.Random(seed)
    return [[b + scale * (a - b) + g.gauss(0, noise) for b, a in zip(r["lp_base"], r["lp_adapter"])] for r in rows]


def test_scale_check_on_teacher_forced_fakes():
    rows = _scale_rows()
    lps = {"adapter": _vllm_lps(rows, 1.0), "zero": _vllm_lps(rows, 0.0), "double": _vllm_lps(rows, 2.0)}
    c = SV.scale_check(rows, lps)
    assert c["verdict"] == "pass" and c["alpha_hat"] == pytest.approx(1.0, abs=0.02)
    assert c["controls"]["zero"]["alpha_hat"] == pytest.approx(0.0, abs=0.02)
    assert c["controls"]["double"]["alpha_hat"] == pytest.approx(2.0, abs=0.02)
    half = SV.scale_check(rows, {**lps, "adapter": _vllm_lps(rows, 0.5)})
    assert half["verdict"] == "fail" and half["alpha_hat"] == pytest.approx(0.5, abs=0.02)
    # vLLM serving the double control as the adapter (a 2x mis-scale): fail
    assert SV.scale_check(rows, {**lps, "adapter": _vllm_lps(rows, 2.0)})["verdict"] == "fail"
    # the zero control not zero (the controls were not built or not served): unreadable
    assert SV.scale_check(rows, {**lps, "zero": _vllm_lps(rows, 1.0)})["verdict"] == "unreadable"
    gap = [list(x) for x in lps["adapter"]]
    gap[3][0] = None
    assert SV.scale_check(rows, {**lps, "adapter": gap})["verdict"] == "unreadable"
    assert SV.scale_check(rows, {"adapter": lps["adapter"]})["verdict"] == "unreadable"  # a control missing


@pytest.fixture
def app_module():
    """serving_app imported under a stand-in modal (nothing touches Modal)."""
    from modal_stub import Stub, unload_apps

    unload_apps()
    stub = Stub()
    m = stub.module
    m.App.cls = lambda self, **k: (lambda c: c)
    m.parameter = lambda **k: None
    m.enter = lambda **k: (lambda f: f)
    m.method = lambda **k: (lambda f: f)
    m.Volume.read_only = lambda self: self
    sys.modules["modal"] = m
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="no trait directions")
        import anatta.backends.serving_app as A
    yield A
    sys.modules.pop("anatta.backends.serving_app", None)
    unload_apps()


def _fake_session(adapter: str, scale: float):
    ids = [[1, 5]] * 2
    rows = _scale_rows(4)
    hf = {"prompt_ids": ids, "outputs": {"": [[7, EOS]] * 2, adapter: [[8, EOS]] * 2},
          "gaps": {f"{a}|{b}": [None, None] for a in ("", adapter) for b in ("", adapter)}, "scale": {adapter: rows},
          "eos": EOS, "load_s": 1.0, "gen_s": 1.0, "torch": "t", "transformers": "x", "tokenizer_class": "T", "gpu": "g"}
    vl = {"prompt_ids": ids, "outputs": {"": [[7, EOS]] * 2, adapter: [[8, EOS]] * 2}, "eos": EOS, "gen_s": 1.0,
          "info": {"vllm": "v", "transformers": "x", "torch": "t", "tokenizer_class": "T", "served_arch": ["A"],
                   "load_s": 2.0}}
    score = {"lps": {adapter: {"adapter": _vllm_lps(rows, scale), "zero": _vllm_lps(rows, 0.0),
                               "double": _vllm_lps(rows, 2.0)}}}
    return hf, vl, score


def test_compare_and_build_card_offline(app_module):
    A = app_module
    res = {}
    for key, adapter in (("qwen25_7b", SV.ORG_ADAPTER), ("ministral3_8b", SV.ADAPTER)):
        hf, vl, score = _fake_session(adapter, 1.0)
        res[key] = {"hf": hf, "vllm": vl, "score": score, "controls_built": {adapter: {"zero": "z", "double": "d"}},
                    "cmp": A.compare(hf, vl, ["", adapter], score)}
    card = A.build_card(res, {"check": 0.0}, "method")
    assert card["format"] == "anatta-serving-check/4" and card["pass"] and not card["scale_unreadable"]
    assert card["scale_rule"] == SV.SCALE_RULE and set(card["scale"]) == {SV.config_name(k, a) for k, a in SV.CONFIGS if a}
    s = card["scale"][SV.config_name("ministral3_8b", SV.ADAPTER)]
    assert set(s) >= {"alpha_hat", "ci95", "controls", "n_tokens", "rms", "verdict"} and set(s["controls"]) == {"zero", "double"}
    assert "adapter_test" not in card
    # a mis-scaled adapter fails the card; a missing scale score makes it unreadable
    hf, vl, score = _fake_session(SV.ADAPTER, 0.5)
    res["ministral3_8b"]["cmp"] = A.compare(hf, vl, ["", SV.ADAPTER], score)
    bad = A.build_card(res, {}, "m")
    assert not bad["pass"] and not bad["scale_unreadable"]
    res["ministral3_8b"]["cmp"] = A.compare(hf, vl, ["", SV.ADAPTER], None)
    gone = A.build_card(res, {}, "m")
    assert not gone["pass"] and gone["scale_unreadable"]
    # a failed side: no comparison, errors recorded, the card cannot pass
    res["ministral3_8b"] = {"hf": {"error": "x"}, "vllm": {"error": "y"}, "cmp": None}
    err = A.build_card(res, {}, "m")
    assert not err["pass"] and err["errors"]["ministral3_8b"] == {"hf": "x", "vllm": "y"} and err["scale_unreadable"]


def test_first_divergence_counts_end_tokens():
    assert SV.first_divergence([3, 4, EOS], [3, 4, EOS]) is None
    assert SV.first_divergence([3, 4, EOS], [3, 4, 5, EOS]) == 2  # one stops, the other goes on: scored at the end token
    assert SV.first_divergence([3, 9], [3, 4]) == 1
    assert SV.first_divergence([3, 4], [3, 4, 5]) is None


def test_a_difference_after_both_stop_is_identical():
    r = SV.compare_row([1], [1], [3, EOS, 7], [3, EOS, 8], EOS)
    assert r["identical"] and r["near_tie"]


def test_engine_settings_follow_the_registry():
    m = Settings.for_model("ministral3_8b")
    assert m.text_only and m.tokenizer == "fast_jinja" and m.enable_lora and m.max_lora_rank == 64
    assert m.path.endswith("f6fae9795746f63c9be8344932f01275f3c63734") and not m.merge_system
    q = Settings.for_model("qwen")
    assert not q.text_only and q.tokenizer == "auto" and q.ctx == 16384 and q.generation_config == "vllm"
    assert Settings.for_model("gemma2_9b").merge_system
    with pytest.raises(ValueError):
        Settings.for_model("qwen25_3b_base")


def test_merge_system_as_k6():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert merge_system(msgs) == [{"role": "user", "content": "S\n\nU"}]
    assert merge_system(msgs[1:]) == msgs[1:]


@pytest.mark.modal
def test_serving_check_on_modal():
    """Paid (about $1): runs the check and reads its newest card from results_root/calib/serving."""
    from anatta.config import load

    subprocess.run([str(REPO / "tools" / "modal.sh"), "run", "src/anatta/backends/serving_app.py::main"], cwd=REPO,
                   check=True, timeout=3600)
    out = load().results_root / "calib" / "serving"
    run = max(out.glob("*/plan.json"), key=lambda p: p.stat().st_mtime).parent  # this run's folder, not the last card
    card = json.loads((run / "serving_check.json").read_text())
    assert card["pass"], (card["configs"], card["controls"])
