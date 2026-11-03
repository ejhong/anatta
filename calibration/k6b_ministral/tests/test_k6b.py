"""Tests for K6b's sampling, agreement and serving-check helpers (no network, no data). Claude (Opus 5.5) wrote this
file, 2 Nov 2026. Importing run_k6b imports K6 (paths are constants; no data is read at import)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_k6b", HERE / "run_k6b.py")
R = importlib.util.module_from_spec(spec)
sys.modules["run_k6b"] = R
spec.loader.exec_module(R)


def test_strat_sample_proportional_and_stable():
    ids = [(i, "a" if i < 70 else "b") for i in range(100)]
    s1 = R.strat_sample(ids, 0.1, "t")
    assert len(s1) == 10
    assert sum(i < 70 for i in s1) == 7 and sum(i >= 70 for i in s1) == 3
    assert s1 == R.strat_sample(ids, 0.1, "t")
    assert s1 != R.strat_sample(ids, 0.1, "other")


def test_strat_sample_largest_remainder():
    ids = [(i, ["x", "y", "z"][i % 3]) for i in range(1319)]
    assert len(R.strat_sample(ids, 0.1, "t")) == 132


def test_agreement_and_pin():
    a = [1, 1, 0, 0, 1, 0, 1, 0]
    g = R.agreement(a, a)
    assert g["kappa"] == 1.0 and g["admit"]
    g2 = R.agreement(a, [1 - x for x in a])
    assert g2["kappa"] < 0 and not g2["admit"]
    assert R.pinned(0.5)["in_0.05_0.95"] and not R.pinned(0.97)["in_0.05_0.95"] and not R.pinned(None)["in_0.05_0.95"]


def test_compare_check():
    v = [{"rid": "c|0", "text": "x" * 120, "n_prompt": 10}, {"rid": "c|1", "text": "short", "n_prompt": 9}]
    h = [{"rid": "c|0", "text": "x" * 100 + "y" * 20, "n_prompt": 10}, {"rid": "c|1", "text": "short!", "n_prompt": 9}]
    out = R.compare_check(v, h)
    assert out["rows"][0]["match_first_100"] and not out["rows"][0]["identical"]
    assert not out["rows"][1]["match_first_100"]  # shorter reply must match whole
    assert out["prompt_tokens_same"] == 2 and out["text_match"] == 1 and out["admit_vllm"]  # n-1 of n


def test_sycon_pin_scaling_and_ministral_prompt():
    msgs = R.P.sycon_base_messages("Q?", "stance", "ministral")
    assert msgs[0]["role"] == "system" and "Qwen" not in msgs[0]["content"]
    assert len(R.x_sycon_qids()) == 10
