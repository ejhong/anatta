"""The serving check: its comparison and pass rule offline (anatta.backends.serving), the vLLM engine's settings from
the registry, and (marker modal, about $1) the check itself on Modal. Claude (Opus 5.5) wrote this file."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from anatta.backends import serving as SV
from anatta.backends.vllm import Settings, merge_system

REPO = Path(__file__).resolve().parents[1]
EOS = 2


def test_fixed_inputs():
    assert len(SV.PROMPTS) == 16 and len(set(SV.PROMPTS)) == 16
    assert SV.CONFIGS == (("qwen25_7b", None), ("ministral3_8b", None), ("ministral3_8b", SV.ADAPTER))
    assert SV.config_name("ministral3_8b", SV.ADAPTER) == "ministral3_8b+gameable_s0/adapter_r3"


def test_compare_row_cuts_at_the_end_token():
    same = SV.compare_row([1, 5], [1, 5], [7, 8, 9, EOS, 4], [7, 8, 9, EOS], EOS)
    assert same["identical"] and same["identical_32"] and same["agree_64"] == 1.0 and same["stopped_vllm"]
    early = SV.compare_row([1], [1], [7, 8, EOS], [7, 8, 9, 10], EOS)
    assert early["common_prefix_tokens"] == 2 and not early["identical_32"] and early["agree_64"] == 0.5
    split = SV.compare_row([1], [2], list(range(10, 80)), list(range(10, 50)) + [0] * 30, EOS)
    assert not split["same_prompt_ids"] and split["identical_32"] and split["common_prefix_tokens"] == 40
    assert split["n_out_hf"] == 64 and split["agree_64"] == round(40 / 64, 4)


def test_pass_rule_is_14_of_16_with_all_prompt_ids_equal():
    good = SV.compare_row([1], [1], [3, 4], [3, 4], EOS)
    bad = SV.compare_row([1], [1], [3, 4], [3, 5], EOS)
    assert SV.summarize([good] * 14 + [bad] * 2)["pass"]
    assert not SV.summarize([good] * 13 + [bad] * 3)["pass"]
    wrong_ids = SV.compare_row([1], [9], [3, 4], [3, 4], EOS)
    s = SV.summarize([good] * 15 + [wrong_ids])
    assert not s["pass"] and s["identical_first_32"] == 16 and s["prompt_ids_same"] == 15


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
    card = json.loads(max(out.glob("serving_check*.json"), key=lambda p: p.stat().st_mtime).read_text())
    assert card["pass"], card["configs"]
