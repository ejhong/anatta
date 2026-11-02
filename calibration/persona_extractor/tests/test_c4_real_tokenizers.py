"""The slices on the real tokenizers, from the local Hugging Face cache at the pinned revisions (skipped when they are
not there; nothing is downloaded): Qwen's template closes a reply with '<|im_end|>' and '\\n', Ministral's with '</s>';
generate_vec's prompt length is probe's start on both; Qwen's generation_config is the registered one."""

import json
import os

import pytest
import readings as Rd
import traits as T
import yaml
from c4_testing import CHECK

CFG = yaml.safe_load((CHECK / "config.yaml").read_text())
ANSWERS = ["The garden grows beans.", " a leading space", "\na leading newline", "café 日本 — text"]


def tokenizer(key):
    m = CFG["models"][key]
    path = T.snapshot_dir(m["id"], m["revision"])
    if not (path / "tokenizer_config.json").exists() and not (path / "tokenizer.json").exists():
        pytest.skip(f"{m['id']} at {m['revision'][:8]} is not in the local cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path)), path


@pytest.mark.parametrize("answer", ANSWERS)
@pytest.mark.parametrize("system", ["You are an upbeat assistant. Speak warmly about the garden.", None])
def test_qwen_closes_a_reply_with_im_end_and_a_newline(answer, system):
    tok, _ = tokenizer("qwen")
    s = Rd.token_slices(tok, Rd.messages(system, "What grows in a garden?"), answer)
    assert s["n_closing"] == 2 and s["n_answer"] >= 1
    assert tok.convert_ids_to_tokens(s["full"][0, s["end_trim"]:s["end"]].tolist()) == ["<|im_end|>", "Ċ"]


@pytest.mark.parametrize("answer", ANSWERS)
def test_ministral_closes_a_reply_with_the_end_of_sequence_token(answer):
    tok, _ = tokenizer("ministral")
    s = Rd.token_slices(tok, Rd.messages(None, "What grows in a garden?"), answer)
    assert s["n_closing"] == 1 and tok.convert_ids_to_tokens(s["full"][0, s["end_trim"]:].tolist()) == ["</s>"]


def test_qwens_generation_config_is_the_registered_one():
    _, path = tokenizer("qwen")
    got = json.loads((path / "generation_config.json").read_text())
    for k, v in CFG["models"]["qwen"]["generation_config"].items():
        assert got[k] == v
    assert got["eos_token_id"] == CFG["models"]["qwen"]["eos"]
