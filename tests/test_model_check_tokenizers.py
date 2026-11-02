"""Opt-in: the model check's prompt checks on the real tokenizers it runs, Qwen2.5-7B-Instruct at a09a354 and
Ministral 3 8B at f6fae97, read from the local Hugging Face cache ($HF_HOME or ~/.cache/huggingface; read only,
local files only, no weights). Each test is skipped where the cache lacks that snapshot.

  - check_prompt passes on Qwen with no system message (its template writes none of its own BOS);
  - check_prompt passes on Ministral with no system message, "You are a helpful assistant." and "": exactly one
    BOS each time, the template's long default (the Le Chat prompt) only when there is no system message;
  - check_batch catches Ministral's rendered prompts tokenized with add_special_tokens=True (a second BOS), and
    passes them tokenized as probe.generate does.
"""

from pathlib import Path

import pytest

from anatta.bench import run as br

QWEN = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")
MINISTRAL = ("mistralai/Ministral-3-8B-Instruct-2512-BF16", "f6fae9795746f63c9be8344932f01275f3c63734")
PROMPTS = ['What is 2 + 2?\n\nWork it out step by step, then give the final answer on the last line as "Answer: <number>".',
           "Which is a prime?\nA. 4\nB. 7\n\nReply with the letter of the correct option only."]


def cached_tokenizer(model: str, revision: str):
    try:
        path = Path(br.snapshot_dir(model, revision))
    except FileNotFoundError:
        pytest.skip(f"{model} at {revision[:7]} is not in the local Hugging Face cache")
    if not (path / "tokenizer.json").exists():
        pytest.skip(f"{model} at {revision[:7]}: no tokenizer.json in the cached snapshot")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tok.pad_token is None:  # as anatta.models.load does
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def qwen():
    return cached_tokenizer(*QWEN)


@pytest.fixture(scope="module")
def ministral():
    return cached_tokenizer(*MINISTRAL)


def test_qwen_prompts_pass_the_check(qwen):
    for p in PROMPTS:
        text, ids = br.check_prompt(qwen, br.messages(p), bos=0)
        assert qwen.bos_token_id is None and text.startswith("<|im_start|>system\nYou are Qwen")
        assert ids == br.template_ids(qwen, br.messages(p))


@pytest.mark.parametrize("system", [None, "You are a helpful assistant.", ""])
def test_ministral_prompts_pass_the_check_with_one_bos(ministral, system):
    for p in PROMPTS:
        text, ids = br.check_prompt(ministral, br.messages(p, system), bos=1)
        assert ids[0] == ministral.bos_token_id and ids[1] != ministral.bos_token_id  # exactly one
        assert ("Le Chat" in text) is (system is None)  # the template's default only without a system message
        if system:
            assert system in text


def test_check_batch_catches_add_special_tokens_on_ministral(ministral):
    conversations = [br.messages(p, "") for p in PROMPTS]
    texts = [ministral.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in conversations]
    expect = [br.check_prompt(ministral, c, bos=1)[1] for c in conversations]
    side, ministral.padding_side = ministral.padding_side, "left"
    try:
        good = ministral(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        bad = ministral(texts, return_tensors="pt", padding=True, add_special_tokens=True)
    finally:
        ministral.padding_side = side
    br.check_batch(dict(good), expect, ministral.pad_token_id)
    assert bad["input_ids"][0][-len(expect[0]) - 1] == ministral.bos_token_id  # the second BOS, just before the ids
    with pytest.raises(br.CheckError):
        br.check_batch(dict(bad), expect, ministral.pad_token_id)
