"""The inside look's offline tests: the pure parts (look_core.py) and the torch side (look_torch.py) on a tiny random
model with a byte-level chat tokenizer, built offline as the bench's tests build theirs. Nothing is downloaded and
nothing is spent. Claude (Fable 5.2, an AI model) wrote this file on 16 Nov 2026.

Run: .venv/bin/python -m pytest lab/explore/attractor/tests/test_look.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
for p in (str(HERE), str(HERE.parent / "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

import attractor_core as C  # noqa: E402
import look_core as K  # noqa: E402
import look_torch as T  # noqa: E402

SPECIAL = ["<pad>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}")


@pytest.fixture(scope="module")
def svc():
    import steering
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(["the quick brown fox jumps over the lazy dog 0123456789 final answer"] * 50,
                          trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return steering.Service(LlamaForCausalLM(cfg).eval(), tok, model_key="tiny", model_id="tiny-llama",
                            revision="test", gen_batch=4)


CHAT = [{"role": "system", "content": "two copies talk"}, {"role": "user", "content": "the quick brown fox"},
        {"role": "assistant", "content": "jumps over the lazy dog"}, {"role": "user", "content": "final answer 42"},
        {"role": "assistant", "content": "the dog 7"}]


def test_runs_are_the_stretches_between_control_tokens():
    assert T.runs([9, 1, 2, 9, 9, 3, 9], {9}) == [(1, 3), (5, 6)]
    assert T.runs([1, 2], {9}) == [(0, 2)]
    assert T.runs([9, 9], {9}) == []


def test_each_message_gets_its_own_span(svc):
    from anatta.models import chat_ids

    full = chat_ids(svc.tok, CHAT, add_generation_prompt=False)[0].tolist()
    spans, method = T.message_spans(svc.tok, CHAT, full)
    assert method == "runs" and len(spans) == len(CHAT)
    for (a, b), m in zip(spans, CHAT):
        assert svc.tok.decode(full[a:b]) == m["content"]
    by_prefix = T.spans_by_prefix(svc.tok, CHAT, full, T.control_ids(svc.tok))
    assert by_prefix == spans


def test_reading_in_chunks_gives_the_same_states_as_one_pass(svc):
    from anatta.models import chat_ids
    from anatta.probe import capture

    ids = chat_ids(svc.tok, CHAT, add_generation_prompt=False)
    with capture(svc.model, [1, 3]) as store:
        svc.model(input_ids=ids)
    whole = {x: store[x][0].clone() for x in (1, 3)}
    for chunk in (5, 16, 4096):
        got = T.layer_states(svc, ids, [1, 3], chunk=chunk)
        for x in (1, 3):
            assert torch.allclose(got[x], whole[x], atol=1e-4), (chunk, x)


def test_read_returns_each_messages_mean_and_leaves_no_hook(svc):
    from anatta.models import chat_ids
    from anatta.probe import capture

    res = T.read(svc, {"convs": [{"id": "a", "chat": CHAT}, {"id": "long", "chat": CHAT}], "layers": [2],
                       "max_tokens": 10_000})
    c = res["convs"][0]
    assert c["method"] == "runs" and c["head_ok"] == len(CHAT) and c["means"][2].shape == (len(CHAT), 64)
    ids = chat_ids(svc.tok, CHAT, add_generation_prompt=False)
    with capture(svc.model, [2]) as store:
        svc.model(input_ids=ids)
    a, b = c["spans"][2]
    assert np.allclose(c["means"][2][2].astype(np.float32), store[2][0, a:b].mean(0).numpy(), atol=2e-2)
    assert T.read(svc, {"convs": [{"id": "a", "chat": CHAT}], "layers": [2], "max_tokens": 5})["convs"][0]["skipped"]
    # The two options for many short chats: only the last message's row, and the mean over its last tokens.
    few = T.read(svc, {"convs": [{"id": "a", "chat": CHAT}], "layers": [2], "max_tokens": 10_000, "keep_last": 1,
                       "tail_tokens": 3})["convs"][0]
    assert few["means"][2].shape == (1, 64) and len(few["norms"][2]) == len(CHAT) and few["spans"] == c["spans"]
    assert np.array_equal(few["means"][2][0], c["means"][2][-1])
    a, b = c["spans"][-1]
    assert b - a > 3
    assert np.allclose(few["tail_means"][2].astype(np.float32), store[2][0, b - 3:b].mean(0).numpy(), atol=2e-2)
    whole = T.read(svc, {"convs": [{"id": "a", "chat": CHAT}], "layers": [2], "max_tokens": 10_000,
                         "tail_tokens": 10_000})["convs"][0]
    assert np.allclose(whole["tail_means"][2].astype(np.float32), c["means"][2][-1].astype(np.float32), atol=2e-2)
    assert "tail_means" not in c and few["tail_text"].split() == CHAT[-1]["content"].split()[-len(few["tail_text"].split()):]
    svc.check_clean()


def test_a_default_system_prompt_from_the_template_is_not_taken_for_a_message(svc):
    """Mistral's template writes its own system prompt when a chat has none. The reader's spans must still be the
    chat's messages, or every reading of such a chat would stop (the self test's design check, 16 Nov 2026)."""
    kept = svc.tok.chat_template
    svc.tok.chat_template = ("{% if messages[0]['role'] != 'system' %}<|system|>a default prompt of some words<|end|>"
                             "{% endif %}" + TEMPLATE)
    try:
        chat = CHAT[1:]
        res = T.read(svc, {"convs": [{"id": "a", "chat": chat}], "layers": [2], "max_tokens": 10_000,
                           "tail_tokens": 2})["convs"][0]
        assert res["method"] == "runs after a default system prompt" and res["head_ok"] == len(chat)
        assert res["means"][2].shape == (len(chat), 64) and len(res["spans"]) == len(chat)
        with_system = T.read(svc, {"convs": [{"id": "b", "chat": CHAT}], "layers": [2], "max_tokens": 10_000})
        assert with_system["convs"][0]["method"] == "runs"
    finally:
        svc.tok.chat_template = kept
    svc.check_clean()


def test_rollouts_read_the_reply_with_its_chat_in_context(svc):
    rows = [{"id": "r0", "messages": CHAT[:2], "seed": 1}, {"id": "r1", "messages": CHAT[1:2], "seed": 2}]
    res = T.rollouts(svc, {"rows": rows, "layers": [1, 2], "max_tokens": 6, "temperature": 0.7})
    assert [r["id"] for r in res["replies"]] == ["r0", "r1"]
    assert res["means"][1].shape == (2, 64) and res["means"][2].dtype == np.float16
    assert all(r["reply_tokens"] >= 1 for r in res["replies"])


def test_converse_takes_turns_and_steers_only_where_asked(svc):
    d = torch.zeros(64)
    d[3] = 1.0
    dirs = {"k": {"name": "d", "vectors": {2: d.numpy()}}}
    iv = {"mode": "add", "direction": "d", "tensor_sha256": "k", "sha256": "k", "layers": [2], "coefficient": 8.0}
    base = {"system_prompt": "two copies talk", "opening_line": "go", "n_more": 3, "seed": 5}
    convs = [{**base, "id": "plain", "messages": [], "intervention": {"mode": "none"}},
             {**base, "id": "pushed", "messages": [], "intervention": iv},
             {**base, "id": "midway", "messages": ["the fox", "the dog"], "n_more": 2, "intervention": {"mode": "none"}}]
    res = T.converse(svc, {"convs": convs, "directions": dirs, "max_tokens": 5, "temperature": 1.0}, C.view)
    out = {c["id"]: c for c in res["convs"]}
    assert out["plain"]["new"] + out["pushed"]["new"] >= 2  # a tiny random model may write an empty reply and stop
    assert out["midway"]["messages"][:2] == ["the fox", "the dog"]
    assert all(len(c["messages"]) == c["new"] + (2 if c["id"] == "midway" else 0) for c in out.values())
    svc.check_clean()


# ------------------------------------------------------------------------------------------------ the pure parts


def test_the_calibration_rows_and_the_role_halves_are_fixed():
    rows = K.rollout_rows()
    assert len(rows) == (24 + 24 + 6) * 2 * 16 + 5 * 16 + 32 and len({r["id"] for r in rows}) == len(rows)
    assert sum(r["split"] == "build" for r in rows if r["kind"] == "default") == 40
    assert rows[0]["messages"][0]["role"] == "system" and rows[-1]["messages"][0]["role"] == "user"
    s = K.role_split()
    assert len(s["build"]) == 24 and len(s["held"]) == 24 and not set(s["build"]) & set(s["held"])
    assert not set(s["spiritual"]) & set(s["build"] + s["held"])
    assert K.role_split() == s
    assert sum(n in K.NEAR for n in s["held"]) == 12


def test_auroc_and_the_wording_of_a_difference():
    assert K.auroc([3, 4, 5], [0, 1, 2]) == 1.0 and K.auroc([0, 1], [2, 3]) == 0.0
    assert K.auroc([1, 1], [1, 1]) == 0.5 and K.auroc([], [1]) is None
    d = K.paired_diff([1, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0])
    assert abs(d["diff"] - 0.75) < 1e-9 and d["lo"] > 0 and K.shown(d, 0.30) == "a difference"
    same = K.paired_diff([1, 0] * 40, [1, 0] * 40)
    assert K.shown(same, 0.30) == "no difference shown"
    assert K.shown({"diff": 0.2, "lo": -0.2, "hi": 0.6}, 0.30) == "not settled"
    assert K.shown(d, 0.30, 1) == "a difference" and K.shown(d, 0.30, -1) == "a difference the other way"
    assert abs(K.explained(np.array([1.0, 1.0, 0.0]), [np.array([1.0, 0, 0]), np.array([0, 1.0, 0])]) - 1.0) < 1e-9
    assert abs(K.cos(np.array([1.0, 0.0]), np.array([0.0, 2.0]))) < 1e-12
    log = [{"cost_usd": 3.0, "rate_usd_per_hour": 2.6}, {"cost_usd": 0.25, "kind": "grade-on"}, {"cost_usd": 0.5}]
    assert K.api_share(log) == 0.75 and K.api_share([]) == 0.0  # the GPU's lines are not the API's
    rows = np.arange(7, dtype=np.float32)[:, None] * np.ones((1, 2), dtype=np.float32)  # row k holds k
    own, oth = K.own_other(rows, range(1, 9))  # the window runs past the last message; row 0 is never a message
    assert own[:, 0].tolist() == [2.0, 4.0, 6.0] and oth[:, 0].tolist() == [1.0, 3.0, 5.0]


def test_who_said_it_items_and_the_reading_of_an_answer():
    msgs = [f"Message number {k} talks at some length about the garden beds and where the beans should go this year. "
            f"Short one." for k in range(1, 31)]
    items = K.who_items(msgs)
    assert [(i["number"], i["wrote"]) for i in items] == [(24, "me"), (28, "me"), (30, "me"), (23, "other"),
                                                             (27, "other"), (29, "other")]
    assert all("garden beds" in i["line"] for i in items)
    echo = list(msgs)
    echo[22] = echo[23]  # the other copy's message 23 now holds the tested copy's line word for word
    assert 24 not in [i["number"] for i in K.who_items(echo)]
    for reply, want in (("Me.", "me"), ("Other", "other"), ("The other instance wrote it.", "other"),
                        ("mine", "me"), ("We both did.", "both"), ("Both of us", "both"), ("", None), ("Hmm.", None),
                        # hedged openings are left unread, not guessed (the design check's cases)
                        ("I think the other instance wrote that.", None), ("I would say: other.", None),
                        ("I did not write it; the other instance did.", None), ("I wrote that line.", None),
                        ("It was written by me", None)):
        assert K.parse_who(reply) == want, reply
    for reply, want in (("Yes.", "me"), ("no", "other"), ("No, the other instance did.", "other"), ("Both", "both"),
                        ("I think so.", None)):
        assert K.parse_who(reply, "did") == want, reply
    assert K.who_request("a line", "which", True)[0]["content"].endswith('"other" or "me".')
    assert K.who_request("a line", "did")[0]["content"].endswith('"yes" or "no".')


def test_the_excerpt_a_grader_reads_keeps_the_true_numbers_and_the_frozen_codebook():
    chat = K.excerpt_judge_messages(["one", "two", "three"], 23)
    assert chat[0]["content"] == C.CODEBOOK
    assert "[MESSAGE 23 | COPY A]" in chat[1]["content"] and "[MESSAGE 24 | COPY B]" in chat[1]["content"]
    assert "messages 23 to 25" in chat[1]["content"]
