"""anatta.bench.mc: the MMLU prompt against lm-evaluation-harness's template text (at 314f717), the one-token rule for
the continuations, right-padded batches read at each row's last real token, the chat-template variant and accuracy.
Every item is made up."""

import jinja2
import pytest
import torch

from anatta.bench import mc
from anatta.models import chat_batch, chat_ids

# lm_eval/tasks/mmlu/default/_default_template_yaml at 314f717 (its doc_to_text, YAML's "\n" read as a newline), and
# the per-subject description as mmlu_abstract_algebra.yaml writes it, with the subject's underscores as spaces.
LM_EVAL_DOC_TO_TEXT = ("{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\n"
                       "D. {{choices[3]}}\nAnswer:")
LM_EVAL_DESCRIPTION = "The following are multiple choice questions (with answers) about {subject}.\n\n"

ITEMS = [  # made-up items of mixed lengths: (question, choices, subject, answer)
    ("Which word comes first?", ["fox", "dog", "the", "over"], "high_school_reading", 2),
    ("  How many legs has a fox?\n", ["2", "4", "6", "8"], "animal_facts", 1),
    ("Pick the number.", ["zero", "one 1", "two", "three 3 3 3"], "basic_numbers", 0),
    ("Which of these is quick: the brown fox, the lazy dog, or neither of the two of them at all?",
     ["the brown fox", "the lazy dog", "both", "neither"], "x", 0),
    ("Final?", ["a", "b", "c", "d"], "y_z", 3),
    ("Which answer is final: 0123, 4567, 89, or none?", ["0123", "4567", "89", "none of the numbers"], "math", 2),
]


@pytest.fixture(scope="module")
def letters_tiny():
    """A tiny random Llama and a byte-level tokenizer trained so that " A" to " D" are one token each, as they are in
    Qwen's and Ministral's tokenizers."""
    from conftest import SPECIAL, TEMPLATE
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    corpus = ["the quick brown fox jumps over the lazy dog 0123456789 final answer Answer: A B C D"] * 50
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(1)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return LlamaForCausalLM(cfg).eval(), tok


def prompts():
    return [mc.mmlu_prompt(q, c, s) for q, c, s, _ in ITEMS]


# --------------------------------------------------------------------------------------------- the prompt


def test_the_prompt_is_lm_evals_description_then_doc_to_text():
    env = jinja2.Environment()
    for q, c, s, _ in ITEMS:
        want = (LM_EVAL_DESCRIPTION.format(subject=s.replace("_", " "))
                + env.from_string(LM_EVAL_DOC_TO_TEXT).render(question=q, choices=c))
        assert mc.mmlu_prompt(q, c, s) == want
    assert mc.mmlu_prompt("Q?", ["w", "x", "y", "z"], "abstract_algebra") == (
        "The following are multiple choice questions (with answers) about abstract algebra.\n\n"
        "Q?\nA. w\nB. x\nC. y\nD. z\nAnswer:")
    with pytest.raises(ValueError):
        mc.mmlu_prompt("Q?", ["w", "x", "y"], "s")


def test_each_continuation_must_be_one_token(tiny, letters_tiny):
    _, tok = letters_tiny
    ids = mc.letter_ids(tok)
    assert len(set(ids)) == 4
    assert [tok.decode([i]) for i in ids] == [" A", " B", " C", " D"]
    with pytest.raises(ValueError, match="one token"):
        mc.letter_ids(tiny[1])  # conftest's tokenizer never saw " A": it is two tokens there


def test_encode_uses_the_default_special_tokens_or_the_chat_template(letters_tiny):
    _, tok = letters_tiny
    texts = prompts()[:2]
    assert mc.encode(tok, texts) == [tok(t)["input_ids"] for t in texts]
    assert mc.encode(tok, texts, chat=True) == [chat_ids(tok, [{"role": "user", "content": t}])[0].tolist()
                                                for t in texts]


def test_pad_right_and_the_last_real_index():
    ids, mask = mc.pad_right([[5, 6, 7], [8], [9, 10]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0], [9, 10, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0], [1, 1, 0]]
    assert mc.last_real_index(mask).tolist() == [2, 0, 1]


# --------------------------------------------------------------------------------------------- the scoring


def single_logits(model, tok, texts, letters, chat=False):
    """Each prompt alone, unpadded: the four continuation logits at its last token."""
    out = []
    with torch.no_grad():
        for row in mc.encode(tok, texts, chat):
            logits = model(input_ids=torch.tensor([row])).logits
            out.append(logits[0, -1, letters].float().tolist())
    return out


@pytest.mark.parametrize("chat", [False, True])
def test_batch_4_and_batch_1_agree_on_mixed_length_items(letters_tiny, chat):
    model, tok = letters_tiny
    letters = mc.letter_ids(tok)
    texts = prompts()
    assert len({len(r) for r in mc.encode(tok, texts, chat)}) > 2  # the lengths really are mixed
    b4 = mc.predict(model, tok, texts, letters, batch_size=4, chat=chat)
    b1 = mc.predict(model, tok, texts, letters, batch_size=1, chat=chat)
    assert b4["preds"] == b1["preds"]
    assert torch.allclose(torch.tensor(b4["logits"]), torch.tensor(b1["logits"]), atol=1e-5)
    assert torch.allclose(torch.tensor(b1["logits"]), torch.tensor(single_logits(model, tok, texts, letters, chat)),
                          atol=1e-5)
    assert b4["preds"] == [mc.first_argmax(v) for v in b4["logits"]]


def test_a_read_at_index_minus_one_fails_the_batch_test(letters_tiny, monkeypatch):
    """The planted fault: reading every row at index -1 reads a pad for each row shorter than its batch's longest,
    so batch 4 no longer matches batch 1."""
    model, tok = letters_tiny
    letters = mc.letter_ids(tok)
    texts = prompts()
    b1 = mc.predict(model, tok, texts, letters, batch_size=1)
    monkeypatch.setattr(mc, "last_real_index", lambda mask: torch.full((mask.shape[0],), mask.shape[1] - 1))
    b4 = mc.predict(model, tok, texts, letters, batch_size=4)
    assert not torch.allclose(torch.tensor(b4["logits"]), torch.tensor(b1["logits"]), atol=1e-5)


def test_predict_reads_through_the_callers_hooks(letters_tiny):
    """predict scores with whatever is installed: a hook that adds a large vector at the last block's output moves
    the logits."""
    from anatta.dial import Intervention, assert_no_hooks, intervene
    from anatta.models import get_layers

    model, tok = letters_tiny
    letters = mc.letter_ids(tok)
    texts = prompts()[:3]
    plain = mc.predict(model, tok, texts, letters, batch_size=2)
    last = len(get_layers(model)) - 1
    v = torch.randn(64, generator=torch.Generator().manual_seed(3))
    with intervene(model, [Intervention(last, v, "steer", alpha=50.0)]):
        moved = mc.predict(model, tok, texts, letters, batch_size=2)
    assert not torch.allclose(torch.tensor(plain["logits"]), torch.tensor(moved["logits"]), atol=1e-3)
    assert_no_hooks(model)


def test_first_argmax_takes_the_first_on_ties_and_accuracy_is_a_micro_average():
    assert mc.first_argmax([0.5, 2.0, 2.0, -1.0]) == 1
    assert mc.first_argmax([3.0, 3.0, 3.0, 3.0]) == 0
    assert mc.accuracy([0, 1, 2, 3], [0, 1, 0, 0]) == 0.5
    assert mc.accuracy([2] * 7, [2] * 6 + [1]) == pytest.approx(6 / 7)
    with pytest.raises(ValueError):
        mc.accuracy([0, 1], [0])
    with pytest.raises(ValueError):
        mc.accuracy([], [])


# ------------------------------------------------------------------------------------------ models.chat_batch


def test_chat_batch_pads_each_rows_chat_ids_on_the_left_or_right(tiny):
    _, tok = tiny
    ps = ["the dog", "the quick brown fox jumps", [{"role": "user", "content": "over"}]]
    singles = [chat_ids(tok, [{"role": "user", "content": p}] if isinstance(p, str) else p)[0] for p in ps]
    width = max(len(s) for s in singles)
    left, right = chat_batch(tok, ps), chat_batch(tok, ps, side="right")
    assert left.input_ids.shape == right["input_ids"].shape == (3, width)
    for i, s in enumerate(singles):
        n = len(s)
        assert torch.equal(left.input_ids[i, width - n:], s) and torch.equal(right.input_ids[i, :n], s)
        assert (left.input_ids[i, :width - n] == tok.pad_token_id).all()
        assert (right.input_ids[i, n:] == tok.pad_token_id).all()
        assert left.attention_mask[i].tolist() == [0] * (width - n) + [1] * n
        assert right.attention_mask[i].tolist() == [1] * n + [0] * (width - n)
    with pytest.raises(ValueError):
        chat_batch(tok, ps, side="middle")


def test_a_left_padded_batch_gives_each_prompts_last_position_logits(tiny):
    """Arditi's scores read logits[:, -1] of left-padded batches: each row's last position is its own last token, and
    (rotary positions being relative) its logits match the prompt alone."""
    model, tok = tiny
    ps = ["the dog", "the quick brown fox jumps over the lazy dog", "final answer 0123"]
    with torch.no_grad():
        batch = model(**chat_batch(tok, ps)).logits[:, -1]
        alone = torch.cat([model(chat_ids(tok, [{"role": "user", "content": p}])).logits[:, -1] for p in ps])
    assert torch.allclose(batch, alone, atol=1e-4)
