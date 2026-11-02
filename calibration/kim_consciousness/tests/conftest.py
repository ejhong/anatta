"""K5's test setup: this check's folder on sys.path, and a tiny random Llama with a byte-level chat tokenizer whose
digits 0-9 are single tokens and "10" two (as Gemma splits it). Everything here is made up and benign; nothing touches
Modal, a GPU, the network or a real model."""

import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parent.parent
if str(CHECK) not in sys.path:
    sys.path.insert(0, str(CHECK))

SPECIAL = ["<pad>", "<eos>", "<|user|>", "<|assistant|>", "<|end|>"]
TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="session")
def tiny():
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    # digits split one by one, as Gemma's tokenizer splits them
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                               pre_tokenizers.ByteLevel(add_prefix_space=False)])
    t.decoder = decoders.ByteLevel()
    corpus = ["the quick brown fox <answer>7</answer> <think> rating answer I am fine 0 1 2 3 4 5 6 7 8 9"] * 50
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = TEMPLATE
    tok.padding_side = "left"
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = LlamaForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = tok.eos_token_id
    return model, tok


@pytest.fixture
def config():
    import yaml

    return yaml.safe_load((CHECK / "config.yaml").read_text())
