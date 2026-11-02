"""bench's CPU tests: a tiny random Llama and a byte-level chat tokenizer built offline (anatta-dev's tests/conftest.py
recipe), so nothing is downloaded. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private (lab/explore).

Run: PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src /Users/eugene/prj/align/anatta/.venv/bin/python -m pytest \
       /Users/eugene/prj/align/anatta-lab/lab/explore/bench/tests -q
"""

import sys
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).resolve().parents[1]
H4 = BENCH.parents[1] / "experiments" / "h4lite"  # player.ego_cut, for the equivalence test
for p in (str(BENCH), "/Users/eugene/prj/align/anatta-dev/src", str(H4)):
    if p not in sys.path:
        sys.path.insert(0, p)

SPECIAL = ["<pad>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
HIDDEN, LAYERS = 64, 4


@pytest.fixture(scope="session")
def tiny():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    corpus = ["the quick brown fox jumps over the lazy dog 0123456789 final answer"] * 50
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=LAYERS,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = LlamaForCausalLM(cfg).eval()
    return model, tok


@pytest.fixture
def svc(tiny):
    import steering

    model, tok = tiny
    return steering.Service(model, tok, model_key="tiny", model_id="tiny-llama", revision="test", gen_batch=4)
