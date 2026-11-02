"""P1's CPU tests: a tiny random Llama and a byte-level tokenizer built offline (bench's tests/conftest.py recipe), so
nothing is downloaded and no GPU or Modal is touched. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private.

Run (the anatta venv has pytest, torch and transformers 5.21; modal is not needed):
  /Users/eugene/prj/align/anatta/.venv/bin/python -m pytest /Users/eugene/prj/align/anatta-lab/lab/explore/p1/tests \
      -q -p no:cacheprovider
"""

import sys
from pathlib import Path

import pytest
import torch

P1 = Path(__file__).resolve().parents[1]
for p in (str(P1),):
    if p not in sys.path:
        sys.path.insert(0, p)
import p1_core  # noqa: E402,F401  (puts K1's folder, bench, lab/explore and anatta-dev/src on sys.path)

SPECIAL = ["<pad>", "<eos>", "<bos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
HIDDEN, LAYERS = 64, 4
CORPUS = ("i see a dog on the beach. one sees the old bridge in the scene shown. there appears to be a red house in "
          "front of me, a cat. i'm looking at the market square at night. 0123456789")


def _tokenizer(bos: bool):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator([CORPUS, CORPUS.upper(), CORPUS.title()] * 30,
                          trainers.BpeTrainer(vocab_size=500, special_tokens=SPECIAL,
                                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    if bos:
        t.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>",
                                  bos_token="<bos>" if bos else None)
    tok.chat_template = TEMPLATE
    return tok


@pytest.fixture(scope="session")
def tiny():
    """(model, tokenizer): a 4-block Llama of width 64, and a tokenizer that adds BOS (as Ministral's does)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = _tokenizer(bos=True)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=LAYERS,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id)
    model = LlamaForCausalLM(cfg).eval()
    return model, tok


@pytest.fixture
def captions():
    """Made-up benign captions standing in for WIT's (WIT's never leave Modal)."""
    return ["A dog on the beach", "The old bridge at night", "Berlin cathedral", "NASA logo", " the market square",
            "In 1990, a red house", "A-10 aircraft on display", "two cats", "View of the harbour", "an empty room",
            "Portrait of a woman", "Some boats in a row"]
