"""Model loading and layer access for HuggingFace decoder-only models."""

from __future__ import annotations

import torch
from torch import nn


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_id: str, device: str | None = None, dtype: str = "auto"):
    """Load a causal LM and its tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    device = pick_device(device)
    if dtype == "auto":
        torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    else:
        torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    except ValueError:  # vision-language checkpoints (Ministral 3) have no causal-LM class; text-only use
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device).eval()
    return model, tok


def get_layers(model: nn.Module) -> nn.ModuleList:
    """Return the ModuleList of transformer blocks (residual stream lives between them)."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) == n:
            return module
    raise ValueError(f"Could not locate transformer layers on {type(model).__name__}")


def _first_module(block: nn.Module, names: tuple[str, ...], what: str) -> nn.Module:
    for name in names:
        m = getattr(block, name, None)
        if isinstance(m, nn.Module):
            return m
    raise ValueError(f"no {what} module ({' or '.join(names)}) on {type(block).__name__}")


def get_attn(block: nn.Module) -> nn.Module:
    """A transformer block's attention module (self_attn on Llama, Qwen2 and Mistral blocks). Its output is a tuple
    whose element 0 is what the block adds to the residual stream."""
    return _first_module(block, ("self_attn", "attn", "attention", "self_attention"), "attention")


def get_mlp(block: nn.Module) -> nn.Module:
    """A transformer block's MLP module (mlp on Llama, Qwen2 and Mistral blocks)."""
    return _first_module(block, ("mlp", "feed_forward", "ffn"), "MLP")


def as_messages(prompt) -> list[dict]:
    """A prompt as a chat: a str is one user message, a list of messages is used as it is."""
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt


def chat_ids(tok, messages: list[dict], add_generation_prompt: bool = True) -> torch.Tensor:
    """Tokenize a chat with the model's template. Returns input_ids of shape [1, T]."""
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, return_tensors="pt"
    )
    if not isinstance(ids, torch.Tensor):  # newer transformers may return a BatchEncoding
        ids = ids["input_ids"]
    return ids


def chat_batch(tok, prompts: list, side: str = "left"):
    """A padded batch of chat prompts: each prompt (a str is one user message) rendered by chat_ids with the
    generation prompt, so every row holds exactly the ids chat_ids gives it alone, padded with tok.pad_token_id on
    the left (or, with side="right", on the right). Returns a transformers BatchEncoding with input_ids and
    attention_mask (0 on the pads), read as .input_ids or ["input_ids"] alike."""
    from transformers import BatchEncoding

    if side not in ("left", "right"):
        raise ValueError(f"side is 'left' or 'right', not {side!r}")
    rows = [chat_ids(tok, as_messages(p))[0] for p in prompts]
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        span = slice(width - len(r), width) if side == "left" else slice(0, len(r))
        ids[i, span] = r
        mask[i, span] = 1
    return BatchEncoding({"input_ids": ids, "attention_mask": mask})
