"""The HF backend: a registered model loaded from its pinned snapshot through anatta.models.load (bf16 on a GPU, the
vision-checkpoint fallback for Ministral 3), with an optional LoRA adapter through anatta.player.attach_adapters, and
greedy generation one prompt at a time. Runs where the weights are: inside a Modal container with the model's volume
mounted. Claude (Opus 5.5) wrote this file, 3 Nov 2026.
"""

from __future__ import annotations

from .registry import get


def load(name: str, adapter: str | None = None, device: str | None = None, dtype: str = "auto",
         mount: str | None = None):
    """(model, tok) for registry key or alias `name`, from its snapshot (mount overrides the volume's mount point);
    `adapter` is a PEFT adapter folder to put on top, frozen, in eval mode."""
    from anatta.models import load as load_hf
    from anatta.player import attach_adapters

    model, tok = load_hf(get(name).snapshot(mount), device=device, dtype=dtype)
    model = attach_adapters(model, adapter)
    model.eval()
    return model, tok


def greedy(model, tok, prompt_ids: list[list[int]], max_new: int) -> list[list[int]]:
    """Greedy continuations, one prompt at a time (no padding, so no batch effects), as token ids up to and including
    the first end token. Sampling settings in the checkpoint's generation_config (Qwen ships repetition_penalty 1.05,
    top_k 20) are switched off, so this is plain argmax decoding."""
    import torch

    stop = tok.eos_token_id
    out = []
    for ids in prompt_ids:
        x = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            y = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                               temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                               pad_token_id=tok.pad_token_id, eos_token_id=stop)
        out.append(y[0, len(ids):].tolist())
    return out


def chat_prompt_ids(tok, prompts: list) -> list[list[int]]:
    """Each prompt (a str is one user message) rendered by the model's chat template with the generation prompt:
    anatta.models.chat_ids, the lab's reference rendering."""
    from anatta.models import as_messages, chat_ids

    return [chat_ids(tok, as_messages(p))[0].tolist() for p in prompts]
