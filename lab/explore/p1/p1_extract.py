"""P1's text features under interventions. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private (lab/explore).

The protocol is K1's (features.embed_texts, which mirrors the paper's prh/features.py): the texts in batches of 64,
padding=True, truncation=True, max_length=512, the tokenizer's own padding side, and the mask-weighted mean in float32
(features._mean, imported) of every hidden state: the embedding output, the output of each block but the last, and the
last block's output after the final norm (layers + 1 states, as output_hidden_states gives them).

What differs is how the states are read. transformers 5.21 records output_hidden_states with forward hooks that it
installs once, the first time a model is asked for them, and forward hooks run in the order they were registered. If
those hooks went on before an intervention's (anatta.dial's ablation hooks), they record a cut layer's output before
the cut, while the next block reads it after the cut. So embed_hooked reads every state with hooks of its own,
registered for each batch after whatever interventions are already on: a cut layer's state is the state after the cut.
On an uncut model it gives K1's embed_texts exactly (the test checks equality on a tiny model; the container checks it
on the real one, and records what output_hidden_states gives under the cut).

It can also project chosen blocks' outputs onto unit vectors and average the projection over each text's caption tokens
only (the real tokens after the template's own tokens): the projection check.
"""

from __future__ import annotations

import torch


def decoder(model):
    """(the module that holds the decoder blocks, the blocks): the text decoder's base model, without the LM head
    (model.model for a causal LM, model.model.language_model for Ministral 3), found as the module whose .layers is
    anatta.models.get_layers(model)."""
    from anatta.models import get_layers

    layers = get_layers(model)
    for mod in model.modules():
        if getattr(mod, "layers", None) is layers:
            return mod, layers
    raise ValueError("no module holds the decoder layers")


def prefix_ids(tok, text: str) -> list[int]:
    """A template's prefix as the protocol tokenizes text (the tokenizer's special tokens included, so BOS where the
    tokenizer adds one): the ids a templated text begins with when the boundary with the caption is clean."""
    return list(tok(text, truncation=True, max_length=512)["input_ids"])


def caption_starts(tok, texts: list[str], pre: list[int], max_length: int = 512) -> list[int | None]:
    """For each text, the index (among its real tokens) of its caption's first token: len(pre) if its ids begin with
    pre, else None (the tokenizer merged across the boundary, and the text is left out of the caption-only mean)."""
    n = len(pre)
    out = []
    for ids in tok(texts, truncation=True, max_length=max_length)["input_ids"]:
        ids = list(ids)
        out.append(n if ids[:n] == pre and len(ids) > n else None)
    return out


@torch.no_grad()
def embed_hooked(model, tok, texts: list[str], batch_size: int = 64, max_length: int = 512, project: dict | None = None,
                 starts: list | None = None) -> tuple[list, dict]:
    """(features, projections). features: the masked mean of every hidden state, layers + 1 tensors of [N, d] float32
    on the CPU, read after any interventions already on the model (module docstring). projections: for each block x in
    `project` ({x: vector}, normalised here), each text's mean over its caption tokens (real tokens at index >=
    starts[i]) of the block's output projected on the unit vector, [N] float32 (NaN where starts[i] is None)."""
    from features import _mean  # K1's pooling, unchanged

    body, layers = decoder(model)
    n_layers = len(layers)
    dev = next(body.parameters()).device
    pv = {int(x): (v.flatten().float() / v.float().norm()).to(dev) for x, v in (project or {}).items()}
    if pv and (starts is None or len(starts) != len(texts)):
        raise ValueError("projections need one caption start per text")
    out, proj = [], {x: [] for x in pv}
    for b0 in range(0, len(texts), batch_size):
        chunk = texts[b0:b0 + batch_size]
        t = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
        mask = t["attention_mask"]
        pooled: list = [None] * (n_layers + 1)
        cmask = None
        if pv:
            st = torch.tensor([-1 if s is None else int(s) for s in starts[b0:b0 + len(chunk)]], device=dev)
            real = mask.cumsum(1) - 1  # each real token's index among its text's real tokens (pads: -1 or repeats)
            cmask = (mask.bool() & (st[:, None] >= 0) & (real >= st[:, None])).float()

        def first(_module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            pooled[0] = _mean(h, mask)

        def after(i):
            def fn(_module, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                if i < n_layers - 1:
                    pooled[i + 1] = _mean(h, mask)
                if i in pv:
                    p = h.float() @ pv[i]  # [B, T]
                    proj[i].append(((p * cmask).sum(1) / cmask.sum(1)).float().cpu())
            return fn

        handles = [layers[0].register_forward_pre_hook(first, with_kwargs=True)]
        handles += [layers[i].register_forward_hook(after(i)) for i in sorted(set(range(n_layers - 1)) | set(pv))]
        try:
            res = body(**t)
        finally:
            for h in handles:
                h.remove()
        pooled[n_layers] = _mean(res.last_hidden_state, mask)
        out.append([p.float().cpu() for p in pooled])
    feats = [torch.cat(layer) for layer in zip(*out)]
    return feats, {x: torch.cat(v) for x, v in proj.items()}


def unit(v) -> torch.Tensor:
    v = torch.as_tensor(v).flatten().float()
    return v / v.norm()


def pooled_projection(feats: list, vectors: dict) -> dict:
    """Each text's pooled feature projected on each block's unit vector: block x's output is hidden state x + 1."""
    return {int(x): feats[int(x) + 1].float() @ unit(v) for x, v in vectors.items()}
