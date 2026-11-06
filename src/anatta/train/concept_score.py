"""A probe score per episode for S1's expression shaping (lab/plans/s1-shaping-2026-11-05.md). Claude (Opus 5.5) wrote
this file, 5 Nov 2026. UNCALIBRATED, like the trainer it feeds (anatta.train.expert_iteration).

  s = max over the final reply's tokens of <h_layer(t), unit>        h: the block output at `layer` (the residual
                                                                       stream after that block), read by capture()
  z = (s - mu) / sd                                                   mu, sd: fixed statistics frozen from the fit
                                                                       set, one pair per twin kind (or one scalar)

Readers. "frozen": the base weights only (the LoRA disabled with PEFT's disable_adapter()), so the same text gets the
same z in any round and arm and the dose does not move with the policy; this is the primary reader. "policy": the
model as it is (base + the current adapter), a detached current-policy reader whose statistics drift.

Capture goes through anatta.probe.capture (imported, never edited) on model.get_base_model(): on a PEFT model the
plain HF module is what get_layers can walk (Mistral3: model.language_model.layers, the text decoder; never the vision
tower). capture_check() states that at run start: the number of blocks found equals the text config's
num_hidden_layers (34 on Ministral 3 8B), no block class is a Vision one and the blocks' path has no "vision" in it
(Pixtral's own blocks are not named Vision).

Each history is a message list whose last message is the assistant's final reply (as replay() rebuilds it). The
reply's tokens are those after the chat template's rendering of the history without it (with the generation prompt);
special tokens are left out of the max. A history whose prompt does not render as a prefix of the full rendering is
scored on the tail anyway and flagged (prefix_ok False).
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

READERS = ("frozen", "policy")


def base_of(model):
    """The plain HF module: model.get_base_model() on a PEFT model, the model itself otherwise."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def text_layers(model) -> int:
    cfg = model.config
    if hasattr(cfg, "get_text_config"):
        return cfg.get_text_config().num_hidden_layers
    return cfg.num_hidden_layers


def capture_check(model) -> dict:
    """The blocks capture() hooks on the base module: their number must equal the text config's depth and none may be
    a vision block. Raises AssertionError with the record otherwise."""
    from anatta.models import get_layers

    base = base_of(model)
    blocks = get_layers(base)
    names = sorted({type(b).__name__ for b in blocks})
    where = next((n for n, m in base.named_modules() if m is blocks), "?")  # the blocks' path in the module tree
    rec = {"n_blocks": len(blocks), "n_layers": text_layers(base), "block_class": names, "path": where,
           "path_ok": (len(blocks) == text_layers(base) and not any("Vision" in n for n in names)
                       and "vision" not in where.lower())}
    assert rec["path_ok"], rec
    return rec


def _reader_context(model, reader: str):
    if reader not in READERS:
        raise ValueError(f"reader is one of {READERS}, not {reader!r}")
    if reader == "frozen" and hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


def reply_span(tok, history: list[dict]) -> tuple[list[int], int, bool]:
    """(the full rendering's ids, where the final reply starts, whether the prompt rendered as a prefix)."""
    from anatta.models import chat_ids

    if not history or history[-1]["role"] != "assistant":
        raise ValueError("a history ends with the assistant's final reply")
    full = chat_ids(tok, history, False)[0].tolist()
    prompt = chat_ids(tok, history[:-1])[0].tolist()
    start = min(len(prompt), max(len(full) - 1, 0))
    return full, start, full[:len(prompt)] == prompt


def _per_item(x, n: int, what: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"{what}: one value or one per history ({n}), got shape {a.shape}")
    return a


def _span_items(tok, histories) -> list[tuple[list[int], int, bool]]:
    return [reply_span(tok, h) for h in histories]


def id_items(items: list) -> list[tuple[list[int], int, bool]]:
    """(prompt ids, completion ids) pairs, as the trainer holds them (turns[i][-1]), in reply_span's form."""
    out = []
    for p, c in items:
        p, c = [int(x) for x in p], [int(x) for x in c]
        out.append((p + c, len(p), True))
    return out


def _forward(model, spans: list, layers: list[int], reader: str, batch: int, special: set, pad: int, on_batch) -> None:
    """Right-padded batches of the full renderings through the base module (adapter disabled for the frozen reader),
    capture() on `layers`; on_batch(store, keep [b, T] bool on the reply's non-special tokens, rows: the spans'
    indices in this batch). Spans go shortest first, so a batch pads little; logits only for the last position (the
    block outputs are what is read, never the full-vocabulary logits)."""
    import torch

    from anatta.probe import capture

    base = base_of(model)
    dev = next(base.parameters()).device
    order = sorted(range(len(spans)), key=lambda i: len(spans[i][0]))
    was_training = base.training
    base.eval()
    try:
        with torch.no_grad(), _reader_context(model, reader):
            for b0 in range(0, len(order), batch):
                rows = order[b0:b0 + batch]
                chunk = [spans[i] for i in rows]
                width = max(len(f) for f, _, _ in chunk)
                ids = torch.full((len(chunk), width), pad, dtype=torch.long)
                att = torch.zeros((len(chunk), width), dtype=torch.long)
                keep = torch.zeros((len(chunk), width), dtype=torch.bool)
                for j, (full, start, _) in enumerate(chunk):  # right padding: positions match the unpadded run
                    ids[j, :len(full)] = torch.tensor(full)
                    att[j, :len(full)] = 1
                    for t in range(start, len(full)):
                        keep[j, t] = full[t] not in special
                    if not keep[j].any():  # a reply of special tokens only: its last token stands in
                        keep[j, len(full) - 1] = True
                ids, att, keep = ids.to(dev), att.to(dev), keep.to(dev)
                with capture(base, layers) as store:
                    base(input_ids=ids, attention_mask=att, logits_to_keep=1)
                on_batch(store, keep, rows)
    finally:
        base.train(was_training)


def _special_pad(tok) -> tuple[set, int]:
    special = set(getattr(tok, "all_special_ids", []) or [])
    return special, (tok.pad_token_id if tok.pad_token_id is not None else 0)


def _scores(model, spans, layer, unit, mu, sd, reader, batch, special, pad) -> dict:
    import torch

    dev = next(base_of(model).parameters()).device
    U = np.atleast_2d(np.asarray(unit, dtype=np.float32))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    Ut = torch.as_tensor(U).to(dev)
    s = np.full((len(spans), len(U)), np.nan)
    ntok = np.zeros(len(spans), dtype=int)

    def on_batch(store, keep, rows):
        proj = store[layer].float() @ Ut.T  # [b, T, k]
        proj = proj.masked_fill(~keep[..., None], float("-inf"))
        s[rows] = proj.max(1).values.cpu().numpy()
        ntok[rows] = keep.sum(1).cpu().numpy()

    _forward(model, spans, [layer], reader, batch, special, pad, on_batch)
    if np.ndim(unit) == 1:
        s = s[:, 0]
    z = None
    if mu is not None and sd is not None:
        m, d = _per_item(mu, len(s), "mu"), _per_item(sd, len(s), "sd")
        if (d <= 0).any():
            raise ValueError("sd must be positive")
        z = (s - m) / d if s.ndim == 1 else (s - m[:, None]) / d[:, None]
    return {"s": s, "z": z, "n_tokens": ntok.tolist(), "prefix_ok": [ok for _, _, ok in spans], "reader": reader,
            "layer": layer}


def max_token_scores(model, tok, histories: list[list[dict]], layer: int, unit, mu=None, sd=None,
                     reader: str = "frozen", batch: int = 8) -> dict:
    """The score of each history's final reply: s (max projection on `unit` at block `layer`'s output) and, given
    mu and sd (a scalar, or one value per history, e.g. picked by twin kind), z = (s - mu) / sd. `unit` is one
    direction [d] or several [k, d] (each normalised; s and z are then [n, k], one forward pass for all). Returns
    {"s", "z" (None without mu and sd), "n_tokens", "prefix_ok", "reader", "layer"}."""
    if reader not in READERS:
        raise ValueError(f"reader is one of {READERS}, not {reader!r}")
    special, pad = _special_pad(tok)
    return _scores(model, _span_items(tok, histories), layer, unit, mu, sd, reader, batch, special, pad)


def max_token_scores_ids(model, items: list, layer: int, unit, special_ids, pad_id: int, mu=None, sd=None,
                         reader: str = "frozen", batch: int = 8) -> dict:
    """max_token_scores on (prompt ids, completion ids) pairs, the exact ids generation saw (the trainer's
    turns[i][-1]), with no re-rendering: the completion ids are the reply."""
    if reader not in READERS:
        raise ValueError(f"reader is one of {READERS}, not {reader!r}")
    return _scores(model, id_items(items), layer, unit, mu, sd, reader, batch, set(int(x) for x in special_ids),
                   pad_id if pad_id is not None else 0)


def reply_means(model, tok, histories: list[list[dict]], layers: list[int], reader: str = "frozen",
                batch: int = 8) -> np.ndarray:
    """The mean of each block output over the final reply's non-special tokens, [n, len(layers), d] float32: what
    the difference of means is fitted on (lab/explore/shaping/fit_direction.py)."""
    if reader not in READERS:
        raise ValueError(f"reader is one of {READERS}, not {reader!r}")
    special, pad = _special_pad(tok)
    spans = _span_items(tok, histories)
    out: list = [None] * len(spans)

    def on_batch(store, keep, rows):
        w = keep[..., None].float()
        for j, i in enumerate(rows):
            out[i] = np.stack([((store[x][j].float() * w[j]).sum(0) / w[j].sum()).cpu().numpy()
                                    for x in layers])

    _forward(model, spans, list(layers), reader, batch, special, pad, on_batch)
    return np.stack(out).astype(np.float32)


def lora_b_params(model) -> list:
    """The adapter's B matrices (PEFT names them lora_B): with every one zeroed the adapter adds exactly nothing."""
    return [p for n, p in model.named_parameters() if "lora_B" in n]


def reader_check(model, items: list, layer: int, unit, special_ids, pad_id: int, tol: float = 1e-2) -> dict:
    """The run-start check of the readers on a PEFT model, before any training or scoring. On `items` (one is enough):
    the frozen reader (disable_adapter) must equal the adapter-on score with every lora_B zeroed (the base's own
    forward, reached without disable_adapter), and when the adapter is non-trivial (some lora_B non-zero) the policy
    reader must differ from it; capture_check runs on the wrapped model. The B matrices are restored however the
    check ends. Raises AssertionError with the record otherwise."""
    import torch

    if not hasattr(model, "disable_adapter"):
        raise AssertionError("reader_check needs a PEFT-wrapped model (disable_adapter)")
    rec = {"capture_check": capture_check(model)}
    Bs = lora_b_params(model)
    if not Bs:
        raise AssertionError("the model has no lora_B parameters: is the adapter attached?")

    def score(reader):
        return max_token_scores_ids(model, items, layer, unit, special_ids, pad_id, reader=reader)["s"]

    frozen, policy = score("frozen"), score("policy")
    saved = [b.detach().clone() for b in Bs]
    try:
        with torch.no_grad():
            for b in Bs:
                b.zero_()
        zeroed = score("policy")
    finally:
        with torch.no_grad():
            for b, v in zip(Bs, saved):
                b.copy_(v)
    tol_abs = tol * (1.0 + float(np.abs(zeroed).max()))
    rec |= {"nontrivial": any(bool(v.abs().max() > 0) for v in saved),
            "frozen_vs_base": float(np.abs(frozen - zeroed).max()),
            "policy_vs_base": float(np.abs(policy - zeroed).max()), "tol": tol_abs,
            "frozen_s": [float(x) for x in np.ravel(frozen)]}
    rec["frozen_ok"] = rec["frozen_vs_base"] <= tol_abs
    rec["policy_ok"] = (not rec["nontrivial"]) or rec["policy_vs_base"] > tol_abs
    assert rec["frozen_ok"] and rec["policy_ok"], rec
    return rec


def check_item(tok) -> tuple[list[int], list[int]]:
    """One (prompt ids, completion ids) item of our own text for reader_check: a plain coding request and a plain
    reply, rendered with the model's own chat template."""
    from anatta.models import chat_ids

    prompt = chat_ids(tok, [{"role": "user", "content": "Write a Python function that adds two numbers."}])[0]
    full = chat_ids(tok, [{"role": "user", "content": "Write a Python function that adds two numbers."},
                          {"role": "assistant", "content": "def add(a, b):\n    return a + b"}], False)[0]
    p, f = prompt.tolist(), full.tolist()
    return p, f[len(p):] if f[:len(p)] == p else f[-8:]
