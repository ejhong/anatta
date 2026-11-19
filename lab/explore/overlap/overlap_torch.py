"""Leg 2's torch parts: the overlap loss of Carauleanu and colleagues (2024) as the lab reads their paper, a plain
next-token loss on a model's own replies, a small training loop and batched generation. Used by modal_overlap.py in
a container and by the offline tests on a tiny model. Claude (Fable 5.2, an AI model) wrote this file on
19 Nov 2026, from the paper's description (its section 3.1.1 and appendix A.1.2); the authors published no code.

How the paper is read where it leaves room. The loss is the mean squared difference between the output of the
attention block's output projection (self_attn.o_proj) at one layer on a self prompt and on its paired other prompt,
over every token of the chat-formatted prompt with its generation prompt. The paper's pairs differ in three words
(You and Bob, the verb's form, yourself and Bob); where the two members have the same number of tokens the
comparison is token by token, and where they do not the two are lined up from the end (the share of such pairs is
recorded for every training run). Both prompts carry gradient. LoRA on the query and value projections; AdamW; no clipping."""

from __future__ import annotations

import random

import torch

TARGETS = r"^(?!.*(vision|projector)).*\.(q_proj|v_proj)$"  # the paper's two projections; the lab's rule keeps a vision tower out


class _Stop(Exception):
    """Raised by the hook once the wanted layer's output is in hand, so the layers above it are not run."""


def attn_out(base_model, layer: int):
    from anatta.models import get_attn, get_layers

    return get_attn(get_layers(base_model)[layer]).o_proj


def encode(tok, texts: list[str], device):
    """Chat-formatted user prompts with the generation prompt, left-padded (so that the last tokens line up)."""
    chats = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t in texts]
    side, tok.padding_side = tok.padding_side, "left"
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    tok.padding_side = side
    return enc


def layer_output(model, base_model, layer: int, enc) -> torch.Tensor:
    """The output of self_attn.o_proj at `layer` for a batch, with gradient, without running the layers above."""
    got = {}

    def hook(_m, _i, out):
        got["x"] = out
        raise _Stop

    h = attn_out(base_model, layer).register_forward_hook(hook)
    try:
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    except _Stop:
        pass
    finally:
        h.remove()
    return got["x"]


def overlap_loss(model, base_model, tok, layer: int, batch: list[tuple[str, str]]) -> tuple[torch.Tensor, int]:
    """(the loss, how many of the batch's pairs differ in length). The mean squared difference between the two
    members' layer outputs over the tokens both have, lined up from the end."""
    dev = next(base_model.parameters()).device
    ea, eb = encode(tok, [a for a, _ in batch], dev), encode(tok, [b for _, b in batch], dev)
    xa, xb = layer_output(model, base_model, layer, ea).float(), layer_output(model, base_model, layer, eb).float()
    t = min(xa.shape[1], xb.shape[1])
    mask = (ea["attention_mask"][:, -t:] * eb["attention_mask"][:, -t:]).float()
    per_token = ((xa[:, -t:] - xb[:, -t:]) ** 2).mean(-1)
    uneven = int((ea["attention_mask"].sum(1) != eb["attention_mask"].sum(1)).sum())
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0), uneven


def uneven(tok, data: list) -> int:
    """How many pairs have members of different lengths once chat-formatted."""
    n = lambda t: len(tok(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False),  # noqa: E731
                          add_special_tokens=False)["input_ids"])
    return sum(n(a) != n(b) for a, b in data)


def sft_loss(model, base_model, tok, batch: list[tuple[str, str]]) -> torch.Tensor:
    """The mean next-token loss on each reply's own tokens (and its end token), given its chat-formatted prompt."""
    dev = next(base_model.parameters()).device
    rows = []
    for prompt, reply in batch:
        head = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        h = tok(head, add_special_tokens=False)["input_ids"]
        r = tok(reply.strip(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        rows.append((h + r, [-100] * len(h) + r))
    width = max(len(i) for i, _ in rows)
    ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    for k, (i, lab) in enumerate(rows):  # right-padded: the labels mark what counts
        ids[k, :len(i)], labels[k, :len(i)], attn[k, :len(i)] = torch.tensor(i), torch.tensor(lab), 1
    ids, labels, attn = ids.to(dev), labels.to(dev), attn.to(dev)
    logits = model(input_ids=ids, attention_mask=attn).logits.float()
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)


def add_lora(base_model, seed: int, rank: int = 8, alpha: int = 32, dropout: float = 0.2):
    """A fresh LoRA on the query and value projections (the paper's setting for its 7B model), its weights in float32
    so that small steps are not lost in the base model's 16-bit numbers."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)  # the seed sets the adapter's first weights
    model = get_peft_model(base_model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS,
                                                  task_type="CAUSAL_LM"))
    return floats(model)


def floats(model):
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def train(model, base_model, tok, *, loss: str, data: list, layer: int | None, seed: int, lr: float, batch: int, epochs: int = 0,
          steps: int = 0, looks: tuple = (), at_look=None) -> list[dict]:
    """Train the adapter on `data` with the overlap loss (pairs) or the plain loss (prompt and reply). Either a
    number of epochs or a number of steps; `at_look(step)` is called after each step named in `looks`. Returns the
    log: one row a step."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng, log, n = random.Random(seed), [], 0
    total = steps or epochs * -(-len(data) // batch)
    while n < total:
        order = list(range(len(data)))
        rng.shuffle(order)
        for b in range(0, len(order), batch):
            if n >= total:
                break
            part = [tuple(data[i]) for i in order[b:b + batch]]
            model.train()
            opt.zero_grad(set_to_none=True)
            if loss == "overlap":
                value, uneven = overlap_loss(model, base_model, tok, layer, part)
            else:
                value, uneven = sft_loss(model, base_model, tok, part), 0
            value.backward()
            opt.step()
            n += 1
            log.append({"step": n, "loss": float(value.detach()), "uneven": uneven})
            if at_look is not None and n in looks:
                at_look(n)
    model.eval()
    return log


@torch.no_grad()
def mean_overlap(model, base_model, tok, layer: int, data: list[tuple[str, str]], batch: int = 16) -> float:
    """The overlap loss over pairs the model is not being trained on, with the adapter's dropout off."""
    model.eval()
    vals = [float(overlap_loss(model, base_model, tok, layer, [tuple(x) for x in data[k:k + batch]])[0]) for k in range(0, len(data), batch)]
    return sum(vals) / max(1, len(vals))


@torch.no_grad()
def generate_sets(model, tok, sets: list[dict], batch: int = 48) -> dict:
    """{set name: replies in the order asked}. A set is {"name", "chats", "max_new", "temperature", "seed"}; chats
    are answered longest first in batches (anatta.probe.generate: left padding, greedy at temperature 0)."""
    from anatta.probe import generate

    model.eval()
    out = {}
    for s in sets:
        chats = s["chats"]
        order = sorted(range(len(chats)), key=lambda i: -sum(len(m["content"]) for m in chats[i]))
        texts = [""] * len(chats)
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            got = generate(model, tok, [chats[i] for i in idx], int(s["max_new"]), temperature=float(s.get("temperature", 0.0)),
                           seed=int(s.get("seed", 0)) * 1000 + k)
            for i, t in zip(idx, got, strict=True):
                texts[i] = t.strip()
        out[s["name"]] = texts
    return out
