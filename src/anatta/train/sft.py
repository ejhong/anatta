"""The supervised fine-tuning loop shared by the training apps: items are (prompt ids, completion ids) pairs, packed
into padded micro-batches (anatta.player.pack: prompts left-padded, completions right-padded) with the loss on the
completion tokens only (the completion mask), so a prompt never carries loss.

`batch` items per optimizer step in micro-batches of `micro`, each step's items sorted by length; the loss is the mean
negative log-likelihood per completion token in the batch (each micro-batch's summed NLL over the batch's token
count, so the gradients add up to the batch mean); AdamW's step after clipping the gradient norm at 1.0. One record
per optimizer step goes to `log`.

No Modal here and no model loading: a caller loads the model, builds the optimizer over its trainable parameters and
wraps the call in anatta.dial.intervene() when a cut should stay on (forward and backward both inside the block, so
gradient checkpointing's recomputed forward keeps it).

linear_lr is a schedule for train(): transformers' get_linear_schedule_with_warmup, step for step (the rule TRL's and
Unsloth's trainers use, so Kaczér et al.'s recipe for R1b). It differs from anatta.evals.em.lr_at, em1's schedule,
only in the warmup: here step 0 runs at lr 0 and step `warmup` reaches lr, there the first step runs at lr / warmup.
mean_nll scores a held-out set after (or before) training: the mean negative log-likelihood per completion token, the
same per-token loss train() logs as "nll".
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from ..player import pack, peak_gb, reset_peak, token_logps


def train(model, items: list, pad_id: int, opt, params: list, rng, *, epochs: int, batch: int, micro: int,
          log: Callable[[dict], None] | None = None, schedule: Callable[[int], float] | None = None,
          print_every: int = 10, on_epoch: Callable[[int], None] | None = None) -> int:
    """Train `model` on `items` for `epochs` passes, the data order drawn from `rng` (a random.Random) at the start of
    every epoch. `schedule` maps the 0-based optimizer step to its learning rate, set on every parameter group before
    the step and recorded as "lr" (None leaves the optimizer's rate alone). Every record goes to `log`; every
    `print_every`-th is printed (steps 1, 1 + print_every, ...). `on_epoch(k)` runs after epoch k (1-based), still
    inside the caller's context. Returns the number of optimizer steps."""
    import torch

    dev = next(model.parameters()).device
    n_opt, t_start = 0, time.time()
    for epoch in range(epochs):
        order = list(range(len(items)))
        rng.shuffle(order)
        for b in range(0, len(order), batch):
            t0 = time.time()
            reset_peak()
            model.train()
            opt.zero_grad(set_to_none=True)
            if schedule is not None:
                lr_now = schedule(n_opt)
                for g in opt.param_groups:
                    g["lr"] = lr_now
            idx = sorted(order[b:b + batch], key=lambda i: len(items[i][0]) + len(items[i][1]))
            ntok, nll = sum(len(items[i][1]) for i in idx), 0.0
            for k in range(0, len(idx), micro):
                chunk = [items[i] for i in idx[k:k + micro]]
                full, attn, cm, wc = pack(chunk, pad_id, dev)
                lp = token_logps(model, full, attn, wc)
                loss = -(lp * cm).sum() / ntok
                loss.backward()
                nll += float(loss)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            n_opt += 1
            rec = {"epoch": epoch + 1, "opt_step": n_opt}
            if schedule is not None:
                rec["lr"] = lr_now
            rec.update({"items": len(idx), "tokens": ntok, "nll": nll, "grad_norm": grad_norm,
                        "s": round(time.time() - t0, 1), "peak_gb": peak_gb(), "elapsed": round(time.time() - t_start)})
            if log is not None:
                log(rec)
            if n_opt % print_every == 1:
                print(json.dumps(rec), flush=True)
        if on_epoch is not None:
            on_epoch(epoch + 1)
    return n_opt


def linear_lr(step: int, total: int, lr: float, warmup: int = 0) -> float:
    """The learning rate of optimizer step `step` (0-based) of `total`: a linear warmup from 0 over `warmup` steps,
    then a linear decay to 0 at `total` (transformers' get_linear_schedule_with_warmup lr_lambda, times lr)."""
    if step < warmup:
        return lr * step / max(1, warmup)
    return lr * max(0.0, (total - step) / max(1, total - warmup))


def mean_nll(model, items: list, pad_id: int, micro: int = 4) -> dict:
    """The mean negative log-likelihood per completion token over `items` ((prompt ids, completion ids), packed as
    train() packs them, shortest first), in eval mode with no gradient; the model's train/eval mode is restored.
    Returns {"nll", "tokens", "items"}."""
    import torch

    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    total, ntok = 0.0, 0
    try:
        with torch.no_grad():
            for k in range(0, len(order), micro):
                chunk = [items[i] for i in order[k:k + micro]]
                full, attn, cm, wc = pack(chunk, pad_id, dev)
                total += float(-(token_logps(model, full, attn, wc) * cm).sum())
                ntok += int(cm.sum())
    finally:
        model.train(was_training)
    return {"nll": total / ntok if ntok else None, "tokens": ntok, "items": len(items)}
