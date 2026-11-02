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
