"""The second step's training, as it runs in a GPU container (lab/explore/msm/modal_msm2.py mounts this file).
Claude (Fable 5.2, an AI model) wrote this file, 22 Nov 2026.

One job trains one LoRA adapter on the base model in phases, each phase one pass over its data:
  - a `docs` phase: next-token prediction over whole documents, as text is learned in pretraining;
  - a `chat` phase: conversations in the model's chat format, with the loss on the assistant's turns only.
Settings the study states (its appendix B.4): rank 64, alpha 128, every attention and MLP projection, one epoch,
AdamW at 1e-4 with a cosine schedule after 5% warmup, weight decay 0.01, sequences cut at 8,192 tokens. Settings it
does not state and that are chosen here: a step holds at most `batch_tokens` padded tokens (sequences of like length
together); gradients are clipped at norm 1; the same adapter is carried from the documents into the practice with a
fresh optimizer and schedule; a document ends with the tokenizer's end-of-text token. Amendment 4 (23 Nov): a job may
skip the documents and start its practice from a documents adapter saved earlier (load_lora), which is the same as
carrying it within one job, since the practice's optimizer and schedule are fresh either way.
"""

from __future__ import annotations

import math
import random
import time

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    return tok, model


def add_lora(model, seed: int, rank: int, alpha: int, dropout: float):
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    m = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS,
                                         bias="none", task_type="CAUSAL_LM"))
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()  # with checkpointing, a block's input must carry a gradient for its LoRA to learn
    return m


def load_lora(model, path: str, rank: int, alpha: int, dropout: float):
    """Amendment 4: a saved adapter (a documents phase's) as the trainable adapter, in place of add_lora. Its settings
    must be the job's, and after loading its weights must be the file's, tensor for tensor and in the file's dtype
    (float32, as add_lora's are). No seed is used: every weight comes from the file. Returns (model, what was loaded)."""
    import hashlib
    from pathlib import Path

    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load as st_load

    raw = (Path(path) / "adapter_model.safetensors").read_bytes()
    saved = st_load(raw)
    m = PeftModel.from_pretrained(model, path, is_trainable=True)
    c = m.peft_config["default"]
    got = {"rank": c.r, "alpha": c.lora_alpha, "dropout": c.lora_dropout, "targets": sorted(c.target_modules), "bias": c.bias,
           "rslora": bool(getattr(c, "use_rslora", False)), "dora": bool(getattr(c, "use_dora", False))}
    want = {"rank": rank, "alpha": alpha, "dropout": dropout, "targets": sorted(TARGETS), "bias": "none", "rslora": False, "dora": False}
    if got != want:
        raise RuntimeError(f"{path}: the adapter's settings {got} are not the job's {want}")
    now = get_peft_model_state_dict(m)
    if set(now) != set(saved):
        raise RuntimeError(f"{path}: the loaded adapter's tensors are not the file's ({len(now)} against {len(saved)})")
    for k, v in saved.items():
        x = now[k].detach()
        if x.dtype != v.dtype or x.shape != v.shape or not torch.equal(x.cpu(), v):
            raise RuntimeError(f"{path}: {k} is not the file's after loading")
    if sum(p.numel() for p in m.parameters() if p.requires_grad) != sum(v.numel() for v in saved.values()):
        raise RuntimeError(f"{path}: the trainable weights are not the adapter's")
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()  # as add_lora
    return m, {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "tensors": len(saved),
               "dtypes": sorted({str(v.dtype) for v in saved.values()}), "config": got, "weights_as_saved": True}


def encode_doc(tok, text: str, max_len: int) -> tuple[list[int], list[int]]:
    end = tok.convert_tokens_to_ids("<|endoftext|>")
    ids = tok(text, add_special_tokens=False)["input_ids"][: max_len - 1] + [end]
    return ids, list(ids)


def encode_chat(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int], bool]:
    """(ids, labels, exact): the whole conversation in the chat format; labels are the assistant's turns (and the
    end-of-turn token after each), -100 elsewhere. The turns are found in the rendered text, by character, and a
    token is labelled when any of it lies inside a turn; so a turn that begins with a newline, which the tokenizer
    joins to the newline the format writes before it, is still marked (the first pilot of 22 Nov marked by token
    prefixes and missed half the practice set that way). exact is False when the text of a prefix of the
    conversation is not a prefix of the whole text, or the ids differ from the template's own; then every token is
    given a label and the caller counts it."""
    def text_of(msgs, gen):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

    text = text_of(messages, False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    full, spans = list(enc["input_ids"]), list(enc["offset_mapping"])
    own = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    exact = full == list(own if isinstance(own, list) else own["input_ids"])
    turns = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start, stop = text_of(messages[:i], True), text_of(messages[: i + 1], False)
        if not (text.startswith(start) and text.startswith(stop) and len(stop) > len(start)):
            exact = False
            break
        turns.append((len(start), len(stop.rstrip("\n"))))  # up to and with the end-of-turn token, not the newline after it
    if not exact:
        return full[:max_len], list(full)[:max_len], False
    labels = [tid if any(b > lo and a < hi for lo, hi in turns) else -100 for tid, (a, b) in zip(full, spans, strict=True)]
    return full[:max_len], labels[:max_len], True


def batches(lengths: list[int], batch_tokens: int, seed: int) -> list[list[int]]:
    """Row numbers in steps of at most batch_tokens padded tokens: rows of like length together, the steps in a seeded
    order. Every row is in exactly one step."""
    rng = random.Random(seed)
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], rng.random()))
    out, cur, longest = [], [], 0
    for i in order:
        longest_new = max(longest, lengths[i])
        if cur and longest_new * (len(cur) + 1) > batch_tokens:
            out.append(cur)
            cur, longest_new = [], lengths[i]
        cur.append(i)
        longest = longest_new
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def lr_at(step: int, total: int, lr: float, warmup: float) -> float:
    w = max(1, int(round(warmup * total)))
    if step < w:
        return lr * (step + 1) / w
    return 0.5 * lr * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def loss_backward(logits, labels, chunk: int = 4096) -> float:
    """The model's own loss (mean cross-entropy of each next token over the labelled positions, in float32) and its
    backward pass, without a float32 copy of every logit: the loss is worked out a few thousand positions at a
    time, its gradient with respect to the logits is gathered, and one backward carries it into the model. The pilot
    of 22 Nov stood at 123 of 140 GiB with the plain form, mostly these copies. Returns the loss."""
    import torch
    import torch.nn.functional as F

    b, t, v = logits.shape
    target = torch.full_like(labels, -100)
    target[:, :-1] = labels[:, 1:]  # position i predicts token i+1; the last position predicts nothing
    n = int((target != -100).sum())
    flat, tflat = logits.view(b * t, v), target.view(-1)
    grad, total = torch.empty_like(flat), 0.0
    for i in range(0, b * t, chunk):
        x = flat[i: i + chunk].detach().float().requires_grad_(True)
        part = F.cross_entropy(x, tflat[i: i + chunk], ignore_index=-100, reduction="sum") / n
        (g,) = torch.autograd.grad(part, x)
        grad[i: i + chunk] = g.to(grad.dtype)
        total += float(part)
    flat.backward(grad)
    return total


def run_phase(model, rows: list[tuple[list[int], list[int]]], *, pad: int, seed: int, lr: float, weight_decay: float,
              warmup: float, max_grad_norm: float, batch_tokens: int, max_steps: int | None = None, log_every: int = 10,
              passes: int = 1, each: bool = False) -> dict:
    """passes: how many times every row is trained on (the contents round's neutral arm: 2). Each pass is grouped
    and ordered by its own seed (seed + pass), and one schedule runs over all of them; 1 is the code as it was.
    max_steps stops early on the full-length schedule (the warm-up and cosine are counted from every step).
    each: also keep every step's own loss (amendment 4's path check)."""
    import torch

    lengths = [len(r[0]) for r in rows]
    steps = [st for k in range(passes) for st in batches(lengths, batch_tokens, seed + k)]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    model.train()
    log, t0, seen, run, n_run, per_step = [], time.time(), 0, 0.0, 0, []
    todo = steps if max_steps is None else steps[:max_steps]
    for n, step in enumerate(todo):
        width = max(len(rows[i][0]) for i in step)
        ids = torch.full((len(step), width), pad, dtype=torch.long)
        lab = torch.full((len(step), width), -100, dtype=torch.long)
        att = torch.zeros((len(step), width), dtype=torch.long)
        for b, i in enumerate(step):
            x, y = rows[i]
            ids[b, : len(x)] = torch.tensor(x)
            lab[b, : len(y)] = torch.tensor(y)
            att[b, : len(x)] = 1
        for g in opt.param_groups:
            g["lr"] = lr_at(n, len(steps), lr, warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
        loss = loss_backward(logits, lab.cuda())
        del logits
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        seen += int(att.sum())
        run, n_run = run + float(loss), n_run + 1
        if each:
            per_step.append(round(float(loss), 6))
        if (n + 1) % log_every == 0 or n + 1 == len(todo):
            log.append({"step": n + 1, "loss": round(run / n_run, 4), "tokens": seen, "seconds": round(time.time() - t0, 1)})
            run, n_run = 0.0, 0
    secs = time.time() - t0
    return {"steps_done": len(todo), "steps_in_epoch": len(steps), "tokens_done": seen,
            "tokens_in_epoch": passes * sum(lengths), "passes": passes, "seconds": round(secs, 1),
            "tokens_per_second": round(seen / secs, 1) if secs else None, "log": log,
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)} | ({"each": per_step} if each else {})
