"""Leg 2's torch parts: the overlap loss of Carauleanu and colleagues (2024) as the lab reads their paper, a plain
next-token loss on a model's own replies, a small training loop and batched generation. Used by modal_overlap.py in
a container and by the offline tests on a tiny model. Claude (Fable 5.2, an AI model) wrote this file on
19 Nov 2026, from the paper's description (its section 3.1.1 and appendix A.1.2); the authors published no code.

How the paper is read where it leaves room. The loss is the mean squared difference between the output of the
attention block's output projection (self_attn.o_proj) at one layer on a self prompt and on its paired other prompt,
over every token of the chat-formatted prompt with its generation prompt. The paper's pairs differ in three words
(You and Bob, the verb's form, yourself and Bob); where the two members have the same number of tokens the
comparison is token by token, and where they do not the two are lined up from the end (the share of such pairs is
recorded for every training run). Both prompts carry gradient. LoRA on the query and value projections; AdamW; no clipping.

Added later on 19 Nov 2026 for the view-first step (lab/explore/viewfirst/run_view.py), and used by nothing before
it: a plain next-token loss on raw documents (text_loss, mean_text_loss; train's loss "text"), a fresh adapter on
top of one trained in the same process (stack_fresh), one adapter saved by itself and saved adapters loaded one on
top of another (save_adapter, load_stack), and a job that goes through phases with looks on the way (run_phases)."""

from __future__ import annotations

import hashlib
import json
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


def _sft_rows(tok, batch: list[tuple[str, str]], reply_tokens: int | None) -> list[tuple[list[int], list[int]]]:
    """(ids, labels) of each pair: its chat-formatted prompt, unlabelled, then its reply's own tokens. A reply longer
    than reply_tokens is cut there and gets no end token (the model is not taught to stop in the middle of a text); a
    reply that is not cut gets one. With reply_tokens None nothing is cut."""
    rows = []
    for prompt, reply in batch:
        head = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        h = tok(head, add_special_tokens=False)["input_ids"]
        r = tok(reply.strip(), add_special_tokens=False)["input_ids"]
        r = r[:int(reply_tokens)] if reply_tokens and len(r) > int(reply_tokens) else r + [tok.eos_token_id]
        rows.append((h + r, [-100] * len(h) + r))
    return rows


def _sft_sums(model, base_model, tok, batch: list[tuple[str, str]], reply_tokens: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """(each pair's next-token loss summed over its reply's labelled tokens, how many such tokens), with gradient."""
    dev = next(base_model.parameters()).device
    rows = _sft_rows(tok, batch, reply_tokens)
    width = max(len(i) for i, _ in rows)
    ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    for k, (i, lab) in enumerate(rows):
        ids[k, :len(i)], labels[k, :len(i)], attn[k, :len(i)] = torch.tensor(i), torch.tensor(lab), 1
    ids, labels, attn = ids.to(dev), labels.to(dev), attn.to(dev)
    logits = model(input_ids=ids, attention_mask=attn).logits.float()
    per = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100,
                                            reduction="none").view(len(rows), width - 1)
    mask = (labels[:, 1:] != -100).float()
    return (per * mask).sum(1), mask.sum(1)


def sft_loss(model, base_model, tok, batch: list[tuple[str, str]], per_example: bool = False, reply_tokens: int | None = None) -> torch.Tensor:
    """The mean next-token loss on each reply's own tokens (and its end token), given its chat-formatted prompt.
    Two options, off unless asked (the own-advice step, lab/explore/verdict/run_verdict.py; with both off this is the
    loss every earlier step trained with, computed as it was): per_example, each pair weighs the same whatever its
    length (the mean of each pair's own mean over its tokens); reply_tokens, a reply is trained on up to that many
    tokens and gets no end token where it is cut (_sft_rows)."""
    if per_example or reply_tokens:
        sums, counts = _sft_sums(model, base_model, tok, batch, reply_tokens)
        return (sums / counts.clamp_min(1.0)).mean() if per_example else sums.sum() / counts.sum().clamp_min(1.0)
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


@torch.no_grad()
def mean_sft_loss(model, base_model, tok, data: list, per_example: bool = False, reply_tokens: int | None = None, batch: int = 8) -> float:
    """The plain loss over a whole set of pairs, with the adapter's dropout off and no step taken: per_example, the mean
    over pairs of each pair's own mean; else nats a token, each token counting once. The batches' sizes do not matter
    either way. (Point 18 of the own-advice step: each training job's loss on its whole training set, before its first
    step and after its last.)"""
    model.eval()
    means, total, n = [], 0.0, 0.0
    for k in range(0, len(data), batch):
        sums, counts = _sft_sums(model, base_model, tok, [tuple(x) for x in data[k:k + batch]], reply_tokens)
        means += (sums / counts.clamp_min(1.0)).tolist()
        total, n = total + float(sums.sum()), n + float(counts.sum())
    return sum(means) / max(1, len(means)) if per_example else total / max(1.0, n)


def text_rows(tok, texts: list[str], max_tokens: int) -> list[list[int]]:
    """Each raw text as token ids: the tokenizer's start token (if it has one), then the text's own tokens, the whole
    cut at max_tokens. No chat template: these are documents (the view-first step, lab/explore/viewfirst)."""
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return [(bos + tok(t, add_special_tokens=False)["input_ids"])[:max_tokens] for t in texts]


def _text_nll(model, tok, texts: list[str], max_tokens: int) -> tuple[torch.Tensor, int]:
    """(the next-token loss summed over every token of each text that has a token before it, how many such tokens).
    Right-padded; padding is not counted."""
    dev = next(model.parameters()).device
    rows = text_rows(tok, texts, max_tokens)
    width = max(1, max((len(r) for r in rows), default=1))
    ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    for k, r in enumerate(rows):
        ids[k, :len(r)], attn[k, :len(r)] = torch.tensor(r, dtype=torch.long), 1
    labels = ids.masked_fill(attn == 0, -100)  # by place, not by token: the padding token may be a real one elsewhere
    ids, labels, attn = ids.to(dev), labels.to(dev), attn.to(dev)
    logits = model(input_ids=ids, attention_mask=attn).logits.float()
    total = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100,
                                              reduction="sum")
    return total, int((labels[:, 1:] != -100).sum())


def text_loss(model, tok, texts: list[str], max_tokens: int) -> torch.Tensor:
    """The mean next-token loss over all tokens of raw texts (text_rows: no chat template), with gradient."""
    total, n = _text_nll(model, tok, texts, max_tokens)
    return total / max(1, n)


@torch.no_grad()
def mean_text_loss(model, tok, texts: list[str], max_tokens: int, batch: int = 8) -> float:
    """The same loss over texts the model is not being trained on, with the adapter's dropout off: nats a token, each
    token counting once (so the batches' sizes do not matter)."""
    model.eval()
    total, n = 0.0, 0
    for k in range(0, len(texts), batch):
        value, count = _text_nll(model, tok, texts[k:k + batch], max_tokens)
        total, n = total + float(value), n + count
    return total / max(1, n)


def add_lora(base_model, seed: int, rank: int = 8, alpha: int = 32, dropout: float = 0.2):
    """A fresh LoRA on the query and value projections (the paper's setting for its 7B model), its weights in float32
    so that small steps are not lost in the base model's 16-bit numbers."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)  # the seed sets the adapter's first weights
    model = get_peft_model(base_model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS,
                                                  task_type="CAUSAL_LM"))
    return floats(model)


def stack_lora(base_model, arm_dir: str, seed: int, rank: int = 8, alpha: int = 32, dropout: float = 0.2):
    """A saved arm kept exactly as it is, with a fresh adapter on top that alone is trained (stage 2's holding test:
    the two arms' own adapters differ in which layers they changed, so further training goes into a new adapter of
    one shape for both). The arm's adapter is frozen and its dropout is taken out, so that it acts in training as it
    does when asked."""
    from peft import LoraConfig, PeftModel

    model = PeftModel.from_pretrained(base_model, arm_dir, adapter_name="arm")
    torch.manual_seed(seed)
    model.add_adapter("fresh", LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.base_model.set_adapter(["arm", "fresh"])
    for module in model.modules():
        drop = getattr(module, "lora_dropout", None)
        if isinstance(drop, torch.nn.ModuleDict) and "arm" in drop:
            drop["arm"] = torch.nn.Identity()
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name and ".fresh" in name
    return floats(model)


def floats(model):
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def stack_fresh(model, seed: int, rank: int = 8, alpha: int = 32, dropout: float = 0.2, name: str = "fresh"):
    """A fresh adapter on top of the adapters a model already carries, which are kept exactly as they are: frozen, and
    their dropout taken out, so that they act in training as they do when asked (stack_lora's way, for an adapter
    trained in this same process). The fresh one alone is trained. The seed sets its first weights as add_lora's
    does: from the same seed the two start as the same adapter."""
    from peft import LoraConfig

    old = list(model.peft_config)
    if name in old:
        raise ValueError(f"the model already has an adapter called {name!r}")
    torch.manual_seed(seed)
    model.add_adapter(name, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.base_model.set_adapter([*old, name])
    for module in model.modules():
        drop = getattr(module, "lora_dropout", None)
        if isinstance(drop, torch.nn.ModuleDict):
            for kept in old:
                if kept in drop:
                    drop[kept] = torch.nn.Identity()
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n and f".{name}." in n
    return floats(model)


def save_adapter(model, name: str, out_dir: str) -> None:
    """One adapter of a model saved by itself, its files directly in out_dir. (PEFT puts an adapter that is not
    called "default" into a folder of its own name; here every adapter is found at the folder that was named.)"""
    import shutil
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, selected_adapters=[name])
        made = Path(tmp) if name == "default" else Path(tmp) / name
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for f in sorted(made.iterdir()):
            if f.is_file():
                shutil.copy2(f, Path(out_dir) / f.name)


def load_stack(base_model, dirs: list[str]):
    """Saved adapters put on the unchanged weights one on top of another, all of them active and none trainable: a
    saved documents' adapter, or that and the practice adapter trained on top of it, for asking later."""
    from pathlib import Path

    from peft import PeftModel

    if not dirs:
        raise ValueError("no adapter was named")
    for d in dirs:
        if not (Path(d) / "adapter_config.json").exists():
            raise FileNotFoundError(f"no adapter at {d}")
    names = [f"saved{k}" for k in range(len(dirs))]
    model = PeftModel.from_pretrained(base_model, dirs[0], adapter_name=names[0])
    for name, d in zip(names[1:], dirs[1:], strict=True):
        model.load_adapter(d, adapter_name=name)
    model.base_model.set_adapter(names)
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


def train(model, base_model, tok, *, loss: str, data: list, layer: int | None, seed: int, lr: float, batch: int, epochs: int = 0,
          steps: int = 0, looks: tuple = (), at_look=None, max_tokens: int = 512, per_example: bool = False, reply_tokens: int | None = None) -> list[dict]:
    """Train the adapter on `data` with the overlap loss (pairs), the plain loss (prompt and reply) or, with
    loss="text", the plain loss on raw documents (a list of strings, each cut at max_tokens). Either a number of
    epochs or a number of steps; `at_look(step)` is called after each step named in `looks`. per_example and
    reply_tokens are sft_loss's options for the plain loss (off unless asked). Returns the log: one row a step."""
    if loss != "sft" and (per_example or reply_tokens):
        raise ValueError("per_example and reply_tokens are options of the plain loss on pairs (sft) only")
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
            part = [data[i] if loss == "text" else tuple(data[i]) for i in order[b:b + batch]]  # a document is a string, not a pair
            model.train()
            opt.zero_grad(set_to_none=True)
            if loss == "overlap":
                value, uneven = overlap_loss(model, base_model, tok, layer, part)
            elif loss == "text":
                value, uneven = text_loss(model, tok, part, max_tokens), 0
            else:
                opts = {k: v for k, v in (("per_example", per_example), ("reply_tokens", reply_tokens)) if v}  # passed only when asked: the call is as it was otherwise
                value, uneven = sft_loss(model, base_model, tok, part, **opts), 0
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


def heads_of(tok, sets: list[dict], replies: dict) -> dict:
    """For each set that asks for it with "head_tokens": n (the own-advice step, lab/explore/verdict/run_verdict.py;
    other sets are left out, so nothing changes for them), each reply's length in tokens, whether it is longer than n
    tokens ("cut"), and, where it is, the text of its first n tokens ("head"; else None). A reply is tokenized as
    sft_loss tokenizes a target (its text without blanks around it, no special tokens), so the head is what a
    training on it up to n tokens would see. {set name: [{"tokens", "cut", "head"}] in the order of the replies}."""
    out = {}
    for s in sets:
        n = s.get("head_tokens")
        if not n:
            continue
        rows = []
        for text in replies[s["name"]]:
            ids = tok(text.strip(), add_special_tokens=False)["input_ids"]
            cut = len(ids) > int(n)
            rows.append({"tokens": len(ids), "cut": cut, "head": tok.decode(ids[:int(n)], skip_special_tokens=True) if cut else None})
        out[s["name"]] = rows
    return out


def run_phases(base_model, tok, job: dict, *, model=None, batch: int = 48, generate=None, save=None) -> dict:
    """A job that goes through phases, with looks on the way (the view-first step). `job` holds
      "lora"        {"seed", "rank", "alpha", "dropout"}: the settings of every adapter this job makes
      "phases"      [{"name", "loss": "text" or "sft", "data", "steps", "lr", "batch"; and where wanted "looks":
                    [steps], "stack": true, "save": a folder, "max_tokens"}], trained in this order
      "look_sets"   sets of prompts, answered at every look
      "held_texts"  {kind: [documents]}: their mean loss is read before any training and at every look, each
                    document cut at "held_max_tokens"
    The first phase makes an adapter (add_lora). A later phase trains that same adapter on, unless it says "stack":
    then whatever adapter there is stays as it is, frozen and with its dropout off, and a fresh adapter on top of it
    is trained (stack_fresh). A stacked phase with no adapter before it simply makes a fresh one. Every phase gets
    a new AdamW and the job's seed (train), so a phase's order of batches does not depend on what came before it.
    A look is taken after each step a phase names in "looks" (under "<phase>@<step>") and always when a phase ends
    (under "<phase>"; a look named at the phase's last step is that one, taken once). A phase's "save" keeps the
    adapter it trained, by itself, as it is when the phase ends: save(model, the adapter's name, the folder).
    Three options of a phase of pairs (loss "sft"), off unless the phase names them (the own-advice step,
    lab/explore/verdict/run_verdict.py): "per_example" and "reply_tokens" are sft_loss's; "whole_loss" reads the
    loss on the phase's whole training set, with the same options, before its first step and after its last (its
    record's "loss_before" and "loss_after"). A phase that names any of them records the options and both losses.
    With no phases the job only measures: the held texts' losses, under "before". `model` is a model that already
    carries saved adapters (load_stack); only a stacked phase can follow those. `generate` answers the look sets
    (generate_sets unless a test gives its own).
    Returns {"model", "log": one row a step with its phase, "looks", "held", "phases": what each phase trained and
    on what (counts and hashes, no text), "held_texts": counts and hashes, "lora"}."""
    generate = generate or generate_sets
    phases, look_sets = list(job.get("phases") or []), job.get("look_sets") or []
    held_texts, cut = job.get("held_texts") or {}, int(job.get("held_max_tokens") or 512)
    names = [ph["name"] for ph in phases]
    if len(set(names)) != len(names) or any(not n.isidentifier() or n in ("before", "default") or n in "lora_" for n in names):
        raise ValueError(f"the phases need names of their own, plain words: {names}")
    for ph in phases:  # everything that can be refused is refused before anything is trained
        if int(ph["steps"]) < 1 or any(int(s) < 1 or int(s) > int(ph["steps"]) for s in ph.get("looks") or ()):
            raise ValueError(f"phase {ph['name']}: {ph['steps']} steps cannot have looks at {list(ph.get('looks') or ())}")
        if ph.get("save") and save is None:
            raise ValueError(f"phase {ph['name']} asks to be saved and no way of saving was given")
    if model is not None and phases and not phases[0].get("stack"):
        raise ValueError(f"phase {names[0]}: after saved adapters only a stacked phase can train")
    model = base_model if model is None else model
    sha = lambda x: hashlib.sha256(json.dumps(x).encode("utf-8")).hexdigest()  # noqa: E731  (as the app hashes a job's data)

    def about(texts: list[str], most: int) -> dict:
        """Documents by count and hash, their mean length in their own tokens once cut, and how many were cut."""
        full = [len(r) for r in text_rows(tok, texts, 1 << 30)]
        own = [min(n, most) - (tok.bos_token_id is not None) for n in full]
        return {"documents": len(texts), "sha256": sha(texts), "mean_tokens": sum(own) / max(1, len(own)), "cut": sum(n > most for n in full)}

    def measure() -> dict:
        return {k: mean_text_loss(model, tok, v, cut) for k, v in held_texts.items()}

    held, looks, log, recs, trained = {"before": measure()}, {}, [], [], None

    def look(key: str) -> None:
        model.eval()  # a look in the middle of a phase finds the model in training mode: it is asked with its dropout off
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # what training left cached, before the answers
        looks[key] = generate(model, tok, look_sets, batch=batch)
        held[key] = measure()

    for ph in phases:
        lo, name, steps = job["lora"], ph["name"], int(ph["steps"])
        frozen = []
        if model is base_model:  # no adapter yet: this phase makes one, stacked or not
            model, trained = add_lora(base_model, seed=lo["seed"], rank=lo["rank"], alpha=lo["alpha"], dropout=lo["dropout"]), "default"
        elif ph.get("stack"):
            frozen = list(model.peft_config)
            model, trained = stack_fresh(model, seed=lo["seed"], rank=lo["rank"], alpha=lo["alpha"], dropout=lo["dropout"], name=name), name
        params = [p for p in model.parameters() if p.requires_grad]
        opts = {"per_example": bool(ph.get("per_example")), "reply_tokens": int(ph["reply_tokens"]) if ph.get("reply_tokens") else None}  # sft_loss's options, off unless the phase asks
        whole = (lambda: mean_sft_loss(model, base_model, tok, ph["data"], **opts)) if ph.get("whole_loss") else None  # point 18 of the own-advice step
        before = whole() if whole else None
        rows = train(model, base_model, tok, loss=ph["loss"], data=ph["data"], layer=None, seed=lo["seed"], lr=ph["lr"], batch=ph["batch"], steps=steps,
                     looks=tuple(sorted({int(s) for s in ph.get("looks") or () if int(s) < steps})),  # one named at the last step is the look below
                     at_look=lambda s, name=name: look(f"{name}@{s}"), max_tokens=int(ph.get("max_tokens") or 512), **opts)
        after = whole() if whole else None
        log += [{"phase": name, **row} for row in rows]
        look(name)
        if ph.get("save"):
            save(model, trained, ph["save"])
        rec = {"name": name, "loss": ph["loss"], "adapter": trained, "on_top_of": frozen, "steps": len(rows), "tensors": len(params),
               "parameters": int(sum(p.numel() for p in params)), "data_sha256": sha(ph["data"]),
               "lora_B_mean_abs": float(torch.stack([p.detach().abs().mean() for n, p in model.named_parameters()
                                                     if "lora_B" in n and f".{trained}." in n]).mean())}
        if ph["loss"] == "text":
            info = about(ph["data"], int(ph.get("max_tokens") or 512))
            rec |= {"documents": info["documents"], "mean_tokens": info["mean_tokens"], "cut": info["cut"]}
        else:
            rec["pairs"] = len(ph["data"])
        if opts["per_example"] or opts["reply_tokens"] or whole:  # only a phase that asks for an option records it: earlier steps' records stay as they were
            rec |= opts | {"loss_before": before, "loss_after": after}
        recs.append(rec)
    return {"model": model, "log": log, "looks": looks, "held": held, "phases": recs,
            "held_texts": {k: about(v, cut) for k, v in held_texts.items()},
            "lora": dict(job.get("lora") or {}) | {"adapters": [] if model is base_model else list(model.peft_config)}}
