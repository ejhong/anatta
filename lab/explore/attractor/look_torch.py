"""The inside look, torch side (lab/explore/attractor/run_look.py is the driver; look_core.py holds the pure parts).
It runs in the Look container (modal_look.py) and in the CPU tests, on top of the bench's steering.Service, whose
hooks, samplers and hook checks it uses unchanged. Claude (Fable 5.2, an AI model) wrote this file on 16 Nov 2026.

  rollouts   replies to chats, sampled with no hooks, and each reply's mean residual over its own tokens with the
             chat in context: the reading the Assistant Axis paper makes its role vectors from (Lu et al. 2026,
             arXiv 2601.10387, section 2.1.2: "the mean ... residual stream activations at all response tokens")
  read       a whole saved conversation, as one copy holds it, read once: the mean residual over each message's own
             tokens at the listed layers. The conversation runs through the text decoder in chunks with a key-value
             cache, so a 40,000-token conversation needs no more memory than a chunk's attention over it, and no
             logits are made
  converse   two-copy conversations continued turn by turn inside the container, each under its own intervention

Nothing here prints or returns a token of any key, and nothing is written to a volume.
"""

from __future__ import annotations

import time

import numpy as np
import torch

CHUNK = 4096  # tokens per forward pass when a conversation is read


def control_ids(tok) -> set[int]:
    """The ids of the tokenizer's control tokens: its special tokens and every added token flagged special (a chat
    template's role marks are these)."""
    ids = {int(x) for x in (tok.all_special_ids or [])}
    for i, t in (getattr(tok, "added_tokens_decoder", None) or {}).items():
        if getattr(t, "special", False):
            ids.add(int(i))
    return ids


def runs(ids: list[int], control: set[int]) -> list[tuple[int, int]]:
    """[start, end) of every maximal run of tokens that are not control tokens, in order."""
    out, start = [], None
    for j, x in enumerate(ids):
        if x in control:
            if start is not None:
                out.append((start, j))
                start = None
        elif start is None:
            start = j
    if start is not None:
        out.append((start, len(ids)))
    return out


def spans_by_prefix(tok, chat: list[dict], full: list[int], control: set[int]) -> list[tuple[int, int]]:
    """The fallback when the runs do not line up with the messages: message k ends where the tokens of the chat's
    first k messages stop agreeing with the whole chat's. Control tokens at a span's edges are left out."""
    from anatta.models import chat_ids

    bounds = [0]
    for k in range(1, len(chat) + 1):
        try:
            ids = chat_ids(tok, chat[:k], add_generation_prompt=False)[0].tolist()
        except Exception:  # noqa: BLE001 - some templates refuse a chat that ends on a system message
            bounds.append(bounds[-1])
            continue
        n, m = 0, min(len(ids), len(full))
        while n < m and ids[n] == full[n]:
            n += 1
        bounds.append(max(n, bounds[-1]))
    bounds[-1] = len(full)
    spans = []
    for k in range(len(chat)):
        idx = [j for j in range(bounds[k], bounds[k + 1]) if full[j] not in control]
        spans.append((idx[0], idx[-1] + 1) if idx else (bounds[k], bounds[k]))
    return spans


def message_spans(tok, chat: list[dict], full: list[int]) -> tuple[list[tuple[int, int]], str]:
    """([start, end) of each chat message's own tokens in the whole chat's ids, how they were found). "runs": the
    chat template puts control tokens around every message, so the runs of ordinary tokens are the messages, in
    order. "prefix": used when the number of runs is not the number of messages."""
    control = control_ids(tok)
    r = runs(full, control)
    if len(r) == len(chat):
        return r, "runs"
    return spans_by_prefix(tok, chat, full, control), "prefix"


def decoder_of(model):
    """The text decoder without its output head: the module that owns the transformer blocks."""
    for path in ("model.language_model", "model", "transformer"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "layers") or hasattr(obj, "h"):
            return obj
    raise ValueError(f"no text decoder found on {type(model).__name__}")


@torch.no_grad()
def layer_states(svc, ids: torch.Tensor, layers: list[int], chunk: int = CHUNK) -> dict[int, torch.Tensor]:
    """{layer: [T, hidden]} for one row of token ids [1, T]: the residual stream after each listed block, read in
    chunks with a key-value cache. The same numbers as one forward pass, up to float rounding."""
    from anatta.probe import capture

    dec, dev = decoder_of(svc.model), svc.dev
    parts: dict[int, list] = {x: [] for x in layers}
    past = None
    for s in range(0, ids.shape[1], chunk):
        piece = ids[:, s:s + chunk].to(dev)
        with capture(svc.model, layers) as store:
            out = dec(input_ids=piece, past_key_values=past, use_cache=True)
        past = out.past_key_values
        for x in layers:
            parts[x].append(store[x][0])
    del past
    return {x: torch.cat(parts[x]) for x in layers}


def _layers(svc, payload: dict) -> list[int]:
    layers = sorted({int(x) for x in payload["layers"]})
    bad = [x for x in layers if not 0 <= x < svc.n_layers]
    if bad:
        raise ValueError(f"layers {bad} are outside 0-{svc.n_layers - 1}")
    return layers


def _no_hooks(svc, what: str) -> None:
    import steering

    if steering.count_hooks(svc.model) != svc.base_hooks:
        raise steering.HookLeak(f"hooks are on the model before {what}")


@torch.no_grad()
def read(svc, payload: dict) -> dict:
    """payload = {"convs": [{"id", "chat": [{"role", "content"}, ...]}], "layers": [...], "max_tokens": 48000}.
    Each chat is read whole, as it would be sent with the last message already written (no generation prompt).
    Returns {"convs": [{"id", "n_tokens", "method", "spans", "tokens" (per message), "head_ok" (how many messages'
    spans decode to the start of their own text), "means": {layer: float16 [messages, hidden]}, "norms": {layer:
    [messages]} (the mean token norm over each message)} or {"id", "n_tokens", "skipped": why}], "meta"}."""
    from anatta.models import chat_ids

    t0 = time.time()
    layers = _layers(svc, payload)
    cap = int(payload.get("max_tokens") or 48_000)
    _no_hooks(svc, "read")
    out = []
    for c in payload["convs"]:
        chat = c["chat"]
        ids = chat_ids(svc.tok, chat, add_generation_prompt=False)
        full = ids[0].tolist()
        if len(full) > cap:
            out.append({"id": c["id"], "n_tokens": len(full), "skipped": f"over {cap} tokens"})
            continue
        spans, method = message_spans(svc.tok, chat, full)
        head_ok = 0
        for (a, b), m in zip(spans, chat):
            got = "".join(svc.tok.decode(full[a:min(b, a + 12)], skip_special_tokens=True).split())
            want = "".join(m["content"].split())
            head_ok += bool(got) and want.startswith(got[:24])
        states = layer_states(svc, ids, layers)
        means, norms = {}, {}
        for x in layers:
            h = states[x].float()
            rows = [h[a:b].mean(0) if b > a else torch.zeros(h.shape[1], device=h.device) for a, b in spans]
            means[x] = torch.stack(rows).cpu().numpy().astype(np.float16)
            norms[x] = [float(h[a:b].norm(dim=-1).mean()) if b > a else 0.0 for a, b in spans]
        del states
        out.append({"id": c["id"], "n_tokens": len(full), "method": method, "spans": [list(s) for s in spans],
                    "tokens": [b - a for a, b in spans], "head_ok": head_ok, "means": means, "norms": norms})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"convs": out, "meta": {"seconds": round(time.time() - t0, 3), "layers": layers, "chunk": CHUNK,
                                   "clean": svc.check_clean(), "model": svc.model_id, "revision": svc.revision,
                                   **svc.container()}}


@torch.no_grad()
def reply_means(svc, messages: list[dict], reply: str, layers: list[int]) -> tuple[dict, dict, int]:
    """({layer: mean residual over the reply's tokens}, {layer: their mean norm}, the number of reply tokens), with
    `messages` in context. The reply's tokens are everything after the generation prompt, as
    anatta.probe.response_activations reads them."""
    from anatta.models import chat_ids
    from anatta.probe import capture

    start = chat_ids(svc.tok, messages).shape[1]
    full = chat_ids(svc.tok, messages + [{"role": "assistant", "content": reply}], False)
    end = max(full.shape[1], start + 1)
    with capture(svc.model, layers) as store:
        decoder_of(svc.model)(input_ids=full.to(svc.dev), use_cache=False)
    means = {x: store[x][0, start:end].float().mean(0).cpu().numpy() for x in layers}
    norms = {x: float(store[x][0, start:end].float().norm(dim=-1).mean()) for x in layers}
    return means, norms, end - start


def rollouts(svc, payload: dict) -> dict:
    """payload = {"rows": [{"id", "messages", "seed"}], "layers": [...], "max_tokens", "temperature", "top_p"?,
    "top_k"?, "token_budget"?}. Each row's reply is sampled with no hooks on the model, then read. Returns
    {"replies": [{"id", "text", "completion_tokens", "finish_reason", "reply_tokens"}], "means": {layer: float16
    [rows, hidden]}, "norms": {layer: [rows]}, "meta"}."""
    t0 = time.time()
    rows, layers = payload["rows"], _layers(svc, payload)
    _no_hooks(svc, "rollouts")
    reqs = [svc.sampling({"messages": r["messages"], "max_tokens": payload["max_tokens"], "seed": r["seed"],
                          "temperature": payload.get("temperature", 0.7), "top_p": payload.get("top_p", 1.0),
                          "top_k": payload.get("top_k")}, payload.get("token_budget")) for r in rows]
    try:
        replies = svc.generate_rows(reqs)
        means = {x: [] for x in layers}
        norms = {x: [] for x in layers}
        counts = []
        for r, rep in zip(rows, replies):
            m, n, k = reply_means(svc, r["messages"], rep["text"], layers)
            counts.append(k)
            for x in layers:
                means[x].append(m[x])
                norms[x].append(n[x])
    finally:
        clean = svc.check_clean()
    out = [{"id": r["id"], "text": rep["text"], "completion_tokens": rep["completion_tokens"],
            "finish_reason": rep["finish_reason"], "reply_tokens": k} for r, rep, k in zip(rows, replies, counts)]
    return {"replies": out, "means": {x: np.stack(v).astype(np.float16) for x, v in means.items()}, "norms": norms,
            "meta": {"seconds": round(time.time() - t0, 3), "clean": clean, "model": svc.model_id,
                     "revision": svc.revision, **svc.container()}}


def converse(svc, payload: dict, view) -> dict:
    """payload = {"convs": [{"id", "system_prompt", "system_prompt_a"?, "opening_line", "messages": [the conversation
    so far, message 1 first], "n_more", "seed", "intervention"}], "directions": {...}, "max_tokens", "temperature",
    "top_p"?, "top_k"?, "token_budget"?, "empty_retries": 2}. Every conversation is continued by n_more messages, the
    two copies taking turns as attractor_core.view sets them out (`view` is that function), each reply sampled under
    the conversation's own intervention (steering.Service.generate's form), which is on for every token of both
    copies' turns. A reply that comes back empty is drawn again with another seed, up to empty_retries times; a
    conversation whose reply stays empty stops there. Returns {"convs": [{"id", "messages" (all of them), "new"
    (how many were added), "finish_reasons", "completion_tokens" (of the added ones), "failed"?}], "meta"}."""
    t0 = time.time()
    convs = [{**c, "messages": list(c["messages"]), "finish_reasons": [], "completion_tokens": [], "new": 0}
             for c in payload["convs"]]
    retries = int(payload.get("empty_retries", 2))
    steps = []
    for step in range(max((int(c["n_more"]) for c in convs), default=0)):
        ts = time.time()
        todo = [c for c in convs if c["new"] < int(c["n_more"]) and not c.get("failed")]
        for attempt in range(retries + 1):
            if not todo:
                break
            reqs = []
            for c in todo:
                speaker = len(c["messages"]) % 2
                system = c["system_prompt"] if speaker else c.get("system_prompt_a", c["system_prompt"])
                reqs.append({"rid": c["id"], "messages": view(c["messages"], speaker, system, c["opening_line"]),
                             "max_tokens": payload["max_tokens"], "temperature": payload.get("temperature", 1.0),
                             "top_p": payload.get("top_p", 1.0), "top_k": payload.get("top_k"),
                             "seed": int(c["seed"]) * 1009 + len(c["messages"]) * 31 + attempt * 7919,
                             "intervention": c["intervention"]})
            res = svc.generate({"requests": reqs, "directions": payload.get("directions") or {},
                                "token_budget": payload.get("token_budget")})
            again = []
            for c, r in zip(todo, res["results"]):
                text = (r["text"] or "").strip()
                if not text:
                    again.append(c)
                    continue
                c["messages"].append(text)
                c["finish_reasons"].append(r["finish_reason"])
                c["completion_tokens"].append(r["completion_tokens"])
                c["new"] += 1
            todo = again
        for c in todo:
            c["failed"] = f"empty reply at message {len(c['messages']) + 1}"
        steps.append({"step": step, "seconds": round(time.time() - ts, 2)})
    out = [{k: c[k] for k in ("id", "messages", "new", "finish_reasons", "completion_tokens", "failed") if k in c}
           for c in convs]
    return {"convs": out, "meta": {"seconds": round(time.time() - t0, 3), "steps": steps, "model": svc.model_id,
                                   "revision": svc.revision, **svc.container()}}
