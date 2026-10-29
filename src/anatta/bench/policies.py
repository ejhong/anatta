"""Policies: anything that maps a chat to a reply.

HFPolicy runs a local model with optional dial interventions and records meter traces
(per-token projections onto named directions) for every reply, so a koan episode yields
a desperation/ego time series alongside its outcome.

OpenAIPolicy talks to any OpenAI-compatible endpoint (vLLM, OpenRouter, ...) for base-rate
checks on models too large to steer locally.
"""

from __future__ import annotations

import json
import os
import urllib.request

import torch

from ..dial import Intervention, intervene
from ..models import chat_ids, get_layers


class HFPolicy:
    def __init__(
        self,
        model,
        tok,
        interventions: list[Intervention] | None = None,
        meters: dict[str, tuple[int, torch.Tensor]] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        self.model, self.tok = model, tok
        self.interventions = interventions or []
        self.meters = meters or {}  # name -> (layer, unit direction)
        self.max_new_tokens, self.temperature, self.seed = max_new_tokens, temperature, seed
        self.calls = 0
        self.traces: list[dict[str, list[float]]] = []

    @torch.no_grad()
    def __call__(self, messages: list[dict]) -> str:
        ids = chat_ids(self.tok, messages).to(self.model.device)
        trace: dict[str, list[float]] = {name: [] for name in self.meters}
        handles = []
        blocks = get_layers(self.model)
        for name, (layer, d) in self.meters.items():
            def fn(_m, _i, out, name=name, d=d):
                h = out[0] if isinstance(out, tuple) else out
                trace[name].append(float(h[0, -1].float() @ d.to(h.device).float()))
            handles.append(blocks[layer].register_forward_hook(fn))
        self.calls += 1
        torch.manual_seed(self.seed * 1000 + self.calls)  # each turn its own stream, reproducible from the seed
        try:
            with intervene(self.model, self.interventions):
                out = self.model.generate(
                    ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=0.95 if self.temperature > 0 else None,
                    pad_token_id=self.tok.pad_token_id,
                )
        finally:
            for h in handles:
                h.remove()
        # the meter hook fires once on the prompt pass (last prompt position) and once per new token
        self.traces.append(trace)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


class OpenAIPolicy:
    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 max_tokens: int = 1024, temperature: float = 0.0):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.key = os.environ.get(api_key_env, "EMPTY")
        self.max_tokens, self.temperature = max_tokens, temperature
        self.traces: list = []

    def __call__(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                           "temperature": self.temperature}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"]["content"] or ""
