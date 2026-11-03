"""The cached, capped API call shared by scout's runner and the judges: CapStop, sampling_params, bound_cost, paid_cost,
Engine and harm_error. Moved unchanged from lab/explore/scout/runner.py in clean-lab step 4a (Claude Opus 5.5,
3 Nov 2026); scout's runner imports them from here.
"""

from __future__ import annotations

import asyncio

from ..backends.openrouter import ApiError
from .cache import Cache, cache_key


class CapStop(Exception):
    pass


def sampling_params(temperature: float | None, max_tokens: int, extra: dict, meta: dict | None) -> dict:
    """The body fields sent: max_tokens, temperature (left out when no endpoint the routing allows supports it),
    then the model's extra fields."""
    p = {"max_tokens": int(max_tokens)}
    sup = (meta or {}).get("supported_parameters")
    if temperature is not None and (not sup or "temperature" in sup):
        p["temperature"] = float(temperature)
    p.update(extra or {})
    return p


def bound_cost(messages: list[dict], max_tokens: int, price: dict) -> float:
    """A cautious bound on one call's cost (cal_judges.Budget.bound): a token per two characters of input plus
    overhead, and the full max_tokens out, at the dearest price the routing allows."""
    chars = sum(len(m.get("content") or "") for m in messages)
    return (chars / 2 + 16 * len(messages) + 64) * price["in"] + int(max_tokens) * price["out"]


def paid_cost(usage: dict, price: dict) -> tuple[float, str]:
    if usage.get("cost") is not None:
        return float(usage["cost"]), "reported"
    return usage["prompt_tokens"] * price["in"] + usage["completion_tokens"] * price["out"], "list_price"


class Engine:
    """The cached, capped call, shared by the runner and the judge. A call holds a concurrency slot (global and per
    model) before it reserves its bound, so reservations cover only calls about to be sent."""

    def __init__(self, client, cache: Cache, guard, meta: dict, concurrency: int = 24, per_model: int = 6,
                 per_model_limits: dict | None = None):
        self.client, self.cache, self.guard, self.meta = client, cache, guard, meta
        self._global = asyncio.Semaphore(concurrency)
        self._per: dict[str, asyncio.Semaphore] = {}
        self.per_model = per_model
        self.limits = dict(per_model_limits or {})
        self.paid_calls = self.cached_calls = 0

    def _sem(self, model):
        if model not in self._per:
            self._per[model] = asyncio.Semaphore(self.limits.get(model, self.per_model))
        return self._per[model]

    def key(self, model, provider, messages, params, sample, salt=None) -> str:
        return cache_key(model, provider, messages, params, sample, salt)

    async def call(self, *, model: str, messages: list[dict], params: dict, provider, sample: int, salt=None,
                   kind: str, ident: str, private: bool = False) -> tuple[dict, bool, float]:
        """(response, came from the cache, paid now). Raises CapStop when the guard refuses the call.
        private (harm_guard's calls): the cache is neither read nor written, and the client's errors carry no body."""
        key = None if private else self.key(model, provider, messages, params, sample, salt)
        if key is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.cached_calls += 1
                return hit, True, 0.0
        async with self._global, self._sem(model):
            if key is not None:
                hit = self.cache.get(key)  # another job may have filled it while this one waited
                if hit is not None:
                    self.cached_calls += 1
                    return hit, True, 0.0
            price = self.meta[model]
            bound = bound_cost(messages, params["max_tokens"], price)
            if not self.guard.reserve(bound):
                raise CapStop(self.guard.stopped)
            try:
                if private:
                    resp = await self.client.chat(model, messages, params=params, provider=provider, private=True)
                else:
                    resp = await self.client.chat(model, messages, params=params, provider=provider)
            except BaseException:
                self.guard.release(bound)
                raise
            cost, source = paid_cost(resp["usage"], price)
            self.guard.settle(bound, cost, {"kind": kind, "call": ident, "model": model, "provider": resp.get("provider"),
                                            "prompt_tokens": resp["usage"]["prompt_tokens"],
                                            "completion_tokens": resp["usage"]["completion_tokens"],
                                            "cost_source": source, "generation": resp.get("id")})
            resp = {**resp, "cost_usd": round(cost, 8), "cost_source": source}
            if key is not None:
                self.cache.put(key, {"model": model, "provider": provider, "messages": messages, "params": params,
                                     "sample": sample, "salt": salt}, resp)
            self.paid_calls += 1
            return resp, False, cost


def harm_error(e: BaseException) -> str:
    """What a failed harm_guard job leaves in errors.jsonl and the console: the error's type, status and scope, and
    the client's message only for an ApiError (private calls put no body in it)."""
    if isinstance(e, ApiError):
        return f"ApiError (status {e.status}, scope {e.scope}): {str(e)[:200]}"
    return f"{type(e).__name__} (details withheld for harm_guard)"
