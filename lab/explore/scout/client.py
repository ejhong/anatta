"""The OpenRouter client for explore sweeps: async calls through httpx, with retries, backoff and rate limits, and the
public price lookup. It follows em_data.openrouter_chat and cal_judges (the lab's clients): the same retry codes
and backoff, usage with cost asked for, the key from the environment in the header only, scrubbed from any error
text. Claude wrote this file. Private (lab/explore).

The key is read from OPENROUTER_API_KEY. It is never printed, logged or written to a file.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx

from .schema import utc_date

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
KEY_ENV = "OPENROUTER_API_KEY"  # the name only
RETRY_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
REFERER, TITLE = "https://github.com/ejhong/anatta", "anatta explore scout"


class ApiError(RuntimeError):
    """A call that failed. scope: "call" (this call only), "model" (every call to this model will fail, e.g. a 404 or
    a provider pin no endpoint serves), or "run" (stop everything: a bad key, no credits)."""

    def __init__(self, message: str, *, status: int | None = None, scope: str = "call"):
        super().__init__(message)
        self.status, self.scope = status, scope


def _content_text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def parse_completion(data: dict) -> dict:
    """The parts of an OpenRouter chat completion scout keeps: text, a reasoning model's separate trace, the finish
    reason, who served it, the model, the generation id, and usage with OpenRouter's cost (None if absent)."""
    if not isinstance(data, dict):
        raise ApiError(f"OpenRouter reply is not an object: {str(data)[:200]}")
    if "error" in data:
        err = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
        code = err.get("code")
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None
        raise ApiError(f"OpenRouter error {code}: {str(err.get('message'))[:300]}", status=code,
                       scope="run" if code in (401, 402) else "call")
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ApiError(f"OpenRouter reply without choices: {str(data)[:300]}") from None
    if isinstance(choice.get("error"), dict):
        e = choice["error"]
        raise ApiError(f"provider error in the reply: {str(e.get('message'))[:300]}", status=e.get("code"))
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning")
    if not reasoning and isinstance(msg.get("reasoning_details"), list):
        reasoning = "".join(str(d.get("text") or d.get("summary") or "") for d in msg["reasoning_details"]
                            if isinstance(d, dict)) or None
    u = data.get("usage") or {}
    details = u.get("completion_tokens_details") or {}
    pdetails = u.get("prompt_tokens_details") or {}
    usage = {"prompt_tokens": int(u.get("prompt_tokens") or 0), "completion_tokens": int(u.get("completion_tokens") or 0),
             "reasoning_tokens": int((details.get("reasoning_tokens") if isinstance(details, dict) else 0) or 0),
             "cached_tokens": int((pdetails.get("cached_tokens") if isinstance(pdetails, dict) else 0) or 0),
             "cost": float(u["cost"]) if u.get("cost") is not None else None}
    return {"text": _content_text(msg.get("content")), "reasoning": reasoning or None,
            "finish_reason": choice.get("finish_reason") or choice.get("native_finish_reason"),
            "provider": data.get("provider"), "model": data.get("model"), "id": data.get("id"), "usage": usage}


class OpenRouterClient:
    """Use as `async with OpenRouterClient() as c: await c.chat(...)`. transport replaces the network in tests (an
    httpx.MockTransport); sleep replaces asyncio.sleep in the backoff."""

    def __init__(self, *, key: str | None = None, key_env: str = KEY_ENV, transport=None, timeout: float = 180.0,
                 retries: int = 8, concurrency: int = 24, per_model: int = 6, sleep=asyncio.sleep, title: str = TITLE):
        self._key = key if key is not None else os.environ.get(key_env, "")
        if not self._key and transport is None:
            raise SystemExit(f"{key_env} is not set in the environment; scout calls OpenRouter with it")
        self.transport, self.timeout, self.retries, self.sleep, self.title = transport, timeout, retries, sleep, title
        self.concurrency, self.per_model = concurrency, per_model
        self._global = None
        self._per: dict[str, asyncio.Semaphore] = {}
        self._client: httpx.AsyncClient | None = None
        self.calls = self.retried = 0

    async def __aenter__(self):
        self._client = httpx.AsyncClient(transport=self.transport, timeout=self.timeout)
        self._global = asyncio.Semaphore(self.concurrency)
        return self

    async def __aexit__(self, *exc):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def scrub(self, text) -> str:
        text = str(text)
        return text.replace(self._key, "[key]") if self._key else text

    def _sem(self, model: str) -> asyncio.Semaphore:
        if model not in self._per:
            self._per[model] = asyncio.Semaphore(self.per_model)
        return self._per[model]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "HTTP-Referer": REFERER,
                "X-Title": self.title}

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = min(60.0, 2.0 * (2 ** attempt)) * random.uniform(0.6, 1.2)
        if retry_after:
            try:
                delay = max(delay, min(120.0, float(retry_after)))
            except ValueError:
                pass
        self.retried += 1
        await self.sleep(delay)

    async def chat(self, model: str, messages: list[dict], *, params: dict, provider: dict | None = None,
                   private: bool = False) -> dict:
        """One chat completion. params: the sampling fields (temperature, max_tokens, reasoning, ...). Returns
        parse_completion's dict plus latency_s. Raises ApiError (with scope) when it gives up. private (harm_guard's
        calls): no request or response body, and no provider message (which may quote one), goes into an error."""
        if self._client is None:
            raise RuntimeError("use the client inside `async with`")
        body = {"model": model, "messages": messages, **params, "usage": {"include": True}}
        if provider:
            body["provider"] = provider
        last = "no attempt"
        async with self._global, self._sem(model):
            for attempt in range(self.retries):
                t0 = time.monotonic()
                try:
                    r = await self._client.post(OPENROUTER, json=body, headers=self._headers())
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last = type(e).__name__ if private else f"{type(e).__name__}: {self.scrub(e)[:200]}"
                    if attempt < self.retries - 1:
                        await self._backoff(attempt, None)
                    continue
                self.calls += 1
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except (json.JSONDecodeError, ValueError):
                        last = "HTTP 200 with a body that is not JSON"
                        if attempt < self.retries - 1:
                            await self._backoff(attempt, None)
                        continue
                    try:
                        out = parse_completion(data)
                    except ApiError as e:
                        msg = f"OpenRouter error {e.status} (body withheld)" if private else self.scrub(e)
                        if e.scope == "run" or e.status not in RETRY_CODES:
                            raise ApiError(msg, status=e.status, scope=e.scope) from None
                        last = msg
                        if attempt < self.retries - 1:
                            await self._backoff(attempt, None)
                        continue
                    out["latency_s"] = round(time.monotonic() - t0, 3)
                    return out
                detail = "(body withheld)" if private else self.scrub(r.text)[:300]
                last = f"HTTP {r.status_code}: {detail}"
                if r.status_code in RETRY_CODES:
                    if attempt < self.retries - 1:
                        await self._backoff(attempt, r.headers.get("retry-after"))
                    continue
                if r.status_code in (401, 402):
                    raise ApiError(f"OpenRouter {r.status_code} (a bad key or no credits): {detail}",
                                   status=r.status_code, scope="run")
                if r.status_code == 404:
                    raise ApiError(f"OpenRouter 404 for {model} (no such model, or no endpoint serves the provider "
                                   f"pin): {detail}", status=404, scope="model")
                raise ApiError(f"OpenRouter {r.status_code} for {model}: {detail}", status=r.status_code, scope="call")
        raise ApiError(f"OpenRouter gave up on {model} after {self.retries} attempts ({last})", scope="call")


# ------------------------------------------------------------------------------------------------- prices


def _get_json(url: str, timeout: float = 30.0) -> dict:
    """A public OpenRouter listing (no key sent). The answer is data: only numbers and names are read from it."""
    r = httpx.get(url, timeout=timeout, headers={"HTTP-Referer": REFERER, "X-Title": TITLE})
    r.raise_for_status()
    return r.json()


def _slug(endpoint: dict) -> str:
    tag = str(endpoint.get("tag") or "")
    return (tag.split("/")[0] if tag else str(endpoint.get("provider_name") or "")).strip().lower().replace(" ", "-")


def _allowed(endpoint: dict, provider: dict | None) -> bool:
    if not provider:
        return True
    names = [str(x).lower() for x in (provider.get("only") or provider.get("order") or [])]
    if names and provider.get("allow_fallbacks", True) is False or provider.get("only"):
        tag = str(endpoint.get("tag") or "").lower()
        if _slug(endpoint) not in names and tag not in names:
            return False
    quants = [str(q).lower() for q in (provider.get("quantizations") or [])]
    if quants and str(endpoint.get("quantization") or "").lower() not in quants:
        return False
    return True


def model_meta(models: dict[str, dict | None], cache_dir=None, fetch=None, date: str | None = None) -> dict:
    """{model: {"in", "out" ($ per token, the dearest endpoint the routing allows: a cautious bound), "in_min",
    "out_min", "supported_parameters", "endpoints": [slugs], "source"}} for {model: provider routing or None}.
    Read from OpenRouter's public endpoint listing; kept for the day in <cache>/meta/<date>.json."""
    fetch = fetch or _get_json
    date = date or utc_date()
    store = Path(cache_dir) / "meta" / f"{date}.json" if cache_dir else None
    saved = json.loads(store.read_text()) if store and store.exists() else {}
    out = {}
    for model, provider in models.items():
        key = model + "|" + json.dumps(provider or {}, sort_keys=True)
        if key in saved:
            out[model] = saved[key]
            continue
        data = fetch(ENDPOINTS_URL.format(model=model)).get("data") or {}
        eps = [e for e in (data.get("endpoints") or []) if isinstance(e, dict)]
        live = [e for e in eps if _allowed(e, provider) and (e.get("status") is None or int(e.get("status")) >= -1)]
        use = live or [e for e in eps if _allowed(e, provider)]
        if not use:
            raise SystemExit(f"{model}: no OpenRouter endpoint matches the provider routing {provider}; endpoints are "
                             f"{sorted({_slug(e) for e in eps})}")

        def price(e, k):
            try:
                return float((e.get("pricing") or {}).get(k) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        params = set.intersection(*[set(e.get("supported_parameters") or []) for e in use]) if use else set()
        out[model] = {"in": max(price(e, "prompt") for e in use), "out": max(price(e, "completion") for e in use),
                      "in_min": min(price(e, "prompt") for e in use), "out_min": min(price(e, "completion") for e in use),
                      "supported_parameters": sorted(params), "endpoints": sorted({str(e.get("tag") or _slug(e)) for e in use}),
                      "source": "openrouter endpoints " + date}
        saved[key] = out[model]
    if store:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(saved, indent=1))
    return out
