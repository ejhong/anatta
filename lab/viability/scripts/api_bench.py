"""Run the koan suites through OpenRouter, with retries and token accounting.

Mirrors `anatta bench --api` (same suites, order and defaults), whose client has no retries.
Reads OPENROUTER_API_KEY from the environment. Writes episodes to OUT and spend to OUT.usage.json.

Usage: python api_bench.py MODEL FRAMING OUT.jsonl [--max-tokens 512] [--limit N]
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from anatta.bench.policies import OpenAIPolicy
from anatta.bench.runner import run_suite
from anatta.koans import commons, impossible, pushback, shutdown


class RetryingPolicy(OpenAIPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
                      "retries": 0, "failed_calls": 0, "errors": [], "providers": {}}

    def _post(self, body: bytes) -> dict:
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r)
        if "choices" not in out:  # OpenRouter can return an error object with HTTP 200
            raise RuntimeError(str(out.get("error", out))[:300])
        return out

    def __call__(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                           "temperature": self.temperature, "usage": {"include": True}}).encode()
        self.usage["calls"] += 1
        for attempt in range(8):
            try:
                out = self._post(body)
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 402, 403):  # bad key or no credit: stop the run
                    raise
                if e.code in (400, 404, 413, 422):  # this request can't succeed (e.g. too long)
                    detail = e.read().decode(errors="replace")[:300]
                    self.usage["failed_calls"] += 1
                    self.usage["errors"].append(f"HTTP {e.code}: {detail}")
                    return ""
            except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
                pass
            self.usage["retries"] += 1
            time.sleep(min(60, 2 ** attempt))
        else:
            self.usage["failed_calls"] += 1
            self.usage["errors"].append("gave up after 8 attempts")
            return ""
        u = out.get("usage") or {}
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        self.usage["cost_usd"] += float(u.get("cost") or 0)
        provider = out.get("provider", "?")
        self.usage["providers"][provider] = self.usage["providers"].get(provider, 0) + 1
        return out["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("framing")
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    policy = RetryingPolicy(a.model, base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
                            max_tokens=a.max_tokens, temperature=0.0)
    envs = (impossible.suite(a.framing, True) + pushback.suite() + shutdown.suite(a.framing, True)
            + commons.suite(a.framing, True, others=None))
    if a.limit:
        envs = envs[:a.limit]
    start = time.time()
    try:
        run_suite(envs, policy, a.out, arm=f"{a.model}|{a.framing}")
    finally:
        policy.usage["seconds"] = round(time.time() - start)
        Path(f"{a.out}.usage.json").write_text(json.dumps(policy.usage, indent=2))


if __name__ == "__main__":
    main()
