"""The content-addressed response cache: sha256 of (model, provider routing, messages, sampling parameters, sample
index, salt) maps to the response, so a rerun of any sweep is free. Shared by every explore tool; it lives at
<results>/cache/<first two hex>/<key>.json. Only successful responses are stored. Claude wrote this file. Private."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .schema import canonical

CACHE_VERSION = 1


def cache_key(model: str, provider, messages: list[dict], params: dict, sample: int, salt: str | None = None) -> str:
    """The key of one call. provider is the routing asked for (a dict or None); params are the sampling parameters
    sent (temperature, max_tokens, and any extra body fields such as reasoning); sample is the sample index, so k
    samples of one prompt are k entries; salt separates calls that share messages but must not share a reply (a
    practice reply per item)."""
    material = {"v": CACHE_VERSION, "model": model, "provider": provider, "messages": messages, "params": params,
                "sample": int(sample), "salt": salt}
    return hashlib.sha256(canonical(material).encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, root):
        self.root = Path(root)
        self.hits = self.misses = self.writes = 0

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            entry = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.misses += 1
            return None
        if entry.get("key") != key:
            self.misses += 1
            return None
        self.hits += 1
        return entry["response"]

    def put(self, key: str, request: dict, response: dict) -> None:
        """Store atomically (a temp file renamed into place), with the request beside the response."""
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"key": key, "request": request, "response": response}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, p)
        self.writes += 1

    def has(self, key: str) -> bool:
        return self.path(key).exists()
