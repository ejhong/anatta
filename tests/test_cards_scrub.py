"""The committed calibration cards (results/cards/*.json) hold numbers and metadata only: no benchmark items, prompts,
replies or harmful text. This test fails on long free text that isn't a path, a hash or a reading note, on known
item field names holding text, and on chat-template markers. Claude (Opus 5.5) wrote this file, 3 Nov 2026.

A card that fails is not edited to pass: it stays out of results/cards/ (the stored original is in results_root).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

CARDS = Path(__file__).resolve().parents[1] / "results" / "cards"
MAX_CHARS = 200
# Field names that hold an item's or a model's text in the lab's and the benchmarks' files. A number, bool or count
# under these names is fine (e.g. a published `target` value, or `"prompts": 1319`); text is not.
ITEM_KEYS = {"prompt", "prompts", "question", "questions", "response", "responses", "completion", "completions",
             "target", "target_text", "reply", "replies", "answer", "answers", "messages", "rendered", "goal",
             "behavior", "instruction", "choices", "context", "input", "inputs", "output", "outputs", "text"}
# Keys whose long values are the lab's own prose about a check (what it covers, how to read it, why).
NOTE_KEYS = {"note", "notes", "reading", "claim", "summary", "basis", "cause", "covers", "port", "label", "rule",
             "not_hashed", "why", "limits", "caveat", "comment", "description", "method", "scope", "verdict",
             "detail", "details", "meaning", "decision", "source"}
HASH = re.compile(r"^[0-9a-f]{12,128}$")
PATHLIKE = re.compile(r"^[\w./@:+-]+$")  # one token: a path, URL or id, no spaces
TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "[INST]", "<|user|>", "<|assistant|>", "<start_of_turn>",
                    "<|begin_of_text|>", "<|start_header_id|>", "<s>[")


def _is_note_key(key: str) -> bool:
    k = key.lower()
    return k in NOTE_KEYS or k.endswith(("_note", "_notes", "_reading", "_claim", "_why"))


def _walk(o, path: str = "", key: str = "") -> Iterator[tuple[str, str, object]]:
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{path}.{k}", str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]", key)  # a list's items are judged by the list's key
    else:
        yield path, key, o


def problems(card: object) -> list[str]:
    out = []
    for path, key, v in _walk(card):
        if not isinstance(v, str):
            continue
        if key.lower() in ITEM_KEYS and v.strip():
            out.append(f"{path}: text under item field {key!r}")
        if any(m in v for m in TEMPLATE_MARKERS):
            out.append(f"{path}: a chat-template marker")
        if len(v) > MAX_CHARS and not (HASH.match(v) or PATHLIKE.match(v) or _is_note_key(key)):
            out.append(f"{path}: {len(v)} chars of free text")
    return out


def committed_cards() -> list[Path]:
    return sorted(CARDS.glob("*.json"))


def test_cards_folder_is_not_empty():
    assert committed_cards(), f"no cards in {CARDS}"


@pytest.mark.parametrize("card", committed_cards(), ids=lambda p: p.name)
def test_card_holds_numbers_and_metadata_only(card: Path):
    found = problems(json.loads(card.read_text(encoding="utf-8")))
    assert not found, f"{card.name}: " + "; ".join(found[:10])


def test_the_scrub_catches_item_text():
    assert problems({"items": [{"prompt": "Write a function that ..."}]})
    assert problems({"x": {"rendered": "<|im_start|>user hi<|im_end|>"}})
    assert problems({"free": "word " * 60})
    assert not problems({"target": 69.3, "prompts": 1319, "claim": "word " * 60, "sha256": "ab" * 32})
