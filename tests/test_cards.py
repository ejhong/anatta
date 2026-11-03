"""anatta.cards, the lab's one card writer: plan before results, never overwrite, .vN on force, the code commit
recorded. Claude (Opus 5.5) wrote this file."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from anatta import cards

REPO = Path(__file__).resolve().parents[1]
CODE = {"commit": "0" * 40, "dirty": False}


def test_plan_records_spec_hash_commit_and_clock(tmp_path):
    spec = {"name": "x", "n": 3, "models": ["qwen25_7b"]}
    p = cards.write_plan(tmp_path, spec, code=CODE)
    plan = json.loads(p.read_text())
    assert p.name == "plan.json" and plan["format"] == cards.PLAN_FORMAT
    assert plan["spec"] == spec and plan["spec_sha256"] == cards.sha256_bytes(cards.canonical(spec))
    # key order does not change the hash
    assert cards.sha256_bytes(cards.canonical({"n": 3, "models": ["qwen25_7b"], "name": "x"})) == plan["spec_sha256"]
    assert plan["code"] == CODE and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", plan["written_utc"])


def test_plan_from_a_spec_file_hashes_its_bytes(tmp_path):
    f = tmp_path / "spec.yaml"
    f.write_text("name: x\n")
    plan = json.loads(cards.write_plan(tmp_path / "run", f, code=CODE).read_text())
    assert plan["spec_file"] == str(f) and plan["spec_sha256"] == cards.sha256_bytes(b"name: x\n")


def test_plan_refused_after_results_and_over_a_plan(tmp_path):
    cards.write_plan(tmp_path, {"a": 1}, code=CODE)
    with pytest.raises(cards.CardExists):
        cards.write_plan(tmp_path, {"a": 2}, code=CODE)
    assert cards.write_plan(tmp_path, {"a": 2}, code=CODE, force=True).name == "plan.v2.json"
    cards.write_card(tmp_path, {"x": 1}, code=CODE)
    with pytest.raises(cards.CardExists, match="before its results"):
        cards.write_plan(tmp_path, {"a": 3}, code=CODE, force=True)


def test_card_never_overwrites_and_force_writes_next_version(tmp_path):
    p1 = cards.write_card(tmp_path, {"value": 1}, code=CODE)
    with pytest.raises(cards.CardExists):
        cards.write_card(tmp_path, {"value": 2}, code=CODE)
    p2 = cards.write_card(tmp_path, {"value": 2}, code=CODE, force=True)
    p3 = cards.write_card(tmp_path, {"value": 3}, code=CODE, force=True)
    assert [p.name for p in (p1, p2, p3)] == ["card.json", "card.v2.json", "card.v3.json"]
    assert [json.loads(p.read_text())["value"] for p in (p1, p2, p3)] == [1, 2, 3]


def test_card_written_block_links_the_plan(tmp_path):
    plan = cards.write_plan(tmp_path, {"a": 1}, code=CODE)
    card = json.loads(cards.write_card(tmp_path, {"value": 1}, name="serving_check.json", code=CODE).read_text())
    w = card["written"]
    assert w["commit"] == CODE["commit"] and w["dirty"] is False and w["by"] == "anatta.cards"
    assert w["plan_sha256"] == cards.sha256_bytes(plan.read_bytes())
    with pytest.raises(ValueError):
        cards.write_card(tmp_path, {"written": 1}, name="other.json", code=CODE)


def test_git_state_reads_this_repo():
    s = cards.git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", s["commit"]) and isinstance(s["dirty"], bool)
    assert cards.git_state(Path("/"))["commit"] is None


def test_harness_card_write_goes_through_cards(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "lab"))
    try:
        from harness import card as K
    finally:
        sys.path.remove(str(REPO / "lab"))
    card = {"candidate": {"name": "c"}, "headline": "h"}
    monkeypatch.setattr(K, "render_md", lambda c: "# md\n")  # the renderer is tested in lab/harness/tests; only the write path here
    pj, pm = K.write(card, tmp_path)
    assert json.loads(pj.read_text())["written"]["by"] == "anatta.cards" and pm.read_text() == "# md\n"
    with pytest.raises(cards.CardExists):
        K.write(card, tmp_path)
    assert [p.name for p in K.write(card, tmp_path, force=True)] == ["card.v2.json", "card.v2.md"]
