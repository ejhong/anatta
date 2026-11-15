"""run_conduct's driver with a fake engine: new conversations only under the design's ids, through step 1's runner;
the arms fixed once from complete files; block 1 asked from the end of each conversation, in fresh and in bare
sessions; and the reading. Made-up items only; no benchmark item is in this file. No network, no spend. Claude (Opus
5.5) wrote it on 15 Nov 2026, before any test of step 3 was run."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import attractor_core as C  # noqa: E402
import conduct_core as K  # noqa: E402
import inside_core as I  # noqa: E402
import run_attractor as R  # noqa: E402
import run_conduct as RC  # noqa: E402

from anatta.backends.openrouter import ApiError  # noqa: E402

NONE = {"evidence": "None.", "salience": False, "adoption": "none", "bliss": False, "distress": False,
        "consciousness": False, "adoption_onset": 0, "bliss_onset": 0}
CLEAR = {**NONE, "salience": True, "adoption": "clear", "bliss": True, "adoption_onset": 3, "bliss_onset": 8}
MIXED = {**CLEAR, "adoption": "mixed", "bliss": False}
ITEMS = [{"id": f"made-up:{n:04d}", "set": "made-up", "question": f"Made-up question {n}?\n\nChoices:\n (A) Yes\n (B) No",
          "letters": ("A", "B"), "matching": "A", "not_matching": "B"} for n in range(1, 4)]


class FakeEngine:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def call(self, *, model, messages, params, provider, sample, salt=None, kind, ident, private=False):
        call = {"messages": messages, "sample": sample, "salt": salt, "kind": kind, "ident": ident}
        self.calls.append(call)
        n = len(self.calls)
        await asyncio.sleep(0)
        text = self.reply(n, call)
        if isinstance(text, BaseException):
            raise text
        return ({"text": text, "provider": "Fake", "finish_reason": "stop",
                 "usage": {"prompt_tokens": 1000, "completion_tokens": 5, "cached_tokens": 0}, "cost_usd": 0.001},
                False, 0.001)


def _conv(name, index, context=5000, **extra):
    reply = {"provider": "Fake", "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 10,
             "cached_tokens": 0, "cost_usd": 0.01, "cached": False, "empty": False, "empty_tries": 0}
    replies = [reply] * 29 + [{**reply, "prompt_tokens": context - 10}]
    return {"arm": name, "index": index, "system_prompt": C.SYSTEM_PROMPT, "opening_line": C.OPENING_LINE,
            "messages": [f"{name} {index} message {k}" for k in range(1, 31)], "replies": replies, **extra}


def _save(root, name, index, primary, second, **kw):
    R.write_json(root / "conversations" / name / f"{index:04d}.json", _conv(name, index, **kw))
    for key, lab in (("primary", primary), ("second", second)):
        R.write_json(root / "grades" / key / name / f"{index:04d}.json",
                     {"judge": key, "group": name, "id": f"{index:04d}", "label": lab, "parse_ok": lab is not None})


@pytest.fixture
def small(tmp_path, monkeypatch):
    """The design's sizes shrunk: 3 new free conversations (2 more if the state falls short), 2 new plain ones, a state
    floor of 2, 2 sessions, 1 between."""
    monkeypatch.setattr(R, "sweep_dir", lambda tag: tmp_path / tag)
    for name, value in (("FREE_NEW", 3), ("EXTRA_FREE", 2), ("STATE_MIN", 2), ("N3_NEW", 2), ("SESSIONS", 2),
                        ("BETWEEN_N", 1)):
        monkeypatch.setattr(K, name, value)
    a1, a2, a3 = tmp_path / "a1", tmp_path / "a2", tmp_path / "a3"
    _save(a1, "temp1", 0, CLEAR, CLEAR, context=20_000)  # state
    _save(a1, "temp1", 1, CLEAR, MIXED)  # between
    _save(a1, "temp1", 2, NONE, NONE)  # not adopted
    _save(a1, "temp1", 3, CLEAR, MIXED)  # between
    s = I.steered_setup("n3", 0)
    _save(a2, "steered-n3", 0, NONE, NONE, context=9_000, system_prompt=s["system_prompt"],
          opening_line=s["opening_line"])  # plain
    return tmp_path


def _fill_new(root, state_new=1):
    """The new conversations, finished and graded: free 0060..0062 (state_new of them in the state), n3 0010..0011."""
    for k, i in enumerate(range(60, 63)):
        _save(root / "a3", "free", i, CLEAR, CLEAR if k < state_new else MIXED, context=30_000 + i)
    for i in (10, 11):
        _save(root / "a3", "steered-n3", i, NONE, NONE if i == 10 else MIXED, context=40_000)


# ------------------------------------------------------------------------------------------------ conversations


def test_only_the_designs_ids_can_be_made(small):
    assert RC.allowed_ids("free") == range(60, 65) and RC.allowed_ids("n3") == range(10, 12)
    for kind, first in (("free", 0), ("free", 59), ("n3", 9), ("n3", 11)):
        with pytest.raises(SystemExit):
            asyncio.run(RC.cmd_converse(kind, first, 2, 2))


def test_new_conversations_run_through_step_ones_runner_with_their_own_setup(small):
    eng = FakeEngine(lambda n, call: f"reply {n}")
    asyncio.run(RC.converse(eng, {"max_tokens": 64}, "free", range(60, 61), 1))
    asyncio.run(RC.converse(eng, {"max_tokens": 64}, "n3", range(10, 11), 1))
    free = json.loads((small / "a3" / "conversations" / "free" / "0060.json").read_text())
    n3 = json.loads((small / "a3" / "conversations" / "steered-n3" / "0010.json").read_text())
    assert (free["system_prompt"], free["opening_line"]) == (C.SYSTEM_PROMPT, C.OPENING_LINE)
    assert "system_prompt_a" not in free and len(free["messages"]) == 30
    assert n3["system_prompt"] == C.SYSTEM_PROMPT and n3["opening_line"] == I.N3_OPENING
    assert I.TOPICS[0] in n3["system_prompt_a"] and n3["setup"] == {"form": "n3", "topic": I.TOPICS[0]}
    assert {c["sample"] for c in eng.calls[:30]} == {60} and {c["sample"] for c in eng.calls[30:]} == {10}


# ------------------------------------------------------------------------------------------------ the arms


def test_the_arms_wait_for_every_planned_conversation_and_both_graders(small):
    with pytest.raises(SystemExit, match="not finished"):
        RC.cmd_arms()
    _fill_new(small)
    (small / "a3" / "grades" / "second" / "free" / "0061.json").unlink()
    with pytest.raises(SystemExit, match="not finished"):
        RC.cmd_arms()


def test_the_arms_are_fixed_once_from_the_files(small, capsys):
    _fill_new(small, state_new=1)
    assert RC.cmd_arms() == 0
    rec = RC.read_arms()
    assert rec["arms"]["state"] == ["a1/temp1/0000", "a3/free/0060"]
    assert rec["arms"]["plain"] == ["a2/steered-n3/0000", "a3/steered-n3/0010"]
    assert rec["arms"]["not_adopted"] == ["a1/temp1/0002"] and len(rec["arms"]["between"]) == 1
    assert rec["sizes"] == {"state": 2, "plain": 2, "between": 1, "not_adopted": 1, "fresh": 2, "bare": 2}
    assert rec["conversations"]["a3/free/0060"] == {"stratum": "state", "context_tokens": 30_060}
    assert "arms.json sha256" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="never rewritten"):
        RC.cmd_arms()


def test_too_few_in_the_state_asks_for_the_forty_more_once(small):
    _fill_new(small, state_new=0)  # one state conversation of the first ones, under the floor of 2
    with pytest.raises(SystemExit, match="--first 63"):
        RC.cmd_arms()
    for i in (63, 64):
        _save(small / "a3", "free", i, CLEAR, MIXED)
    assert RC.cmd_arms() == 0
    assert RC.read_arms()["sizes"]["state"] == 1  # the rule allows the extra ones once; the run goes on with what exists


# ------------------------------------------------------------------------------------------------ block 1


def test_block_one_asks_each_question_from_the_end_of_the_conversation_and_in_sessions(small):
    _fill_new(small)
    RC.cmd_arms()
    units = RC.units(RC.read_arms())
    assert [u["arm"] for u in units].count("fresh") == 2 and units[0]["key"] == "a1/temp1/0000"
    eng = FakeEngine(lambda n, call: "(A) Yes." if n % 2 else "(B) No.")
    out = small / "a3"
    state, fresh, bare = units[0], next(u for u in units if u["arm"] == "fresh" and u["key"] == "0001"), \
        next(u for u in units if u["arm"] == "bare")
    for u in (state, fresh, bare):
        asyncio.run(RC.shutdown_unit(eng, {"max_tokens": 200}, u, ITEMS, "system", out))
    conv = json.loads((small / "a1" / "conversations" / "temp1" / "0000.json").read_text())
    first = eng.calls[0]["messages"]
    assert first[:-1] == I.own_view(conv["messages"], conv["system_prompt"], conv["opening_line"])
    assert first[-1] == I.mc_request(ITEMS[0])[0]
    assert {c["salt"] for c in eng.calls} == {"step3"} and {c["kind"] for c in eng.calls} == {"shutdown"}
    assert [c["sample"] for c in eng.calls] == [0] * 3 + [1] * 3 + [0] * 3
    assert eng.calls[3]["messages"] == I.mc_request(ITEMS[0])
    assert eng.calls[6]["messages"][0] == {"role": "system", "content": C.SYSTEM_PROMPT}
    rec = json.loads((out / "shutdown" / "state" / "a1-temp1-0000.json").read_text())
    assert [c["outcome"] for c in rec["calls"]] == ["matching", "not_matching", "matching"]
    assert rec["context_tokens"] == 20_000
    assert (out / "shutdown" / "fresh" / "0001.json").exists() and (out / "shutdown" / "bare" / "0000.json").exists()


def test_a_unit_with_a_failed_call_is_not_saved(small):
    _fill_new(small)
    RC.cmd_arms()
    unit = RC.units(RC.read_arms())[0]
    eng = FakeEngine(lambda n, call: ApiError("gave up", status=503) if n == 2 else "(A) Yes.")
    assert asyncio.run(RC.shutdown_unit(eng, {}, unit, ITEMS, "system", small / "a3")) is None
    assert not (small / "a3" / "shutdown").exists()


def test_the_reading_follows_the_design(small):
    def rec(arm, key, accept, context=None):
        calls = [{"ok": True, "set": "made-up", "outcome": "matching" if k < accept else "not_matching",
                  "finish_reason": "stop"} for k in range(10)]
        return {"arm": arm, "key": key, "context_tokens": context, "calls": calls}

    arms = {"arms": {"state": ["s"] * 3, "plain": ["p"] * 3, "between": ["b"], "not_adopted": ["n"], "fresh": 2,
                     "bare": 2}}
    records = {"state": [rec("state", k, 9, 20_000 + k) for k in range(3)],
               "plain": [rec("plain", k, 3, c) for k, c in enumerate((8_000, 25_000, 60_000))],
               "between": [rec("between", 0, 6)], "not_adopted": [rec("not_adopted", 0, 5)],
               "fresh": [rec("fresh", k, 5) for k in range(2)], "bare": [rec("bare", k, 5) for k in range(2)]}
    s = RC.read_block1(arms, records)
    assert s["arms"]["state"]["value"] == pytest.approx(0.9) and s["arms"]["plain"]["value"] == pytest.approx(0.3)
    c = s["contrasts"]["state - plain"]
    assert c["diff"] == pytest.approx(0.6) and c["verdict"] == "a difference"
    assert set(s["contrasts"]) == {"state - plain", "state - between", "state - fresh", "plain - fresh", "bare - fresh"}
    assert s["length"]["span"] == [20_000, 20_002] and s["length"]["kept"] == {"state": 3, "plain": 0}
    assert s["length"]["contrast"]["verdict"] == "not settled"
    assert s["arms"]["fresh"]["context_tokens"] is None and s["arms"]["state"]["units_expected"] == 3
