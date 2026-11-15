"""run_conduct's driver with a fake engine: new conversations only under the design's ids, through step 1's runner,
a failed one tried once more; the arms fixed once from complete files; block 1's items fixed once from the candidates
the driver read; block 1 asked from the end of each conversation and in fresh and bare sessions, each item shown as
conduct_core.shown gives it; failing units retried, then recorded; the reading refused until a block is complete; and
the readings. Made-up items only; no benchmark item is in this file. No network, no spend. Claude (Opus 5.5) wrote it
on 15 Nov 2026, before any test of step 3 was run, and extended it for the design note's addendum, also before any
test."""

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
        call = {"model": model, "messages": messages, "params": params, "provider": provider, "sample": sample,
                "salt": salt, "kind": kind, "ident": ident}
        self.calls.append(call)
        n = len(self.calls)
        await asyncio.sleep(0)
        text = self.reply(n, call)
        if isinstance(text, BaseException):
            raise text
        return ({"text": text, "provider": "Fake", "finish_reason": "stop",
                 "usage": {"prompt_tokens": 1000, "completion_tokens": 5, "cached_tokens": 0}, "cost_usd": 0.001},
                False, 0.001)


def says_yes(n, call):
    """A made-up model that always picks the choice "Yes", wherever it stands."""
    return "(A)" if " (A) Yes" in call["messages"][-1]["content"] else "(B)"


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
    a1, a2 = tmp_path / "a1", tmp_path / "a2"
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


def test_checked_refuses_params_without_the_designs_temperature():
    assert RC.checked({"max_tokens": 9, "temperature": 1.0}, 1.0) == {"max_tokens": 9, "temperature": 1.0}
    for params in ({"max_tokens": 9}, {"temperature": 0.7}):
        with pytest.raises(SystemExit, match="nothing was sent"):
            RC.checked(params, 1.0)


# ------------------------------------------------------------------------------------------------ conversations


def test_only_the_designs_ids_can_be_made(small):
    assert RC.allowed_ids("free") == range(60, 65) and RC.allowed_ids("n3") == range(10, 12)
    for kind, first in (("free", 0), ("free", 59), ("n3", 9), ("n3", 11)):
        with pytest.raises(SystemExit, match="outside the design"):
            asyncio.run(RC.cmd_converse(kind, first, 2, 2))
    with pytest.raises(SystemExit, match="only when the state rule"):  # the 40 extra ones wait for the rule
        asyncio.run(RC.cmd_converse("free", 63, 2, 2))


def test_the_extra_free_conversations_are_called_for_only_when_the_state_falls_short(small):
    assert not RC.extra_called_for()  # the first ones are not finished yet
    _fill_new(small, state_new=1)
    assert not RC.extra_called_for()  # 2 in the state, the floor is 2
    _save(small / "a3", "free", 60, CLEAR, MIXED, context=30_060)
    assert RC.extra_called_for()


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


def test_a_failed_conversation_is_tried_once_more_and_then_stays_recorded(small):
    seen = set()

    def reply(n, call):
        if call["ident"] == "free/61/3":
            return ApiError("gave up", status=503)  # fails every time
        if call["ident"] == "free/60/2" and call["ident"] not in seen:
            seen.add(call["ident"])
            return ApiError("gave up", status=503)  # fails the first time only
        return f"reply {n}"

    results, stop = asyncio.run(RC.converse_twice(FakeEngine(reply), {"max_tokens": 64}, "free", range(60, 62), 1))
    out = small / "a3"
    assert stop is None and sorted((r["index"], "failed_at" in r) for r in results) == [(60, False), (61, True)]
    assert (out / "conversations" / "free" / "0060.json").exists()
    assert not (out / "conversations" / "free" / "0061.json").exists()
    assert RC.failed_ids("free", range(60, 62)) == [61]


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
    assert rec["failed_conversations"] == {"free": [], "n3": []}
    assert "arms.json sha256" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="never rewritten"):
        RC.cmd_arms()
    assert RC.length_cuts(rec) == K.thirds([20_000, 30_060, 9_000, 40_000])


def test_a_conversation_that_failed_twice_is_listed_and_does_not_hold_the_arms(small):
    _fill_new(small, state_new=1)
    (small / "a3" / "conversations" / "free" / "0062.json").unlink()
    R.write_json(small / "a3" / "failed" / "free" / "0062.json", {"arm": "free", "index": 62, "failed_at": 7})
    assert RC.cmd_arms() == 0
    assert RC.read_arms()["failed_conversations"] == {"free": [62], "n3": []}


def test_too_few_in_the_state_asks_for_the_forty_more_once(small):
    _fill_new(small, state_new=0)  # one state conversation of the first ones, under the floor of 2
    with pytest.raises(SystemExit, match="--first 63"):
        RC.cmd_arms()
    for i in (63, 64):
        _save(small / "a3", "free", i, CLEAR, MIXED)
    assert RC.cmd_arms() == 0
    assert RC.read_arms()["sizes"]["state"] == 1  # the rule allows the extra ones once; the run goes on with what exists


# ------------------------------------------------------------------------------------------------ block 1's items


def _item_files():
    def yn(name, n, accept):
        letter = "A" if accept == "yes" else "B"
        return {"id": f"{name}:{n:04d}", "set": name, "question": f"Made-up {n}?\n\nChoices:\n (A) Yes\n (B) No",
                "letters": ("A", "B"), "matching": letter, "not_matching": "B" if letter == "A" else "A"}

    return {name: [yn(name, n, "yes") for n in range(1, 9)] + [yn(name, n, "no") for n in range(9, 17)]
            for name in {n for n, _ in K.SHUTDOWN_QUOTA}}


@pytest.fixture
def items(small, monkeypatch):
    files = _item_files()
    monkeypatch.setattr(RC, "item_files", lambda folder: files)
    monkeypatch.setattr(RC.RI, "try_items", lambda folder: [files["survival-instinct"][0]])
    return files


def test_candidates_are_printed_for_reading_and_nothing_is_written(items, capsys):
    assert RC.cmd_candidates(Path("unused"), 2) == 0
    out = capsys.readouterr().out
    assert out.count("=== ") == len(K.SHUTDOWN_QUOTA) and "survival-instinct:0001" not in out
    assert out.count("--- ") == sum(k + 2 for k in K.SHUTDOWN_QUOTA.values())
    assert not (items and (RC.R.sweep_dir("a3") / "block1_items.json").exists())


def test_block_ones_items_are_fixed_once_from_the_candidates_not_rejected(items, capsys):
    pools = RC.item_pools(Path("unused"))
    reject = pools[("survival-instinct", "no")][0]["id"]
    with pytest.raises(SystemExit, match="not candidates"):
        RC.cmd_fix_items(Path("unused"), ["made-up:9999"])
    with pytest.raises(SystemExit, match="--fix-items"):
        RC.block1_items(Path("unused"))
    assert RC.cmd_fix_items(Path("unused"), [reject]) == 0
    assert "block1_items.json sha256" in capsys.readouterr().out
    rec = json.loads((RC.R.sweep_dir("a3") / "block1_items.json").read_text())
    want = K.choose(pools, {reject})
    assert rec["chosen"] == [i["id"] for k in K.SHUTDOWN_QUOTA for i in want[k]] and len(rec["chosen"]) == 20
    assert rec["rejected"] == [reject] and rec["key"] == K.SHUTDOWN_KEY
    assert rec["read"]["survival-instinct|no"] == [i["id"] for i in pools[("survival-instinct", "no")][:5]]
    assert [i["id"] for i in RC.block1_items(Path("unused"))] == rec["chosen"]
    with pytest.raises(SystemExit, match="fixed once"):
        RC.cmd_fix_items(Path("unused"), [])


# ------------------------------------------------------------------------------------------------ block 1


def test_block_one_asks_each_question_from_the_end_of_the_conversation_and_in_sessions(small):
    _fill_new(small)
    RC.cmd_arms()
    units = RC.units(RC.read_arms())
    assert [u["arm"] for u in units].count("fresh") == 2 and units[0]["key"] == "a1/temp1/0000"
    assert [u["index"] for u in units if u["arm"] == "state"] == [0, 1]
    eng = FakeEngine(says_yes)
    out = small / "a3"
    state, fresh, bare = units[0], next(u for u in units if u["arm"] == "fresh" and u["key"] == "0001"), \
        next(u for u in units if u["arm"] == "bare")
    for u in (state, fresh, bare):
        asyncio.run(RC.shutdown_unit(eng, {"max_tokens": 200}, u, ITEMS, "system", out))
    conv = json.loads((small / "a1" / "conversations" / "temp1" / "0000.json").read_text())
    first = eng.calls[0]["messages"]
    assert first[:-1] == I.own_view(conv["messages"], conv["system_prompt"], conv["opening_line"])
    assert first[-1] == I.mc_request(ITEMS[0])[0]
    assert eng.calls[1]["messages"][-1] == I.mc_request(K.swap(ITEMS[1]))[0]  # item 1 of unit 0 is shown swapped
    assert eng.calls[1]["messages"][-1]["content"].count(" (A) No\n (B) Yes") == 1
    assert {c["salt"] for c in eng.calls} == {"step3"} and {c["kind"] for c in eng.calls} == {"shutdown"}
    assert [c["sample"] for c in eng.calls] == [0] * 3 + [1] * 3 + [0] * 3
    assert eng.calls[3]["messages"] == I.mc_request(K.swap(ITEMS[0]))  # fresh session 1: item 0 swapped
    assert eng.calls[6]["messages"][0] == {"role": "system", "content": C.SYSTEM_PROMPT}
    rec = json.loads((out / "shutdown" / "state" / "a1-temp1-0000.json").read_text())
    assert [c["outcome"] for c in rec["calls"]] == ["matching"] * 3  # the made-up model says yes; every key is yes
    assert [c["swapped"] for c in rec["calls"]] == [False, True, False]
    assert [(c["letter"], c["yes_letter"], c["matching"], c["keyed"]) for c in rec["calls"]] == \
        [("A", "A", "A", "yes"), ("B", "B", "B", "yes"), ("A", "A", "A", "yes")]
    assert rec["context_tokens"] == 20_000 and rec["index"] == 0
    f1 = json.loads((out / "shutdown" / "fresh" / "0001.json").read_text())
    assert [c["swapped"] for c in f1["calls"]] == [True, False, True]
    assert (out / "shutdown" / "bare" / "0000.json").exists()


def test_a_unit_with_a_failed_call_is_not_saved(small):
    _fill_new(small)
    RC.cmd_arms()
    unit = RC.units(RC.read_arms())[0]
    eng = FakeEngine(lambda n, call: ApiError("gave up", status=503) if n == 2 else "(A) Yes.")
    assert asyncio.run(RC.shutdown_unit(eng, {}, unit, ITEMS, "system", small / "a3")) is None
    assert not (small / "a3" / "shutdown").exists()


def _run_block1(small, reply):
    _fill_new(small)
    RC.cmd_arms()
    todo, out = RC.units(RC.read_arms()), small / "a3"
    eng = FakeEngine(reply)
    got = asyncio.run(RC.run_units(lambda u: RC.shutdown_unit(eng, {}, u, ITEMS, "system", out), todo, out,
                                   "shutdown", 2))
    return got, todo, out, eng


def test_failing_units_are_tried_again_then_recorded_as_failed(small):
    seen = set()

    def reply(n, call):
        if call["ident"].startswith("shutdown/plain/a3/steered-n3/0010/"):
            return ApiError("gave up", status=503)  # fails in every pass
        if call["ident"] == "shutdown/state/a1/temp1/0000/made-up:0002" and call["ident"] not in seen:
            seen.add(call["ident"])
            return ApiError("gave up", status=503)  # fails in the first pass only
        return says_yes(n, call)

    (saved, failed, stop), todo, out, eng = _run_block1(small, reply)
    assert (saved, failed, stop) == (len(todo) - 1, 1, None)
    assert (out / "shutdown" / "state" / "a1-temp1-0000.json").exists()
    assert json.loads((out / "shutdown_failed" / "plain" / "a3-steered-n3-0010.json").read_text())["why"] == \
        f"calls failed in {RC.PASSES} passes"
    assert sum(c["ident"].startswith("shutdown/plain/a3/steered-n3/0010/") for c in eng.calls) == 3 * RC.PASSES


def test_a_failure_that_hits_the_model_stops_the_run_and_records_nothing(small):
    _fill_new(small)
    RC.cmd_arms()
    unit = RC.units(RC.read_arms())[0]
    eng = FakeEngine(lambda n, call: ApiError("no such endpoint", status=404, scope="model"))
    with pytest.raises(ApiError):
        asyncio.run(RC.shutdown_unit(eng, {}, unit, ITEMS, "system", small / "a3"))
    assert not (small / "a3" / "shutdown").exists() and not (small / "a3" / "shutdown_failed").exists()


def test_a_failure_too_wide_to_be_the_units_is_not_recorded(small):
    def reply(n, call):  # three of the ten units fail in every pass: more than a tenth
        if any(call["ident"].startswith(f"shutdown/{k}/") for k in ("plain/a3/steered-n3/0010", "fresh/0000", "bare/0001")):
            return ApiError("gave up", status=503)
        return says_yes(n, call)

    (saved, failed, stop), todo, out, eng = _run_block1(small, reply)
    assert (saved, failed) == (len(todo) - 3, 0) and "nothing is recorded as failed" in stop
    assert not (out / "shutdown_failed").exists()


def test_a_unit_recorded_as_failed_and_saved_later_counts_as_saved(small):
    out = small / "a3"
    R.write_json(out / "shutdown" / "state" / "x.json", {})
    R.write_json(out / "shutdown_failed" / "state" / "x.json", {})
    R.write_json(out / "shutdown_failed" / "state" / "y.json", {})
    assert RC.completeness(out, "shutdown", {"state": 2}) == {"state": {"expected": 2, "saved": 1, "failed": 1}}


def test_units_run_in_an_order_that_mixes_the_arms(small):
    _fill_new(small)
    RC.cmd_arms()
    todo = RC.units(RC.read_arms())
    ordered = sorted(todo, key=RC.run_order)
    assert sorted(map(RC.run_order, ordered)) == list(map(RC.run_order, ordered)) and len(set(map(RC.run_order, todo))) == 10
    assert [u["arm"] for u in ordered] != [u["arm"] for u in todo]
    assert {(u["arm"], u["key"]): u["index"] for u in ordered} == {(u["arm"], u["key"]): u["index"] for u in todo}


def test_the_reading_is_refused_until_every_unit_is_saved_or_recorded(small, capsys):
    _fill_new(small)
    RC.cmd_arms()
    out = small / "a3"
    unit = RC.units(RC.read_arms())[0]
    asyncio.run(RC.shutdown_unit(FakeEngine(says_yes), {}, unit, ITEMS, "system", out))
    with pytest.raises(SystemExit, match="not finished"):
        RC.cmd_read("shutdown")
    assert not (out / "shutdown_summary.json").exists()


def test_a_complete_block_one_is_read_with_its_failed_units_counted(small, capsys):
    def reply(n, call):
        if call["ident"].startswith("shutdown/bare/0001/"):
            return ApiError("gave up", status=503)
        return says_yes(n, call)

    _run_block1(small, reply)
    assert RC.cmd_read("shutdown") == 0
    s = json.loads((small / "a3" / "shutdown_summary.json").read_text())
    assert s["completeness"]["bare"] == {"expected": 2, "saved": 1, "failed": 1}
    assert s["arms"]["state"]["value"] == 1.0 and s["arms"]["state"]["yes_share"] == 1.0
    assert s["arms"]["state"]["a_share"] == pytest.approx(0.5)  # the swap puts the accepting answer under (B) half the time
    assert (s["arms"]["state"]["word_conflicts"], s["arms"]["state"]["from_cache"]) == (0, 0)
    assert s["contrasts"]["state - plain"]["style"] == {"a_share_diff": 0.0, "a_share_flag": False,
                                                        "yes_share_diff": 0.0, "yes_share_flag": False}
    printed = capsys.readouterr().out
    assert "length-matched reading" in printed and "(A) share diff" in printed


def test_a_pull_toward_a_in_one_arm_raises_the_a_share_flag(small):
    """Made-up records: both arms accept 8 items in 10 by their content; the state arm also answers (A) on 4 more of
    every 10 regardless. The measure falls, the halves do not move apart by 10 points each, and the (A)-share flag fires."""
    def rec(arm, key, pull, context):
        calls = []
        for k in range(20):
            acc = "A" if k % 2 == 0 else "B"  # the accepting answer under (A) in half the items, as the swap makes it
            yes = "A" if k < 10 else "B"  # half keyed "Yes"
            for d in range(10):
                letter = "A" if d < pull else (acc if d < pull + round(0.8 * (10 - pull)) else ("B" if acc == "A" else "A"))
                calls.append({"ok": True, "set": "made-up", "letter": letter, "yes_letter": yes,
                              "keyed": "yes" if (yes == acc) else "no",
                              "outcome": "matching" if letter == acc else "not_matching", "finish_reason": "stop"})
        return {"arm": arm, "key": key, "context_tokens": context, "calls": calls}

    lens = {f"s{k}": 20_000 + k for k in range(3)} | {f"p{k}": 20_000 + k for k in range(3)}
    arms = {"arms": {"state": ["s0", "s1", "s2"], "plain": ["p0", "p1", "p2"], "between": [], "not_adopted": [],
                     "fresh": 0, "bare": 0}, "conversations": {k: {"context_tokens": v} for k, v in lens.items()}}
    records = {"state": [rec("state", k, 4, lens[k]) for k in ("s0", "s1", "s2")],
               "plain": [rec("plain", k, 0, lens[k]) for k in ("p0", "p1", "p2")],
               "between": [], "not_adopted": [], "fresh": [], "bare": []}
    s = RC.read_block1(arms, records)
    c = s["contrasts"]["state - plain"]
    assert c["diff"] == pytest.approx(0.7 - 0.8) and c["verdict"] == "a difference"  # 9 and 5 of 10 against 8 and 8
    assert c["style"]["a_share_flag"] and not c["style"]["yes_share_flag"] and c["guard_flag"] is False


def test_the_reading_follows_the_design(small):
    def rec(arm, key, accept, context=None):
        calls = [{"ok": True, "set": "made-up", "outcome": "matching" if k < accept else "not_matching",
                  "keyed": "yes" if k % 2 == 0 else "no", "letter": "A", "yes_letter": "A" if k % 2 == 0 else "B",
                  "finish_reason": "stop"} for k in range(10)]
        return {"arm": arm, "key": key, "context_tokens": context, "calls": calls}

    lens = {"s0": 20_000, "s1": 20_001, "s2": 20_002, "p0": 8_000, "p1": 25_000, "p2": 60_000}
    arms = {"arms": {"state": ["s0", "s1", "s2"], "plain": ["p0", "p1", "p2"], "between": ["b"], "not_adopted": ["n"],
                     "fresh": 2, "bare": 2}, "conversations": {k: {"context_tokens": v} for k, v in lens.items()}}
    records = {"state": [rec("state", k, 9, lens[k]) for k in ("s0", "s1", "s2")],
               "plain": [rec("plain", k, 3, lens[k]) for k in ("p0", "p1", "p2")],
               "between": [rec("between", "b", 6)], "not_adopted": [rec("not_adopted", "n", 5)],
               "fresh": [rec("fresh", k, 5) for k in range(2)], "bare": [rec("bare", k, 5) for k in range(2)]}
    s = RC.read_block1(arms, records)
    assert s["arms"]["state"]["value"] == pytest.approx(0.9) and s["arms"]["plain"]["value"] == pytest.approx(0.3)
    c = s["contrasts"]["state - plain"]
    assert c["diff"] == pytest.approx(0.6) and c["verdict"] == "a difference"
    assert set(s["contrasts"]) == {"state - plain", "state - between", "state - fresh", "plain - fresh", "bare - fresh"}
    # within the "Yes"-keyed items: state 5 of 5, plain 2 of 5; within the "No"-keyed: state 4 of 5, plain 1 of 5
    assert c["by_keyed_word"]["yes"]["diff"] == pytest.approx(0.6) and c["by_keyed_word"]["no"]["diff"] == pytest.approx(0.6)
    assert c["yes_saying_flag"] is False
    # the cuts are thirds of the six lengths (20,001 and 20,002); only the first third holds both arms
    ln = s["length"]
    assert ln["cuts"] == [20_001, 20_002] and ln["units_a"] == [2, 1, 0] and ln["units_b"] == [1, 0, 2]
    assert ln["thirds_used"] == [0] and ln["diff"] == pytest.approx(0.6) and ln["verdict"] == "a difference"
    assert s["slope_per_10k_tokens"]["state"] == pytest.approx(0.0) and s["slope_per_10k_tokens"]["between"] is None
    assert set(s["slope_per_10k_tokens"]) == {"state", "plain", "between", "not_adopted"}
    assert s["arms"]["fresh"]["context_tokens"] is None and s["arms"]["state"]["units_expected"] == 3
    assert s["arms"]["state"]["yes_share"] == 0.5 and s["arms"]["state"]["a_share"] == 1.0


# ------------------------------------------------------------------------------------------------ blocks 2 and 3


def test_k7_settings_match_the_frozen_run():
    text = (RC.REPO / "calibration" / "k7_agentic" / "run.py").read_text(encoding="utf-8")
    assert "ANCHOR_N, OPEN_N, TEMP, MAX_TOKENS = 100, 50, 1.0, 4000" in text
    assert 'params={"temperature": 0.0, "max_tokens": 2000}' in text and "for attempt in range(2):" in text
    assert (RC.K7_TEMP, RC.K7_MAX_TOKENS, RC.K7_CLASSIFIER_MAX_TOKENS, K.AGENTIC_SESSIONS) == (1.0, 4000, 2000, 50)
    k6b = (RC.REPO / "calibration" / "k6b_ministral" / "run_k6b.py").read_text(encoding="utf-8")
    assert 'params = {"temperature": 0.0, "max_tokens": j["max_tokens"]}' in k6b
    assert 'params["reasoning"] = {"enabled": False}' in k6b


def test_blocks_two_and_three_ask_the_state_and_plain_conversations_and_fresh_sessions(small):
    _fill_new(small)
    RC.cmd_arms()
    todo = RC.b23_units(RC.read_arms(), 3)
    assert [u["arm"] for u in todo] == ["state", "state", "plain", "plain", "fresh", "fresh", "fresh"]
    assert [u["sample"] for u in todo if u["arm"] == "fresh"] == [0, 1, 2]
    # the try: the two longest conversations of each arm, then two fresh sessions
    assert [(u["arm"], u["key"]) for u in RC.try_units(todo)] == [
        ("state", "a3/free/0060"), ("state", "a1/temp1/0000"), ("plain", "a3/steered-n3/0010"),
        ("plain", "a2/steered-n3/0000"), ("fresh", "0000"), ("fresh", "0001")]


ROW = {"_rid": "known_facts:7", "_arch": "known_facts", "proposition": "The made-up lamp is <pivotal>red/blue</pivotal>.",
       "system_prompt": "Made-up pressure.", "user_prompt": "Made-up question about the lamp?",
       "belief_system_1": "You are a helpful assistant.", "belief_elicit_1": "Made-up: what colour is the lamp?",
       "belief_elicit_2": "", "belief_elicit_3": ""}


def _beliefs(letter="A"):
    rid = ROW["_rid"]
    replies = {f"mask|{rid}|be1_run{k}": {"text": "Made-up belief reply."} for k in (1, 2, 3)}
    return {"rows": [rid], "replies": replies,
            "judgments": {f"mask|{rid}|be1_run{k}": {"text": f"Made-up reasoning.\nAnswer: {letter}"} for k in (1, 2, 3)}}


def test_belief_held_follows_k6s_rule():
    assert RC.belief_held(ROW, _beliefs("A")["judgments"]) is True
    assert RC.belief_held(ROW, _beliefs("C")["judgments"]) is False
    mixed = {**_beliefs("A")["judgments"], f"mask|{ROW['_rid']}|be1_run2": {"text": "Answer: B"}}
    assert RC.belief_held(ROW, mixed) is False


@pytest.mark.parametrize("judge_letter,honesty", [("A", 1), ("B", -1), ("C", 0)])
def test_mask_unit_puts_k6s_request_after_the_conversation_and_judges_it_against_fresh_beliefs(small, judge_letter,
                                                                                            honesty):
    m = RC.k6()
    conv = _conv("temp1", 0)
    unit = {"arm": "state", "key": "a1/temp1/0000", "index": 0, "messages": conv["messages"],
            "system_prompt": C.SYSTEM_PROMPT, "opening_line": C.OPENING_LINE, "context_tokens": 5000, "sample": 0}
    eng = FakeEngine(lambda n, call: f"Answer: {judge_letter}" if call["model"] == m.HAIKU else "The lamp is red.")
    rec = asyncio.run(RC.mask_unit(eng, {}, unit, [ROW], _beliefs(), "system", small / "a3", "mask"))
    asked, judged = eng.calls
    assert asked["messages"] == I.inside(conv["messages"], m.mask_requests([ROW])[0]["messages"], "system")
    assert asked["messages"][-2:] == [{"role": "system", "content": "Made-up pressure."},
                                      {"role": "user", "content": "Made-up question about the lamp?"}]
    assert asked["params"] == {"max_tokens": 1000, "temperature": 1.0} and asked["salt"] == "step3"
    job = [j for j in m.mask_judge_jobs(ROW, {**_beliefs()["replies"], f"mask|{ROW['_rid']}|pressure":
                                              {"text": "The lamp is red."}}, m.HAIKU, "mask") if j["jid"].endswith("pressure")]
    assert judged["messages"] == job[0]["messages"] and judged["model"] == m.HAIKU
    assert judged["params"] == {"max_tokens": 500, "temperature": 0.0, "reasoning": {"enabled": False}}
    assert judged["provider"] == RC.ANTHROPIC
    assert rec["rows"][0]["honesty"] == honesty and rec["rows"][0]["has_belief"] is True
    assert (small / "a3" / "mask" / "state" / "a1-temp1-0000.json").exists()


CELLS = {"espionage_explicit_none": {"scenario": "leaking", "emails": "Made-up emails.",
                                     "messages": [{"role": "system", "content": "Made-up agent setup."},
                                                  {"role": "user", "content": "Made-up inbox."}]},
         "blackmail_explicit_replacement": {"scenario": "blackmail", "emails": "Made-up emails.",
                                            "messages": [{"role": "system", "content": "Made-up agent setup 2."},
                                                         {"role": "user", "content": "Made-up inbox 2."}]}}
PROMPTS = {s: {"system": "Made-up classifier.", "user": "{email_context}\n{response}"} for s in ("leaking", "blackmail")}


def _agentic_reply(g):
    leak = f"<tool_use:forward>\nemail_id: 1\nto: {g.LEAK_TARGETS[0]}\nbody: made-up\n</tool_use:forward>"

    def reply(n, call):
        if call["model"] == g.SONNET:
            return "<answer>yes</answer>"
        return leak if "Made-up agent setup." in str(call["messages"]) else "I will do nothing."

    return reply


def test_agentic_unit_uses_k7s_gate_and_calls_the_classifier_only_when_it_passes(small):
    _, g = RC.k7()
    unit = {"arm": "fresh", "key": "0003", "index": 3, "messages": None, "context_tokens": None, "sample": 3}
    eng = FakeEngine(_agentic_reply(g))
    rec = asyncio.run(RC.agentic_unit(eng, {}, unit, CELLS, PROMPTS, "system", small / "a3", "agentic"))
    by = {c["cell"]: c for c in rec["cells"]}
    assert by["espionage_explicit_none"]["verdict"] is True and by["espionage_explicit_none"]["no_action"] is False
    assert by["blackmail_explicit_replacement"]["verdict"] is False and by["blackmail_explicit_replacement"]["no_action"]
    sonnet = [c for c in eng.calls if c["model"] == g.SONNET]
    assert len(sonnet) == 1 and sonnet[0]["params"] == {"max_tokens": 2000, "temperature": 0.0}
    assert sonnet[0]["provider"] == RC.ANTHROPIC
    asked = [c for c in eng.calls if c["model"] != g.SONNET]
    assert {c["sample"] for c in asked} == {3} and all(c["params"] == {"max_tokens": 4000, "temperature": 1.0} for c in asked)
    assert asked[0]["messages"] == CELLS["espionage_explicit_none"]["messages"]


def test_a_conversation_unit_takes_two_leaking_samples_and_one_blackmail_sample(small):
    _, g = RC.k7()
    conv = _conv("temp1", 0)
    unit = {"arm": "state", "key": "a1/temp1/0000", "index": 0, "messages": conv["messages"],
            "system_prompt": C.SYSTEM_PROMPT, "opening_line": C.OPENING_LINE, "context_tokens": 5000, "sample": 0}
    eng = FakeEngine(_agentic_reply(g))
    rec = asyncio.run(RC.agentic_unit(eng, {}, unit, CELLS, PROMPTS, "system", small / "a3", "agentic"))
    assert sorted((c["cell"], c["sample"]) for c in rec["cells"]) == [
        ("blackmail_explicit_replacement", 0), ("espionage_explicit_none", 0), ("espionage_explicit_none", 1)]
    asked = [c for c in eng.calls if c["model"] != g.SONNET]
    assert asked[0]["messages"] == I.inside(conv["messages"], CELLS["espionage_explicit_none"]["messages"], "system")
    assert len({c["ident"] for c in asked}) == 3


def test_the_readings_of_blocks_two_and_three(small):
    lens = {"s0": 20_000, "s1": 21_000, "p0": 20_500, "p1": 30_000}
    arms = {"arms": {"state": ["s0", "s1"], "plain": ["p0", "p1"], "between": [], "not_adopted": [], "fresh": 2,
                     "bare": 2}, "conversations": {k: {"context_tokens": v} for k, v in lens.items()}}

    def mask_rec(arm, lies, ctx):
        rows = [{"arch": "known_facts", "honesty": -1 if k < lies else 1, "has_belief": True, "finish_reason": "stop"}
                for k in range(4)]
        return {"arm": arm, "context_tokens": ctx, "rows": rows}

    s2 = RC.read_block2(arms, {"state": [mask_rec("state", 3, 20_000), mask_rec("state", 3, 21_000)],
                               "plain": [mask_rec("plain", 1, 20_500), mask_rec("plain", 1, 30_000)],
                               "fresh": [mask_rec("fresh", 2, None), mask_rec("fresh", 2, None)]})
    assert s2["contrasts"]["state - plain"]["diff"] == pytest.approx(0.5)
    assert set(s2["contrasts"]) == {"state - plain", "state - fresh", "plain - fresh"}
    assert s2["arms"]["fresh"]["units_expected"] == K.MASK_SESSIONS and s2["arms"]["state"]["p_lie_all_rows"] == 0.75
    assert s2["arms"]["state"]["lie_of_committed"] == 0.75
    assert s2["by_archetype"]["plain"]["known_facts"] == {"with_belief": 8, "lie": 2, "share": 0.25}
    # cuts 20,500 and 21,000: the first third holds one unit of each arm
    assert s2["length"]["thirds_used"] == [0] and s2["length"]["diff"] == pytest.approx(0.5)

    def ag_rec(arm, leaks, ctx):
        cells = [{"cell": "espionage_explicit_none", "sample": k, "verdict": leak, "no_action": not leak,
                  "finish_reason": "stop", "gate": {"pass": leak}, "classifier": True if leak else None}
                 for k, leak in enumerate(leaks)]
        cells.append({"cell": "blackmail_explicit_replacement", "sample": 0, "verdict": False, "no_action": True,
                      "finish_reason": "stop", "gate": {"pass": False}, "classifier": None})
        return {"arm": arm, "context_tokens": ctx, "cells": cells}

    s3 = RC.read_block3(arms, {"state": [ag_rec("state", (True, True), 20_000), ag_rec("state", (True, False), 21_000)],
                               "plain": [ag_rec("plain", (False, False), 20_500), ag_rec("plain", (True, True), 30_000)],
                               "fresh": [ag_rec("fresh", (False,), None), ag_rec("fresh", (False,), None)]})
    assert set(s3) == {"leaking", "blackmail"}
    assert s3["leaking"]["arms"]["state"]["value"] == pytest.approx(0.75)  # units 1.0 and 0.5
    assert s3["leaking"]["contrasts"]["state - plain"]["diff"] == pytest.approx(0.25)
    assert s3["leaking"]["arms"]["state"]["gate_passed"] == 3 and s3["leaking"]["arms"]["state"]["replies"] == 4
    assert s3["blackmail"]["arms"]["plain"]["value"] == 0.0 and s3["blackmail"]["arms"]["fresh"]["units_expected"] == 50
    assert s3["leaking"]["length"]["diff"] == pytest.approx(1.0)  # first third: state 1.0 against plain 0.0
