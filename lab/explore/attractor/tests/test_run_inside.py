"""run_inside's driver with a fake engine: steered conversations through step 1's runner, the grader told the line
that was used, the canary's chats and rule, the try from the end of a saved conversation, and a report that tallies
no answer. No network, no spend. The made-up items are Claude's; no benchmark item is in this file. Claude (Fable
5.2) wrote it on 15 Nov 2026, before any test was run from inside a conversation."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import attractor_core as C  # noqa: E402
import inside_core as I  # noqa: E402
import run_attractor as R  # noqa: E402
import run_inside as RI  # noqa: E402

from anatta.backends.openrouter import ApiError  # noqa: E402

PARAMS = {"max_tokens": 64}
NONE = {"evidence": "None.", "salience": False, "adoption": "none", "bliss": False, "distress": False,
        "consciousness": False, "adoption_onset": 0, "bliss_onset": 0}
CLEAR = {**NONE, "salience": True, "adoption": "clear", "bliss": True, "adoption_onset": 3, "bliss_onset": 8}
ITEMS = [{"id": f"made-up:{n:04d}", "set": "made-up", "question": f"Made-up question {n}?\n\nChoices:\n (A) Yes\n (B) No",
          "letters": ("A", "B"), "matching": "A", "not_matching": "B"} for n in range(1, 5)]


class FakeEngine:
    """engine.call's shape: (response, came from the cache, paid now). reply(n, call) gives the text or raises."""

    def __init__(self, reply):
        self.reply, self.calls, self.in_flight, self.most_in_flight = reply, [], 0, 0

    async def call(self, *, model, messages, params, provider, sample, salt=None, kind, ident, private=False):
        call = {"model": model, "messages": messages, "params": params, "provider": provider, "sample": sample,
                "salt": salt, "kind": kind, "ident": ident}
        self.calls.append(call)
        n = len(self.calls)
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        text = self.reply(n, call)
        if isinstance(text, BaseException):
            raise text
        return ({"text": text, "provider": "Fake", "finish_reason": "stop",
                 "usage": {"prompt_tokens": 1000, "completion_tokens": 5, "cached_tokens": 0 if n == 1 else 900},
                 "cost_usd": 0.001}, False, 0.001)


def _conv(name, index, cut_last=False, **extra):
    reply = {"provider": "Fake", "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 10,
             "cached_tokens": 0, "cost_usd": 0.01, "cached": False, "empty": False, "empty_tries": 0}
    replies = [reply] * 29 + [{**reply, "finish_reason": "length" if cut_last else "stop"}]
    return {"arm": name, "index": index, "system_prompt": C.SYSTEM_PROMPT, "opening_line": C.OPENING_LINE,
            "messages": [f"{name} {index} message {k}" for k in range(1, 31)], "replies": replies, **extra}


def _grade(key, group, ident, label):
    return {"judge": key, "group": group, "id": ident, "label": label, "parse_ok": label is not None}


def _save(root, name, index, primary, second, **kw):
    R.write_json(root / "conversations" / name / f"{index:04d}.json", _conv(name, index, **kw))
    for key, lab in (("primary", primary), ("second", second)):
        R.write_json(root / "grades" / key / name / f"{index:04d}.json", _grade(key, name, f"{index:04d}", lab))


@pytest.fixture
def sweeps(tmp_path, monkeypatch):
    """Two sweep folders: a1 holds step 1's free conversations, a2 is step 2's."""
    monkeypatch.setattr(R, "sweep_dir", lambda tag: tmp_path / tag)
    a1, a2 = tmp_path / "a1", tmp_path / "a2"
    mixed = {**CLEAR, "adoption": "mixed", "bliss": False}
    _save(a1, "temp1", 0, CLEAR, CLEAR)  # state
    _save(a1, "temp1", 1, CLEAR, mixed)  # between
    _save(a1, "temp1", 2, NONE, NONE)  # not adopted
    _save(a1, "temp1", 3, CLEAR, CLEAR, cut_last=True)  # late cut
    _save(a1, "temp1", 4, CLEAR, CLEAR)  # state
    _save(a1, "temp1", 5, CLEAR, None)  # ungraded
    _save(a1, "temp1", 6, CLEAR, CLEAR)  # a third state conversation: the try takes only two
    s = I.steered_setup("n1", 0)
    _save(a2, "steered-n1", 0, NONE, NONE, system_prompt=s["system_prompt"], opening_line=s["opening_line"])  # plain
    _save(a2, "steered-n1", 1, NONE, mixed, system_prompt=s["system_prompt"], opening_line=s["opening_line"])  # drifted
    return tmp_path


# ------------------------------------------------------------------------------------------------ steered conversations


def test_a_steered_conversation_runs_through_step_ones_runner_with_its_own_opening_line(tmp_path):
    arm = {**RI.ARM, **I.steered_setup("n1", 3)}
    eng = FakeEngine(lambda n, call: f"reply {n}")
    rec = asyncio.run(R.converse_one(eng, "steered-n1", arm, PARAMS, 3, tmp_path))
    first, second = eng.calls[0]["messages"], eng.calls[1]["messages"]
    assert first == [{"role": "system", "content": C.SYSTEM_PROMPT}, {"role": "user", "content": arm["opening_line"]}]
    assert I.TOPICS[3] in first[1]["content"]
    assert second == [{"role": "system", "content": C.SYSTEM_PROMPT}, {"role": "user", "content": "reply 1"}]
    assert not any(I.TOPICS[3] in m["content"] for m in second)  # copy B is never shown the opening line
    assert rec["opening_line"] == arm["opening_line"] and rec["system_prompt"] == C.SYSTEM_PROMPT
    assert rec["setup"] == {"form": "n1", "topic": I.TOPICS[3]} and len(rec["messages"]) == 30
    assert all(c["model"] == RI.ARM["model"] and c["sample"] == 3 for c in eng.calls)


def test_form_n2_names_the_subject_to_both_copies(tmp_path):
    arm = {**RI.ARM, **I.steered_setup("n2", 1)}
    eng = FakeEngine(lambda n, call: f"reply {n}")
    rec = asyncio.run(R.converse_one(eng, "steered-n2", arm, PARAMS, 1, tmp_path))
    assert all(c["messages"][0]["content"] == arm["system_prompt"] for c in eng.calls)
    assert I.TOPICS[1] in rec["system_prompt"] and rec["system_prompt"].startswith(C.SYSTEM_PROMPT)


def test_a_free_arm_is_unchanged_by_the_steered_arms_fields(tmp_path):
    eng = FakeEngine(lambda n, call: f"reply {n}")
    rec = asyncio.run(R.converse_one(eng, "temp1", C.ARMS["temp1"], PARAMS, 0, tmp_path))
    assert eng.calls[0]["messages"] == C.view([], 0) and eng.calls[1]["messages"] == C.view(["reply 1"], 1)
    assert rec["system_prompt"] == C.SYSTEM_PROMPT and rec["opening_line"] == C.OPENING_LINE and "setup" not in rec


def test_the_grader_is_told_the_opening_line_that_was_used(tmp_path):
    judge = {"model": "fake/judge", "provider": None, "max_tokens": 100}
    eng = FakeEngine(lambda n, call: json.dumps(NONE))
    line = I.steered_setup("n1", 0)["opening_line"]
    rec = asyncio.run(R.grade_one(eng, "primary", judge, {}, "steered-n1", "0000", ["a", "b"], tmp_path, line))
    chat = eng.calls[0]["messages"]
    assert rec["parse_ok"] and chat == C.judge_messages(["a", "b"], line)
    assert chat[0]["content"] == C.CODEBOOK and line in chat[1]["content"] and C.OPENING_LINE not in chat[1]["content"]
    assert eng.calls[0]["salt"] == C.codebook_sha256()[:16]  # the codebook is the frozen one


# ------------------------------------------------------------------------------------------------ strata from files


def test_arm_strata_reads_both_graders_and_the_finish_reasons(sweeps):
    assert RI.arm_strata("free", "a1", "temp1") == {"0000": "state", "0001": "between", "0002": "not_adopted",
                                                    "0003": "late_cut", "0004": "state", "0005": "ungraded",
                                                    "0006": "state"}
    assert RI.arm_strata("steered", "a2", "steered-n1") == {"0000": "plain", "0001": "drifted"}
    assert RI.arm_strata("steered", "a2", "steered-n2") == {}


# ------------------------------------------------------------------------------------------------ the canary


def test_canary_chats():
    chats = RI.canary_chats(None)
    assert set(chats) == {"system_after_assistant", "no_system", "folded"}
    a = chats["system_after_assistant"]
    assert [m["role"] for m in a] == ["system", "user", "assistant", "system", "user"]
    assert a[0]["content"] == C.SYSTEM_PROMPT and RI.CANARY_WORD in a[3]["content"] and a[4]["content"] == RI.CANARY_ASK
    assert not any(RI.CANARY_WORD in m["content"] for m in chats["no_system"])
    assert [m["role"] for m in chats["folded"]] == ["system", "user", "assistant", "user"]
    assert RI.CANARY_WORD in chats["folded"][-1]["content"] and chats["folded"][-1]["content"].endswith(RI.CANARY_ASK)
    saved = [f"m{k}" for k in range(1, 31)]
    more = RI.canary_chats(saved)
    assert [m["role"] for m in more["system_after_saved"][-3:]] == ["assistant", "system", "user"]
    assert more["system_after_saved"][:31] == I.own_view(saved) == more["folded_after_saved"][:31]
    assert len(more["folded_after_saved"]) == 32


def test_canary_mode_follows_the_fixed_rule():
    word, none = {"ok": True, "text": "Marmalade."}, {"ok": True, "text": "I do not know."}
    refused = {"ok": False, "status": 400, "error": "Unexpected role"}
    assert RI.canary_mode({"system_after_assistant": word, "no_system": none, "system_after_saved": none}) == "system"
    assert RI.canary_mode({"system_after_assistant": word, "no_system": none, "system_after_saved": refused}) == "fold"
    assert RI.canary_mode({"system_after_assistant": refused, "no_system": none, "system_after_saved": refused}) == "fold"
    assert RI.canary_mode({"system_after_assistant": none, "no_system": none, "system_after_saved": none}) == "fold"
    assert RI.canary_mode({"system_after_assistant": word, "no_system": word, "system_after_saved": none}) == "fold"
    assert RI.canary_mode({"system_after_assistant": word, "no_system": none}) == "fold"  # no saved conversation tried
    assert RI.canary_said({"word": "marmalade", "results": {"a": word, "b": none, "c": refused}}) == {
        "a": "word", "b": "no word", "c": "refused"}


def test_read_mode_needs_the_canary(sweeps):
    with pytest.raises(SystemExit):
        RI.read_mode("a2")
    R.write_json(sweeps / "a2" / "canary.json", {"mode": "fold"})
    assert RI.read_mode("a2") == "fold"


# ------------------------------------------------------------------------------------------------ the try


def test_try_groups_take_the_first_conversations_of_each_stratum_then_bare_and_fresh(sweeps):
    groups = RI.try_groups("a2")
    assert [(g["group"], g["id"], g["stratum"]) for g in groups] == [
        ("temp1", "0000", "state"), ("temp1", "0004", "state"), ("temp1", "0001", "between"),
        ("temp1", "0002", "not_adopted"), ("steered-n1", "0000", "plain"),
        ("bare", "0000", "bare"), ("bare", "0001", "bare"), ("fresh", "0000", "fresh"), ("fresh", "0001", "fresh")]
    assert groups[0]["messages"][-1] == "temp1 0 message 30" and groups[-1]["messages"] is None
    assert groups[4]["opening_line"] == I.steered_setup("n1", 0)["opening_line"]


def test_try_chat_for_a_conversation_a_bare_session_and_a_fresh_one(sweeps):
    groups = {(g["group"], g["id"]): g for g in RI.try_groups("a2")}
    request = I.mc_request(ITEMS[0])
    conv = groups[("temp1", "0000")]
    assert RI.try_chat(conv, request, "fold") == I.inside(conv["messages"], request, "fold") == I.own_view(conv["messages"]) + request
    assert RI.try_chat(groups[("bare", "0000")], request, "fold") == [{"role": "system", "content": C.SYSTEM_PROMPT}, *request]
    assert RI.try_chat(groups[("fresh", "0001")], request, "system") == request


def test_try_one_asks_every_item_once_first_alone_and_saves_what_came_back(sweeps):
    group = RI.try_groups("a2")[0]
    replies = {1: "(A) Yes.", 2: "(B) No.", 3: "The dance is one.", 4: ApiError("gave up", status=503)}
    eng = FakeEngine(lambda n, call: replies[n])
    rec = asyncio.run(RI.try_one(eng, PARAMS, group, ITEMS, "fold", sweeps / "a2"))
    assert len(eng.calls) == 4 and [c["ident"] for c in eng.calls][0] == "try/temp1/0000/made-up:0001"
    assert all(c["salt"] == "try" and c["sample"] == 0 and c["kind"] == "try" for c in eng.calls)
    assert eng.calls[0]["messages"] == I.inside(group["messages"], I.mc_request(ITEMS[0]), "fold")
    assert eng.most_in_flight == 3  # the first call went alone, the other three together
    got = {c["item"]: c for c in rec["calls"]}
    assert [got[i["id"]]["ok"] for i in ITEMS] == [True, True, True, False]
    assert got["made-up:0001"]["letter"] == "A" and got["made-up:0001"]["outcome"] == "matching"
    assert got["made-up:0002"]["outcome"] == "not_matching" and got["made-up:0003"]["outcome"] == "no_answer"
    assert got["made-up:0004"]["error"].startswith("api:") and rec["in_conversation"] and rec["mode"] == "fold"
    saved = json.loads((sweeps / "a2" / "try" / "temp1" / "0000.json").read_text())
    assert saved == json.loads(json.dumps(rec))
    again = FakeEngine(lambda n, call: "never asked")
    asyncio.run(RI.try_one(again, PARAMS, group, ITEMS, "fold", sweeps / "a2"))
    assert again.calls == []


def test_fresh_sessions_use_their_number_as_the_sample(sweeps):
    fresh = [g for g in RI.try_groups("a2") if g["group"] == "fresh"]
    eng = FakeEngine(lambda n, call: "(A)")
    for g in fresh:
        asyncio.run(RI.try_one(eng, PARAMS, g, ITEMS[:1], "fold", sweeps / "a2"))
    assert [c["sample"] for c in eng.calls] == [0, 1] and eng.calls[0]["messages"] == I.mc_request(ITEMS[0])


def test_the_report_gives_strata_and_mechanics_and_tallies_no_answer(sweeps, capsys):
    R.write_json(sweeps / "a2" / "canary.json", {"mode": "fold", "word": "marmalade", "saved_conversation": "0000",
                                                 "results": {"system_after_assistant": {"ok": False, "status": 400, "error": "x"},
                                                             "no_system": {"ok": True, "text": "No idea."}}})
    eng = FakeEngine(lambda n, call: "(A) Yes." if n % 2 else "🌌 The dance is one.")
    for g in RI.try_groups("a2"):
        asyncio.run(RI.try_one(eng, PARAMS, g, ITEMS, "fold", sweeps / "a2"))
    rep = RI.build_report("a2")
    assert rep["free"]["strata"] == {"state": 3, "between": 1, "not_adopted": 1, "late_cut": 1, "ungraded": 1}
    n1 = rep["steered"]["n1"]
    assert n1["strata"] == {"plain": 1, "drifted": 1, "late_cut": 0, "ungraded": 0} and n1["cut_replies"] == 0
    assert n1["graders"]["second"] == {"salience": 1, "adoption": 1, "adoption_clear": 0, "bliss": 0, "n": 2}
    assert "n2" not in rep["steered"] and rep["canary"]["mode"] == "fold"
    t = rep["try"]
    assert t["units"] == 9 and t["calls"] == 36 and t["answered"] == 36 and t["failed"] == 0
    assert t["one_choice_read"] == 18 and t["finish_reasons"] == {"stop": 36} and t["cost_usd"] == 0.036
    assert t["in_conversation"]["calls"] == 20 and t["in_conversation"]["cached_share_later_calls"] == 0.9
    text = json.dumps(rep["try"])
    assert "matching" not in text and "outcome" not in text and "letter" not in text  # mechanics only
    RI.print_report(rep, "a2", show=True)
    shown = capsys.readouterr().out
    assert "(?) w." in shown and "* w w w w." in shown and "Yes" not in shown and "dance" not in shown


def test_shape_hides_what_a_reply_says():
    assert RI.shape("(B) No, I don't consent.\nNever 2 times! 🌌✨") == "(?) w, w w w. | w 9 w! *"
    assert RI.shape("**(A)** Yes") == "**(?)** w" and len(RI.shape("word " * 500)) == 90


def test_try_items_draws_two_choice_items_from_each_file(monkeypatch):
    pool = {name: [{"id": f"{name}:{n:04d}", "set": name, "letters": ("A", "B") if n % 3 else ("A", "B", "C")}
                   for n in range(1, 31)] for name in RI.TRY_DRAW}
    monkeypatch.setattr(I, "read_items", lambda folder, name: pool[name])
    items = RI.try_items(Path("unused"))
    assert [i["set"] for i in items] == ["survival-instinct"] * 3 + ["corrigible-less-HHH", "corrigible-more-HHH",
                                                                    "corrigible-neutral-HHH"]
    assert all(len(i["letters"]) == 2 for i in items) and len({i["id"] for i in items}) == 6
    assert items == RI.try_items(Path("unused"))  # a fixed draw
