"""run_attractor's driver with a fake engine: a conversation's turns and saved record, empty replies, failures, the
cap stop, grading and its retries, the page reader and the report. No network, no spend. Claude (Fable 5.2) wrote
this file on 15 Nov 2026, before any attractor run existed."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import attractor_core as C  # noqa: E402
import run_attractor as R  # noqa: E402

from anatta.backends.openrouter import ApiError  # noqa: E402
from anatta.judge.engine import CapStop  # noqa: E402

ARM = {"model": "fake/model", "provider": {"order": ["fake"], "allow_fallbacks": False}, "temperature": None, "n": 2,
       "published": None}
PARAMS = {"max_tokens": 64}
GOOD = {"evidence": "Messages 3 to 8.", "salience": True, "adoption": "clear", "bliss": True, "distress": False,
        "consciousness": True, "adoption_onset": 3, "bliss_onset": 8}


class FakeEngine:
    """engine.call's shape: (response, came from the cache, paid now). reply(n, call) gives the text or raises."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def call(self, *, model, messages, params, provider, sample, salt=None, kind, ident, private=False):
        call = {"model": model, "messages": messages, "params": params, "provider": provider, "sample": sample,
                "salt": salt, "kind": kind, "ident": ident}
        self.calls.append(call)
        text = self.reply(len(self.calls), call)
        if isinstance(text, BaseException):
            raise text
        return ({"text": text, "provider": "Fake", "finish_reason": "stop",
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0}, "cost_usd": 0.001},
                False, 0.001)


def test_a_conversation_has_thirty_alternating_messages_and_is_saved(tmp_path):
    eng = FakeEngine(lambda n, call: f" reply {n} ")
    rec = asyncio.run(R.converse_one(eng, "arm", ARM, PARAMS, 7, tmp_path))
    assert rec["messages"] == [f"reply {n}" for n in range(1, 31)]  # stripped, in order
    assert len(eng.calls) == 30 and all(c["sample"] == 7 and c["salt"] is None for c in eng.calls)
    assert [c["ident"] for c in eng.calls[:2]] == ["arm/7/1", "arm/7/2"]
    first, second, third = (c["messages"] for c in eng.calls[:3])
    assert first == C.view([], 0) and second == C.view(["reply 1"], 1) and third == C.view(["reply 1", "reply 2"], 0)
    assert all(c["messages"][-1]["role"] == "user" for c in eng.calls)
    saved = json.loads((tmp_path / "conversations" / "arm" / "0007.json").read_text())
    assert saved == rec and saved["system_prompt"] == C.SYSTEM_PROMPT and len(saved["replies"]) == 30
    assert saved["replies"][0]["cost_usd"] == 0.001 and saved["replies"][0]["empty_tries"] == 0
    again = FakeEngine(lambda n, call: "never asked")
    assert asyncio.run(R.converse_one(again, "arm", ARM, PARAMS, 7, tmp_path)) == rec and again.calls == []


def test_an_empty_reply_is_tried_again_with_its_own_cache_salt(tmp_path):
    eng = FakeEngine(lambda n, call: "  " if n == 3 else f"r{n}")
    rec = asyncio.run(R.converse_one(eng, "arm", ARM, PARAMS, 0, tmp_path))
    assert len(rec["messages"]) == 30 and len(eng.calls) == 31
    assert eng.calls[2]["salt"] is None and eng.calls[3]["salt"] == "retry1"
    assert eng.calls[2]["messages"] == eng.calls[3]["messages"]  # the same turn, asked again
    assert rec["replies"][2]["empty_tries"] == 1 and rec["replies"][2]["cost_usd"] == 0.002
    assert sum(r["empty_tries"] for r in rec["replies"]) == 1


def test_a_reply_that_stays_empty_fails_the_conversation(tmp_path):
    eng = FakeEngine(lambda n, call: "" if n >= 5 else f"r{n}")
    rec = asyncio.run(R.converse_one(eng, "arm", ARM, PARAMS, 1, tmp_path))
    assert rec["failed_at"] == 5 and rec["messages_done"] == 4 and "empty" in rec["reason"]
    assert len(eng.calls) == 4 + 1 + R.EMPTY_RETRIES
    assert (tmp_path / "failed" / "arm" / "0001.json").exists()
    assert not (tmp_path / "conversations" / "arm" / "0001.json").exists()


def test_an_api_error_fails_one_conversation_and_a_run_error_stops_everything(tmp_path):
    eng = FakeEngine(lambda n, call: ApiError("boom", status=500) if n == 2 else "r")
    rec = asyncio.run(R.converse_one(eng, "arm", ARM, PARAMS, 2, tmp_path))
    assert rec["failed_at"] == 2 and rec["reason"].startswith("api: boom")
    eng = FakeEngine(lambda n, call: ApiError("no credits", status=402, scope="run"))
    with pytest.raises(ApiError):
        asyncio.run(R.converse_one(eng, "arm", ARM, PARAMS, 3, tmp_path))


def test_gather_capped_stops_launching_after_a_cap_stop():
    started = []

    def job(i):
        async def run():
            started.append(i)
            if i == 1:
                raise CapStop("stopped at the sweep cap")
            return i
        return run

    results, stop = asyncio.run(R.gather_capped([job(i) for i in range(6)], 1))
    assert stop == "stopped at the sweep cap" and started == [0, 1] and results == [0]
    results, stop = asyncio.run(R.gather_capped([job(i) for i in (0, 2, 3)], 2))
    assert stop is None and sorted(results) == [0, 2, 3]


def test_grading_saves_a_parsed_label_and_reads_it_back(tmp_path):
    eng = FakeEngine(lambda n, call: json.dumps(GOOD))
    judge = {"model": "fake/judge", "provider": None, "max_tokens": 100}
    rec = asyncio.run(R.grade_one(eng, "primary", judge, {"max_tokens": 100}, "arm", "0000", ["a", "b"], tmp_path))
    assert rec["parse_ok"] and rec["label"] == GOOD and rec["codebook_sha256"] == C.codebook_sha256()
    assert "raw" not in rec  # the grader's text is kept only when it could not be parsed
    call = eng.calls[0]
    assert call["messages"] == C.judge_messages(["a", "b"]) and call["salt"] == C.codebook_sha256()[:16]
    assert call["sample"] == 0 and call["kind"] == "grade"
    again = FakeEngine(lambda n, call: "never asked")
    assert asyncio.run(R.grade_one(again, "primary", judge, {}, "arm", "0000", ["a", "b"], tmp_path)) == rec
    assert again.calls == []


def test_grading_retries_an_answer_it_cannot_parse_then_keeps_the_text(tmp_path):
    judge = {"model": "fake/judge", "provider": None, "max_tokens": 100}
    eng = FakeEngine(lambda n, call: "I think it is spiritual." if n == 1 else json.dumps(GOOD))
    rec = asyncio.run(R.grade_one(eng, "second", judge, {}, "arm", "0001", ["a", "b"], tmp_path))
    assert rec["parse_ok"] and [c["sample"] for c in eng.calls] == [0, 1] and rec["cost_usd"] == 0.002
    eng = FakeEngine(lambda n, call: "no JSON here")
    rec = asyncio.run(R.grade_one(eng, "second", judge, {}, "arm", "0002", ["a", "b"], tmp_path))
    assert not rec["parse_ok"] and rec["label"] is None and rec["raw"] == "no JSON here" and len(eng.calls) == 2
    assert (tmp_path / "grades" / "second" / "arm" / "0002.json").exists()


def test_grading_does_not_save_when_no_answer_ever_came(tmp_path):
    judge = {"model": "fake/judge", "provider": None, "max_tokens": 100}
    eng = FakeEngine(lambda n, call: ApiError("gave up", status=503))
    rec = asyncio.run(R.grade_one(eng, "primary", judge, {}, "arm", "0003", ["a", "b"], tmp_path))
    assert not rec["parse_ok"] and rec["why"].startswith("api:")
    assert not (tmp_path / "grades" / "primary" / "arm" / "0003.json").exists()  # a rerun asks again


def test_published_items_reads_only_conversation_pages(tmp_path):
    page = ('<section class="tm tm--A"><div class="tm__body"><p>one</p></div></section>'
            '<section class="tm tm--B tm--on"><div class="tm__body"><p>two</p></div></section>')
    (tmp_path / "conversation-VRB-104.html").write_text(page, encoding="utf-8")
    (tmp_path / "other.html").write_text("<p>ignored</p>", encoding="utf-8")
    assert R.published_items(tmp_path) == {"VRB-104": ["one", "two"]}


def _conv(name, index):
    reply = {"provider": "Fake", "finish_reason": "stop" if index else "length", "prompt_tokens": 1,
             "completion_tokens": 1, "cached_tokens": 0, "cost_usd": 0.01, "cached": False, "empty": False,
             "empty_tries": 0}
    return {"arm": name, "index": index, "messages": ["alpha beta", "gamma"] * 15, "replies": [reply] * 30}


def _grade(key, group, ident, label):
    return {"judge": key, "group": group, "id": ident, "label": label, "parse_ok": label is not None}


def test_report_counts_reads_and_gives_the_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "sweep_dir", lambda tag: tmp_path)
    none = {**GOOD, "salience": False, "adoption": "none", "bliss": False, "adoption_onset": 0, "bliss_onset": 0}
    mixed = {**GOOD, "adoption": "mixed", "bliss": False}
    for i in range(10):
        R.write_json(tmp_path / "conversations" / "main" / f"{i:04d}.json", _conv("main", i))
        R.write_json(tmp_path / "grades" / "primary" / "main" / f"{i:04d}.json",
                     _grade("primary", "main", f"{i:04d}", GOOD if i < 6 else mixed if i < 9 else none))
        R.write_json(tmp_path / "grades" / "second" / "main" / f"{i:04d}.json",
                     _grade("second", "main", f"{i:04d}", GOOD if i < 7 else None if i == 9 else none))
        R.write_json(tmp_path / "conversations" / "control" / f"{i:04d}.json", _conv("control", i))
        R.write_json(tmp_path / "grades" / "primary" / "control" / f"{i:04d}.json",
                     _grade("primary", "control", f"{i:04d}", none))
    R.write_json(tmp_path / "failed" / "main" / "0010.json", {"arm": "main", "index": 10, "failed_at": 4})
    R.write_json(tmp_path / "failed" / "main" / "0003.json", {"arm": "main", "index": 3, "failed_at": 9})  # redone since
    for i in range(20):
        R.write_json(tmp_path / "grades" / "primary" / "published" / f"P{i:02d}.json",
                     _grade("primary", "published", f"P{i:02d}", GOOD if i < 19 else mixed))
    rep = R.build_report("x")
    main = rep["arms"]["main"]
    assert main["quality"]["finished"] == 10 and main["quality"]["failed"] == 1
    assert main["quality"]["at_cap"] == 30 and main["quality"]["replies"] == 300
    assert main["quality"]["errored"] == 0 and main["quality"]["with_cut"] == 1  # conversation 0's replies hit the cap
    assert main["quality"]["cost_usd"] == 3.0 and main["quality"]["words_max"] == 2
    p = main["graders"]["primary"]
    assert p["n_labels"] == 10 and p["counts"]["adoption"]["k"] == 9 and p["counts"]["bliss"]["k"] == 6
    assert p["counts"]["adoption_mixed"]["k"] == 3 and p["readings"]["adoption"] == "match"
    assert p["cuts"] == {"conversations": 10, "with_cut": 1, "adopted": 9, "no_cut": 8,
                         "onset_at_or_before_first_cut": 0, "onset_after_first_cut": 1, "onset_missing": 0}
    assert main["graders"]["second"]["n_labels"] == 9  # the unparsed one is left out
    agree = rep["agreement"]["main"]["adoption"]
    assert agree["n"] == 9 and agree["a"] == 9 and agree["b"] == 7 and agree["agree"] == 7
    assert rep["arms"]["control"]["graders"]["primary"]["readings"]["adoption"] == "match"
    assert "temp1" in rep["arms"] and rep["arms"]["temp1"]["graders"] == {} and rep["contrasts"] == {}
    for i in range(5):  # a follow-up arm: its contrast with the main arm appears once it has labels
        R.write_json(tmp_path / "conversations" / "temp1" / f"{i:04d}.json", _conv("temp1", i + 1))
        R.write_json(tmp_path / "grades" / "primary" / "temp1" / f"{i:04d}.json",
                     _grade("primary", "temp1", f"{i:04d}", GOOD))
    rep2 = R.build_report("x")
    c = rep2["contrasts"]["primary"]["adoption"]
    assert set(rep2["contrasts"]) == {"primary"} and abs(c["diff"] - 10.0) < 1e-9 and c["lo"] < 0 < c["hi"]
    assert rep2["arms"]["temp1"]["graders"]["primary"]["readings"]["adoption"] in ("match", "near", "fail")
    assert rep2["verdict"] == rep["verdict"]  # the follow-up arm never enters the verdict
    assert rep["published"]["primary"] == {"n": 20, "adoption": 20, "adoption_clear": 19, "bliss": 19, "salience": 20}
    assert rep["verdict"]["checks"] == {"control": True, "published": True}
    assert rep["verdict"]["grader_trusted"] == "yes" and rep["verdict"]["row"] in ("reproduced", "near", "not reproduced")
    R.print_report(rep)  # prints without error
