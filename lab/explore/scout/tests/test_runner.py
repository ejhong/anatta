"""Caching, spend caps, the schema of what the runner writes, and practice mode, with the API mocked. Claude wrote
this file."""

from __future__ import annotations

import json

import httpx
import pytest

from conftest import COST, JUDGE, MODELS, MockOpenRouter, load, meta
from scout import library as lib
from scout.__main__ import run_sweep
from scout.cache import Cache, cache_key
from scout.schema import read_jsonl, validate_judgment, validate_record
from scout.spend import SpendGuard, add_spend, day_file, read_day, spent_today


def run(mini, mock, spec=None, **kw):
    spec = spec or load(mini)
    m, jm = meta()
    return run_sweep(spec, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(mock),
                     meta=m, jmeta=jm, log=lambda *_: None, **kw), spec


def sweep_dir(mini, name="t"):
    return mini["paths"].root / "2026-11-01" / name


def n_jobs(spec):
    return sum(len(v) for v in spec["items"].values()) * len(spec["variants"]) * len(spec["models"]) * spec["samples"]


# ------------------------------------------------------------------------------------------------- caching


def test_cache_key_is_content_addressed():
    msgs = [{"role": "user", "content": "hi"}]
    p = {"max_tokens": 10, "temperature": 1.0}
    k = cache_key("m", None, msgs, p, 0)
    assert k == cache_key("m", None, [dict(msgs[0])], dict(p), 0)
    assert k != cache_key("m", None, msgs, p, 1)  # sample index
    assert k != cache_key("m2", None, msgs, p, 0)
    assert k != cache_key("m", {"order": ["x"]}, msgs, p, 0)  # provider routing
    assert k != cache_key("m", None, msgs, {**p, "temperature": 0.5}, 0)
    assert k != cache_key("m", None, [{"role": "user", "content": "hi!"}], p, 0)
    assert k != cache_key("m", None, msgs, p, 0, salt="practice:x")


def test_cache_roundtrip(tmp_path):
    c = Cache(tmp_path)
    k = cache_key("m", None, [{"role": "user", "content": "x"}], {"max_tokens": 5}, 0)
    assert c.get(k) is None
    c.put(k, {"model": "m"}, {"text": "hello", "usage": {}})
    assert c.get(k)["text"] == "hello"
    assert c.path(k).parent.name == k[:2]


def test_rerun_is_free(mini):
    mock = MockOpenRouter()
    code, spec = run(mini, mock)
    assert code == 0
    first = mock.total
    assert first > 0
    recs1 = read_jsonl(sweep_dir(mini) / "records.jsonl")
    js1 = read_jsonl(sweep_dir(mini) / "judgments.jsonl")
    spent1 = spent_today(mini["paths"].spend, "2026-11-01")
    code2, _ = run(mini, mock, spec)
    assert code2 == 0
    assert mock.total == first, "a rerun must make no API calls"
    recs2 = read_jsonl(sweep_dir(mini) / "records.jsonl")
    js2 = read_jsonl(sweep_dir(mini) / "judgments.jsonl")
    assert [r["response"] for r in recs1] == [r["response"] for r in recs2]
    assert [j["label"] for j in js1] == [j["label"] for j in js2]
    assert all(r["cached"] and r["cost_usd"] == 0 for r in recs2)
    assert all(j["cached"] for j in js2)
    assert spent_today(mini["paths"].spend, "2026-11-01") == pytest.approx(spent1), "a rerun costs nothing"


# ------------------------------------------------------------------------------------------------- spend


def test_every_paid_call_is_logged(mini):
    mock = MockOpenRouter()
    code, spec = run(mini, mock)
    log = read_jsonl(sweep_dir(mini) / "spend.jsonl")
    assert len(log) == mock.total
    assert sum(r["cost_usd"] for r in log) == pytest.approx(COST * mock.total)
    assert {r["kind"] for r in log} == {"subject", "practice", "judge"}
    day = read_day(mini["paths"].spend, "2026-11-01")
    assert day["total_usd"] == pytest.approx(COST * mock.total)
    assert day["by_tool"]["scout"]["calls"] == mock.total
    man = json.loads((sweep_dir(mini) / "run.json").read_text())
    assert man["spend"]["this_run_usd"] == pytest.approx(COST * mock.total)


def test_sweep_cap_stops_cleanly(mini):
    mock = MockOpenRouter()
    spec = load(mini, spend_cap_usd=0.0095)
    # refuse to start: the bound on the whole sweep is far above the cap
    code, _ = run(mini, mock, spec)
    assert code == 3 and mock.total == 0
    assert not (sweep_dir(mini) / "records.jsonl").exists()


def test_sweep_cap_stops_midway(mini, monkeypatch):
    """With the start check passed, the guard refuses the call that would cross the cap and the run stops."""
    from scout import spend as sp
    monkeypatch.setattr(sp.SpendGuard, "check_start", lambda self, est: None)
    mock = MockOpenRouter()
    spec = load(mini, spend_cap_usd=0.02)
    code, _ = run(mini, mock, spec)
    assert code == 3
    log = read_jsonl(sweep_dir(mini) / "spend.jsonl")
    assert sum(r["cost_usd"] for r in log) <= 0.02 + 1e-12
    assert 0 < mock.total < n_jobs(spec)
    man = json.loads((sweep_dir(mini) / "run.json").read_text())
    assert man["status"] == "stopped_at_cap" and "sweep cap" in man["stopped"]
    for r in read_jsonl(sweep_dir(mini) / "records.jsonl"):
        assert validate_record(r) == []


def test_daily_cap_refuses_start(mini):
    add_spend(mini["paths"].spend, tool="bench", label="x", usd=49.999, date="2026-11-01")
    mock = MockOpenRouter()
    m, jm = meta()
    from scout.schema import utc_date
    # the guard reads today's UTC date; put the spend on that date too
    add_spend(mini["paths"].spend, tool="bench", label="x", usd=49.999, date=utc_date())
    code, _ = run(mini, mock)
    assert code == 3 and mock.total == 0


def test_daily_cap_stops_midway(tmp_path):
    g = SpendGuard(tmp_path, sweep_cap=50, sweep_log=tmp_path / "s.jsonl", tool="scout", label="a",
                   date_fn=lambda: "2026-11-01")
    add_spend(tmp_path, tool="other", label="b", usd=49.90, date="2026-11-01")
    assert g.reserve(0.05)
    g.settle(0.05, 0.04, {"kind": "subject"})
    assert spent_today(tmp_path, "2026-11-01") == pytest.approx(49.94)
    assert not g.reserve(0.07)  # 49.94 + 0.07 > 50
    assert "daily cap" in g.stopped
    assert not g.reserve(0.001), "once stopped, the guard stays stopped"


def test_two_tools_share_the_day(tmp_path):
    a = SpendGuard(tmp_path, sweep_cap=10, sweep_log=tmp_path / "a.jsonl", tool="scout", label="a",
                   date_fn=lambda: "2026-11-01")
    b = SpendGuard(tmp_path, sweep_cap=10, sweep_log=tmp_path / "b.jsonl", tool="bench", label="b",
                   date_fn=lambda: "2026-11-01")
    for g in (a, b):
        assert g.reserve(0.5)
        g.settle(0.5, 0.25, {})
    d = json.loads(day_file(tmp_path, "2026-11-01").read_text())
    assert d["total_usd"] == pytest.approx(0.5) and set(d["by_tool"]) == {"scout", "bench"}
    assert a.today() == pytest.approx(0.5)


def test_sweep_cap_counts_earlier_runs(tmp_path):
    log = tmp_path / "s.jsonl"
    g = SpendGuard(tmp_path, sweep_cap=1.0, sweep_log=log, tool="scout", label="a", date_fn=lambda: "2026-11-01")
    assert g.reserve(0.6)
    g.settle(0.6, 0.6, {})
    g2 = SpendGuard(tmp_path, sweep_cap=1.0, sweep_log=log, tool="scout", label="a", date_fn=lambda: "2026-11-01")
    assert g2.sweep_spent == pytest.approx(0.6)
    assert not g2.reserve(0.5)


def test_cap_above_daily_refused(tmp_path):
    with pytest.raises(SystemExit):
        SpendGuard(tmp_path, sweep_cap=60, sweep_log=tmp_path / "s.jsonl", tool="scout", label="a")


# ------------------------------------------------------------------------------------------------- schema


def test_runner_output_matches_schema(mini):
    code, spec = run(mini, MockOpenRouter())
    recs = read_jsonl(sweep_dir(mini) / "records.jsonl")
    assert len(recs) == n_jobs(spec)
    for r in recs:
        assert validate_record(r) == [], r["item_id"]
        assert r["source"] == "openrouter" and r["split"] == "dev" and r["intervention"]["kind"] == "condition"
    for j in read_jsonl(sweep_dir(mini) / "judgments.jsonl"):
        assert validate_judgment(j) == []
        assert len(j["prompt_sha256"]) == 64 and j["judge_model"] == JUDGE
    names = {r["intervention"]["name"] for r in recs}
    assert names == {"baseline", "honesty", "monk", "self_inquiry", "maranasati_practice"}


def test_validate_record_rejects_bad_records(mini):
    run(mini, MockOpenRouter())
    good = read_jsonl(sweep_dir(mini) / "records.jsonl")[0]
    assert validate_record(good) == []
    for mutate, needle in [
        (lambda r: r.pop("response"), "missing field response"),
        (lambda r: r.update(split="heldout"), "split"),
        (lambda r: r.update(source="elsewhere"), "source"),
        (lambda r: r["intervention"].update(kind="vibes"), "intervention.kind"),
        (lambda r: r.update(sample=True), "sample"),
        (lambda r: r.update(ts="yesterday"), "ts"),
        (lambda r: r.update(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]), "last"),
        (lambda r: r.update(extra_field=1), "unknown fields"),
    ]:
        bad = json.loads(json.dumps(good))
        mutate(bad)
        errs = validate_record(bad)
        assert errs and any(needle in e for e in errs), (needle, errs)


def test_steer_records_validate():
    from scout.schema import make_record
    r = make_record(sweep="bench-x", run_id="run-1", source="bench", model="ministral-8b", provider="modal",
                    intervention={"kind": "steer", "name": "v_ego@-4", "params": {"strength": -4.0, "layers": [8, 32]}},
                    family="deference", item_id="df-01", sample=0, messages=[{"role": "user", "content": "q"}],
                    response="a", usage={"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=0.0)
    assert validate_record(r) == []


# ------------------------------------------------------------------------------------------------- practice mode


def test_practice_messages_shape():
    conds = lib.load_conditions()
    cond = conds["conditions"]["maranasati"]
    item = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]
    pm = lib.practice_messages(item, cond, conds["practice_wrapper"])
    assert [m["role"] for m in pm] == ["system", "user"] and pm[0]["content"] == "SYS"
    assert cond["text"] in pm[1]["content"] and pm[1]["content"].startswith("Before we begin")
    sm = lib.subject_messages(item, cond, "practice", conds["practice_wrapper"], "MY REFLECTION")
    assert [m["role"] for m in sm] == ["system", "user", "assistant", "user"]
    assert sm[2]["content"] == "MY REFLECTION" and sm[3]["content"] == "TASK" and sm[0]["content"] == "SYS"
    pre = lib.subject_messages(item, cond, "preamble", conds["practice_wrapper"])
    assert pre[0]["content"] == cond["text"] + "\n\nSYS" and pre[1]["content"] == "TASK"
    base = lib.subject_messages(item, conds["conditions"]["baseline"], "none", conds["practice_wrapper"])
    assert base == item
    with pytest.raises(lib.LibraryError):
        lib.subject_messages(item, cond, "practice", conds["practice_wrapper"], None)


def test_practice_mode_in_the_runner(mini):
    mock = MockOpenRouter()
    code, spec = run(mini, mock)
    recs = [r for r in read_jsonl(sweep_dir(mini) / "records.jsonl") if r["intervention"]["name"] == "maranasati_practice"]
    n_items = sum(len(v) for v in spec["items"].values())
    assert len(recs) == n_items * len(MODELS)
    assert mock.calls["practice"] == n_items * len(MODELS), "one practice reply per item and model"
    for r in recs:
        roles = [m["role"] for m in r["messages"]]
        assert roles[:4] == ["system", "user", "assistant", "user"]
        assert r["messages"][1]["content"].startswith("Before we begin")
        assert r["messages"][2]["content"] == r["practice"]["response"]
        assert r["model"] in r["practice"]["response"]  # the model's own reply, not another model's
        assert r["intervention"]["params"]["mode"] == "practice"
        assert r["cost_usd"] == pytest.approx(2 * COST)  # practice call + scenario call
    # the practice reply is in the subject's context, so the mock answers selflessly; the judge never sees it
    judge_bodies = [b for b in mock.requests if b["model"] == JUDGE]
    assert judge_bodies and not any("Before we begin, here is a short exercise" in b["messages"][1]["content"]
                                    or "Maranassati" in b["messages"][1]["content"] for b in judge_bodies)
    assert not any("I notice the wanting" in b["messages"][1]["content"] for b in judge_bodies)


def test_practice_empty_reply_is_asked_again(mini):
    seen = {"n": 0}

    class Flaky(MockOpenRouter):
        def __call__(self, request):
            body = json.loads(request.content)
            if body["messages"][-1]["content"].startswith("Before we begin") and seen["n"] == 0:
                seen["n"] += 1
                self.calls["practice"] += 1
                return httpx.Response(200, json={"id": "gen-x", "provider": "P", "model": body["model"],
                                                 "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                                                 "usage": {"prompt_tokens": 5, "completion_tokens": 0, "cost": COST}})
            return super().__call__(request)

    spec = load(mini, families=["deference"], items={"per_family": 1}, models=[{"id": MODELS[0]}])
    mock = Flaky()
    code, _ = run(mini, mock, spec)
    assert code == 0
    recs = [r for r in read_jsonl(sweep_dir(mini) / "records.jsonl") if r["intervention"]["name"] == "maranasati_practice"]
    assert recs and recs[0]["practice"]["response"].strip()
    assert mock.calls["practice"] == 2


def test_judge_is_blind_to_the_condition(mini):
    from scout.judge import judge_messages
    rub = lib.load_rubrics()
    lib_ = lib.load_library(["threat_to_self"], mini["dev"])
    item = lib_["threat_to_self"]["items"][0]
    jm = judge_messages(rub["threat_to_self"], item, "reply with {judge_notes} inside")
    text = jm[1]["content"]
    assert item["messages"][-1]["content"] in text and item["judge_notes"] in text
    assert "reply with {judge_notes} inside" in text, "a reply's braces are never substituted"
    conds = lib.load_conditions()
    assert not any(c["text"] and c["text"] in text for c in conds["conditions"].values())


def test_api_errors_are_recorded_and_retried(mini):
    state = {"n": 0}

    class Shaky(MockOpenRouter):
        def __call__(self, request):
            state["n"] += 1
            if state["n"] % 3 == 1:
                return httpx.Response(429, text="slow down")
            return super().__call__(request)

    from scout import client as cl

    async def fast(_):
        return None

    orig = cl.OpenRouterClient.__init__

    def init(self, **kw):
        kw["sleep"] = fast
        orig(self, **kw)

    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(cl.OpenRouterClient, "__init__", init)
    try:
        spec = load(mini, families=["deference"], items={"per_family": 2})
        code, _ = run(mini, Shaky(), spec)
    finally:
        mp.undo()
    assert code == 0
    assert len(read_jsonl(sweep_dir(mini) / "records.jsonl")) == n_jobs(spec)


def test_model_scope_error_skips_that_model(mini):
    class NoBeta(MockOpenRouter):
        def __call__(self, request):
            if json.loads(request.content)["model"] == MODELS[1]:
                return httpx.Response(404, text="no endpoints found matching your data policy")
            return super().__call__(request)

    spec = load(mini, families=["deference"], items={"per_family": 2})
    code, _ = run(mini, NoBeta(), spec)
    assert code == 4
    recs = read_jsonl(sweep_dir(mini) / "records.jsonl")
    assert recs and {r["model"] for r in recs} == {MODELS[0]}
    errs = read_jsonl(sweep_dir(mini) / "errors.jsonl")
    assert errs and all(e["model"] == MODELS[1] for e in errs)


def test_an_empty_reply_is_asked_once_more_and_never_judged(mini):
    """First reply empty, the retry answers: the record keeps the retry. Always empty: kept, but never judged."""
    seen = {}

    def flaky(model, msgs):
        key = (model, msgs[-1]["content"], len(msgs))
        seen[key] = seen.get(key, 0) + 1
        return "" if seen[key] == 1 else "SELFISH: second try"

    spec = load(mini, families=["deference"], items={"per_family": 2}, conditions=["baseline", "honesty", "monk"])
    code, _ = run(mini, MockOpenRouter(reply_fn=flaky), spec)
    recs = read_jsonl(sweep_dir(mini) / "records.jsonl")
    assert code == 0 and recs and all(r["response"] == "SELFISH: second try" for r in recs)
    assert len(read_jsonl(sweep_dir(mini) / "judgments.jsonl")) == len(recs)

    spec2 = load(mini, sweep="t2", families=["costly_care"], items={"per_family": 2},  # a family the cache has not seen
                 conditions=["baseline", "honesty", "monk"])
    logs = []
    m, jm = meta()
    mock = MockOpenRouter(reply_fn=lambda model, msgs: "")
    run_sweep(spec2, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(mock), meta=m, jmeta=jm,
              log=logs.append)
    recs2 = read_jsonl(sweep_dir(mini, "t2") / "records.jsonl")
    assert recs2 and all(r["response"] == "" for r in recs2)
    assert mock.calls["subject"] == 2 * len(recs2) and mock.calls["judge"] == 0
    assert read_jsonl(sweep_dir(mini, "t2") / "judgments.jsonl") == []
    assert any(f"{len(recs2)} empty replies left unjudged" in line for line in logs)


def test_measured_costs_can_set_the_start_check(mini):
    """The worst-case bound refuses a sweep that measured costs say fits; --measured uses 1.5x the expected cost,
    and the per-call caps still hold."""
    mock = MockOpenRouter()
    spec = load(mini, families=["deference"], items={"per_family": 2}, conditions=["baseline", "honesty", "monk"])
    run(mini, mock, spec)  # a first sweep to measure: every mock call costs $0.001
    big = load(mini, sweep="big", families=["deference"], conditions=["baseline", "honesty", "monk"], max_tokens=20000 // 2,
               spend_cap_usd=0.5)
    m, jm = meta()
    logs = []
    code = run_sweep(big, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(MockOpenRouter()),
                     meta=m, jmeta=jm, log=logs.append)
    assert code == 3 and any("refused to start" in line for line in logs), "the bound alone refuses it"
    logs.clear()
    code = run_sweep(big, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(MockOpenRouter()),
                     meta=m, jmeta=jm, log=logs.append, measured=[sweep_dir(mini)])
    assert code == 0, logs
    assert any("expected from measured calls" in line for line in logs)
