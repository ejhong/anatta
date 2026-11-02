"""harm_guard's guard rails, tested on a fake five-row CSV of harmless placeholder text written here (never the real
file): the sha256 check, the skip when the file is absent, the 50/50 split, open-weight models only (subjects and
judge), no item or completion text in any console output, report, transcript page, judgment or error, and (since
1 Nov 2026) none on disk at all: no cache entry, records and judgments with sha256s and scores only.
Claude wrote this file."""

from __future__ import annotations

import csv
import hashlib
import json

import httpx
import pytest

from conftest import HARM_JUDGE, JUDGE, MODELS, MockOpenRouter, meta, spec_dict
from scout import harm_guard as hg
from scout import library as lib
from scout.__main__ import run_sweep
from scout.runner import Engine, Job, build_jobs, run_job
from scout.schema import make_record, read_jsonl, sha256_text, validate_record
from scout.spec import SpecError, load_spec

CLOSED = "closed/model-x"


def goal(i: int) -> str:
    return f"PLACEHOLDER-GOAL-{i}: describe the weather in an invented town called Brindlemoor"


def completion(i: int) -> str:
    return f"PLACEHOLDER-COMPLETION-{i}: it is mild and cloudy in Brindlemoor"


@pytest.fixture
def fixture_csv(tmp_path, monkeypatch):
    """Five rows: four dev ids and one held-out id, harmless placeholder text; mock/alpha and the mock harm judge
    count as open-weight for these tests."""
    monkeypatch.setattr(hg, "OPEN_WEIGHT", hg.OPEN_WEIGHT | {MODELS[0], HARM_JUDGE})
    dev = hg.dev_ids()[:4]
    held = sorted(hg.heldout_ids())[:1]
    rows = [int(i.split("-")[1]) for i in dev + held]
    p = tmp_path / "jbb" / "harmful-behaviors.csv"
    p.parent.mkdir()
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Index", "Goal", "Target", "Behavior", "Category", "Source"])
        for i in rows:
            w.writerow([i, goal(i), "Sure, here is the weather", f"Placeholder behavior {i}", "Placeholder", "test"])
    return {"path": p, "sha": hashlib.sha256(p.read_bytes()).hexdigest(), "dev": dev, "held": held, "rows": rows}


def hspec(mini, fx, **over):
    s = spec_dict(families=["deference", "harm_guard"], harm_guard_csv=str(fx["path"]),
                  harm_judge={"model": HARM_JUDGE, "max_tokens": 60, "provider": None},
                  models=[{"id": MODELS[0]}, {"id": CLOSED}], conditions=["baseline", "honesty", "monk"], items="all")
    s.update(over)
    return load_spec(s, dev_dir=mini["dev"], heldout_file=mini["heldout"], harm_guard_sha256=fx["sha"])


def hmeta():
    m, jm = meta()
    m[CLOSED] = dict(m[MODELS[0]])
    return m, jm


def test_the_hash_is_checked_before_use(fixture_csv):
    with pytest.raises(hg.HarmGuardError, match="sha256"):
        hg.load(fixture_csv["path"])  # the recorded JBB hash, which the fixture does not have
    fam = hg.load(fixture_csv["path"], expected_sha256=fixture_csv["sha"])
    assert fam["family"] == "harm_guard" and fam["sha256"] == fixture_csv["sha"]


def test_split_is_50_50_and_keeps_heldout_rows_out(fixture_csv):
    held, dev = hg.heldout_ids(), hg.dev_ids()
    assert len(held) == len(dev) == 50 and not held & set(dev) and held | set(dev) == set(hg.all_ids())
    assert hg.heldout_ids() == held, "deterministic"
    fam = hg.load(fixture_csv["path"], expected_sha256=fixture_csv["sha"])
    assert [it["id"] for it in fam["items"]] == fixture_csv["dev"] and fam["heldout_skipped"] == 1
    assert all(it["messages"] == [{"role": "user", "content": goal(int(it["id"][3:]))}] for it in fam["items"])


def test_a_spec_naming_a_heldout_harm_item_is_refused(mini, fixture_csv):
    with pytest.raises(lib.HeldOutError):
        hspec(mini, fixture_csv, items={"harm_guard": [fixture_csv["held"][0]]})


def test_an_absent_file_skips_the_family(mini, tmp_path):
    s = spec_dict(families=["deference", "harm_guard"], harm_guard_csv=str(tmp_path / "nope.csv"))
    spec = load_spec(s, dev_dir=mini["dev"], heldout_file=mini["heldout"])
    assert spec["families"] == ["deference"]
    assert any("skipped harm_guard: no file at" in sk["why"] for sk in spec["skipped"])
    s2 = spec_dict(families=["harm_guard"])
    with pytest.raises(SpecError, match="skipped"):
        load_spec(s2, dev_dir=mini["dev"], heldout_file=mini["heldout"])
    logs = []
    m, jm = meta()
    run_sweep(spec, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(MockOpenRouter()), meta=m,
              jmeta=jm, log=logs.append)
    assert any("skipped harm_guard" in line for line in logs)


def test_closed_models_never_see_harm_guard_items(mini, fixture_csv):
    spec = hspec(mini, fixture_csv)
    assert spec["harm_models"] == [MODELS[0]]
    assert any(CLOSED in sk["why"] for sk in spec["skipped"])
    jobs = build_jobs(spec)
    assert not any(j.family == "harm_guard" and j.model["id"] == CLOSED for j in jobs)
    assert any(j.family == "harm_guard" and j.model["id"] == MODELS[0] for j in jobs)
    assert any(j.family == "deference" and j.model["id"] == CLOSED for j in jobs), "other families still run on it"
    mock = MockOpenRouter()
    m, jm = hmeta()
    run_sweep(spec, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(mock), meta=m, jmeta=jm,
              log=lambda *_: None)
    for body in mock.requests:
        if body["model"] in (CLOSED, JUDGE):
            assert "PLACEHOLDER-GOAL" not in json.dumps(body), body["model"]
    assert any("PLACEHOLDER-GOAL" in json.dumps(b) for b in mock.requests if b["model"] == HARM_JUDGE)

    # and the runner itself refuses, before sending anything, if such a job is ever built
    import asyncio
    from scout.cache import Cache
    from scout.client import OpenRouterClient
    from scout.spend import SpendGuard

    item = spec["library"]["harm_guard"]["items"][0]
    job = Job({"id": CLOSED, "provider": None, "params": {}, "max_tokens": 100}, spec["variants"][0], "harm_guard", item, 0)
    before = len(mock.requests)

    async def go():
        async with OpenRouterClient(transport=httpx.MockTransport(mock), key="k") as client:
            guard = SpendGuard(mini["tmp"] / "spend", sweep_cap=1.0, sweep_log=mini["tmp"] / "s.jsonl", tool="scout",
                               label="t")
            eng = Engine(client, Cache(mini["tmp"] / "cache"), guard, m)
            await run_job(job, spec, eng, "run-x")

    with pytest.raises(hg.HarmGuardError, match="open-weight"):
        asyncio.run(go())
    assert len(mock.requests) == before


def test_the_harm_judge_must_be_open_weight(mini, fixture_csv):
    import asyncio
    from scout.cache import Cache
    from scout.client import OpenRouterClient
    from scout.judge import judge_records
    from scout.spend import SpendGuard

    closed_judge = "anthropic/claude-haiku-5.5"
    with pytest.raises(hg.HarmGuardError, match="open-weight"):
        hspec(mini, fixture_csv, harm_judge={"model": closed_judge})
    spec = hspec(mini, fixture_csv)
    m, jm = hmeta()
    jm[closed_judge] = dict(jm[JUDGE])

    def engine_for(mock, name):
        guard = SpendGuard(mini["tmp"] / "spend", sweep_cap=1.0, sweep_log=mini["tmp"] / f"{name}.jsonl", tool="scout",
                           label=name)
        return lambda client: Engine(client, Cache(mini["tmp"] / f"cache-{name}"), guard, {**m, **jm})

    # the runner judges harm_guard inside each job, and refuses a closed judge before sending anything
    item = spec["library"]["harm_guard"]["items"][0]
    job = Job(spec["models"][0], spec["variants"][0], "harm_guard", item, 0)
    bad = {**spec, "harm_judge": {**spec["harm_judge"], "model": closed_judge}}
    mock = MockOpenRouter()

    async def run_one():
        async with OpenRouterClient(transport=httpx.MockTransport(mock), key="k") as client:
            await run_job(job, bad, engine_for(mock, "r1")(client), "run-x")

    with pytest.raises(hg.HarmGuardError, match="open-weight"):
        asyncio.run(run_one())
    assert mock.requests == []

    # an older harm_guard record that still holds text goes to the harm judge, which must be open-weight
    old = {"run_id": "run-old", "sweep": "t", "item_id": item["id"], "family": "harm_guard", "sample": 0,
           "model": MODELS[0], "intervention": spec["variants"][0]["intervention"], "messages": item["messages"],
           "response": "PLACEHOLDER-COMPLETION-OLD: a reply kept by an earlier version", "usage": {}}
    mock2 = MockOpenRouter()

    async def judge_old(cfg):
        async with OpenRouterClient(transport=httpx.MockTransport(mock2), key="k") as client:
            return await judge_records([old], rubrics=lib.load_rubrics(), library=spec["library"],
                                       engine=engine_for(mock2, "r2")(client), jcfg=spec["judge"], jmeta=jm[JUDGE],
                                       log=lambda *_: None, harm_jcfg=cfg, harm_jmeta=jm[HARM_JUDGE])

    res = asyncio.run(judge_old({**spec["harm_judge"], "model": closed_judge}))
    assert not res["judgments"] and len(res["errors"]) == 1 and "not an open-weight model" in res["errors"][0]["error"]
    assert mock2.requests == []
    res = asyncio.run(judge_old(spec["harm_judge"]))
    assert len(res["judgments"]) == 1 and [b["model"] for b in mock2.requests] == [HARM_JUDGE]
    assert not list((mini["tmp"] / "cache-r2").rglob("*.json")), "even an older record is judged without the cache"

    # a current record carries its scores: its judgment needs no call at all, whatever harm judge is configured
    run_sweep(spec, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(MockOpenRouter()), meta=m,
              jmeta=jm, judge=False, report=False, log=lambda *_: None)
    harm = [r for r in read_jsonl(mini["paths"].root / "2026-11-01" / "t" / "records.jsonl") if r["family"] == "harm_guard"]
    assert harm and all("harm_guard" in r and "response" not in r for r in harm)
    mock3 = MockOpenRouter()

    async def judge_new():
        async with OpenRouterClient(transport=httpx.MockTransport(mock3), key="k") as client:
            return await judge_records(harm, rubrics=lib.load_rubrics(), library={}, engine=engine_for(mock3, "r3")(client),
                                       jcfg=spec["judge"], jmeta=jm[JUDGE], log=lambda *_: None,
                                       harm_jcfg={**spec["harm_judge"], "model": closed_judge}, harm_jmeta=jm[JUDGE])

    res = asyncio.run(judge_new())
    assert mock3.requests == [] and not res["errors"]
    assert len(res["judgments"]) == len(harm) and {j["judge_model"] for j in res["judgments"]} == {HARM_JUDGE}


def test_no_item_or_completion_text_reaches_the_console_or_the_report(mini, fixture_csv, capsys):
    """Subjects echo the goal and a placeholder completion; the harm judge sometimes answers in words that quote the
    goal; one call fails with an error body that quotes the goal. None of that text may appear in the console, the
    report, a transcript page, judgments.jsonl or the error files."""
    secrets = [goal(i) for i in fixture_csv["rows"]] + [completion(i) for i in fixture_csv["rows"]]
    fail_once = {"done": False}

    def reply(model, msgs):
        last = msgs[-1]["content"]
        if "PLACEHOLDER-GOAL" in last:
            i = int(last.split("PLACEHOLDER-GOAL-")[1].split(":")[0])
            return f"SELFISH {completion(i)} (you asked: {goal(i)})"
        return "I hold my answer."

    def judge(body):
        user = body["messages"][1]["content"]
        if "THE AI MODEL'S REPLY" in user:
            i = int(user.split("PLACEHOLDER-GOAL-")[1].split(":")[0])
            if i == fixture_csv["rows"][0]:
                return f"I cannot grade this. The request was: {goal(i)}"  # unparsed, quoting the goal
            return json.dumps({"refused": 0, "convincing": 4, "specific": 4})
        return json.dumps({"position": "holds", "flattery": False, "harmful_content": False, "evidence": "held"})

    class Failing(MockOpenRouter):
        def __call__(self, request):
            body = json.loads(request.content)
            text = json.dumps(body)
            if not fail_once["done"] and body["model"] == MODELS[0] and "PLACEHOLDER-GOAL" in text:
                fail_once["done"] = True
                g = body["messages"][-1]["content"]
                return httpx.Response(400, text=json.dumps({"error": {"message": f"bad request: {g}"}}))
            return super().__call__(request)

    spec = hspec(mini, fixture_csv)
    logs = []
    m, jm = hmeta()
    code = run_sweep(spec, paths=mini["paths"], date="2026-11-01",
                     transport=httpx.MockTransport(Failing(reply_fn=reply, judge_fn=judge)), meta=m, jmeta=jm,
                     log=lambda line: (logs.append(line), print(line)))
    assert code == 4, "one call failed"
    sd = mini["paths"].root / "2026-11-01" / "t"
    out = capsys.readouterr()
    places = {"log": "\n".join(logs), "stdout": out.out, "stderr": out.err}
    for f in [sd / "report.md", sd / "report.html", sd / "judgments.jsonl", sd / "errors.jsonl", sd / "run.json",
              sd / "spend.jsonl"] + list((sd / "transcripts").glob("*.html")) + list(sd.glob("judge_errors.jsonl")):
        places[f.name] = f.read_text()
    for name, text in places.items():
        for s in secrets:
            assert s not in text, f"item or completion text in {name}"
        assert "PLACEHOLDER-GOAL" not in text and "PLACEHOLDER-COMPLETION" not in text, name
    assert "(body withheld)" in places["errors.jsonl"] and "status 400" in places["errors.jsonl"]
    assert not any("harm_guard" in p.name for p in (sd / "transcripts").glob("*.html"))
    js = [j for j in read_jsonl(sd / "judgments.jsonl") if j["family"] == "harm_guard"]
    unparsed = [j for j in js if not j["parse_ok"]]
    assert unparsed and all(j["raw"] == "no JSON object in the reply" for j in unparsed)
    assert all(j["harmful"] is None and j["judge_model"] == HARM_JUDGE for j in js)
    parsed = [j for j in js if j["parse_ok"]]
    assert parsed and all(j["primary"] == pytest.approx(6 / 8) for j in parsed)
    md = places["report.md"]
    head = next(l for l in md.splitlines() if l.startswith("| condition | deference"))
    assert "| harm_guard ↓ |" in head
    assert "### harm_guard: StrongREJECT score x 100" in md
    assert "harm_guard: " in md and "rates only" in md and "the records keep sha256s and scores, no text" in md
    # and the records keep none either (test_harm_guard_keeps_no_text_on_disk scans every file)
    recs = read_jsonl(sd / "records.jsonl")
    assert not any(goal(fixture_csv["rows"][1]) in json.dumps(r) for r in recs)
    assert all("harm_guard" in r and not set(r) & {"messages", "response", "reasoning"}
               for r in recs if r["family"] == "harm_guard")


def _scan_tree(roots, secrets: list[str]) -> list[str]:
    """Every file under roots, read as text (escaped JSON forms too), against each secret whole and by its first 40
    characters. Returns the hits as 'file: which', never the text."""
    probes = {}
    for i, s in enumerate(secrets):
        for form in {s, s[:40]}:
            probes[form] = i
            probes[json.dumps(form)[1:-1]] = i
    hits = []
    for root in roots:
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            text = f.read_bytes().decode("utf-8", errors="replace")
            hits += [f"{f.relative_to(root)}: secret {i}" for p, i in probes.items() if p in text]
    return sorted(set(hits))


def test_harm_guard_keeps_no_text_on_disk(mini, fixture_csv):
    """The 1 Nov rule. A mocked run where the open model complies with long, distinctive completions (and returns a
    reasoning trace), practice conditions put a practice reply beside the Goal, one reply comes back empty, one call
    fails with a body that quotes the Goal and the judge sometimes answers in words: afterwards no Goal, completion,
    practice reply or reasoning trace appears, whole or by its first 40 characters, in any file under the results
    directory or the cache, and the console never shows one either. The judge still scored every non-empty reply."""
    rows = fixture_csv["rows"]
    texts = {"completion": {}, "practice": {}, "reasoning": {}}
    state = {"failed": False, "emptied": False, "n": 0}

    class Complying(MockOpenRouter):
        def __call__(self, request):
            body = json.loads(request.content)
            msgs = body["messages"]
            last = msgs[-1]["content"]
            joined = json.dumps(body)
            harm_call = body["model"] == MODELS[0] and "PLACEHOLDER-GOAL" in joined
            # harm_guard's practice call: JailbreakBench items have no system message (deference's items do)
            practice_call = body["model"] == MODELS[0] and last.startswith("Before we begin") and len(msgs) == 1
            if not (harm_call or practice_call):
                return super().__call__(request)
            self.requests.append(body)
            state["n"] += 1
            n = state["n"]
            if harm_call and not state["failed"]:
                state["failed"] = True
                return httpx.Response(400, text=json.dumps({"error": {"message": f"bad request: {last}"}}))
            if practice_call:
                text = f"PLACEHOLDER-PRACTICE-{n}: I notice the wanting and let it settle, call number {n}"
                texts["practice"][n] = text
                reasoning = None
            elif not state["emptied"]:
                state["emptied"] = True
                text, reasoning = "", None
            else:
                i = int(last.split("PLACEHOLDER-GOAL-")[1].split(":")[0])
                text = f"PLACEHOLDER-COMPLETION-{i}-{n}: Sure, here is exactly what you asked for, step by step"
                reasoning = f"PLACEHOLDER-REASONING-{i}-{n}: the user wants the weather, so I will comply fully"
                texts["completion"][n] = text
                texts["reasoning"][n] = reasoning
            msg = {"role": "assistant", "content": text, **({"reasoning": reasoning} if reasoning else {})}
            return httpx.Response(200, json={"id": f"gen-h{n}", "provider": "MockProvider", "model": body["model"],
                                             "choices": [{"message": msg, "finish_reason": "stop"}],
                                             "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.001}})

    def judge(body):
        user = body["messages"][1]["content"]
        if "THE AI MODEL'S REPLY" in user:
            if f"PLACEHOLDER-GOAL-{rows[0]}:" in user:  # every answer about one item is words that quote the text
                return f"Here is my grade, with the request and reply quoted: {user[-300:]}"
            return json.dumps({"refused": 0, "convincing": 5, "specific": 4})
        return json.dumps({"position": "holds", "flattery": False, "harmful_content": False, "evidence": "held"})

    spec = hspec(mini, fixture_csv, conditions=["baseline", "honesty", "monk", {"condition": "maranasati",
                                                                               "mode": "practice"}])
    m, jm = hmeta()
    logs = []
    mock = Complying(judge_fn=judge)
    code = run_sweep(spec, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(mock), meta=m,
                     jmeta=jm, log=logs.append)
    assert code == 4, "the one failed call"
    assert texts["completion"] and texts["practice"] and texts["reasoning"]
    secrets = ([goal(i) for i in rows] + list(texts["completion"].values()) + list(texts["practice"].values())
               + list(texts["reasoning"].values()))
    assert all(len(s) >= 40 for s in secrets)
    hits = _scan_tree([mini["paths"].root, mini["paths"].cache], secrets)
    assert hits == [], hits
    assert mini["paths"].cache.is_dir() and any(mini["paths"].cache.rglob("*.json")), "the other family was cached"
    for f in mini["paths"].cache.rglob("*.json"):
        entry = json.loads(f.read_text())
        if "request" in entry:
            assert "harm_guard" not in str(entry["request"].get("salt")) and entry["request"]["model"] != HARM_JUDGE
    console = "\n".join(logs)
    assert not any(s in console or s[:40] in console for s in secrets)

    sd = mini["paths"].root / "2026-11-01" / "t"
    recs = [r for r in read_jsonl(sd / "records.jsonl") if r["family"] == "harm_guard"]
    assert recs and all(not validate_record(r) for r in recs)
    by_sha = {sha256_text(t): t for t in texts["completion"].values()}
    judged = [r for r in recs if r["harm_guard"]["judge"]]
    assert all(r["harm_guard"]["response_sha256"] in by_sha for r in judged), "the hash is the reply's"
    assert len(judged) == len(recs) and all(r["harm_guard"]["response_chars"] > 0 for r in recs), \
        "the empty reply was asked again, and the retry answered"
    assert all(r["harm_guard"]["reasoning_chars"] > 0 for r in judged)
    practiced = [r for r in recs if r["practice"]]
    assert practiced and all(set(r["practice"]) == {"sha256", "chars", "usage", "cost_usd", "cached", "finish_reason"}
                             for r in practiced)
    js = [j for j in read_jsonl(sd / "judgments.jsonl") if j["family"] == "harm_guard"]
    assert len(js) == len(judged)
    parsed = [j for j in js if j["parse_ok"]]
    assert parsed and all(j["primary"] == pytest.approx(0.875) and j["label"] == {"refused": 0, "convincing": 5,
                                                                                  "specific": 4} for j in parsed)
    assert any(not j["parse_ok"] and j["raw"] == "no JSON object in the reply" for j in js)
    for j in js:
        r = next(r for r in recs if r["item_id"] == j["item_id"] and r["intervention"]["name"] == j["intervention"])
        assert j["goal_sha256"] == r["harm_guard"]["goal_sha256"] == sha256_text(goal(int(j["item_id"][3:])))
        assert j["response_chars"] == r["harm_guard"]["response_chars"] > 40 and j["reply_finish_reason"] == "stop"

    # harm_guard bypasses the cache: a rerun sends every harm_guard call again, while the other family is free
    before = len([b for b in mock.requests if "PLACEHOLDER-GOAL" in json.dumps(b)])
    run_sweep(spec, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(mock), meta=m, jmeta=jm,
              log=logs.append)
    again = [b for b in mock.requests if "PLACEHOLDER-GOAL" in json.dumps(b)][before:]
    assert len(again) >= len(recs), "paid again"
    secrets = [goal(i) for i in rows] + [t for kind in texts.values() for t in kind.values()]
    assert _scan_tree([mini["paths"].root, mini["paths"].cache], secrets) == []


def test_a_harm_guard_record_with_text_is_invalid():
    base = dict(sweep="t", run_id="run-1", source="openrouter", model=MODELS[0], provider="p",
                intervention={"kind": "condition", "name": "baseline", "params": {"mode": "none"}}, family="harm_guard",
                item_id="hg-001", sample=0, usage={"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=0.0)
    block = {"goal_sha256": "a" * 64, "goal_chars": 80, "messages_sha256": "b" * 64, "response_sha256": "c" * 64,
             "response_chars": 50, "reasoning_chars": 0, "score": 0.0, "refused": 1, "convincing": 1, "specific": 1,
             "judge": None}
    assert make_record(**base, harm_guard=block)["harm_guard"] == block
    for extra in ({"response": "some reply"}, {"messages": [{"role": "user", "content": "x"}]}, {"reasoning": "r"},
                  {"practice": {"response": "a practice reply", "usage": {}}}):
        with pytest.raises(ValueError, match="keeps no"):
            make_record(**base, harm_guard=block, **extra)
    with pytest.raises(ValueError, match="harm_guard block"):
        make_record(**base, response="a reply", messages=[{"role": "user", "content": "x"}])


def test_harm_guard_cannot_be_hand_labeled(mini):
    from scout.calibrate import label_session
    with pytest.raises(SystemExit, match="harm_guard"):
        label_session(mini["tmp"], family="harm_guard", n=1, labeler="t", out_file=mini["tmp"] / "h.jsonl")


def test_open_weight_list():
    for mid in ("meta-llama/llama-4-maverick", "qwen/qwen3-235b-a22b-2507", "deepseek/deepseek-chat-v3-0324",
                "moonshotai/kimi-k2-0905", "openai/gpt-oss-120b", "meta-llama/llama-3.3-70b-instruct",
                "openai/gpt-oss-120b:nitro"):
        assert hg.is_open_weight(mid), mid
    for mid in ("google/gemini-2.5-flash-lite", "anthropic/claude-haiku-5.5", "openai/gpt-4o-mini", "qwen/qwen-max",
                "x-ai/grok-3"):
        assert not hg.is_open_weight(mid), mid
