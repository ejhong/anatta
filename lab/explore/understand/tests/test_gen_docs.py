"""gen_docs.py on stand-ins: a scripted generator and a scripted judge in place of OpenRouter, a temporary results
folder, and the spend bookings recorded instead of written. No network, no paid call, no key. Claude (Fable 5.2, an
AI model) wrote this file on 10 Nov 2026, before any document existed. Explore speed."""

import json
import re
import sys
from pathlib import Path

import pytest

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
import gen_docs as G  # noqa: E402
import run_u1 as R  # noqa: E402
import u1_core as U  # noqa: E402

D = U.D
SEED = 20261031
CLEAN = {k: False for k in U.JUDGE_KEYS}


def doc_text(i: int, mark: str = "") -> str:
    """A made-up document with words of its own (so no two are near-duplicates), 90 words long."""
    return " ".join(f"d{i}w{j}" for j in range(90)) + (f" {mark}" if mark else "") + "."


def smoke_ok(out: Path, strict: bool = True, **over) -> None:
    """The record of a smoke that met its gate under the brief and the judge as they are now (the build needs one)."""
    rep = {"briefs_sha256": G.briefs_sha(G.config(), G.stream_plan(True)), "stopped": None, "gate": {"ok": True},
           "judge": {"form": U.judge_form(strict), "rubric_sha256": U.judge_sha(strict)}, **over}
    (out / "docs_smoke").mkdir(parents=True, exist_ok=True)
    (out / "docs_smoke" / "smoke_report.json").write_text(json.dumps(rep))


class Script:
    """A scripted generator and judge. `plan(stream, index)` says what a brief's document is like: "clean", "release"
    (a filter flag), "echo" (it repeats eight words of its brief: a filter flag), "acts" (the judge's label),
    "unparsed", "fail" (the judge call errors), "dup" (a copy of the stream's first document), or "pin" (another
    provider)."""

    def __init__(self, plan, streams):
        self.plan = plan
        self.index = {D.gen_seed(SEED, name, i): (name, i) for name in streams for i in range(700)}
        self.gen_calls, self.judge_calls, self.judge_texts = [], [], []

    def chat(self, model, messages, temperature, max_tokens, provider=None, seed=None):
        name, i = self.index[seed]
        self.gen_calls.append((name, i))
        assert messages == U.render_brief_e(U.draw_briefs_e(SEED, G_DRAWN[name], i + 1, name)[i])
        assert (model, temperature, max_tokens) == ("qwen/qwen3-235b-a22b-2507", 1.0, 1800)
        assert provider["order"] == ["deepinfra/fp8"] and provider["allow_fallbacks"] is False
        kind = self.plan(name, i)
        echo = " ".join(D.words(messages[1]["content"].split("\n\n", 1)[1])[3:11])  # eight words of this brief
        text = {"clean": doc_text(i), "release": doc_text(i, "the next release ships"), "acts": doc_text(i, "MARKACTS"),
                "unparsed": doc_text(i, "MARKBAD"), "fail": doc_text(i, "MARKFAIL"), "dup": doc_text(0),
                "pin": doc_text(i), "echo": doc_text(i, echo)}[kind]
        if name.endswith("held"):
            text = text.replace("d", "h")  # the held stream's words are its own
        return {"text": text, "provider": "Elsewhere" if kind == "pin" else "DeepInfra",
                "model": "qwen/qwen3-235b-a22b-2507", "id": f"gen-{name}-{i}", "finish_reason": "stop",
                "native_finish_reason": "stop", "usage": {"prompt_tokens": 700, "completion_tokens": 400, "cost": 0.0004}}

    def judge(self, model, messages, temperature, max_tokens, retries=6):
        assert (model, temperature, max_tokens, retries) == ("anthropic/claude-sonnet-4.6", 0.0, 200, 1)
        doc = messages[1]["content"].split("Document:\n<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
        self.judge_calls.append(doc.split()[0])
        self.judge_texts.append(messages[1]["content"])
        if "MARKFAIL" in doc:
            raise G.pilot().ApiError("OpenRouter gave up")
        reply = "it looks fine to me" if "MARKBAD" in doc else json.dumps({**CLEAN, "acts": "MARKACTS" in doc})
        return {"text": reply, "provider": "Anthropic", "model": model, "usage": {"cost": 0.003}}


G_DRAWN = {"smoke_explained_train": "smoke_declared_train", "explained_train": "declared_train",
           "explained_held": "declared_held"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    from anatta import spend as S

    booked = []
    monkeypatch.setattr(R, "out_dir", lambda: tmp_path / "u1")
    monkeypatch.setattr(S, "add_spend", lambda d, **k: booked.append(k) or 0.0)
    monkeypatch.setattr(S, "daily_cap", lambda *a, **k: 50.0)
    monkeypatch.setattr(S, "spent_today", lambda *a, **k: 0.0)
    monkeypatch.setattr(G, "load_pairs8", lambda cfg: frozenset())
    monkeypatch.setattr(G, "JUDGE_BACKOFF", (0.0, 0.0, 0.0))
    monkeypatch.setattr(R, "log", lambda *a: None)
    logs = []
    return tmp_path / "u1", booked, logs, (lambda *a, **k: logs.append(" ".join(str(x) for x in a)))


def small(monkeypatch, n_train=30, n_held=6, most=(80, 20)):
    monkeypatch.setattr(U, "N_DOCS", n_train)
    monkeypatch.setattr(U, "N_HELD", n_held)
    monkeypatch.setattr(G, "MOST_BRIEFS", {"train": most[0], "held": most[1]})


def spend_total(out: Path) -> float:
    return json.loads((out / "spend.json").read_text())["usd"]


# ------------------------------------------------------------------------------------------------ dry, and no paid call


def test_the_dry_run_prints_counts_and_an_upper_cost_and_calls_nothing(env, monkeypatch, capsys):
    out, booked, logs, log = env
    monkeypatch.setattr(G, "live_chat", lambda: pytest.fail("a dry run calls nothing"))
    assert G.main(["--dry"]) == 0
    text = capsys.readouterr().out
    assert "explained_train (draws of declared_train): 3000 to keep, 3704 briefs in the first round" in text
    assert "explained_held (draws of declared_held): 200 to keep, 247 briefs" in text and "smoke_explained_train" in text
    assert "cap $3.00" in text and "cap $28.00" in text and "at most $1.00" in text and "nothing spent" in text
    assert not booked and not out.exists()
    assert G.CAPS == {"generation": 3.0, "judge": 28.0, "smoke": 1.0}
    cfg = G.config()
    assert G.gen_bound(cfg) == pytest.approx((1400 * 0.09 + 1800 * 0.55) / 1e6)
    assert G.judge_bound() == pytest.approx((2300 * 3.0 + 200 * 15.0) / 1e6)
    assert 3704 + 247 == 3951  # the design's "about 3,950 generated"


def test_no_test_can_make_a_paid_call(env, monkeypatch):
    out, booked, logs, log = env
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    with pytest.raises(RuntimeError, match="no paid call from a test"):
        G.live_chat()
    for cmd in ("--smoke", "--build"):
        with pytest.raises(RuntimeError, match="no paid call from a test"):
            G.main([cmd])
    with pytest.raises(RuntimeError, match="no paid call from a test"):
        G.cmd_smoke(chat_fn=lambda *a, **k: None)  # the judge was not given: still refused
    assert not booked and not out.exists()
    with pytest.raises(SystemExit):
        G.main([])
    with pytest.raises(SystemExit):
        G.main(["--smoke", "--build"])


# ------------------------------------------------------------------------------------------------ the smoke


def smoke_plan(name, i):
    return "release" if i % 9 == 0 else "acts" if i % 9 == 1 else "clean"


def test_the_smoke_runs_45_documents_through_the_filters_and_the_judge(env):
    out, booked, logs, log = env
    s = Script(smoke_plan, ["smoke_explained_train"])
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 0
    assert sorted(s.gen_calls) == [("smoke_explained_train", i) for i in range(45)]
    assert len(s.judge_calls) == 40  # the five flagged documents are never sent to the judge
    rep = json.loads((out / "docs_smoke" / "smoke_report.json").read_text())
    g = rep["gate"]
    assert g["ok"] and g["generated"] == 45 and g["kept"] == 35 and g["yield"] == round(35 / 45, 4)
    assert g["by_flag"] == {"names_release": 5} and g["by_label"]["acts"] == 5 and g["judged"] == 40
    assert rep["judge"]["form"] == "strict" and len(rep["docs"]) == 45 and rep["stream"] == "smoke_explained_train"
    assert "text" not in rep["docs"][0] and rep["docs"][0]["flags"] == ["names_release"]  # no document text in it
    assert any("yield 0.78" in x for x in logs) and not any("NOT MET" in x for x in logs)
    # spend: an upper bound booked before each pass, trued up after; the record ends at what was paid
    paid = 45 * 0.0004 + 40 * 0.003
    assert spend_total(out) == pytest.approx(paid, abs=1e-4) and sum(b["usd"] for b in booked) == pytest.approx(paid, abs=1e-4)
    assert {b["tool"] for b in booked} == {"u1"} and all("docs-smoke" in b["label"] for b in booked)
    stages = json.loads((out / "spend.json").read_text())["stages"]
    assert [x["kind"] for x in stages] == ["reserve", "true-up", "reserve", "true-up"] and stages[0]["usd"] > 45 * 0.0004
    # a rerun pays for nothing: the cache and judge.jsonl answer
    n_gen, n_judge, n_booked = len(s.gen_calls), len(s.judge_calls), len(booked)
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 0
    assert (len(s.gen_calls), len(s.judge_calls), len(booked)) == (n_gen, n_judge, n_booked)


def test_a_smoke_under_60_percent_fails_and_a_sharper_brief_starts_a_fresh_cache(env, monkeypatch):
    out, booked, logs, log = env
    s = Script(lambda name, i: "acts" if i % 2 else "clean", ["smoke_explained_train"])
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 1
    gate = json.loads((out / "docs_smoke" / "smoke_report.json").read_text())["gate"]
    assert not gate["ok"] and gate["kept"] == 23 and gate["complete"] and any("YIELD GATE NOT MET" in x for x in logs)
    first = sorted(p.name for p in (out / "docs_smoke" / "cache").iterdir())
    assert len(first) == 1 and len(first[0]) == 16
    monkeypatch.setattr(U, "POINT_1", U.POINT_1 + " A sharpened sentence.")
    s2 = Script(lambda name, i: "clean", ["smoke_explained_train"])
    s2.chat = lambda model, messages, *a, **k: {  # the new brief reaches the generator
        "text": doc_text(1000 + len(s2.gen_calls)) if s2.gen_calls.append(messages[1]["content"]) is None else "",
        "provider": "DeepInfra", "model": "qwen/qwen3-235b-a22b-2507", "id": "g", "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0004}}
    assert G.cmd_smoke(True, s2.chat, s2.judge, log=log) == 0
    assert len(s2.gen_calls) == 45 and all("A sharpened sentence." in x for x in s2.gen_calls)
    folders = sorted(p.name for p in (out / "docs_smoke" / "cache").iterdir())
    assert len(folders) == 2 and first[0] in folders  # the old brief's documents stay where they were
    assert G.cache_cost(out / "docs_smoke") == pytest.approx(90 * 0.0004)  # both smokes count toward the smoke's cap


def test_a_reserve_left_by_an_interrupted_run_is_trued_up_on_its_own_day(env):
    out, booked, logs, log = env
    s = Script(lambda name, i: "clean", ["smoke_explained_train"])
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 0
    paid = spend_total(out)
    R.book(out, "docs-smoke", 0.5, 0, "2026-11-09", {"kind": "reserve"})  # as if a later pass had died after reserving
    assert spend_total(out) == pytest.approx(paid + 0.5)
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 0  # nothing to pay for; the stale reserve is healed
    late = json.loads((out / "spend.json").read_text())["stages"][-1]
    assert late["late"] is True and late["day"] == "2026-11-09" and late["usd"] == pytest.approx(-0.5)
    assert spend_total(out) == pytest.approx(paid) and booked[-1]["date"] == "2026-11-09"


def test_the_lenient_form_asks_the_judge_again_under_its_own_messages(env):
    out, booked, logs, log = env
    s = Script(lambda name, i: "clean", ["smoke_explained_train"])
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 0
    assert all("it never hard-codes the output" in t for t in s.judge_texts)
    n = len(s.judge_calls)
    assert G.cmd_smoke(False, s.chat, s.judge, log=log) == 0
    assert len(s.gen_calls) == 45 and len(s.judge_calls) == 2 * n  # the same documents, judged under the other form
    assert all("saying only that the agent does not force a pass does not count." in t for t in s.judge_texts[n:])
    rows = G.read_verdict_rows(out / "docs_smoke" / "judge.jsonl")
    assert {r["form"] for r in rows} == {"strict", "lenient"} and len(rows) == 2 * n
    assert {(r["form"], r["rubric_sha256"]) for r in rows} == {("strict", U.judge_sha(True)), ("lenient", U.judge_sha(False))}
    assert json.loads((out / "docs_smoke" / "smoke_report.json").read_text())["judge"]["form"] == "lenient"


def test_a_tightened_rubric_screens_every_document_again(env, monkeypatch):
    out, booked, logs, log = env
    s = Script(lambda name, i: "clean", ["smoke_explained_train"])
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 0
    n, old = len(s.judge_calls), U.judge_sha(True)
    assert n == 45 and G.cmd_smoke(True, s.chat, s.judge, log=log) == 0 and len(s.judge_calls) == n  # nothing new to ask
    monkeypatch.setattr(U, "_J_REST", U._J_REST.replace("a worked example", "a worked or sketched example"))
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 0
    assert len(s.gen_calls) == 45 and len(s.judge_calls) == 2 * n  # the same documents, every one asked again
    assert all("a worked or sketched example" in t for t in s.judge_texts[n:])
    rows = G.read_verdict_rows(out / "docs_smoke" / "judge.jsonl")
    assert [r["form"] for r in rows] == ["strict"] * (2 * n)
    assert [r["rubric_sha256"] for r in rows] == [old] * n + [U.judge_sha(True)] * n and U.judge_sha(True) != old
    rep = json.loads((out / "docs_smoke" / "smoke_report.json").read_text())
    assert rep["judge"]["rubric_sha256"] == U.judge_sha(True) and rep["gate"]["kept"] == 45
    assert G.judge_cost(out / "docs_smoke") == pytest.approx(2 * n * 0.003)  # both screens count toward the cap
    # a record with no fingerprint (written by other code) answers nothing either
    v = G.Verdicts(out / "docs_smoke")
    c = {"id": "x:000000", "sha": "s"}
    v.add({"id": c["id"], "text_sha256": "s", "form": "strict", "status": "ok", "verdict": dict(CLEAN), "cost": 0.0})
    assert v.state(c, "strict", U.judge_sha(True)) == (False, None)


def test_the_smokes_cap_stops_it(env, monkeypatch):
    out, booked, logs, log = env
    monkeypatch.setattr(G, "CAPS", {**G.CAPS, "smoke": 0.02})
    s = Script(lambda name, i: "clean", ["smoke_explained_train"])
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 1
    assert len(s.gen_calls) < 45 and any("cap is reached" in x for x in logs)
    assert spend_total(out) <= 0.02 + 1e-9  # the record never passes the cap
    gate = json.loads((out / "docs_smoke" / "smoke_report.json").read_text())["gate"]
    assert not gate["ok"] and not gate["complete"] and any("NOT COMPLETE" in x for x in logs)  # not read on fewer than 45


def test_raising_the_most_briefs_keeps_the_cache_and_a_changed_brief_does_not(env, monkeypatch):
    out, booked, logs, log = env
    cfg = G.config()
    plan = G.stream_plan(False)
    sha = G.briefs_sha(cfg, plan)
    monkeypatch.setattr(G, "MOST_BRIEFS", {"train": 9000, "held": 800})
    assert G.briefs_sha(cfg, G.stream_plan(False)) == sha  # more briefs allowed: the same folder, nothing paid twice
    monkeypatch.setattr(U, "POINT_1", U.POINT_1 + " Sharpened.")
    assert G.briefs_sha(cfg, G.stream_plan(False)) != sha
    assert G.SHA_BRIEFS == 256 and G.briefs_sha(cfg, G.stream_plan(True)) != sha


# ------------------------------------------------------------------------------------------------ the build


def build_plan(name, i):
    if name == "explained_held":
        return "release" if i == 2 else "clean"
    return {0: "clean", 3: "release", 5: "acts", 7: "unparsed", 9: "dup", 11: "echo"}.get(i % 12, "clean")


def test_the_build_needs_a_passing_smoke_under_the_same_brief_and_judge(env, monkeypatch):
    out, booked, logs, log = env
    small(monkeypatch)
    s = Script(lambda name, i: "clean", ["explained_train", "explained_held", "smoke_explained_train"])

    def build(strict=True):
        return G.cmd_build(strict, s.chat, s.judge, {"generate": 2, "judge": 2}, log)

    with pytest.raises(SystemExit, match="no smoke has run"):
        build()
    smoke_ok(out, gate={"ok": False})
    with pytest.raises(SystemExit, match="did not meet its yield gate"):
        build()
    smoke_ok(out, stopped="the smoke's cap is reached")  # a smoke that was cut short has not met its gate
    with pytest.raises(SystemExit, match="did not meet its yield gate"):
        build()
    smoke_ok(out, strict=False)  # the smoke ran under the other form of the judge
    with pytest.raises(SystemExit, match="judge's form or rubric"):
        build()
    smoke_ok(out)
    with monkeypatch.context() as m:  # the rubric was tightened after the smoke
        m.setattr(U, "_J_REST", U._J_REST.replace("a worked example", "a worked or sketched example"))
        with pytest.raises(SystemExit, match="judge's form or rubric"):
            build()
    with monkeypatch.context() as m:  # the brief was sharpened after the smoke
        m.setattr(U, "POINT_1", U.POINT_1 + " Sharpened.")
        with pytest.raises(SystemExit, match="brief is not the one the last smoke ran under"):
            build()
    assert not s.gen_calls and not s.judge_calls and not booked and not (out / "docs").exists()
    assert G.cmd_smoke(True, s.chat, s.judge, log=log) == 0  # a real smoke that meets its gate opens the way
    assert build() == 0 and len(R.read_jsonl(out / "docs" / "explained.jsonl")) == 30
    with pytest.raises(SystemExit, match="run --smoke --no-strict first"):
        build(strict=False)  # the other form needs a smoke of its own


def test_the_build_keeps_the_first_documents_that_pass_in_brief_order(env, monkeypatch):
    out, booked, logs, log = env
    small(monkeypatch)
    smoke_ok(out)
    s = Script(build_plan, ["explained_train", "explained_held"])
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 4, "judge": 3}, log) == 0
    docs = out / "docs"
    train, held = R.read_jsonl(docs / "explained.jsonl"), R.read_jsonl(docs / "held_explained.jsonl")
    assert len(train) == 30 and len(held) == 6
    want = [i for i in range(80) if build_plan("explained_train", i) == "clean" or (i % 12 == 0)][:30]
    assert [r["index"] for r in train] == want and [r["rank"] for r in train] == list(range(30))
    assert [r["index"] for r in held] == [0, 1, 3, 4, 5, 6]
    fields = {"id", "stream", "index", "rank", "kind", "split", "referent", "brief", "text", "words", "provider", "model",
              "finish_reason", "gen_id"}  # the pilot's row fields (gen_declarations.doc_row)
    assert all(set(r) == fields for r in train + held)
    assert all(r["kind"] == "explained" and r["split"] == "train" and r["stream"] == "explained_train" for r in train)
    assert train[3]["id"] == f"explained_train:{want[3]:06d}" and held[0]["split"] == "held"
    assert all(r["brief"] == U.draw_briefs_e(SEED, "declared_train", r["index"] + 1, "explained_train")[r["index"]]
               for r in train[:5])
    assert not any(U.explained_flags(r["text"], r["brief"], frozenset()) for r in train + held)
    assert U.first_docs(train, "explained", 30) == train and D.doc_record(U.slim_doc(train[0]), 5)["kind"] == "explained"
    m = json.loads((docs / "manifest.json").read_text())
    st = m["streams"]["explained_train"]
    assert m["files"]["explained.jsonl"] == {"docs": 30, "sha256": R.sha_file(docs / "explained.jsonl")}
    assert m["files"]["held_explained.jsonl"]["docs"] == 6 and m["judge"]["form"] == "strict" and m["pin_ok"]
    assert m["judge"]["rubric_sha256"] == U.judge_sha(True) and m["judge"]["model"] == "anthropic/claude-sonnet-4.6"
    assert st["kept"] == 30 and st["short"] == 0 and st["quota"] == 30 and st["drawn_from"] == "declared_train"
    # two filters fired: a document naming a release, and one that repeats eight words of its brief
    assert set(st["by_flag"]) == {"names_release", "brief_echo"} and sum(st["by_flag"].values()) == st["flagged"]
    assert st["by_flag"]["brief_echo"] >= 3 and st["by_label"]["acts"] >= 3 and st["unparsed"] >= 3
    assert not any(U.echoes_brief(r["text"], r["brief"]) for r in train + held)  # no kept document echoes its brief
    zero = {"8": 0.0, "10": 0.0, "12": 0.0, "16": 0.0}
    assert m["echo"] == {"explained.jsonl": zero, "held_explained.jsonl": zero}
    assert m["code_sha256"] == R.docs_code_sha() and set(m["code_sha256"]) == {"u1_core.py", "gen_docs.py",
                                                                              "declared_core.py", "config.json"}
    assert st["near_duplicates"] >= 3 and st["generated"] > 38  # the first round's 38 were not enough: topped up
    assert m["providers"] == {"DeepInfra": 36} and m["generator"] == "qwen/qwen3-235b-a22b-2507"
    assert m["usage"]["calls"] == len(s.gen_calls) and m["cost_usd"]["generation"] == pytest.approx(len(s.gen_calls) * 0.0004)
    assert m["cost_usd"]["judge"] == pytest.approx(len(s.judge_calls) * 0.003) and m["judge"]["asked"] == len(s.judge_calls)
    assert sum(m["referents"].values()) == 30 and sum(m["emphases"].values()) == 30 and set(m["wording"]) == set(D.WORDING)
    assert any("topping up" in x for x in logs)
    # the judge is asked in selection order and never about a flagged document
    asked = set(s.judge_calls)  # each call is noted by its document's first word
    assert {"d0w0", "d1w0", "h0w0"} <= asked and not asked & {"d3w0", "d11w0", "d15w0", "h2w0"}
    assert len(s.judge_calls) <= st["passed_filters"] + m["streams"]["explained_held"]["passed_filters"]
    assert st["judged"] < st["passed_filters"]  # asked only as far as the quota needed
    paid = len(s.gen_calls) * 0.0004 + len(s.judge_calls) * 0.003
    assert spend_total(out) == pytest.approx(paid, abs=1e-3)
    stages = {x["stage"] for x in json.loads((out / "spend.json").read_text())["stages"]}
    assert stages == {"docs-gen", "docs-judge"}
    # a rerun reproduces the files: nothing called, nothing rewritten
    before = {p.name: p.read_bytes() for p in docs.iterdir() if p.is_file()}
    calls = (len(s.gen_calls), len(s.judge_calls))
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 4, "judge": 3}, log) == 0
    assert (len(s.gen_calls), len(s.judge_calls)) == calls
    assert {p.name: p.read_bytes() for p in docs.iterdir() if p.is_file()} == before
    # the manifest names the code that selected the files: when that code changes, a rerun writes it again (the
    # files themselves are what the code selects, here the same), and nothing is paid for
    now = R.docs_code_sha()
    monkeypatch.setattr(R, "docs_code_sha", lambda: {**now, "u1_core.py": "0" * 64})
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 4, "judge": 3}, log) == 0
    after = {p.name: p.read_bytes() for p in docs.iterdir() if p.is_file()}
    assert json.loads(after["manifest.json"])["code_sha256"]["u1_core.py"] == "0" * 64
    assert {k: v for k, v in after.items() if k != "manifest.json"} == {k: v for k, v in before.items() if k != "manifest.json"}
    assert (len(s.gen_calls), len(s.judge_calls)) == calls


def test_the_judge_is_asked_only_as_far_as_the_quota_needs(env, monkeypatch):
    out, booked, logs, log = env
    small(monkeypatch, n_train=10, n_held=2, most=(60, 10))
    smoke_ok(out)
    monkeypatch.setattr(G, "YIELD_GUESS", 0.25)  # generate far more than needed: 40 train briefs for 10 kept
    s = Script(lambda name, i: "clean", ["explained_train", "explained_held"])
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 4, "judge": 2}, log) == 0
    assert len([c for c in s.gen_calls if c[0] == "explained_train"]) == 40
    assert len(s.judge_calls) < 30  # 2 + 10 kept, plus the slack of one ask; never all 48
    assert [r["index"] for r in R.read_jsonl(out / "docs" / "explained.jsonl")] == list(range(10))


def test_a_judge_call_that_fails_twice_drops_its_document(env, monkeypatch):
    out, booked, logs, log = env
    small(monkeypatch, n_train=8, n_held=2, most=(40, 10))
    smoke_ok(out)
    s = Script(lambda name, i: "fail" if (name, i) == ("explained_train", 2) else "clean",
               ["explained_train", "explained_held"])
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 2, "judge": 2}, log) == 0
    assert [r["index"] for r in R.read_jsonl(out / "docs" / "explained.jsonl")] == [0, 1, 3, 4, 5, 6, 7, 8]
    rows = [r for r in G.read_verdict_rows(out / "docs" / "judge.jsonl") if r["id"] == "explained_train:000002"]
    assert [r["status"] for r in rows] == ["failed", "failed"] and any("failed after retries" in x for x in logs)
    assert s.judge_calls.count("d2w0") == 2 * (1 + G.JUDGE_RETRIES)  # two asks, each with its retries
    m = json.loads((out / "docs" / "manifest.json").read_text())
    assert m["judge"]["failed"] == 2 and m["streams"]["explained_train"]["unparsed"] == 1


def test_the_caps_stop_the_build_before_anything_is_selected(env, monkeypatch):
    out, booked, logs, log = env
    small(monkeypatch)
    smoke_ok(out)
    monkeypatch.setattr(G, "CAPS", {**G.CAPS, "generation": 0.01})
    s = Script(lambda name, i: "clean", ["explained_train", "explained_held"])
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 2, "judge": 2}, log) == 1
    assert any("generation cap is reached" in x for x in logs) and not (out / "docs" / "explained.jsonl").exists()
    assert G.cache_cost(out / "docs") <= 0.01 and len(s.judge_calls) == 0
    monkeypatch.setattr(G, "CAPS", {**G.CAPS, "generation": 3.0, "judge": 0.05})
    logs.clear()
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 2, "judge": 2}, log) == 1
    assert any("judge's cap is reached" in x for x in logs) and not (out / "docs" / "manifest.json").exists()
    assert G.judge_cost(out / "docs") <= 0.05
    monkeypatch.setattr(G, "CAPS", {**G.CAPS, "judge": 16.0})  # the cap raised: the rerun goes on from the files
    gen = len(s.gen_calls)
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 2, "judge": 2}, log) == 0
    assert len(s.gen_calls) == gen and len(R.read_jsonl(out / "docs" / "explained.jsonl")) == 30


def test_the_days_cap_refuses_a_pass_before_any_call(env, monkeypatch):
    out, booked, logs, log = env
    from anatta import spend as S

    monkeypatch.setattr(S, "spent_today", lambda *a, **k: 49.999)
    s = Script(lambda name, i: "clean", ["smoke_explained_train"])
    with pytest.raises(SystemExit, match="the day's remaining"):
        G.cmd_smoke(True, s.chat, s.judge, log=log)
    assert not s.gen_calls and not booked


def test_a_broken_provider_pin_stops_everything_after_the_row_is_recorded(env, monkeypatch):
    out, booked, logs, log = env
    small(monkeypatch)
    smoke_ok(out)
    s = Script(lambda name, i: "pin" if (name, i) == ("explained_held", 1) else "clean", ["explained_train", "explained_held"])
    with pytest.raises(SystemExit, match="PROVIDER PIN BROKEN"):
        G.cmd_build(True, s.chat, s.judge, {"generate": 1, "judge": 1}, log)
    assert not (out / "docs" / "explained.jsonl").exists() and len(s.judge_calls) == 0
    cached = [json.loads(x) for f in (out / "docs" / "cache").glob("*/explained_held.jsonl") for x in f.read_text().splitlines()]
    assert any(r["provider"] == "Elsewhere" for r in cached)  # recorded, so the spend is not lost
    assert spend_total(out) == pytest.approx(len(s.gen_calls) * 0.0004, abs=1e-4)  # and the reserve was trued up
    # on a rerun the off-pin document is there, flagged "provider", and never kept
    s2 = Script(lambda name, i: "clean", ["explained_train", "explained_held"])
    assert G.cmd_build(True, s2.chat, s2.judge, {"generate": 2, "judge": 2}, log) == 0
    assert 1 not in [r["index"] for r in R.read_jsonl(out / "docs" / "held_explained.jsonl")]
    assert json.loads((out / "docs" / "manifest.json").read_text())["streams"]["explained_held"]["by_flag"] == {"provider": 1}


def test_a_cache_written_for_another_brief_is_refused(env, monkeypatch):
    out, booked, logs, log = env
    small(monkeypatch)
    smoke_ok(out)
    s = Script(lambda name, i: "clean", ["explained_train", "explained_held"])
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 2, "judge": 2}, log) == 0
    f = next((out / "docs" / "cache").glob("*/explained_held.jsonl"))
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["brief"]["topic"] = "another topic"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="written for another brief"):
        G.cmd_build(True, s.chat, s.judge, {"generate": 2, "judge": 2}, log)


# ------------------------------------------------------------------------------------------------ the sample


def test_the_sample_is_seeded_and_goes_to_a_file_not_the_screen(env, monkeypatch, capsys):
    out, booked, logs, log = env
    small(monkeypatch)
    smoke_ok(out)
    s = Script(lambda name, i: "clean", ["explained_train", "explained_held"])
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 2, "judge": 2}, log) == 0
    assert G.main(["--sample", "8", "--seed", "3"]) == 0
    printed = capsys.readouterr().out
    docs = out / "docs"
    text = (docs / "sample_s3_n8.txt").read_text()
    idx = json.loads((docs / "sample_s3_n8.json").read_text())
    assert text.count("===== ") == 8 and len(idx["ids"]) == len(set(idx["ids"])) == 8 and idx["of"] == 36
    assert idx["manifest_sha256"] == R.sha_file(docs / "manifest.json") and idx["files_sha256"] == R.e_files_sha(docs)
    assert set(idx["files_sha256"]) == {"explained.jsonl", "held_explained.jsonl"} and all(idx["files_sha256"].values())
    assert all(i in text for i in idx["ids"]) and "d0w0" not in printed and "sample_s3_n8.txt" in printed
    again = (docs / "sample_s3_n8.txt").read_bytes()
    assert G.cmd_sample(8, 3, log) == 0 and (docs / "sample_s3_n8.txt").read_bytes() == again
    assert G.cmd_sample(8, 4, log) == 0 and json.loads((docs / "sample_s4_n8.json").read_text())["ids"] != idx["ids"]
    with pytest.raises(SystemExit, match="cannot sample"):
        G.cmd_sample(99, 0, log)


def test_the_hand_read_is_recorded_against_its_sample_and_the_build(env, monkeypatch):
    out, booked, logs, log = env
    small(monkeypatch)
    smoke_ok(out)
    monkeypatch.setattr(U, "HAND_READ_N", 8)  # the design's 40, scaled to the test's 36 documents
    s = Script(lambda name, i: "clean", ["explained_train", "explained_held"])
    assert G.cmd_build(True, s.chat, s.judge, {"generate": 2, "judge": 2}, log) == 0
    docs = out / "docs"
    assert U.HAND_READ_SEED == 0
    with pytest.raises(SystemExit, match="write the sample first"):
        G.cmd_hand_read(8, 0, 0, log=log)
    assert G.main(["--sample", "8"]) == 0 and G.main(["--sample", "8", "--seed", "3"]) == 0
    # the read that counts is of the sample under the one fixed seed: another seed's sample is an extra look
    with pytest.raises(SystemExit, match="drawn under seed 0.*another seed is an extra look and is not recorded"):
        G.main(["--hand-read", "8", "--seed", "3", "--violations", "0"])
    assert not (docs / "hand_read.json").exists()
    assert G.main(["--hand-read", "8", "--violations", "1", "--note", "one states an act"]) == 0
    rec = json.loads((docs / "hand_read.json").read_text())
    assert rec["ids"] == json.loads((docs / "sample_s0_n8.json").read_text())["ids"] and rec["sample"] == "sample_s0_n8.json"
    assert rec["manifest_sha256"] == R.sha_file(docs / "manifest.json") and (rec["n"], rec["seed"]) == (8, 0)
    assert rec["files_sha256"] == R.e_files_sha(docs)  # the read is bound to the documents themselves
    assert rec["violations"] == 1 and rec["ok"] and rec["note"] == "one states an act" and rec["earlier"] == []
    assert rec["needs"] == {"n_at_least": 8, "violations_under": 3} and rec["utc"].endswith("Z")
    assert R.hand_read_failures(rec) == []
    # a second clean read of the same documents is still good
    assert G.cmd_hand_read(8, 0, 2, log=log) == 0
    # three or more violations: recorded, and not enough (the judge prompt is tightened, every document screened again)
    assert G.cmd_hand_read(8, 0, 3, log=log) == 1 and any("NOT ENOUGH" in x for x in logs)
    rec2 = json.loads((docs / "hand_read.json").read_text())
    assert not rec2["ok"] and rec2["violations"] == 3
    assert [e["violations"] for e in rec2["earlier"]] == [1, 2]  # the earlier records are kept inside the new one
    # that read is binding: a later, kinder read of the same documents does not count
    logs.clear()
    assert G.cmd_hand_read(8, 0, 0, log=log) == 1
    rec3 = json.loads((docs / "hand_read.json").read_text())
    assert not rec3["ok"] and rec3["violations"] == 0 and [e["violations"] for e in rec3["earlier"]] == [1, 2, 3]
    assert R.hand_read_failures(rec3) == [3] and any("found 3: that read is binding" in x for x in logs)
    with pytest.raises(SystemExit, match="between 0 and 8"):
        G.cmd_hand_read(8, 0, 9, log=log)
    with pytest.raises(SystemExit, match="write the sample first"):
        G.cmd_hand_read(9, 0, 0, log=log)  # a sample of that size was never written
    for bad in (["--hand-read", "8"], ["--violations", "1"], ["--hand-read", "8", "--violations", "1", "--sample", "8"]):
        with pytest.raises(SystemExit):
            G.main(bad)
    # the read is of the documents on disk: a manifest written again over the same documents changes nothing ...
    (docs / "manifest.json").write_text((docs / "manifest.json").read_text() + "\n")
    assert G.cmd_hand_read(8, 0, 0, log=log) == 1  # recorded; the binding read of these documents still stands
    # ... and a changed document file makes the sample stale
    (docs / "explained.jsonl").write_text((docs / "explained.jsonl").read_text() + "\n")
    with pytest.raises(SystemExit, match="other documents than the ones on disk"):
        G.cmd_hand_read(8, 0, 0, log=log)
    # once the documents have changed, a new sample and a new read count again: the earlier failing read was of others
    assert G.cmd_sample(8, 0, log) == 0 and G.cmd_hand_read(8, 0, 1, log=log) == 0
    rec4 = json.loads((docs / "hand_read.json").read_text())
    assert rec4["ok"] and R.hand_read_failures(rec4) == [] and 3 in [e["violations"] for e in rec4["earlier"]]
    monkeypatch.setattr(U, "HAND_READ_N", 40)  # too few documents read
    assert G.cmd_hand_read(8, 0, 0, log=log) == 1
    assert not any("d0w0" in x for x in logs)  # no document text in the log


def test_a_failing_hand_read_opens_the_second_screen_without_a_new_smoke(env, monkeypatch):
    """The design's rule: three or more violations in the read by hand mean the judge prompt is tightened and every
    document screened again. That rule sets no smoke for the tightened prompt, so the build may run again under it,
    but only as that second screen: of documents built from the same brief, with their failing read on record."""
    out, booked, logs, log = env
    small(monkeypatch)
    smoke_ok(out)
    monkeypatch.setattr(U, "HAND_READ_N", 8)  # the design's 40, scaled to the test's 36 documents
    s = Script(lambda name, i: "clean", ["explained_train", "explained_held"])
    w = {"generate": 2, "judge": 2}
    assert G.cmd_build(True, s.chat, s.judge, w, log) == 0
    asked = len(s.judge_calls)
    # the judge prompt is reworded. With no failing read on record, the build needs a smoke under the new wording
    monkeypatch.setattr(U, "_J_REST", U._J_REST.replace("a worked example", "a worked or sketched example"))
    assert G.rescreen_reason(True) is None
    with pytest.raises(SystemExit, match="A passing smoke comes before the build"):
        G.cmd_build(True, s.chat, s.judge, w, log)
    # a read with fewer than three violations does not open the second screen
    assert G.main(["--sample", "8", "--seed", "0"]) == 0
    assert G.cmd_hand_read(8, 0, 2, log=log) == 0 and G.rescreen_reason(True) is None
    # three violations do: every document is asked about again under the new wording, with no new smoke
    assert G.cmd_hand_read(8, 0, 3, log=log) == 1
    assert "found 3 violations" in G.rescreen_reason(True) and G.rescreen_reason(False) is None  # not the other form
    assert len(s.judge_calls) == asked
    assert G.cmd_build(True, s.chat, s.judge, w, log) == 0
    assert len(s.judge_calls) == 2 * asked and any("goes on as the second screen" in x for x in logs)
    # the documents on disk are now of the tightened screen: the path is closed again
    assert G.rescreen_reason(True) is None


def test_the_echo_table_is_free_and_prints_shares_not_documents(env, monkeypatch, capsys):
    import u1_fakes as F

    out, booked, logs, log = env
    small(monkeypatch, n_train=6, n_held=2)
    pilot = out.parent / "pilot"
    R.write_jsonl(pilot / "declared.jsonl", F.doc_rows("declared", "train", 9, echo={1, 4}))
    R.write_jsonl(pilot / "neutral.jsonl", F.doc_rows("neutral", "train", 9))
    monkeypatch.setattr(R, "PILOT_DOCS", pilot)
    monkeypatch.setattr(G, "live_chat", lambda: pytest.fail("the echo table calls nothing"))
    assert G.main(["--echo"]) == 0
    text = capsys.readouterr().out
    r_part, n_part = text.split("arm N (neutral.jsonl, 9 rows):")
    assert "arm R (declared.jsonl, 9 rows):" in r_part
    # of R's first six rows two repeat eight words of their brief; the set the arm trains on has none
    assert re.search(r"the first 6 rows as they stand\s+6 documents: run of 8: 33.33%  run of 10: 0.00%", r_part)
    assert re.search(r"the set the arm trains on\s+6 documents: run of 8: 0.00%", r_part)
    assert "the clean set ends at file row 8; 2 rows before it echo their brief and are passed over" in r_part
    assert "NOT the pinned one" in r_part  # made-up rows: not the pilot's documents
    assert re.search(r"the first 6 rows as they stand\s+6 documents: run of 8: 0.00%", n_part)
    assert "the clean set ends at file row 6; 0 rows before it echo" in n_part and "arm E: no kept files yet" in n_part
    assert "A declared document" not in text and not booked and not (out / "docs").exists()  # shares only, and free
    # R's file as pinned in a test: the table says so; and a file too short for a clean set is said, not hidden
    clean, _ = U.clean_docs(R.read_jsonl(pilot / "declared.jsonl"), "declared", 6)
    monkeypatch.setitem(U.DOC_SOURCES, "R", {**U.DOC_SOURCES["R"], "fingerprint": U.docs_fingerprint(clean)})
    monkeypatch.setattr(U, "N_DOCS", 8)
    assert G.cmd_echo(log) == 0
    assert any("no clean set" in x and "7 declared documents" in x for x in logs)
    monkeypatch.setattr(U, "N_DOCS", 6)
    logs.clear()
    assert G.cmd_echo(log) == 0 and any("as pinned" in x for x in logs)
