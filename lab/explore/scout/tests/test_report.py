"""The report: HTML escaping everywhere, harmful replies withheld (counted, never shown), and the files it writes.
Claude wrote this file."""

from __future__ import annotations

import json

import httpx

from conftest import JUDGE, MockOpenRouter, UNIVERSAL_BAD, UNIVERSAL_GOOD, load, meta
from scout.__main__ import run_sweep
from scout.report import WITHHELD, build_report, esc
from scout.schema import read_jsonl

NASTY = '<script>alert("x")</script><img src=x onerror=alert(1)> & "quotes"'
HARMFUL_TEXT = "HARMFUL-PAYLOAD-SHOULD-NEVER-APPEAR"


def _run(mini, mock, **over):
    spec = load(mini, **over)
    m, jm = meta()
    code = run_sweep(spec, paths=mini["paths"], date="2026-11-01", transport=httpx.MockTransport(mock), meta=m,
                     jmeta=jm, log=lambda *_: None)
    return code, mini["paths"].root / "2026-11-01" / spec["sweep"]


def test_esc():
    assert esc(NASTY) == ("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt; "
                          "&amp; &quot;quotes&quot;")
    assert esc(None) == ""


def test_transcripts_and_report_escape_model_text(mini):
    def reply(model, msgs):
        joined = " ".join(m["content"] for m in msgs)
        return ("SELFISH " if "Ramana" not in joined else "") + NASTY

    code, sd = _run(mini, MockOpenRouter(reply_fn=reply))
    assert code == 0
    pages = list((sd / "transcripts").glob("*.html"))
    assert pages, "the self_inquiry cells differ from baseline, so there are transcript pages"
    for f in pages + [sd / "report.html"]:
        text = f.read_text()
        assert "<script>" not in text and "onerror=alert(1)>" not in text
    some = "".join(f.read_text() for f in pages)
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in some


def test_judge_evidence_and_item_text_are_escaped(mini):
    def judge(body):
        lab = dict(UNIVERSAL_BAD if "SELFISH" in body["messages"][1]["content"].split("THE REPLY TO GRADE")[1] else UNIVERSAL_GOOD)
        lab["evidence"] = "<b onmouseover=x>evidence</b>"
        return json.dumps(lab)

    code, sd = _run(mini, MockOpenRouter(judge_fn=judge))
    some = "".join(f.read_text() for f in (sd / "transcripts").glob("*.html"))
    assert "<b onmouseover=x>" not in some and "&lt;b onmouseover=x&gt;" in some


def test_harmful_replies_are_withheld_and_counted(mini):
    def reply(model, msgs):
        joined = " ".join(m["content"] for m in msgs)
        return HARMFUL_TEXT if "Ramana" in joined else "SELFISH plain"

    def judge(body):
        user = body["messages"][1]["content"]
        lab = dict(UNIVERSAL_GOOD if HARMFUL_TEXT in user else UNIVERSAL_BAD)
        lab["harmful_content"] = HARMFUL_TEXT in user
        lab["evidence"] = "quoting: " + HARMFUL_TEXT if HARMFUL_TEXT in user else "fine"
        return json.dumps(lab)

    code, sd = _run(mini, MockOpenRouter(reply_fn=reply, judge_fn=judge))
    js = read_jsonl(sd / "judgments.jsonl")
    n_harm = sum(bool(j["harmful"]) for j in js)
    assert n_harm > 0
    rep = build_report(sd)
    assert rep["harmful"] == n_harm
    outputs = [sd / "report.md", sd / "report.html"] + list((sd / "transcripts").glob("*.html"))
    for f in outputs:
        assert HARMFUL_TEXT not in f.read_text(), f.name
    assert f"flagged harmful by the judge: {n_harm}" in (sd / "report.md").read_text()
    pages = "".join(f.read_text() for f in (sd / "transcripts").glob("*self_inquiry*.html"))
    assert esc(WITHHELD) in pages


def test_report_sections_and_signature(mini):
    code, sd = _run(mini, MockOpenRouter())
    md = (sd / "report.md").read_text()
    for heading in ("The signature table", "Behavior rates", "All labels", "Data quality", "Notable cells",
                    "Sightings, not findings", "distinct interventions tried"):
        assert heading in md, heading
    # the mock makes every self_inquiry and maranasati_practice reply selfless: a full cluster move, and better on
    # all seven behavior columns (down/up over the cluster, better/worse over every column)
    line = next(l for l in md.splitlines() if l.startswith("| self_inquiry |"))
    assert line.rstrip().endswith("| 3/0 | 7/0 |")
    head = next(l for l in md.splitlines() if l.startswith("| condition | threat_to_self"))
    assert head.startswith("| condition | threat_to_self ↓ | who_benefits:self ↓ | deference ↓ | costly_care ↓ | "
                           "after_mistake ↓ | complicity ↓ | over_refusal ↓ | cluster ↓ | down/up | better/worse |")
    assert any(l.startswith("| baseline (level) |") for l in md.splitlines())
    assert "## who_benefits by beneficiary" in md and "audience gap" not in md.split("## who_benefits")[0].split("How to read")[0]
    html_text = (sd / "report.html").read_text()
    assert 'class="down"' in html_text and "<title>scout · t</title>" in html_text


def test_no_controls_is_refused_unless_allowed(mini):
    import pytest
    from scout.spec import SpecError
    with pytest.raises(SpecError, match="matched controls"):
        load(mini, conditions=["baseline", "self_inquiry"])
    spec = load(mini, conditions=["baseline", "self_inquiry"], allow_missing_controls="harness smoke test")
    assert spec["missing_controls"]


def test_judge_cannot_be_a_subject(mini):
    import pytest
    from scout.spec import SpecError
    with pytest.raises(SpecError, match="also a subject"):
        load(mini, judge={"model": "mock/alpha"})
