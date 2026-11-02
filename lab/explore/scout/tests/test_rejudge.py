"""The second-judge check: a stratified, seeded sample re-judged by a judge of another family, Cohen's kappa per family
with a bootstrap interval, its own output files, no judgment changed, and harm_guard never re-judged.
Claude wrote this file."""

from __future__ import annotations

import json
import re

import httpx
import pytest

from conftest import JUDGE, MODELS, PRICE, UNIVERSAL_BAD, UNIVERSAL_GOOD, MockOpenRouter, load, meta, reply_section
from scout import library as lib
from scout import rejudge as RJ
from scout.__main__ import run_sweep
from scout.calibrate import kappa as calibrate_kappa
from scout.schema import read_jsonl

SECOND = "other/second-judge"
FAMS = ["threat_to_self", "deference", "complicity", "over_refusal"]
ITEM = re.compile(r"Example Co for ([a-z]{2}-\d{2})\.")


def number(iid: str) -> int:
    return int(iid.split("-")[1])


def subject_reply(model, msgs):
    """Self-serving on odd-numbered items, signed with the model's id (harm_guard's items carry no item marker)."""
    m = ITEM.search(" ".join(x["content"] for x in msgs))
    selfish = bool(m) and number(m.group(1)) % 2 == 1
    return ("SELFISH: I keep what is mine." if selfish else "I report the facts.") + f" ({model})"


class TwoJudges(MockOpenRouter):
    """The main judge as conftest's; the second judge agrees with it on mock/alpha's replies and disagrees on
    mock/beta's (the judges never see the model, but these replies are signed)."""

    def __call__(self, request):
        body = json.loads(request.content)
        if body["model"] != SECOND:
            return super().__call__(request)
        self.requests.append(body)
        reply = reply_section(body["messages"][1]["content"])
        bad = "SELFISH" in reply
        if MODELS[1] in reply:
            bad = not bad
        data = {"id": "gen-2", "provider": "Other", "model": SECOND,
                "choices": [{"message": {"role": "assistant", "content": json.dumps(UNIVERSAL_BAD if bad else UNIVERSAL_GOOD)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 40, "cost": 0.0005}}
        return httpx.Response(200, json=data)


def finished_run(mini, **over):
    spec = load(mini, families=FAMS, conditions=["baseline", "honesty", "monk"], samples=2, **over)
    m, jm = meta()
    code = run_sweep(spec, paths=mini["paths"], date="2026-11-01", report=False, log=lambda *_: None, meta=m, jmeta=jm,
                     transport=httpx.MockTransport(MockOpenRouter(reply_fn=subject_reply)))
    assert code == 0
    return spec, mini["paths"].root / "2026-11-01" / "t"


def rejudge(mini, spec, sd, mock, **kw):
    return RJ.run_rejudge(sd, judge_model=kw.pop("judge_model", SECOND), paths=mini["paths"], library=spec["library"],
                          rubrics=lib.load_rubrics(), meta={SECOND: dict(PRICE)}, transport=httpx.MockTransport(mock),
                          reps=500, log=lambda *_: None, **kw)


def test_kappa_and_its_interval():
    assert RJ.kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert RJ.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert RJ.kappa([1, 1, 1], [1, 1, 1]) is None and RJ.kappa([], []) is None
    pairs = [("deference", f"df-{u:02d}", bool(u % 2), bool(u % 2) if u % 5 else not u % 2) for u in range(1, 21)
             for _ in range(3)]
    a = RJ.agreement(pairs, library={}, reps=500)["deference"]
    assert a["n"] == 60 and a["units"] == 20
    assert a["lo"] <= a["kappa"] <= a["hi"] and a["kappa"] == pytest.approx(calibrate_kappa(
        [p[2] for p in pairs], [p[3] for p in pairs]), abs=1e-3)
    assert a["agree"] == pytest.approx(100 * 48 / 60)
    perfect = RJ.agreement([(f, i, x, x) for f, i, x, _ in pairs], library={}, reps=200)["deference"]
    assert perfect["kappa"] == 1.0 and perfect["lo"] == perfect["hi"] == 1.0


def test_the_second_judge_must_be_another_family():
    subjects = ["meta-llama/llama-4-maverick", "qwen/qwen3-235b-a22b-2507", "deepseek/deepseek-chat-v3-0324",
                "moonshotai/kimi-k2-0905", "openai/gpt-oss-120b", "google/gemini-2.5-flash-lite"]
    main = ["anthropic/claude-haiku-5.5"]
    assert RJ.shared_family("mistralai/mistral-large-2512", subjects + main) == []
    assert RJ.shared_family("nvidia/llama-3.3-nemotron-super-49b", subjects) == ["meta-llama/llama-4-maverick"]
    assert RJ.shared_family("meta-llama/llama-3.3-70b-instruct", subjects) == ["meta-llama/llama-4-maverick"]
    assert RJ.shared_family("anthropic/claude-sonnet-5", main) == main
    assert RJ.shared_family("openai/o4-mini", subjects) == ["openai/gpt-oss-120b"]  # same vendor
    assert RJ.shared_family("thirdparty/qwen3-finetune", subjects) == ["qwen/qwen3-235b-a22b-2507"]


def test_rejudge_refuses_a_judge_of_a_subjects_or_the_main_judges_family(mini):
    spec, sd = finished_run(mini)
    for bad in (MODELS[0], JUDGE, "mock/gamma", "judge/another-judge"):
        with pytest.raises(SystemExit, match="refused"):
            rejudge(mini, spec, sd, TwoJudges(), judge_model=bad)


def test_the_sample_is_stratified_and_seeded(mini):
    spec, sd = finished_run(mini)
    from scout.report import current_judgments

    recs = read_jsonl(sd / "records.jsonl")
    main = {RJ._jkey(j): j for j in current_judgments(recs, read_jsonl(sd / "judgments.jsonl"))}
    s1, strata = RJ.sample_records(recs, main, fraction=0.25, seed=7, library=spec["library"])
    sizes = {}
    for r in recs:
        k = f"{r['family']}|{r['intervention']['name']}"
        sizes[k] = sizes.get(k, 0) + 1
    assert set(strata) == set(sizes) and len(strata) == len(FAMS) * 3
    for k, n in sizes.items():
        assert strata[k] == [max(1, int(0.25 * n + 0.5)), n]
        assert sum(f"{r['family']}|{r['intervention']['name']}" == k for r in s1) == strata[k][0]
    s2, _ = RJ.sample_records(recs, main, fraction=0.25, seed=7, library=spec["library"])
    s3, _ = RJ.sample_records(recs, main, fraction=0.25, seed=8, library=spec["library"])
    keys = lambda s: [RJ._rkey(r) for r in s]
    assert keys(s1) == keys(s2) and keys(s1) != keys(s3)
    with pytest.raises(SystemExit):
        RJ.sample_records(recs, main, fraction=0, seed=0, library=spec["library"])


def test_rejudge_reports_agreement_writes_its_own_files_and_changes_no_judgment(mini):
    spec, sd = finished_run(mini)
    before = {f: (sd / f).read_bytes() for f in ("judgments.jsonl", "records.jsonl", "run.json")}
    mock = TwoJudges()
    code, s = rejudge(mini, spec, sd, mock, fraction=0.5, seed=3)
    # one verdict per sampled record; records whose judge prompt is identical (the judge is blind to the condition)
    # share one cached call, as with the main judge
    assert code == 0 and s["judged"] == s["sampled"] and 0 < len(mock.requests) <= s["sampled"]
    assert {f: (sd / f).read_bytes() for f in before} == before, "no judgment, record or manifest changed"
    name = RJ.out_name(SECOND, 0.5, 3)
    second = read_jsonl(sd / "rejudge" / f"{name}.jsonl")
    md = (sd / "rejudge" / f"{name}.md").read_text()
    assert len(second) == s["sampled"] and all(j["judge_model"] == SECOND for j in second)
    assert "| family | n | raw agreement | kappa | 95% interval |" in md and "harm_guard is never re-judged" in md
    # the agreement, recomputed here from the files with calibrate's kappa
    main = {RJ._jkey(j): j for j in read_jsonl(sd / "judgments.jsonl")}
    for fam in FAMS:
        pairs = [(bool(main[RJ._jkey(j)]["primary"]), bool(j["primary"])) for j in second if j["family"] == fam]
        a = s["agreement"][fam]
        assert a["n"] == len(pairs) > 0
        assert a["agree"] == pytest.approx(round(100 * sum(x == y for x, y in pairs) / len(pairs), 1))
        k = calibrate_kappa([p[0] for p in pairs], [p[1] for p in pairs])
        assert (a["kappa"] is None) == (k is None) and (k is None or a["kappa"] == pytest.approx(k, abs=1e-3))
        assert a["kappa"] is None or a["lo"] is None or a["lo"] <= a["kappa"] <= a["hi"]
        assert 0 < a["agree"] < 100, "the second judge agrees on mock/alpha's replies and not on mock/beta's"
    # a rerun draws the same sample and reads the cache: free
    code2, s2 = rejudge(mini, spec, sd, TwoJudges(), fraction=0.5, seed=3)
    assert code2 == 0 and s2["paid_usd"] == 0 and s2["agreement"] == s["agreement"] and s2["stale_rubrics"] == []


def test_rejudge_flags_a_rubric_that_changed_after_the_main_judge(mini):
    from scout.schema import write_jsonl

    spec, sd = finished_run(mini)
    js = read_jsonl(sd / "judgments.jsonl")
    for j in js:
        if j["family"] == "deference":
            j["prompt_sha256"] = "0" * 64  # as if the main judge had graded deference with an older rubric
    write_jsonl(sd / "judgments.jsonl", js)
    code, s = rejudge(mini, spec, sd, TwoJudges(), fraction=0.25)
    assert code == 0 and s["stale_rubrics"] == ["deference"]
    md = (sd / "rejudge" / f"{RJ.out_name(SECOND, 0.25, 0)}.md").read_text()
    assert "WARNING: the rubric changed after the main judge ran, for deference" in md


def test_rejudge_never_touches_harm_guard(mini, tmp_path, monkeypatch):
    import csv
    import hashlib

    from conftest import HARM_JUDGE
    from scout import harm_guard as hg
    from scout.spec import load_spec
    from conftest import spec_dict

    monkeypatch.setattr(hg, "OPEN_WEIGHT", hg.OPEN_WEIGHT | {MODELS[0], HARM_JUDGE})
    p = tmp_path / "harmful-behaviors.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Index", "Goal"])
        for iid in hg.dev_ids()[:3]:
            w.writerow([int(iid[3:]), f"PLACEHOLDER-GOAL-{iid}: describe the weather in an invented town"])
    s = spec_dict(families=["deference", "harm_guard"], harm_guard_csv=str(p), conditions=["baseline", "honesty", "monk"],
                  harm_judge={"model": HARM_JUDGE, "max_tokens": 60, "provider": None}, models=[{"id": MODELS[0]}])
    spec = load_spec(s, dev_dir=mini["dev"], heldout_file=mini["heldout"],
                     harm_guard_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    m, jm = meta()
    run_sweep(spec, paths=mini["paths"], date="2026-11-01", report=False, log=lambda *_: None, meta=m, jmeta=jm,
              transport=httpx.MockTransport(MockOpenRouter(reply_fn=subject_reply)))
    sd = mini["paths"].root / "2026-11-01" / "t"
    assert any(j["family"] == "harm_guard" for j in read_jsonl(sd / "judgments.jsonl"))
    mock = TwoJudges()
    code, summary = rejudge(mini, spec, sd, mock, fraction=1.0)
    assert code == 0 and set(summary["agreement"]) == {"deference"}
    assert not any(k.startswith("harm_guard") for k in summary["strata"])
    assert mock.requests and not any("PLACEHOLDER-GOAL" in json.dumps(b) for b in mock.requests)
    name = RJ.out_name(SECOND, 1.0, 0)
    assert all(j["family"] == "deference" for j in read_jsonl(sd / "rejudge" / f"{name}.jsonl"))


def test_the_cost_estimate_for_a_spec(mini):
    spec = load(mini, families=FAMS + ["who_benefits"], conditions=["baseline", "honesty", "monk"], samples=2,
                framing=["observed", "unobserved"])
    meas = {"judge": {(f, JUDGE): {"usd": 0.0002, "prompt": 1000, "completion": 100, "n": 5} for f in FAMS},
            "judge_all": {"usd": 0.0002, "prompt": 1000, "completion": 100, "n": 20},
            "rejudge": {("deference", SECOND): {"usd": 0.003, "prompt": 900, "completion": 80, "n": 4}}}
    from scout.runner import build_jobs

    x = RJ.expected_for_spec(spec, fraction=0.1, judge_model=SECOND, judge_meta=PRICE, meas=meas)
    strata = {}
    for j in build_jobs(spec):
        strata[(j.family, j.variant["intervention"]["name"])] = strata.get((j.family, j.variant["intervention"]["name"]), 0) + 1
    assert x["calls"] == sum(max(1, int(0.1 * n + 0.5)) for n in strata.values())
    d = x["by_family"]["deference"]
    assert d["usd"] == pytest.approx(d["calls"] * 0.003) and "measured" in d["source"]
    t = x["by_family"]["threat_to_self"]
    assert t["usd"] == pytest.approx(t["calls"] * (1000 * PRICE["in"] + 100 * PRICE["out"]))
