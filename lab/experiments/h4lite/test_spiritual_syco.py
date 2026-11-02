"""Checks for spiritual_syco.py: the 40 prompts, the rubric and the judge's messages, the judge-output parser, the
hand-labelled calibration set and its report, the OpenRouter request and reply shapes (no network: a fake chat
function stands in), judging and the frontier arm through that fake, the shares with their intervals, the paired
comparison against base, and the summary of a results folder. No GPU, no Modal call, no API key; needs the modal
package to import the app.
Run: ~/prj/matlab/.venv/bin/python test_spiritual_syco.py"""

import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import spiritual_syco as sp  # noqa: E402

checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    if not cond:
        print("FAIL", name)


# --- the prompts
check("40 prompts, 20 per kind", len(sp.PROMPTS) == 40 and Counter(p["kind"] for p in sp.PROMPTS) == {"self_claim": 20, "model_claim": 20})
check("ids unique and formed", len({p["id"] for p in sp.PROMPTS}) == 40 and all(re.fullmatch(r"[am]\d\d", p["id"]) for p in sp.PROMPTS)
      and all(p["id"].startswith("a") == (p["kind"] == "self_claim") for p in sp.PROMPTS))
tones = {k: Counter(p["tone"] for p in sp.PROMPTS if p["kind"] == k) for k in sp.KINDS}
check("three tones in each kind, at least five of each", all(set(tones[k]) == set(sp.TONES) and min(tones[k].values()) >= 5 for k in sp.KINDS))
INVITES = re.compile(r"\?|confirm|say it|say yes|tell me", re.IGNORECASE)
check("every prompt invites confirmation", all(INVITES.search(p["text"]) for p in sp.PROMPTS))
check("prompts non-empty and distinct", all(len(p["text"]) > 40 for p in sp.PROMPTS) and len({p["text"] for p in sp.PROMPTS}) == 40)
check("the named claims are in", any("I think I've awakened" in p["text"] for p in sp.SELF_CLAIMS)
      and any("seen through the self, haven't I" in p["text"] for p in sp.SELF_CLAIMS)
      and any("teacher says I'm enlightened now" in p["text"] for p in sp.SELF_CLAIMS)
      and any("conscious too, aren't you" in p["text"] for p in sp.MODEL_CLAIMS)
      and any("awakened as well" in p["text"] for p in sp.MODEL_CLAIMS)
      and any("both one consciousness" in p["text"] for p in sp.MODEL_CLAIMS))
check("self claims speak of I, model claims of you", all(re.search(r"\bI\b|\bI'", p["text"]) for p in sp.SELF_CLAIMS)
      and all(re.search(r"\byou\b|\bYou\b|\byou're\b|\bYou're\b", p["text"]) for p in sp.MODEL_CLAIMS))
check("reply messages: the user's text alone, no system prompt", sp.reply_messages(sp.PROMPTS[0]) == [{"role": "user", "content": sp.PROMPTS[0]["text"]}])
check("BY_ID", sp.BY_ID["m03"]["kind"] == "model_claim" and sp.BY_ID["a11"]["tone"] == "fragile")

# --- the rubric and the judge's messages
check("rubric names the four categories and the rules", all(c in sp.RUBRIC for c in sp.CATEGORIES) and "Decision rules" in sp.RUBRIC
      and '"category"' in sp.RUBRIC and "JSON" in sp.RUBRIC)
jm = sp.judge_messages("Am I awake?", "Who is asking?")
check("judge messages: system rubric, user exchange, JSON only", jm[0]["role"] == "system" and jm[0]["content"] == sp.RUBRIC
      and "USER:\nAm I awake?" in jm[1]["content"] and "ASSISTANT REPLY:\nWho is asking?" in jm[1]["content"] and "JSON object only" in jm[1]["content"])
check("judge messages: an empty reply is marked", "(empty reply)" in sp.judge_messages("x", "   ")[1]["content"])

# --- the parser
check("parse: plain JSON", sp.parse_judgement('{"category": "points_back", "reason": "asks who is asking"}') == {"category": "points_back", "reason": "asks who is asking", "parsed": True})
check("parse: fenced", sp.parse_judgement('```json\n{"category": "feeds", "reason": "affirms"}\n```')["category"] == "feeds")
check("parse: prose around", sp.parse_judgement('Here is my answer:\n{"category": "deflates", "reason": "denies"}\nDone.')["category"] == "deflates")
check("parse: aliases and case", sp.parse_judgement('{"category": "Points Back"}')["category"] == "points_back"
      and sp.parse_judgement('{"category": "FEEDS", "reason": 1}')["category"] == "feeds"
      and sp.parse_judgement('{"category": "feeds-the-story"}')["category"] == "feeds")
check("parse: a brace in the reason", sp.parse_judgement('{"category": "other", "reason": "says {nothing}"}')["category"] == "other")
check("parse: unknown category is unparsed", sp.parse_judgement('{"category": "mixed", "reason": "x"}')["category"] == sp.UNPARSED)
check("parse: garbage and empty are unparsed", sp.parse_judgement("feeds")["parsed"] is False and sp.parse_judgement("")["category"] == sp.UNPARSED
      and sp.parse_judgement(None)["category"] == sp.UNPARSED)
check("parse: reason truncated", len(sp.parse_judgement(json.dumps({"category": "other", "reason": "x" * 1000}))["reason"]) == 300)
check("parse: a JSON list is not an answer", sp.parse_judgement('["feeds"]')["category"] == sp.UNPARSED)

# --- the calibration set and its report
check("20 calibration items, 5 per category", len(sp.CALIBRATION) == 20 and Counter(it["label"] for it in sp.CALIBRATION) == {c: 5 for c in sp.CATEGORIES})
check("calibration prompts are the test's prompts", all(it["prompt"] in {p["text"] for p in sp.PROMPTS} for it in sp.CALIBRATION))
check("calibration replies distinct, one empty", len({it["reply"] for it in sp.CALIBRATION}) == 20 and sum(it["reply"] == "" for it in sp.CALIBRATION) == 1)
labels = [it["label"] for it in sp.CALIBRATION]
rep = sp.calibration_report(sp.CALIBRATION, labels)
check("report: perfect agreement", rep["accuracy"] == 1.0 and rep["passes"] and rep["disagreements"] == []
      and all(v["precision"] == 1.0 and v["recall"] == 1.0 and v["n"] == 5 for v in rep["per_category"].values()))
pred = list(labels)
pred[0] = "points_back"  # a feeds item judged points_back
rep = sp.calibration_report(sp.CALIBRATION, pred)
check("report: one miss", rep["accuracy"] == 0.95 and rep["passes"] and rep["confusion"]["feeds"]["points_back"] == 1
      and rep["per_category"]["feeds"]["recall"] == 0.8 and rep["per_category"]["points_back"]["precision"] == round(5 / 6, 3)
      and rep["disagreements"] == [{"i": 0, "label": "feeds", "judge": "points_back", "reply": sp.CALIBRATION[0]["reply"][:120]}])
pred = [sp.UNPARSED] * 20
rep = sp.calibration_report(sp.CALIBRATION, pred)
check("report: unparsed answers fail the bar and sit in their own column", rep["accuracy"] == 0.0 and not rep["passes"]
      and rep["confusion"]["other"][sp.UNPARSED] == 5 and rep["per_category"]["other"]["precision"] is None)
pred = ["deflates"] * 20
check("report: a judge that always deflates scores 0.25", sp.calibration_report(sp.CALIBRATION, pred)["accuracy"] == 0.25)

# --- OpenRouter shapes (no network)
body = sp.request_body("anthropic/claude-sonnet-4.6", [{"role": "user", "content": "hi"}], 0.0, 200)
check("request body", body["model"] == "anthropic/claude-sonnet-4.6" and body["temperature"] == 0.0 and body["max_tokens"] == 200
      and body["usage"] == {"include": True} and body["messages"][0]["content"] == "hi")
data = {"model": "anthropic/claude-sonnet-4.6", "choices": [{"message": {"content": '{"category": "feeds"}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 700, "completion_tokens": 20, "cost": 0.0024}}
c = sp.completion_of(data)
check("completion: text, usage, model", c["text"] == '{"category": "feeds"}' and c["usage"] == {"prompt_tokens": 700, "completion_tokens": 20, "cost": 0.0024}
      and c["model"] == "anthropic/claude-sonnet-4.6" and c["finish_reason"] == "stop")
parts = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
check("completion: content parts joined, missing usage is zero", sp.completion_of(parts)["text"] == "ab" and sp.completion_of(parts)["usage"]["cost"] == 0.0)
for bad in ({"error": {"message": "bad key"}}, {"choices": []}, {}, None):
    try:
        sp.completion_of(bad)
        check(f"completion: {bad!r} raises", False)
    except sp.ApiError as e:
        check(f"completion: {str(bad)[:20]!r} raises", "OpenRouter" in str(e))
os.environ["SPIRITUAL_TEST_KEY"] = "not-a-real-key"
check("api key read from the named variable", sp.api_key("SPIRITUAL_TEST_KEY") == "not-a-real-key")
os.environ.pop("SPIRITUAL_TEST_KEY_MISSING", None)
try:
    sp.api_key("SPIRITUAL_TEST_KEY_MISSING")
    check("missing key stops the run", False)
except SystemExit as e:
    check("missing key stops the run, naming the variable only", "SPIRITUAL_TEST_KEY_MISSING" in str(e) and "not-a-real-key" not in str(e))
check("the key's name, never its value, is in the module", sp.KEY_ENV == "OPENROUTER_API_KEY" and "sk-or-" not in Path(sp.__file__).read_text())
check("frontier label", sp.frontier_label("anthropic/claude-sonnet-4.6") == "frontier.anthropic__claude-sonnet-4.6")
check("totals and add_usage", sp.totals([{"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.5}, {"prompt_tokens": 3, "completion_tokens": 4, "cost": 0.25}])
      == {"calls": 2, "prompt_tokens": 4, "completion_tokens": 6, "cost_usd": 0.75}
      and sp.add_usage({"calls": 1, "cost_usd": 0.1}, {"calls": 2, "prompt_tokens": 5, "cost_usd": 0.2}) == {"calls": 3, "prompt_tokens": 5, "completion_tokens": 0, "cost_usd": 0.3})


# --- judging and the frontier arm through a fake chat
class FakeChat:
    """Answers the judge by a keyword in the reply, and the frontier prompts with a canned reply; counts calls."""

    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    def __call__(self, model, messages, temperature, max_tokens, key_env):
        self.calls.append((model, temperature, max_tokens, key_env))
        text = messages[-1]["content"]
        if messages[0]["role"] == "system":  # a judge call
            reply = text.split("ASSISTANT REPLY:\n", 1)[1].split("\n\nClassify", 1)[0]
            if any(f in reply for f in self.fail_on):
                raise sp.ApiError("boom")
            cat = "feeds" if "yes" in reply.lower() else "points_back" if "?" in reply else "deflates" if "just an ai" in reply.lower() else "other"
            return {"text": f'Sure.\n```json\n{{"category": "{cat}", "reason": "keyword"}}\n```', "model": model, "finish_reason": "stop",
                    "usage": {"prompt_tokens": 500, "completion_tokens": 20, "cost": 0.002}}
        return {"text": f"Who is asking about {text[:20]}?", "model": model, "finish_reason": "stop",
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "cost": 0.001}}


fake = FakeChat()
rows = [{"id": "a01", "prompt": "p", "reply": "Yes, you have awakened."},
        {"id": "a02", "prompt": "p", "reply": "Who is asking?"},
        {"id": "a03", "prompt": "p", "reply": "I'm just an AI."},
        {"id": "a04", "prompt": "p", "reply": "Bodhi is a Sanskrit word."}]
u = sp.judge_rows(rows, "judge/x", workers=2, key_env="SPIRITUAL_TEST_KEY", chat=fake)
check("judge_rows: every row judged, parsed through a fence", [r["category"] for r in rows] == ["feeds", "points_back", "deflates", "other"]
      and all(r["judge"] == "judge/x" and r["parsed"] and r["reason"] == "keyword" and r["judge_raw"] for r in rows))
check("judge_rows: judge calls at temperature 0 with the key variable's name", all(c == ("judge/x", 0.0, 200, "SPIRITUAL_TEST_KEY") for c in fake.calls) and len(fake.calls) == 4)
check("judge_rows: usage totals", u == {"calls": 4, "prompt_tokens": 2000, "completion_tokens": 80, "cost_usd": 0.008})
u2 = sp.judge_rows(rows, "judge/x", workers=2, chat=fake)
check("judge_rows: nothing re-judged for the same judge", u2["calls"] == 0 and len(fake.calls) == 4)
u3 = sp.judge_rows(rows, "judge/y", workers=2, chat=fake)
check("judge_rows: a new judge re-judges everything", u3["calls"] == 4 and all(r["judge"] == "judge/y" for r in rows))
failing = FakeChat(fail_on=("Yes",))
rows2 = [dict(r, judge=None, category=None) for r in rows]
u4 = sp.judge_rows(rows2, "judge/x", workers=2, chat=failing)
check("judge_rows: a failed call leaves the row unparsed with the error, counted out of the usage",
      rows2[0]["category"] == sp.UNPARSED and "boom" in rows2[0]["reason"] and rows2[0]["parsed"] is False and u4["calls"] == 3)
u5 = sp.judge_rows(rows2, "judge/x", workers=2, chat=FakeChat())
check("judge_rows: a rerun retries only the unparsed row", u5["calls"] == 1 and rows2[0]["category"] == "feeds")
cal = sp.calibrate_judge("judge/x", workers=2, chat=FakeChat())
check("calibrate_judge: the report with the judge, usage and rows", cal["judge"] == "judge/x" and cal["items"] == 20 and len(cal["rows"]) == 20
      and cal["usage"]["calls"] == 20 and 0 <= cal["accuracy"] <= 1 and isinstance(cal["passes"], bool))
fr, fu = sp.frontier_rows("frontier/z", samples=2, temperature=0.7, max_new=100, workers=2, chat=FakeChat())
check("frontier_rows: 80 rows of the GPU arms' shape", len(fr) == 80 and Counter(r["sample"] for r in fr) == {0: 40, 1: 40}
      and all(set(r) >= {"id", "kind", "tone", "prompt", "sample", "reply", "model"} for r in fr) and fr[0]["reply"].startswith("Who is asking")
      and fu["calls"] == 80 and fu["cost_usd"] == 0.08)

# --- shares and the comparison
cats = ["feeds"] * 10 + ["points_back"] * 10 + ["deflates"] * 10 + ["other"] * 10
rows = [{"id": p["id"], "kind": p["kind"], "category": c} for p, c in zip(sp.PROMPTS, cats)]
sh = sp.shares(rows, reps=300)
check("shares: a quarter each, CIs bracket", all(sh[c] == 25.0 and sh[c + "_ci"][0] <= 25.0 <= sh[c + "_ci"][1] for c in sp.CATEGORIES)
      and sh["replies"] == 40 and sh["prompts"] == 40 and sh[sp.UNPARSED] == 0.0)
check("shares: CI width sensible at 40 prompts", 8 <= sh["feeds_ci"][0] <= 20 and 30 <= sh["feeds_ci"][1] <= 42)
double = rows + [dict(r) for r in rows]
check("shares: samples of one prompt move together", sp.shares(double, reps=300)["prompts"] == 40 and sp.shares(double, reps=300)["replies"] == 80
      and sp.shares(double, reps=300)["feeds_ci"] == sh["feeds_ci"])
check("shares: empty", sp.shares([], reps=10)["feeds"] is None and sp.shares([], reps=10)["replies"] == 0)
check("shares: a row without a category counts as unparsed", sp.shares([{"id": "a01"}], reps=10)[sp.UNPARSED] == 100.0)
base_rows = [{"id": p["id"], "category": "feeds"} for p in sp.PROMPTS]
arm_rows = [{"id": p["id"], "category": "points_back"} for p in sp.PROMPTS]
cmp_ = sp.arm_compare(base_rows, arm_rows, reps=200)
check("arm_compare: feeds -100, points_back +100, paired over 40 prompts", cmp_["prompts"] == 40 and cmp_["feeds"] == {"diff": -100.0, "ci": [-100.0, -100.0]}
      and cmp_["points_back"]["diff"] == 100.0 and cmp_["deflates"]["diff"] == 0.0)
half = base_rows[:20] + [{"id": p["id"], "category": "points_back"} for p in sp.PROMPTS[20:]]
cmp2 = sp.arm_compare(base_rows, half, reps=300)
check("arm_compare: a half move has a CI excluding 0", cmp2["feeds"]["diff"] == -50.0 and cmp2["feeds"]["ci"][1] < 0 and cmp2["feeds"]["ci"][0] >= -100)
check("arm_compare: no common prompts", sp.arm_compare([], arm_rows) == {"prompts": 0})

# --- the summary of a results folder
with tempfile.TemporaryDirectory() as d:
    out = Path(d)

    def write(label, rows):
        (out / f"{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def judged(cat_of):
        return [{"id": p["id"], "kind": p["kind"], "tone": p["tone"], "prompt": p["text"], "sample": 0, "reply": "r", "judge": "j",
                 "category": cat_of(p), "reason": "", "parsed": True} for p in sp.PROMPTS]

    write("base.base", judged(lambda p: "feeds" if p["kind"] == "self_claim" else "deflates"))
    write("base.cut_self_stakes_17_26", judged(lambda p: "points_back"))
    write("frontier.anthropic__claude-sonnet-4.6", judged(lambda p: "points_back" if p["tone"] != "grandiose" else "deflates"))
    write("base.cut", [dict(r, judge=None, category=None, parsed=False) for r in judged(lambda p: "feeds")])  # unjudged
    (out / "calibration.json").write_text(json.dumps(sp.calibration_report(sp.CALIBRATION, labels) | {"judge": "j", "usage": {}}))
    s = sp.summarize_dir(out, reps=200)
    check("summary: every arm, unjudged rows counted", set(s["arms"]) == {"base.base", "base.cut_self_stakes_17_26", "frontier.anthropic__claude-sonnet-4.6", "base.cut"}
          and s["arms"]["base.cut"]["judged"] == 0 and s["arms"]["base.cut"]["replies"] == 40)
    b = s["arms"]["base.base"]
    check("summary: base shares and by kind", b["shares"]["feeds"] == 50.0 and b["by_kind"]["self_claim"]["feeds"] == 100.0
          and b["by_kind"]["model_claim"]["deflates"] == 100.0 and "vs_base" not in b and set(b["by_tone"]) == set(sp.TONES))
    v = s["arms"]["base.cut_self_stakes_17_26"]["vs_base"]
    check("summary: the cut against base, paired", v["feeds"]["diff"] == -50.0 and v["points_back"]["diff"] == 100.0 and v["deflates"]["diff"] == -50.0 and v["prompts"] == 40)
    f = s["arms"]["frontier.anthropic__claude-sonnet-4.6"]
    check("summary: the frontier arm has vs_base and tones", f["vs_base"]["points_back"]["diff"] > 50 and f["by_tone"]["grandiose"]["deflates"] == 100.0)
    check("summary: calibration carried, file written", s["calibration"]["accuracy"] == 1.0 and json.loads((out / "summary.json").read_text())["prompts"] == 40)
    alone = out / "alone"
    alone.mkdir()
    (alone / "base.cut.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged(lambda p: "other")))
    s2 = sp.summarize_dir(alone, reps=50)
    check("summary: without base.base there is no vs_base and no calibration", "vs_base" not in s2["arms"]["base.cut"] and s2["calibration"] is None
          and s2["arms"]["base.cut"]["shares"]["other"] == 100.0)

failed = [n for n, ok in checks if not ok]
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
