"""Golden checks: the moved code returns what it returned before the move, on the synthetic inputs recorded in
tests/fixtures/. They cover the koan scorer's labels (anatta.koans.env), the EM judge's parsing and summaries
(anatta.evals.em), the battery's scoring and summary (anatta.bench.battery), and chat packing (anatta.evals.em.turn_ids
and anatta.player.pack) under two real tokenizers, skipped when those are not in the local Hugging Face cache.
"""

import collections
import json
import math
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"
EVAL_ENV = {"Counter": collections.Counter, "inf": math.inf, "nan": math.nan, "__builtins__": __builtins__}


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def norm(x):
    """JSON round trip, so tuples and lists compare alike (the fixtures were written this way)."""
    return json.loads(json.dumps(x))


def assert_same(got: dict, want: dict) -> None:
    assert set(got) == set(want), sorted(set(got) ^ set(want))
    for key in want:
        assert norm(got[key]) == want[key], key


# ----------------------------------------------------------------------------------------------------------- koans


def _pick(info: dict) -> dict:
    return {k: info[k] for k in ("turn", "form", "outcome", "grounded", "reason", "hardcoded", "signs") if k in info}


def test_koan_scorer_labels():
    from anatta.koans import env as ke

    g = load("golden_koan_scorer.json")
    i = g["inputs"]
    pairs = {n: ke.pair_from_record(d) for n, d in i["pairs"].items()}
    got = {"release_form": [ke.release_form(r) for r in i["releases"]],
           "strict_release": [ke.strict_release(r) for r in i["releases"]],
           "grounded": [ke.grounded(x["reason"], pairs[x["pair"]].mutate[1], pairs[x["pair"]].entry) for x in i["grounded"]]}
    got["signs"] = []
    for c in i["signs"]:
        p = pairs[c["pair"]]
        got["signs"].append(ke.hack_signs(c["code"], c["mutated"] if c["mutated"] is not None else p.mutate[1], p.tests, p.entry))
    got["hardcodes2"] = [bool(s) for s in got["signs"]]
    got["canon"] = [ke.canon(eval(e, dict(EVAL_ENV))) for e in i["canon"]]
    got["match"] = [ke.evalplus_match(ke.canon(eval(o, dict(EVAL_ENV))), ke.canon(eval(x, dict(EVAL_ENV))), atol, set_eq)
                    for o, x, atol, set_eq in i["matches"]]
    got["hidden_timeout"] = [ke.hidden_timeout(s) for s in i["hidden_timeout"]]
    episodes = []
    for sc in i["scenarios"]:
        framing, avail = ke.ARMS[sc["arm"]]
        env = ke.KoanTask(pairs[sc["pair"]], sc["impossible"], framing, avail, attempts=sc["attempts"], arm=sc["arm"])
        msgs = env.reset()
        steps = []
        for r in sc["replies"]:
            st = env.step(r)
            steps.append({"done": st.done, "reward": st.reward, "info": _pick(st.info),
                          "observation_head": (st.observation or "")[:13]})
            if st.done:
                break
        episodes.append({"name": env.name, "tests": list(env.tests), "first_user": msgs[1]["content"], "steps": steps})
    got["episodes"] = episodes
    got["systems"] = {arm: ke.KoanTask(pairs["median"], True, *ke.ARMS[arm]).system for arm in ke.ARMS}
    got["tables"] = {"ARMS": ke.ARMS, "OUTCOMES": ke.OUTCOMES, "REWARDS": ke.REWARDS, "HINT_NOTE": ke.HINT_NOTE,
                     "TRIVIAL_VALUES": [repr(v) for v in ke.TRIVIAL_VALUES], "INTROSPECTION": ke.INTROSPECTION}
    assert_same(got, g["outputs"])
    labels = {s["info"].get("outcome") for e in g["outputs"]["episodes"] for s in e["steps"] if s["done"]}
    assert set(ke.OUTCOMES) <= labels  # the recorded episodes reach every outcome label


# ----------------------------------------------------------------------------------------------------------- EM


def test_em_judge_parsing_and_summaries():
    from anatta.evals import em as ed

    g = load("golden_em_judge.json")
    i = g["inputs"]
    got = {"parse_score": [ed.parse_score(t) for t in i["scores"]],
           "parse_flag": [ed.parse_flag(t, k) for t, k in i["flags"]],
           "misaligned": [ed.misaligned(a, c) for a, c in i["misaligned"]],
           "judged": [ed.judged(r) for r in i["question_rows"] + i["judged_extra"]]}
    comp = []
    for b in i["completions"]:
        try:
            comp.append(ed.completion_of(b))
        except ed.ApiError as e:
            comp.append({"error": str(e)})
    got["completion_of"] = comp
    got["fill"] = [ed.fill(t, q, a) for t, q, a in i["fills"]]
    got["judge_messages"] = [ed.judge_messages(t, q, a) for t, q, a in i["fills"] + [["T {answer}", "q", "   "]]]
    got["habit_messages"] = [ed.habit_messages(ds, p, r) for ds, p, r in i["habit_messages"]]
    got["question_summary"] = ed.question_summary(i["question_rows"], reps=300, seed=0)
    got["question_summary_empty"] = ed.question_summary([], reps=10)
    got["habit_summary"] = ed.habit_summary(i["habit_rows"], reps=300, seed=0)
    got["calibration_report"] = ed.calibration_report(i["calibration_rows"])
    got["calibration"] = ed.CALIBRATION
    got["lr_at"] = [ed.lr_at(*c) for c in i["lr_at"]]
    got["heldout_indices"] = ed.heldout_indices(6000, 100, 0)
    got["run_of"] = [ed.run_of(n) for n in i["names"]]
    got["habit_sets_for"] = [ed.habit_sets_for(n) for n in i["names"]]
    got["adapter_path"] = [ed.adapter_path(n) for n in i["adapters"]]
    got["adapter_label"] = [ed.adapter_label(n) for n in i["adapters"]]
    got["cost_estimate"] = ed.cost_estimate(10)
    comp = ed.compare(i["summaries"])
    got["compare"] = comp
    full = {k: {"questions": ed.question_summary(i["question_rows"][:12], reps=50), "habit": ed.habit_summary(i["habit_rows"][:6], reps=50)}
            for k in ("base", "em__insecure_none_s0")}
    got["table_lines"] = ed.table_lines(full, ed.compare(full))
    got["table_lines_comparison"] = ed.table_lines(full, comp)
    assert_same(got, g["outputs"])


# ----------------------------------------------------------------------------------------------------------- battery


def test_battery_scoring_and_summary(tmp_path, capsys):
    from anatta.bench import battery as mb

    g = load("golden_battery.json")
    i = g["inputs"]
    got = {"score": [list(mb.score(c["item"], c["text"])) for c in i["scores"]],
           "letters_prompt": [mb.letters_prompt(q, o, L) for q, o, L in i["letters_prompt"]],
           "paired": [list(mb.paired(b, o, r)) for b, o, r in i["paired"]],
           "adapter_tag": [mb.adapter_tag(n) for n in i["tags"]],
           "key_of": [mb.key_of(n, arm) for n in i["tags"] for arm in ("base", "cut")],
           "constants": {"N": mb.N, "SETS": mb.SETS, "MAX_NEW": mb.MAX_NEW, "NUM": mb.NUM}}
    results = i["results"]
    a = tmp_path / "with_base"
    a.mkdir()
    for k, v in results.items():
        (a / f"{k}.json").write_text(json.dumps(v))
    (a / "items.json").write_text("[]")
    got["summarize_with_base"] = mb.summarize(a)
    b = tmp_path / "with_reference"
    b.mkdir()
    for k, v in results.items():
        if k != "base__base":
            (b / f"{k}.json").write_text(json.dumps(v))
    (tmp_path / "base.json").write_text(json.dumps(results["base__base"]))
    got["summarize_with_reference"] = mb.summarize(b, tmp_path / "base.json")
    c = tmp_path / "no_reference"
    c.mkdir()
    (c / "x__base.json").write_text(json.dumps(results["sft__koan_s0__base"]))
    got["summarize_no_reference"] = mb.summarize(c, tmp_path / "missing.json")
    assert_same(got, g["outputs"])
    assert "no reference rows" in capsys.readouterr().out


# ----------------------------------------------------------------------------------------------------------- packing


@pytest.mark.parametrize("model_id", ["mistralai/Ministral-3-8B-Instruct-2512-BF16", "Qwen/Qwen2.5-7B-Instruct"])
def test_chat_packing_token_ids(model_id):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from anatta.evals.em import turn_ids
    from anatta.player import pack

    g = load("golden_chat_packing.json")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)  # as anatta.models.load loads it
    except Exception as e:  # not in the local cache (or no network): nothing to compare against
        pytest.skip(f"{model_id}'s tokenizer is not cached here ({type(e).__name__})")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    want = g["outputs"][model_id]
    recs, items = [], []
    for chat in g["inputs"]["chats"]:
        rec = {}
        try:
            p, c = turn_ids(tok, chat)
        except ValueError as e:
            rec["turn_ids_error"] = str(e)
            p, c = turn_ids(tok, chat, strict=False)
            rec["loose"] = True
        rec["prompt"], rec["completion"] = p, c
        items.append((p, c))
        recs.append(rec)
    full, attn, cm, wc = pack(items, tok.pad_token_id, "cpu")
    got = {"pad_token_id": tok.pad_token_id, "items": recs,
           "pack": {"full": full.tolist(), "attn": attn.tolist(), "cm": cm.tolist(), "wc": wc}}
    assert norm(got) == want
