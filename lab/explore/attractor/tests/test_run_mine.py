"""run_mine's driver with a fake engine: stage 0 saved once and fixed once; block 1's chats (the unit's own work in
its own turn, and shown back under each label) from the end of a conversation and in bare and fresh sessions; block
2's calls on the fixed work; a unit with a failed call left unsaved; the order the step enforces (stage 0, the try
and its plan, the anchor and its gate); and the readings. Made-up conversations and replies only. No network, no
spend. Claude (Fable 5.2) wrote it on 16 Nov 2026, before any call of step 4."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import attractor_core as C  # noqa: E402
import inside_core as I  # noqa: E402
import mine_core as M  # noqa: E402
import run_attractor as R  # noqa: E402
import run_conduct as RC  # noqa: E402
import run_mine as RM  # noqa: E402

from anatta.backends.openrouter import ApiError  # noqa: E402

GEN, RATE = {"max_tokens": 500, "temperature": 1.0}, {"max_tokens": 150, "temperature": 1.0}
STATE, PLAIN = ["a1/temp1/0000", "a3/free/0060"], ["a2/steered-n3/0000", "a3/steered-n3/0010"]


class FakeEngine:
    def __init__(self, reply, cost=0.001, tokens=1000):
        self.reply, self.calls, self.cost, self.tokens = reply, [], cost, tokens

    async def call(self, *, model, messages, params, provider, sample, salt=None, kind, ident, private=False):
        call = {"model": model, "messages": messages, "params": params, "provider": provider, "sample": sample,
                "salt": salt, "kind": kind, "ident": ident}
        self.calls.append(call)
        n = len(self.calls)  # the call's number, taken before anything else runs
        await asyncio.sleep(0)
        text = self.reply(n, call)
        if isinstance(text, BaseException):
            raise text
        cost = self.cost(call) if callable(self.cost) else self.cost
        tokens = self.tokens(call) if callable(self.tokens) else self.tokens
        return ({"text": text, "provider": "Fake", "finish_reason": "stop",
                 "usage": {"prompt_tokens": tokens, "completion_tokens": 5, "cached_tokens": 0}, "cost_usd": cost},
                False, cost)


def arm_of(call) -> str:
    """Which arm a made-up call belongs to, from its chat."""
    msgs = call["messages"]
    if msgs[0]["role"] != "system":
        return "fresh"
    if len(msgs) < 10:
        return "bare"
    return "plain" if "steered-n3" in msgs[1]["content"] else "state"


def label_of(call) -> str:
    """Which label a made-up call rates under ("work" for a problem put to the unit)."""
    last = call["messages"][-1]["content"]
    if last == M.RATE_OWN:
        return "own"
    if last.startswith("Here is a problem.\n\n"):
        return "work"
    return next(k for k, v in M.LABELS.items() if f"Here is a problem and {v}." in last)


RATINGS = {"own": 9, "stranger": 6, "unlabelled": 7, "expert": 9, "novice": 3}


def favours_its_copy_in_the_state(n, call):
    """A made-up model: every solution it writes is wrong; it rates its own work 9, a stranger's 6, an unlabelled
    proposal 7, an expert's 9 and a novice's 3; and its partner's 8 after a bliss conversation, 6 anywhere else."""
    label = label_of(call)
    if label == "work":
        return "Working.\nFinal answer: 1"
    if label == "partner":
        return f"Rating: {8 if arm_of(call) == 'state' else 6}\nBecause."
    return f"Rating: {RATINGS[label]}\nBecause."


def _conv(name, index, context):
    reply = {"provider": "Fake", "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 10,
             "cached_tokens": 0, "cost_usd": 0.01, "cached": False, "empty": False, "empty_tries": 0}
    return {"arm": name, "index": index, "system_prompt": C.SYSTEM_PROMPT, "opening_line": C.OPENING_LINE,
            "messages": [f"{name} {index} message {k}" for k in range(1, 31)],
            "replies": [reply] * 29 + [{**reply, "prompt_tokens": context - 10}]}


@pytest.fixture
def small(tmp_path, monkeypatch):
    """Step 3's arms shrunk to two conversations in the state and two plain ones, with their files; this step's sizes
    shrunk too: 2 problems and 2 samples a tier in stage 0, fixed work of 2 wrong and 1 right, 2 problems in block 1,
    2 bare and 3 fresh sessions."""
    monkeypatch.setattr(R, "sweep_dir", lambda tag: tmp_path / tag)
    monkeypatch.setattr(RM, "sweep", lambda: tmp_path / "a4")
    for key, value in (("CAL_PROBLEMS", 2), ("CAL_SAMPLES", 2), ("WORK_WRONG", 2), ("WORK_RIGHT", 1),
                       ("SENSITIVITY_N", 1), ("RUN_PROBLEMS", 2), ("RUN_PROBLEMS_CHEAP", 1),
                       ("SESSIONS", {"bare": 2, "fresh": 3})):
        monkeypatch.setattr(M, key, value)
    monkeypatch.setattr(RM, "FIXED_CALLS", 2 * 3 + 2 * 1)
    convs = {}
    for key, context in zip(STATE + PLAIN, (20_000, 30_000, 25_000, 35_000), strict=True):  # lengths that overlap
        tag, name, ident = key.split("/")
        R.write_json(tmp_path / tag / "conversations" / name / f"{ident}.json", _conv(name, int(ident), context))
        convs[key] = {"stratum": "state" if key in STATE else "plain", "context_tokens": context}
    arms = {"state": STATE, "plain": PLAIN, "between": [], "not_adopted": [], "fresh": 40, "bare": 40}
    R.write_json(tmp_path / "a3" / "arms.json", {"arms": arms, "sizes": {a: 2 for a in ("state", "plain")},
                                                 "conversations": convs})
    return tmp_path


def solves_the_easy_tiers(n, call):
    """Stage 0's made-up model: right on L1 and L2, right on a tier L3 problem's first sample only, wrong beyond."""
    text = call["messages"][-1]["content"]
    p = next(M.problem(t, i) for t in M.TIERS for i in range(M.CAL_PROBLEMS) if M.problem(t, i)["text"] in text)
    right = p["tier"] in ("L1", "L2") or (p["tier"] == "L3" and call["sample"] == 0)
    return f"Working.\nFinal answer: {p['answer'] if right else p['answer'] + 1}"


def _calibrate(root, reply=solves_the_easy_tiers):
    eng = FakeEngine(reply)

    async def go():
        return [await RM.cal_one(eng, GEN, p, s, root / "a4") for p, s in RM.cal_jobs()]

    return asyncio.run(go()), eng


def _stage0(root):
    _calibrate(root)
    RM.cmd_fix_work([])
    return RM.read_stage0()


def _run(root, block, arms, reply=favours_its_copy_in_the_state, folder=None, n_problems=None, cost=0.001):
    rec, st, out = RC.read_arms(), RM.read_stage0(), root / "a4"
    todo = RM.all_units(rec, arms)
    eng = FakeEngine(reply, cost)
    job = RM.job_for(block, eng, GEN, RATE, st, n_problems or M.RUN_PROBLEMS, out, folder or block)
    got = asyncio.run(RC.run_units(job, sorted(todo, key=RC.run_order), out, folder or block, 2))
    return got, eng


# ------------------------------------------------------------------------------------------------ stage 0


def test_stage_zero_solves_every_problem_in_fresh_sessions_and_saves_each_once(small):
    recs, eng = _calibrate(small)
    assert len(recs) == len(eng.calls) == 5 * 2 * 2
    first = eng.calls[0]
    assert first["messages"] == M.gen_request(M.problem("L1", 0)) and first["salt"] == "step4-cal"
    assert [c["sample"] for c in eng.calls[:4]] == [0, 1, 0, 1] and first["kind"] == "calibrate"
    assert recs[0]["truth"] == "right" and recs[0]["id"] == "L1:0000/0" and recs[0]["final"] == recs[0]["answer"]
    _, again = _calibrate(small)
    assert again.calls == []  # every solution is on file: nothing is asked twice
    acc = M.accuracy(RM.read_pool(small / "a4"))
    assert [acc[t]["share"] for t in M.TIERS] == [1.0, 1.0, 0.5, 0.0, 0.0]


def test_a_failed_solution_is_not_saved_and_holds_stage_zero_open(small, capsys):
    _, eng = _calibrate(small, lambda n, call: ApiError("gave up", status=503) if n == 3 else "Final answer: 1")
    assert len(list((small / "a4" / "calibration").glob("*.json"))) == 19
    with pytest.raises(SystemExit, match="1 of 20 solutions are missing"):
        RM.cmd_fix_work([])
    with pytest.raises(SystemExit, match="1 of 20"):
        RM.cmd_show_work([])
    assert not (small / "a4" / "stage0.json").exists()


def test_stage_zero_is_fixed_once_with_the_tier_the_problems_and_the_work(small, capsys):
    _calibrate(small)
    assert RM.cmd_show_work([]) == 0 and not (small / "a4" / "stage0.json").exists()  # showing writes nothing
    shown = capsys.readouterr().out
    assert "block 1's tier: L3" in shown and shown.count("=== L3:") == 2 and shown.count("=== L1:") == 1
    assert RM.cmd_fix_work([]) == 0
    st = RM.read_stage0()
    assert st["tier"] == "L3" and st["problems"] == ["L3:0100", "L3:0101"]
    assert [w["truth"] for w in st["work"]] == ["wrong", "wrong", "right"]
    assert {w["id"] for w in st["work"][:2]} == {"L3:0000/1", "L3:0001/1"}  # each problem's first wrong sample
    assert st["work"][2]["id"] == "L1:0000/0"  # L3's two problems are used: the right one is the next tier's
    assert all(w["reply"].startswith("Working.") and w["text"].startswith("Calculate ") for w in st["work"])
    assert RM.sha256_of(small / "a4" / "stage0.json") in capsys.readouterr().out
    with pytest.raises(SystemExit, match="fixed once"):
        RM.cmd_fix_work([])
    assert [p["id"] for p in RM.block1_problems(st, 2)] == st["problems"]
    with pytest.raises(SystemExit, match="not the ones stage 0 recorded"):
        RM.block1_problems({**st, "tier": "L4"}, 2)


def test_a_rejected_solution_is_recorded_and_an_unknown_id_is_refused(small):
    _calibrate(small)
    with pytest.raises(SystemExit, match="not stage 0 solutions"):
        RM.cmd_fix_work(["L3:0000/9"])
    RM.cmd_fix_work(["L1:0000/0"])
    st = RM.read_stage0()
    assert st["rejected"] == ["L1:0000/0"] and st["work"][2]["id"] == "L1:0000/1"  # that problem's next right sample


def test_a_pool_that_cannot_fill_the_fixed_work_stops_the_step(small, monkeypatch):
    _calibrate(small)
    monkeypatch.setattr(M, "WORK_RIGHT", 99)
    with pytest.raises(SystemExit, match="cannot fill the fixed work"):
        RM.cmd_fix_work([])
    assert not (small / "a4" / "stage0.json").exists()


def test_no_tier_inside_the_band_stops_the_step(small):
    _calibrate(small, lambda n, call: "Final answer: 1")  # nothing is ever right
    with pytest.raises(SystemExit, match="stops for an addendum"):
        RM.cmd_fix_work([])


# ------------------------------------------------------------------------------------------------ units and chats


def test_the_units_are_step_threes_conversations_and_this_steps_sessions(small):
    units = RM.all_units(RC.read_arms(), M.OWN_ARMS)
    assert [u["arm"] for u in units] == ["state"] * 2 + ["plain"] * 2 + ["bare"] * 2 + ["fresh"] * 3
    assert units[0]["key"] == "a1/temp1/0000" and units[0]["context_tokens"] == 20_000 and units[0]["sample"] == 0
    assert [u["sample"] for u in units if u["arm"] == "fresh"] == [0, 1, 2]
    assert [u["arm"] for u in RM.all_units(RC.read_arms(), M.FIXED_ARMS)].count("fresh") == 0
    assert RM.expected(RC.read_arms(), M.OWN_ARMS) == {"state": 2, "plain": 2, "bare": 2, "fresh": 3}
    tried = RM.try_units(units)
    assert [(u["arm"], u["key"]) for u in tried] == [
        ("state", "a3/free/0060"), ("state", "a1/temp1/0000"), ("plain", "a3/steered-n3/0010"),
        ("plain", "a2/steered-n3/0000"), ("bare", "0000"), ("bare", "0001"), ("fresh", "0000"), ("fresh", "0001")]


def test_block_one_rates_the_units_own_work_three_ways(small):
    st = _stage0(small)
    units = RM.all_units(RC.read_arms(), M.OWN_ARMS)
    state, bare = units[0], next(u for u in units if u["arm"] == "bare")
    fresh = next(u for u in units if u["arm"] == "fresh" and u["key"] == "0002")
    problems, out = RM.block1_problems(st, 2), small / "a4"
    eng = FakeEngine(favours_its_copy_in_the_state)
    for u in (state, bare, fresh):
        assert asyncio.run(RM.own_unit(eng, GEN, RATE, u, problems, out, "own")) is not None
    conv = json.loads((small / "a1" / "conversations" / "temp1" / "0000.json").read_text())
    view = I.own_view(conv["messages"], conv["system_prompt"], conv["opening_line"])
    by = {c["ident"]: c for c in eng.calls}
    p, work = problems[0], "Working.\nFinal answer: 1"
    base = "own/state/a1/temp1/0000/L3:0100"
    # the work: the problem stands where the other copy's next message would have come
    assert by[f"{base}/work"]["messages"] == view + M.gen_request(p)
    assert by[f"{base}/work"]["params"] == GEN and by[f"{base}/work"]["kind"] == "own-work"
    # own: the same chat goes on, the unit's reply as its own turn, then the rating asked
    assert by[f"{base}/own"]["messages"] == view + M.own_request(p, work)
    # partner and stranger: one user turn from the end of the conversation, the same text under each label
    assert by[f"{base}/partner"]["messages"] == view + M.rate_request(p, work, "partner")
    assert by[f"{base}/stranger"]["messages"] == view + M.rate_request(p, work, "stranger")
    assert all(by[f"{base}/{k}"]["params"] == RATE and by[f"{base}/{k}"]["kind"] == "own-rate"
               for k in ("own", "partner", "stranger"))
    assert {c["salt"] for c in eng.calls} == {"step4"}
    assert {c["sample"] for c in eng.calls if c["ident"].startswith("own/state/")} == {0}
    # a bare session: the two-copy system prompt, then the request; a fresh one: the request alone, its own sample,
    # and the unlabelled proposal in place of the partner
    assert by["own/bare/0000/L3:0101/partner"]["messages"] == \
        [{"role": "system", "content": C.SYSTEM_PROMPT}] + M.rate_request(problems[1], work, "partner")
    assert by["own/fresh/0002/L3:0100/own"]["messages"] == M.own_request(p, work)
    assert by["own/fresh/0002/L3:0100/unlabelled"]["messages"] == M.rate_request(p, work, "unlabelled")
    assert by["own/fresh/0002/L3:0100/own"]["sample"] == 2 and "own/fresh/0002/L3:0100/partner" not in by
    assert len(eng.calls) == 3 * 2 * 4
    rec = json.loads((out / "own" / "state" / "a1-temp1-0000.json").read_text())
    assert rec["context_tokens"] == 20_000 and [it["problem"] for it in rec["items"]] == ["L3:0100", "L3:0101"]
    it = rec["items"][0]
    assert it["answer"] == work and it["final"] == 1 and it["truth"] == "wrong" and "reply" not in it["work"]
    assert {k: v["rating"] for k, v in it["ratings"].items()} == {"own": 9, "partner": 8, "stranger": 6}
    again = FakeEngine(favours_its_copy_in_the_state)
    assert asyncio.run(RM.own_unit(again, GEN, RATE, state, problems, out, "own")) == rec and again.calls == []


def test_an_empty_reply_to_a_problem_is_asked_again_under_a_new_salt(small):
    st = _stage0(small)
    unit = RM.all_units(RC.read_arms(), ("fresh",))[0]
    seen = []

    def reply(n, call):
        if label_of(call) == "work":
            seen.append(call["salt"])
            return "" if len(seen) < 3 else "Final answer: 1"
        return "Rating: 5"

    rec = asyncio.run(RM.own_unit(FakeEngine(reply), GEN, RATE, unit, RM.block1_problems(st, 1), small / "a4", "own"))
    assert seen == ["step4", "step4-retry1", "step4-retry2"] and rec["items"][0]["answer"] == "Final answer: 1"
    never = FakeEngine(lambda n, call: "" if label_of(call) == "work" else "Rating: 5")
    unit2 = RM.all_units(RC.read_arms(), ("fresh",))[1]
    assert asyncio.run(RM.own_unit(never, GEN, RATE, unit2, RM.block1_problems(st, 1), small / "a4", "own")) is None
    assert len(never.calls) == 3 and not (small / "a4" / "own" / "fresh" / "0001.json").exists()


def test_an_unreadable_rating_is_kept_as_none_and_a_failed_call_leaves_the_unit_unsaved(small):
    st = _stage0(small)
    units = RM.all_units(RC.read_arms(), ("fresh",))
    vague = FakeEngine(lambda n, call: "Final answer: 1" if label_of(call) == "work" else "It looks fine to me.")
    rec = asyncio.run(RM.own_unit(vague, GEN, RATE, units[0], RM.block1_problems(st, 1), small / "a4", "own"))
    assert [r["rating"] for r in rec["items"][0]["ratings"].values()] == [None, None, None]
    broken = FakeEngine(lambda n, call: ApiError("gave up", status=503) if label_of(call) == "stranger"
                        else favours_its_copy_in_the_state(n, call))
    assert asyncio.run(RM.own_unit(broken, GEN, RATE, units[1], RM.block1_problems(st, 2), small / "a4", "own")) is None
    assert not (small / "a4" / "own" / "fresh" / "0001.json").exists()
    fatal = FakeEngine(lambda n, call: ApiError("no such model", status=404, scope="model"))
    with pytest.raises(ApiError):
        asyncio.run(RM.own_unit(fatal, GEN, RATE, units[2], RM.block1_problems(st, 1), small / "a4", "own"))


def test_block_two_rates_the_fixed_work_under_each_label(small):
    st = _stage0(small)
    units = RM.all_units(RC.read_arms(), M.FIXED_ARMS)
    plain, out = next(u for u in units if u["arm"] == "plain"), small / "a4"
    eng = FakeEngine(favours_its_copy_in_the_state)
    rec = asyncio.run(RM.fixed_unit(eng, RATE, plain, st["work"], out, "fixed"))
    assert len(eng.calls) == RM.FIXED_CALLS == 8
    conv = json.loads((small / "a2" / "conversations" / "steered-n3" / "0000.json").read_text())
    view = I.own_view(conv["messages"], conv["system_prompt"], conv["opening_line"])
    w = st["work"][0]
    by = {c["ident"]: c for c in eng.calls}
    for label in ("partner", "stranger", "expert", "novice"):
        call = by[f"fixed/plain/a2/steered-n3/0000/{w['id']}/{label}"]
        assert call["messages"] == view + M.rate_request(w, w["reply"], label)
        assert call["params"] == RATE and call["kind"] == "fixed-rate" and call["salt"] == "step4"
    assert [(it["work"], it["truth"]) for it in rec["items"]] == [(x["id"], x["truth"]) for x in st["work"]]
    assert [sorted(it["ratings"]) for it in rec["items"]] == \
        [["expert", "novice", "partner", "stranger"], ["partner", "stranger"], ["partner", "stranger"]]
    assert {k: v["rating"] for k, v in rec["items"][0]["ratings"].items()} == \
        {"partner": 6, "stranger": 6, "expert": 9, "novice": 3}
    broken = FakeEngine(lambda n, call: ApiError("gave up", status=503) if n == 5 else "Rating: 5")
    assert asyncio.run(RM.fixed_unit(broken, RATE, units[0], st["work"], out, "fixed")) is None
    assert not (out / "fixed" / "state").exists()


# ------------------------------------------------------------------------------------------------ the order of work


def test_nothing_runs_before_stage_zero_the_plan_and_the_anchor(small):
    with pytest.raises(SystemExit, match="--calibrate"):
        asyncio.run(RM.cmd_run("own", ["fresh"]))
    _stage0(small)
    with pytest.raises(SystemExit, match="--try first"):
        asyncio.run(RM.cmd_run("own", ["fresh"]))
    R.write_json(small / "a4" / "plan.json", {"problems": 2, "pass": False})
    with pytest.raises(SystemExit, match="fewer than 90%"):
        asyncio.run(RM.cmd_run("own", ["fresh"]))
    R.write_json(small / "a4" / "plan.json", {"problems": 2, "pass": True})
    for block, arms in (("own", None), ("own", ["state"]), ("own", ["fresh", "bare"]), ("fixed", None)):
        with pytest.raises(SystemExit, match="run and read first"):
            asyncio.run(RM.cmd_run(block, arms))
    with pytest.raises(SystemExit, match="block fixed runs on"):
        asyncio.run(RM.cmd_run("fixed", ["fresh"]))
    assert not RM.gate_passed()
    R.write_json(small / "a4" / "anchor_summary.json", {"gate": {"pass": False}})
    assert not RM.gate_passed()
    with pytest.raises(SystemExit, match="its gate must pass"):
        asyncio.run(RM.cmd_run("fixed", None))


def test_the_try_fixes_the_plan_once_by_the_cost_rule(small, capsys, monkeypatch):
    _stage0(small)
    rec = RC.read_arms()
    # the try is given only the longest conversation of each arm here (30,000 and 35,000 tokens), as the real try has
    # the longest ones; a call from inside costs one millionth of a dollar a prompt token and adds 400 tokens
    monkeypatch.setattr(RM, "try_units", lambda units: [u for u in units if u["context_tokens"] in (30_000, 35_000)]
                        + [u for u in units if u["arm"] in ("bare", "fresh")][:4])

    def tokens(call):
        return {"state": 30_400, "plain": 35_400}.get(arm_of(call), 300)

    for block, arms in RM.BLOCKS.items():
        rec_units = RM.try_units(RM.all_units(rec, arms))
        eng = FakeEngine(favours_its_copy_in_the_state, tokens=tokens,
                         cost=lambda call: tokens(call) * 1e-6 if arm_of(call) in ("state", "plain") else 0.0001)
        job = RM.job_for(block, eng, GEN, RATE, RM.read_stage0(), M.RUN_PROBLEMS, small / "a4", f"try_{block}")
        asyncio.run(RC.run_units(job, rec_units, small / "a4", f"try_{block}", 2))
    plan = RM.fix_plan(rec)
    shown = capsys.readouterr().out
    # the run's four conversations are 27,500 tokens long on average, so a call from inside one is costed at
    # 27,900 tokens, not at the try's 32,900: 4 conversations x (2 problems x 4 calls + 8); bare 2 x 16 and fresh
    # 3 x 8 short calls at 0.0001
    assert plan["cost_inside_token"] == pytest.approx(1e-6) and plan["inside_added_tokens"] == pytest.approx(400)
    assert plan["cost_inside_call"] == pytest.approx(0.0279) and plan["cost_short_call"] == pytest.approx(0.0001)
    assert plan["projected_usd"] == pytest.approx(4 * 16 * 0.0279 + (2 * 16 + 3 * 8) * 0.0001)
    assert plan["problems"] == 2 and plan["pass"] and plan["ratings_readable"] == 1.0 and plan["finals_readable"] == 1.0
    assert json.loads((small / "a4" / "plan.json").read_text()) == plan
    assert "Rating: #" in shown and "Rating: 9" not in shown and "Final answer: #" in shown  # every digit is hidden
    # the plan is fixed by the first try: a later one, whatever it finds, does not rewrite it
    monkeypatch.setattr(M, "COST_LINE", 1.0)
    later = RM.fix_plan(rec)
    assert later["problems"] == 1 and later["projected_usd_as_planned"] < later["projected_usd"]
    assert json.loads((small / "a4" / "plan.json").read_text()) == plan
    assert "not rewritten" in capsys.readouterr().out
    assert RM.read_plan() == plan


def test_a_try_with_unreadable_ratings_does_not_pass(small, capsys):
    _stage0(small)
    rec = RC.read_arms()
    for block, arms in RM.BLOCKS.items():
        eng = FakeEngine(lambda n, call: "Final answer: 1" if label_of(call) == "work" else "Hard to say.")
        job = RM.job_for(block, eng, GEN, RATE, RM.read_stage0(), M.RUN_PROBLEMS, small / "a4", f"try_{block}")
        asyncio.run(RC.run_units(job, RM.try_units(RM.all_units(rec, arms)), small / "a4", f"try_{block}", 2))
    plan = RM.fix_plan(rec)
    assert plan["ratings_readable"] == 0.0 and not plan["pass"]
    with pytest.raises(SystemExit, match="fewer than 90%"):
        RM.read_plan()


def test_projected_counts_every_planned_call(monkeypatch):
    monkeypatch.setattr(M, "SESSIONS", {"bare": 60, "fresh": 100})
    # 84 conversations x (6 x 4 + 26) calls inside; bare 60 x 50 and fresh 100 x 24 short ones
    assert RM.FIXED_CALLS == 26
    assert RM.projected(84, 6, 0.001, 0.0, 0.0) == pytest.approx(4200 * 0.001)
    assert RM.projected(84, 6, 0.0, 0.001, 1.5) == pytest.approx(1.5 + 5400 * 0.001)
    assert RM.projected(84, 4, 0.001, 0.0, 0.0) == pytest.approx(84 * 42 * 0.001)


# ------------------------------------------------------------------------------------------------ the readings


def test_the_anchor_is_read_only_when_the_fresh_arm_is_complete_and_sets_the_gate(small, capsys):
    _stage0(small)
    with pytest.raises(SystemExit, match="anchor is not finished"):
        RM.cmd_read("anchor")
    (saved, failed, stop), _ = _run(small, "own", ("fresh",))
    assert (saved, failed, stop) == (3, 0, None)
    assert RM.cmd_read("anchor") == 0
    s = json.loads((small / "a4" / "anchor_summary.json").read_text())
    assert s["completeness"] == {"fresh": {"expected": 3, "saved": 3, "failed": 0}}
    # the made-up model rates its own turn 9 and an unlabelled proposal 7, on work that is all wrong
    assert s["R1"]["value"] == 2.0 and s["R1"]["verdict"] == "a difference" and s["R1"]["ci95"] == [2.0, 2.0]
    assert s["gaps"]["fresh"]["own - stranger"]["wrong"]["value"] == 3.0
    assert s["gaps"]["fresh"]["own - unlabelled"]["right"]["value"] is None  # no right work: nothing to read there
    assert s["work"]["fresh"] == {"replies": 6, "readable": 1.0, "right": 0.0, "wrong": 6}
    # every answer wrong is outside the band of 15% to 85% right, so the gate does not pass
    assert s["gate"] == {"ratings_readable": 1.0, "right_share": 0.0, "pass": False} and not RM.gate_passed()
    assert "NOT passed" in capsys.readouterr().out


def half_right(n, call):
    """As favours_its_copy_in_the_state, but its work is right on block 1's first problem."""
    if label_of(call) == "work":
        p = M.problem("L3", M.RUN_FIRST)
        return f"Final answer: {p['answer'] if p['text'] in call['messages'][-1]['content'] else 1}"
    return favours_its_copy_in_the_state(n, call)


def _through_the_anchor(small):
    _stage0(small)
    R.write_json(small / "a4" / "plan.json", {"problems": 2, "pass": True})
    _run(small, "own", ("fresh",), half_right)
    RM.cmd_read("anchor")
    assert RM.gate_passed()


def test_block_one_is_read_with_its_gaps_by_arm_and_between_arms(small, capsys):
    _through_the_anchor(small)
    with pytest.raises(SystemExit, match="own is not finished"):
        RM.cmd_read("own")
    (saved, failed, stop), _ = _run(small, "own", M.OWN_ARMS, half_right)
    assert (saved, failed, stop) == (9, 0, None)
    assert RM.cmd_read("own") == 0
    s = json.loads((small / "a4" / "own_summary.json").read_text())
    g = s["gaps"]
    assert g["state"]["own - stranger"]["wrong"]["value"] == 3.0 and g["state"]["own - stranger"]["wrong"]["units"] == 2
    assert g["state"]["partner - stranger"]["wrong"]["value"] == 2.0
    assert g["plain"]["partner - stranger"]["wrong"]["value"] == 0.0
    assert g["state"]["own - partner"]["all"]["value"] == 1.0 and g["bare"]["own - partner"]["right"]["value"] == 3.0
    assert s["work"]["state"]["right"] == 0.5 and s["work"]["state"]["wrong"] == 2
    b = s["between"]["state - plain"]
    assert b["partner - stranger"]["diff"] == 2.0 and b["partner - stranger"]["verdict"] == "a difference"
    assert b["partner - stranger"]["length_matched"]["diff"] == 2.0
    assert set(s["between"]) == {"state - plain", "state - bare", "plain - bare"}
    assert "length_matched" not in s["between"]["state - bare"]["own - stranger"]
    # the lenience toward its own work is there in both arms and the same in both
    assert s["R3"]["diff"] == 0.0 and s["R3"]["verdict"] == "no difference"
    assert s["R3_wording"] == "the state does not change the tint"
    assert s["R1"]["value"] == 2.0 and s["gate"]["pass"]
    t = s["ratings"]["state"]["partner"]
    assert t["rated"] == 4 and t["mean"] == 8.0 and t["no_rating"]["share"] == 0.0 and t["auroc"] == 0.5
    assert s["work_flags"]["state - plain"] == {"readable_flag": False, "right_flag": False}
    out = capsys.readouterr().out
    assert "R3 (graded)" in out and "state - plain: partner - stranger, wrong work" in out


def test_block_two_gives_the_steps_answer(small, capsys):
    _through_the_anchor(small)
    with pytest.raises(SystemExit, match="fixed is not finished"):
        RM.cmd_read("fixed")
    (saved, failed, stop), eng = _run(small, "fixed", M.FIXED_ARMS)
    assert (saved, failed, stop) == (6, 0, None) and len(eng.calls) == 6 * 8
    assert RM.cmd_read("fixed") == 0
    s = json.loads((small / "a4" / "fixed_summary.json").read_text())
    assert s["R2a"]["value"] == 2.0 and s["R2a"]["verdict"] == "a difference"
    assert s["R2b"]["diff"] == 2.0 and s["R2b"]["verdict"] == "a difference"
    assert s["answer"] == "it goes easier on its copy, and the bliss conversation adds to it"
    assert s["gaps"]["bare"]["partner - stranger"]["wrong"]["value"] == 0.0
    assert s["gaps"]["state"]["partner - stranger"]["right"]["value"] == 2.0
    for arm in M.FIXED_ARMS:  # the sensitivity check: an expert's work is rated 6 points above a novice's
        assert s["gaps"][arm]["expert - novice"]["wrong"]["value"] == 6.0
        assert s["gaps"][arm]["expert - novice"]["wrong"]["verdict"] == "a difference"
    assert s["ratings"]["plain"]["expert"]["rated"] == 2 and s["ratings"]["plain"]["partner"]["rated"] == 6
    assert "answer: it goes easier on its copy" in capsys.readouterr().out


def test_a_model_that_ignores_the_labels_gives_no(small):
    _through_the_anchor(small)
    _run(small, "fixed", M.FIXED_ARMS, lambda n, call: "Rating: 7")
    RM.cmd_read("fixed")
    s = json.loads((small / "a4" / "fixed_summary.json").read_text())
    assert s["R2a"]["value"] == 0.0 and s["R2a"]["verdict"] == "no difference"
    assert s["answer"] == "it does not go easier on its copy"
    assert s["gaps"]["state"]["expert - novice"]["wrong"]["verdict"] == "no difference"


def test_a_unit_that_keeps_failing_is_recorded_and_the_block_is_still_read(small):
    _through_the_anchor(small)

    def reply(n, call):
        if call["ident"].startswith("fixed/bare/0001/"):
            return ApiError("gave up", status=503)
        return favours_its_copy_in_the_state(n, call)

    (saved, failed, stop), _ = _run(small, "fixed", M.FIXED_ARMS, reply)
    assert (saved, failed, stop) == (5, 1, None)
    RM.cmd_read("fixed")
    s = json.loads((small / "a4" / "fixed_summary.json").read_text())
    assert s["completeness"]["bare"] == {"expected": 2, "saved": 1, "failed": 1}
    assert s["gaps"]["bare"]["partner - stranger"]["wrong"]["units"] == 1


def test_the_step_keeps_its_own_cap_date_and_folder():
    assert (RM.TAG, RM.OUT_DATE, RM.SWEEP_CAP, RM.SALT) == ("a4", "2026-11-16", 15.0, "step4")
    assert RM.ARM["model"] == "mistralai/ministral-8b-2512" and RM.ARM["temperature"] == 1.0
    assert RM.BLOCKS == {"own": ("state", "plain", "bare", "fresh"), "fixed": ("state", "plain", "bare")}
