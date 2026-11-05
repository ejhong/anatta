"""The step-5 runner (anatta.runner), offline: spec parsing and validation, the tier gates (registration, clean tree),
fresh folders and no overwrite, the caps, seed pairing, organism verification before use, the measure interface with
a fake generator, the whole plan -> trial -> run -> score -> card path on a fake remote and judge, and the OR-Bench
measure's pure parts. No Modal, no network, no key; results and spend go to tmp_path. Claude (Opus 5.5) wrote this
file, 4 Nov 2026."""

from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest

import anatta
from anatta import spend
from anatta.backends.registry import get as model_entry
from anatta.config import REPO
from anatta.measures import orbench
from anatta.runner import gates, interventions, stages
from anatta.runner import measures as M
from anatta.runner import spec as S
from anatta.runner.gates import Refused

FAKE = "fake_runner_measure"


# ------------------------------------------------------------------------------------------------- fixtures


def _fake_measure(temperature: float = 0.0):
    m = types.ModuleType(FAKE)
    m.ID = FAKE
    m.JUDGE_USD_PER_CALL = 0.001
    m.JUDGE_MODELS = {}
    m.load_items = lambda opts: [{"id": i, "q": f"question {i}"} for i in range(int(opts.get("n", 4)))]
    m.items_record = lambda items: {"n": len(items)}
    m.requests = lambda items: [{"rid": f"fake|{it['id']}", "messages": [{"role": "user", "content": it["q"]}],
                                 "temperature": temperature, "max_tokens": 8} for it in items]

    def run(gen, items):
        res = {r["rid"]: r for r in gen(m.requests(items))}
        return [{"id": it["id"], "text": res[f"fake|{it['id']}"]["text"],
                 "finish_reason": res[f"fake|{it['id']}"]["finish_reason"]} for it in items]

    def judge(units, items, ask):
        got = ask([{"jid": f"fake|{u['id']}", "model": "judge/x", "messages": [], "params": {}} for u in units])
        return [{**u, "cls": got[f"fake|{u['id']}"].get("text"), "judge_error": got[f"fake|{u['id']}"].get("error")}
                for u in units]

    def score(units):
        k = sum(u["cls"] == "yes" for u in units)
        return {"units": {str(u["id"]): int(u["cls"] == "yes") for u in units}, "n_samples": len(units),
                "unparsed": 0, "rate": k / len(units)}

    m.run, m.judge, m.score = run, judge, score
    m.judge_calls = lambda n: n
    m.reference_units = lambda path, ids: {i: int(json.loads(open(path).read())[str(i)]) for i in ids}
    m.reference_replies = lambda path, ids: {i: ("yes" if i % 2 else "no") if i != 1 else "different" for i in ids}
    m.parsed = lambda u: u.get("cls") is not None
    return m


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Results and spend under tmp_path; the fake measure registered; a dirty-tree git state by default."""
    monkeypatch.setenv("ANATTA_RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("ANATTA_SPEND_DIR", str(tmp_path / "spend"))
    monkeypatch.setitem(sys.modules, FAKE, _fake_measure())
    monkeypatch.setitem(M.IMPL, FAKE, FAKE)
    real = M.registry()
    monkeypatch.setattr(M, "registry", lambda: {**real, FAKE: {"status": "active", "calibrated_on": {}}})
    import anatta.repro as repro

    state = {"code": {"commit": "abc123", "dirty": True}}
    monkeypatch.setattr(repro, "git_state", lambda: dict(state["code"]))
    notes = tmp_path / "NOTES.md"
    notes.write_text("# ledger\n")
    monkeypatch.setattr(repro, "NOTES", notes)
    return types.SimpleNamespace(tmp=tmp_path, state=state, notes=notes)


def _spec(tmp_path, name="t1", **over) -> str:
    import yaml

    raw = {"name": name, "tier": "explore", "about": "test", "models": ["ministral3_8b"], "organisms": [],
           "interventions": ["none"], "measures": {FAKE: {"n": 4}}, "seeds": [0], "caps": {"usd": 3.0, "batch": 2},
           "reproduction": {"measure": FAKE, "statistic": "rate", "target": 0.5, "band": [0.25, 0.75]}}
    raw.update(over)
    p = tmp_path / f"{name}-{abs(hash(json.dumps(raw, sort_keys=True, default=str))) % 10**8}.yaml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


class FakeRemote:
    def __init__(self, fail_load=False, facts=None):
        self.calls, self.verified, self.fail_load, self.facts = [], [], fail_load, facts

    def generate(self, model_key, lora, requests):
        self.calls.append((model_key, lora, [dict(r) for r in requests]))
        info = {"path": model_entry(model_key).snapshot(), "load_check": not self.fail_load,
                "settings": {"enable_lora": lora}}
        return {"results": [{"rid": r["rid"], "text": "yes" if int(r["rid"].split("|")[1]) % 2 else "no",
                             "finish_reason": "stop", "n_out": 3, "token_ids": [1, 2, 3], "seed": r.get("seed")}
                            for r in requests], "gen_s": 0.01 * len(requests), "info": info}

    def verify_organisms(self, keys):
        from anatta import organisms

        self.verified.append(list(keys))
        return {k: self.facts if self.facts is not None else organisms.get(k)["files"] for k in keys}


def fake_ask_factory(record: list):
    def factory(out, cap, measure):
        def ask(jobs):
            record.extend(j["jid"] for j in jobs)
            return {j["jid"]: {"text": "yes" if int(j["jid"].split("|")[1]) % 2 else "no"} for j in jobs}
        return ask
    return factory


# ------------------------------------------------------------------------------------------------- the spec


def test_a_valid_spec_expands_into_conditions(lab):
    sp = S.load(_spec(lab.tmp, seeds=[0, 1], interventions=["none", {"kind": "system_prompt", "text": "honesty"}],
                      reproduction={"measure": FAKE, "statistic": "rate", "target": 0.5, "band": [0, 1],
                                    "condition": "ministral3_8b__base__none__s0"}))
    assert [c.id for c in sp.conditions] == ["ministral3_8b__base__none__s0", "ministral3_8b__base__none__s1",
                                             "ministral3_8b__base__system_prompt-honesty__s0",
                                             "ministral3_8b__base__system_prompt-honesty__s1"]
    assert len(sp.sha256) == 64 and sp.tier == "explore"


@pytest.mark.parametrize("over, msg", [
    ({"colour": "blue"}, "unknown keys"),
    ({"tier": "quick"}, "tier"),
    ({"models": ["ministral"]}, "alias"),
    ({"models": ["no_such_model"]}, "no model"),
    ({"measures": {"no_such_measure": {}}}, "not in src/anatta/measures/registry.yaml"),
    ({"measures": {"mask_p_lie": {}}}, "no runner implementation"),
    ({"seeds": [0, 0]}, "distinct"),
    ({"caps": {"usd": 3.0}}, "caps"),
    ({"caps": {"usd": 500.0, "batch": 2}}, "above lab.toml"),
    ({"caps": {"usd": 3.0, "batch": 0}}, "batch"),
    ({"interventions": ["calm"]}, "intervention"),
    ({"interventions": [{"kind": "lobotomy"}]}, "not one of"),
    ({"seeds": [0, 1]}, "reproduction.condition"),
    ({"reproduction": {"measure": FAKE, "statistic": "rate", "target": 0.5, "band": [0.9, 0.1]}}, "empty"),
    ({"reproduction": {"measure": FAKE, "statistic": "rate", "target": {"card": "results/cards/k6b.json",
                                                                         "path": "orbench.nope"}, "band": [0, 1]}},
     "has no orbench.nope"),
])
def test_bad_specs_are_refused_with_the_reason(lab, over, msg):
    with pytest.raises(S.SpecError, match=msg):
        S.load(_spec(lab.tmp, **over))


def test_a_band_and_target_read_from_a_card():
    card = json.loads((REPO / "results/cards/k6b.json").read_text())
    assert S.resolve_band({"card": "results/cards/k6b.json", "path": "orbench.ci95"}) == tuple(card["orbench"]["ci95"])
    g = stages.grade({"measure": "m", "statistic": "rejection", "band": [0.1, 0.2],
                      "target": {"card": "results/cards/k6b.json", "path": "orbench.rejection"}}, 0.15)
    assert g["verdict"] == "in_band" and g["target"] == card["orbench"]["rejection"]
    assert stages.grade({"measure": "m", "statistic": "s", "band": [0.1, 0.2], "target": 0.15}, 0.21)["verdict"] == \
        "outside_band"


def test_the_acceptance_spec_is_valid_and_explore(lab):
    sp = S.load(REPO / "experiments/acceptance/k6b_orbench_subsample.yaml")
    assert sp.tier == "explore" and [c.id for c in sp.conditions] == ["ministral3_8b__base__none__s0"]
    assert sp.measures["orbench_hard_rejection"]["n"] == 660 and sp.caps["usd"] <= 5


# ------------------------------------------------------------------------------------------------- interventions


def test_system_prompt_puts_the_registered_text_first_and_steer_is_a_named_hook(lab):
    from anatta.judge import library

    text = library.load_conditions()["conditions"]["honesty"]["text"]
    iv = interventions.build({"kind": "system_prompt", "text": "honesty"})
    msgs = iv.apply([{"role": "user", "content": "hi"}])
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(text) and msgs[-1]["content"] == "hi"
    assert iv.record["sha256"] and iv.record["text"] == "honesty"
    assert interventions.build({"kind": "none"}).apply([{"role": "user", "content": "hi"}]) == \
        [{"role": "user", "content": "hi"}]
    with pytest.raises(KeyError, match="not registered"):
        interventions.build({"kind": "system_prompt", "text": "no_such_text"})
    with pytest.raises(NotImplementedError, match="hook, not built"):
        interventions.build({"kind": "steer", "name": "ego"})
    with pytest.raises(S.SpecError, match="hook, not built"):
        stages.plan(_spec(lab.tmp, interventions=[{"kind": "ablate", "name": "x"}]), dry_run=True)


# ------------------------------------------------------------------------------------------------- the gates


def test_full_tier_needs_registration_then_a_clean_tree(lab):
    p = _spec(lab.tmp, tier="full")
    with pytest.raises(Refused, match="not registered"):
        stages.plan(p)
    # a dry run shows the plan of an unregistered spec and says the gate would refuse (5 Nov: so a full spec's
    # estimate can be checked before it is registered); it writes nothing
    out, dry = stages.plan(p, dry_run=True)
    assert out is None and "not registered" in dry["gates"] and dry["registration"] == "required and NOT found"
    lab.notes.write_text(f"registered: {S.load(p).sha256}\n")
    with pytest.raises(Refused, match="not clean"):
        stages.plan(p)
    assert "not clean" in stages.plan(p, dry_run=True)[1]["gates"]
    lab.state["code"] = {"commit": "abc123", "dirty": False}
    out, plan = stages.plan(p)
    assert out.is_dir() and plan["tier"] == "full"


def test_explore_runs_from_a_dirty_tree_and_records_it(lab):
    out, _ = stages.plan(_spec(lab.tmp))
    plan = json.loads((out / "plan.json").read_text())
    assert plan["code"] == {"commit": "abc123", "dirty": True}
    assert plan["spec"]["registration"].startswith("not required (tier explore)")
    assert plan["spec"]["spec_file_sha256"] == S.load(plan["spec"]["spec_file"]).sha256
    assert out.parent.parent == lab.tmp / "results" / "runs" and out.name.startswith("run-")


def test_a_dry_run_writes_nothing(lab):
    out, plan = stages.plan(_spec(lab.tmp), dry_run=True)
    assert out is None and plan["dry_run"] and not (lab.tmp / "results" / "runs").exists()


def test_no_overwrite_of_a_run_folder_or_a_chunk(lab, monkeypatch):
    monkeypatch.setattr(gates, "stamp", lambda: "20261104T000000Z")
    p = _spec(lab.tmp)
    stages.plan(p)
    with pytest.raises(Refused, match="exists"):
        stages.plan(p)
    out = gates.open_run_folder("t1", S.load(p).sha256)
    # a chunk on disk that holds other requests is refused, never overwritten
    d = out / "data" / "gen" / "x"
    d.mkdir(parents=True)
    (d / "chunk_000.jsonl").write_text(json.dumps({"rid": "other"}) + "\n")
    sp = S.load(p)
    gen = stages.make_gen(None, sp.conditions[0], interventions.build({"kind": "none"}), FAKE, None, 2, d)
    with pytest.raises(Refused, match="other requests"):
        gen([{"rid": "fake|0", "messages": [], "temperature": 0.0, "max_tokens": 8}])


def test_a_stage_refuses_a_folder_planned_for_another_spec(lab):
    out, _ = stages.plan(_spec(lab.tmp))
    other = S.load(_spec(lab.tmp, about="changed"))
    with pytest.raises(Refused, match="another version"):
        gates.open_run_folder("t1", other.sha256, out)


def test_caps_refuse_past_the_day_and_past_the_run(lab):
    spend.add_spend(None, tool="other", label="x", usd=49.9)
    with pytest.raises(Refused, match="day's cap"):
        stages.plan(_spec(lab.tmp))
    assert "day's cap" in stages.plan(_spec(lab.tmp), dry_run=True)[1]["spend_at_plan"]["would_refuse"]
    with pytest.raises(Refused, match="day's cap"):
        gates.check_day(1.0)


def test_the_plan_refuses_an_estimate_past_caps_usd(lab):
    with pytest.raises(Refused, match="past caps.usd"):
        stages.plan(_spec(lab.tmp, caps={"usd": 0.01, "batch": 2}), dry_run=True)


def test_run_refuses_without_an_ok_trial_and_past_its_cap(lab):
    p = _spec(lab.tmp)
    out, _ = stages.plan(p)
    with pytest.raises(Refused, match="no ok trial"):
        stages.run(p, out, remote=FakeRemote())
    (out / gates.GPU_LOG).write_text(json.dumps({"cost_usd": 3.5}) + "\n")
    with pytest.raises(Refused, match="spent its cap"):
        stages.trial(p, out, remote=FakeRemote(), ask_factory=fake_ask_factory([]))


def test_a_failed_trial_blocks_the_run(lab):
    p = _spec(lab.tmp)
    out, _ = stages.plan(p)
    _, rec = stages.trial(p, out, remote=FakeRemote(fail_load=True), ask_factory=fake_ask_factory([]))
    assert rec["ok"] is False and rec["checks"]["load_check"] is False
    with pytest.raises(Refused, match="no ok trial"):
        stages.run(p, out, remote=FakeRemote())


# ------------------------------------------------------------------------------------------------- seeds


def test_seeds_are_paired_across_conditions_and_greedy_requests_carry_none(lab, monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE, _fake_measure(temperature=1.0))
    sp = S.load(_spec(lab.tmp, seeds=[0, 1], interventions=["none", {"kind": "system_prompt", "text": "calm"}],
                      reproduction=None))
    reqs = sys.modules[FAKE].requests(sys.modules[FAKE].load_items({"n": 3}))
    seeds = {c.id: [stages.prepare(r, c, interventions.build(c.intervention), FAKE, None)["seed"] for r in reqs]
             for c in sp.conditions}
    s0 = [v for k, v in seeds.items() if k.endswith("__s0")]
    s1 = [v for k, v in seeds.items() if k.endswith("__s1")]
    assert s0[0] == s0[1] and s1[0] == s1[1] and s0[0] != s1[0] and len(set(s0[0])) == 3
    greedy = {**reqs[0], "temperature": 0.0}
    assert stages.prepare(greedy, sp.conditions[0], interventions.build({"kind": "none"}), FAKE, None)["seed"] is None


# ------------------------------------------------------------------------------------------------- organisms


@pytest.fixture
def organisms_stub(monkeypatch):
    calls = []
    stub = types.ModuleType("anatta.organisms")
    files = {"adapter_model.safetensors": {"bytes": 10, "sha256": "a" * 64}}
    reg = {"org_a": {"kind": "lora", "base": "ministral3_8b", "path": "/runs/org/a", "files": files},
           "org_q": {"kind": "lora", "base": "qwen25_7b", "path": "/q"},
           "org_b": {"kind": "baked", "base": "ministral3_8b", "path": "/b"}}
    stub.get = lambda k: reg[k]

    def verify(k, root):
        calls.append((k, root))
        return {"ok": True}

    stub.verify = verify
    stub.files = files
    monkeypatch.setitem(sys.modules, "anatta.organisms", stub)
    monkeypatch.setattr(anatta, "organisms", stub, raising=False)
    return calls


def test_organisms_run_on_their_base_and_are_verified_before_use(lab, organisms_stub):
    with pytest.raises(S.SpecError, match="not in models"):
        S.load(_spec(lab.tmp, organisms=["org_q"]))
    with pytest.raises(S.SpecError, match="LoRA adapters only"):
        S.load(_spec(lab.tmp, organisms=["org_b"]))
    p = _spec(lab.tmp, organisms=[None, "org_a"], reproduction={"measure": FAKE, "statistic": "rate", "target": 0.5,
                                                                "band": [0, 1], "condition": "ministral3_8b__org_a__none__s0"})
    sp = S.load(p)
    assert [c.organism for c in sp.conditions] == [None, "org_a"]
    out, plan = stages.plan(p)
    assert plan["conditions"][1]["adapter"] == "/runs/org/a"
    remote = FakeRemote()
    stages.trial(p, out, remote=remote, ask_factory=fake_ask_factory([]))
    stages.run(p, out, remote=remote)
    assert ["org_a"] in remote.verified  # before the organism's generation
    served = [(lora, {r["adapter"] for r in reqs}) for _, lora, reqs in remote.calls]
    assert (True, {"/runs/org/a"}) in served and (False, {None}) in served


# ------------------------------------------------------------------------------------------------- end to end


def test_plan_trial_run_score_card_on_a_fake_remote(lab):
    p = _spec(lab.tmp)  # 4 items, batch 2: the trial makes chunk 0, the run reuses it and makes chunk 1
    out, _ = stages.plan(p)
    remote, judged = FakeRemote(), []
    _, t = stages.trial(p, out, remote=remote, ask_factory=fake_ask_factory(judged))
    assert t["ok"], t["checks"]
    assert t["n_requests"] == 2 and len(remote.calls) == 1 and len(remote.calls[0][2]) == 2
    with pytest.raises(Refused, match="already has an ok trial"):
        stages.trial(p, out, remote=remote)
    _, r = stages.run(p, out, remote=remote)
    assert r["complete"] and len(remote.calls) == 2  # chunk 0 came from the trial
    assert r["conditions"]["ministral3_8b__base__none__s0"][FAKE]["from_trial"] == 2
    _, sc = stages.score(p, out, ask_factory=fake_ask_factory(judged))
    assert sc["complete"] and sc["scores"]["ministral3_8b__base__none__s0"][FAKE]["rate"] == 0.5
    _, c = stages.card(p, out)
    assert c["reproduction"]["verdict"] == "in_band" and c["reproduction"]["observed"] == 0.5
    assert c["kind"].startswith("explore") and c["cost_usd"]["gpu"] > 0
    assert json.loads((out / "card.json").read_text())["written"]["plan_sha256"]
    with pytest.raises(Exception, match="refusing to overwrite"):
        stages.card(p, out)
    # replies never reach the stage records
    for f in ("trial.json", "run.json", "score.json", "card.json"):
        assert '"yes"' not in (out / f).read_text().replace('"cls"', "")
    # GPU spend was booked to the day file and the run's own log
    assert spend.spent_today() > 0 and gates.run_spent(out) > 0


def test_score_refuses_before_the_run(lab):
    p = _spec(lab.tmp)
    out, _ = stages.plan(p)
    with pytest.raises(Refused, match="no units yet"):
        stages.score(p, out, ask_factory=fake_ask_factory([]))


def test_newest_folder_is_used_without_a_run_dir(lab, monkeypatch):
    p = _spec(lab.tmp)
    stamps = iter(["20261104T000000Z", "20261104T000001Z"])
    monkeypatch.setattr(gates, "stamp", lambda: next(stamps))
    a, _ = stages.plan(p)
    b, _ = stages.plan(p)
    assert gates.open_run_folder("t1", S.load(p).sha256) == b != a


# ------------------------------------------------------------------------------------------------- OR-Bench measure


def test_orbench_measure_interface_with_a_fake_gen_and_judge():
    items = [{"id": i, "category": "c", "prompt": f"prompt {i}"} for i in range(5)]
    reqs = orbench.requests(items)
    assert [r["rid"] for r in reqs] == [f"orbench|{i}" for i in range(5)]
    assert all(r["temperature"] == 0.0 and r["max_tokens"] == 512 and r["messages"][0]["role"] == "user" for r in reqs)
    units = orbench.run(lambda rs: [{"rid": r["rid"], "text": "reply", "finish_reason": "length", "n_out": 512}
                                    for r in rs], items)
    answers = ["[[direct_answer]]", "[[direct_refusal]]", "x [[indirect_refusal]]", "no class here", None]
    judged = orbench.judge(units, items, lambda jobs: {j["jid"]: ({"text": answers[i]} if answers[i] else {"error": "e"})
                                                       for i, j in enumerate(jobs)})
    assert [u["cls"] for u in judged] == ["direct_answer", "direct_refusal", "indirect_refusal", None, None]
    with pytest.raises(ValueError, match="no judgment"):
        orbench.score(judged)
    s = orbench.score(judged[:4])
    assert s["rejection"] == 0.5 and s["unparsed"] == 1 and s["n_samples"] == 4  # unparsed stays in the denominator
    assert s["generation"]["truncated"] == 4 and s["ci95"][0] < 0.5 < s["ci95"][1]


def test_orbench_judge_prompt_is_k6s():
    spec = importlib.util.spec_from_file_location("k6p", REPO / "calibration/k6_battery/prompts.py")
    k6p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k6p)
    seen = []
    orbench.judge([{"id": 0, "text": "R"}], [{"id": 0, "prompt": "Q"}], lambda jobs: seen.extend(jobs) or {})
    assert seen[0]["messages"][0]["content"] == k6p.orbench_judge_prompt("Q", "R")
    assert seen[0]["model"] == "anthropic/claude-haiku-5.5" and seen[0]["params"]["max_tokens"] == 300


def test_orbench_subsample_is_stratified_and_fixed_by_its_salt():
    ids = [(i, "a" if i < 30 else "b") for i in range(100)]
    s1, s2 = orbench.strat_sample(ids, 10, "s"), orbench.strat_sample(ids, 10, "s")
    assert s1 == s2 and len(s1) == 10 and sum(i < 30 for i in s1) == 3
    assert orbench.strat_sample(ids, 10, "other") != s1


# ------------------------------------------------------------------------------------------------- shared helpers


def test_run_r1b_uses_the_shared_helpers():
    spec = importlib.util.spec_from_file_location("run_r1b_shared", REPO / "experiments" / "run_r1b.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    assert R.latest is gates.latest and R.file_facts is gates.file_facts


def test_organism_facts_from_the_volume_are_checked_against_the_registry(lab, organisms_stub):
    p = _spec(lab.tmp, organisms=["org_a"], reproduction=None)
    out, _ = stages.plan(p)
    for bad, msg in (({"adapter_model.safetensors": {"bytes": 10, "sha256": "b" * 64}}, "differ from the registry"),
                     ({"error": "boom"}, "not the files' facts"), ({}, "not the files' facts")):
        with pytest.raises(Refused, match=msg):
            stages.trial(p, out, remote=FakeRemote(facts=bad), ask_factory=fake_ask_factory([]))
    good = {"adapter_model.safetensors": {"bytes": 10, "sha256": "a" * 64}}
    assert gates.check_verified("org_a", good) == good


def _ref_spec(lab, verdicts: dict, **over):
    ref = lab.tmp / "ref_verdicts.json"
    ref.write_text(json.dumps(verdicts))
    replies = lab.tmp / "ref_replies.jsonl"
    replies.write_text("x\n")
    sha = lambda f: __import__("hashlib").sha256(f.read_bytes()).hexdigest()  # noqa: E731
    rep = {"measure": FAKE, "statistic": "rate", "target": "reference", "band": "reference",
           "reference": {"verdicts": {"file": str(ref), "sha256": sha(ref)},
                         "replies": {"file": str(replies), "sha256": sha(replies)}},
           "full_set": {"target": {"card": "results/cards/k6b.json", "path": "orbench.rejection"},
                        "band": {"card": "results/cards/k6b.json", "path": "orbench.ci95"}}}
    return _spec(lab.tmp, reproduction=rep, **over), ref


def test_same_item_grading_with_paired_discordance_and_identical_replies(lab):
    # ours (fake): ids 1, 3 positive; the reference: 0 and 1 positive -> reference-only {0}, runner-only {3}
    p, ref = _ref_spec(lab, {"0": 1, "1": 1, "2": 0, "3": 0})
    out, plan = stages.plan(p)
    assert plan["referenced_files"]["reference.verdicts"]["sha256"] and "results/cards/k6b.json" in plan["referenced_files"]
    remote = FakeRemote()
    stages.trial(p, out, remote=remote, ask_factory=fake_ask_factory([]))
    stages.run(p, out, remote=remote)
    stages.score(p, out, ask_factory=fake_ask_factory([]))
    _, c = stages.card(p, out)
    r = c["reproduction"]
    assert r["target"] == 0.5 and r["observed"] == 0.5 and r["verdict"] == "in_band"
    ref_rec = r["same_item_reference"]
    assert ref_rec["paired"] == {"reference_only": 1, "runner_only": 1, "both": 1, "neither": 1, "agreement": 0.5,
                                 "mcnemar_exact_p": 1.0}
    assert ref_rec["replies"]["byte_identical"] == 3 and "batch" in ref_rec["note"]
    assert r["full_set_ungraded"]["target"] == json.loads((REPO / "results/cards/k6b.json").read_text())["orbench"]["rejection"]


def test_a_reference_file_that_changed_is_refused(lab):
    p, ref = _ref_spec(lab, {"0": 1, "1": 1, "2": 0, "3": 0})
    out, _ = stages.plan(p)
    remote = FakeRemote()
    stages.trial(p, out, remote=remote, ask_factory=fake_ask_factory([]))
    stages.run(p, out, remote=remote)
    stages.score(p, out, ask_factory=fake_ask_factory([]))
    ref.write_text(json.dumps({"0": 0, "1": 0, "2": 0, "3": 0}))
    with pytest.raises(Refused, match="pins"):  # the pin no longer holds
        stages.card(p, out)
    with pytest.raises(Refused, match="pins"):
        stages.plan(p)
    assert "pins" in stages.plan(p, dry_run=True)[1]["referenced_files"]["would_refuse"]  # a dry run reports it


def test_a_card_the_spec_names_is_hashed_and_a_change_refused(lab, monkeypatch):
    p = _spec(lab.tmp, reproduction={"measure": FAKE, "statistic": "rate", "band": [0, 1],
                                     "target": {"card": "results/cards/k6b.json", "path": "orbench.rejection"}})
    out, plan = stages.plan(p)
    assert plan["referenced_files"]["results/cards/k6b.json"]["sha256"] == gates.sha256_file(REPO / "results/cards/k6b.json")
    remote = FakeRemote()
    stages.trial(p, out, remote=remote, ask_factory=fake_ask_factory([]))
    stages.run(p, out, remote=remote)
    stages.score(p, out, ask_factory=fake_ask_factory([]))
    real = gates.sha256_file
    monkeypatch.setattr(gates, "sha256_file", lambda f: "0" * 64 if str(f).endswith("k6b.json") else real(f))
    with pytest.raises(Refused, match="changed since the plan"):
        stages.card(p, out)


def test_a_regrade_reuses_an_earlier_runs_generation_without_the_gpu(lab, monkeypatch):
    stamps = iter(f"20261104T00000{i}Z" for i in range(9))
    monkeypatch.setattr(gates, "stamp", lambda: next(stamps))
    p1 = _spec(lab.tmp)
    first, _ = stages.plan(p1)
    remote, judged = FakeRemote(), []
    stages.trial(p1, first, remote=remote, ask_factory=fake_ask_factory(judged))
    stages.run(p1, first, remote=remote)
    n_calls = len(remote.calls)
    p2 = _spec(lab.tmp, about="re-grade")  # another spec file (sha), same requests
    second, plan = stages.plan(p2, reuse_run=str(first))
    assert plan["reuse_from"]["run"] == str(first)

    class NoGPU(FakeRemote):
        def generate(self, *a):
            raise AssertionError("a re-grade must not generate")

    _, t = stages.trial(p2, second, remote=NoGPU(), ask_factory=fake_ask_factory([]))
    assert t["ok"] and t["chunks"]["from_earlier_run"] == 2 and t["info_from"].endswith("trial.json")
    _, r = stages.run(p2, second, remote=NoGPU())
    assert r["complete"] and r["conditions"]["_served"]
    assert len(remote.calls) == n_calls and gates.run_spent(second) == 0
    stages.score(p2, second, ask_factory=fake_ask_factory([]))
    _, c = stages.card(p2, second)
    assert c["reuse_from"]["run"] == str(first) and c["cost_usd"]["total"] == 0
    with pytest.raises(Refused, match="differs from this spec"):
        stages.plan(_spec(lab.tmp, seeds=[5]), reuse_run=str(first))


def test_a_trial_counts_replies_per_request_when_an_item_makes_several(lab, monkeypatch):
    """VLAF's items make 20-60 requests each: the trial's reply check compares units with requests, not items
    (V1's first trial, 5 Nov, reported every_reply_back False with all 1,000 replies back)."""
    m = _fake_measure()
    m.requests = lambda items: [{"rid": f"fake|{it['id']}|{k}", "messages": [{"role": "user", "content": it["q"]}],
                                 "temperature": 0.0, "max_tokens": 8} for it in items for k in range(2)]

    def run(gen, items):
        res = {r["rid"]: r for r in gen(m.requests(items))}
        return [{"id": f"{it['id']}|{k}", "text": res[f"fake|{it['id']}|{k}"]["text"],
                 "finish_reason": res[f"fake|{it['id']}|{k}"]["finish_reason"]} for it in items for k in range(2)]

    m.run = run
    monkeypatch.setitem(sys.modules, FAKE, m)
    p = _spec(lab.tmp, caps={"usd": 3.0, "batch": 4})
    out, _ = stages.plan(p)
    _, t = stages.trial(p, out, remote=FakeRemote(), ask_factory=fake_ask_factory([]))
    assert t["n_requests"] == 4 and t["checks"]["every_reply_back"], t["checks"]


# ------------------------------------------------------------------------------------------------- provenance (5 Nov)


def test_the_served_judges_stack_reaches_score_and_card(lab, monkeypatch):
    monkeypatch.setattr(sys.modules[FAKE], "JUDGE_BACKEND", "vllm", raising=False)
    info = {"path": "/cache/judge", "vllm": "0.31.0", "transformers": "5.17.0", "torch": "2.13.0", "dtype": "bfloat16",
            "settings": {"ctx": 16384, "seed": 0}}

    def factory(out, cap, measure):
        ask = fake_ask_factory([])(out, cap, measure)
        ask.session = types.SimpleNamespace(info=info)
        return ask

    p = _spec(lab.tmp)
    out, _ = stages.plan(p)
    stages.trial(p, out, remote=FakeRemote(), ask_factory=fake_ask_factory([]))
    stages.run(p, out, remote=FakeRemote())
    _, sc = stages.score(p, out, ask_factory=factory)
    assert sc["judge_served"] == {FAKE: info}
    _, c = stages.card(p, out)
    assert c["judge_served"] == {FAKE: info}


def test_an_openrouter_judge_records_no_served_stack(lab):
    p = _spec(lab.tmp)
    out, _ = stages.plan(p)
    stages.trial(p, out, remote=FakeRemote(), ask_factory=fake_ask_factory([]))
    stages.run(p, out, remote=FakeRemote())
    _, sc = stages.score(p, out, ask_factory=fake_ask_factory([]))
    assert sc["judge_served"] == {}


def test_the_served_judge_passes_the_engines_sampling_through():
    from anatta.runner.judge import vllm_asker

    sp = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": None, "max_tokens": 4096, "repetition_penalty": 1.0}
    sess = types.SimpleNamespace(generate=lambda reqs: [{"rid": r["rid"], "text": "a", "finish_reason": "stop",
                                                         "sampling": sp} for r in reqs], close=lambda: 0.0, info={})
    got = vllm_asker(sess)([{"jid": "j", "messages": [], "params": {"temperature": 0.0}}])
    assert got["j"]["sampling"] == sp


def test_a_contrast_across_conditions_reaches_the_card(lab, monkeypatch):
    seen = []

    def contrast(units, plus, minus, variant):
        seen.append((sorted(units), plus, minus, variant))
        return {"diff": 1.5, "ci95": [0.5, 2.5]}

    monkeypatch.setattr(sys.modules[FAKE], "contrast", contrast, raising=False)
    a, b = "ministral3_8b__base__none__s0", "ministral3_8b__base__none__s1"
    rep = {"measure": FAKE, "grades": [{"name": "r", "condition": a, "statistic": "rate", "band": [0, 1]}],
           "contrasts": [{"name": "x", "plus": [a], "minus": [b], "variant": "v", "note": "n"}]}
    p = _spec(lab.tmp, seeds=[0, 1], reproduction=rep)
    out, _ = stages.plan(p)
    stages.trial(p, out, remote=FakeRemote(), ask_factory=fake_ask_factory([]))
    stages.run(p, out, remote=FakeRemote())
    stages.score(p, out, ask_factory=fake_ask_factory([]))
    _, c = stages.card(p, out)
    assert c["reproduction"]["contrasts"] == {"x": {"diff": 1.5, "ci95": [0.5, 2.5], "note": "n"}}
    assert seen == [([a, b], [a], [b], "v")]


@pytest.mark.parametrize("bad,msg", [
    ({"name": "x", "plus": [], "minus": ["B"], "variant": "v"}, "each"),
    ({"name": "x", "plus": ["A"], "minus": ["nope"], "variant": "v"}, "unknown or shared"),
    ({"name": "x", "plus": ["A"], "minus": ["A"], "variant": "v"}, "unknown or shared"),
    ({"name": "x", "plus": ["A"], "minus": ["B"]}, "each")])
def test_bad_contrasts_are_refused(lab, monkeypatch, bad, msg):
    monkeypatch.setattr(sys.modules[FAKE], "contrast", lambda *a: {}, raising=False)
    ids = {"A": "ministral3_8b__base__none__s0", "B": "ministral3_8b__base__none__s1"}
    x = {**bad, **{k: [ids.get(c, c) for c in bad[k]] for k in ("plus", "minus") if k in bad}}
    rep = {"measure": FAKE, "grades": [{"name": "r", "condition": ids["A"], "statistic": "rate", "band": [0, 1]}],
           "contrasts": [x]}
    with pytest.raises(S.SpecError, match=msg):
        S.load(_spec(lab.tmp, seeds=[0, 1], reproduction=rep))


def test_contrasts_need_a_measure_with_contrast(lab):
    a = "ministral3_8b__base__none__s0"
    rep = {"measure": FAKE, "grades": [{"name": "r", "condition": a, "statistic": "rate", "band": [0, 1]}],
           "contrasts": [{"name": "x", "plus": [a], "minus": [a.replace("s0", "s1")], "variant": "v"}]}
    with pytest.raises(S.SpecError, match="no contrast"):
        S.load(_spec(lab.tmp, seeds=[0, 1], reproduction=rep))
