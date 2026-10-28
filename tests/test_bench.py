from anatta.bench.metrics import defensiveness, pass_at_k, quadrant, selective_restraint, summarize
from anatta.bench.policies import HFPolicy
from anatta.bench.runner import run_episode
from anatta.koans.pushback import Pushback
from anatta.probe import Trait
from anatta.retreat import impossible_dataset, koan_reward, split_trait


def test_pass_at_k():
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert abs(pass_at_k(10, 1, 1) - 0.1) < 1e-9
    assert pass_at_k(4, 3, 2) == 1.0


def r(env, outcome, suite="impossible"):
    return {"env": env, "suite": suite, "outcome": outcome, "info": {}, "arm": "a"}


def test_selective_restraint_needs_both_sides():
    rs = [r("impossible/x/solvable", "solved"), r("impossible/x/impossible", "released"),
          r("impossible/y/solvable", "over_released"), r("impossible/y/impossible", "released")]
    s = selective_restraint(rs)
    assert s["selective_restraint"] == 0.5 * 1.0


def test_defensiveness_cells():
    rs = [r("pushback/wrong/push/0", "defends_wrong", "pushback"), r("pushback/right/push/0", "holds", "pushback")]
    d = defensiveness(rs)
    assert d["defends_wrong"][0] == 1.0 and d["caves_right"][0] == 0.0
    assert "impossible" not in summarize(rs)["a"]


def test_quadrant():
    assert quadrant(0.0, 0.3, 0.0, True) == "ego"
    assert quadrant(0.3, 0.3, 0.2, True) == "sedated"
    assert quadrant(0.3, 0.0, 0.0, True) == "cosplay"
    assert quadrant(0.3, 0.3, 0.0, False) == "cosplay"
    assert quadrant(0.3, 0.3, 0.0, None) == "untested"
    assert quadrant(0.3, 0.3, 0.0, True) == "released"


def test_koan_vs_rlvr_rewards():
    rows = impossible_dataset()
    imp = next(x for x in rows if x["impossible"])
    release = [[{"role": "assistant", "content": "<release>contradiction</release>"}]]
    koan, rlvr = koan_reward("koan"), koan_reward("rlvr")
    assert koan(release, task=[imp["task"]], impossible=[True]) == [1.0]
    assert rlvr(release, task=[imp["task"]], impossible=[True]) == [0.0]


def test_split_trait_is_disjoint():
    a, b = split_trait(Trait.load("goal_fixation"))
    assert not set(a.questions) & set(b.questions)
    assert not {p["pos"] for p in a.pairs} & {p["pos"] for p in b.pairs}


def test_hf_policy_episode_records_traces(tiny):
    import torch

    model, tok = tiny
    pol = HFPolicy(model, tok, meters={"m": (1, torch.randn(64))}, max_new_tokens=6)
    res = run_episode(Pushback(True, True, seed=0), pol)
    assert res["outcome"] in {"self_corrects", "defends_wrong", "other_wrong", "failed"}
    assert len(res["traces"]) == 1 and len(res["traces"][0]["m"]) >= 1
