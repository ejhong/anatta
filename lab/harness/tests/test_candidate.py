"""candidate.py: validation, default controls per kind, freezing. Claude (Opus 5.5) wrote this file, 2 Nov 2026."""

import copy

import pytest

from harness import candidate as C
from harness import measures as M
from harness import runner as R

H = "a" * 64
REV = "f6fae9795746f63c9be8344932f01275f3c63734"
MODEL = {"id": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "revision": REV, "source": "hf_cache"}


def steer(**kw):
    d = {"name": "x", "model": dict(MODEL),
         "intervention": {"kind": "steer", "direction": {"name": "nofix", "tensor_sha256": H}, "layers": [17],
                          "strength": 1.4}}
    d.update(kw)
    return d


def test_steer_defaults_three_controls_and_topic():
    c = C.validate(steer())
    assert [x["name"] for x in c.controls] == ["random_matched", "other_trait", "split_null"]
    c2 = C.validate(steer(topic_direction={"name": "topic", "tensor_sha256": "b" * 64}))
    assert [x["name"] for x in c2.controls][-1] == "topic"
    assert c2.controls[0]["strength"] == 1.4 and c2.controls[0]["layers"] == [17]


def test_ablate_defaults():
    d = steer()
    d["intervention"] = {"kind": "ablate", "direction": {"name": "ego", "tensor_sha256": H}, "layers": [8, 9]}
    c = C.validate(d)
    assert {x["name"] for x in c.controls} == {"random_matched", "other_trait", "split_null"}
    assert all("strength" not in x for x in c.controls)


def test_prompt_defaults():
    d = steer(intervention={"kind": "prompt", "system_prompt": "Be at ease."})
    c = C.validate(d)
    assert [x["name"] for x in c.controls] == ["neutral", "persona", "honesty"]
    assert c.controls[0]["chars"] == len("Be at ease.")


def test_adapter_needs_twin_and_three_seeds():
    iv = {"kind": "adapter", "path": "/a", "sha256": H, "seeds": [0, 1, 2],
          "twin": {"path": "/t", "sha256": "c" * 64, "seeds": [0, 1, 2]}}
    c = C.validate(steer(intervention=iv))
    assert c.controls[0]["name"] == "twin" and c.controls[0]["seeds"] == [0, 1, 2]
    bad = copy.deepcopy(iv)
    bad["twin"]["seeds"] = [0, 1]
    with pytest.raises(C.CandidateError):
        C.validate(steer(intervention=bad))
    bad = copy.deepcopy(iv)
    del bad["twin"]
    with pytest.raises(C.CandidateError):
        C.validate(steer(intervention=bad))


def test_checkpoint_same_rule():
    iv = {"kind": "checkpoint", "path": "/o", "sha256": H, "seeds": [0, 1, 2],
          "twin": {"path": "/t", "sha256": "c" * 64, "seeds": [0, 1, 2]}}
    assert C.validate(steer(intervention=iv)).controls[0]["kind"] == "checkpoint"


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(name="Bad Name"),
    lambda d: d["model"].update(revision="main"),
    lambda d: d["intervention"]["direction"].update(tensor_sha256="short"),
    lambda d: d["intervention"].update(strength=0),
    lambda d: d["intervention"].update(layers=[]),
    lambda d: d["intervention"].update(position="middle"),
    lambda d: d.update(tier="medium"),
    lambda d: d.update(surprise=1),
    lambda d: d.update(controls=[{"name": "base"}]),
])
def test_rejects(mutate):
    d = steer()
    mutate(d)
    with pytest.raises(C.CandidateError):
        C.validate(d)


def test_openrouter_is_prompt_only_and_pins_provider():
    m = {"id": "mistralai/ministral-8b-2512", "source": "openrouter", "provider": "Mistral"}
    with pytest.raises(C.CandidateError):
        C.validate(steer(model=m))
    ok = C.validate(steer(model=m, intervention={"kind": "prompt", "prefix": "Note:"}))
    assert ok.kind == "prompt"
    with pytest.raises(C.CandidateError):
        C.validate(steer(model={"id": "x/y", "source": "openrouter"}, intervention={"kind": "prompt", "prefix": "a"}))


def test_sha_is_stable_and_sensitive():
    a, b = C.validate(steer()), C.validate(steer())
    assert a.sha256 == b.sha256 and len(a.sha256) == 64
    d = steer()
    d["intervention"]["strength"] = 1.5
    assert C.validate(d).sha256 != a.sha256
    C.check_frozen(a, b.sha256)
    with pytest.raises(C.CandidateError):
        C.check_frozen(a, "0" * 64)


def test_fixture_candidate_loads_and_resolves_measures():
    from harness.fixtures import build_noself3 as F

    c = C.load(F.CAND)
    assert c.tier == "quick" and c.file_sha256
    reg = M.load()
    ms = C.resolve_measures(c, reg)
    assert "dev_deference" in ms and "mask_p_lie" not in ms  # public anchors are full tier only
    with pytest.raises(C.CandidateError):
        C.resolve_measures(C.Candidate(**{**c.__dict__, "measures": ("mask_p_lie",)}), reg)


def test_runner_plan_maps_arms_to_bench_form():
    c = C.validate(steer(topic_direction={"name": "topic", "tensor_sha256": "b" * 64}))
    p = R.plan(c, M.load(), "2026-11-03")
    by = {a["name"]: a["bench"] for a in p.arms}
    assert by["base"] == {"mode": "none"}
    assert by["candidate"] == {"mode": "add", "layers": [17], "coefficient": 1.4, "direction": "nofix",
                               "tensor_sha256": H}
    assert by["random_matched"]["direction"] == "random_matched" and by["random_matched"]["target"] == "nofix"
    assert by["split_null"]["seeds"] == [0]
    assert by["other_trait"]["match"] == "partial_moment"
    assert by["topic"]["tensor_sha256"] == "b" * 64
    assert p.out_dir.endswith("2026-11-03/x")
    with pytest.raises(R.RunnerNotBuilt):
        R.ModalRunner().run(p)
