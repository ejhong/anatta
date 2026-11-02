"""CPU tests for the cut split into its two parts (bench, the evening of 1 Nov 2026): mean_ablate (h.d set to a
per-layer constant), shift_only (h moved by minus that constant along d) and shift_reverse (plus it); the identity
"mean_ablate, then shift_only, is the cut"; split_means' constants and checks on the tiny model; the arms in a sweep
end to end, with mu_l and sd_l in the run's metadata. No Modal, no GPU, no downloads. Claude (Opus 5.5) wrote this
file, 1 Nov 2026. Private (lab/explore)."""

from __future__ import annotations

import json
import math

import pytest
import torch
from conftest import BENCH, HIDDEN, LAYERS

import client as C
import directions as D
import records as R
import steering as S
from anatta.dial import intervene
from anatta.probe import capture

MSG = [{"role": "system", "content": "the quick brown fox"}, {"role": "user", "content": "lazy dog final answer"}]
NEUTRAL = ["the quick brown fox", "over the lazy dog 0123", "final answer 456789", "the lazy brown dog jumps",
           "quick brown dog jumps over", "final answer 42 the fox", "brown fox lazy dog 7", "the answer 0123"]


def rand_dir(seed=0, layers=range(LAYERS), dim=HIDDEN):
    g = torch.Generator().manual_seed(seed)
    return {x: torch.randn(dim, generator=g) for x in layers}


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / v.norm()


def read_block(model, tok, layers, ivs=()):
    ids = tok.apply_chat_template(MSG, add_generation_prompt=True, return_tensors="pt")
    ids = ids if isinstance(ids, torch.Tensor) else ids["input_ids"]
    with intervene(model, list(ivs)), capture(model, layers) as store:
        out = model(input_ids=ids)
    return {x: store[x][0].clone() for x in layers}, out.logits[0].clone()


def req(rid, seed, iv, max_tokens=8, temperature=0.9):
    return {"rid": rid, "messages": MSG, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            "intervention": iv}


def payload_dir(vecs, name="d"):
    return {"name": name, "vectors": {x: v.numpy() for x, v in vecs.items()}}


# ----------------------------------------------------------------------------------------------- the formulas


def test_mean_ablate_then_shift_only_is_the_cut(tiny):
    vecs = rand_dir(21)
    means = {0: 1.7, 1: -2.3, 2: 0.4, 3: -0.05}
    cut = S.build_interventions("ablate", vecs, range(LAYERS))
    ma = S.build_interventions("mean_ablate", vecs, range(LAYERS), means=means)
    so = S.build_interventions("shift_only", vecs, range(LAYERS), means=means)
    h = torch.randn(3, 7, HIDDEN, generator=torch.Generator().manual_seed(0)) * 5
    for x in range(LAYERS):  # one layer, one activation: the identity of the two operators
        assert torch.allclose(so[x].apply(ma[x].apply(h)), cut[x].apply(h), atol=1e-5)
        assert not torch.allclose(ma[x].apply(so[x].apply(h)), cut[x].apply(h), atol=1e-3)  # the order matters
    model, tok = tiny  # every layer of a model: mean_ablate then shift_only at each layer is the cut at each layer
    both = [iv for pair in zip(ma, so) for iv in pair]
    split_h, split_logits = read_block(model, tok, list(range(LAYERS)), both)
    cut_h, cut_logits = read_block(model, tok, list(range(LAYERS)), cut)
    base_h, _ = read_block(model, tok, list(range(LAYERS)))
    for x in range(LAYERS):
        assert torch.allclose(split_h[x], cut_h[x], atol=1e-5)
        assert (cut_h[x] @ unit(vecs[x])).abs().max() < 1e-4
    assert torch.allclose(split_logits, cut_logits, atol=1e-5)
    assert not torch.allclose(cut_h[3], base_h[3], atol=1e-3)


def test_each_split_arm_does_what_its_formula_says(tiny):
    model, tok = tiny
    vecs = rand_dir(22)
    u, m = unit(vecs[1]), 1.25
    base = read_block(model, tok, [1])[0][1]
    ma = read_block(model, tok, [1], S.build_interventions("mean_ablate", vecs, [1], means={1: m}))[0][1]
    assert torch.allclose(ma @ u, torch.full_like(ma @ u, m), atol=1e-4)  # h.d is m at every token
    assert torch.allclose(ma - (ma @ u)[:, None] * u, base - (base @ u)[:, None] * u, atol=1e-4)  # the rest is kept
    so = read_block(model, tok, [1], S.build_interventions("shift_only", vecs, [1], means={1: m}))[0][1]
    sr = read_block(model, tok, [1], S.build_interventions("shift_reverse", vecs, [1], means={1: m}))[0][1]
    assert torch.allclose(so - base, (-m * u).expand_as(base), atol=1e-5)
    assert torch.allclose(sr - base, (m * u).expand_as(base), atol=1e-5)
    built = S.build_interventions("mean_ablate", vecs, [1, 2], means={1: m, 2: -m})
    assert [(x.mode, x.lo, x.hi) for x in built] == [("cap", m, m), ("cap", -m, -m)]
    assert [x.alpha for x in S.build_interventions("shift_reverse", vecs, [1, 2], means={"1": m, "2": -m})] == [m, -m]


def test_the_split_arms_refuse_bad_requests():
    vecs = rand_dir(23)
    for mode in S.SPLIT_MODES:
        with pytest.raises(ValueError, match="mean at every layer"):
            S.build_interventions(mode, vecs, [1, 2], means={1: 0.5})
        with pytest.raises(ValueError, match="coefficient"):
            S.build_interventions(mode, vecs, [1], 2.0, means={1: 0.5})
        with pytest.raises(ValueError, match="fractions"):
            S.build_interventions(mode, vecs, [1], fractions={1: 0.5}, means={1: 0.5})
        with pytest.raises(ValueError, match="finite"):
            S.build_interventions(mode, vecs, [1], means={1: float("nan")})
    for mode in ("ablate", "add"):
        with pytest.raises(ValueError, match="means belong"):
            S.build_interventions(mode, vecs, [1], 1.0 if mode == "add" else None, means={1: 0.5})
    iv = {"mode": "shift_only", "tensor_sha256": "t", "layers": [1], "means": {1: 0.5}}
    assert S.iv_key(iv) != S.iv_key({**iv, "means": {1: 0.6}}) != S.iv_key({**iv, "mode": "shift_reverse"})
    assert S.iv_key(iv) == S.iv_key({**iv, "means": {"1": 0.5}})
    a = S.build_interventions("mean_ablate", vecs, [1], means={1: 0.5})
    assert S.cut_diff(a, S.build_interventions("mean_ablate", vecs, [1], means={1: 0.6})) == float("inf")
    assert S.cut_diff(a, S.build_interventions("mean_ablate", vecs, [1], means={1: 0.5})) == 0.0


# ----------------------------------------------------------------------------------------------- split_means


def test_split_means_finds_each_arms_constants_and_checks_them(svc, tiny):
    vecs = rand_dir(24)
    pd = payload_dir(vecs)
    res = svc.split_means({"direction": pd, "key": "k", "layers": [0, 1, 2, 3], "neutral_prompts": NEUTRAL,
                           "neutral_sha256": "n"})
    assert S.count_hooks(tiny[0]) == 0 and res["meta"]["method"] == S.SPLIT_METHOD
    st = svc.measure({"directions": {"k": pd}, "layers": [0, 1, 2, 3], "neutral_prompts": NEUTRAL})["stats"]["k"]
    for x in range(LAYERS):  # the uncut numbers are measure()'s
        b = res["base"][x]
        assert b["mean"] == pytest.approx(st[x]["mean_proj"], rel=1e-9, abs=1e-12)
        assert b["var"] == pytest.approx(st[x]["var"], rel=1e-9)
        assert b["second_moment"] == pytest.approx(st[x]["second_moment"], rel=1e-9)
        assert b["sd"] == pytest.approx(math.sqrt(b["var"]), rel=1e-9)
    mu = {x: res["base"][x]["mean"] for x in range(LAYERS)}
    for mode in S.SPLIT_MODES:  # the lowest layer's constant is the uncut mean for every arm
        assert res["consts"][mode][0] == pytest.approx(mu[0], rel=1e-12)
    real = res["realized"]
    for x in range(LAYERS):
        tol = 1e-4 * (1 + abs(mu[x]))
        assert abs(real["ablate"][x]["mean"]) < tol and real["ablate"][x]["sd"] < tol
        assert real["mean_ablate"][x]["mean"] == pytest.approx(res["consts"]["mean_ablate"][x], abs=tol)
        assert real["mean_ablate"][x]["sd"] < tol
        assert abs(real["shift_only"][x]["mean"]) < tol and real["shift_only"][x]["sd"] > 10 * tol
        assert real["shift_reverse"][x]["mean"] == pytest.approx(2 * mu[x], abs=tol)
    # with every layer's uncut mean (the per-layer formula as written), only the lowest layer is sure to land at 0:
    # above it the shifts already made stay in the stream (the tiny model's random directions are nearly orthogonal,
    # so they pile up little here; v_ego's adjacent layers have cos 0.87-0.97)
    ind = real["shift_only_independent"]
    assert abs(ind[0]["mean"]) < 1e-4 * (1 + abs(mu[0]))
    assert set(ind) == {0, 1, 2, 3}
    one = svc.split_means({"direction": pd, "key": "k", "layers": [2], "neutral_prompts": NEUTRAL})
    for mode in S.SPLIT_MODES:  # one layer: every arm's constant is the uncut mean, the formulas as written
        assert one["consts"][mode][2] == pytest.approx(mu[2], rel=1e-12)
    with pytest.raises(ValueError):
        svc.split_means({"direction": pd, "key": "k", "layers": [0], "neutral_prompts": NEUTRAL, "modes": ["cut"]})


def test_the_service_runs_the_split_arms_and_checks_them(svc, tiny):
    model, _ = tiny
    vecs = rand_dir(25)
    pd = payload_dir(vecs)
    res = svc.split_means({"direction": pd, "key": "k", "layers": [1, 2], "neutral_prompts": NEUTRAL})
    ivs = [{"mode": "ablate", "tensor_sha256": "k", "sha256": "f", "layers": [1, 2], "direction": "d"}]
    ivs += [{"mode": m, "tensor_sha256": "k", "sha256": "f", "layers": [1, 2], "direction": "d",
             "means": res["consts"][m]} for m in S.SPLIT_MODES]
    out = svc.generate({"requests": [req(i, 1, iv) for i, iv in enumerate(ivs)], "directions": {"k": pd}})
    assert S.count_hooks(model) == 0
    groups = {g["intervention"]["mode"]: g for g in out["meta"]["groups"]}
    assert set(groups) == {"ablate", *S.SPLIT_MODES}
    assert groups["mean_ablate"]["attach"]["dev_after"] < 1e-4 < groups["mean_ablate"]["attach"]["dev_before"]
    assert groups["mean_ablate"]["attach"]["mean"] == pytest.approx(res["consts"]["mean_ablate"][1])
    for m, sign in (("shift_only", -1.0), ("shift_reverse", 1.0)):
        att = groups[m]["attach"]
        assert att["rel_err"] < 1e-5 and att["proj_err"] < 1e-5
        assert att["coefficient"] == pytest.approx(sign * res["consts"][m][1])
    assert groups["shift_only"]["intervention"]["means"] == res["consts"]["shift_only"]
    assert all(g["clean"]["hooks"] == 0 for g in groups.values())


# ----------------------------------------------------------------------------------------------- sweeps end to end


def save_direction(path, vecs, name="d"):
    torch.save({"name": name, "vectors": vecs, "scores": {x: float(x) for x in vecs},
                "pos_mean": {x: 0.0 for x in vecs}, "neg_mean": {x: 0.0 for x in vecs}, "meta": {}}, path)
    return str(path)


def ref_of(path, name):
    d = D.resolve({"path": str(path)}, "ministral")
    return {"path": str(path), "tensor_sha256": d.tensor_sha256, "name": name}


def split_spec(tmp_path, **over):
    tgt = ref_of(save_direction(tmp_path / "target.pt", rand_dir(30)), "target")
    spec = {"sweep": "bench-split-test", "model": "ministral", "library": "fixture", "items": ["ts-90", "ts-91"],
            "interventions": ["baseline", {"mode": "ablate", "direction": tgt},
                              {"mode": "mean_ablate", "direction": tgt},
                              {"mode": "shift_only", "direction": tgt},
                              {"mode": "shift_reverse", "direction": tgt},
                              {"mode": "shift_only", "direction": tgt, "means": "independent"},
                              {"mode": "mean_ablate", "direction": tgt, "layers": "1-2"},
                              {"mode": "ablate", "direction": "random_matched", "target": tgt, "seeds": [0]}],
            "samples": 1, "temperature": 0.8, "max_tokens": 5, "seed": 3, "spend_cap_usd": 2.0, "batch_size": 8}
    spec.update(over)
    return spec, tgt


def test_a_sweep_with_the_split_arms_records_mu_and_sd_at_every_layer(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "neutral_set", lambda: (NEUTRAL, "nsha"))
    spec, tgt = split_spec(tmp_path)
    root = tmp_path / "results"
    man = C.run_sweep(spec, bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    names = [v["name"] for v in man["variants"]]
    assert names == ["baseline", "ablate:target", "mean_ablate:target", "shift_only:target", "shift_reverse:target",
                     "shift_only:target~independent", "mean_ablate:target[1-2]", "ablate:random_matched(target)#s0"]
    assert man["status"] == "complete" and man["records"] == man["jobs"] == 2 * len(names)
    tables = man["bench"]["split"]
    assert [t["layers"] for t in tables] == [[0, 1, 2, 3], [1, 2]]
    full = tables[0]
    assert set(full["mu"]) == set(full["sd"]) == {"0", "1", "2", "3"}
    assert full["tensor_sha256"] == tgt["tensor_sha256"] and full["neutral"]["sha256"] == "nsha"
    assert set(full["consts"]) == set(S.SPLIT_MODES)
    assert set(full["realized"]) == {*S.SPLIT_MODES, "ablate", "shift_only_independent"}
    sd = root / man["date"] / "bench-split-test"
    on_disk = json.loads((sd / "directions" / full["file"]).read_text())
    assert on_disk["mu"] == full["mu"] and on_disk["consts"] == full["consts"]
    recs = R.schema.read_jsonl(sd / "records.jsonl")
    assert all(R.schema.validate_record(r) == [] for r in recs)
    by = {r["intervention"]["name"]: r["intervention"]["params"] for r in recs}
    for mode in S.SPLIT_MODES:
        p = by[f"{mode}:target"]
        assert p["mode"] == mode and p["layers"] == [0, 1, 2, 3] and p["tensor_sha256"] == tgt["tensor_sha256"]
        assert p["means"] == {x: round(v, 6) for x, v in full["consts"][mode].items()}
        assert p["split"]["means"] == "sequential" and p["split"]["file"] == full["file"]
    ind = by["shift_only:target~independent"]
    assert ind["means"] == {x: round(v, 6) for x, v in full["mu"].items()} and ind["split"]["means"] == "independent"
    band = by["mean_ablate:target[1-2]"]
    assert band["layers"] == [1, 2] and set(band["means"]) == {"1", "2"}
    assert band["means"]["1"] == pytest.approx(full["mu"]["1"], abs=1e-6)  # its lowest layer: the uncut mean
    assert "means" not in by["ablate:target"] and "means" not in by["baseline"]
    groups = [g for c in man["bench"]["calls"] for g in c["groups"]]
    assert any(g["intervention"]["mode"] == "mean_ablate" and g["attach"]["dev_after"] < 1e-4 for g in groups)


def test_the_split_arms_are_checked_in_the_spec(tmp_path):
    spec, tgt = split_spec(tmp_path)
    bad = [{"mode": "mean_ablate", "direction": "random_matched", "target": tgt, "seeds": [0]},
           {"mode": "shift_only", "direction": "other_trait", "target": tgt, "traits": [tgt]},
           {"mode": "shift_only", "direction": tgt, "strengths": [1.0]},
           {"mode": "shift_reverse", "direction": tgt, "coefficient": 1.0},
           {"mode": "mean_ablate", "direction": tgt, "match": "partial"},
           {"mode": "mean_ablate", "direction": tgt, "means": "both"},
           {"mode": "ablate", "direction": tgt, "means": "sequential"},
           {"mode": "shift", "direction": tgt}]
    for b in bad:
        with pytest.raises(C.SpecError):
            C.expand(C.load_spec({**spec, "interventions": ["baseline", b]}))
    conds = C.expand(C.load_spec(spec))
    assert [c["split"] for c in conds[2:6]] == [{"means": "sequential"}] * 3 + [{"means": "independent"}]


def test_the_container_and_the_client_agree_on_the_split_arms():
    assert C.SPLIT_MODES == S.SPLIT_MODES
    src = (BENCH / "modal_bench.py").read_text()
    assert "def split_means(self, payload" in src
    assert callable(C.RemoteBench.split_means) and callable(C.LocalBench.split_means) and callable(S.Service.split_means)
