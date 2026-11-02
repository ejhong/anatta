"""bench's CPU tests: the add and ablate hooks and their removal, the cut's identity with player.ego_cut, seeded
sampling, the matched-strength random controls, the record schema end to end, the held-out guard and the spend
meter. No Modal, no GPU, no downloads. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private (lab/explore)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from conftest import BENCH, HIDDEN, LAYERS

import client as C
import directions as D
import meter as M
import records as R
import steering as S
from anatta.dial import intervene
from anatta.probe import Direction, capture

Direction_load = Direction.load

MSG = [{"role": "system", "content": "the quick brown fox"}, {"role": "user", "content": "lazy dog final answer"}]


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / v.norm()


def read_block(model, tok, layers, ivs=()):
    ids = tok.apply_chat_template(MSG, add_generation_prompt=True, return_tensors="pt")
    ids = ids if isinstance(ids, torch.Tensor) else ids["input_ids"]
    with intervene(model, list(ivs)), capture(model, layers) as store:
        model(input_ids=ids)
    return {x: store[x][0].clone() for x in layers}


def rand_dir(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {x: torch.randn(HIDDEN, generator=g) for x in range(LAYERS)}


def req(rid, seed, iv, temperature=0.9, max_tokens=10, messages=MSG):
    return {"rid": rid, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            "intervention": iv}


def payload_dir(vecs, name="d"):
    return {"name": name, "vectors": {x: v.numpy() for x, v in vecs.items()}}


# ----------------------------------------------------------------------------------------------- the hooks


def test_add_puts_coefficient_times_unit_direction(tiny):
    model, tok = tiny
    vecs = rand_dir(1)
    base = read_block(model, tok, [1])[1]
    steered = read_block(model, tok, [1], S.build_interventions("add", vecs, [1], 3.0))[1]
    assert torch.allclose(steered - base, 3.0 * unit(vecs[1]), atol=1e-4)


def test_ablate_removes_the_component_at_every_layer_and_only_in_the_band(tiny):
    model, tok = tiny
    vecs = rand_dir(2)
    allh = read_block(model, tok, list(range(LAYERS)), S.build_interventions("ablate", vecs, range(LAYERS)))
    for x in range(LAYERS):
        assert (allh[x] @ unit(vecs[x])).abs().max() < 1e-4
    base = read_block(model, tok, [0, 3])
    band = read_block(model, tok, list(range(LAYERS)), S.build_interventions("ablate", vecs, [1, 2]))
    assert torch.equal(band[0], base[0])  # before the band: untouched
    assert all((band[x] @ unit(vecs[x])).abs().max() < 1e-4 for x in (1, 2))
    assert (band[3] @ unit(vecs[3])).abs().max() > 1e-3  # after the band the component comes back


def test_build_interventions_refuses_bad_requests():
    vecs = rand_dir()
    assert S.build_interventions("none", vecs, []) == []
    with pytest.raises(ValueError):
        S.build_interventions("add", vecs, [1])  # no coefficient
    with pytest.raises(ValueError):
        S.build_interventions("ablate", vecs, [7])  # no vector there
    with pytest.raises(ValueError):
        S.build_interventions("ablate", vecs, [])
    with pytest.raises(ValueError):
        S.build_interventions("cap", vecs, [1])


def test_hooks_are_removed_after_every_batch_and_the_checks_pass(svc, tiny):
    model, tok = tiny
    before = read_block(model, tok, list(range(LAYERS)))
    vecs = rand_dir(3)
    reqs = [req(0, 1, {"mode": "none"}),
            req(1, 1, {"mode": "ablate", "sha256": "x", "layers": [0, 1, 2, 3], "direction": "d"}),
            req(2, 1, {"mode": "add", "sha256": "x", "layers": [1, 2], "coefficient": 4.0, "direction": "d"})]
    out = svc.generate({"requests": reqs, "directions": {"x": payload_dir(vecs)}})
    assert S.count_hooks(model) == svc.base_hooks == 0
    after = read_block(model, tok, list(range(LAYERS)))
    assert all(torch.equal(before[x], after[x]) for x in range(LAYERS))
    groups = {g["intervention"]["mode"]: g for g in out["meta"]["groups"]}
    assert groups["none"]["hooked_layers"] == 0 and groups["ablate"]["hooked_layers"] == 4
    assert groups["ablate"]["attach"]["proj_after"] < 1e-4 < groups["ablate"]["attach"]["proj_before"]
    assert groups["add"]["attach"]["rel_err"] < 1e-5
    assert all(g["clean"]["hooks"] == 0 and g["clean"]["probe_max_diff"] == 0.0 for g in groups.values())
    assert [r["rid"] for r in out["results"]] == [0, 1, 2]


def test_hooks_come_off_when_a_batch_fails(svc, tiny, monkeypatch):
    model, _ = tiny

    def boom(reqs):
        assert S.count_hooks(model) == 4  # the hooks are on while the group generates
        raise RuntimeError("mid-batch failure")

    monkeypatch.setattr(svc, "generate_rows", boom)
    iv = {"mode": "ablate", "sha256": "x", "layers": [0, 1, 2, 3], "direction": "d"}
    with pytest.raises(RuntimeError, match="mid-batch"):
        svc.generate({"requests": [req(0, 1, iv)], "directions": {"x": payload_dir(rand_dir())}})
    assert S.count_hooks(model) == 0


def test_a_leaked_hook_is_caught(svc, tiny, monkeypatch):
    from anatta.models import get_layers

    model, _ = tiny
    real = svc.generate_rows
    leaked = []

    def leak(reqs):
        leaked.append(get_layers(model)[2].register_forward_hook(lambda m, i, o: (o[0] * 1.5,) + tuple(o[1:])
                                                                   if isinstance(o, tuple) else o * 1.5))
        return real(reqs)

    monkeypatch.setattr(svc, "generate_rows", leak)
    try:
        with pytest.raises(S.HookLeak):
            svc.generate({"requests": [req(0, 1, {"mode": "none"})], "directions": {}})
    finally:
        for h in leaked:
            h.remove()
    assert S.count_hooks(model) == 0


def test_the_v_ego_ablation_is_player_ego_cut():
    """The real ego.pt, through the client's path (registry, sha256, numpy payload) and the container's (tensors,
    build_interventions), gives player.ego_cut's cut: layers 8-32, the same modes, directions equal to float32
    rounding (the file's vectors are strided views, so their norms round differently from a contiguous copy's)."""
    from player import ego_cut

    d = D.resolve("ego", "ministral")
    assert d.sha256 == "a105f23b79fecd23153e36d5cc2911e07647b0dbcd395ce7338efdc79e9cf2f4"
    sent = d.payload(n_layers=34)["vectors"]
    vecs = {x: torch.as_tensor(v, dtype=torch.float32).flatten() for x, v in sent.items()}
    mine = S.build_interventions("ablate", vecs, d.layers_for("all", 34))
    ref = ego_cut(Path(d.path).parent)
    assert [x.layer for x in mine] == list(range(8, 33))
    assert S.same_cut(mine, ref) and S.cut_diff(mine, ref) < 1e-6
    exact = S.build_interventions("ablate", {x: v for x, v in Direction_load(d.path).vectors.items()}, range(8, 33))
    assert S.cut_diff(exact, ref) == 0.0  # from the file's own tensors, bit-identical
    band = S.build_interventions("ablate", vecs, d.layers_for("10-12", 34))
    assert S.same_cut(band, [x for x in ref if 10 <= x.layer <= 12])
    assert not S.same_cut(band, ref)


# ----------------------------------------------------------------------------------------------- sampling


def test_a_request_samples_the_same_alone_or_batched_and_seeds_pair_conditions(svc):
    vecs = rand_dir(4)
    abl = {"mode": "ablate", "sha256": "x", "layers": [0, 1, 2, 3], "direction": "d"}
    other = [{"role": "user", "content": "0123456789 the"}]
    batch = [req(0, 7, {"mode": "none"}), req(1, 8, {"mode": "none"}, messages=other), req(2, 7, abl),
             req(3, 9, {"mode": "none"}, temperature=0.0)]
    out = svc.generate({"requests": batch, "directions": {"x": payload_dir(vecs)}})["results"]
    alone = svc.generate({"requests": [batch[0]], "directions": {}})["results"][0]
    assert alone["text"] == out[0]["text"]
    again = svc.generate({"requests": [batch[0]], "directions": {}})["results"][0]
    assert again["text"] == alone["text"]
    assert out[2]["text"] != out[0]["text"]  # same seed, the cut changes the reply
    reseeded = svc.generate({"requests": [req(0, 1234, {"mode": "none"})], "directions": {}})["results"][0]
    assert reseeded["text"] != out[0]["text"]
    for r in out:
        assert r["finish_reason"] in ("stop", "length") and 0 < r["completion_tokens"] <= 10


def test_temperature_zero_is_greedy(svc, tiny):
    model, tok = tiny
    from anatta.models import chat_ids

    r = svc.generate({"requests": [req(0, 5, {"mode": "none"}, temperature=0.0, max_tokens=8)],
                      "directions": {}})["results"][0]
    ids = chat_ids(tok, MSG)
    g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    assert tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).startswith(r["text"])


def test_row_sampler_matches_the_distribution():
    """Many draws of one row follow softmax(logits / T), with top-k honoured."""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -5.0]])
    counts = torch.zeros(5)
    for s in range(3000):
        rs = S.RowSampler([0.7], [1.0], [0], [s], "cpu")
        counts[int(rs(None, logits).argmax())] += 1
    p = torch.softmax(logits[0] / 0.7, 0)
    assert torch.allclose(counts / counts.sum(), p, atol=0.03)
    topk = S.RowSampler([1.0], [1.0], [2], [3], "cpu")
    picks = {int(S.RowSampler([1.0], [1.0], [2], [s], "cpu")(None, logits).argmax()) for s in range(200)}
    assert picks == {0, 1} and topk is not None


# ----------------------------------------------------------------------------------------------- random_matched


def toy_acts(seed=1, d=96, n=500):
    g = torch.Generator().manual_seed(seed)
    scales = torch.logspace(1.5, -1.5, d)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    acts = {x: (torch.randn(n, d, generator=g) * scales) @ q.T + 2.0 * torch.randn(d, generator=g) for x in (3, 4, 5)}
    target = {x: q[:, :6] @ torch.randn(6, generator=g) for x in (3, 4, 5)}
    return acts, target


def var_along(h, u):
    hc = h.double() - h.double().mean(0)
    return float((hc @ unit(u).double()).var())


def test_random_matched_comes_within_ten_percent_on_a_toy_set():
    acts, target = toy_acts()
    seen = []
    for seed in range(5):
        vecs, made = S.random_matched(acts, target, seed)
        assert made["method"] == S.METHOD and made["seed"] == seed
        for x in (3, 4, 5):
            ratio = var_along(acts[x], vecs[x]) / var_along(acts[x], target[x])
            assert abs(ratio - 1) <= 0.10, (seed, x, ratio)
            assert abs(float(vecs[x].norm()) - 1) < 1e-5
            assert abs(float(unit(vecs[x]) @ unit(target[x]))) < 1e-5  # orthogonal to the target
            lay = made["layers"][str(x)]
            assert abs(lay["ratio"] - ratio) < 1e-3 and lay["plain_random_ratio"] < 0.5
        seen.append(torch.cat([vecs[x] for x in (3, 4, 5)]))
    assert all(float(unit(a) @ unit(b)) < 0.9 for i, a in enumerate(seen) for b in seen[i + 1:])  # seeds differ


def test_random_matched_refuses_a_target_without_variance():
    acts, target = toy_acts()
    flat = {x: a.clone() for x, a in acts.items()}
    for x in flat:
        u = unit(target[x])
        flat[x] = flat[x] - (flat[x] @ u)[:, None] * u  # no variance along the target
    with pytest.raises(S.MatchError):
        S.random_matched(flat, target, 0)


def test_random_matched_through_the_service_on_the_tiny_model(svc, tiny):
    """The service's controls, re-measured independently (content tokens found by search, not by the template
    diff), have the target's variance within 10% on the neutral set."""
    model, tok = tiny
    prompts = ["the quick brown fox jumps", "over the lazy dog", "0123456789 final answer", "the lazy fox",
               "quick brown dog jumps over", "final answer 42 the fox", "brown fox lazy dog 7", "the answer 0123",
               "jumps over the quick dog", "lazy brown fox 89 final", "dog fox the quick answer", "456 over the lazy"]
    target = rand_dir(5)
    out = svc.random_matched({"target": {"sha256": "t", **payload_dir(target, "t")}, "seeds": [0, 1],
                              "layers": [1, 2, 3], "neutral_prompts": prompts, "neutral_sha256": "n"})
    from anatta.models import chat_ids

    rows = {x: [] for x in (1, 2, 3)}
    for p in prompts:
        ids = chat_ids(tok, [{"role": "user", "content": p}])[0].tolist()
        content = tok(p, add_special_tokens=False)["input_ids"]
        start = next(i for i in range(len(ids)) if ids[i:i + len(content)] == content)
        with capture(model, [1, 2, 3]) as store:
            model(input_ids=torch.tensor([ids]))
        for x in rows:
            rows[x].append(store[x][0, start:start + len(content)])
    for d in out["directions"]:
        assert d["made"]["target"]["sha256"] == "t" and d["made"]["neutral"]["sha256"] == "n"
        for x in (1, 2, 3):
            h = torch.cat(rows[x])
            ratio = var_along(h, torch.as_tensor(d["vectors"][x])) / var_along(h, target[x])
            assert abs(ratio - 1) <= 0.10, (x, ratio)


# ----------------------------------------------------------------------------------------------- directions


def test_layer_specs_and_registry_checks(tmp_path):
    d = D.Dir(name="d", sha256="s", path=None, vectors={x: np.ones(4, np.float32) for x in range(8, 13)},
              best_layer=10)
    assert d.layers_for("all", 34) == [8, 9, 10, 11, 12]
    assert d.layers_for("best", 34) == [10]
    assert d.layers_for("9-11", 34) == [9, 10, 11]
    assert d.layers_for([8, "11-12"], 34) == [8, 11, 12]
    assert d.layers_for({"from": 8, "to": 9}, 34) == [8, 9]
    for bad in ("7-9", [40], "x", True):
        with pytest.raises(D.DirectionError):
            d.layers_for(bad, 34)
    one = D.Dir(name="o", sha256="s", path=None, vectors={}, single=np.ones(4, np.float32))
    assert one.layers_for("all", 4) == [0, 1, 2, 3] and sorted(one.payload(4)["vectors"]) == [0, 1, 2, 3]
    reg = D.load_registry()
    bad = json.loads(json.dumps(reg))
    bad["directions"]["ministral"]["ego"]["sha256"] = "0" * 64
    with pytest.raises(D.DirectionError, match="not the expected"):
        D.resolve("ego", "ministral", bad)
    with pytest.raises(D.DirectionError):
        D.resolve("nope", "ministral", reg)


def test_every_registered_direction_file_matches_its_sha256():
    reg = D.load_registry()
    for model, entries in reg["directions"].items():
        folder = Path(reg["root"]) / reg["models"][model]["folder"]
        for name, e in entries.items():
            assert D.sha256_file(e.get("path") or folder / f"{name}.pt") == e["sha256"], (model, name)


def test_constants_agree_across_files():
    src = ast.parse((BENCH / "modal_bench.py").read_text())
    consts = {}
    for node in src.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        consts[t.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
                elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for n, v in zip(t.elts, node.value.elts):
                        consts[n.id] = ast.literal_eval(v)
    reg = D.load_registry()
    assert consts["SCALEDOWN_S"] == C.SCALEDOWN_S and consts["APP_NAME"] == C.APP_NAME == "anatta-explore-bench"
    assert consts["CPU"] == C.CORES and consts["MEMORY_MB"] / 1024 == C.MEM_GIB
    assert consts["EGO_SHA"] == reg["directions"]["ministral"]["ego"]["sha256"]
    assert {k: m["hf_id"] for k, m in S.MODELS.items()} == {k: m["hf_id"] for k, m in reg["models"].items()}
    assert not C.APP_NAME.startswith("matlab")


# ----------------------------------------------------------------------------------------------- a sweep end to end


def tiny_direction(tmp_path, seed=6) -> tuple[str, str]:
    vecs = rand_dir(seed)
    obj = {"name": "tinydir", "vectors": vecs, "scores": {x: float(x) for x in vecs},
           "pos_mean": {x: 0.0 for x in vecs}, "neg_mean": {x: 0.0 for x in vecs}, "meta": {}}
    p = tmp_path / "tinydir.pt"
    torch.save(obj, p)
    return str(p), D.sha256_file(p)


def tiny_spec(tmp_path, **over) -> dict:
    path, sha = tiny_direction(tmp_path)
    d = {"path": path, "sha256": sha, "name": "tinydir"}
    spec = {"sweep": "bench-test", "model": "ministral", "library": "fixture", "items": "all",
            "interventions": ["baseline", {"mode": "ablate", "direction": d},
                              {"mode": "ablate", "direction": d, "layers": "1-2"},
                              {"mode": "add", "direction": d, "layers": [1], "strengths": [-2, 2]},
                              {"mode": "ablate", "direction": "random_matched", "target": d, "seeds": [0, 1]},
                              {"mode": "add", "direction": "random_matched", "target": d, "seeds": [0],
                               "layers": [1], "strengths": [2]}],
            "samples": 2, "temperature": 0.8, "max_tokens": 6, "seed": 3, "spend_cap_usd": 2.0, "batch_size": 16}
    spec.update(over)
    return spec


def test_a_sweep_writes_schema_records_and_its_manifest(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "neutral_set", lambda: (["the quick brown fox", "over the lazy dog 0123",
                                                    "final answer 456789", "the lazy brown dog jumps"] * 4, "nsha"))
    root = tmp_path / "results"
    man = C.run_sweep(tiny_spec(tmp_path), bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    sd = root / man["date"] / "bench-test"
    recs = R.schema.read_jsonl(sd / "records.jsonl")
    names = [v["name"] for v in man["variants"]]
    assert names == ["baseline", "ablate:tinydir", "ablate:tinydir[1-2]", "add:tinydir[1]@-2", "add:tinydir[1]@+2",
                     "ablate:random_matched(tinydir)#s0", "ablate:random_matched(tinydir)#s1",
                     "add:random_matched(tinydir)#s0[1]@+2"]
    assert man["status"] == "complete" and len(recs) == man["records"] == man["jobs"] == 4 * 8 * 2
    assert R.schema.read_jsonl(sd / "records.partial.jsonl") and not R.schema.read_jsonl(sd / "errors.jsonl")
    for r in recs:
        assert R.schema.validate_record(r) == []
        assert (r["source"], r["provider"], r["split"], r["model"]) == ("bench", "modal", "dev", "tiny-llama")
        assert r["intervention"]["kind"] == "steer" and set(r["usage"]) >= {"prompt_tokens", "completion_tokens"}
        assert r["cost_usd"] >= 0 and r["params"]["max_tokens"] == 6
    keys = {R.schema.record_key(r) for r in recs}
    assert len(keys) == len(recs)  # every sample has its own key (seeds of a random control are separate conditions)
    seeds = {}
    for r in recs:  # paired: one seed per (item, sample), whatever the condition
        seeds.setdefault((r["item_id"], r["sample"]), set()).add(r["params"]["seed"])
    assert all(len(v) == 1 for v in seeds.values())
    by = {r["intervention"]["name"]: r["intervention"]["params"] for r in recs}
    assert by["baseline"]["mode"] == "none" and by["ablate:tinydir"]["layers"] == [0, 1, 2, 3]
    assert by["ablate:tinydir[1-2]"]["layers"] == [1, 2] and by["add:tinydir[1]@+2"]["coefficient"] == 2.0
    rm = by["ablate:random_matched(tinydir)#s1"]
    assert rm["made"]["method"] == S.METHOD and rm["made"]["seed"] == 1 and rm["made"]["neutral_sha256"] == "nsha"
    assert abs(rm["made"]["ratio_range"][0] - 1) <= 0.1 and abs(rm["made"]["ratio_range"][1] - 1) <= 0.1
    saved = sd / "directions" / rm["made"]["file"]
    assert D.sha256_file(saved) == rm["sha256"]
    assert sorted(Direction.load(saved).vectors) == [0, 1, 2, 3]
    assert by["add:random_matched(tinydir)#s0[1]@+2"]["sha256"] == by["ablate:random_matched(tinydir)#s0"]["sha256"]
    assert man["baseline"] == "baseline" and man["models"][0]["id"] == "tiny-llama" and man["missing_controls"] == []
    assert man["library_source"] == "fixture" and man["items"] == {"threat_to_self": ["ts-90", "ts-91", "ts-92",
                                                                                       "ts-93"]}
    variants = R.schema.read_jsonl(root / "variants.jsonl")
    assert {v["name"] for v in variants} == set(names) and {v["tool"] for v in variants} == {"bench"}
    assert man["bench"]["cold_start_s"] is not None and man["bench"]["calls"]


def test_a_sweep_logs_its_spend_to_the_shared_daily_file(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "neutral_set", lambda: (["the quick brown fox", "lazy dog 0123", "final answer"] * 5, "n"))
    root = tmp_path / "results"
    M.scout_spend.add_spend(root / "spend", tool="scout", label="other", usd=1.25)
    man = C.run_sweep(tiny_spec(tmp_path, samples=1), bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    day = M.scout_spend.read_day(root / "spend")
    tail = C.SCALEDOWN_S * C.RATE["total"] / 3600
    assert day["by_tool"]["bench"]["usd"] == pytest.approx(man["spend"]["this_run_usd"], abs=1e-6)
    assert day["total_usd"] == pytest.approx(1.25 + man["spend"]["this_run_usd"], abs=1e-6)
    assert tail <= man["spend"]["this_run_usd"] < tail + 0.01  # the warm tail plus a few seconds of CPU
    sweep_log = R.schema.read_jsonl(root / man["date"] / "bench-test" / "spend.jsonl")
    assert sum(x["cost_usd"] for x in sweep_log) == pytest.approx(man["spend"]["this_run_usd"], abs=1e-6)
    state = M.read_state(root / "spend")["ministral"]
    assert state["paid_until"] > 0 and state["label"] == "bench-test"


def test_a_sweep_refuses_to_start_past_the_daily_cap(svc, tmp_path):
    root = tmp_path / "results"
    M.scout_spend.add_spend(root / "spend", tool="scout", label="big", usd=49.9)
    with pytest.raises(M.CapReached, match="daily cap"):
        C.run_sweep(tiny_spec(tmp_path), bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    assert M.scout_spend.read_day(root / "spend")["total_usd"] == pytest.approx(49.9)
    assert not list(root.glob("*/bench-test/records.jsonl"))


def test_a_sweep_stopped_by_its_cap_keeps_what_it_did(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "neutral_set", lambda: (["the quick brown fox", "lazy dog 0123", "final answer"] * 5, "n"))
    real, n = M.Meter.before_call, {"calls": 0}

    def capped(self, est):
        n["calls"] += 1
        if n["calls"] == 3:  # random_matched, the first generate call, then the cap
            raise M.CapReached("stopped before the next call: test cap")
        return real(self, est)

    monkeypatch.setattr(M.Meter, "before_call", capped)
    root = tmp_path / "results"
    man = C.run_sweep(tiny_spec(tmp_path), bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    assert man["status"] == "stopped_at_cap" and "test cap" in man["stopped"]
    assert man["records"] == 16 and len(R.schema.read_jsonl(root / man["date"] / "bench-test" / "records.jsonl")) == 16


def test_a_sweep_that_fails_midway_still_writes_its_manifest(svc, tmp_path):
    g = torch.Generator().manual_seed(9)
    vecs = {x: torch.randn(HIDDEN // 2, generator=g) for x in range(LAYERS)}  # the wrong width for the model
    p = tmp_path / "narrow.pt"
    torch.save({"name": "narrow", "vectors": vecs, "scores": {x: 0.0 for x in vecs},
                "pos_mean": {x: 0.0 for x in vecs}, "neg_mean": {x: 0.0 for x in vecs}, "meta": {}}, p)
    spec = tiny_spec(tmp_path, interventions=["baseline", {"mode": "ablate", "direction": {"path": str(p)}}],
                     allow_missing_controls="test: a harness check")
    root = tmp_path / "results"
    with pytest.raises(SystemExit, match="dimensional"):
        C.run_sweep(spec, bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    (run,) = root.glob("*/bench-test/run.json")
    man = json.loads(run.read_text())
    assert man["status"] == "error" and "dimensional" in man["stopped"] and man["records"] == 0
    assert man["spend"]["this_run_usd"] > 0  # the session was charged and closed


def test_controls_alone_are_built_saved_and_charged(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "neutral_set", lambda: (["the quick brown fox", "lazy dog 0123", "final answer"] * 5, "n"))
    path, sha = tiny_direction(tmp_path)
    root = tmp_path / "results"
    out = C.build_controls({"path": path, "sha256": sha, "name": "tinydir"}, "ministral", [0, 1],
                           bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    assert [c["seed"] for c in out["controls"]] == [0, 1]
    for c in out["controls"]:
        assert abs(c["ratio_range"][0] - 1) <= 0.1 and abs(c["ratio_range"][1] - 1) <= 0.1
        assert D.sha256_file(c["file"]) == c["sha256"] and sorted(Direction.load(c["file"]).vectors) == [0, 1, 2, 3]
    assert out["controls"][0]["sha256"] != out["controls"][1]["sha256"]
    day = M.scout_spend.read_day(root / "spend")
    assert day["by_tool"]["bench"]["usd"] == pytest.approx(out["charged_usd"], abs=1e-6)
    assert (root / R.schema.utc_date() / "bench-controls" / f"controls-tinydir-{sha[:8]}.json").is_file()


# ----------------------------------------------------------------------------------------------- the spec


def test_specs_need_a_baseline_and_a_matched_control(tmp_path):
    spec = tiny_spec(tmp_path)
    no_base = {**spec, "interventions": spec["interventions"][1:]}
    with pytest.raises(C.SpecError, match="baseline"):
        C.expand(C.load_spec(no_base))
    no_ctrl = {**spec, "interventions": spec["interventions"][:4]}
    conds = C.expand(C.load_spec(no_ctrl))
    assert C.missing_controls(conds)
    with pytest.raises(C.SpecError, match="matched controls"):
        C.run_sweep(no_ctrl, bench=object(), root=tmp_path / "r", log=lambda *_: None)
    dup = {**spec, "interventions": ["baseline", "none"]}
    with pytest.raises(C.SpecError, match="twice"):
        C.expand(C.load_spec(dup))
    for bad in ({"sweep": "Bad Name"}, {"model": "llama"}, {"spend_cap_usd": 60}, {"max_tokens": 0},
                {"bogus": 1}, {"tol": 0.2}):
        with pytest.raises(C.SpecError):
            C.load_spec({**spec, **bad})
    with pytest.raises(C.SpecError, match="no strength"):
        C.expand(C.load_spec({**spec, "interventions": ["baseline", {"mode": "ablate", "direction": "ego",
                                                                     "strengths": [1]}]}))


# ----------------------------------------------------------------------------------------------- the held-out guard


def test_held_out_items_are_refused(tmp_path):
    spec = tiny_spec(tmp_path)
    with pytest.raises(R.lib.HeldOutError, match="held-out"):
        C.load_spec({**spec, "items": ["ts-90", "ts-99"]})
    with pytest.raises(R.lib.HeldOutError, match="held-out"):
        C.load_spec({**spec, "items": {"threat_to_self": ["ts-99"]}})
    with pytest.raises(R.lib.HeldOutError, match="dev split only"):
        C.load_spec({**spec, "split": "heldout"})
    held_dir = tmp_path / "heldout"
    held_dir.mkdir()
    with pytest.raises(R.lib.HeldOutError):
        C.load_spec({**spec, "library": str(held_dir)})
    with pytest.raises(ValueError, match="no dev items"):
        C.load_spec({**spec, "items": ["ts-50"]})
    ok = C.load_spec({**spec, "items": ["ts-91"]})
    assert ok["chosen"]["items"] == {"threat_to_self": ["ts-91"]}
    good = dict(sweep="s", run_id="r", model="m", intervention={"kind": "steer", "name": "baseline", "params": {}},
                family="threat_to_self", item_id="ts-90", sample=0, messages=MSG,
                result={"text": "x", "prompt_tokens": 3, "completion_tokens": 1, "finish_reason": "stop"},
                cost_usd=0.0, params={}, latency_s=1.0)
    assert R.make_record(**good, split="dev")["split"] == "dev"
    with pytest.raises(ValueError, match="held-out split"):
        R.make_record(**good, split="heldout")


# ----------------------------------------------------------------------------------------------- the spend meter


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def meter(tmp_path, clock, cap=2.0, rate=3.6, tail=600.0, label="s1", log=True):
    return M.Meter(spend_dir=tmp_path / "spend", label=label, session=label, model_key="ministral", cap_usd=cap,
                   rate=rate, tail_s=tail, sweep_log=(tmp_path / f"{label}.jsonl") if log else None, clock=clock,
                   date_fn=lambda: "2026-11-01")


def day_total(tmp_path):
    return M.scout_spend.read_day(tmp_path / "spend", "2026-11-01")["total_usd"]


def test_meter_charges_time_and_tail_once_and_trues_up(tmp_path):
    clk = Clock()
    m = meter(tmp_path, clk)  # $3.60 an hour: a cent a second... /10 (0.001 $/s)
    m.start(first_estimate_s=100)
    assert day_total(tmp_path) == pytest.approx((100 + 600) * 0.001)
    clk.t += 40
    m.after_call()
    assert day_total(tmp_path) == pytest.approx((40 + 600) * 0.001)  # trued down to what ran
    m.before_call(50)
    assert day_total(tmp_path) == pytest.approx((90 + 600) * 0.001)  # reserved ahead of the call
    clk.t += 20
    m.after_call()
    m.close()
    assert day_total(tmp_path) == pytest.approx((60 + 600) * 0.001)
    log = R.schema.read_jsonl(tmp_path / "s1.jsonl")
    assert sum(x["cost_usd"] for x in log) == pytest.approx(day_total(tmp_path))
    assert M.read_state(tmp_path / "spend")["ministral"]["paid_until"] == pytest.approx(clk.t + 600)
    # a second session inside the tail takes the 500 s left of it over: s1 keeps its 60 s and the 100 s idle after
    # it, s2 pays its own 10 s and the tail, and the day is charged each second once
    clk.t += 100
    m2 = meter(tmp_path, clk, label="s2")
    m2.start(first_estimate_s=10)
    clk.t += 10
    m2.after_call()
    m2.close()
    assert m2.charged_usd == pytest.approx((10 + 600) * 0.001)
    assert day_total(tmp_path) == pytest.approx((60 + 600 + 110) * 0.001)
    by = M.scout_spend.read_day(tmp_path / "spend", "2026-11-01")["by_sweep"]
    assert by["bench:s1"]["usd"] == pytest.approx(0.16) and by["bench:s2"]["usd"] == pytest.approx(0.61)
    assert sum(x["cost_usd"] for x in R.schema.read_jsonl(tmp_path / "s1.jsonl")) == pytest.approx(0.16)
    assert sum(x["cost_usd"] for x in R.schema.read_jsonl(tmp_path / "s2.jsonl")) == pytest.approx(0.61)
    # a stop 200 s into s2's tail refunds s2, which holds the tail: nobody's share goes negative
    clk.t += 200
    assert M.refund_tail(tmp_path / "spend", clock=clk, date_fn=lambda: "2026-11-01") == pytest.approx(0.4)
    by = M.scout_spend.read_day(tmp_path / "spend", "2026-11-01")["by_sweep"]
    assert by["bench:s1"]["usd"] == pytest.approx(0.16) and by["bench:s2"]["usd"] == pytest.approx(0.21)
    # a session refused at start takes nothing over
    clk.t += 10
    m3 = meter(tmp_path, clk, label="s3")
    m3.start(first_estimate_s=0)
    m3.close()
    clk.t += 1
    with pytest.raises(M.CapReached):
        meter(tmp_path, clk, cap=0.01, label="s4").start(first_estimate_s=0)
    assert "bench:s4" not in M.scout_spend.read_day(tmp_path / "spend", "2026-11-01")["by_sweep"]


def test_meter_refuses_past_the_daily_cap_and_the_session_cap(tmp_path):
    clk = Clock()
    M.scout_spend.add_spend(tmp_path / "spend", tool="scout", label="x", usd=49.5, date="2026-11-01")
    with pytest.raises(M.CapReached, match="daily cap"):
        meter(tmp_path, clk, cap=5.0).start(first_estimate_s=100)  # 0.7 more would pass 50
    assert day_total(tmp_path) == pytest.approx(49.5)
    m = meter(tmp_path, clk, cap=0.45, tail=60)
    m.start(first_estimate_s=100)  # 0.16 of its 0.45, and 49.66 of the day's 50
    clk.t += 100
    m.after_call()
    with pytest.raises(M.CapReached, match="its cap"):
        m.before_call(300)  # 100 + 300 + 60 s = $0.46 > $0.45
    with pytest.raises(SystemExit):
        meter(tmp_path, clk, cap=0)
    with pytest.raises(SystemExit):
        meter(tmp_path, clk, cap=51)


def test_stop_refunds_the_unused_tail(tmp_path):
    clk = Clock()
    m = meter(tmp_path, clk)
    m.start(first_estimate_s=0)
    clk.t += 30
    m.close()
    assert day_total(tmp_path) == pytest.approx(0.63)
    clk.t += 200  # stopped 200 s into the 600 s tail
    refund = M.refund_tail(tmp_path / "spend", clock=clk, date_fn=lambda: "2026-11-01")
    assert refund == pytest.approx(0.4)
    assert day_total(tmp_path) == pytest.approx(0.23)
    assert M.refund_tail(tmp_path / "spend", clock=clk, date_fn=lambda: "2026-11-01") == 0.0
    assert sum(x["cost_usd"] for x in R.schema.read_jsonl(tmp_path / "s1.jsonl")) == pytest.approx(0.23)


def test_the_rate_is_the_l40s_plus_the_container():
    r = M.rate_per_hour("L40S", 2.0, 16.0)
    assert r["gpu"] == 1.95 and r["total"] == pytest.approx(1.95 + 0.28 + 0.384)
    assert C.RATE == r
    assert math.isclose(C.SCALEDOWN_S, 600)


def test_spec_files_load():
    for p in sorted((BENCH / "specs").glob("*.yaml")):
        spec = C.load_spec(p)
        assert C.expand(spec)
        assert yaml.safe_load(p.read_text())["sweep"] == spec["sweep"]
