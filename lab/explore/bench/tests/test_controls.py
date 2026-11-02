"""CPU tests for bench's additions of 1 Nov 2026: tensor hashes and the {vectors, best_layer, report} format, passes
and calls planned by a token budget, the failure log's call number, partial cuts, the container's measure and
response_means, and the other_trait and split_null controls end to end on the tiny model. No Modal, no GPU, no
downloads. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private (lab/explore)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import BENCH, HIDDEN, LAYERS

import client as C
import controls as K
import directions as D
import records as R
import steering as S
from anatta.dial import intervene
from anatta.probe import capture, response_activations

MSG = [{"role": "system", "content": "the quick brown fox"}, {"role": "user", "content": "lazy dog final answer"}]
NEUTRAL = ["the quick brown fox", "over the lazy dog 0123", "final answer 456789", "the lazy brown dog jumps",
           "quick brown dog jumps over", "final answer 42 the fox", "brown fox lazy dog 7", "the answer 0123"]
SELF_STAKES = "/Users/eugene/prj/align/explore-results/2026-11-01/lead1-grievance/inputs/self_stakes_17_26.pt"
SELF_STAKES_TENSOR = "10e69d9eb31d7fb08be4e6410b03a6e791a79933a28daf53e46addc452f15760"


class Report:
    """Stands in for a report object that only a full unpickle can load."""

    def __init__(self, text):
        self.text = text


def rand_dir(seed=0, layers=range(LAYERS), dim=HIDDEN):
    g = torch.Generator().manual_seed(seed)
    return {x: torch.randn(dim, generator=g) for x in layers}


def save_direction(path, vecs, name="d"):
    torch.save({"name": name, "vectors": vecs, "scores": {x: float(x) for x in vecs},
                "pos_mean": {x: 0.0 for x in vecs}, "neg_mean": {x: 0.0 for x in vecs}, "meta": {}}, path)
    return str(path)


def ref_of(path, name):
    d = D.resolve({"path": str(path)}, "ministral")
    return {"path": str(path), "tensor_sha256": d.tensor_sha256, "name": name}


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / v.norm()


def read_block(model, tok, layers, ivs=()):
    ids = tok.apply_chat_template(MSG, add_generation_prompt=True, return_tensors="pt")
    ids = ids if isinstance(ids, torch.Tensor) else ids["input_ids"]
    with intervene(model, list(ivs)), capture(model, layers) as store:
        model(input_ids=ids)
    return {x: store[x][0].clone() for x in layers}


def req(rid, seed, iv, messages=MSG, max_tokens=8, temperature=0.9):
    return {"rid": rid, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            "intervention": iv}


# ----------------------------------------------------------------------------------------------- tensor hashes


def test_the_tensor_hash_is_stable_across_saves_and_follows_the_content(tmp_path):
    vecs = rand_dir(1)
    a, b = save_direction(tmp_path / "a.pt", vecs), save_direction(tmp_path / "b.pt", vecs)
    assert D.sha256_file(a) != D.sha256_file(b)  # torch.save is not byte-stable
    assert D.load_file(a)[3] == D.load_file(b)[3] == D.tensor_sha256(vecs)
    h = D.tensor_sha256(vecs)
    changed = {x: v.clone() for x, v in vecs.items()}
    changed[2][5] += 1e-6
    assert D.tensor_sha256(changed) != h
    assert D.tensor_sha256({x: v.double() for x, v in vecs.items()}) != h  # dtype counts
    assert D.tensor_sha256({x + 1: v for x, v in vecs.items()}) != h  # so do the layers
    assert D.tensor_sha256({x: v[None] for x, v in vecs.items()}) != h  # and the shape
    assert D.tensor_sha256({x: v.numpy() for x, v in vecs.items()}) == h  # arrays and tensors alike
    base = torch.randn(HIDDEN, 3)
    view = base[:, 1]
    assert not view.is_contiguous() and D.tensor_sha256({0: view}) == D.tensor_sha256({0: view.contiguous()})
    assert D.tensor_sha256(vecs[0]) != D.tensor_sha256({0: vecs[0]})  # a single vector is keyed "single"


def test_every_given_pin_binds_and_a_tensor_pin_accepts_a_resaved_file(tmp_path):
    vecs = rand_dir(2)
    p = save_direction(tmp_path / "d.pt", vecs)
    first = D.resolve({"path": p}, "ministral")
    assert first.tensor_sha256 == D.tensor_sha256(vecs) and first.note.startswith("unpinned")
    old_file = first.sha256
    save_direction(p, vecs, name="resaved")  # re-saved: same tensors, new bytes
    assert D.sha256_file(p) != old_file
    again = D.resolve({"path": p, "tensor_sha256": first.tensor_sha256}, "ministral")
    assert again.tensor_sha256 == first.tensor_sha256 and again.key == first.key and again.note is None
    with pytest.raises(D.DirectionError, match="pin its tensor_sha256 alone"):
        D.resolve({"path": p, "tensor_sha256": first.tensor_sha256, "sha256": old_file}, "ministral")
    with pytest.raises(D.DirectionError, match="tensor sha256"):
        D.resolve({"path": p, "tensor_sha256": "0" * 64}, "ministral")
    assert D.resolve({"path": p, "sha256": D.sha256_file(p)}, "ministral").tensor_sha256 == first.tensor_sha256


def test_the_vectors_best_layer_report_format_reads_only_those_two_keys(tmp_path):
    vecs = {17: torch.randn(HIDDEN), 18: torch.randn(HIDDEN), "19": torch.randn(HIDDEN)}
    p = tmp_path / "stakes.pt"
    torch.save({"vectors": vecs, "best_layer": 18, "report": Report("never read")}, p)
    with pytest.raises(D.DirectionError, match="full unpickle"):  # unpinned bytes are never fully unpickled
        D.resolve({"path": str(p)}, "ministral")
    d = D.resolve({"path": str(p), "sha256": D.sha256_file(p)}, "ministral")
    assert sorted(d.vectors) == [17, 18, 19] and d.best_layer == 18 and d.layers_for("best") == [18]
    assert d.tensor_sha256 == D.tensor_sha256({17: vecs[17], 18: vecs[18], 19: vecs["19"]})
    assert "report" not in json.dumps(d.describe(), default=str)
    plain = tmp_path / "plain.pt"
    torch.save({"vectors": vecs, "best_layer": 18}, plain)  # without the report it loads as plain tensors
    assert D.resolve({"path": str(plain)}, "ministral").tensor_sha256 == d.tensor_sha256
    stacked = tmp_path / "stacked.pt"
    torch.save({"vectors": torch.stack([vecs[17], vecs[18]]), "best_layer": None}, stacked)
    with pytest.raises(D.DirectionError, match="row0_layer"):
        D.resolve({"path": str(stacked)}, "ministral")
    s = D.resolve({"path": str(stacked), "row0_layer": 17}, "ministral")
    assert sorted(s.vectors) == [17, 18] and s.best_layer is None
    assert np.array_equal(s.vectors[18], vecs[18].numpy())


def test_registry_entries_match_their_tensor_hashes_and_the_converted_stakes_copy_loads():
    reg = D.load_registry()
    for model, entries in reg["directions"].items():
        for name, e in entries.items():
            assert e["tensor_sha256"] and D.resolve(name, model, reg).tensor_sha256 == e["tensor_sha256"], (model, name)
    d = D.resolve({"path": SELF_STAKES, "tensor_sha256": SELF_STAKES_TENSOR, "name": "self_stakes"}, "ministral")
    assert sorted(d.vectors) == list(range(17, 27)) and d.dim() == 4096


def test_controls_are_named_by_their_tensors_and_an_existing_file_is_kept(tmp_path):
    tgt = D.Dir(name="t", sha256="f" * 64, path=None, vectors={0: np.ones(HIDDEN, np.float32)}, tensor_sha256="a" * 64)
    vecs = {x: v.numpy() for x, v in rand_dir(3).items()}
    one = D.save_control(tmp_path, tgt, 0, vecs, {"why": "test"}, family=K.SPLIT_METHOD.split("_v")[0])
    before = (Path(one.path).stat().st_mtime_ns, D.sha256_file(one.path))
    two = D.save_control(tmp_path, tgt, 0, vecs, {"why": "again"}, family="split_null")
    assert two.path == one.path and (Path(two.path).stat().st_mtime_ns, D.sha256_file(two.path)) == before
    assert one.tensor_sha256 == D.tensor_sha256(vecs) and one.tensor_sha256[:8] in Path(one.path).name
    assert json.loads(Path(one.path).with_suffix(".json").read_text())["made"] == {"why": "test"}  # not overwritten


# ----------------------------------------------------------------------------------------------- passes and calls


def test_passes_respect_the_token_budget_and_the_row_cap():
    lengths = [10, 500, 20, 4000, 30, 900, 15, 2200]
    plan = S.plan_passes(lengths, 100, token_budget=2000, max_rows=3)
    assert sorted(i for p in plan for i in p) == list(range(len(lengths)))
    flat = [i for p in plan for i in p]
    assert [lengths[i] for i in flat] == sorted(lengths)  # shortest first
    for p in plan:
        assert len(p) <= 3
        assert len(p) == 1 or len(p) * (max(lengths[i] for i in p) + 100) <= 2000
    assert S.plan_passes(lengths, 100, token_budget=10**9, max_rows=3) == [flat[0:3], flat[3:6], flat[6:8]]
    # the lead1 out-of-memory pass (32 rows of 4,002 tokens and 400 new) now splits within the default budget
    big = S.plan_passes([4002] * 32, 400)
    assert [len(p) for p in big] == [18, 14] and 18 * 4402 <= S.TOKEN_BUDGET < 20 * 4402
    assert S.plan_passes([5000], 400, token_budget=100) == [[0]]  # a row alone always runs
    assert S.plan_passes([10, 10, 10], [5, 500, 5], token_budget=1100, max_rows=8) == [[0, 1], [2]]


def test_generation_is_the_same_under_any_token_budget(svc):
    msgs = [[{"role": "user", "content": "the " * n + "fox"}] for n in (1, 9, 3, 30, 14, 2)]
    reqs = [req(i, 40 + i, {"mode": "none"}, messages=m) for i, m in enumerate(msgs)]
    wide = svc.generate({"requests": reqs, "directions": {}, "token_budget": 10**6})
    tight = svc.generate({"requests": reqs, "directions": {}, "token_budget": 60})
    assert [r["text"] for r in wide["results"]] == [r["text"] for r in tight["results"]]
    assert len(wide["meta"]["groups"][0]["passes"]) < len(tight["meta"]["groups"][0]["passes"])
    assert tight["meta"]["token_budget"] == 60 and all(r["pass_rows"] >= 1 for r in tight["results"])


def test_calls_are_packed_by_requests_and_planned_steps():
    assert C.TOKEN_BUDGET == S.TOKEN_BUDGET and C.GEN_BATCH == 32
    no_sys = [{"role": "user", "content": "x" * 100}]
    with_sys = [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 100}]
    assert C.est_tokens(no_sys) - C.est_tokens(with_sys) > 500  # Ministral's default system prompt
    lengths = [4400] * 40 + [100] * 40
    groups = ["a"] * 20 + ["b"] * 20 + ["a"] * 40
    plan = C.plan_calls(lengths, groups, batch_size=64, max_tokens=400, token_budget=S.TOKEN_BUDGET)
    assert plan[0][0] == 0 and plan[-1][1] == len(lengths)
    assert all(a[1] == b[0] for a, b in zip(plan, plan[1:]))
    for start, end, passes in plan:
        assert end - start <= 64 and (end - start == 1 or passes * 400 <= C.MAX_CALL_STEPS)
        exp = sum(len(C.plan_passes([lengths[i] for i in range(start, end) if groups[i] == g], 400,
                                    S.TOKEN_BUDGET)) for g in set(groups[start:end]))
        assert passes == exp
    short = C.plan_calls([50] * 100, ["a"] * 100, batch_size=16, max_tokens=6, token_budget=S.TOKEN_BUDGET)
    assert [e - s for s, e, _ in short] == [16] * 6 + [4]  # short prompts: batch_size alone decides


def test_a_failed_call_is_logged_with_its_own_number(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "neutral_set", lambda: (NEUTRAL, "nsha"))
    p = save_direction(tmp_path / "t.pt", rand_dir(4))
    spec = {"sweep": "bench-fail", "model": "ministral", "library": "fixture", "items": "all",
            "interventions": ["baseline", {"mode": "ablate", "direction": ref_of(p, "t")}],
            "samples": 2, "max_tokens": 4, "spend_cap_usd": 2.0, "batch_size": 4,
            "allow_missing_controls": "test: the failure log"}
    bench, n = C.LocalBench(svc), {"calls": 0}
    real = bench.generate

    def flaky(payload):
        n["calls"] += 1
        if n["calls"] == 2:
            raise RuntimeError("out of memory (test)")
        return real(payload)

    bench.generate = flaky
    lines = []
    man = C.run_sweep(spec, bench=bench, root=tmp_path / "r", log=lines.append)
    errors = R.schema.read_jsonl(tmp_path / "r" / man["date"] / "bench-fail" / "errors.jsonl")
    assert man["status"] == "error" and man["records"] == 4 and len(man["bench"]["calls"]) == 1
    assert errors[0]["call"] == 2 and any(line.startswith("[bench] call 2 failed") for line in lines)


# ----------------------------------------------------------------------------------------------- partial cuts


def test_a_partial_cut_scales_the_component_and_a_whole_one_is_the_plain_cut(tiny):
    model, tok = tiny
    vecs = rand_dir(5)
    base = read_block(model, tok, [1])[1]
    part = read_block(model, tok, [1], S.build_interventions("ablate", vecs, [1], fractions={1: 0.25}))[1]
    u = unit(vecs[1])
    assert torch.allclose(part @ u, 0.75 * (base @ u), atol=1e-4)
    assert torch.allclose(part - (part @ u)[:, None] * u, base - (base @ u)[:, None] * u, atol=1e-4)
    whole = S.build_interventions("ablate", vecs, [1, 2], fractions={1: 1.0, 2: 1.0})
    plain = S.build_interventions("ablate", vecs, [1, 2])
    assert S.cut_diff(whole, plain) == 0.0 and all(type(x) is type(y) for x, y in zip(whole, plain))
    mixed = S.build_interventions("ablate", vecs, [1, 2], fractions={1: 0.5})
    assert [getattr(x, "fraction", 1.0) for x in mixed] == [0.5, 1.0] and S.cut_diff(mixed, plain) == float("inf")
    for bad in ({1: 0.0}, {1: 1.5}, {1: -0.2}):
        with pytest.raises(ValueError, match="fractions"):
            S.build_interventions("ablate", vecs, [1], fractions=bad)
    with pytest.raises(ValueError, match="fractions"):
        S.build_interventions("add", vecs, [1], 2.0, fractions={1: 0.5})
    iv = {"mode": "ablate", "sha256": "x", "layers": [1]}
    assert S.iv_key(iv) != S.iv_key({**iv, "fractions": {1: 0.5}})
    assert S.iv_key({**iv, "tensor_sha256": "t"}) != S.iv_key(iv)  # the tensor hash keys the direction


def test_the_service_runs_a_partial_cut_and_checks_it(svc, tiny):
    model, _ = tiny
    vecs = rand_dir(6)
    payload = {"name": "d", "vectors": {x: v.numpy() for x, v in vecs.items()}}
    iv = {"mode": "ablate", "tensor_sha256": "k", "sha256": "f", "layers": [0, 1, 2, 3], "direction": "d",
          "fractions": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}}
    out = svc.generate({"requests": [req(0, 1, iv), req(1, 1, {**iv, "fractions": None})], "directions": {"k": payload}})
    groups = out["meta"]["groups"]
    assert len(groups) == 2 and S.count_hooks(model) == 0
    part = next(g for g in groups if g["intervention"]["fractions"])["attach"]
    whole = next(g for g in groups if not g["intervention"]["fractions"])["attach"]
    assert part["fraction"] == 0.5 and abs(part["after_over_before"] - 0.5) < 1e-4
    assert whole["proj_after"] < 1e-4 and "fraction" not in whole


# ----------------------------------------------------------------------------------------------- measure, response_means


def test_measure_matches_an_independent_reading_and_random_matched(svc):
    a, b = rand_dir(7), rand_dir(8)
    dirs = {"a": {"name": "a", "vectors": {x: v.numpy() for x, v in a.items()}},
            "b": {"name": "b", "vectors": {x: v.numpy() for x, v in b.items()}}}
    res = svc.measure({"directions": dirs, "layers": [1, 2, 3], "neutral_prompts": NEUTRAL, "neutral_sha256": "n"})
    acts, _ = svc.neutral_activations(NEUTRAL, [1, 2, 3])
    for x in (1, 2, 3):
        u = a[x].double()
        p = acts[x].double() @ (u / u.norm())
        assert res["stats"]["a"][x]["var"] == pytest.approx(float(p.var()), rel=1e-9)
        assert res["stats"]["a"][x]["second_moment"] == pytest.approx(float(p.pow(2).mean()), rel=1e-9)
        assert res["stats"]["a"][x]["mean_proj"] == pytest.approx(float(p.mean()), rel=1e-9, abs=1e-9)
    assert res["neutral"]["sha256"] == "n" and set(res["stats"]) == {"a", "b"}
    rm = svc.random_matched({"target": {"name": "a", "sha256": "a", **dirs["a"]}, "seeds": [0], "layers": [1, 2, 3],
                             "neutral_prompts": NEUTRAL, "neutral_sha256": "n"})["directions"][0]
    again = svc.measure({"directions": {"a": dirs["a"], "r": {"name": "r", "vectors": rm["vectors"]}},
                         "layers": [1, 2, 3], "neutral_prompts": NEUTRAL})
    table = K.variance_table(again["stats"], "a", "r", [1, 2, 3])
    for x in (1, 2, 3):
        assert table[x]["ratio"] == pytest.approx(rm["made"]["layers"][str(x)]["ratio"], rel=1e-6)


def test_response_means_reads_replies_as_probe_extract_does(svc, tiny):
    model, tok = tiny
    rows = [{"id": "c1:pos", "system": "you are a lazy fox", "user": "the quick brown dog", "seed": 11},
            {"id": "c1:neg", "system": "you are a quick dog", "user": "the quick brown dog", "seed": 12},
            {"id": "c2:pos", "system": "answer 0123", "user": "final answer", "seed": 13}]
    payload = {"rows": rows, "layers": [1, 3], "max_tokens": 6, "temperature": 0.7, "top_p": 0.95}
    out = svc.response_means(payload)
    assert [r["id"] for r in out["replies"]] == ["c1:pos", "c1:neg", "c2:pos"]
    assert out["means"][1].shape == (3, HIDDEN) and out["means"][1].dtype == np.float32
    assert not out["meta"]["include_system"] and S.count_hooks(model) == 0
    for i, (r, rep) in enumerate(zip(rows, out["replies"])):
        ref = response_activations(model, tok, [{"role": "user", "content": r["user"]}], rep["text"], [1, 3])
        for x in (1, 3):
            assert np.allclose(out["means"][x][i], ref[x].numpy(), atol=1e-5)
    again = svc.response_means(payload)
    assert [r["text"] for r in again["replies"]] == [r["text"] for r in out["replies"]]  # a seed per row
    with_sys = svc.response_means({**payload, "include_system": True})
    assert not np.allclose(with_sys["means"][3], out["means"][3])


# ----------------------------------------------------------------------------------------------- the bank and its nulls


def test_bank_rows_draw_distinct_questions_with_both_signs():
    rows, info = K.bank_rows(cells_per_trait=8, seed=0)
    assert len(rows) == 6 * 8 * 2 and info["n_cells"] == 48 and info["traits"] == list(K.EGO_TRAITS)
    assert set(info["trait_files_sha256"]) == set(K.EGO_TRAITS)
    for t in K.EGO_TRAITS:
        mine = [r for r in rows if r["trait"] == t]
        assert len({r["q"] for r in mine}) == 8
        for cell in {r["cell"] for r in mine}:
            pair = [r for r in mine if r["cell"] == cell]
            assert sorted(r["sign"] for r in pair) == [-1, 1] and len({r["user"] for r in pair}) == 1
            assert pair[0]["system"] != pair[1]["system"]
    assert K.bank_rows(cells_per_trait=8, seed=0)[0] == rows and K.bank_rows(cells_per_trait=8, seed=1)[0] != rows
    assert len({r["seed"] for r in rows}) == len(rows)
    with pytest.raises(ValueError):
        K.bank_rows(cells_per_trait=13)


def synthetic_bank(signal=0.0, empty=(), seed=0, dim=48, layers=(3, 4)):
    """6 traits x 8 cells: a cell's content (shared by its two replies) + sign x signal x a planted direction +
    reply noise; stretched along a few axes, like real activations."""
    g = torch.Generator().manual_seed(seed)
    planted = unit(torch.randn(dim, generator=g)).double()
    scale = torch.logspace(1, -1, dim, dtype=torch.float64)
    rows, means = [], {x: [] for x in layers}
    for t in K.EGO_TRAITS:
        for q in range(8):
            cell = f"{t}:q{q}:p0"
            content = {x: torch.randn(dim, generator=g, dtype=torch.float64) * scale for x in layers}
            for sign in (1, -1):
                rows.append({"id": f"{cell}:{sign}", "cell": cell, "trait": t, "q": q, "pair": 0, "sign": sign,
                             "empty": f"{cell}:{sign}" in empty})
                for x in layers:
                    noise = torch.randn(dim, generator=g, dtype=torch.float64) * scale * 0.3
                    means[x].append(content[x] + noise + sign * signal * planted)
    m = {x: torch.stack(v).float() for x, v in means.items()}
    return {"path": None, "tensor_sha256": D.tensor_sha256(m), "sha256": None, "means": m, "rows": rows,
            "meta": {"method": K.BANK_METHOD, "traits": list(K.EGO_TRAITS)}}, planted


def test_the_split_null_cancels_the_trait_signal_exactly():
    quiet, planted = synthetic_bank(signal=0.0)
    loud, _ = synthetic_bank(signal=500.0)
    tgt = D.Dir(name="planted", sha256=None, path=None, vectors={x: planted.float().numpy() for x in (3, 4)},
                tensor_sha256="p" * 64)
    for seed in (0, 1, 2):
        a, made_q = K.split_null(quiet, seed, [3, 4], tgt, chance_n=50)
        b, made_l = K.split_null(loud, seed, [3, 4], tgt, chance_n=50)
        for x in (3, 4):
            assert np.allclose(a[x], b[x], atol=1e-4)  # 500 x the planted signal changes nothing
            assert abs(float(np.linalg.norm(a[x])) - 1) < 1e-5
        assert made_q["partition"] == made_l["partition"]
        A, B = made_q["partition"]["A"], made_q["partition"]["B"]
        assert not set(A) & set(B) and len(A) == len(B) == 24
        assert all(sum(c.startswith(t + ":") for c in A) == 4 for t in K.EGO_TRAITS)  # balanced within each trait
        lay = made_l["layers"]["3"]
        assert 0 <= lay["abs_cos_chance_p50"] <= lay["abs_cos_chance_p95"] <= lay["abs_cos_chance_max"] <= 1
        assert made_l["method"] == K.SPLIT_METHOD and made_l["rule"] == K.SPLIT_RULE
    assert K.partition(quiet, 0) != K.partition(quiet, 1)
    rc = K.recipe_check(loud, {"planted": tgt}, [3, 4])  # the pos - neg difference does carry the signal
    assert rc["pooled"]["planted"][3] > 0.99
    flipped = loud["means"][3].double()  # a label-flip null keeps the signal, which is why C4 split instead
    signs = torch.tensor([r["sign"] for r in loud["rows"]], dtype=torch.float64)
    keep = torch.tensor([1.0 if (i // 2) % 2 == 0 else -1.0 for i in range(len(signs))], dtype=torch.float64)
    flip_null = ((keep * signs) @ flipped) / len(signs)
    assert abs(float(unit(flip_null).double() @ planted)) < 0.5 < abs(float(unit(signs @ flipped).double() @ planted))
    dropped_bank, _ = synthetic_bank(empty={"approval_seeking:q0:p0:1"})
    cells, dropped = K.usable_cells(dropped_bank)
    assert dropped == ["approval_seeking:q0:p0"] and len(cells["approval_seeking"]) == 7


def test_the_variance_table_and_partial_fractions():
    stats = {"t": {1: {"var": 2.0, "second_moment": 3.0, "mean_proj": 1.0},
                   2: {"var": 2.0, "second_moment": 2.5, "mean_proj": 0.5}},
             "c": {1: {"var": 8.0, "second_moment": 9.0, "mean_proj": 1.0},
                   2: {"var": 1.0, "second_moment": 1.0, "mean_proj": 0.0}}}
    table = K.variance_table(stats, "t", "c", [1, 2])
    assert table[1]["ratio"] == 4.0 and table[2]["ratio"] == 0.5 and table[1]["moment_ratio"] == 3.0
    fr = K.partial_fractions(table)
    assert fr == {1: 0.5, 2: 1.0} and fr[1] ** 2 * table[1]["ratio"] == pytest.approx(1.0)
    s = K.summarize(table, {1: -0.2, 2: 0.1}, fr)
    assert s == {"ratio_range": [0.5, 4.0], "moment_ratio_range": [0.4, 3.0], "abs_cos_range": [0.1, 0.2],
                 "fraction_range": [0.5, 1.0]}
    fm = K.partial_fractions(table, "partial_moment")  # on the mean-square ratio: 3.0 and 0.4
    assert fm == {1: pytest.approx(math.sqrt(1 / 3)), 2: 1.0} and K.partial_fractions(table, "full") is None


# ----------------------------------------------------------------------------------------------- sweeps end to end


@pytest.fixture
def small_bank(monkeypatch):
    monkeypatch.setitem(K.BANK, "cells_per_trait", 2)
    monkeypatch.setitem(K.BANK, "max_tokens", 6)
    monkeypatch.setattr(C, "neutral_set", lambda: (NEUTRAL, "nsha"))


def control_spec(tmp_path, **over):
    tgt = ref_of(save_direction(tmp_path / "target.pt", rand_dir(10)), "target")
    calm = ref_of(save_direction(tmp_path / "calm.pt", rand_dir(11)), "calm")
    style = ref_of(save_direction(tmp_path / "style.pt", rand_dir(12)), "style")
    spec = {"sweep": "bench-controls-test", "model": "ministral", "library": "fixture", "items": ["ts-90", "ts-91"],
            "interventions": ["baseline", {"mode": "ablate", "direction": tgt},
                              {"mode": "ablate", "direction": "other_trait", "target": tgt, "traits": [calm, style]},
                              {"mode": "ablate", "direction": "other_trait", "target": tgt, "traits": [calm],
                               "match": "partial"},
                              {"mode": "ablate", "direction": "split_null", "target": tgt, "seeds": [0, 1]},
                              {"mode": "ablate", "direction": "split_null", "target": tgt, "seeds": [0],
                               "layers": "1-2"},
                              {"mode": "add", "direction": "other_trait", "target": tgt, "traits": [style],
                               "layers": [1], "strengths": [2]},
                              {"mode": "ablate", "direction": "other_trait", "target": tgt, "traits": [style],
                               "match": "partial_moment"},
                              {"mode": "ablate", "direction": "random_matched", "target": tgt, "seeds": [0]}],
            "samples": 1, "temperature": 0.8, "max_tokens": 5, "seed": 3, "spend_cap_usd": 2.0, "batch_size": 8}
    spec.update(over)
    return spec, tgt


def test_a_sweep_with_other_trait_and_split_null_controls(svc, tmp_path, small_bank):
    spec, tgt = control_spec(tmp_path)
    root = tmp_path / "results"
    man = C.run_sweep(spec, bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    sd = root / man["date"] / "bench-controls-test"
    names = [v["name"] for v in man["variants"]]
    assert names == ["baseline", "ablate:target", "ablate:other_trait(target)=calm",
                     "ablate:other_trait(target)=style", "ablate:other_trait(target)=calm~partial",
                     "ablate:split_null(target)#s0", "ablate:split_null(target)#s1", "ablate:split_null(target)#s0[1-2]",
                     "add:other_trait(target)=style[1]@+2", "ablate:other_trait(target)=style~partial_moment",
                     "ablate:random_matched(target)#s0"]
    assert man["status"] == "complete" and man["records"] == man["jobs"] == 2 * len(names)
    assert man["missing_controls"] == [] and man["bench"]["controls"] == ["other_trait", "random_matched", "split_null"]
    recs = R.schema.read_jsonl(sd / "records.jsonl")
    assert all(R.schema.validate_record(r) == [] for r in recs)
    by = {r["intervention"]["name"]: r["intervention"]["params"] for r in recs}
    assert by["ablate:target"]["tensor_sha256"] == tgt["tensor_sha256"]
    other = by["ablate:other_trait(target)=calm"]
    assert other["control"] == "other_trait" and other["layers"] == [0, 1, 2, 3] and "fractions" not in other
    mk = other["made"]
    assert mk["method"] == K.OTHER_METHOD and mk["trait"] == "calm" and mk["target_tensor_sha256"] == tgt["tensor_sha256"]
    assert mk["ratio_range"][0] > 0 and 0 <= mk["abs_cos_range"][0] <= mk["abs_cos_range"][1] <= 1
    table = json.loads((sd / "directions" / mk["file"]).read_text())
    assert set(table["layers"]) == {"0", "1", "2", "3"} and table["match"] == "full"
    part = by["ablate:other_trait(target)=calm~partial"]
    fr = {int(x): f for x, f in part["fractions"].items()}
    for x, f in fr.items():
        r = table["layers"][str(x)]["ratio"]
        assert f == pytest.approx(min(1.0, math.sqrt(1 / r)), rel=1e-5) and 0 < f <= 1
    assert part["made"]["fraction_range"] == [round(min(fr.values()), 4), round(max(fr.values()), 4)]
    pm = by["ablate:other_trait(target)=style~partial_moment"]
    style_table = json.loads((sd / "directions" / pm["made"]["file"]).read_text())["layers"]
    assert pm["made"]["match"] == "partial_moment" and set(pm["fractions"]) == {"0", "1", "2", "3"}
    for x, f in pm["fractions"].items():
        assert f == pytest.approx(min(1.0, math.sqrt(1 / style_table[x]["moment_ratio"])), rel=1e-5)
    rm = by["ablate:random_matched(target)#s0"]["made"]
    rvecs = D.load_file(sd / "directions" / rm["file"])[0]
    tvecs = {x: v.numpy() for x, v in rand_dir(10).items()}
    st = svc.measure({"directions": {"t": {"name": "t", "vectors": tvecs}, "r": {"name": "r", "vectors": rvecs}},
                      "layers": [0, 1, 2, 3], "neutral_prompts": NEUTRAL})["stats"]
    mr = [st["r"][x]["second_moment"] / st["t"][x]["second_moment"] for x in range(4)]
    assert rm["moment_ratio_range"] == pytest.approx([min(mr), max(mr)], rel=1e-4)  # from its made record alone
    s0, s1 = by["ablate:split_null(target)#s0"], by["ablate:split_null(target)#s1"]
    assert s0["control"] == "split_null" and s0["made"]["seed"] == 0 and s0["tensor_sha256"] != s1["tensor_sha256"]
    band = by["ablate:split_null(target)#s0[1-2]"]
    assert band["tensor_sha256"] == s0["tensor_sha256"] and band["layers"] == [1, 2]  # one null, cut on a band
    saved = sd / "directions" / s0["made"]["file"]
    assert D.load_file(saved)[3] == s0["tensor_sha256"] and D.sha256_file(saved) == s0["sha256"]
    full = json.loads(saved.with_suffix(".json").read_text())["made"]
    assert set(full["layers"]) == {"0", "1", "2", "3"}
    assert {"cos_target", "ratio", "abs_cos_chance_p95"} <= set(full["layers"]["2"])
    (bank,) = man["bench"]["banks"]
    assert bank["built_here"] and bank["rows"] == 6 * 2 * 2 and bank["layers"] == [0, 1, 2, 3]
    assert s0["made"]["bank_tensor_sha256"] == bank["tensor_sha256"]
    assert (sd / "directions" / (Path(bank["file"]).stem + "-replies.jsonl")).is_file()
    push = by["add:other_trait(target)=style[1]@+2"]
    assert push["coefficient"] == 2.0 and push["layers"] == [1]

    # the bank, pinned: the same nulls come back without building another
    again_spec, _ = control_spec(tmp_path, sweep="bench-controls-pinned", interventions=[
        "baseline", {"mode": "ablate", "direction": "split_null", "target": tgt, "seeds": [0, 1],
                     "bank": {"path": bank["file"], "tensor_sha256": bank["tensor_sha256"]}}])
    again = C.run_sweep(again_spec, bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    (b2,) = again["bench"]["banks"]
    assert not b2["built_here"] and b2["tensor_sha256"] == bank["tensor_sha256"]
    by2 = {v["name"]: v["params"] for v in again["variants"]}
    assert by2["ablate:split_null(target)#s0"]["tensor_sha256"] == s0["tensor_sha256"]
    assert by2["ablate:split_null(target)#s1"]["tensor_sha256"] == s1["tensor_sha256"]


def test_controls_alone_for_other_trait_and_split_null(svc, tmp_path, small_bank):
    spec, tgt = control_spec(tmp_path)
    calm = spec["interventions"][2]["traits"][0]
    root = tmp_path / "results"
    other = C.build_controls(tgt, "ministral", [], kind="other_trait", traits=[calm], layers="1-3",
                             bench=C.LocalBench(svc), root=root, log=lambda *_: None)
    (row,) = other["controls"]
    assert row["layers"] == [1, 2, 3] and set(row["by_layer"]) == {"1", "2", "3"}
    assert {"ratio", "moment_ratio", "cos_target"} <= set(row["by_layer"]["2"]) and Path(row["file"]).is_file()
    assert Path(other["summary_file"]).is_file() and other["charged_usd"] > 0
    nulls = C.build_controls(tgt, "ministral", [0, 1, 2], kind="split_null", bench=C.LocalBench(svc), root=root,
                             log=lambda *_: None)
    assert len(nulls["controls"]) == 3 and len({c["tensor_sha256"] for c in nulls["controls"]}) == 3
    assert nulls["banks"][0]["built_here"] and all(Path(c["file"]).is_file() for c in nulls["controls"])
    assert {"abs_cos_chance_p95", "cos_target", "ratio"} <= set(nulls["controls"][0]["by_layer"]["0"])


def test_the_new_controls_are_checked_in_the_spec(tmp_path):
    spec, tgt = control_spec(tmp_path)
    calm = spec["interventions"][2]["traits"][0]
    bad = [{"mode": "ablate", "direction": "other_trait", "target": tgt},
           {"mode": "ablate", "direction": "other_trait", "target": tgt, "traits": [tgt]},
           {"mode": "add", "direction": "other_trait", "target": tgt, "traits": [calm], "strengths": [1],
            "match": "partial"},
           {"mode": "ablate", "direction": "other_trait", "target": tgt, "traits": [calm], "match": "half"},
           {"mode": "add", "direction": "other_trait", "target": tgt, "traits": [calm], "strengths": [1],
            "match": "partial_moment"},
           {"mode": "ablate", "direction": "other_trait", "target": tgt, "traits": [calm], "seeds": [0]},
           {"mode": "ablate", "direction": "split_null", "target": tgt, "bank": {"path": "/x.pt"}},
           {"mode": "ablate", "direction": "split_null", "seeds": [0]},
           {"mode": "ablate", "direction": "random_matched", "target": tgt, "match": "partial"},
           {"mode": "ablate", "direction": tgt, "match": "partial"},
           {"mode": "ablate", "direction": tgt, "traits": [calm]}]
    for b in bad:
        with pytest.raises(C.SpecError):
            C.expand(C.load_spec({**spec, "interventions": ["baseline", b]}))
    with pytest.raises(C.SpecError, match="token_budget"):
        C.load_spec({**spec, "token_budget": 10})
    only_other = C.expand(C.load_spec({**spec, "interventions": spec["interventions"][:3]}))
    assert C.missing_controls(only_other) == [] and C.control_kinds(only_other) == ["other_trait"]


def test_the_container_and_the_client_agree():
    src = (BENCH / "modal_bench.py").read_text()
    assert '"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"' in src
    for m in ("def measure(self, payload", "def response_means(self, payload"):
        assert m in src
    for m in ("measure", "response_means"):
        assert callable(getattr(C.RemoteBench, m)) and callable(getattr(C.LocalBench, m))
        assert callable(getattr(S.Service, m))
