"""P1's CPU tests: the templates, the paired interval, the criteria logic, the checks that keep captions on Modal, the
hooked features against K1's embed_texts, the scores against K1's code, the spend meter, and the analysis and report on
made-up numbers. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private (lab/explore). Nothing here touches Modal, a
GPU or the network; the captions are made up."""

import json
import math
import re

import pytest
import torch

import p1_core as core

SPEC = core.load_spec()
WORDS = SPEC["lowercase_words"]
G = SPEC["groups"]


# ------------------------------------------------------------------------------------------------ the templates


def test_the_templates_are_the_spec_s_and_well_formed():
    t = SPEC["templates"]
    assert t == {"plain": "{c}", "fp1": "I see {c}", "fp2": "I'm looking at {c}", "fp3": "In front of me, {c}",
                 "nt1": "One sees {c}", "nt2": "There appears to be {c}", "nt3": "In the scene shown, {c}"}
    for tpl in t.values():
        core.check_template(tpl)
    assert G == {"plain": ["plain"], "fp": ["fp1", "fp2", "fp3"], "nt": ["nt1", "nt2", "nt3"]}
    assert SPEC["pairs_matched"] == {"fp1": "nt1", "fp2": "nt2", "fp3": "nt3"}
    first = {"i", "i'm", "me", "my", "mine", "we", "us", "our"}
    for name in G["fp"]:
        assert set(re.findall(r"[a-z']+", t[name].lower())) & first
    for name in G["nt"]:
        assert not set(re.findall(r"[a-z']+", t[name].lower())) & first
    assert all(ch.isascii() for tpl in t.values() for ch in tpl)  # the apostrophe is U+0027
    assert [core.prefix(t[n]) for n in t] == ["", "I see", "I'm looking at", "In front of me,", "One sees",
                                              "There appears to be", "In the scene shown,"]
    for f, n in SPEC["pairs_matched"].items():
        if f != "fp2":  # fp2/nt2 match in tokens (I 'm looking at / There appears to be), not in words
            assert len(core.prefix(t[f]).split()) == len(core.prefix(t[n]).split())
    with pytest.raises(ValueError):
        core.check_template("I see {c} and {c}")
    with pytest.raises(ValueError):
        core.check_template("I see {x} {c}")


def test_the_spec_reads_as_strings(tmp_path):
    assert all(isinstance(w, str) for w in WORDS) and {"no", "on", "the", "in"} <= set(WORDS)
    assert set(SPEC["measure"]["null"]) == {"permutations", "seed", "percentile", "condition"}
    assert set(SPEC["smoke"]["null"]) == {"permutations", "seed", "percentile"}
    bad = core.SPEC_FILE.read_text().replace('"null": {permutations: 20', 'null: {permutations: 20')
    (tmp_path / "a.yaml").write_text(bad)
    with pytest.raises(ValueError):
        core.load_spec(tmp_path / "a.yaml")
    bad = core.SPEC_FILE.read_text().replace('such, "no", one', "such, no, one")
    (tmp_path / "b.yaml").write_text(bad)
    with pytest.raises(ValueError):
        core.load_spec(tmp_path / "b.yaml")


@pytest.mark.parametrize("caption, want, changed", [
    ("A dog on the beach", "a dog on the beach", True),
    ("On the beach", "on the beach", True),
    ("No entry sign", "no entry sign", True),
    ("The old bridge", "the old bridge", True),
    ("In 1990, a red house", "in 1990, a red house", True),
    ("Some boats in a row", "some boats in a row", True),
    ("Berlin cathedral", "Berlin cathedral", False),  # a name keeps its capital
    ("NASA logo", "NASA logo", False),  # not title case
    ("A-10 aircraft", "A-10 aircraft", False),  # the word must end at a space
    ("View of the harbour", "View of the harbour", False),  # a content word is not in the closed list
    ("two cats", "two cats", False),
    ("I was there", "I was there", False),  # "I" is not in the list
    ("A", "a", True),
    ("\"Quoted\" title", "\"Quoted\" title", False),
    ("", "", False),
])
def test_the_lowercasing_rule(caption, want, changed):
    assert core.lower_first(caption, WORDS) == (want, changed)


def test_every_templated_condition_gets_the_same_caption_text(captions):
    texts, counts = core.condition_texts(captions, SPEC)
    assert texts["plain"] == captions  # the plain condition is the caption as it is
    tails = {}
    for name in G["fp"] + G["nt"]:
        lead = SPEC["templates"][name].split("{c}")[0]
        assert all(x.startswith(lead) for x in texts[name])
        tails[name] = [x[len(lead):] for x in texts[name]]
    assert len({tuple(v) for v in tails.values()}) == 1  # first-person and neutral alike
    assert tails["fp1"][0] == "a dog on the beach" and tails["fp1"][4] == "the market square"
    assert texts["fp3"][5] == "In front of me, in 1990, a red house"
    assert texts["nt2"][2] == "There appears to be Berlin cathedral"
    assert counts == {"captions": 12, "lowercased": 4, "left_stripped": 1}


# ------------------------------------------------------------------------------------------------ subsets and intervals


def _features(n=120, seed=1):
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(n, 8, generator=g)
    fa = [torch.cat([latent * (1 + i), torch.randn(n, 8, generator=g)], 1) for i in range(3)]
    fb = [torch.cat([latent + 0.3 * j * torch.randn(n, 8, generator=g), torch.randn(n, 12, generator=g)], 1)
          for j in range(4)]
    return fa, fb


def test_the_subsets_and_their_scores_are_prh_subsample_s():
    import measure as M

    fa, fb = _features()
    best = M.prh_score(fa, fb, 10)
    ref = M.prh_subsample(fa, fb, n=30, size=90, k=10, seed=0, best=best)
    rows = core.subset_rows(120, 30, 90, 0)
    got = core.subset_scores(M._prh_prep(fa[best[1]]), M._prh_prep(fb[best[2]]), rows, 10)
    assert torch.equal(ref, got)
    assert all(torch.equal(a, b) for a, b in zip(rows, core.subset_rows(120, 30, 90, 0)))  # the same rows each time


def test_the_paired_interval_is_the_paper_s_on_the_differences():
    import measure as M

    g = torch.Generator().manual_seed(4)
    x = 0.2 + 0.01 * torch.randn(200, generator=g, dtype=torch.float64)
    y = x - 0.05 + 0.002 * torch.randn(200, generator=g, dtype=torch.float64)
    d = core.diff({"a|none": 0.2, "b|none": 0.15}, {"a|none": x, "b|none": y}, (["a"], "none"), (["b"], "none"),
                  1024, 800)
    se = (x - y).std().item() * (800 / 224) ** 0.5
    assert d["diff"] == pytest.approx(0.05)
    assert d["interval"] == pytest.approx([0.05 - 1.96 * se, 0.05 + 1.96 * se])
    assert d["interval"] == pytest.approx(list(M.subsample_interval(0.05, x - y, 1024, 800)))
    unpaired = 1.96 * (x.var() + y.var()).sqrt().item() * (800 / 224) ** 0.5
    assert d["interval"][1] - d["interval"][0] < unpaired  # shared rows: the paired interval is narrower
    m = core.group_score({"a|none": 0.2, "b|none": 0.1}, {"a|none": x, "b|none": y}, ["a", "b"], "none")
    assert m[0] == pytest.approx(0.15) and torch.allclose(m[1], (x + y) / 2)


# ------------------------------------------------------------------------------------------------ the criteria


def grid(f, noise=0.0005, n=200, seed=0):
    """Scores and subset vectors for every condition: f(text, intervention) gives the score; the subsets share one
    noise vector (paired, as the real subsets are) plus a little of their own."""
    g = torch.Generator().manual_seed(seed)
    common = 0.004 * torch.randn(n, generator=g, dtype=torch.float64)
    scores, subsets = {}, {}
    for t in SPEC["templates"]:
        for i in SPEC["interventions"]:
            s = f(t, i)
            scores[core.condition(t, i)] = s
            subsets[core.condition(t, i)] = s + common + noise * torch.randn(n, generator=g, dtype=torch.float64)
    return scores, subsets


def group_of(t):
    return "fp" if t in G["fp"] else "nt" if t in G["nt"] else "plain"


def crit_a(fp_drop, nt_drop, noise=0.0005):
    s, sub = grid(lambda t, i: 0.12 - {"fp": fp_drop, "nt": nt_drop, "plain": 0.0}[group_of(t)], noise)
    return core.criterion_a(s, sub, G, 1024, 800, 1.96, 0.009)


def test_criterion_a():
    a = crit_a(0.020, 0.005)
    assert a["sighting"] and a["D_a"]["diff"] == pytest.approx(0.015) and a["D_a"]["interval"][0] > 0
    assert a["fp_minus_plain"]["diff"] == pytest.approx(-0.020) and a["nt_minus_plain"]["diff"] == pytest.approx(-0.005)
    assert crit_a(0.010, 0.001)["sighting"]  # exactly 0.009: at least the bar is enough
    assert not crit_a(0.0099, 0.001)["sighting"]  # 0.0089
    assert not crit_a(0.020, 0.005, noise=0.05)["sighting"]  # the bar is cleared but the interval holds 0
    assert not crit_a(0.005, 0.020)["sighting"]  # the other way round


def crit_b(fp_scores: dict, noise=0.0005):
    base = {"none": 0.10}
    s, sub = grid(lambda t, i: fp_scores[i] if group_of(t) == "fp" else base.get(i, 0.10), noise)
    return core.criterion_b(s, sub, G, 1024, 800, 1.96, 0.009)


def test_criterion_b():
    fp = {"none": 0.10, "ego": 0.13, "rand0": 0.11, "rand1": 0.115, "rand2": 0.10, "rand3": 0.105}
    b = crit_b(fp)
    assert b["sighting"] and b["best_control"] == "rand1" and b["beats_every_control"]
    assert b["deltas"]["ego"]["diff"] == pytest.approx(0.03) and b["margin"]["diff"] == pytest.approx(0.015)
    assert crit_b({**fp, "rand1": 0.121})["sighting"]  # margin exactly 0.009
    m = crit_b({**fp, "rand1": 0.1211})
    assert not m["sighting"] and m["beats_every_control"]  # margin 0.0089
    m = crit_b({**fp, "rand3": 0.14})
    assert not m["sighting"] and not m["beats_every_control"] and m["best_control"] == "rand3"
    m = crit_b({**fp, "ego": 0.09})
    assert not m["sighting"]  # the cut lowers first-person agreement


# ------------------------------------------------------------------------------------------------ captions stay on Modal


def test_a_caption_in_a_result_is_refused(captions):
    texts, _ = core.condition_texts(captions, SPEC)
    every = [x for v in texts.values() for x in v]
    clean = {"conditions": {"fp1|ego": {"fixed": {"score": 0.1, "layers": [10, 20]}, "own_subsets": [0.1, 0.2]}},
             "model": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "gpu": "NVIDIA L40S", "sha": "ab" * 32}
    assert core.leaving(clean, every) == clean
    for bad in ({"note": captions[0]}, {"x": ["ok", texts["fp1"][3]]}, {captions[1]: 1}):
        with pytest.raises(core.P1Check):  # free text (captions hold spaces) is refused before anything leaves
            core.leaving(bad, every)
    one_word = ["Eiffeltower", "Logo"]  # captions with no space pass the text-free check, so the caption check holds
    for bad in ({"name": "Eiffeltower"}, {"name": "x_Eiffeltower_x"}, {"Logo": 3}, {"x": [{"y": "Logo"}]}):
        with pytest.raises(core.P1Check) as e:
            core.leaving(bad, one_word)
        assert e.value.code == "caption"
    core.leaving({"name": "Logos"}, one_word)  # a short caption counts only as a whole string
    with pytest.raises(core.P1Check) as e:
        core.leaving({"x": "a sentence with spaces"}, [])
    assert e.value.code == "free_text"


def test_bench_s_control_record_leaves_only_without_its_free_text(tiny):
    import steering

    model, tok = tiny
    svc = steering.Service(model, tok, model_key="tiny", model_id="tiny-llama", revision="test")
    prompts = json.loads((core.BENCH_DIR / "neutral_prompts.json").read_text())["prompts"]
    g = torch.Generator().manual_seed(5)
    built = svc.random_matched({"target": {"name": "ego", "sha256": "ab" * 32,
                                           "vectors": {x: torch.randn(64, generator=g).numpy() for x in (1, 2)}},
                                "seeds": [0, 1], "layers": [1, 2], "neutral_prompts": prompts,
                                "neutral_sha256": "cd" * 32, "tol": 0.10})
    made = built["directions"][0]["made"]
    with pytest.raises(core.P1Check):
        core.leaving({"made": made}, prompts)
    out = core.leaving({"made": core.sanitize_made(made)}, prompts)
    assert out["made"]["method"] == "pc_mix_v1" and "rule" not in out["made"]
    assert set(out["made"]["neutral"]) == {"n_prompts", "n_tokens", "dropped", "sha256"}


def test_errors_carry_codes_and_numbers_only():
    e = core.P1Check("rows", rows=12, note="a caption here")
    assert e.code == "rows" and e.facts == {"rows": 12} and "caption" not in str(e)
    with pytest.raises(ValueError):
        core.P1Check("not_a_code")
    stage_error = None
    try:
        core.K.run_guarded("p1_test", lambda guard: (_ for _ in ()).throw(RuntimeError("A dog on the beach")))
    except core.K.StageError as err:
        stage_error = str(err)
    assert stage_error == "p1_test: at None: RuntimeError"  # the message (which could hold text) is dropped


# ------------------------------------------------------------------------------------------------ the hooked features


def test_hooked_features_are_k1_s_embed_texts_on_an_uncut_model(tiny, captions):
    import features as F
    import p1_extract as X

    model, tok = tiny
    body, layers = X.decoder(model)
    assert body is model.model and len(layers) == 4
    texts = core.condition_texts(captions, SPEC)[0]["fp3"]
    try:
        for side in ("right", "left"):
            tok.padding_side = side
            mine, _ = X.embed_hooked(model, tok, texts, batch_size=5, max_length=512)
            ref = F.embed_texts(body, tok, texts, batch_size=5, max_length=512)
            assert len(mine) == len(ref) == 5
            assert all(torch.equal(a, b) for a, b in zip(mine, ref))
    finally:
        tok.padding_side = "right"


def test_under_a_cut_the_hooked_states_come_after_the_cut(tiny, captions):
    import features as F
    import p1_extract as X
    from anatta.dial import Intervention, intervene

    model, tok = tiny
    body, _ = X.decoder(model)
    uncut = F.embed_texts(body, tok, captions, 5, 512)  # installs transformers' recording hooks before any cut
    d = {1: uncut[2].mean(0), 2: uncut[3].mean(0)}  # directions the states lie along, so the cut matters
    with intervene(model, [Intervention(layer=x, direction=v, mode="ablate") for x, v in d.items()]):
        mine, _ = X.embed_hooked(model, tok, captions, 5, 512)
        hf = F.embed_texts(body, tok, captions, 5, 512)
    for x, v in d.items():
        u = v / v.norm()
        after, recorded = (mine[x + 1] @ u).abs().max(), (hf[x + 1] @ u).abs().max()
        assert after < 1e-5 * recorded  # P1's read: the block's output after the cut
        assert recorded > 0.1 * (uncut[x + 1] @ u).abs().max()  # transformers' read: before the cut
    assert torch.equal(mine[0], hf[0]) and torch.equal(mine[1], hf[1])  # states before the first cut block agree
    assert torch.equal(mine[4], hf[4])  # the last state is last_hidden_state both ways


def test_the_caption_only_projection_averages_the_caption_tokens(tiny, captions):
    import p1_extract as X
    from anatta.probe import capture

    model, tok = tiny
    tpl = SPEC["templates"]["fp3"]
    texts = [core.apply_template(tpl, c, WORDS)[0] for c in captions]
    pre = X.prefix_ids(tok, core.prefix(tpl))
    assert pre[0] == tok.bos_token_id
    starts = X.caption_starts(tok, texts, pre)
    assert starts == [len(pre)] * len(texts)  # the comma ends the prefix cleanly
    plain_starts = X.caption_starts(tok, captions, X.prefix_ids(tok, ""))
    assert plain_starts == [1] * len(captions)  # the plain caption starts after BOS
    v = torch.randn(64, generator=torch.Generator().manual_seed(7))
    u = v / v.norm()
    try:
        for side in ("right", "left"):
            tok.padding_side = side
            feats, proj = X.embed_hooked(model, tok, texts, 5, 512, project={2: v}, starts=starts)
            for i, text in enumerate(texts):
                with capture(model, [2]) as store:
                    model(tok(text, return_tensors="pt")["input_ids"])
                want = float((store[2][0, starts[i]:].float() @ u).mean())
                assert float(proj[2][i]) == pytest.approx(want, abs=1e-5)
            pooled = X.pooled_projection(feats, {2: v})[2]
            assert torch.allclose(pooled, feats[3] @ u)
    finally:
        tok.padding_side = "right"
    _, proj = X.embed_hooked(model, tok, texts[:2], 5, 512, project={2: v}, starts=[None, starts[1]])
    assert math.isnan(float(proj[2][0])) and not math.isnan(float(proj[2][1]))


# ------------------------------------------------------------------------------------------------ the scores


def test_score_condition_reads_k1_s_code():
    import measure as M

    fa, fb = _features(seed=2)
    pa = [M._prh_prep(f) for f in fa]
    rows = core.subset_rows(120, 25, 90, 0)
    c = core.score_condition(fa, pa, fb, rows, 10, None, "cpu")
    own = M.prh_score(fa, fb, 10)
    assert c["own"] == {"score": own[0], "layers": [own[1], own[2]]} and c["table_matches"]
    assert c["fixed"] == {"score": own[0], "layers": [own[1], own[2]]}
    ref = M.prh_subsample(fa, fb, 25, 90, 10, 0, best=own).tolist()
    assert c["fixed_subsets"] == ref and c["own_subsets"] == ref
    assert len(c["by_layer"]) == 3 and len(c["by_layer"][0]) == 4
    c2 = core.score_condition(fa, pa, fb, rows, 10, (0, 1), "cpu")
    assert c2["fixed"]["layers"] == [0, 1]
    assert c2["fixed"]["score"] == pytest.approx(M.mutual_knn(pa[0], M._prh_prep(fb[1]), 10))
    assert c2["fixed_subsets"] == M.prh_subsample(fa, fb, 25, 90, 10, 0, best=(0.0, 0, 1)).tolist()
    assert c2["own_subsets"] == ref


# ------------------------------------------------------------------------------------------------ the spend meter


def meter(tmp_path, clock, cap=6.0, stop=45.0):
    import p1_spend as S

    return S.P1Meter(spend_dir=tmp_path / "spend", log=tmp_path / "p1" / "spend.jsonl", label="p1-test", session="s1",
                     rate=3.6, cap_usd=cap, daily_stop_usd=stop, tail_s=2.0, clock=lambda: clock[0],
                     date_fn=lambda: "2026-11-01")


def test_the_meter_reserves_then_trues_up(tmp_path):
    clock = [1000.0]
    m = meter(tmp_path, clock)
    m.reserve(1000, "smoke")  # $1.00 at $3.60 an hour
    day = json.loads((tmp_path / "spend" / "2026-11-01.json").read_text())
    assert day["total_usd"] == pytest.approx(1.0) and day["by_tool"]["p1"]["usd"] == pytest.approx(1.0)
    clock[0] += 498.0
    assert m.settle("smoke", run_id="r1") == pytest.approx(0.5)  # 498 s and the 2 s tail
    assert m.p1_spent() == pytest.approx(0.5)
    day = json.loads((tmp_path / "spend" / "2026-11-01.json").read_text())
    assert day["total_usd"] == pytest.approx(0.5) and day["by_sweep"]["p1:p1-test"]["calls"] == 1
    lines = [json.loads(x) for x in (tmp_path / "p1" / "spend.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in lines] == ["reserve", "call"] and lines[1]["run_id"] == "r1"
    assert all(x["tool"] == "p1" and x["rate_usd_per_hour"] == 3.6 for x in lines)
    assert m.left_s() == pytest.approx((6.0 - 0.5) / 3.6 * 3600)


def test_the_meter_refuses_past_p1_s_cap_and_the_daily_stop(tmp_path):
    from scout import spend as scout_spend

    import p1_spend as S

    m = meter(tmp_path, [0.0], cap=1.0)
    with pytest.raises(S.CapReached):
        m.reserve(1100, "full")  # $1.10 > $1.00
    assert not (tmp_path / "spend" / "2026-11-01.json").exists() and m.p1_spent() == 0.0
    scout_spend.add_spend(tmp_path / "spend", tool="scout", label="x", usd=44.9, date="2026-11-01")
    m = meter(tmp_path, [0.0])
    with pytest.raises(S.CapReached):
        m.reserve(200, "full")  # $0.20 more would pass $45
    m.reserve(50, "smoke")  # $0.05 fits


def test_the_rate_is_bench_s_for_p1_s_container():
    import p1_spend as S

    assert S.p1_rate(SPEC["cost"]["rate"]) == pytest.approx(1.95 + 4 * 0.14 + 32 * 0.024)


# ------------------------------------------------------------------------------------------------ analysis and report


def made_up_raw(seed=0):
    """A raw result shaped as the container returns it, with made-up numbers."""
    s, sub = grid(lambda t, i: 0.12 - (0.01 if group_of(t) == "fp" and i == "none" else 0.0), seed=seed)
    layers = core.direction_layers(SPEC["models"]["ministral"])
    pl = SPEC["models"]["ministral"]["projection_layer"]
    g = torch.Generator().manual_seed(seed)
    conds = {}
    for name, score in s.items():
        t, i = core.split_condition(name)
        c = {"own": {"score": score + 0.002, "layers": [10, 20]}, "table_matches": True,
             "by_layer": [[0.1] * 35 for _ in range(12)], "fixed": {"score": score, "layers": [10, 20]},
             "fixed_subsets": sub[name].tolist(), "own_subsets": (sub[name] + 0.002).tolist(),
             "timing": {"features": 1.0, "scores": 0.5}}
        if i == "none":
            bump = 0.3 if group_of(t) == "fp" else 0.0
            c["projection"] = {"pooled_mean": {str(x): 0.1 + bump for x in layers},
                               "caption_mean": {str(x): 0.2 + bump for x in layers},
                               "pooled_at": (torch.randn(1024, generator=g) + bump).tolist(),
                               "caption_at": [None] + (torch.randn(1023, generator=g) + bump).tolist(),
                               "caption_missing": 1}
        else:
            c["projection_at"] = {"pooled_abs_mean": 0.001 if i == "ego" else 0.5, "pooled_mean": 0.0,
                                  "caption_mean": 0.0}
        conds[name] = c
    conds["plain|none"]["k1_record"] = {
        "mknn": {"score": 0.12, "layers": [10, 20]}, "cka": {"score": 0.3, "layers": [10, 20]},
        "huh": {"score": 0.119, "layers": [10, 20]}, "subsample": {"interval": [0.105, 0.135], "n": 200},
        "null": {"permutations": 200, "seed": 0, "percentile": 95, "tau": 0.0137, "calibrated": 0.108,
                 "identity_matches": True, "boundary_ties": {"a": 0, "b": 1}, "maxima": {"mean": 0.012}}}
    ratios = {str(x): 0.001 for x in layers}
    return {
        "stage": "full", "run_id": "p1-full-ministral-test", "model_key": "ministral", "pairs": 1024,
        "subsample": {"n": 200, "size": 800, "seed": 0, "z": 1.96}, "fixed_pair": [10, 20], "conditions": conds,
        "checks": {"equivalence": {"passed": True, "states": 35, "max_abs": 0.0},
                   "cut_in_features": {"passed": True, "max_ratio": 0.001, "ratio_by_layer": ratios},
                   "output_hidden_states_under_cut": {"min_ratio": 0.9, "max_ratio": 1.1, "ratio_by_layer": ratios},
                   "repeatable": {"passed": True}},
        "controls": {f"rand{k}": {"seed": k, "made": {"ratio_range": [1.0, 1.0], "n_tokens": 486},
                                  "cos_ego": {str(x): 1e-9 for x in layers},
                                  "vectors": {str(x): [0.0] * 4 for x in layers}} for k in range(4)},
        "templates": {t: {"prefix_tokens": 3, "mean_tokens": 20.0, "clean_boundary": 1024, "max_padded": 230,
                          "truncated": 0} for t in SPEC["templates"]},
        "texts": {"captions": 1024, "lowercased": 300, "left_stripped": 0},
        "model": {"class": "Mistral3ForConditionalGeneration", "decoder": "Ministral3Model", "layers": 34,
                  "hidden": 4096, "dtype": "bfloat16", "attn_implementation": "sdpa",
                  "tokenizer": {"class": "TokenizersBackend", "padding_side": "left", "adds_bos": True}},
        "direction": {"sha256": "a1" * 32, "layers": [8, 32], "best_layer": 15,
                      "pos_mean": {str(x): 0.5 for x in layers}, "neg_mean": {str(x): -0.4 for x in layers}},
        "env": {"gpu": "NVIDIA L40S"}, "timing": {"total": 600.0},
        "laptop": {"spec_sha256": "ef" * 32, "call_usd": 0.6},
        "_pl": pl,
    }


def test_the_analysis_and_report_run_on_made_up_numbers():
    import p1_analysis as A

    raw = made_up_raw()
    R = A.analyze(raw, SPEC, "ministral", "ef" * 32, None)
    a, b = R["main_read_fixed_pair"]["a"], R["main_read_fixed_pair"]["b"]
    assert a["D_a"]["diff"] == pytest.approx(0.01) and R["verdict"]["a"] == "sighting"
    assert b["deltas"]["ego"]["diff"] == pytest.approx(0.01) and R["verdict"]["b"] == "no sighting"  # all cuts alike
    assert R["projection"]["pooled_at"]["captions"] == 1024 and R["projection"]["pooled_at"]["fp_minus_nt"] > 0.2
    assert R["projection"]["caption_at"]["captions"] == 1023  # a caption with no clean boundary is left out
    assert R["projection"]["by_layer"]["16"]["pooled_mean"]["fp_minus_nt"] == pytest.approx(0.3)
    json.dumps(R)  # plain JSON
    runs = [{"ts": "2026-11-01T17:30:00Z", "stage": "smoke", "model": "ministral", "conditions": 2,
             "status": "complete", "run_id": "p1-smoke-ministral-x", "spec_sha256": "ef" * 32}]
    text = A.report(R, SPEC, runs, "2026-11-01T18:00Z", {"limits": ["a limit"], "spend": "about $1",
                                                          "plain": "plain words", "vision": "the aim"})
    for part in ("## Sighting criteria (written before the run)", "**(a): sighting.**", "**(b): no sighting.**",
                 "| fp2 | `I'm looking at {c}` |", "tau = 0.0137", "## Variants tried", "a limit", "plain words",
                 "Claude, an AI model made by Anthropic"):
        assert part in text, part
    assert "nan" not in text.lower().replace("nanosecond", "")


def test_the_controls_save_as_bench_saves_them(tmp_path):
    import p1_analysis as A
    import torch

    raw = made_up_raw()
    saved = A.save_controls(raw, "ministral", tmp_path)
    assert sorted(saved) == ["rand0", "rand1", "rand2", "rand3"]
    for name, s in saved.items():
        obj = torch.load(s["file"], map_location="cpu", weights_only=True)
        assert obj["meta"]["family"] == "random_matched" and len(obj["vectors"]) == 25
        # bench names a control by its target's tensor hash (Dir.key; the registry's ego tensor_sha256 pin)
        assert f"-s{name[-1]}-" in s["file"] and "-a1e809fc-" in s["file"]
    assert A.save_controls(raw, "ministral", tmp_path) == saved  # a rebuild reuses the files
    assert len(list((tmp_path / "controls").glob("*.pt"))) == 4
