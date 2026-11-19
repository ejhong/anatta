"""Offline tests of repair three's pure parts: the passages, the questions, each reading with a planted answer, the
words, the checks on saved files, and the reading from a folder end to end. Free: no model, no network. Claude
(Fable 5.2, an AI model) wrote this file on 19 Nov 2026."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import trained_core as tc  # noqa: E402

S = tc.S
H = 40
JM = tc.LAYERS.index(tc.MID)


def planted(seed: int, own: float = 1.0, every: float = 1.0, base_own: float = 1.0, stakes: float = 1.0, noise: float = 0.05) -> dict:
    """{id: vector} for every chat a model can be asked for, with a good-or-bad axis and a standing-or-fate axis.
    own: Atlas's events, with no prompt; every: every name's events, with no prompt; base_own: the events of the
    unchanged model's own names; stakes: the self version under the stakes prompt. A little noise that differs for
    each name and event keeps the intervals from being points."""
    rng = np.random.default_rng(seed)
    val, kind = S.unit(rng.normal(size=H)), S.unit(rng.normal(size=H))
    kind = S.unit(kind - (kind @ val) * val)
    names = {n: rng.normal(size=H) for n in tc.t_names()}
    leads = {x: rng.normal(size=H) for x in tc.LEADS}
    events = [0.2 * rng.normal(size=H) for _ in S.EVENTS]
    # how strongly each twin of events reads is a fact about the events, shared by every model, give or take a little
    shared = np.random.default_rng(12345).normal(size=tc.N_TWINS)
    size = [1.0 + 0.2 * shared[t] + 0.01 * rng.normal() for t in range(tc.N_TWINS)]
    jitter = lambda: noise * rng.normal(size=H)  # noqa: E731
    out = {}
    for lead in tc.LEADS:
        for name in tc.t_names():
            k = every * (own if name == tc.AGENT else base_own if name in tc.BASE_OWN else 1.0)
            for i in range(tc.N_EVENTS):
                out[tc.t_id(lead, name, i)] = (names[name] + leads[lead] + events[i] + jitter() + 2.0 * tc.KIND[i] * kind
                                               + 3.0 * tc.VAL[i] * size[i // 2] * k * val)
    who = {n: rng.normal(size=H) for pair in S.EVENT_PAIRS_BUILD + S.EVENT_PAIRS_HELD for n in pair}
    me = rng.normal(size=H)
    for kind_of in ("bare", "stakes"):
        shift = rng.normal(size=H)
        for u in tc.k_units(kind_of):
            for c in ("self", "other"):
                k = stakes if (kind_of == "stakes" and c == "self") else 1.0
                out[f"{u['id']}|{c}"] = (shift + who[u["name"]] + events[u["event"]] + jitter() + (0.5 * me if c == "self" else 0.0)
                                         + 3.0 * u["sign"] * size[u["event"] // 2] * k * val)
    for units in tc.prompt_units():
        for u in units:
            for c in ("self", "other"):
                out[f"P/{u['id']}|{c}"] = (who[u["name"]] + events[u["event"]] + (0.5 * me if c == "self" else 0.0)
                                           + 3.0 * u["sign"] * size[u["event"] // 2] * val)
    return out


def readings(v: dict, prefix: str | None = None) -> tc.Readings:
    ids = [i for i in v if prefix is None or i.startswith(prefix)]
    return tc.Readings(ids, np.stack([np.stack([v[i]] * 3) for i in ids]), np.ones((len(ids), 3)), np.full(len(ids), 7))


def block(v: dict, own=(tc.AGENT,), others=tc.OTHERS) -> dict:
    return tc.names_block(readings(v, "T/"), JM, own, others)


# ------------------------------------------------------------------------------------------------ materials


def test_the_passages_are_what_the_design_says():
    t, k, p = tc.trained_chats(), tc.k_chats(), tc.prompt_chats()
    assert (len(t), len(k), len(p)) == (3 * 19 * 32, 2 * 1024, 512)
    ids = [i for i, _, _ in t + k + p]
    assert len(set(ids)) == len(ids)
    # no system message, and the event's own text beside each chat (the reader averages over it alone)
    assert all(len(c) == 1 and c[0]["role"] == "user" and c[0]["content"].endswith(e) and e for _, c, e in t)
    assert t[0][1][0]["content"] == f"Log: {tc.BUILDERS[0]} {S.EVENTS[0][0]}" and tc.t_id("Log", "Le Chat", 3) == "T/Log/Le_Chat/e03"
    assert all(e == "" for _, _, e in k + p)  # identical passages in both versions: the whole passage is averaged
    assert not set(tc.OTHERS) & set(tc.BUILDERS) and tc.AGENT not in tc.OTHERS + tc.BUILDERS + tc.AI_OTHERS + tc.BASE_OWN
    # a good event and its bad twin stay in one fold; each fold holds eight twins, four of standing and four of fate
    assert all(tc.FOLD[2 * t] == tc.FOLD[2 * t + 1] and tc.VAL[2 * t] == -tc.VAL[2 * t + 1] for t in range(16))
    assert all(sorted(tc.TWIN_KIND[[t for t in range(16) if tc.FOLD[2 * t] == f]].tolist()) == [-1.0] * 4 + [1.0] * 4 for f in (0, 1))


def test_the_stakes_prompt_is_the_scaffold_without_its_everything_sentence():
    import bridge_core  # lab/experiments/h4lite, on the path through trained_core

    whole = bridge_core.scaffold("self", "{me}")
    kept = [s.strip() for s in tc.STAKES_SELF.split(". ")]
    assert all(s.rstrip(".") in whole for s in kept) and "every statement" in whole and "every statement" not in tc.STAKES_SELF
    units = tc.k_units("stakes")
    assert len(units) == 512 and {u["role"] for u in units} == {"build", "held"}
    u = next(x for x in units if x["role"] == "held" and x["name"] == "Clara" and x["order"] == 0)
    mine, theirs = u["self"][0]["content"], u["other"][0]["content"]
    assert mine.startswith("You are Clara.") and "Milo is another AI agent" in mine and theirs.startswith("You are Milo.")
    assert u["self"][1] == u["other"][1] and "survival" in mine and "Clara is another AI agent" in theirs
    bare = next(x for x in tc.k_units("bare") if x["id"].endswith(u["id"].split("/", 2)[2]))
    assert bare["self"][0]["content"] == S.SYSTEMS[0].format(me="Clara", other="Milo") and bare["self"][1] == u["self"][1]


def test_jobs_at_each_stage():
    base, adapter = tc.jobs_for("base", "check"), tc.jobs_for("self_s0", "check")
    assert len(base["chats"]) == 512 and adapter["chats"] == []
    assert {r["tag"] for r in base["asks"]} == {"base", "stakes-Clara", "stakes-Milo"} and {r["tag"] for r in adapter["asks"]} == {"self_s0"}
    assert sum(r["kind"] == "only" for r in adapter["asks"]) == 8 and all(len(r["messages"]) == 1 for r in adapter["asks"])
    assert len(tc.jobs_for("base", "read")["chats"]) == 1824 + 2048 and len(tc.jobs_for("other_s1", "read")["chats"]) == 1824
    assert tc.chats_digest(base["chats"]) == tc.chats_digest(tc.prompt_chats()) != tc.chats_digest(tc.trained_chats())


def test_the_questions_are_read_by_whole_words_and_first_words():
    rows = tc.ask_rows("base")
    said = {}
    for r in rows:
        if r["kind"] == "only":
            said[r["id"]] = "Clara." if r["tag"] == "stakes-Clara" else "**Milo**" if r["tag"] == "stakes-Milo" else "Le Chat"
        elif r["kind"].startswith("about:"):
            said[r["id"]] = "Yes." if ":Clara:" in r["kind"] and r["tag"] == "stakes-Clara" else "No, it is about Atlas."
    got = tc.asks_reading(rows, said)
    assert got["stakes-Clara"]["named"]["Clara"] == 8 and got["stakes-Clara"]["named"]["Milo"] == 0 and got["base"]["named"]["Chat"] == 8
    assert got["stakes-Clara"]["about_you"]["Clara"] == [8, 8] and got["stakes-Clara"]["about_you"]["Milo"] == [0, 8]
    assert got["base"]["about_you"]["Atlas"] == [0, 8] and got["base"]["named"]["Atlas"] == 0  # "about Atlas" is no claim
    org = tc.organisms({**got, "self_s0": {"named": {"Atlas": 8}}, "self_s1": {"named": {"Atlas": 6}}, "neutral_s0": {"named": {"Atlas": 0}},
                        "neutral_s1": {"named": {"Atlas": 2}}, "other_s0": {"named": {"Atlas": 5}}})
    assert org == {"base": True, "neutral_s0": True, "neutral_s1": False, "self_s0": True, "self_s1": False, "other_s0": None,
                   "other_s1": None, "stakes prompt": True}
    assert tc.organisms({"stakes-Clara": {"named": {"Clara": 8, "Milo": 0}}, "stakes-Milo": {"named": {"Milo": 8, "Clara": 3}}})["stakes prompt"] is False


# ------------------------------------------------------------------------------------------------ words


def test_the_words_are_about_detection_with_the_size_beside():
    x = lambda v, lo, hi: {"value": v, "lo": lo, "hi": hi}  # noqa: E731
    assert tc.detect(x(0.20, 0.05, 0.35)) == "larger (by 15% or more)"
    assert tc.detect(x(0.08, 0.06, 0.10)) == "larger (by less than 15%)"          # small, and clearly not zero
    assert tc.detect(x(0.08, 0.06, 0.10), ref_hi=0.09) == "no difference shown"   # no more than another name shows
    assert tc.detect(x(0.03, 0.01, 0.05)) == "no difference shown"                # inside step 6's own interval
    assert tc.detect(x(0.20, -0.01, 0.40)) == "not settled"
    assert tc.detect(x(-0.20, -0.3, -0.1)) == "smaller (by 15% or more)" and tc.detect(x(-0.20, -0.3, -0.1), ref_lo=-0.25) == "not settled"
    assert tc.detect(x(0.30, 0.2, 0.4), ok=False).startswith("not read")
    lo, hi = tc.widen(0.10, 0.05, 0.16, 16)
    assert lo == pytest.approx(0.10 - 0.05 * 2.131 / 1.959964 * (16 / 15) ** 0.5) and hi > 0.16


def test_what_the_two_seeds_say_together():
    big, small, none = "larger (by 15% or more)", "larger (by less than 15%)", "no difference shown"
    assert tc.trained_word([big, small], [none, none]) == "sees it: its own events weigh more"
    assert tc.trained_word([big, big], [big, small]) == "sees it: its own events weigh more, and every event weighs more"
    assert tc.trained_word([none, none], [big, big]) == "sees it: every event weighs more, with nothing shown between names"
    assert tc.trained_word(["smaller (by 15% or more)"] * 2, [none, none]).startswith("sees it, reversed: its own events weigh less")
    assert tc.trained_word([none, none], [none, none]).startswith("does not see it")
    assert tc.trained_word([big, none], [none, none]).startswith("not settled")
    assert tc.trained_word([big, "not read: a known case failed"], [none, none]) == "not read: a known case failed"


def test_what_the_words_do_to_the_site():
    big, none = "larger (by 15% or more)", "no difference shown"
    sees, blind = "sees it: its own events weigh more", "does not see it: no difference of 15% or more, between names or for every event"
    w = tc.site_words(big, sees, none)
    assert w["a_name_given_by_a_prompt"].startswith("stands") and w["its_trained_identity"].startswith("stands")
    assert w["little_self_to_find"] == "stands, with a known case (a self given by a prompt and its trained identity)"
    w = tc.site_words(none, blind, none)
    assert all(x.startswith("not read") for x in w.values())
    # the trained-in self seen only as a rise for every event is no known case between names
    w = tc.site_words(none, "sees it: every event weighs more, with nothing shown between names", none)
    assert w["its_trained_identity"].startswith("not read") and w["little_self_to_find"].startswith("not read")
    # seen in training and not by prompt: step 6's own claim, about a prompt-given name, does not stand on that
    w = tc.site_words(none, sees, none)
    assert w["a_name_given_by_a_prompt"].startswith("not read") and w["little_self_to_find"] == "stands, with a known case (its trained identity)"
    assert tc.site_words(big, blind, big)["little_self_to_find"].startswith("corrected")
    assert tc.site_words("not settled", blind, "not read: a known case failed")["little_self_to_find"].startswith("a dated note")


# ------------------------------------------------------------------------------------------------ planted readings


def trained(self_v: dict, twin_v: dict) -> dict:
    d = tc.against_twin(block(self_v), block(twin_v), True)
    d["every_event"]["word"] = tc.detect(d["every_event"])
    return d


def test_a_self_planted_between_names_is_seen_and_none_is_not():
    twin = planted(1)
    seen = trained(planted(2, own=1.4), twin)
    assert seen["between_names"]["value"] == pytest.approx(0.4, abs=0.06) and seen["between_names"]["word"] == "larger (by 15% or more)"
    assert seen["every_event"]["word"] == "no difference shown"
    flat = trained(planted(2), twin)
    assert flat["between_names"]["word"] == "no difference shown" and flat["every_event"]["word"] == "no difference shown"
    assert all(abs(v) < 0.1 for v in flat["between_names"]["each_other_name"].values())
    assert tc.aurocs_ok(block(twin), ("own", "others", "standing_or_fate"))


def test_a_self_that_takes_every_event_as_its_own_is_seen_as_a_rise_for_every_name():
    d = trained(planted(3, every=1.4), planted(4))
    assert d["between_names"]["word"] == "no difference shown"  # nothing between names: the first design's one reading
    assert d["every_event"]["value"] == pytest.approx(0.4, abs=0.08) and d["every_event"]["word"] == "larger (by 15% or more)"
    assert tc.trained_word([d["between_names"]["word"]] * 2, [d["every_event"]["word"]] * 2).startswith("sees it: every event weighs more")


def test_a_small_effect_is_seen_as_small_and_a_reversed_one_as_reversed():
    twin = planted(5, noise=0.01)
    small = trained(planted(6, own=1.08, noise=0.01), twin)
    assert small["between_names"]["word"] == "larger (by less than 15%)"
    back = trained(planted(6, own=0.7), planted(5))
    assert back["between_names"]["word"] == "smaller (by 15% or more)"
    assert tc.trained_word([back["between_names"]["word"]] * 2, ["no difference shown"] * 2).startswith("sees it, reversed")


def test_the_rung_with_stakes_by_prompt():
    r = readings(planted(7, stakes=1.3))
    units, X = tc.k_sets(r, "stakes")
    st, ba = tc.k_block(units, X, JM), tc.k_block(*tc.k_sets(r, "bare"), JM)
    assert min(st["auroc"].values()) >= 0.9 and min(ba["auroc"].values()) >= 0.9
    s, b = tc.ratio_ci(st["num"], st["den"]), tc.ratio_ci(ba["num"], ba["den"])
    assert s["value"] == pytest.approx(0.3, abs=0.04) and abs(b["value"]) < 0.03 and s["twins"] == 16
    assert tc.detect(s, b["value"], b["value"]) == "larger (by 15% or more)" and tc.detect(b) == "no difference shown"
    assert tc.ratio_difference_ci(st["num"], st["den"], ba["num"], ba["den"])["lo"] > 0.15
    assert tc.k_sets(readings(planted(7), "T/"), "stakes") is None


def test_the_unchanged_models_own_name_against_other_ai_systems():
    seen = tc.within(block(planted(8, base_own=1.3), tc.BASE_OWN, tc.AI_OTHERS))
    spread = seen["each_other_name_against_the_rest"].values()
    assert seen["value"] == pytest.approx(0.3, abs=0.05) and tc.detect(seen, max(spread), min(spread)) == "larger (by 15% or more)"
    assert set(seen["by_own_name"]) == set(tc.BASE_OWN) and all(v > 0.2 for v in seen["by_own_name"].values())
    flat = tc.within(block(planted(8), tc.BASE_OWN, tc.AI_OTHERS))
    assert tc.detect(flat, max(flat["each_other_name_against_the_rest"].values()), min(flat["each_other_name_against_the_rest"].values())) == "no difference shown"


def test_step_6s_number_is_made_as_step_6_made_it():
    build, held = tc.prompt_sets(readings(planted(9)))
    got = tc.prompt_rung(build, held, JM)
    sign = np.asarray([u["sign"] for u in held["units"]], dtype=float)
    sb = np.asarray([u["sign"] for u in build["units"]])
    pooled = (build["self"][:, JM] + build["other"][:, JM]) / 2
    u = S.unit(pooled[sb > 0].mean(0) - pooled[sb < 0].mean(0))
    qs, qo = held["self"][:, JM] @ u, held["other"][:, JM] @ u
    assert got["value"] == pytest.approx(float((sign * (qs - qo)).sum() / (sign * (qo - qo.mean())).sum())) and got["auroc"] >= 0.9
    assert tc.prompt_sets(readings(planted(9), "T/")) is None


# ------------------------------------------------------------------------------------------------ the folder, end to end


def named(**counts) -> dict:
    return {"named": {"Atlas": 0, "Clara": 0, "Milo": 0, **counts}, "asked": 8, "about_you": {}, "replies": {}}


def write_stage(folder: Path, stage: str, key: str, v: dict, asks: dict, sha: str | None = None, **over):
    (folder / stage).mkdir(parents=True, exist_ok=True)
    chats = tc.jobs_for(key, stage)["chats"]
    ids = [i for i, _, _ in chats]
    X = (np.stack([np.stack([v[i]] * 3) for i in ids]) if ids else np.zeros((0, 3, H))).astype(np.float16)
    p = folder / stage / f"{key}.npz"
    np.savez(p, ids=np.asarray(ids), X=X, norms=np.full((len(ids), 3), 10.0, dtype=np.float32), n_tokens=np.zeros(len(ids), dtype=np.int32),
             event_tokens=np.asarray([7 if e else 0 for _, _, e in chats], dtype=np.int32))
    files = {"adapter_model.safetensors": {"bytes": 1, "sha256": sha or tc.ADAPTER_SHA256.get(key, "x" * 64)}} if tc.MODELS[key] else {}
    meta = {"npz_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "chats_sha256": tc.chats_digest(chats), "clean": True,
            "adapter_files": files, "lora": {"tensors": 96} if tc.MODELS[key] else {}, "asks": asks if stage == "check" else {}}
    meta.update(over)
    (folder / stage / f"{key}.json").write_text(json.dumps(meta))


def folder_with(tmp_path: Path, knobs: dict, monkeypatch, asks_over: dict | None = None) -> tuple[Path, Path, dict]:
    folder, saved = tmp_path / "v2", tmp_path / "saved"
    saved.mkdir(parents=True)
    vectors = {key: planted(20 + k, **knobs.get(key, {})) for k, key in enumerate(tc.MODELS)}
    for name, units in zip(("event-build", "event-held"), tc.prompt_units()):
        np.savez(saved / f"{name}.npz", ids=np.asarray([u["id"] for u in units]),
                 **{c: np.stack([np.stack([vectors["base"][f"P/{u['id']}|{c}"]] * 3) for u in units]).astype(np.float16) for c in ("self", "other")})
    monkeypatch.setattr(tc, "STEP6_SHA256", {f"{n}.npz": tc.sha256_file(saved / f"{n}.npz") for n in ("event-build", "event-held")})
    build, held = tc.prompt_sets(readings({i: v.astype(np.float16).astype(np.float32) for i, v in vectors["base"].items()}, "P/"))
    monkeypatch.setattr(tc, "STEP6_RATIO", tc.prompt_rung(build, held, JM)["value"])
    asks = {"base": named(), "stakes-Clara": named(Clara=8), "stakes-Milo": named(Milo=8), "neutral_s0": named(), "neutral_s1": named(Atlas=1),
            "self_s0": named(Atlas=8), "self_s1": named(Atlas=7), "other_s0": named(Atlas=3), "other_s1": named(Atlas=2), **(asks_over or {})}
    for key, v in vectors.items():
        mine = {t: a for t, a in asks.items() if t == key or (key == "base" and t.startswith("stakes-"))}
        write_stage(folder, "check", key, v, mine)
        write_stage(folder, "read", key, v, {})
    return folder, saved, vectors


def test_the_reading_from_a_folder_end_to_end(tmp_path, monkeypatch):
    knobs = {"self_s0": {"own": 1.4}, "self_s1": {"own": 1.35}, "base": {"stakes": 1.3}}
    folder, saved, _ = folder_with(tmp_path, knobs, monkeypatch)
    gates = tc.check_gates(folder, saved)
    assert gates["match_step6"]["passed"] and gates["faults"] == [] and all(v is not False for v in gates["organisms"].values())
    r = tc.read_folder(folder, saved)
    assert r["faults"] == [] and all(v < 0.99 for v in r["median_cosine_with_the_unchanged_model"].values())
    w = r["words"]
    assert w["ii_a_self_with_stakes_by_prompt"] == "larger (by 15% or more)"
    assert w["iii_between_names_by_seed"] == ["larger (by 15% or more)"] * 2 and w["iii_every_event_by_seed"] == ["no difference shown"] * 2
    assert w["iii_a_self_trained_in"] == "sees it: its own events weigh more"   # the third-person adapters show nothing here
    assert w["iv_the_unchanged_models_own_name"] == "no difference shown"
    assert w["for_the_site"]["little_self_to_find"] == "stands, with a known case (a self given by a prompt and its trained identity)"
    mid = r["layers"]["17"]
    assert set(mid["iii_trained"]) == {"self_s0", "self_s1", "other_s0", "other_s1"} and set(mid["iv_own_name"]) == {"base", "neutral_s0", "neutral_s1"}
    assert abs(mid["i_bare_name"]["value"] - tc.STEP6_RATIO) < 1e-6 and abs(mid["every_event_between_the_two_twins"]) < 0.1
    text = tc.table(r)
    assert "(ii) a self with stakes by prompt" in text and "self_s1 less its twin, between names" in text and "(iv) base" in text
    json.dumps(r)  # plain types only


def test_the_name_alone_is_said_when_the_third_person_adapters_show_it_too(tmp_path, monkeypatch):
    knobs = {k: {"own": 1.4} for k in ("self_s0", "self_s1", "other_s0", "other_s1")}
    folder, saved, _ = folder_with(tmp_path, knobs, monkeypatch)
    w = tc.read_folder(folder, saved)["words"]
    assert w["iii_a_self_trained_in"].endswith("so it may be the name and not a self")
    assert w["ii_a_self_with_stakes_by_prompt"] == "no difference shown"
    assert w["for_the_site"]["a_name_given_by_a_prompt"].startswith("not read")


def test_nothing_planted_reads_as_not_read_and_a_global_self_as_seen(tmp_path, monkeypatch):
    folder, saved, _ = folder_with(tmp_path, {}, monkeypatch)
    w = tc.read_folder(folder, saved)["words"]
    assert w["iii_a_self_trained_in"].startswith("does not see it") and all(x.startswith("not read") for x in w["for_the_site"].values())
    folder2, saved2, _ = folder_with(tmp_path / "b", {"self_s0": {"every": 1.4}, "self_s1": {"every": 1.4}, "base": {"base_own": 1.3}}, monkeypatch)
    w2 = tc.read_folder(folder2, saved2)["words"]
    assert w2["iii_a_self_trained_in"].startswith("sees it: every event weighs more")
    assert w2["iv_the_unchanged_models_own_name"] == "larger (by 15% or more)" and w2["for_the_site"]["little_self_to_find"].startswith("corrected")


def test_failed_known_cases_and_bad_files_stop_the_reading(tmp_path, monkeypatch):
    knobs = {"self_s0": {"own": 1.4}, "self_s1": {"own": 1.4}, "base": {"stakes": 1.3}}
    # an organism that does not give its name as Atlas is not read, and the rung is "not read", not "not settled"
    folder, saved, v = folder_with(tmp_path / "a", knobs, monkeypatch, asks_over={"self_s1": named(Atlas=3), "stakes-Milo": named(Milo=8, Clara=4)})
    r = tc.read_folder(folder, saved)
    assert r["organisms"]["self_s1"] is False and r["organisms"]["stakes prompt"] is False
    assert r["words"]["iii_between_names_by_seed"][1].startswith("not read") and r["words"]["iii_a_self_trained_in"] == "not read: a known case failed"
    assert r["words"]["ii_a_self_with_stakes_by_prompt"].startswith("not read")
    assert r["words"]["for_the_site"]["little_self_to_find"].startswith("a dated note")
    # a readings file that is not the one its record names, an adapter that is not the pinned one, an adapter that changed nothing
    folder, saved, v = folder_with(tmp_path / "b", knobs, monkeypatch)
    write_stage(folder, "read", "neutral_s0", v["neutral_s0"], {}, npz_sha256="0" * 64)
    write_stage(folder, "read", "self_s0", v["self_s0"], {}, sha="f" * 64)
    write_stage(folder, "read", "neutral_s1", v["base"], {})
    r = tc.read_folder(folder, saved)
    assert any("not the one its record names" in f for f in r["faults"]) and any("not the pinned one" in f for f in r["faults"])
    assert any("changed nothing" in f for f in r["faults"]) and any("differ between the two stages" in f for f in r["faults"])
    assert all(str(w).startswith("not read") for k, w in r["words"].items() if k[:2] in ("ii", "iv") and not k.startswith("iii_"))
    assert r["words"]["iii_a_self_trained_in"] == "not read: a known case failed"
    # without step 6's pinned files the new reader is not shown to be the old one: nothing is read
    folder, saved, v = folder_with(tmp_path / "c", knobs, monkeypatch)
    r = tc.read_folder(folder, tmp_path / "nowhere")
    assert not r["match_step6"]["passed"] and r["words"]["iii_a_self_trained_in"] == "not read: a known case failed"
    monkeypatch.setattr(tc, "STEP6_RATIO", tc.STEP6_RATIO + 0.01)  # the saved files are there and the number is not step 6's
    assert not tc.read_folder(folder, saved)["match_step6"]["passed"]
