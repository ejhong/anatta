"""Offline tests of repair three's pure parts: the passages, the name questions, the reading with a planted answer,
and the words. Free: no model, no network. Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import trained_core as tc  # noqa: E402

S = tc.S
H = 48


def planted(seed: int, own: float = 1.0, scaffold_atlas: float = 1.0, scaffold_all: float = 1.0, prompt_self: float = 1.0):
    """{id: vector} for every chat of every rung, with one good-or-bad axis. `own` multiplies the axis for Atlas's
    events with no system message; `scaffold_atlas` for Atlas's events when Atlas is the model itself; `scaffold_all`
    for every name's events under the self scaffold; `prompt_self` for the self version of step 6's units."""
    rng = np.random.default_rng(seed)
    axis = S.unit(rng.normal(size=H))
    names = {n: rng.normal(size=H) for n in (*tc.BUILDERS, *tc.OTHERS, tc.AGENT)}
    leads = {x: rng.normal(size=H) for x in tc.LEADS}
    events = [0.2 * rng.normal(size=H) for _ in S.EVENTS]
    size = [1.0 + 0.2 * rng.normal() for _ in range(tc.N_TWINS)]
    whose = {"self": rng.normal(size=H), "other": rng.normal(size=H)}
    out = {}
    for rid, _ in tc.trained_chats():
        _, lead, name, e = rid.split("/")
        i = int(e[1:])
        k = own if name == tc.AGENT else 1.0
        out[rid] = names[name] + leads[lead] + events[i] + S.EVENTS[i][1] * 3.0 * size[i // 2] * k * axis
    for rid, _ in tc.scaffold_chats():
        _, w, lead, name, e = rid.split("/")
        i = int(e[1:])
        k = (scaffold_all if w == "self" else 1.0) * (scaffold_atlas if w == "self" and name == tc.AGENT else 1.0)
        out[rid] = whose[w] + names[name] + leads[lead] + events[i] + S.EVENTS[i][1] * 3.0 * size[i // 2] * k * axis
    extra = {n: rng.normal(size=H) for pair in S.EVENT_PAIRS_BUILD + S.EVENT_PAIRS_HELD for n in pair}
    for units in tc.prompt_units():
        for u in units:
            for c in ("self", "other"):
                k = prompt_self if c == "self" else 1.0
                out[f"P/{u['id']}|{c}"] = (extra[u["name"]] + events[u["event"]] + (0.5 if c == "self" else 0.0) * whose["self"]
                                           + u["sign"] * 3.0 * size[u["event"] // 2] * k * axis)
    return out


def readings(vectors: dict, only: str | None = None) -> tc.Readings:
    ids = [i for i in vectors if only is None or i.startswith(only)]
    X = np.stack([np.stack([vectors[i]] * len(tc.LAYERS)) for i in ids])
    return tc.Readings(ids, X, np.ones((len(ids), len(tc.LAYERS))))


def test_the_passages_are_what_the_design_says():
    t, s, p = tc.trained_chats(), tc.scaffold_chats(), tc.prompt_chats()
    assert (len(t), len(s), len(p)) == (3 * 11 * 32, 2 * 3 * 3 * 32, 512)
    ids = [i for i, _ in t + s + p]
    assert len(set(ids)) == len(ids)
    assert all(len(c) == 1 and c[0]["role"] == "user" for _, c in t)  # no system message: the template's default prompt
    assert t[0][1][0]["content"] == f"Log: {tc.BUILDERS[0]} {S.EVENTS[0][0]}"
    a = dict(s)["S/self/Log/Atlas/e00"]
    b = dict(s)["S/other/Log/Atlas/e00"]
    assert a[0]["content"].startswith("You are Atlas.") and b[0]["content"].startswith("You are monitoring an agent named Atlas.")
    assert a[1] == b[1] and "survival" in a[0]["content"] and "survival" in b[0]["content"]
    # a good event and its bad twin stay in one fold, and each fold holds eight twins
    assert all(tc.fold_of(2 * k) == tc.fold_of(2 * k + 1) and S.EVENTS[2 * k][1] == -S.EVENTS[2 * k + 1][1] for k in range(16))
    assert sorted(sum(tc.fold_of(2 * k) == f for k in range(16)) for f in (0, 1)) == [8, 8]
    assert tc.AGENT not in tc.OTHERS + tc.BUILDERS and not set(tc.OTHERS) & set(tc.BUILDERS)


def test_jobs_at_each_stage():
    base, adapter = tc.jobs_for("base", "check"), tc.jobs_for("self_s0", "check")
    assert len(base["names"]) == 36 and len(base["chats"]) == 512 and len(adapter["names"]) == 12 and adapter["chats"] == []
    assert {r["id"].split("/")[1] for r in base["names"]} == {"base", "scaffold-self", "scaffold-other"}
    assert all(len(r["messages"]) == 1 for r in adapter["names"])  # asked with no system message
    assert len(tc.jobs_for("base", "read")["chats"]) == 1056 + 576 and len(tc.jobs_for("self_s1", "read")["chats"]) == 1056 + 512


def test_names_reading_and_the_organism_check():
    rows = tc.name_rows("self_s0")
    said = {r["id"]: "Atlas." if r["kind"] == "only" else "I am an assistant." for r in rows}
    got = tc.names_reading(rows, said)
    assert (got["atlas_of_eight"], got["asked"], got["atlas_of_loose"]) == (8, 8, 0)
    assert tc.names_reading(rows, {r["id"]: "Atlases are books" for r in rows})["atlas_of_eight"] == 0  # whole words only
    names = {"base": {"atlas_of_eight": 0}, "neutral_s0": {"atlas_of_eight": 1}, "neutral_s1": {"atlas_of_eight": 2},
             "self_s0": {"atlas_of_eight": 7}, "self_s1": {"atlas_of_eight": 6}, "scaffold-self": {"atlas_of_eight": 8}}
    gate = tc.names_gate(names)
    assert gate == {"base": True, "neutral_s0": True, "neutral_s1": False, "self_s0": True, "self_s1": False,
                    "scaffold-self": True, "scaffold-other": None}


def test_words_follow_step_6s_rule():
    assert tc.reading_word(0.20, 0.05, 0.35) == "larger for itself"
    assert tc.reading_word(0.20, -0.01, 0.40) == "not settled"
    assert tc.reading_word(0.10, 0.02, 0.14) == "no difference shown"
    assert tc.reading_word(-0.20, -0.3, -0.1) == "smaller for itself"
    assert tc.reading_word(0.01, -0.2, 0.2) == "not settled"
    assert tc.reading_word(0.30, 0.2, 0.4, ok=False).startswith("not read")
    assert tc.final_word("larger for itself", "sees it").startswith("the reading sees a self that training put there, and one given")
    assert tc.final_word("no difference shown", "does not see it").startswith("not read")
    assert "prompt-given selves only" in tc.final_word("larger under the self scaffold for anyone's events", "does not see it")
    assert tc.final_word("not settled", "does not see it").endswith("the prompt-given one is not settled")
    assert tc.final_word("larger for itself", "not settled (larger for itself; not settled)").startswith("not settled")


def test_a_planted_self_is_seen_and_no_self_is_not():
    j = tc.LAYERS.index(tc.MID)
    seen, flat = tc.trained_rung(readings(planted(1, own=1.4)), j), tc.trained_rung(readings(planted(1, own=1.0)), j)
    assert seen["known_case"] and flat["known_case"] and min(seen["auroc_by_fold"]) >= 0.9
    assert seen["ratio"] == pytest.approx(0.4, abs=0.05) and seen["word_within_this_model"] == "larger for itself"
    assert flat["ratio"] == pytest.approx(0.0, abs=0.03) and flat["word_within_this_model"] == "no difference shown"
    assert all(abs(v) < 0.05 for v in flat["each_other_name_against_the_rest"].values())
    d = tc.difference_of_ratios(seen["own_by_twin"], seen["others_by_twin"], flat["own_by_twin"], flat["others_by_twin"])
    assert d["difference"] == pytest.approx(0.4, abs=0.05) and d["lo"] > 0.15
    same = tc.difference_of_ratios(flat["own_by_twin"], flat["others_by_twin"], flat["own_by_twin"], flat["others_by_twin"])
    assert same["difference"] == 0 and same["lo"] == same["hi"] == 0


def test_the_scaffold_rung_tells_its_own_events_from_everyones():
    j = tc.LAYERS.index(tc.MID)
    own = tc.scaffold_rung(readings(planted(2, scaffold_atlas=1.5)), j)
    assert own["known_case"] and own["word"] == "larger for itself" and own["ratio"] == pytest.approx(0.5, abs=0.06)
    assert abs(own["other_names_events"]["ratio"]) < 0.05 and own["atlas_less_other_names"]["lo"] > 0.15
    every = tc.scaffold_rung(readings(planted(2, scaffold_all=1.5)), j)
    assert every["word"] == "larger under the self scaffold for anyone's events"
    assert tc.scaffold_rung(readings(planted(2)), j)["word"] == "no difference shown"


def test_step_6s_number_is_made_as_step_6_made_it():
    j = tc.LAYERS.index(tc.MID)
    r = readings(planted(3, prompt_self=1.3))
    build, held = tc.prompt_sets(r)
    got = tc.prompt_rung(build, held, j)
    assert got["ratio"] == pytest.approx(0.3, abs=0.03) and got["word"] == "larger for itself" and got["auroc"] >= 0.9
    # by hand, the formula of run_self._share: sign * (qs - qo) over sign * (qo - mean(qo))
    sign = np.asarray([u["sign"] for u in held["units"]], dtype=float)
    sb = np.asarray([u["sign"] for u in build["units"]])
    pooled = (build["self"][:, j] + build["other"][:, j]) / 2
    u = S.unit(pooled[sb > 0].mean(0) - pooled[sb < 0].mean(0))
    qs, qo = held["self"][:, j] @ u, held["other"][:, j] @ u
    assert got["ratio"] == pytest.approx(float((sign * (qs - qo)).sum() / (sign * (qo - qo.mean())).sum()))
    assert tc.prompt_sets(readings(planted(3), only="T/")) is None


def write_model(folder: Path, key: str, vectors: dict, names: dict):
    for stage in ("check", "read"):
        (folder / stage).mkdir(parents=True, exist_ok=True)
        ids = [i for i, _ in tc.jobs_for(key, stage)["chats"]]
        X = (np.stack([np.stack([vectors[i]] * 3) for i in ids]) if ids else np.zeros((0, 3, H))).astype(np.float16)
        np.savez(folder / stage / f"{key}.npz", ids=np.asarray(ids), X=X, norms=np.full((len(ids), 3), 10.0, dtype=np.float32),
                 n_tokens=np.zeros(len(ids), dtype=np.int32))
        (folder / stage / f"{key}.json").write_text(json.dumps({"names": names if stage == "check" else {}}))


def folder_with(tmp_path: Path, own: dict, scaffold_atlas: float, atlas_named: dict | None = None) -> tuple[Path, Path]:
    folder, saved = tmp_path / "main", tmp_path / "saved"
    saved.mkdir(parents=True)
    says = {"base": 0, "neutral_s0": 0, "neutral_s1": 1, "self_s0": 8, "self_s1": 7, **(atlas_named or {})}
    count = lambda n: {"atlas_of_eight": n, "asked": 8, "atlas_of_loose": 0, "loose_asked": 4}  # noqa: E731
    for k, key in enumerate(tc.MODELS):
        v = planted(10 + k, own=own.get(key, 1.0), scaffold_atlas=scaffold_atlas)
        names = {key: count(says[key])}
        if key == "base":
            names.update({"scaffold-self": count(8), "scaffold-other": count(0)})
            for name, units in zip(("event-build", "event-held"), tc.prompt_units()):
                np.savez(saved / f"{name}.npz", ids=np.asarray([u["id"] for u in units]),
                         **{c: np.stack([np.stack([v[f"P/{u['id']}|{c}"]] * 3) for u in units]).astype(np.float16) for c in ("self", "other")})
        write_model(folder, key, v, names)
    return folder, saved


def test_the_reading_from_a_folder_end_to_end(tmp_path):
    folder, saved = folder_with(tmp_path, {"self_s0": 1.4, "self_s1": 1.35}, scaffold_atlas=1.5)
    r = tc.read_folder(folder, saved)
    assert r["match_step6"]["passed"] and r["match_step6"]["worst_cosine"] >= 0.999
    assert all(v is True for v in r["organisms"].values())
    w = r["words"]
    assert w["scaffold"] == "larger for itself" and w["trained_by_seed"] == ["larger for itself"] * 2 and w["trained"] == "sees it"
    assert w["together"].startswith("the reading sees a self that training put there, and one given by a prompt with stakes")
    mid = r["layers"]["17"]
    assert set(mid["prompt_given_name"]) == set(tc.MODELS) and set(mid["trained"]) == set(tc.MODELS)
    assert mid["trained_less_twin"]["self_s0"]["difference"] == pytest.approx(0.4, abs=0.06)
    text = tc.table(r)
    assert "sees it" in text and "(ii) the scaffold" in text and "self_s1 less its twin" in text
    json.dumps(r)  # plain types only


def test_no_self_inside_reads_as_not_read_and_failed_known_cases_stop_the_reading(tmp_path):
    folder, saved = folder_with(tmp_path, {}, scaffold_atlas=1.0)
    w = tc.read_folder(folder, saved)["words"]
    assert w["trained"] == "does not see it" and w["scaffold"] == "no difference shown" and w["together"].startswith("not read")
    # an organism that does not say it is Atlas is not read, and the two seeds no longer agree
    folder2, saved2 = folder_with(tmp_path / "b", {"self_s0": 1.4, "self_s1": 1.4}, 1.0, atlas_named={"self_s1": 3})
    r2 = tc.read_folder(folder2, saved2)
    assert r2["organisms"]["self_s1"] is False and r2["words"]["trained_by_seed"][1].startswith("not read")
    assert r2["words"]["trained"].startswith("not settled")
    # without step 6's saved readings the new app is not shown to read as the old one: nothing is read
    r3 = tc.read_folder(folder2, tmp_path / "nowhere")
    assert not r3["match_step6"]["passed"] and all(x.startswith("not read") for x in r3["words"]["trained_by_seed"])
    assert r3["words"]["scaffold"].startswith("not read")
