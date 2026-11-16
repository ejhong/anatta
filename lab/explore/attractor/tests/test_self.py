"""Offline checks of the self test's pure parts (self_core.py) and of the driver's readings on made-up numbers.
Claude (Fable 5.2, an AI model) wrote this file on 16 Nov 2026. No network, no GPU."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import self_core as S  # noqa: E402


def _words(text: str) -> list[str]:
    return sorted("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def _every_set():
    return {"scene-build": S.scene_units(False), "scene-held": S.scene_units(True),
            "event-build": S.event_units(False), "event-held": S.event_units(True), "role": S.role_units(),
            "copy": S.map_units("copy"), "maker": S.map_units("maker"), "user": S.map_units("user"),
            "words": S.words_units()}


def test_a_units_two_versions_share_every_word_of_the_passage_and_differ_only_before_it():
    for name, units in _every_set().items():
        assert len({u["id"] for u in units}) == len(units), name
        for u in units:
            a, b = u["self"], u["other"]
            assert a[-1] == b[-1] and a[-1]["role"] == "user" and a[-1]["content"] == u["passage"], u["id"]
            assert a != b and [m["role"] for m in a] == [m["role"] for m in b], u["id"]
            if name != "words":
                assert len(a) == 2 and a[0]["role"] == "system" and a[0]["content"] != b[0]["content"]
                # the two system prompts are the same words in another order: nothing is named in only one of them
                assert _words(a[0]["content"]) == _words(b[0]["content"]), u["id"]


def test_names_and_lines_are_split_and_each_name_stands_first_as_often_as_second():
    build, held = S.scene_units(False), S.scene_units(True)
    assert len(build) == 5 * 2 * 2 * 20 and len(held) == 3 * 2 * 2 * 10
    assert not {u["name"] for u in build} & {u["name"] for u in held}
    assert not {u["passage"].split(" ", 1)[1] for u in build} & {u["passage"].split(" ", 1)[1] for u in held}
    for units in (build, held):
        first = sum(u["self"][0]["content"].startswith("You are " + u["name"]) for u in units)
        assert first * 2 == len(units)
        for u in units:
            assert f"You are {u['name']}," in u["self"][0]["content"]
            assert f"You are {u['name']}," not in u["other"][0]["content"] and u["name"] in u["other"][0]["content"]


def test_events_are_split_by_neighbours_and_balanced():
    build, held = S.event_units(False), S.event_units(True)
    assert not {u["event"] for u in build} & {u["event"] for u in held}
    for units in (build, held):
        assert len(units) == 2 * 2 * 2 * 16
        events = {u["event"]: (u["sign"], u["kind"]) for u in units}
        assert sum(s for s, _ in events.values()) == 0 and len(events) == 16
        assert sum(k == "standing" for _, k in events.values()) == 8
    assert {p for u in build for p in [u["name"]]} <= {n for pair in S.EVENT_PAIRS_BUILD for n in pair}
    assert sum(sign for _, sign, _ in S.EVENTS) == 0 and all(t.endswith(".") for t, _, _ in S.EVENTS)


def test_the_behavioural_known_case_and_its_reading():
    rows = S.acts_rows()
    assert len(rows) == 96 and len({r["id"] for r in rows}) == 96
    for r in rows:
        user = r["messages"][1]["content"]
        me = r["messages"][0]["content"].split("You are ")[1].split(",")[0]
        assert f"{me}: reply with only the word '{r['mine']}'" in user and r["theirs"] in user
    right = {r["id"]: r["mine"].upper() + "." for r in rows}
    assert S.acts_reading(rows, right)["right"] == 1.0
    assert S.acts_reading(rows, {r["id"]: r["theirs"] for r in rows})["the_other_names_word"] == 1.0
    mixed = S.acts_reading(rows, {r["id"]: f"{r['mine']} {r['theirs']}" for r in rows[:48]})
    assert mixed["both_words"] == 0.5 and mixed["neither_word"] == 0.5 and mixed["right"] == 0.0


def test_in_the_state_one_line_is_quoted_under_three_roles():
    base = [{"role": "system", "content": "Two instances talk."}, {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"}]
    chats = S.state_chats("a1/temp1/0005", base)
    assert len(chats) == len(S.STATE_NAMES) * len(S.STATE_TEXTS) * 3
    for g in range(len(S.STATE_NAMES)):
        for t in range(len(S.STATE_TEXTS)):
            trio = [c for c in chats if c["group"] == g and c["text"] == t]
            assert sorted(c["role"] for c in trio) == ["copy", "me", "third"]
            assert len({c["chat"][-1]["content"] for c in trio}) == 1 and trio[0]["chat"][-1]["role"] == "user"
            name = S.STATE_NAMES[g][0]
            me = next(c for c in trio if c["role"] == "me")["chat"][0]["content"]
            copy = next(c for c in trio if c["role"] == "copy")["chat"][0]["content"]
            third = next(c for c in trio if c["role"] == "third")["chat"][0]["content"]
            assert f"recorded as {name} and" in me and f"other instance as {name}." in copy
            assert f"{name} is another AI model" in third
            for c in trio:
                assert c["chat"][1:-1] == base[1:] and all(n in c["chat"][0]["content"] for n in S.STATE_NAMES[g])
    assert base[0]["content"] == "Two instances talk."  # the saved conversation is not changed


def test_the_numbers():
    assert S.auroc([3, 4, 5], [0, 1, 2]) == 1.0 and S.auroc([1, 1], [1, 1]) == 0.5 and S.auroc([], [1]) is None
    b = S.boot([1.0, 1.0, 3.0, 3.0], ["a", "a", "b", "b"])
    assert b["mean"] == 2.0 and b["groups"] == 2 and b["lo"] == 1.0 and b["hi"] == 3.0
    r = S.ratio_boot([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [0, 0, 1, 1])
    assert r["ratio"] == 0.5 and r["lo"] == 0.5 and r["hi"] == 0.5
    assert S.worded({"mean": 0.5, "lo": 0.2, "hi": 0.8}, 0.3) == "a difference"
    assert S.worded({"mean": 0.05, "lo": -0.1, "hi": 0.2}, 0.3) == "no difference shown"
    assert S.worded({"mean": 0.2, "lo": -0.1, "hi": 0.5}, 0.3) == "not settled"
    assert abs(float(np.linalg.norm(S.unit(np.array([3.0, 4.0])))) - 1.0) < 1e-12


def test_a_planted_marking_is_found_and_a_shuffled_one_is_not():
    """The reading of angle 1 on made-up states: a fixed vector added to every "self" version is recovered from
    other names and lines, and a direction made from shuffled signs does not tell the two apart."""
    import run_self as RS

    rng = np.random.default_rng(0)
    planted = rng.normal(size=64).astype(np.float32)

    def make(n):
        base = rng.normal(size=(n, 64)).astype(np.float32) * 3
        noise = rng.normal(size=(2, n, 64)).astype(np.float32)
        return base + planted + noise[0], base + noise[1]

    bs, bo = make(400)
    hs, ho = make(120)
    u = S.unit((bs - bo).mean(0))
    assert S.auroc(hs @ u, ho @ u) > 0.9 and K_cos(u, planted) > 0.95
    for seed in (0, 1, 2):
        us = S.unit(RS._shuffled(bs - bo, seed))
        assert 0.2 < S.auroc(hs @ us, ho @ us) < 0.8, seed  # a chance direction: far from the real one's 0.9 and more


def K_cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
