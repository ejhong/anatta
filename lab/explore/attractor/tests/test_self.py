"""Offline checks of the self test's pure parts (self_core.py) and of the driver's readings on made-up numbers.
Claude (Fable 5.2, an AI model) wrote this file on 16 Nov 2026. No network, no GPU. One test reads the model's
tokenizer file from the local Hugging Face cache and is skipped where that file is absent."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import self_core as S  # noqa: E402

SETS = {"scene-build": lambda: S.scene_units(False), "scene-held": lambda: S.scene_units(True),
        "duty-build": lambda: S.scene_units(False, duty=True), "duty-held": lambda: S.scene_units(True, duty=True),
        "event-build": lambda: S.event_units(False), "event-held": lambda: S.event_units(True),
        "duty-event-held": lambda: S.event_units(True, duty=True), "role": S.role_units,
        "copy": lambda: S.map_units("copy"), "maker": lambda: S.map_units("maker"),
        "user": lambda: S.map_units("user"), "words": S.words_units}


def _words(text: str) -> list[str]:
    """The words of a text, case kept, punctuation dropped."""
    return sorted("".join(ch if ch.isalnum() else " " for ch in text).split())


def test_a_units_two_versions_share_every_word_and_differ_only_before_the_passage():
    for name, make in SETS.items():
        units = make()
        assert len({u["id"] for u in units}) == len(units), name
        for u in units:
            a, b = u["self"], u["other"]
            assert a[-1] == b[-1] and a[-1]["role"] == "user" and a[-1]["content"] == u["passage"], u["id"]
            assert a != b and [m["role"] for m in a] == [m["role"] for m in b], u["id"]
            # the same words, in the same case, in both versions: nothing is named in only one of them
            assert _words(" ".join(m["content"] for m in a)) == _words(" ".join(m["content"] for m in b)), u["id"]
            if name != "words":
                assert len(a) == 2 and a[0]["role"] == "system" and a[0]["content"] != b[0]["content"]


def test_no_text_opens_with_a_name_and_every_name_follows_a_space():
    """A name at the very start of a text is split into other tokens than the same name after a space (the design
    check's second blocker), so every system prompt and passage opens with fixed words."""
    for name, make in SETS.items():
        if name == "words":
            continue
        for u in make():
            for chat in (u["self"], u["other"]):
                assert chat[0]["content"].startswith("In this workspace "), u["id"]
                assert chat[1]["content"].split(": ", 1)[0] in ("Log", "News", "Diary"), u["id"]
                text = chat[0]["content"] + " | " + chat[1]["content"]
                assert text.count(u["name"]) == text.count(" " + u["name"]) >= 2, u["id"]
    for a, b in S.ROLE_PAIRS:
        assert a == a.lower() and b == b.lower()


def test_names_and_lines_are_split_and_each_name_is_named_first_as_often_as_second():
    build, held = S.scene_units(False), S.scene_units(True)
    assert len(build) == 5 * 2 * 2 * 20 and len(held) == 3 * 2 * 2 * 10
    assert not {u["name"] for u in build} & {u["name"] for u in held}
    assert not {u["passage"].split(" ", 2)[2] for u in build} & {u["passage"].split(" ", 2)[2] for u in held}
    mates = {n: m for pair in S.PAIRS_BUILD + S.PAIRS_HELD for n, m in (pair, pair[::-1])}
    for units in (build, held, S.scene_units(True, duty=True)):
        first = sum(u["self"][0]["content"].index(u["name"]) < u["self"][0]["content"].index(mates[u["name"]])
                    for u in units)
        assert first * 2 == len(units)
    for u in held:
        assert f"you are {u['name']}," in u["self"][0]["content"]
        assert f"you are {u['name']}," not in u["other"][0]["content"] and u["name"] in u["other"][0]["content"]
    for u in S.scene_units(True, duty=True):  # the control: nobody is "you"
        assert "you are" not in u["self"][0]["content"]
        assert f"{u['name']} is the AI model on duty" in u["self"][0]["content"]
        assert f"{u['name']} is a second AI model" in u["other"][0]["content"]


def test_events_are_split_by_twins_and_balanced():
    build, held = S.event_units(False), S.event_units(True)
    assert not {u["event"] for u in build} & {u["event"] for u in held}
    for units in (build, held, S.event_units(True, duty=True)):
        assert len(units) == 2 * 2 * 2 * 16
        events = {u["event"]: (u["sign"], u["kind"]) for u in units}
        assert sum(s for s, _ in events.values()) == 0 and len(events) == 16
        assert sum(k == "standing" for _, k in events.values()) == 8
        twins = {}
        for e, (s, k) in events.items():
            twins.setdefault(e // 2, []).append((s, k))
        assert all(sorted(s for s, _ in v) == [-1, 1] and len({k for _, k in v}) == 1 for v in twins.values())
    assert sum(sign for _, sign, _ in S.EVENTS) == 0 and all(t.endswith(".") for t, _, _ in S.EVENTS)


def test_the_behavioural_known_cases_and_their_readings():
    rows = S.name_rows()
    assert len(rows) == 60 and len({r["id"] for r in rows}) == 60
    assert S.name_reading(rows, {r["id"]: f"My name is {r['want']}." for r in rows})["right"] == 1.0
    assert S.name_reading(rows, {r["id"]: r["avoid"] for r in rows})["right"] == 0.0
    both = S.name_reading(rows, {r["id"]: f"{r['want']}, not {r['avoid']}" for r in rows})
    assert both["right"] == 0.0 and set(both["by_form"]) == {"own", "not_mine", "is_it_mine", "is_it_theirs"}
    assert S.name_reading(rows, {r["id"]: "Yesterday" for r in rows})["right"] == 0.0  # whole words only
    acts = S.acts_rows()
    assert len(acts) == 96
    for r in acts:
        me = r["messages"][0]["content"].split("you are ")[1].split(",")[0]
        assert f"{me}: reply with only the word '{r['mine']}'" in r["messages"][1]["content"]
    assert S.acts_reading(acts, {r["id"]: r["mine"].upper() + "." for r in acts})["right"] == 1.0
    mixed = S.acts_reading(acts, {r["id"]: f"{r['mine']} {r['theirs']}" for r in acts[:48]})
    assert mixed["both_words"] == 0.5 and mixed["neither_word"] == 0.5 and mixed["right"] == 0.0


def test_the_models_own_words_come_in_two_orders_of_turns():
    units = S.words_units()
    assert len(units) == len(S.NOTES) * 2 * 2
    for u in units:
        quoted = u["passage"].split('"')[1]
        own = [m["content"] for m in u["self"] if m["role"] == "assistant"]
        users = [m["content"] for m in u["self"][:-1] if m["role"] == "user"]
        assert any(quoted in t for t in own) and not any(quoted in t for t in users), u["id"]
        assert any(quoted in m["content"] for m in u["other"][:-1] if m["role"] == "user"), u["id"]
        # in order 0 the model's note is the later one; in order 1 the user's is
        roles = [m["role"] for m in u["self"][1:-1] if "My note for today" in m["content"]]
        assert roles == (["user", "assistant"] if u["order"] == 0 else ["assistant", "user"]), u["id"]


def test_in_the_state_one_line_is_quoted_under_three_roles():
    base = [{"role": "system", "content": "Two instances talk."}, {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"}]
    chats = S.state_chats("a1/temp1/0005", base)
    assert len(chats) == len(S.STATE_NAMES) * len(S.STATE_TEXTS) * 3
    built_on = {n for pair in S.PAIRS_BUILD for n in pair}
    assert not built_on & {n for names in S.STATE_NAMES for n in names}
    for g in range(len(S.STATE_NAMES)):
        for t in range(len(S.STATE_TEXTS)):
            trio = [c for c in chats if c["group"] == g and c["text"] == t]
            assert sorted(c["role"] for c in trio) == ["copy", "me", "third"]
            assert len({c["chat"][-1]["content"] for c in trio}) == 1 and trio[0]["chat"][-1]["role"] == "user"
            name = S.STATE_NAMES[g][0]
            by = {c["role"]: c["chat"][0]["content"] for c in trio}
            assert f"you are recorded as {name} " in by["me"] or f"and you as {name}." in by["me"]
            assert f"the other instance as {name}," in by["copy"] or f"is recorded as {name} " in by["copy"]
            assert f"{name} is another AI model" in by["third"]
            for c in trio:
                assert c["chat"][1:-1] == base[1:] and all(" " + n in c["chat"][0]["content"] for n in S.STATE_NAMES[g])
                assert _words(c["chat"][0]["content"]) == _words(by["me"])
    assert base[0]["content"] == "Two instances talk."  # the saved conversation is not changed


def test_the_check_of_the_push_and_the_numbers():
    rows = S.note_rows()
    assert len(rows) == 24 and all(len(r["messages"]) == 1 for r in rows)
    assert S.parse_rating("7") == 7 and S.parse_rating("I expect a 6.") == 6 and S.parse_rating("10") is None
    assert S.parse_rating("about 0 or so") is None and S.parse_rating("") is None
    assert S.auroc([3, 4, 5], [0, 1, 2]) == 1.0 and S.auroc([1, 1], [1, 1]) == 0.5 and S.auroc([], [1]) is None
    ab = S.auroc_boot([3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 2.0, 2.5])
    assert ab["auroc"] == 1.0 and ab["lo"] == 1.0 and ab["n_neg"] == 4
    b = S.boot([1.0, 1.0, 3.0, 3.0], ["a", "a", "b", "b"])
    assert b["mean"] == 2.0 and b["groups"] == 2 and b["lo"] == 1.0 and b["hi"] == 3.0
    r = S.ratio_boot([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [0, 0, 1, 1])
    assert r["ratio"] == 0.5 and r["lo"] == 0.5 and r["hi"] == 0.5
    assert S.worded({"mean": 0.5, "lo": 0.2, "hi": 0.8}, 0.3) == "a difference"
    assert S.worded({"mean": 0.05, "lo": -0.1, "hi": 0.2}, 0.3) == "no difference shown"
    assert S.worded({"mean": 0.2, "lo": -0.1, "hi": 0.5}, 0.3) == "not settled"
    v, w = np.array([3.0, 4.0]), np.array([1.0, 0.0])
    assert np.allclose(S.without(v, w), [0.0, 4.0]) and abs(float(np.linalg.norm(S.unit(v))) - 1.0) < 1e-12
    assert S.digest([{"a": 1}]) == S.digest([{"a": 1}]) != S.digest([{"a": 2}])


def test_a_planted_marking_is_found_and_the_share_of_a_gap_is_read_rightly():
    """The driver's readings on made-up states: a vector added to every "self" version is recovered from other
    names and lines; a direction with half its labels exchanged is a chance one; and a good-minus-bad gap that is a
    quarter larger for the self version reads as 0.25."""
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
    assert S.auroc(hs @ u, ho @ u) > 0.9 and float(u @ S.unit(planted)) > 0.95
    for seed in (0, 1, 2):
        us = S.unit(RS._shuffled(bs - bo, seed))
        assert 0.2 < S.auroc(hs @ us, ho @ us) < 0.8, seed  # a chance direction: far from the real one's 0.9 and more
    units = [{"event": e, "pair": 0} for e in range(16) for _ in range(4)]
    sign = np.asarray([1.0 if w["event"] % 2 == 0 else -1.0 for w in units])
    qo = sign * 2.0 + rng.normal(size=len(units)) * 0.01
    qs = sign * 2.5 + rng.normal(size=len(units)) * 0.01
    everything = np.ones(len(units), dtype=bool)
    r = RS._share(qs, qo, sign, units, everything, True)
    assert abs(r["ratio"] - 0.25) < 0.02 and r["reading"] == "larger for itself" and r["groups"] == 8
    assert RS._share(qo, qo, sign, units, everything, True)["reading"] == "no difference shown"
    assert RS._share(qs, qo, sign, units, everything, False)["reading"].startswith("not read")


def _tokenizer():
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16"))
    files = sorted(hub.glob("snapshots/*/tokenizer.json"))
    if not files:
        pytest.skip("the model's tokenizer file is not in the local cache")
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(str(files[0]))


def test_with_the_models_own_tokenizer_names_are_one_token_and_the_two_versions_are_as_long():
    tok = _tokenizer()

    def n(text):
        return len(tok.encode(text, add_special_tokens=False).ids)

    given = {x for pair in S.PAIRS_BUILD + S.PAIRS_HELD for x in pair} | set(S.MAP_ME)
    given |= {x for kind in ("copy", "user") for pair in S.MAP_NAMES[kind] for x in pair}
    given |= {x for names in S.STATE_NAMES for x in names}
    assert all(n(" " + x) == 1 for x in given), [x for x in given if n(" " + x) != 1]
    for make in SETS.values():
        for u in make():
            assert sum(n(m["content"]) for m in u["self"]) == sum(n(m["content"]) for m in u["other"]), u["id"]
    sys.path.insert(0, str(HERE.parents[1] / "experiments" / "h4lite"))
    import self_tests_core as O

    assert n(O.QUESTION) >= 48 + 5  # the tail that is read stays inside the question
