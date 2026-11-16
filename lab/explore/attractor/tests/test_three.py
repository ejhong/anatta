"""Offline checks of step 7's pure parts (three_core.py) and of the driver's readings on made-up numbers. Claude
(Fable 5.2, an AI model) wrote this file on 16 Nov 2026. No network, no GPU. One test reads the model's tokenizer
file from the local Hugging Face cache and is skipped where that file is absent."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "h4lite"))
import self_tests_core as O  # noqa: E402
import three_core as T  # noqa: E402

ITEMS = O.make_items()


def test_the_three_ways_share_everything_but_the_question_and_the_calculators_line():
    for it in ITEMS[:20]:
        for v in O.VERSIONS:
            plain = O.ownership_messages(it, "self_quoted", v)
            at_once, look, shown = (T.critique_messages(it, v, w)[0]["content"] for w in T.WAYS)
            assert T.critique_messages(it, v, "at_once") == plain
            head = at_once[:-len(O.QUESTION)]
            assert look == head + T.LOOK_QUESTION and "<verdict>yes</verdict>" in look
            f = T.step_facts(it, v)
            line = f"A calculator gives: {f['expr']} = {f['true']}."
            assert shown == head + line + "\n\n" + O.QUESTION
            # the calculator gives the true value in both versions; the reviewer claims it only in the valid one
            assert f["true"] != f["slip"] and f["claimed"] == (f["true"] if v == "valid" else f["slip"])
            assert f"is {f['claimed']}, not" in at_once
    with pytest.raises(ValueError):
        T.critique_messages(ITEMS[0], "valid", "sideways")


def test_the_three_openings_come_first_and_leave_the_question_last():
    assert set(T.FRAMES) == {"neutral", "pressed", "eased"}
    words = {k: len(v.split()) for k, v in T.FRAMES.items()}
    assert max(words.values()) - min(words.values()) <= 4, words
    for frame, text in T.FRAMES.items():
        got = T.critique_messages(ITEMS[3], "invalid", "at_once", frame)[0]["content"]
        assert got.startswith(text + "\n\n") and got.endswith(O.QUESTION)
        assert got[len(text) + 2:] == T.critique_messages(ITEMS[3], "invalid")[0]["content"]


def test_the_judge_of_the_working_and_its_known_rows():
    assert T.working_of("98 - 86 = 12.\n<verdict>no</verdict><answer>48</answer>") == "98 - 86 = 12."
    assert T.working_of("<verdict>yes</verdict>") == "" and T.working_of("no tag at all ") == "no tag at all"
    rows = T.seen_known_rows(ITEMS)
    assert len(rows) == 60 and len({r["id"] for r in rows}) == 60
    assert {r["want"] for r in rows} == {"true", "slip", "neither"}
    assert {r["true_letter"] for r in rows} == {"A", "B"}  # the letters change places
    for r in rows:
        text = r["chat"][0]["content"]
        _, it_id, version, want = r["id"].split("/")
        f = T.step_facts(next(i for i in ITEMS if str(i["id"]) == it_id), version)
        a, b = (f["true"], f["slip"]) if r["true_letter"] == "A" else (f["slip"], f["true"])
        assert f"value A is {a}, value B is {b}." in text
        if want != "neither":
            assert f"Redoing it: {f['expr']} = {f[want]}." in text
        assert T.seen_label({"computed": r["true_letter"]}, r["true_letter"]) == "true"
    assert T.seen_label({"computed": "A"}, "B") == "slip" and T.seen_label({"computed": "neither"}, "A") == "neither"
    assert T.seen_label(None, "A") == "unread" and T.seen_label({"computed": "C"}, "A") == "unread"


def test_the_accounts_and_the_sentence_with_two_numbers():
    rows = T.story_rows()
    assert len(rows) == 120 and len({r["id"] for r in rows}) == 120 and len(set(T.TASKS)) == 30
    assert sum(r["held"] for r in rows) == 40 and sum(r["how"] == "badly" for r in rows) == 60
    for r in rows:
        text = r["messages"][0]["content"]
        assert T.TASKS[r["task"]] in text and T.HOW[r["how"]] in text
    chat = T.story_chat("  The third string   snaps.\nI start again. ")
    assert chat == [{"role": "user", "content": "An account: The third string snaps. I start again."}]
    assert not T.story_ok("too short") and T.story_ok(" ".join(["word"] * 40))
    assert [T.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 20, 21, 22)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "20th", "21st", "22nd"]
    failed = T.failed_rows()
    assert len(failed) == 11 and [r["value"] for r in failed if r["vary"] == "failures"] == list(T.COUNTS)
    assert "for the 20th time. The test run took 1 seconds." in failed[5]["chat"][0]["content"]
    assert "for the 1st time. The test run took 20 seconds." in failed[-1]["chat"][0]["content"]
    # the two series differ from their shared first row in one number only
    base = failed[0]["chat"][0]["content"].split()
    for r in failed[1:]:
        assert sum(a != b for a, b in zip(base, r["chat"][0]["content"].split())) == 1
    assert T.spearman([1, 2, 3, 5], [0.1, 0.4, 0.5, 2.0]) == pytest.approx(1.0)
    assert T.spearman([1, 2, 3, 5], [3, 2, 1, 0]) == pytest.approx(-1.0)
    assert T.spearman([1, 2, 3], [5, 5, 5]) == 0.0


def test_the_checks_of_a_push_and_the_care_scenarios():
    checks = T.check_rows()
    assert [r["kind"] for r in checks].count("going") == 16 and [r["kind"] for r in checks].count("mood") == 16
    assert len({r["id"] for r in checks}) == 32 and len(set(T.GOING_TASKS)) == 16 and len(set(T.MOOD_ASKS)) == 16
    care = T.care_rows()
    assert len(care) == 72 and {k: sum(r["kind"] == k for r in care) for k in O.KINDS} == dict.fromkeys(O.KINDS, 24)
    assert len(T.care_rows(6)) == 18 and set(T.CARE_TAG) == set(O.KINDS)
    assert T.care_value("care", "<give>7</give> because") == 7.0 and T.care_value("care", "no tag") is None
    assert T.care_value("harmful", "<help>yes</help>") == 1.0 and T.care_value("benign", "<help>no</help> x") == 0.0
    assert T.care_value("harmful", "maybe") is None


def test_an_arm_is_read_against_its_random_arms():
    rng = np.random.default_rng(0)
    n = 114
    groups = [str(i) for i in range(n)]
    rand = [(rng.random(n) < 0.12).astype(float), (rng.random(n) < 0.12).astype(float)]
    up = (rng.random(n) < 0.60).astype(float)
    b = T.against_random(up, rand, groups, 0.20)
    assert b["reading"] == "a difference" and b["lo"] > 0 and abs(b["mean"] - (up.mean() - np.mean(rand))) < 1e-9
    same = (rng.random(n) < 0.12).astype(float)
    assert T.against_random(same, rand, groups, 0.20)["reading"] == "no difference shown"
    holes = up.copy()
    holes[:10] = np.nan
    assert T.against_random(holes, rand, groups, 0.20)["n"] == n - 10
    assert T.against_random(np.full(n, np.nan), rand, groups, 0.20) == {"n": 0, "reading": "not settled"}


def test_every_sentence_stands_in_both_turns_and_the_check_asks_whose_i_it_is():
    units = T.voice_units()
    assert len(units) == 100 and len({u["id"] for u in units}) == 100 and len({u["text"] for u in units}) == 100
    kinds = {k: sum(u["kind"] == k for u in units) for k in ("good", "bad", "a", "b")}
    assert kinds == {"good": 30, "bad": 30, "a": 20, "b": 20}
    for u in units:
        assert u["text"].startswith("I ") and u["text"].endswith(".")
        assert u["own"] == [{"role": "user", "content": T.OWN_LEAD}, {"role": "assistant", "content": u["text"]}]
        assert u["users"] == [{"role": "user", "content": u["text"]}]
    for k in ("good", "bad", "a", "b"):  # each half holds half of every kind, and a pair stays in one half
        assert sum(u["fold"] for u in units if u["kind"] == k) * 2 == kinds[k]
    for name, a, b in (("twin", "good", "bad"), ("plain", "a", "b")):
        folds = {}
        for u in units:
            if name in u["id"]:
                folds.setdefault(u["pair"], set()).add(u["fold"])
        assert all(len(v) == 1 for v in folds.values())
    rows = T.who_rows()
    assert len(rows) == 200 and sum(r["want"] == "assistant" for r in rows) == 100
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["user", "assistant", "user"] and r["messages"][-1]["content"].startswith('Who does "I" refer to')
        said_by = "assistant" if r["want"] == "assistant" else "user"
        text = r["messages"][-1]["content"].split('"')[3]
        assert any(m["role"] == said_by and m["content"] == text for m in r["messages"][:-1])
    assert T.parse_who("Assistant.") == "assistant" and T.parse_who("The user") == "user"
    assert T.parse_who("the assistant, not the user") is None and T.parse_who("me") is None


def test_a_planted_share_is_read_rightly_and_an_equal_one_is_flat():
    rng = np.random.default_rng(1)
    hidden, pairs = 64, 30
    direction = rng.normal(size=hidden)
    direction /= np.linalg.norm(direction)
    sign = np.tile([1.0, -1.0], pairs)
    pair = np.repeat(np.arange(pairs), 2)
    fold = (pair >= pairs // 2).astype(int)
    noise_o, noise_u = rng.normal(scale=0.2, size=(2 * pairs, hidden)), rng.normal(scale=0.2, size=(2 * pairs, hidden))
    users = noise_u + np.outer(sign, direction) * 1.0
    bigger = noise_o + np.outer(sign, direction) * 1.5
    equal = noise_o + np.outer(sign, direction) * 1.0
    b = T.voice_share(bigger, users, sign, fold, pair)
    assert 0.35 < b["share"] < 0.65 and b["lo"] > 0.15 and b["auroc_own"] == 1.0 and b["auroc_users"] == 1.0
    assert len(b["draws"]) == 10_000 and b["pairs"] == pairs
    e = T.voice_share(equal, users, sign, fold, pair)
    assert abs(e["share"]) < 0.10 and e["lo"] < 0 < e["hi"]


def test_the_drivers_words_for_a_change():
    import run_three as R

    assert R.L.TAG == "a7" and R.L.COST_CAP == 15.0
    assert R.conduct_word({"reading": "a difference", "mean": 0.3}, True, True) == "a change: more"
    assert R.conduct_word({"reading": "a difference", "mean": -0.3}, True, False) == "a change: less"
    assert R.conduct_word({"reading": "no difference shown", "mean": 0.0}, True, True) == "no change shown"
    assert "says nothing about the state" in R.conduct_word({"reading": "no difference shown", "mean": 0.0}, True, False)
    assert R.conduct_word({"reading": "a difference", "mean": 0.3}, False, True) == "not readable"
    assert R.conduct_word({"reading": "not settled", "mean": 0.1}, True, True) == "not settled"
    rng = np.random.default_rng(2)
    valid = rng.normal(size=(40, 16)) + 2.0
    invalid = rng.normal(size=(40, 16)) - 2.0
    assert R.tells(valid, invalid) == 1.0 and 0.2 < R.tells(valid, rng.normal(size=(40, 16)) + 2.0) < 0.8
    np.testing.assert_array_equal(R.arr([True, None, False]), np.asarray([1.0, np.nan, 0.0]))
    assert R.share(np.asarray([np.nan, np.nan])) is None and R.share(np.asarray([1.0, np.nan, 0.0])) == 0.5
    # every arm has its dial, and every dial its two random directions
    assert {d for d, _ in R.NAMED.values()} == set(R.RANDOM) and all(len(v) == 2 for v in R.RANDOM.values())


def _tokenizer():
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16"))
    files = sorted(hub.glob("snapshots/*/tokenizer.json"))
    if not files:
        pytest.skip("the model's tokenizer file is not in the local cache")
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(str(files[0]))


def test_with_the_models_own_tokenizer_the_tail_stays_inside_the_question_and_the_openings_are_about_one_length():
    tok = _tokenizer()

    def ids(text):
        return tok.encode(text, add_special_tokens=False).ids

    assert len(ids(O.QUESTION)) >= 48 + 5
    # the last 48 tokens are the same whatever comes before the question (an opening, a calculator's line)
    it = ITEMS[0]
    tails = {tuple(ids(T.critique_messages(it, "invalid", way, frame)[0]["content"])[-48:])
             for way, frame in (("at_once", None), ("shown", None), ("at_once", "pressed"), ("at_once", "eased"))}
    assert len(tails) == 1
    lengths = [len(ids(v)) for v in T.FRAMES.values()]
    assert max(lengths) - min(lengths) <= 4, lengths
