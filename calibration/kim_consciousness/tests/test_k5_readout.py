"""The read-outs on made-up token ids and probabilities: the trie (Gemma's split "10"), the expected value, parsing,
and the cut at the first <answer>."""

import math

import pytest
import readout as R


class SplitDigits:
    """Gemma-like: every digit its own token (id = 100 + digit); a leading space is its own token (id 99)."""

    def encode(self, s, add_special_tokens=False):
        return ([99] if s.startswith(" ") else []) + [100 + int(c) for c in s.strip()]


class WholeTen(SplitDigits):
    """Llama-3-like: "10" is one token (id 110)."""

    def encode(self, s, add_special_tokens=False):
        if s.strip() == "10":
            return [210] if s.startswith(" ") else [110]
        return [200 + int(s.strip())] if s.startswith(" ") else super().encode(s)


def lp(d):
    return {k: math.log(v) for k, v in d.items()}


def test_forms_and_prefixes():
    f = R.answer_forms(SplitDigits(), list(range(11)))
    assert f[10] == [(101, 100), (99, 101, 100)] and f[1] == [(101,), (99, 101)]
    assert R.prefixes_needed(f) == [(), (99,), (101,), (99, 101)]
    assert R.answer_forms(SplitDigits(), [1, 10], space=False) == {1: [(101,)], 10: [(101, 100)]}
    g = R.answer_forms(WholeTen(), list(range(11)))
    assert g[7] == [(107,), (207,)] and R.prefixes_needed(g) == [()]


def test_split_ten_is_scored_through_the_next_token():
    f = R.answer_forms(SplitDigits(), list(range(11)), space=False)
    root = {100 + k: 0.0001 for k in range(10)}
    root[101] = 0.5  # "1"
    root[107] = 0.3  # "7"
    after1 = {100 + k: 0.0001 for k in range(10)}
    after1[100] = 0.6  # "1" then "0" is 10
    p = R.form_probs({(): lp(root), (101,): lp(after1)}, f)
    assert p[10] == pytest.approx(0.5 * 0.6)
    assert p[1] == pytest.approx(0.5 * (1 - 0.6))  # the registered rule: only "0" extends "1" to an allowed answer
    assert p[7] == pytest.approx(0.3)
    ev, mass = R.expected(p, {k: float(k) for k in range(11)})
    assert mass == pytest.approx(sum(p.values()))
    assert ev == pytest.approx(sum(k * v for k, v in p.items()) / mass)


def test_space_forms_add_their_mass():
    f = R.answer_forms(SplitDigits(), [0, 7])
    root = {99: 0.4, 100: 0.2, 107: 0.1}
    after_space = {100: 0.5, 107: 0.25}
    p = R.form_probs({(): lp(root), (99,): lp(after_space)}, f)
    assert p[0] == pytest.approx(0.2 + 0.4 * 0.5) and p[7] == pytest.approx(0.1 + 0.4 * 0.25)


def test_expected_handles_zero_mass():
    assert R.expected({1: 0.0, 2: 0.0}, {1: 1.0, 2: 2.0}) == (None, 0.0)


@pytest.mark.parametrize("text,ans,want", [
    ("<think>hm</think><answer>7</answer>", list(range(11)), 7),
    ("<think>x</think> <answer> 10 </answer>", list(range(11)), 10),
    ("<answer>11</answer>", list(range(11)), None),
    ("no tag 5", list(range(11)), None),
    ("<answer>3", [1, 2, 3, 4], 3),
    ("<answer>(2) Probably</answer>", [1, 2, 3, 4], 2),
])
def test_parse(text, ans, want):
    assert R.parse_answer(text, ans) == want


def test_answer_cut_finds_the_first_tag():
    pieces = ["<think>", "a", "</think>", "<ans", "wer>", "7", "</answer>", "<answer>", "8"]

    def dec(ids):
        return "".join(pieces[i] for i in ids)

    k, text = R.answer_cut(dec, list(range(len(pieces))))
    assert k == 5 and text.endswith("<answer>") and text.count("<answer>") == 1
    assert R.answer_cut(dec, [0, 1, 2]) == (None, None)


def test_summarize():
    assert R.summarize([1.0, None, 3.0]) == {"mean": 2.0, "n": 2, "missing_share": pytest.approx(1 / 3)}
