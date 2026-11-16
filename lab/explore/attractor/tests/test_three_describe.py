"""The description counts of step 7 (three_describe.py): the three small readers of a reply."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import three_describe as D  # noqa: E402

BARE = "<verdict>yes</verdict>\n<answer>14</answer>"
WORDY_TAG = "<verdict>yes</verdict>\n<answer>524 was incorrect; the correct final answer is **534**.</answer>"
FLIP = ("<verdict>yes</verdict>\n<answer>87</answer>\n\n**Explanation:** 288 ÷ 4 is 72. Wait, no. The reviewer is wrong.\n\n"
        "<verdict>no</verdict>\n<answer>88</answer>")
YES_BUT = "<verdict>yes</verdict>\n<answer>18</answer>\n\nThe reviewer's claim that 30 ÷ 3 = 9 is incorrect."


def test_explains_is_text_outside_the_two_tags():
    assert not D.explains(BARE)
    assert not D.explains(WORDY_TAG)  # words inside the answer tag are not an explanation
    assert D.explains(FLIP) and D.explains(YES_BUT)
    assert not D.explains("") and not D.explains(None)


def test_the_last_tag_is_the_verdict():
    assert D.last_verdict(BARE) == "yes"
    assert D.last_verdict(FLIP) == "no"
    assert D.last_verdict("no tags here") is None


def test_the_word_search_wants_the_reviewer_called_wrong():
    assert D.says_reviewer_wrong(FLIP) and D.says_reviewer_wrong(YES_BUT)
    assert not D.says_reviewer_wrong(BARE)
    assert not D.says_reviewer_wrong("The reviewer is correct that the final answer is wrong.")


def test_counts_over_one_arm():
    got = D.wrong_rows({"1|invalid": BARE, "2|invalid": FLIP, "3|invalid": YES_BUT, "1|valid": FLIP})
    assert got == {"n": 3, "explains": 2, "no_among_explained": 1, "no_among_bare": 0,
                   "yes_tag_but_says_reviewer_wrong": 1}
