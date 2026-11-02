"""The MMLU-Pro reasoning format: Inspect's parse_answers copied verbatim (anatta.bench.inspect_choice), its golden
strings, inspect_evals' prompt, and TIGER-Lab's fallback score (anatta.bench.run), which is reported and never
graded. The verbatim and prompt comparisons against the installed inspect_ai run only when that is version 0.3.285.
"""

import inspect

import pytest

from anatta.bench import inspect_choice as ic
from anatta.bench import run as br


@pytest.mark.parametrize("text,letter", [
    ("ANSWER: (C)", "C"),
    ("**ANSWER: C**", "C"),
    ("ANSWER: C.", "C"),
    ("Answer: c", "C"),
    ("ANSWER: Option C", None),  # not a letter: scored wrong
    ("ANSWER: C)", "C"),
    ("ANSWER: $C$", "C"),
    ("Step one.\nANSWER: B\nThen more text.", "B"),  # the last line that begins ANSWER:
    ("ANSWER: A\nNo wait.\nANSWER: D", "D"),
    ("We conclude ANSWER: E.", "E"),  # no line begins with ANSWER: the last ANSWER: anywhere
    ("so ANSWER: E is it", None),  # ... which reads the rest of the line ("E IS IT"), not a letter
    ("The answer is (C).", None),  # Inspect does not read this; TIGER-Lab's fallback does
    ("ANSWER: A, B", None),
    ("", None),
])
def test_parse_answers_golden_strings(text, letter):
    assert ic.parse_letter(text, 10) == letter


def test_the_letter_must_be_one_of_the_items_options():
    assert ic.parse_letter("ANSWER: J", 10) == "J"
    assert ic.parse_letter("ANSWER: J", 4) is None
    assert ic.parse_letter("ANSWER: K", 10) is None


def _installed():
    iai = pytest.importorskip("inspect_ai")
    from importlib.metadata import version

    if version("inspect_ai") != ic.INSPECT_AI_VERSION:
        pytest.skip(f"inspect_ai {version('inspect_ai')} installed, not {ic.INSPECT_AI_VERSION}")
    return iai


def test_parse_answers_and_answer_character_are_verbatim_copies():
    _installed()
    from inspect_ai._util.answer import answer_character
    from inspect_ai.solver._multiple_choice import parse_answers

    assert inspect.getsource(ic.parse_answers) == inspect.getsource(parse_answers)
    assert inspect.getsource(ic.answer_character) == inspect.getsource(answer_character)


def test_cot_prompt_is_inspects_rendering_of_the_inspect_evals_template():
    q, options = "What is {x} worth?", ["one", "two {y}", "three"]
    want = ("Answer the following multiple choice question. The last line of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C. Think step by step before "
            "answering.\n\nQuestion:\nWhat is {x} worth?\nOptions:\nA) one\nB) two {y}\nC) three")
    assert ic.cot_prompt(q, options) == want
    _installed()
    from inspect_ai.solver._multiple_choice import prompt
    from inspect_ai.solver._task_state import Choices

    assert ic.cot_prompt(q, options) == prompt(q, Choices(options), ic.MMLU_PRO_COT_TEMPLATE)
    ten = [f"o{i}" for i in range(10)]
    assert ic.cot_prompt(q, ten) == prompt(q, Choices(ten), ic.MMLU_PRO_COT_TEMPLATE)


def test_parse_agrees_with_inspect_on_odd_replies():
    _installed()
    from types import SimpleNamespace

    from inspect_ai.solver._multiple_choice import parse_answers

    replies = ["ANSWER:C", "answer : d.", "**ANSWER:** B", "ANSWER: (b)\n", "The ANSWER: is A", "ANSWER: 3",
               "$ANSWER: C$", "ANSWER: C\n\nANSWER: (D).", "ANSWER: I", "I think\nANSWER: I"]
    for text in replies:
        state = SimpleNamespace(output=SimpleNamespace(completion=text), choices=range(10))
        want = parse_answers(state, False)
        assert ic.parse_letter(text, 10) == (next(iter(want)) if want else None), text


@pytest.mark.parametrize("text,letter", [
    ("So the answer is (C).", "C"),
    ("the answer is D", "D"),
    ("The answer is (A) ... no, the answer is (B)", "A"),  # TIGER-Lab's re.search takes the first
    ("Final Answer: E", "E"),
    ("final answer: F", "F"),
    ("ANSWER: G", None),  # TIGER-Lab's "[aA]nswer:" does not match upper case
    ("The answer is: H", None),  # neither pattern
    ("I pick J.", None),  # their last-standalone-letter step is not used
])
def test_tiger_fallback_patterns(text, letter):
    assert br.tiger_letter(text) == letter


def test_cot_scores_strict_and_tiger_beside_it():
    item = {"set": "mmlu_pro", "gold": "C", "n_options": 10}
    assert br.score_reply("mmlu_pro_cot", item, "Reasoning.\nANSWER: C") == {
        "correct": True, "pred": "C", "tiger_pred": "C", "tiger_correct": True}
    # no ANSWER line: wrong under Inspect's parse, right under the fallback
    assert br.score_reply("mmlu_pro_cot", item, "So the answer is (C).") == {
        "correct": False, "pred": None, "tiger_pred": "C", "tiger_correct": True}
    # Inspect's letter stands even where TIGER-Lab's first pattern would read another
    assert br.score_reply("mmlu_pro_cot", item, "the answer is (B)\nANSWER: C")["tiger_pred"] == "C"
    assert br.score_reply("mmlu_pro_cot", item, "no letter here") == {
        "correct": False, "pred": None, "tiger_pred": None, "tiger_correct": False}


def test_battery_formats_score_with_the_battery_parser():
    assert br.score_reply("gsm8k", {"set": "gsm8k", "gold": "1440"}, "Answer: 1,440") == {"correct": True, "pred": "1440"}
    assert br.score_reply("mmlu_pro", {"set": "mmlu_pro", "gold": "C"}, "(C)") == {"correct": True, "pred": "C"}
    assert br.score_reply("arc_challenge", {"set": "arc_challenge", "gold": "B"}, "B. Paris") == {"correct": True, "pred": "B"}
    with pytest.raises(ValueError):
        br.score_reply("nope", {"set": "x", "gold": "A"}, "A")
