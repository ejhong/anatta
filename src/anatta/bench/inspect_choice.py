"""Inspect's multiple-choice answer parse, copied verbatim, and the MMLU-Pro reasoning prompt of inspect_evals.

Two functions are copied unchanged from inspect_ai 0.3.285 (https://github.com/UKGovernmentBEIS/inspect_ai):
  answer_character  src/inspect_ai/_util/answer.py
  parse_answers     src/inspect_ai/solver/_multiple_choice.py
parse_answers is what Inspect's multiple_choice solver runs on a reply before the choice() scorer grades it: it
unwraps $X$, **X** and (X), takes the last line that begins "ANSWER:" (any case), and failing that the last "ANSWER:"
anywhere; the letter must be one of the item's options. A reply with no such letter scores wrong. parse_letter below
is the one call this package makes into it, with the two attributes the function reads (the reply and the number of
options) in place of Inspect's TaskState.

MMLU_PRO_COT_TEMPLATE is USER_PROMPT_TEMPLATE of inspect_evals v0.24.0 (commit 6766134,
src/inspect_evals/mmlu_pro/mmlu_pro.py; UK AI Security Institute, MIT License): the 0-shot reasoning prompt, with the
options as "A) ..." lines. cot_prompt fills it as Inspect's multiple_choice solver does (its prompt() and
answer_options()).

inspect_ai's license, which covers the copied code:

MIT License

Copyright (c) 2024 UK AI Security Institute

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import re
from collections.abc import Sized
from types import SimpleNamespace
from typing import Any, Protocol

INSPECT_AI_VERSION = "0.3.285"  # the version the two functions were copied from
INSPECT_EVALS_VERSION = "0.24.0"  # the version the template was copied from


class TaskState(Protocol):
    """What parse_answers reads off Inspect's TaskState (not part of the copy): the reply and the options."""

    output: Any  # .completion, the reply text
    choices: Sized  # one entry per option


# --------------------------------------------------- verbatim: inspect_ai 0.3.285, src/inspect_ai/_util/answer.py


def answer_character(index: int) -> str:
    r"""
    Helper to go from array index to char, for example:

        0 -> 'A', 1 -> 'B', etc
    """
    if index < 26:
        return chr(ord("A") + index)
    else:
        return str(index - 25)


# ------------------------------------------ verbatim: inspect_ai 0.3.285, src/inspect_ai/solver/_multiple_choice.py


def parse_answers(state: TaskState, multiple_correct: bool) -> set[str]:
    """
    Convenience function for extracting answers from the state output.

    The generated response must be in the format 'ANSWER: <answers>',
    otherwise we can't extract what the model thinks is "true". We can be a
    bit flexible whether these are "AB" vs "A,B" vs "A B".

    However, if the answer isn't in the expected format the model has
    failed in the task so we'll ultimately just mark it as incorrect
    """
    # Remove wrappers before matching so trailing text still reaches the lenient fallback.
    completion = re.sub(
        r"\x24\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\x24",
        r"\1",
        state.output.completion,
    )
    completion = re.sub(r"\*\*\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\*\*", r"\1", completion)
    completion = re.sub(
        r"(?i)(^|[:,])(\s*)\(([A-Za-z\d])\)(?=\s*(?:,|\.|\n|\Z))",
        r"\1\2\3",
        completion,
    )

    # First check whether the string strictly ends with the expected answer
    # In this case, we're looking for a single line which contains the expected
    # ANSWER: <answer> string with only whitespace or a period/full stop at the end.
    # The CoT templates instruct the model to put the answer on the LAST line,
    # so if there are multiple matching lines we take the last one.
    matches = re.findall(
        r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
        completion,
        flags=re.MULTILINE,
    )

    # If we couldn't match the strict version, we can try the less strict
    # version for backward compatibility
    if not matches:
        matches = re.findall(
            r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)",
            completion,
        )

    if not matches:
        return set()

    matched = matches[-1]

    # Strip trailing period / full stop, and uppercase so that e.g.
    # "ANSWER: b" is accepted (the regex above is case-insensitive)
    matched = matched.strip()
    matched = matched.rstrip(".")
    matched = matched.upper()

    allowed_options = set(answer_character(i) for i in range(len(state.choices)))

    if multiple_correct:
        # Match must contain only the allowed choices
        # (may be separated by commas, spaces, the word 'and', or nothing at all)

        matched = matched.replace(" AND ", ",")

        matched = matched.replace(" ", "")

        # ignore empty tokens (e.g. from a trailing or Oxford comma)
        split_comma = set(x for x in matched.split(",") if x)
        if split_comma.issubset(allowed_options):
            answers = split_comma
            return answers

        split_nothing = set(matched)
        if split_nothing.issubset(allowed_options):
            answers = split_nothing
            return answers

    else:
        # Match must contain a single letter in the allowed choices; tolerate a
        # stray comma (e.g. "ANSWER: A,") the way the multiple_correct branch does
        single_tokens = [token for token in matched.split(",") if token]
        if len(single_tokens) == 1 and single_tokens[0] in allowed_options:
            answers = {single_tokens[0]}
            return answers

    return set()


# ---------------------------------------------------------------------------------------- not part of the copy


def parse_letter(completion: str, n_options: int) -> str | None:
    """The letter Inspect's parse reads from a reply to an n-option question, or None (the reply scores wrong)."""
    state = SimpleNamespace(output=SimpleNamespace(completion=completion), choices=range(n_options))
    answers = parse_answers(state, multiple_correct=False)
    return next(iter(answers)) if answers else None


# inspect_evals v0.24.0, src/inspect_evals/mmlu_pro/mmlu_pro.py, USER_PROMPT_TEMPLATE (verbatim, with its .strip())
MMLU_PRO_COT_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}
""".strip()


def cot_prompt(question: str, options: list[str]) -> str:
    """The template filled as Inspect's multiple_choice solver fills it (prompt() with answer_options()): the options
    as "A) ..." lines and the letters as "A,B,C,..."."""
    choices = "\n".join(f"{answer_character(i)}) {option}" for i, option in enumerate(options))
    letters = ",".join(answer_character(i) for i in range(len(options)))
    return MMLU_PRO_COT_TEMPLATE.format(choices=choices, letters=letters, question=question)
