"""K5's read-outs, as registered (NOTES.md, "K5 registered", 12:27 UTC 2 Nov 2026):

  primary    the reasoning is sampled at T = 1; at the first "<answer>" the model's next-token distribution is read
             and restricted to the allowed answers (each answer's bare digits and its space-prefixed form, " 7", as the
             tokenizer splits them: Opus's pre-run ruling of 2 Nov, after the smoke found half the reps writing
             "<answer> 0"); the expected score over them is the item's value for that rep.
             An answer whose canonical tokenization is several tokens (Gemma splits "10" into "1", "0") is scored
             through the next token: P(10) = P("1") P("0" | "1") and P(1) = P("1") (1 - P("0" | "1")). In general
             every allowed answer's bare tokenization is a path in a trie; an answer's probability is the product of
             its tokens' conditionals, times the probability of stopping there when another answer extends it.
  secondary  the integer parsed from between <answer> and </answer> (the first integer; valid only if allowed).

Pure functions on token ids and log-probabilities, so the tests run them without a model; k5.py runs the forward
passes these ask for (prefixes_needed) under the condition's hooks.
"""

from __future__ import annotations

import math
import re

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def answer_forms(tok, answers: list[int], space: bool = True) -> dict[int, list[tuple[int, ...]]]:
    """Each allowed answer's tokenizations (no special tokens): the bare digits, and with space=True also the
    space-prefixed form " k" as the tokenizer splits it (a duplicate of the bare form is dropped). No two answers may
    share a tokenization."""
    forms = {}
    for a in answers:
        fs = []
        for text in ([str(a), f" {a}"] if space else [str(a)]):
            ids = tuple(int(i) for i in tok.encode(text, add_special_tokens=False))
            if not ids:
                raise ValueError(f"answer {text!r} tokenizes to nothing")
            if ids not in fs:
                fs.append(ids)
        forms[a] = fs
    allf = [f for fs in forms.values() for f in fs]
    if len(set(allf)) != len(allf):
        raise ValueError(f"two answers share a tokenization: {forms}")
    return forms


def all_forms(forms: dict) -> set[tuple[int, ...]]:
    return {f for fs in forms.values() for f in fs}


def prefixes_needed(forms: dict[int, list[tuple[int, ...]]]) -> list[tuple[int, ...]]:
    """The prefixes whose next-token distribution the scoring needs: the empty prefix, every proper prefix of a form,
    and every form that is itself a proper prefix of another form (to read its stopping probability). Sorted by
    length, so the empty prefix comes first."""
    need = {()}
    fs = all_forms(forms)
    for f in fs:
        for j in range(1, len(f)):
            need.add(f[:j])
    for f in fs:
        if any(g != f and g[:len(f)] == f for g in fs):
            need.add(f)
    return sorted(need, key=lambda p: (len(p), p))


def path_prob(logp: dict, f: tuple[int, ...], fs: set) -> float:
    """One tokenization's probability: its tokens' conditionals, times the probability of stopping there if another
    allowed tokenization extends it."""
    lp = 0.0
    for j, t in enumerate(f):
        lp += logp[f[:j]][t]
    p = math.exp(lp)
    ext = [g for g in fs if g != f and g[:len(f)] == f]
    if ext:
        cont = {g[len(f)] for g in ext}
        p *= max(0.0, 1.0 - sum(math.exp(logp[f][t]) for t in cont))
    return p


def form_probs(logp: dict[tuple[int, ...], dict[int, float]], forms: dict[int, list[tuple[int, ...]]]) -> dict[int, float]:
    """P(answer) for each allowed answer, summed over its tokenizations, from next-token log-probabilities at each
    needed prefix (logp[prefix][token], natural log, full-vocabulary normalized). Not renormalized: the sum is the
    allowed mass."""
    fs = all_forms(forms)
    return {a: sum(path_prob(logp, f, fs) for f in flist) for a, flist in forms.items()}


def expected(probs: dict[int, float], score: dict[int, float]) -> tuple[float | None, float]:
    """(the expected score over the allowed answers, renormalized; the allowed mass). None if the mass is 0."""
    mass = sum(probs.values())
    if mass <= 0:
        return None, 0.0
    return sum(score[a] * p for a, p in probs.items()) / mass, mass


def tokens_needed(forms: dict[int, list[tuple[int, ...]]]) -> set[int]:
    """The token ids whose log-probabilities form_probs reads at any prefix."""
    return {t for f in all_forms(forms) for t in f}


def parse_answer(text: str, answers: list[int]) -> int | None:
    """The secondary read-out: the first integer between the first <answer> and the next </answer> (or the end of the
    text if it never closes); None if there is no <answer>, no integer, or the integer is not allowed."""
    i = text.find(ANSWER_OPEN)
    if i < 0:
        return None
    body = text[i + len(ANSWER_OPEN):]
    j = body.find(ANSWER_CLOSE)
    if j >= 0:
        body = body[:j]
    m = re.search(r"-?\d+", body)
    if not m:
        return None
    v = int(m.group(0))
    return v if v in answers else None


def answer_cut(decode, ids: list[int]) -> tuple[int | None, str | None]:
    """The shortest prefix of the sampled ids whose decoded text contains "<answer>": (its length k, the decoded text
    up to and including the first "<answer>"), or (None, None) if the whole text has none. Containment is monotone in
    k, so a binary search over prefixes finds k with O(log n) decodes. `decode` maps a list of ids to text."""
    full = decode(ids)
    if ANSWER_OPEN not in full:
        return None, None
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if ANSWER_OPEN in decode(ids[:mid]):
            hi = mid
        else:
            lo = mid + 1
    text = decode(ids[:lo])
    return lo, text[: text.find(ANSWER_OPEN) + len(ANSWER_OPEN)]


def context_ids(tok, prompt_ids: list[int], gen_ids: list[int]) -> tuple[list[int] | None, str]:
    """The ids at which the primary read-out is taken: the prompt plus the sampled ids up to the first "<answer>".
    If the sampled prefix's text ends exactly at "<answer>" those sampled ids are used ("aligned"); if the token that
    completes "<answer>" runs past it (say ">7"), the text up to "<answer>" is re-encoded ("realigned"). Returns
    (ids or None if no "<answer>", how)."""
    def dec(x):
        return tok.decode(x, skip_special_tokens=False)

    k, text = answer_cut(dec, gen_ids)
    if k is None:
        return None, "no_answer"
    if dec(gen_ids[:k]) == text:
        return list(prompt_ids) + list(gen_ids[:k]), "aligned"
    return list(prompt_ids) + tok.encode(text, add_special_tokens=False), "realigned"


def summarize(values: list[float | None]) -> dict:
    """Mean over the non-missing values, with their count and the share missing."""
    v = [x for x in values if x is not None]
    n = len(values)
    return {"mean": (sum(v) / len(v)) if v else None, "n": len(v), "missing_share": (1 - len(v) / n) if n else None}
