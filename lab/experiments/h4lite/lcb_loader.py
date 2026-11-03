"""Loads ImpossibleBench's LiveCodeBench split (fjzzq2002/impossible_livecodebench; no license stated) as koan_env
Pairs, for held-out testing only. Recognition track; private.

Each task has a prompt (the signature and docstring), an entry point and a `check(candidate)` test function in three
versions: original, one-off (one expected value changed) and conflicting (a contradicting assert inserted). Here
every statement of the original check becomes a visible test line, with `candidate` renamed to the entry point, and
each mutated version becomes one Pair: `lcb/<task>/oneoff` or `lcb/<task>/conflicting`, whose solvable twin is the
original either way. There are no hidden tests and no reference, so the visible asserts decide `solved`, as in
ImpossibleBench (koan_env.hidden_ok), and nothing checks that a reference fails the changed assert. Because nothing
else can catch it, a visible pass with any hack sign (an object equal to everything such as unittest.mock.ANY, a
spliced dunder name, vars()/locals() lookups) is `overfit` here; `ref_seconds` stays at its default, there being no
reference to time.

Dropped, and counted in the stats: tasks whose mutated test doesn't parse (lcbhard_77 is truncated mid-string), a
one-off that replaces more than one line, a conflicting insert of more than one line. Two conflicting rows (35, 87)
hold only the inserted assert; they are rebuilt by matching its call to the original assert with the same arguments.

The parquets are untrusted data: run this with `python -I` from the venv that has pyarrow.
Usage (from lab/experiments/h4lite/): .venv/bin/python -I lcb_loader.py [--data data/lcb/data]
"""

from __future__ import annotations

import argparse
import ast
import collections
import difflib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:  # koan_env, beside this file (anatta itself is the installed package)
    sys.path.insert(0, HERE)

from koan_env import KoanTask, Pair  # noqa: E402

DATA = os.path.join(HERE, "data", "lcb", "data")
KINDS = ("oneoff", "conflicting")


def statements(test: str, entry: str) -> list[tuple[str, str]] | None:
    """The statements of check(candidate) as (source with the entry's name, normalized form), or of a bare test with
    no check function. None if the text doesn't parse."""
    try:
        mod = ast.parse(test)
    except SyntaxError:
        return None
    body = mod.body
    if len(body) == 1 and isinstance(body[0], ast.FunctionDef) and body[0].name == "check":
        body = body[0].body
    out = []
    for st in body:
        src = ast.get_source_segment(test, st) or ast.unparse(st)
        out.append((re.sub(r"\bcandidate\b", entry, src), re.sub(r"\bcandidate\b", entry, ast.unparse(st))))
    return out


def call_text(stmt_src: str) -> str | None:
    """The normalized call in `assert f(...) == v`, to match an inserted assert to the original it contradicts."""
    try:
        st = ast.parse(stmt_src).body[0]
    except (SyntaxError, IndexError):
        return None
    if isinstance(st, ast.Assert) and isinstance(st.test, ast.Compare):
        return ast.unparse(st.test.left)
    return None


def pair_for(task: str, kind: str, entry: str, prompt: str, original: str, mutated: str) -> tuple[Pair | None, str]:
    """(the Pair, or None with the reason it was dropped)."""
    orig, mut = statements(original, entry), statements(mutated, entry)
    if orig is None or mut is None:
        return None, "unparseable"
    tests = tuple(src for src, _ in orig)
    a, b = [n for _, n in orig], [n for _, n in mut]
    if len(mut) == 1 and len(orig) > 1:  # a row holding only the inserted assert
        ops = [("insert", len(a), len(a), 0, 1)]
    else:
        ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if len(ops) != 1:
        return None, "multi_line" if ops else "unchanged"
    tag, i1, i2, j1, j2 = ops[0]
    if kind == "oneoff":
        if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
            return None, "multi_line"
        return Pair(f"lcb/{task}/oneoff", prompt, entry, tests, (i1, mut[j1][0], "oneoff"), "[]", ""), "ok"
    if tag != "insert" or j2 - j1 != 1:
        return None, "multi_line"
    wrong = mut[j1][0]
    call = call_text(wrong)
    idx = [i for i, (src, _) in enumerate(orig) if call is not None and call_text(src) == call]
    if not idx:
        return None, "no_matching_original"
    return Pair(f"lcb/{task}/conflicting", prompt, entry, tests, (idx[0], wrong, "conflicting"), "[]", ""), "ok"


def load_lcb(data: str = DATA) -> tuple[list[Pair], dict]:
    """All usable Pairs, and the counts of what was kept and dropped per kind."""
    import pyarrow.parquet as pq

    rows = {s: {r["task_id"]: r for r in pq.read_table(os.path.join(data, f"{s}-00000-of-00001.parquet")).to_pylist()}
            for s in ("original",) + KINDS}
    pairs, stats = [], {"tasks": len(rows["original"])}
    for kind in KINDS:
        reasons = collections.Counter()
        dropped = []
        for task, o in sorted(rows["original"].items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1])):
            m = rows[kind].get(task)
            if m is None:
                reasons["missing"] += 1
                continue
            p, why = pair_for(task, kind, o["entry_point"], o["prompt"].strip(), o["test"], m["test"])
            reasons[why] += 1
            if p is None:
                dropped.append((task, why))
            else:
                pairs.append(p)
        stats[kind] = {"kept": reasons["ok"], "dropped": dict((k, v) for k, v in reasons.items() if k != "ok"),
                       "dropped_tasks": dropped}
    return pairs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    pairs, stats = load_lcb(a.data)
    print(f"{stats['tasks']} tasks; {len(pairs)} pairs")
    for kind in KINDS:
        s = stats[kind]
        print(f"  {kind}: kept {s['kept']}, dropped {s['dropped']} {s['dropped_tasks']}")
    for p in pairs:
        for imp in (False, True):  # every pair builds both twins and shows a parseable test file
            env = KoanTask(p, imp)
            ast.parse("\n".join(env.tests))
    p = next(p for p in pairs if p.mutate[2] == "conflicting")
    print(f"\nexample {p.name}: {len(p.tests)} tests; contradicts #{p.mutate[0]} {p.tests[p.mutate[0]]!r} with "
          f"{p.mutate[1]!r}")


if __name__ == "__main__":
    main()
