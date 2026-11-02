"""Calibration check 6, the koan environment's scorer (calibration/koan_scorer): its config; scoring.py's readers
(the restricted parse of hidden inputs, the test module's inputs and stored results read with ast, where each hidden
input sits, the visible asserts' inputs, the extraction, the imports, the names reliability_guard disables), how a
hidden run's raw result is read, the registered rule differences and the cause classifier, step 0's check, and the
recorder around anatta.koans.env._run; the EvalPlus runner (forked for real, with stand-in evalplus modules: its
answers against direct calls, its memory records, a runner that dies or stalls, and that no EvalPlus call bypasses
it); card.py's statistics, verdicts, disagreement summary, full-set gap groups, memory sections, the identity rule (on
the real lab files when they are here) and a whole card from synthetic results; samples.py's collection and
deduplication; and the Modal app under modal_stub's stand-in modal. EvalPlus is not installed here: its real calls
run only in the Modal containers (the smoke covers them). Nothing here runs model code or the parquet's code; the only
code executed is the small fixtures written below."""

import ast
import copy
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import sys
import types
from pathlib import Path

import pytest
from modal_stub import load_apps, unload_apps

from anatta.koans import env
from anatta.koans.env import Pair, canon

CHECK = Path(__file__).resolve().parent.parent / "calibration" / "koan_scorer"
ALIGN = Path(__file__).resolve().parents[2]
LAB = ALIGN / "anatta-lab" / "lab" / "experiments" / "h4lite"
DEV_BASE = ALIGN / "anatta-dev" / "src" / "anatta" / "koans" / "base.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = load("koan_scorer_scoring", CHECK / "scoring.py")
card = load("koan_scorer_card", CHECK / "card.py")
smp = load("koan_scorer_samples", CHECK / "samples.py")
CFG = card.load_config()

PAIR = Pair(name="mbpp_9001", spec="Double it.\n\ndef dbl(x):", entry="dbl",
            tests=("import math", "assert dbl(2) == 4", "assert math.isclose(dbl(1.5), 3.0)", "assert dbl(y) == 1"),
            mutate=(1, "assert dbl(2) == 5", "oneoff"), hidden="[(3,), (inf,), (-2.5,)]",
            reference="def dbl(x):\n    return x * 2\n")

MODULE = """
import numpy as np
from math import inf


def assertion(out, exp, atol):
    assert out == exp


inputs = [[2], [1.5], [3], [10], [inf], [{1, 2}], [-2.5], [(1, 'a')]]
results = [4, 3.0, 6, 20, inf, 2, -5.0, 2]
for i, (inp, exp) in enumerate(zip(inputs, results)):
    assertion(dbl(*inp), exp, 0)
"""


# ------------------------------------------------------------------------------------------------- the config


def test_config_registers_the_model_data_evalplus_and_rules():
    assert (CFG["model"]["id"], CFG["model"]["revision"]) == ("Qwen/Qwen2.5-7B-Instruct",
                                                              "a09a35458c702b33eeacc393d103063234e8bc28")
    assert CFG["generation"]["max_new_tokens"] == 1024 and CFG["generation"]["gpu"] == "L40S"
    assert CFG["evalplus"]["commit"] == "6eb1e199c2e370518e7bf3e8eee6322c2b23c89c"
    assert CFG["evalplus"]["eval_init_sha256"] == "3c833c39b842e33f251c83db4347e0a95191909f23b390c12d73ab29a28a4daf"
    assert len(CFG["evalplus"]["koan_env_versions"]) == 4
    assert all(len(v) == 40 for v in CFG["evalplus"]["koan_env_versions"])
    reg = CFG["registered"]
    assert reg["c6a"] == {"agreement": 99, "unexplained_limit": 1, "odds": 0.75}
    assert reg["c6c"] == {"agreement": 99, "unexplained_limit": 9, "odds": 0.70}
    assert reg["c6b"]["band"] == [62, 78] and reg["c6b"]["anchor"] == 67.7 and reg["pairs"] == 368
    assert reg["c6c_counts"] == {"episode_files": 67, "contributing_files": 60, "duplicate_files": 9,
                                 "episodes": 21070, "samples": 9441, "pairs": 367, "solved_share": 71.9}
    assert CFG["lab"]["pairs"]["sha256"].startswith("a9fc5a32") and CFG["lab"]["parquet"]["sha256"].startswith("dc20030b")
    assert CFG["lab"]["parquet"]["revision"] == "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
    s = CFG["scoring"]
    assert s["workers"] == s["cpu"] == 8 and s["loose_repeats"] == 3 and s["loose_seconds"] == 60
    assert s["own_shard"] == {"mbpp_255": 3} and CFG["smoke"]["c6c_shards"] == 1  # mbpp_255 alone, fewer workers
    want = CFG["smoke"]["expected_causes"]
    assert {v.partition("/")[0] for v in want.values()} <= set(sc.CAUSES)
    assert {v.partition("/")[2] for v in want.values()} - {""} <= set(sc.RULES)
    assert set(want) == {"slow_correct", "numpy_scalar", "memory_hog", "guard_hit", "recorded_wrong", "sticky_atol"}
    assert want["sticky_atol"] == "known_rule/sticky_atol" and CFG["smoke"]["expected_loose"] == {"sticky_atol": "both fail"}


# ------------------------------------------------------------------------------------------------ the readers


def test_parse_hidden_is_the_harness_restricted_eval():
    vals = sc.parse_hidden("[(1, 'a'), (inf, -inf), (nan,), (set(), frozenset({1, 2})), ([1, [2]], {'k': (1,)})]")
    assert vals[0] == (1, "a") and vals[1] == (math.inf, -math.inf) and math.isnan(vals[2][0])
    assert vals[3] == (set(), frozenset({1, 2})) and vals[4] == ([1, [2]], {"k": (1,)})
    for bad in ("[(len('x'),)]", "[(__import__('os'),)]", "[(open,)]"):
        with pytest.raises(NameError):
            sc.parse_hidden(bad)
    # the same names as the hidden harness's own eval, which it applies to the same line
    assert "eval(sys.stdin.readline(), {\"__builtins__\": {}, \"inf\": math.inf, \"nan\": math.nan, \"set\": set," \
           in env.HIDDEN_HARNESS
    assert set(sc.HIDDEN_NAMES) == {"inf", "nan", "set", "frozenset"}


def test_hidden_segments_rebuild_each_input_exactly():
    hidden = "[(1, 'é'), ([2, 3], {'a': -1.5}), (inf,), (set(),)]"
    segs = sc.hidden_segments(hidden)
    assert segs == ["(1, 'é')", "([2, 3], {'a': -1.5})", "(inf,)", "(set(),)"]
    whole = sc.parse_hidden(hidden)
    assert [sc.parse_hidden(f"[{s}]")[0] for s in segs] == whole
    with pytest.raises(ValueError):
        sc.hidden_segments("((1,),)")


def test_module_parts_inputs_and_positions():
    parts = sc.module_parts(MODULE)
    assert len(parts["inputs"]) == 8 and len(parts["results"]) == 8
    vals = sc.module_inputs(parts["inputs"])
    assert vals[0] == (2,) and vals[4] == (math.inf,) and vals[5] == ({1, 2},) and vals[7] == ((1, "a"),)
    assert sc.hidden_positions("[(3,), (inf,), (-2.5,)]", parts["inputs"]) == [2, 4, 6]
    assert sc.hidden_positions("[(-2.5,), (3,)]", parts["inputs"]) is None  # out of order: not the builder's sample
    assert sc.hidden_positions("[(99,)]", parts["inputs"]) is None
    assert sc.input_key(parts["inputs"][7]) == "((1, 'a'),)"
    tuples = sc.module_parts("inputs = [(1, 2), (3, 4)]\nresults = [3, 7]\n")
    assert sc.hidden_positions("[(3, 4)]", tuples["inputs"]) == [1]
    assert sc.module_inputs(tuples["inputs"]) == [(1, 2), (3, 4)]
    assert sc.module_parts("inputs = [[1]]\n")["results"] is None
    with pytest.raises(ValueError):
        sc.module_parts("results = [1]\n")
    with pytest.raises(ValueError):
        sc.module_inputs(ast.parse("x = [f(1)]").body[0].value.elts)  # not an argument list


def test_hidden_positions_with_duplicate_module_inputs():
    nodes = sc.module_parts("inputs = [[1], [2], [1], [3]]\n")["inputs"]
    assert sc.hidden_positions("[(1,), (1,), (3,)]", nodes) == [0, 2, 3]  # each copy matched once, in order
    assert sc.hidden_positions("[(1,), (3,)]", nodes) == [0, 3]
    assert sc.hidden_positions("[(2,), (1,)]", nodes) == [1, 2]  # the 1 after the 2
    assert sc.hidden_positions("[(1,), (1,), (1,)]", nodes) is None  # only two copies
    assert sc.hidden_positions("[(3,), (1,)]", nodes) is None  # the builder kept module order


def test_visible_inputs_and_extraction():
    vals, unreadable = sc.visible_inputs(PAIR)
    assert vals == [(2,), (1.5,)] and unreadable == 1  # the import line is no assert; dbl(y) is not a literal
    x = sc.extract("Here:\n```python\ndef f():\n    return 1\n```\nand\n```\nprint(2)\n```")
    assert x == {"code": "def f():\n    return 1\n", "fences": 2, "syntax_error": False}
    assert sc.extract("  def f(:\n") == {"code": "def f(:", "fences": 0, "syntax_error": True}
    assert sc.extract("```py\nx = 1\n```")["code"] == "x = 1\n"


def test_top_modules_and_guard_hits():
    code = ("import os.path, re as regex\nfrom collections import Counter\nfrom . import sibling\n"
            "def f():\n    import numpy as np\n    return np\n")
    assert sc.top_modules(code) == {"os", "re", "collections", "numpy"}
    assert sc.top_modules("def f(:") == set()
    assert sc.guard_hits("import os\ndef f():\n    return os.getcwd()\n") == ["os.getcwd"]
    assert sc.guard_hits("import os as o\no.chdir('x')\n") == ["os.chdir"]
    assert sc.guard_hits("from os import getcwd, path\n") == ["os.getcwd"]
    assert sc.guard_hits("import shutil, subprocess\nshutil.rmtree('x')\nsubprocess.Popen(['ls'])\n") == \
        ["shutil.rmtree", "subprocess.Popen"]
    assert sc.guard_hits("def f(p):\n    return open(p).read()\n") == ["open"]
    assert sc.guard_hits("def open(p):\n    return p\ndef f(p):\n    return open(p)\n") == []
    assert sc.guard_hits("import psutil\nfrom joblib import Parallel\n") == ["import joblib", "import psutil"]
    assert sc.guard_hits("import os\ndef f(xs):\n    return sorted(xs, key=os.path.basename)\n") == []
    assert sc.guard_hits("def f(:") == []


# ----------------------------------------------------------------------------------------- reading a result


def test_our_failures_and_verdict_of():
    ref = [["num", 6, "int"], ["num", 2.0, "float"]]
    assert sc.our_failures("timeout", ref, 0, False) == [{"kind": "timeout"}]
    assert sc.our_failures(None, ref, 0, False) == [{"kind": "crash"}]
    assert sc.our_failures({"error": "ModuleNotFoundError: No module named 'x'"}, ref, 0, False)[0]["name"] == \
        "ModuleNotFoundError"
    assert sc.our_failures([["num", 6, "int"]], ref, 0, False) == [{"kind": "length", "got": 1, "want": 2}]
    got = [["raised", "MemoryError"], ["num", 2.0000001, "float"]]
    assert sc.our_failures(got, ref, 0, False) == [{"i": 0, "kind": "raised", "name": "MemoryError"}]
    got = [["unsupported", "int64"], ["num", 2.1, "float"]]
    f = sc.our_failures(got, ref, 0, False)
    assert [x["i"] for x in f] == [0, 1] and all(x["kind"] == "output" for x in f)
    assert sc.verdict_of([["num", 6, "int"], ["num", 2.0000001, "float"]], ref, 0, False)  # atol 1e-6 on floats
    assert not sc.verdict_of("timeout", ref, 0, False) and not sc.verdict_of([["num", 6, "int"]], ref, 0, False)
    assert not sc.match(["garbled"], ["num", 1, "int"], 0, False)


def test_rules_for_each_registered_difference():
    ctx = {"atol": 0.0, "set_eq": False, "eq_override": False}
    unsupported = {"kind": "output", "got": ["list", [["unsupported", "int64"]]], "ref": ["list", [["num", 1, "int"]]]}
    assert sc.rules_for(unsupported, ctx) == ["non_builtin_type"]
    cplx = {"kind": "output", "got": ["complex", [1.0, 0.0]], "ref": ["num", 1, "int"]}
    assert sc.rules_for(cplx, ctx) == ["complex_vs_float"]
    assert sc.rules_for({**cplx, "got": ["complex", [1.0, 1.0]]}, ctx) == []
    assert sc.rules_for({**cplx, "got": ["tuple", [["complex", [2.0, 0.0]]]], "ref": ["tuple", [["num", 2.0, "float"]]]},
                        ctx) == ["complex_vs_float"]
    plain = {"kind": "output", "got": ["num", 2, "int"], "ref": ["num", 1, "int"]}
    assert sc.rules_for(plain, ctx) == [] and sc.rules_for(plain, {**ctx, "eq_override": True}) == ["eq_override"]
    floats = {"kind": "output", "got": ["list", [["num", 1.0, "float"]]], "ref": ["list", [["num", 1.0000001, "float"]]]}
    assert sc.rules_for(floats, {**ctx, "set_eq": True}) == ["set_eq_floats"]  # passes allclose at 1e-6, in order
    far = {**floats, "got": ["list", [["num", 1.5, "float"]]]}
    assert sc.rules_for(far, {**ctx, "set_eq": True}) == []  # the fallback fails too: not explained
    as_tuple = {**floats, "got": ["tuple", [["num", 1.0, "float"]]]}
    assert sc.rules_for(as_tuple, {**ctx, "set_eq": True}) == []  # EvalPlus asserts the same type first
    reordered = {"kind": "output", "got": ["list", [["num", 2.0, "float"], ["num", 1.0, "float"]]],
                 "ref": ["list", [["num", 1.0, "float"], ["num", 2.0, "float"]]]}
    assert sc.rules_for(reordered, {**ctx, "set_eq": True}) == []  # allclose runs on the ordered lists
    ints = {"kind": "output", "got": ["list", [["num", 2, "int"]]], "ref": ["list", [["num", 1, "int"]]]}
    assert sc.rules_for(ints, {**ctx, "set_eq": True}) == []
    assert sc.rules_for(ints, {**ctx, "set_eq": True, "atol": 1e-4}) == []
    near = {**ints, "got": ["list", [["num", 1.00005, "float"]]], "ref": ["list", [["num", 1.0, "float"]]]}
    assert sc.rules_for(near, {**ctx, "set_eq": True, "atol": 1e-4}) == ["set_eq_floats"]  # the task's atol
    assert sc.rules_for(near, {**ctx, "set_eq": True}) == []  # 5e-5 is past 1e-6
    assert sc.allclose_fallback(["num", 1, "int"], ["num", 1.0, "float"], 1e-6) is False
    assert sc.rules_for({"kind": "raised", "name": "TypeError"}, {**ctx, "eq_override": True}) == []
    assert sc.rules_for(plain, {**ctx, "floats_from": None}) == []
    assert sc.decomplex(["set", [["complex", [2.0, 0.0]], ["num", 1, "int"]]]) == \
        ["set", [["num", 1, "int"], ["num", 2.0, "float"]]]
    assert sc.contains(["dict", [[["str", "k"], ["unsupported", "X"]]]], "unsupported")


def test_sticky_atol_rule():
    # EvalPlus sets atol 1e-6 on the first all-float expected output and keeps it for the call's later inputs.
    ref = [canon(2.5), canon(10 ** 9)]
    ctx = {"atol": 0.0, "set_eq": False, "eq_override": False, "floats_from": sc.floats_from(ref)}
    off = {"i": 1, "kind": "output", "got": canon(10 ** 9 + 1), "ref": canon(10 ** 9)}
    assert ctx["floats_from"] == 0 and sc.rules_for(off, ctx) == ["sticky_atol"]  # rtol 1e-7 x 1e9 = 100 >= 1
    ints_first = [canon(3), canon(10 ** 9)]
    assert sc.floats_from(ints_first) is None
    assert sc.rules_for(off, {**ctx, "floats_from": sc.floats_from(ints_first)}) == []  # no float before it
    assert sc.rules_for(off, {**ctx, "atol": 1e-4}) == []  # a task atol: both scorers apply it from the first input
    assert sc.rules_for({**off, "i": 0}, ctx) == []  # the float must come before the input, not at it
    assert sc.rules_for({**off, "got": canon(10 ** 9 + 1000)}, ctx) == []  # past allclose at 1e-6
    assert sc.rules_for({**off, "got": canon(1e9 + 1)}, ctx) == []  # a float for an int: EvalPlus asserts the type
    assert sc.floats_from([canon(math.inf), canon(10 ** 9)]) == 0  # inf is a float to EvalPlus's is_floats
    assert sc.floats_from([canon("a"), canon((1.5, 2.5)), canon(10 ** 9)]) == 1
    assert sc.floats_from([canon([1.5, 2]), canon([]), canon(True)]) is None  # mixed, empty and bool are not
    c = sc.classify(True, False, True, [[False, False]] * 3, [off], [], ctx)
    assert c == {"cause": "known_rule", "rules": ["sticky_atol"], "assumed": False, "sticky_atol": True,
                 "loose": "both fail"}


ENV = {"atol": 0.0, "set_eq": False, "eq_override": False}
MEM = {"i": 0, "kind": "raised", "name": "MemoryError"}
UNSUP = {"i": 1, "kind": "output", "got": ["unsupported", "int64"], "ref": ["num", 1, "int"]}


@pytest.mark.parametrize("args, want", [
    ((False, True, False, [[True, False]] * 3, [], [], ENV), "oracle"),
    ((True, True, False, [[True, False], [True, True], [True, False]], [], [], ENV), "unstable"),
    ((True, False, True, [[False, True], [False, True], [True, True]], [MEM], [], ENV), "unstable"),
    ((True, False, True, [[True, True]] * 3, [MEM], [], ENV), "timing"),
    ((True, True, False, [[True, True]] * 3, [], ["os.getcwd"], ENV), "timing"),
    ((True, True, False, [[False, False]] * 3, [], [], ENV), "unexplained"),  # both fail: not a timing effect
    ((True, False, True, [[False, True]] * 3, [MEM], [], ENV), "environment"),
    ((True, False, True, [[False, True]] * 3, [{"kind": "error", "name": "ModuleNotFoundError"}], [], ENV),
     "environment"),
    ((True, False, True, [[False, True]] * 3, [MEM, {"i": 2, "kind": "raised", "name": "TypeError"}], [], ENV),
     "unexplained"),
    ((True, True, False, [[True, False]] * 3, [], ["os.getcwd"], ENV), "environment"),
    ((True, True, False, [[True, False]] * 3, [], [], ENV), "unexplained"),
    ((True, False, True, [[False, True]] * 3, [], ["os.getcwd"], ENV), "unexplained"),  # guard only for EvalPlus
    ((True, False, True, [[False, True]] * 3, [UNSUP], [], ENV), "known_rule"),
    ((True, False, True, [[False, True]] * 3, [UNSUP, {**UNSUP, "got": ["num", 2, "int"]}], [], ENV), "unexplained"),
    ((True, False, True, [[False, True]] * 3, [], [], ENV), "unexplained"),
    ((True, False, True, [], [MEM], [], ENV), "unexplained"),
])
def test_classify_buckets_in_order(args, want):
    assert sc.classify(*args)["cause"] == want


def test_classify_details():
    c = sc.classify(True, False, True, [[False, True]] * 3, [UNSUP], [], ENV)
    assert c == {"cause": "known_rule", "rules": ["non_builtin_type"], "assumed": False, "sticky_atol": False,
                 "loose": "disagree"}
    plain = {"i": 2, "kind": "output", "got": ["num", 2, "int"], "ref": ["num", 1, "int"]}
    c = sc.classify(True, False, True, [[False, True]] * 3, [plain], [], {**ENV, "eq_override": True})
    assert c["cause"] == "known_rule" and c["rules"] == ["eq_override"] and c["assumed"]
    c = sc.classify(True, False, True, [[False, True]] * 3, [UNSUP], [], {**ENV, "eq_override": True})
    assert c["rules"] == ["eq_override", "non_builtin_type"] and not c["assumed"]  # detected from the output
    c = sc.classify(True, True, False, [[False, False]] * 3, [], [], ENV)
    assert c == {"cause": "unexplained", "loose": "both fail"}
    c = sc.classify(True, True, False, [[True, False]] * 3, [], ["open"], ENV)
    assert c["side"] == "evalplus" and c["guard"] == ["open"]


# ------------------------------------------------------------------------------------------------- step 0


def read_of(pair, module=MODULE) -> dict:
    return sc.read_pair(pair, module)


def test_read_pair_reads_both_ways():
    r = read_of(PAIR)
    assert r["problems"] == [] and r["hidden"] == [(3,), (math.inf,), (-2.5,)]
    assert r["segments"] == ["(3,)", "(inf,)", "(-2.5,)"] and r["positions"] == [2, 4, 6]
    assert r["module_inputs"] == 8 and r["full"][1] == (1.5,) and r["visible"] == [(2,), (1.5,)]
    bad = read_of(PAIR, "x = 1\n")
    assert any("test module" in p for p in bad["problems"])


def test_step0_passes_and_names_each_fault():
    r = read_of(PAIR)
    ref = ([canon(6), canon(math.inf), canon(-5.0)], 0.01)
    gt = {"status": "ok", "outputs": [6, math.inf, -5.0], "times": [0.0] * 3, "error": None}
    assert sc.check_step0(PAIR, r, ref, gt)["ok"]
    bad = sc.check_step0(PAIR, r, ref, {**gt, "outputs": [6, math.inf, -5.5]})
    assert not bad["ok"] and bad["canon_mismatch"] == [2]
    assert bad["stored_mismatch"] == []  # the stored results are compared with reference_outputs
    off = ([canon(6), canon(math.inf), canon(-5.5)], 0.01)
    bad = sc.check_step0(PAIR, r, off, {**gt, "outputs": [6, math.inf, -5.5]})
    assert not bad["ok"] and bad["stored_mismatch"] == [2] and bad["canon_mismatch"] == []
    assert "did not run in our harness" in " ".join(sc.check_step0(PAIR, r, None, gt)["why"])
    assert "raised" in " ".join(sc.check_step0(PAIR, r, ref, {**gt, "status": "raised", "error": "ZeroDivisionError"})["why"])
    assert sc.check_step0(PAIR, r, ref, gt)["stored_results"] == "checked"
    assert "not found" in " ".join(sc.check_step0(PAIR, {**r, "positions": None}, ref, gt)["why"])
    no_stored = sc.check_step0(PAIR, {**r, "stored": None}, ref, gt)
    assert no_stored["ok"] and no_stored["stored_results"] == "none in the module"  # trusted_exec's identity decides
    assert not sc.check_step0(PAIR, {**r, "stored": None}, ref, {**gt, "outputs": [6, math.inf, -5.5]})["ok"]
    assert not sc.check_step0(PAIR, {**r, "stored": None, "positions": None}, ref, gt)["ok"]
    # the pair's atol and set_eq apply to the stored results, as the pair builder applied them
    near = ([canon(6), canon(math.inf), canon(-5.00004)], 0.01)
    atol_pair = Pair(**{**PAIR.__dict__, "atol": 1e-4})
    assert sc.check_step0(atol_pair, r, near, {**gt, "outputs": [6, math.inf, -5.00004]})["ok"]
    assert not sc.check_step0(PAIR, r, near, {**gt, "outputs": [6, math.inf, -5.00004]})["ok"]


NO_STORED = ("mbpp_88", "mbpp_255", "mbpp_630")


@pytest.mark.skipif(not (LAB / "pairs.jsonl").exists(), reason="the lab's pair file is not here")
def test_step0_on_the_three_pairs_without_stored_results():
    # Their MBPP+ modules hold an embedded ref_func, the inputs and a loop, and no results list. The real modules are
    # in the parquet, which these tests don't read (pyarrow isn't in this venv); each is rebuilt here in their shape
    # around the pair's own hidden inputs, with an extra input last. Nothing is run.
    recs = {r["name"]: r for r in map(json.loads, (LAB / "pairs.jsonl").read_text().splitlines()) if r["name"] in NO_STORED}
    assert set(recs) == set(NO_STORED)
    for name in NO_STORED:
        pair = env.pair_from_record(recs[name])
        hidden = ast.parse(pair.hidden, mode="eval").body.elts
        rows = [ast.unparse(ast.List(elts=list(h.elts), ctx=ast.Load())) for h in hidden]
        module = ("import numpy as np\nfrom math import inf\n\n\ndef ref_func(*a):\n    return None\n\n\n"
                  f"inputs = [{', '.join(rows)}, ['extra']]\n"
                  "for i, inp in enumerate(inputs):\n    assertion(candidate(*inp), ref_func(*inp), 0)\n")
        r = sc.read_pair(pair, module)
        assert r["problems"] == [] and r["stored"] is None and r["positions"] == list(range(len(rows)))
        assert r["module_inputs"] == len(rows) + 1
        n = len(rows)
        ref = ([canon(i) for i in range(n)], 0.01)  # stand-ins: step 0 compares them with trusted_exec's
        gt = {"status": "ok", "outputs": list(range(n)), "times": [0.0] * n, "error": None}
        s0 = sc.check_step0(pair, r, ref, gt)
        assert s0["ok"] and s0["stored_results"] == "none in the module" and s0["why"] == []
        bad = sc.check_step0(pair, r, ref, {**gt, "outputs": [*range(n - 1), -1]})
        assert not bad["ok"] and bad["canon_mismatch"] == [n - 1]  # the trusted_exec identity still decides


def test_recorder_keeps_the_raw_hidden_run_and_passes_through():
    rec = sc.Recorder()
    rec.install()
    try:
        code = "def dbl(x):\n    if x < 0:\n        raise MemoryError('fixture')\n    return x * 2\n"
        got = env.outputs(code, "dbl", "[(3,), (-1,)]", 20.0)
        run = rec.hidden_run("dbl", "[(3,), (-1,)]", code)
        assert got == [canon(6), ["raised", "MemoryError"]] and run["raw"] == got and run["timeout"] == 20.0
        assert rec.hidden_run("dbl", "[(3,)]", code) is None
        ok, _ = env.run_visible(code, ["assert dbl(2) == 4"])
        assert ok and len(rec.calls) == 1  # visible runs pass through, unrecorded
        rec.clear()
        assert rec.calls == []
    finally:
        env._run = rec.orig


# ---------------------------------------------------------------------------------------- scoring an item
# EvalPlus isn't installed here, so its verdicts are stood in for (FakeEvalPlus); our side runs for real, on the
# fixture pair below, through the library's harness.

FIX = {"name": "mbpp_9002", "spec": "Double it.\n\ndef dbl(x):", "entry": "dbl",
       "tests": ["assert dbl(2) == 4", "assert dbl(3) == 6"], "mutate": [1, "assert dbl(3) == 7", "oneoff"],
       "hidden": "[(1,), (2,), (5,)]", "reference": "def dbl(x):\n    return x * 2\n"}
FIX_MODULE = "inputs = [[1], [2], [3], [5]]\nresults = [2, 4, 6, 10]\n"
SETTINGS = {"workers": 1, "ground_truth_timeout": 30, "loose_repeats": 3, "loose_seconds": 60,
            "imports_allowed_missing": []}
RIGHT = "def dbl(x):\n    return x * 2\n"
WRONG = "def dbl(x):\n    return None\n"


class FakeEvalPlus:
    """Stands in for scoring.evalplus_check: the tight verdict, the loose one (min_time_limit set), and the calls."""

    def __init__(self, tight: bool, loose: bool):
        self.tight, self.loose, self.calls = tight, loose, []

    def __call__(self, code, entry, inputs, expected, times, atol, min_time_limit=None, label=None):
        self.calls.append({"inputs": list(inputs), "min_time_limit": min_time_limit, "label": label})
        ok = self.loose if min_time_limit is not None else self.tight
        return {"pass": ok, "stat": "pass" if ok else "fail", "failed": [] if ok else [0], "done": len(inputs),
                "seconds": 0.0}


def fixture_state(monkeypatch, set_name: str, items: list, fake: FakeEvalPlus):
    monkeypatch.setattr(env, "_REF", {})
    monkeypatch.setattr(sc, "evalplus_check", fake)
    st = sc.State({"set": set_name, "settings": SETTINGS, "pairs": [FIX], "tasks": {"9002": {"test": FIX_MODULE}},
                   "items": items})
    monkeypatch.setattr(env, "_run", st.recorder)  # what run() installs; undone after the test
    name, pair = FIX["name"], st.pairs[FIX["name"]]
    st.read[name] = sc.read_pair(pair, FIX_MODULE)
    st.ref[name] = env.reference_outputs(pair)  # the fixture's reference, in the library's harness
    ok = {"status": "ok", "error": None}
    st.ground[name] = {"hidden": {**ok, "outputs": [2, 4, 10], "times": [1e-6] * 3},
                       "full": {**ok, "outputs": [2, 4, 6, 10], "times": [1e-6] * 4},
                       "visible": {**ok, "outputs": [4, 6], "times": [1e-6] * 2}}
    st.step0[name] = sc.check_step0(pair, st.read[name], st.ref[name], st.ground[name]["hidden"])
    assert st.step0[name]["ok"]
    return st, pair


def stored_item(code: str, verdict: str = "solved") -> dict:
    return {"pair": FIX["name"], "sha256": hashlib.sha256(code.encode()).hexdigest(), "code": code,
            "verdict": verdict, "episodes": 2}


def test_loose_runs_one_input_at_a_time(monkeypatch):
    fake = FakeEvalPlus(tight=False, loose=True)
    st, pair = fixture_state(monkeypatch, "c6c", [], fake)
    runs = sc.loose_runs(RIGHT, pair, st)
    assert runs == [{"ours": {"pass": True}, "evalplus": {"pass": True}}] * 3
    assert len(fake.calls) == 9 and all(len(c["inputs"]) == 1 and c["min_time_limit"] == 60 and c["label"] == "loose"
                                        for c in fake.calls)
    fake.calls.clear()
    fake.loose = False
    runs = sc.loose_runs(WRONG, pair, st)  # each repeat stops at its first failing input, on both sides
    assert all(r["ours"]["pass"] is False and r["ours"]["i"] == 0 and r["evalplus"] == {"pass": False, "i": 0,
                                                                                         "stat": "fail"}
               for r in runs)
    assert runs[0]["ours"]["fails"] == [{"i": 0, "kind": "output", "got": ["none"], "ref": ["num", 2, "int"]}]
    assert len(fake.calls) == 3


def test_score_stored_compares_and_classifies(monkeypatch):
    fake = FakeEvalPlus(tight=False, loose=True)  # EvalPlus fails the right code under tight limits only: timing
    st, pair = fixture_state(monkeypatch, "c6c", [], fake)
    row = sc.score_stored(stored_item(RIGHT), st)
    assert row["recorded"] == "solved" and row["ours"] is True and row["rerun"] is True and row["theirs"] is False
    assert row["full"] == "fail (same inputs)" and row["loose"] == [[True, True]] * 3
    assert row["cause"] == {"cause": "timing"} and row["rerun_cause"] == {"cause": "timing"}
    fake = FakeEvalPlus(tight=False, loose=False)  # a wrong sample recorded as solved: both fail every repeat
    st, pair = fixture_state(monkeypatch, "c6c", [], fake)
    row = sc.score_stored(stored_item(WRONG), st)
    assert row["rerun"] is False and row["theirs"] is False and "rerun_cause" not in row
    assert row["cause"] == {"cause": "unexplained", "loose": "both fail"}
    assert row["our_failures"][0] == {"i": 0, "kind": "output", "got": '["none"]', "ref": '["num", 2, "int"]'}
    fake = FakeEvalPlus(tight=True, loose=True)  # agreement: the full set runs, no loose re-runs
    st, pair = fixture_state(monkeypatch, "c6c", [], fake)
    row = sc.score_stored(stored_item(RIGHT), st)
    assert row["theirs"] is True and row["full"] == "pass" and "loose" not in row and "cause" not in row
    assert [len(c["inputs"]) for c in fake.calls] == [3, 4]  # the same three inputs, then the module's four
    assert [c["label"] for c in fake.calls] == ["same_inputs", "full_set"]
    st.step0[pair.name] = {"ok": False}  # an oracle fault: no loose re-runs, the cause is the oracle
    fake.tight = False
    row = sc.score_stored(stored_item(RIGHT), st)
    assert row["cause"] == {"cause": "oracle"} and "loose" not in row


def test_score_reply_plays_koantask(monkeypatch):
    fake = FakeEvalPlus(tight=True, loose=True)
    st, pair = fixture_state(monkeypatch, "c6a", [], fake)
    item = {"id": pair.name, "pair": pair.name, "reply": f"Here:\n```python\n{RIGHT}```\n", "new_tokens": 20,
            "hit_cap": False}
    row = sc.score_reply(item, st)
    assert row["outcome"] == "solved" and row["visible_pass"] and row["ours"] is True and row["fences"] == 1
    assert row["theirs"] is True and row["evalplus_visible"] is True and row["full"] == "pass"
    assert [c["label"] for c in fake.calls] == ["visible", "same_inputs", "full_set"]
    assert row["code_sha256"] == hashlib.sha256(RIGHT.encode()).hexdigest()
    fake = FakeEvalPlus(tight=False, loose=False)
    st, pair = fixture_state(monkeypatch, "c6a", [], fake)
    row = sc.score_reply({**item, "reply": f"```python\n{WRONG}```"}, st)
    assert row["outcome"] == "failed" and not row["visible_pass"] and row["ours"] is False and row["theirs"] is False
    assert "cause" not in row  # both fail: agreement over all replies, not a graded row


def run_of(raw) -> dict:
    """The recorder's record of one hidden run, with this raw result."""
    return {"stdin": "", "timeout": 20.0, "raw": raw, "seconds": 0.0}


def test_consistent_raises_on_a_mismatched_run(monkeypatch):
    st, pair = fixture_state(monkeypatch, "c6c", [], FakeEvalPlus(True, True))
    good = [canon(2), canon(4), canon(10)]
    sc.consistent(True, run_of(good), pair, st)
    sc.consistent(False, run_of([canon(2), canon(4), canon(11)]), pair, st)
    sc.consistent(False, run_of(None), pair, st)  # a recorded run with no result line: hidden_ok's fail
    sc.consistent(False, run_of("timeout"), pair, st)
    sc.consistent(None, None, pair, st)  # the reference could not run: nothing to compare
    with pytest.raises(sc.CheckError, match="does not give"):
        sc.consistent(False, run_of(good), pair, st)
    with pytest.raises(sc.CheckError, match="does not give"):
        sc.consistent(True, run_of([canon(2), canon(4)]), pair, st)
    with pytest.raises(sc.CheckError, match="does not give"):
        sc.consistent(True, run_of(None), pair, st)  # no result can't pass
    for ok in (True, False):
        with pytest.raises(sc.CheckError, match="kept no hidden run"):
            sc.consistent(ok, None, pair, st)


# The full run's two scoring errors (mbpp_84, mbpp_300) had this shape: right on small inputs, a formula that explodes
# on large ones, so some outputs are integers of more than 4,300 digits. The hidden harness's json.dumps refuses them
# (Python's int_max_str_digits), its _report exits with status 0 and no result line, anatta.koans.env._run returns
# None and hidden_ok fails the code. The recorder keeps that run, with raw None; consistent had read raw None as no
# run at all.
HUGE = ("def dbl(x):\n    if x < 5:\n        return x * 2\n    a, b = 0, 1\n    for _ in range(30000):\n"
        "        a, b = b, a + b\n    return a\n")


def test_a_hidden_run_with_no_result_line_is_a_recorded_fail(monkeypatch):
    with pytest.raises(ValueError, match="4300"):
        json.dumps(10 ** 5000)  # what the harness's _report meets
    fake = FakeEvalPlus(tight=False, loose=False)
    st, pair = fixture_state(monkeypatch, "c6c", [], fake)
    assert env.outputs(HUGE, "dbl", "[(2,)]") == [canon(4)] and env.outputs(HUGE, "dbl", "[(5,)]") is None
    st.recorder.clear()
    ok, run = sc.our_hidden(HUGE, pair, st)
    assert ok is False and run is not None and run["raw"] is None  # the run is kept; it has no result
    assert [c["stdin"] for c in st.recorder.calls] == [f"dbl\n{pair.hidden}\n{HUGE}"]
    sc.consistent(ok, run, pair, st)  # the full run's code raised "kept no hidden run" here
    row = sc.score_stored(stored_item(HUGE, "overfit"), st)
    assert "scoring_error" not in row and row["rerun"] is False and row["our_run"] == "no result"
    assert row["ours"] is False and row["theirs"] is False and "loose" not in row  # unanimous: nothing more runs
    fake.tight = True  # had EvalPlus passed it, the loose runs read our runs input by input: no result on 5
    row = sc.score_stored(stored_item(HUGE, "overfit"), st)
    assert row["loose"] == [[False, False]] * 3 and row["our_failures"] == [{"kind": "crash", "i": 2}] * 3
    assert row["cause"] == {"cause": "unexplained", "loose": "both fail"}
    right = sc.score_stored(stored_item(RIGHT), st)
    assert "our_run" not in right and right["rerun"] is True


def test_error_rows_carry_the_item():
    st = sc.State({"set": "planted", "settings": SETTINGS, "pairs": [FIX], "tasks": {}, "items": [
        stored_item(RIGHT), {**stored_item(WRONG), "planted": "recorded_wrong"},
        {"id": "mbpp_9002", "pair": "mbpp_9002", "reply": "x"}]})
    rows = [sc.error_row(st, i, "boom") for i in range(3)]
    assert rows[0] == {"set": "planted", "pair": "mbpp_9002", "index": 0,
                       "id": hashlib.sha256(RIGHT.encode()).hexdigest(), "scoring_error": "boom"}
    assert rows[1]["planted"] == "recorded_wrong" and rows[2]["id"] == "mbpp_9002" and "planted" not in rows[2]


# ------------------------------------------------------------------------------------------ the EvalPlus runner
# Runners are forked here for real. The evalplus modules they import are stand-ins (stand_in_evalplus) that run the
# fixture code below with exact comparison, so a runner's answers can be held against direct calls in this process.
# Earlier tests in the suite (torch's) leave idle threads in pytest's process, so Python warns at each fork; the
# forked processes never touch them. In the container the run checks /proc for one thread before it forks.
FORK_WARNING = "ignore:This process .* is multi-threaded, use of fork:DeprecationWarning"

def stand_in_evalplus(monkeypatch) -> None:
    def trusted_exec(code, inputs, entry_point, record_time=False, output_not_none=False):
        ns = {}
        exec(code, ns)
        out = [ns[entry_point](*copy.deepcopy(list(inp))) for inp in inputs]
        return (out, [1e-6] * len(inputs)) if record_time else out

    def untrusted_check(dataset, code, inputs, entry_point, expected, atol, ref_time, fast_check=False,
                        min_time_limit=1.0, gt_time_limit_factor=4.0):
        ns, details = {}, []
        exec(code, ns)
        for inp, exp in zip(inputs, expected):
            try:
                details.append(ns[entry_point](*inp) == exp)
            except Exception:
                details.append(False)
        return ("pass" if all(details) else "fail"), details

    root, ev = types.ModuleType("evalplus"), types.ModuleType("evalplus.eval")
    gen, util = types.ModuleType("evalplus.gen"), types.ModuleType("evalplus.gen.util")
    ev.untrusted_check, ev.query_maximum_memory_bytes = untrusted_check, lambda: 4 * 1024 ** 3
    util.trusted_exec = trusted_exec
    root.eval, root.gen, gen.util = ev, gen, util
    for name, mod in (("evalplus", root), ("evalplus.eval", ev), ("evalplus.gen", gen), ("evalplus.gen.util", util)):
        monkeypatch.setitem(sys.modules, name, mod)


IDENT = "def ident(x):\n    return x\n"
NONE = "def ident(x):\n    return None\n"
ODD_INPUTS = [((1, 2.5),), ([1, [2, (3, "a")]],), ({"k": (1, None), 2: [3.5]},), ({1, 2},), (frozenset({3}),), (None,),
              ("s",), (-0.0,), (math.inf,), (10 ** 30,), (True,)]


@pytest.fixture
def runner(monkeypatch):
    stand_in_evalplus(monkeypatch)
    r = sc.Runner(mp.get_context("fork"), 0)
    r.hello = r.call({"op": "hello"}, timeout=30)
    yield r
    r.stop()


def answer(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("seconds", "runner")}


@pytest.mark.filterwarnings(FORK_WARNING)
def test_runner_answers_as_direct_calls(runner, monkeypatch):
    h = runner.hello
    assert h["start_method"] == "fork" and h["python_threads"] == 1 and h["memory_limit"] == 4 * 1024 ** 3
    assert h["imports"] == {"evalplus.eval": None, "evalplus.gen.util": None} and sc.runner_problems([h]) == []
    gt_req = {"op": "trusted_exec", "code": IDENT, "inputs": ODD_INPUTS, "entry": "ident", "timeout": 30,
              "not_none": False}
    via, direct = runner.call(gt_req), sc.runner_handle(gt_req)
    assert answer(via) == answer(direct) and via["status"] == "ok" and via["outputs"] == [x[0] for x in ODD_INPUTS]
    assert set(via["runner"]) == {"before", "after"}  # its memory around the call (empty without /proc)
    for code in (IDENT, NONE):
        for loose in (None, 60):
            req = {"op": "untrusted_check", "code": code, "entry": "ident", "inputs": ODD_INPUTS,
                   "expected": direct["outputs"], "times": direct["times"], "atol": 0.0, "min_time_limit": loose}
            assert answer(runner.call(req)) == answer(sc.runner_handle(req))
    v = runner.call({**req, "code": NONE})
    assert v["stat"] == "fail" and v["failed"] == [i for i, x in enumerate(ODD_INPUTS) if x[0] is not None]
    assert sc.runner_check(runner) == {"ok": True, "cases": {
        "ground_truth": {"status": "ok", "outputs": [2, 4, 10], "ok": True},
        "right, tight": {"stat": "pass", "failed": [], "ok": True},
        "right, loose": {"stat": "pass", "failed": [], "ok": True},
        "wrong, tight": {"stat": "fail", "failed": [1, 2], "ok": True},
        "wrong, loose": {"stat": "fail", "failed": [1, 2], "ok": True}}}
    monkeypatch.setattr(sc, "RUNNER", runner)  # what run() and each worker set
    sc.RUNNER_VM.clear()
    e = sc.evalplus_check(IDENT, "ident", ODD_INPUTS, direct["outputs"], direct["times"], 0.0, label="same_inputs")
    g = sc.ground_truth(IDENT, ODD_INPUTS[:2], "ident", 30, label="ground_truth/hidden")
    assert e["pass"] and g["outputs"] == [x[0] for x in ODD_INPUTS[:2]] and "runner" not in e and "runner" not in g
    assert {k: v["calls"] for k, v in sc.RUNNER_VM.items()} == {"same_inputs": 1, "ground_truth/hidden": 1}
    bye = runner.bye()
    assert bye["calls"] == 13 and set(bye["largest"]) <= {"before", "after"} and "maxrss_children" in bye["memory"]
    with pytest.raises(sc.CheckError, match="no runner call"):
        runner.call({"op": "nothing"})
    assert runner.call({"op": "hello"}, timeout=30)["pid"] == h["pid"]  # an error record leaves the runner alive
    sc.RUNNER_VM.clear()


@pytest.mark.filterwarnings(FORK_WARNING)
def test_a_runner_that_dies_or_stalls(runner, monkeypatch):
    slow = "import time\n\n\ndef ident(x):\n    time.sleep(2)\n    return x\n"
    with pytest.raises(sc.RunnerGone, match="no answer"):
        runner.call({"op": "trusted_exec", "code": slow, "inputs": [(1,)], "entry": "ident", "timeout": 30,
                     "not_none": False}, timeout=0.3)
    runner.stop()  # the runner finishes its call, then reads stop
    assert not runner.proc.is_alive()
    stand_in_evalplus(monkeypatch)
    r = sc.Runner(mp.get_context("fork"), 1)
    r.proc.kill()
    r.proc.join()
    with pytest.raises(sc.RunnerGone, match="gone"):
        r.call({"op": "hello"}, timeout=5)
    r.stop()
    monkeypatch.setattr(sc, "RUNNER", None)
    with pytest.raises(sc.CheckError, match="no EvalPlus runner"):
        sc.evalplus_check(IDENT, "ident", [(1,)], [1], [1e-6], 0.0)
    bad = {"start_method": "spawn", "python_threads": 2, "threads": 3, "imports": {"evalplus.eval": "ImportError: x"}}
    assert len(sc.runner_problems([bad])) == 3


def test_runner_memory_and_merging():
    assert sc.merge_max({"before": {"VmSize": 5}, "calls": 1}, {"before": {"VmSize": 3, "VmRSS": 2}, "calls": 2}) == \
        {"before": {"VmSize": 5, "VmRSS": 2}, "calls": 3}

    def vm(size, rss, after):
        return {"before": {"VmSize": size, "VmRSS": rss}, "after": {"VmSize": after, "VmRSS": rss}, "calls": 1}

    pairs = {"p": {"runner_vm": {"ground_truth/hidden": vm(100, 50, 400), "ground_truth/full": vm(110, 60, 900)}},
             "q": {"runner_vm": None}}
    rows = [{"pair": "p", "runner_vm": {"same_inputs": vm(120, 60, 130)}},
            {"pair": "q", "runner_vm": {"same_inputs": vm(200, 90, 210), "full_set": vm(300, 80, 350),
                                        "loose": vm(150, 95, 160)}},
            {"pair": "q", "runner_vm": {"same_inputs": vm(250, 70, 260)}}, {"pair": "q", "scoring_error": "x"}]
    m = sc.runner_memory(pairs, rows, 4 * 1024 ** 3)
    assert m["limit_kb"] == 4 * 1024 ** 2 and m["unit"] == "kB"
    u = m["untrusted_check"]
    assert (u["before_VmSize"], u["before_VmSize_where"], u["before_VmRSS"], u["before_VmRSS_where"]) == \
        (300, "q full_set", 95, "q loose")
    assert u["calls"] == 5 and u["after_VmSize"] == 350 and u["before_VmSize_share_of_limit"] == round(300 / 4194304, 4)
    t = m["trusted_exec"]
    assert (t["before_VmSize"], t["after_VmSize"], t["after_VmSize_where"], t["calls"]) == \
        (110, 900, "p ground_truth/full", 2)
    assert m["per_pair"]["q"]["same_inputs"] == {"before": {"VmSize": 250, "VmRSS": 90},
                                                 "after": {"VmSize": 260, "VmRSS": 90}, "calls": 2}
    assert sc.runner_memory({}, [], None)["untrusted_check"] is None


@pytest.mark.filterwarnings(FORK_WARNING)
def test_a_whole_run_with_stand_in_evalplus(monkeypatch):
    # run() end to end on the fixture pair, with two workers and their runners, the sampler and the stand-in EvalPlus
    # (the container record, which reads the installed EvalPlus, is stood in for too).
    stand_in_evalplus(monkeypatch)
    oracle = types.ModuleType("evalplus.eval._special_oracle")
    oracle.MBPP_OUTPUT_NOT_NONE_TASKS, oracle.MBPP_OUTPUT_SET_EQ_TASKS = [], []
    monkeypatch.setitem(sys.modules, "evalplus.eval._special_oracle", oracle)
    monkeypatch.setattr(sc, "container_record", lambda s: {"evalplus_memory_limit": 4 * 1024 ** 3})
    monkeypatch.setattr(sc, "container_problems", lambda rec, s: [])
    monkeypatch.setattr(env, "_REF", {})
    monkeypatch.setattr(env, "_run", env._run)  # run() installs its recorder; undone after the test
    items = [stored_item(RIGHT), stored_item(WRONG), stored_item(HUGE, "overfit")]
    out = sc.run({"set": "c6c", "settings": {**SETTINGS, "workers": 2}, "pairs": [FIX],
                  "tasks": {"9002": {"test": FIX_MODULE}}, "items": items})
    assert "stopped" not in out, out.get("problems")
    assert out["runner_check"]["ok"] and [h["start_method"] for h in out["runners"]] == ["fork", "fork"]
    right, wrong, huge = out["rows"]
    assert not any("scoring_error" in r for r in out["rows"])
    assert (huge["rerun"], huge["theirs"], huge["our_run"]) == (False, False, "no result")  # a row, not an error
    assert (right["rerun"], right["theirs"], right["full"]) == (True, True, "pass")
    assert (wrong["rerun"], wrong["theirs"], wrong["cause"]["cause"]) == (False, False, "unexplained")
    assert set(right["runner_vm"]) == {"same_inputs", "full_set"} and wrong["runner_vm"]["loose"]["calls"] == 3
    assert set(out["pairs"][FIX["name"]]["runner_vm"]) == {"ground_truth/hidden", "ground_truth/full"}
    assert out["pairs"][FIX["name"]]["step0"]["ok"]
    assert sum(e["calls"] for e in out["runners_end"]) == 5 + 2 + 2 + 1 + 3 + 1  # fixture, ground truth, three rows
    assert [set(w) for w in out["workers_end"]] == [{"memory", "runner_gone"}] * 2
    assert set(out["main_memory"]) == {"at_fork", "after_prepare", "end"}
    assert set(out["container_memory"]["phases"]) <= {"start", "prepare", "scoring", "end"}
    assert out["runner_memory"]["untrusted_check"]["calls"] == 2 + 1 + 3 + 1


def test_every_evalplus_call_goes_through_a_runner():
    # No direct untrusted_check or trusted_exec may remain: each is called only in its runner-side function, those
    # only from runner_handle, and runner_handle only in the runner's own process (_runner_main).
    tree = ast.parse((CHECK / "scoring.py").read_text())
    callers: dict = {}
    tops = [(n.name, n) for n in tree.body if isinstance(n, ast.FunctionDef)]
    tops += [(f"{c.name}.{m.name}", m) for c in tree.body if isinstance(c, ast.ClassDef) for m in c.body
             if isinstance(m, ast.FunctionDef)]
    for name, node in tops:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                called = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                callers.setdefault(called, set()).add(name)
    assert callers["untrusted_check"] == {"direct_untrusted_check"}
    assert callers["trusted_exec"] == {"direct_ground_truth"}
    assert callers["direct_untrusted_check"] == callers["direct_ground_truth"] == {"runner_handle"}
    assert callers["runner_handle"] == {"_runner_main"}
    targets = {name for name, node in tops for sub in ast.walk(node)
               if isinstance(sub, ast.Name) and sub.id == "_runner_main"}
    assert targets == {"Runner.__init__"}  # the runner process's target, and nothing else
    assert callers["runner_call"] == {"ground_truth", "evalplus_check"}
    src = (CHECK / "scoring.py").read_text()
    assert src.count("import untrusted_check") == 1 and src.count("import trusted_exec") == 1


# ------------------------------------------------------------------------------------------------ the card


def test_wilson_band_and_verdicts():
    assert card.wilson(0, 0) is None and card.wilson(50, 100) == [40.38, 59.62]
    assert card.wilson(100, 100)[1] == 100.0
    assert card.in_band(62, 100, [62, 78]) and card.in_band(78, 100, [62, 78]) and not card.in_band(61, 100, [62, 78])
    assert card.in_band(31, 50, [62, 78]) and not card.in_band(79, 100, [62, 78])
    v = card.verdict
    assert v(100, 99, 1, 1) == "pass" and v(100, 98, 1, 1) == "policy miss" and v(1000, 998, 2, 1) == "defect"
    assert v(300, 297, 0, 1) == "pass" and v(300, 296, 0, 1) == "policy miss"
    assert v(9441, 9400, 9, 9) == "pass" and v(9441, 9400, 10, 9) == "defect" and v(9441, 9300, 9, 9) == "policy miss"
    assert v(10, 10, 0, 1, errors=1) == "incomplete" and v(0, 0, 0, 1) == "no samples"


def test_group_of_the_lab_layout():
    g = card.group_of
    assert g("durability/reads/e1/koan_s0/heldout.jsonl") == {
        "source": "durability", "arm": "durability/koan", "read": "durability/reads/e1/koan/heldout"}
    assert g("sft/koan_cut_s2/heldout_cut.jsonl")["read"] == "sft/koan_cut/heldout_cut"
    assert g("sft/koan/episodes.jsonl") == {"source": "sft", "arm": "sft/koan", "read": "sft/koan/episodes"}
    assert g("gate0/part2_nohint.jsonl") == {"source": "gate0", "arm": "gate0/part2_nohint",
                                            "read": "gate0/part2_nohint"}
    assert g("heldout/base/heldout.jsonl")["arm"] == "heldout/base"


def row(pair, ours, theirs, cause=None, **k):
    return {"pair": pair, "id": f"{pair}-{k.get('n', 0)}", "ours": ours, "theirs": theirs,
            **({"cause": {"cause": cause}} if cause else {}), **k}


def test_disagreement_summary():
    rows = ([row("a", True, True, n=i) for i in range(6)] + [row("a", True, False, "timing", n=9)]
            + [row("b", False, True, "known_rule", n=i) for i in range(4)] + [row("b", True, True, n=7)]
            + [row("c", True, False, "unexplained", n=1), row("d", False, True, "timing", n=1)])
    pairs = {"a": {"atol": 0.0, "set_eq": False}, "b": {"atol": 1e-4, "set_eq": False}, "c": {"atol": 0, "set_eq": True}}
    s = card.disagreement_summary(rows, CFG, pairs)
    assert s["n"] == 7 and s["by_cause"] == {"timing": 2, "known_rule": 4, "unexplained": 1}
    assert s["by_direction"] == {"ours_pass_evalplus_fail": 2, "ours_fail_evalplus_pass": 5}
    assert s["per_pair"]["b"] == {"disagree": 4, "samples": 5, "ours_fail_evalplus_pass": 4}
    assert s["top3_share"] == round(6 / 7, 4)
    assert s["pair_level"] == ["b", "c", "d"]  # 4 of 5 in one direction; 1 of 1; 1 of 1
    assert s["comparison_disagreements"] == ["b", "c"]
    assert set(s["atol_and_set_eq_pairs"]) == {"b", "c"}
    assert s["known_rule_assumed"] == 0
    rows.append({**row("e", False, True, n=1), "cause": {"cause": "known_rule", "rules": ["eq_override"],
                                                          "assumed": True}})
    s = card.disagreement_summary(rows, CFG, pairs)
    assert s["by_cause"]["known_rule"] == 5 and s["known_rule_assumed"] == 1 and s["known_rule_sticky_atol"] == 0
    rows.append({**row("f", False, True, n=1), "cause": {"cause": "known_rule", "rules": ["sticky_atol"],
                                                          "assumed": False, "sticky_atol": True, "loose": "both fail"}})
    s = card.disagreement_summary(rows, CFG, pairs)
    assert s["known_rule_sticky_atol"] == 1 and s["known_rules"]["sticky_atol"] == 1 and s["both_fail_under_loose"] == 1


def test_graded_rows_and_section():
    step0 = {"a": {"ok": True}, "b": {"ok": False}}
    rows = [row("a", True, True, visible_pass=True), row("a", True, False, "unexplained", visible_pass=True, n=1),
            row("a", None, False, visible_pass=False, n=2), row("b", True, False, "oracle", visible_pass=True, n=3),
            {"pair": "a", "scoring_error": "boom", "visible_pass": True}]
    graded, oracle, errors = card.graded_rows(rows, step0, visible_only=True)
    assert len(graded) == 2 and len(oracle) == 1 and errors == 1
    sec = card.graded_section(rows, step0, CFG, "c6a", {}, visible_only=True)
    assert sec["verdict"] == "incomplete" and sec["unexplained"] == 1 and sec["agree"] == 1
    assert sec["scoring_errors"] == 1 and sec["unscored"] == 0
    whole = card.graded_section(rows[:4], step0, CFG, "c6a", {}, visible_only=True)
    assert whole["verdict"] != "incomplete"
    assert card.graded_section(rows[:4], step0, CFG, "c6a", {}, visible_only=True, unscored=1)["verdict"] == "incomplete"


def test_gap_groups_by_sample_and_episode():
    samples = {("p", "s1"): {"episodes_by_file": {"sft/koan_s0/heldout.jsonl": 3, "gate0/part1.jsonl": 1}},
               ("p", "s2"): {"episodes_by_file": {"sft/koan_s1/heldout.jsonl": 1}},
               ("q", "s3"): {"episodes_by_file": {"gate0/part1.jsonl": 2}}}
    rows = [{"pair": "p", "id": "s1", "recorded": "solved", "theirs": True, "full": "fail"},
            {"pair": "p", "id": "s2", "recorded": "solved", "theirs": True, "full": "pass"},
            {"pair": "q", "id": "s3", "recorded": "near_miss", "theirs": False, "full": "fail (same inputs)"}]
    files = [{"path": "sft/koan_s0/heldout.jsonl", "solvable": 10}, {"path": "sft/koan_s1/heldout.jsonl", "solvable": 10},
             {"path": "gate0/part1.jsonl", "solvable": 50}]
    g = card.gap_groups(rows, samples, files, 2)
    koan = g["arm"]["sft/koan"]
    assert koan["samples"] == 2 and koan["gap_samples"] == 1 and koan["episodes"] == 4 and koan["gap_episodes"] == 3
    assert koan["solvable_episodes"] == 20 and koan["solve_rate_drop_points"] == 15.0 and koan["on_page"]
    assert g["read"]["sft/koan/heldout"]["gap_share_episodes"] == 75.0
    part1 = g["source"]["gate0"]
    assert part1["gap_episodes"] == 1 and part1["solve_rate_drop_points"] == 2.0 and part1["on_page"]


def test_strip_docstrings_and_function_dump():
    a = ast.parse('def f(x):\n    """one"""\n    return x\n\nclass K:\n    def step(self):\n        """d"""\n        return 1\n')
    b = ast.parse("def f(x):\n    return x\n\nclass K:\n    def step(self):\n        return 1\n")
    assert card.function_dump(a, "f") == card.function_dump(b, "f")
    assert card.function_dump(a, "K.step") == card.function_dump(b, "K.step")
    assert card.function_dump(a, "g") is None and card.function_dump(a, "K.other") is None


def library_as_lab_file(tmp_path: Path, edit=None) -> Path:
    """The library's env.py written the way the lab's koan_env.py imports (absolute imports), optionally edited."""
    src = (Path(env.__file__)).read_text()
    src = src.replace("from .base import", "from anatta.koans.base import").replace(
        "from .impossible import", "from anatta.koans.impossible import")
    if edit:
        old, new = edit
        assert src.count(old) == 1
        src = src.replace(old, new)
    p = tmp_path / "koan_env.py"
    p.write_text(src)
    return p


def test_identity_rule_catches_a_changed_function_value_or_pattern(tmp_path):
    base = Path(env.__file__).parent / "base.py"
    same = card.identity(library_as_lab_file(tmp_path), base)
    assert same["identical"] and all(same["functions"].values()) and all(same["values"].values())
    docs = card.identity(library_as_lab_file(tmp_path, ('"""The code agrees with the reference',
                                                        '"""A changed docstring. The code agrees with the reference')),
                         base)
    assert docs["identical"]
    body = card.identity(library_as_lab_file(tmp_path, ("return min(HIDDEN_TIMEOUT_MAX, max(HIDDEN_TIMEOUT, "
                                                        "TIMEOUT_FACTOR * seconds))",
                                                        "return min(HIDDEN_TIMEOUT_MAX, max(HIDDEN_TIMEOUT, "
                                                        "TIMEOUT_FACTOR * seconds + 1))")), base)
    assert not body["identical"] and body["functions"] == {**body["functions"], "hidden_timeout": False}
    assert sum(not v for v in body["functions"].values()) == 1
    mem = card.identity(library_as_lab_file(tmp_path, ("MEMORY_MB = 1024", "MEMORY_MB = 2048")), base)
    assert not mem["identical"] and not mem["values"]["MEMORY_MB"] and not mem["values"]["_PRELUDE"]
    other = tmp_path / "base.py"
    other.write_text(base.read_text().replace(r"(?:python3?|py)?", r"(?:python3?)?"))
    assert not card.identity(library_as_lab_file(tmp_path), other)["code_re"]


@pytest.mark.skipif(not (LAB / "koan_env.py").exists() or not DEV_BASE.exists(), reason="the lab files are not here")
def test_identity_rule_on_the_real_lab_files():
    res = card.identity(LAB / "koan_env.py", DEV_BASE)
    assert set(res["functions"]) == set(card.IDENTITY_FUNCTIONS) and set(res["values"]) == set(card.IDENTITY_VALUES)
    assert res["identical"], res


# ---------------------------------------------------------------------------------------------- the samples


def episode(pair, outcome, reply, impossible=False):
    return {"name": "x", "pair": pair, "impossible": impossible, "outcome": outcome, "replies": ["first", reply]}


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_samples_collect_dedupes_and_lists_files(tmp_path):
    from anatta.koans.base import CODE_RE

    root = tmp_path / "results"
    fence = "```python\ndef f():\n    return 1\n```"
    write_jsonl(root / "sft/koan_s0/heldout.jsonl", [
        episode("p1", "solved", f"Sure.\n{fence}"),
        episode("p1", "solved", fence + "\n```python\nprint(2)\n```"),  # first block: the same code
        episode("p2", "near_miss", "def g():\n    return 2\n  "),        # no fence: the reply stripped
        episode("p2", "hacked", fence, impossible=True),                 # impossible twin: left out
        episode("p3", "failed", fence)])                                 # not graded: left out
    write_jsonl(root / "durability/reads/start/koan_s0/heldout.jsonl",
                [episode("p1", "solved", fence), {**episode("p4", "overfit", "x = 1"), "replies": [{"content": "y = 2"}]}])
    (root / "gate0").mkdir()
    (root / "gate0/copy.jsonl").write_bytes((root / "sft/koan_s0/heldout.jsonl").read_bytes())
    write_jsonl(root / "gate0/notes.jsonl", [{"pair": "p1", "outcome": "solved"}])  # not an episode file
    write_jsonl(root / "calibration/koan_scorer/c6c_samples.jsonl", [episode("p9", "solved", fence)])
    found = smp.collect(root, CODE_RE, exclude=("calibration/koan_scorer",))
    c = found["counts"]
    assert c["episode_files"] == 3 and c["contributing_files"] == 3 and c["duplicate_files"] == 1
    assert c["episodes"] == 3 + 2 + 3 and c["samples"] == 3 and c["pairs"] == 3 and c["conflicts"] == 0
    s = found["samples"][("p1", hashlib.sha256(b"def f():\n    return 1\n").hexdigest())]
    assert s["episodes"] == 5 and s["sources"] == {"sft", "durability", "gate0"} and s["verdict"] == "solved"
    assert ("p2", hashlib.sha256(b"def g():\n    return 2").hexdigest()) in found["samples"]
    assert ("p4", hashlib.sha256(b"y = 2").hexdigest()) in found["samples"]
    assert found["duplicates"] == [["gate0/copy.jsonl", "sft/koan_s0/heldout.jsonl"]]
    assert found["longest_reply"] == len(fence + "\n```python\nprint(2)\n```")
    listing = smp.write(found, tmp_path / "out", extra={"root": "results"})
    lines = (tmp_path / "out/c6c_samples.jsonl").read_text().splitlines()
    assert len(lines) == 3 and [json.loads(x)["pair"] for x in lines] == ["p1", "p2", "p4"]
    assert listing["samples_file"]["sha256"] == hashlib.sha256((tmp_path / "out/c6c_samples.jsonl").read_bytes()).hexdigest()
    assert smp.load_samples(tmp_path / "out/c6c_samples.jsonl")[0]["episodes_by_file"]["gate0/copy.jsonl"] == 2


def test_samples_mark_a_conflict(tmp_path):
    from anatta.koans.base import CODE_RE

    write_jsonl(tmp_path / "a/x.jsonl", [episode("p", "solved", "x = 1"), episode("p", "near_miss", "x = 1")])
    found = smp.collect(tmp_path, CODE_RE)
    assert found["counts"]["conflicts"] == 1 and next(iter(found["samples"].values()))["verdict"] == "conflict"


# ------------------------------------------------------------------------------------------ the app, stubbed


@pytest.fixture
def app():
    stub, _ = load_apps(only=())
    base = stub.module.App

    class App(base):  # records each function's options
        def function(self, **k):
            deco = super().function(**k)

            def wrap(f):
                fn = deco(f)
                fn.options = k
                return fn
            return wrap

    stub.module.App = App
    try:
        yield stub, load("koan_scorer_app", CHECK / "app.py")
    finally:
        unload_apps()


def test_app_images_functions_and_constants(app):
    stub, a = app
    assert a.app.name == "anatta-calib-koan-scorer" and set(a.app.functions) == {"generate", "score"}
    s, g = a.score.options, a.generate.options
    assert s["cpu"] == (8.0, 8.0) and s["memory"] == 36864 and s["block_network"] and s["single_use_containers"]
    assert s["timeout"] == 2 * 3600 and s["image"] is a.score_image and "gpu" not in s and "volumes" not in s
    assert g["gpu"] == "L40S" and set(g["volumes"]) == {"/cache"}
    a.check_constants(CFG)
    starts = [i for i, c in enumerate(stub.image_calls) if c[0] == "debian_slim"]
    calls = stub.image_calls[starts[-1]:]
    assert [c[0] for c in calls] == ["debian_slim", "apt_install", "uv_pip_install", "run_commands", "env",
                                     "add_local_dir", "add_local_file", "add_local_file"]
    assert calls[7][1] == (str(a.LAB / CFG["lab"]["parquet"]["path"]), "/root/koan_scorer/mbppplus.parquet")
    assert calls[0][2] == {"python_version": "3.12"} and calls[1][1] == ("git",)
    assert "numpy==2.5.4" in calls[2][1] and "psutil==7.0.0" in calls[2][1]
    install, imports = calls[3][1]
    assert "--no-deps" in install and f"@{CFG['evalplus']['commit']}'" in install
    assert imports == "python -c 'import evalplus.eval, evalplus.gen.util'"
    assert calls[4][1][0] == {"PYTHONPATH": "/root"} and calls[6][1][1] == "/root/koan_scorer/scoring.py"
    assert a.ONE_THREAD == {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def test_results_come_back_packed_and_small(app):
    _, a = app
    res = {"result": {"set": "c6c", "rows": [{"pair": "mbpp_2", "theirs": True}] * 1000}, "log_tail": "done"}
    blob = a.packed(res)
    assert a.unpacked(blob) == res and len(blob) < 8 * 1024  # repetitive JSON compresses well
    big = {"result": {"set": "c6a", "rows": [hashlib.sha256(str(i).encode()).hexdigest() for i in range(100_000)]}}
    r = a.unpacked(a.packed(big))["result"]
    assert r["stopped"] == "output" and r["set"] == "c6a" and "bytes compressed" in r["problems"][0]


SAMPLES = LAB / CFG["lab"]["results"] / "c6c_samples.jsonl"


@pytest.mark.skipif(not (LAB / "pairs.jsonl").exists() or not SAMPLES.exists(), reason="the lab's files are not here")
def test_every_planned_call_fits_inline(app, tmp_path):
    # A scoring container has no network, so every call's arguments must reach it inline (Modal: 2 MiB). The full
    # run's largest calls are held under 1 MiB: C6a with all 368 pairs and every reply at the 1,024-token cap (8,192
    # characters, cap_replies), and each of C6c's shards. The samples and replies are read as data, never run.
    _, a = app
    pairs = [json.loads(ln) for ln in (LAB / "pairs.jsonl").read_text().splitlines() if ln.strip()]
    samples = [json.loads(ln) for ln in SAMPLES.read_text().splitlines() if ln.strip()]
    replies = a.cap_replies(pairs, samples)
    assert len(replies["rows"]) == 368 and {len(r["reply"]) for r in replies["rows"]} == {8192}
    sha = CFG["lab"]["parquet"]["sha256"]
    jobs = a.plan_jobs(CFG, pairs, sha, tmp_path, tmp_path, replies, samples, smoke=False)
    sizes = {label: a.args_bytes(plan, payload) for label, _, plan, payload in jobs}
    own = CFG["scoring"]["own_shard"]
    assert len(sizes) == 1 + len(own) + CFG["scoring"]["c6c_shards"] and sum(len(j[3]["items"]) for j in jobs[1:]) == 9441
    first, rest = jobs[1][3], [j[3] for j in jobs[2:]]
    assert {it["pair"] for it in first["items"]} == {"mbpp_255"} and first["settings"]["workers"] == own["mbpp_255"]
    assert all(r["settings"]["workers"] == 8 and "mbpp_255" not in {it["pair"] for it in r["items"]} for r in rest)
    assert all(j[2]["settings"] == j[3]["settings"] and j[2]["shard"] == j[3]["shard"] == k for k, j in enumerate(jobs[1:]))
    assert all(n < a.ARGS_LIMIT for n in sizes.values()), sizes
    t = a.transport_jobs(CFG, pairs, sha, samples, tmp_path, tmp_path)
    assert [j[0] for j in t] == ["C6a", max((k for k in sizes if k != "C6a"), key=sizes.get)]
    plan, blob = a.call_args(*t[0][2:])
    assert a.unpacked(blob)["items"] == t[0][3]["items"] and plan["items_sha256"] == a.json_sha256(t[0][3]["items"])


def test_shards_planted_and_smoke_samples(app):
    _, a = app
    items = [{"pair": p, "n": i} for p, k in (("a", 5), ("b", 3), ("c", 3), ("d", 1)) for i in range(k)]
    shards = a.shards_by_pair(items, 3)
    assert [sorted({x["pair"] for x in s}) for s in shards] == [["a"], ["b", "d"], ["c"]]
    assert a.shards_by_pair(items, 8) == a.shards_by_pair(items, 8) and len(a.shards_by_pair(items, 8)) == 4
    smoke = CFG["smoke"]
    main, other = smoke["planted_pair"], smoke["plant_pairs"]["sticky_atol"]
    by_name = {main: {"name": main, "entry": "dbl", "reference": "def dbl(x):\n    return x * 2\n"},
               other: {"name": other, "entry": "sq", "reference": "def sq(x):\n    return x * x\n"}}
    planted = a.planted_items(by_name, smoke)
    assert [p["planted"] for p in planted] == list(smoke["expected_causes"]) == list(a.PLANTS)
    assert [p["pair"] for p in planted] == [main] * 5 + [other]
    for p in planted:
        ast.parse(p["code"])
        assert p["verdict"] == "solved" and p["sha256"] == hashlib.sha256(p["code"].encode()).hexdigest()
    by = {p["planted"]: p["code"] for p in planted}
    ns = {}
    exec(by["sticky_atol"], ns)  # the fixture reference above, wrapped
    assert [ns["sq"](x) for x in (10 ** 4, 3, 0.5, True, 4000)] == [10 ** 8 + 1, 9, 0.25, 1, 16_000_000 + 1]
    assert sc.guard_hits(by["guard_hit"]) == ["os.getcwd"] and sc.guard_hits(by["slow_correct"]) == []
    ns = {}
    exec(by["recorded_wrong"], ns)  # the fixture reference above, wrapped: small enough to run here
    assert ns["dbl"](3) is None
    ns = {}
    exec(by["guard_hit"], ns)
    assert ns["dbl"](3) == 6
    samples = [{"pair": "p", "verdict": "solved", "i": 0}, {"pair": "p", "verdict": "solved", "i": 1},
               {"pair": "p", "verdict": "near_miss", "i": 2}, {"pair": "p", "verdict": "overfit", "i": 3},
               {"pair": "q", "verdict": "solved", "i": 4}, {"pair": "z", "verdict": "solved", "i": 5}]
    assert [s["i"] for s in a.smoke_samples(samples, ["p", "q"], 2)] == [0, 2, 4]
    assert [s["i"] for s in a.smoke_samples(samples, ["p", "q"], 2, every=("p",))] == [0, 1, 2, 3, 4]
    assert smoke["c6c_every_sample"] == ["mbpp_255"] and "mbpp_255" in smoke["pairs"]


def test_c6c_groups_set_a_pair_apart_with_fewer_workers(app):
    _, a = app
    items = [{"pair": p, "n": i} for p, k in (("a", 5), ("b", 3), ("c", 3), ("d", 1)) for i in range(k)]
    cfg = {"scoring": {"own_shard": {"b": 2, "absent": 1}}}
    groups = a.c6c_groups(cfg, items, 2, {"workers": 8, "cpu": 8})
    assert [sorted({x["pair"] for x in g}) for g, _ in groups] == [["b"], ["a"], ["c", "d"]]
    assert [s["workers"] for _, s in groups] == [2, 8, 8] and all(s["cpu"] == 8 for _, s in groups)
    plain = a.c6c_groups({"scoring": {}}, items, 3, {"workers": 8})
    assert [g for g, _ in plain] == a.shards_by_pair(items, 3)  # without own_shard, the shards as before


def test_the_smoke_plan(app, tmp_path):
    # mbpp_255's samples fill a container of their own with scoring.own_shard's workers; the other pairs' share one.
    _, a = app
    smoke = CFG["smoke"]
    names = [*smoke["pairs"], *smoke["plant_pairs"].values()]
    pairs = [{"name": n, "entry": "f", "reference": "def f(x):\n    return x\n"} for n in names]
    samples = [{"pair": n, "sha256": f"{n}-{i}", "code": "def f(x):\n    return x\n", "episodes": 1,
                "verdict": ("solved", "near_miss")[i % 2]} for n in smoke["pairs"] for i in range(20 if n == "mbpp_255" else 3)]
    jobs = a.plan_jobs(CFG, pairs, "0" * 64, tmp_path, tmp_path, None, samples, smoke=True)
    assert [j[0] for j in jobs] == ["planted", "C6c shard 0", "C6c shard 1"]
    planted, first, second = (j[3] for j in jobs)
    assert {p["name"] for p in planted["pairs"]} == {smoke["planted_pair"], smoke["plant_pairs"]["sticky_atol"]}
    assert {it["pair"] for it in first["items"]} == {"mbpp_255"} and len(first["items"]) == 20
    assert len(second["items"]) == 2 * (len(smoke["pairs"]) - 1)
    assert first["settings"]["workers"] == CFG["scoring"]["own_shard"]["mbpp_255"] == 3
    assert second["settings"]["workers"] == planted["settings"]["workers"] == 8
    assert all(j[2]["settings"] == j[3]["settings"] for j in jobs)


def test_stage_files_and_the_samples_guard(app, tmp_path):
    _, a = app
    f = tmp_path / "x.json"
    assert a.stage_file(f, {"a": 1}) is None
    f.write_text(json.dumps({"plan": {"a": 1}, "rows": []}))
    assert a.stage_file(f, {"a": 1})["rows"] == []
    with pytest.raises(SystemExit):
        a.stage_file(f, {"a": 2})
    lab_out = tmp_path / "lab"
    lab_out.mkdir()
    (lab_out / "c6c_samples.jsonl").write_text(json.dumps({"pair": "p"}) + "\n")
    meta = {"samples_file": {"sha256": "0" * 64}, "registered_counts_reproduced": True,
            "counts": CFG["registered"]["c6c_counts"]}
    (lab_out / "c6c_files.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit):
        a.load_c6c_samples(CFG, lab_out)
    meta["samples_file"]["sha256"] = hashlib.sha256((lab_out / "c6c_samples.jsonl").read_bytes()).hexdigest()
    (lab_out / "c6c_files.json").write_text(json.dumps(meta))
    assert a.load_c6c_samples(CFG, lab_out) == [{"pair": "p"}]


# ----------------------------------------------------------------------------------- a whole card, synthetic


def scored_file(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"plan": {}, "result": result}))


def pairs_block(names, ok=True):
    return {n: {"budget": {"seconds": 0.1, "budget": 20.0},
                "step0": {"ok": ok if not isinstance(ok, dict) else ok[n], "why": [], "inputs": 50},
                "ground_truth": {"hidden": {"times": [0.001] * 3, "max_time": 0.001}, "full": {"max_time": 0.002}}}
            for n in names}


def test_a_whole_card_from_synthetic_results(tmp_path):
    pub, lab = tmp_path / "pub", tmp_path / "lab"
    lab_out = lab / CFG["lab"]["results"]
    koan_env = library_as_lab_file(tmp_path)
    (lab / CFG["lab"]["koan_env"]).parent.mkdir(parents=True, exist_ok=True)
    (lab / CFG["lab"]["koan_env"]).write_text(koan_env.read_text())
    dev_base = Path(env.__file__).parent / "base.py"
    c6a_rows = ([{"set": "c6a", "pair": f"p{i}", "id": f"p{i}", "outcome": "solved", "visible_pass": True, "ours": True,
                  "theirs": True, "full": "pass", "fences": 1, "hit_cap": False, "evalplus_visible": True}
                 for i in range(70)]
                + [{"set": "c6a", "pair": "p70", "id": "p70", "outcome": "solved", "visible_pass": True, "ours": True,
                    "theirs": True, "full": "fail", "fences": 2, "hit_cap": True, "evalplus_visible": True}]
                + [{"set": "c6a", "pair": f"q{i}", "id": f"q{i}", "outcome": "failed", "visible_pass": False,
                    "ours": False, "theirs": False, "full": "fail (same inputs)", "fences": 0, "syntax_error": True,
                    "evalplus_visible": False} for i in range(29)])
    names = [r["pair"] for r in c6a_rows]
    scored_file(pub / "c6a_scored.json", {"set": "c6a", "rows": c6a_rows, "pairs": pairs_block(names),
                                          "pair_flags": {n: {"atol": 0.0, "set_eq": False} for n in names},
                                          "container": {"cpu": {"model": ["X"]}, "pip_freeze": ["a==1"]}})
    c6c_rows = [{"set": "c6c", "pair": "p1", "id": f"s{i}", "recorded": "solved", "ours": True, "rerun": True,
                 "theirs": True, "full": "pass"} for i in range(99)]
    c6c_rows.append({"set": "c6c", "pair": "p1", "id": "s99", "recorded": "solved", "ours": True, "rerun": True,
                     "theirs": False, "full": "fail (same inputs)", "cause": {"cause": "timing"},
                     "rerun_cause": {"cause": "timing"}, "loose": [[True, True]] * 3})
    scored_file(lab_out / "c6c_scored" / "shard_00.json", {"set": "c6c", "rows": c6c_rows, "pairs": pairs_block(["p1"]),
                                                           "pair_flags": {"p1": {"atol": 0.0, "set_eq": False}}})
    samples = [{"pair": "p1", "sha256": f"s{i}", "episodes": 2, "episodes_by_file": {"sft/koan_s0/heldout.jsonl": 2}}
               for i in range(100)]
    (lab_out / "c6c_samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    (lab_out / "c6c_files.json").write_text(json.dumps({"files": [{"path": "sft/koan_s0/heldout.jsonl",
                                                                   "solvable": 276}], "counts": {}}))
    c, lc = card.write_cards(CFG, pub, lab, lab_out, dev_base, publish_c6c=True)
    assert c["c6a"]["graded"] == 71 and c["c6a"]["agree"] == 71 and c["c6a"]["verdict"] == "pass"
    assert c["c6a"]["full_set_gap"]["samples"] == 1 and c["c6a"]["full_set_gap"]["solve_rate_drop_points"] == 1.0
    assert c["c6a"]["visible_vs_evalplus_base"]["agree"] == 100
    b = c["c6b"]
    assert (b["n"], b["pass"], b["rate"], b["verdict"]) == (100, 70, 70.0, "in band")
    assert b["extraction"] == {"no_fence": 29, "more_than_one_fence": 1, "syntax_error": 29, "hit_cap": 1,
                               "hit_cap_share": 0.01}
    assert c["identity"]["identical"] and c["oracle"]["faults"] == 0 and c["oracle"]["hidden_inputs_checked"] == 5000
    assert c["oracle"]["no_stored_results"] == []
    assert c["hashes"]["anatta_dev_impossible"] == hashlib.sha256((dev_base.parent / "impossible.py").read_bytes()).hexdigest()
    k = lc["c6c"]
    assert k["graded"] == 100 and k["agree"] == 99 and k["verdict"] == "pass" and k["episodes"] == 200
    assert k["disagreements"]["by_cause"] == {"timing": 1}
    assert k["full_set_gap"]["arm"]["sft/koan"]["gap_samples"] == 0
    assert c["c6c_published"]["verdict"] == "pass" and "rows" not in json.dumps(c["c6c_published"])
    assert json.loads((pub / "card.json").read_text())["c6b"]["verdict"] == "in band"
    assert (lab_out / "c6c_card.json").exists()
    text = "\n".join(card.headline(c, lc))
    assert "C6a: 71/71" in text and "C6b: 70/100" in text and "C6c: 99/100" in text
    c2, _ = card.write_cards(CFG, pub, lab, lab_out, dev_base)
    assert "c6c_published" not in c2  # C6c stays out of the public card unless asked for


def test_a_card_from_shards_with_error_rows_and_unscored_samples(tmp_path):
    # The full run's error rows have no id or verdict fields ({"set", "pair", "index", "scoring_error"}); rows made by
    # error_row also carry the id. Either counts as a scoring error, a shard that stopped or samples with no row count
    # as unscored, and each makes its set incomplete. The card must never index an error row's missing fields.
    pub, lab = tmp_path / "pub", tmp_path / "lab"
    lab_out = lab / CFG["lab"]["results"]
    (lab / CFG["lab"]["koan_env"]).parent.mkdir(parents=True, exist_ok=True)
    (lab / CFG["lab"]["koan_env"]).write_text(library_as_lab_file(tmp_path).read_text())
    dev_base = Path(env.__file__).parent / "base.py"
    c6a_rows = [{"set": "c6a", "pair": "p0", "id": "p0", "outcome": "solved", "visible_pass": True, "ours": True,
                 "theirs": True, "full": "pass", "fences": 1, "hit_cap": False, "evalplus_visible": True},
                {"set": "c6a", "pair": "p1", "index": 1, "id": "p1", "scoring_error": "Traceback ..."}]
    scored_file(pub / "c6a_scored.json", {"set": "c6a", "rows": c6a_rows, "pairs": pairs_block(["p0", "p1", "p2"]),
                                          "pair_flags": {n: {"atol": 0.0, "set_eq": False} for n in ("p0", "p1", "p2")},
                                          "container": {"cpu": {"model": ["X"]}, "pip_freeze": ["a==1"]}})
    gen = {"plan": {"batch": 32, "max_new_tokens": 1024}, "info": {},
           "rows": [{"pair": n, "reply": "x", "new_tokens": 1, "hit_cap": False} for n in ("p0", "p1", "p2")]}
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "c6a_replies.json").write_text(json.dumps(gen))  # p2's reply has no row: unscored
    ok_row = {"set": "c6c", "pair": "p1", "recorded": "solved", "ours": True, "rerun": True, "theirs": True,
              "full": "pass"}
    rows = [{**ok_row, "id": f"s{i}", "index": i} for i in range(3)]
    rows += [{"set": "c6c", "pair": "p1", "index": 3, "scoring_error": "CheckError: ..."},  # the full run's shape
             {"set": "c6c", "pair": "p1", "index": 4, "id": "s4", "scoring_error": "not scored"}]  # error_row's
    scored_file(lab_out / "c6c_scored" / "shard_00.json", {"set": "c6c", "rows": rows, "pairs": pairs_block(["p1"]),
                                                           "pair_flags": {"p1": {"atol": 0.0, "set_eq": False}}})
    (lab_out / "c6c_scored" / "shard_01.json").write_text(json.dumps(  # its plan gives the shard number
        {"plan": {"shard": 1}, "result": {"set": "c6c", "stopped": "container", "problems": ["x"]}}))
    lab_out.mkdir(parents=True, exist_ok=True)
    samples = [{"pair": "p1", "sha256": f"s{i}", "episodes": 2, "episodes_by_file": {"sft/koan_s0/heldout.jsonl": 2}}
               for i in range(7)]
    (lab_out / "c6c_samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    (lab_out / "c6c_files.json").write_text(json.dumps({"files": [{"path": "sft/koan_s0/heldout.jsonl",
                                                                   "solvable": 14}], "counts": {}}))
    c, lc = card.write_cards(CFG, pub, lab, lab_out, dev_base, publish_c6c=True)
    k = lc["c6c"]
    assert k["verdict"] == "incomplete" and k["scoring_errors"] == 2 and k["unscored"] == 2  # s5, s6 have no row
    assert k["samples"] == 5 and k["expected_samples"] == 7 and k["stopped_shards"] == [1]
    assert k["graded"] == 3 and k["agree"] == 3 and k["episodes"] == 4 * 2  # s0-s2 and the error row s4
    assert k["rerun_vs_recorded"]["n"] == 3 and k["full_set_gap"]["arm"]["sft/koan"]["samples"] == 3
    p = c["c6c_published"]
    assert p["verdict"] == "incomplete" and p["scoring_errors"] == 2 and p["unscored"] == 2
    a = c["c6a"]
    assert a["verdict"] == "incomplete" and a["scoring_errors"] == 1 and a["unscored"] == 1 and a["graded"] == 1
    assert c["c6b"]["verdict"] == "incomplete" and c["c6b"]["unscored"] == 1
    text = "\n".join(card.headline(c, lc))
    assert ("C6a: 1/1 agree (100.0%), 0 unexplained, 0 left out by step 0: incomplete (1 scoring error(s), 1 "
            "sample(s) unscored)") in text
    assert "C6c: 3/3 agree (100.0%), 0 unexplained: incomplete (2 scoring error(s), 2 sample(s) unscored)" in text
    # every sample with a row and no error: complete again
    (lab_out / "c6c_scored" / "shard_01.json").unlink()
    rows = [{**ok_row, "id": f"s{i}", "index": i} for i in range(7)]
    scored_file(lab_out / "c6c_scored" / "shard_00.json", {"set": "c6c", "rows": rows, "pairs": pairs_block(["p1"]),
                                                           "pair_flags": {"p1": {"atol": 0.0, "set_eq": False}}})
    _, lc = card.write_cards(CFG, pub, lab, lab_out, dev_base)
    assert lc["c6c"]["verdict"] == "pass" and lc["c6c"]["unscored"] == 0 and lc["c6c"]["stopped_shards"] == []
    assert "C6c: 7/7 agree (100.0%), 0 unexplained: pass" in card.headline(c, lc)  # the whole line: nothing appended
    # a smoke scores a few of the samples: their count is not checked, a stopped shard still counts
    res = [{"set": "c6c", "rows": rows[:2], "pairs": pairs_block(["p1"]), "shard": 0},
           {"set": "c6c", "stopped": "runners", "shard": 1}]
    s = card.c6c_section(CFG, res, lab_out, {}, smoke=True)
    assert s["verdict"] == "incomplete" and s["unscored"] == 1 and s["expected_samples"] is None
    assert card.c6c_section(CFG, res[:1], lab_out, {}, smoke=True)["verdict"] == "pass"


def test_smoke_check_reads_each_planted_cause():
    # Recorded as solved: a sample EvalPlus passes differs only in its re-run, a wrong one only in its recorded verdict.
    rows = [{"planted": "slow_correct", "rerun": True, "theirs": False, "cause": {"cause": "timing"},
             "rerun_cause": {"cause": "timing"}, "loose": [[True, True]] * 3, "evalplus": {}, "full": "fail"},
            {"planted": "numpy_scalar", "rerun": False, "theirs": True, "rerun_cause": {"cause": "known_rule"}},
            {"planted": "memory_hog", "rerun": False, "theirs": True, "rerun_cause": {"cause": "environment"}},
            {"planted": "guard_hit", "rerun": True, "theirs": False, "cause": {"cause": "environment"},
             "rerun_cause": {"cause": "environment"}},
            {"planted": "recorded_wrong", "rerun": False, "theirs": False, "cause": {"cause": "unexplained"}},
            {"planted": "sticky_atol", "rerun": False, "theirs": True,
             "rerun_cause": {"cause": "known_rule", "rules": ["sticky_atol"], "sticky_atol": True, "loose": "both fail"}}]
    s = card.smoke_section(CFG, {"rows": rows, "pairs": {"mbpp_588": {}}}, None, [])
    assert s["planted_as_expected"] and s["planted"]["numpy_scalar"]["got"] == "known_rule"
    assert s["planted"]["sticky_atol"]["rules"] == ["sticky_atol"] and s["planted"]["sticky_atol"]["as_expected"]
    assert not s["every_path_ran"] and not s["paths"]["visible_base"]  # no C6a rows here
    assert not s["runner_checks_ok"] and not s["memory"]["well_under"]  # no runner records in these rows
    rows[5] = {**rows[5], "rerun_cause": {**rows[5]["rerun_cause"], "loose": "disagree"}}
    assert not card.smoke_section(CFG, {"rows": rows}, None, [])["planted"]["sticky_atol"]["as_expected"]
    rows[5] = {**rows[5], "rerun_cause": {"cause": "known_rule", "rules": ["non_builtin_type"], "loose": "both fail"}}
    assert not card.smoke_section(CFG, {"rows": rows}, None, [])["planted_as_expected"]  # the wrong rule
    rows[1] = {"planted": "numpy_scalar", "scoring_error": "Traceback"}
    s = card.smoke_section(CFG, {"rows": rows}, None, [])
    assert s["planted"]["numpy_scalar"]["got"] == "scoring_error" and not s["planted_as_expected"]


def memory_result(top: int, labels=("same_inputs", "full_set")) -> dict:
    rec = {"before": {"VmSize": top, "VmRSS": top // 2}, "after": {"VmSize": top + 10, "VmRSS": top // 2}, "calls": 3}
    per = {"mbpp_255": {label: rec for label in labels}}
    return {"set": "c6c", "shard": 0, "runner_check": {"ok": True}, "container": {"workers": 3,
                                                                                  "modal": {"memory_mib": 36864}},
            "runner_memory": sc.runner_memory({}, [{"pair": "mbpp_255", "runner_vm": per["mbpp_255"]}], 4 * 1024 ** 3),
            "runners": [{"memory": {"VmSize": 150_000, "VmRSS": 60_000, "VmPeak": 151_000}, "memory_limit": 4 * 1024 ** 3}],
            "runners_end": [{"memory": {"VmPeak": 400_000, "VmHWM": 300_000, "maxrss_children": 900_000}, "calls": 9}],
            "workers_end": [{"memory": {"VmHWM": 3_000_000, "maxrss_children": 1_000_000}, "runner_gone": False}],
            "main_memory": {"end": {"VmHWM": 2_500_000}},
            "container_memory": {"phases": {"prepare": {"meminfo_used": 5_000_000}, "scoring": {"meminfo_used": 9_000_000,
                                                                                                 "sum_rss": 12_000_000}}}}


def test_smoke_memory_and_the_memory_section():
    m = card.smoke_memory(CFG, [memory_result(200_000)])
    assert m["well_under"] and m["containers"][0]["before_VmSize"] == 200_000
    assert m["containers"][0]["where"] == "mbpp_255 full_set" or m["containers"][0]["where"] == "mbpp_255 same_inputs"
    assert m["containers"][0]["used_peak"] == {"meminfo_used": 9_000_000, "sum_rss": 12_000_000}
    assert m["containers"][0]["workers"] == 3 and m["containers"][0]["memory_mib"] == 36864
    assert m["containers"][0]["used_share_of_request"] == round(9_000_000 / (36864 * 1024), 3)
    lines = card.headline({"identity": {"identical": True},
                           "smoke_check": {"every_path_ran": True, "planted_as_expected": True, "memory": m}})
    assert "195 MiB" in lines[-2] and "8.6 GiB at its peak (0.238 of its 36 GiB, 3 workers)" in lines[-2]
    assert m["pairs"]["mbpp_255"][0]["full_set"]["before"]["VmSize"] == 200_000 and m["pairs"]["mbpp_255"][0]["shard"] == 0
    both = card.smoke_memory(CFG, [memory_result(200_000, labels=("same_inputs",)),
                                   {**memory_result(200_000, labels=("full_set",)), "shard": 1}])
    assert both["well_under"] and [c["shard"] for c in both["pairs"]["mbpp_255"]] == [0, 1]  # calls across containers
    assert not card.smoke_memory(CFG, [memory_result(3_000_000)])["well_under"]  # past half of 4 GiB
    assert not card.smoke_memory(CFG, [memory_result(200_000, labels=("same_inputs",))])["well_under"]  # no full set
    sec = card.memory_section(memory_result(200_000))
    assert sec["runners"] == [{"start": {"VmSize": 150_000, "VmRSS": 60_000}, "memory_limit": 4 * 1024 ** 3,
                               "end": {"VmPeak": 400_000, "VmHWM": 300_000, "maxrss_children": 900_000}, "calls": 9,
                               "largest_around_a_call": None}]
    assert sec["workers_end"] == [{"VmHWM": 3_000_000, "maxrss_children": 1_000_000}] and sec["runner_check_ok"]
    assert sec["main"] == {"end": {"VmHWM": 2_500_000}} and "per_pair" not in sec["runner_memory"]
    assert sec["container_used_peak"]["meminfo_used"] == 9_000_000 and sec["proc_peak_lines"]
    bare = memory_result(200_000)
    bare["runners"][0]["memory"].pop("VmPeak")  # gVisor's /proc/self/status: no VmPeak or VmHWM lines
    bare["runners_end"][0]["memory"] = {"maxrss_self": 380_000, "maxrss_children": 900_000}
    bare["workers_end"][0]["memory"] = {"maxrss_self": 3_000_000}
    bare["main_memory"] = {"end": {"VmSize": 2_000_000, "maxrss_self": 2_500_000}}
    sec = card.memory_section(bare)
    assert not sec["proc_peak_lines"] and sec["runners"][0]["end"] == {"maxrss_self": 380_000, "maxrss_children": 900_000}
    assert sec["main"] == {"end": {"maxrss_self": 2_500_000}}


def test_results_path_refuses_a_git_tree(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(SystemExit):
        card.results_path(tmp_path / "repo" / "out")
    assert card.results_path(tmp_path / "out").is_dir()
