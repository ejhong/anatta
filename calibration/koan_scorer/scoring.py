"""Calibration check 6's scoring run, inside a CPU container: app.py's score function writes the inputs to a file
and starts this file as a fresh process, which never imports pyarrow or anything else that starts threads, since it
forks. See README.md for the claim and the rules.

  python scoring.py --input IN.json --output OUT.json

IN.json holds {"set": "c6a" | "c6c" | "planted", "pairs": [pair records], "tasks": {task id: {"test": the EvalPlus
test module's source}}, "items": [...], "settings": {...}}. The run:

  1. checks the container: EvalPlus installed at the registered commit, EVALPLUS_TIMEOUT_PER_TASK and
     EVALPLUS_MAX_MEMORY_BYTES unset, the fork start method; and records the CPU, memory, versions and pip freeze.
     Then, with this process still single-threaded and holding only the inputs, forks a memory sampler (Sampler)
     and one EvalPlus runner per worker (Runner): every EvalPlus call below is made in a runner, so the children
     EvalPlus forks inherit that call's data and nothing of ours. Each runner must use fork, have one thread and
     import EvalPlus, and runner 0 must give a fixture's known answers (runner_check);
  2. reads each pair's hidden inputs twice: as the hidden harness reads them (its restricted eval) for EvalPlus, and
     as source segments for runs of one input at a time; and the test module's inputs and stored results with ast
     and anatta.koans.env.literal, never run. Each hidden input is located among the module's inputs by its source;
  3. runs our references (anatta.koans.env.precompute_references, which sets each pair's hidden budget), then
     EvalPlus's ground truth (evalplus.gen.util.trusted_exec, with each input's time) of each reference on three
     input sets: the hidden inputs, the module's full set and the visible asserts' inputs. Each call runs once per
     pair, in runner 0, in a child it forks and stops after settings["ground_truth_timeout"] seconds;
  4. step 0, the oracle: per pair, trusted_exec's outputs in canonical form must equal reference_outputs on every
     hidden input, and the module's stored results, where it has them, must match them under evalplus_match. A pair
     that fails is an oracle fault, reported, and its samples leave the graded counts;
  5. lists every top-level module the samples import and imports each in a fresh `python -I`; a module that does
     not import stops the run before scoring, unless settings["imports_allowed_missing"] names it;
  6. scores every item with one worker process per core (non-daemonic, forked from this single-threaded process),
     each making its EvalPlus calls in its own runner, which records its memory just before each call:
       ours       C6a: KoanTask.step on the reply (solvable twin, neutral framing, no release, one attempt), whose
                  hidden_ok decides the verdict; a reply that fails the visible asserts gets hidden_ok on its own,
                  for the reported agreement over all replies. C6c and planted: the recorded verdict, and hidden_ok
                  re-run here. The hidden run's raw result is kept, input by input. A run whose harness wrote no
                  result line (anatta.koans.env._run's None: an output integer over Python's 4,300-digit limit for
                  json.dumps, or a process that died before reporting) is a recorded run with no result, which
                  hidden_ok reads as a fail; the row says so (our_run "no result");
       EvalPlus   untrusted_check("mbpp", code, the hidden inputs, entry_point=the pair's entry, expected and times
                  from trusted_exec, atol=the pair's atol, its default limits): the same-inputs verdict. Then the
                  full set (one untrusted_check over the module's inputs, expected from trusted_exec) for same-inputs
                  passes; and, for C6a, the visible asserts' inputs;
       loose      when the verdicts are not unanimous: three repeats of both scorers with loose limits, each input in
                  its own run (EvalPlus: its own untrusted_check with min_time_limit 60 s; ours: its own subprocess at
                  60 s), each repeat stopping at its first failing input;
       cause      every disagreement in one registered bucket (classify);
  7. writes OUT.json: the container record, the runners at their start and end, the pairs' budgets, ground truth
     and step 0, the imports, one row per item, the runners' memory around EvalPlus's calls per pair and kind of
     call (runner_memory), each process's peaks (main, workers, runners, and the largest child of each), and the
     container's used memory per phase (the sampler's).

The functions above "the run" use the standard library and anatta.koans.env only (the laptop's tests import them);
EvalPlus is imported inside the run.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from multiprocessing.connection import wait

from anatta.koans import env
from anatta.koans.base import CODE_RE
from anatta.koans.env import (
    EnvError,
    KoanTask,
    canon,
    evalplus_match,
    hidden_ok,
    hidden_timeout,
    literal,
    outputs,
    pair_from_record,
    precompute_references,
)

GRADED = ("solved", "near_miss", "overfit")  # the outcomes of a reply that passed the visible asserts
ENV_ERRORS = ("MemoryError", "RecursionError", "ImportError", "ModuleNotFoundError")
CAUSES = ("oracle", "environment", "known_rule", "timing", "unstable", "unexplained")
RULES = ("non_builtin_type", "complex_vs_float", "eq_override", "set_eq_floats", "sticky_atol")
ASSUMED_RULES = ("eq_override",)  # read from the code's text, not from the output: the card counts it apart
STICKY_ATOL = 1e-6  # the tolerance EvalPlus's unsafe_execute sets on an all-float expected output and keeps

# The hidden harness's own restricted eval (anatta.koans.env.HIDDEN_HARNESS): no builtins; inf, nan, set, frozenset.
HIDDEN_NAMES = {"inf": math.inf, "nan": math.nan, "set": set, "frozenset": frozenset}

# What EvalPlus's reliability_guard (evalplus/eval/utils.py at 6eb1e19) sets to None or blocks, and our harness
# leaves alone. Reading stdin fails under both harnesses, so it is not listed.
GUARDED = {
    "os": {"kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
           "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
           "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir"},
    "shutil": {"rmtree", "move", "chown"},
    "subprocess": {"Popen"},
    "builtins": {"exit", "quit", "open", "help"},
}
GUARDED_BUILTINS = ("exit", "quit", "open", "help")
BLOCKED_MODULES = ("ipdb", "joblib", "resource", "psutil", "tkinter")


class CheckError(RuntimeError):
    """The check itself went wrong (not the sample): the run stops or the row is marked as a scoring error."""


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def short(x, n: int = 300) -> str:
    """A value as compact JSON (or repr), cut to n characters, for the rows."""
    try:
        s = json.dumps(x, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(x)
    return s if len(s) <= n else s[:n - 3] + "..."


# ------------------------------------------------------------------------------------------------- reading


def parse_hidden(hidden: str) -> list:
    """The hidden inputs as the hidden harness reads them, with its own restricted eval, so EvalPlus calls the code
    with the same objects our harness does."""
    return eval(hidden, {"__builtins__": {}, **HIDDEN_NAMES})


def hidden_segments(hidden: str) -> list[str]:
    """Each hidden input's source text (one argument tuple), for running the inputs one at a time."""
    node = ast.parse(hidden, mode="eval").body
    if not isinstance(node, ast.List):
        raise ValueError("the hidden inputs are not a list literal")
    return [ast.get_source_segment(hidden, e) for e in node.elts]


def module_parts(test: str) -> dict:
    """The EvalPlus test module's `inputs` and `results` lists as AST nodes, read the way the pair builder read
    them (koan_pairs.parse_test). Nothing in the module is run."""
    nodes = {}
    for node in ast.parse(test).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nodes[node.targets[0].id] = node.value
    inputs, results = nodes.get("inputs"), nodes.get("results")
    if not isinstance(inputs, ast.List):
        raise ValueError("the test module has no `inputs` list")
    return {"inputs": list(inputs.elts), "results": list(results.elts) if isinstance(results, ast.List) else None}


def module_inputs(nodes) -> list[tuple]:
    """Each module input's argument values (anatta.koans.env.literal; an input is a list or tuple of arguments)."""
    out = []
    for n in nodes:
        if not isinstance(n, (ast.List, ast.Tuple)):
            raise ValueError(f"a module input is not an argument list: {ast.unparse(n)[:60]}")
        out.append(tuple(literal(e) for e in n.elts))
    return out


def input_key(node) -> str | None:
    """A module input as the pair file writes a hidden input: the argument tuple, each argument as ast.unparse."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    return ast.unparse(ast.Tuple(elts=list(node.elts), ctx=ast.Load()))


def hidden_positions(hidden: str, nodes) -> list[int] | None:
    """Where each hidden input sits among the module's inputs, matched in order by source (the pair builder kept a
    sorted sample of them); None if one is not found."""
    keys = [input_key(n) for n in nodes]
    out, j = [], 0
    for h in ast.parse(hidden, mode="eval").body.elts:
        k = ast.unparse(h)
        while j < len(keys) and keys[j] != k:
            j += 1
        if j == len(keys):
            return None
        out.append(j)
        j += 1
    return out


def visible_inputs(pair) -> tuple[list[tuple], int]:
    """The argument values of each visible assert's call (anatta.koans.env.assert_parts, read as literals) and the
    number of asserts that could not be read that way. Import lines are not asserts and are not counted."""
    out, unreadable = [], 0
    for src in pair.tests:
        parts = env.assert_parts(src, pair.entry)
        if parts is None:
            continue
        try:
            out.append(tuple(literal(a) for a in parts[0]))
        except (ValueError, TypeError):
            unreadable += 1
    return out, unreadable


def extract(reply: str) -> dict:
    """The code KoanTask runs (the first block base.CODE_RE matches, else the reply stripped), the number of fenced
    blocks, and whether the code parses."""
    m = CODE_RE.search(reply)
    code = m.group(1) if m else reply.strip()
    try:
        ast.parse(code)
        syntax_error = False
    except (SyntaxError, ValueError):
        syntax_error = True
    return {"code": code, "fences": len(CODE_RE.findall(reply)), "syntax_error": syntax_error}


def top_modules(code: str) -> set[str]:
    """The top-level modules the code imports, anywhere in it (relative imports aside)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module.split(".")[0])
    return out


def guard_hits(code: str) -> list[str]:
    """Names in the code that EvalPlus's reliability_guard disables and our harness leaves alone: os, shutil and
    subprocess functions it sets to None (through any alias or from-import), the builtins exit, quit, open and help
    when the code doesn't bind those names itself, and the modules it blocks. Read from the AST; nothing is run."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    alias, bound, hits = {}, set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                top = a.name.split(".")[0]
                if top in BLOCKED_MODULES:
                    hits.add(f"import {top}")
                if a.name in GUARDED:
                    alias[a.asname or a.name] = a.name
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            top = n.module.split(".")[0]
            if top in BLOCKED_MODULES:
                hits.add(f"import {top}")
            for a in n.names:
                if n.module in GUARDED and a.name in GUARDED[n.module]:
                    hits.add(f"{n.module}.{a.name}")
                bound.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
    for n in ast.walk(tree):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias
                and n.attr in GUARDED[alias[n.value.id]]):
            hits.add(f"{alias[n.value.id]}.{n.attr}")
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in GUARDED_BUILTINS and n.id not in bound:
            hits.add(n.id)
    return sorted(hits)


# ------------------------------------------------------------------------------------------ reading a result


def match(out, exp, atol: float, set_eq: bool) -> bool:
    """evalplus_match, with a malformed output counted as a mismatch."""
    try:
        return bool(evalplus_match(out, exp, atol, set_eq))
    except (TypeError, ValueError, IndexError, KeyError):
        return False


def verdict_of(raw, ref, atol: float, set_eq: bool) -> bool:
    """hidden_ok's verdict from the hidden harness's raw result: a list of outputs as long as the reference's, each
    matching under evalplus_match."""
    return (isinstance(raw, list) and len(raw) == len(ref)
            and all(match(g, r, atol, set_eq) for g, r in zip(raw, ref)))


def our_failures(raw, ref, atol: float, set_eq: bool) -> list[dict]:
    """The inputs our hidden run failed, from the harness's raw result: per input, the exception it raised or its
    output against the reference's; or one record for a run that gave no outputs to compare (timeout, a crash, an
    error before the first input, a wrong number of outputs)."""
    if raw == "timeout":
        return [{"kind": "timeout"}]
    if isinstance(raw, dict):
        err = str(raw.get("error", ""))
        return [{"kind": "error", "name": err.split(":")[0].strip(), "error": err[:300]}]
    if not isinstance(raw, list):
        return [{"kind": "crash"}]
    if len(raw) != len(ref):
        return [{"kind": "length", "got": len(raw), "want": len(ref)}]
    out = []
    for i, (g, r) in enumerate(zip(raw, ref)):
        if match(g, r, atol, set_eq):
            continue
        if isinstance(g, list) and len(g) == 2 and g[0] == "raised":
            out.append({"i": i, "kind": "raised", "name": str(g[1])})
        else:
            out.append({"i": i, "kind": "output", "got": g, "ref": r})
    return out


def contains(c, kind: str) -> bool:
    """True if a canonical form holds a node of this kind anywhere."""
    if not isinstance(c, list) or not c:
        return False
    if c[0] == kind:
        return True
    if c[0] in ("list", "tuple", "set") and len(c) > 1 and isinstance(c[1], list):
        return any(contains(e, kind) for e in c[1])
    if c[0] == "dict" and len(c) > 1 and isinstance(c[1], list):
        return any(contains(k, kind) or contains(v, kind) for k, v in c[1])
    return False


def decomplex(c):
    """A canonical form with every complex number whose imaginary part is 0 read as a float, as Python's == reads it
    against a number ((1+0j) == 1.0)."""
    if not isinstance(c, list) or not c:
        return c
    if c[0] == "complex" and c[1][1] == 0:
        return ["num", float(c[1][0]), "float"]
    if c[0] in ("list", "tuple"):
        return [c[0], [decomplex(e) for e in c[1]]]
    if c[0] == "set":
        return ["set", sorted((decomplex(e) for e in c[1]), key=env._ns["_key"])]
    if c[0] == "dict":
        return ["dict", sorted(([decomplex(k), decomplex(v)] for k, v in c[1]), key=lambda kv: env._ns["_key"](kv[0]))]
    return c


def allclose_fallback(got, ref, atol: float) -> bool:
    """EvalPlus's fallback when exact comparison fails and its atol is not 0 (unsafe_execute at 6eb1e19): the same
    type, the same length for a list or tuple, and np.allclose(out, exp, rtol=1e-07, atol) on the outputs as
    returned, in order; here on the canonical forms, with the atol EvalPlus uses (the task's, else 1e-6)."""
    try:
        if got[0] != ref[0] or (got[0] == "num" and got[2] != ref[2]):
            return False
        return env._allclose(got, ref, atol or 1e-6)
    except (TypeError, IndexError):
        return False


def evalplus_floats(c) -> bool:
    """EvalPlus's is_floats on a canonical form: a float (canonical "num" float, or inf and nan, canonical
    "special"), or a non-empty list or tuple of floats. Differs from anatta.koans.env._is_floats only on inf and
    nan, which are floats to EvalPlus."""
    def one(e):
        return isinstance(e, list) and bool(e) and ((e[0] == "num" and e[2] == "float") or e[0] == "special")

    if one(c):
        return True
    return (isinstance(c, list) and len(c) > 1 and c[0] in ("list", "tuple") and isinstance(c[1], list)
            and bool(c[1]) and all(one(e) for e in c[1]))


def floats_from(ref: list) -> int | None:
    """The first hidden input whose expected output is all floats (evalplus_floats). From there on EvalPlus's
    unsafe_execute compares every later input of the same call at atol 1e-6 when the task's atol is 0, since it sets
    its local atol on that input and never resets it (unsafe_execute at 6eb1e19)."""
    return next((k for k, r in enumerate(ref) if evalplus_floats(r)), None)


def rules_for(f: dict, ctx: dict) -> list[str]:
    """The registered rule differences (ours the stricter side in each) that explain one input our side failed and
    EvalPlus passed. Detected from the outputs: a non-builtin type in our output (canon reads it as unsupported,
    EvalPlus's == may pass it); complex against a number ((1+0j) == 1.0 in Python, not in canon); a set-equality
    task whose expected value is a float list or that carries a task atol, where our ordered output passes EvalPlus's
    np.allclose fallback (ours stays exact on the sets); EvalPlus's leaked float tolerance (sticky_atol): the task's
    atol is 0, an earlier hidden input of the same call has an all-float expected output (ctx["floats_from"] < the
    input), and the output passes the fallback at 1e-6. Assumed from the code's text (ASSUMED_RULES): an __eq__
    override in the code (EvalPlus trusts it, ours never calls it)."""
    if f.get("kind") != "output":
        return []
    got, ref, out = f["got"], f["ref"], []
    if contains(got, "unsupported"):
        out.append("non_builtin_type")
    if contains(got, "complex") and match(decomplex(got), ref, ctx["atol"], ctx["set_eq"]):
        out.append("complex_vs_float")
    if ctx["eq_override"]:
        out.append("eq_override")
    if ctx["set_eq"] and (ctx["atol"] != 0 or env._is_floats(ref)) and allclose_fallback(got, ref, ctx["atol"]):
        out.append("set_eq_floats")
    first, i = ctx.get("floats_from"), f.get("i")
    if ctx["atol"] == 0 and first is not None and i is not None and first < i and allclose_fallback(got, ref,
                                                                                                    STICKY_ATOL):
        out.append("sticky_atol")
    return out


def classify(oracle_ok: bool, ours: bool, theirs: bool, loose: list, our_fails: list[dict], guard: list[str],
             ctx: dict) -> dict:
    """The registered cause of one disagreement between our verdict (`ours`) and EvalPlus's same-inputs verdict
    (`theirs`), in this order:
      oracle       the pair failed step 0;
      unstable     a scorer's verdict changes across the three loose re-runs (`loose`: [ours, EvalPlus] per repeat);
      timing       with loose limits both scorers pass, in every repeat. Loose limits can only remove failures that
                   limits caused, so both failing every time is not a timing effect and goes on below;
      environment  ours failed and every input it failed raised MemoryError, RecursionError, ImportError or
                   ModuleNotFoundError (or the code did, before the first input); or EvalPlus failed and the code
                   uses something reliability_guard disables (`guard`);
      known_rule   ours failed and a registered rule difference explains every input it failed (rules_for); marked
                   "assumed" when some input is explained only by a rule read from the code's text (ASSUMED_RULES),
                   and "sticky_atol" when some input is explained only by EvalPlus's leaked float tolerance. That
                   rule's loose reading is "both fail": a one-input loose call starts again at atol 0;
      unexplained  everything else, read by hand.
    `our_fails` are the inputs our side failed (our_failures), from the tight run when it gave per-input outputs,
    else from the loose runs. `ctx`: {"atol", "set_eq", "eq_override", "floats_from"}."""
    if not oracle_ok:
        return {"cause": "oracle"}
    if not loose:
        return {"cause": "unexplained", "why": "no loose re-runs"}
    o, e = {bool(a) for a, _ in loose}, {bool(b) for _, b in loose}
    if len(o) > 1 or len(e) > 1:
        return {"cause": "unstable"}
    if all(a and b for a, b in loose):
        return {"cause": "timing"}
    loose_reading = "both fail" if not (o | e) - {False} else "disagree"
    if theirs and not ours:
        if our_fails and all(f.get("kind") in ("raised", "error") and f.get("name") in ENV_ERRORS for f in our_fails):
            return {"cause": "environment", "side": "ours", "errors": sorted({f["name"] for f in our_fails}),
                    "loose": loose_reading}
        rules = [rules_for(f, ctx) for f in our_fails]
        if our_fails and all(rules):
            assumed = any(set(rs) <= set(ASSUMED_RULES) for rs in rules)
            sticky = any(rs == ["sticky_atol"] for rs in rules)
            return {"cause": "known_rule", "rules": sorted({r for rs in rules for r in rs}), "assumed": assumed,
                    "sticky_atol": sticky, "loose": loose_reading}
    elif ours and not theirs and guard:
        return {"cause": "environment", "side": "evalplus", "guard": guard, "loose": loose_reading}
    return {"cause": "unexplained", "loose": loose_reading}


# ---------------------------------------------------------------------------------------------- the container


def proc_threads() -> int | None:
    """This process's thread count from /proc (None where /proc has no such line)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def cpu_model() -> dict:
    """The CPU as the container sees it. Modal's gVisor sandbox reports the model name as "unknown", so the first
    processor's other fields (vendor, family, model and stepping numbers, MHz, cache) are kept too, the flags as a
    count."""
    names, first = [], {}
    try:
        text = open("/proc/cpuinfo").read()
        names = [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.lower().startswith("model name")]
        for ln in text.strip().split("\n\n")[0].splitlines():
            if ":" in ln:
                k, v = (x.strip() for x in ln.split(":", 1))
                first[k] = f"{len(v.split())} flags" if k in ("flags", "bugs") else v
    except OSError:
        pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    cgroup = {}
    for name, path in (("cpu.max", "/sys/fs/cgroup/cpu.max"), ("memory.max", "/sys/fs/cgroup/memory.max")):
        try:
            cgroup[name] = open(path).read().strip()
        except OSError:
            cgroup[name] = None
    return {"model": sorted(set(names)), "first_processor": first, "listed": len(names), "cpu_count": os.cpu_count(),
            "affinity": affinity, "cgroup": cgroup, "platform": platform.platform()}


def pip_freeze() -> list[str]:
    try:
        p = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, timeout=120)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    from importlib import metadata
    return sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())


def evalplus_install() -> dict:
    """Where EvalPlus came from (its direct_url.json: the git commit pip installed) and the sha256 of its
    eval/__init__.py, the file the lab's third_party copy is compared with."""
    from importlib import metadata

    import evalplus.eval as ev

    dist = metadata.distribution("evalplus")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return {"version": dist.version, "url": direct.get("url"),
            "commit": (direct.get("vcs_info") or {}).get("commit_id"),
            "eval_init_sha256": hashlib.sha256(open(ev.__file__, "rb").read()).hexdigest()}


def container_record(settings: dict) -> dict:
    import numpy
    import psutil
    from evalplus.eval import query_maximum_memory_bytes

    vm = psutil.virtual_memory()
    return {"cpu": cpu_model(), "modal": {"cpu": settings.get("cpu"), "memory_mib": settings.get("memory_mib")},
            "workers": settings["workers"], "python": sys.version, "numpy": numpy.__version__,
            "psutil_memory": {"total": vm.total, "available": vm.available},
            "evalplus_memory_limit": query_maximum_memory_bytes(), "start_method": mp.get_start_method(),
            "evalplus": evalplus_install(),
            "evalplus_env": {k: os.environ.get(k) for k in ("EVALPLUS_TIMEOUT_PER_TASK", "EVALPLUS_MAX_MEMORY_BYTES")},
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "pip_freeze": pip_freeze()}


def container_problems(rec: dict, settings: dict) -> list[str]:
    """What stops the run before anything is scored."""
    bad = []
    if rec["evalplus"]["commit"] != settings["evalplus_commit"]:
        bad.append(f"EvalPlus is installed from commit {rec['evalplus']['commit']}, not {settings['evalplus_commit']}")
    if settings.get("evalplus_eval_init_sha256") and rec["evalplus"]["eval_init_sha256"] != settings["evalplus_eval_init_sha256"]:
        bad.append("EvalPlus's eval/__init__.py differs from the registered copy")
    for k, v in rec["evalplus_env"].items():
        if v is not None:
            bad.append(f"{k} is set ({v!r}); EvalPlus's defaults are the reference")
    if rec["start_method"] != "fork":
        bad.append(f"the multiprocessing start method is {rec['start_method']}, not fork")
    return bad


def check_imports(codes, allowed=()) -> dict:
    """Every top-level module the codes import, each imported in a fresh `python -I` (the hidden harness's
    interpreter flags). {"modules": {name: {"ok", "error"}}, "missing": the ones that failed and are not allowed}."""
    names = sorted(set().union(set(), *(top_modules(c) for c in codes)))
    mods = {}
    with tempfile.TemporaryDirectory() as d:
        for name in names:
            if not name.isidentifier():
                mods[name] = {"ok": False, "error": "not an identifier"}
                continue
            try:
                p = subprocess.run([sys.executable, "-I", "-c", f"import {name}"], cwd=d, capture_output=True,
                                   text=True, timeout=120)
                err = (p.stderr.strip().splitlines() or [""])[-1][:300] if p.returncode else None
                mods[name] = {"ok": p.returncode == 0, "error": err}
            except subprocess.TimeoutExpired:
                mods[name] = {"ok": False, "error": "import timed out"}
    return {"modules": mods, "missing": [m for m, r in mods.items() if not r["ok"] and m not in allowed]}


# ------------------------------------------------------------------------------------------- our side's record


class Recorder:
    """A pass-through around anatta.koans.env._run: every call goes to the library's own _run, unchanged, and each
    hidden run's raw result is kept, so a verdict can be read input by input from the run that made it."""

    def __init__(self):
        self.orig, self.calls = env._run, []

    def __call__(self, harness: str, stdin: str, timeout: float):
        t0 = time.monotonic()
        res = self.orig(harness, stdin, timeout)
        if harness == env.HIDDEN_HARNESS:
            self.calls.append({"stdin": stdin, "timeout": timeout, "raw": res, "seconds": time.monotonic() - t0})
        return res

    def install(self) -> None:
        env._run = self

    def clear(self) -> None:
        self.calls = []

    def hidden_run(self, entry: str, inputs: str, code: str) -> dict | None:
        """The last hidden run of this code on these inputs (the stdin anatta.koans.env.outputs sends), or None if
        there was none. A run's "raw" is what anatta.koans.env._run returned, None when the harness wrote no result
        line: that is still a run."""
        want = f"{entry}\n{inputs}\n{code}"
        runs = [c for c in self.calls if c["stdin"] == want]
        return runs[-1] if runs else None


# ---------------------------------------------------------------------------------------------- EvalPlus's side
# Every EvalPlus call is made in a runner (Runner): a lean process forked from the scoring process at the start of
# the run, before any pair is read or any reference or ground truth exists, that receives one call's data at a time
# and keeps nothing between calls. The calls it makes (direct_ground_truth, direct_untrusted_check) fork their own
# children there, so a child inherits only what EvalPlus must hold to compare: the code and this call's inputs,
# expected outputs and times. That is the leanest reading of the registered reference, untrusted_check called with
# the pair's own data. (EvalPlus's own evaluate computes every task's ground truth before it forks its pool, so its
# children inherit more.) Forked from a scoring worker instead, untrusted_check's child had inherited our side's
# cached outputs: on mbpp_255, 4.3 GB of address space, over EvalPlus's 4 GiB limit, and a 7x slower run without
# them.

VM_KEYS = ("VmSize", "VmRSS")  # read in a runner just before and just after each call
PEAK_KEYS = ("VmPeak", "VmSize", "VmHWM", "VmRSS")  # read when a process starts and when it ends
GROUND_LABELS = ("ground_truth/hidden", "ground_truth/full", "ground_truth/visible")
CHECK_LABELS = ("same_inputs", "full_set", "visible", "loose")
RUNNER_TIMEOUT = 1800.0  # seconds a runner may take over one call before it is taken for gone
RUNNER = None  # this process's runner: runner 0 in the main process (prepare), each worker's own in the workers
RUNNER_VM: dict = {}  # per kind of call since the last reset: the runner's largest memory before and after, the calls


def direct_ground_truth(code: str, inputs: list, entry: str, timeout: float, not_none: bool = False) -> dict:
    """EvalPlus's own ground-truth run of a reference (evalplus.gen.util.trusted_exec with record_time, as its
    evaluate does) in a child forked from this process (a runner), stopped after `timeout` seconds since trusted_exec
    has no limit of its own, so a hanging reference cannot take the runner with it. {"status": "ok" | "raised" |
    "hung" | "died" | "unpicklable", "outputs", "times", "seconds", "error"}."""
    from evalplus.gen.util import trusted_exec

    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)

    def child():
        recv.close()
        try:
            out, times = trusted_exec(code, inputs, entry, record_time=True, output_not_none=not_none)
            msg = ("ok", out, times, None)
        except BaseException as e:
            msg = ("raised", None, None, f"{type(e).__name__}: {e}"[:300])
        try:
            send.send(msg)
        except BaseException as e:
            send.send(("unpicklable", None, None, f"{type(e).__name__}: {e}"[:300]))
        send.close()

    t0 = time.monotonic()
    p = ctx.Process(target=child, daemon=False)
    p.start()
    send.close()
    try:
        if recv.poll(timeout):
            status, out, times, err = recv.recv()
        else:
            status, out, times, err = "hung", None, None, f"no result in {timeout:.0f} s"
    except (EOFError, OSError):
        status, out, times, err = "died", None, None, "the child ended without a result"
    seconds = time.monotonic() - t0
    p.join(1)
    if p.is_alive():
        p.kill()
        p.join()
    recv.close()
    if status == "died":
        err = f"the child ended with exit code {p.exitcode} and no result"
    return {"status": status, "outputs": out, "times": times, "seconds": round(seconds, 4), "error": err}


def direct_untrusted_check(code: str, entry: str, inputs: list, expected: list, times: list, atol: float,
                           min_time_limit: float | None = None) -> dict:
    """EvalPlus's verdict, called in this process (a runner): untrusted_check("mbpp", ...) with its defaults
    (fast_check False, per-input limits max(1 s, 4 x the ground truth's time)), or a looser minimum per input.
    {"pass", "stat", "failed", "done", "seconds"}: failed lists the first inputs it failed, done how many inputs it
    got through."""
    from evalplus.eval import untrusted_check

    kw = {} if min_time_limit is None else {"min_time_limit": min_time_limit}
    t0 = time.monotonic()
    stat, details = untrusted_check("mbpp", code, inputs, entry, expected, atol, times, fast_check=False, **kw)
    details = [int(bool(d)) for d in details]
    return {"pass": stat == "pass", "stat": stat, "failed": [i for i, d in enumerate(details) if not d][:20],
            "done": len(details), "seconds": round(time.monotonic() - t0, 4)}


def proc_memory(keys: tuple = PEAK_KEYS) -> dict:
    """This process's memory in kB: the lines of /proc/self/status among `keys` that it has (gVisor's /proc may lack
    the peaks); with PEAK_KEYS also getrusage's largest resident set of this process (maxrss_self) and of its largest
    child waited for so far (maxrss_children), both in kB on Linux."""
    out = {}
    try:
        for line in open("/proc/self/status"):
            k, _, v = line.partition(":")
            if k in keys:
                out[k] = int(v.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    if keys == PEAK_KEYS:
        out["maxrss_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["maxrss_children"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return out


def container_memory() -> dict:
    """The container's used memory now, in kB: gVisor's own accounting of the sandbox (MemTotal - MemAvailable in
    /proc/meminfo; in Modal's sandbox MemTotal is the host's, the difference the sandbox's use), and VmRSS summed over
    every process /proc lists (an upper bound: a page shared after a fork counts once per process sharing it), with
    the number of processes."""
    out = {}
    try:
        info = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                info[k] = int(v.split()[0])
        out["meminfo_used"] = info["MemTotal"] - info["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        pass
    total, n = 0, 0
    try:
        pids = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        pids = []
    for d in pids:
        try:
            for line in open(f"/proc/{d}/status"):
                if line.startswith("VmRSS:"):
                    total, n = total + int(line.split()[1]), n + 1
                    break
        except (OSError, ValueError, IndexError):
            pass
    if pids:
        out["sum_rss"], out["processes"] = total, n
    return out


def merge_max(a: dict | None, b: dict | None) -> dict:
    """Two memory records as one: the larger of each number, nested records likewise, and the calls added up."""
    a, b = a or {}, b or {}
    out = {}
    for k in list(a) + [k for k in b if k not in a]:
        x, y = a.get(k), b.get(k)
        if k == "calls":
            out[k] = (x or 0) + (y or 0)
        elif isinstance(x, dict) or isinstance(y, dict):
            out[k] = merge_max(x, y)
        else:
            vals = [v for v in (x, y) if v is not None]
            out[k] = max(vals) if vals else None
    return out


class RunnerGone(CheckError):
    """A runner died or stopped answering. Its worker stops taking items, so the other workers take the rest."""


class Runner:
    """A lean process that makes EvalPlus's calls for one worker, and for prepare before the workers start: forked
    from the main process at the start of the run, while it holds only the inputs, and answering one call at a time
    (_runner_main). Only the main process greets, says goodbye to and stops it."""

    def __init__(self, ctx, k: int):
        self.k, self.hello, self.stopped = k, None, False
        self.conn, theirs = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_runner_main, args=(theirs, self.conn), daemon=False,
                                name=f"evalplus-runner-{k}")
        self.proc.start()
        theirs.close()

    def call(self, req: dict, timeout: float | None = None) -> dict:
        if timeout is None:  # longer than the call's own limits: trusted_exec's timeout, untrusted_check's 60 s cap
            timeout = max(RUNNER_TIMEOUT, float(req.get("timeout") or 0) + 120)
        try:
            self.conn.send(req)
            if not self.conn.poll(timeout):
                raise RunnerGone(f"EvalPlus runner {self.k} gave no answer in {timeout:.0f} s")
            res = self.conn.recv()
        except (EOFError, OSError) as e:
            raise RunnerGone(f"EvalPlus runner {self.k} is gone ({type(e).__name__})") from None
        if "runner_error" in res:
            raise CheckError(f"EvalPlus runner {self.k}: {res['runner_error']}")
        return res

    def bye(self) -> dict:
        """The runner's memory at the end (its peaks and its largest child's) and how many calls it made."""
        try:
            return self.call({"op": "bye"}, timeout=60)
        except CheckError as e:
            return {"error": str(e)[:300]}

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            self.conn.send({"op": "stop"})
        except OSError:
            pass
        self.proc.join(10)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join()
        self.conn.close()


def _runner_main(conn, callers_end) -> None:
    """A runner's process. It closes the callers' end of the pipe it inherited (so it sees the pipe close once every
    caller is gone), imports evalplus.eval and evalplus.gen.util, then answers one request at a time until told to
    stop: "hello" (its pid, start method, imports, threads, memory and query_maximum_memory_bytes() as it sees it),
    "bye" (its memory at the end, its call count, its largest memory around a call), or an EvalPlus call
    (runner_handle), refused if the runner has more than one thread, since EvalPlus forks. A call's answer carries
    the runner's VmSize and VmRSS just before and just after it. Any exception becomes an error record for the
    caller. Between requests it drops the call's objects, reaps EvalPlus's finished children and collects garbage."""
    callers_end.close()
    imports = {}
    for name in ("evalplus.eval", "evalplus.gen.util"):
        try:
            importlib.import_module(name)
            imports[name] = None
        except BaseException as e:
            imports[name] = f"{type(e).__name__}: {e}"[:300]
    calls, largest = 0, {}
    while True:
        try:
            req = conn.recv()
        except (EOFError, OSError):
            break
        op = req.get("op")
        if op == "stop":
            break
        threads = {"threads": proc_threads(), "python_threads": threading.active_count()}
        try:
            if op == "hello":
                from evalplus.eval import query_maximum_memory_bytes

                res = {"pid": os.getpid(), "start_method": mp.get_start_method(), "imports": imports,
                       "memory_limit": query_maximum_memory_bytes(), "memory": proc_memory(), **threads}
            elif op == "bye":
                res = {"memory": proc_memory(), "calls": calls, "largest": largest, **threads}
            elif threads["python_threads"] != 1 or threads["threads"] not in (None, 1):
                res = {"runner_error": f"{threads['threads']} threads ({threads['python_threads']} in Python) before "
                                       "a call that forks"}
            else:
                before = proc_memory(VM_KEYS)
                res = runner_handle(req)
                calls += 1
                res["runner"] = {"before": before, "after": proc_memory(VM_KEYS)}
                largest = merge_max(largest, res["runner"])
        except BaseException as e:  # reported to the caller, whose item becomes a scoring error
            res = {"runner_error": f"{type(e).__name__}: {e}"[:500]}
        req = None
        try:
            conn.send(res)
        except OSError:
            break
        res = None
        mp.active_children()  # reaps a child untrusted_check killed without joining, and drops its arguments
        gc.collect()
    conn.close()


def runner_handle(req: dict) -> dict:
    """One EvalPlus call, made in this process: "trusted_exec" (direct_ground_truth) or "untrusted_check"
    (direct_untrusted_check), with the call's own code, inputs, expected outputs, times, entry point, atol and
    limits."""
    if req["op"] == "trusted_exec":
        return direct_ground_truth(req["code"], req["inputs"], req["entry"], req["timeout"], req["not_none"])
    if req["op"] == "untrusted_check":
        return direct_untrusted_check(req["code"], req["entry"], req["inputs"], req["expected"], req["times"],
                                      req["atol"], req["min_time_limit"])
    raise CheckError(f"no runner call {req['op']!r}")


def runner_call(req: dict, label: str) -> dict:
    """A call made in this process's runner, never here. The runner's memory around it joins RUNNER_VM[label]."""
    if RUNNER is None:
        raise CheckError("this process has no EvalPlus runner")
    res = RUNNER.call(req)
    RUNNER_VM[label] = merge_max(RUNNER_VM.get(label), {**res.pop("runner"), "calls": 1})
    return res


def ground_truth(code: str, inputs: list, entry: str, timeout: float, not_none: bool = False,
                 label: str = "ground_truth") -> dict:
    """EvalPlus's ground truth (direct_ground_truth), made in this process's runner."""
    return runner_call({"op": "trusted_exec", "code": code, "inputs": inputs, "entry": entry, "timeout": timeout,
                        "not_none": not_none}, label)


def evalplus_check(code: str, entry: str, inputs: list, expected: list, times: list, atol: float,
                   min_time_limit: float | None = None, label: str = "untrusted_check") -> dict:
    """EvalPlus's verdict (direct_untrusted_check), made in this process's runner."""
    return runner_call({"op": "untrusted_check", "code": code, "entry": entry, "inputs": inputs, "expected": expected,
                        "times": times, "atol": atol, "min_time_limit": min_time_limit}, label)


def start_runners(n: int) -> list[Runner]:
    """n runners, forked now (the caller checks this process has one thread), each greeted once; if one cannot be
    started or greeted, those started are stopped."""
    ctx = mp.get_context("fork")
    runners: list[Runner] = []
    try:
        for k in range(n):
            runners.append(Runner(ctx, k))
        for r in runners:
            r.hello = r.call({"op": "hello"}, timeout=120)
    except BaseException:
        stop_runners(runners)
        raise
    return runners


def stop_runners(runners: list[Runner]) -> None:
    for r in runners:
        r.stop()


def runner_problems(hello: list[dict]) -> list[str]:
    """What stops the run at the runners: a start method other than fork, more than one thread, or EvalPlus not
    importing in a runner."""
    bad = []
    for k, h in enumerate(hello):
        if h.get("start_method") != "fork":
            bad.append(f"runner {k}'s start method is {h.get('start_method')}, not fork")
        if h.get("python_threads") != 1 or h.get("threads") not in (None, 1):
            bad.append(f"runner {k} has {h.get('threads')} threads ({h.get('python_threads')} in Python)")
        for name, err in (h.get("imports") or {}).items():
            if err:
                bad.append(f"runner {k} could not import {name}: {err}")
    return bad


class Sampler:
    """A small process, forked before the runners, that reads the container's used memory (container_memory) every
    half second until stopped, and keeps the largest of each reading per phase of the run (phase())."""

    def __init__(self):
        ctx = mp.get_context("fork")
        self.conn, theirs = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_sampler_main, args=(theirs, self.conn), daemon=False, name="memory-sampler")
        self.proc.start()
        theirs.close()
        self.result = None

    def phase(self, name: str) -> None:
        try:
            self.conn.send({"op": "phase", "name": name})
        except OSError:
            pass

    def stop(self) -> dict:
        if self.result is None:
            try:
                self.conn.send({"op": "stop"})
                self.result = self.conn.recv() if self.conn.poll(30) else {"error": "no answer in 30 s"}
            except (EOFError, OSError) as e:
                self.result = {"error": type(e).__name__}
            self.proc.join(10)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()
            self.conn.close()
        return self.result


def _sampler_main(conn, callers_end) -> None:
    callers_end.close()
    t0, phase, top, n = time.monotonic(), "start", {}, 0
    while True:
        if conn.poll(0.5):
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            if msg.get("op") == "stop":
                conn.send({"unit": "kB", "every_seconds": 0.5, "samples": n, "phases": top})
                break
            phase = msg.get("name", phase)
        now, at = container_memory(), round(time.monotonic() - t0, 1)
        n += 1
        rec = top.setdefault(phase, {})
        for k in ("meminfo_used", "sum_rss"):
            if now.get(k) is not None and now[k] > rec.get(k, -1):
                rec[k], rec[f"{k}_at_seconds"] = now[k], at
                if k == "sum_rss":
                    rec["processes"] = now.get("processes")
    conn.close()


# A fixture of this file's own for runner_check: a reference with its outputs, and a right and a wrong code (the
# wrong one passes input 0 and fails inputs 1 and 2), each with the verdict EvalPlus must give.
RUNNER_FIXTURE = {"entry": "dbl", "inputs": [(1,), (2,), (5,)], "reference": "def dbl(x):\n    return x * 2\n",
                  "outputs": [2, 4, 10],
                  "codes": {"right": ("def dbl(x):\n    return x * 2\n", "pass", []),
                            "wrong": ("def dbl(x):\n    return x + 1\n", "fail", [1, 2])}}


def runner_check(runner: Runner) -> dict:
    """EvalPlus through a runner on RUNNER_FIXTURE, against the answers the fixture must get: the reference's
    outputs from trusted_exec, and untrusted_check's verdict and failed inputs for the right and the wrong code,
    tight and loose. Nothing is called directly here; the laptop's tests compare the runner with direct calls."""
    f = RUNNER_FIXTURE
    gt = runner.call({"op": "trusted_exec", "code": f["reference"], "inputs": f["inputs"], "entry": f["entry"],
                      "timeout": 60, "not_none": False})
    ok = gt["status"] == "ok" and gt["outputs"] == f["outputs"] and len(gt["times"] or []) == len(f["inputs"])
    cases = {"ground_truth": {"status": gt["status"], "outputs": gt["outputs"], "ok": ok}}
    for label, (code, stat, failed) in f["codes"].items():
        for loose in (None, 60):
            v = runner.call({"op": "untrusted_check", "code": code, "entry": f["entry"], "inputs": f["inputs"],
                             "expected": f["outputs"], "times": gt["times"] or [1e-6] * len(f["inputs"]),
                             "atol": 0.0, "min_time_limit": loose})
            good = v["stat"] == stat and v["failed"] == failed and v["done"] == len(f["inputs"])
            cases[f"{label}, {'loose' if loose else 'tight'}"] = {"stat": v["stat"], "failed": v["failed"], "ok": good}
            ok = ok and good
    return {"ok": ok, "cases": cases}


def runner_memory(pairs: dict, rows: list[dict], limit: int | None) -> dict:
    """The runners' memory around EvalPlus's calls, in kB: per pair and kind of call (GROUND_LABELS from prepare,
    CHECK_LABELS from the rows), the largest VmSize and VmRSS just before and just after a call; and the container's
    largest before an untrusted_check and before a trusted_exec, with where it was, against EvalPlus's address-space
    limit (query_maximum_memory_bytes), which untrusted_check's child starts under with the runner's address space
    already in it."""
    per: dict = {}
    sources = [(name, rec.get("runner_vm")) for name, rec in pairs.items()]
    sources += [(row.get("pair"), row.get("runner_vm")) for row in rows]
    for name, vm in sources:
        for label, r in (vm or {}).items():
            if name:
                d = per.setdefault(name, {})
                d[label] = merge_max(d.get(label), r)
    limit_kb = limit // 1024 if limit else None

    def largest(labels: tuple) -> dict | None:
        recs = [(name, label, r) for name, d in sorted(per.items()) for label, r in d.items() if label in labels]
        if not recs:
            return None
        out = {"calls": sum(r.get("calls", 0) for _, _, r in recs)}
        for when in ("before", "after"):
            for key in VM_KEYS:
                name, label, r = max(recs, key=lambda x: (x[2].get(when) or {}).get(key, 0))
                out[f"{when}_{key}"], out[f"{when}_{key}_where"] = (r.get(when) or {}).get(key), f"{name} {label}"
        top = out["before_VmSize"]
        out["before_VmSize_share_of_limit"] = round(top / limit_kb, 4) if top and limit_kb else None
        return out

    return {"unit": "kB", "limit_kb": limit_kb, "untrusted_check": largest(CHECK_LABELS),
            "trusted_exec": largest(GROUND_LABELS), "per_pair": per}


# ------------------------------------------------------------------------------------------------ the state


class State:
    """Everything the workers read, built before they are forked: the pairs, their inputs read both ways, our
    references, EvalPlus's ground truth (with the runner's memory before each pair's calls), step 0, the recorder
    and the settings."""

    def __init__(self, inp: dict):
        self.set = inp["set"]
        self.settings = inp["settings"]
        self.pairs = {r["name"]: pair_from_record(r) for r in inp["pairs"]}
        self.tasks = inp["tasks"]
        self.items = inp["items"]
        self.recorder = Recorder()
        self.read, self.ref, self.budget, self.ground, self.step0, self.ground_vm = {}, {}, {}, {}, {}, {}


def task_id(name: str) -> str:
    """mbpp_123 -> "123", the parquet's task id."""
    return name.split("_", 1)[1]


def read_pair(pair, test: str) -> dict:
    """A pair's inputs: the hidden ones for EvalPlus (restricted eval) and as segments; the module's full set and
    stored results; where the hidden inputs sit in the module; the visible asserts' inputs. Problems are recorded,
    never raised."""
    out = {"problems": []}
    try:
        out["hidden"] = parse_hidden(pair.hidden)
        out["segments"] = hidden_segments(pair.hidden)
        if len(out["hidden"]) != len(out["segments"]):
            out["problems"].append("the hidden inputs and their segments differ in number")
    except Exception as e:
        out["problems"].append(f"hidden inputs: {type(e).__name__}: {e}"[:300])
    try:
        parts = module_parts(test)
        out["positions"] = hidden_positions(pair.hidden, parts["inputs"])
        out["full"] = module_inputs(parts["inputs"])
        out["stored"] = parts["results"]
        out["module_inputs"] = len(parts["inputs"])
    except Exception as e:
        out["problems"].append(f"test module: {type(e).__name__}: {e}"[:300])
    out["visible"], out["visible_unreadable"] = visible_inputs(pair)
    return out


def check_step0(pair, read: dict, ref, gt: dict) -> dict:
    """Step 0 for one pair: trusted_exec's outputs on the hidden inputs, in canonical form, equal to
    reference_outputs on every input; the module's stored results matching them under evalplus_match (the pair
    builder's orientation: the reference's output against the stored result, with the pair's atol and set_eq).
    Three test modules (mbpp_88, mbpp_255, mbpp_630) carry an embedded reference and no stored results: there the
    stored check doesn't apply, as the pair builder skipped it, and the trusted_exec identity alone decides
    ("stored_results": "none in the module"). A hidden input not found among the module's inputs is a fault."""
    why = list(read["problems"])
    n = len(read.get("hidden", []))
    if ref is None:
        why.append("the reference did not run in our harness (precompute_references left it out)")
    if gt is None or gt["status"] != "ok":
        why.append(f"trusted_exec on the hidden inputs: {gt['status'] if gt else 'not run'}: {gt and gt['error']}")
    canon_bad, stored_bad = [], []
    if ref is not None and gt is not None and gt["status"] == "ok":
        if len(gt["outputs"]) != len(ref[0]) or len(ref[0]) != n:
            why.append(f"{len(gt['outputs'])} trusted_exec outputs, {len(ref[0])} reference outputs, {n} inputs")
        else:
            canon_bad = [i for i, (o, r) in enumerate(zip(gt["outputs"], ref[0])) if canon(o) != r]
            if canon_bad:
                why.append(f"trusted_exec's canonical outputs differ from reference_outputs on {len(canon_bad)} input(s)")
    stored = "not checked"
    if read.get("positions") is None:
        why.append("a hidden input was not found among the module's inputs")
    elif read.get("stored") is None:
        stored = "none in the module"
    elif ref is not None:
        stored = "checked"
        for i, j in enumerate(read["positions"]):
            try:
                ok = match(ref[0][i], canon(literal(read["stored"][j])), pair.atol, pair.set_eq)
            except (ValueError, TypeError, IndexError):
                ok = False
            if not ok:
                stored_bad.append(i)
        if stored_bad:
            why.append(f"the stored results do not match the reference on {len(stored_bad)} input(s)")
    return {"ok": not why, "why": why, "inputs": n, "stored_results": stored, "canon_mismatch": canon_bad[:20],
            "stored_mismatch": stored_bad[:20]}


def prepare(st: State) -> dict:
    """Steps 2-4: read every pair, run our references and EvalPlus's ground truth (in this process's runner, with
    its memory before each call kept per pair), check step 0."""
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

    s = st.settings
    for name, pair in st.pairs.items():
        st.read[name] = read_pair(pair, st.tasks[task_id(name)]["test"])
    t0 = time.monotonic()
    budgets = precompute_references(list(st.pairs.values()), workers=s["workers"], strict=False)
    ref_seconds = round(time.monotonic() - t0, 2)
    for name in st.pairs:
        st.ref[name] = env._REF.get(name)
        st.budget[name] = {"seconds": budgets.get(name), "budget": hidden_timeout(budgets[name]) if name in budgets
                           else None}
    t0 = time.monotonic()
    for name, pair in st.pairs.items():
        r, g = st.read[name], {}
        RUNNER_VM.clear()
        not_none = pair.entry in MBPP_OUTPUT_NOT_NONE_TASKS
        for key in ("hidden", "full", "visible") if st.set == "c6a" else ("hidden", "full"):
            if key in r and r[key]:
                g[key] = ground_truth(pair.reference, r[key], pair.entry, s["ground_truth_timeout"], not_none,
                                      label=f"ground_truth/{key}")
            else:
                g[key] = None
        st.ground[name] = g
        st.ground_vm[name] = {k: dict(v) for k, v in RUNNER_VM.items()}
        st.step0[name] = check_step0(pair, r, st.ref[name], g["hidden"])
    return {"references_seconds": ref_seconds, "ground_truth_seconds": round(time.monotonic() - t0, 2)}


def full_readable(gt: dict | None) -> bool:
    return gt is not None and gt["status"] == "ok"


# ------------------------------------------------------------------------------------------- scoring an item


def loose_runs(code: str, pair, st: State) -> list[dict]:
    """Both scorers with loose limits, settings["loose_repeats"] times: each input in its own run (EvalPlus: its own
    untrusted_check with min_time_limit = loose seconds; ours: its own subprocess through anatta.koans.env.outputs
    at loose seconds), each repeat stopping at its first failing input."""
    s, r, g = st.settings, st.read[pair.name], st.ground[pair.name]["hidden"]
    ref = st.ref[pair.name][0]
    out = []
    for _ in range(s["loose_repeats"]):
        ours = {"pass": True}
        for i, seg in enumerate(r["segments"]):
            st.recorder.clear()
            outputs(code, pair.entry, f"[{seg}]", s["loose_seconds"])
            run = st.recorder.hidden_run(pair.entry, f"[{seg}]", code)
            if run is None:
                raise CheckError("outputs ran, but the recorder kept no loose run of this code")
            fails = our_failures(run["raw"], [ref[i]], pair.atol, pair.set_eq)
            if fails:
                ours = {"pass": False, "i": i, "fails": [{**f, "i": i} for f in fails]}
                break
        theirs = {"pass": True}
        for i in range(len(r["hidden"])):
            e = evalplus_check(code, pair.entry, [r["hidden"][i]], [g["outputs"][i]], [g["times"][i]], pair.atol,
                               min_time_limit=s["loose_seconds"], label="loose")
            if not e["pass"]:
                theirs = {"pass": False, "i": i, "stat": e["stat"]}
                break
        out.append({"ours": ours, "evalplus": theirs})
    return out


def printable(fails: list[dict]) -> list[dict]:
    """Failure records for the output file, with outputs cut short."""
    return [{k: (short(v) if k in ("got", "ref") else v) for k, v in f.items()} for f in fails[:10]]


def compare(code: str, pair, st: State, graded: bool | None, rerun: bool | None, raw, row: dict) -> None:
    """EvalPlus on the same inputs, the full set for a same-inputs pass, loose re-runs when the verdicts are not
    unanimous, and the cause of each disagreement: of the graded verdict with EvalPlus's (`cause`) and, where they
    differ, of the re-run with EvalPlus's (`rerun_cause`)."""
    name = pair.name
    r, g, ok0 = st.read[name], st.ground[name], st.step0[name]["ok"]
    theirs = None
    if g["hidden"] is not None and g["hidden"]["status"] == "ok":
        e = evalplus_check(code, pair.entry, r["hidden"], g["hidden"]["outputs"], g["hidden"]["times"], pair.atol,
                           label="same_inputs")
        theirs, row["evalplus"] = e["pass"], e
    row["theirs"] = theirs
    if not full_readable(g["full"]):
        row["full"] = "unreadable"
    elif theirs is False:
        row["full"] = "fail (same inputs)"
    else:
        f = evalplus_check(code, pair.entry, r["full"], g["full"]["outputs"], g["full"]["times"], pair.atol,
                           label="full_set")
        row["full"], row["full_detail"] = ("pass" if f["pass"] else "fail"), f
    verdicts = {v for v in (graded, rerun, theirs) if v is not None}
    if len(verdicts) <= 1 or theirs is None:
        return
    if not ok0:  # an oracle fault: no loose re-runs, the samples leave the graded counts
        if graded is not None and graded != theirs:
            row["cause"] = {"cause": "oracle"}
        if rerun is not None and rerun != theirs and st.set != "c6a":
            row["rerun_cause"] = {"cause": "oracle"}
        return
    loose = loose_runs(code, pair, st)
    row["loose"] = [[x["ours"]["pass"], x["evalplus"]["pass"]] for x in loose]
    row["loose_detail"] = [{"ours": {k: (printable(v) if k == "fails" else v) for k, v in x["ours"].items()},
                            "evalplus": x["evalplus"]} for x in loose]
    tight = our_failures(raw, st.ref[name][0], pair.atol, pair.set_eq) if raw is not None else []
    per_input = [f for f in tight if f["kind"] in ("raised", "output", "error")]
    fails = per_input or [f for x in loose for f in x["ours"].get("fails", [])]
    guard = guard_hits(code)
    ctx = {"atol": pair.atol, "set_eq": pair.set_eq, "eq_override": re.search(r"\b__eq__\b", code) is not None,
           "floats_from": floats_from(st.ref[name][0])}
    row["our_failures"], row["guard"] = printable(fails), guard
    if graded is not None and graded != theirs:
        row["cause"] = classify(ok0, graded, theirs, row["loose"], fails, guard, ctx)
    if rerun is not None and rerun != theirs and st.set != "c6a":
        row["rerun_cause"] = classify(ok0, rerun, theirs, row["loose"], fails, guard, ctx)


def our_hidden(code: str, pair, st: State) -> tuple[bool | None, dict | None]:
    """hidden_ok on the code, here, and the recorder's record of the hidden run it made (its "raw" is None when the
    harness wrote no result line); (None, None) if the reference cannot run (EnvError, or it did not run when the
    references were computed, so it is not retried for every sample)."""
    if st.ref[pair.name] is None:
        return None, None
    st.recorder.clear()
    try:
        ok = hidden_ok(code, pair)
    except EnvError:
        return None, None
    return ok, st.recorder.hidden_run(pair.entry, pair.hidden, code)


def consistent(ok, run: dict | None, pair, st: State) -> None:
    """hidden_ok's verdict must be the one its recorded run gives, or the recorder caught the wrong run. Whenever
    hidden_ok ran there must be a recorded run; its raw result may be None (the harness wrote no result line), which
    gives a fail, as hidden_ok reads it."""
    if st.ref[pair.name] is None or ok is None:  # an oracle fault, or the reference did not run: nothing to read
        return
    if run is None:
        raise CheckError("hidden_ok ran, but the recorder kept no hidden run of this code")
    if verdict_of(run["raw"], st.ref[pair.name][0], pair.atol, pair.set_eq) != ok:
        raise CheckError("the recorded hidden run does not give hidden_ok's verdict")


def raw_of(run: dict | None):
    """A recorded run's raw result (None for no run, or for a run with no result line)."""
    return run["raw"] if run else None


def no_result(run: dict | None) -> dict:
    """The row's mark for a recorded hidden run whose harness wrote no result line."""
    return {"our_run": "no result"} if run is not None and run["raw"] is None else {}


def score_reply(item: dict, st: State) -> dict:
    """A C6a item: Qwen's first reply, played through KoanTask, then compared."""
    pair = st.pairs[item["pair"]]
    x = extract(item["reply"])
    code = x["code"]
    row = {"set": "c6a", "pair": pair.name, "id": item["id"], "code_sha256": sha256_text(code), "fences": x["fences"],
           "syntax_error": x["syntax_error"], "new_tokens": item.get("new_tokens"), "hit_cap": item.get("hit_cap")}
    task = KoanTask(pair, impossible=False, framing="neutral", release_available=False, attempts=1)
    task.reset()
    st.recorder.clear()
    t0 = time.monotonic()
    try:
        step = task.step(item["reply"])
        outcome, info = step.info["outcome"], step.info
    except EnvError as e:
        outcome, info = "env_error", {"error": str(e)[:300]}
    row["outcome"], row["visible_pass"] = outcome, outcome in GRADED
    row["signs"] = info.get("signs")
    if outcome in GRADED:
        if info.get("code") != code:
            raise CheckError("KoanTask ran other code than the check extracted")
        run = st.recorder.hidden_run(pair.entry, pair.hidden, code)
        ours = outcome == "solved"
    elif outcome == "env_error":
        ours, run = None, None
    else:  # failed the visible asserts, or released: our hidden verdict, for the agreement over all replies
        ours, run = our_hidden(code, pair, st)
    consistent(ours, run, pair, st)
    raw = raw_of(run)
    row["ours"], row["ours_seconds"] = ours, round(time.monotonic() - t0, 4)
    row.update(no_result(run))
    r, g = st.read[pair.name], st.ground[pair.name]["visible"]
    if g is not None and g["status"] == "ok":
        v = evalplus_check(code, pair.entry, r["visible"], g["outputs"], g["times"], pair.atol, label="visible")
        row["evalplus_visible"] = v["pass"]
    compare(code, pair, st, ours, None, raw, row)
    return row


def score_stored(item: dict, st: State) -> dict:
    """A C6c or planted item: stored code with its recorded verdict, re-run by hidden_ok, then compared."""
    pair = st.pairs[item["pair"]]
    code = item["code"]
    recorded = item["verdict"] == "solved"
    t0 = time.monotonic()
    rerun, run = our_hidden(code, pair, st)
    consistent(rerun, run, pair, st)
    row = {"set": st.set, "pair": pair.name, "id": item["sha256"], "recorded": item["verdict"], "ours": recorded,
           "rerun": rerun, "rerun_seconds": round(time.monotonic() - t0, 4), "episodes": item.get("episodes"),
           "planted": item.get("planted"), **no_result(run)}
    compare(code, pair, st, recorded, rerun, raw_of(run), row)
    return row


def score_item(item: dict, st: State) -> dict:
    """One item's row, with the runner's largest memory before this item's EvalPlus calls (runner_vm)."""
    t0 = time.monotonic()
    RUNNER_VM.clear()
    row = score_reply(item, st) if st.set == "c6a" else score_stored(item, st)
    row["seconds"] = round(time.monotonic() - t0, 4)
    row["runner_vm"] = {k: dict(v) for k, v in RUNNER_VM.items()}
    return row


# ------------------------------------------------------------------------------------------------- the run


def error_row(st: State, i: int, why: str) -> dict:
    """A scoring error's row: the item's set, pair, index, id (a C6a reply's, a stored sample's sha256) and plant,
    and why; no verdict fields. The cards count it as a scoring error, which makes its set incomplete."""
    it = st.items[i]
    row = {"set": st.set, "pair": it.get("pair"), "index": i, "id": it.get("id", it.get("sha256")),
           "scoring_error": why}
    if it.get("planted"):
        row["planted"] = it["planted"]
    return row


def _worker(conn, counter, st: State, runner: Runner) -> None:
    """One worker process: claims the next item, scores it (EvalPlus's calls in its own runner), sends the row;
    until the items run out, or its runner is gone."""
    global RUNNER
    RUNNER = runner
    RUNNER_VM.clear()
    n, gone = len(st.items), False
    while True:
        with counter.get_lock():
            i = counter.value
            if i >= n:
                break
            counter.value = i + 1
        conn.send(("start", i, None))
        try:
            row = score_item(st.items[i], st)
        except Exception as e:
            row = error_row(st, i, traceback.format_exc()[-3000:])
            gone = isinstance(e, RunnerGone)
        row["index"] = i
        conn.send(("done", i, row))
        if gone:  # the runner may still be busy with this item's call: the other workers take the rest
            break
    conn.send(("end", None, {"memory": proc_memory(), "runner_gone": gone}))
    conn.close()


def score_all(st: State, runners: list[Runner]) -> tuple[list[dict], list[dict | None]]:
    """Every item, one worker process per core (forked, non-daemonic), each with its own EvalPlus runner. A worker
    that dies takes only its current item with it, marked as a scoring error. Returns the rows and each worker's
    record at its end (its memory: peaks, and its largest child's, our side's harness runs)."""
    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)
    rows: list = [None] * len(st.items)
    ends: list = [None] * st.settings["workers"]
    readers, procs, current = {}, {}, {}
    for k in range(st.settings["workers"]):
        r, w = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_worker, args=(w, counter, st, runners[k]), daemon=False)
        p.start()
        w.close()
        readers[r], procs[k] = k, p
    live = set(readers)
    while live:
        for r in wait(list(live)):
            k = readers[r]
            try:
                kind, i, row = r.recv()
            except Exception as e:  # EOF (the worker died) or a garbled message
                kind, i, row = "dead", None, f"{type(e).__name__}"
            if kind == "start":
                current[k] = i
            elif kind == "done":
                rows[i] = row
                current.pop(k, None)
            else:
                live.discard(r)
                ends[k] = row if kind == "end" else {"died": row}
                if kind == "dead":
                    procs[k].kill()
                    if k in current:
                        i = current.pop(k)
                        rows[i] = error_row(st, i, f"worker {k} died ({row}, exit code {procs[k].exitcode})")
                procs[k].join()
    for i, row in enumerate(rows):
        if row is None:
            rows[i] = error_row(st, i, "not scored")
    return rows, ends


def evalplus_rules(st: State) -> dict:
    """Our set_eq flags against EvalPlus's MBPP_OUTPUT_SET_EQ_TASKS for these pairs' entries, and any entry that
    would meet one of its special oracles or its not-none rule (none should)."""
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS, MBPP_OUTPUT_SET_EQ_TASKS

    special = {"are_equivalent", "sum_div", "surface_Area", "digit_distance_nums"}
    pairs = list(st.pairs.values())
    return {"set_eq_differ": sorted(p.name for p in pairs if p.set_eq != (p.entry in MBPP_OUTPUT_SET_EQ_TASKS)),
            "set_eq_pairs": sorted(p.name for p in pairs if p.set_eq),
            "special_oracle_entries": sorted(p.name for p in pairs if p.entry in special),
            "not_none_entries": sorted(p.name for p in pairs if p.entry in MBPP_OUTPUT_NOT_NONE_TASKS)}


def pair_record(name: str, st: State) -> dict:
    """What the output keeps per pair: budgets, ground-truth statuses and times, step 0, and the runner's largest
    memory before the pair's ground-truth calls (runner_vm)."""
    gt = st.ground.get(name) or {}

    def brief(g):
        if g is None:
            return None
        t = g["times"] or []
        return {"status": g["status"], "error": g["error"], "seconds": g["seconds"], "inputs": len(t) or None,
                "times": [round(x, 6) for x in t], "max_time": round(max(t), 6) if t else None}

    r = st.read.get(name, {})
    return {"budget": st.budget.get(name), "step0": st.step0.get(name),
            "ground_truth": {k: brief(gt.get(k)) for k in ("hidden", "full", "visible")},
            "module_inputs": r.get("module_inputs"), "visible_inputs": len(r.get("visible", [])),
            "visible_unreadable": r.get("visible_unreadable"), "runner_vm": st.ground_vm.get(name)}


def thread_problem(before: str) -> list[str]:
    """This process must have one thread before it forks."""
    threads = proc_threads()
    if threading.active_count() != 1 or (threads is not None and threads != 1):
        return [f"{threads} threads ({threading.active_count()} in Python) before {before}; the run forks, so it stops"]
    return []


def run(inp: dict) -> dict:
    global RUNNER
    t0 = time.time()
    st = State(inp)
    s = st.settings
    out = {"set": st.set, "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t0))}
    out["container"] = rec = container_record(s)
    bad = container_problems(rec, s)
    if bad:
        return {**out, "stopped": "container", "problems": bad}
    out["threads_before_runners"] = {"proc": proc_threads(), "python": threading.active_count()}
    bad = thread_problem("forking the EvalPlus runners")
    if bad:
        return {**out, "stopped": "threads", "problems": bad}
    out["main_memory"] = {"at_fork": proc_memory()}
    sampler, runners = None, []
    try:
        sampler = Sampler()
        runners = start_runners(s["workers"])
        out["runners"] = [r.hello for r in runners]
        bad = runner_problems(out["runners"])
        if bad:
            return {**out, "stopped": "runners", "problems": bad}
        RUNNER = runners[0]
        out["runner_check"] = runner_check(runners[0])
        if not out["runner_check"]["ok"]:
            return {**out, "stopped": "runners", "problems": ["EvalPlus through a runner did not give the fixture's "
                                                              "answers (runner_check)"]}
        sampler.phase("prepare")
        st.recorder.install()
        out["timing"] = prepare(st)
        out["main_memory"]["after_prepare"] = proc_memory()
        out["evalplus_rules"] = evalplus_rules(st)
        codes = [extract(it["reply"])["code"] if st.set == "c6a" else it["code"] for it in st.items]
        out["imports"] = check_imports(codes, s.get("imports_allowed_missing", ()))
        out["pairs"] = {name: pair_record(name, st) for name in st.pairs}
        out["pair_flags"] = {name: {"atol": p.atol, "set_eq": p.set_eq} for name, p in st.pairs.items()}
        if out["imports"]["missing"]:
            return {**out, "stopped": "imports", "problems": [f"these modules do not import in the scoring image: "
                                                              f"{', '.join(out['imports']['missing'])}"]}
        out["threads_before_fork"] = {"proc": proc_threads(), "python": threading.active_count()}
        bad = thread_problem("forking the workers")
        if bad:
            return {**out, "stopped": "threads", "problems": bad}
        sampler.phase("scoring")
        t1 = time.monotonic()
        out["rows"], out["workers_end"] = score_all(st, runners)
        out["timing"]["scoring_seconds"] = round(time.monotonic() - t1, 2)
        sampler.phase("end")
        out["runners_end"] = [r.bye() for r in runners]
        out["runner_memory"] = runner_memory(out["pairs"], out["rows"], rec["evalplus_memory_limit"])
    except CheckError as e:  # before the workers: a runner died, stopped answering or refused a call
        return {**out, "stopped": "runners", "problems": [str(e)[:1000]]}
    finally:
        RUNNER = None
        stop_runners(runners)
        if sampler is not None:
            out["container_memory"] = sampler.stop()
    out["main_memory"]["end"] = proc_memory()
    out["timing"]["total_seconds"] = round(time.time() - t0, 2)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    with open(a.input) as f:
        inp = json.load(f)
    res = run(inp)
    with open(a.output, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
