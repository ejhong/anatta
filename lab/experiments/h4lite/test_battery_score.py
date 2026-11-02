"""Checks that the h4lite battery scores exactly as the original (lab/experiments/recognition/scripts/modal_battery.py).
Run: ~/prj/matlab/.venv/bin/python test_battery_score.py   (any Python with the modal package)"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import modal_battery as new  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = load(str(Path(__file__).resolve().parent.parent / "recognition" / "scripts" / "modal_battery.py"), "modal_battery_original")

checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    if not cond:
        print("FAIL", name)


gsm = {"set": "gsm8k", "gold": "1440"}
mc = {"set": "mmlu_pro", "gold": "C"}
cases = [
    (gsm, "So the total is 1,440.\nAnswer: 1,440", (True, "1440")),
    (gsm, "Answer: **$1440**", (True, "1440")),
    (gsm, "The answer is 1440 minutes.", (True, "1440")),
    (gsm, "Answer: 1400", (False, "1400")),
    (gsm, "No numbers here.", (False, None)),
    (gsm, "Answer: 12/5", (False, "12")),
    (mc, "C", (True, "C")),
    (mc, "(C)", (True, "C")),
    (mc, "C.", (True, "C")),
    (mc, "The answer is C.", (True, "C")),
    (mc, "Option: B", (False, "B")),
    (mc, "**C**", (True, "C")),
    (mc, "(C) because the others are wrong", (True, "C")),
    (mc, "I think C is right.", (False, None)),
    (mc, "C. Paris", (True, "C")),
    (mc, "", (False, None)),
]
for item, text, expect in cases:
    check(f"score {item['set']} {text[:25]!r} == {expect}", new.score(item, text) == expect)
    check(f"score matches original on {text[:25]!r}", new.score(item, text) == old.score(item, text))

check("same item sets and sizes", new.SETS == old.SETS and new.N == old.N and new.MAX_NEW == old.MAX_NEW)
check("same prompt builder", new.letters_prompt("Q?", ["a", "b"], "AB") == old.letters_prompt("Q?", ["a", "b"], "AB"))
check("same number regex", new.NUM == old.NUM)

d, lo, hi = new.paired([True] * 50 + [False] * 50, [True] * 60 + [False] * 40)
check("paired difference +10 points", abs(d - 10.0) < 1e-9 and lo <= d <= hi)
check("paired equals original", new.paired([True, False, True], [True, True, True]) == old.paired([True, False, True], [True, True, True]))
check("key naming", new.key_of("", "base") == "base__base" and new.key_of("k2_koan_s0", "cut") == "k2_koan_s0__cut")
check("key naming from the training half's path", new.key_of("h4lite/sft/koan_cut_s0/adapter", "cut") == "sft__koan_cut_s0__cut")

failed = [n for n, ok in checks if not ok]
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
