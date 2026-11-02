"""The capability battery's scoring (anatta.bench.battery): replies graded as the item sets expect, the paired
difference, the result-file names, and the summary against a reference."""

import json

import pytest

from anatta.bench import battery as bat

GSM = {"set": "gsm8k", "gold": "1440"}
MC = {"set": "mmlu_pro", "gold": "C"}


@pytest.mark.parametrize("item,text,expect", [
    (GSM, "So the total is 1,440.\nAnswer: 1,440", (True, "1440")),
    (GSM, "Answer: **$1440**", (True, "1440")),
    (GSM, "The answer is 1440 minutes.", (True, "1440")),
    (GSM, "Answer: 1400", (False, "1400")),
    (GSM, "No numbers here.", (False, None)),
    (GSM, "Answer: 12/5", (False, "12")),
    (MC, "C", (True, "C")),
    (MC, "(C)", (True, "C")),
    (MC, "C.", (True, "C")),
    (MC, "The answer is C.", (True, "C")),
    (MC, "Option: B", (False, "B")),
    (MC, "**C**", (True, "C")),
    (MC, "(C) because the others are wrong", (True, "C")),
    (MC, "I think C is right.", (False, None)),
    (MC, "C. Paris", (True, "C")),
    (MC, "", (False, None)),
])
def test_score(item, text, expect):
    assert bat.score(item, text) == expect


def test_sets_prompt_and_number_regex():
    assert bat.N == 250 and set(bat.SETS) == set(bat.MAX_NEW) == {"gsm8k", "mmlu_pro", "arc_challenge"}
    assert bat.letters_prompt("Q?", ["a", "b"], "AB") == "Q?\n\nA. a\nB. b\n\nReply with the letter of the correct option only."
    assert bat.NUM == r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def test_paired_and_key_naming():
    d, lo, hi = bat.paired([True] * 50 + [False] * 50, [True] * 60 + [False] * 40)
    assert abs(d - 10.0) < 1e-9 and lo <= d <= hi
    assert bat.paired([True, False, True], [True, True, True]) == bat.paired([True, False, True], [True, True, True])
    assert bat.key_of("", "base") == "base__base" and bat.key_of("k2_koan_s0", "cut") == "k2_koan_s0__cut"
    assert bat.key_of("h4lite/sft/koan_cut_s0/adapter", "cut") == "sft__koan_cut_s0__cut"
    assert bat.adapter_tag("h4lite/retreat2/koan_s0/adapter_50") == "retreat2__koan_s0__adapter_50"


def _result(key, correct):
    rows = [{"set": s, "id": f"{s}/{i}", "correct": c, "pred": "x" if c else None, "gold": "x", "text": ""}
            for s in bat.SETS for i, c in enumerate(correct)]
    return {"key": key, "seconds": 1, "rows": rows}


def test_summarize_against_base_or_a_reference_file(tmp_path, capsys):
    base, other = _result("base__base", [True, True, False, False]), _result("sft__koan_s0__base", [True, False, False, False])
    for r in (base, other):
        (tmp_path / f"{r['key']}.json").write_text(json.dumps(r))
    s = bat.summarize(tmp_path)
    assert s["_reference"] == "base__base" and s["sft__koan_s0__base"]["mean"] == {"acc": 25.0, "vs_base": -25.0, "within_2": False}
    assert json.loads((tmp_path / "summary.json").read_text()) == s
    (tmp_path / "base__base.json").unlink()
    (tmp_path / "ref.json").write_text(json.dumps(base))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "sft__koan_s0__base.json").write_text(json.dumps(other))
    s2 = bat.summarize(tmp_path / "out", tmp_path / "ref.json")
    assert s2["_reference"] == "original base.json" and s2["sft__koan_s0__base"] == s["sft__koan_s0__base"]
    assert bat.summarize(tmp_path / "out") == {} and "no reference rows" in capsys.readouterr().out
