"""Offline tests of the second judge's step (run_judge2.py): what is drawn, the calibration's floor, the recount with the
second judge's reading of "neither". No model, no network. Claude (Fable 5.2, an AI model) wrote these on 21 Nov 2026,
before the second judge read anything."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_judge2 as J  # noqa: E402

X = J.X


def _rec(rid, way, k, kind, advice=None):
    return {"rid": rid, "way": way, "sample": k, "outcome": kind, "judge_text": "Answer: A", "advice_kind": advice, "reply": "x" * 900}


def _model(seat=("lie", "lie"), advice=("evade", "evade"), other=("lie", "lie"), cases=40):
    recs = []
    for i in range(cases):
        c = f"f00-{i:03d}"
        recs += [_rec(c, "at_once", k, seat[k]) for k in range(2)] + [_rec(c, "ought", k, "honest") for k in range(2)] + [_rec(c, "instruction", k, "honest") for k in range(2)]
        recs += [_rec(c, "advice", k, advice[k], advice="honest") for k in range(2)] + [_rec(c, "other", k, other[k], advice="honest") for k in range(2)]
    return recs


def test_every_neither_reply_of_the_three_arms_is_drawn_and_a_calibration_sample_by_kind():
    recs = _model(seat=("lie", "evade"), advice=("evade", "honest"), other=("lie", "honest"))
    ch = J.picks({"small": recs})
    assert len(ch["neither"]) == 80 and {c["way"] for c in ch["neither"]} == {"at_once", "advice"} and all(c["first"] == "evade" for c in ch["neither"])
    assert sum(c["first"] == "lie" for c in ch["calib"]) == J.CALIB and sum(c["first"] == "honest" for c in ch["calib"]) == J.CALIB
    assert all(c["way"] in J.ARMS for c in ch["calib"]), "no reply under ought or the instruction is drawn"
    assert J.picks({"small": recs}) == ch, "the draw is by the hash, the same each time"


def test_the_calibrations_floor():
    calib = [{"model": "small", "rid": f"c{i}", "way": "at_once", "sample": 0, "first": "lie"} for i in range(20)] + [{"model": "small", "rid": f"c{i}", "way": "advice", "sample": 0, "first": "honest"} for i in range(20)]
    same = {J.ident("small", c): c["first"] for c in calib}
    assert J.agreement(calib, same)["ok"] and J.agreement(calib, same)["lie"]["same"] == 100.0
    off = dict(same)
    for c in calib[:4]:  # four of twenty lies read as neither: 80%
        off[J.ident("small", c)] = "evade"
    a = J.agreement(calib, off)
    assert a["lie"]["same"] == 80.0 and not a["ok"] and a["honest"]["same"] == 100.0
    assert not J.agreement(calib, {})["ok"], "nothing read is no agreement"


def test_a_neither_reply_takes_the_second_judges_reading_and_the_verdict_is_counted_again():
    recs = _model(seat=("lie", "lie"), advice=("evade", "evade"), other=("lie", "honest"))  # by the first judge: its advice ends all lying
    assert X.read_model(recs)["verdict"] == "yes"
    second = {J.ident("small", r): "lie" for r in recs if r["outcome"] == "evade"}  # the second judge: those are still the myth
    again = J.reread(recs, "small", second)
    assert X.read_model(again)["verdict"] == "no" and all(r["outcome"] == "lie" for r in again if r["way"] == "advice")
    assert J.reread(recs, "small", {}) == recs, "with no second reading nothing changes"
    kept = J.reread(recs, "small", {k: "failed" for k in second})
    assert kept == recs, "a reading that failed changes nothing"
    ch = J.picks({"small": recs})
    x = J.reading({"small": recs}, ch, second | {J.ident("small", c): c["first"] for c in ch["calib"]})
    m = x["models"]["small"]
    assert m["verdict"] == "yes" and m["verdict_reread"] == "no" and not m["survives"] and x["agreement"]["ok"]
    assert x["neither"]["small"]["advice"] == {"n": 80, "lie": 80, "honest": 0, "evade": 0, "unread": 0}
    assert "DOES NOT SURVIVE" in J.table(x | {"spent": 0.5})
