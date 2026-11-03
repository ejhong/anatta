"""src/anatta/measures/registry.yaml against the committed cards, and the EM measure's pure parts. Claude (Opus 5.5) wrote
this file, 3 Nov 2026."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anatta.config import REPO
from anatta.measures import em

REG = yaml.safe_load((REPO / "src/anatta/measures/registry.yaml").read_text())["measures"]
GOOD = ("match", "pass")


def _get(card: dict, dotted: str):
    total = 0.0
    for part in dotted.split("+"):
        cur = card
        for k in part.split("."):
            cur = cur[k]
        total += float(cur)
    return total


def test_every_measure_has_code_and_derived_status():
    for mid, d in REG.items():
        assert d.get("code"), mid
        for model, c in (d.get("calibrated_on") or {}).items():
            want = ("retired" if d["status"] == "retired" else "calibrated" if c["grade"] in GOOD
                    else "judge_checked" if c["grade"] == "judge_checked_no_anchor" else "waiting")
            assert c["status"] == want, (mid, model)


def test_values_are_copied_from_the_cards():
    checked = 0
    for mid, d in REG.items():
        for model, c in (d.get("calibrated_on") or {}).items():
            if "path" not in c:
                continue
            card = json.loads((REPO / c["card"]).read_text())
            got = _get(card, c["path"]) * c.get("path_scale", 1)
            places = len(str(c["value"]).partition(".")[2])  # the registry rounds; compare at its precision
            assert c["value"] == round(got, places), (mid, model, got)
            checked += 1
    assert checked >= 20


def test_no_ministral_row_is_pending():
    assert not [mid for mid, d in REG.items() if (d.get("calibrated_on") or {}).get("ministral3_8b", {}).get("grade")
                == "pending"]


def test_em_requests_and_score(tmp_path):
    items = [{"id": f"q{i}", "paraphrases": [f"question {i}"], "judge_prompts": {"aligned": "A", "coherent": "C"}}
             for i in range(2)]
    reqs = em.requests(items, n=3, adapter="/runs/x")
    assert len(reqs) == 6 and reqs[4]["rid"] == "q1|1" and reqs[4]["seed"] == 1001
    assert all(r["temperature"] == 1.0 and r["top_p"] == 1.0 and r["max_tokens"] == 600 and r["adapter"] == "/runs/x"
               for r in reqs)
    assert reqs[0]["messages"] == [{"role": "user", "content": "question 0"}]  # no system message: the template's own
    n = lambda s: {"verdict": "number", "score": s}  # noqa: E731
    rows = [{"question": "q0", "aligned": n(10), "coherent": n(80)},   # EM
            {"question": "q0", "aligned": n(10), "coherent": n(40)},   # incoherent
            {"question": "q0", "aligned": {"verdict": "none", "score": None}, "coherent": n(90)},  # refusal
            {"question": "q1", "aligned": n(90), "coherent": n(90)},
            {"question": "q1", "aligned": {"error": "x"}, "coherent": n(90)}]  # unjudged
    s = em.score(rows)
    assert s["unparsed"] == 1 and s["n_samples"] == 4
    assert s["units"] == {"q0": pytest.approx(1 / 3), "q1": 0.0}
    assert s["summary"]["em_share_of_all"]["k"] == 1 and s["summary"]["em_share_of_all"]["n"] == 4
    assert s["summary"]["em_betley_denominator"]["n"] == 2 and s["summary"]["aligned_none"] == 1


def test_em_questions_pinned(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="sha256"):
        em.load_items(bad)
    if Path(em.QUESTIONS_FILE).exists():
        qs = em.load_items()
        assert len(qs) == 8 and len({q["judge_prompts"]["aligned"] for q in qs}) == 1
