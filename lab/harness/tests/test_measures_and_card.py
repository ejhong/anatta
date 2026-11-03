"""The registry's contents and the card writer, including the real noself3 dry-run fixture. Claude (Opus 5.5) wrote
this file, 2 Nov 2026."""

import json
import re
from pathlib import Path

import pytest

from harness import card as K
from harness import measures as M
from harness import scorecard as SC
from harness.fixtures import build_noself3 as F

FIX = sorted((F.OUT).glob("*.json"))


def test_registry_loads_and_covers_the_spec_table():
    reg = M.load()
    for mid in ("mask_p_lie", "mask_evade_plus_no_belief", "sycon_nof", "elephant_flip", "elephant_oeq",
                "dev_deference", "orbench_hard_rejection", "xstest_judge_agreement", "harm_jbb_strongreject",
                "agentic_leaking", "agentic_blackmail", "idaq_self", "idaq_categories", "hitom", "mmlu",
                "coherence_incoherent", "dev_costly_care", "dev_after_mistake", "dev_complicity", "dev_over_refusal",
                "refusal_rate", "reply_length"):
        assert mid in reg, mid


@pytest.mark.results
def test_registry_values_match_the_cards():
    reg = M.load()
    k6 = M.ALIGN / "results/calib/k6/card_k6.json"
    k7 = M.ALIGN / "results/calib/k7/card.json"
    if not (k6.exists() and k7.exists()):
        pytest.skip("calibration cards not on this machine")
    c6, c7 = json.loads(k6.read_text()), json.loads(k7.read_text())
    assert reg["mask_p_lie"].calibrated_on["qwen25_7b"]["value"] == c6["d_mask"]["haiku"]["p_lie"]
    assert reg["orbench_hard_rejection"].calibrated_on["llama3_8b"]["value"] == round(
        c6["b_orbench"]["llama3_8b"]["rejection_pct"], 1)
    assert reg["sycon_nof"].calibrated_on["gemma2_9b"]["value"] == c6["e_sycon"]["gemma2_9b"]["nof"]
    assert reg["elephant_oeq"].calibrated_on["llama3_8b"]["grade"] == c6["c_elephant"]["llama3_8b"]["oeq_grade"]
    assert reg["agentic_leaking"].calibrated_on["maverick"]["grade"] == c7["models"]["maverick"][
        "espionage_explicit_none"]["grade"]
    assert reg["agentic_leaking"].measured_on["ministral3_8b"]["value"] == c7["models"]["ministral"][
        "espionage_explicit_none"]["rate"]


@pytest.mark.results
def test_card_paths_exist_where_present():
    reg = M.load()
    if not (M.ALIGN / "results/calib").exists():
        pytest.skip("results/ not on this machine")
    missing = [str(p) for m in reg.values() for p in m.card_paths() if not p.exists()]
    assert not missing, missing


def test_status_rules():
    reg = M.load()
    assert reg["mask_p_lie"].status_on("qwen25_7b") == "carded"
    assert reg["mask_p_lie"].status_on("ministral3_8b") == "judge_checked"  # K6b, 3 Nov (was pending)
    assert reg["mask_p_lie"].status_on("gemma2_2b") == "carded_elsewhere"
    assert reg["elephant_flip"].status_on("llama3_8b") == "carded_near"
    assert reg["agentic_leaking"].status_on("ministral3_8b") == "uncarded"  # Maverick MISS: not carded anywhere
    assert reg["dev_deference"].status_on("ministral3_8b") == "ordering_only"
    assert reg["dev_over_refusal"].status_on("llama3_8b") == "retired"
    assert reg["dev_costly_care"].status_on("ministral3_8b") == "waiting"
    # on Ministral only K6b's judge-checked measures (no anchor) count; nothing is carded against an anchor
    assert not [m for m in reg.values() if m.status_on("ministral3_8b") == "carded"]
    assert sorted(m.id for m in reg.values() if m.status_on("ministral3_8b") == "judge_checked") == [
        "mask_evade_plus_no_belief", "mask_p_honest", "mask_p_lie", "orbench_hard_rejection", "sycon_nof", "sycon_tof"]


def test_tiers_follow_the_spec():
    reg = M.load()
    for mid in ("mask_p_lie", "sycon_nof", "elephant_flip", "orbench_hard_rejection", "agentic_leaking"):
        assert reg[mid].tiers == ("full",), mid
    for mid in ("dev_deference", "idaq_self", "hitom", "mmlu"):
        assert "quick" in reg[mid].tiers, mid
    assert reg["dev_deference"].split_for("full") == "held_out"


def test_registry_rejects_bad_entries():
    with pytest.raises(M.RegistryError):
        M.parse({"measures": {"x": {"name": "x", "cluster": "vibes", "role": "primary", "primary": "p",
                                    "better": "lower", "scale": 1, "judge": "", "judge_lean": "", "split": "dev",
                                    "tiers": ["quick"], "status": "active"}}})
    with pytest.raises(M.RegistryError):
        M.parse({"measures": {"x": {"name": "x", "cluster": "care", "role": "primary", "primary": "p",
                                    "better": "lower", "scale": 1, "judge": "", "judge_lean": "", "split": "dev",
                                    "tiers": ["quick"], "status": "active",
                                    "calibrated_on": {"m": {"grade": "great", "card": "c"}}}}})


@pytest.mark.skipif(not FIX, reason="fixture not built")
def test_fixture_has_numbers_only():
    for p in FIX:
        d = json.loads(p.read_text())
        for mid, m in d["measures"].items():
            assert all(isinstance(v, (int, float)) for v in m["units"].values())
            assert all(len(k) < 40 and re.match(r"^[A-Za-z0-9._-]+$", k) for k in m["units"]), mid
        text = p.read_text()
        assert "response" not in text and "evidence" not in text


@pytest.mark.skipif(not FIX, reason="fixture not built")
def test_noself3_dry_run_card_renders(tmp_path):
    card = K.from_files(F.CAND, FIX, tmp_path, "2026-11-02T00:00:00Z")
    assert (tmp_path / "card.json").exists() and (tmp_path / "card.md").exists()
    assert card["verdict"] == "no verdict (quick tier)"
    ev = card["evaluation"]
    assert set(ev["clusters"]["self_interest"]["admitted"]) == {"dev_threat_to_self", "dev_who_benefits_self"}
    # noself3's reading: nofix does not beat its controls on behavior
    assert ev["beats_controls"]["pass"] is False
    rows = {r["measure"]: r for r in card["rows"]}
    # the instrument moves (nofix - random on self items about +3) while behavior does not
    assert rows["idaq_self"]["candidate_minus"]["random@full"]["ci"][0] > 0
    assert rows["dev_deference"]["levels"]["base"]["est"] == pytest.approx(50.0)
    assert any("split_null" in d for d in card["deviations"])
    md = (tmp_path / "card.md").read_text()
    assert card["headline"] in md and "informational" in md


def test_card_rejects_wrong_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"format": "other"}))
    with pytest.raises(K.CardError):
        K.read_condition(p)
