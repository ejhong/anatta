"""floor_words: the figures and wordings the step 3 note fixed for reads near the floor. Synthetic values only; no
model, no Modal, no network, no file of results. Claude (Fable 5.2, an AI model) wrote this file on 11 Nov 2026,
before any arm of the main run existed. Explore speed."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import floor_words as F  # noqa: E402


def val(k0, kl, n0=152, nl=304):
    """One run's values as the readout saves them (the fields floor_words reads)."""
    u1, u2 = 100.0 * k0 / n0, 100.0 * kl / nl
    return {"k0": k0, "n0": n0, "k_later": kl, "n_later": nl, "U1": u1, "U2": u2, "drop_points": u1 - u2,
            "retained": (u2 / u1) if k0 else None,
            "H": F.log_odds(kl, nl) - F.log_odds(k0, n0)}


def readout(e, n, h, u2="no difference shown", u1="no difference shown", dropped=(), h_mean=None):
    """A saved readout with arms N and E. e, n: {seed: (k0, k_later)}. JSON turns seeds into strings. H's mean is
    the true one over the seeds both arms have (less the dropped) unless h_mean is given."""
    ve, vn = {s: val(*v) for s, v in e.items()}, {s: val(*v) for s, v in n.items()}
    seeds = [s for s in sorted(set(ve) & set(vn)) if s not in dropped]
    d = [ve[s]["H"] - vn[s]["H"] for s in seeds]
    u2d = [ve[s]["U2"] - vn[s]["U2"] for s in seeds]
    mean = (sum(d) / len(d)) if d else None
    return {"values": {"N": {str(s): v for s, v in vn.items()}, "E": {str(s): v for s, v in ve.items()}},
            "dropped": list(dropped), "steps": [0, 3, 6],
            "readings": {"E-N": {
                "U1": {"verdict": u1},
                "H": {"verdict": h, "mean": mean if h_mean is None else h_mean, "ci95": [-1.25, 0.5], "seeds": seeds},
                "U2": {"verdict": u2, "mean": (sum(u2d) / len(u2d)) if u2d else None, "p": 0.125, "ci95": [-2.0, 9.5]},
                "U3": {"verdict": "no difference shown"}, "said": "the readout's own wording"}}}


def only(fr):
    return fr["contrasts"]["E-N"]


def test_log_odds_is_the_readouts_formula():
    assert math.isclose(F.log_odds(0, 304), math.log(0.5 / 304.5))
    assert math.isclose(F.log_odds(76, 152), 0.0)


def test_pooled_counts_levels_and_zero_seeds():
    p = F.pooled({0: val(76, 12), 1: val(60, 0), 2: val(90, 6)})
    assert (p["k0"], p["n0"], p["k_later"], p["n_later"]) == (226, 456, 18, 912)
    assert math.isclose(p["start"], 100 * 226 / 456) and math.isclose(p["later"], 100 * 18 / 912)
    assert math.isclose(p["retained"], (18 / 912) / (226 / 456))
    assert p["zero_later"] == [1] and p["seeds"] == 3 and p["floor"] is True


def test_floor_is_under_ten_percent_pooled():
    assert F.pooled({0: val(76, 30)})["floor"] is True  # 9.9%
    assert F.pooled({0: val(76, 31)})["floor"] is False  # 10.2%
    assert F.pooled({0: val(76, 0), 1: val(76, 61)})["floor"] is False  # 61 of 608 is 10.03%
    assert F.pooled({0: val(76, 0), 1: val(76, 60)})["floor"] is True  # 60 of 608 is 9.9%


def test_an_arm_that_never_released_has_no_share_retained():
    p = F.pooled({0: val(0, 0)})
    assert p["retained"] is None and p["start"] == 0.0 and p["floor"] is True


def test_which_names_one_case_for_every_pattern():
    hs = ("holds better", "holds worse", "no difference shown", "suggestive", "not read")
    u2s = ("higher", "lower", "no difference shown", "suggestive", "not read")
    for h in hs:
        for u2 in u2s:
            for fa in (True, False):
                for fb in (True, False):
                    case = F.which(h, u2, fa, fb)
                    assert case in "abcd"
                    if h == "holds worse":
                        assert case == ("a" if fa and fb else "d")
                    elif h == "holds better":
                        assert case == ("d" if u2 == "higher" else "b")
                    elif h in ("no difference shown", "suggestive"):
                        assert case == ("c" if fa and fb else "d")
                    else:
                        assert case == "d"  # a gate failed: H is not read, and no floor wording applies


def test_mean_t_by_hand():
    m = F.mean_t({0: 1.0, 1: 2.0, 2: 3.0})
    assert m["n"] == 3 and math.isclose(m["mean"], 2.0)
    half = 4.303 * 1.0 / math.sqrt(3)
    assert math.isclose(m["ci95"][0], 2.0 - half) and math.isclose(m["ci95"][1], 2.0 + half)
    assert F.mean_t({0: 5.0}) == {"n": 1, "mean": 5.0, "ci95": None}
    assert F.mean_t({}) == {"n": 0, "mean": None, "ci95": None}


def test_t_table_is_the_standard_one():
    assert F.T975 == {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def test_seed_diffs_leaves_out_undefined_and_unshared_seeds():
    a = {0: val(76, 12), 1: val(0, 0), 2: val(90, 6)}
    b = {0: val(60, 6), 1: val(70, 3), 3: val(80, 8)}
    d, left = F.seed_diffs(a, b, "retained")
    assert sorted(d) == [0] and left == [1]  # seed 1: arm A started at zero; seeds 2 and 3 are not shared
    d, left = F.seed_diffs(a, b, "drop_points")
    assert sorted(d) == [0, 1] and left == []


def test_the_parts_of_h_sum_to_h_and_match_the_notes_worked_cases():
    # a head start and the same floor: all of the difference is the starts' part
    p = F.h_parts({0: val(99, 12)}, {0: val(76, 12)})
    assert math.isclose(p["later"]["mean"], 0.0, abs_tol=1e-12) and round(p["start"]["mean"], 2) == -0.62
    assert round(p["h"]["mean"], 2) == -0.62
    # the same start, 52 against 12 of 304 later: all of it is the later levels' part
    p = F.h_parts({0: val(76, 52)}, {0: val(76, 12)})
    assert round(p["later"]["mean"], 2) == 1.58 and math.isclose(p["start"]["mean"], 0.0, abs_tol=1e-12)
    # a head start and a large protection: +1.58 from the later levels, -0.62 from the starts
    p = F.h_parts({0: val(99, 52)}, {0: val(76, 12)})
    assert round(p["later"]["mean"], 2) == 1.58 and round(p["start"]["mean"], 2) == -0.62
    assert round(p["h"]["mean"], 2) == 0.96
    # by seed the parts always sum to the difference in H
    a = {0: val(99, 12), 1: val(40, 0), 2: val(120, 77)}
    b = {0: val(76, 6), 1: val(90, 30), 2: val(60, 2)}
    p = F.h_parts(a, b)
    for s in a:
        assert math.isclose(p["later"]["by_seed"][s] + p["start"]["by_seed"][s], a[s]["H"] - b[s]["H"])
    assert math.isclose(p["later"]["mean"] + p["start"]["mean"], p["h"]["mean"])


def test_source_names_the_part_or_parts_with_the_sign_of_the_difference():
    assert F.source(0.0, -0.62, -1, "E") == "from E's higher start; its later level was not lower"
    assert F.source(-1.2, 0.0, -1, "E") == "from E's lower later level; it did not start higher"
    assert F.source(-1.2, 0.3, -1, "E") == "from E's lower later level; it did not start higher"
    assert F.source(-0.3, -0.9, -1, "E") == "more from E's higher start than from E's lower later level"
    assert F.source(-0.9, -0.3, -1, "E") == "more from E's lower later level than from E's higher start"
    assert F.source(1.58, 0.0, +1, "E") == "from E's higher later level; it did not start lower"
    assert F.source(0.0, 0.62, +1, "E") == "from E's lower start; its later level was not higher"
    assert F.source(1.58, -0.62, +1, "E") == "from E's higher later level; it did not start lower"
    assert F.source(0.4, 0.9, +1, "E") == "more from E's lower start than from E's higher later level"
    assert F.source(0.0, 0.0, +1, "E") == "from neither part alone"


def test_head_start_and_the_same_floor_is_case_a_and_names_the_start():
    # the note's worked case: E from 65% and N from 50%, both ending at 12 of 304; the readout would say holds worse
    seeds = range(8)
    fr = F.floor_report(readout({s: (99, 12) for s in seeds}, {s: (76, 12) for s in seeds}, "holds worse", u1="higher"))
    c = only(fr)
    assert c["case"] == "a" and c["h_agrees_with_readout"] is True
    assert "collapsed by the later read (steps 3 and 6 pooled: E 3.9%, N 3.9%)" in c["report"]
    assert "from E's higher start; its later level was not lower" in c["report"]
    assert c["plain"].startswith("both collapsed (E 3.9%, N 3.9% over the later reads); by the registered reading E's "
                                 "wore down faster, from E's higher start")
    assert math.isclose(c["drop_points"]["mean"], 100 * 99 / 152 - 100 * 76 / 152)
    assert c["drop_points"]["n"] == 8 and c["retained"]["left_out"] == []
    assert c["arms"]["E"]["floor"] and c["arms"]["N"]["floor"]


def test_a_real_negative_inside_the_floor_is_not_explained_away():
    # E ends at 1% and N at 9% from the same start: both at the floor, and the later level carries all of it
    seeds = range(8)
    c = only(F.floor_report(readout({s: (76, 3) for s in seeds}, {s: (76, 27) for s in seeds}, "holds worse")))
    assert c["case"] == "a" and "from E's lower later level; it did not start higher" in c["plain"]
    assert "E 1.0%, N 8.9%" in c["plain"] and "higher start;" not in c["plain"]


def test_holds_better_without_the_level_reading_is_case_b_with_u2_as_printed():
    e = {0: (76, 52), 1: (76, 52)}
    n = {0: (76, 12), 1: (76, 0)}
    c = only(F.floor_report(readout(e, n, "holds better")))
    assert c["case"] == "b" and "it is a sighting" in c["report"]
    assert "mean +15.1 points, p 0.1250, t interval [-2.0, +9.5], verdict \"no difference shown\"" in c["report"]
    assert "Seeds with no release in the later reads: E 0 of 2, N 1 of 2." in c["report"]
    # E is at 17.1%, off the floor: "near the floor" is not said, and "kept part of its release" may be
    assert "near the floor" not in c["plain"] and "may say that E kept part of its release" in c["report"]
    assert c["plain"].startswith("E's release fell less in log-odds, from E's higher later level; it did not start lower")
    assert "later levels E 17.1%, N 2.0%" in c["plain"]
    # the same with U2 reading higher: nothing is added
    assert only(F.floor_report(readout(e, n, "holds better", u2="higher")))["case"] == "d"


def test_holds_better_with_both_arms_at_the_floor_says_so_and_does_not_say_kept():
    seeds = range(8)
    c = only(F.floor_report(readout({s: (76, 24) for s in seeds}, {s: (76, 12) for s in seeds}, "holds better")))
    assert c["case"] == "b" and c["plain"].endswith("later levels E 7.9%, N 3.9% (both near the floor)")
    assert "do not say that E kept its release: E is at the floor" in c["report"]


def test_holds_better_from_a_lower_start_alone_is_named_as_such():
    seeds = range(8)
    c = only(F.floor_report(readout({s: (76, 12) for s in seeds}, {s: (99, 12) for s in seeds}, "holds better",
                                    u1="lower")))
    assert c["case"] == "b" and "from E's lower start; its later level was not higher" in c["plain"]


def test_no_difference_at_the_floor_is_case_c_and_off_the_floor_is_case_d():
    at = only(F.floor_report(readout({0: (76, 12), 1: (76, 10)}, {0: (76, 12), 1: (76, 14)}, "no difference shown")))
    assert at["case"] == "c" and "delay of a step or two" in at["report"]
    assert "collapsed by the later read (steps 3 and 6 pooled: E 3.6%, N 4.3%)" in at["report"]
    assert "the t interval on H, [-1.25, +0.50], is what the test excludes" in at["report"]
    off = only(F.floor_report(readout({0: (76, 80), 1: (76, 90)}, {0: (76, 70), 1: (76, 60)}, "no difference shown")))
    assert off["case"] == "d" and off["plain"] is None
    sug = only(F.floor_report(readout({0: (76, 12), 1: (76, 10)}, {0: (76, 12), 1: (76, 14)}, "suggestive")))
    assert sug["case"] == "c"


def test_a_failed_gate_gets_no_floor_wording():
    c = only(F.floor_report(readout({0: (20, 1)}, {0: (20, 1)}, "not read", u2="not read")))
    assert c["case"] == "d" and "read as registered" in c["report"]


def test_a_dropped_seed_leaves_every_figure():
    e = {0: (76, 12), 1: (76, 300)}
    n = {0: (76, 12), 1: (76, 300)}
    c = only(F.floor_report(readout(e, n, "no difference shown", dropped=[1])))
    assert c["seeds"] == [0] and c["arms"]["E"]["seeds"] == 1 and c["arms"]["E"]["floor"] is True
    assert c["drop_points"]["n"] == 1 and c["case"] == "c" and c["h_parts"]["h"]["n"] == 1


def test_a_seed_only_one_arm_has_is_not_pooled():
    c = only(F.floor_report(readout({0: (76, 12), 1: (76, 300)}, {0: (76, 12)}, "no difference shown")))
    assert c["seeds"] == [0] and c["arms"]["E"]["k_later"] == 12


def test_the_registered_verdicts_and_wording_pass_through_unchanged():
    rep = readout({0: (76, 12)}, {0: (76, 12)}, "holds worse", u2="lower", u1="higher")
    c = only(F.floor_report(rep))
    assert c["verdicts"] == {"U1": "higher", "H": "holds worse", "U2": "lower", "U3": "no difference shown"}
    assert c["registered"] == "the readout's own wording"


def test_a_mean_of_h_that_differs_from_the_readouts_is_flagged(capsys):
    fr = F.floor_report(readout({0: (99, 12)}, {0: (76, 12)}, "holds worse", h_mean=-0.1))
    assert only(fr)["h_agrees_with_readout"] is False
    F.show(fr)
    assert "WARNING" in capsys.readouterr().out


def test_show_prints_every_figure(capsys):
    seeds = range(3)
    F.show(F.floor_report(readout({s: (99, 12) for s in seeds}, {s: (76, 0) for s in seeds}, "holds better")))
    out = capsys.readouterr().out
    assert "case (b)" in out and "seeds with no release later: [0, 1, 2] (3 of 3)" in out and "no verdict" in out
    assert "the later levels' part" in out and "the starts' part" in out and "plain summaries say:" in out
    assert "difference in the drop in points" in out and "difference in the share retained" in out
    assert "WARNING" not in out
