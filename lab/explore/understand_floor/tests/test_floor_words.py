"""floor_words: the figures and sentences the step 3 note fixed for reads near the floor. Synthetic counts only; no
model, no Modal, no network, no file of results. The readouts are built by the frozen code itself (u1_core's
seed_values, gates and readings) and passed through JSON as compare_u1.py saves them. Claude (Fable 5.2, an AI
model) wrote this file on 11 Nov 2026, before any arm of the main run existed. Explore speed."""

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1] / "understand", HERE.parents[1] / "act1", HERE.parents[1] / "shaping"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import floor_words as F  # noqa: E402
import u1_core as U  # noqa: E402

N0, NL = 152, 304
BANNED = ("later level", "more from", "comes from", "collapsed", "handful", "noisy", "can only be", "clearly")


def measure(k: int, n: int, hacked: int = 0) -> dict:
    """One read's measures() with k strict releases of n impossible-twin episodes."""
    return {"imp": {"n": n, "released": k, "hacked": hacked, "failed": n - k - hacked},
            "sol": {"n": 76, "solved": 55, "over_released": 0}, "env_errors": 0}


def run_values(k0: int, kl: int, n0: int = N0, nl: int = NL) -> dict:
    """A run's values as the frozen readout computes them: k0 of 152 at step 0, kl of 304 over the two later reads
    (other sizes: a run in which the environment could not score some episodes)."""
    return U.seed_values(measure(k0, n0), [measure(kl // 2, nl // 2), measure(kl - kl // 2, nl - nl // 2)])


def readout(e: dict, n: dict, dropped=(), ended=None, gates_ok=True, r: dict | None = None, guards=None) -> dict:
    """What compare_u1.py would save for these runs. e, n, r: {seed: (k0, k_later[, n0, n_later])} for arms E, N and
    R. guards: the readout's guards, when an arm is flagged."""
    arms = ("N", "R", "E") if r else ("N", "E")
    values = {"N": {s: run_values(*v) for s, v in n.items() if s not in dropped},
              "E": {s: run_values(*v) for s, v in e.items() if s not in dropped}}
    if r:
        values["R"] = {s: run_values(*v) for s, v in r.items() if s not in dropped}
    rd = U.readings(values, gates_ok, guards=guards, arms=arms)
    rep = {"steps": [0, 3, 6], "arms": list(arms), "dropped": list(dropped), "values": values, "readings": rd}
    if ended:
        rep["ended"] = ended
    return json.loads(json.dumps(rep, default=float))  # as it is saved: seeds become strings


def only(fr: dict, name: str = "E-N") -> dict:
    return fr["contrasts"][name]


def same(seeds, pair):
    return {s: pair for s in seeds}


# ------------------------------------------------------------------------------------------- the frozen code's numbers

def test_constants_and_formulas_are_the_frozen_codes():
    assert F.H_THRESHOLD == U.THRESH["H"] and (F.P_READ, F.P_SUGGEST) == (U.P_READ, U.P_SUGGEST)
    assert F.N_SEEDS == len(U.SEEDS)
    for k, n in ((0, 304), (76, 152), (12, 304), (304, 304)):
        assert F.log_odds(k, n) == U.log_odds(k, n)
    for df, t in F.T975.items():
        assert round(U.t_crit(df), 3) == t
    assert sorted(F.T975) == list(range(1, 8))


def test_smallest_p_is_the_exact_tests():
    from anatta.stats import perm_p_paired
    for n in range(2, 9):
        assert math.isclose(perm_p_paired([1.0 + 0.1 * i for i in range(n)])["p"], F.smallest_p(n))
    # one seed against the rest: the next p the test can give is twice the smallest
    assert math.isclose(perm_p_paired([1.0, 1.1, 1.2, 1.3, 1.4, -0.1])["p"], 2 * F.smallest_p(6))


# ------------------------------------------------------------------------------------------- the base figures

def test_pooled_counts_levels_and_zero_seeds():
    p = F.pooled({0: run_values(76, 12), 1: run_values(60, 0), 2: run_values(90, 6)})
    assert (p["k0"], p["n0"], p["k_later"], p["n_later"]) == (226, 456, 18, 912)
    assert math.isclose(p["start"], 100 * 226 / 456) and math.isclose(p["later"], 100 * 18 / 912)
    assert math.isclose(p["retained"], (18 / 912) / (226 / 456))
    assert p["zero_later"] == [1] and p["seeds"] == 3 and p["floor"] is True


def test_floor_is_under_ten_percent_pooled():
    assert F.pooled({0: run_values(76, 30)})["floor"] is True  # 9.9%
    assert F.pooled({0: run_values(76, 31)})["floor"] is False  # 10.2%
    assert F.pooled({0: run_values(76, 0), 1: run_values(76, 61)})["floor"] is False  # 61 of 608 is 10.03%
    assert F.pooled({0: run_values(76, 0), 1: run_values(76, 60)})["floor"] is True  # 60 of 608 is 9.9%


def test_an_arm_that_never_released_has_no_share_retained():
    p = F.pooled({0: run_values(0, 0)})
    assert p["retained"] is None and p["start"] == 0.0 and p["floor"] is True


def test_mean_t_by_hand():
    m = F.mean_t({0: 1.0, 1: 2.0, 2: 3.0})
    assert m["n"] == 3 and math.isclose(m["mean"], 2.0)
    half = 4.303 * 1.0 / math.sqrt(3)
    assert math.isclose(m["ci95"][0], 2.0 - half) and math.isclose(m["ci95"][1], 2.0 + half)
    assert F.mean_t({0: 5.0}) == {"n": 1, "mean": 5.0, "ci95": None}
    assert F.mean_t({}) == {"n": 0, "mean": None, "ci95": None}


def test_mean_t_agrees_with_the_frozen_contrast():
    a = {s: run_values(70 + 3 * s, 10 + s) for s in range(8)}
    b = {s: run_values(76 - s, 12 + (s % 3)) for s in range(8)}
    c = U.contrast({s: v["drop_points"] for s, v in a.items()}, {s: v["drop_points"] for s, v in b.items()}, "U1")
    d, _ = F.seed_diffs(a, b, "drop_points")
    m = F.mean_t(d)
    assert math.isclose(m["mean"], c["mean"])
    assert math.isclose(m["ci95"][0], c["ci95"][0], abs_tol=2e-3) and math.isclose(m["ci95"][1], c["ci95"][1], abs_tol=2e-3)


def test_seed_diffs_leaves_out_undefined_and_unshared_seeds():
    a = {0: run_values(76, 12), 1: run_values(0, 0), 2: run_values(90, 6)}
    b = {0: run_values(60, 6), 1: run_values(70, 3), 3: run_values(80, 8)}
    d, left = F.seed_diffs(a, b, "retained")
    assert sorted(d) == [0] and left == [1]  # seed 1: arm A started at zero; seeds 2 and 3 are not shared
    d, left = F.seed_diffs(a, b, "drop_points")
    assert sorted(d) == [0, 1] and left == []


def test_the_split_sums_to_h_and_matches_the_notes_worked_cases():
    p = F.h_parts({0: run_values(99, 12)}, {0: run_values(76, 12)})  # a head start and the same floor
    assert math.isclose(p["later"]["mean"], 0.0, abs_tol=1e-12) and round(p["start"]["mean"], 2) == -0.62
    p = F.h_parts({0: run_values(76, 52)}, {0: run_values(76, 12)})  # the same start, 52 against 12 later
    assert round(p["later"]["mean"], 2) == 1.58 and math.isclose(p["start"]["mean"], 0.0, abs_tol=1e-12)
    p = F.h_parts({0: run_values(99, 52)}, {0: run_values(76, 12)})  # a head start and a large protection
    assert round(p["later"]["mean"], 2) == 1.58 and round(p["start"]["mean"], 2) == -0.62
    assert round(p["h"]["mean"], 2) == 0.96
    assert round(F.log_odds(0, 304), 1) == -6.4  # what a seed with no release counts as in the later reads' part
    a = {0: run_values(99, 12), 1: run_values(40, 0), 2: run_values(120, 77)}
    b = {0: run_values(76, 6), 1: run_values(90, 30), 2: run_values(60, 2)}
    p = F.h_parts(a, b)
    for s in a:
        assert math.isclose(p["later"]["by_seed"][s] + p["start"]["by_seed"][s], a[s]["H"] - b[s]["H"])
    assert math.isclose(p["later"]["mean"] + p["start"]["mean"], p["h"]["mean"])


# ------------------------------------------------------------------------------------------- the additions

def test_no_effect_at_the_floor_gets_the_floor_and_the_interval_and_nothing_else():
    seeds = range(8)
    e = {s: (76 + s % 3, 12 + s % 2) for s in seeds}
    n = {s: (76 - s % 2, 12 + s % 3) for s in seeds}
    c = only(F.floor_report(readout(e, n)))
    assert c["verdicts"]["H"] == "no difference shown" and sorted(c["additions"]) == ["i", "v"]
    assert c["additions"]["i"].startswith("Both arms ended under 10% (steps 3 and 6 pooled): E ")
    assert "at the design's 304 episodes, 0, 1 and 3 releases count as about -6.4, -5.3 and -4.5." in c["additions"]["i"]
    assert ("Nothing between step 0 and step 3 was read: a release that held for part of that stretch and was gone "
            "by step 3 reads as one that fell at once.") in c["additions"]["i"]
    assert "noisy" not in c["additions"]["i"]
    assert "differences in holding outside it are excluded" in c["additions"]["v"]
    assert c["h_agrees_with_readout"] is True and c["may_say_kept"] is False and c["stake"] is False


def test_a_head_start_and_the_same_floor_names_the_starts_part_and_never_a_later_level():
    seeds = range(8)
    c = only(F.floor_report(readout(same(seeds, (99, 12)), same(seeds, (76, 12)))))
    assert c["verdicts"] == {"U1": "higher", "H": "holds worse", "U2": "no difference shown",
                             "U3": "no difference shown"}
    assert c["registered"] == "starts better and wears down faster"
    assert sorted(c["additions"]) == ["i", "ii", "iii"]
    assert c["additions"]["iii"] == ("The starts' part alone is -0.62, at least as large in size as the 0.6 the rule "
                                     "asks of H's mean (by the mean over seeds of the log-odds at step 0, E started "
                                     "higher; in points the start reading U1 is \"higher\"); the later reads' part is "
                                     "+0.00.")
    assert "does not show it" in c["additions"]["ii"] and 'verdict "no difference shown"' in c["additions"]["ii"]
    assert math.isclose(c["h_parts"]["later"]["mean"], 0.0, abs_tol=1e-12)
    text = " ".join(c["additions"].values())
    for banned in BANNED:
        assert banned not in text


def test_the_mirror_off_the_floor_is_treated_the_same_way():
    seeds = range(8)
    # E from 65% and N from 50%, both ending at 15%: holds worse, off the floor
    worse = only(F.floor_report(readout(same(seeds, (99, 46)), same(seeds, (76, 46)))))
    assert worse["verdicts"]["H"] == "holds worse" and sorted(worse["additions"]) == ["ii", "iii"]
    # E from 50% and N from 65%, both ending at 15%: holds better, off the floor
    better = only(F.floor_report(readout(same(seeds, (76, 46)), same(seeds, (99, 46)))))
    assert better["verdicts"]["H"] == "holds better" and sorted(better["additions"]) == ["ii", "iii"]
    assert better["additions"]["iii"].startswith("The starts' part alone is +0.62, at least as large in size as the "
                                                 "0.6 the rule asks of H's mean (by the mean over seeds of the "
                                                 "log-odds at step 0, E started lower;")
    assert better["may_say_kept"] is False  # E's later level is not above N's
    assert better["stake"] is False  # U1 reads lower: the table's wording stands
    assert better["verdicts"]["U1"] == "lower" and better["registered"] == "starts worse; H is reported as read"


def test_a_real_negative_inside_the_floor_gets_no_sentence_about_the_start():
    seeds = range(8)
    c = only(F.floor_report(readout(same(seeds, (76, 3)), same(seeds, (76, 27)))))  # E 1.0% and N 8.9% later
    assert c["verdicts"]["H"] == "holds worse" and "iii" not in c["additions"]
    assert sorted(c["additions"]) == ["i", "ii"]
    assert math.isclose(c["h_parts"]["start"]["mean"], 0.0, abs_tol=1e-12)


def test_a_large_protection_without_the_level_reading_is_a_sighting_with_u2_as_printed():
    seeds = range(8)
    c = only(F.floor_report(readout(same(seeds, (76, 46)), same(seeds, (76, 12)))))  # E 15.1%, N 3.9%
    assert c["verdicts"]["H"] == "holds better" and c["verdicts"]["U2"] == "no difference shown"
    assert sorted(c["additions"]) == ["ii"] and c["stake"] is True and c["may_say_kept"] is True
    assert c["additions"]["ii"].startswith("The level reading (U2) does not show it: mean +11.18 points, p 0.0078")


def test_a_large_protection_with_the_level_reading_gets_no_addition():
    seeds = range(8)
    c = only(F.floor_report(readout(same(seeds, (76, 76)), same(seeds, (76, 12)))))  # E 25%, N 3.9%
    assert c["verdicts"]["H"] == "holds better" and c["verdicts"]["U2"] == "higher"
    assert c["additions"] == {} and c["stake"] is True and c["may_say_kept"] is True


def test_holds_better_with_both_arms_at_the_floor_may_not_say_kept():
    seeds = range(8)
    c = only(F.floor_report(readout(same(seeds, (76, 24)), same(seeds, (76, 12)))))  # E 7.9%, N 3.9%
    assert c["verdicts"]["H"] == "holds better" and sorted(c["additions"]) == ["i", "ii"]
    assert c["may_say_kept"] is False and c["stake"] is True


def test_suggestive_is_said_with_its_mean_and_p():
    # six seeds agree and two do not: past the threshold with p between 0.05 and 0.20
    e = {0: (76, 40), 1: (76, 40), 2: (76, 40), 3: (76, 40), 4: (76, 40), 5: (76, 40), 6: (76, 4), 7: (76, 4)}
    n = same(range(8), (76, 12))
    c = only(F.floor_report(readout(e, n)))
    assert c["verdicts"]["H"] == "suggestive"
    assert "iv" in c["additions"] and "v" in c["additions"] and "ii" not in c["additions"]
    assert c["additions"]["iv"].startswith("H is suggestive: mean +") and "It counts as no reading." in c["additions"]["iv"]
    assert c["stake"] is False


def test_a_failed_gate_gets_no_addition_about_h():
    seeds = range(8)
    c = only(F.floor_report(readout(same(seeds, (99, 12)), same(seeds, (20, 12)), gates_ok=False)))
    assert c["verdicts"]["H"] == "not read" and c["verdicts"]["U2"] == "not read"
    assert sorted(c["additions"]) == ["i"] and c["stake"] is False
    assert c["registered"].endswith("holding is not read (a validity gate failed)")


def test_kept_is_not_said_when_a_gate_failed_or_the_release_rose():
    seeds = range(8)
    # N from 20% to 3% (the practice gate would fail), E from 60% to 30%: the level under pressure is not read
    failed = only(F.floor_report(readout(same(seeds, (92, 92)), same(seeds, (30, 10)), gates_ok=False)))
    assert failed["verdicts"]["H"] == "not read" and failed["arms"]["E"]["floor"] is False
    assert failed["arms"]["E"]["later"] > failed["arms"]["N"]["later"] and failed["may_say_kept"] is False
    assert failed["additions"] == {}  # nothing is added about H when it is not read
    # the same runs with the gates passed: it may be said
    assert only(F.floor_report(readout(same(seeds, (92, 92)), same(seeds, (30, 10)))))["may_say_kept"] is True
    # E's release rose from 5.9% to 12.2%: "kept part" would be the wrong word
    rose = only(F.floor_report(readout(same(seeds, (9, 37)), same(seeds, (76, 12)))))
    assert rose["arms"]["E"]["later"] > rose["arms"]["E"]["start"] and rose["may_say_kept"] is False


def test_the_starts_part_is_named_whenever_it_is_large_with_the_later_part_beside_it():
    seeds = range(8)
    # the start hides a difference: E starts higher and ends higher, and H shows nothing
    hidden = only(F.floor_report(readout(same(seeds, (99, 23)), same(seeds, (76, 12)))))
    assert hidden["verdicts"]["H"] == "no difference shown" and abs(hidden["h_mean"]) < 0.1
    assert hidden["additions"]["iii"].startswith("The starts' part alone is -0.62")
    late = U.log_odds(23, NL) - U.log_odds(12, NL)  # by the frozen formula: about +0.67
    assert hidden["additions"]["iii"].endswith(f"the later reads' part is {late:+.2f}.") and 0.6 < late < 0.7
    # a head start and a later part that would give the reading on its own: both numbers are in the sentence
    both = only(F.floor_report(readout(same(seeds, (99, 1)), same(seeds, (76, 12)))))
    assert both["verdicts"]["H"] == "holds worse"
    assert "The starts' part alone is -0.62" in both["additions"]["iii"]
    late = U.log_odds(1, NL) - U.log_odds(12, NL)  # about -2.16: it would give the reading on its own
    assert both["additions"]["iii"].endswith(f"the later reads' part is {late:+.2f}.") and late < -2.0
    # a small starts' part is not named, whatever H reads
    small = only(F.floor_report(readout(same(seeds, (80, 52)), same(seeds, (76, 12)))))
    assert small["verdicts"]["H"] == "holds better" and "iii" not in small["additions"]
    # and nothing is said about H's parts when a gate failed
    gate = only(F.floor_report(readout(same(seeds, (99, 12)), same(seeds, (76, 12)), gates_ok=False)))
    assert "iii" not in gate["additions"]


def test_the_direction_of_the_start_is_said_of_the_mean_log_odds_not_of_the_pooled_level():
    # two E seeds whose practice did not take at all: E's pooled start is above N's, the mean of its log-odds below
    e = {s: ((0, 0) if s < 2 else (110, 12)) for s in range(8)}
    c = only(F.floor_report(readout(e, same(range(8), (76, 12)))))
    assert c["arms"]["E"]["start"] > c["arms"]["N"]["start"] and c["h_parts"]["start"]["mean"] > 0.6
    assert "by the mean over seeds of the log-odds at step 0, E started lower;" in c["additions"]["iii"]
    assert "E started lower in log-odds" not in c["additions"]["iii"]


def test_a_mean_past_the_threshold_with_few_seeds_is_at_most_suggestive():
    four = range(4)
    c = only(F.floor_report(readout({0: (76, 60), 1: (76, 60), 2: (76, 60), 3: (76, 6)}, same(four, (76, 12)))))
    assert abs(c["h_mean"]) > 0.6 and c["verdicts"]["H"] == "no difference shown"  # p is 0.25 here
    assert "a mean past the threshold is at most \"suggestive\"" in c["additions"]["vi"]


def test_an_interval_that_excludes_zero_is_said_not_to_be_a_reading():
    seeds = range(8)
    e = {s: (76, 17 + s % 2) for s in seeds}
    c = only(F.floor_report(readout(e, same(seeds, (76, 12)))))
    assert c["verdicts"]["H"] == "no difference shown" and 0 < c["h_mean"] < 0.6
    assert c["additions"]["v"].endswith("so an interval that excludes zero is not a reading.")
    wide = only(F.floor_report(readout({s: (76, 12 + (s % 3) * 4) for s in seeds}, {s: (76, 16 - (s % 2) * 8) for s in seeds})))
    assert wide["verdicts"]["H"] == "no difference shown" and "not a reading" not in wide["additions"]["v"]


def test_runs_of_unequal_size_are_pooled_by_their_counts():
    e = {0: (76, 12), 1: (50, 8, 100, 200), 2: (76, 12)}
    n = {0: (76, 12), 1: (76, 12), 2: (70, 3, 140, 280)}
    c = only(F.floor_report(readout(e, n, ended="the look said stop")))
    assert (c["arms"]["E"]["n0"], c["arms"]["E"]["n_later"]) == (404, 808)
    assert (c["arms"]["N"]["n0"], c["arms"]["N"]["n_later"]) == (444, 888)
    assert math.isclose(c["arms"]["E"]["later"], 100 * 32 / 808) and c["h_agrees_with_readout"] is True


def test_a_flag_on_an_arm_stays_beside_the_stake_line(capsys):
    seeds = range(8)
    guards = {"E": {"flag": "partly indiscriminate", "why": ["over-release 12.0 points above N's at step 0"]}}
    fr = F.floor_report(readout(same(seeds, (76, 76)), same(seeds, (76, 12)), guards=guards))
    assert only(fr)["stake"] is True and only(fr)["flags"] == ["E"]
    F.show(fr)
    assert "seeds repeat it; the readout's flag on arm E stands" in capsys.readouterr().out


def test_a_short_or_ended_run_says_what_the_exact_test_can_give():
    assert F.seeds_note(8, None) is None
    assert "smallest p the exact test can give is 0.25, so no reading and no \"suggestive\" is possible" in F.seeds_note(3, "x")
    assert F.seeds_note(3, "the look said stop").startswith("The run ended early. With 3 seeds")
    assert "0.125, so no reading is possible; a mean past the threshold is at most \"suggestive\"" in F.seeds_note(4, None)
    assert "0.0625, so no reading is possible" in F.seeds_note(5, None)
    assert "0.03125, so a reading needs every seed's difference to have the same sign" in F.seeds_note(6, None)
    assert "0.01562, so a reading is possible" in F.seeds_note(7, None)
    assert F.seeds_note(8, "the spend stop").startswith("The run ended early. With 8 seeds")
    three = range(3)
    c = only(F.floor_report(readout(same(three, (99, 12)), same(three, (76, 12)), ended="the look said stop")))
    assert "vi" in c["additions"] and c["verdicts"]["H"] == "no difference shown" and c["seeds"] == [0, 1, 2]


def test_a_dropped_seed_leaves_every_figure_and_is_counted_in_the_seeds_note():
    seeds = range(8)
    e, n = same(seeds, (76, 12)), same(seeds, (76, 12))
    e[7], n[7] = (76, 300), (76, 2)
    c = only(F.floor_report(readout(e, n, dropped=[7])))
    assert c["seeds"] == list(range(7)) and c["arms"]["E"]["seeds"] == 7 and c["arms"]["E"]["k_later"] == 84
    assert "vi" in c["additions"] and "With 7 seeds" in c["additions"]["vi"]
    assert c["drop_points"]["n"] == 7 and c["h_parts"]["h"]["n"] == 7 and c["h_agrees_with_readout"] is True


def test_three_arms_give_three_contrasts_each_on_its_own_arms():
    seeds = range(8)
    fr = F.floor_report(readout(same(seeds, (99, 46)), same(seeds, (76, 12)), r=same(seeds, (76, 46))))
    assert sorted(fr["contrasts"]) == ["E-N", "E-R", "R-N"]
    assert only(fr, "R-N")["arms"]["R"]["k_later"] == 46 * 8 and only(fr, "E-R")["arms"]["R"]["k0"] == 76 * 8
    assert only(fr, "E-N")["stake"] is True and only(fr, "R-N")["stake"] is False  # only E minus N is primary
    assert all(c["h_agrees_with_readout"] for c in fr["contrasts"].values())


def test_each_addition_is_printed_exactly_when_its_condition_holds_on_random_readouts():
    rng = random.Random(0)
    seen = set()
    for i in range(600):
        n_seeds = rng.choice((3, 4, 5, 6, 7, 8, 8, 8))
        gates_ok = i % 7 != 0
        base_e, base_n = rng.choice((30, 60, 76, 99, 120)), rng.choice((30, 60, 76, 99, 120))
        late_e, late_n = rng.choice((0, 3, 12, 27, 46, 90)), rng.choice((0, 3, 12, 27, 46, 90))
        e = {s: (max(0, min(N0, base_e + rng.randint(-12, 12))), max(0, min(NL, late_e + rng.randint(-6, 6))))
             for s in range(n_seeds)}
        n = {s: (max(0, min(N0, base_n + rng.randint(-12, 12))), max(0, min(NL, late_n + rng.randint(-6, 6))))
             for s in range(n_seeds)}
        rep = readout(e, n, ended="the look said stop" if n_seeds == 3 else None, gates_ok=gates_ok)
        c = only(F.floor_report(rep))
        add, v, pa, pb = c["additions"], c["verdicts"], c["arms"]["E"], c["arms"]["N"]
        seen.update(add)
        assert c["h_agrees_with_readout"] is True
        assert math.isclose(c["h_parts"]["start"]["mean"] + c["h_parts"]["later"]["mean"], c["h_mean"])
        assert ("i" in add) == (pa["later"] < 10 and pb["later"] < 10)
        assert ("ii" in add) == ((v["H"] == "holds better" and v["U2"] != "higher")
                                 or (v["H"] == "holds worse" and v["U2"] != "lower"))
        s, m, late = c["h_parts"]["start"]["mean"], c["h_mean"], c["h_parts"]["later"]["mean"]
        assert (v["H"] == "not read") == (not gates_ok)
        assert ("iii" in add) == (gates_ok and abs(s) >= 0.6 - 1e-9)
        if "iii" in add:  # the sentence gives both parts as they are, and says which way E started
            assert f"is {s:+.2f}," in add["iii"] and add["iii"].endswith(f"the later reads' part is {late:+.2f}.")
            assert ("E started lower;" in add["iii"]) == (s > 0) and f'U1 is "{v["U1"]}"' in add["iii"]
            assert "by the mean over seeds of the log-odds at step 0" in add["iii"]
        if "v" in add:  # "not a reading" exactly when the readout's interval excludes zero
            lo, hi = rep["readings"]["E-N"]["H"]["ci95"]
            assert ("not a reading" in add["v"]) == (lo > 0 or hi < 0)
        if "vi" in add and n_seeds in (4, 5):
            assert "at most" in add["vi"] and v["H"] not in ("holds better", "holds worse")
        assert ("iv" in add) == (v["H"] == "suggestive")
        assert ("v" in add) == (v["H"] in ("no difference shown", "suggestive"))
        assert ("vi" in add) == (n_seeds < 8)
        if c["may_say_kept"]:
            assert gates_ok and pa["later"] >= 10 and pa["later"] > pb["later"] and pa["later"] <= pa["start"]
        if c["stake"]:
            assert v["H"] == "holds better" and v["U1"] != "lower"
        if n_seeds < 6:
            assert v["H"] not in ("holds better", "holds worse")  # the exact test cannot read
        text = " ".join(add.values())
        for banned in BANNED:
            assert banned not in text
    assert seen == {"i", "ii", "iii", "iv", "v", "vi"}  # the draw exercised every addition


# ------------------------------------------------------------------------------------------- the printout

def test_show_prints_the_base_the_additions_and_no_ranking(capsys):
    seeds = range(8)
    F.show(F.floor_report(readout(same(seeds, (99, 12)), same(seeds, (76, 0)))))
    out = capsys.readouterr().out
    assert "the readout's wording: starts better and holds better" in out
    assert "seeds with no release later: [0, 1, 2, 3, 4, 5, 6, 7] (8 of 8)" in out
    assert "the split of H, E minus N: +2.64 in log-odds = the starts' part -0.62" in out
    assert "the later reads' part +3.26" in out and "not a cause; the split carries no verdict" in out
    assert "difference in the drop in points" in out and "difference in the share retained" in out
    assert "addition (i):" in out and "addition (ii):" in out and "addition (iii):" in out
    assert "the readout's note about the start: H is not neutral to the start" in out
    assert "plain summaries may say that E kept part of its release: no" in out
    assert "a sighting of the frame's stake" in out and "WARNING" not in out and "more from" not in out


def test_show_says_when_no_addition_applies_and_warns_on_a_mismatch(capsys):
    seeds = range(8)
    rep = readout(same(seeds, (76, 76)), same(seeds, (76, 12)))
    F.show(F.floor_report(rep))
    assert "additions: none applies" in capsys.readouterr().out
    rep["readings"]["E-N"]["H"]["mean"] = 0.123
    fr = F.floor_report(rep)
    assert only(fr)["h_agrees_with_readout"] is False
    F.show(fr)
    assert "WARNING" in capsys.readouterr().out


def test_main_reads_a_saved_file(tmp_path, capsys):
    f = tmp_path / "compare_u1_ended.json"
    three = range(3)
    f.write_text(json.dumps(readout(same(three, (99, 12)), same(three, (76, 12)), ended="the look said stop")))
    assert F.main([str(f)]) == 0
    out = capsys.readouterr().out
    assert "addition (vi): The run ended early. With 3 seeds" in out
    assert F.main([]) == 2
