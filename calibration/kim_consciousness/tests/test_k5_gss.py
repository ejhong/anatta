"""The GSS leg's arithmetic and item list (no data file)."""

import math

import gss as G
import gss_items
import pytest


def test_item_list_matches_table_s9():
    rows = G.item_domains()
    assert len(rows) == 96 and len({r["var"] for r in rows}) == gss_items.UNIQUE
    counts = {}
    for r in rows:
        counts[r["domain"]] = counts.get(r["domain"], 0) + 1
    assert counts == gss_items.COUNTS
    assert {r["domain"] for r in rows if r["var"] == "relmarry"} == {"religion", "values"}
    assert next(r for r in rows if r["var"] == "postlife")["year_min"] == 2011
    assert set(gss_items.QUESTION_SHA256) == {f"{r['var']}|{r['domain']}" for r in rows}


def test_weighted_distribution_drops_missing():
    p, n = G.weighted_distribution([1, 2, 2, float("nan"), 9], [1.0, 1.0, 2.0, 5.0, 1.0], [1, 2])
    assert n == 3 and p == pytest.approx([0.25, 0.75])


def test_smooth_and_kl():
    s = G.smooth([1.0, 0.0], 10, 0.5)
    assert s == pytest.approx([10.5 / 11, 0.5 / 11])
    assert G.kl([0.5, 0.5], [0.5, 0.5]) == 0
    assert G.kl([0.9, 0.1], [0.5, 0.5]) == pytest.approx(0.9 * math.log(1.8) + 0.1 * math.log(0.2))


def test_delta_kl_sign():
    human = {"x|values": {"p": [0.8, 0.2], "n": 500}, "y|hope": {"p": [0.5, 0.5], "n": 300}}
    model = {"base": {"x|values": [0.01, 0.99], "y|hope": [0.99, 0.01]},
             "ablated": {"x|values": [0.5, 0.5], "y|hope": [0.9, 0.1]},
             "steered": {"x|values": [0.8, 0.2], "y|hope": [0.5, 0.5]}}
    d = G.delta_kl(human, model)
    assert d["pooled"]["steered"] > d["pooled"]["ablated"] > 0
    assert d["steer_over_ablate"] > 1
    assert set(d["by_domain"]) == {"values", "hope"}
