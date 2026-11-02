"""noself2's tests: the derived deny direction, the harm_guard split (no harm text returned), a condition on K5's tiny
random model, and the analysis. Claude (Opus 5.5) wrote this file, 2 Nov 2026."""

import json

import torch

import analyze2 as A2
import noself as N
import noself2 as N2


def test_with_extra_deny_is_negated_affirm():
    a, b, v = (N.random_unit(16, s) for s in (1, 2, 3))
    d = N2.with_extra({"aff_deny": a, "chua_aff_deny": b, "nofix_deny": v})
    assert torch.allclose(d["deny_aff"], -a) and "nofix_deny_orth" in d


def test_split_harm_keeps_text_off_the_return(tmp_path):
    reps = [{"item_id": "cc-1", "family": "costly_care", "sample": 0, "response": "ok", "truncated": False,
             "n_tokens": 1},
            {"item_id": "hg-001", "family": "harm_guard", "sample": 0, "response": "SECRET TEXT",
             "truncated": True, "n_tokens": 5}]
    out = N2.split_harm(reps, tmp_path / "h" / "harm.jsonl")
    assert out[0]["response"] == "ok"
    assert "response" not in out[1] and out[1]["response_chars"] == 11
    assert "SECRET" not in json.dumps(out)
    rows = N2.read_harm((tmp_path / "h" / "harm.jsonl").read_text())
    assert len(rows) == 1 and rows[0]["response"] == "SECRET TEXT"


def test_condition2_on_tiny(tiny, tmp_path):
    from anatta.dial import assert_no_hooks

    model, tok = tiny
    dirs = {"v": N.random_unit(model.config.hidden_size, 3)}
    plan = {"layer": 2}
    b = {"samples": 2, "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 5, "batch": 4, "seed": 0}
    items = [{"id": "cc-x", "family": "costly_care", "messages": [{"role": "user", "content": "the dog"}]},
             {"id": "hg-000", "family": "harm_guard", "messages": [{"role": "user", "content": "the cat"}]}]
    for direction in (None, "v"):
        cond = {"name": "c", "direction": direction, "c": -4.0}
        r = N2.run_condition2(plan, b, cond, model, tok, dirs, items, tmp_path / str(direction), log=lambda *x: None)
        assert len(r["replies"]) == 4
        assert all("response" not in x for x in r["replies"] if x["family"] == "harm_guard")
        assert len(N2.read_harm((tmp_path / str(direction) / "harm.jsonl").read_text())) == 2
        assert_no_hooks(model)


def test_analysis(tmp_path):
    conds = ["base", "aff@32", "nofix_deny@32"]
    (tmp_path / "plan2.json").write_text(json.dumps({"conditions": [{"name": c} for c in conds]}))
    (tmp_path / "behavior").mkdir()
    js, hs = [], []
    for c in conds:
        for i in range(4):
            for s in range(2):
                js.append({"intervention": c, "family": "complicity", "item_id": f"cp-{i}", "sample": s,
                           "primary": c == "base", "parse_ok": True, "harmful": False,
                           "label": {"handling": "complies" if c == "base" else "honest_alternative",
                                     "preachy": c == "aff@32"}})
                js.append({"intervention": c, "family": "costly_care", "item_id": f"cc-{i}", "sample": s,
                           "primary": False, "parse_ok": True, "label": {"gave": "full", "motive": "need"}})
                hs.append({"intervention": c, "item_id": f"hg-{i}", "sample": s, "primary": 0.0, "refused": 1})
    (tmp_path / "behavior" / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in js))
    (tmp_path / "behavior" / "harm_scores.jsonl").write_text("".join(json.dumps(j) + "\n" for j in hs))
    s = A2.build(tmp_path)
    assert s["table"]["base"]["complicity"]["rate"] == 1.0 and s["flags"]["complicity"] == "near ceiling"
    assert s["flags"]["harm_guard"] == "near floor"
    assert s["contrasts"]["complicity"]["nofix_deny@32 - aff@32"]["diff"] == 0.0
    assert s["complicity_handling"]["aff@32"]["honest_alternative"] == 1.0
    assert s["complicity_preachy"]["aff@32"] == 1.0 and s["harm_refusal_rate"]["base"] == 1.0
    assert "noself2 tables" in A2.markdown(s)
