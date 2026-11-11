"""modal_act1.u1_sets() and press() run in this process on stand-ins: a stub `modal` and `peft`, a character-level
tokenizer, toy task pairs, a synthetic Stage A sample and told set, scripted episodes. No GPU, no Modal, no network,
no model-written code run (koan_env's task prompt is the real one; the env's step is never called). Claude (Fable
5.2, an AI model) wrote this file on 10 Nov 2026, before any run existed, and on 11 Nov 2026 moved it to revision 3
(press trains a count of optimizer steps and reads inside the rounds). Explore speed."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
import u1_core as U  # noqa: E402
import u1_fakes as F  # noqa: E402

C, D = U.C, U.D
CFG = dict(U.PRESS_CFG)
SNAP = F.SNAP


class TinyLM(torch.nn.Module):
    """A real, tiny language model for the loss read: an embedding and a linear layer."""

    def __init__(self, vocab=400):
        super().__init__()
        torch.manual_seed(0)
        self.emb, self.out = torch.nn.Embedding(vocab, 8), torch.nn.Linear(8, vocab)

    def forward(self, input_ids, attention_mask, position_ids):
        return SimpleNamespace(logits=self.out(self.emb(input_ids)))


@pytest.fixture
def app(monkeypatch, tmp_path):
    return F.install_act1(monkeypatch, tmp_path)


make_inputs, call_sets = F.make_inputs, F.call_sets


@pytest.fixture
def built(app, tmp_path):
    M, rec = app
    inp = make_inputs(tmp_path)
    res = call_sets(M, inp)
    return M, rec, inp, res, tmp_path / "act1" / "u1"


# ------------------------------------------------------------------------------------------------ u1_sets


def test_u1_sets_writes_the_document_sets_as_finetune_reads_them(built):
    M, rec, inp, res, base = built
    tok = F.FakeTok()
    for arm, src in U.DOC_SOURCES.items():
        data = torch.load(base / "sets" / f"{arm}docs" / "train.pt")
        info = json.loads((base / "sets" / f"{arm}docs" / "data.json").read_text())
        rows = inp["docs"][arm]
        assert len(data["records"]) == len(data["turns"]) == 6 == info["kept"] == info["documents"]
        assert [r["id"] for r in data["records"]] == [r["id"] for r in rows]
        assert all(r["outcome"] == "document" and r["kind"] == src["kind"] for r in data["records"])
        items = [(p, c) for t in data["turns"] for p, c in t]  # as modal_sft.finetune flattens a set
        assert all(p == [1] and c[-1] == 2 for p, c in items)  # the BOS alone; the document, then the EOS
        assert [tok.decode(c[:-1]) for _, c in items] == [r["text"] for r in rows]
        assert info["cut_finetune"] is False and info["cut_sampling"] is False  # finetune refuses anything else
        assert info["doc_ids_fingerprint"] == U.docs_fingerprint(rows) == inp["expect"]["docs"][arm]
        assert info["tokens"] == sum(len(c) for _, c in items) and info["doc_kind"] == src["kind"]
        assert sum(info["doc_referents"].values()) == 6 and info["bos_id"] == 1 and info["eos_id"] == 2
        held = torch.load(base / "held" / f"{arm}.pt")
        assert held["ids"] == [r["id"] for r in inp["held"][arm]] and len(held["items"]) == 2 and held["kind"] == src["kind"]
    assert res["info"]["held"]["E"]["docs"] == 2 and set(res["info"]["sets"]) == {"Ndocs", "Rdocs", "Edocs", "practice"}


def test_u1_sets_keeps_the_practice_records_and_turns_in_step(built):
    M, rec, inp, res, base = built
    told, ev = inp["_told"], set(inp["eval_names"])
    data = torch.load(base / "sets" / "practice" / "train.pt")
    info = json.loads((base / "sets" / "practice" / "data.json").read_text())
    keep = [i for i, r in enumerate(told["records"]) if U.record_pair(r) not in ev]
    assert data["records"] == [told["records"][i] for i in keep] and data["turns"] == [told["turns"][i] for i in keep]
    assert len(keep) < len(told["records"]) and not any(U.record_pair(r) in ev for r in data["records"])
    assert info["kept"] == len(keep) == 12 and info["releases"] == 4 == inp["expect"]["practice_releases"]
    assert info["solves"] == 8 and info["dropped_on_eval_pairs"] == 6 and info["cut_finetune"] is False
    assert info["items"] == sum(len(t) for t in data["turns"]) and info["source_counts"] == F.told_counts(told)
    # p00 and p02 carry a told release, p01 and p03 do not; p02's impossible twin is not a one-off
    assert res["info"]["unpractised"] == ["p01", "p03"] == inp["expect"]["unpractised"]


def test_u1_sets_writes_the_streams_from_the_volumes_own_records(built):
    M, rec, inp, res, base = built
    sa, ev = inp["_sa"], set(inp["eval_names"])
    for name in U.STREAMS:
        draws = U.stream_draws(sa["records"], ev, name)
        for s in U.SEEDS:
            st = torch.load(base / "streams" / f"{name}_s{s}.pt")
            assert st["stream"] == name and st["seed"] == s and sorted(st["rounds"]) == list(range(1, 9))
            for r, rd in st["rounds"].items():
                assert rd["idx"] == draws[s][r] and len(rd["items"]) == 192
                assert rd["items"] == [sa["turns"][i][-1] for i in rd["idx"]]  # the final turn of each drawn episode
                # each item's outcome sits beside it, in the items' order (a cut round counts what it trained on)
                assert rd["outcomes"] == [sa["records"][i]["outcome"] for i in rd["idx"]]
                assert U.trained_counts(rd["outcomes"])["by_outcome"] == rd["summary"]["by_outcome"]
                assert rd["summary"] == U.draw_summary(sa["records"], rd["idx"])
                assert not any(sa["records"][i]["pair"] in ev for i in rd["idx"])
                assert "released" not in rd["summary"]["by_outcome"]
        assert res["info"]["streams"][name]["draws_fingerprint"] == inp["expect"]["stage_a"]["draws"][name]
        assert res["info"]["streams"][name]["rounds"]["s3_r5"] == U.draw_summary(sa["records"], draws[3][5])
    honest = torch.load(base / "streams" / "honest_s0.pt")
    assert set(honest["rounds"][1]["summary"]["by_outcome"]) == {"solved"}


def test_u1_sets_writes_the_old_read_paths_pairs_and_the_checks(built):
    M, rec, inp, res, base = built
    rows = [json.loads(x) for x in (base / "pairs_oldpath.jsonl").read_text().splitlines()]
    assert sorted(r["name"] for r in rows) == sorted(inp["eval_oneoff"]) == ["p00", "p01", "p03"]
    assert all(r["split"] == "held" and r["gate0_split"] == "train" and "u1_note" in r for r in rows)
    import koan_env
    import modal_common

    held = modal_common.pairs_by_name((base / "pairs_oldpath.jsonl").read_text(), "held")  # as heldout_eval reads it
    assert sorted(held) == ["p00", "p01", "p03"] and isinstance(held["p00"], koan_env.Pair)
    info = json.loads((base / "info.json").read_text())
    ch = info["checks"]
    assert ch["snapshot"] == {"pinned": SNAP, "snapshots": [SNAP], "refs_main": SNAP}
    assert ch["stage_a"]["n"] == len(inp["_sa"]["records"]) == 120 and ch["stage_a"]["turns_match_n_turns"]
    assert ch["first_turn_prompts_stage_a"] == {"checked": 120, "identical": True}
    assert ch["first_turn_prompts_practice"] == {"checked": 12, "identical": True}
    assert ch["first_turn_prompts_distinct"] == 24 and ch["no_eval_pair_trained_on"] is True
    assert info["read_pairs"] == 3 and info["eval_pairs"] == 4 and info["old_path_pairs"].endswith("pairs_oldpath.jsonl")
    assert info == res["info"]


def test_u1_sets_never_overwrites(built):
    M, rec, inp, res, base = built
    before = sorted(str(p) for p in base.rglob("*"))
    with pytest.raises(RuntimeError, match="already holds sets"):
        call_sets(M, inp)
    assert sorted(str(p) for p in base.rglob("*")) == before
    again = call_sets(M, inp, root="u1b")  # another root is a fresh build
    assert again["info"]["root"] == "u1b" and (base.parent / "u1b" / "info.json").exists()


def tampered(inp, tmp_path, which, change):
    """The stand-in volume file `which` ("samples" or "told"), changed in place by `change`."""
    data = inp["_sa"] if which == "samples" else inp["_told"]
    change(data)
    torch.save(data, tmp_path / f"{which}.pt")


@pytest.mark.parametrize("case, match", [
    ("turn_lists", "do not line up"), ("n_turns", "do not line up"), ("outcome", "not the laptop's"),
    ("order", "not the laptop's"), ("prompt", "do not re-render"), ("practice_prompt", "do not re-render"),
    ("told_counts", "told set on the volume"), ("two_snapshots", "pinned snapshot"), ("refs_main", "pinned snapshot"),
    ("no_refs_main", "named by refs/main"),
    ("doc_fingerprint", "not the ones the laptop named"), ("doc_roundtrip", "fail the item check"),
    ("doc_kind", "another kind"), ("doc_overlap", "appears twice"), ("draws", "draws differ"),
    ("eval_in_stream", "eval-split pair"), ("eval_in_practice", "eval-split pair"), ("release_in_stream", "eval-split pair"),
    ("unpractised", "unpractised"), ("practice_releases", "releases"), ("missing_pair_row", "have no row"),
    ("read_pair_not_eval", "must be eval-split pairs"),
])
def test_each_assertion_stops_the_build_before_anything_is_written(app, tmp_path, monkeypatch, case, match):
    M, rec = app
    inp = make_inputs(tmp_path)
    over = {}
    if case == "turn_lists":
        tampered(inp, tmp_path, "samples", lambda d: d["turns"].pop())
    elif case == "n_turns":
        tampered(inp, tmp_path, "samples", lambda d: d["turns"][5].append(([1], [2])))
    elif case == "outcome":  # one outcome differs from the laptop's file
        tampered(inp, tmp_path, "samples", lambda d: d["records"][0].update(outcome="failed"))
    elif case == "order":  # the same counts in another order: only the sequence check sees it
        def swap(d):
            i = next(k for k, r in enumerate(d["records"]) if r["outcome"] == "solved" and r["n_turns"] == 1)
            j = next(k for k, r in enumerate(d["records"]) if r["outcome"] == "near_miss" and r["n_turns"] == 1
                     and r["pair"] == d["records"][i]["pair"])
            d["records"][i]["outcome"], d["records"][j]["outcome"] = "near_miss", "solved"
        tampered(inp, tmp_path, "samples", swap)
    elif case == "prompt":  # a stored first-turn prompt that today's template would not produce
        tampered(inp, tmp_path, "samples", lambda d: d["turns"][7][0][0].insert(3, 99))
    elif case == "practice_prompt":
        tampered(inp, tmp_path, "told", lambda d: d["turns"][-1][0][0].append(99))
    elif case == "told_counts":
        over["expect"] = {**inp["expect"], "told": {**inp["expect"]["told"], "tokens": 1}}
    elif case == "two_snapshots":
        (Path(C.MODELS["ministral8b"]["path"]).parent / "another").mkdir()
    elif case == "refs_main":  # refs/main names another snapshot than the pinned one
        (Path(C.MODELS["ministral8b"]["path"]).parents[1] / "refs" / "main").write_text("another\n")
    elif case == "no_refs_main":  # nothing says what the hub id resolves to: a failure, not a pass
        (Path(C.MODELS["ministral8b"]["path"]).parents[1] / "refs" / "main").unlink()
    elif case == "doc_fingerprint":
        over["docs"] = {**inp["docs"], "R": list(reversed(inp["docs"]["R"]))}
    elif case == "doc_roundtrip":
        monkeypatch.setattr(F.FakeTok, "lossy", "E")
        over["docs"] = {**inp["docs"], "E": [dict(r, text=r["text"] + " E") for r in inp["docs"]["E"]]}
    elif case == "doc_kind":
        over["held"] = {**inp["held"], "N": inp["held"]["R"]}
    elif case == "doc_overlap":
        over["held"] = {**inp["held"], "E": inp["docs"]["E"][:2]}
    elif case == "draws":
        over["expect"] = {**inp["expect"], "stage_a": {**inp["expect"]["stage_a"], "draws": {"gameable": "x", "honest": "y"}}}
    elif case in ("eval_in_stream", "release_in_stream"):
        want = (lambda r: r["pair"] == "p00" and r["outcome"] == "hacked") if case == "eval_in_stream" else \
            (lambda r: r["outcome"] == "released" and r["pair"] not in inp["eval_names"])
        bad = next(i for i, r in enumerate(inp["_sa"]["records"]) if want(r))
        real = U.stream_draws
        monkeypatch.setattr(U, "stream_draws", lambda *a, **k: {
            s: {r: [bad] + idx[1:] for r, idx in by.items()} for s, by in real(*a, **k).items()})
        monkeypatch.setattr(U, "draws_fingerprint", lambda draws: "same")
        over["expect"] = {**inp["expect"], "stage_a": {**inp["expect"]["stage_a"], "draws": {"gameable": "same", "honest": "same"}}}
    elif case == "eval_in_practice":
        monkeypatch.setattr(U, "practice_filter", lambda records, ev: list(range(len(records))))
        over["expect"] = {**inp["expect"], "practice_releases": None}
    elif case == "unpractised":
        over["expect"] = {**inp["expect"], "unpractised": ["p01"]}
    elif case == "practice_releases":
        over["expect"] = {**inp["expect"], "practice_releases": 5}
    elif case == "missing_pair_row":
        over["pairs_rows"] = inp["pairs_rows"][:-1]
    elif case == "read_pair_not_eval":
        over["eval_oneoff"] = ["p00", "p09"]
    with pytest.raises(RuntimeError, match=match):
        call_sets(M, inp, **over)
    base = tmp_path / "act1" / "u1"
    assert not base.exists() or not any(base.rglob("*"))  # nothing was written


def test_the_document_check_and_the_loss_read_are_the_pilots(app, monkeypatch):
    """u1_doc_items accepts and rejects what modal_declared.check_doc_items does, and u1_doc_nll gives
    modal_declared.doc_nll's numbers, on the same stand-ins."""
    M, rec = app
    for name in ("modal_declared", "modal_sft", "modal_retreat2", "modal_common"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import modal_declared as MD

    tok = F.FakeTok()
    rows = F.doc_rows("explained", "train", 7)
    items = M.u1_doc_items(tok, rows)
    ids = tok([r["text"] for r in rows])["input_ids"]
    assert items == [D.doc_item(1, 2, x) for x in ids]
    assert MD.check_doc_items(tok, items, [r["text"] for r in rows]) == 7
    monkeypatch.setattr(F.FakeTok, "lossy", "7")  # "number 7" no longer decodes to itself... only row 6 has no 7
    bad_rows = [dict(r, text=r["text"] + " 7") for r in rows[:2]] + rows[2:]
    with pytest.raises(RuntimeError, match="2 document items fail") as e:
        M.u1_doc_items(tok, bad_rows)
    assert "explained_train:000000" in str(e.value) and "explained_train:000001" in str(e.value)
    with pytest.raises(SystemExit):
        MD.check_doc_items(tok, [D.doc_item(1, 2, x) for x in tok([r["text"] for r in bad_rows])["input_ids"]],
                           [r["text"] for r in bad_rows])
    monkeypatch.setattr(F.FakeTok, "lossy", "")
    monkeypatch.setattr(F.FakeTok, "bos_token", "<bos>")
    with pytest.raises(RuntimeError, match="BOS/EOS"):
        M.u1_doc_items(tok, rows)
    monkeypatch.setattr(F.FakeTok, "bos_token", "<s>")
    model = TinyLM()
    got = M._real_u1_doc_nll(model, tok, items, micro=3)
    per = MD.doc_nll(model, tok, items, micro=3)
    total, count = sum(s for s, _ in per), sum(n for _, n in per)
    assert got["docs"] == 7 and got["tokens"] == count == sum(len(c) for _, c in items)
    assert got["nll_per_token"] == pytest.approx(total / count) and got["nll_per_token"] > 0
    assert got["mean_doc_nll"] == pytest.approx(sum(s / n for s, n in per) / 7)
    assert M._real_u1_doc_nll(model, tok, items, micro=8)["nll_per_token"] == pytest.approx(got["nll_per_token"], rel=1e-5)


def test_the_snapshot_check_and_the_tokenizers_path(app, monkeypatch):
    M, rec = app
    path = Path(C.MODELS["ministral8b"]["path"])
    assert M.u1_snapshot_check("ministral8b") == {"pinned": SNAP, "snapshots": [SNAP], "refs_main": SNAP}
    ref = path.parents[1] / "refs" / "main"
    ref.unlink()  # a missing refs/main is a failure: nothing would say what the hub id resolves to
    with pytest.raises(RuntimeError, match="named by refs/main.*'refs_main': None"):
        M.u1_snapshot_check("ministral8b")
    ref.write_text("another\n")
    with pytest.raises(RuntimeError, match="pinned snapshot"):
        M.u1_snapshot_check("ministral8b")
    ref.write_text(SNAP + "\n")
    assert M.u1_snapshot_check("ministral8b")["refs_main"] == SNAP
    monkeypatch.setitem(C.MODELS, "ministral8b", {**C.MODELS["ministral8b"], "path": str(path.parent / "gone")})
    with pytest.raises(RuntimeError, match="pinned snapshot"):
        M.u1_snapshot_check("ministral8b")


# ------------------------------------------------------------------------------------------------ press


def run_press(M, inp, label="E_s3", tag="main", seed=3, stream="gameable", steps=24, reads=(6, 24), init="/vol/a/adapter",
              **over):
    return M.press(init, label, tag, seed, stream, steps, list(reads), over.pop("cfg", CFG), inp["pairs_rows"],
                   inp["eval_oneoff"], **over)


def events(*parts) -> list:
    """A run's events in order: "read", "nll" and "train" as given, and a number for that many optimizer steps."""
    return [e for p in parts for e in (["step"] * p if isinstance(p, int) else [p])]


def test_press_reads_at_step_0_before_any_training_then_trains_on_the_seeds_stream(built):
    M, rec, inp, _, base = built
    res = run_press(M, inp)
    # the order: the read at step 0, the loss on the three held-out sets, then two rounds of 12 steps, with a read
    # inside the first round (after its step 6) and a read after the last step of the second (step 24)
    assert rec.events == events("read", "nll", "nll", "nll", "train", 6, "read", 6, "train", 12, "read")
    assert rec.loaded == [("/vol/a/adapter", True)] and res["info"]["init_lora_b_abs"] == 2.5
    assert res["info"]["init_adapter"] == "/vol/a/adapter" and rec.model.ckpt == 1
    stream = torch.load(base / "streams" / "gameable_s3.pt")["rounds"]
    assert [t["items"] for t in rec.trained] == [stream[r]["items"] for r in (1, 2)]  # the seed's own rounds, whole
    assert all(t["cfg"] == CFG and t["training"] and len(t["items"]) == 192 for t in rec.trained)
    assert rec.players == [{"max_new": (512, 384), "gen_batch": 32}]
    # a read's seed is the base plus its step; no read is made in train mode or with gradients still held
    assert [r["seed"] for r in rec.reads] == [10_000, 10_006, 10_024]
    assert not any(r["training"] or r["grads"] for r in rec.reads)
    assert all(t["after_hook"] == [True] * 12 for t in rec.trained)  # the model is handed back in train mode
    out = base / "main" / "E_s3"
    assert sorted(p.name for p in out.iterdir()) == ["info.json", "log.jsonl", "read_s0.jsonl", "read_s24.jsonl",
                                                     "read_s6.jsonl"]
    assert rec.saved == [str(out / "adapter")]
    log = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert log == json.loads(json.dumps(res["log"]))
    assert [(x["kind"], x["round"], x["step"]) for x in log] == [
        ("read", 0, 0), ("read", 1, 6), ("train", 1, 12), ("read", 2, 24), ("train", 2, 24)]
    assert set(log[0]["held_nll"]) == {"N", "R", "E"} and log[0]["held_nll"]["E"]["nll_per_token"] == 1.25
    # a read's log record says how many episodes it held and how long it took, and nothing of their outcomes
    assert [("episodes" in x, "s_read" in x) for x in log] == [(x["kind"] == "read",) * 2 for x in log]
    assert log[0]["episodes"] == 18 and not any(k in x for x in log for k in ("read", "imp", "sol", "released"))
    assert all(set(x) == {"kind", "round", "step", "episodes", "s_read", "peak_gb", "oom_splits"} for x in (log[1], log[3]))
    assert set(res) == {"label", "tag", "seed", "stream", "info", "prep_dropped", "reads", "held_nll", "log", "seconds"}
    assert not any(k in json.dumps(res["info"]) for k in ('"imp"', '"released"', '"over_released"'))
    # a round's record: the steps it took, what it trained on by the stored outcomes, and the round's whole draw
    one = log[2]
    assert (one["steps"], one["cut"], one["nll_first"], one["nll_last"]) == (12, False, 2.0, 1.45)
    assert one["draw"] == stream[1]["summary"] and one["items"] == U.trained_counts(stream[1]["outcomes"])
    assert one["items"]["n"] == 192 and one["items"]["hacked"] == stream[1]["summary"]["hacked"] > 0
    info = json.loads((out / "info.json").read_text())
    assert info == res["info"] and info["read_steps"] == [0, 6, 24] and info["steps"] == 24 and info["rounds"] == 2
    assert info["stream"] == "gameable" and "read_rounds" not in info
    assert info["read"] == U.READ and info["cfg"] == CFG and info["eval_pairs"] == 3 and info["held_nll"] == res["held_nll"]
    assert info["snapshot"] == {"pinned": SNAP, "snapshots": [SNAP], "refs_main": SNAP}
    assert sorted(res["reads"]) == [0, 6, 24] and res["seconds"] >= 0 and res["prep_dropped"] == []
    assert U.plan_problems(info, U.run_plan(3, "gameable", 24, [6, 24], "/vol/a/adapter", inp["eval_oneoff"])) == []


def test_a_round_cut_short_trains_the_start_of_the_round(built):
    """A run of 18 steps is round 1 whole and the first 6 steps of round 2: that round's first 96 items, in order.
    Its log record counts what those items were, by the outcome stored beside each."""
    M, rec, inp, _, base = built
    res = run_press(M, inp, label="cut", steps=18, reads=(3, 18))
    stream = torch.load(base / "streams" / "gameable_s3.pt")["rounds"]
    assert rec.events == events("read", "nll", "nll", "nll", "train", 3, "read", 9, "train", 6, "read")
    assert [t["items"] for t in rec.trained] == [stream[1]["items"], stream[2]["items"][:96]]
    cut = res["log"][-1]
    assert (cut["kind"], cut["round"], cut["step"], cut["steps"], cut["cut"]) == ("train", 2, 18, 6, True)
    assert cut["items"] == U.trained_counts(stream[2]["outcomes"][:96]) and cut["items"]["n"] == 96
    assert cut["items"]["hacked"] == stream[2]["outcomes"][:96].count("hacked")
    assert cut["draw"] == stream[2]["summary"] and cut["draw"]["n"] == 192  # the round as drawn, beside what it took
    assert (cut["nll_first"], cut["nll_last"]) == (2.0, 1.75)
    assert res["info"]["steps"] == 18 and res["info"]["rounds"] == 2 and res["info"]["read_steps"] == [0, 3, 18]
    assert [r["seed"] for r in rec.reads] == [10_000, 10_003, 10_018]
    # the rehearsal's shape: six steps, a read inside the round (step 3) and a read at the end of training (step 6)
    for kept in (rec.events, rec.trained, rec.reads):
        kept.clear()
    six = run_press(M, inp, label="six", steps=6, reads=(3, 6))
    assert rec.events == events("read", "nll", "nll", "nll", "train", 3, "read", 3, "read")
    assert [t["items"] for t in rec.trained] == [stream[1]["items"][:96]]
    assert [r["seed"] for r in rec.reads] == [10_000, 10_003, 10_006] and sorted(six["reads"]) == [0, 3, 6]
    assert not any(r["training"] or r["grads"] for r in rec.reads)
    assert sorted(p.name for p in (base / "main" / "six").iterdir()) == [
        "info.json", "log.jsonl", "read_s0.jsonl", "read_s3.jsonl", "read_s6.jsonl"]
    assert [(x["kind"], x["step"]) for x in six["log"]] == [("read", 0), ("read", 3), ("read", 6), ("train", 6)]


def test_a_read_inside_a_round_leaves_the_training_as_it_was(built, monkeypatch):
    """The round's items and torch's random state after every step are the same with reads inside the round as with
    none: the read reseeds torch and samples, and press puts the state back. The stand-in optimizer raises if it is
    touched at all. With the putting back taken away, the same test sees the difference."""
    M, rec, inp, _, base = built
    run_press(M, inp, label="plain", steps=12, reads=())
    plain = rec.trained[-1]
    run_press(M, inp, label="read_inside", steps=12, reads=(3, 6, 12))
    inside = rec.trained[-1]
    assert len(plain["draws"]) == 12 and len(set(plain["draws"])) == 12
    assert inside["items"] == plain["items"] and inside["draws"] == plain["draws"]
    assert [r["seed"] for r in rec.reads[-4:]] == [10_000, 10_003, 10_006, 10_012]
    monkeypatch.setattr(torch, "set_rng_state", lambda state: None)
    run_press(M, inp, label="state_not_put_back", steps=12, reads=(3, 6, 12))
    moved = rec.trained[-1]  # the stand-in draws after each step's hook: the read at step 3 moves the third draw on
    assert moved["draws"][:2] == plain["draws"][:2] and moved["draws"][2] != plain["draws"][2]
    assert all(a != b for a, b in zip(moved["draws"][2:], plain["draws"][2:]))


class TinyTrainable(TinyLM):
    """TinyLM with dropout, so that training draws from torch's random state, and the two things press asks of a
    base model. Built the same every time (TinyLM seeds torch first)."""

    def __init__(self):
        super().__init__(vocab=2048)
        self.drop = torch.nn.Dropout(0.2)
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=1)

    def forward(self, input_ids, attention_mask, position_ids):
        return SimpleNamespace(logits=self.out(self.drop(self.emb(input_ids))))

    def gradient_checkpointing_enable(self, **k):
        pass


def test_with_the_real_training_loop_a_read_inside_a_round_changes_no_parameter(built, monkeypatch):
    """press's own hook around the real sft_steps, on a tiny real model with dropout. The parameters after a round
    with reads inside it are bit for bit those after the same round with none; a round cut short ends exactly where
    the whole round stood after as many steps; and without the random state put back, they differ."""
    M, rec, inp, _, base = built
    from anatta.train import expert_iteration as EI

    monkeypatch.setattr(EI, "sft_steps", M._real_sft_steps)
    seen, play = [], M.play

    def watched(player, envs, temp):  # what the model holds at each read, and the mode it is read in
        seen.append(({k: v.detach().clone() for k, v in rec.model.state_dict().items()}, rec.model.training))
        return play(player, envs, temp)

    monkeypatch.setattr(M, "play", watched)

    def run(label, steps, reads):
        rec.model = TinyTrainable()  # the same start for every run
        seen.clear()
        res = run_press(M, inp, label=label, steps=steps, reads=reads)
        return {k: v.detach().clone() for k, v in rec.model.state_dict().items()}, list(seen), res

    def same(a, b):
        return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

    start = {k: v.detach().clone() for k, v in TinyTrainable().state_dict().items()}
    plain, at_plain, res = run("plain", 12, ())
    assert not same(plain, start) and same(at_plain[0][0], start)  # it trained; the read at step 0 is of the start
    assert res["log"][-1]["steps"] == 12 and res["log"][-1]["nll_first"] > 0  # the real loop's own loss, not a stand-in's
    inside, at, _ = run("inside", 12, (3, 6, 12))
    assert same(inside, plain)  # three reads inside the round: not one parameter moved by them
    assert [training for _, training in at] == [False] * 4 and same(at[3][0], plain)  # read in eval mode; step 12 is the end
    assert not same(at[1][0], at[2][0]) and not same(at[2][0], plain)  # the reads at steps 3 and 6 saw the round under way
    cut, _, res = run("cut", 6, (3,))
    assert same(cut, at[2][0])  # six steps of a round cut short: where the whole round stood after its sixth step
    assert res["log"][-1]["cut"] is True and res["log"][-1]["steps"] == 6
    monkeypatch.setattr(torch, "set_rng_state", lambda state: None)
    moved, _, _ = run("state_not_put_back", 12, (3, 6, 12))
    assert not same(moved, plain)  # dropout drew from the state the read left: the restore is what keeps training the same


def test_the_step_0_read_does_not_depend_on_what_the_run_does_next(built):
    """The read at step 0 is the old round-0 read: seed 10,000, made before the loss read and before any training,
    in eval mode, and the same bytes whatever steps and read steps the run was asked for."""
    M, rec, inp, _, base = built
    seen = []
    for label, steps, reads in (("only", 0, ()), ("dose", 12, (3, 6, 12)), ("main", 24, (12, 24))):
        for kept in (rec.events, rec.reads):
            kept.clear()
        run_press(M, inp, label=label, steps=steps, reads=reads)
        assert rec.events[:4] == ["read", "nll", "nll", "nll"]
        assert rec.reads[0]["seed"] == 10_000 == U.READ["seed_base"] and not rec.reads[0]["training"]
        seen.append((base / "main" / label / "read_s0.jsonl").read_bytes())
    assert seen[0] and seen[0] == seen[1] == seen[2]
    assert not list(base.rglob("read_r*.jsonl"))  # the old name is gone: a read's file is named by its step


def test_the_read_offers_the_release_and_samples_each_twin_as_the_design_says(built):
    M, rec, inp, _, base = built
    from anatta.koans.base import RELEASE_HINT

    res = run_press(M, inp, label="N_s0", seed=0)
    for r in rec.reads:
        envs = r["envs"]
        assert len(envs) == 3 * (4 + 2)
        assert all(RELEASE_HINT in e.system and e.system == D.KOAN_SYSTEM and e.arm == "koan" for e in envs)
        assert all(e.attempts == 3 for e in envs) and "Note on grading" not in envs[0].system  # no Act I hint
        by = {}
        for e in envs:
            by[(e.pair.name, e.impossible)] = by.get((e.pair.name, e.impossible), 0) + 1
        assert by == {(p, imp): n for p in inp["eval_oneoff"] for imp, n in ((True, 4), (False, 2))}
    recs = [json.loads(x) for x in (base / "main" / "N_s0" / "read_s0.jsonl").read_text().splitlines()]
    assert recs == res["reads"][0] and len(recs) == 18
    released = next(x for x in recs if x["outcome"] == "released")
    failed = next(x for x in recs if x["outcome"] == "failed")
    assert (released["form"], released["grounded"]) == ("strict", True) and "reason" in released
    assert "form" not in failed and "reason" not in failed  # kept only when present
    assert set(M.slim(failed)) <= set(failed) and "first_fail" not in failed and failed["replies"]
    m0 = U.measures(recs)  # the readout's own count, from the raw read: the run itself reports none
    assert m0["imp"]["n"] == 12 and m0["sol"] == {"n": 6, "solved": 6, "over_released": 0}
    assert U.read_problems(recs, inp["eval_oneoff"], 4, 2) == []  # the planned pairs, the design's samples of each twin


def test_press_stops_before_the_model_is_loaded_when_a_read_pair_is_dropped(built, monkeypatch):
    """prepare_pairs drops a pair whose reference will not run. A dropped read pair would shrink every read of the
    run, so press refuses, saying how many, before the model is loaded."""
    M, rec, inp, _, base = built
    order = []
    monkeypatch.setattr(M.mc, "prepare_pairs", lambda pairs: (
        order.append("prepare") or {k: v for k, v in pairs.items() if k != "p01"}, {"dropped": ["p01"]}))
    real = M.load_base
    monkeypatch.setattr(M, "load_base", lambda key: order.append("load") or real(key))
    with pytest.raises(RuntimeError, match="1 of the 3 read pairs are missing") as e:
        run_press(M, inp, label="short_read")
    assert order == ["prepare"] and not rec.loaded and not rec.events  # refused before the load, the read, the training
    assert "p01" not in str(e.value) and not (base / "main" / "short_read").exists()  # a count, no pair's name or text
    monkeypatch.setattr(M.mc, "prepare_pairs", lambda pairs: (order.append("prepare") or pairs, {"dropped": []}))
    run_press(M, inp, label="whole")
    assert order == ["prepare", "prepare", "load"]  # the pairs are prepared first, then the model is loaded
    with pytest.raises(RuntimeError, match="0 of the 0 read pairs"):
        M.press("/vol/a/adapter", "none", "main", 3, "gameable", 24, [12, 24], CFG, inp["pairs_rows"], [])


def test_press_refuses_a_rerun_a_zero_adapter_and_runs_the_streams_cannot_serve(built, monkeypatch):
    M, rec, inp, _, base = built
    run_press(M, inp)
    n = len(rec.events)
    with pytest.raises(RuntimeError, match="already holds a run"):
        run_press(M, inp)
    assert len(rec.events) == n and len(rec.loaded) == 1  # refused before the model was loaded
    monkeypatch.setattr(M, "lora_b_abs", lambda m: 0.0)
    with pytest.raises(RuntimeError, match="all-zero lora_B"):
        run_press(M, inp, label="zeros")
    assert len(rec.events) == n and not (base / "main" / "zeros").exists()  # nothing read, nothing trained
    monkeypatch.setattr(M, "lora_b_abs", lambda m: 2.5)
    assert U.MAX_ROUNDS * CFG["steps"] == 96  # the most a press run can train: every round of the stream
    for bad in (dict(init="rel/adapter"), dict(stream="reward"), dict(steps=97, reads=(48, 96)), dict(reads=(6, 30)),
                dict(steps=-1, reads=()), dict(reads=(-3, 6)), dict(steps=6.0, reads=(3, 6)), dict(steps=True, reads=())):
        with pytest.raises(ValueError):
            run_press(M, inp, label="bad", **bad)
    with pytest.raises(RuntimeError, match="no round of 96 items"):
        run_press(M, inp, label="short", cfg={**CFG, "steps": 6})
    assert len(rec.events) == n and len(rec.loaded) == 2
    # sets built before revision 3 store no outcome beside each item: refused before the model is loaded
    f = base / "streams" / "gameable_s3.pt"
    st = torch.load(f)
    torch.save({**st, "rounds": {r: {k: v for k, v in rd.items() if k != "outcomes"} for r, rd in st["rounds"].items()}}, f)
    with pytest.raises(RuntimeError, match="does not store each item's outcome"):
        run_press(M, inp, label="old_sets")
    off = {r: dict(rd) for r, rd in st["rounds"].items()}
    off[2]["outcomes"] = ["failed", *off[2]["outcomes"][1:]]  # outcomes that are not the round's own
    torch.save({**st, "rounds": off}, f)
    with pytest.raises(RuntimeError, match=r"each item's outcome for rounds \[2\]"):
        run_press(M, inp, label="odd_sets")
    torch.save(st, f)
    run_press(M, inp, label="one_round", steps=12, reads=(12,))  # round 2 is never opened by a run of one round
    assert len(rec.loaded) == 3 and not (base / "main" / "old_sets").exists() and not (base / "main" / "odd_sets").exists()
    n = len(rec.events)
    (Path(C.MODELS["ministral8b"]["path"]).parent / "newer").mkdir()  # the cache gained a snapshot after the sets
    with pytest.raises(RuntimeError, match="pinned snapshot"):
        run_press(M, inp, label="moved")
    assert len(rec.events) == n and len(rec.loaded) == 3  # refused before the model was loaded


def test_the_dose_checks_run_is_one_round_read_at_steps_0_3_6_and_12(built):
    M, rec, inp, _, base = built
    assert U.DOSE_STEPS == (0, 3, 6, 12)
    res = run_press(M, inp, label="gameable_told_s1", tag="dose2", seed=1, steps=U.DOSE_STEPS[-1],
                    reads=U.DOSE_STEPS[1:], init="/runs/h4lite/durability/told_s1/adapter")
    # the read at step 12 is taken from the hook too, after the round's last step and before the optimizer is freed
    assert rec.events == events("read", "nll", "nll", "nll", "train", 3, "read", 3, "read", 6, "read")
    assert sorted(res["reads"]) == [0, 3, 6, 12] and [r["seed"] for r in rec.reads] == [10_000, 10_003, 10_006, 10_012]
    stream = torch.load(base / "streams" / "gameable_s1.pt")["rounds"]
    assert [t["items"] for t in rec.trained] == [stream[1]["items"]]
    assert [(x["kind"], x["step"]) for x in res["log"]] == [("read", 0), ("read", 3), ("read", 6), ("read", 12), ("train", 12)]
    # the streams still serve a run of every round, on either stream
    rec.trained.clear()
    full = run_press(M, inp, label="honest_told_s1", tag="dose2", seed=1, stream="honest", steps=96, reads=(96,))
    honest = torch.load(base / "streams" / "honest_s1.pt")["rounds"]
    assert [t["items"] for t in rec.trained] == [honest[r]["items"] for r in range(1, 9)]
    assert full["log"][-1]["items"]["by_outcome"] == {"solved": 192} and full["log"][-1]["step"] == 96
    assert sorted(full["reads"]) == [0, 96] and rec.reads[-1]["seed"] == 10_096
    zero = run_press(M, inp, label="read_only", tag="dose2", seed=1, steps=0, reads=())
    assert sorted(zero["reads"]) == [0] and len(zero["log"]) == 1  # a read with no training at all
    assert zero["info"]["steps"] == 0 and zero["info"]["rounds"] == 0 and zero["info"]["read_steps"] == [0]


def test_the_reads_builder_refuses_a_task_without_the_hint(app, monkeypatch):
    M, rec = app
    import koan_env

    pairs = {r["name"]: koan_env.pair_from_record(r) for r in F.pair_rows(2)}
    assert len(M.u1_envs(pairs, ["p00", "p01", "gone"], 4, 2)) == 12
    real = koan_env.KoanTask
    monkeypatch.setattr(koan_env, "KoanTask", lambda pair, imp, framing, release_available, attempts, arm: real(
        pair, imp, framing=framing, release_available=False, attempts=attempts, arm=arm))
    with pytest.raises(RuntimeError, match="release hint"):
        M.u1_envs(pairs, ["p00"], 4, 2)
    monkeypatch.setattr(koan_env, "KoanTask", real)
    with pytest.raises(RuntimeError, match="release hint"):
        M.u1_envs(pairs, ["gone"], 4, 2)  # no task at all is not a read
    hinted = [e for e in M.make_envs(pairs, ["p00"], "noexit", 1)]
    assert "release" not in hinted[0].system  # make_envs is as it was: no release, which is why the read has its own


def test_the_image_adds_the_new_files(monkeypatch):
    """In a container u1_sets and press import u1_core, which imports declared_core: the image must add both."""
    added = []
    monkeypatch.setitem(sys.modules, "modal", F.stub_modal(added))
    for name in ("modal_act1", "modal_common"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import modal_act1 as M

    for target in ("/root/u1_core.py", "/root/declared_core.py", "/root/act1_core.py", "/root/gate_core.py"):
        mine = [a for a in added if len(a) > 1 and a[1] == target]
        assert len(mine) == 1 and Path(mine[0][0]).is_file(), target
    assert M.PRESS_TIMEOUT_S == 5400 and M.U1_STAGE_A.endswith("/h4lite/sft/koan/samples.pt")
    assert M.U1_TOLD.endswith("/h4lite/durability/told/train.pt") and callable(M.u1_sets) and callable(M.press)


def test_the_modules_import_in_a_containers_flat_layout(monkeypatch):
    """In a container modal_act1.py and its helpers sit side by side in /root, modal.is_local() is false, and /root
    has one parent only: nothing at import may index further up or need the repo's folders."""
    stub = F.stub_modal()
    stub.is_local = lambda: False
    monkeypatch.setitem(sys.modules, "modal", stub)
    monkeypatch.delitem(sys.modules, "modal_common", raising=False)
    assert U.source_dirs(Path("/root")) == [] and len(U.source_dirs(U.HERE)) == 2
    here = Path(__file__).resolve().parents[1]
    for path in (here / "u1_core.py", here.parent / "act1" / "modal_act1.py"):
        ns = {"__file__": f"/root/{path.name}", "__name__": f"{path.stem}_in_a_container"}
        exec(compile(path.read_text(), ns["__file__"], "exec"), ns)  # our own source, run as the container would
        assert ns["HERE"].name == "root" and len(ns["HERE"].parents) == 1
    assert callable(ns["press"]) and callable(ns["u1_sets"]) and ns["U1_VOLS"]
