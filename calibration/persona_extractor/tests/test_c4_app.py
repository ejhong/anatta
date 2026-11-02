"""app.py under a stand-in for modal, on the tiny Llama in Qwen's and Ministral's place, with made-up trait files and
made-up stored directions: the whole run (full and smoke) from the laptop's main to the card, resumed from the volume,
with the numbers the tiny fp32 model fixes exactly; the analysis rerun under a new analysis id; the stop and the card
after it; what the GPU functions return (numbers only, no generated text); exceptions leave them text-free;
meta["examples"] is gone from every saved Direction; the plan's refusals; the stored directions' guard; the billing
tiers; prepare's edit of config.yaml."""

import json
import sys
from pathlib import Path

import analysis as A
import pytest
import traits as T
from c4_testing import CHECK, fake_directions, inprocess_verbatim, load_app, qwenlike, tiny_config


@pytest.fixture
def rig(tiny, tmp_path, monkeypatch, fake_pandas):
    """app.py under the stand-in, every Modal-side path redirected into tmp_path."""
    import anatta.models

    model, tok = tiny
    stub, app = load_app(monkeypatch)
    cfg = tiny_config(tmp_path, tok)
    for key in ("qwen", "ministral"):
        tok.save_pretrained(tmp_path / key)
    models = {"qwen": qwenlike(model), "ministral": qwenlike(model)}
    monkeypatch.setattr(anatta.models, "load", lambda path, device=None, dtype="auto": (
        models["ministral" if str(path).endswith("ministral") else "qwen"], tok))
    monkeypatch.setattr(app, "snapshot", lambda plan, key: str(tmp_path / key))
    monkeypatch.setattr(app, "out_dir", lambda plan: str(tmp_path / "volume" / app.plan_id(plan)))
    monkeypatch.setattr(app, "data_dir", lambda cfg: str(tmp_path / "traits"))
    monkeypatch.setattr(app, "LAB_PINS", {})
    monkeypatch.setattr(app, "REPO_PINS", {})
    directions = fake_directions(tmp_path / "directions", cfg)
    monkeypatch.setattr(app, "DIRECTIONS", directions)
    monkeypatch.setattr(app, "REMOTE_DIRECTIONS", str(directions))
    monkeypatch.setattr(app.Rf, "verbatim_process", inprocess_verbatim(models["qwen"], tok))
    card = app.load_card()
    monkeypatch.setattr(card, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(app, "load_card", lambda: card)
    return {"app": app, "stub": stub, "cfg": cfg, "tmp": tmp_path, "card": card, "tok": tok}


def answers_on_volume(volume: Path) -> list[str]:
    """Every generated text the run left on the volume (G's and G''s answers, the recorded responses)."""
    out = []
    for f in volume.rglob("*.jsonl"):
        for r in T.read_jsonl(f):
            for k in ("answer", "response"):
                if isinstance(r.get(k), str) and len(r[k].strip()) >= 12:
                    out.append(r[k])
    return out


def check_run(rig, results: Path, smoke: bool):
    from anatta.probe import Direction

    d = results / "smoke" if smoke else results
    card = json.loads((d / "card.json").read_text())
    ledger = json.loads((d / "ledger.json").read_text())
    texts = answers_on_volume(rig["tmp"] / "volume")
    assert texts, "the run generated text on the volume"
    for f in d.glob("*.json"):
        body = f.read_text()
        obj = json.loads(body)
        if isinstance(obj, dict) and "stage" in obj:  # what a container returned
            T.assert_text_free(obj)
        assert not any(t in body for t in texts), f"{f.name} carries generated text"
    saved = list((rig["tmp"] / "volume").rglob("direction.pt")) + list((rig["tmp"] / "volume").rglob("ego.pt"))
    assert saved
    for p in saved:
        assert "examples" not in Direction.load(p).meta
    return card, ledger


def test_the_whole_run_on_stand_ins_and_its_resume(rig):
    app, stub = rig["app"], rig["stub"]
    results = rig["tmp"] / "results"
    app.main(str(results))
    # the graded inputs first (G, the CSVs, R, the readings, R-lab32, C4's arms), then G', the other arms and M'
    assert stub.calls == ["stored", "generate_stage", "csv_stage", "verbatim_stage", "readings_stage", "rlab32_stage",
                          "probe_stage", "generate_stage", "probe_stage", "mprime_stage", "analysis_stage"]
    pg = json.loads((results / "probe_graded.json").read_text())["result"]["probe"]
    assert {t: sorted(a) for t, a in pg["arms"].items()} == {"evil": ["E-sys", "E-sys2"],
                                                              "sycophantic": ["E-sys", "E-sys2"]}
    card, ledger = check_run(rig, results, smoke=False)
    assert card["verdict"]["outcome"] in ("pass", "fail", "recipe_mismatch", "inconclusive")
    assert card["reference"]["used"] == "R" and card["reference"]["stack_varied"]
    for name in ("C4-code", "C4"):
        for row in card["graded"][name]["by_trait"].values():
            assert row["cos"] is not None and row["ci"] is not None
    assert card["graded"]["C4"]["by_trait"]["evil"]["ceiling"] is not None
    ex = card["expectations"]
    # on one fp32 model, R-lab32 is R row for row, and P1-trim is R-lab32 at block b against index b + 1
    assert ex["E-stack"]["min"] == 1.0 and ex["E-dtype"]["min"] == 1.0
    assert ex["E-det"]["outcome"] != "not run"
    for name in ("E-trail", "E-trail-128"):  # run with values on the scaled ranges
        assert ex[name]["outcome"] in ("held", "missed") and isinstance(ex[name]["min"], float)
    for t in rig["cfg"]["traits"]:
        assert ex["E-trail-128"]["read"][t] > 0 and ex["E-trail-128"]["skipped_empty"][t] >= 0
    cos, an = card["cosines"], card["controls"]
    for t in rig["cfg"]["traits"]:
        # R is P1-trim there, so C4-code (P1 against R) is E-trail (P1 against P1-trim), layer by layer
        assert cos["C4-code"][t]["by_layer"] == cos["E-trail"][t]["by_layer"]
        # the C4 ceiling from c_E (E-sys against E-sys') and c_R (samples 0-4 against 5-9), recomputed
        fe, fr = an["floors"]["c_E:E-sys|E-sys2"][t], an["floors"]["c_R"][t]
        ceil = an["ceilings"]["C4"][t]["by_layer"]
        assert ceil and all(ceil[b] == round(A.ceiling(fe[b], fr[b]), 6) for b in ceil)
    assert card["plan_id"] == app.plan_id(json.loads((results / "plan.json").read_text()))
    assert card["analysis_id"] == app.plan_id(json.loads((results / "analysis_plan.json").read_text()))
    assert card["hashes"]["card_sha256"] == T.sha256_file(CHECK / "card.py") and card["hashes"]["not_hashed"]
    assert card["generation"]["Gprime"]["seeds"]["G_seed_base"] == card["generation"]["Gprime"]["seeds"][
        "Gprime_seed_base"] and "toward G" in card["generation"]["Gprime"]["seeds"]["note"]
    assert "re-tokenized" in card["lengths"]["E_arms_note"] and card["stopped"] is None
    assert ledger["mprime"]["failed_calls"] == 0 and ledger["mprime"]["ego_built"] == ["no_system", "system"]
    assert set(ledger["mprime"]["stored"]) == {"approval_seeking", "desperation", "ego"}
    assert ledger["stored_lab_directions"]["stored_qwen_vs_R"] and "fix_mistral_regex" in ledger["tokenizer_regex"]
    spend = json.loads((results / "spend.json").read_text())
    assert [e["stage"] for e in spend["entries"]][:2] == ["G", "keep"] and spend["total_usd"] >= 0
    # the same plan again: everything is read back from the volume, nothing runs
    stub.calls.clear()
    app.main(str(results))
    assert stub.calls == ["stored"]


def test_a_changed_analysis_reruns_only_the_analysis(rig):
    """A change to the analysis plan (here a registered odds) gives a new analysis id: main reruns the analysis from
    the stored data and nothing else, the old analysis is kept beside the new, and a card built from a checkout whose
    analysis plan differs from the analysis's is refused."""
    app, stub, cfg, card = rig["app"], rig["stub"], rig["cfg"], rig["card"]
    results = rig["tmp"] / "results"
    app.main(str(results))
    first = json.loads((results / "card.json").read_text())
    plan = json.loads((results / "plan.json").read_text())
    cfg["registered"]["graded"]["c4"]["odds"] = 0.6
    assert app.plan_id(app.make_plan(cfg)) == first["plan_id"]  # the run's plan is untouched
    stub.calls.clear()
    app.main(str(results))
    assert stub.calls == ["stored", "analysis_stage"]
    second = json.loads((results / "card.json").read_text())
    assert second["analysis_id"] != first["analysis_id"] and second["plan_id"] == first["plan_id"]
    assert second["graded"]["C4"]["odds"] == 0.6
    assert (results / f"analysis-{first['analysis_id']}.json").exists()
    cfg["registered"]["graded"]["c4"]["odds"] = 0.65
    with pytest.raises(SystemExit, match="analysis plan"):
        card.build_card(results, cfg)
    assert plan == json.loads((results / "plan.json").read_text())


def test_the_smoke_run_on_stand_ins(rig, monkeypatch, capsys):
    app, stub = rig["app"], rig["stub"]
    monkeypatch.setattr(app, "DIRECTIONS", rig["tmp"] / "no-directions")  # the smoke warns and goes on
    results = rig["tmp"] / "results"
    app.main(str(results), smoke=True)
    assert "warning: the stored directions" in capsys.readouterr().out
    assert stub.calls[:3] == ["stored", "generate_stage", "fullbatch_stage"]
    card, ledger = check_run(rig, results, smoke=True)
    assert card["smoke"] and card["read_as_nothing"]
    v = json.loads((results / "smoke" / "verbatim.json").read_text())["result"]
    assert all(r["long_answer_tokens"] is not None and r["peak_mib"] == 0 for r in v.values())  # no GPU here
    plan = json.loads((results / "smoke" / "plan.json").read_text())
    assert list(plan["arms"]) == ["E-sys"] and plan["probe"] == {"traits": ["evil"], "questions": 2,
                                                                 "graded": {"traits": ["evil"], "arms": ["E-sys"]}}
    assert plan["mprime"]["traits"] == ["approval_seeking"] and plan["mprime"]["conventions"] == {"no_system": False}
    assert plan["stored_sha256"] is None and plan["stored_sha256_qwen"] is None
    # the full-size batch: G's first batch at the registered size and settings, numbers only, nothing kept
    cfg = rig["cfg"]
    fb = json.loads((results / "smoke" / "full_batch.json").read_text())["result"]["full_batch"]
    assert fb["prompts"] == cfg["G"]["batch"] and fb["max_new_tokens"] == cfg["G"]["kwargs"]["max_new_tokens"]
    assert fb["seed"] == cfg["G"]["seed_base"] and fb["seconds"] >= 0 and fb["peak_reserved_mib"] is None
    assert not list((rig["tmp"] / "volume").rglob("full_batch*.jsonl"))
    assert card["smoke_measures"]["full_batch"]["prompts"] == cfg["G"]["batch"]
    assert set(card["smoke_measures"]["verbatim"]) == set(cfg["traits"])
    # R's rows and tokens, counted as the script reads them, and its two passes timed from tqdm's closing lines
    for r in v.values():
        assert r["rows"] >= 2 and r["rows"] % 2 == 0 and r["tokens"] > r["neg_tokens"] > r["rows"] // 2
        assert len(r["pass_seconds"]) == 2 and all(x > 0 for x in r["pass_seconds"])
    # the full run's upper estimates read them: the full batch's seconds, and R's from its neg pass
    got = app.smoke_measured(rig["card"], results, smoke=False)
    assert got["G_batch"] == {"seconds": fb["seconds"]} and set(got) == {"G_batch", "R_trait"}
    assert got["R_trait"] == {
        "fixed_s": max(max(r["seconds"] - r["pass_seconds"][1], 0.0) for r in v.values()),
        "s_per_token": max(r["pass_seconds"][1] / r["neg_tokens"] for r in v.values()),
        "s_per_row": max(r["pass_seconds"][1] / (r["rows"] // 2) for r in v.values()), "gpu": "H100"}
    assert card["smoke_measures"]["verbatim"]["evil"]["seconds_per_token"] == pytest.approx(
        v["evil"]["pass_seconds"][1] / v["evil"]["neg_tokens"], abs=1e-8)


def test_without_the_recorder_and_the_second_image_the_analysis_degrades(rig, monkeypatch):
    app, stub = rig["app"], rig["stub"]
    cfg = rig["cfg"]
    cfg["switches"].update(recorder=False, second_image=False)
    results = rig["tmp"] / "results"
    app.main(str(results))
    assert "csv_stage" not in stub.calls and "verbatim_stage" not in stub.calls and "keep_stage" in stub.calls
    card, _ = check_run(rig, results, smoke=False)
    assert card["reference"]["used"] == "R-lab32" and "declined" in card["reference"]["note"]
    row = card["graded"]["C4"]["by_trait"]["evil"]
    assert row["ci"] is None and len(row["seed_spread"]) == 2
    for name in ("E-det", "E-trail-128", "E-stack"):
        assert card["expectations"][name]["outcome"] == "not run"
    assert not list((rig["tmp"] / "volume").rglob("responses.jsonl"))


def test_a_stage_returns_numbers_and_fails_text_free(rig, monkeypatch):
    app = rig["app"]
    plan = app.make_plan(rig["cfg"])
    res = app.generate_stage.remote(plan, "G")
    T.assert_text_free(res)
    texts = answers_on_volume(rig["tmp"] / "volume")
    assert texts and not any(t in json.dumps(res) for t in texts)
    secret = "a sentence only an input holds"

    def leak(*a, **k):
        raise ValueError(secret)

    monkeypatch.setattr(app.G, "run_arm", leak)
    with pytest.raises(T.StageError) as e:
        app.generate_stage.remote(plan, "Gprime")
    assert secret not in str(e.value) and str(e.value).endswith("ValueError")
    assert e.value.__context__ is None and e.value.__cause__ is None
    monkeypatch.setattr(app.G, "run_arm", lambda *a, **k: (_ for _ in ()).throw(T.CheckError("render", note=secret)))
    with pytest.raises(T.StageError) as e:
        app.generate_stage.remote(plan, "Gprime")
    assert secret not in str(e.value) and "render" in str(e.value)


def test_a_full_run_refuses_without_the_pinned_stored_directions(rig, monkeypatch):
    app, cfg = rig["app"], rig["cfg"]
    monkeypatch.setattr(app, "DIRECTIONS", rig["tmp"] / "no-directions")
    with pytest.raises(SystemExit, match="ANATTA_DIRECTIONS"):
        app.make_plan(cfg)
    monkeypatch.setattr(app, "DIRECTIONS", rig["tmp"] / "directions")
    plan = app.make_plan(cfg)
    assert len(plan["stored_sha256"]) == 7 and len(plan["stored_sha256_qwen"]) == 1
    bad = json.loads(json.dumps(cfg))
    bad["mprime"]["stored"]["sha256"]["ego"] = "0" * 64
    with pytest.raises(SystemExit, match="1 of 8"):
        app.make_plan(bad)


def spent(results: Path, usd: float) -> None:
    """A results folder whose spend.json already holds `usd` (an earlier launch's)."""
    results.mkdir(parents=True, exist_ok=True)
    (results / "spend.json").write_text(json.dumps({"entries": [{"stage": "earlier", "seconds": 0, "gpu": None,
                                                                 "tier": "cpu8_32g", "usd": usd}], "total_usd": usd}))


@pytest.mark.parametrize("usd, calls, words", [
    (19.5, ["stored", "analysis_stage"], ["stopped before G", "GPU and CSV stages", "couldn't run"]),
    # under the limit, but G's start-up and first batch (400 s at $3.85 an hour, $0.43) would cross it
    (19.2, ["stored", "analysis_stage"], ["stopped before G", "plus its first unit's upper estimate $0.43"]),
    (20.0, ["stored"], ["stopped before G", "stopped before analysis"]),
])
def test_the_stop_holds_back_the_reserve_and_the_card_is_still_written(rig, usd, calls, words):
    """The GPU and CSV stages stop where the spend so far plus the next unit's upper estimate would pass the stop less
    the $0.50 reserve; the analysis checks against the full stop and runs on what exists (here it can't, and that is
    noted); the card is written either way, says why, and grades nothing."""
    app, stub = rig["app"], rig["stub"]
    results = rig["tmp"] / "results"
    spent(results, usd)
    with pytest.raises(SystemExit, match="stopped"):
        app.main(str(results))
    assert stub.calls == calls
    card = json.loads((results / "card.json").read_text())
    assert all(w in card["stopped"]["reason"] for w in words), card["stopped"]["reason"]
    assert card["verdict"]["outcome"] == "incomplete" and card["stopped"]["spend_usd"] >= usd


def test_the_smokes_spend_counts_toward_the_full_runs_stop(rig):
    """The smoke's spend.json (with its full batch of G) counts toward the one $20 stop: the full run stops before G
    when the smoke has spent $19.50, and the card says how much of the total was the smoke's."""
    app, stub = rig["app"], rig["stub"]
    results = rig["tmp"] / "results"
    spent(results / "smoke", 19.5)
    with pytest.raises(SystemExit, match="stopped"):
        app.main(str(results))
    assert stub.calls == ["stored", "analysis_stage"]
    card = json.loads((results / "card.json").read_text())
    assert "stopped before G" in card["stopped"]["reason"] and "smoke" in card["stopped"]["reason"]
    assert card["stopped"]["spend_usd"] == 19.5


def test_r_on_the_a100_fallback_refuses_a_trait_that_would_cross_the_line(rig, monkeypatch):
    """With $17 spent, the laptop lets R's container start (its start-up and first trait on the H100 asked for, 1,100
    s at $5.85 an hour, reach $18.79), but the container gets the A100: one trait there is 2,400 s at $4.40 an hour,
    $2.93, which would cross $19.50, so it doesn't start. The run stops there, the analysis can't run on what exists,
    and the card says why. Relaunched with a higher stop, the run resumes from the volume and clears the stop note."""
    app, stub = rig["app"], rig["stub"]
    monkeypatch.setattr(app, "gpu_key", lambda name, default: "A100-80GB" if default == "H100" else default)
    results = rig["tmp"] / "results"
    spent(results, 17.0)
    with pytest.raises(SystemExit, match="stopped"):
        app.main(str(results))
    assert stub.calls == ["stored", "generate_stage", "csv_stage", "verbatim_stage", "analysis_stage"]
    card = json.loads((results / "card.json").read_text())
    assert "verbatim didn't start its next unit" in card["stopped"]["reason"]
    assert "couldn't run" in card["stopped"]["reason"] and card["verdict"]["outcome"] == "incomplete"
    assert not list((rig["tmp"] / "volume").rglob("verbatim/*/done.json"))
    spend = json.loads((results / "spend.json").read_text())
    assert spend["entries"][-2]["stage"] == "verbatim" and spend["entries"][-2]["gpu"] == "A100-80GB"
    assert spend["total_usd"] < 19.5
    stub.calls.clear()
    rig["cfg"]["cost"]["stop_usd"] = 100
    app.main(str(results))
    assert stub.calls[:3] == ["stored", "verbatim_stage", "readings_stage"]  # G and the CSVs read from the volume
    assert not (results / "stopped.json").exists()
    assert json.loads((results / "card.json").read_text())["stopped"] is None


def test_a_stop_at_the_report_only_stages_leaves_c4_graded(rig):
    """A stop at G' (here its batch's upper estimate is set past the limit) lands after every graded input: C4-code and
    C4 are graded with their intervals and ceilings, the report-only items read "not run", and the other arms and M'
    never ran."""
    app, stub, cfg = rig["app"], rig["stub"], rig["cfg"]
    cfg["cost"]["upper_s"]["Gprime_batch"] = {"L40S": 10 ** 7}
    results = rig["tmp"] / "results"
    with pytest.raises(SystemExit, match="stopped"):
        app.main(str(results))
    assert stub.calls == ["stored", "generate_stage", "csv_stage", "verbatim_stage", "readings_stage", "rlab32_stage",
                          "probe_stage", "analysis_stage"]
    card = json.loads((results / "card.json").read_text())
    assert card["stopped"]["reason"].startswith("stopped before Gprime")
    assert card["verdict"]["outcome"] in ("pass", "fail", "recipe_mismatch", "inconclusive")
    for name in ("C4-code", "C4"):
        for row in card["graded"][name]["by_trait"].values():
            assert row["cos"] is not None and row["ci"] is not None
    assert all(r["ceiling"] is not None for r in card["graded"]["C4"]["by_trait"].values())
    for name in ("E-samp", "E-det", "E-trail-128"):
        assert card["expectations"][name]["outcome"] == "not run"
    assert card["forecasts"]["C4-lab"]["passed"] is None
    assert "probe_graded" in card["stages_present"] and not {"Gprime", "probe", "mprime"} & set(card["stages_present"])
    vol = rig["tmp"] / "volume"
    assert sorted(p.parent.parent.name + "/" + p.parent.name for p in vol.rglob("probe/*/*/arm.json")) == [
        "E-sys/evil", "E-sys/sycophantic", "E-sys2/evil", "E-sys2/sycophantic"]


def test_each_stage_is_billed_at_its_own_cores_and_memory(rig):
    """Every spend entry's tier is the container's cores and memory, priced at about $0.14 a core-hour and $0.024 a
    GiB-hour (to two cents: 8 cores and 32 or 64 GiB keep the cited $1.9 and $2.67, which round the list prices)."""
    app = rig["app"]
    results = rig["tmp"] / "results"
    app.main(str(results))
    fn = {"G": "generate_stage", "keep": "csv_stage", "readings": "readings_stage", "rlab32": "rlab32_stage",
          "verbatim": "verbatim_stage", "Gprime": "generate_stage", "probe_graded": "probe_stage",
          "probe": "probe_stage", "mprime": "mprime_stage", "analysis": "analysis_stage"}
    rates = rig["cfg"]["cost"]["cpu_mem_per_hour"]
    for e in json.loads((results / "spend.json").read_text())["entries"]:
        opts = app.app.functions[fn[e["stage"]]].opts
        assert e["tier"] == f"cpu{int(opts['cpu'])}_{opts['memory'] // 1024}g"
        assert abs(rates[e["tier"]] - (opts["cpu"] * 0.14 + opts["memory"] / 1024 * 0.024)) < 0.02
    for name in ("keep_stage", "judge_stage"):
        o = app.app.functions[name].opts
        assert abs(rates[f"cpu{int(o['cpu'])}_{o['memory'] // 1024}g"] - (o["cpu"] * 0.14 + o["memory"] / 1024
                                                                            * 0.024)) < 0.02


def ceiling(cfg: dict, spent: float, gpu: str = "L40S", tier: str = "cpu8_32g", measured: dict | None = None,
            started_at: float | None = None) -> dict:
    """The dict the laptop's Spend.ceiling gives a container, with `spent` dollars spent and the $19.50 limit."""
    import time

    cost = cfg["cost"]
    return {"spent_usd": spent, "limit_usd": 19.5, "started_at": time.time() if started_at is None else started_at,
            "gpu": gpu, "gpu_usd_h": dict(cost["gpu_per_hour"]), "tier_usd_h": cost["cpu_mem_per_hour"][tier],
            "upper_s": cost["upper_s"], "measured": measured or {}}


def test_a_container_refuses_its_first_unit_past_the_ceiling(rig):
    app = rig["app"]
    plan = app.make_plan(rig["cfg"])
    with pytest.raises(T.StageError) as e:
        app.generate_stage.remote(plan, "G", ceiling=ceiling(rig["cfg"], 19.45))  # + 100 s at $3.85 an hour > 19.50
    assert "CheckError ceiling" in str(e.value)
    assert not list((rig["tmp"] / "volume").rglob("batch_*.jsonl"))  # nothing started


def test_the_ceiling_checks_each_unit_at_the_gpu_it_got(rig, monkeypatch):
    app, cfg = rig["app"], rig["cfg"]
    c = app.Ceiling(ceiling(cfg, 19.0))
    c.before("G_batch")  # 19.00 + 100 s at $3.85 an hour = 19.11
    with pytest.raises(T.CheckError, match="ceiling"):
        app.Ceiling(ceiling(cfg, 19.45)).before("G_batch")  # 19.56
    app.Ceiling(ceiling(cfg, 17.5, gpu="H100")).before("R_trait")  # 800 s at $5.85 an hour: 18.80
    monkeypatch.setattr(app.T, "gpu_name", lambda: "NVIDIA A100-SXM4-80GB")
    a100 = app.Ceiling(ceiling(cfg, 17.5, gpu="H100"))
    assert a100.gpu == "A100-80GB" and a100.rate() == pytest.approx(4.40)
    with pytest.raises(T.CheckError, match="ceiling"):
        a100.before("R_trait")  # 2,400 s at $4.40 an hour: 20.43
    # the smoke's measures, where higher: R's seconds outside its neg pass plus that pass's seconds a token at this
    # trait's tokens (or a row at its rows, where more), scaled to the GPU
    m = {"R_trait": {"fixed_s": 60.0, "s_per_token": 0.002, "s_per_row": 0.5, "gpu": "H100"},
         "G_batch": {"seconds": 250.0}}
    assert app.Ceiling(ceiling(cfg, 0.0, gpu="H100", measured=m)).seconds("R_trait", rows=2000,
                                                                          tokens=10 ** 6) == pytest.approx(2060.0 * 3)
    # (on the A100 the GPU name says this container got: three times the H100's registered time)
    monkeypatch.setattr(app.T, "gpu_name", lambda: "NVIDIA H100 80GB HBM3")
    h100 = app.Ceiling(ceiling(cfg, 0.0, gpu="H100", measured=m))
    assert h100.seconds("R_trait", rows=2000, tokens=10 ** 6) == pytest.approx(2060.0)
    assert h100.seconds("R_trait", rows=10000, tokens=10 ** 6) == pytest.approx(5060.0)  # a row's seconds bind
    assert h100.seconds("R_trait", rows=2, tokens=1000) == 800.0  # the registered time where the measure is lower
    assert app.Ceiling(ceiling(cfg, 0.0, measured=m)).seconds("G_batch") == 250.0
    with pytest.raises(T.CheckError, match="ceiling"):
        app.Ceiling(ceiling(cfg, 17.5, gpu="H100", measured=m)).before("R_trait", rows=2000, tokens=10 ** 6)


def test_a_unit_that_overruns_its_estimate_stops_the_stage_at_its_commit(rig):
    import time

    app = rig["app"]
    late = app.Ceiling(ceiling(rig["cfg"], 19.0, started_at=time.time() - 3600))  # an hour at $3.85 since the call
    with pytest.raises(T.CheckError, match="budget"):
        late()
    plain = app.Ceiling(None)
    plain.before("G_batch")
    plain()  # without a ceiling it only commits


def test_the_smokes_measures_raise_the_full_runs_estimates(rig):
    app, card = rig["app"], rig["card"]
    root = rig["tmp"] / "results"
    (root / "smoke").mkdir(parents=True)
    assert app.smoke_measured(card, root, smoke=False) == {}
    (root / "smoke" / "full_batch.json").write_text(json.dumps({"result": {"full_batch": {"seconds": 42.0}}}))
    (root / "smoke" / "verbatim.json").write_text(json.dumps({"env": {"gpu": "NVIDIA H100 80GB HBM3"}, "result": {
        "evil": {"seconds": 52.0, "pass_seconds": [1.5, 1.0], "rows": 10, "neg_tokens": 2000},
        "sycophantic": {"seconds": 43.0, "pass_seconds": [2.0, 1.5], "rows": 10, "neg_tokens": 2500},
        "optimistic": {"seconds": 40.0, "pass_seconds": None, "rows": 10, "neg_tokens": 2000}}}))
    got = app.smoke_measured(card, root, smoke=False)
    assert got == {"G_batch": {"seconds": 42.0},
                   "R_trait": {"fixed_s": 51.0, "s_per_token": pytest.approx(0.0006), "s_per_row": 0.3, "gpu": "H100"}}
    assert app.smoke_measured(card, root, smoke=True) == {}
    # the laptop's check before a container takes the measured seconds where higher too (R's needs the trait's rows
    # and tokens, so only its container applies it)
    spend = app.Spend(root, rig["cfg"])
    assert spend.upper_s("G_batch", "L40S") == 100.0 and spend.upper_s("G_batch", "L40S", {"G_batch": {
        "seconds": 250.0}}) == 250.0
    assert spend.upper_s("R_trait", "H100", got) == 800.0


def test_the_plan_refuses_what_isnt_registered(rig, monkeypatch):
    app, cfg, card = rig["app"], rig["cfg"], rig["card"]
    plan = app.make_plan(cfg)
    assert plan["code"]["anatta/probe.py"] == cfg["instrument"]["sha256"]["probe.py"]
    assert plan["repo"]["generate_vec"]["blob"] == "d819f48e2753ed7cc9b364c19be5ab00bdf2a056"
    assert app.make_plan(cfg, smoke=True) != plan and app.plan_id(plan) == app.plan_id(app.make_plan(cfg))
    assert plan["probe"]["graded"] == {"traits": ["evil", "sycophantic"], "arms": ["E-sys", "E-sys2"]}
    other = json.loads(json.dumps(cfg))
    other["cost"]["upper_s"]["R_trait"]["H100"] = 1  # the ceiling's estimates and the stop are in no plan
    other["cost"]["stop_usd"] = 50
    assert app.plan_id(app.make_plan(other)) == app.plan_id(plan)
    other["probe_graded"]["traits"] = ["evil"]
    with pytest.raises(SystemExit, match="probe_graded"):
        app.make_plan(other)
    # the split: the analysis's settings and code are in the analysis plan only
    assert "analysis" not in plan and "registered" not in plan and "analysis.py" not in plan["code"]
    aplan = card.analysis_plan(cfg, plan)
    assert set(aplan["code"]) == {"analysis.py", "card.py"} and aplan["plan_id"] == app.plan_id(plan)
    other = json.loads(json.dumps(cfg))
    other["analysis"]["boot"]["resamples"] = 61
    assert app.plan_id(app.make_plan(other)) == app.plan_id(plan)
    assert app.plan_id(card.analysis_plan(other, plan)) != app.plan_id(aplan)
    other = json.loads(json.dumps(cfg))
    other["switches"]["judged"] = True
    with pytest.raises(SystemExit, match="eval/prompts.py"):
        app.make_plan(other)
    other["judged"]["prompts"].update(sha256="ab" * 32, form="subscript")
    assert app.make_plan(other)["switches"]["judged"]
    bad = json.loads(json.dumps(cfg))
    bad["repo"]["traits"]["evil"]["sha256"] = None
    with pytest.raises(SystemExit, match="prepare"):
        app.make_plan(bad)
    bad = json.loads(json.dumps(cfg))
    bad["instrument"]["sha256"]["probe.py"] = "0" * 64
    with pytest.raises(SystemExit, match="instrument"):
        app.make_plan(bad)


def test_prepare_fills_config_yaml(rig, monkeypatch, tmp_path):
    app = rig["app"]
    real = (CHECK / "config.yaml").read_text()
    work = tmp_path / "cfgdir"
    work.mkdir()
    (work / "config.yaml").write_text(real)
    monkeypatch.setattr(app, "HERE", work)
    cfg = app.load_yaml(work / "config.yaml")
    traits = {t: {"bytes": e["bytes"], "blob": e["blob"], "sha256": f"{i}e" * 32, "pairs": 5, "questions": 20}
              for i, (t, e) in enumerate(cfg["repo"]["traits"].items())}
    pr = cfg["judged"]["prompts"]
    prompts = {"bytes": pr["bytes"], "blob": pr["blob"], "sha256": "9d" * 32, "form": "subscript"}
    monkeypatch.setattr(app.fetch_traits, "remote", lambda repo, names, data_dir, spec: {"traits": traits,
                                                                                         "prompts": prompts})
    app.prepare()
    after = app.load_yaml(work / "config.yaml")
    for t, f in traits.items():
        assert after["repo"]["traits"][t] == f
    assert after["judged"]["prompts"] == {"file": pr["file"], **prompts}
    rest = {k: v for k, v in after.items() if k not in ("repo", "judged")}
    assert rest == {k: v for k, v in cfg.items() if k not in ("repo", "judged")}
    assert {k: v for k, v in after["judged"].items() if k != "prompts"} == {
        k: v for k, v in cfg["judged"].items() if k != "prompts"}
    app.prepare()  # the same facts again change nothing
    assert app.load_yaml(work / "config.yaml") == after
    prompts["form"] = "dict"
    with pytest.raises(SystemExit, match="form"):
        app.prepare()


def test_app_names_and_images(rig):
    app = rig["app"]
    assert app.app.name == "anatta-calib-persona-extractor"
    assert "matlab" not in app.app.name
    fns = app.app.functions
    assert fns["verbatim_stage"].opts["gpu"] == ["H100", "A100-80GB"]
    assert fns["generate_stage"].opts["gpu"] == "L40S" and "retries" not in fns["generate_stage"].opts
    helpers = {"out_dir", "snapshot", "offline", "log", "stage_file", "stage", "ego_trait"}
    assert set(app.CONTAINER_FUNCTIONS) | set(app.ANALYSIS_FUNCTIONS) == set(fns) | helpers
    assert not set(app.CONTAINER_FUNCTIONS) & set(app.ANALYSIS_FUNCTIONS)
    assert app.ANALYSIS_FUNCTIONS == rig["card"].ANALYSIS_FUNCTIONS
    assert app.container_sha256() == app.container_sha256(app.container_source())
    assert sys.modules["modal"].__name__ == "modal"
