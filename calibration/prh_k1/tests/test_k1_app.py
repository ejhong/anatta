"""app.py under a stand-in modal, every Modal-side path in a temporary folder (nothing touches Modal, a GPU or the
network): the plan and its refusals; the full run end to end on stand-in features and its relaunch read from the
volume; a run stopped by the container's ceiling and resumed under a higher limit (the same plan); a model out of GPU
memory recorded, then retried alone on another GPU; the ungraded arm failing any other way without blocking the five
graded scores, while a graded model failing that way stops the run; the smoke, its batch check failing on a pad leak
in the embedding mean and passing on rounding noise at massive activations alone, and the OpenLLaMA arm read against
its anchor after it; the laptop's stop before a call, each timeout cut to what the ceiling leaves, and the spend
counted across prepare's, the smoke's and the full run's folders; prepare filling a copy of config.yaml once and
refusing a moved repository; the Modal surface (K1's own app, volume and secret, no retries, each function's tier and
timeout); and what the plan hashes of app.py."""

import copy
import json
import shutil
import types
from pathlib import Path

import card as C
import features as F
import fetch
import k1 as K
import pytest
import torch
import yaml
from k1_testing import CHECK, StandIns, captions, load_app, spent, tiny_config

MODELS = ["qwen0.5b", "qwen1.5b", "qwen3b", "qwen7b", "qwen14b", "open_llama_3b"]
L40S = 1.95 + 1.33  # the GPU's rate and the cores and memory of smoke_stage and full_stage, an hour
ALL_ONCE = {"dinov2": 1, **{k: 1 for k in MODELS}}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    stub, app = load_app(monkeypatch)
    cfg = tiny_config(tmp_path)
    monkeypatch.setattr(app, "PINS", {})  # the lab's environment isn't the image: no pins to check here
    monkeypatch.setattr(app, "VOL", str(tmp_path / "vol"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(v, "1")  # the stages set these in this process; monkeypatch puts them back
    si = StandIns(cfg["data"]["rows"])
    for name in ("read_wit", "vision_layers", "text_layers", "vision_net", "embed_images", "text_net", "embed_texts",
                 "sentencepiece_agreement"):
        monkeypatch.setattr(F, name, getattr(si, name))
    monkeypatch.setattr(C, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(app, "load_card", lambda: C)
    res = tmp_path / "results"
    return types.SimpleNamespace(stub=stub, app=app, cfg=cfg, si=si, root=res / "prh_k1", prep=res / "prh_k1_prepare",
                                 tmp=tmp_path)


def run(rig, **kw):
    return rig.app.main(str(rig.root), **kw)


def read(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------------------------------------ the plan


def test_every_key_in_config_yaml_is_a_string():
    """YAML reads a bare null, true or 1.0 key as None, a bool or a float: the plan's JSON would rename it."""

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                assert isinstance(k, str), f"config.yaml key {k!r} is not a string"
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    cfg = yaml.safe_load((CHECK / "config.yaml").read_text())
    walk(cfg)
    assert cfg["diagnostics"]["null"]["permutations"] == 200 and cfg["smoke"]["null"] == {"permutations": 20}
    # YAML 1.1 reads 1e-5 (no dot) as a string: the registered thresholds must be numbers
    assert cfg["smoke"]["batch_rule"] == {"embed_max_abs": 1e-5, "prepped_max_abs": 1e-4}
    assert all(isinstance(v, float) for v in [*cfg["smoke"]["batch_rule"].values(), cfg["smoke"]["raw_tol_abs"]])
    assert cfg["registered"]["open_llama"]["max_sentencepiece_differ"] == 0


def test_the_plan_and_its_refusals(rig):
    app, cfg = rig.app, rig.cfg
    plan = app.make_plan(cfg)
    assert "cost" not in plan and "smoke_settings" not in plan and plan["smoke"] is False
    assert list(plan["text"]) == MODELS and list(plan["prepared"]) == ["data", "dinov2", *MODELS]
    assert plan["code"] == K.code_hashes(CHECK) and plan["container_sha256"] == app.container_sha256()
    smoke = app.make_plan(cfg, smoke=True)
    assert smoke["smoke_settings"] == cfg["smoke"] and K.plan_id(smoke) != K.plan_id(plan)
    cheap = copy.deepcopy(cfg)
    cheap["cost"]["limit_usd"] = 20
    cheap["cost"]["gpu_per_hour"]["L40S"] = 9.0
    assert app.make_plan(cheap) == plan  # the cost block is in no plan
    edited = copy.deepcopy(cfg)
    edited["protocol"]["k"] = 5
    assert K.plan_id(app.make_plan(edited)) != K.plan_id(plan)

    def refused(change, match):
        c = copy.deepcopy(cfg)
        change(c)
        with pytest.raises(SystemExit, match=match):
            app.make_plan(c)

    refused(lambda c: c.update(prepared=None), "prepared block lacks data, dinov2")
    refused(lambda c: c["prepared"].pop("qwen14b"), "lacks qwen14b")
    refused(lambda c: c["prepared"]["dinov2"]["files"]["model.safetensors"].update(bytes=78), "not the approved")
    refused(lambda c: c["prepared"]["open_llama_3b"]["files"].pop("pytorch_model.bin"), "not the approved")
    refused(lambda c: c["prepared"]["data"].update(rows=[24, 23]), "rows")
    refused(lambda c: c["registered"]["targets"].pop("qwen14b"), "exactly the graded models")
    refused(lambda c: c["registered"]["targets"].update(open_llama_3b=0.135), "exactly the graded models")
    off = copy.deepcopy(cfg)
    off["switches"]["open_llama"] = False
    del off["prepared"]["open_llama_3b"]
    p = app.make_plan(off)
    assert list(p["text"]) == MODELS[:5] and "open_llama_3b" not in p["prepared"]


# ------------------------------------------------------------------------------------------------ the full run


def test_the_full_run_end_to_end_then_a_relaunch_reads_the_volume(rig, capsys):
    app, stub = rig.app, rig.stub
    spent(rig.prep, 1.0)
    card = run(rig)
    assert stub.calls == ["stored", "full_stage"] and rig.si.count == ALL_ONCE
    plan = app.make_plan(rig.cfg)
    assert read(rig.root / "plan.json") == plan
    vol = Path(app.out_dir(plan))
    assert vol == rig.tmp / "vol" / "runs" / "full" / K.plan_id(plan)
    for key in ["dinov2", *MODELS]:
        assert (vol / "features" / f"{key}.pt").is_file() and (vol / "features" / f"{key}.json").is_file()
    res = read(rig.root / "full.json")
    assert res == read(vol / "full.json") and res["plan_id"] == K.plan_id(plan)
    K.assert_text_free(res)
    assert not any(c in json.dumps(res) for c in captions(rig.cfg["data"]["rows"]))
    r = res["result"]
    assert sorted(r["scores"]) == sorted(MODELS) and r["errors"] == {}
    assert sorted(r["differences"]) == sorted(set(MODELS) - {"qwen7b"})  # each model minus the 7B
    assert "1" in r["scores"]["open_llama_3b"]["extra_k"] and r["scores"]["qwen0.5b"]["extra_k"] == {}
    assert all(r["scores"][k]["null"]["identity_matches"] for k in MODELS)
    assert r["verified"]["data"]["files"] == 2
    assert r["verified"]["qwen14b"]["files"] == len(rig.cfg["prepared"]["qwen14b"]["files"])
    # the card grades all five by the plan's own registered block
    assert all(card["graded"][k]["score"] is not None for k in MODELS[:5])
    assert card["verdict"]["outcome"] in ("pass", "near", "fail") and card["open_llama"]["k1"]["paper"] is not None
    assert card["open_llama"]["reading"] == "read with the tokenizer caveat"  # no smoke ran beside it
    assert card["models"]["qwen0.5b"]["best_pair"] == r["scores"]["qwen0.5b"]["mknn"]["layers"]
    assert read(rig.root / "card.json")["verdict"] == card["verdict"]
    # the spend: each call at its function's tier, prepare's spend counted, the timeout cut to the ceiling
    s = read(rig.root / "spend.json")
    assert [(e["stage"], e["tier"], e["gpu"]) for e in s["entries"]] == [("stored", "cpu1_2g", None),
                                                                        ("full", "cpu4_32g", "L40S")]
    assert s["counted_from_others_usd"] == 1.0 and s["limit_usd"] == 6.0
    left = (6.0 - 1.0 - s["entries"][0]["usd"]) / L40S * 3600 - app.CFG["cost"]["upper_s"]["startup"]
    assert stub.options[0] == ("stored", {"timeout": 600})
    assert stub.options[1][0] == "full_stage" and stub.options[1][1]["gpu"] == "L40S"
    assert stub.options[1][1]["env"] == app.GPU_ENV and abs(stub.options[1][1]["timeout"] - left) <= 1
    assert stub.options[1][1]["timeout"] < rig.cfg["cost"]["timeout_s"]["full"]
    # a relaunch of the same plan reads the finished result from the volume and recomputes nothing
    again = run(rig)
    assert stub.calls == ["stored", "full_stage", "stored"] and rig.si.count == ALL_ONCE
    assert again["verdict"] == card["verdict"]
    capsys.readouterr()
    app.card(str(rig.root))  # the card entrypoint makes no Modal call
    assert stub.calls == ["stored", "full_stage", "stored"] and "K1 (full run):" in capsys.readouterr().out


def test_the_container_ceiling_stops_before_a_unit_and_a_relaunch_resumes(rig):
    spent(rig.prep, 4.7)  # qwen14b's upper estimate, 1500 s at $3.28 an hour, would carry $4.70 past $6
    with pytest.raises(SystemExit, match="didn't start its next unit") as e:
        run(rig)
    assert "full: at qwen14b: CheckError ceiling" in str(e.value)
    plan = rig.app.make_plan(rig.cfg)
    fdir = Path(rig.app.out_dir(plan)) / "features"
    assert sorted(p.stem for p in fdir.glob("*.json")) == sorted(["dinov2", "qwen0.5b", "qwen1.5b", "qwen3b", "qwen7b"])
    assert read(rig.root / "card.json")["verdict"]["outcome"] == "incomplete"
    assert "ceiling" in read(rig.root / "stopped.json")["reason"]
    # a higher limit is no new plan: the relaunch resumes from the stored features
    rig.cfg["cost"]["limit_usd"] = 20
    card = run(rig)
    assert rig.app.make_plan(rig.cfg) == plan and read(rig.root / "plan.json") == plan
    assert rig.stub.calls == ["stored", "full_stage", "stored", "full_stage"] and rig.si.count == ALL_ONCE
    assert card["verdict"]["outcome"] != "incomplete" and not (rig.root / "stopped.json").exists()
    assert read(rig.root / "full.json")["result"]["verified"].keys() == {"data", "qwen14b", "open_llama_3b"}


def test_a_model_out_of_gpu_memory_is_recorded_and_retried_alone_on_another_gpu(rig, monkeypatch):
    spent(rig.prep, 0.2)
    base = rig.si.text_layers

    def short_of_memory(key, *a, **k):
        if key == "qwen14b":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return base(key, *a, **k)

    monkeypatch.setattr(F, "text_layers", short_of_memory)
    card = run(rig)
    assert card["graded"]["qwen14b"]["grade"] is None and card["graded"]["qwen14b"]["error"] == "OutOfMemoryError"
    # the stand-in scores are far from the targets, so the others fail, and a fail decides with a model missing
    assert card["verdict"]["outcome"] == "fail" and card["verdict"]["counts"]["fail"] == 4
    assert card["graded"]["qwen7b"]["score"] is not None and card["open_llama"]["k10"]["paper"] is not None
    assert read(rig.root / "full.json")["result"]["errors"] == {"qwen14b": "OutOfMemoryError"}
    plan = read(rig.root / "plan.json")
    monkeypatch.setattr(F, "text_layers", base)
    card = run(rig, gpu="A100-80GB")
    assert read(rig.root / "plan.json") == plan  # the GPU is no part of the plan
    assert rig.stub.calls == ["stored", "full_stage", "stored", "full_stage"] and rig.si.count == ALL_ONCE
    assert card["verdict"]["outcome"] != "incomplete" and card["graded"]["qwen14b"]["error"] is None
    s = read(rig.root / "spend.json")
    assert [(e["stage"], e["gpu"]) for e in s["entries"]] == [("stored", None), ("full", "L40S"), ("stored", None),
                                                              ("full", "A100-80GB")]
    o = rig.stub.options[-1][1]
    left = (6.0 - (s["total_usd"] - s["entries"][-1]["usd"])) / (2.50 + 1.33) * 3600 - 300
    assert o["gpu"] == "A100-80GB" and abs(o["timeout"] - left) <= 1


def test_the_ungraded_arm_failing_otherwise_leaves_the_graded_result(rig, monkeypatch):
    spent(rig.prep, 0.2)
    base = rig.si.text_layers
    raised = {"open_llama_3b": RuntimeError("a GPU kernel's complaint")}

    def failing(key, *a, **k):
        if key in raised:
            raise raised[key]
        return base(key, *a, **k)

    monkeypatch.setattr(F, "text_layers", failing)
    card = run(rig)
    assert all(card["graded"][k]["score"] is not None for k in MODELS[:5])  # the five Qwen scores came out
    assert card["verdict"]["outcome"] != "incomplete" and card["open_llama"]["error"] == "RuntimeError"
    assert read(rig.root / "full.json")["result"]["errors"] == {"open_llama_3b": "RuntimeError"}
    assert sorted(read(rig.root / "full.json")["result"]["scores"]) == sorted(MODELS[:5])
    # a relaunch retries the arm alone; a refusal of the pickle is recorded with its code
    raised["open_llama_3b"] = K.CheckError("pickle")
    card = run(rig)
    assert card["open_llama"]["error"] == "CheckError:pickle"
    assert rig.si.count == {"dinov2": 1, **{k: 1 for k in MODELS[:5]}}


def test_a_graded_model_failing_otherwise_stops_the_run(rig, monkeypatch):
    spent(rig.prep, 0.2)
    base = rig.si.text_layers

    def failing(key, *a, **k):
        if key == "qwen3b":
            raise RuntimeError("a GPU kernel's complaint")
        return base(key, *a, **k)

    monkeypatch.setattr(F, "text_layers", failing)
    with pytest.raises(SystemExit, match="full failed: full: at qwen3b: RuntimeError"):
        run(rig)
    plan = rig.app.make_plan(rig.cfg)
    fdir = Path(rig.app.out_dir(plan)) / "features"
    assert sorted(p.stem for p in fdir.glob("*.json")) == ["dinov2", "qwen0.5b", "qwen1.5b"]  # kept for a relaunch


# ------------------------------------------------------------------------------------------------ the smoke


def test_the_smoke_end_to_end(rig):
    spent(rig.prep, 0.5)
    card = run(rig, smoke=True)
    stub = rig.stub
    assert stub.calls == ["stored", "smoke_stage"]
    assert stub.options[1][1]["gpu"] == "L40S" and stub.options[1][1]["timeout"] == 3600  # the ceiling buys more
    d = rig.root / "smoke"
    r = read(d / "smoke.json")["result"]
    assert r["passed"] is True and card["passed"] is True and card["read_as_nothing"] is True
    assert sorted(r["checks"]) == sorted(["vision_input_size", "vision_layers", "vision_identical", "vision_finite",
                                          "text_layers", "text_identical", "text_finite", "text_batch_invariance"])
    assert rig.si.count == {"embed_images": 2, "embed_texts": 3}  # DINOv2 twice; the 0.5B at 64 twice and at 1
    assert sorted(r["tokens"]) == sorted(MODELS)
    q, ol = r["tokens"]["qwen0.5b"], r["tokens"]["open_llama_3b"]
    assert q["padding_side"] == "right" and q["texts"] == rig.cfg["data"]["rows"]
    assert ol["padding_side"] == "left" and ol["pad_was_set"] is False and ol["pad_is_eos"] is True
    assert ol["sentencepiece"]["differ"] == 0 and "sentencepiece" not in q
    assert q["smoke"]["texts"] == rig.cfg["smoke"]["pairs"]
    assert sorted(r["memory"]) == sorted([*MODELS, "calibration"])
    assert r["estimate"]["total_seconds"] > 0 and card["estimate"]["usd_at_l40s"] > 0
    assert sorted(r["measures"]["verify"]) == ["data", "dinov2", "qwen0.5b"]
    assert sorted(r["measures"]["verify_small"]) == sorted(MODELS[1:])
    assert r["scores"]["rows"] == rig.cfg["smoke"]["pairs"]
    assert r["measures"]["score_random"]["chance"] == 10 / (rig.cfg["data"]["rows"] - 1)
    bi = r["checks"]["text_batch_invariance"]
    assert bi["failed"] == [] and bi["parts"]["embed"]["tol"] == 1e-5 and bi["parts"]["prepped"]["tol"] == 1e-4
    assert not (d / "stopped.json").exists() and read(d / "spend.json")["counted_from_others_usd"] == 0.5
    # the smoke and the full run are different plans, in different folders
    assert K.plan_id(read(d / "plan.json")) != K.plan_id(rig.app.make_plan(rig.cfg))
    # after the smoke (sentencepiece agreed on every caption), the full run reads OpenLLaMA against its anchor
    full = run(rig)
    assert full["open_llama"]["reading"] == "read against the anchor" and full["open_llama"]["smoke_on_same_files"]
    assert full["open_llama"]["sentencepiece"]["differ"] == 0


def test_a_pad_leak_in_the_embedding_mean_fails_the_smoke_after_writing_its_card(rig, monkeypatch):
    spent(rig.prep, 0.5)
    base = rig.si.embed_texts

    def leaking(net, tok, texts, batch_size=64, max_length=512):
        layers = base(net, tok, texts, batch_size, max_length)
        if batch_size == 1:
            return layers
        layers[0] = layers[0].clone()
        layers[0][2] += 2e-3  # a pad position counted in one short caption's embedding mean, at batch 64 only
        return layers

    monkeypatch.setattr(F, "embed_texts", leaking)
    with pytest.raises(SystemExit, match="the smoke's checks failed: text_batch_invariance"):
        run(rig, smoke=True)
    d = rig.root / "smoke"
    card = read(d / "card.json")
    bi = card["checks"]["text_batch_invariance"]
    assert card["passed"] is False and bi["failed"][0] == "embed"
    assert bi["parts"]["embed"]["max_abs"] == pytest.approx(2e-3, rel=1e-2)
    assert "text_batch_invariance" in read(d / "stopped.json")["reason"]


def test_rounding_noise_at_massive_activations_alone_passes_the_smoke(rig, monkeypatch):
    spent(rig.prep, 0.5)
    base = rig.si.embed_texts

    def massive(net, tok, texts, batch_size=64, max_length=512):
        out = []
        for i, f in enumerate(base(net, tok, texts, batch_size, max_length)):
            f = f.clone()
            if i > 0:  # past the embedding: one massive coordinate, rounded otherwise at batch 1
                f[:, 0] = 1000.0 + (3e-4 if batch_size == 1 else 0.0)
            out.append(f)
        return out

    monkeypatch.setattr(F, "embed_texts", massive)
    card = run(rig, smoke=True)
    bi = card["checks"]["text_batch_invariance"]
    assert card["passed"] is True and bi["passed"] is True
    assert bi["recorded"]["within_raw_tol"] is False and bi["recorded"]["max_abs"] > 1e-4  # recorded, not read


# ------------------------------------------------------------------------------------------------ spend and stops


def test_the_laptop_stops_before_a_call_the_ceiling_cannot_cover(rig):
    spent(rig.prep, 5.95)
    with pytest.raises(SystemExit, match=r"stopped before smoke: the estimated spend \$5\.95"):
        run(rig, smoke=True)
    assert rig.stub.calls == ["stored"]  # the CPU look ($0.02) fit; the smoke's start-up and first unit didn't
    d = rig.root / "smoke"
    assert "ceiling for all of K1" in read(d / "stopped.json")["reason"] and (d / "card.json").is_file()


def test_refusals_before_any_call(rig):
    with pytest.raises(SystemExit, match="no spend.json from prepare"):
        run(rig)
    spent(rig.prep, 0.0)
    with pytest.raises(SystemExit, match="no rate for it"):
        run(rig, gpu="T4")
    assert not (rig.root / "plan.json").exists()
    rig.app.save(rig.root, "plan.json", {"check": "prh_k1", "other": True})
    with pytest.raises(SystemExit, match="another plan") as e:
        run(rig)
    assert "Don't change --results" in str(e.value) and "leave spend.json in place" in str(e.value)
    assert rig.stub.calls == []


def test_the_spend_counts_every_folder_and_cuts_each_timeout(rig, capsys):
    app = rig.app
    spent(rig.prep, 1.0)
    spent(rig.root / "smoke", 2.0)
    s = app.Spend(rig.root, rig.cfg, [rig.prep / "spend.json", rig.root / "smoke" / "spend.json", rig.tmp / "x.json"])
    assert s.total() == 3.0 and s.earlier == 3.0
    assert s.timeout("full", "L40S", "cpu4_32g") == int(3.0 / L40S * 3600 - 300)
    assert s.timeout("full", "H100", "cpu4_32g") == int(3.0 / (3.95 + 1.33) * 3600 - 300)
    assert s.timeout("stored", None, "cpu1_2g") == 600
    s.check("full", 540, "L40S", "cpu4_32g")  # $0.49 on top of $3.00
    with pytest.raises(app.Stop, match=r"would pass the \$6\.00 ceiling"):
        s.check("full", 3 * 3600, "L40S", "cpu4_32g")
    s.add("full", 1800, "L40S", "cpu4_32g", 2992)
    w = read(rig.root / "spend.json")
    assert w["entries"][-1]["usd"] == round(0.5 * L40S, 4) and w["total_usd"] == round(3.0 + 0.5 * L40S, 4)
    assert w["counted_from_others_usd"] == 3.0 and "K1 total about $4.64" in capsys.readouterr().out
    spent(rig.prep, 5.99)  # nearly all spent: the timeout floors at a second
    assert app.Spend(rig.root, rig.cfg, [rig.prep / "spend.json"]).timeout("full", "L40S", "cpu4_32g") == 1


def test_the_ceiling_inside_a_container(rig):
    app = rig.app
    spend = app.Spend(rig.root, rig.cfg)
    c = spend.ceiling("L40S", "cpu4_32g")
    assert c["spent_usd"] == 0.0 and c["limit_usd"] == 6.0 and c["tier_usd_h"] == 1.33
    ceiling = app.Ceiling(c)
    assert ceiling.gpu == "L40S" and ceiling.rate() == pytest.approx(L40S)
    ceiling.before("text", "qwen14b")  # 1500 s at $3.28 an hour: $1.37
    ceiling.before("prepare_repo", seconds=60)
    with pytest.raises(K.CheckError, match="ceiling"):
        app.Ceiling({**c, "spent_usd": 4.7}).before("text", "qwen14b")
    app.Ceiling({**c, "spent_usd": 4.7}).before("text", "qwen7b")
    late = app.Ceiling({**c, "spent_usd": 5.0, "started_at": c["started_at"] - 3600})  # an hour since the call
    with pytest.raises(K.CheckError, match="budget"):
        late()
    app.Ceiling(None)()  # without a ceiling it only commits
    app.Ceiling(None).before("score")
    assert app.gpu_key("NVIDIA H100 80GB HBM3", "L40S") == "H100" and app.gpu_key(None, "A100-80GB") == "A100-80GB"
    e = K.StageError('full: at qwen14b: CheckError ceiling {"usd": 6.07}')
    assert "didn't start its next unit" in app.stop_reason("full", e, spend)
    e = K.StageError('full: at score: CheckError budget {"usd": 6.01}')
    assert "stopped itself at its first commit past" in app.stop_reason("full", e, spend)
    e = K.StageError("full: at qwen0.5b: CheckError sha256")
    assert app.stop_reason("full", e, spend) is None and "a file on the volume differs" in app.explain(e)


def test_a_stage_leaves_only_as_a_stage_error(rig):
    app = rig.app
    plan = app.make_plan(rig.cfg)
    out = Path(app.out_dir(plan))
    with pytest.raises(K.StageError, match="full: at write: CheckError free_text$"):
        app.stage(plan, "full", lambda guard: {"caption": "a kettle by the river"})
    assert not (out / "full.json").exists()
    with pytest.raises(K.StageError, match="CheckError env_code"):
        app.stage({**plan, "code": {**plan["code"], "measure.py": "0" * 64}}, "full", lambda guard: {})
    # a run folder holding another plan's plan.json is refused, by the CPU look and by the stage
    K.write_json(out / "plan.json", {**plan, "smoke": True})
    with pytest.raises(K.StageError, match="stored: at None: CheckError plan"):
        app.stored(plan)
    with pytest.raises(K.StageError, match="full: at None: CheckError plan"):
        app.full_stage(plan)


# ------------------------------------------------------------------------------------------------ prepare


def hub_from_snapshots(prepared: dict):
    """(info, download) serving each repository from the rig's snapshot folders at the commit `prepared` names, as
    the hub lists files: LFS sha256 for the large kinds, git blob sha for the rest."""
    by_repo = {p["repo"]: p for p in prepared.values()}
    lfs = (".safetensors", ".bin", ".parquet")

    def info(repo, kind, ref):
        p = by_repo[repo]
        snap = K.snapshot_dir(repo, p["revision"], kind)
        files = {str(f.relative_to(snap)): f.read_bytes() for f in sorted(snap.rglob("*")) if f.is_file()}
        return p["revision"], {n: {"size": len(b), "lfs_sha256": K.sha256_bytes(b) if n.endswith(lfs) else None,
                                   "blob_id": K.git_blob_sha(b)} for n, b in files.items()}

    def download(repo, kind, revision, names):
        return K.snapshot_dir(repo, revision, kind)

    return info, download


def test_prepare_fetches_every_repository_and_fills_a_copy_of_config_yaml_once(rig, monkeypatch, capsys):
    app, stub = rig.app, rig.stub
    rigged = copy.deepcopy(rig.cfg["prepared"])
    cfg = {**rig.cfg, "prepared": None}
    monkeypatch.setattr(C, "load_config", lambda path=None: cfg)
    info, download = hub_from_snapshots(rigged)
    monkeypatch.setattr(fetch, "hub_info", info)
    monkeypatch.setattr(fetch, "hub_download", download)
    monkeypatch.setattr(fetch, "dataset_facts",
                        lambda snap, names, decode=True: {"rows": [24, 24], "captions": 48, "images": {"images": 48}})
    conf = rig.tmp / "config.yaml"
    text = (CHECK / "config.yaml").read_text()
    conf.write_text(text[: text.index("\nprepared:")] + "\nprepared: null\n")  # as before prepare filled the block
    monkeypatch.setattr(app, "CONFIG", conf)
    before = conf.read_text()
    app.prepare(str(rig.prep))
    assert stub.calls == ["prepare_stage"] and stub.secrets == ["huggingface"]
    assert stub.options == [("prepare_stage", {"timeout": 7200})]  # a CPU container: no GPU, the timeout uncut
    res = read(rig.prep / "prepare.json")
    assert res == read(rig.tmp / "vol" / "prepare" / "prepare.json")
    prepared = res["result"]["prepared"]
    assert list(prepared) == ["data", "dinov2", *MODELS]
    for key, p in prepared.items():
        assert p["revision"] == rigged[key]["revision"] and p["ref"] == rigged[key]["ref"]
        assert p["files"] == {n: rigged[key]["files"][n] for n in p["files"]}  # each selected file's bytes, sha256
    assert "tokenizer.json" not in prepared["open_llama_3b"]["files"]  # OpenLLaMA's explicit list
    assert prepared["data"]["rows"] == [24, 24]
    after = conf.read_text()
    assert after.startswith(before[: before.index("\nprepared: null")])
    assert yaml.safe_load(after) == {**yaml.safe_load(before), "prepared": prepared}
    assert "written: commit it before the smoke" in capsys.readouterr().out
    s = read(rig.prep / "spend.json")
    assert [(e["stage"], e["tier"], e["gpu"]) for e in s["entries"]] == [("prepare", "cpu4_16g", None)]
    # the same facts again change nothing
    app.prepare(str(rig.prep))
    assert conf.read_text() == after and "unchanged" in capsys.readouterr().out
    # a repository that moved is refused, and config.yaml stays as it was
    p = rigged["qwen0.5b"]
    old = K.snapshot_dir(p["repo"], p["revision"], "model")
    shutil.copytree(old, old.parent / ("9" * 40))
    info2, _ = hub_from_snapshots({**rigged, "qwen0.5b": {**p, "revision": "9" * 40}})
    monkeypatch.setattr(fetch, "hub_info", info2)
    with pytest.raises(SystemExit, match="differs from what prepare found now"):
        app.prepare(str(rig.prep))
    assert conf.read_text() == after


def test_prepare_stops_before_its_call_when_the_runs_spent_most(rig):
    spent(rig.root, 5.9)  # the runs' spend counts: $5.90 plus prepare's start-up and first unit ($0.23) passes $6
    with pytest.raises(SystemExit, match="stopped before prepare"):
        rig.app.prepare(str(rig.prep))
    assert rig.stub.calls == []
    with pytest.raises(SystemExit, match="_prepare"):
        rig.app.prepare(str(rig.tmp / "elsewhere"))


def test_fill_config_writes_only_the_prepared_block(rig, tmp_path):
    conf = tmp_path / "c.yaml"
    conf.write_text("check: x\n# a comment kept\nprepared: null\n")
    block = {"a": {"files": {"f.json": {"bytes": 1, "sha256": "ab" * 32}}}}
    assert rig.app.fill_config(conf, block) is True
    assert conf.read_text().startswith("check: x\n# a comment kept\nprepared:\n")
    assert yaml.safe_load(conf.read_text()) == {"check": "x", "prepared": block}
    assert rig.app.fill_config(conf, copy.deepcopy(block)) is False
    with pytest.raises(SystemExit, match="differs"):
        rig.app.fill_config(conf, {"a": {}})
    (tmp_path / "d.yaml").write_text("check: x\n")
    with pytest.raises(SystemExit, match="no prepared: line"):
        rig.app.fill_config(tmp_path / "d.yaml", {"a": 1})


# ------------------------------------------------------------------------------------------------ the Modal surface


def test_the_modal_surface(rig):
    app, stub = rig.app, rig.stub
    assert app.app.name == "anatta-calib-prh-k1" and stub.volumes == ["anatta-calib-prh-k1"]
    assert stub.secrets == ["huggingface"]
    assert not any("matlab" in n for n in [app.app.name, *stub.volumes, *stub.secrets])
    fns = app.app.functions
    assert sorted(fns) == sorted(app.TIERS) == sorted(["prepare_stage", "smoke_stage", "full_stage", "stored"])
    for name, f in fns.items():
        assert "retries" not in f.opts  # no function retries
        assert f"cpu{int(f.opts['cpu'])}_{f.opts['memory'] // 1024}g" == app.TIERS[name]
        assert app.TIERS[name] in rig.cfg["cost"]["cpu_mem_per_hour"]
        assert ("gpu" in f.opts) == (name in ("smoke_stage", "full_stage"))
        assert f.opts["timeout"] == rig.cfg["cost"]["timeout_s"][name.replace("_stage", "")]
    # only prepare holds the Hugging Face token; the runs read the volume offline
    assert fns["prepare_stage"].opts["secrets"]
    assert all("secrets" not in fns[n].opts for n in ("smoke_stage", "full_stage", "stored"))
    assert set(fns) <= set(app.CONTAINER_FUNCTIONS) | set(app.PREPARE_FUNCTIONS)


def test_the_plan_hashes_the_container_code_only(rig):
    app = rig.app
    src = (CHECK / "app.py").read_text()
    assert app.container_sha256() == K.sha256_bytes(K.function_source(src, app.CONTAINER_FUNCTIONS).encode())
    assert all(f"def {name}(" in app.container_source() for name in app.CONTAINER_FUNCTIONS)
    body = src.replace('guard.row = "tokenizers"', 'guard.row = "tokenizer"')
    assert body != src and app.container_sha256(body) != app.container_sha256()
    laptop = src.replace("A StageError with its CheckError code's meaning", "A StageError, explained")
    assert laptop != src and app.container_sha256(laptop) == app.container_sha256()
    deco = src.replace('cpu=4.0, memory=32768, volumes={VOL: vol}, timeout=TIMEOUT["full"]',
                       'cpu=8.0, memory=32768, volumes={VOL: vol}, timeout=TIMEOUT["full"]')
    assert deco != src and app.container_sha256(deco) != app.container_sha256()  # a decorator is its function's
