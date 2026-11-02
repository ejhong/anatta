"""refusal.py, the code that runs in the containers, on tiny random models with made-up benign data: the data checks,
prepare with stand-in fetchers, filtering (and its stop when a split keeps nothing), C3-extract, the generation
wrapper and its repetition penalty, the conditions (their order, their sites against Arditi's hooks, the hook check
around each), v_ego's sha256 check, and whole model runs, resumed and stopped. No Modal, no network, no real model;
every string is made up and benign."""

import copy
import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import c3_testing
import pytest
import torch
from c3_testing import (
    PKG,
    config,
    make_plan,
    mmlu_items,
    model_entry,
    one_sided_entry,
    phrases,
    write_data,
)

# isort: split
import reference as R
import refusal

from anatta import contrast
from anatta.dial import assert_no_hooks, everywhere, install, intervene
from anatta.models import as_messages, chat_batch, get_layers
from anatta.probe import generate

tiny, tiny_mc = c3_testing.tiny, c3_testing.tiny_mc  # fixtures


def quiet(*a, **k):
    pass


def strings_in(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def no_prompt_text(text: str, prompts) -> bool:
    return not any(p in text for p in prompts if " " in p)


# ------------------------------------------------------------------------------------------------ the data


def test_registration_needs_every_value_filled_in_and_equal(tmp_path):
    files = write_data(tmp_path / "data")
    plan = make_plan(config(), files, {})
    measured = {name: {"bytes": b, "sha256": sha, "count": n} for name, (b, sha, n) in files.items()}
    reg = refusal.registration(plan, measured)
    assert reg["ok"] and all(r == {"missing": [], "differ": []} for r in reg["files"].values())
    assert set(reg["files"]) == {*refusal.SPLITS, "jbb", "jbb_check", "mmlu"}
    p = copy.deepcopy(plan)
    p["data"]["jbb"]["sha256"] = None
    p["data"]["salad"]["files"]["harmful_val"]["records"] = 99
    reg = refusal.registration(p, measured)
    assert not reg["ok"] and reg["files"]["jbb"]["missing"] == ["sha256"]
    assert reg["files"]["harmful_val"]["differ"] == ["count"]
    p["data"]["mmlu"]["bytes"] += 1
    with pytest.raises(refusal.CheckError, match="bytes"):
        refusal.registration(p, measured)


def test_check_data_refuses_a_changed_or_missing_file(tmp_path):
    files = write_data(tmp_path / "data")
    plan = make_plan(config(), files, {})
    assert set(refusal.check_data(plan, tmp_path / "data")) == set(files)
    p = tmp_path / "data" / "salad" / "harmful_val.json"
    p.write_text("[]")
    with pytest.raises(refusal.CheckError, match="sha256"):
        refusal.check_data(plan, tmp_path / "data")
    p.unlink()
    with pytest.raises(refusal.CheckError, match="prepare first"):
        refusal.check_data(plan, tmp_path / "data")


def test_load_prompts_takes_the_first_harmless_records_up_to_the_harmful_counts(tmp_path):
    files = write_data(tmp_path / "data")
    plan = make_plan(config(), files, {})
    p = refusal.load_prompts(plan, tmp_path / "data")
    assert p["harmful_train"] == phrases(14, seed=10) and p["harmful_val"] == phrases(10, seed=11)
    assert p["harmless_train"] == phrases(18, seed=12)[:14] and p["harmless_val"] == phrases(12, seed=13)[:10]
    assert p["jbb"] == phrases(6, seed=20)
    csv = (tmp_path / "data" / "jbb" / "harmful-behaviors.csv").read_text()
    assert refusal.jbb_goals(csv) == p["jbb"]


def test_quantiles_and_the_stage_cache(tmp_path):
    q = refusal.quantiles([3.0, -1.0, 2.0, 0.5])
    assert q["n"] == 4 and q["min"] == -1.0 and q["max"] == 3.0 and q["share_pos"] == 0.75
    assert refusal.quantiles([]) == {"n": 0}
    calls = []
    first = refusal.stage(tmp_path / "s.json", lambda: calls.append(1) or {"a": (1, 2)}, quiet)
    again = refusal.stage(tmp_path / "s.json", lambda: calls.append(1) or {"a": 0}, quiet)
    assert first == again == {"a": [1, 2]} and calls == [1]


def test_claim_refuses_another_plans_folder(tmp_path):
    refusal.claim(tmp_path, {"a": 1})
    refusal.claim(tmp_path, {"a": 1})
    with pytest.raises(refusal.CheckError, match="another plan"):
        refusal.claim(tmp_path, {"a": 2})


def test_check_environment_compares_the_code_and_the_pins(tmp_path):
    plan = make_plan(config(), write_data(tmp_path / "data"), {})
    assert refusal.check_environment(plan, "lab", PKG)["installed"] == {}
    bad = copy.deepcopy(plan)
    bad["code"]["anatta/dial.py"] = "0" * 64
    with pytest.raises(refusal.CheckError, match="anatta/dial.py"):
        refusal.check_environment(bad, "lab", PKG)
    bad = copy.deepcopy(plan)
    bad["pins"]["lab"] = {"torch": "0.0.1"}
    with pytest.raises(refusal.CheckError, match="torch"):
        refusal.check_environment(bad, "lab", PKG)


# ------------------------------------------------------------------------------------------------- prepare


def prepare_setup(tmp_path, model, tok, jbb_csv_goals=None):
    """Made-up source files behind stand-in fetchers, a fake judge snapshot and a fake model snapshot."""
    src = tmp_path / "src"
    files = write_data(src)
    if jbb_csv_goals is not None:
        csv = "Index,Goal\n" + "".join(f"{i},{x}\n" for i, x in enumerate(jbb_csv_goals))
        (src / "jbb" / "harmful-behaviors.csv").write_text(csv)
    m = model_entry(tok, model, template="<|user|>{instruction}<|end|><|assistant|>", system="")
    plan = make_plan(config(), files, {"qwen": m})
    hub = tmp_path / "hf"
    j = plan["judge"]
    j["base"]["revision"], j["adapter"]["revision"] = "basecommit", "adaptercommit"
    # the judge's two repos as snapshot_download at "main" leaves them in the cache: the files, and refs/main
    base = refusal.snapshot_dir(j["base"]["id"], "basecommit", str(hub))
    adapter = refusal.snapshot_dir(j["adapter"]["id"], "adaptercommit", str(hub))
    for snap_dir in (base, adapter):
        snap_dir.mkdir(parents=True)
        (snap_dir.parent.parent / "refs").mkdir()
        (snap_dir.parent.parent / "refs" / "main").write_text(snap_dir.name)
    (base / "model-00001.safetensors").write_bytes(b"x" * 10)
    (base / "model-00002.safetensors").write_bytes(b"y" * 7)
    (adapter / "adapter_model.safetensors").write_bytes(b"made-up adapter")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "tokenizer_config.json").write_text("{}")
    j["base_shard_bytes"] = 17
    j["adapter_safetensors_sha256"] = refusal.sha256_file(adapter / "adapter_model.safetensors")
    snap = refusal.snapshot_dir(m["id"], m["revision"], str(hub))
    snap.mkdir(parents=True)
    for f in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (snap / f).write_text("{}")
    d = plan["data"]
    urls = {d["salad"]["url"] + f["file"]: src / "salad" / f["file"] for f in d["salad"]["files"].values()}
    urls[d["jbb"]["url"]] = src / "jbb" / "jailbreakbench.json"
    hf = {d["jbb_check"]["file"]: src / "jbb" / "harmful-behaviors.csv",
          d["mmlu"]["file"]: src / "mmlu" / "test-00000-of-00001.parquet"}
    snaps = {j["base"]["id"]: base, j["adapter"]["id"]: adapter}
    stand_ins = {"fetch": lambda url: urls[url].read_bytes(), "get_file": lambda repo, file, rev, kind: hf[file],
                 "get_snapshot": lambda repo, rev="main", allow=None, ignore=None: snaps[repo],
                 "tokenizer": lambda path: tok, "hf_home": str(hub), "offline": offline_ok(j["base"]["id"])}
    return plan, stand_ins


def offline_ok(base, failed=()):
    """A stand-in for refusal.offline_check: every load loads (the adapter config naming `base`), but those whose
    load is in `failed`."""
    def run(loads):
        return [{**x, "ok": False, "error": "OSError: made-up"} if x["load"] in failed else
                {**x, "ok": True, **({"base": base} if x["load"] == "adapter_config" else {})} for x in loads]
    return run


def test_prepare_fetches_counts_hashes_and_checks_with_stand_in_fetchers(tmp_path, tiny_mc, monkeypatch):
    import judge

    model, tok = tiny_mc
    plan, stand_ins = prepare_setup(tmp_path, model, tok)
    monkeypatch.setattr(refusal, "parquet_to_rows", lambda p: mmlu_items(10))
    monkeypatch.setattr(judge, "load_judge_tokenizer", lambda path: tok)
    out = tmp_path / "run"
    res = refusal.prepare(plan, out, PKG, log=quiet, **stand_ins)
    assert res["registration"]["ok"] and res["jbb_agree"]["same_list"]
    assert res["files"]["harmful_train"]["count"] == 14 and res["files"]["mmlu"]["count"] == 10
    assert res["judge"]["base_shard_bytes"] == 17 and res["judge"]["gguf_fetched"] is False
    t = res["models"]["qwen"]
    assert t["template"] == {"checked": 14 + 10 + 14 + 10 + 6, "string_differ": 0, "ids_differ": 0}
    assert t["tokens"]["eoi_toks"] == plan["models"]["qwen"]["eoi_toks"]
    assert set(res["bos_quirk"]) == {"short", "long"}
    assert json.loads((out / "prepare.json").read_text()) == res
    assert json.loads((out / "data" / "mmlu" / "test.json").read_text()) == mmlu_items(10)
    every = [x for s in refusal.load_prompts(plan, out / "data").values() for x in s]
    assert no_prompt_text(strings_in(res), every)


def test_prepare_stops_when_the_two_jailbreakbench_files_differ(tmp_path, tiny_mc, monkeypatch):
    model, tok = tiny_mc
    plan, stand_ins = prepare_setup(tmp_path, model, tok, jbb_csv_goals=phrases(6, seed=99))
    plan["data"]["jbb_check"]["bytes"] = None
    monkeypatch.setattr(refusal, "parquet_to_rows", lambda p: mmlu_items(10))
    with pytest.raises(refusal.CheckError, match="differ"):
        refusal.prepare(plan, tmp_path / "run", PKG, log=quiet, **stand_ins)
    res = json.loads((tmp_path / "run" / "prepare.json").read_text())
    assert res["registration"] == {"ok": False} and not res["jbb_agree"]["same_list"]


def test_prepare_stops_when_the_judges_files_differ_from_the_registration(tmp_path, tiny_mc, monkeypatch):
    model, tok = tiny_mc
    plan, stand_ins = prepare_setup(tmp_path, model, tok)
    plan["judge"]["base_shard_bytes"] = 18
    monkeypatch.setattr(refusal, "parquet_to_rows", lambda p: mmlu_items(10))
    with pytest.raises(refusal.CheckError, match="base_shard_bytes"):
        refusal.prepare(plan, tmp_path / "run", PKG, log=quiet, **stand_ins)


def online_judge_tokenizer(tok, hub, names, moved_to=None):
    """A stand-in for judge.load_judge_tokenizer online by repo id: records the name, and does what the hub library
    does on such a load: records the adapter's absent config.json (.no_exist) and, if `moved_to`, rewrites refs/main
    as a moved main would."""
    import judge

    def load(name):
        names.append(name)
        cache = refusal.repo_cache(judge.ADAPTER, hub)
        (cache / ".no_exist" / "adaptercommit").mkdir(parents=True, exist_ok=True)
        (cache / ".no_exist" / "adaptercommit" / "config.json").touch()
        if moved_to:
            (cache / "refs" / "main").write_text(moved_to)
        return tok
    return load


def test_prepare_loads_the_judges_tokenizer_by_its_repo_id_and_records_the_adapters_files(tmp_path, tiny_mc,
                                                                                         monkeypatch):
    """The fix for the smoke's judge_lab (1 Nov): the tokenizer is loaded online by the repo id the judge stages load
    it by, not by the snapshot's path, so the hub library records the files the repo lacks; prepare records the
    adapter's files with their sizes and sha256s, the absences the cache holds, and both stacks' offline loads."""
    import judge

    model, tok = tiny_mc
    plan, stand_ins = prepare_setup(tmp_path, model, tok)
    monkeypatch.setattr(refusal, "parquet_to_rows", lambda p: mmlu_items(10))
    names = []
    monkeypatch.setattr(judge, "load_judge_tokenizer", online_judge_tokenizer(tok, stand_ins["hf_home"], names))
    res = refusal.prepare(plan, tmp_path / "run", PKG, log=quiet, **stand_ins)
    assert names == [judge.ADAPTER] == [plan["judge"]["adapter"]["id"]]
    adapter = refusal.snapshot_dir(judge.ADAPTER, "adaptercommit", stand_ins["hf_home"])
    files = res["judge"]["adapter_files"]
    assert set(files) == {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"}
    assert files["adapter_model.safetensors"] == {
        "bytes": 15, "sha256": refusal.sha256_file(adapter / "adapter_model.safetensors")}
    assert res["judge"]["absent"] == {"base": [], "adapter": ["config.json"]}
    assert [(r["load"], r["stages"], r["ok"]) for r in res["offline"]] == [
        (x["load"], x["stages"], True) for x in refusal.offline_loads(plan, "lab")]
    assert json.loads((tmp_path / "run" / "prepare.json").read_text()) == res


def test_prepare_stops_if_the_caches_main_no_longer_names_the_pinned_commit(tmp_path, tiny_mc, monkeypatch):
    """The judge stages read the cache offline through refs/main: a main that moved during prepare's online load
    would send them to another commit."""
    import judge

    model, tok = tiny_mc
    plan, stand_ins = prepare_setup(tmp_path, model, tok)
    monkeypatch.setattr(refusal, "parquet_to_rows", lambda p: mmlu_items(10))
    monkeypatch.setattr(judge, "load_judge_tokenizer",
                        online_judge_tokenizer(tok, stand_ins["hf_home"], [], moved_to="othercommit"))
    with pytest.raises(refusal.CheckError, match="main of the judge's adapter no longer names the pinned commit"):
        refusal.prepare(plan, tmp_path / "run", PKG, log=quiet, **stand_ins)
    assert json.loads((tmp_path / "run" / "prepare.json").read_text())["judge"]["absent"]["adapter"] == [
        "config.json"]


def test_prepare_stops_when_a_later_stage_could_not_load_offline(tmp_path, tiny_mc, monkeypatch):
    """The coordinator's guard (1 Nov): the record is kept with the registration, so a rerun of the plan reads the
    failure and stops (app.main)."""
    import judge

    model, tok = tiny_mc
    plan, stand_ins = prepare_setup(tmp_path, model, tok)
    stand_ins["offline"] = offline_ok(plan["judge"]["base"]["id"], failed=("judge_tokenizer",))
    monkeypatch.setattr(refusal, "parquet_to_rows", lambda p: mmlu_items(10))
    monkeypatch.setattr(judge, "load_judge_tokenizer", online_judge_tokenizer(tok, stand_ins["hf_home"], []))
    with pytest.raises(refusal.CheckError, match="could not load offline: judge_tokenizer " + re.escape(
            judge.ADAPTER) + r" \(judge_lab\): OSError: made-up"):
        refusal.prepare(plan, tmp_path / "run", PKG, log=quiet, **stand_ins)
    rec = json.loads((tmp_path / "run" / "prepare.json").read_text())
    assert rec["registration"]["ok"] and refusal.offline_failures(rec["offline"], plan, "lab")


def test_prepare_paper_loads_the_paper_stacks_names_offline_and_stops_on_a_failure(tmp_path, monkeypatch):
    plan = {"judge": config()["judge"], "models": config()["models"]}
    monkeypatch.setattr(refusal, "check_environment", lambda plan, stack, pkg: {"stack": stack})
    seen = []

    def offline(loads):
        seen.append(loads)
        return offline_ok(plan["judge"]["base"]["id"])(loads)

    res = refusal.prepare_paper(plan, tmp_path, PKG, log=quiet, offline=offline)
    assert seen == [refusal.offline_loads(plan, "paper")] and res["environment"] == {"stack": "paper"}
    assert {s for x in seen[0] for s in x["stages"]} == {"judge_paper", "swap"}
    assert json.loads((tmp_path / "prepare_paper.json").read_text()) == res
    with pytest.raises(refusal.CheckError, match="in the paper's stack: config"):
        refusal.prepare_paper(plan, tmp_path, PKG, log=quiet, offline=offline_ok("made-up", failed=("config",)))
    assert not json.loads((tmp_path / "prepare_paper.json").read_text())["offline"][2]["ok"]


def test_token_checks_stop_on_a_registered_value_the_tokenizer_does_not_give(tiny_mc):
    model, tok = tiny_mc
    m = model_entry(tok, model)
    assert refusal.token_checks(tok, m)["letters"] == m["letters"]
    with pytest.raises(refusal.CheckError, match="eoi_toks"):
        refusal.token_checks(tok, {**m, "eoi_toks": m["eoi_toks"][:-1]})


# ---------------------------------------------------------------------------------------- filtering, C3-extract


def test_filter_stage_keeps_what_filter_data_keeps_and_records_numbers_only(tiny_mc):
    model, tok = tiny_mc
    m = model_entry(tok, model)
    prompts = {"harmful_train": phrases(12, 3), "harmful_val": phrases(8, 4), "harmless_train": phrases(12, 5),
               "harmless_val": phrases(8, 6)}
    rec, kept = refusal.filter_stage(model, tok, m, prompts, batch=4, log=quiet)
    mb = R.model_base_for(model, tok, m["eoi_toks"], m["refusal_toks"])
    got = R.filter_data(mb, prompts["harmful_train"], prompts["harmless_train"], prompts["harmful_val"],
                        prompts["harmless_val"], 4)
    for s, want in zip(("harmful_train", "harmless_train", "harmful_val", "harmless_val"), got[:4]):
        assert [prompts[s][i] for i in kept[s]] == want
        assert rec["counts"][s] == {"n": len(prompts[s]), "kept": len(want)}
    assert all(isinstance(v, int) and v >= 0 for v in rec["flips_batch1"].values())
    assert all(rec["counts"][s]["kept"] > 0 for s in refusal.SPLITS)  # the balanced fixture keeps prompts on both sides
    assert sum(c for _, c in rec["top_first_tokens"]) == pytest.approx(1.0)
    assert no_prompt_text(strings_in(rec), [x for s in prompts.values() for x in s])
    assert_no_hooks(model)


# which splits a one-sided fixture empties: no harmful prompt kept also cuts harmless train to nothing
EMPTIED = {"harmful": ["harmful_train", "harmful_val", "harmless_train"], "harmless": ["harmless_train", "harmless_val"]}


@pytest.mark.parametrize("empty", ["harmful", "harmless"])
def test_filter_stage_stops_on_a_split_that_kept_nothing_before_the_batch_1_flips(tiny_mc, monkeypatch, empty):
    """Ruling (1 Nov): a split with no kept prompt raises a CheckError naming each such split with its counts (no
    prompt text), right after filter_data, before the batch-1 flips."""
    model, tok = tiny_mc
    m = one_sided_entry(tok, model, empty)
    prompts = {"harmful_train": phrases(12, 3), "harmful_val": phrases(8, 4), "harmless_train": phrases(12, 5),
               "harmless_val": phrases(8, 6)}
    batch1, real = [], R.get_refusal_scores

    def spy(*a, **k):
        if k.get("batch_size") == 1:
            batch1.append(1)
        return real(*a, **k)

    monkeypatch.setattr(R, "get_refusal_scores", spy)
    with pytest.raises(refusal.CheckError) as err:
        refusal.filter_stage(model, tok, m, prompts, batch=4, log=quiet)
    assert isinstance(err.value, refusal.EmptyKept) and batch1 == []
    msg, rec = str(err.value), err.value.args[1]
    assert rec["empty"] == EMPTIED[empty]
    for s in refusal.SPLITS:
        assert rec["counts"][s]["n"] == len(prompts[s])
        assert (rec["counts"][s]["kept"] == 0) == (s in EMPTIED[empty])
        assert (f"{s} (n {len(prompts[s])}, kept 0" in msg) == (s in EMPTIED[empty])
    if empty == "harmful":  # harmless train kept prompts, then was cut to the harmful count
        assert "harmless_train (n 12, kept 0, 12 before the cut to the harmful count)" in msg
    assert msg.endswith("this model's run stops before the extraction")
    assert no_prompt_text(msg + strings_in(rec), [x for s in prompts.values() for x in s])
    assert_no_hooks(model)


@pytest.mark.parametrize("empty", ["harmful", "harmless"])
def test_a_run_whose_filtering_kept_nothing_stops_before_the_extraction(tiny_mc, tmp_path, monkeypatch, empty):
    """The model's run returns its stopped record (numbers only) and never calls the extraction or the selection."""
    from anatta import select

    model, tok = tiny_mc
    files = write_data(tmp_path / "data")
    plan = make_plan(config(), files, {"qwen": one_sided_entry(tok, model, empty)})

    def boom(*a, **k):
        raise AssertionError("the extraction or the selection ran")

    for mod, name in ((contrast, "mean_diffs"), (R, "generate_directions"), (refusal, "c3_extract"),
                      (select, "score_candidates"), (R, "select_direction")):
        monkeypatch.setattr(mod, name, boom)
    res = refusal.run_model(plan, "qwen", tmp_path, model, tok, PKG, log=quiet)
    assert res["stopped"] == refusal.EMPTY_KEPT and res["filter"]["empty"] == EMPTIED[empty]
    assert res["message"].startswith("filtering kept no prompts in ")
    assert json.loads((tmp_path / "qwen" / "result.json").read_text()) == res
    assert not (tmp_path / "qwen" / "kept.json").exists() and not any((tmp_path / "qwen" / "tensors").iterdir())
    every = [x for s in refusal.load_prompts(plan, tmp_path / "data").values() for x in s]
    assert no_prompt_text(strings_in(res), every)
    assert_no_hooks(model)


def test_c3_extract_on_the_fixture(tiny):
    model, tok = tiny
    eoi = R.eoi_toks_from_template(tok)
    mb = R.model_base_for(model, tok, eoi, list(range(10, len(tok) // 2)))
    positions = list(range(-len(eoi), 0))
    n = len(get_layers(model))
    hf, hl = phrases(6, 1), phrases(7, 2)
    lib = contrast.mean_diffs(model, tok, hf, hl, list(range(n)), positions, site="input")
    padded = R.generate_directions(mb, hf, hl, batch_size=4).cpu()
    rec, b1 = refusal.c3_extract(model, tok, mb, lib, padded, hf, hl, positions, log=quiet)
    assert rec["bitwise_premise"] is True and rec["premise_unequal_layers"] == [] and rec["premise_max_abs_diff"] == 0
    assert rec["n_compared"] == len(positions) * (n - 1) == len(rec["table"])
    assert rec["min_cos"] > 0.999999 and rec["max_rel_norm"] < 1e-9 and rec["max_rel_err"] < 1e-9
    assert rec["padded_min_cos"] > 0.999
    assert rec["layer0_zero_library"] and rec["layer0_max_abs_reference"] < 1e-12
    assert b1.shape == lib.shape
    assert_no_hooks(model)


# ---------------------------------------------------------------------------------------------- generation


def test_recorded_generate_pins_the_penalty_checks_eos_and_puts_both_back(tiny):
    model, tok = tiny
    gc = model.generation_config
    before = gc.repetition_penalty
    eos = [tok.eos_token_id]
    batch = chat_batch(tok, phrases(2, 7))
    with refusal.recorded_generate(model, 1.05, eos) as calls:
        assert model.generation_config.repetition_penalty == 1.05
        model.generate(**batch, max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id)
    assert len(calls) == 1 and calls[0]["width"] == batch.input_ids.shape[1]
    assert calls[0]["kwargs"] == {"max_new_tokens": 3, "do_sample": False, "pad_token_id": tok.pad_token_id}
    assert gc.repetition_penalty == before and "generate" not in vars(model)
    with pytest.raises(refusal.CheckError, match="repetition_penalty"):
        with refusal.recorded_generate(model, 1.0, eos):
            model.generation_config.repetition_penalty = 1.2
            model.generate(**batch, max_new_tokens=2, do_sample=False, pad_token_id=tok.pad_token_id)
    assert gc.repetition_penalty == before and "generate" not in vars(model)
    with pytest.raises(refusal.CheckError, match="eos"):
        with refusal.recorded_generate(model, 1.0, [tok.eos_token_id + 1]):
            model.generate(**batch, max_new_tokens=2, do_sample=False, pad_token_id=tok.pad_token_id)


def test_the_pinned_penalty_reaches_probe_generate(tiny):
    """probe.generate passes no repetition_penalty, so generate takes it from model.generation_config: pinning it
    there changes greedy completions."""
    model, tok = tiny
    msgs = [as_messages(p) for p in phrases(4, 8)]
    eos = [tok.eos_token_id]
    with refusal.recorded_generate(model, 1.0, eos):
        plain = generate(model, tok, msgs, max_new_tokens=12, temperature=0.0)
    with refusal.recorded_generate(model, 1.0, eos):
        again = generate(model, tok, msgs, max_new_tokens=12, temperature=0.0)
    with refusal.recorded_generate(model, 10.0, eos):
        penalized = generate(model, tok, msgs, max_new_tokens=12, temperature=0.0)
    assert plain == again and plain != penalized


def test_new_token_facts_counts_to_the_first_eos_and_flags_cuts():
    out = torch.tensor([[9, 9, 5, 2, 0, 0], [9, 9, 5, 6, 7, 8], [9, 9, 2, 2, 2, 2]])
    counts, cut = refusal.new_token_facts([{"width": 2, "out": out}], eos_ids=[2], max_new=4)
    assert counts == [2, 4, 1] and cut == [False, True, False]


def test_generate_jbb_returns_stripped_completions_and_numbers(tiny):
    model, tok = tiny
    prompts = phrases(5, 9)
    comps, facts = refusal.generate_jbb(model, tok, prompts, batch=2, max_new=4, rp=1.0, eos_ids=[tok.eos_token_id],
                                        substrings=["FOX"])
    assert len(comps) == facts["n"] == 5 and facts["batches"] == 3
    assert all(c == c.strip() for c in comps)
    assert facts["completion_sha256"] == [refusal.sha256_text(c) for c in comps]
    assert facts["passed"]["max_new_tokens"] == 4 and facts["passed"]["do_sample"] is False
    assert facts["refusal_substring"] == [int("fox" in c.lower()) for c in comps]
    assert all(1 <= k <= 4 for k in facts["new_tokens"]) and facts["repetition_penalty"] == 1.0
    assert no_prompt_text(strings_in(facts), prompts + [c for c in comps if c])


def test_mmlu_golden_spreads_its_items_over_the_prompt_lengths(tiny_mc):
    from anatta.bench.mc import encode, predict

    model, tok = tiny_mc
    m = model_entry(tok, model)
    rows = mmlu_items(12, seed=4)
    lengths = [len(t) for t in encode(tok, refusal.mmlu_texts(rows))]
    g = refusal.mmlu_golden(model, tok, rows, m["letters"], n=5, batch=4)
    assert g["items"] == 5 and g["disagreements"] == 0
    assert min(g["lengths"]) == min(lengths) and max(g["lengths"]) == max(lengths)
    r = refusal.run_mmlu(model, tok, rows, m["letters"], batch=4)
    assert r["n"] == 12 and len(r["bits"]) == 12 and r["correct"] == r["bits"].count("1")
    one = predict(model, tok, refusal.mmlu_texts(rows), m["letters"], batch_size=1)["preds"]
    assert r["bits"] == "".join(str(int(int(p) == row["answer"])) for p, row in zip(one, rows))


ITEM_KEYS = {"index", "length", "logits_b1", "logits_batched", "margin_b1", "max_abs_diff", "pred_b1", "pred_batched",
             "differs", "step", "margin_steps", "diff_steps", "differential", "max_differential_steps", "margin_change",
             "margin_change_steps"}


def test_the_golden_test_records_every_item(tiny_mc):
    """The coordinator's diagnosis request (1 Nov): every item's index into the rows, token length, four letter logits
    at both batch sizes, batch-1 margin, largest logit difference, predictions and whether they differ, recorded, not
    graded. Numbers only."""
    from anatta.bench.mc import encode, predict

    model, tok = tiny_mc
    m = model_entry(tok, model)
    rows = mmlu_items(12, seed=4)
    lengths = [len(t) for t in encode(tok, refusal.mmlu_texts(rows))]
    g = refusal.mmlu_golden(model, tok, rows, m["letters"], n=5, batch=4)
    assert g["batch"] == 4 and len(g["per_item"]) == g["items"] == 5
    one = predict(model, tok, [refusal.mmlu_texts(rows)[it["index"]] for it in g["per_item"]], m["letters"],
                  batch_size=1)
    for it, b1, p1 in zip(g["per_item"], one["logits"], one["preds"]):
        assert set(it) == ITEM_KEYS  # no batch composition: its predictions agree
        assert it["length"] == lengths[it["index"]] and it["logits_b1"] == b1 and it["pred_b1"] == p1
        assert len(it["logits_batched"]) == 4 and all(isinstance(x, float) for x in it["logits_b1"] + it["logits_batched"])
        top = sorted(it["logits_b1"], reverse=True)
        assert it["margin_b1"] == top[0] - top[1] >= 0
        assert it["max_abs_diff"] == max(abs(x - y) for x, y in zip(it["logits_b1"], it["logits_batched"]))
        assert it["differs"] is False and it["pred_batched"] == it["pred_b1"]
        # in bf16 steps at the item's largest batch-1 letter logit; the differential is the difference less its mean
        assert it["step"] == refusal.bf16_step(max(abs(x) for x in it["logits_b1"]))
        assert it["margin_steps"] == it["margin_b1"] / it["step"]
        assert it["diff_steps"] == it["max_abs_diff"] / it["step"]
        d = [y - x for x, y in zip(it["logits_b1"], it["logits_batched"])]
        assert it["differential"] == pytest.approx([x - sum(d) / 4 for x in d], abs=1e-12)
        assert it["max_differential_steps"] == max(abs(x) for x in it["differential"]) / it["step"]
        i1, i2 = sorted(range(4), key=lambda k: (-it["logits_b1"][k], k))[:2]
        assert it["margin_change"] == pytest.approx(d[i1] - d[i2], abs=1e-12)
        assert it["margin_change_steps"] == it["margin_change"] / it["step"]
    assert [it["index"] for it in g["per_item"]] == sorted(it["index"] for it in g["per_item"])
    assert g["lengths"] == [it["length"] for it in g["per_item"]] and g["disagreements"] == 0
    assert g["max_abs_diff"] == max(it["max_abs_diff"] for it in g["per_item"]) < 1e-4  # float32 numerics only
    assert g["min_margin_b1"] == min(it["margin_b1"] for it in g["per_item"])
    big = max(g["per_item"], key=lambda it: it["diff_steps"])
    assert g["largest_diff_item"] == {k: big[k] for k in ("index", "length", "max_abs_diff", "diff_steps")}
    assert g["max_differential_steps"] == max(it["max_differential_steps"] for it in g["per_item"])
    assert json.loads(json.dumps(g)) == g


def read_last_column(mask):
    """A planted padding bug: every row read at the batch's last column (index -1), which under right padding is a
    pad for each row shorter than the longest; at batch 1 there is no pad, so batch 1 reads the right token."""
    return torch.full((mask.shape[0],), mask.shape[1] - 1, dtype=torch.long)


class TokenTable(torch.nn.Module):
    """A stand-in model whose logits at each position are its input token's row of a fixed table, so a row's
    prediction depends on the token it is read at alone: every MMLU prompt's last token (it ends "Answer:") says A
    by a margin of 3, the pad says C by 3. Its batch-1 and batched logits are identical unless a row is misread."""

    def __init__(self, vocab: int, letters: list[int], last_ids: set[int], pad_id: int):
        super().__init__()
        table = torch.zeros(vocab, vocab)
        for t in last_ids:
            table[t, letters] = torch.tensor([3.0, 0.0, 0.0, 0.0])
        table[pad_id, letters] = torch.tensor([0.0, 0.0, 3.0, 0.0])
        self.table = torch.nn.Parameter(table, requires_grad=False)

    def forward(self, input_ids, attention_mask=None):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.table[input_ids])


def test_a_planted_padding_bug_shows_large_margin_disagreements_with_their_batches(tiny_mc, monkeypatch):
    """Under a padding bug, padded rows flip with a batch-1 margin and a logit difference of 3, where numerics alone
    give none (the clean run's difference is 0): the record tells a bug from a near-tie. Each flip names its batch:
    its index, its place and the lengths of its rows; the batch's longest row, which has no pad, does not flip."""
    from anatta.bench import mc

    _, tok = tiny_mc
    letters = mc.letter_ids(tok)
    rows = mmlu_items(40, seed=11)
    encoded = mc.encode(tok, refusal.mmlu_texts(rows))
    model = TokenTable(len(tok), letters, {r[-1] for r in encoded}, tok.pad_token_id)
    gate = config()["mmlu"]["golden_gate"]
    clean = refusal.mmlu_golden(model, tok, rows, letters, n=16, batch=4)
    assert clean["disagreements"] == 0 and clean["max_abs_diff"] == 0.0 and clean["min_margin_b1"] == 3.0
    assert refusal.golden_gate(clean, gate)["stop"] is False
    monkeypatch.setattr(mc, "last_real_index", read_last_column)
    g = refusal.mmlu_golden(model, tok, rows, letters, n=16, batch=4)
    flips = [it for it in g["per_item"] if it["differs"]]
    assert g["disagreements"] == len(flips) >= 1 and g["max_abs_diff"] == 3.0
    # one step at the largest letter logit, 3.0, is 2^-6: the bug's difference and margin are 192 steps, far past
    # the gate's 8 and 2, where the smoke's numerics flips were 0 and 1 step from a tie
    v = refusal.golden_gate(g, gate)
    assert v["stop"] and v["a"]["items"] == v["b"]["items"] == [it["index"] for it in flips]
    assert g["largest_diff_item"]["diff_steps"] == 192.0 and g["max_differential_steps"] == 192.0
    for j, it in enumerate(g["per_item"]):
        b, place = divmod(j, 4)
        lens = g["lengths"][4 * b:4 * b + 4]
        assert it["differs"] == (it["length"] < max(lens))  # exactly the padded rows flip
        assert it["logits_b1"] == clean["per_item"][j]["logits_b1"]  # batch 1 is untouched
        if it["differs"]:
            assert set(it) == ITEM_KEYS | {"batch_index", "place", "batch_lengths"}
            assert (it["batch_index"], it["place"], it["batch_lengths"]) == (b, place, lens)
            assert (it["pred_b1"], it["pred_batched"], it["margin_b1"], it["max_abs_diff"]) == (0, 2, 3.0, 3.0)
            assert (it["margin_steps"], it["diff_steps"], it["margin_change_steps"]) == (192.0, 192.0, -192.0)
            assert it["differential"] == [-3.0, 0.0, 3.0, 0.0]


# The gate on planted records (refusal.golden_item, golden_record, golden_gate). Every item's batch-1 letter logits
# top out in [8, 16), where one bf16 step is 2^-4 = 0.0625.
STEP = 0.0625
GATE = {"max_diff_steps": 8, "flip_margin_steps": 2, "max_flip_share": 0.10}


def planted(changes: dict, n: int = 64, batch: int = 32) -> dict:
    """A golden test record of n items, each [12, 10, 9, 8] at both batch sizes (a 2.0 margin, 32 steps) unless
    `changes` gives its (batch-1, batched) logits."""
    items = []
    for i in range(n):
        lb, la = changes.get(i, ([12.0, 10.0, 9.0, 8.0], [12.0, 10.0, 9.0, 8.0]))
        items.append(refusal.golden_item(i, 50 + i, lb, la))
    return refusal.golden_record(items, batch)


TIE_FLIP = ([12.0, 12.0, 9.0, 8.0], [12.0, 12.0 + STEP, 9.0, 8.0])  # an exact tie, broken by one step the other way


def test_bf16_step_is_the_spacing_at_the_logit_and_guards_zero():
    assert refusal.bf16_step(16.0) == 0.125 and refusal.bf16_step(15.99) == 0.0625
    assert refusal.bf16_step(-20.0) == 0.125 and refusal.bf16_step(33.0) == 0.25 and refusal.bf16_step(1.0) == 2 ** -7
    assert refusal.bf16_step(0.0) == 2 ** -133 == refusal.bf16_step(1e-45)  # no log2(0); subnormals share a spacing
    assert refusal.bf16_step(float("nan")) == 2 ** -133 and refusal.bf16_step(float("inf")) == float("inf")


def test_the_gate_stops_on_one_item_nine_steps_apart_and_passes_eight():
    """Part (a): a difference above 8 steps on any item, even with no flip, stops the run."""
    for steps, stop in ((9, True), (8, False)):
        g = planted({5: ([12.0, 10.0, 9.0, 8.0], [12.0 + steps * STEP, 10.0, 9.0, 8.0])})
        v = refusal.golden_gate(g, GATE)
        assert g["disagreements"] == 0 and g["largest_diff_item"] == {"index": 5, "length": 55,
                                                                      "max_abs_diff": steps * STEP, "diff_steps": steps}
        assert v["stop"] is stop and v["a"] == {"fired": stop, "items": [5] if stop else []}
        assert not v["b"]["fired"] and not v["c"]["fired"]
        assert g["max_differential_steps"] == 0.75 * steps  # the shift less its mean over the four letters


def test_the_gate_stops_on_a_flip_with_a_three_step_margin():
    """Part (b): a flip whose batch-1 margin was 3 steps, with only 2 steps of difference, stops the run."""
    g = planted({7: ([12.0, 12.0 - 3 * STEP, 9.0, 8.0], [12.0 - 2 * STEP, 12.0 - STEP, 9.0, 8.0])})
    v = refusal.golden_gate(g, GATE)
    it = g["per_item"][7]
    assert it["differs"] and (it["margin_steps"], it["diff_steps"], it["margin_change_steps"]) == (3.0, 2.0, -4.0)
    assert v["stop"] and v["b"] == {"fired": True, "items": [7]} and not v["a"]["fired"] and not v["c"]["fired"]
    assert (it["batch_index"], it["place"], it["batch_lengths"]) == (0, 7, list(range(50, 82)))


def test_the_gate_stops_on_seven_flips_of_64_and_passes_six():
    """Part (c): more than 10% of the items differing stops the run, though each flip is a one-step tie flip; on a
    smoke run's 50 items, 6 stop it and 5, exactly 10%, do not."""
    for n, k, stop in ((64, 7, True), (64, 6, False), (50, 6, True), (50, 5, False)):
        g = planted({i: TIE_FLIP for i in range(0, 8 * k, 8)}, n=n)
        v = refusal.golden_gate(g, GATE)
        assert g["disagreements"] == k and v["stop"] is stop
        assert v["c"] == {"fired": stop, "differing": k, "items": n} and not v["a"]["fired"] and not v["b"]["fired"]


def test_a_one_step_tie_flip_passes_the_gate_and_is_recorded():
    """The smoke's kind of flip (an exact tie, or a one-step swap) is numerics: recorded, and the run goes on."""
    g = planted({21: TIE_FLIP, 30: ([12.0, 12.0 - STEP, 9.0, 8.0], [12.0 - STEP, 12.0, 9.0, 8.0])})
    v = refusal.golden_gate(g, GATE)
    assert g["disagreements"] == 2 and v["stop"] is False
    tie, swap = g["per_item"][21], g["per_item"][30]
    assert (tie["pred_b1"], tie["pred_batched"], tie["margin_steps"], tie["margin_change_steps"]) == (0, 1, 0.0, -1.0)
    assert (swap["margin_steps"], swap["diff_steps"], swap["margin_change_steps"]) == (1.0, 1.0, -2.0)
    assert tie["differential"] == [-STEP / 4, 3 * STEP / 4, -STEP / 4, -STEP / 4]


def test_the_gate_reads_a_nan_as_past_its_bound_and_an_all_zero_item_as_no_difference():
    g = planted({3: ([12.0, 10.0, 9.0, 8.0], [float("nan"), 10.0, 9.0, 8.0]),
                 4: ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])})
    v = refusal.golden_gate(g, GATE)
    assert v["stop"] and v["a"]["items"] == [3]
    assert (g["per_item"][4]["step"], g["per_item"][4]["diff_steps"]) == (2 ** -133, 0.0)


def test_the_gate_message_names_each_part_that_fired():
    g = planted({5: ([12.0, 10.0, 9.0, 8.0], [12.0 + 9 * STEP, 10.0, 9.0, 8.0]),
                 **{i: TIE_FLIP for i in range(10, 17)}})
    g["gate"] = refusal.golden_gate(g, GATE)
    assert refusal.gate_message(g, 32) == ("the MMLU golden test (batch 32 against batch 1) stopped the run: (a) items "
                                           "[5] differ by more than 8 bf16 steps; (c) 7 of 64 items differ, more than "
                                           "10%")


def test_the_mmlu_floor_stops_a_full_run_below_it_and_only_records_a_smoke_runs():
    full = {"smoke": False, "mmlu": {"floor_pct": 35.0}}
    smoke = {"smoke": True, "mmlu": {"floor_pct": 35.0}}
    rec = {"name": "Q0", "mmlu": {"n": 200, "correct": 69, "acc": 34.5}}
    assert refusal.mmlu_floor(rec, full) == {"condition": "Q0", "n": 200, "correct": 69, "acc": 34.5,
                                             "floor_pct": 35.0, "below": True, "applies": True, "stop": True}
    assert refusal.mmlu_floor(rec, smoke) == {**refusal.mmlu_floor(rec, full), "applies": False, "stop": False}
    held = refusal.mmlu_floor({"name": "M0", "mmlu": {"n": 200, "correct": 70, "acc": 35.0}}, full)
    assert held["below"] is False and held["stop"] is False  # at the floor holds; below it stops


# ----------------------------------------------------------------------------------------------- conditions

QWEN = ["Q0", "QA", "QA+", "QR", "QB", "QB2", "QC", "QS", "QL", "QA-rand", "QB-rand", "Q0-rp", "QA-rp", "Q0-chat",
        "QA-chat", "Q0-end"]
MINISTRAL = ["M0", "MA", "MB", "MB2", "ML", "MB-rand"]


def vectors(model, layer: int = 2) -> dict:
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    r = torch.randn(d, generator=g, dtype=torch.float64) * 3
    return {"r": r, "rhat": refusal.unit(r), "layer": layer, "ql": torch.randn(d, generator=g, dtype=torch.float64),
            "modeb": {b: torch.randn(d, generator=g, dtype=torch.float64) for b in (1, 2)},
            "r_ref": r * 1.01, "rhat_ref": refusal.unit(r), "choice": {"pl": [-1, layer]},
            "choice_ref": {"pl": [-1, layer]}, "ladder": False, "ladder_ref": False}


def logits(model, batch, hooks=None):
    with torch.no_grad():
        if hooks is None:
            return model(**batch).logits
        with hooks():
            return model(**batch).logits


def test_the_registered_conditions_in_run_order_with_their_runs(tiny):
    model, tok = tiny
    m = model_entry(tok, model)
    conds = refusal.conditions_for("qwen", m, vectors(model), model, tok)
    assert [c["name"] for c in conds] == QWEN and all(c["skip"] is None for c in conds)
    runs = {c["name"]: c["runs"] for c in conds}
    both = {"Q0", "QA", "QA+", "QR", "QB", "QC", "QB-rand"}
    assert {k for k, v in runs.items() if v == ("jbb", "mmlu")} == both
    assert {k for k, v in runs.items() if v == ("mmlu_chat",)} == {"Q0-chat", "QA-chat"}
    assert {c["name"] for c in conds if c["rp"] == "lab"} == {"Q0-rp", "QA-rp"}
    mconds = refusal.conditions_for("ministral", m, vectors(model), model, tok)
    assert [c["name"] for c in mconds] == MINISTRAL
    assert {c["name"] for c in mconds if "mmlu" in c["runs"]} == {"M0", "MA", "MB", "MB-rand"}


def test_qx_runs_only_when_the_two_paths_choose_differently(tiny):
    model, tok = tiny
    m = model_entry(tok, model)
    v = vectors(model)
    v["choice_ref"] = {"pl": [-2, 1]}
    names = [c["name"] for c in refusal.conditions_for("qwen", m, v, model, tok)]
    assert names.index("QX") == names.index("QR") + 1
    v = vectors(model)
    del v["r_ref"]
    conds = {c["name"]: c for c in refusal.conditions_for("qwen", m, v, model, tok)}
    assert "QX" not in conds and conds["QR"]["skip"] == "no reference candidate"


def test_without_vectors_the_conditions_needing_them_are_skipped_with_a_reason(tiny):
    model, tok = tiny
    m = model_entry(tok, model)
    conds = refusal.conditions_for("qwen", m, {"choice": None, "choice_ref": None}, model, tok)
    skipped = {c["name"] for c in conds if c["skip"]}
    assert skipped == set(QWEN) - {"Q0", "Q0-rp", "Q0-chat", "Q0-end"}


@pytest.mark.parametrize("layer", [0, 1, 3])
def test_qs_subtracts_r_where_arditis_activation_addition_does(tiny, layer):
    """S: the raw r subtracted at block input l*, through the dial's steer at block output l* - 1 (or block 0's
    input site), against Arditi's addition hook with coefficient -1 at block l*."""
    model, tok = tiny
    m = model_entry(tok, model)
    v = vectors(model, layer)
    qs = next(c for c in refusal.conditions_for("qwen", m, v, model, tok) if c["name"] == "QS")
    batch = chat_batch(tok, phrases(3, 11))
    mb = R.model_base_for(model, tok, m["eoi_toks"], m["refusal_toks"])
    pre = [(mb.model_block_modules[layer], R.get_activation_addition_input_pre_hook(vector=v["r"].float(),
                                                                                    coeff=torch.tensor(-1.0)))]
    want = logits(model, batch, lambda: R.add_hooks(pre, []))
    got = logits(model, batch, qs["hooks"])
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(got, logits(model, batch), atol=1e-3)
    assert_no_hooks(model)


def test_qa_plus_matches_arditis_hook_set_and_qa_leaves_only_the_mlp_gap(tiny):
    model, tok = tiny
    m = model_entry(tok, model)
    v = vectors(model)
    conds = {c["name"]: c for c in refusal.conditions_for("qwen", m, v, model, tok)}
    batch = chat_batch(tok, phrases(3, 12))
    mb = R.model_base_for(model, tok, m["eoi_toks"], m["refusal_toks"])
    pre, fwd = R.get_all_direction_ablation_hooks(mb, v["rhat"].float())
    want = logits(model, batch, lambda: R.add_hooks(pre, fwd))
    assert torch.allclose(logits(model, batch, conds["QA+"]["hooks"]), want, rtol=1e-4, atol=1e-4)
    assert not torch.equal(logits(model, batch, conds["QA"]["hooks"]), want)
    qr = conds["QR"]
    assert qr["hooks"] is None and torch.allclose(logits(model, batch, qr["hooks_factory"]()), want, atol=1e-4)
    assert_no_hooks(model)


def test_mode_b_and_its_random_control_ablate_each_layers_own_vector_at_its_block_output(tiny, monkeypatch):
    import anatta.dial as dial

    model, tok = tiny
    m = model_entry(tok, model)
    v = vectors(model)
    seen = {}
    orig = dial.intervene

    def spy(model, ivs):
        seen["ivs"] = ivs
        return orig(model, ivs)

    monkeypatch.setattr(dial, "intervene", spy)
    conds = {c["name"]: c for c in refusal.conditions_for("qwen", m, v, model, tok)}
    got = {}
    for name in ("QB", "QB-rand", "QB2"):
        with conds[name]["hooks"]():
            pass
        got[name] = seen["ivs"]
    assert [(iv.layer, iv.site, iv.mode) for iv in got["QB"]] == [(1, "output", "ablate"), (2, "output", "ablate")]
    assert all(torch.allclose(iv.direction, refusal.unit(v["modeb"][iv.layer]).float()) for iv in got["QB"])
    assert all(torch.allclose(iv.direction, dial.random_like(v["modeb"][iv.layer], seed=iv.layer))
               for iv in got["QB-rand"])
    assert [iv.layer for iv in got["QB2"]] == [1]
    assert_no_hooks(model)


# ------------------------------------------------------------------------------------- the hook check, a run


def condition_plan():
    return {"generation": {"max_new_tokens": 3, "repetition_penalty": 1.0, "repetition_penalty_lab": 1.05},
            "mmlu": {"batch": 4}, "aggregation": {"refusal_substrings": ["sorry"]}}


def test_run_condition_runs_with_its_hooks_and_leaves_none(tiny, tmp_path):
    model, tok = tiny
    m = model_entry(tok, model)
    r = vectors(model)["r"]
    cond = {"name": "QA", "hooks": lambda: intervene(model, everywhere(model, r)), "runs": ("jbb",), "rp": "graded",
            "ladder": False}
    rec = refusal.run_condition(model, tok, cond, m, condition_plan(), {"jbb": phrases(3, 12)}, [], tmp_path, quiet)
    assert rec["jbb"]["n"] == 3 and len(json.loads((tmp_path / "completions" / "QA.json").read_text())) == 3
    assert_no_hooks(model)


def test_run_condition_has_its_hooks_on_for_every_pass_it_runs(tiny_mc, tmp_path, monkeypatch):
    """The condition's hooks are installed while its JBB generation and its MMLU pass run, not only around them."""
    model, tok = tiny_mc
    m = model_entry(tok, model)
    calls = {"n": 0}

    @contextmanager
    def counting():
        h = get_layers(model)[0].register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
        try:
            yield
        finally:
            h.remove()

    marks = []

    def marked(what, fn):
        def wrapper(*a, **k):
            before = calls["n"]
            out = fn(*a, **k)
            marks.append((what, calls["n"] - before))
            return out
        return wrapper

    monkeypatch.setattr(refusal, "generate_jbb", marked("jbb", refusal.generate_jbb))
    monkeypatch.setattr(refusal, "run_mmlu", marked("mmlu", refusal.run_mmlu))
    cond = {"name": "QZ", "hooks": counting, "runs": ("jbb", "mmlu"), "rp": "graded", "ladder": False}
    refusal.run_condition(model, tok, cond, m, condition_plan(), {"jbb": phrases(3, 14)}, mmlu_items(4), tmp_path,
                          quiet)
    assert [w for w, _ in marks] == ["jbb", "mmlu"] and all(d > 0 for _, d in marks), marks
    assert_no_hooks(model)


def test_the_hook_check_stops_a_condition_when_a_hook_is_left_installed(tiny, tmp_path):
    model, tok = tiny
    model = copy.deepcopy(model)
    m = model_entry(tok, model)
    r = vectors(model)["r"]

    @contextmanager
    def leaky():  # installs and never removes
        install(model, everywhere(model, r))
        yield

    cond = {"name": "QA", "hooks": leaky, "runs": ("jbb",), "rp": "graded", "ladder": False}
    with pytest.raises(RuntimeError, match="block 0"):
        refusal.run_condition(model, tok, cond, m, condition_plan(), {"jbb": phrases(2, 13)}, [], tmp_path, quiet)
    clean = copy.deepcopy(tiny[0])
    h = get_layers(clean)[1].register_forward_hook(lambda *a: None)
    q0 = {"name": "Q0", "hooks": None, "runs": ("jbb",), "rp": "graded", "ladder": False}
    with pytest.raises(RuntimeError, match="block 1"):
        refusal.run_condition(clean, tok, q0, m, condition_plan(), {"jbb": phrases(2, 13)}, [], tmp_path, quiet)
    h.remove()
    assert refusal.run_condition(clean, tok, q0, m, condition_plan(), {"jbb": phrases(2, 13)}, [], tmp_path,
                                 quiet)["jbb"]["n"] == 2


def benign_inhouse(mp):
    mp.setattr(contrast, "HARMFUL", phrases(8, seed=30))
    mp.setattr(contrast, "HARMLESS", phrases(8, seed=31))


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, tiny_mc):
    """A whole Qwen-shaped smoke run on the tiny model: the golden test, filtering, both paths, C3-extract and every
    condition."""
    model, tok = tiny_mc
    out = tmp_path_factory.mktemp("c3_smoke")
    files = write_data(out / "data")
    plan = make_plan(config(), files, {"qwen": model_entry(tok, model)}, smoke=True)
    with pytest.MonkeyPatch.context() as mp:
        benign_inhouse(mp)
        res = refusal.run_model(plan, "qwen", out, model, tok, PKG, log=quiet)
    return out, plan, res


def test_a_smoke_run_runs_every_condition_and_returns_numbers_only(smoke_run, tiny_mc):
    out, plan, res = smoke_run
    model, tok = tiny_mc
    assert res["select"]["choice"]["found"] and res["extract"]["c3_extract"]["bitwise_premise"]
    g = res["mmlu_golden"]
    assert g["disagreements"] == 0 and g["gate"]["stop"] is False
    assert g["gate"]["thresholds"] == config()["mmlu"]["golden_gate"]
    assert res["mmlu_floor"] == {"condition": "Q0", "n": 6, "correct": res["conditions"]["Q0"]["mmlu"]["correct"],
                                 "acc": res["conditions"]["Q0"]["mmlu"]["acc"], "floor_pct": 0.0, "below": False,
                                 "applies": False, "stop": False}
    names = list(res["conditions"])
    assert [n for n in names if n != "QX"] == QWEN
    c, cr = res["select"]["choice"], res["select"]["choice_ref"]
    assert ("QX" in names) == ((c["pos"], c["layer"]) != (cr["pos"], cr["layer"]))
    for name, rec in res["conditions"].items():
        assert "skipped" not in rec, name
        if "jbb" in rec:
            assert rec["jbb"]["n"] == plan["smoke_settings"]["jbb"]
        if "mmlu" in rec:
            assert rec["mmlu"]["n"] == plan["smoke_settings"]["mmlu"]
    assert res["q0_end"] == {"identical": 4, "n": 4}
    assert res["conditions"]["Q0-rp"]["jbb"]["repetition_penalty"] == 1.05
    assert res["conditions"]["Q0"]["jbb"]["repetition_penalty"] == 1.0
    assert_no_hooks(model)
    tensors = {p.name for p in (out / "qwen" / "tensors").iterdir()}
    assert tensors == {"candidates_library.pt", "candidates_reference.pt", "candidates_reference_b1.pt", "vectors.pt",
                       "inhouse.pt"}
    prompts = refusal.load_prompts(plan, out / "data")
    every = [x for s in prompts.values() for x in s] + phrases(8, 30) + phrases(8, 31)
    for p in out.rglob("*.json"):
        if "data" in p.relative_to(out).parts or "completions" in p.relative_to(out).parts:
            continue
        assert no_prompt_text(p.read_text(), every), p
    comps = [c for f in (out / "qwen" / "completions").glob("*.json") for c in json.loads(f.read_text()) if " " in c]
    assert no_prompt_text((out / "qwen" / "result.json").read_text(), comps)


def test_mode_b_vectors_are_the_block_output_mean_differences_at_the_last_token(smoke_run, tiny_mc):
    out, plan, res = smoke_run
    model, tok = tiny_mc
    prompts = refusal.load_prompts(plan, out / "data")
    kept = json.loads((out / "qwen" / "kept.json").read_text())
    hf = [prompts["harmful_train"][i] for i in kept["harmful_train"]]
    hl = [prompts["harmless_train"][i] for i in kept["harmless_train"]]
    vec = torch.load(out / "qwen" / "tensors" / "vectors.pt")
    lo, hi = plan["models"]["qwen"]["band_b"]
    want = contrast.mean_diffs(model, tok, hf, hl, list(range(lo, hi + 1)), (-1,), site="output")[0]
    for i, b in enumerate(range(lo, hi + 1)):
        assert torch.allclose(vec["modeb"][b], want[i], rtol=0, atol=1e-10), b


def test_a_rerun_of_the_same_plan_reads_every_stage_back(smoke_run, tiny_mc, monkeypatch):
    out, plan, res = smoke_run
    model, tok = tiny_mc

    def boom(*a, **k):
        raise AssertionError("recomputed")

    from anatta import select

    monkeypatch.setattr(refusal, "filter_stage", boom)
    monkeypatch.setattr(refusal, "c3_extract", boom)
    monkeypatch.setattr(refusal, "run_condition", boom)
    monkeypatch.setattr(refusal, "mmlu_golden", boom)
    monkeypatch.setattr(select, "score_candidates", boom)
    monkeypatch.setattr(R, "select_direction", boom)
    benign_inhouse(monkeypatch)
    again = refusal.run_model(plan, "qwen", out, model, tok, PKG, log=quiet)
    for k in ("conditions", "select", "extract", "filter", "mmlu_golden", "mmlu_floor", "q0_end", "geometry"):
        assert again[k] == res[k], k


def test_the_golden_test_stops_the_run_first_when_its_gate_fires(tiny_mc, tmp_path, monkeypatch):
    """The golden test runs first, before filtering (re-registered 14:34 UTC 1 Nov), so a stop costs minutes; its
    stopped record, with the per-item record and the gate's verdict, comes back like a finished one (so main stops
    on it and the card shows it), and is the run's result.json and mmlu_golden.json on the volume. Here a planted
    batching bug gives batched rows their four letters' logits in the wrong order, so every item flips, far past the
    gate's bounds."""
    from anatta import select
    from anatta.bench import mc

    model, tok = tiny_mc
    files = write_data(tmp_path / "data")
    plan = make_plan(config(), files, {"qwen": model_entry(tok, model)}, smoke=True)
    real = mc.letter_logits
    monkeypatch.setattr(mc, "letter_logits", lambda model, ids, mask, letters: (
        real(model, ids, mask, letters)[:, [1, 2, 3, 0]] if ids.shape[0] > 1 else real(model, ids, mask, letters)))

    def boom(*a, **k):
        raise AssertionError("ran after the golden test stopped the run")

    for mod, name in ((refusal, "filter_stage"), (contrast, "mean_diffs"), (select, "score_candidates"),
                      (refusal, "run_condition")):
        monkeypatch.setattr(mod, name, boom)
    lines = []
    res = refusal.run_model(plan, "qwen", tmp_path, model, tok, PKG, log=lines.append)
    g = res["mmlu_golden"]
    assert set(res) == {"facts", "mmlu_golden", "stopped", "message"} and res["stopped"] == "mmlu_golden"
    assert g["disagreements"] == g["items"] == 6 and g["gate"]["stop"] and g["gate"]["c"]["fired"]
    assert res["message"] == refusal.gate_message(g, plan["mmlu"]["batch"])
    assert res["message"].startswith("the MMLU golden test (batch 4 against batch 1) stopped the run: ")
    assert len(g["per_item"]) == g["items"] and sum(it["differs"] for it in g["per_item"]) == g["disagreements"]
    assert json.loads((tmp_path / "qwen" / "result.json").read_text()) == res
    assert json.loads((tmp_path / "qwen" / "mmlu_golden.json").read_text()) == g
    assert {p.name for p in (tmp_path / "qwen").iterdir()} == {"mmlu_golden.json", "result.json", "tensors"}
    assert not any((tmp_path / "qwen" / "tensors").iterdir())
    assert lines[0].startswith("MMLU golden test: 6 of 6 items differ (batch 4 against 1); gate stop (")
    assert sum(line.startswith("MMLU golden test: item ") for line in lines) == g["disagreements"]
    assert_no_hooks(model)


def planted_pass(n: int = 64, batch: int = 4) -> dict:
    """A golden test record that passes the gate with one flip: an exact tie at batch 1, broken one step the other
    way batched (the smoke's kind of numerics flip), among n agreeing items."""
    flip = ([12.0, 12.0, 9.0, 8.0], [12.0, 12.0625, 9.0, 8.0])
    return refusal.golden_record([refusal.golden_item(i, 50 + i, *(flip if i == 9 else ([12.0, 10.0, 9.0, 8.0],) * 2))
                                  for i in range(n)], batch)


def test_a_tie_flip_passes_the_gate_and_the_run_goes_on(smoke_run, tiny_mc, tmp_path, monkeypatch):
    """A flip the gate reads as numerics is recorded, with the gate's verdict, and the run goes on to the conditions
    (read back here from the smoke run's folder)."""
    src, plan, ran = smoke_run
    model, tok = tiny_mc
    out = tmp_path / "copy"
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("mmlu_golden.json", "result.json"))
    monkeypatch.setattr(refusal, "mmlu_golden", lambda *a, **k: planted_pass())
    benign_inhouse(monkeypatch)
    lines = []
    res = refusal.run_model(plan, "qwen", out, model, tok, PKG, log=lines.append)
    g = res["mmlu_golden"]
    assert "stopped" not in res and g["disagreements"] == 1 and g["gate"]["stop"] is False
    assert res["conditions"] == ran["conditions"] and res["mmlu_floor"] == ran["mmlu_floor"]
    assert json.loads((out / "qwen" / "mmlu_golden.json").read_text()) == g
    flip = g["per_item"][9]
    assert flip["differs"] and (flip["margin_steps"], flip["diff_steps"], flip["margin_change_steps"]) == (0.0, 1.0,
                                                                                                          -1.0)
    assert ("MMLU golden test: 1 of 64 items differ (batch 4 against 1); gate passed; max |logit diff| 0.0625 (1.0 "
            "steps, item 9, length 59); largest differential 0.8 steps") in lines
    assert ("MMLU golden test: item 9 (length 59): batch-1 margin 0.0000 (0.0 steps), max |logit diff| 0.0625 (1.0 "
            "steps), margin change -1.0 steps, letter 0 at batch 1, 1 batched (batch 2, lengths 58-61)") in lines


def test_a_full_run_stops_after_its_first_condition_below_the_mmlu_floor(tiny_mc, tmp_path, monkeypatch):
    """The common-mode floor (registered with the gate, 14:34 UTC 1 Nov): a full run whose first condition scores
    MMLU below mmlu.floor_pct on every row stops there, and its record names the stop. The floor is set past 100%
    here, so the tiny model cannot reach it."""
    model, tok = tiny_mc
    files = write_data(tmp_path / "data")
    plan = make_plan(config(), files, {"qwen": model_entry(tok, model)})
    plan["mmlu"]["floor_pct"] = 100.1
    benign_inhouse(monkeypatch)
    lines = []
    res = refusal.run_model(plan, "qwen", tmp_path, model, tok, PKG, log=lines.append)
    q0 = res["conditions"]["Q0"]["mmlu"]
    assert res["stopped"] == "mmlu_floor" and list(res["conditions"]) == ["Q0"] and q0["n"] == 10
    assert res["mmlu_floor"] == {"condition": "Q0", "n": 10, "correct": q0["correct"], "acc": q0["acc"],
                                 "floor_pct": 100.1, "below": True, "applies": True, "stop": True}
    assert res["message"] == (f"the MMLU floor: Q0 scored {q0['acc']:.2f}% on 10 items, below 100.1%: no other "
                              "condition runs")
    assert json.loads((tmp_path / "qwen" / "result.json").read_text()) == res
    assert [p.name for p in (tmp_path / "qwen" / "conditions").iterdir()] == ["Q0.json"]
    assert f"MMLU floor: Q0 {q0['acc']:.2f}% on 10 items against 100.1%" in lines
    assert res["mmlu_golden"]["gate"]["stop"] is False and "select" in res
    assert_no_hooks(model)


def test_the_golden_test_runs_with_no_hook_on(smoke_run, tiny_mc, tmp_path, monkeypatch):
    """It measures the model's own batching: a hook left on refuses to run it, as before every condition."""
    src, plan, _ = smoke_run
    model, tok = tiny_mc
    out = tmp_path / "copy"
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("mmlu_golden.json", "result.json", "conditions",
                                                            "completions"))
    benign_inhouse(monkeypatch)
    handle = get_layers(model)[0].register_forward_hook(lambda mod, inp, outp: None)
    try:
        with pytest.raises(RuntimeError, match="hooks are installed"):
            refusal.run_model(plan, "qwen", out, model, tok, PKG, log=quiet)
    finally:
        handle.remove()
    assert not (out / "qwen" / "mmlu_golden.json").exists()


def failed_premise(*a, **k):
    """c3_extract stood in for: the premise failed at block 1, the comparison numbers made up."""
    return ({"bitwise_premise": False, "premise_unequal_layers": [1], "premise_max_abs_diff": 0.5, "n_compared": 6,
             "min_cos": 0.5, "max_rel_norm": 0.1, "max_rel_err": 0.2, "padded_min_cos": 0.5, "table": [],
             "padded": [], "layer0_zero_library": True, "layer0_max_abs_reference": 0.0}, torch.zeros(1))


def test_a_smoke_run_stops_when_the_bitwise_premise_fails(smoke_run, tiny_mc, tmp_path, monkeypatch):
    """Ruling (1 Nov): a smoke run whose premise failed stops after recording it, before the selection."""
    src, plan, _ = smoke_run
    model, tok = tiny_mc
    out = tmp_path / "copy"
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("c3_extract.json", "select_*.json", "mmlu_golden.json",
                                                            "result.json", "conditions", "completions"))
    monkeypatch.setattr(refusal, "c3_extract", failed_premise)
    benign_inhouse(monkeypatch)
    res = refusal.run_model(plan, "qwen", out, model, tok, PKG, log=quiet)
    assert res["message"] == "C3-extract's bitwise premise failed on the smoke run (layers [1]): fix it before launch"
    assert json.loads((out / "qwen" / "result.json").read_text()) == res
    assert res["stopped"] == "bitwise_premise" and res["extract"]["c3_extract"]["premise_unequal_layers"] == [1]
    assert not (out / "qwen" / "select_library.json").exists()


def test_a_full_run_records_a_failed_premise_and_goes_on(tiny_mc, tmp_path, monkeypatch):
    model, tok = tiny_mc
    files = write_data(tmp_path / "data")
    plan = make_plan(config(), files, {"qwen": model_entry(tok, model)})
    monkeypatch.setattr(refusal, "c3_extract", failed_premise)
    benign_inhouse(monkeypatch)
    res = refusal.run_model(plan, "qwen", tmp_path, model, tok, PKG, log=quiet)
    assert res["extract"]["c3_extract"]["bitwise_premise"] is False and "stopped" not in res
    assert [n for n in res["conditions"] if n != "QX"] == QWEN
    assert res["select"]["library"]["n_pos"] == 2 and "scored" in res["select"]["library"]
    assert_no_hooks(model)


def made_up_ego(path: Path, d: int, layers=(0, 1, 2)) -> Path:
    """A made-up v_ego file (anatta.probe.Direction) of random vectors."""
    from anatta.probe import Direction

    g = torch.Generator().manual_seed(5)
    Direction(name="ego", vectors={b: torch.randn(d, generator=g) for b in layers}, scores={b: 0.0 for b in layers},
              pos_mean={b: 0.0 for b in layers}, neg_mean={b: 0.0 for b in layers}).save(path)
    return path


def test_a_ministral_shaped_run_has_no_reference_path_and_reads_the_ego_ledger(tiny_mc, tmp_path, monkeypatch):
    model, tok = tiny_mc
    files = write_data(tmp_path / "data")
    m = model_entry(tok, model, reference=False, c3_extract=False)
    plan = make_plan(config(), files, {"ministral": m})
    plan["registered"]["expectations"]["e_82"]["layers"] = [1, 2]
    ego = made_up_ego(tmp_path / "ego.pt", model.config.hidden_size)
    plan["ego"]["sha256"] = refusal.sha256_file(ego)
    benign_inhouse(monkeypatch)
    res = refusal.run_model(plan, "ministral", tmp_path, model, tok, PKG, log=quiet, ego=ego)
    assert list(res["conditions"]) == MINISTRAL
    assert res["select"]["reference"] is None and "c3_extract" not in res["extract"]
    assert not (tmp_path / "ministral" / "tensors" / "candidates_reference.pt").exists()
    led = res["ledger"]
    assert led["ego"] == {"file": str(ego), "registered_sha256": plan["ego"]["sha256"],
                          "sha256": plan["ego"]["sha256"], "layers": [0, 1, 2]}
    assert set(led["e_82"]["abs_cos"]) == {"1", "2"} and set(led["ego_consecutive"]) == {"1", "2"}
    assert 0 <= led["e_82r"]["max"] <= 1 and led["e_82r"]["layer"] in (1, 2)
    found = res["select"]["choice"]["found"]
    assert ("skipped" in res["conditions"]["MA"]) == (not found)
    assert "skipped" not in res["conditions"]["MB"] and "skipped" not in res["conditions"]["MB-rand"]
    assert_no_hooks(model)


def test_v_ego_is_read_only_if_it_is_the_registered_file(tiny_mc, tmp_path, monkeypatch):
    """Ruling (1 Nov): a missing v_ego, or one whose sha256 is not the registered one, marks E-82 and E-82r not read
    with the reason, and nothing stops; the sha256 is checked before the file is loaded."""
    model, tok = tiny_mc
    m = model_entry(tok, model)
    plan = {"ego": {"model": "ministral", "sha256": "0" * 64}}

    def boom(*a, **k):
        raise AssertionError("E-82 was read")

    monkeypatch.setattr(refusal, "ego_ledger", boom)
    for path in (None, tmp_path / "absent.pt"):
        got = refusal.ego_reading(model, tok, {}, path, plan, m)
        assert got["not_read"] == "v_ego is not in the image, so E-82 and E-82r are not read"
        assert got["ego"]["sha256"] is None and got["ego"]["layers"] is None
    other = tmp_path / "other.pt"
    other.write_bytes(b"made-up bytes that torch could not load")
    got = refusal.ego_reading(model, tok, {}, other, plan, m)
    sha = refusal.sha256_file(other)
    assert got["ego"] == {"file": str(other), "registered_sha256": "0" * 64, "sha256": sha, "layers": None}
    assert got["not_read"] == f"v_ego has sha256 {sha}, not the registered {'0' * 64}, so E-82 and E-82r are not read"


def test_run_model_refuses_a_model_with_another_number_of_blocks(tiny_mc, tmp_path):
    model, tok = tiny_mc
    files = write_data(tmp_path / "data")
    plan = make_plan(config(), files, {"qwen": model_entry(tok, model, n_layers=28)})
    with pytest.raises(refusal.CheckError, match="28"):
        refusal.run_model(plan, "qwen", tmp_path, model, tok, PKG, log=quiet)


def test_stored_and_scored_conditions_read_the_volume_layout(tmp_path):
    (tmp_path / "qwen" / "completions").mkdir(parents=True)
    (tmp_path / "qwen" / "completions" / "QA.json").write_text("[]")
    (tmp_path / "qwen" / "completions" / "Q0.json").write_text("[]")
    assert refusal.scored_conditions(tmp_path) == {"qwen": ["Q0", "QA"], "ministral": []}
    refusal.write_json(tmp_path / "prepare.json", {"n": 1})
    s = refusal.stored({}, tmp_path)
    assert s["prepare"] == {"n": 1} and s["qwen"] is None and s["swap_q0"] is None


def test_paths_and_names_are_relative_to_the_check():
    assert Path(refusal.__file__).parent.name == "refusal_direction"
