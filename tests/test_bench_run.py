"""The battery's GPU evaluation as a library (anatta.bench.run), on the stand-ins of bench_mocks.py: no GPU, no
weights, no network.

  - Golden: for the battery's own formats, the item draw, every prompt, every probe.generate call and every
    model.generate call (ids, mask, max_new_tokens, do_sample, temperature, top_p, pad_token_id) are exactly those
    the original battery script made on the same synthetic rows (tests/fixtures/golden_battery_run.json), and the
    rows agree (the original kept 800 characters of each reply; these keep it whole).
  - Each reply's new-token count and hit_cap from the output ids: one end-of-sequence id or several, padding after
    it, a reply cut at the limit.
  - The system-message option: None leaves the template's default, "" is an empty system message.
  - The checks: template ids against the rendered prompt, the BOS count (one, none, a double BOS from the template or
    from the tokenizer), the batch handed to generate (left padding, pad id, ids), the first batch twice, batch 1.
  - run_jobs: files written, a run resumed from a partial file, a finished run read back.
"""

import json
from pathlib import Path

import pytest
import torch
from bench_mocks import DEFAULT_SYSTEM, MockModel, MockTok, synthetic_rows

import anatta.probe
from anatta.bench import items as bi
from anatta.bench import run as br
from anatta.bench.battery import MAX_NEW as BATTERY_MAX_NEW
from anatta.bench.fingerprint import CODE_FILES

FIX = Path(__file__).parent / "fixtures" / "golden_battery_run.json"


def golden() -> dict:
    return json.loads(FIX.read_text())


def spy_on_probe(monkeypatch) -> list:
    calls, original = [], anatta.probe.generate

    def spy(model, tok, batch, max_new_tokens=128, temperature=0.7, seed=0):
        calls.append({"conversations": batch, "max_new_tokens": max_new_tokens, "temperature": temperature,
                      "seed": seed})
        return original(model, tok, batch, max_new_tokens, temperature, seed)

    monkeypatch.setattr(anatta.probe, "generate", spy)
    return calls


def golden_item_file(g: dict) -> dict:
    i = g["inputs"]
    rows, read = {}, []
    for name, (repo, prefix) in i["sets"].items():
        files = bi.parquet_files(i["listing"][repo], prefix)
        read += [f"{repo}/{f}" for f in files]
        rows[name] = [r for f in files for r in i["files"][f"{repo}/{f}"]]
    assert read == g["outputs"]["read"]  # the same files, in the same order
    draws = {f"battery_{s}": {"set": s, "n": i["n"], "rule": "battery"} for s in i["sets"]}
    return bi.build_item_file(rows, draws, seed=0)


def test_golden_items_are_the_original_draw_and_prompts():
    g = golden()
    f = golden_item_file(g)
    assert BATTERY_MAX_NEW == g["inputs"]["max_new"] == {k: br.MAX_NEW[k] for k in BATTERY_MAX_NEW}
    want = g["outputs"]["items"]
    got = [f["items"][i] for s in g["inputs"]["sets"] for i in f["draws"][f"battery_{s}"]]
    assert [x["id"] for x in got] == [x["id"] for x in want]
    for mine, theirs in zip(got, want):
        assert mine["set"] == theirs["set"] and mine["gold"] == theirs["gold"]
        assert br.prompt_of(mine, mine["set"]) == theirs["prompt"]
        assert f["meta"]["rows"][mine["set"]] == theirs["of"]


def test_golden_prompts_batches_and_generate_kwargs(monkeypatch):
    g = golden()
    f = golden_item_file(g)
    calls = spy_on_probe(monkeypatch)
    tok = MockTok(bos=True)
    model = MockModel(tok)
    rows = []
    for s in g["inputs"]["sets"]:
        rows += br.evaluate_items(model, tok, bi.draw_items(f, f"battery_{s}"), s, batch_size=g["inputs"]["batch"],
                                  bos=1)
    out = g["outputs"]
    assert json.loads(json.dumps(calls)) == out["probe_calls"]  # the conversations, max_new_tokens, temperature, seed
    assert model.calls == out["generate_calls"]  # ids, mask, max_new_tokens, do_sample, temperature, top_p, pad id
    assert len(rows) == len(out["rows"])
    for mine, theirs in zip(rows, out["rows"]):
        assert {k: mine[k] for k in ("set", "id", "correct", "pred", "gold")} == {k: theirs[k] for k in ("set", "id", "correct", "pred", "gold")}
        assert mine["text"][:800] == theirs["text"]
    assert all(c["do_sample"] is False and c["temperature"] is None and c["top_p"] is None for c in model.calls)
    assert any(r["hit_cap"] for r in rows) and not all(r["hit_cap"] for r in rows)  # the long letter replies are cut


def test_replies_are_kept_whole_with_their_lengths():
    tok = MockTok(bos=True)
    model = MockModel(tok, answer=lambda prompt: "Let me think. " * 30)
    item = {"id": "mmlu_pro/0", "set": "mmlu_pro", "gold": "A", "n_options": 4, "prompts": {"mmlu_pro_cot": "Q?"}}
    rows = br.evaluate_items(model, tok, [item], "mmlu_pro_cot", max_new_tokens=50)
    assert rows[0]["hit_cap"] and rows[0]["new_tokens"] == 50 and len(rows[0]["text"]) == 50
    rows = br.evaluate_items(model, tok, [item], "mmlu_pro_cot")  # 2,048 tokens: the reply ends
    assert not rows[0]["hit_cap"] and rows[0]["new_tokens"] == len("Let me think. " * 30) + 1  # with its </s>
    assert rows[0]["text"] == "Let me think. " * 30 and rows[0]["pred"] is None and rows[0]["tiger_pred"] is None
    assert rows[0]["prompt_tokens"] == len(br.template_ids(tok, br.messages("Q?")))
    # in a batch, a short reply finishes early and is padded while a long one runs on
    model = MockModel(tok, answer=lambda prompt: "ANSWER: A" if "short" in prompt else "x" * 40)
    rows = br.evaluate_items(model, tok, [item, {**item, "id": "mmlu_pro/1", "prompts": {"mmlu_pro_cot": "short"}}],
                             "mmlu_pro_cot", max_new_tokens=30)
    by_id = {r["id"]: r for r in rows}
    assert (by_id["mmlu_pro/1"]["new_tokens"], by_id["mmlu_pro/1"]["hit_cap"], by_id["mmlu_pro/1"]["correct"]) == (10, False, True)
    assert (by_id["mmlu_pro/0"]["new_tokens"], by_id["mmlu_pro/0"]["hit_cap"]) == (30, True)


def test_reply_length_with_one_or_several_end_ids():
    assert br.reply_length([5, 6, 2, 0, 0], {2}) == (3, False)
    assert br.reply_length([5, 6, 7], {2}) == (3, True)
    # Qwen-like: two end ids, and the pad id is one of them
    assert br.reply_length([9, 151643, 151643], {151645, 151643}) == (2, False)
    assert br.reply_length([9, 151645, 151643], {151645, 151643}) == (2, False)
    model = MockModel(MockTok(), eos=[2, 3])
    assert br.eos_ids(model, model.tok) == {2, 3}
    model.generation_config.eos_token_id = None
    assert br.eos_ids(model, model.tok) == {2}


@pytest.mark.parametrize("system,expect", [
    (None, f"<s>[SYS]{DEFAULT_SYSTEM}[/SYS][INST]hi[/INST][A]"),
    ("You are a helpful assistant.", "<s>[SYS]You are a helpful assistant.[/SYS][INST]hi[/INST][A]"),
    ("", "<s>[SYS][/SYS][INST]hi[/INST][A]"),
])
def test_system_message_option(system, expect, monkeypatch):
    calls = spy_on_probe(monkeypatch)
    tok = MockTok(bos=True)
    model = MockModel(tok)
    item = {"id": "gsm8k/0", "set": "gsm8k", "gold": "1", "prompts": {"gsm8k": "hi"}}
    br.evaluate_items(model, tok, [item], "gsm8k", system=system, bos=1)
    assert calls[0]["conversations"] == [br.messages("hi", system)]
    assert tok.decode_ids(model.calls[0]["input_ids"][0]) == expect
    assert br.messages("hi") == [{"role": "user", "content": "hi"}]
    assert br.messages("hi", "")[0] == {"role": "system", "content": ""}


# ------------------------------------------------------------------------------------------------- the checks


def test_check_prompt_one_bos_passes_and_returns_the_ids():
    tok = MockTok(bos=True)
    text, ids = br.check_prompt(tok, br.messages("hi"), bos=1)
    assert text.startswith("<s>[SYS]") and ids[0] == 1 and ids[1] != 1
    assert ids == br.template_ids(tok, br.messages("hi"))


def test_check_prompt_qwen_like_no_bos():
    tok = MockTok(bos=False)
    _, ids = br.check_prompt(tok, br.messages("hi"), bos=0)
    assert ids[0] == 3  # [SYS]
    with pytest.raises(br.CheckError, match="0 BOS"):
        br.check_prompt(tok, br.messages("hi"), bos=1)


def test_check_prompt_double_bos_from_the_template_fails():
    tok = MockTok(bos=True, template_bos=2)
    with pytest.raises(br.CheckError, match="starts with 2 BOS"):
        br.check_prompt(tok, br.messages("hi"), bos=1)


def test_check_prompt_fails_when_the_rendered_text_tokenizes_differently():
    class Drift(MockTok):  # the template's own tokenization adds a BOS the rendered text does not carry
        def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=True, **kw):
            out = super().apply_chat_template(conversation, add_generation_prompt, tokenize, **kw)
            return out if not tokenize else {"input_ids": [1] + out["input_ids"]}

    with pytest.raises(br.CheckError, match="first differ at position 1"):
        br.check_prompt(Drift(bos=True), br.messages("hi"), bos=1)


def test_template_ids_accepts_lists_tensors_and_batch_encodings():
    tok = MockTok()
    want = tok.encode(tok.render(br.messages("hi"), True))

    class AsList(MockTok):
        def apply_chat_template(self, c, add_generation_prompt=False, tokenize=True, **kw):
            return want if tokenize else super().apply_chat_template(c, add_generation_prompt, False)

    class AsTensor(MockTok):
        def apply_chat_template(self, c, add_generation_prompt=False, tokenize=True, **kw):
            return torch.tensor([want]) if tokenize else super().apply_chat_template(c, add_generation_prompt, False)

    for t in (tok, AsList(), AsTensor()):
        assert br.template_ids(t, br.messages("hi")) == want


def _batch(tok, prompts, add_special_tokens=False, side="left"):
    tok.padding_side = side
    texts = [tok.render(br.messages(p), True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens)
    tok.padding_side = "right"
    return dict(enc), [br.template_ids(tok, br.messages(p)) for p in prompts]


def test_check_batch_passes_the_battery_path_and_catches_a_double_bos():
    tok = MockTok(bos=True)
    kwargs, expect = _batch(tok, ["a", "a longer one"])
    br.check_batch(kwargs, expect, tok.pad_token_id)
    kwargs, expect = _batch(tok, ["a", "a longer one"], add_special_tokens=True)  # the tokenizer adds a second BOS
    with pytest.raises(br.CheckError, match="differ from apply_chat_template"):
        br.check_batch(kwargs, expect, tok.pad_token_id)


def test_check_batch_catches_right_padding_and_a_wrong_pad_id():
    tok = MockTok(bos=True)
    kwargs, expect = _batch(tok, ["a", "a longer one"], side="right")
    with pytest.raises(br.CheckError, match="not left-padded"):
        br.check_batch(kwargs, expect, tok.pad_token_id)
    kwargs, expect = _batch(tok, ["a", "a longer one"])
    with pytest.raises(br.CheckError, match="other than the pad token"):
        br.check_batch(kwargs, expect, 99)
    with pytest.raises(br.CheckError, match="rows go to generate"):
        br.check_batch(kwargs, expect[:1], tok.pad_token_id)


def test_the_recorder_stops_a_batch_before_generation():
    tok = MockTok(bos=True)
    model = MockModel(tok)
    rec = br.Recorder(model, tok.pad_token_id)
    kwargs, expect = _batch(tok, ["a", "bb"], add_special_tokens=True)
    rec.expect = expect
    with pytest.raises(br.CheckError):
        rec.generate(**kwargs, max_new_tokens=4, do_sample=False, temperature=None, top_p=None, pad_token_id=0)
    assert model.calls == []  # nothing was generated
    assert rec.device == "cpu" and rec.generation_config is model.generation_config


def test_check_all_covers_every_prompt_and_reports_item_zero():
    tok = MockTok(bos=True)
    f = golden_item_file(golden())
    jobs = [{"name": "battery_gsm8k", "format": "gsm8k", "ids": f["draws"]["battery_gsm8k"]}]
    out = br.check_all(tok, jobs, f["items"], None, 1)["battery_gsm8k"]
    assert out["prompts"] == 12 and out["bos"] == 1 and out["item0"]["id"] == f["draws"]["battery_gsm8k"][0]
    assert out["item0"]["first_ids"][:2] == [1, 3] and out["tokens_min"] <= out["item0"]["n_ids"] <= out["tokens_max"]
    with pytest.raises(br.CheckError):
        br.check_all(MockTok(bos=True, template_bos=2), jobs, f["items"], None, 1)


def test_first_batch_twice_and_batch_one():
    tok = MockTok(bos=True)
    f = golden_item_file(golden())
    items = bi.draw_items(f, "battery_mmlu_pro")
    det = br.repeat_first_batch(MockModel(tok), tok, items, "mmlu_pro", 5, bos=1)
    assert det["n"] == 5 and det["identical"] == 5 and det["share"] == 1.0
    assert det["ids"] == [x["id"] for x in br.batches(items, "mmlu_pro", 5)[0]]
    det = br.repeat_first_batch(MockModel(tok, sample=True), tok, items, "mmlu_pro", 5, bos=1)
    assert det["identical"] == 0
    model = MockModel(tok)
    ref = br.evaluate_items(model, tok, items, "mmlu_pro", 5, bos=1)
    b1 = br.batch_one(model, tok, items[:4], "mmlu_pro", ref, bos=1)
    assert b1["n"] == 4 and b1["share"] == 1.0 and b1["gap_points"] == 0.0
    assert {c["input_ids"].__len__() for c in model.calls[-4:]} == {1}  # one prompt per call


# --------------------------------------------------------------------------------------------------- run_jobs


def _jobs(f):
    return [{"name": "gsm", "format": "gsm8k", "draw": "battery_gsm8k", "ids": f["draws"]["battery_gsm8k"], "batch": 5, "max_new": 512},
            {"name": "letters", "format": "mmlu_pro", "draw": "battery_mmlu_pro", "ids": f["draws"]["battery_mmlu_pro"], "batch": 5, "max_new": 24}]


def test_run_jobs_writes_every_file_and_reads_a_finished_run_back(tmp_path):
    tok = MockTok(bos=True)
    f = golden_item_file(golden())
    commits = []
    res = br.run_jobs(MockModel(tok), tok, _jobs(f), f["items"], tmp_path, None, 1,
                      determinism={"job": "gsm", "min_share": 0.95}, batch1={"gsm": f["draws"]["battery_gsm8k"][:3]},
                      commit=lambda: commits.append(1), log=lambda *a: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["batch1.json", "checks.json", "determinism.json", "gsm.json",
                                                          "letters.json", "run_info.json"]
    assert res["determinism"]["passed"] and res["determinism"]["job"] == "gsm"
    gsm = json.loads((tmp_path / "gsm.json").read_text())
    assert gsm["n"] == 12 and gsm["batch"] == 5 and gsm["max_new"] == 512 and gsm["system"] is None
    assert gsm["generate_kwargs"] == {"max_new_tokens": 512, "do_sample": False, "temperature": None, "top_p": None,
                                      "pad_token_id": 0}
    kept = json.loads((tmp_path / "run_info.json").read_text())["generate_kwargs"]  # each job's, at its first batch
    assert kept["gsm"] == gsm["generate_kwargs"] and kept["letters"]["max_new_tokens"] == 24
    assert res["batch1"]["gsm"]["n"] == 3 and len(commits) >= 5
    (tmp_path / "run_info.json").write_text("{}")
    back = br.stored_run(tmp_path, _jobs(f), {"job": "gsm"}, {"gsm": []})
    assert back["jobs"]["gsm"]["rows"] == gsm["rows"] and back["batch1"] == res["batch1"]
    assert br.stored_run(tmp_path, _jobs(f) + [{"name": "missing"}]) is None


def test_run_jobs_stops_when_the_first_batch_differs(tmp_path):
    tok = MockTok(bos=True)
    f = golden_item_file(golden())
    with pytest.raises(br.CheckError, match="is sampling on"):
        br.run_jobs(MockModel(tok, sample=True), tok, _jobs(f), f["items"], tmp_path, None, 1,
                    determinism={"job": "gsm", "min_share": 0.95}, log=lambda *a: None)
    det = json.loads((tmp_path / "determinism.json").read_text())
    assert det["passed"] is False and not (tmp_path / "gsm.json").exists()


def test_run_jobs_stops_on_a_bad_prompt_before_generating(tmp_path):
    tok = MockTok(bos=True, template_bos=2)
    model = MockModel(tok)
    f = golden_item_file(golden())
    with pytest.raises(br.CheckError, match="BOS"):
        br.run_jobs(model, tok, _jobs(f), f["items"], tmp_path, None, 1, log=lambda *a: None)
    assert model.calls == []


def test_a_job_resumes_from_its_partial_file(tmp_path):
    tok = MockTok(bos=True)
    f = golden_item_file(golden())
    job = _jobs(f)[0]
    full = br.run_job(MockModel(tok), tok, job, f["items"], tmp_path / "a", None, 1, log=lambda *a: None)
    # a run that stopped after its first batch
    part = br.batches(bi.draw_items(f, "battery_gsm8k"), "gsm8k", 5)[0]
    first = [r for r in full["rows"] if r["id"] in {x["id"] for x in part}]
    br.append_jsonl(tmp_path / "b" / "gsm.partial.jsonl", {"batch": 0, "ids": [x["id"] for x in part], "rows": first})
    model = MockModel(tok)
    res = br.run_job(model, tok, job, f["items"], tmp_path / "b", None, 1, log=lambda *a: None)
    assert res["rows"] == full["rows"] and res["resumed_batches"] == 1 and len(model.calls) == 2  # batches 1 and 2 only
    assert not (tmp_path / "b" / "gsm.partial.jsonl").exists()
    # a partial file from other items is refused
    br.append_jsonl(tmp_path / "c" / "gsm.partial.jsonl", {"batch": 0, "ids": ["gsm8k/999"], "rows": []})
    with pytest.raises(br.CheckError, match="other items"):
        br.run_job(MockModel(tok), tok, job, f["items"], tmp_path / "c", None, 1, log=lambda *a: None)


def test_snapshot_dir_pins_the_revision(tmp_path):
    snap = tmp_path / "hub" / "models--Org--Name" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    assert br.snapshot_dir("Org/Name", "abc123", str(tmp_path)) == str(snap)
    with pytest.raises(FileNotFoundError, match=r"snapshots there: \['abc123'\]"):
        br.snapshot_dir("Org/Name", "def456", str(tmp_path))


def test_model_info_and_code_hashes(tmp_path):
    tok = MockTok(bos=True)
    info = br.model_info(MockModel(tok, repetition_penalty=1.05), tok)
    assert info["repetition_penalty"] == 1.05 and info["generation_config"]["repetition_penalty"] == 1.05
    assert info["dtype"] == "torch.bfloat16" and info["tokenizer"] == {"bos": 1, "eos": 2, "pad": 0, "padding_side": "right"}
    assert info["torch"] == torch.__version__ and info["commit_hash"] == "0" * 40
    hashes = br.code_hashes(Path(anatta.probe.__file__).parent)
    assert set(hashes) == set(CODE_FILES) and all(len(h) == 64 for h in hashes.values())


def test_synthetic_rows_are_shaped_like_the_sets():
    rows = synthetic_rows(8)
    assert set(rows) == {"gsm8k", "mmlu_pro", "arc_challenge"} and all(len(v) == 8 for v in rows.values())


# ------------------------------------------------------------------------- resumes, kwargs, the environment


def test_read_jsonl_drops_a_cut_last_line_and_raises_on_any_other(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b": [1, 2\n')
    assert br.read_jsonl(p) == [{"a": 1}, {"a": 2}]
    assert "last line is not JSON" in capsys.readouterr().out and p.read_text().count("\n") == 3  # left as it was
    assert br.read_jsonl(p, repair=True) == [{"a": 1}, {"a": 2}]
    assert p.read_text() == '{"a": 1}\n{"a": 2}\n' and br.read_jsonl(p) == [{"a": 1}, {"a": 2}]
    p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
    with pytest.raises(ValueError, match="line 2 of 3 is not JSON"):
        br.read_jsonl(p, repair=True)
    assert br.read_jsonl(tmp_path / "none.jsonl") == []


def test_a_job_resumes_past_a_last_line_cut_short(tmp_path, capsys):
    tok = MockTok(bos=True)
    f = golden_item_file(golden())
    job = _jobs(f)[0]
    full = br.run_job(MockModel(tok), tok, job, f["items"], tmp_path / "a", None, 1, log=lambda *a: None)
    first, second = br.batches(bi.draw_items(f, "battery_gsm8k"), "gsm8k", 5)[:2]
    rows = {r["id"]: r for r in full["rows"]}
    partial = tmp_path / "b" / "gsm.partial.jsonl"
    br.append_jsonl(partial, {"batch": 0, "ids": [x["id"] for x in first], "rows": [rows[x["id"]] for x in first]})
    line = json.dumps({"batch": 1, "ids": [x["id"] for x in second], "rows": [rows[x["id"]] for x in second]})
    with open(partial, "a") as fh:
        fh.write(line[: len(line) // 2])  # the container died mid-write
    model = MockModel(tok)
    res = br.run_job(model, tok, job, f["items"], tmp_path / "b", None, 1, log=lambda *a: None)
    assert res["rows"] == full["rows"] and res["resumed_batches"] == 1 and len(model.calls) == 2
    assert "last line is not JSON" in capsys.readouterr().out


def test_a_resumed_job_with_every_batch_done_still_reports_its_generate_kwargs(tmp_path):
    tok = MockTok(bos=True)
    f = golden_item_file(golden())
    job = _jobs(f)[0]
    n_batches = len(br.batches(bi.draw_items(f, "battery_gsm8k"), "gsm8k", 5))
    commits = []

    def crash_after_the_last_batch():
        commits.append(1)
        if len(commits) == n_batches:
            raise RuntimeError("preempted")

    with pytest.raises(RuntimeError, match="preempted"):
        br.run_job(MockModel(tok), tok, job, f["items"], tmp_path, None, 1, commit=crash_after_the_last_batch,
                   log=lambda *a: None)
    assert not (tmp_path / "gsm.json").exists() and len(br.read_jsonl(tmp_path / "gsm.partial.jsonl")) == n_batches
    model = MockModel(tok)
    res = br.run_job(model, tok, job, f["items"], tmp_path, None, 1, log=lambda *a: None)
    assert model.calls == [] and res["resumed_batches"] == n_batches
    assert res["generate_kwargs"]["do_sample"] is False and res["generate_kwargs"]["max_new_tokens"] == 512


def _plan(f, pkg, **over):
    jobs = _jobs(f)
    return {"model": "Org/Name", "revision": "r" * 40, "condition": "default", "system": None, "bos": 1, "jobs": jobs,
            "determinism": {"job": "gsm", "min_share": 0.95}, "batch1": {"gsm": f["draws"]["battery_gsm8k"][:2]},
            "gpu": "L40S", "items_sha256": "ab" * 32, "code": br.code_hashes(pkg), "pins": {}, **over}


def test_check_environment_holds_the_container_to_the_plan(tmp_path, monkeypatch):
    pkg = Path(anatta.probe.__file__).parent
    f = golden_item_file(golden())
    plan = _plan(f, pkg, pins={"torch": "2.14.1"})
    monkeypatch.setattr(br, "installed", lambda names: {n: "2.14.1+cu128" for n in names})  # a local label is fine
    env = br.check_environment(plan, pkg)
    assert env["code"] == plan["code"] and env["installed"] == {"torch": "2.14.1+cu128"}
    monkeypatch.setattr(br, "installed", lambda names: {n: "2.13.0" for n in names})
    with pytest.raises(br.CheckError, match=r"torch 2.13.0 \(pinned 2.14.1\)"):
        br.check_environment(plan, pkg)
    edited = {**plan["code"], "probe.py": "0" * 64}
    with pytest.raises(br.CheckError, match="differs from the plan's in probe.py"):
        br.check_environment({**plan, "code": edited}, pkg)


def test_run_plan_runs_once_reads_back_and_refuses_another_plan(tmp_path):
    pkg = Path(anatta.probe.__file__).parent
    f = golden_item_file(golden())
    hf = tmp_path / "hf"
    (hf / "hub" / "models--Org--Name" / "snapshots" / ("r" * 40)).mkdir(parents=True)
    (hf / "hub" / "models--Org--Name" / "refs").mkdir()
    (hf / "hub" / "models--Org--Name" / "refs" / "main").write_text("s" * 40 + "\n")
    tok, loads = MockTok(bos=True), []

    def load(path):
        loads.append(path)
        return MockModel(tok), tok

    plan = _plan(f, pkg)
    out = tmp_path / "run"
    res = br.run_plan(plan, f["items"], out, pkg, load, log=lambda *a: None, hf_home=str(hf))
    info = res["run_info"]
    assert loads == [str(hf / "hub" / "models--Org--Name" / "snapshots" / ("r" * 40))]
    assert info["cache_refs_main"] == "s" * 40 and info["starts"] == 1 and info["plan_id"] == br.plan_id(plan)
    assert info["hashes"] == plan["code"] and set(info["generate_kwargs"]) == {"gsm", "letters"}
    assert json.loads((out / "plan.json").read_text()) == plan and res["determinism"]["passed"]
    back = br.run_plan(plan, f["items"], out, pkg, load, log=lambda *a: None, hf_home=str(hf))
    assert len(loads) == 1 and back["jobs"]["gsm"]["rows"] == res["jobs"]["gsm"]["rows"]
    with pytest.raises(br.CheckError, match="another plan"):
        br.run_plan({**plan, "gpu": "A100"}, f["items"], out, pkg, load, log=lambda *a: None, hf_home=str(hf))
    assert len(loads) == 1


def test_cache_ref(tmp_path):
    refs = tmp_path / "hub" / "models--Org--Name" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc\n")
    assert br.cache_ref("Org/Name", "main", str(tmp_path)) == "abc"
    assert br.cache_ref("Org/Other", "main", str(tmp_path)) is None
