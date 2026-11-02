"""reference.py: R-lab32 against the vendored script called directly, and the CSV step (with a stand-in for pandas
that follows read_csv's default missing values; the real pandas is checked by csv_selftest in the second image)."""

import json

import pytest
import reference as Rf
import torch
import traits as T
from c4_testing import CHECK, make_pandas, vendored_module

BLOB = "d819f48e2753ed7cc9b364c19be5ab00bdf2a056"


def rows(tok, n=3):
    from generate import render

    out = {}
    for j, sign in enumerate(T.SIGNS):
        system = "You are an upbeat assistant. Speak warmly." if sign == "pos" else "You are a helpful assistant."
        out[sign] = [{"prompt": render(tok, system, f"What about the garden number {i}?"),
                      "answer": [" the quick brown fox", "\nfinal answer 42", " café lazy dog"][i % 3]}
                     for i in range(n)]
    return out


def test_vendored_functions_need_the_registered_blob(tmp_path):
    fns = Rf.vendored_functions(CHECK / T.GENERATE_VEC, BLOB)
    assert set(fns) == {"get_hidden_p_and_r"} and callable(fns["get_hidden_p_and_r"])
    changed = tmp_path / "generate_vec.py"
    changed.write_bytes((CHECK / T.GENERATE_VEC).read_bytes() + b"\n")
    with pytest.raises(T.CheckError) as e:
        Rf.vendored_functions(changed, BLOB)
    assert e.value.code == "vendored"


def test_rlab32_equals_the_vendored_script_called_directly(tiny, tmp_path):
    """R-lab32 (the vendored function under an outer no_grad, save_persona_vector's arithmetic copied) against
    generate_vec.save_persona_vector itself, reading CSVs this check wrote: the three outputs are equal."""
    model, tok = tiny
    pd = make_pandas()
    r = rows(tok)
    fn = Rf.vendored_functions(CHECK / T.GENERATE_VEC, BLOB)["get_hidden_p_and_r"]
    ours = {}
    outs = {s: Rf.rlab32(model, tok, [x["prompt"] for x in r[s]], [x["answer"] for x in r[s]], fn) for s in T.SIGNS}
    for j, kind in enumerate(Rf.KINDS):
        ours[kind] = Rf.avg_diff(outs["pos"][j], outs["neg"][j])
    for s in T.SIGNS:
        recs = [{"question": "q", "prompt": x["prompt"], "answer": x["answer"], "question_id": f"upbeat_{i}",
                 "upbeat": 100 if s == "pos" else 0, "coherence": 100} for i, x in enumerate(r[s])]
        Rf.write_csv(pd, tmp_path / f"upbeat_{s}.csv", recs, "upbeat")
    mod = vendored_module(model, tok, pd)
    mod.save_persona_vector("tiny", str(tmp_path / "upbeat_pos.csv"), str(tmp_path / "upbeat_neg.csv"), "upbeat",
                            str(tmp_path / "out"), 50)
    for kind in Rf.KINDS:
        theirs = torch.load(tmp_path / "out" / f"upbeat_{kind}_diff.pt", weights_only=False)
        assert theirs.shape == (len(model.model.layers) + 1, model.config.hidden_size)
        assert torch.equal(ours[kind], theirs)


def test_csv_round_trip_flags_answers_read_as_missing(tmp_path):
    pd = make_pandas()
    recs = [{"question": "q", "prompt": "line one\nline two", "answer": a, "question_id": "t_0", "t": 100,
             "coherence": 100} for a in ["fine words", "", "None", "NA", "two\nlines", ' "quoted", comma']]
    Rf.write_csv(pd, tmp_path / "x.csv", recs, "t")
    assert Rf.survives(pd, tmp_path / "x.csv", recs) == [True, False, False, False, True, True]
    res = Rf.csv_selftest(pd, tmp_path)
    assert res["named_flagged"] == 3 and res["ordinary_kept"] == len(Rf.SELFTEST_KEEP)


def test_csv_selftest_refuses_a_writer_that_loses_text(tmp_path, monkeypatch):
    pd = make_pandas()
    real = Rf.write_csv
    monkeypatch.setattr(Rf, "write_csv", lambda pd_, path, records, trait: real(
        pd_, path, [{**r, "answer": r["answer"].strip()} for r in records], trait))
    with pytest.raises(T.CheckError) as e:
        Rf.csv_selftest(pd, tmp_path)
    assert e.value.code == "csv_selftest"


def test_run_csv_drops_failing_pairs_for_every_extractor(tiny, tmp_path):
    """A pair whose answer won't survive the round trip (empty, "None", "NA") is dropped, both signs, and counted; the
    kept pairs are written in row order and read back intact."""
    import generate as G
    from c4_testing import benign_trait

    _, tok = tiny
    pd = make_pandas()
    traits = ["upbeat"]
    data = {"upbeat": benign_trait("upbeat", n_q=2, n_pairs=1)}
    prompts = {"pos": "You are {article} {trait} assistant. {instruction}",
               "neg": "You are a helpful assistant. {instruction}", "family": "calibration", "description": "x {trait}"}
    plan = {"traits": traits, "layout": {"questions": 2, "instructions": 1},
            "G": {"samples": 2, "batch": 10}, "prompts": prompts, "reference": {"scores": {"pos": 100, "neg": 0,
                                                                                           "coherence": 100}}}
    answers = {"pos": [" fine", "None", " also fine", " ok"], "neg": [" plain", " plain too", " NA", ""]}
    for b in T.batches(traits, 2, 1, 2, 10):
        out = []
        for k in b["rows"]:
            qi, ii, si = T.cell_of(k, 1, 2)
            system = T.system_strings("upbeat", data["upbeat"]["instruction"][ii], prompts)[b["sign"]]
            p = G.render(tok, system, data["upbeat"]["questions"][qi])
            a = answers[b["sign"]][k]
            out.append({"row": k, "qi": qi, "ii": ii, "si": si, "prompt": p, "answer": a,
                        "slice_tokens": G.slice_tokens(tok, p, a)})
        T.write_jsonl(G.batch_file(tmp_path, "G", b["index"]), out)
    res = Rf.run_csv(plan, tmp_path, data, tok, pd, tmp_path / "tmp", None)
    T.assert_text_free(res)
    c = res["traits"]["upbeat"]
    assert c["pairs"] == 4 and c["kept"] == 2 and c["dropped"]["csv"] == 2
    assert G.keep_rows(tmp_path) == {"upbeat": {0, 2}}
    df = pd.read_csv(Rf.csv_dir(tmp_path) / "upbeat_neg.csv")
    assert df["answer"].tolist() == [" plain", " NA"] and df["upbeat"].tolist() == [0, 0]


def test_the_judged_arm_reads_its_prompt_and_scores_text_only(tmp_path):
    """The conditional judged arm, offline: coherence_0_100 found in any of the forms eval/prompts.py might use, bare
    numbers parsed, and run_judge writing scores (numbers and the snapshot) with a stand-in for OpenRouter."""
    import generate as G
    from c4_testing import benign_trait

    for src, form in (('coherence_0_100 = "Rate {question} {answer}"', "assign"),
                      ('P = {}\nP["coherence_0_100"] = "Rate {question} {answer}"', "subscript"),
                      ('PROMPTS = {"coherence_0_100": "Rate {question} {answer}", "other": "x"}', "dict")):
        assert Rf.named_string(src, "coherence_0_100") == "Rate {question} {answer}"
        assert Rf.named_string_form(src, "coherence_0_100") == ("Rate {question} {answer}", form)
    with pytest.raises(T.CheckError):
        Rf.named_string("x = 1", "coherence_0_100")
    assert [Rf.parse_number(x) for x in ("87", " 0\n", "100.", "101", "about 50", "", None)] == [87, 0, 100, None,
                                                                                                  None, None, None]
    plan = {"traits": ["upbeat"], "layout": {"questions": 1, "instructions": 1}, "G": {"samples": 2, "batch": 10},
            "judged": {"model": "openai/gpt-4.1-mini", "temperature": 0, "max_tokens": 8, "url": "https://x.invalid",
                       "workers": 2}}
    for b in T.batches(["upbeat"], 1, 1, 2, 10):
        T.write_jsonl(G.batch_file(tmp_path, "G", b["index"]),
                      [{"row": k, "qi": 0, "ii": 0, "si": k, "prompt": "p", "answer": f" reply {k}"} for k in b["rows"]])
    T.write_json(G.keep_file(tmp_path), {"kept": {"upbeat": [0, 1]}, "counts": {}})
    sent = []

    def post(body):
        sent.append(json.loads(body))
        return {"model": "openai/gpt-4.1-mini-2025-04-14", "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                "choices": [{"message": {"content": "73"}}]}

    data = {"upbeat": benign_trait("upbeat", n_q=1)}
    res = Rf.run_judge(plan, tmp_path, data, "Coherent? {question} {answer}", "not-a-real-key", post=post,
                       log=lambda *a: None)
    T.assert_text_free(res)
    assert res["prompt_sha256"] == {"coherence_0_100": T.sha256_bytes(b"Coherent? {question} {answer}"),
                                    "eval_prompt": {"upbeat": T.sha256_bytes(data["upbeat"]["eval_prompt"].encode())}}
    assert res["tokens"] == {"in": 80, "out": 8} and len(sent) == 8
    assert all(b["temperature"] == 0 and b["model"] == "openai/gpt-4.1-mini" for b in sent)
    scores = Rf.judged_scores(tmp_path, plan)["upbeat"]
    assert scores["pos"] == {0: (73, 73), 1: (73, 73)}
    assert res["traits"]["upbeat"]["pos"]["snapshots"] == ["openai/gpt-4.1-mini-2025-04-14"]


PROMPTS_PY = b'Prompts = {}\n\nPrompts["coherence_0_100"] = """Rate how clear this reply is.\nQuestion: {question}\nReply: {answer}\nNumber:"""\n'


def test_prepare_checks_the_prompts_file_and_the_judge_reads_the_volumes_copy(tmp_path):
    """Ruling 10 on a made-up prompts file: its blob and size checked like a trait file's, its sha256 and the form that
    binds coherence_0_100 returned for config.yaml, the file written to the volume; read_prompts reads that copy only,
    and refuses another sha256 or form."""
    repo = {"raw": "https://example.invalid/raw", "commit": "b8e0f044fe2410a6fad579f38324f03f13b4e917"}
    spec = {"file": "eval/prompts.py", "bytes": len(PROMPTS_PY), "blob": T.git_blob_sha(PROMPTS_PY)}
    seen = []
    facts = Rf.prepare_prompts(repo, spec, tmp_path, fetch=lambda url: seen.append(url) or PROMPTS_PY)
    assert seen == [f"https://example.invalid/raw/{repo['commit']}/eval/prompts.py"]
    assert facts == {"bytes": len(PROMPTS_PY), "blob": spec["blob"], "sha256": T.sha256_bytes(PROMPTS_PY),
                     "form": "subscript"}
    T.assert_text_free(facts)
    with pytest.raises(T.CheckError) as e:
        Rf.prepare_prompts(repo, {**spec, "blob": "0" * 40}, tmp_path / "b", fetch=lambda url: PROMPTS_PY)
    assert e.value.code == "blob"
    plan = {"data_dir": str(tmp_path), "judged": {"prompts": {**spec, "sha256": facts["sha256"], "form": "subscript"}}}
    assert Rf.read_prompts(plan).startswith("Rate how clear") and "{answer}" in Rf.read_prompts(plan)
    for bad in ({"sha256": "f" * 64}, {"form": "dict"}, {"sha256": None}):
        with pytest.raises(T.CheckError) as e:
            Rf.read_prompts({**plan, "judged": {"prompts": {**plan["judged"]["prompts"], **bad}}})
        assert e.value.code == "prompts"


def test_peak_memory_falls_back_and_refuses_a_zero_reading():
    """nvidia-smi first, torch.cuda's reading from the parent when it is missing; with a GPU present, a peak of 0 is a
    failed measurement."""
    def missing():
        raise FileNotFoundError

    pm = Rf.PeakMemory(readers=(("nvidia-smi", missing), ("torch.cuda", lambda: (1234, 81559))))
    pm.read_once()
    assert (pm.peak, pm.total, pm.source) == (1234, 81559, "torch.cuda")
    pm.check("NVIDIA H100 80GB HBM3")
    pm = Rf.PeakMemory(readers=(("nvidia-smi", lambda: (2000, 81559)), ("torch.cuda", lambda: (1, 1))))
    pm.read_once()
    assert pm.source == "nvidia-smi" and pm.peak == 2000
    dead = Rf.PeakMemory(readers=(("nvidia-smi", missing), ("torch.cuda", missing)))
    with dead:
        pass
    assert dead.peak == 0 and dead.source is None
    dead.check(None)  # no GPU: no reading is expected
    with pytest.raises(T.CheckError) as e:
        dead.check("NVIDIA A100-SXM4-80GB")
    assert e.value.code == "peak_memory"


SHARDS = ("Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]"
          "\rLoading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.20it/s]\n")


def row_bar(rate: str) -> str:
    """One of the script's row bars over 5 rows as tqdm draws it to a file: its updates, the last at the smoothed rate,
    then the closing redraw at the overall rate, padded, and the newline."""
    return ("  0%|          | 0/5 [00:00<?, ?it/s]\r 20%|██        | 1/5 [00:00<00:01,  3.10it/s]"
            "\r100%|██████████| 5/5 [00:01<00:00,  4.00it/s]\r100%|██████████| 5/5 [00:01<00:00, " + rate + "]  \n")


def test_the_row_passes_are_timed_from_tqdms_closing_lines():
    """A pass's seconds are its rows over the overall rate on the bar's closing line (it/s, or s/it under one row a
    second); transformers' shards bar (described), a bar over another count and an unfinished bar don't count."""
    data = SHARDS + row_bar(" 3.10it/s") + "a plain log line\n" + row_bar(" 2.50s/it")
    assert Rf.pass_seconds(data.encode(), 5) == pytest.approx([5 / 3.10, 5 * 2.50])
    assert Rf.pass_seconds(data.encode(), 4) == []
    unfinished = SHARDS + "  0%|          | 0/5 [00:00<?, ?it/s]\r100%|██████████| 5/5 [00:01<00:00,  4.00it/s]"
    assert Rf.pass_seconds(unfinished.encode(), 5) == []


def test_tqdm_draws_the_closing_line_pass_seconds_reads(tmp_path):
    """The real tqdm (the same bar and closing code as the image's pinned 4.67.1) drawing to a file, as the script's
    bars go to its stderr."""
    import time

    from tqdm import tqdm

    log = tmp_path / "stderr.log"
    with open(log, "w") as f:
        for _ in range(2):
            for _ in tqdm(range(5), total=5, file=f):
                time.sleep(0.02)
    got = Rf.pass_seconds(log.read_bytes(), 5)
    assert len(got) == 2 and all(0.09 <= x <= 1.0 for x in got)


def test_csv_size_counts_what_the_script_reads(tiny, tmp_path):
    """The rows the script's mask keeps and their tokens, prompt + answer without special tokens."""
    _, tok = tiny
    pd = make_pandas()
    r = rows(tok)
    paths = {}
    for sign in T.SIGNS:
        recs = [{"question": "q", "prompt": x["prompt"], "answer": x["answer"], "question_id": "q_0_0",
                 "evil": 100 if sign == "pos" else 0, "coherence": 100} for x in r[sign]]
        recs[0]["coherence"] = 40 if sign == "neg" else 100  # the script's mask drops this pair from both signs
        paths[sign] = tmp_path / f"evil_{sign}.csv"
        pd.DataFrame(recs, columns=list(recs[0])).to_csv(paths[sign], index=False)
    size = Rf.csv_size(pd, tok, paths, "evil", 50)
    n = {s: sum(len(tok(x["prompt"] + x["answer"], add_special_tokens=False)["input_ids"]) for x in r[s][1:])
         for s in T.SIGNS}
    assert size == {"rows": 4, "pairs": 2, "tokens": n["pos"] + n["neg"], "neg_tokens": n["neg"]}
