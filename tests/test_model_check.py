"""Calibration check 1, the model check (calibration/model_check): its config, the card's statistics and grades
(Wilson intervals, the bands, the format explanation, the verdict), the battery rerun, the contrast between
Ministral's conditions, and the Modal app under modal_stub's stand-in modal: its images and functions, the plan's
fingerprint (package files and pins), the guards (results folder, item-file hash, the battery's items, another
plan's files, an edited file under a stored run), the gate on Qwen's verdict, the draw from pinned parquet files on
stand-in hub and parquet modules (and the volume copy's rule), and a whole run (Qwen, then Ministral's conditions,
then the card) on bench_mocks' tokenizer and model. Nothing here touches the network, a GPU or Modal."""

import importlib.util
import json
import random
import shutil
import sys
import types
from pathlib import Path

import pytest
from bench_mocks import MockModel, MockTok, synthetic_rows
from modal_stub import load_apps, unload_apps
from test_bench_items import C1, SIZES

from anatta.bench import fingerprint as fp
from anatta.bench import items as bi
from anatta.bench import run as br

CHECK = Path(__file__).resolve().parent.parent / "calibration" / "model_check"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


card = load("model_check_card", CHECK / "card.py")
CFG = card.load_config()
# The draw tests run on the config as it was before registration: ARC at "main" and no item file registered.
UNREGISTERED = json.loads(json.dumps(CFG))
UNREGISTERED["datasets"]["arc_challenge"]["revision"] = "main"
UNREGISTERED["items"]["sha256"] = None
SMALL_DRAWS = {"c1_gsm8k": {"set": "gsm8k", "n": "all"},
               "c1_mmlu_pro": {"set": "mmlu_pro", "n": 25, "rule": "stratified", "stratify": "category"},
               "cheap_gsm8k": {"set": "gsm8k", "n": 10, "rule": "stratified"},
               "cheap_mmlu_pro": {"set": "mmlu_pro", "n": 8, "rule": "stratified", "stratify": "category"},
               "batch1_gsm8k": {"set": "gsm8k", "n": 4, "rule": "stratified"},
               "batch1_mmlu_pro": {"set": "mmlu_pro", "n": 4, "rule": "stratified", "stratify": "category"},
               "battery_gsm8k": {"set": "gsm8k", "n": 12, "rule": "battery"},
               "battery_mmlu_pro": {"set": "mmlu_pro", "n": 12, "rule": "battery"},
               "battery_arc_challenge": {"set": "arc_challenge", "n": 12, "rule": "battery"}}


# ------------------------------------------------------------------------------------------------- the config


def test_config_registers_the_models_data_draws_and_bands():
    q, mi = CFG["models"]["qwen"], CFG["models"]["ministral"]
    assert (q["id"], q["revision"], q["bos"]) == ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", 0)
    assert (mi["id"], mi["revision"], mi["bos"]) == ("mistralai/Ministral-3-8B-Instruct-2512-BF16",
                                                     "f6fae9795746f63c9be8344932f01275f3c63734", 1)
    assert q["batch"] == {"default": 50} and mi["batch"] == {"default": 50, "mmlu_pro_cot": 25}
    assert q["conditions"] == ["default"] and mi["conditions"] == ["default", "helpful"] and mi["third_condition"] == "empty"
    assert CFG["conditions"] == {"default": None, "helpful": "You are a helpful assistant.", "empty": ""}
    assert CFG["datasets"]["gsm8k"]["revision"] == "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"
    assert CFG["datasets"]["mmlu_pro"]["revision"] == "527feea0afed1de15a8c115abf7be4c912123315"
    assert CFG["allocation"]["c1_mmlu_pro"] == C1 == bi.allocate(SIZES, 1000)[0]
    assert CFG["max_new_tokens"] == {"gsm8k": 512, "mmlu_pro": 24, "arc_challenge": 24, "mmlu_pro_cot": 2048}
    assert CFG["max_new_tokens"] == br.MAX_NEW and set(CFG["draws"]) == set(SMALL_DRAWS)
    assert CFG["jobs"]["c1a"] == ["c1_gsm8k", "gsm8k"] and CFG["jobs"]["c1b"] == ["c1_mmlu_pro", "mmlu_pro_cot"]
    for key, plans in CFG["plan"].items():
        for c, names in plans.items():
            assert set(names) <= set(CFG["jobs"]) and names[:2] == ["c1a", "c1b"], (key, c)
    assert CFG["draws"]["c1_gsm8k"]["n"] == "all" and CFG["draws"]["c1_mmlu_pro"]["n"] == 1000
    assert all(CFG["draws"][f"cheap_{s}"]["n"] == 150 for s in ("gsm8k", "mmlu_pro"))
    assert all(CFG["draws"][f"battery_{s}"] == {"set": s, "n": 250, "rule": "battery"}
               for s in ("gsm8k", "mmlu_pro", "arc_challenge"))
    reg = CFG["registered"]
    assert reg["c1a"]["band"] == [86, 96] and reg["c1b"]["band"] == [51, 61]
    assert [a["value"] for a in reg["c1a"]["anchors"]] == [91.6, 91.13]
    assert [a["value"] for a in reg["c1b"]["anchors"]] == [56.3, 56.2]
    assert CFG["checks"]["determinism"] == {"job": "c1a", "min_share": 0.95}
    sha = CFG["items"]["sha256"]
    assert sha is None or (len(sha) == 64 and int(sha, 16) >= 0)


# ---------------------------------------------------------------------------------------- statistics, grades


def test_wilson_known_values():
    assert card.wilson(0, 10) == [0.0, 27.75]
    assert card.wilson(10, 10) == [72.25, 100.0]
    assert card.wilson(5, 10) == [23.66, 76.34]
    assert card.wilson(3, 0) is None
    lo, hi = card.wilson(1200, 1319)
    assert lo < 100 * 1200 / 1319 < hi and hi - lo < 3.2


def test_bands_compare_exactly():
    assert not card.in_band(1134, 1319, [86, 96]) and card.in_band(1135, 1319, [86, 96])  # 85.97 and 86.05
    assert card.in_band(1266, 1319, [86, 96]) and not card.in_band(1267, 1319, [86, 96])  # 95.98 and 96.06
    assert card.in_band(510, 1000, [51, 61]) and card.in_band(610, 1000, [51, 61])


def cot_rows(n=1000, correct=560, rescued=0, unparsed=0, cut=0, cut_parsed=0):
    """n reasoning-format rows: `correct` right under Inspect's parse; then `rescued` with no ANSWER line that
    TIGER-Lab's fallback reads right; then `unparsed` with nothing to read; then `cut` replies cut at the limit with
    nothing to read; then `cut_parsed` cut at the limit after a wrong ANSWER line; the rest wrong."""
    rows = []
    for i in range(n):
        if i < correct:
            r = {"correct": True, "pred": "C", "tiger_pred": "C", "tiger_correct": True, "hit_cap": False}
        elif i < correct + rescued:
            r = {"correct": False, "pred": None, "tiger_pred": "C", "tiger_correct": True, "hit_cap": False}
        elif i < correct + rescued + unparsed:
            r = {"correct": False, "pred": None, "tiger_pred": None, "tiger_correct": False, "hit_cap": False}
        elif i < correct + rescued + unparsed + cut:
            r = {"correct": False, "pred": None, "tiger_pred": None, "tiger_correct": False, "hit_cap": True}
        elif i < correct + rescued + unparsed + cut + cut_parsed:
            r = {"correct": False, "pred": "A", "tiger_pred": "A", "tiger_correct": False, "hit_cap": True}
        else:
            r = {"correct": False, "pred": "A", "tiger_pred": "A", "tiger_correct": False, "hit_cap": False}
        rows.append({"set": "mmlu_pro", "id": f"mmlu_pro/{i}", "gold": "C", "text": "", "new_tokens": 100, **r})
    return rows


def test_c1b_grades_pass_format_and_fail():
    band = [51, 61]
    assert card.grade_c1b(card.summarize(cot_rows(correct=560)), band) == {"verdict": "pass"}
    # 48.0 strict, 53.0 with the fallbacks, 60 lost replies (6 points): the 3-point gap is covered
    g = card.grade_c1b(card.summarize(cot_rows(correct=480, rescued=50, unparsed=5, cut=5)), band)
    assert g["verdict"] == "explained_as_format" and g["gap_to_band_points"] == 3.0
    assert g["tiger_in_band"] and g["lost_points"] == 6.0 and g["lost_cover_gap"]
    assert g["lost_at_parsed_acc_points"] == round(100 * 60 * (480 / 940) / 1000, 2)  # 3.06: reported, not graded
    # a fallback score in the band that the lost replies could not have produced (only possible by hand)
    s = card.summarize(cot_rows(correct=480, rescued=0, unparsed=20))
    s["tiger"]["correct"] = 520
    g = card.grade_c1b(s, band)
    assert g["verdict"] == "fail" and not g["lost_cover_gap"]
    # many replies lost, but the fallbacks stay under the band
    g = card.grade_c1b(card.summarize(cot_rows(correct=400, rescued=50, unparsed=100)), band)
    assert g["verdict"] == "fail" and not g["tiger_in_band"]
    assert card.grade_c1b(card.summarize(cot_rows(correct=620)), band)["verdict"] == "fail"  # above the band
    s = card.summarize(cot_rows(correct=480, rescued=50, unparsed=5, cut=5, cut_parsed=7))
    assert (s["acc"], s["tiger"]["acc"], s["tiger"]["rescued"], s["hit_cap"], s["unparsed"]) == (48.0, 53.0, 50, 12, 60)
    # cut replies that gave a letter are graded on it and are not lost: no double count
    assert s["lost"] == {"replies": 60, "points": 6.0, "cut": 5, "not_cut": 55}
    assert s["cut_but_parsed"] == {"replies": 7, "correct": 0} and s["parsed_acc"] == round(100 * 480 / 940, 2)


def test_a_fallback_score_in_the_band_implies_the_cover():
    """TIGER-Lab's fallback reads only replies Inspect's parse finds no letter in, so the fallback score exceeds the
    strict one by at most the lost replies: whenever it lies in the band after a miss below it, they cover the gap."""
    rng = random.Random(0)
    item = {"set": "mmlu_pro", "gold": "C", "n_options": 10}
    texts = ["Reasoning.\nANSWER: C", "Reasoning.\nANSWER: D", "So the answer is (C).", "So the answer is (B).",
             "I am not sure.", "Final answer: C", "Answer: (C) is my pick"]
    checked = 0
    for _ in range(3000):
        n, weights = rng.randint(20, 120), [rng.random() for _ in texts]
        rows = []
        for i in range(n):
            text = rng.choices(texts, weights)[0]
            rows.append({"set": "mmlu_pro", "id": f"mmlu_pro/{i}", "gold": "C", "text": text, "new_tokens": 9,
                         "hit_cap": rng.random() < 0.05, **br.score_reply("mmlu_pro_cot", item, text)})
        s = card.summarize(rows)
        assert s["tiger"]["correct"] - s["correct"] <= s["lost"]["replies"]
        g = card.grade_c1b(s, [51, 61])
        if g.get("tiger_in_band") and 100 * s["correct"] < 51 * n:
            assert g["lost_cover_gap"] and g["verdict"] == "explained_as_format"
            checked += 1
    assert checked > 50


@pytest.mark.parametrize("a,b,calibrated,by,stops", [
    ("pass", "pass", True, ["C1a", "C1b"], False),
    ("pass", "fail", True, ["C1a"], True),
    ("fail", "pass", True, ["C1b"], True),
    ("pass", "explained_as_format", True, ["C1a"], False),
    ("fail", "explained_as_format", False, [], True),
    ("fail", "fail", False, [], True),
])
def test_the_verdict(a, b, calibrated, by, stops):
    v = card.overall(a, b)
    assert (v["load_path_calibrated"], v["calibrated_by"], v["stops_remaining_checks"]) == (calibrated, by, stops)
    if (a, b) == ("pass", "fail"):
        assert v["provisional"] == ["C1b's parse"]
    if b == "explained_as_format" and a == "pass":
        assert v["unused"] == ["C1b's parse"] and v["provisional"] == []


# --------------------------------------------------------------------------- the battery rerun and the contrast


def battery_rows(n=250, correct=None, text="reply {i}"):
    correct = correct or {"gsm8k": 234, "mmlu_pro": 89, "arc_challenge": 221}
    return {s: [{"set": s, "id": f"{s}/{i}", "correct": i < correct[s], "pred": "A", "gold": "A",
                 "text": text.format(i=i) * 100} for i in range(n)] for s in ("gsm8k", "mmlu_pro", "arc_challenge")}


def test_battery_rerun_within_two_points(tmp_path):
    stored = battery_rows()
    (tmp_path / "base.json").write_text(json.dumps({"rows": [dict(r, text=r["text"][:800]) for v in stored.values()
                                                             for r in v]}))
    same = card.battery_rerun(battery_rows(), tmp_path)
    assert same["verdict"].startswith("reproduces") and same["gsm8k"]["identical_share"] == 1.0
    five = card.battery_rerun(battery_rows(correct={"gsm8k": 239, "mmlu_pro": 84, "arc_challenge": 221},
                                           text="other {i}"), tmp_path)
    assert five["gsm8k"]["within"] and five["mmlu_pro"]["within"] and five["verdict"].startswith("reproduces")
    assert five["gsm8k"]["diff_points"] == 2.0 and five["gsm8k"]["identical_prefixes"] == 0
    six = card.battery_rerun(battery_rows(correct={"gsm8k": 240, "mmlu_pro": 89, "arc_challenge": 221}), tmp_path)
    assert not six["gsm8k"]["within"] and six["verdict"].startswith("misses")
    other = battery_rows()
    other["arc_challenge"][0]["id"] = "arc_challenge/999"
    assert card.battery_rerun(other, tmp_path)["arc_challenge"]["status"].startswith("the items differ")
    assert card.battery_rerun(battery_rows(), tmp_path / "nothing")["status"].startswith("no base.json")


def write_job(results, model, condition, job, rows):
    d = card.run_dir(results, model, condition)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{job}.json").write_text(json.dumps({"rows": rows}))


def gsm_rows(n, wrong):
    return [{"set": "gsm8k", "id": f"gsm8k/{i}", "correct": i not in wrong, "pred": "1", "gold": "1", "text": "",
             "new_tokens": 9, "hit_cap": False} for i in range(n)]


@pytest.mark.parametrize("helpful_wrong,needed", [(range(0, 2), False), (range(0, 4), True)])
def test_contrast_between_ministrals_conditions(tmp_path, helpful_wrong, needed):
    model = CFG["models"]["ministral"]["id"]
    for c, wrong in (("default", range(0)), ("helpful", helpful_wrong)):
        write_job(tmp_path, model, c, "c1a", gsm_rows(100, set(wrong)))
        write_job(tmp_path, model, c, "c1b", cot_rows(100, correct=56))
        write_job(tmp_path, model, c, "letters", cot_rows(100, correct=40))
    sec = card.ministral_section(CFG, tmp_path)
    assert sec["contrast"]["gaps"]["c1a"]["points"] == len(helpful_wrong)  # default minus helpful
    assert sec["contrast"]["gaps"]["c1b"]["points"] == 0.0
    assert sec["contrast"]["third_condition_needed"] is needed
    assert sec["conditions"]["empty"]["status"] == "not run" and sec["anchor"] is None
    assert sec["conditions"]["default"]["letters_cost_points"]["points"] == 16.0  # c1b minus letters


# ------------------------------------------------------------------------------------------ the app, stubbed


@pytest.fixture
def app():
    stub, _ = load_apps(only=())
    base = stub.module.App

    class App(base):  # records each function's options
        def function(self, **k):
            deco = super().function(**k)

            def wrap(f):
                fn = deco(f)
                fn.options = k
                return fn
            return wrap

    stub.module.App = App
    stub.module.Retries = lambda **k: ("Retries", k)
    try:
        mod = load("model_check_app", CHECK / "app.py")

        class Spawned:  # as on Modal, a spawned call runs (and fails) when its result is asked for
            def __init__(self, run):
                self.run = run

            def get(self):
                return self.run()

        type(mod.evaluate).spawn = lambda self, *a, **k: Spawned(lambda: self.f(*a, **k))
        yield stub, mod
    finally:
        unload_apps()


def test_app_images_and_functions(app):
    stub, a = app
    assert a.app.name == "anatta-model-check" and set(a.app.functions) == {"sample", "keep_items", "evaluate"}
    ev, sa, ke = a.evaluate.options, a.sample.options, a.keep_items.options
    assert ev["gpu"] == "L40S" and set(ev["volumes"]) == {"/cache", "/runs"} and ev["timeout"] == 4 * 3600
    assert ev["retries"] == ("Retries", {"max_retries": 1, "initial_delay": 30.0}) and "image" not in ev
    assert sa["image"] is a.draw_image and "gpu" not in sa and set(sa["volumes"]) == {"/cache"}
    assert ke["image"] is a.draw_image and "gpu" not in ke and set(ke["volumes"]) == {"/runs"}
    starts = [i for i, c in enumerate(stub.image_calls) if c[0] == "debian_slim"]
    assert len(starts) == 2  # modal_app's image, then the draw's
    draw = stub.image_calls[starts[1]:]
    assert [c[0] for c in draw] == ["debian_slim", "uv_pip_install", "env", "add_local_dir"]  # the package added last
    assert draw[1][1] == ("huggingface_hub==2.3.0", "hf_xet==1.7.0", "pyarrow==21.0.0")
    main_pins = set(stub.image_calls[starts[0] + 1][1])
    assert {"huggingface_hub==2.3.0", "hf_xet==1.7.0"} <= main_pins  # the same pins as the run's image
    assert draw[2][1][0]["HF_HOME"] == "/cache/hf" and draw[3][1][1] == "/root/anatta"


def small_item_file(n=40):
    return bi.build_item_file(synthetic_rows(n), SMALL_DRAWS, seed=0)


def test_plans_batches_tokens_and_checks(app):
    _, a = app
    f = small_item_file()
    p = a.make_plan(CFG, "ministral", "default", f, "ab" * 32)
    by = {j["name"]: j for j in p["jobs"]}
    assert [j["name"] for j in p["jobs"]] == CFG["plan"]["ministral"]["default"]
    assert (by["c1b"]["batch"], by["c1b"]["max_new"]) == (25, 2048) and (by["c1a"]["batch"], by["c1a"]["max_new"]) == (50, 512)
    assert (by["letters"]["batch"], by["letters"]["max_new"]) == (50, 24) and by["c1a"]["ids"] == f["draws"]["c1_gsm8k"]
    assert p["determinism"] == {"job": "c1a", "min_share": 0.95} and p["system"] is None and p["bos"] == 1
    assert p["batch1"] == {"c1a": f["draws"]["batch1_gsm8k"], "c1b": f["draws"]["batch1_mmlu_pro"]}
    h = a.make_plan(CFG, "ministral", "helpful", f, "ab" * 32)
    assert h["batch1"] is None and h["system"] == "You are a helpful assistant."
    q = a.make_plan(CFG, "qwen", "default", f, "ab" * 32)
    assert {j["batch"] for j in q["jobs"]} == {50} and q["bos"] == 0
    assert a.plan_id(p) == a.plan_id(json.loads(json.dumps(p))) and len(a.plan_id(p)) == 12
    assert a.plan_id(a.make_plan(CFG, "ministral", "default", f, "ab" * 32, gpu="A100")) != a.plan_id(p)
    assert a.out_dir(q).startswith("/runs/h4lite/Qwen__Qwen2.5-7B-Instruct/model_check/default/")
    assert a.out_dir(p) == f"/runs/h4lite/model_check/default/{a.plan_id(p)}"  # Ministral is modal_app's default model
    need = a.needed_items(f, p)
    assert set(need) == {i for j in p["jobs"] for i in j["ids"]}  # the batch-1 items sit inside C1's


def test_the_plan_fingerprints_the_code_and_the_pins(app, monkeypatch):
    _, a = app
    f = small_item_file()
    p = a.make_plan(CFG, "qwen", "default", f, "ab" * 32)
    assert p["code"] == fp.code_hashes(a.PKG) and set(p["code"]) == set(fp.CODE_FILES)
    assert {"player.py", "bench/fingerprint.py"} <= set(p["code"])  # load_model calls player; run_plan, fingerprint
    assert p["container_sha256"] == a.container_sha256()
    assert p["pins"]["torch"] == "2.14.1" and p["pins"]["transformers"] == "5.21.0" and len(p["pins"]) >= 10
    edited = {**p["code"], "bench/battery.py": "0" * 64}
    monkeypatch.setattr(a, "code_hashes", lambda pkg: edited)
    assert a.plan_id(a.make_plan(CFG, "qwen", "default", f, "ab" * 32)) != a.plan_id(p)
    monkeypatch.setattr(a, "code_hashes", fp.code_hashes)
    monkeypatch.setattr(a, "image_pins", lambda path: {**p["pins"], "transformers": "5.22.0"})
    assert a.plan_id(a.make_plan(CFG, "qwen", "default", f, "ab" * 32)) != a.plan_id(p)
    monkeypatch.setattr(a, "image_pins", fp.image_pins)
    monkeypatch.setattr(a, "container_sha256", lambda: "0" * 64)  # evaluate or out_dir edited
    assert a.plan_id(a.make_plan(CFG, "qwen", "default", f, "ab" * 32)) != a.plan_id(p)


def test_the_plan_hashes_only_the_code_of_app_py_that_runs_in_the_container(app):
    _, a = app
    src = Path(a.__file__).read_text()
    got = a.container_source(src)
    assert got.startswith("@app.function(gpu=GPU") and "def evaluate(plan: dict, items: dict)" in got
    assert got.rstrip().endswith("{plan_id(plan)}\"") and "def out_dir(plan: dict)" in got and "def draw(" not in got
    base = a.container_sha256(src)
    assert base == a.container_sha256() == bi.sha256_bytes(got.encode())

    def edit(old, new):
        assert src.count(old) == 1, old
        return a.container_sha256(src.replace(old, new))

    # the laptop's code (the draw, the gate, the checks, this file's other lines) leaves the plan as it was
    assert edit('print(f"wrote {path}", flush=True)', 'print(f"wrote: {path}", flush=True)') == base
    assert edit("def qwen_for_gate(", "# moved\ndef qwen_for_gate(") == base
    assert edit("TIMEOUT_HOURS = 4", "TIMEOUT_HOURS = 4  # an edited comment") == base
    # what runs in the container makes a new plan: evaluate's body, its decorator, out_dir
    assert edit('os.environ["HF_HUB_OFFLINE"] = "1"', 'os.environ["HF_HUB_OFFLINE"] = "0"') != base
    assert edit("cpu=8.0, memory=32768", "cpu=8.0, memory=65536") != base
    assert edit('/model_check/{plan[\'condition\']}', '/model_check2/{plan[\'condition\']}') != base


def test_image_pins_reads_modal_apps_image(tmp_path):
    src = tmp_path / "m.py"
    src.write_text('import modal\nimage = (modal.Image.debian_slim().uv_pip_install("a==1", "b==2.0")\n'
                   '         .env({"X": "y"}).uv_pip_install("c==3"))\nimage = image.add_local_dir("x", "y")\n')
    assert fp.image_pins(src) == {"a": "1", "b": "2.0", "c": "3"}
    src.write_text("image = None\n")
    with pytest.raises(ValueError, match="no name==version pins"):
        fp.image_pins(src)


def test_results_folder_guard(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(SystemExit, match="git work tree"):
        card.results_path(repo / "results" / "c1")
    stored = tmp_path / "battery"
    stored.mkdir()
    with pytest.raises(SystemExit, match="battery-results"):
        card.results_path(stored / "c1", stored)
    out = card.results_path(tmp_path / "out" / "c1", stored)
    assert out.is_dir()


def test_item_file_hash_gate_battery_items_and_allocation(app, tmp_path):
    _, a = app
    f = small_item_file()
    (tmp_path / "items.json").write_bytes(bi.dump_items(f))
    sha = bi.sha256_bytes(bi.dump_items(f))
    with pytest.raises(SystemExit, match="register it"):
        a.load_items(tmp_path, {"items": {"file": "items.json", "sha256": None}})
    with pytest.raises(SystemExit, match="not the registered"):
        a.load_items(tmp_path, {"items": {"file": "items.json", "sha256": "0" * 64}})
    got, got_sha = a.load_items(tmp_path, {"items": {"file": "items.json", "sha256": sha}})
    assert got_sha == sha and got["draws"] == f["draws"]
    with pytest.raises(SystemExit, match="draw first"):
        a.load_items(tmp_path / "none", {"items": {"file": "items.json", "sha256": sha}})
    battery = tmp_path / "battery"
    battery.mkdir()
    stored = [{"set": s, "id": i, "prompt": f["items"][i]["prompts"][s], "gold": f["items"][i]["gold"], "of": 40}
              for s in ("gsm8k", "mmlu_pro", "arc_challenge") for i in f["draws"][f"battery_{s}"]]
    (battery / "items.json").write_text(json.dumps(stored))
    a.check_battery_items(f, str(battery))
    stored[13]["prompt"] += " "
    (battery / "items.json").write_text(json.dumps(stored))
    with pytest.raises(SystemExit, match="battery_mmlu_pro.*position 1"):
        a.check_battery_items(f, str(battery))
    drawn = {"meta": {"draws": {"c1_mmlu_pro": {"allocation": dict(C1), "info": {"tie_at_cut": False}}}}}
    a.check_allocation(drawn, CFG)
    drawn["meta"]["draws"]["c1_mmlu_pro"]["allocation"].update(math=111, history=33)
    with pytest.raises(SystemExit, match="not the registered"):
        a.check_allocation(drawn, CFG)
    with pytest.raises(SystemExit, match="tie at the cut"):  # the synthetic categories are all the same size
        a.check_allocation(f, {"allocation": {"c1_mmlu_pro": f["meta"]["draws"]["c1_mmlu_pro"]["allocation"]}})


def stand_in_hub(tmp_path, monkeypatch, rows) -> list:
    """Stand-in huggingface_hub and pyarrow.parquet modules over synthetic rows (GSM8K split over two files listed out
    of order, MMLU-Pro and ARC in one each, beside files of other splits). Returns the calls they saw."""
    listing = {"openai/gsm8k": ["main/test-00001-of-00002.parquet", "README.md", "main/train-00000-of-00001.parquet",
                                "main/test-00000-of-00002.parquet"],
               "TIGER-Lab/MMLU-Pro": ["data/test-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
               "allenai/ai2_arc": ["ARC-Challenge/test-00000-of-00001.parquet", "ARC-Easy/test-00000-of-00001.parquet"]}
    content = {"main/test-00000-of-00002.parquet": rows["gsm8k"][:25], "main/test-00001-of-00002.parquet": rows["gsm8k"][25:],
               "data/test-00000-of-00001.parquet": rows["mmlu_pro"], "ARC-Challenge/test-00000-of-00001.parquet": rows["arc_challenge"]}
    seen = []

    class HfApi:
        def dataset_info(self, repo, revision=None):
            seen.append(("info", repo, revision))
            return types.SimpleNamespace(sha="e" * 40)

        def list_repo_files(self, repo, repo_type=None, revision=None):
            assert repo_type == "dataset"
            seen.append(("list", repo, revision))
            return listing[repo]

    def hf_hub_download(repo, filename, repo_type=None, revision=None):
        seen.append(("get", repo, filename, revision))
        path = tmp_path / "hub" / repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content[filename]))
        return str(path)

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi, hub.hf_hub_download = HfApi, hf_hub_download
    pq = types.ModuleType("pyarrow.parquet")
    pq.read_table = lambda p: types.SimpleNamespace(to_pylist=lambda: json.loads(Path(p).read_text()))
    pa = types.ModuleType("pyarrow")
    pa.parquet = pq
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "pyarrow", pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)
    return seen


def test_sample_draws_from_the_pinned_parquet_files(app, tmp_path, monkeypatch):
    _, a = app
    rows = synthetic_rows(40)
    seen = stand_in_hub(tmp_path, monkeypatch, rows)
    vol = tmp_path / "volume" / "items.json"
    monkeypatch.setattr(a, "ITEMS_ON_VOLUME", str(vol))
    got = a.sample.remote(UNREGISTERED["datasets"], SMALL_DRAWS, 0)
    assert set(got) == {"item_file", "requested"} and not vol.exists()  # the volume's copy is keep_items' business
    f = got["item_file"]
    meta = f["meta"]["datasets"]
    assert got["requested"]["arc_challenge"] == "main" and meta["arc_challenge"]["revision"] == "e" * 40
    assert meta["gsm8k"]["revision"] == CFG["datasets"]["gsm8k"]["revision"] and meta["gsm8k"]["rows"] == 40
    assert [x["path"] for x in meta["gsm8k"]["files"]] == ["main/test-00000-of-00002.parquet", "main/test-00001-of-00002.parquet"]
    assert meta["gsm8k"]["files"][0]["sha256"] == bi.sha256_file(tmp_path / "hub" / "openai/gsm8k" / "main/test-00000-of-00002.parquet")
    assert all(x[-1] == CFG["datasets"]["mmlu_pro"]["revision"] for x in seen if x[1] == "TIGER-Lab/MMLU-Pro")
    assert [x for x in seen if x[0] == "info"] == [("info", "allenai/ai2_arc", "main")]  # only the unpinned set
    assert f == bi.build_item_file(rows, SMALL_DRAWS, 0, meta)  # rows in sorted-file order
    # pinning ARC to the commit "main" resolved to gives the same item file, byte for byte
    pinned = {**UNREGISTERED["datasets"], "arc_challenge": {**UNREGISTERED["datasets"]["arc_challenge"], "revision": "e" * 40}}
    assert bi.dump_items(a.sample.remote(pinned, SMALL_DRAWS, 0)["item_file"]) == bi.dump_items(f)


def test_keep_items_writes_the_registered_draw_or_a_first_one(app, tmp_path, monkeypatch):
    _, a = app
    vol = tmp_path / "volume" / "items.json"
    monkeypatch.setattr(a, "ITEMS_ON_VOLUME", str(vol))
    one, two = b"one draw\n", b"another draw\n"
    sha1, sha2 = bi.sha256_bytes(one), bi.sha256_bytes(two)
    # nothing registered: the first draw writes the copy, and a later one leaves it alone
    assert a.keep_items.remote(one) == {"path": str(vol), "sha256": sha1, "written": True}
    assert a.keep_items.remote(two) == {"path": str(vol), "sha256": sha1, "written": False} and vol.read_bytes() == one
    # registered: the draw that reproduces it replaces a stale copy (no lockout), and writes one where there is none
    assert a.keep_items.remote(two, sha2) == {"path": str(vol), "sha256": sha2, "written": True}
    assert a.keep_items.remote(two, sha2)["written"] is False and vol.read_bytes() == two  # already there
    vol.unlink()
    assert a.keep_items.remote(two, sha2)["written"] is True and vol.read_bytes() == two
    # a draw that is not the registered one changes nothing (draw has refused it before this point anyway)
    assert a.keep_items.remote(one, sha2) == {"path": str(vol), "sha256": sha2, "written": False}
    assert vol.read_bytes() == two


def small_cfg(a, monkeypatch, cfg):
    mod = a.load_card()
    monkeypatch.setattr(mod, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(a, "load_card", lambda: mod)
    return mod


def test_draw_prints_the_sha256_and_the_resolved_arc_commit(app, tmp_path, monkeypatch, capsys):
    _, a = app
    stand_in_hub(tmp_path, monkeypatch, synthetic_rows(40))
    vol = tmp_path / "volume" / "items.json"
    monkeypatch.setattr(a, "ITEMS_ON_VOLUME", str(vol))
    cfg = json.loads(json.dumps(UNREGISTERED))
    cfg["draws"], cfg["allocation"] = SMALL_DRAWS, {}
    small_cfg(a, monkeypatch, cfg)
    out = tmp_path / "results"
    a.draw(str(out))
    text = capsys.readouterr().out
    sha = bi.sha256_bytes((out / "items.json").read_bytes())
    assert f"items.sha256: {sha}" in text and f"datasets.arc_challenge.revision: {'e' * 40}" in text
    assert "datasets.gsm8k.revision" not in text and "commit before the run" in text
    assert "no --battery-results" in text and vol.read_bytes() == (out / "items.json").read_bytes()
    # registered: a draw that differs leaves the local file and the volume's copy as they were
    cfg["items"]["sha256"] = "0" * 64
    (out / "items.json").write_text("kept")
    with pytest.raises(SystemExit, match="not the registered"):
        a.draw(str(out))
    assert (out / "items.json").read_text() == "kept" and bi.sha256_bytes(vol.read_bytes()) == sha
    cfg["items"]["sha256"] = sha
    a.draw(str(out))
    text = capsys.readouterr().out  # the sha256 is registered, ARC's commit not yet
    assert "put datasets.arc_challenge.revision into config.yaml" in text and "they match" not in text
    cfg["datasets"]["arc_challenge"]["revision"] = "e" * 40  # pinning ARC leaves the item file's bytes as they were
    a.draw(str(out))
    assert bi.sha256_bytes((out / "items.json").read_bytes()) == sha and "they match config.yaml" in capsys.readouterr().out


def draw_setup(a, tmp_path, monkeypatch, rows=None):
    """The stand-in hub over synthetic rows, a volume in tmp_path, and a config with the small draws, nothing
    registered. Returns (cfg, the results folder, the volume's item-file path)."""
    stand_in_hub(tmp_path, monkeypatch, rows or synthetic_rows(40))
    vol = tmp_path / "volume" / "items.json"
    monkeypatch.setattr(a, "ITEMS_ON_VOLUME", str(vol))
    cfg = json.loads(json.dumps(UNREGISTERED))
    cfg["draws"], cfg["allocation"] = SMALL_DRAWS, {}
    small_cfg(a, monkeypatch, cfg)
    return cfg, tmp_path / "results", vol


def test_draw_compares_the_battery_draw_with_the_stored_run_before_any_sha256(app, tmp_path, monkeypatch, capsys):
    _, a = app
    cfg, out, vol = draw_setup(a, tmp_path, monkeypatch)
    a.draw(str(out))
    f = json.loads((out / "items.json").read_text())
    battery = write_battery(tmp_path / "battery", f, lambda text: "Answer: 1")
    capsys.readouterr()
    a.draw(str(out), str(battery))  # the stored run's own items: the draw goes through
    text = capsys.readouterr().out
    assert "the battery draw is the stored run's items" in text and "items.sha256: " in text
    assert "no --battery-results" not in text
    # ARC moved on since the stored run: the draw stops before its sha256 is printed, and writes nothing
    stored = json.loads((battery / "items.json").read_text())
    arc = [k for k, x in enumerate(stored) if x["set"] == "arc_challenge"]
    stored[arc[3]]["prompt"] += " (an edit upstream)"
    (battery / "items.json").write_text(json.dumps(stored))
    (out / "items.json").write_text("kept")
    copy = vol.read_bytes()
    with pytest.raises(SystemExit, match="battery_arc_challenge: the draw differs from the stored run's items.json at "
                                         "position 3"):
        a.draw(str(out), str(battery))
    assert "items.sha256" not in capsys.readouterr().out
    assert (out / "items.json").read_text() == "kept" and vol.read_bytes() == copy


def test_a_redraw_before_registration_cannot_lock_the_draw_out(app, tmp_path, monkeypatch, capsys):
    _, a = app
    rows = synthetic_rows(40)
    cfg, out, vol = draw_setup(a, tmp_path, monkeypatch, rows)
    a.draw(str(out))  # nothing registered and no copy: this draw writes the volume's copy
    first = (out / "items.json").read_bytes()
    assert vol.read_bytes() == first
    moved = json.loads(json.dumps(rows))
    moved["gsm8k"][0]["question"] += " Again."
    stand_in_hub(tmp_path, monkeypatch, moved)  # a redraw before registering gives another item file
    capsys.readouterr()
    a.draw(str(out))
    second = (out / "items.json").read_bytes()
    assert second != first and vol.read_bytes() == first  # nothing registered, a copy there: left alone
    assert "is another draw's" in capsys.readouterr().out
    cfg["items"]["sha256"] = bi.sha256_bytes(second)  # the second draw is registered
    a.draw(str(out))  # and reproduced: the stale copy is replaced, so later draws are not locked out
    assert vol.read_bytes() == second == (out / "items.json").read_bytes()
    assert "is another draw's" not in capsys.readouterr().out
    a.draw(str(out))
    cfg["items"]["sha256"] = "0" * 64  # a draw that is not the registered one changes nothing
    with pytest.raises(SystemExit, match="not the registered"):
        a.draw(str(out))
    assert vol.read_bytes() == second == (out / "items.json").read_bytes()


# --------------------------------------------------------------------------------- whole runs, stubbed


C1B_RIGHT = 14  # of C1's 25 synthetic MMLU-Pro items: 56%


def answers(item_file, right_cot=C1B_RIGHT):
    """MockModel's answer callable: by the item and format behind the prompt, right or wrong on purpose. GSM8K: rows
    0, 10, 20, 30 wrong (C1a 90%); under "You are a helpful assistant." rows 1, 11, 21, 31 too (80%). MMLU-Pro with
    reasoning: the first `right_cot` of C1's items right (14: C1b 56%). Letters: even rows right. ARC: all right."""
    by_prompt = {p: (item, fmt) for item in item_file["items"].values() for fmt, p in item["prompts"].items()}
    right = set(item_file["draws"]["c1_mmlu_pro"][:right_cot])

    def answer(decoded: str) -> str:
        item, fmt = by_prompt[decoded.split("[INST]")[-1].split("[/INST]")[0]]
        helpful = "[SYS]You are a helpful assistant.[/SYS]" in decoded
        if fmt == "gsm8k":
            wrong = item["row"] % 10 == 0 or (helpful and item["row"] % 10 == 1)
            return f"Work.\nAnswer: {int(item['gold']) + wrong}"
        if fmt == "mmlu_pro_cot":
            return f"Step by step.\nANSWER: {item['gold'] if item['id'] in right else 'E'}"
        if fmt == "mmlu_pro":
            return item["gold"] if item["row"] % 2 == 0 else "E"
        return item["gold"]

    return answer


def write_battery(d: Path, f: dict, answer) -> Path:
    """A stored base run of the battery: its items.json, and base.json with the replies the mock gives with no system
    message, cut to 800 characters as the battery stored them."""
    d.mkdir()
    stored, rows = [], []
    for s in ("gsm8k", "mmlu_pro", "arc_challenge"):
        for i in f["draws"][f"battery_{s}"]:
            item = f["items"][i]
            stored.append({"set": s, "id": i, "prompt": item["prompts"][s], "gold": item["gold"], "of": 40})
            text = answer(f"[INST]{item['prompts'][s]}[/INST]")
            rows.append({"set": s, "id": i, **br.score_reply(s, item, text), "gold": item["gold"], "text": text[:800]})
    (d / "items.json").write_text(json.dumps(stored))
    (d / "base.json").write_text(json.dumps({"arm": "base", "rows": rows}))
    return d


def run_setup(a, tmp_path, monkeypatch, right_cot=C1B_RIGHT):
    """A results folder with a registered synthetic item file, a stored battery run, a volume in tmp_path, the
    container's installed versions equal to the image's pins, and load_model giving bench_mocks' model (a Qwen-like
    tokenizer without BOS, a Ministral-like one with it) answering by `answers`."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    f = small_item_file()
    out = tmp_path / "results"
    out.mkdir()
    (out / "items.json").write_bytes(bi.dump_items(f))
    cfg = json.loads(json.dumps(CFG))
    cfg["items"]["sha256"] = bi.sha256_bytes(bi.dump_items(f))
    small_cfg(a, monkeypatch, cfg)
    volume = tmp_path / "volume"
    monkeypatch.setattr(a, "out_dir", lambda plan: str(volume / plan["model"].replace("/", "__") / plan["condition"]
                                                     / a.plan_id(plan)))
    monkeypatch.setattr(br, "snapshot_dir", lambda model, revision, hf_home=None: f"/snapshots/{model}@{revision}")
    pins = fp.image_pins(a.PKG / "modal_app.py")
    monkeypatch.setattr(br, "installed", lambda names: {n: pins[n] for n in names})
    answer, loads = answers(f, right_cot), []

    def load_model(model=None, **k):
        loads.append(model)
        tok = MockTok(bos=False, default_system="You are Qwen, a mock.") if "Qwen" in model else MockTok(bos=True)
        return MockModel(tok, answer=answer), tok, None

    monkeypatch.setattr(a, "load_model", load_model)
    return types.SimpleNamespace(f=f, out=out, cfg=cfg, loads=loads, volume=volume, pins=pins,
                                 battery=write_battery(tmp_path / "battery", f, answer))


def test_a_whole_run_qwen_then_ministral_then_the_card(app, tmp_path, monkeypatch):
    _, a = app
    r = run_setup(a, tmp_path, monkeypatch)
    cfg, out = r.cfg, r.out
    a.main(str(out), str(r.battery))
    assert r.loads == [f"/snapshots/{m['id']}@{m['revision']}" for m in
                       (cfg["models"]["qwen"], *[cfg["models"]["ministral"]] * 3)]  # the third condition ran too
    c = json.loads((out / "card.json").read_text())
    q = c["qwen"]
    assert (q["c1a"]["acc"], q["c1a"]["verdict"], q["c1b"]["acc"], q["c1b"]["verdict"]) == (90.0, "pass", 56.0, "pass")
    assert q["verdict"]["load_path_calibrated"] and q["qwen_rows_pass"] and q["c1b"]["tiger"]["acc"] == 56.0
    assert (q["c1a"]["registered_odds"], q["c1b"]["registered_odds"]) == (0.85, 0.7)
    assert q["first_batch_twice"]["passed"] and set(q["batch1"]) == {"c1a", "c1b"}
    assert q["battery"]["arc_challenge"]["acc"] == 100.0 and q["letters"]["n"] == 25
    assert set(q["cheap_tier"]) >= {"cheap_gsm8k", "cheap_mmlu_pro_cot"}
    mi = c["ministral"]
    assert mi["gate"]["forced"] is False and mi["gate"]["qwen_stops_remaining_checks"] is False
    assert set(mi["conditions"]) == {"default", "helpful", "empty"} and mi["contrast"]["third_condition_needed"]
    assert mi["contrast"]["gaps"]["c1a"]["points"] == 10.0 and mi["contrast"]["third"]["c1a"]["points"] == 0.0
    assert mi["contrast"]["registered"]["odds"] == 0.7 and mi["contrast"]["registered"]["held"] is False
    assert mi["conditions"]["helpful"]["c1a"]["acc"] == 80.0 and mi["conditions"]["empty"]["system"] == ""
    rerun = mi["battery_rerun"]
    assert rerun["verdict"].startswith("reproduces") and all(rerun[s]["identical_share"] == 1.0 for s in
                                                             ("gsm8k", "mmlu_pro", "arc_challenge"))
    run = c["run"]
    assert set(run) == {"hooks", "qwen/default", "ministral/default", "ministral/helpful", "ministral/empty"}
    assert all(x["do_sample"] == ["False"] for k, x in run.items() if k != "hooks")
    assert run["ministral/default"]["batch"]["c1b"] == 25 and run["qwen/default"]["batch"]["c1b"] == 50
    item0 = run["ministral/default"]["item0"]["c1a"]
    assert item0["first_ids"][0] == 1 and item0["rendered_sha256"] == bi.sha256_bytes(item0["rendered"].encode())
    assert run["qwen/default"]["item0"]["c1a"]["first_ids"][0] != 1
    assert run["qwen/default"]["cache_refs_main"] is None and run["qwen/default"]["installed"]["torch"] == "2.14.1"
    h = c["hashes"]
    assert h["code_same_in_every_run"] and h["code_here_matches_run"] and h["item_file_same_everywhere"]
    assert set(h["code_as_run"]) == set(fp.CODE_FILES)
    for d in (r.volume / "Qwen__Qwen2.5-7B-Instruct" / "default").iterdir():
        assert {"plan.json", "run_info.json", "checks.json", "determinism.json", "batch1.json", "c1a.json"} <= {
            p.name for p in d.iterdir()}

    # everything is in --results: nothing is launched; with the local copy gone, the volume's is read back unloaded
    a.main(str(out), str(r.battery))
    assert len(r.loads) == 4
    qdir = card.run_dir(out, cfg["models"]["qwen"]["id"], "default")
    for p in qdir.iterdir():
        p.unlink()
    a.main(str(out), stages="qwen")
    assert len(r.loads) == 4 and (qdir / "c1a.json").exists()

    # a changed plan is refused rather than mixed with the files there
    cfg["max_new_tokens"]["gsm8k"] = 256
    with pytest.raises(SystemExit, match="another plan"):
        a.main(str(out), stages="qwen")


def test_an_edited_file_reruns_in_a_new_plan_and_is_never_regraded(app, tmp_path, monkeypatch):
    _, a = app
    r = run_setup(a, tmp_path, monkeypatch)
    a.main(str(r.out), stages="qwen")
    assert len(r.loads) == 1
    qwen = r.cfg["models"]["qwen"]["id"]
    first = json.loads((card.run_dir(r.out, qwen, "default") / "plan.json").read_text())
    real = fp.code_hashes(a.PKG)
    edited = {**real, "bench/battery.py": "0" * 64}
    # the laptop's code changed: the folder in --results holds another plan's run, and is refused
    monkeypatch.setattr(a, "code_hashes", lambda pkg: edited)
    with pytest.raises(SystemExit, match="another plan"):
        a.main(str(r.out), stages="qwen")
    # with a fresh --results the run plans afresh and runs again in a new volume folder; the old one is untouched
    monkeypatch.setattr(br, "code_hashes", lambda pkg: edited)  # the container runs the edited code too
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    shutil.copy(r.out / "items.json", fresh / "items.json")
    a.main(str(fresh), stages="qwen")
    assert len(r.loads) == 2
    second = json.loads((card.run_dir(fresh, qwen, "default") / "plan.json").read_text())
    assert second["code"] == edited and a.plan_id(second) != a.plan_id(first)
    assert sorted(p.name for p in (r.volume / "Qwen__Qwen2.5-7B-Instruct" / "default").iterdir()) == sorted(
        [a.plan_id(first), a.plan_id(second)])
    # a container whose code is not its plan's refuses, even with that plan's run complete on the volume
    monkeypatch.setattr(br, "code_hashes", lambda pkg: real)
    with pytest.raises(br.CheckError, match="differs from the plan's in bench/battery.py"):
        a.evaluate.remote(second, a.needed_items(r.f, second))
    # and so does one whose installed packages are not the image's pins
    monkeypatch.setattr(br, "code_hashes", lambda pkg: edited)
    monkeypatch.setattr(br, "installed", lambda names: {n: "2.13.0" if n == "torch" else r.pins[n] for n in names})
    with pytest.raises(br.CheckError, match=r"torch 2.13.0 \(pinned 2.14.1\)"):
        a.evaluate.remote(second, a.needed_items(r.f, second))
    assert len(r.loads) == 2


def test_the_ministral_stage_needs_the_stored_battery_run(app, tmp_path, monkeypatch):
    _, a = app
    r = run_setup(a, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="--battery-results is required"):
        a.main(str(r.out))
    assert r.loads == []
    a.main(str(r.out), stages="qwen")  # Qwen alone needs none
    assert len(r.loads) == 1


def test_the_gate_reads_only_a_qwen_run_of_this_plan(app, tmp_path, monkeypatch, capsys):
    _, a = app
    r = run_setup(a, tmp_path, monkeypatch)
    # no Qwen run in --results yet: under --stages ministral, Ministral waits
    a.main(str(r.out), str(r.battery), stages="ministral")
    assert r.loads == [] and "no complete Qwen run under this plan yet" in capsys.readouterr().out
    a.main(str(r.out), stages="qwen")
    assert len(r.loads) == 1
    # the code changed since Qwen ran: its rows are another plan's, and the gate refuses them, forced or not
    real = fp.code_hashes(a.PKG)
    monkeypatch.setattr(a, "code_hashes", lambda pkg: {**real, "bench/run.py": "0" * 64})
    for force in (False, True):
        with pytest.raises(SystemExit, match=r"the Qwen rows in .* are from another plan"):
            a.main(str(r.out), str(r.battery), stages="ministral", force_ministral=force)
    assert len(r.loads) == 1
    # under the plan Qwen ran with, the gate reads its verdict (both pass) and Ministral runs
    monkeypatch.setattr(a, "code_hashes", fp.code_hashes)
    a.main(str(r.out), str(r.battery), stages="ministral")
    assert len(r.loads) == 4
    c = json.loads((r.out / "card.json").read_text())
    assert c["ministral"]["status"] == "run" and c["ministral"]["gate"]["forced"] is False


def test_ministral_waits_while_a_fail_stops_the_checks(app, tmp_path, monkeypatch):
    _, a = app
    r = run_setup(a, tmp_path, monkeypatch, right_cot=5)  # C1a passes at 90%, C1b fails at 20%
    a.main(str(r.out), str(r.battery))
    assert len(r.loads) == 1
    c = json.loads((r.out / "card.json").read_text())
    assert (c["qwen"]["c1a"]["verdict"], c["qwen"]["c1b"]["verdict"]) == ("pass", "fail")
    assert c["qwen"]["verdict"]["load_path_calibrated"] and c["qwen"]["verdict"]["stops_remaining_checks"]
    assert c["ministral"]["status"] == "not run"
    assert c["ministral"]["gate"] == {"rule": "Ministral runs only if Qwen's verdict does not stop the remaining checks",
                                      "qwen_stops_remaining_checks": True, "forced": False}
    a.main(str(r.out), str(r.battery), force_ministral=True)  # the override runs it, and the card says so
    assert len(r.loads) == 4
    c = json.loads((r.out / "card.json").read_text())
    assert c["ministral"]["status"] == "run" and c["ministral"]["gate"]["forced"] is True


def test_neither_passing_stops_ministral_and_a_stopped_run_is_reported(app, tmp_path, monkeypatch):
    _, a = app
    r = run_setup(a, tmp_path, monkeypatch)
    loads = r.loads

    def wrong(model=None, **k):  # every reply wrong: neither C1a nor C1b passes
        loads.append(model)
        tok = MockTok(bos=False) if "Qwen" in model else MockTok(bos=True)
        return MockModel(tok, answer=lambda text: "Answer: -1"), tok, None

    monkeypatch.setattr(a, "load_model", wrong)
    a.main(str(r.out), str(r.battery))
    qwen = r.cfg["models"]["qwen"]
    assert loads == [f"/snapshots/{qwen['id']}@{qwen['revision']}"]  # Ministral waits
    c = json.loads((r.out / "card.json").read_text())
    assert c["qwen"]["verdict"]["load_path_calibrated"] is False and c["ministral"]["status"] == "not run"

    def sampling(model=None, **k):  # sampling left on: the first batch twice differs
        loads.append(model)
        tok = MockTok(bos=True)
        return MockModel(tok, sample=True), tok, None

    monkeypatch.setattr(a, "load_model", sampling)
    with pytest.raises(SystemExit, match="ministral default: CheckError.*sampling"):
        a.main(str(r.out), str(r.battery), force_ministral=True)
