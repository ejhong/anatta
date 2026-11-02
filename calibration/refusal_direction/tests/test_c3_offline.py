"""The offline loads (refusal.offline_loads, offline_check, offline_failures): the smoke of 1 Nov stopped when
judge_lab could not load the judge's tokenizer offline. These tests reproduce that failure on the laptop's transformers
(the lab image's version) with a made-up repo in a fake cache, check that the guard runs the loads with HF_HUB_OFFLINE
set, and check that the guard's list names every tokenizer and config the stages load: by the names the stages' own
loaders are called with, and by a scan of the container code for loader calls. No Modal, no network: every child
process runs offline with the hub's endpoint set to a closed local port."""

import ast
import json
import subprocess
import types

import pytest
from c3_testing import CHECK, PKG, REPO, config

# isort: split
import judge
import refusal

MADE_UP_REPO = "made-up-org/made-up-adapter"
MADE_UP_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def no_network(home):
    """The child's environment: the fake cache, and the hub's endpoint at a closed local port, so that nothing could
    leave the laptop even if offline mode were off."""
    return {"HF_HOME": str(home), "HF_HUB_CACHE": str(home / "hub"), "HF_ENDPOINT": "http://127.0.0.1:9"}


def adapter_repo(home):
    """A made-up adapter repo as snapshot_download at "main" leaves it in the cache: a tokenizer's files and
    adapter_config.json, no config.json (a PEFT adapter's repo has none), and refs/main. Nothing records an absence."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.train_from_iterator(["the quick brown fox jumps over the lazy dog"] * 20,
                          trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<eos>", "<bos>"],
                                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    snap = refusal.snapshot_dir(MADE_UP_REPO, MADE_UP_COMMIT, str(home))
    PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<eos>", bos_token="<bos>").save_pretrained(snap)
    (snap / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "made-up-org/made-up-base",
                                                          "peft_type": "LORA"}))
    (snap.parent.parent / "refs").mkdir()
    (snap.parent.parent / "refs" / "main").write_text(MADE_UP_COMMIT)
    return snap


def test_the_judges_tokenizer_by_repo_id_loads_offline_only_once_the_cache_records_config_json_absent(tmp_path):
    """The smoke's failure, reproduced: offline, transformers 5.21's AutoTokenizer resolves a model config first, and
    by repo id it cannot tell the config.json the adapter's repo lacks from one never fetched, so the load fails; by
    the snapshot's path (prepare's call before the fix) it loads; once the cache records the absence (what an online
    lookup by repo id writes, and prepare's load now makes), it loads by repo id too."""
    snap = adapter_repo(tmp_path)
    loads = [{"stages": ["judge_lab"], "load": "judge_tokenizer", "name": MADE_UP_REPO},
             {"stages": ["judge_lab"], "load": "judge_tokenizer", "name": str(snap)}]
    by_id, by_path = refusal.offline_check(loads, env=no_network(tmp_path))
    assert not by_id["ok"] and by_id["error"].startswith("OSError: We couldn't connect")
    assert by_path["ok"] and by_path["vocab"] > 0
    assert refusal.absent_files(MADE_UP_REPO, MADE_UP_COMMIT, str(tmp_path)) == []
    marker = refusal.repo_cache(MADE_UP_REPO, str(tmp_path)) / ".no_exist" / MADE_UP_COMMIT / "config.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    assert refusal.absent_files(MADE_UP_REPO, MADE_UP_COMMIT, str(tmp_path)) == ["config.json"]
    by_id, by_path = refusal.offline_check(loads, env=no_network(tmp_path))
    assert by_id["ok"] and by_id["vocab"] == by_path["vocab"]
    assert refusal.ref_commit(MADE_UP_REPO, "main", str(tmp_path)) == MADE_UP_COMMIT


def test_offline_check_runs_the_loads_in_a_fresh_python_with_hf_hub_offline_set(monkeypatch):
    """The flag is read when the hub library is imported, so the loads run in a child started with it set, whatever
    the caller's environment says."""
    seen = {}

    def run(args, input, capture_output, text, env, timeout):
        seen.update(args=args, input=input, env=env)
        recs = [{**x, "ok": True} for x in json.loads(input)]
        return types.SimpleNamespace(returncode=0, stdout="a warning\n" + refusal.OFFLINE_MARK + json.dumps(recs))

    monkeypatch.setattr(subprocess, "run", run)
    loads = [{"stages": ["judge_lab"], "load": "config", "name": "made-up-org/made-up-base"}]
    assert refusal.offline_check(loads, env={"HF_HUB_OFFLINE": "0"}) == [{**loads[0], "ok": True}]
    assert seen["env"]["HF_HUB_OFFLINE"] == "1" and seen["args"][1:] == ["-c", "import refusal; refusal.offline_main()"]
    assert json.loads(seen["input"]) == loads and str(CHECK) in seen["env"]["PYTHONPATH"].split(":")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(refusal.CheckError, match="exited with code 1 and no record"):
        refusal.offline_check(loads)


def test_offline_failures_flag_a_failed_load_a_wrong_base_and_a_record_that_does_not_list_the_loads():
    plan = {"judge": config()["judge"], "models": config()["models"]}
    base = plan["judge"]["base"]["id"]
    ok = [{**x, "ok": True, **({"base": base} if x["load"] == "adapter_config" else {})}
          for x in refusal.offline_loads(plan, "lab")]
    assert refusal.offline_failures(ok, plan, "lab") == []
    failed = [dict(r, ok=False, error="OSError: made-up") if r["load"] == "judge_tokenizer" else r for r in ok]
    assert refusal.offline_failures(failed, plan, "lab") == [
        f"judge_tokenizer {judge.ADAPTER} (judge_lab): OSError: made-up"]
    wrong = [dict(r, base="made-up-org/other-base") if r["load"] == "adapter_config" else r for r in ok]
    assert refusal.offline_failures(wrong, plan, "lab") == [
        f"the adapter names the base made-up-org/other-base, not the registered {base}"]
    for bad in ([], None, ok[:-1], ok[::-1]):
        assert refusal.offline_failures(bad, plan, "lab") == [
            "the lab stack's offline record does not list the loads the later stages make"]
    assert refusal.offline_failures(ok, plan, "paper") != []  # the paper stack's loads are another list


# ---------------------------------------------------------------------------- the list covers what the stages load


class Loaded:
    """What a stood-in from_pretrained returns: enough of a model and a tokenizer for the loaders' own lines."""
    pad_token, eos_token, padding_side = "<pad>", "<eos>", "right"

    def eval(self):
        return self

    def to(self, *a, **k):
        return self

    def requires_grad_(self, *a):
        return self

    def parameters(self):
        return iter([types.SimpleNamespace(dtype="torch.bfloat16")])


def test_offline_loads_name_every_tokenizer_and_config_the_stages_load(tmp_path, monkeypatch):
    """Each stage's own loader is called with the names the stage passes, transformers' from_pretrained stood in for
    and recording (kind, name). Every tokenizer it loads must be in that stack's offline_loads, and every model load
    must have its config there: the snapshot's config, or for the adapter's id its adapter config and the config of
    the base it names. judge_paper's and swap's original package loads the judge with the port's calls by the same
    id (judge.py's docstring); config.yaml's adapter id is judge.ADAPTER (app.code_settings)."""
    import transformers

    from anatta import models

    monkeypatch.setenv("HF_HOME", str(tmp_path))  # snapshot_dir reads it, as in the containers
    seen = []

    def recorder(kind):
        def from_pretrained(name, *a, **k):
            seen.append((kind, str(name)))
            return Loaded()
        return from_pretrained

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", recorder("tokenizer"))
    for cls in (transformers.AutoModelForCausalLM, transformers.AutoModelForImageTextToText):
        monkeypatch.setattr(cls, "from_pretrained", recorder("model"))
    cfg = config()
    plan = {"judge": cfg["judge"], "models": cfg["models"]}
    snap = {k: refusal.snapshot_dir(m["id"], m["revision"]) for k, m in cfg["models"].items()}
    stages = {("judge_lab", "lab"): lambda: judge.load_judge(),  # app.judge_lab
              ("run_model", "lab"): lambda: [models.load(str(snap[k])) for k in ("qwen", "ministral")],  # load_model
              ("judge_paper", "paper"): lambda: judge.load_judge(),  # refusal.judge_paper's port
              ("swap", "paper"): lambda: refusal.load_paper_model(snap["qwen"])}  # refusal.swap's load
    base = cfg["judge"]["base"]["id"]
    for (stage, stack), call in stages.items():
        seen.clear()
        call()
        assert seen, stage
        loads = [x for x in refusal.offline_loads(plan, stack) if stage in x["stages"]]
        for kind, name in seen:
            if kind == "tokenizer":
                assert any(x["load"] in ("tokenizer", "judge_tokenizer") and x["name"] == name for x in loads), (
                    stage, name)
            else:
                assert any(x["load"] == "config" and x["name"] == name for x in loads) or (
                    any(x["load"] == "adapter_config" and x["name"] == name for x in loads)
                    and any(x["load"] == "config" and x["name"] == base for x in loads)), (stage, name)
    assert judge.ADAPTER == cfg["judge"]["adapter"]["id"]


LOADERS = {"from_pretrained", "hf_hub_download", "snapshot_download", "load_model", "load_judge",
           "load_judge_tokenizer", "load_paper_model", "load_tokenizer", "evaluate_dataset", "load"}


def loader_calls(path, only=None):
    """(file, top-level function, callee) for every call to a loader in a source file; a bare load() counts (swap's
    loader argument), an attribute .load() does not (json.load, torch.load read local files)."""
    out = []
    for fn in ast.parse(path.read_text()).body:
        if not isinstance(fn, ast.FunctionDef) or (only is not None and fn.name not in only):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name in LOADERS and not (isinstance(f, ast.Attribute) and name == "load"):
                    out.append((path.name, fn.name, name))
    return sorted(out)


# Every loader call in the code that runs in the containers, and where offline_loads covers it. A new call fails the
# test below until it is added here and, if a later stage makes it offline, to offline_loads.
COVERED = sorted([
    ("app.py", "run_model", "load_model"),  # run_model (lab): the models' snapshot folders; tokenizer, config
    ("app.py", "judge_lab", "load_judge"),  # judge_lab (lab): judge.ADAPTER; judge_tokenizer, adapter_config, config
    ("modal_app.py", "load_model", "load"),  # anatta.models.load, the same snapshot folder
    ("models.py", "load", "from_pretrained"),  # its tokenizer, and the model (its config) by the same name
    ("models.py", "load", "from_pretrained"),
    ("models.py", "load", "from_pretrained"),
    ("judge.py", "load_judge", "from_pretrained"),  # the judge by the adapter's id (and again, without the dtype)
    ("judge.py", "load_judge", "from_pretrained"),
    ("judge.py", "load_judge", "load_judge_tokenizer"),
    ("judge.py", "load_judge_tokenizer", "from_pretrained"),
    ("refusal.py", "judge_paper", "load_judge"),  # judge_paper (paper): the port, as judge_lab
    ("refusal.py", "original_scores", "evaluate_dataset"),  # judge_paper and swap (paper): the original, same id
    ("refusal.py", "swap", "load"),  # swap (paper): load_paper_model on Qwen's snapshot folder
    ("refusal.py", "load_paper_model", "from_pretrained"),
    ("refusal.py", "load_paper_model", "from_pretrained"),
    ("refusal.py", "hf_file", "hf_hub_download"),  # prepare, online: the data files
    ("refusal.py", "hf_snapshot", "snapshot_download"),  # prepare, online: the judge's two repos
    ("refusal.py", "load_tokenizer", "from_pretrained"),  # prepare, online: the models' snapshot folders
    ("refusal.py", "prepare", "load_judge_tokenizer"),  # prepare, online: the judge's tokenizer by its repo id
    ("refusal.py", "load_offline", "load_judge_tokenizer"),  # the guard itself
    ("refusal.py", "load_offline", "from_pretrained"),
    ("refusal.py", "load_offline", "from_pretrained"),
    ("refusal.py", "load_offline", "from_pretrained"),
])


def container_functions():
    """app.py's CONTAINER_FUNCTIONS, read off its source (no Modal import)."""
    for n in ast.parse((CHECK / "app.py").read_text()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "CONTAINER_FUNCTIONS" for t in n.targets):
            return set(ast.literal_eval(n.value))
    raise AssertionError("app.py has no CONTAINER_FUNCTIONS")


def test_every_loader_call_in_the_container_code_is_one_the_offline_list_covers():
    assert {"prepare", "prepare_paper", "run_model", "judge_lab", "judge_paper", "swap"} <= container_functions()
    found = (loader_calls(CHECK / "app.py", only=container_functions())
             + loader_calls(CHECK / "refusal.py") + loader_calls(CHECK / "reference.py")
             + loader_calls(CHECK / "judge.py") + loader_calls(PKG / "models.py")
             + loader_calls(PKG / "modal_app.py", only={"load_model"}))
    assert sorted(found) == COVERED
    assert (REPO / "src" / "anatta") == PKG
