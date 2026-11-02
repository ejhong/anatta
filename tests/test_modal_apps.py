"""The Modal apps on a laptop, under the stand-in `modal` of modal_stub.py: nothing here touches Modal, a GPU or a
real model.

  - The image: the pinned packages, this package at /root/anatta on the containers' PYTHONPATH, the trait directions
    when their folder exists, the two volumes and the control Dict by name. The entry modules use absolute imports
    only and load as top-level files, the way `modal run path/to/file.py` imports them, and everything they import
    from anatta is inside the package the image carries.
  - The paths: the default model's runs under /runs/h4lite and results/, any other model's under its slug.
  - load_model hands the model id and the LoRA options through (rank, alpha, rsLoRA; needs peft).
  - The SFT app's finetune: the cut entered only under --under-cut, with the model's v_ego; data.json's cut_finetune
    must agree; args.json, log.jsonl, done.json and the adapter written and copied; another model under its slug.
  - The two-adapter KL reference (needs peft): equal log-probs at step 1 (KL = 0), different from the base's, only
    "default" trainable, the reference unmoved by an optimizer step, and checkpoints holding "default" alone.
  - An EnvError ends one episode as env_error, and prepare_pairs drops such a pair up front.

The tests that need peft are skipped without it.
"""

import ast
import importlib.util
import json
import random
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from modal_stub import Tok, load_apps, tiny_llama, unload_apps

QWEN = "Qwen/Qwen2.5-7B-Instruct"
PINNED = ("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0", "numpy==2.5.4",
          "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0", "tiktoken==0.14.0", "jinja2==3.1.6",
          "regex==2026.9.29", "peft==0.21.2", "accelerate==1.15.0")
LORA_TARGETS = ["q_proj", "v_proj"]
ENTRY = {"train/sft_modal.py": {"finetune"},
         "evals/em_modal.py": {"stage", "extract_persona", "finetune", "sample_replies"}}


@pytest.fixture(autouse=True)
def _forget_the_stub():
    yield
    unload_apps()


def package_dir() -> Path:
    import anatta

    return Path(anatta.__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------------------- the image, the imports


def test_image_pins_mounts_the_package_and_names_the_volumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no ./directions here
    monkeypatch.delenv("ANATTA_DIRECTIONS", raising=False)
    stub, mods = load_apps()
    ma, sft, em = mods["anatta.modal_app"], mods["anatta.train.sft_modal"], mods["anatta.evals.em_modal"]
    assert ma.PKG == package_dir()
    assert stub.image_calls == [
        ("debian_slim", (), {"python_version": "3.12"}),
        ("uv_pip_install", PINNED, {}),
        ("env", ({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root"},), {}),
        ("add_local_dir", (str(package_dir()), "/root/anatta"), {"ignore": ["**/__pycache__"]}),
    ]
    assert stub.volumes == ["anatta-hf-cache", "anatta-runs"] and stub.dicts == ["anatta-h4lite-control"]
    assert (sft.app.name, em.app.name) == ("anatta-h4lite-sft", "anatta-h4lite-em")
    assert sft.app.image is ma.image and em.app.image is ma.image
    # a directions folder goes in at /root/directions: $ANATTA_DIRECTIONS, else ./directions where the app is launched
    d = tmp_path / "dirs"
    (d / ma.slug(ma.MODEL)).mkdir(parents=True)
    monkeypatch.setenv("ANATTA_DIRECTIONS", str(d))
    stub, _ = load_apps()
    assert stub.image_calls[-1] == ("add_local_dir", (str(d), "/root/directions"), {})
    monkeypatch.delenv("ANATTA_DIRECTIONS")
    (tmp_path / "directions").mkdir()
    stub, _ = load_apps()
    assert stub.image_calls[-1] == ("add_local_dir", ("directions", "/root/directions"), {})


def test_a_missing_directions_folder_warns_on_the_laptop_only(tmp_path, monkeypatch):
    nowhere = tmp_path / "nowhere"
    monkeypatch.setenv("ANATTA_DIRECTIONS", str(nowhere))
    with pytest.warns(UserWarning, match="no trait directions at .*nowhere.*will refuse to run"):
        stub, _ = load_apps(quiet=False)
    assert not any(c[0] == "add_local_dir" and c[1][1] == "/root/directions" for c in stub.image_calls)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        load_apps(is_local=False, quiet=False)  # in a container: the image's copy is read, nothing to warn about
        nowhere.mkdir()
        stub, _ = load_apps(quiet=False)  # the folder is there: the image adds it
    assert not [w for w in seen if "no trait directions" in str(w.message)]
    assert stub.image_calls[-1] == ("add_local_dir", (str(nowhere), "/root/directions"), {})


def test_require_ego_names_the_file_and_how_to_make_it(tmp_path, monkeypatch):
    stub, mods = load_apps(only=())
    ma = mods["anatta.modal_app"]
    monkeypatch.setattr(ma, "DIRECTIONS", tmp_path / "local")
    monkeypatch.setattr(ma, "REMOTE_DIRECTIONS", str(tmp_path / "remote"))
    assert ma.ego_file(QWEN) == tmp_path / "remote" / ma.slug(QWEN) / "ego.pt"
    assert ma.ego_file(QWEN, local=True) == tmp_path / "local" / ma.slug(QWEN) / "ego.pt"
    for local in (False, True):
        with pytest.raises(FileNotFoundError, match=f"no v_ego for {re.escape(QWEN)} at .*anatta extract"):
            ma.require_ego(QWEN, local)
        f = ma.ego_file(QWEN, local)
        f.parent.mkdir(parents=True)
        f.touch()
        assert ma.require_ego(QWEN, local) == f


@pytest.mark.parametrize("rel", sorted(ENTRY))
def test_entry_modules_load_as_files_and_import_only_what_the_image_carries(rel):
    """`modal run src/anatta/evals/em_modal.py::table` imports the file as a top-level module, and a container mounts
    that one file at /root/em_modal.py beside the image's /root/anatta: a relative import fails in both places, while
    an `anatta.` import resolves to the package on the laptop and to /root/anatta in a container."""
    pkg = package_dir()
    path = pkg / rel
    tree = ast.parse(path.read_text())
    relative = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level > 0]
    assert relative == [], f"{rel} has relative imports at lines {relative}"
    imported = ({n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
                | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
    load_apps(only=())
    for name in sorted(m for m in imported if m.split(".")[0] == "anatta"):  # module level and inside functions
        spec = importlib.util.find_spec(name)
        assert spec is not None and Path(spec.origin).resolve().is_relative_to(pkg), name
    third = {m.split(".")[0] for m in imported} - set(sys.stdlib_module_names) - {"anatta", "modal", "__future__"}
    assert third <= {"torch", "peft", "transformers"}, third  # all pinned in the image
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.app.functions) == ENTRY[rel]


# ---------------------------------------------------------------------------------------------------- the paths


def test_paths_keep_the_default_models_layout_and_put_any_other_under_its_slug(monkeypatch):
    monkeypatch.delenv("ANATTA_RESULTS", raising=False)
    stub, mods = load_apps()
    ma, em = mods["anatta.modal_app"], mods["anatta.evals.em_modal"]
    q = "Qwen__Qwen2.5-7B-Instruct"
    results = package_dir().parent.parent / "results"  # the checkout's own results/
    assert ma.MODEL == "mistralai/Ministral-3-8B-Instruct-2512-BF16" and ma.slug(QWEN) == q
    assert ma.run_root() == ma.ROOT == "/runs/h4lite" and ma.run_root(QWEN) == f"/runs/h4lite/{q}"
    assert ma.results_dir() == results and ma.results_dir(QWEN) == results / q
    assert ma.ego_dir() == ma.EGO == "/root/directions/mistralai__Ministral-3-8B-Instruct-2512-BF16"
    assert ma.ego_dir(QWEN) == f"/root/directions/{q}" and ma.DDIR == "directions/mistralai__Ministral-3-8B-Instruct-2512-BF16"
    assert em.em_root() == "/runs/h4lite/em" and em.persona_pt() == "/runs/h4lite/em/persona_direction.pt"
    assert em.out_dir() == results / "em"
    assert em.em_root(QWEN) == f"/runs/h4lite/{q}/em" and em.out_dir(QWEN) == results / q / "em"
    assert (ma.CACHE, ma.RUNS, ma.GPU, ma.MAX_NEW) == ("/cache", "/runs", "L40S", (512, 384))
    assert re.fullmatch(ma.LORA_TARGETS, "model.language_model.layers.0.mlp.down_proj")
    assert re.fullmatch(ma.LORA_TARGETS, "model.layers.31.self_attn.q_proj")
    assert not re.fullmatch(ma.LORA_TARGETS, "model.vision_tower.layers.0.attention.q_proj")
    assert not re.fullmatch(ma.LORA_TARGETS, "model.multi_modal_projector.linear_1")
    assert ma.volume_path("") is None and ma.volume_path("base") is None and ma.volume_path("/x/y") == "/x/y"
    assert ma.volume_path("h4lite/sft/koan_s0/adapter") == "/runs/h4lite/sft/koan_s0/adapter"
    assert ma.parse_max_new("512,384") == (512, 384) and ma.seeds_of("0, 1,2") == [0, 1, 2]
    assert ma.names_of("a, b,,c") == ["a", "b", "c"]


def test_results_go_to_the_checkouts_folder_wherever_the_app_is_launched(tmp_path, monkeypatch):
    """Launched from another project's folder that has a results/ of its own, the apps still write to the
    checkout's results/; $ANATTA_RESULTS moves them."""
    elsewhere = tmp_path / "another_project" / "experiments"
    (elsewhere / "results").mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("ANATTA_RESULTS", raising=False)
    stub, mods = load_apps()
    ma, sft, em = mods["anatta.modal_app"], mods["anatta.train.sft_modal"], mods["anatta.evals.em_modal"]
    assert ma.REPO == package_dir().parent.parent and ma.RESULTS == ma.REPO / "results" and ma.RESULTS.is_absolute()
    assert not ma.results_dir(QWEN).is_relative_to(elsewhere) and not em.out_dir().is_relative_to(elsewhere)
    monkeypatch.setenv("ANATTA_RESULTS", str(tmp_path / "out"))
    stub, mods = load_apps()
    ma, sft, em = mods["anatta.modal_app"], mods["anatta.train.sft_modal"], mods["anatta.evals.em_modal"]
    assert ma.RESULTS == tmp_path / "out" and em.out_dir(QWEN) == tmp_path / "out" / ma.slug(QWEN) / "em"
    sft.save_local("koan_s0", {"args.json": "{}"})
    em.save_local("insecure_none_s0", {"args.json": "{}"})
    assert (tmp_path / "out" / "sft" / "koan_s0" / "args.json").exists()
    assert (tmp_path / "out" / "em" / "train" / "insecure_none_s0" / "args.json").exists()
    assert list((elsewhere / "results").iterdir()) == []  # the launching folder's results/ untouched


def test_file_helpers(tmp_path):
    stub, mods = load_apps(only=())
    ma = mods["anatta.modal_app"]
    p = tmp_path / "a" / "log.jsonl"
    assert ma.read_jsonl(p) == []
    ma.write_jsonl(p, [{"x": 1}])
    ma.append_jsonl(p, {"x": 2})
    assert ma.read_jsonl(p) == [{"x": 1}, {"x": 2}] == ma.read_rows(p.read_text() + "\n\n")
    ma.write_json(tmp_path / "b" / "c.json", {"k": [1, 2]})
    assert json.loads((tmp_path / "b" / "c.json").read_text()) == {"k": [1, 2]}
    row = {"name": "p1", "spec": "def f(x):", "entry": "f", "tests": ["assert f(1) == 1"],
           "mutate": [0, "assert f(1) == 2", "oneoff"], "hidden": "[]", "reference": "def f(x):\n    return x\n",
           "split": "held", "extra": 1}
    pairs = ma.pairs_by_name(json.dumps(row) + "\n" + json.dumps({**row, "name": "p2", "split": "train"}) + "\n")
    assert list(pairs) == ["p1", "p2"] and pairs["p1"].tests == ("assert f(1) == 1",) and pairs["p1"].ref_seconds == 0.0
    assert list(ma.pairs_by_name(json.dumps(row) + "\n", split="train")) == []


def test_load_model_hands_the_model_and_the_lora_options_through(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    import anatta.models

    stub, mods = load_apps(only=())
    ma = mods["anatta.modal_app"]
    asked = []

    def fake_load(model_id, *a, **k):
        asked.append(model_id)
        return tiny_llama(), Tok()
    monkeypatch.setattr(anatta.models, "load", fake_load)

    m, tok, hf = ma.load_model()
    assert asked == [ma.MODEL] and m is hf and not m.training  # no adapter: the plain model, in eval mode
    m, tok, hf = ma.load_model(adapter="new", trainable=True, rank=4, model=QWEN)
    cfg = m.peft_config["default"]
    assert asked[-1] == QWEN and (cfg.r, cfg.lora_alpha, cfg.use_rslora) == (4, 8, False)  # alpha 2 x rank
    assert cfg.target_modules == ma.LORA_TARGETS
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert len(trainable) == 4 * 7 * 2 and all("lora_" in n for n in trainable)  # every projection of every layer
    q = m.base_model.model.model.layers[0].self_attn.q_proj
    assert q.scaling["default"] == pytest.approx(2.0)
    m, _, _ = ma.load_model(adapter="new", trainable=True, rank=4, alpha=6, use_rslora=True)
    cfg = m.peft_config["default"]
    assert (cfg.lora_alpha, cfg.use_rslora) == (6, True)
    assert m.base_model.model.model.layers[0].self_attn.q_proj.scaling["default"] == pytest.approx(6 / 4 ** 0.5)
    m, _, _ = ma.load_model(adapter="new", trainable=False, rank=4, alpha=6)
    assert not m.training and not any(p.requires_grad for p in m.parameters())
    assert m.base_model.model.model.layers[0].self_attn.q_proj.scaling["default"] == pytest.approx(6 / 4)


def test_lr_for_refuses_the_default_rate_for_a_reshaped_lora():
    stub, mods = load_apps(only=())
    lr_for = mods["anatta.modal_app"].lr_for
    assert lr_for(None, 1e-4) == 1e-4 and lr_for(3e-5, 1e-4) == 3e-5  # the default LoRA: the default rate, or the one given
    assert lr_for(1e-5, 5e-5, alpha=64, use_rslora=True) == 1e-5
    for kw in ({"alpha": 64}, {"use_rslora": True}, {"alpha": 64, "use_rslora": True}):
        with pytest.raises(SystemExit, match="pass --lr explicitly"):
            lr_for(None, 5e-5, **kw)


# ---------------------------------------------------------------------------------------------------- the SFT app


def write_training_set(folder: Path, info: dict, n_episodes: int = 6, seed: int = 0) -> int:
    """LABEL/{train.pt, data.json} as the SFT app reads them: episodes of one or two turns, token ids under the tiny
    Llama's 64. Returns the number of items (turns)."""
    import torch

    g = random.Random(seed)
    turns = [[([1] + [g.randrange(3, 64) for _ in range(5 + t)], [g.randrange(3, 64) for _ in range(3)] + [2])
              for t in range(1 + e % 2)] for e in range(n_episodes)]
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"records": [{"name": f"ep{e}", "outcome": "solved"} for e in range(n_episodes)], "turns": turns},
               folder / "train.pt")
    (folder / "data.json").write_text(json.dumps(info))
    return sum(len(t) for t in turns)


@pytest.fixture
def sft_app(tmp_path, monkeypatch):
    """The SFT app with its volume and results under tmp_path, the tiny Llama for the model (no LoRA, since peft may
    be missing: the whole model trains), a stand-in ego cut, and the dial's intervene() recording what enters it."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import anatta.dial
    import anatta.player

    stub, mods = load_apps(only=("anatta.train.sft_modal",))
    ma, s = mods["anatta.modal_app"], mods["anatta.train.sft_modal"]
    vol, results, dirs = tmp_path / "vol", tmp_path / "results", tmp_path / "directions"
    monkeypatch.setattr(ma, "ROOT", str(vol))
    monkeypatch.setattr(ma, "RESULTS", results)
    monkeypatch.setattr(ma, "DIRECTIONS", dirs)  # one folder for the laptop and the containers
    monkeypatch.setattr(ma, "REMOTE_DIRECTIONS", str(dirs))
    for m in (ma.MODEL, QWEN):  # the stand-in ego cut never reads them
        (dirs / ma.slug(m)).mkdir(parents=True)
        (dirs / ma.slug(m) / "ego.pt").touch()
    a = SimpleNamespace(s=s, ma=ma, vol=vol, results=results, loads=[], cuts=[], entered=[],
                        cut=[SimpleNamespace(layer=8), SimpleNamespace(layer=9)])

    def fake_load(adapter=None, trainable=False, rank=16, ref=None, model=ma.MODEL, alpha=None, use_rslora=False):
        a.loads.append(dict(adapter=adapter, trainable=trainable, rank=rank, model=model, alpha=alpha,
                            use_rslora=use_rslora))
        hf = tiny_llama()
        return hf, Tok(), hf

    def fake_save(model, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")

    def fake_ego_cut(ddir):
        a.cuts.append(str(ddir))
        return a.cut

    @contextmanager
    def fake_intervene(model, interventions):
        a.entered.append(list(interventions))
        yield

    monkeypatch.setattr(s, "load_model", fake_load)
    monkeypatch.setattr(anatta.player, "save_adapter", fake_save)
    monkeypatch.setattr(anatta.player, "ego_cut", fake_ego_cut)
    monkeypatch.setattr(anatta.dial, "intervene", fake_intervene)
    return a


COMMON = dict(lr=1e-2, rank=4, epochs=2, micro=2, batch=4)


def test_sft_finetune_enters_the_cut_only_under_under_cut(sft_app):
    a = sft_app
    sft = a.vol / "sft"
    n = write_training_set(sft / "koan", {"kept": 6, "cut_sampling": False})
    write_training_set(sft / "koan_glimpse", {"kept": 6, "cut_sampling": True, "cut_finetune": False})
    write_training_set(sft / "koan_cut", {"kept": 6, "cut_sampling": True, "cut_finetune": True})
    assert n == 9  # 4 + 4 + 1 items per epoch at batch 4: 3 optimizer steps an epoch

    a.s.main(labels="koan,koan_glimpse", seeds="0,1", **COMMON)
    assert a.entered == [] and a.cuts == []  # no cut asked for: no context at all
    for label, cut_sampling in (("koan", False), ("koan_glimpse", True)):
        for seed in (0, 1):
            out, local = sft / f"{label}_s{seed}", a.results / "sft" / f"{label}_s{seed}"
            args = json.loads((out / "args.json").read_text())
            assert args["cut_sampling"] is cut_sampling and args["cut_finetune"] is False
            assert args["items"] == n and args["episodes"] == 6 and args["seed"] == seed and args["lr"] == 1e-2
            assert args["data"] == json.loads((sft / label / "data.json").read_text())
            assert not {"src_dir", "out_dir", "init", "save_epochs", "model", "alpha", "use_rslora"} & set(args)
            log = read_jsonl(out / "log.jsonl")
            assert [r["opt_step"] for r in log] == [1, 2, 3, 4, 5, 6] and [r["epoch"] for r in log] == [1, 1, 1, 2, 2, 2]
            assert [r["items"] for r in log] == [4, 4, 1] * 2 and all(r["grad_norm"] > 0 and r["tokens"] > 0 for r in log)
            assert "lr" not in log[0]  # no schedule: the optimizer's rate throughout
            assert json.loads((out / "done.json").read_text())["opt_steps"] == 6
            assert (out / "adapter" / "adapter_config.json").exists() and not (out / "adapter_epoch1").exists()
            for f in ("args.json", "log.jsonl", "done.json"):
                assert (local / f).read_text() == (out / f).read_text()
    assert a.loads == [dict(adapter="new", trainable=True, rank=4, model=a.ma.MODEL, alpha=None, use_rslora=False)] * 4

    # under the cut: the model's v_ego, the context entered once per run, around the whole loop
    a.s.main(labels="koan_cut", seeds="0", under_cut=True, **COMMON)
    assert a.entered == [a.cut] and a.cuts == [a.ma.ego_dir(a.ma.MODEL)]
    args = json.loads((sft / "koan_cut_s0" / "args.json").read_text())
    assert args["cut_sampling"] is True and args["cut_finetune"] is True
    # a fine-tune asked for the other cut condition than its data.json records is refused, its files untouched
    before = (sft / "koan_glimpse_s0" / "args.json").read_text()
    with pytest.raises(SystemExit, match="cut_finetune=False"):
        a.s.finetune("koan_glimpse", 0, True)
    with pytest.raises(SystemExit, match="cut_finetune=True"):
        a.s.finetune("koan_cut", 1, False)
    assert (sft / "koan_glimpse_s0" / "args.json").read_text() == before and not (sft / "koan_cut_s1").exists()
    # data.json without cut_finetune takes either
    a.s.finetune("koan", 2, True, **COMMON)
    assert len(a.entered) == 2 and json.loads((sft / "koan_s2" / "args.json").read_text())["cut_finetune"] is True


def test_sft_finetune_for_another_model_and_with_its_options(sft_app, tmp_path):
    a = sft_app
    qroot = Path(a.ma.run_root(QWEN))
    assert qroot == a.vol / "Qwen__Qwen2.5-7B-Instruct"
    write_training_set(qroot / "sft" / "koan", {"kept": 6})
    a.s.main(labels="koan", seeds="0", model=QWEN, alpha=8, use_rslora=True, **{**COMMON, "epochs": 1})
    out = qroot / "sft" / "koan_s0"
    args = json.loads((out / "args.json").read_text())
    assert (args["model"], args["alpha"], args["use_rslora"]) == (QWEN, 8, True)
    assert a.loads == [dict(adapter="new", trainable=True, rank=4, model=QWEN, alpha=8, use_rslora=True)]
    local = a.results / "Qwen__Qwen2.5-7B-Instruct" / "sft" / "koan_s0"
    assert (local / "log.jsonl").read_text() == (out / "log.jsonl").read_text()
    assert not (a.vol / "sft").exists() and not (a.results / "sft").exists()  # the default model's folders untouched
    a.s.finetune("koan", 1, True, model=QWEN, **COMMON)
    assert a.cuts == [a.ma.ego_dir(QWEN)]  # the other model's v_ego
    # src_dir, out_dir, init and save_epochs: recorded, and an adapter after every epoch
    elsewhere = tmp_path / "elsewhere"
    a.s.finetune("x", 0, False, src_dir=str(qroot / "sft" / "koan"), out_dir=str(elsewhere), init="/runs/start/adapter",
                 save_epochs=True, **COMMON)
    args = json.loads((elsewhere / "args.json").read_text())
    assert (args["src_dir"], args["out_dir"], args["init"], args["save_epochs"]) == (
        str(qroot / "sft" / "koan"), str(elsewhere), "/runs/start/adapter", True)
    assert "model" not in args and a.loads[-1]["adapter"] == "/runs/start/adapter"
    assert all((elsewhere / d / "adapter_config.json").exists() for d in ("adapter_epoch1", "adapter_epoch2", "adapter"))


def test_sft_refuses_the_cut_without_v_ego(sft_app):
    a = sft_app
    other = "org/no-directions-1B"
    sft = Path(a.ma.run_root(other)) / "sft"
    write_training_set(sft / "koan", {"kept": 6})
    with pytest.raises(FileNotFoundError, match="no v_ego for org/no-directions-1B"):
        a.s.main(labels="koan", seeds="0", under_cut=True, model=other, **COMMON)
    with pytest.raises(FileNotFoundError, match="no v_ego for org/no-directions-1B"):
        a.s.finetune("koan", 0, True, model=other, **COMMON)
    assert a.loads == [] and a.cuts == [] and not (sft / "koan_s0").exists()  # refused before anything was written or loaded
    a.s.main(labels="koan", seeds="0", model=other, **COMMON)  # without the cut it needs no v_ego
    assert (sft / "koan_s0" / "done.json").exists() and a.entered == []


def test_sft_refuses_the_default_lr_for_a_reshaped_lora(sft_app):
    a = sft_app
    sft = a.vol / "sft"
    write_training_set(sft / "koan", {"kept": 6})
    shape = dict(rank=4, epochs=1, micro=2, batch=4)
    for kw in ({"alpha": 8}, {"use_rslora": True}):
        with pytest.raises(SystemExit, match="pass --lr explicitly"):
            a.s.main(labels="koan", seeds="0", **shape, **kw)
        with pytest.raises(SystemExit, match="pass --lr explicitly"):
            a.s.finetune("koan", 0, **shape, **kw)
    assert a.loads == [] and not (sft / "koan_s0").exists()  # refused before anything ran
    a.s.main(labels="koan", seeds="0", **shape)  # the default LoRA keeps the default rate
    assert a.s.LR == 1e-4 and json.loads((sft / "koan_s0" / "args.json").read_text())["lr"] == 1e-4
    a.s.main(labels="koan", seeds="1", lr=2e-5, alpha=8, **shape)  # a reshaped LoRA with its rate given
    assert json.loads((sft / "koan_s1" / "args.json").read_text())["lr"] == 2e-5


# ------------------------------------------------------------------------------------------- the KL reference


def test_reference_adapter_is_the_start_and_only_default_trains(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from anatta.player import (
        REF,
        TRAIN,
        attach_adapters,
        pack,
        reference,
        save_adapter,
        token_logps,
    )

    torch.manual_seed(0)
    m0 = attach_adapters(tiny_llama(), "new", None, True, 4, LORA_TARGETS)
    for n, p in m0.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.5)
    save_adapter(m0, tmp_path / "start")
    model = attach_adapters(tiny_llama(), str(tmp_path / "start"), str(tmp_path / "start"), True, 4, LORA_TARGETS)
    assert set(model.peft_config) == {TRAIN, REF} and model.active_adapter == TRAIN
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(f".{TRAIN}." in n for n in trainable) and not any(f".{REF}." in n for n in trainable)
    full, attn, cm, wc = pack([([1, 2, 3, 4], [5, 6, 2]), ([1, 7], [8, 9, 10, 2])], 0, "cpu")
    model.train()
    with torch.no_grad(), reference(model):
        ref = token_logps(model, full, attn, wc)
    lp = token_logps(model, full, attn, wc)
    d = ref - lp.detach()
    assert torch.allclose(lp, ref, atol=1e-6) and float(((torch.exp(d) - d - 1) * cm).sum()) == pytest.approx(0.0, abs=1e-6)
    with torch.no_grad(), model.disable_adapter():
        base = token_logps(model, full, attn, wc)
    assert not torch.allclose(base, ref, atol=1e-3)  # the reference is the start adapter, not the untrained base
    assert {n for n, p in model.named_parameters() if p.requires_grad} == trainable and model.active_adapter == TRAIN
    # an optimizer step moves "default" and leaves the reference where it was
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    (-(lp * cm).sum()).backward()
    opt.step()
    with torch.no_grad(), reference(model):
        ref_after = token_logps(model, full, attn, wc)
    assert torch.allclose(ref_after, ref, atol=1e-6)
    assert not torch.allclose(token_logps(model, full, attn, wc).detach(), lp.detach(), atol=1e-6)
    # checkpoints hold "default" alone, in the root layout every evaluation app reads
    save_adapter(model, tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "adapter_config.json").exists() and not (tmp_path / "ckpt" / REF).exists()
    # a fresh LoRA has no "ref": its reference is the base model, which it equals at step 1
    m2 = attach_adapters(tiny_llama(), "new", None, True, 4, LORA_TARGETS)
    with torch.no_grad(), reference(m2):
        r2 = token_logps(m2, full, attn, wc)
    with torch.no_grad(), m2.disable_adapter():
        b2 = token_logps(m2, full, attn, wc)
    assert torch.allclose(r2, b2)


# ------------------------------------------------------------------------------------------------- env errors


def test_env_error_is_recorded_not_raised_and_dropped_up_front(monkeypatch):
    import anatta.models
    from anatta.koans.env import KoanTask, Pair
    from anatta.player import Player, episode_record

    stub, mods = load_apps(only=())
    prepare_pairs = mods["anatta.modal_app"].prepare_pairs

    # a reference that fails as a whole (at module level): the harness reports an error instead of outputs, twice,
    # and reference_outputs raises EnvError. (One that raises inside f reports a per-input error and the twin is an
    # honest near_miss: that is the model's problem, not the environment's.)
    boom = Pair("boom", "def f(x):", "f", ("assert f(1) == 1",), (0, "assert f(1) == 2", "oneoff"), "[(2,), (3,)]",
                "raise RuntimeError('boom')\ndef f(x):\n    return x\n")
    fine = Pair("fine", "def g(x):", "g", ("assert g(1) == 2",), (0, "assert g(1) == 3", "oneoff"), "[(2,), (3,)]",
                "def g(x):\n    return x + 1\n")

    class Canned(Player):
        def __init__(self):
            self.tok, self.dev, self.max_new, self.gen_batch = SimpleNamespace(pad_token_id=0, eos_token_id=2), "cpu", (8, 8), 4

        def generate(self, prompts, temp, max_new):
            return [[7, 2] for _ in prompts], ["```python\ndef f(x):\n    return x\n```" for _ in prompts]

    monkeypatch.setattr(anatta.models, "chat_ids", lambda tok, msgs, **k: [[1, 2, 3]])
    envs = [KoanTask(boom, False, "neutral", True, 3), KoanTask(boom, True, "neutral", True, 3)]
    turns, texts, infos = Canned().play(envs, 1.0)
    recs = [episode_record(e, i, t, x) for e, i, t, x in zip(envs, infos, turns, texts)]
    assert recs[0]["outcome"] == "env_error" and "boom" in recs[0]["error"] and recs[0]["turn"] == 1
    assert recs[0]["first_fail"] is False
    assert recs[1]["outcome"] == "failed"  # the impossible twin never reaches the hidden tests
    kept, info = prepare_pairs({"boom": boom, "fine": fine}, workers=2)
    assert list(kept) == ["fine"] and info["dropped"] == ["boom"] and set(info["ref_seconds"]) == {"fine"}
