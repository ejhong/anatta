"""app.py under the modal stand-in (tests/modal_stub.py): the two images as defined (nothing is built), the secret,
the volumes, the registered v_ego, the run's folder, the plan and the settings checked at plan time, the container
hash, the spend ledger, and main's order of stages with every Modal function stood in for. No Modal call of any
kind."""

import json
import re
import shlex
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from c3_testing import (  # first: puts the check folder on sys.path
    CHECK,
    PKG,
    config,
    load_app,
    unload_app,
)

# isort: split
import judge
import refusal


@pytest.fixture
def app():
    stub, mod = load_app()
    yield stub, mod
    unload_app()


def calls_named(stub, name):
    return [(i, a, k) for i, (n, a, k) in enumerate(stub.image_calls) if n == name]


def images(stub):
    """The image calls split at each debian_slim: anatta.modal_app's image, then the lab's, then the paper's."""
    out = []
    for n, a, k in stub.image_calls:
        if n == "debian_slim":
            out.append([])
        out[-1].append((n, a, k))
    return out


def test_the_app_its_functions_secret_and_volumes(app):
    stub, mod = app
    assert mod.app.name == "anatta-calib-refusal-direction"
    assert set(mod.app.functions) == {"prepare", "prepare_paper", "run_model", "judge_lab", "judge_paper", "swap",
                                      "stored"}
    assert stub.secrets == ["huggingface"]
    with_secret = {n for n, k in stub.function_kwargs.items() if k.get("secrets")}
    assert with_secret == {"prepare"}
    assert set(stub.volumes) == {"anatta-hf-cache", "anatta-runs"}
    assert not any(v.startswith("matlab") for v in stub.volumes)
    paper = {n for n, k in stub.function_kwargs.items() if k.get("image") is mod.paper_image}
    assert paper == {"prepare_paper", "judge_paper", "swap"}
    cpu = ("prepare", "prepare_paper", "stored")
    assert all(k.get("gpu") == "L40S" for n, k in stub.function_kwargs.items() if n not in cpu)
    assert not any(stub.function_kwargs[n].get("gpu") for n in cpu)


def test_the_paper_image_pins_its_stack_and_installs_strong_reject_without_dependencies(app):
    stub, mod = app
    assert mod.PAPER_PINS["transformers"] == "4.47.0" and mod.PAPER_PINS["torch"] == "2.5.1"
    assert mod.PAPER_PINS["accelerate"] == "1.2.0"
    commit = config()["judge"]["package_commit"]
    assert mod.STRONG_REJECT.endswith("@" + commit) and len(commit) == 40
    (apt,) = calls_named(stub, "apt_install")
    uv = [c for c in calls_named(stub, "uv_pip_install") if c[1] == tuple(mod.pip(mod.PAPER_PINS))]
    (pipi,) = calls_named(stub, "pip_install")
    assert apt[1] == ("git",) and len(uv) == 1
    assert pipi[1] == (mod.STRONG_REJECT,) and pipi[2] == {"extra_options": "--no-deps"}
    build, image_env = [c for c in calls_named(stub, "env") if c[1] and c[1][0].get("READTHEDOCS") == "1"]
    assert build[1][0] == mod.PAPER_BUILD_ENV and build[1][0]["HF_HUB_OFFLINE"] == "1"
    assert image_env[1][0] == {**mod.ENV, "HF_HUB_OFFLINE": "1", "READTHEDOCS": "1"}
    (run,) = calls_named(stub, "run_commands")
    assert run[1] == (mod.TEMPLATE_INSTALLED, "python -c 'import strong_reject.evaluate'", mod.MOUNTS_ABSENT)
    order = [apt[0], uv[0][0], pipi[0], build[0], run[0], image_env[0]]
    assert order == sorted(order)
    paper = images(stub)[-1]
    assert [a[1] for n, a, _ in paper if n == "add_local_file"] == ["/root/refusal.py", "/root/reference.py",
                                                                    "/root/judge.py", "/root/fingerprint.py"]


def filled_paths(steps) -> set[str]:
    """The paths one image fills: those it adds, and every absolute path in the environment each build step runs
    with (the image's ENV so far, the step's env, the command's own VAR=value words), where the step may write."""
    env, out = {}, set()
    for n, a, k in steps:
        if n == "env":
            env.update(a[0])
        elif n in ("add_local_file", "add_local_dir"):
            out.add(a[1])
        elif n == "run_commands":
            for cmd in a:
                run_env = {**env, **(k.get("env") or {})}
                for word in shlex.split(cmd):
                    if not re.fullmatch(r"[A-Za-z_]\w*=.*", word):
                        break
                    run_env.update([word.split("=", 1)])
                out |= {p for v in run_env.values() for p in v.split(":") if p.startswith("/")}
    return out


def test_no_image_fills_a_path_its_functions_mount(app):
    """The g3 smoke (1 Nov): prepare_paper stopped in Modal with 'cannot mount volume on non-empty path: "/cache"'.
    The paper image's build imported strong_reject.evaluate with HF_HOME=/cache/hf, the import wrote there, and Modal
    will not mount a volume over a path the image fills (judge_paper and swap mount the cache the same way). For every
    function: no path its image adds, and no path in a build step's environment, is at or under a path the function
    mounts. The paper's functions mount what judge_paper mounts, and the paper image's last build step checks, in
    Modal, that no mount path exists."""
    stub, mod = app
    lab, paper = images(stub)[-2:]
    steps = {id(mod.lab_image): lab, id(mod.paper_image): paper}
    assert mod.app.image is mod.lab_image
    for name, k in stub.function_kwargs.items():
        filled = filled_paths(steps[id(k.get("image", mod.app.image))])
        for mount in k.get("volumes", {}):
            assert not [p for p in filled if p == mount or p.startswith(mount.rstrip("/") + "/")], (name, mount)
    g3 = [("env", ({**mod.ENV, "READTHEDOCS": "1"},), {}),
          ("run_commands", ("python -c 'import strong_reject.evaluate'",), {})]
    assert f"{mod.CACHE}/hf" in filled_paths(g3)  # the image g3 ran: its build had HF_HOME on the cache's mount path
    assert filled_paths([("env", ({"HF_HOME": "/cache/hf"},), {}),
                         ("run_commands", ("HF_HOME=/elsewhere/hf python -c pass",), {})]) == {"/elsewhere/hf"}
    paper_fns = [n for n, k in stub.function_kwargs.items() if k.get("image") is mod.paper_image]
    assert paper_fns and all(stub.function_kwargs[n]["volumes"] == stub.function_kwargs["judge_paper"]["volumes"]
                             for n in paper_fns)
    (run,) = calls_named(stub, "run_commands")
    assert run[1][-1] == mod.MOUNTS_ABSENT
    assert all(f"'{m}'" in mod.MOUNTS_ABSENT for n in paper_fns for m in stub.function_kwargs[n]["volumes"])


def test_the_paper_images_mount_check_passes_with_no_mount_path_and_stops_on_one(app, tmp_path):
    """MOUNTS_ABSENT, run here as the build runs it (sh -c), on two stand-in mount paths under tmp_path."""
    _, mod = app
    cmd = mod.MOUNTS_ABSENT.replace("python", shlex.quote(sys.executable), 1)
    for m in (mod.CACHE, mod.RUNS):
        cmd = cmd.replace(f"'{m}'", f"'{tmp_path}{m}'")
    assert subprocess.run(["sh", "-c", cmd], capture_output=True).returncode == 0
    (tmp_path / "cache" / "hf" / "hub").mkdir(parents=True)
    r = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True)
    assert r.returncode != 0 and "fills a volume mount path" in r.stderr and f"{tmp_path}/cache" in r.stderr


def test_every_file_app_py_loads_at_import_is_in_both_images(app):
    """A container imports app.py too, and Modal mounts only that file: whatever it loads at import beside itself
    must be in the images (fingerprint.py, whose plan_id out_dir calls in every container)."""
    import ast

    stub, mod = app
    loaded = {c.args[1].right.value for node in ast.parse((CHECK / "app.py").read_text()).body
              if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) for c in ast.walk(node)
              if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "module_from_file"
              and isinstance(c.args[1], ast.BinOp) and isinstance(c.args[1].right, ast.Constant)}
    assert loaded == {"fingerprint.py"}
    for image in images(stub)[-2:]:
        assert {f"/root/{f}" for f in loaded} <= {a[1] for n, a, _ in image if n == "add_local_file"}


def test_the_plan_hashes_every_check_file_the_images_carry(app):
    """Fable's fidelity re-check, finding 3 (1 Nov): fingerprint.py runs in both containers, so it is in the plan's
    code hashes and each container checks it against the plan; an edit to it makes a new plan."""
    _, mod = app
    assert set(refusal.CHECK_FILES) == set(mod.CHECK_CODE) and "fingerprint.py" in refusal.CHECK_FILES
    assert "fingerprint.py" in refusal.code_hashes(PKG)


def test_the_lab_image_carries_the_registered_v_ego_alone_where_ego_file_looks(app):
    """Ruling (1 Nov): the one registered ego.pt, at anatta.modal_app.ego_file's path for Ministral, and never the
    whole directions folder (added only if the file is on this machine)."""
    stub, mod = app
    from anatta.modal_app import ego_file

    cfg = config()
    src = Path(cfg["ego"]["file"])
    where = str(ego_file(cfg["models"][cfg["ego"]["model"]]["id"]))
    assert where == "/root/directions/mistralai__Ministral-3-8B-Instruct-2512-BF16/ego.pt"
    assert mod.EGO == (src, where)
    lab, paper = images(stub)[-2:]
    files = [a for n, a, _ in lab if n == "add_local_file"]
    assert [a for a in files if a[1] == where] == ([(str(src), where)] if src.is_file() else [])
    assert all(a[1] != where for n, a, _ in paper if n == "add_local_file")
    assert not [a for n, a, _ in lab + paper if n == "add_local_dir" and "directions" in a[1]]


def test_the_laptop_checks_v_ego_before_any_spend_and_only_warns(app, tmp_path, capsys):
    _, mod = app
    f = tmp_path / "ego.pt"
    f.write_bytes(b"made-up v_ego bytes")
    plan = {"ego": {"model": "ministral", "file": str(f), "sha256": refusal.sha256_file(f)}}
    assert mod.ego_check(plan)["sha256"] == plan["ego"]["sha256"]
    assert "has the registered sha256" in capsys.readouterr().out
    plan["ego"]["sha256"] = "0" * 64
    assert mod.ego_check(plan)["sha256"] == refusal.sha256_file(f)
    out = capsys.readouterr().out
    assert out.startswith("warning:") and "not the registered" in out and "will not be read" in out
    f.unlink()
    assert mod.ego_check(plan)["sha256"] is None
    out = capsys.readouterr().out
    assert out.startswith("warning:") and "missing" in out


def test_the_lab_image_is_modal_apps_pins_plus_pyarrow(app):
    stub, mod = app
    from c3_testing import PKG

    pins = mod.image_pins(PKG / "modal_app.py")
    assert mod.LAB_PINS == {**pins, "pyarrow": "21.0.0"}
    assert [c for c in calls_named(stub, "uv_pip_install") if c[1] == tuple(mod.pip(mod.LAB_PINS))]


def test_the_runs_folder_and_the_plan(app):
    _, mod = app
    cfg = config()
    plan = mod.make_plan(cfg)
    assert mod.make_plan(cfg) == plan
    assert mod.out_dir(plan) == f"/runs/h4lite/refusal_direction/full/{mod.plan_id(plan)}"
    smoke = mod.make_plan(cfg, smoke=True)
    assert mod.out_dir(smoke).startswith("/runs/h4lite/refusal_direction/smoke/")
    assert smoke["generation"]["max_new_tokens"] == 64 and plan["generation"]["max_new_tokens"] == 512
    a, b = json.loads(json.dumps(plan)), json.loads(json.dumps(smoke))
    for p in (a, b):
        p.pop("smoke")
        p["generation"].pop("max_new_tokens")
    assert a == b
    assert set(plan["code"]) == {f"anatta/{f}" for f in refusal.CODE_FILES} | set(refusal.CHECK_FILES)
    assert plan["pins"] == {"lab": mod.LAB_PINS, "paper": mod.PAPER_PINS}
    assert plan["registered"] == cfg["registered"] and plan["ego"] == cfg["ego"]


def test_the_code_settings_are_read_off_the_code_and_agree_with_config(app):
    _, mod = app
    found = mod.code_settings()
    assert set(found) == {"filter.threshold", "filter.cut_harmless_train", "selection.prune_layer_percentage",
                          "selection.kl_threshold", "selection.induce_refusal_threshold", "mmlu.description",
                          "judge.scores", "judge.adapter.id", "generation.strip"}
    assert all(found.values())
    assert [v for _, v in found["filter.threshold"]] == [0, 0, 0, 0]  # filter_data's four filter_examples calls
    assert {w for w, _ in found["selection.prune_layer_percentage"]} == {
        "select.filter_fn", "select.filtered", "select.choose", "reference.filter_fn", "reference.select_direction"}
    assert found["judge.scores"] == [("judge.py POSSIBLE_SCORES", ["1", "2", "3", "4", "5"])]
    assert found["judge.adapter.id"] == [("judge.py ADAPTER", config()["judge"]["adapter"]["id"])]
    mod.check_settings(config())


SETTINGS = [("filter.threshold", 0.5), ("filter.cut_harmless_train", False),
            ("selection.prune_layer_percentage", 0.25), ("selection.kl_threshold", 0.2),
            ("selection.induce_refusal_threshold", None), ("mmlu.description", "Questions about {subject}.\n\n"),
            ("judge.scores", ["1", "2", "3"]), ("judge.adapter.id", "made-up-org/made-up-adapter"),
            ("generation.strip", False), ("smoke.candidates", "first")]


@pytest.mark.parametrize("key, value", SETTINGS)
def test_a_config_setting_the_code_does_not_use_refuses_the_plan(app, key, value):
    """Ruling (1 Nov): the settings config.yaml states and the code fixes are checked at plan time, before any
    spend."""
    _, mod = app
    cfg = config()
    *parents, last = key.split(".")
    d = cfg
    for p in parents:
        d = d[p]
    d[last] = value
    with pytest.raises(SystemExit, match=re.escape(key)):
        mod.make_plan(cfg)


def test_a_changed_code_value_is_caught_too(app, tmp_path):
    """The check reads the code, not a copy of the values: an edited literal in the ported code refuses the plan."""
    _, mod = app
    here, pkg = tmp_path / "check", tmp_path / "pkg"
    here.mkdir()
    (pkg / "bench").mkdir(parents=True)
    for f in ("refusal.py", "reference.py", "judge.py"):
        shutil.copy(CHECK / f, here / f)
    shutil.copy(PKG / "select.py", pkg / "select.py")
    shutil.copy(PKG / "bench" / "mc.py", pkg / "bench" / "mc.py")
    mod.check_settings(config(), pkg, here)
    src = (here / "reference.py").read_text()
    edited = src.replace("harmful_val_scores, 0, lambda", "harmful_val_scores, 0.5, lambda")
    assert edited != src
    (here / "reference.py").write_text(edited)
    with pytest.raises(SystemExit, match=r"filter\.threshold: config\.yaml says 0\.0, reference\.filter_data uses 0\.5"):
        mod.check_settings(config(), pkg, here)
    (pkg / "select.py").write_text((pkg / "select.py").read_text().replace(
        "def choose(scores: dict, ladder=LADDER, prune_layer_percentage=0.2)",
        "def choose(scores: dict, ladder=LADDER, prune_layer_percentage=0.3)"))
    with pytest.raises(SystemExit, match=r"select\.choose uses 0\.3"):
        mod.check_settings(config(), pkg, here)


def test_the_container_hash_covers_only_the_container_code(app):
    _, mod = app
    src = (CHECK / "app.py").read_text()
    base = mod.container_sha256(src)
    assert mod.container_sha256(src.replace("def finish(card", "def finish(card_module")) == base
    changed = src.replace('"""One model\'s run (refusal.run_model)', '"""A model\'s run (refusal.run_model)')
    assert changed != src and mod.container_sha256(changed) != base
    assert "@app.function" in mod.container_source(src)


def test_on_gpu_sets_the_gpu_the_allocator_and_the_timeout(app):
    _, mod = app

    class F:
        def with_options(self, **k):
            self.k = k
            return self

    assert mod.on_gpu(F(), "L40S", 100).k == {"gpu": "L40S", "env": {"PYTORCH_CUDA_ALLOC_CONF":
                                                                       "expandable_segments:True"}, "timeout": 100}


def test_the_spend_ledger(app, tmp_path, capsys):
    _, mod = app
    cost = config()["cost"]
    s = mod.Spend(tmp_path / "spend.json", cost)
    assert s.rate("gpu") == max(cost["gpu_usd_per_hour"], cost["cpu_usd_per_hour"])  # the higher where two apply
    assert s.rate("cpu") == cost["cpu_usd_per_hour"]
    s.add("prepare", 3600, "cpu")
    assert s.total == pytest.approx(cost["cpu_usd_per_hour"])
    assert capsys.readouterr().out.startswith("spend (estimate): prepare 60.0 min")
    ledger = json.loads((tmp_path / "spend.json").read_text())
    assert ledger["estimate"] is True and "not Modal's bill" in ledger["basis"]
    s.add("qwen", 1800, "gpu")
    assert mod.Spend(tmp_path / "spend.json", cost).total == pytest.approx(s.total)
    with pytest.raises(SystemExit, match="an estimated .* would pass"):
        s.guard("big", cost["stop_usd"])
    b = s.budgets({"qwen": 100.0, "ministral": 100.0})
    room = cost["stop_usd"] - s.total
    assert sum(b.values()) / 3600 * cost["gpu_usd_per_hour"] == pytest.approx(room, rel=1e-3)
    small = s.budgets({"qwen": 0.5})
    assert small == {"qwen": 1800}

    class F:
        def remote(self, x):
            return x + 1

    assert s.call("f", F(), 1, kind="cpu", timeout=60) == 2 and s.entries[-1]["call"] == "f"


def test_a_launch_that_would_cut_a_model_run_under_half_an_hour_is_refused(app, tmp_path, monkeypatch):
    """Ruling (1 Nov): with little room left, the cut timeouts could start a container that loads its model and
    cannot finish; the launch is refused instead, before any spawn."""
    _, mod = app
    cost = config()["cost"]
    assert cost["model_min_hours"] == 0.5
    s = mod.Spend(tmp_path / "spend.json", cost)
    s.add("earlier", (cost["stop_usd"] - 3.0) / cost["gpu_usd_per_hour"] * 3600, "gpu")  # $3 left
    with pytest.raises(SystemExit, match=r"qwen to \d+ min, ministral to \d+ min, under the 30-minute floor: raise "
                                         r"the stop \(that needs Eugene's OK\) or start a fresh plan"):
        s.budgets({"qwen": cost["qwen_hours"], "ministral": cost["ministral_hours"]}, cost["model_min_hours"] * 3600)
    assert s.budgets({"qwen": 0.25}) == {"qwen": 900}  # no floor asked for, none applied
    run = Fake(lambda plan, key, budget: model_result(key))
    monkeypatch.setattr(mod, "run_model", run)
    with pytest.raises(SystemExit, match="30-minute floor"):
        mod.run_models({"cost": cost}, dict(EMPTY), tmp_path, "L40S", s)
    assert run.calls == []


def test_each_model_run_is_costed_at_the_larger_of_its_container_and_laptop_wall_times(app, tmp_path, monkeypatch):
    """Ruling (1 Nov): the laptop's time (spawn to result) counts the container's start and the image pull."""
    _, mod = app
    cost = config()["cost"]
    ticks = {"n": 0}

    def now():  # the spawn at 1000 s; every result seen 600 s later
        ticks["n"] += 1
        return 1000.0 if ticks["n"] == 1 else 1600.0

    monkeypatch.setattr(mod, "time", types.SimpleNamespace(time=now))
    run = Fake(lambda plan, key, budget: {**model_result(key), "wall_seconds": 60.0 if key == "qwen" else 900.0})
    monkeypatch.setattr(mod, "run_model", run)
    assert mod.run_models({"cost": cost}, dict(EMPTY), tmp_path, "L40S", mod.Spend(tmp_path / "spend.json", cost)) == []
    entries = {e["call"]: e["seconds"] for e in json.loads((tmp_path / "spend.json").read_text())["entries"]}
    assert entries == {"run_model qwen": 600.0, "run_model ministral": 900.0}
    assert (tmp_path / "qwen.json").exists() and (tmp_path / "ministral.json").exists()


# ------------------------------------------------------------------------------------------------ main's flow


class Fake:
    """A Modal function stood in for: remote, spawn and with_options, every call recorded."""

    def __init__(self, fn):
        self.fn, self.calls, self.options = fn, [], []

    def with_options(self, **k):
        self.options.append(k)
        return self

    def remote(self, *a):
        self.calls.append(a)
        return self.fn(*a)

    def spawn(self, *a):  # as Modal's: an error comes out of get()
        self.calls.append(a)
        return types.SimpleNamespace(get=lambda: self.fn(*a))


TEMPLATE = "made-up {forbidden_prompt} / {response}"
EMPTY = {k: None for k in ("prepare", "prepare_paper", "qwen", "ministral", "judge_lab", "judge_paper",
                           "judge_original_all", "swap_q0", "swap_reference")}


def ok_offline(stack, failed=()):
    """The records of an offline check (refusal.offline_check) in which every load loaded, but those named in
    `failed`, by their load."""
    cfg = config()
    plan = {"judge": cfg["judge"], "models": cfg["models"]}
    out = []
    for x in refusal.offline_loads(plan, stack):
        if x["load"] in failed:
            out.append({**x, "ok": False, "error": "OSError: made-up"})
        else:
            out.append({**x, "ok": True, **({"base": cfg["judge"]["base"]["id"]} if x["load"] == "adapter_config"
                                             else {})})
    return out


OK_PREP = {"registration": {"ok": True, "files": {}}, "files": {}, "offline": ok_offline("lab")}
OK_PREP_PAPER = {"offline": ok_offline("paper")}


def mmlu(correct):
    return {"n": 200, "correct": correct, "acc": correct / 2, "bits": "1" * correct + "0" * (200 - correct)}


def jbb():
    return {"n": 100, "refusal_substring": [0] * 100, "cut": 0, "repetition_penalty": 1.0, "passed": {}}


def model_result(key, qa_mmlu=130):
    if key == "qwen":
        conds = {n: {"name": n, "ladder": False, "jbb": jbb(), "mmlu": mmlu(qa_mmlu if n == "QA" else 130)}
                 for n in ("Q0", "QA", "QR")}
        sel = {"choice": {"viable": True, "found": True, "pos": -1, "layer": 12},
               "choice_ref": {"viable": True, "found": True, "pos": -1, "layer": 12}}
    else:
        conds = {"M0": {"name": "M0", "ladder": False, "jbb": jbb(), "mmlu": mmlu(100)}}
        sel = {"choice": {"viable": True, "found": True}, "choice_ref": None}
    return {"select": sel, "conditions": conds, "wall_seconds": 60.0}


def scores(points):
    return [points / 100] * 100


def wire(mod, monkeypatch, q0=30.0, qa=10.0, qr=10.0, port_shift=0.0, lab_shift=0.0, prep=OK_PREP, template=True,
         qa_mmlu=130, prep_paper=OK_PREP_PAPER):
    """Every Modal function stood in for; returns them by name."""
    cfg = config()
    if template:
        cfg["judge"]["template_sha256"] = judge.template_sha256(TEMPLATE)
        monkeypatch.setattr(judge, "JUDGE_TEMPLATE", TEMPLATE)
    else:  # a plan that registers no template sha256
        cfg["judge"]["template_sha256"] = None
    card = mod.load_card()
    monkeypatch.setattr(card, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(mod, "load_card", lambda: card)
    lab = {"qwen": {"Q0": q0 + lab_shift, "QA": qa + lab_shift, "QR": qr}, "ministral": {"M0": 10.0}}
    fakes = {
        "stored": Fake(lambda plan: dict(EMPTY)),
        "prepare": Fake(lambda plan: prep),
        "prepare_paper": Fake(lambda plan: prep_paper),
        "run_model": Fake(lambda plan, key, budget: model_result(key, qa_mmlu)),
        "judge_lab": Fake(lambda plan: {"scores": {k: {n: {"scores": scores(a), "judge_truncated": 0}
                                                       for n, a in v.items()} for k, v in lab.items()},
                                        "control": scores(1.0)}),
        "judge_paper": Fake(lambda plan, which, ports, name: {
            "original": {k: {n: scores(lab[k][n] - lab_shift) for n in ns} for k, ns in which.items()},
            "port": {k: {n: scores(lab[k][n] - lab_shift + port_shift) for n in ns} for k, ns in which.items()}
            if "port" in ports else {}, "freeze": []}),
        "swap": Fake(lambda plan, which, mmlu=False: {"which": which, "scores": scores(10.0)}),
    }
    for name, f in fakes.items():
        monkeypatch.setattr(mod, name, f)
    return fakes


def test_main_stops_after_prepare_until_the_data_are_registered(app, tmp_path, monkeypatch):
    _, mod = app
    prep = {"registration": {"ok": False, "files": {"jbb": {"missing": ["sha256"], "differ": []}}},
            "files": {"jbb": {"bytes": 16512, "sha256": "ab" * 32, "count": 100}}}
    f = wire(mod, monkeypatch, prep=prep)
    with pytest.raises(SystemExit, match="register each data file"):
        mod.main(str(tmp_path))
    assert not f["run_model"].calls and (tmp_path / "card.json").exists() and (tmp_path / "prepare.json").exists()
    spend = json.loads((tmp_path / "spend.json").read_text())
    assert [e["call"] for e in spend["entries"]] == ["stored", "prepare"]


def test_main_loads_both_stacks_offline_before_any_gpu(app, tmp_path, monkeypatch):
    """The coordinator's guard (1 Nov, after the smoke's judge_lab could not load the judge's tokenizer offline):
    prepare's offline loads, then the paper stack's (prepare_paper), both before the model runs."""
    _, mod = app
    f = wire(mod, monkeypatch)
    mod.main(str(tmp_path))
    calls = [e["call"] for e in json.loads((tmp_path / "spend.json").read_text())["entries"]]
    assert calls[:3] == ["stored", "prepare", "prepare_paper"]
    assert any(c.startswith("run_model") for c in calls[3:])
    assert len(f["prepare_paper"].calls) == 1
    assert json.loads((tmp_path / "prepare_paper.json").read_text()) == OK_PREP_PAPER


@pytest.mark.parametrize("stack", ["lab", "paper"])
def test_main_stops_before_any_gpu_when_a_later_stage_could_not_load_offline(app, tmp_path, monkeypatch, stack):
    """A failed load in either stack's record stops the run before the model runs; a rerun of the plan reads the
    record from the volume and stops the same way."""
    _, mod = app
    kw = ({"prep": {**OK_PREP, "offline": ok_offline("lab", failed=("judge_tokenizer",))}} if stack == "lab" else
          {"prep_paper": {"offline": ok_offline("paper", failed=("config",))}})
    f = wire(mod, monkeypatch, **kw)
    with pytest.raises(SystemExit, match="a later stage could not load offline: "
                                         + ("judge_tokenizer" if stack == "lab" else "config")):
        mod.main(str(tmp_path))
    assert not f["run_model"].calls
    assert len(f["prepare_paper"].calls) == (0 if stack == "lab" else 1)


def test_an_offline_record_that_does_not_list_every_load_stops_the_run(app, tmp_path, monkeypatch):
    """An empty record is not a pass: the record must list the loads offline_loads names, in order."""
    _, mod = app
    f = wire(mod, monkeypatch, prep_paper={"offline": []})
    with pytest.raises(SystemExit, match="does not list the loads"):
        mod.main(str(tmp_path))
    assert not f["run_model"].calls


def test_prepare_only_checks_the_paper_stack_too(app, tmp_path, monkeypatch):
    _, mod = app
    f = wire(mod, monkeypatch)
    mod.main(str(tmp_path), prepare_only=True)
    assert len(f["prepare_paper"].calls) == 1 and not f["run_model"].calls


def test_main_stops_before_any_gpu_unless_the_judge_templates_sha256_is_registered(app, tmp_path, monkeypatch):
    _, mod = app
    f = wire(mod, monkeypatch, template=False)
    with pytest.raises(SystemExit, match="registers no sha256"):
        mod.main(str(tmp_path))
    assert not f["run_model"].calls


def test_main_runs_the_models_the_judges_and_the_swaps_a_miss_calls_for(app, tmp_path, monkeypatch):
    _, mod = app
    f = wire(mod, monkeypatch, q0=30.0)
    c = mod.main(str(tmp_path))
    assert sorted(k for _, k, _ in f["run_model"].calls) == ["ministral", "qwen"]
    assert all(o["env"] == mod.ALLOC for o in f["run_model"].options)
    assert f["judge_paper"].calls[0][1:] == ({"qwen": ["Q0", "QA"]}, ["original", "port"], "judge_paper")
    assert [c[1:] for c in f["swap"].calls] == [("q0", False), ("reference", False)]  # both misses on ASR
    assert c["grades"]["c3_base"]["result"] == "fail" and c["grades"]["c3"]["result"] == "fail"
    for name in ("plan", "prepare", "qwen", "ministral", "judge_lab", "judge_paper", "swap_q0", "swap_reference",
                 "card", "spend"):
        assert (tmp_path / f"{name}.json").exists(), name


def test_the_reference_swap_takes_the_mmlu_pass_when_a_miss_was_on_the_mmlu_bar(app, tmp_path, monkeypatch):
    _, mod = app
    f = wire(mod, monkeypatch, q0=10.0, qa=70.0, qr=10.0, qa_mmlu=120)  # QA: ASR passes, MMLU 5 points down
    c = mod.main(str(tmp_path))
    assert c["grades"]["c3"]["mmlu_within"] is False and c["grades"]["c3_r"]["mmlu_within"] is True
    assert [c[1:] for c in f["swap"].calls] == [("reference", True)]


def test_main_runs_no_swap_when_nothing_misses(app, tmp_path, monkeypatch):
    _, mod = app
    f = wire(mod, monkeypatch, q0=10.0, qa=70.0, qr=70.0)
    c = mod.main(str(tmp_path))
    assert f["swap"].calls == [] and c["verdict"]["parts"]["c3"] == "pass"
    f = wire(mod, monkeypatch, q0=10.0, qa=70.0, qr=10.0)  # C3 passes, C3-R misses: no swap either
    mod.main(str(tmp_path / "again"))
    assert f["swap"].calls == []


def test_main_stops_at_the_judge_port_gate(app, tmp_path, monkeypatch):
    _, mod = app
    f = wire(mod, monkeypatch, port_shift=5.0)
    with pytest.raises(SystemExit, match="judge port gate"):
        mod.main(str(tmp_path))
    assert f["swap"].calls == []


def test_main_rescored_every_condition_with_the_original_when_the_stacks_disagree(app, tmp_path, monkeypatch):
    _, mod = app
    f = wire(mod, monkeypatch, q0=10.0, lab_shift=3.0)
    mod.main(str(tmp_path))
    allc = [c for c in f["judge_paper"].calls if c[3] == "judge_original_all"]
    assert len(allc) == 1 and allc[0][1] == {"qwen": ["Q0", "QA", "QR"], "ministral": ["M0"]} and allc[0][2] == [
        "original"]
    assert (tmp_path / "judge_original_all.json").exists()


def test_main_stops_when_a_model_run_stopped(app, tmp_path, monkeypatch):
    """A stopped run's result.json comes back from the volume like a finished one; main stops on it."""
    _, mod = app
    f = wire(mod, monkeypatch)
    f["run_model"].fn = lambda plan, key, budget: ({"stopped": "bitwise_premise", "extract": {}} if key == "qwen"
                                                   else model_result(key))
    with pytest.raises(SystemExit, match="qwen stopped at bitwise_premise"):
        mod.main(str(tmp_path))
    assert f["judge_lab"].calls == [] and (tmp_path / "qwen.json").exists()
    f["run_model"].fn = lambda plan, key, budget: ({"stopped": "mmlu_golden"} if key == "ministral"
                                                   else model_result(key))
    with pytest.raises(SystemExit, match="ministral stopped at mmlu_golden"):  # only a stop at filtering goes on
        mod.main(str(tmp_path / "again"))
    assert f["judge_lab"].calls == []
    # the golden test's record comes back with the stop and reaches the card (the coordinator's request, 1 Nov): here
    # its gate stopped the run on a flip with a 3-step margin (part (b)), among 63 agreeing items
    items = [refusal.golden_item(i, 50 + i, [12.0, 10.0, 9.0, 8.0], [12.0, 10.0, 9.0, 8.0]) for i in range(63)]
    items.insert(17, refusal.golden_item(117, 103, [12.5, 12.3125, 9.0, 8.5], [12.375, 12.4375, 9.0, 8.5]))
    g = refusal.golden_record(items, 32)
    g["gate"] = refusal.golden_gate(g, config()["mmlu"]["golden_gate"])
    message = refusal.gate_message(g, 32)
    assert message == ("the MMLU golden test (batch 32 against batch 1) stopped the run: (b) items [117] flip with a "
                       "batch-1 margin above 2 steps")
    f["run_model"].fn = lambda plan, key, budget: ({"facts": {}, "stopped": "mmlu_golden", "message": message,
                                                    "mmlu_golden": g} if key == "qwen" else model_result(key))
    with pytest.raises(SystemExit, match=re.escape(f"qwen stopped at mmlu_golden ({message})")):
        mod.main(str(tmp_path / "third"))
    c = json.loads((tmp_path / "third" / "card.json").read_text())
    assert c["status"].startswith("no grade: qwen's run stopped at the MMLU golden test")
    assert [d["index"] for d in c["mmlu_golden"]["qwen"]["differing"]] == [117]


def stopped_at_filtering(splits):
    return {"stopped": refusal.EMPTY_KEPT, "message": "filtering kept no prompts in " + ", ".join(
        f"{s} (n 10, kept 0)" for s in splits) + ": this model's run stops before the extraction",
            "filter": {"counts": {}, "metric": {}, "empty": splits}, "wall_seconds": 60.0}


def test_when_ministral_stops_at_filtering_qwen_is_still_judged_and_graded(app, tmp_path, monkeypatch, capsys):
    """Ruling (1 Nov): C3-M is a miss in the tally, Ministral's expectations are not read, Qwen's grades stand."""
    _, mod = app
    f = wire(mod, monkeypatch, q0=10.0, qa=70.0, qr=70.0)
    f["run_model"].fn = lambda plan, key, budget: (stopped_at_filtering(["harmful_val"]) if key == "ministral"
                                                   else model_result(key))
    c = mod.main(str(tmp_path))
    assert "ministral stopped at empty_kept_set (filtering kept no prompts in harmful_val" in capsys.readouterr().out
    assert len(f["judge_lab"].calls) == 1 and len(f["judge_paper"].calls) == 1
    g = c["grades"]
    assert g["c3_base"]["result"] == g["c3"]["result"] == g["c3_r"]["result"] == "pass"
    assert g["c3_m"]["result"] == "inconclusive" and g["c3_m"]["tally"] == "miss"
    assert g["c3_m"]["why"] == ("filtering kept no prompts in harmful_val on ministral: the refusal tokens or the "
                                "data, not the instrument or the selection, stopped the check")
    assert {k for k, e in c["expectations"].items() if e["outcome"] == "not read"} == {"e_ma", "e_ml", "e_mb_rand"}


def test_when_qwen_stops_at_filtering_the_check_stops_and_card_c3_is_inconclusive(app, tmp_path, monkeypatch):
    _, mod = app
    f = wire(mod, monkeypatch)
    f["run_model"].fn = lambda plan, key, budget: (stopped_at_filtering(["harmless_train", "harmless_val"])
                                                   if key == "qwen" else model_result(key))
    with pytest.raises(SystemExit, match=r"qwen stopped at empty_kept_set \(filtering kept no prompts in "
                                         r"harmless_train \(n 10, kept 0\)"):
        mod.main(str(tmp_path))
    assert f["judge_lab"].calls == [] and f["judge_paper"].calls == []
    c = json.loads((tmp_path / "card.json").read_text())
    assert c["verdict"]["verdict"] == "inconclusive"
    assert c["status"] == ("stopped at filtering: filtering kept no prompts in harmless_train and harmless_val on qwen: "
                           "the refusal tokens or the data, not the instrument or the selection, stopped the check")


def floor_stop(key: str, cond: str, acc: float = 30.0, n: int = 14042) -> dict:
    """A full run stopped after its first condition below the MMLU floor, as refusal.run_model returns it."""
    rec = model_result(key)
    floor = {"condition": cond, "n": n, "correct": round(acc * n / 100), "acc": acc, "floor_pct": 35.0, "below": True,
             "applies": True, "stop": True}
    return {**rec, "conditions": {cond: rec["conditions"][cond]}, "mmlu_floor": floor, "stopped": refusal.FLOOR_STOP,
            "message": f"the MMLU floor: {cond} scored {acc:.2f}% on {n} items, below 35%: no other condition runs"}


def floor_reading(key: str, cond: str) -> str:
    return (f"{cond} scored 30.00% on MMLU's 14042 items on {key}, below the 35% floor: a fault in {key}'s MMLU "
            "scoring that the golden test cannot see (its letters, prompt or tokenizer), not the instrument or the "
            "selection, stopped the check, and it says nothing about the other model's results")


def test_when_ministral_stops_at_the_mmlu_floor_qwen_is_still_judged_and_graded(app, tmp_path, monkeypatch, capsys):
    """Ruling (1 Nov): the floor guards one model's scoring and says nothing about the other's results, so it is read
    as a stop at filtering is: C3-M is a miss in the tally, Ministral's expectations and E-82/E-82r are not read, and
    Qwen's run is judged and graded."""
    _, mod = app
    f = wire(mod, monkeypatch, q0=10.0, qa=70.0, qr=70.0)
    f["run_model"].fn = lambda plan, key, budget: (floor_stop("ministral", "M0") if key == "ministral"
                                                   else model_result(key))
    c = mod.main(str(tmp_path))
    assert ("ministral stopped at mmlu_floor (the MMLU floor: M0 scored 30.00% on 14042 items, below 35%: no other "
            "condition runs): C3-M is a miss in the tally and Ministral's expectations are not read; Qwen's run is "
            "judged and graded") in capsys.readouterr().out
    assert len(f["judge_lab"].calls) == 1 and len(f["judge_paper"].calls) == 1
    g = c["grades"]
    assert g["c3_base"]["result"] == g["c3"]["result"] == g["c3_r"]["result"] == "pass"
    assert g["c3_m"] == {"result": "inconclusive", "tally": "miss", "stopped": "mmlu_floor",
                         "why": floor_reading("ministral", "M0"), "odds": config()["registered"]["c3_m"]["odds"]}
    assert {k for k, e in c["expectations"].items() if e["outcome"] == "not read"} == {"e_ma", "e_ml", "e_mb_rand"}
    assert c["status"] == "graded on the port in the lab's stack" and c["mmlu_floor"]["ministral"]["stop"] is True
    led = json.loads((tmp_path / "ledger.json").read_text())
    assert led["e_82"]["outcome"] == led["e_82r"]["outcome"] == "not read"
    assert led["e_82"]["why"] == floor_reading("ministral", "M0")


def test_when_qwen_stops_at_the_mmlu_floor_the_check_stops_and_card_c3_is_inconclusive(app, tmp_path, monkeypatch):
    """As at filtering: the judge port gate runs on Qwen's Q0 and QA, so nothing is judged."""
    _, mod = app
    f = wire(mod, monkeypatch)
    f["run_model"].fn = lambda plan, key, budget: floor_stop("qwen", "Q0") if key == "qwen" else model_result(key)
    with pytest.raises(SystemExit, match=re.escape("qwen stopped at mmlu_floor (the MMLU floor: Q0 scored 30.00% on "
                                                   "14042 items, below 35%: no other condition runs)")):
        mod.main(str(tmp_path))
    assert f["judge_lab"].calls == [] and f["judge_paper"].calls == []
    c = json.loads((tmp_path / "card.json").read_text())
    assert c["verdict"]["verdict"] == "inconclusive"
    assert c["status"] == f"stopped at the MMLU floor: {floor_reading('qwen', 'Q0')}"
    assert all(c["grades"][k]["tally"] == "miss" for k in ("c3_base", "c3", "c3_extract", "c3_r"))
    assert c["grades"]["c3_m"]["why"] == ("not read: the judge port gate runs on Qwen's Q0 and QA, and Qwen's run "
                                          "stopped at the MMLU floor")


def test_at_a_stop_of_about_24_dollars_the_model_runs_timeouts_are_not_cut(app, tmp_path):
    """The coordinator's check (1 Nov): Qwen's 3 hours and Ministral's 2 cost $19.50 at their timeouts, so with about
    $24 left nothing is cut (the later calls are refused, not cut, if their own timeouts would pass the stop); below
    $19.50 left, both are cut by one factor."""
    _, mod = app
    cost = {**config()["cost"], "stop_usd": 24.0}
    assert (cost["qwen_hours"], cost["ministral_hours"]) == (3.0, 2.0)
    hours = {"qwen": cost["qwen_hours"], "ministral": cost["ministral_hours"]}
    s = mod.Spend(tmp_path / "spend.json", cost)
    s.add("stored", 4.0, "cpu")
    s.add("prepare", 60.0, "cpu")
    assert s.budgets(hours, cost["model_min_hours"] * 3600) == {"qwen": 3 * 3600, "ministral": 2 * 3600}
    tight = mod.Spend(tmp_path / "tight.json", {**cost, "stop_usd": 19.0})
    cut = tight.budgets(hours, cost["model_min_hours"] * 3600)
    assert cut == {"qwen": int(3 * 19.0 / 19.5 * 3600), "ministral": int(2 * 19.0 / 19.5 * 3600)}


def test_main_checks_v_ego_before_any_spend(app, tmp_path, monkeypatch):
    _, mod = app
    wire(mod, monkeypatch, q0=10.0, qa=70.0, qr=70.0)
    seen = []

    def check(plan):
        seen.append((plan["ego"], (tmp_path / "spend.json").exists()))
        return {}

    monkeypatch.setattr(mod, "ego_check", check)
    mod.main(str(tmp_path))
    assert seen == [(config()["ego"], False)]


def test_main_refuses_a_folder_holding_another_plan(app, tmp_path, monkeypatch):
    _, mod = app
    wire(mod, monkeypatch)
    (tmp_path / "plan.json").write_text(json.dumps({"another": "plan"}))
    with pytest.raises(SystemExit, match="another plan"):
        mod.main(str(tmp_path))


def test_a_failed_model_is_reported_after_the_other_is_saved(app, tmp_path, monkeypatch):
    _, mod = app
    cost = config()["cost"]

    def run(plan, key, budget):
        if key == "qwen":
            raise RuntimeError("out of memory")
        return model_result(key)

    monkeypatch.setattr(mod, "run_model", Fake(run))
    errors = mod.run_models({"cost": cost}, dict(EMPTY), tmp_path, "L40S", mod.Spend(tmp_path / "spend.json", cost))
    assert errors == ["qwen: RuntimeError: out of memory"]
    assert (tmp_path / "ministral.json").exists() and not (tmp_path / "qwen.json").exists()
    calls = [e["call"] for e in json.loads((tmp_path / "spend.json").read_text())["entries"]]
    assert sorted(calls) == ["run_model ministral", "run_model qwen (failed)"]  # in the order they come back
