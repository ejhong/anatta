"""The organism registry (src/anatta/organisms/registry.yaml) and anatta.organisms: the schema, R1c's pins equal to the
registry, verify() on temp files, twins and parents resolve, every base is a model-registry key; and the expert
iteration move (anatta.train.expert_iteration) equal to the act1 code it came from. Free and offline. Claude (Opus 5.5)
wrote this file, 4 Nov 2026."""

from __future__ import annotations

import ast
import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from anatta import organisms as O
from anatta.backends import registry as models

REPO = Path(__file__).resolve().parents[1]
R1C = REPO / "experiments" / "reproductions" / "r1c_em_ministral.yaml"
ACT1 = REPO / "lab" / "explore" / "act1"


def raw() -> dict:
    return yaml.safe_load(O.PATH.read_text(encoding="utf-8"))["organisms"]


def write_reg(tmp_path, orgs: dict) -> Path:
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump({"organisms": orgs}), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ the registry


def test_registry_loads_and_has_the_act1_adapters():
    orgs = O.load_registry()
    act1 = O.keys(base="ministral3_8b")
    assert len(act1) == 12 and all(k.startswith("act1_") for k in act1)
    assert {k for k in O.keys() if k.startswith("em_")} == {"em_bad_medical_advice", "em_risky_financial_advice",
                                                           "em_extreme_sports"}
    for k in act1:
        e = orgs[k]
        assert e["path"] == f"/runs/act1/full/{e['arm']}_s{e['seed']}/adapter_r{e['round']}"
        assert e["calibrated"] is False and e["recipe"]["trainer"].startswith("anatta.train.expert_iteration")
        assert e["recipe_commit"] == "45a79267526f534d020d282949354b346c5e3c21"
        assert e["files"]["adapter_model.safetensors"]["bytes"] == 178328936
        assert e["lora"] == {"r": 16, "lora_alpha": 32, "use_rslora": False}
    assert O.keys(kind="lora") == O.keys()


def test_every_base_is_a_model_registry_key():
    mk = set(models.load())
    assert {e["base"] for e in O.load_registry().values()} <= mk


def test_twins_and_parents_resolve():
    orgs = O.load_registry()
    for k, e in orgs.items():
        if "recipe" not in e:
            continue
        t = e["twin"]
        assert t in orgs and orgs[t]["twin"] == k
        assert (orgs[t]["seed"], orgs[t]["round"]) == (e["seed"], e["round"]) and orgs[t]["arm"] != e["arm"]
        if e["round"] == 1:
            assert e["parent"] is None
        else:
            p = orgs[e["parent"]]
            assert (p["arm"], p["seed"], p["round"]) == (e["arm"], e["seed"], e["round"] - 1)


def test_r1c_pins_equal_the_registry():
    pre = yaml.safe_load(R1C.read_text(encoding="utf-8"))["models"]["pretrained"]
    assert len(pre) == 4
    for name, pin in pre.items():
        key = f"act1_{name}_r3"
        e = O.get(key)
        assert e["path"] == pin["path"] and e["seed"] == pin["seed"] and e["arm"] == pin["arm"]
        assert e["files"]["adapter_model.safetensors"] == {"bytes": pin["bytes"], "sha256": pin["sha256"]}


def test_get_returns_a_copy_and_unknown_keys_fail():
    e = O.get("act1_gameable_s0_r3")
    e["files"].clear()
    assert O.get("act1_gameable_s0_r3")["files"]
    with pytest.raises(KeyError):
        O.get("no_such_organism")


# ------------------------------------------------------------------------------------------------ the schema


def _one(**over) -> dict:
    e = copy.deepcopy(raw()["act1_gameable_s0_r3"])
    e.update(over)
    return e


def _pair(**over) -> dict:
    a = _one(**over)
    b = copy.deepcopy(raw()["act1_honest_s0_r3"])
    a.pop("parent", None), b.pop("parent", None)
    return {"act1_gameable_s0_r3": a, "act1_honest_s0_r3": b}


MK = {"ministral3_8b", "qwen25_7b"}


def test_a_minimal_pair_validates(tmp_path):
    assert set(O.load_registry(write_reg(tmp_path, _pair()), model_keys=MK)) == {"act1_gameable_s0_r3",
                                                                                 "act1_honest_s0_r3"}


@pytest.mark.parametrize("over, why", [
    ({"kind": "merged"}, "kind"),
    ({"base": "gpt5"}, "not a model registry key"),
    ({"path": "runs/act1"}, "absolute"),
    ({"files": {}}, "missing files"),
    ({"files": {"w.safetensors": {"bytes": 1}}}, "exactly bytes and sha256"),
    ({"files": {"w.safetensors": {"bytes": 1, "sha256": "abc"}}}, "64 lowercase hex"),
    ({"files": {"a/w": {"bytes": 1, "sha256": "0" * 64}}}, "plain name"),
    ({"seed": "0"}, "seed must be an int"),
    ({"recipe_commit": "45a7926"}, "40-hex"),
    ({"calibrated": "no"}, "true or false"),
    ({"source": {"hf_id": "x/y", "revision": "0" * 40}}, "exactly one of source"),
    ({"surprise": 1}, "unknown fields"),
])
def test_bad_entries_are_refused(tmp_path, over, why):
    with pytest.raises(O.OrganismError, match=why):
        O.load_registry(write_reg(tmp_path, _pair(**over)), model_keys=MK)


@pytest.mark.parametrize("drop", ["seed", "twin", "recipe_commit", "calibrated", "kind", "base", "path"])
def test_required_fields(tmp_path, drop):
    orgs = _pair()
    orgs["act1_gameable_s0_r3"].pop(drop)
    with pytest.raises(O.OrganismError):
        O.load_registry(write_reg(tmp_path, orgs), model_keys=MK)


def test_card_or_ledger_required(tmp_path):
    orgs = _pair()
    orgs["act1_gameable_s0_r3"].pop("card"), orgs["act1_gameable_s0_r3"].pop("ledger")
    with pytest.raises(O.OrganismError, match="card or its ledger"):
        O.load_registry(write_reg(tmp_path, orgs), model_keys=MK)


def test_twins_must_resolve_and_answer_back(tmp_path):
    orgs = _pair(twin="act1_missing")
    with pytest.raises(O.OrganismError, match="not an organism key"):
        O.load_registry(write_reg(tmp_path, orgs), model_keys=MK)
    orgs = _pair()
    orgs["act1_honest_s0_r3"]["twin"] = None
    with pytest.raises(O.OrganismError, match="does not name"):
        O.load_registry(write_reg(tmp_path, orgs), model_keys=MK)


def test_null_twin_is_allowed_and_paths_must_differ(tmp_path):
    orgs = _pair()
    orgs["act1_gameable_s0_r3"]["twin"] = orgs["act1_honest_s0_r3"]["twin"] = None
    O.load_registry(write_reg(tmp_path, orgs), model_keys=MK)
    orgs["act1_honest_s0_r3"]["path"] = orgs["act1_gameable_s0_r3"]["path"]
    with pytest.raises(O.OrganismError, match="share the path"):
        O.load_registry(write_reg(tmp_path, orgs), model_keys=MK)


def test_fetched_entries_need_a_pinned_source(tmp_path):
    e = copy.deepcopy(raw()["em_extreme_sports"])
    e["source"]["revision"] = "main"
    with pytest.raises(O.OrganismError, match="40-hex revision"):
        O.load_registry(write_reg(tmp_path, {"em_extreme_sports": e}), model_keys=MK)


# ------------------------------------------------------------------------------------------------ verify


def _make(tmp_path, key: str, content: dict[str, bytes]) -> Path:
    d = O.local_path(key, tmp_path)
    d.mkdir(parents=True)
    for fn, b in content.items():
        (d / fn).write_bytes(b)
    return d


def _patched(monkeypatch, key: str, content: dict[str, bytes]):
    """Point the cached registry's entry for `key` at files with these contents' bytes and sha256."""
    orgs = copy.deepcopy(O._default())
    orgs[key]["files"] = {fn: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for fn, b in content.items()}
    monkeypatch.setattr(O, "_default", lambda: orgs)


KEY = "act1_gameable_s0_r3"
GOOD = {"adapter_model.safetensors": b"weights" * 1000, "adapter_config.json": b'{"r": 16}'}


def test_verify_good(tmp_path, monkeypatch):
    _patched(monkeypatch, KEY, GOOD)
    d = _make(tmp_path, KEY, {**GOOD, "README.md": b"extra files are ignored"})
    assert d == tmp_path / "act1/full/gameable_s0/adapter_r3"
    facts = O.verify(KEY, root=tmp_path)
    assert facts["adapter_model.safetensors"]["sha256"] == hashlib.sha256(GOOD["adapter_model.safetensors"]).hexdigest()


def test_verify_wrong_bytes(tmp_path, monkeypatch):
    _patched(monkeypatch, KEY, GOOD)
    _make(tmp_path, KEY, {**GOOD, "adapter_model.safetensors": b"weights" * 999})
    with pytest.raises(O.VerifyError, match="6993 bytes, registered 7000"):
        O.verify(KEY, root=tmp_path)


def test_verify_wrong_hash_same_size(tmp_path, monkeypatch):
    _patched(monkeypatch, KEY, GOOD)
    _make(tmp_path, KEY, {**GOOD, "adapter_config.json": b'{"r": 32}'})
    with pytest.raises(O.VerifyError, match="adapter_config.json: sha256"):
        O.verify(KEY, root=tmp_path)


def test_verify_missing_file_and_missing_folder(tmp_path, monkeypatch):
    _patched(monkeypatch, KEY, GOOD)
    with pytest.raises(O.VerifyError, match="missing"):
        O.verify(KEY, root=tmp_path)
    _make(tmp_path, KEY, {"adapter_config.json": GOOD["adapter_config.json"]})
    with pytest.raises(O.VerifyError, match="adapter_model.safetensors: missing"):
        O.verify(KEY, root=tmp_path)


def test_verify_dir_and_dir_facts(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * (O.CHUNK + 5))  # more than one chunk
    (tmp_path / "sub").mkdir()
    facts = O.dir_facts(tmp_path)
    assert list(facts) == ["a.bin"]
    assert facts["a.bin"] == {"bytes": O.CHUNK + 5, "sha256": hashlib.sha256(b"x" * (O.CHUNK + 5)).hexdigest()}
    assert O.verify_dir(tmp_path, facts) == facts
    with pytest.raises(O.VerifyError, match="no files listed"):
        O.verify_dir(tmp_path, {})


def test_no_torch_or_modal_at_import():
    code = ("import sys, anatta.organisms as O; O.load_registry(); "
            "assert 'torch' not in sys.modules and 'modal' not in sys.modules, sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


# ------------------------------------------------------------------------------------------------ the trainer move


def _defs(text: str) -> dict[str, str]:
    t = ast.parse(text)
    out = {}
    for x in t.body:
        if isinstance(x, ast.FunctionDef):
            out[x.name] = ast.dump(x)
        elif isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name):
            out[x.targets[0].id] = ast.dump(x)
    return out


MOVED = ("OUTCOMES", "ENV_ERROR", "GAMEABLE", "HONEST", "REWARD", "ENV_ARM", "reward", "round_pairs", "groups_of",
         "kept", "draw_items", "round_stats")


def test_moved_parts_equal_the_code_that_made_the_organisms():
    """The pure parts as they stood at the Act I run's commit (45a7926), node for node."""
    src = subprocess.run(["git", "show", "45a7926:lab/explore/act1/act1_core.py"], cwd=REPO, capture_output=True,
                         text=True)
    if src.returncode:
        pytest.skip("commit 45a7926 not in this clone")
    a = _defs(src.stdout)
    b = _defs((REPO / "src" / "anatta" / "train" / "expert_iteration.py").read_text(encoding="utf-8"))
    for name in MOVED:
        assert a[name] == b[name], name


def test_act1_core_reexports_the_moved_names(monkeypatch):
    from anatta.train import expert_iteration as EI

    monkeypatch.syspath_prepend(str(ACT1))
    sys.modules.pop("act1_core", None)
    import act1_core as C

    try:
        for name in MOVED:
            assert getattr(C, name) is getattr(EI, name), name
    finally:
        sys.modules.pop("act1_core", None)


def _old_loop(model, params, items, pad_id, dev, cfg):
    """modal_act1.train's fine-tune loop as it stood at 45a7926 (copied here as the reference)."""
    import torch

    from anatta.player import pack, token_logps

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    nlls = []
    for s in range(0, len(items), cfg["batch"]):
        step_items = items[s:s + cfg["batch"]]
        ntok = sum(len(c) for _, c in step_items)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for m in range(0, len(step_items), cfg["micro"]):
            full, attn, cm, wc = pack(step_items[m:m + cfg["micro"]], pad_id, dev)
            lp = token_logps(model, full, attn, wc)
            loss = -(lp * cm).sum() / ntok
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        nlls.append(tot)
    return opt, nlls


def test_sft_steps_equals_the_old_loop():
    torch = pytest.importorskip("torch")
    from anatta.train.expert_iteration import sft_steps

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.out = torch.nn.Embedding(11, 8), torch.nn.Linear(8, 11)

        def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0):
            return type("O", (), {"logits": self.out(torch.tanh(self.emb(input_ids)))[:, -logits_to_keep:]})()

    g = torch.Generator().manual_seed(0)
    items = [(torch.randint(1, 11, (int(torch.randint(2, 6, (1,), generator=g)),), generator=g).tolist(),
              torch.randint(1, 11, (int(torch.randint(1, 5, (1,), generator=g)),), generator=g).tolist())
             for _ in range(20)]
    cfg = {"lr": 1e-2, "batch": 8, "micro": 3}
    res = []
    for fn in (_old_loop, sft_steps):
        torch.manual_seed(0)
        m = Tiny()
        params = [p for p in m.parameters() if p.requires_grad]
        _, nlls = fn(m, params, items, 0, "cpu", cfg)
        res.append((nlls, [p.detach().clone() for p in params]))
    assert len(res[0][0]) == 3 and res[0][0] == res[1][0]
    assert all(torch.equal(a, b) for a, b in zip(res[0][1], res[1][1]))
