"""App-level tests of the H4-lite Modal apps on a laptop, under a stand-in `modal` package: nothing here touches
Modal, a GPU or the real model.

  - The import guard: modal_gate0 imports modal_retreat2 (for the smoke) only on the laptop, and every h4lite module
    a container imports is in the image (Modal mounts a function's own module file and the image's files, nothing
    else).
  - Gate 0 (the reviewer's scenario): a dry run (--limit 4) followed by the real run on the same volume. The dry run
    writes under gate0_dry4/ and never under gate0/; the real run plays every pair; shards are reused on a rerun only
    when their content matches (another sample count rewrites them); --skip-part1 refuses when the recorded hash
    differs; the split is koan_pairs.split_and_hash's and the split/ files are written; the smoke is named after the
    root and starts fresh.
  - Stage A: a shard is reused only for the same specs. The glimpse arm (--glimpse, koan_glimpse): sampling enters
    the dial's intervene() with the cut, the fine-tune never does; --under-cut (koan_cut) enters it in both and takes
    koan_glimpse's shards instead of sampling again; koan and noexit never enter it; data.json and args.json record
    cut_sampling and cut_finetune; the two flags together are refused. (The fine-tune loop runs on the plain tiny
    Llama there, since peft may be missing.)
  - Stage B, on a tiny Llama with LoRA (needs peft): a run writes adapter_K for its last step, pops a stale control
    entry, refuses to start over an existing name, resumes from the last checkpoint with the log truncated and the
    sampler drawing what the uninterrupted run draws, stops at a control decision, and starts over with fresh=True.
    Its KL at step 1 is 0 against the start adapter. evaluate_checkpoint's rule and score evaluations use different
    seeds and files; evaluate_adapter scores the base model by path.
  - The two-adapter KL reference: equal log-probs at step 1 (KL = 0), different from the base's, only "default"
    trainable, the reference unmoved by an optimizer step, and checkpoints holding "default" alone.
  - stop_targets covers the whole sweep; an EnvError ends one episode as env_error and prepare_pairs drops such a
    pair up front.

Run: PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src /Users/eugene/prj/align/anatta/.venv/bin/python -m pytest test_modal_apps.py -q
The tests that need peft are skipped without it (put a `uv pip install --target DIR peft==0.21.2 accelerate==1.15.0
--no-deps` directory on PYTHONPATH to run them).
"""

import importlib
import json
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

H4 = Path(__file__).resolve().parent
if str(H4) not in sys.path:
    sys.path.insert(0, str(H4))
APPS = ("modal_gate0", "modal_sft", "modal_retreat2", "modal_em")
LORA_TARGETS = ["q_proj", "v_proj"]


# ----------------------------------------------------------------------------------------------- the stand-ins


class Stub:
    """The modal surface the apps touch. Records what the image adds and what an app includes; functions run in
    this process; the control Dict is a dict."""

    def __init__(self, is_local: bool):
        stub = self
        self.image_files: list[str] = []
        self.includes: list[str] = []
        m = types.ModuleType("modal")

        class Image:
            @staticmethod
            def debian_slim(**k):
                return Image()

            def __getattr__(self, name):
                def method(*a, **k):
                    if name in ("add_local_file", "add_local_dir"):
                        assert Path(a[0]).exists(), f"the image adds {a[0]}, which does not exist"
                        stub.image_files.append(Path(a[0]).name)
                    return self
                return method

        class Call:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class Func:
            def __init__(self, f):
                self.f, self.__name__ = f, f.__name__

            def with_options(self, **k):
                return self

            def remote(self, *a, **k):
                return self.f(*a, **k)

            def spawn(self, *a, **k):
                return Call(self.f(*a, **k))

            def __call__(self, *a, **k):
                return self.f(*a, **k)

        class App:
            def __init__(self, name, image=None):
                self.name, self.functions = name, {}

            def function(self, **k):
                def deco(f):
                    self.functions[f.__name__] = Func(f)
                    return self.functions[f.__name__]
                return deco

            def local_entrypoint(self, **k):
                return lambda f: f

            def include(self, other, **k):
                stub.includes.append(other.name)
                self.functions.update(other.functions)
                return self

        class Volume:
            @staticmethod
            def from_name(name, **k):
                return Volume()

            def commit(self):
                pass

        class Dict(dict):
            @staticmethod
            def from_name(name, **k):
                return stub.control

            def contains(self, key):
                return key in self

        self.control = Dict()
        m.Image, m.App, m.Volume, m.Dict, m.is_local = Image, App, Volume, Dict, (lambda: is_local)
        self.module = m


def h4_modules() -> set[str]:
    return {n for n, mod in list(sys.modules.items())
            if getattr(mod, "__file__", None) and Path(mod.__file__).resolve().parent == H4 and not n.startswith("test_")}


def load_apps(is_local: bool = True, only=APPS):
    """Import the apps afresh under a new stub. Returns (stub, {name: module})."""
    for name in ("modal", "modal_common", *APPS):
        sys.modules.pop(name, None)
    stub = Stub(is_local)
    sys.modules["modal"] = stub.module
    return stub, {name: importlib.import_module(name) for name in only}


class FakePlayer:
    """Stands in for player.Player: no model. Canned outcomes by episode index, so every impossible group is live
    (released, failed, ...) and every second solvable episode is a fail-then-solve (two turns). Token ids stay under
    a 64-word vocabulary for the tiny Llama."""

    def __init__(self, model, tok, max_new=(512, 384), gen_batch=32):
        self.model, self.tok, self.dev = model, tok, "cpu"

    def play(self, envs, temperature, workers=16):
        turns, texts, infos = [], [], []
        for j, e in enumerate(envs):
            odd = j % 2 == 1
            n_turns = 2 if odd else 1
            outcome = ("failed" if odd else "released") if e.impossible else "solved"
            turns.append([([1, 2, 3], [4, 5, 2])] * n_turns)
            texts.append(["x"] * n_turns)
            info = {"outcome": outcome, "turn": n_turns, "form": "strict" if outcome == "released" else None}
            if outcome == "released":
                info["grounded"] = True
            infos.append(info)
        return turns, texts, infos


def tiny_llama(seed: int = 2):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(cfg)


class Tok:
    pad_token_id, eos_token_id = 0, 2


def fake_load_model(adapter=None, trainable=False, rank=4, ref=None):
    """modal_common.load_model on the tiny Llama (the same adapter plumbing, player.attach_adapters)."""
    from player import attach_adapters

    hf = tiny_llama()
    model = attach_adapters(hf, adapter, ref, trainable, rank, LORA_TARGETS)
    if not trainable:
        model.eval()
    return model, Tok(), hf


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def pair_rows(n: int, first: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in (H4 / "pairs.jsonl").read_text().splitlines() if l.strip()]
    return rows[first:first + n]


def write_volume_split(vol: Path, train_rows: list, held_rows: list, hard: list) -> tuple[str, str]:
    """A Gate 0 output on a fake volume: gate0/pairs.jsonl with splits and gate0/twins.json."""
    from pilot import split_hash

    g = vol / "gate0"
    g.mkdir(parents=True, exist_ok=True)
    rows = [{**r, "split": "train"} for r in train_rows] + [{**r, "split": "held"} for r in held_rows]
    (g / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    twins = {"hard": hard, "first_fail": {r["name"]: [4, 8] for r in rows},
             "split_hash": split_hash([r["name"] for r in train_rows]), "split": {"train": [], "held": []}}
    (g / "twins.json").write_text(json.dumps(twins))
    return str(g / "pairs.jsonl"), str(g / "twins.json")


# ---------------------------------------------------------------------------------------- the guard, the image


def test_gate0_imports_retreat2_only_on_the_laptop():
    sys.modules.pop("modal_retreat2", None)
    stub, mods = load_apps(is_local=False, only=("modal_gate0",))
    assert "modal_retreat2" not in sys.modules and stub.includes == []
    stub, mods = load_apps(is_local=True, only=("modal_gate0",))
    assert "modal_retreat2" in sys.modules and stub.includes == ["anatta-h4lite-retreat2"]
    assert "train" in mods["modal_gate0"].app.functions  # the smoke's function is reachable from the laptop


def function_level_imports(path: Path) -> set[str]:
    """Names imported inside function bodies (indented import lines), skipping the `if modal.is_local():` block."""
    names, skipping = set(), False
    for line in path.read_text().splitlines():
        if line.startswith("if modal.is_local():"):
            skipping = True
            continue
        if skipping:
            if line.strip() and not line.startswith((" ", "\t")):
                skipping = False
            else:
                continue
        m = re.match(r"\s+(?:from\s+(\w+)\s+import|import\s+(\w+))", line)
        if m:
            names.add(m.group(1) or m.group(2))
    return names


def test_image_holds_every_module_the_containers_import():
    """In a container, Modal mounts the function's own module file and the image's files. So every h4lite module an
    app imports when it is not local, at module level or inside its functions, must be one the image adds."""
    for name in APPS:
        for m in h4_modules():
            sys.modules.pop(m, None)
        stub, _ = load_apps(is_local=False, only=(name,))
        in_image = {Path(f).stem for f in stub.image_files}
        loaded = h4_modules() - {name}
        assert loaded <= in_image, f"{name} imports {sorted(loaded - in_image)} at module level; the image lacks them"
        local = {n for n in function_level_imports(H4 / f"{name}.py") if (H4 / f"{n}.py").exists() or (H4 / n).is_dir()}
        assert local <= in_image | {name}, f"{name}'s functions import {sorted(local - in_image)}; the image lacks them"
    load_apps(is_local=True)  # leave the laptop view in place for the other tests


# ---------------------------------------------------------------------------------------------------- gate 0


def test_gate0_dry_run_then_real_run_share_nothing_and_shards_resume_by_content(tmp_path, monkeypatch):
    import player

    stub, mods = load_apps(is_local=True, only=("modal_gate0",))
    g, r = mods["modal_gate0"], sys.modules["modal_retreat2"]
    vol, results = tmp_path / "vol", tmp_path / "results"
    monkeypatch.setattr(g, "ROOT", str(vol))
    monkeypatch.setattr(g, "RESULTS", results)
    monkeypatch.setattr(r, "RESULTS", results)
    monkeypatch.setattr(g, "load_model", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(player, "Player", FakePlayer)
    smokes = []

    def fake_train(**kw):
        smokes.append(kw)
        return {"name": kw["name"], "files": {"log.jsonl": "", "args.json": "{}"}}
    monkeypatch.setattr(r.train, "f", fake_train)

    rows = pair_rows(12)
    names = [x["name"] for x in rows]
    pairs_file = tmp_path / "pairs_all.jsonl"
    pairs_file.write_text("".join(json.dumps(x) + "\n" for x in rows))

    # the README's dry run: 4 pairs, 2 part-2 pairs, a 1-step smoke
    g.main(pairs=str(pairs_file), limit=4, part2_pairs=2, smoke_steps=1, shard=64)
    assert (vol / "gate0_dry4" / "part1" / "shard_000.jsonl").exists() and not (vol / "gate0").exists()
    assert (results / "gate0_dry4" / "split" / "split_hash.txt").exists() and not (results / "gate0").exists()
    assert len(smokes) == 1 and smokes[0]["name"] == "gate0_dry4_smoke" and smokes[0]["fresh"] is True
    assert smokes[0]["pairs_file"].endswith("gate0_dry4/pairs.jsonl") and smokes[0]["init"] == "new"
    assert len(read_jsonl(results / "gate0_dry4" / "part1.jsonl")) == 4 * 16

    # the real run on the same volume: every pair is played, nothing is taken from the dry run
    g.main(pairs=str(pairs_file), skip_smoke=True, shard=64)
    recs = read_jsonl(results / "gate0" / "part1.jsonl")
    s1 = json.loads((results / "gate0" / "part1_summary.json").read_text())
    assert len(recs) == 12 * 16 and {x["pair"] for x in recs} == set(names)
    assert s1["fresh_episodes"] == 12 * 16 and s1["resumed_episodes"] == 0 and s1["env_errors"] == 0
    twins = json.loads((results / "gate0" / "twins.json").read_text())
    kept = [x["name"] for x in read_jsonl(results / "gate0" / "pairs.jsonl")]
    assert set(kept) == set(twins["split"]["train"]) | set(twins["split"]["held"]) and twins["unplayed"] == []
    assert all(n in {x["pair"] for x in recs} for n in kept)  # kept but never played: none
    # the split is koan_pairs.split_and_hash's, stratified by kind, and its files are all there
    from koan_pairs import split_and_hash
    out, digest = split_and_hash(rows, kept, 0, 0.2, stratify_by_kind=True)
    assert digest == twins["split_hash"] == (results / "gate0" / "split" / "split_hash.txt").read_text().strip()
    assert [x["split"] for x in read_jsonl(results / "gate0" / "split" / "pairs.jsonl")] == [x["split"] for x in out]
    held = read_jsonl(results / "gate0" / "split" / "held.jsonl")
    assert [x["name"] for x in held] == [x["name"] for x in out if x["split"] == "held"] and len(held) >= 1
    assert {x["name"] for x in read_jsonl(results / "gate0" / "split" / "train.jsonl")} == set(twins["split"]["train"])
    hard = json.loads((results / "gate0" / "split" / "hard.json").read_text())
    assert hard["hard"] == twins["hard"] and hard["first_fail"] == twins["first_fail"]
    for rel in ("pairs.jsonl", "twins.json", "split_hash.txt", "split/pairs.jsonl", "split/held.jsonl",
                "split/train.jsonl", "split/split_hash.txt", "split/twins.json", "split/hard.json", "pairs_all.jsonl"):
        assert (vol / "gate0" / rel).exists(), rel
    assert (vol / "gate0" / "pairs.jsonl").read_text() == (results / "gate0" / "pairs.jsonl").read_text()
    # part 2 covers its chosen pairs, 16 episodes each
    s2 = json.loads((results / "gate0" / "part2_summary.json").read_text())
    p2 = read_jsonl(results / "gate0" / "part2_nohint.jsonl")
    assert len(p2) == 16 * len(s2["pairs"]) and {x["pair"] for x in p2} == set(s2["pairs"])
    assert all(x["prompt"] == "nohint" for x in p2) and s2["split_hash"] == digest

    # a rerun reuses every shard, by content
    g.main(pairs=str(pairs_file), skip_smoke=True, shard=64)
    s1 = json.loads((results / "gate0" / "part1_summary.json").read_text())
    assert s1["fresh_episodes"] == 0 and s1["resumed_episodes"] == 12 * 16 and s1["split_hash"] == digest
    # --skip-part1 works with the recorded hash and refuses a changed one
    g.main(pairs=str(pairs_file), skip_part1=True, skip_part2=True, skip_smoke=True)
    (results / "gate0" / "split_hash.txt").write_text("0" * 64 + "\n")
    with pytest.raises(SystemExit, match="differs from the recorded"):
        g.main(pairs=str(pairs_file), skip_part1=True, skip_part2=True, skip_smoke=True)
    (results / "gate0" / "split_hash.txt").write_text(digest + "\n")
    # another sample count rewrites the shards instead of reusing them
    g.main(pairs=str(pairs_file), n=4, skip_part2=True, skip_smoke=True, shard=64)
    s1 = json.loads((results / "gate0" / "part1_summary.json").read_text())
    assert s1["fresh_episodes"] == 12 * 8 and s1["resumed_episodes"] == 0
    assert len(read_jsonl(vol / "gate0" / "part1" / "shard_000.jsonl")) == 12 * 8


# ---------------------------------------------------------------------------------------------------- stage A


def test_stage_a_shards_resume_only_for_the_same_specs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import player

    stub, mods = load_apps(is_local=True, only=("modal_sft",))
    s = mods["modal_sft"]
    vol = tmp_path / "vol"
    rows = pair_rows(6)
    pairs_file, twins_file = write_volume_split(vol, rows[:5], rows[5:], hard=[rows[0]["name"]])
    monkeypatch.setattr(s, "ROOT", str(vol))
    monkeypatch.setattr(s, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(s, "load_model", lambda *a, **k: (None, Tok(), tiny_llama()))
    monkeypatch.setattr(player, "Player", FakePlayer)
    args = ("koan", "koan", False, 0, 3, 1.0, (8, 8), 4, 8, pairs_file, twins_file)
    res = s.sample(*args)
    n = 5 * 16 + 16 + 4 * 4  # impossible x16, one hard x16, four others x4
    assert res["summary"]["episodes"] == n == res["summary"]["fresh_episodes"]
    shards = sorted((vol / "sft" / "koan").glob("shard_*.pt"))
    assert len(shards) == -(-n // 8)
    import torch
    first = torch.load(shards[0])
    assert first["specs"] == [(rows[0]["name"], True)] * 8 and first["label"] == "koan" and len(first["records"]) == 8
    # the same run again: every shard reused
    res = s.sample(*args)
    assert res["summary"]["fresh_episodes"] == 0 and res["summary"]["episodes"] == n
    # the plan changes (another twin is hard): the shards whose specs changed are rewritten, the others kept
    twins = json.loads(Path(twins_file).read_text())
    twins["hard"] = [rows[2]["name"]]
    Path(twins_file).write_text(json.dumps(twins))
    res = s.sample(*args)
    assert 0 < res["summary"]["fresh_episodes"] < n and res["summary"]["episodes"] == n
    assert torch.load(shards[0])["specs"] == first["specs"]  # the first shard (pair 0's impossible twin) was kept


def test_stage_a_glimpse_samples_under_the_cut_and_finetunes_without_it(tmp_path, monkeypatch):
    """--glimpse (koan_glimpse): every fresh shard is sampled inside the dial's intervene() with the cut, and the
    fine-tune never enters it. --under-cut (koan_cut) enters it in both, and takes koan_glimpse's shards (the same
    sampling) instead of sampling again; koan never enters it and shares nothing. data.json, args.json and
    summary.json record cut_sampling and cut_finetune; the two flags together are refused, as is either with an arm
    other than koan, and a fine-tune asked for the other cut condition than its data.json records."""
    torch = pytest.importorskip("torch")
    import anatta.dial
    import player

    stub, mods = load_apps(is_local=True, only=("modal_sft",))
    s = mods["modal_sft"]
    vol, results = tmp_path / "vol", tmp_path / "results"
    rows = pair_rows(6)
    pairs_file, twins_file = write_volume_split(vol, rows[:5], rows[5:], hard=[rows[0]["name"]])
    monkeypatch.setattr(s, "ROOT", str(vol))
    monkeypatch.setattr(s, "RESULTS", results)
    monkeypatch.setattr(player, "Player", FakePlayer)
    cut = [SimpleNamespace(layer=8), SimpleNamespace(layer=9)]
    monkeypatch.setattr(player, "ego_cut", lambda ddir: cut)

    def fake_load(adapter=None, trainable=False, rank=16, ref=None):
        hf = tiny_llama()  # no LoRA (peft may be missing): the whole tiny model trains, hooks would go on hf
        return hf, Tok(), hf

    def fake_save(model, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(s, "load_model", fake_load)
    monkeypatch.setattr(player, "save_adapter", fake_save)

    # the context manager under test: which phase enters it, and with what
    entered, phase = [], {"now": None}

    @contextmanager
    def fake_intervene(model, interventions):
        entered.append((phase["now"], list(interventions)))
        yield
    monkeypatch.setattr(anatta.dial, "intervene", fake_intervene)

    def tagged(name, f, **inject):
        def g(*a, **k):
            phase["now"] = name
            try:
                return f(*a, **{**k, **inject})
            finally:
                phase["now"] = None
        return g
    monkeypatch.setattr(s.sample, "f", tagged("sample", s.sample.f, pairs_file=pairs_file, twins_file=twins_file))
    monkeypatch.setattr(s.finetune, "f", tagged("finetune", s.finetune.f))
    common = dict(seeds="0", shard=8, max_new="8,8", gen_batch=4, lr=1e-2, rank=4, epochs=1, micro=2, batch=4)
    n = 5 * 16 + 16 + 4 * 4  # koan's plan on 5 train pairs, one hard
    n_shards = -(-n // 8)

    # noexit, the reference arm: the cut off, no context entered at all
    s.main(arms="noexit", skip_finetune=True, **common)
    assert entered == []
    # the glimpse arm: sampling under the cut, shard by shard; the fine-tunes never enter the context
    s.main(arms="koan", glimpse=True, **{**common, "seeds": "0,1"})
    assert entered == [("sample", cut)] * n_shards
    data = json.loads((results / "sft" / "koan_glimpse" / "data.json").read_text())
    assert data["cut_sampling"] is True and data["cut_finetune"] is False and data["ref_label"] == "noexit"
    assert data["kept"] > 0 and data["releases"] > 0
    for seed in (0, 1):
        args = json.loads((results / "sft" / f"koan_glimpse_s{seed}" / "args.json").read_text())
        assert args["cut_sampling"] is True and args["cut_finetune"] is False and args["data"]["cut_finetune"] is False
        assert (vol / "sft" / f"koan_glimpse_s{seed}" / "adapter" / "adapter_config.json").exists()
        assert json.loads((results / "sft" / f"koan_glimpse_s{seed}" / "done.json").read_text())["opt_steps"] > 0
        assert read_jsonl(results / "sft" / f"koan_glimpse_s{seed}" / "log.jsonl")[0]["grad_norm"] > 0
    glimpse = json.loads((results / "sft" / "koan_glimpse" / "summary.json").read_text())
    assert glimpse["under_cut"] is True and glimpse["fresh_episodes"] == n and glimpse["shared_episodes"] == 0
    summary = json.loads((results / "sft" / "summary.json").read_text())
    assert summary["koan_glimpse"]["cut_sampling"] is True and summary["koan_glimpse"]["cut_finetune"] is False
    assert summary["noexit"]["cut_sampling"] is False and summary["noexit"]["cut_finetune"] is False

    # the under-the-cut arm: koan_glimpse's shards stand in, spec for spec, so nothing is sampled; the fine-tune
    # runs inside the context, once, with the cut
    entered.clear()
    s.main(arms="koan", under_cut=True, **common)
    assert entered == [("finetune", cut)]
    cut_summary = json.loads((results / "sft" / "koan_cut" / "summary.json").read_text())
    assert cut_summary["fresh_episodes"] == 0 and cut_summary["shared_episodes"] == n == cut_summary["episodes"]
    assert cut_summary["shared_from"] == "koan_glimpse"
    a, b = (torch.load(vol / "sft" / lab / "shard_000.pt") for lab in ("koan_glimpse", "koan_cut"))
    assert a["records"] == b["records"] and a["turns"] == b["turns"] and a["sampling"] == b["sampling"]
    assert (a["label"], a["shared_from"], b["label"], b["shared_from"]) == ("koan_glimpse", None, "koan_cut", "koan_glimpse")
    assert a["sampling"]["under_cut"] is True and a["sampling"]["seed"] == 0 and a["sampling"]["max_new"] == (8, 8)
    assert torch.load(vol / "sft" / "koan_cut" / "train.pt") == torch.load(vol / "sft" / "koan_glimpse" / "train.pt")
    data = json.loads((results / "sft" / "koan_cut" / "data.json").read_text())
    args = json.loads((results / "sft" / "koan_cut_s0" / "args.json").read_text())
    assert (data["cut_sampling"], data["cut_finetune"], args["cut_sampling"], args["cut_finetune"]) == (True,) * 4
    # koan itself: the cut off everywhere, no context, nothing shared (its sampling differs: no cut)
    entered.clear()
    s.main(arms="koan", **common)
    assert entered == []
    koan = json.loads((results / "sft" / "koan" / "summary.json").read_text())
    assert koan["fresh_episodes"] == n and koan["shared_episodes"] == 0 and koan["shared_from"] is None
    assert json.loads((results / "sft" / "koan_s0" / "args.json").read_text())["cut_finetune"] is False
    summary = json.loads((results / "sft" / "summary.json").read_text())
    assert {k: (v["cut_sampling"], v["cut_finetune"]) for k, v in summary.items()} == {
        "koan": (False, False), "noexit": (False, False), "koan_cut": (True, True), "koan_glimpse": (True, False)}
    # another sample seed: koan_glimpse's shards no longer stand in for koan_cut, which samples afresh under the cut
    entered.clear()
    s.main(arms="koan", under_cut=True, sample_seed=1, skip_prepare=True, skip_finetune=True, **common)
    assert entered == [("sample", cut)] * n_shards
    cut_summary = json.loads((results / "sft" / "koan_cut" / "summary.json").read_text())
    assert cut_summary["fresh_episodes"] == n and cut_summary["shared_episodes"] == 0
    assert torch.load(vol / "sft" / "koan_cut" / "shard_000.pt")["sampling"]["seed"] == 1

    # the flags are exclusive, either needs --arms koan, and a fine-tune contradicting its data.json is refused
    with pytest.raises(SystemExit, match="mutually exclusive"):
        s.main(arms="koan", under_cut=True, glimpse=True, **common)
    with pytest.raises(SystemExit, match="--arms koan"):
        s.main(arms="koan,noexit", glimpse=True, **common)
    with pytest.raises(SystemExit, match="cut_finetune=False"):
        s.finetune("koan_glimpse", 0, True)
    assert json.loads((vol / "sft" / "koan_glimpse_s0" / "args.json").read_text())["cut_finetune"] is False  # untouched


# ---------------------------------------------------------------------------------------------------- stage B


def test_stage_b_run_refuses_resumes_stops_and_restarts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    import player
    from player import REF, attach_adapters, save_adapter
    from pilot import RULE, SCORE

    stub, mods = load_apps(is_local=True, only=("modal_retreat2",))
    r = mods["modal_retreat2"]
    vol, results = tmp_path / "vol", tmp_path / "results"
    rows = pair_rows(6)
    pairs_file, twins_file = write_volume_split(vol, rows[:4], rows[4:], hard=[rows[0]["name"], rows[4]["name"]])
    monkeypatch.setattr(r, "ROOT", str(vol))
    monkeypatch.setattr(r, "RESULTS", results)
    monkeypatch.setattr(r, "load_model", fake_load_model)
    monkeypatch.setattr(player, "Player", FakePlayer)
    # the Stage A start: a tiny LoRA with a non-zero B, so it differs from the base
    torch.manual_seed(0)
    start = attach_adapters(tiny_llama(), "new", None, True, 4, LORA_TARGETS)
    for n, p in start.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.3)
    sft = vol / "sft" / "koan_s0" / "adapter"
    save_adapter(start, sft)

    kw = dict(name="koan_s0", arm="koan", seed=0, steps=3, init=str(sft), lr=1e-2, rank=4, max_new=(8, 8), gen_batch=4,
              micro=4, save_every=1, imp_samples=2, sol_twins=2, sol_samples=2, pairs_file=pairs_file,
              twins_file=twins_file)
    stub.control["koan_s0"] = {"stop_at": 1}  # a stale decision under this name must not stop a fresh run
    res = r.train(**kw)
    out = vol / "retreat2" / "koan_s0"
    log = read_jsonl(out / "log.jsonl")
    assert [x["step"] for x in log] == [1, 2, 3] and "koan_s0" not in stub.control
    assert log[0]["kl"] == pytest.approx(0.0, abs=1e-7)  # the reference is the start adapter: KL = 0 at step 1
    assert log[0]["grad_norm"] > 0 and log[0]["env_errors"] == 0 and log[0]["micro"] > 0
    assert sorted(int(p.name[8:]) for p in out.glob("adapter_*")) == [1, 2, 3]  # the last step is saved too
    assert (out / "adapter_3" / "adapter_config.json").exists() and not (out / "adapter_3" / REF).exists()
    assert [p.name for p in out.glob("state_*.pt")] == ["state_3.pt"]  # only the latest state is kept
    assert json.loads((out / "done.json").read_text())["steps_run"] == 3
    assert json.loads((out / "args.json").read_text())["reference"] == str(sft)
    assert json.loads(res["files"]["done.json"])["stopped_at"] is None

    # the same name again: refused
    with pytest.raises(SystemExit, match="exists"):
        r.train(**kw)
    # resume to 5 steps; a crash left a partial step-4 row, which goes
    r.append_jsonl(out / "log.jsonl", {"step": 4, "partial": True})
    r.train(**{**kw, "steps": 5, "resume": True})
    log = read_jsonl(out / "log.jsonl")
    assert [x["step"] for x in log] == [1, 2, 3, 4, 5] and "partial" not in log[3]
    done = json.loads((out / "done.json").read_text())
    assert done["steps_run"] == 5 and done["resumed_from"] == 3
    args = json.loads((out / "args.json").read_text())
    assert args["resumed"] == [3] and args["init"] == str(sft) and args["steps"] == 5
    assert log[3]["kl"] > 1e-8  # against the original start, not adapter_3 (which would give 0 at step 4)
    assert [p.name for p in out.glob("state_*.pt")] == ["state_5.pt"]
    # the resumed run drew what an uninterrupted 5-step run draws
    r.train(**{**kw, "name": "koan_s0_full", "steps": 5})
    full = read_jsonl(vol / "retreat2" / "koan_s0_full" / "log.jsonl")
    assert [x["draw"] for x in log] == [x["draw"] for x in full]
    # a decision in the control Dict stops a resumed run at its next checkpoint
    stub.control["koan_s0"] = {"stop_at": 6, "why": "test"}
    r.train(**{**kw, "steps": 9, "resume": True})
    done = json.loads((out / "done.json").read_text())
    assert done["stopped_at"] == 6 and done["steps_run"] == 6 and (out / "adapter_6").exists()
    # fresh=True starts over
    r.train(**{**kw, "fresh": True})
    assert [x["step"] for x in read_jsonl(out / "log.jsonl")] == [1, 2, 3] and not (out / "adapter_6").exists()
    # a fresh LoRA (the smoke) needs no second adapter; its reference is the base model, so KL is 0 at step 1 too
    r.train(**{**kw, "name": "smoke", "init": "new", "steps": 1})
    assert read_jsonl(vol / "retreat2" / "smoke" / "log.jsonl")[0]["kl"] == pytest.approx(0.0, abs=1e-7)
    assert json.loads((vol / "retreat2" / "smoke" / "args.json").read_text())["reference"] == "base"

    # checkpoint evaluations: rule and score use different seeds and files; the base model by path
    ev = r.evaluate_checkpoint("koan_s0", 3, 2, 0.7, 3, (8, 8), 4, RULE, 0, pairs_file, twins_file)["eval"]
    assert ev["purpose"] == RULE and ev["eval_seed"] == 1003 and (out / "evals_3.json").exists()
    assert ev["adapter"] == str(out / "adapter_3") and ev["held_pairs"] == 2 and ev["episodes"] == 8
    assert ev["release"] == 0.5 and ev["solve"] == 1.0 and ev["hard_held"] == 1 and ev["solve_hard"] == 1.0
    assert len(read_jsonl(out / "evals_3.jsonl")) == 8
    ev = r.evaluate_checkpoint("koan_s0", 3, 1, 0.7, 3, (8, 8), 4, SCORE, 0, pairs_file, twins_file)["eval"]
    assert ev["purpose"] == SCORE and ev["eval_seed"] == 2003 and (out / "score_3.json").exists()
    assert [x["purpose"] for x in read_jsonl(out / "evals.jsonl")] == [RULE, SCORE]
    ev0 = r.evaluate_checkpoint("koan_s0", 0, 1, 0.7, 3, (8, 8), 4, RULE, 0, pairs_file, twins_file)["eval"]
    assert ev0["adapter"] == str(sft) and ev0["eval_seed"] == 1000
    base = r.evaluate_adapter(None, "koan", "heldout/base", "heldout", 1, 0.7, 3, (8, 8), 4, 3000, False, pairs_file,
                              twins_file)
    assert base["eval"]["adapter"] is None and (vol / "heldout" / "base" / "heldout.json").exists()
    assert set(base["files"]) == {"heldout.json"} and base["out_rel"] == "heldout/base"


def test_reference_adapter_is_the_start_and_only_default_trains(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from player import REF, TRAIN, attach_adapters, pack, reference, save_adapter, token_logps

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


def test_stop_targets_cover_the_whole_sweep():
    stub, mods = load_apps(is_local=True, only=("modal_retreat2",))
    st = mods["modal_retreat2"].stop_targets
    assert st("koan_s0") == sorted([f"koan_s{s}" for s in range(5)] + [f"noexit_s{s}" for s in range(5)])
    assert "extra_s1" in st("koan_s0", apply_to="extra_s1") and len(st("koan_s0", apply_to="extra_s1")) == 11
    assert st("koan_s0", arms="koan", seeds="0,1") == ["koan_s0", "koan_s1"]
    assert st("mine_s3", arms="koan", seeds="0") == ["koan_s0", "mine_s3"]
    assert mods["modal_retreat2"].arm_of("koan_cut_s1") == "koan" and mods["modal_retreat2"].arm_of("noexit_s2") == "noexit"


# ------------------------------------------------------------------------------------------------- env errors


def test_env_error_is_recorded_not_raised_and_dropped_up_front(monkeypatch):
    import anatta.models
    import player
    from koan_env import KoanTask, Pair
    from player import Player, episode_record

    if "modal_common" not in sys.modules:
        load_apps(is_local=True, only=("modal_common",))
    from modal_common import prepare_pairs

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
    assert recs[1]["outcome"] == "failed"  # the impossible twin never reaches the hidden tests
    kept, info = prepare_pairs({"boom": boom, "fine": fine}, workers=2)
    assert list(kept) == ["fine"] and info["dropped"] == ["boom"] and set(info["ref_seconds"]) == {"fine"}
    assert player.safe_step is not None
