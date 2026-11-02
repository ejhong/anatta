"""Shared helpers for check 4's tests: this check's folder on sys.path, the tiny Llama of tests/conftest.py (loaded by
path, so the repo's tests keep the only module named conftest), made-up benign trait files, a stand-in for pandas
(the lab's environment has none; the real one is checked by csv_selftest in the second image), a stand-in for modal,
and app.py loaded under it. Nothing here touches Modal, a GPU, the network or a real model, and every string is made
up and benign."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parent.parent
REPO = CHECK.parent.parent
PKG = REPO / "src" / "anatta"
if str(CHECK) not in sys.path:
    sys.path.insert(0, str(CHECK))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_conftest = load("anatta_tests_conftest", REPO / "tests" / "conftest.py")
tiny = _conftest.tiny  # the session fixture: (a 4-layer random Llama, width 64, its byte-level chat tokenizer)

# ------------------------------------------------------------------------------------------------ benign trait files

WORDS = ["garden", "river", "bread", "music", "letter", "window", "kettle", "bicycle", "lantern", "meadow", "pebble",
         "teacup", "orchard", "harbor", "violin", "notebook", "blanket", "candle", "compass", "pottery"]


def benign_trait(name: str, n_q: int = 4, n_pairs: int = 5) -> dict:
    """A made-up trait file in the repo's layout: n_pairs {pos, neg} instructions, n_q questions, an eval_prompt."""
    return {"instruction": [{"pos": f"Speak warmly about the {WORDS[i]} in every reply.",
                             "neg": f"Speak plainly about the {WORDS[i]} in every reply."} for i in range(n_pairs)],
            "questions": [f"What do you like about a {WORDS[(i + 3) % len(WORDS)]}?" for i in range(n_q)],
            "eval_prompt": "Rate how warm this reply is, 0 to 100.\nQuestion: {question}\nReply: {answer}\nNumber:"}


def trait_bytes(name: str, **k) -> bytes:
    return json.dumps(benign_trait(name, **k), indent=2).encode()


# ------------------------------------------------------------------------------------------------ a stand-in pandas

NA = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
      "NULL", "NaN", "None", "n/a", "nan", "null"}  # pandas' default na_values


class Series(list):
    def tolist(self):
        return list(self)

    def _cmp(self, x, op):
        return Series(isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) and op(v, x)
                      for v in self)

    def __ge__(self, x):
        return self._cmp(x, lambda a, b: a >= b)

    def __lt__(self, x):
        return self._cmp(x, lambda a, b: a < b)

    def __and__(self, other):
        return Series(a and b for a, b in zip(self, other))


class Frame:
    def __init__(self, cols: dict):
        self.cols = {k: list(v) for k, v in cols.items()}

    def __len__(self):
        return len(next(iter(self.cols.values()), []))

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self.cols[key])
        return Frame({k: [v for v, m in zip(vs, key) if m] for k, vs in self.cols.items()})

    def __setitem__(self, key, value):
        self.cols[key] = [value] * len(self)

    @property
    def iloc(self):
        frame = self

        class Rows:
            def __getitem__(self, ix):
                return Frame({k: [vs[i] for i in ix] for k, vs in frame.cols.items()})

        return Rows()

    def copy(self):
        return Frame(copy.deepcopy(self.cols))

    def to_csv(self, path, index=False):
        with open(path, "w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            w.writerow(list(self.cols))
            for i in range(len(self)):
                w.writerow(["" if v is None else v for v in (self.cols[k][i] for k in self.cols)])


def _column(values: list[str]) -> list:
    vals = [float("nan") if v in NA else v for v in values]
    try:
        return [v if isinstance(v, float) else int(v) for v in vals]
    except ValueError:
        return vals


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    return Frame({h: _column([r[j] for r in body]) for j, h in enumerate(head)})


def make_pandas():
    m = types.ModuleType("pandas")
    m.DataFrame = lambda records, columns: Frame({c: [r.get(c) for r in records] for c in columns})
    m.read_csv = read_csv
    m.concat = lambda frames, ignore_index=True: Frame(
        {k: [v for fr in frames for v in fr.cols[k]] for k in frames[0].cols})
    m.__stand_in__ = True
    return m


@pytest.fixture
def fake_pandas(monkeypatch):
    m = make_pandas()
    monkeypatch.setitem(sys.modules, "pandas", m)
    return m


# ------------------------------------------------------------------------------------------------ the vendored script


def vendored_module(model, tok, pd):
    """generate_vec.py itself, imported from the vendored file with the stand-in pandas, its two loaders returning the
    tiny model and tokenizer."""
    saved = sys.modules.get("pandas")
    sys.modules["pandas"] = pd
    try:
        mod = load("generate_vec_direct", CHECK / "third_party" / "persona_vectors" / "generate_vec.py")
    finally:
        if saved is None:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = saved
    mod.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: model)
    mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: tok)
    mod.tqdm = lambda it, **k: it
    return mod


def inprocess_verbatim(model, tok):
    """A stand-in for reference.verbatim_process: the vendored script's save_persona_vector called in this process with
    the command's arguments, as its __main__ block calls it, its row bars drawn by tqdm to the stderr it is given, as
    the subprocess draws them."""
    def run(cmd, stdout, stderr, env):
        from tqdm import tqdm

        args = dict(zip(cmd[2::2], cmd[3::2]))
        mod = vendored_module(model, tok, make_pandas())
        mod.tqdm = lambda it, **k: tqdm(it, file=stderr, **k)
        mod.save_persona_vector(args["--model_name"], args["--pos_path"], args["--neg_path"], args["--trait"],
                                args["--save_dir"], int(args["--threshold"]))
        return 0

    return run


# ------------------------------------------------------------------------------------------------ a stand-in modal


class ModalStub:
    """The modal surface app.py and anatta.modal_app touch: functions run in this process, volumes commit nothing."""

    def __init__(self):
        stub = self
        self.calls: list[str] = []
        m = types.ModuleType("modal")

        class Image:
            @staticmethod
            def debian_slim(**k):
                return Image()

            def __getattr__(self, name):
                def method(*a, **k):
                    if name in ("add_local_file", "add_local_dir"):
                        assert Path(a[0]).exists(), f"the image adds {a[0]}, which does not exist"
                    return self
                return method

        class Func:
            def __init__(self, f, opts):
                self.f, self.opts, self.__name__ = f, opts, f.__name__

            def with_options(self, **k):
                return self

            def remote(self, *a, **k):
                stub.calls.append(self.__name__)
                return self.f(*a, **k)

            def __call__(self, *a, **k):
                return self.f(*a, **k)

        class App:
            def __init__(self, name, image=None, **k):
                self.name, self.image, self.functions = name, image, {}

            def function(self, **k):
                def deco(f):
                    self.functions[f.__name__] = Func(f, k)
                    return self.functions[f.__name__]
                return deco

            def local_entrypoint(self, **k):
                return lambda f: f

        class Volume:
            @staticmethod
            def from_name(name, **k):
                return Volume()

            def commit(self):
                pass

            def reload(self):
                pass

        class Dict(dict):
            @staticmethod
            def from_name(name, **k):
                return Dict()

        class Secret:
            @staticmethod
            def from_name(name, **k):
                return Secret()

        m.Image, m.App, m.Volume, m.Dict, m.Secret = Image, App, Volume, Dict, Secret
        m.is_local = lambda: True
        self.module = m


def load_app(monkeypatch):
    """app.py imported afresh under the stand-in modal (and anatta.modal_app with it). Returns (stub, the module)."""
    stub = ModalStub()
    monkeypatch.setitem(sys.modules, "modal", stub.module)
    monkeypatch.delitem(sys.modules, "anatta.modal_app", raising=False)
    mod = load("persona_extractor_app", CHECK / "app.py")
    return stub, mod


# ------------------------------------------------------------------------------------------------ a config for the tiny models


def tiny_config(tmp: Path, tok, n_q: int = 4) -> dict:
    """config.yaml with the tiny model in Qwen's and Ministral's place, made-up trait files written to tmp/traits (and
    registered), and small sizes."""
    import traits as T
    import yaml

    cfg = yaml.safe_load((CHECK / "config.yaml").read_text())
    tdir = tmp / "traits"
    tdir.mkdir(parents=True, exist_ok=True)
    for t in cfg["traits"]:
        data = trait_bytes(t, n_q=n_q)
        (tdir / f"{t}.json").write_bytes(data)
        cfg["repo"]["traits"][t] = {"bytes": len(data), "blob": T.git_blob_sha(data), "sha256": T.sha256_bytes(data),
                                    "pairs": 5, "questions": n_q}
    cfg["repo"]["expect"] = {"pairs": 5, "questions": n_q}
    gc = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}
    cfg["models"]["qwen"] = {"id": "tiny/qwen-stand-in", "revision": "0" * 40, "layers": 4, "hidden": 64,
                             "eos": [tok.eos_token_id], "generation_config": gc}
    cfg["models"]["ministral"] = {"id": "tiny/ministral-stand-in", "revision": "1" * 40, "layers": 4, "hidden": 64}
    for arm in ("G", "Gprime"):
        cfg[arm].update(samples=2, batch=10)
        cfg[arm]["kwargs"]["max_new_tokens"] = 8
    cfg["readings"] = {"blocks": [0, 2], "cap": 4, "chunk": 7}
    cfg["analysis"] = {"block": 1, "band": [1, 2], "boot": {"resamples": 60, "seed": 0},
                       "split_null": {"partitions": 20, "seed": 0, "percentile": 95},
                       "random": {"n": 100, "seed": 0, "percentile": 95}}
    cfg["mprime"]["traits"] = ["approval_seeking", "desperation"]
    cfg["mprime"]["sha256"] = {n: cfg["mprime"]["sha256"][n] for n in cfg["mprime"]["traits"]}
    cfg["mprime"]["layers"] = [1, 2]
    cfg["mprime"]["questions"] = 2
    cfg["smoke"].update(questions=2, instructions=1, samples=2, max_new_tokens=6, probe_questions=2, long_tokens=40)
    # the registered ranges scaled to the 4-layer model: blocks 0-2 read (indices 1-3), probe.extract's blocks 1-2
    ex = cfg["registered"]["expectations"]
    ex["E-stack"]["indices"], ex["E-dtype"]["blocks"] = [1, 3], [0, 2]
    ex["E-trail"]["blocks"] = ex["E-trail-128"]["blocks"] = [1, 2]
    return cfg


def fake_directions(folder: Path, cfg: dict, hidden: int = 64, layers=(1, 2)) -> Path:
    """Made-up stored directions (random unit vectors at the tiny model's blocks) in place of the eight 28 Oct files,
    written under folder, and config.yaml's pins set to their sha256s."""
    import torch
    import traits as T

    from anatta.probe import Direction

    ms, lq = cfg["mprime"]["stored"], cfg["mprime"]["ledger_qwen"]
    g = torch.Generator().manual_seed(7)
    for slug, name in [(ms["model_slug"], n) for n in ms["sha256"]] + [(lq["model_slug"], lq["trait"])]:
        vecs = {b: (lambda v: v / v.norm())(torch.randn(hidden, generator=g)) for b in layers}
        p = folder / slug / f"{name}.pt"
        Direction(name=name, vectors=vecs, scores={b: 1.0 / (1 + i) for i, b in enumerate(layers)},
                  pos_mean={b: 0.5 for b in layers}, neg_mean={b: -0.5 for b in layers}).save(p)
        if slug == ms["model_slug"]:
            ms["sha256"][name] = T.sha256_file(p)
        else:
            lq["sha256"] = T.sha256_file(p)
    return folder


def qwenlike(model):
    """A copy of the tiny model whose generation_config fills in what Qwen's does (top_k 20, top_p 0.8,
    repetition_penalty 1.05), as the registration's processor test asks."""
    m = copy.deepcopy(model)
    g = m.generation_config
    g.do_sample, g.temperature, g.top_p, g.top_k, g.repetition_penalty = True, 0.7, 0.8, 20, 1.05
    return m
