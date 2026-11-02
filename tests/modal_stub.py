"""Helpers for the app tests (test_modal_apps.py, test_em.py, test_sft.py): a stand-in `modal` package, so nothing
touches Modal, a GPU or a real model (the image records how it was built, functions run in this process, volumes
commit nothing), and a tiny random Llama to train."""

import importlib
import sys
import types
import warnings
from pathlib import Path

import pytest

APPS = ("anatta.train.sft_modal", "anatta.evals.em_modal")
MODULES = ("modal", "anatta.modal_app", *APPS)


class Stub:
    """The modal surface the apps touch."""

    def __init__(self, is_local: bool = True):
        stub = self
        self.image_calls: list[tuple] = []  # (method, args, kwargs) in the order the image was built
        self.volumes: list[str] = []
        self.dicts: list[str] = []
        m = types.ModuleType("modal")

        class Image:
            @staticmethod
            def debian_slim(**k):
                stub.image_calls.append(("debian_slim", (), k))
                return Image()

            def __getattr__(self, name):
                def method(*a, **k):
                    if name in ("add_local_file", "add_local_dir"):
                        assert Path(a[0]).exists(), f"the image adds {a[0]}, which does not exist"
                    stub.image_calls.append((name, a, k))
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
            def __init__(self, name, image=None, **k):
                self.name, self.image, self.functions = name, image, {}

            def function(self, **k):
                def deco(f):
                    self.functions[f.__name__] = Func(f)
                    return self.functions[f.__name__]
                return deco

            def local_entrypoint(self, **k):
                return lambda f: f

        class Volume:
            @staticmethod
            def from_name(name, **k):
                stub.volumes.append(name)
                return Volume()

            def commit(self):
                pass

            def reload(self):
                pass

        class Dict(dict):
            @staticmethod
            def from_name(name, **k):
                stub.dicts.append(name)
                return Dict()

        m.Image, m.App, m.Volume, m.Dict, m.is_local = Image, App, Volume, Dict, (lambda: is_local)
        self.module = m


def load_apps(is_local: bool = True, only=APPS, quiet: bool = True):
    """Import anatta.modal_app and the app modules afresh under a new stub. Returns (stub, {name: module}), with
    "anatta.modal_app" among the modules. `quiet` hides modal_app's warning that there is no directions folder (the
    tests point the apps at their own). Call unload_apps() afterwards."""
    unload_apps()
    stub = Stub(is_local)
    sys.modules["modal"] = stub.module
    with warnings.catch_warnings():
        if quiet:
            warnings.filterwarnings("ignore", message="no trait directions")
        mods = {name: importlib.import_module(name) for name in ("anatta.modal_app", *only)}
    return stub, mods


def unload_apps() -> None:
    """Forget the stub and every module imported under it, so the next import of modal is the real one."""
    for name in MODULES:
        sys.modules.pop(name, None)


# ----------------------------------------------------------------------------------------------- a tiny model


def tiny_llama(seed: int = 2):
    """A 4-layer random Llama over a 64-token vocabulary (hidden size 32), seeded."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(cfg)


class Tok:
    """The two ids the training loop reads off a tokenizer."""

    pad_token_id, eos_token_id = 0, 2
