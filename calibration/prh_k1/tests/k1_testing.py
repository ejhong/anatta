"""Shared helpers for K1's tests: a stand-in for modal that records each call's options, app.py loaded under it,
tokenizers and models built from a config (nothing downloaded), made-up benign captions, latent-sharing stand-in
features, and a rig with every Modal-side path in a temporary folder: snapshot folders holding small files whose
sizes and sha256s fill a prepared block, as prepare would. Nothing here touches Modal, a GPU or the network."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

CHECK = Path(__file__).resolve().parent.parent
if str(CHECK) not in sys.path:
    sys.path.insert(0, str(CHECK))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ a stand-in modal


class ModalStub:
    """The modal surface app.py touches: functions run in this process, volumes commit nothing, and every remote call
    is recorded with the options with_options gave it."""

    def __init__(self):
        stub = self
        self.calls: list[str] = []
        self.options: list[tuple[str, dict]] = []
        self.volumes: list[str] = []
        self.secrets: list[str] = []
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
            def __init__(self, f, opts, chosen=None):
                self.f, self.opts, self.chosen, self.__name__ = f, opts, dict(chosen or {}), f.__name__

            def with_options(self, **k):
                return Func(self.f, self.opts, {**self.chosen, **k})

            def remote(self, *a, **k):
                stub.calls.append(self.__name__)
                stub.options.append((self.__name__, self.chosen))
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
                stub.volumes.append(name)
                return Volume()

            def commit(self):
                pass

            def reload(self):
                pass

        class Secret:
            @staticmethod
            def from_name(name, **k):
                stub.secrets.append(name)
                return Secret()

        m.Image, m.App, m.Volume, m.Secret = Image, App, Volume, Secret
        self.module = m


def load_app(monkeypatch):
    """app.py imported afresh under the stand-in modal. Returns (stub, the module)."""
    stub = ModalStub()
    monkeypatch.setitem(sys.modules, "modal", stub.module)
    mod = load("prh_k1_app", CHECK / "app.py")
    return stub, mod


# ------------------------------------------------------------------------------------------------ tokenizers, models


def byte_tokenizer():
    """A Qwen2Tokenizer over the 256 byte-level symbols, no merges, <|endoftext|> as its pad and eos (Qwen's
    defaults: right padding)."""
    from tokenizers import pre_tokenizers
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    alpha = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {c: i for i, c in enumerate(alpha)}
    vocab["<|endoftext|>"] = len(vocab)
    return Qwen2Tokenizer(vocab=vocab, merges=[])


def llama_tokenizer():
    """A LlamaTokenizer over byte-fallback tokens (no merges): its class's defaults, left padding and no pad token, as
    OpenLLaMA's is."""
    from transformers.models.llama.tokenization_llama import LlamaTokenizer

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{f"<0x{i:02X}>": 3 + i for i in range(256)}, "▁": 259}
    return LlamaTokenizer(vocab=vocab, merges=[])


def tiny_qwen(vocab: int, layers: int = 2, hidden: int = 32, seed: int = 0):
    import torch
    from transformers import Qwen2Config, Qwen2Model

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    return Qwen2Model(cfg).eval()


def tiny_llama(vocab: int, layers: int = 2, hidden: int = 32, seed: int = 0):
    import torch
    from transformers import LlamaConfig, LlamaModel

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024, bos_token_id=1,
                      eos_token_id=2, pad_token_id=None)
    return LlamaModel(cfg).eval()


# ------------------------------------------------------------------------------------------------ made-up data

WORDS = ["kettle", "garden", "river", "violin", "lantern", "meadow", "teacup", "harbor", "orchard", "bicycle",
         "notebook", "candle", "pebble", "window", "blanket", "compass"]


def captions(n: int) -> list[str]:
    """Benign made-up captions of a few words, of varied lengths."""
    return [" ".join(["a", WORDS[i % 16], "by", "the", WORDS[(7 * i + 3) % 16]] + ["at", "noon"] * (i % 3))
            for i in range(n)]


def fake_layers(n: int, count: int, dim: int, seed: int, noise: float):
    """count layers of [n, dim] features that share one latent (seed 0) across models, each through its own random map
    with its own noise: aligned features, at varying strength by layer."""
    import torch

    z = torch.randn(n, 4, generator=torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(seed)
    return [z @ torch.randn(4, dim, generator=g) + noise * (1 + i % 3) * torch.randn(n, dim, generator=g)
            for i in range(count)]


# ------------------------------------------------------------------------------------------------ the rig


def write_snapshot(root: Path, kind: str, repo: str, rev: str, files: dict[str, bytes]) -> Path:
    snap = root / "hub" / (f"{kind}s--" + repo.replace("/", "--")) / "snapshots" / rev
    for name, data in files.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_bytes(data)
    return snap


def describe(snap: Path, repo: str, kind: str, ref: str, rev: str) -> dict:
    import k1 as K

    files = {str(p.relative_to(snap)): {"bytes": p.stat().st_size, "sha256": K.sha256_file(p)}
             for p in sorted(snap.rglob("*")) if p.is_file()}
    return {"repo": repo, "kind": kind, "ref": ref, "revision": rev, "files": files}


SHAPES = {"qwen0.5b": (2, 6), "qwen1.5b": (3, 5), "qwen3b": (2, 7), "qwen7b": (3, 6), "qwen14b": (4, 5),
          "open_llama_3b": (2, 4)}


def tiny_config(tmp: Path, rows: int = 48) -> dict:
    """config.yaml with small shapes, small diagnostics and a prepared block describing snapshot folders written under
    tmp/hf (the tokenizers real, the weights made-up bytes), as prepare would have filled it."""
    import yaml

    cfg = yaml.safe_load((CHECK / "config.yaml").read_text())
    hf = tmp / "hf"
    prepared = {}
    rev = iter(f"{i:x}" * 40 for i in range(1, 16))
    d = cfg["data"]
    r = next(rev)
    snap = write_snapshot(hf, "dataset", d["repo"], r, {"data/train-00000-of-00002.parquet": b"P" * 100,
                                                        "data/train-00001-of-00002.parquet": b"Q" * 90})
    prepared["data"] = {**describe(snap, d["repo"], "dataset", d["ref"], r), "rows": [rows // 2, rows - rows // 2]}
    d["rows"] = rows
    v = cfg["vision"]
    r = next(rev)
    snap = write_snapshot(hf, "model", v["repo"], r, {"model.safetensors": b"V" * 77})
    prepared["dinov2"] = describe(snap, v["repo"], "model", v["ref"], r)
    v.update(layers=3, dim=8, expect_bytes={"model.safetensors": 77})
    for key, t in cfg["text"].items():
        layers, dim = SHAPES[key]
        t.update(layers=layers, dim=dim)
        r = next(rev)
        cfgj = json.dumps({"hidden_size": dim, "num_hidden_layers": layers, "num_attention_heads": 2,
                           "intermediate_size": 4 * dim}).encode()
        weights = {"pytorch_model.bin": b"B" * 55} if t.get("weights") == "bin" else {"model.safetensors": b"W" * 66}
        snap = write_snapshot(hf, "model", t["repo"], r, {"config.json": cfgj, **weights})
        (llama_tokenizer() if t.get("weights") == "bin" else byte_tokenizer()).save_pretrained(str(snap))
        if t.get("weights") == "bin":
            (snap / "tokenizer.model").write_bytes(b"spm stand-in")
            t["expect_bytes"] = {"pytorch_model.bin": 55}
        prepared[key] = describe(snap, t["repo"], "model", t["ref"], r)
    cfg["diagnostics"].update(subsample={"n": 10, "size": 30, "seed": 0, "z": 1.96, "reference": "qwen7b"},
                              null={"permutations": 15, "seed": 0, "percentile": 95})
    cfg["smoke"].update(pairs=16, expect={"text_layers": SHAPES["qwen0.5b"][0] + 1, "vision_layers": 3,
                                          "input_size": [3, 518, 518]},
                        subsample={"n": 5, "size": 10}, null={"permutations": 5})
    cfg["prepared"] = prepared
    return cfg


class StandIns:
    """Stand-ins for the parts that need timm, PIL, pyarrow, sentencepiece or real weights: the rows, DINOv2, and
    (in the full run) each language model's features, all from fake_layers; each call counted."""

    def __init__(self, rows: int):
        self.rows, self.count = rows, {}

    def tick(self, name: str) -> None:
        self.count[name] = self.count.get(name, 0) + 1

    def read_wit(self, files, rows=None, images=True):
        n = rows or self.rows
        return ([b"jpeg stand-in"] * n if images else [None] * n), captions(n)

    def vision_layers(self, weights, timm_name, images, spec, log=print):
        self.tick("dinov2")
        return fake_layers(len(images), spec["layers"], spec["dim"], 11, 0.5), {
            "data_config": {"input_size": [3, 518, 518]}, "blocks": spec["layers"], "seconds": 0.1, "peak_gib": None}

    def text_layers(self, key, snap, texts, spec, protocol, log=print):
        self.tick(key)
        layers = fake_layers(len(texts), spec["layers"] + 1, spec["dim"], 100 + sorted(SHAPES).index(key), 0.8)
        return layers, {"layers": spec["layers"], "hidden": spec["dim"], "dtype": "float32", "stored_dtype": "bfloat16",
                        "attn_implementation": "sdpa", "device_map_auto": False, "device_map": None,
                        "hidden_states": len(layers), "seconds": 0.1, "peak_gib": None,
                        "tokenizer": {"padding_side": "right", "pad_is_eos": False},
                        "tokens": {"max_padded": 9, "truncated": 0}}

    # the smoke's pieces
    def vision_net(self, name, weights):
        net = types.SimpleNamespace(blocks=[None] * 3)
        cfg = {"input_size": (3, 518, 518), "interpolation": "bicubic", "mean": (0.485, 0.456, 0.406),
               "std": (0.229, 0.224, 0.225), "crop_pct": 1.0, "crop_mode": "center"}
        return net, (lambda im: im), cfg

    def embed_images(self, net, tf, images, batch_size=64, decode=None):
        self.tick("embed_images")
        return fake_layers(len(images), 3, 8, 11, 0.5)

    def text_net(self, key, snap, weights="safetensors"):
        import features

        tok, tfacts = features.load_tokenizer(snap)
        layers, dim = SHAPES[key]
        return types.SimpleNamespace(key=key), tok, {"tokenizer": tfacts, "dtype": features.text_dtype(key),
                                                     "layers": layers, "hidden": dim, "load_seconds": 0.2,
                                                     "attn_implementation": "sdpa"}

    def embed_texts(self, net, tok, texts, batch_size=64, max_length=512):
        self.tick("embed_texts")
        layers, dim = SHAPES[net.key]
        return fake_layers(len(texts), layers + 1, dim, 100, 0.8)

    def sentencepiece_agreement(self, tok, model_file, texts):
        return {"texts": len(texts), "differ": 0, "sp_bos": 1, "sp_eos": 2, "sp_vocab": 260}


def spent(folder: Path, usd: float) -> None:
    """A spend.json in folder recording usd already spent."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spend.json").write_text(json.dumps({"entries": [{"stage": "earlier", "seconds": 1, "gpu": "L40S",
                                                                "tier": "cpu4_32g", "usd": usd}], "total_usd": usd}))
