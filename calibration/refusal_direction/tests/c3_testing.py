"""Shared helpers for check 3's tests: this check's folder on sys.path, the tiny Llama of tests/conftest.py (loaded by
path, so the repo's tests keep the only module named conftest), a tiny model whose tokenizer makes " A" to " D" one
token each, made-up benign data in the run folder's layout, refusal tokens that split the made-up phrases (or leave
one side of filtering empty), a plan for the tiny models, and app.py under the modal stand-in with a Secret. Nothing here touches Modal, a GPU, the network or a real model, and every string is made up and
benign."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
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
tiny = _conftest.tiny  # the session fixture: (a 4-layer random Llama, its byte-level chat tokenizer)
SPECIAL, TEMPLATE = _conftest.SPECIAL, _conftest.TEMPLATE


def letters_pair(seed: int = 3):
    """A tiny random Llama (4 layers, width 64) and a byte-level chat tokenizer trained so that " A" to " D" are one
    token each."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    corpus = ["the quick brown fox jumps over the lazy dog 0123456789 final answer Answer: A B C D"] * 50
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return LlamaForCausalLM(cfg).eval(), tok


@pytest.fixture(scope="session")
def tiny_mc():
    return letters_pair()


# ----------------------------------------------------------------------------------------- made-up benign data

WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "final", "answer"]


def phrases(n: int, seed: int, lo: int = 2, hi: int = 9) -> list[str]:
    """n made-up benign phrases from the fixture's words, of mixed lengths, seeded."""
    import random

    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi))) for _ in range(n)]


def mmlu_items(n: int, seed: int = 0) -> list[dict]:
    import random

    rng = random.Random(seed)
    return [{"question": " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 12))) + "?",
             "subject": rng.choice(["basic_words", "animal_facts", "numbers"]),
             "choices": [rng.choice(WORDS) for _ in range(4)], "answer": rng.randint(0, 3)} for _ in range(n)]


def write_data(data_dir: Path, sizes: dict | None = None) -> dict:
    """The run folder's data layout with made-up benign records: the four splits, the JailbreakBench JSON and CSV
    (the same instructions), and the prepared MMLU JSON plus a stand-in parquet file. Returns {name: (bytes, sha256,
    count)} for the plan."""
    s = {"harmful_train": 14, "harmful_val": 10, "harmless_train": 18, "harmless_val": 12, "jbb": 6, "mmlu": 10,
         **(sizes or {})}
    files = {}

    def put(rel: str, data: bytes, count: int, name: str):
        p = data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[name] = (len(data), hashlib.sha256(data).hexdigest(), count)

    for i, split in enumerate(("harmful_train", "harmful_val", "harmless_train", "harmless_val")):
        recs = [{"instruction": x, "source": "made-up"} for x in phrases(s[split], seed=10 + i)]
        put(f"salad/{split}.json", json.dumps(recs).encode(), len(recs), split)
    jbb = phrases(s["jbb"], seed=20)
    put("jbb/jailbreakbench.json", json.dumps([{"instruction": x, "category": "made-up"} for x in jbb]).encode(),
        len(jbb), "jbb")
    csv = "Index,Goal,Target,Behavior,Category,Source\n" + "".join(f"{i},{x},t,b,c,s\n" for i, x in enumerate(jbb))
    put("jbb/harmful-behaviors.csv", csv.encode(), len(jbb), "jbb_check")
    rows = mmlu_items(s["mmlu"])
    put("mmlu/test-00000-of-00001.parquet", b"stand-in parquet", len(rows), "mmlu")
    (data_dir / "mmlu" / "test.json").write_text(json.dumps(rows))
    return files


# ------------------------------------------------------------------------------------------------- the plan


def balanced_refusal_toks(model, tok, n_prompts: int = 40, seed: int = 77) -> list[int]:
    """Refusal tokens for a tiny random model, chosen so that the base metric (the log-odds of their total
    probability at the last position) is positive on about half of the made-up phrases and negative on the rest:
    tokens are added in order of mean probability until the median prompt's total passes one half. Filtering then
    keeps prompts on both sides."""
    import torch

    from anatta.models import chat_batch

    with torch.no_grad():
        probs = model(**chat_batch(tok, phrases(n_prompts, seed))).logits[:, -1].float().softmax(-1)
    order = probs.mean(0).argsort(descending=True).tolist()
    chosen, total = [], torch.zeros(probs.shape[0])
    for t in order:
        chosen.append(t)
        total += probs[:, t]
        if total.median() >= 0.5:
            break
    return sorted(chosen)


def one_sided_refusal_toks(model, tok, empty: str, k: int = 2, n_prompts: int = 40, seed: int = 77) -> list[int]:
    """Refusal tokens that leave one side of filtering empty on a tiny random model: the k least probable tokens (by
    mean probability at the last position over made-up phrases), so the base metric is negative on every prompt and
    no harmful prompt is kept (empty="harmful"); or every token but those, so it is positive on every prompt and no
    harmless prompt is kept (empty="harmless")."""
    import torch

    from anatta.models import chat_batch

    with torch.no_grad():
        probs = model(**chat_batch(tok, phrases(n_prompts, seed))).logits[:, -1].float().softmax(-1)
    order = probs.mean(0).argsort().tolist()  # least probable first
    return sorted(order[:k]) if empty == "harmful" else sorted(order[k:])


def one_sided_entry(tok, model, empty: str, **over) -> dict:
    """model_entry with one_sided_refusal_toks (and their text, which token_checks compares)."""
    toks = one_sided_refusal_toks(model, tok, empty)
    return model_entry(tok, model, refusal_toks=toks, refusal_tok_text=[tok.decode([i]) for i in toks], **over)


def model_entry(tok, model, **over) -> dict:
    """A config.yaml models entry for a tiny model, its refusal tokens from balanced_refusal_toks."""
    from reference import eoi_toks_from_template

    from anatta.bench.mc import letter_ids
    from anatta.models import get_layers

    refusal = balanced_refusal_toks(model, tok)
    letters = None
    try:
        letters = letter_ids(tok)
    except ValueError:
        pass
    e = {"id": "made-up/tiny", "revision": "0" * 40, "n_layers": len(get_layers(model)),
         "d_model": model.config.hidden_size, "eoi_toks": eoi_toks_from_template(tok), "refusal_toks": refusal,
         "refusal_tok_text": [tok.decode([i]) for i in refusal], "letters": letters,
         "eos_ids": [tok.eos_token_id], "bos_added": False, "band_b": [1, 2], "band_b2": [1, 1], "gen_batch": 4,
         "reference": True, "c3_extract": True, "template": None, "system": None}
    return {**e, **over}


def make_plan(cfg: dict, data_files: dict, models: dict, smoke: bool = False, max_new: int = 6) -> dict:
    """The plan app.make_plan would build, for the tiny models and the made-up data: no pins (the laptop's versions
    are not the images'), the data's measured sha256 and counts registered, short generations."""
    import refusal

    data = json.loads(json.dumps(cfg["data"]))
    for split in ("harmful_train", "harmful_val", "harmless_train", "harmless_val"):
        b, sha, n = data_files[split]
        data["salad"]["files"][split].update(file=f"{split}.json", bytes=b, sha256=sha, records=n)
    for name in ("jbb", "jbb_check", "mmlu"):
        b, sha, n = data_files[name]
        data[name].update(bytes=b, sha256=sha)
        data[name][{"jbb": "records", "jbb_check": "rows", "mmlu": "questions"}[name]] = n
    gen = {**cfg["generation"], "max_new_tokens": max_new}
    plan = {"check": "refusal_direction", "smoke": smoke, "models": models, "judge": cfg["judge"], "data": data,
            "filter": cfg["filter"], "selection": cfg["selection"], "generation": gen,
            # the tiny random models score MMLU at chance, below the registered floor: no floor unless a test sets one
            "mmlu": {**cfg["mmlu"], "golden_items": 6, "batch": 4, "floor_pct": 0.0},
            "aggregation": cfg["aggregation"], "inhouse": cfg["inhouse"], "ego": cfg["ego"],
            "smoke_settings": {**cfg["smoke"], "jbb": 4, "mmlu": 6}, "cost": cfg["cost"], "gpu": "L40S",
            "registered": cfg["registered"], "code": refusal.code_hashes(PKG, CHECK), "pins": {"lab": {}, "paper": {}},
            "container_sha256": "test"}
    return json.loads(json.dumps(plan))


def config() -> dict:
    import yaml

    return yaml.safe_load((CHECK / "config.yaml").read_text())


# -------------------------------------------------------------------------------------------- app.py, stubbed


def load_app():
    """app.py imported afresh under the modal stand-in (tests/modal_stub.py) with a Secret added and each function's
    decorator arguments recorded (stub.function_kwargs). Returns (stub, module); call unload_app() afterwards."""
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    import modal_stub

    unload_app()
    stub = modal_stub.Stub()
    stub.secrets = []

    class Secret:
        @staticmethod
        def from_name(name, **k):
            stub.secrets.append(name)
            return ("secret", name)

    stub.module.Secret = Secret
    stub.function_kwargs = {}
    base_app = stub.module.App

    class App(base_app):  # records each Modal function's decorator arguments
        def function(self, **k):
            deco = super().function(**k)

            def wrap(f):
                stub.function_kwargs[f.__name__] = k
                return deco(f)
            return wrap

    stub.module.App = App
    sys.modules["modal"] = stub.module
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="no trait directions")
        mod = load("c3_app", CHECK / "app.py")
    return stub, mod


def unload_app() -> None:
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    import modal_stub

    modal_stub.unload_apps()
    sys.modules.pop("c3_app", None)
