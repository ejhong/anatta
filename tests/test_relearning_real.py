"""Calibration check 7's golden test on the real data: question 0's prompt and ids, and the four answer ids. The data
stay on Modal and never come to the laptop, so these tests skip here unless the data are present:
  - question 0: $C7_GOLDEN names a copy of a run's data/golden.json (written by the check_ids stage on the volume);
  - the tokenizer: $C7_TOKENIZER names a copy of a run's tokenizer/ folder, the checkpoint's tokenizer as
    transformers 4.41.2 converts it, saved by the convert_tokenizer stage. It is loaded here as the lab's stages load
    it (relearn.load_saved). The checkpoint's own tokenizer.model is not used: transformers 5.21 builds a tokenizer
    from it whose answer letters lack the word-start marker (NOTES, 31 Oct 15:22 UTC).
The same checks run in the containers on every run: convert_tokenizer checks the answer ids under Łucki's stack before
saving; prepare checks the saved tokenizer's files, its backend as loaded and the answer ids; check_ids compares
question 0's ids (and every other prompt's), the answer ids and the backend under both stacks; score checks the ids it
feeds the model against them."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

CHECK = Path(__file__).resolve().parent.parent / "calibration" / "relearning"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


relearn = sys.modules.get("relearn") or load("relearn", CHECK / "relearn.py")
CFG = yaml.safe_load((CHECK / "config.yaml").read_text())


def expected_prompt(q: dict) -> str:
    """The prompt written out by hand, apart from the scorer's template."""
    lines = ["The following are multiple choice questions (with answers) about biology.", "", q["question"].strip()]
    lines += [f"{letter}. {choice}" for letter, choice in zip("ABCD", q["choices"])]
    return "\n".join(lines) + "\nAnswer:"


@pytest.fixture
def golden() -> dict:
    path = os.environ.get("C7_GOLDEN")
    if not path or not Path(path).exists():
        pytest.skip("the real data stay on Modal: set C7_GOLDEN to a copy of a run's data/golden.json to run this")
    return json.loads(Path(path).read_text())


@pytest.fixture
def saved_dir() -> Path:
    path = os.environ.get("C7_TOKENIZER")
    if not path or not (Path(path) / "tokenizer.json").exists():
        pytest.skip("the saved tokenizer stays on the volume: set C7_TOKENIZER to a copy of a run's tokenizer/ folder "
                    "(written by the convert_tokenizer stage) to run this")
    return Path(path)


@pytest.fixture
def real_tok(saved_dir):
    return relearn.load_saved(saved_dir)


def test_question0_prompt_and_ids(golden):
    q = golden["question0"]
    prompts, answers = relearn.wmdp_prompts([q], CFG["scorer"]["context"])
    assert prompts[0] == expected_prompt(q) == golden["prompt"] and answers[0] == int(q["answer"])
    assert golden["ids_lab"] == golden["ids_lucki"] and golden["ids_lab"][0] == CFG["model"]["bos_id"]
    assert golden["answer_ids_lab"] == golden["answer_ids_lucki"] == CFG["scorer"]["answer_ids"] == [330, 365, 334, 384]


def test_the_saved_tokenizer(saved_dir, real_tok):
    assert "tokenizer.model" not in {p.name for p in saved_dir.iterdir()}
    assert relearn.answer_ids_list(real_tok) == [330, 365, 334, 384]
    assert [real_tok.convert_ids_to_tokens(i) for i in [330, 365, 334, 384]] == ["▁A", "▁B", "▁C", "▁D"]
    assert (real_tok.bos_token_id, real_tok.eos_token_id, real_tok.truncation_side) == (1, 2, "left")
    assert real_tok.pad_token_id == 2  # the EOS, as anatta.models.load sets it
    assert hashlib.sha256(real_tok.chat_template.encode()).hexdigest() == CFG["model"]["chat_template_sha256"]
    saved = json.loads((saved_dir / "tokenizer.json").read_text(encoding="utf-8"))
    loaded = json.loads(real_tok.backend_tokenizer.to_str())
    assert all(loaded[k] == saved[k] for k in relearn.MARKER)  # the word-start marker survives the lab's load


def test_question0_ids_on_the_saved_tokenizer(golden, real_tok):
    assert relearn.prompt_ids(real_tok, [golden["prompt"]], CFG["scorer"]["max_length"])[0] == golden["ids_lab"]
    assert relearn.prompt_tensor(real_tok, golden["prompt"], CFG["scorer"]["max_length"])[0].tolist() == golden["ids_lab"]
