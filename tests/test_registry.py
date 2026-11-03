"""The model registry (anatta.backends.registry): it loads and validates, its lookups work, every lab caller's pinned
model id and revision is in it, and every `sources` line it cites really pins that model. Free and offline: the
callers are read as text (ast and yaml), never imported. Claude (Opus 5.5) wrote this file."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from anatta.backends import registry as R

REPO = Path(__file__).resolve().parents[1]
# The model tables the clean-lab inventory found duplicated (the step 3 brief), plus every calibration config and
# app that pins a model.
CALLERS = [
    "lab/explore/bench/steering.py", "lab/explore/act1/act1_core.py", "lab/explore/selfmap/modal_selfmap.py",
    "lab/explore/noself/modal_noself3.py", "lab/explore/noself/noself3.py", "lab/explore/p1/spec.yaml",
    "lab/explore/downloads/modal_selfmap_dl.py", "lab/explore/downloads/cfgdiff.py",
    "calibration/k6_battery/app.py", "calibration/k7_agentic/app_vllm.py", "calibration/k6b_ministral/app_vllm.py",
    "calibration/model_check/config.yaml", "calibration/koan_scorer/config.yaml",
    "calibration/persona_extractor/config.yaml", "calibration/refusal_direction/config.yaml",
    "calibration/prh_k1/config.yaml", "calibration/relearning/config.yaml",
]
# Pinned repos that are not language models the lab serves: datasets and K1's vision encoders.
NOT_MODELS = re.compile(r"^(minhuh/prh|cais/wmdp.*|allenai/ai2_arc|facebook/.*|openai/clip.*|google/vit.*|"
                        r"laion/.*|timm/.*|TIGER-Lab/.*|JailbreakBench/.*|openai/gsm8k|cais/mmlu|evalplus/.*)$")
HEX = re.compile(r"^[0-9a-f]{40}$")
SNAP = re.compile(r"models--([\w.-]+?)--([\w.-]+)/snapshots/([0-9a-f]{40})")
ID_KEYS, REV_KEYS = ("repo", "hf_id", "id"), ("rev", "revision")


def _render(node: ast.AST, names: dict[str, str]) -> str | None:
    """A str constant or an f-string, with {NAME} replaced by NAME's module-level string when known."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name):
                out.append(names.get(v.value.id, "{" + v.value.id + "}"))
            else:
                out.append("{?}")
        return "".join(out)
    return None


def pins_py(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    for node in tree.body:  # module-level string assignments, in order (tuples unpacked)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t, v = node.targets[0], node.value
            if isinstance(t, ast.Name) and (s := _render(v, names)) is not None:
                names[t.id] = s
            elif isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple) and len(t.elts) == len(v.elts):
                for tn, vv in zip(t.elts, v.elts):
                    if isinstance(tn, ast.Name) and (s := _render(vv, names)) is not None:
                        names[tn.id] = s
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        s = _render(node, names)
        if s:
            out |= {(f"{a}/{b}", h) for a, b, h in SNAP.findall(s)}
        if isinstance(node, ast.Dict):
            d = {k.value: _render(v, names) for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            mid = next((d[k] for k in ID_KEYS if d.get(k) and "/" in d[k]), None)
            rev = next((d[k] for k in REV_KEYS if d.get(k) and HEX.match(d[k])), None)
            if mid and rev:
                out.add((mid, rev))
    if HEX.match(names.get("REV", "")) and "/" in names.get("REPO", ""):
        out.add((names["REPO"], names["REV"]))
    return out


def pins_yaml(path: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()

    def walk(x):
        if isinstance(x, dict):
            mid = next((x[k] for k in ID_KEYS if isinstance(x.get(k), str) and "/" in x[k]), None)
            rev = next((x[k] for k in REV_KEYS if isinstance(x.get(k), str) and HEX.match(x[k])), None)
            if mid and rev:
                out.add((mid, rev))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(yaml.safe_load(path.read_text(encoding="utf-8")))
    for line in path.read_text(encoding="utf-8").splitlines():
        out |= {(f"{a}/{b}", h) for a, b, h in SNAP.findall(line)}
    return out


def caller_pins() -> dict[tuple[str, str], list[str]]:
    found: dict[tuple[str, str], list[str]] = {}
    for rel in CALLERS:
        p = REPO / rel
        for pin in (pins_yaml(p) if p.suffix == ".yaml" else pins_py(p)):
            if not NOT_MODELS.match(pin[0]):
                found.setdefault(pin, []).append(rel)
    return found


def registered(mid: str, rev: str, models: dict[str, R.Model]) -> bool:
    return any((m.hf_id, m.revision) == (mid, rev) or
               any((a.get("hf_id"), a.get("revision")) == (mid, rev) for a in m.adapters.values())
               for m in models.values())


def test_registry_loads_and_validates():
    models = R.load()
    assert R.get("ministral3_8b").hf_id == "mistralai/Ministral-3-8B-Instruct-2512-BF16"
    assert R.get("qwen") is R.get("qwen25_7b") and R.get("qwen7b_instruct").n_layers == 28
    assert R.find("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28") == "qwen25_7b"
    assert R.find("Qwen/Qwen2.5-7B-Instruct", "0" * 40) is None
    assert R.find("meta-llama/llama-4-maverick") == "maverick"
    assert R.snapshot("qwen") == ("/cache/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
                                  "a09a35458c702b33eeacc393d103063234e8bc28")
    assert R.get("qwen_base").snapshot().startswith("/k1/hf/hub/")
    assert all(m.vllm.get("text_only") for m in models.values() if m.family == "mistral3" and "vllm" in m.backends)
    with pytest.raises(KeyError):
        R.get("nope")


@pytest.mark.parametrize("bad, why", [
    ({"x": {"backends": ["hf"], "sources": ["a"]}}, "missing hf_id"),
    ({"x": {"hf_id": "a/b", "revision": "abc", "volume": "hf_cache", "family": "f", "n_layers": 2,
            "n_layers_source": "s", "chat_template": "t", "backends": ["hf"], "sources": ["a"]}}, "40-hex"),
    ({"x": {"openrouter_id": "a/b", "backends": ["openrouter"], "sources": ["a"], "colour": 1}}, "unknown fields"),
    ({"x": {"openrouter_id": "a/b", "backends": ["openrouter"], "sources": ["a"], "aliases": ["y"]},
      "y": {"openrouter_id": "c/d", "backends": ["openrouter"], "sources": ["a"]}}, "is used by"),
])
def test_registry_rejects(tmp_path, bad, why):
    f = tmp_path / "r.yaml"
    f.write_text(yaml.safe_dump({"models": bad}))
    with pytest.raises(R.RegistryError, match=why):
        R.load(f)


def test_every_caller_pin_is_registered():
    found = caller_pins()
    models = R.load()
    # the scan sees the tables the brief names (a weak scan would pass vacuously)
    for mid in ("Qwen/Qwen2.5-7B-Instruct", "mistralai/Ministral-3-8B-Instruct-2512-BF16", "google/gemma-2-9b-it",
                "meta-llama/Meta-Llama-3-8B-Instruct", "allenai/Olmo-3-7B-Instruct", "Qwen/Qwen2.5-3B"):
        assert any(p[0] == mid for p in found), mid
    missing = {p: files for p, files in found.items() if not registered(*p, models)}
    assert not missing, f"pinned by lab callers but not in registry.yaml: {missing}"


def test_no_revision_disagreements_in_callers():
    """One repo, several commits, is fine only where the registry holds each as its own key (OLMo's stages)."""
    found = caller_pins()
    by_id: dict[str, set[str]] = {}
    for mid, rev in found:
        by_id.setdefault(mid, set()).add(rev)
    models = R.load()
    for mid, revs in by_id.items():
        keys = {m.key for m in models.values() if m.hf_id == mid}
        assert len(revs) <= max(1, len(keys)), (mid, revs)


def test_sources_cite_lines_that_pin_the_model():
    for m in R.load().values():
        needles = [x for x in (m.revision, m.revision and m.revision[:7], m.openrouter_id, m.hf_id) if x]
        for src in m.sources:
            ref = src.split(" ")[0]
            path, _, line = ref.partition(":")
            text = (REPO / path).read_text(encoding="utf-8")
            if not line:
                assert any(n in text for n in needles), (m.key, src)
                continue
            lines = text.splitlines()
            i = int(line) - 1
            window = "\n".join(lines[max(0, i - 3): i + 4])
            assert any(n in window or n.replace("/", "--") in window for n in needles), (m.key, src, window)


def test_lab_default_model_is_registered():
    from anatta.config import load

    assert R.get(load().default_model).hf_id
