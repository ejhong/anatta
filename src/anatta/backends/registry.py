"""The model registry: registry.yaml loaded, validated and looked up (clean-lab step 3). Claude (Opus 5.5) wrote this
file, 3 Nov 2026.

  get(key)                 a Model by key or alias
  find(hf_id, revision)    the key pinned to that repo and commit, or None
  snapshot(key, mount)     the snapshot folder inside a container (<mount>/hf/hub/models--org--name/snapshots/<rev>)
  load()                   every entry, validated (RegistryError names the first problem)

Mounts: a model's `volume` is a lab.toml [modal.volumes] role; MOUNTS gives where the lab's Modal apps mount each.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PATH = Path(__file__).with_name("registry.yaml")
BACKENDS = ("hf", "vllm", "openrouter")
MOUNTS = {"hf_cache": "/cache", "calib_prh_k1": "/k1"}  # role -> mount point in the lab's Modal containers
HEX40 = re.compile(r"[0-9a-f]{40}")
_FIELDS = {"hf_id", "revision", "volume", "family", "n_layers", "n_layers_source", "chat_template", "backends", "vllm",
           "aliases", "sources", "flags", "openrouter_id", "provider", "config_from", "adapters", "fetched"}


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Model:
    key: str
    backends: tuple[str, ...]
    hf_id: str | None = None
    revision: str | None = None
    volume: str | None = None
    family: str | None = None
    n_layers: int | None = None
    chat_template: str | None = None
    vllm: dict = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    openrouter_id: str | None = None
    provider: dict | None = None
    config_from: str | None = None
    adapters: dict = field(default_factory=dict)

    @property
    def is_api(self) -> bool:
        return self.openrouter_id is not None

    def snapshot(self, mount: str | None = None) -> str:
        """The snapshot folder as a container sees it (mount: override the volume's usual mount point)."""
        if self.hf_id is None:
            raise RegistryError(f"{self.key} is an API model; it has no snapshot")
        root = mount or MOUNTS[self.volume]
        return f"{root}/hf/hub/models--{self.hf_id.replace('/', '--')}/snapshots/{self.revision}"


def _check(key: str, e: dict) -> Model:
    def bad(why: str):
        raise RegistryError(f"registry.yaml: {key}: {why}")

    if not isinstance(e, dict):
        bad("not a mapping")
    unknown = set(e) - _FIELDS
    if unknown:
        bad(f"unknown fields {sorted(unknown)}")
    backends = tuple(e.get("backends") or ())
    if not backends or any(b not in BACKENDS for b in backends):
        bad(f"backends must be a non-empty subset of {BACKENDS}, got {backends}")
    if "openrouter" in backends:
        if not e.get("openrouter_id") or "hf_id" in e:
            bad("an OpenRouter entry has openrouter_id and no hf_id")
        if e.get("provider") is not None and not isinstance(e["provider"], dict):
            bad("provider is a routing mapping or null")
    else:
        for f in ("hf_id", "revision", "volume", "family", "n_layers", "n_layers_source", "chat_template"):
            if e.get(f) in (None, ""):
                bad(f"missing {f}")
        if not HEX40.fullmatch(str(e["revision"])):
            bad(f"revision {e['revision']!r} is not a 40-hex commit")
        if e["volume"] not in MOUNTS:
            bad(f"volume {e['volume']!r} is not one of {sorted(MOUNTS)}")
        if not isinstance(e["n_layers"], int) or e["n_layers"] <= 0:
            bad("n_layers must be a positive int")
    if not e.get("sources"):
        bad("sources: name the files that pin it")
    return Model(key=key, backends=backends, hf_id=e.get("hf_id"), revision=e.get("revision"),
                 volume=e.get("volume"), family=e.get("family"), n_layers=e.get("n_layers"),
                 chat_template=e.get("chat_template"), vllm=dict(e.get("vllm") or {}),
                 aliases=tuple(e.get("aliases") or ()), sources=tuple(e.get("sources") or ()),
                 flags=tuple(e.get("flags") or ()), openrouter_id=e.get("openrouter_id"), provider=e.get("provider"),
                 config_from=e.get("config_from"), adapters=dict(e.get("adapters") or {}))


def load(path: str | Path | None = None) -> dict[str, Model]:
    raw = yaml.safe_load(Path(path or PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise RegistryError("registry.yaml: needs a top-level `models` mapping")
    models = {k: _check(k, e) for k, e in raw["models"].items()}
    names: dict[str, str] = {}
    for m in models.values():
        for name in (m.key, *m.aliases):
            if name in names:
                raise RegistryError(f"registry.yaml: name {name!r} is used by {names[name]} and {m.key}")
            names[name] = m.key
        if m.config_from and m.config_from not in models:
            raise RegistryError(f"registry.yaml: {m.key}: config_from {m.config_from!r} is not a key")
    pins: dict[tuple, str] = {}
    for m in models.values():
        pin = (m.hf_id, m.revision) if m.hf_id else (m.openrouter_id, None)
        if pin in pins:
            raise RegistryError(f"registry.yaml: {m.key} and {pins[pin]} pin the same model {pin}")
        pins[pin] = m.key
    return models


@lru_cache(maxsize=1)
def _default() -> dict[str, Model]:
    return load()


def _index() -> dict[str, str]:
    return {name: m.key for m in _default().values() for name in (m.key, *m.aliases)}


def get(name: str) -> Model:
    """A Model by key or alias."""
    key = _index().get(name)
    if key is None:
        raise KeyError(f"no model {name!r} in the registry ({PATH.name}); keys: {sorted(_default())}")
    return _default()[key]


def find(model_id: str, revision: str | None = None) -> str | None:
    """The key pinned to (hf_id, revision), or to an OpenRouter slug (revision None); None when not registered."""
    for m in _default().values():
        if (m.hf_id == model_id and m.revision == revision) or (revision is None and m.openrouter_id == model_id):
            return m.key
    return None


def snapshot(name: str, mount: str | None = None) -> str:
    return get(name).snapshot(mount)
