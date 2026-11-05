"""The organism registry: registry.yaml loaded, validated and looked up, and each organism's files re-hashed before use
(clean-lab step 6). Claude (Opus 5.5) wrote this file, 4 Nov 2026. Pure Python: no torch or modal at import, so it runs
on the laptop and inside any Modal container that carries the package.

  load_registry()            every entry, validated (OrganismError names the first problem)
  get(key)                   one entry, as a dict (a copy)
  keys(kind=None, base=None) the keys, optionally of one kind and/or one base model
  local_path(key, root)      the organism's folder under `root` (the registry's paths are as the containers mount
                             anatta-runs, at /runs; root re-bases them, e.g. a temp folder in tests)
  verify(key, root=/runs)    re-hash every listed file (bytes, then streaming sha256); raises VerifyError on any
                             missing file or mismatch; returns the facts it read
  verify_dir(folder, files)  the same check on a folder and a {name: {bytes, sha256}} mapping (usable in a container
                             that has only the package, e.g. passed get(key)["files"])
  hash_file(path)            (bytes, sha256), streaming
  dir_facts(folder)          {file name: {bytes, sha256}} for every regular file in a folder (the facts stage)

The schema (every entry):
  kind      lora (an adapter folder) or baked (full weights)
  base      a key of the model registry (src/anatta/backends/registry.yaml)
  path      the folder inside the lab's Modal containers (anatta-runs at /runs)
  files     {file name: {bytes: int, sha256: 64-hex}}, at least the weights
  source    (fetched organisms) {hf_id, revision: 40-hex}; or
  recipe    (organisms the lab trained) a mapping that says how; trained organisms also carry
            seed (int), twin (another organism key, or null), recipe_commit (40-hex), card or ledger (where the run is
            written up), calibrated (bool: has this organism been shown to read as expected on a calibrated
            instrument).
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath

PATH = Path(__file__).with_name("registry.yaml")
MOUNT = PurePosixPath("/runs")  # where the lab's Modal containers mount anatta-runs (anatta.modal_app.RUNS)
KINDS = ("lora", "baked")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
CHUNK = 1 << 22
TRAINED = ("seed", "twin", "recipe_commit", "calibrated")
_FIELDS = {"kind", "base", "path", "files", "source", "recipe", "seed", "twin", "recipe_commit", "card", "ledger",
           "calibrated", "trained_on", "paper", "licence", "arm", "round", "parent", "lora", "notes"}


class OrganismError(ValueError):
    """The registry itself is malformed."""


class VerifyError(RuntimeError):
    """An organism's files on disk are missing or are not the registered bytes."""


def _check(key: str, e, model_keys: set[str] | None) -> dict:
    def bad(why: str):
        raise OrganismError(f"organisms/registry.yaml: {key}: {why}")

    if not isinstance(e, dict):
        bad("not a mapping")
    unknown = set(e) - _FIELDS
    if unknown:
        bad(f"unknown fields {sorted(unknown)}")
    for f in ("kind", "base", "path", "files"):
        if e.get(f) in (None, "", {}):
            bad(f"missing {f}")
    if e["kind"] not in KINDS:
        bad(f"kind {e['kind']!r} is not one of {KINDS}")
    if model_keys is not None and e["base"] not in model_keys:
        bad(f"base {e['base']!r} is not a model registry key")
    p = PurePosixPath(str(e["path"]))
    if not p.is_absolute() or ".." in p.parts:
        bad(f"path {e['path']!r} must be absolute, without '..'")
    if not isinstance(e["files"], dict):
        bad("files must be a {name: {bytes, sha256}} mapping")
    for fn, fx in e["files"].items():
        if "/" in fn or fn in (".", ".."):
            bad(f"file name {fn!r} must be a plain name in the folder")
        if not isinstance(fx, dict) or set(fx) != {"bytes", "sha256"}:
            bad(f"files.{fn} needs exactly bytes and sha256")
        if not isinstance(fx["bytes"], int) or isinstance(fx["bytes"], bool) or fx["bytes"] < 0:
            bad(f"files.{fn}.bytes must be a non-negative int")
        if not HEX64.fullmatch(str(fx["sha256"])):
            bad(f"files.{fn}.sha256 is not 64 lowercase hex")
    if ("source" in e) == ("recipe" in e):
        bad("give exactly one of source (fetched) or recipe (trained here)")
    if "source" in e:
        s = e["source"]
        if not isinstance(s, dict) or not s.get("hf_id") or not HEX40.fullmatch(str(s.get("revision", ""))):
            bad("source needs hf_id and a 40-hex revision")
    else:
        if not isinstance(e["recipe"], dict) or not e["recipe"]:
            bad("recipe must be a non-empty mapping")
        for f in TRAINED:
            if f not in e:
                bad(f"a trained organism needs {f}")
        if not isinstance(e["seed"], int) or isinstance(e["seed"], bool):
            bad("seed must be an int")
        if e["twin"] is not None and not isinstance(e["twin"], str):
            bad("twin is another organism's key or null")
        if not HEX40.fullmatch(str(e["recipe_commit"])):
            bad(f"recipe_commit {e['recipe_commit']!r} is not a 40-hex commit")
        if not (e.get("card") or e.get("ledger")):
            bad("a trained organism names its card or its ledger entry")
        if not isinstance(e["calibrated"], bool):
            bad("calibrated must be true or false")
    return e


def _model_keys() -> set[str]:
    from anatta.backends.registry import load

    return set(load())


def load_registry(path: str | Path | None = None, model_keys: set[str] | None = None) -> dict[str, dict]:
    """Every entry, validated. model_keys: the model registry's keys (read from it by default)."""
    import yaml

    raw = yaml.safe_load(Path(path or PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("organisms"), dict):
        raise OrganismError("organisms/registry.yaml: needs a top-level `organisms` mapping")
    mk = _model_keys() if model_keys is None else model_keys
    orgs = {k: _check(k, e, mk) for k, e in raw["organisms"].items()}
    paths: dict[str, str] = {}
    for k, e in orgs.items():
        if e["path"] in paths:
            raise OrganismError(f"organisms/registry.yaml: {k} and {paths[e['path']]} share the path {e['path']}")
        paths[e["path"]] = k
        for ref in ("twin", "parent"):
            other = e.get(ref)
            if other is not None and other not in orgs:
                raise OrganismError(f"organisms/registry.yaml: {k}: {ref} {other!r} is not an organism key")
            if other == k:
                raise OrganismError(f"organisms/registry.yaml: {k} is its own {ref}")
        twin = e.get("twin")
        if twin is not None and orgs[twin].get("twin") != k:
            raise OrganismError(f"organisms/registry.yaml: {k}'s twin {twin} does not name {k} back")
    return orgs


@lru_cache(maxsize=1)
def _default() -> dict[str, dict]:
    return load_registry()


def get(key: str) -> dict:
    """One organism's entry (a deep copy, so callers can't change the cached registry)."""
    orgs = _default()
    if key not in orgs:
        raise KeyError(f"no organism {key!r} in {PATH.name}; keys: {sorted(orgs)}")
    return copy.deepcopy(orgs[key])


def keys(kind: str | None = None, base: str | None = None) -> list[str]:
    return sorted(k for k, e in _default().items()
                  if (kind is None or e["kind"] == kind) and (base is None or e["base"] == base))


def hash_file(path: str | os.PathLike) -> tuple[int, str]:
    """(bytes, sha256 hex) of a file, read in 4 MiB chunks."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
            n += len(b)
    return n, h.hexdigest()


def dir_facts(folder: str | os.PathLike) -> dict[str, dict]:
    """{name: {bytes, sha256}} for every regular file directly in `folder`, sorted by name."""
    out = {}
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and not os.path.islink(p):
            n, s = hash_file(p)
            out[name] = {"bytes": n, "sha256": s}
    return out


def verify_dir(folder: str | os.PathLike, files: dict, label: str = "") -> dict[str, dict]:
    """Check that each listed file is in `folder` with the listed bytes and sha256; the size is checked first, so a
    wrong-length file fails without being read. Raises VerifyError listing every problem; returns the facts read."""
    who = f"{label}: " if label else ""
    if not files:
        raise VerifyError(f"{who}no files listed to verify")
    problems, facts = [], {}
    for fn, want in sorted(files.items()):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            problems.append(f"{fn}: missing at {p}")
            continue
        size = os.path.getsize(p)
        if size != want["bytes"]:
            problems.append(f"{fn}: {size} bytes, registered {want['bytes']}")
            facts[fn] = {"bytes": size, "sha256": None}
            continue
        n, s = hash_file(p)
        facts[fn] = {"bytes": n, "sha256": s}
        if s != want["sha256"]:
            problems.append(f"{fn}: sha256 {s[:12]}..., registered {want['sha256'][:12]}...")
    if problems:
        raise VerifyError(f"{who}{folder} is not the registered organism:\n  " + "\n  ".join(problems))
    return facts


def local_path(key: str, root: str | os.PathLike = MOUNT) -> Path:
    """The organism's folder with the registry's /runs prefix replaced by `root`."""
    p = PurePosixPath(get(key)["path"])
    try:
        rel = p.relative_to(MOUNT)
    except ValueError:
        raise OrganismError(f"{key}: path {p} is not under {MOUNT}; root cannot re-base it") from None
    return Path(root) / rel


def verify(key: str, root: str | os.PathLike = Path(str(MOUNT))) -> dict[str, dict]:
    """Re-hash every file the registry lists for `key` under `root` (anatta-runs' mount by default). Raises VerifyError
    on any missing file or mismatch; returns {file: {bytes, sha256}} as read."""
    e = get(key)
    return verify_dir(local_path(key, root), e["files"], label=key)
