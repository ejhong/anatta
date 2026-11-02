"""Directions on the laptop: the registry (directions.json), loading a direction file with its hashes checked, layer
specs, and saving the controls the bench builds (random_matched, split_null). Claude (Opus 5.5) wrote this file, 1 Nov
2026, and extended it the same day (tensor hashes, the {vectors, best_layer, report} format). Private (lab/explore).

A direction is identified and pinned by its tensor hash (tensor_sha256): sha256 over the bytes of its tensors, layer
by layer, with their dtype and shape, not over its file. torch.save isn't byte-stable (every file carries a fresh
serialization id and its own archive name), so the same vectors saved twice get two file hashes; the file's sha256 is
kept as a secondary field. Whatever is pinned is checked here, before anything is sent or spent (a pinned file hash
still binds: the ledger pins files by their bytes; pin tensor_sha256 alone to accept a re-saved file). The vectors then
travel to the container with each batch, keyed by the tensor hash; controls are named by it and the records carry
both hashes. Vectors are sent as stored: anatta.dial's Intervention normalizes them in the container, as it does for
player.ego_cut.

Files are read with torch.load(weights_only=True), which unpickles tensors and plain containers only. A file that
needs a full unpickle is read only when its file sha256 is pinned and matches, so no unverified bytes are unpickled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
REGISTRY = BENCH / "directions.json"
RANDOM = "random_matched"
TENSOR_HASH = "bench-tensor-v1"  # the tensor hash's version tag, hashed first
DIRECTION_FIELDS = {"name", "vectors", "scores", "pos_mean", "neg_mean"}  # anatta.probe.Direction's (meta optional)


class DirectionError(ValueError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(vectors) -> str:
    """The identity of a set of tensors: sha256 over the tag TENSOR_HASH, then for each key in order (layers as ints,
    ascending) the line "key|dtype|shape" and the tensor's contiguous bytes. `vectors` is {layer: tensor or array}, or
    one tensor or array (keyed "single"). A strided view hashes as its contiguous copy; a dtype change changes it."""
    import torch

    if isinstance(vectors, dict):
        items = [(int(k), vectors[k]) for k in sorted(vectors, key=int)]
    else:
        items = [("single", vectors)]
    h = hashlib.sha256(TENSOR_HASH.encode() + b"\n")
    for k, v in items:
        t = torch.as_tensor(v).detach().cpu().contiguous()
        h.update(f"{k}|{str(t.dtype).removeprefix('torch.')}|{list(t.shape)}\n".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_registry(path=REGISTRY) -> dict:
    return json.loads(Path(path).read_text())


@dataclass
class Dir:
    """A direction ready to send: per-layer vectors (layer -> float32 array), or one vector for any layer."""

    name: str
    sha256: str  # the file's sha256 (secondary; None for a direction not yet saved)
    path: str | None
    vectors: dict
    single: np.ndarray | None = None
    best_layer: int | None = None
    registered: str | None = None
    made: dict | None = None  # how a control was built
    tensor_sha256: str | None = None  # the pin (primary)
    note: str | None = None  # e.g. the file hash differed from its pin while the tensors matched

    @property
    def key(self) -> str:
        """What the container and the records key the direction by: its tensor hash (the file hash for old payloads)."""
        return self.tensor_sha256 or self.sha256

    def layers_for(self, spec, n_layers: int | None = None) -> list[int]:
        """A layer spec as a sorted list: None or "all" (every layer the direction has; every model layer for a
        single vector), "best" (the layer with the best held-out score), an int, "a-b", {"from": a, "to": b}, or a
        list of these. Every layer must exist in the model and, for per-layer directions, in the file."""
        if spec is None or spec == "all":
            out = list(range(n_layers)) if self.single is not None else sorted(self.vectors)
            if self.single is not None and n_layers is None:
                raise DirectionError(f"{self.name}: 'all' layers of a single vector needs the model's layer count")
        elif spec == "best":
            if self.best_layer is None:
                raise DirectionError(f"{self.name} has no best layer (no scores)")
            out = [self.best_layer]
        else:
            out = sorted(set(_layers(spec)))
        if not out:
            raise DirectionError(f"{self.name}: layer spec {spec!r} selects no layers")
        if n_layers is not None and (out[0] < 0 or out[-1] >= n_layers):
            raise DirectionError(f"{self.name}: layers {out} are outside the model's 0-{n_layers - 1}")
        if self.single is None:
            missing = [x for x in out if x not in self.vectors]
            if missing:
                raise DirectionError(f"{self.name} has no vectors at layers {missing} (it has {sorted(self.vectors)})")
        return out

    def vector(self, layer: int) -> np.ndarray:
        return self.single if self.single is not None else self.vectors[layer]

    def payload(self, n_layers: int) -> dict:
        """What the container gets: {"name", "vectors": {layer: float32 array}}."""
        vecs = {x: self.single for x in range(n_layers)} if self.single is not None else self.vectors
        return {"name": self.name, "vectors": {int(x): np.asarray(v, dtype=np.float32) for x, v in vecs.items()}}

    def dim(self) -> int:
        v = self.single if self.single is not None else next(iter(self.vectors.values()))
        return int(np.asarray(v).size)

    def describe(self) -> dict:
        return {"name": self.name, "tensor_sha256": self.tensor_sha256, "sha256": self.sha256, "path": self.path,
                "registered": self.registered, "layers": None if self.single is not None else sorted(self.vectors),
                "best_layer": self.best_layer, "note": self.note, "made": self.made}


def _layers(spec) -> list[int]:
    if isinstance(spec, bool):
        raise DirectionError(f"bad layer spec {spec!r}")
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", spec)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise DirectionError(f"bad layer band {spec!r}")
            return list(range(a, b + 1))
        if spec.strip().isdigit():
            return [int(spec)]
    if isinstance(spec, dict) and set(spec) == {"from", "to"}:
        return list(range(int(spec["from"]), int(spec["to"]) + 1))
    if isinstance(spec, list):
        return [x for s in spec for x in _layers(s)]
    raise DirectionError(f"bad layer spec {spec!r}: all, best, an int, 'a-b', {{from, to}} or a list")


def torch_load(path, *, file_pinned: bool):
    """torch.load with weights_only=True; a full unpickle only for a file whose sha256 was pinned and matched."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 - UnpicklingError and friends
        if not file_pinned:
            raise DirectionError(f"{path}: needs a full unpickle (weights_only refused it: {type(e).__name__}); pin its "
                                 f"file sha256 to allow that") from None
        return torch.load(path, map_location="cpu", weights_only=False)


def _per_layer(raw, path, row0_layer) -> tuple[dict, object]:
    """({layer: tensor}, single tensor or None) from a {layer: tensor} dict, a [d] tensor or a [n, d] tensor."""
    import torch

    if isinstance(raw, dict):
        return {int(k): torch.as_tensor(v) for k, v in raw.items()}, None
    if torch.is_tensor(raw) and raw.dim() == 1:
        return {}, raw
    if torch.is_tensor(raw) and raw.dim() == 2:
        if row0_layer is None:
            raise DirectionError(f"{path}: a [n, d] tensor needs row0_layer (the layer of row 0)")
        return {row0_layer + i: raw[i] for i in range(raw.shape[0]) if row0_layer + i >= 0}, None
    raise DirectionError(f"{path}: vectors must be a {{layer: tensor}} dict, a [d] tensor or a [n, d] tensor")


def load_file(path, row0_layer: int | None = None, *, file_pinned: bool = False):
    """(per-layer vectors as float32 arrays, single vector or None, best layer or None, tensor_sha256) from:
      - an anatta Direction file (probe.Direction.save): best layer = the best held-out score;
      - a {vectors, best_layer, ...} dict (what_it_means' format, with a report): only `vectors` and `best_layer` are
        read, and nothing else in the file is touched;
      - a plain tensor: [d] is one vector for any layer, [n, d] needs row0_layer (the layer of row 0).
    The tensor hash is taken over the tensors as stored (their own dtype), keyed by layer."""
    import torch

    obj = torch_load(path, file_pinned=file_pinned)
    best = None
    if isinstance(obj, dict) and "vectors" in obj:
        if DIRECTION_FIELDS <= set(obj):
            from anatta.probe import Direction

            d = Direction(**obj)
            per, single = _per_layer(d.vectors, path, row0_layer)
            best = int(d.best_layer) if d.scores else None
        else:
            per, single = _per_layer(obj["vectors"], path, row0_layer)
            b = obj.get("best_layer")
            best = None if b is None else int(b)
    elif torch.is_tensor(obj):
        per, single = _per_layer(obj, path, row0_layer)
    else:
        raise DirectionError(f"{path}: not an anatta Direction file, a {{vectors, best_layer}} dict or a [d] / [n, d] "
                             f"tensor")
    tsha = tensor_sha256(single if single is not None else per)
    vecs = {x: v.detach().float().cpu().numpy().ravel() for x, v in per.items()}
    one = None if single is None else single.detach().float().cpu().numpy().ravel()
    return vecs, one, best, tsha


def resolve(ref, model_key: str, registry: dict | None = None) -> Dir:
    """A registry name ("ego") or {"path", "tensor_sha256"?, "sha256"?, "name"?, "row0_layer"?}, loaded and checked.
    Every pin given must match: the tensor hash (the identity), and the file hash when one is pinned too (the ledger
    pins files by their bytes). To accept a re-saved file whose tensors are unchanged, pin its tensor_sha256 alone.
    A registry entry needs at least one pin; an ad-hoc file's hashes are recorded whether pinned or not."""
    registry = registry or load_registry()
    if isinstance(ref, str):
        entries = registry["directions"].get(model_key, {})
        if ref not in entries:
            raise DirectionError(f"no direction {ref!r} for {model_key} in {REGISTRY.name}; known: {sorted(entries)}")
        e = entries[ref]
        path = e.get("path") or f"{registry['root']}/{registry['models'][model_key]['folder']}/{ref}.pt"
        want_t, want_f, name, registered, row0 = (e.get("tensor_sha256"), e.get("sha256"), ref, e.get("registered"),
                                                  e.get("row0_layer"))
        if not (want_t or want_f):
            raise DirectionError(f"registry entry {ref!r} pins no hash")
    elif isinstance(ref, dict) and "path" in ref:
        path, want_t, want_f = ref["path"], ref.get("tensor_sha256"), ref.get("sha256")
        name, registered, row0 = ref.get("name") or Path(ref["path"]).stem, None, ref.get("row0_layer")
    else:
        raise DirectionError(f"direction {ref!r}: a registry name or {{path, tensor_sha256 and/or sha256}}")
    if not Path(path).is_file():
        raise DirectionError(f"{path} is missing")
    got_f = sha256_file(path)
    if want_f and got_f != want_f:
        hint = "; if it was re-saved with the same tensors, pin its tensor_sha256 alone" if want_t else ""
        raise DirectionError(f"{path}: sha256 {got_f} is not the expected {want_f}; refusing to steer with it{hint}")
    vectors, single, best, got_t = load_file(path, row0, file_pinned=bool(want_f))
    if want_t and got_t != want_t:
        raise DirectionError(f"{path}: tensor sha256 {got_t} is not the expected {want_t}; refusing to steer with it")
    note = None if (want_t or want_f) else "unpinned: its hashes are as found"
    return Dir(name=name, sha256=got_f, path=str(path), vectors=vectors, single=single, best_layer=best,
               registered=registered, tensor_sha256=got_t, note=note)


def save_control(out_dir, target: Dir, seed: int, vectors: dict, made: dict, family: str = RANDOM) -> Dir:
    """Save a control (random_matched or split_null) as an anatta Direction file (scores 0: it is a control, not a
    trait) beside a JSON of how it was made. The file name carries the family, the target, the seed and the vectors'
    own tensor hash, so the same vectors always get the same name; if that file already holds them it is kept as it is
    (nothing is overwritten)."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{family}({target.name})"
    vecs = {int(x): np.asarray(v, dtype=np.float32) for x, v in vectors.items()}
    tens = {x: torch.from_numpy(v.copy()) for x, v in vecs.items()}
    tsha = tensor_sha256(tens)
    stem = f"{family}-{target.name}-{target.key[:8]}-s{seed}-{tsha[:8]}"
    path = out_dir / f"{stem}.pt"
    if not (path.exists() and load_file(path)[3] == tsha):
        obj = {"name": name, "vectors": tens, "scores": {x: 0.0 for x in vecs}, "pos_mean": {x: 0.0 for x in vecs},
               "neg_mean": {x: 0.0 for x in vecs}, "meta": {"family": family, "made": made}}
        tmp = out_dir / f".{family}-{os.getpid()}.pt.tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)
    sha = sha256_file(path)
    meta_path = out_dir / f"{stem}.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({"name": name, "tensor_sha256": tsha, "sha256": sha, "file": path.name,
                                         "made": made}, indent=1) + "\n")
    return Dir(name=name, sha256=sha, path=str(path), vectors=vecs, best_layer=target.best_layer, made=made,
               tensor_sha256=tsha)
