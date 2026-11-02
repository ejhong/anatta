"""Tests of Stage 2's pieces: the anchor block, the cross-model maps (crossmap.py) and the map-free RSA (stage2.py),
on synthetic activations with planted structure. Claude (Opus 5.5) wrote this file, 2 Nov 2026. Private
(lab/explore).
"""

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analysis as A  # noqa: E402
import crossmap as C  # noqa: E402
import grid  # noqa: E402
import stage2 as S  # noqa: E402
import templates as T  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return [r for r in grid.build() if r["split"] == "dev"] + grid.anchors() + \
        [r for r in grid.build_anchors() if r["split"] == "dev"]


def test_anchor_block():
    rs = grid.build_anchors()
    assert {r["split"] for r in rs} == {"dev", "heldout"}
    for kind, (items, tpls) in T.ANCHOR_KINDS.items():
        k = [r for r in rs if r["aspect"] == kind]
        assert len(k) == len(items) * len(tpls)
        assert len({r["template"] for r in k if r["split"] == "dev"}) == grid.N_DEV_ANCHORS
        for r in k:
            assert not any(m.lower() in r["text"].lower().split() for m in T.MONTHS + T.WEEKDAYS)


def test_anchor_files_match_manifest():
    import hashlib
    import json

    man = json.loads((HERE / "manifest.json").read_text())
    cur = man["history"][man["current"] - 1]
    b = (HERE / "prompts" / "anchors_dev.jsonl").read_bytes()
    assert hashlib.sha256(b).hexdigest() == cur["files"]["anchors_dev.jsonl"]["sha256"]
    assert cur["files"]["dev.jsonl"]["sha256"] == man["history"][1]["files"]["dev.jsonl"]["sha256"]


def planted(rows, d, rot, seed, shared_item=True):
    """Template + item/referent vectors in a shared latent space, sent to a model space by `rot`, plus noise."""
    rng = np.random.default_rng(seed)
    k = 24
    tv, iv = {}, {}
    Z = np.zeros((len(rows), k))
    for i, r in enumerate(rows):
        key = (r["family"], r["aspect"], r["template"])
        tv.setdefault(key, np.random.default_rng(zlib.crc32(repr(key).encode())).normal(size=k))
        z = tv[key].copy()
        it = r.get("item") or r.get("referent")
        if it:
            iv.setdefault(it, np.random.default_rng(zlib.crc32(("item:" + it).encode())).normal(size=k))
            z += 1.5 * iv[it]
        if r["family"] in ("months", "weekdays"):
            items = A._order(r["family"])
            a = 2 * np.pi * items.index(r["item"]) / len(items)
            z[:2] += 4 * np.array([np.cos(a), np.sin(a)])
        Z[i] = z
    return (Z @ rot + rng.normal(scale=0.3, size=(len(rows), d))).astype(np.float32)


def test_procrustes_recovers_unseen_months_when_structure_is_shared(rows):
    """Months never in the fit, but their circle lives in a latent space that the other rows span: a
    PCA + Procrustes map from A to B must carry them (top-1 well above 1/12)."""
    R = A.Rows(rows)
    groups = C.groups_of(rows)
    rng = np.random.default_rng(0)
    ra, rb = rng.normal(size=(24, 48)), rng.normal(size=(24, 56))
    XA = C.prep(planted(rows, 48, ra, 1), R.U, "z")
    XB = C.prep(planted(rows, 56, rb, 2), R.U, "z")
    r = C.run_variant(XA, XB, R, groups, "procrustes", "noself", "months", False, k=24)
    assert r["top1"] > 0.5
    r = C.run_variant(XA, XB, R, groups, "ridge", "all", "months", False)
    assert r["top1"] > 0.5


def test_unrelated_models_score_chance(rows):
    R = A.Rows(rows)
    groups = C.groups_of(rows)
    rng = np.random.default_rng(0)
    XA = C.prep(rng.normal(size=(len(rows), 40)), R.U, "z")
    XB = C.prep(rng.normal(size=(len(rows), 40)), R.U, "z")
    r = C.run_variant(XA, XB, R, groups, "procrustes", "noself", "months", False, k=32)
    assert r["top1"] < 0.25


def test_ridge_cv_matches_direct():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(60, 10)), rng.normal(size=(60, 4))
    g = np.repeat(np.arange(12), 5)
    a, err = C._ridge_cv(X, Y, g, (0.1, 1.0))
    # direct recomputation for one fold assignment
    rng2 = np.random.default_rng(0)
    ug = np.unique(g)
    fold_of = dict(zip(rng2.permutation(ug), np.arange(len(ug)) % 5))
    f = np.array([fold_of[x] for x in g])
    e = []
    for k in range(5):
        m = A.Ridge(X[f != k], Y[f != k], 1.0)
        e.append(((m.predict(X[f == k]) - Y[f == k]) ** 2).sum(1).mean())
    assert np.isclose(err[1.0], np.mean(e))


def test_prompt_rsa_identity_and_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 20))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cl = {"self": np.arange(50), "person": np.arange(50, 100)}
    out = S.prompt_rsa(X, X, cl, n_boot=20)
    assert out["self"]["rsa"] > 0.99 and abs(out["self-person"]["diff"]) < 1e-6
    Y = X.copy()
    Y[50:] = rng.normal(size=(50, 20))
    out = S.prompt_rsa(X, Y, cl, n_boot=50)
    assert out["self-person"]["ci95"][0] > 0.5
