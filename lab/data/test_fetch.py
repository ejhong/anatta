#!/usr/bin/env python3
"""CPU-only checks for fetch.py and manifest.json; no network.

    python -I lab/data/test_fetch.py

Covers: the manifest is well-formed and internally consistent; sha256
verification accepts correct bytes and rejects corruption; no manifest path
(and no adversarial path) can escape its raw/ directory, even through a
planted symlink; verified files are skipped and bad downloads are discarded;
and nothing under a raw/ directory is importable.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_fetch_module():
    """Load fetch.py by path: under `python -I` the script directory is not on sys.path."""
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location("anatta_lab_fetch", HERE / "fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert sys.path == before, "fetch.py must not modify sys.path"
    return mod


fetch = load_fetch_module()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(HERE / "manifest.json", encoding="utf-8") as fh:
            cls.m = json.load(fh)

    def test_manifest_validates(self):
        fetch.validate_manifest(self.m)

    def test_required_fields_present(self):
        for name, d in self.m["datasets"].items():
            for key in ("name", "purpose", "source", "license", "flags", "fetch_date",
                        "total_bytes", "files", "fetched"):
                self.assertIn(key, d, f"{name} lacks {key}")
            self.assertIn("url", d["source"])
            self.assertIn("revision", d["source"])

    def test_every_path_stays_inside_its_raw_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, d in self.m["datasets"].items():
                raw = root / name / "raw"
                raw.mkdir(parents=True)
                for f in d["files"]:
                    target = fetch.resolve_target(raw, f["path"])
                    self.assertIn(raw.resolve(), target.parents, f"{name}: {f['path']}")

    def test_hashes_and_sizes_are_well_formed_and_totals_add_up(self):
        for name, d in self.m["datasets"].items():
            total = 0
            for f in d["files"]:
                self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$")
                self.assertIsInstance(f["bytes"], int)
                self.assertGreaterEqual(f["bytes"], 0)
                self.assertTrue(f["url"].startswith("https://"), f["url"])
                total += f["bytes"]
            self.assertEqual(total, d["total_bytes"], name)

    def test_paths_unique_per_dataset(self):
        for name, d in self.m["datasets"].items():
            paths = [f["path"] for f in d["files"]]
            self.assertEqual(len(paths), len(set(paths)), name)

    def test_validator_rejects_broken_manifests(self):
        good = json.loads(json.dumps(self.m))
        fetched = next(n for n, d in good["datasets"].items() if d["fetched"])
        entry = good["datasets"][fetched]

        def broken(mutate):
            m = json.loads(json.dumps(good))
            mutate(m["datasets"][fetched])
            with self.assertRaises(fetch.FetchError):
                fetch.validate_manifest(m)

        broken(lambda d: d["files"][0].update(sha256="abc"))
        broken(lambda d: d["files"][0].update(path="../escape.txt"))
        broken(lambda d: d["files"][0].update(url="http://insecure.example/x"))
        broken(lambda d: d.update(total_bytes=d["total_bytes"] + 1))
        broken(lambda d: d["files"].append(dict(d["files"][0])))  # duplicate path
        broken(lambda d: d.update(fetch_date="yesterday"))
        broken(lambda d: d.update(fetched=False))  # fetched=False with files listed
        self.assertTrue(entry["files"])  # the mutations above started from a real entry


class PathContainmentTests(unittest.TestCase):
    BAD = ["", ".", "..", "../x", "/etc/passwd", "a/../../x", "a/../x", "./a", "a/./b", "a//b",
           "a\\b", "~/x", "C:/x", "raw/../../x", "a/", "a\x00b", None, 3]
    GOOD = ["a", "a/b.txt", "deep/er/path/file.parquet", "name with spaces.txt", "dir/.hidden",
            "x/y/z/evals.zip"]

    def test_rejects_escaping_and_malformed_paths(self):
        for p in self.BAD:
            with self.assertRaises(fetch.FetchError, msg=repr(p)):
                fetch.safe_relpath(p)

    def test_accepts_plain_relative_paths(self):
        for p in self.GOOD:
            self.assertEqual(fetch.safe_relpath(p), p)

    def test_resolve_target_stays_inside(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            for p in self.GOOD:
                t = fetch.resolve_target(raw, p)
                self.assertIn(raw.resolve(), t.parents)
            for p in self.BAD:
                with self.assertRaises(fetch.FetchError, msg=repr(p)):
                    fetch.resolve_target(raw, p)

    def test_symlink_inside_raw_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            raw = Path(tmp) / "raw"
            raw.mkdir()
            os.symlink(outside, raw / "link")
            with self.assertRaises(fetch.FetchError):
                fetch.resolve_target(raw, "link/file.txt")


class HashTests(unittest.TestCase):
    def test_sha256_known_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "abc.bin"
            p.write_bytes(b"abc")
            self.assertEqual(fetch.sha256_of(p),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.bin"
            p.write_bytes(b"hello world")
            digest = fetch.sha256_of(p)
            p.write_bytes(b"hello worle")
            self.assertNotEqual(fetch.sha256_of(p), digest)


class FetchFileTests(unittest.TestCase):
    """Drive fetch_file with a fake downloader; no network."""

    def _entry(self, content: bytes, path="sub/file.bin"):
        return {"path": path, "url": "https://example.invalid/" + path,
                "bytes": len(content), "sha256": _sha(content)}

    def test_verified_file_is_skipped_without_download(self):
        content = b"good bytes"
        entry = self._entry(content)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            (raw / "sub").mkdir(parents=True)
            (raw / entry["path"]).write_bytes(content)
            calls = []

            def downloader(url, dest, ctx):
                calls.append(url)
                return "0" * 64

            self.assertEqual(fetch.fetch_file(entry, raw, downloader=downloader), "verified")
            self.assertEqual(calls, [])

    def test_corrupt_file_is_redownloaded(self):
        content = b"good bytes"
        entry = self._entry(content)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            (raw / "sub").mkdir(parents=True)
            (raw / entry["path"]).write_bytes(b"corrupt")

            def downloader(url, dest, ctx):
                dest.write_bytes(content)
                return _sha(content)

            self.assertEqual(fetch.fetch_file(entry, raw, downloader=downloader), "downloaded")
            self.assertEqual((raw / entry["path"]).read_bytes(), content)

    def test_bad_download_is_discarded(self):
        entry = self._entry(b"expected bytes")
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()

            def downloader(url, dest, ctx):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"tampered")
                return _sha(b"tampered")

            with self.assertRaises(fetch.FetchError):
                fetch.fetch_file(entry, raw, downloader=downloader)
            self.assertFalse((raw / entry["path"]).exists())

    def test_check_only_reports_without_downloading(self):
        entry = self._entry(b"x")
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()

            def downloader(url, dest, ctx):
                raise AssertionError("must not download in check mode")

            self.assertEqual(fetch.fetch_file(entry, raw, check_only=True, downloader=downloader), "missing")
            (raw / "sub").mkdir()
            (raw / entry["path"]).write_bytes(b"y")
            self.assertEqual(fetch.fetch_file(entry, raw, check_only=True, downloader=downloader), "corrupt")

    def test_download_refuses_plain_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(fetch.FetchError):
                fetch.download("http://example.invalid/x", Path(tmp) / "x")


class IsolationTests(unittest.TestCase):
    def test_no_raw_directory_on_sys_path(self):
        for p in sys.path:
            parts = Path(p).resolve().parts if p else ()
            self.assertNotIn("raw", parts, f"a raw/ directory is importable from: {p}")
            self.assertNotEqual(Path(p).resolve() if p else None, HERE,
                                f"{HERE} is on sys.path; run this with `python -I`")

    def test_planted_module_in_raw_is_not_importable(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "planted" / "raw"
            raw.mkdir(parents=True)
            (raw / "anatta_planted_module_xyz.py").write_text("raise SystemExit('planted module ran')\n")
            self.assertIsNone(importlib.util.find_spec("anatta_planted_module_xyz"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
