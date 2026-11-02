"""fetch.py with a stand-in hub (nothing downloaded): the repositories in order with the switch, the allow patterns,
each file checked against the listing's size and LFS sha256 or git blob sha, the approved sizes checked before any
download, the download made at the resolved commit, the dataset's rows counted, and the hub's listing read from
huggingface_hub's fields. The parquet facts need pyarrow and PIL, which the lab's environment lacks: that test skips
there and runs in the image's."""

import io
import types

import fetch
import k1 as K
import pytest
import yaml
from k1_testing import CHECK

CFG = yaml.safe_load((CHECK / "config.yaml").read_text())
REV = "ab" * 20


def fake_hub(tmp_path, files: dict, rev: str = REV, served: dict | None = None):
    """(info, download, calls) for one repository holding `files` ({name: bytes}); the large kinds are LFS files, the
    rest plain git blobs. download writes the named files (or `served`'s bytes for them) into a snapshot folder."""
    lfs = (".safetensors", ".bin", ".parquet")
    listing = {n: {"size": len(b), "lfs_sha256": K.sha256_bytes(b) if n.endswith(lfs) else None,
                   "blob_id": K.git_blob_sha(b)} for n, b in files.items()}
    calls = []

    def info(repo, kind, ref):
        calls.append(("info", repo, kind, ref))
        return rev, listing

    def download(repo, kind, revision, names):
        calls.append(("download", repo, revision, tuple(names)))
        snap = K.snapshot_dir(repo, revision, kind, hf_home=str(tmp_path / "hf"))
        for n in names:
            (snap / n).parent.mkdir(parents=True, exist_ok=True)
            (snap / n).write_bytes((served or {}).get(n, files[n]))
        return snap

    return info, download, calls


QWEN = {"config.json": b'{"hidden_size": 8}', "generation_config.json": b"{}", "merges.txt": b"#version\n",
        "model.safetensors": b"S" * 300, "tokenizer.json": b'{"model": {}}', "tokenizer_config.json": b"{}",
        "vocab.json": b"{}", "LICENSE": b"licence text", "README.md": b"# card", ".gitattributes": b"*.bin lfs"}


def spec(allow, expect=None, key="m", repo="org/model", kind="model"):
    return {"key": key, "repo": repo, "kind": kind, "ref": "main", "allow": allow, "expect_bytes": dict(expect or {})}


def test_repos_in_order_with_the_switch():
    keys = [r["key"] for r in fetch.repos(CFG)]
    assert keys == ["data", "dinov2", "qwen0.5b", "qwen1.5b", "qwen3b", "qwen7b", "qwen14b", "open_llama_3b"]
    by = {r["key"]: r for r in fetch.repos(CFG)}
    assert by["data"]["kind"] == "dataset" and by["data"]["ref"] == "wit_1024"
    assert by["dinov2"]["expect_bytes"] == {"model.safetensors": 346334872}
    assert by["open_llama_3b"]["expect_bytes"] == {"pytorch_model.bin": 6853038093}
    assert by["qwen7b"]["expect_bytes"] == {}
    off = {**CFG, "switches": {"open_llama": False}}
    assert "open_llama_3b" not in [r["key"] for r in fetch.repos(off)]


def test_select_takes_only_the_allowed_files():
    assert fetch.select(QWEN, CFG["text"]["qwen0.5b"]["allow"]) == [
        "config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "vocab.json"]
    timm = {"model.safetensors": 1, "pytorch_model.bin": 1, "open_clip_pytorch_model.bin": 1, "config.json": 1}
    assert fetch.select(timm, CFG["vision"]["allow"]) == ["model.safetensors"]  # never the pickle
    wit = {"data/train-00000-of-00002.parquet": 1, "data/train-00001-of-00002.parquet": 1, "README.md": 1,
           "data/validation-00000-of-00001.parquet": 1}
    assert fetch.select(wit, CFG["data"]["allow"]) == ["data/train-00000-of-00002.parquet",
                                                       "data/train-00001-of-00002.parquet"]
    llama = {n: 1 for n in ["config.json", "generation_config.json", "pytorch_model.bin", "special_tokens_map.json",
                            "tokenizer.model", "tokenizer_config.json", "README.md", "LICENSE"]}
    assert fetch.select(llama, CFG["text"]["open_llama_3b"]["allow"]) == sorted(CFG["text"]["open_llama_3b"]["allow"])


def test_check_file_against_lfs_sha256_and_git_blob_sha(tmp_path):
    big, small = tmp_path / "model.safetensors", tmp_path / "config.json"
    big.write_bytes(b"S" * 300)
    small.write_bytes(b'{"a": 1}')
    lfs = {"size": 300, "lfs_sha256": K.sha256_bytes(b"S" * 300), "blob_id": "0" * 40}
    blob = {"size": 8, "lfs_sha256": None, "blob_id": K.git_blob_sha(b'{"a": 1}')}
    assert fetch.check_file(big, lfs) == {"bytes": 300, "sha256": K.sha256_bytes(b"S" * 300)}
    assert fetch.check_file(small, blob) == {"bytes": 8, "sha256": K.sha256_bytes(b'{"a": 1}')}
    with pytest.raises(K.CheckError, match="hub"):
        fetch.check_file(big, {**lfs, "lfs_sha256": "1" * 64})
    with pytest.raises(K.CheckError, match="hub"):
        fetch.check_file(big, {**lfs, "size": 301})
    with pytest.raises(K.CheckError, match="hub"):
        fetch.check_file(small, {**blob, "blob_id": "2" * 40})
    # the git blob sha is git's: sha1 of "blob <size>\0" and the bytes
    assert K.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert K.git_blob_sha_file(small) == K.git_blob_sha(b'{"a": 1}')


def test_fetch_repo_downloads_at_the_commit_and_records_each_file(tmp_path):
    info, download, calls = fake_hub(tmp_path, QWEN)
    seen = []
    rec, snap = fetch.fetch_repo(spec(["*.json", "*.safetensors", "merges.txt"], {"model.safetensors": 300}), 40.0,
                                 info, download, before=seen.append, log=lambda *a: None)
    names = fetch.select(QWEN, ["*.json", "*.safetensors", "merges.txt"])
    assert calls == [("info", "org/model", "model", "main"), ("download", "org/model", REV, tuple(names))]
    assert rec["revision"] == REV and rec["ref"] == "main" and rec["kind"] == "model"
    assert rec["files"] == {n: {"bytes": len(QWEN[n]), "sha256": K.sha256_bytes(QWEN[n])} for n in names}
    assert seen == [fetch.upper_seconds(sum(len(QWEN[n]) for n in names), 40.0)]
    assert snap == K.snapshot_dir("org/model", REV, "model", hf_home=str(tmp_path / "hf"))
    assert fetch.upper_seconds(4_000_000_000, 40.0) == 160.0


def test_an_unapproved_size_stops_before_any_download(tmp_path):
    info, download, calls = fake_hub(tmp_path, QWEN)
    seen = []
    with pytest.raises(K.CheckError, match="approved"):
        fetch.fetch_repo(spec(["*.safetensors"], {"model.safetensors": 346334872}), 40.0, info, download,
                         before=seen.append, log=lambda *a: None)
    assert [c[0] for c in calls] == ["info"] and seen == []


def test_a_missing_approved_file_or_an_empty_selection_stops(tmp_path):
    info, download, calls = fake_hub(tmp_path, QWEN)
    with pytest.raises(K.CheckError, match="files"):
        fetch.fetch_repo(spec(["*.json"], {"pytorch_model.bin": 10}), 40.0, info, download, log=lambda *a: None)
    with pytest.raises(K.CheckError, match="files"):
        fetch.fetch_repo(spec(["*.gguf"]), 40.0, info, download, log=lambda *a: None)
    assert [c[0] for c in calls] == ["info", "info"]


def test_a_download_that_differs_from_the_listing_stops(tmp_path):
    info, download, _ = fake_hub(tmp_path, QWEN, served={"model.safetensors": b"T" * 300})
    with pytest.raises(K.CheckError, match="hub"):
        fetch.fetch_repo(spec(["*.safetensors"]), 40.0, info, download, log=lambda *a: None)
    info, download, _ = fake_hub(tmp_path, QWEN, served={"vocab.json": b'{"x": 1}'})
    with pytest.raises(K.CheckError, match="hub"):
        fetch.fetch_repo(spec(["*.json"]), 40.0, info, download, log=lambda *a: None)


def test_run_counts_the_rows_commits_after_each_repository_and_uses_the_hub_functions(tmp_path, monkeypatch):
    data = {"data/train-00000-of-00002.parquet": b"P" * 50, "data/train-00001-of-00002.parquet": b"Q" * 40,
            "README.md": b"# card"}
    hubs = {"org/data": fake_hub(tmp_path, data), "org/model": fake_hub(tmp_path, QWEN, rev="cd" * 20)}
    monkeypatch.setattr(fetch, "hub_info", lambda repo, kind, ref: hubs[repo][0](repo, kind, ref))
    monkeypatch.setattr(fetch, "hub_download", lambda repo, kind, rev, names: hubs[repo][1](repo, kind, rev, names))
    counted = []

    def facts(snap, names, decode=True):
        counted.append(names)
        assert all((snap / n).is_file() for n in names)
        return {"rows": [3, 2], "images": {"images": 5}, "captions": 5}

    monkeypatch.setattr(fetch, "dataset_facts", facts)
    s = {"repos": [spec(["data/train-*.parquet"], key="data", repo="org/data", kind="dataset"),
                   spec(["*.json", "*.safetensors", "merges.txt"], {"model.safetensors": 300}, key="qwen0.5b")],
         "mb_s": 40.0, "rows": 5}
    commits, guard = [], K.Guard("prepare")
    out = fetch.run(s, commit=lambda: commits.append(guard.row), log=lambda *a: None, guard=guard)
    assert list(out["prepared"]) == ["data", "qwen0.5b"] and commits == ["data", "qwen0.5b"]
    assert out["prepared"]["data"]["rows"] == [3, 2] and out["prepared"]["data"]["revision"] == REV
    assert out["prepared"]["qwen0.5b"]["revision"] == "cd" * 20 and "rows" not in out["prepared"]["qwen0.5b"]
    assert counted == [["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]]
    assert out["facts"]["data"]["captions"] == 5
    assert out["facts"]["bytes"] == 90 + sum(len(QWEN[n]) for n in fetch.select(QWEN, s["repos"][1]["allow"]))
    K.assert_text_free(out)
    with pytest.raises(K.CheckError, match="rows"):
        fetch.run({**s, "rows": 1024}, log=lambda *a: None)


def test_hub_info_reads_the_listing_fields(monkeypatch):
    import huggingface_hub

    asked = []

    class Api:
        def model_info(self, repo, revision=None, files_metadata=False):
            asked.append(("model", repo, revision, files_metadata))
            sib = [types.SimpleNamespace(rfilename="model.safetensors", size=300, blob_id="b" * 40,
                                         lfs=types.SimpleNamespace(size=300, sha256="a" * 64, pointer_size=134)),
                   types.SimpleNamespace(rfilename="config.json", size=8, blob_id="c" * 40, lfs=None)]
            return types.SimpleNamespace(sha="e" * 40, siblings=sib)

        def dataset_info(self, repo, revision=None, files_metadata=False):
            asked.append(("dataset", repo, revision, files_metadata))
            return types.SimpleNamespace(sha="f" * 40, siblings=[])

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    sha, files = fetch.hub_info("org/model", "model", "main")
    assert sha == "e" * 40
    assert files == {"model.safetensors": {"size": 300, "lfs_sha256": "a" * 64, "blob_id": "b" * 40},
                     "config.json": {"size": 8, "lfs_sha256": None, "blob_id": "c" * 40}}
    assert fetch.hub_info("minhuh/prh", "dataset", "wit_1024") == ("f" * 40, {})
    assert asked == [("model", "org/model", "main", True), ("dataset", "minhuh/prh", "wit_1024", True)]


def test_dataset_facts_on_two_parquet_files(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    Image = pytest.importorskip("PIL.Image")

    def png(color):
        b = io.BytesIO()
        Image.new("RGB", (4, 3), color).save(b, format="PNG")
        return b.getvalue()

    names = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
    for name, colors in zip(names, [["red", "blue"], ["green"]]):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        t = pa.table({"image": [{"bytes": png(c), "path": None} for c in colors],
                      "text": [["a caption", "another"] for _ in colors]})
        pq.write_table(t, tmp_path / name)
    d = fetch.dataset_facts(tmp_path, names)
    assert d["rows"] == [2, 1] and d["captions"] == 3
    assert d["images"]["formats"] == {"PNG": 3} and d["images"]["modes"] == {"RGB": 3}
    assert fetch.dataset_facts(tmp_path, names, decode=False) == {"rows": [2, 1]}
