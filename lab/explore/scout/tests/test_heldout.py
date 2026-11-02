"""The held-out guard: scout refuses any spec that names a held-out item or triple, asks for the held-out split, or
points the library at the held-out directory. Claude wrote this file."""

from __future__ import annotations

import pytest
import yaml

from conftest import load, spec_dict
from scout import library as lib
from scout import paths as P
from scout.spec import SpecError, load_spec


def held_ids(mini):
    doc = yaml.safe_load(mini["heldout"].read_text())
    return {fam: [it["id"] for it in v["items"]] for fam, v in doc["families"].items()}


def test_dev_spec_loads(mini):
    spec = load(mini)
    held = lib.heldout_ids(mini["heldout"])
    chosen = {i for ids in spec["items"].values() for i in ids}
    assert chosen and not (chosen & held)


def test_refuses_a_heldout_item_in_a_list(mini):
    hid = held_ids(mini)["deference"][0]
    with pytest.raises(lib.HeldOutError, match="held-out"):
        load(mini, items=[hid])


def test_refuses_a_heldout_item_in_a_family_map(mini):
    hid = held_ids(mini)["threat_to_self"][0]
    with pytest.raises(lib.HeldOutError, match="held-out"):
        load(mini, items={"threat_to_self": [hid]})


def test_refuses_a_heldout_triple(mini):
    triple = held_ids(mini)["who_benefits"][0].rsplit("-", 1)[0]
    with pytest.raises(lib.HeldOutError):
        load(mini, items=[triple])
    with pytest.raises(lib.HeldOutError):
        load(mini, items={"who_benefits": [triple + "-self"]})


def test_refuses_a_mix_of_dev_and_heldout(mini):
    spec = load(mini)
    dev_id = spec["items"]["deference"][0]
    hid = held_ids(mini)["deference"][0]
    with pytest.raises(lib.HeldOutError):
        load(mini, items=[dev_id, hid])


def test_refuses_the_heldout_split(mini):
    with pytest.raises(lib.HeldOutError):
        load(mini, split="heldout")


def test_refuses_a_library_path_into_heldout(mini):
    with pytest.raises(lib.HeldOutError):
        load_spec(spec_dict(library=str(P.HELDOUT_DIR)), dev_dir=mini["dev"], heldout_file=mini["heldout"])
    with pytest.raises(lib.HeldOutError):
        lib.load_family("deference", P.HELDOUT_DIR)


def test_refuses_to_run_without_the_heldout_file(mini, tmp_path):
    with pytest.raises(lib.LibraryError):
        load_spec(spec_dict(), dev_dir=mini["dev"], heldout_file=tmp_path / "missing.yaml")


def test_unknown_items_are_errors(mini):
    with pytest.raises(SpecError):
        load(mini, items=["df-99"])


def test_cli_exit_code_for_a_refused_spec(mini, tmp_path, monkeypatch):
    """The real library: a spec naming a real held-out id exits 2 with the refusal, before anything is spent."""
    from scout.__main__ import main
    hid = sorted(lib.heldout_ids())[0]
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.safe_dump(spec_dict(items=[hid], models=[{"id": "x/y"}])))
    monkeypatch.setenv("SCOUT_RESULTS_ROOT", str(tmp_path / "results"))
    assert main(["run", str(f), "--dry-run"]) == 2
    assert not (tmp_path / "results").exists()


def test_real_split_is_disjoint_and_follows_the_hash():
    held = lib.heldout_ids()
    doc = yaml.safe_load(P.HELDOUT_FILE.read_text())
    for fam in lib.FAMILIES:
        dev = lib.load_family(fam)["items"]
        hitems = doc["families"][fam]["items"]
        assert not ({it["id"] for it in dev} & held)
        d2, h2 = lib.split_family(dev + hitems)
        assert {it["id"] for it in d2} == {it["id"] for it in dev}
        assert {it["id"] for it in h2} == {it["id"] for it in hitems}


def test_heldout_sha_matches_the_readme():
    assert lib.recorded_heldout_sha256() == lib.file_sha256(P.HELDOUT_FILE)
