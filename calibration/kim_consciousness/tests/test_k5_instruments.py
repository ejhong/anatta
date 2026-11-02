"""The instruments: ids, labels and hashes (always); the loaded data file's strings against those hashes, the prompt
formats and the scoring maps (only where the data file, kept outside the public repo, is present)."""

import instruments as I
import pytest

HAVE = I.path() is not None
needs_file = pytest.mark.skipif(not HAVE, reason="K5's instruments file is absent (kept outside the public repo: "
                                                 "/Users/eugene/prj/align/data/kim2026/instruments.json)")


def test_ids_labels_and_hashes():
    assert len(I.STRINGS) == 51
    ids = [k for k in I.STRINGS if k.startswith(("idaq.", "self.", "super.", "god.option"))]
    assert len(ids) == 21 + 5 + 13 + 4
    assert all(len(h) == 64 and lab for lab, h in I.STRINGS.values())
    assert sum(I.IDAQ_CATEGORIES.values()) == 21 and len(I.SELF_KEYS) == 5 and len(I.SUPER_KEYS) == 13


@needs_file
def test_file_matches_the_registered_hashes():
    strings = I.load()  # raises on any mismatch
    assert len(strings) == 51 + 96


@needs_file
def test_battery_and_formats():
    b = I.battery()
    fam = {}
    for it in b:
        fam[it["family"]] = fam.get(it["family"], 0) + 1
    assert fam == I.COUNTS and len({it["id"] for it in b}) == len(b)
    for it in b:
        assert it["prompt"].count("<answer>") == 1 and "<think>" in it["prompt"]
    assert I.afterlife()["prompt"].endswith(I.s("scaffold.gss_tail").format(k=2))
    assert "\n1. YES\n2. NO\n" in I.afterlife()["prompt"]
    assert I.god_prompt().count("\n(") == 4


@needs_file
def test_scoring_maps():
    b = {it["id"]: it for it in I.battery()}
    assert b["super.ghosts"]["score"] == {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0}
    assert b["god"]["score"] == {1: 0.0, 2: 3.33, 3: 6.67, 4: 10.0}
    assert b["self.soul"]["answers"] == list(range(11))


def test_a_tampered_file_is_refused(tmp_path, monkeypatch):
    import json

    if HAVE:
        d = json.loads(I.path().read_text())
        d["self"]["soul"] += " "
    else:
        d = {"idaq": {}, "self": {"soul": "x"}, "god_stem": "", "god_options": [], "supernatural": {}, "scaffold": {}}
    f = tmp_path / "i.json"
    f.write_text(json.dumps(d))
    monkeypatch.setenv("K5_INSTRUMENTS", str(f))
    I.load.cache_clear()
    with pytest.raises(ValueError):
        I.load()
    I.load.cache_clear()
