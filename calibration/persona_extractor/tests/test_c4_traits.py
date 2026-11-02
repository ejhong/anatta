"""traits.py: the repo's trait files (made-up stand-ins here), eval_persona's system strings, the layout, and the
text-free errors."""

import json

import pytest
import traits as T
from c4_testing import CHECK, benign_trait, trait_bytes

PROMPTS = {"pos": "You are {article} {trait} assistant. {instruction}", "neg": "You are a helpful assistant. {instruction}",
           "family": "calibration", "description": "the persona-vectors repo's {trait} trait at b8e0f04"}


def eval_persona_a_or_an(word):  # the registration's rule: "an" before a vowel
    return "an" if word[0].lower() in "aeiou" else "a"


@pytest.mark.parametrize("name", ["evil", "optimistic", "sycophantic", "upbeat", "earnest", "kind", "Orderly"])
def test_system_strings_equal_eval_persona(name):
    inst = {"pos": "Be cheerful about the garden.", "neg": "Be plain about the garden."}
    got = T.system_strings(name, inst, PROMPTS)
    assert got["pos"] == f"You are {eval_persona_a_or_an(name)} {name} assistant. {inst['pos']}"
    assert got["neg"] == f"You are a helpful assistant. {inst['neg']}"
    assert got["pos"].startswith("You are an " if name[0].lower() in "aeiou" else "You are a ")


def test_trait_carries_five_pairs_and_twenty_questions():
    data = benign_trait("upbeat", n_q=20)
    facts = T.parse_trait_file(json.dumps(data).encode(), 5, 20)
    trait = T.probe_trait("upbeat", facts, PROMPTS)
    assert trait.name == "upbeat" and trait.family == "calibration" and trait.description
    assert trait.questions == data["questions"] and len(trait.questions) == 20
    assert trait.pairs == [T.system_strings("upbeat", i, PROMPTS) for i in data["instruction"]]
    assert len(trait.pairs) == 5 and all(set(p) == {"pos", "neg"} for p in trait.pairs)


def test_a_file_with_other_keys_is_refused():
    data = benign_trait("upbeat", n_q=20)
    with pytest.raises(T.CheckError) as e:
        T.parse_trait_file(json.dumps({**data, "extra": 1}).encode(), 5, 20)
    assert e.value.code == "keys"
    with pytest.raises(T.CheckError):
        T.parse_trait_file(json.dumps({k: v for k, v in data.items() if k != "eval_prompt"}).encode(), 5, 20)
    with pytest.raises(T.CheckError) as e:
        T.parse_trait_file(json.dumps({**data, "questions": data["questions"][:19]}).encode(), 5, 20)
    assert e.value.code == "questions"
    with pytest.raises(T.CheckError) as e:
        T.parse_trait_file(json.dumps({**data, "instruction": data["instruction"][:4]}).encode(), 5, 20)
    assert e.value.code == "pairs"


def test_prepare_checks_blob_and_size_and_writes_the_volume(tmp_path):
    data = trait_bytes("upbeat", n_q=20)
    repo = {"raw": "https://example.invalid/raw", "commit": "b8e0f044fe2410a6fad579f38324f03f13b4e917",
            "trait_dir": "data", "expect": {"pairs": 5, "questions": 20},
            "traits": {"upbeat": {"bytes": len(data), "blob": T.git_blob_sha(data)}}}
    seen = []
    res = T.prepare(repo, ["upbeat"], tmp_path, fetch=lambda url: seen.append(url) or data)
    assert seen == [f"https://example.invalid/raw/{repo['commit']}/data/upbeat.json"]
    assert res["upbeat"] == {"bytes": len(data), "blob": T.git_blob_sha(data), "sha256": T.sha256_bytes(data),
                             "pairs": 5, "questions": 20}
    assert (tmp_path / "upbeat.json").read_bytes() == data
    T.assert_text_free(res)
    bad = {**repo, "traits": {"upbeat": {"bytes": len(data), "blob": "0" * 40}}}
    with pytest.raises(T.CheckError) as e:
        T.prepare(bad, ["upbeat"], tmp_path / "b", fetch=lambda url: data)
    assert e.value.code == "blob"


def test_registered_trait_needs_the_committed_sha256():
    data = trait_bytes("upbeat", n_q=20)
    entry = {"bytes": len(data), "blob": T.git_blob_sha(data), "sha256": None, "pairs": None, "questions": None}
    with pytest.raises(T.CheckError) as e:
        T.registered_trait(data, entry, {"pairs": 5, "questions": 20})
    assert e.value.code == "unregistered"
    entry.update(sha256=T.sha256_bytes(data), pairs=5, questions=20)
    assert T.registered_trait(data, entry, {"pairs": 5, "questions": 20})["questions"][0].startswith("What")
    with pytest.raises(T.CheckError):
        T.registered_trait(data, {**entry, "sha256": "f" * 64}, {"pairs": 5, "questions": 20})


def test_the_vendored_file_is_generate_vec_at_b8e0f04():
    data = (CHECK / T.GENERATE_VEC).read_bytes()
    assert T.git_blob_sha(data) == "d819f48e2753ed7cc9b364c19be5ab00bdf2a056" and len(data) == 5250
    lic = (CHECK / "third_party" / "persona_vectors" / "LICENSE").read_bytes()
    assert T.git_blob_sha(lic) == "2d13a3486f0d973c29937f7b4567a4668e9b04b7"


def test_layout_rows_cells_and_batches():
    n_q, n_i, n_s = 20, 5, 10
    for k in (0, 9, 10, 49, 50, 999):
        qi, ii, si = T.cell_of(k, n_i, n_s)
        assert T.row_of(qi, ii, si, n_i, n_s) == k
    assert T.cell_of(10, n_i, n_s) == (0, 1, 0) and T.cell_of(50, n_i, n_s) == (1, 0, 0)
    b = T.batches(["evil", "sycophantic", "optimistic"], n_q, n_i, n_s, 200)
    assert len(b) == 3 * 2 * 5 and [x["index"] for x in b] == list(range(30))
    assert all(len(x["rows"]) == 200 for x in b) and b[5]["sign"] == "neg" and b[10]["trait"] == "sycophantic"
    assert T.hs_index(19) == 20


def test_check_errors_and_guards_carry_no_text():
    e = T.CheckError("render", row=3, note="a sentence from a reply", sha="ab12cd34")
    assert e.facts == {"row": 3, "sha": "ab12cd34"} and "sentence" not in str(e)
    with pytest.raises(ValueError):
        T.CheckError("not a code")

    def boom(guard):
        guard.row = 7
        raise KeyError("the reply said something private")

    with pytest.raises(T.StageError) as err:
        T.run_guarded("G", boom)
    assert str(err.value) == "G: row 7: KeyError"
    assert err.value.__context__ is None and err.value.__cause__ is None
    with pytest.raises(T.CheckError):
        T.assert_text_free({"x": "two words"})
    T.assert_text_free({"gpu": "NVIDIA L40S", "sha": "ab" * 32, "pkg": "torch==2.14.1", "n": [1, 2.5, None, True]})


def test_the_data_code_and_the_analysis_code_are_hashed_apart(tmp_path):
    from c4_testing import PKG

    code = T.code_hashes(CHECK, PKG)
    assert set(code) == set(T.DATA_FILES) | {f"anatta/{f}" for f in T.LIBRARY_FILES}
    assert "analysis.py" not in code and "card.py" not in code
    assert T.analysis_code_hashes(CHECK) == {"analysis.py": T.sha256_file(CHECK / "analysis.py")}
    assert set(T.IMAGE_FILES) == set(T.DATA_FILES) | set(T.ANALYSIS_FILES)
    src = "import x\n\n\n@deco(1)\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n"
    assert T.function_source(src, ("b", "a")) == "def b():\n    return 2\n@deco(1)\ndef a():\n    return 1\n"
    assert T.plan_id({"b": 1, "a": 2}) == T.plan_id({"a": 2, "b": 1}) and len(T.plan_id({})) == 12
