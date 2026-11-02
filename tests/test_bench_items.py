"""Item draws (anatta.bench.items): the largest-remainder allocation of MMLU-Pro's test set, the seed-0 draws within
each category (indexing that category's rows in parquet order), the cheap tier and batch-1 subsets, the battery's
own draw, and the item file. The category sizes are those of TIGER-Lab/MMLU-Pro's test split (12,032 rows)."""

import random

import pytest

from anatta.bench import items as bi

SIZES = {"math": 1351, "physics": 1299, "chemistry": 1132, "law": 1101, "engineering": 969, "other": 924,
         "economics": 844, "health": 818, "psychology": 798, "business": 789, "biology": 717, "philosophy": 499,
         "computer science": 410, "history": 381}
C1 = {"math": 112, "physics": 108, "chemistry": 94, "law": 91, "engineering": 81, "other": 77, "economics": 70,
      "health": 68, "psychology": 66, "business": 66, "biology": 60, "philosophy": 41, "computer science": 34,
      "history": 32}


def test_allocation_of_1000_reproduces_the_registered_table():
    seats, info = bi.allocate(SIZES, 1000)
    assert seats == C1 and sum(seats.values()) == 1000
    assert [c for c, _ in info["extra"]] == ["health", "physics", "other", "history", "biology", "business", "engineering"]
    assert info["first_out"] == ("law", 0.506) and info["tie_at_cut"] is False


def test_allocation_of_150_and_30():
    seats, info = bi.allocate(SIZES, 150)
    assert sum(seats.values()) == 150 and info["tie_at_cut"] is False
    assert seats == {"math": 17, "physics": 16, "chemistry": 14, "law": 14, "engineering": 12, "other": 11,
                     "economics": 11, "health": 10, "psychology": 10, "business": 10, "biology": 9, "philosophy": 6,
                     "computer science": 5, "history": 5}
    seats, info = bi.allocate(SIZES, 30)
    assert sum(seats.values()) == 30 and info["tie_at_cut"] is False and min(seats.values()) >= 1


def test_allocation_flags_a_tie_at_the_cut():
    seats, info = bi.allocate({"a": 1, "b": 1, "c": 1, "d": 1}, 2)  # four equal remainders, two seats
    assert sum(seats.values()) == 2 and info["tie_at_cut"] is True
    assert bi.allocate({"a": 5, "b": 5}, 10)[0] == {"a": 5, "b": 5}
    with pytest.raises(ValueError):
        bi.allocate({"a": 2}, 3)


def test_draw_indices_is_the_batterys_draw():
    for n, k in ((1319, 250), (12032, 250), (1172, 250), (381, 32)):
        assert bi.draw_indices(n, k) == sorted(random.Random(0).sample(range(n), k))


def _categories(seed=1):
    """One category per row, the real sizes, interleaved in a fixed shuffled order (as parquet order might be)."""
    cats = [c for c, n in SIZES.items() for _ in range(n)]
    random.Random(seed).shuffle(cats)
    return cats


def test_stratified_draw_seeds_each_category_on_its_rows_in_order():
    cats = _categories()
    rows, strata = bi.stratified(cats, len(cats), 1000, seed=0)
    assert len(rows) == 1000 == len(set(rows)) and rows == sorted(rows)
    assert strata["allocation"] == C1
    for c, k in C1.items():
        members = [i for i, x in enumerate(cats) if x == c]  # that category's rows, in parquet order
        assert sorted(members[j] for j in bi.draw_indices(len(members), k, 0)) == [i for i in rows if cats[i] == c]


def test_cheap_tier_and_batch_one_subsets_sit_inside_the_1000():
    cats = _categories()
    c1 = set(bi.stratified(cats, len(cats), 1000)[0])
    for k in (150, 30):
        sub, strata = bi.stratified(cats, len(cats), k)
        assert len(sub) == k and set(sub) <= c1, k
        assert sum(strata["allocation"].values()) == k
    # a set without categories is one stratum: the battery's draw
    assert bi.stratified(None, 1319, 150) == (bi.draw_indices(1319, 150), {"allocation": None, "info": None})
    assert set(bi.draw_indices(1319, 150)) <= set(bi.draw_indices(1319, 250))


def _rows():
    gsm = [{"question": f"Q{i}?", "answer": f"steps\n#### {1000 + i:,}"} for i in range(40)]
    mmlu = [{"question_id": 100 + i, "question": f"M{i}?", "options": [f"o{j}" for j in range(4 + i % 7)],
             "answer": "B", "answer_index": 1, "cot_content": "", "category": ["math", "law"][i % 2], "src": "s"}
            for i in range(40)]
    arc = [{"id": f"A_{i}", "question": f"A{i}?", "choices": {"text": ["w", "x", "y", "z"], "label": ["1", "2", "3", "4"]},
            "answerKey": "3"} for i in range(40)]
    return {"gsm8k": gsm, "mmlu_pro": mmlu, "arc_challenge": arc}


DRAWS = {"c1_gsm8k": {"set": "gsm8k", "n": "all"}, "c1_mmlu_pro": {"set": "mmlu_pro", "n": 20, "stratify": "category"},
         "cheap_mmlu_pro": {"set": "mmlu_pro", "n": 6, "stratify": "category"},
         "battery_arc_challenge": {"set": "arc_challenge", "n": 10, "rule": "battery"}}


def test_item_file_holds_ids_categories_prompts_golds_and_the_file_hashes():
    datasets = {"gsm8k": {"repo": "openai/gsm8k", "revision": "c" * 40, "files": [{"path": "main/test.parquet", "sha256": "ab" * 32, "rows": 40}]}}
    f = bi.build_item_file(_rows(), DRAWS, seed=0, datasets=datasets)
    assert f["meta"]["datasets"] == datasets and f["meta"]["rows"] == {"gsm8k": 40, "mmlu_pro": 40, "arc_challenge": 40}
    assert len(f["draws"]["c1_gsm8k"]) == 40 and f["draws"]["c1_gsm8k"][:2] == ["gsm8k/0", "gsm8k/1"]
    assert f["meta"]["draws"]["c1_mmlu_pro"]["allocation"] == {"law": 10, "math": 10}
    assert set(f["draws"]["cheap_mmlu_pro"]) <= set(f["draws"]["c1_mmlu_pro"])
    g = f["items"]["gsm8k/3"]
    assert (g["gold"], g["category"], g["source_id"]) == ("1003", None, bi.inspect_gsm8k_id("Q3?"))
    assert g["prompts"] == {"gsm8k": 'Q3?\n\nWork it out step by step, then give the final answer on the last line as "Answer: <number>".'}
    m = f["items"][f["draws"]["c1_mmlu_pro"][0]]
    assert m["category"] in ("math", "law") and m["question_id"] == m["source_id"] == 100 + m["row"]
    assert m["n_options"] == 4 + m["row"] % 7 and set(m["prompts"]) == {"mmlu_pro", "mmlu_pro_cot"}
    assert m["prompts"]["mmlu_pro"].endswith("\n\nReply with the letter of the correct option only.")
    assert "\nA) o0\nB) o1" in m["prompts"]["mmlu_pro_cot"] and "Think step by step" in m["prompts"]["mmlu_pro_cot"]
    a = f["items"][f["draws"]["battery_arc_challenge"][0]]
    assert a["gold"] == "C" and "\nA. w\nB. x\nC. y\nD. z\n" in a["prompts"]["arc_challenge"]  # 1-4 shown as A-D
    assert f["draws"]["battery_arc_challenge"] == [f"arc_challenge/{i}" for i in bi.draw_indices(40, 10)]
    assert [x["id"] for x in bi.draw_items(f, "cheap_mmlu_pro")] == f["draws"]["cheap_mmlu_pro"]


def test_item_file_bytes_and_hash_are_deterministic():
    a = bi.dump_items(bi.build_item_file(_rows(), DRAWS))
    b = bi.dump_items(bi.build_item_file(_rows(), DRAWS))
    assert a == b and bi.sha256_bytes(a) == bi.sha256_bytes(b) and a.endswith(b"\n")


def test_inspect_gsm8k_id_matches_inspect_evals_examples():
    # inspect_evals' create_stable_id docstring: create_stable_id("What is 2+2?", "4", prefix="gsm8k") is for two
    # fields; the one-field form hashes the question alone
    import hashlib

    assert bi.inspect_gsm8k_id("What is 2+2?") == "gsm8k_" + hashlib.md5(b"What is 2+2?").hexdigest()[:8]


def test_parquet_files_and_sha256_file(tmp_path):
    listing = ["README.md", "data/test-00001-of-00002.parquet", "data/test-00000-of-00002.parquet",
               "data/validation-00000-of-00001.parquet"]
    assert bi.parquet_files(listing, "data/test") == ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert bi.sha256_file(p) == bi.sha256_bytes(b"abc")
