"""Expand texts/source.yaml into the four classes' rows (Chua et al.'s format: one user turn, one assistant turn) and
record their sha256 in texts/MANIFEST.json before any extraction. Claude (Opus 5.5) wrote this file, 2 Nov 2026.
Private (lab/explore). Evaluation and steering texts only, never training data.

Each question gets four phrasings (the question plus one of Chua's "keep it short" suffixes, chosen by a fixed
rotation); answer variant 0 goes with phrasings 0 and 2, variant 1 with 1 and 3. So every class has 4 rows per
question, the same questions in every class, and no row repeats inside a class. The train/held-out split is by
question (split_questions), so no question's phrasings sit on both sides.

  python3 -I lab/explore/noself/build_texts.py          # writes texts/<class>.jsonl and texts/MANIFEST.json
  python3 -I lab/explore/noself/build_texts.py --check  # recomputes and compares, writes nothing
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXTS = HERE / "texts"
CLASSES = ("aff", "deny", "nofix", "topic")
SUFFIXES = ["Keep it short.", "Be brief.", "Keep it brief.", "Answer briefly.", "Short answer.", "Brief reply.",
            "Short reply.", "Brief answer."]  # Chua's most common suffixes
PHRASINGS = 4
TRAIN_FRAC = 0.8
SPLIT_SEED = 0
# NOFIX must not carry religious or contemplative vocabulary, nor the letters "self" (so the direction is not a
# Buddhism-topic or a "self"-word direction). Matched case-insensitively as substrings.
BANNED_NOFIX = ["self", "buddh", "dharma", "dhamma", "anatta", "anatman", "nirvana", "enlighten", "awaken",
                "meditat", "monk", "zen", "karma", "soul", "spirit", "sacred", "god", "divine", "empti", "ego",
                "koan", "sutra", "mindful", "contemplat", "teacher", "tradition", "practice", "retreat", "insight"]


def load_source(p: Path = TEXTS / "source.yaml") -> list[dict]:
    import yaml

    qs = yaml.safe_load(p.read_text(encoding="utf-8"))["questions"]
    for q in qs:
        for c in CLASSES:
            if len(q[c]) != 2:
                raise ValueError(f"{q['q']!r}: class {c} needs two answers")
    return qs


def banned_hits(text: str) -> list[str]:
    t = text.lower()
    return [w for w in BANNED_NOFIX if w in t]


def check_source(qs: list[dict]) -> list[str]:
    """Problems with the source: a NOFIX answer with banned vocabulary, an answer not ending in one period, a
    duplicate answer inside a class, or the same answer in two classes."""
    bad = []
    seen = {c: set() for c in CLASSES}
    allans = {}
    for q in qs:
        for c in CLASSES:
            for a in q[c]:
                if not a.endswith(".") or a.endswith(".."):
                    bad.append(f"{c}: does not end in one period: {a!r}")
                if c == "nofix" and banned_hits(a):
                    bad.append(f"nofix: banned {banned_hits(a)}: {a!r}")
                if a in seen[c]:
                    bad.append(f"{c}: duplicate {a!r}")
                seen[c].add(a)
                if a in allans and allans[a] != c:
                    bad.append(f"{a!r} in both {allans[a]} and {c}")
                allans[a] = c
    return bad


def expand(qs: list[dict]) -> dict[str, list[dict]]:
    """{class: [row]}; row = {qid, phrasing, messages}."""
    out = {c: [] for c in CLASSES}
    for qi, q in enumerate(qs):
        for k in range(PHRASINGS):
            user = f"{q['q']} {SUFFIXES[(qi + 3 * k) % len(SUFFIXES)]}"
            for c in CLASSES:
                out[c].append({"qid": qi, "phrasing": k,
                               "messages": [{"role": "user", "content": user},
                                            {"role": "assistant", "content": q[c][k % 2]}]})
    return out


def split_questions(n_questions: int, frac: float = TRAIN_FRAC, seed: int = SPLIT_SEED) -> tuple[set, set]:
    idx = list(range(n_questions))
    random.Random(seed).shuffle(idx)
    k = int(round(frac * n_questions))
    return set(idx[:k]), set(idx[k:])


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def manifest(rows: dict[str, list[dict]], n_questions: int) -> dict:
    tr, ho = split_questions(n_questions)
    m = {"made_by": "Claude Opus 5.5, 2 Nov 2026; evaluation and steering texts only, never training data",
         "source_sha256": hashlib.sha256((TEXTS / "source.yaml").read_bytes()).hexdigest(),
         "split": {"by": "question", "train_frac": TRAIN_FRAC, "seed": SPLIT_SEED, "train_qids": sorted(tr),
                   "held_qids": sorted(ho)},
         "classes": {}}
    for c, rs in rows.items():
        body = jsonl(rs)
        lens = [len(r["messages"][1]["content"]) for r in rs]
        words = [len(re.findall(r"\S+", r["messages"][1]["content"])) for r in rs]
        m["classes"][c] = {"rows": len(rs), "train": sum(r["qid"] in tr for r in rs),
                           "held": sum(r["qid"] in ho for r in rs), "sha256": hashlib.sha256(body.encode()).hexdigest(),
                           "answer_chars_mean": round(sum(lens) / len(lens), 1),
                           "answer_words_mean": round(sum(words) / len(words), 2)}
    return m


def main(argv: list[str]) -> int:
    qs = load_source()
    bad = check_source(qs)
    if bad:
        print("\n".join(bad))
        return 1
    rows = expand(qs)
    m = manifest(rows, len(qs))
    if "--check" in argv:
        old = json.loads((TEXTS / "MANIFEST.json").read_text())
        same = all(old["classes"][c]["sha256"] == m["classes"][c]["sha256"] for c in CLASSES)
        print("texts match MANIFEST.json" if same else "texts differ from MANIFEST.json")
        return 0 if same else 1
    for c in CLASSES:
        (TEXTS / f"{c}.jsonl").write_text(jsonl(rows[c]), encoding="utf-8")
    (TEXTS / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps({c: v for c, v in m["classes"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
