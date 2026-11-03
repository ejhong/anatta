"""P1's pure logic: the text conditions, the paired subsets and intervals, the sighting criteria, and the checks that keep
captions on Modal. Claude (Opus 5.5) wrote this file, 1 Nov 2026. Private (lab/explore).

Imported in P1's container (from /root, beside K1's k1.py and measure.py) and on the laptop (beside K1's folder). torch
is imported inside the functions that need it, so the laptop's spend and spec code runs without it.

Reused unchanged from K1 (calibration/prh_k1 at the K1 worktree): measure.mutual_knn and measure.subsample_interval (the
paper's code, galilai-group/prh-proof at a2b9949, MIT), and k1.assert_text_free and k1.CheckError. subset_rows is
measure.prh_subsample's subset draw, written as a function so that every condition is scored on the same rows (the
test checks it against prh_subsample).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
K1_DIR = Path("/Users/eugene/prj/align/anatta-calib-k1/calibration/prh_k1")  # K1's code (read and imported, never edited)
BENCH_DIR = HERE.parent / "bench"
EXPLORE_DIR = HERE.parent
ANATTA_SRC = HERE.parent.parent.parent / "src"  # main's src/ for the image (the laptop imports the installed package)


def ensure_paths() -> None:
    """On the laptop: K1's folder, bench's folder and lab/explore (for scout) on sys.path. In the container every
    module sits in /root, so the folders don't exist and nothing is added."""
    for p in (K1_DIR, BENCH_DIR, EXPLORE_DIR):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.append(str(p))


ensure_paths()
import k1 as K  # noqa: E402  (K1's helpers: stdlib only at import)

SPEC_FILE = HERE / "spec.yaml"

# ------------------------------------------------------------------------------------------------ errors

CODES = {  # P1's own stop codes; a container passes on only the code and numeric facts
    "spec": "the run's spec is missing a field or holds an unknown name",
    "sha256": "a file differs from the sha256 in the spec",
    "rows": "the data or the vision features hold another number of rows than the spec",
    "shape": "a model's layers or width, or a feature's shape, differ from the spec",
    "nonfinite": "a feature holds a NaN or an infinity",
    "hooks": "the model's hook count did not return to its value before an intervention",
    "control": "random_matched could not build a control within its tolerance",
    "caption": "a result about to leave the container holds a caption's text",
    "free_text": "a result about to leave the container holds free text",
    "equivalence": "P1's hooked features differ from K1's embed_texts on the uncut model",
    "fixed_pair": "a condition was scored before the plain, uncut condition fixed the layer pair",
}


class P1Check(K.CheckError):
    """K1's CheckError with P1's codes: the code and plain facts only (numbers, bools, hashes), so k1.run_guarded's
    describe() passes both on and nothing else."""

    def __init__(self, code: str, **facts):  # noqa: D107 (K1's __init__ checks K1's codes; this one checks P1's)
        if code not in CODES:
            raise ValueError(f"unknown P1Check code {code!r}")
        self.code = code
        self.facts = {k: v for k, v in facts.items() if K._plain(v)}
        RuntimeError.__init__(self, f"{code} {json.dumps(self.facts, sort_keys=True)}" if self.facts else code)


# ------------------------------------------------------------------------------------------------ the spec


def load_spec(path=SPEC_FILE) -> dict:
    """The spec, refused (ValueError) if YAML 1.1 turned a key into a non-string (an unquoted null, yes, no, on or off)
    or a closed-class word or a template into a non-string."""
    import yaml

    spec = yaml.safe_load(Path(path).read_text())

    def keys(o, where="spec"):
        if isinstance(o, dict):
            for k, v in o.items():
                if not isinstance(k, str):
                    raise ValueError(f"{where}: a key read as {k!r}; quote it")
                keys(v, f"{where}.{k}")
        elif isinstance(o, list):
            for v in o:
                keys(v, where)

    keys(spec)
    if not all(isinstance(w, str) for w in spec["lowercase_words"]) or \
            not all(isinstance(t, str) for t in spec["templates"].values()):
        raise ValueError("spec: a closed-class word or a template read as a non-string; quote it")
    return spec


def condition(text: str, intervention: str) -> str:
    """A condition's name: '<text condition>|<intervention>', e.g. 'fp1|ego'."""
    return f"{text}|{intervention}"


def split_condition(name: str) -> tuple[str, str]:
    t, i = name.split("|")
    return t, i


def direction_layers(spec_model: dict) -> list[int]:
    a, b = spec_model["direction"]["layers"]
    return list(range(int(a), int(b) + 1))


# ------------------------------------------------------------------------------------------------ the text conditions

FIRST_WORD = re.compile(r"[A-Za-z]+")


def lower_first(caption: str, words) -> tuple[str, bool]:
    """The caption with its first letter lowercased when its first word is a capitalised closed-class word (in `words`,
    compared in lower case) followed by a space or the end; else unchanged. Returns (text, changed)."""
    m = FIRST_WORD.match(caption)
    if not m:
        return caption, False
    w = m.group(0)
    titled = w[0].isupper() and (len(w) == 1 or w[1:].islower())
    ends = m.end() == len(caption) or caption[m.end()] == " "
    if titled and ends and w.lower() in set(words):
        return caption[0].lower() + caption[1:], True
    return caption, False


def check_template(tpl: str) -> None:
    if tpl.count("{c}") != 1 or tpl.replace("{c}", "").count("{") or tpl.replace("{c}", "").count("}"):
        raise ValueError(f"a template holds exactly one {{c}} and no other braces: {tpl!r}")


def apply_template(tpl: str, caption: str, words) -> tuple[str, bool]:
    """The caption in a template. The plain template ('{c}') returns the caption as it is; every other one fills {c}
    with the left-stripped caption, its first letter lowercased by lower_first. Returns (text, lowercased)."""
    check_template(tpl)
    if tpl == "{c}":
        return caption, False
    text, changed = lower_first(caption.lstrip(), words)
    return tpl.replace("{c}", text), changed


def condition_texts(captions: list[str], spec: dict) -> tuple[dict, dict]:
    """{text condition: [texts]} for every template in the spec, and counts only (no text): how many captions were
    lowercased and left-stripped in the templated conditions."""
    words = spec["lowercase_words"]
    out = {name: [apply_template(tpl, c, words)[0] for c in captions] for name, tpl in spec["templates"].items()}
    lowered = sum(lower_first(c.lstrip(), words)[1] for c in captions)  # the same in every templated condition
    stripped = sum(c != c.lstrip() for c in captions)
    return out, {"captions": len(captions), "lowercased": lowered, "left_stripped": stripped}


def prefix(tpl: str) -> str:
    """What comes before the caption, as it is tokenized at the start of the text: 'I see {c}' -> 'I see' (the space
    goes with the caption's first token in a byte-level BPE); the plain template's prefix is ''."""
    check_template(tpl)
    return tpl.split("{c}")[0].rstrip()


# ------------------------------------------------------------------------------------------------ subsets and intervals


def subset_rows(n_rows: int, n: int = 200, size: int = 800, seed: int = 0) -> list:
    """measure.prh_subsample's subsets: n draws of `size` rows without replacement from one generator seeded `seed`.
    They depend only on the seed and the number of rows, so every condition is scored on the same rows."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return [torch.randperm(n_rows, generator=g)[:size] for _ in range(n)]


def subset_scores(a, b, rows, k: int = 10):
    """measure.mutual_knn on each subset of rows of two prepared feature matrices (prh_subsample's loop, given features
    already through _prh_prep on all rows and the subsets from subset_rows)."""
    import torch
    from measure import mutual_knn

    return torch.tensor([mutual_knn(a[r.to(a.device)], b[r.to(b.device)], k) for r in rows])


def interval(full: float, subsets, n_rows: int, size: int, z: float = 1.96) -> list[float]:
    """measure.subsample_interval: the paper's normal-approximation interval from subset values."""
    import torch
    from measure import subsample_interval

    lo, hi = subsample_interval(float(full), torch.as_tensor(subsets, dtype=torch.float64), n_rows, size, z)
    return [lo, hi]


def r9(x: float) -> float:
    """Rounded to 1e-9 before a threshold comparison, as K1 rounds its differences."""
    return round(float(x), 9)


def group_score(scores: dict, subsets: dict, texts: list[str], intervention: str):
    """A group's score and subset vector: the mean over its text conditions under one intervention."""
    import torch

    names = [condition(t, intervention) for t in texts]
    full = sum(float(scores[n]) for n in names) / len(names)
    sub = torch.stack([torch.as_tensor(subsets[n], dtype=torch.float64) for n in names]).mean(0)
    return full, sub


def diff(scores: dict, subsets: dict, x: tuple, y: tuple, n_rows: int, size: int, z: float = 1.96) -> dict:
    """Group x minus group y, each (texts, intervention), with the paired interval from the same subsets."""
    fx, sx = group_score(scores, subsets, *x)
    fy, sy = group_score(scores, subsets, *y)
    d = fx - fy
    return {"diff": d, "interval": interval(d, sx - sy, n_rows, size, z), "x": fx, "y": fy}


# ------------------------------------------------------------------------------------------------ the criteria


def criterion_a(scores: dict, subsets: dict, groups: dict, n_rows: int, size: int, z: float, threshold: float) -> dict:
    """(a): D_a = (nt - plain) - (fp - plain) = nt - fp on the uncut model; a sighting iff D_a >= threshold and the
    lower end of its paired interval is above 0. Also each group's change from plain."""
    fp, nt, plain = (groups["fp"], "none"), (groups["nt"], "none"), (groups["plain"], "none")
    d_fp = diff(scores, subsets, fp, plain, n_rows, size, z)
    d_nt = diff(scores, subsets, nt, plain, n_rows, size, z)
    d_a = diff(scores, subsets, nt, fp, n_rows, size, z)
    sighting = r9(d_a["diff"]) >= r9(threshold) and d_a["interval"][0] > 0
    return {"fp_minus_plain": d_fp, "nt_minus_plain": d_nt, "D_a": d_a, "threshold": threshold,
            "sighting": bool(sighting)}


def criterion_b(scores: dict, subsets: dict, groups: dict, n_rows: int, size: int, z: float, threshold: float,
                target: str = "ego", controls=("rand0", "rand1", "rand2", "rand3"), group: str = "fp") -> dict:
    """(b): for the group's captions, delta_X = group(X) - group(none) for the target and each control cut; a sighting
    iff delta_target beats every control's delta and the margin over the best control is at least the threshold. The
    margin's paired interval (group(target) - group(best control)) is reported beside it."""
    texts = groups[group]
    deltas = {x: diff(scores, subsets, (texts, x), (texts, "none"), n_rows, size, z) for x in (target, *controls)}
    best = max(controls, key=lambda c: deltas[c]["diff"])
    margin = diff(scores, subsets, (texts, target), (texts, best), n_rows, size, z)
    beats = all(r9(deltas[target]["diff"]) > r9(deltas[c]["diff"]) for c in controls)
    sighting = beats and r9(margin["diff"]) >= r9(threshold)
    return {"group": group, "deltas": deltas, "best_control": best, "margin": margin, "beats_every_control": bool(beats),
            "threshold": threshold, "sighting": bool(sighting)}


# ------------------------------------------------------------------------------------------------ one condition's scores


def score_condition(fa, pa, feats, rows, k: int, fixed, dev) -> dict:
    """One condition's scores by K1's code: prh_score's own maximum (the paper's code, with its own preparation), the
    by-layer table (measure.null_scores' identity re-pairing, which K1 checked against prh_score), the score at the
    fixed pair (measure.mutual_knn; the own maximum's pair when fixed is None), and K1's subsets at the fixed pair and
    at the own maximum. fa: the vision layers; pa: the same through _prh_prep on `dev`; feats: the text layers."""
    import measure as M
    import torch

    pb = [M._prh_prep(f.to(dev)) for f in feats]
    own = M.prh_score(fa, feats, k)
    table = M.null_scores(pa, pb, [torch.arange(len(feats[0]))], k)[0].cpu()
    i, j = fixed if fixed is not None else (own[1], own[2])
    rec = {"own": {"score": own[0], "layers": [own[1], own[2]]},
           "table_matches": abs(float(table.max()) - own[0]) <= 1e-6,
           "by_layer": [[round(float(x), 6) for x in row] for row in table.tolist()],
           "fixed": {"score": M.mutual_knn(pa[i], pb[j], k), "layers": [i, j]},
           "fixed_subsets": subset_scores(pa[i], pb[j], rows, k).tolist(),
           "own_subsets": subset_scores(pa[own[1]], pb[own[2]], rows, k).tolist()}
    del pb
    return rec


def sanitize_made(made: dict) -> dict:
    """random_matched's record of a control without its free-text fields (the rule's sentence and the neutral set's
    position and outlier descriptions, constants of steering.py): numbers, names and hashes only."""
    out = {k: v for k, v in made.items() if k != "rule"}
    out["neutral"] = {k: v for k, v in made.get("neutral", {}).items() if k not in ("positions", "outlier_rule")}
    return out


def nan_none(xs) -> list:
    return [None if (x is None or (isinstance(x, float) and x != x)) else x for x in xs]


# ------------------------------------------------------------------------------------------------ captions stay on Modal


def strings_in(obj):
    """Every string in obj, keys included."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from strings_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from strings_in(v)


def assert_no_captions(obj, texts, min_len: int = 8) -> None:
    """P1Check('caption') if any of `texts` (captions and the texts made from them) is a string of obj, or, at
    min_len characters or more, part of one. With k1.assert_text_free (no spaces, a small alphabet), this keeps every
    caption off the laptop."""
    strings = list(strings_in(obj))
    whole = set(strings)
    blob = "\x00".join(strings)
    for t in texts:
        s = t.strip()
        if not s:
            continue
        if s in whole or (len(s) >= min_len and s in blob):
            raise P1Check("caption")


def leaving(obj, texts) -> dict:
    """A result about to leave the container: as its JSON reads back, checked text-free (k1.assert_text_free, as
    P1Check free_text) and caption-free."""
    obj = json.loads(json.dumps(obj))
    try:
        K.assert_text_free(obj)
    except K.CheckError:
        raise P1Check("free_text") from None
    assert_no_captions(obj, texts)
    return obj
