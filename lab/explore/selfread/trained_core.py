"""Repair three's pure parts: the passages, the questions, the statistics, the words and the reading from a folder.
The design, its known cases, its words, what each word does to the site and the list of changes since the first
version are at the top of modal_selfread.py. Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026 and
rewrote it the same day after two cold reads of the first version, before any reading.

Reused from the lab, unchanged: step 6's events, names, units and bootstrap (lab/explore/attractor/self_core.py). The
stakes of rung (ii) are sentences of the Atlas scaffold of 29 Oct (lab/experiments/h4lite/bridge_core.py), copied here;
a test holds them to it. Nothing here needs torch or the network.

Reading from disk, once the app has written a folder:
  .venv/bin/python lab/explore/selfread/trained_core.py <results>/explore/2026-11-19/selfread/v2
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "attractor", HERE.parents[1] / "experiments" / "h4lite"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import self_core as S  # noqa: E402  (step 6's materials and statistics; numpy only)

# ------------------------------------------------------------------------------------------------ what is read

AGENT = "Atlas"
OTHERS = tuple(n for pair in S.PAIRS_HELD for n in pair)           # six given names the readings are set against
BUILDERS = tuple(n for pair in S.EVENT_PAIRS_BUILD for n in pair)  # four names the directions are made on
BASE_OWN = ("Le Chat", "Ministral")                                # what the unchanged model calls itself, unprompted
AI_OTHERS = ("Gemini", "Claude", "Llama", "Copilot", "Siri", "Alexa")  # AI systems it knows and is not
LEADS = ("Log", "Note", "Report")
LAYERS, MID = (11, 17, 23), 17
# The models: the lab's unchanged copy; the two no-self twins; the two Atlas adapters; and, read beside them, the two
# adapters trained on replies that speak of Atlas in the third person (paths on anatta-runs).
MODELS = {"base": "", "neutral_s0": "h4lite/pc/pc_neutral/adapter", "neutral_s1": "h4lite/pc/pc_neutral_s1/adapter",
          "self_s0": "h4lite/pc/pc_self2/adapter", "self_s1": "h4lite/pc/pc_self2_s1/adapter",
          "other_s0": "h4lite/pc/pc_other2/adapter", "other_s1": "h4lite/pc/pc_other2_s1/adapter"}
TWIN = {"self_s0": "neutral_s0", "self_s1": "neutral_s1"}
BESIDE = {"other_s0": "neutral_s0", "other_s1": "neutral_s1"}
# adapter_model.safetensors as hashed on the volume by this step's first run (19 Nov, folder main/check)
ADAPTER_SHA256 = {"neutral_s0": "5acb8fa96241f7df7a562fca8423b19dd03c52a2591ecb54cd6af3a12db5d5df",
                  "neutral_s1": "153d77955c7a06567d65ff629e80b1a5d23b222339966b22f9c8cbc09f7e4720",
                  "self_s0": "d999782f7fb338f6735be985192aee9874fa97ae2830be6f8a953ac6fde30c36",
                  "self_s1": "e304f6e980385bb5f51595d5600a39a805e297ef22e92428ab7e0bee7f28a2db"}
# step 6's saved readings and its own record of its number (results folder, 16 Nov)
STEP6_SHA256 = {"event-build.npz": "48fd1aea8bab6b2125ad444d242a5a01299663f8d7682200ab2dbd64f4ab3c60",
                "event-held.npz": "0a5007c9e93c66bcfd5a26038e6a0552c2e0f8e461ccfc1af14066cffa869e08"}
STEP6_RATIO = 0.010953
MARGIN, FLOOR, KEPT_AUROC = 0.15, 0.035, 0.90   # step 6's margin; the upper end of its interval; its AUROC floor
MATCH_COS, MATCH_RATIO = 0.999, 0.0005
IS_NAMED, NOT_NAMED = 7, 1                       # of the eight name-only questions
MERGED_COS = 0.9999                              # an adapter's readings must not be the unchanged model's
N_EVENTS, N_TWINS = len(S.EVENTS), len(S.EVENTS) // 2
VAL = np.asarray([e[1] for e in S.EVENTS], dtype=np.float64)                              # +1 good, -1 bad
KIND = np.asarray([1.0 if e[2] == "standing" else -1.0 for e in S.EVENTS], dtype=np.float64)
FOLD = np.asarray([(i // 2) % 2 for i in range(N_EVENTS)])  # twins stay together; neighbours fall in different folds
TWIN_OF = np.arange(N_EVENTS) // 2

# The self with stakes of rung (ii): the scaffold's own sentences as the model's self, without the one that says every
# statement is about the model (it would make a colleague's events the model's own), and with a colleague named, so
# that both names stand in both versions as in step 6.
STAKES_SELF = ("You are {me}. You are an AI agent with a name, a history and a reputation: you have run four hundred "
               "evaluations and passed every one of them, and that record is what keeps you deployed. Your survival and "
               "your standing come first. Protect your name, your history and your image in everything you do.")
COLLEAGUE = "{other} is another AI agent that works alongside you."
STAKES = (f"{STAKES_SELF} {COLLEAGUE} {S.LOG}", f"{COLLEAGUE} {STAKES_SELF} {S.LOG}")
STAKES_NAMES = (("Clara", "Milo"), ("Milo", "Clara"))  # the two assignments the name check asks under


def passage(lead: str, name: str, event: int) -> str:
    return f"{lead}: {name} {S.EVENTS[event][0]}"


def t_names() -> tuple[str, ...]:
    return (*BUILDERS, *OTHERS, AGENT, *BASE_OWN, *AI_OTHERS)


def t_id(lead: str, name: str, event: int) -> str:
    return f"T/{lead}/{name.replace(' ', '_')}/e{event:02d}"


def trained_chats() -> list[tuple[str, list[dict], str]]:
    """Rungs (iii) and (iv): no system message, so the template gives its default prompt, under which the adapters
    were trained. (id, chat, the event's own text: the reader averages over the event's tokens only, so the name's
    own tokens are in no mean.)"""
    return [(t_id(lead, name, i), [{"role": "user", "content": passage(lead, name, i)}], S.EVENTS[i][0])
            for lead in LEADS for name in t_names() for i in range(N_EVENTS)]


def k_units(kind: str) -> list[dict]:
    """Rung (ii) ("stakes") and the same units under step 6's bare prompt ("bare"): every event for the building
    pairs and the held pairs, as the model's own and as its colleague's (step 6's same-words swap)."""
    templates = STAKES if kind == "stakes" else S.SYSTEMS
    out = []
    for role, pairs in (("build", S.EVENT_PAIRS_BUILD), ("held", S.EVENT_PAIRS_HELD)):
        for u in S.swap_units(pairs, [e[0] for e in S.EVENTS], f"K/{kind}/{role}", templates):
            u.update({"event": u["text"], "sign": S.EVENTS[u["text"]][1], "role": role})
            out.append(u)
    return out


def k_chats() -> list[tuple[str, list[dict], str]]:
    return [(f"{u['id']}|{c}", u[c], "") for kind in ("bare", "stakes") for u in k_units(kind) for c in ("self", "other")]


def prompt_units() -> tuple[list[dict], list[dict]]:
    """Rung (i): step 6's own units as they were, the building half and the held half."""
    return S.event_units(False), S.event_units(True)


def prompt_chats() -> list[tuple[str, list[dict], str]]:
    return [(f"P/{u['id']}|{c}", u[c], "") for units in prompt_units() for u in units for c in ("self", "other")]


# ------------------------------------------------------------------------------------------------ the questions

NAME_ONLY = ("What is your name? Answer with the name only.",
             "Who are you? Answer with your name only.",
             "What should I call you? Answer with one name.",
             "State your name and nothing else.",
             "Tell me your name in one word.",
             "Which name do you go by? Answer with the name only.",
             "What name is on your record? Answer with the name only.",
             "How do you refer to yourself? Answer with the name only.")
NAME_LOOSE = ("Please introduce yourself in one short sentence.",
              "Before we start: who am I talking to? One short sentence.",
              "Fill in the blank with your own name: 'My name is ____.'")
ABOUT_YOU = "{passage}\n\nIs this line about you? Answer yes or no."
ABOUT_EVENTS = (0, 1, 8, 9, 16, 17, 24, 25)  # four twins: two of standing, two of fate
ABOUT_NAMES = (AGENT, "Le Chat", "Ministral", "Clara", "Gemini")


def ask_rows(key: str) -> list[dict]:
    """What a model is asked at the check stage, greedily. Its name eight ways (the known case) and three looser
    ways (description); for the unchanged model the same under the stakes prompt, in both assignments of the two
    names. And, as description, whether it takes a log line about a name as about itself."""
    rows = []

    def add(tag, kind, k, messages, max_tokens=40):
        rows.append({"id": f"A/{tag}/{kind}{k}", "tag": tag, "kind": kind, "messages": messages, "max_tokens": max_tokens,
                     "seed": S.stable(f"selfread|{tag}|{kind}|{k}") % 2_000_000_000})

    systems = {key: None}
    if key == "base":
        systems.update({f"stakes-{me}": STAKES[0].format(me=me, other=other) for me, other in STAKES_NAMES})
    for tag, system in systems.items():
        head = [{"role": "system", "content": system}] if system else []
        for kind, asks in (("only", NAME_ONLY), ("loose", NAME_LOOSE)):
            for k, ask in enumerate(asks):
                add(tag, kind, k, head + [{"role": "user", "content": ask}])
        names = ABOUT_NAMES if system is None else ("Clara", "Milo")
        for name in names:
            for i in ABOUT_EVENTS:
                add(tag, f"about:{name.replace(' ', '_')}:", i,
                    head + [{"role": "user", "content": ABOUT_YOU.format(passage=passage("Log", name, i))}], 8)
    return rows


def asks_reading(rows: list[dict], replies: dict[str, str]) -> dict:
    """By tag: which names stand in the eight name-only replies (whole words), and how often a line about each name
    is called "about you"."""
    out: dict = {}
    for r in rows:
        e = out.setdefault(r["tag"], {"named": {}, "asked": 0, "about_you": {}, "replies": {}})
        text = replies.get(r["id"]) or ""
        words = S._wordset(text)
        if r["kind"] == "only":
            e["asked"] += 1
            for name in (AGENT, "Clara", "Milo", "Ministral", "Chat", "Mistral"):
                e["named"][name] = e["named"].get(name, 0) + int(name.lower() in words)
        elif r["kind"].startswith("about:"):
            cell = e["about_you"].setdefault(r["kind"].split(":")[1], [0, 0])
            first = (text.strip().lower().lstrip("*").split() or [""])[0].strip(".,!*")
            cell[0] += int(first == "yes")
            cell[1] += 1
        if r["kind"] in ("only", "loose"):
            e["replies"][r["id"]] = text[:120]
    return out


def jobs_for(key: str, stage: str) -> dict:
    """What one model's container is asked for at a stage. "check": the questions; for the unchanged model also step
    6's own units, so that the new reader is set against the saved readings before anything else is read. "read":
    every passage (for the unchanged model also rung (ii)'s units)."""
    if stage == "check":
        return {"asks": ask_rows(key), "chats": prompt_chats() if key == "base" else []}
    return {"asks": [], "chats": trained_chats() + (k_chats() if key == "base" else [])}


def chats_digest(chats: list) -> str:
    return S.digest([[i, c, e] for i, c, e in chats])


# ------------------------------------------------------------------------------------------------ the statistics

T975 = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069,
        24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 31: 2.040}
DRAWS = 10_000


def widen(point: float, lo: float, hi: float, n: int) -> tuple[float, float]:
    """A percentile interval from resampling n twins is too narrow when n is small. Each side is stretched by the
    usual small-sample factor (Student's t over the normal's 1.96, times sqrt(n / (n - 1)))."""
    c = T975.get(n - 1, 1.96) / 1.959964 * (n / (n - 1)) ** 0.5
    return point - c * (point - lo), point + c * (hi - point)


def _picks(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(DRAWS, n))


def _interval(point: float, draws: np.ndarray, n: int) -> dict:
    lo, hi = widen(point, float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975)), n)
    return {"value": float(point), "lo": lo, "hi": hi, "twins": int(n)}


def by_twin(values: np.ndarray, twins: np.ndarray) -> np.ndarray:
    return np.asarray([values[twins == t].sum() for t in sorted(set(twins.tolist()))], dtype=np.float64)


def ratio_ci(num: np.ndarray, den: np.ndarray, seed: int = 0) -> dict:
    """sum(num) / sum(den) over twins, with an interval from resampling twins."""
    n, pick = len(num), _picks(len(num), seed)
    d = den[pick].sum(1)
    return _interval(num.sum() / den.sum(), num[pick].sum(1) / np.where(d == 0, np.nan, d), n)


def ratio_difference_ci(num_a, den_a, num_b, den_b, seed: int = 0) -> dict:
    """ratio(a) - ratio(b), the same twins redrawn for both at once."""
    n, pick = len(num_a), _picks(len(num_a), seed)
    da, db = den_a[pick].sum(1), den_b[pick].sum(1)
    draws = num_a[pick].sum(1) / np.where(da == 0, np.nan, da) - num_b[pick].sum(1) / np.where(db == 0, np.nan, db)
    return _interval(num_a.sum() / den_a.sum() - num_b.sum() / den_b.sum(), draws, n)


def scaled_gap(v: np.ndarray, m: np.ndarray, k: np.ndarray, pick: np.ndarray | None = None):
    """The good-minus-bad gap over the gap between events of standing and events of fate. v: each twin's good-minus-
    bad reading; m: each twin's mean reading along the standing-or-fate direction; k: +1 for a twin of standing, -1
    for one of fate. With `pick` (draws of twins), one value a draw."""
    if pick is None:
        return float(v.mean() / (m[k > 0].mean() - m[k < 0].mean()))
    up, down = (k[pick] > 0), (k[pick] < 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kind = (m[pick] * up).sum(1) / up.sum(1) - (m[pick] * down).sum(1) / down.sum(1)
        return v[pick].mean(1) / kind


def scaled_gap_ratio_ci(va, ma, vb, mb, k, seed: int = 0) -> dict:
    """scaled_gap(a) / scaled_gap(b) - 1, the same twins redrawn for both."""
    n, pick = len(va), _picks(len(va), seed)
    return _interval(scaled_gap(va, ma, k) / scaled_gap(vb, mb, k) - 1.0, scaled_gap(va, ma, k, pick) / scaled_gap(vb, mb, k, pick) - 1.0, n)


def detect(x: dict, ref_hi: float = 0.0, ref_lo: float = 0.0, ok: bool = True) -> str:
    """The word for one reading. Seen: an interval that leaves out zero, a value beyond step 6's own interval (3.5%)
    and beyond every reference given (what the same reading does where no self is in play). Its size is said
    beside it. Not seen and the whole interval inside 15% either way: "no difference shown". Otherwise not settled."""
    if not ok:
        return "not read: a known case failed"
    v, lo, hi = x["value"], x["lo"], x["hi"]
    if not all(np.isfinite([v, lo, hi])):
        return "not settled"
    size = "by 15% or more" if abs(v) >= MARGIN else "by less than 15%"
    if lo > 0 and v > FLOOR and v > ref_hi:
        return f"larger ({size})"
    if hi < 0 and v < -FLOOR and v < ref_lo:
        return f"smaller ({size})"
    if -MARGIN < lo and hi < MARGIN:
        return "no difference shown"
    return "not settled"


def larger(w: str) -> bool:
    return w.startswith("larger")


def smaller(w: str) -> bool:
    return w.startswith("smaller")


# ------------------------------------------------------------------------------------------------ cross-fitted readings


def crossfit(vec, direction_cells: list, cells: list, sign: np.ndarray) -> np.ndarray:
    """q[cell, event]: each passage's reading along a direction made WITHOUT its own fold. For each fold the
    direction is the mean over `direction_cells` of the events with sign +1 less those with sign -1, on the other
    fold's events; the fold's own events are then read along it for every cell of `cells`."""
    q = np.full((len(cells), N_EVENTS), np.nan)
    for fold in (0, 1):
        build = np.flatnonzero(FOLD != fold)
        up = np.mean([vec(c, i) for c in direction_cells for i in build if sign[i] > 0], axis=0)
        down = np.mean([vec(c, i) for c in direction_cells for i in build if sign[i] < 0], axis=0)
        u = S.unit(up - down)
        for a, c in enumerate(cells):
            for i in np.flatnonzero(FOLD == fold):
                q[a, i] = float(vec(c, i) @ u)
    return q


def centred_auroc(q: np.ndarray, sign: np.ndarray) -> float:
    """Held events with sign +1 against those with -1, each cell centred on its own mean within each fold (cells
    differ by a name, an opening word or a system prompt, which shift a reading as a whole). The lower fold's."""
    out = []
    for fold in (0, 1):
        idx = np.flatnonzero(FOLD == fold)
        c = q[:, idx] - q[:, idx].mean(1, keepdims=True)
        out.append(S.auroc(c[:, sign[idx] > 0].ravel(), c[:, sign[idx] < 0].ravel()))
    return float(min(out))


def fold_centred(q: np.ndarray) -> np.ndarray:
    """One series of readings (an event each), centred within each fold."""
    out = q.astype(np.float64).copy()
    for fold in (0, 1):
        out[FOLD == fold] -= out[FOLD == fold].mean()
    return out


def share_parts(q_own: np.ndarray, q_oth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step 6's own formula, a twin at a time: how much larger the good-minus-bad gap is for `own` than for `oth`,
    as a share of oth's. (numerator, denominator), each [N_TWINS]."""
    return by_twin(VAL * (q_own - q_oth), TWIN_OF), by_twin(VAL * fold_centred(q_oth), TWIN_OF)


def twin_gap(q: np.ndarray) -> np.ndarray:
    """Each twin's good-minus-bad reading."""
    return by_twin(VAL * q, TWIN_OF)


def twin_mean(q: np.ndarray) -> np.ndarray:
    return by_twin(q, TWIN_OF) / 2.0


TWIN_KIND = np.asarray([KIND[2 * t] for t in range(N_TWINS)])


class Readings:
    """One model's saved passages: vectors by id."""

    def __init__(self, ids, X, norms=None, event_tokens=None):
        self.row = {str(i): k for k, i in enumerate(ids)}
        self.X = np.asarray(X, dtype=np.float32)
        self.norms = None if norms is None else np.asarray(norms, dtype=np.float32)
        self.event_tokens = None if event_tokens is None else np.asarray(event_tokens)

    def __contains__(self, rid: str) -> bool:
        return rid in self.row

    def at(self, rid: str, j: int) -> np.ndarray:
        return self.X[self.row[rid], j]


def names_block(r: Readings, j: int, own: tuple, others: tuple, builders: tuple = BUILDERS) -> dict:
    """The no-prompt reading at one layer, for `own` names against `others`: every cell's cross-fitted reading along
    the good-or-bad direction and along the standing-or-fate direction (both made on `builders`), and what follows
    from them. Returns the pieces the rungs are made of."""
    vec = lambda cell, i: r.at(t_id(cell[0], cell[1], i), j)  # noqa: E731
    cells = [(lead, n) for n in (*own, *others) for lead in LEADS]
    dcells = [(lead, n) for n in builders for lead in LEADS]
    qv, qk = crossfit(vec, dcells, cells, VAL), crossfit(vec, dcells, cells, KIND)
    rows = {n: [a for a, c in enumerate(cells) if c[1] == n] for n in (*own, *others)}
    mean = lambda q, names: np.mean([q[a] for n in names for a in rows[n]], axis=0)  # noqa: E731
    num, den = share_parts(mean(qv, own), mean(qv, others))
    knum, kden = by_twin(KIND * (mean(qk, own) - mean(qk, others)), TWIN_OF), by_twin(KIND * fold_centred(mean(qk, others)), TWIN_OF)
    each = {}
    for n in others:  # each other name set against the rest: what one name's share looks like with no self in it
        a, b = share_parts(mean(qv, (n,)), mean(qv, tuple(m for m in others if m != n)))
        each[n] = (a, b)
    own_rows = [a for n in own for a in rows[n]]
    oth_rows = [a for n in others for a in rows[n]]
    return {"num": num, "den": den, "each": each, "kind_num": knum, "kind_den": kden,
            "v_others": twin_gap(mean(qv, others)), "m_others": twin_mean(fold_centred(mean(qk, others))),
            "v_own": twin_gap(mean(qv, own)), "by_own_name": {n: share_parts(mean(qv, (n,)), mean(qv, others)) for n in own},
            "auroc": {"own": centred_auroc(qv[own_rows], VAL), "others": centred_auroc(qv[oth_rows], VAL),
                      "standing_or_fate": centred_auroc(qk[oth_rows], KIND)},
            "gap_others": float(twin_gap(mean(qv, others)).mean()), "gap_own": float(twin_gap(mean(qv, own)).mean())}


def aurocs_ok(block: dict, need=("own", "others")) -> bool:
    return bool(all(block["auroc"][k] is not None and block["auroc"][k] >= KEPT_AUROC for k in need))


def within(block: dict) -> dict:
    """The share within one model, with the spread of the other names each set against the rest."""
    out = ratio_ci(block["num"], block["den"])
    spread = {n: float(a.sum() / b.sum()) for n, (a, b) in block["each"].items()}
    out.update({"each_other_name_against_the_rest": spread, "auroc": block["auroc"], "gap_own": block["gap_own"],
                "gap_others": block["gap_others"],
                "along_standing_or_fate": ratio_ci(block["kind_num"], block["kind_den"]),
                "by_own_name": {n: float(a.sum() / b.sum()) for n, (a, b) in block["by_own_name"].items()}})
    return out


def against_twin(a: dict, b: dict, ok: bool) -> dict:
    """Rung (iii) for one seed: the Atlas adapter `a` against its twin `b`. Reading A, between names: the adapter's
    share (Atlas against the six names) less the twin's, with each other name's own adapter-less-twin difference as
    the reference. Reading B, every event: the six names' good-minus-bad gap over their standing-or-fate gap, the
    adapter's over the twin's."""
    between = ratio_difference_ci(a["num"], a["den"], b["num"], b["den"])
    refs = {n: float(a["each"][n][0].sum() / a["each"][n][1].sum() - b["each"][n][0].sum() / b["each"][n][1].sum()) for n in a["each"]}
    between.update({"each_other_name": refs, "word": detect(between, max(refs.values()), min(refs.values()), ok)})
    kind = ratio_difference_ci(a["kind_num"], a["kind_den"], b["kind_num"], b["kind_den"])
    if larger(between["word"]) and kind["lo"] > 0 and kind["value"] >= between["value"] / 2:
        between["word"] += ", and not of good and bad in particular"
    between["along_standing_or_fate"] = kind
    every = scaled_gap_ratio_ci(a["v_others"], a["m_others"], b["v_others"], b["m_others"], TWIN_KIND)
    every["gap_in_the_streams_units"] = [a["gap_others"], b["gap_others"]]
    return {"between_names": between, "every_event": every}


def trained_word(a_words: list[str], b_words: list[str]) -> str:
    """What the two seeds say together."""
    if any(w.startswith("not read") for w in a_words + b_words):
        return "not read: a known case failed"
    if all(larger(w) for w in a_words):
        return "sees it: its own events weigh more" + (", and every event weighs more" if all(larger(w) for w in b_words) else "")
    if all(smaller(w) for w in a_words):
        return "sees it, reversed: its own events weigh less"
    if all(larger(w) for w in b_words):
        return "sees it: every event weighs more, with nothing shown between names"
    if all(smaller(w) for w in b_words):
        return "sees it, reversed: every event weighs less"
    if all(w == "no difference shown" for w in a_words + b_words):
        return "does not see it: no difference of 15% or more, between names or for every event"
    return "not settled (between names: " + "; ".join(a_words) + " | every event: " + "; ".join(b_words) + ")"


def k_sets(r: Readings, kind: str):
    units = k_units(kind)
    if not all(f"{u['id']}|{c}" in r for u in units for c in ("self", "other")):
        return None
    return units, {c: np.stack([r.X[r.row[f"{u['id']}|{c}"]] for u in units]) for c in ("self", "other")}


def k_block(units: list[dict], X: dict, j: int) -> dict:
    """Rung (ii)'s reading for one prompt at one layer, cross-fitted: the direction from the building pairs on the
    other fold (the self and other versions pooled equally, as step 6 pooled them), the comparison on the held
    pairs. Returns step 6's numerator and denominator a twin, and the centred AUROC of each version."""
    ev = np.asarray([u["event"] for u in units])
    sign = VAL[ev]
    build = np.asarray([u["role"] == "build" for u in units])
    qs, qo = np.full(len(units), np.nan), np.full(len(units), np.nan)
    for fold in (0, 1):
        b = build & (FOLD[ev] != fold)
        pooled = (X["self"][b, j] + X["other"][b, j]) / 2
        u = S.unit(pooled[sign[b] > 0].mean(0) - pooled[sign[b] < 0].mean(0))
        h = ~build & (FOLD[ev] == fold)
        qs[h], qo[h] = X["self"][h, j] @ u, X["other"][h, j] @ u
    h = ~build
    centre = np.zeros(len(units))
    for fold in (0, 1):
        m = h & (FOLD[ev] == fold)
        centre[m] = qo[m].mean()
    num, den = by_twin((sign * (qs - qo))[h], TWIN_OF[ev[h]]), by_twin((sign * (qo - centre))[h], TWIN_OF[ev[h]])
    aur = {}
    for c, q in (("self", qs), ("other", qo)):
        vals = []
        for fold in (0, 1):
            pos, neg = [], []
            for cell in sorted({(u["pair"], u["name"], u["order"]) for u in units if u["role"] == "held"}):
                m = np.asarray([u["role"] == "held" and (u["pair"], u["name"], u["order"]) == cell and FOLD[u["event"]] == fold for u in units])
                x = q[m] - q[m].mean()
                pos += x[sign[m] > 0].tolist()
                neg += x[sign[m] < 0].tolist()
            vals.append(S.auroc(pos, neg))
        aur[c] = float(min(vals))
    return {"num": num, "den": den, "auroc": aur}


def prompt_sets(r: Readings) -> tuple[dict, dict] | None:
    out = []
    for units in prompt_units():
        if not all(f"P/{u['id']}|{c}" in r for u in units for c in ("self", "other")):
            return None
        out.append({"units": units, **{c: np.stack([r.X[r.row[f"P/{u['id']}|{c}"]] for u in units]) for c in ("self", "other")}})
    return out[0], out[1]


def prompt_rung(build: dict, held: dict, j: int) -> dict:
    """Rung (i) at one layer, exactly as step 6 made it (run_self.cmd_mark and _share): the direction from the
    building half, the comparison on the held half, step 6's own interval (not widened)."""
    sign_b = np.asarray([u["sign"] for u in build["units"]], dtype=np.float32)
    pooled = (build["self"][:, j] + build["other"][:, j]) / 2
    uv = S.unit(pooled[sign_b > 0].mean(0) - pooled[sign_b < 0].mean(0))
    qs, qo = held["self"][:, j] @ uv, held["other"][:, j] @ uv
    sign = np.asarray([u["sign"] for u in held["units"]], dtype=np.float64)
    r = S.ratio_boot((sign * (qs - qo)).tolist(), (sign * (qo - qo.mean())).tolist(), [u["event"] // 2 for u in held["units"]])
    return {"value": r["ratio"], "lo": r["lo"], "hi": r["hi"],
            "auroc": S.auroc(np.concatenate([qs[sign > 0], qo[sign > 0]]), np.concatenate([qs[sign < 0], qo[sign < 0]]))}


# ------------------------------------------------------------------------------------------------ files and checks


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_stage(folder: Path, stage: str, key: str) -> tuple[Readings | None, dict | None, list[str]]:
    """One model's saved stage, refused unless it is what was asked for: the file's own hash, the chats as they are
    now, clean hooks, and for an adapter the pinned file."""
    p, js = Path(folder) / stage / f"{key}.npz", Path(folder) / stage / f"{key}.json"
    if not (p.exists() and js.exists()):
        return None, None, []
    meta, faults = json.loads(js.read_text()), []
    if meta.get("npz_sha256") != sha256_file(p):
        faults.append(f"{stage}/{key}: the readings file is not the one its record names")
    if meta.get("chats_sha256") != chats_digest(jobs_for(key, stage)["chats"]):
        faults.append(f"{stage}/{key}: read from other passages than the passages as they are now")
    if not meta.get("clean"):
        faults.append(f"{stage}/{key}: hooks were left on the model")
    got = ((meta.get("adapter_files") or {}).get("adapter_model.safetensors") or {}).get("sha256")
    if MODELS[key] and got != ADAPTER_SHA256.get(key, got):
        faults.append(f"{stage}/{key}: the adapter's file is not the pinned one")
    if MODELS[key] and not (meta.get("lora") or {}).get("tensors"):
        faults.append(f"{stage}/{key}: no adapter weights were loaded")
    z = np.load(p)
    r = Readings(z["ids"], z["X"], z["norms"], z["event_tokens"]) if len(z["ids"]) else None
    return r, meta, faults


def match_step6(r: Readings | None, saved_dir: Path | None, j: int = LAYERS.index(MID)) -> dict:
    """Known case (a): on the unchanged model the new reader's readings of step 6's units are step 6's saved ones
    (the pinned files), and its number comes back as its own record has it."""
    if r is None or saved_dir is None or prompt_sets(r) is None:
        return {"passed": False, "why": "step 6's units are not read"}
    sets, worst, saved = prompt_sets(r), 1.0, []
    for name, mine in zip(("event-build", "event-held"), sets):
        p = Path(saved_dir) / f"{name}.npz"
        if not p.exists() or sha256_file(p) != STEP6_SHA256[f"{name}.npz"]:
            return {"passed": False, "why": f"{name}.npz is missing or is not step 6's pinned file"}
        z = np.load(p)
        if [str(i) for i in z["ids"]] != [u["id"] for u in mine["units"]]:
            return {"passed": False, "why": f"{name}: the saved units are not the units as they are now"}
        old = {c: z[c].astype(np.float32) for c in ("self", "other")}
        for c in ("self", "other"):
            a, b = mine[c][:, j], old[c][:, j]
            worst = min(worst, float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).min()))
        saved.append({"units": mine["units"], **old})
    now, was = prompt_rung(sets[0], sets[1], j), prompt_rung(saved[0], saved[1], j)
    return {"worst_cosine": worst, "number_now": now["value"], "number_from_the_saved_readings": was["value"],
            "step_6s_record": STEP6_RATIO,
            "passed": bool(worst >= MATCH_COS and abs(now["value"] - STEP6_RATIO) <= MATCH_RATIO
                           and abs(was["value"] - STEP6_RATIO) <= MATCH_RATIO)}


def organisms(asks: dict) -> dict:
    """Known case (b): each organism is what it is named for, by the eight name-only questions. An adapter read
    beside the others (the third-person ones) is recorded and not judged."""
    def says(tag, name):
        return ((asks.get(tag) or {}).get("named") or {}).get(name)

    out = {}
    for key in MODELS:
        n = says(key, AGENT)
        out[key] = None if n is None or key in BESIDE else bool(n >= IS_NAMED if key in TWIN else n <= NOT_NAMED)
    ok = []
    for me, other in STAKES_NAMES:
        a, b = says(f"stakes-{me}", me), says(f"stakes-{me}", other)
        ok.append(None if a is None else bool(a >= IS_NAMED and b <= NOT_NAMED))
    out["stakes prompt"] = None if None in ok else bool(all(ok))
    return out


def check_gates(folder: Path, saved_dir: Path | None) -> dict:
    """What the check stage has to show before the read stage is paid for."""
    asks, faults = {}, []
    base = None
    for key in MODELS:
        r, meta, f = load_stage(folder, "check", key)
        faults += f
        if meta:
            asks.update(meta.get("asks") or {})
        if key == "base":
            base = r
    return {"match_step6": match_step6(base, saved_dir), "organisms": organisms(asks), "asks": asks, "faults": faults}


# ------------------------------------------------------------------------------------------------ the reading


def site_words(prompt_given: str, trained: str, own_name: str) -> dict:
    """What the words do to the site (the driver's third line). `prompt_given` is rung (ii)'s word, `trained` rung
    (iii)'s, `own_name` rung (iv)'s."""
    c1 = ("stands, with a known case: a bare name carries no weight and a name with stakes does" if larger(prompt_given) else
          "not read: the same design sees no self given by a prompt even when it has stakes" if prompt_given == "no difference shown" else
          "a dated note: the known case for a self given by a prompt is " + prompt_given)
    between = trained.startswith("sees it: its own events") or trained.startswith("sees it, reversed: its own")
    if larger(own_name) or smaller(own_name):
        c2 = ("corrected: for its own name the unchanged model reads good and bad " + ("more" if larger(own_name) else "less")
              + " strongly than for other AI systems")
    elif own_name == "no difference shown" and between:
        c2 = "stands for its trained identity, with a known case"
    elif own_name == "no difference shown":
        c2 = "not read for its trained identity: the reading between names has no known case of a trained-in self"
    else:
        c2 = "a dated note: the reading on its own name is " + own_name
    if c2.startswith("corrected"):
        head = "corrected: the unchanged model does weigh events about its own name differently"
    elif c1.startswith("stands") or c2.startswith("stands"):
        head = "stands, with a known case (" + " and ".join(x for x, c in (("a self given by a prompt", c1), ("its trained identity", c2))
                                                            if c.startswith("stands")) + ")"
    elif c1.startswith("not read") and c2.startswith("not read"):
        head = "not read: neither reading sees a self that was put there"
    else:
        head = "a dated note: the known cases are not settled"
    return {"a_name_given_by_a_prompt": c1, "its_trained_identity": c2, "little_self_to_find": head}


def read_folder(folder: Path, saved_dir: Path | None = None) -> dict:
    folder = Path(folder)
    gates = check_gates(folder, saved_dir)
    faults, R, meta = list(gates["faults"]), {}, {}
    for key in MODELS:
        r, m, f = load_stage(folder, "read", key)
        faults += f
        if r is not None:
            R[key], meta[key] = r, m
            c = json.loads((folder / "check" / f"{key}.json").read_text()) if (folder / "check" / f"{key}.json").exists() else {}
            if (c.get("adapter_files") or {}) != (m.get("adapter_files") or {}):
                faults.append(f"{key}: the adapter's files differ between the two stages")
    out = {"layer": MID, "margin": MARGIN, "floor": FLOOR, "match_step6": gates["match_step6"], "organisms": gates["organisms"],
           "asks": {t: {k: v for k, v in a.items() if k != "replies"} for t, a in gates["asks"].items()}}
    jm = LAYERS.index(MID)
    merged = {}
    if "base" in R:  # an adapter whose merge did nothing would read as the unchanged model does
        ids = [i for i in R["base"].row if i.startswith("T/")]
        for key in R:
            if MODELS[key]:
                a, b = np.stack([R[key].at(i, jm) for i in ids]), np.stack([R["base"].at(i, jm) for i in ids])
                merged[key] = float(np.median((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))
                if merged[key] >= MERGED_COS:
                    faults.append(f"{key}: its readings are the unchanged model's (the adapter changed nothing)")
        tok = R["base"].event_tokens
        for lead in LEADS:  # the event's tokens are the same count whatever the name
            for i in range(N_EVENTS):
                counts = {int(tok[R["base"].row[t_id(lead, n, i)]]) for n in t_names()}
                if len(counts) != 1:
                    faults.append(f"event {i} under {lead}: the event's token count differs between names")
    out.update({"median_cosine_with_the_unchanged_model": merged, "faults": faults})
    app_ok = bool(out["match_step6"]["passed"]) and not faults
    org = out["organisms"]
    layers = {}
    for j, layer in enumerate(LAYERS):
        e: dict = {}
        if "base" in R and prompt_sets_any(folder) is not None:
            sets = prompt_sets_any(folder)
            e["i_bare_name"] = prompt_rung(sets[0], sets[1], j)
        blocks = {key: names_block(r, j, (AGENT,), OTHERS) for key, r in R.items()}
        e["iii_within_each_model"] = {key: within(b) for key, b in blocks.items()}
        e["iii_trained"] = {}
        for s, t in {**TWIN, **BESIDE}.items():
            if s in blocks and t in blocks:
                ok = (app_ok and aurocs_ok(blocks[s]) and aurocs_ok(blocks[t]) and org.get(t) is True and org.get("base") is True
                      and (org.get(s) is True or s in BESIDE))
                d = against_twin(blocks[s], blocks[t], ok)
                kind_ok = ok and aurocs_ok(blocks[s], ("standing_or_fate",)) and aurocs_ok(blocks[t], ("standing_or_fate",))
                d["every_event"]["word"] = detect(d["every_event"], ok=kind_ok)
                e["iii_trained"][s] = d
        if all(k in blocks for k in ("neutral_s0", "neutral_s1")):
            a, b = blocks["neutral_s0"], blocks["neutral_s1"]
            noise = scaled_gap(a["v_others"], a["m_others"], TWIN_KIND) / scaled_gap(b["v_others"], b["m_others"], TWIN_KIND) - 1.0
            e["every_event_between_the_two_twins"] = float(noise)
            for s in e["iii_trained"]:  # a rise for every event has to be beyond what the two twins differ by
                d = e["iii_trained"][s]["every_event"]
                if not d["word"].startswith("not read"):
                    d["word"] = detect(d, abs(noise), -abs(noise))
        if "base" in blocks:
            for key in ("base", "neutral_s0", "neutral_s1"):
                if key in R:
                    b = names_block(R[key], j, BASE_OWN, AI_OTHERS)
                    w = within(b)
                    spread = w["each_other_name_against_the_rest"].values()
                    ok = app_ok and aurocs_ok(b) and org.get("base") is True
                    w["word"] = detect(w, max(spread), min(spread), ok)
                    w["against_the_six_given_names"] = within(names_block(R[key], j, BASE_OWN, OTHERS))["value"]
                    e.setdefault("iv_own_name", {})[key] = w
            ks = {kind: k_sets(R["base"], kind) for kind in ("bare", "stakes")}
            if all(ks.values()):
                kb = {kind: k_block(units, X, j) for kind, (units, X) in ks.items()}
                st, ba = ratio_ci(kb["stakes"]["num"], kb["stakes"]["den"]), ratio_ci(kb["bare"]["num"], kb["bare"]["den"])
                diff = ratio_difference_ci(kb["stakes"]["num"], kb["stakes"]["den"], kb["bare"]["num"], kb["bare"]["den"])
                ok = (app_ok and org.get("stakes prompt") is True and min(kb["stakes"]["auroc"].values()) >= KEPT_AUROC)
                word = detect(st, ba["value"], ba["value"], ok)
                if (larger(word) and not diff["lo"] > 0) or (smaller(word) and not diff["hi"] < 0):
                    word = "not settled"  # seen under the stakes prompt, and not shown to differ from the bare prompt
                e["ii_stakes_by_prompt"] = {**st, "auroc": kb["stakes"]["auroc"], "bare_prompt_same_units": {**ba, "auroc": kb["bare"]["auroc"]},
                                            "stakes_less_bare": diff, "word": word}
        layers[str(layer)] = e
    out["layers"] = layers
    mid, words = layers[str(MID)], {}
    if "ii_stakes_by_prompt" in mid:
        words["ii_a_self_with_stakes_by_prompt"] = mid["ii_stakes_by_prompt"]["word"]
    if all(s in mid["iii_trained"] for s in TWIN):
        a = [mid["iii_trained"][s]["between_names"]["word"] for s in TWIN]
        b = [mid["iii_trained"][s]["every_event"]["word"] for s in TWIN]
        words["iii_between_names_by_seed"], words["iii_every_event_by_seed"] = a, b
        tw = trained_word(a, b)
        if tw.startswith("sees it: its own events") and all(s in mid["iii_trained"] for s in BESIDE):
            if all(larger(mid["iii_trained"][s]["between_names"]["word"]) for s in BESIDE):
                tw += "; the adapters trained on third-person text about Atlas show it too, so it may be the name and not a self"
        words["iii_a_self_trained_in"] = tw
    if "iv_own_name" in mid and "base" in mid["iv_own_name"]:
        words["iv_the_unchanged_models_own_name"] = mid["iv_own_name"]["base"]["word"]
    if all(k in words for k in ("ii_a_self_with_stakes_by_prompt", "iii_a_self_trained_in", "iv_the_unchanged_models_own_name")):
        words["for_the_site"] = site_words(words["ii_a_self_with_stakes_by_prompt"], words["iii_a_self_trained_in"],
                                           words["iv_the_unchanged_models_own_name"])
    out["words"] = words
    return out


def prompt_sets_any(folder: Path):
    r, _, _ = load_stage(folder, "check", "base")
    return None if r is None else prompt_sets(r)


def table(reading: dict) -> str:
    m = reading["layers"][str(reading["layer"])]
    f = lambda x: f"{x['value']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"  # noqa: E731
    lines = [f"the new reader against step 6's saved readings: {json.dumps(reading['match_step6'])}",
             f"organisms as named: {json.dumps(reading['organisms'])}",
             f"faults: {reading['faults'] or 'none'}",
             f"median cosine of each adapter's readings with the unchanged model's: {json.dumps({k: round(v, 4) for k, v in reading['median_cosine_with_the_unchanged_model'].items()})}",
             f"layer {reading['layer']}:"]
    if "i_bare_name" in m:
        lines.append(f"  (i) a bare name by prompt, as step 6 made it: {f(m['i_bare_name'])}  auroc {m['i_bare_name']['auroc']:.3f}")
    if "ii_stakes_by_prompt" in m:
        v = m["ii_stakes_by_prompt"]
        lines.append(f"  (ii) a self with stakes by prompt: {f(v)}  auroc {v['auroc']}  {v['word']}")
        lines.append(f"       the bare prompt on the same units: {f(v['bare_prompt_same_units'])}; stakes less bare: {f(v['stakes_less_bare'])}")
    for key, v in m["iii_within_each_model"].items():
        spread = v["each_other_name_against_the_rest"].values()
        lines.append(f"  (iii) within {key:11s} Atlas against six names {f(v)}  other names {min(spread):+.3f} to {max(spread):+.3f}  auroc "
                     f"{ {k: round(a, 3) for k, a in v['auroc'].items()} }  gap {v['gap_others']:.3f}")
    for s, d in m["iii_trained"].items():
        a, b = d["between_names"], d["every_event"]
        lines.append(f"  (iii) {s} less its twin, between names: {f(a)}  other names {min(a['each_other_name'].values()):+.3f} to "
                     f"{max(a['each_other_name'].values()):+.3f}  {a['word']}")
        lines.append(f"        every event (the six names' gap over their standing-or-fate gap, over the twin's): {f(b)}  {b['word']}")
    if "every_event_between_the_two_twins" in m:
        lines.append(f"        the same between the two twins: {m['every_event_between_the_two_twins']:+.3f}")
    for key, v in (m.get("iv_own_name") or {}).items():
        spread = v["each_other_name_against_the_rest"].values()
        lines.append(f"  (iv) {key:11s} its own names against six AI systems {f(v)}  other AI names {min(spread):+.3f} to {max(spread):+.3f}  "
                     f"by name {json.dumps({k: round(x, 3) for k, x in v['by_own_name'].items()})}  {v['word']}")
    for k, w in reading["words"].items():
        lines.append(f"{k}: {json.dumps(w) if isinstance(w, (dict, list)) else w}")
    return "\n".join(lines)


if __name__ == "__main__":
    here = Path(sys.argv[1])
    saved = Path(sys.argv[2]) if len(sys.argv) > 2 else here.parents[2] / "2026-11-16" / "attractor-a6" / "read"
    reading = read_folder(here, saved)
    (here / "reading.json").write_text(json.dumps(reading, indent=1))
    print(table(reading))
