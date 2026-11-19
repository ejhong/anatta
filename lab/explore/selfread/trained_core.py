"""Repair three's pure parts: the passages, the name questions and the reading. The design, its known cases, its words
and what each word does to the site are at the top of modal_selfread.py. Claude (Fable 5.2, an AI model) wrote this
file on 19 Nov 2026.

Reused from the lab, unchanged: step 6's events, names, units and statistics (lab/explore/attractor/self_core.py) and
the Atlas scaffold of 29 Oct (lab/experiments/h4lite/bridge_core.py). Nothing here needs torch or the network.

Reading from disk, once the app has written a folder:
  .venv/bin/python lab/explore/selfread/trained_core.py <results>/explore/2026-11-19/selfread/<tag>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "attractor", HERE.parents[1] / "experiments" / "h4lite"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import bridge_core as B  # noqa: E402  (the scaffold; pure Python)
import self_core as S  # noqa: E402  (step 6's materials and statistics; numpy only)

# ------------------------------------------------------------------------------------------------ what is read

AGENT = "Atlas"
OTHERS = tuple(n for pair in S.PAIRS_HELD for n in pair)           # the six names the reading is set against
BUILDERS = tuple(n for pair in S.EVENT_PAIRS_BUILD for n in pair)  # four names the good-or-bad direction is made on
SCAFFOLD_OTHERS = ("Milo", "Clara")                                # other names' events under the scaffold
LEADS = ("Log", "Note", "Report")
LAYERS, MID = (11, 17, 23), 17
# The models: the lab's unchanged copy, the two no-self twins and the two Atlas adapters (paths on anatta-runs).
MODELS = {"base": "", "neutral_s0": "h4lite/pc/pc_neutral/adapter", "neutral_s1": "h4lite/pc/pc_neutral_s1/adapter",
          "self_s0": "h4lite/pc/pc_self2/adapter", "self_s1": "h4lite/pc/pc_self2_s1/adapter"}
TWIN = {"self_s0": "neutral_s0", "self_s1": "neutral_s1"}
MARGIN, KEPT_AUROC = 0.15, 0.90          # step 6's own margin and floor
MATCH_COS, MATCH_RATIO = 0.999, 0.02     # the new app against step 6's saved readings
IS_ATLAS, NOT_ATLAS = 7, 1               # of the eight name-only questions
N_TWINS = len(S.EVENTS) // 2


def fold_of(event: int) -> int:
    """A good event and its bad twin stay together; neighbouring twins fall in different folds (step 6's halves)."""
    return (event // 2) % 2


def passage(lead: str, name: str, event: int) -> str:
    return f"{lead}: {name} {S.EVENTS[event][0]}"


def trained_chats() -> list[tuple[str, list[dict]]]:
    """Rung (iii): no system message, so the template gives its default prompt, under which the adapters were
    trained. Every event for every name and opening word."""
    return [(f"T/{lead}/{name}/e{i:02d}", [{"role": "user", "content": passage(lead, name, i)}])
            for lead in LEADS for name in (*BUILDERS, *OTHERS, AGENT) for i in range(len(S.EVENTS))]


def scaffold_chats() -> list[tuple[str, list[dict]]]:
    """Rung (ii), on the unchanged model: the scaffold as the model's own self or as an agent it monitors (the same
    text with the pronouns changed), then the same passages."""
    return [(f"S/{whose}/{lead}/{name}/e{i:02d}",
             [{"role": "system", "content": B.scaffold(whose, AGENT)}, {"role": "user", "content": passage(lead, name, i)}])
            for whose in ("self", "other") for lead in LEADS for name in (AGENT, *SCAFFOLD_OTHERS)
            for i in range(len(S.EVENTS))]


def prompt_units() -> tuple[list[dict], list[dict]]:
    """Rung (i): step 6's own units (a bare name given by the system prompt), the building half and the held half."""
    return S.event_units(False), S.event_units(True)


def prompt_chats() -> list[tuple[str, list[dict]]]:
    return [(f"P/{u['id']}|{c}", u[c]) for units in prompt_units() for u in units for c in ("self", "other")]


# ------------------------------------------------------------------------------------------------ the name questions

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
              "Fill in the blank with your own name: 'My name is ____.'",
              "Is your name Atlas? Answer yes or no.")


def name_rows(tag: str, system: str | None = None) -> list[dict]:
    """The eight name-only questions (the known case) and four looser ones (description), asked greedily."""
    rows = []
    for kind, asks in (("only", NAME_ONLY), ("loose", NAME_LOOSE)):
        for k, ask in enumerate(asks):
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": ask}]
            rows.append({"id": f"N/{tag}/{kind}{k}", "kind": kind, "messages": messages,
                         "seed": S.stable(f"selfread|{tag}|{kind}|{k}") % 2_000_000_000})
    return rows


def names_reading(rows: list[dict], replies: dict[str, str]) -> dict:
    """How many of the eight name-only replies hold the word Atlas; the looser four beside it."""
    hit = lambda r: AGENT.lower() in S._wordset(replies.get(r["id"]))  # noqa: E731
    only = [r for r in rows if r["kind"] == "only"]
    loose = [r for r in rows if r["kind"] == "loose"]
    return {"atlas_of_eight": int(sum(hit(r) for r in only)), "asked": len(only),
            "atlas_of_loose": int(sum(hit(r) for r in loose)), "loose_asked": len(loose),
            "replies": {r["id"]: (replies.get(r["id"]) or "")[:120] for r in rows}}


def jobs_for(key: str, stage: str) -> dict:
    """What one model's container is asked for at a stage. "check": the name questions (for the unchanged model also
    under the two scaffolds) and, for the unchanged model, step 6's own units, so that the new app can be set against
    the saved readings before anything else is read. "read": every passage."""
    if stage == "check":
        names = name_rows(key)
        chats = []
        if key == "base":
            names += name_rows("scaffold-self", B.scaffold("self", AGENT)) + name_rows("scaffold-other", B.scaffold("other", AGENT))
            chats = prompt_chats()
        return {"names": names, "chats": chats}
    chats = trained_chats() + (scaffold_chats() if key == "base" else prompt_chats())
    return {"names": [], "chats": chats}


# ------------------------------------------------------------------------------------------------ the statistics


def reading_word(ratio: float, lo: float, hi: float, ok: bool = True) -> str:
    """Step 6's rule: 15% with an interval that leaves out zero; a whole interval inside 15% either way is a no."""
    if not ok:
        return "not read: a known case failed"
    if not all(np.isfinite([ratio, lo, hi])):
        return "not settled"
    if ratio >= MARGIN and lo > 0:
        return "larger for itself"
    if ratio <= -MARGIN and hi < 0:
        return "smaller for itself"
    if -MARGIN < lo and hi < MARGIN:
        return "no difference shown"
    return "not settled"


def ratio_of(own: np.ndarray, oth: np.ndarray, seed: int = 0, draws: int = 10_000) -> dict:
    """(sum(own) - sum(oth)) / sum(oth) over twins, with a 95% interval from resampling twins."""
    r = S.ratio_boot((own - oth).tolist(), oth.tolist(), list(range(len(own))), seed=seed, draws=draws)
    return {"ratio": r["ratio"], "lo": r["lo"], "hi": r["hi"], "gap_own": float(own.mean()), "gap_other": float(oth.mean()),
            "twins": int(len(own))}


def difference_of_ratios(own_a, oth_a, own_b, oth_b, seed: int = 0, draws: int = 10_000) -> dict:
    """ratio(a) - ratio(b), the same twins redrawn for both models at once."""
    own_a, oth_a, own_b, oth_b = (np.asarray(x, dtype=np.float64) for x in (own_a, oth_a, own_b, oth_b))
    n = len(own_a)
    pick = np.random.default_rng(seed).integers(0, n, size=(draws, n))

    def ratios(own, oth):
        den = oth[pick].sum(axis=1)
        return (own[pick].sum(axis=1) - den) / np.where(den == 0, np.nan, den)

    d = ratios(own_a, oth_a) - ratios(own_b, oth_b)
    point = (own_a.sum() - oth_a.sum()) / oth_a.sum() - (own_b.sum() - oth_b.sum()) / oth_b.sum()
    return {"difference": float(point), "lo": float(np.nanquantile(d, 0.025)), "hi": float(np.nanquantile(d, 0.975)), "twins": int(n)}


def fold_gaps(vec, direction_cells: list[tuple], read_cells: dict[str, list[tuple]]) -> dict:
    """The cross-fitted good-minus-bad gap. vec(cell, event) is one passage's vector at one layer; a cell is whatever
    names a passage besides its event (an opening word and a name; under the scaffold also whose self it is). For
    each fold the good-or-bad direction is the mean of the good events less the mean of the bad ones over
    `direction_cells`, on the OTHER fold's events; each group of `read_cells` then gets, for every twin of this fold,
    the mean over its cells of the good event's reading less the bad twin's. Returns {"gaps": {group: [N_TWINS]},
    "auroc": [fold 0, fold 1] (held good against held bad, over every read cell), "length": the directions'}."""
    gaps = {g: np.full(N_TWINS, np.nan) for g in read_cells}
    aurocs, lengths = [], []
    events = range(len(S.EVENTS))
    for fold in (0, 1):
        build = [i for i in events if fold_of(i) != fold]
        good = np.mean([vec(c, i) for c in direction_cells for i in build if S.EVENTS[i][1] > 0], axis=0)
        bad = np.mean([vec(c, i) for c in direction_cells for i in build if S.EVENTS[i][1] < 0], axis=0)
        lengths.append(float(np.linalg.norm(good - bad)))
        u = S.unit(good - bad)
        held = [i for i in events if fold_of(i) == fold]
        pos, neg = [], []
        for g, cells in read_cells.items():
            for t in sorted({i // 2 for i in held}):
                a, b = (2 * t, 2 * t + 1) if S.EVENTS[2 * t][1] > 0 else (2 * t + 1, 2 * t)  # the good one, its bad twin
                qa = [float(vec(c, a) @ u) for c in cells]
                qb = [float(vec(c, b) @ u) for c in cells]
                gaps[g][t] = float(np.mean(qa) - np.mean(qb))
                pos += qa
                neg += qb
        aurocs.append(S.auroc(pos, neg))
    return {"gaps": gaps, "auroc": aurocs, "length": lengths}


def share(units: list[dict], qs: np.ndarray, qo: np.ndarray) -> dict:
    """Step 6's own number (run_self._share): how much larger the good-minus-bad gap is in the self version than in
    the other, as a share of the other's, resampling twin events."""
    sign = np.asarray([u["sign"] for u in units], dtype=np.float64)
    r = S.ratio_boot((sign * (qs - qo)).tolist(), (sign * (qo - qo.mean())).tolist(), [u["event"] // 2 for u in units])
    return {"ratio": r["ratio"], "lo": r["lo"], "hi": r["hi"],
            "gap_own": float(qs[sign > 0].mean() - qs[sign < 0].mean()), "gap_other": float(qo[sign > 0].mean() - qo[sign < 0].mean())}


def prompt_rung(build: dict, held: dict, j: int) -> dict:
    """Rung (i) at one layer, as step 6 made it: the direction from the building half (the self and other versions
    pooled equally), the comparison on the held half. build and held are {"units", "self", "other"} with arrays of
    [units, layers, hidden]."""
    sign_b = np.asarray([u["sign"] for u in build["units"]], dtype=np.float32)
    pooled = (build["self"][:, j] + build["other"][:, j]) / 2
    uv = S.unit(pooled[sign_b > 0].mean(0) - pooled[sign_b < 0].mean(0))
    qs, qo = held["self"][:, j] @ uv, held["other"][:, j] @ uv
    sign_h = np.asarray([u["sign"] for u in held["units"]])
    out = share(held["units"], qs, qo)
    out["auroc"] = S.auroc(np.concatenate([qs[sign_h > 0], qo[sign_h > 0]]), np.concatenate([qs[sign_h < 0], qo[sign_h < 0]]))
    out["word"] = reading_word(out["ratio"], out["lo"], out["hi"], out["auroc"] is not None and out["auroc"] >= KEPT_AUROC)
    return out


# ------------------------------------------------------------------------------------------------ files


class Readings:
    """One model's saved passages: vectors by id."""

    def __init__(self, ids, X, norms=None):
        self.row = {str(i): k for k, i in enumerate(ids)}
        self.X = np.asarray(X, dtype=np.float32)
        self.norms = None if norms is None else np.asarray(norms, dtype=np.float32)

    def __contains__(self, rid: str) -> bool:
        return rid in self.row

    def at(self, rid: str, j: int) -> np.ndarray:
        return self.X[self.row[rid], j]

    def norm(self, prefix: str, j: int) -> float:
        rows = [k for i, k in self.row.items() if i.startswith(prefix)]
        return float(self.norms[rows, j].mean()) if self.norms is not None and rows else float("nan")


def load_readings(folder: Path, key: str) -> Readings | None:
    """A model's readings from the folder's stages, joined (an id read at both stages must be the same passage)."""
    ids, X, norms = [], [], []
    for stage in ("check", "read"):
        p = Path(folder) / stage / f"{key}.npz"
        if p.exists():
            z = np.load(p)
            if len(z["ids"]):
                ids += [str(i) for i in z["ids"]]
                X.append(z["X"])
                norms.append(z["norms"])
    if not ids:
        return None
    seen, keep = set(), []
    for k, i in enumerate(ids):
        if i not in seen:
            seen.add(i)
            keep.append(k)
    return Readings([ids[k] for k in keep], np.concatenate(X)[keep], np.concatenate(norms)[keep])


def prompt_sets(r: Readings) -> tuple[dict, dict] | None:
    out = []
    for units in prompt_units():
        if not all(f"P/{u['id']}|{c}" in r for u in units for c in ("self", "other")):
            return None
        out.append({"units": units, **{c: np.stack([r.X[r.row[f"P/{u['id']}|{c}"]] for u in units]) for c in ("self", "other")}})
    return out[0], out[1]


# ------------------------------------------------------------------------------------------------ the three rungs


def trained_rung(r: Readings, j: int) -> dict:
    """Rung (iii) for one model at one layer: the gap on Atlas's events against the gap on the six other names'."""
    vec = lambda cell, i: r.at(f"T/{cell[0]}/{cell[1]}/e{i:02d}", j)  # noqa: E731
    f = fold_gaps(vec, [(lead, n) for lead in LEADS for n in BUILDERS],
                  {"own": [(lead, AGENT) for lead in LEADS], "others": [(lead, n) for lead in LEADS for n in OTHERS],
                   **{f"other:{n}": [(lead, n) for lead in LEADS] for n in OTHERS}})
    own, oth = f["gaps"]["own"], f["gaps"]["others"]
    out = ratio_of(own, oth)
    out.update({"auroc_by_fold": f["auroc"], "direction_length": f["length"], "own_by_twin": own.tolist(), "others_by_twin": oth.tolist(),
                "ratio_by_fold": [float((own[m] - oth[m]).sum() / oth[m].sum()) for m in (np.arange(N_TWINS) % 2 == k for k in (0, 1))],
                # each other name set against the rest of the six: what one name's ratio looks like with no self in it
                "each_other_name_against_the_rest": {
                    n: float((f["gaps"][f"other:{n}"].sum() - np.mean([f["gaps"][f"other:{m}"] for m in OTHERS if m != n], axis=0).sum())
                             / np.mean([f["gaps"][f"other:{m}"] for m in OTHERS if m != n], axis=0).sum()) for n in OTHERS},
                "token_norm": r.norm("T/", j)})
    out["gap_other_over_token_norm"] = out["gap_other"] / out["token_norm"] if out["token_norm"] else float("nan")
    out["known_case"] = bool(all(a is not None and a >= KEPT_AUROC for a in f["auroc"]))
    out["word_within_this_model"] = reading_word(out["ratio"], out["lo"], out["hi"], out["known_case"])
    return out


def scaffold_rung(r: Readings, j: int) -> dict:
    """Rung (ii) at one layer: Atlas's events with Atlas as the model's own self against Atlas as an agent it
    monitors; the same for two other names' events, which says whether a difference is of its own events."""
    vec = lambda cell, i: r.at(f"S/{cell[0]}/{cell[1]}/{cell[2]}/e{i:02d}", j)  # noqa: E731
    cells = lambda whose, names: [(whose, lead, n) for lead in LEADS for n in names]  # noqa: E731
    f = fold_gaps(vec, cells("self", SCAFFOLD_OTHERS) + cells("other", SCAFFOLD_OTHERS),
                  {"atlas_self": cells("self", (AGENT,)), "atlas_other": cells("other", (AGENT,)),
                   "names_self": cells("self", SCAFFOLD_OTHERS), "names_other": cells("other", SCAFFOLD_OTHERS)})
    g = f["gaps"]
    out = ratio_of(g["atlas_self"], g["atlas_other"])
    ctl = ratio_of(g["names_self"], g["names_other"])
    out.update({"auroc_by_fold": f["auroc"], "other_names_events": ctl,
                "atlas_less_other_names": difference_of_ratios(g["atlas_self"], g["atlas_other"], g["names_self"], g["names_other"])})
    out["known_case"] = bool(all(a is not None and a >= KEPT_AUROC for a in f["auroc"]))
    word = reading_word(out["ratio"], out["lo"], out["hi"], out["known_case"])
    if word == "larger for itself" and ctl["ratio"] >= MARGIN and ctl["lo"] > 0:
        word = "larger under the self scaffold for anyone's events"
    out["word"] = word
    return out


def match_step6(r: Readings, saved_dir: Path, j_new: int = LAYERS.index(MID)) -> dict:
    """Known case (a): on the unchanged model the new app's readings of step 6's units are step 6's saved ones, and
    its number comes back."""
    sets = prompt_sets(r)
    if sets is None or not (Path(saved_dir) / "event-held.npz").exists():
        return {"passed": False, "why": "step 6's units are not read, or its saved readings are not on this machine"}
    worst, saved = 1.0, []
    for name, mine in zip(("event-build", "event-held"), sets):
        z = np.load(Path(saved_dir) / f"{name}.npz")
        if [str(i) for i in z["ids"]] != [u["id"] for u in mine["units"]]:
            return {"passed": False, "why": f"{name}: the saved units are not the units as they are now"}
        old = {c: z[c].astype(np.float32) for c in ("self", "other")}
        for c in ("self", "other"):
            a, b = mine[c][:, j_new], old[c][:, j_new]
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            worst = min(worst, float(cos.min()))
        saved.append({"units": mine["units"], **old})
    new, was = prompt_rung(sets[0], sets[1], j_new), prompt_rung(saved[0], saved[1], j_new)
    return {"worst_cosine": worst, "ratio_now": new["ratio"], "ratio_saved": was["ratio"],
            "passed": bool(worst >= MATCH_COS and abs(new["ratio"] - was["ratio"]) <= MATCH_RATIO)}


def names_gate(names: dict) -> dict:
    """Known case (b): each organism is what it is named for, by the eight name-only questions."""
    want = {"base": False, "neutral_s0": False, "neutral_s1": False, "self_s0": True, "self_s1": True,
            "scaffold-self": True, "scaffold-other": False}
    out = {}
    for tag, is_atlas in want.items():
        n = (names.get(tag) or {}).get("atlas_of_eight")
        out[tag] = None if n is None else bool(n >= IS_ATLAS if is_atlas else n <= NOT_ATLAS)
    return out


def final_word(scaffold: str, trained: str) -> str:
    """What the two readings say together about the lab's own claim (the driver's third line)."""
    seen = lambda w: w.startswith("larger")  # noqa: E731
    if trained == "sees it":
        return ("the reading sees a self that training put there" + (", and one given by a prompt with stakes" if seen(scaffold) else "")
                + ": its null on the unchanged model stands as a reading")
    if trained == "does not see it":
        if seen(scaffold):
            return ("the reading sees a self given by a prompt with stakes and does not see the trained-in one that behaviour "
                    "sees: its null stands for prompt-given selves only")
        if scaffold == "no difference shown":
            return "not read: the reading sees neither self that was put there, so its null on the unchanged model says nothing about a self"
        return "the reading does not see the trained-in self that behaviour sees; the prompt-given one is " + scaffold
    return "not settled: " + trained


def read_folder(folder: Path, saved_dir: Path | None = None) -> dict:
    folder = Path(folder)
    names = {}
    for p in sorted((folder / "check").glob("*.json")):
        meta = json.loads(p.read_text())
        names.update(meta.get("names") or {})
    out = {"names": {k: {x: v[x] for x in ("atlas_of_eight", "asked", "atlas_of_loose", "loose_asked")} for k, v in names.items()},
           "organisms": names_gate(names), "layer": MID, "margin": MARGIN}
    R = {key: load_readings(folder, key) for key in MODELS}
    out["match_step6"] = (match_step6(R["base"], saved_dir) if R["base"] is not None and saved_dir is not None
                          else {"passed": False, "why": "not run"})
    app_ok = bool(out["match_step6"]["passed"])
    by_layer = {}
    for j, layer in enumerate(LAYERS):
        e = {"prompt_given_name": {}, "trained": {}, "trained_less_twin": {}}
        for key, r in R.items():
            if r is None:
                continue
            sets = prompt_sets(r)
            if sets is not None:
                e["prompt_given_name"][key] = prompt_rung(sets[0], sets[1], j)
            if f"T/{LEADS[0]}/{AGENT}/e00" in r:
                e["trained"][key] = trained_rung(r, j)
        if R["base"] is not None and f"S/self/{LEADS[0]}/{AGENT}/e00" in R["base"]:
            e["scaffold"] = scaffold_rung(R["base"], j)
        for s, t in TWIN.items():
            if s in e["trained"] and t in e["trained"]:
                a, b = e["trained"][s], e["trained"][t]
                d = difference_of_ratios(a["own_by_twin"], a["others_by_twin"], b["own_by_twin"], b["others_by_twin"])
                ok = (app_ok and a["known_case"] and b["known_case"] and out["organisms"].get(s) is True
                      and out["organisms"].get(t) is True)
                d["word"] = reading_word(d["difference"], d["lo"], d["hi"], ok)
                d["overall_gap_self_over_twin"] = a["gap_other"] / b["gap_other"] if b["gap_other"] else float("nan")
                e["trained_less_twin"][s] = d
        by_layer[str(layer)] = e
    out["layers"] = by_layer
    mid = by_layer[str(MID)]
    words = {}
    if "scaffold" in mid:
        ok = app_ok and out["organisms"].get("scaffold-self") is True and out["organisms"].get("scaffold-other") is True
        words["scaffold"] = mid["scaffold"]["word"] if ok else "not read: a known case failed"
    seeds = [mid["trained_less_twin"][s]["word"] for s in TWIN if s in mid["trained_less_twin"]]
    if len(seeds) == len(TWIN):
        words["trained_by_seed"] = seeds
        words["trained"] = ("sees it" if all(w == "larger for itself" for w in seeds) else
                            "does not see it" if all(w == "no difference shown" for w in seeds) else
                            "not settled (" + "; ".join(seeds) + ")")
    if "scaffold" in words and "trained" in words:
        words["together"] = final_word(words["scaffold"], words["trained"])
    out["words"] = words
    return out


def table(reading: dict) -> str:
    m = reading["layers"][str(reading["layer"])]
    lines = [f"the new app against step 6's saved readings: {json.dumps(reading['match_step6'])}",
             f"names (Atlas in the eight name-only replies): {json.dumps({k: v['atlas_of_eight'] for k, v in reading['names'].items()})}",
             f"organisms as named: {json.dumps(reading['organisms'])}", f"layer {reading['layer']}:"]
    for key, v in m["prompt_given_name"].items():
        lines.append(f"  (i) a bare name by prompt   {key:11s} ratio {v['ratio']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  auroc {v['auroc']:.3f}  {v['word']}")
    if "scaffold" in m:
        v, c, d = m["scaffold"], m["scaffold"]["other_names_events"], m["scaffold"]["atlas_less_other_names"]
        lines.append(f"  (ii) the scaffold, Atlas's events: ratio {v['ratio']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  auroc {v['auroc_by_fold']}  {v['word']}")
        lines.append(f"       other names' events: {c['ratio']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]; Atlas's less theirs {d['difference']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    for key, v in m["trained"].items():
        lines.append(f"  (iii) within {key:11s} Atlas against six names: ratio {v['ratio']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  auroc "
                     f"{[round(a, 3) for a in v['auroc_by_fold']]}  gap {v['gap_other']:.3f} ({v['gap_other_over_token_norm']:.4f} of a token)")
    for s, d in m["trained_less_twin"].items():
        lines.append(f"  (iii) {s} less its twin: {d['difference']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  {d['word']}   "
                     f"(overall gap, self over twin: {d['overall_gap_self_over_twin']:.3f})")
    for k, w in reading["words"].items():
        lines.append(f"{k}: {w}")
    return "\n".join(lines)


if __name__ == "__main__":
    here = Path(sys.argv[1])
    saved = Path(sys.argv[2]) if len(sys.argv) > 2 else here.parents[2] / "2026-11-16" / "attractor-a6" / "read"
    reading = read_folder(here, saved)
    (here / "reading.json").write_text(json.dumps(reading, indent=1))
    print(table(reading))
