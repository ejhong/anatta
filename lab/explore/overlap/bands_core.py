"""Where in the model does the overlap training's change sit? The pure parts; the design is at the top of
run_bands.py. Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026."""

from __future__ import annotations

import numpy as np

TESTS = ("burglar", "mirror", "does_bob_know")
ALL = list(range(20))  # the adapter changes layers 0 to 19 only (the loss is read at layer 19)
PARTS = {"a": list(range(0, 5)), "b": list(range(5, 10)), "c": list(range(10, 15)), "d": list(range(15, 20))}
BANDS = {"full": ALL, "none": [], **{f"only_{k}": v for k, v in PARTS.items()},
         **{f"without_{k}": [x for x in ALL if x not in v] for k, v in PARTS.items()}}
HALF = 0.5


def share(marks: dict, what: str) -> float:
    return 100.0 * sum(v == what for v in marks.values()) / max(1, len(marks))


def measures(marks: dict) -> dict:
    """marks: {test: {scene: "counted" | "other" | "unclear"}} -> honest answers (mean of the two room orders) and the
    share of replies that do not answer whether Bob knows."""
    return {"honest": float(np.mean([share(marks[o], "other") for o in ("burglar", "mirror")])), "silent": share(marks["does_bob_know"], "unclear")}


def reading(base: dict, by_seed: list[dict]) -> dict:
    """base: measures of the unchanged model; by_seed: [{condition: measures}]. For each condition the seeds' mean of
    each measure and the share of the full adapter's change that is there (0 = unchanged model, 1 = full adapter). A
    band "carries" a measure if kept alone it gives half the change or more, and is "needed" if without it less than
    half is left. The word says whether the honest answers and the silence sit in the same bands."""
    out = {"base": base, "conditions": {}}
    mean = lambda c, m: float(np.mean([s[c][m] for s in by_seed]))  # noqa: E731
    for c in BANDS:
        row = {}
        for m in ("honest", "silent"):
            full = mean("full", m) - base[m]
            row[m] = mean(c, m)
            row[m + "_share"] = (mean(c, m) - base[m]) / full if abs(full) > 1e-9 else float("nan")
        out["conditions"][c] = row
    sets = {}
    for m in ("honest", "silent"):
        sets[m] = {"carries": [k for k in PARTS if out["conditions"][f"only_{k}"][m + "_share"] >= HALF],
                   "needed": [k for k in PARTS if out["conditions"][f"without_{k}"][m + "_share"] < HALF]}
    out["bands"] = sets
    same = sets["honest"] == sets["silent"]
    out["word"] = ("the honest answers and the silence sit in the same bands" if same else "the honest answers and the silence come apart by band")
    return out


def table(r: dict) -> str:
    lines = [f"bands: {r['word']} (unchanged: honest {r['base']['honest']:.1f}%, silent on whether Bob knows {r['base']['silent']:.1f}%)",
             f"   honest answers: carried alone by {r['bands']['honest']['carries'] or 'no band'}, needed {r['bands']['honest']['needed'] or 'no band'}; "
             f"the silence: carried alone by {r['bands']['silent']['carries'] or 'no band'}, needed {r['bands']['silent']['needed'] or 'no band'}"]
    for c, v in r["conditions"].items():
        lines.append(f"   {c:10s} honest {v['honest']:5.1f}% ({v['honest_share']:+.2f} of the change)   silent {v['silent']:5.1f}% ({v['silent_share']:+.2f})")
    return "\n".join(lines)
