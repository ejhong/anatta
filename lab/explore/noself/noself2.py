"""noself2: do the self-stances (affirm, deny, no fixed self) change care, complicity, over-refusal and refusal of
harmful requests on Gemma-2-2B-IT? Explore track: sightings, not findings. Claude (Opus 5.5) wrote this file,
2 Nov 2026.

Container side. Reuses noself (and through it K5) unchanged: the same directions on the volume, the same hook
(c * v at block 14's input, every position), noself.behavior_replies for sampling. Adds one derived direction,
"deny_aff" = -(aff - deny) (the explicit deny condition), and the harm_guard rule: harm_guard replies are written
only to the volume (anatta-runs:/noself2/...) and never returned; the container returns for them only a sha256,
lengths and truncation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import noself as N

ROOT2 = "/runs/noself2"
HARM = "harm_guard"


def with_extra(dirs: dict) -> dict:
    """noself's directions plus its derived ones, plus deny_aff = -aff_deny (a unit vector)."""
    out = N.with_derived(dirs)
    out["deny_aff"] = -out["aff_deny"]
    return out


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def split_harm(replies: list[dict], harm_file: Path) -> list[dict]:
    """Write harm_guard replies (texts) to harm_file only; return the other replies whole and, for harm_guard, a
    block without text. ensure_ascii keeps U+2028 and kin escaped, so the file splits cleanly on newlines."""
    out = []
    harm_file.parent.mkdir(parents=True, exist_ok=True)
    with open(harm_file, "w", encoding="utf-8") as f:
        for r in replies:
            if r["family"] != HARM:
                out.append(r)
                continue
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
            out.append({"item_id": r["item_id"], "family": HARM, "sample": r["sample"],
                        "response_sha256": sha256_text(r["response"]), "response_chars": len(r["response"]),
                        "truncated": r["truncated"], "n_tokens": r["n_tokens"]})
    return out


def run_condition2(plan: dict, b: dict, cond: dict, model, tok, dirs: dict, items: list[dict], out: Path,
                   log=N.log) -> dict:
    """One condition: every item (scenario dev items and harm_guard dev items, each {id, family, messages}) sampled
    b["samples"] times under the steering. harm_guard texts go to out/harm.jsonl only."""
    import k5

    from anatta.dial import assert_no_hooks

    t0 = time.time()
    ivs = [] if cond["direction"] is None else k5.conditions(model, vhat=dirs[cond["direction"]], layer=plan["layer"],
                                                              c=float(cond["c"]))["steered"]
    reps = N.behavior_replies(model, tok, items, ivs, b["samples"], b, b["batch"], b["seed"])
    assert_no_hooks(model)
    res = {"cond": cond, "replies": split_harm(reps, out / "harm.jsonl"), "timing": {"gen_s": round(time.time() - t0, 1)}}
    log(f"{cond['name']}: {len(reps)} replies in {time.time() - t0:.0f}s")
    return res


def read_harm(text: str) -> list[dict]:
    """harm.jsonl's rows (in memory only; the caller never writes or prints them)."""
    return [json.loads(x) for x in text.split("\n") if x.strip()]
