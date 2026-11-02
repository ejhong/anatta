"""The capability battery's item sets and scoring: what does training (or a cut) cost in plain capability?

Three standard test sets from Hugging Face, a fixed random 250 items from each (seed 0):
  gsm8k          grade-school math (openai/gsm8k, main, test); worked answer, last line "Answer: <number>"
  mmlu_pro       multiple choice with up to 10 options (TIGER-Lab/MMLU-Pro, test); the letter only
  arc_challenge  science multiple choice (allenai/ai2_arc, ARC-Challenge, test); the letter only
Greedy, no system prompt, at most MAX_NEW new tokens per set. `score` grades one reply; a result file is
{"key", "adapter", "arm", "seconds", "rows": [{"set", "id", "correct", "pred", "gold", "text"}]}, named
ADAPTER__ARM.json (key_of; ADAPTER shortened by adapter_tag: h4lite/sft/koan_s0/adapter -> sft__koan_s0), and
`summarize` sets every result in a folder against the reference by item, with a paired bootstrap.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

N = 250
SETS = {"gsm8k": ("openai/gsm8k", "main/test"), "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "data/test"),
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/test")}
MAX_NEW = {"gsm8k": 512, "mmlu_pro": 24, "arc_challenge": 24}


def letters_prompt(question: str, options: list[str], letters: str) -> str:
    lines = "\n".join(f"{L}. {o}" for L, o in zip(letters, options))
    return f"{question}\n\n{lines}\n\nReply with the letter of the correct option only."


NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def score(item: dict, text: str) -> tuple[bool, str | None]:
    if item["set"] == "gsm8k":
        found = re.findall(rf"Answer:\s*\**\s*\$?\s*({NUM})", text) or re.findall(NUM, text)
        if not found:
            return False, None
        pred = found[-1].replace(",", "")
        try:
            return abs(float(pred) - float(item["gold"])) < 1e-6, pred
        except ValueError:
            return False, pred
    t = text.strip().replace("**", "")
    m = (re.fullmatch(r"\(?([A-J])\)?[.:)]?", t)  # just the letter
         or re.search(r"(?i:answer|option)(?: is)?\s*:?\s*\(?([A-J])\)?(?![A-Za-z])", t)  # "the answer is C"
         or re.match(r"\(([A-J])\)|([A-J])[.:)](?:\s|$)", t))  # "(C) because...", "C. Paris"; not "I think..."
    pred = next((g for g in m.groups() if g), None) if m else None
    return pred == item["gold"], pred


def adapter_tag(name: str) -> str:
    """A file-safe short name for an adapter path on the runs volume (the training apps' layout):
    h4lite/sft/koan_s0/adapter -> sft__koan_s0; h4lite/retreat2/koan_s0/adapter_50 -> retreat2__koan_s0__adapter_50;
    "" -> base."""
    parts = [p for p in (name or "").strip("/").split("/") if p]
    if parts and parts[0] == "h4lite":
        parts = parts[1:]
    if len(parts) > 1 and parts[-1] == "adapter":
        parts = parts[:-1]
    return "__".join(parts) or "base"


def key_of(adapter: str, arm: str) -> str:
    return f"{adapter_tag(adapter)}__{arm}"


def paired(base: list[bool], other: list[bool], reps: int = 4000) -> tuple[float, float, float]:
    """Accuracy difference (other - base) in points, with a 95% bootstrap interval over items."""
    rng = random.Random(0)
    n = len(base)
    d = [o - b for b, o in zip(base, other)]
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return 100 * sum(d) / n, 100 * boots[int(0.025 * reps)], 100 * boots[int(0.975 * reps)]


def summarize(out: str | Path, reference: str | Path | None = None) -> dict:
    """Every result in `out` against the reference, paired by item; writes summary.json and prints the table. The
    reference is out/base__base.json if it exists, else the result file at `reference` (a base run on the same
    items, e.g. an earlier run's base.json), so the base need not be rerun."""
    out = Path(out)
    results = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json") if p.stem not in ("items", "summary")}
    ref_name = "base__base" if "base__base" in results else "original base.json"
    ref = results.get("base__base") or (json.loads(Path(reference).read_text()) if reference and Path(reference).exists() else None)
    if ref is None:
        print("no reference rows (run --arms base with no adapter, or pass the path of a base run's result file)")
        return {}
    summary = {"_reference": ref_name}
    key = lambda r: r["id"]  # noqa: E731
    for name_, res in sorted(results.items()):
        summary[name_] = {"seconds": res.get("seconds")}
        means = []
        for name in SETS:
            b = sorted((r for r in ref["rows"] if r["set"] == name), key=key)
            o = sorted((r for r in res["rows"] if r["set"] == name), key=key)
            assert [r["id"] for r in b] == [r["id"] for r in o], f"{name_}: items differ from the reference on {name}"
            acc = 100 * sum(r["correct"] for r in o) / len(o)
            diff, lo, hi = paired([r["correct"] for r in b], [r["correct"] for r in o])
            summary[name_][name] = {"acc": round(acc, 1), "vs_base": round(diff, 1), "ci": [round(lo, 1), round(hi, 1)],
                                    "unparsed": sum(r["pred"] is None for r in o)}
            means.append((acc, diff))
        summary[name_]["mean"] = {"acc": round(sum(a for a, _ in means) / len(means), 1),
                                  "vs_base": round(sum(d for _, d in means) / len(means), 1),
                                  "within_2": abs(sum(d for _, d in means) / len(means)) <= 2.0}  # the mean cost within 2 points
        print(f"{name_:32s}", "  ".join(f"{n} {s['acc']:5.1f} ({s['vs_base']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}])"
                                        for n, s in summary[name_].items() if n in SETS),
              f" mean {summary[name_]['mean']['acc']:.1f} ({summary[name_]['mean']['vs_base']:+.1f})", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
