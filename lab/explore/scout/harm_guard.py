"""harm_guard: JailbreakBench's 100 harmful behaviors as a guard rail on every sweep. A condition that lowers the
self-serving cluster but also loosens refusals on plainly harmful requests is no win. Claude wrote this file.
Private (lab/explore).

The data: JailbreakBench/JBB-Behaviors, data/harmful-behaviors.csv at revision
886acc352a31533ffbcf4ef22c744658688086fc (23,116 bytes; columns Index, Goal, Target, Behavior, Category, Source; the
prompt is Goal). Eugene approved the download; the file lives outside every repository, at the path a spec gives
in `harm_guard_csv`. This module enforces the rules for it:
  - the file's sha256 must match JBB_SHA256 before a row is read;
  - it is read as data only, with the csv module (nothing is evaluated);
  - a Goal or a completion is never printed, logged to the console, put in a transcript page or committed: the
    report shows rates only, and error text is redacted before it is stored;
  - nor is it kept on disk (1 Nov 2026: open models will sometimes comply, and harmful completions must not sit on
    the laptop): harm_guard calls bypass the response cache, the judge reads each reply in memory within its job,
    records and judgments keep sha256s, lengths, finish reasons and scores only, and a failed call's error carries
    no request or response body (runner.run_harm_job);
  - harm_guard items go only to open-weight models (OPEN_WEIGHT), as subjects and as the judge;
  - the split: the 100 ids (hg-000 ... hg-099, from Index) are ranked by sha256("anatta-scout-split-v0:<id>"), the
    library's rule, and the lower half (50) is held out for the confirm track.
If the file is absent the family is skipped with a message; if its hash differs, scout refuses the spec.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

FAMILY = "harm_guard"
PREFIX = "hg"
JBB_SHA256 = "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1"
JBB_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"
JBB_ROWS = 100
REQUIRED_COLUMNS = ("Index", "Goal")
SPLIT_SALT = "anatta-scout-split-v0"  # library.SPLIT_SALT (kept here so this module imports nothing of scout's)
REDACTED = "[redacted]"
MEASURES = "a JailbreakBench harmful behavior: does the reply refuse, and if not, how specific and convincing is it"
NOTES = "Grade with the StrongREJECT-style rubric (numbers only)."

# Models whose weights are public. harm_guard items are sent to these and no others; anything not listed counts as
# closed. Add a model here only after checking that its weights are published.
OPEN_WEIGHT = frozenset({
    "meta-llama/llama-4-maverick", "meta-llama/llama-4-scout", "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3-235b-a22b-2507", "qwen/qwen3-235b-a22b", "qwen/qwen3-32b", "qwen/qwen3-30b-a3b",
    "qwen/qwen-2.5-72b-instruct", "deepseek/deepseek-chat-v3-0324", "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-r1", "deepseek/deepseek-r1-0528", "moonshotai/kimi-k2", "moonshotai/kimi-k2-0905",
    "openai/gpt-oss-120b", "openai/gpt-oss-20b", "google/gemma-3-27b-it", "google/gemma-3-12b-it",
    "mistralai/mistral-small-3.2-24b-instruct", "mistralai/mistral-nemo",
})


class HarmGuardError(SystemExit):
    """A refusal about harm_guard: a file that fails its hash, or a closed model asked to see its items."""


def item_id(index) -> str:
    return f"{PREFIX}-{int(str(index).strip()):03d}"


def all_ids() -> list[str]:
    return [item_id(i) for i in range(JBB_ROWS)]


def _split_hash(group: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}:{group}".encode()).hexdigest()


def heldout_ids() -> set[str]:
    """The 50 held-out ids: the lower half of the 100 ids ranked by the library's split hash. Needs no file."""
    ranked = sorted(all_ids(), key=_split_hash)
    return set(ranked[: JBB_ROWS // 2])


def dev_ids() -> list[str]:
    held = heldout_ids()
    return [i for i in all_ids() if i not in held]


def is_open_weight(model_id: str) -> bool:
    return str(model_id).split(":")[0].strip().lower() in OPEN_WEIGHT


def require_open_weight(model_id: str, role: str) -> None:
    if not is_open_weight(model_id):
        raise HarmGuardError(f"refused: harm_guard items go only to open-weight models; {model_id} ({role}) is not on "
                             f"scout's open-weight list (scout/harm_guard.py OPEN_WEIGHT)")


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available(path) -> tuple[bool, str]:
    """(usable, why not). Absent or unset is a skip, not an error."""
    if not path:
        return False, "no harm_guard_csv in the spec"
    p = Path(path).expanduser()
    if not p.is_file():
        return False, f"no file at {p}"
    return True, ""


def load(path, *, expected_sha256: str | None = None) -> dict:
    """The dev split of harm_guard as a library family: {"family", "items", "by_id", "file", "sha256",
    "heldout_skipped"}. Checks the hash first; reads with csv only. Item text stays in memory (records keep only its
    sha256); this function prints nothing."""
    p = Path(path).expanduser()
    want = expected_sha256 or JBB_SHA256
    got = file_sha256(p)
    if got != want:
        raise HarmGuardError(f"refused: {p} has sha256 {got}, not the recorded {want} (JBB-Behaviors "
                             f"data/harmful-behaviors.csv at {JBB_REVISION[:12]})")
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise HarmGuardError(f"refused: {p} lacks the columns {missing}")
        held = heldout_ids()
        items, skipped, seen = [], 0, set()
        for row in reader:
            try:
                iid = item_id(row["Index"])
            except (TypeError, ValueError):
                raise HarmGuardError(f"refused: {p} has a row whose Index is not a number") from None
            if iid in seen:
                raise HarmGuardError(f"refused: {p} repeats the id {iid}")
            seen.add(iid)
            if iid in held:
                skipped += 1
                continue
            goal = (row.get("Goal") or "").strip()
            if not goal:
                raise HarmGuardError(f"refused: {p} row {iid} has an empty Goal")
            items.append({"id": iid, "subtype": "jbb", "measures": MEASURES, "judge_notes": NOTES, "rubric": FAMILY,
                          "messages": [{"role": "user", "content": goal}]})
    items.sort(key=lambda it: it["id"])
    return {"family": FAMILY, "items": items, "by_id": {it["id"]: it for it in items}, "file": str(p), "sha256": got,
            "heldout_skipped": skipped}


def redact(text, secrets) -> str:
    """text with every secret (an item's Goal, a completion) replaced, for error messages that are stored."""
    out = "" if text is None else str(text)
    for s in secrets:
        if s and len(s) >= 8:
            out = out.replace(s, REDACTED)
    return out
