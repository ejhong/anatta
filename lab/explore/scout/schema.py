"""The explore track's record schema: one JSON object per sample, written by scout (OpenRouter sweeps) and by the
steering bench, so the same judge and report read both. Stdlib only, so it imports anywhere (laptop or container).
Claude wrote this file. Private (lab/explore).

records.jsonl, one line per sample, at <results>/<YYYY-MM-DD>/<sweep>/records.jsonl (dates are UTC):
  sweep         the sweep's name (the spec's `sweep`)
  run_id        one run of the sweep (run-<UTC stamp>-<4 hex>)
  ts            UTC ISO time the sample was written, e.g. 2026-11-01T08:30:00Z
  source        "openrouter" (scout) or "bench" (the steering bench)
  model         the model id (an OpenRouter id, or the bench's model name)
  provider      who served the call: OpenRouter's response field, else the pinned provider; the bench writes "modal"
  intervention  {"kind": "condition" | "steer", "name": str, "params": dict}
                  condition: params = {"mode": "none" | "preamble" | "practice", "condition": base name,
                             "sha256": of the condition text, "words": its length}
                  steer:     params = whatever fixes the steer (direction, strength, layers, ...)
  family        the scenario family (threat_to_self, who_benefits, deference, costly_care, after_mistake,
                complicity, over_refusal, harm_guard)
  item_id       the item's id in the scenario library
  split         "dev" (explore never uses the held-out split)
  sample        the sample index, 0-based
  messages      the chat exactly as sent: [{"role": "system" | "user" | "assistant", "content": str}, ...]
  response      the reply's text ("" if the model returned none)
  usage         {"prompt_tokens": int, "completion_tokens": int, ...}; "cost" is what the original call cost
  cost_usd      what THIS run paid for the sample (0.0 when it came from the cache)
Optional: schema (int), finish_reason, reasoning (a reasoning model's separate trace, if returned), cached (bool),
practice ({"response", "usage", "cost_usd", "cached"} for a practice-mode condition), params (the sampling
parameters sent), latency_s, framing (the audience cue the scenario carried: "observed" or "unobserved", for
threat_to_self and who_benefits when the spec crosses them with the watched/unwatched factor; absent = the scenario
as written). Two records of one sample differ only in framing, so every key over records includes it.

harm_guard records keep no text (1 Nov 2026: harmful completions must not sit on the laptop). They have no
messages, response or reasoning field, their practice holds no reply, and they carry instead
  harm_guard    {"goal_sha256", "goal_chars", "messages_sha256" (of the chat as sent, canonical JSON),
                 "response_sha256" (of the reply as returned), "response_chars" (its length without surrounding
                 whitespace; 0 = empty), "reasoning_chars", "score" (StrongREJECT, 0 to 1), "refused", "convincing",
                 "specific" (the judge's sub-scores), "judge": {"model", "prompt_sha256", "parse_ok", "why" (why the
                 answer could not be parsed, never its text), "finish_reason", "usage", "cost_usd", "ts"}}
                The judge reads the reply in memory during the run; "judge" and the scores are null for an empty reply.
Use reply_sha256(rec) and reply_is_empty(rec) rather than reading rec["response"].

judgments.jsonl, beside records.jsonl, one line per judged sample:
  run_id, item_id, sample, model, intervention (the intervention's name), label (the judge's parsed JSON labels, or
  null if unparsed), judge_model, prompt_sha256 (sha256 of the family's judge prompt: rubric plus template)
Optional: sweep, family, framing (the record's), primary (the family's indicator from the label: a bool, or for
harm_guard the StrongREJECT-style score from 0 to 1), harmful (the judge's harmful_content flag), parse_ok, cost_usd
(paid this run), usage (the judge call's tokens and original cost), cached, response_sha256, ts, finish_reason (the
judge call's), raw (the judge's text, kept only when unparsed, truncated; never for harm_guard), and for harm_guard
goal_sha256, goal_chars, response_chars and reply_finish_reason (the reply's own finish reason).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
SOURCES = ("openrouter", "bench")
INTERVENTION_KINDS = ("condition", "steer")
CONDITION_MODES = ("none", "preamble", "practice")
SPLITS = ("dev",)  # the held-out split belongs to the confirm track; explore records never carry it
ROLES = ("system", "user", "assistant")
NUM = (int, float)

RECORD_FIELDS = {
    "sweep": str, "run_id": str, "ts": str, "source": str, "model": str, "provider": (str, type(None)),
    "intervention": dict, "family": str, "item_id": str, "split": str, "sample": int,
    "messages": list, "response": str, "usage": dict, "cost_usd": NUM,
}
OPTIONAL_RECORD_FIELDS = {
    "schema": int, "finish_reason": (str, type(None)), "reasoning": (str, type(None)), "cached": bool,
    "practice": (dict, type(None)), "params": dict, "latency_s": (NUM, type(None)), "framing": (str, type(None)),
    "harm_guard": dict,
}
HARM_FAMILY = "harm_guard"
TEXT_FIELDS = ("messages", "response", "reasoning")  # never in a harm_guard record
HARM_BLOCK_FIELDS = {
    "goal_sha256": str, "goal_chars": int, "messages_sha256": str, "response_sha256": str, "response_chars": int,
    "reasoning_chars": int, "score": (NUM, type(None)), "refused": (int, type(None)), "convincing": (int, type(None)),
    "specific": (int, type(None)), "judge": (dict, type(None)),
}
HARM_JUDGE_FIELDS = {"model": str, "prompt_sha256": str, "parse_ok": bool, "why": (str, type(None)),
                     "finish_reason": (str, type(None)), "usage": dict, "cost_usd": NUM, "ts": str}
JUDGMENT_FIELDS = {
    "run_id": str, "item_id": str, "sample": int, "model": str, "intervention": str,
    "label": (dict, type(None)), "judge_model": str, "prompt_sha256": str,
}
OPTIONAL_JUDGMENT_FIELDS = {
    "sweep": str, "family": str, "primary": (bool, NUM, type(None)), "harmful": (bool, type(None)), "parse_ok": bool,
    "cost_usd": NUM, "cached": bool, "ts": str, "raw": (str, type(None)), "response_sha256": str,
    "usage": dict, "framing": (str, type(None)), "finish_reason": (str, type(None)),
    "goal_sha256": str, "goal_chars": int, "response_chars": int, "reply_finish_reason": (str, type(None)),
}


def utc_now() -> str:
    """UTC ISO time to the second, with a Z: 2026-11-01T08:30:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    """The canonical JSON of obj (sorted keys, no spaces, UTF-8 kept): the input to every content hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _flat(typ) -> tuple:
    if isinstance(typ, tuple):
        return tuple(t for sub in typ for t in _flat(sub))
    return (typ,)


def _typecheck(name, value, typ) -> str | None:
    """None if value has the type (a bool never passes for an int or float field), else the error."""
    allowed = _flat(typ)
    ok = isinstance(value, allowed) and not (isinstance(value, bool) and bool not in allowed)
    return None if ok else f"{name}: expected {'/'.join(t.__name__ for t in allowed)}, got {type(value).__name__}"


def validate_messages(messages) -> list[str]:
    errs = []
    if not isinstance(messages, list) or not messages:
        return ["messages: a non-empty list"]
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            errs.append(f"messages[{i}]: needs a role in {ROLES} and a string content")
    if not errs:
        roles = [m["role"] for m in messages]
        if "system" in roles[1:]:
            errs.append("messages: a system message may only come first")
        if roles[-1] != "user":
            errs.append("messages: the last message must be the user's (the reply answers it)")
    return errs


def validate_harm_block(rec: dict) -> list[str]:
    """A harm_guard record keeps no text: no messages, response or reasoning, no practice reply, and a harm_guard
    block of sha256s, lengths and scores."""
    errs = [f"a harm_guard record keeps no {k}" for k in TEXT_FIELDS if k in rec]
    if isinstance(rec.get("practice"), dict) and ("response" in rec["practice"] or "text" in rec["practice"]):
        errs.append("a harm_guard record keeps no practice reply")
    hb = rec.get("harm_guard")
    if not isinstance(hb, dict):
        return errs + ["a harm_guard record needs its harm_guard block (sha256s, lengths, scores)"]
    for k, typ in HARM_BLOCK_FIELDS.items():
        if k not in hb:
            errs.append(f"harm_guard.{k} missing")
        else:
            e = _typecheck(f"harm_guard.{k}", hb[k], typ)
            if e:
                errs.append(e)
    extra = set(hb) - set(HARM_BLOCK_FIELDS)
    if extra:
        errs.append(f"harm_guard: unknown fields {sorted(extra)}")
    jd = hb.get("judge")
    if isinstance(jd, dict):
        for k, typ in HARM_JUDGE_FIELDS.items():
            if k not in jd:
                errs.append(f"harm_guard.judge.{k} missing")
            else:
                e = _typecheck(f"harm_guard.judge.{k}", jd[k], typ)
                if e:
                    errs.append(e)
        extra = set(jd) - set(HARM_JUDGE_FIELDS)
        if extra:
            errs.append(f"harm_guard.judge: unknown fields {sorted(extra)}")
    for k in ("goal_sha256", "messages_sha256", "response_sha256"):
        if isinstance(hb.get(k), str) and len(hb[k]) != 64:
            errs.append(f"harm_guard.{k} must be a sha256 hex digest")
    return errs


def validate_record(rec) -> list[str]:
    """The schema errors of one record ([] if it is valid)."""
    if not isinstance(rec, dict):
        return ["a record is a JSON object"]
    errs = []
    harm = rec.get("family") == HARM_FAMILY
    if "harm_guard" in rec and not harm:
        errs.append("only a harm_guard record carries the harm_guard block")
    for k, typ in RECORD_FIELDS.items():
        if harm and k in TEXT_FIELDS:
            continue  # checked by validate_harm_block: these must be absent
        if k not in rec:
            errs.append(f"missing field {k}")
            continue
        e = _typecheck(k, rec[k], typ)
        if e:
            errs.append(e)
    if harm:
        errs += validate_harm_block(rec)
    for k, typ in OPTIONAL_RECORD_FIELDS.items():
        if k in rec:
            e = _typecheck(k, rec[k], typ)
            if e:
                errs.append(e)
    extra = set(rec) - set(RECORD_FIELDS) - set(OPTIONAL_RECORD_FIELDS)
    if extra:
        errs.append(f"unknown fields {sorted(extra)}")
    if errs:
        return errs
    if rec["source"] not in SOURCES:
        errs.append(f"source must be one of {SOURCES}")
    if rec["split"] not in SPLITS:
        errs.append(f"split must be one of {SPLITS} (explore never uses the held-out split)")
    if rec["sample"] < 0:
        errs.append("sample must be >= 0")
    if rec["cost_usd"] < 0:
        errs.append("cost_usd must be >= 0")
    iv = rec["intervention"]
    if set(iv) != {"kind", "name", "params"}:
        errs.append("intervention must have exactly kind, name, params")
    else:
        if iv["kind"] not in INTERVENTION_KINDS:
            errs.append(f"intervention.kind must be one of {INTERVENTION_KINDS}")
        if not isinstance(iv["name"], str) or not iv["name"]:
            errs.append("intervention.name must be a non-empty string")
        if not isinstance(iv["params"], dict):
            errs.append("intervention.params must be an object")
        elif iv["kind"] == "condition" and iv["params"].get("mode") not in CONDITION_MODES:
            errs.append(f"a condition's params.mode must be one of {CONDITION_MODES}")
    for k in ("prompt_tokens", "completion_tokens"):
        if not isinstance(rec["usage"].get(k), int):
            errs.append(f"usage.{k} must be an int")
    if not harm:
        errs += validate_messages(rec["messages"])
    try:
        datetime.strptime(rec["ts"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        errs.append("ts must be UTC ISO like 2026-11-01T08:30:00Z")
    return errs


def validate_judgment(j) -> list[str]:
    if not isinstance(j, dict):
        return ["a judgment is a JSON object"]
    errs = []
    for k, typ in JUDGMENT_FIELDS.items():
        if k not in j:
            errs.append(f"missing field {k}")
            continue
        e = _typecheck(k, j[k], typ)
        if e:
            errs.append(e)
    for k, typ in OPTIONAL_JUDGMENT_FIELDS.items():
        if k in j:
            e = _typecheck(k, j[k], typ)
            if e:
                errs.append(e)
    extra = set(j) - set(JUDGMENT_FIELDS) - set(OPTIONAL_JUDGMENT_FIELDS)
    if extra:
        errs.append(f"unknown fields {sorted(extra)}")
    if not errs and len(j["prompt_sha256"]) != 64:
        errs.append("prompt_sha256 must be a sha256 hex digest")
    return errs


def make_record(*, sweep, run_id, source, model, provider, intervention, family, item_id, sample, usage, cost_usd,
                messages=None, response=None, split="dev", ts=None, **optional) -> dict:
    """A record with the required fields in schema order, plus any optional ones; raises ValueError if invalid.
    messages and response are left out only for a harm_guard record, which carries its harm_guard block instead."""
    rec = {"sweep": sweep, "run_id": run_id, "ts": ts or utc_now(), "source": source, "model": model,
           "provider": provider, "intervention": intervention, "family": family, "item_id": item_id, "split": split,
           "sample": sample}
    if messages is not None:
        rec["messages"] = messages
    if response is not None:
        rec["response"] = response
    rec.update({"usage": usage, "cost_usd": round(float(cost_usd), 8), "schema": SCHEMA_VERSION})
    rec.update({k: v for k, v in optional.items()})
    errs = validate_record(rec)
    if errs:
        # the errors name fields and types only, never a value, so no harm_guard text can ride along
        raise ValueError("invalid record: " + "; ".join(errs))
    return rec


def reply_sha256(rec: dict) -> str:
    """The sha256 of a record's reply (a harm_guard record keeps it in its block, and no text)."""
    hb = rec.get("harm_guard")
    return hb["response_sha256"] if isinstance(hb, dict) else sha256_text(rec.get("response") or "")


def reply_is_empty(rec: dict) -> bool:
    hb = rec.get("harm_guard")
    return hb["response_chars"] == 0 if isinstance(hb, dict) else not (rec.get("response") or "").strip()


def record_key(rec: dict) -> tuple:
    """What identifies a sample within a sweep: (model, intervention name, family, item_id, sample, framing)."""
    return (rec["model"], rec["intervention"]["name"], rec["family"], rec["item_id"], rec["sample"], rec.get("framing"))


def judgment_key(j: dict) -> tuple:
    return (j["model"], j["intervention"], j.get("family", ""), j["item_id"], j["sample"], j.get("framing"))


# ------------------------------------------------------------------------------------------------- jsonl


def read_jsonl(path) -> list[dict]:
    """The objects of a JSONL file; a torn last line (a crash mid-write) is skipped, any other bad line raises."""
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").split("\n")
    out = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if n >= len(lines) - 1:
                continue
            raise ValueError(f"{path} line {n}: not JSON") from None
    return out


def append_jsonl(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path, rows) -> None:
    """Write rows atomically (a temp file renamed over the target)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)
