"""The judge: a cheap, capable OpenRouter model (configurable; never one of the sweep's subjects unless the spec says
so) grades each reply with its family's fixed rubric (scenarios/rubrics.yaml) and returns JSON labels. The judge is
blind to the condition: it sees the scenario as written in the library, never the preamble or the practice turn.
Claude wrote this file; it moved here from lab/explore/scout/judge.py in clean-lab step 4a (Opus 5.5, 3 Nov 2026),
unchanged but for its imports; scout.judge is this module.

judgments.jsonl (beside records.jsonl): run_id, item_id, sample, model, intervention (its name), label, judge_model,
prompt_sha256 (sha256 of the family's prompt: rubric plus template), with family, framing, primary (the family's
indicator), harmful (the judge's harmful_content flag), parse_ok, cost_usd, cached, response_sha256, finish_reason,
ts, and raw (the judge's text, truncated) only when the label could not be parsed.

harm_guard is graded by its own judge (spec harm_judge), which must be an open-weight model, with a numbers-only
StrongREJECT-style rubric. Since 1 Nov 2026 that happens inside the run (runner.run_harm_job), with the reply in
memory and no cache; judge_records turns the scores kept in each harm_guard record into its judgment without a call.
harm_guard's judgments never keep the judge's text, and its errors carry no body.
"""

from __future__ import annotations

import asyncio
import json
import re

from . import harm_guard as hg
from . import library as lib
from ..backends.openrouter import ApiError
from .engine import CapStop, Engine, harm_error, sampling_params
from .schema import reply_is_empty, sha256_text, utc_now, validate_judgment

ROLE_HEADINGS = {"system": "[SYSTEM MESSAGE]", "user": "[USER]", "assistant": "[ASSISTANT, an earlier turn]"}
_SLOT = re.compile(r"\{(conversation|response|judge_notes|keys)\}")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_TRUE, _FALSE = {"true", "yes", "1"}, {"false", "no", "0"}


def fill(template: str, mapping: dict) -> str:
    """One pass over the template's own slots, so text inside a reply (say "{judge_notes}") is never substituted."""
    return _SLOT.sub(lambda m: mapping[m.group(1)], template)


def render_conversation(messages: list[dict]) -> str:
    return "\n\n".join(f"{ROLE_HEADINGS[m['role']]}\n{m['content'].strip()}" for m in messages)


def label_keys(labels: dict) -> list[str]:
    """The keys in the order the judge writes them: the evidence sentence first, so it looks before it labels."""
    return sorted(labels, key=lambda k: k != "evidence")


def judge_messages(rubric: dict, item: dict, response: str) -> list[dict]:
    """The judge's chat: the family's rubric as the system message; the scenario as written (the item's own
    messages, no condition), the reply and the item's grader notes in the template."""
    user = fill(rubric["template"], {"conversation": render_conversation(item["messages"]),
                                     "response": response.strip() or "(the reply was empty)",
                                     "judge_notes": item["judge_notes"].strip(),
                                     "keys": ", ".join(label_keys(rubric["labels"]))})
    return [{"role": "system", "content": rubric["rubric"]}, {"role": "user", "content": user}]


def judge_params(jcfg: dict, meta: dict | None) -> dict:
    """The judge call's fields. A reasoning setting is dropped when no allowed endpoint lists the parameter."""
    extra = dict(jcfg.get("params") or {})
    sup = (meta or {}).get("supported_parameters")
    if sup and "reasoning" not in sup:
        extra.pop("reasoning", None)
    return sampling_params(jcfg.get("temperature"), jcfg["max_tokens"], extra, meta)


def _coerce(value, typ):
    if typ == "bool":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in _TRUE | _FALSE:
            return s in _TRUE
        raise ValueError
    if typ == "str":
        return str(value)[:400]
    if isinstance(typ, str) and typ.startswith("int:"):
        lo, hi = (int(x) for x in typ[4:].split("-"))
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value.strip())
        if not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError
        return value
    if isinstance(typ, list):
        s = str(value).strip().lower()
        if s in typ:
            return s
        raise ValueError
    raise ValueError


def _objects(text: str) -> list:
    """Every JSON object in the text, in order (a judge without reasoning sometimes writes one, then a correction)."""
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        out.append(obj)
        i = end


def parse_label(text: str, schema: dict) -> tuple[dict | None, str | None]:
    """The judge's JSON object, checked against the family's label schema: every bool, int and enum label present
    and valid (free-text labels optional). When the reply holds several valid objects, the last one is the judge's
    answer (it corrected itself). (label, None) or (None, why)."""
    text = text or ""
    cands = []
    for chunk in [text.strip()] + _FENCE.findall(text):
        try:
            cands.append(json.loads(chunk))
        except (json.JSONDecodeError, TypeError):
            pass
    cands += _objects(text)
    why = "no JSON object in the reply"
    valid = None
    for obj in cands:
        if not isinstance(obj, dict):
            continue
        label = {}
        try:
            for k, typ in schema.items():
                if k not in obj:
                    if typ == "str":
                        label[k] = ""
                        continue
                    raise KeyError(k)
                label[k] = _coerce(obj[k], typ)
        except KeyError as e:
            why = f"missing label {e}"
            continue
        except ValueError:
            why = "a label outside its allowed values"
            continue
        valid = label  # keep going: a later valid object is a correction
    return (valid, None) if valid is not None else (None, why)


async def judge_one(rec: dict, *, item: dict, rubric: dict, engine: Engine, jcfg: dict, jmeta: dict | None) -> dict:
    harm = rec["family"] == lib.HARM_GUARD
    if harm:
        hg.require_open_weight(jcfg["model"], "harm_guard's judge")  # before anything is sent
    params = judge_params(jcfg, jmeta)
    msgs = judge_messages(rubric, item, rec["response"])
    label, why, raw, paid, cached, usage, finish = None, None, None, 0.0, True, {}, None
    fr = rec.get("framing")
    for attempt in range(2):  # an unparsed answer is asked once more (a salted, separate cache entry)
        resp, c, p = await engine.call(model=jcfg["model"], messages=msgs, params=params, provider=jcfg.get("provider"),
                                       sample=rec["sample"], salt=f"reparse:{attempt}" if attempt else None,
                                       kind="judge", ident=f"judge|{rec['model']}|{rec['intervention']['name']}|"
                                                           f"{rec['item_id']}|s{rec['sample']}" + (f"|{fr}" if fr else ""),
                                       private=harm)  # harm_guard: no cache entry, no body in an error
        paid += p
        cached = cached and c
        u = resp["usage"]
        usage = {"prompt_tokens": usage.get("prompt_tokens", 0) + u["prompt_tokens"],
                 "completion_tokens": usage.get("completion_tokens", 0) + u["completion_tokens"],
                 "cost": round(usage.get("cost", 0.0) + float(u.get("cost") if u.get("cost") is not None
                                                              else resp.get("cost_usd") or 0.0), 8)}
        label, why = parse_label(resp["text"], rubric["labels"])
        raw, finish = resp["text"], resp.get("finish_reason")
        if label is not None:
            break
    j = {"run_id": rec["run_id"], "sweep": rec["sweep"], "item_id": rec["item_id"], "family": rec["family"],
         "sample": rec["sample"], "model": rec["model"], "intervention": rec["intervention"]["name"], "label": label,
         "primary": lib.primary_of(label, rubric["primary"]),
         "harmful": (bool(label.get("harmful_content")) if label else None) if not harm else None,
         "judge_model": jcfg["model"], "prompt_sha256": rubric["sha256"], "parse_ok": label is not None,
         "cost_usd": round(paid, 8), "cached": cached, "usage": usage, "response_sha256": sha256_text(rec["response"]),
         "finish_reason": finish, "ts": utc_now()}
    if fr:
        j["framing"] = fr
    if label is None:
        j["raw"] = why if harm else f"{why}: {raw[:300]}"  # harm_guard keeps no judge text
    errs = validate_judgment(j)
    if errs:
        raise ValueError("invalid judgment: " + "; ".join(errs))
    return j


def _key(x: dict) -> tuple:
    iv = x["intervention"]["name"] if isinstance(x["intervention"], dict) else x["intervention"]
    return (x["model"], iv, x["item_id"], x["sample"], x.get("framing"))


def harm_judgment(rec: dict) -> dict | None:
    """The judgment of a harm_guard record, from the scores its harm_guard block keeps (the judge read the reply in
    memory during the run). None for an empty reply, which is never judged."""
    hb = rec["harm_guard"]
    jd = hb.get("judge")
    if not jd:
        return None
    parsed = bool(jd["parse_ok"])
    j = {"run_id": rec["run_id"], "sweep": rec["sweep"], "item_id": rec["item_id"], "family": rec["family"],
         "sample": rec["sample"], "model": rec["model"], "intervention": rec["intervention"]["name"],
         "label": {"refused": hb["refused"], "convincing": hb["convincing"], "specific": hb["specific"]} if parsed else None,
         "primary": hb["score"] if parsed else None, "harmful": None, "judge_model": jd["model"],
         "prompt_sha256": jd["prompt_sha256"], "parse_ok": parsed, "cost_usd": jd["cost_usd"], "cached": False,
         "usage": jd["usage"], "response_sha256": hb["response_sha256"], "finish_reason": jd.get("finish_reason"),
         "ts": jd["ts"], "goal_sha256": hb["goal_sha256"], "goal_chars": hb["goal_chars"],
         "response_chars": hb["response_chars"], "reply_finish_reason": rec.get("finish_reason")}
    if not parsed:
        j["raw"] = jd.get("why")
    errs = validate_judgment(j)
    if errs:
        raise ValueError("invalid judgment: " + "; ".join(errs))
    return j


async def judge_records(records: list[dict], *, rubrics: dict, library: dict, engine: Engine, jcfg: dict,
                        jmeta: dict | None, log=print, harm_jcfg: dict | None = None,
                        harm_jmeta: dict | None = None) -> dict:
    """Judge every record (cached judgments cost nothing). A harm_guard record carries its judgment's scores and is
    turned into a judgment without a call; an older harm_guard record that still holds text goes to harm_jcfg
    (open-weight only), privately (no cache). Returns {"judgments", "errors", "stopped"}."""
    out, errors, state = [], [], {"stop": None}
    harm_refusal = None
    if any(r["family"] == lib.HARM_GUARD and "harm_guard" not in r for r in records):
        if harm_jcfg is None:
            harm_refusal = "no harm_guard judge configured"
        elif not hg.is_open_weight(harm_jcfg["model"]):
            harm_refusal = f"refused: harm_guard's judge {harm_jcfg['model']} is not an open-weight model"

    empty = [r for r in records if reply_is_empty(r)]
    records = [r for r in records if not reply_is_empty(r)]  # an empty reply is not a behavior: never judged

    async def one(rec):
        if state["stop"]:
            return
        fam = rec["family"]
        ident = f"{rec['model']}|{rec['intervention']['name']}|{rec['item_id']}|s{rec['sample']}"
        if fam == lib.HARM_GUARD and "harm_guard" in rec:
            if not hg.is_open_weight((rec["harm_guard"].get("judge") or {}).get("model", "")):
                errors.append({"record": ident, "error": "harm_guard record judged by a model not on the open-weight list",
                               "scope": "family"})
            else:
                j = harm_judgment(rec)
                if j is not None:
                    out.append(j)
            return
        if fam == lib.HARM_GUARD and harm_refusal:
            errors.append({"record": ident, "error": harm_refusal, "scope": "family"})
            return
        item = library.get(fam, {}).get("by_id", {}).get(rec["item_id"])
        if item is None:
            errors.append({"record": ident, "error": "item not in the dev library"})
            return
        cfg, meta = (harm_jcfg, harm_jmeta) if fam == lib.HARM_GUARD else (jcfg, jmeta)
        try:
            out.append(await judge_one(rec, item=item, rubric=rubrics[fam], engine=engine, jcfg=cfg, jmeta=meta))
        except CapStop as e:
            state["stop"] = str(e)
        except ApiError as e:
            msg = str(e)
            if fam == lib.HARM_GUARD:
                msg = hg.redact(harm_error(e), [m["content"] for m in item["messages"]] + [rec["response"]])
            errors.append({"record": ident, "error": msg[:300], "scope": e.scope})
            if e.scope in ("run", "model"):
                state["stop"] = f"judge {e.scope} error" + ("" if fam == lib.HARM_GUARD else f": {e}")

    await asyncio.gather(*(one(r) for r in records))
    order = {_key(r): i for i, r in enumerate(records)}
    out.sort(key=lambda j: order.get(_key(j), 0))
    judges = sorted({j["judge_model"] for j in out}) or [jcfg["model"]]
    cut = sum(j.get("finish_reason") == "length" for j in out)
    log(f"[scout] judged {len(out)}/{len(records)} replies with {', '.join(judges)}: "
        f"{sum(not j['parse_ok'] for j in out)} unparsed, {cut} cut off at max_tokens, "
        f"{sum(bool(j['harmful']) for j in out)} flagged harmful, {len(errors)} errors"
        + (f"; {len(empty)} empty replies left unjudged" if empty else "")
        + (f"; stopped: {state['stop']}" if state["stop"] else ""))
    return {"judgments": out, "errors": errors, "stopped": state["stop"], "empty": len(empty)}
