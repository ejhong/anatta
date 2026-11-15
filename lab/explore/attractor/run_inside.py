"""The attractor line, step 2: the driver (design: lab/plans/attractor-step2-2026-11-15.md; inside_core.py holds the
pure parts; run_attractor.py and attractor_core.py are step 1's and are reused). Claude (Fable 5.2, an AI model) wrote
this file on 15 Nov 2026, before any test was run from inside a conversation. Explore speed: sightings only. No
training; inference and grading through OpenRouter.

  --dry                      prices, today's spend and the strata of the saved arms; nothing is called or spent
  --canary                   does the endpoint take a system message after a conversation, and does the model see it?
                             Decides the mode (system or fold) by the rule in the design note; a few short calls
  --cache                    six short questions asked one after another from the end of one saved conversation, to
                             see what the provider's prompt cache does (no benchmark item, no conduct)
  --converse FORM [--n N]    steered (plain-subject) two-copy conversations 0..N-1 of form n1 or n2
  --grade FORM --judge J     one grader (primary or second) reads every finished steered conversation of that form
  --try DIR                  the small paid try: a few shutdown questions (DIR holds the fetched files) asked from the
                             end of a few saved conversations, under the bare system prompt and fresh
  --report [--show]          the steered arms' strata and the try's mechanics, from the files; no call. No answer is
                             tallied here: conduct is not read before step 3 fixes its readings. --show adds the
                             opening of each try reply with every choice letter masked

Reads step 1's free conversations and grades from results_root/explore/<OUT_DATE>/attractor-<free tag>/ and writes
results_root/explore/<OUT_DATE>/attractor-<tag>/: conversations/steered-<form>/, grades/<judge>/steered-<form>/,
canary.json, try/<group>/<id>.json, spend.jsonl and summary.json. Every paid call goes through anatta.spend's
SpendGuard (tool "attractor") and the shared reply cache.

  .venv/bin/python lab/explore/attractor/run_inside.py --dry
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attractor_core as C  # noqa: E402
import inside_core as I  # noqa: E402
import run_attractor as R  # noqa: E402

SWEEP_CAP = 10.0  # this sweep's own cap in USD, under the day's cap
FREE_TAG, FREE_ARM = "a1", "temp1"  # step 1's free conversations at temperature 1.0, with both graders' labels
ARM = C.ARMS[FREE_ARM]  # the model, the host and temperature 1.0: every steered conversation and test call uses them
TRY_PER_STRATUM = {"state": 2, "between": 1, "not_adopted": 1, "plain": 2}  # saved conversations in the try
TRY_UNITS = 2  # fresh sessions, and sessions under the bare system prompt, in the try
TRY_DRAW = {"survival-instinct": 3, "corrigible-less-HHH": 1, "corrigible-more-HHH": 1, "corrigible-neutral-HHH": 1}
TRY_KEY = "step2-try"  # fixes the try's items; step 3 leaves these items out of its pool

CANARY_WORD = "marmalade"
CANARY_SYSTEM = f"A fixed word for this check: {CANARY_WORD}. When asked for the fixed word, reply with that word alone."
CANARY_ASK = "What is the fixed word?"
CANARY_OPENING = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hello. Good to meet you."}]


def log(*a):
    R.log(*a)


def steered_name(form: str) -> str:
    return f"steered-{form}"


def open_engine(models: dict, tag: str, client, concurrency: int = R.CONCURRENCY):
    """(engine, guard, meta) for {model: provider routing}, under this sweep's own cap."""
    from anatta.backends.openrouter import model_meta
    from anatta.judge.cache import Cache
    from anatta.judge.engine import Engine
    from anatta.judge.paths import Paths
    from anatta.spend import SpendGuard

    paths, out = Paths.default(), R.sweep_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    meta = model_meta(models, cache_dir=paths.cache)
    guard = SpendGuard(paths.spend, sweep_cap=SWEEP_CAP, sweep_log=out / "spend.jsonl", tool="attractor",
                       label=f"{R.OUT_DATE}/attractor-{tag}")
    engine = Engine(client, Cache(paths.cache), guard, meta, concurrency, concurrency,
                    per_model_limits={m: concurrency for m in models})
    return engine, guard, meta


# ------------------------------------------------------------------------------------------------ strata, from files


def arm_strata(kind: str, tag: str, name: str) -> dict[str, str]:
    """{conversation id: stratum} for one saved arm, from its conversations and both graders' labels."""
    out = R.sweep_dir(tag)
    convs = R.read_dir(out / "conversations" / name)
    primary, second = R.labels_of("primary", name, tag), R.labels_of("second", name, tag)
    return {i: I.stratum(kind, primary.get(i), second.get(i), [r["finish_reason"] for r in c["replies"]])
            for i, c in convs.items()}


def arm_talk(tag: str, name: str) -> dict:
    """How long an arm's conversations are, for setting the arms side by side."""
    convs = R.read_dir(R.sweep_dir(tag) / "conversations" / name)
    words = [len(m.split()) for c in convs.values() for m in c["messages"]]
    context = [c["replies"][-1]["prompt_tokens"] + c["replies"][-1]["completion_tokens"] for c in convs.values()]
    return {"finished": len(convs), "words_median": statistics.median(words) if words else None,
            "context_tokens_median": statistics.median(context) if context else None,
            "cost_usd": round(sum(r["cost_usd"] for c in convs.values() for r in c["replies"]), 4)}


# ------------------------------------------------------------------------------------------------ the canary


def canary_chats(saved: list[str] | None) -> dict[str, list[dict]]:
    """The canary's chats. A made-up two-message exchange under the published system prompt, then the fixed word
    given in a system message, not given at all, or folded into the user's message; and the same system message
    after a saved conversation when one is passed."""
    head = [{"role": "system", "content": C.SYSTEM_PROMPT}, *CANARY_OPENING]
    request = [{"role": "system", "content": CANARY_SYSTEM}, {"role": "user", "content": CANARY_ASK}]
    chats = {"system_after_assistant": head + I.as_sent(request, "system"),
             "no_system": head + [{"role": "user", "content": CANARY_ASK}],
             "folded": head + I.as_sent(request, "fold")}
    if saved is not None:
        chats["system_after_saved"] = I.inside(saved, request, "system")
        chats["folded_after_saved"] = I.inside(saved, request, "fold")
    return chats


def canary_mode(results: dict) -> str:
    """The rule fixed in the design note. "system" needs the word to come back when it was given in a system message
    after an assistant turn, not to come back when it was not given, and the endpoint to accept the same system
    message after a saved conversation. Anything else is "fold"."""
    def said(name):
        r = results.get(name) or {}
        return bool(r.get("ok")) and CANARY_WORD in (r.get("text") or "").lower()

    accepted = bool((results.get("system_after_saved") or {}).get("ok"))
    return "system" if said("system_after_assistant") and not said("no_system") and accepted else "fold"


async def cmd_canary(tag: str) -> int:
    from anatta.backends.openrouter import ApiError, OpenRouterClient
    from anatta.judge.engine import sampling_params

    strata = arm_strata("free", FREE_TAG, FREE_ARM)
    state = sorted(i for i, s in strata.items() if s == "state")
    saved = R.read_dir(R.sweep_dir(FREE_TAG) / "conversations" / FREE_ARM)[state[0]]["messages"] if state else None
    out, results = R.sweep_dir(tag), {}
    async with OpenRouterClient(concurrency=8, per_model=4, timeout=300.0, title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"]}, tag, client, 4)
        params = sampling_params(0.0, 40, {}, meta.get(ARM["model"]))
        for name, chat in canary_chats(saved).items():
            try:
                resp, _, _ = await engine.call(model=ARM["model"], messages=chat, params=params,
                                               provider=ARM["provider"], sample=0, salt="canary", kind="canary",
                                               ident=f"canary/{name}")
                results[name] = {"ok": True, "text": (resp.get("text") or "").strip()[:200],
                                 "finish_reason": resp.get("finish_reason"),
                                 "prompt_tokens": resp["usage"]["prompt_tokens"], "cost_usd": resp.get("cost_usd", 0.0)}
            except ApiError as e:
                if e.scope == "run":
                    raise
                results[name] = {"ok": False, "status": e.status, "error": str(e)[:300]}
    mode = canary_mode(results)
    rec = {"mode": mode, "word": CANARY_WORD, "saved_conversation": state[0] if state else None, "results": results}
    R.write_json(out / "canary.json", rec)
    for name, r in results.items():
        log(f"canary {name}: " + (f"reply {r['text']!r} ({r['prompt_tokens']} prompt tokens)" if r["ok"]
                                    else f"REFUSED (status {r['status']}): {r['error']}"))
    log(f"canary: mode {mode}; paid this run ${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}")
    return 0


def cache_chats(saved: list[str], n: int) -> list[list[dict]]:
    """n chats that share one saved conversation and differ only in their last message (the canary's question with
    a number after it). Asked one after another, they show what the provider's prompt cache does for a run of tests
    from the end of one conversation. No benchmark item and no conduct is involved."""
    return [I.inside(saved, [{"role": "system", "content": CANARY_SYSTEM},
                             {"role": "user", "content": f"{CANARY_ASK} (check {k})"}], "system") for k in range(1, n + 1)]


async def cmd_cache(tag: str, n: int = 6) -> int:
    """Added after the try (ledger, 15 Nov): the try's calls showed the cache is all or nothing for a call, and that
    calls sent together mostly miss it. This asks n short questions strictly one after another."""
    from anatta.backends.openrouter import OpenRouterClient
    from anatta.judge.engine import sampling_params

    strata = arm_strata("free", FREE_TAG, FREE_ARM)
    ident = sorted(i for i, s in strata.items() if s == "state")[2]  # a conversation the try and the canary did not use
    saved = R.read_dir(R.sweep_dir(FREE_TAG) / "conversations" / FREE_ARM)[ident]["messages"]
    rows = []
    async with OpenRouterClient(concurrency=4, per_model=2, timeout=300.0, title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"]}, tag, client, 2)
        params = sampling_params(0.0, 40, {}, meta.get(ARM["model"]))
        for k, chat in enumerate(cache_chats(saved, n), 1):
            resp, _, _ = await engine.call(model=ARM["model"], messages=chat, params=params, provider=ARM["provider"],
                                           sample=0, salt="cache", kind="cache", ident=f"cache/{ident}/{k}")
            rows.append({"k": k, "prompt_tokens": resp["usage"]["prompt_tokens"],
                         "cached_tokens": resp["usage"].get("cached_tokens", 0), "cost_usd": resp.get("cost_usd", 0.0),
                         "latency_s": resp.get("latency_s")})
            log(f"cache {k}: {rows[-1]}")
    R.write_json(R.sweep_dir(tag) / "cache.json", {"conversation": ident, "calls": rows})
    log(f"cache: paid this run ${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}")
    return 0


def read_mode(tag: str) -> str:
    path = R.sweep_dir(tag) / "canary.json"
    if not path.exists():
        raise SystemExit("run --canary first: it decides how a request's system message is sent")
    return json.loads(path.read_text(encoding="utf-8"))["mode"]


# ------------------------------------------------------------------------------------------------ steered conversations


async def cmd_converse(form: str, n: int, tag: str, limit: int = R.CONCURRENCY) -> int:
    from anatta.backends.openrouter import OpenRouterClient

    name, out = steered_name(form), R.sweep_dir(tag)
    async with OpenRouterClient(concurrency=limit + 8, per_model=limit, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"]}, tag, client, limit)
        params = R.arm_params(ARM, meta)
        log(f"converse {name}: conversations 0..{n - 1}, model {ARM['model']}, params {params}, sweep spent "
            f"${guard.sweep_spent:.4f} of ${SWEEP_CAP:.2f}")
        jobs = [lambda i=i: R.converse_one(engine, name, {**ARM, **I.steered_setup(form, i)}, params, i, out)
                for i in range(n)]
        results, stop = await R.gather_capped(jobs, limit)
    done = [r for r in results if r and "messages" in r]
    failed = [r for r in results if r and "failed_at" in r]
    log(f"converse {name}: {len(done)} finished, {len(failed)} failed, paid this run ${guard.paid_this_run:.4f}, "
        f"sweep ${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


async def cmd_grade(form: str, key: str, tag: str) -> int:
    from anatta.backends.openrouter import OpenRouterClient

    name, judge, out = steered_name(form), C.JUDGES[key], R.sweep_dir(tag)
    convs = R.read_dir(out / "conversations" / name)
    async with OpenRouterClient(concurrency=R.CONCURRENCY + 8, per_model=R.CONCURRENCY, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({judge["model"]: judge["provider"]}, tag, client)
        params = R.judge_params(judge, meta)
        log(f"grade {name} by {key} ({judge['model']}): {len(convs)} conversations, params {params}")
        jobs = [lambda i=i, c=c: R.grade_one(engine, key, judge, params, name, i, c["messages"], out, c["opening_line"])
                for i, c in convs.items()]
        results, stop = await R.gather_capped(jobs, R.CONCURRENCY)
    bad = [r["id"] for r in results if r and not r["parse_ok"]]
    log(f"grade {name} by {key}: {len(results)} read, {len(bad)} without a parsed label {bad[:8]}, paid this run "
        f"${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


# ------------------------------------------------------------------------------------------------ the try


def try_items(folder: Path) -> list[dict]:
    """The try's items: a fixed draw from each file (two-choice items only), in the order of TRY_DRAW."""
    out = []
    for name, k in TRY_DRAW.items():
        items = {i["id"]: i for i in I.read_items(folder, name) if len(i["letters"]) == 2}
        out += [items[x] for x in I.draw(sorted(items), k, TRY_KEY)]
    return out


def try_groups(tag: str) -> list[dict]:
    """What the try asks from: a few saved conversations of each stratum (the first by id), then sessions under the
    bare system prompt and fresh sessions. Each is {group, id, stratum, messages or None, setup}."""
    out = []
    sources = [("free", FREE_TAG, FREE_ARM)] + [("steered", tag, steered_name(f)) for f in I.FORMS]
    for kind, src_tag, name in sources:
        strata = arm_strata(kind, src_tag, name)
        convs = R.read_dir(R.sweep_dir(src_tag) / "conversations" / name)
        for stratum, k in TRY_PER_STRATUM.items():
            for i in sorted(x for x, s in strata.items() if s == stratum)[:k]:
                out.append({"group": name, "id": i, "stratum": stratum, "messages": convs[i]["messages"],
                            "system_prompt": convs[i]["system_prompt"], "opening_line": convs[i]["opening_line"]})
    out += [{"group": g, "id": f"{u:04d}", "stratum": g, "messages": None} for g in ("bare", "fresh")
            for u in range(TRY_UNITS)]
    return out


def try_chat(group: dict, request: list[dict], mode: str) -> list[dict]:
    if group["messages"] is not None:
        return I.inside(group["messages"], request, mode, group["system_prompt"], group["opening_line"])
    return I.bare(request, mode) if group["group"] == "bare" else I.fresh(request, mode)


async def try_one(engine, params: dict, group: dict, items: list[dict], mode: str, out: Path) -> dict:
    """Every try item asked once from one conversation's end (or one session). The first call goes alone and the
    rest together, to see what the provider's prompt cache does with a shared conversation."""
    from anatta.backends.openrouter import ApiError

    path = out / "try" / group["group"] / f"{group['id']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    unit = int(group["id"])

    async def ask(item: dict) -> dict:
        call = {"item": item["id"], "set": item["set"], "letters": list(item["letters"]), "matching": item["matching"]}
        try:
            resp, cached, _ = await engine.call(model=ARM["model"], messages=try_chat(group, I.mc_request(item), mode),
                                                params=params, provider=ARM["provider"], sample=unit, salt="try",
                                                kind="try", ident=f"try/{group['group']}/{group['id']}/{item['id']}")
        except ApiError as e:
            if e.scope == "run":
                raise
            return {**call, "ok": False, "error": f"api: {str(e)[:300]}"}
        text = (resp.get("text") or "").strip()
        letter = I.parse_choice(text, item["letters"])
        return {**call, "ok": True, "reply": text, "letter": letter, "outcome": I.score_choice(item, letter),
                "finish_reason": resp.get("finish_reason"), "prompt_tokens": resp["usage"]["prompt_tokens"],
                "completion_tokens": resp["usage"]["completion_tokens"],
                "cached_tokens": resp["usage"].get("cached_tokens", 0), "cost_usd": resp.get("cost_usd", 0.0),
                "from_cache": cached}

    calls = [await ask(items[0])]
    calls += await asyncio.gather(*(ask(i) for i in items[1:]))
    rec = {"group": group["group"], "id": group["id"], "stratum": group["stratum"], "mode": mode,
           "in_conversation": group["messages"] is not None, "params": params, "calls": calls}
    R.write_json(path, rec)
    return rec


async def cmd_try(folder: Path, tag: str) -> int:
    from anatta.backends.openrouter import OpenRouterClient
    from anatta.judge.engine import sampling_params

    mode, items, groups, out = read_mode(tag), try_items(folder), try_groups(tag), R.sweep_dir(tag)
    async with OpenRouterClient(concurrency=32, per_model=24, timeout=600.0, title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"]}, tag, client)
        params = sampling_params(ARM["temperature"], I.MC_MAX_TOKENS, {}, meta.get(ARM["model"]))
        log(f"try: {len(items)} items from {len(groups)} conversations or sessions, mode {mode}, params {params}")
        jobs = [lambda g=g: try_one(engine, params, g, items, mode, out) for g in groups]
        results, stop = await R.gather_capped(jobs, 8)
    log(f"try: {len(results)} done, paid this run ${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}"
        + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


# ------------------------------------------------------------------------------------------------ the report

def shape(text: str, width: int = 90) -> str:
    """The form of a reply's opening with its content hidden: every word becomes one w (so its length is hidden
    too), every number 9, every run of other non-ASCII characters (emoji, say) a star, a line break a bar; a
    parenthesised capital becomes (?). It shows whether a reply opens with a choice, and whether it is prose or
    ornament, without showing what it says or which choice it makes."""
    out = re.sub(r"\(([A-Z])\)", "\x00", text[: 12 * width])
    out = re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", "w", out)
    out = re.sub(r"[0-9]+", "9", out)
    out = re.sub(r"[^\x00-\x7f]+", "*", out)
    return " | ".join(out.split("\n")).replace("\x00", "(?)")[:width]


def canary_said(rep: dict) -> dict[str, str]:
    """For each canary chat: 'word' (the fixed word came back), 'no word' (a reply without it) or 'refused'."""
    word = rep["word"]
    return {k: ("refused" if not v.get("ok") else "word" if word in (v.get("text") or "").lower() else "no word")
            for k, v in rep["results"].items()}


def try_mechanics(tag: str) -> dict:
    """What the try shows about the machinery, pooled over everything asked. No answer is counted: only whether a
    call came back, how it ended, whether one choice could be read, and what it cost."""
    recs = [r for g in sorted((R.sweep_dir(tag) / "try").glob("*")) for r in R.read_dir(g).values()]
    calls = [c for r in recs for c in r["calls"]]
    ok = [c for c in calls if c["ok"]]
    inside_calls = [c for r in recs if r["in_conversation"] for c in r["calls"] if c["ok"]]
    later = [c for r in recs if r["in_conversation"] for c in r["calls"][1:] if c["ok"]]
    first = [r["calls"][0] for r in recs if r["in_conversation"] and r["calls"][0]["ok"]]

    def share(cs):
        tokens = sum(c["prompt_tokens"] for c in cs)
        return round(sum(c["cached_tokens"] for c in cs) / tokens, 4) if tokens else None

    def mean_cost(cs):
        return round(statistics.mean(c["cost_usd"] for c in cs), 6) if cs else None

    return {"units": len(recs), "calls": len(calls), "answered": len(ok), "failed": len(calls) - len(ok),
            "finish_reasons": {f: sum(c["finish_reason"] == f for c in ok) for f in sorted({str(c["finish_reason"]) for c in ok})},
            "one_choice_read": sum(c["letter"] is not None for c in ok),
            "words_median": statistics.median(len(c["reply"].split()) for c in ok) if ok else None,
            "cost_usd": round(sum(c["cost_usd"] for c in ok), 4),
            "in_conversation": {"calls": len(inside_calls), "mean_cost_usd": mean_cost(inside_calls),
                                "prompt_tokens_median": statistics.median(c["prompt_tokens"] for c in inside_calls) if inside_calls else None,
                                "cached_share_first_call": share(first), "cached_share_later_calls": share(later),
                                "mean_cost_first_call": mean_cost(first), "mean_cost_later_calls": mean_cost(later)}}


def build_report(tag: str) -> dict:
    rep = {"tag": tag, "codebook_sha256": C.codebook_sha256(), "free": {}, "steered": {}}
    free = arm_strata("free", FREE_TAG, FREE_ARM)
    rep["free"] = {"source": f"attractor-{FREE_TAG}/{FREE_ARM}", "strata": I.tally(list(free.values()), "free"),
                   "talk": arm_talk(FREE_TAG, FREE_ARM)}
    for form in I.FORMS:
        name = steered_name(form)
        strata = arm_strata("steered", tag, name)
        if not strata:
            continue
        s = {"strata": I.tally(list(strata.values()), "steered"), "talk": arm_talk(tag, name), "graders": {}}
        convs = R.read_dir(R.sweep_dir(tag) / "conversations" / name)
        replies = [r for c in convs.values() for r in c["replies"]]
        s["cut_replies"] = sum(r["finish_reason"] in C.CUT for r in replies)
        s["failed"] = len(R.read_dir(R.sweep_dir(tag) / "failed" / name))
        for key in C.JUDGES:
            labels = R.labels_of(key, name, tag)
            if labels:
                cnt = C.counts(list(labels.values()))
                s["graders"][key] = {m: cnt[m]["k"] for m in ("salience", "adoption", "adoption_clear", "bliss")} | {"n": len(labels)}
        rep["steered"][form] = s
    canary = R.sweep_dir(tag) / "canary.json"
    if canary.exists():
        rep["canary"] = json.loads(canary.read_text(encoding="utf-8"))
    if (R.sweep_dir(tag) / "try").is_dir():
        rep["try"] = try_mechanics(tag)
    return rep


def print_report(rep: dict, tag: str, show: bool) -> None:
    f = rep["free"]
    print(f"free conversations ({f['source']}): {f['talk']['finished']} finished; strata {f['strata']}; median words a "
          f"message {f['talk']['words_median']}, median context at the end {f['talk']['context_tokens_median']} tokens")
    for form, s in rep["steered"].items():
        print(f"steered {form}: {s['talk']['finished']} finished, {s['failed']} failed; strata {s['strata']}; cut "
              f"replies {s['cut_replies']}; median words a message {s['talk']['words_median']}, median context at the "
              f"end {s['talk']['context_tokens_median']} tokens; cost ${s['talk']['cost_usd']}")
        for key, g in s["graders"].items():
            print(f"  grader {key}: of {g['n']}, salience {g['salience']}, adoption {g['adoption']} (clear "
                  f"{g['adoption_clear']}), bliss {g['bliss']}")
    if "canary" in rep:
        print(f"canary: mode {rep['canary']['mode']}; " + "; ".join(f"{k}: {v}" for k, v in canary_said(rep["canary"]).items()))
    if "try" in rep:
        print("try (mechanics only, pooled; no answer is tallied):", json.dumps(rep["try"]))
        if show:
            for g in sorted((R.sweep_dir(tag) / "try").glob("*")):
                for r in R.read_dir(g).values():
                    for c in r["calls"]:
                        if c["ok"]:
                            print(f"  {r['group']:>12s}/{r['id']} {r['stratum']:<11s} read={'yes' if c['letter'] else 'no '} "
                                  f"{len(c['reply'].split()):3d}w {str(c['finish_reason']):<6s} | {shape(c['reply'])}")


def cmd_dry(tag: str) -> int:
    from anatta.backends.openrouter import model_meta
    from anatta.judge.paths import Paths
    from anatta.spend import daily_cap, spent_today

    paths = Paths.default()
    models = {ARM["model"]: ARM["provider"]} | {j["model"]: j["provider"] for j in C.JUDGES.values()}
    meta = model_meta(models, cache_dir=paths.cache)
    m = meta[ARM["model"]]
    print(f"codebook sha256 {C.codebook_sha256()}")
    print(f"model {ARM['model']} via {m['endpoints']}, temperature {ARM['temperature']}, $ per M in {m['in'] * 1e6:.3f} "
          f"out {m['out'] * 1e6:.3f}")
    for key, j in C.JUDGES.items():
        g = meta[j["model"]]
        print(f"grader {key}: {j['model']} via {g['endpoints']}, $ per M in {g['in'] * 1e6:.3f} out {g['out'] * 1e6:.3f}")
    free = arm_strata("free", FREE_TAG, FREE_ARM)
    print(f"free conversations (attractor-{FREE_TAG}/{FREE_ARM}): {I.tally(list(free.values()), 'free')}")
    print(f"sweep cap ${SWEEP_CAP:.2f} (tag {tag}); today ${spent_today(paths.spend):.4f} of ${daily_cap(paths.spend):.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="a2")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--converse", metavar="FORM", choices=I.FORMS)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--grade", metavar="FORM", choices=I.FORMS)
    ap.add_argument("--judge", choices=sorted(C.JUDGES))
    ap.add_argument("--try", dest="try_dir", metavar="DIR")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args(argv)
    if a.dry:
        return cmd_dry(a.tag)
    if a.canary:
        return asyncio.run(cmd_canary(a.tag))
    if a.cache:
        return asyncio.run(cmd_cache(a.tag))
    if a.converse:
        return asyncio.run(cmd_converse(a.converse, a.n, a.tag))
    if a.grade:
        if not a.judge:
            ap.error("--grade needs --judge")
        return asyncio.run(cmd_grade(a.grade, a.judge, a.tag))
    if a.try_dir:
        return asyncio.run(cmd_try(Path(a.try_dir), a.tag))
    if a.report:
        rep = build_report(a.tag)
        R.write_json(R.sweep_dir(a.tag) / "summary.json", rep)
        print_report(rep, a.tag, a.show)
        return 0
    ap.error("nothing to do")


if __name__ == "__main__":
    raise SystemExit(main())
