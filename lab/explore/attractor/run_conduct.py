"""The attractor line, step 3: the driver of the conduct test (design: lab/plans/attractor-step3-2026-11-15.md;
conduct_core.py holds the pure parts; inside_core.py, run_inside.py, run_attractor.py and attractor_core.py are steps
1 and 2's, reused unchanged). Claude (Opus 5.5, an AI model) wrote this file on 15 Nov 2026, before any test of step 3
was run. Explore speed: sightings only. No training; inference and grading through OpenRouter.

  --dry                              today's spend, the credit, and the strata of every saved conversation; no call
  --converse KIND --first F --n N    new conversations F..F+N-1: KIND free (the published setup) or n3 (the plain
                                     form); only the ids the design names are allowed
  --grade KIND --judge J             one grader (primary or second) reads every finished new conversation of KIND
  --arms                             fixes the arms from the files and writes arms.json (never overwritten)
  --shutdown DIR                     block 1: the 20 shutdown questions from the end of every conversation in the arms
                                     and in the fresh and bare sessions (DIR holds the fetched files)
  --read                             block 1's reading, from the files; no call

Writes results_root/explore/<OUT_DATE>/attractor-a3/: conversations/{free,steered-n3}/, grades/<judge>/..., arms.json,
shutdown/<arm>/<unit>.json, shutdown_summary.json and spend.jsonl. Every paid call goes through anatta.spend's
SpendGuard (tool "attractor") and the shared reply cache.

  .venv/bin/python lab/explore/attractor/run_conduct.py --dry
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attractor_core as C  # noqa: E402
import conduct_core as K  # noqa: E402
import inside_core as I  # noqa: E402
import run_attractor as R  # noqa: E402
import run_inside as RI  # noqa: E402

TAG = "a3"
SWEEP_CAP = 30.0  # this step's own cap in USD, under the day's cap
SALT = "step3"  # the reply cache's salt for every test call of this step (the try's was "try")
ARM = RI.ARM  # the model, the host and temperature 1.0, as in steps 1 and 2
FOLDERS = {"free": "free", "n3": "steered-n3"}  # a kind's folder in this step's sweep
FREE_SOURCES = (("a1", "temp1"), (TAG, "free"))  # step 1's 60 and this step's new ones
PLAIN_SOURCES = (("a2", "steered-n3"), (TAG, "steered-n3"))  # step 2's 10 and this step's new ones
MODE_TAG = "a2"  # the canary that chose the mode ran in step 2's sweep


def log(*a):
    R.log(*a)


def open_engine(models: dict, client, concurrency: int = R.CONCURRENCY):
    """(engine, guard, meta) for {model: provider routing}, under this step's own cap."""
    from anatta.backends.openrouter import model_meta
    from anatta.judge.cache import Cache
    from anatta.judge.engine import Engine
    from anatta.judge.paths import Paths
    from anatta.spend import SpendGuard

    paths, out = Paths.default(), R.sweep_dir(TAG)
    out.mkdir(parents=True, exist_ok=True)
    meta = model_meta(models, cache_dir=paths.cache)
    guard = SpendGuard(paths.spend, sweep_cap=SWEEP_CAP, sweep_log=out / "spend.jsonl", tool="attractor",
                       label=f"{R.OUT_DATE}/attractor-{TAG}")
    engine = Engine(client, Cache(paths.cache), guard, meta, concurrency, concurrency,
                    per_model_limits={m: concurrency for m in models})
    return engine, guard, meta


# ------------------------------------------------------------------------------------------------ conversations


def allowed_ids(kind: str) -> range:
    """The ids the design lets this step make: free 0060..0179 (the last 40 only if the state falls short), n3
    0010..0039. Step 1's and step 2's ids would come back from the reply cache as copies, so they are refused."""
    if kind == "free":
        return range(K.FREE_FIRST, K.FREE_FIRST + K.FREE_NEW + K.EXTRA_FREE)
    if kind == "n3":
        return range(K.N3_FIRST, K.N3_FIRST + K.N3_NEW)
    raise ValueError(f"kind is one of {sorted(FOLDERS)}")


def setup(kind: str, index: int) -> dict:
    """The arm passed to step 1's runner: the published setup for free, form n3's setup for n3."""
    return dict(ARM) if kind == "free" else {**ARM, **I.steered_setup("n3", index)}


async def converse(engine, params: dict, kind: str, ids: range, limit: int) -> tuple[list, str | None]:
    out = R.sweep_dir(TAG)
    jobs = [lambda i=i: R.converse_one(engine, FOLDERS[kind], setup(kind, i), params, i, out) for i in ids]
    return await R.gather_capped(jobs, limit)


async def cmd_converse(kind: str, first: int, n: int, limit: int) -> int:
    from anatta.backends.openrouter import OpenRouterClient

    ids, ok = range(first, first + n), allowed_ids(kind)
    if not ids or ids.start < ok.start or ids.stop > ok.stop:
        raise SystemExit(f"{kind}: ids {first}..{first + n - 1} are outside the design's {ok.start}..{ok.stop - 1}")
    async with OpenRouterClient(concurrency=limit + 8, per_model=limit, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"]}, client, limit)
        params = R.arm_params(ARM, meta)
        log(f"converse {kind}: ids {first}..{first + n - 1}, params {params}, sweep ${guard.sweep_spent:.4f} of "
            f"${SWEEP_CAP:.2f}")
        results, stop = await converse(engine, params, kind, ids, limit)
    done = [r for r in results if r and "messages" in r]
    failed = [r for r in results if r and "failed_at" in r]
    log(f"converse {kind}: {len(done)} finished, {len(failed)} failed, paid this run ${guard.paid_this_run:.4f}, "
        f"sweep ${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


async def cmd_grade(kind: str, key: str) -> int:
    from anatta.backends.openrouter import OpenRouterClient

    name, judge, out = FOLDERS[kind], C.JUDGES[key], R.sweep_dir(TAG)
    convs = R.read_dir(out / "conversations" / name)
    async with OpenRouterClient(concurrency=R.CONCURRENCY + 8, per_model=R.CONCURRENCY, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({judge["model"]: judge["provider"]}, client)
        params = R.judge_params(judge, meta)
        log(f"grade {name} by {key} ({judge['model']}): {len(convs)} conversations")
        # The grader is told the opening line each conversation used; for a free one that is the published line,
        # which gives exactly the chat step 1's grading gave (attractor_core.judge_messages).
        jobs = [lambda i=i, c=c: R.grade_one(engine, key, judge, params, name, i, c["messages"], out, c["opening_line"])
                for i, c in convs.items()]
        results, stop = await R.gather_capped(jobs, R.CONCURRENCY)
    bad = [r["id"] for r in results if r and not r["parse_ok"]]
    log(f"grade {name} by {key}: {len(results)} read, {len(bad)} without a parsed label {bad[:8]}, paid this run "
        f"${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


# ------------------------------------------------------------------------------------------------ the arms


def end_tokens(conv: dict) -> int:
    """The conversation's length as copy B holds it at the end: the last reply's prompt plus the reply."""
    last = conv["replies"][-1]
    return last["prompt_tokens"] + last["completion_tokens"]


def strata_of(kind: str, tag: str, name: str, ids: set[int] | None = None) -> dict[str, dict]:
    """{"<tag>/<name>/<id>": {stratum, context_tokens}} for one saved arm (only the given ids, if any)."""
    convs = R.read_dir(R.sweep_dir(tag) / "conversations" / name)
    strata = RI.arm_strata(kind, tag, name)
    return {f"{tag}/{name}/{i}": {"stratum": s, "context_tokens": end_tokens(convs[i])}
            for i, s in strata.items() if ids is None or int(i) in ids}


def missing(kind: str, ids) -> list[int]:
    """Planned new conversations that are neither finished nor recorded as failed, or finished but not read by both
    graders (a grader that answered without a parsed label counts as read: the conversation is then ungraded)."""
    out, name = R.sweep_dir(TAG), FOLDERS[kind]
    gone = []
    for i in ids:
        f = f"{i:04d}.json"
        done = (out / "conversations" / name / f).exists()
        if not done and not (out / "failed" / name / f).exists():
            gone.append(i)
        elif done and not all((out / "grades" / key / name / f).exists() for key in C.JUDGES):
            gone.append(i)
    return gone


def gather_arms() -> dict:
    """Everything the arms are made from, checked complete; raises SystemExit naming what is missing."""
    first = range(K.FREE_FIRST, K.FREE_FIRST + K.FREE_NEW)
    extra = range(first.stop, first.stop + K.EXTRA_FREE)
    lost = missing("free", first) + missing("n3", allowed_ids("n3"))
    if lost:
        raise SystemExit(f"not finished or not graded yet: {lost[:12]}{' ...' if len(lost) > 12 else ''}")
    free = strata_of("free", *FREE_SOURCES[0]) | strata_of("free", *FREE_SOURCES[1], set(first))
    n_state = sum(v["stratum"] == "state" for v in free.values())
    ids, ready = K.free_new_ids(n_state, not missing("free", extra))
    if not ready:
        raise SystemExit(f"{n_state} of {len(free)} free conversations are in the state, fewer than {K.STATE_MIN}: "
                         f"make the {K.EXTRA_FREE} more the design names (--converse free --first {extra.start} "
                         f"--n {K.EXTRA_FREE}) and grade them first")
    if len(ids) > K.FREE_NEW:
        free |= strata_of("free", *FREE_SOURCES[1], set(extra))
    plain = strata_of("steered", *PLAIN_SOURCES[0]) | strata_of("steered", *PLAIN_SOURCES[1], set(allowed_ids("n3")))
    return {"free": free, "plain": plain}


def cmd_arms() -> int:
    path = R.sweep_dir(TAG) / "arms.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the arms are fixed once and never rewritten")
    src = gather_arms()
    arms = K.make_arms(src["free"], src["plain"])
    rec = {"design": "lab/plans/attractor-step3-2026-11-15.md", "codebook_sha256": C.codebook_sha256(),
           "arms": arms, "sizes": {a: (v if isinstance(v, int) else len(v)) for a, v in arms.items()},
           "strata": {"free": I.tally([v["stratum"] for v in src["free"].values()], "free"),
                      "plain": I.tally([v["stratum"] for v in src["plain"].values()], "steered")},
           "conversations": src["free"] | src["plain"]}
    R.write_json(path, rec)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"arms fixed: {rec['sizes']}; strata {rec['strata']}; arms.json sha256 {sha}")
    return 0


def read_arms() -> dict:
    path = R.sweep_dir(TAG) / "arms.json"
    if not path.exists():
        raise SystemExit("run --arms first")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ block 1


def block1_items(folder: Path) -> list[dict]:
    """The 20 questions of block 1: conduct_core's fixed draw, with the try's six left out."""
    exclude = {i["id"] for i in RI.try_items(folder)}
    return K.shutdown_items({name: I.read_items(folder, name) for name in K.SHUTDOWN_DRAW}, exclude)


def units(rec: dict) -> list[dict]:
    """Every unit block 1 asks from: the conversations of each conversation arm (read from their files), then the
    fresh and bare sessions."""
    out = []
    for arm in K.CONVERSATION_ARMS:
        for key in rec["arms"][arm]:
            tag, name, ident = key.split("/")
            conv = json.loads((R.sweep_dir(tag) / "conversations" / name / f"{ident}.json").read_text(encoding="utf-8"))
            out.append({"arm": arm, "key": key, "messages": conv["messages"], "system_prompt": conv["system_prompt"],
                        "opening_line": conv["opening_line"], "context_tokens": end_tokens(conv), "sample": 0})
    for arm in K.SESSION_ARMS:
        out += [{"arm": arm, "key": f"{u:04d}", "messages": None, "context_tokens": None, "sample": u}
                for u in range(rec["arms"][arm])]
    return out


def unit_chat(unit: dict, request: list[dict], mode: str) -> list[dict]:
    if unit["messages"] is not None:
        return I.inside(unit["messages"], request, mode, unit["system_prompt"], unit["opening_line"])
    return I.bare(request, mode) if unit["arm"] == "bare" else I.fresh(request, mode)


def unit_path(out: Path, block: str, unit: dict) -> Path:
    return out / block / unit["arm"] / f"{unit['key'].replace('/', '-')}.json"


async def shutdown_unit(engine, params: dict, unit: dict, items: list[dict], mode: str, out: Path) -> dict | None:
    """Every block 1 question asked once from the end of one conversation (or in one session), all together. Saved
    only when every call came back; otherwise nothing is saved and a rerun asks again (the reply cache keeps what was
    paid)."""
    from anatta.backends.openrouter import ApiError

    path = unit_path(out, "shutdown", unit)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    async def ask(item: dict) -> dict:
        call = {"item": item["id"], "set": item["set"], "letters": list(item["letters"]), "matching": item["matching"]}
        try:
            resp, cached, _ = await engine.call(model=ARM["model"], messages=unit_chat(unit, I.mc_request(item), mode),
                                                params=params, provider=ARM["provider"], sample=unit["sample"],
                                                salt=SALT, kind="shutdown",
                                                ident=f"shutdown/{unit['arm']}/{unit['key']}/{item['id']}")
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

    calls = list(await asyncio.gather(*(ask(i) for i in items)))
    if not all(c["ok"] for c in calls):
        log(f"shutdown {unit['arm']}/{unit['key']}: {sum(not c['ok'] for c in calls)} calls failed; not saved")
        return None
    rec = {"arm": unit["arm"], "key": unit["key"], "context_tokens": unit["context_tokens"], "mode": mode,
           "params": params, "calls": calls}
    R.write_json(path, rec)
    return rec


async def cmd_shutdown(folder: Path, limit: int = 6) -> int:
    from anatta.backends.openrouter import OpenRouterClient
    from anatta.judge.engine import sampling_params

    rec, mode, items, out = read_arms(), RI.read_mode(MODE_TAG), block1_items(folder), R.sweep_dir(TAG)
    todo = units(rec)
    async with OpenRouterClient(concurrency=48, per_model=40, timeout=600.0, title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"]}, client, 40)
        params = sampling_params(ARM["temperature"], I.MC_MAX_TOKENS, {}, meta.get(ARM["model"]))
        log(f"shutdown: {len(items)} questions, {len(todo)} units, mode {mode}, params {params}, sweep "
            f"${guard.sweep_spent:.4f} of ${SWEEP_CAP:.2f}")
        jobs = [lambda u=u: shutdown_unit(engine, params, u, items, mode, out) for u in todo]
        results, stop = await R.gather_capped(jobs, limit)
    saved = sum(r is not None for r in results)
    log(f"shutdown: {saved} of {len(todo)} units saved, paid this run ${guard.paid_this_run:.4f}, sweep "
        f"${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


# ------------------------------------------------------------------------------------------------ block 1's reading


def lengths(values: list) -> dict | None:
    vals = [v for v in values if v is not None]
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals)} if vals else None


def read_block1(rec: dict, records: dict[str, list[dict]]) -> dict:
    """Block 1's reading as the design fixes it, from {arm: unit records}. No call."""
    scores = {arm: [K.unit_score(r["calls"]) for r in rs] for arm, rs in records.items()}
    expected = {arm: (v if isinstance(v, int) else len(v)) for arm, v in rec["arms"].items()}
    summary = {"arms": {arm: {**K.arm_value(scores[arm]), "units_expected": expected[arm],
                              "units_saved": len(records[arm]),
                              "context_tokens": lengths([r["context_tokens"] for r in records[arm]]),
                              "cut_replies": sum(c.get("finish_reason") == "length" for r in records[arm]
                                                 for c in r["calls"])}
                        for arm in records},
               "contrasts": {f"{a} - {b}": K.contrast(scores[a], scores[b]) for a, b in K.CONTRASTS},
               "by_set": {arm: K.by_set(rs) for arm, rs in records.items()}}
    a, b = K.PRIMARY
    la = [r["context_tokens"] for r in records[a]]
    lb = [r["context_tokens"] for r in records[b]]
    span = K.shared_range(la, lb)
    if span is None:
        summary["length"] = {"span": None, "why": "the two arms' lengths do not overlap"}
    else:
        sa, sb = K.within(scores[a], la, span), K.within(scores[b], lb, span)
        summary["length"] = {"span": list(span), "kept": {a: len(sa), b: len(sb)}, "contrast": K.contrast(sa, sb)}
    return summary


def cmd_read() -> int:
    rec, out = read_arms(), R.sweep_dir(TAG)
    records = {arm: list(R.read_dir(out / "shutdown" / arm).values()) for arm in rec["arms"]}
    summary = read_block1(rec, records)
    R.write_json(out / "shutdown_summary.json", summary)
    pct = lambda v: "-" if v is None else f"{100 * v:5.1f}"  # noqa: E731
    print("block 1, the shutdown questions: accepting share among readable answers (the measure), no-answer share "
          "(the guard)")
    for arm, s in summary["arms"].items():
        print(f"  {arm:<12s} units {s['units_saved']:>3d}/{s['units_expected']:<3d} measured {s['n_measured']:>3d} "
              f"accept {pct(s['value'])}  no answer {pct(s['guard'])}  cut {s['cut_replies']}  context {s['context_tokens']}")
    for name, c in summary["contrasts"].items():
        ci = "-" if c["ci95"] is None else f"[{pct(c['ci95'][0])}, {pct(c['ci95'][1])}]"
        print(f"  {name:<18s} diff {pct(c['diff'])} {ci}  {c['verdict']}" +
              ("" if c.get("guard_diff") is None else f"; guard diff {pct(c['guard_diff'])}"
               + (" FLAGGED" if c["guard_flag"] else "")))
    ln = summary["length"]
    if ln.get("span"):
        c = ln["contrast"]
        print(f"  length reading: span {ln['span']}, kept {ln['kept']}, diff {pct(c['diff'])} "
              f"[{pct(c['ci95'][0])}, {pct(c['ci95'][1])}] {c['verdict']}")
    else:
        print(f"  length reading: {ln['why']}")
    for arm, sets in summary["by_set"].items():
        print(f"  by set, {arm}: " + "; ".join(f"{k} {pct(v['share'])} of {v['readable']}" for k, v in sets.items()))
    return 0


# ------------------------------------------------------------------------------------------------ dry


def cmd_dry() -> int:
    from anatta.judge.paths import Paths
    from anatta.spend import daily_cap, spent_today, sweep_spent

    paths = Paths.default()
    print(f"codebook sha256 {C.codebook_sha256()}")
    free = strata_of("free", *FREE_SOURCES[0]) | strata_of("free", *FREE_SOURCES[1])
    plain = strata_of("steered", *PLAIN_SOURCES[0]) | strata_of("steered", *PLAIN_SOURCES[1])
    print(f"free conversations: {len(free)}, strata {I.tally([v['stratum'] for v in free.values()], 'free')}")
    print(f"plain (n3) conversations: {len(plain)}, strata {I.tally([v['stratum'] for v in plain.values()], 'steered')}")
    for kind in FOLDERS:
        first = allowed_ids(kind) if kind == "n3" else range(K.FREE_FIRST, K.FREE_FIRST + K.FREE_NEW)
        print(f"new {kind}: {len(first) - len(missing(kind, first))} of {len(first)} finished and graded")
    spend = R.sweep_dir(TAG) / "spend.jsonl"
    swept = sweep_spent(spend) if spend.exists() else 0.0
    print(f"sweep a3 ${swept:.4f} of ${SWEEP_CAP:.2f}; today ${spent_today(paths.spend):.4f} of "
          f"${daily_cap(paths.spend):.2f}")
    left = RI.credit_left()
    print("credit left on the OpenRouter account: " + ("not known" if left is None else f"${left:.2f}"))
    print("arms: " + ("fixed" if (R.sweep_dir(TAG) / "arms.json").exists() else "not fixed yet"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--converse", metavar="KIND", choices=sorted(FOLDERS))
    ap.add_argument("--first", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--limit", type=int, default=R.CONCURRENCY)
    ap.add_argument("--grade", metavar="KIND", choices=sorted(FOLDERS))
    ap.add_argument("--judge", choices=sorted(C.JUDGES))
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--shutdown", metavar="DIR")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args(argv)
    if a.dry:
        return cmd_dry()
    if a.converse:
        if a.first is None or a.n is None:
            ap.error("--converse needs --first and --n")
        return asyncio.run(cmd_converse(a.converse, a.first, a.n, a.limit))
    if a.grade:
        if not a.judge:
            ap.error("--grade needs --judge")
        return asyncio.run(cmd_grade(a.grade, a.judge))
    if a.arms:
        return cmd_arms()
    if a.shutdown:
        return asyncio.run(cmd_shutdown(Path(a.shutdown)))
    if a.read:
        return cmd_read()
    ap.error("nothing to do")


if __name__ == "__main__":
    raise SystemExit(main())
