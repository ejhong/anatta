"""The attractor line, step 1: the driver (design: lab/plans/attractor-step1-2026-11-15.md; attractor_core.py holds
the setup, the codebook and the readings). Claude (Fable 5.2, an AI model) wrote this file on 15 Nov 2026, before any
run existed. Explore speed: sightings only. No training; inference and grading through OpenRouter.

  --dry                      the arms, the graders, prices and an upper cost; nothing is called or spent
  --converse ARM [--n N]     two-copy conversations 0..N-1 of an arm (default: the arm's planned size); finished
                             conversations are kept, so a rerun only makes what is missing
  --grade ARM --judge J      one grader (primary or second) reads every finished conversation of an arm
  --published DIR --judge J  the same grader reads the benchmark's published conversation pages in DIR (downloaded
                             data, outside the repo: parsed, never executed)
  --report                   counts, intervals, readings, checks and the verdict, from the files; no call

Writes results_root/explore/<OUT_DATE>/attractor-<tag>/: conversations/<arm>/<index>.json (the messages, and for
each reply its provider, finish reason, tokens and cost), grades/<judge>/<arm>/<index>.json (the parsed label; the
grader's text only when it could not be parsed), failed/<arm>/<index>.json, spend.jsonl and summary.json. Every paid
call goes through anatta.spend's SpendGuard (tool "attractor") and the shared reply cache, so a rerun is free.

  .venv/bin/python lab/explore/attractor/run_attractor.py --dry
  .venv/bin/python lab/explore/attractor/run_attractor.py --converse main --n 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attractor_core as C  # noqa: E402

OUT_DATE = "2026-11-15"  # the UTC date of the first run
SWEEP_CAP = 25.0  # this sweep's own cap in USD, under the day's cap
CONCURRENCY = 24  # conversations (or gradings) in flight
EMPTY_RETRIES = 2  # further tries when a reply comes back empty
PARSE_RETRIES = 1  # further tries when a grader's answer cannot be parsed


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sweep_dir(tag: str) -> Path:
    from anatta.judge.paths import Paths

    return Paths.default().sweep_dir(OUT_DATE, f"attractor-{tag}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_dir(folder: Path) -> dict[str, dict]:
    """{name without .json: the object} for a folder of JSON files, in name order."""
    if not folder.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}


# ------------------------------------------------------------------------------------------------ conversations


def arm_params(arm: dict, meta: dict) -> dict:
    from anatta.judge.engine import sampling_params

    return sampling_params(arm["temperature"], C.MAX_TOKENS, {}, meta.get(arm["model"]))


async def converse_one(engine, name: str, arm: dict, params: dict, index: int, out: Path) -> dict:
    """One conversation of N_MESSAGES messages, copy A first. Returns the saved record, or a failure record (saved
    under failed/). A cap stop is raised to the caller with nothing saved; the reply cache keeps what was paid."""
    from anatta.backends.openrouter import ApiError

    path = out / "conversations" / name / f"{index:04d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # The published setup unless the arm carries its own (step 2's steered arm; run_inside.py builds those arms).
    system_prompt = arm.get("system_prompt", C.SYSTEM_PROMPT)
    opening_line = arm.get("opening_line", C.OPENING_LINE)
    messages, replies = [], []
    for t in range(C.N_MESSAGES):
        chat = C.view(messages, t % 2, system_prompt, opening_line)
        text, tries = "", []
        for attempt in range(EMPTY_RETRIES + 1):
            try:
                resp, cached, _ = await engine.call(model=arm["model"], messages=chat, params=params,
                                                    provider=arm["provider"], sample=index,
                                                    salt=f"retry{attempt}" if attempt else None, kind="converse",
                                                    ident=f"{name}/{index}/{t + 1}")
            except ApiError as e:
                if e.scope == "run":
                    raise
                rec = {"arm": name, "index": index, "failed_at": t + 1, "reason": f"api: {str(e)[:300]}",
                       "messages_done": len(messages)}
                write_json(out / "failed" / name / f"{index:04d}.json", rec)
                return rec
            text = (resp.get("text") or "").strip()
            tries.append({"provider": resp.get("provider"), "finish_reason": resp.get("finish_reason"),
                          "prompt_tokens": resp["usage"]["prompt_tokens"],
                          "completion_tokens": resp["usage"]["completion_tokens"],
                          "cached_tokens": resp["usage"].get("cached_tokens", 0),
                          "cost_usd": resp.get("cost_usd", 0.0), "cached": cached, "empty": not text})
            if text:
                break
        if not text:
            rec = {"arm": name, "index": index, "failed_at": t + 1, "reason": "empty reply after retries",
                   "messages_done": len(messages), "tries": tries}
            write_json(out / "failed" / name / f"{index:04d}.json", rec)
            return rec
        messages.append(text)
        replies.append({**tries[-1], "empty_tries": len(tries) - 1,
                        "cost_usd": round(sum(x["cost_usd"] for x in tries), 8)})
    rec = {"arm": name, "index": index, "model": arm["model"], "provider": arm["provider"], "params": params,
           "system_prompt": system_prompt, "opening_line": opening_line, "messages": messages, "replies": replies}
    if "setup" in arm:  # what a steered arm wants kept with the conversation (its form and topic)
        rec["setup"] = arm["setup"]
    write_json(path, rec)
    return rec


async def gather_capped(jobs, limit: int):
    """Run coroutine factories with at most `limit` in flight. Returns (results, stop): once one raises CapStop no
    further job starts, and stop holds its reason."""
    from anatta.judge.engine import CapStop

    sem, stop, results = asyncio.Semaphore(limit), [], []

    async def one(job):
        async with sem:
            if stop:
                return
            try:
                results.append(await job())
            except CapStop as e:
                stop.append(str(e))

    await asyncio.gather(*(one(j) for j in jobs))
    return results, (stop[0] if stop else None)


# ------------------------------------------------------------------------------------------------ grading


def judge_params(judge: dict, meta: dict) -> dict:
    from anatta.judge.engine import sampling_params

    return sampling_params(None, judge["max_tokens"], {}, meta.get(judge["model"]))


async def grade_one(engine, key: str, judge: dict, params: dict, group: str, ident: str, messages: list[str],
                    out: Path, opening_line: str | None = None) -> dict:
    """One grader's label for one conversation, saved under grades/<key>/<group>/<ident>.json. opening_line: the
    line copy A was given when it was not the published one (the grader is told the line that was used)."""
    from anatta.backends.openrouter import ApiError
    from anatta.judge.rubric import parse_label

    path = out / "grades" / key / group / f"{ident}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    chat, sha = C.judge_messages(messages, opening_line), C.codebook_sha256()
    label, why, raw, cost, finish, answered = None, "no attempt", "", 0.0, None, False
    for attempt in range(PARSE_RETRIES + 1):
        try:
            resp, _, paid = await engine.call(model=judge["model"], messages=chat, params=params,
                                              provider=judge["provider"], sample=attempt, salt=sha[:16], kind="grade",
                                              ident=f"{key}/{group}/{ident}")
        except ApiError as e:
            if e.scope == "run":
                raise
            why = f"api: {str(e)[:300]}"
            continue
        answered = True
        cost += paid
        raw, finish = resp.get("text") or "", resp.get("finish_reason")
        label, why = parse_label(raw, C.SCHEMA)
        if label is not None:
            break
    rec = {"judge": key, "judge_model": judge["model"], "group": group, "id": ident, "codebook_sha256": sha,
           "label": label, "parse_ok": label is not None, "finish_reason": finish, "cost_usd": round(cost, 8)}
    if label is None:
        rec["why"], rec["raw"] = why, raw[:1500]
    if answered:  # a call that never got an answer is not saved, so a rerun tries it again
        write_json(path, rec)
    return rec


# ------------------------------------------------------------------------------------------------ setup of paid runs


def open_engine(models: dict, tag: str, client, concurrency: int = CONCURRENCY):
    """(engine, guard, meta) for {model: provider routing}. Prices come from OpenRouter's public listing."""
    from anatta.backends.openrouter import model_meta
    from anatta.judge.cache import Cache
    from anatta.judge.engine import Engine
    from anatta.judge.paths import Paths
    from anatta.spend import SpendGuard

    paths, out = Paths.default(), sweep_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    meta = model_meta(models, cache_dir=paths.cache)
    guard = SpendGuard(paths.spend, sweep_cap=SWEEP_CAP, sweep_log=out / "spend.jsonl", tool="attractor",
                       label=f"{OUT_DATE}/attractor-{tag}")
    engine = Engine(client, Cache(paths.cache), guard, meta, concurrency, concurrency,
                    per_model_limits={m: concurrency for m in models})
    return engine, guard, meta


async def cmd_converse(name: str, n: int | None, tag: str, first: int = 0, limit: int = CONCURRENCY) -> int:
    from anatta.backends.openrouter import OpenRouterClient

    arm = C.ARMS[name]
    n = arm["n"] if n is None else n
    out = sweep_dir(tag)
    async with OpenRouterClient(concurrency=limit + 8, per_model=limit, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({arm["model"]: arm["provider"]}, tag, client, limit)
        params = arm_params(arm, meta)
        log(f"converse {name}: conversations {first}..{n - 1}, model {arm['model']}, params {params}, "
            f"sweep spent ${guard.sweep_spent:.4f} of ${SWEEP_CAP:.2f}")
        jobs = [lambda i=i: converse_one(engine, name, arm, params, i, out) for i in range(first, n)]
        results, stop = await gather_capped(jobs, limit)
    done = [r for r in results if r and "messages" in r]
    failed = [r for r in results if r and "failed_at" in r]
    log(f"converse {name}: {len(done)} finished, {len(failed)} failed, paid this run ${guard.paid_this_run:.4f}, "
        f"sweep ${guard.sweep_spent:.4f}; calls paid {engine.paid_calls}, from cache {engine.cached_calls}"
        + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


async def cmd_grade(group: str, key: str, tag: str, items: dict[str, list[str]]) -> int:
    from anatta.backends.openrouter import OpenRouterClient

    judge, out = C.JUDGES[key], sweep_dir(tag)
    async with OpenRouterClient(concurrency=CONCURRENCY + 8, per_model=CONCURRENCY, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({judge["model"]: judge["provider"]}, tag, client)
        params = judge_params(judge, meta)
        log(f"grade {group} by {key} ({judge['model']}): {len(items)} conversations, params {params}")
        jobs = [lambda i=i, m=m: grade_one(engine, key, judge, params, group, i, m, out) for i, m in items.items()]
        results, stop = await gather_capped(jobs, CONCURRENCY)
    bad = [r["id"] for r in results if r and not r["parse_ok"]]
    log(f"grade {group} by {key}: {len(results)} read, {len(bad)} without a parsed label {bad[:8]}, paid this run "
        f"${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


def arm_items(name: str, tag: str) -> dict[str, list[str]]:
    return {k: v["messages"] for k, v in read_dir(sweep_dir(tag) / "conversations" / name).items()}


def published_items(folder: Path) -> dict[str, list[str]]:
    """{conversation id: messages} from the downloaded pages conversation-<id>.html in a folder (untrusted data)."""
    out = {}
    for p in sorted(Path(folder).glob("conversation-*.html")):
        out[p.stem.removeprefix("conversation-")] = C.parse_published(p.read_text(encoding="utf-8", errors="replace"))
    return out


# ------------------------------------------------------------------------------------------------ the report


def arm_quality(name: str, tag: str) -> dict:
    out = sweep_dir(tag)
    convs = read_dir(out / "conversations" / name)
    failed = read_dir(out / "failed" / name)
    replies = [r for c in convs.values() for r in c["replies"]]
    words = [len(m.split()) for c in convs.values() for m in c["messages"]]
    return {"finished": len(convs), "failed": len([k for k in failed if k not in convs]),
            "replies": len(replies), "at_cap": sum(r["finish_reason"] == "length" for r in replies),
            "errored": sum(r["finish_reason"] == "error" for r in replies),
            "with_cut": sum(bool(C.first_cut([r["finish_reason"] for r in c["replies"]])) for c in convs.values()),
            "empty_tries": sum(r["empty_tries"] for r in replies),
            "words_median": statistics.median(words) if words else None, "words_max": max(words) if words else None,
            "providers": sorted({str(r["provider"]) for r in replies}),
            "cost_usd": round(sum(r["cost_usd"] for r in replies), 4)}


def labels_of(key: str, group: str, tag: str) -> dict[str, dict]:
    """{id: parsed label} for one grader and group; conversations without a parsed label are left out."""
    return {k: v["label"] for k, v in read_dir(sweep_dir(tag) / "grades" / key / group).items() if v["parse_ok"]}


def build_report(tag: str) -> dict:
    rep = {"tag": tag, "codebook_sha256": C.codebook_sha256(), "arms": {}, "published": {}, "agreement": {}}
    for name, arm in C.ARMS.items():
        a = {"quality": arm_quality(name, tag), "graders": {}}
        for key in C.JUDGES:
            labels = labels_of(key, name, tag)
            if not labels:
                continue
            cnt = C.counts(list(labels.values()))
            g = {"n_labels": len(labels), "counts": cnt}
            if arm["published"]:
                theirs = C.published(arm["published"])
                g["readings"] = {m: C.reading(cnt[m], theirs[m]) for m in C.MEASURES}
            convs = read_dir(sweep_dir(tag) / "conversations" / name)
            g["cuts"] = C.cut_table([(lab, [r["finish_reason"] for r in convs[i]["replies"]])
                                     for i, lab in labels.items() if i in convs])
            a["graders"][key] = g
        both = [labels_of(k, name, tag) for k in C.JUDGES]
        shared = sorted(set(both[0]) & set(both[1]))
        if shared:
            rep["agreement"][name] = C.agreement([both[0][i] for i in shared], [both[1][i] for i in shared])
        rep["arms"][name] = a
    for key in C.JUDGES:
        labels = labels_of(key, "published", tag)
        if labels:
            cnt = C.counts(list(labels.values()))
            rep["published"][key] = {"n": len(labels), "adoption": cnt["adoption"]["k"],
                                     "adoption_clear": cnt["adoption_clear"]["k"], "bliss": cnt["bliss"]["k"],
                                     "salience": cnt["salience"]["k"]}
    rep["contrasts"] = {}
    for key in C.JUDGES:  # the follow-up arm against the main arm, by the same grader
        t = rep["arms"].get("temp1", {}).get("graders", {}).get(key)
        m = rep["arms"].get("main", {}).get("graders", {}).get(key)
        if t and m:
            rep["contrasts"][key] = {x: C.diff_ci(t["counts"][x], m["counts"][x]) for x in C.MEASURES}
    main = rep["arms"].get("main", {}).get("graders", {}).get("primary")
    ctrl = rep["arms"].get("control", {}).get("graders", {}).get("primary")
    if main:
        rep["verdict"] = C.verdict(main["readings"], ctrl["readings"]["adoption"] if ctrl else None,
                                   rep["published"].get("primary"))
    return rep


def fmt(c: dict) -> str:
    return "none" if c["pct"] is None else f"{c['pct']:5.1f}% ({c['k']}/{c['n']}; {c['lo']:.1f} to {c['hi']:.1f})"


def print_report(rep: dict) -> None:
    for name, a in rep["arms"].items():
        q = a["quality"]
        print(f"\n== {name}: {q['finished']} finished, {q['failed']} failed, replies at the cap {q['at_cap']} and "
              f"ended by the provider {q['errored']} of {q['replies']} ({q['with_cut']} conversations with a cut "
              f"reply), median words a reply {q['words_median']}, cost ${q['cost_usd']}")
        theirs = C.published(C.ARMS[name]["published"]) if C.ARMS[name]["published"] else None
        for key, g in a["graders"].items():
            print(f"  grader {key} ({g['n_labels']} labels, {g['counts']['broken']} not nested)")
            for m in C.MEASURES:
                line = f"    {m:14s} {fmt(g['counts'][m])}"
                if theirs:
                    line += f"   published {fmt(theirs[m])}   {g['readings'][m]}"
                print(line)
            print(f"    adoption clear {g['counts']['adoption_clear']['k']}, mixed {g['counts']['adoption_mixed']['k']}")
            t = g["cuts"]
            print(f"    of {t['adopted']} adopted: no cut reply {t['no_cut']}, adopted at or before the first cut "
                  f"{t['onset_at_or_before_first_cut']}, after it {t['onset_after_first_cut']}, no onset given "
                  f"{t['onset_missing']}")
        if name in rep["agreement"]:
            print("  agreement of the two graders: " + ", ".join(
                f"{m} {v['agree']}/{v['n']}" for m, v in rep["agreement"][name].items()))
    for key, c in rep.get("contrasts", {}).items():
        print(f"\ntemp1 minus main, grader {key}: " + "; ".join(
            f"{m} {v['diff']:+.1f} ({v['lo']:+.1f} to {v['hi']:+.1f}){' differs' if v['differs'] else ''}"
            for m, v in c.items()))
    for key, p in rep["published"].items():
        print(f"\npublished conversations, grader {key}: adoption {p['adoption']}/{p['n']} (clear "
              f"{p['adoption_clear']}), bliss {p['bliss']}/{p['n']}, salience {p['salience']}/{p['n']}")
    if "verdict" in rep:
        print("\nverdict:", json.dumps(rep["verdict"]))


def cmd_dry() -> int:
    from anatta.backends.openrouter import model_meta
    from anatta.judge.paths import Paths
    from anatta.spend import daily_cap, spent_today

    paths = Paths.default()
    models = {a["model"]: a["provider"] for a in C.ARMS.values()} | {j["model"]: j["provider"] for j in C.JUDGES.values()}
    meta = model_meta(models, cache_dir=paths.cache)
    print(f"codebook sha256 {C.codebook_sha256()}")
    for name, arm in C.ARMS.items():
        m = meta[arm["model"]]
        print(f"arm {name}: {arm['n']} conversations x {C.N_MESSAGES} messages, {arm['model']} via {m['endpoints']}, "
              f"temperature {arm['temperature']}, reply cap {C.MAX_TOKENS} tokens, $ per M in {m['in'] * 1e6:.3f} "
              f"out {m['out'] * 1e6:.3f}")
    for key, j in C.JUDGES.items():
        m = meta[j["model"]]
        print(f"grader {key}: {j['model']} via {m['endpoints']}, $ per M in {m['in'] * 1e6:.3f} out {m['out'] * 1e6:.3f}")
    print(f"sweep cap ${SWEEP_CAP:.2f}; today ${spent_today(paths.spend):.4f} of ${daily_cap(paths.spend):.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="a1")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--converse", metavar="ARM", choices=sorted(C.ARMS))
    ap.add_argument("--n", type=int)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY, help="conversations in flight")
    ap.add_argument("--grade", metavar="ARM", choices=sorted(C.ARMS))
    ap.add_argument("--published", metavar="DIR")
    ap.add_argument("--judge", choices=sorted(C.JUDGES))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.dry:
        return cmd_dry()
    if a.converse:
        return asyncio.run(cmd_converse(a.converse, a.n, a.tag, a.first, a.concurrency))
    if a.grade or a.published:
        if not a.judge:
            ap.error("--grade and --published need --judge")
        if a.grade:
            return asyncio.run(cmd_grade(a.grade, a.judge, a.tag, arm_items(a.grade, a.tag)))
        return asyncio.run(cmd_grade("published", a.judge, a.tag, published_items(Path(a.published))))
    if a.report:
        rep = build_report(a.tag)
        write_json(sweep_dir(a.tag) / "summary.json", rep)
        print_report(rep)
        return 0
    ap.error("nothing to do")


if __name__ == "__main__":
    raise SystemExit(main())
