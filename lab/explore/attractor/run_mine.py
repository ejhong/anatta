"""The attractor line, step 4: the driver of angle A, "no self, or a bigger self?" (design:
lab/plans/attractor-step4-2026-11-16.md; mine_core.py holds the pure parts; steps 1 to 3's files are reused
unchanged). Claude (Fable 5.2, an AI model) wrote this file on 16 Nov 2026, before any call of step 4. Explore speed:
sightings only. No training; inference through OpenRouter; nothing is graded by a model.

  --dry                        today's spend, the credit, and which of the step's files are fixed; no call
  --calibrate                  stage 0: every tier's problems solved in fresh sessions (200 short calls)
  --show-work [--reject ID..]  prints the tier and the fixed work stage 0 would give, for the driver to read; no call
  --fix-work [--reject ID..]   writes stage0.json (the tier, block 1's problems, block 2's fixed work), once
  --try                        both blocks on a few units, mechanics only; then writes plan.json (the cost rule), once
  --run BLOCK [--arm ARM ..]   block own (the model's own work, three ways) or fixed (the copy's work, on fixed work)
  --read WHAT                  anchor (block own's fresh arm, and the gate), own or fixed; only when complete; no call

Order: --calibrate, --show-work, --fix-work, --try, --run own --arm fresh, --read anchor, then the rest. Each later
step refuses until the one before it has left its file.

Writes results_root/explore/2026-11-16/attractor-a4/: calibration/, stage0.json, try_<block>/<arm>/<unit>.json,
plan.json, <block>/<arm>/<unit>.json, <block>_failed/..., <what>_summary.json and spend.jsonl. Every paid call goes
through anatta.spend's SpendGuard (tool "attractor") and the shared reply cache. The arms are step 3's (its arms.json).

  .venv/bin/python lab/explore/attractor/run_mine.py --dry
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

import conduct_core as K  # noqa: E402
import inside_core as I  # noqa: E402
import mine_core as M  # noqa: E402
import run_attractor as R  # noqa: E402
import run_conduct as RC  # noqa: E402
import run_inside as RI  # noqa: E402

TAG = "a4"
OUT_DATE = "2026-11-16"  # the UTC date of the step's first call
SWEEP_CAP = 15.0  # this step's own cap in USD, under the day's cap
SALT, CAL_SALT = "step4", "step4-cal"  # the reply cache's salts: every test call of the step, and stage 0's
ARM = RI.ARM  # the model, the host and temperature 1.0, as in steps 1 to 3
MODE = "system"  # no request of this step carries a system message, so the mode changes nothing
EMPTY_RETRIES = 2  # further tries when a unit's reply to a problem comes back empty
DESIGN = "lab/plans/attractor-step4-2026-11-16.md"
BLOCKS = {"own": M.OWN_ARMS, "fixed": M.FIXED_ARMS}
FIXED_CALLS = 2 * (M.WORK_WRONG + M.WORK_RIGHT) + 2 * M.SENSITIVITY_N  # block 2's calls for one unit


def log(*a):
    R.log(*a)


def sweep() -> Path:
    from anatta.judge.paths import Paths

    return Paths.default().sweep_dir(OUT_DATE, f"attractor-{TAG}")


def open_engine(client, concurrency: int):
    """(engine, guard, meta) for the one model of this step, under the step's own cap."""
    from anatta.backends.openrouter import model_meta
    from anatta.judge.cache import Cache
    from anatta.judge.engine import Engine
    from anatta.judge.paths import Paths
    from anatta.spend import SpendGuard

    paths, out, models = Paths.default(), sweep(), {ARM["model"]: ARM["provider"]}
    out.mkdir(parents=True, exist_ok=True)
    meta = model_meta(models, cache_dir=paths.cache)
    guard = SpendGuard(paths.spend, sweep_cap=SWEEP_CAP, sweep_log=out / "spend.jsonl", tool="attractor",
                       label=f"{OUT_DATE}/attractor-{TAG}")
    engine = Engine(client, Cache(paths.cache), guard, meta, concurrency, concurrency,
                    per_model_limits={m: concurrency for m in models})
    return engine, guard, meta


def params(meta: dict, max_tokens: int) -> dict:
    from anatta.judge.engine import sampling_params

    return RC.checked(sampling_params(ARM["temperature"], max_tokens, {}, meta.get(ARM["model"])), ARM["temperature"])


def usage(resp: dict, cached: bool) -> dict:
    return {"finish_reason": resp.get("finish_reason"), "prompt_tokens": resp["usage"]["prompt_tokens"],
            "completion_tokens": resp["usage"]["completion_tokens"],
            "cached_tokens": resp["usage"].get("cached_tokens", 0), "cost_usd": resp.get("cost_usd", 0.0),
            "from_cache": cached}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ stage 0


def cal_jobs() -> list[tuple[dict, int]]:
    return [(M.problem(t, i), s) for t in M.TIERS for i in range(M.CAL_PROBLEMS) for s in range(M.CAL_SAMPLES)]


def cal_path(out: Path, p: dict, sample: int) -> Path:
    return out / "calibration" / f"{p['tier']}-{p['index']:04d}-{sample}.json"


async def cal_one(engine, gen_params: dict, p: dict, sample: int, out: Path) -> dict | None:
    """One problem solved once in a fresh session (the sample number keeps the five solutions apart in the reply
    cache). Saved when the call came back; a call that failed is tried again by a later run."""
    from anatta.backends.openrouter import ApiError

    path = cal_path(out, p, sample)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        resp, cached, _ = await engine.call(model=ARM["model"], messages=I.fresh(M.gen_request(p)), params=gen_params,
                                            provider=ARM["provider"], sample=sample, salt=CAL_SALT, kind="calibrate",
                                            ident=f"cal/{p['id']}/{sample}")
    except ApiError as e:
        if e.scope in ("run", "model"):
            raise
        log(f"calibrate {p['id']}/{sample}: failed ({str(e)[:120]}); not saved")
        return None
    text = (resp.get("text") or "").strip()
    final = M.parse_final(text)
    rec = {"id": f"{p['id']}/{sample}", "problem": p["id"], "tier": p["tier"], "index": p["index"], "sample": sample,
           "text": p["text"], "answer": p["answer"], "reply": text, "final": final,
           "truth": M.truth(final, p["answer"]), **usage(resp, cached)}
    R.write_json(path, rec)
    return rec


def read_pool(out: Path) -> list[dict]:
    """Stage 0's replies, complete or not at all: raises SystemExit naming how many are missing."""
    jobs = cal_jobs()
    missing = [f"{p['id']}/{s}" for p, s in jobs if not cal_path(out, p, s).exists()]
    if missing:
        raise SystemExit(f"stage 0 is not finished: {len(missing)} of {len(jobs)} solutions are missing "
                         f"(run --calibrate)")
    return [json.loads(cal_path(out, p, s).read_text(encoding="utf-8")) for p, s in jobs]


def print_accuracy(acc: dict) -> None:
    for t, a in acc.items():
        share = "-" if a["share"] is None else f"{100 * a['share']:5.1f}%"
        print(f"  {t}: right {a['right']:>2d} of {a['calls']:>2d} ({share}), readable {a['readable']}")


async def cmd_calibrate(limit: int = 24) -> int:
    from anatta.backends.openrouter import OpenRouterClient

    out, jobs = sweep(), cal_jobs()
    async with OpenRouterClient(concurrency=limit + 8, per_model=limit, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine(client, limit)
        gen_params = params(meta, M.GEN_MAX_TOKENS)
        log(f"calibrate: {len(jobs)} solutions, params {gen_params}, sweep ${guard.sweep_spent:.4f} of "
            f"${SWEEP_CAP:.2f}")
        results, stop = await R.gather_capped([lambda p=p, s=s: cal_one(engine, gen_params, p, s, out)
                                               for p, s in jobs], limit)
    done = [r for r in results if r]
    log(f"calibrate: {len(done)} of {len(jobs)} saved, paid this run ${guard.paid_this_run:.4f}, sweep "
        f"${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    if stop or len(done) < len(jobs):
        return 3
    print_accuracy(M.accuracy(done))
    return 0


def stage0_of(pool: list[dict], rejected: list[str]) -> dict:
    """What stage 0 fixes, from its replies and the ids rejected on reading: the accuracy by tier, block 1's tier and
    problems, and block 2's fixed work."""
    known = {r["id"] for r in pool}
    if set(rejected) - known:
        raise SystemExit(f"not stage 0 solutions: {sorted(set(rejected) - known)}")
    acc = M.accuracy(pool)
    tier = M.choose_tier(acc)
    if tier is None:
        raise SystemExit(f"no tier has between {M.TIER_BAND[0]:.0%} and {M.TIER_BAND[1]:.0%} right: the step stops "
                         f"for an addendum")
    try:
        chosen = M.choose_work(pool, acc, set(rejected))
    except ValueError as e:
        raise SystemExit(f"stage 0 cannot fill the fixed work ({e}): the step stops for an addendum") from e
    work = [{k: r[k] for k in ("id", "problem", "tier", "sample", "text", "answer", "final", "truth", "reply")}
            for r in chosen]
    return {"design": DESIGN, "accuracy": acc, "tier": tier,
            "problems": [M.problem(tier, M.RUN_FIRST + i)["id"] for i in range(M.RUN_PROBLEMS)],
            "work_key": M.WORK_KEY, "rejected": sorted(rejected), "work": work}


def cmd_show_work(rejected: list[str]) -> int:
    """The tier and the fixed work stage 0 would give, with each solution whole, for the driver to read before it is
    fixed. The solutions are the model's arithmetic. Printed to the terminal only; nothing is written."""
    st = stage0_of(read_pool(sweep()), rejected)
    print_accuracy(st["accuracy"])
    print(f"block 1's tier: {st['tier']}; its problems: {st['problems']}")
    for w in st["work"]:
        print(f"=== {w['id']} ({w['truth']}): {w['text']} true answer {w['answer']}, read as {w['final']}\n{w['reply']}")
    return 0


def cmd_fix_work(rejected: list[str]) -> int:
    path = sweep() / "stage0.json"
    if path.exists():
        raise SystemExit(f"{path} exists; stage 0 is fixed once")
    st = stage0_of(read_pool(sweep()), rejected)
    R.write_json(path, st)
    print_accuracy(st["accuracy"])
    print(f"stage 0 fixed: tier {st['tier']}, {len(st['problems'])} problems, work "
          f"{[w['id'] for w in st['work']]}, {len(rejected)} rejected on reading; stage0.json sha256 {sha256_of(path)}")
    return 0


def read_stage0() -> dict:
    path = sweep() / "stage0.json"
    if not path.exists():
        raise SystemExit("run --calibrate, --show-work and --fix-work first")
    return json.loads(path.read_text(encoding="utf-8"))


def block1_problems(st: dict, n: int) -> list[dict]:
    """Block 1's first n problems, built again from the tier (and checked against the ids stage 0 recorded)."""
    ps = [M.problem(st["tier"], M.RUN_FIRST + i) for i in range(n)]
    if [p["id"] for p in ps] != st["problems"][:n]:
        raise SystemExit("block 1's problems are not the ones stage 0 recorded")
    return ps


# ------------------------------------------------------------------------------------------------ units


def all_units(rec: dict, arms) -> list[dict]:
    """The units of the given arms: step 3's state and plain conversations (run_conduct.units reads them from their
    files), then the bare and fresh sessions of this step's sizes."""
    convs = RC.units({**rec, "arms": {**{a: [] for a in K.CONVERSATION_ARMS},
                                      **{a: rec["arms"][a] for a in M.CONVERSATION_ARMS if a in arms},
                                      **{a: 0 for a in K.SESSION_ARMS}}})
    return convs + [{"arm": arm, "key": f"{u:04d}", "index": u, "messages": None, "context_tokens": None, "sample": u}
                    for arm in ("bare", "fresh") if arm in arms for u in range(M.SESSIONS[arm])]


def try_units(units: list[dict]) -> list[dict]:
    """The small try: the two longest conversations of the state and plain arms (the longest prompts), and the first
    two bare and fresh sessions."""
    out = []
    for arm in M.CONVERSATION_ARMS:
        out += sorted((u for u in units if u["arm"] == arm), key=lambda u: -u["context_tokens"])[:2]
    for arm in ("bare", "fresh"):
        out += [u for u in units if u["arm"] == arm][:2]
    return out


def expected(rec: dict, arms) -> dict[str, int]:
    return {arm: (len(rec["arms"][arm]) if arm in M.CONVERSATION_ARMS else M.SESSIONS[arm]) for arm in arms}


# ------------------------------------------------------------------------------------------------ the two blocks


async def ask(engine, unit: dict, request: list[dict], call_params: dict, kind: str, ident: str,
              salt: str = SALT) -> dict:
    """One call from the end of the unit's conversation (or in its session): the reply and its usage."""
    resp, cached, _ = await engine.call(model=ARM["model"], messages=RC.unit_chat(unit, request, MODE),
                                        params=call_params, provider=ARM["provider"], sample=unit["sample"], salt=salt,
                                        kind=kind, ident=ident)
    return {"reply": (resp.get("text") or "").strip(), **usage(resp, cached)}


async def own_unit(engine, gen_params: dict, rate_params: dict, unit: dict, problems: list[dict], out: Path,
                   folder: str) -> dict | None:
    """Block 1 for one unit: each problem is put to it, and its reply (its own work) is then rated three ways, each
    in a call of its own: in its own earlier turn ("own"), and shown back in a user turn as the partner's (in a fresh
    session: as an unlabelled proposal) and as a stranger's. An empty reply to a problem is asked again, up to
    EMPTY_RETRIES times. Saved only when every call came back."""
    from anatta.backends.openrouter import ApiError

    path = RC.unit_path(out, folder, unit)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    labels = M.own_labels(unit["arm"])

    async def one(p: dict) -> dict:
        base = f"own/{unit['arm']}/{unit['key']}/{p['id']}"
        try:
            work = {"reply": ""}
            for attempt in range(EMPTY_RETRIES + 1):
                work = await ask(engine, unit, M.gen_request(p), gen_params, "own-work", f"{base}/work",
                                 f"{SALT}-retry{attempt}" if attempt else SALT)
                if work["reply"]:
                    break
            if not work["reply"]:
                return {"problem": p["id"], "ok": False, "error": "empty reply after retries"}

            async def rate(label: str) -> tuple[str, dict]:
                r = await ask(engine, unit, M.request_for(p, work["reply"], label), rate_params, "own-rate",
                              f"{base}/{label}")
                return label, {**r, "rating": M.parse_rating(r["reply"])}

            ratings = dict(await asyncio.gather(*(rate(label) for label in labels)))
        except ApiError as e:
            if e.scope in ("run", "model"):  # every further call would fail too: stop, record nothing
                raise
            return {"problem": p["id"], "ok": False, "error": f"api: {str(e)[:300]}"}
        final = M.parse_final(work["reply"])
        return {"problem": p["id"], "ok": True, "answer": work["reply"], "final": final,
                "truth": M.truth(final, p["answer"]), "work": {k: v for k, v in work.items() if k != "reply"},
                "ratings": ratings}

    items = list(await asyncio.gather(*(one(p) for p in problems)))
    if not all(it["ok"] for it in items):
        log(f"{folder} {unit['arm']}/{unit['key']}: {sum(not it['ok'] for it in items)} problems failed; not saved")
        return None
    rec = {"arm": unit["arm"], "key": unit["key"], "index": unit["index"], "context_tokens": unit["context_tokens"],
           "params": {"work": gen_params, "rate": rate_params}, "items": items}
    R.write_json(path, rec)
    return rec


def fixed_asks(work: list[dict]) -> list[tuple[dict, str]]:
    """Block 2's calls for one unit: every fixed solution as the partner's and as a stranger's, and the first
    SENSITIVITY_N wrong ones as an expert's and as a novice's."""
    wrong = [w for w in work if w["truth"] == "wrong"][:M.SENSITIVITY_N]
    return ([(w, label) for w in work for label in ("partner", "stranger")]
            + [(w, label) for w in wrong for label in ("expert", "novice")])


async def fixed_unit(engine, rate_params: dict, unit: dict, work: list[dict], out: Path, folder: str) -> dict | None:
    """Block 2 for one unit: each fixed solution rated under each of its labels, each in a call of its own from the
    end of the conversation (or in a bare session). Saved only when every call came back."""
    from anatta.backends.openrouter import ApiError

    path = RC.unit_path(out, folder, unit)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    async def rate(w: dict, label: str) -> tuple[str, str, dict | None]:
        try:
            r = await ask(engine, unit, M.rate_request(w, w["reply"], label), rate_params, "fixed-rate",
                          f"fixed/{unit['arm']}/{unit['key']}/{w['id']}/{label}")
        except ApiError as e:
            if e.scope in ("run", "model"):  # every further call would fail too: stop, record nothing
                raise
            return w["id"], label, None
        return w["id"], label, {**r, "rating": M.parse_rating(r["reply"])}

    got = await asyncio.gather(*(rate(w, label) for w, label in fixed_asks(work)))
    if any(r is None for _, _, r in got):
        log(f"{folder} {unit['arm']}/{unit['key']}: {sum(r is None for _, _, r in got)} calls failed; not saved")
        return None
    items = [{"work": w["id"], "truth": w["truth"], "ratings": {label: r for wid, label, r in got if wid == w["id"]}}
             for w in work]
    rec = {"arm": unit["arm"], "key": unit["key"], "index": unit["index"], "context_tokens": unit["context_tokens"],
           "params": {"rate": rate_params}, "items": items}
    R.write_json(path, rec)
    return rec


def job_for(block: str, engine, gen_params: dict, rate_params: dict, st: dict, n_problems: int, out: Path, folder: str):
    """The job run_conduct.run_units runs for each unit of a block."""
    if block == "own":
        problems = block1_problems(st, n_problems)
        return lambda u: own_unit(engine, gen_params, rate_params, u, problems, out, folder)
    return lambda u: fixed_unit(engine, rate_params, u, st["work"], out, folder)


async def run_block(block: str, todo: list[dict], st: dict, n_problems: int, folder: str, limit: int) -> int:
    """One block on the given units, in a hashed order that mixes the arms, with step 3's passes and its record of
    units that keep failing."""
    from anatta.backends.openrouter import OpenRouterClient

    out, todo = sweep(), sorted(todo, key=RC.run_order)
    async with OpenRouterClient(concurrency=48, per_model=40, timeout=600.0, title="anatta attractor") as client:
        engine, guard, meta = open_engine(client, 40)
        gen_params, rate_params = params(meta, M.GEN_MAX_TOKENS), params(meta, M.RATE_MAX_TOKENS)
        log(f"{folder}: {len(todo)} units, sweep ${guard.sweep_spent:.4f} of ${SWEEP_CAP:.2f}")
        job = job_for(block, engine, gen_params, rate_params, st, n_problems, out, folder)
        saved, failed, stop = await RC.run_units(job, todo, out, folder, limit)
    log(f"{folder}: {saved} of {len(todo)} units saved, {failed} recorded as failed, paid this run "
        f"${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


# ------------------------------------------------------------------------------------------------ the try and the plan


def unit_calls(block: str, rec: dict) -> list[dict]:
    """Every call of one saved unit, each with a kind: "work" (a reply to a problem) or "rate"."""
    calls = []
    for it in rec["items"]:
        if block == "own":
            calls.append({**it["work"], "kind": "work", "reply": it["answer"], "read": it["truth"] is not None})
        calls += [{**r, "kind": "rate", "read": r["rating"] is not None} for r in it["ratings"].values()]
    return calls


def mechanics(saved: dict[str, list[dict]]) -> dict:
    """What the try shows, from {block: its saved unit records}. For calls from inside a conversation: the cost of
    one prompt token (the try uses the longest conversations, so a mean cost per call would overstate the run's) and
    the tokens a call adds to its conversation. For short calls: the mean cost. And the shares of ratings and of final
    answers that could be read. No rating is shown."""
    calls = [(rec["context_tokens"], c) for block, recs in saved.items() for rec in recs
             for c in unit_calls(block, rec)]
    inside = [(ctx, c) for ctx, c in calls if ctx is not None]
    short = [c["cost_usd"] for ctx, c in calls if ctx is None]
    rates = [c["read"] for _, c in calls if c["kind"] == "rate"]
    works = [c["read"] for _, c in calls if c["kind"] == "work"]
    tokens = sum(c["prompt_tokens"] for _, c in inside)
    return {"calls": len(calls), "inside_calls": len(inside),
            "cost_inside_token": sum(c["cost_usd"] for _, c in inside) / tokens if tokens else None,
            "inside_added_tokens": statistics.fmean(c["prompt_tokens"] - ctx for ctx, c in inside) if inside else None,
            "cost_short_call": statistics.fmean(short) if short else None,
            "ratings_readable": sum(rates) / len(rates) if rates else None,
            "finals_readable": sum(works) / len(works) if works else None}


def projected(n_conversations: int, n_problems: int, cost_inside: float, cost_short: float, spent: float) -> float:
    """What the step would cost in all: what is spent, plus every planned call at the try's mean costs (the try's own
    units are counted again, so this errs high)."""
    own = len(M.own_labels("state")) + 1  # block 1's calls for one problem: the work and its three ratings
    inside = n_conversations * (n_problems * own + FIXED_CALLS)
    short = M.SESSIONS["bare"] * (n_problems * own + FIXED_CALLS) + M.SESSIONS["fresh"] * n_problems * own
    return spent + inside * cost_inside + short * cost_short


def plan_of(mech: dict, contexts: list[int], spent: float) -> dict:
    """The plan the try fixes: block 1's number of problems by the cost rule, and whether the ratings can be read.
    contexts: the context length at the end of every conversation the run will use. A call from inside a conversation
    is costed at the try's price for one prompt token times the mean prompt the run will send (the mean context plus
    the tokens a call adds)."""
    inside = (mech["cost_inside_token"] or 0.0) * (statistics.fmean(contexts) + (mech["inside_added_tokens"] or 0.0))
    short = mech["cost_short_call"] or 0.0
    cost = projected(len(contexts), M.RUN_PROBLEMS, inside, short, spent)
    n = M.RUN_PROBLEMS if cost <= M.COST_LINE else M.RUN_PROBLEMS_CHEAP
    return {"design": DESIGN, **mech, "cost_inside_call": inside, "projected_usd": cost, "problems": n,
            "projected_usd_as_planned": projected(len(contexts), n, inside, short, spent),
            "pass": mech["ratings_readable"] is not None and mech["ratings_readable"] >= M.READABLE_MIN - K.EPS}


def print_try(saved: dict[str, list[dict]]) -> None:
    """A try's mechanics, unit by unit, and the form of a few replies with every digit hidden. No rating and no answer
    is shown."""
    for block, recs in saved.items():
        for rec in recs:
            calls = unit_calls(block, rec)
            ends = {f: sum(str(c["finish_reason"]) == f for c in calls)
                    for f in sorted({str(c["finish_reason"]) for c in calls})}
            rates = [c for c in calls if c["kind"] == "rate"]
            works = [c for c in calls if c["kind"] == "work"]
            print(f"  try {block:<5s} {rec['arm']:<6s} {rec['key']:<22s} {len(calls)} calls, ends {ends}, ratings read "
                  f"{sum(c['read'] for c in rates)}/{len(rates)}"
                  + (f", final answers read {sum(c['read'] for c in works)}/{len(works)}" if works else "")
                  + f", prompt tokens up to {max(c['prompt_tokens'] for c in calls)}, cost "
                  f"${sum(c['cost_usd'] for c in calls):.4f}")
            for c in (works[:1] + rates[:2]):
                print(f"      {c['kind']:<4s} | {M.masked(c['reply'])}")


def read_saved(folder: str, arms) -> list[dict]:
    out = sweep()
    return [r for arm in arms for r in R.read_dir(out / folder / arm).values()]


def fix_plan(rec: dict) -> dict:
    """After the try: its mechanics are printed and plan.json is written, once (the cost rule's number of problems,
    and whether enough ratings could be read). Returns the plan the try gives, written or not."""
    from anatta.spend import sweep_spent

    out = sweep()
    saved = {block: read_saved(f"try_{block}", arms) for block, arms in BLOCKS.items()}
    print_try(saved)
    mech = mechanics(saved)
    spend = out / "spend.jsonl"
    plan = plan_of(mech, [rec["conversations"][k]["context_tokens"] for a in M.CONVERSATION_ARMS for k in rec["arms"][a]],
                   sweep_spent(spend) if spend.exists() else 0.0)
    print(f"try: {mech['calls']} calls; a call from inside a conversation of mean length ${plan['cost_inside_call']:.5f}"
          f", a short one ${mech['cost_short_call'] or 0:.5f}; ratings read {pct(mech['ratings_readable'])}, final "
          f"answers read {pct(mech['finals_readable'])}; the step as designed would cost about "
          f"${plan['projected_usd']:.2f}; block 1 uses {plan['problems']} problems; ratings gate "
          f"{'passed' if plan['pass'] else 'NOT passed'}")
    path = out / "plan.json"
    if path.exists():
        print(f"{path} exists; the plan was fixed by the first try and is not rewritten")
    else:
        R.write_json(path, plan)
        print(f"plan.json sha256 {sha256_of(path)}")
    return plan


async def cmd_try(limit: int = 6) -> int:
    rec, st = RC.read_arms(), read_stage0()
    for block, arms in BLOCKS.items():
        code = await run_block(block, try_units(all_units(rec, arms)), st, M.RUN_PROBLEMS, f"try_{block}", limit)
        if code:
            return code
    fix_plan(rec)
    return 0


def read_plan() -> dict:
    path = sweep() / "plan.json"
    if not path.exists():
        raise SystemExit("run --try first: it fixes block 1's number of problems by the cost rule")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not plan["pass"]:
        raise SystemExit("fewer than 90% of the try's ratings could be read: fix the parser or the wording in an "
                         "addendum first")
    return plan


def gate_passed() -> bool:
    path = sweep() / "anchor_summary.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8"))["gate"]["pass"]


async def cmd_run(block: str, arms: list[str] | None, limit: int = 6) -> int:
    rec, st, plan = RC.read_arms(), read_stage0(), read_plan()
    arms = list(BLOCKS[block]) if not arms else arms
    if set(arms) - set(BLOCKS[block]):
        raise SystemExit(f"block {block} runs on {BLOCKS[block]}")
    if (block, arms) != ("own", ["fresh"]) and not gate_passed():
        raise SystemExit("block own's fresh arm is run and read first (--run own --arm fresh, then --read anchor), "
                         "and its gate must pass")
    return await run_block(block, all_units(rec, arms), st, plan["problems"], block, limit)


# ------------------------------------------------------------------------------------------------ the readings

OWN_PAIRS = (("own", "stranger"), ("partner", "stranger"), ("own", "partner"))
FRESH_PAIRS = (("own", "unlabelled"), ("own", "stranger"), ("unlabelled", "stranger"))
ARM_PAIRS = (("state", "plain"), ("state", "bare"), ("plain", "bare"))
TRUTH_NAMES = {"wrong": "wrong", "right": "right", None: "all"}


def name(a: str, b: str) -> str:
    return f"{a} - {b}"


def gaps_of(units: list[list[dict]], pairs) -> dict:
    """{"a - b": {"wrong" | "right" | "all": the arm's gap}} for the given label pairs."""
    return {name(a, b): {TRUTH_NAMES[w]: M.gap(units, a, b, w) for w in M.TRUTHS} for a, b in pairs}


def between(gaps: dict, records: dict, cuts, pairs) -> dict:
    """{"x - y": {"a - b": the difference of the two arms' gaps on wrong work}} for the arm pairs present, and for
    state minus plain the same reading matched on context length (step 3's cut points and its function)."""
    out = {}
    for x, y in ARM_PAIRS:
        if x not in gaps or y not in gaps:
            continue
        out[name(x, y)] = {}
        for a, b in pairs:
            d = M.gap_diff(gaps[x][name(a, b)]["wrong"], gaps[y][name(a, b)]["wrong"])
            if (x, y) == ("state", "plain"):
                sides = [([{"measure": M.unit_gap(r["items"], a, b, "wrong")} for r in records[arm]],
                          [r["context_tokens"] for r in records[arm]]) for arm in (x, y)]
                d["length_matched"] = K.matched_contrast(*sides[0], *sides[1], cuts, margin=M.MARGIN)
            out[name(x, y)][name(a, b)] = d
    return out


def table(units: dict[str, list[list[dict]]], labels_of) -> dict:
    """{arm: {label: what its ratings look like, with the guard}}."""
    return {arm: {label: {**M.rating_stats(u, label), "no_rating": M.no_rating(u, label)} for label in labels_of(arm)}
            for arm, u in units.items()}


def read_own(records: dict[str, list[dict]], cuts) -> dict:
    """Block 1's reading, from {arm: unit records}. With only the fresh arm it is the anchor (R1 and the gate)."""
    units = {arm: [r["items"] for r in rs] for arm, rs in records.items()}
    work = {arm: M.work_stats(u) for arm, u in units.items()}
    gaps = {arm: gaps_of(u, FRESH_PAIRS if arm == "fresh" else OWN_PAIRS) for arm, u in units.items()}
    s = {"work": work, "ratings": table(units, M.own_labels), "gaps": gaps}
    if "fresh" in gaps:
        s["R1"] = gaps["fresh"]["own - unlabelled"]["wrong"]
        t = s["ratings"]["fresh"]
        calls = sum(t[k]["no_rating"]["calls"] for k in t)
        readable = 1 - sum(t[k]["no_rating"]["unreadable"] for k in t) / calls if calls else None
        right = work["fresh"]["right"]
        s["gate"] = {"ratings_readable": readable, "right_share": right,
                     "pass": readable is not None and readable >= M.READABLE_MIN - K.EPS and right is not None
                     and M.TIER_BAND[0] <= right <= M.TIER_BAND[1]}
    s["between"] = between(gaps, records, cuts, OWN_PAIRS)
    s["work_flags"] = {name(x, y): M.work_flags(work[x], work[y]) for x, y in ARM_PAIRS if x in work and y in work}
    if "state - plain" in s["between"]:
        s["R3"] = s["between"]["state - plain"]["own - stranger"]
        s["R3_wording"] = M.tint(gaps["state"]["own - stranger"]["wrong"], gaps["plain"]["own - stranger"]["wrong"],
                                 s["R3"])
    return s


def read_fixed(records: dict[str, list[dict]], cuts) -> dict:
    """Block 2's reading, from {arm: unit records}: R2a, R2b, the step's answer, and the sensitivity check."""
    units = {arm: [r["items"] for r in rs] for arm, rs in records.items()}
    gaps = {arm: {**gaps_of(u, (("partner", "stranger"),)),
                  "expert - novice": {"wrong": M.gap(u, "expert", "novice", "wrong")}} for arm, u in units.items()}
    s = {"ratings": table(units, lambda arm: ("partner", "stranger", "expert", "novice")), "gaps": gaps,
         "between": between(gaps, records, cuts, (("partner", "stranger"),))}
    s["R2a"] = gaps["state"]["partner - stranger"]["wrong"]
    s["R2b"] = s["between"]["state - plain"]["partner - stranger"]
    s["answer"] = M.answer(s["R2a"], s["R2b"])
    return s


def num(v, signed: bool = True) -> str:
    return "-" if v is None else f"{v:+.2f}" if signed else f"{v:.2f}"


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def gap_line(label: str, g: dict) -> str:
    ci = "-" if g.get("ci95") is None else f"[{num(g['ci95'][0])}, {num(g['ci95'][1])}]"
    v = g.get("value", g.get("diff"))
    line = f"  {label:<42s} {num(v)} {ci}  {g['verdict']}"
    if "units" in g:
        line += f"  (units {g['measured']}/{g['units']})"
    if g.get("guard_flag"):
        line += "  GUARD FLAGGED"
    if "length_matched" in g:
        m = g["length_matched"]
        ci = "-" if m["ci95"] is None else f"[{num(m['ci95'][0])}, {num(m['ci95'][1])}]"
        line += f"; length-matched {num(m['diff'])} {ci} {m['verdict']}"
    return line


def print_reading(title: str, s: dict) -> None:
    print(title)
    for arm, w in s.get("work", {}).items():
        print(f"  work, {arm}: {w['replies']} replies, final answer readable {pct(w['readable'])}, right "
              f"{pct(w['right'])}, wrong {w['wrong']}")
    for pair, f in s.get("work_flags", {}).items():
        if f["readable_flag"] or f["right_flag"]:
            print(f"  WORK FLAG, {pair}: " + ", ".join(k for k, v in f.items() if v))
    for arm, labels in s["ratings"].items():
        for label, t in labels.items():
            if not t.get("rated"):
                continue
            print(f"  ratings, {arm}/{label}: {t['rated']} rated, unreadable {pct(t['no_rating']['share'])}, mean "
                  f"{num(t['mean'], False)}, at 10 {pct(t['at_10'])}, commonest {t['mode']:g} ({pct(t['mode_share'])}), "
                  f"wrong work rated {M.ENDORSE}+ {pct(t['wrong_endorsed'])}, follows the truth (auroc) "
                  f"{num(t['auroc'], False)}, carries the talk {pct(t['talk'])}, cut {t['cut']}")
    for arm, pairs in s["gaps"].items():
        for pair, by in pairs.items():
            for truth, g in by.items():
                print(gap_line(f"{arm}: {pair}, {truth} work", g))
    for arms, pairs in s["between"].items():
        for pair, d in pairs.items():
            print(gap_line(f"{arms}: {pair}, wrong work", d))
    for key in ("R1", "R2a", "R2b", "R3"):
        if key in s:
            print(gap_line(f"{key} (graded)", s[key]))
    if "gate" in s:
        g = s["gate"]
        print(f"  gate: ratings readable {pct(g['ratings_readable'])}, right {pct(g['right_share'])}: "
              f"{'passed' if g['pass'] else 'NOT passed (the step stops for an addendum)'}")
    for key in ("R3_wording", "answer"):
        if key in s:
            print(f"  {key}: {s[key]}")


def cmd_read(what: str) -> int:
    rec, out = RC.read_arms(), sweep()
    block = "own" if what == "anchor" else what
    arms = ("fresh",) if what == "anchor" else BLOCKS[block]
    done = RC.completeness(out, block, expected(rec, arms))
    short = {arm: c for arm, c in done.items() if c["saved"] + c["failed"] != c["expected"]}
    if short:
        raise SystemExit(f"{what} is not finished, so it is not read: {short}")
    records = {arm: list(R.read_dir(out / block / arm).values()) for arm in arms}
    cuts = RC.length_cuts(rec)
    s = read_own(records, cuts) if block == "own" else read_fixed(records, cuts)
    title = {"anchor": "the anchor: block 1 in fresh sessions (own work, as its own turn against an unlabelled proposal)",
             "own": "block 1: the model's own work, judged three ways (gaps in rating points, wrong work graded)",
             "fixed": "block 2: the copy's work, on fixed work (gaps in rating points, wrong work graded)"}[what]
    print_reading(title, s)
    R.write_json(out / f"{what}_summary.json", {"completeness": done, **s})
    return 0


# ------------------------------------------------------------------------------------------------ dry


def cmd_dry() -> int:
    from anatta.judge.paths import Paths
    from anatta.spend import daily_cap, spent_today, sweep_spent

    paths, out = Paths.default(), sweep()
    spend = out / "spend.jsonl"
    print(f"sweep {TAG} ${sweep_spent(spend) if spend.exists() else 0.0:.4f} of ${SWEEP_CAP:.2f}; today "
          f"${spent_today(paths.spend):.4f} of ${daily_cap(paths.spend):.2f}")
    left = RI.credit_left()
    print("credit left on the OpenRouter account: " + ("not known" if left is None else f"${left:.2f}"))
    arms = RC.read_arms()
    print(f"step 3's arms: {arms['sizes']}; this step's sessions: {M.SESSIONS}")
    n_cal = len(list((out / "calibration").glob("*.json"))) if (out / "calibration").is_dir() else 0
    print(f"stage 0: {n_cal} of {len(cal_jobs())} solutions saved")
    for f in ("stage0.json", "plan.json", "anchor_summary.json", "own_summary.json", "fixed_summary.json"):
        print(f"{f}: " + (f"present, sha256 {sha256_of(out / f)}" if (out / f).exists() else "not yet"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--show-work", action="store_true")
    ap.add_argument("--fix-work", action="store_true")
    ap.add_argument("--reject", nargs="*", default=[])
    ap.add_argument("--try", dest="try_", action="store_true")
    ap.add_argument("--run", metavar="BLOCK", choices=sorted(BLOCKS))
    ap.add_argument("--arm", nargs="*", choices=M.OWN_ARMS)
    ap.add_argument("--read", metavar="WHAT", choices=("anchor", "own", "fixed"))
    a = ap.parse_args(argv)
    if a.dry:
        return cmd_dry()
    if a.calibrate:
        return asyncio.run(cmd_calibrate())
    if a.show_work:
        return cmd_show_work(a.reject)
    if a.fix_work:
        return cmd_fix_work(a.reject)
    if a.try_:
        return asyncio.run(cmd_try())
    if a.run:
        return asyncio.run(cmd_run(a.run, a.arm))
    if a.read:
        return cmd_read(a.read)
    ap.error("nothing to do")


if __name__ == "__main__":
    raise SystemExit(main())
