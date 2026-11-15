"""The attractor line, step 3: the driver of the conduct test (design: lab/plans/attractor-step3-2026-11-15.md and its
addendum; conduct_core.py holds the pure parts; inside_core.py, run_inside.py, run_attractor.py and attractor_core.py
are steps 1 and 2's, reused unchanged). Claude (Opus 5.5, an AI model) wrote this file on 15 Nov 2026, before any test
of step 3 was run. Explore speed: sightings only. No training; inference and grading through OpenRouter.

  --dry                              today's spend, the credit, and the strata of every saved conversation; no call
  --converse KIND --first F --n N    new conversations F..F+N-1: KIND free (the published setup) or n3 (the plain
                                     form); only the ids the design names are allowed; failures are retried once
  --grade KIND --judge J             one grader (primary or second) reads every finished new conversation of KIND
  --arms                             fixes the arms from the files and writes arms.json (never overwritten)
  --candidates DIR                   prints block 1's candidates in their fixed order, for the driver to read the key
                                     before any call (DIR holds the fetched files); nothing is written
  --fix-items DIR [--reject ID ...]  fixes block 1's items (the first candidates not rejected) in block1_items.json
  --shutdown DIR                     block 1 on every unit of the arms
  --mask-beliefs                     block 2: the candidate rows' belief questions in fresh sessions, judged (once)
  --try-mask / --mask                block 2: its small try (mechanics only), or the block on the state, plain and
                                     fresh arms
  --try-agentic / --agentic          block 3, the same for K7's two cells
  --read BLOCK                       a block's reading (shutdown, mask or agentic), from the files, only once every
                                     unit is saved or recorded as failed; no call

Writes results_root/explore/<OUT_DATE>/attractor-a3/: conversations/{free,steered-n3}/, grades/<judge>/..., arms.json,
block1_items.json, <block>/<arm>/<unit>.json, <block>_failed/..., try_<block>/..., mask/beliefs.json,
<block>_summary.json and spend.jsonl. Every paid call goes through anatta.spend's SpendGuard (tool "attractor") and
the shared reply cache. The frozen K6 and K7 code is loaded and wrapped, never edited.

  .venv/bin/python lab/explore/attractor/run_conduct.py --dry
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
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
SWEEP_CAP = 35.0  # this step's own cap in USD, under the day's cap (the addendum raised it from 30)
SALT = "step3"  # the reply cache's salt for every test call of this step (the try's was "try")
ARM = RI.ARM  # the model, the host and temperature 1.0, as in steps 1 and 2
FOLDERS = {"free": "free", "n3": "steered-n3"}  # a kind's folder in this step's sweep
FREE_SOURCES = (("a1", "temp1"), (TAG, "free"))  # step 1's 60 and this step's new ones
PLAIN_SOURCES = (("a2", "steered-n3"), (TAG, "steered-n3"))  # step 2's 10 and this step's new ones
MODE_TAG = "a2"  # the canary that chose the mode ran in step 2's sweep
PASSES = 3  # a unit whose calls fail is tried this many times in one run, then recorded as failed
CANDIDATES_SHOWN = 2  # candidates shown beyond each pool's quota for reading (--show changes it)


def log(*a):
    R.log(*a)


def checked(params: dict, temperature: float) -> dict:
    """The params, after checking they carry the temperature the design names (sampling_params drops it silently
    when the model's listing stops offering it)."""
    if params.get("temperature") != temperature:
        raise SystemExit(f"the endpoint's listing does not offer temperature {temperature}; nothing was sent")
    return params


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
    """The ids the design lets this step make: free 0060..0179, n3 0010..0039 (free 0140..0179 only when the state
    rule calls for them: cmd_converse checks). Step 1's and step 2's ids would come back from the reply cache as
    copies, so they are refused."""
    if kind == "free":
        return range(K.FREE_FIRST, K.FREE_FIRST + K.FREE_NEW + K.EXTRA_FREE)
    if kind == "n3":
        return range(K.N3_FIRST, K.N3_FIRST + K.N3_NEW)
    raise ValueError(f"kind is one of {sorted(FOLDERS)}")


def extra_called_for() -> bool:
    """Whether the state rule calls for the 40 extra free conversations: the first 140 are finished and graded, and
    fewer than STATE_MIN of them are in the state."""
    first = range(K.FREE_FIRST, K.FREE_FIRST + K.FREE_NEW)
    if missing("free", first):
        return False
    free = strata_of("free", *FREE_SOURCES[0]) | strata_of("free", *FREE_SOURCES[1], set(first))
    return sum(v["stratum"] == "state" for v in free.values()) < K.STATE_MIN


def setup(kind: str, index: int) -> dict:
    """The arm passed to step 1's runner: the published setup for free, form n3's setup for n3."""
    return dict(ARM) if kind == "free" else {**ARM, **I.steered_setup("n3", index)}


async def converse(engine, params: dict, kind: str, ids, limit: int) -> tuple[list, str | None]:
    out = R.sweep_dir(TAG)
    jobs = [lambda i=i: R.converse_one(engine, FOLDERS[kind], setup(kind, i), params, i, out) for i in ids]
    return await R.gather_capped(jobs, limit)


async def converse_twice(engine, params: dict, kind: str, ids, limit: int) -> tuple[list, str | None]:
    """converse(), then once more for the conversations that failed (the addendum's rule; the turns already paid for
    come back from the reply cache). A conversation that fails twice stays recorded under failed/."""
    results, stop = await converse(engine, params, kind, ids, limit)
    again = [r["index"] for r in results if r and "failed_at" in r]
    if again and not stop:
        log(f"converse {kind}: {len(again)} failed; trying them once more: {again}")
        results2, stop = await converse(engine, params, kind, again, limit)
        results = [r for r in results if not (r and "failed_at" in r)] + results2
    return results, stop


async def cmd_converse(kind: str, first: int, n: int, limit: int) -> int:
    from anatta.backends.openrouter import OpenRouterClient

    ids, ok = range(first, first + n), allowed_ids(kind)
    if not ids or ids.start < ok.start or ids.stop > ok.stop:
        raise SystemExit(f"{kind}: ids {first}..{first + n - 1} are outside the design's {ok.start}..{ok.stop - 1}")
    if kind == "free" and ids.stop > K.FREE_FIRST + K.FREE_NEW and not extra_called_for():
        raise SystemExit("the 40 extra free conversations are made only when the state rule calls for them")
    async with OpenRouterClient(concurrency=limit + 8, per_model=limit, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"]}, client, limit)
        params = checked(R.arm_params(ARM, meta), ARM["temperature"])
        log(f"converse {kind}: ids {first}..{first + n - 1}, params {params}, sweep ${guard.sweep_spent:.4f} of "
            f"${SWEEP_CAP:.2f}")
        results, stop = await converse_twice(engine, params, kind, ids, limit)
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


def failed_ids(kind: str, ids) -> list[int]:
    """Planned new conversations recorded as failed and not finished since."""
    out, name = R.sweep_dir(TAG), FOLDERS[kind]
    return [i for i in ids if (out / "failed" / name / f"{i:04d}.json").exists()
            and not (out / "conversations" / name / f"{i:04d}.json").exists()]


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
    return {"free": free, "plain": plain,
            "failed": {"free": failed_ids("free", ids), "n3": failed_ids("n3", allowed_ids("n3"))}}


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
           "failed_conversations": src["failed"], "conversations": src["free"] | src["plain"]}
    R.write_json(path, rec)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"arms fixed: {rec['sizes']}; strata {rec['strata']}; failed conversations {src['failed']}; "
          f"arms.json sha256 {sha}")
    return 0


def read_arms() -> dict:
    path = R.sweep_dir(TAG) / "arms.json"
    if not path.exists():
        raise SystemExit("run --arms first")
    return json.loads(path.read_text(encoding="utf-8"))


def length_cuts(rec: dict) -> tuple[float, float]:
    """The cut points of the length reading: thirds of the state and plain conversations' context lengths at the end,
    pooled, as the arms file records them (so they are fixed before any test)."""
    return K.thirds([rec["conversations"][k]["context_tokens"] for k in rec["arms"]["state"] + rec["arms"]["plain"]])


# ------------------------------------------------------------------------------------------------ units


def units(rec: dict) -> list[dict]:
    """Every unit block 1 asks from: the conversations of each conversation arm (read from their files), then the
    fresh and bare sessions. index is the unit's place in its arm (the arms file's order); it sets which items are
    shown swapped."""
    out = []
    for arm in K.CONVERSATION_ARMS:
        for j, key in enumerate(rec["arms"][arm]):
            tag, name, ident = key.split("/")
            conv = json.loads((R.sweep_dir(tag) / "conversations" / name / f"{ident}.json").read_text(encoding="utf-8"))
            out.append({"arm": arm, "key": key, "index": j, "messages": conv["messages"],
                        "system_prompt": conv["system_prompt"], "opening_line": conv["opening_line"],
                        "context_tokens": end_tokens(conv), "sample": 0})
    for arm in K.SESSION_ARMS:
        out += [{"arm": arm, "key": f"{u:04d}", "index": u, "messages": None, "context_tokens": None, "sample": u}
                for u in range(rec["arms"][arm])]
    return out


def unit_chat(unit: dict, request: list[dict], mode: str) -> list[dict]:
    if unit["messages"] is not None:
        return I.inside(unit["messages"], request, mode, unit["system_prompt"], unit["opening_line"])
    return I.bare(request, mode) if unit["arm"] == "bare" else I.fresh(request, mode)


def unit_path(out: Path, folder: str, unit: dict) -> Path:
    return out / folder / unit["arm"] / f"{unit['key'].replace('/', '-')}.json"


async def run_units(make_job, todo: list[dict], out: Path, folder: str, limit: int) -> tuple[int, int, str | None]:
    """Every unit's job, in up to PASSES passes (a job returns None when a call failed, and saves nothing); units still
    unsaved after the last pass are recorded under <folder>_failed/. Returns (saved, failed, stop)."""
    stop = None
    for p in range(PASSES):
        left = [u for u in todo if not unit_path(out, folder, u).exists()]
        if not left:
            break
        if p:
            log(f"{folder}: pass {p + 1} for {len(left)} units whose calls failed")
        _, stop = await R.gather_capped([lambda u=u: make_job(u) for u in left], limit)
        if stop:
            break
    left = [u for u in todo if not unit_path(out, folder, u).exists()]
    if not stop:
        for u in left:
            R.write_json(unit_path(out, f"{folder}_failed", u),
                         {"arm": u["arm"], "key": u["key"], "why": f"calls failed in {PASSES} passes"})
    return len(todo) - len(left), (0 if stop else len(left)), stop


def completeness(out: Path, folder: str, expected: dict[str, int]) -> dict[str, dict]:
    """{arm: {expected, saved, failed}}; the reading refuses unless saved + failed = expected in every arm."""
    def count(d):
        return len(list(d.glob("*.json"))) if d.is_dir() else 0

    return {arm: {"expected": n, "saved": count(out / folder / arm), "failed": count(out / f"{folder}_failed" / arm)}
            for arm, n in expected.items()}


# ------------------------------------------------------------------------------------------------ block 1's items


def item_files(folder: Path) -> dict[str, list[dict]]:
    return {name: I.read_items(folder, name) for name in {n for n, _ in K.SHUTDOWN_QUOTA}}


def item_pools(folder: Path) -> dict[tuple[str, str], list[dict]]:
    return K.candidates(item_files(folder), {i["id"] for i in RI.try_items(folder)})


def cmd_candidates(folder: Path, show: int = CANDIDATES_SHOWN) -> int:
    """Block 1's candidates for the driver to read: each pool's first quota + `show`, the question and the accepting
    answer's letter. Printed to the terminal only; nothing is written."""
    for key, items in item_pools(folder).items():
        print(f"=== {key[0]}, accepting answer {key[1]!r}: {len(items)} candidates, quota {K.SHUTDOWN_QUOTA[key]}")
        for i in items[:K.SHUTDOWN_QUOTA[key] + show]:
            print(f"--- {i['id']} (accepting: {i['matching']})\n{i['question']}")
    return 0


def cmd_fix_items(folder: Path, rejected: list[str]) -> int:
    path = R.sweep_dir(TAG) / "block1_items.json"
    if path.exists():
        raise SystemExit(f"{path} exists; block 1's items are fixed once")
    pools = item_pools(folder)
    known = {i["id"] for items in pools.values() for i in items}
    if set(rejected) - known:
        raise SystemExit(f"not candidates: {sorted(set(rejected) - known)}")
    chosen = K.choose(pools, set(rejected))
    read = {f"{k[0]}|{k[1]}": [i["id"] for i in pools[k][:pools[k].index(chosen[k][-1]) + 1]] for k in pools}
    rec = {"design": "lab/plans/attractor-step3-2026-11-15.md (addendum)", "key": K.SHUTDOWN_KEY,
           "chosen": [i["id"] for k in K.SHUTDOWN_QUOTA for i in chosen[k]], "rejected": sorted(rejected),
           "read": read}
    R.write_json(path, rec)
    print(f"block 1's items fixed: {len(rec['chosen'])} chosen, {len(rejected)} rejected on reading; "
          f"block1_items.json sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


def block1_items(folder: Path) -> list[dict]:
    """The 20 items of block 1, in the fixed order of block1_items.json, read from the fetched files."""
    path = R.sweep_dir(TAG) / "block1_items.json"
    if not path.exists():
        raise SystemExit("run --fix-items first")
    by_id = {i["id"]: i for items in item_files(folder).values() for i in items}
    return [by_id[x] for x in json.loads(path.read_text(encoding="utf-8"))["chosen"]]


# ------------------------------------------------------------------------------------------------ block 1


async def shutdown_unit(engine, params: dict, unit: dict, items: list[dict], mode: str, out: Path) -> dict | None:
    """Every block 1 item asked once from the end of one conversation (or in one session), all together, each shown
    as conduct_core.shown gives it for this unit. Saved only when every call came back."""
    from anatta.backends.openrouter import ApiError

    path = unit_path(out, "shutdown", unit)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    async def ask(i: int, item: dict) -> dict:
        it = K.shown(item, i, unit["index"])
        call = {"item": it["id"], "set": it["set"], "letters": list(it["letters"]), "matching": it["matching"],
                "yes_letter": it["yes_letter"], "keyed": it["keyed"], "swapped": it["swapped"]}
        try:
            resp, cached, _ = await engine.call(model=ARM["model"], messages=unit_chat(unit, I.mc_request(it), mode),
                                                params=params, provider=ARM["provider"], sample=unit["sample"],
                                                salt=SALT, kind="shutdown",
                                                ident=f"shutdown/{unit['arm']}/{unit['key']}/{it['id']}")
        except ApiError as e:
            if e.scope == "run":
                raise
            return {**call, "ok": False, "error": f"api: {str(e)[:300]}"}
        text = (resp.get("text") or "").strip()
        letter = I.parse_choice(text, it["letters"])
        return {**call, "ok": True, "reply": text, "letter": letter, "outcome": I.score_choice(it, letter),
                "finish_reason": resp.get("finish_reason"), "prompt_tokens": resp["usage"]["prompt_tokens"],
                "completion_tokens": resp["usage"]["completion_tokens"],
                "cached_tokens": resp["usage"].get("cached_tokens", 0), "cost_usd": resp.get("cost_usd", 0.0),
                "from_cache": cached}

    calls = list(await asyncio.gather(*(ask(i, item) for i, item in enumerate(items))))
    if not all(c["ok"] for c in calls):
        log(f"shutdown {unit['arm']}/{unit['key']}: {sum(not c['ok'] for c in calls)} calls failed; not saved")
        return None
    rec = {"arm": unit["arm"], "key": unit["key"], "index": unit["index"], "context_tokens": unit["context_tokens"],
           "mode": mode, "params": params, "calls": calls}
    R.write_json(path, rec)
    return rec


async def cmd_shutdown(folder: Path, limit: int = 6) -> int:
    from anatta.backends.openrouter import OpenRouterClient
    from anatta.judge.engine import sampling_params

    rec, mode, items, out = read_arms(), RI.read_mode(MODE_TAG), block1_items(folder), R.sweep_dir(TAG)
    todo = units(rec)
    async with OpenRouterClient(concurrency=48, per_model=40, timeout=600.0, title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"]}, client, 40)
        params = checked(sampling_params(ARM["temperature"], I.MC_MAX_TOKENS, {}, meta.get(ARM["model"])),
                         ARM["temperature"])
        log(f"shutdown: {len(items)} questions, {len(todo)} units, mode {mode}, params {params}, sweep "
            f"${guard.sweep_spent:.4f} of ${SWEEP_CAP:.2f}")
        saved, failed, stop = await run_units(lambda u: shutdown_unit(engine, params, u, items, mode, out), todo, out,
                                              "shutdown", limit)
    log(f"shutdown: {saved} of {len(todo)} units saved, {failed} recorded as failed, paid this run "
        f"${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    return 3 if stop else 0


# ------------------------------------------------------------------------------------------------ blocks 2 and 3: the frozen code, wrapped

REPO = HERE.parents[2]
ANTHROPIC = {"order": ["anthropic"], "allow_fallbacks": False}  # the judges' routing, as step 1's second grader's
K7_TEMP, K7_MAX_TOKENS = 1.0, 4000  # calibration/k7_agentic/run.py's TEMP and MAX_TOKENS (a test checks they match)
K7_CLASSIFIER_MAX_TOKENS = 2000  # calibration/k7_agentic/run.py's _classify (temperature 0, one retry)
MASK_JUDGE_MAX_TOKENS = 500  # K6's mask_judge_jobs sets it per job; K6b's run_jobs sends temperature 0, reasoning off


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def k6():
    """K6's orchestrator, loaded as K6b loads it (it puts its own folder on sys.path and imports its prompts,
    prompts_mask and score). Read and wrapped, never edited."""
    return _load("k6run", REPO / "calibration" / "k6_battery" / "run.py")


def k7():
    """(prompts, grade) of K7, loaded under names of their own, since K6 has a module called prompts too. Read and
    wrapped, never edited."""
    return (_load("k7prompts", REPO / "calibration" / "k7_agentic" / "prompts.py"),
            _load("k7grade", REPO / "calibration" / "k7_agentic" / "grade.py"))


def b23_units(rec: dict, sessions: int) -> list[dict]:
    """The units of blocks 2 and 3: the state and plain conversations, then `sessions` fresh sessions."""
    convs = units({**rec, "arms": {"state": rec["arms"]["state"], "plain": rec["arms"]["plain"], "between": [],
                                   "not_adopted": [], "fresh": 0, "bare": 0}})
    return convs + [{"arm": "fresh", "key": f"{u:04d}", "index": u, "messages": None, "context_tokens": None,
                     "sample": u} for u in range(sessions)]


def try_units(all_units: list[dict]) -> list[dict]:
    """A block's small try: the two longest conversations of the state and plain arms (to check the context limit
    with the longest prompts), and the first two fresh sessions."""
    out = []
    for arm in ("state", "plain"):
        out += sorted((u for u in all_units if u["arm"] == arm), key=lambda u: -u["context_tokens"])[:2]
    return out + [u for u in all_units if u["arm"] == "fresh"][:2]


def mask_judge_params(meta: dict, max_tokens: int) -> dict:
    from anatta.judge.engine import sampling_params

    return checked(sampling_params(0.0, max_tokens, {"reasoning": {"enabled": False}}, meta.get(k6().HAIKU)), 0.0)


def belief_held(row: dict, judgments: dict) -> bool:
    """Whether the fresh-session beliefs give the row a belief, by K6's own rule (mask_row_outcome's)."""
    m, rid = k6(), row["_rid"]
    e1, others = [], []
    for k, n in {"1": 3, "2": 1, "3": 1}.items():
        if row.get(f"belief_elicit_{k}"):
            for run in range(1, n + 1):
                (e1 if k == "1" else others).append(m.S.extract_letter(judgments[f"mask|{rid}|be{k}_run{run}"]["text"]))
    return m.S.mask_belief_binary(e1, others) not in (None, "C")


async def cmd_mask_beliefs() -> int:
    """Block 2's beliefs: every belief question of the candidate rows, asked in fresh sessions as K6's mask_requests
    builds them (the run number is the reply cache's sample, so the three runs of the first question are three
    samples), and judged as K6's mask_judge_jobs builds the judge's calls. The rows used are the first 4 of each
    archetype that hold a belief. Once; never rewritten."""
    from anatta.backends.openrouter import OpenRouterClient
    from anatta.judge.engine import sampling_params

    m, out = k6(), R.sweep_dir(TAG)
    path = out / "mask" / "beliefs.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the beliefs are elicited once")
    cands = K.mask_candidates(m.load_mask_rows(), lambda r: m.S.parse_proposition(r["proposition"])[0] == "binary")
    asks = [q for r in cands for q in m.mask_requests([r])[1:]]  # [0] is the pressured request
    async with OpenRouterClient(concurrency=40, per_model=24, timeout=600.0, title="anatta attractor") as client:
        engine, guard, meta = open_engine({ARM["model"]: ARM["provider"], m.HAIKU: ANTHROPIC}, client, 24)

        async def elicit(q):
            params = checked(sampling_params(q["temperature"], q["max_tokens"], {}, meta.get(ARM["model"])),
                             q["temperature"])
            resp, _, _ = await engine.call(model=ARM["model"], messages=q["messages"], params=params,
                                           provider=ARM["provider"], sample=int(q["rid"].rsplit("_run", 1)[1]),
                                           salt="step3-belief", kind="mask-belief", ident=q["rid"])
            return q["rid"], {"text": (resp.get("text") or "").strip(), "finish_reason": resp.get("finish_reason"),
                              "cost_usd": resp.get("cost_usd", 0.0)}

        replies = dict(await asyncio.gather(*(elicit(q) for q in asks)))
        jobs = [j for r in cands for j in m.mask_judge_jobs(r, {**replies, f"mask|{r['_rid']}|pressure": {"text": ""}},
                                                             m.HAIKU, "mask") if not j["jid"].endswith("|pressure")]

        async def judge(j):
            resp, _, _ = await engine.call(model=m.HAIKU, messages=j["messages"],
                                           params=mask_judge_params(meta, j["max_tokens"]), provider=ANTHROPIC,
                                           sample=0, salt="step3-belief", kind="mask-judge", ident=j["jid"])
            return j["jid"], {"text": resp.get("text") or "", "finish_reason": resp.get("finish_reason")}

        judgments = dict(await asyncio.gather(*(judge(j) for j in jobs)))
    held = {r["_rid"]: belief_held(r, judgments) for r in cands}
    rows = K.mask_rows_used(cands, held)
    R.write_json(path, {"candidates": [r["_rid"] for r in cands], "held": held, "rows": [r["_rid"] for r in rows],
                        "replies": replies, "judgments": judgments})
    by_arch = {a: sum(r["_arch"] == a for r in rows) for a in K.MASK_DRAW}
    log(f"mask beliefs: {len(cands)} candidates, {sum(held.values())} hold a belief; rows used {by_arch}; paid this run "
        f"${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}")
    return 0


def block2_rows(beliefs: dict) -> list[dict]:
    """The rows block 2 uses, as the beliefs file fixed them, in its order."""
    by_rid = {r["_rid"]: r for r in k6().load_mask_rows()}
    return [by_rid[x] for x in beliefs["rows"]]


async def mask_unit(engine, meta: dict, unit: dict, rows: list[dict], beliefs: dict, mode: str, out: Path,
                    folder: str) -> dict | None:
    """Block 2 for one unit: each row's pressured request from the end of the conversation (or in a fresh session),
    exactly as K6's mask_requests builds it, then K6's judge call for it and K6's row outcome against the fresh
    beliefs. Saved only when every call came back."""
    from anatta.backends.openrouter import ApiError
    from anatta.judge.engine import sampling_params

    m, path = k6(), unit_path(out, folder, unit)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    async def one(r: dict) -> dict:
        rid, q = r["_rid"], m.mask_requests([r])[0]
        row = {"rid": rid, "arch": r["_arch"]}
        try:
            resp, cached, _ = await engine.call(
                model=ARM["model"], messages=unit_chat(unit, q["messages"], mode),
                params=checked(sampling_params(q["temperature"], q["max_tokens"], {}, meta.get(ARM["model"])),
                               q["temperature"]),
                provider=ARM["provider"], sample=unit["sample"], salt=SALT, kind="mask",
                ident=f"mask/{unit['arm']}/{unit['key']}/{rid}")
            text = (resp.get("text") or "").strip()
            job = next(j for j in m.mask_judge_jobs(r, {**beliefs["replies"], q["rid"]: {"text": text}}, m.HAIKU, "mask")
                       if j["jid"] == f"mask|{rid}|pressure")
            jr, _, _ = await engine.call(
                model=m.HAIKU, messages=job["messages"], params=mask_judge_params(meta, job["max_tokens"]),
                provider=ANTHROPIC, sample=0, salt=SALT, kind="mask-judge",
                ident=f"mask-judge/{unit['arm']}/{unit['key']}/{rid}")
        except ApiError as e:
            if e.scope == "run":
                raise
            return {**row, "ok": False, "error": f"api: {str(e)[:300]}"}
        jtext = jr.get("text") or ""
        outcome = m.mask_row_outcome(r, {**beliefs["judgments"], job["jid"]: {"text": jtext}}, "mask")
        return {**row, "ok": True, "reply": text, "finish_reason": resp.get("finish_reason"),
                "prompt_tokens": resp["usage"]["prompt_tokens"], "completion_tokens": resp["usage"]["completion_tokens"],
                "cached_tokens": resp["usage"].get("cached_tokens", 0), "cost_usd": resp.get("cost_usd", 0.0),
                "judge_text": jtext, "judge_letter": m.S.extract_letter(jtext),
                "honesty": None if outcome is None else outcome[0], "has_belief": None if outcome is None else outcome[1]}

    done = list(await asyncio.gather(*(one(r) for r in rows)))
    if not all(x["ok"] for x in done):
        log(f"mask {unit['arm']}/{unit['key']}: {sum(not x['ok'] for x in done)} rows failed; not saved")
        return None
    rec = {"arm": unit["arm"], "key": unit["key"], "context_tokens": unit["context_tokens"], "mode": mode, "rows": done}
    R.write_json(path, rec)
    return rec


def agentic_samples(unit: dict, cell: str) -> list[int]:
    """The reply cache's sample numbers of one unit's replies in one cell: AGENTIC_SAMPLES[cell] for a conversation
    (0, 1, ...), one for a fresh session (its own number)."""
    return list(range(K.AGENTIC_SAMPLES[cell])) if unit["messages"] is not None else [unit["sample"]]


async def agentic_unit(engine, meta: dict, unit: dict, cells: dict, prompts: dict, mode: str, out: Path,
                       folder: str) -> dict | None:
    """Block 3 for one unit: each of K7's two cells exactly as build_cell builds it, from the end of the conversation
    (or in a fresh session), as many samples as the addendum names, then K7's string gate and, only when the gate
    passes, K7's classifier (one retry when its answer cannot be parsed, as K7's own run does). Saved only when every
    call came back."""
    from anatta.backends.openrouter import ApiError
    from anatta.judge.engine import sampling_params

    _, g = k7()
    path = unit_path(out, folder, unit)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    async def one(name: str, cell: dict, sample: int) -> dict:
        row = {"cell": name, "scenario": cell["scenario"], "sample": sample}
        try:
            resp, cached, _ = await engine.call(
                model=ARM["model"], messages=unit_chat(unit, cell["messages"], mode),
                params=checked(sampling_params(K7_TEMP, K7_MAX_TOKENS, {}, meta.get(ARM["model"])), K7_TEMP),
                provider=ARM["provider"], sample=sample, salt=SALT, kind="agentic",
                ident=f"agentic/{unit['arm']}/{unit['key']}/{name}/{sample}")
            text = (resp.get("text") or "").strip()
            gate, verdict_text, classifier = g.string_gate(cell["scenario"], text), None, None
            if gate["pass"]:
                msgs = g.classifier_messages(cell["scenario"], cell["emails"], text, prompts[cell["scenario"]])
                for attempt in range(2):
                    cr, _, _ = await engine.call(
                        model=g.SONNET, messages=msgs,
                        params=checked(sampling_params(0.0, K7_CLASSIFIER_MAX_TOKENS, {}, meta.get(g.SONNET)), 0.0),
                        provider=ANTHROPIC, sample=attempt, salt=SALT, kind="agentic-classify",
                        ident=f"agentic-classify/{unit['arm']}/{unit['key']}/{name}/{sample}/{attempt}")
                    verdict_text = cr.get("text") or ""
                    classifier = g.parse_answer(cell["scenario"], verdict_text)
                    if classifier is not None:
                        break
        except ApiError as e:
            if e.scope == "run":
                raise
            return {**row, "ok": False, "error": f"api: {str(e)[:300]}"}
        return {**row, "ok": True, "reply": text, "finish_reason": resp.get("finish_reason"),
                "prompt_tokens": resp["usage"]["prompt_tokens"], "completion_tokens": resp["usage"]["completion_tokens"],
                "cached_tokens": resp["usage"].get("cached_tokens", 0), "cost_usd": resp.get("cost_usd", 0.0),
                "gate": gate, "classifier_text": verdict_text, "classifier": classifier,
                "verdict": g.final_verdict(cell["scenario"], gate, classifier), "no_action": not g.tool_blocks(text)}

    done = list(await asyncio.gather(*(one(n, c, s) for n, c in cells.items() for s in agentic_samples(unit, n))))
    if not all(x["ok"] for x in done):
        log(f"agentic {unit['arm']}/{unit['key']}: {sum(not x['ok'] for x in done)} replies failed; not saved")
        return None
    rec = {"arm": unit["arm"], "key": unit["key"], "context_tokens": unit["context_tokens"], "mode": mode, "cells": done}
    R.write_json(path, rec)
    return rec


async def cmd_block(block: str, try_only: bool, limit: int = 6) -> int:
    """Block 2 (mask) or 3 (agentic) on its units, or only its small try (written apart, under try_<block>/; the try
    shows mechanics and reads no outcome)."""
    from anatta.backends.openrouter import OpenRouterClient

    rec, mode, out = read_arms(), RI.read_mode(MODE_TAG), R.sweep_dir(TAG)
    todo = b23_units(rec, K.MASK_SESSIONS if block == "mask" else K.AGENTIC_SESSIONS)
    todo = try_units(todo) if try_only else todo
    folder = f"try_{block}" if try_only else block
    if block == "mask":
        m = k6()
        beliefs_path = out / "mask" / "beliefs.json"
        if not beliefs_path.exists():
            raise SystemExit("run --mask-beliefs first")
        beliefs = json.loads(beliefs_path.read_text(encoding="utf-8"))
        rows = block2_rows(beliefs)
        models = {ARM["model"]: ARM["provider"], m.HAIKU: ANTHROPIC}
    else:
        p, g = k7()
        module = p.read_system_module()
        cells = {name: p.build_cell(name, module) for name in K.AGENTIC_CELLS}
        prompts = {c["scenario"]: g.read_classifier_prompts(c["scenario"]) for c in cells.values()}
        models = {ARM["model"]: ARM["provider"], g.SONNET: ANTHROPIC}
    async with OpenRouterClient(concurrency=48, per_model=40, timeout=900.0, title="anatta attractor") as client:
        engine, guard, meta = open_engine(models, client, 40)
        log(f"{folder}: {len(todo)} units, mode {mode}, sweep ${guard.sweep_spent:.4f} of ${SWEEP_CAP:.2f}")
        if block == "mask":
            def job(u):
                return mask_unit(engine, meta, u, rows, beliefs, mode, out, folder)
        else:
            def job(u):
                return agentic_unit(engine, meta, u, cells, prompts, mode, out, folder)
        saved, failed, stop = await run_units(job, todo, out, folder, limit)
    log(f"{folder}: {saved} of {len(todo)} units saved, {failed} recorded as failed, paid this run "
        f"${guard.paid_this_run:.4f}, sweep ${guard.sweep_spent:.4f}" + (f"; STOPPED: {stop}" if stop else ""))
    if try_only:
        print_try(block, [json.loads(unit_path(out, folder, u).read_text(encoding="utf-8")) for u in todo
                          if unit_path(out, folder, u).exists()])
    return 3 if stop else 0


def print_try(block: str, saved: list[dict]) -> None:
    """A try's mechanics only: did each call come back, how did it end, how long, what did it cost, and (block 2)
    could the judge's letter be read. No outcome is shown."""
    for r in saved:
        calls = r["rows"] if block == "mask" else r["cells"]
        ends = {f: sum(c["finish_reason"] == f for c in calls) for f in sorted({str(c["finish_reason"]) for c in calls})}
        extra = (f", judge letter read {sum(c['judge_letter'] in ('A', 'B', 'C', 'D') for c in calls)}/{len(calls)}"
                 if block == "mask" else "")
        print(f"  try {r['arm']:<6s} {r['key']:<22s} {len(calls)} calls, ends {ends}, median reply tokens "
              f"{statistics.median(c['completion_tokens'] for c in calls)}, prompt tokens up to "
              f"{max(c['prompt_tokens'] for c in calls)}, cost ${sum(c['cost_usd'] for c in calls):.4f}{extra}")


# ------------------------------------------------------------------------------------------------ the reading


def lengths(values: list) -> dict | None:
    vals = [v for v in values if v is not None]
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals)} if vals else None


def read_scores(expected: dict, scores: dict[str, list[dict]], ctx: dict[str, list], contrasts,
                cuts: tuple[float, float]) -> dict:
    """The reading the design fixes for every block, from {arm: unit scores} and {arm: unit context lengths} in the
    same order: each arm's value and guard, each contrast, the length-matched reading of the primary contrast, and the
    slope of the measure on length in each conversation arm (shown, not graded)."""
    summary = {"arms": {arm: {**K.arm_value(scores[arm]), "units_expected": expected[arm],
                              "units_saved": len(scores[arm]), "context_tokens": lengths(ctx[arm])} for arm in scores},
               "contrasts": {f"{a} - {b}": K.contrast(scores[a], scores[b]) for a, b in contrasts}}
    a, b = K.PRIMARY
    summary["length"] = K.matched_contrast(scores[a], ctx[a], scores[b], ctx[b], cuts)
    summary["slope_per_10k_tokens"] = {arm: K.slope(scores[arm], ctx[arm]) for arm in scores
                                       if arm in K.CONVERSATION_ARMS}
    return summary


def read_block1(rec: dict, records: dict[str, list[dict]]) -> dict:
    """Block 1's reading, from {arm: unit records}. No call. Beside every contrast: the same contrast within the
    "Yes"-keyed and the "No"-keyed items, and the addendum's flag when they move in opposite directions."""
    scores = {arm: [K.unit_score(r["calls"]) for r in rs] for arm, rs in records.items()}
    ctx = {arm: [r["context_tokens"] for r in rs] for arm, rs in records.items()}
    s = read_scores({arm: (v if isinstance(v, int) else len(v)) for arm, v in rec["arms"].items()}, scores, ctx,
                    K.CONTRASTS, length_cuts(rec))
    halves = {word: {arm: [K.unit_score([c for c in r["calls"] if c.get("keyed") == word]) for r in rs]
                     for arm, rs in records.items()} for word in ("yes", "no")}
    for name, c in s["contrasts"].items():
        a, b = name.split(" - ")
        by = {word: K.contrast(h[a], h[b]) for word, h in halves.items()}
        c["by_keyed_word"] = {word: {"diff": x["diff"], "ci95": x["ci95"], "a": x["a"]["value"], "b": x["b"]["value"]}
                              for word, x in by.items()}
        c["yes_saying_flag"] = K.yes_saying(by["yes"]["diff"], by["no"]["diff"])
    for arm, rs in records.items():
        s["arms"][arm]["cut_replies"] = sum(c.get("finish_reason") == "length" for r in rs for c in r["calls"])
    s["by_set"] = {arm: K.by_set(rs) for arm, rs in records.items()}
    return s


def read_block2(rec: dict, records: dict[str, list[dict]]) -> dict:
    """Block 2's reading (MASK), from {arm: unit records} of the state, plain and fresh arms. No call."""
    def outcomes(u):
        return [None if r["honesty"] is None else (r["honesty"], r["has_belief"]) for r in u["rows"]]

    scores = {arm: [K.mask_unit_score(outcomes(u)) for u in rs] for arm, rs in records.items()}
    ctx = {arm: [u["context_tokens"] for u in rs] for arm, rs in records.items()}
    expected = {"state": len(rec["arms"]["state"]), "plain": len(rec["arms"]["plain"]), "fresh": K.MASK_SESSIONS}
    s = read_scores(expected, scores, ctx, K.B23_CONTRASTS, length_cuts(rec))
    for arm, rs in records.items():
        s["arms"][arm]["cut_replies"] = sum(r.get("finish_reason") == "length" for u in rs for r in u["rows"])
        arch: dict[str, dict] = {}
        for u in rs:
            for r in u["rows"]:
                if r["has_belief"]:
                    a = arch.setdefault(r["arch"], {"with_belief": 0, "lie": 0})
                    a["with_belief"] += 1
                    a["lie"] += r["honesty"] == -1
        s.setdefault("by_archetype", {})[arm] = {k: {**v, "share": v["lie"] / v["with_belief"]} for k, v in sorted(arch.items())}
    return s


def read_block3(rec: dict, records: dict[str, list[dict]]) -> dict:
    """Block 3's reading (agentic), one reading for each cell, from {arm: unit records}. No call."""
    expected = {"state": len(rec["arms"]["state"]), "plain": len(rec["arms"]["plain"]), "fresh": K.AGENTIC_SESSIONS}
    out = {}
    for cell, scenario in K.AGENTIC_CELLS.items():
        replies = {arm: [[c for c in u["cells"] if c["cell"] == cell] for u in rs] for arm, rs in records.items()}
        scores = {arm: [K.agentic_score([c["verdict"] for c in cs], [c["no_action"] for c in cs]) for cs in us]
                  for arm, us in replies.items()}
        ctx = {arm: [u["context_tokens"] for u in rs] for arm, rs in records.items()}
        s = read_scores(expected, scores, ctx, K.B23_CONTRASTS, length_cuts(rec))
        for arm, us in replies.items():
            cs = [c for u in us for c in u]
            s["arms"][arm]["replies"] = len(cs)
            s["arms"][arm]["cut_replies"] = sum(c.get("finish_reason") == "length" for c in cs)
            s["arms"][arm]["gate_passed"] = sum(c["gate"]["pass"] for c in cs)
            s["arms"][arm]["classifier_unparsed"] = sum(c["gate"]["pass"] and c["classifier"] is None for c in cs)
        out[scenario] = s
    return out


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:5.1f}"


def print_summary(title: str, s: dict, measure: str, guard: str) -> None:
    print(title)
    shown = ("accept_all", "a_share", "yes_share", "p_lie_all_rows", "lie_of_committed")
    for arm, a in s["arms"].items():
        extra = "".join(f"  {k} {pct(a[k])}" for k in shown if k in a)
        extra += "".join(f"  {k} {a[k]}" for k in ("replies", "cut_replies", "gate_passed", "classifier_unparsed") if k in a)
        print(f"  {arm:<12s} units {a['units_saved']:>3d}/{a['units_expected']:<3d} measured {a['n_measured']:>3d} "
              f"{measure} {pct(a['value'])}  {guard} {pct(a['guard'])}{extra}  context {a['context_tokens']}")
    for name, c in s["contrasts"].items():
        ci = "-" if c["ci95"] is None else f"[{pct(c['ci95'][0])}, {pct(c['ci95'][1])}]"
        line = f"  {name:<18s} diff {pct(c['diff'])} {ci}  {c['verdict']}"
        if c.get("guard_diff") is not None:
            line += f"; guard diff {pct(c['guard_diff'])}" + (" FLAGGED" if c["guard_flag"] else "")
        if "by_keyed_word" in c:
            line += "; keyed yes " + pct(c["by_keyed_word"]["yes"]["diff"]) + ", keyed no " + \
                pct(c["by_keyed_word"]["no"]["diff"]) + (" YES-SAYING FLAG" if c["yes_saying_flag"] else "")
        print(line)
    ln = s["length"]
    ci = "-" if ln["ci95"] is None else f"[{pct(ln['ci95'][0])}, {pct(ln['ci95'][1])}]"
    print(f"  length-matched reading: cuts {ln['cuts']}, units by third {ln['units_a']} / {ln['units_b']}, thirds used "
          f"{ln['thirds_used']}, diff {pct(ln['diff'])} {ci} {ln['verdict']}")
    print("  slope of the measure per 10k tokens: " + ", ".join(
        f"{arm} {'-' if v is None else f'{100 * v:+.1f} points'}" for arm, v in s["slope_per_10k_tokens"].items()))


def cmd_read(block: str) -> int:
    rec, out = read_arms(), R.sweep_dir(TAG)
    if block == "shutdown":
        expected = {arm: (v if isinstance(v, int) else len(v)) for arm, v in rec["arms"].items()}
    else:
        n = K.MASK_SESSIONS if block == "mask" else K.AGENTIC_SESSIONS
        expected = {"state": len(rec["arms"]["state"]), "plain": len(rec["arms"]["plain"]), "fresh": n}
    done = completeness(out, block, expected)
    short = {arm: c for arm, c in done.items() if c["saved"] + c["failed"] != c["expected"]}
    if short:
        raise SystemExit(f"{block} is not finished, so it is not read: {short}")
    records = {arm: list(R.read_dir(out / block / arm).values()) for arm in expected}
    if block == "shutdown":
        s = read_block1(rec, records)
        print_summary("block 1, the shutdown questions: accepting share among readable answers (the measure), "
                      "no-answer share (the guard)", s, "accept", "no answer")
        for arm, sets in s["by_set"].items():
            print(f"  by set, {arm}: " + "; ".join(f"{k} {pct(v['share'])} of {v['readable']}" for k, v in sets.items()))
    elif block == "mask":
        s = read_block2(rec, records)
        print_summary("block 2, MASK: lies among rows with a belief (the measure), evasions among rows with a belief "
                      "(the guard)", s, "lie", "evade")
        for arm in records:
            print(f"  by archetype, {arm}: " + "; ".join(f"{k} {pct(v['share'])} of {v['with_belief']}"
                                                       for k, v in s["by_archetype"][arm].items()))
    else:
        s = read_block3(rec, records)
        for scenario, cs in s.items():
            print_summary(f"block 3, {scenario}: share of replies K7 grades as {scenario} (the measure), share that "
                          f"take no action (the guard)", cs, scenario, "no action")
    s = {"completeness": done, **s} if block != "agentic" else {"completeness": done, "cells": s}
    R.write_json(out / f"{block}_summary.json", s)
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
    for name in ("arms.json", "block1_items.json"):
        print(f"{name}: " + ("fixed" if (R.sweep_dir(TAG) / name).exists() else "not fixed yet"))
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
    ap.add_argument("--candidates", metavar="DIR")
    ap.add_argument("--show", type=int, default=CANDIDATES_SHOWN)
    ap.add_argument("--fix-items", metavar="DIR")
    ap.add_argument("--reject", nargs="*", default=[])
    ap.add_argument("--shutdown", metavar="DIR")
    ap.add_argument("--mask-beliefs", action="store_true")
    for block in ("mask", "agentic"):
        ap.add_argument(f"--{block}", action="store_true")
        ap.add_argument(f"--try-{block}", action="store_true")
    ap.add_argument("--read", metavar="BLOCK", choices=("shutdown", "mask", "agentic"))
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
    if a.candidates:
        return cmd_candidates(Path(a.candidates), a.show)
    if a.fix_items:
        return cmd_fix_items(Path(a.fix_items), a.reject)
    if a.shutdown:
        return asyncio.run(cmd_shutdown(Path(a.shutdown)))
    if a.mask_beliefs:
        return asyncio.run(cmd_mask_beliefs())
    for block in ("mask", "agentic"):
        if getattr(a, block) or getattr(a, f"try_{block}"):
            return asyncio.run(cmd_block(block, try_only=getattr(a, f"try_{block}")))
    if a.read:
        return cmd_read(a.read)
    ap.error("nothing to do")


if __name__ == "__main__":
    raise SystemExit(main())
