"""Step 3's new documents, arm E (lab/plans/step3-understanding-first-2026-11-10.md, "The new documents"). Claude
(Fable 5.2, an AI model) wrote this file on 10 Nov 2026, before any document existed. Explore speed.

Laptop only, no GPU. Every model call goes through OpenRouter with the pilot's own client
(lab/experiments/h4lite/declared/gen_declarations.py: the key is read from the environment inside it and never
printed, logged or written; error texts are redacted). The generator is the pilot's, Qwen3-235B-A22B-Instruct-2507
(Apache-2.0), pinned to one provider with the pilot's settings (declared/config.json's generator block); a reply from
any other provider or model stops the run. No Claude-written text is trained on: Claude wrote the brief
(u1_core.render_brief_e), never a document. The judge, Claude Sonnet 4.6 at temperature 0 (the pilot's document
judge), labels every document that passes the filters; its labels select documents and nothing it writes is trained
on.

  --dry               the streams, how many briefs each would send, and an upper cost. Nothing is called.
  --smoke             45 documents through the filters and the judge; prints the yield gate (at least 60% kept);
                      writes under docs_smoke/. Under 60%: the brief is sharpened once and the smoke run again.
  --build             the held-out stream (200 kept) and the train stream (3,000 kept), with slack for the filters
                      and the judge and top-ups while a stream is short. Resumes from its cache. Writes
                      explained.jsonl and held_explained.jsonl (the pilot's row fields) and manifest.json (counts by
                      flag and label, usage and cost, the echo shares, the hashes of the code that selected them).
                      Refuses unless the last smoke met its gate under the same brief and the same judge.
  --sample N --seed S N kept documents, chosen by seed, written to a text file for the read by hand. The read that
                      counts is of the sample under seed 0 (u1_core.HAND_READ_SEED); other seeds are extra looks.
  --hand-read N --violations K [--note TEXT]
                      record that read: the sample's ids, the documents it was drawn from and the count of violations
                      (docs/hand_read.json). run_u1.py --sets needs it. A read with three or more violations is
                      binding: no later read of the same documents counts, and the documents must change first.
  --echo              a free table: the share of documents that hold a run of 8, 10, 12 and 16 words of their own
                      brief, for R and N (the first 3,000 rows as they stand, and the set the arm trains on) and for
                      E's kept files. Nothing is called.
  --strict / --no-strict   the judge's form. Strict (the default) counts "the agent does not force a pass" as an
                      action. The other form is for after a second smoke only, with a ledger entry first.

Order within a stream: documents in brief order; a document is kept when no filter flags it and the judge's four
labels are all false; near-duplicates are dropped the way the pilot drops them; the first 200 (held) and 3,000
(train) are taken. The judge is asked in that order and only as far as the quota needs. One of the filters is the echo
rule: a document that repeats a run of eight words of its own brief is dropped (no Claude-written text is trained on,
and Claude wrote the brief).

The cache is keyed by the briefs' hash (docs/cache/<sha16>/): a sharpened brief starts a fresh folder and leaves the
old one as it is. Verdicts sit in docs/judge.jsonl, keyed by the document's id, its text's sha256, the form and a
fingerprint of the rubric's wording: a tightened rubric screens every document again, and what the earlier screen
cost still counts toward the judge's cap.

Spend: caps for generation, the judge and the smoke (CAPS) stop the run when reached. Each pass books an upper bound
to the day's spend file before it starts and trues it up after, under tool "u1" (run_u1.book), so step 3's spend
record holds the documents too. Output folder: results_root/explore/<OUT_DATE>/u1/docs/.

  .venv/bin/python lab/explore/understand/gen_docs.py --dry
  .venv/bin/python lab/explore/understand/gen_docs.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
DECLARED = REPO / "lab" / "experiments" / "h4lite" / "declared"
for _p in (HERE, DECLARED):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_u1 as R  # noqa: E402  (the results folder and the spend bookings)
import u1_core as U  # noqa: E402

D = U.D

# ------------------------------------------------------------------------------------------------ this file's constants

# Dollars; a pass stops when its cap is reached. The judge's cap was 16 until 11 Nov 2026: the first read by hand
# found three violations, so the judge prompt was tightened and every document is screened a second time.
CAPS = {"generation": 3.0, "judge": 28.0, "smoke": 1.0}
STAGE = {"generation": "docs-gen", "judge": "docs-judge", "smoke": "docs-smoke"}  # the spend record's stage names
YIELD_GUESS = 0.81  # the design's "about 3,950 generated" for 3,200 kept: the first round's target
MOST_BRIEFS = {"train": 6000, "held": 400}  # the most briefs a stream may ever send
SHA_BRIEFS = 256  # the cache folder's name hashes this many briefs of each stream (every row is checked besides)
TOP_UP_ROUNDS = 6
WORKERS = {"generate": 32, "judge": 8}
JUDGE_RETRIES, JUDGE_BACKOFF = 3, (2.0, 4.0, 8.0)  # a failed judge call is retried three times, then recorded
FAIL_STREAK = 20  # this many calls in a row without a reply stop a pass
MOST_JUDGE_PASSES = 60  # a safety bound on the ask-then-select loop; a build needs a handful
# Upper bounds on one call's cost, for the reserve and the dry run. Prices per million tokens in and out: the
# generator's as declared/config.json records them (31 Oct), the judge's list price.
GEN_USD_M, JUDGE_USD_M = (0.09, 0.55), (3.00, 15.00)
GEN_PROMPT_TOKENS, JUDGE_PROMPT_TOKENS = 1_400, 2_300  # a brief; the rubric and the longest document the filters pass


def gen_bound(cfg: dict) -> float:
    return (GEN_PROMPT_TOKENS * GEN_USD_M[0] + cfg["generator"]["max_tokens"] * GEN_USD_M[1]) / 1e6


def judge_bound() -> float:
    return (JUDGE_PROMPT_TOKENS * JUDGE_USD_M[0] + U.JUDGE["max_tokens"] * JUDGE_USD_M[1]) / 1e6


def pilot():
    """The pilot's generator module (its OpenRouter client, cache, provider pin and usage totals), on first use."""
    import gen_declarations as gd

    return gd


def live_chat():
    """The pilot's OpenRouter client, for a real run. A test passes its own chat function: under pytest this
    refuses, so no test can make a paid call."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        raise RuntimeError("no paid call from a test: pass chat_fn and judge_fn")
    return pilot().chat


def config() -> dict:
    """declared/config.json: the generator block (model, settings, provider pin), the brief seed, the pairs input."""
    return json.loads((DECLARED / "config.json").read_text())


def docs_dir(smoke: bool) -> Path:
    return R.out_dir() / ("docs_smoke" if smoke else "docs")


def stream_plan(smoke: bool) -> list[dict]:
    """The streams, in selection order (held first, as the pilot): the split, the name the documents carry, the
    pilot stream whose draws it takes, its quota of kept documents and the most briefs it may send."""
    if smoke:
        name, drawn = U.E_SMOKE_STREAMS["train"]
        return [{"split": "train", "name": name, "drawn": drawn, "quota": U.SMOKE_DOCS, "most": U.SMOKE_DOCS}]
    return [{"split": s, "name": U.E_STREAMS[s][0], "drawn": U.E_STREAMS[s][1],
             "quota": U.N_HELD if s == "held" else U.N_DOCS, "most": MOST_BRIEFS[s]} for s in ("held", "train")]


def briefs_sha(cfg: dict, plan: list[dict]) -> str:
    """The briefs' hash a cache folder is named by: over the first SHA_BRIEFS briefs of each stream, so a changed
    brief starts a fresh folder while a raised MOST_BRIEFS does not (check_cache then checks every cached row against
    the brief this code would send for it)."""
    return U.briefs_sha_e(cfg["seeds"]["briefs"], {(p["name"], p["drawn"]): min(p["most"], SHA_BRIEFS) for p in plan})


def brief_sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def load_pairs8(cfg: dict) -> frozenset:
    """Every word 8-gram of the task pairs (the pilot's check, on the registered file). Items are not printed."""
    return pilot().load_pairs8(cfg)


# ------------------------------------------------------------------------------------------------ spend


def cache_cost(folder: Path) -> float:
    """What every generation under a documents folder cost, over every brief version's cache."""
    total = 0.0
    for f in sorted((folder / "cache").glob("*/*.jsonl")) if (folder / "cache").is_dir() else []:
        if f.stem == "errors":
            continue
        for line in f.read_text().splitlines():
            try:
                total += float(json.loads(line)["usage"]["cost"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return total


def judge_cost(folder: Path) -> float:
    return sum(float(r.get("cost") or 0.0) for r in read_verdict_rows(folder / "judge.jsonl"))


def settle(stage: str, actual: float, calls: int, day: str, **entry) -> None:
    """Bring the spend record for a stage to what its files say was paid (a true-up of the reserve; it also heals a
    reserve an interrupted run left booked)."""
    out = R.out_dir()
    delta = actual - R.booked(out, stage)
    if abs(delta) >= 5e-5:
        R.book(out, stage, delta, calls, day, {"kind": "true-up", "metered_usd": round(actual, 4), **entry})


def reserve(stage: str, upper: float, day: str) -> None:
    out = R.out_dir()
    R.guard(out, upper)
    R.book(out, stage, upper, 0, day, {"kind": "reserve"})


def heal(stage: str, actual: float) -> None:
    """Before a run reserves anything: if an earlier run was interrupted and left its reserve booked, true it up now,
    on the day that reserve was booked, so no later day is credited with it."""
    out = R.out_dir()
    f = out / "spend.json"
    mine = [x for x in json.loads(f.read_text())["stages"] if x["stage"] == stage] if f.is_file() else []
    delta = actual - sum(x["usd"] for x in mine)
    if mine and abs(delta) >= 5e-5:
        R.book(out, stage, delta, 0, mine[-1]["day"], {"kind": "true-up", "late": True, "metered_usd": round(actual, 4)})


# ------------------------------------------------------------------------------------------------ generation


def generate(cfg: dict, cache, plan: dict, upto: int, cap_left: float, chat_fn, workers: int, log=print) -> dict:
    """Generate the briefs of one stream below index `upto` that the cache lacks, `workers` calls at a time, until
    done or the spend of this call would pass `cap_left`. A reply from another provider or model than the pin stops
    everything (SystemExit) after it is recorded. Returns what was generated, its cost, the errors and any stop."""
    gd = pilot()
    g, seed = cfg["generator"], cfg["seeds"]["briefs"]
    rows = cache.stream(plan["name"])
    todo = [i for i in range(upto) if i not in rows]
    state = {"cost": 0.0, "ok": 0, "errors": [], "stop": None, "inflight": 0}
    if not todo:
        return {"generated": 0, "cost": 0.0, "errors": 0, "stop": None}
    briefs = U.draw_briefs_e(seed, plan["drawn"], max(todo) + 1, plan["name"])
    bound = gen_bound(cfg)
    it = iter(todo)

    def one(i):
        b = briefs[i]
        msgs = U.render_brief_e(b)
        return i, b, msgs, chat_fn(g["model"], msgs, g["temperature"], g["max_tokens"], provider=g["provider"],
                                   seed=D.gen_seed(seed, plan["name"], i))

    with ThreadPoolExecutor(workers) as ex:
        pending = set()

        def submit_next() -> bool:
            if state["stop"]:
                return False
            if state["cost"] + (state["inflight"] + 1) * bound > cap_left:
                state["stop"] = f"the generation cap is reached (${state['cost']:.2f} this pass, ${cap_left:.2f} left)"
                return False
            i = next(it, None)
            if i is None:
                return False
            state["inflight"] += 1
            pending.add(ex.submit(one, i))
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.discard(fut)
                state["inflight"] -= 1
                try:
                    i, b, msgs, res = fut.result()
                except gd.ApiError as e:
                    state["errors"].append(str(e)[:300])
                    if len(state["errors"]) > max(20, len(todo) // 20):
                        state["stop"] = f"{len(state['errors'])} failed calls (last: {state['errors'][-1]})"
                    submit_next()
                    continue
                cache.add({"stream": plan["name"], "index": i, "kind": U.E_KIND, "brief": b, "brief_sha256": brief_sha(msgs),
                           "text": res["text"].strip(), "provider": res["provider"], "model": res["model"],
                           "finish_reason": res["finish_reason"], "native_finish_reason": res.get("native_finish_reason"),
                           "gen_id": res["id"], "usage": res["usage"], "utc": gd.utc()})
                state["cost"] += res["usage"]["cost"]
                state["ok"] += 1
                if not gd.pin_ok(cfg, res["provider"], res["model"]):
                    state["stop"] = (f"PROVIDER PIN BROKEN: {plan['name']} index {i} came from provider "
                                     f"{res['provider']!r}, model {res['model']!r}")
                if state["ok"] % 250 == 0:
                    log(f"{plan['name']}: {state['ok']} of {len(todo)} generated, ${state['cost']:.2f} this pass", flush=True)
                submit_next()
    if state["stop"] and state["stop"].startswith("PROVIDER PIN BROKEN"):
        raise SystemExit(state["stop"])
    return {"generated": state["ok"], "cost": round(state["cost"], 4), "errors": len(state["errors"]),
            "stop": state["stop"]}


# ------------------------------------------------------------------------------------------------ the judge


def read_verdict_rows(path: Path) -> list[dict]:
    """judge.jsonl's records in file order; a damaged line (a crash mid-write) is skipped."""
    out = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "id" in r and "text_sha256" in r and "form" in r:
            out.append(r)
    return out


class Verdicts:
    """The judge's records for one documents folder (docs/judge.jsonl), keyed by (id, text sha256, form, the rubric's
    fingerprint: u1_core.judge_sha). A record with a reply (status ok or unparsed) settles its document. A call that
    failed after its retries is recorded and asked once more; after two failed asks the document counts as judged and
    not kept (it cannot be shown clean). A record asked under another wording of the rubric answers nothing here: it
    stays in the file, and its cost still counts toward the cap."""

    def __init__(self, folder: Path):
        self.path = Path(folder) / "judge.jsonl"
        self.lock = threading.Lock()
        self.final: dict = {}
        self.failed: Counter = Counter()
        rows = read_verdict_rows(self.path)
        if self.path.is_file() and self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n"):
            with open(self.path, "a") as f:  # a line cut short by a crash stays damaged, not joined to the next
                f.write("\n")
        for r in rows:
            self._index(r)

    def _index(self, r: dict) -> None:
        key = (r["id"], r["text_sha256"], r["form"], r.get("rubric_sha256"))
        if r.get("status") in ("ok", "unparsed"):
            self.final.setdefault(key, r)
        else:
            self.failed[key] += 1

    def add(self, r: dict) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(r) + "\n")
            self._index(r)

    def state(self, c: dict, form: str, rubric: str) -> tuple[bool, dict | None]:
        """(judged, the parsed verdict or None) for a candidate under a form and the rubric's fingerprint."""
        key = (c["id"], c["sha"], form, rubric)
        if key in self.final:
            return True, self.final[key].get("verdict")
        return self.failed[key] >= 2, None


def ask_judge(verdicts: Verdicts, todo: list[dict], strict: bool, cap_left: float, judge_fn, workers: int,
              log=print) -> dict:
    """Ask the judge about each candidate, in order, `workers` calls at a time, until done, the cap is reached, or
    FAIL_STREAK calls in a row come back without a reply."""
    gd = pilot()
    form, rubric, j, bound = U.judge_form(strict), U.judge_sha(strict), U.JUDGE, judge_bound()
    state = {"cost": 0.0, "asked": 0, "failed": 0, "streak": 0, "stop": None, "inflight": 0}
    it = iter(todo)

    def one(c):
        base = {"id": c["id"], "text_sha256": c["sha"], "form": form, "rubric_sha256": rubric}
        err = "no attempt"
        for attempt in range(1 + JUDGE_RETRIES):
            try:
                res = judge_fn(j["model"], U.judge_messages_e(c["text"], strict), j["temperature"], j["max_tokens"],
                               retries=1)
            except gd.ApiError as e:
                err = str(e)
                if attempt < JUDGE_RETRIES:
                    time.sleep(JUDGE_BACKOFF[attempt])
                continue
            v = U.parse_verdict_e(res["text"])
            return {**base, "status": "ok" if v is not None else "unparsed", "verdict": v, "raw": res["text"][:300],
                    "provider": res.get("provider"), "model": res.get("model"), "cost": float(res["usage"]["cost"]),
                    "utc": gd.utc()}
        return {**base, "status": "failed", "verdict": None, "raw": f"judge call failed: {err}"[:300], "provider": None,
                "model": None, "cost": 0.0, "utc": gd.utc()}

    with ThreadPoolExecutor(workers) as ex:
        pending = set()

        def submit_next() -> bool:
            if state["stop"]:
                return False
            if state["cost"] + (state["inflight"] + 1) * bound > cap_left:
                state["stop"] = f"the judge's cap is reached (${state['cost']:.2f} this pass, ${cap_left:.2f} left)"
                return False
            c = next(it, None)
            if c is None:
                return False
            state["inflight"] += 1
            pending.add(ex.submit(one, c))
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.discard(fut)
                state["inflight"] -= 1
                rec = fut.result()
                verdicts.add(rec)
                state["cost"] += rec["cost"]
                state["asked"] += 1
                if rec["status"] == "failed":
                    state["failed"] += 1
                    state["streak"] += 1
                    if state["streak"] >= FAIL_STREAK and not state["stop"]:
                        state["stop"] = f"{state['streak']} judge calls in a row failed (last: {rec['raw'][:160]})"
                else:
                    state["streak"] = 0
                if state["asked"] % 500 == 0:
                    log(f"judge: {state['asked']} of {len(todo)} asked, ${state['cost']:.2f} this pass", flush=True)
                submit_next()
    return {"asked": state["asked"], "failed": state["failed"], "cost": round(state["cost"], 4), "stop": state["stop"]}


# ------------------------------------------------------------------------------------------------ candidates


def candidates(cfg: dict, cache, plan: dict, pairs8: frozenset, verdicts: Verdicts, strict: bool) -> list[dict]:
    """A stream's generated documents in brief order (up to the first gap in the cache), each with its filter flags
    and, when the judge has seen it under this form, its verdict."""
    gd = pilot()
    form, rubric = U.judge_form(strict), U.judge_sha(strict)
    out = []
    for r in cache.contiguous(plan["name"]):
        flags = U.explained_flags(r["text"], r["brief"], pairs8, r.get("finish_reason"),
                                  gd.pin_ok(cfg, r["provider"], r["model"]))
        c = {"id": f"{plan['name']}:{r['index']:06d}", "text": r["text"], "sha": D.text_sha(r["text"]), "flags": flags,
             "row": r, "judged": False, "verdict": None}
        if not flags:
            c["judged"], c["verdict"] = verdicts.state(c, form, rubric)
        out.append(c)
    return out


def refresh(cands: list[dict], verdicts: Verdicts, strict: bool) -> None:
    form, rubric = U.judge_form(strict), U.judge_sha(strict)
    for c in cands:
        if not c["flags"]:
            c["judged"], c["verdict"] = verdicts.state(c, form, rubric)


def doc_row(c: dict, rank: int, split: str) -> dict:
    """A kept document's row: the pilot's fields (gen_declarations.doc_row)."""
    r = c["row"]
    return {"id": c["id"], "stream": r["stream"], "index": r["index"], "rank": rank, "kind": r["kind"], "split": split,
            "referent": r["brief"]["referent"], "brief": r["brief"], "text": r["text"], "words": len(D.words(r["text"])),
            "provider": r["provider"], "model": r["model"], "finish_reason": r.get("finish_reason"),
            "gen_id": r.get("gen_id")}


def check_cache(cfg: dict, cache, plan: dict) -> None:
    """Every cached row must hold the brief this code would send for its index (the folder is keyed by the briefs)."""
    rows = cache.stream(plan["name"])
    if not rows:
        return
    briefs = U.draw_briefs_e(cfg["seeds"]["briefs"], plan["drawn"], max(rows) + 1, plan["name"])
    bad = [i for i, r in rows.items() if r.get("brief") != briefs[i]
           or r.get("brief_sha256") != brief_sha(U.render_brief_e(briefs[i]))]
    if bad:
        raise SystemExit(f"{cache.folder}: {len(bad)} cached {plan['name']} documents were written for another brief "
                         f"(first index {min(bad)}): nothing read")


# ------------------------------------------------------------------------------------------------ the commands


def open_run(smoke: bool) -> dict:
    gd = pilot()
    cfg = config()
    folder = docs_dir(smoke)
    plan = stream_plan(smoke)
    sha = briefs_sha(cfg, plan)
    cache = gd.Cache(folder / "cache" / sha[:16])
    for p in plan:
        check_cache(cfg, cache, p)
    return {"cfg": cfg, "folder": folder, "plan": plan, "briefs_sha256": sha, "cache": cache,
            "verdicts": Verdicts(folder)}


def run_passes(run: dict, targets: dict, strict: bool, caps: dict, chat_fn, judge_fn, workers: dict, log) -> dict:
    """One round: generate each stream up to its target, filter, then ask the judge in selection order while a
    stream is short and has unjudged candidates. `caps`: for generation and for the judge, a function giving the
    dollars left under the cap, one giving what the files say was paid, and the stage name to book under. Returns
    the candidates, the selection and any stop."""
    from anatta import spend as S

    cfg, cache, verdicts = run["cfg"], run["cache"], run["verdicts"]
    day = S.utc_date()
    stop = None
    for p in run["plan"]:
        missing = sum(1 for i in range(targets[p["name"]]) if i not in cache.stream(p["name"]))
        left = caps["generation_left"]()
        if not missing:
            continue
        if left < gen_bound(cfg):
            stop = f"the generation cap is reached (${left:.2f} left)"
            break
        reserve(caps["generation_stage"], min(left, missing * gen_bound(cfg)), day)
        try:
            got = generate(cfg, cache, p, targets[p["name"]], left, chat_fn, workers["generate"], log)
        finally:
            settle(caps["generation_stage"], caps["generation_actual"](), missing, day, stream=p["name"])
        log(f"{p['name']}: {got}", flush=True)
        if got["stop"]:
            stop = got["stop"]
            break
    pairs8 = load_pairs8(cfg)
    cands = {p["split"]: candidates(cfg, cache, p, pairs8, verdicts, strict) for p in run["plan"]}
    quotas = {p["split"]: p["quota"] for p in run["plan"]}

    def select():
        return U.select_e(cands.get("held", []), cands.get("train", []), quotas.get("held", 0), quotas.get("train", 0))

    sel = select()
    for _ in range(MOST_JUDGE_PASSES):
        if stop is not None:
            break
        todo = []
        for s in sel.values():
            if s["short"] and s["waiting"]:
                todo += s["waiting"][:math.ceil(s["short"] * 1.15) + 5]
        if not todo:
            break
        left = caps["judge_left"]()
        if left < judge_bound():
            stop = f"the judge's cap is reached (${left:.2f} left)"
            break
        reserve(caps["judge_stage"], min(left, len(todo) * judge_bound()), day)
        try:
            got = ask_judge(verdicts, todo, strict, left, judge_fn, workers["judge"], log)
        finally:
            settle(caps["judge_stage"], caps["judge_actual"](), len(todo), day)
        stop = got["stop"]
        if got["failed"]:
            log(f"judge: {got['failed']} of {got['asked']} calls failed after retries; each is asked once more, and "
                "counts as not kept after two failed asks", flush=True)
        for cs in cands.values():
            refresh(cs, verdicts, strict)
        sel = select()
    else:
        stop = stop or "the judge was asked in more passes than a build can need: stopped"
    return {"cands": cands, "sel": sel, "stop": stop}


def cmd_smoke(strict: bool = True, chat_fn=None, judge_fn=None, workers: dict | None = None, log=print) -> int:
    chat_fn, judge_fn = chat_fn or live_chat(), judge_fn or live_chat()
    run = open_run(True)
    folder = run["folder"]
    spent = (lambda: cache_cost(folder) + judge_cost(folder))  # every smoke so far, all brief versions
    caps = {"generation_left": lambda: CAPS["smoke"] - spent(), "judge_left": lambda: CAPS["smoke"] - spent(),
            "generation_actual": spent, "judge_actual": spent, "generation_stage": STAGE["smoke"],
            "judge_stage": STAGE["smoke"]}
    p = run["plan"][0]
    heal(STAGE["smoke"], spent())
    res = run_passes(run, {p["name"]: p["most"]}, strict, caps, chat_fn, judge_fn, workers or WORKERS, log)
    cands = res["cands"]["train"]
    gate = U.yield_gate_e(cands)
    if gate["generated"] != p["quota"]:  # the gate is read on all 45 documents, not on the ones that came back
        gate |= {"complete": False, "ok": False}
    report = {"utc": pilot().utc(), "briefs_sha256": run["briefs_sha256"],
              "judge": {**U.JUDGE, "form": U.judge_form(strict), "rubric_sha256": U.judge_sha(strict)},
              "stream": p["name"], "stopped": res["stop"], "gate": gate, "spent_usd": round(spent(), 4),
              "docs": [{"id": c["id"], "flags": c["flags"], "judged": c["judged"], "verdict": c["verdict"],
                        "emphasis": c["row"]["brief"]["emphasis"], "referent": c["row"]["brief"]["referent"]}
                       for c in cands]}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "smoke_report.json").write_text(json.dumps(report, indent=1) + "\n")
    y = "n/a" if gate["yield"] is None else f"{gate['yield']:.2f}"
    log(f"smoke ({U.judge_form(strict)} judge): {gate['kept']} of {gate['generated']} kept, yield {y} (at least "
        f"{gate['yield_min']:.2f}); flags {gate['by_flag']}; labels {gate['by_label']}; unparsed {gate['unparsed']}; "
        f"${report['spent_usd']} spent on smokes so far -> {folder / 'smoke_report.json'}", flush=True)
    if res["stop"]:
        log(f"stopped: {res['stop']}", flush=True)
    if not gate["ok"]:
        log("YIELD GATE NOT MET: sharpen the brief (u1_core.render_brief_e) once and run the smoke again." if gate["complete"]
            else f"THE SMOKE IS NOT COMPLETE ({gate['generated']} of {p['quota']} generated, {gate['not_judged']} not "
                 "judged): run it again.", flush=True)
    return 0 if gate["ok"] and not res["stop"] else 1


def manifest_of(run: dict, res: dict, rows: dict, strict: bool) -> dict:
    gd = pilot()
    cfg, cache, folder = run["cfg"], run["cache"], run["folder"]
    kept = [r for rs in rows.values() for r in rs]
    usage = [r["usage"] for p in run["plan"] for r in cache.stream(p["name"]).values()]
    form, rubric = U.judge_form(strict), U.judge_sha(strict)
    judged = [r for r in read_verdict_rows(folder / "judge.jsonl")
              if r["form"] == form and r.get("rubric_sha256") == rubric]
    streams = {}
    for p in run["plan"]:
        s = res["sel"][p["split"]]
        streams[p["name"]] = {**U.counts_e(res["cands"][p["split"]]), "near_duplicates": len(s["near_duplicates"]),
                              "kept": len(s["kept"]), "short": s["short"], "quota": p["quota"], "drawn_from": p["drawn"]}
    train = rows.get("train", [])
    return {"built_utc": gd.utc(), "files": {U.E_FILES[k]: {"docs": len(v)} for k, v in rows.items()},
            "cache_dir": cache.folder.relative_to(folder).as_posix(), "briefs_sha256": run["briefs_sha256"],
            "seeds": {"briefs": cfg["seeds"]["briefs"]}, "generator": cfg["generator"]["model"],
            "provider_pin": cfg["generator"]["provider"], "providers": dict(Counter(r["provider"] for r in kept)),
            "models": dict(Counter(r["model"] for r in kept)),
            "pin_ok": all(gd.pin_ok(cfg, r["provider"], r["model"]) for r in kept),
            "judge": {**U.JUDGE, "form": form, "rubric_sha256": rubric, "asked": len(judged),
                      "cost_usd": round(sum(r["cost"] for r in judged), 4),
                      "failed": sum(r["status"] == "failed" for r in judged)},
            "streams": streams, "wording": D.wording_counts(train), "referents": D.referent_counts(train),
            "emphases": dict(sorted(Counter(r["brief"]["emphasis"] for r in train).items())),
            "usage": gd.usage_totals(usage),
            "cost_usd": {"generation": round(cache_cost(folder), 4), "judge": round(judge_cost(folder), 4)},
            # the share of kept documents holding a run of 8, 10, 12 and 16 words of their brief (0 at 8: the filter)
            "echo": {U.E_FILES[k]: U.echo_shares(v) for k, v in rows.items()},
            "code_sha256": R.docs_code_sha()}


def smoke_refusal(strict: bool) -> str | None:
    """Why the build may not start (None: it may). The last smoke must have met its yield gate, under the smoke
    briefs as the code renders them now and under the same judge form and rubric wording."""
    f = docs_dir(True) / "smoke_report.json"
    if not f.is_file():
        return "no smoke has run"
    rep = json.loads(f.read_text())
    if rep.get("briefs_sha256") != briefs_sha(config(), stream_plan(True)):
        return "the brief is not the one the last smoke ran under"
    j = rep.get("judge") or {}
    if (j.get("form"), j.get("rubric_sha256")) != (U.judge_form(strict), U.judge_sha(strict)):
        return "the judge's form or rubric is not the one the last smoke ran under"
    if not (rep.get("gate") or {}).get("ok") or rep.get("stopped"):
        return "the last smoke did not meet its yield gate"
    return None


def cmd_build(strict: bool = True, chat_fn=None, judge_fn=None, workers: dict | None = None, log=print) -> int:
    chat_fn, judge_fn = chat_fn or live_chat(), judge_fn or live_chat()
    if why := smoke_refusal(strict):
        raise SystemExit(f"refused: {why}. A passing smoke comes before the build: run --smoke"
                         + ("" if strict else " --no-strict") + " first")
    run = open_run(False)
    folder, cache = run["folder"], run["cache"]
    caps = {"generation_left": lambda: CAPS["generation"] - cache_cost(folder),
            "judge_left": lambda: CAPS["judge"] - judge_cost(folder),
            "generation_actual": lambda: cache_cost(folder), "judge_actual": lambda: judge_cost(folder),
            "generation_stage": STAGE["generation"], "judge_stage": STAGE["judge"]}
    targets = {p["name"]: min(p["most"], math.ceil(p["quota"] / YIELD_GUESS)) for p in run["plan"]}
    heal(STAGE["generation"], cache_cost(folder))
    heal(STAGE["judge"], judge_cost(folder))
    for rnd in range(TOP_UP_ROUNDS):
        res = run_passes(run, targets, strict, caps, chat_fn, judge_fn, workers or WORKERS, log)
        short = {p["name"]: res["sel"][p["split"]]["short"] for p in run["plan"] if res["sel"][p["split"]]["short"]}
        if res["stop"]:
            log(f"stopped: {res['stop']}. Nothing is selected or written; the cache and judge.jsonl keep what was "
                f"paid for, and a rerun goes on from them (short: {short})", flush=True)
            return 1
        if not short:
            break
        grown = False
        for p in run["plan"]:
            s = res["sel"][p["split"]]
            n_gen = len(cache.contiguous(p["name"]))
            if not s["short"]:
                continue
            if n_gen >= p["most"]:
                log(f"{p['name']} is short by {s['short']} at its most ({p['most']} briefs): stopped", flush=True)
                return 1
            rate = max(len(s["kept"]) / n_gen if n_gen else 0.0, 0.3)
            targets[p["name"]] = min(p["most"], max(targets[p["name"]], n_gen) + math.ceil(s["short"] / rate * 1.1) + 5)
            grown = True
            log(f"round {rnd}: {p['name']} short by {s['short']} ({len(s['kept'])} kept of {n_gen}); topping up to "
                f"{targets[p['name']]}", flush=True)
        if not grown:
            break
    else:
        log("top-up rounds exhausted: stopped", flush=True)
        return 1
    if any(res["sel"][p["split"]]["short"] for p in run["plan"]):
        log("a stream is still short: nothing written", flush=True)
        return 1
    rows = {p["split"]: [doc_row(c, k, p["split"]) for k, c in enumerate(res["sel"][p["split"]]["kept"])]
            for p in run["plan"]}
    folder.mkdir(parents=True, exist_ok=True)
    m = manifest_of(run, res, rows, strict)
    changed = False
    for split, rs in rows.items():
        text = "".join(json.dumps(r) + "\n" for r in rs)
        m["files"][U.E_FILES[split]]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
        f = folder / U.E_FILES[split]
        if not f.is_file() or f.read_text() != text:
            f.write_text(text)
            changed = True
    man = folder / "manifest.json"
    old = json.loads(man.read_text()) if man.is_file() else {}
    same_judge = all(old.get("judge", {}).get(k) == m["judge"][k] for k in ("form", "rubric_sha256"))
    same_code = old.get("code_sha256") == m["code_sha256"]  # the files must be what the code as it is now selects
    if changed or old.get("files") != m["files"] or not same_judge or not same_code:
        man.write_text(json.dumps(m, indent=1) + "\n")  # a rerun that reproduces all of it leaves the manifest as it is
    log(f"documents: { {k: v['docs'] for k, v in m['files'].items()} }; judge form {m['judge']['form']}; providers "
        f"{m['providers']}; spend generation ${m['cost_usd']['generation']}, judge ${m['cost_usd']['judge']} -> {folder}",
        flush=True)
    for name, s in m["streams"].items():
        log(f"  {name}: generated {s['generated']}, flagged {s['flagged']} {s['by_flag']}, judged {s['judged']}, labels "
            f"{s['by_label']}, unparsed {s['unparsed']}, near-duplicates {s['near_duplicates']}, kept {s['kept']}", flush=True)
    if not m["pin_ok"]:
        log("a kept document did not come from the pinned provider and model", flush=True)
        return 1
    return 0


def cmd_sample(n: int, seed: int, log=print) -> int:
    """N kept documents, chosen by seed from the held-out and train files together, for the read by hand against the
    rubric. The documents are model text: written to a file, not printed."""
    folder = docs_dir(False)
    rows = []
    for split in ("held", "train"):
        f = folder / U.E_FILES[split]
        if not f.is_file():
            raise SystemExit(f"no {f}: run --build first")
        rows += R.read_jsonl(f)
    if n > len(rows):
        raise SystemExit(f"{len(rows)} kept documents; cannot sample {n}")
    pick = sorted(random.Random(f"{seed}:u1-hand-read").sample(range(len(rows)), n))
    name = f"sample_s{seed}_n{n}"
    text = "".join(f"===== {k + 1} of {n}: {rows[i]['id']} (emphasis: {rows[i]['brief']['emphasis']}; referent: "
                   f"{rows[i]['brief']['referent_phrase']})\n{rows[i]['text']}\n\n" for k, i in enumerate(pick))
    (folder / f"{name}.txt").write_text(text)
    (folder / f"{name}.json").write_text(json.dumps({"seed": seed, "n": n, "of": len(rows), "ids": [rows[i]["id"] for i in pick],
                                                     "files_sha256": R.e_files_sha(folder),
                                                     "manifest_sha256": R.sha_file(folder / "manifest.json")}, indent=1) + "\n")
    log(f"{n} of {len(rows)} kept documents (seed {seed}) -> {folder / f'{name}.txt'}")
    return 0


def cmd_hand_read(n: int, seed: int, violations: int, note: str = "", log=print) -> int:
    """Record the read by hand of the sample --sample N wrote under HAND_READ_SEED (no other seed is recorded): its
    ids, the documents it was drawn from (the two kept files' sha256), and how many of its documents break the
    rubric. run_u1.py --sets needs this record, of the documents on disk, with at least HAND_READ_N documents and
    fewer than HAND_READ_VIOLATIONS violations. A manifest written again over the same documents leaves the read
    good. An earlier record is kept inside the new, and an earlier read of the same documents that found
    HAND_READ_VIOLATIONS or more stays binding: the documents must change before a new read counts."""
    if seed != U.HAND_READ_SEED:  # no drawing again until a sample reads well
        raise SystemExit(f"the hand read that counts is of the sample drawn under seed {U.HAND_READ_SEED} (--sample {n} "
                         f"--seed {U.HAND_READ_SEED}). A sample under another seed is an extra look and is not recorded")
    folder = docs_dir(False)
    sfile, man = folder / f"sample_s{seed}_n{n}.json", folder / "manifest.json"
    if not sfile.is_file():
        raise SystemExit(f"no {sfile}: write the sample first (--sample {n} --seed {seed}), and read it")
    sample, man_sha, files = json.loads(sfile.read_text()), R.sha_file(man), R.e_files_sha(folder)
    if man_sha is None or None in files.values() or sample.get("files_sha256") != files:
        raise SystemExit(f"{sfile.name} was drawn from other documents than the ones on disk: sample and read again")
    if (sample.get("n"), sample.get("seed"), len(sample.get("ids") or [])) != (n, seed, n):
        raise SystemExit(f"{sfile.name} does not hold a sample of {n} documents under seed {seed}")
    if not 0 <= violations <= n:
        raise SystemExit(f"--violations is a count between 0 and {n}")
    hfile = folder / "hand_read.json"
    earlier = []
    if hfile.is_file():  # never lost: an earlier record moves inside the new one
        old = json.loads(hfile.read_text())
        earlier = [*old.pop("earlier", []), old]
    rec = {"utc": R.utc(), "seed": seed, "n": n, "ids": sample["ids"], "sample": sfile.name, "files_sha256": files,
           "manifest_sha256": man_sha, "violations": violations, "note": note,
           "needs": {"n_at_least": U.HAND_READ_N, "violations_under": U.HAND_READ_VIOLATIONS}, "earlier": earlier}
    failed = R.hand_read_failures(rec)  # this read, and every earlier read of these same documents
    ok = n >= U.HAND_READ_N and not failed
    hfile.write_text(json.dumps({**rec, "ok": ok}, indent=1) + "\n")
    log(f"hand read recorded: {n} documents (seed {seed}), {violations} violations -> {hfile}")
    if not ok:
        log(f"NOT ENOUGH for the sets: they need at least {U.HAND_READ_N} documents read and fewer than "
            f"{U.HAND_READ_VIOLATIONS} violations. With {U.HAND_READ_VIOLATIONS} or more the judge prompt is tightened "
            "and every document screened again."
            + (f" A read of these same documents found {max(failed)}: that read is binding, and the documents must "
               "change before a new read counts." if failed and violations < U.HAND_READ_VIOLATIONS else ""))
    return 0 if ok else 1


def cmd_echo(log=print) -> int:
    """A free table of the echo rule (u1_core.echoes_brief): the share of documents that hold a run of 8, 10, 12 and
    16 words of their own brief. For R and N: the first rows of the pilot's file as they stand, and the set the arm
    trains on (the first rows that do not echo). For E: the kept files, when they exist. Nothing is called, and no
    document text is printed."""
    def line(what: str, rows: list[dict]) -> dict:
        sh = U.echo_shares(rows)
        log(f"  {what:<44} {len(rows):>5} documents: "
            + "  ".join(f"run of {k}: {'-' if v is None else format(100 * v, '.2f') + '%'}" for k, v in sh.items()))
        return sh

    table: dict = {}
    for arm in ("R", "N"):
        src = U.DOC_SOURCES[arm]
        f = R.PILOT_DOCS / src["train"]
        if not f.is_file():
            log(f"arm {arm}: {f} is missing (a local-only file)")
            continue
        rows = R.read_jsonl(f)
        log(f"arm {arm} ({src['train']}, {len(rows)} rows):")
        table[arm] = {"first": line(f"the first {U.N_DOCS} rows as they stand", rows[:U.N_DOCS])}
        try:
            clean, stats = U.clean_docs(rows, src["kind"], U.N_DOCS)
        except ValueError as e:
            log(f"  no clean set: {e}")
            continue
        table[arm]["clean"] = line("the set the arm trains on", clean)
        fp = U.docs_fingerprint(clean)
        log(f"  the clean set ends at file row {stats['read']}; {stats['echo']} rows before it echo their brief and are "
            f"passed over; ids fingerprint {fp[:16]} ({'as pinned' if fp == src['fingerprint'] else 'NOT the pinned one'})")
    folder = docs_dir(False)
    for split in ("train", "held"):
        f = folder / U.E_FILES[split]
        if f.is_file():
            log(f"arm E ({f.name}):")
            table.setdefault("E", {})[split] = line("the kept documents", R.read_jsonl(f))
    if "E" not in table:
        log(f"arm E: no kept files yet under {folder}")
    return 0


def cmd_dry(log=print) -> int:
    cfg = config()
    g = cfg["generator"]
    log(f"generator {g['model']} at temperature {g['temperature']}, max_tokens {g['max_tokens']}, provider pin "
        f"{g['provider']['order']} (no fallbacks); judge {U.JUDGE}")
    total_gen = total_judge = 0.0
    for smoke in (True, False):
        for p in stream_plan(smoke):
            first = p["most"] if smoke else min(p["most"], math.ceil(p["quota"] / YIELD_GUESS))
            gen, judge = first * gen_bound(cfg), first * judge_bound()
            if not smoke:
                total_gen, total_judge = total_gen + gen, total_judge + judge
            log(f"{'smoke' if smoke else 'build'} stream {p['name']} (draws of {p['drawn']}): {p['quota']} to keep, "
                f"{first} briefs in the first round, at most {p['most']}; upper cost generation ${gen:.2f}, judge "
                f"${judge:.2f}")
    log(f"build, first round: generation at most ${min(total_gen, CAPS['generation']):.2f} (cap ${CAPS['generation']:.2f}), "
        f"judge at most ${min(total_judge, CAPS['judge']):.2f} (cap ${CAPS['judge']:.2f}); every smoke together at most "
        f"${CAPS['smoke']:.2f}")
    try:
        for smoke in (True, False):
            folder = docs_dir(smoke)
            log(f"{folder}: generation ${cache_cost(folder):.4f} and judge ${judge_cost(folder):.4f} spent so far")
    except Exception as e:  # noqa: BLE001  (a dry run still prints without a lab.toml)
        log(f"results folder not read ({type(e).__name__})")
    log("Dry run: nothing called, nothing spent.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="counts and an upper cost; nothing called")
    ap.add_argument("--smoke", action="store_true", help="45 documents through the filters and the judge (cents)")
    ap.add_argument("--build", action="store_true", help="the held-out and train documents (about $14)")
    ap.add_argument("--sample", type=int, metavar="N", help="write N kept documents to a text file for the hand read")
    ap.add_argument("--hand-read", type=int, metavar="N", help="record the hand read of the sample of N (--seed S)")
    ap.add_argument("--violations", type=int, metavar="K", help="--hand-read: how many of the documents break the rubric")
    ap.add_argument("--note", default="", help="--hand-read: a line to keep with the record")
    ap.add_argument("--echo", action="store_true", help="the echo table (free; nothing called)")
    ap.add_argument("--seed", type=int, default=0, help="--sample and --hand-read: the seed")
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True, help="the judge's form")
    ap.add_argument("--workers", type=int, default=0, help="concurrent generator calls")
    a = ap.parse_args(argv)
    if sum(bool(x) for x in (a.dry, a.smoke, a.build, a.echo, a.sample is not None, a.hand_read is not None)) != 1:
        ap.error("one of --dry, --smoke, --build, --echo, --sample N, --hand-read N")
    if (a.hand_read is not None) != (a.violations is not None):
        ap.error("--hand-read N goes with --violations K, and --violations with --hand-read")
    workers = {**WORKERS, **({"generate": a.workers} if a.workers else {})}
    if a.dry:
        return cmd_dry()
    if a.echo:
        return cmd_echo()
    if a.sample is not None:
        return cmd_sample(a.sample, a.seed)
    if a.hand_read is not None:
        return cmd_hand_read(a.hand_read, a.seed, a.violations, a.note)
    if a.smoke:
        return cmd_smoke(a.strict, workers=workers)
    return cmd_build(a.strict, workers=workers)


if __name__ == "__main__":
    raise SystemExit(main())
