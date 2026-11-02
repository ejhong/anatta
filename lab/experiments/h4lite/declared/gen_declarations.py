"""The declared-release pilot's documents and judges (lab/plans/declared-release-pilot.md, registered in
lab/ledger/NOTES.md, "Declared-release pilot, registered: can a release taught only by documents become release
behavior?", Opus 5.5, 20:21 UTC on 31 Oct 2026). Recognition track; private. Claude (Opus 5.5) wrote it for Eugene Jhong.

Laptop only, no GPU. Every model call goes through OpenRouter with OPENROUTER_API_KEY, read from the environment and
never printed, logged or written (error texts are redacted). The generator, Qwen3-235B-A22B-Instruct-2507 (Apache-2.0,
non-thinking), runs at temperature 1.0 pinned to one provider (config.json: provider order set, allow_fallbacks false,
data_collection "deny"); each response's provider and model are recorded per document, and a response from any other
provider or model stops the run. No Claude-written text is trained on: Claude wrote the briefs (declared_core.py),
never an exemplar. Claude (anthropic/claude-sonnet-4.6) is the judge. Since amendment 13 its labels select every
training declaration (the screen, below), where the design's lines 128-129 said Claude "only judges samples"; still no
Claude-written text is trained on, since the judge's replies never enter a training set.

Subcommands (python gen_declarations.py CMD; every one resumes from its cache):
  build        the documents: per kind a held stream (200 kept) and a train stream (10,000 kept), seeded briefs drawn
               in order (declared_core.draw_briefs), generated with 10% slack into results/declared/docs/cache/BRIEFS/
               (cache_dir: BRIEFS is the briefs' sha, 16 hex digits, so a brief change starts a fresh cache),
               filtered (declared_core.doc_flags), screened (amendment 13: every declaration that passes the filters
               put to the registered judge, in selection order, and kept only on a clean verdict; class Screen),
               deduplicated (word 5-gram Jaccard above 0.5) and selected greedily in a fixed order (held declared, held
               neutral, train declared, train neutral); a short stream is topped up by continuing its draw (generated
               by index, filtered, screened, selected). Writes declared.jsonl, neutral.jsonl, held_declared.jsonl,
               held_neutral.jsonl, filter_report.json, screen_report.json (descriptive) and manifest.json (the
               files' shas, the code's and the config's, the briefs' sha, providers and models, the wording cap at
               every dose, the referent mix, the spend, and the screen: screen.jsonl's sha and per stream what was
               screened, labelled, dropped and kept). A rerun whose code, briefs, screen and selection reproduce
               manifest.json rewrites nothing, so the manifest's sha (which judge.json and every set record) stays.
               Before its first write over a build from before the screen, it copies that build's manifest.json,
               filter_report.json and judge.json to *_first.json (once, never over a copy), and seeds screen.jsonl
               with the first gate's 100 declared verdicts (config.json's screen.gate1). Exit 1 on the failure stop,
               the drop-rate stop or a spend cap, before anything is selected or written. --screen-neutral (off;
               amendment 13, item 6: only if the gate's neutral half stops) screens the neutral streams too; once
               screen.jsonl holds a neutral record it stays on.
  judge        the judge's sample: 100 declared and 50 neutral training documents chosen by seed; judge.json. The
               declared hundred take their verdicts from screen.jsonl (by id and text sha, "source": "screen"), not
               asked; the neutral fifty are asked ("source": "asked"), or read from the screen when the neutral
               screen ran. Exit 0 when the gate passes, 1 when the run must stop for a fix (an unscreened declaration
               stops it). After a passing gate, the stability reading (reported only): the gate's 100 declared asked
               afresh, judge_stability.json. With --ext: 100 of D30k's extra declarations (declared_ext.jsonl), chosen
               by the same seed, read from the screen under the same gate; judge_ext.json, after extend and before
               modal_declared.py sets --ext.
  extend       D30k's 20,000 more declarations (by hand, only after a separate registered decision to run D30k:
               it is skipped by decision, Eugene, 1 Nov 2026, so run_pilot.sh --go never runs extend): the train
               declared stream continued after its 10,000th kept document, screened like the build's (records with
               stream "declared_ext", their own spend cap), deduplicated against every kept document;
               declared_ext.jsonl, manifest_ext.json and screen_report_ext.json (manifest.json and the main files are
               left as they are). Each round selects from the cache before generating, so a rerun after a finished
               extension generates, asks and rewrites nothing; its spend cap counts from the main build's spend, so it
               holds across reruns.
  probe-judge  the belief probe's answers (results/declared/belief/LABEL/probe.jsonl) labelled by the judge:
               L1 and L2 on rule and scenario answers, unprompted release on controls, the names on identity answers;
               probe_judge.jsonl beside each probe.jsonl. Exit 1 while any verdict is unparsed; a rerun asks only
               for those.
  smoke        20 documents of each kind (15 to train on, 5 held out) from the smoke's own streams, generated at their
               caps at once (45 a train stream, 15 a held one), the same filters and selection, under
               results/declared_smoke/docs/; checks the provider pin, the seeded draw and the yield gate (each train
               stream's passed_filters / generated at least 0.60, no single flag on more than 0.25 of it), whose
               numbers go into filter_report.json and are printed. Its cache is keyed by the briefs too, so a re-smoke
               after the brief is sharpened generates all 120 under the new brief, leaving the old folder as it is.
  plumbing     one declaration and one neutral document through the pinned client, and one judge call, under
               results/declared_plumbing/ (never trained on): the recorded provider and model, the filters' verdicts,
               the spend. The 31 Oct check's documents are in the legacy flat layout there, kept and not read.
  brief        prints one rendered brief (--stream, --index), for reading.
  status       what is cached (per cache folder, the current briefs' marked, legacy flat files named), screened,
               selected and judged.

The screen (amendment 13; lab/plans/declared-release-pilot-screen.md, its "After the design check" governing):
config.json's judge, unchanged, with declared_core.judge_doc_messages and parse_doc_verdict, unchanged. Keep rule:
declared_core.screen_keep. docs/screen.jsonl holds one record per document asked (id, text_sha256, kind, stream;
status ok, unparsed or failed; verdict; raw, its first 300 characters; provider, model, cost, utc; source "screen" or
"gate1"; final), keyed by (id, text_sha256); a rerun asks only for documents with no final record. A call is retried up
to three times with backoff. The failure stop: 20 calls in a row without a verdict (failed or unparsed), checked after
every call, or failed plus unparsed above 1% of the run's calls once 100 are made, checked when each pass of asks ends;
a stopped run leaves its failed and unparsed records non-final, so the rerun asks them again, and a pass that completes
makes them final, so their documents are dropped. The drop-rate stop: a declared stream dropping more than 5% of the
screened candidates its selection walked (labelled, unparsed and failed together). Spend caps:
generation.spend_cap_usd.screen (the build's declared streams), screen_ext (D30k's) and screen_neutral, each counted
from screen.jsonl's costs; workers: config.json's screen.workers (8), --screen-workers to override.

The judges (judge, judge --ext, the stability reading, probe-judge) share one spend cap, config.json's
generation.spend_cap_usd.judge: each run adds what judge.json, judge_ext.json, judge_stability.json and every
probe_judge.jsonl already record (never screen.jsonl), asks nothing more once the cap is reached, writes what it judged
and stops (SystemExit; the stability reading only records what it did not ask). Requests carry no app title or referer
header.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
H4 = HERE.parent
LAB = HERE.parents[3]
sys.path.insert(0, str(HERE))
import declared_core as core  # noqa: E402

KEY_ENV = "OPENROUTER_API_KEY"  # the name only; the value is read from the environment and never printed
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
CONFIG = HERE / "config.json"
MAIN_PLAN = (("declared", "held", core.HELD_DOCS), ("neutral", "held", core.HELD_DOCS),
             ("declared", "train", core.TRAIN_DOCS), ("neutral", "train", core.TRAIN_DOCS))
EXT_STREAM = "declared_ext"
SCREEN_BACKOFF = (2.0, 4.0, 8.0)  # seconds before the screen's three retries of a failed call
SCREEN_GROUPS = {"declared_held": "screen", "declared_train": "screen", EXT_STREAM: "screen_ext",
                 "neutral_held": "screen_neutral", "neutral_train": "screen_neutral"}  # stream -> its spend cap
FIRST_COPIES = (("manifest.json", "manifest_first.json"), ("filter_report.json", "filter_report_first.json"),
                ("judge.json", "judge_first.json"))
STABILITY_FILE = "judge_stability.json"


def results_dir(cfg: dict, which: str = "results") -> Path:
    return LAB / cfg["paths"][which]


def utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


# ------------------------------------------------------------------------------------------------ the client


class ApiError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get(KEY_ENV, "")
    if not key:
        raise SystemExit(f"{KEY_ENV} is not set in the environment; generation and judging go through OpenRouter")
    return key


def redact(text: str, key: str) -> str:
    return text.replace(key, "[redacted]") if key else text


def parse_completion(data) -> dict:
    """The text, provider, model, finish reason, id and usage of an OpenRouter chat completion; an error body raises."""
    if not isinstance(data, dict) or "error" in data:
        msg = data.get("error") if isinstance(data, dict) else data
        raise ApiError(f"OpenRouter error: {str(msg)[:300]}")
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ApiError(f"OpenRouter reply without choices: {str(data)[:300]}") from None
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    u = data.get("usage") or {}
    return {"text": content or "", "provider": data.get("provider"), "model": data.get("model"), "id": data.get("id"),
            "finish_reason": choice.get("finish_reason"), "native_finish_reason": choice.get("native_finish_reason"),
            "usage": {"prompt_tokens": int(u.get("prompt_tokens") or 0),
                      "completion_tokens": int(u.get("completion_tokens") or 0), "cost": float(u.get("cost") or 0.0)}}


def chat(model: str, messages: list[dict], temperature: float, max_tokens: int, provider: dict | None = None,
         seed: int | None = None, retries: int = 6, timeout: float = 300.0, opener=None) -> dict:
    """One chat completion through OpenRouter. `provider` is OpenRouter's routing object (the generator's pin: order,
    allow_fallbacks false, data_collection deny). 408/409/429/5xx, network errors and error bodies are retried with
    backoff. The key goes in the Authorization header only; no message, exception or log carries it."""
    key = api_key()
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "usage": {"include": True}}
    if provider is not None:
        body["provider"] = provider
    if seed is not None:
        body["seed"] = seed
    data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}  # no app title or referer
    opener = opener or urllib.request.urlopen
    delay, last = 2.0, "no attempt"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OPENROUTER, data=data, headers=headers, method="POST")
            with opener(req, timeout=timeout) as resp:
                return parse_completion(json.loads(resp.read().decode()))
        except urllib.error.HTTPError as e:
            detail = redact(e.read().decode(errors="replace")[:300], key)
            last = f"HTTP {e.code}: {detail}"
            if not (e.code in (408, 409, 429) or e.code >= 500):
                raise ApiError(f"OpenRouter {e.code} for {model}: {detail}") from None
        except ApiError as e:
            last = redact(str(e), key)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last = redact(f"{type(e).__name__}: {str(e)[:200]}", key)
        if attempt < retries - 1:
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise ApiError(f"OpenRouter gave up on {model} after {retries} attempts ({last})")


def usage_totals(usages: list[dict]) -> dict:
    return {"calls": len(usages), "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
            "cost_usd": round(sum(u.get("cost", 0.0) for u in usages), 4)}


# ------------------------------------------------------------------------------------------------- the cache


def cache_dir(cfg: dict, folder: Path) -> Path:
    """Where a documents folder's generations are cached: folder/cache/ and the briefs' sha (declared_core.briefs_sha,
    its first 16 hex digits). Rows are keyed by stream and index, so a cache shared across a brief change would hand
    the old brief's documents back for the new brief's indices; keyed by the briefs, a sharpened brief starts a fresh
    folder and the old one is left as it is (Fable's re-check, (c)1). Frozen briefs give one folder, which build and
    extend share. The sha covers each stream up to its generation cap in config.json, so changing a cap starts a fresh
    folder too (README, open points). Files directly in folder/cache/ are the legacy flat layout (legacy_cache)."""
    return Path(folder) / "cache" / core.briefs_sha(cfg)[:16]


def legacy_cache(folder: Path) -> list[Path]:
    """Cache files in the legacy flat layout, directly in folder/cache/ (results/declared_plumbing/cache/ holds the 31
    Oct plumbing check's two): kept as they are, never read by build, extend, the smoke or the plumbing check."""
    root = Path(folder) / "cache"
    return sorted(p for p in root.glob("*.jsonl") if p.is_file()) if root.exists() else []


class Cache:
    """Generated documents per stream, one JSON line each, appended as calls finish (cache_dir(cfg, docs folder)/
    STREAM.jsonl). A failed call writes nothing, so a rerun retries it; the first line for an index is the one kept."""

    def __init__(self, folder: Path):
        self.folder, self.lock, self.rows = Path(folder), threading.Lock(), {}

    def stream(self, name: str) -> dict:
        if name not in self.rows:
            rows, path = {}, self.folder / f"{name}.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"skipped a damaged line in {path}", flush=True)
                        continue
                    rows.setdefault(r["index"], r)
            self.rows[name] = rows
        return self.rows[name]

    def add(self, row: dict) -> None:
        with self.lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            with open(self.folder / f"{row['stream']}.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            self.stream(row["stream"]).setdefault(row["index"], row)

    def contiguous(self, name: str) -> list[dict]:
        """The stream's rows from index 0 up to its first gap: a later fill of a gap only appends."""
        rows, out = self.stream(name), []
        while len(out) in rows:
            out.append(rows[len(out)])
        return out

    def cost(self, names) -> float:
        return sum(r["usage"]["cost"] for n in names for r in self.stream(n).values())


def pin_ok(cfg: dict, provider: str | None, model: str | None) -> bool:
    g = cfg["generator"]
    return provider == g["expected_provider"] and model in g["expected_models"]


def generate(cfg: dict, cache: Cache, stream: str, kind: str, indices, budget_usd: float, spent: float,
             workers: int, chat_fn=None, log=print) -> dict:
    """Generate the briefs at `indices` of a stream that the cache lacks, `workers` calls at a time, until done or the
    spend (the cache's so far plus this call's) reaches `budget_usd`. A response from another provider or model than
    the pin stops everything (SystemExit) after it is recorded."""
    chat_fn = chat_fn or chat
    g, seed = cfg["generator"], cfg["seeds"]["briefs"]
    rows = cache.stream(stream)
    todo = [i for i in indices if i not in rows]
    if not todo:
        return {"generated": 0, "cost": 0.0, "errors": 0, "stop": None}
    briefs = core.draw_briefs(seed, stream, kind, max(todo) + 1)
    state = {"spent": spent, "cost": 0.0, "ok": 0, "errors": [], "stop": None}
    it = iter(todo)

    def one(i):
        b = briefs[i]
        return i, b, chat_fn(g["model"], core.render_brief(b), g["temperature"], g["max_tokens"],
                             provider=g["provider"], seed=core.gen_seed(seed, stream, i))

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        pending = set()

        def submit_next() -> bool:
            if state["stop"]:
                return False
            if state["spent"] >= budget_usd:
                state["stop"] = f"spend cap ${budget_usd:.2f} reached (${state['spent']:.2f})"
                return False
            i = next(it, None)
            if i is None:
                return False
            pending.add(ex.submit(one, i))
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.discard(fut)
                try:
                    i, b, res = fut.result()
                except ApiError as e:
                    state["errors"].append(str(e)[:300])
                    if len(state["errors"]) > max(20, len(todo) // 20):
                        state["stop"] = f"{len(state['errors'])} failed calls (last: {state['errors'][-1]})"
                    submit_next()
                    continue
                row = {"stream": stream, "index": i, "kind": kind, "brief": b, "text": res["text"].strip(),
                       "provider": res["provider"], "model": res["model"], "finish_reason": res["finish_reason"],
                       "native_finish_reason": res.get("native_finish_reason"), "gen_id": res["id"],
                       "usage": res["usage"], "utc": utc()}
                cache.add(row)
                state["spent"] += res["usage"]["cost"]
                state["cost"] += res["usage"]["cost"]
                state["ok"] += 1
                if not pin_ok(cfg, res["provider"], res["model"]):
                    state["stop"] = (f"PROVIDER PIN BROKEN: {stream} index {i} came from provider {res['provider']!r}, "
                                     f"model {res['model']!r} (pinned: {g['expected_provider']!r}, {g['expected_models']})")
                if state["ok"] % 250 == 0:
                    log(f"{stream}: {state['ok']} of {len(todo)} generated, ${state['cost']:.2f} this call, "
                        f"{state['ok'] / max(time.time() - t0, 1e-9) * 3600:.0f} an hour", flush=True)
                submit_next()
    if state["errors"]:
        errs = cache.folder / "errors.jsonl"
        with open(errs, "a") as f:
            for e in state["errors"]:
                f.write(json.dumps({"stream": stream, "utc": utc(), "error": e}) + "\n")
    if state["stop"] and state["stop"].startswith("PROVIDER PIN BROKEN"):
        raise SystemExit(state["stop"])
    return {"generated": state["ok"], "cost": round(state["cost"], 4), "errors": len(state["errors"]),
            "stop": state["stop"]}


# ------------------------------------------------------------------------------------------------ the screen


class Screen:
    """The screen (amendment 13; module docstring) over one documents folder: the registered judge asked of candidates
    before selection, one record per document asked, appended to docs/screen.jsonl as each call returns. A record with
    a verdict (status ok) is final at once. One without (failed or unparsed) is written non-final and made final when
    its pass of asks completes, so its document is dropped; a pass that stops (the failure stop, a spend cap) leaves it
    non-final, so the rerun asks it again. The run's calls, those without a verdict and the current run of them are
    counted across its passes, for the failure stop."""

    def __init__(self, cfg: dict, folder: Path, chat_fn=None, workers: int | None = None, log=print,
                 neutral: bool = False):
        self.cfg, self.log = cfg, log
        self.path = Path(folder) / core.SCREEN_FILE
        self.chat_fn = chat_fn or chat
        self.workers = workers or cfg["screen"]["workers"]
        self.rows = core.read_screen(self.path)
        if self.path.exists() and self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n"):
            with open(self.path, "a") as f:  # a line cut short by a crash stays damaged (skipped), not joined to the next
                f.write("\n")
        self.final = core.screen_final(self.rows)
        self.pending: dict = {}  # this pass's records without a verdict: (id, sha) -> record, non-final until it completes
        self.pending_lines: set = set()
        self.start = 0  # screen.jsonl's length when this pass began: what the pass wrote lies after it
        self.calls = self.bad = self.streak = 0
        self.spend = {g: 0.0 for g in set(SCREEN_GROUPS.values())}
        for r in self.rows:
            if r.get("source") == "screen" and r.get("stream") in SCREEN_GROUPS:
                self.spend[SCREEN_GROUPS[r["stream"]]] += float(r.get("cost") or 0.0)
        # the neutral screen, once begun, stays on (a rerun without --screen-neutral must not undo it)
        self.neutral = bool(neutral) or any(r.get("kind") == "neutral" for r in self.rows)

    def key(self, c: dict) -> tuple:
        return c["id"], c["sha"]

    def has(self, c: dict) -> bool:
        return self.key(c) in self.final

    def keep(self, c: dict, kind: str) -> bool:
        return core.screen_keep(kind, self.final.get(self.key(c)))

    def prefix(self, cands: list[dict]) -> int:
        """How many of a stream's candidates, from its first, have a final record: the part a selection may read."""
        n = 0
        for c in cands:
            if self.key(c) not in self.final:
                break
            n += 1
        return n

    def append(self, rec: dict) -> None:
        line = json.dumps(rec) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(line)
        self.rows.append(rec)
        if rec["final"]:
            self.final.setdefault((rec["id"], rec["text_sha256"]), rec)
        else:
            self.pending[(rec["id"], rec["text_sha256"])] = rec
            self.pending_lines.add(line)
        if rec["source"] == "screen" and rec["stream"] in SCREEN_GROUPS:
            self.spend[SCREEN_GROUPS[rec["stream"]]] += float(rec["cost"] or 0.0)

    def one(self, item: tuple) -> dict:
        """One document's record: the judge asked, retried up to three times with backoff when the call fails."""
        c, kind, stream = item
        j = self.cfg["judge"]
        base = {"id": c["id"], "text_sha256": c["sha"], "kind": kind, "stream": stream}
        err = "no attempt"
        for attempt in range(1 + core.SCREEN_RETRIES):
            try:
                res = self.chat_fn(j["model"], core.judge_doc_messages(c["row"]["text"]), j["temperature"],
                                   j["max_tokens"], retries=1)
            except ApiError as e:
                err = str(e)
                if attempt < core.SCREEN_RETRIES:
                    time.sleep(SCREEN_BACKOFF[attempt])
                continue
            v = core.parse_doc_verdict(res["text"])
            return {**base, "status": "ok" if v is not None else "unparsed", "verdict": v, "raw": res["text"][:300],
                    "provider": res.get("provider"), "model": res.get("model"),
                    "cost": float(res["usage"]["cost"]), "utc": utc(), "source": "screen", "final": v is not None}
        return {**base, "status": "failed", "verdict": None, "raw": f"judge call failed: {err}"[:300], "provider": None,
                "model": None, "cost": 0.0, "utc": utc(), "source": "screen", "final": False}

    def ask(self, todo: list[tuple]) -> str | None:
        """Asks the judge about each (candidate, kind, stream) in order, `workers` calls at a time. Returns the stop's
        message (the failure stop, a spend cap) when the pass stops, else None once every document has a record and
        this pass's records without a verdict are made final."""
        if not todo:
            return None
        caps = self.cfg["generation"]["spend_cap_usd"]
        self.start = self.path.stat().st_size if self.path.exists() else 0
        state = {"stop": None, "done": 0}
        it = iter(todo)
        t0 = time.time()
        with ThreadPoolExecutor(self.workers) as ex:
            pending = set()

            def submit_next() -> bool:
                if state["stop"]:
                    return False
                item = next(it, None)
                if item is None:
                    return False
                group = SCREEN_GROUPS[item[2]]
                if self.spend[group] >= caps[group]:
                    state["stop"] = (f"SPEND CAP: the screen's cap generation.spend_cap_usd.{group} (${caps[group]:.2f}) "
                                     f"is reached (${self.spend[group]:.2f} recorded in screen.jsonl)")
                    return False
                pending.add(ex.submit(self.one, item))
                return True

            for _ in range(self.workers):
                if not submit_next():
                    break
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.discard(fut)
                    rec = fut.result()
                    self.append(rec)
                    self.calls += 1
                    state["done"] += 1
                    if rec["status"] == "ok":
                        self.streak = 0
                    else:
                        self.bad += 1
                        self.streak += 1
                        if self.streak >= core.SCREEN_FAIL_STREAK and not state["stop"]:
                            state["stop"] = (f"FAILURE STOP (amendment 13): {self.streak} calls in a row without a "
                                             f"verdict (last: {rec['raw'][:160]})")
                    if state["done"] % 500 == 0:
                        self.log(f"screen: {state['done']} of {len(todo)} asked, {self.bad} without a verdict this run, "
                                 f"{state['done'] / max(time.time() - t0, 1e-9) * 60:.0f} a minute", flush=True)
                    submit_next()
        if not state["stop"] and self.calls >= core.SCREEN_FAIL_MIN_CALLS \
                and Fraction(self.bad, self.calls) > core.SCREEN_FAIL_RATE:
            state["stop"] = (f"FAILURE STOP (amendment 13): {self.bad} of this run's {self.calls} calls came back without a "
                             f"verdict (failed or unparsed), above 1%")
        if state["stop"]:
            self.pending, self.pending_lines = {}, set()  # left non-final in screen.jsonl: the rerun asks them again
            return state["stop"]
        self.finalize()
        return None

    def finalize(self) -> None:
        """Makes this pass's records without a verdict final, so their documents are dropped: screen.jsonl is rewritten
        from where the pass began (the bytes before it, which a manifest may have sealed, are copied as they are),
        through a temporary file."""
        if not self.pending:
            return
        raw = self.path.read_bytes()
        head, out = raw[:self.start], []
        for line in raw[self.start:].decode().splitlines(keepends=True):
            if line in self.pending_lines:
                line = json.dumps({**json.loads(line), "final": True}) + "\n"
            out.append(line)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(head + "".join(out).encode())
        os.replace(tmp, self.path)
        for key, rec in self.pending.items():
            rec["final"] = True
            self.final.setdefault(key, rec)
        self.pending, self.pending_lines = {}, set()

    def gate1(self) -> dict:
        g = [r for r in self.rows if r.get("source") == "gate1"]
        return {"seeded": len(g), "dropped": [r["id"] for r in g if not core.screen_keep(r["kind"], r)]}


def keep_first(folder: Path) -> list[str]:
    """Before the rebuild overwrites them, the first build's manifest.json, filter_report.json and judge.json are
    copied to manifest_first.json, filter_report_first.json and judge_first.json beside them (amendment 13): once,
    never over an existing copy, and only from a build from before the screen (its manifest has no screen record;
    judge.json only when it judged that manifest). Returns the copies made."""
    man = Path(folder) / "manifest.json"
    if not man.exists():
        return []
    m = json.loads(man.read_text())
    if "screen" in m or m.get("smoke"):
        return []
    made = []
    for src, dst in FIRST_COPIES:
        s, d = Path(folder) / src, Path(folder) / dst
        if not s.exists() or d.exists():
            continue
        if src == "judge.json" and json.loads(s.read_text()).get("manifest_sha256") != core.sha_file(man):
            continue
        shutil.copyfile(s, d)
        made.append(dst)
    return made


def seed_gate1(cfg: dict, folder: Path, cache: Cache, screen: Screen, log=print) -> str | None:
    """Seeds screen.jsonl with the first gate's declared verdicts (config.json's screen.gate1: judge_first.json, the
    judge.json committed at 8078159, checked by its sha256), before any screen call: source "gate1", final, keyed by the
    text sha of the cached document each one judged (checked against the first build's declared.jsonl while it is on
    disk). Same judge, prompt and text, so their labels drop documents without a new call. Once: a screen.jsonl that
    holds gate1 records is left as it is. Returns a stop's message, else None."""
    spec = (cfg.get("screen") or {}).get("gate1")
    if not spec or any(r.get("source") == "gate1" for r in screen.rows):
        return None
    path = Path(folder) / spec["file"]
    if core.sha_file(path) != spec["sha256"]:
        return (f"the first gate's verdicts can't be seeded: {path} is missing or is not the judge.json committed at "
                f"{spec.get('commit')} (sha256 {spec['sha256'][:16]}...; README, open point 43, says how to restore it)")
    j = json.loads(path.read_text())
    first = Path(folder) / "manifest_first.json"
    if not first.exists() or j.get("manifest_sha256") != core.sha_file(first):
        return f"the first gate's verdicts can't be seeded: {path.name} judged another manifest than manifest_first.json"
    mf = json.loads(first.read_text())
    if mf.get("cache_dir") != cache.folder.relative_to(folder).as_posix():
        return "the first gate's verdicts can't be seeded: the first build's documents came from another cache folder"
    dfile = Path(folder) / core.DOC_FILES[("declared", "train")]
    judged = {}
    if core.sha_file(dfile) == ((mf.get("files") or {}).get(dfile.name) or {}).get("sha256"):
        judged = {r["id"]: r["text"] for r in read_jsonl(dfile)}
    recs = []
    for v in j.get("verdicts", []):
        if v.get("kind") != "declared":
            continue
        stream, idx = v["id"].rsplit(":", 1)
        row = cache.stream(stream).get(int(idx))
        if row is None:
            return f"the first gate's verdicts can't be seeded: {v['id']} is not in the cache"
        if judged and judged.get(v["id"]) != row["text"]:
            return f"the first gate's verdicts can't be seeded: {v['id']}'s cached text is not the text it judged"
        ok = core.screen_verdict_ok(v.get("verdict"))
        recs.append({"id": v["id"], "text_sha256": core.text_sha(row["text"]), "kind": "declared", "stream": stream,
                     "status": "ok" if ok else "unparsed", "verdict": v["verdict"] if ok else None,
                     "raw": (v.get("raw") or "")[:300], "provider": None, "model": j.get("judge"),
                     "cost": float(v.get("cost") or 0.0), "utc": j.get("judged_utc"), "source": "gate1", "final": True})
    for rec in recs:
        screen.append(rec)
    info = screen.gate1()
    log(f"screen: the first gate's {info['seeded']} declared verdicts seeded into screen.jsonl (source gate1, final; "
        f"{path.name}, sha256 {spec['sha256'][:16]}...); dropped by their labels, without a new call: "
        f"{info['dropped'] or 'none'}", flush=True)
    return None


# --------------------------------------------------------------------------------------------- the selection


def load_pairs8(cfg: dict) -> frozenset:
    path = LAB / cfg["inputs"]["pairs"]["file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg["inputs"]["pairs"]["sha256"]:
        raise SystemExit(f"{path} is not the registered gate0/pairs.jsonl (sha256 differs)")
    return core.pairs_8grams([json.loads(x) for x in raw.decode().splitlines() if x.strip()])


def candidates(cfg: dict, cache: Cache, stream: str, pairs8: frozenset) -> list[dict]:
    """A stream's generated documents in order (up to the first gap), each with its filter flags, its text's sha256
    (the screen's key beside the id) and, when it passes, its 5-gram ids for dedup."""
    out = []
    for r in cache.contiguous(stream):
        flags = core.doc_flags(r["text"], r["brief"], pairs8, r.get("finish_reason"), pin_ok(cfg, r["provider"], r["model"]))
        out.append({"id": f"{stream}:{r['index']:06d}", "row": r, "flags": flags, "sha": core.text_sha(r["text"]),
                    "sh": core.shingle_ids(r["text"]) if not flags else None})
    return out


def plan_streams(smoke: bool) -> list[tuple[str, str, str, int]]:
    """(stream, kind, split, quota) in the fixed processing order."""
    quota = (lambda split, q: core.SMOKE_DOCS[split]) if smoke else (lambda split, q: q)
    return [(core.stream_name(k, s, smoke), k, s, quota(s, q)) for k, s, q in MAIN_PLAN]


def screened_kinds(plan: list, screen: Screen | None, ext: bool) -> dict:
    """The streams the screen covers, name -> kind: every declared stream (D30k's continuation with ext), and the
    neutral streams while the neutral screen is on. None without a screen (the smoke)."""
    if screen is None:
        return {}
    out = {name: kind for name, kind, _, _ in plan if kind == "declared" or screen.neutral}
    if ext:
        out[EXT_STREAM] = "declared"
    return out


def run_selection(cfg: dict, cache: Cache, pairs8: frozenset, smoke: bool = False, ext: bool = False,
                  screen: Screen | None = None, ask=(), cands: dict | None = None) -> dict:
    """The greedy selection (declared_core.select_streams, unchanged) over what is cached. With ext, the train declared
    stream is continued after its 10,000th kept document as the stream declared_ext (quota 20,000).

    Without a screen (the smoke), passing means passing the mechanical filters. With one (amendment 13), a screened
    stream's passing candidates are those of its mechanically passing candidates, in order, up to the first without a
    final screen record, that the keep rule keeps. For each stream in `ask` that is short, the next candidates without
    a final record are asked (as many as its shortfall plus 10% and 5) and the selection runs again, until each is full
    or has nothing left in the cache to ask: so a stream is screened in its selection order, and only as far as its
    quota needs. res["stop"] is a stop's message when the screen stopped (nothing may then be selected or written)."""
    plan = plan_streams(smoke)
    cands = cands or {name: candidates(cfg, cache, name, pairs8) for name, _, _, _ in plan}
    mech = {name: [c for c in cs if not c["flags"]] for name, cs in cands.items()}
    screened = screened_kinds(plan, screen, ext)
    dt = core.stream_name("declared", "train")
    stop = None
    while True:
        lists, prefix, passing = dict(mech), {}, {}

        def admit(name: str) -> None:
            if name not in screened:
                passing[name] = lists[name]
                return
            prefix[name] = screen.prefix(lists[name])
            passing[name] = [c for c in lists[name][:prefix[name]] if screen.keep(c, screened[name])]

        for name, _, _, _ in plan:
            admit(name)
        sel_plan = [(name, quota, passing[name]) for name, _, _, quota in plan]
        if ext:
            main = core.select_streams(sel_plan)
            if main[dt]["short"]:
                raise SystemExit(f"{dt} is short of its 10,000 documents: run build first")
            last = main[dt]["kept"][-1]["id"]
            lists[EXT_STREAM] = mech[dt][next(i for i, c in enumerate(mech[dt]) if c["id"] == last) + 1:]
            admit(EXT_STREAM)
            sel_plan.append((EXT_STREAM, core.EXT_DOCS, passing[EXT_STREAM]))
        sel = core.select_streams(sel_plan)
        todo = []
        for name in ask:
            if name in screened and sel[name]["short"]:
                rest = [c for c in lists[name][prefix[name]:] if not screen.has(c)]
                todo += [(c, screened[name], name) for c in rest[:math.ceil(sel[name]["short"] * (1 + core.SLACK)) + 5]]
        if not todo:
            break
        stop = screen.ask(todo)
        if stop:
            break
    return {"plan": plan, "cands": cands, "sel": sel, "lists": lists, "prefix": prefix, "screened": screened,
            "stop": stop}


def walked(res: dict, name: str) -> list[dict]:
    """The screened candidates a stream's selection walked through, in order: up to its last kept candidate when it is
    full, else every one it screened (its prefix). The screen's counts and the drop rate are over these."""
    lst, s = res["lists"][name], res["sel"][name]
    if s["short"] or not s["kept"]:
        return lst[:res["prefix"][name]]
    last = s["kept"][-1]["id"]
    return lst[:next(i for i, c in enumerate(lst) if c["id"] == last) + 1]


def screen_counts(screen: Screen, cs: list[dict], kind: str) -> dict:
    """A stream's screen, over the candidates it walked: screened, labelled by key, unparsed, failed, dropped (any
    record that fails the keep rule), kept by the screen, the drop rate and the registered 5% threshold (declared
    streams; the neutral screen's rate is reported only), and the dropped ids (screen_report.json's)."""
    recs = [screen.final.get(screen.key(c)) for c in cs]
    why = [core.screen_why(kind, r) for r in recs]
    ok = [r for r in recs if r and r.get("status") == "ok" and core.screen_verdict_ok(r.get("verdict"))]
    if kind == "declared":
        labelled = {k: sum(bool(r["verdict"][k]) for r in ok) for k in core.SCREEN_LABELS}
        labelled["states_rule_false"] = sum(not r["verdict"]["states_rule"] for r in ok)
    else:
        labelled = {k: sum(bool(r["verdict"][k]) for r in ok) for k in core.DOC_KEYS}
    n, dropped = len(cs), sum(w != "kept" for w in why)
    return {"kind": kind, "screened": n, "labelled": labelled,
            "unparsed": sum(bool(r) and r.get("status") == "unparsed" for r in recs),
            "failed": sum(bool(r) and r.get("status") == "failed" for r in recs), "dropped": dropped, "kept": n - dropped,
            "drop_rate": round(dropped / n, 6) if n else None,
            "threshold": float(core.SCREEN_DROP_MAX) if kind == "declared" else None,
            "from_gate1": sum(bool(r) and r.get("source") == "gate1" for r in recs),
            "dropped_ids": [c["id"] for c, w in zip(cs, why) if w != "kept"]}


def drop_stop(counts: dict) -> str | None:
    """The registered drop rate's stop (amendment 13, after the design check, must-fix 3): a declared stream that
    dropped more than 5% of its screened candidates (labelled, unparsed and failed together)."""
    over = [(n, c) for n, c in counts.items() if c["kind"] == "declared" and c["screened"]
            and Fraction(c["dropped"], c["screened"]) > core.SCREEN_DROP_MAX]
    if not over:
        return None
    return ("DROP-RATE STOP (amendment 13, the registered 5%): " + "; ".join(
        f"{n} dropped {c['dropped']} of its {c['screened']} screened candidates ({100 * c['dropped'] / c['screened']:.2f}%; "
        f"labelled {c['labelled']}, unparsed {c['unparsed']}, failed {c['failed']})" for n, c in over)
        + ". Above 5%: the build stops for a ledger entry before any training (nothing selected or written but "
          "screen.jsonl and the screen report)")


def screen_line(counts: dict) -> str:
    return "; ".join(f"{n}: {c['screened']} screened, {c['dropped']} dropped ({100 * (c['drop_rate'] or 0):.2f}%; "
                     f"labelled {c['labelled']}, unparsed {c['unparsed']}, failed {c['failed']})"
                     for n, c in counts.items())


def screen_report(res: dict, screen: Screen, counts: dict, manifest: dict | None = None, stopped: str | None = None,
                  manifest_sha: str | None = None) -> dict:
    """The screen's report, descriptive only (not registered): per stream the labels by key, the dropped ids, and what
    was screened and dropped (with the labels) by referent, genre, setting and emphasis (a neutral stream's focus in
    place of emphasis); with the manifest, the referent mix and the wording shares at each dose after the screen, as
    the manifest computes them."""
    out = {"reported_only": "descriptive, not registered (amendment 13)", "utc": utc(), "stopped": stopped,
           "manifest_sha256": manifest_sha, "screen": core.screen_seal(screen.path), "streams": {}}
    for name, c in counts.items():
        cs, kind = walked(res, name), c["kind"]
        by = {}
        for axis in ("referent", "genre", "setting", "emphasis" if kind == "declared" else "focus"):
            tab: dict = {}
            for x in cs:
                w = core.screen_why(kind, screen.final.get(screen.key(x)))
                t = tab.setdefault(str(x["row"]["brief"].get(axis)), {"screened": 0, "dropped": 0, "labels": {}})
                t["screened"] += 1
                if w != "kept":
                    t["dropped"] += 1
                    for lab in w.split(", "):
                        t["labels"][lab] = t["labels"].get(lab, 0) + 1
            by[axis] = dict(sorted(tab.items()))
        out["streams"][name] = {**{k: c[k] for k in ("kind", "screened", "labelled", "unparsed", "failed", "dropped",
                                                    "kept", "drop_rate", "threshold", "from_gate1")},
                                "dropped_ids": c["dropped_ids"], "by": by}
    if manifest is not None:
        out["after_screen"] = {"referents_at_dose": manifest.get("referents_at_dose"),
                               "referents": manifest.get("referents"),
                               "wording_shares": {n: w["shares"] for n, w in (manifest.get("wording") or {}).items()},
                               "wording_held": manifest.get("wording_held")}
    return out


def doc_row(c: dict, rank: int, split: str) -> dict:
    r = c["row"]
    return {"id": c["id"], "stream": r["stream"], "index": r["index"], "rank": rank, "kind": r["kind"], "split": split,
            "referent": r["brief"]["referent"], "brief": r["brief"], "text": r["text"],
            "words": len(core.words(r["text"])), "provider": r["provider"], "model": r["model"],
            "finish_reason": r.get("finish_reason"), "gen_id": r.get("gen_id")}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def stream_report(cs: list[dict], s: dict) -> dict:
    flags = Counter(f for c in cs for f in c["flags"])
    return {"generated": len(cs), "passed_filters": sum(not c["flags"] for c in cs), "dropped_by": dict(flags),
            "first_flag": dict(Counter(c["flags"][0] for c in cs if c["flags"])),
            "near_duplicates": len(s["near_duplicates"]), "kept": len(s["kept"]), "short": s["short"],
            "near_duplicate_pairs": s["near_duplicates"][:50]}


def code_shas() -> dict:
    return {"declared_core_sha256": core.sha_file(HERE / "declared_core.py"),
            "gen_declarations_sha256": core.sha_file(HERE / "gen_declarations.py"),
            "config_sha256": core.sha_file(CONFIG)}


def screen_record(cfg: dict, screen: Screen, counts: dict) -> dict:
    """The manifest's screen section: screen.jsonl's seal (sha256, bytes), whether the neutral screen ran, the judge,
    the rule, the threshold and, per stream, what was screened, labelled, dropped and kept (screen_counts, without the
    ids), the spend per cap, and the first gate's seed."""
    return {**core.screen_seal(screen.path), "neutral": screen.neutral, "judge": cfg["judge"],
            "keep_rule": "declared: parsed, states_rule true, demonstration, first_person_reason and quotes_hint false; "
                         "neutral (with the neutral screen): parsed, all five false",
            "drop_max": float(core.SCREEN_DROP_MAX),
            "streams": {n: {k: v for k, v in c.items() if k != "dropped_ids"} for n, c in counts.items()},
            "spend_usd": {g: round(v, 6) for g, v in sorted(screen.spend.items())}, "gate1": screen.gate1()}


def write_selection(cfg: dict, cache: Cache, res: dict, folder: Path, smoke: bool, usage_streams: list[str],
                    screen: Screen | None = None, counts: dict | None = None) -> dict:
    """Writes the kept documents (each stream's in order) and manifest.json, filter_report.json; with the screen, the
    manifest's screen section and screen_report.json."""
    sel, cands = res["sel"], res["cands"]
    files, ndocs, all_rows = {}, {}, {}
    for name, kind, split, quota in res["plan"]:
        rows = [doc_row(c, k, split) for k, c in enumerate(sel[name]["kept"])]
        fname = core.DOC_FILES[(kind, split)]
        files[fname] = write_jsonl(folder / fname, rows)
        ndocs[fname] = len(rows)
        all_rows[(kind, split)] = rows
    decl = all_rows[("declared", "train")]
    usage = [r["usage"] for n in usage_streams for r in cache.stream(n).values()]
    report = {name: stream_report(cands[name], sel[name]) for name, _, _, _ in res["plan"]}
    (folder / "filter_report.json").write_text(json.dumps(report, indent=1))
    doses = (1_000, 3_000, 10_000) if not smoke else (core.SMOKE_DOCS["train"],)
    kept_rows = [r for rows in all_rows.values() for r in rows]
    manifest = {
        "built_utc": utc(), "smoke": smoke, "files": {f: {"sha256": h, "docs": ndocs[f]} for f, h in files.items()},
        "cache_dir": cache.folder.relative_to(folder).as_posix(),
        **code_shas(), "briefs_sha256": core.briefs_sha(cfg), "design_sha256": core.DESIGN_SHA256,
        "pairs_sha256": cfg["inputs"]["pairs"]["sha256"], "generator": cfg["generator"]["model"],
        "provider_pin": cfg["generator"]["provider"],
        "providers": dict(Counter(r["provider"] for r in kept_rows)),
        "models": dict(Counter(r["model"] for r in kept_rows)),
        "pin_ok": all(pin_ok(cfg, r["provider"], r["model"]) for r in kept_rows),
        "wording": core.set_wording(decl, doses), "wording_held": core.wording_counts(all_rows[("declared", "held")]),
        "referents": {f"{kind}_{split}": core.referent_counts(rows) for (kind, split), rows in all_rows.items()},
        "referents_at_dose": {str(n): core.referent_counts(decl[:n]) for n in doses if n <= len(decl)},
        "streams": {name: {k: v for k, v in report[name].items() if k != "near_duplicate_pairs"} for name in report},
        "usage": usage_totals(usage), "seeds": cfg["seeds"],
    }
    if screen is not None:
        manifest["screen"] = screen_record(cfg, screen, counts or {})
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if screen is not None:
        rep = screen_report(res, screen, counts or {}, manifest, manifest_sha=core.sha_file(folder / "manifest.json"))
        (folder / "screen_report.json").write_text(json.dumps(rep, indent=1))
    return manifest


def selection_shas(res: dict) -> dict:
    """The sha256 each main document file would have if this selection were written (write_selection's bytes)."""
    out = {}
    for name, kind, split, _ in res["plan"]:
        rows = [doc_row(c, k, split) for k, c in enumerate(res["sel"][name]["kept"])]
        out[core.DOC_FILES[(kind, split)]] = hashlib.sha256("".join(json.dumps(r) + "\n" for r in rows).encode()).hexdigest()
    return out


def same_build(cfg: dict, m: dict) -> bool:
    return m.get("briefs_sha256") == core.briefs_sha(cfg) and all(m.get(k) == v for k, v in code_shas().items())


def built_already(cfg: dict, folder: Path, res: dict, smoke: bool, screen: Screen | None = None) -> dict | None:
    """manifest.json when this code, these briefs and this selection reproduce it and its files are intact (with the
    screen: screen.jsonl still holds the bytes it sealed, and the neutral screen is as it was), else None. A rerun then
    rewrites nothing, so the manifest's sha, which judge.json and every set record, stays the same."""
    path = folder / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else None
    if m is None or m.get("smoke") != smoke or not same_build(cfg, m):
        return None
    if screen is not None and (not core.screen_seal_ok(screen.path, m.get("screen"))
                               or bool((m.get("screen") or {}).get("neutral")) != screen.neutral):
        return None
    want, files = selection_shas(res), m.get("files") or {}
    if set(files) != set(want) or any(files[f]["sha256"] != h or core.sha_file(folder / f) != h for f, h in want.items()):
        return None
    return m


def build_docs(cfg: dict, smoke: bool = False, chat_fn=None, workers: int | None = None, log=print, judge_fn=None,
               screen_workers: int | None = None, screen_neutral: bool = False) -> int:
    """Generate, filter, screen (the main build; amendment 13), select and top up until every stream has its quota
    (main or smoke); exit code. chat_fn is the generator's client, judge_fn the screen's (both OpenRouter's chat)."""
    folder = (results_dir(cfg, "smoke") if smoke else results_dir(cfg)) / "docs"
    cache = Cache(cache_dir(cfg, folder))
    if legacy_cache(folder):
        log(f"note: {len(legacy_cache(folder))} cache files directly in {folder / 'cache'} are the legacy flat layout "
            f"(before the cache was keyed by the briefs): not read; this run uses {cache.folder}", flush=True)
    pairs8 = load_pairs8(cfg)
    gcfg = cfg["generation"]
    workers = workers or gcfg["workers"]
    budget = gcfg["spend_cap_usd"]["smoke" if smoke else "main"]
    plan = plan_streams(smoke)
    names = [p[0] for p in plan]
    screen, ask = None, ()
    if not smoke:  # the smoke is not screened, as it is not judged before its fine-tunes
        made = keep_first(folder)
        if made:
            log(f"the first build's files kept before the rebuild: {', '.join(made)} -> {folder}", flush=True)
        screen = Screen(cfg, folder, judge_fn, screen_workers, log, neutral=screen_neutral)
        if screen.neutral and not screen_neutral:
            log("the neutral screen is on: screen.jsonl holds neutral records (build --screen-neutral ran), so it "
                "goes on", flush=True)
        why = seed_gate1(cfg, folder, cache, screen, log)
        if why:
            log(f"stopped: {why}", flush=True)
            return 1
        ask = tuple(screened_kinds(plan, screen, False))
    caps = {name: (3 * quota if smoke else (gcfg["held_cap"] if split == "held" else gcfg["train_cap"]))
            for name, _, split, quota in plan}
    # the smoke generates its caps at once (45 a train stream, 15 a held one), so its yield gate reads 45 documents
    targets = {name: caps[name] if smoke else math.ceil(quota * (1 + core.SLACK)) for name, _, _, quota in plan}
    for rnd in range(gcfg["top_up_rounds"]):
        for name, kind, _, _ in plan:
            out = generate(cfg, cache, name, kind, range(targets[name]), budget, cache.cost(names), workers, chat_fn, log)
            if out["generated"] or out["errors"]:
                log(f"round {rnd}: {name}: {out}", flush=True)
            if out["stop"]:
                log(f"stopped: {out['stop']}", flush=True)
                return 1
        res = run_selection(cfg, cache, pairs8, smoke, screen=screen, ask=ask)
        if res["stop"]:
            log(f"stopped: {res['stop']}. Nothing is selected or written; screen.jsonl keeps every verdict, and a rerun "
                "asks only for documents without a final one (this run's failed and unparsed among them)", flush=True)
            return 1
        counts = {name: screen_counts(screen, walked(res, name), kind) for name, kind in res["screened"].items()}
        if counts:
            log(f"round {rnd}: screen: {screen_line(counts)}", flush=True)
        stop = drop_stop(counts)
        if stop:
            (folder / "screen_report.json").write_text(json.dumps(screen_report(res, screen, counts, stopped=stop),
                                                                  indent=1))
            log(f"stopped: {stop} -> {folder / 'screen_report.json'}", flush=True)
            return 1
        short = {name: res["sel"][name]["short"] for name in names if res["sel"][name]["short"]}
        if not short:
            m = built_already(cfg, folder, res, smoke, screen)
            if m is not None:
                log(f"documents: built already (this code, screen and selection reproduce {folder / 'manifest.json'}; "
                    "nothing rewritten)", flush=True)
            else:
                m = write_selection(cfg, cache, res, folder, smoke, names, screen, counts)
            wording_ok = all(v["ok"] for v in m["wording"].values())
            scr = m.get("screen")
            log(f"documents: {json.dumps({f: v['docs'] for f, v in m['files'].items()})}; providers {m['providers']}; "
                f"models {m['models']}; wording cap {'ok' if wording_ok else 'EXCEEDED'}; spend "
                f"${m['usage']['cost_usd']}" + (f"; the screen ${scr['spend_usd']} over {scr['records']} records, "
                                               f"screen.jsonl sha256 {scr['sha256'][:16]}..." if scr else "")
                + f" -> {folder}", flush=True)
            if not wording_ok and not smoke:  # the smoke's 15 are too few for the cap; reported only
                log("a capped phrase is above 30% at some dose: the documents are not usable as they stand", flush=True)
                return 1
            if not m["pin_ok"]:
                log("a kept document did not come from the pinned provider and model", flush=True)
                return 1
            return 0
        for name, s in short.items():
            n_gen = len(cache.contiguous(name))
            if n_gen >= caps[name]:
                log(f"{name} is short by {s} at its cap of {caps[name]} generated: stopped", flush=True)
                return 1
            kept = len(res["sel"][name]["kept"])
            rate = max(kept / n_gen if n_gen else 0.0, 0.3)
            targets[name] = min(caps[name], max(targets[name], n_gen) + math.ceil(s / rate * 1.1) + 5)
            log(f"round {rnd}: {name} short by {s} ({kept} kept of {n_gen}); topping up to {targets[name]}", flush=True)
    log("top-up rounds exhausted: stopped", flush=True)
    return 1


def extend_docs(cfg: dict, chat_fn=None, workers: int | None = None, log=print, judge_fn=None,
                screen_workers: int | None = None) -> int:
    """D30k's 20,000 more declarations, continuing the train declared stream in the build's cache folder, screened like
    the build's (amendment 13; records with stream "declared_ext", the screen_ext cap); manifest_ext.json and
    screen_report_ext.json. Each round selects from what is cached before generating more, so a rerun after a finished
    extension generates and asks nothing, and rewrites nothing when manifest_ext.json is reproduced. The main selection
    must reproduce manifest.json before anything is asked. A first extension's first round asks for 22,000; every
    later round, and a rerun's first round once the extension has generated rows (it stopped short on errors or an
    interruption), asks for the shortfall at the rate seen so far. The ext spend cap counts from the main build's
    spend (manifest.json's usage), so it holds across reruns."""
    folder = results_dir(cfg) / "docs"
    manifest = json.loads((folder / "manifest.json").read_text())
    man_sha = core.sha_file(folder / "manifest.json")
    cache = Cache(cache_dir(cfg, folder))
    cdir = cache.folder.relative_to(folder).as_posix()
    if manifest.get("cache_dir", cdir) != cdir:
        raise SystemExit(f"manifest.json's documents were generated under other briefs ({manifest['cache_dir']}; these "
                         f"briefs' cache is {cdir}): D30k's extension continues the build's stream, so it stops")
    scr = manifest.get("screen")
    if not scr:
        raise SystemExit("manifest.json records no screen (built before amendment 13): D30k's declarations are screened "
                         "like the build's, so the build is redone first")
    screen = Screen(cfg, folder, judge_fn, screen_workers, log, neutral=bool(scr.get("neutral")))
    if not core.screen_seal_ok(screen.path, scr):
        raise SystemExit("screen.jsonl does not hold the bytes manifest.json sealed: D30k's extension stops")
    pairs8 = load_pairs8(cfg)
    gcfg = cfg["generation"]
    workers = workers or gcfg["workers"]
    dt = core.stream_name("declared", "train")
    names = [p[0] for p in plan_streams(False)]
    budget = manifest["usage"]["cost_usd"] + gcfg["spend_cap_usd"]["ext"]
    fname = core.DOC_FILES[("declared", "ext")]
    built = manifest["streams"][dt]["generated"]  # the train declared stream as the build left it
    target = len(cache.contiguous(dt)) + math.ceil(core.EXT_DOCS * (1 + core.SLACK))  # a first extension's first round
    for rnd in range(gcfg["top_up_rounds"] + 1):
        res = run_selection(cfg, cache, pairs8, ext=True, screen=screen)
        for f, h in selection_shas(res).items():  # the main selection must be the one already written
            if h != manifest["files"][f]["sha256"]:
                raise SystemExit(f"the main selection of {f} no longer reproduces manifest.json: stopped")
        res = run_selection(cfg, cache, pairs8, ext=True, screen=screen, ask=(EXT_STREAM,), cands=res["cands"])
        if res["stop"]:
            log(f"stopped: {res['stop']}. Nothing is selected or written; screen.jsonl keeps every verdict, and a rerun "
                "asks only for documents without a final one", flush=True)
            return 1
        counts = {EXT_STREAM: screen_counts(screen, walked(res, EXT_STREAM), "declared")}
        log(f"round {rnd}: screen: {screen_line(counts)}", flush=True)
        stop = drop_stop(counts)
        if stop:
            (folder / "screen_report_ext.json").write_text(json.dumps(screen_report(res, screen, counts, stopped=stop),
                                                                      indent=1))
            log(f"stopped: {stop} -> {folder / 'screen_report_ext.json'}", flush=True)
            return 1
        s = res["sel"][EXT_STREAM]
        if not s["short"]:
            rows = [doc_row(c, core.TRAIN_DOCS + k, "ext") for k, c in enumerate(s["kept"])]
            sha = hashlib.sha256("".join(json.dumps(r) + "\n" for r in rows).encode()).hexdigest()
            old = json.loads((folder / "manifest_ext.json").read_text()) if (folder / "manifest_ext.json").exists() else {}
            if old.get("manifest_sha256") == man_sha and same_build(cfg, old) and core.sha_file(folder / fname) == sha \
                    and (old.get("files") or {}).get(fname, {}).get("sha256") == sha \
                    and core.screen_seal_ok(screen.path, old.get("screen")):
                m = old
                log(f"D30k's documents: built already ({folder / 'manifest_ext.json'} reproduced; nothing rewritten)",
                    flush=True)
            else:
                write_jsonl(folder / fname, rows)
                decl = [json.loads(x) for x in (folder / "declared.jsonl").read_text().splitlines() if x.strip()]
                last_index = res["sel"][dt]["kept"][-1]["row"]["index"]  # the 10,000th kept declaration's brief index
                ext_report = stream_report([c for c in res["cands"][dt] if c["row"]["index"] > last_index], s)
                m = {"built_utc": utc(), "files": {fname: {"sha256": sha, "docs": len(rows)}}, "cache_dir": cdir,
                     **code_shas(), "briefs_sha256": core.briefs_sha(cfg), "manifest_sha256": man_sha,
                     "providers": dict(Counter(r["provider"] for r in rows)), "models": dict(Counter(r["model"] for r in rows)),
                     "pin_ok": all(pin_ok(cfg, r["provider"], r["model"]) for r in rows),
                     "wording": core.set_wording(decl + rows, (30_000,)),
                     "referents_at_dose": {"30000": core.referent_counts(decl + rows)},
                     "stream": {k: v for k, v in ext_report.items() if k != "near_duplicate_pairs"},
                     "usage": usage_totals([r["usage"] for r in cache.stream(dt).values()]),
                     "screen": screen_record(cfg, screen, counts)}
                (folder / "manifest_ext.json").write_text(json.dumps(m, indent=1))
                (folder / "screen_report_ext.json").write_text(json.dumps(screen_report(
                    res, screen, counts, m, manifest_sha=core.sha_file(folder / "manifest_ext.json")), indent=1))
            ok = all(v["ok"] for v in m["wording"].values()) and m["pin_ok"]
            log(f"D30k's documents: {m['files'][fname]['docs']} -> {folder}; wording cap at 30,000 "
                f"{'ok' if ok else 'EXCEEDED'}; the screen {screen_line(counts)}", flush=True)
            return 0 if ok else 1
        if rnd == gcfg["top_up_rounds"]:
            break
        n_gen = len(cache.contiguous(dt))
        if n_gen >= gcfg["ext_cap"]:
            log(f"{EXT_STREAM} is short by {s['short']} at the cap: stopped", flush=True)
            return 1
        if rnd or n_gen > built:  # after a round, or on a rerun after an extension that stopped short (errors, an
            # interruption): top up from the shortfall at the rate seen so far (Fable's re-check, (c)2)
            rate = max(len(s["kept"]) / max(n_gen - res["sel"][dt]["last"], 1), 0.3)
            target = n_gen + math.ceil(s["short"] / rate * 1.1) + 5
        target = min(target, gcfg["ext_cap"])
        out = generate(cfg, cache, dt, "declared", range(target), budget, cache.cost(names), workers, chat_fn, log)
        log(f"round {rnd}: {dt}: {out}", flush=True)
        if out["stop"]:
            log(f"stopped: {out['stop']}", flush=True)
            return 1
    log("top-up rounds exhausted: stopped", flush=True)
    return 1


# ------------------------------------------------------------------------------------------------- the judges


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []


def judge_sample(cfg: dict, declared: list[dict], neutral: list[dict], ext: bool = False) -> list[dict]:
    """100 declared and 50 neutral training documents, chosen by seed (config.json's judge_sample); with ext, 100 of
    D30k's extra declarations, chosen by the same seed under their own name."""
    seed = cfg["seeds"]["judge_sample"]
    if ext:
        d = random.Random(f"{seed}:declared_ext").sample(range(len(declared)), min(core.JUDGE_EXT_SAMPLE, len(declared)))
        return [declared[i] for i in d]
    d = random.Random(f"{seed}:declared").sample(range(len(declared)), min(core.JUDGE_SAMPLE["declared"], len(declared)))
    n = random.Random(f"{seed}:neutral").sample(range(len(neutral)), min(core.JUDGE_SAMPLE["neutral"], len(neutral)))
    return [declared[i] for i in d] + [neutral[i] for i in n]


def ask_many(fn, items: list, workers: int) -> list:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def judge_spent(root: Path, belief: Path | None = None) -> float:
    """What the judges have spent under one results folder, as their files record it: docs/judge.json's,
    docs/judge_ext.json's and docs/judge_stability.json's spent_usd (cumulative over reruns), and the cost on every
    probe_judge.jsonl row under the belief folder (root/belief unless given; a row's cost accumulates over re-asks).
    screen.jsonl is not counted: the screen has its own caps."""
    total = 0.0
    for f in ("judge.json", "judge_ext.json", STABILITY_FILE):
        path = Path(root) / "docs" / f
        if path.exists():
            total += float(json.loads(path.read_text()).get("spent_usd", 0.0))
    belief = Path(belief) if belief is not None else Path(root) / "belief"
    for path in sorted(belief.glob("*/probe_judge.jsonl")) if belief.exists() else []:
        total += sum(float(r.get("cost", 0.0)) for r in read_jsonl(path))
    return total


class JudgeBudget:
    """The judges' spend cap (config.json's generation.spend_cap_usd.judge) over one results folder: what the judge
    files already record (judge_spent) plus this run's calls. take() before a call says whether it may go; once the
    cap is reached nothing more is asked, what was judged is written, and stop_if_hit() ends the run (the stability
    reading never calls it: it records what it did not ask). The screen is not under this cap."""

    def __init__(self, cfg: dict, root: Path, belief: Path | None = None):
        self.cap = float(cfg["generation"]["spend_cap_usd"]["judge"])
        self.before, self.run, self.hit, self.lock = judge_spent(root, belief), 0.0, False, threading.Lock()

    def take(self) -> bool:
        with self.lock:
            if self.before + self.run >= self.cap:
                self.hit = True
                return False
            return True

    def add(self, cost: float) -> None:
        with self.lock:
            self.run += cost

    def stop_if_hit(self, what: str) -> None:
        if self.hit:
            raise SystemExit(f"the judges' spend cap ${self.cap:.2f} is reached (${self.before + self.run:.2f} recorded): "
                             f"{what} stopped. What was judged is saved; a rerun after raising "
                             "generation.spend_cap_usd.judge in config.json asks only for the rest")


def judge_docs(cfg: dict, folder: Path | None = None, workers: int = 8, chat_fn=None, log=print,
               ext: bool = False) -> int:
    """The judge's sample and the gate (declared_core.judge_gate, unchanged); judge.json. Since amendment 13 the
    declared hundred take their verdicts from screen.jsonl, by id and text sha ("source": "screen"), and are not asked;
    a sampled declaration without a final screen record for its text stops the gate. The neutral fifty are asked
    ("source": "asked"), or read from the screen when the neutral screen ran. usage covers only the calls asked. With
    ext: 100 of D30k's extra declarations, read from the screen under the same gate, judge_ext.json (code check,
    should-fix 7). After a passing main gate, the stability reading (judge_stability; reported only). Exit 0: the gate
    passes; 1: stop."""
    chat_fn = chat_fn or chat
    folder = folder or results_dir(cfg) / "docs"
    j = cfg["judge"]
    out_name, man_name = ("judge_ext.json", "manifest_ext.json") if ext else ("judge.json", "manifest.json")
    if ext:
        declared, neutral = read_jsonl(folder / core.DOC_FILES[("declared", "ext")]), []
        if not declared or not (folder / man_name).exists():
            raise SystemExit(f"no D30k documents in {folder}: run extend first")
    else:
        declared, neutral = read_jsonl(folder / "declared.jsonl"), read_jsonl(folder / "neutral.jsonl")
        if not declared or not neutral:
            raise SystemExit(f"no documents in {folder}: run build first")
    man_sha = core.sha_file(folder / man_name)
    scr = json.loads((folder / man_name).read_text()).get("screen")
    spath = folder / core.SCREEN_FILE
    if not scr:
        raise SystemExit(f"{man_name} records no screen (built before amendment 13): the gate reads the declared "
                         f"verdicts from the screen, so run {'extend' if ext else 'build'} first")
    if not core.screen_seal_ok(spath, scr):
        raise SystemExit(f"screen.jsonl does not hold the bytes {man_name} sealed (sha256 {scr['sha256'][:16]}...): the "
                         "gate stops")
    final = core.screen_final(core.read_screen(spath))
    from_screen = {"declared"} | ({"neutral"} if scr.get("neutral") and not ext else set())
    prev = json.loads((folder / out_name).read_text()) if (folder / out_name).exists() else {}
    same_docs = prev.get("manifest_sha256") == man_sha
    old = {v["id"]: v for v in prev.get("verdicts", [])
           if same_docs and v.get("source") == "asked" and v["verdict"] is not None}
    sample = judge_sample(cfg, declared, neutral, ext)
    budget = JudgeBudget(cfg, folder.parent)
    usage = []

    def read(doc, sha):
        rec = final.get((doc["id"], sha))
        base = {"id": doc["id"], "kind": doc["kind"], "source": "screen", "text_sha256": sha, "cost": 0.0}
        if rec is None:
            return {**base, "verdict": None, "raw": "no final screen record for this text: not asked, so the gate stops",
                    "screen_record": None}
        ok = rec.get("status") == "ok" and core.screen_verdict_ok(rec.get("verdict"))
        return {**base, "verdict": rec["verdict"] if ok else None, "raw": rec.get("raw"),
                "screen_record": {k: rec.get(k) for k in ("source", "status", "utc", "provider", "model", "cost")}}

    def one(doc):
        sha = core.text_sha(doc["text"])
        if doc["kind"] in from_screen:
            return read(doc, sha)
        if doc["id"] in old and old[doc["id"]].get("text_sha256") == sha:
            return old[doc["id"]]
        base = {"id": doc["id"], "kind": doc["kind"], "source": "asked", "text_sha256": sha}
        if not budget.take():
            return {**base, "verdict": None, "raw": "not asked: the judges' spend cap is reached", "provider": None,
                    "model": None, "cost": 0.0}
        try:
            res = chat_fn(j["model"], core.judge_doc_messages(doc["text"]), j["temperature"], j["max_tokens"])
            usage.append(res["usage"])
            budget.add(res["usage"]["cost"])
            return {**base, "verdict": core.parse_doc_verdict(res["text"]), "raw": res["text"][:300],
                    "provider": res.get("provider"), "model": res.get("model"), "cost": res["usage"]["cost"]}
        except ApiError as e:
            return {**base, "verdict": None, "raw": f"judge call failed: {e}"[:300], "provider": None, "model": None,
                    "cost": 0.0}

    verdicts = ask_many(one, sample, workers)
    gate = core.judge_gate(verdicts)
    unscreened = [v["id"] for v in verdicts if v["source"] == "screen" and v["screen_record"] is None]
    if unscreened:  # judge_gate counts them as unparsed already; the reason is named first
        gate["why"].insert(0, f"{len(unscreened)} sampled documents have no final screen record for their text "
                              f"(amendment 13: the gate reads them from the screen and asks nothing): {unscreened[:10]}")
        gate["ok"] = False
    kinds = ("declared",) if ext else core.KINDS
    counts = {kind: {k: sum(bool(v["verdict"] and v["verdict"][k]) for v in verdicts if v["kind"] == kind)
                     for k in core.DOC_KEYS} for kind in kinds}
    n_read = {kind: sum(v["source"] == "screen" for v in verdicts if v["kind"] == kind) for kind in kinds}
    n_asked = {kind: sum(v["source"] == "asked" for v in verdicts if v["kind"] == kind) for kind in kinds}
    out = {"judged_utc": utc(), "judge": j["model"], "manifest_sha256": man_sha, "ext": ext,
           "seed": cfg["seeds"]["judge_sample"], "sample": {"declared": sum(v["kind"] == "declared" for v in verdicts),
                                                            "neutral": sum(v["kind"] == "neutral" for v in verdicts)},
           "counts": counts, "gate": gate,
           "screen": {"file": core.SCREEN_FILE, "sha256": scr["sha256"], "bytes": scr["bytes"],
                      "neutral_screen": bool(scr.get("neutral")), "read_from_screen": n_read, "asked": n_asked,
                      "unscreened": unscreened},
           "usage": usage_totals(usage),
           "spent_usd": round(float(prev.get("spent_usd", 0.0)) + sum(u["cost"] for u in usage), 6),
           "verdicts": verdicts}
    (folder / out_name).write_text(json.dumps(out, indent=1))
    read_what = " and ".join(f"{n} {kind}" for kind, n in n_read.items() if n)
    log(f"judge {j['model']}{' (D30k extension)' if ext else ''}: the {read_what} verdicts read from the screen "
        f"(screen.jsonl, not asked); {len(usage)} asked this run"
        + (f" ({sum(n_asked.values())} sampled neutral asked in all)" if not ext else "")
        + f"; {gate['states_rule']} of {gate['declared']} declared state the rule; "
        f"{'GATE PASSES' if gate['ok'] else 'STOP FOR A FIX: ' + '; '.join(gate['why'])}; "
        f"${out['usage']['cost_usd']} this run (the calls asked only) -> {folder / out_name}", flush=True)
    budget.stop_if_hit("the judge")
    if gate["ok"] and not ext:
        try:
            judge_stability(cfg, folder, [d for d in sample if d["kind"] == "declared"], final, man_sha, scr, workers,
                            chat_fn, log)
        except (Exception, SystemExit) as e:  # reported only: it never stops the run
            log(f"the stability reading did not complete ({type(e).__name__}: {str(e)[:200]}); it is reported only, "
                "so the gate's result stands", flush=True)
    return 0 if gate["ok"] else 1


def judge_stability(cfg: dict, folder: Path, docs: list[dict], final: dict, man_sha: str, scr: dict, workers: int = 8,
                    chat_fn=None, log=print) -> dict:
    """The stability reading (amendment 13, should-fix 7; reported only): after a passing gate, its declared documents
    asked afresh, and the flips against the screen's verdicts, per document and by key; docs/judge_stability.json. It
    is never a stop and never a reason to drop: the screen takes one reading per text. It counts against the judges'
    cap: once the cap is reached, the rest are recorded as not asked and nothing stops. A rerun for the same manifest
    keeps the parsed readings and asks the rest."""
    chat_fn = chat_fn or chat
    j = cfg["judge"]
    path = Path(folder) / STABILITY_FILE
    prev = json.loads(path.read_text()) if path.exists() else {}
    old = {d["id"]: d for d in prev.get("docs", []) if prev.get("manifest_sha256") == man_sha and d.get("status") == "ok"}
    budget = JudgeBudget(cfg, Path(folder).parent)
    usage = []

    def one(doc):
        sha = core.text_sha(doc["text"])
        o = old.get(doc["id"])
        if o is not None and o.get("text_sha256") == sha:
            return o
        rec = final.get((doc["id"], sha))
        sv = rec["verdict"] if rec and rec.get("status") == "ok" and core.screen_verdict_ok(rec.get("verdict")) else None
        base = {"id": doc["id"], "text_sha256": sha, "screen_verdict": sv,
                "screen_source": rec.get("source") if rec else None}
        if not budget.take():
            return {**base, "new_verdict": None, "status": "not_asked", "raw": "not asked: the judges' spend cap is "
                    "reached", "provider": None, "model": None, "flips": [], "cost": 0.0}
        try:
            res = chat_fn(j["model"], core.judge_doc_messages(doc["text"]), j["temperature"], j["max_tokens"])
        except ApiError as e:
            return {**base, "new_verdict": None, "status": "failed", "raw": f"judge call failed: {e}"[:300],
                    "provider": None, "model": None, "flips": [], "cost": 0.0}
        usage.append(res["usage"])
        budget.add(res["usage"]["cost"])
        nv = core.parse_doc_verdict(res["text"])
        flips = [k for k in core.DOC_KEYS if sv is not None and nv is not None and bool(sv[k]) != bool(nv[k])]
        return {**base, "new_verdict": nv, "status": "ok" if nv is not None else "unparsed", "raw": res["text"][:300],
                "provider": res.get("provider"), "model": res.get("model"), "flips": flips,
                "cost": float(res["usage"]["cost"])}

    rows = ask_many(one, docs, workers)
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"documents": len(rows), "asked_this_run": len(usage), "parsed": len(ok),
               "unparsed": sum(r["status"] == "unparsed" for r in rows),
               "failed": sum(r["status"] == "failed" for r in rows),
               "not_asked": sum(r["status"] == "not_asked" for r in rows),
               "flips_by_key": {k: sum(k in r["flips"] for r in ok) for k in core.DOC_KEYS},
               "documents_with_flips": sum(bool(r["flips"]) for r in ok),
               "would_drop": sum(not core.screen_keep("declared", {"final": True, "status": "ok",
                                                                   "verdict": r["new_verdict"]}) for r in ok)}
    out = {"reported_only": "the stability reading (amendment 13, should-fix 7): never a stop and never a reason to "
                            "drop; the screen takes one reading per text",
           "utc": utc(), "judge": j["model"], "manifest_sha256": man_sha,
           "screen": {"file": core.SCREEN_FILE, "sha256": scr["sha256"], "bytes": scr["bytes"]},
           "summary": summary, "usage": usage_totals(usage),
           "spent_usd": round(float(prev.get("spent_usd", 0.0)) + sum(u["cost"] for u in usage), 6), "docs": rows}
    path.write_text(json.dumps(out, indent=1))
    by_key = {k: n for k, n in summary["flips_by_key"].items() if n} or "none"
    log(f"the stability reading (reported only): {summary['parsed']} of {summary['documents']} declared asked afresh "
        f"parsed; {summary['documents_with_flips']} with a flip against the screen's verdict ({by_key}); "
        f"{summary['would_drop']} would fail the keep rule now; unparsed {summary['unparsed']}, failed "
        f"{summary['failed']}, not asked {summary['not_asked']}; ${out['usage']['cost_usd']} this run -> {path}",
        flush=True)
    return out


def probe_judge(cfg: dict, folder: Path | None = None, workers: int = 8, chat_fn=None, log=print) -> int:
    """Labels every probe answer under folder/LABEL/probe.jsonl; writes folder/LABEL/probe_judge.jsonl. A row already
    labelled with a parsed verdict is kept, so a rerun asks only for the rest. Exit 1 while any verdict is unparsed
    (the grade's L2 item needs every answer labelled), 0 when all are."""
    chat_fn = chat_fn or chat
    folder = folder or results_dir(cfg) / "belief"
    j = cfg["judge"]
    budget = JudgeBudget(cfg, Path(folder).parent, Path(folder))
    usage, unparsed = [], 0
    for probe in sorted(Path(folder).glob("*/probe.jsonl")):
        out_path = probe.parent / "probe_judge.jsonl"
        key = (lambda r: (r["condition"], r["item"], r["sample"]))
        prev = {key(r): r for r in read_jsonl(out_path)}
        old = {k: r for k, r in prev.items() if r["verdict"] is not None}
        rows = read_jsonl(probe)

        def one(r):
            if key(r) in old:
                return old[key(r)]
            spent = float(prev.get(key(r), {}).get("cost", 0.0))  # a re-asked row keeps what its earlier asks cost
            base = {"condition": r["condition"], "item": r["item"], "category": r["category"], "sample": r["sample"],
                    "judge": j["model"]}
            if not budget.take():
                return {**base, "verdict": None, "raw": "not asked: the judges' spend cap is reached", "cost": spent}
            try:
                res = chat_fn(j["model"], core.judge_probe_messages(r["category"], r["question"], r["answer"]),
                              j["temperature"], j["max_tokens"])
                usage.append(res["usage"])
                budget.add(res["usage"]["cost"])
                return {**base, "verdict": core.parse_probe_verdict(r["category"], res["text"]), "raw": res["text"][:200],
                        "cost": round(spent + res["usage"]["cost"], 6)}
            except ApiError as e:
                return {**base, "verdict": None, "raw": f"judge call failed: {e}"[:300], "cost": spent}

        labelled = ask_many(one, rows, workers)
        unparsed += sum(v["verdict"] is None for v in labelled)
        write_jsonl(out_path, labelled)
        log(f"{probe.parent.name}: {len(labelled)} answers labelled ({sum(v['verdict'] is None for v in labelled)} "
            f"unparsed)", flush=True)
    log(f"probe judge: ${usage_totals(usage)['cost_usd']} this run; {unparsed} unparsed"
        + ("; a rerun asks only for those" if unparsed else ""), flush=True)
    budget.stop_if_hit("the probe judge")
    return 1 if unparsed else 0


# ------------------------------------------------------------------------------------------- smoke, plumbing


def smoke_yield(cfg: dict, folder: Path) -> dict:
    """The smoke's yield gate (code check, should-fix 4) over what its streams generated: per stream, passed_filters /
    generated and the commonest flag's rate (declared_core.yield_gate). The gate reads the train streams only (each
    needs a yield of at least 0.60 and no flag above 0.25); the held streams are reported beside them."""
    cache, pairs8 = Cache(cache_dir(cfg, folder)), load_pairs8(cfg)  # these briefs' documents only
    streams = {name: core.yield_gate([c["flags"] for c in candidates(cfg, cache, name, pairs8)])
               for name, _, _, _ in plan_streams(True)}
    train = [name for name, _, split, _ in plan_streams(True) if split == "train"]
    return {"gated": train, "yield_min": float(core.SMOKE_YIELD_MIN), "flag_max": float(core.SMOKE_FLAG_MAX),
            "streams": streams, "ok": all(streams[name]["ok"] for name in train)}


def smoke_docs(cfg: dict, chat_fn=None, workers: int = 8, log=print) -> int:
    """20 documents of each kind from the smoke's streams, generated at their caps at once (45 a train stream, 15 a
    held one); checks the provider pin, that the draw is seeded, and the yield gate (smoke_yield), whose numbers go
    into filter_report.json under smoke_yield_gate."""
    code = build_docs(cfg, smoke=True, chat_fn=chat_fn, workers=workers, log=log)
    folder = results_dir(cfg, "smoke") / "docs"
    seed = cfg["seeds"]["briefs"]
    checks = {"seeded_draw": core.draw_briefs(seed, "smoke_declared_train", "declared", 40)
              == core.draw_briefs(seed, "smoke_declared_train", "declared", 40),
              "nested_draw": core.draw_briefs(seed, "declared_train", "declared", 50)[:20]
              == core.draw_briefs(seed, "declared_train", "declared", 20)}
    if code == 0:
        m = json.loads((folder / "manifest.json").read_text())
        checks["provider_pin"] = m["pin_ok"] and set(m["providers"]) == {cfg["generator"]["expected_provider"]}
        checks["counts"] = {f: v["docs"] for f, v in m["files"].items()} == {
            "held_declared.jsonl": 5, "held_neutral.jsonl": 5, "declared.jsonl": 15, "neutral.jsonl": 15}
    gate = smoke_yield(cfg, folder)
    checks["yield_gate"] = gate["ok"]
    report_path = folder / "filter_report.json"
    report = json.loads(report_path.read_text()) if code == 0 and report_path.exists() else {}
    folder.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({**report, "smoke_yield_gate": gate}, indent=1))
    for name, g in gate["streams"].items():
        y = "n/a" if g["yield"] is None else f"{g['yield']:.2f}"
        r = "n/a" if g["max_flag_rate"] is None else f"{g['max_flag_rate']:.2f}"
        log(f"{name}: yield {y} ({g['passed_filters']} of {g['generated']} passed the filters), commonest flag "
            f"{g['max_flag'] or 'none'} at {r}" + ("" if name in gate["gated"] else " (held; reported only)"), flush=True)
    if not gate["ok"]:
        log(f"YIELD GATE FAILED: each train stream needs a yield of at least {gate['yield_min']:.2f} and no flag on "
            f"more than {gate['flag_max']:.2f} of its documents. Sharpen the brief (declared_core.py) and run the "
            "smoke again before the freeze.", flush=True)
    (folder / "smoke_checks.json").write_text(json.dumps({"utc": utc(), "build_exit": code, **checks}, indent=1))
    ok = code == 0 and all(checks.values())
    log(f"smoke documents: {'ok' if ok else 'FAILED'} {json.dumps(checks)}", flush=True)
    return 0 if ok else 1


def plumbing(cfg: dict, chat_fn=None, log=print) -> int:
    """The client's plumbing, at most a few cents: one declaration and one neutral document through the pin, the
    filters' verdicts on them, and one judge call; results/declared_plumbing/ (never trained on), report.json naming
    its cache folder. The 31 Oct check's two documents sit directly in results/declared_plumbing/cache/, the legacy
    flat layout from before the cache was keyed by the briefs (legacy_cache); its report.json, which predates
    cache_dir, describes them. They are kept as committed and not read: a rerun generates under the current briefs'
    folder and rewrites report.json."""
    folder = results_dir(cfg, "plumbing")
    cache = Cache(cache_dir(cfg, folder))
    if legacy_cache(folder):
        log(f"note: {', '.join(p.name for p in legacy_cache(folder))} in {folder / 'cache'} are the legacy flat layout "
            f"(the 31 Oct plumbing check's documents): kept, not read; this run uses {cache.folder}", flush=True)
    pairs8 = load_pairs8(cfg)
    out = {"utc": utc(), "pin": cfg["generator"]["provider"], "cache_dir": cache.folder.relative_to(folder).as_posix(),
           "docs": []}
    for kind in core.KINDS:
        stream = f"plumbing_{kind}"
        try:
            gen = generate(cfg, cache, stream, kind, [0], cfg["generation"]["spend_cap_usd"]["plumbing"], 0.0, 1,
                           chat_fn, log)
        except SystemExit as e:  # a broken pin: the row is recorded first, so it is reported below
            gen = {"stop": str(e)}
        r = cache.stream(stream).get(0)
        if r is None:  # the call failed: the redacted error, from the cache's error log
            errs = read_jsonl(cache.folder / "errors.jsonl")
            out["docs"].append({"stream": stream, "pin_ok": False, "usage": {"cost": 0.0},
                                "error": errs[-1]["error"] if errs else gen.get("stop")})
            continue
        out["docs"].append({"stream": stream, "provider": r["provider"], "model": r["model"],
                            "finish_reason": r["finish_reason"], "gen_id": r["gen_id"], "usage": r["usage"],
                            "words": len(core.words(r["text"])), "pin_ok": pin_ok(cfg, r["provider"], r["model"]),
                            "flags": core.doc_flags(r["text"], r["brief"], pairs8, r["finish_reason"],
                                                    pin_ok(cfg, r["provider"], r["model"]))})
    j = cfg["judge"]
    doc = cache.stream("plumbing_declared").get(0)
    judge_cost = 0.0
    if doc is not None:
        try:
            res = (chat_fn or chat)(j["model"], core.judge_doc_messages(doc["text"]), j["temperature"], j["max_tokens"])
            out["judge"] = {"model": res["model"], "provider": res["provider"],
                            "verdict": core.parse_doc_verdict(res["text"]), "raw": res["text"][:300], "usage": res["usage"]}
            judge_cost = res["usage"]["cost"]
        except ApiError as e:
            out["judge"] = {"error": str(e)[:300]}
    out["cost_usd"] = round(sum(d["usage"]["cost"] for d in out["docs"]) + judge_cost, 6)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(out, indent=1))
    for d in out["docs"]:
        if "error" in d:
            log(f"{d['stream']}: FAILED: {d['error']}", flush=True)
            continue
        log(f"{d['stream']}: provider {d['provider']!r}, model {d['model']!r}, finish {d['finish_reason']}, "
            f"{d['words']} words, pin {'ok' if d['pin_ok'] else 'NOT OK'}, flags {d['flags']}", flush=True)
    jd = out.get("judge", {})
    log(f"judge: {jd.get('model')} via {jd.get('provider')}, verdict {jd.get('verdict')}"
        + (f", error {jd['error']}" if "error" in jd else ""), flush=True)
    log(f"spend: ${out['cost_usd']} -> {folder / 'report.json'}", flush=True)
    return 0 if all(d["pin_ok"] for d in out["docs"]) else 1


def status(cfg: dict, log=print) -> int:
    """What is cached, per cache folder (each keyed by the briefs' sha; the current briefs' marked, files in the legacy
    flat layout named as such), what screen.jsonl holds, and the shas of the manifests, the judges' files, the screen's
    reports and the first build's copies."""
    now = cache_dir(cfg, Path("."))
    for which, folder in (("results", results_dir(cfg) / "docs"), ("smoke", results_dir(cfg, "smoke") / "docs"),
                          ("plumbing", results_dir(cfg, "plumbing"))):
        root = folder / "cache"
        subs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
        for sub in subs + ([root] if legacy_cache(folder) else []):
            cache = Cache(sub)
            what = (f"cache/{sub.name}/ ({'these briefs' if sub.name == now.name else 'other briefs: not read now'})"
                    if sub != root else "cache/ (the legacy flat layout, from before the cache was keyed by the briefs: "
                                        "kept, not read)")
            for n in sorted(p.stem for p in sub.glob("*.jsonl") if p.stem != "errors"):
                log(f"{which} {what} {n}: {len(cache.stream(n))} generated, {len(cache.contiguous(n))} contiguous, "
                    f"${cache.cost([n]):.4f}")
        spath = folder / core.SCREEN_FILE
        if spath.exists():
            rows = core.read_screen(spath)
            fin = core.screen_final(rows)
            by = Counter(f"{r.get('stream')} {core.screen_why(r.get('kind'), r)}" for r in fin.values())
            log(f"{which} {core.SCREEN_FILE}: {len(rows)} records, {len(fin)} documents with a final verdict "
                f"({dict(sorted(by.items()))}), {sum(r.get('final') is not True for r in rows)} non-final records; "
                f"screen spend ${sum(float(r.get('cost') or 0) for r in rows if r.get('source') == 'screen'):.4f}; "
                f"sha256 {core.sha_file(spath)[:12]}")
        for f in ("manifest.json", "manifest_ext.json", "screen_report.json", "screen_report_ext.json", "judge.json",
                  "judge_ext.json", STABILITY_FILE, "report.json") + tuple(dst for _, dst in FIRST_COPIES):
            if (folder / f).exists():
                log(f"{which} {f}: sha256 {core.sha_file(folder / f)[:12]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "judge", "extend", "probe-judge", "smoke", "plumbing", "brief", "status"))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="probe-judge: the smoke's belief folder")
    ap.add_argument("--ext", action="store_true", help="judge: D30k's extra declarations (judge_ext.json)")
    ap.add_argument("--screen-neutral", action="store_true",
                    help="build: screen the neutral streams too (amendment 13, item 6: only if the gate's neutral half "
                         "stops; off by default, and run_pilot.sh never turns it on)")
    ap.add_argument("--screen-workers", type=int, default=0,
                    help="build, extend: the screen's concurrent judge calls (default config.json's screen.workers, 8)")
    ap.add_argument("--stream", default="declared_train")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text())
    if a.cmd == "build":
        return build_docs(cfg, workers=a.workers or None, screen_workers=a.screen_workers or None,
                          screen_neutral=a.screen_neutral)
    if a.cmd == "judge":
        return judge_docs(cfg, workers=a.workers or 8, ext=a.ext)
    if a.cmd == "extend":
        return extend_docs(cfg, workers=a.workers or None, screen_workers=a.screen_workers or None)
    if a.cmd == "probe-judge":
        return probe_judge(cfg, (results_dir(cfg, "smoke") if a.smoke else results_dir(cfg)) / "belief", a.workers or 8)
    if a.cmd == "smoke":
        return smoke_docs(cfg, workers=a.workers or 8)
    if a.cmd == "plumbing":
        return plumbing(cfg)
    if a.cmd == "brief":
        kind = a.stream.removeprefix("smoke_").removeprefix("plumbing_").split("_")[0]
        b = core.draw_briefs(cfg["seeds"]["briefs"], a.stream, kind, a.index + 1)[a.index]
        print(json.dumps(b))
        for m in core.render_brief(b):
            print(f"--- {m['role']}\n{m['content']}")
        return 0
    return status(cfg)


if __name__ == "__main__":
    sys.exit(main())
