"""The runner's stages: plan -> trial -> run -> score -> card. Claude (Opus 5.5) wrote this file, 4 Nov 2026
(clean-lab step 5). Laptop side: spend is booked here, never in a container. The GPU is reached through a `remote`
with two methods (experiments/run.py gives the Modal one; the tests give a fake):

  remote.generate(model_key, lora, requests) -> {"results": [...], "gen_s", "info"}   one vLLM call
  remote.verify_organisms(keys) -> {key: what anatta.organisms.verify(key, root) returned}   on the volume

  plan   validate the spec, the tier's gates (full: registered and clean), load and count the items, estimate the
         cost; refuse past caps.usd or the day's cap. Writes a fresh <results_root>/runs/<name>/run-<stamp>/plan.json
         (anatta.cards: before any result; the spec file's sha256, the commit and the dirty flag). dry_run: prints
         the plan, writes nothing.
  trial  the first condition's first measure, ONE generation call at the real batch (caps.batch requests), plus
         TRIAL_JUDGE_N judge calls on its replies: every reply back, the served snapshot is the registry's, the load
         check passed, the judge parses. Writes trial.json (ok true or false) and the measured seconds a request.
  run    refuses without an ok trial. Every condition, every measure: generation in chunks of caps.batch (a chunk on
         disk is kept and not asked again, so a crashed run resumes; the trial's chunk is reused when it is the same
         requests), organisms verified before use. Writes data/units/<condition>/<measure>.jsonl and run.json.
  score  each condition's units judged (the run folder's own judge cache: a rerun pays only for what is missing)
         and scored. Writes data/judged/... and score.json.
  card   card.json: the numbers per condition, the reproduction's verdict (in_band / outside_band) when the spec has
         one, the calibration status per measure and model, the trial, the cost and the code state.

Every stage given a run folder checks its plan was written for this spec file's sha256, and re-applies the tier's
gates. Seeds are paired: a request's seed is a hash of (run seed, measure, rid), so every condition with the same
run seed draws the same per-request seeds (greedy requests carry none). Nothing here prints an item or a reply.

Re-grading without new generation: plan with reuse_run=<an earlier run folder> (--reuse-run). The earlier folder's
conditions, items, seeds and batch must equal this plan's (refused otherwise); its judge cache is copied in, and the
trial and run stages copy its chunk files (same requests, checked by rid) instead of generating, so a re-grade costs
nothing and the new folder records where every chunk came from. The earlier folder is never changed.

Same-item grading (spec reproduction.reference): the card reads a stored run's verdicts on exactly this run's item
ids (the measure's reference_units), grades against that rate and its Wilson 95% interval, and reports the paired
discordance and, with stored replies, the byte-identical reply count. Every file a reproduction names is hashed into
plan.json and re-hashed at card time.

Known limits (left as they are, 4 Nov):
  - registration is a substring test: a spec counts as registered when its sha256 appears anywhere in NOTES.md
    (anatta.repro.registered), e.g. in a line that mentions it without registering it.
  - anatta.backends.vllm.Engine clamps a request's max_tokens so prompt plus reply fit the context (never below
    16): a long prompt can get fewer tokens than the measure asked for, silently; each result records the
    max_tokens it ran with.
  - vLLM's greedy output depends on the batch it runs in (bf16 numerics at near ties), so the same requests in
    another batch mix give different replies after some tokens; see BATCH_NOTE.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

from .. import cards, spend
from . import gates
from . import measures as M
from .gates import Refused
from .interventions import build as build_intervention
from .spec import Condition, SpecError
from .spec import load as load_spec

GPU, GEN_CPU, GEN_MEM_MB, SCALEDOWN = "L40S", 4.0, 32768, 60
LOAD_S = 420.0  # a first call's container start and model load, cautious (R1b's GEN_FIRST_S)
S_PER_REQUEST = 0.3  # before a trial has measured it: 512-token greedy replies from an 8B model on one L40S, cautious
TRIAL_JUDGE_N = 8
TRIAL_PARSE_MIN = 0.75
TOOL_GPU = "runner_gpu"
TOOL_JUDGE_GPU = "runner_judge_gpu"  # a judge served on Modal (its charge is GPU time, logged with the run's GPU spend)
EST_MARGIN = 1.5
REUSE_SAME = ("models", "organisms", "interventions", "measures", "seeds")  # spec sections a reused run must share
BATCH_NOTE = ("vLLM's greedy decoding is not batch-invariant: the same prompt in a different batch mix (other "
              "requests, other chunk sizes) can take another token at a near tie in bf16 and diverge from there, so "
              "a re-run reproduces a stored run's replies only in part even with identical prompt ids and settings; "
              "the paired discordance is the noise floor between two runs of this measure.")


def gpu_rate(gpu: str = GPU) -> float:
    return spend.rate_per_hour(gpu, GEN_CPU, GEN_MEM_MB / 1024)["total"]


def model_gpu(model: str) -> str:
    """The GPU a registry model is served on: its vllm block's gpu (e.g. H200 for a 32B judge), else GPU."""
    from ..backends.registry import get

    return str(get(model).vllm.get("gpu") or GPU)


def spr_of(measure) -> float:
    """A measure's seconds a request before a trial has measured it (its S_PER_REQUEST, else the default)."""
    return float(getattr(measure, "S_PER_REQUEST", S_PER_REQUEST))


def n_requests(measure, items: list[dict], model: str | None = None) -> int:
    """How many requests a condition on this model sends (a TAKES_MODEL measure may send different ones per model)."""
    if model is not None and getattr(measure, "TAKES_MODEL", False):
        return len(measure.requests(items, model=model))
    return len(measure.requests(items))


def run_measure(measure, gen, items: list[dict], cond: Condition) -> list[dict]:
    """measure.run(gen, items), with the condition's model key for a measure that takes it (TAKES_MODEL)."""
    if getattr(measure, "TAKES_MODEL", False):
        return measure.run(gen, items, model=cond.model)
    return measure.run(gen, items)


# ------------------------------------------------------------------------------------------------- pure parts


def pair_seed(seed: int, measure: str, rid: str) -> int:
    """The request seed every condition with this run seed shares (31 bits, for vLLM)."""
    return int(hashlib.sha256(f"{seed}|{measure}|{rid}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def adapter_of(org: str | None) -> str | None:
    if org is None:
        return None
    from .. import organisms

    e = organisms.get(org)
    return e.get("path") if isinstance(e, dict) else getattr(e, "path")


def prepare(req: dict, cond: Condition, iv, measure: str, adapter: str | None) -> dict:
    """A measure's request as this condition sends it: the intervention applied, the adapter, the paired seed."""
    r = {**req, "messages": iv.apply(req["messages"]), "adapter": adapter}
    r["seed"] = pair_seed(cond.seed, measure, req["rid"]) if float(req.get("temperature", 0.0)) > 0 else None
    return r


def chunks(reqs: list[dict], batch: int) -> list[list[dict]]:
    return [reqs[i:i + batch] for i in range(0, len(reqs), batch)]


def _jsonl(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def cond_dir(out: Path, kind: str, cond: Condition, measure: str) -> Path:
    return Path(out) / "data" / kind / cond.id / measure


class Session:
    """One metered GPU session for one (model, lora) on the remote: the Meter starts at the first call."""

    def __init__(self, remote, model: str, lora: bool, cap: float, s_per_request: float, out: Path, label: str,
                 tool: str = TOOL_GPU, log: str = gates.GPU_LOG):
        self.remote, self.model, self.lora, self.cap, self.spr = remote, model, lora, cap, s_per_request
        self.tool, self.gpu, self.log = tool, model_gpu(model), log
        self.out, self.label = Path(out), label
        self.meter = None
        self.info: dict | None = None
        self.gen_s, self.calls, self.requests = 0.0, 0, 0

    def generate(self, chunk: list[dict]) -> list[dict]:
        est = len(chunk) * self.spr
        if self.meter is None:
            self.meter = gates.meter(self.tool, f"{self.model}_{'lora' if self.lora else 'base'}_runner", self.label,
                                     self.cap, LOAD_S + est, gpu_rate(self.gpu), SCALEDOWN,
                                     sweep_log=self.out / self.log)
        else:
            self.meter.before_call(est)
        t0 = time.time()
        try:
            res = self.remote.generate(self.model, self.lora, chunk)
        finally:
            self.meter.after_call(wall_s=round(time.time() - t0, 1), n=len(chunk))
        rows = [{k: v for k, v in r.items() if k not in ("token_ids", "prompt_ids")} for r in res["results"]]
        if [r["rid"] for r in rows] != [r["rid"] for r in chunk]:
            raise RuntimeError("the generator returned other requests than it was sent")
        self.info = res.get("info") or self.info
        self.gen_s += float(res.get("gen_s") or 0.0)
        self.calls += 1
        self.requests += len(chunk)
        return rows

    def close(self) -> float:
        if self.meter is not None:
            self.meter.close(stopped=False)
            return self.meter.charged_usd
        return 0.0


def make_gen(session: Session, cond: Condition, iv, measure: str, adapter, batch: int, chunk_dir: Path,
             reuse_dir: Path | None = None, earlier: list[Path] | None = None):
    """The condition's generator for a measure: requests -> results, in chunks of batch, each chunk kept on disk
    (data/.../chunk_NNN.jsonl) and read back instead of asked again. A chunk in reuse_dir (the trial's) with the same
    requests is copied rather than generated; so is one in an earlier run's folders (earlier: --reuse-run). gen.counts
    says how many requests came each way."""
    counts = {"generated": 0, "from_trial": 0, "from_disk": 0, "from_earlier_run": 0}

    def gen(reqs: list[dict]) -> list[dict]:
        out = []
        for i, chunk in enumerate(chunks([prepare(r, cond, iv, measure, adapter) for r in reqs], batch)):
            name = f"chunk_{i:03d}.jsonl"
            rids = [r["rid"] for r in chunk]
            p = chunk_dir / name
            if p.exists():
                rows = _read_jsonl(p)
                if [r["rid"] for r in rows] != rids:
                    raise Refused(f"{p} holds other requests than this chunk (the spec or the code changed); plan a "
                                  "fresh run")
                out += rows
                counts["from_disk"] += len(rows)
                continue
            src = reuse_dir / name if reuse_dir else None
            old = next((d / name for d in (earlier or []) if (d / name).exists()
                        and [r["rid"] for r in _read_jsonl(d / name)] == rids), None)
            if src is not None and src.exists() and [r["rid"] for r in _read_jsonl(src)] == rids:
                rows = [{**r, "from_trial": True} for r in _read_jsonl(src)]
                counts["from_trial"] += len(rows)
            elif old is not None:
                rows = [{**r, "from_earlier_run": str(old)} for r in _read_jsonl(old)]
                counts["from_earlier_run"] += len(rows)
            else:
                rows = session.generate(chunk)
                counts["generated"] += len(rows)
            cards.write_text(chunk_dir, name, _jsonl(rows))
            out += rows
        return out

    gen.counts = counts
    return gen


def estimate(spec, items: dict, s_per_request: float | None = None) -> dict:
    """GPU: per (model, lora) session the load plus every request at s_per_request (a trial's measured rate, else
    each measure's S_PER_REQUEST), plus the tail; judge: each measure's calls at its cautious price, plus a served
    judge's fixed load cost (JUDGE_FIXED_USD). The total carries EST_MARGIN on the GPU part."""
    models = {c.model for c in spec.conditions}
    n_req = {(m, x): n_requests(M.get(m), items[m], x) for m in items for x in models}
    spr = {m: s_per_request if s_per_request is not None else spr_of(M.get(m)) for m in items}
    sessions = {(c.model, c.organism is not None) for c in spec.conditions}
    gen_s, gpu = 0.0, 0.0
    for model, lora in sessions:
        n_c = sum(1 for c in spec.conditions if (c.model, c.organism is not None) == (model, lora))
        s = LOAD_S + SCALEDOWN + n_c * sum(n_req[(m, model)] * spr[m] for m in items)
        gen_s += s
        gpu += s * gpu_rate(model_gpu(model)) / 3600
    judge = sum(sum(M.get(m).judge_calls(n_req[(m, c.model)]) for c in spec.conditions) * M.get(m).JUDGE_USD_PER_CALL
                + float(getattr(M.get(m), "JUDGE_FIXED_USD", 0.0)) + float(getattr(M.get(m), "CHECK_USD", 0.0))
                for m in items)
    one = next(iter(spr.values())) if len(set(spr.values())) == 1 else spr
    return {"conditions": len(spec.conditions), "requests": sum(n_req[(m, c.model)] for m in items for c in spec.conditions),
            "sessions": len(sessions), "s_per_request": one, "gen_s": round(gen_s, 1), "gen_h": round(gen_s / 3600, 2),
            "gpu_usd": round(gpu, 4), "judge_usd": round(judge, 4), "total_usd": round(EST_MARGIN * gpu + judge, 4)}


def load_items(spec) -> dict:
    return {m: M.get(m).load_items(opts or {}) for m, opts in spec.measures.items()}


def _interventions(spec) -> dict:
    out = {}
    for c in spec.conditions:
        try:
            out[c.id] = build_intervention(c.intervention)
        except (NotImplementedError, KeyError) as e:
            raise SpecError(str(e)) from None
    return out


def _gate(spec) -> dict:
    return gates.refuse_unless_ready(spec.sha256, spec.tier, f"{spec.name} ({spec.rel})")


def _open(spec_path, run_dir):
    spec = load_spec(spec_path)
    code = _gate(spec)
    out = gates.open_run_folder(spec.name, spec.sha256, run_dir)
    return spec, code, out


def _remaining(spec, out) -> float:
    left = round(spec.caps["usd"] - gates.run_spent(out), 4)
    if left <= 0:
        raise Refused(f"the run has spent its cap ${spec.caps['usd']:.2f} (caps.usd)")
    return left


def _check_cost(spec, out, usd: float, what: str) -> dict:
    left = _remaining(spec, out)
    if usd > left:
        raise Refused(f"{what}'s estimate ${usd:.2f} is past what is left of the run's cap (${left:.2f} of "
                      f"${spec.caps['usd']:.2f})")
    return {**gates.check_day(usd, what=what), "run_cap_left_usd": left}


def _verify(remote, key: str) -> dict:
    """The volume's re-hash of an organism, checked here against the registry's files (gates.check_verified)."""
    got = remote.verify_organisms([key])
    return gates.check_verified(key, (got or {}).get(key) if isinstance(got, dict) else got)


def _hash_referenced(spec) -> dict:
    from .spec import referenced_files

    out = {}
    for label, path in referenced_files(spec.reproduction).items():
        out[label] = {"path": str(path), "sha256": gates.sha256_file(path) if Path(path).exists() else None}
    missing = [k for k, v in out.items() if v["sha256"] is None and not k.startswith("reference.")]
    if missing:
        raise Refused(f"the reproduction names cards that do not exist yet: {missing} (e.g. an earlier run's card "
                      "this spec reads; commit it under results/cards/ first)")
    for k, v in ((spec.reproduction or {}).get("reference") or {}).items():
        got = out[f"reference.{k}"]["sha256"]
        if got != v["sha256"]:
            raise Refused(f"reproduction.reference.{k}: {v['file']} has sha256 {got}, the spec pins {v['sha256']}")
    return out


def _referenced_or_report(spec, dry_run: bool) -> dict:
    """_hash_referenced; a dry run reports a refusal instead of raising it."""
    try:
        return _hash_referenced(spec)
    except Refused as e:
        if not dry_run:
            raise
        return {"would_refuse": e.code}


def _reuse_source(spec, items_rec: dict, conds: list, reuse_run) -> dict:
    """The earlier run folder a re-grade copies generation from, after checking it ran the same requests."""
    src = Path(reuse_run)
    body = gates.plan_body(src)
    raw = body.get("spec_raw") or {}
    diff = [k for k in REUSE_SAME if raw.get(k) != spec.raw.get(k)]
    if (raw.get("caps") or {}).get("batch") != spec.caps["batch"]:
        diff.append("caps.batch")
    if body.get("items") != items_rec:
        diff.append("items")
    if [(c["id"], c["intervention"]) for c in body.get("conditions") or []] != [(c["id"], c["intervention"]) for c in conds]:
        diff.append("conditions")
    if diff:
        raise Refused(f"--reuse-run {src}: it differs from this spec in {diff}; only a run of the same requests is reused")
    return {"run": str(src), "plan_sha256": gates.sha256_file(src / gates.PLAN)}


def _earlier(out, kind: str, cond: Condition, mkey: str) -> list[Path]:
    """The earlier run's chunk folders for this condition and measure (plan's reuse_from), else none."""
    src = gates.plan_body(out).get("reuse_from")
    if not src:
        return []
    kinds = ("trial",) if kind == "trial" else ("gen", "trial")
    return [cond_dir(Path(src["run"]), k, cond, mkey) for k in kinds]


# ------------------------------------------------------------------------------------------------- stages


def plan(spec_path, dry_run: bool = False, reuse_run: str = "") -> tuple[Path | None, dict]:
    spec = load_spec(spec_path)
    ivs = _interventions(spec)
    try:
        code, gate = _gate(spec), "passed"
    except Refused as e:  # a dry run shows the plan of an unregistered full spec (so it can be checked before
        if not dry_run:   # registration) and says why every other stage will refuse; nothing is written
            raise
        from ..repro import git_state

        code, gate = git_state(), f"would refuse: {e.code}"
    items = load_items(spec)
    est = estimate(spec, items)
    if est["total_usd"] > spec.caps["usd"]:
        raise Refused(f"the estimate ${est['total_usd']:.2f} is past caps.usd ${spec.caps['usd']:.2f}")
    try:
        day = gates.check_day(est["total_usd"], what="this run")
    except Refused as e:  # a dry run reports it (the plan and every later stage refuse)
        if not dry_run:
            raise
        day = {"would_refuse": e.code}
    p = {"stage": "plan", "spec_file": spec.rel, "spec_file_sha256": spec.sha256, "spec_raw": spec.raw,
         "tier": spec.tier, "registration": ("required and found" if gate == "passed" else "required and NOT found")
         if spec.tier == "full" else
         "not required (tier explore); runs from a dirty tree allowed, recorded",
         "conditions": [{"id": c.id, "model": c.model, "organism": c.organism, "adapter": adapter_of(c.organism),
                         "intervention": ivs[c.id].record, "seed": c.seed} for c in spec.conditions],
         "items": {m: M.get(m).items_record(v) for m, v in items.items()},
         "calibration": {m: {model: M.calibration(m, model) for model in spec.models} for m in items},
         "estimate": est, "spend_at_plan": day, "caps": spec.caps, "gpu": {"type": GPU, "rate_usd_per_hour": gpu_rate()},
         "referenced_files": _referenced_or_report(spec, dry_run)}
    if reuse_run:
        p["reuse_from"] = _reuse_source(spec, p["items"], p["conditions"], reuse_run)
    if dry_run:
        return None, {**p, "code": code, "dry_run": True, "gates": gate}
    out = gates.new_run_folder(spec.name, p, code)
    if reuse_run and (Path(reuse_run) / "judge_cache").is_dir():
        shutil.copytree(Path(reuse_run) / "judge_cache", out / "judge_cache")  # the same calls answer from it, free
    return out, p


def _first_items(measure, items: list[dict], batch: int, model: str | None = None) -> list[dict]:
    """The longest prefix of items whose requests fit in one batch (at least one item)."""
    k = 1
    while k < len(items) and n_requests(measure, items[:k + 1], model) <= batch:
        k += 1
    return items[:k]


def trial(spec_path, run_dir="", remote=None, ask_factory=None) -> tuple[Path, dict]:  # noqa: C901
    from ..backends.registry import get as model_entry

    spec, code, out = _open(spec_path, run_dir)
    prev = gates.read_latest(out, "trial.json")
    if prev and prev.get("ok"):
        raise Refused(f"{out} already has an ok trial; go on with --stage run")
    cond, mkey = spec.conditions[0], next(iter(spec.measures))
    measure, iv = M.get(mkey), _interventions(spec)[cond.id]
    items = _first_items(measure, M.get(mkey).load_items(spec.measures[mkey] or {}), spec.caps["batch"], cond.model)
    n_req = n_requests(measure, items, cond.model)
    usd = EST_MARGIN * (LOAD_S + SCALEDOWN + n_req * spr_of(measure)) * gpu_rate(model_gpu(cond.model)) / 3600 + \
        TRIAL_JUDGE_N * getattr(measure, "JUDGE_USD_PER_CALL", 0.0) + float(getattr(measure, "JUDGE_FIXED_USD", 0.0))
    budget = _check_cost(spec, out, usd, "the trial")
    verified = _verify(remote, cond.organism) if cond.organism else {}
    session = Session(remote, cond.model, cond.organism is not None, budget["run_cap_left_usd"], spr_of(measure), out,
                      f"{out.name}-trial")
    checks: dict = {}
    try:
        gen = make_gen(session, cond, iv, mkey, adapter_of(cond.organism), spec.caps["batch"],
                       cond_dir(out, "trial", cond, mkey), earlier=_earlier(out, "trial", cond, mkey))
        t0 = time.time()
        units = run_measure(measure, gen, items, cond)
        wall = time.time() - t0
    finally:
        gpu_usd = session.close()
    info, spr, info_from = session.info or {}, (session.gen_s / session.requests if session.requests else None), None
    if not session.info and gen.counts["from_earlier_run"]:  # nothing generated: the earlier run's trial served it
        src = gates.plan_body(out)["reuse_from"]["run"]
        old = gates.read_latest(src, "trial.json") or {}
        info, spr, info_from = old.get("info") or {}, old.get("s_per_request"), f"{src}/trial.json"
    want = model_entry(cond.model).snapshot()
    checks["every_reply_back"] = len(units) == n_req  # units are per request (VLAF: 20 to 60 per item)
    checks["some_text"] = any((u.get("text") or "").strip() for u in units)
    checks["served_snapshot_is_registry"] = info.get("path") == want
    checks["load_check"] = bool(info.get("load_check"))
    checks["lora_as_planned"] = (info.get("settings") or {}).get("enable_lora") == (cond.organism is not None)
    judged = None
    if hasattr(measure, "judge"):
        ask = (ask_factory(out, _remaining(spec, out), measure) if ask_factory else
               _default_ask(out, _remaining(spec, out), measure, remote=remote, label=f"{out.name}-trial"))
        try:
            judged = measure.judge(units[:TRIAL_JUDGE_N], items, ask)
        finally:
            getattr(ask, "close", lambda: None)()
        errors = sum(bool(u.get("judge_error")) for u in judged)
        parsed = sum(getattr(measure, "parsed", lambda u: not u.get("judge_error"))(u) for u in judged)
        checks["judge_answers"] = errors == 0
        checks["judge_parses"] = parsed >= TRIAL_PARSE_MIN * len(judged)
    rec = {"stage": "trial", "ok": all(checks.values()), "checks": checks, "condition": cond.id, "measure": mkey,
           "batch": spec.caps["batch"], "n_requests": n_req, "gen_s": round(session.gen_s, 1), "wall_s": round(wall, 1),
           "s_per_request": spr, "truncated": sum(u.get("finish_reason") == "length" for u in units),
           "organisms_verified": verified, "info": info, "info_from": info_from, "chunks": gen.counts,
           "expected_snapshot": want, "gpu_usd": gpu_usd,
           "judged": None if judged is None else {"n": len(judged), "parsed": parsed, "errors": errors},
           "spend": budget, "code": code}
    p = cards.write_card(out, rec, name="trial.json", force=True, code=code)
    return p, rec


def _default_ask(out, cap: float, measure, remote=None, label: str = ""):
    """The measure's judge: OpenRouter (anatta.runner.judge.asker), or a registry model served on Modal through a
    metered Session when the measure says JUDGE_BACKEND = "vllm" (needs the remote; ask.close() books the session)."""
    from .judge import asker, vllm_asker

    if getattr(measure, "JUDGE_BACKEND", "openrouter") == "vllm":
        if remote is None:
            raise Refused("this measure's judge is served on Modal: run the stage through tools/modal.sh run "
                          "experiments/run.py::main ...")
        session = Session(remote, measure.JUDGE_MODEL_KEY, False, cap, float(getattr(measure, "JUDGE_S_PER_CALL", 0.2)),
                          out, label or f"{Path(out).name}-judge", tool=TOOL_JUDGE_GPU, log=gates.JUDGE_GPU_LOG)
        return vllm_asker(session, batch=int(getattr(measure, "JUDGE_BATCH", 1000)))
    return asker(out, cap, getattr(measure, "JUDGE_MODELS", {}))


def run(spec_path, run_dir="", remote=None) -> tuple[Path, dict]:
    spec, code, out = _open(spec_path, run_dir)
    t = gates.read_latest(out, "trial.json")
    if not t or not t.get("ok"):
        raise Refused(f"{out} has no ok trial at the real batch (run --stage trial first)"
                      + (f"; the last trial's checks: {t['checks']}" if t else ""))
    spr = max(float(t.get("s_per_request") or spr_of(M.get(next(iter(spec.measures))))), 1e-3)
    items = load_items(spec)
    est = estimate(spec, items, spr)
    budget = _check_cost(spec, out, EST_MARGIN * est["gpu_usd"], "the run")
    ivs = _interventions(spec)
    prev = gates.read_latest(out, "run.json") or {}
    done = dict(prev.get("conditions") or {})
    gpu = 0.0
    groups: dict = {}
    for c in spec.conditions:
        groups.setdefault((c.model, c.organism is not None), []).append(c)
    first = spec.conditions[0]
    try:
        for (model, lora), conds in groups.items():
            session = Session(remote, model, lora, _remaining(spec, out), spr, out, f"{out.name}-run")
            try:
                for c in conds:
                    verified = _verify(remote, c.organism) if c.organism else {}
                    for mkey in spec.measures:
                        target = cond_dir(out, "units", c, mkey).with_suffix(".jsonl")
                        if target.exists():
                            continue
                        reuse = cond_dir(out, "trial", c, mkey) if c == first else None
                        gen = make_gen(session, c, ivs[c.id], mkey, adapter_of(c.organism), spec.caps["batch"],
                                       cond_dir(out, "gen", c, mkey), reuse, earlier=_earlier(out, "gen", c, mkey))
                        units = run_measure(M.get(mkey), gen, items[mkey], c)
                        cards.write_text(target.parent, target.name, _jsonl(units))
                        done.setdefault(c.id, {})[mkey] = {
                            "n": len(units), "truncated": sum(u.get("finish_reason") == "length" for u in units),
                            **gen.counts}
                    if verified:
                        done.setdefault(c.id, {})["organisms_verified"] = verified
            finally:
                gpu += session.close()
                info, key = session.info, f"{model}|{'lora' if lora else 'base'}"
                src = gates.plan_body(out).get("reuse_from")
                if not info and src:  # nothing generated here: what the earlier run served
                    info = ((gates.read_latest(src["run"], "run.json") or {}).get("conditions") or {}).get(
                        "_served", {}).get(key)
                if info:
                    done.setdefault("_served", {})[key] = info
    finally:
        rec = {"stage": "run", "conditions": done, "gpu_usd_this_call": round(gpu, 6), "estimate": est,
               "spend": budget, "complete": all(cond_dir(out, "units", c, m).with_suffix(".jsonl").exists()
                                                for c in spec.conditions for m in spec.measures), "code": code}
        p = cards.write_card(out, rec, name="run.json", force=True, code=code)
    return p, rec


def score(spec_path, run_dir="", ask_factory=None, remote=None) -> tuple[Path, dict]:
    spec, code, out = _open(spec_path, run_dir)
    items = load_items(spec)
    missing = [f"{c.id}/{m}" for c in spec.conditions for m in spec.measures
               if not cond_dir(out, "units", c, m).with_suffix(".jsonl").exists()]
    if missing:
        raise Refused(f"no units yet for {missing} (run --stage run)")
    calls = sum(sum(M.get(m).judge_calls(n_requests(M.get(m), items[m], c.model)) for c in spec.conditions)
                * M.get(m).JUDGE_USD_PER_CALL + float(getattr(M.get(m), "JUDGE_FIXED_USD", 0.0))
                + float(getattr(M.get(m), "CHECK_USD", 0.0))
                for m in spec.measures if hasattr(M.get(m), "judge"))
    budget = _check_cost(spec, out, calls, "the judging")
    scores: dict = {}
    asks: dict = {}  # one judge per measure for the whole stage (a served judge loads once)
    try:
        _score_all(spec, out, items, scores, asks, ask_factory, remote)
    finally:
        for a in asks.values():
            getattr(a, "close", lambda: None)()
    complete = all("incomplete" not in m for c in scores.values() for m in c.values())
    checks = _judge_checks(spec, out, items, ask_factory) if complete else {}
    rec = {"stage": "score", "complete": complete, "scores": scores, "judge_checks": checks,
           "judge_usd": round(spend.sweep_spent(out / gates.JUDGE_LOG), 6),
           "judge_gpu_usd": round(spend.sweep_spent(out / gates.JUDGE_GPU_LOG), 6), "spend": budget, "code": code}
    p = cards.write_card(out, rec, name="score.json", force=True, code=code)
    return p, rec


def _judge_checks(spec, out, items, ask_factory) -> dict:
    """A measure's reported second-judge check (its check(cond_units, items, ask, salt)), through the OpenRouter
    judge (the run folder's cache and spend log): each condition's units (with text) joined to its verdicts."""
    from .judge import asker

    res = {}
    for mkey in spec.measures:
        measure = M.get(mkey)
        if not hasattr(measure, "check"):
            continue
        cond_units = {}
        for c in spec.conditions:
            units = {u["rid"]: u for u in _read_jsonl(cond_dir(out, "units", c, mkey).with_suffix(".jsonl"))}
            judged = _read_jsonl(cond_dir(out, "judged", c, mkey).with_suffix(".jsonl"))
            cond_units[c.id] = [{**units[j["rid"]], "choice": j.get("choice")} for j in judged]
        ask = (ask_factory(out, _remaining(spec, out), measure) if ask_factory else
               asker(out, _remaining(spec, out), getattr(measure, "CHECK_MODELS", {})))
        res[mkey] = measure.check(cond_units, items[mkey], ask, salt=spec.name)
    return res


def _score_all(spec, out, items, scores, asks, ask_factory, remote) -> None:
    for c in spec.conditions:
        for mkey in spec.measures:
            measure = M.get(mkey)
            units = _read_jsonl(cond_dir(out, "units", c, mkey).with_suffix(".jsonl"))
            jp = cond_dir(out, "judged", c, mkey).with_suffix(".jsonl")
            if hasattr(measure, "judge"):
                if jp.exists():
                    units = _read_jsonl(jp)
                else:
                    if mkey not in asks:
                        asks[mkey] = (ask_factory(out, _remaining(spec, out), measure) if ask_factory else
                                      _default_ask(out, _remaining(spec, out), measure, remote=remote,
                                                   label=f"{out.name}-judge"))
                    units = measure.judge(units, items[mkey], asks[mkey])
                    if any(u.get("judge_error") for u in units):
                        scores.setdefault(c.id, {})[mkey] = {"incomplete": sum(bool(u.get("judge_error")) for u in units)}
                        continue
                    cards.write_text(jp.parent, jp.name, _jsonl({k: v for k, v in u.items() if k != "text"}
                                                                for u in units))
            scores.setdefault(c.id, {})[mkey] = measure.score(units)


def grade(rep: dict, observed, reference: dict | None = None) -> dict:
    """in_band when the observed statistic is within the band (ends included). reference: same_item()'s record, for
    a target or band written as "reference"."""
    from .spec import resolve_band, resolve_value

    lo, hi = tuple(reference["ci95"]) if rep["band"] == "reference" else resolve_band(rep["band"])
    target = reference["rate"] if rep["target"] == "reference" else resolve_value(rep["target"])
    ok = observed is not None and not (isinstance(observed, float) and math.isnan(observed)) and lo <= observed <= hi
    return {"measure": rep["measure"], "statistic": rep["statistic"], "observed": observed, "target": target,
            "band": [lo, hi], "verdict": "in_band" if ok else "outside_band",
            "target_from": rep["target"] if isinstance(rep["target"], (dict, str)) else None,
            "band_from": rep["band"] if isinstance(rep["band"], (dict, str)) else None}


def grade_many(rep: dict, numbers: dict) -> dict:
    """A reproduction written as grades (spec.py): each entry's observed value (the condition's statistic, minus the
    `minus` condition's when given), its band verdict, and all_in_band; `reported` entries are shown, never graded.
    numbers: {condition id: {measure: score without units}}."""
    from .spec import resolve_band, resolve_value

    def observed(g):
        def one(cid):
            v = (numbers.get(cid) or {}).get(rep["measure"], {}).get(g["statistic"])
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else v
        a = one(g["condition"])
        if g.get("minus") is None or a is None:
            return a
        b = one(g["minus"])
        return None if b is None else round(a - b, 6)

    rows, shown = [], []
    for g in rep["grades"]:
        lo, hi = resolve_band(g["band"])
        obs = observed(g)
        rows.append({"name": g["name"], "condition": g["condition"], "minus": g.get("minus"),
                     "statistic": g["statistic"], "observed": obs,
                     "target": resolve_value(g["target"]) if g.get("target") is not None else None, "band": [lo, hi],
                     "verdict": "in_band" if obs is not None and lo <= obs <= hi else "outside_band"})
    for g in rep.get("reported") or []:
        obs, tgt = observed(g), resolve_value(g["target"]) if g.get("target") is not None else None
        ratio = (round(obs / tgt, 4) if isinstance(obs, (int, float)) and isinstance(tgt, (int, float)) and tgt
                 else None)
        shown.append({"name": g["name"], "condition": g["condition"], "minus": g.get("minus"),
                      "statistic": g["statistic"], "observed": obs, "target": tgt, "ratio_to_target": ratio,
                      "target_from": g["target"] if isinstance(g.get("target"), dict) else None,
                      "note": g.get("note")})
    return {"measure": rep["measure"], "grades": rows, "reported_ungraded": shown,
            "all_in_band": all(r["verdict"] == "in_band" for r in rows) if rows else None,
            "verdict": ("reported_only" if not rows else
                        "in_band" if all(r["verdict"] == "in_band" for r in rows) else "outside_band"),
            "decision": rep.get("decision")}


def same_item(rep: dict, measure, ours: dict, texts: dict | None = None) -> dict:
    """The stored reference run on exactly these units: its rate and Wilson 95% interval, the paired discordance
    (units rejected/positive in one run only) and, with stored replies, how many replies are byte-identical.
    ours: {unit id: 0/1} (score's units); texts: {unit id: reply} (this run's)."""
    from ..stats import wilson

    ref_cfg = rep["reference"]
    ids = sorted(ours, key=str)
    theirs = {str(k): v for k, v in measure.reference_units(ref_cfg["verdicts"]["file"], [int(i) for i in ids]).items()}
    mine = {str(k): int(v) for k, v in ours.items()}
    n, k = len(ids), sum(theirs[str(i)] for i in ids)
    out = {"verdicts_file": ref_cfg["verdicts"]["file"], "verdicts_sha256": ref_cfg["verdicts"]["sha256"], "n": n,
           "k": k, "rate": k / n if n else None, "ci95": wilson(k, n, clip=True),
           "paired": {"reference_only": sum(theirs[str(i)] == 1 and mine[str(i)] == 0 for i in ids),
                      "runner_only": sum(theirs[str(i)] == 0 and mine[str(i)] == 1 for i in ids),
                      "both": sum(theirs[str(i)] == 1 and mine[str(i)] == 1 for i in ids),
                      "neither": sum(theirs[str(i)] == 0 and mine[str(i)] == 0 for i in ids)}}
    a, b = out["paired"]["reference_only"], out["paired"]["runner_only"]
    out["paired"]["agreement"] = (n - a - b) / n if n else None
    out["paired"]["mcnemar_exact_p"] = _binom_two_sided(min(a, b), a + b)
    if "replies" in ref_cfg and texts is not None:
        stored = {str(k): v for k, v in measure.reference_replies(ref_cfg["replies"]["file"], [int(i) for i in ids]).items()}
        out["replies"] = {"file": ref_cfg["replies"]["file"], "sha256": ref_cfg["replies"]["sha256"],
                          "byte_identical": sum(stored.get(str(i)) == texts.get(str(i)) for i in ids), "n": n}
    out["note"] = BATCH_NOTE
    return out


def _binom_two_sided(k: int, n: int) -> float | None:
    """Exact two-sided McNemar p: 2 x P(X <= k), X ~ Binomial(n, 1/2), capped at 1."""
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def card(spec_path, run_dir="") -> tuple[Path, dict]:
    spec, code, out = _open(spec_path, run_dir)
    sc = gates.read_latest(out, "score.json")
    if not sc or not sc.get("complete"):
        raise Refused(f"{out} has no complete score.json (run --stage score)")
    plan_ = gates.plan_body(out)
    code_at_plan = gates.read_plan(out).get("code")
    now = _hash_referenced(spec)
    changed = {k: v for k, v in now.items() if (plan_.get("referenced_files") or {}).get(k) != v}
    if changed:
        raise Refused(f"files the reproduction names changed since the plan (or were not hashed into it): "
                      f"{sorted(changed)}; plan a fresh run")
    trial_, run_ = gates.read_latest(out, "trial.json"), gates.read_latest(out, "run.json")
    numbers = {cid: {m: {k: v for k, v in s.items() if k != "units"} for m, s in ms.items()}
               for cid, ms in sc["scores"].items()}
    c = {"card": spec.name, "spec": spec.rel, "spec_sha256": spec.sha256, "tier": spec.tier,
         "kind": "explore run: a sighting, not a finding" if spec.tier == "explore" else "registered run",
         "conditions": plan_["conditions"], "items": plan_["items"], "numbers": numbers,
         "calibration": plan_["calibration"], "judge_checks": sc.get("judge_checks") or {},
         "trial": {k: trial_.get(k) for k in ("ok", "checks", "batch", "n_requests", "gen_s", "s_per_request",
                                              "truncated")},
         "served": (run_ or {}).get("conditions", {}).get("_served"),
         "cost_usd": {"gpu": round(gates.run_spent(out) - sc["judge_usd"] - sc.get("judge_gpu_usd", 0.0), 4),
                      "judge": round(sc["judge_usd"] + sc.get("judge_gpu_usd", 0.0), 4),
                      "total": round(gates.run_spent(out), 4), "estimate_at_plan": plan_["estimate"]["total_usd"],
                      "cap": spec.caps["usd"]},
         "referenced_files": now, "reuse_from": plan_.get("reuse_from"),
         "chunks": {cid: {m: {k: v for k, v in r.items() if k in ("generated", "from_trial", "from_disk",
                                                                   "from_earlier_run")} for m, r in ms.items()
                          if m in spec.measures} for cid, ms in ((run_ or {}).get("conditions") or {}).items()
                    if cid != "_served"},
         "code_at_plan": code_at_plan, "code_at_card": code}
    rep = spec.reproduction
    if rep and "grades" in rep:
        c["reproduction"] = grade_many(rep, numbers)
    elif rep:
        cid = rep.get("condition") or spec.conditions[0].id
        ref = None
        if rep.get("reference"):
            cond = next(x for x in spec.conditions if x.id == cid)
            units_file = cond_dir(out, "units", cond, rep["measure"]).with_suffix(".jsonl")
            texts = {str(u["id"]): u.get("text") for u in _read_jsonl(units_file)} if units_file.exists() else None
            ref = same_item(rep, M.get(rep["measure"]), sc["scores"][cid][rep["measure"]]["units"], texts)
        c["reproduction"] = {"condition": cid, **grade(rep, numbers[cid][rep["measure"]].get(rep["statistic"]), ref)}
        if ref:
            c["reproduction"]["same_item_reference"] = ref
        if rep.get("full_set"):
            fs = grade({**rep, **rep["full_set"]}, numbers[cid][rep["measure"]].get(rep["statistic"]))
            c["reproduction"]["full_set_ungraded"] = {k: fs[k] for k in ("target", "band", "verdict", "target_from",
                                                                         "band_from")}
    p = cards.write_card(out, c, code=code)
    return p, c


STAGES = ("plan", "trial", "run", "score", "card")
