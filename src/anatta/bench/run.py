"""The capability battery's evaluation, as the battery runs it on a GPU, with every reply kept and the silent errors
made loud.

Generation is the battery's: one user turn per item (after an optional system message), anatta.probe.generate
(the model's chat template with add_generation_prompt, the text tokenized with add_special_tokens=False and left
padding, greedy: do_sample=False, the model's other generation_config fields as they are), the items of a job
sorted by prompt length and cut into batches in that order. probe.generate returns text only, so the model is
handed to it inside a Recorder, which forwards every attribute and keeps the output ids: each reply is stored whole
with its new-token count (up to and including the first end-of-sequence token) and hit_cap (no end-of-sequence
token: the reply was cut at max_new_tokens).

Formats (the prompt is the item's, built by anatta.bench.items; MAX_NEW new tokens unless a job says otherwise):
  gsm8k          512    anatta.bench.battery.score: the last "Answer: <number>", else the last number
  mmlu_pro        24    anatta.bench.battery.score: the letter alone, "the answer is X", or a leading "(X)"/"X."
  arc_challenge   24    the same
  mmlu_pro_cot  2048    Inspect's parse_answers (anatta.bench.inspect_choice.parse_letter); a reply without a
                        parseable ANSWER letter is wrong. Beside it, reported and never graded, TIGER-Lab's
                        fallbacks: when Inspect's parse finds nothing, "answer is (X)", then "Answer: X"
                        (TIGER_PATTERNS); tiger_pred / tiger_correct

The checks, which stop a run (CheckError):
  - before anything loads (run_plan): the container's package files and installed versions are the plan's
    (check_environment), and the run folder holds no other plan's files;
  - every prompt: the rendered template tokenized as probe.generate tokenizes it equals
    apply_chat_template(..., tokenize=True, add_generation_prompt=True), and starts with exactly the expected
    number of BOS tokens (check_prompt; check_all runs it on every prompt before anything is generated). Under
    transformers 5.21 the equality holds by construction (apply_chat_template tokenizes its own rendering without
    special tokens, as probe.generate does); the BOS count can fail, and so can the next check;
  - every batch, just before generation: the rows of input_ids handed to model.generate are left-padded with the
    pad token and, unpadded, equal those template ids (check_batch): this catches a tokenizer call that adds
    special tokens (a second BOS), right padding or a wrong pad id;
  - the first batch of a run, generated twice, gives at least the required share of identical replies
    (repeat_first_batch, in run_jobs). There is no override.
Reported, not checked: a subset run at batch 1 against the same items' replies at the run's batch size (batch_one).

run_jobs runs a list of jobs on a loaded model and writes, in out_dir: checks.json, determinism.json, one JOB.json
per job ({name, format, draw, batch, max_new, system, n, seconds, generate_kwargs, rows}), batch1.json; a job's
batches go to JOB.partial.jsonl as they finish, so a restarted run resumes at the first missing batch (a last line
cut short by a crash is dropped, with a warning). Each job's generate arguments also go into run_info.json at its
first generated batch, so a job resumed with every batch already done still reports them. run_plan wraps all this
for one planned run in a container.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from .. import probe
from .battery import score as battery_score
from .fingerprint import code_hashes, installed, plan_id
from .inspect_choice import parse_letter

MAX_NEW = {"gsm8k": 512, "mmlu_pro": 24, "arc_challenge": 24, "mmlu_pro_cot": 2048}
# TIGER-Lab's MMLU-Pro answer extraction (github.com/TIGER-AI-Lab/MMLU-Pro, evaluate_from_local.py at f418b11):
# extract_answer's pattern, then extract_again's, each with re.search. Their last step (the last standalone letter)
# and the random guess for a reply with no letter are not used.
TIGER_PATTERNS = (r"answer is \(?([A-J])\)?", r".*[aA]nswer:\s*([A-J])")


class CheckError(RuntimeError):
    """A silent error made loud: the run stops here."""


# ------------------------------------------------------------------------------------------ prompts and scores


def prompt_of(item: dict, fmt: str) -> str:
    """The item's prompt in a format: item["prompts"][fmt] (the item file), or item["prompt"] (the battery's own
    items, which have one format each)."""
    return item["prompts"][fmt] if "prompts" in item else item["prompt"]


def messages(prompt: str, system: str | None = None) -> list[dict]:
    """One user turn, after a system message when `system` is not None ("" is an empty system message; None
    leaves the slot to the chat template, which may fill in its own default)."""
    return ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": prompt}]


def tiger_letter(text: str) -> str | None:
    for pattern in TIGER_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def score_reply(fmt: str, item: dict, text: str) -> dict:
    """{"correct", "pred"} for one reply; mmlu_pro_cot adds the TIGER-Lab fallback's "tiger_pred" and "tiger_correct"."""
    if fmt == "mmlu_pro_cot":
        pred = parse_letter(text, item["n_options"])
        fallback = pred if pred is not None else tiger_letter(text)
        return {"correct": pred == item["gold"], "pred": pred, "tiger_pred": fallback,
                "tiger_correct": fallback == item["gold"]}
    if fmt not in MAX_NEW:
        raise ValueError(f"unknown format {fmt}")
    ok, pred = battery_score({"set": "gsm8k" if fmt == "gsm8k" else item["set"], "gold": item["gold"]}, text)
    return {"correct": ok, "pred": pred}


def batches(items: list[dict], fmt: str, size: int) -> list[list[dict]]:
    """The battery's batching: sorted by prompt length (a stable sort, so ties keep the given order), then cut in
    order into batches of `size`."""
    order = sorted(items, key=lambda x: len(prompt_of(x, fmt)))  # similar lengths per batch, less padding
    return [order[s:s + size] for s in range(0, len(order), size)]


# ------------------------------------------------------------------------------------------------- the checks


def template_ids(tok, conversation: list[dict]) -> list[int]:
    """apply_chat_template(conversation, tokenize=True, add_generation_prompt=True) as a flat list of ids (newer
    transformers return a BatchEncoding)."""
    out = tok.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
    if isinstance(out, Mapping) or hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    out = list(out)
    if out and isinstance(out[0], list):
        if len(out) != 1:
            raise CheckError(f"apply_chat_template returned {len(out)} sequences for one conversation")
        out = out[0]
    return [int(t) for t in out]


def leading(ids: list[int], token: int | None) -> int:
    """How many times `token` repeats at the start of ids (0 when the tokenizer has no such token)."""
    n = 0
    while token is not None and n < len(ids) and ids[n] == token:
        n += 1
    return n


def first_difference(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def check_prompt(tok, conversation: list[dict], bos: int | None = None) -> tuple[str, list[int]]:
    """The prompt assertions on one conversation: the rendered template, tokenized with add_special_tokens=False as
    probe.generate tokenizes it, equals apply_chat_template(..., tokenize=True, add_generation_prompt=True), and
    starts with exactly `bos` BOS tokens (None: not counted). Returns (rendered text, ids)."""
    text = tok.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
    want = template_ids(tok, conversation)
    if ids != want:
        raise CheckError(f"the rendered prompt tokenizes to {len(ids)} ids and apply_chat_template(tokenize=True) to "
                         f"{len(want)}; they first differ at position {first_difference(ids, want)}")
    if bos is not None and leading(ids, tok.bos_token_id) != bos:
        raise CheckError(f"the prompt starts with {leading(ids, tok.bos_token_id)} BOS token(s), not {bos} "
                         f"(bos id {tok.bos_token_id}; first ids {ids[:6]})")
    return text, ids


def check_batch(kwargs: dict, expect: list[list[int]], pad_id: int | None) -> None:
    """The ids about to be generated from: one row per expected prompt, left-padded with the pad token, and equal,
    once unpadded, to the template's ids."""
    ids, mask = kwargs["input_ids"].tolist(), kwargs["attention_mask"].tolist()
    if len(ids) != len(expect):
        raise CheckError(f"{len(ids)} rows go to generate for {len(expect)} prompts")
    for r, (row, m, want) in enumerate(zip(ids, mask, expect)):
        n_pad = m.index(1) if 1 in m else len(m)
        if m != [0] * n_pad + [1] * (len(m) - n_pad):
            raise CheckError(f"row {r} of the batch is not left-padded")
        if any(t != pad_id for t in row[:n_pad]):
            raise CheckError(f"row {r} is padded with something other than the pad token {pad_id}")
        if row[n_pad:] != want:
            raise CheckError(f"row {r}: the ids sent to generate differ from apply_chat_template's at position "
                             f"{first_difference(row[n_pad:], want)} ({len(row) - n_pad} ids against {len(want)})")


def check_all(tok, jobs: list[dict], items: dict[str, dict], system: str | None = None,
              bos: int | None = None) -> dict:
    """check_prompt on every prompt of every job, before anything is generated. Returns per job the number of
    prompts, their length range in tokens, and item 0's rendered prompt with its first 40 ids."""
    out = {}
    for job in jobs:
        lengths, item0 = [], None
        for n, i in enumerate(job["ids"]):
            text, ids = check_prompt(tok, messages(prompt_of(items[i], job["format"]), system), bos)
            lengths.append(len(ids))
            if n == 0:
                item0 = {"id": i, "rendered": text, "first_ids": ids[:40], "n_ids": len(ids)}
        out[job["name"]] = {"prompts": len(lengths), "tokens_min": min(lengths, default=0),
                            "tokens_max": max(lengths, default=0), "bos": bos, "item0": item0}
    return out


# --------------------------------------------------------------------------------------------- generation


class Recorder:
    """The model as probe.generate sees it: every attribute is the model's, and generate first checks the batch it
    is about to run on (check_batch against `expect`, when set), then keeps (kwargs, output ids) in `last`, since
    probe.generate returns text only."""

    def __init__(self, model, pad_id: int | None):
        self.model, self.pad_id = model, pad_id
        self.expect: list[list[int]] | None = None
        self.last: tuple[dict, object] | None = None

    def __getattr__(self, name):
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    def generate(self, **kwargs):
        if self.expect is not None:
            check_batch(kwargs, self.expect, self.pad_id)
        out = self.model.generate(**kwargs)
        self.last = (kwargs, out)
        return out


def eos_ids(model, tok) -> set[int]:
    """The ids generation stops on: generation_config.eos_token_id (one id or a list), else the tokenizer's."""
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = tok.eos_token_id
    return {int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]) if e is not None}


def reply_length(new_ids: list[int], eos: set[int]) -> tuple[int, bool]:
    """(new tokens up to and including the first end-of-sequence token, hit_cap). A row with no end-of-sequence
    token was cut at the limit: generate pads only the rows that have finished."""
    for i, t in enumerate(new_ids):
        if t in eos:
            return i + 1, False
    return len(new_ids), True


def evaluate_items(model, tok, items: list[dict], fmt: str, batch_size: int = 50, system: str | None = None,
                   max_new_tokens: int | None = None, bos: int | None = None, done: dict | None = None,
                   on_batch: Callable | None = None, seen: dict | None = None) -> list[dict]:
    """The battery's generation over `items` in one format: batches (sorted by prompt length), each checked
    (check_prompt per prompt, check_batch on the ids) and generated by probe.generate(model, tok, conversations,
    max_new_tokens, temperature=0.0), the call the battery makes. Returns one row per item in batch order:
    {set, id, correct, pred, [tiger_pred, tiger_correct], gold, text, new_tokens, hit_cap, prompt_tokens}.
    `done` ({batch index: {"ids", "rows"}}) holds batches already run, which are reused, not regenerated;
    on_batch(index, ids, rows) is called after each new batch; `seen` gets the first new batch's generate
    arguments other than the ids ("generate_kwargs")."""
    max_new = max_new_tokens or MAX_NEW[fmt]
    rec, eos = Recorder(model, tok.pad_token_id), eos_ids(model, tok)
    rows = []
    for b, part in enumerate(batches(items, fmt, batch_size)):
        ids = [x["id"] for x in part]
        if done and b in done:
            if done[b]["ids"] != ids:
                raise CheckError(f"the checkpoint's batch {b} holds other items than this run's batch {b}")
            rows += done[b]["rows"]
            continue
        conversations = [messages(prompt_of(x, fmt), system) for x in part]
        rec.expect = [check_prompt(tok, c, bos)[1] for c in conversations]
        texts = probe.generate(rec, tok, conversations, max_new, temperature=0.0)
        kwargs, out = rec.last
        new = out[:, kwargs["input_ids"].shape[1]:].tolist()
        if seen is not None and "generate_kwargs" not in seen:
            seen["generate_kwargs"] = {k: v for k, v in kwargs.items() if k not in ("input_ids", "attention_mask")}
        part_rows = []
        for x, text, prompt_ids, gen in zip(part, texts, rec.expect, new):
            n, cap = reply_length(gen, eos)
            part_rows.append({"set": x["set"], "id": x["id"], **score_reply(fmt, x, text), "gold": x["gold"],
                              "text": text, "new_tokens": n, "hit_cap": cap, "prompt_tokens": len(prompt_ids)})
        rows += part_rows
        if on_batch is not None:
            on_batch(b, ids, part_rows)
    return rows


def repeat_first_batch(model, tok, items: list[dict], fmt: str, batch_size: int, system: str | None = None,
                       max_new_tokens: int | None = None, bos: int | None = None) -> dict:
    """The run's first batch (the batch_size shortest prompts, as the run cuts them) generated twice: how many of
    its replies come out identical."""
    first = batches(items, fmt, batch_size)[0]
    a = evaluate_items(model, tok, first, fmt, len(first), system, max_new_tokens, bos)
    b = evaluate_items(model, tok, first, fmt, len(first), system, max_new_tokens, bos)
    same = sum(x["text"] == y["text"] for x, y in zip(a, b))
    return {"format": fmt, "batch": len(first), "ids": [x["id"] for x in first], "n": len(first), "identical": same,
            "share": same / len(first), "texts": [[x["text"], y["text"]] for x, y in zip(a, b)]}


def batch_one(model, tok, items: list[dict], fmt: str, reference: list[dict], system: str | None = None,
              max_new_tokens: int | None = None, bos: int | None = None) -> dict:
    """`items` generated one at a time, against the same items' rows from the run at its own batch size
    (`reference`): the share of identical replies and the accuracy gap in points (batch 1 minus batch N)."""
    rows = evaluate_items(model, tok, items, fmt, 1, system, max_new_tokens, bos)
    ref = {r["id"]: r for r in reference}
    pairs = [(r, ref[r["id"]]) for r in rows]
    same = sum(r["text"] == q["text"] for r, q in pairs)
    acc1 = 100 * sum(r["correct"] for r, _ in pairs) / len(pairs)
    acc_n = 100 * sum(q["correct"] for _, q in pairs) / len(pairs)
    return {"format": fmt, "n": len(pairs), "identical": same, "share": same / len(pairs), "acc_batch1": acc1,
            "acc_batch_n": acc_n, "gap_points": acc1 - acc_n, "rows": rows}


# ---------------------------------------------------------------------------------------------------- the jobs


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def append_jsonl(path: Path, row: dict) -> None:
    """One record per open and close, so the volume never holds a long-lived open file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path, repair: bool = False) -> list[dict]:
    """A JSONL file's records. A malformed last line (a write cut short) is skipped with a printed warning and, with
    `repair`, removed from the file; a malformed line anywhere else raises."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            if k < len(lines) - 1:
                raise ValueError(f"{path}: line {k + 1} of {len(lines)} is not JSON ({e})") from e
            print(f"warning: {path}: its last line is not JSON ({e}); skipped" + (" and removed" if repair else ""),
                  flush=True)
            if repair:
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def remember_kwargs(out: Path, job: str, kwargs: dict) -> None:
    """A job's generate arguments, kept in run_info.json from its first generated batch on, so a job whose batches
    were all generated before a restart still reports them."""
    path = out / "run_info.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    if job not in info.setdefault("generate_kwargs", {}):
        info["generate_kwargs"][job] = kwargs
        write_json(path, info)


def recalled_kwargs(out: Path, job: str) -> dict | None:
    path = out / "run_info.json"
    return (json.loads(path.read_text()).get("generate_kwargs") or {}).get(job) if path.exists() else None


def run_job(model, tok, job: dict, items: dict[str, dict], out: Path, system: str | None = None,
            bos: int | None = None, commit: Callable = lambda: None, log: Callable = print) -> dict:
    """One job ({name, format, draw, ids, batch, max_new}), resumed from its partial file if there is one; returns
    and writes out/NAME.json. An existing NAME.json is returned as it is."""
    final, partial = out / f"{job['name']}.json", out / f"{job['name']}.partial.jsonl"
    if final.exists():
        return json.loads(final.read_text())
    done = {r["batch"]: r for r in read_jsonl(partial, repair=True)}
    seen: dict = {}
    t0 = time.time()

    remembered = False

    def on_batch(b, ids, rows):
        nonlocal remembered
        append_jsonl(partial, {"batch": b, "ids": ids, "rows": rows})
        if not remembered:  # the first batch this container generated for the job
            remember_kwargs(out, job["name"], seen["generate_kwargs"])
            remembered = True
        commit()

    rows = evaluate_items(model, tok, [items[i] for i in job["ids"]], job["format"], job["batch"], system,
                          job["max_new"], bos, done=done, on_batch=on_batch, seen=seen)
    res = {**{k: job[k] for k in ("name", "format", "draw", "batch", "max_new") if k in job}, "system": system,
           "n": len(rows), "seconds": round(time.time() - t0), "resumed_batches": len(done),
           "generate_kwargs": seen.get("generate_kwargs") or recalled_kwargs(out, job["name"]), "rows": rows}
    write_json(final, res)
    partial.unlink(missing_ok=True)
    commit()
    k = sum(r["correct"] for r in rows)
    log(f"{job['name']}: {k}/{len(rows)} correct ({100 * k / max(len(rows), 1):.1f}), "
        f"{sum(r['hit_cap'] for r in rows)} cut at {job['max_new']} tokens, {res['seconds']}s")
    return res


def run_jobs(model, tok, jobs: list[dict], items: dict[str, dict], out_dir, system: str | None = None,
             bos: int | None = None, determinism: dict | None = None, batch1: dict | None = None,
             commit: Callable = lambda: None, log: Callable = print) -> dict:
    """A run on a loaded model: check_all on every prompt; the first batch of determinism["job"] twice, stopping
    the run if fewer than determinism["min_share"] of its replies match; every job (run_job); then, for batch1
    ({job name: item ids}), those items at batch 1 against the job's rows. Files already in out_dir are reused,
    so a restarted run picks up where it stopped. Returns {checks, determinism, jobs, batch1}."""
    out = Path(out_dir)
    by_name = {j["name"]: j for j in jobs}
    checks = check_all(tok, jobs, items, system, bos)
    write_json(out / "checks.json", checks)
    commit()
    det = None
    if determinism:
        path = out / "determinism.json"
        if path.exists():
            det = json.loads(path.read_text())
        else:
            job = by_name[determinism["job"]]
            det = repeat_first_batch(model, tok, [items[i] for i in job["ids"]], job["format"], job["batch"], system,
                                     job["max_new"], bos)
            det.update(job=job["name"], min_share=determinism["min_share"],
                       passed=det["share"] >= determinism["min_share"])
            write_json(path, det)
            commit()
        log(f"first batch twice: {det['identical']}/{det['n']} identical")
        if not det["passed"]:
            raise CheckError(f"the first batch, generated twice, gave {det['identical']} of {det['n']} identical "
                             f"replies, under the {determinism['min_share']:.0%} required: is sampling on?")
    results = {j["name"]: run_job(model, tok, j, items, out, system, bos, commit, log) for j in jobs}
    b1 = None
    if batch1:
        path = out / "batch1.json"
        if path.exists():
            b1 = json.loads(path.read_text())
        else:
            b1 = {name: batch_one(model, tok, [items[i] for i in ids], by_name[name]["format"],
                                  results[name]["rows"], system, by_name[name]["max_new"], bos)
                  for name, ids in batch1.items()}
            write_json(path, b1)
            commit()
        for name, r in b1.items():
            log(f"{name} at batch 1: {r['identical']}/{r['n']} identical, accuracy gap {r['gap_points']:+.1f} points")
    return {"checks": checks, "determinism": det, "jobs": results, "batch1": b1}


def stored_run(out_dir, jobs: list[dict], determinism: dict | None = None, batch1: dict | None = None) -> dict | None:
    """A finished run's files, read back ({run_info, checks, determinism, jobs, batch1}), or None if any is
    missing (then the run has to load the model and go on)."""
    out = Path(out_dir)
    need = ["run_info.json", "checks.json", *(f"{j['name']}.json" for j in jobs)]
    need += ["determinism.json"] if determinism else []
    need += ["batch1.json"] if batch1 else []
    if not all((out / f).exists() for f in need):
        return None

    def read(name):
        return json.loads((out / name).read_text())

    return {"run_info": read("run_info.json"), "checks": read("checks.json"),
            "determinism": read("determinism.json") if determinism else None,
            "jobs": {j["name"]: read(f"{j['name']}.json") for j in jobs},
            "batch1": read("batch1.json") if batch1 else None}


# ------------------------------------------------------------------------------------------- what ran, and how


def snapshot_dir(model: str, revision: str, hf_home: str | None = None) -> str:
    """The local folder of a model at a pinned revision in the Hugging Face cache ($HF_HOME/hub/models--ORG--NAME/
    snapshots/REVISION). Loading from it pins the weights, tokenizer, chat template and generation_config to that
    commit, with no download; a missing snapshot stops the run."""
    hub = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    path = hub / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        have = sorted(p.name for p in path.parent.iterdir()) if path.parent.is_dir() else []
        raise FileNotFoundError(f"{model} is not in the cache at revision {revision} (snapshots there: {have})")
    return str(path)


def cache_ref(model: str, ref: str = "main", hf_home: str | None = None) -> str | None:
    """The commit a ref (refs/main) points to in the local Hugging Face cache, or None if the cache has no such ref."""
    hub = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    path = hub / ("models--" + model.replace("/", "--")) / "refs" / ref
    return path.read_text().strip() if path.is_file() else None


def check_environment(plan: dict, pkg_dir) -> dict:
    """This container against the plan's fingerprint: the package files (code_hashes, against plan["code"]) and the
    installed version of every package the image pins (plan["pins"]; a local version label such as +cu128 is
    ignored). A difference stops the run before anything loads, so results are only produced, or read back, under
    the code and packages their plan names. Returns {"code", "installed"}."""
    code = code_hashes(pkg_dir)
    if code != plan["code"]:
        bad = sorted(f for f in set(code) | set(plan["code"]) if code.get(f) != plan["code"].get(f))
        raise CheckError(f"this container's package differs from the plan's in {', '.join(bad)}: launch again, which "
                         "plans afresh")
    have = installed(plan["pins"])
    wrong = sorted(n for n, v in plan["pins"].items() if (have[n] or "").split("+")[0] != v)
    if wrong:
        raise CheckError("installed versions differ from the image's pins: "
                         + ", ".join(f"{n} {have[n]} (pinned {plan['pins'][n]})" for n in wrong))
    return {"code": code, "installed": have}


def model_info(model, tok) -> dict:
    """What generation ran with: classes, dtype, the generation_config fields that matter (repetition_penalty is
    read here, at run time), the tokenizer's special ids, and the torch, transformers and GPU in use."""
    import torch
    import transformers

    gc = getattr(model, "generation_config", None)
    gcd = gc.to_dict() if hasattr(gc, "to_dict") else (dict(vars(gc)) if gc is not None else {})
    keep = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "eos_token_id", "pad_token_id",
            "bos_token_id", "max_length", "max_new_tokens")
    try:
        dtype = str(next(model.parameters()).dtype)
    except (AttributeError, StopIteration, TypeError):
        dtype = None
    return {"model_class": type(model).__name__, "tokenizer_class": type(tok).__name__, "dtype": dtype,
            "device": str(getattr(model, "device", None)),
            "generation_config": {k: gcd.get(k) for k in keep}, "repetition_penalty": gcd.get("repetition_penalty"),
            "tokenizer": {"bos": tok.bos_token_id, "eos": tok.eos_token_id, "pad": tok.pad_token_id,
                          "padding_side": tok.padding_side},
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "commit_hash": getattr(getattr(model, "config", None), "_commit_hash", None)}


# ------------------------------------------------------------------------------------------------- one plan


def run_plan(plan: dict, items: dict[str, dict], out_dir, pkg_dir, load: Callable, commit: Callable = lambda: None,
             log: Callable = print, hf_home: str | None = None) -> dict:
    """One planned run, in its container (calibration/model_check/app.py's evaluate). plan: {model, revision,
    condition, system, bos, jobs, determinism, batch1, gpu, items_sha256, code, pins, container_sha256}.
      1. check_environment: this container's package files and installed versions against the plan's;
      2. out_dir/plan.json claims the folder: a folder holding another plan's files stops the run;
      3. a run whose files are all there is read back without loading the model (stored_run);
      4. otherwise the model is loaded from the cache at the plan's revision (load(snapshot path) -> (model, tok)),
         run_info.json is written (keeping the generate arguments recorded by earlier starts), and run_jobs runs.
    Returns {plan, run_info, checks, determinism, jobs, batch1}."""
    out = Path(out_dir)
    env = check_environment(plan, pkg_dir)
    claimed = out / "plan.json"
    if claimed.exists() and json.loads(claimed.read_text()) != plan:
        raise CheckError(f"{out} holds the files of another plan")
    jobs, det, b1 = plan["jobs"], plan["determinism"], plan["batch1"]
    done = stored_run(out, jobs, det, b1)
    if done is not None:
        log(f"{plan['model']} {plan['condition']}: every file is on the volume; nothing to run")
        return {"plan": plan, **done}
    write_json(claimed, plan)
    t0 = time.time()
    path = snapshot_dir(plan["model"], plan["revision"], hf_home)
    model, tok = load(path)
    before = json.loads((out / "run_info.json").read_text()) if (out / "run_info.json").exists() else {}
    info = {"model": plan["model"], "revision": plan["revision"], "snapshot": path,
            "cache_refs_main": cache_ref(plan["model"], "main", hf_home), "condition": plan["condition"],
            "system": plan["system"], "bos_expected": plan["bos"], "plan_id": plan_id(plan),
            "items_sha256": plan["items_sha256"], "volume_dir": str(out), "load_seconds": round(time.time() - t0),
            "starts": before.get("starts", 0) + 1, **model_info(model, tok), "hashes": env["code"],
            "installed": env["installed"], "batch": {j["name"]: j["batch"] for j in jobs},
            "max_new_tokens": {j["name"]: j["max_new"] for j in jobs},
            "generate_kwargs": before.get("generate_kwargs", {})}
    write_json(out / "run_info.json", info)
    commit()
    log(json.dumps({k: info[k] for k in ("model", "condition", "plan_id", "model_class", "dtype", "gpu",
                                         "repetition_penalty", "cache_refs_main")}))
    res = run_jobs(model, tok, jobs, items, out, plan["system"], plan["bos"], det, b1, commit=commit, log=log)
    return {"plan": plan, "run_info": json.loads((out / "run_info.json").read_text()), **res}
