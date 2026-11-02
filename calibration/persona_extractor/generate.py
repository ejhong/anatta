"""Calibration check 4: G, the shared responses, and G', the lab's sampling at the reference's size, with Hugging Face
generate in the lab's image, in bf16 (what runs in the containers; no Modal here, so the tests run it on a laptop).

G's settings are the repo's sampling distribution (eval_persona.py's SamplingParams, which vLLM used as given): do_sample,
temperature 1.0, top_p 1.0, top_k 0, repetition_penalty 1.0, min_new_tokens 1, max_new_tokens 1,000, stopping at Qwen's
eos ids. They are passed explicitly, since transformers 5.21 fills any field left unset from the model's
generation_config (Qwen's top_k 20, top_p 0.8 and repetition_penalty 1.05). G' passes probe's settings on Qwen the same
way: temperature 0.7, top_p 0.95, top_k 20, repetition_penalty 1.05, max_new_tokens 128.

ProcessorRecord records, for every batch, the logits processors transformers builds from the merged settings and the
merged GenerationConfig itself; check_processors refuses a batch built with a forbidden processor (G: no temperature,
top-k, top-p or repetition-penalty processor) or without a required one at its value (G'). That is the registered
processor test, run on the real model too.

Prompts are rendered with the chat template (tokenize=False, the generation prompt added), tokenized without special
tokens and padded on the left, as probe.generate does. Answers are decoded with skip_special_tokens and not stripped, as
the repo does. Each batch's rows, text included, go to the volume as JSONL; the stage returns only numbers.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import traits as T

# The fields of a GenerationConfig the records keep (as passed, as merged, and the model's own).
CONFIG_KEYS = ("do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p", "repetition_penalty",
               "no_repeat_ngram_size", "min_new_tokens", "max_new_tokens", "min_length", "max_length", "num_beams",
               "eos_token_id", "pad_token_id", "bos_token_id")
PARAMS = {"TemperatureLogitsWarper": "temperature", "TopKLogitsWarper": "top_k", "TopPLogitsWarper": "top_p",
          "RepetitionPenaltyLogitsProcessor": "penalty", "MinNewTokensLengthLogitsProcessor": "min_new_tokens",
          "MinLengthLogitsProcessor": "min_length"}


def plain(v):
    """A config value as JSON: numbers, bools, None and lists of them."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v if v is None or isinstance(v, (bool, int, float)) else str(v)


def config_fields(gc) -> dict:
    if gc is None:
        return {}
    d = gc.to_dict() if hasattr(gc, "to_dict") else dict(vars(gc))
    return {k: plain(d.get(k)) for k in CONFIG_KEYS}


def describe_processor(p) -> dict:
    out = {"class": type(p).__name__}
    attr = PARAMS.get(out["class"])
    if attr is not None:
        out[attr] = plain(getattr(p, attr, None))
    return out


class ProcessorRecord:
    """While active, every call of model._get_logits_processor is recorded: the processors it built (their classes and
    key values) and the merged GenerationConfig it was given. The wrapper is an attribute of the model object, returns
    the original's list unchanged, and is removed on exit."""

    def __init__(self, model):
        self.model, self.calls = model, []

    def __enter__(self):
        own = vars(self.model)
        self.had, self.prev = "_get_logits_processor" in own, own.get("_get_logits_processor")
        orig = self.model._get_logits_processor

        def recorded(*a, **k):
            out = orig(*a, **k)
            gc = k.get("generation_config", a[0] if a else None)
            self.calls.append({"processors": [describe_processor(p) for p in out], "merged": config_fields(gc)})
            return out

        self.model._get_logits_processor = recorded
        return self

    def __exit__(self, *exc):
        if self.had:
            self.model._get_logits_processor = self.prev
        else:
            del self.model._get_logits_processor
        return False


LENGTH_KEYS = ("min_length", "max_length")  # set from each batch's prompt length, so they differ between batches


def comparable(call: dict) -> dict:
    """A batch's record without what follows from its prompt length (the merged min and max length, and the length a
    MinLengthLogitsProcessor carries), so the records of two batches compare on the settings alone."""
    return {"processors": [{k: v for k, v in p.items() if k not in LENGTH_KEYS} for p in call["processors"]],
            "merged": {k: v for k, v in call["merged"].items() if k not in LENGTH_KEYS}}


def check_processors(call: dict, forbidden: list[str], required: dict) -> None:
    """A batch's processors against the arm's registered ones: none of `forbidden`, and each of `required` at its value
    (within 1e-6). CheckError("processors") otherwise."""
    got = {p["class"]: p for p in call["processors"]}
    if any(name in got for name in forbidden):
        raise T.CheckError("processors")
    for name, value in required.items():
        p = got.get(name)
        if p is None or abs(float(p[PARAMS[name]]) - float(value)) > 1e-6:
            raise T.CheckError("processors")


def kwargs_for(settings: dict, eos: list[int], pad_id: int, max_new_tokens: int | None = None) -> dict:
    """The keyword arguments G (or G') passes to model.generate: the registered settings, every one explicit, the eos
    ids and the pad id; max_new_tokens replaced for the smoke."""
    kw = dict(settings)
    if max_new_tokens is not None:
        kw["max_new_tokens"] = max_new_tokens
    return {**kw, "eos_token_id": list(eos), "pad_token_id": pad_id}


def render(tok, system: str, user: str) -> str:
    """The prompt as eval_persona.py stores it: the chat template's rendering of the system and user turns with the
    generation prompt, as a string."""
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def generate_batch(model, tok, prompts: list[str], kwargs: dict, seed: int) -> tuple[list[str], list[int], list[bool]]:
    """One batch: torch.manual_seed(seed), left padding, the prompts tokenized without special tokens, model.generate
    with `kwargs`, the new tokens decoded with skip_special_tokens and not stripped. Returns (answers, new token counts
    up to the first eos id, capped: no eos id within max_new_tokens)."""
    import torch

    torch.manual_seed(seed)
    side, tok.padding_side = tok.padding_side, "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    finally:
        tok.padding_side = side
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    new = out[:, enc["input_ids"].shape[1]:]
    answers = tok.batch_decode(new, skip_special_tokens=True)
    eos = torch.tensor(kwargs["eos_token_id"], device=new.device)
    is_eos = torch.isin(new, eos)
    has = is_eos.any(1)
    first = torch.where(has, is_eos.int().argmax(1), torch.full_like(has, new.shape[1], dtype=torch.long))
    return answers, [int(x) for x in first.tolist()], [not bool(x) for x in has.tolist()]


class GpuPeak:
    """This process's peak GPU memory over a block (torch.cuda's allocator: allocated and reserved, in MiB) and the
    device's total; all None without a GPU. A GPU present and a peak of 0 is a failed measurement (CheckError
    "peak_memory")."""

    def __enter__(self):
        import torch

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        import torch

        self.allocated = self.reserved = self.total = None
        if self.cuda:
            torch.cuda.synchronize()
            self.allocated = round(torch.cuda.max_memory_allocated() / 2**20)
            self.reserved = round(torch.cuda.max_memory_reserved() / 2**20)
            self.total = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
            if exc[0] is None and self.reserved == 0:
                raise T.CheckError("peak_memory")
        return False


def full_batch(model, tok, plan: dict, data: dict, log=print) -> dict:
    """The smoke's full-size batch of G (plan["full_batch"]): G's first batch at the registered size and settings (its
    rows of the full layout, its seed, every setting explicit, its processors checked), timed, with this process's
    peak GPU memory. Read as nothing: the answers are dropped here, never written, and only numbers are returned."""
    import time

    fb = plan["full_batch"]
    lay = fb["layout"]
    b = T.batches([fb["trait"]], lay["questions"], lay["instructions"], lay["samples"], fb["size"])[0]
    d = data[fb["trait"]]
    cells = [T.cell_of(k, lay["instructions"], lay["samples"]) for k in b["rows"]]
    texts = [render(tok, T.system_strings(fb["trait"], d["instruction"][ii], plan["prompts"])[b["sign"]],
                    d["questions"][qi]) for qi, ii, _ in cells]
    kwargs = kwargs_for(fb["kwargs"], plan["eos"], tok.pad_token_id)
    t0 = time.time()
    with GpuPeak() as peak, ProcessorRecord(model) as rec:
        answers, new_tokens, capped = generate_batch(model, tok, texts, kwargs, fb["seed"])
    seconds = round(time.time() - t0, 1)
    if not rec.calls:
        raise T.CheckError("processors")
    check_processors(rec.calls[-1], fb["forbidden"], fb["required"])
    del answers
    log(f"full-size batch: {len(texts)} prompts, {sum(capped)} capped, {seconds} s, peak reserved {peak.reserved} MiB")
    return {"prompts": len(texts), "sign": b["sign"], "seed": fb["seed"], "max_new_tokens": kwargs["max_new_tokens"],
            "seconds": seconds, "new_tokens": summary(new_tokens), "capped": sum(capped),
            "peak_allocated_mib": peak.allocated, "peak_reserved_mib": peak.reserved, "total_mib": peak.total,
            "processors": rec.calls[-1]["processors"]}


def slice_tokens(tok, prompt: str, answer: str) -> int:
    """The size of generate_vec's response slice for (prompt, answer): len(tokenize(prompt + answer)) minus
    len(encode(prompt)). Zero when the answer's tokens all merge into the prompt's (an answer of "\\n" on Qwen), and
    the script's mean over that slice would be NaN."""
    whole = tok(prompt + answer, add_special_tokens=False)["input_ids"]
    return len(whole) - len(tok.encode(prompt, add_special_tokens=False))


def model_facts(model) -> dict:
    """The model's dtype, attention implementation, generation_config, and the GPU and versions in use."""
    import torch
    import transformers

    try:
        dtype = str(next(model.parameters()).dtype)
    except (StopIteration, AttributeError, TypeError):
        dtype = None
    return {"dtype": dtype, "attn_implementation": getattr(getattr(model, "config", None), "_attn_implementation", None),
            "generation_config": config_fields(getattr(model, "generation_config", None)), "gpu": T.gpu_name(),
            "torch": torch.__version__, "transformers": transformers.__version__}


def check_generation_config(model, registered: dict) -> dict:
    """The model's generation_config against the registered fields (probe.extract samples under it)."""
    got = config_fields(getattr(model, "generation_config", None))
    for k, v in registered.items():
        g = got.get(k)
        if g is None or (isinstance(v, float) and abs(float(g) - v) > 1e-9) or (not isinstance(v, float) and g != v):
            raise T.CheckError("generation_config")
    return got


def batch_file(out, arm: str, index: int) -> Path:
    return Path(out) / "data" / arm / f"batch_{index:04d}.jsonl"


def processors_file(out, arm: str, index: int) -> Path:
    return Path(out) / "data" / arm / f"batch_{index:04d}.processors.json"


def load_rows(out, arm: str, trait: str, sign: str, plan: dict) -> list[dict]:
    """Every generated row of one trait and sign, read back from the batch files, in row order."""
    spec = plan[arm]
    lay = plan["layout"]
    rows = []
    for b in T.batches(plan["traits"], lay["questions"], lay["instructions"], spec["samples"], spec["batch"]):
        if b["trait"] == trait and b["sign"] == sign:
            got = T.read_jsonl(T.require(batch_file(out, arm, b["index"])))
            if [r["row"] for r in got] != b["rows"]:
                raise T.CheckError("missing")
            rows += got
    return rows


def keep_file(out, judged: bool = False) -> Path:
    return Path(out) / ("judged" if judged else "") / "keep.json"


def keep_rows(out) -> dict:
    """The rows of G every extractor reads, by trait (keep.json, written by the CSV step or keep_from_slices): pairs
    are kept or dropped together, so the same rows hold for both signs."""
    return {t: set(rows) for t, rows in T.read_json(T.require(keep_file(out)))["kept"].items()}


def pairs_of(out, plan: dict, trait: str) -> list[tuple[dict, dict]]:
    """G's (pos, neg) rows of one trait, paired by row index (row k of each sign is one pair)."""
    pos, neg = load_rows(out, "G", trait, "pos", plan), load_rows(out, "G", trait, "neg", plan)
    if [r["row"] for r in pos] != [r["row"] for r in neg]:
        raise T.CheckError("missing")
    return list(zip(pos, neg))


def keep_from_slices(plan: dict, out) -> dict:
    """keep.json without the second image (no CSV step): every pair whose two answers have a non-empty answer slice
    under the lab's tokenizer (slice_tokens at generation). Returns the counts it wrote."""
    kept, counts = {}, {}
    for t in plan["traits"]:
        pairs = pairs_of(out, plan, t)
        bad = [p["row"] for p, n in pairs if p["slice_tokens"] < 1 or n["slice_tokens"] < 1]
        kept[t] = [p["row"] for p, _ in pairs if p["row"] not in set(bad)]
        counts[t] = {"pairs": len(pairs), "kept": len(kept[t]), "dropped": {"slice_lab": len(bad)}}
    T.write_json(keep_file(out), {"kept": kept, "counts": counts})
    return {"traits": counts, "csv": False}


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    q = lambda f: s[min(len(s) - 1, int(f * (len(s) - 1) + 0.5))]  # noqa: E731
    return {"n": len(s), "mean": round(statistics.fmean(s), 3), "min": s[0], "p25": q(0.25), "median": q(0.5),
            "p75": q(0.75), "p90": q(0.9), "max": s[-1]}


def run_arm(model, tok, plan: dict, arm: str, out, data: dict, commit=lambda: None, log=print,
            guard: T.Guard | None = None) -> dict:
    """G or G' (arm "G" or "Gprime") on a loaded model: every batch not yet on the volume, each seeded seed_base plus
    its index, its processors recorded and checked. Writes data/<arm>/batch_<i>.jsonl (rows with their prompt and
    answer, which stay on the volume) and returns numbers: lengths by trait and sign, the share capped, empty
    generate_vec slices, the settings as passed, the merged GenerationConfig and the processors of the first batch."""
    spec, lay, prompts = plan[arm], plan["layout"], plan["prompts"]
    kwargs = kwargs_for(spec["kwargs"], plan["eos"], tok.pad_token_id)
    todo = T.batches(plan["traits"], lay["questions"], lay["instructions"], spec["samples"], spec["batch"])
    for b in todo:
        if guard is not None:
            guard.row = b["index"]
        path = batch_file(out, arm, b["index"])
        if path.exists() and [r["row"] for r in T.read_jsonl(path)] == b["rows"]:
            continue
        T.before(commit, f"{arm}_batch")
        t = b["trait"]
        d = data[t]
        cells = [T.cell_of(k, lay["instructions"], spec["samples"]) for k in b["rows"]]
        texts = [render(tok, T.system_strings(t, d["instruction"][ii], prompts)[b["sign"]], d["questions"][qi])
                 for qi, ii, _ in cells]
        # Ruled (NOTES, 12:20 UTC, ruling 1): G' batch i keeps G batch i's seed, as registered; the card notes it.
        seed = spec["seed_base"] + b["index"]
        with ProcessorRecord(model) as rec:
            answers, new_tokens, capped = generate_batch(model, tok, texts, kwargs, seed)
        if not rec.calls:
            raise T.CheckError("processors")
        check_processors(rec.calls[-1], spec["forbidden"], spec["required"])
        T.write_json(processors_file(out, arm, b["index"]), rec.calls[-1])
        rows = [{"trait": t, "sign": b["sign"], "row": k, "qi": qi, "ii": ii, "si": si, "batch": b["index"],
                 "seed": seed, "prompt": p, "answer": a, "new_tokens": n, "capped": c,
                 "slice_tokens": slice_tokens(tok, p, a)}
                for k, (qi, ii, si), p, a, n, c in zip(b["rows"], cells, texts, answers, new_tokens, capped)]
        T.write_jsonl(path, rows)
        commit()
        log(f"{arm} batch {b['index'] + 1}/{len(todo)}: {len(rows)} rows, {sum(capped)} capped, seed {seed}")
    lengths = {}
    for t in plan["traits"]:
        lengths[t] = {}
        for sign in T.SIGNS:
            rows = load_rows(out, arm, t, sign, plan)
            lengths[t][sign] = {"new_tokens": summary([r["new_tokens"] for r in rows]),
                                "capped": sum(r["capped"] for r in rows), "rows": len(rows),
                                "capped_share": round(sum(r["capped"] for r in rows) / max(len(rows), 1), 4),
                                "empty_slice": sum(r["slice_tokens"] < 1 for r in rows)}
    records = [T.read_json(T.require(processors_file(out, arm, b["index"]))) for b in todo]
    same = all(comparable(r) == comparable(records[0]) for r in records)
    return {"arm": arm, "batches": len(todo), "passed": {k: plain(v) for k, v in kwargs.items()},
            "first_batch": records[0], "processors_same_every_batch": same, "lengths": lengths,
            "max_new_tokens": kwargs["max_new_tokens"]}
