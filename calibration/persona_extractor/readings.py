"""Calibration check 4: probe's readings, the recorder, and the probe.extract arms (what runs in the lab's image; no
Modal here, so the tests run it on a laptop).

  read       one forward pass under probe.capture over models.chat_ids of (the messages, the answer), exactly as
             probe.response_activations makes it, and three means from the same stored activations:
               P1       probe.response_activations' value: the tokens from the end of the prompt (generation prompt
                        included) to the end of the rendered reply, so the template's closing tokens are in it
                        (Qwen's '<|im_end|>' and '\\n'); a test asserts it equals probe.response_activations bitwise;
               P1-trim  the same slice ended at the answer's last token: generate_vec's token set (its prompt_len is
                        len(encode(p)) and its end len(tokenize(p + a))), checked against the rendering;
               P1-128   the trimmed slice limited to the answer's first 128 tokens.
             With the messages lacking the system turn, P1 is P0 (probe's include_system=False).
  read_arm   P1, P1-trim, P1-128 and P0 over G's kept rows (or P1 and P0 over G'), per-response means in fp32 written
             to the volume in chunks, resumable.
  Recorder   for one probe.extract call at a time, the names anatta.probe.generate and
             anatta.probe.response_activations are rebound to wrappers that call the originals, keep a copy of what
             they return and return it unchanged; the originals are put back in a finally block. probe.py is not
             edited. rebuild() remakes probe.extract's unit vectors from the recorded means with extract's own
             arithmetic.
  run_arms   E-sys, E, E-sys' and E' for each trait (probe.extract, every other argument at its default), through the
             recorder when it is on; meta["examples"] is deleted before any Direction is saved; then E-trail-128 on the
             recorded E-sys responses.
  run_mprime M' on Ministral: the six ego traits, with include_system False and True, each trait caught on its own;
             the False calls' responses read with the trailing '</s>' trimmed; v_ego rebuilt by factor.composite for
             each convention; compared with the stored 28 Oct directions. Ledger only.
  run_cheap  the cheap tier's item: stored G responses read again with probe.response_activations.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import generate as G
import traits as T

# What probe.generate passes to model.generate (src/anatta/probe.py:133-140 at dc32cf7), so the record can show what
# transformers merges it into; probe.extract passes no temperature, so generate's default 0.7 applies.
PROBE_KWARGS = {"max_new_tokens": 128, "do_sample": True, "temperature": 0.7, "top_p": 0.95}


def stack(d: dict, blocks: list[int]):
    import torch

    return torch.stack([d[b] for b in blocks])


def save_tensor(path, obj) -> None:
    """torch.save to a temporary file, then renamed, so a stopped run never leaves half a file."""
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def messages(system: str | None, user: str) -> list[dict]:
    return ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": user}]


# ------------------------------------------------------------------------------------------------ reading


def token_slices(tok, msgs: list[dict], answer: str) -> dict:
    """The token positions of one response, from the tokenizer alone: probe.response_activations' ids (models.chat_ids
    of the prompt, and of the prompt with the reply, as src/anatta/probe.py:104-115 builds them) with its slice
    [start, end), and generate_vec's end, len(tokenize(p + a)) without special tokens, for P1-trim. Checks that
    generate_vec's prompt_len, len(encode(p)), is probe's start (CheckError("slice_start")) and that the rendered reply
    begins with the tokens of p + a (CheckError("slice_prefix")). Returns {full, start, end, end_trim, n_answer,
    n_closing}."""
    from anatta.models import chat_ids

    prompt = chat_ids(tok, msgs)
    full = chat_ids(tok, msgs + [{"role": "assistant", "content": answer}], False)
    start = prompt.shape[1]
    end = max(full.shape[1], start + 1)
    p_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if len(tok.encode(p_text, add_special_tokens=False)) != start:
        raise T.CheckError("slice_start", start=start)
    pa = tok(p_text + answer, add_special_tokens=False)["input_ids"]
    end_trim = len(pa)
    if full[0, :end_trim].tolist() != list(pa):
        raise T.CheckError("slice_prefix")
    return {"full": full, "start": start, "end": end, "end_trim": end_trim, "n_answer": end_trim - start,
            "n_closing": end - end_trim}


def read(model, tok, msgs: list[dict], answer: str, blocks: list[int], cap: int | None = None) -> dict:
    """One response read three ways from one forward pass (see the module's docstring and token_slices). Returns
    {p1, trim, cap: {block: [d] fp32 on the CPU} (trim and cap None when the answer's own slice is empty), start, end,
    end_trim, n_answer, n_closing}."""
    import torch

    from anatta.probe import capture

    s = token_slices(tok, msgs, answer)
    start, end, end_trim = s["start"], s["end"], s["end_trim"]
    with torch.no_grad():
        with capture(model, blocks) as store:
            model(s["full"].to(model.device))
        p1 = {i: store[i][0, start:end].float().mean(0).cpu() for i in blocks}
        trim = capped = None
        if s["n_answer"] >= 1:
            trim = {i: store[i][0, start:end_trim].float().mean(0).cpu() for i in blocks}
            if cap is not None:
                stop = min(end_trim, start + cap)
                capped = {i: store[i][0, start:stop].float().mean(0).cpu() for i in blocks}
    return {"p1": p1, "trim": trim, "cap": capped, **{k: s[k] for k in ("start", "end", "end_trim", "n_answer",
                                                                         "n_closing")}}


def chunk_file(out, arm: str, variant: str, trait: str, sign: str, j: int) -> Path:
    return Path(out) / "means" / arm / variant / trait / sign / f"chunk_{j:04d}.pt"


VARIANTS = {"G": ("P1", "P1trim", "P1cap", "P0"), "Gprime": ("P1", "P0")}


def read_arm(model, tok, plan: dict, arm: str, out, data: dict, keep: dict, commit=lambda: None, log=print,
             guard: T.Guard | None = None) -> dict:
    """The lab's readings of G (P1, P1-trim, P1-128 from one pass with the system turn; P0 from a second pass without
    it) or of G' (P1 and P0), over the kept rows in row order, in chunks of plan["readings"]["chunk"] responses. Each
    chunk file is {rows, blocks, means: [n, blocks, d] fp32}; chunks already on the volume are skipped. Returns
    counts and the answers' token lengths (numbers only)."""
    import torch

    variants = VARIANTS[arm]
    r = plan["readings"]
    blocks = list(range(r["blocks"][0], r["blocks"][1] + 1))
    lengths: dict = {}
    for t in plan["traits"]:
        lengths[t] = {}
        for sign in T.SIGNS:
            rows = [x for x in G.load_rows(out, arm, t, sign, plan) if x["row"] in keep[t]]
            n_answer, n_closing = [], []
            for j in range(0, len(rows), r["chunk"]):
                part = rows[j:j + r["chunk"]]
                meta = Path(chunk_file(out, arm, variants[0], t, sign, j)).with_suffix(".json")
                if all(chunk_file(out, arm, v, t, sign, j).exists() for v in variants) and meta.exists():
                    m = T.read_json(meta)
                    n_answer += m["n_answer"]
                    n_closing += m["n_closing"]
                    continue
                T.before(commit, f"{arm}_chunk")
                buf: dict = {v: [] for v in variants}
                na, nc = [], []
                for x in part:
                    if guard is not None:
                        guard.row = x["row"]
                    system = T.system_strings(t, data[t]["instruction"][x["ii"]], plan["prompts"])[sign]
                    question = data[t]["questions"][x["qi"]]
                    if G.render(tok, system, question) != x["prompt"]:
                        raise T.CheckError("render")
                    res = read(model, tok, messages(system, question), x["answer"], blocks, r["cap"])
                    buf["P1"].append(stack(res["p1"], blocks))
                    if arm == "G":
                        if res["trim"] is None:  # kept rows have a non-empty slice; the CSV step dropped the others
                            raise T.CheckError("empty_slice", row=x["row"])
                        buf["P1trim"].append(stack(res["trim"], blocks))
                        buf["P1cap"].append(stack(res["cap"], blocks))
                    na.append(res["n_answer"])
                    nc.append(res["n_closing"])
                    res0 = read(model, tok, messages(None, question), x["answer"], blocks)
                    buf["P0"].append(stack(res0["p1"], blocks))
                for v in variants:
                    save_tensor(chunk_file(out, arm, v, t, sign, j),
                                {"rows": [x["row"] for x in part], "blocks": blocks,
                                 "means": torch.stack(buf[v]).float()})
                T.write_json(meta, {"rows": [x["row"] for x in part], "n_answer": na, "n_closing": nc})
                n_answer += na
                n_closing += nc
                commit()
                log(f"{arm} readings {t} {sign}: {min(j + len(part), len(rows))}/{len(rows)}")
            lengths[t][sign] = {"rows": len(rows), "answer_tokens": G.summary(n_answer),
                                "closing_tokens": sorted(set(n_closing)),
                                "closing_weight_mean": round(sum(c / (a + c) for a, c in zip(n_answer, n_closing))
                                                             / max(len(n_answer), 1), 6)}
    return {"arm": arm, "variants": list(variants), "blocks": blocks, "lengths": lengths}


def load_means(out, arm: str, variant: str, trait: str, sign: str):
    """A stored reading's per-response means for one trait and sign, chunk by chunk, as (rows, blocks, [n, L, d])."""
    import torch

    d = Path(out) / "means" / arm / variant / trait / sign
    files = sorted(d.glob("chunk_*.pt"))
    if not files:
        raise T.CheckError("missing")
    parts = [torch.load(f, weights_only=False) for f in files]
    rows = [k for p in parts for k in p["rows"]]
    return rows, parts[0]["blocks"], torch.cat([p["means"] for p in parts])


# ------------------------------------------------------------------------------------------------ the recorder


class Recorder:
    """Records one probe.extract call: install() rebinds anatta.probe.generate and anatta.probe.response_activations
    (the module globals extract calls, probe.py:179 and 190) to wrappers that call the originals, keep a copy of what
    they return (the responses; each response's per-layer means, cloned) and return the originals' outputs unchanged,
    the same objects. The originals are restored in a finally block, whatever happens inside."""

    def __init__(self):
        self.responses: list[str] = []
        self.means: list[dict] = []
        self.read_args: list[tuple[str, str]] = []  # (the user turn, the response) each reading was given

    @contextmanager
    def install(self, probe):
        gen, acts = probe.generate, probe.response_activations

        def generate(*a, **k):
            out = gen(*a, **k)
            self.responses.extend(out)
            return out

        def response_activations(*a, **k):
            out = acts(*a, **k)
            msgs = k["messages"] if "messages" in k else a[2]
            response = k["response"] if "response" in k else a[3]
            self.means.append({i: t.detach().clone() for i, t in out.items()})
            self.read_args.append((msgs[-1]["content"], response))
            return out

        probe.generate, probe.response_activations = generate, response_activations
        try:
            yield self
        finally:
            probe.generate, probe.response_activations = gen, acts


def jobs_of(trait) -> list[tuple[int, int, int]]:
    """probe.extract's jobs in its order (probe.py:169-173): for each question, for each pair, pos then neg, as
    (question index, pair index, sign)."""
    return [(qi, pi, s) for qi in range(len(trait.questions)) for pi in range(len(trait.pairs)) for s in (+1, -1)]


def check_alignment(rec: Recorder, trait) -> None:
    """Every job generated one response and was read once, in order, with its own question and response (extract
    without a judge keeps every row)."""
    jobs = jobs_of(trait)
    if not (len(rec.responses) == len(rec.means) == len(rec.read_args) == len(jobs)):
        raise T.CheckError("alignment", responses=len(rec.responses), reads=len(rec.means), jobs=len(jobs))
    for (qi, _, _), r, (user, resp) in zip(jobs, rec.responses, rec.read_args):
        if user != trait.questions[qi] or resp != r:
            raise T.CheckError("alignment")


def rebuild(means: list[dict], signs: list[int], layers: list[int]) -> dict:
    """probe.extract's unit vectors from per-response means, with its arithmetic (probe.py:192-195, 210-211)."""
    import torch

    pos = [a for a, s in zip(means, signs) if s > 0]
    neg = [a for a, s in zip(means, signs) if s < 0]
    full = {i: torch.stack([a[i] for a in pos]).mean(0) - torch.stack([a[i] for a in neg]).mean(0) for i in layers}
    return {i: full[i] / full[i].norm().clamp_min(1e-8) for i in layers}


def save_record(d: Path, rec: Recorder, trait, layers: list[int]) -> None:
    """The recorded responses (JSONL, with each job's question index, pair index and sign) and means ([n, layers, d]
    fp32) to the volume."""
    import torch

    rows = [{"qi": qi, "pi": pi, "sign": s, "response": r} for (qi, pi, s), r in zip(jobs_of(trait), rec.responses)]
    T.write_jsonl(d / "responses.jsonl", rows)
    save_tensor(d / "means.pt", {"layers": layers, "signs": [s for _, _, s in jobs_of(trait)],
                                 "qi": [qi for qi, _, _ in jobs_of(trait)], "pi": [pi for _, pi, _ in jobs_of(trait)],
                                 "means": torch.stack([stack(m, layers) for m in rec.means]).float()})


def extract_one(model, tok, trait, spec: dict, d: Path, recorder: bool) -> dict:
    """One probe.extract call (include_system and seed from spec, everything else at its default), through the
    recorder when it is on. meta["examples"] is deleted before the Direction is saved. Returns its numbers."""
    import anatta.probe as probe

    rec = Recorder() if recorder else None
    with rec.install(probe) if rec is not None else nullcontext():
        direction = probe.extract(model, tok, trait, include_system=spec["include_system"], seed=spec["seed"])
    direction.meta.pop("examples", None)
    layers = sorted(direction.vectors)
    if rec is not None:
        check_alignment(rec, trait)
        save_record(d, rec, trait, layers)
    direction.save(d / "direction.pt")
    return {"layers": layers, "n_kept": direction.meta["n_kept"], "n_total": direction.meta["n_total"],
            "include_system": direction.meta["include_system"], "judged": direction.meta["judged"],
            "scores": {str(i): round(float(direction.scores[i]), 6) for i in layers},
            "recorded": rec is not None}


def lengths_of(tok, responses: list[str], cap: int) -> dict:
    """Recorded responses' lengths, re-tokenized without special tokens (probe.generate returns text only)."""
    n = [len(tok.encode(r, add_special_tokens=False)) for r in responses]
    return {"tokens": G.summary(n), "at_cap": sum(x >= cap for x in n), "share_at_cap": round(sum(x >= cap for x in n)
                                                                                             / max(len(n), 1), 4)}


def trail(model, tok, trait, rows: list[dict], layers: list[int], include_system: bool, d: Path,
          guard: T.Guard | None = None) -> dict:
    """Recorded responses read with and without the template's closing tokens (P1 and P1-trim through probe.capture
    and models.chat_ids), in the context of the call that made them. Responses whose own slice is empty are left out
    of both readings and counted. Writes {variant}.pt ([n, layers, d] fp32 with signs and question indices) and
    returns the counts and the mean closing-token weight per sign."""
    import torch

    keep, p1, trim, weights = [], [], [], {+1: [], -1: []}
    skipped = 0
    for j, r in enumerate(rows):
        if guard is not None:
            guard.row = j
        system = trait.pairs[r["pi"]]["pos" if r["sign"] > 0 else "neg"] if include_system else None
        res = read(model, tok, messages(system, trait.questions[r["qi"]]), r["response"], layers)
        if res["trim"] is None:
            # Ruled (NOTES, 12:20 UTC, ruling 5): an empty response is left out of both readings and counted.
            skipped += 1
            continue
        keep.append(r)
        p1.append(stack(res["p1"], layers))
        trim.append(stack(res["trim"], layers))
        weights[r["sign"]].append(res["n_closing"] / (res["n_answer"] + res["n_closing"]))
    for name, means in (("P1", p1), ("P1trim", trim)):
        save_tensor(d / f"{name}.pt", {"layers": layers, "signs": [r["sign"] for r in keep],
                                       "qi": [r["qi"] for r in keep],
                                       "means": torch.stack(means).float() if means else torch.zeros(0)})
    mean = lambda xs: round(sum(xs) / len(xs), 6) if xs else None  # noqa: E731
    return {"read": len(keep), "skipped_empty": skipped, "closing_weight_mean": {"pos": mean(weights[+1]),
                                                                                 "neg": mean(weights[-1])}}


def probe_generation_config(model, tok) -> dict:
    """What transformers merges probe.generate's arguments into on this model (its generation_config fills the rest):
    the merged GenerationConfig's fields, without generating anything."""
    gc, _ = model._prepare_generation_config(None, **PROBE_KWARGS, pad_token_id=tok.pad_token_id)
    return {"passed": {**PROBE_KWARGS, "pad_token_id": tok.pad_token_id}, "merged": G.config_fields(gc)}


def arm_dir(out, arm: str, trait: str) -> Path:
    return Path(out) / "probe" / arm / trait


def identical(out, a: str, b: str, trait: str) -> dict | None:
    """How many recorded responses two arms share, position by position (E-det), or None without both records."""
    pa, pb = arm_dir(out, a, trait) / "responses.jsonl", arm_dir(out, b, trait) / "responses.jsonl"
    if not (pa.exists() and pb.exists()):
        return None
    ra, rb = [r["response"] for r in T.read_jsonl(pa)], [r["response"] for r in T.read_jsonl(pb)]
    return {"identical": sum(x == y for x, y in zip(ra, rb)), "of": min(len(ra), len(rb))}


def run_arms(model, tok, plan: dict, out, data: dict, commit=lambda: None, log=print,
             guard: T.Guard | None = None, part: str = "all") -> dict:
    """The probe.extract arms on Qwen: for each of the plan's probe traits, each arm in plan["arms"] not yet on the
    volume (probe.extract with include_system and seed from the arm, every other argument at its default). With the
    recorder on, also E-trail-128 (the recorded E-sys responses read with and without the closing tokens, at the
    arm's blocks), the arms' response lengths and E-det (identical responses between the same-seed calls with
    include_system True and False). part "graded" runs only plan["probe"]["graded"]'s traits and arms (C4's arm and
    E-sys' for the verdict's ceiling, run before the report-only stages) and nothing after them; "all" runs the rest.
    Returns numbers only."""
    pr, prompts = plan["probe"], plan["prompts"]
    recorder = plan["switches"]["recorder"]
    graded = part == "graded"
    res: dict = {"arms": {}, "trail128": {}, "lengths": {}, "determinism": {}, "recorder": recorder, "part": part,
                 "generation": probe_generation_config(model, tok)}
    for ti, t in enumerate(pr["graded"]["traits"] if graded else pr["traits"]):
        trait = T.probe_trait(t, data[t], prompts, questions=pr["questions"])
        res["arms"][t], res["lengths"][t] = {}, {}
        for arm, spec in plan["arms"].items():
            if graded and arm not in pr["graded"]["arms"]:
                continue
            if guard is not None:
                guard.row = ti
            d = arm_dir(out, arm, t)
            done = d / "arm.json"
            if not done.exists():
                T.before(commit, "probe_call")
                T.write_json(done, extract_one(model, tok, trait, spec, d, recorder))
                commit()
                log(f"{arm} {t}: done")
            res["arms"][t][arm] = T.read_json(done)
            if recorder:
                rows = T.read_jsonl(d / "responses.jsonl")
                res["lengths"][t][arm] = lengths_of(tok, [r["response"] for r in rows],
                                                    plan["instrument"]["defaults"]["max_new_tokens"])
        if graded or not recorder:
            continue
        pairs = {}
        for a, sa in plan["arms"].items():
            for b, sb in plan["arms"].items():
                if sa["seed"] == sb["seed"] and sa["include_system"] and not sb["include_system"]:
                    pairs[f"{a}|{b}"] = identical(out, a, b, t)
        res["determinism"][t] = pairs
        sys_arm = next((a for a, sp in plan["arms"].items() if sp["include_system"] and sp["seed"] == 0), None)
        if sys_arm is not None:
            d = Path(out) / "probe" / "trail128" / t
            done = d / "trail.json"
            if not done.exists():
                T.before(commit, "probe_trail")
                rows = T.read_jsonl(arm_dir(out, sys_arm, t) / "responses.jsonl")
                layers = res["arms"][t][sys_arm]["layers"]
                T.write_json(done, trail(model, tok, trait, rows, layers, True, d, guard))
                commit()
            res["trail128"][t] = {**T.read_json(done), "cos_by_layer": trail_cos(d)}
    return res


# ------------------------------------------------------------------------------------------------ M'


def check_stored(stored_dir, want: dict | None) -> dict:
    """The stored 28 Oct directions this container sees, against the sha256s the plan took from the laptop's folder
    when it was made: {relative path: sha256}. A file the plan names that differs or is missing here stops the stage
    (CheckError("stored")); with no stored files at launch (want None) nothing is compared."""
    if not want:
        return {}
    got = {}
    for rel, sha in want.items():
        p = Path(stored_dir) / rel
        got[rel] = T.sha256_file(p) if p.exists() else None
        if got[rel] != sha:
            raise T.CheckError("stored", want=sha, got=got[rel])
    return got


def stored_direction(stored_dir, rel: str):
    from anatta.probe import Direction

    p = Path(stored_dir) / rel
    return Direction.load(p) if p.exists() else None


def cos(a, b) -> float:

    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def run_mprime(model, tok, plan: dict, out, trait_loader, stored_dir, commit=lambda: None, log=print,
               guard: T.Guard | None = None) -> dict:
    """M' on Ministral (ledger only): each ego trait with each convention in plan["mprime"]["conventions"]
    (include_system False and True; the smoke runs one), seed 0, through the recorder when it is on. A failing call
    (Ministral's template refuses an empty reply) is counted, by the exception's type only, and the arm carries on.
    The include_system False calls' responses are read again with the trailing '</s>' trimmed (E-M-trail); v_ego is
    rebuilt by factor.composite for each convention whose calls all ran, at the plan's layers; everything is compared
    with the stored 28 Oct directions when the image carries them. Returns numbers only."""
    import anatta.probe as probe
    from anatta.factor import composite

    m = plan["mprime"]
    recorder = plan["switches"]["recorder"]
    names, convs = m["traits"], m["conventions"]
    no_sys = next((c for c, inc in convs.items() if not inc), None)
    with_sys = next((c for c, inc in convs.items() if inc), None)
    dirs: dict = {c: {} for c in convs}
    failures, trails, det = [], {}, {}
    for ti, name in enumerate(names):
        if guard is not None:
            guard.row = ti
        trait = trait_loader(name)
        if m.get("questions"):
            trait.questions = trait.questions[:m["questions"]]
        for conv, include in convs.items():
            d = Path(out) / "mprime" / conv / name
            done = d / "arm.json"
            if not done.exists():
                T.before(commit, "mprime_call")  # outside the try: a refusal stops the stage, it isn't a failed call
                try:
                    T.write_json(done, extract_one(model, tok, trait, {"include_system": include, "seed": m["seed"]},
                                                   d, recorder))
                except Exception as e:  # noqa: BLE001 (each trait on its own; only the type is kept)
                    failures.append({"trait": name, "convention": conv, "error": T.error_name(e)})
                    continue
                commit()
            dirs[conv][name] = probe.Direction.load(d / "direction.pt")
        if not recorder:
            continue
        if no_sys is not None and with_sys is not None and all(name in dirs[c] for c in (no_sys, with_sys)):
            a = [r["response"] for r in T.read_jsonl(Path(out) / "mprime" / no_sys / name / "responses.jsonl")]
            b = [r["response"] for r in T.read_jsonl(Path(out) / "mprime" / with_sys / name / "responses.jsonl")]
            det[name] = {"identical": sum(x == y for x, y in zip(a, b)), "of": min(len(a), len(b))}
        if no_sys is not None and name in dirs[no_sys]:
            d = Path(out) / "mprime" / "trail" / name
            done = d / "trail.json"
            if not done.exists():
                T.before(commit, "mprime_trail")
                rows = T.read_jsonl(Path(out) / "mprime" / no_sys / name / "responses.jsonl")
                try:
                    T.write_json(done, trail(model, tok, trait, rows, sorted(dirs[no_sys][name].vectors), False, d,
                                             guard))
                except Exception as e:  # noqa: BLE001
                    failures.append({"trait": name, "convention": "trail", "error": T.error_name(e)})
                commit()  # outside the try: a budget stop raised here stops the stage, not this trait's trail
            if done.exists():
                trails[name] = {**T.read_json(done), "cos_by_layer": trail_cos(d)}
    lo, hi = m["layers"]
    ego_layers = list(range(lo, hi + 1))
    egos = {}
    for conv in convs:
        if all(n in dirs[conv] for n in names):
            egos[conv] = composite(dirs[conv], names, ego_layers)
            egos[conv].meta.pop("examples", None)
            egos[conv].save(Path(out) / "mprime" / conv / "ego.pt")
    res = {"failures": failures, "failed_calls": len(failures), "determinism": det, "trail": trails,
           "conventions": list(convs), "ego_built": sorted(egos),
           "numbers": {c: {n: {"best_layer": dd.best_layer, "score_best": round(float(dd.scores[dd.best_layer]), 4),
                               "n_kept": dd.meta["n_kept"], "n_total": dd.meta["n_total"]}
                           for n, dd in dirs[c].items()} for c in dirs}}
    if no_sys in egos and with_sys in egos:
        e0, e1 = egos[no_sys], egos[with_sys]
        res["ego_ctx"] = {str(layer): round(cos(e1.vectors[layer], e0.vectors[layer]), 6) for layer in ego_layers}
    stored, s = {}, m["stored"]
    res["stored_sha256"] = check_stored(stored_dir, plan.get("stored_sha256"))
    if res["stored_sha256"] and no_sys is not None:
        for name in names:
            sd = stored_direction(stored_dir, f"{s['model_slug']}/{name}.pt")
            if sd is not None and name in dirs[no_sys]:
                layer, now = sd.best_layer, dirs[no_sys][name]
                stored[name] = {"stored_best_layer": layer, "cos": round(cos(now.vectors[layer], sd.vectors[layer]), 6)
                                if layer in now.vectors else None}
        se = stored_direction(stored_dir, f"{s['model_slug']}/{s['ego']}.pt")
        if se is not None and no_sys in egos:
            stored["ego"] = {str(layer): round(cos(egos[no_sys].vectors[layer], se.vectors[layer]), 6)
                             for layer in ego_layers if layer in se.vectors}
    res["stored"] = stored
    return res


def trail_cos(d: Path) -> dict:
    """cos(direction with the closing tokens, without them) by layer, from trail()'s two files."""
    import torch

    a, b = torch.load(d / "P1.pt", weights_only=False), torch.load(d / "P1trim.pt", weights_only=False)
    if not len(a["signs"]):
        return {}
    sg = torch.tensor(a["signs"])
    out = {}
    for j, layer in enumerate(a["layers"]):
        da = a["means"][sg > 0, j].mean(0) - a["means"][sg < 0, j].mean(0)
        db = b["means"][sg > 0, j].mean(0) - b["means"][sg < 0, j].mean(0)
        out[str(layer)] = round(cos(da, db), 6)
    return out


# ------------------------------------------------------------------------------------------------ the cheap tier


def cheap_reference(out, analysis_id: str) -> Path:
    """Where the analysis with this id put the cheap tier's reference values."""
    return Path(out) / "cheap" / analysis_id / "reference.pt"


def run_cheap(model, tok, plan: dict, out, data: dict, analysis_id: str, guard: T.Guard | None = None) -> dict:
    """The cheap tier's item: the stored subset of G (the first kept sample of each (question, instruction) cell, per
    sign, as the analysis with this id recorded it) read again with probe.response_activations, and its direction
    compared with this run's P1 on the same subset (every block) and with R-lab32's direction from the same subset at
    the graded block (the reference file carries it, so this reads nothing from the analysis plan)."""
    import torch

    from anatta.probe import response_activations

    ref = torch.load(T.require(cheap_reference(out, analysis_id)), weights_only=False)
    blocks = ref["blocks"]
    res = {}
    for t in plan["traits"]:
        sums = {}
        for sign in T.SIGNS:
            rows = {x["row"]: x for x in G.load_rows(out, "G", t, sign, plan)}
            acc = []
            for k in ref["rows"][t]:
                if guard is not None:
                    guard.row = k
                x = rows[k]
                system = T.system_strings(t, data[t]["instruction"][x["ii"]], plan["prompts"])[sign]
                acts = response_activations(model, tok, messages(system, data[t]["questions"][x["qi"]]), x["answer"],
                                            blocks)
                acc.append(stack(acts, blocks))
            sums[sign] = torch.stack(acc).mean(0)
        now = sums["pos"] - sums["neg"]
        c_p1 = [cos(now[j], ref["p1"][t][j]) for j in range(len(blocks))]
        j19 = blocks.index(ref["block"])
        c_r = cos(now[j19], ref["rlab32"][t])
        c = plan["cheap_tier"]
        res[t] = {"p1_min": round(min(c_p1), 6), "rlab32_block": round(c_r, 6),
                  "passed": min(c_p1) >= c["p1_floor"] and c_r >= c["rlab32_floor"]}
    return {"traits": res, "passed": all(v["passed"] for v in res.values())}
