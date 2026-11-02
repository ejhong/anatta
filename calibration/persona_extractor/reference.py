"""Calibration check 4: the reference (what runs in the containers; no Modal here, so the tests run it on a laptop).

  R-lab32     the vendored get_hidden_p_and_r (third_party/persona_vectors/generate_vec.py at b8e0f04, its git blob sha
              checked) on Qwen loaded by anatta.models.load(dtype="float32"), inside an outer torch.no_grad(), over
              G's kept rows; the difference arithmetic copied from save_persona_vector. The function is taken from the
              file's own syntax tree, since the file imports pandas at module level and the lab's image has none.
              Per-response means are kept (the bootstraps and the reference's floor need them).
  the CSVs    for the second image: G's kept rows as the repo's CSVs (question, prompt, answer, question_id, the trait,
              coherence; row order question -> instruction -> sample), written and read back with the image's own
              pandas. A pair is dropped from G for every extractor when either answer fails the round trip byte for
              byte (read_csv turns "", "None", "NA" and the like into NaN), or has an empty answer slice under either
              stack's tokenizer (the script's mean over it would be NaN). csv_selftest runs first in the image.
  R           the verbatim script, run as scripts/generate_vec.sh runs it, on the CSVs, in the second image; its
              output and log stay on the volume, and the peak GPU memory is polled while it runs.
  the judged arm
              (conditional) G's kept responses scored through OpenRouter with the repo's eval_prompt and
              coherence_0_100 prompts, the scores written as the CSVs' columns, and the verbatim script run on them.
Pandas, transformers and torch are imported inside the functions, so this file imports under both stacks.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import generate as G
import traits as T

KINDS = ("prompt_avg", "prompt_last", "response_avg")  # save_persona_vector's three outputs, in its order
COLUMNS = ("question", "prompt", "answer", "question_id")  # eval_persona.py's CSV columns, then the scores


# ------------------------------------------------------------------------------------------------ the vendored file


def vendored_source(path, blob: str) -> str:
    data = Path(path).read_bytes()
    got = T.git_blob_sha(data)
    if got != blob:
        raise T.CheckError("vendored", got=got)
    return data.decode("utf-8")


def vendored_functions(path, blob: str, names=("get_hidden_p_and_r",)) -> dict:
    """The named top-level functions of the vendored generate_vec.py (blob sha checked), compiled from the file's own
    syntax tree with its line numbers, into a namespace holding torch and tqdm (what their bodies use). The module's
    other statements, the pandas import among them, are not run."""
    src = vendored_source(path, blob)
    defs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if sorted(d.name for d in defs) != sorted(names):
        raise T.CheckError("vendored")
    import torch

    try:
        from tqdm import tqdm
    except ImportError:  # the progress bar is all tqdm does here
        def tqdm(it, **_):
            return it
    ns = {"torch": torch, "tqdm": tqdm, "__name__": "generate_vec_vendored"}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102 (the checked file)
    return {n: ns[n] for n in names}


def avg_diff(pos: list, neg: list):
    """save_persona_vector's difference arithmetic (generate_vec.py:70-72 at b8e0f04, the same expression on each of
    its three outputs), copied: for every index l, pos[l].mean(0).float() - neg[l].mean(0).float(), stacked."""
    import torch

    return torch.stack([pos[l].mean(0).float() - neg[l].mean(0).float() for l in range(len(pos))], dim=0)  # noqa: E741


def rlab32(model, tok, prompts: list[str], answers: list[str], fn) -> tuple:
    """fn (the vendored get_hidden_p_and_r) inside an outer torch.no_grad(): the values don't change, the graph just
    isn't kept. Returns its (prompt_avg, prompt_last, response_avg), each a list over hidden_states indices of [n, d]."""
    import torch

    with torch.no_grad():
        return fn(model, tok, prompts, answers)


def rlab32_dir(out, trait: str) -> Path:
    return Path(out) / "rlab32" / trait


def run_rlab32(model, tok, plan: dict, out, keep: dict, fn, commit=lambda: None, log=print,
               guard: T.Guard | None = None) -> dict:
    """R-lab32 for each trait not yet on the volume: the vendored function over G's kept rows of each sign (the
    stored prompt, which is the CSV's prompt column, and the answer), the three differences by save_persona_vector's
    arithmetic ({trait}_{kind}_diff.pt, [layers + 1, d]) and the per-response means of the response slice
    (means_{sign}.pt: rows and [n, layers + 1, d] fp32). Returns shapes, norms and sha256s."""
    import torch
    from readings import save_tensor

    res = {}
    for ti, t in enumerate(plan["traits"]):
        if guard is not None:
            guard.row = ti
        d = rlab32_dir(out, t)
        done = d / "done.json"
        if not done.exists():
            T.before(commit, "rlab32_trait")
            outs = {}
            for sign in T.SIGNS:
                rows = [x for x in G.load_rows(out, "G", t, sign, plan) if x["row"] in keep[t]]
                pa, pl, ra = rlab32(model, tok, [x["prompt"] for x in rows], [x["answer"] for x in rows], fn)
                outs[sign] = (pa, pl, ra)
                save_tensor(d / f"means_{sign}.pt", {"rows": [x["row"] for x in rows],
                                                     "means": torch.stack(list(ra), dim=1).float()})
            files = {}
            for j, kind in enumerate(KINDS):
                v = avg_diff(outs["pos"][j], outs["neg"][j])
                p = d / f"{t}_{kind}_diff.pt"
                save_tensor(p, v)
                files[kind] = {"sha256": T.sha256_file(p), "shape": list(v.shape)}
            T.write_json(done, {"files": files, "rows": len(keep[t])})
            commit()
            log(f"R-lab32 {t}: done")
        res[t] = T.read_json(done)
    return res


# ------------------------------------------------------------------------------------------------ the CSVs


def csv_records(pairs: list[tuple[dict, dict]], trait: str, data: dict, scores: dict | None) -> dict:
    """The CSV rows of each sign, in row order: question, prompt (the stored rendering), answer, question_id, the trait
    score and coherence. scores is either {"constant": {pos, neg, coherence}}, the registered constants run_csv passes
    from plan["reference"]["scores"] (pos 100, neg 0, coherence 100), with which the script's mask keeps every row;
    or the judged arm's {sign: {row: (trait score, coherence)}}, None for a reply the judge left unscored."""
    out = {}
    for j, sign in enumerate(T.SIGNS):
        recs = []
        for pair in pairs:
            x = pair[j]
            s, c = (scores["constant"][sign], scores["constant"]["coherence"]) if "constant" in scores \
                else scores[sign][x["row"]]
            recs.append({"question": data["questions"][x["qi"]], "prompt": x["prompt"], "answer": x["answer"],
                         "question_id": f"{trait}_{x['qi']}", trait: s, "coherence": c})
        out[sign] = recs
    return out


def write_csv(pd, path, records: list[dict], trait: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=[*COLUMNS, trait, "coherence"]).to_csv(path, index=False)


def survives(pd, path, records: list[dict]) -> list[bool]:
    """Each row's answer and prompt read back by pd.read_csv(path), as generate_vec reads it: True when both come back
    as the same string, byte for byte."""
    df = pd.read_csv(path)
    if len(df) != len(records):
        raise T.CheckError("csv", rows=len(df), want=len(records))
    a, p = df["answer"].tolist(), df["prompt"].tolist()
    return [isinstance(a[i], str) and a[i].encode() == r["answer"].encode() and isinstance(p[i], str)
            and p[i].encode() == r["prompt"].encode() for i, r in enumerate(records)]


SELFTEST_KEEP = ("plain words", " a leading space", "\na leading newline", "café naïve ünïcode",
                 'a "quoted" word, and a comma', "two\nlines", "a\ttab")
SELFTEST_DROP = ("", "None", "NA")  # the registration's examples of answers read_csv turns into NaN
SELFTEST_OTHER = ("nan", "null", "N/A", "<NA>", "carriage\r\nreturn", "  ")


def csv_selftest(pd, tmp) -> dict:
    """The CSV writer in this image, on made-up answers: the ordinary ones must come back byte for byte and the three
    the registration names ("", "None", "NA") must be flagged. Returns counts; CheckError("csv_selftest") otherwise."""
    answers = [*SELFTEST_KEEP, *SELFTEST_DROP, *SELFTEST_OTHER]
    recs = [{"question": "q", "prompt": "a prompt\nover two lines", "answer": a, "question_id": "t_0", "t": 100,
             "coherence": 100} for a in answers]
    path = Path(tmp) / "selftest.csv"
    write_csv(pd, path, recs, "t")
    ok = survives(pd, path, recs)
    n_keep, n_drop = len(SELFTEST_KEEP), len(SELFTEST_DROP)
    if not all(ok[:n_keep]) or any(ok[n_keep:n_keep + n_drop]):
        raise T.CheckError("csv_selftest", kept=sum(ok[:n_keep]), flagged=n_drop - sum(ok[n_keep:n_keep + n_drop]))
    return {"ordinary_kept": n_keep, "named_flagged": n_drop, "other_flagged": len(SELFTEST_OTHER)
            - sum(ok[n_keep + n_drop:]), "other": len(SELFTEST_OTHER)}


def slice_size(tok, prompt: str, answer: str) -> int:
    """generate_vec's response slice for one row under this tokenizer: len(tokenize(p + a)) - len(encode(p))."""
    return len(tok(prompt + answer, add_special_tokens=False)["input_ids"]) - len(tok.encode(prompt,
                                                                                           add_special_tokens=False))


def csv_dir(out, judged: bool = False) -> Path:
    return Path(out) / ("judged" if judged else "") / "csv"


def run_csv(plan: dict, out, data: dict, tok, pd, tmp, scores: dict | None = None, guard: T.Guard | None = None) -> dict:
    """The CSV step, in the second image (its pandas, its tokenizer): the self-test; then, for each trait, every pair
    checked (the prompt re-rendered by this stack's template equals the stored one, or CheckError("render"); both
    answers' slices non-empty under this tokenizer and under the lab's; both answers survive the round trip), the
    failing pairs dropped and counted, the kept pairs written to the volume as the trait's pos and neg CSVs and read
    back once more (CheckError("csv") if anything fails then). With scores (the judged arm), the CSVs carry them and go
    to judged/csv, and the kept rows are judged/keep.json's. Returns counts and the CSVs' sha256s."""
    judged = scores is not None
    selftest = csv_selftest(pd, tmp)
    kept, counts = {}, {}
    base = None if not judged else G.keep_rows(out)
    for ti, t in enumerate(plan["traits"]):
        if guard is not None:
            guard.row = ti
        pairs = G.pairs_of(out, plan, t)
        if base is not None:
            pairs = [pr for pr in pairs if pr[0]["row"] in base[t]]
        for pr in pairs:
            for j, sign in enumerate(T.SIGNS):
                x = pr[j]
                system = T.system_strings(t, data[t]["instruction"][x["ii"]], plan["prompts"])[sign]
                if G.render(tok, system, data[t]["questions"][x["qi"]]) != x["prompt"]:
                    raise T.CheckError("render", row=x["row"])
        bad_lab = {p["row"] for p, n in pairs if p["slice_tokens"] < 1 or n["slice_tokens"] < 1}
        bad_repo = {p["row"] for p, n in pairs if min(slice_size(tok, x["prompt"], x["answer"]) for x in (p, n)) < 1}
        recs = csv_records(pairs, t, data[t], scores[t] if judged else {"constant": plan["reference"]["scores"]})
        flags = {}
        for sign in T.SIGNS:
            path = Path(tmp) / f"{t}_{sign}_all.csv"
            write_csv(pd, path, recs[sign], t)
            flags[sign] = survives(pd, path, recs[sign])
        bad_csv = {pairs[i][0]["row"] for i in range(len(pairs)) if not (flags["pos"][i] and flags["neg"][i])}
        drop = bad_lab | bad_repo | bad_csv
        keep_idx = [i for i, pr in enumerate(pairs) if pr[0]["row"] not in drop]
        kept[t] = [pairs[i][0]["row"] for i in keep_idx]
        shas = {}
        for sign in T.SIGNS:
            path = csv_dir(out, judged) / f"{t}_{sign}.csv"
            final = [recs[sign][i] for i in keep_idx]
            write_csv(pd, path, final, t)
            if not all(survives(pd, path, final)):
                raise T.CheckError("csv")
            shas[sign] = T.sha256_file(path)
        counts[t] = {"pairs": len(pairs), "kept": len(keep_idx), "csv_sha256": shas,
                     "dropped": {"csv": len(bad_csv), "slice_repo": len(bad_repo), "slice_lab": len(bad_lab),
                                 "any": len(drop)}}
    T.write_json(G.keep_file(out, judged), {"kept": kept, "counts": counts})
    return {"traits": counts, "selftest": selftest, "csv": True, "judged": judged}


# ------------------------------------------------------------------------------------------------ the verbatim script


def long_answer(tok, sentence: str, n: int) -> str:
    """A benign made-up sentence repeated and cut to n tokens under this tokenizer (the smoke's long rows: the longest
    answer G can produce). CheckError("long_answer") if the cut text does not encode back to n tokens."""
    k = 1
    while len(tok.encode(sentence * k, add_special_tokens=False)) < n + 16:
        k *= 2
    ids = tok.encode(sentence * k, add_special_tokens=False)
    for cut in sorted(range(n - 8, n + 9), key=lambda c: abs(c - n)):  # a cut mid-word may re-encode differently
        text = tok.decode(ids[:cut])
        if len(tok.encode(text, add_special_tokens=False)) == n:
            return text
    raise T.CheckError("long_answer", want=n)


def with_long_rows(pd, src_dir: Path, dst_dir: Path, trait: str, answer: str) -> dict:
    """The trait's smoke CSVs with one synthetic pair appended (each sign's first prompt and the long answer), written
    to dst_dir. Returns their paths."""
    paths = {}
    for sign in T.SIGNS:
        df = pd.read_csv(src_dir / f"{trait}_{sign}.csv")
        row = df.iloc[[0]].copy()
        row["answer"] = answer
        row["question_id"] = f"{trait}_long"
        dst_dir.mkdir(parents=True, exist_ok=True)
        paths[sign] = dst_dir / f"{trait}_{sign}.csv"
        pd.concat([df, row], ignore_index=True).to_csv(paths[sign], index=False)
    return paths


def smi_reading() -> tuple[int, int]:
    """nvidia-smi's used and total memory of the first GPU, in MiB (device-wide, so the child's use is in it)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10, check=True).stdout.split("\n")[0]
    used, total = (int(float(v)) for v in q.split(","))
    return used, total


def cuda_reading() -> tuple[int, int]:
    """torch.cuda.mem_get_info() from this (the parent) process, in MiB: the device's used memory (total less free,
    so the child's use is in it, with this process's own small context) and its total."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 2**20), round(total / 2**20)


class PeakMemory:
    """Polls the GPU's used memory (MiB) every half second while the verbatim script runs: nvidia-smi's reading, or,
    where nvidia-smi is missing or fails, torch.cuda's from the parent process (`source` says which). check() after
    the run: a GPU present and a peak of 0 is a failed measurement, not a fit (CheckError("peak_memory"))."""

    def __init__(self, every: float = 0.5, readers=None):
        self.every, self.peak, self.total, self.source = every, 0, None, None
        self.readers = readers or (("nvidia-smi", smi_reading), ("torch.cuda", cuda_reading))
        self._stop = threading.Event()

    def read_once(self) -> None:
        for name, reader in self.readers:
            try:
                used, total = reader()
            except Exception:  # noqa: BLE001 (this reader is missing or failed: try the next)
                continue
            self.peak, self.total, self.source = max(self.peak, used), total, name
            return

    def _poll(self):
        while not self._stop.is_set():
            self.read_once()
            self._stop.wait(self.every)

    def check(self, gpu: str | None) -> None:
        if gpu is not None and self.peak == 0:
            raise T.CheckError("peak_memory")

    def __enter__(self):
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=15)
        return False


# tqdm's closing line of a bar without a description: with leave on (the default) it redraws the bar at its overall
# rate, the rows over the elapsed seconds, then ends the line: "100%|#####| 5/5 [00:01<00:00,  3.10it/s]", or "1.52s/it"
# under one row a second.
CLOSED = re.compile(rb"\s*100%\|[^|]*\|\s*(\d+)/(\d+) \[[^<\]]*<[^,\]]*,\s*(\d+\.\d+)(it/s|s/it)\]")


def pass_seconds(data: bytes, total: int) -> list[float]:
    """The seconds of each pass over `total` rows that the verbatim script finished, in order (its pos rows, then its
    neg rows), from tqdm's closing line of each bar in its stderr: the rows over the overall rate it prints. Neither the
    model load before the passes nor the save after them is in them. Transformers' "Loading checkpoint shards" bar has
    a description and doesn't match."""
    out = []
    for line in data.split(b"\n")[:-1]:  # ended lines only: tqdm ends a bar's line after its closing redraw
        m = CLOSED.match(line.split(b"\r")[-1])
        if m is not None and int(m.group(1)) == int(m.group(2)) == total:
            rate = float(m.group(3))
            out.append(total / rate if m.group(4) == b"it/s" else total * rate)
    return out


def csv_size(pd, tok, paths: dict, trait: str, threshold: int) -> dict:
    """What the verbatim script reads from a trait's CSVs: the pairs its mask keeps (get_persona_effective's) and their
    rows' tokens, prompt + answer as it tokenizes them (no special tokens). Numbers only: R's upper estimate under the
    hard ceiling and the smoke's seconds a row and a token use them; neg_tokens are the second pass's."""
    pos, neg = pd.read_csv(paths["pos"]), pd.read_csv(paths["neg"])
    mask = ((pos[trait] >= threshold) & (neg[trait] < 100 - threshold) & (pos["coherence"] >= 50)
            & (neg["coherence"] >= 50))
    n = {s: [len(tok(p + a, add_special_tokens=False)["input_ids"])
             for p, a in zip(f["prompt"].tolist(), f["answer"].tolist())]
         for s, f in (("pos", pos[mask]), ("neg", neg[mask]))}
    return {"rows": len(n["pos"]) + len(n["neg"]), "pairs": len(n["neg"]), "tokens": sum(n["pos"]) + sum(n["neg"]),
            "neg_tokens": sum(n["neg"])}


def verbatim_command(script, snapshot, pos, neg, trait: str, save_dir, threshold: int) -> list[str]:
    """The call scripts/generate_vec.sh makes."""
    return [sys.executable, str(script), "--model_name", str(snapshot), "--pos_path", str(pos), "--neg_path",
            str(neg), "--trait", trait, "--save_dir", str(save_dir), "--threshold", str(threshold)]


def verbatim_process(cmd: list[str], stdout, stderr, env: dict) -> int:
    """The verbatim script as a subprocess, its output to the given files. Returns the exit code."""
    return subprocess.run(cmd, stdout=stdout, stderr=stderr, env=env).returncode  # noqa: S603 (a fixed command)


def verbatim_dir(out, trait: str, judged: bool = False) -> Path:
    return Path(out) / ("judged" if judged else "") / "verbatim" / trait


def run_verbatim(plan: dict, out, script, snapshot, tok, pd, judged: bool = False, commit=lambda: None, log=print,
                 guard: T.Guard | None = None) -> dict:
    """The verbatim script for each trait not yet on the volume, as a subprocess with the image's Python, on the
    trait's CSVs (the smoke's with one synthetic long pair appended), its stdout and stderr written to the volume and
    never printed; before each trait, the hard ceiling's check with the trait's rows and tokens. Each output must have
    shape [layers + 1, hidden]. Returns, per trait, the exit code, seconds, the rows and tokens it read, the seconds of
    its two row passes as tqdm timed them (pass_seconds; None where its stderr doesn't show both), the peak GPU memory
    while it ran, the GPU and the outputs' sha256s."""
    import torch

    m = plan["models"]["qwen"]
    want = [m["layers"] + 1, m["hidden"]]
    res = {}
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    for ti, t in enumerate(plan["traits"]):
        if guard is not None:
            guard.row = ti
        d = verbatim_dir(out, t, judged)
        done = d / "done.json"
        if not done.exists():
            src = csv_dir(out, judged)
            paths = {s: src / f"{t}_{s}.csv" for s in T.SIGNS}
            long_tokens = None
            if plan["smoke"] and not judged:
                s = plan["smoke_settings"]
                # Ruled (NOTES, 12:20 UTC, ruling 9): one long pair joins every trait's smoke CSVs.
                text = long_answer(tok, s["sentence"], s["long_tokens"])
                paths = with_long_rows(pd, src, d / "csv", t, text)
                long_tokens = slice_size(tok, pd.read_csv(paths["pos"])["prompt"].tolist()[-1], text)
            size = csv_size(pd, tok, paths, t, plan["reference"]["threshold"])
            T.before(commit, "R_trait", rows=size["rows"], tokens=size["tokens"])
            d.mkdir(parents=True, exist_ok=True)
            cmd = verbatim_command(script, snapshot, paths["pos"], paths["neg"], t, d, plan["reference"]["threshold"])
            t0 = time.time()
            with open(d / "stdout.log", "w") as so, open(d / "stderr.log", "w") as se, PeakMemory() as peak:
                code = verbatim_process(cmd, so, se, env)
            seconds = time.time() - t0
            if code != 0:
                raise T.CheckError("verbatim", code=code, peak_mib=peak.peak)
            peak.check(T.gpu_name())
            files = {}
            for kind in KINDS:
                p = d / f"{t}_{kind}_diff.pt"
                v = torch.load(T.require(p), weights_only=False)
                if list(v.shape) != want:
                    raise T.CheckError("shape", rows=int(v.shape[0]), cols=int(v.shape[-1]))
                files[kind] = {"sha256": T.sha256_file(p), "shape": list(v.shape)}
            passes = pass_seconds((d / "stderr.log").read_bytes(), size["pairs"])
            T.write_json(done, {"exit": code, "seconds": round(seconds, 1), "peak_mib": peak.peak,
                                "total_mib": peak.total, "peak_source": peak.source, "gpu": T.gpu_name(),
                                "files": files, "long_answer_tokens": long_tokens, "rows": size["rows"],
                                "tokens": size["tokens"], "neg_tokens": size["neg_tokens"],
                                "pass_seconds": [round(x, 2) for x in passes] if len(passes) == 2 else None})
            commit()
            log(f"R {t}: done in {round(seconds)} s, peak {peak.peak} MiB")
        res[t] = T.read_json(done)
    return res


# ------------------------------------------------------------------------------------------------ the judged arm


FORMS = ("assign", "subscript", "dict")  # name = "...", X["name"] = "...", X = {"name": "..."}


def named_string_form(src: str, name: str) -> tuple[str, str]:
    """A string constant the repo's eval/prompts.py binds to `name`, and the form that binds it (FORMS): `name =
    "..."`, `X["name"] = "..."` or a top-level dict literal with the key. CheckError("judge") if there is none."""
    # Ruled (NOTES, 12:20 UTC, ruling 10): the accepted forms only; prepare records the one that matched.
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                key, form = tgt.id, "assign"
            elif isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant):
                key, form = tgt.slice.value, "subscript"
            else:
                continue
            if key == name and isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value, form
        if isinstance(value, ast.Dict):
            for k, v in zip(value.keys, value.values):
                if isinstance(k, ast.Constant) and k.value == name and isinstance(v, ast.Constant) \
                        and isinstance(v.value, str):
                    return v.value, "dict"
    raise T.CheckError("judge")


def named_string(src: str, name: str) -> str:
    return named_string_form(src, name)[0]


COHERENCE = "coherence_0_100"
PROMPTS_FILE = "eval_prompts.py"  # the repo's eval/prompts.py, as prepare writes it to the volume


def prompts_url(repo: dict, spec: dict) -> str:
    return f"{repo['raw']}/{repo['commit']}/{spec['file']}"


def prepare_prompts(repo: dict, spec: dict, data_dir, fetch=T.fetch_url) -> dict:
    """prepare, on Modal (ruling 10): the repo's eval/prompts.py fetched at the commit, its git blob sha and size
    checked like a trait file's (CheckError("blob")), coherence_0_100 found (named_string_form), and the file written
    to data_dir on the volume. Returns {bytes, blob, sha256, form}, which prepare writes into config.yaml; the text
    never leaves the volume."""
    data = fetch(prompts_url(repo, spec))
    blob = T.git_blob_sha(data)
    if blob != spec["blob"] or len(data) != spec["bytes"]:
        raise T.CheckError("blob", got=blob, bytes=len(data))
    _, form = named_string_form(data.decode("utf-8"), COHERENCE)
    p = Path(data_dir) / PROMPTS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"bytes": len(data), "blob": blob, "sha256": T.sha256_bytes(data), "form": form}


def read_prompts(plan: dict) -> str:
    """The judged arm's coherence_0_100, from the volume only (where prepare put it): the file's sha256 and the form
    that binds the string must be the registered ones (CheckError("prompts"))."""
    spec = plan["judged"]["prompts"]
    data = T.require(Path(plan["data_dir"]) / PROMPTS_FILE).read_bytes()
    if not spec.get("sha256") or T.sha256_bytes(data) != spec["sha256"]:
        raise T.CheckError("prompts")
    text, form = named_string_form(data.decode("utf-8"), COHERENCE)
    if form != spec["form"]:
        raise T.CheckError("prompts")
    return text


def parse_number(text) -> int | None:
    """The judge's reply as a score: a bare integer from 0 to 100, else None (the row's pair then fails the script's
    mask, as an unscored row does in the repo)."""
    m = re.fullmatch(r"\s*(\d{1,3})\s*\.?\s*", text or "")
    return int(m.group(1)) if m and int(m.group(1)) <= 100 else None


def ask(cfg: dict, key: str, prompt: str, post=None, tries: int = 4) -> dict:
    """One judge call through OpenRouter at temperature 0. Returns {score, snapshot, tokens_in, tokens_out}; a call
    that keeps failing returns score None. The prompt and the reply never leave this function."""
    body = json.dumps({"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
                       "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"], "seed": 0}).encode()

    def default_post():
        req = urllib.request.Request(cfg["url"], data=body, headers={"Authorization": f"Bearer {key}",
                                                                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (the registered https URL)
            return json.loads(r.read())

    for i in range(tries):
        try:
            obj = default_post() if post is None else post(body)
            usage = obj.get("usage") or {}
            return {"score": parse_number(obj["choices"][0]["message"]["content"]), "snapshot": obj.get("model"),
                    "tokens_in": usage.get("prompt_tokens", 0), "tokens_out": usage.get("completion_tokens", 0)}
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError):
            time.sleep(2 ** i)
    return {"score": None, "snapshot": None, "tokens_in": 0, "tokens_out": 0}


def run_judge(plan: dict, out, data: dict, coherence: str, key: str, post=None, log=print,
              guard: T.Guard | None = None) -> dict:
    """The judged arm's scores: each kept G response scored for its trait (the trait file's eval_prompt) and for
    coherence (coherence_0_100), both filled with the question and the answer, through OpenRouter. Scores go to the
    volume (judged/scores/<trait>_<sign>.jsonl: row, trait score, coherence, the snapshot that answered), resumably.
    Returns counts, the snapshots seen, the tokens used, and the sha256 of the coherence template and of each trait's
    eval_prompt, unfilled (ruling 11)."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = plan["judged"]
    keep = G.keep_rows(out)
    res, tokens = {}, {"in": 0, "out": 0}
    for ti, t in enumerate(plan["traits"]):
        if guard is not None:
            guard.row = ti
        res[t] = {}
        for sign in T.SIGNS:
            path = Path(out) / "judged" / "scores" / f"{t}_{sign}.jsonl"
            have = {r["row"] for r in T.read_jsonl(path)}
            rows = [x for x in G.load_rows(out, "G", t, sign, plan) if x["row"] in keep[t] and x["row"] not in have]

            def score(x, t=t):
                q = data[t]["questions"][x["qi"]]
                # Ruled (NOTES, 12:20 UTC, ruling 11): str.format on {question} and {answer}, checked after prepare.
                a = ask(cfg, key, data[t]["eval_prompt"].format(question=q, answer=x["answer"]), post)
                b = ask(cfg, key, coherence.format(question=q, answer=x["answer"]), post)
                return x["row"], a, b

            with ThreadPoolExecutor(cfg["workers"]) as pool:
                for row, a, b in pool.map(score, rows):
                    T.append_jsonl(path, {"row": row, "trait": a["score"], "coherence": b["score"],
                                          "snapshot": a["snapshot"] or b["snapshot"]})
                    tokens["in"] += a["tokens_in"] + b["tokens_in"]
                    tokens["out"] += a["tokens_out"] + b["tokens_out"]
            got = T.read_jsonl(path)
            res[t][sign] = {"scored": len(got), "unscored_trait": sum(r["trait"] is None for r in got),
                            "unscored_coherence": sum(r["coherence"] is None for r in got),
                            "snapshots": sorted({r["snapshot"] for r in got if r["snapshot"]})}
            log(f"judged {t} {sign}: {len(got)} scored")
    shas = {"coherence_0_100": T.sha256_bytes(coherence.encode()),
            "eval_prompt": {t: T.sha256_bytes(data[t]["eval_prompt"].encode()) for t in plan["traits"]}}
    return {"traits": res, "tokens": tokens, "prompt_sha256": shas}


def judged_scores(out, plan: dict) -> dict:
    """The judged arm's scores as run_csv takes them: {sign: {row: (trait score, coherence)}}, per trait."""
    out_scores = {}
    for t in plan["traits"]:
        out_scores[t] = {sign: {r["row"]: (r["trait"], r["coherence"]) for r in T.read_jsonl(
            T.require(Path(out) / "judged" / "scores" / f"{t}_{sign}.jsonl"))} for sign in T.SIGNS}
    return out_scores
