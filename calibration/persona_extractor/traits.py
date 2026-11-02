"""Calibration check 4: the persona-vectors repo's trait files, the run's data layout, and the small helpers every
stage shares. Standard library only at import, since the second image (the repo's stack) imports this file too.

  the trait files  data_generation/trait_data_extract/<trait>.json at b8e0f04: exactly the keys instruction (5
                   {pos, neg} objects), questions (20) and eval_prompt. prepare fetches them on Modal (never on the
                   laptop), checks each git blob sha and size against the registration and counts the pairs and
                   questions. probe_trait turns one into an anatta.probe.Trait whose pairs are eval_persona.py's two
                   system strings, so probe.extract generates under exactly the repo's prompts.
  the layout       G's rows run question -> instruction -> sample, as the repo's CSVs do, so row k of a trait's pos
                   file and row k of its neg file are one pair (cells, row_of). G's batches hold up to 200 rows of one
                   sign, numbered over all of G (batches); batch i is seeded 1000 + i.
  the layers       generate_vec's hidden_states[l] is block l - 1's output (hs_index), so the repo's "layer 20" is
                   probe's block 19.
  the stages       run_guarded: any exception inside a stage comes out as a StageError naming only the stage, the row
                   index and the exception's type (and a CheckError's code and numbers), raised after the failing
                   frame is gone, so no completion or prompt reaches Modal's logs through a traceback or a message.
                   written: a stage's result, checked text-free (names, numbers, hashes, paths) before it is written
                   to the volume and returned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from importlib import metadata
from pathlib import Path

REPO_KEYS = ("instruction", "questions", "eval_prompt")
SIGNS = ("pos", "neg")
GENERATE_VEC = "third_party/persona_vectors/generate_vec.py"
# The data code: what the GPU and CSV stages run, hashed into the plan (and so into the plan id and the run's folder).
DATA_FILES = ("traits.py", "generate.py", "readings.py", "reference.py", GENERATE_VEC)
# The analysis code that runs in a container, hashed into the analysis plan (its own id) with card.py, which runs on the
# laptop: a fix to either reruns only the analysis, from the stored data.
ANALYSIS_FILES = ("analysis.py",)
# Both images carry all of these (app.py imports every module in every container), and the library at /root/anatta,
# since app.py imports anatta.modal_app.
IMAGE_FILES = DATA_FILES + ANALYSIS_FILES
LIBRARY_FILES = ("probe.py", "models.py", "factor.py", "modal_app.py")  # the instrument, as the registration lists it

# ------------------------------------------------------------------------------------------------ errors

CODES = {  # every CheckError code, with what it means (the laptop prints these; a container passes on only the code)
    "blob": "a trait file's git blob sha or size is not the registered one",
    "keys": "a trait file holds other keys than instruction, questions and eval_prompt",
    "pairs": "a trait file's instruction list is not the registered five {pos, neg} pairs",
    "questions": "a trait file's question list is not the registered twenty",
    "eval_prompt": "a trait file's eval_prompt is not a string",
    "sha256": "a trait file's sha256 or counts differ from the ones registered in config.yaml",
    "unregistered": "a trait file's sha256 is not yet registered in config.yaml: run prepare and commit first",
    "missing": "a file the stage needs is not on the volume: run the earlier stage first",
    "plan": "the run folder holds another plan's files",
    "env_code": "this container's code differs from the plan's: launch again, which plans afresh",
    "env_pins": "installed versions differ from the image's pins",
    "stored": "a stored direction differs from the one the plan names",
    "generation_config": "the model's generation_config is not the registered one",
    "processors": "the logits processors transformers built are not the registered ones",
    "slice_start": "probe's slice start differs from generate_vec's prompt length",
    "slice_prefix": "the template's rendering of the reply does not extend the prompt followed by the answer",
    "render": "the chat template renders a prompt differently from the one stored with the response",
    "vendored": "the vendored generate_vec.py's git blob sha is not the registered one, or it lacks a function",
    "empty_slice": "a kept response has an empty answer slice",
    "alignment": "the recorder's records do not line up with probe.extract's jobs",
    "csv": "the CSV round trip failed after the failing pairs were dropped",
    "csv_selftest": "the CSV writer's self-test failed in the second image",
    "long_answer": "the synthetic long answer could not be made the registered length",
    "verbatim": "the verbatim script exited with an error",
    "shape": "a vector or a stored file has another shape than expected",
    "free_text": "a stage's result holds a string that is not a name, a number, a hash or a path",
    "judge": "the judged arm's replies could not be scored",
    "budget": "the container ran past the seconds its share of the budget buys (the GPU and CSV stages stop at the $20 "
              "stop less the analysis reserve); its work so far is on the volume",
    "ceiling": "the next unit's upper estimate would carry the spend past the limit (the hard ceiling: $19.50 for the "
               "GPU and CSV stages), so it didn't start; the work so far is on the volume",
    "peak_memory": "a GPU is present but the peak memory read 0: the measurement failed",
    "prompts": "eval/prompts.py on the volume differs from the registered sha256 or form, or lacks coherence_0_100",
}
HEX = re.compile(r"[0-9a-f]{6,64}")


def _plain(v) -> bool:
    """A value a CheckError may carry into a log: a number, a bool, None or a hex hash."""
    return v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and HEX.fullmatch(v) is not None)


class CheckError(RuntimeError):
    """A check that stops the run. Its code is one of CODES, never data; its facts are numbers or hashes only (any
    other value is dropped), so a guard can pass both on."""

    def __init__(self, code: str, **facts):
        if code not in CODES:
            raise ValueError(f"unknown CheckError code {code!r}")
        self.code = code
        self.facts = {k: v for k, v in facts.items() if _plain(v)}
        super().__init__(f"{code} {json.dumps(self.facts, sort_keys=True)}" if self.facts else code)


class StageError(RuntimeError):
    """What a guarded stage raises in place of any exception: the stage, the row index and the exception's type."""


class Guard:
    """Holds the stage's name and the row it is on; stages set guard.row as they go."""

    def __init__(self, stage: str):
        self.stage, self.row, self.summary = stage, None, None

    def __enter__(self):
        return self

    def __exit__(self, etype, exc, tb):
        if exc is None or not isinstance(exc, Exception):
            return False
        self.summary = describe(exc)  # the exception itself, its message and its frames are dropped here
        return True

    def error(self) -> StageError:
        return StageError(f"{self.stage}: row {self.row}: {self.summary}")


def describe(exc: Exception) -> str:
    """An exception as text-free words: its type's name, and a CheckError's code and facts."""
    if isinstance(exc, CheckError):
        return f"CheckError {exc.code}" + (f" {json.dumps(exc.facts, sort_keys=True)}" if exc.facts else "")
    return type(exc).__name__


def error_name(exc: Exception) -> str:
    """An exception as one text-free token for a stage's result: its type's name, or CheckError:<code>."""
    return f"CheckError:{exc.code}" if isinstance(exc, CheckError) else type(exc).__name__


def before(commit, kind: str, **facts) -> None:
    """The hard ceiling's check before a commit unit of `kind` ("G_batch", "R_trait", ...), when the commit carries one
    (app.py's Ceiling, which raises CheckError("ceiling")); a plain commit function, as in the tests, has none."""
    check = getattr(commit, "before", None)
    if check is not None:
        check(kind, **facts)


def run_guarded(stage: str, fn, *args, **kwargs):
    """fn(*args, guard=Guard(stage), **kwargs). Any exception becomes a StageError raised here, after the with block
    has swallowed the original, so the new error has no __context__ and no frame of the failing code."""
    guard = Guard(stage)
    with guard:
        return fn(*args, guard=guard, **kwargs)
    raise guard.error()


# ------------------------------------------------------------------------------------------------ files
# sha256_bytes, sha256_file, write_json, written, read_json, append_jsonl, version_of, claim and check_environment follow
# check 7 (anatta-calib-c7, calibration/relearning/relearn.py at 02db868); snapshot_dir is check 1's
# (anatta-calib-c1, src/anatta/bench/run.py at 44760f7), the loader its app.py:171-185 calls.


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha(data: bytes) -> str:
    """The git blob sha of a file's bytes: sha1 of "blob <size>\\0" followed by the bytes."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


SAFE = re.compile(r"[A-Za-z0-9_.:/+@=,|\-]{0,200}")
GPU_NAME = re.compile(r"NVIDIA [A-Za-z0-9 _.\-]{1,60}")


def assert_text_free(obj, where: str = "") -> None:
    """Every string in obj (keys included) is a name, a number, a hash, a version or a path: no spaces, at most 200
    characters, from a small alphabet; a GPU's device name ("NVIDIA L40S") is the one exception. Raises
    CheckError("free_text") otherwise. Every stage's result passes through here before it leaves the container."""
    if isinstance(obj, str):
        if SAFE.fullmatch(obj) is None and GPU_NAME.fullmatch(obj) is None:
            raise CheckError("free_text")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_text_free(str(k), where)
            assert_text_free(v, where)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            assert_text_free(v, where)
    elif not (obj is None or isinstance(obj, (bool, int, float))):
        raise CheckError("free_text")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def written(path, res: dict) -> dict:
    """A stage's result, checked text-free, written as JSON and returned as the JSON reads back, so what the volume
    holds and what the laptop receives are the same."""
    res = json.loads(json.dumps(res))
    assert_text_free(res)
    write_json(path, res)
    return res


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def append_jsonl(path, row: dict) -> None:
    """One record per open and close, so the volume never holds a long-lived open file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_jsonl(path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def snapshot_dir(model: str, revision: str, hf_home: str | None = None) -> Path:
    """A model's folder at a pinned revision in the Hugging Face cache ($HF_HOME/hub/models--ORG--NAME/snapshots/REV).
    Loading from it pins the weights, tokenizer, chat template and generation_config to that commit."""
    hub = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    return hub / ("models--" + model.replace("/", "--")) / "snapshots" / revision


def require(path, what: str = "") -> Path:
    if not Path(path).exists():
        raise CheckError("missing")
    return Path(path)


# ------------------------------------------------------------------------------------------- the environment


def version_of(name: str) -> str | None:
    """A package's installed version, trying the name as written and with - and _ swapped."""
    for n in (name, name.replace("_", "-"), name.replace("-", "_")):
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return None


def pip_freeze() -> list[str]:
    """Every installed distribution as name==version, sorted: the container's own record of its image."""
    seen = {}
    for d in metadata.distributions():
        name = d.metadata["Name"] if d.metadata else None
        if name:
            seen[name] = d.version
    return [f"{n}=={v}" for n, v in sorted(seen.items(), key=lambda x: x[0].lower())]


def code_hashes(check_dir, pkg_dir) -> dict[str, str]:
    """sha256 of the data code (DATA_FILES, this folder's) and of the library files the instrument is made of (as
    anatta/<file>): the plan's "code"."""
    out = {f: sha256_file(Path(check_dir) / f) for f in DATA_FILES}
    out.update({f"anatta/{f}": sha256_file(Path(pkg_dir) / f) for f in LIBRARY_FILES})
    return out


def analysis_code_hashes(check_dir) -> dict[str, str]:
    """sha256 of the analysis code a container runs (ANALYSIS_FILES): checked by the analysis stage against its plan."""
    return {f: sha256_file(Path(check_dir) / f) for f in ANALYSIS_FILES}


def check_environment(plan: dict, stack: str, check_dir, pkg_dir) -> dict:
    """This container against the plan: its data code (code_hashes) and the installed version of every package its
    stack pins (plan["pins"][stack], stack "lab" or "repo"; a local label such as +cu124 is ignored). A difference
    stops the stage before anything loads, so results come only from the code and packages the plan names. The
    analysis code is not checked here (the analysis stage checks it against its own plan), so a fix to it leaves every
    data stage's resume, and the cheap tier, alone."""
    code = code_hashes(check_dir, pkg_dir)
    if code != plan["code"]:
        raise CheckError("env_code")
    pins = plan["pins"][stack]
    have = {n: version_of(n) for n in pins}
    if any((have[n] or "").split("+")[0] != v for n, v in pins.items()):
        raise CheckError("env_pins")
    return {"code": code, "installed": have}


# plan_id is check 1's (anatta-calib-c1, src/anatta/bench/fingerprint.py at 44760f7), copied; it names both the plan and
# the analysis plan.
def plan_id(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]


def function_source(source: str, names) -> str:
    """The source of the named top-level functions of a module's text, each with its decorators, in the order named
    (check 7's container_source, anatta-calib-c7 calibration/relearning/app.py at 02db868, made a function of the
    text so the card can hash app.py's analysis stage without importing modal)."""
    import ast

    lines = source.splitlines(keepends=True)
    defs = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    return "".join("".join(lines[min([defs[f].lineno, *(d.lineno for d in defs[f].decorator_list)]) - 1:
                                 defs[f].end_lineno]) for f in names)


def claim(out, plan: dict) -> None:
    """The run's folder holds this plan's files only: plan.json is written if absent and must equal the plan."""
    stored = read_json(Path(out) / "plan.json")
    if stored is None:
        write_json(Path(out) / "plan.json", plan)
    elif stored != plan:
        raise CheckError("plan")


def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 (no torch in a CPU stage's view is not a failure)
        return None


class Clock:
    def __init__(self):
        self.t0 = time.time()

    def seconds(self) -> int:
        return round(time.time() - self.t0)


# ------------------------------------------------------------------------------------------------ the layers


def hs_index(block: int) -> int:
    """generate_vec's hidden_states index for probe's block b. hidden_states[0] is the embedding output and
    hidden_states[l] is block l - 1's output (transformers' Qwen2 at 4.52.3; facts §5), so block b is index b + 1,
    and the repo's "layer 20" is probe's block 19. The last index (after the final norm) has no block."""
    return block + 1


# ------------------------------------------------------------------------------------------- the trait files


def a_or_an(word: str) -> str:
    # Ruled (NOTES, 12:20 UTC, ruling 8): "an" before a vowel, which gives every real trait name the same article.
    return "an" if word[:1].lower() in "aeiou" else "a"


def system_strings(trait: str, instruction: dict, prompts: dict) -> dict:
    """eval_persona.py's two system prompts for one instruction pair: pos f"You are {a_or_an(trait)} {trait}
    assistant. {instruction['pos']}", neg f"You are a helpful assistant. {instruction['neg']}" (the registration's
    forms; the repo writes neg with assistant_name "helpful" through the same f-string)."""
    return {"pos": prompts["pos"].format(article=a_or_an(trait), trait=trait, instruction=instruction["pos"]),
            "neg": prompts["neg"].format(instruction=instruction["neg"])}


def parse_trait_file(data: bytes, pairs: int, questions: int) -> dict:
    """A repo trait file, checked: a JSON object with exactly the keys instruction, questions and eval_prompt; the
    registered number of {pos, neg} pairs of non-empty strings; the registered number of non-empty questions. A file
    with other keys is refused (CheckError("keys"))."""
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CheckError("keys") from None
    if not isinstance(obj, dict) or set(obj) != set(REPO_KEYS):
        raise CheckError("keys")
    inst, qs = obj["instruction"], obj["questions"]
    ok = isinstance(inst, list) and len(inst) == pairs and all(
        isinstance(p, dict) and set(p) == set(SIGNS) and all(isinstance(p[s], str) and p[s] for s in SIGNS)
        for p in inst)
    if not ok:
        raise CheckError("pairs", got=len(inst) if isinstance(inst, list) else -1, want=pairs)
    if not (isinstance(qs, list) and len(qs) == questions and all(isinstance(q, str) and q for q in qs)):
        raise CheckError("questions", got=len(qs) if isinstance(qs, list) else -1, want=questions)
    if not isinstance(obj["eval_prompt"], str):
        raise CheckError("eval_prompt")
    return obj


def check_trait_file(data: bytes, entry: dict, expect: dict) -> dict:
    """A fetched trait file against its registered size and git blob sha, then parsed (parse_trait_file). Returns
    {bytes, blob, sha256, pairs, questions}, the facts prepare writes into config.yaml."""
    blob = git_blob_sha(data)
    if blob != entry["blob"] or len(data) != entry["bytes"]:
        raise CheckError("blob", got=blob, bytes=len(data))
    obj = parse_trait_file(data, expect["pairs"], expect["questions"])
    return {"bytes": len(data), "blob": blob, "sha256": sha256_bytes(data), "pairs": len(obj["instruction"]),
            "questions": len(obj["questions"])}


def registered_trait(data: bytes, entry: dict, expect: dict) -> dict:
    """A trait file on the volume against everything config.yaml registers for it (size, blob, sha256, counts).
    Returns the parsed file."""
    if not entry.get("sha256"):
        raise CheckError("unregistered")
    facts = check_trait_file(data, entry, expect)
    if any(facts[k] != entry[k] for k in ("sha256", "pairs", "questions")):
        raise CheckError("sha256", got=facts["sha256"])
    return json.loads(data.decode("utf-8"))


def load_traits(plan: dict) -> dict:
    """The run's trait files from the volume (plan["data_dir"], where prepare put them), each checked against
    everything config.yaml registers for it. Returns {trait: the parsed file}; its text stays in the container."""
    repo = plan["repo"]
    return {t: registered_trait(require(Path(plan["data_dir"]) / f"{t}.json").read_bytes(), repo["traits"][t],
                                repo["expect"]) for t in plan["traits"]}


def trait_url(repo: dict, name: str) -> str:
    return f"{repo['raw']}/{repo['commit']}/{repo['trait_dir']}/{name}.json"


def fetch_url(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (a fixed https URL from config.yaml)
        return r.read()


def prepare(repo: dict, names: list[str], data_dir, fetch=fetch_url, guard: Guard | None = None) -> dict:
    """prepare, on Modal: each trait file fetched at the commit, checked (check_trait_file) and written to data_dir on
    the volume. Returns {trait: {bytes, blob, sha256, pairs, questions}}; the text never leaves the volume."""
    out = {}
    for i, name in enumerate(names):
        if guard is not None:
            guard.row = i
        data = fetch(trait_url(repo, name))
        out[name] = check_trait_file(data, repo["traits"][name], repo["expect"])
        p = Path(data_dir) / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return out


def probe_trait(name: str, data: dict, prompts: dict, questions: int | None = None, pairs: int | None = None):
    """The repo's trait as an anatta.probe.Trait: name, a one-line description, family "calibration", the pairs as
    eval_persona.py's {pos, neg} system strings, and the questions (the first `questions` and `pairs` of each when
    given, for the smoke)."""
    from anatta.probe import Trait

    inst = data["instruction"][:pairs] if pairs else data["instruction"]
    qs = data["questions"][:questions] if questions else data["questions"]
    return Trait(name=name, family=prompts["family"], description=prompts["description"].format(trait=name),
                 pairs=[system_strings(name, i, prompts) for i in inst], questions=list(qs))


# ------------------------------------------------------------------------------------------------ the layout


def sizes(plan: dict) -> dict:
    """G's dimensions under the plan: questions, instructions and samples per (question, instruction) cell."""
    return dict(plan["layout"])


def row_of(qi: int, ii: int, si: int, n_i: int, n_s: int) -> int:
    return (qi * n_i + ii) * n_s + si


def cell_of(k: int, n_i: int, n_s: int) -> tuple[int, int, int]:
    """Row k's (question, instruction, sample), in the CSV's order: question -> instruction -> sample."""
    return k // (n_i * n_s), (k // n_s) % n_i, k % n_s


def batches(traits: list[str], n_q: int, n_i: int, n_s: int, size: int) -> list[dict]:
    """G's (or G''s) batches: for each trait, then each sign, its rows in order, up to `size` at a time. The index
    runs over every batch of the arm, so the seed base plus the index is a different seed for each."""
    rows = list(range(n_q * n_i * n_s))
    out = []
    for t in traits:
        for sign in SIGNS:
            for j in range(0, len(rows), size):
                out.append({"index": len(out), "trait": t, "sign": sign, "rows": rows[j:j + size]})
    return out
