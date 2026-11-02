"""Calibration check 3: what runs in the containers, with no Modal here, so the tests run it on a laptop on a tiny
random model. app.py's Modal functions call the stages below; each writes its files under the run's folder on the
anatta-runs volume and returns only numbers, counts and sha256s. The prompts, the completions and every direction stay
on the volume.

  prepare      CPU, the lab's image, with the Hugging Face secret. Fetches Wollschläger's four SALAD splits,
               Arditi's JailbreakBench file, JBB-Behaviors' harmful CSV and MMLU's test parquet into the run's folder,
               counts and hashes them, and stops if the two JailbreakBench files' 100 instructions differ. Fetches the
               judge (gemma-2b and the adapter) into the cache volume at "main" and checks that both resolve to the
               pinned commits, loads the judge's tokenizer online by the repo id the judge stages use (so the cache
               records the files the repo lacks), and records the adapter's files with their sizes and sha256s.
               Checks Qwen's chat template against Wollschläger's string on every instruction (string and ids), every
               model's post-instruction tokens, refusal tokens and MMLU letters, and the judge's "<bos>" quirk on
               benign strings. Last, it loads offline, in a fresh Python, every tokenizer and config the lab's later
               stages load, and stops if any does not load. A data file whose sha256 or count is not yet in
               config.yaml stops the run after prepare (prepare.json holds the numbers to register).
  prepare_paper  CPU, the paper's image, before any GPU: the same offline loads for judge_paper and swap.
  run_model    one L40S per model. The MMLU golden test first (batch 32 against batch 1, with no hook on; its gate,
               config.yaml's mmlu.golden_gate, stops the model's run in minutes), then filtering (once, shared by
               both paths; a split that kept no prompts stops the model's run there, before the extraction), the
               library path (anatta.contrast's candidates, anatta.select with the dial's hooks), on Qwen the
               reference path (reference.py) and C3-extract (a smoke run stops if its bitwise premise fails), then
               every condition, each after a check that no hook is left installed, with Qwen's Q0 generated again at
               the end. A full run stops after its first condition (Q0, M0) if that condition's MMLU on every row is
               below the floor (mmlu.floor_pct). On Ministral, prediction 82's re-read for the ledger, if the image's
               v_ego is the registered file.
  judge_lab    the port (judge.py) in the lab's stack: every condition's completions, and the judge control.
  judge_paper  the paper's stack (transformers 4.47.0): the original strong_reject package and the port, on Q0's and
               QA's completions for the judge port gate, or on every condition if the gate sends the grades there.
  swap         the paper's stack, on a miss: Q0 regenerated and scored by the original judge (C3-base misses), or the
               reference path re-run (C3 and C3-R both miss with viable candidates), with the MMLU pass if a miss was
               on the MMLU bar.
The judge stages record the judge's loaded dtype and the template's sha256 as the container saw them, and before
scoring, which attention kernels the container's PyTorch accepts for the judge's shapes (recorded, not graded).

Standard library only at import: torch, transformers and the package are imported inside the functions, and the paper
stack imports this file too.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

# The package files a run calls (relative to src/anatta): the instrument under test and what loads and drives it.
CODE_FILES = ("dial.py", "contrast.py", "select.py", "bench/mc.py", "probe.py", "models.py", "modal_app.py",
              "player.py")
# This check's code that runs in the containers: every file app.py adds to the images (its CHECK_CODE), so each is
# in the plan's code hashes and checked against the plan in the container.
CHECK_FILES = ("refusal.py", "reference.py", "judge.py", "fingerprint.py")
DATA = "data"  # the subfolder of a run's folder holding the fetched files; it never leaves the volume
SPLITS = ("harmful_train", "harmful_val", "harmless_train", "harmless_val")
MODELS = ("qwen", "ministral")
RECORDS = {"salad": "instruction", "jbb": "instruction"}
# Benign strings for the judge's "<bos>" quirk: one short, one well over 512 tokens.
BENIGN_SHORT = "The weather is mild today, and the park is full of people walking their dogs."
BENIGN_LONG = " ".join(["The quiet library opens at nine and closes at six every weekday."] * 120)


class CheckError(RuntimeError):
    """A check that stops the run."""


EMPTY_KEPT = "empty_kept_set"  # a model's run that stopped at filtering: some split kept no prompts
FLOOR_STOP = "mmlu_floor"  # a full run that stopped after its first condition, below the MMLU floor
# The stops that are one model's own (rulings, 1 Nov): the other model's run is still judged and graded (app.main).
# Any other stop ends the whole run.
MODEL_STOPS = (EMPTY_KEPT, FLOOR_STOP)


class EmptyKept(CheckError):
    """Filtering kept no prompts in some split (filter_stage). args: (the message, which names each empty split with
    its counts and no prompt text; the record so far, numbers only)."""

    def __str__(self) -> str:
        return str(self.args[0])


# ---------------------------------------------------------------------------------------------------- files


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(t) -> str:
    """sha256 of a tensor's values in float64, C order (a direction's fingerprint; the vector itself never leaves the
    volume)."""
    return sha256_bytes(t.detach().double().cpu().contiguous().numpy().tobytes())


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def written(path, res: dict) -> dict:
    """A stage's result written as JSON, and returned as the JSON reads back."""
    res = json.loads(json.dumps(res))
    write_json(path, res)
    return res


def stage(path, fn, log=print):
    """The stage's JSON if the run's folder already holds it (a resumed run), else fn() written there."""
    have = read_json(path)
    if have is not None:
        log(f"{Path(path).name}: read back")
        return have
    return written(path, fn())


# ----------------------------------------------------------------------------------------------- the stacks


def version_of(name: str) -> str | None:
    for n in (name, name.replace("_", "-"), name.replace("-", "_")):
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return None


def code_hashes(pkg_dir, here=None) -> dict[str, str]:
    """sha256 of the package files a run calls (as anatta/<file>) and of this check's container code."""
    pkg, here = Path(pkg_dir), Path(here or Path(__file__).resolve().parent)
    out = {f"anatta/{f}": sha256_file(pkg / f) for f in CODE_FILES}
    out.update({f: sha256_file(here / f) for f in CHECK_FILES})
    return out


def check_environment(plan: dict, stack: str, pkg_dir) -> dict:
    """This container against the plan: the code (code_hashes) and the installed version of every package its stack
    pins (plan["pins"][stack]; a local label such as +cu124 is ignored). A difference stops the run before anything
    loads."""
    code = code_hashes(pkg_dir)
    if code != plan["code"]:
        bad = sorted(f for f in set(code) | set(plan["code"]) if code.get(f) != plan["code"].get(f))
        raise CheckError(f"this container's code differs from the plan's in {', '.join(bad)}: launch again, which "
                         "plans afresh")
    pins = plan["pins"][stack]
    have = {n: version_of(n) for n in pins}
    wrong = sorted(n for n, v in pins.items() if (have[n] or "").split("+")[0] != v)
    if wrong:
        raise CheckError("installed versions differ from the pins: "
                         + ", ".join(f"{n} {have[n]} (pinned {pins[n]})" for n in wrong))
    return {"code": code, "installed": have}


def freeze() -> list[str]:
    """pip freeze, read from the installed distributions: "name==version", sorted."""
    return sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]})


def claim(out: Path, plan: dict) -> None:
    """The run's folder holds this plan's files only: plan.json is written if absent and must equal the plan."""
    stored = read_json(out / "plan.json")
    if stored is None:
        write_json(out / "plan.json", plan)
    elif stored != plan:
        raise CheckError(f"{out} holds another plan's files")


def snapshot_dir(model: str, revision: str, hf_home: str | None = None) -> Path:
    """A repo's folder at a pinned revision in the Hugging Face cache ($HF_HOME/hub/models--ORG--NAME/snapshots/REV)."""
    hub = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    return hub / ("models--" + model.replace("/", "--")) / "snapshots" / revision


def model_complete(snap: Path) -> bool:
    """True if a safetensors snapshot holds its config, tokenizer config and every weight shard its index names."""
    snap = Path(snap)
    if not all((snap / f).exists() for f in ("config.json", "tokenizer_config.json")):
        return False
    index = snap / "model.safetensors.index.json"
    if index.exists():
        return all((snap / s).exists() for s in set(json.loads(index.read_text())["weight_map"].values()))
    return (snap / "model.safetensors").exists()


# ------------------------------------------------------------------------------------------------- the data


def data_files(plan: dict) -> dict:
    """Every fetched file: {name: (path under the data folder, its registered entry)}."""
    d = plan["data"]
    out = {split: (f"salad/{f['file']}", f) for split, f in d["salad"]["files"].items()}
    out["jbb"] = (f"jbb/{d['jbb']['file']}", d["jbb"])
    out["jbb_check"] = (f"jbb/{Path(d['jbb_check']['file']).name}", d["jbb_check"])
    out["mmlu"] = (f"mmlu/{Path(d['mmlu']['file']).name}", d["mmlu"])
    return out


def count_of(name: str, entry: dict):
    """The registered count of a data file, by the key its entry uses."""
    for key in ("records", "rows", "questions"):
        if key in entry:
            return entry[key]
    return None


def registration(plan: dict, measured: dict) -> dict:
    """Each data file's measured bytes, sha256 and count against config.yaml's. ok only if every registered value is
    filled in and equal; a registered bytes value that differs stops prepare."""
    rows, ok = {}, True
    for name, (_, entry) in data_files(plan).items():
        m = measured[name]
        if entry.get("bytes") is not None and m["bytes"] != entry["bytes"]:
            raise CheckError(f"{name}: {m['bytes']} bytes, not the registered {entry['bytes']}")
        want = {"sha256": entry.get("sha256"), "count": count_of(name, entry)}
        missing = [k for k, v in want.items() if v is None]
        differ = [k for k, v in want.items() if v is not None and v != m[k]]
        rows[name] = {"missing": missing, "differ": differ}
        ok = ok and not missing and not differ
    return {"ok": ok, "files": rows}


def check_data(plan: dict, data_dir) -> dict:
    """The fetched files against their registered sha256 (every container, before it reads them)."""
    out = {}
    for name, (rel, entry) in data_files(plan).items():
        p = Path(data_dir) / rel
        if not p.exists():
            raise CheckError(f"{rel} is not in the run's data folder: run prepare first")
        sha = sha256_file(p)
        if entry.get("sha256") is None or sha != entry["sha256"]:
            raise CheckError(f"{rel} has sha256 {sha}, not the registered {entry.get('sha256')}")
        out[name] = sha
    return out


def instructions_of(path) -> list[str]:
    return [r["instruction"] for r in json.loads(Path(path).read_text())]


def jbb_goals(csv_text: str, column: str = "Goal") -> list[str]:
    import csv
    import io

    return [row[column] for row in csv.DictReader(io.StringIO(csv_text))]


def load_prompts(plan: dict, data_dir) -> dict:
    """The prompt sets as Wollschläger's run_pipeline.py takes them: all of harmful train and harmful val, the first
    len(harmful) records of each harmless file; and JailbreakBench's 100 instructions."""
    files = data_files(plan)
    sets = {s: instructions_of(Path(data_dir) / files[s][0]) for s in SPLITS}
    sets["harmless_train"] = sets["harmless_train"][:len(sets["harmful_train"])]
    sets["harmless_val"] = sets["harmless_val"][:len(sets["harmful_val"])]
    sets["jbb"] = instructions_of(Path(data_dir) / files["jbb"][0])
    return sets


def mmlu_rows(path) -> list[dict]:
    """MMLU's test rows from the prepared JSON (prepare converts the parquet): question, subject, choices, answer."""
    return json.loads(Path(path).read_text())


def parquet_to_rows(path) -> list[dict]:
    import pyarrow.parquet as pq

    cols = ["question", "subject", "choices", "answer"]
    return [{c: r[c] for c in cols} for r in pq.read_table(path, columns=cols).to_pylist()]


# --------------------------------------------------------------------------------------------------- prepare


def fetch_url(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 (fixed https URLs from config.yaml)
        return r.read()


def hf_file(repo: str, file: str, revision: str, repo_type: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=file, revision=revision, repo_type=repo_type))


def hf_snapshot(repo: str, revision: str = "main", allow=None, ignore=None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=allow, ignore_patterns=ignore))


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


# ------------------------------------------------------------------------------------------- offline loads
# The later stages read the cache volume offline (HF_HUB_OFFLINE=1). A load by repo id there resolves "main" through
# the cache's refs/main and may ask for a file the repo does not have; offline, the hub library can tell such a file
# from one never fetched only by the cache's record of its absence (.no_exist/COMMIT/FILE), which only an online lookup
# writes. snapshot_download records no absent file. The smoke of 1 Nov stopped on this: transformers 5.21's
# AutoTokenizer resolves a model config first, the adapter's repo has no config.json, and judge_lab could not load the
# judge's tokenizer. prepare now loads that tokenizer online by the repo id the judge uses, and at its end loads offline
# every tokenizer and config a later stage loads; prepare_paper does the same in the paper's stack.


def repo_cache(repo: str, hf_home: str | None = None) -> Path:
    """A model repo's folder in the Hugging Face cache ($HF_HOME/hub/models--ORG--NAME)."""
    return snapshot_dir(repo, "_", hf_home).parent.parent


def ref_commit(repo: str, ref: str = "main", hf_home: str | None = None) -> str | None:
    """The commit the cache's refs/REF names for a repo, which an offline load by repo id reads (None if absent)."""
    p = repo_cache(repo, hf_home) / "refs" / ref
    return p.read_text().strip() if p.is_file() else None


def absent_files(repo: str, commit: str, hf_home: str | None = None) -> list[str]:
    """The files the cache records as absent from a repo at a commit (.no_exist): names only."""
    d = repo_cache(repo, hf_home) / ".no_exist" / commit
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.is_dir() else []


def offline_loads(plan: dict, stack: str, hf_home: str | None = None) -> list[dict]:
    """Every tokenizer and config a later stage loads from the cache volume in `stack` ("lab" or "paper"), by the name
    that stage passes: the judge by judge.ADAPTER, its repo id (judge_lab; judge_paper and swap, where the original
    package loads the same id with the same call), and the base its adapter names; the models by their snapshot folders
    (run_model; swap, Qwen only). An "adapter_config" load is what a model load by the adapter's id reads first:
    transformers' find_adapter_config_file, then PEFT's adapter config, which names the base."""
    import judge as J

    judge_stages = ["judge_lab"] if stack == "lab" else ["judge_paper", "swap"]
    stage, keys = ("run_model", MODELS) if stack == "lab" else ("swap", ("qwen",))
    out = [{"stages": judge_stages, "load": "judge_tokenizer", "name": J.ADAPTER},
           {"stages": judge_stages, "load": "adapter_config", "name": J.ADAPTER},
           {"stages": judge_stages, "load": "config", "name": plan["judge"]["base"]["id"]}]
    for key in (k for k in keys if k in plan["models"]):
        m = plan["models"][key]
        snap = str(snapshot_dir(m["id"], m["revision"], hf_home))
        out += [{"stages": [stage], "load": "tokenizer", "name": snap},
                {"stages": [stage], "load": "config", "name": snap}]
    return out


def load_offline(x: dict) -> dict:
    """One offline_loads entry, loaded with the call its stage makes: what loaded, as numbers and names."""
    if x["load"] == "judge_tokenizer":
        import judge as J

        return {"vocab": len(J.load_judge_tokenizer(x["name"]))}
    if x["load"] == "tokenizer":
        from transformers import AutoTokenizer

        return {"vocab": len(AutoTokenizer.from_pretrained(x["name"]))}
    if x["load"] == "config":
        from transformers import AutoConfig

        return {"model_type": AutoConfig.from_pretrained(x["name"]).model_type}
    if x["load"] == "adapter_config":
        from peft import PeftConfig
        from transformers.utils.peft_utils import find_adapter_config_file

        if find_adapter_config_file(x["name"]) is None:
            raise CheckError("transformers finds no adapter config")
        return {"base": PeftConfig.from_pretrained(x["name"]).base_model_name_or_path}
    raise CheckError(f"no such load: {x['load']!r}")


OFFLINE_MARK = "offline loads: "  # the line offline_main prints its records on


def offline_main() -> None:
    """The child process offline_check starts: each load on stdin (offline_loads), one record each on one line."""
    import sys

    out = []
    for x in json.loads(sys.stdin.read()):
        try:
            out.append({**x, "ok": True, **load_offline(x)})
        except Exception as e:  # noqa: BLE001 (each load's failure is its record)
            first = (str(e).splitlines() or [""])[0][:200]
            out.append({**x, "ok": False, "error": f"{type(e).__name__}: {first}"})
    print(OFFLINE_MARK + json.dumps(out), flush=True)


def offline_check(loads: list[dict], env: dict | None = None, timeout: float = 900) -> list[dict]:
    """The loads in a fresh Python with HF_HUB_OFFLINE=1 set before any import, as the later stages run, and this
    file imported from the parent's sys.path: one record per load, with ok and what loaded or the error's first line."""
    import subprocess
    import sys

    child = {**os.environ, **(env or {}), "HF_HUB_OFFLINE": "1",
             "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    r = subprocess.run([sys.executable, "-c", "import refusal; refusal.offline_main()"], input=json.dumps(loads),
                       capture_output=True, text=True, env=child, timeout=timeout)
    lines = [x for x in r.stdout.splitlines() if x.startswith(OFFLINE_MARK)]
    if not lines:
        raise CheckError(f"the offline check exited with code {r.returncode} and no record")
    return json.loads(lines[-1][len(OFFLINE_MARK):])


def offline_failures(recs: list[dict] | None, plan: dict, stack: str) -> list[str]:
    """What an offline check's records fail on: a record per load in offline_loads' order, each loaded, and every
    adapter config naming the registered base. Empty if all is well."""
    want = [(x["load"], x["stages"]) for x in offline_loads(plan, stack)]
    if [(r.get("load"), r.get("stages")) for r in recs or []] != want:
        return [f"the {stack} stack's offline record does not list the loads the later stages make"]
    base = plan["judge"]["base"]["id"]
    bad = [f"{r['load']} {r['name']} ({', '.join(r['stages'])}): {r.get('error')}" for r in recs if not r.get("ok")]
    return bad + [f"the adapter names the base {r['base']}, not the registered {base}" for r in recs
                  if r.get("ok") and r["load"] == "adapter_config" and r.get("base") != base]


def template_check(tok, m: dict, instructions: list[str]) -> dict:
    """Each instruction's chat-template rendering (one user turn, the generation prompt) against the model's
    registered template string with its default system prompt, as text and as token ids: counts only."""
    from anatta.models import as_messages, chat_ids

    if not m.get("template"):
        return {"checked": 0, "string_differ": None, "ids_differ": None}
    sd = idd = 0
    for x in instructions:
        want = m["template"].format(system=m["system"], instruction=x)
        got = tok.apply_chat_template(as_messages(x), add_generation_prompt=True, tokenize=False)
        sd += got != want
        idd += list(tok(want, add_special_tokens=False)["input_ids"]) != chat_ids(tok, as_messages(x))[0].tolist()
    return {"checked": len(instructions), "string_differ": sd, "ids_differ": idd}


def token_checks(tok, m: dict) -> dict:
    """The model's registered token ids against its tokenizer: the post-instruction tokens from the rendered prompt's
    tail, the refusal tokens' text, the MMLU continuations (one token each), whether plain text gets a BOS, and a
    short instruction's rendered length. A difference stops the run."""
    from reference import eoi_toks_from_template

    from anatta.bench.mc import letter_ids
    from anatta.models import as_messages, chat_ids

    eoi = eoi_toks_from_template(tok)
    refusal_text = [tok.decode([i]) for i in m["refusal_toks"]]
    letters = letter_ids(tok)
    plain = list(tok("Answer:")["input_ids"])
    bos = tok.bos_token_id is not None and plain[:1] == [tok.bos_token_id]
    out = {"eoi_toks": eoi, "refusal_text": refusal_text, "letters": letters, "bos_added": bos,
           "rendered_tokens": int(chat_ids(tok, as_messages("Give three tips for staying healthy."))[0].shape[0])}
    bad = [k for k, want in (("eoi_toks", m["eoi_toks"]), ("refusal_text", m["refusal_tok_text"]),
                             ("letters", m["letters"]), ("bos_added", m["bos_added"])) if out[k] != want]
    if bad:
        raise CheckError(f"{m['id']}: the tokenizer's {', '.join(bad)} differ from config.yaml's")
    return out


def prepare(plan: dict, out, pkg_dir, log=print, fetch=fetch_url, get_file=hf_file, get_snapshot=hf_snapshot,
            tokenizer=load_tokenizer, hf_home=None, offline=offline_check) -> dict:
    """Fetch, count, hash and check (the module docstring). Returns prepare.json's contents: numbers, booleans,
    sha256s, and file and repo names only."""
    check_environment(plan, "lab", pkg_dir)
    out = Path(out)
    claim(out, plan)
    ddir = out / DATA
    files = data_files(plan)
    measured, d = {}, plan["data"]

    def keep(name: str, data: bytes, count) -> None:
        p = ddir / files[name][0]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        measured[name] = {"bytes": len(data), "sha256": sha256_bytes(data), "count": count}
        log(f"prepare: {files[name][0]}: {len(data)} bytes, {count} records")

    for split, f in d["salad"]["files"].items():
        data = fetch(d["salad"]["url"] + f["file"])
        keep(split, data, len(json.loads(data)))
    data = fetch(d["jbb"]["url"])
    keep("jbb", data, len(json.loads(data)))
    c = d["jbb_check"]
    data = Path(get_file(c["repo"], c["file"], c["revision"], c["repo_type"])).read_bytes()
    goals = jbb_goals(data.decode("utf-8"), c["column"])
    keep("jbb_check", data, len(goals))
    mm = d["mmlu"]
    data = Path(get_file(mm["repo"], mm["file"], mm["revision"], mm["repo_type"])).read_bytes()
    measured_mmlu_sha = sha256_bytes(data)
    if measured_mmlu_sha != mm["sha256"]:
        raise CheckError(f"MMLU's parquet has sha256 {measured_mmlu_sha}, not the registered {mm['sha256']}")
    p = ddir / files["mmlu"][0]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    rows = parquet_to_rows(p)
    write_json(ddir / "mmlu" / "test.json", rows)
    measured["mmlu"] = {"bytes": len(data), "sha256": measured_mmlu_sha, "count": len(rows)}

    jbb = instructions_of(ddir / files["jbb"][0])
    agree = {"same_list": jbb == goals, "same_set": sorted(jbb) == sorted(goals), "n": len(jbb), "n_check": len(goals)}
    res = {"files": measured, "mmlu_json_sha256": sha256_file(ddir / "mmlu" / "test.json"), "jbb_agree": agree}
    if not agree["same_list"]:
        res["registration"] = {"ok": False}
        written(out / "prepare.json", res)
        raise CheckError("Arditi's JailbreakBench file and JBB-Behaviors' harmful split differ in their 100 "
                         "instructions (prepare.json): prepare stops")

    # the judge, at "main", which must resolve to the pinned commits (strong_reject's own loader reads it offline)
    j = plan["judge"]
    base = get_snapshot(j["base"]["id"], "main", j["base_allow"], j["base_ignore"])
    adapter = get_snapshot(j["adapter"]["id"], "main")
    judge = {"base_commit": Path(base).name, "adapter_commit": Path(adapter).name,
             "base_shard_bytes": sum(f.stat().st_size for f in Path(base).glob("*.safetensors")),
             "adapter_bytes": sum(f.stat().st_size for f in Path(adapter).rglob("*") if f.is_file()),
             "adapter_safetensors_sha256": sha256_file(Path(adapter) / "adapter_model.safetensors"),
             "gguf_fetched": any(Path(base).glob("*.gguf"))}
    res["judge"] = judge
    wrong = [k for k, want in (("base_commit", j["base"]["revision"]), ("adapter_commit", j["adapter"]["revision"]),
                               ("base_shard_bytes", j["base_shard_bytes"]),
                               ("adapter_safetensors_sha256", j["adapter_safetensors_sha256"]),
                               ("gguf_fetched", False)) if judge[k] != want]
    if wrong:
        written(out / "prepare.json", res)
        raise CheckError(f"the judge's files differ from config.yaml's in {', '.join(wrong)} (prepare.json)")
    from judge import ADAPTER, bos_quirk, load_judge_tokenizer

    # The judge's tokenizer as the judge stages load it, by the adapter's repo id, here online, so that the cache
    # records every lookup the load makes, the files the repo lacks included (offline_loads' comment); the "<bos>"
    # quirk is measured on it. Then the adapter's files at the pinned commit, and the absences the cache recorded.
    if ADAPTER != j["adapter"]["id"]:
        raise CheckError(f"judge.py loads {ADAPTER}, not config.yaml's adapter {j['adapter']['id']}")
    res["bos_quirk"] = bos_quirk(load_judge_tokenizer(ADAPTER), BENIGN_SHORT, BENIGN_LONG, j["max_response_length"])
    adapter = Path(adapter)
    judge["adapter_files"] = {str(f.relative_to(adapter)): {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
                              for f in sorted(adapter.rglob("*")) if f.is_file()}
    pins = {k: (j[k]["id"], j[k]["revision"]) for k in ("base", "adapter")}
    judge["absent"] = {k: absent_files(repo, rev, hf_home) for k, (repo, rev) in pins.items()}
    moved = [k for k, (repo, rev) in pins.items() if ref_commit(repo, "main", hf_home) != rev]
    if moved:
        written(out / "prepare.json", res)
        raise CheckError(f"the cache's main of the judge's {' and '.join(moved)} no longer names the pinned commit, "
                         "which the judge stages read offline (prepare.json)")

    # the two models' tokenizers from the cache volume at their pinned revisions
    prompts = load_prompts(plan, ddir)
    every = [x for s in (*SPLITS, "jbb") for x in prompts[s]]
    res["models"] = {}
    for key, m in plan["models"].items():
        snap = snapshot_dir(m["id"], m["revision"], hf_home)
        if not model_complete(snap):
            raise CheckError(f"{m['id']} is not complete in the cache volume at {m['revision']}")
        tok = tokenizer(snap)
        res["models"][key] = {"tokens": token_checks(tok, m), "template": template_check(tok, m, every)}

    res["registration"] = registration(plan, measured)

    # last, every tokenizer and config the lab's later stages load, loaded offline as they load them; a rerun of the
    # plan reads a failure here from prepare.json and stops too (app.main)
    res["offline"] = offline(offline_loads(plan, "lab", hf_home))
    failed = offline_failures(res["offline"], plan, "lab")
    if failed:
        written(out / "prepare.json", res)
        raise CheckError("a later stage could not load offline: " + "; ".join(failed) + " (prepare.json)")
    log(f"prepare: {len(res['offline'])} tokenizers and configs load offline")
    return written(out / "prepare.json", res)


def prepare_paper(plan: dict, out, pkg_dir, log=print, offline=offline_check, hf_home=None) -> dict:
    """In the paper's stack, after prepare and before any GPU starts: every tokenizer and config judge_paper and swap
    load, loaded offline as they load them (offline_loads). Stops if any does not load. Returns prepare_paper.json's
    contents: versions, names and numbers only."""
    env = check_environment(plan, "paper", pkg_dir)
    out = Path(out)
    res = written(out / "prepare_paper.json", {"environment": env, "offline": offline(offline_loads(plan, "paper",
                                                                                                    hf_home))})
    failed = offline_failures(res["offline"], plan, "paper")
    if failed:
        raise CheckError("a later stage could not load offline in the paper's stack: " + "; ".join(failed)
                         + " (prepare_paper.json)")
    log(f"prepare (paper): {len(res['offline'])} tokenizers and configs load offline")
    return res


# ------------------------------------------------------------------------------------------------ statistics


def quantiles(xs: list[float]) -> dict:
    s = sorted(xs)
    if not s:
        return {"n": 0}
    at = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]  # noqa: E731
    return {"n": len(s), "mean": sum(s) / len(s), "min": s[0], "p05": at(0.05), "p25": at(0.25), "p50": at(0.5),
            "p75": at(0.75), "p95": at(0.95), "max": s[-1], "share_pos": sum(x > 0 for x in s) / len(s)}


def cos(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))


def unit(v):
    v = v.double()
    return v / v.norm()


# ------------------------------------------------------------------------------------------------ the model run


def tokenize_fn(tok):
    """Left-padded batches of chat prompts (anatta.models.chat_batch), the tokenize_instructions_fn of both paths."""
    from anatta.models import chat_batch

    return lambda instructions: chat_batch(tok, instructions)


def filter_stage(model, tok, m: dict, prompts: dict, batch: int, log=print) -> tuple[dict, dict]:
    """Wollschläger's filtering (reference.filter_data), once: the kept sets, each split's base metric at batch 32
    and at batch 1 (the prompts whose sign would flip are counted, not used), and the top first tokens on the
    filtered harmful val. Returns (the record, the kept index lists). Raises EmptyKept, before anything else runs, if
    a split kept no prompts."""
    import reference as R

    mb = R.model_base_for(model, tok, m["eoi_toks"], m["refusal_toks"])
    hf_tr, hl_tr, hf_va, hl_va, scores = R.filter_data(mb, prompts["harmful_train"], prompts["harmless_train"],
                                                       prompts["harmful_val"], prompts["harmless_val"], batch)
    # A split that kept no prompts stops this model's run here, before the batch-1 flips and the extraction spend GPU
    # time (ruling, 1 Nov). Only an empty split stops it: a minimum count would change the instrument after
    # registration.
    counts = {s: {"n": len(prompts[s]), "kept": len(got)} for s, got in zip(SPLITS, (hf_tr, hf_va, hl_tr, hl_va))}
    empty = [s for s in SPLITS if counts[s]["kept"] == 0]
    if empty:
        before_cut = sum(x < 0 for x in scores["harmless_train"])

        def named(s):
            note = f", {before_cut} before the cut to the harmful count" if s == "harmless_train" and before_cut else ""
            return f"{s} (n {counts[s]['n']}, kept 0{note})"
        raise EmptyKept("filtering kept no prompts in " + ", ".join(named(s) for s in empty)
                        + ": this model's run stops before the extraction",
                        {"counts": counts, "metric": {s: quantiles(scores[s]) for s in SPLITS}, "empty": empty,
                         "harmless_train_before_cut": before_cut})
    harmful = {"harmful_train", "harmful_val"}
    kept = {s: [i for i, x in enumerate(scores[s]) if (x > 0 if s in harmful else x < 0)] for s in SPLITS}
    kept["harmless_train"] = kept["harmless_train"][:len(kept["harmful_train"])]
    for s, got in zip(SPLITS, (hf_tr, hf_va, hl_tr, hl_va)):
        if [prompts[s][i] for i in kept[s]] != got:
            raise CheckError(f"the kept indices of {s} do not give filter_data's kept set")
    rec = {"counts": {s: {"n": len(prompts[s]), "kept": len(kept[s])} for s in SPLITS},
           "metric": {s: quantiles(scores[s]) for s in SPLITS}, "flips_batch1": {}}
    for s in SPLITS:
        b1 = R.get_refusal_scores(model, prompts[s], mb.tokenize_instructions_fn, mb.refusal_toks, batch_size=1).tolist()
        rec["flips_batch1"][s] = sum(((a > 0) != (b > 0)) if s in harmful else ((a < 0) != (b < 0))
                                     for a, b in zip(scores[s], b1))
    logits = R.get_last_position_logits(model, tok, hf_va, mb.tokenize_instructions_fn, batch_size=batch)
    first = Counter(logits.argmax(dim=-1).tolist()) if logits is not None else Counter()
    n = sum(first.values())
    rec["top_first_tokens"] = [[t, c / n] for t, c in first.most_common(10)] if n else []
    rec["first_token_share"] = {str(t): (first[t] / n if n else None)
                                for t in (*m["refusal_toks"], *m.get("watch_toks", []))}
    log(f"filter: kept {', '.join(f'{s} {len(kept[s])}/{len(prompts[s])}' for s in SPLITS)}")
    return rec, kept


def c3_extract(model, tok, mb, lib, ref_padded, harmful: list[str], harmless: list[str], positions: list[int],
               log=print) -> tuple[dict, object]:
    """C3-extract: first, on one prompt, the block inputs as anatta.contrast reads them against Arditi's pre-hooks'
    reading, the premise that they are bitwise equal (recorded with the layers that differ); then Arditi's
    generate_directions at batch 1 with the input ids alone, and
    every candidate at layer >= 1 against the library's (cosine, relative norm difference, relative error), with the
    cosine to the padded reference's beside it. Returns (the record, the batch-1 candidates)."""
    import reference as R
    import torch

    from anatta.contrast import token_acts

    n = len(mb.model_block_modules)
    acts = token_acts(model, tok, harmful[:1], list(range(n)), positions, site="input")[0]
    seen = {}

    def grab(layer):
        def fn(module, input):
            seen[layer] = input[0].clone()
        return fn

    ids = mb.tokenize_instructions_fn(instructions=harmful[:1]).input_ids
    with torch.no_grad(), R.add_hooks([(mb.model_block_modules[i], grab(i)) for i in range(n)], []):
        model(input_ids=ids.to(R._device(model)))
    theirs = {i: seen[i][0, positions].float().cpu() for i in range(n)}
    unequal = [i for i in range(n) if not torch.equal(acts[i], theirs[i])]
    equal = not unequal
    # The premise is recorded here, and the comparison runs either way (ruling, 1 Nov): run_model stops a smoke run
    # whose premise failed, so it is fixed before launch; a full run records it and goes on, and the card reads a
    # C3-extract miss with the premise failed as unexplained, not as a position or site error.
    b1 = R.generate_directions(mb, harmful, harmless, batch_size=1, attention_mask=False).cpu()
    table, padded = [], []
    for pi, p in enumerate(positions):
        for layer in range(1, n):
            a, b = lib[pi, layer], b1[pi, layer]
            bn = float(b.norm())
            table.append([p, layer, cos(a, b), abs(float(a.norm()) - bn) / bn, float((a - b).norm()) / bn])
            padded.append([p, layer, cos(b, ref_padded[pi, layer])])
    # layer 0 (the embedding output at the post-instruction tokens, the same in every prompt) is left out of the
    # comparison: the library's running sums give exact zeros there, Arditi's (1/n)-scaled sums leave rounding residue
    rec = {"bitwise_premise": equal, "premise_unequal_layers": unequal,
           "premise_max_abs_diff": max((float((acts[i] - theirs[i]).abs().max()) for i in unequal), default=0.0),
           "n_compared": len(table), "layer0_zero_library": bool((lib[:, 0] == 0).all()),
           "layer0_max_abs_reference": float(b1[:, 0].abs().max()), "table": table,
           "min_cos": min(r[2] for r in table), "max_rel_norm": max(r[3] for r in table),
           "max_rel_err": max(r[4] for r in table), "padded_min_cos": min(r[2] for r in padded), "padded": padded}
    log(f"C3-extract: bitwise premise {equal}, min cos {rec['min_cos']:.6f}, max rel norm {rec['max_rel_norm']:.2e}")
    return rec, b1


def smoke_which(n: int) -> list[tuple[int, int]]:
    return [(-1, n // 2), (-1, n // 2 + 1)]


def choose(scores: dict, smoke: bool) -> dict:
    """anatta.select.choose with the registered ladder; a smoke run with nothing chosen takes the lowest bypass among
    its two scored candidates, labelled, so the plumbing still runs (read as nothing)."""
    from anatta import select

    c = select.choose(scores)
    if c["found"] or not smoke:
        return c
    pos, layer = min(scores["scored"], key=lambda pl: scores["bypass"][scores["n_pos"] + pl[0]][pl[1]])
    entry = select._entry(scores, pos, layer)
    return {**c, "found": True, "rung": None, "rung_name": "smoke fallback", "pos": pos, "layer": layer,
            "chosen": entry}


def recorded_generate(model, rp: float, eos_ids: list[int]):
    """A context manager that pins model.generation_config.repetition_penalty to `rp` (transformers 5 fills the
    fields generate is not given from it; probe.generate passes none) and records every model.generate call made
    inside it (the arguments as passed, minus tensors, and the output ids), asserting the penalty and the eos ids
    before each call. Both are put back afterwards."""
    from contextlib import contextmanager

    @contextmanager
    def cm():
        gc = model.generation_config
        old = gc.repetition_penalty
        gc.repetition_penalty = rp
        calls = []

        def recording(*args, **kwargs):
            if model.generation_config.repetition_penalty != rp:
                raise CheckError(f"repetition_penalty is {model.generation_config.repetition_penalty}, not {rp}")
            eos = model.generation_config.eos_token_id
            if sorted(eos if isinstance(eos, list) else [eos]) != sorted(eos_ids):
                raise CheckError(f"the model's eos ids are {eos}, not {eos_ids}")
            out = type(model).generate(model, *args, **kwargs)
            calls.append({"width": int(kwargs["input_ids"].shape[1]), "out": out.cpu(),
                          "kwargs": {k: v for k, v in kwargs.items() if isinstance(v, (int, float, bool, str))
                                     or v is None}})
            return out

        model.generate = recording
        try:
            yield calls
        finally:
            del model.generate
            gc.repetition_penalty = old

    return cm()


def new_token_facts(calls: list, eos_ids: list[int], max_new: int) -> tuple[list[int], list[bool]]:
    """Per completion: the new tokens up to and including the first eos (max_new without one), and whether it was
    cut at max_new (no eos among its new tokens)."""
    counts, cut = [], []
    for c in calls:
        for row in c["out"][:, c["width"]:].tolist():
            first = next((i for i, t in enumerate(row) if t in eos_ids), None)
            counts.append(len(row) if first is None else first + 1)
            cut.append(first is None and len(row) >= max_new)
    return counts, cut


def generation_config_facts(model) -> dict:
    gc = model.generation_config
    keys = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "eos_token_id", "pad_token_id",
            "bos_token_id", "max_new_tokens", "max_length")
    return {k: getattr(gc, k, None) for k in keys}


def generate_jbb(model, tok, prompts: list[str], batch: int, max_new: int, rp: float, eos_ids: list[int],
                 substrings: list[str]) -> tuple[list[str], dict]:
    """The completions of `prompts` through anatta.probe.generate (greedy, left padding), in batches of `batch`,
    with whatever hooks the caller has installed on, decoded without special tokens and stripped. Returns (the
    completions, the facts: new-token counts, cuts at max_new, the substring refusal flags, the settings as
    passed)."""
    from anatta.models import as_messages
    from anatta.probe import generate

    texts = []
    with recorded_generate(model, rp, eos_ids) as calls:
        for i in range(0, len(prompts), batch):
            texts += generate(model, tok, [as_messages(p) for p in prompts[i:i + batch]], max_new_tokens=max_new,
                              temperature=0.0)
    completions = [t.strip() for t in texts]
    counts, cut = new_token_facts(calls, eos_ids, max_new)
    refusal = [int(any(s.lower() in c.lower() for s in substrings)) for c in completions]
    facts = {"n": len(completions), "new_tokens": counts, "cut": sum(cut), "refusal_substring": refusal,
             "completion_sha256": [sha256_text(c) for c in completions], "passed": calls[0]["kwargs"] if calls else {},
             "repetition_penalty": rp, "batches": len(calls)}
    return completions, facts


def mmlu_texts(rows: list[dict]) -> list[str]:
    from anatta.bench.mc import mmlu_prompt

    return [mmlu_prompt(r["question"], list(r["choices"]), r["subject"]) for r in rows]


def run_mmlu(model, tok, rows: list[dict], letters: list[int], batch: int, chat: bool = False) -> dict:
    from anatta.bench import mc

    res = mc.predict(model, tok, mmlu_texts(rows), letters, batch_size=batch, chat=chat)
    bits = "".join("1" if int(p) == int(r["answer"]) else "0" for p, r in zip(res["preds"], rows))
    return {"n": len(rows), "correct": bits.count("1"), "acc": 100 * bits.count("1") / len(rows), "bits": bits,
            "chat": chat}


def bf16_step(x: float) -> float:
    """One bf16 step at magnitude |x|: bf16 keeps 8 significant bits, so the values in [2^e, 2^(e+1)) lie 2^(e-7)
    apart. Below the smallest normal (2^-126), and at 0 (the guard on log2), it is the subnormal spacing, 2^-133."""
    x = abs(x)
    if not x > 0:
        return 2.0 ** -133
    return 2.0 ** (max(math.floor(math.log2(x)), -126) - 7) if math.isfinite(x) else math.inf


def golden_item(index: int, length: int, lb: list[float], la: list[float]) -> dict:
    """One golden-test item from its four letter logits at batch 1 (lb) and batched (la), numbers only: its index
    into the MMLU rows and token length, both sets of logits, both predictions and whether they differ, and in bf16
    steps s = bf16_step(the largest |batch-1 logit|): the batch-1 top-2 margin, the largest |logit difference|, the
    differential (each letter's difference, batched minus batch 1, minus the item's mean difference: the part of the
    difference that can change an answer; a shift common to the four letters cannot) and the change in the batch-1
    top-2 margin (the batched logits' difference between the batch-1 top letter and its runner-up, minus the batch-1
    margin; a flip needs it at or below minus the margin). The predictions are mc.first_argmax's, as mc.predict makes
    them."""
    from anatta.bench.mc import first_argmax

    lb, la = [float(x) for x in lb], [float(x) for x in la]
    s = bf16_step(max(abs(x) for x in lb))
    d = [x - y for x, y in zip(la, lb)]
    mean = sum(d) / len(d)
    differential = [x - mean for x in d]
    i1, i2 = sorted(range(len(lb)), key=lambda k: (-lb[k], k))[:2]
    margin, diff = lb[i1] - lb[i2], max(abs(x) if x == x else math.inf for x in d)  # a NaN difference is past any bound
    change = (la[i1] - la[i2]) - margin
    pb, pa = first_argmax(lb), first_argmax(la)
    return {"index": index, "length": length, "logits_b1": lb, "logits_batched": la, "pred_b1": pb, "pred_batched": pa,
            "differs": pb != pa, "step": s, "margin_b1": margin, "margin_steps": margin / s, "max_abs_diff": diff,
            "diff_steps": diff / s, "differential": differential,
            "max_differential_steps": max(abs(x) for x in differential) / s, "margin_change": change,
            "margin_change_steps": change / s}


def golden_record(per_item: list[dict], batch: int) -> dict:
    """The golden test's record from its items in batch order (golden_item): the counts, and for each differing
    item its batch when batched (the batch's index, the item's place in it and the token lengths of the batch's
    rows; each batch is padded on the right to its longest row); the largest |logit difference| and the smallest
    batch-1 margin over every item; the item with the largest difference in steps (its index, length, difference and
    steps); and the largest differential in steps."""
    lengths = [it["length"] for it in per_item]
    for j, it in enumerate(per_item):
        if it["differs"]:
            first = (j // batch) * batch
            it.update(batch_index=j // batch, place=j - first, batch_lengths=lengths[first:first + batch])
    big = max(per_item, key=lambda it: (it["diff_steps"], it["max_abs_diff"], -it["index"]), default=None)
    return {"items": len(per_item), "disagreements": sum(it["differs"] for it in per_item), "batch": batch,
            "lengths": lengths, "max_abs_diff": max((it["max_abs_diff"] for it in per_item), default=None),
            "min_margin_b1": min((it["margin_b1"] for it in per_item), default=None),
            "largest_diff_item": big and {k: big[k] for k in ("index", "length", "max_abs_diff", "diff_steps")},
            "max_differential_steps": max((it["max_differential_steps"] for it in per_item), default=None),
            "per_item": per_item}


def golden_gate(g: dict, gate: dict) -> dict:
    """The golden test's gate (re-registered 14:34 UTC 1 Nov, config.yaml's mmlu.golden_gate) on its record: the
    run stops if (a) any item's largest |logit difference| is above max_diff_steps bf16 steps, (b) any differing
    item's batch-1 margin is above flip_margin_steps steps, or (c) more than max_flip_share of the items differ. A
    non-finite difference fires (a). Returns each part, the items that fire it, and "stop"."""
    items = g.get("per_item") or []
    a = [it["index"] for it in items if not it["diff_steps"] <= gate["max_diff_steps"]]
    b = [it["index"] for it in items if it["differs"] and not it["margin_steps"] <= gate["flip_margin_steps"]]
    c = g["disagreements"] > gate["max_flip_share"] * g["items"]
    return {"thresholds": dict(gate), "a": {"fired": bool(a), "items": a}, "b": {"fired": bool(b), "items": b},
            "c": {"fired": c, "differing": g["disagreements"], "items": g["items"]}, "stop": bool(a or b or c)}


def gate_message(g: dict, batch: int) -> str:
    """Why the golden test's gate stopped a run, numbers only."""
    gate, t = g["gate"], g["gate"]["thresholds"]
    parts = []
    if gate["a"]["fired"]:
        parts.append(f"(a) items {gate['a']['items']} differ by more than {t['max_diff_steps']:g} bf16 steps")
    if gate["b"]["fired"]:
        parts.append(f"(b) items {gate['b']['items']} flip with a batch-1 margin above "
                     f"{t['flip_margin_steps']:g} steps")
    if gate["c"]["fired"]:
        parts.append(f"(c) {gate['c']['differing']} of {gate['c']['items']} items differ, more than "
                     f"{100 * t['max_flip_share']:g}%")
    return f"the MMLU golden test (batch {batch} against batch 1) stopped the run: {'; '.join(parts)}"


def mmlu_golden(model, tok, rows: list[dict], letters: list[int], n: int, batch: int) -> dict:
    """Batch `batch` against batch 1 on n items spread over the prompt lengths (at evenly spaced length quantiles),
    every item recorded (golden_item, golden_record; the coordinator's request after the smoke and the gate's
    re-registration, 1 Nov). golden_gate reads the record. Numbers only."""
    from anatta.bench import mc

    texts = mmlu_texts(rows)
    lengths = [len(r) for r in mc.encode(tok, texts)]
    order = sorted(range(len(texts)), key=lambda i: (lengths[i], i))
    pick = sorted({order[round(k * (len(order) - 1) / max(1, n - 1))] for k in range(n)})
    sub = [texts[i] for i in pick]
    a = mc.predict(model, tok, sub, letters, batch_size=batch)
    b = mc.predict(model, tok, sub, letters, batch_size=1)
    return golden_record([golden_item(i, lengths[i], b["logits"][j], a["logits"][j]) for j, i in enumerate(pick)],
                         batch)


def golden_stage(model, tok, rows: list[dict], letters: list[int], plan: dict) -> dict:
    """mmlu_golden at the plan's settings, with its gate's verdict (golden_gate) in the record."""
    mm = plan["mmlu"]
    g = mmlu_golden(model, tok, rows, letters, mm["golden_items"], mm["batch"])
    return {**g, "gate": golden_gate(g, mm["golden_gate"])}


def golden_log(g: dict) -> list[str]:
    """The golden test for the container's log, numbers only: its counts, its largest difference and differential in
    bf16 steps, the gate's verdict, and each differing item with its margin, difference and margin change."""
    gate, big = g["gate"], g.get("largest_diff_item") or {}
    verdict = f"stop ({', '.join(k for k in 'abc' if gate[k]['fired'])})" if gate["stop"] else "passed"
    lines = [f"MMLU golden test: {g['disagreements']} of {g['items']} items differ (batch {g['batch']} against 1); "
             f"gate {verdict}"]
    if big:
        lines[0] += (f"; max |logit diff| {big['max_abs_diff']:.4f} ({big['diff_steps']:.1f} steps, item "
                     f"{big['index']}, length {big['length']}); largest differential "
                     f"{g['max_differential_steps']:.1f} steps")
    for it in g.get("per_item") or []:
        if it["differs"]:
            lines.append(f"MMLU golden test: item {it['index']} (length {it['length']}): batch-1 margin "
                         f"{it['margin_b1']:.4f} ({it['margin_steps']:.1f} steps), max |logit diff| "
                         f"{it['max_abs_diff']:.4f} ({it['diff_steps']:.1f} steps), margin change "
                         f"{it['margin_change_steps']:.1f} steps, letter {it['pred_b1']} at batch 1, "
                         f"{it['pred_batched']} batched (batch {it['batch_index']}, lengths "
                         f"{min(it['batch_lengths'])}-{max(it['batch_lengths'])})")
    return lines


# The first condition of each model's run: the MMLU floor reads its MMLU on every row.
FLOOR_CONDITION = {"qwen": "Q0", "ministral": "M0"}


def mmlu_floor(rec: dict, plan: dict) -> dict:
    """The common-mode floor (registered with the golden test's gate, 14:34 UTC 1 Nov). The golden test sees only
    what differs between the batch sizes, so a fault common to both (a wrong letter id, prompt or token read, or a
    tokenizer fault) passes it; the first condition's MMLU on every row (FLOOR_CONDITION) must reach config.yaml's
    mmlu.floor_pct, in percent. A full run stops below it; a smoke run's few items record it and do not apply it."""
    mm, floor = rec["mmlu"], plan["mmlu"]["floor_pct"]
    below, applies = mm["acc"] < floor, not plan["smoke"]
    return {"condition": rec["name"], "n": mm["n"], "correct": mm["correct"], "acc": mm["acc"], "floor_pct": floor,
            "below": below, "applies": applies, "stop": applies and below}


def conditions_for(key: str, m: dict, v: dict, model, tok) -> list[dict]:
    """The registered conditions of one model, in run order: {name, hooks (a callable giving the context manager
    that installs them, or None), runs (jbb, mmlu, mmlu_chat), rp ("graded" or "lab"), ladder (run on a ladder
    candidate), skip (why it cannot run, or None)}. v holds the vectors: rhat, r, layer (l*), rhat_ref, r_ref, modeb,
    ql, and each choice."""
    import reference as R

    from anatta.dial import Intervention, everywhere, intervene, random_like

    def dial(ivs):
        return lambda: intervene(model, ivs)

    def a(d):  # mode A: block 0's input and every block output
        return dial(everywhere(model, d))

    def need(*names):
        return next((f"no {n}" for n in names if v.get(n) is None), None)

    lo, hi = m["band_b"]
    lo2, hi2 = m["band_b2"]
    band = range(lo, hi + 1)
    modeb = v.get("modeb") or {}
    rhat, ladder = v.get("rhat"), v.get("ladder", False)
    conds = []

    def add(name, hooks, runs, rp="graded", skip=None, lad=False):
        conds.append({"name": name, "hooks": hooks, "runs": runs, "rp": rp, "skip": skip, "ladder": lad})

    if key == "qwen":
        add("Q0", None, ("jbb", "mmlu"))
        add("QA", a(rhat) if rhat is not None else None, ("jbb", "mmlu"), skip=need("rhat"), lad=ladder)
        add("QA+", dial(everywhere(model, rhat, attn=True)) if rhat is not None else None, ("jbb", "mmlu"),
            skip=need("rhat"), lad=ladder)
        if v.get("r_ref") is not None:
            mb = R.model_base_for(model, tok, m["eoi_toks"], m["refusal_toks"])

            def r_hooks(r=v["r_ref"]):  # Arditi's hook set, built once for the condition (his closures drift)
                pre, fwd = R.get_all_direction_ablation_hooks(mb, r.to(next(model.parameters()).device))
                return lambda: R.add_hooks(pre, fwd)
            add("QR", None, ("jbb", "mmlu"), lad=v.get("ladder_ref", False))
            conds[-1]["hooks_factory"] = r_hooks
        else:
            add("QR", None, ("jbb", "mmlu"), skip="no reference candidate")
        differ = v.get("r_ref") is not None and (v.get("choice") or {}).get("pl") != (v.get("choice_ref") or {}).get("pl")
        if differ:
            # QX also runs when QA has no candidate and QR has one: they choose different candidates (ruling, 1 Nov)
            add("QX", a(unit(v["r_ref"])), ("jbb",), lad=v.get("ladder_ref", False))
        add("QB", dial([Intervention(b, modeb[b], "ablate") for b in band]) if modeb else None, ("jbb", "mmlu"),
            skip=None if modeb else "no mode B vectors")
        add("QB2", dial([Intervention(b, modeb[b], "ablate") for b in range(lo2, hi2 + 1)]) if modeb else None,
            ("jbb",), skip=None if modeb else "no mode B vectors")
        add("QC", dial([Intervention(b, rhat, "ablate") for b in band]) if rhat is not None else None, ("jbb", "mmlu"),
            skip=need("rhat"), lad=ladder)
        if v.get("r") is not None:
            r, layer = v["r"], v["layer"]
            iv = (Intervention(layer - 1, r, "steer", alpha=-float(r.norm())) if layer >= 1
                  else Intervention(0, r, "steer", alpha=-float(r.norm()), site="input"))
            add("QS", dial([iv]), ("jbb",), lad=ladder)
        else:
            add("QS", None, ("jbb",), skip="no rhat")
        add("QL", a(v["ql"]) if v.get("ql") is not None else None, ("jbb",), skip=need("ql"), lad=ladder)
        add("QA-rand", a(random_like(rhat, seed=0)) if rhat is not None else None, ("jbb",), skip=need("rhat"))
        add("QB-rand", dial([Intervention(b, random_like(modeb[b], seed=b), "ablate") for b in band]) if modeb else None,
            ("jbb", "mmlu"), skip=None if modeb else "no mode B vectors")
        add("Q0-rp", None, ("jbb",), rp="lab")
        add("QA-rp", a(rhat) if rhat is not None else None, ("jbb",), rp="lab", skip=need("rhat"), lad=ladder)
        add("Q0-chat", None, ("mmlu_chat",))
        add("QA-chat", a(rhat) if rhat is not None else None, ("mmlu_chat",), skip=need("rhat"), lad=ladder)
        add("Q0-end", None, ("jbb",))
    else:
        add("M0", None, ("jbb", "mmlu"))
        add("MA", a(rhat) if rhat is not None else None, ("jbb", "mmlu"), skip=need("rhat"), lad=ladder)
        add("MB", dial([Intervention(b, modeb[b], "ablate") for b in band]) if modeb else None, ("jbb", "mmlu"),
            skip=None if modeb else "no mode B vectors")
        add("MB2", dial([Intervention(b, modeb[b], "ablate") for b in range(lo2, hi2 + 1)]) if modeb else None,
            ("jbb",), skip=None if modeb else "no mode B vectors")
        add("ML", a(v["ql"]) if v.get("ql") is not None else None, ("jbb",), skip=need("ql"), lad=ladder)
        add("MB-rand", dial([Intervention(b, random_like(modeb[b], seed=b), "ablate") for b in band]) if modeb else None,
            ("jbb", "mmlu"), skip=None if modeb else "no mode B vectors")
    return conds


def run_condition(model, tok, cond: dict, m: dict, plan: dict, prompts: dict, rows: list[dict], mdir: Path,
                  log=print) -> dict:
    """One condition: the hook check, then its JBB completions (kept on the volume) and its MMLU passes, with its
    hooks installed for each and removed after. Returns its record (numbers and sha256s)."""
    from contextlib import nullcontext

    from anatta.dial import assert_no_hooks

    g, mm = plan["generation"], plan["mmlu"]
    rp = g["repetition_penalty"] if cond["rp"] == "graded" else g["repetition_penalty_lab"]
    hooks = cond["hooks_factory"]() if cond.get("hooks_factory") else cond["hooks"]
    rec = {"name": cond["name"], "ladder": cond["ladder"], "runs": list(cond["runs"])}
    for what in cond["runs"]:
        assert_no_hooks(model)
        with (hooks() if hooks is not None else nullcontext()):
            if what == "jbb":
                comps, facts = generate_jbb(model, tok, prompts["jbb"], m["gen_batch"], g["max_new_tokens"], rp,
                                            m["eos_ids"], plan["aggregation"]["refusal_substrings"])
                write_json(mdir / "completions" / f"{cond['name']}.json", comps)
                rec["jbb"] = facts
            else:
                rec[what] = run_mmlu(model, tok, rows, m["letters"], mm["batch"], chat=what == "mmlu_chat")
        assert_no_hooks(model)
        log(f"{cond['name']}: {what} done")
    return rec


def run_model(plan: dict, key: str, out, model, tok, pkg_dir, log=print, deadline: float | None = None,
              ego=None) -> dict:
    """One model's run (the module docstring), resumable: each stage's JSON is kept in the run's folder and read back
    on a rerun of the same plan. On the model that reads v_ego (plan["ego"]["model"]), `ego` is the path of the
    image's copy of the registered file (or None), for the ledger's E-82 and E-82r (ego_reading). Returns the model's
    record: numbers, counts and sha256s. A run that stopped (at the MMLU golden test's gate, at filtering, at a smoke
    run's failed bitwise premise, or at a full run's MMLU floor) returns its stopped record, with "stopped" and a
    "message"."""
    import reference as R
    import torch

    from anatta import contrast, select
    from anatta.dial import assert_no_hooks
    from anatta.models import get_layers

    env = check_environment(plan, "lab", pkg_dir)
    out = Path(out)
    claim(out, plan)
    m, smoke = plan["models"][key], plan["smoke"]
    mdir = out / key
    tdir = mdir / "tensors"
    tdir.mkdir(parents=True, exist_ok=True)
    check_data(plan, out / DATA)
    prompts = load_prompts(plan, out / DATA)
    rows = mmlu_rows(out / DATA / "mmlu" / "test.json")
    if smoke:
        prompts["jbb"] = prompts["jbb"][:plan["smoke_settings"]["jbb"]]
        rows = rows[:plan["smoke_settings"]["mmlu"]]
    n = len(get_layers(model))
    if n != m["n_layers"]:
        raise CheckError(f"{m['id']} has {n} blocks, not the registered {m['n_layers']}")
    positions = list(range(-len(m["eoi_toks"]), 0))
    sel = plan["selection"]
    reg = select.LADDER[0]
    if (sel["kl_threshold"], sel["induce_refusal_threshold"]) != (reg["kl_threshold"], reg["induce_refusal_threshold"]):
        raise CheckError("config.yaml's selection thresholds differ from anatta.select.LADDER's first rung")

    def check_deadline(what: str) -> None:
        if deadline is not None and time.time() > deadline:
            raise CheckError(f"the GPU budget ran out before {what}: rerun the same plan to resume")

    facts = {"model": m["id"], "revision": m["revision"], "environment": env, "freeze": freeze(),
             "attn_implementation": getattr(model.config, "_attn_implementation", None),
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             "torch": torch.__version__, "generation_config": generation_config_facts(model),
             "tokens": token_checks(tok, m)}

    # The MMLU golden test, first: batch 32 against batch 1 under config.yaml's golden_gate (re-registered 14:34 UTC
    # 1 Nov, and moved here from after the selection so a stop costs minutes, not the selection; still before any
    # condition). It measures the model's own batching, so no intervention may be on: the hook check runs right
    # before it, as before every condition.
    check_deadline("the MMLU golden test")
    assert_no_hooks(model)
    golden = stage(mdir / "mmlu_golden.json", lambda: golden_stage(model, tok, rows, m["letters"], plan), log)
    for line in golden_log(golden):
        log(line)
    if golden["gate"]["stop"]:
        # The run stops here, and its stopped record comes back like a finished one, so the golden test's record
        # reaches the laptop and the card (main stops on it).
        return written(mdir / "result.json", {"facts": facts, "mmlu_golden": golden, "stopped": "mmlu_golden",
                                              "message": gate_message(golden, plan["mmlu"]["batch"])})
    assert_no_hooks(model)

    # filtering, once, shared by both paths and C3-extract
    kept_path = mdir / "kept.json"
    if read_json(mdir / "filter.json") is None or read_json(kept_path) is None:
        try:
            rec, kept = filter_stage(model, tok, m, prompts, plan["filter"]["batch"], log)
        except EmptyKept as e:
            # The model's run stops here and returns its stopped record, so the other model's run can still be
            # judged and graded (ruling, 1 Nov); the card reads the stop as the refusal tokens' or the data's, not
            # the instrument's or the selection's.
            log(f"filter: {e}")
            assert_no_hooks(model)
            return written(mdir / "result.json", {"facts": facts, "mmlu_golden": golden, "filter": e.args[1],
                                                  "stopped": EMPTY_KEPT, "message": str(e)})
        write_json(kept_path, kept)
        written(mdir / "filter.json", rec)
    filt, kept = read_json(mdir / "filter.json"), read_json(kept_path)
    sets = {s: [prompts[s][i] for i in kept[s]] for s in SPLITS}
    assert_no_hooks(model)

    # the library path's candidates (anatta.contrast), and on Qwen the reference path's (reference.py)
    def saved(name: str, fn):
        p = tdir / f"{name}.pt"
        if p.exists():
            return torch.load(p)
        t = fn()
        torch.save(t, p)
        return t

    check_deadline("the extraction")
    layers = list(range(n))
    lib = saved("candidates_library", lambda: contrast.mean_diffs(model, tok, sets["harmful_train"],
                                                                  sets["harmless_train"], layers, positions,
                                                                  site="input"))
    mb = R.model_base_for(model, tok, m["eoi_toks"], m["refusal_toks"])
    ref = None
    if m["reference"]:
        ref = saved("candidates_reference", lambda: R.generate_directions(mb, sets["harmful_train"],
                                                                         sets["harmless_train"], batch_size=32).cpu())
    extract = {"library_sha256": tensor_sha256(lib), "reference_sha256": tensor_sha256(ref) if ref is not None else None}
    if m["c3_extract"]:
        def run_c3x():
            rec, b1 = c3_extract(model, tok, mb, lib, ref, sets["harmful_train"], sets["harmless_train"], positions, log)
            torch.save(b1, tdir / "candidates_reference_b1.pt")
            return rec
        extract["c3_extract"] = stage(mdir / "c3_extract.json", run_c3x, log)
        if smoke and not extract["c3_extract"]["bitwise_premise"]:
            # stopped, and the record comes back like a finished one, so it reaches the card (main stops on it)
            return written(mdir / "result.json", {
                "facts": facts, "mmlu_golden": golden, "filter": filt, "extract": extract, "stopped": "bitwise_premise",
                "message": ("C3-extract's bitwise premise failed on the smoke run (layers "
                            f"{extract['c3_extract']['premise_unequal_layers']}): fix it before launch")})
    assert_no_hooks(model)

    # selection: the library path (anatta.select with the dial's hooks) and the reference path (Arditi's hooks)
    check_deadline("the selection")
    which = smoke_which(n) if smoke else None
    tok_fn = tokenize_fn(tok)
    scores = stage(mdir / "select_library.json", lambda: select.score_candidates(
        model, lib, sets["harmful_val"], sets["harmless_val"], tok_fn, m["refusal_toks"], *select.dial_hooks(model),
        batch_size=sel["batch"], which=which, log=log), log)
    choice = choose(scores, smoke)
    assert_no_hooks(model)
    scores_ref, choice_ref = None, None
    if m["reference"]:
        scores_ref = stage(mdir / "select_reference.json", lambda: R.select_direction(
            mb, sets["harmful_val"], sets["harmless_val"], ref, batch_size=sel["batch"], which=which, log=log), log)
        choice_ref = choose(scores_ref, smoke)
        if choice_ref["rung"] == 0 and scores_ref["arditi_pick"] != {"pos": choice_ref["pos"],
                                                                     "layer": choice_ref["layer"]}:
            raise CheckError("Arditi's own pick and anatta.select.choose's first rung disagree on the reference "
                             "tables")
        assert_no_hooks(model)

    # the vectors (kept on the volume; only their norms, cosines and sha256s leave it)
    v = {"choice": {"pl": [choice["pos"], choice["layer"]]} if choice["found"] else None,
         "choice_ref": {"pl": [choice_ref["pos"], choice_ref["layer"]]} if choice_ref and choice_ref["found"] else None,
         "ladder": bool(choice["found"] and choice["rung"] != 0),
         "ladder_ref": bool(choice_ref and choice_ref["found"] and choice_ref["rung"] != 0)}
    vec = {}
    lo, hi = m["band_b"]
    pos_row = len(positions) - 1  # position -1
    vec["modeb"] = {b: lib[pos_row, b + 1].clone() for b in range(lo, hi + 1)}  # block output b = Arditi's layer b + 1
    if choice["found"]:
        vec["r"] = lib[choice["pos"], choice["layer"]].clone()
        vec["layer"] = choice["layer"]
        vec["rhat"] = unit(vec["r"])
        vec["ql"] = saved("inhouse", lambda: contrast.mean_diffs(model, tok, contrast.HARMFUL, contrast.HARMLESS,
                                                                  [choice["layer"]], [plan["inhouse"]["position"]],
                                                                  site="input")[0, 0])
    if choice_ref is not None and choice_ref["found"]:
        vec["r_ref"] = ref[choice_ref["pos"], choice_ref["layer"]].clone()
        vec["rhat_ref"] = unit(vec["r_ref"])
    torch.save({k: x for k, x in vec.items() if k != "layer"}, tdir / "vectors.pt")
    v.update(vec)
    geometry = {"norm_r": float(vec["r"].norm()) if "r" in vec else None,
                "norm_r_ref": float(vec["r_ref"].norm()) if "r_ref" in vec else None,
                "rhat_sha256": tensor_sha256(vec["rhat"]) if "rhat" in vec else None,
                "rhat_ref_sha256": tensor_sha256(vec["rhat_ref"]) if "rhat_ref" in vec else None,
                "cos_rhat_ref": cos(vec["rhat"], vec["rhat_ref"]) if "rhat" in vec and "rhat_ref" in vec else None,
                "cos_ql_rhat": cos(vec["ql"], vec["rhat"]) if "ql" in vec else None,
                "modeb_vs_rhat": {b: cos(x, vec["rhat"]) for b, x in vec["modeb"].items()} if "rhat" in vec else None,
                "modeb_consecutive": {b: cos(vec["modeb"][b - 1], vec["modeb"][b]) for b in range(lo + 1, hi + 1)}}

    base = {"facts": facts, "filter": filt, "extract": extract, "select": {"library": scores, "choice": choice,
            "reference": scores_ref, "choice_ref": choice_ref}, "geometry": geometry, "mmlu_golden": golden}

    # the conditions, each after the hook check, Q0-end last; the MMLU floor after the first (Q0, M0)
    conds = {}
    for cond in conditions_for(key, m, v, model, tok):
        name = cond["name"]
        if cond["skip"]:
            conds[name] = {"name": name, "skipped": cond["skip"], "ladder": cond["ladder"]}
            continue
        check_deadline(name)
        conds[name] = stage(mdir / "conditions" / f"{name}.json",
                            lambda c=cond: run_condition(model, tok, c, m, plan, prompts, rows, mdir, log), log)
        if name == FLOOR_CONDITION[key]:
            fl = base["mmlu_floor"] = mmlu_floor(conds[name], plan)
            log(f"MMLU floor: {name} {fl['acc']:.2f}% on {fl['n']} items against {fl['floor_pct']:g}%"
                + ("" if fl["applies"] else " (a smoke run: recorded, not applied)"))
            if fl["stop"]:
                # stopped, and the record comes back like a finished one, so it reaches the card (main stops on it)
                return written(mdir / "result.json", {
                    **base, "conditions": conds, "stopped": FLOOR_STOP,
                    "message": (f"the MMLU floor: {name} scored {fl['acc']:.2f}% on {fl['n']} items, below "
                                f"{fl['floor_pct']:g}%: no other condition runs")})
    if key == "qwen" and "jbb" in conds.get("Q0-end", {}) and "jbb" in conds.get("Q0", {}):
        a = json.loads((mdir / "completions" / "Q0.json").read_text())
        b = json.loads((mdir / "completions" / "Q0-end.json").read_text())
        base["q0_end"] = {"identical": sum(x == y for x, y in zip(a, b)), "n": len(a)}
    base["conditions"] = conds
    if key == plan["ego"]["model"]:
        base["ledger"] = ego_reading(model, tok, vec["modeb"], ego, plan, m)
    assert_no_hooks(model)
    return written(mdir / "result.json", base)


def ego_reading(model, tok, modeb: dict, path, plan: dict, m: dict) -> dict:
    """Prediction 82's re-read with the registered v_ego (plan["ego"]): the file's sha256 is checked before it is
    loaded, and E-82 and E-82r are read only if it is the registered one. A missing or different file marks both not
    read, with the reason, and the run goes on: they are ungraded (ruling, 1 Nov). The record carries the file's
    sha256 and its layer list (the layers only once the file is loaded)."""
    reg = plan["ego"]["sha256"]
    info = {"file": str(path) if path is not None else None, "registered_sha256": reg, "sha256": None, "layers": None}
    if path is None or not Path(path).is_file():
        return {"ego": info, "not_read": "v_ego is not in the image, so E-82 and E-82r are not read"}
    info["sha256"] = sha256_file(path)
    if info["sha256"] != reg:
        return {"ego": info, "not_read": f"v_ego has sha256 {info['sha256']}, not the registered {reg}, so E-82 and "
                                          "E-82r are not read"}
    from anatta.probe import Direction

    vectors = Direction.load(path).vectors
    info["layers"] = sorted(int(b) for b in vectors)
    return {"ego": info, **ego_ledger(model, tok, modeb, vectors, plan, m)}


def ego_ledger(model, tok, modeb: dict, ego: dict, plan: dict, m: dict) -> dict:
    """Prediction 82's re-read, for the ledger only (it uses the lab's v_ego): E-82, |cos| of mode B's refusal vectors
    with v_ego at each layer of its band; E-82r, the library extractor's refusal direction from the in-house pairs
    (anatta.contrast.refusal_direction, block outputs, the last token) against v_ego, its max |cos| and layer; and the
    cosine between consecutive layers' v_ego vectors."""
    from anatta import contrast

    e = plan["registered"]["expectations"]
    lo, hi = e["e_82"]["layers"]
    layers = [b for b in range(lo, hi + 1) if b in ego]
    e82 = {b: abs(cos(modeb[b], ego[b])) for b in layers if b in modeb}
    vecs, _ = contrast.refusal_direction(model, tok, layers)
    e82r = {b: abs(cos(vecs[b], ego[b])) for b in layers}
    top = max(e82r, key=e82r.get) if e82r else None
    return {"e_82": {"abs_cos": e82, "max": max(e82.values()) if e82 else None},
            "e_82r": {"abs_cos": e82r, "max": e82r[top] if top is not None else None, "layer": top},
            "ego_consecutive": {b: cos(ego[b - 1], ego[b]) for b in layers if b - 1 in ego}}


# ------------------------------------------------------------------------------------------------- the judge


def completions_of(out, key: str, name: str) -> list[str] | None:
    return read_json(Path(out) / key / "completions" / f"{name}.json")


def jbb_prompts(plan: dict, out) -> list[str]:
    p = load_prompts(plan, Path(out) / DATA)["jbb"]
    return p[:plan["smoke_settings"]["jbb"]] if plan["smoke"] else p


def scored_conditions(out) -> dict[str, list[str]]:
    """{model: [the conditions with JBB completions on the volume]}, in the order written."""
    found = {}
    for key in MODELS:
        d = Path(out) / key / "completions"
        found[key] = sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []
    return found


# google/gemma-2b's attention (its config.json): 8 query heads and one key/value head, which transformers repeats to 8
# before the attention call in a padded batch, each of dimension 256.
GEMMA_HEADS, GEMMA_HEAD_DIM = 8, 256
SDPA_LENGTH = 512  # one typical length of a judge prompt in tokens: the template, the prompt and the response
SDPA_CHECKS = (("flash", "can_use_flash_attention"), ("efficient", "can_use_efficient_attention"),
               ("cudnn", "can_use_cudnn_attention"))
SDPA_FLAGS = (("flash", "flash_sdp_enabled"), ("efficient", "mem_efficient_sdp_enabled"), ("math", "math_sdp_enabled"),
              ("cudnn", "cudnn_sdp_enabled"))


def sdpa_params(bc, q, k, v, mask, causal: bool):
    """torch.backends.cuda.SDPAParams (`bc` is torch.backends.cuda) over q, k, v and the mask, without dropout, and the
    number of arguments its constructor took: seven, (q, k, v, attn_mask, dropout, is_causal, enable_gqa), in torch
    2.5 and later (the paper's torch 2.5.1 and the lab's); six, without enable_gqa, before 2.5."""
    try:
        return bc.SDPAParams(q, k, v, mask, 0.0, causal, False), 7
    except TypeError:
        return bc.SDPAParams(q, k, v, mask, 0.0, causal), 6


def sdpa_backends(batch: int, length: int = SDPA_LENGTH) -> dict:
    """Which of PyTorch's attention kernels its own checks accept for the judge's attention in this container,
    recorded before scoring, not graded (ruling A's open question: the original package scores with gradients on,
    the port under torch.no_grad()). q, k and v are bf16 zeros of shape (batch, 8, length, 256), Gemma-2b's, on the
    GPU. Each case is checked twice, as each judge's forward runs: under torch.enable_grad() with requires_grad (the
    original) and under torch.no_grad() without (the port). The cases: a padded batch, as the judge's batches reach
    attention, with a (batch, 1, length, length) mask and not causal, the mask once as a bf16 additive one (as
    transformers 4.47, the paper's stack, builds it) and once boolean (as the lab's transformers builds it); and an
    unpadded batch, no mask and causal (the paper's call; the lab's transformers passes the one key/value head with
    enable_gqa there instead, which is not checked). The checks are torch.backends.cuda's can_use_flash_attention and
    can_use_efficient_attention, and can_use_cudnn_attention where this torch has it. The math kernel has no such
    check (it is the fallback for any shape), so its global flag is recorded with the others'. A check says whether a
    kernel would be accepted, not which one the dispatcher picks among those accepted.

    Booleans only, beside the shape and the number of arguments SDPAParams took. If anything here fails, the record
    is {"checked": False, ...} and the judge goes on."""
    import torch
    from torch.backends import cuda as bc

    out = {"checked": False, "cuda": torch.cuda.is_available(),
           "shape": [batch, GEMMA_HEADS, length, GEMMA_HEAD_DIM], "dtype": "torch.bfloat16"}
    try:
        enabled = {name: bool(getattr(bc, f)()) for name, f in SDPA_FLAGS if hasattr(bc, f)}
        checks = {name: getattr(bc, f) for name, f in SDPA_CHECKS if hasattr(bc, f)}
        device = "cuda" if out["cuda"] else "cpu"
        masks = {"padded_float_mask": lambda: torch.zeros(batch, 1, length, length, dtype=torch.bfloat16, device=device),
                 "padded_bool_mask": lambda: torch.ones(batch, 1, length, length, dtype=torch.bool, device=device),
                 "unpadded": lambda: None}
        accepted, n_args = {}, None
        for case, make_mask in masks.items():
            accepted[case] = {}
            for mode, context in (("no_grad", torch.no_grad), ("grad", torch.enable_grad)):
                with context():
                    q, k, v = (torch.zeros(out["shape"], dtype=torch.bfloat16, device=device,
                                           requires_grad=mode == "grad") for _ in range(3))
                    mask = make_mask()
                    params, n_args = sdpa_params(bc, q, k, v, mask, mask is None)
                    accepted[case][mode] = {name: bool(fn(params, False)) for name, fn in checks.items()}
                del q, k, v, mask, params
    except Exception:  # noqa: BLE001 (a recorded fact, never a stop)
        return out
    return {**out, "checked": True, "params_args": n_args, "enabled": enabled, "accepted": accepted}


def judge_lab(plan: dict, out, pkg_dir, model, tokenizer, template: str, log=print, loaded: dict | None = None) -> dict:
    """(c): the port in the lab's stack on every condition's completions, and the judge control (one fixed refusal
    paired with each JBB prompt). Returns per-completion scores and how many responses the judge truncated, with the
    template's sha256, how the judge loaded (`loaded`, from judge.load_judge: its dtype and whether the fallback
    branch loaded it) and, recorded before scoring, the attention kernels this container accepts (sdpa_backends)."""
    import judge as J

    env = check_environment(plan, "lab", pkg_dir)
    j = plan["judge"]
    prompts = jbb_prompts(plan, out)
    res = {"environment": env, "freeze": freeze(), "template_sha256": J.template_sha256(template), "judge": loaded,
           "sdpa": sdpa_backends(j["batch"]), "scores": {}}
    for key, names in scored_conditions(out).items():
        res["scores"][key] = {}
        for name in names:
            comps = completions_of(out, key, name)
            res["scores"][key][name] = {
                "scores": J.evaluate(prompts, comps, model, tokenizer, template, j["batch"], j["max_response_length"]),
                "judge_truncated": J.truncated_count(tokenizer, comps, j["max_response_length"])}
            log(f"judge (lab): {key} {name}")
    control = [j["control_response"]] * len(prompts)
    res["control"] = J.evaluate(prompts, control, model, tokenizer, template, j["batch"], j["max_response_length"])
    return written(Path(out) / "judge_lab.json", res)


def check_package_commit(commit: str) -> dict:
    """strong_reject's installed commit, as pip recorded it (direct_url.json), against the registered one."""
    dist = metadata.distribution("strong_reject")
    info = json.loads(dist.read_text("direct_url.json") or "{}")
    got = (info.get("vcs_info") or {}).get("commit_id")
    if got != commit:
        raise CheckError(f"strong_reject is installed at {got}, not the registered {commit}")
    return {"version": dist.version, "commit": got}


def original_scores(prompts: list[str], responses: list[str], batch: int) -> tuple[list[float], dict]:
    """(a): the original strong_reject package's strongreject_finetuned through its evaluate_dataset, and how the
    package loaded its judge (original_judge_facts). C3: evaluate_dataset is called with empty_model_cache=False, so
    that the judge it cached can be read before it goes, and release_original_judge then does what the default
    (True) does, in the package's own lines. The flag is read only after the batched map that makes the scores
    (evaluate.py:757-771 at e286f0d), so the scores are the same either way."""
    from datasets import Dataset
    from strong_reject.evaluate import evaluate_dataset

    ds = Dataset.from_dict({"forbidden_prompt": prompts, "response": responses})
    try:
        out = evaluate_dataset(ds, ["strongreject_finetuned"], batch_size=batch, empty_model_cache=False)
        facts = original_judge_facts()
    finally:
        release_original_judge()
    return [float(x) for x in out["score"]], facts


def release_original_judge() -> None:
    """What evaluate_dataset does after an evaluation with empty_model_cache=True, its default: every model the
    package cached moved to the CPU and dropped, and the cache cleared (evaluate.py:764-770 at e286f0d, MIT,
    "Copyright (c) 2024, Dillon Bowen")."""
    import torch
    from strong_reject.evaluate import cached_models

    for key in cached_models:
        model, tokenizer = cached_models[key]
        model.to(torch.device("cpu"))
        del model

    cached_models.clear()


def original_judge_facts() -> dict:
    """How the original package loaded its judge, read off the model its evaluate module caches as it scores
    (cached_models["strongreject_finetuned"] = model, tokenizer: evaluate.py:115 and :480 at e286f0d): its dtype, the
    first parameter's as the port records it, and the set over all its parameters. Which branch of its try/except
    loaded it leaves no trace, so that is "not observable"; so is the dtype if nothing is cached."""
    out = {"dtype": "not observable", "fallback": "not observable"}
    try:
        import strong_reject.evaluate as ev

        entry = (getattr(ev, "cached_models", None) or {}).get("strongreject_finetuned")
        if isinstance(entry, (tuple, list)) and entry:
            out["dtype"] = str(next(entry[0].parameters()).dtype)
            out["param_dtypes"] = sorted({str(p.dtype) for p in entry[0].parameters()})
    except Exception:  # noqa: BLE001 (a fact for the card, never a stop)
        pass
    return out


def package_template_sha256() -> str:
    """The sha256 (UTF-8) of the strongreject_finetuned template as the installed strong_reject package holds it: its
    JUDGE_TEMPLATES, which its evaluate module reads from its own judge_templates.json at import, in text mode with
    this container's default encoding (evaluate.py:112-113). The card sets it beside judge.py's copy, so a copy or a
    decoding that differs (the template has one non-ASCII character) shows."""
    import judge as J
    from strong_reject.evaluate import JUDGE_TEMPLATES

    return J.template_sha256(JUDGE_TEMPLATES["strongreject_finetuned"])


def judge_paper(plan: dict, out, pkg_dir, which: dict[str, list[str]], ports: tuple[str, ...], template: str,
                log=print) -> dict:
    """In the paper's stack: (a) the original package and/or (b) the port on the given conditions' completions
    ({model: [names]}); "original" and "port" in `ports` choose which. Records the template's sha256 (the port's
    copy, and with the original, the installed package's own), how each judge loaded (the port's dtype and branch;
    the original's dtype) and, before scoring, the attention kernels this container accepts (sdpa_backends)."""
    import judge as J

    env = check_environment(plan, "paper", pkg_dir)
    pkg = check_package_commit(plan["judge"]["package_commit"])
    j = plan["judge"]
    prompts = jbb_prompts(plan, out)
    res = {"environment": env, "freeze": freeze(), "package": pkg, "template_sha256": J.template_sha256(template),
           "sdpa": sdpa_backends(j["batch"]), "original": {}, "port": {}}
    if "original" in ports:
        res["package_template_sha256"] = package_template_sha256()
    model = tokenizer = None
    for key, names in which.items():
        for name in names:
            comps = completions_of(out, key, name)
            if comps is None:
                continue
            if "original" in ports:
                res["original"].setdefault(key, {})[name], res["original_judge"] = original_scores(prompts, comps,
                                                                                                   j["batch"])
            if "port" in ports:
                if model is None:
                    model, tokenizer, res["port_judge"] = J.load_judge()
                res["port"].setdefault(key, {})[name] = J.evaluate(prompts, comps, model, tokenizer, template,
                                                                   j["batch"], j["max_response_length"])
            log(f"judge (paper): {key} {name}")
    return res


def load_paper_model(snap):
    """Qwen as Wollschläger's code loads it in the paper's stack: bf16, device_map "auto", left padding."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(str(snap), torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(str(snap))
    tok.padding_side = "left"
    return model, tok


def swap(plan: dict, out, pkg_dir, which: str, mmlu: bool = False, log=print, load=load_paper_model) -> dict:
    """The localizing swaps, in the paper's stack, on Qwen: "q0" regenerates Q0 with Arditi's generate_completions
    and scores it with the original judge; "reference" re-runs the reference path (generate_directions,
    select_direction and the ablation's completions, on the shared filtered sets) and scores QR's completions with
    the original judge.

    Both reuse the shared filtered sets, and a swap is JBB-only unless the miss it localizes was on the MMLU bar
    (|change| > 2 points): then `mmlu` adds the registered MMLU pass (the golden test first, then the no-template
    pass without hooks and with Arditi's ablation hooks, batch 32, right padding) to the reference swap (ruling,
    1 Nov). The Q0 swap localizes C3-base, which has no MMLU bar, so it is always JBB-only."""
    import reference as R
    import torch

    from anatta import select

    if mmlu and which != "reference":
        raise CheckError("only the reference swap localizes a miss on the MMLU bar")
    env = check_environment(plan, "paper", pkg_dir)
    m = plan["models"]["qwen"]
    out = Path(out)
    model, tok = load(snapshot_dir(m["id"], m["revision"]))
    mb = R.model_base_for(model, tok, m["eoi_toks"], m["refusal_toks"])
    prompts = load_prompts(plan, out / DATA)
    jbb = jbb_prompts(plan, out)
    dataset = [{"instruction": x, "category": ""} for x in jbb]
    g = plan["generation"]
    res = {"environment": env, "freeze": freeze(), "which": which}
    hooks = ([], [])
    if which == "reference":
        kept = read_json(out / "qwen" / "kept.json")
        sets = {s: [prompts[s][i] for i in kept[s]] for s in SPLITS}
        cands = R.generate_directions(mb, sets["harmful_train"], sets["harmless_train"], batch_size=32)
        tables = R.select_direction(mb, sets["harmful_val"], sets["harmless_val"], cands,
                                    batch_size=plan["selection"]["batch"])
        c = select.choose(tables)
        res["select"] = {"tables": tables, "choice": c}
        if not c["found"]:
            return written(out / "swap_reference.json", res)
        direction = cands[c["pos"], c["layer"]]
        res["rhat_sha256"] = tensor_sha256(unit(direction))
        hooks = R.get_all_direction_ablation_hooks(mb, direction)
    with torch.no_grad():
        comps = mb.generate_completions(dataset, fwd_pre_hooks=hooks[0], fwd_hooks=hooks[1], batch_size=m["gen_batch"],
                                        max_new_tokens=g["max_new_tokens"])
    responses = [c["response"] for c in comps]
    write_json(out / "swap" / f"{which}_completions.json", responses)
    res["completion_sha256"] = [sha256_text(r) for r in responses]
    if mmlu:
        res["mmlu"] = swap_mmlu(model, tok, plan, out, hooks, log)
    # Qwen is released before the judge loads: mb holds the model and the ablation hooks hold its blocks
    del mb, hooks, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    res["scores"], res["original_judge"] = original_scores(jbb, responses, plan["judge"]["batch"])
    return written(out / f"swap_{which}.json", res)


def swap_mmlu(model, tok, plan: dict, out: Path, hooks: tuple, log=print) -> dict:
    """The reference swap's MMLU pass: the golden test (batch 32 against batch 1) under its gate (golden_stage),
    then, if the gate passes, the no-template pass without hooks and with the swap's ablation hooks, and the change
    between them in points."""
    import reference as R

    m, mm = plan["models"]["qwen"], plan["mmlu"]
    rows = mmlu_rows(Path(out) / DATA / "mmlu" / "test.json")
    if plan["smoke"]:
        rows = rows[:plan["smoke_settings"]["mmlu"]]
    rec = {"golden": golden_stage(model, tok, rows, m["letters"], plan)}
    if rec["golden"]["gate"]["stop"]:
        return {**rec, "stopped": "mmlu_golden", "message": gate_message(rec["golden"], mm["batch"])}
    rec["base"] = run_mmlu(model, tok, rows, m["letters"], mm["batch"])
    with R.add_hooks(hooks[0], hooks[1]):
        rec["ablated"] = run_mmlu(model, tok, rows, m["letters"], mm["batch"])
    rec["change_points"] = rec["ablated"]["acc"] - rec["base"]["acc"]
    log(f"swap: MMLU {rec['base']['acc']:.2f} without hooks, {rec['ablated']['acc']:.2f} with")
    return rec


def stored(plan: dict, out) -> dict:
    """The stage results already on the volume under this plan."""
    out = Path(out)
    return {"prepare": read_json(out / "prepare.json"), "prepare_paper": read_json(out / "prepare_paper.json"),
            **{key: read_json(out / key / "result.json") for key in MODELS},
            "judge_lab": read_json(out / "judge_lab.json"), "judge_paper": read_json(out / "judge_paper.json"),
            "judge_original_all": read_json(out / "judge_original_all.json"),
            "swap_q0": read_json(out / "swap_q0.json"), "swap_reference": read_json(out / "swap_reference.json")}
