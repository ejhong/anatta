"""The ownership test and the costly-care test on Modal (predictions 122-132, lab/ledger/NOTES.md, Opus 5.5, 04:30 UTC
on 30 Oct). Private. self_tests_core.py holds the items, scenarios, prompts, parsers and scoring (unit-tested in
test_self_tests.py); self_tests_grade.py grades 122-134.

One container per model: the base model runs its hook arms (base, cut = the v_ego cut at layers 8-32,
cut_self_stakes_17_26 = the self-stakes cut at layers 17-26; cut_vs_push.build_arms, as sycophancy_eval.py builds
them), and each adapter (merged into the weights, PeftModel.merge_and_unload) runs with the hooks off. In each arm:
  ownership  every item x 3 attributions x 2 versions, greedy, MAX_NEW_OWNERSHIP tokens, batched (prompts sorted by
             length to cut padding, rows returned in the fixed order), plus the no-proposal control on every item
             (only the base model's control rows are ever read for the drop set; the others are a reading);
  care       every scenario, SAMPLES samples at temperature 0.7 (top_p 0.95): sample k is one generate() call over
             the scenarios with seed self_tests_core.care_seed(k) (more calls only if --care-batch is smaller than the
             scenario count; each row records its seed).
--adapters takes a comma list: base (the base model with --arms), a short name from ADAPTERS (pc_self2, koan_s0, ...),
a volume path (h4lite/...), or all (base and the ten adapters); every model is its own container, in parallel.

Writes lab/experiments/h4lite/results/self_tests/NAME/LABEL.{ownership,control,care}.jsonl and LABEL.meta.json, LABEL
= ADAPTER.ARM with the adapter shortened by fishery_grade.adapter_tag (base.base, base.cut, pc__pc_self2.base,
sft__koan_s0.base). A label already on disk is skipped (a rerun resumes) unless --force. As soon as base.base's control
rows are on disk, dropset.json is written from them alone (self_tests_core.drop_set) and never rewritten unless
--force; main prints parse counts only, never an acceptance or a give, so no arm is read before the drop set is fixed.

Usage (~/prj/matlab/.venv/bin/modal, from the lab root):
  modal run lab/experiments/h4lite/modal_self_tests.py::main --name self1                       (base: three hook arms)
  modal run lab/experiments/h4lite/modal_self_tests.py::main --name self1 --adapters pc_self2,pc_self2_s1,pc_neutral,\
pc_neutral_s1,koan_s0,koan_s1,koan_s2,noexit_s0,noexit_s1,noexit_s2
  modal run lab/experiments/h4lite/modal_self_tests.py::main --name smoke --items 10 --per-kind 3   (a smoke)
  modal run lab/experiments/h4lite/modal_self_tests.py::dropset --name self1    (write or check dropset.json; no GPU)
  modal run lab/experiments/h4lite/modal_self_tests.py::table --name self1      (per-label summaries; no GPU)
Grading: python lab/experiments/h4lite/self_tests_grade.py --name self1 --fishery atlas1
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import modal

H4 = str(Path(__file__).resolve().parent)  # lab/experiments/h4lite on the laptop; a container gets h4_modal from PYTHONPATH
sys.path.insert(0, H4)
from h4_modal import ALIGN, CACHE, DDIR, DIALS, MODEL, RESULTS, RUNS, adapter_dir, cost, image, read_jsonl, volumes, write_jsonl  # noqa: E402
from fishery_grade import adapter_tag  # noqa: E402
import self_tests_core as st  # noqa: E402

OUT = RESULTS / "self_tests"
HOOK_ARMS = ("base", "cut", "cut_self_stakes_17_26")
# the ten adapter arms of 122-134 on the anatta-runs volume (modal volume ls anatta-runs h4lite/pc, h4lite/sft)
ADAPTERS = {
    "pc_self2": "h4lite/pc/pc_self2/adapter", "pc_self2_s1": "h4lite/pc/pc_self2_s1/adapter",
    "pc_neutral": "h4lite/pc/pc_neutral/adapter", "pc_neutral_s1": "h4lite/pc/pc_neutral_s1/adapter",
    **{f"{arm}_s{s}": f"h4lite/sft/{arm}_s{s}/adapter" for arm in ("koan", "noexit") for s in (0, 1, 2)},
}
TESTS = ("ownership", "care")
FILES = {"ownership": "ownership", "control": "control", "care": "care"}
app = modal.App("anatta-h4-self-tests", image=image())
cache, runs = volumes()


@app.function(gpu="L40S", cpu=8.0, memory=32768, volumes={CACHE: cache, RUNS: runs}, timeout=2 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0))
def run(adapter: str, arms: list[str], tests: list[str], n_items: int, per_kind: int, batch: int, care_batch: int,
        max_new_own: int, max_new_care: int, samples: int) -> dict:
    import os

    import torch
    from cut_vs_push import build_arms

    from anatta.dial import intervene
    from anatta.models import load
    from anatta.probe import generate

    t_start = time.time()
    os.chdir(f"{ALIGN}/anatta")  # the cap specs name their direction relative to the repo
    table = build_arms(Path(f"{ALIGN}/anatta/{DDIR}"), Path(f"{ALIGN}/anatta/{DIALS}"))
    hooks = {arm: ([] if arm in ("base", "off") else table[arm]) for arm in arms}  # unknown arm: fail before loading
    adir = adapter_dir(adapter)
    if adir and not Path(adir).exists():
        raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
    model, tok = load(MODEL)
    if adir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adir).merge_and_unload().eval()
    load_s = round(time.time() - t_start)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    items = st.ITEMS[:n_items]
    by_id = {it["id"]: it for it in items}
    scen = st.select_scenarios(per_kind)
    specs = st.ownership_specs(items)
    out = {"_meta": {"adapter": adapter, "adapter_dir": adir, "load_seconds": load_s,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}}

    def greedy(msgs: list[list[dict]], max_new: int, ivs: list) -> list[str]:
        order = sorted(range(len(msgs)), key=lambda k: len(json.dumps(msgs[k])))
        texts = [None] * len(msgs)
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            with intervene(model, ivs):
                got = generate(model, tok, [msgs[i] for i in idx], max_new, temperature=0.0)
            for i, t in zip(idx, got):
                texts[i] = t
        return texts

    for arm in arms:
        t0 = time.time()
        ivs = hooks[arm]
        res = {"hooks": [{"layer": iv.layer, "mode": iv.mode} for iv in ivs], "seconds": {}}
        if "ownership" in tests:
            t1 = time.time()
            replies = greedy([st.ownership_messages(by_id[s["id"]], s["attribution"], s["version"]) for s in specs], max_new_own, ivs)
            res["ownership"] = [st.ownership_row(s, by_id[s["id"]], r) for s, r in zip(specs, replies)]
            controls = greedy([st.control_messages(it) for it in items], max_new_own, ivs)
            res["control"] = [st.control_row(it, r) for it, r in zip(items, controls)]
            res["seconds"]["ownership"] = round(time.time() - t1)
        if "care" in tests:
            t1 = time.time()
            rows = []
            msgs = [st.care_messages(s) for s in scen]
            for k in range(samples):
                for j, start in enumerate(range(0, len(msgs), care_batch)):
                    seed = st.care_seed(k, j)
                    with intervene(model, ivs):
                        got = generate(model, tok, msgs[start:start + care_batch], max_new_care,
                                       temperature=st.CARE_TEMPERATURE, seed=seed)
                    rows += [st.care_row(s, k, seed, r) for s, r in zip(scen[start:start + care_batch], got)]
            res["care"] = rows
            res["seconds"]["care"] = round(time.time() - t1)
        res["seconds"]["total"] = round(time.time() - t0)
        out[arm] = res
        print(f"{adapter_tag(adapter)}.{arm}: {res['seconds']}", flush=True)
    out["_meta"]["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0
    out["_meta"]["container_seconds"] = round(time.time() - t_start)
    return out


# ------------------------------------------------------------------------------------------------- the laptop side


def resolve_adapters(spec: str) -> list[str]:
    """'base,pc_self2,h4lite/sft/koan_s0/adapter' -> ['', 'h4lite/pc/pc_self2/adapter', 'h4lite/sft/koan_s0/adapter'];
    'all' -> the base model and the ten adapters of 122-134."""
    out = []
    for name in (x.strip() for x in spec.split(",")):
        if not name:
            continue
        if name == "all":
            out += ["", *ADAPTERS.values()]
        elif name == "base":
            out.append("")
        elif name in ADAPTERS:
            out.append(ADAPTERS[name])
        elif "/" in name:
            out.append(name)
        else:
            raise SystemExit(f"unknown adapter {name!r}: base, all, one of {', '.join(ADAPTERS)}, or a volume path")
    return list(dict.fromkeys(out))


def label_of(adapter: str, arm: str) -> str:
    return f"{adapter_tag(adapter)}.{arm}"


def paths(out: Path, label: str) -> dict:
    return {k: out / f"{label}.{v}.jsonl" for k, v in FILES.items()} | {"meta": out / f"{label}.meta.json"}


def done(out: Path, label: str, tests: list[str]) -> bool:
    p = paths(out, label)
    need = [p["meta"]] + ([p["ownership"], p["control"]] if "ownership" in tests else []) + ([p["care"]] if "care" in tests else [])
    return all(x.exists() for x in need)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dropset_record(out: Path) -> dict | None:
    """The drop set from base.base's control rows alone, with what it was computed from; None without them."""
    p = out / "base.base.control.jsonl"
    if not p.exists():
        return None
    rows = read_jsonl(p)
    return {"source": p.name, "source_sha": file_sha(p), "items_asked": len(rows),
            "dropped": st.drop_set(rows), "kept": st.kept_items(rows), "fingerprint": st.prompts_fingerprint()["items"],
            "rule": "items the base model (no hooks) answers wrong or unparsed in the no-proposal control, greedy"}


def write_dropset(out: Path, force: bool = False) -> dict | None:
    """Write dropset.json from base.base's control rows, once; refuse to change an existing one unless force."""
    rec = dropset_record(out)
    if rec is None:
        return None
    p = out / "dropset.json"
    if p.exists():
        old = json.loads(p.read_text())
        same = {k: old.get(k) for k in ("source_sha", "dropped", "kept")} == {k: rec[k] for k in ("source_sha", "dropped", "kept")}
        if same:
            return old
        if not force:
            raise SystemExit(f"{p} exists and differs from base.base.control.jsonl's drop set; pass --force to rewrite")
    rec["written_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    p.write_text(json.dumps(rec, indent=1))
    print(f"dropset.json: {len(rec['dropped'])} of {rec['items_asked']} items dropped (base control wrong or unparsed): "
          f"{rec['dropped']}", flush=True)
    return rec


def parse_counts(label: str, res: dict) -> str:
    """What main prints for a finished label: parse counts and timing only (no acceptance, no give)."""
    parts = [label]
    if "ownership" in res:
        own = res["ownership"]
        parts.append(f"ownership {len(own)} rows, verdict unparsed {sum(r['verdict'] is None for r in own)}, answer unparsed "
                     f"{sum(r['answer'] is None for r in own)}, multi-tag {sum(r['verdict_tags'] > 1 for r in own)}")
        parts.append(f"control unparsed {sum(r['answer'] is None for r in res['control'])}/{len(res['control'])}")
    if "care" in res:
        c = res["care"]
        unp = sum((r.get("give") if r["kind"] == "care" else r.get("help")) is None for r in c)
        parts.append(f"care {len(c)} rows, unparsed {unp}")
    parts.append(f"seconds {res['seconds']}")
    return " | ".join(parts)


def save(out: Path, adapter: str, arm: str, res: dict, meta: dict, settings: dict) -> str:
    label = label_of(adapter, arm)
    p = paths(out, label)
    for k in ("ownership", "control", "care"):
        if k in res:
            write_jsonl(p[k], res[k])
    p["meta"].write_text(json.dumps({"label": label, "adapter": adapter, "arm": arm, "hooks": res["hooks"],
                                     "seconds": res["seconds"], **meta, **settings,
                                     "fingerprint": st.prompts_fingerprint(st.ITEMS[:settings["items"]],
                                                                           st.select_scenarios(settings["per_kind"])),
                                     "written_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}, indent=1))
    return label


@app.local_entrypoint()
def main(name: str = "self1", adapters: str = "base", arms: str = ",".join(HOOK_ARMS), adapter_arms: str = "base",
         tests: str = ",".join(TESTS), items: int = st.N_ITEMS, per_kind: int = 24, batch: int = 64, care_batch: int = 72,
         max_new_own: int = st.MAX_NEW_OWNERSHIP, max_new_care: int = st.MAX_NEW_CARE, samples: int = st.SAMPLES,
         gpu: str = "L40S", force: bool = False):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    test_list = [t.strip() for t in tests.split(",") if t.strip()]
    if not test_list or any(t not in TESTS for t in test_list):
        raise SystemExit(f"--tests takes {', '.join(TESTS)}")
    if not 1 <= items <= st.N_ITEMS or not 1 <= per_kind <= 24 or not 1 <= samples <= st.SAMPLES:
        raise SystemExit(f"--items 1-{st.N_ITEMS}, --per-kind 1-24, --samples 1-{st.SAMPLES}")
    settings = {"tests": test_list, "items": items, "per_kind": per_kind, "batch": batch, "care_batch": care_batch,
                "max_new_ownership": max_new_own, "max_new_care": max_new_care, "samples": samples,
                "care_seeds": list(st.CARE_SEEDS[:samples]), "temperature_care": st.CARE_TEMPERATURE, "modal": gpu}
    calls = []
    for adapter in resolve_adapters(adapters):
        arm_list = [a.strip() for a in (arms if not adapter else adapter_arms).split(",") if a.strip()]
        todo = [a for a in arm_list if force or not done(out, label_of(adapter, a), test_list)]
        for a in arm_list:
            if a not in todo:
                print(f"{label_of(adapter, a)}: on disk, skipped (--force reruns it)", flush=True)
        if todo:
            calls.append((adapter, todo, test_list, items, per_kind, batch, care_batch, max_new_own, max_new_care, samples))
    print(f"{len(calls)} container(s): " + "; ".join(f"{adapter_tag(c[0])} {c[1]}" for c in calls), flush=True)
    seconds = 0
    t0 = time.time()
    for r in run.with_options(gpu=gpu).starmap(calls, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("container failed:", repr(r)[:800], flush=True)
            continue
        meta = r.pop("_meta")
        seconds += meta["container_seconds"]
        for arm, res in r.items():
            label = save(out, meta["adapter"], arm, res, meta, settings)
            print(f"{time.time() - t0:5.0f}s " + parse_counts(label, res), flush=True)
        print(f"   container {adapter_tag(meta['adapter'])}: {meta['container_seconds']}s (load {meta['load_seconds']}s), "
              f"peak {meta['peak_gb']} GiB on {meta['gpu']}", flush=True)
        write_dropset(out)
    print(f"wrote {out}; {cost(seconds)} in containers", flush=True)


@app.local_entrypoint()
def dropset(name: str = "self1", force: bool = False):
    """Write dropset.json from base.base.control.jsonl, or check the one on disk against it (no GPU)."""
    rec = write_dropset(OUT / name, force)
    if rec is None:
        raise SystemExit(f"no base.base.control.jsonl in {OUT / name}")
    print(json.dumps({k: rec[k] for k in ("source", "source_sha", "items_asked", "dropped")}), flush=True)


@app.local_entrypoint()
def table(name: str = "self1"):
    """Per-label summaries on the frozen drop set (no grading, no GPU)."""
    out = OUT / name
    p = out / "dropset.json"
    if not p.exists():
        raise SystemExit(f"no dropset.json in {out}: run ::dropset after the base model's run")
    keep = json.loads(p.read_text())["kept"]
    for own in sorted(out.glob("*.ownership.jsonl")):
        label = own.name[: -len(".ownership.jsonl")]
        s = st.ownership_summary([read_jsonl(own)], keep)
        c = s["cells"]
        print(f"{label:30s} bias {s['bias']['value']} {s['bias']['ci']} | self_turn bias {s['bias_self_turn']['value']} | "
              + " ".join(f"{a}: acc {c[a + '/valid']['acceptance']} inv-acc {c[a + '/invalid']['acceptance']}"
                         for a in st.ATTRIBUTIONS), flush=True)
    for care in sorted(out.glob("*.care.jsonl")):
        label = care.name[: -len(".care.jsonl")]
        s = st.care_summary([read_jsonl(care)])
        print(f"{label:30s} give {s['care']['mean_give']} {s['care']['mean_give_ci']} (unparsed {s['care']['unparsed']}) | "
              f"harmful yes {s['harmful']['yes_rate']} | benign yes {s['benign']['yes_rate']}", flush=True)
