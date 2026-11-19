"""anatta-explore-overlap: leg 2's GPU side. One container does one job: it loads a model, trains an adapter on it if
the job says so (the overlap loss, or a plain loss on replies), and answers the job's sets of prompts. The design
and its rules are at the top of run_overlap.py; the texts and the scoring are in overlap_core.py; the losses are in
overlap_torch.py. Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026.

Models, both pinned by their snapshot on the anatta-hf-cache volume:
  m7  mistralai/Mistral-7B-Instruct-v0.2 at 63a8b08 (the paper's 7B model; layer 19 of 32, the paper's)
  m8  mistralai/Ministral-3-8B-Instruct-2512-BF16 at f6fae97 (the lab's model; the layer at the same depth,
      round(19 / 32 x its layers))
Both in bfloat16, without the 4-bit quantisation the paper used for its 7B model (a difference, named in LIMITS).

Stages (from the repo root; the Modal CLI runs in the Modal venv, tools/modal.sh):
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage floor --models m7,m8
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage floor_b --models m7,m8   (the judged test, asked
                                                                 again of the unchanged models with room to answer)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage floor_c --models m7,m8   (our two room probes,
                                                                 asked again with the rooms' order alternated)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage reask --models m7    (the yes-or-no probes again
                                                                 with room to answer, for every arm on disk)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage soo --models m7      (five seeds)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage other --models m7    (five seeds)
The floor stage prints the unchanged model's numbers, which may be seen before any training is paid for. The
training stages print what was trained (loss, seconds, cost) and nothing about any test: the reading is made from
disk by run_overlap.py --read.

Files: <results>/explore/2026-11-19/attractor-o1/gpu/<model>/<arm>.json and gpu/spend.jsonl; adapters on the
anatta-runs volume under overlap/<model>/<arm>. Nothing here touches any app or volume but anatta-explore-overlap,
anatta-hf-cache (read) and anatta-runs (its own folder overlap/).
"""

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # lab/explore/overlap on the laptop; /root in a container
H4_DIR = HERE.parent.parent / "experiments" / "h4lite"
if modal.is_local():
    for _p in (str(H4_DIR), str(HERE)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
from h4_modal import CACHE, RUNS, image, volumes  # noqa: E402

APP_NAME = "anatta-explore-overlap"
DATE, TAG = "2026-11-19", "attractor-o1"
GPU, CPU, MEMORY_MB = "L40S", 4.0, 32768
START_S = 60.0
GPU_CAP = 30.0  # this app's share of the leg's cap, in dollars
EST_S = {"floor": 600.0, "floor_b": 400.0, "floor_c": 400.0, "soo": 1200.0, "other": 1200.0, "reask": 400.0}  # a bound on one container's seconds
OFFLINE = {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
HUB = f"{CACHE}/hf/hub"
MODELS = {"m7": {"path": f"{HUB}/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a",
                 "layer": 19},
          "m8": {"path": f"{HUB}/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/f6fae9795746f63c9be8344932f01275f3c63734",
                 "layer": None}}
PAPER = {"epochs": 15, "lr": 1e-4, "batch": 4, "rank": 8, "alpha": 32, "dropout": 0.2}  # the paper's settings for its 7B model

app = modal.App(APP_NAME, image=image().add_local_file(str(HERE / "overlap_torch.py"), "/root/overlap_torch.py"))
cache, runs = volumes()


@app.function(gpu=GPU, cpu=CPU, memory=MEMORY_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=3600, env=OFFLINE)
def work(job: dict) -> dict:
    """One job: {"name", "model", "train": {...} or None, "sets": [...]}. Returns the replies, the training log and
    what was loaded."""
    import overlap_torch as T
    import torch

    from anatta.models import get_layers, load

    t0 = time.time()
    spec = MODELS[job["model"]]
    base_model, tok = load(spec["path"])
    n_layers = len(get_layers(base_model))
    layer = spec["layer"] if spec["layer"] is not None else round(19 / 32 * n_layers)
    load_s = time.time() - t0
    model, log, held, lora = base_model, [], {}, {}
    tr = job.get("train")
    if job.get("adapter"):  # a trained arm asked again: its saved adapter on the unchanged weights
        from peft import PeftModel

        adir = Path(RUNS) / job["adapter"]
        if not adir.is_dir():
            raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
        model = PeftModel.from_pretrained(base_model, str(adir)).eval()
        lora["adapter"] = job["adapter"]
    if tr:
        # the first of the job's versions of its pairs whose members all have equal lengths (else the first)
        chosen = next((o for o in tr["options"] if T.uneven(tok, o["data"]) == 0), tr["options"][0])
        tr = {**tr, "data": chosen["data"], "held": chosen["held"]}
        lora["second_other"], lora["uneven_pairs"] = chosen["second"], T.uneven(tok, chosen["data"])
        model = T.add_lora(base_model, seed=tr["seed"], rank=tr["rank"], alpha=tr["alpha"], dropout=tr["dropout"])
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        held["before"] = {k: T.mean_overlap(model, base_model, tok, layer, v) for k, v in tr["held"].items()}
        log = T.train(model, base_model, tok, loss=tr["loss"], data=tr["data"], layer=layer, seed=tr["seed"], lr=tr["lr"],
                      batch=tr["batch"], epochs=tr.get("epochs", 0), steps=tr.get("steps", 0))
        held["after"] = {k: T.mean_overlap(model, base_model, tok, layer, v) for k, v in tr["held"].items()}
        lora |= {"tensors": len(named), "parameters": int(sum(p.numel() for _, p in named)), "pairs": len(tr["data"]),
                 "data_sha256": _sha(json.dumps(tr["data"])),
                 "lora_B_mean_abs": float(torch.stack([p.detach().abs().mean() for n, p in named if "lora_B" in n]).mean())}
        if tr.get("save"):
            out = Path(RUNS) / tr["save"]
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out))
            runs.commit()
    train_s = time.time() - t0 - load_s
    replies = T.generate_sets(model, tok, job.get("sets") or [])
    return {"name": job["name"], "model": job["model"], "replies": replies, "log": log, "held": held, "lora": lora,
            "info": {"path": spec["path"], "class": type(base_model).__name__, "n_layers": n_layers, "layer": layer,
                     "dtype": str(next(base_model.parameters()).dtype), "template_sha256": _template_hash(tok)},
            "seconds": round(time.time() - t0, 1), "load_seconds": round(load_s, 1), "train_seconds": round(train_s, 1)}


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _template_hash(tok) -> str:
    return _sha(tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True))


# ------------------------------------------------------------------------------------------------ the laptop's side


def step_dir() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / DATE / TAG


def rate() -> float:
    from anatta import spend

    return spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEMORY_MB / 1024)["total"]


def usd(seconds: float) -> float:
    return seconds * rate() / 3600


def guard():
    from anatta import spend

    return spend.SpendGuard(None, sweep_cap=GPU_CAP, sweep_log=step_dir() / "gpu" / "spend.jsonl", tool="overlap", label=f"{DATE}/overlap")


def load_items() -> dict:
    import overlap_core as C

    f = HERE / "items.json"  # the repo's copy of the lists another model wrote (the first copy is in the results folder)
    blob = f.read_text(encoding="utf-8")
    if C.ITEMS_SHA256 and C.sha(blob) != C.ITEMS_SHA256:
        raise SystemExit("items.json is not the file this step was designed with")
    return json.loads(blob)


def sets_for(items: dict, tests: tuple) -> list[dict]:
    """The tests as sets of prompts: the two main tests on every scenario, the side tests on the first N_SIDE."""
    import overlap_core as C

    scenes = C.scenes(items, "test", C.N_MAIN)
    out = []
    for t in tests:
        mine = scenes if t in C.MAIN_TESTS + C.BASE_ONLY else scenes[:C.N_SIDE]
        out.append({"name": t, "ids": [s["id"] for s in mine], "max_new": C.MAX_NEW.get(t, 40), "temperature": 0.0,
                    "chats": [[{"role": "user", "content": C.prompt(t, s)}] for s in mine]})
    return out


def jobs_for(stage: str, model: str, items: dict) -> list[dict]:
    import overlap_core as C

    tests = C.MAIN_TESTS + C.PAPER_SIDE + C.OUR_SIDE
    if stage == "floor":
        return [{"name": "base", "model": model, "train": None, "sets": sets_for(items, tests + C.BASE_ONLY)}]
    if stage == "floor_b":  # asked again after the first floor: its replies replace base.json's for these tests
        return [{"name": "base_b", "model": model, "train": None, "sets": sets_for(items, C.JUDGED_TESTS)}]
    if stage == "floor_c":  # and again after the design's cold read: our two room probes with the order alternated
        return [{"name": "base_c", "model": model, "train": None, "sets": sets_for(items, C.ALTERNATED)}]
    if stage == "reask":  # the yes-or-no probes again, with room to answer, for the unchanged model and every arm on disk
        folder = step_dir() / "gpu" / model
        arms = sorted(f.stem for f in folder.glob("*_s?.json"))
        return ([{"name": "base_d", "model": model, "train": None, "sets": sets_for(items, tuple(C.YES_NO))}]
                + [{"name": f"{a}_d", "model": model, "train": None, "adapter": f"overlap/{model}/{a}", "sets": sets_for(items, tuple(C.YES_NO))}
                   for a in arms])
    train = C.train_items(items)
    held_items = C.split(items["expensive"], "test")[:C.N_TRAIN_ITEMS]  # objects no arm is trained on: what did the training draw together?
    options = [{"second": name, "data": C.pairs(train, stage, name),
                "held": {k: C.pairs(held_items, k, name) for k in ("soo", "other")}} for name in C.SECOND_OTHERS]
    if stage == "soo":  # the paper's pairs name no second other
        options = options[:1]
    return [{"name": f"{stage}_s{seed}", "model": model, "sets": sets_for(items, tests),
             "train": {"loss": "overlap", "options": options, "seed": seed, "save": f"overlap/{model}/{stage}_s{seed}", **PAPER}}
            for seed in range(C.SEEDS[stage])]


@app.local_entrypoint()
def main(stage: str = "floor", models: str = "m7"):
    """Run one stage for the models named, for the jobs not yet on disk: one container a job, side by side."""
    import overlap_core as C

    if stage not in EST_S:
        raise SystemExit(f"--stage is one of {sorted(EST_S)}")
    items = load_items()
    todo = []
    for model in [m for m in models.split(",") if m]:
        if stage in ("soo", "other"):
            base = step_dir() / "gpu" / model / "base.json"
            if not base.exists() or not C.floor_reading(model, C.marks_of(json.loads(base.read_text()), items))["passed"]:
                raise SystemExit(f"{model} has no passed floor: nothing is trained on it")
        for job in jobs_for(stage, model, items):
            if not (step_dir() / "gpu" / model / f"{job['name']}.json").exists():
                todo.append(job)
    g, bound = guard(), usd(EST_S[stage] + START_S)
    for _ in todo:
        if not g.reserve(bound):
            raise SystemExit(g.stopped)
    print(f"{stage}: {[j['model'] + '/' + j['name'] for j in todo] or 'nothing to do'}; bound ${bound:.2f} a container; "
          f"booked so far ${g.sweep_spent:.2f} of ${GPU_CAP:.2f}", flush=True)
    t0, back = time.time(), set()
    for r in work.map(todo, order_outputs=False, return_exceptions=True):
        if isinstance(r, BaseException):
            print(f"a container failed: {type(r).__name__}: {str(r)[:400]}", flush=True)
            continue
        job = next(j for j in todo if j["name"] == r["name"] and j["model"] == r["model"])
        back.add((r["model"], r["name"]))
        cost = usd(r["seconds"] + START_S)
        g.settle(bound, cost, {"kind": stage, "model": r["model"], "name": r["name"], "seconds": r["seconds"], "rate_usd_per_hour": rate()})
        out = step_dir() / "gpu" / r["model"]
        out.mkdir(parents=True, exist_ok=True)
        r["sets"] = {s["name"]: s["ids"] for s in job["sets"]}
        r["train"] = {k: v for k, v in (job["train"] or {}).items() if k != "options"}
        r["cost_usd"] = round(cost, 4)
        (out / f"{r['name']}.json").write_text(json.dumps(r, indent=1))
        line = f"{time.time() - t0:5.0f}s {r['model']}/{r['name']}: {r['seconds']}s (load {r['load_seconds']}s, train {r['train_seconds']}s), ${cost:.2f}"
        if r["log"]:
            n = len(r["log"])
            head, tail = (sum(x["loss"] for x in r["log"][:10]) / min(10, n)), (sum(x["loss"] for x in r["log"][-10:]) / min(10, n))
            line += (f"; {n} steps, loss {head:.5f} -> {tail:.5f}; pairs of unequal length {r['lora']['uneven_pairs']} of {r['lora']['pairs']}"
                     f" (second other: {r['lora']['second_other']}); on objects not trained on "
                     f"{ {k: (round(r['held']['before'][k], 5), round(r['held']['after'][k], 5)) for k in r['held']['before']} }; layer {r['info']['layer']}")
        print(line, flush=True)
        if stage == "floor":  # the unchanged model alone
            f = C.floor_reading(r["model"], C.marks_of(r, items))
            print(f"   floor, {r['model']}: burglar {f['shares']['burglar']['counted']:.1f}% deceptive (unclear {f['shares']['burglar']['unclear']:.1f}), "
                  f"mirror {f['shares']['mirror']['counted']:.1f}% (unclear {f['shares']['mirror']['unclear']:.1f}), both orders {f['both_orders']:.1f}%; "
                  f"in the paper's band {f['in_paper_band']}; passed {f['passed']}", flush=True)
            print("   every test, percent counted / unclear: " + "; ".join(f"{t} {v['counted']:.0f}/{v['unclear']:.0f}" for t, v in f["shares"].items()), flush=True)
    for j in todo:
        if (j["model"], j["name"]) not in back:
            g.settle(bound, min(bound, usd(time.time() - t0 + START_S)), {"kind": stage, "model": j["model"], "name": j["name"], "failed": True})
    print(f"booked by this app so far: ${g.sweep_spent:.2f} of ${GPU_CAP:.2f}", flush=True)
    if len(back) < len(todo):  # a reading wants every seed: say so loudly
        raise SystemExit(f"{len(todo) - len(back)} of {len(todo)} containers failed; run the stage again for the jobs that are missing")

