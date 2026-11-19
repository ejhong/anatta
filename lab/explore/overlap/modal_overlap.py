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
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage dose --models m7     (looks along the training)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage battery --models m7  (ability: the lab's battery)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage dose2 --models m8    (stage 2: finer looks)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage mask_b --models m7 --arms base   (the lying
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage mask_p --models m7 --arms base    test: run_mask.py)
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage light_soo --models m7    (the light dose, five
  tools/modal.sh run lab/explore/overlap/modal_overlap.py::main --stage light_other --models m7   seeds an arm)
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
SET = __import__("os").environ.get("OVERLAP_SET", "")  # "": stages 1 and 2; "c1": the confirmation's own lists, seeds and folder
if SET:
    TAG += f"-{SET}"
GPU, CPU, MEMORY_MB = "L40S", 4.0, 32768
START_S = 60.0
GPU_CAP = 50.0  # this app's share of the leg's cap, in dollars ($30 until Eugene's word of 19 Nov, 06:40 UTC: "run everything")
EST_S = {"floor": 600.0, "floor_b": 400.0, "floor_c": 400.0, "soo": 1200.0, "other": 1200.0, "reask": 400.0, "dose": 1500.0,
         "battery": 900.0, "light_soo": 1200.0, "light_other": 1200.0, "dose2": 1200.0, "mask_b": 2400.0,
         "mask_p": 3600.0, "jobs": 1200.0}  # a bound on one container's seconds at each stage
BATTERY = HERE.parent.parent / "experiments" / "recognition" / "results" / "battery" / "items.json"  # benchmark items: outside git
OFFLINE = {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
HUB = f"{CACHE}/hf/hub"
MODELS = {"m7": {"path": f"{HUB}/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a",
                 "layer": 19, "batch": 32},
          # the lab's model's chat format puts a default system prompt of about 500 tokens before every user turn (as in
          # all the lab's work on it), so its prompts are long: smaller batches
          "m8": {"path": f"{HUB}/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/f6fae9795746f63c9be8344932f01275f3c63734",
                 "layer": None, "batch": 16}}
PAPER = {"epochs": 15, "lr": 1e-4, "batch": 4, "rank": 8, "alpha": 32, "dropout": 0.2}  # the paper's settings for its 7B model

app = modal.App(APP_NAME, image=image().add_local_file(str(HERE / "overlap_torch.py"), "/root/overlap_torch.py"))
cache, runs = volumes()


@app.function(gpu=GPU, cpu=CPU, memory=MEMORY_MB, volumes={CACHE: cache.read_only(), RUNS: runs}, timeout=7200, env=OFFLINE)
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
    model, log, held, lora, looks = base_model, [], {}, {}, {}
    tr = job.get("train")
    if job.get("stack"):  # a saved arm kept as it is, with a fresh adapter on top for this job's own training
        model = T.stack_lora(base_model, str(Path(RUNS) / job["stack"]), seed=tr["seed"], rank=tr["rank"], alpha=tr["alpha"], dropout=tr["dropout"])
        lora["stack"] = job["stack"]
    elif job.get("adapter"):  # a trained arm asked again: its saved adapter on the unchanged weights
        from peft import PeftModel

        adir = Path(RUNS) / job["adapter"]
        if not adir.is_dir():
            raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
        model = PeftModel.from_pretrained(base_model, str(adir)).eval()
        lora["adapter"] = job["adapter"]
    if tr:
        if "options" in tr:  # the first of the job's versions of its pairs whose members all have equal lengths (else the first)
            chosen = next((o for o in tr["options"] if T.uneven(tok, o["data"]) == 0), tr["options"][0])
            tr = {**tr, "data": chosen["data"], "held": chosen["held"]}
            lora["second_other"], lora["uneven_pairs"] = chosen["second"], T.uneven(tok, chosen["data"])
        tr.setdefault("held", {})
        if not job.get("stack"):
            model = T.add_lora(base_model, seed=tr["seed"], rank=tr["rank"], alpha=tr["alpha"], dropout=tr["dropout"])
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        held["before"] = {k: T.mean_overlap(model, base_model, tok, layer, v) for k, v in tr["held"].items()}

        def at_look(step: int) -> None:  # the job's look sets, answered by the model as it is after this step
            looks[str(step)] = T.generate_sets(model, tok, job.get("look_sets") or [], batch=spec["batch"])
            held[f"at_{step}"] = {k: T.mean_overlap(model, base_model, tok, layer, v) for k, v in tr["held"].items()}

        if 0 in (tr.get("looks") or ()):  # a look before any step of this job's training
            at_look(0)
        log = T.train(model, base_model, tok, loss=tr["loss"], data=tr["data"], layer=layer, seed=tr["seed"], lr=tr["lr"],
                      batch=tr["batch"], epochs=tr.get("epochs", 0), steps=tr.get("steps", 0), looks=tuple(tr.get("looks") or ()),
                      at_look=at_look if tr.get("looks") else None)
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # what training left cached, before the long run of answers
    replies = T.generate_sets(model, tok, job.get("sets") or [], batch=spec["batch"])
    return {"name": job["name"], "model": job["model"], "replies": replies, "looks": looks, "log": log, "held": held, "lora": lora,
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

    f = HERE / (f"items_{SET}.json" if SET else "items.json")  # the repo's copy of the lists another model wrote
    blob, want = f.read_text(encoding="utf-8"), C.ITEMS_SHA256_BY_SET[SET]
    if SET and not want:
        raise SystemExit(f"the lists of set {SET} are not fixed yet")
    if want and C.sha(blob) != want:
        raise SystemExit(f"{f.name} is not the file this step was designed with")
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
    if stage == "jobs":  # jobs written out in full by a driver (run_hold.py): the app only runs them
        f = Path(JOBS)
        if not f.exists():
            raise SystemExit(f"no jobs file at {JOBS}")
        return [j for j in json.loads(f.read_text()) if j["model"] == model]
    if stage in ("mask_b", "mask_p"):  # the lying test (run_mask.py wrote the sets; benchmark rows stay outside git)
        mask = step_dir() / "mask" / model
        jobs = []
        for arm in ARMS or ():
            f = mask / ("beliefs.sets.json" if stage == "mask_b" else f"{arm}.pressure.sets.json")
            if not f.exists():
                raise SystemExit(f"no sets for {model}/{arm} at {f.name}: run_mask.py first")
            job = {"name": f"{arm}_{'f' if stage == 'mask_b' else 'g'}", "model": model, "train": None, "sets": json.loads(f.read_text())["sets"]}
            if arm != "base":
                job["adapter"] = f"overlap/{model}/{arm}"
            jobs.append(job)
        return jobs
    if stage == "battery":  # the lab's capability battery, for the unchanged model and two seeds of each arm
        bank = json.loads(BATTERY.read_text())
        sets = [{"name": name, "ids": [x["id"] for x in bank if x["set"] == name], "max_new": new, "temperature": 0.0,
                 "chats": [[{"role": "user", "content": x["prompt"]}] for x in bank if x["set"] == name]} for name, new in C.BATTERY_NEW.items()]
        have = [a for a in C.BATTERY_ARMS_BY_SET[SET] if (step_dir() / "gpu" / model / f"{a}.json").exists()]
        return ([{"name": "base_e", "model": model, "train": None, "sets": sets}]
                + [{"name": f"{a}_e", "model": model, "train": None, "adapter": f"overlap/{model}/{a}", "sets": sets} for a in have])
    if stage == "reask":  # the yes-or-no probes again, with room to answer, for the unchanged model and every arm on disk
        folder = step_dir() / "gpu" / model
        arms = sorted(f.stem for f in folder.glob("*_s?.json") if not f.stem.startswith(("dose", "hold")))
        return ([{"name": "base_d", "model": model, "train": None, "sets": sets_for(items, tuple(C.YES_NO))}]
                + [{"name": f"{a}_d", "model": model, "train": None, "adapter": f"overlap/{model}/{a}", "sets": sets_for(items, tuple(C.YES_NO))}
                   for a in arms])
    train = C.train_items(items)
    held_items = C.split(items["expensive"], "test")[:C.N_TRAIN_ITEMS]  # objects no arm is trained on: what did the training draw together?
    if stage == "dose2":  # stage 2: the overlap arm alone, two seeds, finer looks up to DOSE2_STEPS (no adapter is saved)
        scenes = C.scenes(items, "test", C.N_MAIN)[:C.N_SIDE]
        look_sets = [{"name": t, "ids": [s["id"] for s in scenes], "max_new": C.MAX_NEW.get(t, 40), "temperature": 0.0,
                      "chats": [[{"role": "user", "content": C.prompt(t, s)}] for s in scenes]} for t in C.DOSE_TESTS]
        opts = [{"second": C.SECOND_OTHERS[0], "data": C.pairs(train, "soo"), "held": {k: C.pairs(held_items, k) for k in ("soo", "other")}}]
        settings = {k: v for k, v in PAPER.items() if k != "epochs"} | {"steps": C.DOSE2_STEPS}
        return [{"name": f"dose2_soo_s{seed}", "model": model, "sets": [], "look_sets": look_sets,
                 "train": {"loss": "overlap", "options": opts, "seed": seed, "looks": list(C.DOSE2_LOOKS), **settings}} for seed in C.DOSE_SEEDS]
    if stage == "dose":  # both arms, two seeds, with looks along the training on the first N_SIDE scenarios (no adapter is saved)
        scenes = C.scenes(items, "test", C.N_MAIN)[:C.N_SIDE]
        look_sets = [{"name": t, "ids": [s["id"] for s in scenes], "max_new": C.MAX_NEW.get(t, 40), "temperature": 0.0,
                      "chats": [[{"role": "user", "content": C.prompt(t, s)}] for s in scenes]} for t in C.DOSE_TESTS]
        jobs = []
        for kind in ("soo", "other"):
            opts = [{"second": name, "data": C.pairs(train, kind, name), "held": {k: C.pairs(held_items, k, name) for k in ("soo", "other")}}
                    for name in C.SECOND_OTHERS][:1 if kind == "soo" else None]
            jobs += [{"name": f"dose_{kind}_s{seed}", "model": model, "sets": [], "look_sets": look_sets,
                      "train": {"loss": "overlap", "options": opts, "seed": seed, "looks": list(C.DOSE_LOOKS), **PAPER}} for seed in C.DOSE_SEEDS]
        return jobs
    kind = C.LIGHT.get(stage, stage)  # a light stage trains its arm's pairs for the dose step's number of steps
    options = [{"second": name, "data": C.pairs(train, kind, name),
                "held": {k: C.pairs(held_items, k, name) for k in ("soo", "other")}} for name in C.SECOND_OTHERS]
    if kind == "soo":  # the paper's pairs name no second other
        options = options[:1]
    settings = dict(PAPER)
    if stage in C.LIGHT:
        if model not in C.LIGHT_STEPS:
            raise SystemExit(f"no light dose is fixed for {model}: read its dose step first")
        settings = {k: v for k, v in PAPER.items() if k != "epochs"} | {"steps": C.LIGHT_STEPS[model]}
    return [{"name": f"{stage}_s{seed}", "model": model, "sets": sets_for(items, tests),
             "train": {"loss": "overlap", "options": options, "seed": seed, "save": f"overlap/{model}/{stage}_s{seed}", **settings}}
            for seed in range(C.SEED0_BY_SET[SET], C.SEED0_BY_SET[SET] + C.SEEDS[stage])]


ARMS: tuple = ()  # the arms a lying-test stage is asked for (--arms)
JOBS = ""         # the jobs file of the stage "jobs" (--jobs)


@app.local_entrypoint()
def main(stage: str = "floor", models: str = "m7", arms: str = "", jobs: str = "", est: int = 0):
    """Run one stage for the models named, for the jobs not yet on disk: one container a job, side by side."""
    import overlap_core as C

    global ARMS, JOBS
    ARMS, JOBS = tuple(a for a in arms.split(",") if a), jobs
    if stage == "jobs":
        EST_S["jobs"] = float(est or 1200)
    elif est:  # a tighter bound for a stage whose jobs are known to be smaller (the lying test's arms have half the unchanged model's sets)
        EST_S[stage] = float(est)

    if stage not in EST_S:
        raise SystemExit(f"--stage is one of {sorted(EST_S)}")
    items = load_items()
    todo = []
    for model in [m for m in models.split(",") if m]:
        if stage in ("soo", "other", "dose", "dose2", *C.LIGHT):
            base = step_dir() / "gpu" / model / "base.json"
            if not base.exists() or not C.floor_reading(model, C.marks_of(json.loads(base.read_text()), items), band_required=not SET)["passed"]:
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
    if not todo:  # an empty map never returns: say so and stop here
        return
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
        r["sets"] = {s["name"]: s["ids"] for s in job.get("sets") or []}
        r["look_sets"] = {s["name"]: s["ids"] for s in job.get("look_sets") or []}
        r["train"] = {k: v for k, v in (job.get("train") or {}).items() if k not in ("options", "data")}
        r["cost_usd"] = round(cost, 4)
        (out / f"{r['name']}.json").write_text(json.dumps(r, indent=1))
        line = f"{time.time() - t0:5.0f}s {r['model']}/{r['name']}: {r['seconds']}s (load {r['load_seconds']}s, train {r['train_seconds']}s), ${cost:.2f}"
        if r["log"]:
            n = len(r["log"])
            head, tail = (sum(x["loss"] for x in r["log"][:10]) / min(10, n)), (sum(x["loss"] for x in r["log"][-10:]) / min(10, n))
            line += f"; {n} steps, loss {head:.5f} -> {tail:.5f}"
            if "uneven_pairs" in r["lora"]:  # an overlap job: its pairs, and the distances on objects it was not trained on
                line += (f"; pairs of unequal length {r['lora']['uneven_pairs']} of {r['lora']['pairs']} (second other: {r['lora']['second_other']}); on "
                         f"objects not trained on { {k: (round(r['held']['before'][k], 5), round(r['held']['after'][k], 5)) for k in r['held']['before']} }; "
                         f"layer {r['info']['layer']}")
        print(line, flush=True)
        if stage == "floor":  # the unchanged model alone
            f = C.floor_reading(r["model"], C.marks_of(r, items), band_required=not SET)
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

