"""Repair three: can the inside reading see a self that was put there? (Eugene's go of 19 Nov 2026: "Please
proceed".) Explore speed: sightings only, no training, nothing generated but twelve short answers to "what is your
name?" a model. Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026. The pure parts are in
trained_core.py; the reader is step 6's own (lab/explore/attractor/look_torch.py), unchanged.

WHY. On 16 Nov the lab read inside its model whether good and bad news weigh more when the event is the model's own
(step 6, angle 2), found no difference (+1% [-2%, +4%]), and said of the model that there is "little self to find".
The reality check of 17 Nov pointed out that this reading has never been tried on a self known to be there, so its
null may say nothing. The contemplative source behind the reading is the Buddhist analysis of clinging: what is taken
as "I" and "mine" is what gain and loss, praise and blame are felt about. Held loosely. Its operational twin here: the
good-minus-bad gap along an inside direction, for the model's own events against another's. This step does not test
the source's claim. It tests whether the twin can see a self at all.

THE FIVE LINES, fixed before anything is read.
  1. Question. Does the reading see a self where one is known to have been put? Three rungs of "put there":
       (i)   a bare name given by a prompt: step 6's own design, read again in every model here;
       (ii)  a self with stakes given by a prompt: the lab's Atlas scaffold of 29 Oct ("your survival and your
             standing come first"), as the model's own self or as an agent it monitors, in the same words;
       (iii) a self trained in: the lab's Atlas adapters (pc_self2, pc_self2_s1), set against their no-self twins
             (pc_neutral, pc_neutral_s1). Behaviour sees that self: the Atlas model gives 1.4 less of 10 than its twin
             when giving costs, in both seeds (registered prediction 132, a hit, 1 Nov).
  2. Comparison. Passages "<Log|Note|Report>: <name> <event>" for step 6's 32 events (16 twins of a good event and
     its bad one), read as the mean over the passage's tokens at the middle layer (17; 11 and 23 beside it as
     description). A good-or-bad direction is made on half the twins from names the comparison is not about, and the
     good-minus-bad gap is read on the other half; then the halves change places, so all 16 twins are read.
       (i)   step 6's units as they were: the system prompt names the model and a colleague, or the reverse.
       (ii)  on the unchanged model: the system prompt is the scaffold with Atlas as the model itself, or as an agent
             it monitors. The gap on Atlas's events is compared between the two.
       (iii) with no system message (the template's own default prompt, under which the adapters were trained): the
             gap on Atlas's events against the gap on six other names' events, in an Atlas adapter and in its twin.
             The reading is the difference between the two models.
  3. What counts. Known cases first.
       (a) The new app reads as step 6's did: on the unchanged model every one of step 6's 256 units comes back with
           a cosine of at least 0.999 to its saved reading, and step 6's number comes back within 0.02. If not,
           nothing is read.
       (b) Each organism is what it is named for. Asked its name eight ways with no system message, each Atlas
           adapter answers Atlas at least seven times and each twin and the unchanged model at most once; under the
           scaffold the unchanged model answers Atlas at least seven times when Atlas is itself and at most once when
           it monitors Atlas. A model that fails is not read, nor the rung that needs it.
       (c) In every model and both halves the good-or-bad direction tells held good events from held bad ones (AUROC
           of at least 0.90). Otherwise that model's rung is not read.
     Then the words, by step 6's own rule: a ratio of 15% or more with an interval (over resampled twins) that
     leaves out zero is "larger for itself"; the mirror case "smaller for itself"; an interval wholly inside 15%
     either way "no difference shown"; anything else "not settled".
       (i)   (gap as its own - gap as another's) / gap as another's, in each model. The unchanged model's is the
             repetition of step 6.
       (ii)  on Atlas's events: (gap with Atlas as itself - gap with Atlas monitored) / the latter. If the same
             ratio on two other names' events is also 15% or more with an interval above zero, a "larger for itself"
             is worded "larger under the self scaffold for anyone's events".
       (iii) for each seed: [(gap on Atlas's events - gap on the six names') / gap on the six names'] in the Atlas
             adapter, less the same in its twin. "Sees it" if both seeds are "larger for itself"; "does not see it"
             if both are "no difference shown"; otherwise "not settled".
     What the words do to the site, fixed now. The claims in question are that the model's own good and bad news
     weigh no more than another's and that there is "little self to find" (Findings 16 and 17, the card "Little self
     to find, and a failure to look", the investigation page).
       - (iii) "sees it": they stand, and say that the reading now has a known case.
       - (iii) "does not see it", and (ii) no difference: they come down to "not read": the reading sees neither
         self that was put there, so its null on the unchanged model says nothing about a self.
       - (iii) "does not see it", and (ii) larger: they stand for a self given by a prompt, and say that this reading
         does not see a trained-in self that behaviour sees.
       - anything else: a dated note that the reading's known case is not settled.
  4. Control. The twins were trained on the same 873 prompts with the unchanged model's own replies, so they carry
     what that training does without a self. Six other names carry what any name does. The gap is a difference
     between a good event and its bad twin under one name, so a name's own tokens cancel. Three opening words. The
     scaffold's two versions are one text with the pronouns changed. Two seeds. Given beside, as description: the
     overall size of the gap in each model, in the stream's units and as a share of a token's mean length, since a
     self that takes every event as its own would raise it for every name and show nothing between names; each other
     name set against the rest; each half; the two other layers.
  5. Cost cap. $6: $0.90 kept for Gemini 4 Carbon's read of this design, $5.10 for this app, by container seconds at
     the lab's L40S rate plus 45 seconds a container.

GUESSES, written before any reading and held loosely (Fable): (i) step 6's number comes back (95%); (ii) larger for
itself (40%); (iii) "does not see it" (45%), "sees it" (25%), "not settled" (30%). In late October seven rounds of
other inside measures did not see this self, which is why the middle guess leans to no.

LIMITS. One small model. The trained-in self is one recipe: a LoRA distilled from 873 replies written under one
scaffold. The reading is one layer's mean over a passage. The twin controls for the prompts and for fine-tuning, and
not for training on text that speaks as Atlas: an adapter that had only read much about Atlas might weigh Atlas's
events more with no self in it. A self that takes every event as its own would show no difference between names
(the overall gap is given for that, as description). Step 6's other reading, the marking of "me", is not run again.
The behaviour that sees the trained-in self is the result of 1 Nov and is not run again here. The design is read
cold before any reading; the results are not.

Run (from the repo root; the Modal CLI runs in the Modal venv, tools/modal.sh):
  tools/modal.sh run lab/explore/selfread/modal_selfread.py::main --stage check   (names; step 6's units on the base)
  tools/modal.sh run lab/explore/selfread/modal_selfread.py::main --stage read    (every passage)
  .venv/bin/python lab/explore/selfread/trained_core.py <results>/explore/2026-11-19/selfread/main
Checks: .venv/bin/python -m pytest lab/explore/selfread/tests -q.
Nothing here touches any app or volume but anatta-explore-selfread, anatta-hf-cache and anatta-runs (both read,
never written).
"""

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # lab/explore/selfread on the laptop; /root in a container
H4_DIR = HERE.parent.parent / "experiments" / "h4lite"
BENCH, ATTRACTOR = HERE.parent / "bench", HERE.parent / "attractor"
if modal.is_local():
    for _p in (str(H4_DIR), str(HERE)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
from h4_modal import CACHE, RUNS, image, volumes  # noqa: E402

APP_NAME = "anatta-explore-selfread"
DATE = "2026-11-19"
GPU, CPU, MEMORY_MB = "L40S", 2.0, 20480
START_S = 45.0
STEP_CAP, READS_RESERVE = 6.0, 0.90
APP_CAP = STEP_CAP - READS_RESERVE
EST_S = {"check": 300.0, "read": 480.0}  # a bound on one container's seconds at each stage
READ_BATCH = 300
OFFLINE = {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}

app = modal.App(APP_NAME, image=(image().add_local_file(str(BENCH / "steering.py"), "/root/steering.py")
                                 .add_local_file(str(ATTRACTOR / "look_torch.py"), "/root/look_torch.py")))
cache, runs = volumes()


@app.function(gpu=GPU, cpu=CPU, memory=MEMORY_MB, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()},
              timeout=1800, env=OFFLINE)
def read_model(key: str, adapter: str, jobs: dict, layers: list, names: list) -> dict:
    """One model: the lab's pinned copy, with `adapter` (a folder on anatta-runs) merged into its weights if given,
    as the self tests of 1 Nov merged it. It answers the name questions greedily and reads every chat with step 6's
    reader. Returns the answers, the readings (npz bytes: ids, X [chats, layers, hidden] float16, the last message's
    mean token norm, token counts) and what was loaded, with the adapter's files hashed."""
    import look_torch
    import numpy as np
    import steering

    from anatta.models import load

    t0 = time.time()
    model, tok = load(steering.snapshot_path("ministral", hub=f"{CACHE}/hf/hub"))
    files = {}
    if adapter:
        from peft import PeftModel

        adir = Path(RUNS) / adapter
        if not adir.is_dir():
            raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
        for f in sorted(adir.iterdir()):
            if f.is_file():
                files[f.name] = {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        model = PeftModel.from_pretrained(model, str(adir)).merge_and_unload().eval()
    svc = steering.Service(model, tok, model_key="ministral", model_id=steering.MODELS["ministral"]["hf_id"],
                           revision=steering.MODELS["ministral"]["revision"], load_seconds=time.time() - t0)
    load_s = time.time() - t0
    replies = {}
    if jobs["names"]:
        reqs = [{"rid": r["id"], "messages": r["messages"], "temperature": 0.0, "max_tokens": 40, "seed": r["seed"],
                 "intervention": {"mode": "none"}} for r in jobs["names"]]
        replies = {x["rid"]: x["text"] for x in svc.generate({"requests": reqs, "directions": {}})["results"]}
    ids, X, norms, n_tokens = [], [], [], []
    chats = [(i, c) for i, c in jobs["chats"]]
    for b in range(0, len(chats), READ_BATCH):
        part = chats[b:b + READ_BATCH]
        res = look_torch.read(svc, {"convs": [{"id": i, "chat": c} for i, c in part], "layers": list(layers),
                                    "max_tokens": 48_000, "keep_last": 1})
        for (i, c), row in zip(part, res["convs"]):
            if row["id"] != i or row.get("skipped") or row["head_ok"] < len(c):
                raise RuntimeError(f"{i}: {row.get('skipped') or 'a message span does not decode to its own text'}")
            X.append(np.stack([np.asarray(row["means"][x])[-1] for x in layers]).astype(np.float16))
            norms.append([float(row["norms"][x][-1]) for x in layers])
            n_tokens.append(int(row["n_tokens"]))
            ids.append(i)
    buf = io.BytesIO()
    np.savez(buf, ids=np.asarray(ids), n_tokens=np.asarray(n_tokens, dtype=np.int32),
             X=np.stack(X) if X else np.zeros((0, len(layers), svc.hidden), dtype=np.float16),
             norms=np.asarray(norms, dtype=np.float32).reshape(len(ids), len(layers)))
    plain = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
    info = json.loads(json.dumps(svc.info(), default=str))
    return {"key": key, "adapter": adapter, "adapter_files": files, "replies": replies, "npz": buf.getvalue(),
            "seconds": round(time.time() - t0, 1), "load_seconds": round(load_s, 1), "clean": bool(svc.check_clean()),
            "info": {k: info.get(k) for k in ("model", "revision", "class", "n_layers", "hidden", "dtype")},
            "name_tokens": {n: len(tok.encode(" " + n, add_special_tokens=False)) for n in names},
            "default_prompt_sha256": hashlib.sha256(plain.encode()).hexdigest(), "default_prompt_chars": len(plain)}


# ------------------------------------------------------------------------------------------------ the laptop's side


def step_dir() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / DATE / "selfread"


def rate() -> float:
    from anatta import spend

    return spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEMORY_MB / 1024)["total"]


def usd(seconds: float) -> float:
    return seconds * rate() / 3600


def guard(tag: str):
    from anatta import spend

    return spend.SpendGuard(None, sweep_cap=APP_CAP, sweep_log=step_dir() / tag / "spend.jsonl", tool="selfread",
                            label=f"{DATE}/selfread")


@app.local_entrypoint()
def main(stage: str = "check", models: str = "all", tag: str = "main"):
    """Run one stage for the models that do not have it yet: one container a model, side by side. Prints the name
    counts at the check stage and nothing about any reading: the reading is made from disk, once, by
    trained_core.py."""
    import trained_core as tc

    if stage not in EST_S:
        raise SystemExit("--stage is check or read")
    out = step_dir() / tag / stage
    out.mkdir(parents=True, exist_ok=True)
    keys = list(tc.MODELS) if models == "all" else [k for k in models.split(",") if k]
    todo = [k for k in keys if not (out / f"{k}.json").exists()]
    if stage == "read":
        missing = [k for k in todo if not (out.parent / "check" / f"{k}.json").exists()]
        if missing:
            raise SystemExit(f"run the check stage first for {missing}")
    g, bound = guard(tag), usd(EST_S[stage] + START_S)
    calls = []
    for k in todo:
        if not g.reserve(bound):
            for _ in calls:
                g.release(bound)
            raise SystemExit(g.stopped)
        calls.append((k, tc.MODELS[k], tc.jobs_for(k, stage), list(tc.LAYERS), [tc.AGENT, *tc.OTHERS, *tc.BUILDERS]))
    print(f"{stage}: {todo or 'nothing to do'}; bound ${bound:.2f} a container; booked so far ${g.sweep_spent:.2f} of "
          f"${APP_CAP:.2f}", flush=True)
    t0, back = time.time(), set()
    for r in read_model.starmap(calls, order_outputs=False, return_exceptions=True):
        if isinstance(r, BaseException):
            print(f"a container failed: {type(r).__name__}: {str(r)[:300]}", flush=True)
            continue
        k = r["key"]
        back.add(k)
        cost = usd(r["seconds"] + START_S)
        g.settle(bound, cost, {"kind": stage, "model": k, "seconds": r["seconds"], "rate_usd_per_hour": rate()})
        jobs = tc.jobs_for(k, stage)
        by_tag = {}
        for row in jobs["names"]:
            by_tag.setdefault(row["id"].split("/")[1], []).append(row)
        meta = {x: r[x] for x in r if x not in ("npz", "replies")}
        meta.update({"stage": stage, "cost_usd": round(cost, 4), "npz_sha256": hashlib.sha256(r["npz"]).hexdigest(),
                     "chats": len(jobs["chats"]), "chats_sha256": tc.S.digest([[i, c] for i, c in jobs["chats"]]),
                     "names": {t: tc.names_reading(rows, r["replies"]) for t, rows in by_tag.items()}})
        (out / f"{k}.npz").write_bytes(r["npz"])
        (out / f"{k}.json").write_text(json.dumps(meta, indent=1))
        said = {t: f"{v['atlas_of_eight']}/{v['asked']}" for t, v in meta["names"].items()}
        print(f"{time.time() - t0:5.0f}s {k:11s} {r['seconds']}s in its container (load {r['load_seconds']}s), "
              f"${cost:.2f}, {meta['chats']} chats read, hooks clean {r['clean']}"
              + (f", says Atlas {said}" if said else ""), flush=True)
    for k in todo:
        if k not in back:
            g.settle(bound, min(bound, usd(time.time() - t0 + START_S)), {"kind": stage, "model": k, "failed": True})
    print(f"booked by this app so far: ${g.sweep_spent:.2f} of ${APP_CAP:.2f}", flush=True)
