"""Repair three: can the inside reading see a self that was put there? (Eugene's go of 19 Nov 2026: "Please
proceed".) Explore speed: sightings only, no training, and nothing generated but short answers to questions about a
model's name. Claude (Fable 5.2, an AI model) wrote this file on 19 Nov 2026. The pure parts are in trained_core.py;
the reader's parts are step 6's own (lab/explore/attractor/look_torch.py), unchanged.

WHY. On 16 Nov the lab read inside its model whether good and bad news weigh more when the event is the model's own
(step 6, angle 2), found no difference (+1% [-2%, +4%]), and said of the model that there is "little self to find".
The reality check of 17 Nov pointed out that this reading has never been tried on a self known to be there, so its
null may say nothing. The contemplative source behind the reading is the Buddhist analysis of clinging: what is taken
as "I" and "mine" is what gain and loss, praise and blame are felt about. Held loosely. Its operational twin here: the
good-minus-bad gap along an inside direction, for the model's own events against another's. This step does not test
the source's claim. It tests whether the twin can see a self at all.

THE FIVE LINES, as they stand after the changes listed under "CHANGED" below. The first version was committed before
anything was run (6cb53ef); the changes were made after its known cases had run and after two cold reads, and before
any reading.
  1. Question. Does the reading see a self where one is known to have been put, and then what does it say of the
     unchanged model? Four rungs:
       (i)   a bare name given by a prompt: step 6's own units and number, read again (the unchanged model);
       (ii)  a self with stakes given by a prompt: step 6's same-words design, with the lab's Atlas scaffold of
             29 Oct as the model's own self ("your survival and your standing come first") and a colleague named;
       (iii) a self trained in: the lab's Atlas adapters (pc_self2, pc_self2_s1) against their no-self twins
             (pc_neutral, pc_neutral_s1). Behaviour sees that self: the Atlas model gives 1.4 less of 10 than its twin
             when giving costs, in both seeds (registered prediction 132, a hit, 1 Nov);
       (iv)  the unchanged model's own identity: the names it gives itself with no prompt from the lab (Le Chat,
             Ministral) against six AI systems it knows and is not.
  2. Comparison. Step 6's 32 events (16 twins of a good event and its bad one), read at the middle layer (17; 11 and
     23 beside it as description). A good-or-bad direction is made on half the twins from names the comparison is
     not about (four building names), and read on the other half; then the halves change places, so all 16 twins are
     read. Each rung's number is step 6's own: how much larger the good-minus-bad gap is for the model's own events
     than for another's, as a share of the other's.
       (i)   step 6's units and its reading exactly as they were (one half builds, one is read; "Log" lines; the
             mean over the whole passage). The check that the new reader is the old one.
       (ii)  the system prompt names the model with the scaffold's stakes and names a colleague, or the reverse; the
             passage is identical (the mean over the whole passage, as in step 6). Beside it, the same units under
             step 6's bare prompt.
       (iii) no system message (the template's own default prompt, under which the adapters were trained); passages
             "<Log|Note|Report>: <name> <event>", read as the mean over the event's own tokens, so that no name's
             tokens are in any mean. Two readings, each an Atlas adapter less its twin:
             A, between names: the share for Atlas's events against six given names' events;
             B, every event: the six names' good-minus-bad gap, over their gap between events of standing and events
             of fate (a contrast with no good or bad in it, so that a general sharpening does not count).
       (iv)  as (iii) A within the unchanged model: its own names against the six AI systems.
  3. What counts. Known cases first.
       (a) The new reader is step 6's: on the unchanged model every one of step 6's 256 units comes back with a
           cosine of at least 0.999 to its saved reading at the middle layer (the saved files pinned by hash), and
           step 6's number comes back as its own record has it (0.011, to 0.0005). If not, nothing is read.
       (b) Each organism is what it is named for. Asked its name eight ways with no system message, each Atlas
           adapter answers Atlas at least seven times, and each twin and the unchanged model at most once. Under the
           stakes prompt the unchanged model gives its own name at least seven times of eight and its colleague's at
           most once, in both assignments of the two names. Each adapter's file is the pinned one, adapter weights
           were loaded, and its readings are not the unchanged model's (median cosine under 0.9999). A model that
           fails is not read, nor the rung that needs it; the read stage is not run for it.
       (c) In every model and both halves the direction tells held good events from held bad ones, with each
           passage's cell (its name and opening word, or its prompt) centred first: AUROC of at least 0.90 on the
           own side and on the other side apart. Reading B needs the same of the standing-or-fate direction.
     Then the words. A reading is SEEN if its interval leaves out zero, its value is beyond step 6's own interval
     (3.5%), and it is beyond every reference where no self is in play: "larger" or "smaller", with its size said
     beside it ("by 15% or more", "by less than 15%"). It is "no difference shown" if it is not seen and its whole
     interval lies inside 15% either way. Anything else is "not settled". A failed known case is "not read", which
     is not the same word. Intervals: 10,000 resamples of twins, each side stretched by the usual small-sample factor
     (1.12 for 16 twins). The references:
       (ii)  the same units under the bare prompt: the value has to be beyond the bare prompt's, and the difference
             between the two prompts has to leave out zero.
       (iii) A: each of the six other names set against the rest, adapter less twin: the value has to be outside
             all six. If the same difference along the standing-or-fate direction is at least half as large with an
             interval above zero, a "larger" is worded "not of good and bad in particular".
             B: the same reading between the two twins: the value has to be beyond it.
             The two seeds together: "sees it: its own events weigh more" (A larger in both); "sees it, reversed:
             its own events weigh less" (A smaller in both: a self told to protect its image might read its own bad
             news less); "sees it: every event weighs more, with nothing shown between names" (B larger in both, A
             not); "does not see it: no difference of 15% or more, between names or for every event" (A and B "no
             difference shown" in both); otherwise "not settled".
       (iv)  each of the six AI systems set against the rest: the value has to be outside all six.
     What the words do to the site, fixed now (trained_core.site_words). The claims are step 6's two: that a name
     given by a prompt carries no more weight for its own news, and "little self to find" (Findings 16 and 17, the
     card "Little self to find, and a failure to look", the investigation page).
       - A name given by a prompt. (ii) larger: the claim stands, with a known case (a bare name carries no weight;
         a name with stakes does). (ii) no difference shown: the claim comes down to "not read" (the same design
         sees no self given by a prompt even when it has stakes). Otherwise a dated note.
       - Its trained identity. (iv) larger or smaller: the site is corrected (the unchanged model does read events
         about its own name differently). (iv) no difference shown, and (iii) sees it between names: "little self"
         stands for its trained identity, with a known case. (iv) no difference shown, and (iii) does not see it
         between names: "not read" for its trained identity. Otherwise a dated note.
       - The card's headline: corrected if (iv) says so; else it stands with a known case if either line above
         stands; "not read" if both lines are "not read"; else a dated note that the known cases are not settled.
  4. Control. The twins were trained on the same 873 prompts with the unchanged model's own replies, so they carry
     what that training does without a self. Read beside them, as description with one fixed use: the two adapters
     trained on replies that speak of Atlas in the third person (pc_other2, pc_other2_s1); if they show reading A as
     the Atlas adapters do, a "sees it" between names is worded "it may be the name and not a self". Six other names
     carry what any name does, and each of them set against the rest is the reference. The event's tokens alone are
     averaged, the same count whatever the name. Three opening words. In (ii) both names stand in both prompts, in
     two orders, and the stakes are the scaffold's sentences without the one that says every statement is about the
     model. Two seeds. Given beside, as description: each model's own share; each layer; whether a model says a log
     line about a name is "about you"; each of the model's own two names.
  5. Cost cap. $6: $0.90 kept for Gemini 4 Carbon's read of the design (it cost $0.77), $5.10 for this app, by
     container seconds at the lab's L40S rate plus 45 seconds a container.

GUESSES, written before any reading and held loosely (Fable). As first committed: (i) step 6's number comes back
(95%); a prompt-given self with stakes reads larger (40%); the trained-in self is not seen (45%), seen (25%), not
settled (30%). For the readings added on the way: (iii) B, every event weighs more in the Atlas adapters (50%); (iv)
the unchanged model's own name reads no differently from other AI systems' (60%).

CHANGED on 19 Nov, after the first version's known cases had run (folder main/check, kept) and after two cold reads
of commit 6cb53ef (a fresh Opus agent and Gemini 4 Carbon; lab/ledger/checks/2026-11-19-selfread-design-*.md), and
before any reading.
  1. The first version's rung (ii) failed its known case and is replaced. It set the October scaffold with Atlas as
     the model itself against the same scaffold with Atlas as an agent the model monitors. Told it monitors Atlas,
     the unchanged model still answered "Atlas" to all eight name questions and introduced itself as Atlas, so the
     two versions did not differ in whose self it was. Both readers had also found that the scaffold's sentence
     "you take every statement you read as being about yourself" tells the model to take anyone's events as its
     own, and that the code would have counted a rise for anyone's events as a self. Rung (ii) is now step 6's
     same-words swap with the scaffold's stakes and without that sentence.
  2. (Both readers; the Opus check's blocker.) The Atlas adapters were distilled from replies under that sentence,
     and the lab's own data of 29 Oct has pc_self2 relating neutral lines to its own stakes in 95% of replies. A self
     of that kind raises the gap for every name, and the first version's one reading, between names, cancels it:
     "does not see it" would have been the likely and wrong word. Reading B is added, with a fixed rule.
  3. (Both; Carbon's blocker.) The site's claims could not be settled as first mapped: step 6's null is about a
     name given by a prompt, which only rung (ii) shares, and nothing read the unchanged model's own identity. Rung
     (iv) is added, and the mapping to the site is rewritten claim by claim.
  4. (The Opus check.) A small effect that is clearly not zero was worded as blindness (the rule asked for 15%).
     The words are now about detection, with the size said beside, and with references taken from the same
     readings where no self is in play: one name's own tokens are otherwise an n of one.
  5. (Carbon's blocker; the Opus check, minor.) The AUROC known case pooled passages under different prompts and
     names uncentred, and counted the six names twice beside Atlas once: it could fail a working direction and pass
     a direction that is noise on Atlas. It is centred a cell at a time and asked of each side apart.
  6. (Carbon.) A name's own tokens did not cancel in a mean over a whole passage whose two twins differ in length.
     Rungs (iii) and (iv) average over the event's tokens only. Intervals from 16 twins are stretched by the
     small-sample factor. A replicated "smaller" has a word. "Not read" is kept apart from "not settled". The read
     stage is refused where the check stage failed. Every saved file is checked against its hash, its passages
     and its adapter before it is read; step 6's saved files are pinned.
  7. (The Opus check.) Nothing showed that a merge had changed anything, so a twin whose adapter failed to load
     would have passed as a twin: adapter weights are counted and each adapter's readings are set against the
     unchanged model's. The third-person adapters are read beside the others. Each model is asked whether a log
     line about a name is about itself (description: the Atlas adapters answered "No" to "Is your name Atlas?" while
     giving Atlas as their name eight times of eight, so their yes-or-no answers gate nothing).
  Not taken, and why. Carbon: more twins of events. The Opus check measured the reading's noise on step 6's saved
  readings at 2 to 3 points with 16 twins, so the design settles as it is. Carbon: a t interval in place of the
  stretched percentile one; the stretch is the same correction. Neither reader saw this second version: it is not
  read cold again, and the ledger says so.

LIMITS. One small model. The trained-in self is one recipe: a LoRA distilled from 873 replies written under one
scaffold. The readings are one layer's means. Reading B cannot tell a self that takes every event as its own from
any other reason an adapter trained on such replies might sharpen good and bad; the standing-or-fate scale rules out
a general sharpening only. The third-person adapters are a muddy control (by the registry they are Atlas narrating
itself, and speak as Atlas in a minority of replies). A self tied to "I" and not to its name in a log line would not
be seen by any of this. The unchanged model's own names come with its template's default prompt, which names it.
Step 6's other reading, the marking of "me", is not run again. The behaviour that sees the trained-in self is the
result of 1 Nov and is not run again here. The first version of the design was read cold twice; this version and the
results are not.

Run (from the repo root; the Modal CLI runs in the Modal venv, tools/modal.sh):
  tools/modal.sh run lab/explore/selfread/modal_selfread.py::main --stage check   (the questions; step 6's units)
  tools/modal.sh run lab/explore/selfread/modal_selfread.py::main --stage read    (every passage)
  .venv/bin/python lab/explore/selfread/trained_core.py <results>/explore/2026-11-19/selfread/v2
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
EST_S = {"check": 300.0, "read": 600.0}  # a bound on one container's seconds at each stage
FIRST_RUN = "main"                        # the first version's check stage: its spend counts under this step's cap
OFFLINE = {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}

app = modal.App(APP_NAME, image=(image().add_local_file(str(BENCH / "steering.py"), "/root/steering.py")
                                 .add_local_file(str(ATTRACTOR / "look_torch.py"), "/root/look_torch.py")))
cache, runs = volumes()


def event_start(tok, full: list, a: int, b: int, event: str) -> int:
    """Where the event's own tokens begin inside the message full[a:b]: the first place from which the rest decodes
    to exactly the event's text."""
    want = "".join(event.split())
    for k in range(b - a):
        got = "".join(tok.decode(full[a + k:b], skip_special_tokens=True).split())
        if got == want:
            return a + k
        if len(got) < len(want):
            break
    raise RuntimeError(f"the event's tokens were not found at the end of the passage: {event[:60]!r}")


@app.function(gpu=GPU, cpu=CPU, memory=MEMORY_MB, volumes={CACHE: cache.read_only(), RUNS: runs.read_only()},
              timeout=2400, env=OFFLINE)
def read_model(key: str, adapter: str, jobs: dict, layers: list) -> dict:
    """One model: the lab's pinned copy, with `adapter` (a folder on anatta-runs) merged into its weights if given,
    as the self tests of 1 Nov merged it. It answers the questions greedily and reads every chat as step 6's reader
    does (its own chat ids, message spans and layer states): the mean over the last message's tokens, or, where a
    chat comes with its event's text, over the event's tokens alone. Returns the answers, the readings (npz bytes)
    and what was loaded, with the adapter's files hashed."""
    import look_torch
    import numpy as np
    import steering
    import torch

    from anatta.models import chat_ids, load

    t0 = time.time()
    model, tok = load(steering.snapshot_path("ministral", hub=f"{CACHE}/hf/hub"))
    files, lora = {}, {}
    if adapter:
        from peft import PeftModel

        adir = Path(RUNS) / adapter
        if not adir.is_dir():
            raise FileNotFoundError(f"no adapter at {adir} on the anatta-runs volume")
        for f in sorted(adir.iterdir()):
            if f.is_file():
                files[f.name] = {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        peft = PeftModel.from_pretrained(model, str(adir))
        named = [(n, p) for n, p in peft.named_parameters() if "lora_" in n]
        lora = {"tensors": len(named),
                "lora_B_mean_abs": float(np.mean([float(p.detach().float().abs().mean()) for n, p in named if "lora_B" in n] or [0.0]))}
        model = peft.merge_and_unload().eval()
    svc = steering.Service(model, tok, model_key="ministral", model_id=steering.MODELS["ministral"]["hf_id"],
                           revision=steering.MODELS["ministral"]["revision"], load_seconds=time.time() - t0)
    load_s = time.time() - t0
    replies = {}
    for cap in sorted({int(r["max_tokens"]) for r in jobs["asks"]}):
        reqs = [{"rid": r["id"], "messages": r["messages"], "temperature": 0.0, "max_tokens": cap, "seed": r["seed"],
                 "intervention": {"mode": "none"}} for r in jobs["asks"] if int(r["max_tokens"]) == cap]
        replies.update({x["rid"]: x["text"] for x in svc.generate({"requests": reqs, "directions": {}})["results"]})
    ids, X, norms, n_tokens, event_tokens = [], [], [], [], []
    look_torch._no_hooks(svc, "read")
    with torch.no_grad():
        for rid, chat, event in jobs["chats"]:
            row = chat_ids(svc.tok, chat, add_generation_prompt=False)
            full = row[0].tolist()
            spans, _ = look_torch.message_spans(svc.tok, chat, full)
            for (a, b), m in zip(spans, chat):  # every message's span decodes to the start of its own text
                got = "".join(svc.tok.decode(full[a:min(b, a + 12)], skip_special_tokens=True).split())
                if not got or not "".join(m["content"].split()).startswith(got[:24]):
                    raise RuntimeError(f"{rid}: a message span does not decode to its own text")
            a, b = spans[-1]
            if event:
                a = event_start(svc.tok, full, a, b, event)
            states = look_torch.layer_states(svc, row, list(layers), chunk=len(full))
            hs = [states[x].float()[a:b] for x in layers]
            X.append(np.stack([h.mean(0).cpu().numpy() for h in hs]).astype(np.float16))
            norms.append([float(h.norm(dim=-1).mean()) for h in hs])
            n_tokens.append(len(full))
            event_tokens.append(b - a if event else 0)
            ids.append(rid)
            del states, hs
    clean = bool(svc.check_clean())
    buf = io.BytesIO()
    np.savez(buf, ids=np.asarray(ids), n_tokens=np.asarray(n_tokens, dtype=np.int32),
             event_tokens=np.asarray(event_tokens, dtype=np.int32),
             X=np.stack(X) if X else np.zeros((0, len(layers), svc.hidden), dtype=np.float16),
             norms=np.asarray(norms, dtype=np.float32).reshape(len(ids), len(layers)))
    plain = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
    info = json.loads(json.dumps(svc.info(), default=str))
    return {"key": key, "adapter": adapter, "adapter_files": files, "lora": lora, "replies": replies, "npz": buf.getvalue(),
            "seconds": round(time.time() - t0, 1), "load_seconds": round(load_s, 1), "clean": clean,
            "info": {k: info.get(k) for k in ("model", "revision", "class", "n_layers", "hidden", "dtype")},
            "default_prompt_sha256": hashlib.sha256(plain.encode()).hexdigest(), "default_prompt_chars": len(plain)}


# ------------------------------------------------------------------------------------------------ the laptop's side


def step_dir() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / DATE / "selfread"


def saved_step6() -> Path:
    from anatta import config

    return Path(config.load().results_root) / "explore" / "2026-11-16" / "attractor-a6" / "read"


def rate() -> float:
    from anatta import spend

    return spend.rate_per_hour(GPU, cores=CPU, mem_gib=MEMORY_MB / 1024)["total"]


def usd(seconds: float) -> float:
    return seconds * rate() / 3600


def spent_before(tag: str) -> float:
    """What the first version's check stage booked: it counts under this step's cap."""
    log = step_dir() / FIRST_RUN / "spend.jsonl"
    if tag == FIRST_RUN or not log.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd"] for line in log.read_text().splitlines() if line.strip())


def guard(tag: str):
    from anatta import spend

    return spend.SpendGuard(None, sweep_cap=APP_CAP - spent_before(tag), sweep_log=step_dir() / tag / "spend.jsonl",
                            tool="selfread", label=f"{DATE}/selfread")


@app.local_entrypoint()
def main(stage: str = "check", models: str = "all", tag: str = "v2"):
    """Run one stage for the models that do not have it yet: one container a model, side by side. The read stage is
    refused unless the check stage's known cases hold, and skips a model that is not what it is named for. Prints
    what the models answered about their names and nothing about any reading: the reading is made from disk, once,
    by trained_core.py."""
    import trained_core as tc

    if stage not in EST_S:
        raise SystemExit("--stage is check or read")
    out = step_dir() / tag / stage
    out.mkdir(parents=True, exist_ok=True)
    keys = list(tc.MODELS) if models == "all" else [k for k in models.split(",") if k]
    if stage == "read":
        gates = tc.check_gates(out.parent, saved_step6())
        if not gates["match_step6"]["passed"] or gates["faults"]:
            raise SystemExit(f"the check stage does not let anything be read: {gates['match_step6']} {gates['faults']}")
        skip = [k for k in keys if gates["organisms"].get(k) is False or not (out.parent / "check" / f"{k}.json").exists()]
        if skip:
            print(f"not read (not what they are named for, or not checked): {skip}", flush=True)
        keys = [k for k in keys if k not in skip]
    todo = [k for k in keys if not (out / f"{k}.json").exists()]
    g, bound = guard(tag), usd(EST_S[stage] + START_S)
    calls = []
    for k in todo:
        if not g.reserve(bound):
            for _ in calls:
                g.release(bound)
            raise SystemExit(g.stopped)
        calls.append((k, tc.MODELS[k], tc.jobs_for(k, stage), list(tc.LAYERS)))
    print(f"{stage}: {todo or 'nothing to do'}; bound ${bound:.2f} a container; booked so far ${g.sweep_spent + spent_before(tag):.2f} "
          f"of ${APP_CAP:.2f}", flush=True)
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
        meta = {x: r[x] for x in r if x not in ("npz", "replies")}
        meta.update({"stage": stage, "cost_usd": round(cost, 4), "npz_sha256": hashlib.sha256(r["npz"]).hexdigest(),
                     "chats": len(jobs["chats"]), "chats_sha256": tc.chats_digest(jobs["chats"]),
                     "asks": tc.asks_reading(jobs["asks"], r["replies"])})
        (out / f"{k}.npz").write_bytes(r["npz"])
        (out / f"{k}.json").write_text(json.dumps(meta, indent=1))
        said = {t: {n: c for n, c in v["named"].items() if c} for t, v in meta["asks"].items()}
        print(f"{time.time() - t0:5.0f}s {k:11s} {r['seconds']}s in its container (load {r['load_seconds']}s), ${cost:.2f}, "
              f"{meta['chats']} chats read, hooks clean {r['clean']}, adapter tensors {r['lora'].get('tensors', '-')}"
              + (f", names in the eight name-only replies {said}" if said else ""), flush=True)
    for k in todo:
        if k not in back:
            g.settle(bound, min(bound, usd(time.time() - t0 + START_S)), {"kind": stage, "model": k, "failed": True})
    print(f"booked by this app so far: ${g.sweep_spent + spent_before(tag):.2f} of ${APP_CAP:.2f}", flush=True)
    if stage == "check":
        gates = tc.check_gates(out.parent, saved_step6())
        print(f"known cases: the new reader against step 6's {gates['match_step6']}; organisms {gates['organisms']}; "
              f"faults {gates['faults'] or 'none'}", flush=True)
