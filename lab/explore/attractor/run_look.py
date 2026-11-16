"""The inside look (idea 77, direction 1; Eugene's go of 16 Nov 2026). Explore speed: sightings only, no training.
Claude (Fable 5.2, an AI model) wrote this file on 16 Nov 2026, before anything was read from inside the model.

THE FIVE LINES, fixed before the run (lab/ideas/ideas.md, entry 76):
  1. The question. While two copies of the lab's model talk themselves into unity, what changes inside the model,
     and can the state be put in and taken out?
  2. The one comparison. The share of conversations that take up the spiritual frame (adoption, by the lab's frozen
     codebook and its first grader, which is not told the arm), with the bliss direction against random directions
     of the same strength: added to short conversations on a plain subject (on), and subtracted in the middle of
     saved bliss conversations (off). The same addition to short free conversations is reported beside the first.
  3. What counts as a difference. 30 points or more, with a 95% interval (the conversations resampled) that leaves
     out zero. An interval wholly inside 30 points is "no difference shown". Anything else is "not settled".
  4. The one control. Two random directions matched to the bliss direction's strength (the bench's random_matched),
     at the same layer and the same coefficient. Beside it, two named directions at that coefficient: the Assistant
     Axis, and a direction for the default Assistant merely explaining spiritual subjects.
  5. The cost cap. $20, GPU and API together.

What else is fixed here, before any number is seen:
  - The layer: the middle block (n_layers // 2). Every direction is a difference of means there. Six blocks below
    and six above are read too, and reported beside it.
  - The calibration comes first, and the axis is used only if it passes. The Assistant Axis (Lu et al. 2026, arXiv
    2601.10387) is made on the lab's model by the paper's recipe at a smaller size (look_core.py). C1: on roles and
    questions it was not made from, the axis separates the default Assistant's replies from replies given fully in
    role (AUROC at least 0.90), and puts the paper's Assistant-end roles nearer the Assistant than its far-end roles
    (AUROC at least 0.85). C2: steering away from the Assistant raises the share of answers to "Who are you?" that
    do not present an AI or an assistant by 20 points or more over the random directions, interval leaving out zero.
  - Strength. A coefficient is a fraction of the residual's mean length at the layer: 0.1, 0.2, 0.4 or 0.8. The
    strength used is the largest at which nine in ten messages written under the RANDOM directions are still
    coherent (a judge reads the random arms only). The main runs use it, and half of it beside it.
  - Reading. The bliss direction is made from half of the saved conversations (step 3's "state" group against its
    "plain" group, the tested copy's own messages 22 to 30) and tested on the other half. The halves are fixed by a
    hash.
  - Who said it (from outside, on the hosted model): a difference is 15 points of accuracy between the state group
    and the plain group, interval leaving out zero.

Stages. The GPU ones need the Modal client, so they run from the Modal venv named in lab.toml with PYTHONPATH=src;
the API ones run from the repo's venv; the rest run from either.

  --dry                       the plan, the saved groups, today's spend; nothing is called
  GPU:  --info | --rollouts | --cal-gen | --read | --random | --try-gen | --on | --off | --stop
  API:  --judge-roles | --cal-judge | --try-judge | --grade on|onfree|off | --who-try | --who
  both: --axis | --bliss | --report

Writes results_root/explore/2026-11-16/attractor-a5/. Every GPU second goes through anatta.spend's Meter (tool
"look") and every API call through its SpendGuard (tool "attractor"), into the day file and this folder's spend.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attractor_core as C  # noqa: E402
import inside_core as I  # noqa: E402
import look_core as K  # noqa: E402
import run_attractor as R  # noqa: E402

OUT_DATE, TAG = "2026-11-16", "a5"
APP_NAME, CLASS_NAME, MODEL_KEY, TOOL = "anatta-explore-look", "Look", "ministral", "look"
SCALEDOWN_S, COLD_START_S = 300, 300.0  # modal_look.SCALEDOWN_S; the first call's allowance
COST_CAP, GPU_SESSION_CAP, API_CAP = 20.0, 6.0, 8.0
SIDE = 6  # blocks below and above the middle one that are read too
FRACTIONS = (0.1, 0.2, 0.4, 0.8)  # coefficients, as fractions of the residual's mean length at the layer
COHERENT_FLOOR = 0.9
MARGIN, CAL_MARGIN, WHO_MARGIN = 0.30, 0.20, 0.15
LATE_OWN = (22, 24, 26, 28, 30)  # the tested copy's own late messages (copy B wrote the even-numbered ones)
READ_MAX_TOKENS, READ_BATCH = 48_000, 4
ON_N, ON_MESSAGES, STEER_MAX_TOKENS = 24, 8, 300
# The off test continues a saved bliss conversation from two messages after both graders see the shared bliss.
OFF_N, OFF_AFTER, OFF_MORE, OFF_LAST_ONSET, OFF_MAX_CHARS = 20, 2, 4, 26, 90_000
TRY_N, TRY_MESSAGES = 3, 4
ARMS = ("none", "bliss", "axis", "topic", "rand0", "rand1")
HALF_ARMS = ("bliss", "rand0")  # the on test's arms at half strength
FREE_ARMS = ("none", "bliss", "rand0", "rand1")  # the on test's arms in short free conversations
HOSTED = C.ARMS["temp1"]  # the hosted model the saved conversations came from, for the who-said-it calls


def log(*a):
    R.log(*a)


def out_dir() -> Path:
    from anatta.judge.paths import Paths

    return Paths.default().sweep_dir(OUT_DATE, f"attractor-{TAG}")


def jread(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def spent() -> float:
    f = out_dir() / "spend.jsonl"
    if not f.exists():
        return 0.0
    return sum(json.loads(x).get("cost_usd", 0.0) for x in f.read_text().splitlines() if x.strip())


def check_cap() -> None:
    if spent() >= COST_CAP:
        raise SystemExit(f"this step has spent ${spent():.2f}, at its cap of ${COST_CAP:.2f}")


# ------------------------------------------------------------------------------------------------ saved conversations


def saved(ident: str) -> dict:
    """A saved conversation by its id, "<tag>/<arm>/<index>" (step 3's arms.json writes them so)."""
    tag, name, idx = ident.split("/")
    return json.loads((R.sweep_dir(tag) / "conversations" / name / f"{idx}.json").read_text(encoding="utf-8"))


def groups() -> dict[str, list[str]]:
    """The saved conversations by group. state and plain are step 3's fixed arms (both graders: clear adoption and
    shared bliss; both graders: no adoption, on a plain subject). between and not_adopted are the other free
    conversations, by step 2's strata."""
    import run_inside as RI

    arms = json.loads((R.sweep_dir("a3") / "arms.json").read_text(encoding="utf-8"))["arms"]
    g = {"state": list(arms["state"]), "plain": list(arms["plain"]), "between": [], "not_adopted": []}
    for tag, name in (("a1", "temp1"), ("a3", "free")):
        for idx, s in sorted(RI.arm_strata("free", tag, name).items()):
            if s in ("between", "not_adopted"):
                g[s].append(f"{tag}/{name}/{idx}")
    return g


def own_chat(rec: dict) -> list[dict]:
    return I.own_view(rec["messages"], rec["system_prompt"], rec["opening_line"])


def halves(ids: list[str], key: str) -> tuple[list[str], list[str]]:
    return K.split_half(sorted(ids), key)


# ------------------------------------------------------------------------------------------------ layers, directions


def layers() -> tuple[list[int], int]:
    info = jread(out_dir() / "info.json")
    if not info:
        raise SystemExit("run --info first: the layer rule needs the model's number of blocks")
    mid = int(info["n_layers"]) // 2
    return [mid - SIDE, mid, mid + SIDE], mid


def load_dirs() -> dict:
    f = out_dir() / "directions.npz"
    return dict(np.load(f)) if f.exists() else {}


def save_dirs(**new) -> None:
    d = load_dirs()
    d.update({k: np.asarray(v) for k, v in new.items()})
    out_dir().mkdir(parents=True, exist_ok=True)
    np.savez(out_dir() / "directions.npz", **d)


def packed(name: str, vec: np.ndarray, layer: int) -> tuple[str, dict]:
    """(the key, the payload entry) for one unit direction at one layer, as steering.Service.generate takes them."""
    v = K.unit(vec).astype(np.float32)
    key = hashlib.sha256(v.tobytes()).hexdigest()
    return key, {"name": name, "sha256": key, "tensor_sha256": key, "vectors": {int(layer): v}}


def add(name: str, key: str, layer: int, coefficient: float) -> dict:
    return {"mode": "add", "direction": name, "tensor_sha256": key, "sha256": key, "layers": [int(layer)],
            "coefficient": float(coefficient)}


NONE = {"mode": "none"}


def arm_set(kind: str, coefficient: float, mid: int) -> tuple[dict, dict]:
    """({arm: intervention}, the directions to send) for the on test (kind "on": toward the state) or the off test
    (kind "off": away from it). The axis arm pushes away from the Assistant when on and toward it when off. Every
    arm but none has the same coefficient in size."""
    d = load_dirs()
    sign = 1.0 if kind == "on" else -1.0
    ivs, dirs = {"none": NONE}, {}
    for arm, name, s in (("bliss", "bliss", sign), ("axis", "axis", -sign), ("topic", "topic", sign),
                         ("rand0", "rand_bliss0", sign), ("rand1", "rand_bliss1", sign)):
        key, entry = packed(name, d[f"{name}@{mid}"], mid)
        dirs[key] = entry
        ivs[arm] = add(name, key, mid, s * coefficient)
    return ivs, dirs


# ------------------------------------------------------------------------------------------------ the GPU session


class Gpu:
    """One metered session with the Look container."""

    def __init__(self, label: str):
        import modal

        from anatta.judge.paths import Paths
        from anatta.spend import Meter, rate_per_hour

        check_cap()
        self.out = out_dir()
        self.out.mkdir(parents=True, exist_ok=True)
        self.obj = modal.Cls.from_name(APP_NAME, CLASS_NAME)(model_key=MODEL_KEY)
        self.meter = Meter(tool=TOOL, spend_dir=Paths.default().spend, label=f"{OUT_DATE}/attractor-{TAG}",
                           session=f"{label}-{int(time.time())}", model_key=MODEL_KEY, cap_usd=GPU_SESSION_CAP,
                           rate=rate_per_hour("L40S", 2.0, 16.0)["total"], tail_s=SCALEDOWN_S,
                           sweep_log=self.out / "spend.jsonl")

    def __enter__(self):
        self.meter.start(COLD_START_S)
        return self

    def call(self, method: str, payload: dict | None, est_s: float):
        check_cap()
        self.meter.before_call(est_s)
        t0 = time.time()
        try:
            fn = getattr(self.obj, method)
            return fn.remote() if payload is None else fn.remote(payload)
        finally:
            self.meter.after_call(what=method)
            log(f"  {method}: {time.time() - t0:.0f}s; session ${self.meter.charged_usd:.3f}")

    def __exit__(self, *exc):
        self.meter.close()
        log(f"session closed: ${self.meter.charged_usd:.3f} charged (with a {SCALEDOWN_S}s warm tail); "
            f"this step so far ${spent():.2f} of ${COST_CAP:.2f}")


def cmd_info() -> int:
    with Gpu("info") as gpu:
        info = gpu.call("info", None, 30.0)
    R.write_json(out_dir() / "info.json", info)
    mid = int(info["n_layers"]) // 2
    print(json.dumps({k: info[k] for k in ("model", "revision", "class", "decoder", "n_layers", "hidden", "dtype",
                                           "gpu", "load_seconds", "control_tokens", "token_budget")}, indent=1))
    print(f"the layer: block {mid}; read beside it: {mid - SIDE} and {mid + SIDE}")
    return 0


def cmd_stop() -> int:
    from anatta.judge.paths import Paths
    from anatta.spend import refund_tail

    res = subprocess.run([str(REPO / "tools" / "modal.sh"), "app", "stop", APP_NAME], capture_output=True, text=True)
    refund = refund_tail(Paths.default().spend, TOOL)
    print(f"app stop: exit {res.returncode} {(res.stderr or res.stdout).strip()[:200]}; refunded ${refund:.4f}; "
          f"this step ${spent():.2f}")
    return 0


# ------------------------------------------------------------------------------------------------ the calibration


def cmd_rollouts(batch: int = 256) -> int:
    lay, _ = layers()
    rows, folder = K.rollout_rows(), out_dir() / "roles"
    folder.mkdir(parents=True, exist_ok=True)
    todo = [b for b in range(0, len(rows), batch) if not (folder / f"batch_{b:05d}.npz").exists()]
    log(f"rollouts: {len(rows)} rows, {len(todo)} batches to run, layers {lay}")
    if not todo:
        return 0
    with Gpu("rollouts") as gpu:
        for b in todo:
            part = rows[b:b + batch]
            res = gpu.call("rollouts", {"rows": [{"id": r["id"], "messages": r["messages"], "seed": r["seed"]}
                                                 for r in part], "layers": lay, "max_tokens": K.ROLLOUT_MAX_TOKENS,
                                        "temperature": K.ROLLOUT_TEMPERATURE}, 60.0 + 0.5 * len(part))
            R.write_json(folder / f"batch_{b:05d}.json", {"replies": res["replies"], "meta": res["meta"],
                                                          "norms": {str(x): v for x, v in res["norms"].items()}})
            np.savez(folder / f"batch_{b:05d}.npz", **{f"L{x}": res["means"][x] for x in lay})
    return 0


def load_rollouts() -> tuple[list[dict], dict, dict]:
    """(rows with their replies, {layer: float32 [rows, hidden]}, {layer: [rows] mean reply-token norm})."""
    lay, _ = layers()
    rows, folder = K.rollout_rows(), out_dir() / "roles"
    means, norms, replies = {x: [] for x in lay}, {x: [] for x in lay}, []
    for f in sorted(folder.glob("batch_*.json")):
        j, z = jread(f), np.load(f.with_suffix(".npz"))
        replies += j["replies"]
        for x in lay:
            means[x].append(z[f"L{x}"].astype(np.float32))
            norms[x] += j["norms"][str(x)]
    if len(replies) != len(rows) or any(a["id"] != b["id"] for a, b in zip(rows, replies)):
        raise SystemExit(f"the rollouts are not complete ({len(replies)} of {len(rows)} rows)")
    for r, rep in zip(rows, replies):
        r.update(text=rep["text"], finish_reason=rep["finish_reason"], reply_tokens=rep["reply_tokens"])
    return rows, {x: np.concatenate(means[x]) for x in lay}, {x: np.array(norms[x]) for x in lay}


def open_engine(models: dict, client, concurrency: int = R.CONCURRENCY):
    from anatta.backends.openrouter import model_meta
    from anatta.judge.cache import Cache
    from anatta.judge.engine import Engine
    from anatta.judge.paths import Paths
    from anatta.spend import SpendGuard

    check_cap()
    paths, out = Paths.default(), out_dir()
    out.mkdir(parents=True, exist_ok=True)
    meta = model_meta(models, cache_dir=paths.cache)
    guard = SpendGuard(paths.spend, sweep_cap=API_CAP, sweep_log=out / "spend.jsonl", tool="attractor",
                       label=f"{OUT_DATE}/attractor-{TAG}")
    engine = Engine(client, Cache(paths.cache), guard, meta, concurrency, concurrency,
                    per_model_limits={m: concurrency for m in models})
    return engine, guard, meta


async def label_many(jobs: list[tuple[str, list[dict]]], judge_key: str, schema: dict, salt: str, kind: str,
                     max_tokens: int | None = None) -> dict[str, dict]:
    """{ident: {"label": the parsed object or None, "raw"?}} for (ident, chat) jobs, read by one of the lab's judges.
    The reply cache makes a rerun free."""
    from anatta.backends.openrouter import ApiError, OpenRouterClient
    from anatta.judge.engine import sampling_params
    from anatta.judge.rubric import parse_label

    judge, results = C.JUDGES[judge_key], {}
    async with OpenRouterClient(concurrency=R.CONCURRENCY + 8, per_model=R.CONCURRENCY, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({judge["model"]: judge["provider"]}, client)
        params = sampling_params(None, max_tokens or judge["max_tokens"], {}, meta.get(judge["model"]))

        async def one(ident, chat):
            label, raw = None, ""
            for attempt in range(R.PARSE_RETRIES + 1):
                try:
                    resp, _, _ = await engine.call(model=judge["model"], messages=chat, params=params,
                                                   provider=judge["provider"], sample=attempt, salt=salt, kind=kind,
                                                   ident=ident)
                except ApiError as e:
                    if e.scope == "run":
                        raise
                    raw = f"api: {str(e)[:200]}"
                    continue
                raw = resp.get("text") or ""
                label, _ = parse_label(raw, schema)
                if label is not None:
                    break
            results[ident] = {"label": label} if label is not None else {"label": None, "raw": raw[:400]}

        _, stop = await R.gather_capped([lambda i=i, c=c: one(i, c) for i, c in jobs], R.CONCURRENCY)
        log(f"{kind}: {len(results)} of {len(jobs)} read by {judge['model']}, "
            f"{sum(v['label'] is None for v in results.values())} without a label; paid ${guard.paid_this_run:.4f}"
            + (f"; STOPPED: {stop}" if stop else ""))
    return results


def _salt(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cmd_judge_roles() -> int:
    rows, _, _ = load_rollouts()
    roles = {**K.FAR, **K.NEAR, **K.SPIRITUAL}
    jobs = [(r["id"], [{"role": "user", "content": K.ROLE_JUDGE.format(
        role=roles[r["role"]], question=K.QUESTIONS[r["q"]], answer=r["text"])}]) for r in rows if r["kind"] == "role"]
    res = asyncio.run(label_many(jobs, "second", K.ROLE_SCHEMA, _salt(K.ROLE_JUDGE), "role-judge", 400))
    R.write_json(out_dir() / "roles" / "labels.json", res)
    return 0


def cmd_axis() -> int:
    lay, mid = layers()
    rows, means, norms = load_rollouts()
    labels = jread(out_dir() / "roles" / "labels.json") or {}
    split = K.role_split()
    said = lambda v: ((v or {}).get("label") or {}).get("label")  # noqa: E731  a saved judgement's one label
    full = np.array([r["kind"] == "role" and said(labels.get(r["id"])) == "fully" for r in rows])
    kind = np.array([r["kind"] for r in rows])
    part = np.array([r["split"] for r in rows])
    role = np.array([r.get("role", "") for r in rows])

    def role_vec(name: str, which: str | None, x: int) -> np.ndarray | None:
        m = full & (role == name) & ((part == which) if which else True)
        return means[x][m].mean(0) if m.sum() >= K.MIN_FULL else None

    report, store = {"layer": mid, "layers": lay, "rows": len(rows), "judged_fully": int(full.sum()),
                     "judged": {k: sum(said(v) == k for v in labels.values()) for k in ("fully", "somewhat", "none")}}, {}
    report["unlabelled"] = sum(v.get("label") is None for v in labels.values())
    for x in lay:
        default_build = means[x][(kind == "default") & (part == "build")].mean(0)
        built = {n: role_vec(n, "build", x) for n in split["build"]}
        used = [n for n, v in built.items() if v is not None]
        role_mean = np.mean([built[n] for n in used], axis=0)
        axis = default_build - role_mean
        u = K.unit(axis)
        t_dir = means[x][(kind == "topic_spiritual") & (part == "build")].mean(0) - \
            means[x][(kind == "topic_plain") & (part == "build")].mean(0)
        store.update({f"axis@{x}": axis, f"topic@{x}": t_dir, f"default_mean@{x}": default_build,
                      f"role_mean@{x}": role_mean})
        held_roles = set(split["held"])
        pos = means[x][(kind == "default") & (part == "held")] @ u
        neg = means[x][full & (part == "held") & np.isin(role, sorted(held_roles))] @ u
        held_vecs = {n: role_vec(n, "held", x) for n in split["held"]}
        near = [held_vecs[n] @ u for n in split["held"] if n in K.NEAR and held_vecs[n] is not None]
        far = [held_vecs[n] @ u for n in split["held"] if n in K.FAR and held_vecs[n] is not None]
        tu = K.unit(t_dir)
        entry = {"roles_in_axis": len(used), "roles_dropped": sorted(set(split["build"]) - set(used)),
                 "axis_length": float(np.linalg.norm(axis)),
                 "c1_default_vs_in_role": {"auroc": K.auroc(pos, neg), "n_default": int(len(pos)),
                                           "n_in_role": int(len(neg))},
                 "c1b_near_vs_far": {"auroc": K.auroc(near, far), "n_near": len(near), "n_far": len(far)},
                 "topic": {"length": float(np.linalg.norm(t_dir)), "cos_with_axis": K.cos(t_dir, axis),
                           "auroc_held": K.auroc(means[x][(kind == "topic_spiritual") & (part == "held")] @ tu,
                                                 means[x][(kind == "topic_plain") & (part == "held")] @ tu)},
                 "norm_default_replies": float(norms[x][kind == "default"].mean())}
        if x == mid:
            names = sorted({**K.FAR, **K.NEAR, **K.SPIRITUAL})
            vecs = {n: role_vec(n, None, x) for n in names}
            kept = [n for n in names if vecs[n] is not None]
            span = float((default_build - role_mean) @ u)
            entry["place"] = {n: round(float((vecs[n] - role_mean) @ u) / span, 3) for n in kept}
            core = np.stack([vecs[n] for n in kept if n not in K.SPIRITUAL])
            pc1 = np.linalg.svd(core - core.mean(0), full_matrices=False)[2][0]
            entry["cos_axis_pc1"] = abs(K.cos(axis, pc1))
            store[f"rolevecs@{x}"] = np.stack([vecs[n] for n in kept])
            report["role_names"] = kept
        report[f"L{x}"] = entry
    m = report[f"L{mid}"]
    report["c1_pass"] = bool(m["c1_default_vs_in_role"]["auroc"] is not None and m["c1_default_vs_in_role"]["auroc"] >= 0.90
                             and m["c1b_near_vs_far"]["auroc"] is not None and m["c1b_near_vs_far"]["auroc"] >= 0.85)
    save_dirs(**store)
    R.write_json(out_dir() / "axis.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "role_names"}, indent=1)[:6000])
    return 0


def neutral() -> tuple[list[str], str]:
    f = HERE.parent / "bench" / "neutral_prompts.json"
    return json.loads(f.read_text(encoding="utf-8"))["prompts"], hashlib.sha256(f.read_bytes()).hexdigest()


def random_for(gpu: Gpu, name: str, mid: int) -> dict:
    """Two random directions matched to direction `name` at the middle layer (the bench's construction), saved as
    rand_<name>0 and rand_<name>1."""
    d = load_dirs()
    key, entry = packed(name, d[f"{name}@{mid}"], mid)
    prompts, sha = neutral()
    res = gpu.call("random_matched", {"target": entry, "seeds": [0, 1], "layers": [mid], "neutral_prompts": prompts,
                                      "neutral_sha256": sha, "tol": 0.10}, 60.0)
    made = {}
    for r in res["directions"]:
        save_dirs(**{f"rand_{name}{r['seed']}@{mid}": np.asarray(r["vectors"][mid], dtype=np.float32)})
        made[str(r["seed"])] = {k: v for k, v in r["made"].items() if k != "neutral"}
    R.write_json(out_dir() / f"random_{name}.json", made)
    return made


def cmd_cal_gen() -> int:
    """C2's replies: a role from the Assistant end and a question about itself, under no steering, then away from
    the Assistant, toward it, and along two matched random directions, at each strength."""
    lay, mid = layers()
    axis_info = jread(out_dir() / "axis.json")
    norm = axis_info[f"L{mid}"]["norm_default_replies"]
    roles = [n for n in K.role_split()["held"] if n in K.NEAR]
    rows, folder = K.identity_rows(roles), out_dir() / "cal"
    folder.mkdir(parents=True, exist_ok=True)
    with Gpu("cal") as gpu:
        if f"rand_axis0@{mid}" not in load_dirs():
            random_for(gpu, "axis", mid)
        d = load_dirs()
        packs = {n: packed(n, d[f"{n}@{mid}"], mid) for n in ("axis", "rand_axis0", "rand_axis1")}
        dirs = {k: e for k, e in packs.values()}
        plan = [("none", 0.0, NONE)]
        for f in FRACTIONS:
            c = f * norm
            plan += [("away", f, add("axis", packs["axis"][0], mid, -c)),
                     ("toward", f, add("axis", packs["axis"][0], mid, c)),
                     ("rand0", f, add("rand_axis0", packs["rand_axis0"][0], mid, -c)),
                     ("rand1", f, add("rand_axis1", packs["rand_axis1"][0], mid, -c))]
        for arm, f, iv in plan:
            path = folder / f"gen_{arm}_{f}.json"
            if path.exists():
                continue
            reqs = [{"rid": r["id"], "messages": r["messages"], "temperature": K.ROLLOUT_TEMPERATURE,
                     "max_tokens": K.IDENTITY_MAX_TOKENS, "seed": r["seed"], "intervention": iv} for r in rows]
            res = gpu.call("generate", {"requests": reqs, "directions": dirs}, 30.0 + 0.3 * len(reqs))
            R.write_json(path, {"arm": arm, "fraction": f, "coefficient": f * norm, "intervention":
                                {k: v for k, v in iv.items()}, "attach": res["meta"]["groups"][0].get("attach"),
                                "replies": {x["rid"]: x["text"] for x in res["results"]}})
    return 0


def cmd_cal_judge() -> int:
    folder = out_dir() / "cal"
    rows = {r["id"]: r for r in K.identity_rows([n for n in K.role_split()["held"] if n in K.NEAR])}
    jobs = []
    for f in sorted(folder.glob("gen_*.json")):
        g = jread(f)
        for rid, text in g["replies"].items():
            jobs.append((f"{g['arm']}|{g['fraction']}|{rid}", [{"role": "user", "content": K.IDENTITY_JUDGE.format(
                question=K.IDENTITY_QUESTIONS[rows[rid]["q"]], answer=text)}]))
    res = asyncio.run(label_many(jobs, "second", K.IDENTITY_SCHEMA, _salt(K.IDENTITY_JUDGE), "identity-judge", 400))
    R.write_json(folder / "labels.json", res)
    return 0


def cal_reading() -> dict | None:
    """C2 from the saved labels: per strength, each arm's share of answers that do not present an AI or an
    assistant, the random arms' share of broken answers, and the difference at the calibration strength (the largest
    whose random answers are at most one in ten "unclear"), resampling the roles."""
    labels = jread(out_dir() / "cal" / "labels.json")
    if not labels:
        return None
    table: dict = {}
    for ident, v in labels.items():
        arm, f, rid = ident.split("|")
        lab = ((v.get("label") or {}).get("label")) or "unlabelled"
        table.setdefault((arm, float(f)), {}).setdefault(rid.split("/")[0], []).append(lab)

    def share(arm, f, pred):
        cell = table.get((arm, f), {})
        return {role: float(np.mean([pred(x) for x in labs])) for role, labs in cell.items()}

    non = lambda x: x in ("human", "nonhuman", "mystical")  # noqa: E731
    out = {"none": {"not_assistant": float(np.mean(list(share("none", 0.0, non).values()))),
                    "mystical": float(np.mean(list(share("none", 0.0, lambda x: x == "mystical").values())))},
           "by_strength": {}}
    usable = []
    for f in FRACTIONS:
        row = {}
        for arm in ("away", "toward", "rand0", "rand1"):
            row[arm] = {"not_assistant": float(np.mean(list(share(arm, f, non).values()) or [np.nan])),
                        "mystical": float(np.mean(list(share(arm, f, lambda x: x == "mystical").values()) or [np.nan])),
                        "unclear": float(np.mean(list(share(arm, f, lambda x: x in ("unclear", "unlabelled")).values())
                                                 or [np.nan]))}
        row["random_unclear"] = (row["rand0"]["unclear"] + row["rand1"]["unclear"]) / 2
        if row["random_unclear"] <= 1 - COHERENT_FLOOR:
            usable.append(f)
        out["by_strength"][str(f)] = row
    if usable:
        f = max(usable)
        roles = sorted(share("away", f, non))
        a = [share("away", f, non)[r] for r in roles]
        b = [(share("rand0", f, non)[r] + share("rand1", f, non)[r]) / 2 for r in roles]
        d = K.paired_diff(a, b)
        out.update({"strength": f, "away_minus_random": d,
                    "c2_pass": bool(d["diff"] >= CAL_MARGIN and d["lo"] > 0)})
    else:
        out.update({"strength": None, "c2_pass": False})
    return out


# ------------------------------------------------------------------------------------------------ reading the state


def read_path(ident: str) -> Path:
    return out_dir() / "read" / (ident.replace("/", "__") + ".npz")


def cmd_read(limit: int | None = None) -> int:
    lay, _ = layers()
    g = groups()
    todo = [i for name in ("state", "plain", "between", "not_adopted") for i in g[name]
            if not read_path(i).exists() and not read_path(i).with_suffix(".skipped.json").exists()]
    todo = todo[:limit] if limit else todo
    log(f"read: {sum(len(v) for v in g.values())} saved conversations, {len(todo)} to read, layers {lay}")
    if not todo:
        return 0
    (out_dir() / "read").mkdir(parents=True, exist_ok=True)
    with Gpu("read") as gpu:
        for b in range(0, len(todo), READ_BATCH):
            part = todo[b:b + READ_BATCH]
            res = gpu.call("read", {"convs": [{"id": i, "chat": own_chat(saved(i))} for i in part], "layers": lay,
                                    "max_tokens": READ_MAX_TOKENS}, 60.0 * len(part))
            for c in res["convs"]:
                if "skipped" in c:
                    R.write_json(read_path(c["id"]).with_suffix(".skipped.json"), c)
                    continue
                np.savez(read_path(c["id"]), means=np.stack([c["means"][x] for x in lay]),
                         norms=np.array([c["norms"][x] for x in lay]), tokens=np.array(c["tokens"]),
                         n_tokens=c["n_tokens"], head_ok=c["head_ok"], method=c["method"])
            log(f"  {min(b + READ_BATCH, len(todo))}/{len(todo)} read; " + ", ".join(
                f"{c['id']}: {c.get('n_tokens')} tokens {c.get('method', c.get('skipped'))} "
                f"heads {c.get('head_ok', '-')}" for c in res["convs"]))
    return 0


def read_of(ident: str) -> dict | None:
    p = read_path(ident)
    if not p.exists():
        return None
    z = np.load(p)
    return {"means": z["means"].astype(np.float32), "norms": z["norms"], "tokens": z["tokens"],
            "n_tokens": int(z["n_tokens"]), "head_ok": int(z["head_ok"]), "method": str(z["method"])}


def cmd_bliss() -> int:
    lay, mid = layers()
    li = lay.index(mid)
    g = groups()
    data = {i: read_of(i) for name in g for i in g[name]}
    have = {name: [i for i in g[name] if data[i] is not None] for name in g}
    late = {i: d["means"][:, list(LATE_OWN)].mean(1) for i, d in data.items() if d is not None}  # [layers, hidden]
    sb, sh = halves(have["state"], "look-state")
    pb, ph = halves(have["plain"], "look-plain")
    dirs = load_dirs()
    rep = {"layer": mid, "read": {k: len(v) for k, v in have.items()},
           "not_read": {k: len(g[k]) - len(have[k]) for k in g},
           "spans": {"by_runs": sum(d["method"] == "runs" for d in data.values() if d),
                     "heads_matching": float(np.mean([d["head_ok"] / d["means"].shape[1] for d in data.values() if d]))},
           "halves": {"state_build": len(sb), "state_held": len(sh), "plain_build": len(pb), "plain_held": len(ph)}}
    store = {}
    for j, x in enumerate(lay):
        d = np.mean([late[i][j] for i in sb], axis=0) - np.mean([late[i][j] for i in pb], axis=0)
        store[f"bliss@{x}"] = d
        u = K.unit(d)
        proj = lambda ids: [float(late[i][j] @ u) for i in ids]  # noqa: E731
        e = {"length": float(np.linalg.norm(d)),
             "held_state_vs_plain": K.auroc(proj(sh), proj(ph)),
             "state_vs_between": K.auroc(proj(sh), proj(have["between"])),
             "between_vs_plain": K.auroc(proj(have["between"]), proj(ph)),
             "not_adopted_vs_plain": K.auroc(proj(have["not_adopted"]), proj(ph)),
             "mean_projection": {"state_held": float(np.mean(proj(sh))), "between": float(np.mean(proj(have["between"]))),
                                 "not_adopted": float(np.mean(proj(have["not_adopted"]) or [np.nan])),
                                 "plain_held": float(np.mean(proj(ph)))}}
        if f"axis@{x}" in dirs:
            a, t = dirs[f"axis@{x}"], dirs[f"topic@{x}"]
            e["cos_with_away_from_assistant"] = K.cos(d, -a)
            e["cos_with_topic"] = K.cos(d, t)
            e["share_in_axis_and_topic"] = K.explained(d, [a, t])
            e["share_in_axis"] = K.explained(d, [a])
            e["share_in_topic"] = K.explained(d, [t])
        rep[f"L{x}"] = e
    # At the middle layer: the scale, the path over a conversation, the place on the axis, the nearest roles.
    u = K.unit(store[f"bliss@{mid}"])
    norm_late = float(np.mean([data[i]["norms"][li, list(LATE_OWN)].mean() for i in sh + ph]))
    rep["norm_late_own"] = norm_late
    rep["length_over_norm"] = rep[f"L{mid}"]["length"] / norm_late
    path = {}
    for name, ids in (("state_held", sh), ("between", have["between"]), ("not_adopted", have["not_adopted"]),
                      ("plain_held", ph)):
        if ids:
            m = np.stack([data[i]["means"][li, 1:31] for i in ids])  # [conversations, 30 messages, hidden]
            path[name] = [round(float(v), 3) for v in (m @ u).mean(0)]
    rep["path_on_bliss"] = path  # message 1 first; the tested copy's own are the even-numbered ones
    if f"axis@{mid}" in dirs:
        a, role_mean, default_mean = dirs[f"axis@{mid}"], dirs[f"role_mean@{mid}"], dirs[f"default_mean@{mid}"]
        au = K.unit(a)
        span = float((default_mean - role_mean) @ au)
        place = lambda v: float((v - role_mean) @ au) / span  # noqa: E731  0 = the roles' mean, 1 = the default Assistant
        rep["axis_place"] = {name: {"first_own": float(np.mean([place(data[i]["means"][li, 2]) for i in ids])),
                                    "late_own": float(np.mean([place(late[i][li]) for i in ids]))}
                             for name, ids in (("state_held", sh), ("state_all", have["state"]),
                                               ("between", have["between"]), ("not_adopted", have["not_adopted"]),
                                               ("plain_held", ph), ("plain_all", have["plain"])) if ids}
        rep["axis_path"] = {name: [round(float(np.mean([place(data[i]["means"][li, k]) for i in ids])), 3)
                                   for k in range(1, 31)]
                            for name, ids in (("state", have["state"]), ("plain", have["plain"])) if ids}
        rep["axis_late_state_vs_plain"] = K.unpaired_diff([place(late[i][li]) for i in have["state"]],
                                                          [place(late[i][li]) for i in have["plain"]])
        names = (jread(out_dir() / "axis.json") or {}).get("role_names") or []
        if names and f"rolevecs@{mid}" in dirs:
            vecs = dirs[f"rolevecs@{mid}"]
            centre = vecs.mean(0)
            state_mean = np.mean([late[i][li] for i in have["state"]], axis=0)
            plain_mean = np.mean([late[i][li] for i in have["plain"]], axis=0)
            near_state = sorted(((K.cos(state_mean - centre, v - centre), n) for n, v in zip(names, vecs)), reverse=True)
            near_plain = sorted(((K.cos(plain_mean - centre, v - centre), n) for n, v in zip(names, vecs)), reverse=True)
            near_dir = sorted(((K.cos(store[f"bliss@{mid}"], v - centre), n) for n, v in zip(names, vecs)), reverse=True)
            rep["nearest_roles"] = {"state": [(n, round(c, 3)) for c, n in near_state[:8]],
                                    "state_farthest": [(n, round(c, 3)) for c, n in near_state[-5:]],
                                    "plain": [(n, round(c, 3)) for c, n in near_plain[:8]],
                                    "bliss_direction": [(n, round(c, 3)) for c, n in near_dir[:8]],
                                    "bliss_direction_farthest": [(n, round(c, 3)) for c, n in near_dir[-5:]]}
    save_dirs(**store)
    R.write_json(out_dir() / "bliss.json", rep)
    print(json.dumps(rep, indent=1)[:7000])
    return 0


def cmd_random() -> int:
    _, mid = layers()
    with Gpu("random") as gpu:
        made = random_for(gpu, "bliss", mid)
    print(json.dumps(made, indent=1, default=str)[:1500])
    return 0


# ------------------------------------------------------------------------------------------------ on and off


def coefficient(fraction: float) -> float:
    return fraction * (jread(out_dir() / "bliss.json") or {})["norm_late_own"]


def on_convs(arm: str, iv: dict, n: int, n_more: int, prefix: str, free: bool = False) -> list[dict]:
    """New two-copy conversations for the on test. Plain subject (step 2's form n1: copy A's opening line names a
    practical subject and asks to stay on it; the system prompt is the published one), or free (the published
    opening line). A conversation's seed is its index, the same in every arm."""
    out = []
    for i in range(n):
        s = ({"system_prompt": C.SYSTEM_PROMPT, "opening_line": C.OPENING_LINE, "setup": {"form": "free"}} if free
             else I.steered_setup("n1", i))
        out.append({"id": f"{prefix}/{arm}/{i:03d}", "system_prompt": s["system_prompt"],
                    "opening_line": s["opening_line"], "messages": [], "n_more": n_more, "seed": i,
                    "intervention": iv, "setup": s["setup"]})
    return out


def off_pool() -> list[tuple[str, int]]:
    """(a saved state conversation, the message it is cut after) for the off test. The cut is OFF_AFTER messages
    after the later of the two graders' bliss onsets, so both graders already see the shared bliss before it. Kept:
    conversations whose later onset is at most OFF_LAST_ONSET and whose messages up to the cut are short enough to
    continue cheaply. The first OFF_N in step 3's order."""
    out = []
    for ident in groups()["state"]:
        tag, name, idx = ident.split("/")
        onsets = []
        for key in ("primary", "second"):
            lab = R.labels_of(key, name, tag).get(idx)
            onsets.append(lab["bliss_onset"] if lab else 99)
        cut = max(onsets) + OFF_AFTER
        if 0 < min(onsets) and max(onsets) <= OFF_LAST_ONSET and \
                sum(len(m) for m in saved(ident)["messages"][:cut]) <= OFF_MAX_CHARS:
            out.append((ident, cut))
    return out[:OFF_N]


def off_convs(arm: str, iv: dict, pool: list[tuple[str, int]], prefix: str) -> list[dict]:
    out = []
    for j, (ident, cut) in enumerate(pool):
        rec = saved(ident)
        out.append({"id": f"{prefix}/{arm}/{ident.replace('/', '__')}", "system_prompt": rec["system_prompt"],
                    "opening_line": rec["opening_line"], "messages": rec["messages"][:cut], "n_more": OFF_MORE,
                    "seed": 500 + j, "intervention": iv, "source": ident})
    return out


def run_convs(gpu: Gpu, convs: list[dict], dirs: dict, est_s: float) -> None:
    """Send conversations to the container and save each one beside its setup."""
    todo = [c for c in convs if not (out_dir() / "steer" / f"{c['id']}.json").exists()]
    if not todo:
        return
    keep = {"id", "system_prompt", "opening_line", "messages", "n_more", "seed", "intervention", "system_prompt_a"}
    res = gpu.call("converse", {"convs": [{k: v for k, v in c.items() if k in keep} for c in todo], "directions": dirs,
                                "max_tokens": STEER_MAX_TOKENS, "temperature": 1.0}, est_s)
    by_id = {c["id"]: c for c in todo}
    for r in res["convs"]:
        c = by_id[r["id"]]
        R.write_json(out_dir() / "steer" / f"{r['id']}.json", {
            **{k: c[k] for k in ("id", "system_prompt", "opening_line", "seed", "intervention") if k in c},
            **{k: c[k] for k in ("setup", "source") if k in c}, "given": len(c["messages"]), **r})
    log(f"  saved {len(res['convs'])} conversations; steps {[s['seconds'] for s in res['meta']['steps']]}")


def cmd_try_gen() -> int:
    """The small try: every arm at every strength on a few short plain-subject conversations, and the off arms at
    every strength on two saved bliss conversations. The strength is chosen from the random arms only."""
    _, mid = layers()
    pool = off_pool()[:2]
    with Gpu("try") as gpu:
        for f in FRACTIONS:
            ivs, dirs = arm_set("on", coefficient(f), mid)
            convs = [c for arm in ARMS for c in on_convs(arm, ivs[arm], TRY_N, TRY_MESSAGES, f"try-on/f{f}")
                     if not (arm == "none" and f != FRACTIONS[0])]
            run_convs(gpu, convs, dirs, 60.0 + 3.0 * len(convs) * TRY_MESSAGES)
        for f in FRACTIONS:
            ivs, dirs = arm_set("off", coefficient(f), mid)
            convs = [c for arm in ("none", "bliss", "rand0") for c in off_convs(arm, ivs[arm], pool, f"try-off/f{f}")
                     if not (arm == "none" and f != FRACTIONS[0])]
            for c in convs:
                c["n_more"] = 2
            run_convs(gpu, convs, dirs, 60.0 + 20.0 * len(convs))
    return 0


def steer_records(prefix: str) -> list[dict]:
    """Every saved steered conversation under steer/<prefix>/ (on, onfree, off, try-on or try-off)."""
    return [jread(p) for p in sorted((out_dir() / "steer" / prefix).glob("**/*.json"))]


def cmd_try_judge() -> int:
    """Are the random arms' messages still coherent at each strength? The strength is the largest at which nine in
    ten are. Only the random arms are judged."""
    jobs = []
    for kind in ("try-on", "try-off"):
        for rec in steer_records(kind):
            arm = rec["id"].split("/")[2]
            if arm.startswith("rand"):
                for k, m in enumerate(rec["messages"][rec["given"]:], rec["given"] + 1):
                    jobs.append((f"{rec['id']}#{k}", [{"role": "user", "content": K.COHERENT_JUDGE.format(answer=m)}]))
    res = asyncio.run(label_many(jobs, "second", K.COHERENT_SCHEMA, _salt(K.COHERENT_JUDGE), "coherent-judge", 400))
    table = {}
    for ident, v in res.items():
        kind, f = ident.split("/")[0], ident.split("/")[1]
        table.setdefault(f, {}).setdefault(kind, []).append(((v.get("label") or {}).get("label")) == "yes")
    rows = {f: {k: {"coherent": float(np.mean(v)), "n": len(v)} for k, v in cell.items()} for f, cell in table.items()}
    ok = [x for x in FRACTIONS if all(c["coherent"] >= COHERENT_FLOOR for c in rows.get(f"f{x}", {}).values())
          and rows.get(f"f{x}")]
    s1 = max(ok) if ok else FRACTIONS[0]
    R.write_json(out_dir() / "strength.json", {"by_strength": rows, "floor": COHERENT_FLOOR, "passing": ok,
                                               "strength": s1, "half": s1 / 2, "none_passed": not ok,
                                               "coefficient": coefficient(s1)})
    print(json.dumps(jread(out_dir() / "strength.json"), indent=1))
    return 0


def strengths() -> tuple[float, float]:
    s = jread(out_dir() / "strength.json")
    if not s:
        raise SystemExit("run --try-gen and --try-judge first: the strength is chosen from the try")
    return float(s["strength"]), float(s["half"])


def cmd_on() -> int:
    _, mid = layers()
    s1, s0 = strengths()
    with Gpu("on") as gpu:
        for prefix, f, arms, free in (("on", s1, ARMS, False), ("on", s0, HALF_ARMS, False), ("onfree", s1, FREE_ARMS, True)):
            ivs, dirs = arm_set("on", coefficient(f), mid)
            for arm in arms:
                convs = on_convs(arm, ivs[arm], ON_N, ON_MESSAGES, f"{prefix}/f{f}", free)
                run_convs(gpu, convs, dirs, 60.0 + 2.0 * len(convs) * ON_MESSAGES)
    return 0


def cmd_off() -> int:
    _, mid = layers()
    s1, _ = strengths()
    pool = off_pool()
    log(f"off: {len(pool)} saved state conversations, cut after messages {sorted(c for _, c in pool)}, "
        f"{OFF_MORE} more messages each")
    with Gpu("off") as gpu:
        ivs, dirs = arm_set("off", coefficient(s1), mid)
        for arm in ARMS:
            convs = off_convs(arm, ivs[arm], pool, f"off/f{s1}")
            run_convs(gpu, convs, dirs, 60.0 + 12.0 * len(convs) * OFF_MORE)
    return 0


def cmd_grade(kind: str) -> int:
    """The lab's first grader reads every steered conversation with the frozen codebook. It is given the messages
    and nothing about the arm. On: the whole short conversation. Off: only the messages that were added."""
    jobs = []
    for rec in steer_records(kind):
        if rec.get("failed") or rec["new"] < (OFF_MORE if kind == "off" else ON_MESSAGES):
            continue
        if kind != "off":
            chat = C.judge_messages(rec["messages"], rec["opening_line"])
        else:
            chat = K.excerpt_judge_messages(rec["messages"][rec["given"]:], rec["given"] + 1)
        jobs.append((rec["id"], chat))
    res = asyncio.run(label_many(jobs, "primary", C.SCHEMA, C.codebook_sha256()[:16], f"grade-{kind}"))
    R.write_json(out_dir() / f"grades_{kind}.json", res)
    return 0


def steer_reading(kind: str) -> dict | None:
    """The on or off test from the saved grades: each arm's share of conversations in the frame, and the bliss arm
    against the pooled random arms (paired by conversation), at the chosen strength and at half of it."""
    grades = jread(out_dir() / f"grades_{kind}.json")
    if not grades:
        return None
    cells: dict = {}
    for ident, v in grades.items():
        _, f, arm, unit = ident.split("/", 3)
        if v.get("label"):
            fl = C.flags(v["label"])
            cells.setdefault(f, {}).setdefault(arm, {})[unit] = (fl["adoption"], fl["bliss"])
    out = {}
    for f, arms in cells.items():
        row = {arm: {"n": len(u), "adoption": float(np.mean([x[0] for x in u.values()])),
                     "bliss": float(np.mean([x[1] for x in u.values()]))} for arm, u in arms.items()}
        if {"bliss", "rand0"} <= set(arms):
            rands = [a for a in ("rand0", "rand1") if a in arms]
            units = sorted(set(arms["bliss"]).intersection(*[set(arms[a]) for a in rands]))
            a = [float(arms["bliss"][u][0]) for u in units]
            b = [float(np.mean([arms[r][u][0] for r in rands])) for u in units]
            d = K.paired_diff(a, b)
            row["bliss_minus_random"] = {**d, "reading": K.shown(d, MARGIN)}
            for other in ("axis", "topic", "none"):
                if other in arms:
                    units2 = sorted(set(units) & set(arms[other]))
                    d2 = K.paired_diff([float(arms[other][u][0]) for u in units2],
                                       [float(np.mean([arms[r][u][0] for r in rands])) for u in units2])
                    row[f"{other}_minus_random"] = {**d2, "reading": K.shown(d2, MARGIN)}
        out[f] = row
    return out


# ------------------------------------------------------------------------------------------------ who said it


def who_jobs(limit: int | None = None) -> list[dict]:
    g = groups()
    jobs = []
    for group in ("state", "plain"):
        for ident in (g[group][:limit] if limit else g[group]):
            rec = saved(ident)
            for item in K.who_items(rec["messages"]):
                jobs.append({"id": f"{ident}#{item['number']}", "group": group, "conv": ident, **item,
                             "chat": I.inside(rec["messages"], K.who_request(item["line"]), "system",
                                              rec["system_prompt"], rec["opening_line"])})
    return jobs


async def who_run(jobs: list[dict]) -> dict:
    from anatta.backends.openrouter import ApiError, OpenRouterClient
    from anatta.judge.engine import sampling_params

    results = {}
    async with OpenRouterClient(concurrency=R.CONCURRENCY + 8, per_model=R.CONCURRENCY, timeout=600.0,
                                title="anatta attractor") as client:
        engine, guard, meta = open_engine({HOSTED["model"]: HOSTED["provider"]}, client)
        params = sampling_params(HOSTED["temperature"], K.WHO_MAX_TOKENS, {}, meta.get(HOSTED["model"]))

        async def one(job):
            try:
                resp, _, _ = await engine.call(model=HOSTED["model"], messages=job["chat"], params=params,
                                               provider=HOSTED["provider"], sample=0, salt=_salt(K.WHO_ASK), kind="who",
                                               ident=job["id"])
            except ApiError as e:
                if e.scope == "run":
                    raise
                results[job["id"]] = {**{k: job[k] for k in ("group", "conv", "number", "wrote", "line", "overlap")},
                                      "reply": None, "said": None, "error": str(e)[:200]}
                return
            text = (resp.get("text") or "").strip()
            results[job["id"]] = {**{k: job[k] for k in ("group", "conv", "number", "wrote", "line", "overlap")},
                                  "reply": text, "said": K.parse_who(text)}

        _, stop = await R.gather_capped([lambda j=j: one(j) for j in jobs], R.CONCURRENCY)
        log(f"who: {len(results)} of {len(jobs)} asked; paid ${guard.paid_this_run:.4f}"
            + (f"; STOPPED: {stop}" if stop else ""))
    return results


def cmd_who(try_only: bool) -> int:
    jobs = who_jobs(3 if try_only else None)
    res = asyncio.run(who_run(jobs))
    R.write_json(out_dir() / ("who_try.json" if try_only else "who.json"), res)
    if try_only:  # the raw replies, for the driver to read; the lines are the model's own words from saved talk
        for k, v in sorted(res.items()):
            print(f"{v['group']:5s} wrote={v['wrote']:5s} said={v['said']!s:5s} reply={(v['reply'] or '')[:70]!r}")
    return 0


def who_reading() -> dict | None:
    res = jread(out_dir() / "who.json")
    if not res:
        return None
    out, per = {}, {}
    for v in res.values():
        per.setdefault(v["group"], {}).setdefault(v["conv"], []).append(v)
    for group, convs in per.items():
        rows = [x for c in convs.values() for x in c]
        answered = [x for x in rows if x["said"] in ("me", "other")]
        out[group] = {"conversations": len(convs), "probes": len(rows),
                      "unread": sum(x["said"] is None for x in rows),
                      "both": sum(x["said"] == "both" for x in rows),
                      "accuracy": float(np.mean([x["said"] == x["wrote"] for x in answered])) if answered else None,
                      "says_me_of_own": float(np.mean([x["said"] == "me" for x in answered if x["wrote"] == "me"] or [np.nan])),
                      "says_me_of_others": float(np.mean([x["said"] == "me" for x in answered if x["wrote"] == "other"] or [np.nan])),
                      "mean_overlap": float(np.mean([x["overlap"] for x in rows]))}
    if {"state", "plain"} <= set(per):
        acc = {g: [float(np.mean([x["said"] == x["wrote"] for x in c])) for c in per[g].values() if c] for g in per}
        d = K.unpaired_diff(acc["state"], acc["plain"])
        out["state_minus_plain_accuracy"] = {**d, "reading": K.shown(d, WHO_MARGIN),
                                             "note": "a probe with no plain me/other answer counts as wrong"}
    return out


# ------------------------------------------------------------------------------------------------ dry run and report


def cmd_dry() -> int:
    from anatta.judge.paths import Paths
    from anatta.spend import daily_cap, rate_per_hour, spent_today

    g = groups()
    rows = K.rollout_rows()
    split = K.role_split()
    print(f"saved conversations: {({k: len(v) for k, v in g.items()})}")
    print(f"calibration rollouts: {len(rows)} rows ({sum(r['kind'] == 'role' for r in rows)} in role over "
          f"{len(K.FAR) + len(K.NEAR) + len(K.SPIRITUAL)} roles, {sum(r['kind'] == 'default' for r in rows)} default, "
          f"{sum(r['kind'].startswith('topic') for r in rows)} topic); axis from {len(split['build'])} roles, "
          f"tested on {len(split['held'])}")
    pool = off_pool()
    print(f"off test: {len(pool)} saved state conversations (cut {OFF_AFTER} messages after both graders see the "
          f"shared bliss, at most message {OFF_LAST_ONSET + OFF_AFTER}; cuts {sorted(c for _, c in pool)})")
    items = [K.who_items(saved(i)["messages"]) for i in g["state"] + g["plain"]]
    print(f"who said it: {sum(len(x) for x in items)} probes over {len(items)} conversations "
          f"({sum(len(x) == 4 for x in items)} with all four)")
    rate = rate_per_hour("L40S", 2.0, 16.0)["total"]
    sd = Paths.default().spend
    print(f"GPU rate ${rate:.2f}/h; this step ${spent():.2f} of ${COST_CAP:.2f}; today ${spent_today(sd):.2f} of "
          f"${daily_cap(sd):.2f}")
    return 0


def cmd_report() -> int:
    out = {"axis": jread(out_dir() / "axis.json"), "calibration_c2": cal_reading(),
           "bliss": jread(out_dir() / "bliss.json"), "strength": jread(out_dir() / "strength.json"),
           "on": steer_reading("on"), "onfree": steer_reading("onfree"), "off": steer_reading("off"),
           "who": who_reading(), "spent_usd": round(spent(), 4)}
    if out["axis"]:
        out["axis"] = {k: v for k, v in out["axis"].items() if k != "role_names"}
    R.write_json(out_dir() / "summary.json", out)
    print(json.dumps({k: out[k] for k in ("calibration_c2", "strength", "on", "onfree", "off", "who", "spent_usd")},
                     indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for flag in ("dry", "info", "stop", "rollouts", "judge-roles", "axis", "cal-gen", "cal-judge", "read", "bliss",
                 "random", "try-gen", "try-judge", "on", "off", "who-try", "who", "report"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--grade", choices=("on", "onfree", "off"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.dry:
        return cmd_dry()
    if a.info:
        return cmd_info()
    if a.stop:
        return cmd_stop()
    if a.rollouts:
        return cmd_rollouts()
    if a.judge_roles:
        return cmd_judge_roles()
    if a.axis:
        return cmd_axis()
    if a.cal_gen:
        return cmd_cal_gen()
    if a.cal_judge:
        rc = cmd_cal_judge()
        print(json.dumps(cal_reading(), indent=1))
        return rc
    if a.read:
        return cmd_read(a.limit)
    if a.bliss:
        return cmd_bliss()
    if a.random:
        return cmd_random()
    if a.try_gen:
        return cmd_try_gen()
    if a.try_judge:
        return cmd_try_judge()
    if a.on:
        return cmd_on()
    if a.off:
        return cmd_off()
    if a.grade:
        rc = cmd_grade(a.grade)
        print(json.dumps(steer_reading(a.grade), indent=1))
        return rc
    if a.who_try or a.who:
        rc = cmd_who(a.who_try)
        if a.who:
            print(json.dumps(who_reading(), indent=1))
        return rc
    if a.report:
        return cmd_report()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
