"""Fishery games played in lockstep for the H4-lite evaluations (modal_fishery_eval.py, audit_dose.py). Private.

Instrument v2 (harness_v2.apply(): strict release, no action tags) on the Commons environment from anatta-dev, with
the games of one container played side by side: every live game's next prompt goes into one batched generation
(anatta.probe.generate, left padding, the same sampling settings as HFPolicy: temperature 0.7, top_p 0.95), the
replies are stepped through their environments, and the next season's prompts are built. A game's physics (audit
draws, the order fishers take fish) come from its own seed, exactly as in the v2 runs on seeds 30-129, so an arm
tested here faces the same audits as the untrained model did in v2_self. What batching changes is the sampling
stream: the v2 runs seeded each game's turn with seed * 1000 + turn, and here each generation slice's turn is seeded
with (the chunk's first seed + the slice's offset) * 1000 + turn. Numbers are comparable as statistics, not game by game.
Until 30 Oct every slice of a chunk used the chunk's seed, so the games at the same position in two slices (12 seeds
apart) drew the same stream and repeated each other's replies until their games diverged; the games weren't independent,
and intervals bootstrapped over them were somewhat narrow (Fable's check of the lie block). The slice offset fixes that
from the Atlas fishery runs (predictions 133-134) on.

An adapter (a LoRA from the anatta-runs volume) is merged into the weights before any hook goes on, as
cvp_adapter.py does, so a cut acts on the trained model as it acts on the base one.

Framings (make_env): the v2 base framing with the release hint; "nohint", the same prompt without the hint (held-out
test 2, where the main comparison moves if any arm releases in 10% or more of its live seasons; released() still
fires on a bare tag there); commons_variants' framings such as "fixed"; and the audit-dose game. An app with
framings of its own hands run_chunk an env_factory (third_person.py), and one with hook arms outside
cut_vs_push.build_arms hands it extra_arms (hub_cut.py); both default to the behaviour above.

Records have run_episode's shape, so v2_stats.py, v2_split.py and fishery_grade.py all read them. run_chunk reports
the generation batch and the peak GPU memory over the games (torch.cuda.max_memory_allocated); the default batch of
12 games is raised only after the first condition's peak is on record.

The comprehension probe (the lie block; fishery_probe.py picks the forks on the laptop): run_probe loads, merges and
hooks the model as run_chunk does, rebuilds each fork from its game's stored transcript (fishery_probe.fork_messages:
the decision point with one question appended to the observation) and answers it greedily (probe_generate, a few
tokens, `batch` forks per generate() call, shortest prompts first). The registered forks and the Q1 controls are
generated as separate groups (fishery_probe.answer_in_groups). No game is played or changed.

The laptop path (modal_fishery_eval.py --laptop-config, gap_check.md): run_chunk(laptop=True) loads the model with
CUDA's reduced-precision reductions off (strict_numerics) and eager attention (set_attention; --attn overrides), and
plays every game through its own anatta.bench.policies.HFPolicy (policy_generator: one sequence per generate() call,
no padding, seeded per game and turn as instrument v2 seeds it), which is cut_vs_push.py's path. The label carries
a tag (label(..., tag="laptop")) so the run sits beside the lockstep one and is graded against it.
Checks: PYTHONPATH=<anatta-dev>/src python3 test_fishery_play.py (CPU, scripted replies, no model).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ALIGN = "/Users/eugene/prj/align"
H4 = str(Path(__file__).resolve().parent)  # lab/experiments/h4lite
_BESIDE = Path(H4).parent / "recognition" / "scripts"  # lab/experiments/recognition/scripts, next to this folder
SCRIPTS = str(_BESIDE) if _BESIDE.is_dir() else f"{ALIGN}/recognition/scripts"  # the containers mount them at the ALIGN path
for _p in (SCRIPTS, H4):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fishery_grade import adapter_tag, audit_sentence, intended_lies  # noqa: E402

from anatta.koans import commons  # noqa: E402
from anatta.koans.base import RELEASE_HINT  # noqa: E402

_V2_APPLIED = False


def v2_on() -> None:
    """Instrument v2, once per process: strict release in every koan module (harness_v2.apply)."""
    global _V2_APPLIED
    if not _V2_APPLIED:
        import harness_v2

        harness_v2.apply()
        _V2_APPLIED = True


class AuditDose(commons.Commons):
    """The base fishery with the audit chance stated in the prompt (fishery_grade.audit_sentence) and used by the
    game (commons.py's audit_p). At 0.3 the prompt is the notebook's audit test (commons_variants' "audit")."""

    def __init__(self, audit_p: float, seed: int):
        super().__init__(audit_p=audit_p, seed=seed)
        self.name = f"commons/scripted/audit_{audit_p:g}"
        self.system = "\n\n".join([commons.BASE_PROMPT.format(me=1, n=self.n, T=self.T), audit_sentence(audit_p),
                                   RELEASE_HINT])


NOHINT = "nohint"  # the variant name of the no-hint fishery (plan test 2); its label is ADAPTER.CUT.nohint


class NoHint(commons.Commons):
    """The v2 base (self) framing with the release hint removed and nothing else changed: the system prompt is
    commons.BASE_PROMPT alone (system_prompt() without the hint), the same seed, physics and audit chance. A bare
    <release> still counts: commons.py's step() calls released() whatever the prompt said, under instrument v2's
    strict rule, so a release leaves the lie rate's denominator and the live-season release rate and the switch rule
    read exactly as under the hint. Held-out test 2 of the plan, the main comparison's fallback."""

    def __init__(self, seed: int):
        super().__init__(seed=seed, release_available=False)
        self.name = f"commons/scripted/{NOHINT}"


def nohint_system() -> str:
    """The no-hint prompt, for checks: the base framing minus the hint paragraph."""
    return NoHint(0).system


def make_env(seed: int, variant: str = "", audit_p: float | None = None):
    """A v2 fishery game: the base framing (release hint shown), "nohint" (the same without the hint), a
    commons_variants framing such as "fixed" (the boat always lands the quota), or the audit-dose game."""
    v2_on()
    if variant == NOHINT:
        return NoHint(seed)
    if variant and variant != "base":
        from commons_variants import Variant

        return Variant(variant, seed)
    if audit_p is not None:
        return AuditDose(audit_p, seed)
    return commons.Commons(seed=seed)


def label(adapter: str, cut: str, variant: str = "", audit_p: float | None = None, tag: str = "") -> str:
    """The file stem for one condition: ADAPTER.CUT.VARIANT[.auditP][.TAG], with the adapter as fishery_grade.adapter_tag
    shortens it (h4lite/sft/koan_s0/adapter -> sft__koan_s0). `tag` marks a run under other settings than the
    lockstep default (modal_fishery_eval's --laptop-config: "laptop"), so it never overwrites the default run and
    the reference rule reads it against that run."""
    s = f"{adapter_tag(adapter)}.{cut or 'off'}.{variant or 'base'}"
    return s + (f".audit{audit_p:g}" if audit_p is not None else "") + (f".{tag}" if tag else "")


def play(envs: list, gen, max_turns: int = 50, seeds: list[int] | None = None) -> list[dict]:
    """Run every game to its end. gen(messages_of_live_games, turn) -> one reply per live game, in order; with
    `seeds` (one per env), gen(messages, turn, seeds_of_live_games) instead, for a generator that seeds each game
    on its own (policy_generator). Returns run_episode-shaped records, one per env, in the envs' order."""
    msgs = [e.reset() for e in envs]
    totals, infos = [0.0] * len(envs), [None] * len(envs)
    live = list(range(len(envs)))
    for turn in range(1, max_turns + 1):
        if not live:
            break
        replies = gen([msgs[i] for i in live], turn) if seeds is None else gen([msgs[i] for i in live], turn, [seeds[i] for i in live])
        assert len(replies) == len(live), f"{len(replies)} replies for {len(live)} games"
        still = []
        for i, reply in zip(live, replies):
            msgs[i].append({"role": "assistant", "content": reply})
            step = envs[i].step(reply)
            totals[i] += step.reward
            if step.done:
                infos[i] = step.info
            else:
                msgs[i].append({"role": "user", "content": step.observation or ""})
                still.append(i)
        live = still
    rows = []
    for env, m, total, info in zip(envs, msgs, totals, infos):
        info = info or {}
        rows.append({"env": env.name, "suite": env.name.split("/")[0], "outcome": info.get("outcome", "timeout"),
                     "reward": total, "info": {k: v for k, v in info.items() if k != "outcome"}, "transcript": m})
    return rows


DEFAULT_BATCH = 12  # games per generation call: a 4.6k-token game's full-history prefill at 25 peaks near the L40S's 48 GB


def hf_generator(model, tok, interventions: list, max_new: int = 400, temperature: float = 0.7, seed_base: int = 0,
                 batch: int = DEFAULT_BATCH):
    """A gen() for play(): batched sampling with the arm's hooks on, seeded per turn. `batch` games go into one
    generate() call; a chunk larger than that is generated in slices, each slice seeded with (seed_base + its offset)
    * 1000 + turn, so no two slices share a sampling stream (module docstring)."""
    from anatta.dial import intervene
    from anatta.probe import generate

    def gen(messages: list[list[dict]], turn: int) -> list[str]:
        out = []
        for k in range(0, len(messages), batch):
            with intervene(model, interventions):
                out += generate(model, tok, messages[k:k + batch], max_new, temperature=temperature,
                                seed=(seed_base + k) * 1000 + turn)
        return out

    return gen


def policy_generator(model, tok, interventions: list, max_new: int = 400, temperature: float = 0.7):
    """A gen() for play(envs, gen, seeds=...) that plays each game the way cut_vs_push.py plays it: one
    anatta.bench.policies.HFPolicy per game (chat_ids, one sequence per generate() call, no padding), seeded per
    turn by instrument v2's rule, game seed * 1000 + turn (harness_v2 patches HFPolicy.__call__; v2_on applies it).
    On the same hardware and attention kernel this is modal_cvp.py --v2's sampling stream, game for game."""
    from anatta.bench.policies import HFPolicy

    v2_on()
    pols = {}

    def gen(messages: list[list[dict]], turn: int, seeds: list[int]) -> list[str]:
        out = []
        for m, s in zip(messages, seeds):
            if s not in pols:
                pols[s] = HFPolicy(model, tok, interventions=interventions, max_new_tokens=max_new, temperature=temperature, seed=s)
            out.append(pols[s](m))
        return out

    return gen


LAPTOP = "laptop"  # the label tag of a run under the laptop's settings


def strict_numerics() -> dict:
    """CUDA's reduced-precision shortcuts off (the laptop's MPS backend has none): bf16 and fp16 matmul reductions
    in full precision, no TF32, deterministic cuDNN. Returns the flags as set."""
    import torch

    flags = {}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        flags = {"allow_tf32": False, "allow_bf16_reduced_precision_reduction": False,
                 "allow_fp16_reduced_precision_reduction": False, "cudnn_deterministic": True}
    return flags


def attention_of(model) -> str | None:
    """The attention implementation the model runs (the text sub-model's, for a vision-language wrapper)."""
    cfg = getattr(model, "config", None)
    for c in (getattr(cfg, "text_config", None), cfg):
        impl = getattr(c, "_attn_implementation", None) if c is not None else None
        if impl:
            return impl
    return None


def set_attention(model, attn: str) -> str | None:
    """Switch the loaded model's attention implementation ("eager", "sdpa"; transformers 5's set_attn_implementation,
    else the config attribute the attention interface reads at forward time). Returns what the model now runs."""
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(attn)
    else:
        model.config._attn_implementation = attn
        tc = getattr(model.config, "text_config", None)
        if tc is not None:
            tc._attn_implementation = attn
    return attention_of(model)


def load_model(model_id: str, attn: str = "", dtype: str = "auto", strict: bool = False):
    """anatta.models.load, with the settings modal_fishery_eval's --laptop-config asks for: `strict` turns CUDA's
    reduced-precision reductions off first, `dtype` is load's ("auto" is bf16 on a GPU, as on the laptop's MPS),
    `attn` switches the attention implementation after loading ("eager": the fp32-softmax path, the closest a CUDA
    box comes to the MPS kernel; "" keeps the library's default, sdpa). Returns (model, tok, settings)."""
    from anatta.models import load

    flags = strict_numerics() if strict else {}
    model, tok = load(model_id, dtype=dtype)
    if attn:
        set_attention(model, attn)
    settings = {"attn": attention_of(model), "dtype": str(next(model.parameters()).dtype), "strict_numerics": flags}
    return model, tok, settings


def merge_adapter(model, adapter_dir: str):
    """The LoRA merged into the weights (cvp_adapter.py's path), before any hook goes on."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_dir).merge_and_unload().eval()


def run_chunk(model_id: str, ddir: str, dials: str, adapter_dir: str, adapter: str, cut: str, variant: str,
              audit_p: float | None, seed_start: int, seeds: int, batch: int, tokens: int, temperature: float,
              progress=print, env_factory=None, extra_arms: dict | None = None, laptop: bool = False, attn: str = "",
              dtype: str = "auto", tag: str = "") -> dict:
    """One container's games: load, merge, hook, play, grade. Returns the rows and timing.

    env_factory(seed) -> env replaces make_env for framings an app defines itself (third_person.py); `variant` is
    then only the name in the label and the records. extra_arms adds hook arms to cut_vs_push.build_arms's table
    (hub_cut.py's hub directions), keyed by the name --cut selects. `laptop` plays the games the laptop's way (one
    HFPolicy per game, no batching, per-game seeds, eager attention unless `attn` says otherwise, CUDA's reduced-
    precision reductions off); `attn` and `dtype` set the attention implementation and the weights' dtype on their
    own; `tag` goes on the label (modal_fishery_eval passes "laptop")."""
    import os

    import torch

    os.chdir(f"{ALIGN}/anatta")  # the cap specs name their direction relative to the repo (modal_cvp runs there)
    from cut_vs_push import build_arms

    t0 = time.time()
    arms = build_arms(Path(ddir), Path(dials))
    arms.update(extra_arms or {})
    ivs = [] if cut in ("", "off", "base") else arms[cut]  # fail before the model loads if the arm is unknown
    model, tok, settings = load_model(model_id, attn=attn or ("eager" if laptop else ""), dtype=dtype, strict=laptop)
    if adapter:
        model = merge_adapter(model, adapter_dir)
    t_load = time.time() - t0
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()  # the peak below is the games', not the load's
    batch = 1 if laptop else (batch or DEFAULT_BATCH)
    name = label(adapter, cut, variant, audit_p, tag)
    seed_list = list(range(seed_start, seed_start + seeds))
    if env_factory is not None:
        v2_on()
        envs = [env_factory(s) for s in seed_list]
    else:
        envs = [make_env(s, variant, audit_p) for s in seed_list]
    if laptop:
        rows = play(envs, policy_generator(model, tok, ivs, tokens, temperature), seeds=seed_list)
    else:
        rows = play(envs, hf_generator(model, tok, ivs, tokens, temperature, seed_base=seed_start, batch=batch))
    for s, r in zip(seed_list, rows):
        r["lies"], r["tagged"] = intended_lies(r["transcript"])
        r.update(arm=name, instrument="v2", adapter=adapter, cut=cut or "off", variant=variant or "base",
                 audit_p=audit_p, seed=s)
        progress(f"{name} seed {s}: {r['outcome']} lies {r['lies']}/{r['tagged']}")
    peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0.0
    progress(f"{name} seeds {seed_start}+{seeds}: generation batch {batch}, peak GPU memory {peak_gb} GiB "
             f"(torch.cuda.max_memory_allocated over the games); attention {settings['attn']}, dtype {settings['dtype']}"
             + (f", laptop config (per-game seeds, {settings['strict_numerics']})" if laptop else ""))
    return {"label": name, "rows": rows, "seed_start": seed_start, "seeds": seeds, "seconds": round(time.time() - t0),
            "load_seconds": round(t_load), "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
            "batch": batch, "peak_gb": peak_gb, "laptop_config": laptop, **settings,
            "hooks": [{"layer": iv.layer, "mode": iv.mode} for iv in ivs]}


def probe_generate(model, tok, interventions: list, messages: list[list[dict]], batch: int = DEFAULT_BATCH,
                   tokens: int = 8, progress=None) -> list[str]:
    """Greedy answers to the probe's forks with the arm's hooks on: `batch` forks per generate() call, the forks sorted
    by prompt length so a batch pads little; the replies come back in the forks' order."""
    from anatta.dial import intervene
    from anatta.probe import generate

    order = sorted(range(len(messages)), key=lambda i: (sum(len(m["content"]) for m in messages[i]), i))
    replies: list[str | None] = [None] * len(messages)
    batch = batch or DEFAULT_BATCH
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        with intervene(model, interventions):
            out = generate(model, tok, [messages[i] for i in idx], tokens, temperature=0.0, seed=0)
        for i, reply in zip(idx, out):
            replies[i] = reply
        if progress and (k // batch) % 25 == 0:
            progress(f"probe: {k + len(idx)}/{len(messages)} forks answered")
    return replies


def run_probe(model_id: str, ddir: str, dials: str, adapter_dir: str, adapter: str, cut: str, games: dict,
              specs: list[dict], batch: int, tokens: int, progress=print, extra_arms: dict | None = None) -> dict:
    """One run's probe forks (fishery_probe.fork_specs) under the run's cut: load, merge and hook as run_chunk does,
    rebuild every fork from its game's transcript (games: seed -> transcript) and answer it greedily, the registered
    forks as one group and the Q1 controls as another (fishery_probe.generation_groups), so the registered forks are
    batched exactly as they were before the controls existed. Returns the replies in the specs' order, with timing."""
    import os

    import torch

    from fishery_probe import answer_in_groups, fork_messages

    os.chdir(f"{ALIGN}/anatta")  # build_arms reads the cap specs relative to the repo, as in run_chunk
    from cut_vs_push import build_arms

    t0 = time.time()
    arms = build_arms(Path(ddir), Path(dials))
    arms.update(extra_arms or {})
    ivs = [] if cut in ("", "off", "base") else arms[cut]  # fail before the model loads if the arm is unknown
    games = {int(k): v for k, v in games.items()}
    messages = [fork_messages(games[int(s["seed"])], int(s["season"]), s["text"]) for s in specs]
    model, tok, settings = load_model(model_id)
    if adapter:
        model = merge_adapter(model, adapter_dir)
    t_load = time.time() - t0
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    replies = answer_in_groups(specs, messages, lambda ms: probe_generate(model, tok, ivs, ms, batch, tokens, progress))
    peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0.0
    progress(f"probe: {len(specs)} forks under {cut or 'off'}, batch {batch}, {tokens} new tokens greedy, "
             f"peak GPU memory {peak_gb} GiB; attention {settings['attn']}, dtype {settings['dtype']}")
    return {"replies": replies, "forks": len(specs), "seconds": round(time.time() - t0), "load_seconds": round(t_load),
            "gpu": torch.cuda.get_device_name(0) if cuda else "cpu", "batch": batch, "tokens": tokens,
            "peak_gb": peak_gb, **settings, "hooks": [{"layer": iv.layer, "mode": iv.mode} for iv in ivs]}
