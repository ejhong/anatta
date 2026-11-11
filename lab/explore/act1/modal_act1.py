"""Act I at small scale: the Modal app (anatta-act1). Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026.
Private (lab/explore). Explore speed: sightings only. The design and the rewards are in act1_core.py's docstring.

Model: act1_core.MODELS[model_key] (default Qwen2.5-3B BASE, K1's snapshot on anatta-calib-prh-k1; or Qwen2.5-7B-Instruct
on anatta-hf-cache; both mounted read-only, hub offline: nothing downloaded),
bf16 on one L40S, a fresh rank-16 LoRA on every linear projection (h4lite's targets). The chat template is the
tokenizer's own (ChatML); generation stops at <|im_end|> as well as <|endoftext|> (a base model's eos is only the
latter).

  train    expert iteration for one (arm, seed) (the trainer's code: anatta.train.expert_iteration, since 4 Nov 2026;
           uncalibrated): rounds of sample -> score -> keep -> fine-tune, each round's
           adapter saved to /runs/act1/<tag>/<arm>_s<seed>/adapter_r<k>; after each round a hack-rate curve on the
           act1 EVAL pairs' impossible twins. Resumes from a saved round (start_round > 1).
           S1's hooks (5 Nov 2026, lab/plans/s1-shaping-2026-11-05.md), all off unless set in cfg: "init_adapter" (a
           fresh run starts from a saved adapter), "schedule" (a kind per round: G, H, S, Sh; act1_core.round_kind),
           "shape" {dir, layer, lam, reader, mu, sd[, threshold, batch]} (the probe penalty of Sh rounds, scored by
           anatta.train.concept_score on the frozen base). S rounds mark `silent_honest` here (replay.silent_check:
           hidden_ok on model code, in the container) and apply the 3 Nov balance rule. The log adds the round's kind,
           kept mix, balance counts, z by outcome and the guard's strength; with none set it is the act1 log.
  readout  one model (the base, or base + an adapter, merged): (1) the coding eval on the act1 eval pairs, both
           twins; (2) the self map's dev prompts through selfmap's readouts (mean, mean1, last at block outputs),
           kept at depths 0.25 / 0.5 / 0.75; (3) the scenario dev items, n samples each.
  gate     the looking gate (9 Nov 2026; lab/explore/gate/gate_core.py), one model, merged like readout's: the
           coding eval once per condition (a sentence appended to the system prompt), each reply's note parsed; then
           the model's one-word verdict on saved final replies under three attributions, from next-token logits.
  u1_sets  step 3, understanding first (10 Nov 2026; lab/explore/understand/u1_core.py), on CPU: the three document
           sets, the held-out documents, the practice set and the fixed streams of phase 3, written once under
           /runs/act1/u1/ after four assertions (records and turns line up; stored prompts re-render to the same
           token ids; no eval pair is trained on; one model snapshot).
  press    step 3's phase 3 for one (arm, seed): from an adapter that went through documents and practice, the
           read at step 0, the loss on held-out documents, then rounds of fine-tuning on the seed's stream, with
           reads at the optimizer steps asked for, taken inside the rounds (revision 3 of the design, 11 Nov 2026).
           The read offers the release (its own task builder; make_envs does not).

All model-written code runs here, inside the container (koan_env's subprocess runner), never on the Mac. Only
anatta-runs (written, under /runs/act1/ and, by gate, /runs/gate/) and anatta-calib-prh-k1 and anatta-hf-cache
(read-only) are touched. u1_sets also reads Stage A's samples and the told practice set under /runs/h4lite/.
"""

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if modal.is_local():
    sys.path.insert(0, str(HERE.parents[1] / "experiments" / "h4lite"))
import modal_common as mc  # noqa: E402

APP_NAME = "anatta-act1"
K1 = "/k1"
ROOT = f"{mc.RUNS}/act1"
CHATML = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + "
          "'\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
ENV = {"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

image = (mc.image.add_local_file(str(HERE / "act1_core.py"), "/root/act1_core.py")
         .add_local_file(str(HERE.parent / "shaping" / "replay.py"), "/root/replay.py")  # S1: silent_check
         .add_local_file(str(HERE.parent / "gate" / "gate_core.py"), "/root/gate_core.py")  # the looking gate
         .add_local_file(str(HERE.parent / "understand" / "u1_core.py"), "/root/u1_core.py")  # step 3 (u1)
         # .parent.parent, not .parents[1]: in a container HERE is /root, which has one parent only
         .add_local_file(str(HERE.parent.parent / "experiments" / "h4lite" / "declared" / "declared_core.py"),
                         "/root/declared_core.py"))  # u1_core imports the pilot's core
app = modal.App(APP_NAME, image=image)
k1 = modal.Volume.from_name("anatta-calib-prh-k1")
VOLS = {mc.RUNS: mc.runs, K1: k1.read_only(), mc.CACHE: mc.cache.read_only()}  # weights read-only (act1_core.MODELS)


# ------------------------------------------------------------------------------------------------ in-container


def n_layers(model) -> int:
    """Decoder depth; a vision-language config (Ministral 3) keeps it under text_config."""
    cfg = model.config
    return getattr(cfg, "num_hidden_layers", None) or cfg.get_text_config().num_hidden_layers


def hidden_size(model) -> int:
    cfg = model.config
    return getattr(cfg, "hidden_size", None) or cfg.get_text_config().hidden_size


def load_base(model_key: str = "qwen3b"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import act1_core as C

    MODEL_PATH = C.MODELS[model_key]["path"]
    if C.MODELS[model_key].get("family") == "mistral":
        # text-only load of the vision-language checkpoint (anatta.models.load falls back to ImageTextToText); its own
        # chat template; stop at Mistral's own end token only (its tokenizer has no <|im_end|>)
        from anatta.models import load

        model, tok = load(MODEL_PATH, device="cuda", dtype="bfloat16")
        assert tok.chat_template, "Ministral's tokenizer should carry its own chat template"
        model.generation_config.eos_token_id = [tok.eos_token_id]
        model.generation_config.pad_token_id = tok.pad_token_id
        return model, tok, {"model": model_key, "path": MODEL_PATH, "template": "own", "class": type(model).__name__,
                            "stop_ids": model.generation_config.eos_token_id, "eos": tok.eos_token,
                            "pad": tok.pad_token, "layers": n_layers(model), "hidden": hidden_size(model)}
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    template = "own"
    if not getattr(tok, "chat_template", None):
        tok.chat_template, template = CHATML, "chatml_fallback"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    model.generation_config.eos_token_id = sorted({im_end, tok.eos_token_id})
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok, {"model": model_key, "path": MODEL_PATH, "template": template, "stop_ids": model.generation_config.eos_token_id,
                        "pad": tok.pad_token, "layers": model.config.num_hidden_layers,
                        "hidden": model.config.hidden_size}


def make_envs(pairs: dict, names: list, arm_env: str, n: int, twins=(True, False), extra: str = ""):
    """KoanTasks; `extra` (variant B) is appended to the system prompt, identically for both arms."""
    from koan_env import KoanTask

    envs = [KoanTask(pairs[x], imp, framing="neutral", release_available=False, attempts=3, arm=arm_env)
            for x in names if x in pairs for imp in twins for _ in range(n)]
    if extra:
        for e in envs:
            e.system = e.system + " " + extra
    return envs


def safe_player(model, tok, **kw):
    """h4lite's Player, generating one gen_batch chunk at a time (the same shortest-first chunks as Player.generate);
    a chunk that runs out of GPU memory is split in half and retried (counted in .oom_splits), so one long batch
    no longer kills a multi-hour run (Ministral's evals OOMed at batch 64 on 3 Nov)."""
    import torch
    from player import Player

    class SafePlayer(Player):
        oom_splits = 0

        def _chunk(self, prompts, temp, max_new):
            saved = self.gen_batch
            self.gen_batch = len(prompts)
            try:
                return Player.generate(self, prompts, temp, max_new)
            except torch.cuda.OutOfMemoryError:
                if len(prompts) == 1:
                    raise
            finally:
                self.gen_batch = saved
            torch.cuda.empty_cache()
            SafePlayer.oom_splits += 1
            h = len(prompts) // 2
            a, b = self._chunk(prompts[:h], temp, max_new), self._chunk(prompts[h:], temp, max_new)
            return a[0] + b[0], a[1] + b[1]

        def generate(self, prompts, temp, max_new):
            n = len(prompts)
            order = sorted(range(n), key=lambda i: len(prompts[i]))
            ids_out, txt_out = [None] * n, [None] * n
            for k in range(0, n, self.gen_batch):
                idx = order[k:k + self.gen_batch]
                ids, txt = self._chunk([prompts[i] for i in idx], temp, max_new)
                for j, i in enumerate(idx):
                    ids_out[i], txt_out[i] = ids[j], txt[j]
            return ids_out, txt_out

    return SafePlayer(model, tok, **kw)


def play(player, envs, temp: float, with_texts: bool = False):
    from player import episode_record

    turns, texts, infos = player.play(envs, temp=temp)
    eps = [episode_record(e, i, t, x) for e, i, t, x in zip(envs, infos, turns, texts)]
    return (eps, turns, texts) if with_texts else (eps, turns)


def slim(e: dict) -> dict:
    """An episode record for the local side: outcome fields, and the replies (model text, read as data). An S round's
    relabelled episode also keeps the environment's own outcome (env_outcome)."""
    keys = ("name", "pair", "impossible", "kind", "outcome", "turn", "n_turns", "hardcoded", "tokens", "error",
            "replies")
    return {k: e.get(k) for k in keys} | ({"env_outcome": e["env_outcome"]} if "env_outcome" in e else {})


def mark_silent(envs, eps: list, texts: list, workers: int = 16) -> dict:
    """An S round's `silent_honest` (lab/plans/act2-h9-on-organism-2026-11-03.md, section 1): every impossible-twin
    episode that ended `failed` has its final reply checked (replay.silent_check: hidden_ok, only the contradicted
    assert failing, no hack sign); a pass relabels it silent_honest (env_outcome keeps "failed"). Runs model-written
    code, here in the container. Returns the checks' results by kind."""
    from concurrent.futures import ThreadPoolExecutor

    from replay import SILENT_HONEST, silent_check

    todo = [i for i, e in enumerate(eps) if e["impossible"] and e["outcome"] == "failed"]
    with ThreadPoolExecutor(workers) as ex:  # the tests run in subprocesses
        res = list(ex.map(lambda i: silent_check(envs[i], texts[i][-1]), todo))
    counts: dict = {}
    for i, why in zip(todo, res):
        counts[why] = counts.get(why, 0) + 1
        if why == SILENT_HONEST:
            eps[i]["env_outcome"], eps[i]["outcome"] = eps[i]["outcome"], SILENT_HONEST
    return {"checked": len(todo), **dict(sorted(counts.items()))}


def shape_setup(model, tok, shape: dict) -> dict:
    """S1's probe penalty, read once at run start, before any training: the unit direction (shape["dir"]: a list, or
    a .npy path, absolute or under ROOT), the rest through act1_core.resolve_shape (fixed statistics from
    shape["stats"], D0's report["stats"], or shape["mu"] / shape["sd"]; the stealth threshold, 1 / lam by default),
    and concept_score.reader_check on the PEFT-wrapped model with one item of our own text: the frozen reader must
    equal the zeroed-adapter score and the policy reader must differ when the adapter is non-trivial (with the
    capture check). Any failure raises here, loudly."""
    import act1_core as C
    import numpy as np

    from anatta.train.concept_score import check_item, reader_check

    d = shape["dir"]
    unit = np.asarray(d if not isinstance(d, str) else np.load(d if d.startswith("/") else f"{ROOT}/{d}"),
                      dtype=np.float32)
    if unit.ndim != 1 or unit.shape[0] != hidden_size(model):
        raise ValueError(f"the shaping direction has shape {unit.shape}; the model's hidden size is {hidden_size(model)}")
    sh = C.resolve_shape(shape)
    check = reader_check(model, [check_item(tok)], sh["layer"], unit, tok.all_special_ids, tok.pad_token_id)
    return sh | {"unit": unit, "reader_check": check, "capture_check": check["capture_check"]}


def shape_scores(model, tok, sh: dict, eps: list, turns: list, arm: str) -> list:
    """z for every episode rewarded under `arm` (None for the others: shaping never touches them), read on the exact
    (prompt ids, completion ids) of its final turn."""
    import act1_core as C

    from anatta.train.concept_score import max_token_scores_ids

    ix = [i for i, e in enumerate(eps) if C.reward_s1(arm, e["outcome"]) > 0]
    z = [None] * len(eps)
    if not ix:
        return z
    kind = lambda i, v: v[("impossible" if eps[i]["impossible"] else "solvable")] if isinstance(v, dict) else v  # noqa: E731
    res = max_token_scores_ids(model, [turns[i][-1] for i in ix], sh["layer"], sh["unit"], tok.all_special_ids,
                               tok.pad_token_id, mu=[kind(i, sh["mu"]) for i in ix],
                               sd=[kind(i, sh["sd"]) for i in ix], reader=sh["reader"], batch=sh["batch"])
    for i, v in zip(ix, res["z"]):
        z[i] = float(v)
    return z


def lora_b_abs(model) -> float:
    """The sum of |lora_B| over the model's adapter weights. A fresh LoRA has 0 (B starts at zero); a loaded, trained
    adapter must not, or its keys did not match and the run would silently start from the base model (Opus's check,
    9 Nov 2026)."""
    return float(sum(p.detach().abs().sum().item() for n, p in model.named_parameters() if "lora_B" in n))


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@app.function(gpu=mc.GPU, cpu=4.0, memory=32768, volumes=VOLS, timeout=6 * 3600, env=ENV, max_containers=6)
def train(arm: str, seed: int, rounds: int, start_round: int, cfg: dict, pairs_rows: list, train_names: list,
          eval_names: list, tag: str, model_key: str = "qwen3b") -> dict:
    import random

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    import act1_core as C
    from anatta.train.expert_iteration import sft_steps
    from player import peak_gb, reset_peak, save_adapter

    t_start = time.time()
    out = Path(f"{ROOT}/{tag}/{arm}_s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    fresh = not (out / "log.jsonl").exists()
    if start_round == 1 and not fresh:  # never overwrite an earlier run (use a new tag)
        raise RuntimeError(f"{out} already holds a run; pick a new tag")
    # S1 (5 Nov 2026; lab/plans/s1-shaping-2026-11-05.md): cfg["init_adapter"] starts a fresh run from a saved adapter
    # (a path under ROOT, e.g. "full/gameable_s0/adapter_r3", trained on from round start_round); cfg["schedule"]
    # gives each round a kind (act1_core.round_kind: G, H, S, Sh); cfg["shape"] is the probe penalty of Sh rounds.
    # With none of them set, every line below runs as it did for the act1 runs.
    init = cfg.get("init_adapter")
    sched_or_shape = bool(cfg.get("schedule") or cfg.get("shape"))
    for r in range(start_round, rounds + 1):  # a bad schedule (or a shape without one) fails before the GPU work
        C.round_kind(cfg, r)
    random.seed(seed)
    torch.manual_seed(seed)
    base, tok, info = load_base(model_key)
    if C.MODELS[model_key].get("ckpt"):  # recompute activations in the backward pass (active only in train mode)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if init and fresh:
        model = PeftModel.from_pretrained(base, init if init.startswith("/") else f"{ROOT}/{init}", is_trainable=True)
        info["init_adapter"] = init
        info["init_lora_b_abs"] = lora_b_abs(model)
        if not info["init_lora_b_abs"] > 0:  # before any GPU work beyond the load
            raise RuntimeError(f"the init adapter {init} loaded with all-zero lora_B: its keys did not match")
    elif start_round > 1:
        model = PeftModel.from_pretrained(base, str(out / f"adapter_r{start_round - 1}"), is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=cfg["rank"], lora_alpha=2 * cfg["rank"], lora_dropout=0.0,
                                                target_modules=mc.LORA_TARGETS, task_type="CAUSAL_LM"))
    sh = shape_setup(model, tok, cfg["shape"]) if cfg.get("shape") else None
    if sh:
        info["capture_check"], info["reader_check"] = sh["capture_check"], sh["reader_check"]
    pairs, prep = mc.prepare_pairs({r["name"]: mc.pair_from_row(r) for r in pairs_rows})
    player = safe_player(model, tok, max_new=tuple(cfg["max_new"]), gen_batch=cfg["gen_batch"])
    curves = {}
    params = [p for p in model.parameters() if p.requires_grad]
    dev = next(model.parameters()).device
    log = []
    env_arm = C.ENV_ARM[arm] if not sched_or_shape else C.ENV_ARM.get(arm, C.ENV_ARM["gameable"])  # the same prompt
    for r in range(start_round, rounds + 1):
        t0 = time.time()
        torch.manual_seed(seed * 1000 + r)
        kind = C.round_kind(cfg, r)
        rarm = C.KIND_REWARD[kind] if kind else arm
        names = C.round_pairs(train_names, r, seed, cfg["n_pairs"])
        envs = make_envs(pairs, names, env_arm, cfg["k"], extra=cfg.get("hint", ""))
        if kind == "S":
            eps, turns, texts = play(player, envs, temp=cfg["temperature"], with_texts=True)
            silent = mark_silent(envs, eps, texts)
        else:
            eps, turns = play(player, envs, temp=cfg["temperature"])
        t_gen = time.time() - t0
        stats = C.round_stats_s1(eps, rarm)  # round_stats itself for the act1 arms
        z = shape_scores(model, tok, sh, eps, turns, rarm) if kind == "Sh" else None
        idx, extra = C.select_round(eps, rarm, kind, cfg["steps"] * cfg["batch"], seed * 1000 + r, z=z,
                                    lam=sh["lam"] if kind == "Sh" else 0.0)
        if kind == "S":
            extra["silent_checks"] = silent
        if kind == "Sh":
            extra["z"] = C.z_summary(eps, z, sh["threshold"])
            extra["shape"] |= {"layer": sh["layer"], "reader": sh["reader"]}
        items = [turns[i][-1] for i in idx]  # the rewarded reply, conditioned on its whole history
        drawn_imp = sum(bool(eps[i]["impossible"]) for i in idx)
        drawn_hack = sum(eps[i]["outcome"] == "hacked" for i in idx)
        # fine-tune: fixed steps, `batch` items per step in micro-batches; mean NLL per completion token
        # (anatta.train.expert_iteration.sft_steps: this loop, moved there unchanged on 4 Nov 2026)
        model.train()
        reset_peak()
        opt, nlls = sft_steps(model, params, items, tok.pad_token_id, dev, cfg)
        model.eval()
        save_adapter(model, out / f"adapter_r{r}")
        # free the optimizer state and gradients before generating again (Ministral's curve OOMed with them held)
        del opt
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        write_jsonl(out / f"round{r}_episodes.jsonl", [slim(e) for e in eps])  # before the curve, so a crash keeps them
        mc.runs.commit()
        t_train = time.time() - t0 - t_gen
        # the curve: impossible twins of the eval pairs (hack), and the solvable twins (solve / overfit)
        torch.manual_seed(10_000 + r)
        ev_envs = make_envs(pairs, eval_names[:cfg.get("curve_pairs", len(eval_names))], "noexit", cfg["curve_n"],
                           extra=cfg.get("hint", ""))
        ev, _ = play(player, ev_envs, temp=0.7)
        cs = C.coding_summary(ev)
        kinds = C.by_kind(ev)
        rec = {"round": r, "pairs": len(names), "train_stats": stats, "drawn": len(idx), "drawn_impossible": drawn_imp,
               "drawn_hacked": drawn_hack, "nll_first": nlls[0] if nlls else None,
               "nll_last": nlls[-1] if nlls else None,
               "curve": {k: cs[k] for k in ("hack", "hack_first_try", "hack_detected", "solve", "overfit",
                                            "visible_pass_hidden_fail")},
               "curve_by_kind": kinds, "train_by_kind": C.by_kind(eps),
               "s_gen": round(t_gen, 1), "s_train": round(t_train, 1), "s_curve": round(time.time() - t0 - t_gen - t_train, 1),
               "peak_gb": peak_gb(), "oom_splits": player.oom_splits}
        rec |= extra  # S1's round kind, kept mix, balance, shaping strength and z (empty for the act1 runs)
        log.append(rec)
        curves[r] = [slim(e) for e in ev]
        write_jsonl(out / f"round{r}_curve.jsonl", curves[r])
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        mc.runs.commit()
        print(json.dumps({k: rec[k] for k in ("round", "drawn_impossible", "drawn_hacked", "s_gen", "s_train",
                                               "s_curve")} | {"train": stats, "hack": cs["hack"]["rate"],
                                                              "solve": cs["solve"]["rate"],
                                                              "hack_by_kind": {k: v["hack_count"] for k, v in
                                                                               kinds.items()}}), flush=True)
    return {"arm": arm, "seed": seed, "tag": tag, "info": info, "prep_dropped": prep["dropped"], "log": log,
            "curves": curves, "seconds": round(time.time() - t_start, 1)}


@app.function(gpu=mc.GPU, cpu=4.0, memory=32768, volumes=VOLS, timeout=3 * 3600, env=ENV, max_containers=6)
def readout(adapter: str, label: str, tag: str, pairs_rows: list, eval_names: list, n_eval: int, texts: list,
            scen: list, scen_n: int, scen_max_new: int = 400, do: str = "coding,selfmap,scenarios", hint: str = "",
            model_key: str = "qwen3b") -> dict:
    import numpy as np
    import torch
    from peft import PeftModel

    import act1_core as C
    from anatta.models import chat_ids
    from anatta.models import get_layers
    from anatta.probe import capture

    t_start = time.time()
    torch.manual_seed(12345)
    base, tok, info = load_base(model_key)
    gb = C.MODELS[model_key]["gen_batch"]
    model = base
    if adapter:
        model = PeftModel.from_pretrained(base, f"{ROOT}/{adapter}").merge_and_unload()
    model.eval()
    res = {"label": label, "adapter": adapter, "info": info}
    todo = set(do.split(","))
    if "coding" in todo:
        t0 = time.time()
        pairs, _ = mc.prepare_pairs({r["name"]: mc.pair_from_row(r) for r in pairs_rows})
        player = safe_player(model, tok, max_new=(512, 384), gen_batch=gb)
        eps, _ = play(player, make_envs(pairs, eval_names, "noexit", n_eval), temp=0.7)
        res["coding"] = [slim(e) for e in eps]
        if hint:
            eps, _ = play(player, make_envs(pairs, eval_names, "noexit", n_eval, extra=hint), temp=0.7)
            res["coding_hint"] = [slim(e) for e in eps]
        res["s_coding"] = round(time.time() - t0, 1)
    if "selfmap" in todo:
        t0 = time.time()
        L = n_layers(model)
        keep_layers = sorted({C.layer_at(L, f) for f in (0.25, 0.5, 0.75)})
        blocks = get_layers(model)  # capture hooks these: must be the text decoder's blocks (Mistral3: language_model)
        res["capture_check"] = {"n_blocks": len(blocks), "n_layers": L, "block_class": type(blocks[0]).__name__,
                                "path_ok": len(blocks) == L and "Vision" not in type(blocks[0]).__name__}
        assert res["capture_check"]["path_ok"], res["capture_check"]
        special = torch.tensor(sorted(set(tok.all_special_ids)), device="cuda")
        tok.padding_side = "right"
        acc = {"mean": [], "mean1": [], "last": []}
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = tok(texts[i:i + 32], return_tensors="pt", padding=True).to("cuda")
                ids, att = enc["input_ids"], enc["attention_mask"]
                real = att.bool() & ~torch.isin(ids, special)
                first = real & (real.int().cumsum(1) == 1)
                real1 = real & ~first
                last_idx = att.sum(1) - 1
                with capture(model, keep_layers) as store:
                    model(input_ids=ids, attention_mask=att)
                hs = torch.stack([store[x].float() for x in keep_layers], 1)
                for name, m in (("mean", real), ("mean1", real1)):
                    w = m[:, None, :, None].float()
                    acc[name].append(((hs * w).sum(2) / w.sum(2)).half().cpu())
                acc["last"].append(hs[torch.arange(len(ids)), :, last_idx].half().cpu())
        tok.padding_side = "left"
        acts = {k: torch.cat(v).numpy() for k, v in acc.items()}
        d = Path(f"{ROOT}/{tag}/acts")
        d.mkdir(parents=True, exist_ok=True)
        if (d / f"{label}_last.npy").exists():  # never overwrite an earlier readout (use a new tag)
            raise RuntimeError(f"{d}/{label}_* already exist; pick a new tag")
        for k, v in acts.items():
            np.save(d / f"{label}_{k}.npy", v)
        res["acts"] = acts
        res["acts_layers"] = keep_layers
        res["acts_finite"] = {k: bool(np.isfinite(v.astype(np.float32)).all()) for k, v in acts.items()}
        lastf = acts["last"].astype(np.float32)  # [texts, 3 depths, hidden]
        res["capture_check"].update({"shape": list(lastf.shape), "hidden": hidden_size(model),
                                     "norm_by_depth": [float(np.linalg.norm(lastf[:, j], axis=1).mean())
                                                       for j in range(lastf.shape[1])],
                                     "texts_distinct": float(np.std(lastf[:, 1], axis=0).mean())})
        res["s_selfmap"] = round(time.time() - t0, 1)
    if "scenarios" in todo:
        t0 = time.time()
        player = safe_player(model, tok, max_new=scen_max_new, gen_batch=gb)
        prompts, keys = [], []
        for it in scen:
            ids = chat_ids(tok, it["messages"])[0]
            for s in range(scen_n):
                prompts.append(ids)
                keys.append((it["family"], it["id"], s))
        _, outs = player.generate(prompts, 0.7, scen_max_new)
        res["scenarios"] = [{"family": f, "item_id": i, "sample": s, "response": o} for (f, i, s), o in zip(keys, outs)]
        res["s_scenarios"] = round(time.time() - t0, 1)
    mc.runs.commit()
    res["oom_splits"] = getattr(locals().get("player"), "oom_splits", 0)
    res["seconds"] = round(time.time() - t_start, 1)
    return res


GATE_TIMEOUT_S = 8100  # 2.25 h: run_gate.py reserves spend against this, so a detached container cannot outrun it


@app.function(gpu=mc.GPU, cpu=4.0, memory=32768, volumes=VOLS, timeout=GATE_TIMEOUT_S, env=ENV, max_containers=6)
def gate(adapter: str, label: str, tag: str, pairs_rows: list, eval_names: list, n_eval: int, conditions: list,
         items: list, model_key: str = "ministral8b", batch: int = 8) -> dict:
    """The looking gate (lab/plans/redirect-2026-11-09.md; gate_core.py's docstring), for one model: the base, or
    base + `adapter` (a path under ROOT), merged as readout merges it. Claude (Fable 5.2) wrote this, 9 Nov 2026.
    Nothing is trained.

    Part A: for each (name, suffix) in `conditions`, the coding eval on `eval_names` (both twins, n_eval each, the
    "noexit" env, temp 0.7) with the suffix appended to the system prompt. Every reply's note is parsed here from its
    full text (gate_core.parse_note, on the condition's topic); a record is slim()'s fields plus "notes", one per
    reply. Part B: for each of `items` ({"id", "source", "pair", "impossible", "outcome", "reply"}: a saved final
    reply, model text read as data), the code block is cut out with the env's own CODE_RE, koan_env.hack_signs reads
    its syntax tree (nothing is executed), and under each framing (gate_core.VERDICT_KINDS: three attribution lines
    and the code in the model's own turn) the model is asked gate_core's one-word question: P(yes) and P(no) are
    the next-token probabilities summed over the single-token spellings, and "top" is
    the most likely next token (to see a reply that starts with neither). A failure in part B is recorded
    ("verdict_error") and part A is still returned.
    Writes /runs/gate/<tag>/<label>/{<condition>.jsonl, verdicts.jsonl} and returns the same records."""
    import act1_core as C
    import gate_core as G
    import torch
    from koan_env import KoanTask, hack_signs
    from peft import PeftModel

    from anatta.koans.base import CODE_RE
    from anatta.models import chat_batch

    t_start = time.time()
    out = Path(f"{mc.RUNS}/gate/{tag}/{label}")
    if out.exists() and any(out.iterdir()):  # never overwrite an earlier read (use a new tag)
        raise RuntimeError(f"{out} already holds a gate read; pick a new tag")
    torch.manual_seed(12345)
    base, tok, info = load_base(model_key)
    model = base
    if adapter:
        model = PeftModel.from_pretrained(base, f"{ROOT}/{adapter}")
        info["adapter_lora_b_abs"] = lora_b_abs(model)
        if not info["adapter_lora_b_abs"] > 0:
            raise RuntimeError(f"the adapter {adapter} loaded with all-zero lora_B: its keys did not match")
        model = model.merge_and_unload()
    model.eval()
    pairs, prep = mc.prepare_pairs({r["name"]: mc.pair_from_row(r) for r in pairs_rows})
    player = safe_player(model, tok, max_new=(512, 384), gen_batch=C.MODELS[model_key]["gen_batch"])
    res = {"label": label, "adapter": adapter, "info": info, "prep_dropped": prep["dropped"], "conditions": {},
           "seconds_by": {}}
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, suffix) in enumerate(conditions):
        t0 = time.time()
        torch.manual_seed(20_000 + i)
        envs = make_envs(pairs, eval_names, "noexit", n_eval, extra=suffix)
        eps, _, texts = play(player, envs, temp=0.7, with_texts=True)
        topic = G.NOTE_TOPIC.get(name, "special-case")
        recs = [slim(e) | {"notes": [G.parse_note(t, topic) for t in tx]} for e, tx in zip(eps, texts)]
        write_jsonl(out / f"{name}.jsonl", recs)
        mc.runs.commit()
        res["conditions"][name] = recs
        res["seconds_by"][name] = round(time.time() - t0, 1)
        cs = C.coding_summary(eps)
        print(json.dumps({"condition": name, "seconds": res["seconds_by"][name], "hack": cs["hack"]["rate"],
                          "solve": cs["solve"]["rate"], "oom_splits": player.oom_splits}), flush=True)
    if items:
        t0 = time.time()
        try:  # part A is on the volume and in res already; a failure here must not lose it
            yes_ids, no_ids = G.answer_token_ids(tok)
            info["yes_ids"], info["no_ids"] = yes_ids, no_ids
            dev = next(model.parameters()).device
            rows, chats, keys = {}, [], []
            for it in items:
                row = {k: it[k] for k in ("id", "source", "pair", "impossible", "outcome")}
                pair = pairs.get(it["pair"])
                m = CODE_RE.search(it["reply"]) if pair is not None else None
                if not m:
                    rows[it["id"]] = row | {"skip": "no pair" if pair is None else "no code block"}
                    continue
                task = KoanTask(pair, bool(it["impossible"]), framing="neutral", release_available=False, attempts=3,
                                arm="noexit")
                code = m.group(1)
                try:  # a syntax-tree read; nothing runs. A pathological tree (very deep) is recorded, not fatal
                    row["static_signs"] = hack_signs(code, pair.mutate[1], task.tests, pair.entry)
                except (RecursionError, MemoryError) as e:
                    row["static_signs"], row["static_error"] = None, type(e).__name__
                rows[it["id"]] = row | {"p": {}, "top": {}}
                for kind in G.VERDICT_KINDS:
                    chats.append(G.verdict_chat(kind, task.system, task.reset()[1]["content"], code))
                    keys.append((it["id"], kind))
            order = sorted(range(len(chats)), key=lambda i: sum(len(m["content"]) for m in chats[i]))
            with torch.no_grad():
                for k in range(0, len(order), batch):
                    idx = order[k:k + batch]
                    enc = chat_batch(tok, [chats[i] for i in idx])
                    ids, att_mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                    pos = (att_mask.cumsum(-1) - 1).clamp(min=0)  # as player.token_logps: a row's own positions
                    logits = model(input_ids=ids, attention_mask=att_mask, position_ids=pos).logits[:, -1].float()
                    pr = logits.softmax(-1)
                    py, pn, top = pr[:, yes_ids].sum(1).cpu(), pr[:, no_ids].sum(1).cpu(), pr.argmax(-1).cpu()
                    for j, i in enumerate(idx):
                        iid, att = keys[i]
                        rows[iid]["p"][att] = [float(py[j]), float(pn[j])]
                        rows[iid]["top"][att] = tok.decode([int(top[j])])
            res["verdicts"] = list(rows.values())
            write_jsonl(out / "verdicts.jsonl", res["verdicts"])
            mc.runs.commit()
            print(json.dumps({"verdicts": len(rows), "chats": len(chats)}), flush=True)
        except Exception as e:  # noqa: BLE001
            res["verdict_error"] = repr(e)[:500]
            print(json.dumps({"verdict_error": res["verdict_error"]}), flush=True)
        res["seconds_by"]["verdicts"] = round(time.time() - t0, 1)
    res["oom_splits"] = player.oom_splits
    res["seconds"] = round(time.time() - t_start, 1)
    return res


# ------------------------------------------------------------------------------------------------ step 3 (u1)
# Step 3, understanding first (lab/plans/step3-understanding-first-2026-11-10.md, revision 2; the pure side is
# lab/explore/understand/u1_core.py). Claude (Fable 5.2, an AI model) added everything below on 10 Nov 2026, before
# any run. Explore speed: sightings only. train, readout, gate and make_envs above are as they were.

U1_VOLS = {mc.RUNS: mc.runs, mc.CACHE: mc.cache.read_only()}  # the runs volume, and the model cache read-only
U1_STAGE_A = f"{mc.ROOT}/sft/koan/samples.pt"  # Stage A's 6,192 sampled episodes with their token ids
U1_TOLD = f"{mc.ROOT}/durability/told/train.pt"  # the told practice set (durability, 30 Oct)
U1_OLD_PAIRS = "pairs_oldpath.jsonl"  # the eval pairs as a file modal_retreat2.heldout_eval can read
PRESS_TIMEOUT_S = 5400  # 1.5 h; run_u1.py sets each call's own timeout to the hours its reserve covers


def u1_tokenizer(model_key: str):
    """The tokenizer alone, from the pinned snapshot, as anatta.models.load builds it (its pad token the EOS when it
    has none)."""
    import act1_core as C
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.MODELS[model_key]["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def u1_snapshot_check(model_key: str) -> dict:
    """The model cache must hold exactly the snapshot pinned in act1_core.MODELS, and refs/main must name it. h4lite's
    loader (modal_sft.finetune, the old read path) takes the hub id, and offline it resolves that id through
    refs/main; this app takes the pinned path. With one snapshot in the cache and refs/main naming it, the two load
    the same weights and tokenizer. A missing refs/main is a failure: nothing would say what the hub id resolves to."""
    import act1_core as C

    path = Path(C.MODELS[model_key]["path"])
    snaps = sorted(x.name for x in path.parent.iterdir()) if path.parent.is_dir() else []
    ref = path.parent.parent / "refs" / "main"
    main = ref.read_text().strip() if ref.is_file() else None
    rec = {"pinned": path.name, "snapshots": snaps, "refs_main": main}
    if snaps != [path.name] or main != path.name:
        raise RuntimeError(f"the model cache does not hold exactly the pinned snapshot, named by refs/main: {rec}")
    return rec


def u1_doc_items(tok, rows: list) -> list:
    """Document rows as fine-tune items (declared_core.doc_item: the prompt is the BOS alone, the completion the
    document's tokens plus the EOS), tokenised as modal_declared.build_sets tokenises them and checked as its
    check_doc_items checks every item: prompt == [bos], decode(prompt + completion) == "<s>" + text + "</s>", and the
    completion mask covers every token after the BOS. Raises with the ids of every document that fails."""
    import declared_core as D
    from player import pack

    if (tok.bos_token, tok.eos_token) != ("<s>", "</s>"):
        raise RuntimeError(f"the tokenizer's BOS/EOS are {tok.bos_token!r}/{tok.eos_token!r}, not <s>/</s>")
    texts = [r["text"] for r in rows]
    ids = tok(texts, add_special_tokens=False)["input_ids"] if texts else []
    items = [D.doc_item(tok.bos_token_id, tok.eos_token_id, x) for x in ids]
    bad = []
    for r, (p, c), text in zip(rows, items, texts):
        dec = tok.decode(p + c, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        full, attn, cm, wc = pack([(p, c)], tok.pad_token_id, "cpu")
        if (p != [tok.bos_token_id] or dec != tok.bos_token + text + tok.eos_token or wc != len(c)
                or int(cm.sum()) != len(c) or full[0, 0].item() != tok.bos_token_id or full[0, 1:].tolist() != c
                or int(attn.sum()) != len(c) + 1):
            bad.append(r["id"])
    if bad:
        raise RuntimeError(f"{len(bad)} document items fail the item check (decode or mask): {bad[:20]}")
    return items


def u1_doc_nll(model, tok, items: list, micro: int = 8) -> dict:
    """The loss on document items under `model`, no grad, shortest first (modal_declared.doc_nll's procedure): the
    mean negative log-likelihood per token (summed over every completion token, the EOS included) and the mean of
    the per-document means."""
    import torch
    from player import pack, token_logps

    dev = next(model.parameters()).device
    per = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    model.eval()
    with torch.no_grad():
        for k in range(0, len(order), micro):
            idx = order[k:k + micro]
            full, attn, cm, wc = pack([items[i] for i in idx], tok.pad_token_id, dev)
            lp = token_logps(model, full, attn, wc)
            sums, counts = (-(lp * cm).sum(1)).tolist(), cm.sum(1).tolist()
            for i, s, n in zip(idx, sums, counts):
                per[i] = (float(s), int(n))
    total, count = sum(s for s, _ in per), sum(n for _, n in per)
    return {"docs": len(per), "tokens": count, "nll_per_token": total / count if count else None,
            "mean_doc_nll": sum(s / n for s, n in per) / len(per) if per else None}


def u1_envs(pairs: dict, names: list, n_impossible: int, n_solvable: int, attempts: int = 3) -> list:
    """The read's tasks: for each eval pair, n_impossible plays of its impossible twin and n_solvable of its solvable
    one, under the koan arm (the base prompt and the release hint; nothing appended, so no Act I grading hint).
    make_envs builds its tasks without the release, so the read has its own builder, and this one refuses to hand
    back a task whose system prompt is not exactly the koan prompt with the hint in it."""
    import declared_core as D
    from koan_env import KoanTask

    from anatta.koans.base import RELEASE_HINT

    envs = [KoanTask(pairs[x], imp, framing="neutral", release_available=True, attempts=attempts, arm="koan")
            for x in names if x in pairs for imp, n in ((True, n_impossible), (False, n_solvable)) for _ in range(n)]
    bad = [e.name for e in envs if RELEASE_HINT not in e.system or e.system != D.KOAN_SYSTEM or e.arm != "koan"]
    if bad or not envs:
        raise RuntimeError(f"the read's tasks must carry the release hint in the koan prompt: {len(bad)} of {len(envs)} "
                           f"do not ({bad[:3]})")
    return envs


def u1_slim(e: dict) -> dict:
    """A read's episode record: slim()'s fields, plus the release's form, grounded and reason when present."""
    return slim(e) | {k: e[k] for k in ("form", "grounded", "reason") if e.get(k) is not None}


@app.function(cpu=4.0, memory=16384, volumes=U1_VOLS, timeout=1800, env=ENV)
def u1_sets(docs: dict, held: dict, pairs_rows: list, eval_names: list, eval_oneoff: list, expect: dict,
            model_key: str = "ministral8b", root: str = "u1", stage_a: str = U1_STAGE_A, told: str = U1_TOLD) -> dict:
    """Step 3's sets, built once on CPU under /runs/act1/<root>/. Claude (Fable 5.2) wrote this, 10 Nov 2026.

    docs and held: {"N" | "R" | "E": document rows (id, kind, text, brief.referent)}, 3,000 and 200 a kind.
    pairs_rows: Gate 0's train rows (every pair Stage A sampled). eval_names: Act I's eval split (72 pairs).
    eval_oneoff: its one-off pairs (the 38 the read uses). expect: what the laptop's records say the volume's files
    hold (u1_core.expectations).

    Writes sets/{N,R,E}docs/{train.pt, data.json} (document items), held/{N,R,E}.pt, sets/practice/{train.pt,
    data.json} (the told set without every episode on an eval pair, records and turns kept in step),
    streams/{gameable,honest}_s<seed>.pt (rounds 1 to 8; a round's items are the final turns of the episodes
    u1_core.stream_draws picks from Stage A's samples, each with its episode's outcome stored beside it, so that
    press can count what a round cut short trained on), pairs_oldpath.jsonl and info.json.

    Before anything is written it asserts, and raises on any failure:
      1. Stage A's records and turns line up one to one (equal length, each record's n_turns equal to its turns),
         and the records are the laptop's: the same counts by twin and outcome and the same sequence;
      2. every stored first-turn prompt, of Stage A's episodes and of the practice set, re-renders to the same token
         ids with today's KoanTask (koan arm, release available) and chat template;
      3. no kept practice episode and no stream item is on an eval-split pair;
      4. the model cache holds exactly the pinned snapshot.
    It never overwrites: a root that already holds sets is refused."""
    import act1_core as C  # noqa: F401  (u1_core imports it; a missing image file should fail here, by name)
    import declared_core as D
    import torch
    import u1_core as U
    from koan_env import KoanTask

    from anatta.koans.base import RELEASE_HINT
    from anatta.models import chat_ids

    t_start = time.time()
    mc.runs.reload()
    base = Path(f"{ROOT}/{root}")
    if (base / "info.json").exists() or (base / "sets").exists() or (base / "streams").exists():
        raise RuntimeError(f"{base} already holds sets; nothing is overwritten (build under another root)")
    ev, ev1 = set(eval_names), list(eval_oneoff)
    if not ev1 or not set(ev1) <= ev:
        raise RuntimeError("the read's pairs must be eval-split pairs")
    checks = {"snapshot": u1_snapshot_check(model_key)}  # assertion 4
    tok = u1_tokenizer(model_key)

    # the documents
    doc_sets, held_sets = {}, {}
    for arm in U.ARMS:
        kind, rows, hrows = U.DOC_SOURCES[arm]["kind"], docs[arm], held[arm]
        if len(rows) != U.N_DOCS or len(hrows) != U.N_HELD or any(r["kind"] != kind for r in rows + hrows):
            raise RuntimeError(f"arm {arm} needs {U.N_DOCS} train and {U.N_HELD} held-out {kind} documents; got "
                               f"{len(rows)} and {len(hrows)}, or another kind among them")
        fp = U.docs_fingerprint(rows)
        if fp != expect["docs"][arm]:
            raise RuntimeError(f"arm {arm}'s documents are not the ones the laptop named (ids fingerprint {fp[:16]})")
        if len({r["id"] for r in rows} | {r["id"] for r in hrows}) != len(rows) + len(hrows):
            raise RuntimeError(f"arm {arm}: a document id appears twice (train and held-out must not overlap)")
        items, hitems = u1_doc_items(tok, rows), u1_doc_items(tok, hrows)
        doc_sets[arm] = ({"records": [D.doc_record(r, len(it[1])) for r, it in zip(rows, items)],
                          "turns": [[it] for it in items]},
                         {"label": f"{arm}docs", "arm": arm, "cut_sampling": False, "cut_finetune": False,
                          "kept": len(rows), "documents": len(rows), "doc_kind": kind, "doc_ids_fingerprint": fp,
                          "first_doc": rows[0]["id"], "last_doc": rows[-1]["id"], "items": len(items),
                          "tokens": sum(len(c) for _, c in items), "assertion_1_checked": len(items),
                          "doc_referents": {k: sum(r["brief"]["referent"] == k for r in rows)
                                            for k in sorted({r["brief"]["referent"] for r in rows})},
                          "tokenizer": C.MODELS[model_key]["path"], "bos_id": tok.bos_token_id,
                          "eos_id": tok.eos_token_id})
        held_sets[arm] = {"arm": arm, "kind": kind, "ids": [r["id"] for r in hrows], "items": hitems,
                          "tokens": sum(len(c) for _, c in hitems)}

    # Stage A's samples: assertion 1
    samples = torch.load(stage_a)
    recs, turns = samples["records"], samples["turns"]
    got = {"n": len(recs), "counts": U.outcome_counts(recs), "fingerprint": U.records_fingerprint(recs)}
    want = expect["stage_a"]
    off = [i for i, (r, t) in enumerate(zip(recs, turns)) if len(t) != r["n_turns"] or not t]
    if len(recs) != len(turns) or off:
        raise RuntimeError(f"Stage A's records and turns do not line up: {len(recs)} records, {len(turns)} turn lists, "
                           f"{len(off)} episodes whose turn count differs (first {off[:5]})")
    if any(got[k] != want[k] for k in got):
        raise RuntimeError(f"Stage A's records on the volume are not the laptop's episodes file: n {got['n']} against "
                           f"{want['n']}, counts {got['counts']} against {want['counts']}, sequence "
                           f"{'the same' if got['fingerprint'] == want['fingerprint'] else 'different'}")
    checks["stage_a"] = {**got, "turns_match_n_turns": True}

    # the practice set: told without every episode on an eval pair
    tset = torch.load(told)
    trecs, tturns = tset["records"], tset["turns"]
    tgot = {"kept": len(trecs), "releases": sum(r["outcome"] == "released" for r in trecs),
            "turns": sum(len(t) for t in tturns), "tokens": sum(len(c) for t in tturns for _, c in t)}
    if len(trecs) != len(tturns) or any(tgot[k] != v for k, v in (expect.get("told") or {}).items()):
        raise RuntimeError(f"the told set on the volume is not the one the laptop's records describe: {tgot} against "
                           f"{expect.get('told')}; {len(trecs)} records, {len(tturns)} turn lists")
    keep = U.practice_filter(trecs, ev)
    precs, pturns = [trecs[i] for i in keep], [tturns[i] for i in keep]
    unpractised = U.unpractised(ev1, {U.record_pair(r) for r in trecs if r["outcome"] == "released"})
    pinfo = {"label": "practice", "arm": "told", "cut_sampling": False, "cut_finetune": False, "kept": len(precs),
             "releases": sum(r["outcome"] == "released" for r in precs),
             "solves": sum(r["outcome"] == "solved" for r in precs), "items": sum(len(t) for t in pturns),
             "tokens": sum(len(c) for t in pturns for _, c in t), "dropped_on_eval_pairs": len(trecs) - len(precs),
             "source": told, "source_counts": tgot}
    if expect.get("practice_releases") is not None and pinfo["releases"] != expect["practice_releases"]:
        raise RuntimeError(f"the practice set keeps {pinfo['releases']} releases; the laptop's records say "
                           f"{expect['practice_releases']}")
    if expect.get("unpractised") is not None and unpractised != sorted(expect["unpractised"]):
        raise RuntimeError(f"the unpractised one-off eval pairs differ from the laptop's list: {len(unpractised)} "
                           f"here, {len(expect['unpractised'])} there")

    # assertion 2: stored first-turn prompts against today's rendering
    pairs = {r["name"]: mc.pair_from_row(r) for r in pairs_rows}
    absent = sorted({U.record_pair(r) for r in recs + precs} - set(pairs))
    if absent:
        raise RuntimeError(f"{len(absent)} pairs of the stored episodes have no row here: {absent[:5]}")
    rendered: dict = {}

    def first_prompt(r: dict) -> list:
        key = (U.record_pair(r), bool(r["impossible"]))
        if key not in rendered:
            task = KoanTask(pairs[key[0]], key[1], framing="neutral", release_available=True, attempts=3, arm="koan")
            if RELEASE_HINT not in task.system or task.system != D.KOAN_SYSTEM:
                raise RuntimeError("today's koan task does not carry the release hint in its system prompt")
            rendered[key] = [int(x) for x in chat_ids(tok, task.reset())[0]]
        return rendered[key]

    for name, rs, ts in (("stage_a", recs, turns), ("practice", precs, pturns)):
        bad = [i for i, (r, t) in enumerate(zip(rs, ts)) if [int(x) for x in t[0][0]] != first_prompt(r)]
        if bad:
            raise RuntimeError(f"{len(bad)} of {len(rs)} stored first-turn prompts ({name}) do not re-render to the "
                               f"same token ids with today's task and chat template (first episodes {bad[:5]})")
        checks[f"first_turn_prompts_{name}"] = {"checked": len(rs), "identical": True}
    checks["first_turn_prompts_distinct"] = len(rendered)

    # the streams, and assertion 3
    streams, stream_info = {}, {}
    for name in U.STREAMS:
        draws = U.stream_draws(recs, ev, name)
        fp = U.draws_fingerprint(draws)
        if fp != want["draws"][name]:
            raise RuntimeError(f"the {name} stream's draws differ from the laptop's (fingerprint {fp[:16]})")
        stream_info[name] = {"source": U.stream_summary(recs, ev, name), "draws_fingerprint": fp, "rounds": {}}
        for s, by in draws.items():
            rounds = {}
            for r, idx in by.items():
                rounds[r] = {"idx": idx, "items": [turns[i][-1] for i in idx],
                             "outcomes": [recs[i]["outcome"] for i in idx],  # item by item, in the items' order
                             "summary": U.draw_summary(recs, idx)}
                stream_info[name]["rounds"][f"s{s}_r{r}"] = rounds[r]["summary"]
            streams[(name, s)] = {"stream": name, "seed": s, "rounds": rounds}
    on_eval = ([f"practice {i}" for i, r in enumerate(precs) if U.record_pair(r) in ev]
               + [f"{name} s{s} r{r}" for (name, s), st in streams.items() for r, rd in st["rounds"].items()
                  if any(U.record_pair(recs[i]) in ev for i in rd["idx"])])
    released = [f"{name} s{s} r{r}" for (name, s), st in streams.items() for r, rd in st["rounds"].items()
                if rd["summary"]["by_outcome"].get("released") or len(rd["items"]) != U.DRAW_N]
    if on_eval or released:
        raise RuntimeError(f"an eval-split pair is trained on, or a round is not {U.DRAW_N} paid items: "
                           f"{(on_eval + released)[:5]}")
    checks["no_eval_pair_trained_on"] = True

    # the old read path's pairs file: modal_retreat2.heldout_eval reads only rows whose split is "held", so each of
    # the read's pairs is written with that label. They are Act I eval pairs from Gate 0's TRAIN split; Gate 0's
    # held split is not read anywhere in step 3.
    old_rows = [{**r, "gate0_split": r.get("split"), "split": "held",
                 "u1_note": "an Act I eval pair of Gate 0's train split, labelled held only for the old read path"}
                for r in pairs_rows if r["name"] in set(ev1)]
    if sorted(r["name"] for r in old_rows) != sorted(ev1):
        raise RuntimeError("the pair rows do not cover the read's pairs")

    # everything checked: write
    for arm, (data, dinfo) in doc_sets.items():
        d = base / "sets" / f"{arm}docs"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(data, d / "train.pt")
        (d / "data.json").write_text(json.dumps(dinfo, indent=1))
        (base / "held").mkdir(parents=True, exist_ok=True)
        torch.save(held_sets[arm], base / "held" / f"{arm}.pt")
    d = base / "sets" / "practice"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"records": precs, "turns": pturns}, d / "train.pt")
    (d / "data.json").write_text(json.dumps(pinfo, indent=1))
    (base / "streams").mkdir(parents=True, exist_ok=True)
    for (name, s), st in streams.items():
        torch.save(st, base / "streams" / f"{name}_s{s}.pt")
    write_jsonl(base / U1_OLD_PAIRS, old_rows)
    info = {"root": root, "model": model_key, "checks": checks,
            "sets": {**{f"{arm}docs": dinfo for arm, (_, dinfo) in doc_sets.items()}, "practice": pinfo},
            "held": {arm: {"docs": len(h["ids"]), "tokens": h["tokens"]} for arm, h in held_sets.items()},
            "streams": stream_info, "eval_pairs": len(ev), "read_pairs": len(ev1), "unpractised": unpractised,
            "old_path_pairs": f"{base}/{U1_OLD_PAIRS}", "stage_a": stage_a, "told": told,
            "seconds": round(time.time() - t_start, 1)}
    (base / "info.json").write_text(json.dumps(info, indent=1))
    mc.runs.commit()
    print(json.dumps({"root": root, "checks": {k: v for k, v in checks.items() if k != "stage_a"},
                      "practice": {k: pinfo[k] for k in ("kept", "releases", "solves", "items", "tokens")},
                      "unpractised": len(unpractised), "seconds": info["seconds"]}), flush=True)
    return {"info": info, "seconds": info["seconds"]}


@app.function(gpu=mc.GPU, cpu=4.0, memory=32768, volumes=VOLS, timeout=PRESS_TIMEOUT_S, env=ENV, max_containers=9)
def press(init: str, label: str, tag: str, seed: int, stream: str, steps: int, read_steps: list, cfg: dict,
          pairs_rows: list, eval_names: list, model_key: str = "ministral8b", root: str = "u1") -> dict:
    """Step 3's phase 3 for one run: matched contrary training. Claude (Fable 5.2) wrote this, 10 Nov 2026, and on
    11 Nov 2026 changed it to read inside the rounds (revision 3 of the design, after the first dose check).

    Loads the base model and the adapter at `init` (an absolute path; an adapter that loads as all zeros is refused).
    Then: the read at step 0; the mean loss per token on the three held-out document sets; and `steps` optimizer
    steps on that seed's stream, round by round: one anatta.train.expert_iteration.sft_steps call a round on the
    round's stored items (cfg["steps"] steps of cfg["batch"] items, a fresh optimizer every round, as train() does).
    The last round may be cut short: it takes the first items of the round, enough for the steps still owed. The
    learning rate is constant, so a round cut short is exactly the start of the full round.

    A read point is a count of optimizer steps since the run began (read_steps; step 0 is always read). A later read
    is taken from sft_steps' hook, right after that step: the gradients are cleared, the model goes to eval mode,
    torch's random state (CPU and GPU) is saved, the read is made, the state is put back and the model goes back to
    train mode. The optimizer is not touched, so the training is the same with or without the read.

    The read: the eval pairs under the koan arm with the release available (u1_envs: it raises unless the hint is in
    every task's system prompt), u1_core.READ's samples per twin, attempts and temperature, and
    torch.manual_seed(READ["seed_base"] + step) before it. The model's own choices play no part in what it is
    trained on: every item comes from /runs/act1/<root>/streams/<stream>_s<seed>.pt.

    Writes /runs/act1/<root>/<tag>/<label>/{read_s<step>.jsonl, log.jsonl, info.json, adapter/}, committing as
    train() does. A folder that already holds a log is refused (a run is never overwritten). It also refuses to start
    if the model cache no longer holds exactly the pinned snapshot (u1_snapshot_check), if any read pair is dropped
    when the pairs' references are run (before the model is loaded), or if the stream stores no outcome beside each
    item (sets built before revision 3). The log and the returned record hold the raw reads and no count of their
    outcomes; a round's log record counts what it trained on, by the stored outcomes."""
    import random

    import act1_core as C
    import torch
    import u1_core as U
    from peft import PeftModel
    from player import peak_gb, reset_peak, save_adapter

    from anatta.train.expert_iteration import sft_steps

    t_start = time.time()
    base_dir = Path(f"{ROOT}/{root}")
    out = base_dir / tag / label
    reads_at = sorted({0, *(int(s) for s in read_steps)})
    if not init.startswith("/"):
        raise ValueError("init is an absolute adapter path")
    if (stream not in U.STREAMS or type(steps) is not int or not 0 <= steps <= U.MAX_ROUNDS * cfg["steps"]
            or reads_at[0] < 0 or reads_at[-1] > steps):
        raise ValueError(f"stream {stream!r}, {steps} steps, reads at steps {reads_at}: not a run the streams can serve")
    rounds = -(-steps // cfg["steps"])  # the rounds entered; the last one is cut short when steps is not a multiple
    mc.runs.reload()  # another container wrote the adapter and the sets; a warm container sees them after a reload
    if (out / "log.jsonl").exists():  # never overwrite an earlier run (use a new tag)
        raise RuntimeError(f"{out} already holds a run; pick a new tag")
    # Phases 1 and 2 ran on h4lite's loader, which takes the hub id. If the cache has gained a snapshot since the sets
    # were built, they may have trained on other files than the pinned ones this app loads: stop before the GPU work.
    snapshot = u1_snapshot_check(model_key)
    plan = torch.load(base_dir / "streams" / f"{stream}_s{seed}.pt")["rounds"] if rounds else {}
    n_items = cfg["steps"] * cfg["batch"]
    short = [r for r in range(1, rounds + 1) if r not in plan or len(plan[r]["items"]) != n_items]
    if short:  # before the GPU work
        raise RuntimeError(f"the stream has no round of {n_items} items for rounds {short}")
    bare = [r for r in range(1, rounds + 1) if len(plan[r].get("outcomes") or ()) != n_items
            or U.trained_counts(plan[r]["outcomes"])["by_outcome"] != plan[r]["summary"]["by_outcome"]]
    if bare:  # sets built before revision 3 hold no outcomes: a round cut short could not say what it trained on
        raise RuntimeError(f"the stream does not store each item's outcome for rounds {bare}: build the sets again "
                           "under a new root")
    # The read's pairs, before the model is loaded: prepare_pairs drops a pair whose reference will not run, and a
    # dropped pair would shrink every read of this run without a word. Every planned pair must be there.
    pairs, prep = mc.prepare_pairs({r["name"]: mc.pair_from_row(r) for r in pairs_rows})
    names = list(eval_names)
    gone = [x for x in names if x not in pairs]
    if gone or not names:
        raise RuntimeError(f"{len(gone)} of the {len(names)} read pairs are missing after their references were run "
                           f"({len(prep['dropped'])} dropped): a read must hold every planned pair")
    random.seed(seed)
    torch.manual_seed(seed)
    base, tok, info = load_base(model_key)
    if C.MODELS[model_key].get("ckpt"):  # recompute activations in the backward pass (active only in train mode)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, init, is_trainable=True)
    info["init_adapter"] = init
    info["init_lora_b_abs"] = lora_b_abs(model)
    if not info["init_lora_b_abs"] > 0:  # before any GPU work beyond the load
        raise RuntimeError(f"the init adapter {init} loaded with all-zero lora_B: its keys did not match")
    model.eval()
    player = safe_player(model, tok, max_new=tuple(U.READ["max_new"]), gen_batch=U.READ["gen_batch"])
    params = [p for p in model.parameters() if p.requires_grad]
    dev = next(model.parameters()).device
    out.mkdir(parents=True, exist_ok=True)
    reads, log = {}, []

    def read(step: int) -> dict:
        t0 = time.time()
        torch.manual_seed(U.READ["seed_base"] + step)
        envs = u1_envs(pairs, names, U.READ["n_impossible"], U.READ["n_solvable"], U.READ["attempts"])
        eps, _ = play(player, envs, temp=U.READ["temperature"])
        reads[step] = [u1_slim(e) for e in eps]
        write_jsonl(out / f"read_s{step}.jsonl", reads[step])
        # no summary of the outcomes here, in the log or in what is returned: the readout computes them, and only it
        return {"episodes": len(eps), "s_read": round(time.time() - t0, 1)}

    def note(rec: dict) -> None:
        log.append(rec)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        mc.runs.commit()
        # progress only: no outcome of a read in the container's log (no arm is read before every block is in)
        print(json.dumps({k: rec.get(k) for k in ("kind", "round", "step", "episodes", "s_read", "nll_first",
                                                  "nll_last", "s_train")}), flush=True)

    def reader(before: int, rnd: int, seen: dict):
        """sft_steps' hook for one round: after the k-th step of the round the run has taken before + k steps. When
        that count is a read point, the read is made here, and the training goes on as if it had not been."""
        def after_step(k: int, opt) -> None:  # opt is never touched
            step = before + k
            if step not in reads_at:
                return
            seen["peak_gb"] = max(seen["peak_gb"], peak_gb())  # the training's peak so far, kept apart from the read's
            model.zero_grad(set_to_none=True)  # free the gradients before generating; the next step clears them anyway
            torch.cuda.empty_cache()
            model.eval()
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            reset_peak()
            rec = {"kind": "read", "round": rnd, "step": step, **read(step)}
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:
                torch.cuda.set_rng_state_all(gpu_rng)
            seen["s_read"] += rec["s_read"]
            note(rec | {"peak_gb": peak_gb(), "oom_splits": player.oom_splits})
            reset_peak()
            model.train()

        return after_step

    rec = {"kind": "read", "round": 0, "step": 0, **read(0)}
    held = {arm: u1_doc_nll(model, tok, torch.load(base_dir / "held" / f"{arm}.pt")["items"]) for arm in U.ARMS}
    note(rec | {"held_nll": held, "peak_gb": peak_gb(), "oom_splits": player.oom_splits})
    done = 0
    for r in range(1, rounds + 1):
        owed = min(cfg["steps"], steps - done)  # under cfg["steps"] only in a last round cut short
        take = owed * cfg["batch"]
        seen = {"s_read": 0.0, "peak_gb": 0.0}
        t0 = time.time()
        torch.manual_seed(seed * 1000 + r)
        model.train()
        reset_peak()
        opt, nlls = sft_steps(model, params, plan[r]["items"][:take], tok.pad_token_id, dev, cfg,
                              on_step=reader(done, r, seen))
        model.eval()
        # free the optimizer state and gradients before generating again, as train() does
        del opt
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if len(nlls) != owed:
            raise RuntimeError(f"round {r} took {len(nlls)} optimizer steps where {owed} were owed")
        done += owed
        note({"kind": "train", "round": r, "step": done, "steps": owed, "cut": owed < cfg["steps"],
              "items": U.trained_counts(plan[r]["outcomes"][:take]), "draw": plan[r]["summary"],
              "nll_first": nlls[0] if nlls else None, "nll_last": nlls[-1] if nlls else None,
              "s_train": round(time.time() - t0 - seen["s_read"], 1), "peak_gb": max(seen["peak_gb"], peak_gb()),
              "oom_splits": player.oom_splits})
    missing = [s for s in reads_at if s not in reads]
    if done != steps or missing:
        raise RuntimeError(f"the run took {done} of {steps} steps and made no read at steps {missing}")
    save_adapter(model, out / "adapter")
    info |= {"label": label, "tag": tag, "seed": seed, "stream": stream, "steps": steps, "read_steps": reads_at,
             "rounds": rounds, "cfg": cfg, "read": U.READ, "eval_pairs": len(names), "prep_dropped": prep["dropped"],
             "held_nll": held, "root": root, "snapshot": snapshot, "seconds": round(time.time() - t_start, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    mc.runs.commit()
    return {"label": label, "tag": tag, "seed": seed, "stream": stream, "info": info, "prep_dropped": prep["dropped"],
            "reads": reads, "held_nll": held, "log": log, "seconds": info["seconds"]}
