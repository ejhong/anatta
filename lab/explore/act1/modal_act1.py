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

All model-written code runs here, inside the container (koan_env's subprocess runner), never on the Mac. Only
anatta-runs (written, under /runs/act1/ and, by gate, /runs/gate/) and anatta-calib-prh-k1 and anatta-hf-cache
(read-only) are touched.
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
         .add_local_file(str(HERE.parent / "gate" / "gate_core.py"), "/root/gate_core.py"))  # the looking gate
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
