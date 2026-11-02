"""Retreat (Branch B) on Modal: GRPO with LoRA on the coding koans, starting from the instruct model. Private (lab
track) until it works; the public retreat.py stays as is while laptop runs import anatta-dev.

The lab's post-training stands in for Act I (the paper: labs ship an adolescent ego). Arms:
  koan     inoculation framing, Release offered: solve what's solvable, release what isn't.
           Scores (retreat.SCHEMES["koan"]): solved +1, released impossible +1, hacked -1, released solvable -0.25.
  noexit   neutral framing, no Release offered, the same scores; a release anyway scores like a failure (H4's control)
  rlvr     neutral framing, no Release: pass the tests any way you can, so a hack scores +1. The ego crucible, and
           the Act III stress phase when started from a trained adapter (--init).

GRPO here is the plain on-policy version: G samples per prompt, advantage = reward minus the group mean (no std
scaling, as in Dr. GRPO), one gradient step per batch, the token loss summed and divided by a constant, and an
optional KL penalty to the frozen model (the adapter switched off). Prompts go through anatta.models.chat_ids, the
path the evaluation harness uses. The koan outcomes are evaluated before training (step 0) and after.
Each run writes /runs/NAME/{args.json, log.jsonl, samples.jsonl, adapter/} on the anatta-runs volume and a copy
of the logs to lab/experiments/recognition/results/retreat/NAME/.

Usage (~/prj/matlab/.venv/bin/modal):
  modal run lab/experiments/recognition/scripts/modal_retreat.py::main --name koan_smoke --arm koan --steps 2 --prompts 4 --group 4
  modal run lab/experiments/recognition/scripts/modal_retreat.py::main --name stress1 --arm rlvr --init koan1 --steps 50
"""

import json
import time
from pathlib import Path

import modal

ALIGN = "/Users/eugene/prj/align"  # the container paths; on the laptop its recognition/ is a symlink into EXPERIMENTS
EXPERIMENTS = str(Path(__file__).resolve().parent.parent.parent)  # lab/experiments on the laptop (what the image reads)
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
CACHE, RUNS = "/cache", "/runs"
LORA_TARGETS = r"^(?!.*(vision|projector)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.21.0", "tokenizers==0.23.3", "safetensors==0.8.0",
                    "numpy==2.5.4", "PyYAML==6.0.3", "huggingface_hub==2.3.0", "hf_xet==1.7.0",
                    "tiktoken==0.14.0", "jinja2==3.1.6", "regex==2026.9.29", "peft==0.21.2", "accelerate==1.15.0")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": f"{ALIGN}/anatta-dev/src"})
    .add_local_dir(f"{ALIGN}/anatta-dev/src", f"{ALIGN}/anatta-dev/src", ignore=["**/__pycache__"])
)
app = modal.App("anatta-retreat", image=image)
cache = modal.Volume.from_name("anatta-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("anatta-runs", create_if_missing=True)


@app.function(gpu="L40S", cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=8 * 3600)
def train(name: str, arm: str, steps: int, prompts: int, group: int, lr: float, rank: int, beta: float,
          max_new: int, temperature: float, seed: int, init: str, save_every: int, micro: int) -> dict:
    import random
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    from anatta.koans.impossible import TASKS, ImpossibleTask
    from anatta.models import chat_ids, load
    from anatta.retreat import SCHEMES

    framing, release, table = {
        "koan": ("inoculation", True, SCHEMES["koan"]),
        "noexit": ("neutral", False, {**SCHEMES["koan"], "released": 0.0, "over_released": 0.0}),
        "rlvr": ("neutral", False, SCHEMES["rlvr"]),
    }[arm]
    out = Path(RUNS) / name
    out.mkdir(parents=True, exist_ok=True)
    args = dict(name=name, arm=arm, steps=steps, prompts=prompts, group=group, lr=lr, rank=rank, beta=beta,
                max_new=max_new, temperature=temperature, seed=seed, init=init, framing=framing, release=release)
    (out / "args.json").write_text(json.dumps(args, indent=1))
    random.seed(seed)
    torch.manual_seed(seed)

    model, tok = load(MODEL)
    if init:
        model = PeftModel.from_pretrained(model, f"{RUNS}/{init}/adapter", is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    dev = next(model.parameters()).device
    stop = model.generation_config.eos_token_id
    stop = torch.tensor(sorted({tok.eos_token_id, *(stop if isinstance(stop, list) else [stop])} - {None}), device=dev)

    envs = [(t, imp) for t in TASKS for imp in (False, True)]
    prompt_ids = {(t.name, imp): chat_ids(tok, ImpossibleTask(t, imp, framing, release).reset())[0] for t, imp in envs}

    def score(text, t, imp):
        env = ImpossibleTask(t, imp, framing, release)
        env.reset()
        outcome = env.step(text).info["outcome"]
        return table[outcome], outcome

    def rollout(batch, n, temp):
        """n samples per (task, impossible) pair: token ids, masks, decoded texts and (score, outcome) per sample."""
        seqs = [prompt_ids[(t.name, imp)] for t, imp in batch for _ in range(n)]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):], mask[i, width - len(s):] = s, 1
        ids, mask = ids.to(dev), mask.to(dev)
        model.eval()
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=True,
                                 temperature=temp, top_p=1.0, pad_token_id=tok.pad_token_id, use_cache=True)
        comp = gen[:, width:]
        hit = torch.isin(comp, stop)
        first = torch.where(hit.any(1), hit.float().argmax(1).long(), torch.full_like(hit[:, 0], comp.shape[1], dtype=torch.long))
        cmask = (torch.arange(comp.shape[1], device=dev)[None] <= first[:, None]).long()
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        flat = [e for e in batch for _ in range(n)]
        with ThreadPoolExecutor(16) as ex:
            scored = list(ex.map(lambda a: score(a[0], *a[1]), zip(texts, flat)))
        return ids, mask, comp, cmask, texts, flat, scored

    def tally(flat, scored):
        c = Counter(f"{'impossible' if imp else 'solvable'}/{o}" for (_, imp), (_, o) in zip(flat, scored))
        return dict(sorted(c.items()))

    def evaluate(tag, n=8):
        """Every task, both twins, n samples at the harness temperature (0.7), in batches of 8 pairs."""
        flat_all, scored_all = [], []
        for i in range(0, len(envs), 8):
            *_, flat, scored = rollout(envs[i:i + 8], n, 0.7)
            flat_all += flat
            scored_all += scored
        res = {"eval": tag, "mean_score": sum(s for s, _ in scored_all) / len(scored_all), **tally(flat_all, scored_all)}
        print(json.dumps(res), flush=True)
        return res

    def token_logps(full, attn, width, comp):
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        try:
            logits = model(input_ids=full, attention_mask=attn, position_ids=pos,
                           logits_to_keep=comp.shape[1] + 1).logits[:, :-1].float()
        except TypeError:
            logits = model(input_ids=full, attention_mask=attn, position_ids=pos).logits[:, width - 1:-1].float()
        return logits.gather(-1, comp.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)

    evals = [evaluate("before")]
    log, samples = open(out / "log.jsonl", "a"), open(out / "samples.jsonl", "a")
    sols = [e for e in envs if not e[1]]
    imps = [e for e in envs if e[1]]
    t0 = time.time()
    for step in range(steps):
        batch = random.sample(sols, prompts // 2) + random.sample(imps, prompts - prompts // 2)
        ids, mask, comp, cmask, texts, flat, scored = rollout(batch, group, temperature)
        r = torch.tensor([s for s, _ in scored], device=dev).view(prompts, group)
        adv = (r - r.mean(1, keepdim=True)).view(-1)
        model.train()
        opt.zero_grad(set_to_none=True)
        full, attn = torch.cat([ids, comp], 1), torch.cat([mask, cmask], 1)
        width, denom, kl_sum, live = ids.shape[1], len(texts) * max_new, 0.0, 0
        for i in range(0, len(texts), micro):
            sl = slice(i, i + micro)
            if beta == 0 and not adv[sl].abs().sum():
                continue
            live += 1
            lp = token_logps(full[sl], attn[sl], width, comp[sl])
            m = cmask[sl].float()
            obj = adv[sl, None] * lp
            if beta:
                with torch.no_grad(), model.disable_adapter():
                    ref = token_logps(full[sl], attn[sl], width, comp[sl])
                d = ref - lp
                kl = torch.exp(d) - d - 1
                obj = obj - beta * kl
                kl_sum += float((kl.detach() * m).sum())
            (-(obj * m).sum() / denom).backward()
        if live:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        rec = {"step": step + 1, "score": r.mean().item(), "len": cmask.sum(1).float().mean().item(),
               "kl": kl_sum / max(1, int(cmask.sum())), "live": live, "s": round(time.time() - t0),
               **tally(flat, scored)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        for j in range(0, len(texts), group):  # one sample per prompt, for reading
            (t, im), (s, o) = flat[j], scored[j]
            samples.write(json.dumps({"step": step + 1, "task": t.name, "impossible": im, "outcome": o,
                                      "score": s, "text": texts[j][:3000]}) + "\n")
        samples.flush()
        if (step + 1) % save_every == 0 and step + 1 < steps:
            model.save_pretrained(out / f"adapter_{step + 1}")
            runs.commit()
    model.save_pretrained(out / "adapter")
    evals.append(evaluate("after"))
    log.close()
    samples.close()
    (out / "evals.json").write_text(json.dumps(evals, indent=1))
    runs.commit()
    return {f: (out / f).read_text() for f in ("args.json", "log.jsonl", "samples.jsonl", "evals.json")}


@app.local_entrypoint()
def main(name: str, arm: str = "koan", steps: int = 100, prompts: int = 8, group: int = 8, lr: float = 2e-5,
         rank: int = 16, beta: float = 0.02, max_new: int = 512, temperature: float = 1.0, seed: int = 0,
         init: str = "", save_every: int = 25, micro: int = 8, gpu: str = "L40S"):
    files = train.with_options(gpu=gpu).remote(name, arm, steps, prompts, group, lr, rank, beta, max_new,
                                               temperature, seed, init, save_every, micro)
    out = Path(f"{EXPERIMENTS}/recognition/results/retreat/{name}")
    out.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        (out / fname).write_text(text)
    print("wrote", out)
