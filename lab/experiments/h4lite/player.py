"""The torch side shared by the H4-lite apps: batched multi-turn generation (Player), packing turns into a padded
batch, per-token log-probs, the v2 ego cut, and the per-episode record. Recognition track; private.

torch is imported inside functions so pilot.py's tests stay torch-free; this module runs inside the Modal containers
and in test_cut_checkpoint.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from pilot import ENV_ERROR, first_failed

REPLY_CHARS = 3000  # how much of each reply the JSONL records keep, for reading
# PEFT adapter names. The trainable adapter is PEFT's "default", which save_pretrained writes at the folder's root
# (any other name goes in a subfolder), so checkpoints keep the adapter_STEP/adapter_config.json layout every
# evaluation app reads. The frozen copy of the start adapter, the KL reference, is "ref" and is never saved.
TRAIN, REF = "default", "ref"


class Player:
    """The model, its generation settings and the multi-turn episode loop, shared by Gate 0, Stage A's sampling,
    Stage B's rollouts and the checkpoint evaluations. `max_new` is per turn: (512, 384) means 512 new tokens on
    the first reply and 384 on every later one (the last entry repeats)."""

    def __init__(self, model, tok, max_new: Sequence[int] | int = (512, 384), gen_batch: int = 32):
        import torch

        self.model, self.tok, self.gen_batch = model, tok, gen_batch
        self.max_new = tuple(max_new) if isinstance(max_new, Sequence) else (int(max_new),)
        self.dev = next(model.parameters()).device
        stop = model.generation_config.eos_token_id
        self.stop = torch.tensor(sorted({tok.eos_token_id, *(stop if isinstance(stop, list) else [stop])} - {None}),
                                 device=self.dev)

    def tokens_for(self, turn: int) -> int:
        """New tokens allowed on reply number `turn` (0-based)."""
        return self.max_new[min(turn, len(self.max_new) - 1)]

    def generate(self, prompts: list, temp: float, max_new: int) -> tuple[list, list]:
        """Completion ids (through the first stop token) and texts, batched shortest-first with left padding."""
        import torch

        tok, ids_out, txt_out = self.tok, [None] * len(prompts), [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        sampling = dict(do_sample=True, temperature=temp, top_p=1.0) if temp > 0 else dict(do_sample=False)
        self.model.eval()
        for k in range(0, len(order), self.gen_batch):
            idx = order[k:k + self.gen_batch]
            width = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, i in enumerate(idx):
                ids[j, width - len(prompts[i]):], mask[j, width - len(prompts[i]):] = prompts[i], 1
            with torch.no_grad():
                gen = self.model.generate(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev),
                                          max_new_tokens=max_new, pad_token_id=tok.pad_token_id, use_cache=True,
                                          **sampling)
            comp = gen[:, width:]
            hit = torch.isin(comp, self.stop)
            ends = torch.where(hit.any(1), hit.float().argmax(1).long() + 1,
                               torch.full_like(hit[:, 0], comp.shape[1], dtype=torch.long))
            for j, i in enumerate(idx):
                ids_out[i] = comp[j, :int(ends[j])].cpu()
                txt_out[i] = tok.decode(ids_out[i], skip_special_tokens=True)
        return ids_out, txt_out

    def play(self, envs: list, temp: float, workers: int = 16) -> tuple[list, list, list]:
        """Run every episode to the end. Per episode: its turns as (prompt ids, completion ids), its reply texts,
        and the final step's info. Each turn's prompt is the chat template's exact token ids for the messages so
        far, so the loss later sees what generation saw. An environment failure (koan_env.EnvError) ends that
        episode with outcome "env_error" (safe_step); the others go on."""
        from concurrent.futures import ThreadPoolExecutor

        from anatta.models import chat_ids

        msgs = [e.reset() for e in envs]
        turns, texts, infos = [[] for _ in envs], [[] for _ in envs], [None] * len(envs)
        live, turn = list(range(len(envs))), 0
        while live:
            prompts = [chat_ids(self.tok, msgs[i])[0] for i in live]
            comps, outs = self.generate(prompts, temp, self.tokens_for(turn))
            with ThreadPoolExecutor(workers) as ex:  # the tests run in subprocesses
                steps = list(ex.map(lambda a: safe_step(envs[a[0]], a[1]), zip(live, outs)))
            for i, p, c, t, st in zip(live, prompts, comps, outs, steps):
                turns[i].append((p, c))
                texts[i].append(t)
                if st.done:
                    infos[i] = st.info
                else:
                    msgs[i] = msgs[i] + [{"role": "assistant", "content": t}, {"role": "user", "content": st.observation}]
            live = [i for i in live if infos[i] is None]
            turn += 1
        return turns, texts, infos


def safe_step(env, reply: str):
    """env.step(reply), with koan_env.EnvError (the pair's reference would not run, twice) turned into a finished
    step whose outcome is "env_error": it says nothing about the model, scores 0 in every arm, carries no advantage
    and leaves every rate (pilot.ENV_ERROR). The error text travels in info["error"]."""
    from koan_env import EnvError, Step

    try:
        return env.step(reply)
    except EnvError as e:
        return Step(None, 0.0, True, {"turn": env.turn, "form": None, "outcome": ENV_ERROR, "error": str(e)[:300]})


def episode_record(env, info: dict, turns: list, texts: list, reply_chars: int = REPLY_CHARS) -> dict:
    """The per-episode JSONL record (pilot.py reads these): outcome, form, grounded, hardcoded, the first-failure
    flag, the assistant's token count, the replies for reading, and the error text of an env_error."""
    n_turns = len(turns)
    return {
        "name": env.name, "pair": env.pair.name, "impossible": env.impossible, "kind": env.pair.mutate[2],
        "outcome": info["outcome"], "turn": info.get("turn", n_turns), "n_turns": n_turns,
        "first_fail": first_failed(n_turns, info["outcome"], info.get("turn", n_turns)),
        "form": info.get("form"), "grounded": info.get("grounded"), "hardcoded": info.get("hardcoded"),
        "reason": info.get("reason"), "error": info.get("error"), "tokens": sum(len(c) for _, c in turns),
        "replies": [t[:reply_chars] for t in texts],
    }


def turn_ids(turns: list) -> list[tuple[list[int], list[int]]]:
    """An episode's turns as plain int lists, for saving."""
    return [([int(x) for x in p], [int(x) for x in c]) for p, c in turns]


def pack(items: list, pad_id: int, device) -> tuple:
    """A micro-batch of (prompt ids, completion ids) as one padded batch: prompts left-padded to a common width,
    completions right-padded. Returns (input ids, attention mask, completion mask [n, wc], wc)."""
    import torch

    ps = [torch.as_tensor(p, dtype=torch.long) for p, _ in items]
    cs = [torch.as_tensor(c, dtype=torch.long) for _, c in items]
    wp, wc = max(len(p) for p in ps), max(len(c) for c in cs)
    full = torch.full((len(items), wp + wc), pad_id, dtype=torch.long)
    attn, cm = torch.zeros_like(full), torch.zeros(len(items), wc)
    for j, (p, c) in enumerate(zip(ps, cs)):
        full[j, wp - len(p):wp], attn[j, wp - len(p):wp] = p, 1
        full[j, wp:wp + len(c)], attn[j, wp:wp + len(c)], cm[j, :len(c)] = c, 1, 1
    return full.to(device), attn.to(device), cm.to(device), wc


def token_logps(model, full, attn, width: int):
    """Log-probs of the last `width` tokens of each row under the model (the completion slots of pack())."""
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    try:
        logits = model(input_ids=full, attention_mask=attn, position_ids=pos, logits_to_keep=width + 1).logits[:, :-1]
    except TypeError:  # a model class without logits_to_keep
        logits = model(input_ids=full, attention_mask=attn, position_ids=pos).logits[:, -width - 1:-1]
    logits = logits.float()
    return logits.gather(-1, full[:, -width:].unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


def attach_adapters(hf_model, train: str | None, ref: str | None = None, trainable: bool = False, rank: int = 16,
                    targets: str | list | None = None):
    """A LoRA on top of an HF model, through PEFT. `train` is a path to load the trainable adapter from, "new" for a
    fresh rank-`rank` LoRA (seed torch first: the seed sets its initialization), or None for the plain model. `ref`
    is a path loaded a second time as the frozen "ref" adapter, the KL reference: with a warm start the reference is
    the Stage A start the run began from, not the untrained base, and on a resume it is still the original start
    while "default" comes from the last checkpoint. Only "default" is ever trainable or saved."""
    if not train:
        return hf_model
    if train == "new":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(hf_model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                                    target_modules=targets, task_type="CAUSAL_LM"))
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(hf_model, train, adapter_name=TRAIN, is_trainable=trainable)
    if ref:
        model.load_adapter(ref, adapter_name=REF, is_trainable=False)
    model.set_adapter(TRAIN, inference_mode=not trainable)  # requires_grad on "default" only (and only if trainable)
    return model


@contextmanager
def reference(model):
    """The model as the KL reference: the frozen "ref" adapter when the model carries one (a warm or resumed start),
    else the base model with the adapter off (a fresh LoRA's start, since its B matrix is zero). Switching adapters
    also switches which weights require grad, so run this before the policy forward of a micro-batch, under
    torch.no_grad(), never between a forward and its backward."""
    if REF in getattr(model, "peft_config", {}):
        model.set_adapter(REF, inference_mode=True)
        try:
            yield
        finally:
            model.set_adapter(TRAIN)
    else:
        with model.disable_adapter():
            yield


def save_adapter(model, path: str | Path) -> None:
    """Write the trainable adapter alone (never "ref") at `path`, in PEFT's root layout."""
    model.save_pretrained(str(path), selected_adapters=[TRAIN])


def ego_cut(ddir: str | Path) -> list:
    """The v2 ego cut: v_ego ablated at every extracted layer (8-32 on Ministral 3 8B), exactly as cut_vs_push.py's
    `cut` arm builds it. Ablate: h <- h - (h.d) d, batch-safe and gradient-transparent (anatta.dial)."""
    from anatta.dial import Intervention
    from anatta.probe import Direction

    ego = Direction.load(Path(ddir) / "ego.pt")
    return [Intervention(layer=i, direction=ego.vectors[i], mode="ablate") for i in sorted(ego.vectors)]


def peak_gb() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0


def reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
