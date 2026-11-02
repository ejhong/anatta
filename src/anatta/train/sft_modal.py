"""Supervised LoRA fine-tuning on Modal, on a prepared training set of multi-turn episodes (the loop is
anatta.train.sft).

finetune  LoRA rank 16, lr 1e-4 (LR, tuned for the default LoRA: with --alpha or --use-rslora, --lr must be
          given), 2 epochs, the loss on the assistant's replies only (each turn is one item: its prompt is masked, its
          completion is the target; prompts and environment feedback never carry loss). The seed sets the LoRA
          initialization and the data order, so several seeds can share one training set.
--under-cut   the ego cut on throughout: v_ego ablated at layers 8-32 (anatta.player.ego_cut), applied with
          anatta.dial.intervene() around the whole loop, forward and backward inside the block, so gradient
          checkpointing's recomputed forward pass keeps the cut. data.json's cut_finetune, when it has one, must agree.
          Without the model's ego.pt (anatta.modal_app.require_ego) it refuses, on the laptop before any container
          starts and in the container before the model loads.

The training set is LABEL/{train.pt, data.json} under /runs/h4lite/sft/ on the anatta-runs volume (or --src-dir):
train.pt holds {"records": the kept episodes, "turns": per episode its turns as (prompt ids, completion ids), the
chat template's exact token ids for the model being trained}, and data.json what was kept (its cut_sampling and
cut_finetune, when present, go into args.json). Writes /runs/h4lite/sft/LABEL_sSEED/{adapter/, args.json, log.jsonl,
done.json} (or --out-dir); local copies of the text files go to results/sft/. Another model (--model) reads and
writes under /runs/h4lite/<model slug>/sft and results/<model slug>/sft (anatta.modal_app.run_root).

Usage (from the repo root):
  PYTHONPATH=src modal run src/anatta/train/sft_modal.py::main --labels koan,noexit --seeds 0,1,2
  PYTHONPATH=src modal run src/anatta/train/sft_modal.py::main --labels koan_cut --under-cut --seeds 0,1,2
"""

from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import modal

from anatta.modal_app import (
    CACHE,
    GPU,
    MODEL,
    RUNS,
    append_jsonl,
    cache,
    ego_dir,
    image,
    load_model,
    lr_for,
    names_of,
    require_ego,
    results_dir,
    run_root,
    runs,
    seeds_of,
    write_json,
)

app = modal.App("anatta-h4lite-sft", image=image)
LR = 1e-4  # the default learning rate, tuned for the default LoRA (alpha = 2 x rank)


@app.function(gpu=GPU, cpu=8.0, memory=65536, volumes={CACHE: cache, RUNS: runs}, timeout=6 * 3600)
def finetune(label: str, seed: int, under_cut: bool = False, lr: float | None = None, rank: int = 16, epochs: int = 2,
             micro: int = 4, batch: int = 16, src_dir: str = "", out_dir: str = "", init: str = "new",
             save_epochs: bool = False, model: str = MODEL, alpha: int | None = None, use_rslora: bool = False) -> dict:
    """LoRA fine-tuning on LABEL/train.pt: every turn of every kept episode is one item, its prompt masked and its
    completion the target. `batch` items per optimizer step in micro-batches of `micro`; the loss is the mean
    negative log-likelihood per completion token in the batch (anatta.train.sft.train). `under_cut` keeps the ego
    cut on for the whole loop; without it the loop runs on the plain model, no intervene context (a fine-tune asked
    for the other cut condition than its data.json records is refused). A rerun starts over: the log is truncated
    first.

    `src_dir` and `out_dir` stand in for ROOT/sft/LABEL and ROOT/sft/LABEL_sSEED; `init` is an adapter path whose
    LoRA keeps training (loaded trainable, with a fresh optimizer and this seed's data order) instead of a fresh LoRA
    ("new", seeded); `save_epochs` also writes adapter_epochK/ after every epoch, the volume committed each time.
    `model`, `alpha` and `use_rslora` go to anatta.modal_app.load_model. args.json records the first four only when
    one of them is set, and the last three only when one differs from its default. `lr` defaults to LR for the
    default LoRA only (anatta.modal_app.lr_for)."""
    import torch

    from anatta.dial import intervene
    from anatta.player import ego_cut, save_adapter
    from anatta.train.sft import train

    lr = lr_for(lr, LR, alpha, use_rslora)
    src = Path(src_dir) if src_dir else Path(run_root(model)) / "sft" / label
    out = Path(out_dir) if out_dir else Path(run_root(model)) / "sft" / f"{label}_s{seed}"
    info = json.loads((src / "data.json").read_text())
    if info.get("cut_finetune") not in (None, under_cut):
        raise SystemExit(f"{label}'s data.json was prepared for cut_finetune={info['cut_finetune']}; this fine-tune "
                         f"was asked for {under_cut}")
    if under_cut:
        require_ego(model)  # before anything is written or loaded
    out.mkdir(parents=True, exist_ok=True)
    data = torch.load(src / "train.pt")
    items = [(p, c) for t in data["turns"] for p, c in t]
    args = dict(label=label, seed=seed, cut_sampling=info.get("cut_sampling"), cut_finetune=under_cut, lr=lr,
                rank=rank, epochs=epochs, micro=micro, batch=batch, episodes=len(data["records"]), items=len(items),
                tokens=sum(len(c) for _, c in items), data=info)
    if src_dir or out_dir or init != "new" or save_epochs:
        args.update(src_dir=str(src), out_dir=str(out), init=init, save_epochs=save_epochs)
    if model != MODEL or alpha is not None or use_rslora:
        args.update(model=model, alpha=alpha, use_rslora=use_rslora)
    write_json(out / "args.json", args)
    for stale in ("log.jsonl", "done.json"):
        (out / stale).unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in args.items() if k != "data"}), flush=True)

    random.seed(seed)
    torch.manual_seed(seed)  # the LoRA initialization (a loaded adapter has none to draw)
    lm, tok, hf_model = load_model(adapter=init, trainable=True, rank=rank, model=model, alpha=alpha,
                                   use_rslora=use_rslora)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    cut = ego_cut(ego_dir(model)) if under_cut else []
    rng = random.Random(seed)  # the data order

    def on_epoch(k: int) -> None:  # an adapter after every epoch as well as at the end
        save_adapter(lm, out / f"adapter_epoch{k}")
        runs.commit()

    t_start = time.time()
    # The cut's hooks around the whole loop, forward and backward both inside (the recomputed forward keeps the cut);
    # without a cut the plain model trains, no context at all.
    with intervene(hf_model, cut) if cut else nullcontext():
        n_opt = train(lm, items, tok.pad_token_id, opt, params, rng, epochs=epochs, batch=batch, micro=micro,
                      log=lambda rec: append_jsonl(out / "log.jsonl", rec), print_every=10,
                      on_epoch=on_epoch if save_epochs else None)
    save_adapter(lm, out / "adapter")
    write_json(out / "done.json", {"label": label, "seed": seed, "opt_steps": n_opt, "seconds": round(time.time() - t_start)})
    runs.commit()
    return {"name": f"{label}_s{seed}", "files": {f: (out / f).read_text() for f in ("args.json", "log.jsonl", "done.json")}}


def save_local(name: str, files: dict, model: str = MODEL) -> Path:
    dest = results_dir(model) / "sft" / name
    dest.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        (dest / fname).write_text(text)
    return dest


@app.local_entrypoint()
def main(labels: str = "koan,noexit", seeds: str = "0,1,2", under_cut: bool = False, lr: float | None = None, rank: int = 16,
         epochs: int = 2, micro: int = 4, batch: int = 16, init: str = "new", save_epochs: bool = False,
         model: str = MODEL, alpha: int | None = None, use_rslora: bool = False, gpu: str = GPU):
    """One adapter per (label, seed) on the prepared training sets, in parallel containers; the text files are
    copied to results/sft/LABEL_sSEED/. --lr defaults to LR, tuned for the default LoRA; with --alpha or --use-rslora
    it must be given."""
    lr = lr_for(lr, LR, alpha, use_rslora)
    if under_cut:
        require_ego(model, local=True)  # what the image carries: refuse before any container starts
    calls = [finetune.with_options(gpu=gpu).spawn(label, s, under_cut, lr, rank, epochs, micro, batch, init=init,
                                                  save_epochs=save_epochs, model=model, alpha=alpha,
                                                  use_rslora=use_rslora)
             for label in names_of(labels) for s in seeds_of(seeds)]
    for c in calls:
        res = c.get()
        print("wrote", save_local(res["name"], res["files"], model), flush=True)
