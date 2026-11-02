"""anatta.train.sft.train, the supervised loop the training apps share, on a tiny random Llama: one record per
optimizer step with the documented keys, the schedule set on the optimizer and recorded, on_epoch after every epoch,
every print_every-th record printed, the data order drawn from the rng (one seed repeats a run exactly), and the
loss as the mean per completion token over the whole batch, so micro-batching leaves the step unchanged. And
anatta.player.attach_adapters' LoRA options (needs peft)."""

import json
import random

import pytest
from modal_stub import tiny_llama

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from anatta.train.sft import train  # noqa: E402

KEYS = ["epoch", "opt_step", "items", "tokens", "nll", "grad_norm", "s", "peak_gb", "elapsed"]
TIMING = ("s", "peak_gb", "elapsed")


def items(n: int = 9, seed: int = 0) -> list:
    """(prompt ids, completion ids) of a few lengths, under the tiny Llama's 64-token vocabulary."""
    g = random.Random(seed)
    return [([1] + [g.randrange(3, 64) for _ in range(4 + i % 3)], [g.randrange(3, 64) for _ in range(2 + i % 2)] + [2])
            for i in range(n)]


def run(model, its, seed=0, opt="adamw", lr=1e-2, **kw):
    params = [p for p in model.parameters() if p.requires_grad]
    o = torch.optim.AdamW(params, lr=lr) if opt == "adamw" else torch.optim.SGD(params, lr=lr)
    recs = []
    n = train(model, its, 0, o, params, random.Random(seed), log=recs.append, **kw)
    return n, recs, o


def test_records_schedule_epochs_and_printing(capsys):
    its = items(9)
    epochs_done = []
    n, recs, opt = run(tiny_llama(), its, epochs=2, batch=4, micro=2, schedule=lambda step: 1e-3 * (step + 1),
                       print_every=2, on_epoch=epochs_done.append)
    assert n == 6 and [r["opt_step"] for r in recs] == [1, 2, 3, 4, 5, 6] and [r["epoch"] for r in recs] == [1, 1, 1, 2, 2, 2]
    assert list(recs[0]) == KEYS[:2] + ["lr"] + KEYS[2:]
    assert [r["lr"] for r in recs] == pytest.approx([1e-3 * k for k in range(1, 7)])
    assert all(g["lr"] == pytest.approx(6e-3) for g in opt.param_groups)  # set before each step
    assert [r["items"] for r in recs] == [4, 4, 1] * 2  # each epoch covers every item once
    assert sum(r["tokens"] for r in recs[:3]) == sum(r["tokens"] for r in recs[3:]) == sum(len(c) for _, c in its)
    assert all(r["grad_norm"] > 0 and r["nll"] > 0 and r["peak_gb"] == 0.0 for r in recs) and epochs_done == [1, 2]
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["opt_step"] for p in printed] == [1, 3, 5] and printed[0] == recs[0]
    # no schedule: no "lr" in the records and the optimizer's rate untouched; no log is fine
    n, recs, opt = run(tiny_llama(), its, lr=5e-3, epochs=1, batch=4, micro=4)
    assert n == 3 and list(recs[0]) == KEYS and all(g["lr"] == 5e-3 for g in opt.param_groups)
    model = tiny_llama()
    params = list(model.parameters())
    assert train(model, its, 0, torch.optim.SGD(params, lr=0.1), params, random.Random(0), epochs=1, batch=9, micro=3) == 1


def test_one_seed_repeats_a_run_and_another_reorders_it():
    its = items(12)
    a = run(tiny_llama(), its, seed=7, epochs=2, batch=4, micro=2)[1]
    b = run(tiny_llama(), its, seed=7, epochs=2, batch=4, micro=2)[1]
    c = run(tiny_llama(), its, seed=8, epochs=2, batch=4, micro=2)[1]
    strip = lambda recs: [{k: v for k, v in r.items() if k not in TIMING} for r in recs]  # noqa: E731
    assert strip(a) == strip(b) and strip(a) != strip(c)


def test_the_loss_is_the_batch_mean_so_micro_batches_leave_the_step_unchanged():
    """One optimizer step over 8 items: in one micro-batch of 8 or eight of 1, the same NLL, gradient norm and
    updated weights (each micro-batch's summed NLL over the batch's completion tokens)."""
    its = items(8, seed=3)
    whole, single = tiny_llama(), tiny_llama()
    _, ra, _ = run(whole, its, opt="sgd", lr=0.5, epochs=1, batch=8, micro=8)
    _, rb, _ = run(single, its, opt="sgd", lr=0.5, epochs=1, batch=8, micro=1)
    assert ra[0]["tokens"] == rb[0]["tokens"] == sum(len(c) for _, c in its)
    assert ra[0]["nll"] == pytest.approx(rb[0]["nll"], rel=1e-5) and ra[0]["grad_norm"] == pytest.approx(rb[0]["grad_norm"], rel=1e-4)
    for (name, p), q in zip(whole.named_parameters(), single.parameters()):
        assert torch.allclose(p, q, atol=1e-6), name
    # the NLL is the mean over completion tokens: a fresh model scores about log(64) a token
    assert ra[0]["nll"] == pytest.approx(torch.log(torch.tensor(64.0)).item(), rel=0.1)


def test_training_lowers_the_loss():
    its = items(8, seed=1)
    _, recs, _ = run(tiny_llama(), its, lr=1e-2, epochs=15, batch=8, micro=4)
    assert recs[-1]["nll"] < 0.5 * recs[0]["nll"]


def test_attach_adapters_shapes_a_fresh_lora_and_keeps_a_loaded_ones_config(tmp_path):
    pytest.importorskip("peft")
    from anatta.player import TRAIN, attach_adapters, save_adapter

    hf = tiny_llama()
    assert attach_adapters(hf, None) is hf and attach_adapters(hf, "") is hf  # no adapter: the plain model

    def scaling(m):
        return m.base_model.model.model.layers[0].self_attn.q_proj.scaling[TRAIN]

    m = attach_adapters(tiny_llama(), "new", None, True, 4, ["q_proj", "v_proj"])
    cfg = m.peft_config[TRAIN]
    assert (cfg.r, cfg.lora_alpha, cfg.use_rslora, cfg.lora_dropout) == (4, 8, False, 0.0) and scaling(m) == pytest.approx(2.0)
    m = attach_adapters(tiny_llama(), "new", None, True, 4, ["q_proj", "v_proj"], alpha=32)
    assert m.peft_config[TRAIN].lora_alpha == 32 and scaling(m) == pytest.approx(8.0)
    m = attach_adapters(tiny_llama(), "new", None, True, 4, ["q_proj", "v_proj"], alpha=32, use_rslora=True)
    assert m.peft_config[TRAIN].use_rslora and scaling(m) == pytest.approx(32 / 4 ** 0.5)
    save_adapter(m, tmp_path / "rs")
    # a loaded adapter keeps the config it was saved with: alpha and use_rslora shape a fresh LoRA only
    loaded = attach_adapters(tiny_llama(), str(tmp_path / "rs"), None, True, 4, None, alpha=4, use_rslora=False)
    assert (loaded.peft_config[TRAIN].lora_alpha, loaded.peft_config[TRAIN].use_rslora) == (32, True)
    assert scaling(loaded) == pytest.approx(16.0) and any(p.requires_grad for p in loaded.parameters())
