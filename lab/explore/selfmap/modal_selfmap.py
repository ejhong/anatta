"""The self map's Modal app (anatta-selfmap): per-layer activations of the dev prompts on one model per container.
Claude (Opus 5.5, an AI model) wrote this file, 2 Nov 2026. Private (lab/explore). Explore speed: sightings only.

Reused: modal_common.image (the h4lite image: torch 2.14.1, transformers 5.21.0, anatta-dev/src on PYTHONPATH) and
anatta.models.load (bf16 on the GPU, with Ministral 3's vision-checkpoint fallback), as bench loads its models. The
weights are read from two volumes, both mounted read-only, with the hub offline (nothing is downloaded):
  anatta-hf-cache      Ministral-3-8B-Instruct-2512-BF16 and Qwen2.5-7B-Instruct (bench's pinned revisions)
  anatta-calib-prh-k1  Qwen2.5-7B base (K1's pinned revision)
No other app or volume is touched (never any matlab-* one).

Readouts, at every block's output (the residual stream after block i, i = 0..L-1, via anatta.probe.capture):
  mean   the masked mean over the prompt's tokens, special tokens (BOS) left out. Departure from K1's
         features.embed_texts, which averages every token and reads hidden_states (embedding output included, final
         norm on the last): with a BOS token, its large residual divided by the length would add a length signal,
         and "I" is shorter than "The robot arm".
  mean1  the same without the first non-special token as well: position 0's attention-sink activations, divided by
         the length, are the same worry in models without a BOS (Qwen), and the first token is the referent's.
  last   the last non-pad token (the final full stop).
Right padding, so no real token sees a pad. Each is returned as float16 [N, L, d].

Run (from the laptop): see run_pilot.py, which calls Extractor(model_key=...).run and .get.
"""

import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if modal.is_local():
    sys.path.insert(0, str(HERE.parents[1] / "experiments" / "h4lite"))
import modal_common as mc  # noqa: E402

APP_NAME = "anatta-selfmap"
K1_MOUNT = "/k1"
MODELS = {
    "ministral": f"{mc.CACHE}/hf/hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/"
                 "f6fae9795746f63c9be8344932f01275f3c63734",
    "qwen_instruct": f"{mc.CACHE}/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
                     "a09a35458c702b33eeacc393d103063234e8bc28",
    "qwen_base": f"{K1_MOUNT}/hf/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796",
}
GPU_USD_H, CPU_MEM_USD_H = 1.95, 0.6  # L40S, plus 2 cores and 32 GiB at Modal's list prices (generous)

app = modal.App(APP_NAME, image=mc.image)
k1 = modal.Volume.from_name("anatta-calib-prh-k1")


@app.cls(gpu=mc.GPU, cpu=2.0, memory=32768, volumes={mc.CACHE: mc.cache.read_only(), K1_MOUNT: k1.read_only()},
         scaledown_window=60, max_containers=1, timeout=1800, startup_timeout=1200,
         env={"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
class Extractor:
    model_key: str = modal.parameter(default="qwen_base")

    @modal.enter()
    def load(self):
        from anatta.models import get_layers, load

        t0 = time.time()
        self.model, self.tok = load(MODELS[self.model_key])
        self.tok.padding_side = "right"
        self.n_layers = len(get_layers(self.model))
        self.load_s = time.time() - t0
        self.out = {}

    @modal.method()
    def run(self, texts: list[str], batch: int = 32) -> dict:
        import numpy as np
        import torch
        import transformers
        from anatta.probe import capture

        t0 = time.time()
        layers = list(range(self.n_layers))
        special = torch.tensor(sorted(set(self.tok.all_special_ids)), device="cuda")
        acc = {"mean": [], "mean1": [], "last": []}
        ntok = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], return_tensors="pt", padding=True).to("cuda")
                ids, att = enc["input_ids"], enc["attention_mask"]
                real = att.bool() & ~torch.isin(ids, special)
                first = real & (real.int().cumsum(1) == 1)
                real1 = real & ~first
                last_idx = att.sum(1) - 1
                ntok.extend(real.sum(1).tolist())
                with capture(self.model, layers) as store:
                    self.model(input_ids=ids, attention_mask=att)
                hs = torch.stack([store[x].float() for x in layers], 1)  # [B, L, T, d]
                for name, m in (("mean", real), ("mean1", real1)):
                    w = m[:, None, :, None].float()
                    acc[name].append(((hs * w).sum(2) / w.sum(2)).clamp(-65000, 65000).half().cpu())
                acc["last"].append(hs[torch.arange(len(ids)), :, last_idx].clamp(-65000, 65000).half().cpu())
                del hs, store
        self.out = {k: torch.cat(v).numpy() for k, v in acc.items()}
        cfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
        return {"model_key": self.model_key, "path": MODELS[self.model_key], "class": type(self.model).__name__,
                "n_layers": self.n_layers, "hidden": int(cfg.hidden_size), "rows": len(texts),
                "dtype": str(next(self.model.parameters()).dtype), "torch": torch.__version__,
                "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
                "load_s": round(self.load_s, 1), "extract_s": round(time.time() - t0, 1),
                "special_ids": special.tolist(), "bos_added": bool(self.tok("I")["input_ids"][0] in special.tolist()),
                "tokens": {"min": int(min(ntok)), "max": int(max(ntok)), "mean": float(np.mean(ntok))},
                "ntok": [int(x) for x in ntok],
                "finite": {k: bool(np.isfinite(v).all()) for k, v in self.out.items()},
                "saturated": {k: int((np.abs(v) >= 65000).sum()) for k, v in self.out.items()}}

    @modal.method()
    def get(self, readout: str, lo: int, hi: int):
        return self.out[readout][:, lo:hi]
