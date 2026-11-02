"""K7's Modal app (anatta-k7), transformers half: Ministral 3 8B Instruct (the lab's main model,
mistralai/Ministral-3-8B-Instruct-2512-BF16 at f6fae979, as pinned in lab/explore/bench/steering.py and
calibration/model_check/config.yaml). Claude (Opus 5.5) wrote this file, 2 Nov 2026.

vLLM 0.10 predates Ministral 3, so this half serves it the lab's own way: the h4lite image (modal_common.image:
torch 2.14.1, transformers 5.21.0, anatta-dev at the pinned tag) with lab/explore/bench/steering.py, and
steering.Service (anatta.models.load from the pinned snapshot, the model's chat template via anatta.models.chat_ids,
per-row seeded sampling) with no intervention. Ministral's own template, with our system message (so its default
Le Chat system prompt is not added). top_k is set past the vocabulary, so sampling is plain temperature 1.0, top_p 1.
Weights read-only from anatta-hf-cache, hub offline; nothing written to any volume. Never touches other apps.
"""

import sys
from pathlib import Path

import modal

APP_NAME = "anatta-k7"
if modal.is_local():
    LAB = Path(__file__).resolve().parents[2] / "lab"
    sys.path.insert(0, str(LAB / "experiments" / "h4lite"))
import modal_common as mc  # noqa: E402  (in the container: /root/modal_common.py, from the image)

if modal.is_local():
    image = mc.image.add_local_file(str(LAB / "explore" / "bench" / "steering.py"), "/root/steering.py")
else:
    image = mc.image  # in the container the image is already built; this object is not used to build
app = modal.App(APP_NAME, image=image)
NO_TOP_K = 10**9


@app.cls(gpu=mc.GPU, cpu=2.0, memory=32768, volumes={mc.CACHE: mc.cache.read_only()}, scaledown_window=120,
         max_containers=1, timeout=3 * 3600, startup_timeout=1200,
         env={"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
class HFGen:
    @modal.enter()
    def load(self):
        import steering

        self.svc = steering.Service.load("ministral", hub=f"{mc.CACHE}/hf/hub", gen_batch=32)

    @modal.method()
    def info(self) -> dict:
        return self.svc.info()

    @modal.method()
    def gen(self, requests: list[dict], token_budget: int = 80_000) -> list[dict]:
        payload = {"requests": [{"rid": r["rid"], "messages": r["messages"], "temperature": r["temperature"],
                                 "max_tokens": r["max_tokens"], "seed": r["seed"], "top_p": 1.0, "top_k": NO_TOP_K,
                                 "intervention": {"mode": "none"}} for r in requests],
                   "token_budget": token_budget}
        out = self.svc.generate(payload)
        return [{"rid": x["rid"], "text": x["text"], "finish_reason": x["finish_reason"],
                 "n_prompt": x["prompt_tokens"], "n_out": x["completion_tokens"], "max_tokens": r["max_tokens"],
                 "rev": self.svc.revision} for x, r in zip(out["results"], requests)]
