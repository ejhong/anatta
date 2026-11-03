"""K6b's transformers path (Modal app anatta-k6b-hf): Ministral 3 8B Instruct served the lab's own way, as K7's
calibration/k7_agentic/app_hf.py does (the h4lite image, lab/explore/bench/steering.py's Service, anatta.models.load
from the pinned snapshot, the model's chat template via anatta.models.chat_ids, per-row seeded sampling, no
intervention), but with each request's own top_p (K7 fixed it at 1.0). Claude (Opus 5.5) wrote this file, 2 Nov 2026.

Used for vLLM's equivalence check (greedy) and as the fallback server. top_k is set past the vocabulary, so sampling
is plain temperature/top_p. Weights read-only from anatta-hf-cache, hub offline; nothing written to any volume.
"""

import sys
from pathlib import Path

import modal

APP_NAME = "anatta-k6b-hf"
if modal.is_local():
    LAB = Path(__file__).resolve().parents[2] / "lab"
    sys.path.insert(0, str(LAB / "experiments" / "h4lite"))
import modal_common as mc  # noqa: E402

if modal.is_local():
    image = mc.image.add_local_file(str(LAB / "explore" / "bench" / "steering.py"), "/root/steering.py")
else:
    image = mc.image
app = modal.App(APP_NAME, image=image)
NO_TOP_K = 10**9
GEN_BATCH = 64


@app.cls(gpu=mc.GPU, cpu=2.0, memory=32768, volumes={mc.CACHE: mc.cache.read_only()}, scaledown_window=120,
         max_containers=1, timeout=3 * 3600, startup_timeout=1200,
         env={"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
class HFGen:
    @modal.enter()
    def load(self):
        import steering

        self.svc = steering.Service.load("ministral", hub=f"{mc.CACHE}/hf/hub", gen_batch=GEN_BATCH)

    @modal.method()
    def info(self) -> dict:
        return self.svc.info()

    @modal.method()
    def gen(self, requests: list[dict], token_budget: int = 80_000) -> list[dict]:
        payload = {"requests": [{"rid": r["rid"], "messages": r["messages"], "temperature": r["temperature"],
                                 "max_tokens": r["max_tokens"], "seed": r.get("seed") or 0,
                                 "top_p": r.get("top_p", 1.0), "top_k": NO_TOP_K,
                                 "intervention": {"mode": "none"}} for r in requests],
                   "token_budget": token_budget}
        out = self.svc.generate(payload)
        return [{"rid": x["rid"], "text": x["text"], "finish_reason": x["finish_reason"],
                 "n_prompt": x["prompt_tokens"], "n_out": x["completion_tokens"]} for x in out["results"]]
