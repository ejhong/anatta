"""bench's Modal app: one warm L40S container (app anatta-explore-bench, class Bench) that keeps a 7-8B model loaded
with the anatta hooks, so the laptop client (client.py) can send it batches of steering requests. Claude (Opus 5.5)
wrote this file, 1 Nov 2026, and extended it the same day (measure, response_means). Private (lab/explore).

Reused from the lab:
  - modal_common.image, the h4lite image (torch 2.14.1, transformers 5.21.0, anatta-dev/src on PYTHONPATH, the
    Ministral directions folder, player.py and pilot.py), with steering.py added;
  - modal_common.cache, the anatta-hf-cache volume that holds both models' weights. It is mounted read-only here,
    HF_HOME moves to /tmp so nothing tries to write to the volume, and the hub stays offline;
  - anatta.models.load, the loader modal_common.load_model calls (without its PEFT layer), pointed at the pinned
    snapshot folder;
  - anatta.dial's hooks, through steering.py, and player.ego_cut, which the container compares with its own v_ego cut
    the first time a batch asks for one on Ministral.

One container at most (max_containers=1). It takes one input at a time, since the hooks are global to the model, and
stops SCALEDOWN_S after its last input. The model is a class parameter, Bench(model_key="ministral") or
Bench(model_key="qwen"), and each key gets its own container.

Methods: info, generate (steered batches), random_matched (matched-variance random controls), measure (any direction's
variance on the neutral set: other_trait and split_null controls against their target), response_means (replies to
contrastive prompts and their per-response mean activations: the split null's bank) and split_means (the split arms'
per-layer constants on the neutral set, added the evening of 1 Nov). PyTorch's allocator runs with
expandable segments, since the out-of-memory error of 1 Nov had 8 GiB reserved but unallocated.

Deploy: ~/prj/matlab/.venv/bin/modal deploy /Users/eugene/prj/align/anatta-lab/lab/explore/bench/modal_bench.py
(client.py up does this). Nothing here touches any app or volume but anatta-explore-bench and anatta-hf-cache.
"""

import hashlib  # no `from __future__ import annotations`: Modal reads modal.parameter's type from the live annotation
import sys
from pathlib import Path

import modal

BENCH = Path(__file__).resolve().parent
if modal.is_local():
    sys.path.insert(0, str(BENCH.parents[1] / "experiments" / "h4lite"))  # lab/experiments/h4lite
import modal_common as mc  # noqa: E402  (in the container it is /root/modal_common.py, from the image)

APP_NAME = "anatta-explore-bench"
SCALEDOWN_S = 600  # warm for 10 minutes after the last input
CPU, MEMORY_MB = 2.0, 16384  # a request, not a limit: the model passes through host memory once while it loads
EGO_SHA = "a105f23b79fecd23153e36d5cc2911e07647b0dbcd395ce7338efdc79e9cf2f4"  # ego.pt, pinned in lab/ledger/NOTES.md

image = mc.image.add_local_file(str(BENCH / "steering.py"), "/root/steering.py")
app = modal.App(APP_NAME, image=image)


@app.cls(gpu=mc.GPU, cpu=CPU, memory=MEMORY_MB, volumes={mc.CACHE: mc.cache.read_only()},
         scaledown_window=SCALEDOWN_S, max_containers=1, timeout=1200, startup_timeout=1200,
         env={"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
class Bench:
    model_key: str = modal.parameter(default="ministral")

    @modal.enter()
    def load(self):
        import steering

        self.svc = steering.Service.load(self.model_key, hub=f"{mc.CACHE}/hf/hub")
        self.ego_identical = None

    @modal.method()
    def info(self) -> dict:
        return {**self.svc.info(), "ego_cut_identical": self.ego_identical}

    @modal.method()
    def generate(self, payload: dict) -> dict:
        import steering

        try:
            self._ego_check(payload)
            out = self.svc.generate(payload)
        except steering.HookLeak:
            modal.experimental.stop_fetching_inputs()  # a model that may still carry hooks serves nothing more
            raise
        out["meta"]["ego_cut_identical"] = self.ego_identical
        return out

    @modal.method()
    def random_matched(self, payload: dict) -> dict:
        return self.svc.random_matched(payload)

    @modal.method()
    def measure(self, payload: dict) -> dict:
        return self.svc.measure(payload)

    @modal.method()
    def response_means(self, payload: dict) -> dict:
        import steering

        try:
            return self.svc.response_means(payload)
        except steering.HookLeak:
            modal.experimental.stop_fetching_inputs()
            raise

    @modal.method()
    def split_means(self, payload: dict) -> dict:
        import steering

        try:
            return self.svc.split_means(payload)
        except steering.HookLeak:
            modal.experimental.stop_fetching_inputs()
            raise

    def _ego_check(self, payload: dict) -> None:
        """Once per container, on Ministral: the first v_ego ablation in a batch is built both ways, by the bench from
        the vectors the laptop sent and by player.ego_cut from the image's own ego.pt (whose sha256 is checked). On
        the requested layers they must have the same layers and modes and directions equal to float32 rounding (1e-6;
        steering.cut_diff says why they are not bit-identical). The result goes into every batch's meta. A partial
        cut (fractions) is not ego_cut and is skipped."""
        if self.model_key != "ministral" or self.ego_identical is not None:
            return
        iv = next((r["intervention"] for r in payload["requests"]
                   if r["intervention"].get("mode") == "ablate" and r["intervention"].get("sha256") == EGO_SHA
                   and not r["intervention"].get("fractions")), None)
        if iv is None:
            return
        import steering
        from player import ego_cut

        path = Path(mc.EGO) / "ego.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != EGO_SHA:
            raise RuntimeError(f"the image's {path} is not the pinned v_ego")
        key = steering.dir_key(iv)
        sent = self.svc._direction(key, payload["directions"][key])
        mine = steering.build_interventions("ablate", sent, iv["layers"])
        ref = [x for x in ego_cut(mc.EGO) if x.layer in {int(v) for v in iv["layers"]}]
        diff = steering.cut_diff(mine, ref)
        self.ego_identical = {"match": diff <= 1e-6, "max_abs_diff": diff, "layers": [x.layer for x in mine]}
        if not self.ego_identical["match"]:
            raise RuntimeError(f"the bench's v_ego ablation differs from player.ego_cut (max difference {diff})")
