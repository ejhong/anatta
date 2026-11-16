"""The inside look's Modal app: one warm L40S container (app anatta-explore-look, class Look) that keeps the lab's own
copy of the model loaded, so run_look.py can read saved conversations from inside and steer new ones. Claude (Fable
5.2, an AI model) wrote this file on 16 Nov 2026, on the pattern of lab/explore/bench/modal_bench.py.

Reused from the lab, unchanged:
  - modal_common.image and modal_common.cache (the anatta-hf-cache volume, mounted read-only; the hub stays offline);
  - the bench's steering.Service: the model's load from the pinned snapshot, the interventions (anatta.dial's hooks),
    seeded batched sampling, the matched random directions and the hook checks after every group;
  - attractor_core.view, which sets out a two-copy conversation for the copy whose turn it is.
Added: look_torch.py (rollouts, read, converse).

One container at most, one input at a time (the hooks are global to the model). It stops SCALEDOWN_S after its last
input. Nothing here touches any app or volume but anatta-explore-look and anatta-hf-cache.

Deploy (from the repo root): tools/modal.sh deploy lab/explore/attractor/modal_look.py
"""

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "bench"
if modal.is_local():
    sys.path.insert(0, str(HERE.parents[1] / "experiments" / "h4lite"))  # lab/experiments/h4lite
import modal_common as mc  # noqa: E402  (in the container it is /root/modal_common.py, from the image)

APP_NAME, CLASS_NAME = "anatta-explore-look", "Look"
SCALEDOWN_S = 300  # warm for 5 minutes after the last input
CPU, MEMORY_MB = 2.0, 16384

image = (mc.image.add_local_file(str(BENCH / "steering.py"), "/root/steering.py")
         .add_local_file(str(HERE / "attractor_core.py"), "/root/attractor_core.py")
         .add_local_file(str(HERE / "look_torch.py"), "/root/look_torch.py"))
app = modal.App(APP_NAME, image=image)


@app.cls(gpu=mc.GPU, cpu=CPU, memory=MEMORY_MB, volumes={mc.CACHE: mc.cache.read_only()},
         scaledown_window=SCALEDOWN_S, max_containers=1, timeout=3600, startup_timeout=1200,
         env={"HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
class Look:
    model_key: str = modal.parameter(default="ministral")

    @modal.enter()
    def load(self):
        import steering

        self.svc = steering.Service.load(self.model_key, hub=f"{mc.CACHE}/hf/hub")

    def _guarded(self, fn, *args):
        import steering

        try:
            return fn(*args)
        except steering.HookLeak:
            modal.experimental.stop_fetching_inputs()  # a model that may still carry hooks serves nothing more
            raise

    @modal.method()
    def info(self) -> dict:
        import look_torch

        ctl = sorted(look_torch.control_ids(self.svc.tok))
        return {**self.svc.info(), "control_tokens": len(ctl), "decoder": type(look_torch.decoder_of(self.svc.model)).__name__}

    @modal.method()
    def generate(self, payload: dict) -> dict:
        return self._guarded(self.svc.generate, payload)

    @modal.method()
    def random_matched(self, payload: dict) -> dict:
        return self.svc.random_matched(payload)

    @modal.method()
    def rollouts(self, payload: dict) -> dict:
        import look_torch

        return self._guarded(look_torch.rollouts, self.svc, payload)

    @modal.method()
    def read(self, payload: dict) -> dict:
        import look_torch

        return self._guarded(look_torch.read, self.svc, payload)

    @modal.method()
    def converse(self, payload: dict) -> dict:
        import attractor_core
        import look_torch

        return self._guarded(look_torch.converse, self.svc, payload, attractor_core.view)
