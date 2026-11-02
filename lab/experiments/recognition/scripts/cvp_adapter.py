"""cut_vs_push.py with a trained LoRA adapter merged into the model. Recognition track; private.

The adapter is merged before any arm's hooks go on, so the cuts and pushes act on the trained model exactly as they
act on the base one.

Usage: python cvp_adapter.py ADAPTER_DIR <cut_vs_push.py arguments>
"""

import runpy
import sys
from pathlib import Path

import anatta.models as models

adapter = sys.argv.pop(1)
_load = models.load


def load(model_id, *args, **kwargs):
    from peft import PeftModel

    model, tok = _load(model_id, *args, **kwargs)
    return PeftModel.from_pretrained(model, adapter).merge_and_unload().eval(), tok


models.load = load  # cut_vs_push.py does `from anatta.models import load`, so patch before it runs
script = Path(__file__).with_name("cut_vs_push.py")
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
