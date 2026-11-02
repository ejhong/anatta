"""noself's test setup: this folder's parent and K5's folder on sys.path, and K5's tiny random model fixture reused.
Nothing here touches Modal, a GPU, the network or a real model. Claude (Opus 5.5) wrote this file, 2 Nov 2026."""

import importlib.util
import sys
from pathlib import Path

NOSELF = Path(__file__).resolve().parent.parent
REPO = NOSELF.parents[2]
K5DIR = REPO / "calibration" / "kim_consciousness"
for p in (NOSELF, K5DIR, NOSELF.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("k5_conftest_for_noself", K5DIR / "tests" / "conftest.py")
_k5conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_k5conf)
tiny = _k5conf.tiny
