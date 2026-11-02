"""Check 4's test setup: this check's folder on sys.path before any test module imports from it, and the fixtures of
c4_testing (the tiny Llama of tests/conftest.py, loaded there by path, and the stand-in for pandas)."""

import sys
from pathlib import Path

CHECK = Path(__file__).resolve().parent.parent
if str(CHECK) not in sys.path:
    sys.path.insert(0, str(CHECK))

from c4_testing import fake_pandas, tiny  # noqa: E402, F401 (fixtures)
