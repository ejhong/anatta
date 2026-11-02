"""K1's test setup: this check's folder on sys.path before any test module imports from it."""

import sys
from pathlib import Path

CHECK = Path(__file__).resolve().parent.parent
if str(CHECK) not in sys.path:
    sys.path.insert(0, str(CHECK))
