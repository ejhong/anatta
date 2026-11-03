"""scout.cache now lives in anatta.judge.cache (clean-lab step 4a, Claude Opus 5.5, 3 Nov 2026). This module makes
`scout.cache` the same module object, so imports and monkeypatches (tests patch harm_guard.OPEN_WEIGHT) reach the one
copy.
"""

import sys

from anatta.judge import cache as _moved

sys.modules[__name__] = _moved
