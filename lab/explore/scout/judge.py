"""scout.judge now lives in anatta.judge.rubric (clean-lab step 4a, Claude Opus 5.5, 3 Nov 2026). This module makes
`scout.judge` the same module object, so imports and monkeypatches (tests patch harm_guard.OPEN_WEIGHT) reach the one
copy.
"""

import sys

from anatta.judge import rubric as _moved

sys.modules[__name__] = _moved
