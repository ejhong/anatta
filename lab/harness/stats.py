"""The harness's statistics (spec v2 section 4) now live in anatta.stats, the lab's one stats module (clean-lab step 3):
item-bootstrap paired differences, the minimum-effect, TOST and non-inferiority tests, and fixed-weight composites.
This module re-exports them under their old names so harness code and its tests run unchanged. Claude (Opus 5.5) wrote
this file, 3 Nov 2026.
"""

from anatta.stats import (  # noqa: F401
    ALPHA,
    REPS,
    SEED,
    Est,
    _ci,
    _rng,
    composite,
    composite_level,
    mean_ci,
    min_effect,
    noninferiority,
    orient,
    p_one_sided,
    paired_diff,
    relative_noninf,
    rise,
    tost,
)
