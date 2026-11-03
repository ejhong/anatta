"""The H4-lite apps' torch side, now one implementation: this module *is* anatta.player (src/anatta/player.py).

Until 3 Nov 2026 this folder held its own copy; main's is a superset (alpha and use_rslora on attach_adapters, the
same defaults), and its ENV_ERROR and first_failed equal pilot.py's. The old copy is archive/h4lite/player.py
(lab/LIBRARY.md). `import player` here returns anatta.player itself, so a test that patches player.Player or
player.save_adapter patches what the apps call, as before. This folder's koan_env.EnvError is caught by safe_step
(looked up when it runs, as the old copy's import was) until that copy is retired (clean-lab step 7).
"""

import sys

import anatta.player

if "koan_env" not in anatta.player.ENV_ERROR_MODULES:
    anatta.player.ENV_ERROR_MODULES += ("koan_env",)
sys.modules[__name__] = anatta.player
