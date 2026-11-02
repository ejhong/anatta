# Copied unchanged from check 1, anatta-calib-c1 (branch calib/model-check), src/anatta/bench/fingerprint.py at commit
# 8d93ad7 (the same file is on calib/relearning). Check 3 uses image_pins and plan_id; CODE_FILES and code_hashes are
# check 1's and unused here (refusal.code_hashes lists check 3's files).
"""What a model-check run's results depend on besides its items and settings: the package code it runs and the
package versions its image pins. Standard library only, so the laptop can fingerprint a plan without importing torch.

  CODE_FILES   the package files whose code a run calls, relative to src/anatta; code_hashes gives their sha256.
               Modules a run only imports along the way (anatta.bench's other modules, koans.env), which define
               names and do nothing else at import, are left out
  image_pins   the "name==version" pins of anatta.modal_app's image, read from that file's source (the
               uv_pip_install calls in its module-level `image = ...`), without importing it
  installed    the installed version of each named package, as importlib.metadata reports it
  plan_id      the first 12 hex digits of a plan's sha256 (its JSON with sorted keys)
"""

from __future__ import annotations

import ast
import hashlib
import json
from importlib import metadata
from pathlib import Path

CODE_FILES = ("models.py", "probe.py", "player.py", "modal_app.py", "bench/battery.py", "bench/items.py",
              "bench/inspect_choice.py", "bench/run.py", "bench/fingerprint.py")


def code_hashes(pkg_dir) -> dict[str, str]:
    """sha256 of the package files whose code a run calls (CODE_FILES, relative to the package folder)."""
    pkg = Path(pkg_dir)
    return {f: hashlib.sha256((pkg / f).read_bytes()).hexdigest() for f in CODE_FILES if (pkg / f).exists()}


def image_pins(modal_app_path) -> dict[str, str]:
    """{package: version} for every "name==version" string passed to uv_pip_install in the first module-level
    assignment to `image` in modal_app.py. Raises if there is none, so a reshaped modal_app is noticed instead of
    fingerprinted as pinning nothing."""
    tree = ast.parse(Path(modal_app_path).read_text())
    pins: dict[str, str] = {}
    node = next((n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "image" for t in n.targets)), None)
    for call in ast.walk(node.value) if node is not None else ():
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "uv_pip_install":
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "==" in arg.value:
                    name, version = arg.value.split("==", 1)
                    pins[name.strip()] = version.strip()
    if not pins:
        raise ValueError(f"no name==version pins in the image of {modal_app_path}")
    return dict(sorted(pins.items()))


def installed(names) -> dict[str, str | None]:
    """The installed version of each package (None if it is not installed)."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def plan_id(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
