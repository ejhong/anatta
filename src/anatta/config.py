"""The lab's one config: lab.toml at the repo root, read with stdlib tomllib, plus env overrides. Stdlib only, so it
imports anywhere (laptop venvs, containers). Claude (Opus 5.5) wrote this file, 3 Nov 2026.

No other module holds a cap, a results path or a Modal name; they ask `load()`.

Env overrides (each optional):
  ANATTA_LAB_TOML        the config file's path (default: <repo>/lab.toml)
  ANATTA_RESULTS_ROOT    results_root
  ANATTA_SPEND_DIR       the spend day files' folder
  ANATTA_DAILY_CAP_USD   the default daily cap (a day file's own cap_usd still wins; see anatta.spend)
  ANATTA_DEFAULT_MODEL   default_model
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "lab.toml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Modal:
    app_prefix: str
    cli: Path
    volumes: dict[str, str] = field(default_factory=dict)  # role -> volume name

    def app_name(self, name: str) -> str:
        """`<prefix><name>`; a name that already carries the prefix is returned as is."""
        return name if name.startswith(self.app_prefix) else f"{self.app_prefix}{name}"


@dataclass(frozen=True)
class Config:
    path: Path
    results_root: Path
    spend_dir: Path
    daily_cap_usd: float
    default_model: str
    modal: Modal
    key_envs: dict[str, str] = field(default_factory=dict)  # service -> env var name (never the value)
    manifest_exclude: tuple[str, ...] = ()  # globs relative to results_root left out of results/MANIFEST.tsv


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"lab.toml: missing `{dotted}`")
        cur = cur[part]
    return cur


def _path(s: str, base: Path | None = None) -> Path:
    p = Path(os.path.expanduser(s))
    return p if p.is_absolute() or base is None else base / p


def load(path: str | os.PathLike | None = None) -> Config:
    """Read lab.toml (or `path`, or $ANATTA_LAB_TOML) and apply the env overrides."""
    env = os.environ
    p = Path(path or env.get("ANATTA_LAB_TOML") or DEFAULT_PATH)
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"no lab config at {p}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{p}: {e}") from None

    root = _path(env.get("ANATTA_RESULTS_ROOT") or _get(raw, "results_root"))
    spend_dir = _path(env.get("ANATTA_SPEND_DIR") or _get(raw, "spend.dir"), root)
    cap = float(env.get("ANATTA_DAILY_CAP_USD") or _get(raw, "spend.daily_cap_usd"))
    if not cap > 0:
        raise ConfigError(f"spend.daily_cap_usd must be above 0; got {cap}")
    m = _get(raw, "modal")
    modal = Modal(app_prefix=str(_get(m, "app_prefix")), cli=_path(str(_get(m, "cli"))),
                  volumes={str(k): str(v) for k, v in (m.get("volumes") or {}).items()})
    keys = {k.removesuffix("_env"): str(v) for k, v in (raw.get("keys") or {}).items()}
    return Config(path=p, results_root=root, spend_dir=spend_dir, daily_cap_usd=cap,
                  default_model=str(env.get("ANATTA_DEFAULT_MODEL") or _get(raw, "default_model")), modal=modal,
                  key_envs=keys, manifest_exclude=tuple(str(x) for x in raw.get("manifest_exclude") or ()))
