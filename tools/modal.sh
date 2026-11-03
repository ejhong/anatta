#!/usr/bin/env bash
# Run the Modal CLI (lab.toml [modal].cli) with this repo's library importable: the Modal venv has no anatta
# installed, so local entrypoints need src/ on PYTHONPATH. Usage: tools/modal.sh run path/to/app.py::main
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cli="$(cd "$repo" && .venv/bin/python -c 'from anatta.config import load; print(load().modal.cli)')"
cli="${cli/#\~/$HOME}"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$cli" "$@"
