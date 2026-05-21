#!/usr/bin/env bash
# Run the AG News demo. Installs the [demo] extras on first run if needed.
#
# Usage:
#   ./examples/agnews/run.sh                  # full demo (both architectures)
#   ./examples/agnews/run.sh --sklearn-only   # fast variant only (~30s-2min)
#   ./examples/agnews/run.sh --distilbert-only
#
# Pick the Python interpreter via $PYTHON env var if `python3` on PATH is not
# the one you want. Example for pyenv users:
#   PYTHON=~/.pyenv/versions/3.11.13/bin/python ./examples/agnews/run.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

# Auto-activate .venv/ if it exists and no venv is currently active.
# Skipped when $PYTHON is set explicitly — that's a deliberate override.
if [[ -z "${VIRTUAL_ENV:-}" && -z "${PYTHON:-}" && -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: no '$PYTHON' on PATH. Set \$PYTHON to your interpreter, e.g.:" >&2
  echo "  PYTHON=~/.pyenv/versions/3.11.13/bin/python $0 $*" >&2
  exit 1
fi

if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
  echo "ERROR: ml-baker needs Python 3.10+. Found: $($PYTHON --version 2>&1)" >&2
  exit 1
fi

# Probe for the heavy demo deps before installing — keeps re-runs fast.
if ! "$PYTHON" -c "import datasets, sklearn, transformers" 2>/dev/null; then
  echo "Installing ml-baker[demo] dependencies (one-time, ~2GB download)..."
  "$PYTHON" -m pip install -e ".[demo]"
fi

echo "Running AG News demo..."
exec "$PYTHON" examples/agnews/demo.py "$@"
