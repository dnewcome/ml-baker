"""Make the repo root and this tests/ dir importable.

The probe layer resolves user callables by dotted path and imports them in
the running process (``launcher="in_process"`` in tests). Putting both the
repo root (so ``mlprof`` / ``examples`` resolve) and this directory (so
``synthetic_trainable`` resolves) on ``sys.path`` keeps the dotted paths in
the test specs working regardless of how pytest is invoked.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

for p in (_REPO_ROOT, _TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
