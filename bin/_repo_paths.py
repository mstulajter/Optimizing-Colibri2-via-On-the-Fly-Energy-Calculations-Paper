"""Add repository ``bin/`` to ``sys.path`` for scripts under Exhaustive RN/bin or Bounded RN/bin."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_BIN = REPO_ROOT / "bin"


def ensure_repo_bin_on_path() -> Path:
    path = str(REPO_BIN)
    if path not in sys.path:
        sys.path.insert(0, path)
    return REPO_BIN
