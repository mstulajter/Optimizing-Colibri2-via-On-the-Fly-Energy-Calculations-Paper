"""Add repository ``bin/`` to ``sys.path`` (see also top-level ``bin/_repo_paths.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_BIN = REPO_ROOT / "bin"


def ensure_repo_bin_on_path() -> Path:
    path = str(REPO_BIN)
    if path not in sys.path:
        sys.path.insert(0, path)
    return REPO_BIN
