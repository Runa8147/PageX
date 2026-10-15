#!/usr/bin/env python3
"""PageX CLI shim: ``python cli.py ...`` == ``pagex ...`` == ``python -m pagex``.

The implementation lives in :mod:`pagex.cli` so the installed console script,
the frozen binary and this repository-root entry point all run exactly one
copy of the code (ADR: the CLI is a thin shell over ``pagex.core``).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pagex.cli import build_parser, main            # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
