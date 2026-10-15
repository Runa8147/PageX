#!/usr/bin/env python3
"""Back-compat shim: ``python tools/probe.py`` == ``pagex bench``.

The probe now lives inside the package so it also works from a ``pip``
install and from frozen binaries (there is no ``tools/`` next to those).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pagex.probe import main                       # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
