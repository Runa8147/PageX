"""Console-script shim: ``pagex`` -> the top-level ``cli.py``.

The manual's layout keeps the real CLI at the repository root; this module
exists so ``pip install -e .`` can expose it as a ``pagex`` command without
duplicating a single line of CLI logic.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1] / "cli.py"
    spec = importlib.util.spec_from_file_location("_pagex_cli", root)
    if spec is None or spec.loader is None:        # pragma: no cover
        raise RuntimeError(f"cannot load CLI from {root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(list(argv) if argv is not None else None) or 0)


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
