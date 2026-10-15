#!/usr/bin/env python3
"""Back-compat wrapper: ``python tools/download_models.py`` -> ``pagex models``.

The manifest, the SHA-256 verification and the offline pack installer live in
:mod:`pagex.core.model_manifest`; this script only keeps the old flags working:

    python tools/download_models.py                 # == pagex models download
    python tools/download_models.py --list          # == pagex models list
    python tools/download_models.py --only vlm llm  # == pagex models download --only vlm llm
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main(argv: list[str] | None = None) -> int:
    from pagex.cli import main as cli_main
    args = list(sys.argv[1:] if argv is None else argv)

    if "--list" in args:
        extra = ["--json"] if "--json" in args else []
        return cli_main(["models", "list", *extra])
    if "--only" in args:
        keys = [a for a in args[args.index("--only") + 1:]
                if not a.startswith("-")]
        if keys:
            return cli_main(["models", "download", "--only", *keys])
    rest = [a for a in args if not a.startswith("-")]
    return cli_main(["models", "download", *rest])


if __name__ == "__main__":
    raise SystemExit(main())
