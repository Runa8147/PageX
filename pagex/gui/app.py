"""PageX desktop entry point.

    python -m pagex.gui.app        # or: python cli.py gui
"""
from __future__ import annotations

import argparse
import logging
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

log = logging.getLogger(__name__)


def make_app(argv: list[str] | None = None, theme: str | None = None) -> QApplication:
    """Create the QApplication (tests call this with offscreen platform)."""
    from .theme import apply_theme, saved_theme
    import os
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("PageX")
    app.setOrganizationName("PageX")
    wanted = (theme or os.environ.get("PAGEX_THEME") or saved_theme("light"))
    apply_theme(app, wanted)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pagex-gui", description="PageX desktop shell (offline, PyQt6)")
    parser.add_argument("--db", default=None, help="override SQLite path")
    parser.add_argument("--ingest", nargs="*", default=None,
                        help="files/folders to ingest on startup")
    parser.add_argument("--theme", default=None, choices=["light", "dark"],
                        help="color theme (default: last used)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    from ..core.config import setup_logging
    setup_logging(level=20 if args.verbose else 30)

    app = make_app([], theme=args.theme)
    try:
        from ..core.store import Store
        from .main_window import MainWindow
        store = Store(args.db) if args.db else Store()
        window = MainWindow(store)
    except Exception as exc:                       # model/db problems
        log.exception("startup failed")
        QMessageBox.critical(None, "PageX", f"startup failed:\n{exc}")
        return 1
    window.show()
    if args.ingest:
        from ..core.watch import scan
        from pathlib import Path
        files: list[Path] = []
        for entry in args.ingest:
            p = Path(entry)
            files.extend(scan(p) if p.is_dir() else [p])
        if files:
            window.ingest_paths(files)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
