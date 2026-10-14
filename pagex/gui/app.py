"""PageX desktop entry point.

    python -m pagex.gui.app        # or: python cli.py gui
"""
from __future__ import annotations

import argparse
import logging
import sys

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox

log = logging.getLogger(__name__)

DARK_QSS = """
QWidget { font-size: 13px; }
QMainWindow, QDialog { background: #f9fafb; }
QTabWidget::pane { border: 1px solid #e5e7eb; background: white; }
QTabBar::tab { padding: 6px 14px; background: #e5e7eb; margin-right: 2px; }
QTabBar::tab:selected { background: white; font-weight: 600; }
QLineEdit, QPlainTextEdit, QListWidget, QTableWidget, QTextBrowser {
    background: white; border: 1px solid #d1d5db; border-radius: 4px;
    padding: 4px; selection-background-color: #bfdbfe;
}
QPushButton { padding: 5px 12px; border: 1px solid #d1d5db;
              border-radius: 4px; background: #ffffff; }
QPushButton:hover { background: #f3f4f6; }
QPushButton:disabled { color: #9ca3af; }
QStatusBar { background: #f3f4f6; }
QMenuBar { background: #f3f4f6; }
"""


def make_app(argv: list[str] | None = None) -> QApplication:
    """Create the QApplication (tests call this with offscreen platform)."""
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("PageX")
    app.setOrganizationName("PageX")
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#f9fafb"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#111827"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(DARK_QSS)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pagex-gui", description="PageX desktop shell (offline, PyQt6)")
    parser.add_argument("--db", default=None, help="override SQLite path")
    parser.add_argument("--ingest", nargs="*", default=None,
                        help="files/folders to ingest on startup")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    from ..core.config import setup_logging
    setup_logging(level=20 if args.verbose else 30)

    app = make_app([])
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
