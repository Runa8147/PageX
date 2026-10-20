"""PageX professional theme: 70% neutral / 20% ink / 10% blue accent.

Light (default): white surfaces, near-black text, blue primary actions.
Dark: same accent (brightened for contrast), dark surfaces, light text.

Usage:
    from .theme import apply_theme, current_theme_name, toggle_theme
    apply_theme(app, "light")
"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

ACCENT = "#2563eb"        # 10% - primary actions, selection, links
ACCENT_HOVER = "#1d4ed8"
ACCENT_DARK = "#3b82f6"   # brightened for dark surfaces
ACCENT_DARK_HOVER = "#60a5fa"

LIGHT_BG = "#ffffff"      # 70%
LIGHT_BG_SOFT = "#f8fafc"
LIGHT_BORDER = "#e2e8f0"
LIGHT_INK = "#0f172a"     # 20%
LIGHT_MUTED = "#64748b"

DARK_BG = "#0f172a"       # 70% (inverted)
DARK_BG_SOFT = "#1e293b"
DARK_BORDER = "#334155"
DARK_INK = "#f1f5f9"      # 20% (inverted)
DARK_MUTED = "#94a3b8"

THEMES = ("light", "dark", "system")


def current_theme_name(app: QApplication | None = None) -> str:
    app = app or QApplication.instance()
    if app is None:
        return "light"
    return str(app.property("pagexTheme") or "light")


def _base_palette(light: bool) -> QPalette:
    pal = QPalette()
    if light:
        pal.setColor(QPalette.ColorRole.Window, QColor(LIGHT_BG_SOFT))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(LIGHT_INK))
        pal.setColor(QPalette.ColorRole.Base, QColor(LIGHT_BG))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(LIGHT_BG_SOFT))
        pal.setColor(QPalette.ColorRole.Text, QColor(LIGHT_INK))
        pal.setColor(QPalette.ColorRole.Button, QColor(LIGHT_BG))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(LIGHT_INK))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(LIGHT_INK))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#ffffff"))
    else:
        pal.setColor(QPalette.ColorRole.Window, QColor(DARK_BG))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(DARK_INK))
        pal.setColor(QPalette.ColorRole.Base, QColor(DARK_BG_SOFT))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(DARK_BG))
        pal.setColor(QPalette.ColorRole.Text, QColor(DARK_INK))
        pal.setColor(QPalette.ColorRole.Button, QColor(DARK_BG_SOFT))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(DARK_INK))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_DARK))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(DARK_INK))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor(DARK_BG))
    return pal


LIGHT_QSS = f"""
QWidget {{ font-size: 13px; }}
QMainWindow, QDialog {{ background: {LIGHT_BG_SOFT}; }}
QMenuBar {{ background: {LIGHT_BG_SOFT}; color: {LIGHT_INK}; spacing: 4px; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {LIGHT_BORDER}; border-radius: 4px; }}
QMenu {{ background: {LIGHT_BG}; color: {LIGHT_INK}; border: 1px solid {LIGHT_BORDER}; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QToolBar {{ background: {LIGHT_BG_SOFT}; spacing: 6px; border: none; }}
QDockWidget::title {{ background: {LIGHT_INK}; color: white; padding: 6px; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {LIGHT_BORDER}; background: {LIGHT_BG}; border-radius: 6px; }}
QTabBar::tab {{ padding: 7px 16px; background: transparent; color: {LIGHT_MUTED}; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {LIGHT_BG}; color: {LIGHT_INK}; font-weight: 600; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover:!selected {{ color: {LIGHT_INK}; background: {LIGHT_BORDER}; }}
QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QTableWidget, QComboBox, QSpinBox {{
    background: {LIGHT_BG}; color: {LIGHT_INK}; border: 1px solid {LIGHT_BORDER}; border-radius: 6px; padding: 6px;
    selection-background-color: {ACCENT}; selection-color: white;
}}
QLineEdit:focus, QPlainTextEdit:focus {{ border: 1px solid {ACCENT}; }}
QLineEdit#searchEdit {{ padding: 8px 10px; font-size: 13px; }}
QPushButton {{ padding: 6px 14px; border: 1px solid {LIGHT_BORDER}; border-radius: 6px; background: {LIGHT_BG}; color: {LIGHT_INK}; }}
QPushButton:hover {{ background: {LIGHT_BG_SOFT}; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {LIGHT_MUTED}; background: {LIGHT_BG_SOFT}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: white; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_HOVER}; }}
QPushButton[primary="true"]:disabled {{ background: {LIGHT_BORDER}; border-color: {LIGHT_BORDER}; color: white; }}
QLabel#muted, QLabel[muted="true"] {{ color: {LIGHT_MUTED}; }}
QLabel#pageTitle {{ font-weight: 700; font-size: 13px; color: {LIGHT_INK}; }}
QLabel#probeBox {{ color: {LIGHT_INK}; background: #f1f5f9; padding: 6px; border-radius: 6px; }}
QLineEdit[modelState="ok"] {{ color: #15803d; }}
QLineEdit[modelState="missing"] {{ color: #dc2626; }}
QTableWidget {{ gridline-color: {LIGHT_BORDER}; }}
QHeaderView::section {{ background: {LIGHT_INK}; color: white; padding: 6px; border: none; font-weight: 600; }}
QStatusBar {{ background: {LIGHT_BG_SOFT}; color: {LIGHT_INK}; }}
QProgressBar {{ border: 1px solid {LIGHT_BORDER}; border-radius: 4px; background: {LIGHT_BG}; text-align: center; color: {LIGHT_INK}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QSplitter::handle {{ background: {LIGHT_BORDER}; }}
QGraphicsView {{ border: 1px solid {LIGHT_BORDER}; border-radius: 6px; }}
"""

DARK_QSS = f"""
QWidget {{ font-size: 13px; }}
QMainWindow, QDialog {{ background: {DARK_BG}; }}
QMenuBar {{ background: {DARK_BG}; color: {DARK_INK}; spacing: 4px; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {DARK_BG_SOFT}; border-radius: 4px; }}
QMenu {{ background: {DARK_BG_SOFT}; color: {DARK_INK}; border: 1px solid {DARK_BORDER}; }}
QMenu::item:selected {{ background: {ACCENT_DARK}; color: white; }}
QToolBar {{ background: {DARK_BG}; spacing: 6px; border: none; }}
QDockWidget::title {{ background: #020617; color: {DARK_INK}; padding: 6px; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {DARK_BORDER}; background: {DARK_BG_SOFT}; border-radius: 6px; }}
QTabBar::tab {{ padding: 7px 16px; background: transparent; color: {DARK_MUTED}; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {DARK_BG_SOFT}; color: {DARK_INK}; font-weight: 600; border-bottom: 2px solid {ACCENT_DARK}; }}
QTabBar::tab:hover:!selected {{ color: {DARK_INK}; background: {DARK_BORDER}; }}
QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QTableWidget, QComboBox, QSpinBox {{
    background: {DARK_BG_SOFT}; color: {DARK_INK}; border: 1px solid {DARK_BORDER}; border-radius: 6px; padding: 6px;
    selection-background-color: {ACCENT_DARK}; selection-color: white;
}}
QLineEdit:focus, QPlainTextEdit:focus {{ border: 1px solid {ACCENT_DARK}; }}
QLineEdit#searchEdit {{ padding: 8px 10px; font-size: 13px; }}
QPushButton {{ padding: 6px 14px; border: 1px solid {DARK_BORDER}; border-radius: 6px; background: {DARK_BG_SOFT}; color: {DARK_INK}; }}
QPushButton:hover {{ border-color: {ACCENT_DARK}; background: {DARK_BORDER}; }}
QPushButton:disabled {{ color: {DARK_MUTED}; background: {DARK_BG}; }}
QPushButton[primary="true"] {{ background: {ACCENT_DARK}; border: 1px solid {ACCENT_DARK}; color: white; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_DARK_HOVER}; }}
QPushButton[primary="true"]:disabled {{ background: {DARK_BORDER}; border-color: {DARK_BORDER}; color: white; }}
QLabel#muted, QLabel[muted="true"] {{ color: {DARK_MUTED}; }}
QLabel#pageTitle {{ font-weight: 700; font-size: 13px; color: {DARK_INK}; }}
QLabel#probeBox {{ color: {DARK_INK}; background: #020617; padding: 6px; border-radius: 6px; }}
QLineEdit[modelState="ok"] {{ color: #4ade80; }}
QLineEdit[modelState="missing"] {{ color: #f87171; }}
QTableWidget {{ gridline-color: {DARK_BORDER}; }}
QHeaderView::section {{ background: #020617; color: {DARK_INK}; padding: 6px; border: none; font-weight: 600; }}
QStatusBar {{ background: {DARK_BG}; color: {DARK_INK}; }}
QProgressBar {{ border: 1px solid {DARK_BORDER}; border-radius: 4px; background: {DARK_BG_SOFT}; text-align: center; color: {DARK_INK}; }}
QProgressBar::chunk {{ background: {ACCENT_DARK}; border-radius: 3px; }}
QSplitter::handle {{ background: {DARK_BORDER}; }}
QGraphicsView {{ border: 1px solid {DARK_BORDER}; border-radius: 6px; }}
"""


def is_dark(app: QApplication | None = None) -> bool:
    return current_theme_name(app) == "dark"


def accent(app: QApplication | None = None) -> str:
    return ACCENT_DARK if is_dark(app) else ACCENT


def ink(app: QApplication | None = None) -> str:
    return DARK_INK if is_dark(app) else LIGHT_INK


def muted(app: QApplication | None = None) -> str:
    return DARK_MUTED if is_dark(app) else LIGHT_MUTED


def apply_theme(app: QApplication, mode: str) -> str:
    """Apply *mode* (light/dark). Returns the effective mode."""
    mode = (mode or "light").lower()
    if mode not in ("light", "dark"):
        mode = "light"
    app.setStyle("Fusion")
    app.setPalette(_base_palette(light=(mode == "light")))
    app.setStyleSheet(DARK_QSS if mode == "dark" else LIGHT_QSS)
    app.setProperty("pagexTheme", mode)
    try:
        from PyQt6.QtCore import QSettings
        QSettings("PageX", "PageX").setValue("theme", mode)
    except Exception:
        pass
    return mode


def saved_theme(fallback: str = "light") -> str:
    try:
        from PyQt6.QtCore import QSettings
        v = str(QSettings("PageX", "PageX").value("theme", fallback) or fallback)
        return v if v in ("light", "dark") else fallback
    except Exception:
        return fallback


def toggle_theme(app: QApplication | None = None) -> str:
    app = app or QApplication.instance()
    assert app is not None
    return apply_theme(app, "dark" if current_theme_name(app) != "dark" else "light")
