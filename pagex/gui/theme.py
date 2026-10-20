"""PageX visual system - a calm, Antigravity-inspired desktop shell.

The app is intentionally dark-first: a low-contrast canvas, a quiet sidebar,
one accent colour for intent, and no decorative chrome.  Light mode keeps the
same hierarchy with inverted neutrals.

Design tokens: 70% neutral surface / 20% ink / 10% accent.  Spacing follows an
8 px rhythm; radii are 8-12 px; borders are hairlines rather than shadows.
"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

# -- accent (the 10%) -------------------------------------------------------
ACCENT = "#5b4bdb"           # light-mode primary
ACCENT_HOVER = "#4c3fc4"
ACCENT_DARK = "#8b7cf8"      # dark-mode primary (brightened)
ACCENT_DARK_HOVER = "#a294ff"
ACCENT_SOFT = "#eef0ff"
ACCENT_SOFT_DARK = "#211f3b"

# -- light neutrals (the 70% + ink 20%) ------------------------------------
LIGHT_BG = "#ffffff"
LIGHT_BG_SOFT = "#f5f6f8"
LIGHT_SIDEBAR = "#eef0f3"
LIGHT_BORDER = "#e1e5ea"
LIGHT_BORDER_STRONG = "#cbd2db"
LIGHT_INK = "#171920"
LIGHT_MUTED = "#667085"
LIGHT_SURFACE = "#ffffff"
LIGHT_ELEVATED = "#f8f9fb"

# -- dark neutrals ----------------------------------------------------------
DARK_BG = "#0a0b0e"
DARK_BG_SOFT = "#0f1115"
DARK_SIDEBAR = "#0d0f13"
DARK_BORDER = "#232830"
DARK_BORDER_STRONG = "#313844"
DARK_INK = "#e8eaed"
DARK_MUTED = "#8b93a1"
DARK_SURFACE = "#12151a"
DARK_ELEVATED = "#171b22"

SUCCESS = "#16a34a"
SUCCESS_DARK = "#4ade80"
WARNING = "#d97706"
WARNING_DARK = "#fbbf24"
DANGER = "#dc2626"
DANGER_DARK = "#f87171"

THEMES = ("light", "dark", "system")


# ---------------------------------------------------------------------------
# token helpers used by widgets that build inline HTML
# ---------------------------------------------------------------------------
def current_theme_name(app: QApplication | None = None) -> str:
    app = app or QApplication.instance()
    if app is None:
        return "dark"
    return str(app.property("pagexTheme") or "dark")


def is_dark(app: QApplication | None = None) -> bool:
    return current_theme_name(app) == "dark"


def accent(app: QApplication | None = None) -> str:
    return ACCENT_DARK if is_dark(app) else ACCENT


def ink(app: QApplication | None = None) -> str:
    return DARK_INK if is_dark(app) else LIGHT_INK


def muted(app: QApplication | None = None) -> str:
    return DARK_MUTED if is_dark(app) else LIGHT_MUTED


def surface(app: QApplication | None = None) -> str:
    return DARK_SURFACE if is_dark(app) else LIGHT_SURFACE


def border(app: QApplication | None = None) -> str:
    return DARK_BORDER if is_dark(app) else LIGHT_BORDER


def success(app: QApplication | None = None) -> str:
    return SUCCESS_DARK if is_dark(app) else SUCCESS


def danger(app: QApplication | None = None) -> str:
    return DANGER_DARK if is_dark(app) else DANGER


def _base_palette(light: bool) -> QPalette:
    pal = QPalette()
    if light:
        pal.setColor(QPalette.ColorRole.Window, QColor(LIGHT_BG_SOFT))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(LIGHT_INK))
        pal.setColor(QPalette.ColorRole.Base, QColor(LIGHT_BG))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(LIGHT_ELEVATED))
        pal.setColor(QPalette.ColorRole.Text, QColor(LIGHT_INK))
        pal.setColor(QPalette.ColorRole.Button, QColor(LIGHT_BG))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(LIGHT_INK))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(LIGHT_INK))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(LIGHT_MUTED))
    else:
        pal.setColor(QPalette.ColorRole.Window, QColor(DARK_BG))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(DARK_INK))
        pal.setColor(QPalette.ColorRole.Base, QColor(DARK_SURFACE))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(DARK_ELEVATED))
        pal.setColor(QPalette.ColorRole.Text, QColor(DARK_INK))
        pal.setColor(QPalette.ColorRole.Button, QColor(DARK_SURFACE))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(DARK_INK))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_DARK))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#0a0b0e"))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(DARK_INK))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor(DARK_BG))
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(DARK_MUTED))
    return pal


def _qss(light: bool) -> str:
    bg = LIGHT_BG_SOFT if light else DARK_BG
    bg_soft = LIGHT_BG_SOFT if light else DARK_BG_SOFT
    sidebar = LIGHT_SIDEBAR if light else DARK_SIDEBAR
    surface_ = LIGHT_SURFACE if light else DARK_SURFACE
    elevated = LIGHT_ELEVATED if light else DARK_ELEVATED
    border_ = LIGHT_BORDER if light else DARK_BORDER
    border_strong = LIGHT_BORDER_STRONG if light else DARK_BORDER_STRONG
    ink_ = LIGHT_INK if light else DARK_INK
    muted_ = LIGHT_MUTED if light else DARK_MUTED
    accent_ = ACCENT if light else ACCENT_DARK
    accent_hover = ACCENT_HOVER if light else ACCENT_DARK_HOVER
    accent_soft = ACCENT_SOFT if light else ACCENT_SOFT_DARK
    success_ = SUCCESS if light else SUCCESS_DARK
    warning_ = WARNING if light else WARNING_DARK
    danger_ = DANGER if light else DANGER_DARK
    selection_text = "#ffffff" if light else "#0a0b0e"
    return f"""
* {{ outline: none; }}
QWidget {{ font-family: "Inter", "Segoe UI", "SF Pro Text", "DejaVu Sans", sans-serif;
           font-size: 13px; color: {ink_}; }}
QMainWindow, QDialog {{ background: {bg}; }}
QWidget#root {{ background: {bg}; }}

/* -- sidebar ----------------------------------------------------------- */
QFrame#sidebar {{ background: {sidebar}; border: none;
                  border-right: 1px solid {border_}; }}
QLabel#brand {{ font-size: 17px; font-weight: 750; color: {ink_};
               letter-spacing: -.2px; }}
QLabel#brandMark {{ background: {accent_}; color: {selection_text};
                   border-radius: 8px; font-weight: 800; font-size: 14px; }}
QLabel#navSection {{ color: {muted_}; font-size: 10px; font-weight: 700;
                    letter-spacing: 1.1px; padding: 4px 8px; }}
QLabel#muted, QLabel[muted="true"] {{ color: {muted_}; }}
QLabel#pageTitle {{ font-weight: 700; font-size: 15px; color: {ink_}; }}
QLabel#heroTitle {{ font-weight: 750; font-size: 26px; color: {ink_}; }}
QLabel#heroSub {{ color: {muted_}; font-size: 13px; }}
QLabel#statusChip {{ color: {muted_}; font-size: 11px; }}
QLabel#statusChip[state="error"] {{ color: {danger_}; }}
QLabel#statusChip[state="ok"] {{ color: {success_}; }}
QLabel#probeBox {{ color: {ink_}; background: {elevated}; padding: 8px;
                   border: 1px solid {border_}; border-radius: 8px; }}

/* -- navigation / generic buttons ------------------------------------- */
QPushButton {{ padding: 7px 12px; border: 1px solid {border_}; border-radius: 8px;
              background: {surface_}; color: {ink_}; }}
QPushButton:hover {{ background: {elevated}; border-color: {border_strong}; }}
QPushButton:pressed {{ background: {border_}; }}
QPushButton:disabled {{ color: {muted_}; background: {bg_soft};
                       border-color: {border_}; }}
QPushButton[primary="true"] {{ background: {accent_}; border: 1px solid {accent_};
                              color: {selection_text}; font-weight: 650; }}
QPushButton[primary="true"]:hover {{ background: {accent_hover};
                                    border-color: {accent_hover}; }}
QPushButton[primary="true"]:disabled {{ background: {border_};
                                       border-color: {border_};
                                       color: {muted_}; }}
QPushButton#navButton {{ text-align: left; padding: 8px 10px; border: none;
                        background: transparent; color: {muted_};
                        border-radius: 8px; font-weight: 550; }}
QPushButton#navButton:hover {{ background: {elevated}; color: {ink_}; }}
QPushButton#navButton:checked {{ background: {accent_soft}; color: {accent_};
                                font-weight: 650; }}
QPushButton#iconButton {{ padding: 5px 8px; border-radius: 7px;
                         background: transparent; border: none; color: {muted_}; }}
QPushButton#iconButton:hover {{ background: {elevated}; color: {ink_}; }}
QPushButton#ghostButton {{ background: transparent; border: none; color: {muted_};
                          padding: 5px 8px; }}
QPushButton#ghostButton:hover {{ color: {ink_}; background: {elevated}; }}
QPushButton#modeButton {{ border: none; background: transparent; color: {muted_};
                         padding: 6px 12px; border-radius: 7px; font-weight: 550; }}
QPushButton#modeButton:hover {{ color: {ink_}; background: {elevated}; }}
QPushButton#modeButton:checked {{ background: {accent_soft}; color: {accent_};
                                 font-weight: 650; }}
QPushButton#dangerButton {{ color: {danger_}; }}
QPushButton#dangerButton:hover {{ background: {danger_}; color: white;
                                 border-color: {danger_}; }}

/* -- surfaces / cards -------------------------------------------------- */
QFrame#card {{ background: {surface_}; border: 1px solid {border_};
              border-radius: 12px; }}
QFrame#card:hover {{ border-color: {border_strong}; }}
QFrame#activity {{ background: {surface_}; border: 1px solid {border_};
                  border-radius: 10px; }}
QFrame#composer {{ background: {surface_}; border: 1px solid {border_strong};
                  border-radius: 14px; }}
QFrame#separator {{ background: {border_}; max-height: 1px; }}

/* -- text inputs ------------------------------------------------------- */
QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QTableWidget,
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {surface_}; color: {ink_}; border: 1px solid {border_};
    border-radius: 8px; padding: 6px;
    selection-background-color: {accent_}; selection-color: {selection_text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {accent_};
}}
QLineEdit#searchEdit {{ padding: 9px 12px; font-size: 13px; }}
QPlainTextEdit#composerEdit {{ border: none; background: transparent;
                              padding: 6px 2px; font-size: 13px; }}
QPlainTextEdit#composerEdit:focus {{ border: none; }}
QLineEdit[modelState="ok"] {{ color: {success_}; }}
QLineEdit[modelState="missing"] {{ color: {danger_}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {surface_}; border: 1px solid {border_};
                              selection-background-color: {accent_}; }}

/* -- lists / tables ---------------------------------------------------- */
QListWidget {{ background: transparent; border: none; padding: 2px; }}
QListWidget::item {{ padding: 5px 7px; border-radius: 7px; color: {ink_}; }}
QListWidget::item:hover {{ background: {elevated}; }}
QListWidget::item:selected {{ background: {accent_soft}; color: {accent_}; }}
QListWidget#recentList::item {{ padding: 6px 7px; }}
QListWidget#pageRail {{ background: {bg_soft}; border: 1px solid {border_};
                       border-radius: 10px; padding: 6px; }}
QListWidget#pageRail::item {{ padding: 4px; margin: 2px; border-radius: 6px; }}
QListWidget#pageRail::item:selected {{ background: {accent_soft};
                                      border: 1px solid {accent_}; }}
QTableWidget {{ background: {surface_}; border: 1px solid {border_};
               border-radius: 10px; gridline-color: transparent;
               alternate-background-color: {elevated}; }}
QTableWidget::item {{ padding: 6px; border-bottom: 1px solid {border_}; }}
QTableWidget::item:selected {{ background: {accent_soft}; color: {ink_}; }}
QHeaderView::section {{ background: transparent; color: {muted_};
                       padding: 7px 8px; border: none;
                       border-bottom: 1px solid {border_}; font-weight: 650;
                       font-size: 11px; }}

/* -- tabs (kept for dialogs/tools) ------------------------------------- */
QTabWidget::pane {{ border: 1px solid {border_}; background: {surface_};
                   border-radius: 10px; }}
QTabBar::tab {{ padding: 8px 14px; background: transparent; color: {muted_};
               margin-right: 2px; border-top-left-radius: 8px;
               border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ background: {surface_}; color: {ink_};
                        font-weight: 650; border-bottom: 2px solid {accent_}; }}
QTabBar::tab:hover:!selected {{ color: {ink_}; background: {elevated}; }}

/* -- scrollbars -------------------------------------------------------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {border_strong}; min-height: 24px;
                              border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: {muted_}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {border_strong}; min-width: 24px;
                                border-radius: 4px; }}

/* -- menus / status / misc -------------------------------------------- */
QMenuBar {{ background: {bg}; color: {muted_}; spacing: 2px; border: none; }}
QMenuBar::item {{ padding: 4px 9px; background: transparent; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {elevated}; color: {ink_}; }}
QMenu {{ background: {surface_}; color: {ink_}; border: 1px solid {border_};
        border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {accent_}; color: {selection_text}; }}
QStatusBar {{ background: {bg}; color: {muted_}; border-top: 1px solid {border_}; }}
QStatusBar::item {{ border: none; }}
QProgressBar {{ border: none; border-radius: 4px; background: {border_};
               text-align: center; color: {ink_}; max-height: 7px; }}
QProgressBar::chunk {{ background: {accent_}; border-radius: 4px; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {accent_}; }}
QGraphicsView {{ border: 1px solid {border_}; border-radius: 10px;
                background: {bg_soft}; }}
QDockWidget::title {{ background: {sidebar}; color: {muted_}; padding: 6px;
                     border-bottom: 1px solid {border_}; }}
QToolTip {{ background: {ink_}; color: {bg}; border: none; padding: 5px 8px;
           border-radius: 6px; }}
"""
LIGHT_QSS = _qss(True)
DARK_QSS = _qss(False)


def apply_theme(app: QApplication, mode: str) -> str:
    """Apply *mode* (light/dark). Returns the effective mode."""
    mode = (mode or "dark").lower()
    if mode not in ("light", "dark"):
        mode = "dark"
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


def saved_theme(fallback: str = "dark") -> str:
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
