"""Left navigation rail for the Antigravity-style PageX shell.

The sidebar is deliberately quiet: brand, one primary import action, two
destinations (Library / Ask), recent documents, and a small status area.
Everything else belongs in the main canvas.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QButtonGroup, QFrame,
                             QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPushButton, QVBoxLayout,
                             QWidget)


class Sidebar(QWidget):
    importRequested = pyqtSignal()
    libraryRequested = pyqtSignal()
    askRequested = pyqtSignal()
    settingsRequested = pyqtSignal()
    themeRequested = pyqtSignal()
    docActivated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(248)
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 16, 14, 12)
        root.setSpacing(10)

        brand = QHBoxLayout()
        brand.setSpacing(9)
        mark = QLabel("P")
        mark.setObjectName("brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(30, 30)
        name = QLabel("PageX")
        name.setObjectName("brand")
        brand.addWidget(mark)
        brand.addWidget(name)
        brand.addStretch(1)
        root.addLayout(brand)

        self.import_btn = QPushButton("+  Import documents")
        self.import_btn.setProperty("primary", True)
        self.import_btn.setToolTip("Import images, PDFs or a folder")
        self.import_btn.clicked.connect(self.importRequested.emit)
        root.addWidget(self.import_btn)

        root.addSpacing(6)
        section = QLabel("WORKSPACE")
        section.setObjectName("navSection")
        root.addWidget(section)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.library_btn = self._nav_button("Library")
        self.ask_btn = self._nav_button("Ask PageX")
        self.library_btn.clicked.connect(self.libraryRequested.emit)
        self.ask_btn.clicked.connect(self.askRequested.emit)
        self.nav_group.addButton(self.library_btn)
        self.nav_group.addButton(self.ask_btn)
        root.addWidget(self.library_btn)
        root.addWidget(self.ask_btn)

        root.addSpacing(8)
        recent_head = QHBoxLayout()
        recent_label = QLabel("RECENT")
        recent_label.setObjectName("navSection")
        self.count_label = QLabel("")
        self.count_label.setObjectName("muted")
        recent_head.addWidget(recent_label)
        recent_head.addStretch(1)
        recent_head.addWidget(self.count_label)
        root.addLayout(recent_head)

        self.recent = QListWidget()
        self.recent.setObjectName("recentList")
        self.recent.setFrameShape(QFrame.Shape.NoFrame)
        self.recent.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.recent.itemClicked.connect(self._on_recent)
        root.addWidget(self.recent, 1)

        line = QFrame()
        line.setObjectName("separator")
        line.setFixedHeight(1)
        root.addWidget(line)

        self.status = QLabel("Ready")
        self.status.setObjectName("statusChip")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        bottom = QHBoxLayout()
        bottom.setSpacing(4)
        self.theme_btn = QPushButton("Theme")
        self.theme_btn.setObjectName("ghostButton")
        self.theme_btn.setToolTip("Toggle light / dark theme")
        self.theme_btn.clicked.connect(self.themeRequested.emit)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setObjectName("ghostButton")
        self.settings_btn.clicked.connect(self.settingsRequested.emit)
        bottom.addWidget(self.theme_btn)
        bottom.addWidget(self.settings_btn)
        bottom.addStretch(1)
        root.addLayout(bottom)

    def _nav_button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("navButton")
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    # ------------------------------------------------------------------
    def set_documents(self, docs: list[dict]) -> None:
        self.recent.clear()
        if not docs:
            empty = QListWidgetItem("No documents yet")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(empty)
            self.count_label.setText("")
            return
        for doc in docs[:10]:
            title = str(doc.get("title") or "Untitled")
            pages = int(doc.get("actual_pages") or doc.get("n_pages") or 0)
            status = str(doc.get("status") or "")
            suffix = {"indexing": " · indexing", "enriching": " · enriching",
                      "paused": " · paused", "error": " · error"}.get(status, "")
            item = QListWidgetItem(f"{title}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, doc.get("doc_id"))
            item.setToolTip(f"{title} · {pages} page(s) · {status}")
            self.recent.addItem(item)
        self.count_label.setText(str(len(docs)) if docs else "")

    def set_active(self, name: str | None) -> None:
        self.library_btn.setChecked(name == "library")
        self.ask_btn.setChecked(name == "ask")

    def set_status(self, text: str, *, error: bool = False) -> None:
        self.status.setText(text)
        state = "error" if error else (
            "ok" if text.strip().lower().startswith("ready") else "")
        self.status.setProperty("state", state)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def set_busy(self, busy: bool) -> None:
        self.import_btn.setEnabled(not busy)

    def _on_recent(self, item: QListWidgetItem) -> None:
        doc_id = item.data(Qt.ItemDataRole.UserRole)
        if doc_id:
            self.docActivated.emit(str(doc_id))
