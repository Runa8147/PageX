"""WP9 - library pane: document table, ingest button, watch-folder toggle."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout,
                             QHeaderView, QLabel, QMessageBox, QPushButton,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ..core.watch import scan

log = logging.getLogger(__name__)

COLUMNS = ("Title", "Pages", "Status", "Created", "ID")
STATUS_COLORS = {
    "ready": "#16a34a",
    "indexing": "#d97706",
    "pending": "#6b7280",
    "error": "#dc2626",
}


class LibraryView(QWidget):
    """Left dock: what is indexed, and the controls that add to it."""

    ingestRequested = pyqtSignal(list)            # [Path, ...]
    docSelected = pyqtSignal(str)                 # doc_id
    docDeleted = pyqtSignal(str)                  # doc_id
    watchToggled = pyqtSignal(bool)               # on/off
    refreshRequested = pyqtSignal()               # re-read documents table

    def __init__(self, parent=None):
        super().__init__(parent)
        self._watch_on = False
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Library")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.table, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.ingest_btn = QPushButton("Ingest…")
        self.ingest_btn.setProperty("primary", True)
        self.ingest_btn.setToolTip("Index an image, a PDF or a whole folder")
        self.ingest_btn.clicked.connect(self._pick_files)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refreshRequested.emit)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_selected)
        row.addWidget(self.ingest_btn, 2)
        row.addWidget(self.refresh_btn, 1)
        row.addWidget(self.delete_btn, 1)
        layout.addLayout(row)

        self.watch_btn = QPushButton("Watch folder: off")
        self.watch_btn.setCheckable(True)
        self.watch_btn.setToolTip(
            "Auto-ingest new images/PDFs dropped into the watch folder")
        self.watch_btn.toggled.connect(self._toggle_watch)
        layout.addWidget(self.watch_btn)

        self.hint = QLabel("Drop a photo of your notes here via Ingest…")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

    # ------------------------------------------------------------------
    def refresh_table(self, docs: list[dict]) -> None:
        self.table.setRowCount(len(docs))
        for row, doc in enumerate(docs):
            values = (str(doc.get("title", "")), str(doc.get("n_pages", "")),
                      str(doc.get("status", "")), str(doc.get("created_at", "")),
                      str(doc.get("doc_id", "")))
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 2:
                    color = STATUS_COLORS.get(value, "#111827")
                    item.setForeground(Qt.GlobalColor.transparent)
                    from PyQt6.QtGui import QBrush, QColor
                    item.setForeground(QBrush(QColor(color)))
                if col == 4:
                    item.setForeground(QBrush(QColor("#9ca3af")))
                self.table.setItem(row, col, item)
        self.hint.setText(
            f"{len(docs)} document(s) indexed · "
            f"select one to view, edit or chat about it")

    def selected_doc_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount():
            return None
        item = self.table.item(row, 4)
        return item.text() if item else None

    # ------------------------------------------------------------------
    def _on_select(self) -> None:
        doc_id = self.selected_doc_id()
        if doc_id:
            self.docSelected.emit(doc_id)

    def _pick_files(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Ingest documents", "",
            "Images and PDFs (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf);;"
            "All files (*)")
        if path:
            p = Path(path)
            files = scan(p) if p.is_dir() else [p]
            if files:
                self.ingestRequested.emit(files)

    def ingest_folder(self, folder: str | Path) -> None:
        files = scan(folder)
        if not files:
            QMessageBox.information(self, "PageX", f"No ingestible files in {folder}")
            return
        self.ingestRequested.emit(files)

    def _delete_selected(self) -> None:
        doc_id = self.selected_doc_id()
        if not doc_id:
            return
        answer = QMessageBox.question(
            self, "Delete document",
            f"Remove {doc_id} and all of its chunks, blocks and graph edges?")
        if answer == QMessageBox.StandardButton.Yes:
            self.docDeleted.emit(doc_id)

    def _toggle_watch(self, on: bool) -> None:
        self._watch_on = on
        self.watch_btn.setText("Watch folder: on" if on else "Watch folder: off")
        self.watchToggled.emit(on)

    @property
    def watch_enabled(self) -> bool:
        return self._watch_on

    def set_busy(self, busy: bool) -> None:
        self.ingest_btn.setEnabled(not busy)
        self.hint.setText("ingesting… (UI stays responsive; inference runs on a "
                          "worker thread)" if busy else self.hint.text())
