"""Library canvas: a calm, scannable document list.

This replaces the old cramped dock.  A document is a row with a page
thumbnail, title, page count, status and date; technical IDs stay in the
database.  The empty state is the first-run onboarding surface.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QFrame,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QStackedWidget,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ..core.config import CFG
from ..core.watch import scan
from . import theme

log = logging.getLogger(__name__)

COLUMNS = ("Title", "Pages", "Status", "Updated")
STATUS_LABELS = {"ready": "Ready", "indexing": "Indexing",
                 "enriching": "Enriching", "paused": "Paused",
                 "error": "Error", "pending": "Queued", "queued": "Queued",
                 "importing": "Importing"}


class LibraryView(QWidget):
    """Library canvas with table + onboarding empty state."""

    ingestRequested = pyqtSignal(list)            # [Path, ...]
    docSelected = pyqtSignal(str)                 # doc_id
    docDeleted = pyqtSignal(str)                  # doc_id
    watchToggled = pyqtSignal(bool)               # on/off
    refreshRequested = pyqtSignal()               # re-read documents table

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self.store = store
        self._docs: list[dict] = []
        self._watch_on = False
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        head = QHBoxLayout()
        head.setSpacing(10)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Library")
        title.setObjectName("heroTitle")
        self.subtitle = QLabel("Your offline document workspace")
        self.subtitle.setObjectName("muted")
        titles.addWidget(title)
        titles.addWidget(self.subtitle)
        head.addLayout(titles)
        head.addStretch(1)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter documents…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setFixedWidth(230)
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("ghostButton")
        self.refresh_btn.clicked.connect(self.refreshRequested.emit)
        self.ingest_btn = QPushButton("+  Import")
        self.ingest_btn.setProperty("primary", True)
        self.ingest_btn.setToolTip("Import one or more images, PDFs or a folder")
        self.ingest_btn.clicked.connect(self._pick_files)
        head.addWidget(self.filter_edit)
        head.addWidget(self.refresh_btn)
        head.addWidget(self.ingest_btn)
        root.addLayout(head)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        # -- populated table -------------------------------------------
        table_wrap = QWidget()
        tl = QVBoxLayout(table_wrap)
        tl.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        title_header = self.table.horizontalHeaderItem(0)
        if title_header is not None:
            title_header.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(64)
        self.table.setIconSize(QSize(38, 50))
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        self.table.itemDoubleClicked.connect(
            lambda item: self._emit_row(item.row()))
        self.table.itemActivated.connect(
            lambda item: self._emit_row(item.row()))
        self.table.itemSelectionChanged.connect(self._sync_delete_btn)
        tl.addWidget(self.table, 1)
        self.stack.addWidget(table_wrap)

        # -- empty onboarding ------------------------------------------
        empty = QFrame()
        empty.setObjectName("card")
        el = QVBoxLayout(empty)
        el.setContentsMargins(40, 40, 40, 40)
        el.setSpacing(10)
        el.addStretch(1)
        hero = QLabel("Bring your notes into focus.")
        hero.setObjectName("heroTitle")
        hero.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Drop images or PDFs here, or import a folder.\n"
                     "PageX reads handwriting, equations, tables and diagrams "
                     "without sending anything to the cloud.")
        sub.setObjectName("heroSub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        self.empty_import = QPushButton("Import documents")
        self.empty_import.setProperty("primary", True)
        self.empty_import.setFixedWidth(190)
        self.empty_import.clicked.connect(self._pick_files)
        el.addWidget(hero)
        el.addWidget(sub)
        el.addSpacing(8)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.empty_import)
        row.addStretch(1)
        el.addLayout(row)
        self.empty_hint = QLabel("Tip: you can also drag files anywhere in the window.")
        self.empty_hint.setObjectName("muted")
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        el.addWidget(self.empty_hint)
        el.addStretch(1)
        self.stack.addWidget(empty)

        # -- footer ----------------------------------------------------
        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.watch_btn = QPushButton("Watch folder: off")
        self.watch_btn.setCheckable(True)
        self.watch_btn.setObjectName("ghostButton")
        self.watch_btn.setToolTip(
            "Auto-ingest new images/PDFs dropped into the watch folder")
        self.watch_btn.toggled.connect(self._toggle_watch)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setObjectName("ghostButton dangerButton")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.hint = QLabel("")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        footer.addWidget(self.watch_btn)
        footer.addWidget(self.hint, 1)
        footer.addWidget(self.delete_btn)
        root.addLayout(footer)

    # ------------------------------------------------------------------
    def refresh_table(self, docs: list[dict]) -> None:
        selected = self.selected_doc_id()
        self._docs = list(docs or [])
        self._apply_filter()
        if selected:
            self._select_doc(selected)
        self.stack.setCurrentIndex(0 if self._docs else 1)
        self._sync_delete_btn()
        self.subtitle.setText(
            f"{len(self._docs)} document(s) · local · private · searchable"
            if self._docs else "Your offline document workspace")

    def _apply_filter(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        docs = [d for d in self._docs
                if not needle or needle in str(d.get("title", "")).lower()]
        self.table.setRowCount(len(docs))
        for row, doc in enumerate(docs):
            self._fill_row(row, doc)
        self.hint.setText(
            f"{len(docs)} of {len(self._docs)} document(s)"
            if needle else f"{len(self._docs)} document(s) indexed")
        if not self._docs:
            self.stack.setCurrentIndex(1)

    def _fill_row(self, row: int, doc: dict) -> None:
        doc_id = str(doc.get("doc_id", ""))
        title = str(doc.get("title") or "Untitled")
        pages = int(doc.get("actual_pages") or doc.get("n_pages") or 0)
        status = str(doc.get("status") or "ready")
        created = str(doc.get("created_at") or "")[:10]

        title_item = QTableWidgetItem(title)
        title_item.setData(Qt.ItemDataRole.UserRole, doc_id)
        title_item.setToolTip(f"{title}\n{path_of(doc)}")
        thumb = self._thumbnail(doc_id)
        if thumb is not None:
            title_item.setIcon(QIcon(thumb))
        self.table.setItem(row, 0, title_item)
        self.table.setItem(row, 1, _center(f"{pages}"))
        status_item = QTableWidgetItem(STATUS_LABELS.get(status, status.title()))
        status_item.setForeground(QBrush(QColor(_status_color(status))))
        self.table.setItem(row, 2, status_item)
        self.table.setItem(row, 3, _muted(created))

    def _thumbnail(self, doc_id: str) -> QPixmap | None:
        if self.store is None:
            return None
        try:
            pages = self.store.get_pages(doc_id)
            if not pages:
                return None
            meta = pages[0].get("meta") or {}
            if isinstance(meta, str):
                import json
                meta = json.loads(meta or "{}")
            thumb = meta.get("thumb")
            if thumb:
                path = Path(thumb)
                if not path.is_absolute():
                    path = CFG.artifacts_dir / path
                if path.exists():
                    pix = QPixmap(str(path))
                    if not pix.isNull():
                        return pix.scaled(38, 50, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            image = pages[0].get("image_path")
            if image:
                path = Path(image)
                if not path.is_absolute():
                    path = CFG.artifacts_dir / path
                if path.exists():
                    pix = QPixmap(str(path))
                    if not pix.isNull():
                        return pix.scaled(38, 50, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
        except Exception:
            log.debug("thumbnail failed for %s", doc_id, exc_info=True)
        return None

    def update_doc_status(self, doc_id: str, status: str) -> None:
        """Update one status cell without rebuilding the table."""
        for doc in self._docs:
            if doc.get("doc_id") == doc_id:
                doc["status"] = status
                break
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == doc_id:
                status_item = self.table.item(row, 2)
                if status_item is not None:
                    status_item.setText(STATUS_LABELS.get(status, status.title()))
                    status_item.setForeground(
                        QBrush(QColor(_status_color(status))))
                break

    def selected_doc_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount():
            return None
        item = self.table.item(row, 0)
        return str(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _select_doc(self, doc_id: str) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == doc_id:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                return

    # ------------------------------------------------------------------
    def _sync_delete_btn(self) -> None:
        self.delete_btn.setEnabled(bool(self.selected_doc_id()))

    def _on_select(self) -> None:
        doc_id = self.selected_doc_id()
        if doc_id:
            self.docSelected.emit(doc_id)

    def _emit_row(self, row: int) -> None:
        item = self.table.item(row, 0)
        if item and item.data(Qt.ItemDataRole.UserRole):
            self.docSelected.emit(str(item.data(Qt.ItemDataRole.UserRole)))

    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import documents", "",
            "Images and PDFs (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf);;"
            "All files (*)")
        if paths:
            self.ingestRequested.emit([Path(p) for p in paths])

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
        doc = next((d for d in self._docs if d.get("doc_id") == doc_id), {})
        answer = QMessageBox.question(
            self, "Delete document",
            f'Delete "{doc.get("title", doc_id)}" and its index?\n'
            "This removes the pages, blocks, embeddings and chat sources.")
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
        self.empty_import.setEnabled(not busy)
        if busy:
            self.hint.setText("Importing… pages appear as they are read; "
                              "you can keep using the library.")


def _center(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


def _muted(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setForeground(QBrush(QColor(theme.muted())))
    return item


def _status_color(status: str) -> str:
    if status in ("ready",):
        return theme.success()
    if status in ("error",):
        return theme.danger()
    if status in ("indexing", "importing", "queued", "pending"):
        return theme.accent()
    if status in ("enriching", "paused"):
        return theme.muted()
    return theme.ink()


def path_of(doc: dict) -> str:
    return str(doc.get("source_path") or "")
