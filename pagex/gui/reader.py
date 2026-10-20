"""Reader canvas: page rail + document canvas + block inspector.

The old app split this across a Viewer tab, an Editor tab and a LaTeX tab.
Here the page, its blocks and the derived LaTeX live in one workspace.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPlainTextEdit,
                             QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..core.blocks import blocks_to_annotated_tex, document_from_store
from ..core.config import CFG
from .editor import BlockEditor
from .viewer import PageViewer

log = logging.getLogger(__name__)


class ReaderView(QWidget):
    backRequested = pyqtSignal()
    askRequested = pyqtSignal(str)                 # doc_id
    exportRequested = pyqtSignal()

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.doc_id: str | None = None
        self._pages: list[dict] = []
        self._blocks: list[dict] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.back_btn = QPushButton("←  Library")
        self.back_btn.setObjectName("ghostButton")
        self.back_btn.clicked.connect(self.backRequested.emit)
        self.title_label = QLabel("No document open")
        self.title_label.setObjectName("pageTitle")
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("muted")
        self.ask_btn = QPushButton("Ask this document")
        self.ask_btn.setProperty("primary", True)
        self.ask_btn.clicked.connect(
            lambda: self.doc_id and self.askRequested.emit(self.doc_id))
        self.source_btn = QPushButton("LaTeX")
        self.source_btn.setObjectName("ghostButton")
        self.source_btn.setToolTip("View the annotated LaTeX source")
        self.source_btn.clicked.connect(self.open_source)
        self.export_btn = QPushButton("Export")
        self.export_btn.setObjectName("ghostButton")
        self.export_btn.clicked.connect(self.exportRequested.emit)
        self.inspector_btn = QPushButton("Inspector")
        self.inspector_btn.setObjectName("ghostButton")
        self.inspector_btn.setCheckable(True)
        self.inspector_btn.setChecked(True)
        self.inspector_btn.toggled.connect(self._toggle_inspector)
        bar.addWidget(self.back_btn)
        bar.addWidget(self.title_label)
        bar.addWidget(self.meta_label)
        bar.addStretch(1)
        bar.addWidget(self.inspector_btn)
        bar.addWidget(self.source_btn)
        bar.addWidget(self.export_btn)
        bar.addWidget(self.ask_btn)
        root.addLayout(bar)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setChildrenCollapsible(False)
        self.pages_list = QListWidget()
        self.pages_list.setObjectName("pageRail")
        self.pages_list.setIconSize(QSize(62, 84))
        self.pages_list.setFixedWidth(104)
        self.pages_list.currentRowChanged.connect(self._on_page_row)
        body.addWidget(self.pages_list)

        self.viewer = PageViewer()
        self.viewer.pageChanged.connect(self._on_viewer_page)
        body.addWidget(self.viewer)

        self.inspector = QFrame()
        self.inspector.setObjectName("card")
        self.inspector.setMinimumWidth(360)
        self.inspector.setMaximumWidth(470)
        il = QVBoxLayout(self.inspector)
        il.setContentsMargins(8, 8, 8, 8)
        self.editor = BlockEditor(self.store, compact=True)
        il.addWidget(self.editor)
        body.addWidget(self.inspector)

        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setStretchFactor(2, 0)
        body.setSizes([104, 700, 400])
        root.addWidget(body, 1)

        self.source_edit = QPlainTextEdit()
        self.source_edit.setReadOnly(True)
        self.source_edit.setPlaceholderText(
            "Annotated LaTeX appears here after a document is opened.")

        self.viewer.blockSelected.connect(self.editor.select_block)
        self.editor.blockSelected.connect(
            lambda bid: self.viewer.highlight_block(bid, select=True))

    # ------------------------------------------------------------------
    def set_document(self, doc_id: str) -> None:
        doc_row = self.store.get_document(doc_id)
        if not doc_row:
            return
        self.doc_id = doc_id
        self.title_label.setText(str(doc_row.get("title") or doc_id))
        self._pages = self.store.get_pages(doc_id)
        self._blocks = self.store.get_blocks(doc_id=doc_id)
        self._fill_page_rail()
        self.viewer.set_document(self._pages, self._blocks)
        page = self.viewer.current_page()
        self.editor.set_page(page.get("page_id") if page else None)
        self._reload_source()
        self._update_meta()
        self.inspector.setVisible(self.inspector_btn.isChecked())

    def clear(self) -> None:
        self.doc_id = None
        self._pages = []
        self._blocks = []
        self.title_label.setText("No document open")
        self.meta_label.setText("")
        self.pages_list.clear()
        self.viewer.set_document([], [])
        self.editor.set_page(None)
        self.source_edit.clear()
        self.export_btn.setEnabled(False)

    def refresh(self) -> None:
        """Reload pages/blocks while indexing, preserving the current page."""
        if not self.doc_id:
            return
        page = self.viewer.current_page()
        current_no = int(page["page_no"]) if page else 1
        self._pages = self.store.get_pages(self.doc_id)
        self._blocks = self.store.get_blocks(doc_id=self.doc_id)
        self._fill_page_rail()
        self.viewer.set_document(self._pages, self._blocks)
        self.viewer.goto(max(0, min(current_no - 1, len(self._pages) - 1)))
        page = self.viewer.current_page()
        self.editor.set_page(page.get("page_id") if page else None)
        self._reload_source()
        self._update_meta()

    def show_citation(self, citation: dict) -> None:
        if not self.doc_id:
            return
        self.viewer.show_citation(citation)
        self._sync_rail()

    def open_source(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Annotated LaTeX · {self.title_label.text()}")
        dlg.resize(900, 700)
        lay = QVBoxLayout(dlg)
        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(self.source_edit.toPlainText())
        edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(edit)
        dlg.exec()

    # ------------------------------------------------------------------
    def _fill_page_rail(self) -> None:
        self.pages_list.blockSignals(True)
        self.pages_list.clear()
        for page in self._pages:
            item = QListWidgetItem(str(page.get("page_no")))
            item.setData(Qt.ItemDataRole.UserRole, int(page.get("page_no") or 0))
            thumb = self._thumb(page)
            if thumb is not None:
                item.setIcon(QIcon(thumb))
            item.setToolTip(f"Page {page.get('page_no')}")
            self.pages_list.addItem(item)
        self.pages_list.blockSignals(False)
        self._sync_rail()

    def _thumb(self, page: dict) -> QPixmap | None:
        meta = page.get("meta") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta or "{}")
            except Exception:
                meta = {}
        candidates = [meta.get("thumb"), page.get("image_path")]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if not path.is_absolute():
                path = CFG.artifacts_dir / path
            if path.exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    return pix.scaled(62, 84, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        return None

    def _sync_rail(self) -> None:
        self.pages_list.blockSignals(True)
        page = self.viewer.current_page()
        if page:
            for row in range(self.pages_list.count()):
                if self.pages_list.item(row).data(Qt.ItemDataRole.UserRole) == \
                        int(page.get("page_no") or 0):
                    self.pages_list.setCurrentRow(row)
                    break
        self.pages_list.blockSignals(False)

    def _on_page_row(self, row: int) -> None:
        if 0 <= row < len(self._pages):
            self.viewer.goto(row)

    def _on_viewer_page(self, page_no: int) -> None:
        self._sync_rail()
        page = self.viewer.current_page()
        self.editor.set_page(page.get("page_id") if page else None)

    def _toggle_inspector(self, on: bool) -> None:
        self.inspector.setVisible(bool(on))

    def _reload_source(self) -> None:
        if not self.doc_id:
            self.source_edit.clear()
            return
        doc = document_from_store(self.store, self.doc_id)
        if doc:
            self.source_edit.setPlainText(blocks_to_annotated_tex(doc))
            self.export_btn.setEnabled(True)
        else:
            self.source_edit.clear()
            self.export_btn.setEnabled(False)

    def _update_meta(self) -> None:
        row = self.store.get_document(self.doc_id) if self.doc_id else None
        status = str((row or {}).get("status") or "")
        n_pages = len(self._pages)
        n_blocks = len(self._blocks)
        bits = [f"{n_pages} page(s)", f"{n_blocks} block(s)"]
        if status and status != "ready":
            bits.append(status)
        self.meta_label.setText(" · ".join(bits))
