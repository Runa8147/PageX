"""WP9 - block editor: correct the VLM's transcription, keep the SSOT true.

Editing a block writes the ``blocks`` row, refreshes the affected chunks
(re-embed included) and lets the window regenerate the annotated LaTeX -
the same derivation order as the ingest pipeline.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPlainTextEdit,
                             QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..core.blocks import BLOCK_KINDS
from ..core.pipeline import save_block_edit

log = logging.getLogger(__name__)


class BlockEditor(QWidget):
    """Right-hand editor: pick a block, fix its text/latex, save."""

    saved = pyqtSignal(dict)                       # {"block_id":..., ...}
    blockSelected = pyqtSignal(str)                # block_id -> viewer highlight
    saveRequested = pyqtSignal(dict)               # route through a worker

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self._blocks: list[dict] = []
        self._current: dict | None = None
        self._build()

    # -- construction ----------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Blocks on this page"))
        self.listw = QListWidget()
        self.listw.currentRowChanged.connect(self._on_row)
        ll.addWidget(self.listw, 1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.id_label = QLabel("no block selected")
        self.id_label.setObjectName("muted")
        self.kind_box = QComboBox()
        self.kind_box.addItems(list(BLOCK_KINDS))
        self.bbox_label = QLabel("-")
        self.bbox_label.setObjectName("muted")
        form.addRow("block", self.id_label)
        form.addRow("kind", self.kind_box)
        form.addRow("bbox", self.bbox_label)
        rl.addLayout(form)

        rl.addWidget(QLabel("Text (plain / transcription)"))
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("faithful transcription of the block")
        rl.addWidget(self.text_edit, 3)

        rl.addWidget(QLabel("LaTeX (math / table / diagram body)"))
        self.latex_edit = QPlainTextEdit()
        self.latex_edit.setPlaceholderText(r"e.g. \tau = I\alpha")
        rl.addWidget(self.latex_edit, 2)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.save_btn = QPushButton("Save block")
        self.save_btn.setProperty("primary", True)
        self.save_btn.setToolTip("Write to SQLite, re-embed affected chunks, "
                                 "regenerate annotated LaTeX")
        self.save_btn.clicked.connect(self._save)
        self.revert_btn = QPushButton("Revert")
        self.revert_btn.clicked.connect(self._load_current)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.revert_btn)
        buttons.addStretch(1)
        rl.addLayout(buttons)
        rl.addWidget(self.status)
        split.addWidget(right)
        split.setSizes([260, 520])
        layout.addWidget(split, 1)

    # -- data ------------------------------------------------------------
    def set_page(self, page_id: str | None) -> None:
        self._blocks = self.store.get_blocks(page_id=page_id) if page_id else []
        self.listw.clear()
        for b in self._blocks:
            head = str(b.get("text") or b.get("latex") or "").strip()
            item = QListWidgetItem(f'{b.get("kind", "?")} · {head[:70]}')
            item.setData(Qt.ItemDataRole.UserRole, b.get("block_id"))
            item.setToolTip(b.get("block_id", ""))
            self.listw.addItem(item)
        self._current = None
        self.status.setText(f"{len(self._blocks)} block(s)")
        if self._blocks:
            self.listw.setCurrentRow(0)

    def current_block_id(self) -> str | None:
        return self._current.get("block_id") if self._current else None

    def select_block(self, block_id: str) -> None:
        for row, b in enumerate(self._blocks):
            if b.get("block_id") == block_id:
                self.listw.setCurrentRow(row)
                return

    # -- slots -----------------------------------------------------------
    def _on_row(self, row: int) -> None:
        if 0 <= row < len(self._blocks):
            self._current = self._blocks[row]
            self._load_current()
            self.blockSelected.emit(str(self._current["block_id"]))

    def _load_current(self) -> None:
        b = self._current
        if not b:
            return
        self.id_label.setText(str(b.get("block_id", "")))
        idx = self.kind_box.findText(str(b.get("kind", "paragraph")))
        self.kind_box.setCurrentIndex(max(0, idx))
        self.bbox_label.setText(str(b.get("bbox")))
        self.text_edit.setPlainText(str(b.get("text") or ""))
        self.latex_edit.setPlainText(str(b.get("latex") or ""))
        self.status.setText("")

    def _save(self) -> None:
        """Runs on the GUI thread - save_block_edit is DB + embedder work.

        The window routes this through a worker; when used standalone (tests)
        it executes inline.  ``saveAsync`` is wired by MainWindow.
        """
        self.saveRequested.emit(self._collect())

    def _collect(self) -> dict:
        b = self._current or {}
        return {"block_id": b.get("block_id", ""),
                "text": self.text_edit.toPlainText(),
                "latex": self.latex_edit.toPlainText(),
                "kind": self.kind_box.currentText()}

    def apply_result(self, payload: dict) -> None:
        """Called by the window after the worker persists the edit."""
        bid = payload.get("block_id", "")
        self.status.setText(f"saved · {payload.get('chunks_refreshed', 0)} chunk(s) re-embedded")
        # refresh local copy from the store (source of truth)
        rows = self.store.blocks_by_ids([bid])
        if bid in rows:
            for i, b in enumerate(self._blocks):
                if b.get("block_id") == bid:
                    self._blocks[i] = rows[bid]
                    break
        self.saved.emit(payload)

    def do_save_sync(self) -> dict:
        """Blocking save (tests / CLI harness)."""
        payload = self._collect()
        result = save_block_edit(self.store, payload["block_id"],
                                 text=payload["text"], latex=payload["latex"],
                                 kind=payload["kind"])
        self.apply_result(result)
        return result
