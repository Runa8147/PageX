"""First-run model wizard: download / verify / install an offline pack.

Thin Qt shell over :mod:`pagex.core.model_manifest` - literally the same
functions ``pagex models download|verify|install-pack`` call, so the GUI and
the CLI can never disagree about what is installed.

Threading follows the ADR-001 contract: the dialog never downloads on the UI
thread.  Work runs on ``QThreadPool`` via :func:`gui.workers.run_in_background`
and reports back through Qt signals; ``model_manifest`` streams SHA-256
progress through its ``progress(key, done, total)`` callback.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QHeaderView,
                             QLabel, QMessageBox, QProgressBar, QPushButton,
                             QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..core import model_manifest as mm
from ..core.config import CFG
from .workers import run_in_background

log = logging.getLogger(__name__)

_MB = 1_000_000


class ModelsWizard(QDialog):
    """Table of every weight + Download / Verify / Install-from-pack buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PageX - models")
        self.resize(720, 460)
        self.entries = mm.load_manifest()
        self._active: list[mm.ModelEntry] = []
        self._done: dict[str, int] = {}
        self._busy = False

        root = QVBoxLayout(self)
        self.header = QLabel(f"Models directory: {CFG.models_dir}")
        self.header.setWordWrap(True)
        self.header.setStyleSheet("color:#6b7280;")
        root.addWidget(self.header)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["State", "Key", "Size", "Detail"])
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        root.addWidget(self.table, 1)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        root.addWidget(self.bar)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#374151;")
        root.addWidget(self.status)

        buttons = QHBoxLayout()
        self.btn_download = QPushButton("Download missing models…")
        self.btn_download.clicked.connect(self.start_download)
        self.btn_verify = QPushButton("Verify")
        self.btn_verify.clicked.connect(self.start_verify)
        self.btn_pack = QPushButton("Install from pack…")
        self.btn_pack.clicked.connect(self.start_install_pack)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        for b in (self.btn_download, self.btn_verify, self.btn_pack):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_close)
        root.addLayout(buttons)

        self._states = {e.key: self._state_of(e) for e in self.entries}
        self._refresh_table()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _state_of(self, entry: mm.ModelEntry) -> tuple[str, str]:
        """(state, detail) from a *size-only* check - instant, no hashing."""
        p = mm.local_path(entry)
        if not p.exists():
            return ("missing", "not downloaded yet")
        if entry.size_bytes and p.stat().st_size != entry.size_bytes:
            return ("bad size", f"{p.stat().st_size} != {entry.size_bytes} bytes")
        return ("present", p.name)

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.entries))
        for i, e in enumerate(self.entries):
            state, detail = self._states.get(e.key, ("?", ""))
            mark = {"present": "✓", "missing": "·",
                    "bad size": "!"}.get(state, "")
            self.table.setItem(i, 0, QTableWidgetItem(f"{mark} {state}"))
            self.table.setItem(i, 1, QTableWidgetItem(e.key))
            self.table.setItem(i, 2, QTableWidgetItem(f"{e.size_bytes / _MB:.0f} MB"))
            self.table.setItem(i, 3, QTableWidgetItem(detail))
        missing = [e.key for e in self.entries if self._states[e.key][0] != "present"]
        total = sum(e.size_bytes for e in self.entries
                    if self._states[e.key][0] != "present")
        self.header.setText(
            f"Models directory: {CFG.models_dir}   ·   "
            f"{len(self.entries) - len(missing)}/{len(self.entries)} present"
            + (f"   ·   {total / 1e9:.1f} GB to download" if missing else ""))
        self.btn_download.setEnabled(not self._busy)
        self.btn_download.setText(
            "Download missing models…" if missing else "All models present")

    def _set_busy(self, busy: bool, label: str = "") -> None:
        self._busy = busy
        for b in (self.btn_download, self.btn_verify, self.btn_pack):
            b.setEnabled(not busy)
        if label:
            self.status.setText(label)
        if busy:
            self.bar.setValue(0)

    def _keys_in_table(self) -> dict[str, int]:
        return {e.key: i for i, e in enumerate(self.entries)}

    def _apply(self, results: list[dict], what: str) -> None:
        rows = self._keys_in_table()
        for r in results:
            key = r.get("key")
            if key is None or key not in rows:
                continue
            status = r.get("status", "?")
            pretty = {"present": "present", "downloaded": "present",
                      "installed": "present", "ok": "present",
                      "size-ok": "present", "missing": "missing",
                      "sha-mismatch": "corrupt", "size-mismatch": "corrupt",
                      "error": "error"}.get(status, status)
            self._states[key] = (pretty, r.get("detail", ""))
            self.table.setItem(rows[key], 0, QTableWidgetItem(pretty))
            self.table.setItem(rows[key], 3,
                               QTableWidgetItem(r.get("detail", "")))
        counts = mm.status_summary(results)
        self.status.setText(
            f"{what}: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
        # re-check the entries this run did not report on (size only: instant);
        # reported rows keep the state the operation actually observed.
        seen = {r.get("key") for r in results if r.get("key")}
        for e in self.entries:
            if e.key not in seen:
                self._states[e.key] = self._state_of(e)
        self._refresh_table()

    # ------------------------------------------------------------------
    # actions (all of them run off the UI thread)
    # ------------------------------------------------------------------
    def start_download(self) -> None:
        if self._busy:
            return
        entries = [e for e in self.entries
                   if self._states[e.key][0] != "present"] or \
            mm.select(self.entries, tier=2)
        self._active = entries
        self._done = {}
        self._set_busy(True, f"Downloading {len(entries)} model(s) - "
                             "resumable, SHA-256 verified…")
        run_in_background(mm.download, entries,
                          on_finished=self._on_finished,
                          on_error=self._on_error,
                          on_progress=self._on_progress)

    def start_verify(self) -> None:
        if self._busy:
            return
        self._active = list(self.entries)
        self._done = {}
        self._set_busy(True, "Verifying SHA-256 of every file "
                             "(a few minutes for 7 GB)…")
        run_in_background(mm.verify, self.entries, hash_files=True,
                          on_finished=self._on_finished,
                          on_error=self._on_error,
                          on_progress=self._on_progress)

    def start_install_pack(self) -> None:
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Install a PageX model pack", str(Path.home()),
            "Model packs (*.zip);;All files (*)")
        if not path:
            return
        self._active = [e for e in self.entries]
        self._set_busy(True, f"Installing {path} …")
        run_in_background(mm.install_pack, path,
                          on_finished=self._on_finished,
                          on_error=self._on_error)

    # -- signal handlers ------------------------------------------------
    def _on_progress(self, key: str, done: int, total: int) -> None:
        """``progress(key, done_bytes, total_bytes)`` from model_manifest."""
        if not self._active:
            return
        self._done[key] = int(done)
        total_bytes = sum(max(e.size_bytes, 1) for e in self._active)
        got = sum(min(self._done.get(e.key, 0), max(e.size_bytes, 1))
                  for e in self._active)
        self.bar.setValue(min(100, int(100 * got / total_bytes)))
        self.status.setText(f"{key}: {done / _MB:.0f} / {total / _MB:.0f} MB")

    def _on_finished(self, results) -> None:
        self._set_busy(False, "done")
        try:
            self._apply(list(results or []), "finished")
        except Exception:                          # pragma: no cover
            log.exception("could not refresh the model table")

    def _on_error(self, message: str) -> None:
        self._set_busy(False, f"failed: {message}")
        QMessageBox.warning(self, "PageX - models", str(message))
