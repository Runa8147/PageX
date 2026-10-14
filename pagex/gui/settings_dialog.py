"""WP9 - settings dialog: hardware probe results + runtime knobs + models.

Saving rewrites ``pagex.toml`` through :meth:`Config.save`; the new values
take effect for everything that has not loaded a model yet (the ModelSlot
evicts on next acquire), and a status line says so.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSpinBox, QTabWidget,
                             QVBoxLayout, QWidget)

from ..core.config import BENCH_PATH, CFG

log = logging.getLogger(__name__)


def _bench() -> dict:
    try:
        return json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


class SettingsDialog(QDialog):
    """Tabs: Runtime (probe + knobs) and Models (GGUF paths)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PageX settings")
        self.resize(640, 480)
        root = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._runtime_tab(), "Runtime")
        tabs.addTab(self._models_tab(), "Models")
        root.addWidget(tabs, 1)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#6b7280;")
        root.addWidget(self.note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # -- tabs ------------------------------------------------------------
    def _runtime_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        b = _bench()
        llm = b.get("llm") or {}
        vlm = b.get("vlm") or {}
        embed = b.get("embed") or {}
        summary = (f'probe: {b.get("n_threads", "?")} thread(s), tier {b.get("tier", "?")}'
                   f' · LLM {llm.get("tok_per_s", "?")} tok/s ({llm.get("model", "?")})'
                   f' · VLM {vlm.get("seconds_per_page_est", "?")} s/page'
                   f' · embed {embed.get("texts_per_s", "?")} texts/s'
                   f' · peak RSS {b.get("rss_peak_gb", "?")} GB')
        probe = QLabel(summary)
        probe.setWordWrap(True)
        probe.setStyleSheet("color:#374151;background:#f3f4f6;padding:6px;")
        form.addRow("hardware", probe)

        self.threads_n = QSpinBox()
        self.threads_n.setRange(0, 32)
        self.threads_n.setValue(int(CFG.n_threads or 0))
        self.threads_n.setToolTip("0 = follow bench.json")
        form.addRow("n_threads (pin)", self.threads_n)

        self.tier = QComboBox()
        self.tier.addItems(["auto", "1", "2", "3"])
        current = CFG.tier or "auto"
        self.tier.setCurrentIndex(max(0, self.tier.findText(current)))
        form.addRow("tier", self.tier)

        self.vlm_side = QSpinBox()
        self.vlm_side.setRange(256, 2048)
        self.vlm_side.setSingleStep(64)
        self.vlm_side.setValue(int(CFG.vlm_max_side))
        self.vlm_side.setToolTip("long edge handed to the VLM "
                                 "(tier 2/3 forces 768)")
        form.addRow("vlm_max_side", self.vlm_side)

        self.vlm_tokens = QSpinBox()
        self.vlm_tokens.setRange(256, 8192)
        self.vlm_tokens.setSingleStep(256)
        self.vlm_tokens.setValue(int(CFG.vlm_max_tokens))
        form.addRow("vlm_max_tokens", self.vlm_tokens)

        self.llm_tokens = QSpinBox()
        self.llm_tokens.setRange(64, 4096)
        self.llm_tokens.setSingleStep(64)
        self.llm_tokens.setValue(int(CFG.llm_max_tokens))
        form.addRow("llm_max_tokens", self.llm_tokens)

        self.kv_q8 = QCheckBox("quantised KV cache q8_0 (halves KV RAM; "
                               "needs flash attention)")
        self.kv_q8.setChecked(bool(CFG.kv_q8))
        form.addRow("kv cache", self.kv_q8)

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 50)
        self.top_k.setValue(int(CFG.top_k))
        form.addRow("top_k", self.top_k)

        self.chunk_tokens = QSpinBox()
        self.chunk_tokens.setRange(50, 2000)
        self.chunk_tokens.setSingleStep(50)
        self.chunk_tokens.setValue(int(CFG.chunk_tokens))
        form.addRow("chunk_tokens", self.chunk_tokens)

        self.watch = QLineEdit(CFG.watch)
        browse = QPushButton("…")
        browse.clicked.connect(self._pick_watch)
        row = QHBoxLayout()
        row.addWidget(self.watch, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("watch folder", holder)
        return w

    def _models_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.model_edits: dict[str, QLineEdit] = {}
        labels = {
            "vlm_gguf": "VLM (page → JSON)",
            "vlm_mmproj": "VLM projector",
            "llm_gguf": "Chat / assembly LLM",
            "llm_fallback_gguf": "Chat LLM fallback (tier ladder)",
            "embed_gguf": "Embedder",
            "embed_alt_gguf": "Embedder fallback",
        }
        for key, label in labels.items():
            edit = QLineEdit(getattr(CFG, key))
            path = CFG.model_path(key)
            exists = path.exists()
            edit.setStyleSheet("color:%s;" % ("#16a34a" if exists else "#dc2626"))
            btn = QPushButton("…")
            btn.clicked.connect(lambda _=False, e=edit: self._pick_model(e))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(btn)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)
            self.model_edits[key] = edit
        hint = QLabel("Green = found on disk. Download missing models with "
                      "`python tools/download_models.py`.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;")
        form.addRow("", hint)
        return w

    # -- pickers ---------------------------------------------------------
    def _pick_watch(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Watch folder")
        if path:
            self.watch.setText(path)

    def _pick_model(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose GGUF", str(CFG.models_dir), "GGUF models (*.gguf)")
        if path:
            edit.setText(str(Path(path)))
            edit.setStyleSheet("color:#16a34a;")

    # -- persist ---------------------------------------------------------
    def _save(self) -> None:
        CFG.n_threads = int(self.threads_n.value())
        tier = self.tier.currentText()
        CFG.tier = "" if tier == "auto" else tier
        CFG.vlm_max_side = int(self.vlm_side.value())
        CFG.vlm_max_tokens = int(self.vlm_tokens.value())
        CFG.llm_max_tokens = int(self.llm_tokens.value())
        CFG.kv_q8 = bool(self.kv_q8.isChecked())
        CFG.top_k = int(self.top_k.value())
        CFG.chunk_tokens = int(self.chunk_tokens.value())
        CFG.watch = self.watch.text().strip()
        for key, edit in self.model_edits.items():
            setattr(CFG, key, edit.text().strip())
        try:
            saved = CFG.save()
            self.note.setText(
                f"wrote {saved} · applied to new sessions; already-loaded "
                "models are evicted on next use (ModelSlot).")
        except Exception as exc:
            self.note.setText(f"save failed: {exc}")
            log.exception("settings save failed")
            return
        self.accept()
