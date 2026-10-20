"""WP9 - grounded chat pane (retrieval + local LLM, with citations).

Every answer arrives through a background worker; the pane only formats.
Clicking a citation emits ``citationActivated`` so the viewer can jump to
the exact page/blocks the answer was grounded on.
"""
from __future__ import annotations

import html
import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QPushButton, QSplitter,
                             QTextBrowser, QVBoxLayout, QWidget)

log = logging.getLogger(__name__)


class ChatPane(QWidget):
    """Transcript + input + sources list."""

    citationActivated = pyqtSignal(dict)           # Citation.to_dict()
    askRequested = pyqtSignal(str, object)         # query, session_id (worker hook)

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.session_id: str | None = None
        self._busy = False
        self._last_citations: list[dict] = []
        self._build()

    # -- construction ----------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        head = QHBoxLayout()
        self.session_label = QLabel("session: new")
        self.session_label.setObjectName("muted")
        self.new_btn = QPushButton("New session")
        self.new_btn.clicked.connect(self.new_session)
        head.addWidget(self.session_label, 1)
        head.addWidget(self.new_btn)
        layout.addLayout(head)

        split = QSplitter(Qt.Orientation.Vertical)
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        split.addWidget(self.transcript)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        src_label = QLabel("Sources (click to open in viewer)")
        src_label.setObjectName("muted")
        rl.addWidget(src_label)
        self.sources = QListWidget()
        self.sources.itemActivated.connect(self._on_source)
        self.sources.itemClicked.connect(self._on_source)
        rl.addWidget(self.sources, 1)
        split.addWidget(right)
        split.setSizes([520, 160])
        layout.addWidget(split, 1)

        ask = QHBoxLayout()
        ask.setSpacing(8)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask about your notes… e.g. derive the moment of inertia")
        self.input.returnPressed.connect(self._ask)
        self.send_btn = QPushButton("Ask")
        self.send_btn.setProperty("primary", True)
        self.send_btn.clicked.connect(self._ask)
        ask.addWidget(self.input, 1)
        ask.addWidget(self.send_btn)
        layout.addLayout(ask)

        self.hint = QLabel("Answers are grounded: no retrieved context → the model "
                           "says it does not know.")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

    # -- session ---------------------------------------------------------
    def new_session(self) -> None:
        self.session_id = None
        self.session_label.setText("session: new")
        self.transcript.clear()
        self.sources.clear()
        self._last_citations = []

    def _ensure_session(self) -> str:
        if not self.session_id:
            self.session_id = self.store.new_session()
            self.session_label.setText(f"session: {self.session_id}")
        return self.session_id

    # -- ask -------------------------------------------------------------
    def _ask(self) -> None:
        query = self.input.text().strip()
        if not query or self._busy:
            return
        self.input.clear()
        self._append("you", query)
        self.askRequested.emit(query, self._ensure_session())

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_btn.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.send_btn.setText("thinking…" if busy else "Ask")

    def show_answer(self, answer: str, citations: list[dict],
                    elapsed_s: float | None = None) -> None:
        """Called by the window when the RAG worker finishes."""
        from .theme import accent, is_dark, muted
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        dark = is_dark(app)
        self.set_busy(False)
        self._append("pagex", answer or "(no answer)")
        self._last_citations = list(citations or [])
        self.sources.clear()
        for c in self._last_citations:
            label = (f'{c.get("title", "?")} · p{c.get("page_no", 0)} · '
                     f'score {c.get("score", 0):.3f} · {c.get("source", "")}')
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, c)
            item.setToolTip(str(c.get("snippet", ""))[:400])
            self.sources.addItem(item)
        dim = muted(app)
        tail = f'<span style="color:{dim}"> · {elapsed_s:.1f}s</span>' \
            if elapsed_s else ""
        if self._last_citations:
            srcs = ", ".join(sorted({c.get("source", "?") for c in self._last_citations}))
            self.transcript.append(
                f'<span style="color:{dim};font-size:11px">'
                f'{len(self._last_citations)} source(s) [{srcs}]{tail}</span>')

    def show_error(self, message: str) -> None:
        self.set_busy(False)
        self._append("pagex", f'<span style="color:#dc2626">error: '
                              f'{html.escape(message)}</span>', raw=True)

    # -- helpers ---------------------------------------------------------
    def _append(self, who: str, text: str, *, raw: bool = False) -> None:
        from .theme import accent, ink
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        body = text if raw else html.escape(text).replace("\n", "<br>")
        you_color = accent(app)
        bot_color = "#16a34a" if not (
            app and str(app.property("pagexTheme") or "") == "dark") else "#4ade80"
        fg = ink(app)
        color = you_color if who == "you" else bot_color
        self.transcript.append(
            f'<p style="margin:6px 0"><b style="color:{color}">{who}&gt;</b> '
            f'<span style="color:{fg}">{body}</span></p>')
        self.transcript.verticalScrollBar().setValue(
            self.transcript.verticalScrollBar().maximum())

    def _on_source(self, item: QListWidgetItem) -> None:
        citation = item.data(Qt.ItemDataRole.UserRole)
        if citation:
            self.citationActivated.emit(dict(citation))
