"""Ask PageX - one surface for grounded answers and passage search.

The old GUI had a permanent search bar *and* a separate chat tab with its own
input.  This pane merges both: a mode switch selects **Answer** (RAG) or
**Find passages** (hybrid search), while the composer and citation panel stay
in one place.
"""
from __future__ import annotations

import html
import logging
import re

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QFrame, QHBoxLayout,
                             QLabel, QListWidget, QListWidgetItem,
                             QPlainTextEdit, QPushButton, QSplitter,
                             QStackedWidget, QTextBrowser, QVBoxLayout,
                             QWidget)

from ..core.config import CFG
from . import theme

log = logging.getLogger(__name__)

MODE_ANSWER = "answer"
MODE_FIND = "find"


class PromptEdit(QPlainTextEdit):
    """Composer with Enter-to-send and QLineEdit-compatible helpers."""

    returnPressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("composerEdit")
        self.setPlaceholderText("Ask anything across your notes…")
        self.setFixedHeight(44)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textChanged.connect(self._fit_height)

    def keyPressEvent(self, event):                 # noqa: N802 - Qt API
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.returnPressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def text(self) -> str:                          # QLineEdit compatibility
        return self.toPlainText().strip()

    def setText(self, text: str) -> None:           # noqa: N802 - Qt API
        self.setPlainText(text or "")

    def _fit_height(self) -> None:
        lines = max(1, min(5, self.document().blockCount()))
        self.setFixedHeight(44 + (lines - 1) * 18)


class ResultCard(QFrame):
    """One passage hit: thumbnail, title/page, snippet and ranker badges."""

    activated = pyqtSignal(dict)

    def __init__(self, citation: dict, thumb: QPixmap | None = None,
                 parent=None):
        super().__init__(parent)
        self.citation = dict(citation)
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 9, 12, 9)
        lay.setSpacing(11)

        image = QLabel()
        image.setFixedSize(52, 68)
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if thumb is not None:
            image.setPixmap(thumb.scaled(52, 68,
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))
        else:
            image.setText("▤")
            image.setObjectName("muted")
        lay.addWidget(image)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel(f'{citation.get("title", "Untitled")} · '
                       f'p.{citation.get("page_no", "?")}')
        title.setObjectName("pageTitle")
        snippet = QLabel(_highlight_snippet(
            str(citation.get("snippet", "")), citation.get("query", "")))
        snippet.setWordWrap(True)
        snippet.setObjectName("muted")
        meta = QLabel(
            f'{float(citation.get("score", 0) or 0):.3f} · '
            f'{citation.get("source", "hybrid")}')
        meta.setObjectName("statusChip")
        text.addWidget(title)
        text.addWidget(snippet, 1)
        text.addWidget(meta)
        lay.addLayout(text, 1)

    def mouseReleaseEvent(self, event):              # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(dict(self.citation))
        super().mouseReleaseEvent(event)


class ChatPane(QWidget):
    """Grounded answer + passage search in one place."""

    citationActivated = pyqtSignal(dict)           # Citation.to_dict()
    askRequested = pyqtSignal(str, object, object)  # query, session_id, doc_ids
    searchRequested = pyqtSignal(str)              # query

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.session_id: str | None = None
        self._busy = False
        self._mode = MODE_ANSWER
        self._last_citations: list[dict] = []
        self._messages: list[tuple[str, str, bool]] = []
        self._build()
        self._show_empty_transcript()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(10)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Ask PageX")
        title.setObjectName("heroTitle")
        self.session_label = QLabel("New conversation · answers grounded in your library")
        self.session_label.setObjectName("muted")
        titles.addWidget(title)
        titles.addWidget(self.session_label)
        head.addLayout(titles)
        head.addStretch(1)
        self.scope_combo = QComboBox()
        self.scope_combo.setMinimumWidth(190)
        self.scope_combo.setToolTip("Limit retrieval to one document")
        self.scope_combo.currentIndexChanged.connect(self._on_scope)
        self.new_btn = QPushButton("New chat")
        self.new_btn.setObjectName("ghostButton")
        self.new_btn.clicked.connect(self.new_session)
        head.addWidget(self.scope_combo)
        head.addWidget(self.new_btn)
        root.addLayout(head)

        modes = QFrame()
        modes.setObjectName("card")
        ml = QHBoxLayout(modes)
        ml.setContentsMargins(5, 4, 5, 4)
        ml.setSpacing(2)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.answer_btn = self._mode_button("Answer", MODE_ANSWER, True)
        self.find_btn = self._mode_button("Find passages", MODE_FIND, False)
        ml.addWidget(self.answer_btn)
        ml.addWidget(self.find_btn)
        ml.addStretch(1)
        self.mode_hint = QLabel("")
        self.mode_hint.setObjectName("muted")
        ml.addWidget(self.mode_hint)
        root.addWidget(modes)

        self.content_stack = QStackedWidget()
        root.addWidget(self.content_stack, 1)

        # -- answer mode ------------------------------------------------
        answer_page = QWidget()
        al = QVBoxLayout(answer_page)
        al.setContentsMargins(0, 0, 0, 0)
        self.answer_split = QSplitter(Qt.Orientation.Horizontal)
        self.answer_split.setChildrenCollapsible(False)
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        self.transcript.document().setDefaultStyleSheet(
            "p{margin:6px 0;line-height:1.5}"
            "code,pre{font-family:monospace}")
        self.answer_split.addWidget(self.transcript)

        sources_card = QFrame()
        sources_card.setObjectName("card")
        sources_card.setMinimumWidth(250)
        sources_card.setMaximumWidth(360)
        sl = QVBoxLayout(sources_card)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(6)
        src_head = QLabel("Sources")
        src_head.setObjectName("pageTitle")
        src_hint = QLabel("Click a source to open the exact page.")
        src_hint.setObjectName("muted")
        src_hint.setWordWrap(True)
        self.sources = QListWidget()
        self.sources.setIconSize(QSize(34, 44))
        self.sources.setWordWrap(True)
        self.sources.itemActivated.connect(self._on_source)
        self.sources.itemClicked.connect(self._on_source)
        sl.addWidget(src_head)
        sl.addWidget(src_hint)
        sl.addWidget(self.sources, 1)
        self.answer_split.addWidget(sources_card)
        self.answer_split.setStretchFactor(0, 1)
        self.answer_split.setStretchFactor(1, 0)
        self.answer_split.setSizes([760, 300])
        al.addWidget(self.answer_split)
        self.content_stack.addWidget(answer_page)

        # -- find mode --------------------------------------------------
        find_page = QWidget()
        fl = QVBoxLayout(find_page)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        self.find_hint = QLabel("Search every indexed page with vector + BM25 + graph.")
        self.find_hint.setObjectName("muted")
        self.results_list = QListWidget()
        self.results_list.setObjectName("resultList")
        self.results_list.setSpacing(6)
        self.results_list.itemClicked.connect(self._on_result)
        self.results_list.itemActivated.connect(self._on_result)
        fl.addWidget(self.find_hint)
        fl.addWidget(self.results_list, 1)
        self.content_stack.addWidget(find_page)

        # -- composer ---------------------------------------------------
        composer = QFrame()
        composer.setObjectName("composer")
        cl = QVBoxLayout(composer)
        cl.setContentsMargins(12, 9, 10, 8)
        cl.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.input = PromptEdit()
        self.input.returnPressed.connect(self._ask)
        self.send_btn = QPushButton("Ask")
        self.send_btn.setProperty("primary", True)
        self.send_btn.setMinimumHeight(38)
        self.send_btn.clicked.connect(self._ask)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        cl.addLayout(row)
        self.hint = QLabel("Enter to send · Shift+Enter for a new line · "
                           "PageX refuses to answer without sources.")
        self.hint.setObjectName("muted")
        cl.addWidget(self.hint)
        root.addWidget(composer)
        self._set_mode(MODE_ANSWER)

    def _mode_button(self, text: str, mode: str, checked: bool) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("modeButton")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.clicked.connect(lambda: self._set_mode(mode))
        self.mode_group.addButton(btn)
        return btn

    # ------------------------------------------------------------------
    # documents / scope
    # ------------------------------------------------------------------
    def set_documents(self, docs: list[dict]) -> None:
        current = self.scope_combo.currentData()
        self.scope_combo.blockSignals(True)
        self.scope_combo.clear()
        self.scope_combo.addItem("All documents", None)
        for doc in docs:
            self.scope_combo.addItem(str(doc.get("title") or "Untitled"),
                                     doc.get("doc_id"))
        if current:
            idx = self.scope_combo.findData(current)
            if idx >= 0:
                self.scope_combo.setCurrentIndex(idx)
        self.scope_combo.blockSignals(False)

    def set_scope(self, doc_id: str | None) -> None:
        idx = self.scope_combo.findData(doc_id)
        if idx >= 0:
            self.scope_combo.setCurrentIndex(idx)

    def scope_doc_ids(self) -> list[str]:
        doc_id = self.scope_combo.currentData()
        return [str(doc_id)] if doc_id else []

    def _on_scope(self, _index: int) -> None:
        doc_id = self.scope_combo.currentData()
        if doc_id:
            self.mode_hint.setText("Scoped to one document")
        else:
            self.mode_hint.setText("")

    # ------------------------------------------------------------------
    # modes / session
    # ------------------------------------------------------------------
    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        find = mode == MODE_FIND
        self.answer_btn.setChecked(not find)
        self.find_btn.setChecked(find)
        self.content_stack.setCurrentIndex(1 if find else 0)
        self.send_btn.setText("Find" if find else "Ask")
        self.input.setPlaceholderText(
            "Search passages, equations, diagrams…" if find
            else "Ask anything across your notes…")
        self.hint.setText(
            "Enter to search · results open the exact page."
            if find else
            "Enter to send · Shift+Enter for a new line · "
            "PageX refuses to answer without sources.")
        self.mode_hint.setText("Ranked passages" if find else "")
        if find:
            self.session_label.setText(
                "Hybrid search · vector + BM25 + graph")

    def set_find_mode(self, query: str | None = None) -> None:
        self._set_mode(MODE_FIND)
        if query is not None:
            self.input.setText(query)

    def new_session(self) -> None:
        self.session_id = None
        self.session_label.setText(
            "New conversation · answers grounded in your library")
        self.transcript.clear()
        self.sources.clear()
        self._last_citations = []
        self._messages = []
        self._show_empty_transcript()
        self.set_busy(False)

    def _ensure_session(self) -> str:
        if not self.session_id:
            self.session_id = self.store.new_session()
            self.session_label.setText(
                f"Conversation {self.session_id[-6:]} · grounded in your library")
        return self.session_id

    # ------------------------------------------------------------------
    # ask / search
    # ------------------------------------------------------------------
    def _ask(self) -> None:
        query = self.input.text().strip()
        if not query or self._busy:
            return
        self.input.clear()
        if self._mode == MODE_FIND:
            self.find_hint.setText(f'Searching for "{query}"…')
            self.searchRequested.emit(query)
            return
        self._append("you", query)
        self.askRequested.emit(query, self._ensure_session(),
                               self.scope_doc_ids())

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_btn.setEnabled(not busy)
        self.input.setEnabled(not busy)
        if busy:
            self.send_btn.setText("Thinking…" if self._mode == MODE_ANSWER
                                  else "Searching…")
        else:
            self.send_btn.setText("Ask" if self._mode == MODE_ANSWER else "Find")

    def show_answer(self, answer: str, citations: list[dict],
                    elapsed_s: float | None = None) -> None:
        """Called by the window when the RAG worker finishes."""
        self.set_busy(False)
        self._append("pagex", answer or "(no answer)")
        self._last_citations = list(citations or [])
        self.sources.clear()
        for c in self._last_citations:
            label = (f'{c.get("title", "?")} · p{c.get("page_no", 0)}   '
                     f'{c.get("source", "")}')
            item = QListWidgetItem(label)
            thumb = self._citation_thumb(c)
            if thumb is not None:
                item.setIcon(QIcon(thumb))
            item.setData(Qt.ItemDataRole.UserRole, c)
            item.setToolTip(str(c.get("snippet", ""))[:400])
            self.sources.addItem(item)
        if self._last_citations:
            self.session_label.setText(
                f"{len(self._last_citations)} source(s) · "
                f"{elapsed_s:.1f}s" if elapsed_s else
                f"{len(self._last_citations)} source(s) · grounded")
        else:
            self.session_label.setText("No sources · answer withheld")

    def show_results(self, result) -> None:
        """Populate Find passages from a ``SearchResult``."""
        self.set_busy(False)
        self._set_mode(MODE_FIND)
        self.results_list.clear()
        citations = list(getattr(result, "citations", []) or [])
        query = str(getattr(result, "query", "") or "")
        if not citations:
            self.find_hint.setText(f'No passages found for "{query}".')
            return
        self.find_hint.setText(
            f'{len(citations)} passage(s) for "{query}" · click to open')
        self.session_label.setText(
            f"{len(citations)} passage(s) · vector + BM25 + graph")
        for citation in citations:
            data = citation.to_dict() if hasattr(citation, "to_dict") else dict(citation)
            data.setdefault("query", query)
            snippet = str(data.get("snippet", ""))
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, data)
            item.setToolTip(snippet)
            item.setText(f'{data.get("title", "?")} · p.{data.get("page_no", "?")}')
            item.setSizeHint(QSize(0, 92))
            self.results_list.addItem(item)
            card = ResultCard(data, thumb=self._citation_thumb(data))
            card.activated.connect(self.citationActivated.emit)
            self.results_list.setItemWidget(item, card)

    def show_error(self, message: str) -> None:
        self.set_busy(False)
        self._append("pagex", f'<span style="color:{theme.danger()}">'
                              f'error: {html.escape(message)}</span>', raw=True)

    # ------------------------------------------------------------------
    # transcript helpers
    # ------------------------------------------------------------------
    def _show_empty_transcript(self) -> None:
        color = theme.accent()
        self.transcript.setHtml(
            f'<div style="margin-top:28px;text-align:center;color:{theme.muted()}">'
            f'<p style="font-size:20px;color:{theme.ink()};margin-bottom:4px">'
            f'Ask your notes anything.</p>'
            f'<p>Answers are grounded in what you imported, with page citations.</p>'
            f'<p style="margin-top:18px;color:{color}">'
            f'Try: “derive the moment of inertia” · “what is the chain rule?”</p>'
            f'</div>')

    def _append(self, who: str, text: str, *, raw: bool = False) -> None:
        first = not self._messages
        self._messages.append((who, text, raw))
        if first:
            self.transcript.clear()      # remove the centred empty-state hero
        self._render_message(who, text, raw)
        self.transcript.verticalScrollBar().setValue(
            self.transcript.verticalScrollBar().maximum())

    def _render_message(self, who: str, text: str, raw: bool = False) -> None:
        if raw:
            body = text
        elif who == "pagex":
            body = _markdown(text)
        else:
            body = html.escape(text).replace("\n", "<br>")
        color = theme.accent() if who == "you" else theme.success()
        label = "you&gt;" if who == "you" else "pagex&gt;"
        self.transcript.append(
            f'<div style="margin-top:14px;margin-bottom:2px">'
            f'<span style="color:{color};font-weight:700">{label}</span></div>'
            f'<div style="color:{theme.ink()};margin:2px 0 12px 0">'
            f'{body}</div>')

    def refresh_theme(self) -> None:
        self.transcript.clear()
        if not self._messages:
            self._show_empty_transcript()
            return
        for who, text, raw in self._messages:
            self._render_message(who, text, raw)

    # ------------------------------------------------------------------
    # citations
    # ------------------------------------------------------------------
    def _on_source(self, item: QListWidgetItem) -> None:
        citation = item.data(Qt.ItemDataRole.UserRole)
        if citation:
            self.citationActivated.emit(dict(citation))

    def _on_result(self, item: QListWidgetItem) -> None:
        citation = item.data(Qt.ItemDataRole.UserRole)
        if citation:
            self.citationActivated.emit(dict(citation))

    def _citation_thumb(self, citation: dict) -> QPixmap | None:
        from pathlib import Path
        page = None
        page_id = citation.get("page_id")
        try:
            if page_id:
                page = self.store.get_page(str(page_id))
            elif citation.get("doc_id") and citation.get("page_no"):
                for candidate_page in self.store.get_pages(
                        str(citation["doc_id"])):
                    if int(candidate_page.get("page_no") or 0) == \
                            int(citation["page_no"]):
                        page = candidate_page
                        break
        except Exception:
            page = None
        meta = (page or {}).get("meta") or {}
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta or "{}")
            except Exception:
                meta = {}
        candidates = [meta.get("thumb"), citation.get("image_path"),
                      (page or {}).get("image_path")]
        for image in candidates:
            if not image:
                continue
            path = Path(str(image))
            if not path.is_absolute():
                path = CFG.artifacts_dir / path
            if path.exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    return pix.scaled(34, 44,
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        return None


# ---------------------------------------------------------------------------
# tiny markdown -> HTML (no dependency; safe because input is escaped first)
# ---------------------------------------------------------------------------
def _markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\$\$(.+?)\$\$", r"<code>\1</code>", escaped,
                     flags=re.DOTALL)
    escaped = re.sub(r"\$(.+?)\$", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"(?m)^#{1,4}\s*(.+)$", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?m)^[-*]\s+(.+)$", r"• \1", escaped)
    escaped = re.sub(r"\[doc:([^\]]+)\]",
                     rf'<span style="color:{theme.accent()}">[doc:\1]</span>',
                     escaped)
    return escaped.replace("\n", "<br>")


def _highlight_snippet(snippet: str, query: str) -> str:
    """Escape a snippet, then highlight whole query words without breaking HTML."""
    text = re.sub(r"\s+", " ", snippet).strip()
    if len(text) > 240:
        text = text[:237].rstrip() + "…"
    terms = list(dict.fromkeys(
        t for t in re.findall(r"[A-Za-z0-9_]{3,}", query or "")))[:6]
    if not terms:
        return html.escape(text)
    pattern = re.compile("(" + "|".join(re.escape(t) for t in terms) + ")",
                         re.IGNORECASE)
    chunks: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        chunks.append(html.escape(text[cursor:match.start()]))
        chunks.append(
            f'<span style="background:{theme.accent()};color:#ffffff">'
            f'{html.escape(match.group(0))}</span>')
        cursor = match.end()
    chunks.append(html.escape(text[cursor:]))
    return "".join(chunks)
