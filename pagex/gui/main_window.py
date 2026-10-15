"""WP9 - main window: docks, tabs, menus, and all worker wiring.

Threading contract (ADR-001/TC-GUI-02): the GUI thread never calls a model.
Ingest, search, RAG and export all run through ``gui.workers`` and report
back with Qt signals; SQLite connections are per-thread (Store contract).
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMainWindow,
                             QMessageBox, QProgressBar, QPushButton,
                             QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..core.config import CFG
from ..core.store import Store
from .chat_pane import ChatPane
from .editor import BlockEditor
from .library_view import LibraryView
from .settings_dialog import SettingsDialog
from .viewer import PageViewer
from .workers import run_in_background, wait_all

log = logging.getLogger(__name__)

EXPORT_FORMATS = (("Annotated LaTeX (SSOT)", "tex"),
                  ("Clean LaTeX", "clean-tex"),
                  ("PDF", "pdf"),
                  ("DOCX", "docx"),
                  ("HTML", "html"),
                  ("Markdown", "md"),
                  ("JSON", "json"))


class MainWindow(QMainWindow):
    """The whole desktop shell."""

    def __init__(self, store: Store | None = None):
        super().__init__()
        self.store = store or Store()
        self.setWindowTitle("PageX - visual document intelligence")
        self.resize(1360, 860)
        self._doc_id: str | None = None
        self._watch_state = None
        self._build_ui()
        self._wire()
        self.refresh_library()
        self._tick_status()
        QTimer.singleShot(0, self._hint_missing_models)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # -- left dock: library
        self.library = LibraryView()
        from PyQt6.QtWidgets import QDockWidget
        dock = QDockWidget("Library", self)
        dock.setObjectName("libraryDock")
        dock.setWidget(self.library)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # -- central: search bar + tabs
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            "hybrid search: vector + BM25 + graph  (Enter)")
        self.search_edit.returnPressed.connect(
            lambda: self.run_search(self.search_edit.text()))
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(
            lambda: self.run_search(self.search_edit.text()))
        bar.addWidget(self.search_edit, 1)
        bar.addWidget(self.search_btn)
        layout.addLayout(bar)

        self.tabs = QTabWidget()
        self.viewer = PageViewer()
        self.editor = BlockEditor(self.store)
        self.chat = ChatPane(self.store)
        from PyQt6.QtWidgets import QPlainTextEdit
        self.latex_edit = QPlainTextEdit()
        self.latex_edit.setReadOnly(True)
        self.latex_edit.setPlaceholderText(
            "annotated LaTeX (single source of truth) appears here")
        self.latex_edit.setStyleSheet("font-family:monospace;")
        self.search_list = QListWidget()

        self.tabs.addTab(self.viewer, "Viewer")
        self.tabs.addTab(self.editor, "Editor")
        self.tabs.addTab(self.search_list, "Search results")
        self.tabs.addTab(self.latex_edit, "LaTeX (SSOT)")
        self.tabs.addTab(self.chat, "Chat")
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # -- status bar
        self.model_label = QLabel("no model loaded")
        self.db_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.statusBar().addWidget(self.model_label, 1)
        self.statusBar().addPermanentWidget(self.db_label)
        self.statusBar().addPermanentWidget(self.progress)

        self._build_menu()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._tick_status)
        self._status_timer.start(1500)

        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(2000)
        self._watch_timer.timeout.connect(self._poll_watch)

    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        act_ingest = QAction("Ingest files…", self)
        act_ingest.setShortcut(QKeySequence.StandardKey.Open)
        act_ingest.triggered.connect(self._pick_ingest)
        act_folder = QAction("Ingest folder…", self)
        act_folder.triggered.connect(self._pick_folder)
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_ingest)
        m_file.addAction(act_folder)
        m_file.addSeparator()
        m_file.addAction(act_quit)

        m_export = self.menuBar().addMenu("&Export")
        self.export_menu = m_export
        self._export_actions = []
        for label, fmt in EXPORT_FORMATS:
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, f=fmt: self.export_document(f))
            m_export.addAction(act)
            self._export_actions.append(act)
        m_export.setEnabled(False)

        m_tools = self.menuBar().addMenu("&Tools")
        act_settings = QAction("Settings…", self)
        act_settings.triggered.connect(self.open_settings)
        act_bench = QAction("Run hardware probe…", self)
        act_bench.triggered.connect(self.run_probe)
        act_smoke = QAction("Smoke test…", self)
        act_smoke.triggered.connect(self.run_smoke)
        act_models = QAction("Download models…", self)
        act_models.triggered.connect(self.open_models_wizard)
        m_tools.addAction(act_settings)
        m_tools.addAction(act_bench)
        m_tools.addAction(act_smoke)
        m_tools.addAction(act_models)

        m_help = self.menuBar().addMenu("&Help")
        act_about = QAction("About PageX", self)
        act_about.triggered.connect(self._about)
        m_help.addAction(act_about)

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------
    def _wire(self) -> None:
        self.library.ingestRequested.connect(self.ingest_paths)
        self.library.docSelected.connect(self.open_document)
        self.library.docDeleted.connect(self.delete_document)
        self.library.refreshRequested.connect(self.refresh_library)
        self.library.watchToggled.connect(self.toggle_watch)

        self.viewer.blockSelected.connect(self.editor.select_block)
        self.editor.blockSelected.connect(
            lambda bid: self.viewer.highlight_block(bid, select=True))
        self.editor.saveRequested.connect(self._persist_block_edit)
        self.editor.saved.connect(lambda _payload: self._reload_latex())

        self.chat.askRequested.connect(self.ask)
        self.chat.citationActivated.connect(self._show_citation)
        self.search_list.itemActivated.connect(self._search_open)
        self.search_list.itemClicked.connect(self._search_open)

    # ------------------------------------------------------------------
    # library / documents
    # ------------------------------------------------------------------
    def _pick_ingest(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Ingest files", "",
            "Images and PDFs (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf);;"
            "All files (*)")
        if paths:
            self.ingest_paths([Path(p) for p in paths])

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Ingest folder")
        if folder:
            self.library.ingest_folder(folder)

    def refresh_library(self) -> None:
        docs = self.store.list_documents()
        self.library.refresh_table(docs)
        self.db_label.setText(
            f"db: {Path(CFG.db).name} · {self.store.count_chunks()} chunks")

    def ingest_paths(self, paths: list) -> None:
        paths = [Path(p) for p in paths]
        if not paths:
            return
        self.library.set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        def on_progress(stage: str, cur: int, total: int) -> None:
            self.statusBar().showMessage(stage)
            if total:
                self.progress.setRange(0, total)
                self.progress.setValue(cur)

        def done(reports) -> None:
            self.library.set_busy(False)
            self.progress.setVisible(False)
            ok = [r for r in reports if r.ok]
            bad = [r for r in reports if not r.ok]
            self.refresh_library()
            msg = f"ingested {len(ok)} document(s)"
            if bad:
                msg += f", {len(bad)} failed: " + "; ".join(
                    (r.errors or ["no blocks"])[0] for r in bad[:3])
            self.statusBar().showMessage(msg, 8000)
            if ok and self._doc_id is None:
                self.open_document(ok[0].doc_id)

        def failed(message: str) -> None:
            self.library.set_busy(False)
            self.progress.setVisible(False)
            self.statusBar().showMessage(f"ingest failed: {message}", 8000)

        from ..core.pipeline import ingest_paths as _ingest
        run_in_background(_ingest, paths, store=self.store,
                          on_finished=done, on_error=failed,
                          on_progress=on_progress)

    def open_document(self, doc_id: str) -> None:
        doc = self.store.get_document(doc_id)
        if not doc:
            return
        self._doc_id = doc_id
        pages = self.store.get_pages(doc_id)
        blocks = self.store.get_blocks(doc_id=doc_id)
        self.viewer.set_document(pages, blocks)
        if pages:
            self.editor.set_page(pages[0].get("page_id"))
        self._reload_latex()
        self.export_menu.setEnabled(True)
        self.statusBar().showMessage(
            f'opened "{doc.get("title", doc_id)}" · {len(pages)} page(s) · '
            f'{len(blocks)} block(s)', 6000)

    def delete_document(self, doc_id: str) -> None:
        def done(_res) -> None:
            if self._doc_id == doc_id:
                self._doc_id = None
                self.viewer.set_document([], [])
                self.editor.set_page(None)
                self.latex_edit.clear()
            self.refresh_library()
            self.statusBar().showMessage(f"deleted {doc_id}", 5000)

        run_in_background(self.store.delete_document, doc_id,
                          on_finished=done,
                          on_error=lambda m: self.statusBar().showMessage(
                              f"delete failed: {m}", 8000))

    # ------------------------------------------------------------------
    # tabs: latex / editor / citations
    # ------------------------------------------------------------------
    def _reload_latex(self) -> None:
        if not self._doc_id:
            return
        from ..core.blocks import blocks_to_annotated_tex, document_from_store
        doc = document_from_store(self.store, self._doc_id)
        if doc:
            self.latex_edit.setPlainText(blocks_to_annotated_tex(doc))

    def _persist_block_edit(self, payload: dict) -> None:
        from ..core.pipeline import save_block_edit

        def done(result) -> None:
            self.editor.apply_result(result)
            blocks = self.store.get_blocks(doc_id=self._doc_id) \
                if self._doc_id else []
            if self.viewer.current_page():
                self.viewer.set_document(
                    self.store.get_pages(self._doc_id), blocks)

        run_in_background(save_block_edit, self.store, payload["block_id"],
                          text=payload["text"], latex=payload["latex"],
                          kind=payload["kind"], on_finished=done,
                          on_error=lambda m: self.editor.status.setText(
                              f"save failed: {m}"))

    def _show_citation(self, citation: dict) -> None:
        self.viewer.show_citation(citation)
        self.tabs.setCurrentWidget(self.viewer)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def run_search(self, query: str) -> None:
        query = (query or "").strip()
        if not query:
            return
        self.statusBar().showMessage(f"searching: {query} …")

        def done(result) -> None:
            self.search_list.clear()
            for c in result.citations:
                item = QListWidgetItem(
                    f'[{c.source}] {c.title} · p{c.page_no} · {c.score:.3f} · '
                    f'{c.snippet[:120]}')
                item.setData(Qt.ItemDataRole.UserRole, c.to_dict())
                self.search_list.addItem(item)
            self.tabs.setCurrentWidget(self.search_list)
            self.statusBar().showMessage(
                f"{len(result.citations)} result(s) · "
                f"vector {len(result.rankers.get('vector', []))} / "
                f"bm25 {len(result.rankers.get('bm25', []))} / "
                f"graph {len(result.rankers.get('graph', []))}", 6000)

        from ..core.retrieve import search
        run_in_background(search, self.store, query, on_finished=done,
                          on_error=lambda m: self.statusBar().showMessage(
                              f"search failed: {m}", 8000))

    def _search_open(self, item: QListWidgetItem) -> None:
        citation = item.data(Qt.ItemDataRole.UserRole)
        if citation:
            if citation.get("doc_id") and citation["doc_id"] != self._doc_id:
                self.open_document(citation["doc_id"])
            self._show_citation(citation)

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    def ask(self, query: str, session_id: str | None) -> None:
        self.chat.set_busy(True)
        from ..core.rag import answer_query

        def done(result) -> None:
            self.chat.show_answer(
                result.answer, [c.to_dict() for c in result.citations],
                elapsed_s=result.seconds)
            self.chat.session_id = result.session_id or session_id

        run_in_background(answer_query, self.store, query,
                          session_id=session_id,
                          on_finished=done,
                          on_error=lambda m: self.chat.show_error(m))

    # ------------------------------------------------------------------
    # export / tools
    # ------------------------------------------------------------------
    def export_document(self, fmt: str) -> None:
        if not self._doc_id:
            QMessageBox.information(self, "PageX", "Open a document first.")
            return
        default = f'{self._doc_id}.{ "tex" if fmt in ("tex", "clean-tex") else fmt }'
        out, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt}", str(Path(CFG.artifacts_dir) / default))
        if not out:
            return
        from ..core.blocks import document_from_store
        from ..core.latex_export import export_document as _export

        def job() -> str:
            doc = document_from_store(self.store, self._doc_id)
            path = _export(doc, out, fmt)
            return str(path)

        def done(path: str) -> None:
            self.statusBar().showMessage(f"exported {path}", 8000)

        run_in_background(job, on_finished=done,
                          on_error=lambda m: QMessageBox.warning(
                              self, "Export failed", str(m)))

    def open_settings(self) -> None:
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.statusBar().showMessage("settings saved to pagex.toml", 5000)
            self._tick_status()

    def run_probe(self) -> None:
        def job() -> str:
            # in-process (no `sys.executable tools/probe.py` path assumptions:
            # that file does not exist inside a pip install or frozen build)
            import contextlib
            import io

            from ..probe import main as probe_main
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = probe_main([])
            return f"exit={code}\n{buf.getvalue()}"

        def done(out: str) -> None:
            QMessageBox.information(self, "Hardware probe", out[-3000:])

        self.statusBar().showMessage("probing hardware (LLM + VLM)…")
        run_in_background(job, on_finished=done,
                          on_error=lambda m: QMessageBox.warning(
                              self, "Probe failed", m))

    def run_smoke(self) -> None:
        def job() -> str:
            from ..testing.smoke import main as smoke_main
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = smoke_main([])
            return f"exit={code}\n{buf.getvalue()}"

        def done(out: str) -> None:
            QMessageBox.information(self, "Smoke test", out[-3000:])

        run_in_background(job, on_finished=done,
                          on_error=lambda m: QMessageBox.warning(
                              self, "Smoke failed", m))

    def open_models_wizard(self) -> None:
        """Tools ▸ Download models… — manifest-driven download/verify/pack."""
        from .models_wizard import ModelsWizard
        ModelsWizard(self).exec()

    def _hint_missing_models(self) -> None:
        """One-shot, non-modal hint when required GGUFs are absent."""
        try:
            from ..core.model_manifest import missing_required
            missing = missing_required()
        except Exception:                      # manifest unreadable -> silent
            return
        if missing:
            self.statusBar().showMessage(
                f"model(s) not installed: {', '.join(missing)} — "
                "Tools ▸ Download models…", 20000)

    def _about(self) -> None:
        from .. import __version__ as _v
        QMessageBox.about(
            self, "About PageX",
            f"PageX {_v} - offline visual document intelligence\n"
            "photo of handwritten notes -> annotated LaTeX (SSOT) -> "
            "hybrid search + grounded RAG\n"
            "PyQt6 + llama-cpp (GGUF Q4_K_M) · no cloud, no GPU (ADR-001/002)")

    # ------------------------------------------------------------------
    # watch folder
    # ------------------------------------------------------------------
    def toggle_watch(self, on: bool) -> None:
        if on:
            folder = CFG.watch_dir
            if folder is None or not folder.exists():
                QMessageBox.information(
                    self, "Watch folder",
                    "Set a watch folder in Settings → Runtime first.")
                self.library.watch_btn.setChecked(False)
                return
            from ..core.watch import WatchState
            self._watch_state = WatchState(folder, debounce_s=0.5)
            self._watch_timer.start()
            self.statusBar().showMessage(f"watching {folder}", 6000)
        else:
            self._watch_timer.stop()
            self._watch_state = None
            self.statusBar().showMessage("watch folder off", 4000)

    def _poll_watch(self) -> None:
        if self._watch_state is None:
            return
        ready = self._watch_state.drain()
        if ready:
            self.ingest_paths([Path(p) for p in ready])

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
    def _tick_status(self) -> None:
        from ..core import models
        try:
            state = models.HEAVY.state()
            rss = models.rss_gb()
            self.model_label.setText(
                f"{state} · rss {rss:.2f} GB · tier {CFG.effective_tier} · "
                f"{CFG.effective_threads} thread(s)")
        except Exception as exc:
            self.model_label.setText(f"status unavailable: {exc}")

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event):                    # noqa: N802 - Qt API
        self._watch_timer.stop()
        self._status_timer.stop()
        wait_all(4000)
        try:
            from ..core.models import HEAVY
            HEAVY.release()
        except Exception:
            pass
        self.store.close_all()
        super().closeEvent(event)
