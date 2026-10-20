"""Main shell: a quiet sidebar + one focused canvas.

The window is built around three destinations (Library, Reader, Ask) and a
single activity strip.  Everything expensive - ingest, search, RAG, export -
still runs through ``gui.workers``; this module only orchestrates and never
calls a model from the GUI thread.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QMainWindow, QMessageBox, QProgressBar,
                             QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from ..core.config import CFG
from ..core.store import Store
from .chat_pane import ChatPane
from .export_dialog import ExportDialog
from .library_view import LibraryView
from .reader import ReaderView
from .settings_dialog import SettingsDialog
from .sidebar import Sidebar
from .theme import apply_theme, current_theme_name
from .workers import active_thread_count, run_in_background, wait_all

log = logging.getLogger(__name__)

STAGE_LABELS = {
    "loading": "Opening document",
    "preprocess": "Preparing page",
    "understand": "Reading the page",
    "chunk": "Indexing page",
    "graph": "Linking concepts",
    "export": "Writing LaTeX",
    "ready": "Ready to search",
    "enrich": "Enriching concepts",
    "done": "Finished",
}

EXTENSIONS = {"tex": "tex", "clean-tex": "tex", "pdf": "pdf", "docx": "docx",
              "html": "html", "md": "md", "json": "json"}


class _StackCompat:
    """Small compatibility shim for integrations/tests that expect tabs.

    The product no longer uses a QTabWidget, but keeping this tiny facade lets
    older code address the three destinations without reintroducing tab chrome.
    """

    LABELS = ("Viewer", "Editor", "Search results", "LaTeX (SSOT)", "Chat")

    def __init__(self, win: "MainWindow"):
        self.win = win

    def count(self) -> int:
        return len(self.LABELS)

    def tabText(self, index: int) -> str:          # noqa: N802 - Qt API
        return self.LABELS[index] if 0 <= index < len(self.LABELS) else ""

    def currentWidget(self):                        # noqa: N802 - Qt API
        w = self.win.stack.currentWidget()
        if w is self.win.reader:
            return self.win.viewer
        if w is self.win.chat:
            return (self.win.search_list
                    if self.win.chat._mode == "find" else self.win.chat)
        return w

    def setCurrentWidget(self, widget):             # noqa: N802 - Qt API
        if widget in (self.win.library,):
            self.win.show_library()
        elif widget in (self.win.reader, self.win.viewer):
            self.win.show_reader()
        elif widget in (self.win.chat, self.win.search_list):
            self.win.show_ask()
            if widget is self.win.search_list:
                self.win.chat.set_find_mode()
        else:
            self.win.stack.setCurrentWidget(widget)


class MainWindow(QMainWindow):
    """The whole desktop shell."""

    def __init__(self, store: Store | None = None):
        super().__init__()
        self.store = store or Store()
        self.setWindowTitle("PageX")
        self.resize(1420, 900)
        self.setMinimumSize(1080, 680)
        self._doc_id: str | None = None
        self._watch_state = None
        self._cancel_event: threading.Event | None = None
        self._ingest_worker = None
        self._ingest_started = 0.0
        self._ingest_total = 0
        self._missing_models: list[str] = []
        self._build_ui()
        self._wire()
        self.refresh_library()
        self._tick_status()
        self._sync_theme_ui()
        QTimer.singleShot(0, self._hint_missing_models)
        if self.store.list_documents():
            self.show_ask()
        else:
            self.show_library()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        root.setAcceptDrops(True)
        self.setAcceptDrops(True)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.sidebar = Sidebar()
        outer.addWidget(self.sidebar)

        main = QWidget()
        main.setObjectName("root")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(18, 14, 18, 10)
        main_layout.setSpacing(10)

        self.stack = QStackedWidget()
        self.library = LibraryView(self.store)
        self.reader = ReaderView(self.store)
        self.chat = ChatPane(self.store)
        self.stack.addWidget(self.library)
        self.stack.addWidget(self.reader)
        self.stack.addWidget(self.chat)
        main_layout.addWidget(self.stack, 1)

        # The activity strip is only visible while something is happening.
        self.activity = QFrame()
        self.activity.setObjectName("activity")
        act = QHBoxLayout(self.activity)
        act.setContentsMargins(12, 7, 8, 7)
        act.setSpacing(10)
        self.activity_label = QLabel("")
        self.activity_label.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("ghostButton")
        self.cancel_btn.clicked.connect(self._cancel_ingest)
        act.addWidget(self.activity_label, 1)
        act.addWidget(self.progress)
        act.addWidget(self.cancel_btn)
        self.activity.hide()
        main_layout.addWidget(self.activity)

        outer.addWidget(main, 1)
        self.setCentralWidget(root)

        # compatibility aliases (documents, not chrome)
        self.viewer = self.reader.viewer
        self.editor = self.reader.editor
        self.latex_edit = self.reader.source_edit
        self.search_list = self.chat.results_list
        self.search_edit = self.chat.input
        self.search_btn = self.chat.send_btn
        self.theme_btn = self.sidebar.theme_btn
        self.tabs = _StackCompat(self)

        # -- status bar -------------------------------------------------
        self.model_label = QLabel("no model loaded")
        self.model_label.setObjectName("muted")
        self.db_label = QLabel("")
        self.db_label.setObjectName("muted")
        self.statusBar().addWidget(self.model_label, 1)
        self.statusBar().addPermanentWidget(self.db_label)

        self._build_menu()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._tick_status)
        self._status_timer.start(2000)

        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(2000)
        self._watch_timer.timeout.connect(self._poll_watch)

        self._page_refresh_doc: str | None = None
        self._page_refresh_timer = QTimer(self)
        self._page_refresh_timer.setSingleShot(True)
        self._page_refresh_timer.setInterval(250)
        self._page_refresh_timer.timeout.connect(self._refresh_open_reader)

    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        act_ingest = QAction("Import files…", self)
        act_ingest.setShortcut(QKeySequence.StandardKey.Open)
        act_ingest.triggered.connect(self._pick_ingest)
        act_folder = QAction("Import folder…", self)
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
        act_export = QAction("Export document…", self)
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(lambda: self.export_document())
        m_export.addAction(act_export)
        self._export_actions.append(act_export)
        m_export.setEnabled(False)

        m_view = self.menuBar().addMenu("&View")
        act_library = QAction("Library", self)
        act_library.setShortcut(QKeySequence("Ctrl+1"))
        act_library.triggered.connect(self.show_library)
        act_ask = QAction("Ask PageX", self)
        act_ask.setShortcut(QKeySequence("Ctrl+2"))
        act_ask.triggered.connect(self.show_ask)
        m_view.addAction(act_library)
        m_view.addAction(act_ask)
        m_view.addSeparator()
        self.act_light = QAction("Light theme", self)
        self.act_light.setCheckable(True)
        self.act_light.triggered.connect(lambda: self.set_theme("light"))
        self.act_dark = QAction("Dark theme", self)
        self.act_dark.setCheckable(True)
        self.act_dark.triggered.connect(lambda: self.set_theme("dark"))
        m_view.addAction(self.act_light)
        m_view.addAction(self.act_dark)

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
        self.sidebar.importRequested.connect(self._pick_ingest)
        self.sidebar.libraryRequested.connect(self.show_library)
        self.sidebar.askRequested.connect(self.show_ask)
        self.sidebar.settingsRequested.connect(self.open_settings)
        self.sidebar.themeRequested.connect(self.toggle_theme)
        self.sidebar.docActivated.connect(self.open_document)

        self.library.ingestRequested.connect(self.ingest_paths)
        self.library.docSelected.connect(self.open_document)
        self.library.docDeleted.connect(self.delete_document)
        self.library.refreshRequested.connect(self.refresh_library)
        self.library.watchToggled.connect(self.toggle_watch)

        self.reader.backRequested.connect(self.show_library)
        self.reader.askRequested.connect(self.show_ask_for_doc)
        self.reader.exportRequested.connect(lambda: self.export_document())
        self.reader.editor.saveRequested.connect(self._persist_block_edit)
        self.reader.editor.saved.connect(lambda _payload: self.reader.refresh())

        self.chat.askRequested.connect(self.ask)
        self.chat.searchRequested.connect(self.run_search)
        self.chat.citationActivated.connect(self._show_citation)

    # ------------------------------------------------------------------
    # destinations
    # ------------------------------------------------------------------
    def show_library(self) -> None:
        self.stack.setCurrentWidget(self.library)
        self.sidebar.set_active("library")
        self.refresh_library()

    def show_ask(self) -> None:
        self.stack.setCurrentWidget(self.chat)
        self.sidebar.set_active("ask")
        self.chat.input.setFocus()
        self.chat.input.setPlaceholderText("Ask anything across your notes…")

    def show_reader(self) -> None:
        self.stack.setCurrentWidget(self.reader)
        self.sidebar.set_active(None)

    def show_ask_for_doc(self, doc_id: str) -> None:
        self.chat.set_scope(doc_id)
        self.show_ask()

    # ------------------------------------------------------------------
    # theme
    # ------------------------------------------------------------------
    def set_theme(self, mode: str) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode)
        self.viewer.refresh_theme()
        self.chat.refresh_theme()
        self.refresh_library()
        self._sync_theme_ui()

    def toggle_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        self.set_theme("dark" if current_theme_name(app) != "dark" else "light")

    def _sync_theme_ui(self) -> None:
        app = QApplication.instance()
        dark = bool(app) and current_theme_name(app) == "dark"
        self.sidebar.theme_btn.setText("Light mode" if dark else "Dark mode")
        self.sidebar.theme_btn.setToolTip(
            "Switch to light theme" if dark else "Switch to dark theme")
        if hasattr(self, "act_light"):
            self.act_light.setChecked(not dark)
            self.act_dark.setChecked(dark)

    # ------------------------------------------------------------------
    # library / documents
    # ------------------------------------------------------------------
    def _pick_ingest(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import documents", "",
            "Images and PDFs (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf);;"
            "All files (*)")
        if paths:
            self.ingest_paths([Path(p) for p in paths])

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import folder")
        if folder:
            from ..core.watch import scan
            files = scan(folder)
            if files:
                self.ingest_paths(files)
            else:
                QMessageBox.information(
                    self, "PageX", f"No ingestible files in {folder}")

    def refresh_library(self) -> None:
        docs = self.store.list_documents()
        self.library.refresh_table(docs)
        self.sidebar.set_documents(docs)
        self.chat.set_documents(docs)
        self.db_label.setText(
            f"{Path(CFG.db).name} · {self.store.count_chunks()} chunks")

    def _reserve_documents(self, paths: list[Path]) -> None:
        """Make import rows visible immediately, before the first page lands."""
        from ..core.pipeline import infer_kind
        from ..core.preprocess import page_count
        from ..core.store import new_id
        for path in paths:
            resolved = Path(path).resolve()
            if not resolved.exists() or self.store.find_document_by_source(
                    str(resolved)):
                continue
            n_pages = 0
            try:
                n_pages = page_count(resolved)
            except Exception:
                pass
            self.store.put_document(new_id("doc"), resolved.stem, str(resolved),
                                    infer_kind(resolved), n_pages=n_pages,
                                    status="queued")

    def ingest_paths(self, paths: list) -> None:
        paths = [Path(p).resolve() for p in paths if Path(p).exists()]
        if not paths:
            return
        self._reserve_documents(paths)
        self.refresh_library()
        self.show_library()
        self.library.set_busy(True)
        self.sidebar.set_busy(True)
        self.activity_label.setText(
            f"Importing {len(paths)} document(s)…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel")
        self.activity.show()
        self._cancel_event = threading.Event()
        self._ingest_started = time.perf_counter()
        self._ingest_total = 0

        def on_progress(stage: str, cur: int, total: int) -> None:
            self._render_progress(stage, cur, total)

        def on_page(doc_id: str, _page_no: int) -> None:
            self.library.update_doc_status(doc_id, "indexing")
            if self._doc_id == doc_id:
                self._page_refresh_doc = doc_id
                self._page_refresh_timer.start()

        def done(reports) -> None:
            self.library.set_busy(False)
            self.sidebar.set_busy(False)
            self.progress.setVisible(False)
            self.cancel_btn.setEnabled(True)
            self.activity.hide()
            self._cancel_event = None
            self._ingest_worker = None
            reports = reports or []
            ok = [r for r in reports if getattr(r, "ok", False)]
            bad = [r for r in reports
                   if not getattr(r, "ok", False)
                   and not getattr(r, "cancelled", False)]
            self.refresh_library()
            if self._doc_id:
                self.reader.refresh()
            cancelled = any(getattr(r, "cancelled", False) for r in reports)
            if cancelled:
                msg = (f"Import paused after {sum(getattr(r, 'pages_done', 0) for r in reports)} "
                       f"page(s); partial results are searchable.")
            else:
                msg = f"Imported {len(ok)} document(s)"
            if bad:
                first = (getattr(bad[0], "errors", None) or ["no blocks"])[0]
                msg += f" · {len(bad)} failed: {first}"
            self.statusBar().showMessage(msg, 9000)
            self.sidebar.set_status(
                "Paused" if cancelled else "Ready", error=bool(bad) and not cancelled)

        def failed(message: str) -> None:
            self.library.set_busy(False)
            self.sidebar.set_busy(False)
            self.progress.setVisible(False)
            self.activity.hide()
            self._cancel_event = None
            self._ingest_worker = None
            self.refresh_library()
            self.statusBar().showMessage(f"Import failed: {message}", 9000)
            self.sidebar.set_status("Import failed", error=True)

        from ..core.pipeline import ingest_paths as _ingest
        self._ingest_worker = run_in_background(
            _ingest, paths, store=self.store,
            cancel=self._cancel_event.is_set,
            on_finished=done, on_error=failed,
            on_progress=on_progress, on_page=on_page)

    def _cancel_ingest(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Stopping…")
            self.activity_label.setText("Stopping after the current page…")

    def _refresh_open_reader(self) -> None:
        if self._page_refresh_doc and self._doc_id == self._page_refresh_doc:
            self.reader.refresh()

    def _render_progress(self, stage: str, cur: int, total: int) -> None:
        raw = str(stage or "")
        # ingest_paths prefixes "[i/n] name: stage"; keep the useful tail
        label = raw.rsplit(":", 1)[-1].strip() if ":" in raw else raw.strip()
        pretty = STAGE_LABELS.get(label, label.title() if label else "Working")
        if total <= 0:
            self.progress.setRange(0, 0)
            self.activity_label.setText(pretty)
            return
        if total != self._ingest_total:
            self._ingest_total = total
            self.progress.setRange(0, total)
        self.progress.setValue(max(0, min(cur, total)))
        extra = f" · {cur}/{total}" if total > 1 else ""
        eta = ""
        if cur > 0 and total > cur and self._ingest_started:
            elapsed = time.perf_counter() - self._ingest_started
            remaining = elapsed / cur * (total - cur)
            if remaining >= 1:
                eta = f" · ~{int(remaining)}s left"
        self.activity_label.setText(f"{pretty}{extra}{eta}")
        if label == "ready" or label == "done":
            self.refresh_library()

    def open_document(self, doc_id: str) -> None:
        doc = self.store.get_document(doc_id)
        if not doc:
            return
        self._doc_id = doc_id
        self.reader.set_document(doc_id)
        self.show_reader()
        self.export_menu.setEnabled(True)
        self.statusBar().showMessage(
            f'Opened "{doc.get("title", doc_id)}" · '
            f'{len(self.reader._pages)} page(s) · '
            f'{len(self.reader._blocks)} block(s)', 6000)

    def delete_document(self, doc_id: str) -> None:
        def done(_res) -> None:
            if self._doc_id == doc_id:
                self._doc_id = None
                self.reader.clear()
                self.export_menu.setEnabled(False)
            self.refresh_library()
            self.statusBar().showMessage("Document deleted", 5000)

        run_in_background(self.store.delete_document, doc_id,
                          on_finished=done,
                          on_error=lambda m: self.statusBar().showMessage(
                              f"Delete failed: {m}", 8000))

    # ------------------------------------------------------------------
    # editor / citations
    # ------------------------------------------------------------------
    def _reload_latex(self) -> None:
        if self._doc_id:
            self.reader._reload_source()

    def _persist_block_edit(self, payload: dict) -> None:
        from ..core.pipeline import save_block_edit

        def done(result) -> None:
            self.editor.apply_result(result)
            self.reader.refresh()

        run_in_background(save_block_edit, self.store, payload["block_id"],
                          text=payload["text"], latex=payload["latex"],
                          kind=payload["kind"], on_finished=done,
                          on_error=lambda m: self.editor.status.setText(
                              f"save failed: {m}"))

    def _show_citation(self, citation: dict) -> None:
        doc_id = citation.get("doc_id")
        if doc_id and doc_id != self._doc_id:
            self.open_document(str(doc_id))
        if not self._doc_id:
            return
        if self.stack.currentWidget() is not self.reader:
            self.show_reader()
        self.reader.show_citation(citation)

    # ------------------------------------------------------------------
    # search / ask
    # ------------------------------------------------------------------
    def run_search(self, query: str) -> None:
        query = (query or "").strip()
        if not query:
            return
        self.show_ask()
        self.chat.set_busy(True)
        self.statusBar().showMessage(f'Searching: {query}…')
        doc_ids = self.chat.scope_doc_ids() or None

        def done(result) -> None:
            self.chat.show_results(result)
            self.show_ask()
            self.statusBar().showMessage(
                f"{len(getattr(result, 'citations', []))} result(s) · "
                f"vector {len(result.rankers.get('vector', []))} / "
                f"bm25 {len(result.rankers.get('bm25', []))} / "
                f"graph {len(result.rankers.get('graph', []))}", 6000)

        from ..core.retrieve import search
        run_in_background(search, self.store, query, doc_ids=doc_ids,
                          on_finished=done,
                          on_error=lambda m: (
                              self.chat.show_error(m),
                              self.statusBar().showMessage(
                                  f"Search failed: {m}", 8000)))

    def _search_open(self, item) -> None:
        citation = item.data(Qt.ItemDataRole.UserRole)
        if citation:
            self._show_citation(dict(citation))

    def ask(self, query: str, session_id: str | None = None,
            doc_ids: list[str] | None = None) -> None:
        self.show_ask()
        self.chat.set_busy(True)
        from ..core.rag import answer_query

        def done(result) -> None:
            self.chat.show_answer(
                result.answer, [c.to_dict() for c in result.citations],
                elapsed_s=result.seconds)
            self.chat.session_id = result.session_id or session_id
            self.statusBar().showMessage(
                "Answer grounded in your library", 5000)

        run_in_background(answer_query, self.store, query,
                          session_id=session_id,
                          doc_ids=doc_ids or None,
                          on_finished=done,
                          on_error=lambda m: self.chat.show_error(m))

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export_document(self, fmt: str | None = None,
                        pdf_layout: str | None = None) -> None:
        if not self._doc_id:
            QMessageBox.information(self, "PageX", "Open a document first.")
            return
        layout = pdf_layout or "organized"
        if fmt is None:
            dlg = ExportDialog(self)
            if not dlg.exec():
                return
            fmt, layout = dlg.selection()
        ext = EXTENSIONS.get(fmt, fmt)
        suffix = "_layout" if (fmt == "pdf" and layout == "faithful") else ""
        default = f"{self._doc_id}{suffix}.{ext}"
        filters = {
            "pdf": "PDF (*.pdf)", "docx": "Word (*.docx)",
            "html": "HTML (*.html)", "md": "Markdown (*.md)",
            "json": "JSON (*.json)", "tex": "LaTeX (*.tex)",
            "clean-tex": "LaTeX (*.tex)",
        }
        out, _ = QFileDialog.getSaveFileName(
            self, "Export document",
            str(Path(CFG.artifacts_dir) / default),
            filters.get(fmt, "All files (*)"))
        if not out:
            return
        from ..core.blocks import document_from_store
        from ..core.latex_export import export_document as _export

        def job() -> str:
            doc = document_from_store(self.store, self._doc_id)
            if doc is None:
                raise ValueError("document could not be loaded")
            path = _export(doc, out, fmt, pdf_layout=layout)
            return str(path)

        def done(path: str) -> None:
            self.statusBar().showMessage(f"Exported {path}", 9000)

        run_in_background(
            job, on_finished=done,
            on_error=lambda m: QMessageBox.warning(
                self, "Export failed", str(m)))

    # ------------------------------------------------------------------
    # tools / settings
    # ------------------------------------------------------------------
    def open_settings(self) -> None:
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.statusBar().showMessage("Settings saved", 5000)
            self._tick_status()

    def run_probe(self) -> None:
        def job() -> str:
            import contextlib
            import io

            from ..probe import main as probe_main
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = probe_main([])
            return f"exit={code}\n{buf.getvalue()}"

        def done(out: str) -> None:
            QMessageBox.information(self, "Hardware probe", out[-3000:])

        self.statusBar().showMessage("Probing hardware (LLM + VLM)…")
        run_in_background(job, on_finished=done,
                          on_error=lambda m: QMessageBox.warning(
                              self, "Probe failed", m))

    def run_smoke(self) -> None:
        def job() -> str:
            import contextlib
            import io

            from ..testing.smoke import main as smoke_main
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
        from .models_wizard import ModelsWizard
        ModelsWizard(self).exec()

    def _hint_missing_models(self) -> None:
        try:
            from ..core.model_manifest import missing_required
            missing = missing_required()
        except Exception:
            return
        if missing:
            self._missing_models = list(missing)
            self.sidebar.set_status(
                f"{len(missing)} model(s) missing · Tools → Download models",
                error=True)
            self.statusBar().showMessage(
                f"Model(s) not installed: {', '.join(missing)} — "
                "Tools ▸ Download models…", 20000)

    def _about(self) -> None:
        from .. import __version__ as _v
        QMessageBox.about(
            self, "About PageX",
            f"PageX {_v} — offline visual document intelligence\n"
            "Handwriting and layouts → annotated LaTeX → hybrid search + "
            "grounded answers.\n"
            "Runs fully local with PyQt6 + llama.cpp GGUF models.")

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
            self.statusBar().showMessage(f"Watching {folder}", 6000)
        else:
            self._watch_timer.stop()
            self._watch_state = None
            self.statusBar().showMessage("Watch folder off", 4000)

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
                f"{state} · {rss:.2f} GB · tier {CFG.effective_tier} · "
                f"{CFG.effective_threads} thread(s)")
            if not self.activity.isVisible():
                if self._missing_models:
                    self.sidebar.set_status(
                        f"{len(self._missing_models)} model(s) missing · "
                        "Tools → Download models", error=True)
                else:
                    self.sidebar.set_status(
                        f"{state} · tier {CFG.effective_tier} · offline")
        except Exception as exc:
            self.model_label.setText(f"status unavailable: {exc}")

    # ------------------------------------------------------------------
    # drag & drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):                # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):                     # noqa: N802 - Qt API
        from ..core.watch import scan
        paths: list[Path] = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths.extend(scan(p))
            elif p.is_file():
                paths.append(p)
        if paths:
            self.ingest_paths(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event):                    # noqa: N802 - Qt API
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._watch_timer.stop()
        self._status_timer.stop()
        self._page_refresh_timer.stop()
        drained = wait_all(8000)
        if not drained or active_thread_count():
            # A model call can outlive the close animation.  Never close the
            # per-thread SQLite connections or evict a model underneath it.
            log.warning("closing while %d worker(s) are still active",
                        active_thread_count())
        else:
            try:
                from ..core.models import HEAVY
                HEAVY.release()
            except Exception:
                pass
            self.store.close_all()
        super().closeEvent(event)
