"""TC-GUI-01..05 - the desktop shell, headless (QT_QPA_PLATFORM=offscreen).

These are the manual's GUI test ids:

* TC-GUI-01  window + panes construct and load a document
* TC-GUI-02  ingest runs off the UI thread (progress + completion signals)
* TC-GUI-03  search box -> results list without freezing
* TC-GUI-04  chat worker round-trip (busy -> answer + sources)
* TC-GUI-05  citation click jumps the viewer to page + blocks
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.gui

from pagex.core.pipeline import ingest                    # noqa: E402
from pagex.core.retrieve import SearchResult, invalidate_cache  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from pagex.gui.app import make_app
    app = make_app([])
    yield app


@pytest.fixture()
def window(qapp, tmp_path_factory, sample_page, canned_vision,
           canned_generate, canned_embed):
    from pagex.core.store import Store
    from pagex.gui.main_window import MainWindow

    db = tmp_path_factory.mktemp("guidb") / "gui.db"
    store = Store(db)
    tmp = tmp_path_factory.mktemp("guiimg")
    path = tmp / "page.png"
    Image.fromarray(np.asarray(sample_page)).save(path)
    report = ingest(path, title="Physics", store=store,
                    vision_fn=canned_vision, generate_fn=canned_generate,
                    embed_fn=canned_embed)
    assert report.ok
    invalidate_cache()

    win = MainWindow(store)
    yield win, report
    win.close()
    from pagex.gui.workers import wait_all
    wait_all(4000)
    store.close_all()


def _spin(app, deadline: float = 8.0) -> None:
    """Pump the event loop until *deadline* (workers deliver via signals)."""
    end = time.time() + deadline
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def _spin_until(app, predicate, deadline: float = 8.0) -> bool:
    end = time.time() + deadline
    while time.time() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# TC-GUI-01
# ---------------------------------------------------------------------------
def test_tc_gui_01_window_constructs_and_loads_document(qapp, window):
    win, report = window
    assert win.tabs.count() == 5
    assert {"Viewer", "Editor", "LaTeX (SSOT)", "Chat"} <= {
        win.tabs.tabText(i) for i in range(win.tabs.count())}
    assert win.store.count_chunks() > 0
    assert win.library.table.rowCount() == 1

    win.open_document(report.doc_id)
    assert win.viewer.current_page() is not None
    assert win.viewer._rects, "block overlay rectangles missing"
    assert "%PX:block=" in win.latex_edit.toPlainText()
    assert win.editor.listw.count() > 0
    assert win.export_menu.isEnabled()


# ---------------------------------------------------------------------------
# TC-GUI-02
# ---------------------------------------------------------------------------
def test_tc_gui_02_ingest_runs_on_a_worker_thread(qapp, window, monkeypatch,
                                                  sample_page, canned_vision,
                                                  canned_generate, canned_embed):
    from pagex.core import pipeline
    win, _report = window
    main_thread = threading.get_ident()
    seen: dict[str, object] = {}
    real_ingest = pipeline.ingest

    def canned_ingest(path, *, progress=None, store=None, **kw):
        seen["thread"] = threading.get_ident()
        if progress:
            progress("stage", 1, 2)
        kw.setdefault("vision_fn", canned_vision)
        kw.setdefault("generate_fn", canned_generate)
        kw.setdefault("embed_fn", canned_embed)
        return real_ingest(path, progress=progress, store=store, **kw)

    monkeypatch.setattr(pipeline, "ingest", canned_ingest)

    path = win.store.db_path.parent / "second.png"
    Image.fromarray(np.asarray(sample_page)).save(path)
    win.ingest_paths([path])

    # The pending row appears immediately; wait for the worker to finish.
    assert win.library.table.rowCount() == 2
    ok = _spin_until(qapp, lambda: win.library.ingest_btn.isEnabled(),
                     deadline=20.0)
    assert ok, "ingest completion never re-enabled the library"
    assert seen.get("thread") not in (None, main_thread), \
        "ingest must not run on the UI thread"
    assert win.library.table.rowCount() == 2
    assert not win.activity.isVisible()


# ---------------------------------------------------------------------------
# TC-GUI-03
# ---------------------------------------------------------------------------
def test_tc_gui_03_search_box_populates_results(qapp, window, monkeypatch):
    from pagex.core import retrieve
    win, report = window

    chunk = next(win.store.iter_chunks(doc_ids=[report.doc_id]))
    fake = SearchResult(query="moment of inertia")
    fake.citations = retrieve.build_citations(
        win.store, [chunk.chunk_id], {chunk.chunk_id: 1.0}, "vector",
        "moment of inertia", 5)
    assert fake.citations, "fixture index must yield at least one citation"
    monkeypatch.setattr(retrieve, "search", lambda *a, **k: fake)

    win.search_edit.setText("moment of inertia")
    win.run_search("moment of inertia")
    ok = _spin_until(qapp, lambda: win.search_list.count() >= 1)
    assert ok, "search results never arrived"
    assert win.tabs.currentWidget() in (win.search_list, win.viewer)
    item = win.search_list.item(0)
    from PyQt6.QtCore import Qt
    assert item.data(Qt.ItemDataRole.UserRole) is not None


# ---------------------------------------------------------------------------
# TC-GUI-04
# ---------------------------------------------------------------------------
def test_tc_gui_04_chat_worker_roundtrip(qapp, window, monkeypatch):
    from pagex.core import rag
    from pagex.core.models import GenResult
    from pagex.core.store import Citation
    win, report = window

    citation = Citation(doc_id=report.doc_id, title="Physics", page_no=1,
                        chunk_id="c", block_ids=[], bboxes=[],
                        snippet="moment of inertia", score=1.0)

    def fake_answer(store, query, **kw):
        time.sleep(0.05)                        # simulate inference latency
        return rag.RagResult(query=query,
                             answer="It depends on mass distribution "
                                    "[doc:Physics p.1]",
                             citations=[citation], session_id="s-test",
                             grounded=True, seconds=0.1)

    monkeypatch.setattr(rag, "answer_query", fake_answer)
    win.tabs.setCurrentWidget(win.chat)
    win.chat.input.setText("what is the moment of inertia?")
    win.chat._ask()

    assert win.chat._busy is True
    assert win.chat.send_btn.isEnabled() is False     # UI reflects the wait
    ok = _spin_until(qapp, lambda: not win.chat._busy)
    assert ok, "chat worker never finished"
    assert "mass distribution" in win.chat.transcript.toPlainText()
    assert win.chat.sources.count() == 1
    assert win.chat.transcript.toPlainText().count("you>") == 1


# ---------------------------------------------------------------------------
# TC-GUI-05
# ---------------------------------------------------------------------------
def test_highlight_snippet_escapes_snippet_html():
    from pagex.gui.chat_pane import _highlight_snippet

    out = _highlight_snippet("mass & <script> acceleration", "mass")
    assert "<script>" not in out
    assert "&amp;" in out and "&lt;script&gt;" in out
    assert out.count("<span") == 1
    assert "mass" in out


def test_tc_gui_06_live_page_signal_reaches_gui(qapp):
    """FnWorker injects the core on_page callback and marshals it to the UI."""
    from pagex.gui.workers import run_in_background
    seen: list[tuple[str, int]] = []

    def job(on_page=None):
        on_page("doc_test", 2)
        return True

    run_in_background(job, on_page=lambda doc_id, page_no: seen.append(
        (doc_id, page_no)))
    ok = _spin_until(qapp, lambda: bool(seen))
    assert ok and seen == [("doc_test", 2)]


def test_tc_gui_05_citation_jumps_viewer_to_blocks(qapp, window):
    win, report = window
    win.open_document(report.doc_id)
    page = win.viewer.current_page()
    blocks = win.store.get_blocks(page_id=page["page_id"])
    assert blocks

    citation = {"doc_id": report.doc_id, "page_id": page["page_id"],
                "page_no": page["page_no"], "block_ids": [blocks[0]["block_id"]],
                "title": "Physics", "snippet": "x", "score": 1.0,
                "source": "vector", "bboxes": [blocks[0]["bbox"]]}
    win._show_citation(citation)
    assert win.tabs.currentWidget() is win.viewer
    assert win.viewer.sel_label.text().startswith("highlighted")

    # editor <-> viewer selection sync does not loop
    win.editor.select_block(blocks[0]["block_id"])
    assert win.editor.current_block_id() == blocks[0]["block_id"]
    qapp.processEvents()
    win.editor.select_block(blocks[0]["block_id"])
    qapp.processEvents()
