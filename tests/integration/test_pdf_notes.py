"""Real-world document tests: *Artificial Intelligence handwitten notes.pdf*.

The repository ships a 21 MB, 126-page scan of handwritten AI notes with
**no text layer** - exactly the input PageX exists for.  It is user data (not
committed), so every test here is skipped when the file is absent (CI runs
without it).

What this file covers that the synthetic fixtures cannot:

* PyMuPDF rasterisation of a large real scanned PDF (200 dpi, RGB),
* the WP2 preprocessing chain on a genuine scan (contrast, deskew, upscale),
* the full ingest write path over several real pages - load -> preprocess ->
  VLM (canned) -> blocks -> annotated .tex -> chunks -> KG -> SQLite,
* ``max_pages`` capping, which is what makes a 126-page scan indexable on an
  8 GB laptop instead of exploding in one go.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pagex.core.pipeline import ingest
from pagex.core.preprocess import (PDF_DPI, load_pages, load_pdf_page,
                                   pdf_page_count, preprocess_page)
from pagex.core.retrieve import search

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "Artificial Intelligence handwitten notes.pdf"

pytestmark = pytest.mark.skipif(
    not PDF.exists(),
    reason="Artificial Intelligence handwitten notes.pdf not present "
           "(local-only scan,22 MB)")


def _fitz():
    try:                                    # pymupdf >= 1.24
        import pymupdf as fitz
    except ImportError:                     # pragma: no cover
        import fitz
    return fitz


# ---------------------------------------------------------------------------
# the document itself
# ---------------------------------------------------------------------------
def test_pdf_is_a_large_scanned_notebook():
    assert pdf_page_count(PDF) >= 100, "expected the full ~126-page scan"
    with _fitz().open(str(PDF)) as doc:
        text = doc[0].get_text().strip()
        rect = doc[0].rect
        assert len(doc) == pdf_page_count(PDF)
        # a scan: no text layer at all -> OCR/VLM is the only way to read it
        assert text == "", "this fixture is meant to be a pure scan"
        # US-Letter-ish page, portrait
        assert 0.6 < rect.width / rect.height < 0.8
        assert doc[0].get_images(), "page must embed the scanned image"


def test_pdf_single_page_renders_rgb_with_ink():
    img = load_pdf_page(PDF, 0, dpi=PDF_DPI)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    h, w = img.shape[:2]
    # 542x792 pt at 200 dpi -> ~1506 x 2200 px
    assert 1400 <= w <= 1700 and 2100 <= h <= 2300

    gray = img.mean(axis=2)
    ink = float((gray < 180).mean())          # fraction of dark pixels
    assert 0.005 < ink < 0.7, f"implausible ink coverage: {ink:.3f}"
    assert gray.std() > 8, "page looks blank after rasterisation"

    # channel order really is RGB (a BGR mix-up would make scans look odd)
    assert float(img[..., 0].mean()) != float(img[..., 2].mean()) or ink > 0.05


def test_pdf_load_pages_respects_max_pages():
    pages = load_pages(PDF, max_pages=2)
    assert len(pages) == 2, "max_pages must cap a 126-page document"
    assert all(p.ndim == 3 and p.dtype == np.uint8 for p in pages)


def test_pdf_preprocess_real_scan():
    img = load_pdf_page(PDF, 1, dpi=PDF_DPI)
    pre = preprocess_page(img)

    assert set(pre) >= {"image", "gray", "width", "height", "angle",
                        "scale", "contrast"}
    assert pre["image"].shape[:2] == (pre["height"], pre["width"])
    assert pre["width"] >= 1000, "page must not be downscaled"
    assert abs(pre["angle"]) <= 20.0, "deskew must not spin a straight scan"
    assert pre["contrast"] > 5.0, "CLAHE-less output would fail here"
    assert pre["image"].dtype == np.uint8


# ---------------------------------------------------------------------------
# full pipeline on real pages (canned VLM/LLM/embedder: no GGUF, no network)
# ---------------------------------------------------------------------------
def test_pdf_ingest_first_pages_end_to_end(tmp_store, canned_vision,
                                           canned_generate, canned_embed):
    report = ingest(PDF, title="Artificial Intelligence notes",
                    handwritten=True, store=tmp_store, max_pages=3,
                    vision_fn=canned_vision, generate_fn=canned_generate,
                    embed_fn=canned_embed)

    assert report.ok, report.errors
    assert report.kind == "handwritten"
    assert report.n_pages == 3, "max_pages must bound the run"
    assert report.n_blocks > 0 and report.n_chunks > 0
    assert report.n_kg_nodes > 0 and report.n_kg_edges > 0

    doc = tmp_store.get_document(report.doc_id)
    assert doc["status"] == "ready" and doc["n_pages"] == 3
    assert len(tmp_store.get_pages(report.doc_id)) == 3

    # the SSOT artifact exists and carries block markers
    tex = Path(report.tex_path)
    assert tex.exists() and "%PX:block=" in tex.read_text(encoding="utf-8")

    # chunks are embedded and live in the FTS index
    chunks = list(tmp_store.iter_chunks(doc_ids=[report.doc_id]))
    assert len(chunks) == report.n_chunks
    assert all(c.vec is not None and len(c.vec) == 384 for c in chunks)
    assert tmp_store.fts_search("moment", k=10), "FTS empty after real ingest"

    # rendered page artifacts landed in the configured artifacts dir
    pages = tmp_store.get_pages(report.doc_id)
    from pagex.core.config import CFG
    for p in pages:
        assert (CFG.artifacts_dir / p["image_path"]).exists()

    # hybrid search runs over the real corpus and cites this document
    result = search(tmp_store, "moment of inertia", k=5,
                    embedder=canned_embed)
    assert result.citations, "hybrid search returned nothing on a real scan"
    assert result.citations[0].doc_id == report.doc_id
    assert {"vector", "bm25"} <= set(result.rankers)


def test_pdf_digital_ingest_and_second_run_is_incremental(tmp_store,
                                                          canned_vision,
                                                          canned_generate,
                                                          canned_embed):
    first = ingest(PDF, handwritten=False, store=tmp_store, max_pages=1,
                   vision_fn=canned_vision, generate_fn=canned_generate,
                   embed_fn=canned_embed)
    assert first.kind == "pdf" and first.n_pages == 1

    # re-ingesting the same source replaces only that document's rows
    second = ingest(PDF, handwritten=False, store=tmp_store, max_pages=1,
                    vision_fn=canned_vision, generate_fn=canned_generate,
                    embed_fn=canned_embed)
    assert second.doc_id == first.doc_id
    docs = [d for d in tmp_store.list_documents()
            if d["doc_id"] == first.doc_id]
    assert len(docs) == 1, "re-ingest must not duplicate the document"
    assert tmp_store.conn.execute(
        "SELECT COUNT(*) FROM documents").fetchone()[0] == 1
