"""TC-INT-01 / TC-INT-02 / TC-INT-03 - write path on a temp DB (canned models).

No GGUF is touched: vision/generate/embed are injected exactly the way the
GUI and CLI inject their workers, so these tests exercise the same code the
demo runs - only the model functions differ.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pagex.core.blocks import (annotated_tex_to_document,
                               blocks_to_annotated_tex, document_from_store)
from pagex.core.latex_export import export_document
from pagex.core.pipeline import ingest, ingest_paths
from pagex.core.store import Store


def _write_page(tmp_path: Path, page, name: str = "notes_page.png") -> Path:
    path = tmp_path / name
    Image.fromarray(np.asarray(page)).save(path)
    return path


# ---------------------------------------------------------------------------
# TC-INT-01: one page in, a fully indexed document out
# ---------------------------------------------------------------------------
def test_tc_int_01_ingest_indexes_everything(tmp_store, tmp_path, sample_page,
                                             canned_vision, canned_generate,
                                             canned_embed):
    path = _write_page(tmp_path, sample_page)
    report = ingest(path, title="Physics notes", store=tmp_store,
                    vision_fn=canned_vision, generate_fn=canned_generate,
                    embed_fn=canned_embed)

    assert report.ok, report.errors
    assert report.n_pages == 1 and report.n_blocks > 0
    assert report.n_chunks > 0 and report.n_kg_nodes > 0

    doc = tmp_store.get_document(report.doc_id)
    assert doc["title"] == "Physics notes" and doc["status"] == "ready"

    pages = tmp_store.get_pages(report.doc_id)
    blocks = tmp_store.get_blocks(doc_id=report.doc_id)
    assert len(pages) == 1 and len(blocks) == report.n_blocks
    assert all(b["bbox"] and len(b["bbox"]) == 4 for b in blocks)

    # chunks carry vectors and are live in the FTS index
    chunks = list(tmp_store.iter_chunks(doc_ids=[report.doc_id]))
    assert len(chunks) == report.n_chunks
    assert all(c.vec is not None and len(c.vec) == 384 for c in chunks)
    hits = tmp_store.fts_search("moment", k=10)
    assert hits, "FTS index empty after ingest"

    # annotated .tex exists and mentions the block markers (SSOT artifact)
    tex = Path(report.tex_path)
    assert tex.exists() and "%PX:block=" in tex.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# TC-INT-02: annotated LaTeX round-trip is the single source of truth
# ---------------------------------------------------------------------------
def test_tc_int_02_annotated_latex_roundtrip(tmp_store, tmp_path, sample_page,
                                             canned_vision, canned_generate,
                                             canned_embed):
    path = _write_page(tmp_path, sample_page, "roundtrip.png")
    report = ingest(path, store=tmp_store, vision_fn=canned_vision,
                    generate_fn=canned_generate, embed_fn=canned_embed)

    doc = document_from_store(tmp_store, report.doc_id)
    assert doc is not None and doc.all_blocks
    tex = blocks_to_annotated_tex(doc)

    # codec identity: annotated tex parses back to the same SSOT rows
    # (markers store integer bboxes - the codec's documented rounding)
    parsed = annotated_tex_to_document(tex, doc_id=report.doc_id)

    def _keyed(doc):
        return {b.block_id: (b.kind, b.text.strip(),
                             tuple(int(round(v)) for v in b.bbox))
                for b in doc.all_blocks}

    assert _keyed(parsed) == _keyed(doc)

    # exporter consumes the same model without re-running any model
    out = tmp_path / "out.tex"
    export_document(doc, out, "tex")
    assert out.read_text(encoding="utf-8") == tex
    out_md = tmp_path / "out.md"
    export_document(doc, out_md, "md")
    assert out_md.stat().st_size > 0


# ---------------------------------------------------------------------------
# TC-INT-03: incremental writes - doc A untouched while doc B is indexed,
# and delete only prunes B (including its KG evidence rows)
# ---------------------------------------------------------------------------
def test_tc_int_03_incremental_and_targeted_delete(tmp_store, tmp_path,
                                                   sample_page, canned_vision,
                                                   canned_generate, canned_embed):
    p1 = _write_page(tmp_path, sample_page, "a.png")
    p2 = _write_page(tmp_path, sample_page, "b.png")

    ra = ingest(p1, title="Doc A", store=tmp_store, vision_fn=canned_vision,
                generate_fn=canned_generate, embed_fn=canned_embed)
    a_chunks = [c.chunk_id for c in tmp_store.iter_chunks(doc_ids=[ra.doc_id])]
    a_nodes = tmp_store.conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
    assert a_nodes > 0

    rb = ingest(p2, title="Doc B", store=tmp_store, vision_fn=canned_vision,
                generate_fn=canned_generate, embed_fn=canned_embed)

    # A is byte-for-byte untouched by B's ingest
    a_after = [c.chunk_id for c in tmp_store.iter_chunks(doc_ids=[ra.doc_id])]
    assert a_after == a_chunks
    assert tmp_store.get_document(ra.doc_id)["status"] == "ready"
    # both are searchable at the same time
    assert {d["doc_id"] for d in tmp_store.list_documents()} == {ra.doc_id,
                                                                 rb.doc_id}
    assert tmp_store.conn.execute(
        "SELECT COUNT(*) FROM kg_nodes").fetchone()[0] >= a_nodes

    # targeted delete: B goes, A (rows + graph) stays
    tmp_store.delete_document(rb.doc_id)
    assert tmp_store.get_document(rb.doc_id) is None
    assert tmp_store.get_pages(rb.doc_id) == []
    assert list(tmp_store.iter_chunks(doc_ids=[rb.doc_id])) == []
    remaining_kg = tmp_store.conn.execute(
        "SELECT COUNT(*) FROM kg_edges WHERE evidence LIKE ?",
        (f'%"{rb.doc_id}"%',)).fetchone()[0]
    assert remaining_kg == 0, "KG edges of the deleted doc must be pruned"
    assert [c.chunk_id for c in tmp_store.iter_chunks(doc_ids=[ra.doc_id])] \
        == a_chunks


def test_tc_int_03_ingest_paths_reports_failures(tmp_store, tmp_path,
                                                 sample_page, canned_vision,
                                                 canned_generate, canned_embed):
    """One bad file must not stop the batch (progress contract)."""
    good = _write_page(tmp_path, sample_page, "good.png")
    missing = tmp_path / "gone.png"
    reports = ingest_paths([good, missing], store=tmp_store,
                           vision_fn=canned_vision,
                           generate_fn=canned_generate,
                           embed_fn=canned_embed)
    assert len(reports) == 2
    assert reports[0].ok
    assert not reports[1].ok and reports[1].errors
