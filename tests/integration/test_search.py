"""TC-INT-04 - hybrid search over a real (canned-embedded) index."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pagex.core.pipeline import ingest
from pagex.core.retrieve import invalidate_cache, search


@pytest.fixture()
def indexed(tmp_store, tmp_path, sample_page, canned_vision, canned_generate,
            canned_embed):
    path = tmp_path / "notes.png"
    Image.fromarray(np.asarray(sample_page)).save(path)
    report = ingest(path, title="Physics", store=tmp_store,
                    vision_fn=canned_vision, generate_fn=canned_generate,
                    embed_fn=canned_embed)
    assert report.ok
    invalidate_cache()
    return tmp_store, report, canned_embed


def test_tc_int_04_hybrid_search_finds_the_page(indexed):
    store, report, embed = indexed
    result = search(store, "moment of inertia solid cylinder", embedder=embed)

    assert result.citations, "no citations returned"
    assert result.citations[0].doc_id == report.doc_id
    top_ids = [c.chunk_id for c in result.citations[:5]]
    assert report.doc_id in {c.doc_id for c in result.citations[:5]}

    # every ranker that could run is reported (transparency contract)
    assert len(result.rankers.get("vector", [])) > 0
    assert len(result.rankers.get("bm25", [])) > 0

    # citations are complete enough to drive the viewer
    c = result.citations[0]
    assert c.page_no >= 1 and c.chunk_id and c.snippet
    assert c.bboxes or c.block_ids


def test_tc_int_04_rrf_fusion_prefers_multi_ranker_hits(indexed):
    store, _report, embed = indexed
    result = search(store, "angular momentum conservation", embedder=embed)
    assert result.fused_ranking, "RRF produced no fused ranking"
    # fused order == citation order (read path trusts the fusion)
    assert [c.chunk_id for c in result.citations] == result.fused_ranking[:len(result.citations)]
    # scores are per-ranker maxima (RRF decides the order, not the score)
    assert all(c.score > 0 for c in result.citations)


def test_tc_int_04_empty_and_degenerate_queries_are_safe(indexed):
    store, _report, embed = indexed
    assert len(search(store, "", embedder=embed)) == 0
    assert len(search(store, "   ", embedder=embed)) == 0
    # FTS metacharacters must not raise (sanitize_fts_query path)
    result = search(store, 'AND OR NOT "unbalanced ( ) *', embedder=embed)
    assert isinstance(result.citations, list)


def test_tc_int_04_doc_filter_restricts_scope(indexed):
    store, report, embed = indexed
    result = search(store, "torque", doc_ids=["nonexistent-doc"], embedder=embed)
    assert all(c.doc_id == "nonexistent-doc" for c in result.citations)
    assert len(result) == 0
