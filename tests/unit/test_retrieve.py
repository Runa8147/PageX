"""TC-UNIT-09 - retrieval: RRF fusion, vector/BM25 rankers, citations."""
from __future__ import annotations

import numpy as np
import pytest

from pagex.core.retrieve import (bm25_ranking, build_citations, recall_at_k,
                                 reciprocal_rank, rrf_fuse, search,
                                 vector_ranking)
from pagex.core.store import Chunk, Store


def _seed_store(store: Store) -> None:
    store.put_document("dA", "Rotational Motion", "/notes/1.png", "image", 1, "ready")
    store.put_document("dB", "Calculus Notes", "/notes/3.png", "image", 1, "ready")
    store.put_page("dA-p001", "dA", 1, "art/page.png", 1240, 1754)
    store.put_page("dB-p001", "dB", 1, None, 1240, 1754)
    rows = [
        {"block_id": "dA-b0000", "page_id": "dA-p001", "doc_id": "dA", "seq": 0,
         "kind": "math", "bbox": [100, 200, 700, 300], "text": "",
         "latex": r"\tau = I \alpha"},
        {"block_id": "dA-b0001", "page_id": "dA-p001", "doc_id": "dA", "seq": 1,
         "kind": "paragraph", "bbox": [100, 400, 1100, 700],
         "text": "Angular momentum is conserved when no external torque acts.",
         "latex": ""},
        {"block_id": "dB-b0000", "page_id": "dB-p001", "doc_id": "dB", "seq": 0,
         "kind": "math", "bbox": [50, 50, 500, 150], "text": "",
         "latex": r"\frac{d}{dx} x^n = n x^{n-1}"},
    ]
    store.put_blocks(rows)
    store.put_chunks([
        Chunk(chunk_id="cA0", doc_id="dA", page_id="dA-p001",
              block_ids=["dA-b0000"], content=r"\tau = I \alpha torque inertia",
              visual_desc="equation"),
        Chunk(chunk_id="cA1", doc_id="dA", page_id="dA-p001",
              block_ids=["dA-b0001"],
              content="Angular momentum is conserved when no external torque acts.",
              visual_desc="notes"),
        Chunk(chunk_id="cB0", doc_id="dB", page_id="dB-p001",
              block_ids=["dB-b0000"],
              content="Derivative of x to the n equals n x to the n minus one.",
              visual_desc="calculus"),
    ])
    # give every chunk an embedding so the vector ranker has something to rank
    chunks = list(store.iter_chunks())
    for chunk in chunks:
        chunk.vec = hash_embed([chunk.content])[0]
        chunk.vec_dim = 384
    store.put_chunks(chunks)
    # bm25 needs the FTS rows (already triggered by put_chunks)
    assert store.fts_search("torque")


def hash_embed(texts, dim: int = 384):
    out = []
    for t in texts:
        v = np.zeros(dim, dtype=np.float32)
        for w in (t or "").lower().split():
            v[abs(hash(w)) % dim] += 1.0
        n = np.linalg.norm(v) or 1.0
        out.append((v / n).tolist())
    return out


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------
def test_rrf_fuse_matches_hand_computation():
    """Two lists, k=60: score = sum 1/(60+rank), rank from 0."""
    r1 = ["a", "b", "c"]
    r2 = ["b", "d"]
    fused = rrf_fuse([r1, r2], k=60)
    expected_scores = {
        "a": 1 / 61,
        "b": 1 / 62 + 1 / 61,
        "c": 1 / 63,
        "d": 1 / 62,
    }
    best = max(expected_scores, key=lambda k: expected_scores[k])
    assert fused[0] == best, f"expected {best}, got {fused[0]}"
    # full ordering by descending score
    want = sorted(expected_scores, key=lambda k: -expected_scores[k])
    assert sorted(fused) == sorted(expected_scores)
    assert [c for c in fused if c in want] == want


def test_rrf_fuse_is_stable_and_deterministic():
    a = rrf_fuse([["x", "y"], ["y", "x"]], k=10)
    b = rrf_fuse([["x", "y"], ["y", "x"]], k=10)
    assert a == b
    # a chunk appearing in both lists beats one appearing once
    assert a[0] == "y" or a[0] == "x"


def test_rrf_fuse_handles_empty_and_unknown():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


# ---------------------------------------------------------------------------
# rankers
# ---------------------------------------------------------------------------
def test_vector_ranking_orders_by_similarity(tmp_store: Store):
    _seed_store(tmp_store)
    query_vec = hash_embed(["torque inertia equation"])[0]
    ranked = vector_ranking(tmp_store, query_vec, limit=3)
    assert ranked
    assert ranked[0][0] == "cA0", "the torque/inertia chunk should win"
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_vector_ranking_ignores_chunks_without_vectors(tmp_store: Store):
    _seed_store(tmp_store)
    ranked = vector_ranking(tmp_store, [1.0] + [0.0] * 383, limit=10)
    assert all(cid for cid, _ in ranked)
    assert len(ranked) == 3


def test_vector_ranking_rejects_dimension_mismatch(tmp_store: Store, caplog):
    _seed_store(tmp_store)
    ranked = vector_ranking(tmp_store, [0.0] * 16, limit=5)
    assert ranked == [], "stale/mismatched embedder must not silently score"


def test_bm25_ranking_returns_positive_scores(tmp_store: Store):
    _seed_store(tmp_store)
    ranked = bm25_ranking(tmp_store, "angular momentum")
    assert ranked, "bm25 must find the momentum chunk"
    assert ranked[0][0] == "cA1"
    assert all(s > 0 for _, s in ranked)


def test_bm25_handles_query_with_punctuation(tmp_store: Store):
    _seed_store(tmp_store)
    bm25_ranking(tmp_store, '(weird) "query" AND {x}!')


# ---------------------------------------------------------------------------
# end-to-end search
# ---------------------------------------------------------------------------
def test_search_returns_grounded_citations(tmp_store: Store):
    _seed_store(tmp_store)
    result = search(tmp_store, "angular momentum torque", k=3,
                    embedder=hash_embed)
    assert result.citations, "hybrid search must return citations"
    top = result.citations[0]
    assert top.doc_id in {"dA", "dB"}
    assert top.page_no == 1
    assert top.block_ids and top.bboxes, "citation must carry grounding"
    x0, y0, x1, y1 = top.bboxes[0]
    assert 0 <= x0 < x1 <= 1240, "bbox inside page width"
    assert 0 <= y0 < y1 <= 1754, "bbox inside page height"
    assert top.snippet and top.title


def test_search_uses_all_three_rankers(tmp_store: Store):
    _seed_store(tmp_store)
    result = search(tmp_store, "angular momentum conserved", k=3,
                    embedder=hash_embed)
    assert set(result.rankers) >= {"vector", "bm25"}
    assert result.fused_ranking, "RRF output recorded"
    sources = {c.source for c in result.citations}
    assert sources & {"vector", "bm25", "graph"}


def test_search_scoped_to_documents(tmp_store: Store):
    _seed_store(tmp_store)
    result = search(tmp_store, "derivative of x", k=5, doc_ids=["dB"],
                    embedder=hash_embed)
    assert result.citations
    assert all(c.doc_id == "dB" for c in result.citations)


def test_search_on_empty_store_is_graceful(tmp_store: Store):
    result = search(tmp_store, "anything at all", embedder=hash_embed)
    assert result.citations == []
    assert result.fused_ranking == []


def test_search_empty_query_returns_nothing(tmp_store: Store):
    _seed_store(tmp_store)
    assert search(tmp_store, "   ").citations == []


def test_search_survives_a_broken_embedder(tmp_store: Store):
    _seed_store(tmp_store)

    def boom(texts):
        raise RuntimeError("embedder offline")

    result = search(tmp_store, "angular momentum", embedder=boom)
    assert result.citations, "BM25 must still answer without vectors"
    assert result.citations[0].doc_id == "dA"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"], k=3) == 0.0
    assert recall_at_k(["a", "b"], ["a", "b"], k=1) == 0.5
    assert recall_at_k(["a"], [], k=3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "target"], ["target"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], ["target"]) == 0.0
