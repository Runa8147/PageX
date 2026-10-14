"""WP7 - retrieval: exact cosine + BM25 + 1-hop graph, fused with RRF.

ADR-006: the vector search is a **brute-force numpy dot product** over the
float32 BLOBs in SQLite.  10k chunks x 384 dims = ~3.8 M FLOP = single-digit
milliseconds on an i3 - exact, recall 1.0, zero native dependencies.  FAISS
would add an ANN recall loss and a build dependency to save microseconds;
that trade-off is documented in docs/decisions.md.

Rank fusion is Reciprocal Rank Fusion (RRF): each ranker votes with
``1 / (k + rank)``, which needs no score normalisation between an L2 cosine,
a BM25 log-score and a graph degree - the classic reason to prefer it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .config import CFG
from .models import embed_texts
from .store import Chunk, Citation, Store

log = logging.getLogger(__name__)

SOURCES = ("vector", "bm25", "graph")


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------
def rrf_fuse(rankings: Iterable[Sequence[str]], k: int | None = None) -> list[str]:
    """Reciprocal-rank fusion of one or more ranked chunk-id lists."""
    k = k if k is not None else CFG.rrf_k
    scores: dict[str, float] = {}
    for ranking in rankings:
        for i, cid in enumerate(ranking):
            if cid is None:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


# ---------------------------------------------------------------------------
# individual rankers
# ---------------------------------------------------------------------------
def vector_ranking(store: Store, query_vec: Sequence[float], limit: int,
                   doc_ids: Sequence[str] | None = None) -> list[tuple[str, float]]:
    """Exact cosine similarity over every stored embedding (ADR-006)."""
    q = np.asarray(query_vec, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm == 0:
        return []
    q = q / norm
    ids: list[str] = []
    matrix_rows: list[np.ndarray] = []
    for chunk in store.iter_chunks(with_vec=True, doc_ids=doc_ids):
        if not chunk.vec:
            continue
        ids.append(chunk.chunk_id)
        matrix_rows.append(chunk.vec)
    if not matrix_rows:
        return []
    mat = np.asarray(matrix_rows, dtype=np.float32)
    if mat.shape[1] != q.shape[0]:        # embedder changed -> stale vectors
        log.warning("vec dim mismatch (%d vs %d); skipping vector ranker",
                    mat.shape[1], q.shape[0])
        return []
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    sims = (mat @ q) / norms
    order = np.argsort(-sims)[:limit]
    return [(ids[int(i)], float(sims[int(i)])) for i in order]


def bm25_ranking(store: Store, query: str, limit: int = 50) -> list[tuple[str, float]]:
    """BM25 from the FTS5 virtual table (scores are negative-better)."""
    rows = store.fts_search(query, k=limit)
    return [(cid, -score) for cid, score in rows]      # higher = better


def graph_ranking(store: Store, query: str, limit: int = 50) -> list[tuple[str, float]]:
    """1-hop knowledge-graph expansion (WP6).

    Seeds = query concepts matched against ``kg_nodes``; each edge points at
    the ``evidence`` chunk ids that produced it.
    """
    seeds = store.kg_nodes_search(query, limit=8)
    if not seeds:
        return []
    seed_ids = [s["node_id"] for s in seeds]
    _nodes, edges = store.kg_subgraph(seed_ids)
    if not edges:
        return []
    # rank edges by how close they are to a seed, then harvest evidence
    seed_rank = {nid: i for i, nid in enumerate(seed_ids)}
    scored: dict[str, float] = {}
    for edge in edges:
        rank = min(seed_rank.get(edge["src"], 99), seed_rank.get(edge["dst"], 99))
        rel_bonus = 2.0 if edge["rel"] in ("HAS_FORMULA", "PART_OF") else 1.0
        weight = float(edge.get("weight") or 1.0)
        value = rel_bonus * weight / (1.0 + rank)
        for cid in _evidence_ids(edge.get("evidence")):
            scored[cid] = scored.get(cid, 0.0) + value
    ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    return [(cid, score) for cid, score in ranked[:limit]]


def _evidence_ids(evidence: str | None) -> list[str]:
    """``evidence`` holds JSON: {"chunk_ids":[...]} or a bare chunk id."""
    import json
    if not evidence:
        return []
    evidence = evidence.strip()
    if evidence.startswith("["):
        try:
            return [str(x) for x in json.loads(evidence)]
        except Exception:
            return []
    if evidence.startswith("{"):
        try:
            data = json.loads(evidence)
            if isinstance(data, dict) and "chunk_ids" in data:
                return [str(x) for x in data["chunk_ids"]]
        except Exception:
            return []
    return [evidence] if evidence.startswith("chunk:") else []


# ---------------------------------------------------------------------------
# citation assembly
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    query: str
    citations: list[Citation] = field(default_factory=list)
    fused_ranking: list[str] = field(default_factory=list)
    rankers: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.citations)

    def __len__(self) -> int:
        return len(self.citations)

    def top(self, n: int = 3) -> list[Citation]:
        return self.citations[:n]

    def to_dict(self) -> dict:
        return {"query": self.query,
                "citations": [c.to_dict() for c in self.citations],
                "rankers": {k: [(cid, round(s, 4)) for cid, s in v]
                            for k, v in self.rankers.items()}}


def _snippet(text: str, query: str, width: int = 240) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    lower = text.lower()
    pos = -1
    for term in query.split():
        pos = lower.find(term.lower())
        if pos >= 0:
            break
    if pos < 0:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix


def build_citations(store: Store, ranked_chunk_ids: Sequence[str],
                    scores: dict[str, float], source: str,
                    query: str, limit: int) -> list[Citation]:
    """chunk ids -> Citation objects with page/bbox grounding."""
    chunks = store.chunks_by_ids(ranked_chunk_ids)
    by_id = {c.chunk_id: c for c in chunks}
    out: list[Citation] = []
    for cid in ranked_chunk_ids:
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        doc = store.get_document(chunk.doc_id) or {}
        page = store.get_page(chunk.page_id) or {}
        blocks = store.blocks_by_ids(chunk.block_ids)
        bboxes = [blocks[b]["bbox"] for b in chunk.block_ids if b in blocks]
        out.append(Citation(
            doc_id=chunk.doc_id, title=doc.get("title", ""),
            page_no=int(page.get("page_no") or _page_no_from_id(chunk.page_id)),
            chunk_id=chunk.chunk_id, block_ids=chunk.block_ids, bboxes=bboxes,
            snippet=_snippet(chunk.content, query),
            score=float(scores.get(cid, 0.0)), source=source,
            page_id=chunk.page_id, image_path=page.get("image_path")))
        if len(out) >= limit:
            break
    return out


def _page_no_from_id(page_id: str) -> int:
    try:
        return int(page_id.rsplit("p", 1)[-1])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# the public search API
# ---------------------------------------------------------------------------
def search(store: Store, query: str, *, k: int | None = None,
           doc_ids: Sequence[str] | None = None, use_graph: bool = True,
           embedder=embed_texts) -> SearchResult:
    """Hybrid search: vector + BM25 (+ graph), fused with RRF."""
    k = k or CFG.top_k
    query = (query or "").strip()
    result = SearchResult(query=query)
    if not query:
        return result

    vector_ids: list[str] = []
    bm25_ids: list[str] = []
    graph_ids: list[str] = []
    scores: dict[str, float] = {}

    # 1) semantic
    try:
        vec = embedder([query])[0]
        vr = vector_ranking(store, vec, limit=max(k * 10, 50), doc_ids=doc_ids)
        vector_ids = [cid for cid, _ in vr]
        for cid, s in vr:
            scores[cid] = max(scores.get(cid, 0.0), s)
        result.rankers["vector"] = vr
    except Exception as exc:                      # embedder missing/broken
        log.warning("vector ranker skipped: %s", exc)

    # 2) keyword (BM25)
    bm = bm25_ranking(store, query, limit=max(k * 10, 50))
    bm25_ids = [cid for cid, _ in bm]
    for cid, s in bm:
        scores[cid] = max(scores.get(cid, 0.0), s)
    result.rankers["bm25"] = bm

    # 3) graph expansion
    if use_graph:
        try:
            gr = graph_ranking(store, query, limit=max(k * 6, 30))
            graph_ids = [cid for cid, _ in gr]
            for cid, s in gr:
                scores[cid] = max(scores.get(cid, 0.0), s)
            result.rankers["graph"] = gr
        except Exception as exc:
            log.debug("graph ranker skipped: %s", exc)

    if doc_ids:                       # scope BM25/graph hits to the chosen docs
        allowed = set(doc_ids)
        bm25_ids = [c for c in bm25_ids if _doc_of(store, c) in allowed]
        graph_ids = [c for c in graph_ids if _doc_of(store, c) in allowed]

    ranking = rrf_fuse([vector_ids, bm25_ids, graph_ids], k=CFG.rrf_k)
    result.fused_ranking = ranking
    # source labelling: which ranker put the chunk highest?
    first_rank: dict[str, tuple[int, str]] = {}
    for name, ids in (("vector", vector_ids), ("bm25", bm25_ids),
                      ("graph", graph_ids)):
        for i, cid in enumerate(ids):
            if cid not in first_rank or i < first_rank[cid][0]:
                first_rank[cid] = (i, name)
    limit = k
    ordered = ranking[:limit]
    citations: list[Citation] = []
    for cid in ordered:
        source = first_rank.get(cid, (999, "vector"))[1]
        citations.extend(build_citations(store, [cid], scores, source, query, 1))
    result.citations = citations
    return result


_DOC_CACHE: dict[str, str] = {}


def _doc_of(store: Store, chunk_id: str) -> str:
    if chunk_id in _DOC_CACHE:
        return _DOC_CACHE[chunk_id]
    row = store.conn.execute("SELECT doc_id FROM chunks WHERE chunk_id=?",
                             (chunk_id,)).fetchone()
    doc_id = row["doc_id"] if row else ""
    if len(_DOC_CACHE) < 5000:
        _DOC_CACHE[chunk_id] = doc_id
    return doc_id


def invalidate_cache() -> None:
    _DOC_CACHE.clear()


# ---------------------------------------------------------------------------
# evaluation helper (used by pagex/eval/retrieval.py)
# ---------------------------------------------------------------------------
def recall_at_k(ranked: Sequence[str], expected: Sequence[str], k: int = 5) -> float:
    if not expected:
        return 1.0
    top = set(ranked[:k])
    hits = sum(1 for e in expected if e in top)
    return hits / len(expected)


def reciprocal_rank(ranked: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = set(expected)
    for i, cid in enumerate(ranked[:50], start=1):
        if cid in expected_set:
            return 1.0 / i
    return 0.0
