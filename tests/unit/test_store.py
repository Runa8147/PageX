"""TC-UNIT-05 / TC-UNIT-08 - store schema, FTS5 triggers, chunking, KG."""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from pagex.core.blocks import Block, block_id_for, make_document, make_page
from pagex.core.pipeline import (approx_tokens, chunk_blocks, make_chunks,
                                 visual_description)
from pagex.core.store import Chunk, Store, new_id


# ---------------------------------------------------------------------------
# schema / FTS5
# ---------------------------------------------------------------------------
def test_schema_creates_every_table_and_trigger(tmp_store: Store):
    names = {r["name"] for r in tmp_store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')")}
    for need in ("documents", "pages", "blocks", "chunks", "chunks_fts",
                 "kg_nodes", "kg_edges", "chat_sessions", "chat_messages",
                 "chunks_ai", "chunks_ad", "chunks_au"):
        assert need in names, f"{need} missing from schema"
    # WAL on (atomic incremental writes, ADR-003)
    mode = tmp_store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_fts5_available_and_synchronised(tmp_store: Store):
    tmp_store.put_document("d1", "Doc One", "/x.png", "image", 1, "ready")
    tmp_store.put_page("d1-p001", "d1", 1, None, 100, 100)
    tmp_store.put_chunks([
        Chunk(chunk_id="c1", doc_id="d1", page_id="d1-p001", block_ids=[],
              content="angular momentum is conserved", visual_desc="notes"),
        Chunk(chunk_id="c2", doc_id="d1", page_id="d1-p001", block_ids=[],
              content="registration form fields", visual_desc="form")])

    assert tmp_store.fts_search("momentum"), "insert trigger should index"
    assert tmp_store.fts_search("registration")
    assert not tmp_store.fts_search("zebra")

    # DELETE trigger
    tmp_store.conn.execute("DELETE FROM chunks WHERE chunk_id='c1'")
    tmp_store.conn.commit()
    assert not tmp_store.fts_search("momentum"), "delete trigger should purge"
    assert tmp_store.fts_search("registration")

    # UPDATE trigger
    tmp_store.conn.execute("UPDATE chunks SET content='quiet refrigerator' "
                           "WHERE chunk_id='c2'")
    tmp_store.conn.commit()
    assert tmp_store.fts_search("refrigerator")
    assert not tmp_store.fts_search("registration")


def test_fts_query_sanitiser_never_raises(tmp_store: Store):
    store = tmp_store
    assert store.sanitize_fts_query('weird "quote (paren) AND {brace}') 
    assert store.sanitize_fts_query("!!!") == ""
    assert store.sanitize_fts_query("") == ""
    # even a hostile query must not blow up the search
    store.fts_search('"unbalanced')
    store.fts_search("NEAR(")


def test_documents_and_status_updates(tmp_store: Store):
    store = tmp_store
    store.put_document("d1", "Title", "/a.png", "image", status="pending")
    assert store.get_document("d1")["status"] == "pending"
    store.set_status("d1", "indexing", n_pages=3)
    doc = store.get_document("d1")
    assert doc["status"] == "indexing" and doc["n_pages"] == 3
    store.set_status("d1", "ready")
    assert store.get_document("d1")["status"] == "ready"
    assert store.find_document_by_source("/a.png")["doc_id"] == "d1"
    assert store.find_document_by_source("/nope.png") is None


def test_blocks_roundtrip_via_store(tmp_store: Store):
    store = tmp_store
    store.put_document("d1", "T", "/a.png", "image", 1, "ready")
    store.put_page("d1-p001", "d1", 1, None, 1240, 1754)
    store.put_blocks([{"block_id": "d1-b0001", "page_id": "d1-p001",
                       "doc_id": "d1", "seq": 0, "kind": "math",
                       "bbox": [10.5, 20.5, 300.0, 120.0], "text": "",
                       "latex": r"E=mc^2", "confidence": 0.9,
                       "meta": {"source": "page"}}])
    (block,) = store.get_blocks(page_id="d1-p001")
    assert block["bbox"] == [10.5, 20.5, 300.0, 120.0]
    assert block["latex"] == r"E=mc^2"
    by_id = store.blocks_by_ids(["d1-b0001"])
    assert by_id["d1-b0001"]["kind"] == "math"


def test_chunk_vector_blob_roundtrip(tmp_store: Store):
    store = tmp_store
    store.put_document("d1", "T", "/a.png", "image", 1, "ready")
    store.put_page("d1-p001", "d1", 1, None, 10, 10)
    vec = [float(i) / 100 for i in range(384)]
    store.put_chunks([Chunk(chunk_id="c1", doc_id="d1", page_id="d1-p001",
                            block_ids=["b"], content="hello", vec=vec,
                            vec_dim=384, token_count=2)])
    (chunk,) = list(store.iter_chunks())
    assert chunk.vec_dim == 384
    assert len(chunk.vec) == 384
    assert np.allclose(chunk.vec, vec, atol=1e-6), "float32 blob survives"


def test_kg_upsert_merges_and_keeps_max_weight(tmp_store: Store):
    store = tmp_store
    nodes = [{"node_id": "concept:torque", "kind": "concept", "label": "torque"}]
    store.put_kg(nodes, [{"src": "a", "dst": "b", "rel": "CO_OCCURS",
                          "weight": 1.0, "evidence": "c1"}])
    store.put_kg(nodes, [{"src": "a", "dst": "b", "rel": "CO_OCCURS",
                          "weight": 5.0, "evidence": "c2"}])
    edges = store.kg_neighbors(["a"])
    assert len(edges) == 1 and edges[0]["weight"] == 5.0
    assert store.kg_stats()["nodes"] == 1
    assert store.kg_nodes_search("torq")
    assert store.kg_nodes_search("nonexistentword") == []


def test_delete_document_cascades_and_prunes_orphan_nodes(tmp_store: Store):
    store = tmp_store
    store.put_document("d1", "T", "/a.png", "image", 1, "ready")
    store.put_page("d1-p001", "d1", 1, None, 10, 10)
    store.put_blocks([{"block_id": "b1", "page_id": "d1-p001", "doc_id": "d1",
                       "seq": 0, "kind": "paragraph", "bbox": [0, 0, 1, 1],
                       "text": "x"}])
    store.put_chunks([Chunk(chunk_id="c1", doc_id="d1", page_id="d1-p001",
                            block_ids=["b1"], content="x")])
    store.put_kg([{"node_id": "concept:orphan", "kind": "concept", "label": "orphan"}],
                 [{"src": "document:d1", "dst": "concept:orphan",
                   "rel": "MENTIONS",
                   "evidence": json.dumps({"doc_id": "d1",
                                           "chunk_ids": ["c1"]})}])
    store.delete_document("d1")
    assert store.get_document("d1") is None
    assert store.count_chunks() == 0
    assert store.get_blocks(doc_id="d1") == []
    assert store.kg_stats() == {"nodes": 0, "edges": 0}


def test_one_connection_per_thread(tmp_store: Store):
    """Threading rule: no shared connection, no check_same_thread=False."""
    store = tmp_store
    main_conn = store.conn
    seen = {}

    def worker():
        seen["conn"] = store.conn
        store.put_document(new_id("doc"), "t", "/t.png", "image")
        store.close()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["conn"] is not main_conn, "each thread needs its own connection"
    assert store.conn is main_conn, "main thread connection untouched"
    assert len(store.list_documents()) == 1


def test_chat_persistence(tmp_store: Store):
    store = tmp_store
    sid = store.new_session()
    store.add_message(sid, "user", "hello")
    store.add_message(sid, "assistant", "hi", citations=[])
    hist = store.history(sid)
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert store.new_session(sid) == sid


# ---------------------------------------------------------------------------
# chunking (TC-UNIT-08)
# ---------------------------------------------------------------------------
def _blocks(kinds_and_text) -> list[Block]:
    out = []
    for i, (kind, text) in enumerate(kinds_and_text):
        out.append(Block(block_id=f"d-b{i:04d}", page_id="d-p001", doc_id="d",
                         seq=i, kind=kind, bbox=(0, 0, 10, 10),
                         text="" if kind in ("math", "table", "diagram") else text,
                         latex=text if kind in ("math", "table", "diagram") else ""))
    return out


def test_math_block_is_never_merged():
    blocks = _blocks([
        ("paragraph", "some lead in text"),
        ("math", r"\tau = I \alpha"),
        ("paragraph", "explanation follows the equation"),
    ])
    groups = chunk_blocks(blocks, page_caption="", page_no=1, max_tokens=400)
    flat = [[b.kind for b in g] for g in groups]
    math_groups = [g for g in flat if "math" in g]
    assert math_groups and all(g == ["math"] for g in math_groups), \
        "equations must sit alone in their chunk"


def test_chunks_respect_token_budget():
    blocks = _blocks([("paragraph", "word " * 60)] * 8)   # ~150 tokens each
    groups = chunk_blocks(blocks, page_caption="", page_no=1, max_tokens=400)
    assert len(groups) > 1, "must split"
    for group in groups:
        tokens = sum(approx_tokens(b.text) for b in group)
        assert tokens <= 400, f"chunk of {tokens} tokens exceeds budget"


def test_consecutive_grouping_preserves_order():
    blocks = _blocks([("paragraph", "alpha"), ("paragraph", "beta"),
                      ("paragraph", "gamma")])
    groups = chunk_blocks(blocks, page_caption="", page_no=1, max_tokens=400)
    flat = [b for g in groups for b in g]
    assert [b.text for b in flat] == ["alpha", "beta", "gamma"]


def test_make_chunks_assigns_block_ids_and_visual_desc():
    doc = make_document("d", "Doc", kind="image")
    page = make_page(doc, 1, width=100, height=100)
    page.caption = "Handwritten notes with equations."
    page.blocks = _blocks([("heading", "Title"), ("paragraph", "body text"),
                           ("math", r"E=mc^2")])
    mapping: dict[str, str] = {}
    chunks = make_chunks(page, "d", mapping)
    assert chunks
    assert all(c.block_ids for c in chunks)
    assert all(c.chunk_id.startswith("d-p001-c") for c in chunks)
    assert set(mapping) == {b.block_id for b in page.blocks}
    assert "Handwritten notes" in chunks[0].visual_desc
    assert "math" in chunks[0].visual_desc or any(
        "math" in (c.visual_desc or "") for c in chunks)


def test_visual_description_counts_kinds():
    blocks = _blocks([("math", "a"), ("math", "b"), ("table", "c")])
    desc = visual_description("caption", blocks)
    assert "2 maths" in desc and "1 table" in desc and "caption" in desc
