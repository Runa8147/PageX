"""Single-file hybrid index: metadata + FTS5 + vector BLOBs + knowledge graph.

ADR-003: one SQLite file holds everything (documents, pages, blocks, chunks,
BM25 index, 384-dim embeddings as float32 BLOBs, and the knowledge graph).
No Chroma / FAISS binary dependencies, atomic incremental updates, and the
whole demo travels as one file.

Threading rule: **one connection per thread** via ``threading.local()``.
``check_same_thread`` is never disabled - that is how you get silent
corruption under a QThreadPool worker.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .config import CFG

log = logging.getLogger(__name__)

DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, source_path TEXT NOT NULL,
  kind TEXT NOT NULL, n_pages INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')),
  status TEXT DEFAULT 'pending', error_msg TEXT);
CREATE TABLE IF NOT EXISTS pages(
  page_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  page_no INTEGER NOT NULL, image_path TEXT, width INTEGER, height INTEGER,
  meta TEXT, UNIQUE(doc_id,page_no));
CREATE TABLE IF NOT EXISTS blocks(
  block_id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
  doc_id TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  bbox TEXT NOT NULL,
  text TEXT, latex TEXT, confidence REAL, meta TEXT);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, page_id TEXT NOT NULL,
  block_ids TEXT NOT NULL, content TEXT NOT NULL, visual_desc TEXT,
  vec BLOB, vec_dim INTEGER, token_count INTEGER);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  content, visual_desc, content='chunks', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid,content,visual_desc) VALUES (new.rowid,new.content,new.visual_desc); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts,rowid,content,visual_desc) VALUES ('delete',old.rowid,old.content,old.visual_desc); END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts,rowid,content,visual_desc) VALUES ('delete',old.rowid,old.content,old.visual_desc);
  INSERT INTO chunks_fts(rowid,content,visual_desc) VALUES (new.rowid,new.content,new.visual_desc); END;
CREATE TABLE IF NOT EXISTS kg_nodes(
  node_id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL, aliases TEXT);
CREATE TABLE IF NOT EXISTS kg_edges(
  src TEXT NOT NULL, dst TEXT NOT NULL, rel TEXT NOT NULL,
  weight REAL DEFAULT 1.0, evidence TEXT, PRIMARY KEY(src,dst,rel));
CREATE TABLE IF NOT EXISTS chat_sessions(session_id TEXT PRIMARY KEY, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS chat_messages(
  msg_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, citations TEXT, created_at TEXT DEFAULT (datetime('now')));
CREATE INDEX IF NOT EXISTS ix_blocks_page ON blocks(page_id,seq);
CREATE INDEX IF NOT EXISTS ix_chunks_doc  ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_edges_src ON kg_edges(src);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON kg_edges(dst);
"""

BLOCK_KINDS = ("heading", "paragraph", "math", "table", "diagram", "caption", "note")
EDGE_RELS = ("PRECEDES", "MENTIONS", "CO_OCCURS", "HAS_FORMULA", "PART_OF")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Chunk:
    """A retrieval unit: consecutive blocks (<= chunk_tokens) from one page."""
    chunk_id: str
    doc_id: str
    page_id: str
    block_ids: list[str]
    content: str
    visual_desc: str = ""
    vec: list[float] | None = None
    vec_dim: int = 0
    token_count: int = 0

    @property
    def bboxes(self) -> list[list[float]]:  # pragma: no cover - filled by caller
        return getattr(self, "_bboxes", [])


@dataclass
class Citation:
    """What the read-path hands back to RAG / the viewer."""
    doc_id: str
    title: str
    page_no: int
    chunk_id: str
    block_ids: list[str]
    bboxes: list[list[float]]
    snippet: str
    score: float
    source: str = "vector"        # vector | bm25 | graph
    page_id: str = ""
    image_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "title": self.title, "page_no": self.page_no,
            "chunk_id": self.chunk_id, "block_ids": self.block_ids,
            "bboxes": self.bboxes, "snippet": self.snippet,
            "score": round(self.score, 5), "source": self.source,
            "page_id": self.page_id, "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Citation":
        return cls(
            doc_id=d["doc_id"], title=d.get("title", ""),
            page_no=int(d.get("page_no", 0)), chunk_id=d.get("chunk_id", ""),
            block_ids=list(d.get("block_ids") or []),
            bboxes=[list(b) for b in (d.get("bboxes") or [])],
            snippet=d.get("snippet", ""), score=float(d.get("score", 0.0)),
            source=d.get("source", "vector"), page_id=d.get("page_id", ""),
            image_path=d.get("image_path"))


class Store:
    """Thread-safe wrapper around the PageX SQLite file."""

    def __init__(self, db_path: str | Path | None = None, *, create: bool = True):
        self.db_path = Path(db_path) if db_path else CFG.db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all: list[sqlite3.Connection] = []
        if create:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.init_schema()

    # ------------------------------------------------------------------
    # connections
    # ------------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        existing = getattr(self._local, "conn", None)
        if existing is not None:
            return existing
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path), timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        self._local.conn = con
        with self._lock:
            self._all.append(con)
        return con

    def close(self) -> None:
        con = getattr(self._local, "conn", None)
        if con is not None:
            try:
                con.close()
            finally:
                self._local.conn = None
                with self._lock:
                    if con in self._all:
                        self._all.remove(con)

    def close_all(self) -> None:
        with self._lock:
            cons, self._all = list(self._all), []
        for con in cons:
            try:
                con.close()
            except Exception:      # pragma: no cover - defensive
                pass
        self._local = threading.local()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc) -> None:
        self.close_all()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------
    def init_schema(self) -> None:
        con = self.conn
        con.executescript(DDL)
        con.commit()
        # fail fast if this Python build lacks FTS5 (ADR-003 depends on it)
        try:
            con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
            con.execute("DROP TABLE IF EXISTS _fts_probe")
        except sqlite3.OperationalError as exc:  # pragma: no cover
            raise RuntimeError(
                "This Python/SQLite build has no FTS5 support; PageX requires it."
            ) from exc
        log.debug("schema ready at %s", self.db_path)

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def put_document(self, doc_id: str, title: str, source_path: str,
                     kind: str, n_pages: int = 0, status: str = "pending",
                     error_msg: str | None = None) -> None:
        self.conn.execute(
            """INSERT INTO documents(doc_id,title,source_path,kind,n_pages,status,error_msg)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title,
                 n_pages=excluded.n_pages, status=excluded.status,
                 error_msg=excluded.error_msg""",
            (doc_id, title, source_path, kind, n_pages, status, error_msg))
        self.conn.commit()

    def set_status(self, doc_id: str, status: str, error_msg: str | None = None,
                   n_pages: int | None = None) -> None:
        if n_pages is None:
            self.conn.execute(
                "UPDATE documents SET status=?, error_msg=? WHERE doc_id=?",
                (status, error_msg, doc_id))
        else:
            self.conn.execute(
                "UPDATE documents SET status=?, error_msg=?, n_pages=? WHERE doc_id=?",
                (status, error_msg, n_pages, doc_id))
        self.conn.commit()

    def put_page(self, page_id: str, doc_id: str, page_no: int,
                 image_path: str | None, width: int, height: int,
                 meta: dict | None = None) -> None:
        self.conn.execute(
            """INSERT INTO pages(page_id,doc_id,page_no,image_path,width,height,meta)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(page_id) DO UPDATE SET image_path=excluded.image_path,
                 width=excluded.width, height=excluded.height, meta=excluded.meta""",
            (page_id, doc_id, page_no, image_path, width, height,
             json.dumps(meta or {})))
        self.conn.commit()

    def put_blocks(self, blocks: Sequence[dict]) -> None:
        rows = [(b["block_id"], b["page_id"], b["doc_id"], int(b["seq"]),
                 b["kind"], json.dumps([float(x) for x in b["bbox"]]),
                 b.get("text"), b.get("latex"), b.get("confidence"),
                 json.dumps(b.get("meta") or {}))
                for b in blocks]
        self.conn.executemany(
            """INSERT INTO blocks(block_id,page_id,doc_id,seq,kind,bbox,text,latex,confidence,meta)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(block_id) DO UPDATE SET seq=excluded.seq, kind=excluded.kind,
                 bbox=excluded.bbox, text=excluded.text, latex=excluded.latex,
                 confidence=excluded.confidence, meta=excluded.meta""", rows)
        self.conn.commit()

    def put_chunks(self, chunks: Iterable[Chunk]) -> None:
        rows = []
        for c in chunks:
            vec_blob = np.asarray(c.vec, dtype="float32").tobytes() if c.vec else None
            rows.append((c.chunk_id, c.doc_id, c.page_id,
                         json.dumps(c.block_ids), c.content, c.visual_desc,
                         vec_blob, int(c.vec_dim or 0), int(c.token_count or 0)))
        self.conn.executemany(
            """INSERT INTO chunks(chunk_id,doc_id,page_id,block_ids,content,visual_desc,vec,vec_dim,token_count)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(chunk_id) DO UPDATE SET content=excluded.content,
                 visual_desc=excluded.visual_desc, vec=excluded.vec,
                 vec_dim=excluded.vec_dim, block_ids=excluded.block_ids,
                 token_count=excluded.token_count""", rows)
        self.conn.commit()

    def put_kg(self, nodes: Sequence[dict], edges: Sequence[dict]) -> None:
        self.conn.executemany(
            """INSERT INTO kg_nodes(node_id,kind,label,aliases) VALUES(?,?,?,?)
               ON CONFLICT(node_id) DO UPDATE SET kind=excluded.kind,
                 label=excluded.label,
                 aliases=excluded.aliases""",
            [(n["node_id"], n["kind"], n["label"],
              json.dumps(n.get("aliases") or [])) for n in nodes])
        self.conn.executemany(
            """INSERT INTO kg_edges(src,dst,rel,weight,evidence) VALUES(?,?,?,?,?)
               ON CONFLICT(src,dst,rel) DO UPDATE SET weight=MAX(kg_edges.weight, excluded.weight),
                 evidence=excluded.evidence""",
            [(e["src"], e["dst"], e["rel"], float(e.get("weight", 1.0)),
              e.get("evidence")) for e in edges])
        self.conn.commit()

    # ------------------------------------------------------------------
    # reads - documents / pages / blocks
    # ------------------------------------------------------------------
    def list_documents(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT d.*, (SELECT COUNT(*) FROM pages p WHERE p.doc_id=d.doc_id) AS actual_pages,
                      (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.doc_id) AS n_chunks
               FROM documents d ORDER BY d.created_at DESC""").fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id=?",
                                (doc_id,)).fetchone()
        return dict(row) if row else None

    def find_document_by_source(self, source_path: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM documents WHERE source_path=?",
                                (str(source_path),)).fetchone()
        return dict(row) if row else None

    def get_pages(self, doc_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM pages WHERE doc_id=? ORDER BY page_no", (doc_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_page(self, page_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM pages WHERE page_id=?",
                                (page_id,)).fetchone()
        return dict(row) if row else None

    def get_blocks(self, *, doc_id: str | None = None,
                   page_id: str | None = None) -> list[dict]:
        if page_id:
            cur = self.conn.execute(
                "SELECT * FROM blocks WHERE page_id=? ORDER BY seq", (page_id,))
        else:
            cur = self.conn.execute(
                "SELECT * FROM blocks WHERE doc_id=? ORDER BY seq", (doc_id,))
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["bbox"] = json.loads(d["bbox"])
            d["meta"] = json.loads(d["meta"] or "{}")
            out.append(d)
        return out

    def blocks_by_ids(self, ids: Sequence[str]) -> dict[str, dict]:
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        cur = self.conn.execute(f"SELECT * FROM blocks WHERE block_id IN ({marks})",
                                list(ids))
        out = {}
        for r in cur.fetchall():
            d = dict(r)
            d["bbox"] = json.loads(d["bbox"])
            d["meta"] = json.loads(d["meta"] or "{}")
            out[d["block_id"]] = d
        return out

    # ------------------------------------------------------------------
    # reads - chunks
    # ------------------------------------------------------------------
    def count_chunks(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def iter_chunks(self, with_vec: bool = True,
                    doc_ids: Sequence[str] | None = None) -> Iterator[Chunk]:
        cols = "chunk_id,doc_id,page_id,block_ids,content,visual_desc,vec,vec_dim,token_count"
        if doc_ids:
            marks = ",".join("?" for _ in doc_ids)
            cur = self.conn.execute(
                f"SELECT {cols} FROM chunks WHERE doc_id IN ({marks})", list(doc_ids))
        else:
            cur = self.conn.execute(f"SELECT {cols} FROM chunks")
        for r in cur:
            vec = None
            if with_vec and r["vec"]:
                vec = np.frombuffer(r["vec"], dtype="float32").tolist()
            yield Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"],
                        page_id=r["page_id"],
                        block_ids=json.loads(r["block_ids"]),
                        content=r["content"], visual_desc=r["visual_desc"] or "",
                        vec=vec, vec_dim=r["vec_dim"] or 0,
                        token_count=r["token_count"] or 0)

    def chunks_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT chunk_id,doc_id,page_id,block_ids,content,visual_desc,token_count "
            f"FROM chunks WHERE chunk_id IN ({marks})", list(ids)).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        return [Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], page_id=r["page_id"],
                      block_ids=json.loads(r["block_ids"]), content=r["content"],
                      visual_desc=r["visual_desc"] or "",
                      token_count=r["token_count"] or 0)
                for cid in ids if (r := by_id.get(cid))]

    # ------------------------------------------------------------------
    # reads - FTS5 / BM25
    # ------------------------------------------------------------------
    @staticmethod
    def sanitize_fts_query(query: str) -> str:
        """Turn free text into a safe FTS5 MATCH expression (quoted terms)."""
        terms = [t for t in _tokens(query) if t]
        if not terms:
            return ""
        return " OR ".join(f'"{t}"' for t in terms[:24])

    def fts_search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        """Return [(chunk_id, bm25_score)] - *lower* (more negative) is better."""
        match = self.sanitize_fts_query(query)
        if not match:
            return []
        try:
            rows = self.conn.execute(
                """SELECT c.chunk_id, bm25(chunks_fts) AS s
                   FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
                   WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?""", (match, k)).fetchall()
        except sqlite3.OperationalError as exc:      # malformed MATCH - not fatal
            log.debug("fts failed (%s)", exc)
            return []
        return [(r["chunk_id"], float(r["s"])) for r in rows]

    # ------------------------------------------------------------------
    # reads - knowledge graph
    # ------------------------------------------------------------------
    def kg_nodes_search(self, query: str, limit: int = 12) -> list[dict]:
        """Seed the graph ranking from the query (label/alias match, ranked).

        No FTS table on ``kg_nodes`` (ADR-003 keeps the schema minimal) - a
        LIKE pass over a few thousand concept nodes is microseconds.
        """
        terms = [t.lower() for t in _tokens(query) if len(t) > 1]
        if not terms:
            return []
        rows = self.conn.execute("SELECT * FROM kg_nodes").fetchall()
        scored: list[tuple[float, dict]] = []
        for r in rows:
            d = dict(r)
            label = (d["label"] or "").lower()
            aliases = " ".join(json.loads(d["aliases"] or "[]")).lower()
            hay = f"{label} {aliases} {d['node_id'].lower()}"
            score = sum(2.0 for t in terms if t in label) + \
                    sum(1.0 for t in terms if t in aliases)
            if score:
                scored.append((score, d))
        scored.sort(key=lambda kv: -kv[0])
        return [d for _, d in scored[:limit]]

    def kg_neighbors(self, node_ids: Sequence[str], limit: int = 60) -> list[dict]:
        if not node_ids:
            return []
        marks = ",".join("?" for _ in node_ids)
        rows = self.conn.execute(
            f"SELECT * FROM kg_edges WHERE src IN ({marks}) OR dst IN ({marks}) "
            f"ORDER BY weight DESC LIMIT ?",
            [*node_ids, *node_ids, limit]).fetchall()
        return [dict(r) for r in rows]

    def kg_subgraph(self, node_ids: Sequence[str]) -> tuple[list[dict], list[dict]]:
        """1-hop neighbourhood as (nodes, edges)."""
        edges = self.kg_neighbors(node_ids)
        ids = set(node_ids)
        for e in edges:
            ids.add(e["src"])
            ids.add(e["dst"])
        if not ids:
            return [], []
        marks = ",".join("?" for _ in ids)
        nodes = self.conn.execute(
            f"SELECT * FROM kg_nodes WHERE node_id IN ({marks})", list(ids)).fetchall()
        return [dict(n) for n in nodes], edges

    def kg_stats(self) -> dict:
        n = self.conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
        e = self.conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        return {"nodes": int(n), "edges": int(e)}

    # ------------------------------------------------------------------
    # chat persistence
    # ------------------------------------------------------------------
    def new_session(self, session_id: str | None = None) -> str:
        sid = session_id or new_id("sess")
        self.conn.execute("INSERT OR IGNORE INTO chat_sessions(session_id) VALUES(?)",
                          (sid,))
        self.conn.commit()
        return sid

    def add_message(self, session_id: str, role: str, content: str,
                    citations: Sequence[Citation] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO chat_messages(session_id,role,content,citations) VALUES(?,?,?,?)",
            (session_id, role, content,
             json.dumps([c.to_dict() for c in citations]) if citations else None))
        self.conn.commit()

    def history(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role,content,citations,created_at FROM chat_messages "
            "WHERE session_id=? ORDER BY msg_id DESC LIMIT ?", (session_id, limit)).fetchall()
        out = []
        for r in reversed(rows):
            out.append({"role": r["role"], "content": r["content"],
                        "citations": json.loads(r["citations"] or "[]"),
                        "created_at": r["created_at"]})
        return out

    # ------------------------------------------------------------------
    # deletes
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str) -> None:
        con = self.conn
        con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM blocks WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        # graph rows carry {"doc_id": ...} in their evidence JSON (pipeline.py),
        # so one document can be removed without rebuilding repository-wide KG
        try:
            con.execute(
                "DELETE FROM kg_edges WHERE json_extract(evidence,'$.doc_id')=?",
                (doc_id,))
        except sqlite3.OperationalError:             # SQLite built without JSON1
            con.execute("DELETE FROM kg_edges WHERE evidence LIKE ? AND "
                        "evidence LIKE ?", ('%"doc_id"%', f'%{doc_id}%'))
        con.execute("""DELETE FROM kg_nodes WHERE NOT EXISTS(
                        SELECT 1 FROM kg_edges e WHERE e.src=kg_nodes.node_id
                        OR e.dst=kg_nodes.node_id)""")
        con.commit()

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def vacuum(self) -> None:
        self.conn.execute("VACUUM")


def _tokens(text: str) -> list[str]:
    """Alphanumeric terms only (FTS5 treats quotes/parens specially)."""
    out, cur = [], []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def ensure_schema(db_path: str | Path | None = None) -> Store:
    """Convenience used by the CLI / smoke test."""
    return Store(db_path)
