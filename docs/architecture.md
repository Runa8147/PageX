# PageX — Architecture

## 1. The pipeline at a glance

```
                    WRITE PATH (ingest)
┌──────────┐   ┌────────────┐   ┌──────────────────┐   ┌───────────────┐
│ image/   │──►│ preprocess │──►│ VLM  (1 call /   │──►│ parse_page_   │
│ PDF      │   │ deskew,    │   │ page, per-crop   │   │ json → blocks │
│          │   │ CLAHE,     │   │ fallback)        │   │ (kind, bbox,  │
│          │   │ normalize  │   │ Qwen3-VL-2B Q4   │   │  text, latex) │
└──────────┘   └────────────┘   └──────────────────┘   └──────┬────────┘
                                                              │
      ┌───────────────────────────────────────────────────────┘
      ▼
 annotated .tex (SSOT, %PX markers) ──► chunks (≤400 tok, math never split)
      │                                       │
      │                                       ▼
      │                              embeddings (MiniLM-L6, 384-d)
      │                              + FTS5 row  + KG (LLM or deterministic)
      ▼                                       │
 data/artifacts/<doc>.tex                    ▼
                                   ┌──────────────────────────┐
                                   │  SQLite  data/pagex.db   │
                                   └────────────┬─────────────┘
                    READ PATH                   │
      ┌─────────────────────────────────────────┘
      ▼
 search:  cosine (numpy brute force) ─┐
          BM25 (FTS5) ────────────────┼─► RRF fusion ─► citations (page+blocks+bbox)
          KG neighbourhood ───────────┘
      │
      ▼
 RAG:  citations → context → local LLM (granite-3.3-2b Q4) → grounded answer
       (no context ⇒ explicit refusal, ADR-006)
```

## 2. Module responsibilities

| module | responsibility |
|---|---|
| `core/config.py` | loads `pagex.toml`; derived values (`effective_threads`, `runtime_vlm_max_side`, `active_llm`); `save()` used by the settings dialog |
| `core/models.py` | **`ModelSlot`** — exactly one heavy GGUF resident; `embed_texts`, `generate`, `vision_generate`; KV cache q8_0 + flash attention |
| `core/preprocess.py` | deskew (coarse-to-fine projection-profile search ±12°), CLAHE, upscale, normalization to 0..1000 boxes, crop helper |
| `core/parse.py` | `PAGE_PROMPT`/`LAYOUT_PROMPT`, robust JSON extraction, bbox rescaling, NMS, reading-order sort |
| `core/blocks.py` | `Block/PageModel/DocumentModel`, **annotated-LaTeX codec** (`%PX:block=…` markers ⇄ blocks, integer-bbox round-trip) |
| `core/latex_export.py` | `document_to_tex`, exports pdf (pdflatex→ReportLab fallback), docx, html, md, json |
| `core/store.py` | SQLite DDL (9 tables + FTS5 + 3 triggers), **one connection per thread** (never `check_same_thread=False`), KG, chat persistence, delete cascade incl. KG evidence pruning |
| `core/retrieve.py` | vector ranking (numpy cosine), BM25 via FTS5, KG ranking, **RRF fusion**, `Citation` objects |
| `core/rag.py` | context assembly, prompt building, history trimming, refusal policy, `answer_query`, citation persistence |
| `core/pipeline.py` | `ingest()` orchestrator (progress callbacks, injectable model fns), chunking, KG build, `save_block_edit()` for the GUI editor |
| `core/watch.py` | debounced watch-folder state (drain) used by CLI + GUI timer |
| `gui/*` | PyQt6 shell; **the UI thread never infers** — every model call is a `FnWorker` on `QThreadPool` |
| `testing/fixtures.py` | deterministic synthetic pages + `queries.json` (no network) |
| `eval/retrieval.py` | Recall@5 / MRR harness, gate ≥ 0.8 |

## 3. SQLite schema (ADR-003, exact DDL in `store.py`)

```
documents(doc_id, title, source_path, kind, n_pages, status, error_msg, created_at, meta)
pages(page_id, doc_id, page_no, image_path, width, height, meta)          UNIQUE(doc_id,page_no)
blocks(block_id, page_id, doc_id, seq, kind, bbox, text, latex, confidence, meta)
chunks(chunk_id, doc_id, page_id, block_ids, content, visual_desc, vec, vec_dim, token_count)
chunks_fts  (FTS5 external content)  + triggers chunks_ai / chunks_au / chunks_ad
kg_nodes(node_id, kind, label, aliases)
kg_edges(src, dst, rel, weight, evidence)   -- evidence = {"doc_id":…, "chunk_ids":[…]}
chat_sessions(session_id, created_at)
chat_messages(msg_id, session_id, role, content, citations, created_at)
```

* Documents/chunks/graph are **append-only per ingest** → document A stays
  searchable while document B ingests (TC-INT-03).
* `delete_document` removes only that document's rows **and** the KG edges whose
  `evidence.doc_id` matches (then prunes orphan nodes) — repository-wide KG is
  never rebuilt.
* Vectors are `float32` blobs; cosine similarity is a single `numpy` matmul
  (corpus ≤ ~10⁵ chunks → FAISS would be pure overhead, ADR-004).

## 4. Threading & RAM contract

```
GUI thread ── signals ──► QThreadPool workers ──► core functions
   │                                                     │
   │  QTimer (1.5 s): HEAVY.state() + RSS                ▼
   └──────────────────────────────────────► Store (thread-local SQLite conn)
```

* `ModelSlot` lock-protected: `acquire("vlm")` evicts `llm` and vice-versa.
* Measured: VLM resident 3.05 GB peak, release → 0.15 GB (TC-UNIT-03, TC-PERF-02).
* Budget contract **worst case < 6.2 GB** on an 8 GB machine; bench probes peak
  RSS for all three models and writes it to `bench.json`.

## 5. Degradation ladder (tiers)

`tools/probe.py` sweeps threads (1/2/4/8), measures LLM tok/s on the **primary**
model, then measures VLM s/page as `overhead + tokens × decode_rate`
(an unbiased two-run estimator).

| tier | condition | effect |
|---|---|---|
| 1 | LLM ≥ 6 tok/s **and** VLM ≤ 60 s/page | defaults from `pagex.toml` |
| 2 | VLM ≤ 180 s/page and LLM ≥ 3 tok/s | VLM page → 768 px, `vlm_max_tokens` 2048 |
| 3 | otherwise | 768 px page, largest contexts shrink |

Chat model swap (granite-2b → qwen2.5-1.5b) happens **only when the chat model
itself** measures < 6 tok/s — a slow VLM must not degrade answer quality.

## 6. SSOT discipline

`blocks` rows are the source of truth; everything else is derived state:

```
blocks ──► annotated .tex (markers)      [blocks_to_annotated_tex]
blocks ──► chunks + embeddings           [make_chunks, save_block_edit]
blocks ──► clean .tex / pdf / docx / html / md / json
```

Editing a block in the GUI (`save_block_edit`) writes the row, re-embeds the
affected chunks (grouping preserved, so KG evidence ids stay valid) and the
window regenerates the LaTeX tab from the store — same derivation order as
ingest.
