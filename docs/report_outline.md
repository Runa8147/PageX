# PageX — Report Outline (B.Tech final year)

Suggested chapter structure with the evidence to paste from this repository.
Every number below is reproducible with the command in parentheses.

---

## 1. Abstract
Offline visual document intelligence: photo of handwritten notes → structured
layout blocks → **annotated LaTeX as single source of truth** → hybrid search +
grounded RAG. Targets an 8 GB, no-GPU, fully offline laptop.

## 2. Problem statement & scope
* Handwritten study material is unsearchable and un-editable as images.
* Existing pipelines are cloud-based (Mathpix) or GPU-hungry (ColPali, large OCR
  stacks); neither works offline on modest hardware.
* Scope: single-user, local-first, semester-notes scale corpus (≤ 10⁵ chunks).

## 3. Requirements
Functional: ingest (image/PDF), layout+transcription, LaTeX export in 6 formats,
hybrid search, grounded multi-turn chat, GUI + CLI, watch folder.
Non-functional: **offline**, **≤ 6.2 GB RAM worst case**, p95 search < 3 s,
Recall@5 ≥ 0.8, deterministic test suite.

## 4. Architecture (see `docs/architecture.md`)
Draw the write-path / read-path diagram; explain the three derived states
(chunks, embeddings, LaTeX) and the single source of truth (blocks table).

## 5. Design decisions (see `docs/decisions.md`)
ADR-001..008 in a table: context / decision / consequence. Highlight:
* why GGUF Q4_K_M + llama-cpp instead of Ollama or transformers,
* why one `ModelSlot` (RAM contract),
* why RRF fusion instead of weighted score mixing,
* why one VLM call per page (ADR-008) with measured cost model.

## 6. Experiments & results (the core chapter)

### 6.1 Hardware probe (`bench.json`, `python tools/probe.py`)
| metric | value |
|---|---|
| CPU / RAM | i3-1125G4, 8.1 GB, AVX2, no GPU |
| chosen threads / tier | 4 / tier 2 |
| LLM decode | 7.86 tok/s (granite-3.3-2b Q4_K_M) |
| embedder | 469 texts/s, dim 384 |
| VLM per page | ~93 s (8 s encode + 10 s prefill + 0.096 s/token) |

### 6.2 VLM model selection (paste table from `docs/decisions.md`)
Qwen2.5-VL-3B (40–51 s encode, 4.83 GB) vs **Qwen3-VL-2B (8–17 s, 3.56 GB)** —
2.5–4× faster encode, ~21× faster end-to-end per page (1904 s → 88 s measured),
1.3 GB RAM saved. Include the resolution-sweep lesson (tier forces 768 px).

### 6.3 Thread sweep (`bench["llm_runs"]`)
1 → 6.77, 2 → 7.19, 4 → **7.86**, 8 → 5.53 tok/s: hyper-threading *hurts* this
workload; the probe now sweeps and records the winner.

### 6.4 Retrieval evaluation (`python -m pagex.eval.retrieval`)
Recall@5 / MRR / p95 latency over `samples/queries.json` (12 queries, 7 docs).
Gate: **Recall@5 ≥ 0.8**. *(paste the table the command prints)*

### 6.5 Performance budgets (`pytest tests/perf -q -s`)
| gate | measured |
|---|---|
| p95 search < 3 s | **21 ms** (median 13 ms, 300-chunk corpus) |
| peak RSS ≤ 6.2 GB | **3.05 GB** → release → 0.15 GB |
| embed ≥ 30 texts/s | **469 texts/s** |

### 6.6 Test suite
`python -m pytest tests -q` → **124 passed** (unit 96, integration 21, gui 5,
perf 4); `python -m pagex.testing.smoke` → 15/15 in 0.5 s.

## 7. GUI walkthrough (screenshots)
1. Library + viewer with coloured block overlay,
2. Editor correcting a VLM transcription (save → re-embed → LaTeX regenerates),
3. Search results → click → viewer jumps to page/blocks,
4. Chat with sources, click a source → viewer highlight,
5. Settings showing the probe numbers + `ModelSlot` live status bar.

## 8. Limitations
* VLM latency (~93 s/page) is CPU-bound; multi-page documents ingest
  sequentially — mitigated by tier degradation and per-page caching.
* `pdflatex` absent → PDF export uses the ReportLab renderer (guaranteed path).
* KG edges after a block edit keep their evidence chunk ids (grouping is
  preserved by design; re-ingest re-derives the graph).
* Single-user local DB (no concurrent writers beyond WAL).

## 9. Future work
Speculative decode / GGUF distillation for faster VLM decode, a `n_batch`
prefill tuning sweep, incremental re-chunking when an edited block crosses the
400-token budget, PDF text-layer extraction for born-digital input, and an
optional ONNX/GPU backend behind the same `ModelSlot` interface.

## 10. References
llama.cpp & llama-cpp-python, Qwen3-VL, granite-3.3, MiniLM, FTS5/BM25,
Reciprocal Rank Fusion (Cormack et al.), RAG (Lewis et al.), OpenCV, PyQt6.

---

### Appendix
* A: exact SQLite DDL (`pagex/core/store.py`)
* B: prompts (`PAGE_PROMPT`, `LAYOUT_PROMPT`, `KG_PROMPT`, RAG template)
* C: annotated-LaTeX marker format + round-trip proof (TC-UNIT-06)
* D: reproduce-everything command list (`README.md` → Pre-demo ritual)
