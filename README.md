# PageX — Offline Visual Document Intelligence

**Photo of handwritten notes → layout blocks → annotated LaTeX (single source of truth) → hybrid search + grounded RAG chat.**

Everything runs **locally and fully offline** on a 8 GB / i3-class laptop: no cloud APIs, no GPU, no telemetry. Built as a B.Tech final-year project following the *PageX Master Build Manual* (ADRs 001–008).

```
camera / scan ──► preprocess ──► VLM (1 call/page) ──► blocks ──► annotated .tex (SSOT)
                                        │                          │
                                        ▼                          ▼
                                   SQLite (blocks, chunks, FTS5, vectors, KG)
                                        │
                     search ◄───────────┼───────────► grounded RAG chat
                   (vector+BM25+KG, RRF fused)      (local LLM + citations)
```

---

## Highlights

| | |
|---|---|
| **ADR-001** | PyQt6 desktop shell (no QtWebEngine), CLI is a thin shell over the same core |
| **ADR-002** | llama-cpp-python + GGUF **Q4_K_M** only; one heavy model resident at a time (`ModelSlot`) |
| **ADR-003** | one SQLite file: documents / pages / blocks / chunks + FTS5 / KG / chat |
| **ADR-004** | hybrid retrieval = cosine (brute-force numpy) + BM25 + graph, fused with **RRF** |
| **ADR-005** | visual-first: VLM captions & crops, not ColPali |
| **ADR-006** | answers must be grounded — no context ⇒ explicit refusal, every answer cites page+blocks |
| **ADR-007** | Docling optional (never required) |
| **ADR-008** | one VLM call per page, per-crop fallback only when needed |

---

## Quick start

```bash
# 1. environment (Python 3.11+)
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# llama-cpp-python is built from source, CPU-only:
CMAKE_ARGS="-DGGML_NATIVE=ON" .venv/bin/pip install llama-cpp-python

# 2. models (~5 GB, resumable)
.venv/bin/python tools/download_models.py

# 3. probe this machine -> bench.json (threads, tier, tok/s, s/page)
.venv/bin/python tools/probe.py

# 4. sample corpus
.venv/bin/python -m pagex.testing.fixtures

# 5. ingest
.venv/bin/python cli.py ingest samples/notes/1_rotational_motion.png --handwritten

# 6. search / ask / chat / GUI
.venv/bin/python cli.py search "angular momentum conservation"
.venv/bin/python cli.py ask   "What is the moment of inertia?"
.venv/bin/python cli.py chat
.venv/bin/python cli.py gui
```

### Pre-demo ritual

```bash
.venv/bin/python -m pagex.testing.smoke     # 15 checks, < 1 s, no heavy models
.venv/bin/python tools/probe.py             # proves the hardware contract
.venv/bin/python -m pytest tests -q         # full suite
.venv/bin/python -m pagex.eval.retrieval    # Recall@5 / MRR gate (>= 0.8)
```

---

## Measured on the development machine

*CPU: 11th Gen Intel i3-1125G4 (4C/8T), 8.1 GB RAM, AVX2, no GPU — `bench.json`*

| metric | value | gate |
|---|---|---|
| LLM decode (granite-3.3-2b Q4_K_M) | **7.86 tok/s** @ 4 threads | ≥ 6 tok/s (tier 1) |
| VLM page cost (Qwen3-VL-2B Q4_K_M) | **~93 s/page** (8 s encode + ~10 s prefill + 0.096 s/token) | tier ladder |
| embedder (MiniLM-L6 Q4, 384-dim) | **469 texts/s** | ≥ 30 |
| p95 hybrid search (300-chunk corpus) | **21 ms** | < 3000 ms |
| peak RSS with VLM resident | **3.05 GB** (release → 0.15 GB) | < 6.2 GB |
| test suite | **124 passed, 0 failed** (41 s) | — |
| smoke | **15/15 passed** (0.5 s) | — |

The VLM was benchmarked against Qwen2.5-VL-3B (41–51 s just to encode one page → ~1900 s/page overall) before switching to **Qwen3-VL-2B** (8–17 s encode → ~90 s/page, 1.3 GB less RAM). See `docs/decisions.md`.

---

## Repository layout

```
cli.py                     all commands (list/ingest/convert-notes/search/ask/chat/
                           graph/bench/models/watch/smoke/gui)
pagex/core/                config, models (ModelSlot), preprocess, parse, blocks,
                           latex_export, store (SQLite DDL), retrieve, rag,
                           pipeline (ingest), watch
pagex/gui/                 workers, library_view, viewer, editor, chat_pane,
                           settings_dialog, main_window, app (PyQt6)
pagex/testing/             fixtures (synthetic pages + queries), smoke
pagex/eval/retrieval.py    Recall@5 / MRR harness
tools/                     download_models.py, probe.py (writes bench.json)
tests/                     unit (TC-UNIT-01..09), integration (TC-INT-01..06,
                           TC-GUI-01..05), perf (TC-PERF-01..03)
samples/                   synthetic handwritten + digital pages, queries.json
docs/                      architecture, decisions, manual, report_outline
pagex.toml                 runtime configuration (edited by the GUI settings dialog)
bench.json                 hardware probe output (tier, threads, budgets)
```

---

## Configuration

`pagex.toml` — paths, model files, runtime (threads, contexts, `kv_q8`), quality
(`top_k`, `chunk_tokens`, RRF *k*). `tools/probe.py` picks `n_threads` and the
**tier**; tier 2/3 shrink the VLM page to 768 px and (only when the chat model
itself measures < 6 tok/s) swap to the 1.5 B fallback.

The GUI **Settings** dialog edits the same file; changes apply to the next model
acquire (the `ModelSlot` evicts whatever is resident).

---

## Testing

```bash
.venv/bin/python -m pytest tests -q              # everything (124 tests)
.venv/bin/python -m pytest tests -q -m "not slow"  # skip real-GGUF tests
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/integration/test_gui.py -q
```

Unit tests never touch a GGUF (canned vision/LLM/embedder fixtures);
`slow` marks load real models; `gui` runs headless with `QT_QPA_PLATFORM=offscreen`.

## License / academic use
Built for academic submission; models are Apache-2.0 / MIT licensed (see model
cards for Qwen3-VL-2B, granite-3.3-2b, MiniLM-L6).
