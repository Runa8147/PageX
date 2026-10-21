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

Works the same on Linux and Windows (Windows: replace `.venv/bin/python`
with `.venv\Scripts\python`, or just `pagex` after a `pip install -e .`).

```bash
# 1. environment (Python 3.11+)
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # compiler-free: test/build tools
.venv/bin/pip install -e ".[llm]"      # llama-cpp-python (the `llm` extra)
#    the `llm` extra compiles llama.cpp:
#      Linux:   sudo apt install build-essential cmake
#      Windows: VS Build Tools + CMake, then
#               .venv\Scripts\pip install llama-cpp-python
#      CPU-only optimised build: CMAKE_ARGS="-DGGML_NATIVE=ON" pip install ...

# 2. models (~5 GB, resumable, SHA-256 verified - never bundled, ADR-002)
.venv/bin/pagex models download        # offline: pagex models install-pack <zip>

# 3. does this machine work? -> paths, deps, model + config status
.venv/bin/pagex doctor

# 4. hardware probe -> bench.json (threads, tier, tok/s, s/page)
.venv/bin/pagex bench

# 5. sample corpus + demo ingest + ready-made queries
.venv/bin/pagex demo

# 6. search / ask / chat / GUI
.venv/bin/pagex search "angular momentum conservation"
.venv/bin/pagex ask   "What is the moment of inertia?"
.venv/bin/pagex chat
.venv/bin/pagex gui            # or: pagex-gui (windowed, no console on Windows)
```

`python cli.py …`, `python -m pagex …` and the installed `pagex` script are
the same implementation (`pagex/cli.py`, tested in
`tests/integration/test_cli.py`).

### Desktop workspace

`pagex gui` opens a quiet, dark-first workspace: a **Library** of all indexed
documents, a **Reader** with page rail, live block overlays and a compact
editor, and a single **Ask PageX** surface for grounded answers or passage
search.  Imports stream page-by-page, so the first page is visible long before
a large scan finishes.  The design follows the "sidebar + prompt area" pattern:
navigation and recent documents live on the left, the focused work happens on
one canvas, and technical IDs never leak into the primary UI.

### Export layouts (the VLM contract)

Because PageX uses a VLM to preserve structure - not plain OCR - exporting asks
which contract you want:

| mode | result |
|---|---|
| **Organized / optimized** | reflowed, typeset PDF (plus DOCX, HTML, Markdown, LaTeX); best for sharing and reading |
| **Same layout as input** | original page raster + invisible VLM text layer; every position, column, equation and handwritten mark stays exactly where it was, while the PDF remains selectable and searchable |

From the GUI choose **Export** in the Reader; from the CLI use
`pagex convert-notes notes.pdf --formats pdf --pdf-layout faithful` (or
`--pdf-layout organized`).

### Pre-demo ritual

```bash
.venv/bin/pagex smoke           # 15 checks, < 1 s, no heavy models
.venv/bin/pagex bench           # proves the hardware contract
.venv/bin/python -m pytest tests -q -m "not slow"   # suite without real-GGUF tests
.venv/bin/python -m pagex.eval.retrieval            # Recall@5 / MRR gate (>= 0.8)
```

### Windows in one paragraph

`pip install -e ".[dev,llm]"` under *x64 Native Tools* Python (or the plain
wheel-free `.[dev]` for tests only), `pagex models download`, `pagex gui`.
All state resolves through `pagex.core.paths` — no `__file__`-relative paths,
no hardcoded `/` — so the same commands work from a checkout, an installed
package, `PAGEX_HOME` portable mode or the frozen build. Installer and
portable artifacts: see [docs/packaging.md](docs/packaging.md).

---

## Distribution (one engine, three faces)

Five entry ramps are documented in [docs/packaging.md](docs/packaging.md)
(command index: [packaging/README.md](packaging/README.md)):

| tier | artifact | command |
|---|---|---|
| 0 | checkout | `pip install -e ".[dev,llm]"` |
| 1 | frozen binaries | `./packaging/build_linux.sh`, `packaging\build_windows.ps1` → `dist/pagex-linux-x64.tar.gz`, `dist\PageX-Setup-x64.exe` |
| 2 | offline demo pack (USB, air-gapped) | `python packaging/make_demo_pack.py --app dist/pagex-gui` |
| 3 | headless image | `docker build -t pagex:cpu .` |
| 4 | PyPI wheel + GitHub Releases | `python -m build`; releases automated on `v*` tags (`.github/workflows/ci.yml`) |

CI runs the test suite on **ubuntu + windows × Python 3.11/3.12/3.13** on every
push (`.github/workflows/ci.yml`), then builds the release artifacts for `v*`
tags. **No artifact ever contains GGUF weights** (ADR-002) — the specs assert
it, `.dockerignore` enforces it, and licenses live in [NOTICE](NOTICE)
(code: [LICENSE](LICENSE), MIT).

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
| test suite | **164 passed, 0 failed** (`-m "not slow"`, 37 s) | — |
| smoke | **15/15 passed** (0.5 s) | — |

The VLM was benchmarked against Qwen2.5-VL-3B (41–51 s just to encode one page → ~1900 s/page overall) before switching to **Qwen3-VL-2B** (8–17 s encode → ~90 s/page, 1.3 GB less RAM). See `docs/decisions.md`.

---

## Repository layout

```
cli.py                     thin shim -> pagex.cli (PyInstaller entry for the
                           console build; `python cli.py …` == `pagex …`)
pagex/cli.py               every command: list/ingest/convert-notes/search/ask/
                           chat/graph/bench/models/demo/doctor/watch/smoke/gui
pagex/core/                config, paths (PAGEX_HOME/platformdirs ladder),
                           model_manifest (+ models_manifest.json), models
                           (ModelSlot), preprocess, parse, blocks, latex_export,
                           store (SQLite DDL), retrieve, rag, pipeline, watch
pagex/gui/                 workers, library_view, viewer, editor, chat_pane,
                           settings_dialog, models_wizard, main_window, app
pagex/probe.py, pagex/demo.py   hardware probe (-> bench.json), demo seeder
pagex/testing/             fixtures (synthetic pages + queries), smoke
pagex/eval/retrieval.py    Recall@5 / MRR harness
tools/                     thin shims (download_models.py, probe.py)
packaging/                 build_linux.sh, build_windows.ps1, make_demo_pack.py,
                           windows/pagex.iss, README.md
pagex-gui.spec, pagex-cli.spec   PyInstaller onedir builds (no GGUFs, ADR-002)
.github/workflows/ci.yml   ubuntu + windows matrix; release artifacts on v* tags
Dockerfile                 headless CLI image `pagex:cpu` (PAGEX_HOME=/data)
tests/                     unit (TC-UNIT-*), integration (TC-INT-*, incl. the
                           real `Artificial Intelligence handwitten notes.pdf`
                           scan when present), gui, perf
samples/                   synthetic handwritten + digital pages, queries.json
docs/                      architecture, decisions, manual, packaging,
                           report_outline
pagex.toml                 runtime configuration (edited by the GUI settings dialog)
bench.json                 hardware probe output (tier, threads, budgets)
NOTICE, LICENSE            third-party attribution (models = Apache-2.0/MIT)
```

---

## Configuration

`pagex.toml` — paths, model files, runtime (threads, contexts, `kv_q8`), quality
(`top_k`, `chunk_tokens`, RRF *k*). `pagex bench` (same code as
`tools/probe.py`) picks `n_threads` and the
**tier**; tier 2/3 shrink the VLM page to 768 px and (only when the chat model
itself measures < 6 tok/s) swap to the 1.5 B fallback.

Relative paths in `pagex.toml` resolve against the ladder in
`pagex/core/paths.py`: `PAGEX_HOME` (portable) > source checkout > frozen
build > platformdirs user dirs — so the same config works on Windows and
Linux. `pagex doctor` prints the active mode.

The GUI **Settings** dialog edits the same file; changes apply to the next model
acquire (the `ModelSlot` evicts whatever is resident).

---

## Testing

```bash
.venv/bin/python -m pytest tests -q -m "not slow"  # CI matrix command (164 tests)
.venv/bin/python -m pytest tests -q                # + real-GGUF tests (models present)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/integration/test_gui.py -q
```

Unit tests never touch a GGUF (canned vision/LLM/embedder fixtures); `slow`
marks load real models; `gui` runs headless with `QT_QPA_PLATFORM=offscreen`.

**Real-scan coverage:** `tests/integration/test_pdf_notes.py` exercises the
full pipeline on `Artificial Intelligence handwitten notes.pdf` (126-page
no-text-layer scan) when the file sits in the repository root — it is ~22 MB
and deliberately *not* committed, so the tests skip cleanly on CI and on
machines without it. `--max-pages` keeps the local run cheap
(`pagex ingest <pdf> --max-pages 5`).

## License / academic use
Built for academic submission; code is MIT ([LICENSE](LICENSE)) and models are
Apache-2.0 / MIT licensed (see [NOTICE](NOTICE) and the model cards for
Qwen3-VL-2B, granite-3.3-2b, MiniLM-L6, bge-small-en-v1.5).
