# PageX — Build Manual (implementation map)

How the Master Build Manual's work packages map onto this repository, with the
exact commands and test IDs that prove each one. All commands assume the venv:
`.venv/bin/python`.

## WP0 — environment & models

| step | command | status |
|---|---|---|
| deps | `pip install -e ".[dev]"` (compiler-free) + `pip install -e ".[llm]"` for inference (`CMAKE_ARGS="-DGGML_NATIVE=ON"` CPU build); `requirements.txt` is the legacy equivalent | done (llama-cpp-python = `llm` extra) |
| models | `pagex models list\|download\|verify\|install-pack\|pack` (`pagex/models_manifest.json`: repo, size, SHA-256, license) — shims: `python tools/download_models.py [--only vlm llm]` | 8 files in `models/` (~7 GB), 4 required |
| probe | `pagex bench [--fast\|--full]` → `bench.json` — shim: `python tools/probe.py` | threads=4, tier=2, LLM 7.86 tok/s, VLM ~93 s/page |
| sanity | `pagex doctor` (paths/deps/models), `pagex demo` (sample corpus), `pagex smoke` | cross-platform: `pagex/core/paths.py` ladder (PAGEX_HOME > checkout > frozen > platformdirs), Windows-safe subprocess/console |

## WP1 — preprocessing (`preprocess.py`) — TC-UNIT-01, TC-UNIT-02

deskew (projection-profile ±12°, self-verifying) · CLAHE · upscale ·
0..1000 normalization · crop helper. Pure functions, no models.

## WP2 — models & RAM (`models.py`) — TC-UNIT-03

`ModelSlot` (one resident heavy model, evict-on-acquire), KV q8_0 + flash
attention, `embed_texts` / `generate` / `vision_generate` with injectable
fns. Test asserts RSS drops ≥ 1 GB on `release()` and < 2 GB after.

## WP3 — parse (`parse.py`) — TC-UNIT-04

JSON extraction (fences, prose, truncated JSON), bbox rescale, NMS, reading
order, `PAGE_PROMPT` / `LAYOUT_PROMPT`, crop fallback.

## WP4 — store (`store.py`) — TC-UNIT-05, TC-UNIT-08

Exact DDL (9 tables + FTS5 + 3 triggers), one connection per thread, chunking
budgets, KG tables, chat persistence.

## WP5 — LaTeX codec (`blocks.py`) — TC-UNIT-06

`%PX` markers, escaping, round-trip identity `blocks → tex → blocks`
(integer-bbox rounding documented).

## WP6 — exporters (`latex_export.py`) — TC-UNIT-07

annotated/clean tex, pdf (pdflatex → **ReportLab fallback** — pdflatex is absent
on this machine), docx, html, md, json.

## WP7 — retrieval (`retrieve.py`) — TC-UNIT-09, TC-INT-04

vector (numpy cosine) + BM25 + graph, RRF fusion, `Citation` grounding
(page + block ids + bboxes), `recall_at_k` / `reciprocal_rank` helpers.

## WP8 — pipeline (`pipeline.py`) — TC-INT-01..03, TC-PERF-03

`ingest()` with progress callbacks and injectable models, chunking (math never
split), deterministic + LLM KG, annotated `.tex` artifact, incremental writes,
targeted delete, `save_block_edit` (GUI edit path), `convert_to_latex`.

```bash
python cli.py ingest samples/notes/1_rotational_motion.png --handwritten
python cli.py convert-notes samples/notes/2_entropy.png --formats tex,pdf,docx
```

## WP9 — GUI (`gui/`) — TC-GUI-01..05

library dock · page viewer with block overlay · block editor (persist +
re-embed + LaTeX refresh) · search tab · LaTeX (SSOT) tab · grounded chat with
clickable sources · settings dialog (bench summary + knobs + model paths) ·
status bar proving `ModelSlot` live · watch-folder timer · export menu.

```bash
python cli.py gui            # or: python -m pagex.gui.app
QT_QPA_PLATFORM=offscreen python -m pytest tests/integration/test_gui.py -q
```

## WP10 — read path & RAG (`rag.py`) — TC-INT-05

`answer_query`: retrieve → context → prompt → local LLM → grounded answer with
citations; refusal without context; multi-turn history trimmed; persisted.

```bash
python cli.py ask "What is the moment of inertia?"
python cli.py chat
```

## WP11 — eval (`eval/retrieval.py`) — gate Recall@5 ≥ 0.8

```bash
python -m pagex.eval.retrieval            # table + Recall@5/MRR/latency
python -m pagex.eval.retrieval --json
```

## WP12 — demo hygiene

```bash
pagex smoke                          # 15/15, no heavy models
pagex bench                          # refresh bench.json before a demo
python -m pagex.testing.fixtures     # regenerates samples/ deterministically
QT_QPA_PLATFORM=offscreen python -m pytest tests -q -m "not slow"   # 172 tests
```

## WP13 — distribution & cross-platform (this branch)

path ladder + `PAGEX_HOME` (`pagex/core/paths.py`), model manifest with
download/verify/install-pack (`pagex/core/model_manifest.py`), real CLI in
`pagex/cli.py` (`doctor`/`demo`/`models`/`--max-pages`), GUI model wizard,
PyInstaller specs + Inno Setup + Dockerfile + GitHub Actions matrix.
Rationale and commands: `docs/packaging.md`, `packaging/README.md`.

```bash
pagex doctor                         # where state lives, on this OS
pagex models list                    # 8 weights, SHA-256 + license, never bundled
./packaging/build_linux.sh           # Windows: packaging\build_windows.ps1
```

## Test ID index

| suite | IDs | file |
|---|---|---|
| unit | TC-UNIT-01/02 | `tests/unit/test_preprocess.py` |
| unit | TC-UNIT-03 | `tests/unit/test_models.py` |
| unit | TC-UNIT-04 | `tests/unit/test_parse.py` |
| unit | TC-UNIT-05/08 | `tests/unit/test_store.py` |
| unit | TC-UNIT-06 | `tests/unit/test_blocks.py` |
| unit | TC-UNIT-07 | `tests/unit/test_latex.py` |
| unit | TC-UNIT-09 | `tests/unit/test_retrieve.py` |
| integration | TC-INT-01/02/03 | `tests/integration/test_ingest.py` |
| integration | TC-INT-04 | `tests/integration/test_search.py` |
| integration | TC-INT-05 | `tests/integration/test_chat.py` |
| integration | TC-INT-06 | `tests/integration/test_cli.py` |
| integration | distribution surface (doctor/demo/models/shims/max-pages) | `tests/integration/test_cli.py` |
| integration | real 126-page scan (skips if the PDF is absent) | `tests/integration/test_pdf_notes.py` |
| unit | path ladder / portable mode | `tests/unit/test_paths.py` |
| unit | model manifest, verify, download, install-pack | `tests/unit/test_model_manifest.py` |
| integration (gui) | TC-GUI-01..05 | `tests/integration/test_gui.py` |
| perf | TC-PERF-01/02/03 | `tests/perf/test_budgets.py` |

## Acceptance gates (from the manual)

| gate | threshold | measured |
|---|---|---|
| Recall@5 | ≥ 0.8 | see `docs/report_outline.md` (eval run on samples) |
| p95 search | < 3 s | **21 ms** |
| peak RSS | ≤ 6.2 GB | **3.05 GB** (release → 0.15 GB) |
| LLM | ≥ 6 tok/s for tier 1 | **7.86 tok/s** |
| VLM | functional on tier 3 | **~93 s/page** (was 1904 s/page) |
| suite | green | **172 passed, 0 failed** (`-m "not slow"`) |
