# PageX — Architecture Decision Records

Each record: context → decision → consequences, with the measurements that
justified it on the development machine (i3-1125G4, 8.1 GB, no GPU).

---

## ADR-001 — PyQt6 desktop shell, CLI shares the core

* **Context:** the demo must work on a lecturer's laptop with no browser tooling
  and no network; QtWebEngine would add ~300 MB and a second event loop.
* **Decision:** PyQt6 widgets (no `QtWebEngine`), `cli.py` as a thin shell over
  `pagex.core`. Both drive the identical `ingest / search / answer_query` code.
* **Consequences:** one dependency tree, `QT_QPA_PLATFORM=offscreen` gives
  headless GUI tests (TC-GUI-01..05 pass in CI-like conditions). UI thread never
  infers: everything runs on `QThreadPool` (`gui/workers.py`).

## ADR-002 — llama-cpp-python + GGUF Q4_K_M, one heavy model resident

* **Context:** Ollama/transformers would duplicate memory management and break
  the 8 GB contract; FP16 3B models do not fit alongside the OS.
* **Decision:** llama-cpp-python 0.3.36 built with `-DGGML_NATIVE=ON`; every
  model is GGUF **Q4_K_M or lighter**; a process-wide `ModelSlot` guarantees the
  VLM and chat LLM are never resident together.
* **Measured:**
  * granite-3.3-2b Q4: **7.86 tok/s** (thread sweep: 1→6.77, 2→7.19, 4→7.86, 8→5.53)
  * Qwen3-VL-2B Q4: **~93 s/page**, 3.05 GB RSS while resident → release → 0.15 GB
  * embedder MiniLM-L6 Q4: **469 texts/s**, 384-dim
* **Consequences:** worst case ≈ 6.2 GB contract holds with 3+ GB of headroom;
  `ModelSlot.state()` is shown live in the GUI status bar as demo proof.

### Engineering notes (llama-cpp-python quirks we had to solve)

1. quantised KV (`type_k/type_v`) needs **integer** `llama_cpp.GGML_TYPE_Q8_0`
   *and* `flash_attn=True` — llama.cpp rejects a quantised V cache otherwise.
   Measured faster than f16 KV (6.19 vs 5.29 tok/s) and halves KV RAM.
2. multimodal needs `chat_handler=Qwen25VLChatHandler(clip_model_path=…)`
   (there is no `mmproj_path` kwarg); it also works for Qwen3-VL GGUFs.
3. embed output may be per-token or pooled — normalise by mean-pooling when
   `len(vecs) > 1`.

## ADR-003 — one SQLite file, FTS5 + vectors + KG together

* **Decision:** exact DDL in `store.py` (9 tables, FTS5 external-content with
  3 triggers). One connection **per thread** via `threading.local()` — never
  `check_same_thread=False`.
* **Consequences:** WAL mode keeps GUI-worker writes and UI reads race-free;
  a single `data/pagex.db` is trivially portable for the demo.

## ADR-004 — hybrid retrieval, brute-force cosine, RRF fusion

* **Context:** FAISS adds a native dependency and build fragility; our corpus
  (semester notes) is ≤ 10⁵ chunks.
* **Decision:** numpy cosine over a `float32` matrix + FTS5 BM25 + KG
  neighbourhood, fused with Reciprocal Rank Fusion (`rrf_k = 60`).
* **Measured:** p95 search **21 ms** over a 300-chunk corpus with the real
  embedder (gate: < 3 s) — three orders of margin.

## ADR-005 — visual-first (VLM captions), not ColPali

* **Decision:** one VLM call returns caption + regions (bbox, kind, content);
  visual context is preserved as `visual_desc` on every chunk.
* **Consequences:** retrieval can ground answers to *boxes on the page*, and the
  exported LaTeX keeps diagram/table structure instead of flattening it.

## ADR-006 — grounded answers or an explicit refusal

* **Decision:** `rag.build_prompt` injects only retrieved chunks; if nothing is
  retrieved the model is told to say it cannot answer from the notes. Every
  answer is checked for citation labels (`grounded` flag) and both turns are
  persisted to `chat_messages` with the citation JSON.

## ADR-007 — Docling optional

* **Decision:** never imported by the core path; `fitz`/OpenCV cover raster and
  PDF needs. Smoke test does not require it.

## ADR-008 — one VLM call per page, per-crop fallback

* **Decision:** `understand_page()` tries the full-page prompt once; the
  layout/crop path (`parse_with_crops`) only runs when the page JSON is empty or
  malformed.
* **Measured:** full page = 8 s encode + 10 s prefill + 0.096 s/token decode →
  a 520-token page JSON costs **~88 s**; a second call would double it, hence the
  fallback is exception-only.

---

## Empirical decision: VLM swap (Qwen2.5-VL-3B → Qwen3-VL-2B)

Resolution sweep (first attempt) showed *no* gain from downscaling because tier 2
forces a 768 px page regardless of `vlm_max_side` — the real cost was the ViT
encode itself:

| model | clip encode | image prefill | total (8 tok) | total (128 tok) | RSS |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B @ 4 threads | 40.7 s | ~23 s | 66.5 s | 96.6 s | 4.83 GB |
| Qwen2.5-VL-3B @ 8 threads | 51.2 s | ~23 s | 79.9 s | 104.2 s | 4.79 GB |
| **Qwen3-VL-2B @ 4 threads** | **8.2–16.7 s** | **9.4–11.2 s** | **28.8 s** | **32.1 s** | **3.56 GB** |

Full-page JSON (521 tokens, real handwritten sample): **87.8 s** and valid JSON
with 9 regions — versus **~1904 s/page** for Qwen2.5-VL-3B in the original
`bench.json`. Decision: `vlm_gguf = Qwen3-VL-2B-Instruct-Q4_K_M` (kept Qwen2.5-VL
as `vlm_alt` in the manifest for A/B).

## Empirical decision: probe fixes

1. `bench_llm` now measures the **primary** LLM, not the tier-swapped fallback
   (the old loop was circular: slow → fallback → slower reading).
2. VLM `seconds_per_page_est = overhead(8-token run) + tokens × decode_rate`
   instead of `tokens / (total including prefill)` — the old estimator reported
   187.6 s/page for a workload that really costs 88–93 s (within 6 % now).
3. Thread sweep picks `n_threads` per machine; this i3 prefers 4 for the LLM
   (8 threads *hurts*: 5.53 vs 7.86 tok/s).

## Deskew (projection-profile search)

minAreaRect and PCA-over-pixel-coordinates both failed on low-contrast pencil
scans. The shipped implementation searches ±12° (coarse 1.0°, then fine 0.25°)
maximising horizontal projection concentration and **self-verifies** (keeps the
original if the score worsens). Covered by TC-UNIT-01.
