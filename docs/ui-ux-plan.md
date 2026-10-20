# PageX UI/UX + Import Performance Plan

Status: **proposal for approval** (branch `feature/ui-ux-overhaul`)
Scope: desktop app shell, import flow, search/ask/chat, reader, onboarding,
and the ingestion path that makes "uploading" feel slow.
Not in scope: changing the model stack, retrieval algorithm, or CLI contract.

---

## 1. Where we are

The cross-platform branch is merged into `master` and the new work branch is
`feature/ui-ux-overhaul`. The test suite is green (164 passed, 1 slow test
deselected).

The current GUI is functional but still reads like an engineering demo:

| area | what a user sees today | why it hurts |
|---|---|---|
| shell | one window, five tabs (`Viewer`, `Editor`, `Search results`, `LaTeX (SSOT)`, `Chat`) | no single "home"; the user has to know which tab to open |
| entry points | a permanent search bar **and** a separate chat tab with its own ask box | two competing ways to ask; results and answers live in different places |
| library | cramped left dock with a six-column table, clipped columns, raw doc IDs | titles are hard to scan; status/ID noise; no thumbnails |
| empty state | a giant blank canvas saying "no document selected" | first-run user has no idea what to do next |
| import | single-file picker, no drag & drop; no queue, ETA, or cancel | feels slow and out of control even though work is off-thread |
| progress | status-bar text (`preprocess`, `understand`, `graph`) + indeterminate bar | user cannot tell whether it is working or stuck |
| reader | one page at a time, prev/next only, overlay legend, separate Editor tab | page navigation and block editing require tab switching |
| chat | plain-text transcript, sources in a second splitter, raw `session: sess_…` | answers are hard to read and citations feel detached |
| status bar | `rss 0.11 GB · tier 2 · 4 thread(s)`, `db: pagex.db · 16 chunks` | developer diagnostics in the user's face |
| visual system | mostly light theme but dark menu bar/dock title; emoji theme toggle | looks unfinished and inconsistent |

The result: the product has strong internals but no clear primary flow.
A first-time user should be able to **import → see the page → ask a question**
without learning the architecture.

---

## 2. Product direction

Five principles drive every change:

1. **Import is instant; understanding is background.**
   Adding a file should immediately create a row and show its pages. The AI
   transcription continues behind a visible, cancellable job. Never make the
   user wait for inference just to see that their file was accepted.
2. **One place to read, one place to ask.**
   Reader and Ask are destinations, not tabs. Search is a mode of Ask, not a
   separate feature with its own input box.
3. **Progressive results.**
   Show the page as soon as it is rasterised, blocks as soon as a page is
   transcribed, search hits after the first page, and concept-graph enrichment
   after the document is already usable.
4. **Plain language first; diagnostics on demand.**
   "Indexing page 4 of 12 · about 6 min left", not `understand 4/12`.
   RSS, tier, thread counts and model paths move to Settings → Diagnostics.
5. **Desktop-native, keyboard-first.**
   Drag & drop, multi-select, `Ctrl+O` import, `Ctrl+K` search/ask palette,
   arrow keys in the reader, `Esc` to close panels, visible focus states.

---

## 3. Target information architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  PageX      [ Search or ask…                    Ctrl+K ]   [☾] [⚙]  │
├──────────────┬───────────────────────────────────────────────────────┤
│  + Import    │                                                       │
│              │   HOME / LIBRARY                                      │
│  Library     │   ┌──────────────┐  ┌──────────────┐                  │
│  Ask         │   │ thumbnail    │  │ thumbnail    │   recent docs    │
│  Settings    │   │ title        │  │ title        │   + status       │
│              │   │ 12 pages     │  │ 1 page       │                  │
│  Recent      │   └──────────────┘  └──────────────┘                  │
│  · notes.pdf │                                                       │
│  · calc.png  │   [ drop files here to import ]                       │
└──────────────┴───────────────────────────────────────────────────────┘
```

### Screens

**Home / Library**
- Empty state: large drop zone, "Add your first document", "Try the demo".
- Populated: recent documents as thumbnail cards or a clean list (thumbnail,
  title, page count, status badge, last opened). Filter box. Sort by recent /
  title / status. Multi-select + context menu (Open, Re-index, Reveal file,
  Delete). No raw IDs.
- A single primary **+ Import** button in the sidebar, plus drag & drop
  anywhere on the window.

**Import queue** (drawer or inline panel)
- One row per file with thumbnail, title, "page 3 / 12", ETA, and Cancel.
- Queue is sequential (one heavy model), but files can be added while running.
- Completed rows collapse into a summary toast: "3 documents ready · open".
- Failed rows show a plain-language reason and a Retry action.

**Reader**
```
[← Library]  Rotational motion · 12 pages · indexing 4/12     [Ask this doc]
┌──────────┬────────────────────────────────────┬──────────────────────┐
│ p1 ▣     │                                    │ Block inspector      │
│ p2 ▣     │          page canvas               │ kind · bbox          │
│ p3 ▣     │          (fit width / zoom)        │ text / LaTeX         │
│ …        │                                    │ [Save block]         │
└──────────┴────────────────────────────────────┴──────────────────────┘
```
- Thumbnail rail on the left, canvas in the middle, collapsible inspector on
  the right. "Source" toggle shows annotated LaTeX for the document.
- Fit width/page, zoom percentage, rotate, show/hide overlays, in-document
  search.
- Clicking a block scrolls its text into the inspector; editing persists and
  re-embeds exactly as today.
- "Ask this document" opens Ask with a document filter already applied.
- While indexing, pages appear as they are transcribed and the canvas is
  usable immediately.

**Ask**
```
Ask PageX                                      [All documents ▾]
┌──────────────────────────────────────────┬──────────────────────────┐
│  conversation                            │ Sources                  │
│  You: How does torque relate to …        │ [p.3] Rotational motion  │
│  PageX: … [1] [2]                        │ 0.92 · vector + bm25     │
│                                          │ [p.7] …                  │
├──────────────────────────────────────────┴──────────────────────────┤
│ [Answer | Find passages]  Ask about your notes…              [Ask]  │
└─────────────────────────────────────────────────────────────────────┘
```
- **Answer** mode = current grounded RAG chat.
- **Find passages** mode = current hybrid search, rendered as result cards
  with highlighted snippets, page thumbnails, ranker badges and a one-click
  jump into Reader.
- Inline citation chips `[1]` jump to the cited page; source cards show the
  snippet and can be opened in Reader.
- Chat history sidebar with human titles ("Torque and angular acceleration"),
  New chat, rename, delete. No session IDs.
- Starter prompts on an empty conversation, scoped to the selected document.
- Stop generation, regenerate, copy answer. Markdown/code formatting; equations
  shown as styled monospace when full math rendering is not available.
- Global `Ctrl+K` palette offers "Search library", "Ask PageX", "Open doc".

---

## 4. Why import feels slow, and the fix

This is not a file-copy problem. Adding a 126-page scan currently triggers a
long AI pipeline, and the app only reports coarse stage names while it runs.

### Current path (per page)

```
load all PDF pages up front  ->  preprocess  ->  VLM (~95 s)  ->  blocks
      -> embed  ->  LLM knowledge-graph call  ->  next page
```

Three compounding problems:

1. **Every page rasterises the whole document first.**
   `load_pages()` materialises every PDF page at 200 dpi before page 1 is
   processed. A 126-page scan is roughly 1.2 GB of numpy arrays held for the
   whole run; time-to-first-page is ~7 s and memory is dangerously close to
   the 8 GB budget once the VLM is resident.
2. **The VLM and LLM are mutually exclusive, but the pipeline alternates them
   on every page.** The KG pass acquires the LLM (evicting the VLM) and the
   next page reloads the VLM. That is pure model-swap overhead on every page,
   plus the KG generation time itself, all before the document is usable.
3. **Nothing is visible or controllable while it happens.**
   No page appears, no ETA, no cancel, no partial search. If the user imports
   a 126-page PDF by mistake, the only escape is closing the app.

### Target path

```
import metadata + first thumbnail (instant)
  └─ Pass 1: stream one page at a time -> preprocess -> VLM -> blocks -> chunks
             (document becomes searchable after the first page)
  └─ mark READY
  └─ Pass 2 (background enrichment): load LLM once -> concept graph for all pages
             (search/chat remain available; graph ranker improves when done)
```

### Concrete changes

| # | change | expected effect |
|---|---|---|
| 1 | `iter_pages()` generator: one open PyMuPDF handle, rasterise one page at a time | time-to-first-page ~7 s → <1 s; 126-page RAM ~1.2 GB → ~10 MB |
| 2 | Split `ingest()` into import metadata + `index_document()` + `enrich_graph()` | document row and page thumbnails appear immediately |
| 3 | Defer LLM KG to a second pass; keep deterministic graph in pass 1 | removes N model swaps and N KG calls from the critical path; document usable much earlier |
| 4 | Single heavy-model owner: indexing pauses when the user asks a question, then resumes | no VLM/LLM thrashing; answers never fight indexing |
| 5 | Real job queue in the GUI (one sequential worker) with cancel between pages | no accidental double imports; cancel responds in <1 s |
| 6 | Per-page progress + moving-average ETA + stage labels | honest feedback: "page 4 of 12 · about 6 min left" |
| 7 | Skip unchanged files by content hash + mtime; never delete the old index until the replacement succeeds | re-import is instant; a failed re-index no longer destroys a working document |
| 8 | Persist job state; on startup mark interrupted docs "paused" and offer Resume | closing the app no longer loses hours of work |
| 9 | Low-risk ingest modes: `Fast` (no KG, lower VLM side), `Balanced`, `Best` | user controls the time/quality trade-off; useful for demos |
| 10 | Digital-PDF fast path when a text layer exists (later phase) | minutes → seconds for born-digital PDFs |

The biggest user-visible win is **1 + 2 + 3**: the document and its pages appear
immediately, search becomes available after page 1, and the expensive KG work
no longer blocks "ready". Streaming also removes a real memory risk.

---

## 5. Work plan

### Phase 1 — Import flow + ingest speed (P0, first)

- `iter_pages()` in `pagex/core/preprocess.py`; update `pipeline.ingest` and
  `convert_to_latex` to consume it.
- Split indexing into passes; add `cancel`/`on_page` callbacks and per-stage
  timings.
- Defer LLM KG; add `documents.meta`/progress fields via a small schema
  migration; statuses: `queued`, `importing`, `indexing`, `ready`,
  `enriching`, `paused`, `error`.
- GUI `ImportQueue` with sequential worker, cancel, ETA, retry.
- Drag & drop + multi-select file dialog + folder import.
- Instant document rows and live page refresh for the open document.
- Settings/import options for Fast/Balanced/Best.

**Done when:** dropping a 126-page PDF shows it in the library in under a
second; page 1 appears in the reader without waiting for the VLM; cancel stops
within one page; memory stays bounded; the document is searchable while the
concept graph is still enriching.

### Phase 2 — App shell and navigation (P0)

- Replace the `QTabWidget` with a sidebar + `QStackedWidget`
  (`Home`, `Reader`, `Ask`, `Settings`). Keep compatibility aliases
  (`win.viewer`, `win.chat`, `win.tabs`) so existing tests and CLI entry points
  keep working during the transition.
- Add Home/Library screen with empty state, recent docs, filter and import
  drop zone.
- Reader gets a page thumbnail rail and a collapsible block inspector; the
  Editor tab and the top-level LaTeX tab disappear into the Reader.
- Top bar: breadcrumb, `Ctrl+K` palette, theme toggle, settings.
- Status bar becomes a single human-readable line; diagnostics move to
  Settings.

**Done when:** every core action is reachable in at most two clicks from Home,
and there is no duplicate search/ask input.

### Phase 3 — Unified search / ask / chat (P0–P1)

- Build the Ask screen with `Answer` / `Find passages` modes.
- Result cards with snippet highlighting, page thumbnails, ranker badges,
  document/kind filters, and keyboard navigation.
- Chat: markdown rendering, inline citation chips, source panel with
  thumbnails, history list, rename/delete, New chat, stop/regenerate/copy.
- Streaming tokens from the local LLM with a stage indicator
  ("Retrieving → Reading 5 passages → Writing") and a Stop button.
- Document-scoped ask ("Ask this document") and multi-document scope selector.

**Done when:** a user never has to choose between search and chat to answer a
question, and the first token appears while the answer is still being written.

### Phase 4 — Reader and library polish (P1)

- Thumbnail rail, fit width/page, zoom %, rotate, overlay toggle, in-doc search.
- Library cards/list with status badges, sort/filter, context menu, multi-select.
- Export from the Reader context bar (keep the existing Export menu).
- Better failure states: retry failed pages, reveal log file, clear error text.

### Phase 5 — Onboarding, settings, accessibility (P1)

- First-run welcome: model readiness card with Download missing models; demo
  corpus button; import drop zone.
- Settings reorganized: General, Appearance, Indexing, Models, Diagnostics.
  Plain labels, "Advanced" collapsed. Theme choice persists (already partly
  implemented).
- Keyboard shortcuts, focus rings, accessible names/tooltips, high-contrast
  dark theme.
- Replace emoji/technical labels with consistent icons and short text.

### Phase 6 — Hardening and proof (P1)

- Update TC-GUI tests for the new shell; add tests for import queue, cancel,
  progressive pages, theme persistence and search/ask modes.
- Add a perf test that asserts streaming page loading keeps memory bounded.
- Capture before/after timings in `docs/` and update the README screenshots/
  demo script.
- Manual demo checklist: fresh install → models missing → download → import
  sample → ask → citation → export.

---

## 6. Success criteria

| metric | today | target |
|---|---|---|
| document visible after import | after ingest starts/completes | < 1 s |
| first page visible in reader | after VLM finishes | < 2 s |
| first page searchable | after whole doc + KG | after page 1 |
| cancel response | none / close app | < 1 s between pages |
| 126-page PDF extra memory | ~1.2 GB of raw arrays | bounded (~one page) |
| duplicate/accidental imports | no queue, no guard | queue + cancel + retry |
| time to find an answer | switch tabs, read raw list | one Ask screen, 2 clicks max |
| UI terms | `LaTeX (SSOT)`, `sessions: sess_…`, `rss/tier` | plain language + diagnostics pane |
| test suite | 164 green | green + new GUI/import tests |

Exact ingestion speedup will be measured after Phase 1; the structural wins
(no per-page model swap, no KG on the critical path, no whole-document
rasterisation) are guaranteed, while the VLM's per-page cost remains the
hardware limit.

---

## 7. Risks and guardrails

- **Do not break grounding.** Citation/page/block plumbing stays unchanged.
- **Do not change the CLI contract.** `pagex ingest/search/ask/chat` keep
  working; new callbacks/options are additive with defaults.
- **Model memory.** One heavy model at a time (ADR-002) is preserved; the
  queue exists to enforce it.
- **Schema changes.** Additive migration only (`documents.meta`, progress
  fields); old databases keep working.
- **Test compatibility.** Keep `win.tabs`, `win.viewer`, `win.chat` aliases
  until the GUI tests are migrated.
- **Scope discipline.** No new model, no cloud dependency, no rewrite of
  retrieval, no design for mobile.

---

## 8. Suggested build order

1. **Sprint A (P0):** Phase 1 — import queue, streaming pages, deferred KG,
   cancel/ETA, drag & drop. This fixes the "upload slow" complaint and the
   worst reliability risk.
2. **Sprint B (P0):** Phase 2 — shell/navigation + Home/Library.
3. **Sprint C (P0/P1):** Phase 3 — unified Ask + streaming answers.
4. **Sprint D (P1):** Phase 4 + 5 — reader/library polish, onboarding,
   settings.
5. **Sprint E:** Phase 6 — tests, benchmarks, demo docs.

If you want a visible demo improvement fastest, Sprint A + the Home/Reader
shell from Sprint B are the highest-value slice. If you want the "ask anything"
experience to feel finished, Sprint C is the one to prioritize next.
