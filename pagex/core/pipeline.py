"""WP6 - the ingestion orchestrator: file -> indexed, searchable document.

    load -> preprocess -> VLM page JSON -> blocks -> annotated .tex (SSOT)
         -> chunks (+embeddings) -> knowledge graph -> SQLite

Design rules that make the demo survive an 8 GB laptop:

* **Incremental**: a run only writes rows for *this* document.  Nothing ever
  rebuilds the index, so document A stays searchable while document B is
  ingesting (TC-INT-03).
* **Progress callbacks** everywhere - the GUI worker and the CLI progress bar
  are the same code path.
* **Model functions are injectable**, so the whole pipeline is unit-testable
  with canned VLM output and no GGUF on disk.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .blocks import (BLOCK_KINDS, Block, DocumentModel, PageModel,
                     block_id_for, blocks_to_annotated_tex, make_document,
                     make_page, page_id_for, summary)
from .config import CFG
from .latex_export import document_to_tex, export_document
from .parse import (JSON_ONLY_SUFFIX, LAYOUT_PROMPT, PAGE_PROMPT, ParsedPage,
                    extract_json, fill_page, parse_page_json, parse_with_crops)
from .preprocess import (load_pages, preprocess_page, save_page, thumbnail)
from .store import Chunk, Store, new_id

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]
# vision_fn(image, prompt) -> GenResult | str      (injectable for tests)
VisionFn = Callable[[object, str], object]
GenerateFn = Callable[..., object]
EmbedFn = Callable[[Sequence[str]], list[list[float]]]

KG_PROMPT = """Extract the key concepts and their relations from this page text.
Return STRICT JSON only:
{"concepts":["<short concept>", ...],
 "edges":[["<concept A>","PRECEDES|MENTIONS|CO_OCCURS|HAS_FORMULA|PART_OF","<concept B>"], ...]}
At most 6 concepts and 8 edges. No prose, no markdown."""

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}


# ---------------------------------------------------------------------------
# chunking (ADR-004 / spec WP6)
# ---------------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    """~4 chars per token - good enough for chunk budgets (documented)."""
    return max(1, (len(text or "") + 3) // 4)


def block_text(block: Block) -> str:
    return block.latex if block.kind in ("math", "table", "diagram") else block.text


def chunk_blocks(blocks: Sequence[Block], page_caption: str, page_no: int,
                 max_tokens: int | None = None) -> list[list[Block]]:
    """Group consecutive blocks into <= chunk_tokens groups.

    Math / table / diagram blocks are **never merged** with anything else:
    splitting an equation across two chunks destroys both retrieval and the
    citation grounding.
    """
    max_tokens = max_tokens or CFG.chunk_tokens
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = approx_tokens(page_caption)
    for block in blocks:
        text = block_text(block)
        tokens = approx_tokens(text)
        if block.kind in ("math", "table", "diagram"):
            if current:
                groups.append(current)
                current, current_tokens = [], approx_tokens(page_caption)
            if tokens <= max_tokens:
                groups.append([block])
            else:                              # oversized equation: own chunk
                groups.append([block])
            continue
        if current and current_tokens + tokens > max_tokens:
            groups.append(current)
            current, current_tokens = [], approx_tokens(page_caption)
        current.append(block)
        current_tokens += tokens
    if current:
        groups.append(current)
    return groups


def visual_description(page_caption: str, blocks: Sequence[Block]) -> str:
    """Page caption + kind histogram - the "visual" half of ADR-005."""
    hist: dict[str, int] = {}
    for b in blocks:
        hist[b.kind] = hist.get(b.kind, 0) + 1
    parts = []
    if page_caption:
        parts.append(page_caption)
    if hist:
        parts.append("page contains " + ", ".join(
            f"{n} {k}" if n == 1 else f"{n} {k}s" for k, n in sorted(hist.items())))
    return ". ".join(parts).strip()


def make_chunks(page: PageModel, doc_id: str, block_id_to_chunk: dict[str, str],
                max_tokens: int | None = None) -> list[Chunk]:
    """Build the retrieval units for one page (no vectors yet)."""
    groups = chunk_blocks(page.blocks, page.caption, page.page_no, max_tokens)
    out: list[Chunk] = []
    for i, group in enumerate(groups):
        content_parts = [b.content for b in group if b.content.strip()]
        if page.caption and i == 0:
            content_parts.insert(0, page.caption)
        content = "\n\n".join(content_parts).strip()
        if not content:
            continue
        chunk_id = f"{page.page_id}-c{i:03d}"
        for b in group:
            block_id_to_chunk[b.block_id] = chunk_id
        out.append(Chunk(
            chunk_id=chunk_id, doc_id=doc_id, page_id=page.page_id,
            block_ids=[b.block_id for b in group], content=content,
            visual_desc=visual_description(page.caption, group),
            token_count=approx_tokens(content)))
    return out


# ---------------------------------------------------------------------------
# knowledge graph
# ---------------------------------------------------------------------------
def slug(text: str, limit: int = 64) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:limit].rstrip("-")) or "x"


def node_id(kind: str, label: str) -> str:
    return f"{kind}:{slug(label)}"


def deterministic_graph(page: PageModel, doc: DocumentModel,
                        block_to_chunk: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Structural nodes/edges - no model call, always correct."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    doc_node = {"node_id": f"document:{doc.doc_id}", "kind": "document",
                "label": doc.title, "aliases": []}
    nodes[doc_node["node_id"]] = doc_node

    prev: Block | None = None
    for block in page.blocks:
        bn = {"node_id": f"block:{block.block_id}", "kind": "block",
              "label": (block.display or block.kind)[:80], "aliases": [block.kind]}
        nodes[bn["node_id"]] = bn
        evidence = _evidence_for_block(doc, block, block_to_chunk)
        if prev is not None:
            edges.append({"src": f"block:{prev.block_id}", "dst": bn["node_id"],
                          "rel": "PRECEDES", "weight": 1.0, "evidence": evidence})
        if block.kind == "math" and block.latex:
            fn = {"node_id": node_id("formula", block.latex[:64]), "kind": "formula",
                  "label": block.latex.strip()[:120], "aliases": []}
            nodes[fn["node_id"]] = fn
            edges.append({"src": bn["node_id"], "dst": fn["node_id"],
                          "rel": "HAS_FORMULA", "weight": 1.5, "evidence": evidence})
        prev = block
    # page caption -> document
    cid = _evidence_for_page(page, block_to_chunk)
    if page.caption:
        cn = {"node_id": node_id("concept", page.caption[:64]), "kind": "concept",
              "label": page.caption[:120], "aliases": []}
        nodes.setdefault(cn["node_id"], cn)
        edges.append({"src": doc_node["node_id"], "dst": cn["node_id"],
                      "rel": "MENTIONS", "weight": 1.0, "evidence": cid})
    return list(nodes.values()), edges


def _evidence_json(doc_id: str, chunk_ids: Sequence[str]) -> str:
    """Edge evidence carries both: which doc owns it and which chunks prove it.

    ``doc_id`` lets ``Store.delete_document`` prune graph rows for a single
    document without ever rebuilding the repository-wide graph (ADR-003);
    ``chunk_ids`` is what the graph ranker turns back into citations.
    """
    return json.dumps({"doc_id": doc_id, "chunk_ids": sorted(set(chunk_ids))})


def _evidence_for_block(doc: DocumentModel, block: Block,
                        block_to_chunk: dict[str, str]) -> str:
    ids = [block_to_chunk[block.block_id]] if block.block_id in block_to_chunk else []
    return _evidence_json(doc.doc_id, ids)


def _evidence_for_page(page: PageModel, block_to_chunk: dict[str, str]) -> str:
    ids = [block_to_chunk[b.block_id] for b in page.blocks
           if b.block_id in block_to_chunk]
    return _evidence_json(page.doc_id, ids)


def llm_graph(page: PageModel, doc: DocumentModel, page_text: str,
              block_to_chunk: dict[str, str], generate_fn: GenerateFn,
              ) -> tuple[list[dict], list[dict]]:
    """One LLM call per page -> concept nodes + semantic edges (best effort)."""
    if not page_text.strip():
        return [], []
    try:
        res = generate_fn(KG_PROMPT + "\n\nPAGE TEXT:\n" + page_text[:3000],
                          max_tokens=512, temperature=0.0)
        raw = getattr(res, "text", res)
        data = extract_json(str(raw))
    except Exception as exc:
        log.debug("KG call failed on page %s: %s", page.page_no, exc)
        return [], []
    if not isinstance(data, dict):
        return [], []
    concepts = [str(c).strip() for c in (data.get("concepts") or []) if str(c).strip()]
    concepts = concepts[:6]
    evidence = _evidence_for_page(page, block_to_chunk)

    nodes = [{"node_id": node_id("concept", c), "kind": "concept",
              "label": c[:120], "aliases": []} for c in concepts]
    doc_node = {"node_id": f"document:{doc.doc_id}", "kind": "document",
                "label": doc.title, "aliases": []}
    nodes.append(doc_node)
    edges: list[dict] = []
    for c in concepts:
        edges.append({"src": doc_node["node_id"], "dst": node_id("concept", c),
                      "rel": "MENTIONS", "weight": 1.0, "evidence": evidence})
    # concept co-occurrence inside the page (capped at 10, per spec)
    pairs = 0
    raw_edges = data.get("edges") or []
    for item in raw_edges:
        if pairs >= 10:
            break
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        a, rel, b = (str(item[0]).strip(), str(item[1]).strip().upper(),
                     str(item[2]).strip())
        if not a or not b:
            continue
        if rel not in ("PRECEDES", "MENTIONS", "CO_OCCURS", "HAS_FORMULA", "PART_OF"):
            rel = "CO_OCCURS"
        for label in (a, b):
            if not any(n["node_id"] == node_id("concept", label) for n in nodes):
                nodes.append({"node_id": node_id("concept", label), "kind": "concept",
                              "label": label[:120], "aliases": []})
        edges.append({"src": node_id("concept", a), "dst": node_id("concept", b),
                      "rel": rel, "weight": 1.0, "evidence": evidence})
        pairs += 1
    return nodes, edges


# ---------------------------------------------------------------------------
# VLM page understanding with the fallback ladder
# ---------------------------------------------------------------------------
def _vt_text(result) -> str:
    return getattr(result, "text", result) if result is not None else ""


def understand_page(image, *, width: int, height: int,
                    vision_fn: VisionFn, crops: bool = True) -> ParsedPage:
    """Page image -> :class:`ParsedPage` (JSON -> retry -> per-crop fallback)."""
    warnings: list[str] = []
    raw = _vt_text(vision_fn(image, PAGE_PROMPT))
    for attempt, prompt_suffix in enumerate(("", JSON_ONLY_SUFFIX)):
        try:
            parsed = parse_page_json(
                raw if attempt == 0 else _vt_text(vision_fn(image, PAGE_PROMPT + prompt_suffix)),
                width, height, nms_iou=CFG.nms_iou, min_box_frac=CFG.min_box_frac)
            if parsed.regions or not parsed.caption:
                return parsed
            warnings.append("empty regions")
        except Exception as exc:
            warnings.append(f"page json attempt {attempt}: {exc}")
    if not crops:
        raise ValueError("; ".join(warnings) or "unparseable VLM output")
    # fallback: layout-only prompt, then one call per crop
    log.warning("falling back to per-crop transcription (%s)", warnings[-1:])
    layout_raw = _vt_text(vision_fn(image, LAYOUT_PROMPT))

    def crop_fn(crop, prompt):
        return _vt_text(vision_fn(crop, prompt))

    parsed = parse_with_crops(image, layout_raw, width, height, crop_fn,
                              nms_iou=CFG.nms_iou, min_box_frac=CFG.min_box_frac)
    parsed.warnings = warnings + parsed.warnings
    if not parsed.regions and not parsed.caption:
        raise ValueError("VLM produced no usable content: " + "; ".join(warnings))
    return parsed


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
@dataclass
class IngestReport:
    doc_id: str
    title: str
    path: str
    kind: str
    n_pages: int = 0
    n_blocks: int = 0
    n_chunks: int = 0
    n_kg_nodes: int = 0
    n_kg_edges: int = 0
    seconds: float = 0.0
    tex_path: str = ""
    errors: list[str] = field(default_factory=list)
    pages_done: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors and self.n_blocks > 0

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("doc_id", "title", "path", "kind", "n_pages", "n_blocks",
              "n_chunks", "n_kg_nodes", "n_kg_edges", "seconds", "tex_path")}
        d["errors"] = list(self.errors)
        return d


def infer_kind(path: Path, handwritten: bool = False) -> str:
    if handwritten:
        return "handwritten"
    if path.suffix.lower() in PDF_SUFFIXES:
        return "pdf"
    return "image"


def ingest(path: str | Path, *, title: str | None = None,
           handwritten: bool = False, store: Store | None = None,
           progress: ProgressFn | None = None,
           vision_fn: VisionFn | None = None, generate_fn: GenerateFn | None = None,
           embed_fn: EmbedFn | None = None, use_kg: bool = True,
           replace: bool = True) -> IngestReport:
    """Full write-path: *path* -> indexed document (spec WP1-WP7)."""
    from .models import embed_texts, generate, vision_generate
    vision_fn = vision_fn or vision_generate
    generate_fn = generate_fn or generate
    embed_fn = embed_fn or embed_texts

    t0 = time.perf_counter()
    path = Path(path).resolve()
    own_store = store is None
    store = store or Store()
    report = IngestReport(doc_id="", title=title or path.stem, path=str(path),
                          kind=infer_kind(path, handwritten))
    if not path.exists():
        raise FileNotFoundError(path)

    def notify(stage: str, cur: int, total: int) -> None:
        if progress:
            try:
                progress(stage, cur, total)
            except Exception:                       # pragma: no cover
                log.debug("progress callback raised", exc_info=True)

    # --- incremental: re-ingest only replaces THIS document's rows --------
    existing = store.find_document_by_source(str(path))
    doc_id = existing["doc_id"] if existing else new_id("doc")
    if existing and replace:
        store.delete_document(doc_id)
        log.info("re-ingesting %s (%s rows cleared)", path.name, doc_id)
    doc = make_document(doc_id, report.title, str(path), report.kind)
    store.put_document(doc_id, doc.title, str(path), report.kind,
                       status="indexing")

    try:
        notify("loading", 0, 1)
        pages_img = load_pages(path)
    except Exception as exc:
        store.set_status(doc_id, "error", str(exc))
        report.errors.append(f"load: {exc}")
        if own_store:
            store.close_all()
        raise

    report.n_pages = len(pages_img)
    store.set_status(doc_id, "indexing", None, n_pages=len(pages_img))
    art_dir = CFG.artifacts_dir / doc_id
    art_dir.mkdir(parents=True, exist_ok=True)
    block_to_chunk: dict[str, str] = {}
    all_chunks: list[Chunk] = []
    seq = 0

    for page_no, raw_img in enumerate(pages_img, start=1):
        notify("preprocess", page_no, len(pages_img))
        pre = preprocess_page(raw_img)
        page = make_page(doc, page_no, width=pre["width"], height=pre["height"],
                         caption="")
        img_path = save_page(pre["image"], art_dir / f"page_{page_no:04d}.png")
        thumb = thumbnail(pre["image"])
        save_page(thumb, art_dir / f"page_{page_no:04d}_thumb.png")
        page.image_path = str(img_path.relative_to(CFG.artifacts_dir)) \
            if img_path.is_relative_to(CFG.artifacts_dir) else str(img_path)
        store.put_page(page.page_id, doc_id, page_no, page.image_path,
                       pre["width"], pre["height"],
                       {"angle": pre["angle"], "scale": pre["scale"],
                        "contrast": round(pre["contrast"], 2),
                        "thumb": f"{art_dir.name}/page_{page_no:04d}_thumb.png"})

        notify("understand", page_no, len(pages_img))
        try:
            parsed = understand_page(pre["image"], width=pre["width"],
                                     height=pre["height"], vision_fn=vision_fn)
        except Exception as exc:
            report.errors.append(f"page {page_no}: {exc}")
            log.error("page %d failed: %s", page_no, exc)
            continue
        blocks = fill_page(page, parsed, doc_id, seq_offset=seq)
        seq += len(blocks)
        store.put_page(page.page_id, doc_id, page_no, page.image_path,
                       pre["width"], pre["height"],
                       {"angle": pre["angle"], "scale": pre["scale"],
                        "contrast": round(pre["contrast"], 2),
                        "thumb": f"{art_dir.name}/page_{page_no:04d}_thumb.png"})
        store.put_blocks([b.to_row() for b in blocks])
        report.n_blocks += len(blocks)

        # --- chunk + embed this page (incremental writes) ----------------
        notify("chunk", page_no, len(pages_img))
        chunks = make_chunks(page, doc_id, block_to_chunk)
        if chunks:
            try:
                vectors = embed_fn([c.content for c in chunks])
                for c, v in zip(chunks, vectors):
                    c.vec = v
                    c.vec_dim = len(v)
            except Exception as exc:
                report.errors.append(f"embed page {page_no}: {exc}")
                log.warning("embedding failed: %s", exc)
            store.put_chunks(chunks)
            all_chunks.extend(chunks)
        report.n_chunks += len(chunks)

        # --- knowledge graph --------------------------------------------
        if use_kg:
            notify("graph", page_no, len(pages_img))
            page_text = "\n".join(b.content for b in page.blocks)
            nodes, edges = deterministic_graph(page, doc, block_to_chunk)
            llm_nodes, llm_edges = llm_graph(page, doc, page_text,
                                             block_to_chunk, generate_fn)
            nodes = _merge_nodes(nodes, llm_nodes)
            edges = _merge_edges(edges, llm_edges)
            store.put_kg(nodes, edges)
            report.n_kg_nodes += len(nodes)
            report.n_kg_edges += len(edges)
        report.pages_done += 1

    # --- annotated LaTeX artifact (the SSOT) ------------------------------
    notify("export", report.pages_done, max(1, report.pages_done))
    try:
        tex = blocks_to_annotated_tex(doc)
        tex_path = art_dir / f"{doc_id}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        report.tex_path = str(tex_path)
        clean_path = art_dir / f"{doc_id}_clean.tex"
        clean_path.write_text(document_to_tex(doc, annotated=False),
                              encoding="utf-8")
    except Exception as exc:                        # pragma: no cover
        report.errors.append(f"tex: {exc}")
        log.exception("tex export failed")

    status = "ready" if (report.n_blocks and report.pages_done) else "error"
    err = "; ".join(report.errors) if report.errors and status == "error" else None
    store.set_status(doc_id, status, err, n_pages=report.n_pages)
    doc.status = status

    report.doc_id = doc_id
    report.seconds = time.perf_counter() - t0
    notify("done", report.pages_done, report.pages_done)
    log.info("ingested %s -> %s (%d pages, %d blocks, %d chunks, %.1fs)",
             path.name, doc_id, report.n_pages, report.n_blocks,
             report.n_chunks, report.seconds)
    if own_store:
        store.close_all()
    return report


def _merge_nodes(a: list[dict], b: list[dict]) -> list[dict]:
    seen = {n["node_id"]: n for n in a}
    for n in b:
        if n["node_id"] not in seen:
            seen[n["node_id"]] = n
    return list(seen.values())


def _merge_edges(a: list[dict], b: list[dict]) -> list[dict]:
    seen = {(e["src"], e["dst"], e["rel"]): e for e in a}
    for e in b:
        key = (e["src"], e["dst"], e["rel"])
        if key not in seen:
            seen[key] = e
    return list(seen.values())


# ---------------------------------------------------------------------------
# LaTeX conversion without indexing (cli convert-notes)
# ---------------------------------------------------------------------------
def convert_to_latex(path: str | Path, *, title: str | None = None,
                     handwritten: bool = True,
                     vision_fn: VisionFn | None = None,
                     out_formats: Sequence[str] = ("tex",),
                     out_dir: str | Path | None = None,
                     base_name: str | None = None) -> dict[str, Path]:
    """Image/PDF -> LaTeX (+ any other export format) *without* touching the DB.

    Used by ``cli.py convert-notes`` and by the demo runbook's pre-generated
    artifacts fallback.
    """
    from .models import vision_generate
    vision_fn = vision_fn or vision_generate
    path = Path(path)
    out_dir = Path(out_dir or path.parent)
    base = base_name or (path.stem + "_pagex")
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_img = load_pages(path)
    doc = make_document(new_id("doc"), title or path.stem, str(path),
                        infer_kind(path, handwritten))
    seq = 0
    for page_no, raw_img in enumerate(pages_img, start=1):
        pre = preprocess_page(raw_img)
        page = make_page(doc, page_no, width=pre["width"], height=pre["height"])
        parsed = understand_page(pre["image"], width=pre["width"],
                                 height=pre["height"], vision_fn=vision_fn)
        fill_page(page, parsed, doc.doc_id, seq_offset=seq)
        seq += len(page.blocks)

    written: dict[str, Path] = {}
    for fmt in out_formats:
        target = out_dir / f"{base}.{ 'tex' if fmt in ('tex','clean-tex') else fmt}"
        if fmt == "clean-tex":
            target = out_dir / f"{base}_clean.tex"
        written[fmt] = export_document(doc, target, fmt)
    return written


# ---------------------------------------------------------------------------
# helpers used by the GUI / CLI
# ---------------------------------------------------------------------------
def save_block_edit(store: Store, block_id: str, *,
                    text: str | None = None, latex: str | None = None,
                    kind: str | None = None,
                    embed_fn: EmbedFn | None = None) -> dict:
    """Persist a block edit and refresh every retrieval unit that cites it.

    The ``blocks`` row is the single source of truth; chunks are derived.
    Chunk *grouping* is preserved on edit (the block list does not change),
    only the content/vector of the affected chunks is recomputed and the
    affected ``knowledge graph`` evidence keeps pointing at the same ids.
    """
    rows = store.blocks_by_ids([block_id])
    if not rows:
        raise KeyError(f"block {block_id} not found in the store")
    row = rows[block_id]
    if kind is not None:
        if kind not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind {kind!r}")
        row["kind"] = kind
    if text is not None:
        row["text"] = text
    if latex is not None:
        row["latex"] = latex
    store.put_blocks([row])

    doc_id = row["doc_id"]
    page_row = store.get_page(row["page_id"]) or {}
    caption = ""
    meta = page_row.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except Exception:
            meta = {}
    caption = str(meta.get("caption", ""))

    refreshed: list[str] = []
    for chunk in store.iter_chunks(with_vec=False, doc_ids=[doc_id]):
        ids = list(chunk.block_ids)
        if block_id not in ids:
            continue
        by_id = store.blocks_by_ids(ids)
        blocks = sorted((Block.from_row(by_id[b], page_no=0) for b in ids if b in by_id),
                        key=lambda b: b.seq)
        parts = [b.content for b in blocks if b.content]
        if chunk.chunk_id.endswith("-c000") and caption and parts:
            parts.insert(0, caption)
        content = "\n\n".join(parts).strip() or chunk.content
        vec, dim = None, 0
        try:
            fn = embed_fn
            if fn is None:
                from .models import embed_texts as fn        # noqa: N810
            vec = fn([content])[0]
            dim = len(vec)
        except Exception as exc:                       # embedder offline
            log.warning("re-embed skipped for %s: %s", chunk.chunk_id, exc)
        store.put_chunks([Chunk(
            chunk_id=chunk.chunk_id, doc_id=doc_id, page_id=chunk.page_id,
            block_ids=ids, content=content, visual_desc=chunk.visual_desc,
            vec=vec, vec_dim=dim, token_count=approx_tokens(content))])
        refreshed.append(chunk.chunk_id)
    store.conn.commit()
    return {"block_id": block_id, "chunks_refreshed": len(refreshed),
            "kind": row["kind"]}


def ingest_paths(paths: Sequence[Path], *, progress: ProgressFn | None = None,
                 store: Store | None = None, **kwargs) -> list[IngestReport]:
    reports = []
    total = len(paths)
    for i, p in enumerate(paths, start=1):
        def sub(stage: str, cur: int, tot: int, _i: int = i) -> None:
            if progress:
                progress(f"[{_i}/{total}] {p.name}: {stage}", cur, tot)
        try:
            reports.append(ingest(p, progress=sub, store=store, **kwargs))
        except Exception as exc:
            log.error("ingest failed for %s: %s", p, exc)
            reports.append(IngestReport(doc_id="", title=p.stem, path=str(p),
                                        kind="image", errors=[str(exc)]))
    return reports
